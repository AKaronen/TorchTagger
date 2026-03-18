import math
import os

import numpy as np
import yaml

try:
    from datasets import interleave_datasets, load_dataset
except ImportError:
    interleave_datasets = None
    load_dataset = None

from .collide_helpers import create_jet_datasets
from .common import (
    coerce_inputs_to_bcf,
    prepare_output_dir,
    resolve_class_labels,
    save_generic_dataset_metadata,
    save_numpy_partitions,
    to_one_hot_labels,
)


class HuggingFaceDataParser:
    """Parse datasets loaded through Hugging Face datasets to NumPy partitions."""

    parser_name = "huggingface"

    def __init__(self, data_config):
        self.data_config = data_config

    @staticmethod
    def _ensure_list(value, name):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        raise TypeError(f"'{name}' must be a string or list of strings.")

    @staticmethod
    def _dedupe_keep_order(items):
        seen = set()
        out = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def _cfg_or_global(self, cfg, *keys):
        for key in keys:
            if cfg.get(key) is not None:
                return cfg.get(key)
            if self.data_config.get(key) is not None:
                return self.data_config.get(key)
        return None

    @staticmethod
    def _load_collide_fields_map():
        fields_file = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "collide_fields.yml")
        )
        if not os.path.exists(fields_file):
            raise FileNotFoundError(f"collide_fields.yml not found at '{fields_file}'.")

        with open(fields_file, "r") as handle:
            field_map = yaml.safe_load(handle) or {}

        if not isinstance(field_map, dict):
            raise ValueError(
                "collide_fields.yml must contain a mapping of field groups."
            )

        return field_map

    def _resolve_hf_columns(self, cfg):
        explicit_columns = self._cfg_or_global(cfg, "hf_columns", "columns")
        if explicit_columns is not None:
            cols = self._ensure_list(explicit_columns, "hf_columns")
            return self._dedupe_keep_order(cols)

        selected_groups = self._cfg_or_global(
            cfg,
            "hf_collide_field_groups",
            "collide_field_groups",
        )
        selected_fields = self._cfg_or_global(
            cfg,
            "hf_collide_fields",
            "collide_fields",
        )

        selected_groups = self._ensure_list(selected_groups, "hf_collide_field_groups")
        selected_fields = self._ensure_list(selected_fields, "hf_collide_fields")

        if not selected_groups and not selected_fields:
            return None

        field_map = self._load_collide_fields_map()
        available_groups = sorted(field_map.keys())
        resolved = []

        for group in selected_groups:
            if group not in field_map:
                raise KeyError(
                    f"Unknown collide field group '{group}'. Available groups: {available_groups}"
                )
            group_fields = field_map[group]
            if not isinstance(group_fields, list):
                raise ValueError(
                    f"Group '{group}' in collide_fields.yml must map to a list of fields."
                )
            resolved.extend(group_fields)

        known_fields = set()
        if isinstance(field_map.get("all_fields"), list):
            known_fields = set(field_map["all_fields"])

        for field in selected_fields:
            if known_fields and field not in known_fields:
                raise KeyError(
                    f"Unknown collide field '{field}'. Use names listed in collide_fields.yml."
                )
            resolved.append(field)

        return self._dedupe_keep_order(resolved)

    def _load_single_dataset(self, cfg):
        if load_dataset is None:
            raise ImportError(
                "datasets is required for HuggingFaceDataParser. Install it with 'pip install datasets'."
            )

        dataset_name = cfg.get("hf_dataset") or cfg.get("dataset") or cfg.get("path")
        if not dataset_name:
            raise ValueError(
                "Hugging Face parser requires 'hf_dataset' (or 'dataset')."
            )

        kwargs = {"split": cfg.get("hf_split", cfg.get("split", "train"))}
        if cfg.get("hf_name", cfg.get("name")) is not None:
            kwargs["name"] = cfg.get("hf_name", cfg.get("name"))
        if cfg.get("hf_data_dir", cfg.get("data_dir")) is not None:
            kwargs["data_dir"] = cfg.get("hf_data_dir", cfg.get("data_dir"))
        if cfg.get("hf_data_files", cfg.get("data_files")) is not None:
            kwargs["data_files"] = cfg.get("hf_data_files", cfg.get("data_files"))

        resolved_columns = self._resolve_hf_columns(cfg)
        if resolved_columns is not None:
            kwargs["columns"] = resolved_columns

        return load_dataset(dataset_name, **kwargs)

    @staticmethod
    def _resolve_column(column_names, configured_name, candidates, kind):
        if configured_name:
            if configured_name not in column_names:
                raise KeyError(
                    f"Configured {kind} column '{configured_name}' not found in dataset columns {column_names}."
                )
            return configured_name

        for candidate in candidates:
            if candidate in column_names:
                return candidate

        raise KeyError(
            f"Could not infer {kind} column. Tried {candidates} in dataset columns {column_names}."
        )

    def make_data(self):
        outdir = self.data_config.get("outdir", "training_data/")
        ratio = float(self.data_config.get("ratio", 1.0))
        chunk_size = int(self.data_config.get("chunk_size", 10000))
        seed = int(self.data_config.get("seed", 42))
        test_split = float(self.data_config.get("test_split", 0.2))
        input_layout = self.data_config.get("input_layout", "BCF")

        if not prepare_output_dir(outdir):
            return

        signal_cfg = self.data_config.get("hf_signal", None)
        background_cfg = self.data_config.get("hf_background", None)

        if signal_cfg and background_cfg:
            signal_ds = self._load_single_dataset(signal_cfg)
            background_ds = self._load_single_dataset(background_cfg)

            if self.data_config.get("hf_apply_collide_mapping", False):
                max_pf = int(self.data_config.get("max_pf", 32))
                l1_features = self.data_config.get("l1_features", None)
                if l1_features is None:
                    raise ValueError(
                        "hf_apply_collide_mapping=True requires 'l1_features' in data config."
                    )
                signal_ds, background_ds = create_jet_datasets(
                    signal_ds, background_ds, (max_pf, l1_features)
                )

            if interleave_datasets is None:
                raise ImportError(
                    "datasets.interleave_datasets is required for hf_signal/hf_background flow."
                )

            probs = self.data_config.get("hf_interleave_probabilities", [0.5, 0.5])
            dataset = interleave_datasets(
                [signal_ds, background_ds],
                probabilities=probs,
                seed=seed,
            )
        else:
            dataset = self._load_single_dataset(self.data_config)

        if self.data_config.get("shuffle", True):
            dataset = dataset.shuffle(seed=seed)

        total_entries = len(dataset)
        if total_entries == 0:
            raise ValueError("Loaded Hugging Face dataset is empty.")

        target_entries = total_entries
        if ratio < 1.0:
            target_entries = max(1, int(math.ceil(total_entries * ratio)))

        input_column = self._resolve_column(
            dataset.column_names,
            self.data_config.get("hf_input_column", None),
            ("nn_inputs", "inputs", "x"),
            "input",
        )
        target_column = self._resolve_column(
            dataset.column_names,
            self.data_config.get("hf_target_column", None),
            ("class_label", "targets", "label", "y"),
            "target",
        )

        extra_columns = [
            col
            for col in self.data_config.get("hf_extra_columns", [])
            if col in dataset.column_names
        ]

        n_classes = self.data_config.get("num_classes", None)
        configured_inputs = self.data_config.get("feature_labels", [])

        input_chunks = []
        label_chunks = []
        class_labels = None
        buffer_inputs = []
        buffer_labels = []
        buffer_extras = {name: [] for name in extra_columns}

        for idx, entry in enumerate(dataset):
            if idx >= target_entries:
                break

            buffer_inputs.append(entry[input_column])
            buffer_labels.append(entry[target_column])
            for name in extra_columns:
                buffer_extras[name].append(entry[name])

            if len(buffer_inputs) < chunk_size and idx + 1 < target_entries:
                continue

            labels, n_classes = to_one_hot_labels(buffer_labels, n_classes)

            if class_labels is None:
                class_labels = resolve_class_labels(
                    self.data_config.get("class_labels", None), n_classes
                )

            input_chunks.append(
                coerce_inputs_to_bcf(buffer_inputs, input_layout=input_layout)
            )
            label_chunks.append(labels.astype(np.float32, copy=False))

            processed = min(idx + 1, target_entries)
            print(
                f"Processed {processed}/{target_entries} entries | {np.round(processed / target_entries * 100, 1)}%"
            )

            buffer_inputs = []
            buffer_labels = []
            buffer_extras = {name: [] for name in extra_columns}

        if not input_chunks:
            raise ValueError("No Hugging Face samples were loaded.")

        inputs = np.concatenate(input_chunks, axis=0).astype(np.float32, copy=False)
        targets = np.concatenate(label_chunks, axis=0).astype(np.float32, copy=False)

        if class_labels is None:
            class_labels = resolve_class_labels(
                self.data_config.get("class_labels", None), targets.shape[1]
            )

        if not configured_inputs:
            configured_inputs = [f"f{i}" for i in range(inputs.shape[2])]

        save_generic_dataset_metadata(
            outdir,
            class_labels,
            inputs=configured_inputs,
            extras=extra_columns,
        )
        save_numpy_partitions(
            outdir,
            inputs,
            targets,
            feature_labels=configured_inputs,
            class_labels=class_labels,
            extra_vars=extra_columns,
            test_split=test_split,
            seed=seed,
            shuffle=self.data_config.get("shuffle", True),
        )
