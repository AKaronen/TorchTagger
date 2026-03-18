import math
import os

import numpy as np

import h5py

from .common import (
    coerce_inputs_to_bcf,
    prepare_output_dir,
    resolve_class_labels,
    save_generic_dataset_metadata,
    save_numpy_partitions,
    to_one_hot_labels,
)


class H5DataParser:
    """Parse HDF5 files and save NumPy train/test partitions."""

    parser_name = "h5"
    _extensions = (".h5", ".hdf5")

    def __init__(self, data_config):
        self.data_config = data_config

    @staticmethod
    def _list_h5_files(data_path):
        if os.path.isfile(data_path):
            return [data_path]

        files = []
        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            if os.path.isfile(path) and path.lower().endswith(H5DataParser._extensions):
                files.append(path)
        return files

    @staticmethod
    def _resolve_key(handle, configured_key, candidates, kind):
        if configured_key:
            if configured_key not in handle:
                raise KeyError(
                    f"Configured {kind} key '{configured_key}' was not found in {handle.filename}."
                )
            return configured_key

        for key in candidates:
            if key in handle:
                return key
        raise KeyError(
            f"Could not infer {kind} key in {handle.filename}. Tried: {candidates}"
        )

    def make_data(self):
        if h5py is None:
            raise ImportError(
                "h5py is required for H5DataParser. Install it with 'pip install h5py'."
            )

        data_path = self.data_config["data_path"]
        outdir = self.data_config.get("outdir", "training_data/")
        ratio = float(self.data_config.get("ratio", 1.0))
        chunk_size = int(self.data_config.get("chunk_size", 10000))
        configured_inputs = self.data_config.get("feature_labels", [])
        test_split = float(self.data_config.get("test_split", 0.2))
        seed = int(self.data_config.get("seed", 42))
        input_layout = self.data_config.get("input_layout", "BCF")

        files = self._list_h5_files(data_path)
        if not files:
            raise FileNotFoundError(f"No .h5/.hdf5 files found in '{data_path}'.")

        if not prepare_output_dir(outdir):
            return

        file_specs = []
        total_entries = 0
        configured_extra_keys = self.data_config.get("h5_extra_keys", [])

        for path in files:
            with h5py.File(path, "r") as h5f:
                input_key = self._resolve_key(
                    h5f,
                    self.data_config.get("h5_input_key"),
                    ("nn_inputs", "inputs", "x"),
                    "input",
                )
                target_key = self._resolve_key(
                    h5f,
                    self.data_config.get("h5_target_key"),
                    ("class_label", "targets", "label", "y"),
                    "target",
                )

                valid_extras = [k for k in configured_extra_keys if k in h5f]
                n_entries = int(h5f[input_key].shape[0])
                total_entries += n_entries
                file_specs.append(
                    (path, input_key, target_key, valid_extras, n_entries)
                )

        if total_entries == 0:
            raise ValueError("H5 input contains zero entries.")

        target_entries = total_entries
        if ratio < 1.0:
            target_entries = max(1, int(math.ceil(total_entries * ratio)))

        processed = 0
        n_classes = self.data_config.get("num_classes", None)
        input_chunks = []
        label_chunks = []
        class_labels = None
        metadata_extras = []

        for path, input_key, target_key, extra_keys, n_entries in file_specs:
            if processed >= target_entries:
                break

            print("Processing file:", path)
            max_entries = min(n_entries, target_entries - processed)

            with h5py.File(path, "r") as h5f:
                inputs_ds = h5f[input_key]
                labels_ds = h5f[target_key]

                for start in range(0, max_entries, chunk_size):
                    stop = min(start + chunk_size, max_entries)
                    labels, n_classes = to_one_hot_labels(
                        labels_ds[start:stop], n_classes
                    )

                    if class_labels is None:
                        class_labels = resolve_class_labels(
                            self.data_config.get("class_labels", None), n_classes
                        )
                        metadata_extras = list(extra_keys)

                    inputs_chunk = coerce_inputs_to_bcf(
                        inputs_ds[start:stop], input_layout=input_layout
                    )
                    input_chunks.append(inputs_chunk)
                    label_chunks.append(labels.astype(np.float32, copy=False))

                    processed += stop - start

                    print(
                        f"Processed {processed}/{target_entries} entries | {np.round(processed / target_entries * 100, 1)}%"
                    )

        if not input_chunks:
            raise ValueError("No H5 samples were loaded.")

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
            extras=metadata_extras,
        )
        save_numpy_partitions(
            outdir,
            inputs,
            targets,
            feature_labels=configured_inputs,
            class_labels=class_labels,
            extra_vars=metadata_extras,
            test_split=test_split,
            seed=seed,
            shuffle=self.data_config.get("shuffle", True),
        )
