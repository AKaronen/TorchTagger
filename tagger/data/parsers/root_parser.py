import gc
import json
import math
import os
import shutil

import awkward as ak
import numpy as np
import uproot
import yaml

from ..config import EXTRA_FIELDS, FILTER_PATTERN, INPUT_TAG, N_PARTICLES, CLASS_LABELS
from .common import (
    coerce_inputs_to_bcf,
    prepare_output_dir,
    resolve_class_labels,
    save_generic_dataset_metadata,
    save_numpy_partitions,
)


def _define_target(data, all_labels: list = CLASS_LABELS):
    """
    Splits data by particle flavor and applies conditions for each category.
    Also creates pT and mass targets.
    """
    class_labels = dict(zip(all_labels, range(len(all_labels))))
    labels = np.zeros(shape=(len(data), len(class_labels)), dtype=np.float32)

    for i, jet in enumerate(data["fj_label"]):
        if jet in all_labels:
            labels[i, class_labels[jet]] = 1.0
        else:
            print(f"Warning: Unknown fj_label '{jet}' encountered.")
    data = ak.with_field(data, labels, "class_label")

    pt_ratio = ak.nan_to_num(
        data["jet_genmatch_pt"] / data["jet_pt_phys"], nan=0, posinf=0, neginf=0
    )
    data["target_pt"] = np.clip(pt_ratio, 0.3, 3)
    data["target_pt_phys"] = np.clip(
        ak.nan_to_num(data["jet_genmatch_pt"], nan=0, posinf=0, neginf=0), 0, 2000
    )

    mass_ratio = ak.nan_to_num(
        data["jet_genmatch_mass"] / data["jet_mass"], nan=0, posinf=0, neginf=0
    )
    data["target_mass"] = np.clip(mass_ratio, 0.3, 3)
    data["target_mass_phys"] = np.clip(
        ak.nan_to_num(data["jet_genmatch_mass"], nan=0, posinf=0, neginf=0), 0, 256
    )

    jet_ptmin_gen, jet_massmin_gen = (
        (data["target_pt_phys"] > 15.0),
        (data["target_mass_phys"] > 5.0),
    )

    return data[jet_ptmin_gen & jet_massmin_gen], class_labels


def get_unique_fj_labels(infile, tree="outnano/Jets", step_size="500 MB"):
    unique_labels = set()

    for arrays in uproot.iterate(
        infile,
        treepath=tree,
        expressions=["fj_label"],
        step_size=step_size,
        how="zip",
    ):
        fj_labels = arrays["fj_label"]
        flat = ak.flatten(fj_labels, axis=None)
        flat = ak.Array(flat)
        unique_labels.update(list(set(flat.tolist())))

    return sorted(unique_labels)


def _get_pfcand_fields(tag):
    current_dir = os.path.dirname(__file__)
    pfcand_fields_path = os.path.normpath(
        os.path.join(current_dir, "..", "pfcand_fields.yml")
    )

    with open(pfcand_fields_path, "r") as handle:
        pfcand_fields = yaml.safe_load(handle)

    return pfcand_fields[tag]


def _pad_fill(array, target):
    """Pad an array to target length and then fill it with 0s."""
    return ak.fill_none(ak.pad_none(array, target, axis=1, clip=True), 0)


def _make_nn_inputs(data_split, tag, n_parts):
    features = _get_pfcand_fields(tag)
    inputs_list = []

    for field in features:
        field_array = data_split["jet_pfcand"][field]
        padded_filled_array = ak.values_astype(
            _pad_fill(field_array, n_parts), np.float32
        )
        inputs_list.append(padded_filled_array[:, :, np.newaxis])

    from math import pi

    pt = data_split["jet_pfcand"]["pt"]
    deta = data_split["jet_pfcand"]["deta"]
    dphi = data_split["jet_pfcand"]["dphi"]

    energy = pt * np.cosh(deta * pi / 720)
    px = pt * np.cos(dphi * pi / 720)
    py = pt * np.sin(dphi * pi / 720)
    pz = pt * np.sinh(deta * pi / 720)

    inputs_list.append(
        ak.values_astype(_pad_fill(energy, n_parts)[:, :, np.newaxis], np.float32)
    )
    inputs_list.append(
        ak.values_astype(_pad_fill(px, n_parts)[:, :, np.newaxis], np.float32)
    )
    inputs_list.append(
        ak.values_astype(_pad_fill(py, n_parts)[:, :, np.newaxis], np.float32)
    )
    inputs_list.append(
        ak.values_astype(_pad_fill(pz, n_parts)[:, :, np.newaxis], np.float32)
    )

    inputs = ak.concatenate(inputs_list, axis=2)
    data_split["nn_inputs"] = inputs


def _save_chunk_metadata(metadata_file, chunk, entries, outfile):
    chunk_info = {"chunk": chunk, "entries": entries, "file": outfile}

    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as handle:
            content = handle.read()
            metadata = json.loads(content) if content.strip() else []
    else:
        metadata = []

    metadata.append(chunk_info)

    with open(metadata_file, "w") as handle:
        json.dump(metadata, handle, indent=4)


def _save_dataset_metadata(outdir, class_labels, tag, extras):
    dataset_metadata_file = os.path.join(outdir, "variables.json")

    metadata = {
        "outputs": class_labels,
        "inputs": _get_pfcand_fields(tag) + ["e", "px", "py", "pz"],
        "extras": _get_pfcand_fields(extras),
    }

    with open(dataset_metadata_file, "w") as handle:
        json.dump(metadata, handle, indent=4)


def _process_chunk(data_split, tag, extras, n_parts, chunk, outdir):
    """Process a chunk and save it to a ROOT file (legacy path)."""
    _make_nn_inputs(data_split, tag, n_parts)
    extra_features = _get_pfcand_fields(extras)

    save_fields = [
        "nn_inputs",
        "class_label",
    ] + extra_features

    filtered_data = {field: data_split[field] for field in save_fields}

    outfile = os.path.join(outdir, f"data_chunk_{chunk}.root")
    with uproot.recreate(outfile) as handle:
        handle.mktree("data", filtered_data)
        print(f"Saved chunk {chunk} to {outfile}")

    metadata_file = os.path.join(outdir, "metadata.json")
    _save_chunk_metadata(metadata_file, chunk, len(data_split), outfile)

    del data_split, filtered_data, outfile
    gc.collect()


def extract_array(tree, field, entry_stop):
    """Extract an array from a tree with a limit on number of entries."""
    return tree[field].array(entry_stop=entry_stop)


def extract_nn_inputs(data, input_vars, n_parts=16, n_entries=None):
    """Extract nn inputs based on input_vars list."""
    inputs_list = []

    for field in input_vars:
        field_array = extract_array(data, f"jet_pfcand_{field}", n_entries)

        padded_filled_array = _pad_fill(field_array, n_parts)
        inputs_list.append(padded_filled_array[:, :, np.newaxis])

    return ak.concatenate(inputs_list, axis=2)


def group_id_values(event_id, *arrays, num_elements=2):
    """Group values according to event id and filter short groups."""
    sorted_indices = ak.argsort(event_id)
    sorted_event_id = event_id[sorted_indices]

    _, counts = np.unique(sorted_event_id, return_counts=True)

    grouped_id = ak.unflatten(sorted_event_id, counts)
    grouped_arrays = [ak.unflatten(arr[sorted_indices], counts) for arr in arrays]

    mask = ak.num(grouped_id) >= num_elements
    filtered_grouped_arrays = [arr[mask] for arr in grouped_arrays]

    return grouped_id[mask], filtered_grouped_arrays


class RootDataParser:
    """ROOT parser producing train/test NPZ partitions."""

    parser_name = "root"

    def __init__(self, data_config):
        self.data_config = data_config

    def make_data(self):
        data_path = self.data_config["data_path"]
        outdir = self.data_config.get("outdir", "training_data/")
        tag = self.data_config.get("tag", INPUT_TAG)
        extras = self.data_config.get("extras", EXTRA_FIELDS)
        n_parts = int(self.data_config.get("n_particles", N_PARTICLES))
        ratio = float(self.data_config.get("ratio", 1.0))
        step_size = self.data_config.get("step_size", "100MB")
        tree = self.data_config.get("tree", "outnano/Jets")
        test_split = float(self.data_config.get("test_split", 0.2))
        seed = int(self.data_config.get("seed", 42))
        shuffle = self.data_config.get("shuffle", True)
        if not prepare_output_dir(outdir):
            return

        root_files = []
        for name in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, name)
            if os.path.isfile(path) and name.lower().endswith(".root"):
                root_files.append(path)

        if not root_files:
            raise FileNotFoundError(f"No ROOT files found in '{data_path}'.")

        features = _get_pfcand_fields(tag) + ["e", "px", "py", "pz"]
        extra_features = _get_pfcand_fields(extras)

        input_chunks = []
        label_chunks = []
        class_labels = None
        processed_entries = 0
        total_entries = 0
        entry_counts = {}
        for infile in root_files:
            num_entries = uproot.open(infile)[tree].num_entries
            total_entries += num_entries
            entry_counts[infile] = num_entries
        if ratio < 1.0:
            total_entries = int(total_entries * ratio)
        for infile in root_files:
            num_entries = entry_counts[infile]
            print("Processing file:", infile)
            for data in uproot.iterate(
                infile,
                filter_name=FILTER_PATTERN,
                how="zip",
                step_size=step_size,
                max_workers=8,
            ):
                if shuffle:
                    indices = np.arange(len(data))
                    np.random.shuffle(indices)
                    data = data[indices]
                if ratio < 1:
                    n_rows = len(data)
                    limit = int(math.ceil(n_rows * ratio))
                    data = data[:limit]

                processed_entries += len(data)
                remaining = total_entries - processed_entries
                if remaining < 0:
                    data = data[:remaining]

                jet_cut = (
                    (data["jet_pt_phys"] > 15)
                    & (np.abs(data["jet_eta_phys"]) < 2.4)
                    & (data["jet_reject"] == 0)
                )
                data = data[jet_cut]
                if len(data) == 0:
                    continue

                data, class_labels = _define_target(data)
                if len(data) == 0:
                    continue

                _make_nn_inputs(data, tag, n_parts)
                input_chunks.append(coerce_inputs_to_bcf(np.asarray(data["nn_inputs"])))
                label_chunks.append(np.asarray(data["class_label"], dtype=np.float32))

                print(
                    f"Processed {processed_entries}/{total_entries} entries | {np.round(processed_entries / total_entries * 100, 1)}%"
                )
                if processed_entries >= total_entries:
                    break

        if not input_chunks:
            raise ValueError("No valid ROOT samples were produced after selection.")

        inputs = np.concatenate(input_chunks, axis=0).astype(np.float32, copy=False)
        targets = np.concatenate(label_chunks, axis=0).astype(np.float32, copy=False)

        if class_labels is None:
            class_labels = resolve_class_labels(
                self.data_config.get("class_labels"), targets.shape[1]
            )

        save_generic_dataset_metadata(
            outdir,
            class_labels,
            inputs=features,
            extras=extra_features,
        )
        save_numpy_partitions(
            outdir,
            inputs,
            targets,
            feature_labels=features,
            class_labels=class_labels,
            extra_vars=extra_features,
            test_split=test_split,
            seed=seed,
            shuffle=self.data_config.get("shuffle", True),
        )
