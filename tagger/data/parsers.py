# Data parsers for different dataset formats (ROOT, Numpy, HuggingFace, etc.)

# Python
import gc
import json
import os
import shutil

import re


# Third party
import numpy as np
from sklearn.model_selection import train_test_split
import uproot
import yaml
import awkward as ak


# ROOT Dataset configuration
from .config import EXTRA_FIELDS, FILTER_PATTERN, INPUT_TAG, N_PARTICLES, CLASS_LABELS


################################### ROOT Dataset parsing ###################################


def _define_target(data, all_labels: list = CLASS_LABELS):
    """
    Splits data by particle flavor and applies conditions for each category. Also creates the pT target.

    Parameters:
        data (awkward array): The input data to split.

    Returns:
        dict: A dictionary containing the split data by label.
    """

    class_labels = dict(zip(all_labels, range(len(all_labels))))
    labels = np.zeros(shape=(len(data), len(class_labels)), dtype=np.float32)

    for i, jet in enumerate(data["fj_label"]):
        if jet in all_labels:
            labels[i, class_labels[jet]] = 1.0
        else:
            print(f"Warning: Unknown fj_label '{jet}' encountered.")
    data = ak.with_field(data, labels, "class_label")

    # Set pT targets
    pt_ratio = ak.nan_to_num(
        data["jet_genmatch_pt"] / data["jet_pt_phys"], nan=0, posinf=0, neginf=0
    )
    data["target_pt"] = np.clip(pt_ratio, 0.3, 3)
    data["target_pt_phys"] = np.clip(
        ak.nan_to_num(data["jet_genmatch_pt"], nan=0, posinf=0, neginf=0), 0, 2000
    )

    # Set mass targets
    mass_ratio = ak.nan_to_num(
        data["jet_genmatch_mass"] / data["jet_mass"], nan=0, posinf=0, neginf=0
    )
    data["target_mass"] = np.clip(mass_ratio, 0.3, 3)
    data["target_mass_phys"] = np.clip(
        ak.nan_to_num(data["jet_genmatch_mass"], nan=0, posinf=0, neginf=0), 0, 256
    )

    # Apply pt_cut and mass_cut
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
        expressions=["fj_label"],  # only load fj_label
        step_size=step_size,
        how="zip",
    ):
        fj_labels = arrays["fj_label"]
        # Flatten jagged array
        flat = ak.flatten(fj_labels, axis=None)
        # Use ak.Array to ensure it's an awkward array
        flat = ak.Array(flat)
        # Convert to Python list and add unique entries
        unique_labels.update(list(set(flat.tolist())))

    return sorted(unique_labels)


def _get_pfcand_fields(tag):
    # Get the directory of the current file (tools.py)
    current_dir = os.path.dirname(__file__)

    # Construct the path to pfcand_fields.yml relative to tools.py
    pfcand_fields_path = os.path.join(current_dir, "pfcand_fields.yml")

    # Load the YAML file as a dictionary
    with open(pfcand_fields_path, "r") as file:
        pfcand_fields = yaml.safe_load(file)

    return pfcand_fields[tag]


def _pad_fill(array, target):
    """
    pad an array to target length and then fill it with 0s
    """
    return ak.fill_none(ak.pad_none(array, target, axis=1, clip=True), 0)


def _make_nn_inputs(data_split, tag, n_parts):
    features = _get_pfcand_fields(tag)

    # Concatenate all the inputs
    inputs_list = []

    # Vertically stacked them to create input sets
    # https://awkward-array.org/doc/main/user-guide/how-to-restructure-concatenate.html
    # Also pad and fill them with 0 to the number of constituents we are using (nconstit)
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

    # batch_size, n_particles, n_features
    inputs = ak.concatenate(inputs_list, axis=2)
    data_split["nn_inputs"] = inputs

    return


def _save_chunk_metadata(metadata_file, chunk, entries, outfile):
    chunk_info = {"chunk": chunk, "entries": entries, "file": outfile}

    # Load existing metadata or start a new list
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            content = f.read()
            metadata = json.loads(content) if content.strip() else []
    else:
        metadata = []

    metadata.append(chunk_info)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    return


def _save_dataset_metadata(outdir, class_labels, tag, extras):
    dataset_metadata_file = os.path.join(outdir, "variables.json")

    metadata = {
        "outputs": class_labels,
        "inputs": _get_pfcand_fields(tag) + ["e", "px", "py", "pz"],
        "extras": _get_pfcand_fields(extras),
    }

    with open(dataset_metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    return


def _process_chunk(data_split, tag, extras, n_parts, chunk, outdir):
    """
    Process chunk of data_split to save/parse it for training datasets
    """

    # Create the NN inputs
    _make_nn_inputs(data_split, tag, n_parts)
    extra_features = _get_pfcand_fields(extras)

    # Save them to a root file
    save_fields = [
        "nn_inputs",
        "class_label",
    ] + extra_features

    # Filter the data_split to only include save_fields
    filtered_data = {field: data_split[field] for field in save_fields}

    # Save chunk to files
    outfile = os.path.join(outdir, f"data_chunk_{chunk}.root")
    with uproot.recreate(outfile) as f:
        f.mktree("data", filtered_data)
        print(f"Saved chunk {chunk} to {outfile}")

    # Log metadata
    metadata_file = os.path.join(outdir, "metadata.json")
    _save_chunk_metadata(
        metadata_file, chunk, len(data_split), outfile
    )  # Chunk, Entries, Outfile

    del data_split, filtered_data, outfile
    # Delete the variables to save memory
    gc.collect()

    return


# >>>>>>FUNCTIONS THAT SHOULD BE USED EXTERNALLY!<<<<<<<


def extract_array(tree, field, entry_stop):
    """
    Extracts an array from the tree with a limit on the number of entries.
    """
    return tree[field].array(entry_stop=entry_stop)


def extract_nn_inputs(data, input_vars, n_parts=16, n_entries=None):
    """
    Extract nn inputs based on the input_vars list
    """

    # Concatenate all the inputs
    inputs_list = []

    for field in input_vars:
        field_array = extract_array(data, f"jet_pfcand_{field}", n_entries)

        padded_filled_array = _pad_fill(field_array, n_parts)
        inputs_list.append(padded_filled_array[:, :, np.newaxis])

    # batch_size, n_particles, n_features
    inputs = ak.concatenate(inputs_list, axis=2)

    return inputs


def group_id_values(event_id, *arrays, num_elements=2):
    """
    Group values according to event id.
    Filter out events that has less than num_elements
    """

    # Use ak.argsort to sort based on event_id
    sorted_indices = ak.argsort(event_id)
    sorted_event_id = event_id[sorted_indices]

    # Find unique event_ids and counts manually
    unique_event_id, counts = np.unique(sorted_event_id, return_counts=True)

    # Use ak.unflatten to group the arrays by counts
    grouped_id = ak.unflatten(sorted_event_id, counts)
    grouped_arrays = [ak.unflatten(arr[sorted_indices], counts) for arr in arrays]

    # Filter out groups that don't have at least num_elements elements
    mask = ak.num(grouped_id) >= num_elements
    filtered_grouped_arrays = [arr[mask] for arr in grouped_arrays]

    return grouped_id[mask], filtered_grouped_arrays


def make_data(
    data_path,
    outdir="training_data/",
    tag=INPUT_TAG,
    extras=EXTRA_FIELDS,
    n_parts=N_PARTICLES,
    ratio=1.0,
    step_size="100MB",
    tree="outnano/jets",
):
    """
    Process the data set in chunks from the input ntuples file.

    Parameters:
        infile (str): The input file path.
        outdir (str): The output directory.
        tag (str): Input tags to use from pfcands, defined in pfcand_fields.yml.
        extras (str): Extra fields to store for plotting, defined in pfcand_fields.yml
        n_parts (int): Number of constituent particles to use for tagging.
        fraction (float) : fraction from (0-1) of data to process for training/testing
        step_size (str): Step size for uproot iteration.
    """

    # Check if output dir already exists, remove if so
    if os.path.exists(outdir):
        confirm = input(
            f"The directory '{outdir}' already exists. Do you want to delete it and continue? [y/n]: "
        )
        if confirm.lower() == "y":
            shutil.rmtree(outdir)
            print(f"Deleted existing directory: {outdir}")
        else:
            print("Exiting without making changes.")
            return

    # Create output training dataset
    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)

    total_num_entries = 0
    chunk = 0
    for infile in os.listdir(data_path):
        if not infile.endswith(".root"):
            continue
        infile = os.path.join(data_path, infile)
        print("Processing file:", infile)

        # Loop through the entries
        num_entries = uproot.open(infile)[tree].num_entries
        total_num_entries += uproot.open(infile)[tree].num_entries
        num_entries_done = 0

        for data in uproot.iterate(
            infile,
            filter_name=FILTER_PATTERN,
            how="zip",
            step_size=step_size,
            max_workers=8,
        ):
            num_entries_done += len(data)  # count before cuts

            # Define jet kinematic cuts
            jet_cut = (
                (data["jet_pt_phys"] > 15)
                & (np.abs(data["jet_eta_phys"]) < 2.4)
                & (data["jet_reject"] == 0)
            )
            data = data[jet_cut]

            data, class_labels = _define_target(data)

            # If first chunk then save metadata of the dataset
            if chunk == 0:
                _save_dataset_metadata(outdir, class_labels, tag, extras)

            # Process and save training data for a given feature set
            _process_chunk(
                data,
                tag=tag,
                extras=extras,
                n_parts=n_parts,
                chunk=chunk,
                outdir=outdir,
            )

            # Number of chunk for indexing files
            chunk += 1
            print(
                f"Processed {num_entries_done}/{num_entries} entries | {np.round(num_entries_done / num_entries * 100, 1)}%"
            )
            print("Total entries processed:", total_num_entries)
            if num_entries_done / total_num_entries >= ratio:
                break
