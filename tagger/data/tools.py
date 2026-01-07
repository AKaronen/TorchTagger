# Python
import gc
import json
import os
import shutil

import re
import awkward as ak

# Third party
import numpy as np
from sklearn.model_selection import train_test_split
import uproot
import yaml
from typing import Any

# Dataset configuration
from .config import EXTRA_FIELDS, FILTER_PATTERN, INPUT_TAG, N_PARTICLES

gc.set_threshold(0)

# >>>>>>>>>>>>>>>>>>>PRIVATE FUNCTIONS<<<<<<<<<<<<<<<<<<<<<<
all_labels = [
    "Top_Wcs",
    "H_ss",
    "Top_Wev",
    "Top_Wtaumv",
    "Top_bWtauhv",
    "Top_bWtauev",
    "Top_bWs",
    "Top_Wmv",
    "Top_bWqq",
    "H_gg",
    "H_bb",
    "Top_bWc",
    "H_tauhtaum",
    "H_mm",
    "H_tauhtaue",
    "QCD_others",
    "H_ee",
    "Top_bWcs",
    "Top_Wtauhv",
    "Top_bWq",
    "H_tauhtauh",
    "Top_bWev",
    "Top_Wqq",
    "Hext_aa",
    "H_cc",
    "Top_bWtaumv",
    "Top_Wtauev",
    "H_qq",
    "Top_bWmv",
]


def _add_response_vars(data):
    data["jet_ptUncorr_div_ptGen"] = ak.nan_to_num(
        data["jet_pt_phys"] / data["jet_genmatch_pt"],
        copy=True,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    data["jet_ptCorr_div_ptGen"] = ak.nan_to_num(
        data["jet_pt_corr"] / data["jet_genmatch_pt"],
        copy=True,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    data["jet_ptRaw_div_ptGen"] = ak.nan_to_num(
        data["jet_pt_raw"] / data["jet_genmatch_pt"],
        copy=True,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _define_target(data, all_labels: None):
    """
    Splits data by particle flavor and applies conditions for each category. Also creates the pT target.

    Parameters:
        data (awkward array): The input data to split.

    Returns:
        dict: A dictionary containing the split data by label.
    """

    # genmatch_base = (data["jet_genmatch_pt"] >= 0) | (
    #    data["jet_genmatch_mass"] >= 0
    # )  # Only jets matched to a gen jet
    # data = data[genmatch_base]

    # Define conditions for each label
    # conditions = {
    #    "TP": (data["jet_genmatch_Nprongs"] >= 2),
    #    "BKG": (data["jet_genmatch_Nprongs"] < 2),
    # }

    # Automatically generate class labels based on the order of keys in conditions
    # class_labels = {label: idx for idx, label in enumerate(conditions)}    # {"H": 0, "W": 1, "Z": 2, "Two-prong": 3, "Background": 4}

    class_labels = dict(zip(all_labels, range(len(all_labels))))
    labels = np.zeros(shape=(len(data), len(class_labels)), dtype=np.float32)

    # print(labels.shape)
    # for i, jet in enumerate(data["fj_label"]):
    #    jet = jet[0].tolist()
    #    onehot = np.array([x for x in jet.values()])
    #    labels[i] = onehot

    for i, jet in enumerate(data["fj_label"]):
        # jet_labels = ak.to_list(jet)
        # jet_labels = list(set(jet_labels))
        # Fill one-hot
        if jet in all_labels:
            labels[i, class_labels[jet]] = 1.0
        else:
            print(f"Warning: Unknown fj_label '{jet}' encountered.")
    data = ak.with_field(data, labels, "class_label")
    # Assign numeric values based on conditions using awkward's where function
    # for label, condition in conditions.items():
    #    data["class_label"] = ak.where(
    #        condition, class_labels[label], data["class_label"]
    #    )

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


def _split_flavor(data):
    """
    Splits data by particle flavor and applies conditions for each category. Also creates the pT target.

    Parameters:
        data (awkward array): The input data to split.

    Returns:
        dict: A dictionary containing the split data by label.
    """

    genmatch_pt_base = data["jet_genmatch_pt"] > 0

    # Define conditions for each label
    conditions = {
        "b": (
            genmatch_pt_base
            & (data["jet_muflav"] == 0)
            & (data["jet_tauflav"] == 0)
            & (data["jet_elflav"] == 0)
            & (data["jet_genmatch_hflav"] == 5)
        ),  # Bottom
        "charm": (
            genmatch_pt_base
            & (data["jet_muflav"] == 0)
            & (data["jet_tauflav"] == 0)
            & (data["jet_elflav"] == 0)
            & (data["jet_genmatch_hflav"] == 4)
        ),  # Charm
        "light": (
            genmatch_pt_base
            & (data["jet_muflav"] == 0)
            & (data["jet_tauflav"] == 0)
            & (data["jet_elflav"] == 0)
            & (data["jet_genmatch_hflav"] == 0)
            & (
                (abs(data["jet_genmatch_pflav"]) == 0)
                | (abs(data["jet_genmatch_pflav"]) == 1)
                | (abs(data["jet_genmatch_pflav"]) == 2)
                | (abs(data["jet_genmatch_pflav"]) == 3)
            )
        ),  # uds
        "gluon": (
            genmatch_pt_base
            & (data["jet_muflav"] == 0)
            & (data["jet_tauflav"] == 0)
            & (data["jet_elflav"] == 0)
            & (data["jet_genmatch_hflav"] == 0)
            & (data["jet_genmatch_pflav"] == 21)
        ),  # Gluon
        "taup": (
            genmatch_pt_base
            & (data["jet_muflav"] == 0)
            & (data["jet_tauflav"] == 1)
            & (data["jet_taucharge"] > 0)
            & (data["jet_elflav"] == 0)
        ),  # Tau +
        "taum": (
            genmatch_pt_base
            & (data["jet_muflav"] == 0)
            & (data["jet_tauflav"] == 1)
            & (data["jet_taucharge"] < 0)
            & (data["jet_elflav"] == 0)
        ),  # Tau -
        "muon": (
            genmatch_pt_base
            & (data["jet_muflav"] == 1)
            & (data["jet_tauflav"] == 0)
            & (data["jet_elflav"] == 0)
        ),  # muon
        "electron": (
            genmatch_pt_base
            & (data["jet_muflav"] == 0)
            & (data["jet_tauflav"] == 0)
            & (data["jet_elflav"] == 1)
        ),  # electron
    }

    # Automatically generate class labels based on the order of keys in conditions
    class_labels = {label: idx for idx, label in enumerate(conditions)}

    # Initialize the new array in data for numeric labels with default -1 for unmatched entries
    data["class_label"] = ak.full_like(data["jet_genmatch_pt"], -1)

    # Assign numeric values based on conditions using awkward's where function
    for label, condition in conditions.items():
        data["class_label"] = ak.where(
            condition, class_labels[label], data["class_label"]
        )

    # Set pt regression target
    hadrons = (
        conditions["b"]
        | conditions["charm"]
        | conditions["light"]
        | conditions["gluon"]
    )
    leptons = (
        conditions["taup"]
        | conditions["taum"]
        | conditions["muon"]
        | conditions["electron"]
    )

    hadron_pt_ratio = ak.nan_to_num(
        data["jet_genmatch_pt"] / data["jet_pt_phys"], nan=0, posinf=0, neginf=0
    )
    lepton_pt_ratio = ak.nan_to_num(
        (data["jet_genmatch_lep_vis_pt"] / data["jet_pt_phys"]),
        nan=0,
        posinf=0,
        neginf=0,
    )

    hadron_pt = ak.nan_to_num(data["jet_genmatch_pt"], nan=0, posinf=0, neginf=0)
    lepton_pt = ak.nan_to_num(
        (data["jet_genmatch_lep_vis_pt"]), nan=0, posinf=0, neginf=0
    )

    data["target_pt"] = np.clip(
        hadrons * hadron_pt_ratio + leptons * lepton_pt_ratio, 0.3, 2
    )
    data["target_pt_phys"] = hadrons * hadron_pt + leptons * lepton_pt

    # Apply pt_cut
    jet_ptmin_gen = data["target_pt_phys"] > 5.0
    for key in conditions:
        conditions[key] = conditions[key] & jet_ptmin_gen

    # Sanity check for data consistency
    split_data_sum = sum(
        sum(conditions[label]) for label, condition in conditions.items()
    )
    if split_data_sum != len(data[jet_ptmin_gen]):
        raise ValueError(
            f"""Data splitting error: Total entries ({split_data_sum})
            do not match the filtered data length ({len(data[jet_ptmin_gen])})."""
        )

    return data[jet_ptmin_gen], class_labels


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
        "target_pt",
        "target_pt_phys",
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


def get_targets(target_labels, all_labels):
    """
    Get the targets based on given label names. Combine multiple labels if needed.
    (e.g. target_labels = ['bb', 'cc', 'qq', 'qcd'], all_labels = ['Top_bWqq', 'Top_Wqq', 'H_qq', 'H_bb', 'H_cc', 'QCD_others'] -> [3, 4, [0,1,2], 5])
    """
    target_indices = []
    for target in target_labels:
        target = target.lower()
        idx = np.where([target in label.lower() for label in all_labels])[0].tolist()
        if len(idx) == 0:
            combined_idx = []
            for i, label in enumerate(all_labels):
                label = label.lower()
                if target in label:
                    combined_idx.append(i)
            target_indices.append(combined_idx)
        else:
            target_indices.append(idx)
    return target_indices


def to_ML(
    data,
    val_ratio=None,
    n_particles=None,
    shuffle_constits=False,
    use_jets=False,
    target_labels=None,
    all_labels=all_labels,
):
    """
    Take in the data from make_data (loaded by load_data) and make them ready for training.
    """

    constit_data = np.asarray(data["nn_inputs"])

    constit_feats = (
        constit_data if n_particles is None else constit_data[:, :n_particles, :]
    )
    if shuffle_constits:
        indices = np.arange(constit_feats.shape[1])
        constit_feats[:, indices] = constit_feats[:, np.random.permutation(indices)]
    if use_jets:
        try:
            X = (constit_feats, np.asarray(data["nn_jet_inputs"]))
        except KeyError:
            raise KeyError(
                "Error: jet-level features not found in data. Please check your dataset or the tag used."
            )
    else:
        X = constit_feats

    # Get target indices
    if target_labels is None:
        target_labels = all_labels
    targets = get_targets(target_labels, all_labels)
    y = np.zeros((len(data), len(targets)), dtype=np.float32)
    for idx, t in enumerate(targets):
        if isinstance(t, list):
            # Combine multiple labels
            combined = np.sum(np.asarray(data["class_label"][:, t]), axis=1)
            combined = np.clip(combined, 0, 1)  # Ensure binary values
            y[:, idx] = combined
        else:
            y[:, idx] = np.asarray(data["class_label"][:, t])
    # remove entries where all targets are 0
    valid_samples = np.where(np.sum(y, axis=1) > 0)[0]
    X = X[valid_samples]
    y = y[valid_samples]
    print(f"Class distribution after filtering: {y.sum(axis=0)} ")

    if val_ratio is not None and val_ratio > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None

    return X_train, y_train, X_val, y_val


def load_np_data(
    path,
    percentage=None,
    val_ratio=None,
    n_particles=None,
    shuffle_constits=False,
    fields=None,
    target_labels=None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    Load a specified percentage of the dataset using numpy files.
    Parameters:
        X_path (str): The path to the numpy file containing input features.
        Y_path (str): The path to the numpy file containing target labels.
        percentage (float): The percentage of TOTAL data to load (0-100).
        val_ratio (float): how much of the total data would be used for validation (0-1) [Default = 0.2]
        n_particles (int, optional): Number of constituent particles to load. If None, load all.
        fields (list, optional): Specific features to load. If None, load all features.
        shuffle_constits (bool, optional): Whether to shuffle constituent particles within each jet.
    Returns:
        train_data (np.ndarray): The training data.
        test_data (np.ndarray): The testing data.
        input_vars (list): List of input variable names.

    """
    if percentage is None:
        percentage = 100.0
    if val_ratio is None:
        val_ratio = 0.2
    print("Loading data from: ", path)
    print("Loading percentage: ", percentage)
    print("With validation ratio of: ", val_ratio)
    data = np.load(path)
    X = data["inputs"]
    Y = data["targets"]
    input_vars = data.get("feature_labels", None)
    class_labels = data.get("class_labels", None)
    extra_vars = data.get("extra_vars", None)
    # Choose specific fields if provided and exist in data
    if fields is not None and input_vars is not None:
        feature_labels = input_vars
        # Get indices of the requested fields
        field_indices = [
            np.where(feature_labels == f)[0][0] for f in fields if f in feature_labels
        ]
        X = X[:, :, field_indices]

    # Shuffle and select the specified percentage of data
    total_data_len = len(X)
    data_to_load = int(np.ceil((percentage / 100) * total_data_len))
    indices = np.arange(total_data_len)
    np.random.shuffle(indices)
    X = X[indices[:data_to_load]]
    Y = Y[indices[:data_to_load]]

    # Limit to n_particles if specified and shuffle if requested
    if n_particles is not None:
        X = X[:, :n_particles, :]
    if shuffle_constits:
        X = np.array([np.random.permutation(x) for x in X])

    # Split into training and validation sets
    if val_ratio > 0:
        total_data_len = len(X)
        split_index = int((1 - val_ratio) * total_data_len)
        train_X = X[:split_index, :, :]
        train_Y = Y[:split_index, :]
        train_data = {"inputs": train_X, "targets": train_Y}
        val_X = X[split_index:, :, :]
        val_Y = Y[split_index:, :]
        val_data = {"inputs": val_X, "targets": val_Y}

        return train_data, val_data, class_labels, input_vars, extra_vars
    data = {"inputs": X, "targets": Y}
    return data, None, class_labels, input_vars, extra_vars


def load_data(
    path,
    percentage=None,
    fields=None,
):
    """
    Load a specified percentage of the dataset using uproot.concatenate.

    Parameters:
        outdir (str): The output directory containing the data chunks.
        percentage (float): The percentage of TOTAL data to load (0-100).
        fields (list, optional): Specific fields to load. If None, load all fields.

    Returns:
        awkward.Array: Concatenated data arrays from selected chunks.
    """
    if percentage is None:
        percentage = 100.0

    print("Loading data from: ", path)
    print("Loading percentage: ", percentage)

    # Load metadata to determine chunks to load
    metadata_file = os.path.join(path, "metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    total_chunks = len(metadata)
    chunks_to_load = int(np.ceil((percentage / 100) * total_chunks))

    # Collect the file paths for the chunks to load
    # chunk_files = [metadata[i]["file"] for i in range(chunks_to_load)]

    chunk_files = [
        metadata[int(i * np.floor((1 / (percentage / 100))))]["file"] + ":data"
        for i in range(chunks_to_load)
    ]

    # Use uproot.concatenate to load and combine data from multiple files
    data = uproot.concatenate(chunk_files, filter_name=fields, library="ak")
    # print(ak.sum(data["class_label"], axis=0))
    # Load corresponding metadata for classlabels/input variables
    data_metadata_file = os.path.join(path, "variables.json")
    with open(data_metadata_file, "r") as f:
        variables = json.load(f)
        class_labels = variables["outputs"]
        input_vars = variables["inputs"]
        extra_vars = variables["extras"]

    return data, class_labels, input_vars, extra_vars


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

            # Add additional response variables
            # _add_response_vars(data)
            # Split data into all the training classes
            # data_split, class_labels = _split_flavor(data)
            data, class_labels = _define_target(data, all_labels)

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
