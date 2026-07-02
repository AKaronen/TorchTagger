import json
import os
import shutil

import numpy as np


def prepare_output_dir(outdir):
    """Prepare output directory with an overwrite confirmation prompt."""
    if os.path.exists(outdir):
        confirm = input(
            f"The directory '{outdir}' already exists. Do you want to delete it and continue? [y/n]: "
        )
        if confirm.lower() == "y":
            shutil.rmtree(outdir)
            print(f"Deleted existing directory: {outdir}")
        else:
            print("Exiting without making changes.")
            return False
    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)
    return True


def class_labels_to_metadata(class_labels):
    if isinstance(class_labels, dict):
        return class_labels
    return dict(zip(class_labels, range(len(class_labels))))


def resolve_class_labels(config_labels, n_classes):
    if isinstance(config_labels, dict):
        return config_labels
    if isinstance(config_labels, (list, tuple)):
        return list(config_labels)
    return [f"class_{idx}" for idx in range(n_classes)]


def save_generic_dataset_metadata(outdir, class_labels, inputs=None, extras=None):
    dataset_metadata_file = os.path.join(outdir, "variables.json")
    metadata = {
        "outputs": class_labels_to_metadata(class_labels),
        "inputs": list(inputs or []),
        "extras": list(extras or []),
    }
    with open(dataset_metadata_file, "w") as handle:
        json.dump(metadata, handle, indent=4)


def class_labels_to_list(class_labels):
    if isinstance(class_labels, dict):
        return [name for name, _ in sorted(class_labels.items(), key=lambda kv: kv[1])]
    return [str(label) for label in class_labels]


def save_numpy_partitions(
    outdir,
    inputs,
    targets,
    feature_labels,
    class_labels,
    extra_vars=None,
    test_split=0.2,
    seed=42,
    shuffle=True,
):
    """Save datasets as train.npz and test.npz in (B, C, F) format."""
    if len(inputs) == 0:
        raise ValueError("No samples available to save.")

    split = float(test_split)
    if split < 0.0:
        split = 0.0
    if split > 0.99:
        split = 0.99

    n_samples = len(inputs)
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(indices)

    if n_samples == 1 or split == 0.0:
        test_count = 0
    else:
        test_count = int(np.round(n_samples * split))
        test_count = min(max(test_count, 1), n_samples - 1)

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    feature_labels_arr = np.asarray(feature_labels or [], dtype=str)
    class_labels_arr = np.asarray(class_labels_to_list(class_labels), dtype=str)
    if extra_vars is not None:
        extra_vars_arr = np.asarray(extra_vars or [], dtype=str)

    np.savez_compressed(
        os.path.join(outdir, "train.npz"),
        inputs=inputs[train_idx],
        targets=targets[train_idx],
        feature_labels=feature_labels_arr,
        class_labels=class_labels_arr,
        extra_vars=extra_vars_arr if extra_vars is not None else None,
    )

    np.savez_compressed(
        os.path.join(outdir, "test.npz"),
        inputs=inputs[test_idx],
        targets=targets[test_idx],
        feature_labels=feature_labels_arr,
        class_labels=class_labels_arr,
        extra_vars=extra_vars_arr if extra_vars is not None else None,
    )

    print(
        f"Saved train/test NumPy partitions: train={len(train_idx)} samples, test={len(test_idx)} samples"
    )
    print(f"Input shape (B, C, F): {inputs.shape}")


def to_one_hot_labels(labels, n_classes=None):
    """Normalize class labels into one-hot arrays."""
    arr = np.asarray(labels)

    if arr.ndim >= 2 and arr.shape[-1] > 1:
        out = arr.astype(np.float32)
        return out, int(out.shape[-1])

    flat = arr.reshape(-1)
    if flat.size == 0:
        classes = int(n_classes or 0)
        return np.zeros((0, classes), dtype=np.float32), classes

    if n_classes is None:
        n_classes = int(np.max(flat)) + 1

    out = np.zeros((len(flat), n_classes), dtype=np.float32)
    idx = flat.astype(np.int64)
    valid = (idx >= 0) & (idx < n_classes)
    out[np.arange(len(idx))[valid], idx[valid]] = 1.0
    return out, int(n_classes)
