import torch
from torch.utils.data import DataLoader
from tagger.train.cli import parse_cli
from tagger.data.datasets import ConstituentsDataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tagger.data.tools import load_np_data, load_data, to_ML
import gc

gc.set_threshold(0)
# TODO: add type hints
# TODO: distributed training, multiple GPUs (limit to single GPU for now)


def get_data(data_config, mode="train"):
    X, y, X_val, y_val = None, None, None, None
    class_labels, input_vars, extra_vars = None, None, None
    use_np_loader = False
    path = Path(data_config["data_path"])
    if path.is_dir():
        # check for train.npz
        if (path / "train.npz").exists():
            use_np_loader = True
            path = path / f"{mode}.npz"
    path = path.resolve().as_posix()
    if use_np_loader:
        (data_train, data_val, class_labels, input_vars, extra_vars) = load_np_data(
            path=path,
            val_ratio=data_config.get("validation_split", None),
            percentage=data_config.get("percentage", None),
            n_particles=data_config.get("n_particles", None),
            fields=data_config.get("fields", None),
            shuffle_constits=data_config.get("shuffle_constits", False),
            target_labels=data_config.get("target_labels", None),
        )
        X = data_train["inputs"]
        y = data_train["targets"]
        X_val = data_val["inputs"]
        y_val = data_val["targets"]
    else:
        data, class_labels, input_vars, extra_vars = load_data(
            path=path,
            percentage=data_config.get("percentage", None),
            fields=data_config.get("fields", None),
        )  # returns full data with all existing labels
        # Make into ML-like data for training
        X, y, X_val, y_val = to_ML(
            data,
            all_labels=class_labels,
            val_ratio=data_config.get("validation_split", None),
            target_labels=data_config.get("target_labels", None),
            n_particles=data_config.get("n_particles", None),
            shuffle_constits=data_config.get("shuffle_constits", False),
        )
        del data
        gc.collect()
        # Optionally balance classes by downsampling
        if data_config.get("balance_classes", False):
            import numpy as np

            train_indices = []
            val_indices = []
            train_lim = np.min(y.sum(axis=0)).astype(int)  # per class
            val_lim = np.min(y_val.sum(axis=0)).astype(int)
            for c in range(len(class_labels)):
                t_idx = torch.where(torch.argmax(torch.from_numpy(y), axis=1) == c)[0]
                v_idx = torch.where(torch.argmax(torch.from_numpy(y_val), axis=1) == c)[
                    0
                ]
                if len(t_idx) >= train_lim:
                    train_indices.append(t_idx[torch.randperm(len(t_idx))[:train_lim]])
                if len(v_idx) >= val_lim:
                    val_indices.append(v_idx[torch.randperm(len(v_idx))[:val_lim]])
            train_indices = torch.cat(train_indices).numpy()
            val_indices = torch.cat(val_indices).numpy()
            X = X[train_indices]
            y = y[train_indices]
            X_val = X_val[val_indices]
            y_val = y_val[val_indices]
        print(
            f"Training samples per class: {y.sum(axis=0)}\nValidation samples per class: {y_val.sum(axis=0)}"
        )
    class_labels = data_config.get("target_labels", class_labels)
    return X, y, X_val, y_val, class_labels, input_vars, extra_vars


def train(config, device=None, **kwargs):
    training_config = config.get("training_config", {})
    data_config = config.get("data_config", {})
    out_dir = config.get("output", "output")
    # logger = kwargs.get("logger", None)

    X_train, y_train, X_val, y_val, class_labels, input_vars, extra_vars = get_data(
        data_config, mode="train"
    )
    model = load_model_from_config(config, recreate=True)

    model.set_labels(
        input_vars,
        extra_vars,
        class_labels,
    )
    model.compile_model()

    dataset = ConstituentsDataset(X_train, y_train)
    val_dataset = ConstituentsDataset(X_val, y_val)
    X_sample, y_sample = dataset[0]
    input_shape = X_sample.shape
    output_shape = y_sample.shape
    print(f"Input shape: {input_shape}, output shape: {output_shape}")
    sample_weights = None
    sampler = None
    # if data_config.get("balance_classes", False):
    class_weights = y_train.sum(axis=0) / y_train.shape[0]
    class_weights = 1.0 / (class_weights + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_labels)
    sample_weights = torch.tensor(class_weights, dtype=torch.float32)
    sample_weights = sample_weights[y_train.argmax(axis=1)]
    print(f"Sample weights: {class_weights}")
    # print(sample_weights.test)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=False
    )

    train_loader = DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        sampler=sampler if sampler is not None else None,
        shuffle=False if sample_weights is not None else True,
        collate_fn=model.collate_fn if hasattr(model, "collate_fn") else None,
    )
    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config["batch_size"],
            shuffle=False,
            collate_fn=model.collate_fn if hasattr(model, "collate_fn") else None,
        )
    print(f"Training model on device: {device}")
    print(
        f"Training for {training_config['epochs']} epochs with batch_size={data_config['batch_size']}"
    )

    history = model.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        device=device,
        extras=kwargs["extras"] if "extras" in kwargs else {},
    )
    save_path = Path(out_dir) / "model.pt"
    model.save(save_path)
    print(f"Model training complete, model saved to {save_path}")
    if config.get("run_config", {}).get("plotting", False):
        model.plot_training_history(history=history)
    return model


def test(config, device=None, **kwargs):
    extras = kwargs.get("extras", {})
    # build model from config, don't recreate output dir
    model = load_model_from_config(config, recreate=False)
    # load weights from checkpoint
    model_path = extras.get(
        "ckpt_path", Path(config.get("output", "output")) / "model.pt"
    )
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load(model_path)

    data_config = config.get("data_config", {})
    data_config["validation_split"] = 0.0  # Use all data for testing
    X_test, y_test, _, _, class_labels, input_vars, extra_vars = get_data(
        data_config, mode="test"
    )
    model.set_labels(
        input_vars,
        extra_vars,
        class_labels,
    )
    test_dataset = ConstituentsDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        collate_fn=model.collate_fn if hasattr(model, "collate_fn") else None,
    )
    print(f"Testing model on device: {device}")
    results, loss = model.eval(
        test_loader=test_loader,
        device=device,
        extras=extras,
    )
    print(f"Test results: {results}, Test loss: {loss}")

    return results


def load_model_from_config(config, recreate=False):
    try:
        from tagger.model.common import from_cfg
    except Exception:
        raise RuntimeError("tagger.model.common.from_cfg not available in PYTHONPATH")
    model = from_cfg(config, recreate=recreate)
    print(f"Model created: {model}")

    return model


def main():
    # Parse CLI args and load YAML if provided (moved to tagger.train.cli)
    mode, cfg, extras = parse_cli()
    # print("Mode:", mode)
    # print("Extras:", extras)
    # print(f"Configuration for {mode}: {cfg}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "train":
        train(cfg, device=device, extras=extras)
    elif mode == "test":
        # run testing
        test(cfg, device=device, extras=extras)

    else:  # Should not reach here due to argparse enforcement
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
