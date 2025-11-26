import torch
from torch.utils.data import DataLoader
from tagger.train.cli import parse_cli
from tagger.data.datasets import ConstituentsDataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tagger.data.tools import load_np_data

# TODO: add type hints
# TODO: distributed training, multiple GPUs (limit to single GPU for now)


def train(model, config, device=None, **kwargs):
    training_config = config.get("training_config", {})
    data_config = config.get("data_config", {})
    output = config.get("output", "output")
    logger = kwargs["logger"] if "logger" in kwargs else None
    if data_config.get("np_data", True):  # Default to np_data for now
        path = Path(data_config["data_path"])
        if path.is_dir():
            path = (
                path / "train.npz"
            )  # assuming standard naming, otherwise user should provide full path
        path = path.as_posix()
        data_train, data_val = load_np_data(
            path=path,
            val_ratio=data_config["validation_split"],
            percentage=data_config["percentage"],
            n_particles=data_config.get("n_particles", None),
            fields=data_config.get("fields", None),
            shuffle_constits=data_config.get("shuffle_constits", False),
        )
        X_train = data_train["inputs"]
        y_train = data_train["targets"]
        X_val = data_val["inputs"]
        y_val = data_val["targets"]

        # print(
        #    f"Class distribution in training set: {y_train.sum(axis=0)}, Class distribution in validation set: {y_val.sum(axis=0)}"
        # )
    else:
        raise RuntimeError(
            "Only --np-data prototype path is implemented for this runner"
        )

    dataset = ConstituentsDataset(X_train, y_train)
    val_dataset = ConstituentsDataset(X_val, y_val)
    X_sample, y_sample = dataset[0]
    input_shape = X_sample.shape
    output_shape = y_sample.shape
    print(f"Input shape: {input_shape}, output shape: {output_shape}")
    # create DataLoader now that collate is known
    train_loader = DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        collate_fn=(lambda batch: (model.collate_fn(batch)))
        if hasattr(model, "collate_fn")
        else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        collate_fn=(lambda batch: (model.collate_fn(batch)))
        if hasattr(model, "collate_fn")
        else None,
    )
    print(f"Training model on device: {device}")
    print(
        f"Training for {training_config['epochs']} epochs with batch_size={data_config['batch_size']}"
    )

    history = model.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        device=device,
        logger=logger,
        extras=kwargs["extras"] if "extras" in kwargs else {},
    )
    save_path = Path(output) / "model.pt"
    model.save(save_path)
    print(f"Model training complete, model saved to {save_path}")
    if config.get("run_config", {}).get("plotting", False):
        model.plot_training_history(history=history)
    return model


def test(model, config, device=None, **kwargs):
    data_config = config.get("data_config", {})
    output = config.get("output", "output")
    if data_config.get("np_data", True):  # Default to np_data for now
        try:
            from tagger.data.tools import load_np_data
        except Exception:
            raise RuntimeError(
                "loader tagger.data.tools.load_np_data not available in PYTHONPATH"
            )
        # Heuristic: look for files in the provided folder
        path = Path(data_config["data_path"])
        if path.is_dir():
            path = (
                path / "test.npz"
            )  # assuming standard naming, otherwise user should provide full path
        path = path.as_posix()
        data_test, _ = load_np_data(
            path=path,
            val_ratio=0.0,
            percentage=100,
            n_particles=data_config.get("n_particles", None),
            fields=data_config.get("fields", None),
            shuffle_constits=data_config.get("shuffle_constits", False),
        )
        X_test = data_test["inputs"]
        y_test = data_test["targets"]

    else:
        raise RuntimeError(
            "Only --np-data prototype path is implemented for this runner"
        )

    dataset = ConstituentsDataset(X_test, y_test)
    X_sample, y_sample = dataset[0]
    input_shape = X_sample.shape
    output_shape = y_sample.shape
    print(f"Input shape: {input_shape}, output shape: {output_shape}")
    # create DataLoader now that collate is known
    test_loader = DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        collate_fn=(lambda batch: (model.collate_fn(batch, device)))
        if hasattr(model, "collate_fn")
        else None,
    )

    metrics = model.evaluate(
        test_loader=test_loader,
        device=device,
        eval_metrics=model.model_config.get("eval_metrics", ["accuracy"]),
    )
    print(f"Test metrics: {metrics}")
    # save metrics to output
    metrics_path = Path(output) / "test" / "test_metrics.txt"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    return model


def load_model_from_config(config, output, recreate=False):
    try:
        from tagger.model.common import from_cfg
    except Exception:
        raise RuntimeError("tagger.model.common.from_cfg not available in PYTHONPATH")
    model = from_cfg(config, output, recreate=recreate)
    return model


def main():
    # Parse CLI args and load YAML if provided (moved to tagger.train.cli)
    mode, cfg, extras = parse_cli()
    # print("Mode:", mode)
    # print("Extras:", extras)
    # print(f"Configuration for {mode}: {cfg}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "train":
        output = cfg.get("output", "output")
        model = load_model_from_config(cfg, output, recreate=True)
        print(f"Model created: {model}")
        logger = None
        if cfg.get("run_config", {}).get("logger", False):
            log_path = Path(output) / "logs"
            log_path = log_path
            log_path.mkdir(parents=True, exist_ok=True)
            print(
                f"TensorBoard logging enabled. To view, run: tensorboard --logdir={log_path}"
            )
            logger = SummaryWriter(log_dir=log_path, flush_secs=30)
        train(model, cfg, device=device, extras=extras, logger=logger)
    elif mode == "test":
        # build model from config, don't recreate output dir
        output = cfg.get("output", "output")
        model = load_model_from_config(cfg, output, recreate=False)

        # load weights from checkpoint
        model_path = extras.get("ckpt_path", Path(output) / "model.pth")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model.load(model_path)

        # run testing
        test(model, cfg, device=device, extras=extras)

    else:  # Should not reach here due to argparse enforcement
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
