import torch
from torch.utils.data import DataLoader
from tagger.train.cli import parse_cli
from tagger.data.datasets import ConstituentsDataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tagger.data.tools import load_np_data, load_data, to_ML
import numpy as np


# TODO: add type hints
# TODO: distributed training, multiple GPUs (limit to single GPU for now)


def get_data(data_config):
    X_train, y_train, X_val, y_val = None, None, None, None
    class_labels, input_vars, extra_vars = None, None, None
    if data_config.get("np_data", True):  # Default to np_data for now
        path = Path(data_config["data_path"])
        if path.is_dir():
            path = (
                path / "train.npz"
            )  # assuming standard naming, otherwise user should provide full path
        path = path.as_posix()
        data_train, data_val = load_np_data(
            path=path,
            val_ratio=data_config.get("validation_split", None),
            percentage=data_config.get("percentage", None),
            n_particles=data_config.get("n_particles", None),
            fields=data_config.get("fields", None),
            shuffle_constits=data_config.get("shuffle_constits", False),
            target_labels=data_config.get("target_labels", None),
        )
        X_train = data_train["inputs"]
        y_train = data_train["targets"]
    else:
        path = Path(data_config["data_path"]).as_posix()
        data, class_labels, input_vars, extra_vars = load_data(
            path=path,
            percentage=data_config.get("percentage", None),
            fields=data_config.get("fields", None),
        )  # returns full data with all existing labels
        # Make into ML-like data for training
        X_train, y_train, X_val, y_val = to_ML(
            data,
            all_labels=class_labels,
            val_ratio=data_config.get("validation_split", None),
            target_labels=data_config.get("target_labels", None),
            n_particles=data_config.get("n_particles", None),
            shuffle_constits=data_config.get("shuffle_constits", False),
        )
        # if data_config.get("balance_classes", False):
        #    train_indices = []
        #    val_indices = []
        #    train_lim = np.min(y_train.sum(axis=0)).astype(int)  # per class
        #    val_lim = np.min(y_val.sum(axis=0)).astype(int)
        #    for c in range(len(class_labels)):
        #        t_idx = torch.where(
        #            torch.argmax(torch.from_numpy(y_train), axis=1) == c
        #        )[0]
        #        v_idx = torch.where(torch.argmax(torch.from_numpy(y_val), axis=1) == c)[
        #            0
        #        ]
        #        if len(t_idx) >= train_lim:
        #            train_indices.append(t_idx[torch.randperm(len(t_idx))[:train_lim]])
        #        if len(v_idx) >= val_lim:
        #            val_indices.append(v_idx[torch.randperm(len(v_idx))[:val_lim]])
        #    train_indices = torch.cat(train_indices).numpy()
        #    val_indices = torch.cat(val_indices).numpy()
        #    X_train = X_train[train_indices]
        #    y_train = y_train[train_indices]
        #    X_val = X_val[val_indices]
        #    y_val = y_val[val_indices]
        print(
            f"Training samples per class: {y_train.sum(axis=0)}\nValidation samples per class: {y_val.sum(axis=0)}"
        )
        class_labels = data_config.get("target_labels", class_labels)
    return X_train, y_train, X_val, y_val, class_labels, input_vars, extra_vars


def train(model, config, device=None, **kwargs):
    training_config = config.get("training_config", {})
    data_config = config.get("data_config", {})
    output = config.get("output", "output")
    logger = kwargs.get("logger", None)

    X_train, y_train, X_val, y_val, class_labels, input_vars, extra_vars = get_data(
        data_config
    )

    model.set_labels(
        input_vars,
        extra_vars,
        class_labels,
    )

    dataset = ConstituentsDataset(X_train, y_train)
    val_dataset = ConstituentsDataset(X_val, y_val)
    X_sample, y_sample = dataset[0]
    input_shape = X_sample.shape
    output_shape = y_sample.shape
    print(f"Input shape: {input_shape}, output shape: {output_shape}")
    sample_weights = None
    sampler = None
    if data_config.get("balance_classes", False):
        class_weights = y_train.sum(axis=0) / y_train.shape[0]
        class_weights = 1.0 / (class_weights + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_labels)
        sample_weights = torch.tensor(class_weights, dtype=torch.float32)
        sample_weights = sample_weights[y_train.argmax(axis=1)]
        print(f"Sample weights: {class_weights}")
        # print(sample_weights.test)
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights)
        )

    train_loader = DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        sampler=sampler if sampler is not None else None,
        shuffle=False if sample_weights is not None else True,
        collate_fn=(lambda batch: model.collate_fn(batch))
        if hasattr(model, "collate_fn")
        else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        collate_fn=(lambda batch: model.collate_fn(batch))
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

        if model.model_config.get("from_pretrained", False):
            pretrained_path = model.model_config.get("ckpt_path", None)
            if pretrained_path is None:
                raise ValueError(
                    "Model config specifies from_pretrained=True but no pretrained_model_path provided"
                )
            pretrained_path = Path(pretrained_path)
            if not pretrained_path.exists():
                raise FileNotFoundError(
                    f"Pretrained model file not found: {pretrained_path}"
                )
            model.load(pretrained_path)
            if model.model_config.get("fine_tune", False):
                for param in model.jet_model.named_parameters():
                    if not param[0].startswith("fc"):
                        param[1].requires_grad = False
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
