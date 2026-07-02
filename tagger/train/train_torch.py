"""PyTorch Lightning training entrypoint for TorchTagger models."""

from typing import Any
from pathlib import Path

import gc
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from tagger.model.JetTagModel import JetTagModel
from tagger.train.cli import parse_cli
from tagger.data.datasets import ConstituentsDataset
from tagger.data.tools import load_np_data, load_data, to_ML

gc.set_threshold(0)


def get_data(
    data_config, mode="train"
) -> tuple[Any, Any, Any | None, Any | None, Any, Any, Any]:
    """Load data from config (NumPy or ROOT format).

    Returns:
        X, y, X_val, y_val, class_labels, input_vars, extra_vars
    """
    X, y, X_val, y_val = None, None, None, None
    class_labels, input_vars, extra_vars = None, None, None
    use_np_loader = False
    path = Path(data_config["data_path"])

    if path.is_dir():
        if (path / "train.npz").exists():
            use_np_loader = True
            path = path / f"{mode}.npz"

    path = path.resolve().as_posix()

    if use_np_loader:
        data_train, data_val, class_labels, input_vars, extra_vars = load_np_data(
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
        )
        X, y, X_val, y_val = to_ML(
            data,
            val_ratio=data_config.get("validation_split", None),
            target_labels=data_config.get("target_labels", None),
            n_particles=data_config.get("n_particles", None),
            shuffle_constits=data_config.get("shuffle_constits", False),
        )
        del data
        gc.collect()

        # Optional class balancing by downsampling
        if data_config.get("balance_classes", False):
            import numpy as np

            train_indices = []
            val_indices = []
            val_lim = None
            train_lim = np.min(y.sum(axis=0)).astype(int)
            if X_val is not None and y_val is not None:
                val_lim = np.min(y_val.sum(axis=0)).astype(int)

            for c in range(len(class_labels)):
                t_idx = torch.where(torch.argmax(torch.from_numpy(y), axis=1) == c)[0]
                if len(t_idx) >= train_lim:
                    train_indices.append(t_idx[torch.randperm(len(t_idx))[:train_lim]])

                if val_lim is not None:
                    v_idx = torch.where(
                        torch.argmax(torch.from_numpy(y_val), axis=1) == c
                    )[0]
                    if len(v_idx) >= val_lim:
                        val_indices.append(v_idx[torch.randperm(len(v_idx))[:val_lim]])

            train_indices = torch.cat(train_indices).numpy()
            if val_lim is not None and X_val is not None and y_val is not None:
                val_indices = torch.cat(val_indices).numpy()
                X_val = X_val[val_indices]
                y_val = y_val[val_indices]
            X = X[train_indices]
            y = y[train_indices]

    print(
        f"Training samples per class: {y.sum(axis=0)}\n"
        f"Validation samples per class: {y_val.sum(axis=0) if y_val is not None else 0}"
    )
    class_labels = data_config.get("target_labels", class_labels)
    return X, y, X_val, y_val, class_labels, input_vars, extra_vars


def get_dataloaders(data_config, X_train, y_train, X_val, y_val, model):
    """Create PyTorch DataLoaders with WeightedRandomSampler."""
    # Create datasets
    train_dataset = ConstituentsDataset(X_train, y_train)

    # Compute class weights for balanced sampling
    class_weights = y_train.sum(axis=0) / y_train.shape[0]
    class_weights = 1.0 / (class_weights + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    sample_weights = torch.tensor(class_weights, dtype=torch.float32)
    sample_weights = sample_weights[y_train.argmax(axis=1)]
    print(f"Class weights: {class_weights}")

    # Weighted sampler for balanced training
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=False
    )

    # Train loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        sampler=sampler,
        shuffle=False,  # Disable shuffle when using sampler
        collate_fn=model.collate_fn if hasattr(model, "collate_fn") else None,
    )

    # Validation loader (no sampling)
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = ConstituentsDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config["batch_size"],
            shuffle=False,
            collate_fn=model.collate_fn if hasattr(model, "collate_fn") else None,
        )

    return train_loader, val_loader


def train(config, device=None, **kwargs) -> JetTagModel:
    """Train a model using PyTorch Lightning.

    Args:
        config: Configuration dictionary
        device: Device to train on (legacy, Lightning handles this)
        **kwargs: Extra arguments including 'extras' dict

    Returns:
        Trained model
    """
    training_config = config.get("training_config", {})
    data_config = config.get("data_config", {})
    run_config = config.get("run_config", {})
    out_dir = Path(config.get("output", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, y_train, X_val, y_val, class_labels, input_vars, extra_vars = get_data(
        data_config, mode="train"
    )

    # Create and compile model
    model = load_model_from_config(config, recreate=True)
    model.set_labels(input_vars, extra_vars, class_labels)
    model.compile_model()

    # Show model info
    X_sample, y_sample = ConstituentsDataset(X_train, y_train)[0]
    print(f"Input shape: {X_sample.shape}, Output shape: {y_sample.shape}")

    # Create data loaders
    train_loader, val_loader = get_dataloaders(
        data_config, X_train, y_train, X_val, y_val, model
    )

    print(f"Training for {training_config.get('epochs', 10)} epochs")

    # Configure Lightning callbacks and logger
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-model",
        monitor="val_loss" if val_loader else "train_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback (if configured)
    if training_config.get("callbacks", {}).get("EarlyStopping"):
        early_stop_config = training_config["callbacks"]["EarlyStopping"]
        early_stop = EarlyStopping(
            monitor=early_stop_config.get("monitor", "val_loss"),
            patience=early_stop_config.get("patience", 3),
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stop)

    # TensorBoard logger
    logger = None
    if run_config.get("logger", False):
        logger = TensorBoardLogger(
            save_dir=str(out_dir),
            name="logs",
            version=None,
        )

    # Create Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=training_config.get("epochs", 10),
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=training_config.get("log_interval", 50),
        enable_progress_bar=run_config.get("verbose", 1) >= 1,
        enable_model_summary=run_config.get("verbose", 1) >= 1,
        # Optional: uncomment for mixed precision
        # precision="16-mixed",
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save final checkpoint
    save_path = out_dir / "model.pt"
    model.save(save_path)
    print(f"Model training complete, saved to {save_path}")

    # Plot training history if requested
    if run_config.get("plotting", False):
        # Note: In Lightning, training history is in trainer.logger,
        # but for backward compatibility we could reconstruct it
        pass

    return model


def test(config, device=None, **kwargs):
    """Test a model using PyTorch Lightning.

    Args:
        config: Configuration dictionary
        device: Device to test on (legacy, Lightning handles this)
        **kwargs: Extra arguments including 'extras' dict

    Returns:
        Test results
    """
    extras = kwargs.get("extras", {})
    run_config = config.get("run_config", {})

    # Load model from config (don't recreate output dir)
    model = load_model_from_config(config, recreate=False)

    # Load checkpoint
    model_path = extras.get(
        "ckpt_path", Path(config.get("output", "output")) / "model.pt"
    )
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load(model_path)

    # Load test data
    data_config = config.get("data_config", {}).copy()
    data_config["validation_split"] = 0.0  # Use all data for testing
    X_test, y_test, _, _, class_labels, input_vars, extra_vars = get_data(
        data_config, mode="test"
    )

    model.set_labels(input_vars, extra_vars, class_labels)

    # Create test loader
    test_dataset = ConstituentsDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        collate_fn=model.collate_fn if hasattr(model, "collate_fn") else None,
    )

    print(f"Testing model")

    # Create Lightning Trainer for testing
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_progress_bar=run_config.get("verbose", 1) >= 1,
    )

    # Run test
    results = trainer.test(model, dataloaders=test_loader)

    print(f"Test results: {results}")
    return results


def load_model_from_config(config, recreate=False) -> JetTagModel:
    """Load a model from configuration.

    Args:
        config: Configuration dictionary
        recreate: Whether to recreate the output directory

    Returns:
        JetTagModel instance
    """
    try:
        from tagger.model.common import from_cfg
    except Exception:
        raise RuntimeError("tagger.model.common.from_cfg not available in PYTHONPATH")

    model = from_cfg(config, recreate=recreate)
    print(f"Model created: {model}")
    return model


def main() -> None:
    """Main entry point for training/testing."""
    mode, cfg, extras = parse_cli()

    if mode == "train":
        train(cfg, extras=extras)
    elif mode == "test":
        test(cfg, extras=extras)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
