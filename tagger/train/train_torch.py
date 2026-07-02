"""PyTorch Lightning training entrypoint for TorchTagger models."""

from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from tagger.model.JetTagModel import JetTagModel
from tagger.train.cli import parse_cli
from tagger.data.datasets import ConstituentsDataset
from tagger.data.datamodule import TorchTaggerDataModule


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

    # Build datamodule and load data
    datamodule = TorchTaggerDataModule(data_config)
    datamodule.setup(stage="fit")

    # Create and compile model
    model = load_model_from_config(config, recreate=True)
    model.set_labels(datamodule.input_vars, datamodule.extra_vars, datamodule.class_labels)
    model.compile_model()
    datamodule.set_model(model)  # for collate_fn support

    # Show input/output shape from first sample
    X_sample, y_sample = datamodule._train_dataset[0]
    print(f"Input shape: {X_sample.shape}, Output shape: {y_sample.shape}")

    print(f"Training for {training_config.get('epochs', 10)} epochs")

    # Configure Lightning callbacks and logger
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        filename="best-model",
        monitor="val_loss" if datamodule.has_val_data else "train_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    if training_config.get("callbacks", {}).get("EarlyStopping"):
        early_stop_config = training_config["callbacks"]["EarlyStopping"]
        early_stop = EarlyStopping(
            monitor=early_stop_config.get("monitor", "val_loss"),
            patience=early_stop_config.get("patience", 3),
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stop)

    logger = None
    if run_config.get("logger", False):
        logger = TensorBoardLogger(
            save_dir=str(out_dir),
            name="logs",
            version=None,
        )

    trainer = pl.Trainer(
        max_epochs=training_config.get("epochs", 10),
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=training_config.get("log_interval", 50),
        enable_progress_bar=run_config.get("verbose", 1) >= 1,
        enable_model_summary=run_config.get("verbose", 1) >= 1,
    )

    trainer.fit(model, datamodule=datamodule)

    save_path = out_dir / "model.pt"
    model.save(save_path)
    print(f"Model training complete, saved to {save_path}")

    if run_config.get("plotting", False):
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

    model = load_model_from_config(config, recreate=False)

    model_path = extras.get(
        "ckpt_path", Path(config.get("output", "output")) / "model.pt"
    )
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load(model_path)

    data_config = config.get("data_config", {})
    datamodule = TorchTaggerDataModule(data_config)
    datamodule.setup(stage="test")
    datamodule.set_model(model)

    model.set_labels(datamodule.input_vars, datamodule.extra_vars, datamodule.class_labels)

    print("Testing model")

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_progress_bar=run_config.get("verbose", 1) >= 1,
    )

    results = trainer.test(model, datamodule=datamodule)
    print(f"Test results: {results}")
    return results


def load_model_from_config(config, recreate=False) -> JetTagModel:
    """Load a model from configuration."""
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
