"""PyTorch Lightning DataModule for TorchTagger."""

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from tagger.data.datasets import ConstituentsDataset
from tagger.data.tools import load_np_data, load_data, to_ML


class TorchTaggerDataModule(pl.LightningDataModule):
    """LightningDataModule for TorchTagger data loading pipeline."""

    REQUIRED_KEYS = ["data_path", "batch_size"]

    def __init__(self, data_config: dict) -> None:
        super().__init__()
        for key in self.REQUIRED_KEYS:
            if key not in data_config:
                raise ValueError(f"Missing required data_config key: '{key}'")
        self.data_config = data_config
        self.model = None  # Set via set_model() before training for collate_fn

        # Populated after setup()
        self.class_labels: Any = None
        self.input_vars: Any = None
        self.extra_vars: Any = None

        self._train_dataset: ConstituentsDataset | None = None
        self._val_dataset: ConstituentsDataset | None = None
        self._test_dataset: ConstituentsDataset | None = None
        self._train_y: np.ndarray | None = None
        self._class_weights: np.ndarray | None = None

    def set_model(self, model) -> None:
        """Provide model reference so dataloaders can use model.collate_fn."""
        self.model = model

    @property
    def has_val_data(self) -> bool:
        return self._val_dataset is not None

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", "validate", None) and self._train_dataset is None:
            X, y, X_val, y_val, class_labels, input_vars, extra_vars = _load_data(
                self.data_config, mode="train"
            )
            self.class_labels = class_labels
            self.input_vars = input_vars
            self.extra_vars = extra_vars
            self._train_y = y
            self._train_dataset = ConstituentsDataset(X, y)
            self._class_weights = self._compute_class_weights(y)
            if X_val is not None and y_val is not None:
                self._val_dataset = ConstituentsDataset(X_val, y_val)

        if stage in ("test", None) and self._test_dataset is None:
            test_config = {**self.data_config, "validation_split": 0.0}
            X_test, y_test, _, _, class_labels, input_vars, extra_vars = _load_data(
                test_config, mode="test"
            )
            if self.class_labels is None:
                self.class_labels = class_labels
                self.input_vars = input_vars
                self.extra_vars = extra_vars
            self._test_dataset = ConstituentsDataset(X_test, y_test)

    def _compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        epsilon = self.data_config.get("class_weight_epsilon", 1e-6)
        weights = y.sum(axis=0) / y.shape[0]
        weights = 1.0 / (weights + epsilon)
        weights = weights / weights.sum() * len(weights)
        print(f"Class weights: {weights}")
        return weights

    @property
    def _collate_fn(self):
        return getattr(self.model, "collate_fn", None) if self.model else None

    def train_dataloader(self) -> DataLoader:
        if self.data_config.get("weighted_sampling", True):
            sample_weights = torch.tensor(self._class_weights, dtype=torch.float32)
            sample_weights = sample_weights[self._train_y.argmax(axis=1)]
            sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, len(sample_weights), replacement=False
            )
            return DataLoader(
                self._train_dataset,
                batch_size=self.data_config["batch_size"],
                sampler=sampler,
                shuffle=False,  # mutually exclusive with sampler
                collate_fn=self._collate_fn,
            )
        return DataLoader(
            self._train_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self._val_dataset is None:
            return None
        return DataLoader(
            self._val_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            collate_fn=self._collate_fn,
        )


def _load_data(
    data_config: dict, mode: str = "train"
) -> tuple[Any, Any, Any | None, Any | None, Any, Any, Any]:
    """Load data from config (NumPy .npz or ROOT format).

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

        if data_config.get("balance_classes", False):
            train_lim = np.min(y.sum(axis=0)).astype(int)
            train_indices = []
            for c in range(len(class_labels)):
                t_idx = torch.where(torch.argmax(torch.from_numpy(y), axis=1) == c)[0]
                if len(t_idx) >= train_lim:
                    train_indices.append(t_idx[torch.randperm(len(t_idx))[:train_lim]])
            X = X[torch.cat(train_indices).numpy()]
            y = y[torch.cat(train_indices).numpy()]

            if X_val is not None and y_val is not None:
                val_lim = np.min(y_val.sum(axis=0)).astype(int)
                val_indices = []
                for c in range(len(class_labels)):
                    v_idx = torch.where(
                        torch.argmax(torch.from_numpy(y_val), axis=1) == c
                    )[0]
                    if len(v_idx) >= val_lim:
                        val_indices.append(v_idx[torch.randperm(len(v_idx))[:val_lim]])
                X_val = X_val[torch.cat(val_indices).numpy()]
                y_val = y_val[torch.cat(val_indices).numpy()]

    print(
        f"Training samples per class: {y.sum(axis=0)}\n"
        f"Validation samples per class: "
        f"{y_val.sum(axis=0) if y_val is not None else 0}"
    )
    class_labels = data_config.get("target_labels", class_labels)
    return X, y, X_val, y_val, class_labels, input_vars, extra_vars
