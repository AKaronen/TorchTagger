"""Jet Tag Model base class and additional functionality for model registering
Adapted from Keras style API for model building and training
Migrated to PyTorch Lightning
Written 28/05/2025, cebrown@cern.ch
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torchmetrics
from torchmetrics.classification import Accuracy, AUROC, ConfusionMatrix
from torchmetrics import MetricCollection
from tqdm import tqdm


class JetTagModel(pl.LightningModule):
    """Parent Class for Jet Tag Models

    Adapted from Keras style API, now subclasses pl.LightningModule.
    """

    def __init__(self, config, output_dir: str):
        """
        Args:
            config: Configuration dictionary
            output_dir (str): Saving directory for model artefacts
        """
        super().__init__()
        self.config = config
        self.output_directory = output_dir

        # Model architecture
        self.model = None
        self.hls_model = None
        self.log_interval = None

        # Labels and metadata
        self.input_vars = []
        self.extra_vars = []
        self.class_labels = []

        # Higher level config dictionaries
        self.run_config = {}
        self.model_config = {}
        self.quantization_config = {}
        self.training_config = {}
        self.hls4ml_config = {}
        self.data_config = {}

        # Training attributes
        self.optimizer_name = None
        self.optimizer_params = {}
        self.lr_scheduler_cfg = {}
        self.lr_scheduler = None
        self.loss_fn = None
        self.lr = None
        self.batch_size = None
        self.epochs = None
        self.betas = None
        self.weight_decay = None
        self.momentum = None
        self.scheduler = None
        self.stop_training = False

        # Callbacks and logger
        self.callbacks = []
        self.logger_obj = None
        self.log_on_epoch = None
        self.log_interval = None

        # Metrics (Lightning-style MetricCollections)
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

        # Other
        self.verbose = None

    def compile_model(self, **kwargs):
        """
        Compile the model, adding loss function and configuring training.
        """
        self.quantization_config = self.config.get("quantization_config", {})
        self.training_config = self.config.get("training_config", {})
        self.run_config = self.config.get("run_config", {})
        self.model_config = self.config.get("model_config", {})
        self.hls4ml_config = self.config.get("hls4ml_config", {})
        self.data_config = self.config.get("data_config", {})

        self.verbose = self.run_config.get("verbose", 1)

        # Build the model architecture
        self.build_model(model_cfg=self.model_config)

        # Configure optimizer, loss function, and metrics
        # Note: configure_optimizer() is called by Lightning during setup
        self.loss_fn = self.configure_loss()
        self.configure_metrics()

        # Legacy callback/logger configuration (can be extended by subclasses)
        if self.training_config.get("callbacks", None):
            self.configure_callbacks()
        if self.run_config.get("logger", False):
            self.configure_logger()

    @abstractmethod
    def build_model(self, **kwargs):
        """
        Build the model layers, must be written for child class
        """

    @abstractmethod
    def hls4ml_convert(self, **kwargs):
        """
        Convert the model in hls4ml
        Must be written for child class
        """

    def set_labels(self, input_vars: str, extra_vars: str, class_labels: str):
        """Set internal labels

        Args:
            input_vars (str): Input variable names
            extra_vars (str): Extra variable names
            class_labels (str): Class label names
        """
        self.input_vars = input_vars
        self.extra_vars = extra_vars
        self.class_labels = class_labels

    def plot_training_history(self, history: dict):
        """Plot the training history of the model"""

        out_dir = self.output_directory
        plot_path = os.path.join(out_dir, "plots/training")
        os.makedirs(plot_path, exist_ok=True)
        try:
            from tagger.plot.basic import training_history

            print(f"Plotting training history to {plot_path}")
            training_history(plot_path, history)
        except ImportError:
            print(
                "Could not import training_history from tagger.plot.basic. Skipping training history plot."
            )

    def configure_optimizer(self):
        """Configure optimizer for the model"""
        from torch.optim import Adam, AdamW, SGD, lr_scheduler

        # Set training attributes from config
        for key, value in self.training_config.items():
            setattr(self, key, value)
        if not hasattr(self, "optimizer_params"):
            self.optimizer_params = {}

        # Get optimizer name (may be set by setattr above)
        optimizer_name = self.training_config.get("optimizer", "adamw")

        print(
            f"Configuring optimizer: {optimizer_name} with params: {self.optimizer_params}"
        )

        if optimizer_name == "adam":
            optimizer = Adam(self.model.parameters(), **self.optimizer_params)
        elif optimizer_name == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                **self.optimizer_params,
            )
        elif optimizer_name == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                **self.optimizer_params,
            )
        else:
            raise ValueError(
                f"{optimizer_name} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # Configure scheduler
        scheduler_name = self.training_config.get("scheduler", "cosine")

        if scheduler_name == "cosine":
            self.lr_scheduler_cfg = self.training_config.get(
                "scheduler_params", {"T_max": self.training_config.get("epochs", 50)}
            )
            lr_sched = lr_scheduler.CosineAnnealingLR(optimizer, **self.lr_scheduler_cfg)
        elif scheduler_name == "reduce_on_plateau":
            self.lr_scheduler_cfg = self.training_config.get("scheduler_params", {})
            lr_sched = lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, **self.lr_scheduler_cfg
            )
        elif scheduler_name == "cosine_with_warmup":
            self.lr_scheduler_cfg = self.training_config.get("scheduler_params", {})
            T_max = self.training_config.get("epochs", 50) - self.lr_scheduler_cfg.get(
                "warmup_steps", 10
            )
            linear = lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.05,
                total_iters=self.lr_scheduler_cfg.get("warmup_steps", 10),
            )
            cosine = lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=T_max,
                eta_min=self.lr_scheduler_cfg.get("eta_min", 0),
            )
            lr_sched = lr_scheduler.SequentialLR(
                optimizer=optimizer,
                schedulers=[linear, cosine],
                milestones=[self.lr_scheduler_cfg.get("warmup_steps", 10)],
            )
        else:
            lr_sched = lr_scheduler.LambdaLR(optimizer, lambda _: 1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_sched,
                "monitor": "val_loss" if scheduler_name == "reduce_on_plateau" else None,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def configure_loss(self) -> torch.nn.Module:
        """Configure loss function for the model"""
        loss = self.training_config.get("loss", "cross_entropy")
        if loss == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif loss == "mse":
            return torch.nn.MSELoss()
        else:
            raise ValueError(
                f"Loss function {loss} not supported. ['cross_entropy', 'mse']"
            )

    def configure_callbacks(self):
        """Configure callbacks for the model (legacy support)"""
        self.callbacks = []
        callback_configs = self.training_config.get("callbacks", [])
        if self.verbose and self.verbose > 1:
            print(f"Configuring {callback_configs} callbacks")
        # Note: Lightning's built-in callbacks are used instead
        # This is kept for backward compatibility

    def configure_logger(self):
        """Configure logger for the model (legacy support)"""
        if self.callbacks is None:
            self.callbacks = []
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        # Note: Lightning's logger is configured by Trainer
        # This is kept for backward compatibility

    def configure_metrics(self):
        """Configure metrics using torchmetrics.MetricCollection"""
        metric_configs = self.training_config.get(
            "metrics",
            [
                {"accuracy": None},
            ],
        )

        # Get number of classes
        num_classes = (
            self.model.n_classes
            if hasattr(self.model, "n_classes")
            else len(self.class_labels)
        )

        # Build metric collections for train/val/test
        metrics_dict = {}
        for m in metric_configs:
            if isinstance(m, dict):
                m_type = list(m.keys())[0]
                m_cfg = m[m_type] or {}
            else:
                m_type = m
                m_cfg = {}

            if m_type == "accuracy":
                metrics_dict["accuracy"] = Accuracy(
                    task="multiclass", num_classes=num_classes
                )
            elif m_type == "auroc":
                metrics_dict["auroc"] = AUROC(task="multiclass", num_classes=num_classes)
            elif m_type == "confusion_matrix":
                metrics_dict["cm"] = ConfusionMatrix(
                    task="multiclass", num_classes=num_classes
                )

        # Create separate metric collections for each phase
        self.train_metrics = MetricCollection(metrics_dict, prefix="train_")
        self.val_metrics = MetricCollection(metrics_dict, prefix="val_")
        self.test_metrics = MetricCollection(metrics_dict, prefix="test_")

        if self.verbose and self.verbose > 1:
            print(f"Configured metrics: {list(metrics_dict.keys())}")

    def summary(self):
        """Print the model summary"""
        print("Model Summary:")
        print(f"{'=' * 80}")
        print(
            f"{'Module':15} {'Input':15}  {'Output':15} {'# Parameters':15} {'# Trainable':15}"
        )
        print(f"{'=' * 80}")

        def get_first_layer_size(layer) -> list[Any] | Any | Literal["N/A"]:
            for children in layer.children():
                try:
                    if hasattr(children, "weight"):
                        if hasattr(children, "in_features"):
                            return [children.in_features]
                        return list(children.weight.size())
                    else:
                        return get_first_layer_size(children)
                except AttributeError:
                    continue
            return "N/A"

        def get_last_layer_size(layer) -> list[Any] | Any | Literal["N/A"]:
            for children in reversed(list(layer.children())):
                try:
                    if hasattr(children, "weight"):
                        if hasattr(children, "out_features"):
                            return [children.out_features]
                        return list(children.weight.size())
                    else:
                        return get_last_layer_size(children)
                except AttributeError:
                    continue
            return "N/A"

        for children in self.model.named_children():
            layer_name = children[0]
            layer = children[1]
            layer_params = sum(p.numel() for p in layer.parameters())
            trainable_params = sum(
                p.numel() for p in layer.parameters() if p.requires_grad
            )

            if (
                hasattr(layer, "named_children")
                and len(list(layer.named_children())) > 0
            ):
                layer_input_size = get_first_layer_size(layer)
                layer_output_size = get_last_layer_size(layer)
            elif hasattr(layer, "weight"):
                layer_input_size = (
                    [layer.in_features]
                    if hasattr(layer, "in_features")
                    else list(layer.weight.size())
                )
                layer_output_size = (
                    [layer.out_features]
                    if hasattr(layer, "out_features")
                    else list(layer.weight.size())
                )
            else:
                layer_input_size = "N/A"
                layer_output_size = "N/A"
            print(
                f"{layer_name:15}{str(layer_input_size):15}{str(layer_output_size):15}{layer_params:10}{trainable_params:15}",
            )

        print(f"{'=' * 80}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

    def training_step(self, batch, batch_idx):
        """Lightning training step"""
        outputs, targets = self.shared_step(batch)
        loss = self.loss_fn(outputs, targets)

        # Update metrics
        self.train_metrics.update(outputs, targets)

        # Log loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Lightning validation step"""
        outputs, targets = self.shared_step(batch)
        loss = self.loss_fn(outputs, targets)

        # Update metrics
        self.val_metrics.update(outputs, targets)

        # Log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Lightning test step"""
        outputs, targets = self.shared_step(batch)
        loss = self.loss_fn(outputs, targets)

        # Update metrics
        self.test_metrics.update(outputs, targets)

        # Log loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Compute and log aggregated metrics
        if self.train_metrics:
            train_metric_vals = self.train_metrics.compute()
            self.log_dict(train_metric_vals, on_epoch=True)
            self.train_metrics.reset()

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Compute and log aggregated metrics
        if self.val_metrics:
            val_metric_vals = self.val_metrics.compute()
            self.log_dict(val_metric_vals, on_epoch=True)
            self.val_metrics.reset()

    def on_test_epoch_end(self):
        """Called at the end of test epoch"""
        # Compute and log aggregated metrics
        if self.test_metrics:
            test_metric_vals = self.test_metrics.compute()
            self.log_dict(test_metric_vals, on_epoch=True)
            self.test_metrics.reset()

    def shared_step(self, batch):
        """Shared step for training and evaluation

        Args:
            batch: Input batch. Format can be list/tuple (inputs, targets) or dict with keys "inputs" and "targets".
        Returns:
            tuple: outputs, targets
        """
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            xb, yb = batch[0], batch[1]
        elif isinstance(batch, dict):
            xb, yb = batch["inputs"], batch["targets"]
        else:
            raise ValueError("Unsupported batch format. Expected list, tuple, or dict.")

        # Move to device (Lightning handles this automatically, but explicit is clearer)
        if isinstance(xb, torch.Tensor) and isinstance(yb, torch.Tensor):
            inputs, targets = xb, yb
        elif (
            isinstance(xb, (list, tuple))
            and all(isinstance(x, torch.Tensor) for x in xb)
            and isinstance(yb, torch.Tensor)
        ):
            inputs = xb
            targets = yb
        else:
            raise ValueError(
                "Unsupported input format. Expected tensors or list/tuple of tensors."
            )

        if self.model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before training."
            )

        # Forward pass
        outputs = (
            self.model(*inputs)
            if isinstance(inputs, (list, tuple))
            else self.model(inputs)
        )
        return outputs, targets

    def on_epoch_begin(self, epoch, global_step=None, logs=None):
        """Hook for begin of epoch actions (legacy support)"""
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, global_step, logs)

    def on_epoch_end(self, epoch, global_step=None, logs=None):
        """Hook for end of epoch actions (legacy support)"""
        for cb in self.callbacks:
            try:
                cb.on_epoch_end(epoch, global_step, logs)
            except Exception as e:
                if hasattr(cb, "on_exception"):
                    self.stop_training = cb.on_exception(e, self.model)
                else:
                    raise e

    def on_batch_begin(self, global_step=None, logs=None):
        """Hook for begin of batch actions (legacy support)"""
        for cb in self.callbacks:
            try:
                cb.on_batch_begin(global_step, logs)
            except Exception as e:
                if hasattr(cb, "on_exception"):
                    self.stop_training = cb.on_exception(e, self.model)
                else:
                    raise e

    def on_batch_end(self, global_step=None, logs=None):
        """Hook for end of batch actions (legacy support)"""
        for cb in self.callbacks:
            try:
                cb.on_batch_end(global_step, logs)
            except Exception as e:
                if hasattr(cb, "on_exception"):
                    self.stop_training = cb.on_exception(e, self.model)
                else:
                    raise e

    def save(self, out_dir: str | Path = "model_output"):
        """Save the model to the output directory

        Args:
            out_dir (str | Path): Where to save the model. Defaults to "model_output".
        """
        raise NotImplementedError(
            "Save method not implemented for base JetTagModel. Must be implemented in child class."
        )

    def load(self, out_dir: str | Path = "model_output"):
        """Load the model from the output directory

        Args:
            out_dir (str | Path): Where to load the model from. Defaults to "model_output".
        """
        raise NotImplementedError(
            "Load method not implemented for base JetTagModel. Must be implemented in child class."
        )


################################################################################
class JetModelFactory:
    """The factory class for creating Jet Tag Models"""

    registry = {}
    """ Internal registry for available Jet Tag Models """

    @classmethod
    def register(cls, name: str):
        """Decorator for registering new jet tag models

        Args:
            name (str): Name of the model
        """

        def inner_wrapper(wrapped_class: JetTagModel):
            if name in cls.registry:
                print(f"Jet Tagger Model {name} already exists. Will replace it")
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_JetTagModel(
        cls, name: str, config: dict, folder: str, **kwargs
    ) -> "JetTagModel":
        """Factory command to create the Jet Tag Model"""
        try:
            jettag_class = cls.registry[name]
        except KeyError:
            raise ValueError(
                f"Jet Tag Model {name} not found in registry. Available models: {list(cls.registry.keys())}"
            )
        model = jettag_class(config=config, output_dir=folder, **kwargs)
        return model

    @classmethod
    def list_registered_models(cls) -> list:
        """List all registered models in the JetModelFactory

        Returns:
            list: List of registered model strings
        """
        return list(cls.registry.keys())
