"""Jet Tag Model base class and additional functionality for model registering
Adapted from Keras style API for model building and training
Written 28/05/2025, cebrown@cern.ch
"""

import functools
import json
import os
from abc import ABC, abstractmethod
import torch
import tagger.src.metrics as metrics


class JetTagModel(ABC):
    """Parent Class for Jet Tag Models

    Abstract Base Class not for use directly
    """

    def __init__(self, config, output_dir: str):
        """
        Args:
            output_dir (str): Saving directory for model artefacts
        """
        self.config = config
        self.output_directory = output_dir

        #
        self.model = None
        self.hls_model = None
        self.log_interval = None

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

        # Training attribute
        self.optimizer = {}
        self.history = None
        self.lr_scheduler_cfg = {}
        self.lr_scheduler = None
        self.loss_fn = None
        self.lr = None
        self.batch_size = None
        self.epochs = None
        self.betas = None
        self.weight_decay = None
        self.momentum = None

    def compile_model(self, **kwargs):
        """
        Compile the model, adding loss function and callbacks
        """

        self.quantization_config = self.config.get("quantization_config", {})
        self.training_config = self.config.get("training_config", {})
        self.run_config = self.config.get("run_config", {})
        self.model_config = self.config.get("model_config", {})
        self.hls4ml_config = self.config.get("hls4ml_config", {})
        self.data_config = self.config.get("data_config", {})
        self.build_model()
        optim = self.configure_optimizer()
        self.optimizer = optim["optimizer"]
        self.lr_scheduler = optim["lr_scheduler"]
        self.loss_fn = self.configure_loss()

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

    def save_decorator(save_func):
        """Decorator used to include additional
        saving functionality for child classes
        """

        @functools.wraps(save_func)
        def wrapper(self, out_dir: str = "None"):
            """Wrapper adding saving functionality

            Args:
                out_dir (str): Where to save the model. Defaults to
                None but overridden to output_directory.
            """
            if out_dir == "None":
                out_dir = self.output_directory
            # Save additional jsons associated with model
            # Dump input variables
            with open(os.path.join(out_dir, "input_vars.json"), "w") as f:
                json.dump(self.input_vars, f, indent=4)
            # Dump extra variables
            with open(os.path.join(out_dir, "extra_vars.json"), "w") as f:
                json.dump(self.extra_vars, f, indent=4)
            # Dump class variables
            with open(os.path.join(out_dir, "class_labels.json"), "w") as f:
                json.dump(self.class_labels, f, indent=4)
            # Do the rest of the saving, defined in child class
            save_func(self, out_dir)

        return wrapper

    def load_decorator(load_func):
        """Decorator used to include additional
        loading functionality for child classes
        """

        @functools.wraps(load_func)
        def wrapper(self, out_dir: str = "None"):
            """Wrapper adding loading functionality

            Args:
                out_dir (str): Where to load the model from. Defaults to
                None but overridden to output_directory.
            """
            if out_dir == "None":
                out_dir = self.output_directory
            # Save additional jsons associated with model
            # Dump input variables
            with open(os.path.join(out_dir, "input_vars.json"), "r") as f:
                self.input_vars = json.load(f)
            # Dump extra variables
            with open(os.path.join(out_dir, "class_labels.json"), "r") as f:
                self.class_labels = json.load(f)
            # Dump class variables
            with open(os.path.join(out_dir, "extra_vars.json"), "r") as f:
                self.extra_vars = json.load(f)
            # Do the rest of the loading, defined in child class
            load_func(self, out_dir)

        return wrapper

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

    def plot_training_history(self, history: dict = None):
        """Plot the training history of the model"""

        out_dir = self.output_directory
        # Produce some basic plots with the training for diagnostics
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
        # Configure optimizer
        print(
            f"Configuring optimizer: {self.optimizer} with params: {self.optimizer_params}"
        )
        if self.optimizer == "adam":
            optimizer = Adam(self.model.parameters(), **self.optimizer_params)
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                **self.optimizer_params,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                **self.optimizer_params,
            )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        if self.scheduler == "cosine":
            self.lr_scheduler_cfg = self.training_config.get("scheduler_params", {})
            self.lr_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, **self.lr_scheduler_cfg
            )
        elif self.scheduler == "reduce_on_plateau":
            self.lr_scheduler_cfg = self.training_config.get("scheduler_params", {})
            self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, **self.lr_scheduler_cfg
            )
        elif self.scheduler == "cosine_with_warmup":  # cosine decay with linear warmup
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
            self.lr_scheduler = lr_scheduler.SequentialLR(
                optimizer=optimizer,
                schedulers=[linear, cosine],
                milestones=[self.lr_scheduler_cfg.get("warmup_steps", 10)],
            )
        else:
            self.lr_scheduler = lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        return {"optimizer": optimizer, "lr_scheduler": self.lr_scheduler}

    def configure_loss(self):
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
        """Configure callbacks for the model"""
        pass

    def configure_logger(self):
        """Configure logger for the model"""
        pass

    def configure_metrics(self):
        """Configure metrics for the model"""
        pass

    def summary(self):
        """Print the model summary"""
        print("Model Summary:")
        print("--------------------------------------------")
        print("Layer Name   --   Size   --   # Parameters  ")
        print("--------------------------------------------")
        for name, param in self.model.named_parameters():
            print(f"{name} -- {param.size()} -- {param.numel()}")

        print("--------------------------------------------")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader = None,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> None:
        """
        Tensorflowesque fit method for training the model
        Args:
            train_loader: DataLoader for training data
            validation_loader: DataLoader for validation data
            device: Device to run the training on
        """
        epochs = self.training_config.get("epochs", 10)
        logs = {}
        global_step = 0

        for epoch in range(epochs):
            self.callbacks.on_epoch_begin(epoch)
            self.model.train()
            running_loss = 0.0
            total = 0
            training_targets = []
            training_outputs = []
            # Training loop
            for batch_idx, batch in enumerate(train_loader):
                self.callbacks.on_batch_begin(batch_idx)
                self.optimizer.zero_grad()
                outputs, targets = self.shared_step(batch, device)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total += targets.size(0)
                global_step += 1

                training_outputs.append(outputs.detach().cpu())
                training_targets.append(targets.detach().cpu())

                self.callbacks.on_batch_end(batch_idx)

            self.callbacks.on_training_step_end(global_step)

            training_outputs = torch.cat(training_outputs)
            training_targets = torch.cat(training_targets)
            logs.update(self.metrics(training_outputs, training_targets))

            if validation_loader is not None:
                logs.update(self.eval(validation_loader, device=device, mode="val"))

            self.on_epoch_end(epoch)

        self.history = logs
        return self.history

    def eval(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cpu"),
        mode: str = "validation",
        **kwargs,
    ) -> dict:
        """
        Tensorflowesque eval method for evaluating the model
        Args:
            data_loader: DataLoader for evaluation data
            device: Device to run the evaluation on
            mode: Mode string for evaluation (e.g., 'validation' or 'test')
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        running_loss = 0.0
        total = 0
        outputs = []
        targets = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                output, target = self.shared_step(batch, device)
                loss = self.loss_fn(output, target)
                running_loss += loss.item()
                total += targets.size(0)
                _, predicted = torch.max(output, 1)
                outputs.append(output.cpu())
                targets.append(target.cpu())

        outputs = torch.cat(outputs)
        targets = torch.cat(targets)

        eval_metrics = self.metrics(outputs, targets, mode=mode)
        eval_metrics[f"{mode}_loss"] = running_loss / total
        return eval_metrics

    def on_epoch_begin(self, epoch: int):
        """Hook for actions at the beginning of an epoch"""
        pass

    def on_epoch_end(self, epoch: int):
        """Hook for actions at the end of an epoch"""
        pass

    def on_training_step_end(self, step: int):
        """Hook for actions at the end of a training step"""
        pass

    def on_validation_step_end(self, step: int):
        """Hook for actions at the end of a validation step"""
        pass

    def shared_step(self, batch, device: torch.device):
        """Shared step for training and evaluation

        Args:
            batch: Input batch
            device: Device to run the step on
        Returns:
            tuple: outputs, targets
        """
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            xb, yb = batch[0], batch[1]
        elif isinstance(batch, dict):
            xb, yb = batch["inputs"], batch["targets"]
        else:
            raise ValueError("Unsupported batch format. Expected list, tuple, or dict.")

        inputs, targets = xb.to(device), yb.to(device)
        outputs = (
            self.model(*inputs)
            if isinstance(inputs, (list, tuple))
            else self.model(inputs)
        )
        return outputs, targets


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
        jettag_class = cls.registry[name]
        model = jettag_class(config=config, output_dir=folder, **kwargs)

        return model

    @classmethod
    def list_registered_models(cls) -> list:
        """List all registered models in the JetModelFactory

        Returns:
            list: List of registered model strings
        """
        return list(cls.registry.keys())
