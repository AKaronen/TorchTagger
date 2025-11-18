"""Jet Tag Model base class and additional functionality for model registering

Written 28/05/2025, cebrown@cern.ch
"""

import functools
import json
import os
from abc import ABC, abstractmethod
import torch


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
        self.jet_model = None
        self.hls_jet_model = None
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
    def fit(self, **kwargs) -> None:
        """
        Fit the model to the training data
        Must be written for child class
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
            optimizer = Adam(self.jet_model.parameters(), **self.optimizer_params)
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.jet_model.parameters(),
                **self.optimizer_params,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.jet_model.parameters(),
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
