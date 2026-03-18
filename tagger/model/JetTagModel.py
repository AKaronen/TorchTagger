"""Jet Tag Model base class and additional functionality for model registering
Adapted from Keras style API for model building and training
Written 28/05/2025, cebrown@cern.ch
"""

import functools
import json
import os
from abc import ABC, abstractmethod
import torch
from src.metrics import (
    JetTagMetrics,
    ClassificationAccuracy,
    AUROC,
    ConfusionMatrix,
)
from src.callbacks import TBLogger, EarlyStopping, ModelCheckpoint
from tqdm import tqdm


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
        self.stop_training = False

        # Callbacks and logger
        self.callbacks = None
        self.logger = None
        self.log_on_epoch = None
        self.log_interval = None

        # Metrics
        self.metrics = None

        # Other
        self.verbose = None  # Verbosity level; 1=normal, 2=more, 0=silent

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

        self.verbose = self.run_config.get("verbose", 1)
        # Build the model (create the model architecture)
        self.build_model(model_cfg=self.model_config)

        # Configure optimizer, loss function, callbacks, logger, and metrics
        optim = self.configure_optimizer()
        self.optimizer = optim["optimizer"]
        self.lr_scheduler = optim["lr_scheduler"]
        self.loss_fn = self.configure_loss()
        if self.training_config.get("callbacks", None):
            self.configure_callbacks()
        if self.run_config.get("logger", False):
            self.configure_logger()
        self.configure_metrics()

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
            self.lr_scheduler_cfg = self.training_config.get(
                "scheduler_params", {"T_max": self.training_config.get("epochs", 50)}
            )
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

    def configure_loss(self) -> torch.nn.Module:
        """Configure loss function for the model"""
        loss = self.training_config.get(
            "loss", "cross_entropy"
        )  # Default to cross entropy
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
        self.callbacks = []
        callback_configs = self.training_config.get("callbacks", [])
        if self.verbose > 1:
            print(f"Configuring {callback_configs} callbacks")
        for cb_type in callback_configs:
            cb_cfg = callback_configs[cb_type]
            if cb_type == "EarlyStopping":
                cb = EarlyStopping(**cb_cfg)
            elif cb_type == "ModelCheckpoint":
                cb = (
                    ModelCheckpoint(**cb_cfg)
                    if "filepath" in cb_cfg
                    else ModelCheckpoint(
                        filepath=os.path.join(self.output_directory, "best_model.pt"),
                        **cb_cfg,
                    )
                )
            else:
                raise ValueError(f"Callback type {cb_type} not recognized.")
            self.callbacks.append(cb)

    def configure_logger(self):
        """Configure logger for the model"""
        if self.callbacks is None:
            self.callbacks = []
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        tb_log_dir = os.path.join(self.output_directory, "logs")
        self.logger = TBLogger(
            log_dir=tb_log_dir,
            log_interval=self.log_interval,
            log_on_epoch=self.log_on_epoch,
            flush_secs=30,
        )
        self.callbacks.append(self.logger)

    def configure_metrics(self):
        """Configure metrics for the model"""
        metric_configs = self.training_config.get(
            "metrics",
            [
                {"accuracy": None},
            ],
        )
        self.metrics = JetTagMetrics({})
        for m in metric_configs:
            if isinstance(m, dict):
                m_type = list(m.keys())[0]
                m_cfg = m[m_type]
            else:
                m_type = m
                m_cfg = None
            default_kwargs = {
                "class_names": self.class_labels,
                "num_classes": self.model.n_classes
                if hasattr(self.model, "n_classes")
                else len(self.class_labels),
            }  # Provide default kwargs
            if m_cfg is None:
                m_cfg = default_kwargs
            else:
                m_cfg.update(default_kwargs)
            if m_type == "accuracy":
                metric = ClassificationAccuracy(**m_cfg)
            elif m_type == "auroc":
                metric = AUROC(**m_cfg)
            elif m_type == "confusion_matrix":
                metric = ConfusionMatrix(**m_cfg)
            else:
                raise ValueError(f"Metric type {m_type} not recognized.")
            self.metrics.add_metric(
                metric.name if hasattr(metric, "name") else m_type, metric
            )
        if self.verbose > 1:
            print(f"Configured metrics: {list(self.metrics.keys())}")

    def summary(self):
        """Print the model summary"""
        print("Model Summary:")
        print(f"{'=' * 80}")
        print(
            f"{'Module':15} {'Input':15}  {'Output':15} {'# Parameters':15} {'# Trainable':15}"
        )
        print(f"{'=' * 80}")

        def get_first_layer_size(layer):
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

        def get_last_layer_size(layer):
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
        history = {}
        global_step = 0

        for epoch in range(1, epochs + 1):
            self.on_epoch_begin(epoch, global_step, logs)
            self.model.train()
            running_loss = 0.0
            training_targets = []
            training_outputs = []
            train_metrics = {}
            # Training loop
            for idx, batch in tqdm(
                iterable=enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}/{epochs}",
                ncols=80,
                leave=False,
            ):
                self.on_batch_begin(global_step=global_step, logs=logs)
                self.optimizer.zero_grad()
                outputs, targets = self.shared_step(batch, device)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                global_step += 1

                training_outputs.append(outputs.cpu())
                training_targets.append(targets.cpu())

                # Log training metrics
                if self.logger:
                    if self.log_interval and (global_step % self.log_interval) == 0:
                        train_metrics = self.metrics(
                            torch.cat(training_outputs),
                            torch.cat(training_targets),
                            mode="train",
                        )
                        train_metrics["train/loss"] = running_loss / (idx + 1)
                self.on_batch_end(global_step=global_step, logs=train_metrics)

            # End of epoch metrics
            train_metrics = self.metrics(
                torch.cat(training_outputs),
                torch.cat(training_targets),
                mode="train",
            )

            training_outputs = torch.cat(training_outputs)
            training_targets = torch.cat(training_targets)
            logs.update(train_metrics)
            logs["train_loss"] = running_loss / len(train_loader)
            if validation_loader is not None:
                eval_metrics, eval_loss = self.eval(
                    validation_loader, device=device, mode="val"
                )
                logs.update(eval_metrics)
                logs["val_loss"] = eval_loss
            try:
                self.lr_scheduler.step(metrics=logs["val_loss"])
            except TypeError:
                self.lr_scheduler.step()

            # Update history
            for key, value in logs.items():
                if key not in history:
                    history[key] = []
                history[key].append(value)
            if self.verbose >= 1:
                print(
                    f"Epoch {epoch}/{epochs} - "
                    + ", ".join(
                        [
                            f"{k}: {v[-1]:.4f}"
                            for k, v in history.items()
                            if "loss" in k
                            or "accuracy" in k
                            and isinstance(v[-1], float)
                        ]
                    )
                )
            self.on_epoch_end(epoch, global_step=global_step, logs=logs)
            if self.stop_training:
                break

        self.history = history
        return self.history

    def eval(
        self,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cpu"),
        mode: str = "val",
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
            float: Evaluation loss
        """
        self.model.eval()
        running_loss = 0.0
        outputs = []
        targets = []
        with torch.no_grad():
            for batch in data_loader:
                output, target = self.shared_step(batch, device)
                loss = self.loss_fn(output, target)
                running_loss += loss.item()
                outputs.append(output.cpu())
                targets.append(target.cpu())

        outputs = torch.cat(outputs)
        targets = torch.cat(targets)

        eval_metrics = self.metrics(outputs, targets, mode=mode)
        eval_loss = running_loss / len(data_loader)
        return eval_metrics, eval_loss

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
        if isinstance(xb, torch.Tensor) and isinstance(yb, torch.Tensor):
            inputs, targets = xb.to(device), yb.to(device)
        elif (
            isinstance(xb, (list, tuple))
            and all(isinstance(x, torch.Tensor) for x in xb)
            and isinstance(yb, torch.Tensor)
        ):
            inputs = [x.to(device) for x in xb]
            targets = yb.to(device)
        else:
            raise ValueError(
                "Unsupported input format. Expected tensors or list/tuple of tensors."
            )
        outputs = (
            self.model(*inputs)
            if isinstance(inputs, (list, tuple))
            else self.model(inputs)
        )
        return outputs, targets

    def on_epoch_begin(self, epoch, global_step=None, logs=None):
        """Hook for begin of epoch actions

        Args:
            epoch (int): Current epoch number
            global_step (int, optional): Current global step number
            logs (dict, optional): Dictionary of logs containing monitored metrics
        """
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, global_step, logs)
        # Can be extended in child classes for additional functionality

    def on_epoch_end(self, epoch, global_step=None, logs=None):
        """Hook for end of epoch actions

        Args:
            epoch (int): Current epoch number
            global_step (int, optional): Current global step number
            logs (dict, optional): Dictionary of logs containing monitored metrics
        """
        for cb in self.callbacks:
            try:
                cb.on_epoch_end(epoch, global_step, logs)
            except Exception as e:
                if hasattr(cb, "on_exception"):
                    self.stop_training = cb.on_exception(e, self.model)
                else:
                    raise e
        # Can be extended in child classes for additional functionality

    def on_batch_begin(self, global_step=None, logs=None):
        """Hook for begin of batch actions
        Args:
            global_step (int, optional): Current global step
            logs (dict, optional): Dictionary of logs containing monitored metrics
        """
        for cb in self.callbacks:
            try:
                cb.on_batch_begin(global_step, logs)
            except Exception as e:
                if hasattr(cb, "on_exception"):
                    self.stop_training = cb.on_exception(e, self.model)
                else:
                    raise e
        # Can be extended in child classes for additional functionality

    def on_batch_end(self, global_step=None, logs=None):
        """Hook for end of batch actions

        Args:
            global_step (int, optional): Current global step
            logs (dict, optional): Dictionary of logs containing monitored metrics
        """
        for cb in self.callbacks:
            try:
                cb.on_batch_end(global_step, logs)
            except Exception as e:
                if hasattr(cb, "on_exception"):
                    self.stop_training = cb.on_exception(e, self.model)
                else:
                    raise e
        # Can be extended in child classes for additional functionality


################################--------------------------------------####################################
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
