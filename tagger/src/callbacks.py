import torch


class Callbacks:
    """Collection of callback functions for model training and evaluation"""

    def __init__(
        self,
    ):
        self.name = "callbacks"
        self.callbacks = []

    def on_epoch_start(self, epoch, logs=None):
        """Called at the beginning of each epoch"""
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, logs=None):
        """Called at the beginning of training"""
        for callback in self.callbacks:
            callback.on_batch_begin(logs)

    def on_batch_end(self, logs=None):
        """Called at the end of training"""
        for callback in self.callbacks:
            callback.on_batch_end(logs)

    def on_training_step_end(self, logs=None):
        """Called at the end of each training step"""
        for callback in self.callbacks:
            callback.on_training_step_end(logs)

    def on_validation_step_end(self, logs=None):
        """Called at the end of each validation step"""
        for callback in self.callbacks:
            callback.on_validation_step_end(logs)


class Callback:
    """Base class for callbacks"""

    def __init__(self, name: str):
        self.name = name

    def on_epoch_start(self, epoch, logs=None):
        """
        Called at the beginning of each epoch
        Args:
            epoch (int): Current epoch number
            logs (dict, optional): Dictionary of logs
        """
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch
        Args:
            epoch (int): Current epoch number
            logs (dict, optional): Dictionary of logs
        """
        pass

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of each batch
        Args:
            batch (int): Current batch number
            logs (dict, optional): Dictionary of logs
        """
        pass

    def on_batch_begin(self, logs=None):
        """
        Called at the beginning of training
        Args:
            logs (dict, optional): Dictionary of logs
        """
        pass

    def on_training_step_end(self, logs=None):
        """
        Called at the end of each training step
        Args:
            logs (dict, optional): Dictionary of logs
        """
        pass

    def on_validation_step_end(self, logs=None):
        """
        Called at the end of each validation step
        Args:
            logs (dict, optional): Dictionary of logs
        """
        pass


class TBLogger:
    """Tensorboard Logger for logging metrics"""

    def __init__(self, log_dir: str, **kwargs):
        """Initialize TBLogger callback
        Args:
            log_dir (str): Directory to save TensorBoard logs
            **kwargs: Additional arguments for SummaryWriter
        """
        super().__init__("tb_logger")
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir, **kwargs)
        self.step = 0

    def log(self, logs: dict):
        """Log metrics to TensorBoard
        Args:
            logs (dict): Dictionary of metrics to log
        """
        for key, value in logs.items():
            self.writer.add_scalar(key, value, self.step)
        self.step += 1

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()


class EarlyStopping(Callback):
    """Early stopping callback to stop training when a monitored metric stops improving"""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """Initialize EarlyStopping callback
        Args:
            monitor (str, optional): Metric to monitor. Defaults to 'val_loss'.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 5.
            min_delta (float, optional): Minimum change in the monitored metric to qualify as an improvement. Defaults to 0.0.
            mode (str, optional): One of {'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has stopped increasing. Defaults to 'min'.
        """
        super().__init__("early_stopping")
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.num_bad_epochs = 0
        if mode == "min":
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_value = float("inf")
        elif mode == "max":
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_value = -float("inf")
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def on_epoch_end(self, epoch, logs=None):
        """Check if training should be stopped at the end of an epoch
        Args:
            epoch (int): Current epoch number
            logs (dict, optional): Dictionary of logs containing monitored metrics
        """
        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        if self.monitor_op(current_value, self.best_value):
            self.best_value = current_value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"No improvement in {self.patience} consecutive epochs."
            )
            raise StopIteration


class ModelCheckpoint(Callback):
    """Model checkpoint callback to save the model at specified intervals"""

    def __init__(
        self,
        filepath: str,
        save_best_only: bool = False,
        monitor: str = "val_loss",
        mode: str = "min",
        frequency: int = 1,
    ):
        """Initialize ModelCheckpoint callback
        Args:
            filepath (str): Path to save the model file
            save_best_only (bool, optional): If True, only save the model when the monitored metric improves. Defaults to False.
            monitor (str, optional): Metric to monitor. Defaults to 'val_loss'.
            mode (str, optional): One of {'min', 'max'}. In 'min' mode, the monitored metric is expected to decrease; in 'max' mode it is expected to increase. Defaults to 'min'.
            frequency (int, optional): Save the model every 'frequency' epochs. Defaults to 1.
        """
        super().__init__("model_checkpoint")
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best_value = None
        self.frequency = frequency
        if mode == "min":
            self.monitor_op = lambda current, best: current < best
            self.best_value = float("inf")
        elif mode == "max":
            self.monitor_op = lambda current, best: current > best
            self.best_value = -float("inf")
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def on_epoch_end(self, epoch, logs=None):
        """Save the model at the end of an epoch
        Args:
            epoch (int): Current epoch number
            logs (dict, optional): Dictionary of logs containing monitored metrics
        """
        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        if (epoch + 1) % self.frequency == 0:
            if self.save_best_only:
                if self.monitor_op(current_value, self.best_value):
                    torch.save(logs["model_state_dict"], self.filepath)
                    print(f"Model improved. Saved to {self.filepath}")
            else:
                torch.save(logs["model_state_dict"], self.filepath)
                print(f"Model saved to {self.filepath}")
