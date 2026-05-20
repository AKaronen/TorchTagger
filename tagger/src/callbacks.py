import torch  # type: ignore


class Callback:
    """Base class for callbacks"""

    def __init__(self, name: str):
        self.name = name

    def on_epoch_begin(self, epoch, global_step, logs=None):
        """
        Called at the beginning of each epoch
        Args:
            epoch (int): Current epoch number
            global_step (int): Current global step number
            logs (dict, optional): Dictionary of logs
        """
        pass

    def on_epoch_end(self, epoch, global_step, logs=None):
        """
        Called at the end of each epoch
        Args:
            epoch (int): Current epoch number
            global_step (int): Current global step number
            logs (dict, optional): Dictionary of logs
        """
        pass

    def on_batch_end(self, global_step, logs=None):
        """
        Called at the end of each batch
        Args:
            global_step (int): Current global step number
            logs (dict, optional): Dictionary of logs
        """
        pass

    def on_batch_begin(self, global_step, logs=None):
        """
        Called at the beginning of training
        Args:
            global_step (int): Current global step number
            logs (dict, optional): Dictionary of logs
        """
        pass


class TBLogger(Callback):
    """Tensorboard Logger for logging metrics"""

    def __init__(
        self,
        log_dir: str,
        log_interval: int | None,
        log_on_epoch: bool | None,
        **kwargs,
    ):
        """Initialize TBLogger callback
        Args:
            log_dir (str): Directory to save TensorBoard logs
            log_interval (int, optional): Interval (in steps) to log metrics. Defaults to None.
            log_on_epoch (bool, optional): Whether to log metrics at the end of each epoch. Defaults to True.
            **kwargs: Additional arguments for SummaryWriter
        """
        super().__init__("tb_logger")
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        self.writer = SummaryWriter(log_dir=log_dir, **kwargs)
        self.log_interval = log_interval
        self.log_on_epoch = log_on_epoch if log_on_epoch is not None else True
        print(
            f"TensorBoard logging enabled. To view, run: tensorboard --logdir={log_dir}"
        )

    def log(self, step, logs: dict):
        """Log scalar metrics to TensorBoard
        Args:
            logs (dict): Dictionary of metrics to log
        """

        for key, value in logs.items():
            if "_" in key:
                key = key.replace("_", "/")
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    self.add_scalar(key, value.item(), step)
            if isinstance(value, (int, float)):
                self.add_scalar(key, value, step)

    def add_scalar(self, tag, scalar_value, global_step=None):
        """Add a scalar value to TensorBoard
        Args:
            tag (str): Tag name for the scalar
            scalar_value (float): Scalar value to log
            global_step (int, optional): Global step number
        """
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        """Add multiple scalar values to TensorBoard
        Args:
            main_tag (str): Main tag name for the scalars
            tag_scalar_dict (dict): Dictionary of tag names and scalar values
            global_step (int, optional): Global step number
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def add_figure(self, tag, figure, global_step=None):
        """Add a figure to TensorBoard
        Args:
            tag (str): Tag name for the figure
            figure (matplotlib.figure.Figure): Figure object to log
            global_step (int, optional): Global step number
        """
        self.writer.add_figure(tag, figure, global_step)

    def on_epoch_end(self, epoch, global_step=None, logs=None):
        """Log metrics at the end of each epoch
        Args:
            epoch (int): Current epoch number (not necessarily used)
            global_step (int, optional): Current global step number
            logs (dict, optional): Dictionary of logs containing metrics
        """
        if logs is not None and self.log_on_epoch:
            self.log(global_step, logs)

    def on_batch_end(self, global_step, logs=None):
        """Log metrics at the end of each batch
        Args:
            global_step (int): Current global step number
            logs (dict, optional): Dictionary of logs containing metrics
        """

        if (
            logs is not None
            and self.log_interval
            and (global_step) % self.log_interval == 0
        ):
            self.log(global_step, logs)

    def log_confusion_matrix(self, step, cm, class_names=None):
        """Log confusion matrix to TensorBoard
        Args:
            step (int): Current step number
            cm (torch.Tensor): Confusion matrix tensor
            class_names (list, optional): List of class names for labeling axes
        """
        fig = self._plot_confusion_matrix(cm, class_names)
        self.writer.add_figure("Confusion_Matrix", fig, step)

    def log_roc_curve(self, step, fpr, tpr, auc, class_name):
        """Log ROC curve to TensorBoard
        Args:
            step (int): Current step number
            fpr (array-like): False positive rates
            tpr (array-like): True positive rates
            auc (float): Area under the ROC curve
            class_name (str): Name of the class
        """
        fig = self._plot_roc_curve(fpr, tpr, auc, class_name)
        self.writer.add_figure(f"ROC_Curve_{class_name}", fig, step)

    def _plot_confusion_matrix(self, cm, class_names=None):
        """Plot confusion matrix
        Args:
            cm (torch.Tensor): Confusion matrix tensor
            class_names (list, optional): List of class names for labeling axes
        Returns:
            matplotlib.figure.Figure: Figure object containing the confusion matrix plot
        """
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore

        if class_names is not None:
            plt.xticks(
                ticks=np.arange(len(class_names)),
                labels=class_names,
                rotation=45 if len(class_names) > 10 else 0,
            )
            plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
        fig, ax = plt.subplots()
        cax = ax.matshow(cm.numpy(), cmap=plt.cm.Blues)  # type: ignore
        fig.colorbar(cax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (i, j), z in np.ndenumerate(cm.numpy()):
            ax.text(j, i, f"{z}", ha="center", va="center")

        plt.tight_layout()

        return fig

    def _plot_roc_curve(self, fpr, tpr, auc, class_name):
        """Plot ROC curve for a single class
        Args:
            fpr (array-like): False positive rates
            tpr (array-like): True positive rates
            auc (float): Area under the ROC curve
            class_name (str): Name of the class
        Returns:
            matplotlib.figure.Figure: Figure object containing the ROC curve plot
        """
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {class_name}")
        ax.legend(loc="lower right")
        plt.tight_layout()

        return fig


class EarlyStopping(Callback):
    """Early stopping callback to stop training when a monitored metric stops improving"""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        save_last: bool = False,
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
        self.stop_training = False
        self.save_last = save_last
        if mode == "min":
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_value = float("inf")
        elif mode == "max":
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_value = -float("inf")
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def on_epoch_end(self, epoch, global_step=None, logs=None):
        """Check if training should be stopped at the end of an epoch
        Args:
            epoch (int): Current epoch number
            global_step (int, optional): Current global step number
            logs (dict, optional): Dictionary of logs containing monitored metrics
        """
        if logs is None:
            return

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

    def reset(self):
        """Reset the early stopping state"""
        self.best_value = float("inf") if self.mode == "min" else -float("inf")
        self.num_bad_epochs = 0

    def on_exception(self, exception, model):
        """Handle early stopping exception
        Args:
            exception (Exception): The exception that was raised
        """
        if isinstance(exception, StopIteration):
            self.stop_training = True
            if self.save_last:
                torch.save(model.state_dict(), "last.pt")
        return self.stop_training


class ModelCheckpointException(Exception):
    """Custom exception for model checkpointing"""

    pass


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

    def on_epoch_end(self, epoch, global_step=None, logs=None):
        """Save the model at the end of an epoch
        Args:
            epoch (int): Current epoch number
            global_step (int, optional): Current global step number
            logs (dict, optional): Dictionary of logs containing monitored metrics
        """
        if logs is None:
            return
        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        if (epoch + 1) % self.frequency == 0:
            if self.save_best_only:
                if self.monitor_op(current_value, self.best_value):
                    self.best_value = current_value
                    raise ModelCheckpointException()

            else:
                raise ModelCheckpointException()

    def on_exception(self, exception, model):
        """Handle model saving exception
        Args:
            exception (Exception): The exception that was raised
        """
        if isinstance(exception, ModelCheckpointException):
            torch.save(model.state_dict(), self.filepath)
            # print(f"Model checkpoint saved to {self.filepath}")
        return False  # Do not stop training
