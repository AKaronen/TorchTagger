from _collections_abc import dict_keys
import torch
import numpy as np
from sklearn.metrics import auc, roc_curve


class Metric:
    """Base class for metrics"""

    def __init__(self, name: str):
        self.name = name
        self.state_dict = {self.name: []}

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        raise NotImplementedError(
            "Metric __call__ method must be implemented in subclass"
        )

    def state_dict(self) -> dict:
        """Get the state dictionary of the metric

        Returns:
            dict: State dictionary
        """
        return self.state_dict

    def reset(self):
        """Reset the state dictionary of the metric"""
        self.state_dict = {self.name: []}


class JetTagMetrics(dict[str, Metric]):
    """Collection of metrics to be computed together"""

    def __init__(self, metrics: dict):
        """Initialize MetricCollection

        Args:
            metrics (dict): Dictionary of metric name to Metric instance
        """
        super().__init__(metrics)
        self.metrics = metrics

    def __call__(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, mode: str = ""
    ) -> dict:
        """Compute all metrics in the collection

        Args:
            y_pred (torch.Tensor): Predicted labels
            y_true (torch.Tensor): True labels

        Returns:
            dict: Dictionary of metric name to computed value
        """
        results = {}
        for name, metric in self.metrics.items():
            results[f"{mode}_{name}"] = metric(y_true, y_pred)
        self.metric_state = results
        return results

    def __iter__(self):
        return iter(self.metrics)

    def __getitem__(self, key: str) -> Metric:
        return self.metrics[key]

    def __contains__(self, key: str) -> bool:
        return key in self.metrics

    def get_metric(self, name: str) -> "Metric":
        """Get a specific metric by name

        Args:
            name (str): Name of the metric

        Returns:
            Metric: The requested metric instance
        """
        return self.metrics.get(name, None)

    def add_metric(self, name: str, metric: Metric):
        """Add a new metric to the collection

        Args:
            name (str): Name of the metric
            metric (Metric): Metric instance to add
        """
        self.metrics[name] = metric

    def reset(self):
        """Reset all metrics in the collection"""
        for name, metric in self.metrics.items():
            if hasattr(metric, "state_dict") and hasattr(metric, "reset"):
                metric.reset()

    def state_dict(self) -> dict:
        """Get the state dictionary of all metrics

        Returns:
            dict: State dictionary
        """
        state = self.metric_state
        for name, metric in self.metrics.items():
            if hasattr(metric, "state_dict"):
                state[name] = metric.state_dict()
        return state

    def keys(self):
        return self.metrics.keys()


class ClassificationAccuracy(Metric):
    """Classification accuracy metric"""

    def __init__(
        self,
        num_classes: int,
        class_names: list | None = None,
        reduction: str = "macro",
        return_per_class: bool = False,
        name: str = "accuracy",
        **kwargs,
    ):
        """Initialize ClassificationAccuracy metric
        Args:
            num_classes (int): Number of classes
            class_names (list, optional): List of class names. Defaults to None.
            reduction (str, optional): Reduction method ('macro', 'micro', 'weighted'). Defaults to 'macro'.
            return_per_class (bool, optional): Whether to return per-class accuracy. Defaults to False.
        Returns:
            float: Accuracy value
            dict (optional): Per-class accuracy if return_per_class is True
        """
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.class_names = class_names
        self.reduction = reduction
        self.return_per_class = return_per_class
        self.state_dict = {self.name: []}

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        accuracy, per_class_acc = self.classification_accuracy(y_pred, y_true)

        if self.return_per_class:
            self.state_dict[self.name].append((per_class_acc))
            return per_class_acc

        self.state_dict[self.name].append(accuracy)
        return accuracy

    def classification_accuracy(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ):
        """Compute per-class accuracy given true and predicted labels

        Args:
            y_true (torch.Tensor): True labels (can be one-hot or class indices)
            y_pred (torch.Tensor): Predicted labels (can be logits or class indices)

        Returns:
            float: Accuracy value based on reduction method
            dict: Dictionary with per-class accuracy
        """
        class_names = self.class_names
        if class_names is None:
            class_names = [f"{i}" for i in range(self.num_classes)]
        if y_true.dim() > 1:
            y_true = torch.argmax(y_true, dim=1)
        if y_pred.dim() > 1:
            y_pred = torch.argmax(y_pred, dim=1)
        per_class_acc = dict(zip(class_names, [0.0] * len(class_names)))
        accuracy = 0.0
        if self.reduction == "micro":
            correct = (y_true == y_pred).sum().item()
            total = y_true.size(0)
            accuracy = correct / total
            for cls_idx, cls_name in enumerate(
                class_names
            ):  # Compute per-class accuracy as well
                cls_mask = y_true == cls_idx
                correct = (y_true[cls_mask] == y_pred[cls_mask]).sum().item()
                total = cls_mask.sum().item()
                acc = correct / total if total > 0 else 0.0
                per_class_acc[cls_name] = acc
        elif self.reduction == "weighted":
            class_counts = torch.bincount(y_true, minlength=self.num_classes).float()
            total_counts = class_counts.sum().item()
            for cls_idx, cls_name in enumerate(class_names):
                cls_mask = y_true == cls_idx
                correct = (y_true[cls_mask] == y_pred[cls_mask]).sum().item()
                total = cls_mask.sum().item()
                acc = correct / total if total > 0 else 0.0
                weight = (
                    class_counts[cls_idx].item() / total_counts
                    if total_counts > 0
                    else 0.0
                )
                per_class_acc[cls_name] = acc * weight
                accuracy += acc * weight
        else:  # macro
            for cls_idx, cls_name in enumerate(class_names):
                cls_mask = y_true == cls_idx
                correct = (y_true[cls_mask] == y_pred[cls_mask]).sum().item()
                total = cls_mask.sum().item()
                acc = correct / total if total > 0 else 0.0
                per_class_acc[cls_name] = acc
            accuracy = np.mean(list(per_class_acc.values()))
        return accuracy, per_class_acc


class ConfusionMatrix(Metric):
    """Confusion matrix metric"""

    def __init__(
        self,
        num_classes: int,
        class_names: list | None = None,
        name: str = "confusion_matrix",
        **kwargs,
    ):
        """Initialize ConfusionMatrix metric

        Args:
            num_classes (int): Number of classes
            class_names (list, optional): List of class names. Defaults to None.
        """
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.class_names = class_names
        self.state_dict = {self.name: None}

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute confusion matrix given true and predicted labels

        Args:
            y_pred (torch.Tensor): Predicted labels
            y_true (torch.Tensor): True labels

        Returns:
            torch.Tensor: Confusion matrix
        """
        if y_true.dim() > 1:
            y_true = torch.argmax(y_true, dim=1)
        if y_pred.dim() > 1:
            y_pred = torch.argmax(y_pred, dim=1)
        confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
        )
        for t, p in zip(y_true.view(-1), y_pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        self.state_dict[self.name] = confusion_matrix
        return confusion_matrix


class AUROC(Metric):
    """Area Under the Receiver Operating Characteristic Curve (AUROC) metric"""

    def __init__(
        self,
        num_classes: int,
        class_names: list | None = None,
        name: str = "auroc",
        **kwargs,
    ):
        """Initialize AUROC metric
        Args:
            num_classes (int): Number of classes
            class_names (list, optional): List of class names. Defaults to None.
        """
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

        if class_names is None:
            class_names = [f"class_{i}" for i in range(self.num_classes)]

        self.class_names = class_names

        self.roc_dict = (
            {class_label: 0 for class_label in self.class_names}
            if self.class_names
            else {}
        )

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Compute AUROC given true and predicted labels

        Args:
            y_pred (torch.Tensor): Predicted labels
            y_true (torch.Tensor): True labels (can be one-hot or class indices)

        Returns:
            dict: Dictionary with ROC curve data per class
        """
        if y_true.dim() > 1:
            y_true = torch.argmax(y_true, dim=1)
        if y_pred.dim() > 1:
            y_pred = torch.argmax(y_pred, dim=1)

        self.roc_dict = self.compute_auroc(y_true, y_pred)
        self.state_dict[self.name] = self.roc_dict
        return self.roc_dict

    def compute_auroc(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ):
        """Compute ROC curves given true and predicted labels

        Args:
            y_true (torch.Tensor): True labels (can be one-hot or class indices)
            y_pred (torch.Tensor): Predicted labels
        Returns:
            dict: Dictionary  with ROC curve data (fpr, tpr, auc) per class
        """

        roc_dict = {}
        for cls_idx, cls_name in enumerate(self.class_names):
            if y_true.dim() == 1:
                y_true_c = (y_true == cls_idx).numpy()
            else:
                y_true_c = y_true[
                    :, cls_idx
                ].numpy()  # True binary labels for the current class
            if y_pred.dim() == 1:
                y_score = (y_pred == cls_idx).numpy().astype(float)
            else:
                y_score = y_pred[
                    :, cls_idx
                ].numpy()  # Predicted probabilities for the current class
            # Compute FPR, TPR, and AUC
            fpr, tpr, _ = roc_curve(y_true_c, y_score)
            roc_auc = auc(fpr, tpr)
            roc_dict[cls_name] = (fpr, tpr, roc_auc)

        return roc_dict


class MAE(Metric):
    """Mean Absolute Error (MAE) metric"""

    def __init__(self):
        """Initialize MAE metric"""
        super().__init__("mae")

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Compute MAE given true and predicted values

        Args:
            y_pred (torch.Tensor): Predicted values
            y_true (torch.Tensor): True values


        Returns:
            float: MAE value
        """
        mae = torch.mean(torch.abs(y_true - y_pred)).item()
        return mae


class MSE(Metric):
    """Mean Squared Error (MSE) metric"""

    def __init__(self):
        """Initialize MSE metric"""
        super().__init__("mse")

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """Compute MSE given true and predicted values

        Args:
            y_pred (torch.Tensor): Predicted values
            y_true (torch.Tensor): True values

        Returns:
            float: MSE value
        """
        mse = torch.mean((y_true - y_pred) ** 2).item()
        return mse
