import torch

try:
    from torch_scatter import scatter_add

    _HAS_SCATTER = True
except Exception:
    _HAS_SCATTER = False


def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Sum values in `data` according to `segment_ids`.

    Args:
        data: (E, D) tensor
        segment_ids: (E,) long tensor with values in [0, num_segments)
        num_segments: int

    Returns:
        (num_segments, D) tensor
    """
    if _HAS_SCATTER:
        return scatter_add(data, segment_ids, dim=0, dim_size=num_segments)
    # fallback using index_add_
    device = data.device
    out = torch.zeros((num_segments, data.size(1)), device=device, dtype=data.dtype)
    out.index_add_(0, segment_ids, data)
    return out


def unsorted_segment_mean(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    if _HAS_SCATTER:
        sums = scatter_add(data, segment_ids, dim=0, dim_size=num_segments)
        counts = scatter_add(
            torch.ones((data.size(0), 1), device=data.device, dtype=data.dtype),
            segment_ids,
            dim=0,
            dim_size=num_segments,
        )
        return sums / counts.clamp(min=1.0)
    sums = unsorted_segment_sum(data, segment_ids, num_segments)
    # compute counts
    device = data.device
    counts = torch.zeros((num_segments, 1), device=device, dtype=data.dtype)
    ones = torch.ones((data.size(0), 1), device=device, dtype=data.dtype)
    counts.index_add_(0, segment_ids, ones)
    return sums / counts.clamp(min=1.0)


def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy given true and predicted labels

    Args:
        y_true (torch.Tensor): True labels (can be one-hot or class indices)
        y_pred (torch.Tensor): Predicted labels

    Returns:
        float: Accuracy value
    """
    if y_true.dim() > 1:
        y_true = torch.argmax(y_true, dim=1)
    if y_pred.dim() > 1:
        y_pred = torch.argmax(y_pred, dim=1)
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    accuracy = correct / total
    return accuracy


def per_class_accuracy(y_true, y_pred, class_names):
    """Calculate per-class accuracy given true and predicted labels

    Args:
        y_true (torch.Tensor): True labels (can be one-hot or class indices)
        y_pred (torch.Tensor): Predicted labels
        num_classes (int): Number of classes

    Returns:
        dict: Dictionary with per-class accuracy
    """
    if y_true.dim() > 1:
        y_true = torch.argmax(y_true, dim=1)
    if y_pred.dim() > 1:
        y_pred = torch.argmax(y_pred, dim=1)
    per_class_acc = dict(zip(class_names, [0.0] * len(class_names)))
    for cls_idx, cls_name in enumerate(class_names):
        cls_mask = y_true == cls_idx
        correct = (y_true[cls_mask] == y_pred[cls_mask]).sum().item()
        total = cls_mask.sum().item()
        acc = correct / total if total > 0 else 0.0
        per_class_acc[cls_name] = acc
    return per_class_acc


def compute_confusion_matrix(y_true, y_pred, class_names):
    """Compute confusion matrix given true and predicted labels

    Args:
        y_true (torch.Tensor): True labels (can be one-hot or class indices)
        y_pred (torch.Tensor): Predicted labels
        class_names (list): List of class names

    Returns:
        torch.Tensor: Confusion matrix
    """
    if y_true.dim() > 1:
        y_true = torch.argmax(y_true, dim=1)
    if y_pred.dim() > 1:
        y_pred = torch.argmax(y_pred, dim=1)
    num_classes = len(class_names)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def plot_confusion_matrix(cm, class_names, normalize: str | bool = False):
    """Plot confusion matrix using matplotlib

    Args:
        cm (torch.Tensor): Confusion matrix
        class_names (list): List of class names
        normalize (str | bool): Whether to normalize the confusion matrix.
            If True or "trues", normalize by true labels (rows).
            If "preds", normalize by predicted labels (columns).
            If False, do not normalize.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    format_str = "d"
    if normalize or normalize == "trues":
        cm = cm.float()
        cm = cm / cm.sum(dim=1, keepdim=True).clamp(min=1.0)
        format_str = ".2f"
    elif normalize == "preds":
        cm = cm.float()
        cm = cm / cm.sum(dim=0, keepdim=True).clamp(min=1.0)
        format_str = ".2f"

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm.numpy(), interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.size(0)),
        yticks=np.arange(cm.size(0)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.size(0)):
        for j in range(cm.size(1)):
            ax.text(
                j,
                i,
                format(cm[i, j].item(), format_str),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return fig
