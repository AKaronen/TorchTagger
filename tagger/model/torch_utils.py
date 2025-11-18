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
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    accuracy = correct / total
    return accuracy
