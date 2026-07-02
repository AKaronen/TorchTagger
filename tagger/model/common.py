import os
import shutil

# import tensorflow as tf
# import tensorflow_model_optimization as tfmot
import yaml

# from qkeras.qlayers import QDense
# from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D

from tagger.model.JetTagModel import JetModelFactory, JetTagModel
import torch


#
# class AAtt(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
#    """Attention Layer class
#
#    Args:
#        tf.keras.layers.Layer (_type_): tensorflow layer wrapper
#        tfmot.sparsity.keras.PrunableLayer (_type_): prunable layer wrapper
#    """
#
#    def __init__(self, d_model=16, nhead=2, bits=9, bits_int=2, alpha_val=1, **kwargs):
#        super(AAtt, self).__init__(**kwargs)
#
#        self.d_model = d_model
#        self.n_head = nhead
#        self.bits = bits
#        self.bits_int = bits_int
#        self.alpha_val = alpha_val
#
#        self.qD = QDense(self.d_model, **kwargs)
#        self.kD = QDense(self.d_model, **kwargs)
#        self.vD = QDense(self.d_model, **kwargs)
#        self.outD = QDense(self.d_model, **kwargs)
#
#    def get_config(self):
#        base_config = super().get_config()
#        config = {
#            "d_model": (self.d_model),
#            "nhead": (self.n_head),
#            "bits": (self.bits),
#            "bits_int": (self.bits_int),
#            "alpha_val": (self.alpha_val),
#        }
#        return {**base_config, **config}
#
#    def call(self, input):
#        """Call the layer
#
#        Args:
#            input (_type_): input to layer
#
#        Returns:
#            _type_: Output of layer
#        """
#        input_shape = input.shape
#        shape_ = (-1, input_shape[1], self.n_head, self.d_model // self.n_head)
#        perm_ = (0, 2, 1, 3)
#
#        q = self.qD(input)
#        q = tf.reshape(q, shape=shape_)
#        q = tf.transpose(q, perm=perm_)
#
#        k = self.kD(input)
#        k = tf.reshape(k, shape=shape_)
#        k = tf.transpose(k, perm=perm_)
#
#        v = self.vD(input)
#        v = tf.reshape(v, shape=shape_)
#        v = tf.transpose(v, perm=perm_)
#
#        a = tf.matmul(q, k, transpose_b=True)
#        a = tf.nn.softmax(a / q.shape[3] ** 0.5, axis=3)
#
#        out = tf.matmul(a, v)
#        out = tf.transpose(out, perm=perm_)
#        out = tf.reshape(out, shape=(-1, input_shape[1], self.d_model))
#        out = self.outD(out)
#
#        return out
#
#    # define all prunable weights
#    def get_prunable_weights(self):
#        return (
#            self.qD._trainable_weights
#            + self.kD._trainable_weights
#            + self.vD._trainable_weights
#            + self.outD._trainable_weights
#        )
#
#
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Attention pooling for PyTorch.
    Accepts input of shape (B, N, d) and returns (B, d).
    Uses nn.LazyLinear so no input dim is required at construction.
    """

    def __init__(self, bits=None, bits_int=None, alpha_val=None, **kwargs):
        super().__init__()
        # use built-in lazy linear to avoid manual lazy build
        self._linear = nn.LazyLinear(1, bias=False)
        # keep params for compatibility with previous signature
        self.bits = bits
        self.bits_int = bits_int
        self.alpha_val = alpha_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d)
        a = self._linear(x).squeeze(-1)  # (B, N)
        a = F.softmax(a, dim=1)  # (B, N)
        out = torch.bmm(a.unsqueeze(1), x)  # (B, 1, d)
        return out.squeeze(1)  # (B, d)

    def get_prunable_weights(self):
        # mimic TF interface: return list of parameters that could be pruned
        return (
            [self._linear.weight]
            if hasattr(self._linear, "weight") and self._linear.weight is not None
            else []
        )


class MeanPool(nn.Module):
    """Global average pooling along dim=1 for inputs (B, N, d) -> (B, d).
    Implemented using the built-in AdaptiveAvgPool1d.
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d) -> transpose to (B, d, N) for 1d pooling
        y = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, d)
        return y


class MaxPool(nn.Module):
    """Global max pooling along dim=1 for inputs (B, N, d) -> (B, d).
    Implemented using the built-in AdaptiveMaxPool1d.
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, d)
        return y


def choose_aggregator(
    choice: str, name: str = None, bits=9, bits_int=2, alpha_val=1, **common_args
) -> nn.Module:
    """Choose the aggregator PyTorch module based on an input string.
    Returns a nn.Module that maps (B, N, d) -> (B, d).
    """
    if choice not in ["mean", "max", "attention"]:
        raise ValueError(
            "Given aggregation string is not implemented in choose_aggregator(). "
            "Available: 'mean', 'max', 'attention'."
        )
    if choice == "mean":
        return MeanPool(name=name)
    elif choice == "max":
        return MaxPool(name=name)
    elif choice == "attention":
        return AttentionPooling(
            bits=bits, bits_int=bits_int, alpha_val=alpha_val, **common_args
        )


def from_cfg(config: dict, recreate: bool = True) -> JetTagModel:
    """Create a model directly from a yaml input file

    Args:
        config (dict): Configuration dictionary
        recreate (bool, optional): Rewrite the output directory?. Defaults to True.

    Returns:
        JetTagModel: The model
    """
    folder = config.get("output", "output")
    # Create a model based on what is specified in the yaml 'model' field
    # Model must be registered for this to function
    model = JetModelFactory.create_JetTagModel(
        config["model"], config=config, folder=folder
    )
    if recreate:
        # Remove output dir if exists
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Re-created existing directory: {folder}.")
        # Create dir to save results
        os.makedirs(folder, exist_ok=True)
        # Copy yaml into folder using a cross-platform method
        try:
            yaml.safe_dump(config, open(os.path.join(folder, "config.yaml"), "w"))
        except Exception:
            # best-effort: write a friendly message but don't fail the factory
            print(f"Warning: could not write config to {folder}")
    return model
