"""PyTorch DeepSet model child class of JetTagModel."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from tagger.model.JetTagModel import JetModelFactory, JetTagModel
import tqdm
from tagger.model.torch_utils import (
    calculate_accuracy,
    per_class_accuracy,
    compute_confusion_matrix,
    plot_confusion_matrix,
)


# Register the model in the factory with the string name corresponding to what is in the yaml config
@JetModelFactory.register("DeepSetModel")
class DeepSetModel(JetTagModel):
    """DeepSetModel implemented with PyTorch"""

    class DeepSetNet(nn.Module):
        def __init__(
            self,
            input_features: int,
            conv_channels: list,
            classifier_layers: list,
            aggregator: str,
            num_classes: int,
        ):
            super().__init__()
            # Conv1d expects (batch, channels, seq_len). We'll treat features as channels.
            self.bn = nn.BatchNorm1d(input_features)
            self.conv_layers = nn.ModuleList()
            in_ch = input_features
            for i, out_ch in enumerate(conv_channels):
                # kernel_size=1 conv to emulate Dense applied per particle
                self.conv_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=1))
                in_ch = out_ch

            self.aggregator_type = (
                aggregator.lower() if isinstance(aggregator, str) else "mean"
            )

            # Attention aggregator params
            try:
                from tagger.model.common import choose_aggregator
            except ImportError as e:
                raise ImportError(
                    "Could not import choose_aggregator from tagger.model.common"
                ) from e
            self.aggregator = choose_aggregator(self.aggregator_type, name="pool")
            # classifier MLP
            mlp_layers = []
            in_feat = in_ch
            for i, out_feat in enumerate(classifier_layers):
                mlp_layers.append(nn.Linear(in_feat, out_feat))
                mlp_layers.append(nn.ReLU())
                in_feat = out_feat
            mlp_layers.append(nn.Linear(in_feat, num_classes))
            self.classifier = nn.Sequential(*mlp_layers)

        def forward(self, x):
            # x: (batch, seq_len, features) -> convert to (batch, features, seq_len) for Conv1d and BatchNorm1d
            if x.dim() == 3:
                x = x.permute(0, 2, 1)
            x = self.bn(x)
            for conv in self.conv_layers:
                x = conv(x)
                x = F.relu(x)

            pooled = self.aggregator(
                x.permute(0, 2, 1)
            )  # back to (batch, seq_len, channels) for aggregator

            # classifier expects (batch, features)
            out = self.classifier(pooled)
            return out

    def build_model(self, model_cfg=None):
        """Build PyTorch model from `self.model_config`.

        Expects `model_config` to contain either explicit `n_features` and
        `n_classes` entries, or else the model can be constructed later by
        `fit` when data shapes are known.
        """
        if model_cfg is not None:
            self.model_config = model_cfg

        conv_channels = self.model_config.get("conv1d_layers", [32, 64])
        classifier_layers = self.model_config.get("classification_layers", [64])
        aggregator = self.model_config.get("aggregator", "mean")

        n_features = self.model_config.get("n_features", None)
        n_classes = self.model_config.get("n_classes", None)

        self.model = DeepSetModel.DeepSetNet(
            n_features, conv_channels, classifier_layers, aggregator, n_classes
        )

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(_device)
        print(self.summary())

    def _prune_model(self):
        """Apply simple global L1 unstructured pruning according to config."""
        try:
            amount = float(self.training_config.get("final_sparsity", 0.0))
        except Exception:
            amount = 0.0
        if amount <= 0.0:
            return

        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                parameters_to_prune.append((module, "weight"))

        if parameters_to_prune:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )

    def compile_model(self, num_samples: int = 0):
        """Backward-compatible compile hook. Use `configure_optimizer` from
        `JetTagModel` to set optimizer and scheduler when running via the
        training pipeline.
        """
        super().compile_model()
        # Apply pruning if requested
        self._prune_model()

    # Decorated with save decorator for added functionality
    # @JetTagModel.save_decorator
    def save(self, path):
        """Save the model to the specified path."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path, device=torch.device("cpu")):
        """Load the model from the specified path."""
        self.build_model()
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Model loaded from {path}")

    def _build_keras_model(self, keras, QConv1D, QDense):
        """Build HGQ2 Keras 3 equivalent of this DeepSetModel for hls4ml export."""
        import numpy as np

        n_features = self.model_config.get("n_features")
        conv_channels = self.model_config.get("conv1d_layers", [32, 64])
        classifier_layers = self.model_config.get("classification_layers", [64])
        n_classes = self.model_config.get("n_classes")
        n_particles = self.model_config.get(
            "n_particles", self.data_config.get("n_particles", 32)
        )

        inputs = keras.Input(shape=(n_particles, n_features))
        x = inputs
        for out_ch in conv_channels:
            x = QConv1D(out_ch, 1, activation="relu", padding="same")(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        for out_feat in classifier_layers:
            x = QDense(out_feat, activation="relu")(x)
        outputs = QDense(n_classes)(x)
        keras_model = keras.Model(inputs, outputs)

        # Run a dummy forward pass to initialise all HGQ2 quantizer state variables
        dummy = np.zeros((1, n_particles, n_features), dtype="float32")
        keras_model(dummy)
        return keras_model

    def _transfer_weights_to_keras(self, keras_model):
        """Transfer trained PyTorch weights to the Keras equivalent model.

        Conv1d: PyTorch (out, in, 1) -> Keras (1, in, out)
        Linear: PyTorch (out, in)    -> Keras (in, out)
        Bias:   same shape for both

        Uses direct .assign() because HGQ2 layers carry extra quantizer
        state variables that make set_weights() incompatible.
        """
        pt_conv_layers = list(self.model.conv_layers)
        pt_linears = [m for m in self.model.classifier if isinstance(m, nn.Linear)]

        keras_conv = [l for l in keras_model.layers if "conv1d" in l.name.lower()]
        keras_dense = [l for l in keras_model.layers if "dense" in l.name.lower()]

        for pt_layer, k_layer in zip(pt_conv_layers, keras_conv):
            w = pt_layer.weight.detach().cpu().numpy()  # (out, in, 1)
            b = pt_layer.bias.detach().cpu().numpy()
            k_layer.kernel.assign(w.transpose(2, 1, 0))  # → (1, in, out)
            k_layer.bias.assign(b)

        for pt_layer, k_layer in zip(pt_linears, keras_dense):
            w = pt_layer.weight.detach().cpu().numpy()  # (out, in)
            b = pt_layer.bias.detach().cpu().numpy()
            k_layer.kernel.assign(w.T)  # → (in, out)
            k_layer.bias.assign(b)

    def hls4ml_convert(self, firmware_dir: str, build: bool = False, **kwargs):
        """Convert trained DeepSetModel to HLS firmware using HGQ2 + hls4ml.

        Requires: hgq2, hls4ml, keras with KERAS_BACKEND=torch.
        Only 'mean' aggregator is supported for HLS export.
        """
        import os
        os.environ.setdefault("KERAS_BACKEND", "torch")
        try:
            import keras
            from hgq.layers import QConv1D, QDense
            import hls4ml
        except ImportError as exc:
            raise ImportError(
                "hls4ml_convert requires hgq2, hls4ml, and keras. "
                "Install with: pip install hgq2 hls4ml"
            ) from exc

        aggregator = self.model_config.get("aggregator", "mean")
        if aggregator != "mean":
            raise ValueError(
                f"hls4ml_convert only supports 'mean' aggregator, got '{aggregator}'"
            )

        keras_model = self._build_keras_model(keras, QConv1D, QDense)
        if self.model is not None:
            self._transfer_weights_to_keras(keras_model)

        precision = self.quantization_config.get("precision", "ap_fixed<16,6>")
        reuse = int(self.quantization_config.get("reuse_factor", 1))
        hls_config = hls4ml.utils.config_from_keras_model(
            keras_model, default_precision=precision
        )
        hls_config["Model"]["ReuseFactor"] = reuse

        backend = self.hls4ml_config.get("backend", "Vivado")
        hls_model = hls4ml.converters.convert_from_keras_model(
            keras_model,
            hls_config=hls_config,
            output_dir=firmware_dir,
            backend=backend,
        )
        hls_model.write()
        if build:
            hls_model.build(csim=False, synth=True)
        return hls_model

    def on_epoch_end(self, epoch, global_step=None, logs=None):
        """Callback at the end of each epoch."""
        if self.logger is not None:
            self.logger.add_scalar(
                "model/learning_rate",
                self.optimizer.param_groups[0]["lr"],
                global_step,
            )
            if "confusion_matrix" in self.metrics:
                cm = self.metrics.metric_state["val_confusion_matrix"]
                self.logger.add_figure(
                    "val/confusion_matrix",
                    plot_confusion_matrix(
                        cm.cpu(),
                        class_names=self.class_labels,  # assuming classes are 0..N-1
                        normalize=True,
                    ),
                    global_step,
                )
        super().on_epoch_end(epoch, global_step, logs)
        return
