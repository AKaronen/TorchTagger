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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
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

    def hls4ml_convert(self, firmware_dir: str, build: bool = False):
        raise NotImplementedError(
            "HLS4ML conversion not supported for PyTorch DeepSetModel."
        )

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
