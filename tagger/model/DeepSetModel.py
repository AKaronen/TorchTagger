"""PyTorch DeepSet model child class

Converted from the original TensorFlow/Keras implementation to PyTorch.
Preserves the public API expected by the surrounding code: `build_model`,
`compile_model`, `fit`, `save`, `load`, and `hls4ml_convert` (stubbed).

Written 21/11/2025 by GitHub Copilot (converted to PyTorch)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from tagger.model.JetTagModel import JetModelFactory, JetTagModel


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
            # x: (batch, seq_len, features) -> convert to (batch, features, seq_len)
            if x.dim() == 3:
                x = x.permute(0, 2, 1)

            for conv in self.conv_layers:
                x = conv(x)
                x = F.relu(x)

            pooled = self.aggregator(
                x.permute(0, 2, 1)
            )  # back to (batch, seq_len, channels) for aggregator

            # classifier expects (batch, features)
            out = self.classifier(pooled)
            return out

    def build_model(self):
        """Build PyTorch model from `self.model_config`.

        Expects `model_config` to contain either explicit `n_features` and
        `n_classes` entries, or else the model can be constructed later by
        `fit` when data shapes are known.
        """
        conv_channels = self.model_config.get("conv1d_layers", [32, 64])
        classifier_layers = self.model_config.get("classification_layers", [64])
        aggregator = self.model_config.get("aggregator", "mean")

        n_features = self.model_config.get("n_features", None)
        n_classes = self.model_config.get("n_classes", None)

        if n_features is None or n_classes is None:
            # Defer building until fit when data shapes are available
            self.model = None
            print(
                "DeepSetModel: n_features or n_classes not set in model_config; deferring build until fit()."
            )
            return

        self.model = DeepSetModel.DeepSetNet(
            n_features, conv_channels, classifier_layers, aggregator, n_classes
        )
        # Keep compatibility with JetTagModel.configure_optimizer which expects `jet_model`
        self.jet_model = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(self.model)

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

    def fit(
        self,
        train_loader,
        validation_loader=None,
        device=torch.device("cpu"),
        logger=None,
        extras=None,
        resume_training=False,
        **kwargs,
    ):
        """Train method compatible with `train_torch.train`.

        Expects `train_loader` and optional `validation_loader` to be
        `torch.utils.data.DataLoader` instances producing `(X, y)` tuples.
        """
        # Load configs into attributes like JetTagModel.configure_optimizer would

        # Move model to device
        self.device = device
        self.model.to(self.device)

        epochs = int(self.training_config.get("epochs", 10))
        patience = int(self.training_config.get("EarlyStopping_patience", 10))
        best_val_loss = float("inf")
        epochs_no_improve = 0

        history = {"train_loss": [], "val_loss": []}

        start_epoch = 1
        for epoch in range(start_epoch, epochs + 1):
            self.model.train()
            total_loss = 0.0
            n_samples = 0
            for batch in train_loader:
                # Support collated batches in tuple form
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    xb, yb = batch[0], batch[1]
                elif isinstance(batch, dict):
                    xb, yb = batch["inputs"], batch["targets"]
                else:
                    raise ValueError("Unsupported batch format from train_loader")
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                # If yb is one-hot, convert
                if yb.dim() > 1 and yb.shape[-1] > 1:
                    y_true = torch.argmax(yb, dim=-1)
                else:
                    y_true = yb.squeeze().long()

                self.optimizer.zero_grad()
                outputs = self.model(xb)
                loss = self.loss_fn(outputs, y_true)
                loss.backward()
                self.optimizer.step()

                batch_size = xb.size(0)
                total_loss += loss.item() * batch_size
                n_samples += batch_size

            train_loss = total_loss / n_samples if n_samples > 0 else total_loss
            history["train_loss"].append(train_loss)

            val_loss = None
            if validation_loader is not None:
                self.model.eval()
                total_val = 0.0
                n_val = 0
                with torch.no_grad():
                    for batch in validation_loader:
                        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            xb, yb = batch[0], batch[1]
                        elif isinstance(batch, dict):
                            xb, yb = batch["inputs"], batch["targets"]
                        else:
                            raise ValueError(
                                "Unsupported batch format from validation_loader"
                            )
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        if yb.dim() > 1 and yb.shape[-1] > 1:
                            y_true = torch.argmax(yb, dim=-1)
                        else:
                            y_true = yb.squeeze().long()
                        outputs = self.model(xb)
                        loss = self.loss_fn(outputs, y_true)
                        bs = xb.size(0)
                        total_val += loss.item() * bs
                        n_val += bs
                val_loss = total_val / n_val if n_val > 0 else total_val
                history["val_loss"].append(val_loss)
                # Scheduler step
                if self.lr_scheduler is not None:
                    # ReduceLROnPlateau requires metric; others accept step
                    try:
                        self.lr_scheduler.step(val_loss)
                    except TypeError:
                        self.lr_scheduler.step()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            if self.run_config.get("verbose", 1):
                if val_loss is not None:
                    print(
                        f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}"
                    )
                else:
                    print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}")

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.history = history
        return history

    # Decorated with save decorator for added functionality
    # @JetTagModel.save_decorator
    def save(self, out_dir: str = "None"):
        """Save the PyTorch model state_dict and minimal metadata."""
        os.makedirs(os.path.join(out_dir, "model"), exist_ok=True)
        export_path = os.path.join(out_dir, "model", "model.pt")
        torch.save({"model_state_dict": self.model.state_dict()}, export_path)
        print(f"Model saved to {export_path}")

    # @JetTagModel.load_decorator
    def load(self, out_dir: str = "None"):
        """Load the PyTorch model state_dict. Assumes `build_model` was called"""
        load_path = os.path.join(out_dir, "model", "model.pt")
        checkpoint = torch.load(load_path, map_location="cpu")
        if not hasattr(self, "model"):
            raise RuntimeError(
                "Model architecture not built. Call build_model before load."
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        print(f"Model loaded from {load_path}")

    def hls4ml_convert(self, firmware_dir: str, build: bool = False):
        """HLS conversion is not implemented for PyTorch models in this repository."""
        raise NotImplementedError(
            "HLS4ML conversion not supported for PyTorch DeepSetModel."
        )
