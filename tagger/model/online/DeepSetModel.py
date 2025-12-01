"""PyTorch DeepSet model child class of JetTagModel."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from tagger.model.JetTagModel import JetModelFactory, JetTagModel
import tqdm
from tagger.model.torch_utils import calculate_accuracy


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

        self.jet_model = DeepSetModel.DeepSetNet(
            n_features, conv_channels, classifier_layers, aggregator, n_classes
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.jet_model.to(self.device)
        print(self.jet_model)
        print(
            "# Model parameters:", sum(p.numel() for p in self.jet_model.parameters())
        )

    def _prune_model(self):
        """Apply simple global L1 unstructured pruning according to config."""
        try:
            amount = float(self.training_config.get("final_sparsity", 0.0))
        except Exception:
            amount = 0.0
        if amount <= 0.0:
            return

        parameters_to_prune = []
        for name, module in self.jet_model.named_modules():
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
        **kwargs,
    ) -> dict[str, list]:
        """Train the model using the provided DataLoader.
        Args:
            train_loader: DataLoader for training data.
            validation_loader: DataLoader for validation data.
            device: Device to run training on.
            logger: Optional logger for training metrics.
            extras: Additional data for training (not used here).
            resume_training: Whether to resume training from a checkpoint (not used here).
            **kwargs: Additional keyword arguments.
        Returns:
            history: Dictionary containing training history.
        """

        # Move model to device
        self.device = device
        self.jet_model.to(self.device)
        epochs = int(self.training_config.get("epochs", 30))
        patience = int(self.training_config.get("EarlyStopping_patience", 5))
        best_val_loss = float("inf")
        epochs_no_improve = 0

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        start_epoch = 1
        best_model = self.jet_model.state_dict()
        for epoch in range(start_epoch, epochs + 1):
            self.jet_model.train()
            total_loss = 0.0
            train_acc = 0.0
            n_samples = 0
            cur_lr = self.optimizer.param_groups[0]["lr"]
            for batch_idx, batch in tqdm.tqdm(
                enumerate(iterable=train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}/{epochs}",
                ncols=80,
                leave=False,
            ):
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
                outputs = self.jet_model(xb)
                loss = self.loss_fn(outputs, y_true)
                loss.backward()
                self.optimizer.step()

                batch_size = xb.size(0)
                total_loss += loss.item() * batch_size
                n_samples += batch_size
                train_acc += calculate_accuracy(outputs, y_true) * batch_size

            train_loss = total_loss / n_samples if n_samples > 0 else total_loss
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc / n_samples if n_samples > 0 else 0)

            val_loss = None
            if validation_loader is not None:
                self.jet_model.eval()
                total_val = 0.0
                n_val = 0
                val_acc = 0.0
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
                        outputs = self.jet_model(xb)
                        loss = self.loss_fn(outputs, y_true)
                        bs = xb.size(0)
                        total_val += loss.item() * bs
                        n_val += bs
                        val_acc += calculate_accuracy(outputs, y_true) * bs
                val_loss = total_val / n_val if n_val > 0 else total_val
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc / n_val if n_val > 0 else 0)
                # Scheduler step
                if self.lr_scheduler is not None:
                    # ReduceLROnPlateau requires metric; others accept step
                    try:
                        self.lr_scheduler.step(val_loss)
                    except TypeError:
                        self.lr_scheduler.step()
                    if self.optimizer.param_groups[0]["lr"] != cur_lr:
                        cur_lr = self.optimizer.param_groups[0]["lr"]
                        print(f"Learning rate adjusted to {cur_lr:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = self.jet_model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if val_loss is not None:
                    print(
                        f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} | train_acc: {history['train_acc'][-1]:.4f} | val_loss: {val_loss:.4f} | val_acc: {history['val_acc'][-1]:.4f}"
                    )
                    if logger is not None:
                        logger.add_scalar("train/loss", train_loss, epoch)
                        logger.add_scalar(
                            "train/accuracy", history["train_acc"][-1], epoch
                        )
                        logger.add_scalar("val/loss", val_loss, epoch)
                        logger.add_scalar("val/accuracy", history["val_acc"][-1], epoch)

                else:
                    print(
                        f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} | train_acc: {history['train_acc'][-1]:.4f}"
                    )

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                # Restore best model
                self.jet_model.load_state_dict(best_model)
                break

        self.history = history
        return history

    def evaluate(
        self, test_loader, device=torch.device("cpu"), eval_metrics=None
    ) -> dict[str, float]:
        """Evaluate the model on the test data loader.

        Args:
            test_loader: DataLoader for test data.
            device: Device to run evaluation on.
            eval_metrics: List of metrics to compute. #TODO: expand beyond accuracy and loss.

        Returns:
            Dictionary of computed metrics.
        """
        self.jet_model.eval()
        self.jet_model.to(device)
        total_loss = 0.0
        n_samples = 0
        correct = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    xb, yb = batch[0], batch[1]
                elif isinstance(batch, dict):
                    xb, yb = batch["inputs"], batch["targets"]
                else:
                    raise ValueError("Unsupported batch format from test_loader")
                xb = xb.to(device)
                yb = yb.to(device)
                if yb.dim() > 1 and yb.shape[-1] > 1:
                    y_true = torch.argmax(yb, dim=-1)
                else:
                    y_true = yb.squeeze().long()
                outputs = self.jet_model(xb)
                loss = self.loss_fn(outputs, y_true)
                bs = xb.size(0)
                total_loss += loss.item() * bs
                n_samples += bs
                correct += (outputs.argmax(dim=1) == y_true).sum().item()

        avg_loss = total_loss / n_samples if n_samples > 0 else total_loss
        metrics = {"loss": avg_loss}

        if eval_metrics and "accuracy" in eval_metrics:
            accuracy = correct / n_samples if n_samples > 0 else 0.0
            metrics["accuracy"] = accuracy

        return metrics

    # Decorated with save decorator for added functionality
    # @JetTagModel.save_decorator
    def save(self, path):
        """Save the model to the specified path."""
        torch.save(self.jet_model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path, device=torch.device("cpu")):
        """Load the model from the specified path."""
        self.build_model()
        self.jet_model.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Model loaded from {path}")

    def hls4ml_convert(self, firmware_dir: str, build: bool = False):
        raise NotImplementedError(
            "HLS4ML conversion not supported for PyTorch DeepSetModel."
        )
