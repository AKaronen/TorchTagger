import os
import torch
import torch.nn as nn
from typing import Tuple

from tagger.model.torch_utils import (
    calculate_accuracy,
    unsorted_segment_sum,
    unsorted_segment_mean,
)
from tagger.model.JetTagModel import JetTagModel, JetModelFactory


import numpy as np
from scipy.sparse import coo_matrix
import tqdm


def get_adj_matrix(n_nodes, batch_size, edge_mask):
    rows, cols = [], []
    for batch_idx in range(batch_size):
        nn = batch_idx * n_nodes
        x = coo_matrix(edge_mask[batch_idx])
        rows.append(nn + x.row)
        cols.append(nn + x.col)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    edges = [torch.LongTensor(rows), torch.LongTensor(cols)]
    return edges


class LGEB(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        n_node_attr=0,
        dropout=0.0,
        c_weight=1.0,
        last_layer=False,
    ):
        super().__init__()
        self.c_weight = c_weight
        self.last_layer = last_layer

        # simple MLPs
        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + 2, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.phi_m = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())
        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )
        layer = nn.Linear(n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi_x = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), layer)
        self.last_layer = last_layer
        if last_layer:
            del self.phi_x

    def minkowski_feats(
        self, edges: Tuple[torch.Tensor, torch.Tensor], x: torch.Tensor
    ):
        i, j = edges
        x_diff = x[i] - x[j]
        norms = self.normsq4(x_diff).unsqueeze(1)
        dots = self.dotsq4(x[i], x[j]).unsqueeze(1)
        norms, dots = self.psi(norms), self.psi(dots)
        return norms, dots, x_diff

    def normsq4(self, p):
        r"""Minkowski square norm
        `\|p\|^2 = p[0]^2-p[1]^2-p[2]^2-p[3]^2`
        """
        psq = torch.pow(p, 2)
        return 2 * psq[..., 0] - psq.sum(dim=-1)

    def dotsq4(self, p, q):
        r"""Minkowski inner product
        `<p,q> = p[0]q[0]-p[1]q[1]-p[2]q[2]-p[3]q[3]`
        """
        psq = p * q
        return 2 * psq[..., 0] - psq.sum(dim=-1)

    def psi(self, p):
        """`\psi(p) = Sgn(p) \cdot \log(|p| + 1)`"""
        return torch.sign(p) * torch.log(torch.abs(p) + 1)

    def m_model(self, hi, hj, norms, dots):
        out = torch.cat([hi, hj, norms, dots], dim=-1)
        out = self.phi_e(out)
        w = self.phi_m(out)
        return out * w

    def h_model(self, h, edges, m, node_attr):
        i, j = edges
        # m: (E, hidden)
        agg = unsorted_segment_sum(m, i, num_segments=h.size(0))
        agg = torch.cat([h, agg, node_attr], dim=1)
        out = h + self.phi_h(agg)
        return out

    def x_model(self, x, edges, x_diff, m):
        i, j = edges
        trans = x_diff * self.phi_x(m)
        trans = torch.clamp(trans, min=-100.0, max=100.0)
        agg = unsorted_segment_mean(trans, i, num_segments=x.size(0))
        out = x + self.c_weight * agg
        return out

    def forward(self, h, x, edges, node_attr=None):
        i, j = edges
        norms, dots, x_diff = self.minkowski_feats(edges, x)
        m = self.m_model(h[i], h[j], norms, dots)
        if not self.last_layer:
            x = self.x_model(x, edges, x_diff, m)
        h = self.h_model(h, edges, m, node_attr)
        return h, x, m


class LorentzNetModelTorch(nn.Module):
    def __init__(
        self, n_scalar, n_hidden, n_class=2, n_layers=6, c_weight=1e-3, dropout=0.0
    ):
        super(LorentzNetModelTorch, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList(
            [
                LGEB(
                    n_hidden,
                    n_hidden,
                    n_hidden,
                    n_node_attr=n_scalar,
                    dropout=dropout,
                    c_weight=c_weight,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            self.dropout,
            nn.Linear(n_hidden, n_class),
        )

    def forward(self, scalars, x, edges, node_mask, edge_mask):
        # scalars: (B, N, n_scalar)
        b, nn, _ = scalars.size()

        scalars = scalars.view(b * nn, -1)
        x = x.view(b * nn, -1)
        edge_mask = edge_mask.view(b * nn * nn, -1)
        node_mask = node_mask.view(b * nn, 1)

        h = self.embedding(scalars)

        for i in range(self.n_layers):
            h, x, _ = self.LGEBs[i](h, x, edges, node_attr=scalars)
        # reshape back
        h = h * node_mask
        h = h.view(-1, nn, self.n_hidden)
        h = h.mean(dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)


@JetModelFactory.register("LorentzNet")
class LorentzNet(JetTagModel):
    def build_model(self):
        """Build model based on model_config loaded from YAML."""

        n_scalar = self.model_config.get("n_scalar", 4)
        n_class = self.model_config.get("n_classes", 2)
        # read hyperparams from model_config if present
        n_hidden = (
            self.model_config.get("n_hidden", 64)
            if hasattr(self, "model_config")
            else 64
        )
        n_layers = (
            self.model_config.get("n_layers", 3) if hasattr(self, "model_config") else 3
        )
        c_weight = (
            self.model_config.get("c_weight", 1e-3)
            if hasattr(self, "model_config")
            else 1e-3
        )
        dropout = (
            self.model_config.get("dropout", 0.0)
            if hasattr(self, "model_config")
            else 0.0
        )

        self.jet_model = LorentzNetModelTorch(
            n_scalar, n_hidden, n_class, n_layers, c_weight, dropout
        )
        print(self.jet_model)
        print(
            "# of trainable parameters:",
            sum(p.numel() for p in self.jet_model.parameters() if p.requires_grad),
        )

    def fit(
        self, train_loader, device=torch.device("cpu"), resume_training=False, **kwargs
    ):
        """Fit the model using training data provided in kwargs"""
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
        if "epochs" in kwargs:
            epochs = kwargs["epochs"]
        else:
            epochs = self.epochs
        if "validation_loader" in kwargs:
            val_loader = kwargs["validation_loader"]
            self.history = {
                "train_loss": [],
                "val_loss": [],
                "train_acc": [],
                "val_acc": [],
            }
        else:
            val_loader = None
            self.history = {"train_loss": [], "train_acc": []}
        if resume_training and "checkpoint_path" in kwargs:
            checkpoint_path = kwargs["checkpoint_path"]
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.jet_model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed training from epoch {start_epoch}")
        self.jet_model.to(device)
        self.best_val_loss = float("inf")
        self.best_model_state = None
        for epoch in range(start_epoch if resume_training else 1, epochs + 1):
            self.jet_model.train()
            total_loss = 0.0
            train_acc = 0.0
            cur_lr = self.optimizer.param_groups[0]["lr"]
            for batch_idx, (batch, target) in tqdm.tqdm(
                enumerate(iterable=train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}/{epochs}",
                ncols=80,
                leave=False,
            ):
                scalars, x, edges, node_mask, edge_mask = batch
                self.optimizer.zero_grad()
                output = self.jet_model(scalars, x, edges, node_mask, edge_mask)

                for item in (scalars, x, edges[0], edges[1], node_mask, edge_mask):
                    item.cpu()

                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                preds = output.argmax(dim=1)
                train_acc += calculate_accuracy(target, preds)
                if self.log_interval > 0 and (batch_idx + 1) % self.log_interval == 0:
                    if self.logger:
                        self.logger.add_scalar(
                            "Train/Loss",
                            total_loss / (batch_idx + 1),
                            (epoch - 1) * len(train_loader) + batch_idx + 1,
                        )
                        self.logger.add_scalar(
                            "Train/Accuracy",
                            train_acc / (batch_idx + 1),
                            (epoch - 1) * len(train_loader) + batch_idx + 1,
                        )
            self.history["train_loss"].append(total_loss / len(train_loader))
            self.history["train_acc"].append(train_acc / len(train_loader))

            if val_loader is not None:
                self.jet_model.eval()
                val_loss = 0.0
                val_acc = 0.0
                with torch.no_grad():
                    for batch, target in val_loader:
                        scalars, x, edges, node_mask, edge_mask = batch
                        output = self.jet_model(scalars, x, edges, node_mask, edge_mask)
                        loss = self.loss_fn(output, target)
                        val_loss += loss.item()
                        preds = output.argmax(dim=1)
                        val_acc += calculate_accuracy(target, preds)
                avg_val_loss = val_loss / len(val_loader)
                avg_val_acc = val_acc / len(val_loader)
                self.history["val_loss"].append(avg_val_loss)
                self.history["val_acc"].append(avg_val_acc)
                if self.lr_scheduler is not None:
                    # ReduceLROnPlateau requires metric; others accept step
                    try:
                        self.lr_scheduler.step(val_loss)
                    except TypeError:
                        self.lr_scheduler.step()
                    if self.optimizer.param_groups[0]["lr"] != cur_lr:
                        cur_lr = self.optimizer.param_groups[0]["lr"]
                        print(f"Learning rate adjusted to {cur_lr}")
                if self.logger:
                    self.logger.add_scalar(
                        "Val/Loss",
                        avg_val_loss,
                        epoch * len(train_loader),
                    )
                    self.logger.add_scalar(
                        "Val/Accuracy",
                        avg_val_acc,
                        epoch * len(train_loader),
                    )
                    self.logger.add_scalar(
                        "Learning Rate",
                        self.optimizer.param_groups[0]["lr"],
                        epoch * len(train_loader),
                    )
                print(
                    f"Epoch {epoch}/{epochs}, Training Loss: {self.history['train_loss'][-1]:.4f}, Training Accuracy: {self.history['train_acc'][-1]:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_acc:.4f}"
                )

            else:
                if self.training_config.get("scheduler", None) == "reduce_on_plateau":
                    self.lr_scheduler.step(self.history["train_loss"][-1])
                elif hasattr(self, "lr_scheduler"):
                    self.lr_scheduler.step()
                print(
                    f"Epoch {epoch}/{epochs}, Training Loss: {self.history['train_loss'][-1]:.4f}, Training Accuracy: {self.history['train_acc'][-1]:.4f}"
                )
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.best_model_state = self.jet_model.state_dict()
            if epoch % 5 == 0:
                os.makedirs(os.path.join(self.output_directory, "model"), exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.best_model_state
                        if self.best_model_state is not None
                        else self.jet_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    f"{os.path.join(self.output_directory, 'model', 'best_model.pth')}",  # save only latest checkpoint
                )
            torch.cuda.empty_cache()
        return self.history

    def hls4ml_convert(self, **kwargs):
        raise NotImplementedError("HLS4ML conversion not implemented yet")

    def collate_fn(self, batch):
        """Collate function for LorentzNet model
        Args:
            batch (list): List of samples
            device (torch.device): Device to move tensors to
        Returns:
            tuple: Collated batch (scalars, x, edges, node_mask, edge_mask)
        """
        labels = torch.from_numpy(np.array([item[1] for item in batch]))
        p4s = torch.from_numpy(np.array([item[0][:, :4] for item in batch]))
        scalars = torch.from_numpy(np.array([item[0][:, 4:] for item in batch]))
        node_masks = torch.from_numpy(
            np.array([item[0][:, 5] != 0 for item in batch])
        )  # assuming pT=0 means padding
        batch_size, n_nodes, _ = p4s.size()
        atom_mask = node_masks.bool()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edges = get_adj_matrix(n_nodes, batch_size, edge_mask)
        # everything to device
        scalars = scalars.to(self.device)
        p4s = p4s.to(self.device)
        edge_mask = edge_mask.to(self.device)
        node_masks = node_masks.float().to(self.device)
        edges = [item.to(self.device) for item in edges]
        labels = labels.to(self.device)
        return (scalars, p4s, edges, node_masks, edge_mask), labels

    def save(self, path):
        """Save the model to the specified path."""
        torch.save(self.jet_model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path, device=torch.device("cpu")):
        """Load the model from the specified path."""
        self.build_model()
        self.jet_model.load_state_dict(torch.load(path, map_location=device))
        self.jet_model.to(device)
        print(f"Model loaded from {path}")
