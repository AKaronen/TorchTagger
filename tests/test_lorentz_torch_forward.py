import torch
import numpy as np
from TrainTagger_v2.tagger.model.LorentzNet import LorentzNetModelTorch


def make_complete_graph_indices(n_nodes, device=None):
    rows = []
    cols = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            rows.append(i)
            cols.append(j)
    rows = torch.tensor(rows, dtype=torch.long, device=device)
    cols = torch.tensor(cols, dtype=torch.long, device=device)
    return rows, cols


def test_lorentz_forward_shapes():
    device = torch.device("cpu")
    B = 2
    N = 139
    n_scalar = 8
    n_hidden = 16
    n_class = 3

    scalars = torch.randn(B, N, n_scalar, device=device)
    # x per node (B*N, 4) flattening approach
    x = torch.randn(B, N, 4, device=device)
    rows, cols = make_complete_graph_indices(B * N, device=device)
    # node_mask: (B, N)
    node_mask = torch.ones(B, N, device=device)
    edge_mask = torch.ones(B, N, N, device=device)

    print("Batch shapes:", scalars.shape, x.shape, node_mask.shape, edge_mask.shape, rows.shape, cols.shape)
    model = LorentzNetModelTorch(n_scalar, n_hidden, n_class, n_layers=6)
    out = model(scalars, x, (rows, cols), node_mask, edge_mask, n_nodes=N)
    assert out.shape == (B, n_class)
