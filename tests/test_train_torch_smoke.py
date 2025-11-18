import sys
import os
import importlib
import types
import numpy as np


def test_train_torch_smoke(tmp_path, monkeypatch):
    # Import the training module
    train_mod = importlib.import_module("tagger.train.train_torch")

    # Provide a fake load_np_data implementation used by train_torch
    def fake_load_np_data(X_path, y_path, percentage=100):
        N = 8
        n_nodes = 6
        features = 6
        X = np.random.randn(N, n_nodes, features).astype(np.float32)
        labels = np.zeros((N, 2), dtype=np.float32)
        labels[: N // 2, 0] = 1.0
        labels[N // 2 :, 1] = 1.0
        split = int(0.75 * N)
        data_train = {"inputs": X[:split], "targets": labels[:split]}
        data_test = {"inputs": X[split:], "targets": labels[split:]}
        return data_train, data_test

    monkeypatch.setattr(train_mod, "load_np_data", fake_load_np_data)

    # Ensure the LorentzNet model module is imported so the factory registry
    # contains the 'LorentzNet_torch' entry.
    importlib.import_module("tagger.model.LorentzNet_torch")

    out_dir = tmp_path / "out"
    out_dir_str = str(out_dir)

    # Locate the repo YAML config we added under tagger/model/configs
    import tagger

    yaml_path = os.path.join(
        os.path.dirname(tagger.__file__), "model", "configs", "lorentznet_smoke.yaml"
    )

    # Prepare argv for a short run using the YAML/factory path
    args = [
        "train_torch.py",
        "--data",
        "./data/train",
        "--np-data",
        "--config",
        yaml_path,
        "--batch-size",
        "2",
        "--epochs",
        "1",
        "-o",
        out_dir_str,
    ]
    monkeypatch.setattr(sys, "argv", args)

    # Run main (should not raise)
    train_mod.main()

    # Check checkpoint written
    model_path = os.path.join(out_dir_str, "model.pth")
    assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"
