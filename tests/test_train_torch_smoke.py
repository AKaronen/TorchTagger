import importlib
import numpy as np
import pytest


def test_train_torch_smoke(tmp_path, monkeypatch):
    import tagger.data.datamodule as dm_mod
    import tagger.model.offline.LorentzNet  # register model in factory

    N, n_nodes, features, n_classes = 16, 6, 6, 2
    X = np.random.randn(N, n_nodes, features).astype(np.float32)
    labels = np.zeros((N, n_classes), dtype=np.float32)
    labels[: N // 2, 0] = 1.0
    labels[N // 2 :, 1] = 1.0
    split = int(0.75 * N)
    X_train, y_train = X[:split], labels[:split]
    X_val, y_val = X[split:], labels[split:]
    class_labels = ["class_0", "class_1"]

    def fake_load_data(data_config, mode="train"):
        return X_train, y_train, X_val, y_val, class_labels, None, None

    monkeypatch.setattr(dm_mod, "_load_data", fake_load_data)

    out_dir = str(tmp_path / "out")
    config = {
        "model": "LorentzNet",
        "output": out_dir,
        "run_config": {"verbose": 0},
        "training_config": {"epochs": 1, "log_interval": 10},
        "data_config": {"data_path": str(tmp_path), "batch_size": 4},
        "model_config": {
            "n_scalar": 2,  # 6 total features: 4 for 4-vector + 2 scalars
            "n_hidden": 16,
            "n_layers": 2,
            "c_weight": 1e-3,
            "dropout": 0.0,
            "n_classes": n_classes,
        },
    }

    train_mod = importlib.import_module("tagger.train.train_torch")
    model = train_mod.train(config)

    assert (tmp_path / "out" / "model.pt").exists(), "model.pt not found"
    print("Smoke test passed")
