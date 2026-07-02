"""End-to-end smoke test: train DeepSetModel 1 epoch, then convert to hls4ml."""
import importlib
import os
import numpy as np
import pytest


def test_deepset_hls4ml(tmp_path, monkeypatch):
    import tagger.data.datamodule as dm_mod
    import tagger.model.online.DeepSetModel  # register in factory

    N, n_particles, n_features, n_classes = 16, 20, 3, 2
    rng = np.random.default_rng(0)
    X = rng.uniform(0.01, 1.0, (N, n_particles, n_features)).astype(np.float32)
    labels = np.zeros((N, n_classes), dtype=np.float32)
    labels[::2, 0] = 1.0
    labels[1::2, 1] = 1.0
    split = int(0.75 * N)
    X_train, y_train = X[:split], labels[:split]
    X_val, y_val = X[split:], labels[split:]
    input_vars = ["pt", "etarel", "phirel"]

    def fake_load_data(data_config, mode="train"):
        return X_train, y_train, X_val, y_val, ["cls0", "cls1"], input_vars, None

    monkeypatch.setattr(dm_mod, "_load_data", fake_load_data)

    out_dir = str(tmp_path / "out")
    cfg = {
        "model": "DeepSetModel",
        "output": out_dir,
        "run_config": {"verbose": 0, "logger": False, "plotting": False},
        "model_config": {
            "n_features": n_features,
            "n_classes": n_classes,
            "n_particles": n_particles,
            "conv1d_layers": [16, 32],
            "classification_layers": [16],
            "aggregator": "mean",
        },
        "quantization_config": {"precision": "ap_fixed<16,6>", "reuse_factor": 1},
        "training_config": {
            "epochs": 1,
            "optimizer": "adam",
            "optimizer_params": {"lr": 1e-3},
            "loss": "cross_entropy",
            "metrics": [{"accuracy": None}],
        },
        "data_config": {"data_path": str(tmp_path), "batch_size": 4},
        "hls4ml_config": {},
    }

    train_mod = importlib.import_module("tagger.train.train_torch")
    model = train_mod.train(cfg)

    assert os.path.exists(os.path.join(out_dir, "model.pt")), "Checkpoint missing"

    hls_out = str(tmp_path / "hls_out")
    hls_model = model.hls4ml_convert(firmware_dir=hls_out, build=False)
    assert hls_model is not None, "hls4ml conversion returned None"

    # write() emits HLS source; check firmware directory and a generated file
    hls_src = os.path.join(hls_out, "firmware")
    assert os.path.isdir(hls_src), f"HLS firmware directory not found at {hls_src}"
    assert any(f.endswith(".cpp") for f in os.listdir(hls_src)), "No .cpp file in firmware/"
