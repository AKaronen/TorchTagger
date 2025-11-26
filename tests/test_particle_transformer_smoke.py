import sys
import os
import importlib
import numpy as np


def test_particle_transformer_smoke(tmp_path, monkeypatch):
    # Import the training module
    train_mod = importlib.import_module("tagger.train.train_torch")

    # Provide a fake load_np_data implementation used by train_torch
    def fake_load_np_data(
        path=None,
        val_ratio=0.2,
        percentage=100,
        n_particles=None,
        fields=None,
        shuffle_constits=False,
    ):
        N = 16
        n_nodes = 20
        features = 16
        X = np.random.randn(N, n_nodes, features).astype(np.float32)
        labels = np.zeros((N, 2), dtype=np.float32)
        labels[: N // 2, 0] = 1.0
        labels[N // 2 :, 1] = 1.0
        split = int((1 - val_ratio) * N)
        data_train = {"inputs": X[:split], "targets": labels[:split]}
        data_test = {"inputs": X[split:], "targets": labels[split:]}
        return data_train, data_test

    # Monkeypatch the loader expected by train_torch
    monkeypatch.setattr(train_mod, "load_np_data", fake_load_np_data)

    # Ensure the ParticleTransformer model module is imported so factory registry contains it
    importlib.import_module("tagger.model.ParticleTransformer")

    out_dir = tmp_path / "out"
    out_dir_str = str(out_dir)

    # Locate the YAML config we added under tagger/model/configs
    import tagger

    yaml_path = os.path.join(
        os.path.dirname(tagger.__file__),
        "model",
        "configs",
        "ParticleTransformer",
        "particle_transformer.yaml",
    )

    # Prepare argv for a short run using the YAML/factory path
    args = [
        "train_torch.py",
        "train",
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
    model_path = os.path.join(out_dir_str, "model.pt")
    assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"
