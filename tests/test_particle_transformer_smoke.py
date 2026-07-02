import importlib
import numpy as np
import pytest


def test_particle_transformer_smoke(tmp_path, monkeypatch):
    import tagger.data.datamodule as dm_mod
    import tagger.model.offline.ParticleTransformer  # register model in factory

    # PT collate_fn needs specific physics feature names in order
    # "kin" mode uses: log_pt, log_e, log_ptrel, log_erel, deltaR, deta, dphi
    # Lorentz vectors: px, py, pz, e
    # Raw inputs needed: pt, e, pt_rel, deta, dphi, px, py, pz
    input_vars = ["pt", "e", "pt_rel", "deta", "dphi", "px", "py", "pz"]
    n_features = len(input_vars)
    N, n_nodes, n_classes = 16, 20, 2

    rng = np.random.default_rng(42)
    X = rng.uniform(0.01, 1.0, (N, n_nodes, n_features)).astype(np.float32)
    labels = np.zeros((N, n_classes), dtype=np.float32)
    # Interleave classes so both appear in train and val splits
    labels[::2, 0] = 1.0
    labels[1::2, 1] = 1.0
    split = int(0.75 * N)
    X_train, y_train = X[:split], labels[:split]
    X_val, y_val = X[split:], labels[split:]
    class_labels = ["class_0", "class_1"]

    def fake_load_data(data_config, mode="train"):
        return X_train, y_train, X_val, y_val, class_labels, input_vars, None

    monkeypatch.setattr(dm_mod, "_load_data", fake_load_data)

    out_dir = str(tmp_path / "out")

    import tagger
    import os
    import yaml

    yaml_path = os.path.join(
        os.path.dirname(tagger.__file__),
        "model",
        "configs",
        "ParticleTransformer",
        "particle_transformer.yaml",
    )

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    cfg["output"] = out_dir
    cfg["run_config"] = {"verbose": 0, "logger": False, "plotting": False}
    cfg["training_config"]["epochs"] = 1
    cfg["training_config"]["metrics"] = [{"accuracy": None}]  # skip AUROC/CM to avoid state accumulation issues
    cfg["model_config"]["from_pretrained"] = False
    cfg["model_config"]["pf_input_dim"] = 7  # "kin" mode produces 7 features
    cfg["model_config"]["input_dim"] = 4     # lorentz vector (px,py,pz,e)
    cfg["model_config"]["n_classes"] = n_classes
    cfg["model_config"]["features"] = "kin"
    cfg["data_config"] = {
        "data_path": str(tmp_path),
        "batch_size": 4,
    }

    train_mod = importlib.import_module("tagger.train.train_torch")
    train_mod.train(cfg)

    model_path = os.path.join(out_dir, "model.pt")
    assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"
