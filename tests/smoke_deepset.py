import os
import yaml
import numpy as np
import torch
from tagger.train.train_torch import train


def main():
    # Create test data
    data_dir = os.path.join(os.getcwd(), "test_data_deepset")
    os.makedirs(data_dir, exist_ok=True)

    n_samples = 200
    seq_len = 150
    n_features = 3
    n_classes = 3

    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y_idx = np.random.randint(0, n_classes, size=(n_samples,))
    y = np.eye(n_classes)[y_idx].astype(np.float32)
    if not os.path.exists(
        os.path.join(data_dir, "jetConstituent_150.npy")
    ) or not os.path.exists(os.path.join(data_dir, "jetConstituent_150_target.npy")):
        np.save(os.path.join(data_dir, "jetConstituent_150.npy"), X)
        np.save(os.path.join(data_dir, "jetConstituent_150_target.npy"), y)

    # Load config
    cfg_path = os.path.join(
        os.getcwd(), "tagger", "model", "configs", "DeepSet", "deepset.yaml"
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Ensure data_path points to generated data
    cfg["data_config"]["data_path"] = data_dir
    cfg["output"] = os.path.join(os.getcwd(), "output_deepset")

    try:
        from tagger.model.common import fromConfig
    except Exception:
        raise RuntimeError("tagger.model.common.fromConfig not available in PYTHONPATH")
    model = fromConfig(cfg, folder=cfg["output"], recreate=True)
    print(f"Model created: {model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run training (uses train_torch.train internals)
    model = train(model=model, config=cfg, device=device)
    assert model is not None
    print("Training complete.")


if __name__ == "__main__":
    main()
