import os


import torch
from torch.utils.data import DataLoader
from tagger.train.cli import parse_cli
from tagger.data.datasets import ConstituentsDataset


def train(model, config, device=None, **kwargs):
    training_config = config.get("training_config", {})
    data_config = config.get("data_config", {})
    output = config.get("output", "output")
    if data_config.get("np_data", True):  # Default to np_data for now
        try:
            from tagger.data.tools import load_np_data
        except Exception:
            raise RuntimeError(
                "loader tagger.data.tools.load_np_data not available in PYTHONPATH"
            )
        # Heuristic: look for files in the provided folder
        X_path = os.path.join(data_config["data_path"], "jetConstituent_150.npy")
        y_path = os.path.join(data_config["data_path"], "jetConstituent_150_target.npy")
        data_train, data_val = load_np_data(
            X_path,
            y_path,
            test_ratio=data_config["validation_split"],
            percentage=data_config["percentage"],
        )
        X_train = data_train["inputs"]
        y_train = data_train["targets"]
        X_val = data_val["inputs"]
        y_val = data_val["targets"]

        print(
            f"Class distribution in training set: {y_train.sum(axis=0)}, Class distribution in validation set: {y_val.sum(axis=0)}"
        )
    else:
        raise RuntimeError(
            "Only --np-data prototype path is implemented for this runner"
        )

    dataset = ConstituentsDataset(X_train, y_train)
    val_dataset = ConstituentsDataset(X_val, y_val)
    X_sample, y_sample = dataset[0]
    input_shape = X_sample.shape
    output_shape = y_sample.shape
    print(f"Input shape: {input_shape}, output shape: {output_shape}")
    # create DataLoader now that collate is known
    train_loader = DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: (
            model.collate_fn(batch, device) if hasattr(model, "collate_fn") else batch
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: (
            model.collate_fn(batch, device) if hasattr(model, "collate_fn") else batch
        ),
    )
    print(f"Training model on device: {device}")
    print(
        f"Training for {training_config['epochs']} epochs with batch_size={data_config['batch_size']}"
    )

    history = model.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        device=device,
    )
    model.save(os.path.join(output, "model.pth"))
    print(f"Model training complete, model saved to {output}/model.pth")
    if config.get("run_config", {}).get("plotting", False):
        model.plot_training_history(history=history)
    return model


def main():
    # Parse CLI args and load YAML if provided (moved to tagger.train.cli)
    mode, cfg, extras = parse_cli()
    print("Mode:", mode)
    print("Extras:", extras)

    print(f"Configuration for {mode}: {cfg}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "train":
        output = cfg.get("output", "output")
        # Load data
        os.makedirs(output, exist_ok=True)
        # Create model

        try:
            from tagger.model.common import fromConfig
        except Exception:
            raise RuntimeError(
                "tagger.model.common.fromConfig not available in PYTHONPATH"
            )
        model = fromConfig(cfg, output)
        print(f"Model created: {model}")

        train(
            model,
            cfg,
            device=device,
            extras=extras,
        )

    elif mode == "eval":
        pass  # TODO: implement eval mode

    elif mode == "test":
        pass  # TODO: implement test mode
    else:  # Should not reach here due to argparse enforcement
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
