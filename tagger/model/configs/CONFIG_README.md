Config file guide for TrainTagger_v2
===================================

This document describes the YAML config structure used by the PyTorch training
runner `tagger/train/train_torch.py`. Put model configs in
`TrainTagger_v2/tagger/model/configs/` and pass the path with `--config`.

Top-level keys
--------------

- `model` (string)
  - Name of the model factory to use (example: `LorentzNet`). When `--config`
    is provided the runner will call the project's `fromYaml(...)` factory and
    then `build_model(input_shape, output_shape)` on the returned wrapper.

- `run_config` (map) — optional
  - free-form debugging/runtime flags. Example: `debug: False`.

- `model_config` (map)
  - Model-specific hyperparameters. These keys depend on the model type.
  - Example for LorentzNet:
    - `n_scalar` (int)
    - `n_hidden` (int)
    - `n_layers` (int)
    - `c_weight` (float)
    - `dropout` (float)
    - `n_classes` (int)

- `training_config` (map)
  - Training-level parameters used by the runner. Supported keys:
    - `optimizer` (string) — one of: `adam` (default), `adamw`, `sgd`.
      - `optimizer_params` parameters related to the chosen optimizer e.g.
        - `momentum` (float) — used when `optimizer: sgd` (default 0.0).
        - `lr` or `learning_rate` (float) — learning rate (default 1e-3)
        - `weight_decay` (float) — weight decay for optimizers (default 0.0).
    - `epochs` (int) — number of training epochs (default 3).
    - `loss` (string) loss function to use during training ("cross_entropy" or "mse")
    - `scheduler` (string) the learning rate scheduler to be used
      - `scheduler_params` hyperparameters to be provided for the scheduler

- `data_config`
  - `batch_size` (int) — batch size (default 8).
  - `validation_split` (float) — fraction of training to keep as validation.
  - `percentage` (int) percentage of data loaded (for quick tests)
  - `np_data` (bool) indicate the use of numpy data (required for loading)
  - `data_path` path to training data
- `quantization_config`, `hls4ml_config`, etc.
  - Other model-specific or post-processing configs may be included; the
    runner ignores unknown top-level keys but model factories may read them.

### How runner resolves values
--------------------------

- CLI arguments override YAML values.
- Defaults are used where needed

Minimal example (LorentzNet smoke)

---------------------------------;

```yaml
model: LorentzNet
run_config:
  debug: False
model_config:
  n_hidden: 16
  n_layers: 2
  c_weight: 0.001
  dropout: 0.0
training_config:
  optimizer: adam
  lr: 0.001
  epochs: 3
  batch_size: 8
  weight_method: none
  validation_split: 0.0
```

