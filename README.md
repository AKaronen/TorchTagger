# TorchTagger

Jet-flavour tagging framework for CMS L1T, built on **PyTorch Lightning**. Supports three model architectures (DeepSet, LorentzNet, ParticleTransformer), a unified YAML-based config system, class-balanced data loading, and FPGA firmware export via HGQ2 + hls4ml.

---

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models](#models)
- [Configuration Reference](#configuration-reference)
- [Training](#training)
- [Evaluation](#evaluation)
- [FPGA Export (hls4ml)](#fpga-export-hls4ml)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)

---

## Installation

```bash
conda env create -f environment.yml
conda activate tagger
```

The environment installs:
- PyTorch + PyTorch Lightning + torchmetrics + TensorBoard
- Data tools: uproot4, numpy, pandas, pyarrow
- FPGA export: hgq2, hls4ml (CMS L1T fork)

> **GPU:** Lightning auto-detects CUDA. No extra steps needed.

---

## Quick Start

```bash
# Train DeepSetModel for 1000 epochs on hls4ml jet data
python tagger/train/train_torch.py train -c tagger/model/configs/DeepSet/deepset32.yaml

# Evaluate a saved checkpoint
python tagger/train/train_torch.py test -c tagger/model/configs/DeepSet/deepset32.yaml
```

Both commands expect `data_config.data_path` in the YAML to point to a directory
containing the dataset. See [Configuration Reference](#configuration-reference) for
how to override values.

---

## Models

Three models are registered in `JetModelFactory`:

| Key | File | Description |
|-----|------|-------------|
| `DeepSetModel` | `tagger/model/online/DeepSetModel.py` | Permutation-invariant DeepSet with Conv1d per-particle layers and configurable aggregator. Supports hls4ml export. |
| `LorentzNet` | `tagger/model/offline/LorentzNet.py` | Lorentz-equivariant GNN using Lorentz Graph Equivariant Blocks (LGEB). Processes 4-momenta with Minkowski metric. |
| `ParticleTransformer` | `tagger/model/offline/ParticleTransformer.py` | Transformer with per-particle embeddings and pairwise interaction terms. Supports loading CMS-pretrained checkpoints. |

### Selecting a model

Set the top-level `model:` key in your YAML config to the factory key above.

---

## Configuration Reference

All configs are YAML files. Pass one with `-c path/to/config.yaml`. CLI arguments
can override any key: `--training_config.epochs 50`.

### Top-level keys

```yaml
model: DeepSetModel           # factory key (required)
output: ./output/run_name     # directory for checkpoints and logs

run_config:
  verbose: True               # print training progress
  plotting: True              # save metric plots after training
  logger: True                # enable TensorBoard logger
  debug: False

model_config: {}              # model-specific (see per-model section below)
training_config: {}           # optimizer, scheduler, callbacks, metrics
data_config: {}               # data path, batch size, splits
quantization_config: {}       # HLS precision (DeepSetModel export only)
hls4ml_config: {}             # HLS backend settings (DeepSetModel export only)
```

### `training_config`

```yaml
training_config:
  epochs: 100
  optimizer: adamw             # adam | adamw | sgd
  optimizer_params:
    lr: 1e-4                   # learning rate
    weight_decay: 0.01         # weight decay (adamw/sgd)
    momentum: 0.9              # momentum (sgd only)
  loss: cross_entropy          # cross_entropy | mse
  scheduler: cosine            # cosine | reduce_on_plateau | cosine_with_warmup
  scheduler_params:
    eta_min: 1e-6
    T_max: 100                 # cosine: period in epochs
    # reduce_on_plateau: mode, factor, patience, min_lr
    # cosine_with_warmup: warmup_steps, eta_min
  callbacks:
    EarlyStopping:
      monitor: val_loss
      patience: 15
    ModelCheckpoint:
      monitor: val_loss
      save_best_only: true
  metrics:
    - accuracy:
    - auroc:
    - confusion_matrix:
```

### `data_config`

```yaml
data_config:
  data_path: ./data/train      # directory with .npy/.npz files (required)
  batch_size: 128              # (required)
  validation_split: 0.2        # fraction held out for validation
  np_data: true                # true = numpy format; false = ROOT format
  n_particles: 32              # max constituents per jet
  fields: [pt, etarel, phirel] # input feature names
  target_labels: [g, q, w, z, t]  # class labels
  weighted_sampling: true      # WeightedRandomSampler for class balance
  class_weight_epsilon: 1e-6   # epsilon in weight computation
  percentage: 100              # % of available data to load (for quick tests)
```

### `model_config` — per model

**DeepSetModel**
```yaml
model_config:
  n_features: 3                # input features per particle (must match fields)
  n_classes: 5                 # number of output classes
  n_particles: 32              # particles per jet (needed for hls4ml export)
  conv1d_layers: [32, 32, 32]  # channels in per-particle Conv1d stack
  classification_layers: [32]  # MLP hidden layer sizes
  aggregator: mean             # mean | max | attention
```

**LorentzNet**
```yaml
model_config:
  n_scalar: 12     # scalar features per particle (beyond 4-momentum)
  n_hidden: 64     # hidden dimension for LGEB layers
  n_layers: 5      # number of LGEB layers
  c_weight: 5e-3   # weight for coordinate update term
  dropout: 0.3
  n_classes: 5
```

**ParticleTransformer**
```yaml
model_config:
  input_dim: 4          # Lorentz 4-vector dimension (px, py, pz, e)
  pf_input_dim: 7       # particle-flow features (depends on 'features' mode)
  n_classes: 5
  features: kin         # kin (7 f) | kinpid (13 f) | full (17 f)
  embed_dims: [128, 512, 128]
  pair_embed_dims: [64, 64, 64]
  num_heads: 8
  num_layers: 8
  num_cls_layers: 2
  fc_params: [[128, 0.1], [64, 0.1]]   # MLP head: [out_dim, dropout]
  from_pretrained: true
  ckpt_path: ./pre-trained/ParticleTransformer/ParT_kin.pt
```

### `quantization_config` (DeepSetModel hls4ml export only)

```yaml
quantization_config:
  precision: ap_fixed<16,6>   # HLS fixed-point precision for all layers
  reuse_factor: 1             # hls4ml ReuseFactor (higher = less area, more latency)
```

### `hls4ml_config` (DeepSetModel hls4ml export only)

```yaml
hls4ml_config:
  backend: Vivado             # Vivado | VivadoAccelerator | Catapult
```

---

## Training

### From CLI

```bash
python tagger/train/train_torch.py train -c path/to/config.yaml
```

Outputs written to `output/` (or the `output:` key in the YAML):
```
output/
├── model.pt                        # final state dict
├── checkpoints/
│   └── best-model.ckpt             # best checkpoint (monitored metric)
└── lightning_logs/                 # TensorBoard event files
```

### From Python

```python
import yaml
from tagger.train.train_torch import train

with open("tagger/model/configs/DeepSet/deepset32.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["data_config"]["data_path"] = "/path/to/data"
cfg["output"] = "/path/to/output"

model = train(cfg)
```

`train()` returns the trained `JetTagModel` instance.

---

## Evaluation

```bash
python tagger/train/train_torch.py test -c path/to/config.yaml
```

Or from Python:

```python
from tagger.train.train_torch import test
results = test(cfg)
```

The test runner loads the best checkpoint from `output/checkpoints/best-model.ckpt`
and evaluates on the test split.

---

## FPGA Export (hls4ml)

Only **DeepSetModel** with `aggregator: mean` supports FPGA export. The pipeline
uses HGQ2 (quantization-aware Keras 3 layers) and hls4ml to generate Vivado HLS C++.

### Requirements

```bash
pip install hgq2 hls4ml
```

> `KERAS_BACKEND=torch` is set automatically by `hls4ml_convert`.

### Usage

```python
import yaml
from tagger.train.train_torch import train

with open("tagger/model/configs/DeepSet/deepset32.yaml") as f:
    cfg = yaml.safe_load(f)

# Add quantization config
cfg["quantization_config"] = {"precision": "ap_fixed<16,6>", "reuse_factor": 1}
cfg["model_config"]["n_particles"] = 32   # must be set for fixed HLS input shape

model = train(cfg)

# Export — writes firmware/ to the given directory
hls_model = model.hls4ml_convert(
    firmware_dir="./hls4ml_output",
    build=False,   # set True to run Vivado synthesis
)
```

The generated `firmware/` directory contains:
```
firmware/
├── myproject.cpp / myproject.h   # top-level HLS project
├── defines.h
├── nnet_utils/                   # hls4ml layer templates
└── weights/                      # trained weight headers
```

### Config reference for export

```yaml
model_config:
  n_particles: 32          # fixed input length (required for HLS)
  aggregator: mean         # only 'mean' supported

quantization_config:
  precision: ap_fixed<16,6>
  reuse_factor: 1

hls4ml_config:
  backend: Vivado
```

---

## Running Tests

```bash
cd TorchTagger
python -m pytest tests/ -v
```

| Test | What it covers |
|------|---------------|
| `test_train_torch_smoke.py` | 1-epoch DeepSet training end-to-end |
| `test_particle_transformer_smoke.py` | 1-epoch ParticleTransformer training |
| `test_lorentz_torch_forward.py` | LorentzNet forward pass |
| `test_deepset_hls4ml.py` | Train DeepSet then export to HLS firmware |

---

## Project Structure

```
TorchTagger/
├── tagger/
│   ├── train/
│   │   ├── train_torch.py      # main entry point: train(), test()
│   │   └── cli.py              # argument parser
│   ├── model/
│   │   ├── JetTagModel.py      # pl.LightningModule base class + JetModelFactory
│   │   ├── common.py           # aggregators, from_cfg() factory helper
│   │   ├── online/
│   │   │   └── DeepSetModel.py # DeepSet + hls4ml_convert()
│   │   └── offline/
│   │       ├── LorentzNet.py
│   │       └── ParticleTransformer.py
│   ├── data/
│   │   ├── datamodule.py       # TorchTaggerDataModule (pl.LightningDataModule)
│   │   ├── datasets.py         # ConstituentsDataset
│   │   ├── tools.py            # load_data(), load_np_data()
│   │   └── parsers/            # ROOT and HuggingFace parsers
│   └── plot/                   # plotting utilities
├── tests/                      # pytest test suite
├── environment.yml
├── CHANGELOG.md
└── README.md
```

---

## Extending with a new model

1. Create `tagger/model/online/MyModel.py` (or `offline/`).
2. Subclass `JetTagModel` and implement `build_model()`, `save()`, `load()`, `hls4ml_convert()`.
3. Register with the factory:

```python
from tagger.model.JetTagModel import JetModelFactory, JetTagModel

@JetModelFactory.register("MyModel")
class MyModel(JetTagModel):
    def build_model(self, model_cfg=None): ...
    def hls4ml_convert(self, **kwargs): raise NotImplementedError
    def save(self, path): torch.save(self.model.state_dict(), path)
    def load(self, path, device=torch.device("cpu")): ...
```

4. Add a YAML config under `tagger/model/configs/MyModel/`.
5. Import the module in your training script or test to trigger registration.
