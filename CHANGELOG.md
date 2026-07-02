# Changelog

All notable changes to TorchTagger are documented here, organised by migration phase.

---

## [Phase 5] — 2026-07-02 — HGQ2 + hls4ml Spike (DeepSetModel)

### Added
- `DeepSetModel._build_keras_model()` — constructs a Keras 3 equivalent of the trained
  PyTorch DeepSet model using HGQ2-quantized layers (`QConv1D`, `QDense`,
  `GlobalAveragePooling1D`). A dummy forward pass initialises all HGQ2 quantizer
  state variables so weight transfer is possible immediately.
- `DeepSetModel._transfer_weights_to_keras()` — copies trained PyTorch weights into the
  Keras model. Uses `.assign()` directly on `kernel`/`bias` attributes instead of
  `set_weights()` because HGQ2 layers carry extra quantizer state variables
  (beta, ebops, k, i, f, …) that make `set_weights()` incompatible.
- `DeepSetModel.hls4ml_convert(firmware_dir, build=False)` — full end-to-end export
  pipeline: build Keras model → transfer weights → `convert_from_keras_model()` →
  `hls_model.write()`. Returns the hls4ml `ModelGraph`. Set `build=True` to invoke
  Vivado HLS synthesis.
- `tests/test_deepset_hls4ml.py` — smoke test covering 1-epoch Lightning training
  followed by `hls4ml_convert()`, asserting that `firmware/` and a `.cpp` file are
  produced.

### Notes
- `KERAS_BACKEND=torch` is set via `os.environ.setdefault` inside `hls4ml_convert`;
  no manual environment setup required at call time.
- Only `aggregator: mean` is supported for HLS export; other aggregators raise a
  clear `ValueError`.
- hls4ml `Permute`/`Transpose` layers crash the bit_exact optimiser pass — the Keras
  model therefore uses `channels_last` format throughout with no permutation ops.
- `quantization_config.precision` (default `ap_fixed<16,6>`) and
  `quantization_config.reuse_factor` (default `1`) in the YAML control HLS synthesis.

---

## [Phase 4] — 2026-07-02 — Data Loading Refactor (LightningDataModule)

### Added
- `tagger/data/datamodule.py` — new `TorchTaggerDataModule(pl.LightningDataModule)`.
  - Validates required keys `data_path` and `batch_size` at construction time.
  - `setup(stage)` loads data for `'fit'`, `'validate'`, and `'test'` stages.
  - `train_dataloader()` supports `WeightedRandomSampler` for class balancing
    (configurable via `data_config.weighted_sampling`, default `True`).
  - `set_model(model)` hooks in a model's custom `collate_fn` (required for
    LorentzNet and ParticleTransformer).
  - `has_val_data` property used by `train_torch.py` to pick the right
    `ModelCheckpoint` monitor key.
- `tagger/data/config.py` — inline re-export of `CLASS_LABELS`, `EXTRA_FIELDS`,
  `FILTER_PATTERN`, `INPUT_TAG`, `N_PARTICLES`. Fixes a pre-existing
  `ModuleNotFoundError` caused by `tools.py` importing from this missing module.
  Created as a workaround because `parsers/__init__.py` has a broken legacy import
  that prevents importing from `parsers/defaults.py` directly.

### Changed
- `tagger/train/train_torch.py` — `get_data()` (107 lines) and `get_dataloaders()`
  (40 lines) removed. `train()` and `test()` now instantiate `TorchTaggerDataModule`
  and pass it to `trainer.fit()` / `trainer.test()`.

### Bugfixes (pre-existing, found during Phase 4 testing)
- `JetTagModel.configure_optimizer` renamed to `configure_optimizers` — Lightning
  requires the plural form; the old name silently prevented Lightning from finding
  the optimizer, causing a `MisconfigurationException`.
- `JetTagModel.shared_step` tuple check relaxed — LorentzNet batches contain a list
  of edge tensors mixed with a regular tensor, which previously raised
  `ValueError: Unsupported input format`.
- `JetTagModel.training/validation/test_step` — one-hot targets converted to class
  indices before `metrics.update()` to fix
  `ValueError: Expected preds to have one more dimension than target`.
- `JetTagModel.on_*_epoch_end` — non-scalar metrics (e.g. `ConfusionMatrix`, which
  returns a 2-D tensor) are now filtered out before `log_dict` to prevent Lightning
  from rejecting them.
- `JetTagModel.configure_optimizers` — YAML scalar notation (`1e-4`) is read by
  PyYAML as a string in some contexts; values in `optimizer_params` and
  `scheduler_params` are now coerced to `float`.
- `LorentzNet.build_model(self)` — signature widened to `build_model(self, **kwargs)`
  to match the `JetTagModel.compile_model()` call of `self.build_model(model_cfg=…)`.
- `ParticleTransformer.build_model` / `DeepSetModel.build_model` — removed
  `self.device = torch.device(…)` which raises `AttributeError` in PyTorch ≥ 2.0
  where `nn.Module.device` is a read-only property.

### New config keys (TorchTaggerDataModule)
- `data_config.weighted_sampling` (bool, default `True`)
- `data_config.class_weight_epsilon` (float, default `1e-6`)

---

## [Phase 3] — 2026-07-01 — PyTorch Lightning Migration

### Changed
- `tagger/model/JetTagModel.py` — base class changed from `ABC` to
  `pl.LightningModule`. Added `training_step()`, `validation_step()`,
  `test_step()`, `configure_optimizers()`, `on_train_epoch_end()`,
  `on_validation_epoch_end()`, `on_test_epoch_end()`. Replaced 140-line manual
  training loop with Lightning lifecycle hooks.
- `tagger/model/offline/LorentzNet.py` — renamed `self.jet_model` → `self.model`
  for contract consistency. Deleted 140-line custom `fit()` method.
- `tagger/train/train_torch.py` — rewrote training entry point to use
  `pl.Trainer.fit()` with `ModelCheckpoint` and `EarlyStopping` callbacks,
  TensorBoardLogger, and auto device detection.

### Added
- `torchmetrics.MetricCollection` used for accuracy, AUROC, and ConfusionMatrix
  across all models.
- `JetModelFactory` registry pattern with `@JetModelFactory.register("ModelName")`
  decorator for all models.

---

## [Phase 1 & 2] — 2026-06-30 — TF Cleanup and Import Fixes

### Changed
- Removed TensorFlow/Keras imports across all model files.
- Fixed broken factory registry and import chain.
- `tagger/model/offline/ParticleTransformer.py` — implemented full
  ParticleTransformer architecture in PyTorch (`ParticleTransformer`,
  `ParticleTransformerModel`).
- `tagger/model/online/DeepSetModel.py` — fully re-implemented using `nn.Conv1d`
  (per-particle), configurable aggregator, and MLP classifier.

---

## [Pre-refactor baseline] — legacy commits

Earlier commits (`ecf0b74`, `e1fd91a`, `d0eae91`, ...) represent the original
TensorFlow/Keras codebase and iterative PyTorch prototyping. These are preserved in
git history but superseded by the Phase 1–5 refactor.
