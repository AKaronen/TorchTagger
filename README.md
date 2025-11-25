# TensorFlow → PyTorch Migration TODO


This file captures the migration plan and tracked tasks for converting the project from TensorFlow/Keras to PyTorch. It mirrors the tracked todo list used during the migration planning.

## High-level status
- [x] Inventory TensorFlow usages
- [x] Produce migration plan
- [x] Prototype LorentzNet conversion
- [x] Convert training script
- [x] Convert dataloaders
- [ ] Replace model-specific TF utilities (pruning/quant/qkeras/hls4ml)
- [ ] Update notebooks and examples
- [ ] CI and packaging updates
- [ ] Run validation tests
- [ ] Final cleanup and docs

---

## Tasks

1. Add PyTorch LorentzNet implementation
   - File: `tagger/model/LorentzNet.py`
   - Implement LGEB and LorentzNetModel as `torch.nn.Module` matching TF forward signature.

2. Implement segment-aggregation utilities
   - File: `tagger/model/torch_utils.py`
   - Provide `unsorted_segment_sum` / `unsorted_segment_mean` using `torch_scatter` if available, otherwise fallback to `index_add_`.

3. Register PyTorch model in factory
   - File: `tagger/model/common.py`
   - Add registry entry (e.g. `LorentzNet`) or backend selection logic.
   

4. Add a TorchDataset wrapper
   - File: `tagger/data/torch_dataset.py`
   - Wrap `load_np_data` outputs into `torch.utils.data.Dataset` and implement collate_fn if needed.

5. Add minimal training runner
   - Option: add `--backend pytorch` branch in `tagger/train/train.py` or create `train_torch.py`.
   - Implement DataLoader, optimizer, loss, simple training loop, checkpointing with `torch.save`.

6. Add unit tests

7. Add README snippet / dev notes

---

### Other
- Convert remaining models: `DeepSetModel.py`, `DeepSetModelHGQ.py`, `InteractionNetModel.py`, `TransformerModel.py`
- Convert full training script and callbacks to PyTorch (EarlyStopping, ReduceLROnPlateau, sample weights).
- Replace/decide on TF-specific tooling (tfmot, qkeras, hls4ml).
- Update notebooks and examples
- Update `environment.yml` and CI to install PyTorch (and optionally `torch_scatter`).
- Add a CI smoke test that runs a 1-epoch smoke training for each model.

---

## Decisions to make
- Whether to add `torch_scatter` as dependency (recommended for performance/clarity). If not, use `index_add_` fallback.
- How to handle `hls4ml` (keep Keras export path or document temporary limitation).
- Registry approach: support both TF and PyTorch in model factory during transition, or replace in-place.
- Pytorch-Lightning?
- CLI: require config, commandline arguments as override?
- Refactor and cleanup tagger module
- Use TorchMetrics or manually code metrics?

## Quick commands (dev)


To run: `python ./tagger/train/train_torch.py [train|test] -c $PATH_TO_CONFIG$`
