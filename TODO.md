# TODOs

## Remaining Migration Work

- [ ] **ParticleNet** — `tagger/model/offline/ParticleNet.py` is orphaned (no `@JetModelFactory.register` decorator, not wired into training pipeline). Either add `JetTagModel` wrapper and register, or delete. Deferred to Phase 7.
- [ ] **Update notebooks and examples** — existing notebooks reference the old TF/Keras API.
- [ ] **CI smoke tests** — add per-model 1-epoch smoke tests to CI pipeline.

## Future Work

- [ ] Knowledge Distillation model
- [ ] k-fold cross-validation
- [ ] Hyperparameter optimisation (Optuna / Ray Tune)
- [ ] Pruning support for DeepSetModel (L1 unstructured pruning is partially implemented via `_prune_model()`, not yet wired to hls4ml export)
- [ ] hls4ml export for LorentzNet and ParticleTransformer
- [ ] `num_workers > 0` in DataLoaders (currently `0` for Windows compatibility)
- [ ] Distributed training (multi-GPU)
- [ ] Data augmentation config
- [ ] Clean up plotting (`plot/` module, controlled from config)
- [ ] ROOT data loading integration for training (currently numpy-only path is primary)
