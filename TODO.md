# TODOs

## Phase 2+ Migration Work

- [ ] ParticleNet integration — `tagger/model/offline/ParticleNet.py` is orphaned (no `@JetModelFactory.register` decorator, not integrated into training pipeline). Either add decorator + `JetTagModel` wrapper OR delete. Defer to Phase 7 cleanup.

## Future Work

- [ ] Implement rest of the models
- [x] Figure out evaluation metrics (craft own or pre-built e.g. TorchMetrics?)
- [x] Use `model.evaluate()` during training for validation?
- [ ] Implement a generic Knowledge Distillation model
- [ ] Implement k-fold?
- [ ] Implement hyperparameter optimization
- [ ] Implement quantization and pruning (PQuant, HGQ2)?
- [x] Integrate ROOT-data loading.
- [ ] Clean up plotting stuff (plots controlled from config?)
- [ ] Documentation for usage.
  