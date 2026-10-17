# Round2 implementation diff

## Added

- `round2/RACFNet.py`: `CCCScale`, `RTFScale`, and `RACFNet` with three configuration switches.
- `round2/round2_smoke.py`: batch-1/batch-4 interface checks, 30-step checks for all modes, gradient and update audits, Softmax validation, VRAM and timing.
- `round2/round2_train.py`: resumable FULL 200-epoch fine-tuning from Round1 Tnet with original losses and B0 QAnet training-state behavior.
- `round2/round2_infer.py`: formal 1000-image inference, raw reliability tensors, maps, visual candidates, and single-modality degradation checks.
- `round2/round2_evaluate.py`: Round1-compatible eight-metric evaluation and FULL-minus-B0 deltas.

## Preserved

- All Round1 checkpoints and outputs.
- Every reusable Tnet parameter.
- Original raw-logit quality-guided fusion equation.
- Six IIA modules, ER, CBAM cascade, loss composition, and 14 outputs.
- VDT-2048 data, 384x384 input, Adam settings, and Round1 postprocessing/evaluator semantics.

## Deliberately not added

- No auxiliary reliability loss.
- No extra dataset.
- No FULL hyperparameter selection on the 1000-image test set.
- No 200-epoch CCC-only or RTF-only training in Round2.
