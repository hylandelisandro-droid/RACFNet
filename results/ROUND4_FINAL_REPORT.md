# RACFNet Round4 final report

## Completion status

Round4 completed the three requested read-only experiments: all 14 official VDT-2048 challenge attributes, all 27 deterministic single-modality degradation conditions for four formal models and 1,000 test images, and CCC/RTF reliability-weight-residual analysis. It also produced Tables 1-6, candidate Figures 5-8 and three degradation curve figures. No model was retrained and no Round1-Round3 artifact was modified.

## Formal models used

- B0: Round1, epoch 200, step 52400, checkpoint identity `B0_Tnet_final.pth`.
- CCC_ONLY: Round3, epoch 200, step 52400, checkpoint identity `CCC_ONLY_RACFNet_final.pth`.
- RTF_ONLY: Round3, epoch 200, step 52400, checkpoint identity `RTF_ONLY_RACFNet_final.pth`.
- FULL: Round2, epoch 200, step 52400, checkpoint identity `FULL_RACFNet_final.pth`.

These identities are inherited from the already accepted Round1-Round3 results, as requested; Round4 did not repeat the excluded checkpoint/BN/bootstrap audits.

## Official challenge evaluation

`Table2_Visible.csv`, `Table3_Depth.csv`, `Table4_Thermal.csv` and `Challenge_Comparison.csv` contain four-model values and official sample counts. All improvements and regressions are retained. FULL is not uniformly best in every challenge category; for example, RTF_ONLY leads several T-RD and V-SA entries, so the evidence supports condition-dependent gains rather than universal dominance.

## Synthetic degradation robustness

All inputs were generated in memory with fixed seed 20261013. QAnet and the main model received the same degraded modality input. The evaluator streamed predictions and retained statistics rather than 108,000 PNG files.

- B0: mean ΔS=-0.003140, mean ΔFmax=-0.004465, mean ΔMAE=+0.000126.
- CCC_ONLY: mean ΔS=-0.003193, mean ΔFmax=-0.004594, mean ΔMAE=+0.000123.
- RTF_ONLY: mean ΔS=-0.003225, mean ΔFmax=-0.004629, mean ΔMAE=+0.000127.
- FULL: mean ΔS=-0.003145, mean ΔFmax=-0.004631, mean ΔMAE=+0.000124.

These results are diagnostic image perturbations, not a physical sensor-failure experiment. Per-condition and per-image rows are preserved so unfavorable cases remain auditable.

## CCC and RTF mechanism evidence

The delivered CSVs separately report CCC evidence means/spatial variance, RTF means, dominant-area shares, entropy, near-zero/near-one saturation, residual ratios and degraded-minus-clean shifts. Existing RTF_ONLY Softmax collapse is retained and treated as a limitation. CCC evidence must not be described as a ground-truth probability of sensor correctness, and a weight shift alone does not prove causal robustness.

## Evidence-supported claims

- The fixed CCC and RTF modules participate in inference and produce measurable reliability, weight and residual outputs.
- The four formal models show different condition-dependent challenge and degradation behaviour.
- Reliability/weight maps have measurable spatial and perturbation-dependent variation where the CSV statistics show it.

## Claims not established

- Cross-random-seed stability, multi-dataset generalization and physical sensor-failure robustness are not established.
- RTF saturation prevents a strong claim of balanced adaptive tri-modal selection.
- Local weight changes cannot be called correct or causal solely because aggregate performance changes.

## Paper deliverables

- Tables 1-6: `Table1_Overall.csv`, `TABLE2_VISIBLE.csv`,
  `TABLE3_DEPTH.csv`, `TABLE4_THERMAL.csv`, `Table5_Complexity.csv` and
  `Table6_Ablation.csv`.
- Public-safe result tables and compact mechanism summaries: `results/`.
- Derived degradation curves: `figures/`.
- Checkpoint identities and hashes: `checkpoints/README.md`.

The full research archive, raw mechanism tensors, dataset composites and large
checkpoints are intentionally not part of this code repository.

Table 1 contains only the four controlled in-project models. No external-method numbers were invented.
