# Round3 method and Figure 2 consistency audit

This is a read-only audit. Round3 does not change `CCCScale`, `RTFScale`, IIA, CBAM, ER, the prediction heads, or the loss.

## Direct code mapping

- The three actual modal triplets are `(x1_1v,x1_1d,x1_1t)`, `(x2_1v,x2_1d,x2_1t)`, and `(x3_1v,x3_1d,x3_1t)`.
- Every triplet entering CCC/RTF has shape `[B,128,96,96]` for 384x384 inputs. They are three decoder-path integration nodes, not three different spatial resolutions.
- `Q_d` and `Q_t` inside an active innovation are `sigmoid(resize(QAnet logits))`; the original quality-guided base equations continue to use raw pooled logits.
- CCC outputs calibrated `(Fv,Fd,Ft)` plus `(Rv,Rd,Rt)` and pairwise maps `(Avd,Avt,Adt)`.
- FULL passes CCC-calibrated features and reliabilities into RTF. RTF_ONLY passes raw modal features and independently generated `(Qv,Qd,Qt)` into RTF.
- RTF creates one weighted feature and one projected `Delta`, then adds it to two distinct mixed branches using separate `gamma_d` and `gamma_t`. Both enhanced branches enter the original bidirectional IIA blocks.
- IIA, CBAM, ER, and all prediction heads are unchanged.

## Fourteen-output supervision

The forward interface is unchanged: one edge output, nine unimodal outputs, and four fused outputs. `full_loss` applies BCE-with-logits to the edge output, structure loss to the nine unimodal predictions and three intermediate fused predictions, and twice-weighted structure loss to the final fused prediction.

## Figure 2 corrections required before publication

1. The labels `Scale 1 (1/8)`, `Scale 2 (1/16)`, and `Scale 3 (1/32)` imply three spatial resolutions, but the implemented tensors are all 96x96 (1/4 of 384). Relabel them as three decoder integration paths/nodes, or redraw the sizes from verified tensors.
2. The figure's single `F_base + Delta F_rtf -> Fout` line hides the implemented dual output. The code produces `X_d_new=X_d_base+gamma_d*Delta` and `X_t_new=X_t_base+gamma_t*Delta`, then feeds both to IIA.
3. The original quality-guided path is not bypassed. In CCC_ONLY it consumes calibrated features; in RTF_ONLY it consumes raw features. Its quality terms retain raw-logit B0 semantics.
4. The diagram should not imply that a single final saliency loss and a single edge loss are the only supervised outputs. Training uses all 14 outputs as described above.
5. `Rv/Rd/Rt` are calibrated reliability evidence, not ground-truth sensor correctness probabilities; figure/caption wording should preserve this distinction.

The Figure 2 artwork was not modified in Round3.
