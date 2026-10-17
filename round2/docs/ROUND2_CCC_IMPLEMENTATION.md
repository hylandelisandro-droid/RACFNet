# CCC implementation

Each scale performs the fixed V2.0 computation:

1. Build `Qd` and `Qt` by sigmoid after resizing raw QAnet logits, and learn `Qv` through a 128-to-1 convolution plus sigmoid.
2. Project the three modality features independently from 128 to 32 channels and L2-normalize along channels with epsilon `1e-6`.
3. Compute `Avd`, `Avt`, and `Adt` from pixelwise cosine similarity mapped from `[-1,1]` to `[0,1]`.
4. Produce independent consistency evidence `Cv/Cd/Ct` and parallel self-information evidence `Sv/Sd/St`.
5. Calibrate `Rv/Rd/Rt` from quality, consistency, and self evidence.
6. Apply independent 128-to-128 residual projections gated by reliability and scaled by learned `gamma_v/gamma_d/gamma_t`, initialized to 0.1.

Residual projection weights use a small nonzero normal initialization (`std=1e-3`); other convolutions use PyTorch's normal convolution initialization. The implementation exposes auxiliary tensors only when explicitly requested, preserving the original prediction interface by default.
