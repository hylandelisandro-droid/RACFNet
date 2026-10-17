# RTF implementation

At each scale, RTF independently projects RGB, Depth, and Thermal from 128 to 32 channels. The concatenation of the three 32-channel features and three one-channel reliability maps has exactly 99 channels. A 3x3 99-to-32 convolution, ReLU, and 1x1 32-to-3 convolution predict modality logits.

Softmax is applied on channel dimension 1. Runtime validation rejects a maximum absolute `Wv+Wd+Wt-1` error of `1e-5` or greater. The weighted feature is

`Wv*Fv + Wd*Fd + Wt*Ft`.

Its 128-to-128 projection is added to both original QSFNet mixed branches through independent learned RTF gamma scalars initialized to 0.1. The RTF delta projection uses small nonzero normal initialization (`std=1e-3`). IIA, ER, and all prediction heads are unchanged.

`RTF_ONLY` obtains `Qv` from a dedicated 128-to-1 convolution and obtains `Qd/Qt` from sigmoid-resized QAnet logits without calling CCC. `FULL` consumes CCC-calibrated features and `Rv/Rd/Rt` directly.
