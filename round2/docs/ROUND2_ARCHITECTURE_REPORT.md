# RACFNet Round2 architecture report

RACFNet preserves the complete QSFNet Tnet backbone, unimodal decoders, six IIA blocks, CBAM cascade, ER block, prediction heads, and the original 14-output training interface. CCC and RTF are inserted immediately before the three original quality-guided mixed branches.

At 384x384 input, source inspection and runtime validation use these three independent feature triplets:

| Scale | RGB | Depth | Thermal | Channels | Spatial size |
|---|---|---|---|---:|---:|
| 0 | `x1_1v` | `x1_1d` | `x1_1t` | 128 | 96x96 |
| 1 | `x2_1v` | `x2_1d` | `x2_1t` | 128 | 96x96 |
| 2 | `x3_1v` | `x3_1d` | `x3_1t` | 128 | 96x96 |

No channel adapter is added. Each scale has independent CCC and RTF parameters. The original raw-QAnet-logit mixed path remains present. CCC calibration changes the three modality features by a small learned residual; RTF adds a second small learned residual to both original mixed branches before their existing `trans*_d/t` and IIA calls.

Supported configurations are `CCC_ONLY`, `RTF_ONLY`, and `FULL`. Round2 formally trains only `FULL`.
