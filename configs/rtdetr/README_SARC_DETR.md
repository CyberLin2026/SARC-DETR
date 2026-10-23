# SARC-DETR

This repository keeps the original RT-DETR training, decoder, head, losses, and
post-processing path unchanged. SARC-DETR is added as a neck plug-in between the
backbone and `RTDETRTransformer`.

## Configs

- `sarc_detr_r50vd_6x_coco.yml`: full SARC-DETR with SAC and training-time RCC.
- `sarc_detr_accel_r50vd_6x_coco.yml`: SARC-DETR-Accel with the same SAC gate and
  a single-path depthwise-separable RCC deployment branch.

## Modules

- `SimilarityAdaptiveConvolution` implements SAC. It estimates a local prototype
  with a 3x3 average window, computes an RKHS-style similarity gate, applies a
  sigmoid compressor, and normalizes the gated response per channel.
- `ReceptiveFieldCrossConvolution` implements RCC. The full model uses horizontal
  `1 x K` and vertical `K x 1` strip transforms, a content-adaptive directional
  gate, and a compact 3x3 isotropic closure. The default span is `K=5`.
- `SARCDETRAccelHybridEncoder` implements the lightweight deployment variant.

The paper reports the following VisDrone2019-DET efficiency targets:

| Method | Params (M) | FLOPs (G) |
| --- | ---: | ---: |
| RT-DETR | 41.48 | 68.19 |
| SARC-DETR | 55.62 | 99.76 |
| SARC-DETR-Accel | 42.23 | 68.55 |

Use `tools/sarc_detr_stats.py` in a PaddlePaddle environment to build the model,
run a dummy forward pass, and print parameter counts:

```bash
python tools/sarc_detr_stats.py -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml
python tools/sarc_detr_stats.py -c configs/rtdetr/sarc_detr_accel_r50vd_6x_coco.yml
```
