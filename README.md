# SARC-DETR for Remote Sensing Object Detection

This repository is based on `Get1star/PSWP-DETR` and keeps the original
PaddleDetection-style RT-DETR training pipeline intact. The SARC-DETR changes are
implemented as neck plug-ins between the backbone and `RTDETRTransformer`, so the
decoder, loss, matching, post-processing, and training flow remain unchanged.

## What Is Implemented

The repository provides two SARC variants:

- `SARC-DETR`: full training-time SARC-DETR with Similarity Adaptive Convolution
  (SAC) and Receptive Field Cross Convolution (RCC).
- `SARC-DETR-Accel`: lightweight deployment-oriented SARC-DETR. It keeps the same
  model interface and uses a single-path lightweight RCC branch to keep parameter
  count and FLOPs close to RT-DETR.

Main files:

- `ppdet/modeling/transformers/sarc_detr_encoder.py`
- `configs/rtdetr/sarc_detr_r50vd_6x_coco.yml`
- `configs/rtdetr/sarc_detr_accel_r50vd_6x_coco.yml`
- `tools/sarc_detr_stats.py`

## Algorithm Overview

SARC-DETR adds a Receptive-Field Optimizer to RT-DETR. The optimizer contains two
modules: SAC and RCC.

### Similarity Adaptive Convolution (SAC)

SAC is designed for dense remote sensing scenes where neighboring objects,
background clutter, motion blur, and object boundaries can contaminate each
other's features.

For each spatial location, SAC estimates a local prototype with a `3 x 3`
average window. It then compares the center feature with that local prototype
using a positive-definite kernel, compresses the similarity with a sigmoid gate,
and multiplies the gate back into the feature map. A per-channel affine
normalizer stabilizes the output range.

Conceptually:

```text
P(u, v) = AvgPool_3x3(F0)(u, v)
g(u, v) = sigmoid(k(F0(u, v), P(u, v)))
F1(u, v) = g(u, v) * F0(u, v)
Y = Norm(F1)
```

The default implementation uses an RBF-style similarity because the paper reports
it as the strongest kernel choice.

### Receptive Field Cross Convolution (RCC)

RCC builds a large directional receptive field without using an expensive full
`K x K` large kernel. It uses horizontal `1 x K` and vertical `K x 1` strip
convolutions, then fuses both directions with a content-adaptive gate. A compact
`3 x 3` closure mixes the fused result.

Conceptually:

```text
Fh = Conv_1xK(X)
Fv = Conv_Kx1(X)
a = softmax(DirectionGate(GAP(concat(Fh, Fv))))
Y = Conv_3x3(a_h * Fh + a_v * Fv)
```

The full SARC-DETR config uses `K=5`, matching the paper's recommended
directional span.

### SARC-DETR-Accel

SARC-DETR-Accel is the lightweight variant. The paper reports it as a
reparameterized deployment version whose overhead is close to RT-DETR. This
repository exposes it as `SARCDETRAccelHybridEncoder`, with a lightweight
single-path RCC branch for efficient inference and stable integration with the
existing PaddleDetection export flow.

Paper efficiency targets on VisDrone2019-DET:

| Method | Params (M) | FLOPs (G) | FPS |
| --- | ---: | ---: | ---: |
| RT-DETR | 41.48 | 68.19 | 42.6 |
| SARC-DETR | 55.62 | 99.76 | 22.7 |
| SARC-DETR-Accel | 42.23 | 68.55 | 42.1 |

## Environment

The original project targets PaddlePaddle:

```text
paddlepaddle-gpu==2.4.2
Python 3.8 or a compatible PaddlePaddle 2.4.x environment
CUDA matching your PaddlePaddle wheel
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use CPU only, install a CPU PaddlePaddle wheel that is compatible with
your Python version, then set `use_gpu: false` in `configs/runtime.yml`.

## Dataset Preparation

The default dataset config is:

```text
configs/datasets/coco_detection.yml
```

It is currently set for VisDrone2019-DET converted to COCO format:

```yaml
metric: COCO
num_classes: 10

TrainDataset:
  image_dir: VisDrone2019-DET-train
  anno_path: train.json
  dataset_dir: /root/autodl-tmp/visdrone

EvalDataset:
  image_dir: VisDrone2019-DET-val
  anno_path: val.json
  dataset_dir: /root/autodl-tmp/visdrone
```

Expected layout:

```text
/root/autodl-tmp/visdrone/
  train.json
  val.json
  VisDrone2019-DET-train/
    *.jpg
  VisDrone2019-DET-val/
    *.jpg
```

For a different COCO-style dataset, update:

- `num_classes`
- `TrainDataset.dataset_dir`
- `TrainDataset.image_dir`
- `TrainDataset.anno_path`
- `EvalDataset.dataset_dir`
- `EvalDataset.image_dir`
- `EvalDataset.anno_path`
- `TestDataset.dataset_dir`
- `TestDataset.anno_path`

If your annotations are not in COCO format, convert them first. The repository
contains helper scripts under `tools/`, including `tools/x2coco.py`.

## Training

The original caveat still applies: the PSWP-DETR baseline was intended to train
with 4 GPUs.

### Train SARC-DETR

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py \
  -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml \
  --eval
```

Single-GPU training is also possible for debugging, but you may need to reduce
batch size in `configs/rtdetr/_base_/rtdetr_reader.yml`:

```bash
python tools/train.py \
  -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml \
  --eval
```

### Train SARC-DETR-Accel

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py \
  -c configs/rtdetr/sarc_detr_accel_r50vd_6x_coco.yml \
  --eval
```

### Resume Training

```bash
python tools/train.py \
  -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml \
  -r output/sarc_detr_r50vd_6x_coco/latest
```

### Useful Training Options

Override config values from the command line with `-o`:

```bash
python tools/train.py \
  -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml \
  -o TrainReader.batch_size=1 LearningRate.base_lr=0.00005
```

Enable AMP:

```bash
python tools/train.py \
  -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml \
  --amp
```

## Evaluation

Set the checkpoint path with `-o weights=...` or edit the `weights` field in the
config.

Evaluate SARC-DETR:

```bash
python tools/eval.py \
  -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml \
  -o weights=output/sarc_detr_r50vd_6x_coco/model_final
```

Evaluate SARC-DETR-Accel:

```bash
python tools/eval.py \
  -c configs/rtdetr/sarc_detr_accel_r50vd_6x_coco.yml \
  -o weights=output/sarc_detr_accel_r50vd_6x_coco/model_final
```

For large remote sensing images, sliced evaluation is often useful:

```bash
python tools/eval.py \
  -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml \
  --slice_infer \
  --slice_size 640 640 \
  --overlap_ratio 0.25 0.25 \
  -o weights=output/sarc_detr_r50vd_6x_coco/model_final
```

## Inference

Run inference on one image:

```bash
python tools/infer.py \
  -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml \
  --infer_img path/to/image.jpg \
  --output_dir output/infer_sarc \
  --draw_threshold 0.5 \
  -o weights=output/sarc_detr_r50vd_6x_coco/model_final
```

Run inference on a directory:

```bash
python tools/infer.py \
  -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml \
  --infer_dir path/to/images \
  --output_dir output/infer_sarc \
  --draw_threshold 0.5 \
  --save_results True \
  -o weights=output/sarc_detr_r50vd_6x_coco/model_final
```

Use sliced inference for high-resolution aerial images:

```bash
python tools/infer.py \
  -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml \
  --infer_img path/to/large_image.jpg \
  --slice_infer \
  --slice_size 640 640 \
  --overlap_ratio 0.25 0.25 \
  --combine_method nms \
  --match_threshold 0.6 \
  --output_dir output/slice_infer_sarc \
  -o weights=output/sarc_detr_r50vd_6x_coco/model_final
```

## Export

Export a trained model for Paddle inference:

```bash
python tools/export_model.py \
  -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml \
  --output_dir output_inference \
  -o weights=output/sarc_detr_r50vd_6x_coco/model_final
```

Export the Accel variant:

```bash
python tools/export_model.py \
  -c configs/rtdetr/sarc_detr_accel_r50vd_6x_coco.yml \
  --output_dir output_inference \
  -o weights=output/sarc_detr_accel_r50vd_6x_coco/model_final
```

## Parameter Count and Forward Check

Use the helper script to build the model, print parameter counts, and run a dummy
forward pass:

```bash
python tools/sarc_detr_stats.py \
  -c configs/rtdetr/sarc_detr_r50vd_6x_coco.yml
```

For parameter count only:

```bash
python tools/sarc_detr_stats.py \
  -c configs/rtdetr/sarc_detr_accel_r50vd_6x_coco.yml \
  --params-only
```

Use the same input size as your paper or experiment when comparing FLOPs and
throughput.

## Configuration Notes

The SARC configs inherit the original RT-DETR base config:

```text
configs/rtdetr/_base_/rtdetr_r50vd.yml
```

Only the neck is replaced:

```yaml
DETR:
  backbone: ResNet
  neck: SARCDETRHybridEncoder
  transformer: RTDETRTransformer
  detr_head: DINOHead
  post_process: DETRPostProcess
```

This keeps the model path close to the baseline and reduces the risk of changing
accuracy due to unrelated decoder, loss, or data-flow modifications.

## Troubleshooting

- `ModuleNotFoundError: No module named 'paddle'`: install a compatible
  PaddlePaddle or PaddlePaddle-GPU wheel.
- CUDA extension errors: verify CUDA, PaddlePaddle, and GPU driver compatibility.
- Out of memory: reduce `TrainReader.batch_size`, use fewer decoder layers for
  debugging, or reduce input size.
- Wrong class count: update `num_classes` in
  `configs/datasets/coco_detection.yml`.
- Empty evaluation results: check `dataset_dir`, `image_dir`, and `anno_path`.
- Git or checkpoint path issues: pass paths explicitly with `-o weights=...`.

## Acknowledgement

This codebase follows the PaddleDetection RT-DETR style and is adapted from
PSWP-DETR. SARC-DETR is implemented as a minimally invasive neck replacement so
that training and inference remain compatible with the original repository.
