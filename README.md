# LDR-Net

LDR-Net is a PyTorch training skeleton for a lesion-aware chest X-ray model with two internal stages:

1. A lesion localization stage that predicts lesion queries, boxes, labels, and uncertainty.
2. A disease reasoning stage that consumes lesion tokens together with anatomy and global image context.

This repository is structured as a practical starting point for research and experimentation. It includes:

- A runnable `LesionDiseaseNet` model skeleton
- A Hugging Face ConvNeXtV2 CheXpert backbone as the default encoder
- Dataset and collation utilities for mixed lesion/disease supervision
- DETR-style matching and multi-task losses
- Training and evaluation loops
- A CLI training entrypoint and a synthetic smoke-test mode

## Project Layout

```text
ldr_net/
  data/
  engine/
  losses/
  models/
  utils/
configs/
scripts/
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Format

The default dataset reader expects JSONL metadata, one sample per line:

```json
{
  "image_path": "images/sample_0001.png",
  "boxes": [[0.42, 0.48, 0.18, 0.12]],
  "labels": [3],
  "disease_labels": [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "has_detection": true,
  "sample_id": "sample_0001"
}
```

Conventions:

- `boxes` use normalized `cx, cy, w, h`
- `labels` are lesion class ids aligned with the detection head
- `disease_labels` is a multi-hot vector aligned with the disease head
- `lesion_concept_labels` can store image-level lesion concepts when available
- `has_detection` can be `false` for weakly supervised disease-only samples

Images may be PNG, JPG, or any format supported by `torchvision.io.read_image`.

## VinDr Conversion

The repo now includes a Kaggle-friendly VinDr converter that turns the `annotations_*.csv` and `image_labels_*.csv` files into the JSONL format used by the loader.

Expected source layout:

```text
vindr_physionet_384/
  train/
  test/
  annotations_train.csv
  annotations_test.csv
  image_labels_train.csv
  image_labels_test.csv
```

Run:

```bash
python scripts/convert_vindr_to_jsonl.py --source-root vindr_physionet_384
```

This writes:

- `data/vindr_physionet_384/train.jsonl`
- `data/vindr_physionet_384/test.jsonl`
- `data/vindr_physionet_384/metadata.json`

The converter treats `COPD`, `Lung tumor`, `Pneumonia`, `Tuberculosis`, `Other diseases`, and `No finding` as diseases. The remaining image-level label columns are stored as lesion concepts. Bounding-box classes come from the unique `class_name` values in the annotation CSVs.

## Training

Update [`configs/ldr_net_v1.yaml`](/d:/Github%20Repos/LDR-Net/configs/ldr_net_v1.yaml) and run:

```bash
python scripts/train.py --config configs/ldr_net_v1.yaml
```

For a CPU smoke test with synthetic data:

```bash
python scripts/train.py --config configs/ldr_net_smoke.yaml --synthetic
```

For VinDr stage-1 training after conversion:

```bash
python scripts/train.py --config configs/vindr_stage1.yaml
```

## Notes

- The default backbone is `shreydan/CheXpert-5-convnextv2-tiny-384`, which is a fine-tuned Hugging Face checkpoint built on `facebook/convnextv2-tiny-22k-384`.
- The smoke config still uses `resnet18` so a quick local sanity check does not depend on downloading weights.
- Pretrained CXR backbones, stronger augmentations, prototype regularization, and semi-supervised training can be layered in later without changing the overall package structure.
