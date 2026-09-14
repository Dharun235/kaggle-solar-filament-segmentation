# Solar Filament Segmentation

Reproducible baseline for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

The repository separates model code from data preparation, validation, post-processing, and submission generation. The included model is a small patch U-Net intended as a clear baseline; replace it when experimenting with stronger models.

## Competition task

Predict one binary instance mask for each detected solar filament in every 2048×2048 H-alpha test image.

The competition uses Panoptic Quality (PQ):

```text
PQ = sum(IoU of matched pairs) / (TP + 0.5*FP + 0.5*FN)
```

Instances are matched at `IoU > 0.5`. Fragmentation, merging, false positives, and missed filaments therefore matter; foreground pixel Dice alone is not an adequate model-selection metric. The organizers also assess morphology, method description, and code quality. See the [official evaluation description](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/prizes).

## Repository layout

```text
main.py                              Full pipeline controller
models/patch_unet.py                 Small trainable patch U-Net baseline
models/simple_cv.py                  Training-free CPU smoke baseline
scripts/audit_data.py                Input-data checks
scripts/prepare_data.py              COCO parsing and grouped manifests
scripts/postprocess.py               Filtering, de-overlap, PQ, and RLE
scripts/audit_submission.py          Final CSV and mask validation
scripts/evaluate_official.py         Standalone local PQ evaluator
scripts/model_adapter_template.py    Model output contract
artifacts/                           Generated outputs; not tracked
```

## Data

Accept the competition rules and download the data from Kaggle. Do not commit competition data or hidden/test labels.

Expected layout:

```text
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/*.jpeg
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/*.jpeg
```

The public discussions mention overlap between some public MAGFiLO releases and competition test images. Do not use external test ground truth or metadata for training, validation, threshold selection, or pseudo-labeling. Use only competition-provided training annotations unless competition rules explicitly allow another source.

## Installation

```bash
conda create -n solar python=3.11 -y
conda activate solar
python -m pip install -r requirements.txt
```

For Kaggle, attach competition data, enable a GPU, clone this repository into `/kaggle/working`, and run from repository root. Kaggle notebook cells use `%cd` and `!`; normal terminals do not.

## Run complete pipeline

Set `DATA_ROOT` to directory containing `train/` and `test/`.

```bash
DATA_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026
RUN_DIR=artifacts/runs/unet_baseline

python main.py \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR" \
  --confidence-grid 0.20 0.25 0.30 0.35 0.40 0.50 \
  --model-command 'python models/patch_unet.py \
    --train {train_manifest} \
    --val {val_manifest} \
    --test {test_manifest} \
    --raw-val {raw_val} \
    --raw-test {raw_test} \
    --run-dir {run_dir} \
    --device auto \
    --epochs 15 \
    --samples-per-image 2 \
    --batch-size 8 \
    --stride 512 \
    --infer-batch 16 \
    --threshold 0.35 \
    --min-area 80 \
    --max-candidates 20'
```

`--device auto` selects CUDA, Apple MPS, or CPU. Use `--device cuda` on Kaggle when GPU is enabled.

Fast smoke test:

```bash
python main.py \
  --data-root "$DATA_ROOT" \
  --run-dir artifacts/runs/smoke \
  --model-command 'python models/patch_unet.py \
    --train {train_manifest} --val {val_manifest} --test {test_manifest} \
    --raw-val {raw_val} --raw-test {raw_test} --run-dir {run_dir} \
    --device auto --epochs 1 --samples-per-image 1 --max-train-rows 50 \
    --batch-size 8 --stride 1024 --infer-batch 16 --threshold 0.35 \
    --max-candidates 20'
```

Controller stages:

```text
data audit → manifests/folds → model training/inference
→ validation PQ and confidence selection → test CSV → submission audit
```

Each run records `run.json`, `state.json`, `metric.json`, raw predictions, masks, `validation.csv`, and `submission.csv` under its run directory.

## Validation design

`prepare_data.py` groups records by physical JPEG and month before assigning folds. This prevents duplicate annotator records for one physical image from crossing train and validation folds.

Training manifests retain annotated records. Validation and test inference contain one record per physical image. The evaluator reads every annotator record from the COCO file and scores the same prediction separately against each annotation record, then averages those scores. It does not union annotations or give duplicate inference credit.

`postprocess.py` selects confidence on validation PQ, applies a minimum-area filter and instance limit, then greedily removes overlapping pixels from lower-ranked masks. This produces non-overlapping instance masks before RLE encoding.

## Model output contract

Any model can replace `models/patch_unet.py` if it writes two JSONL files: one validation file and one test file. Each line represents one physical image:

```json
{"image_id":"20140609195854Bh","instances":[
  {"score":0.82,"mask_path":"path/to/mask.png"}
]}
```

Requirements:

- `image_id` is the JPEG stem, without `.jpeg`.
- Each mask is a binary PNG with shape `2048×2048`.
- Each instance has numeric `score` and `mask_path`.
- Exactly one prediction record is written per physical image.
- The model writes both `{raw_val}` and `{raw_test}`.

See `scripts/model_adapter_template.py` for the interface.

## Submission format and checks

Final file:

```csv
filament_id,segmentation_rle
20140609195854Bh_1,...
```

There is one row per predicted filament. `filament_id` must be `<test-image-stem>_<instance-number>`. Masks use COCO-compatible column-major RLE. Empty masks, unknown image IDs, duplicate IDs, invalid RLE, and overlapping predicted pixels are rejected by the audit.

Run audit before uploading:

```bash
python scripts/audit_submission.py \
  --submission artifacts/runs/unet_baseline/submission.csv \
  --test-images "$DATA_ROOT/test/test_images"
```

Upload `submission.csv` on the competition submission page. Keep code, configuration, checkpoint, validation result, and submission description together for reproducibility. See [Kaggle competition workflow](https://www.kaggle.com/docs/competitions).

## Manual evaluation

```bash
python scripts/evaluate_official.py \
  --predictions artifacts/runs/unet_baseline/raw_val.jsonl \
  --ground-truth "$DATA_ROOT/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
```

Use validation PQ for model and post-processing decisions. Hidden leaderboard score remains final result.

## Reproducibility and scope

- Included U-Net fixes random seed to `42`.
- Generated artifacts, checkpoints, submissions, raw data, and credentials are excluded from Git.
- Baseline is not a claimed winning solution.
- Stronger next steps are model changes: instance-aware detection/segmentation, better consensus targets, higher-resolution refinement, test-time augmentation, or carefully validated ensembles. Judge every change with the same PQ protocol and leakage controls.

## License and competition data

This repository contains code only. Competition data, annotations, and third-party weights remain subject to their original Kaggle or author licenses and are not redistributed here.
