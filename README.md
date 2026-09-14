# Solar Filament Segmentation pipeline

Model-neutral pipeline for MAGFiLO/Kaggle 2026. Model code stops at one contract:
write raw instance predictions to JSON. Everything else is implemented here.

## Pipeline

```text
audit → grouped folds/manifests → model adapter → calibrated postprocess
      → legal non-overlapping masks → COCO RLE → local PQ → submission audit
```

## Setup

```bash
conda activate /Users/dharunkumar/Dev/kaggle/solar-fil/.conda-envs/solar
python -m pip install -r requirements.txt
```

Data expected at `MAGFiLO_1.0_Kaggle_2026/`.

The competition dataset is intentionally excluded from Git. Download it after accepting
the Kaggle rules, then place it at that path. Never commit Kaggle credentials, raw data,
generated masks, checkpoints, or submissions.

## Run

Single controller:

```bash
python main.py
```

Fast Mac smoke model, no GPU/training required:

```bash
python main.py --model-command \
  'python models/simple_cv.py --train {train_manifest} --val {val_manifest} \
   --test {test_manifest} --raw-val {raw_val} --raw-test {raw_test} --run-dir {run_dir}'
```

Patch U-Net model, community-style baseline:

```bash
python main.py --epochs 3 --model-command \
  'python models/patch_unet.py --train {train_manifest} --val {val_manifest} \
   --test {test_manifest} --raw-val {raw_val} --raw-test {raw_test} --run-dir {run_dir}'
```

Without model command, controller prepares data and prints model-hook instructions. Full run
expects model command to create two JSONL files. Command supports these placeholders:
`{train_manifest}`, `{val_manifest}`, `{test_manifest}`, `{raw_val}`, `{raw_test}`,
`{run_dir}`, `{fold}`.

```bash
python main.py --model-command \
  'python models/run_model.py --train {train_manifest} --val {val_manifest} \
   --test {test_manifest} --raw-val {raw_val} --raw-test {raw_test} --run-dir {run_dir}'
```

Each run stores metadata, validation CSV, and final submission under
`artifacts/runs/latest/`. This is lightweight experiment tracking: git SHA, arguments,
paths, and outputs stay together without requiring an MLflow server.

Official-protocol validation uses strict `IoU > 0.5` matching and scores each prediction
set separately against each annotator record, then averages those record scores. Run the
standalone evaluator when checking an existing prediction file:

```bash
python scripts/evaluate_official.py --predictions artifacts/raw/val_fold0.jsonl \
  --ground-truth MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
```

Manual stages, useful while developing model code:

```bash
python scripts/prepare_data.py --folds 5
```

This writes `artifacts/manifests/`. Train any instance-segmentation
model using those files. Model adapter must write raw predictions:

```json
{"image_id":"20140609195854Bh","instances":[
  {"score":0.82,"mask_path":".../mask_001.png"}
]}
```

`image_id` is test filename stem without `.jpeg`. Masks must be binary PNGs at original
`2048x2048` resolution. Multiple JSON lines allowed, one image per line.

Process validation predictions and tune thresholds:

```bash
python scripts/postprocess.py \
  --predictions artifacts/raw/val_fold0.jsonl \
  --output artifacts/pred/val_fold0.csv \
  --confidence-grid 0.10 0.15 0.20 0.25 0.30 0.40 0.50 \
  --ground-truth MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
```

Generate test submission:

```bash
python scripts/postprocess.py \
  --predictions artifacts/raw/test.jsonl \
  --output submission.csv \
  --confidence 0.20 --max-instances 10
python scripts/audit_submission.py --submission submission.csv \
  --test-images MAGFiLO_1.0_Kaggle_2026/test/test_images
```

Local PQ uses IoU threshold 0.5 and official formula. Images with no predicted filament
produce no CSV row. No overlapping predicted pixels allowed.
