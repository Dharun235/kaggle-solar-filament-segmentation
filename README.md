# Solar Filament Segmentation

Reproducible baseline for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

Pipeline:

```text
audit data → create grouped folds → train/infer model → tune confidence + instance cap on PQ
→ create RLE submission → audit submission
```

The included model is a small patch U-Net. Replace only the model command when testing another model.

## 1. Data

Accept the competition rules and download the data from Kaggle. Expected layout:

```text
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/*.jpeg
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/test_images/*.jpeg
```

Do not commit competition data, test labels, checkpoints, or submissions. Do not use public datasets containing competition test labels. This avoids leakage and follows the competition rules.

## 2. Run on your own PC

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m unittest discover -v
```

Set `DATA_ROOT` to the directory containing `train/` and `test/`, then run:

```bash
DATA_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026

python main.py \
  --data-root "$DATA_ROOT" \
  --run-dir artifacts/runs/unet_pc_pq_v2 \
  --confidence-grid 0.20 0.25 0.30 0.35 0.40 0.50 \
  --max-instances-grid 1 2 3 4 5 8 10 \
  --model-command 'python models/patch_unet.py \
    --train {train_manifest} --val {val_manifest} --test {test_manifest} \
    --raw-val {raw_val} --raw-test {raw_test} --run-dir {run_dir} \
    --ground-truth {ground_truth} --validate-every 3 \
    --device auto --epochs 15 --samples-per-image 2 \
    --batch-size 8 --stride 512 --infer-batch 16 \
    --threshold 0.35 --min-area 80 --max-candidates 20'
```

`--device auto` selects CUDA, Apple MPS, or CPU.

Training uses reproducible fresh crops each epoch and samples filament centers from
one foreground pixel. Every three epochs (and at the final epoch), the model evaluates
full validation images against every independent annotator record. It selects the
checkpoint and pixel threshold by mean PQ, using confidence 0.2 and at most 10 instances.
`--threshold-grid` defaults to `0.2 0.3 0.35 0.4 0.5 0.6`; `--threshold` is also included.
The controller then tunes instance confidence and cap for the selected model.
Test inference runs once, using the selected checkpoint and pixel threshold.
Validation increases runtime; `--validate-every` controls the frequency. Cached current
and selected validation maps use about 4 GiB for 123 images. Keep the same fold for
comparison with the existing baseline; use a new run directory to preserve it.

## 3. Run on Kaggle GPU

Create a Kaggle Notebook, attach the competition data, and select **GPU** under notebook settings. Run these cells separately.

Reference notebook: [solar-fil Kaggle notebook, version 349781629](https://www.kaggle.com/code/dharun235/solar-fil?scriptVersionId=349781629).

Clone repository:

```python
%cd /kaggle/working
!rm -rf solar-fil
!git clone https://github.com/Dharun235/kaggle-solar-filament-segmentation.git solar-fil
%cd /kaggle/working/solar-fil
```

Run full pipeline:

```python
!python main.py \
  --data-root /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026 \
  --run-dir /kaggle/working/artifacts/runs/unet_pq_v2 \
  --confidence-grid 0.20 0.25 0.30 0.35 0.40 0.50 \
  --max-instances-grid 1 2 3 4 5 8 10 \
  --model-command 'python models/patch_unet.py \
    --train {train_manifest} --val {val_manifest} --test {test_manifest} \
    --raw-val {raw_val} --raw-test {raw_test} --run-dir {run_dir} \
    --ground-truth {ground_truth} --validate-every 3 \
    --device cuda --epochs 15 --samples-per-image 2 \
    --batch-size 8 --stride 512 --infer-batch 16 \
    --threshold 0.35 --min-area 80 --max-candidates 20'
```

Confirm log contains:

```text
device=cuda
DONE
submission=/kaggle/working/artifacts/runs/unet_pq_v2/submission.csv
```

## 4. Check and submit

Run submission audit:

```python
!python scripts/audit_submission.py \
  --submission /kaggle/working/artifacts/runs/unet_pq_v2/submission.csv \
  --test-images /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images
```

Expected final line:

```text
submission=ok
```

Upload this file on Kaggle:

```text
/kaggle/working/artifacts/runs/unet_pq_v2/submission.csv
```

For a Kaggle Notebook, use **Save Version → Save & Run All** before submitting. For a classic competition, upload the CSV from the competition’s **Submit Predictions** page.

## Outputs

Each run is stored under its `--run-dir`:

```text
run.json                 run configuration and Git commit
state.json               current pipeline stage
validation.csv           confidence/PQ/instance-cap sweep
validation_preview.csv   RLE preview for the selected validation settings
selected_confidence.json selected validation threshold and instance cap
submission.csv           final Kaggle file
patch_unet.pt            checkpoint with best validation PQ
selected_model.json      selected epoch, pixel threshold, and selection PQ
checkpoint_metrics.json  epoch/pixel-threshold PQ sweep
probabilities/val/*.npy   selected checkpoint validation probabilities (float32)
```

Validation uses the competition's updated Panoptic Quality (PQ) metric with strict `IoU > 0.5` one-to-one instance matching. PQ is `sum(matched IoU) / (TP + 0.5 FP + 0.5 FN)` and penalizes false positives, missed filaments, fragmentation, and over-merging. The pipeline tunes both confidence and the maximum number of instances on validation data. See the [official evaluation details](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking).

The organizers report that the leaderboard was rescored on August 12, 2026. The organizers also evaluate reproducibility, segmentation quality, and the source repository. Do not use public test annotations or other ground-truth metadata for inference, even though community discussions report overlap between some test images and the public MAGFiLO release.

## Model interface

Custom model must write one JSONL record per physical image to both `{raw_val}` and `{raw_test}`:

```json
{"image_id":"20140609195854Bh","instances":[
  {"score":0.82,"mask_path":"path/to/2048x2048_binary_mask.png"}
]}
```

See `scripts/model_adapter_template.py`.

## Repository structure

```text
main.py                       Complete pipeline controller
models/patch_unet.py          Baseline model
models/simple_cv.py           Fast CPU smoke baseline
scripts/audit_data.py         Data validation
scripts/prepare_data.py       COCO parsing and folds
scripts/postprocess.py        PQ, filtering, de-overlap, RLE
scripts/audit_submission.py   Submission validation
tests/test_metric.py          PQ tests
```
