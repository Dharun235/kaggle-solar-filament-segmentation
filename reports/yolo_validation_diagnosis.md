# YOLO validation diagnosis — 2026-09-16

The largest observed weakness is missed filaments, especially small objects. Mask
extent errors are the second useful target. Confidence and instance-cap tuning
have already been tested and are unlikely to provide a large improvement alone.

## Evidence and reproducibility

Replayed the downloaded YOLOv8-S `best.pt` on all 123 month-grouped fold0 photos,
evaluated separately against all 197 annotator records. Training photos remain
excluded. The checkpoint SHA256 is
`aebef8e47380842e5935ab6ee5aef0430e159f7e3144c703a828e1d6165bcd80`.
Ultralytics 8.4.152, input 2048, native masks, detection confidence floor 0.1,
NMS IoU 0.7, final confidence 0.2, cap 16, minimum area 80, confidence-ordered
pixel exclusion. CPU replay reproduced Kaggle PQ **0.4012144075536765 exactly**.

Saved local artifacts under `output/yolo_diagnosis/`: `summary.json`,
`validation.csv`, `ground_truth.csv`, `predictions.csv`, `predictions/*.json`,
`examples.png`. Replay script: `output/diagnose_yolo.py`. Original downloaded
training logs, grid and both checkpoints: `output/yolo_diagnosis_source/`.

Counts below are per annotator record, not unique physical objects. Diagnostic
categories can overlap and should not be added together.

| Quantity | Result |
|---|---:|
| Matched ground-truth instances | 741 / 1,303 |
| Missed ground-truth instances | 562 |
| False-positive predictions | 444 |
| Pooled precision | 62.5% |
| Pooled recall | 56.9% |
| Mean IoU of matched pairs | 0.672 |

The competition-style PQ is averaged by annotation record; multiplying the pooled
precision/recall-derived detection score by pooled IoU will not reproduce it.

## 1. Detection and small-object recall — highest priority

340/562 misses (60.5%) have best retained-mask IoU below 0.1. This indicates little
useful mask overlap, not proof that the detector emitted no bounding box. Even
with confidence 0.1 and cap 100, 262 of these still have no mask exceeding IoU 0.5.
286/562 misses (50.9%) have annotated area below 1,000 pixels.

| Ground-truth area, pixels | Matched / total | Recall |
|---|---:|---:|
| <400 | 26 / 105 | 24.8% |
| 400–999 | 184 / 391 | 47.1% |
| 1,000–2,999 | 346 / 537 | 64.4% |
| 3,000–7,999 | 160 / 231 | 69.3% |
| ≥8,000 | 25 / 39 | 64.1% |

Selected visual examples show an unpredicted small dark filament and a missed
large faint curved filament near the limb. They illustrate failure types; they
are deliberately selected examples, not a random visual sample.

A useful subsequent model experiment would test improved small-object/context
handling, such as crop-based inference combined with full-image predictions.
It needs validation of both new detections and duplicate suppression; it is not
an established improvement and is not part of the currently launched run.

## 2. Mask extent and the strict matching threshold

155 misses have best IoU between 0.3 and 0.5; 98 are above 0.4 but fail the strict
IoU >0.5 criterion. Of partial matches with IoU >0.1, 78 have a best predicted
mask more than twice the GT area and 33 have one less than half the GT area.
45 misses have at least two predicted masks each covering 10% of the annotated
object. These are fragmentation indicators, not definitive split diagnoses.

Visual inspection confirms examples of excess mask area, incomplete extent and
several pieces along a curved filament. Both over- and under-segmentation occur,
so blanket dilation or closing is not justified. A targeted refinement experiment
should report changes separately for these groups, including lost true positives.

## 3. False positives and annotator disagreement

192/444 false positives have best IoU ≤0.1 for the evaluated annotator record.
100/444 false positives (22.5%) match another annotator's object on the same photo.
Thus some penalized detections reflect annotation disagreement; they cannot all
be described as hallucinated objects. An illustrative large false-positive mask
near the solar limb follows faint image structure without a matching annotation
in the shown record. Its physical validity cannot be established from this audit.

## What the saved experiments already rule out

- Confidence 0.1: PQ 0.371660 versus 0.401214 at 0.2. It can rescue 79 currently
  missed GT objects, but its additional errors outweigh the benefit overall.
- Confidence 0.3: PQ 0.387706. Simply filtering more detections also hurts.
- Cap 8: 0.399066; cap 16: 0.401214; cap 100: 0.401063. The cap is not the main bottleneck.
- Tuned final checkpoint: 0.401080, essentially the same as the selected checkpoint.
  There is no evidence yet that different checkpoint selection produces a large gain.

## Change implemented and launched

Commit `f9be86a` retains completed epochs 5/10/15/20/25/30 and compares them plus
YOLO-mAP-best and final weights using native-resolution PQ on all 197 annotator
records. Evaluation occurs after training to limit GPU memory use. It now saves
raw validation masks/confidences and selected per-record `validation.csv`.

The previous run selected the training checkpoint by combined box-and-mask mAP
against one annotator per photo, then compared only best and last by all-annotator
PQ. The change broadens checkpoint selection using the actual target metric.
It fixes an evaluation mismatch, not the underlying detection/mask errors by itself.
The larger search can overfit this validation fold; improved validation PQ does
not guarantee improved public or private test scores.

All nine existing tests passed, and the new checkpoint-retention regression test
passed with the three existing YOLO tests. Python compilation and diff checks passed.
Pushed Kaggle notebook version 2, `dharun235/solar-fil-yolo-instance`; status verified
RUNNING. Architecture, training duration, split and inference settings are unchanged.
No new competition submission was made.
