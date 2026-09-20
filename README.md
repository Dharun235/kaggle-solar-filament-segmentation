# Solar Filament Segmentation

Maintained solution for the Kaggle Solar Filament Segmentation Challenge 2026. The repository contains pretrained YOLOv8-L instance segmentation with COCO-consistent polygon-mask rasterization.

## Result

The previous YOLOv8-S COCO-mask pipeline reached legacy local validation PQ **0.41944** and public score **0.36**. The current pipeline uses community-style YOLOv8-L inference and organizer-compatible PQ; validation must be regenerated. YOLO11m-seg was rejected after public score **0.35**.

## Run

The Kaggle notebook is [notebooks/yolo_instance/yolo-instance.ipynb](notebooks/yolo_instance/yolo-instance.ipynb). It installs the pinned Ultralytics version, verifies CUDA, prepares the grouped fold, trains at 2048px, selects checkpoints using full-fold PQ, creates `submission.csv`, and audits disjoint masks.

```bash
kaggle kernels push -p notebooks/yolo_instance
```

The local entry point is `models/yolo_instance.py`. It requires the competition dataset and a CUDA GPU. Use `--data-pipeline coco`; this is the only maintained data pipeline.

## Repository contents

- `models/yolo_instance.py`: training, validation, checkpoint selection, test inference and RLE export.
- `models/yolo_data_pipeline.py`: COCO-consistent instance-mask formatting.
- `scripts/pipeline_lib.py`: shared physical-image helpers.
- `scripts/postprocess.py`: PQ scoring and exclusive mask postprocessing.
- `scripts/audit_submission.py`: submission validation.
- `notebooks/yolo_instance/`: maintained Kaggle GPU notebook.
- `reports/lessons_and_experiments.md`: complete history, community lessons, tested methods and measured results.

Generated outputs, checkpoints, local environments and competition submissions are intentionally not tracked.
