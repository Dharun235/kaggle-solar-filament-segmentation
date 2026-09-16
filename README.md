# Solar Filament Segmentation

Maintained solution for the Kaggle Solar Filament Segmentation Challenge 2026. The repository now contains one production pipeline: pretrained YOLOv8-S instance segmentation with COCO-consistent polygon-mask rasterization.

## Result

The maintained COCO-mask pipeline reached validation PQ **0.41944** on the leakage-safe month-grouped fold (584 training photos, 123 validation photos, 197 annotator records). The previous YOLO preprocessing scored 0.40121; disk-intensity normalization scored 0.38818 and was removed. The best confirmed public score before the COCO submission was 0.34; the COCO submission is pending.

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
- `tests/`: metric, postprocessing and data-pipeline tests.
- `reports/lessons_and_experiments.md`: complete history, community lessons, tested methods and measured results.

Generated outputs, checkpoints, local environments and competition submissions are intentionally not tracked.
