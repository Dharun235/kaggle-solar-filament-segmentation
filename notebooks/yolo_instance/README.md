# YOLO instance segmentation experiment

Pretrained YOLOv8-S segmentation, 2048px inputs, batch1, 30 epochs, AdamW,
full-resolution output masks, training mask_ratio2. No mosaic or copy-paste.
Ultralytics version: 8.4.152. Model size is chosen for T4 memory at native input resolution.

Uses the same month-grouped fold0 as the baseline:584 training photos (957 separate
annotator samples),123 validation photos (197 annotator records). Training retains
individual filament polygons as one class. Different annotator records are separate
samples rather than duplicate objects in one image. YOLO's built-in mAP uses one
annotation record per validation photo; final PQ scores every annotator record.

Retains checkpoints after epochs 5, 10, 15, 20, 25 and 30. After training, compares
all retained checkpoints plus best-by-YOLO-mAP and final checkpoints using native-resolution PQ with
confidence .1/.2/.3/.4/.5 and caps4/8/16/100. Removes overlaps in descending detection
confidence order and drops masks smaller than80 pixels after exclusion. Uses boxes.conf,
not boxes.cls, for scores. The baseline cap4 is only one candidate, not imposed on YOLO.

Checkpoint selection uses all 197 annotator records on the same 123 held-out photos.
It runs after training to avoid loading an extra model alongside the training model.
`validation_predictions/<checkpoint>.jsonl` saves pre-exclusion instance masks as
COCO RLE and detection confidence for every validation photo, including empty predictions.
`validation.csv` records the selected model's PQ for each annotator record.
`validation_grid.json` is saved after each evaluated checkpoint. The expanded
checkpoint search adds validation time and does not constitute an independent test.

The final CSV is checked during generation for nonempty masks and disjoint pixels.
No training performance or GPU memory claim is made until the Kaggle run completes.
No automatic competition submission. Run status: /kaggle/working/yolo_instance/status.json.

Launch: `kaggle kernels push -p notebooks/yolo_instance`

References: https://docs.ultralytics.com/tasks/segment/
https://docs.ultralytics.com/datasets/segment/
