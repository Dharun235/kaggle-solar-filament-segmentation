# Lessons learned and methods tried

Updated 2026-09-16. This is the single experiment log; update it rather than adding
separate diagnostic reports. Public scores below are the last confirmed results,
not a fresh leaderboard check.

## Evaluation rules we keep

- Use month-grouped fold0, seed42: 584 training photos / 957 annotator records;
  123 validation photos / 197 annotator records; 180 test photos.
- Multiple annotations of one photo stay in the same split. Evaluate annotators
  separately; merging their objects changes the task.
- Score native-resolution instance masks with PQ and strict IoU >0.5. Resized Dice
  and YOLO mAP are useful diagnostics but are not the selection target.
- A full-data experiment still holds out validation photos. Training on those
  photos invalidates validation; it is not a fair way to improve the score.
- Compare on the same photos and annotations. Repeated tuning on one fold can
  overfit it, and local gains need not transfer to the hidden test set.
- Audit submission masks for valid RLE, nonempty masks and disjoint pixels.
  Some test photos can have no retained predictions; audit success is not accuracy.

## Model progression

| Method | Validation PQ | Confirmed public score | Interpretation |
|---|---:|---:|---|
| Early U-Net submissions | Not recorded here | 0.04 → 0.06 | Earlier baseline iterations |
| Custom U-Net, stride512, version349911418 | 0.140628; retuned 0.143089 | 0.12 | Small one-channel network trained from scratch |
| Same U-Net weights, stride256/max blending | 0.170947 | Not submitted | Better overlapping inference |
| Community U-Net++ EfficientNet-B3, original split | 0.274272 on source split | 0.23 | Annotation-level split allowed shared photos across train/validation |
| U-Net++ EfficientNet-B3, grouped fold0 | 0.330773 | Not verified here | Same training recipe, leakage-safe split; version350156671 |
| YOLOv8-S instance segmentation, grouped fold0 | **0.401214** | **0.34** | Best confirmed model so far |

The previous community U-Net++ scored 0.286305 on 101 annotator records whose
photos were absent from training. Its retrospective comparison against the original
U-Net used only 20 common unseen photos / 22 records: 0.362326 versus 0.223628.
**The 0.3623 result was a subset score, not full-fold validation.** Even unseen-photo
scoring does not undo a checkpoint-selection procedure that used a leaky split.

Grouped U-Net++ had 693 true positives, 804 false positives and 610 misses.
Pipeline changes include pretraining, resolution, training duration and selection;
these results do not isolate the effect of architecture alone.

## Original U-Net: experiments and decisions

Training used foreground-centered 512px crops, BCE + Dice and 15 epochs. Fixed
foreground x/y sampling so the coordinates come from the same foreground pixel;
made augmentations vary reproducibly by epoch. Added periodic validation PQ,
best-checkpoint restoration and validation-driven threshold/cap selection.

All inference experiments below used the same saved checkpoint and full grouped
validation fold, with no test labels.

| Experiment | Validation PQ | Decision |
|---|---:|---|
| Stride512, max blending, tuned threshold/cap | 0.143089 | Superseded |
| Stride256, max blending, tuned threshold/cap | **0.170947** | Retained |
| Stride256, mean blending | 0.126939 | Rejected |
| Stride256, center-weighted blending | 0.166466 | Rejected |
| Closing 3×3 / 5×5 / 9×9 | 0.166928 / 0.156362 / 0.126155 | Rejected |
| Best probability-guided exclusive mask growth | 0.169459 | Rejected |
| Rank by confidence × area^0.25, cap4 | 0.176467 | Exploratory; bootstrap improvement interval included zero |

Retained settings for these weights: stride256, max blending, threshold0.4,
confidence0.2, min-area80, cap4. Re-tune for new weights rather than treating them
as universal settings.

Diagnosis: 276 matches, 1,027 misses, 505 false positives. A matching candidate
existed outside the top4 for 211 misses. Of 220 partial matches, 148 predicted less
than half the annotated area. Blanket closing/growth did not repair the problem.

## Community reproduction and U-Net++

Reproduced the final branch of Talha Celik's public notebook: ImageNet-pretrained
EfficientNet-B3 encoder, SMP U-Net++, random512 crops, flip/rotate90, normalization
[-1,1], BCE + Dice, AdamW1e-3, weight decay1e-4, cosine schedule, AMP, batch8,
10 epochs. Checkpoint selection used resized512 validation Dice. Final inference
used whole2048 images, threshold0.5, min-area150, no instance cap or morphology.

Created a second notebook using our month-grouped split while retaining that
training/inference recipe. This yielded the comparable 0.330773 result above.

Community lessons: notebook titles and displayed scores do not establish which
code produced a result. The source contains several experiments; reproduce a
specific branch rather than mixing them. Public examples are not verified recipes
of the current top competitors. Morphology from a notebook must be tested locally.

References inspected:
- https://www.kaggle.com/code/realtalhacelik/solar-filaments-u-net-baseline
- https://www.kaggle.com/code/anthonytherrien/magfilo-blended-tile-u-net-for-solar-filaments
- https://www.kaggle.com/code/hdjojo/solar-filament-seg-inference

The blended-tile notebook's inspected latest branch used a pretrained EfficientViT
encoder and large whole-image inputs; we did not reproduce it. The YOLO inference
example used a private YOLOv8-L checkpoint; its training recipe was unavailable.
Our YOLOv8-S experiment is our own controlled implementation, not its exact reproduction.

## Community discussions: what we learned and what we did

These are retained findings from earlier research in this project, not a fresh
review of every thread. Kaggle's discussion pages did not expose their text when
reopened during this consolidation; the links preserve provenance for later review.

| Discussion or public analysis | Lesson | Effect on our work |
|---|---|---|
| Organizer metric clarification | Optimize instance PQ with strict IoU >0.5; pixel Dice alone can reward masks that fail instance matching | Added native-resolution PQ evaluation, threshold/cap tuning, and now periodic YOLO checkpoint selection by PQ |
| Historical leaderboard/rescoring discussion | Old reported scores may refer to an earlier scoring state | Do not treat the old Dice/rescoring issue as an explanation for current errors or compare old notebook numbers blindly |
| Inter-annotator EDA | Experts disagree about object identity, extent and presence; a prediction can be valid for one record and penalized for another | Preserve independent annotator records; quantify cross-annotator matches rather than calling every FP a hallucination |
| Small-object and mask-size EDA | Small masks are sensitive to boundary errors and resolution | Evaluate at full resolution and stratify recall by object area; YOLO's small-object weakness is now measured locally |
| Community patch/whole-image notebooks | Crops help memory and training, but inference context and blending affect filament continuity | Tested overlap, max/mean/weighted blending and whole-image U-Net++ |
| Community closing/refinement examples | Visually smoother masks are not necessarily better instance predictions | Tested closing and guided growth; rejected them when our PQ fell |
| Public YOLO inference | Direct instance segmentation produces separate masks and confidence without connected-component grouping | Implemented a separate controlled YOLOv8-S experiment; did not assume access to private training weights/recipes |
| Public-data overlap discussion | External data can contain competition test images or labels | Avoided test-label reuse; validation-driven experiments only |

Reference links:
- [Metric clarification](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/727186)
- [Organizer discussion](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/724503)
- [Historical rescoring discussion](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/733912)
- [Public-data overlap discussion](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062)
- [Community annotation/size EDA](https://www.kaggle.com/code/xiaoleilian/hardcore-eda-filament-segmentation-2026/log)

Earlier EDA notes reported inter-annotator PQ around0.389 on80 photos. That is a
sample agreement statistic, not a model performance ceiling. Our own YOLO audit
provides direct evidence:100 false-positive annotator comparisons match a different
annotator on the same photo. Neither result proves which annotation is physically correct.

We considered but did not implement the inspected EfficientViT whole-image recipe,
the private YOLOv8-L training recipe, or a Mask R-CNN plus U-Net refinement pipeline.
The last was only a public notebook lead, not an inspected or validated implementation.
Larger models, test-time augmentation, ensembles and crop-plus-whole-image YOLO
inference remain hypotheses here; none should be listed as a demonstrated gain.

## YOLO: current recipe, diagnosis and changes

Pretrained YOLOv8-S segmentation, Ultralytics8.4.152, 2048 inputs, batch1, 30epochs,
AdamW1e-3, mask_ratio2, overlap_mask=False, no mosaic/mixup/copy-paste, mild
scale/translation/brightness and flips. Annotator records are separate training
samples. Native-resolution inference uses detection confidence, NMS IoU0.7,
max_det100 and score-ordered pixel exclusion. Use boxes.conf, not boxes.cls.

Selected best.pt, confidence0.2, cap16, min-area80. Test submission: 1,209 masks,
175/180 photos with predictions, zero overlapping pixels. Full CPU replay on
123 photos / 197 records reproduced PQ **0.4012144075536765 exactly**.

| Diagnostic | Finding |
|---|---|
| Matches / misses / false positives | 741 / 562 / 444 |
| Pooled precision / recall | 62.5% / 56.9% |
| Mean IoU of matched pairs | 0.672 |
| Misses with best retained-mask IoU <0.1 | 340 / 562 |
| Misses smaller than1,000 pixels | 286 / 562 |
| Misses near the matching boundary, IoU >0.4 and ≤0.5 | 98 |
| False positives matching another annotator | 100 / 444 |
| Misses with multiple pieces covering ≥10% each | 45 |

Recall by GT area: <400px:26/105 (24.8%); 400–999:184/391 (47.1%);
1,000–2,999:346/537 (64.4%); 3,000–7,999:160/231 (69.3%); ≥8,000:25/39 (64.1%).
Counts are per annotator record; diagnostic categories overlap. Mean PQ is averaged
by record and is not the product of these pooled detection/IoU summaries.

The main target is missed objects, especially small filaments. Selected visual
examples also showed faint large misses, excess/incomplete mask extent and fragmented
curves. Of partial misses (IoU>0.1), 78 had masks over twice GT area and 33 had masks
under half GT area. Blanket dilation is not justified. Apparent false positives
include annotation disagreement, not only incorrect detections.

Already tested: confidence0.1 gives PQ0.371660, confidence0.3 gives0.387706 versus
0.401214 at0.2. Lower confidence exposes candidate matches for79 misses but hurts
PQ overall. Caps8/16/100 give0.399066/0.401214/0.401063: cap tuning is not the main
bottleneck. The final checkpoint, retuned at confidence0.3, gives0.401080.

Implemented in commit f9be86a and launched as YOLO notebook version2 on2026-09-16:
retain epochs5/10/15/20/25/30, then compare these plus mAP-best and final checkpoints
using full-fold PQ after training. Save raw validation masks/confidences and selected
per-record validation.csv. This fixes the mismatch between training's one-annotator
box/mask mAP and all-annotator PQ selection; no improvement has yet been established.
Last verified remote state was RUNNING. No automatic competition submission.

Potential future experiment, not yet tried: crop-based inference combined with
whole-image predictions to improve small-object recall, with explicit duplicate
handling and full-fold validation. Avoid changing many factors simultaneously.

## Operational lessons and artifact policy

- A successful pipeline can be followed by a failing notebook cell: an extra
  indented `test_images` line caused a post-pipeline IndentationError in an early run.
  Compile notebook cells and check subprocess return codes.
- Check existing submissions before uploading to avoid duplicates. Expired Kaggle
  OAuth may require `kaggle auth login --force`; signed download links also expire.
- Remote RUNNING status alone does not establish the current epoch or GPU health.
- Keep this one experiment document. Preserve submission CSVs locally; generated
  validation outputs, masks, logs, downloaded checkpoints and scratch experiments
  were deleted on2026-09-16 at the user's request. Source code, notebooks, required
  setup READMEs, environments and original competition data were retained.
- The remote Kaggle notebook and its stored versions are unaffected by local cleanup.
  Regenerating detailed diagnostics will require downloading/replaying weights again.

## Preserved submission files

Paths and SHA256 checksums recorded during cleanup:

- `artifacts/runs/simple-eval-fixed4/submission.csv` — `9c012d799a43151d8bb74cc745528c7a7a0c980bc0d48c994a01f40dcde1f3a8`

- `artifacts/runs/unet_kaggle/submission.csv` — `998906fd805ed978df6d8d7b02034ee8d9e4e55b2b3b3aea5461d250bff9474b`

- `output/349911418_submission.csv` — `c217dd735ef1447fb76c7c951d98a05c4e7a17aaf09e54d8c6c80c20a6d5905f`

- `output/community_exact_results/community_exact/submission_fullres.csv` — `c02e6aa72a0b1ff50b547a03aa5dfd7832fab587c65ee39274e28d7b0fcc3e5a`

- `output/submission.csv` — `801b6945f9729430f400033840df4719729be6db4f35a16ff359cf3e1df6201b`

- `output/submission_unetplusplus_grouped.csv` — `191313ff6e38fedb96709495e8f5c8840b7d8179c469c50f7c725ffc9149199a`

- `output/submission_unetplusplus_previous.csv` — `c02e6aa72a0b1ff50b547a03aa5dfd7832fab587c65ee39274e28d7b0fcc3e5a`

- `output/submission_yolo.csv` — `5fde2b858dfcf7ed12efa6074f8aaaef63e0ebebfdd961a90b01ac8881897763`

- `output/yolo_results/yolo_instance/submission.csv` — `5fde2b858dfcf7ed12efa6074f8aaaef63e0ebebfdd961a90b01ac8881897763`
