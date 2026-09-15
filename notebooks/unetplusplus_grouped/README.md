# U-Net++ grouped-fold validation

Retrains the successful community U-Net++/EfficientNet-B3 recipe on the original
month-grouped fold0 (seed42, five folds), so all123 validation photographs and197
annotation records are unseen in training. 584 training photographs,957 annotator samples.

Architecture, ImageNet initialization, random512 crops, augmentations, normalization,
BCE+Dice, AdamW, 10epochs, resized-Dice checkpoint selection and full2048 inference
are unchanged from ../community_exact. Uses threshold0.5, min-area150 and no instance
cap. This compares full pipelines, not architecture in isolation. Dependency versions
are recorded; training randomness is not fully seeded in the original recipe.

The notebook checks the validation-photograph hash against our original fold.
Outputs under /kaggle/working/unetplusplus_grouped:
- best_solar_unetplusplus_patch.pth
- validation_summary.json: PQ and difference from0.1709468443
- validation.csv: metrics per annotation record
- validation_preview.csv: predicted validation instance RLEs
- submission.csv (also submission_fullres.csv): test predictions, not auto-submitted
- split.json and environment.txt

Source and attribution: Talha Celik,
https://www.kaggle.com/code/realtalhacelik/solar-filaments-u-net-baseline
Apache-2.0, see ../community_exact/LICENSE. Changed split and added evaluation/export.

Launch: kaggle kernels push -p notebooks/unetplusplus_grouped
