# Validation experiments — 2026-09-15

All experiments reuse checkpoint SHA256 `8bdb8279119d8e08200274d9c224723c2f4262684662a5263759a648739743b6`
from Kaggle run 349911418. Validation covers 123 physical images and 197 independent
annotator records. PQ uses strict IoU > 0.5 matching; no test labels were used.

| Experiment | Best validation PQ |
|---|---:|
| Stride 512, max blending, tuned threshold/cap | 0.143089 |
| Stride 256, max blending, tuned threshold/cap | **0.170947** |
| Stride 256, mean blending, tuned threshold, cap4 | 0.126939 |
| Stride 256, center-weighted blending, tuned threshold, cap4 | 0.166466 |
| Closing 3x3, baseline settings | 0.166928 |
| Closing 5x5, baseline settings | 0.156362 |
| Closing 9x9, baseline settings | 0.126155 |
| Best tested exclusive probability-guided growth | 0.169459 |

Retained settings: stride256, max blending, threshold0.4, confidence0.2,
min-area80, cap4. These settings are specific to this checkpoint; new training runs
still select checkpoints and tune postprocessing on validation.

Size-aware ranking (confidence multiplied by area^0.25, cap4) scored 0.176467,
but its paired image-bootstrap improvement interval included zero (-0.00436 to
0.01547). It remains exploratory and is not enabled. These are repeated comparisons
on one validation fold, not independent estimates of hidden-test performance.

The baseline matched 276 annotations, missed 1027, and produced 505 unmatched
predictions across annotator comparisons. 211 missed annotations had a matching
candidate outside the retained top4. Among 220 partial matches, 148 predicted less
than half the annotated area. Closing9x9 reduced matches to200 and increased false
positives to578. Generic expansion did not repair the missing filament sections.

Community morphology reference:
https://www.kaggle.com/code/realtalhacelik/solar-filaments-u-net-baseline
The downloaded notebook uses square closing (9x9, later5x5); it changes several
other components simultaneously and does not isolate a current-PQ benefit.
Our closing experiments adapt this idea while retaining our inference and ranking.

Raw experimental maps, checkpoints, CSVs, and diagnostic images remain under ignored
`output/`; competition data and predictions are not published in this repository.
