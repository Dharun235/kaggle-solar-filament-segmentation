# Community final-branch reproduction

Source: https://www.kaggle.com/code/realtalhacelik/solar-filaments-u-net-baseline
Author: Talha Celik. Source license: Apache-2.0; see LICENSE.

The source contains several experiments rather than one configuration. This notebook
retains its final U-Net++ branch: EfficientNet-B3 ImageNet encoder, 512px random crops,
horizontal/vertical/90-degree augmentation, normalization to [-1,1], BCE + SMP Dice,
AdamW(lr=1e-3, weight_decay=1e-4), cosine schedule T_max10, AMP, batch8, 10 epochs,
and best checkpoint chosen by Dice on resized 512px validation images. Inference is
full-image 2048px, threshold0.5, minimum area150, without an instance cap or morphology.
Source training cell28 and inference cell58 are copied unchanged (zero-based indices).

This is a reproduction of that branch, not a bitwise reproducible training run:
the original dependencies are unpinned and its training RNG is not fully seeded.
Installed packages and the actual split are saved with each run.

Added only execution setup, isolated output paths, and PQ reporting. The source's
annotation-ID split can place the same photograph in both sets; it is intentionally
retained for fidelity. Report `unseen_photo_pq` separately. Neither score is directly
comparable with our original grouped validation fold. Prior model experiments and
plots are omitted. No automatic submission is performed.

Kaggle push: `kaggle kernels push -p notebooks/community_exact`
Outputs are under `/kaggle/working/community_exact`.

Source notebook SHA256: 7903fbe59116b38b44cbe39248f31de5c2d51f066b217d1d21169c10cd375d50
