#!/usr/bin/env python3
"""Fast, training-free Mac baseline.

Finds locally dark connected regions in grayscale H-alpha images. It is intentionally
simple: useful smoke baseline and model contract reference, not final leaderboard model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def detect(path: Path, output_dir: Path, delta: float, min_area: int, max_candidates: int) -> list[dict]:
    # Work at half resolution for CPU speed, restore masks to Kaggle resolution.
    original_size = (2048, 2048)
    small_size = (1024, 1024)
    gray = np.asarray(Image.open(path).convert("L").resize(small_size, Image.Resampling.BILINEAR), dtype=np.float32)
    # Uniform filter gives useful local background at a fraction of Gaussian cost on CPU.
    background = ndimage.uniform_filter(gray, size=19, mode="nearest")
    darkness = background - gray
    # Filaments are locally dark. Morphology joins broken thin pixels.
    binary = darkness >= delta
    binary = ndimage.binary_closing(binary, structure=np.ones((3, 3)), iterations=2)
    binary = ndimage.binary_opening(binary, structure=np.ones((3, 3)), iterations=1)
    labels, count = ndimage.label(binary, structure=np.ones((3, 3)))
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for label in range(1, count + 1):
        mask = labels == label
        area = int(mask.sum())
        if area < min_area:
            continue
        # Confidence reflects local contrast; postprocess calibrates threshold later.
        score = float(np.clip(darkness[mask].mean() / 40.0, 0.01, 0.99))
        candidates.append((score, area, mask))
    instances = []
    for score, area, mask in sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)[:max_candidates]:
        mask_path = output_dir / f"{path.stem}_{len(instances):03d}.png"
        full_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(original_size, Image.Resampling.NEAREST)
        Image.fromarray(np.asarray(full_mask), mode="L").save(mask_path)
        instances.append({"score": score, "mask_path": str(mask_path)})
    return instances


def run(manifest: Path, output: Path, masks: Path, delta: float, min_area: int, max_candidates: int):
    import json
    output.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open() as src, output.open("w") as dst:
        for line in src:
            row = json.loads(line)
            instances = detect(Path(row["path"]), masks / row["physical_id"], delta, min_area, max_candidates)
            dst.write(json.dumps({"image_id": row["physical_id"], "instances": instances}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--raw-val", type=Path, required=True)
    ap.add_argument("--raw-test", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--delta", type=float, default=10.0)
    ap.add_argument("--min-area", type=int, default=32)
    ap.add_argument("--max-candidates", type=int, default=30)
    args = ap.parse_args()
    run(args.val, args.raw_val, args.run_dir / "masks/val", args.delta, args.min_area, args.max_candidates)
    run(args.test, args.raw_test, args.run_dir / "masks/test", args.delta, args.min_area, args.max_candidates)
    print(f"wrote {args.raw_val} and {args.raw_test}")


if __name__ == "__main__":
    main()
