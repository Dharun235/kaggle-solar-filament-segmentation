from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm

try:
    from .pipeline_lib import load_json, physical_stem, polygons_to_mask, write_mask, jsonl_write
except ImportError:
    from pipeline_lib import load_json, physical_stem, polygons_to_mask, write_mask, jsonl_write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("MAGFiLO_1.0_Kaggle_2026"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode-masks", action="store_true")
    args = ap.parse_args()
    train = args.root / "train"
    test_images = args.root / "test/test_images"
    ann_path = train / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    print("[1/3] loading COCO annotations", flush=True)
    data = load_json(ann_path)
    image_dir = train / "train_images"
    files = {p.stem: p for p in image_dir.glob("*.jpeg")}
    test_files = sorted(test_images.glob("*.jpeg"))
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann)

    # Group by physical JPEG, then month. Prevent duplicate annotator leakage.
    groups = defaultdict(list)
    for im in tqdm(data["images"], desc="COCO image records", unit="image"):
        stem = physical_stem(im["file_name"])
        month = im.get("date_captured", stem[:6])[:7]
        groups[month].append(im)
    months = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(months)
    month_folds = {month: i % args.folds for i, month in enumerate(months)}
    out = Path("artifacts/manifests")
    out.mkdir(parents=True, exist_ok=True)
    print("[2/3] building grouped manifests", flush=True)
    image_rows = []
    for im in data["images"]:
        stem = physical_stem(im["file_name"])
        month = im.get("date_captured", stem[:6])[:7]
        image_rows.append({"image_id": im["id"], "physical_id": stem, "file_name": im["file_name"],
                           "path": str(files.get(stem, image_dir / im["file_name"])),
                           "width": im["width"], "height": im["height"], "month": month,
                           "fold": month_folds[month], "annotations": by_image.get(im["id"], [])})
    jsonl_write(out / "train.jsonl", image_rows)
    jsonl_write(out / "test.jsonl", [{"image_id": p.stem, "physical_id": p.stem, "file_name": p.name,
                                      "path": str(p), "width": 2048, "height": 2048} for p in test_files])
    for fold in range(args.folds):
        jsonl_write(out / f"train_fold{fold}.jsonl", [r for r in image_rows if r["fold"] != fold])
        jsonl_write(out / f"val_fold{fold}.jsonl", [r for r in image_rows if r["fold"] == fold])
    if args.decode_masks:
        print("[3/3] decoding masks", flush=True)
        mask_root = Path("artifacts/masks")
        for row in image_rows:
            for i, ann in enumerate(row["annotations"]):
                mask = polygons_to_mask(ann["segmentation"], row["height"], row["width"])
                write_mask(mask_root / row["image_id"] / f"{i:04d}.png", mask)
    print(f"train records={len(image_rows)} annotations={len(data['annotations'])} test={len(test_files)} folds={args.folds}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
