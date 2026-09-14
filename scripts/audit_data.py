from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

try:
    from .pipeline_lib import load_json, polygons_to_mask
except ImportError:
    from pipeline_lib import load_json, polygons_to_mask


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=Path("MAGFiLO_1.0_Kaggle_2026")); ap.add_argument("--decode-all", action="store_true"); args = ap.parse_args()
    train = args.root / "train"; test = args.root / "test/test_images"
    d = load_json(train / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    train_files = {p.stem for p in (train / "train_images").glob("*.jpeg")}; test_files = {p.stem for p in test.glob("*.jpeg")}
    refs = {Path(i["file_name"]).stem for i in d["images"]}; ids = {a["image_id"] for a in d["annotations"]}
    missing = refs - train_files
    print(f"train_jpegs={len(train_files)} test_jpegs={len(test_files)} coco_records={len(d['images'])} annotations={len(d['annotations'])}")
    print(f"missing_train_files={len(missing)} categories={Counter(a['category_id'] for a in d['annotations'])}")
    if refs & test_files: print(f"WARNING train/test physical overlap={len(refs & test_files)}")
    if args.decode_all:
        for a in d["annotations"]: polygons_to_mask(a["segmentation"])
        print("polygon_decode=ok")


if __name__ == "__main__": main()
