from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .pipeline_lib import decode_rle
except ImportError:
    from pipeline_lib import decode_rle


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--submission", type=Path, required=True); ap.add_argument("--test-images", type=Path, required=True); ap.add_argument("--require-all-images", action="store_true"); args = ap.parse_args()
    expected = {p.stem for p in args.test_images.glob("*.jpeg")}; seen = defaultdict(list); ids=set(); errors=[]
    with args.submission.open(newline="") as f:
        rows=list(csv.DictReader(f))
    if set(rows[0]) != {"filament_id","segmentation_rle"} if rows else True: errors.append("columns must be filament_id,segmentation_rle")
    for row in rows:
        fid=row["filament_id"]
        if fid in ids: errors.append(f"duplicate filament_id={fid}")
        ids.add(fid)
        match=re.fullmatch(r"(.+)_([1-9][0-9]*)", fid)
        if not match: errors.append(f"invalid filament_id={fid}"); continue
        stem=match.group(1)
        if stem not in expected: errors.append(f"unknown image_id={stem}")
        try: mask=decode_rle(row["segmentation_rle"])
        except Exception as e: errors.append(f"{fid}: invalid RLE: {e}"); continue
        if not mask.any(): errors.append(f"{fid}: empty mask")
        seen[stem].append(mask)
    missing=expected-set(seen)
    if args.require_all_images and missing: errors.append(f"missing images={len(missing)}")
    overlap=0
    for stem,masks in seen.items():
        occ=np.zeros((2048,2048),bool)
        for m in masks:
            overlap += int((occ&m).sum()); occ |= m
    if overlap: errors.append(f"overlapping pixels={overlap}")
    print(f"rows={len(rows)} images={len(seen)}/{len(expected)} overlap_pixels={overlap}")
    if errors:
        for e in errors: print("ERROR",e)
        raise SystemExit(1)
    print("submission=ok")


if __name__ == "__main__": main()
