#!/usr/bin/env python3
"""Organizer-protocol local PQ evaluator.

Scores each physical prediction against every independent annotator record, using strict
IoU > 0.5 one-to-one matching, then averages record scores. This mirrors the organizer's
self-evaluation protocol described in the competition notebook/discussion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .pipeline_lib import jsonl_read, load_json, physical_stem, polygons_to_mask, read_mask
    from .postprocess import greedy_exclusive, gt_by_stem, pq_score
except ImportError:
    from pipeline_lib import jsonl_read, load_json, physical_stem, polygons_to_mask, read_mask
    from postprocess import greedy_exclusive, gt_by_stem, pq_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--confidence", type=float, default=0.20)
    ap.add_argument("--max-instances", type=int, default=10)
    ap.add_argument("--min-area", type=int, default=16)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    gt = gt_by_stem(args.ground_truth)
    scores = []
    missing = []
    for rec in jsonl_read(args.predictions):
        stem = physical_stem(rec["image_id"])
        if stem not in gt:
            missing.append(stem)
            continue
        # Load only masks used by postprocess; avoid duplicate full-image reads.
        for item in rec.get("instances", []):
            item["_mask"] = read_mask(item["mask_path"] if "mask_path" in item else item["mask"])
        pred = [m for _, m in greedy_exclusive(rec.get("instances", []), args.confidence,
                                                args.max_instances, args.min_area)]
        for anns in gt[stem]:
            scores.append(pq_score([polygons_to_mask(a["segmentation"]) for a in anns], pred))
    result = {"official_pq": float(np.mean(scores)) if scores else None,
              "annotation_record_scores": len(scores), "missing_prediction_images": sorted(set(missing)),
              "confidence": args.confidence, "iou_match_threshold": ">0.5"}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
