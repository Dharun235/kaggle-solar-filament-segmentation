from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

try:
    from .pipeline_lib import encode_mask, jsonl_read, load_json, physical_stem, read_mask, polygons_to_mask
except ImportError:
    from pipeline_lib import encode_mask, jsonl_read, load_json, physical_stem, read_mask, polygons_to_mask
from pycocotools import mask as mask_utils


def greedy_exclusive(items, confidence, max_instances, min_area):
    kept = []
    occupied = np.zeros((2048, 2048), dtype=bool)
    for item in sorted(items, key=lambda x: float(x.get("score", 0)), reverse=True):
        score = float(item.get("score", 0))
        if score < confidence or len(kept) >= max_instances:
            continue
        mask = item.get("_mask")
        if mask is None:
            mask = read_mask(item["mask_path"] if "mask_path" in item else item["mask"])
        if mask.shape != occupied.shape:
            raise ValueError(f"mask shape {mask.shape}; expected (2048, 2048)")
        mask &= ~occupied
        if int(mask.sum()) < min_area:
            continue
        kept.append((score, mask))
        occupied |= mask
    return kept


def gt_by_stem(path):
    data = load_json(path)
    anns_by_id = {}
    for ann in data["annotations"]:
        anns_by_id.setdefault(ann["image_id"], []).append(ann)
    out = {}
    for im in data["images"]:
        out.setdefault(physical_stem(im["file_name"]), []).append(anns_by_id.get(im["id"], []))
    return out


def expected_stems(path):
    return {physical_stem(row["physical_id"]) for row in jsonl_read(path)}


def pq_score(gt, pred, threshold=0.5):
    gt_rles = [mask_utils.encode(np.asarray(m, dtype=np.uint8, order="F")[:, :, None])[0] for m in gt]
    pred_rles = [mask_utils.encode(np.asarray(m, dtype=np.uint8, order="F")[:, :, None])[0] for m in pred]
    return pq_score_rles(gt_rles, pred_rles, threshold)


def pq_score_rles(gt_rles, pred_rles, threshold=0.5):
    # pycocotools signature is iou(dt, gt, iscrowd); transpose to [gt, pred].
    ious = mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)).T if gt_rles and pred_rles else np.empty((len(gt_rles), len(pred_rles)))
    pairs = []
    for i in range(len(gt_rles)):
        for j in range(len(pred_rles)):
            if ious[i, j] > 0:
                pairs.append((float(ious[i, j]), i, j))
    used_g, used_p, tp_iou = set(), set(), []
    for iou, i, j in sorted(pairs, reverse=True):
        # Official scorer uses strict IoU > 0.5 matching.
        if iou > threshold and i not in used_g and j not in used_p:
            used_g.add(i); used_p.add(j); tp_iou.append(iou)
    return (sum(tp_iou) / (len(tp_iou) + 0.5 * (len(pred_rles) - len(used_p)) + 0.5 * (len(gt_rles) - len(used_g)))
            if gt_rles or pred_rles else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--confidence", type=float, default=0.2)
    ap.add_argument("--confidence-grid", type=float, nargs="*")
    ap.add_argument("--max-instances", type=int, default=10)
    ap.add_argument("--max-instances-grid", type=int, nargs="*",
                    help="candidate instance caps to tune on validation PQ")
    ap.add_argument("--min-area", type=int, default=16)
    ap.add_argument("--ground-truth", type=Path)
    ap.add_argument("--selected-confidence-file", type=Path)
    ap.add_argument("--metrics-output", type=Path,
                    help="CSV file for validation PQ results across the parameter grid")
    ap.add_argument("--expected-manifest", type=Path)
    args = ap.parse_args()
    records = list(jsonl_read(args.predictions))
    if not records:
        raise SystemExit("prediction JSONL empty")
    pred_stems = [physical_stem(rec["image_id"]) for rec in records]
    duplicates = sorted({s for s in pred_stems if pred_stems.count(s) > 1})
    if duplicates:
        raise SystemExit(f"duplicate prediction records for physical images={len(duplicates)}")
    pred_stems = set(pred_stems)
    if args.expected_manifest:
        missing = expected_stems(args.expected_manifest) - pred_stems
        if missing:
            raise SystemExit(f"missing prediction records={len(missing)}")
    gt = gt_by_stem(args.ground_truth) if args.ground_truth else None
    confidence_grid = args.confidence_grid or [args.confidence]
    max_instances_grid = args.max_instances_grid or [args.max_instances]
    best = None
    metrics = []
    for conf in confidence_grid:
        for max_instances in max_instances_grid:
            if max_instances < 0:
                raise SystemExit("max instances must be non-negative")
            scores = []
            for rec in tqdm(records, desc=f"PQ threshold {conf:.2f}, cap {max_instances}", unit="image", leave=False):
                kept = greedy_exclusive(rec.get("instances", []), conf, max_instances, args.min_area)
                stem = physical_stem(rec["image_id"])
                if gt is not None and stem in gt:
                    pm = [m for _, m in kept]
                    # Score each independent annotator record separately; never merge labels.
                    scores.extend(pq_score([polygons_to_mask(a["segmentation"]) for a in anns], pm)
                                  for anns in gt[stem])
            mean = float(np.mean(scores)) if scores else None
            metrics.append({"confidence": conf, "max_instances": max_instances, "panoptic_quality": mean})
            if best is None or (mean is not None and mean > best[0]):
                best = (mean, conf, max_instances)
    conf = best[1] if gt is not None else args.confidence
    max_instances = best[2] if gt is not None else args.max_instances
    if args.selected_confidence_file:
        args.selected_confidence_file.parent.mkdir(parents=True, exist_ok=True)
        args.selected_confidence_file.write_text(json.dumps({
            "confidence": conf, "max_instances": max_instances,
            "local_pq": best[0], "metric": "panoptic_quality", "iou_match": ">0.5",
        }) + "\n")
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["confidence", "max_instances", "panoptic_quality"])
            writer.writeheader()
            writer.writerows(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["filament_id", "segmentation_rle"])
        rows = 0
        for rec in tqdm(records, desc="write RLE submission", unit="image"):
            kept = greedy_exclusive(rec.get("instances", []), conf, max_instances, args.min_area)
            for idx, (_, mask) in enumerate(kept, 1):
                writer.writerow([f"{physical_stem(rec['image_id'])}_{idx}", encode_mask(mask)])
                rows += 1
    print(f"wrote={args.output} rows={rows} confidence={conf:.3f} max_instances={max_instances} local_pq={best[0]}")


if __name__ == "__main__":
    main()
