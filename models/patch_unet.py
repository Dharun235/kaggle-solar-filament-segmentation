#!/usr/bin/env python3
"""Small patch U-Net baseline inspired by community pipeline.

Train on filament-centered 512px patches. Infer with overlapping 512px tiles and max
stitching. Emits model-neutral JSONL consumed by postprocess.py.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Allow `python models/patch_unet.py ...` from controller.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pipeline_lib import polygons_to_mask, physical_stem
from scripts.postprocess import gt_by_stem, pq_score_rles
from pycocotools import mask as mask_utils


def normalize(image):
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [1, 99])
    return np.clip((image - lo) / max(hi - lo, 1.0), 0, 1)


def load_row(row):
    image = normalize(np.asarray(Image.open(row["path"]).convert("L")))
    mask = np.zeros((row["height"], row["width"]), dtype=bool)
    for ann in row.get("annotations", []):
        mask |= polygons_to_mask(ann["segmentation"], row["height"], row["width"])
    return image, mask


class PatchDataset(Dataset):
    def __init__(self, manifest, patch=512, samples_per_image=3, seed=42, max_rows=None):
        self.rows = [json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()]
        if max_rows:
            self.rows = self.rows[:max_rows]
        self.patch, self.seed = patch, seed
        self.epoch = 0
        self.indices = [(i, j) for i in range(len(self.rows)) for j in range(samples_per_image)]
        self._cached_row = None
        self._cached_data = None

    def __len__(self): return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        row_i, repeat = self.indices[index]
        # Two samples per image are adjacent. Avoid rereading JPEG and rasterizing
        # every polygon twice; this was starving CUDA during training.
        if self._cached_row != row_i:
            self._cached_row = row_i
            self._cached_data = load_row(self.rows[row_i])
        image, mask = self._cached_data
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index, repeat]))
        ys, xs = np.nonzero(mask)
        if len(xs) and rng.random() < 0.75:
            pixel = rng.integers(len(xs))
            cx, cy = int(xs[pixel]), int(ys[pixel])
        else:
            cx, cy = int(rng.integers(0, 2048)), int(rng.integers(0, 2048))
        x0 = min(max(cx - self.patch // 2, 0), 2048 - self.patch)
        y0 = min(max(cy - self.patch // 2, 0), 2048 - self.patch)
        image, mask = image[y0:y0+self.patch, x0:x0+self.patch], mask[y0:y0+self.patch, x0:x0+self.patch]
        if rng.random() < 0.5: image, mask = image[:, ::-1], mask[:, ::-1]
        if rng.random() < 0.5: image, mask = image[::-1], mask[::-1]
        return torch.from_numpy(np.ascontiguousarray(image[None])).float(), torch.from_numpy(np.ascontiguousarray(mask[None])).float()


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(a,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.Conv2d(b,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU())
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__(); self.e1=Block(1,16); self.e2=Block(16,32); self.e3=Block(32,64); self.pool=nn.MaxPool2d(2); self.u2=nn.ConvTranspose2d(64,32,2,2); self.d2=Block(64,32); self.u1=nn.ConvTranspose2d(32,16,2,2); self.d1=Block(32,16); self.out=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); x=self.u2(c); x=self.d2(torch.cat([x,b],1)); x=self.u1(x); x=self.d1(torch.cat([x,a],1)); return self.out(x)


def loss_fn(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits); inter = (p * target).sum((1,2,3)); dice = (2*inter + 1) / (p.sum((1,2,3)) + target.sum((1,2,3)) + 1)
    return bce + (1 - dice.mean())


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def predict_image(model, path, device, stride, infer_batch):
    image = normalize(np.asarray(Image.open(path).convert("L")))
    logits = np.full((2048,2048), -20., np.float32)
    coords = [(y,x) for y in range(0, 2048-512+1, stride) for x in range(0, 2048-512+1, stride)]
    if coords[-1][0] != 1536: coords += [(1536,x) for x in range(0,1536+1,stride) if x != 1536] + [(y,1536) for y in range(0,1536,stride)] + [(1536,1536)]
    for start in range(0, len(coords), infer_batch):
        batch_coords = coords[start:start + infer_batch]
        batch = np.stack([image[y:y+512, x:x+512] for y, x in batch_coords])[:, None]
        probs = torch.sigmoid(model(torch.from_numpy(batch).float().to(device))).cpu().numpy()[:, 0]
        for (y, x), prob in zip(batch_coords, probs):
            logits[y:y+512,x:x+512]=np.maximum(logits[y:y+512,x:x+512], prob)
    return logits


def extract_candidates(prob, threshold, min_area, max_candidates):
    labels, _ = ndimage.label(prob >= threshold, structure=np.ones((3, 3)))
    # Rank components without allocating one full-frame mask for every speck.
    areas = np.bincount(labels.ravel())
    sums = np.bincount(labels.ravel(), weights=prob.ravel())
    candidates = [(float(np.clip(sums[i] / areas[i], .01, .99)), int(areas[i]), i)
                  for i in range(1, len(areas)) if areas[i] >= min_area]
    return labels, sorted(candidates, reverse=True)[:max_candidates]


def validation_ground_truth(path, rows):
    annotations = gt_by_stem(path)
    result = {}
    for row in rows:
        stem = physical_stem(row["physical_id"])
        if stem not in annotations:
            raise ValueError(f"Validation image missing from ground truth: {stem}")
        result[stem] = [[mask_utils.merge(mask_utils.frPyObjects(
            ann["segmentation"], row["height"], row["width"])) for ann in record]
            for record in annotations[stem]]
    return result


def validate(model, rows, gt, device, args, thresholds, probability_dir):
    probability_dir.mkdir(parents=True, exist_ok=True)
    scores = {threshold: [] for threshold in thresholds}
    model.eval()
    for row in tqdm(rows, desc="checkpoint validation PQ", unit="image"):
        prob = predict_image(model, row["path"], device, args.stride, args.infer_batch)
        np.save(probability_dir / f"{row['physical_id']}.npy", prob)
        for threshold in thresholds:
            labels, candidates = extract_candidates(prob, threshold, args.min_area, args.max_candidates)
            pred = [mask_utils.encode(np.asfortranarray(labels == label, dtype=np.uint8))
                    for score, _, label in candidates if score >= args.selection_confidence]
            pred = pred[:args.selection_max_instances]
            scores[threshold].extend(pq_score_rles(record, pred)
                                     for record in gt[physical_stem(row["physical_id"])])
    return [{"threshold": threshold, "pq": float(np.mean(values))}
            for threshold, values in scores.items()]


def write_predictions(model, manifest, output, mask_root, device, threshold, min_area,
                      max_candidates, stride, infer_batch, probability_dir=None):
    rows = [json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w") as dst:
        for row in tqdm(rows, desc=f"inference {manifest.stem}", unit="image"):
            prob = (np.load(probability_dir / f"{row['physical_id']}.npy") if probability_dir else
                    predict_image(model, row["path"], device, stride, infer_batch))
            labels, candidates = extract_candidates(prob, threshold, min_area, max_candidates)
            instances = []
            out_dir = Path(mask_root) / row["physical_id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            for score, area, label in candidates:
                p = out_dir / f"{len(instances):03d}.png"
                Image.fromarray(((labels == label)*255).astype(np.uint8)).save(p)
                instances.append({"score": score, "mask_path": str(p)})
            dst.write(json.dumps({"image_id": row["physical_id"], "instances": instances})+"\n")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train",type=Path,required=True); ap.add_argument("--val",type=Path,required=True); ap.add_argument("--test",type=Path,required=True); ap.add_argument("--raw-val",type=Path,required=True); ap.add_argument("--raw-test",type=Path,required=True); ap.add_argument("--run-dir",type=Path,required=True); ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--batch-size",type=int,default=4); ap.add_argument("--samples-per-image",type=int,default=3); ap.add_argument("--max-train-rows",type=int); ap.add_argument("--device",default="auto"); ap.add_argument("--threshold",type=float,default=.35); ap.add_argument("--min-area",type=int,default=80); ap.add_argument("--max-candidates",type=int,default=20); ap.add_argument("--stride",type=int,default=256); ap.add_argument("--infer-batch",type=int,default=8)
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--threshold-grid", type=float, nargs="+", default=[.2, .3, .35, .4, .5, .6])
    ap.add_argument("--validate-every", type=int, default=3)
    ap.add_argument("--selection-confidence", type=float, default=.2)
    ap.add_argument("--selection-max-instances", type=int, default=10)
    args = ap.parse_args()
    if args.epochs < 1 or args.validate_every < 1 or not 1 <= args.stride <= 512:
        ap.error("epochs and validate-every must be positive; stride must be 1..512")
    if args.min_area < 1 or args.max_candidates < 1 or args.selection_max_instances < 1:
        ap.error("area and instance limits must be positive")
    thresholds = sorted(set(args.threshold_grid + [args.threshold]))
    if any(not 0 < t < 1 for t in thresholds):
        ap.error("pixel thresholds must be between 0 and 1")
    val_rows = [json.loads(x) for x in args.val.read_text().splitlines() if x.strip()]
    if not val_rows:
        ap.error("validation manifest must not be empty")
    gt = validation_ground_truth(args.ground_truth, val_rows)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.run_dir / "patch_unet.pt"
    current_probs = args.run_dir / "probabilities/current_val"
    best_probs = args.run_dir / "probabilities/val"
    best = None
    history = []
    random.seed(42); np.random.seed(42); torch.manual_seed(42); device=device_for(args.device); print(f"device={device}")
    ds=PatchDataset(args.train,samples_per_image=args.samples_per_image,max_rows=args.max_train_rows); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=(device.type == "cuda"))
    model=UNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    for epoch in range(args.epochs):
        ds.set_epoch(epoch)
        model.train(); total=0.
        progress = tqdm(loader, desc=f"train epoch {epoch + 1}/{args.epochs}", unit="batch")
        for x,y in progress:
            opt.zero_grad(); z=loss_fn(model(x.to(device, non_blocking=True)),y.to(device, non_blocking=True)); z.backward(); opt.step(); total += z.detach().item()
            progress.set_postfix(loss=f"{z.detach().item():.4f}")
        print(f"epoch={epoch+1}/{args.epochs} loss={total/max(len(loader),1):.4f}",flush=True)
        if (epoch + 1) % args.validate_every == 0 or epoch + 1 == args.epochs:
            metrics = validate(model, val_rows, gt, device, args, thresholds, current_probs)
            history.extend(dict(epoch=epoch + 1, **metric) for metric in metrics)
            selected = max(metrics, key=lambda metric: metric["pq"])
            if best is None or selected["pq"] > best["pq"]:
                best = dict(epoch=epoch + 1, **selected,
                            confidence=args.selection_confidence,
                            max_instances=args.selection_max_instances)
                torch.save(model.state_dict(), ckpt)
                shutil.copytree(current_probs, best_probs, dirs_exist_ok=True)
                (args.run_dir / "selected_model.json").write_text(json.dumps(best, indent=2) + "\n")
            (args.run_dir / "checkpoint_metrics.json").write_text(json.dumps(history, indent=2) + "\n")
            print(f"validation epoch={epoch+1} pq={selected['pq']:.5f} "
                  f"threshold={selected['threshold']} best_epoch={best['epoch']}", flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    write_predictions(model, args.val, args.raw_val, args.run_dir / "masks/val", device,
                      best["threshold"], args.min_area, args.max_candidates,
                      args.stride, args.infer_batch, probability_dir=best_probs)
    write_predictions(model, args.test, args.raw_test, args.run_dir / "masks/test", device,
                      best["threshold"], args.min_area, args.max_candidates,
                      args.stride, args.infer_batch)
    print(f"checkpoint={ckpt}")


if __name__ == "__main__": main()
