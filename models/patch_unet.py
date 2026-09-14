#!/usr/bin/env python3
"""Small patch U-Net baseline inspired by community pipeline.

Train on filament-centered 512px patches. Infer with overlapping 512px tiles and max
stitching. Emits model-neutral JSONL consumed by postprocess.py.
"""

from __future__ import annotations

import argparse
import json
import random
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
from scripts.pipeline_lib import polygons_to_mask


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
        self.indices = [(i, j) for i in range(len(self.rows)) for j in range(samples_per_image)]
        self._cached_row = None
        self._cached_data = None

    def __len__(self): return len(self.indices)

    def __getitem__(self, index):
        row_i, repeat = self.indices[index]
        # Two samples per image are adjacent. Avoid rereading JPEG and rasterizing
        # every polygon twice; this was starving CUDA during training.
        if self._cached_row != row_i:
            self._cached_row = row_i
            self._cached_data = load_row(self.rows[row_i])
        image, mask = self._cached_data
        rng = np.random.default_rng(self.seed + index * 1009 + repeat)
        ys, xs = np.nonzero(mask)
        if len(xs) and rng.random() < 0.75:
            cx, cy = int(xs[rng.integers(len(xs))]), int(ys[rng.integers(len(ys))])
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


def write_predictions(model, manifest, output, mask_root, device, threshold, min_area, max_candidates, stride, infer_batch):
    rows=[json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w") as dst:
        for row in tqdm(rows, desc=f"inference {manifest.stem}", unit="image"):
            prob=predict_image(model, row["path"], device, stride, infer_batch)
            labels, n=ndimage.label(prob >= threshold, structure=np.ones((3,3)))
            candidates=[]
            for label in range(1,n+1):
                mask=labels==label; area=int(mask.sum())
                if area < min_area: continue
                score=float(np.clip(prob[mask].mean(), .01, .99)); candidates.append((score,area,mask))
            instances=[]; out_dir=Path(mask_root)/row["physical_id"]; out_dir.mkdir(parents=True,exist_ok=True)
            for score,area,mask in sorted(candidates,reverse=True)[:max_candidates]:
                p=out_dir/f"{len(instances):03d}.png"; Image.fromarray((mask*255).astype(np.uint8)).save(p); instances.append({"score":score,"mask_path":str(p)})
            dst.write(json.dumps({"image_id":row["physical_id"],"instances":instances})+"\n")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--train",type=Path,required=True); ap.add_argument("--val",type=Path,required=True); ap.add_argument("--test",type=Path,required=True); ap.add_argument("--raw-val",type=Path,required=True); ap.add_argument("--raw-test",type=Path,required=True); ap.add_argument("--run-dir",type=Path,required=True); ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--batch-size",type=int,default=4); ap.add_argument("--samples-per-image",type=int,default=3); ap.add_argument("--max-train-rows",type=int); ap.add_argument("--device",default="auto"); ap.add_argument("--threshold",type=float,default=.35); ap.add_argument("--min-area",type=int,default=80); ap.add_argument("--max-candidates",type=int,default=20); ap.add_argument("--stride",type=int,default=256); ap.add_argument("--infer-batch",type=int,default=8); args=ap.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42); device=device_for(args.device); print(f"device={device}")
    ds=PatchDataset(args.train,samples_per_image=args.samples_per_image,max_rows=args.max_train_rows); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=(device.type == "cuda"))
    model=UNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train(); total=0.
        progress = tqdm(loader, desc=f"train epoch {epoch + 1}/{args.epochs}", unit="batch")
        for x,y in progress:
            opt.zero_grad(); z=loss_fn(model(x.to(device, non_blocking=True)),y.to(device, non_blocking=True)); z.backward(); opt.step(); total += z.detach().item()
            progress.set_postfix(loss=f"{z.detach().item():.4f}")
        print(f"epoch={epoch+1}/{args.epochs} loss={total/max(len(loader),1):.4f}",flush=True)
    ckpt=args.run_dir/"patch_unet.pt"; ckpt.parent.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),ckpt)
    model.eval(); write_predictions(model,args.val,args.raw_val,args.run_dir/"masks/val",device,args.threshold,args.min_area,args.max_candidates,args.stride,args.infer_batch); write_predictions(model,args.test,args.raw_test,args.run_dir/"masks/test",device,args.threshold,args.min_area,args.max_candidates,args.stride,args.infer_batch)
    print(f"checkpoint={ckpt}")


if __name__ == "__main__": main()
