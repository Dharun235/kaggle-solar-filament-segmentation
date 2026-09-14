from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

SIZE = (2048, 2048)


def load_json(path: str | Path):
    with Path(path).open() as f:
        return json.load(f)


def image_stem(name: str) -> str:
    return Path(name).stem


def physical_stem(name: str) -> str:
    stem = image_stem(name)
    # COCO validation IDs prepend six-digit annotator batch, e.g. 040301-foo.jpeg.
    return re.sub(r"^\d{6}-", "", stem)


def encode_mask(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=np.uint8, order="F")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    rle = mask_utils.encode(mask[:, :, None])[0]
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def decode_rle(counts: str, size=SIZE) -> np.ndarray:
    return np.asarray(mask_utils.decode({"size": list(size), "counts": counts.encode("ascii")})).astype(bool)


def polygons_to_mask(segmentation, height=2048, width=2048) -> np.ndarray:
    rles = mask_utils.frPyObjects(segmentation, height, width)
    return np.asarray(mask_utils.decode(rles).any(axis=2) if isinstance(rles, list) else mask_utils.decode(rles)).astype(bool)


def read_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8) > 0


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def jsonl_read(path: str | Path):
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL line {line_no}: {e}") from e


def jsonl_write(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
