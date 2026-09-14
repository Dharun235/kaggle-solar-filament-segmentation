"""Model adapter contract.

Implement model-specific training/inference elsewhere. This file documents the only
interface required by postprocess.py; keeping it JSONL makes model swaps cheap.
"""

from pathlib import Path
import json


def write_prediction(out, image_id: str, instances: list[dict]) -> None:
    """Write one image prediction.

    Each instance needs `score` and either:
      * `mask_path`: binary PNG, exactly 2048x2048; or
      * `mask`: path to binary PNG (alias accepted by postprocess).
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("a") as f:
        f.write(json.dumps({"image_id": image_id, "instances": instances}) + "\n")


if __name__ == "__main__":
    raise SystemExit("Template only. Replace with model training/inference adapter.")
