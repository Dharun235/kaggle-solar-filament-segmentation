#!/usr/bin/env python3
"""One-command controller for the Solar Filament pipeline.

Model-specific code stays outside this controller. It receives manifests and must write
JSONL predictions described in scripts/model_adapter_template.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "MAGFiLO_1.0_Kaggle_2026"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(cmd: list[str], label: str, env: dict[str, str]) -> None:
    print(f"\n== {label} ==\n$ {shlex.join(cmd)}", flush=True)
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode:
        raise SystemExit(f"{label} failed, exit={result.returncode}")
    print(f"{label}: OK ({time.time() - started:.1f}s)", flush=True)


def render_model_command(template: str, values: dict[str, Path | str]) -> list[str]:
    rendered = template.format(**{k: str(v) for k, v in values.items()})
    return shlex.split(rendered)


def write_run_metadata(path: Path, args: argparse.Namespace, run_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_at": utc_now(),
        "git_sha": git_sha(),
        "cwd": str(ROOT),
        "run_dir": str(run_dir),
        "argv": sys.argv,
        "config": vars(args),
    }
    metadata["config"]["data_root"] = str(metadata["config"]["data_root"])
    path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run complete solar-filament pipeline")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/runs/latest")
    p.add_argument("--fold", type=int, default=0, help="validation fold used for model run")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--confidence", type=float, default=0.20)
    p.add_argument("--min-area", type=int, default=16)
    p.add_argument("--max-instances", type=int, default=10)
    p.add_argument("--confidence-grid", type=float, nargs="+",
                   default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    p.add_argument("--model-command", help="shell-like command with placeholders; see README")
    p.add_argument("--raw-val", type=Path, help="skip model command and use existing validation JSONL")
    p.add_argument("--raw-test", type=Path, help="skip model command and use existing test JSONL")
    p.add_argument("--skip-audit", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.fold < args.folds:
        raise SystemExit("--fold must be between 0 and --folds-1")
    data_root = args.data_root.resolve()
    run_dir = args.run_dir.resolve()
    manifests = ROOT / "artifacts/manifests"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SOLAR_DATA_ROOT"] = str(data_root)
    env["SOLAR_RUN_DIR"] = str(run_dir)
    write_run_metadata(run_dir / "run.json", args, run_dir)

    # 1. Verify data and create leakage-safe train/val/test manifests.
    run([sys.executable, "scripts/audit_data.py", "--root", str(data_root)], "data audit", env)
    run([sys.executable, "scripts/prepare_data.py", "--root", str(data_root), "--folds", str(args.folds)],
        "manifest preparation", env)

    raw_val = (args.raw_val or run_dir / "raw_val.jsonl").resolve()
    raw_test = (args.raw_test or run_dir / "raw_test.jsonl").resolve()
    model_values = {
        "train_manifest": manifests / f"train_fold{args.fold}.jsonl",
        "val_manifest": manifests / f"val_fold{args.fold}.jsonl",
        "test_manifest": manifests / "test.jsonl",
        "raw_val": raw_val,
        "raw_test": raw_test,
        "run_dir": run_dir,
        "fold": args.fold,
    }

    # 2. Model hook. Controller does not assume YOLO, U-Net, Torch, or framework.
    if args.model_command:
        run(render_model_command(args.model_command, model_values), "model command", env)
    elif not raw_val.exists() or not raw_test.exists():
        print("\nPreparation complete. Model command missing.")
        print("Required placeholders: {train_manifest} {val_manifest} {test_manifest} {raw_val} {raw_test} {run_dir} {fold}")
        print("Example:")
        print("python main.py --model-command 'python models/run_model.py --train {train_manifest} --val {val_manifest} --test {test_manifest} --raw-val {raw_val} --raw-test {raw_test} --run-dir {run_dir}'")
        return

    if not raw_val.exists() or not raw_test.exists():
        raise SystemExit(f"model must create both files: {raw_val}, {raw_test}")

    # 3. Validation postprocess: calibrates confidence against official-style PQ.
    val_csv = run_dir / "validation.csv"
    selected_confidence = run_dir / "selected_confidence.json"
    gt = data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    run([sys.executable, "scripts/postprocess.py", "--predictions", str(raw_val), "--output", str(val_csv),
         "--confidence-grid", *map(str, args.confidence_grid), "--max-instances", str(args.max_instances),
         "--min-area", str(args.min_area), "--ground-truth", str(gt),
         "--selected-confidence-file", str(selected_confidence),
         "--expected-manifest", str(manifests / f"val_fold{args.fold}.jsonl")], "validation PQ", env)
    calibrated_confidence = json.loads(selected_confidence.read_text())["confidence"]
    (run_dir / "metric.json").write_text(json.dumps({
        "protocol": "organizer_self_evaluation",
        "iou_match": ">0.5",
        "aggregation": "mean over annotator records",
        "confidence": calibrated_confidence,
    }, indent=2) + "\n")

    # 4. Test postprocess: RLE, non-overlap, Kaggle column format.
    submission = run_dir / "submission.csv"
    run([sys.executable, "scripts/postprocess.py", "--predictions", str(raw_test), "--output", str(submission),
         "--confidence", str(calibrated_confidence), "--max-instances", str(args.max_instances),
         "--min-area", str(args.min_area), "--expected-manifest", str(manifests / "test.jsonl")], "test postprocess", env)

    # 5. Final submission audit.
    if not args.skip_audit:
        run([sys.executable, "scripts/audit_submission.py", "--submission", str(submission),
             "--test-images", str(data_root / "test/test_images")],
            "submission audit", env)
    print(f"\nDONE\nrun={run_dir}\nsubmission={submission}", flush=True)


if __name__ == "__main__":
    main()
