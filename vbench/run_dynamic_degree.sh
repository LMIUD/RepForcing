#!/usr/bin/env bash
# Modified for the RepForcing source release; see NOTICE.md.
# Evaluate Dynamic Degree for a directory containing 100 MP4 videos.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON="${PYTHON:-python}"

export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-${PROJECT_ROOT}/.cache/vbench}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

exec "$PYTHON" -B - "$PROJECT_ROOT" "$@" <<'PY'
import argparse
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

project_root = Path(sys.argv.pop(1))
parser = argparse.ArgumentParser(
    prog="bash vbench/run_dynamic_degree.sh",
    description="Evaluate Dynamic Degree on 100 MP4 videos using VBench custom_input mode.",
    epilog="Select GPUs with CUDA_VISIBLE_DEVICES; override the interpreter and cache with PYTHON and VBENCH_CACHE_DIR.",
)
parser.add_argument("videos_dir", type=Path, help="Directory containing 100 MP4 videos; subdirectories are not scanned")
parser.add_argument("output_dir", type=Path, nargs="?", help="Optional new output directory; existing directories are rejected")
args = parser.parse_args()

videos_dir = args.videos_dir.expanduser().resolve()
if not videos_dir.is_dir():
    parser.error(f"Video directory does not exist: {videos_dir}")
videos = sorted(p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")
if len(videos) != 100:
    parser.error(f"Expected 100 MP4 videos, found {len(videos)} in {videos_dir}")
extra_media = [p.name for p in videos_dir.iterdir() if p.suffix.lower() in (".gif", ".jpg", ".png")]
if extra_media:
    parser.error("VBench would also read images or GIFs in this directory; use a directory containing only the target MP4 videos")
cache_dir = Path(os.environ["VBENCH_CACHE_DIR"]).expanduser().resolve()
os.environ["VBENCH_CACHE_DIR"] = str(cache_dir)
weight = cache_dir / "raft_model/models/raft-things.pth"
if not weight.is_file():
    parser.error(f"RAFT weights not found; automatic download is disabled: {weight}")

if args.output_dir is None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = project_root / "vbench/results" / f"dynamic_degree_{videos_dir.name}_{stamp}"
else:
    output_dir = args.output_dir.expanduser().resolve()
if output_dir.exists():
    parser.error(f"Output directory already exists; refusing to overwrite: {output_dir}")

import torch
import vbench
from vbench import VBench

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; evaluation has not started")
metadata = Path(vbench.__file__).parent / "VBench_full_info.json"
if not metadata.is_file():
    raise FileNotFoundError(metadata)
output_dir.mkdir(parents=True, exist_ok=False)
manifest = {
    "metric": "dynamic_degree",
    "mode": "custom_input",
    "videos_dir": str(videos_dir),
    "video_files": [p.name for p in videos],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all visible devices"),
    "gpu_name": torch.cuda.get_device_name(0),
    "vbench_version": importlib.metadata.version("vbench"),
    "raft_weight": str(weight),
    "python": sys.executable,
}
with (output_dir / "run_manifest.json").open("x", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
print(f"Video directory: {videos_dir}")
print(f"Video count: {len(videos)}")
print(f"GPU: {manifest['cuda_visible_devices']} ({manifest['gpu_name']})")
print(f"Output directory: {output_dir}")
started = time.monotonic()
evaluator = VBench(torch.device("cuda:0"), str(metadata), str(output_dir))
evaluator.evaluate(
    videos_path=str(videos_dir),
    name="dynamic_degree",
    dimension_list=["dynamic_degree"],
    mode="custom_input",
)
result_file = output_dir / "dynamic_degree_eval_results.json"
result = json.loads(result_file.read_text(encoding="utf-8"))
score, details = result["dynamic_degree"]
expected = {p.resolve() for p in videos}
observed = {Path(item["video_path"]).resolve() for item in details}
if len(details) != 100 or observed != expected:
    raise RuntimeError("Results do not cover all 100 input videos; no aggregate score will be produced")
values = [item["video_results"] for item in details]
if any(value not in (True, False, 0, 1) for value in values):
    raise RuntimeError("Per-video dynamic decisions must be binary")
score = float(score)
if not math.isfinite(score) or abs(score - sum(values) / len(values)) > 1e-8:
    raise RuntimeError("Aggregate score does not match the per-video decisions")
summary = {
    "videos_dir": str(videos_dir),
    "num_videos": len(details),
    "dynamic_videos": int(sum(values)),
    "dynamic_degree": score,
    "dynamic_degree_pct": 100 * score,
    "elapsed_seconds": time.monotonic() - started,
    "vbench_version": manifest["vbench_version"],
}
with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
print(f"\nDynamic Degree: {100 * score:.2f}")
print(f"Videos classified as dynamic: {sum(values)}/100")
print(f"Summary: {output_dir / 'summary.json'}")
print(f"Per-video results: {result_file}")
PY
