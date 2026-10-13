#!/usr/bin/env python3
"""Score VBench-standard videos and aggregate Quality/Semantic/Total."""

import argparse
import json
import os

import torch

from constants import (
    DIM_WEIGHT,
    NORMALIZE_DIC,
    QUALITY_LIST,
    QUALITY_WEIGHT,
    SEMANTIC_LIST,
    SEMANTIC_WEIGHT,
    TASK_INFO,
)


def scalar(value):
    if isinstance(value, list) and value:
        value = value[0]
    return float(value) if isinstance(value, (int, float)) else None


def aggregate_scores(raw_results):
    scores = {}
    for key, value in raw_results.items():
        value = scalar(value)
        if value is not None:
            scores[key.replace("_", " ")] = value

    missing = [key for key in TASK_INFO if key not in scores]
    if missing:
        raise RuntimeError(f"Cannot aggregate incomplete VBench results; missing: {missing}")

    normalized = {}
    for key in TASK_INFO:
        bounds = NORMALIZE_DIC[key]
        normalized[key] = (
            (scores[key] - bounds["Min"])
            / (bounds["Max"] - bounds["Min"])
            * DIM_WEIGHT[key]
        )

    quality = sum(normalized[key] for key in QUALITY_LIST) / sum(
        DIM_WEIGHT[key] for key in QUALITY_LIST
    )
    semantic = sum(normalized[key] for key in SEMANTIC_LIST) / sum(
        DIM_WEIGHT[key] for key in SEMANTIC_LIST
    )
    total = (
        quality * QUALITY_WEIGHT + semantic * SEMANTIC_WEIGHT
    ) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

    return {
        "quality_score": quality,
        "semantic_score": semantic,
        "total_score": total,
        "quality_pct": quality * 100,
        "semantic_pct": semantic * 100,
        "total_pct": total * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dimensions", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--name", default="eval")
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Write selected-dimension results without requiring all 16 dimensions.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    videos = [name for name in os.listdir(args.videos_dir) if name.endswith(".mp4")]
    if not videos:
        raise RuntimeError(f"No MP4 videos found in {args.videos_dir}")
    print(f"Found {len(videos)} MP4 files in {args.videos_dir}", flush=True)

    from vbench import VBench
    import vbench

    full_info_path = os.path.join(os.path.dirname(vbench.__file__), "VBench_full_info.json")
    if not os.path.isfile(full_info_path):
        raise FileNotFoundError(f"VBench metadata not found: {full_info_path}")

    evaluator = VBench(torch.device(args.device), full_info_path, args.output_dir)
    dimensions = args.dimensions or evaluator.build_full_dimension_list()
    print(f"Evaluating {len(dimensions)} dimensions: {dimensions}", flush=True)
    evaluator.evaluate(
        videos_path=args.videos_dir,
        name=args.name,
        dimension_list=dimensions,
        mode="vbench_standard",
    )

    result_path = os.path.join(args.output_dir, f"{args.name}_eval_results.json")
    if not os.path.isfile(result_path):
        raise FileNotFoundError(f"VBench result file was not produced: {result_path}")
    with open(result_path, encoding="utf-8") as handle:
        results = json.load(handle)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    if args.allow_partial:
        print(
            f"Partial evaluation complete ({len(dimensions)} dimensions); "
            "skipping aggregate.",
            flush=True,
        )
        return

    aggregate = aggregate_scores(results)
    with open(os.path.join(args.output_dir, "aggregate.json"), "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    print("\n=== VBench aggregate (official) ===")
    print(f"Quality  : {aggregate['quality_pct']:.2f}")
    print(f"Semantic : {aggregate['semantic_pct']:.2f}")
    print(f"Total    : {aggregate['total_pct']:.2f}")
    print(f"Saved: {args.output_dir}/aggregate.json", flush=True)


if __name__ == "__main__":
    main()
