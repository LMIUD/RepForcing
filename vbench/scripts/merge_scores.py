#!/usr/bin/env python3
"""Merge independently evaluated VBench dimensions and compute official aggregates."""

import argparse
import json
import os

from constants import TASK_INFO
from score import aggregate_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    merged = {}
    missing = []
    for display_name in TASK_INFO:
        dimension = display_name.replace(" ", "_")
        result_path = os.path.join(
            args.partial_root,
            dimension,
            f"{dimension}_eval_results.json",
        )
        if not os.path.isfile(result_path):
            missing.append(dimension)
            continue
        with open(result_path, encoding="utf-8") as handle:
            result = json.load(handle)
        if dimension not in result:
            missing.append(dimension)
            continue
        merged[dimension] = result[dimension]

    if missing:
        raise RuntimeError(f"Missing independently scored dimensions: {missing}")

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, f"{args.name}_eval_results.json")
    summary_path = os.path.join(args.output_dir, "summary.json")
    aggregate_path = os.path.join(args.output_dir, "aggregate.json")

    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False)

    aggregate = aggregate_scores(merged)
    with open(aggregate_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    print(f"Quality  : {aggregate['quality_pct']:.2f}")
    print(f"Semantic : {aggregate['semantic_pct']:.2f}")
    print(f"Total    : {aggregate['total_pct']:.2f}")
    print(f"Saved: {aggregate_path}")


if __name__ == "__main__":
    main()
