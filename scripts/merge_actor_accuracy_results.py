#!/usr/bin/env python3
"""Merge multiple actor-accuracy shard result JSON files into one summary."""

import argparse
import glob
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge sharded actor-accuracy result files")
    parser.add_argument(
        "--input_pattern",
        type=str,
        required=True,
        help="Glob pattern for shard json files, e.g. /path/model_shard*.json",
    )
    parser.add_argument("--output_file", type=str, required=True, help="Merged output json path")
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(args.input_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {args.input_pattern}")

    merged = {
        "metadata": {
            "source_files": files,
        },
        "metrics": {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
        },
        "mismatches": [],
    }

    for path in files:
        with open(path, "r") as f:
            payload = json.load(f)
        merged["metrics"]["correct"] += int(payload["metrics"]["correct"])
        merged["metrics"]["total"] += int(payload["metrics"]["total"])
        merged["mismatches"].extend(payload.get("mismatches", []))

    total = merged["metrics"]["total"]
    correct = merged["metrics"]["correct"]
    merged["metrics"]["accuracy"] = correct / total if total else 0.0

    output_path = Path(args.output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(json.dumps(merged["metrics"], indent=2))


if __name__ == "__main__":
    main()
