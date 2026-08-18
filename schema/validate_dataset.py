#!/usr/bin/env python3
"""Validate dataset.jsonl against the frozen Day-1 schema (required fields)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = [
    "benchmark_id",
    "suite",
    "loop_id",
    "gpu_safe",
    "trip_count_eval",
    "trip_count_unknown",
    "loop_nest_depth",
    "fp_op_count",
    "int_op_count",
    "arithmetic_intensity",
    "load_inst_count",
    "store_inst_count",
    "stride_one_accesses",
    "strided_accesses",
    "divergence_score",
    "transfer_bytes_est",
    "transfer_compute_ratio",
]


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/dataset.jsonl")
    n = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        missing = [k for k in REQUIRED if k not in rec]
        if missing:
            print(f"line {i}: missing {missing}")
            return 1
        n += 1
    print(f"OK: {n} records satisfy required schema fields ({path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
