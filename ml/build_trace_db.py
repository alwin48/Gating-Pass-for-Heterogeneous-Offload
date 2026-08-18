#!/usr/bin/env python3
"""Build data/trace_db.json from dataset records (hardware-fallback traces)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ds = ROOT / "data" / "dataset.jsonl"
    rows = [json.loads(l) for l in ds.read_text(encoding="utf-8").splitlines() if l.strip()]
    db = {}
    for r in rows:
        # Prefer non-mutant canonical ids
        bid = r["benchmark_id"]
        if "_m" in bid:
            continue
        db[bid] = {
            "t_cpu_ms": r["t_cpu_ms"],
            "t_gpu_ms": r["t_gpu_ms"],
            "trip_count_eval": r.get("trip_count_eval"),
            "speedup_actual": r.get("speedup_actual"),
            "source": "simulated_or_profiled_trace",
            "platform_note": "Default traces for demo/CI without live GPU; replace with AWS g4dn medians when available.",
        }
    out = ROOT / "data" / "trace_db.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(db, indent=2), encoding="utf-8")
    print(f"Wrote {len(db)} traces → {out}")


if __name__ == "__main__":
    main()
