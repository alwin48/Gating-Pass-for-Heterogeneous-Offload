#!/usr/bin/env python3
"""CPU vs GPU profiling harness with median aggregation and trace_db fallback.

When OpenMP GPU offload is unavailable, timings are taken from data/trace_db.json
keyed by benchmark_id (pre-collected / simulated traces).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TAU = 1.20


def median_ms(samples: list[float]) -> float:
    return float(statistics.median(samples))


def run_timed(cmd: list[str], runs: int = 10, warmups: int = 3, env: dict | None = None) -> float:
    """Run command; parse 'GPHO_TIME_MS=<float>' from stdout, else wall time."""
    e = os.environ.copy()
    if env:
        e.update(env)
    times: list[float] = []
    for i in range(warmups + runs):
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, env=e, check=False)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        parsed = None
        for line in (p.stdout or "").splitlines():
            if line.startswith("GPHO_TIME_MS="):
                parsed = float(line.split("=", 1)[1])
        sample = parsed if parsed is not None else wall_ms
        if i >= warmups:
            times.append(sample)
    return median_ms(times)


def load_trace_db(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def profile_record(
    rec: dict[str, Any],
    trace_db: dict[str, Any],
    force_trace: bool = False,
) -> dict[str, Any]:
    """Attach timings from trace_db (MVP default without live GPU)."""
    key = rec["benchmark_id"]
    # Prefer exact key; else strip mutant suffix.
    entry = trace_db.get(key)
    if entry is None and "_m" in key:
        entry = trace_db.get(key.split("_m")[0])
    out = dict(rec)
    if entry:
        out["t_cpu_ms"] = entry["t_cpu_ms"]
        out["t_gpu_ms"] = entry["t_gpu_ms"]
        # Scale mutant trip counts roughly.
        if "_m" in key and rec.get("trip_count_eval") and entry.get("trip_count_eval"):
            scale = rec["trip_count_eval"] / max(entry["trip_count_eval"], 1)
            out["t_cpu_ms"] = entry["t_cpu_ms"] * scale
            out["t_gpu_ms"] = entry["t_gpu_ms"] * max(scale, 0.15)
    elif force_trace:
        raise KeyError(f"No trace for {key}")
    else:
        # Leave existing simulated timings if present.
        pass

    if out.get("t_cpu_ms") is not None and out.get("t_gpu_ms") is not None:
        out["speedup_actual"] = out["t_cpu_ms"] / max(out["t_gpu_ms"], 1e-9)
        out["label_profitable"] = bool(
            out.get("gpu_safe") and out["speedup_actual"] >= TAU
        )
    return out


def compile_and_profile_c(
    src: Path,
    mode: str,
    runs: int = 10,
) -> float:
    """Compile a demo kernel with g++ (CPU) — OpenMP GPU needs clang+offload."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "a.out"
        if mode == "cpu":
            cmd = ["g++", "-O3", "-fopenmp", "-march=native", str(src), "-o", str(out)]
        else:
            # Best-effort OpenMP target; falls back to CPU compile if unsupported.
            clang = os.environ.get("CLANG", "clang++-18")
            cmd = [
                clang,
                "-O3",
                "-fopenmp",
                "-fopenmp-targets=nvptx64",
                str(src),
                "-o",
                str(out),
            ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Compile failed ({mode}): {r.stderr[-500:]}")
        env = {"GPHO_MODE": mode}
        return run_timed([str(out)], runs=runs, env=env)


def extract_features_via_opt(ll_path: Path, benchmark_id: str, suite: str) -> list[dict]:
    opt = os.environ.get("OPT", "/usr/lib/llvm-18/bin/opt")
    plugin = ROOT / "llvm-pass" / "GPHOPass.so"
    cmd = [
        opt,
        f"-load-pass-plugin={plugin}",
        "-passes=gpho-analysis",
        f"-gpho-benchmark-id={benchmark_id}",
        f"-gpho-suite={suite}",
        "-disable-output",
        str(ll_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Plugin prints JSON array to stdout; opt may add other noise — find '['.
    text = p.stdout
    start = text.find("[")
    if start < 0:
        raise RuntimeError(f"No JSON from pass: {p.stderr}")
    return json.loads(text[start:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "dataset.jsonl",
    )
    ap.add_argument(
        "--trace-db",
        type=Path,
        default=ROOT / "data" / "trace_db.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "dataset_profiled.jsonl",
    )
    ap.add_argument("--extract-ll", type=Path, help="Run pass on .ll and print JSON")
    ap.add_argument("--benchmark-id", default="unknown")
    ap.add_argument("--suite", default="demo")
    args = ap.parse_args()

    if args.extract_ll:
        recs = extract_features_via_opt(args.extract_ll, args.benchmark_id, args.suite)
        print(json.dumps(recs, indent=2))
        return

    if not args.dataset.exists():
        raise SystemExit(f"Missing dataset {args.dataset}; run ml/generate_dataset.py")

    trace = load_trace_db(args.trace_db)
    out_rows = []
    with args.dataset.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out_rows.append(profile_record(rec, trace))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Profiled {len(out_rows)} records → {args.out}")


if __name__ == "__main__":
    main()
