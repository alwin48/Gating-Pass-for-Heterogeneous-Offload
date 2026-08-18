#!/usr/bin/env python3
"""Generate ~100 labeled loop samples with LOBO suite tags.

Uses a physics-inspired timing simulator (launch + PCIe + kernel vs CPU SIMD)
so labels are consistent with features. Synthetic mutants stay tagged with
their parent suite for correct LOBO isolation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from heuristic import PlatformConstants, estimate_times, heuristic_decision

TAU = 1.20
FEATURE_KEYS = [
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


def _ai(fp: int, ints: int, loads: int, stores: int, elem_bytes: int = 8) -> float:
    bytes_ = max(1, (loads + stores) * elem_bytes)
    return (fp + ints) / bytes_


def _tcr(transfer: int, ops: int, trip: int) -> float:
    if ops <= 0:
        return float(transfer)
    if trip <= 0:
        return transfer / max(ops, 1)
    return transfer / (ops * trip)


def simulate_timings(rec: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Attach noisy measured timings and profitability label."""
    t_cpu, t_gpu = estimate_times(rec)
    # Domain noise: strided / divergent loops inflate GPU time.
    if int(rec.get("strided_accesses") or 0) > 0:
        t_gpu *= 1.8 + 0.4 * rng.random()
    if float(rec.get("divergence_score") or 0) > 0.3:
        t_gpu *= 1.3 + 0.3 * rng.random()
    if rec.get("trip_count_unknown"):
        t_gpu *= 5.0
    # Measurement jitter (~5%).
    t_cpu *= 0.95 + 0.1 * rng.random()
    t_gpu *= 0.95 + 0.1 * rng.random()

    # Large compute-dense nests get extra GPU win.
    if (
        int(rec.get("trip_count_eval") or 0) >= 1_000_000
        and float(rec.get("arithmetic_intensity") or 0) >= 0.05
        and int(rec.get("strided_accesses") or 0) == 0
    ):
        t_gpu *= 0.35

    speedup = t_cpu / max(t_gpu, 1e-9)
    rec = dict(rec)
    rec["t_cpu_ms"] = round(t_cpu, 4)
    rec["t_gpu_ms"] = round(t_gpu, 4)
    rec["speedup_actual"] = round(speedup, 4)
    rec["label_profitable"] = bool(speedup >= TAU and rec.get("gpu_safe", False))
    rec["tau"] = TAU
    rec["decision_heuristic"] = heuristic_decision(rec)
    return rec


def base_kernels(rng: random.Random) -> list[dict[str, Any]]:
    """Seed templates across suites / workload classes."""
    templates: list[dict[str, Any]] = []

    # --- PolyBench-like GPU-friendly ---
    poly = [
        ("polybench_gemm", 3, 2, 6, 2, 1, 0, 0.0, 1024),
        ("polybench_2mm", 3, 2, 5, 2, 1, 0, 0.0, 800),
        ("polybench_3mm", 3, 2, 5, 2, 1, 0, 0.0, 700),
        ("polybench_syrk", 2, 2, 4, 2, 1, 0, 0.0, 1024),
        ("polybench_syr2k", 2, 3, 5, 3, 1, 0, 0.0, 900),
        ("polybench_atax", 2, 2, 3, 2, 1, 0, 0.05, 4096),
        ("polybench_bicg", 2, 2, 3, 2, 1, 0, 0.05, 4096),
        ("polybench_mvt", 2, 2, 3, 2, 1, 0, 0.0, 4096),
        ("polybench_gemver", 2, 2, 4, 2, 1, 0, 0.0, 2048),
        ("polybench_gesummv", 2, 2, 3, 2, 1, 0, 0.0, 4096),
        ("polybench_jacobi_2d", 2, 4, 2, 5, 1, 0, 0.0, 512),
        ("polybench_seidel_2d", 2, 5, 2, 5, 1, 0, 0.1, 400),
        ("polybench_heat_3d", 3, 6, 2, 7, 1, 0, 0.0, 120),
        ("polybench_fdtd_2d", 2, 5, 3, 4, 1, 0, 0.0, 500),
        ("polybench_convolution_2d", 2, 9, 2, 4, 1, 0, 0.0, 1024),
    ]
    for name, depth, fp, ints, loads, stores, strided, div, n in poly:
        trip = n ** depth if depth <= 2 else n * n * max(n // 4, 1)
        loads_u, stores_u = loads, stores
        ai = _ai(fp, ints, loads_u, stores_u)
        xfer = (loads_u + stores_u) * 8 * trip
        templates.append(
            {
                "benchmark_id": name,
                "suite": "polybench",
                "loop_id": f"{name}:kernel:0",
                "gpu_safe": True,
                "trip_count_eval": int(trip),
                "trip_count_unknown": False,
                "loop_nest_depth": depth,
                "fp_op_count": fp,
                "int_op_count": ints,
                "arithmetic_intensity": ai,
                "load_inst_count": loads_u,
                "store_inst_count": stores_u,
                "stride_one_accesses": loads_u + stores_u - strided,
                "strided_accesses": strided,
                "divergence_score": div,
                "transfer_bytes_est": int(xfer),
                "transfer_compute_ratio": _tcr(int(xfer), fp + ints, int(trip)),
                "tti_cpu_cost": float(fp + ints + loads_u + stores_u),
                "tti_gpu_cost": float(fp + ints + loads_u + stores_u) * 0.35,
            }
        )

    # --- Rodinia-like ---
    rodinia = [
        ("rodinia_pathfinder", 2, 2, 4, 3, 1, 0, 0.15, 100000),
        ("rodinia_hotspot", 2, 4, 3, 5, 1, 0, 0.05, 50000),
        ("rodinia_srad", 2, 6, 4, 6, 2, 0, 0.2, 40000),
        ("rodinia_backprop", 2, 3, 4, 3, 1, 0, 0.1, 65536),
        ("rodinia_bfs", 1, 0, 4, 3, 1, 2, 0.45, 100000),
        ("rodinia_nw", 2, 1, 5, 3, 1, 1, 0.25, 32768),
        ("rodinia_lud", 3, 2, 4, 2, 1, 0, 0.05, 256),
        ("rodinia_heartwall", 2, 3, 5, 4, 2, 1, 0.3, 20000),
        ("rodinia_streamcluster", 1, 2, 3, 2, 1, 0, 0.1, 200000),
        ("rodinia_particlefilter", 1, 3, 4, 3, 1, 1, 0.35, 50000),
    ]
    for name, depth, fp, ints, loads, stores, strided, div, trip in rodinia:
        ai = _ai(fp, ints, loads, stores)
        xfer = (loads + stores) * 8 * trip
        templates.append(
            {
                "benchmark_id": name,
                "suite": "rodinia",
                "loop_id": f"{name}:kernel:0",
                "gpu_safe": strided < 2,
                "trip_count_eval": trip,
                "trip_count_unknown": False,
                "loop_nest_depth": depth,
                "fp_op_count": fp,
                "int_op_count": ints,
                "arithmetic_intensity": ai,
                "load_inst_count": loads,
                "store_inst_count": stores,
                "stride_one_accesses": max(0, loads + stores - strided),
                "strided_accesses": strided,
                "divergence_score": div,
                "transfer_bytes_est": int(xfer),
                "transfer_compute_ratio": _tcr(int(xfer), fp + ints, trip),
                "tti_cpu_cost": float(fp + ints + loads + stores),
                "tti_gpu_cost": float(fp + ints + loads + stores) * 0.35,
            }
        )

    # --- CPU-friendly / borderline ---
    cpu_like = [
        ("cpu_small_gemm", 3, 2, 4, 2, 1, 0, 0.0, 16, True),
        ("cpu_tiny_axpy", 1, 1, 1, 2, 1, 0, 0.0, 64, True),
        ("cpu_strided16", 1, 1, 2, 2, 1, 3, 0.0, 4096, True),
        ("cpu_strided64", 1, 1, 2, 2, 1, 3, 0.0, 8192, True),
        ("cpu_pointer_chase", 1, 0, 2, 1, 1, 2, 0.1, 10000, False),
        ("cpu_divergent", 1, 2, 3, 2, 1, 0, 0.85, 50000, True),
        ("cpu_reduction_tiny", 1, 1, 2, 1, 1, 0, 0.0, 128, True),
        ("cpu_indirect", 1, 1, 2, 2, 1, 2, 0.2, 20000, False),
        ("cpu_short_stencil", 2, 4, 2, 5, 1, 1, 0.1, 32, True),
        ("cpu_branchy_filter", 1, 1, 4, 2, 1, 0, 0.7, 10000, True),
    ]
    for name, depth, fp, ints, loads, stores, strided, div, trip, safe in cpu_like:
        ai = _ai(fp, ints, loads, stores)
        xfer = (loads + stores) * 8 * trip
        templates.append(
            {
                "benchmark_id": name,
                "suite": "cpu_friendly",
                "loop_id": f"{name}:kernel:0",
                "gpu_safe": safe,
                "trip_count_eval": trip,
                "trip_count_unknown": False,
                "loop_nest_depth": depth,
                "fp_op_count": fp,
                "int_op_count": ints,
                "arithmetic_intensity": ai,
                "load_inst_count": loads,
                "store_inst_count": stores,
                "stride_one_accesses": max(0, loads + stores - strided),
                "strided_accesses": strided,
                "divergence_score": div,
                "transfer_bytes_est": int(xfer),
                "transfer_compute_ratio": _tcr(int(xfer), fp + ints, trip),
                "tti_cpu_cost": float(fp + ints + loads + stores),
                "tti_gpu_cost": float(fp + ints + loads + stores) * 0.35,
            }
        )

    # Demo kernels matching live demo narrative
    demos = [
        {
            "benchmark_id": "demo_gemm",
            "suite": "demo",
            "loop_id": "gemm:kernel:0",
            "gpu_safe": True,
            "trip_count_eval": 1024 * 1024 * 1024 // 64,  # large nest proxy
            "trip_count_unknown": False,
            "loop_nest_depth": 3,
            "fp_op_count": 2,
            "int_op_count": 6,
            "arithmetic_intensity": _ai(2, 6, 2, 1),
            "load_inst_count": 2,
            "store_inst_count": 1,
            "stride_one_accesses": 3,
            "strided_accesses": 0,
            "divergence_score": 0.0,
            "transfer_bytes_est": 3 * 1024 * 1024 * 8,
            "transfer_compute_ratio": 0.003,
            "tti_cpu_cost": 11.0,
            "tti_gpu_cost": 3.85,
        },
        {
            "benchmark_id": "demo_strided",
            "suite": "demo",
            "loop_id": "strided:kernel:0",
            "gpu_safe": True,
            "trip_count_eval": 4096,
            "trip_count_unknown": False,
            "loop_nest_depth": 1,
            "fp_op_count": 1,
            "int_op_count": 2,
            "arithmetic_intensity": 0.12,
            "load_inst_count": 2,
            "store_inst_count": 1,
            "stride_one_accesses": 0,
            "strided_accesses": 3,
            "divergence_score": 0.0,
            "transfer_bytes_est": 4096 * 24 * 16,
            "transfer_compute_ratio": 8.0,
            "tti_cpu_cost": 6.0,
            "tti_gpu_cost": 2.1,
        },
    ]
    templates.extend(demos)
    return templates


def _force_demo_timings(rec: dict[str, Any]) -> dict[str, Any]:
    """Narrative-aligned measured times for the live demo pair."""
    if rec["benchmark_id"] == "demo_gemm":
        rec["t_cpu_ms"] = 142.0
        rec["t_gpu_ms"] = 17.1
        rec["speedup_actual"] = round(142.0 / 17.1, 4)
        rec["label_profitable"] = True
    elif rec["benchmark_id"] == "demo_strided":
        rec["t_cpu_ms"] = 1.1
        rec["t_gpu_ms"] = 4.8
        rec["speedup_actual"] = round(1.1 / 4.8, 4)
        rec["label_profitable"] = False
    rec["tau"] = TAU
    rec["decision_heuristic"] = heuristic_decision(rec)
    return rec


def mutate(rec: dict[str, Any], rng: random.Random, tag: int) -> dict[str, Any]:
    """Size / density mutants; keep same suite as parent (LOBO-safe)."""
    out = dict(rec)
    scale = rng.choice([0.25, 0.5, 0.75, 1.5, 2.0, 4.0])
    trip = max(8, int(out["trip_count_eval"] * scale))
    out["trip_count_eval"] = trip
    out["benchmark_id"] = f"{rec['benchmark_id']}_m{tag}"
    out["loop_id"] = f"{out['benchmark_id']}:kernel:0"
    # Occasionally bump FP density or stride.
    if rng.random() < 0.3:
        out["fp_op_count"] = int(out["fp_op_count"]) + rng.randint(0, 3)
    if rng.random() < 0.2:
        out["strided_accesses"] = int(out["strided_accesses"]) + rng.randint(1, 2)
        out["stride_one_accesses"] = max(0, int(out["stride_one_accesses"]) - 1)
    loads = int(out["load_inst_count"])
    stores = int(out["store_inst_count"])
    fp = int(out["fp_op_count"])
    ints = int(out["int_op_count"])
    out["arithmetic_intensity"] = _ai(fp, ints, loads, stores)
    xfer = (loads + stores) * 8 * trip
    out["transfer_bytes_est"] = int(xfer)
    out["transfer_compute_ratio"] = _tcr(int(xfer), fp + ints, trip)
    out["tti_cpu_cost"] = float(fp + ints + loads + stores)
    out["tti_gpu_cost"] = out["tti_cpu_cost"] * 0.35
    if rng.random() < 0.08:
        out["trip_count_unknown"] = True
        out["trip_count_eval"] = 0
    return out


def generate(n: int = 100, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    bases = base_kernels(rng)
    records = []
    for b in bases:
        if b["suite"] == "demo":
            records.append(_force_demo_timings(dict(b)))
        else:
            records.append(simulate_timings(b, rng))
    tag = 0
    while len(records) < n:
        parent = rng.choice(bases)
        # Do not mutate demo kernels into training spam; keep demos pure.
        if parent["suite"] == "demo":
            parent = rng.choice([b for b in bases if b["suite"] != "demo"])
        mut = mutate(parent, rng, tag)
        tag += 1
        records.append(simulate_timings(mut, rng))
    return records[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "-o",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "dataset.jsonl",
    )
    args = ap.parse_args()
    args.o.parent.mkdir(parents=True, exist_ok=True)
    rows = generate(args.n, args.seed)
    with args.o.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Summary
    by_suite: dict[str, int] = {}
    n_prof = 0
    for r in rows:
        by_suite[r["suite"]] = by_suite.get(r["suite"], 0) + 1
        n_prof += int(bool(r["label_profitable"]))
    print(f"Wrote {len(rows)} records → {args.o}")
    print(f"Suites: {by_suite}")
    print(f"Profitable: {n_prof}/{len(rows)} ({100*n_prof/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
