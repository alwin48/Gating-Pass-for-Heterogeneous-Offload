#!/usr/bin/env python3
"""GPHO Analytical Heuristic Cost Model.

Implements a deterministic first-principles hardware performance equation
modeling multi-threaded CPU SIMD execution vs. GPU kernel launch latency,
PCIe Host-to-Device/Device-to-Host transfer time, and massively parallel SIMT
compute throughput.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PlatformConstants:
    """Hardware architecture calibration constants.

    Default values model a modern enterprise heterogeneous node:
    - Host CPU: 8-core AVX-512 @ 2.5 GHz (e.g. Intel Xeon / AMD EPYC)
    - Device GPU: NVIDIA PCIe Gen4 x16 accelerator (2560 CUDA cores @ 1.5 GHz)
    """

    w_simd: float = 8.0          # AVX-512 double-precision vector lanes
    f_cpu_ghz: float = 2.5       # CPU clock frequency in GHz
    c_cores: float = 8.0         # Multithreaded CPU physical core count
    t_launch_us: float = 15.0    # GPU runtime driver/kernel launch overhead (microseconds)
    b_pcie_gbs: float = 16.0     # Effective bidirectional PCIe Gen4 bandwidth (GB/s)
    n_cuda_cores: float = 2560.0 # GPU compute units / CUDA cores
    f_gpu_ghz: float = 1.5       # GPU core clock frequency in GHz
    eta_occupancy: float = 0.65  # Practical warp occupancy / execution efficiency
    tau: float = 1.20            # Speedup threshold required for offload (>= 1.20x)


def estimate_times(
    rec: Mapping[str, Any],
    plat: PlatformConstants | None = None,
) -> tuple[float, float]:
    """Calculate analytical execution times in milliseconds: (T_CPU_ms, T_GPU_ms)."""
    plat = plat or PlatformConstants()

    trip = int(rec.get("trip_count_eval") or 0)
    trip_unknown = bool(rec.get("trip_count_unknown", False))

    if trip_unknown or trip <= 0:
        # Unknown loop bounds -> fallback to CPU baseline with penalty
        return 1.0, 1e9

    cpu_cost = float(
        rec.get("tti_cpu_cost")
        or (rec.get("fp_op_count", 0) + rec.get("int_op_count", 0))
    )
    gpu_cost = float(rec.get("tti_gpu_cost") or (cpu_cost * 0.35))
    bytes_xfer = float(rec.get("transfer_bytes_est") or 0)

    # 1. CPU Execution Time: T_CPU = (N * Cost_CPU) / (W_SIMD * f_CPU * C_cores)
    denom_cpu = plat.w_simd * (plat.f_cpu_ghz * 1e9) * plat.c_cores
    t_cpu_s = (trip * max(cpu_cost, 1.0)) / max(denom_cpu, 1.0)
    # Scale relative TTI units into realistic millisecond ranges
    t_cpu_ms = t_cpu_s * 1e3 * 50.0

    # 2. GPU Execution Time: T_GPU = T_launch + T_PCIe + T_Compute
    t_launch_ms = plat.t_launch_us / 1000.0
    t_xfer_ms = (bytes_xfer / (plat.b_pcie_gbs * 1e9)) * 1e3
    denom_gpu = plat.n_cuda_cores * (plat.f_gpu_ghz * 1e9) * plat.eta_occupancy
    t_kern_s = (trip * max(gpu_cost, 0.1)) / max(denom_gpu, 1.0)
    t_kern_ms = t_kern_s * 1e3 * 50.0

    t_gpu_ms = t_launch_ms + t_xfer_ms + t_kern_ms
    return max(t_cpu_ms, 1e-6), max(t_gpu_ms, 1e-6)


def heuristic_decision(
    rec: Mapping[str, Any],
    plat: PlatformConstants | None = None,
) -> str:
    """Return deterministic analytical gating decision: 'GPU' or 'CPU'."""
    plat = plat or PlatformConstants()

    # Rule 1: Dependence safety hard gate
    if not rec.get("gpu_safe", False):
        return "CPU"

    # Rule 2: Trip count unknown gate
    if rec.get("trip_count_unknown", False):
        return "CPU"

    # Rule 3: Stride penalty & arithmetic intensity threshold
    ai = float(rec.get("arithmetic_intensity") or 0.0)
    strided = int(rec.get("strided_accesses") or 0)
    if strided > 0 and ai < 1.0:
        return "CPU"

    # Rule 4: Analytical cost comparison with tau = 1.20 margin
    t_cpu_ms, t_gpu_ms = estimate_times(rec, plat)
    speedup = t_cpu_ms / max(t_gpu_ms, 1e-9)

    if speedup >= plat.tau:
        return "GPU"
    return "CPU"


def evaluate_record(
    rec: Mapping[str, Any],
    plat: PlatformConstants | None = None,
) -> dict[str, Any]:
    """Comprehensive analytical evaluation report for a single loop record."""
    plat = plat or PlatformConstants()
    decision = heuristic_decision(rec, plat)
    t_cpu_ms, t_gpu_ms = estimate_times(rec, plat)
    speedup = t_cpu_ms / max(t_gpu_ms, 1e-9)

    reasons = []
    if not rec.get("gpu_safe", False):
        reasons.append("Loop-carried dependence violation (Unsafe)")
    if rec.get("trip_count_unknown", False):
        reasons.append("Unknown trip count (Conservative CPU stance)")
    if int(rec.get("strided_accesses", 0)) > 0 and float(rec.get("arithmetic_intensity", 0)) < 1.0:
        reasons.append("Un-coalesced memory stride with low arithmetic intensity (< 1.0)")
    if speedup < plat.tau:
        reasons.append(f"Predicted speedup ({speedup:.2f}x) does not clear margin tau={plat.tau:.2f}x")

    return {
        "loop_id": rec.get("loop_id", "unknown"),
        "decision": decision,
        "t_cpu_ms": round(t_cpu_ms, 4),
        "t_gpu_ms": round(t_gpu_ms, 4),
        "predicted_speedup": round(speedup, 3),
        "tau_threshold": plat.tau,
        "is_gpu_safe": bool(rec.get("gpu_safe", False)),
        "trip_count": int(rec.get("trip_count_eval") or 0),
        "reasons": reasons if decision == "CPU" else ["High compute density and coalesced memory access"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GPHO Analytical Heuristic Evaluator")
    parser.add_argument("input", type=Path, help="Path to JSON or JSONL file with loop metrics")
    parser.add_argument("--tau", type=float, default=1.20, help="Profitability speedup threshold")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file {args.input} does not exist.", file=sys.stderr)
        sys.exit(1)

    plat = PlatformConstants(tau=args.tau)
    content = args.input.read_text(encoding="utf-8").strip()

    records: list[dict[str, Any]] = []
    if content.startswith("["):
        records = json.loads(content)
    elif content.startswith("{"):
        records = [json.loads(content)]
    else:
        records = [json.loads(line) for line in content.splitlines() if line.strip()]

    print(f"--- GPHO Analytical Cost Baseline Evaluation (tau = {plat.tau:.2f}) ---")
    for idx, rec in enumerate(records):
        eval_res = evaluate_record(rec, plat)
        print(f"\n[Loop #{idx+1}]: {eval_res['loop_id']}")
        print(f"  Decision:           {eval_res['decision']}")
        print(f"  T_CPU (est):        {eval_res['t_cpu_ms']:.3f} ms")
        print(f"  T_GPU (est):        {eval_res['t_gpu_ms']:.3f} ms")
        print(f"  Predicted Speedup:  {eval_res['predicted_speedup']:.2f}x")
        print(f"  Diagnostics:        {'; '.join(eval_res['reasons'])}")


if __name__ == "__main__":
    main()
