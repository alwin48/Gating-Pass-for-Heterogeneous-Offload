#!/usr/bin/env python3
"""GPHO analytical heuristic cost model (Phase 9).

Platform defaults are documented; calibrate on target hardware when possible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PlatformConstants:
    """Default constants for an AWS g4dn-like / PCIe Gen4 x16 + mid GPU node."""

    w_simd: float = 8.0  # AVX-512 DP lanes
    f_cpu_ghz: float = 2.5
    c_cores: float = 8.0
    t_launch_us: float = 15.0
    b_pcie_gbs: float = 16.0
    n_cuda_cores: float = 2560.0  # T4-ish
    f_gpu_ghz: float = 1.5
    eta_occupancy: float = 0.65
    tau: float = 1.20


def estimate_times(rec: Mapping[str, Any], plat: PlatformConstants | None = None) -> tuple[float, float]:
    """Return (T_CPU_est_ms, T_GPU_est_ms)."""
    plat = plat or PlatformConstants()
    trip = int(rec.get("trip_count_eval") or 0)
    if rec.get("trip_count_unknown") or trip <= 0:
        # Unknown bounds → force CPU path with high GPU estimate.
        return 1.0, 1e9

    cpu_cost = float(rec.get("tti_cpu_cost") or (rec.get("fp_op_count", 0) + rec.get("int_op_count", 0)))
    gpu_cost = float(rec.get("tti_gpu_cost") or cpu_cost * 0.35)
    bytes_xfer = float(rec.get("transfer_bytes_est") or 0)

    # T_CPU ≈ N * cost / (SIMD * freq * cores)  → convert to ms
    denom_cpu = plat.w_simd * plat.f_cpu_ghz * 1e9 * plat.c_cores
    t_cpu_s = (trip * max(cpu_cost, 1.0)) / max(denom_cpu, 1.0)
    # Scale: TTI costs are relative; apply empirical µs/cost factor so ms are realistic.
    t_cpu_ms = t_cpu_s * 1e3 * 50.0  # calibration knob for relative TTI units

    t_launch_ms = plat.t_launch_us / 1000.0
    t_xfer_ms = (bytes_xfer / (plat.b_pcie_gbs * 1e9)) * 1e3
    denom_gpu = plat.n_cuda_cores * plat.f_gpu_ghz * 1e9 * plat.eta_occupancy
    t_kern_s = (trip * max(gpu_cost, 0.1)) / max(denom_gpu, 1.0)
    t_kern_ms = t_kern_s * 1e3 * 50.0
    t_gpu_ms = t_launch_ms + t_xfer_ms + t_kern_ms
    return max(t_cpu_ms, 1e-6), max(t_gpu_ms, 1e-6)


def heuristic_decision(rec: Mapping[str, Any], plat: PlatformConstants | None = None) -> str:
    """Return 'GPU' or 'CPU'."""
    plat = plat or PlatformConstants()
    if not rec.get("gpu_safe", False):
        return "CPU"
    if rec.get("trip_count_unknown", True):
        return "CPU"

    ai = float(rec.get("arithmetic_intensity") or 0.0)
    strided = int(rec.get("strided_accesses") or 0)
    if strided > 0 and ai < 1.0:
        return "CPU"

    t_cpu, t_gpu = estimate_times(rec, plat)
    if t_cpu > plat.tau * t_gpu:
        return "GPU"
    return "CPU"
