#!/usr/bin/env python3
"""Generate evaluation charts (Phase 25) from dataset + LOBO report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ds = ROOT / "data" / "dataset.jsonl"
    report = json.loads((ROOT / "ml" / "artifacts" / "lobo_report.json").read_text())
    rows = [json.loads(l) for l in ds.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)

    out_dir = ROOT / "ml" / "artifacts" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Speedup distribution (actual) — proxy correlation panel
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    safe = df[df["gpu_safe"] == True]  # noqa: E712
    ax.scatter(safe["trip_count_eval"], safe["speedup_actual"], alpha=0.6, s=20)
    ax.axhline(1.2, color="red", linestyle="--", label="tau=1.2")
    ax.set_xscale("log")
    ax.set_xlabel("trip_count_eval")
    ax.set_ylabel("speedup_actual")
    ax.set_title("Speedup vs Trip Count")
    ax.legend()

    # 2. Confusion-style bars from micro LOBO
    ax = axes[0, 1]
    ml = report["ml_lobo_micro"]
    he = report["heuristic_lobo_micro"]
    labels = ["TP", "FP", "FN", "TN"]
    ml_v = [ml["tp"], ml["fp"], ml["fn"], ml["tn"]]
    he_v = [he["tp"], he["fp"], he["fn"], he["tn"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, ml_v, 0.4, label="ML (asymmetric)")
    ax.bar(x + 0.2, he_v, 0.4, label="Heuristic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("LOBO Micro Confusion Counts")
    ax.legend()

    # 3. Ablation F1
    ax = axes[1, 0]
    abl = report["ablation"]
    names = [a["model"] for a in abl]
    f1s = [a["lobo_f1"] for a in abl]
    ax.barh(names, f1s)
    ax.set_xlim(0, 1)
    ax.set_title("Feature Ablation LOBO F1")
    ax.set_xlabel("F1")

    # 4. Cumulative runtime: always-GPU vs gated
    ax = axes[1, 1]
    always_gpu = df["t_gpu_ms"].sum()
    # Gated: use GPU iff label_profitable else CPU (oracle gate) AND ML would need model;
    # Approximate gated with profitable→GPU else CPU.
    gated = np.where(df["label_profitable"], df["t_gpu_ms"], df["t_cpu_ms"]).sum()
    always_cpu = df["t_cpu_ms"].sum()
    ax.bar(["Always CPU", "Always GPU", "Oracle Gate"], [always_cpu, always_gpu, gated])
    ax.set_ylabel("Cumulative ms")
    ax.set_title("Suite Runtime (oracle gate proxy)")

    fig.tight_layout()
    path = out_dir / "eval_panel.png"
    fig.savefig(path, dpi=120)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
