#!/usr/bin/env python3
"""GPHO CLI: gate decision + Tree SHAP diagnostics for a feature JSON record."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from heuristic import estimate_times, heuristic_decision  # noqa: E402

FEATURE_NOTES = {
    "transfer_compute_ratio": "High Transfer Overhead",
    "strided_accesses": "Un-coalesced Memory",
    "arithmetic_intensity": "Compute Density",
    "trip_count_eval": "Loop Bounds",
    "trip_count_unknown": "Unknown Trip Count",
    "divergence_score": "SIMT Divergence Risk",
    "transfer_bytes_est": "Host-Device Transfer Volume",
    "fp_op_count": "FP Compute",
    "load_inst_count": "Memory Read Pressure",
    "store_inst_count": "Memory Write Pressure",
}


def load_model(path: Path):
    blob = joblib.load(path)
    return blob["model"], blob["features"], blob.get("alpha_fp", 3.5)


def shap_top(model, features: list[str], x_row: np.ndarray, k: int = 4) -> list[dict[str, Any]]:
    """Tree SHAP via XGBoost pred_contribs (avoids shap/xgboost version skew)."""
    boost = model.get_booster()
    import xgboost as xgb

    dmat = xgb.DMatrix(x_row.reshape(1, -1), feature_names=features)
    # Last column is bias; preceding columns are per-feature contributions.
    contribs = boost.predict(dmat, pred_contribs=True).reshape(-1)
    vals = contribs[:-1]
    order = np.argsort(-np.abs(vals))[:k]
    out = []
    for i in order:
        out.append(
            {
                "feature": features[i],
                "value": float(x_row[i]),
                "impact": float(vals[i]),
                "note": FEATURE_NOTES.get(features[i], ""),
            }
        )
    return out

def decide(rec: dict[str, Any], model, features: list[str], thresh: float = 0.50) -> dict[str, Any]:
    if not rec.get("gpu_safe", False):
        return {
            "decision": "CPU",
            "reason": "Not GPU-Safe (DependenceAnalysis)",
            "confidence": 1.0,
            "predicted_speedup": None,
            "shap_top_factors": [],
        }
    if rec.get("trip_count_unknown", False):
        return {
            "decision": "CPU",
            "reason": "Unknown trip count — conservative CPU stance",
            "confidence": 1.0,
            "predicted_speedup": None,
            "shap_top_factors": [],
        }

    x = np.array([float(rec.get(c, 0) if rec.get(c) is not None else 0) for c in features], dtype=float)
    # bools
    if "trip_count_unknown" in features:
        idx = features.index("trip_count_unknown")
        x[idx] = float(int(bool(rec.get("trip_count_unknown", False))))

    proba = float(model.predict_proba(x.reshape(1, -1))[0, 1])
    decision = "GPU" if proba >= thresh else "CPU"
    # Map probability mass above thresh to a crude speedup hint via heuristic times.
    t_cpu, t_gpu = estimate_times(rec)
    heur_s = t_cpu / max(t_gpu, 1e-9)
    # Blend: if ML says CPU, report min(heur, 1.0)-ish; if GPU, max(heur, tau).
    if decision == "GPU":
        pred_s = max(heur_s, 1.2) * (0.7 + 0.3 * proba)
    else:
        pred_s = min(heur_s, 1.15) * (1.2 - 0.4 * proba)

    factors = shap_top(model, features, x)
    return {
        "decision": decision,
        "reason": "ML asymmetric gate" if decision == "GPU" else "ML rejected offload",
        "confidence": proba if decision == "GPU" else 1.0 - proba,
        "predicted_speedup": round(pred_s, 3),
        "ml_proba_profitable": round(proba, 4),
        "shap_top_factors": factors,
        "decision_heuristic": heuristic_decision(rec),
    }


def format_report(rec: dict[str, Any], result: dict[str, Any]) -> str:
    loop = rec.get("loop_id", "?")
    src = rec.get("source_file", rec.get("benchmark_id", ""))
    lines = [
        "[GPHO Pass Optimization Report]",
        f"Target Loop: {loop} (benchmark: {src})",
    ]
    if result["decision"] == "GPU":
        lines.append(
            f"Decision: GPU OFF-LOADING ACCEPTED (Confidence: {100*result['confidence']:.1f}%)"
        )
    else:
        lines.append(
            f"Decision: GPU OFF-LOADING REJECTED (Confidence: {100*result['confidence']:.1f}%)"
        )
    if result.get("predicted_speedup") is not None:
        base = "GPU" if result["decision"] == "GPU" else "CPU Baseline Recommended"
        lines.append(f"Predicted Speedup: {result['predicted_speedup']:.2f}x ({base})")
    lines.append(f"Heuristic Decision: {result.get('decision_heuristic', 'n/a')}")
    lines.append("")
    lines.append("Primary Factor Drivers (Tree SHAP Analysis):")
    for fac in result.get("shap_top_factors") or []:
        sign = "+" if fac["impact"] >= 0 else "-"
        note = fac.get("note") or ""
        note_s = f" [{note}]" if note else ""
        lines.append(
            f"  [{sign}] {fac['feature']} ({fac['value']})  ==> {fac['impact']:+.3f} impact{note_s}"
        )
    if rec.get("speedup_actual") is not None:
        lines.append("")
        lines.append(
            f"Measured: T_CPU={rec['t_cpu_ms']:.2f} ms  T_GPU={rec['t_gpu_ms']:.2f} ms  "
            f"Speedup={rec['speedup_actual']:.2f}x  label_profitable={rec.get('label_profitable')}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="JSON object, JSON array, or JSONL file")
    ap.add_argument(
        "--model",
        type=Path,
        default=ROOT / "ml" / "artifacts" / "gpho_xgb.joblib",
    )
    ap.add_argument("--json-out", type=Path, help="Write machine-readable decision JSON")
    ap.add_argument("--index", type=int, default=0, help="Record index if input is a list")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Run: python ml/train_eval.py")

    model, features, _ = load_model(args.model)
    text = args.input.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        data = json.loads(text)
        rec = data[args.index]
    elif "\n" in text and not text.startswith("{"):
        lines = [json.loads(l) for l in text.splitlines() if l.strip()]
        rec = lines[args.index]
    else:
        # JSONL single-line or object
        try:
            rec = json.loads(text)
            if isinstance(rec, list):
                rec = rec[args.index]
        except json.JSONDecodeError:
            rec = json.loads(text.splitlines()[args.index])

    result = decide(rec, model, features)
    report = format_report(rec, result)
    print(report)
    if args.json_out:
        payload = {**rec, **result}
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
