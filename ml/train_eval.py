#!/usr/bin/env python3
"""Train asymmetric XGBoost gate + evaluate vs heuristic under LOBO-CV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from heuristic import heuristic_decision

ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLS = [
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

# Asymmetric weight: false positives (predict GPU when CPU) cost alpha× more.
ALPHA_FP = 2.5
PROBA_THRESH = 0.50


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    # Only GPU-safe loops enter ML labeling path; unsafe are forced CPU.
    df["trip_count_unknown"] = df["trip_count_unknown"].astype(int)
    df["gpu_safe"] = df["gpu_safe"].astype(bool)
    df["label_profitable"] = df["label_profitable"].astype(int)
    return df


def make_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].astype(float).to_numpy()
    y = df["label_profitable"].to_numpy().astype(int)
    return X, y


def train_xgb(X: np.ndarray, y: np.ndarray, seed: int = 42) -> xgb.XGBClassifier:
    # Asymmetry via sample_weight on negatives (not scale_pos_weight — that
    # crushed recall when combined with ALPHA_FP).
    clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=2,
    )
    w = np.where(y == 0, ALPHA_FP, 1.0).astype(float)
    clf.fit(X, y, sample_weight=w)
    return clf


def predict_gate(clf: xgb.XGBClassifier, X: np.ndarray, proba_thresh: float = PROBA_THRESH) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_labels, proba_positive)."""
    proba = clf.predict_proba(X)[:, 1]
    return (proba >= proba_thresh).astype(int), proba


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
        "fp_rate": float(fp / max(fp + tn, 1)),
        "n": int(len(y_true)),
    }


def lobo_cv(df: pd.DataFrame, seed: int = 42) -> list[dict[str, Any]]:
    """Leave-One-Benchmark-Out by suite (mutants stay with parent suite)."""
    suites = sorted(s for s in df["suite"].unique() if s != "demo")
    results = []
    for held in suites:
        train_df = df[df["suite"] != held]
        test_df = df[df["suite"] == held]
        # Filter: ML only on gpu_safe rows; unsafe → forced CPU pred.
        train_safe = train_df[train_df["gpu_safe"]].copy()
        if train_safe.empty or test_df.empty:
            continue
        Xtr, ytr = make_xy(train_safe)
        clf = train_xgb(Xtr, ytr, seed=seed)

        y_true = []
        y_ml = []
        y_heu = []
        for _, row in test_df.iterrows():
            true = int(row["label_profitable"])
            if not row["gpu_safe"] or row["trip_count_unknown"]:
                pred = 0
                proba = 0.0
            else:
                X = row[FEATURE_COLS].astype(float).to_numpy().reshape(1, -1)
                pred_arr, proba_arr = predict_gate(clf, X)
                pred = int(pred_arr[0])
                proba = float(proba_arr[0])
            heu = 1 if heuristic_decision(row.to_dict()) == "GPU" else 0
            y_true.append(true)
            y_ml.append(pred)
            y_heu.append(heu)

        y_true_a = np.array(y_true)
        fold = {
            "held_out_suite": held,
            "ml": metrics(y_true_a, np.array(y_ml)),
            "heuristic": metrics(y_true_a, np.array(y_heu)),
        }
        results.append(fold)
    return results


def ablation(df: pd.DataFrame, seed: int = 42) -> list[dict[str, Any]]:
    """Simple feature-subset ablation under LOBO (macro-avg precision)."""
    subsets = {
        "A_trip_only": ["trip_count_eval", "trip_count_unknown"],
        "B_trip_strides": [
            "trip_count_eval",
            "trip_count_unknown",
            "stride_one_accesses",
            "strided_accesses",
        ],
        "C_full_static": FEATURE_COLS,
    }
    out = []
    safe = df[df["gpu_safe"]].copy()
    suites = sorted(s for s in safe["suite"].unique() if s != "demo")
    for name, cols in subsets.items():
        precs, recalls, f1s, fps = [], [], [], []
        for held in suites:
            tr = safe[safe["suite"] != held]
            te = safe[safe["suite"] == held]
            if tr.empty or te.empty:
                continue
            Xtr = tr[cols].astype(float).to_numpy()
            ytr = tr["label_profitable"].to_numpy().astype(int)
            Xte = te[cols].astype(float).to_numpy()
            yte = te["label_profitable"].to_numpy().astype(int)
            clf = train_xgb(Xtr, ytr, seed=seed)
            # Re-fit with only subset columns via temporary model
            w = np.where(ytr == 0, ALPHA_FP, 1.0)
            clf = xgb.XGBClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.1,
                objective="binary:logistic",
                random_state=seed,
                n_jobs=2,
            )
            clf.fit(Xtr, ytr, sample_weight=w)
            pred = (clf.predict_proba(Xte)[:, 1] >= PROBA_THRESH).astype(int)
            m = metrics(yte, pred)
            precs.append(m["precision"])
            recalls.append(m["recall"])
            f1s.append(m["f1"])
            fps.append(m["fp_rate"])
        out.append(
            {
                "model": name,
                "features": cols,
                "lobo_precision": float(np.mean(precs) if precs else 0),
                "lobo_recall": float(np.mean(recalls) if recalls else 0),
                "lobo_f1": float(np.mean(f1s) if f1s else 0),
                "lobo_fp_rate": float(np.mean(fps) if fps else 0),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "dataset.jsonl",
    )
    ap.add_argument("--model-out", type=Path, default=ROOT / "ml" / "artifacts" / "gpho_xgb.joblib")
    ap.add_argument("--report-out", type=Path, default=ROOT / "ml" / "artifacts" / "lobo_report.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_jsonl(args.dataset)
    # Train final model on all non-demo gpu-safe rows.
    train_df = df[(df["suite"] != "demo") & (df["gpu_safe"])].copy()
    X, y = make_xy(train_df)
    clf = train_xgb(X, y, seed=args.seed)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURE_COLS, "alpha_fp": ALPHA_FP}, args.model_out)

    folds = lobo_cv(df, seed=args.seed)
    abl = ablation(df, seed=args.seed)

    def micro_from_folds(folds: list[dict], system: str) -> dict[str, float]:
        tp = sum(f[system]["tp"] for f in folds)
        fp = sum(f[system]["fp"] for f in folds)
        fn = sum(f[system]["fn"] for f in folds)
        tn = sum(f[system]["tn"] for f in folds)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        return {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "fp_rate": float(fp / max(fp + tn, 1)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    summary = {
        "alpha_fp": ALPHA_FP,
        "n_train": int(len(train_df)),
        "n_total": int(len(df)),
        "lobo_folds": folds,
        "ml_lobo_micro": micro_from_folds(folds, "ml"),
        "heuristic_lobo_micro": micro_from_folds(folds, "heuristic"),
        "ablation": abl,
        "note": "Prefer micro metrics over macro when some folds predict all-CPU (precision=0 artifact).",
    }
    args.report_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "lobo_folds"}, indent=2))
    print(f"Model → {args.model_out}")
    print(f"Report → {args.report_out}")


if __name__ == "__main__":
    main()
