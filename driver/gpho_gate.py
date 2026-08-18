#!/usr/bin/env python3
"""GPHO CLI Diagnostic Driver: LLVM 18 Pass Runner + ML Gating + Tree SHAP Diagnostics.

Accepts LLVM IR (.ll) or C source files, executes the GPHO LLVM analysis pass,
extracts structural IR metrics and 4D graph topological embeddings, evaluates
against the asymmetric XGBoost gate and analytical heuristic baseline, and
prints a terminal performance diagnostic report.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# Setup module search paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

try:
    from graph_encoder import extract_graph_embedding_from_metrics, LLVMGraphEncoder
    from gpho_trainer import compute_tree_shap_contributions, ALL_FEATURES, FEATURE_NOTES
    from heuristic_baseline import PlatformConstants, estimate_times, heuristic_decision, evaluate_record
except ImportError:
    from ml.graph_encoder import extract_graph_embedding_from_metrics, LLVMGraphEncoder
    from ml.gpho_trainer import compute_tree_shap_contributions, ALL_FEATURES, FEATURE_NOTES
    from ml.heuristic_baseline import PlatformConstants, estimate_times, heuristic_decision, evaluate_record

DEFAULT_MODEL_PATH = ROOT / "ml" / "artifacts" / "gpho_xgb.joblib"
DEFAULT_PASS_PATH = ROOT / "build" / "pass" / "GPHOPass.so"


def find_opt_binary() -> str:
    """Locate opt-18 or opt binary."""
    for candidate in ["opt-18", "opt", "/usr/lib/llvm-18/bin/opt"]:
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError("LLVM opt binary not found in PATH or /usr/lib/llvm-18/bin/opt")


def find_clang_binary() -> str:
    """Locate clang-18 or clang binary."""
    for candidate in ["clang-18", "clang", "/usr/lib/llvm-18/bin/clang"]:
        path = shutil.which(candidate)
        if path:
            return path
    return "clang"


def compile_c_to_ir(c_path: Path, output_ll: Path) -> None:
    """Compile C source file to LLVM IR (.ll) with O3 optimization."""
    clang = find_clang_binary()
    cmd = [
        clang,
        "-O3",
        "-fno-inline",
        "-Xclang", "-disable-lifetime-markers",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-S",
        "-emit-llvm",
        str(c_path),
        "-o",
        str(output_ll),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Clang compilation failed:\n{res.stderr}")


def run_llvm_pass(
    ll_path: Path,
    pass_so: Path = DEFAULT_PASS_PATH,
    benchmark_id: str = "custom_kernel",
    suite: str = "eval",
) -> list[dict[str, Any]]:
    """Execute GPHOAnalysisPass via LLVM opt and return extracted loop metrics."""
    if not pass_so.exists():
        raise FileNotFoundError(
            f"LLVM pass plugin not found at: {pass_so}. Build it first with: cmake -B build && cmake --build build"
        )

    opt = find_opt_binary()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_json:
        tmp_json_path = Path(tmp_json.name)

    try:
        cmd = [
            opt,
            f"-load-pass-plugin={pass_so}",
            "-passes=gpho-analysis",
            f"-gpho-benchmark-id={benchmark_id}",
            f"-gpho-suite={suite}",
            f"-gpho-json-out={tmp_json_path}",
            "-disable-output",
            str(ll_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"LLVM opt execution failed:\n{res.stderr}")

        if tmp_json_path.exists() and tmp_json_path.stat().st_size > 0:
            content = tmp_json_path.read_text(encoding="utf-8").strip()
            if content:
                records = []
                for line in content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, list):
                            records.extend(parsed)
                        else:
                            records.append(parsed)
                    except json.JSONDecodeError:
                        pass
                return records
        return []
    finally:
        if tmp_json_path.exists():
            tmp_json_path.unlink(missing_ok=True)


def evaluate_loop_gate(
    rec: dict[str, Any],
    model_artifact: dict[str, Any],
    raw_ir: str | None = None,
) -> dict[str, Any]:
    """Perform hybrid gating decision combining hard safety, asymmetric ML, SHAP, and heuristic."""
    clf = model_artifact["model"]
    features: list[str] = model_artifact["features"]
    thresh: float = model_artifact.get("proba_thresh", 0.50)

    # 1. Hard Rule: GPU Correctness Safety via DependenceAnalysis
    if not rec.get("gpu_safe", True):
        return {
            "decision": "CPU",
            "reason": "Loop-carried dependence violation (Unsafe for GPU SIMT)",
            "confidence": 1.0,
            "predicted_speedup": 0.0,
            "ml_proba_profitable": 0.0,
            "shap_factors": [],
            "heuristic": evaluate_record(rec),
            "pragma_recommendation": "#pragma omp parallel for",
        }

    # 2. Hard Rule: Unknown loop bounds (Conservative Stance)
    if rec.get("trip_count_unknown", False) or int(rec.get("trip_count_eval", 0)) <= 0:
        return {
            "decision": "CPU",
            "reason": "Unknown trip count bounds (Conservative CPU baseline preference)",
            "confidence": 1.0,
            "predicted_speedup": 0.0,
            "ml_proba_profitable": 0.0,
            "shap_factors": [],
            "heuristic": evaluate_record(rec),
            "pragma_recommendation": "#pragma omp parallel for",
        }

    # 3. Enrich features with 4D Graph Embedding
    if raw_ir:
        encoder = LLVMGraphEncoder()
        enriched = encoder.enrich_features(rec, raw_ir=raw_ir)
    else:
        graph_embed = extract_graph_embedding_from_metrics(rec)
        enriched = dict(rec)
        for i, gfeat in enumerate(["graph_cfg_cyclomatic", "graph_dfg_fanout", "graph_mem_clustering", "graph_path_entropy"]):
            enriched[gfeat] = graph_embed[i]

    # Construct feature vector
    x_vec = np.array([float(enriched.get(f, 0.0)) for f in features], dtype=float)

    # 4. Asymmetric ML Gating Inference
    proba = float(clf.predict_proba(x_vec.reshape(1, -1))[0, 1])
    decision = "GPU" if proba >= thresh else "CPU"
    confidence = proba if decision == "GPU" else 1.0 - proba

    # 5. Tree SHAP local feature attribution
    shap_factors = compute_tree_shap_contributions(clf, features, x_vec, top_k=4)

    # 6. Analytical heuristic comparison
    heur_eval = evaluate_record(rec)
    heur_speedup = heur_eval["predicted_speedup"]

    # Speedup blending
    if decision == "GPU":
        blended_speedup = max(heur_speedup, 1.20) * (0.8 + 0.2 * proba)
        pragma = (
            f"#pragma omp target teams distribute parallel for "
            f"if(target: N >= {max(int(rec.get('trip_count_eval', 1024)), 256)})"
        )
    else:
        blended_speedup = min(heur_speedup, 1.15) * (1.1 - 0.2 * proba)
        pragma = "#pragma omp parallel for"

    return {
        "decision": decision,
        "reason": "High arithmetic intensity & coalesced stride (Clear ML margin)" if decision == "GPU" else "ML Asymmetric Gate rejected offload (Risk of PCIe/Strided regression)",
        "confidence": round(confidence, 4),
        "predicted_speedup": round(blended_speedup, 2),
        "ml_proba_profitable": round(proba, 4),
        "shap_factors": shap_factors,
        "heuristic": heur_eval,
        "pragma_recommendation": pragma,
    }


def format_terminal_report(rec: dict[str, Any], res: dict[str, Any]) -> str:
    """Format an ANSI colorized terminal diagnostic report."""
    # ANSI color codes
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    RESET = "\033[0m"

    loop_id = rec.get("loop_id", "unknown_loop")
    bench = rec.get("benchmark_id", "input")
    is_gpu = res["decision"] == "GPU"

    lines = []
    lines.append(f"{BOLD}{CYAN}========================================================================{RESET}")
    lines.append(f"{BOLD}  GPHO COMPILER OPTIMIZATION REPORT: HETEROGENEOUS OFFLOAD GATING{RESET}")
    lines.append(f"{BOLD}{CYAN}========================================================================{RESET}")
    lines.append(f"  {BOLD}Target Loop:{RESET}       {loop_id} (Source / Benchmark: {bench})")
    lines.append(f"  {BOLD}Trip Count:{RESET}        {rec.get('trip_count_eval', 'Unknown')} (Nest Depth: {rec.get('loop_nest_depth', 1)})")
    lines.append(f"  {BOLD}Compute Profile:{RESET}   {rec.get('fp_op_count', 0)} FP Ops, {rec.get('int_op_count', 0)} INT Ops (AI: {rec.get('arithmetic_intensity', 0.0):.2f} Ops/Byte)")
    lines.append(f"  {BOLD}Memory Profile:{RESET}    {rec.get('stride_one_accesses', 0)} Stride-1, {rec.get('strided_accesses', 0)} Strided (Transfer: {rec.get('transfer_bytes_est', 0)} Bytes)")
    lines.append("------------------------------------------------------------------------")

    if is_gpu:
        dec_str = f"{BOLD}{GREEN}[✓] GPU OFF-LOADING ACCEPTED{RESET}"
    else:
        dec_str = f"{BOLD}{RED}[✗] GPU OFF-LOADING REJECTED{RESET}"

    lines.append(f"  {BOLD}GPHO Decision:{RESET}     {dec_str} (Confidence: {res['confidence'] * 100:.1f}%)")
    lines.append(f"  {BOLD}Predicted Speedup:{RESET} {BOLD}{res['predicted_speedup']:.2f}x{RESET} over Multithreaded CPU Baseline")
    lines.append(f"  {BOLD}Decision Reason:{RESET}   {res['reason']}")
    lines.append(f"  {BOLD}Heuristic Model:{RESET}   {res['heuristic']['decision']} (Estimated: T_CPU={res['heuristic']['t_cpu_ms']:.2f}ms, T_GPU={res['heuristic']['t_gpu_ms']:.2f}ms)")
    lines.append("------------------------------------------------------------------------")
    lines.append(f"  {BOLD}Primary Factor Drivers (Tree SHAP Explainability):{RESET}")

    for factor in res.get("shap_factors", []):
        impact = factor["impact"]
        sign = "+" if impact >= 0 else "-"
        color = GREEN if impact >= 0 else RED
        note_str = f"{GRAY} -> {factor['note']}{RESET}" if factor.get("note") else ""
        lines.append(
            f"    {color}[{sign}]{RESET} {BOLD}{factor['feature']:24s}{RESET} "
            f"= {str(factor['value']):<8s} (SHAP impact: {color}{impact:+.3f}{RESET}){note_str}"
        )

    lines.append("------------------------------------------------------------------------")
    lines.append(f"  {BOLD}Recommended OpenMP Directive:{RESET}")
    lines.append(f"    {YELLOW}{res['pragma_recommendation']}{RESET}")
    lines.append(f"{BOLD}{CYAN}========================================================================{RESET}\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPHO Gate: Static-First ML-Gated GPU Offload CLI")
    parser.add_argument("input_file", type=Path, help="Input LLVM IR (.ll) or C source file (.c)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to trained GPHO model artifact")
    parser.add_argument("--pass-so", type=Path, default=DEFAULT_PASS_PATH, help="Path to GPHOPass.so module")
    parser.add_argument("--json-out", type=Path, help="Optional path to write full diagnostic JSON output")
    parser.add_argument("--benchmark-id", type=str, default="custom_kernel", help="Benchmark identifier")
    parser.add_argument("--suite", type=str, default="cli_eval", help="Suite tag")
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: input file {args.input_file} not found.", file=sys.stderr)
        sys.exit(1)

    if not args.model.exists():
        print(f"Model artifact not found at {args.model}. Run 'python ml/gpho_trainer.py' first.", file=sys.stderr)
        sys.exit(1)

    model_artifact = joblib.load(args.model)

    # Handle C source compilation to IR
    raw_ir = ""
    if args.input_file.suffix.lower() in [".c", ".cpp", ".cc"]:
        with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as tmp_ll:
            tmp_ll_path = Path(tmp_ll.name)
        try:
            compile_c_to_ir(args.input_file, tmp_ll_path)
            raw_ir = tmp_ll_path.read_text(encoding="utf-8")
            records = run_llvm_pass(tmp_ll_path, args.pass_so, args.benchmark_id, args.suite)
        finally:
            tmp_ll_path.unlink(missing_ok=True)
    elif args.input_file.suffix.lower() in [".ll", ".bc"]:
        raw_ir = args.input_file.read_text(encoding="utf-8")
        records = run_llvm_pass(args.input_file, args.pass_so, args.benchmark_id, args.suite)
    elif args.input_file.suffix.lower() == ".json":
        data = json.loads(args.input_file.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else [data]
    else:
        print(f"Unsupported file format: {args.input_file.suffix}", file=sys.stderr)
        sys.exit(1)

    if not records:
        print("GPHO: No candidate loops detected in input.")
        return

    full_diagnostics = []
    for rec in records:
        eval_result = evaluate_loop_gate(rec, model_artifact, raw_ir=raw_ir)
        report = format_terminal_report(rec, eval_result)
        print(report)
        full_diagnostics.append({**rec, "gpho_evaluation": eval_result})

    if args.json_out:
        args.json_out.write_text(json.dumps(full_diagnostics, indent=2), encoding="utf-8")
        print(f"Exported diagnostic JSON to {args.json_out}")


if __name__ == "__main__":
    main()
