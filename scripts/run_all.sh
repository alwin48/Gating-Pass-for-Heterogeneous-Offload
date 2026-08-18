#!/usr/bin/env bash
# End-to-end: build pass → generate dataset → train → demo
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"

bash "$ROOT/llvm-pass/build.sh"
"$PY" "$ROOT/ml/generate_dataset.py" -n 100 -o "$ROOT/data/dataset.jsonl"
"$PY" "$ROOT/ml/build_trace_db.py"
"$PY" "$ROOT/harness/profile.py" --dataset "$ROOT/data/dataset.jsonl" --out "$ROOT/data/dataset_profiled.jsonl"
"$PY" "$ROOT/ml/train_eval.py" --dataset "$ROOT/data/dataset.jsonl"
"$PY" "$ROOT/ml/plot_eval.py"
bash "$ROOT/scripts/demo.sh"
echo "OK: full pipeline complete"
