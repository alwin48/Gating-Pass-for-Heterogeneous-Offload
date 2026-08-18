#!/usr/bin/env bash
# Build the out-of-tree GPHO LLVM 18 pass plugin.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LLVM_CONFIG="${LLVM_CONFIG:-/usr/lib/llvm-18/bin/llvm-config}"
CXX="${CXX:-g++}"
OUT="${ROOT}/GPHOPass.so"
# shellcheck disable=SC2046
$CXX -shared -fPIC $($LLVM_CONFIG --cxxflags) -o "$OUT" "$ROOT/GPHOAnalysisPass.cpp" $($LLVM_CONFIG --ldflags) -Wl,-znodelete
echo "Built $OUT"
