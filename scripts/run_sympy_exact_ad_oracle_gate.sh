#!/usr/bin/env bash
# SymPy differential oracle on the exact-AD surface (JIT + AOT).
# Emits ICC runtime_event traces into $TRACE_DIR/sympy_exact_ad_oracle.jsonl:
# one per case and engine, plus the summary `sympy_exact_ad_oracle_pilot`.
# Set ESHKOL_SYMPY_PYTHON to an interpreter that can import SymPy; without
# one the summary records SKIP with the reason and the script exits 77.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
ESHKOL_RUN="${ESHKOL_RUN:-$BUILD_DIR/eshkol-run}"
if [ ! -x "$ESHKOL_RUN" ]; then
  echo "INFRA: eshkol-run not found at $ESHKOL_RUN (set BUILD_DIR or ESHKOL_RUN)" >&2
  exit 2
fi
mkdir -p "$TRACE_DIR"
export ESHKOL_PATH="${ESHKOL_PATH:-$REPO_ROOT/lib}"
export ESHKOL_LIB_DIR="${ESHKOL_LIB_DIR:-$BUILD_DIR}"
exec python3 "$REPO_ROOT/scripts/sympy_exact_ad_oracle.py" \
  --eshkol-run "$ESHKOL_RUN" --lib-dir "$BUILD_DIR" \
  --trace-dir "$TRACE_DIR" --work-dir "$BUILD_DIR/sympy-oracle-work" "$@"
