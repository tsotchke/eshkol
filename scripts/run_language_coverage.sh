#!/usr/bin/env bash
# Run the deterministic language-surface tracker and publish fresh ICC evidence.
# The committed policy is a one-way ratchet: callers may raise the threshold,
# but language_coverage.py will never accept a value below the policy floor.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
TRACE_DIR=${ICC_TRACE_DIR:-"$REPO_ROOT/scripts/icc_traces"}
TRACE_FILE=${LANGUAGE_COVERAGE_TRACE:-"$TRACE_DIR/language_surface_coverage.jsonl"}

cd "$REPO_ROOT"
python3 scripts/gen_language_surface.py --check
python3 scripts/language_coverage.py \
    --trace "$TRACE_FILE" \
    "$@"
