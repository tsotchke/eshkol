#!/usr/bin/env bash
# Run the v1.3-evolve smoke probes and ask ICC for trace-aware readiness.
#
# Plain `icc readiness --repo eshkol_lang --target v1.3-evolve` does not know
# where this repository writes smoke traces. This wrapper is the canonical
# release check: refresh the trace, then pass scripts/icc_traces explicitly.
set -euo pipefail

cd "$(dirname "$0")/.."

ICC_BIN="${ICC_BIN:-/Users/tyr/Desktop/infinite_context_coder/bin/icc}"
TRACE_DIR="${TRACE_DIR:-scripts/icc_traces}"
READINESS_JSON="$(mktemp "${TMPDIR:-/tmp}/eshkol-v13-readiness.XXXXXX.json")"
: "${READINESS_JSON:?READINESS_JSON must be set}"
trap 'rm -f "$READINESS_JSON"' EXIT

scripts/run_icc_smoke.sh

"$ICC_BIN" readiness \
  --repo eshkol_lang \
  --target v1.3-evolve \
  --trace-dir "$TRACE_DIR" \
  --format json > "$READINESS_JSON"

"$ICC_BIN" readiness \
  --repo eshkol_lang \
  --target v1.3-evolve \
  --trace-dir "$TRACE_DIR" \
  --format markdown

status="$(jq -r '.status // ""' "$READINESS_JSON")"
if [ "$status" != "ready" ]; then
  echo "v1.3-evolve readiness is $status, expected ready" >&2
  exit 1
fi
