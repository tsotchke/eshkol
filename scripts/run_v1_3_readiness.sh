#!/usr/bin/env bash
# Run the v1.3-evolve smoke probes and ask ICC for trace-aware readiness.
#
# Plain `icc readiness --repo eshkol_lang --target v1.3-evolve` does not know
# where this repository writes smoke traces. This wrapper is the canonical
# release check: refresh the trace, then pass scripts/icc_traces explicitly.
set -euo pipefail

cd "$(dirname "$0")/.."

ICC_BIN="${ICC_BIN:-/Users/tyr/Desktop/infinite_context_coder/bin/icc}"
ICC_REPO="${ICC_REPO:-eshkol_lang}"
TRACE_DIR="${TRACE_DIR:-scripts/icc_traces}"
ARCH_MODEL="${ARCH_MODEL:-.icc/architecture-model.yaml}"
ARCH_TRACE_GLOB="${ARCH_TRACE_GLOB:-.icc/runtime-traces-oracle-view/*architecture-model-verify-*.jsonl}"
READINESS_JSON="$(mktemp "${TMPDIR:-/tmp}/eshkol-v13-readiness.XXXXXX.json")"
: "${READINESS_JSON:?READINESS_JSON must be set}"
trap 'rm -f -- "${READINESS_JSON:?}"' EXIT

BUILD_DIR="${BUILD_DIR:-build}" TRACE_DIR="$TRACE_DIR" \
  scripts/run_mono_equiv_ad_taylor_gate.sh
scripts/run_icc_smoke.sh

# Emit a fresh, runtime-backed architecture verdict. Readiness consumes only
# the newest verdict so old red investigations remain auditable without
# poisoning the current release candidate.
"$ICC_BIN" architecture-verify \
  --repo "$ICC_REPO" \
  --model "$ARCH_MODEL" \
  --trace-dir "$TRACE_DIR" \
  --emit-trace \
  --format markdown

"$ICC_BIN" readiness \
  --repo "$ICC_REPO" \
  --target v1.3-evolve \
  --trace-dir "$TRACE_DIR" \
  --trace-latest "$ARCH_TRACE_GLOB" \
  --format json > "${READINESS_JSON:?}"

"$ICC_BIN" readiness \
  --repo "$ICC_REPO" \
  --target v1.3-evolve \
  --trace-dir "$TRACE_DIR" \
  --trace-latest "$ARCH_TRACE_GLOB" \
  --format markdown

status="$(jq -r '.status // ""' "$READINESS_JSON")"
if [ "$status" != "ready" ]; then
  echo "v1.3-evolve readiness is $status, expected ready" >&2
  exit 1
fi
