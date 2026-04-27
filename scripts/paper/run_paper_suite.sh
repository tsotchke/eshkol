#!/usr/bin/env bash
# run_paper_suite.sh — reproduce every number, table, and trace in the SDNC paper.
#
# Part of the artifact package for
#   "The Self-Differentiating Neural Computer: Computable Transformers
#    via Analytical Weight Construction" (tsotchke, 2026)
#
# Usage:
#   bash scripts/paper/run_paper_suite.sh           # full suite
#   bash scripts/paper/run_paper_suite.sh --quick   # skip heavy comparisons
#
# Expected wall time on 2023 M2 Max: under 5 minutes for full suite.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="$REPO_ROOT/artifacts/paper/outputs"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "SDNC Paper Artifact — Full Reproducibility Suite"
echo "=============================================="
echo "Repo HEAD:   $(git rev-parse HEAD)"
echo "Repo tag:    $(git describe --tags --always)"
echo "Output dir:  $OUTPUT_DIR"
echo "=============================================="
echo

echo "[1/4] Export weights + dump VM and matrix-forward traces (single run)..."
# A single weight_matrices invocation runs the 74-test suite once and emits
# both per-step traces. This is faster than calling dump_vm_trace.sh and
# dump_transformer_trace.sh separately (each of which runs the full suite).
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-paper}"
bash scripts/paper/export_weights.sh "$OUTPUT_DIR/weights.qlmw"

echo "    running 74-test suite with both trace flags..."
ESHKOL_WEIGHTS_OUT="$OUTPUT_DIR/weights.qlmw" \
"$BUILD_DIR/tools/weight_matrices" \
    --trace-vm "$OUTPUT_DIR/vm-traces.jsonl" \
    --trace-transformer "$OUTPUT_DIR/transformer-traces.jsonl" \
    > "$BUILD_DIR.suite_trace.log" 2>&1
if ! grep -q "74 passed, 0 failed" "$BUILD_DIR.suite_trace.log"; then
    echo "    ERROR: 74-test suite did not pass; tail:"
    tail -20 "$BUILD_DIR.suite_trace.log" | sed 's/^/      /'
    exit 1
fi
echo "    vm-traces:          $(wc -l < "$OUTPUT_DIR/vm-traces.jsonl" | tr -d ' ') lines"
echo "    transformer-traces: $(wc -l < "$OUTPUT_DIR/transformer-traces.jsonl" | tr -d ' ') lines"

echo "[2/4] Compare traces (fieldwise + ordinal output match)..."
python3 scripts/paper/compare_traces.py \
    --vm "$OUTPUT_DIR/vm-traces.jsonl" \
    --transformer "$OUTPUT_DIR/transformer-traces.jsonl" \
    --out "$OUTPUT_DIR/comparison-report.json" \
    --coverage-out "$OUTPUT_DIR/opcode-coverage.json"

echo "[3/4] Regenerate paper tables..."
mkdir -p "$OUTPUT_DIR/tables"
python3 scripts/paper/gen_paper_tables.py \
    --comparison "$OUTPUT_DIR/comparison-report.json" \
    --coverage "$OUTPUT_DIR/opcode-coverage.json" \
    --weights "$OUTPUT_DIR/weights.qlmw" \
    --out-dir "$OUTPUT_DIR/tables"

echo "[4/4] Done."
echo
echo "=============================================="
echo "Suite complete. Output checksums:"
echo "=============================================="
for f in "$OUTPUT_DIR"/weights.qlmw "$OUTPUT_DIR"/*.jsonl "$OUTPUT_DIR"/*.json; do
    if [[ -f "$f" ]]; then
        shasum -a 256 "$f"
    fi
done
echo

echo "Tables regenerated to: $OUTPUT_DIR/tables/"
ls -1 "$OUTPUT_DIR/tables/" 2>/dev/null || echo "  (no tables — gen_paper_tables.py may have failed)"
echo
echo "Done."
