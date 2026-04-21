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

echo "[1/5] Export weights from ISA specification..."
bash scripts/paper/export_weights.sh "$OUTPUT_DIR/weights.qlmw"

echo "[2/5] Dump reference VM traces..."
bash scripts/paper/dump_vm_trace.sh "$OUTPUT_DIR/vm-traces.jsonl"

echo "[3/5] Dump compiled-transformer traces..."
bash scripts/paper/dump_transformer_trace.sh "$OUTPUT_DIR/transformer-traces.jsonl"

echo "[4/5] Compare traces (fieldwise exact)..."
python3 scripts/paper/compare_traces.py \
    --vm "$OUTPUT_DIR/vm-traces.jsonl" \
    --transformer "$OUTPUT_DIR/transformer-traces.jsonl" \
    --out "$OUTPUT_DIR/comparison-report.json" \
    --coverage-out "$OUTPUT_DIR/opcode-coverage.json"

echo "[5/5] Regenerate paper tables..."
mkdir -p "$OUTPUT_DIR/tables"
python3 scripts/paper/gen_paper_tables.py \
    --comparison "$OUTPUT_DIR/comparison-report.json" \
    --coverage "$OUTPUT_DIR/opcode-coverage.json" \
    --weights "$OUTPUT_DIR/weights.qlmw" \
    --out-dir "$OUTPUT_DIR/tables"

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
