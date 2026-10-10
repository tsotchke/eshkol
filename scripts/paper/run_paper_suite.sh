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
# --quick still exports the weights, still runs the full verification suite
# with both trace flags (so every number in the paper still gets re-proved),
# and still writes vm-traces.jsonl / transformer-traces.jsonl. It skips the
# fieldwise VM-vs-transformer trace comparison (compare_traces.py) and the
# paper-table regeneration that consumes that comparison's output, since
# those are the two steps whose cost scales with trace size rather than with
# the fixed 71-program suite.
#
# Expected wall time on 2023 M2 Max: under 5 minutes for full suite.

set -euo pipefail

QUICK=0
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        *)
            echo "usage: $0 [--quick]" >&2
            exit 2
            ;;
    esac
done

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

if [[ "$QUICK" -eq 1 ]]; then
    STEP_TOTAL=2
else
    STEP_TOTAL=4
fi

echo "[1/$STEP_TOTAL] Export weights + dump VM and matrix-forward traces (single run)..."
# A single weight_matrices invocation runs the verification suite once and emits
# both per-step traces. This is faster than calling dump_vm_trace.sh and
# dump_transformer_trace.sh separately (each of which runs the full suite).
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-paper}"
bash scripts/paper/export_weights.sh "$OUTPUT_DIR/weights.qlmw"

echo "    running verification suite with both trace flags..."
ESHKOL_WEIGHTS_OUT="$OUTPUT_DIR/weights.qlmw" \
"$BUILD_DIR/tools/weight_matrices" \
    --trace-vm "$OUTPUT_DIR/vm-traces.jsonl" \
    --trace-transformer "$OUTPUT_DIR/transformer-traces.jsonl" \
    > "$BUILD_DIR.suite_trace.log" 2>&1
result_line="$(grep -E "=== Results: [0-9]+ passed, 0 failed ===" "$BUILD_DIR.suite_trace.log" | tail -1 || true)"
if [[ -z "$result_line" ]]; then
    echo "    ERROR: verification suite did not pass; tail:"
    tail -20 "$BUILD_DIR.suite_trace.log" | sed 's/^/      /'
    exit 1
fi
passed="$(printf '%s\n' "$result_line" | sed -E 's/.*Results: ([0-9]+) passed, 0 failed.*/\1/')"
echo "    $passed/$passed verification passes with trace flags."
echo "    vm-traces:          $(wc -l < "$OUTPUT_DIR/vm-traces.jsonl" | tr -d ' ') lines"
echo "    transformer-traces: $(wc -l < "$OUTPUT_DIR/transformer-traces.jsonl" | tr -d ' ') lines"

if [[ "$QUICK" -eq 1 ]]; then
    echo "[2/$STEP_TOTAL] --quick: skipping compare_traces.py and paper-table regeneration."
    echo
    echo "=============================================="
    echo "Quick suite complete (verification + traces only). Output checksums:"
    echo "=============================================="
    for f in "$OUTPUT_DIR"/weights.qlmw "$OUTPUT_DIR"/*.jsonl; do
        if [[ -f "$f" ]]; then
            shasum -a 256 "$f"
        fi
    done
    echo
    echo "Re-run without --quick for the fieldwise trace comparison and regenerated paper tables."
    echo "Done."
    exit 0
fi

echo "[2/$STEP_TOTAL] Compare traces (fieldwise + ordinal output match)..."
python3 scripts/paper/compare_traces.py \
    --vm "$OUTPUT_DIR/vm-traces.jsonl" \
    --transformer "$OUTPUT_DIR/transformer-traces.jsonl" \
    --out "$OUTPUT_DIR/comparison-report.json" \
    --coverage-out "$OUTPUT_DIR/opcode-coverage.json"

echo "[3/$STEP_TOTAL] Regenerate paper tables..."
mkdir -p "$OUTPUT_DIR/tables"
python3 scripts/paper/gen_paper_tables.py \
    --comparison "$OUTPUT_DIR/comparison-report.json" \
    --coverage "$OUTPUT_DIR/opcode-coverage.json" \
    --weights "$OUTPUT_DIR/weights.qlmw" \
    --out-dir "$OUTPUT_DIR/tables"

echo "[4/$STEP_TOTAL] Done."
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
