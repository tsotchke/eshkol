#!/usr/bin/env bash
# run_sdnc_finetune_gate.sh — full gate for exact-forward control fine-tuning
# of the constructed SDNC interpreter (docs/design/adr/0032-exact-forward-control-fine-tuning.md).
#
# Runs, against an existing build:
#   1. weight_matrices --control-finetune --artifacts DIR
#      records every transition of the 127-program verifier battery, damages
#      the ADD dispatch, checks that the exact control derivative is
#      identically zero, repairs with smoothed-backward-ranked /
#      exact-forward-accepted threshold moves, then requires the full battery
#      and every recorded transition byte for byte;
#   2. the sdnc_finetune_unit_test binary (VJP rule, bit-identity, tied gates,
#      file formats, synthetic repair);
#   3. the core.sdnc surface test (tests/sdnc/sdnc_finetune_test.esk) through
#      eshkol-run -r on the artifacts of step 1.
#
# Writes ICC runtime events to scripts/icc_traces/sdnc_finetune.jsonl:
#   {"kind":"sdnc_control_finetune","name":"battery_repair"|"unit"|"surface",
#    "value":"PASS"|"FAIL","snippet":"..."}
# and the measured numbers line to stdout.
#
# Usage: BUILD_DIR=build bash scripts/run_sdnc_finetune_gate.sh
set -u
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in /*) BUILD="$BUILD_DIR" ;; *) BUILD="$REPO_ROOT/$BUILD_DIR" ;; esac
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
OUT="$TRACE_DIR/sdnc_finetune.jsonl"
ART="$BUILD/sdnc_finetune"
mkdir -p "$TRACE_DIR" "$ART" "$BUILD/sdnc_finetune_unit"
: > "$OUT"

emit() {  # name verdict snippet
    python3 - "$OUT" "$1" "$2" "$3" <<'PY'
import json, sys
out, name, verdict, snippet = sys.argv[1:5]
with open(out, "a") as f:
    f.write(json.dumps({"kind": "sdnc_control_finetune", "name": name,
                        "value": verdict, "snippet": snippet,
                        "confidence": 1.0}) + "\n")
PY
}

status=0

WM="$BUILD/tools/weight_matrices"
LOG="$ART/control_finetune.log"
if [ -x "$WM" ] && "$WM" --control-finetune --artifacts "$ART" > "$LOG" 2>&1 &&
   grep -q "Control fine-tune: 11 passed, 0 failed" "$LOG"; then
    emit battery_repair PASS "$(grep '^SDNC_FINETUNE ' "$LOG" | sed 's/^SDNC_FINETUNE //')"
else
    emit battery_repair FAIL "$(tail -3 "$LOG" 2>/dev/null | tr '\n' ' ')"
    status=1
fi
grep '^SDNC_FINETUNE ' "$LOG" 2>/dev/null

UNIT="$BUILD/sdnc_finetune_unit_test"
ULOG="$ART/unit.log"
if [ -x "$UNIT" ] && "$UNIT" "$BUILD/sdnc_finetune_unit" > "$ULOG" 2>&1 &&
   grep -q "Results: 21 passed, 0 failed" "$ULOG"; then
    emit unit PASS "$(grep 'Results:' "$ULOG")"
else
    emit unit FAIL "$(tail -2 "$ULOG" 2>/dev/null | tr '\n' ' ')"
    status=1
fi

RUN="$BUILD/eshkol-run"
SLOG="$ART/surface.log"
if [ -x "$RUN" ] && [ -f "$ART/transitions.sdtr" ] &&
   ESHKOL_SDNC_ARTIFACTS="$ART" "$RUN" -r "$REPO_ROOT/tests/sdnc/sdnc_finetune_test.esk" "-L$BUILD" > "$SLOG" 2>&1 &&
   grep -q "Results: 16 passed, 0 failed" "$SLOG"; then
    emit surface PASS "$(grep 'Results:' "$SLOG")"
else
    emit surface FAIL "$(tail -2 "$SLOG" 2>/dev/null | tr '\n' ' ')"
    status=1
fi

echo "SDNC fine-tune gate trace written: $OUT (status $status)"
exit $status
