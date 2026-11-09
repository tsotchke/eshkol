#!/usr/bin/env bash
#
# core.sblc acceptance gate (ADR-0017, Stochastic Binary Lambda Calculus,
# first slice).
#
# Runs tests/stdlib/sblc_test.esk under BOTH execution modes:
#   1. JIT : eshkol-run -r <test>
#   2. AOT : eshkol-run -o <bin> <test> && <bin>
#
# The suite is self-asserting and prints one "SBLC-GATE <oracle> PASS|FAIL"
# line per oracle (codec, mass conservation, coin traces, semantics, sampler
# statistics, conditioning receipts) plus an explicit "RESULT: PASS". This
# gate additionally requires the two engines' standard output to be
# byte-identical: it contains the full conditioning receipts and a seeded
# sample trace, so equal bytes are the fixed-seed JIT/AOT parity check.
#
# Emits ICC runtime_event traces (kind "sblc") to $TRACE_DIR/sblc.jsonl:
#   sblc_codec_oracle, sblc_mass_conservation, sblc_coin_trace_oracle,
#   sblc_semantics, sblc_sampler_statistics, sblc_conditioning_receipt
#     PASS only when the oracle line is PASS on BOTH engines;
#   sblc_jit_aot_parity
#     PASS only when both engines pass and their stdout is byte-identical;
#   sblc_first_slice
#     PASS only when every event above is PASS.
#
# Environment:
#   BUILD_DIR  build directory holding eshkol-run (default: build)
#   TRACE_DIR  trace directory (default: scripts/icc_traces)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
RUN="$BUILD_DIR/eshkol-run"
TEST="tests/stdlib/sblc_test.esk"

# Resolve (require core.sblc) against THIS checkout's lib/.
export ESHKOL_PATH="$PROJECT_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"
export ESHKOL_JIT_CACHE=0

if [ ! -x "$RUN" ]; then
    echo "ERROR: $RUN not found or not executable (set BUILD_DIR?)." >&2
    exit 2
fi

ESHKOL_TEST_ISOLATION_NO_TRAP=1
ESHKOL_TEST_LIB="$SCRIPT_DIR/lib/test_isolation.sh"
if [ ! -r "$ESHKOL_TEST_LIB" ]; then
    echo "FATAL: cannot read $ESHKOL_TEST_LIB" >&2
    echo "       (the shared test isolation and failure-detection prelude)." >&2
    echo "       Refusing to run: without it this gate would report a" >&2
    echo "       meaningless PASS." >&2
    exit 2
fi
source "$ESHKOL_TEST_LIB"
# shellcheck source=lib/checked_write.sh
. "$(dirname "$ESHKOL_TEST_LIB")/checked_write.sh"
eshkol_test_isolation_init "sblc-gate"
trap eshkol_test_isolation_cleanup EXIT

TRACE_DIR="${TRACE_DIR:-$PROJECT_ROOT/scripts/icc_traces}"
. "$SCRIPT_DIR/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$PROJECT_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/sblc.jsonl"
eshkol_require_output_file_path "$TRACE_FILE"
: > "$TRACE_FILE"

emit_event() { # name PASS|FAIL snippet
    eshkol_require_output_file_path "$TRACE_FILE"
    python3 -c '
import json, sys
print(json.dumps({"kind": "sblc", "name": sys.argv[1], "value": sys.argv[2],
                  "snippet": sys.argv[3], "confidence": 0.95}, ensure_ascii=False))
' "$1" "$2" "$3" >> "$TRACE_FILE"
}

JIT_OUT="$ESHKOL_TEST_TMPDIR/sblc.jit.out"
JIT_ERR="$ESHKOL_TEST_TMPDIR/sblc.jit.err"
AOT_OUT="$ESHKOL_TEST_TMPDIR/sblc.aot.out"
AOT_ERR="$ESHKOL_TEST_TMPDIR/sblc.aot.err"
AOT_BIN="$ESHKOL_TEST_TMPDIR/sblc_aot"

# $1 label, $2 stdout file, $3 exit status. Returns 0 when the run passed.
sblc_check_output() {
    local label="$1" out="$2" rc="$3"
    if [ "$rc" -ne 0 ]; then
        echo "$label: FAIL (exit status $rc)"
        return 1
    fi
    if eshkol_test_output_has_failure "$out"; then
        echo "$label: FAIL (test reported failures)"
        eshkol_test_output_failures "$out" "" 10 | sed 's/^/  /'
        return 1
    fi
    if eshkol_test_output_is_silent "$out"; then
        echo "$label: FAIL (no output; absence of a verdict is not a pass)"
        return 1
    fi
    if ! grep -qx 'RESULT: PASS' "$out"; then
        echo "$label: FAIL (missing explicit RESULT: PASS verdict)"
        return 1
    fi
    echo "$label: PASS"
    return 0
}

echo "========================================="
echo "  core.sblc acceptance gate (ADR-0017)"
echo "========================================="

echo ""
echo "--- $TEST: JIT (-r) ----------------------"
eshkol_require_output_file_path "$JIT_OUT"
"$RUN" -r "$TEST" > "$JIT_OUT" 2> "$JIT_ERR"
JIT_RC=$?
sed -n '/^SBLC-GATE /p;/^sblc_test: /p;/^RESULT: /p' "$JIT_OUT"
JIT_OK=0
sblc_check_output "JIT" "$JIT_OUT" "$JIT_RC" && JIT_OK=1
[ "$JIT_OK" -eq 1 ] || tail -n 20 "$JIT_ERR" | sed 's/^/  stderr: /'

echo ""
echo "--- $TEST: AOT (compile + run) -----------"
AOT_OK=0
if "$RUN" -o "$AOT_BIN" "$TEST" > "$ESHKOL_TEST_TMPDIR/sblc.aot.compile" 2>&1; then
    eshkol_require_output_file_path "$AOT_OUT"
    "$AOT_BIN" > "$AOT_OUT" 2> "$AOT_ERR"
    AOT_RC=$?
    sed -n '/^SBLC-GATE /p;/^sblc_test: /p;/^RESULT: /p' "$AOT_OUT"
    sblc_check_output "AOT" "$AOT_OUT" "$AOT_RC" && AOT_OK=1
    [ "$AOT_OK" -eq 1 ] || tail -n 20 "$AOT_ERR" | sed 's/^/  stderr: /'
else
    tail -n 20 "$ESHKOL_TEST_TMPDIR/sblc.aot.compile" | sed 's/^/  /'
    echo "AOT: FAIL (compile)"
fi

echo ""
echo "--- oracles --------------------------------"
ALL_OK=1
for oracle in codec_oracle mass_conservation coin_trace_oracle semantics \
              sampler_statistics conditioning_receipt; do
    if [ "$JIT_OK" -eq 1 ] && [ "$AOT_OK" -eq 1 ] &&
       grep -qx "SBLC-GATE $oracle PASS" "$JIT_OUT" &&
       grep -qx "SBLC-GATE $oracle PASS" "$AOT_OUT"; then
        emit_event "sblc_$oracle" PASS "$TEST: SBLC-GATE $oracle PASS on JIT and AOT"
        echo "  sblc_$oracle: PASS"
    else
        emit_event "sblc_$oracle" FAIL "$TEST: SBLC-GATE $oracle not PASS on both JIT and AOT"
        echo "  sblc_$oracle: FAIL"
        ALL_OK=0
    fi
done

if [ "$JIT_OK" -eq 1 ] && [ "$AOT_OK" -eq 1 ] && cmp -s "$JIT_OUT" "$AOT_OUT"; then
    DIGEST="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$JIT_OUT")"
    emit_event sblc_jit_aot_parity PASS \
        "JIT and AOT stdout byte-identical (receipts, seeded sample trace); sha256 $DIGEST"
    echo "  sblc_jit_aot_parity: PASS (sha256 $DIGEST)"
else
    emit_event sblc_jit_aot_parity FAIL "JIT and AOT stdout differ or a run failed"
    echo "  sblc_jit_aot_parity: FAIL"
    if [ -s "$JIT_OUT" ] && [ -s "$AOT_OUT" ]; then
        diff "$JIT_OUT" "$AOT_OUT" | head -20 | sed 's/^/  /'
    fi
    ALL_OK=0
fi

if [ "$ALL_OK" -eq 1 ]; then
    emit_event sblc_first_slice PASS \
        "ADR-0017 first slice: codec, mass, coin-trace, semantics, sampler, receipt oracles and JIT/AOT parity all PASS"
    echo ""
    echo "SBLC GATE: PASS (trace: $TRACE_FILE)"
    exit 0
fi
emit_event sblc_first_slice FAIL "at least one SBLC oracle or the JIT/AOT parity check did not pass"
echo ""
echo "SBLC GATE: FAIL (trace: $TRACE_FILE)"
exit 1
