#!/usr/bin/env bash
#
# core.transcript acceptance gate: content-addressed transcripts of
# certified mathematical quantities, cross-checked by an independent
# pure-Python replay.
#
# Runs, from an existing CMake build:
#   1. The CTest entries that own the behaviour, in native JIT and AOT:
#        transcript_jit / transcript_aot          tests/stdlib/transcript_test.esk
#        dw_group_cohomology_order_eight_jit/aot   examples/mathematics_group_cohomology_order_eight.esk
#        dw_cohomology_rings_cup_products_jit/aot   examples/mathematics_cohomology_rings_cup_products.esk
#        hodge_fermat_sweep_default_jit/aot        examples/mathematics_fermat_hodge_classes_sweep.esk
#        multimodular_jit/aot                      tests/stdlib/multimodular_test.esk
#      (the last four are the programs wired to core.transcript; this gate
#      proves their own math still passes, not merely that recording is wired)
#   2. transcript_replay_check: scripts/math_crosscheck/replay_check.py
#      against every committed fixture in scripts/math_crosscheck/fixtures/,
#      a pure-Python-stdlib independent recomputation of each claim plus a
#      tamper check on its hash.
#
# It writes one event to $TRACE_DIR/v14_math_transcripts.jsonl:
#   {"kind": "v1_4_connection", "name": "v14_math_transcripts", "value": PASS|FAIL}
# PASS only when all eleven CTest entries ran and passed. Absence of a test
# is a FAIL, never a skip.
#
#   BUILD_DIR  (env, default "build")   the CMake build directory
#   TRACE_DIR  (env, default scripts/icc_traces)
#   JOBS       (env, default 2)          parallel CTest jobs

set -u
set -o pipefail
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) BUILD_DIR_PATH="$BUILD_DIR" ;;
    *)  BUILD_DIR_PATH="$REPO_ROOT/$BUILD_DIR" ;;
esac
if [ ! -x "$BUILD_DIR_PATH/eshkol-run" ]; then
    echo "run_transcript_gate.sh: $BUILD_DIR_PATH/eshkol-run not found; build first" >&2
    exit 2
fi
BUILD_DIR_PATH="$(cd "$BUILD_DIR_PATH" && pwd -P)" || exit 2

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/v14_math_transcripts.jsonl"

WORK="$BUILD_DIR_PATH/test-work/transcript-gate"
mkdir -p "$WORK"
export ESHKOL_PATH="$REPO_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

TESTS='^(transcript_jit|transcript_aot|dw_group_cohomology_order_eight_jit|dw_group_cohomology_order_eight_aot|dw_cohomology_rings_cup_products_jit|dw_cohomology_rings_cup_products_aot|hodge_fermat_sweep_default_jit|hodge_fermat_sweep_default_aot|multimodular_jit|multimodular_aot|transcript_replay_check)$'
EXPECTED=11

emit_event() { # PASS|FAIL snippet
    python3 - "$TRACE_FILE" "$1" "$2" <<'PY'
import json, os, sys
path, value, snippet = sys.argv[1], sys.argv[2], sys.argv[3]
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(json.dumps({"kind": "v1_4_connection", "name": "v14_math_transcripts", "value": value,
                        "snippet": snippet[:2000], "confidence": 0.95}, ensure_ascii=False) + "\n")
os.replace(tmp, path)
PY
}

status=0
summary=""

ctest_log="$WORK/ctest.log"
(cd "$BUILD_DIR_PATH" && ctest -R "$TESTS" --output-on-failure -j "${JOBS:-2}") > "$ctest_log" 2>&1
ctest_status=$?
ran=$(grep -cE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$ctest_log" || true)
passed=$(grep -cE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: .* Passed' "$ctest_log" || true)
summary="ctest ${passed}/${EXPECTED} passed (ran ${ran}, exit ${ctest_status})"
if [ "$ctest_status" -ne 0 ] || [ "$ran" -ne "$EXPECTED" ] || [ "$passed" -ne "$EXPECTED" ]; then
    status=1
fi

if [ "$status" -eq 0 ]; then
    emit_event PASS "$summary"
    echo "core.transcript gate: PASS ($summary)"
else
    emit_event FAIL "$summary"
    echo "core.transcript gate: FAIL ($summary)"
    tail -n 60 "$ctest_log"
fi
exit "$status"
