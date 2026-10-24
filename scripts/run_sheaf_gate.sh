#!/usr/bin/env bash
#
# core.sheaf acceptance gate (ADR-0033, v1.4.0-connection).
#
# Runs, from an existing CMake build:
#   1. the CTest entries that own the behaviour, in native JIT and AOT:
#        sheaf_jit / sheaf_aot (tests/stdlib/sheaf_test.esk) and the two
#        examples written on the library,
#        sheaf_finite_site_sheafification_{jit,aot} and
#        sheaf_finite_spaces_cech_descent_{jit,aot};
#   2. every fenced example on docs/reference/stdlib/sheaf.md and
#      docs/reference/stdlib/integer_linalg.md through the documentation
#      example harness, on the JIT and as AOT binaries, comparing each
#      `;; =>` annotation with what the build printed.
#
# It writes one event to $TRACE_DIR/v14_core_sheaf.jsonl:
#   {"kind": "v1_4_connection", "name": "v14_core_sheaf", "value": PASS|FAIL}
# PASS only when all six CTest entries ran and passed and every documentation
# example matched. Absence of a test is a FAIL, never a skip.
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
    echo "run_sheaf_gate.sh: $BUILD_DIR_PATH/eshkol-run not found; build first" >&2
    exit 2
fi
BUILD_DIR_PATH="$(cd "$BUILD_DIR_PATH" && pwd -P)" || exit 2

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/v14_core_sheaf.jsonl"

WORK="$BUILD_DIR_PATH/test-work/sheaf-gate"
mkdir -p "$WORK"
export ESHKOL_PATH="$REPO_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

TESTS='^(sheaf_jit|sheaf_aot|sheaf_finite_site_sheafification_jit|sheaf_finite_site_sheafification_aot|sheaf_finite_spaces_cech_descent_jit|sheaf_finite_spaces_cech_descent_aot)$'
EXPECTED=6

emit_event() { # PASS|FAIL snippet
    python3 - "$TRACE_FILE" "$1" "$2" <<'PY'
import json, os, sys
path, value, snippet = sys.argv[1], sys.argv[2], sys.argv[3]
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(json.dumps({"kind": "v1_4_connection", "name": "v14_core_sheaf", "value": value,
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

doc_log="$WORK/doc-examples.log"
python3 "$REPO_ROOT/scripts/doc_audit/check_doc_examples.py" \
    --eshkol-run "$BUILD_DIR_PATH/eshkol-run" \
    --engines jit,aot \
    --only docs/reference/stdlib/sheaf.md,docs/reference/stdlib/integer_linalg.md \
    --work-dir "$WORK/doc-examples" \
    --no-trace > "$doc_log" 2>&1
doc_status=$?
summary="$summary; documentation examples exit ${doc_status}: $(tail -n 1 "$doc_log")"
[ "$doc_status" -eq 0 ] || status=1

if [ "$status" -eq 0 ]; then
    emit_event PASS "$summary"
    echo "core.sheaf gate: PASS ($summary)"
else
    emit_event FAIL "$summary"
    echo "core.sheaf gate: FAIL ($summary)"
    tail -n 40 "$ctest_log"
    tail -n 40 "$doc_log"
fi
exit "$status"
