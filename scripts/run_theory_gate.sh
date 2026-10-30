#!/usr/bin/env bash
#
# core.theory acceptance gate (ADR-0039: algebraic theories as data).
#
# Runs, from an existing CMake build:
#   1. the CTest entries that own the behaviour, in native JIT and AOT:
#        theory_jit / theory_aot              tests/stdlib/theory_test.esk
#        theory_topos_jit / theory_topos_aot  tests/stdlib/theory_topos_test.esk
#        ring_jit / ring_aot                  tests/stdlib/ring_test.esk (core.ring's
#                                             law gate, which runs the theory law check)
#   2. every fenced example on docs/reference/stdlib/theory.md through the
#      documentation example harness, on the JIT and as AOT binaries,
#      comparing each `;; =>` annotation with what the build printed.
#
# It writes one event to $TRACE_DIR/v14_algebraic_theories.jsonl:
#   {"kind": "v1_4_connection", "name": "v14_algebraic_theories", "value": PASS|FAIL}
# PASS only when all six CTest entries ran and passed and every documentation
# example matched. Absence of a test is a FAIL, never a skip.
#
#   BUILD_DIR  (env, default "build")   the CMake build directory
#   TRACE_DIR  (env, default scripts/icc_traces)
#   JOBS       (env, default 2)          parallel CTest jobs
#   DOC_TIMEOUT (env, default 600)       per-example limit in seconds: an AOT
#                                        example compiles the library closure
#                                        (core.ring, core.dbsp, core.sheaf.site)

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
    echo "run_theory_gate.sh: $BUILD_DIR_PATH/eshkol-run not found; build first" >&2
    exit 2
fi
BUILD_DIR_PATH="$(cd "$BUILD_DIR_PATH" && pwd -P)" || exit 2

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/v14_algebraic_theories.jsonl"

WORK="$BUILD_DIR_PATH/test-work/theory-gate"
mkdir -p "$WORK"
export ESHKOL_PATH="$REPO_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

TESTS='^(theory_jit|theory_aot|theory_topos_jit|theory_topos_aot|ring_jit|ring_aot)$'
EXPECTED=6
PAGES='docs/reference/stdlib/theory.md'

emit_event() { # PASS|FAIL snippet
    python3 - "$TRACE_FILE" "$1" "$2" <<'PY'
import json, os, sys
path, value, snippet = sys.argv[1], sys.argv[2], sys.argv[3]
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(json.dumps({"kind": "v1_4_connection", "name": "v14_algebraic_theories", "value": value,
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
    --timeout "${DOC_TIMEOUT:-600}" \
    --only "$PAGES" \
    --work-dir "$WORK/doc-examples" \
    --no-trace > "$doc_log" 2>&1
doc_status=$?
# The harness reports "examples=N executed=E passed=P failed=F" under a
# "scope=... paths=..." header for every gated scope; the counts that matter
# are those under the header naming this page. An example that was not
# executed is not evidence, so all of them must run and pass.
doc_counts=$(awk -v pages="paths=$PAGES" 'index($0, "doc example gate: ") && index($0, pages) {grab=1; next}
    grab && match($0, /examples=[0-9]+ executed=[0-9]+ passed=[0-9]+/) {print substr($0, RSTART, RLENGTH); grab=0}' "$doc_log" | tail -n 1)
doc_examples=$(printf '%s' "$doc_counts" | sed -E 's/examples=([0-9]+).*/\1/')
doc_executed=$(printf '%s' "$doc_counts" | sed -E 's/.*executed=([0-9]+).*/\1/')
doc_passed=$(printf '%s' "$doc_counts" | sed -E 's/.*passed=([0-9]+).*/\1/')
summary="$summary; documentation examples ${doc_counts:-none reported} (exit ${doc_status})"
if [ "$doc_status" -ne 0 ] || [ -z "$doc_counts" ] || [ "${doc_examples:-0}" -eq 0 ] \
   || [ "$doc_executed" -ne "$doc_examples" ] || [ "$doc_passed" -ne "$doc_examples" ]; then
    status=1
fi

if [ "$status" -eq 0 ]; then
    emit_event PASS "$summary"
    echo "core.theory gate: PASS ($summary)"
else
    emit_event FAIL "$summary"
    echo "core.theory gate: FAIL ($summary)"
    tail -n 40 "$ctest_log"
    tail -n 40 "$doc_log"
fi
exit "$status"
