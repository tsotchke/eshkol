#!/usr/bin/env bash
#
# core.monoidal acceptance gate (ADR-0041, v1.4.0-connection).
#
# Runs, from an existing CMake build:
#   1. the CTest entries that own the behaviour, in native JIT and AOT:
#        monoidal_test (monoidal, braided, symmetric, compact closed and
#        dagger structure, FinVect, string diagrams) and zx_test (soundness
#        of every ZX rewrite rule over the phases k pi/4, circuits,
#        simplification) ({jit,aot});
#   2. every fenced example on docs/reference/stdlib/monoidal.md and zx.md
#      (pages of the documentation gate's `reference` scope) through the
#      documentation example harness, on the JIT and as AOT binaries,
#      comparing each `;; =>` annotation with what the build printed;
#   3. the Moonlab cross-check zx_moonlab_crosscheck from a build configured
#      with -DESHKOL_QUANTUM_ENABLED=ON, when QUANTUM_BUILD_DIR names one.
#
# It writes two events:
#   $TRACE_DIR/v14_monoidal_zx.jsonl
#     {"kind": "v1_4_connection", "name": "v14_monoidal_zx", "value": PASS|FAIL}
#     PASS only when all four CTest entries ran and passed, every
#     documentation example matched, and the Moonlab cross-check, when a
#     quantum build was given, passed. Absence of a test is a FAIL.
#   $TRACE_DIR/v14_monoidal_zx_moonlab.jsonl
#     value PASS or FAIL from the quantum build, or SKIP with the reason
#     (no quantum build given) in its snippet.
#
#   BUILD_DIR  (env, default "build")   the CMake build directory
#   TRACE_DIR  (env, default scripts/icc_traces)
#   JOBS       (env, default 2)          parallel CTest jobs
#   QUANTUM_BUILD_DIR (env, optional)    a CMake build with Moonlab enabled
#   DOC_TIMEOUT (env, default 360)       per-example limit for the reference
#              pages (each example is also compiled as an AOT binary)

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
    echo "run_monoidal_gate.sh: $BUILD_DIR_PATH/eshkol-run not found; build first" >&2
    exit 2
fi
BUILD_DIR_PATH="$(cd "$BUILD_DIR_PATH" && pwd -P)" || exit 2

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/v14_monoidal_zx.jsonl"
MOONLAB_TRACE_FILE="$TRACE_DIR/v14_monoidal_zx_moonlab.jsonl"

WORK="$BUILD_DIR_PATH/test-work/monoidal-gate"
mkdir -p "$WORK"
export ESHKOL_PATH="$REPO_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

TESTS='^((monoidal_test|zx_test)_(jit|aot))$'
EXPECTED=4
PAGES="docs/reference/stdlib/monoidal.md docs/reference/stdlib/zx.md"

emit_event() { # file name PASS|FAIL|SKIP snippet
    python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, os, sys
path, name, value, snippet = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(json.dumps({"kind": "v1_4_connection", "name": name, "value": value,
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
only_list=$(printf '%s' "$PAGES" | tr ' ' ',')
python3 "$REPO_ROOT/scripts/doc_audit/check_doc_examples.py" \
    --eshkol-run "$BUILD_DIR_PATH/eshkol-run" \
    --engines jit,aot \
    --scope reference \
    --only "$only_list" \
    --timeout "${DOC_TIMEOUT:-360}" \
    --work-dir "$WORK/doc-examples" \
    --no-trace > "$doc_log" 2>&1
doc_status=$?
# The harness reports "examples=N executed=E passed=P failed=F" for the whole
# scope while executing only these pages; an example that was not executed is
# not evidence, so every example on the two pages (counted by the same
# extractor) must run and pass.
doc_counts=$(grep -oE 'examples=[0-9]+ executed=[0-9]+ passed=[0-9]+' "$doc_log" | tail -n 1)
doc_examples=$(python3 - "$REPO_ROOT" $PAGES <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/scripts/doc_audit")
import extract_examples
records = extract_examples.extract(sys.argv[1], sys.argv[2:])
print(sum(1 for r in records if r.get("klass") != "data-file"))
PY
)
doc_executed=$(printf '%s' "$doc_counts" | sed -E 's/.*executed=([0-9]+).*/\1/')
doc_passed=$(printf '%s' "$doc_counts" | sed -E 's/.*passed=([0-9]+).*/\1/')
summary="$summary; documentation examples on monoidal.md and zx.md ${doc_examples:-?}, harness ${doc_counts:-none reported} (exit ${doc_status})"
if [ "$doc_status" -ne 0 ] || [ -z "$doc_counts" ] || [ "${doc_examples:-0}" -eq 0 ] \
   || [ "$doc_executed" -ne "$doc_examples" ] || [ "$doc_passed" -ne "$doc_examples" ]; then
    status=1
fi

# Moonlab: only from a build that links it
if [ -n "${QUANTUM_BUILD_DIR:-}" ] && [ -x "$QUANTUM_BUILD_DIR/eshkol-run" ]; then
    moonlab_log="$WORK/moonlab.log"
    (cd "$QUANTUM_BUILD_DIR" && ctest -R '^zx_moonlab_crosscheck$' --output-on-failure -V) > "$moonlab_log" 2>&1
    moonlab_status=$?
    moonlab_line=$(grep -oE 'zx_moonlab_crosscheck: circuits .*' "$moonlab_log" | tail -n 1)
    if [ "$moonlab_status" -eq 0 ] && grep -q 'zx_moonlab_crosscheck_test: PASS' "$moonlab_log"; then
        emit_event "$MOONLAB_TRACE_FILE" v14_monoidal_zx_moonlab PASS "$moonlab_line"
        summary="$summary; Moonlab cross-check PASS ($moonlab_line)"
    else
        emit_event "$MOONLAB_TRACE_FILE" v14_monoidal_zx_moonlab FAIL "exit $moonlab_status; $moonlab_line"
        summary="$summary; Moonlab cross-check FAIL (exit $moonlab_status)"
        status=1
    fi
else
    emit_event "$MOONLAB_TRACE_FILE" v14_monoidal_zx_moonlab SKIP "not run: QUANTUM_BUILD_DIR does not name a build with -DESHKOL_QUANTUM_ENABLED=ON"
    summary="$summary; Moonlab cross-check not run (no quantum build given)"
fi

if [ "$status" -eq 0 ]; then
    emit_event "$TRACE_FILE" v14_monoidal_zx PASS "$summary"
    echo "core.monoidal gate: PASS ($summary)"
else
    emit_event "$TRACE_FILE" v14_monoidal_zx FAIL "$summary"
    echo "core.monoidal gate: FAIL ($summary)"
    tail -n 40 "$ctest_log"
    tail -n 40 "$doc_log"
fi
exit "$status"
