#!/usr/bin/env bash
#
# Quantum register types, regions, circuit compilation and QAOA gate
# (ADR-0048, criterion v20_quantum_types).
#
# Runs, from existing CMake builds:
#   1. the CTest entries that own the behaviour: the compile-time corpus on
#      the native and bytecode engines (quantum_types_corpus) and the
#      library suites in native JIT and AOT (quantum_circuit_test,
#      quantum_optimize_test, quantum_route_test, qaoa_test,
#      quantum_register_test);
#   2. every fenced example on docs/reference/stdlib/quantum.md through the
#      documentation example harness, on the JIT and as AOT binaries;
#   3. qaoa_moonlab_crosscheck from a build configured with
#      -DESHKOL_QUANTUM_ENABLED=ON (QUANTUM_BUILD_DIR): Moonlab expectations
#      and Moonlab-adjoint gradients against the exact state vector. QAOA
#      through Moonlab is part of the criterion, so a missing quantum build
#      is a FAIL, not a skip.
#
# It writes $TRACE_DIR/v20_quantum_types.jsonl
#   {"kind": "v2_0_starlight", "name": "v20_quantum_types", "value": PASS|FAIL}
#
#   BUILD_DIR  (env, default "build")   the CMake build directory
#   TRACE_DIR  (env, default scripts/icc_traces)
#   JOBS       (env, default 2)          parallel CTest jobs
#   QUANTUM_BUILD_DIR (env)              a CMake build with Moonlab enabled
#   DOC_TIMEOUT (env, default 360)       per-example limit

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
    echo "run_quantum_types_gate.sh: $BUILD_DIR_PATH/eshkol-run not found; build first" >&2
    exit 2
fi
BUILD_DIR_PATH="$(cd "$BUILD_DIR_PATH" && pwd -P)" || exit 2

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/v20_quantum_types.jsonl"

WORK="$BUILD_DIR_PATH/test-work/quantum-types-gate"
mkdir -p "$WORK"
export ESHKOL_PATH="$REPO_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

TESTS='^(((quantum_circuit_test|quantum_optimize_test|quantum_route_test|qaoa_test|quantum_register_test)_(jit|aot))|quantum_types_corpus)$'
EXPECTED=11
PAGES="docs/reference/stdlib/quantum.md"

emit_event() { # file name PASS|FAIL|SKIP snippet
    python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, os, sys
path, name, value, snippet = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(json.dumps({"kind": "v2_0_starlight", "name": name, "value": value,
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
summary="$summary; documentation examples on quantum.md ${doc_examples:-?}, harness ${doc_counts:-none reported} (exit ${doc_status})"
if [ "$doc_status" -ne 0 ] || [ -z "$doc_counts" ] || [ "${doc_examples:-0}" -eq 0 ] \
   || [ "$doc_executed" -ne "$doc_examples" ] || [ "$doc_passed" -ne "$doc_examples" ]; then
    status=1
fi

# Moonlab: required
if [ -n "${QUANTUM_BUILD_DIR:-}" ] && [ -x "$QUANTUM_BUILD_DIR/eshkol-run" ]; then
    moonlab_log="$WORK/moonlab.log"
    (cd "$QUANTUM_BUILD_DIR" && ctest -R '^qaoa_moonlab_crosscheck$' --output-on-failure -V) > "$moonlab_log" 2>&1
    moonlab_status=$?
    moonlab_line=$(grep -oE 'qaoa-moonlab: max .*' "$moonlab_log" | tail -n 1)
    if [ "$moonlab_status" -eq 0 ] && grep -q 'qaoa_moonlab_crosscheck_test: PASS' "$moonlab_log"; then
        summary="$summary; Moonlab QAOA cross-check PASS ($moonlab_line)"
    else
        summary="$summary; Moonlab QAOA cross-check FAIL (exit $moonlab_status; $moonlab_line)"
        status=1
    fi
else
    summary="$summary; Moonlab QAOA cross-check not run: QUANTUM_BUILD_DIR does not name a build with -DESHKOL_QUANTUM_ENABLED=ON"
    status=1
fi

if [ "$status" -eq 0 ]; then
    emit_event "$TRACE_FILE" v20_quantum_types PASS "$summary"
    echo "quantum types gate: PASS ($summary)"
else
    emit_event "$TRACE_FILE" v20_quantum_types FAIL "$summary"
    echo "quantum types gate: FAIL ($summary)"
    tail -n 40 "$ctest_log"
    tail -n 40 "$doc_log"
fi
exit "$status"
