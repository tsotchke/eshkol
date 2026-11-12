#!/usr/bin/env bash
#
# core.ring / core.multimodular acceptance gate (ADR-0034, library rung).
#
# Runs, from an existing CMake build:
#   1. the CTest entries that own the behaviour, in native JIT and AOT:
#        ring_jit / ring_aot             tests/stdlib/ring_test.esk (the law gate)
#        ring_ieee_jit / ring_ieee_aot   tests/stdlib/ring_ieee_test.esk (IEEE tier)
#        multimodular_jit / _aot         tests/stdlib/multimodular_test.esk
#        rings_functor_of_points_jit / _aot  the example written on the libraries
#   2. every fenced example on docs/reference/stdlib/ring.md and
#      docs/reference/stdlib/multimodular.md through the documentation example
#      harness, on the JIT and as AOT binaries, comparing each `;; =>`
#      annotation with what the build printed.
#
# It writes one event to $TRACE_DIR/v14_core_ring.jsonl:
#   {"kind": "v1_4_connection", "name": "v14_core_ring", "value": PASS|FAIL}
# PASS only when all eight CTest entries ran and passed and every documentation
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
    echo "run_ring_gate.sh: $BUILD_DIR_PATH/eshkol-run not found; build first" >&2
    exit 2
fi
BUILD_DIR_PATH="$(cd "$BUILD_DIR_PATH" && pwd -P)" || exit 2

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/v14_core_ring.jsonl"

WORK="$BUILD_DIR_PATH/test-work/ring-gate"
mkdir -p "$WORK"
export ESHKOL_PATH="$REPO_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

TESTS='^(ring_jit|ring_aot|ring_ieee_jit|ring_ieee_aot|multimodular_jit|multimodular_aot|rings_functor_of_points_jit|rings_functor_of_points_aot)$'
EXPECTED=8
PAGES='docs/reference/stdlib/ring.md,docs/reference/stdlib/multimodular.md'

emit_event() { # PASS|FAIL snippet
    python3 - "$TRACE_FILE" "$1" "$2" <<'PY'
import json, os, sys
path, value, snippet = sys.argv[1], sys.argv[2], sys.argv[3]
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(json.dumps({"kind": "v1_4_connection", "name": "v14_core_ring", "value": value,
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
    --scope reference \
    --only "$PAGES" \
    --work-dir "$WORK/doc-examples" \
    --no-trace > "$doc_log" 2>&1
doc_status=$?
# The harness reports "examples=N executed=E passed=P failed=F" for the whole
# `reference` scope while executing only these two pages (the scope also
# covers core.sheaf/core.topos's reference pages; --only restricts execution
# to this gate's pages without changing the scope's declared total). The
# count that matters is how many of THESE two pages' examples exist, counted
# independently the same way run_topos_gate.sh does for its own page; an
# example that was not executed is not evidence, so all of them must run and
# pass.
doc_counts=$(grep -oE 'examples=[0-9]+ executed=[0-9]+ passed=[0-9]+' "$doc_log" | tail -n 1)
doc_examples=$(python3 - "$REPO_ROOT" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/scripts/doc_audit")
import extract_examples
records = extract_examples.extract(sys.argv[1], ["docs/reference/stdlib/ring.md", "docs/reference/stdlib/multimodular.md"])
print(sum(1 for r in records if r.get("klass") != "data-file"))
PY
)
doc_executed=$(printf '%s' "$doc_counts" | sed -E 's/.*executed=([0-9]+).*/\1/')
doc_passed=$(printf '%s' "$doc_counts" | sed -E 's/.*passed=([0-9]+).*/\1/')
summary="$summary; documentation examples on ring.md/multimodular.md ${doc_examples:-?}, harness ${doc_counts:-none reported} (exit ${doc_status})"
if [ "$doc_status" -ne 0 ] || [ -z "$doc_counts" ] || [ "${doc_examples:-0}" -eq 0 ] \
   || [ "$doc_executed" -ne "$doc_examples" ] || [ "$doc_passed" -ne "$doc_examples" ]; then
    status=1
fi

if [ "$status" -eq 0 ]; then
    emit_event PASS "$summary"
    echo "core.ring gate: PASS ($summary)"
else
    emit_event FAIL "$summary"
    echo "core.ring gate: FAIL ($summary)"
    tail -n 40 "$ctest_log"
    tail -n 40 "$doc_log"
fi
exit "$status"
