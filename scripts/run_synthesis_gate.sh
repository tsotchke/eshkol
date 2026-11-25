#!/usr/bin/env bash
#
# v1.7.0-synthesis acceptance gate (ADR-0052).
#
# Runs, from an existing CMake build, the CTest entries that own the behaviour:
#   synthesis_test_{jit,aot}        core.synthesis (tests/stdlib/synthesis_test.esk)
#   gnn_test_{jit,aot}              ml.autograd and ml.gnn (tests/ml/gnn_test.esk)
#   synthesis_holes_test_{jit,aot}  ?? completion (tests/types/synthesis_holes_test.esk)
#   synthesis_hole_diagnostics      the completion diagnostic
#   synthesis_holes_strict_types    completions pass the checker with type errors fatal
#   synthesis_hole_refusal_{jit,aot} a hole with no completion stops compilation
#   synthesis_hole_vm_named         the bytecode compiler names an unbound hole
#
# It writes one event to $TRACE_DIR/v17_synthesis.jsonl:
#   {"kind": "v1_7_synthesis", "name": "v17_synthesis", "value": PASS|FAIL}
# PASS only when all eleven entries ran and passed. A missing entry is a FAIL.
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
    echo "run_synthesis_gate.sh: $BUILD_DIR_PATH/eshkol-run not found; build first" >&2
    exit 2
fi
BUILD_DIR_PATH="$(cd "$BUILD_DIR_PATH" && pwd -P)" || exit 2

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/v17_synthesis.jsonl"

WORK="$BUILD_DIR_PATH/test-work/synthesis-gate"
mkdir -p "$WORK"

TESTS='^(synthesis_test_jit|synthesis_test_aot|gnn_test_jit|gnn_test_aot|synthesis_holes_test_jit|synthesis_holes_test_aot|synthesis_hole_diagnostics|synthesis_holes_strict_types|synthesis_hole_refusal_jit|synthesis_hole_refusal_aot|synthesis_hole_vm_named)$'
EXPECTED=11

emit_event() { # PASS|FAIL snippet
    python3 - "$TRACE_FILE" "$1" "$2" <<'PY'
import json, os, sys
path, value, snippet = sys.argv[1], sys.argv[2], sys.argv[3]
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(json.dumps({"kind": "v1_7_synthesis", "name": "v17_synthesis", "value": value,
                        "snippet": snippet[:2000], "confidence": 0.95}, ensure_ascii=False) + "\n")
os.replace(tmp, path)
PY
}

ctest_log="$WORK/ctest.log"
(cd "$BUILD_DIR_PATH" && ctest -R "$TESTS" --output-on-failure -j "${JOBS:-2}") > "$ctest_log" 2>&1
ctest_status=$?
ran=$(grep -cE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$ctest_log" || true)
passed=$(grep -cE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: .* Passed' "$ctest_log" || true)
summary="ctest ${passed}/${EXPECTED} passed (ran ${ran}, exit ${ctest_status})"
if [ "$ctest_status" -eq 0 ] && [ "$ran" -eq "$EXPECTED" ] && [ "$passed" -eq "$EXPECTED" ]; then
    emit_event PASS "$summary"
    echo "v1.7 synthesis gate: PASS ($summary)"
    exit 0
fi
emit_event FAIL "$summary"
echo "v1.7 synthesis gate: FAIL ($summary)"
tail -n 60 "$ctest_log"
exit 1
