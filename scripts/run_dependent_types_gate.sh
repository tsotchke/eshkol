#!/usr/bin/env bash
#
# Dependent type enforcement gate (ADR-0042, v1.9.0-types).
#
# Runs, from an existing CMake build, the three CTest entries that own the
# behaviour:
#   type_term_test                      the core term language
#   typesystem_dependent_corpus         every dep_*/bidir_* fixture: positives
#                                       compile and run, negatives are refused
#                                       with their diagnostic
#   dependent_type_engine_parity_gate   the bytecode VM refuses and runs the
#                                       same programs as the native engine
#
# Writes one event to $TRACE_DIR/v19_dependent_enforcement.jsonl:
#   {"kind": "v1_9_types", "name": "v19_dependent_enforcement", "value": PASS|FAIL}
# PASS only when all three ran and passed. A missing test is a FAIL.
#
#   BUILD_DIR  (env, default "build")   the CMake build directory
#   TRACE_DIR  (env, default scripts/icc_traces)
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
    echo "run_dependent_types_gate.sh: $BUILD_DIR_PATH/eshkol-run not found; build first" >&2
    exit 2
fi
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/v19_dependent_enforcement.jsonl"
WORK="$BUILD_DIR_PATH/test-work/dependent-types-gate"
mkdir -p "$WORK"

TESTS='^(type_term_test|typesystem_dependent_corpus|dependent_type_engine_parity_gate)$'
EXPECTED=3

log="$WORK/ctest.log"
(cd "$BUILD_DIR_PATH" && ctest -R "$TESTS" --output-on-failure) > "$log" 2>&1
status=$?
ran=$(grep -cE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$log" || true)
passed=$(grep -cE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: .* Passed' "$log" || true)
summary="ctest ${passed}/${EXPECTED} passed (ran ${ran}, exit ${status})"
value=PASS
if [ "$status" -ne 0 ] || [ "$ran" -ne "$EXPECTED" ] || [ "$passed" -ne "$EXPECTED" ]; then
    value=FAIL
fi
python3 - "$TRACE_FILE" "$value" "$summary" <<'PY'
import json, os, sys
path, value, snippet = sys.argv[1], sys.argv[2], sys.argv[3]
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(json.dumps({"kind": "v1_9_types", "name": "v19_dependent_enforcement",
                        "value": value, "snippet": snippet[:2000], "confidence": 0.95},
                       ensure_ascii=False) + "\n")
os.replace(tmp, path)
PY
echo "dependent types gate: $value ($summary)"
[ "$value" = PASS ] || { tail -n 40 "$log"; exit 1; }
