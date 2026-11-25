#!/usr/bin/env bash
#
# Structural types gate (ADR-0046, v1.9.0-types): row polymorphism, higher-rank types, session types.
#
# Runs, from an existing CMake build, the two CTest entries that own the
# behaviour:
#   structural_types_test   the solver contract (rows, protocols, duality, levels)
#   types_ext_gate          every tests/types_ext fixture on JIT, AOT and VM:
#                           positives run with exact output and principal types,
#                           negatives are refused with their diagnostic
#
# Writes one event to $TRACE_DIR/v19_structural_types.jsonl:
#   {"kind": "v1_9_types", "name": "v19_structural_types", "value": PASS|FAIL}
# PASS only when both ran and passed. A missing test is a FAIL.
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
    echo "run_structural_types_gate.sh: $BUILD_DIR_PATH/eshkol-run not found; build first" >&2
    exit 2
fi
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/v19_structural_types.jsonl"
WORK="$BUILD_DIR_PATH/test-work/structural-types-gate"
mkdir -p "$WORK"

TESTS='^(structural_types_test|types_ext_gate)$'
EXPECTED=2

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
    f.write(json.dumps({"kind": "v1_9_types", "name": "v19_structural_types",
                        "value": value, "snippet": snippet[:2000], "confidence": 0.95},
                       ensure_ascii=False) + "\n")
os.replace(tmp, path)
PY
echo "structural types gate: $value ($summary)"
[ "$value" = PASS ] || { tail -n 40 "$log"; exit 1; }
