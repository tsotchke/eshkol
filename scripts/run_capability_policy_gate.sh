#!/usr/bin/env bash
#
# Capability policy acceptance gate.
#
# The policy's whole claim is: a program that installs no policy behaves as it
# always has, and once a policy is active every capability it does not name is
# refused — at the shell, process-creation, socket and foreign-code sinks as
# well as the file and environment surfaces, on every engine.
#
# Runs, from an existing CMake build, the three CTest entries that own that
# claim:
#   capability_sink_coverage_test   each sink class with no policy, under a
#                                   policy that withholds the capability, and
#                                   under one that grants it, on every engine
#                                   that carries the sink — plus the case a
#                                   policy granting only file-read must also
#                                   hold for the shell
#   process_child_bookkeeping_test  (process-spawn-pty …) hands back the
#                                   pseudo-terminal descriptor on every engine
#                                   and repeated spawns stay within budget
#   process_child_env_test          a child environment overlay reaches the
#                                   child and leaves the spawning program's own
#                                   environment as it was
#
# It writes one event to $TRACE_DIR/v14_capability_policy.jsonl:
#   {"kind": "v1_4_connection", "name": "v14_capability_policy", "value": PASS|FAIL}
# PASS only when all three ran and passed. A test that did not run is a FAIL,
# never a skip — absence of evidence is not evidence.
#
#   BUILD_DIR  (env, default "build")             the CMake build directory
#   TRACE_DIR  (env, default scripts/icc_traces)
#   JOBS       (env, default 2)                   parallel CTest jobs

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
    echo "run_capability_policy_gate.sh: $BUILD_DIR_PATH/eshkol-run not found; build first" >&2
    exit 2
fi
BUILD_DIR_PATH="$(cd "$BUILD_DIR_PATH" && pwd -P)" || exit 2

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
. "$REPO_ROOT/scripts/lib/evidence_paths.sh"
eshkol_evidence_abs_var TRACE_DIR "$REPO_ROOT" || exit $?
mkdir -p "$TRACE_DIR"
TRACE_FILE="$TRACE_DIR/v14_capability_policy.jsonl"

WORK="$BUILD_DIR_PATH/test-work/capability-policy-gate"
mkdir -p "$WORK"
export ESHKOL_PATH="$REPO_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

TESTS='^(capability_sink_coverage_test|process_child_bookkeeping_test|process_child_env_test)$'
EXPECTED=3

emit_event() { # PASS|FAIL snippet
    python3 - "$TRACE_FILE" "$1" "$2" <<'PY'
import json, os, sys
path, value, snippet = sys.argv[1], sys.argv[2], sys.argv[3]
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(json.dumps({"kind": "v1_4_connection", "name": "v14_capability_policy", "value": value,
                        "snippet": snippet[:2000], "confidence": 0.95}, ensure_ascii=False) + "\n")
os.replace(tmp, path)
PY
}

status=0
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
    echo "capability policy gate: PASS ($summary)"
else
    emit_event FAIL "$summary"
    echo "capability policy gate: FAIL ($summary)"
    tail -n 60 "$ctest_log"
fi
exit "$status"
