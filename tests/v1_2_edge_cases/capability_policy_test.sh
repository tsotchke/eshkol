#!/usr/bin/env bash
# capability_policy_test.sh — the capability policy, one scenario per process.
#
# A policy only ever narrows: installing one while another is active keeps the
# names both hold, and an active policy cannot be cleared by the code it
# governs. So a single process can walk down one chain of policies and no
# further, and each scenario in
# tests/v1_2_edge_cases/support/capability_policy_scenarios.esk starts from a
# fresh one.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"
SCENARIOS="$ROOT/tests/v1_2_edge_cases/support/capability_policy_scenarios.esk"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

mkdir -p "$ROOT/.scratch"
WORK="$ROOT/.scratch/capability_policy_test.$$"
mkdir -p "$WORK"

failures=0
ran=0

for scenario in no-policy denies-unlisted narrows clear-refused empty-policy \
                file-read file-write file-read-after-write file-write-cleanup \
                env-read env-write subprocess-shell subprocess-with-shell; do
    ran=$((ran + 1))
    case_work="$WORK/$scenario"
    mkdir -p "$case_work"
    if ESHKOL_CAPABILITY_SCENARIO="$scenario" ESHKOL_CAPABILITY_WORK="$case_work" \
       ESHKOL_JIT_CACHE=0 "$RUN" -r "$SCENARIOS" >"$WORK/$scenario.out" 2>&1; then
        if grep -q "^FAIL: " "$WORK/$scenario.out"; then
            echo "FAIL: scenario $scenario"
            grep "^FAIL: " "$WORK/$scenario.out" | sed 's/^/    /'
            failures=$((failures + 1))
        else
            echo "PASS: scenario $scenario"
        fi
    else
        echo "FAIL: scenario $scenario did not run"
        sed -n '1,20p' "$WORK/$scenario.out" | sed 's/^/    /'
        failures=$((failures + 1))
    fi
done

echo "scenarios: $ran, failures: $failures"
if [ "$failures" -ne 0 ]; then
    echo "FAIL: capability_policy_test"
    exit 1
fi
echo "PASS: capability_policy_test"
