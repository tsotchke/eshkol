#!/usr/bin/env bash
set -eu -o pipefail

runner="${BUILD_DIR:-build-tsan}/eshkol-run"
export TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1
out=$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-.}}/tsan-concurrency.XXXXXXXX")
python3 scripts/p8/gen_concurrency_fuzz.py --out "$out" --seed 8805 --repeats 20 --shapes 10

cases=("$out"/*.esk)
[[ ${#cases[@]} -eq 10 ]]
status=0
for fixture in "${cases[@]}"; do
    log="${fixture%.esk}.log"
    expected=$(awk '/^;; CHECKS: [0-9]+$/ {print $3}' "$fixture")
    if ! "$runner" -r "$fixture" | tee "$log"; then
        status=1
    elif [[ ! $expected =~ ^[1-9][0-9]*$ ]] || ! grep -Fxq "SUMMARY $expected/0" "$log"; then
        echo "Incomplete concurrency checks in $fixture" >&2
        status=1
    fi
done
for fixture in threads_mutex_concurrency_test channels_test threads_semaphore_barrier_test atomics_contention_test; do
    "$runner" -r "tests/v1_2_edge_cases/$fixture.esk" || status=1
done
exit "$status"
