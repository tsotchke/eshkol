#!/usr/bin/env bash
# run_differential_fuzz.sh — seeded random differential fuzzing (P1).
#
# Drives scripts/gen_differential.py: generate bounded, deterministic random
# programs, run each on every native execution axis (jit, jit-nocache,
# aot-o0, aot-o2), diff pairwise, and auto-shrink any divergent program to a
# minimal repro in tests/differential/found/NNN_shrunk.esk.
#
# Usage: scripts/run_differential_fuzz.sh --seed S --count N [extra gen args]
# Trace: scripts/icc_traces/differential_fuzz.jsonl (kind=differential_smoke)
set -u
cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build}"

SEED=""
COUNT=100
PASSTHRU=()
while [ $# -gt 0 ]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --count) COUNT="$2"; shift 2 ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done
if [ -z "$SEED" ]; then
    echo "usage: scripts/run_differential_fuzz.sh --seed S [--count N] [gen_differential.py args]" >&2
    exit 2
fi

exec python3 scripts/gen_differential.py \
    --seed "$SEED" --count "$COUNT" --build-dir "$BUILD_DIR" \
    ${PASSTHRU[@]+"${PASSTHRU[@]}"}
