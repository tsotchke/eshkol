#!/usr/bin/env bash
# run_generative_differential.sh — thin wrapper around the generative
# multi-oracle differential harness (adversarial testing pillar P7c).
#
# Exists so the harness can be an ICC `action:` and a scripts/run_icc_smoke.sh
# probe with a stable command line. All real logic lives in the Python harness
# (scripts/run_generative_differential.py); see tests/generative-diff/README.md.
#
# Default invocation runs in regression mode against the committed baseline
# (tests/generative-diff/baseline.txt): exit 0 iff no NEW divergence appears.
# Pass any flags to override (e.g. --count 60 with no --baseline for a full
# discovery run that is RED while any divergence remains).
#
# Usage:
#   scripts/run_generative_differential.sh                 # smoke + baseline
#   scripts/run_generative_differential.sh --count 60      # full discovery
#   scripts/run_generative_differential.sh <any harness flags...>
set -u

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
BASELINE="$REPO_ROOT/tests/generative-diff/baseline.txt"

if [ "$#" -eq 0 ]; then
    exec python3 "$REPO_ROOT/scripts/run_generative_differential.py" \
        --smoke --baseline "$BASELINE"
fi

exec python3 "$REPO_ROOT/scripts/run_generative_differential.py" "$@"
