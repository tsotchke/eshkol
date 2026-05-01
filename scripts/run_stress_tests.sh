#!/bin/bash
# run_stress_tests.sh — drive the long-running stress harnesses.
#
# Not wired into scripts/run_all_tests.sh because these take minutes
# to hours. Use from CI separately, e.g.:
#
#   bash scripts/run_stress_tests.sh                 # each harness once
#   bash scripts/run_stress_tests.sh --repeat 10     # ten laps
#   bash scripts/run_stress_tests.sh --hours 24      # repeat for ≤24 h
#
# Exits 0 iff every harness in every lap prints `RESULT: OK`.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

ESHKOL_RUN="${ESHKOL_RUN:-./build/eshkol-run}"
if [[ ! -x "$ESHKOL_RUN" ]]; then
    echo "eshkol-run not found at $ESHKOL_RUN — build first." >&2
    exit 2
fi

REPEAT=1
HOURS=0
while (( "$#" )); do
    case "$1" in
        --repeat) REPEAT="$2"; shift 2 ;;
        --hours)  HOURS="$2";  shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ "$HOURS" != "0" ]]; then
    DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
else
    DEADLINE=0
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
FAIL=0; PASS=0; LAP=0

HARNESSES=(
    tests/stress/stress_fd_exhaustion.esk
    tests/stress/stress_alloc_loop.esk
    tests/stress/stress_parallel_at_scale.esk
)

while true; do
    LAP=$((LAP+1))
    echo ""
    echo "=== Stress lap $LAP ==="
    for h in "${HARNESSES[@]}"; do
        printf "  %-50s " "$(basename $h)"
        if "$ESHKOL_RUN" -r "$h" 2>/dev/null | tail -1 | grep -q "RESULT: OK"; then
            echo -e "${GREEN}PASS${NC}"
            PASS=$((PASS+1))
        else
            echo -e "${RED}FAIL${NC}"
            FAIL=$((FAIL+1))
        fi
    done
    if [[ "$LAP" -ge "$REPEAT" && "$HOURS" == "0" ]]; then break; fi
    if [[ "$DEADLINE" != "0" && "$(date +%s)" -ge "$DEADLINE" ]]; then break; fi
done

echo ""
echo "========================================="
echo "  Stress run complete: $LAP lap(s)"
echo "  Passed: $PASS   Failed: $FAIL"
echo "========================================="

[[ "$FAIL" == "0" ]] && exit 0 || exit 1
