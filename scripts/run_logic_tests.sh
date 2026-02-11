#!/bin/bash
# Run Eshkol Consciousness Engine tests — v1.1-accelerate
set -e

ESHKOL="./build/eshkol-run"
PASS=0
FAIL=0

echo "=== Eshkol Consciousness Engine Tests ==="
echo ""

for test in tests/logic/*.esk; do
    name=$(basename "$test" .esk)
    printf "  %-30s " "$name"

    # Compile
    if $ESHKOL "$test" -o /tmp/eshkol_logic_test 2>/dev/null; then
        # Run
        if /tmp/eshkol_logic_test >/tmp/eshkol_logic_output 2>&1; then
            echo "PASS"
            PASS=$((PASS + 1))
        else
            echo "FAIL (runtime error)"
            cat /tmp/eshkol_logic_output
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL (compile error)"
        $ESHKOL "$test" -o /tmp/eshkol_logic_test 2>&1 | head -20
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit $FAIL
