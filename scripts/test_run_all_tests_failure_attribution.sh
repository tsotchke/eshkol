#!/usr/bin/env bash
# Self-test for the complete-suite failure attribution contract.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
. "$SCRIPT_DIR/lib/test_failure_attribution.sh"

WORK_DIR="$REPO_ROOT/.scratch/failure-attribution-self-test"
mkdir -p "$WORK_DIR"
PASS=0
FAIL=0

check_records() {
    local name="$1"
    local input="$2"
    local expected="$3"
    local actual

    actual="$(eshkol_extract_failure_records self-test "$input")"
    if [ "$actual" = "$expected" ]; then
        echo "PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $name"
        echo "  expected: $expected"
        echo "  actual:   $actual"
        FAIL=$((FAIL + 1))
    fi
}

PASSING_THEN_BARE="$WORK_DIR/passing-then-bare.log"
cat > "$PASSING_THEN_BARE" <<'EOF'
Testing frechet_mean_surface_regression.esk PASS
FAIL
EOF
check_records "bare aggregate FAIL is not attached to the last passing test" \
    "$PASSING_THEN_BARE" ""

SAME_LINE="$WORK_DIR/same-line.log"
cat > "$SAME_LINE" <<'EOF'
Testing actual_failure.esk RUNTIME FAIL
EOF
check_records "same-line failure keeps its named test" \
    "$SAME_LINE" "self-test	actual_failure.esk	RUNTIME FAIL"

SPLIT_CRASH="$WORK_DIR/split-crash.log"
cat > "$SPLIT_CRASH" <<'EOF'
Testing process_tree_surface_regression.esk 
RUNTIME FAIL (exit 139)
EOF
check_records "split crash uses the immediately preceding test" \
    "$SPLIT_CRASH" "self-test	process_tree_surface_regression.esk	RUNTIME FAIL"

STALE_SPLIT="$WORK_DIR/stale-split.log"
cat > "$STALE_SPLIT" <<'EOF'
Testing passing_test.esk PASS
diagnostic output from the passing test
RUNTIME FAIL (exit 139)
EOF
check_records "split crash does not use a stale test header" \
    "$STALE_SPLIT" ""

NAMED_LIST="$WORK_DIR/named-list.log"
cat > "$NAMED_LIST" <<'EOF'
Testing frechet_mean_surface_regression.esk PASS
  assertion: FAIL
Failed Tests:
  - actual_failure.esk
EOF
check_records "named failure list keeps assertion output tied to its listed test" \
    "$NAMED_LIST" "self-test	actual_failure.esk	FAIL"

GENERIC_NAME="$WORK_DIR/generic-name.log"
cat > "$GENERIC_NAME" <<'EOF'
  logic_case FAIL (assertion)
EOF
check_records "legacy stem-only result keeps its explicit test name" \
    "$GENERIC_NAME" "self-test	logic_case	FAIL"

DUPLICATE="$WORK_DIR/duplicate.log"
cat > "$DUPLICATE" <<'EOF'
Testing actual_failure.esk RUNTIME FAIL
Failed Tests:
  - actual_failure.esk
EOF
check_records "same test is emitted once when result and summary both name it" \
    "$DUPLICATE" "self-test	actual_failure.esk	RUNTIME FAIL"

rm -f -- "$PASSING_THEN_BARE" "$SAME_LINE" "$SPLIT_CRASH" "$STALE_SPLIT" \
    "$NAMED_LIST" "$GENERIC_NAME" "$DUPLICATE"
rmdir "$WORK_DIR" 2>/dev/null || true

if [ "$FAIL" -eq 0 ]; then
    echo "failure-attribution self-test: PASS ($PASS checks)"
    exit 0
fi

echo "failure-attribution self-test: FAIL ($FAIL checks failed)"
exit 1
