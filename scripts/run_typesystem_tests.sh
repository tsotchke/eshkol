#!/bin/bash

# Eshkol Type System Negative Test Suite
# Tests that the type checker correctly detects and reports errors.
#
# Each .esk test file has header comments declaring expectations:
#   ;; EXPECT-MODE: strict-types | unsafe | default
#   ;; EXPECT-STDERR: <pattern>         (this pattern must appear in stderr)
#   ;; EXPECT-NO-STDERR: <pattern>      (this pattern must NOT appear in stderr)
#
# A test PASSES when all EXPECT-STDERR patterns are found
# and all EXPECT-NO-STDERR patterns are absent.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
PASS=0
FAIL=0

declare -a FAILED_TESTS

echo "========================================="
echo "  Eshkol Type System Test Suite"
echo "========================================="
echo ""

# Ensure build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}Error: build directory not found. Run cmake first.${NC}"
    exit 1
fi

if [ ! -f "build/eshkol-run" ]; then
    echo -e "${RED}Error: eshkol-run not found. Run make first.${NC}"
    exit 1
fi

echo "Testing type checker enforcement..."
echo ""

for test_file in tests/typesystem/*.esk; do
    [ -e "$test_file" ] || continue

    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    # Clean up stale temp files
    rm -f a.out a.out.tmp.o

    # Extract metadata from header comments
    mode=$(grep '^;; EXPECT-MODE:' "$test_file" | head -1 | sed 's/;; EXPECT-MODE: *//')

    # Build compiler flags
    flags="-L./build"
    case "$mode" in
        strict-types) flags="$flags --strict-types" ;;
        unsafe)       flags="$flags --unsafe" ;;
        default)      ;; # no extra flags
        *)            ;; # no extra flags
    esac

    # Compile, capturing stderr separately
    ./build/eshkol-run "$test_file" $flags -o /tmp/typesystem_test_bin > /dev/null 2>/tmp/typesystem_test_stderr.txt
    compile_exit=$?

    # Check all EXPECT-STDERR patterns
    test_passed=true

    while IFS= read -r line; do
        pattern=$(echo "$line" | sed 's/;; EXPECT-STDERR: *//')
        if [ -n "$pattern" ]; then
            if ! grep -qF "$pattern" /tmp/typesystem_test_stderr.txt 2>/dev/null; then
                test_passed=false
            fi
        fi
    done < <(grep '^;; EXPECT-STDERR:' "$test_file")

    # Check all EXPECT-NO-STDERR patterns
    while IFS= read -r line; do
        pattern=$(echo "$line" | sed 's/;; EXPECT-NO-STDERR: *//')
        if [ -n "$pattern" ]; then
            if grep -qF "$pattern" /tmp/typesystem_test_stderr.txt 2>/dev/null; then
                test_passed=false
            fi
        fi
    done < <(grep '^;; EXPECT-NO-STDERR:' "$test_file")

    if $test_passed; then
        echo -e "${GREEN}PASS${NC}"
        ((PASS++)) || true
    else
        echo -e "${RED}FAIL${NC}"
        FAILED_TESTS+=("$test_name")
        ((FAIL++)) || true
        # Show stderr for debugging
        if [ -s /tmp/typesystem_test_stderr.txt ]; then
            echo "    stderr: $(head -3 /tmp/typesystem_test_stderr.txt)"
        else
            echo "    stderr: (empty)"
        fi
    fi
done

echo ""
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
echo -e "Total Tests:    $(( PASS + FAIL ))"
echo -e "${GREEN}Passed:         $PASS${NC}"
echo -e "${RED}Failed:         $FAIL${NC}"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
fi

TOTAL=$(( PASS + FAIL ))
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(( PASS * 100 / TOTAL ))
    echo ""
    echo "Pass Rate: ${PASS_RATE}%"
fi

echo ""

# Clean up
rm -f /tmp/typesystem_test_stderr.txt /tmp/typesystem_test_bin a.out

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    exit 1
fi
