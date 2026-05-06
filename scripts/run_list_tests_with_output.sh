#!/bin/bash

# Test Suite Validation Script with Output Capture
# Runs all tests and saves their output for verification

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/test_output_helpers.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
COMPILE_FAIL=0

# Output directory
OUTPUT_DIR="list_test_outputs"
test_output_prepare_dir "$OUTPUT_DIR"

# Results file
RESULTS_FILE="$OUTPUT_DIR/test_results_summary.txt"
test_output_reset_file "$RESULTS_FILE" "$OUTPUT_DIR" "test results file"

# Results arrays
declare -a FAILED_TESTS
declare -a RUNTIME_ERRORS

echo "========================================="
echo "  Eshkol Test Suite with Output Capture"
echo "========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Honour $BUILD_DIR (CI passes it via the matrix); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: build directory not found. Run cmake first.${NC}"
    exit 1
fi

# Check if compiler exists
if [ ! -f "$BUILD_DIR/eshkol-run" ]; then
    echo -e "${RED}Error: eshkol-run not found. Run make first.${NC}"
    exit 1
fi

echo "Testing all files in tests/lists/ directory..."
echo ""

# Run each test
for test_file in tests/lists/*.esk; do
    test_name=$(basename "$test_file")
    output_file="$OUTPUT_DIR/${test_name%.esk}_output.txt"
    
    printf "Testing %-50s " "$test_name"
    
    # Clear output file
    test_output_reset_file "$output_file" "$OUTPUT_DIR" "list test output file"
    compile_output_file="${output_file}.compile"
    test_output_reset_file "$compile_output_file" "$OUTPUT_DIR" "list compile output file"
    
    # Add header to output file
    echo "========================================" >> "$output_file"
    echo "Test: $test_name" >> "$output_file"
    echo "Time: $(date)" >> "$output_file"
    echo "========================================" >> "$output_file"
    echo "" >> "$output_file"
    
    # Try to compile
    if ./"$BUILD_DIR"/eshkol-run -L./"$BUILD_DIR" "$test_file" > "$compile_output_file" 2>&1; then
        # Compilation succeeded, try to run
        echo "COMPILATION: SUCCESS" >> "$output_file"
        echo "" >> "$output_file"
        echo "OUTPUT:" >> "$output_file"
        echo "----------------------------------------" >> "$output_file"
        
        if ./a.out >> "$output_file" 2>&1; then
            # Check if there were any errors in output
            if grep -q "error:" "$output_file"; then
                echo -e "${YELLOW}⚠ RUNTIME ERROR${NC}"
                echo "STATUS: RUNTIME ERROR" >> "$output_file"
                RUNTIME_ERRORS+=("$test_name")
                ((FAIL++))
                
                # Add to summary
                test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "❌ $test_name - RUNTIME ERROR"
            else
                echo -e "${GREEN}✅ PASS${NC}"
                echo "STATUS: PASS" >> "$output_file"
                ((PASS++))
                
                # Add to summary
                test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "✅ $test_name - PASS"
            fi
        else
            echo -e "${RED}❌ RUNTIME FAIL${NC}"
            echo "STATUS: RUNTIME FAIL (segfault or crash)" >> "$output_file"
            FAILED_TESTS+=("$test_name")
            ((FAIL++))
            
            # Add to summary
            test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "❌ $test_name - RUNTIME FAIL"
        fi
    else
        echo -e "${RED}❌ COMPILE FAIL${NC}"
        echo "COMPILATION: FAILED" >> "$output_file"
        echo "" >> "$output_file"
        cat "$compile_output_file" >> "$output_file"
        FAILED_TESTS+=("$test_name")
        ((COMPILE_FAIL++))
        ((FAIL++))
        
        # Add to summary
        test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "❌ $test_name - COMPILE FAIL"
    fi
    
    # Clean up compile output
    rm -f -- "$compile_output_file"
done

echo ""
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
echo -e "Total Tests:    $(( PASS + FAIL ))"
echo -e "${GREEN}Passed:         $PASS${NC}"
echo -e "${RED}Failed:         $FAIL${NC}"
echo -e "  Compile Failures: $COMPILE_FAIL"
echo -e "  Runtime Errors:   ${#RUNTIME_ERRORS[@]}"
echo ""

# Add summary to results file
test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" ""
test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "========================================="
test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "SUMMARY"
test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "========================================="
test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "Total Tests: $(( PASS + FAIL ))"
test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "Passed: $PASS"
test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "Failed: $FAIL"
test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "  Compile Failures: $COMPILE_FAIL"
test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "  Runtime Errors: ${#RUNTIME_ERRORS[@]}"

if [ $FAIL -gt 0 ]; then
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
    
    if [ ${#RUNTIME_ERRORS[@]} -gt 0 ]; then
        echo "Runtime Errors:"
        for test in "${RUNTIME_ERRORS[@]}"; do
            echo "  - $test"
        done
        echo ""
    fi
fi

# Calculate pass rate
TOTAL=$(( PASS + FAIL ))
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(( PASS * 100 / TOTAL ))
    echo "Pass Rate: ${PASS_RATE}%"
    test_output_append_line "$RESULTS_FILE" "$OUTPUT_DIR" "Pass Rate: ${PASS_RATE}%"
fi

echo ""
echo -e "${BLUE}Test outputs saved in: $OUTPUT_DIR${NC}"
echo -e "${BLUE}Summary file: $RESULTS_FILE${NC}"
echo ""

# Clean up
rm -f a.out

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
