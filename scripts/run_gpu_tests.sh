#!/bin/bash

# Eshkol GPU Test Suite
# Runs all GPU and softfloat tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
COMPILE_FAIL=0

# Results array
declare -a FAILED_TESTS
declare -a RUNTIME_ERRORS

# Exact-Ozaki certification state (reported explicitly in the summary so the
# headline exact-GEMM claim can never be silently unverified).
CERT_STATUS="not reached"
CERT_LOG="${TMPDIR:-/tmp}/eshkol_ozaki_certification_output.txt"

# Is a real GPU device present on this host? Mirrors the capability checks in
# tests/gpu/gpu_correctness_gate.sh (steps 1 and 4) rather than inventing a new
# rule. Sets GPU_SKIP_REASON when it returns nonzero.
GPU_SKIP_REASON=""
gpu_device_present() {
    GPU_SKIP_REASON=""
    case "$(uname -s)" in
        Darwin)
            if ! xcrun -sdk macosx --show-sdk-path >/dev/null 2>&1; then
                GPU_SKIP_REASON="no macOS SDK — Metal unavailable"
                return 1
            fi
            if ! otool -L "$BUILD_DIR/eshkol-run" 2>/dev/null | grep -q '/Metal\.framework/'; then
                GPU_SKIP_REASON="$BUILD_DIR/eshkol-run is not linked against Metal — configure with -DESHKOL_GPU_ENABLED=ON"
                return 1
            fi
            ;;
        Linux|MINGW*|MSYS*|CYGWIN*)
            if command -v nvidia-smi >/dev/null 2>&1; then
                if [ -z "$(nvidia-smi -L 2>/dev/null)" ]; then
                    GPU_SKIP_REASON="nvidia-smi present but reports no GPU device"
                    return 1
                fi
            elif [ -e /dev/nvidiactl ] || [ -e /dev/nvidia0 ] || [ -e /dev/nvhost-gpu ]; then
                : # Jetson/L4T: device node without nvidia-smi
            else
                GPU_SKIP_REASON="no NVIDIA device node and no nvidia-smi GPU — CUDA toolchain without a runtime device"
                return 1
            fi
            ;;
        *)
            GPU_SKIP_REASON="GPU execution is not supported on $(uname -s)"
            return 1
            ;;
    esac
    return 0
}

# The exact-Ozaki certificate. This fixture is NOT a plain pass/fail program:
# it only means anything when driven by tests/gpu/ozaki_certification_gate.sh,
# which pins the whole contract (CPU BLAS must MISMATCH the i128 oracle, Metal
# exact must match it with mismatches=0, exactly one init and one dispatch line,
# no CPU fallback) across both JIT and AOT. It used to be skipped here by
# FILENAME, unconditionally, which left the headline exact-GEMM/Ozaki-II claim
# with no automated verification anywhere. It now runs whenever a GPU device is
# actually present, and the skip is LOUD when it is not.
run_ozaki_certification() {
    printf "Testing %-50s " "ozaki_certification_test.esk"
    if ! gpu_device_present; then
        echo -e "${YELLOW}SKIPPED${NC}"
        echo -e "${YELLOW}  >>> EXACT-OZAKI CERTIFICATION NOT VERIFIED BY THIS RUN${NC}"
        echo -e "${YELLOW}  >>> reason: $GPU_SKIP_REASON${NC}"
        echo -e "${YELLOW}  >>> the exact-GEMM/Ozaki-II claim has NO evidence from this host${NC}"
        CERT_STATUS="NOT RUN — $GPU_SKIP_REASON"
        return 0
    fi
    local cert_bin
    case "$BUILD_DIR" in
        /*) cert_bin="$BUILD_DIR/eshkol-run" ;;
        *)  cert_bin="$PWD/$BUILD_DIR/eshkol-run" ;;
    esac
    if ESHKOL_RUN="$cert_bin" \
            ./tests/gpu/ozaki_certification_gate.sh > "$CERT_LOG" 2>&1; then
        if grep -q '^SKIP:' "$CERT_LOG"; then
            echo -e "${YELLOW}SKIPPED${NC}"
            echo -e "${YELLOW}  >>> EXACT-OZAKI CERTIFICATION NOT VERIFIED BY THIS RUN${NC}"
            grep '^SKIP:' "$CERT_LOG" | sed 's/^/  >>> /'
            CERT_STATUS="NOT RUN — $(grep -m1 '^SKIP:' "$CERT_LOG")"
        else
            echo -e "${GREEN}PASS${NC}"
            grep '^PASS:' "$CERT_LOG" | sed 's/^/    /'
            CERT_STATUS="VERIFIED (JIT+AOT, CPU-BLAS mismatch vs Metal exact mismatches=0)"
            ((PASS++)) || true
        fi
    else
        echo -e "${RED}CERTIFICATION FAIL${NC}"
        tail -40 "$CERT_LOG" | sed 's/^/    /'
        FAILED_TESTS+=("ozaki_certification_test.esk (certification gate)")
        CERT_STATUS="FAILED — see the gate output above"
        ((FAIL++)) || true
    fi
}

echo "========================================="
echo "  Eshkol GPU Test Suite"
echo "========================================="
echo ""

# Determine which build directory to use
# Override with: BUILD_DIR=build-cuda ./scripts/run_gpu_tests.sh
BUILD_DIR="${BUILD_DIR:-build}"

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: build directory '$BUILD_DIR' not found. Run cmake first.${NC}"
    exit 1
fi

# Check if compiler exists
if [ ! -f "$BUILD_DIR/eshkol-run" ]; then
    echo -e "${RED}Error: eshkol-run not found in '$BUILD_DIR'. Run make first.${NC}"
    exit 1
fi

echo -e "${GREEN}Using build directory: $BUILD_DIR${NC}"
echo ""
echo "Testing all files in tests/gpu/ directory..."
echo ""

# Run each test
for test_file in tests/gpu/*.esk; do
    test_name=$(basename "$test_file")
    if [ "$test_name" = "ozaki_certification_test.esk" ]; then
        run_ozaki_certification
        continue
    fi
    printf "Testing %-50s " "$test_name"

    # Clean up stale temp files before each test
    rm -f a.out a.out.tmp.o

    # Try to compile
    if ./"$BUILD_DIR"/eshkol-run "$test_file" -L./"$BUILD_DIR" > /dev/null 2>&1; then
        # Compilation succeeded, try to run
        if [ "$test_name" = "cuda_host_sync_regression_test.esk" ]; then
            runtime_cmd=(env ESHKOL_GPU_THRESHOLD=1 ESHKOL_GPU_VERBOSE=1 ./a.out)
        else
            runtime_cmd=(./a.out)
        fi

        if "${runtime_cmd[@]}" > /tmp/gpu_test_output.txt 2>&1; then
            # Check for FAIL markers in output
            if grep -qE "^FAIL:|Failed:[[:space:]]+[1-9]" /tmp/gpu_test_output.txt; then
                echo -e "${YELLOW}FAIL MARKER${NC}"
                RUNTIME_ERRORS+=("$test_name")
                ((FAIL++)) || true
            else
                echo -e "${GREEN}PASS${NC}"
                ((PASS++)) || true
            fi
        else
            echo -e "${RED}RUNTIME FAIL${NC}"
            FAILED_TESTS+=("$test_name")
            ((FAIL++)) || true
        fi
    else
        echo -e "${RED}COMPILE FAIL${NC}"
        FAILED_TESTS+=("$test_name")
        ((COMPILE_FAIL++)) || true
        ((FAIL++)) || true
    fi
done

echo ""
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
TOTAL=$(( PASS + FAIL ))
echo -e "Total Tests:        $TOTAL"
echo -e "${GREEN}Passed:             $PASS${NC}"
echo -e "${RED}Failed:             $FAIL${NC}"
echo -e "  Compile Failures: $COMPILE_FAIL"
echo -e "  Runtime Errors:   ${#RUNTIME_ERRORS[@]}"
echo ""

if [ $FAIL -gt 0 ]; then
    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo "Failed Tests:"
        for test in "${FAILED_TESTS[@]}"; do
            echo "  - $test"
        done
        echo ""
    fi

    if [ ${#RUNTIME_ERRORS[@]} -gt 0 ]; then
        echo "Tests with FAIL markers:"
        for test in "${RUNTIME_ERRORS[@]}"; do
            echo "  - $test"
        done
        echo ""
    fi
fi

if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(( PASS * 100 / TOTAL ))
    echo "Pass Rate: ${PASS_RATE}%"
fi

# Always state the exact-Ozaki certification verdict, including when it did not
# run — an unverified headline claim must be visible, not silent.
case "$CERT_STATUS" in
    VERIFIED*) echo -e "${GREEN}Ozaki exact-GEMM certification: $CERT_STATUS${NC}" ;;
    FAILED*)   echo -e "${RED}Ozaki exact-GEMM certification: $CERT_STATUS${NC}" ;;
    *)         echo -e "${YELLOW}Ozaki exact-GEMM certification: $CERT_STATUS${NC}" ;;
esac

echo ""

# Clean up
rm -f /tmp/gpu_test_output.txt "$CERT_LOG" a.out a.out.tmp.o

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
