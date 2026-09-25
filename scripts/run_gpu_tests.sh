#!/usr/bin/env bash

# Eshkol GPU Test Suite
# Runs all GPU and softfloat tests
#
# GPU verdict contract: exit 0 and a terminal "PASS: <test name>" are
# required. A FAIL token anywhere (including bare or indented FAIL) fails.
# A terminal SKIP: is counted separately. The correctness payload is run by
# gpu_correctness_gate.sh, which compares GPU execution with a CPU reference.
# The must-fail canary and --self-test exercise both failure paths.

set -e

# Per-run, per-repo-root isolation for temp files and build artifacts.
# Two suites (two worktrees, two agents, CI plus a local run) must never share
# a scratch path or a build artifact — see scripts/lib/test_isolation.sh.
# Sourcing must be checked *before* the fact: bash 3.2 (macOS) exits the
# shell when `source` cannot find its file, so a trailing `|| {...}` never
# runs there. A suite with no prelude has no failure detection and no
# scratch isolation, and must refuse to run rather than report a PASS.
ESHKOL_TEST_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test_isolation.sh"
if [ ! -r "$ESHKOL_TEST_LIB" ]; then
    echo "FATAL: cannot read $ESHKOL_TEST_LIB" >&2
    echo "       (the shared test isolation and failure-detection prelude)." >&2
    echo "       Refusing to run: without it this suite would report a" >&2
    echo "       meaningless PASS." >&2
    exit 2
fi
source "$ESHKOL_TEST_LIB"
# shellcheck source=lib/checked_write.sh
. "$(dirname "$ESHKOL_TEST_LIB")/checked_write.sh"
ESHKOL_TEST_TMP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.scratch"
mkdir -p "$ESHKOL_TEST_TMP_ROOT"
export ESHKOL_TEST_TMP_ROOT
eshkol_test_isolation_init "gpu"

# One parser grades both GPU programs and the differential shell gate. Its
# return codes are 0=PASS, 1=FAIL, 2=SKIP. A success line from an earlier
# check cannot certify a program that stopped before its final verdict.
gpu_verdict() { # output-file, required pass line
    local file="$1" expected="$2"
    [ -f "$file" ] || return 1
    LC_ALL=C awk -v expected="PASS: $expected" '
        /[^[:space:]]/ { last=$0; sub(/^[[:space:]]+/, "", last); sub(/[[:space:]]+$/, "", last) }
        /(^|[^[:alnum:]_])FAIL(:|[[:space:]]|$)/ { failed=1 }
        /^[[:space:]]*SKIP:/ { skipped=1 }
        END {
            if (failed) exit 1
            if (last ~ /^SKIP: /) exit 2
            if (skipped) exit 1
            if (last == expected) exit 0
            exit 1
        }
    ' "$file"
}

if [ "${1:-}" = "--self-test" ]; then
    probe="$ESHKOL_TEST_TMPDIR/verdict-probe.txt"
    gpu_probe() { # label, expected return code, output, optional pass name
        local label="$1" want="$2" output="$3" name="${4:-gpu_probe}" got=0
        printf '%s\n' "$output" > "$probe"
        gpu_verdict "$probe" "$name" || got=$?
        if [ "$got" -ne "$want" ]; then
            echo "FAIL: $label (got $got, expected $want)"
            exit 1
        fi
        echo "ok: $label"
    }
    gpu_probe "terminal pass" 0 'PASS: gpu_probe'
    gpu_probe "bare FAIL after an earlier pass" 1 $'PASS: gpu_probe\nFAIL'
    gpu_probe "indented FAIL marker" 1 $'PASS: gpu_probe\n  check: FAIL\nPASS: gpu_probe'
    gpu_probe "missing terminal pass" 1 $'PASS: gpu_probe\nRESULT checksum 1'
    gpu_probe "single-run payload cannot certify differential" 1 'PASS: gpu_correctness_gate self-checks' 'gpu_correctness_gate.sh'
    gpu_probe "differential failure overrides pass" 1 $'FAIL: GPU-vs-CPU mismatch\nPASS: gpu_correctness_gate.sh' 'gpu_correctness_gate.sh'
    gpu_probe "absent GPU is skipped" 2 'SKIP: no GPU device'
    echo 'PASS: run_gpu_tests.sh --self-test'
    exit 0
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
SKIP=0
COMPILE_FAIL=0

# Results array
declare -a FAILED_TESTS
declare -a RUNTIME_ERRORS

# Exact-Ozaki certification state (reported explicitly in the summary so the
# headline exact-GEMM claim can never be silently unverified).
CERT_STATUS="not reached"
CERT_LOG="$ESHKOL_TEST_TMPDIR/ozaki_certification_output.txt"

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
        ((SKIP++)) || true
        return 0
    fi
    local cert_bin
    case "$BUILD_DIR" in
        /*) cert_bin="$BUILD_DIR/eshkol-run" ;;
        *)  cert_bin="$PWD/$BUILD_DIR/eshkol-run" ;;
    esac
    local cert_rc=0 cert_verdict=0
    ESHKOL_RUN="$cert_bin" TMPDIR="$ESHKOL_TEST_REPO_ROOT/.scratch" \
        ./tests/gpu/ozaki_certification_gate.sh > "$CERT_LOG" 2>&1 || cert_rc=$?
    gpu_verdict "$CERT_LOG" "ozaki_certification_gate.sh" || cert_verdict=$?
    if [ "$cert_rc" -eq 0 ] && [ "$cert_verdict" -ne 1 ]; then
        if [ "$cert_verdict" -eq 2 ]; then
            echo -e "${YELLOW}SKIPPED${NC}"
            echo -e "${YELLOW}  >>> EXACT-OZAKI CERTIFICATION NOT VERIFIED BY THIS RUN${NC}"
            grep '^SKIP:' "$CERT_LOG" | sed 's/^/  >>> /'
            CERT_STATUS="NOT RUN — $(grep -m1 '^SKIP:' "$CERT_LOG")"
            ((SKIP++)) || true
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

# The gate canary — see the VERDICT CONTRACT header, item 4. Runs before the
# main suite and is excluded from it (it is SUPPOSED to fail, so it must not
# be graded by the normal PASS-required loop below). CANARY_HARD_FAIL forces
# this script's own exit code to 1 at the very end regardless of every other
# result: a harness that cannot fail cannot be trusted to have passed.
CANARY_NAME="gate_canary_must_fail.esk"
CANARY_HARD_FAIL=0
run_gate_canary() {
    local canary_path="tests/gpu/$CANARY_NAME"
    printf "Canary  %-50s " "$CANARY_NAME (must fail)"

    if [ ! -f "$canary_path" ]; then
        echo -e "${RED}MISSING${NC}"
        echo -e "${RED}  >>> cannot prove this harness can still fail a real defect${NC}"
        CANARY_HARD_FAIL=1
        return 0
    fi

    eshkol_test_reset_bin
    if ! ./"$BUILD_DIR"/eshkol-run "$canary_path" -L./"$BUILD_DIR" -o "$ESHKOL_TEST_BIN" > /dev/null 2>&1; then
        echo -e "${RED}COMPILE FAIL${NC}"
        echo -e "${RED}  >>> the canary must compile and then fail at runtime — it did neither${NC}"
        CANARY_HARD_FAIL=1
        return 0
    fi

    local canary_rc=0
    eshkol_require_output_file_path "$ESHKOL_TEST_OUT"
    "$ESHKOL_TEST_BIN" > "$ESHKOL_TEST_OUT" 2>&1 || canary_rc=$?

    if [ "$canary_rc" -eq 0 ]; then
        echo -e "${RED}DID NOT FAIL (exit 0)${NC}"
        echo -e "${RED}  >>> VERDICT PIPELINE IS BROKEN: a deliberately-wrong result exited 0${NC}"
        CANARY_HARD_FAIL=1
        return 0
    fi

    if ! eshkol_test_output_has_failure "$ESHKOL_TEST_OUT"; then
        echo -e "${RED}FAILED WITH NO FAIL: MARKER${NC}"
        echo -e "${RED}  >>> exit code was non-zero but no FAIL: line was printed —${NC}"
        echo -e "${RED}  >>> the marker grammar itself is broken${NC}"
        CANARY_HARD_FAIL=1
        return 0
    fi

    echo -e "${GREEN}RED as expected (exit $canary_rc)${NC}"
    echo -e "${GREEN}  >>> harness confirmed live: a failing kernel does turn this gate red${NC}"
}

# tests/gpu/gpu_correctness_gate.sh's SKIP path deliberately writes NO trace
# record to scripts/icc_traces/gpu_execution.jsonl (see its header comment and
# .icc/completion-oracles.yaml's `gpu-execution` criterion, severity: high) —
# that silence is what lets the oracle read a GPU-less host as "no evidence
# yet" instead of a false PASS. Its `--self-test` mode asserts that contract
# directly (skip() writes nothing; fail()/a PASS record are distinguishable),
# and tests a planted numeric mismatch, with no build and no GPU required, so a future edit that makes SKIP look
# like PASS is caught here rather than by an oracle silently regressing.
run_gate_self_test() {
    printf "Canary  %-50s " "gpu_correctness_gate.sh --self-test"
    if [ ! -x tests/gpu/gpu_correctness_gate.sh ]; then
        echo -e "${RED}MISSING/NOT EXECUTABLE${NC}"
        CANARY_HARD_FAIL=1
        return 0
    fi
    eshkol_require_output_file_path "$ESHKOL_TEST_OUT"
    if tests/gpu/gpu_correctness_gate.sh --self-test > "$ESHKOL_TEST_OUT" 2>&1; then
        echo -e "${GREEN}PASS${NC}"
    else
        echo -e "${RED}FAIL${NC}"
        sed 's/^/    /' "$ESHKOL_TEST_OUT"
        CANARY_HARD_FAIL=1
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

run_gate_canary
run_gate_self_test
echo ""

echo "Testing all files in tests/gpu/ directory..."
echo ""

# Run each test
for test_file in tests/gpu/*.esk; do
    test_name=$(basename "$test_file")
    if [ "$test_name" = "$CANARY_NAME" ]; then
        continue  # handled by run_gate_canary above, not graded here
    fi
    if [ "$test_name" = "gpu_correctness_gate.esk" ]; then
        continue  # payload is graded by the CPU-vs-GPU differential below
    fi
    if [ "$test_name" = "ozaki_certification_test.esk" ]; then
        run_ozaki_certification
        continue
    fi
    # This regression executes metal_softfloat.h, which is not built by the
    # CUDA/CPU backends. Do not count absent Metal execution as a pass.
    if [ "$test_name" = "sf64_div_quotient_regression_test.esk" ] &&
       [ "$(uname -s)" != "Darwin" ]; then
        printf "Testing %-50s " "$test_name"
        echo -e "${YELLOW}SKIPPED — Metal-only regression; not measured on this platform${NC}"
        continue
    fi
    printf "Testing %-50s " "$test_name"

    # Clean up stale temp files before each test
    eshkol_test_reset_bin
    # Try to compile
    if ./"$BUILD_DIR"/eshkol-run "$test_file" -L./"$BUILD_DIR" -o "$ESHKOL_TEST_BIN" > /dev/null 2>&1; then
        # Compilation succeeded, try to run
        if [ "$test_name" = "cuda_ozaki_correctness_test.esk" ]; then
            runtime_cmd=(env ESHKOL_GPU_THRESHOLD=1 ESHKOL_GPU_VERBOSE=1 ESHKOL_CUDA_F64_KERNEL=ozaki-int8 "$ESHKOL_TEST_BIN")
        elif [ "$test_name" = "cuda_host_sync_regression_test.esk" ]; then
            runtime_cmd=(env ESHKOL_GPU_THRESHOLD=1 ESHKOL_GPU_VERBOSE=1 "$ESHKOL_TEST_BIN")
        else
            runtime_cmd=("$ESHKOL_TEST_BIN")
        fi

        eshkol_require_output_file_path "$ESHKOL_TEST_OUT"
        if "${runtime_cmd[@]}" > "$ESHKOL_TEST_OUT" 2>&1; then
            verdict=0
            gpu_verdict "$ESHKOL_TEST_OUT" "${test_name%.esk}" || verdict=$?
            case "$verdict" in
                0) echo -e "${GREEN}PASS${NC}"; ((PASS++)) || true ;;
                2) echo -e "${YELLOW}SKIPPED${NC}"; tail -1 "$ESHKOL_TEST_OUT" | sed 's/^/    /'; ((SKIP++)) || true ;;
                *) echo -e "${RED}FAIL/NO TERMINAL VERDICT${NC}"
                   tail -12 "$ESHKOL_TEST_OUT" | sed 's/^/    /'
                   FAILED_TESTS+=("$test_name")
                   ((FAIL++)) || true ;;
            esac
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

printf "Testing %-50s " "gpu_correctness_gate.sh (GPU vs CPU)"
diff_rc=0
diff_gpu_dir="$ESHKOL_TEST_TMPDIR/gpu-enabled"
diff_cpu_dir="$ESHKOL_TEST_TMPDIR/gpu-cpuref"
diff_reuse=0
# The supplied build is a valid reference for its configured backend. Reuse
# it on that side of the differential; configure only the opposite backend.
if grep -q '^ESHKOL_GPU_ENABLED:BOOL=OFF$' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    diff_cpu_dir="$BUILD_DIR"
    diff_reuse=1
elif grep -q '^ESHKOL_GPU_ENABLED:BOOL=ON$' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    diff_gpu_dir="$BUILD_DIR"
    diff_reuse=1
fi
case "$BUILD_DIR" in
    /*) diff_deps_dir="$BUILD_DIR/_deps" ;;
    *)  diff_deps_dir="$ESHKOL_TEST_REPO_ROOT/$BUILD_DIR/_deps" ;;
esac
BUILD_DIR_GPU="$diff_gpu_dir" BUILD_DIR_CPU="$diff_cpu_dir" REUSE_BUILDS="$diff_reuse" \
    GPU_GATE_SOURCE_DEPS_ROOT="$diff_deps_dir" \
    ESHKOL_TEST_TMP_ROOT="$ESHKOL_TEST_REPO_ROOT/.scratch" \
    ./tests/gpu/gpu_correctness_gate.sh > "$ESHKOL_TEST_OUT" 2>&1 || diff_rc=$?
diff_verdict=0
gpu_verdict "$ESHKOL_TEST_OUT" "gpu_correctness_gate.sh" || diff_verdict=$?
if [ "$diff_rc" -ne 0 ] || [ "$diff_verdict" -eq 1 ]; then
    echo -e "${RED}FAIL${NC}"
    tail -20 "$ESHKOL_TEST_OUT" | sed 's/^/    /'
    FAILED_TESTS+=("gpu_correctness_gate.sh")
    ((FAIL++)) || true
elif [ "$diff_verdict" -eq 2 ]; then
    echo -e "${YELLOW}SKIPPED — GPU execution not certified${NC}"
    tail -1 "$ESHKOL_TEST_OUT" | sed 's/^/    /'
    ((SKIP++)) || true
else
    echo -e "${GREEN}PASS${NC}"
    ((PASS++)) || true
fi

echo ""
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
TOTAL=$(( PASS + FAIL + SKIP ))
echo -e "Total Tests:        $TOTAL"
echo -e "${GREEN}Passed:             $PASS${NC}"
echo -e "${YELLOW}Skipped:            $SKIP${NC}"
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

if [ "$CANARY_HARD_FAIL" -eq 1 ]; then
    echo -e "${RED}Gate canary + self-test: at least one verdict-pipeline check FAILED — see above${NC}"
else
    echo -e "${GREEN}Gate canary ($CANARY_NAME): confirmed RED, and gpu_correctness_gate.sh --self-test: PASS${NC}"
fi

echo ""

# Clean up
# CERT_LOG now lives inside $ESHKOL_TEST_TMPDIR, which the isolation trap
# removes wholesale, so it needs no separate unlink here.
eshkol_checked_rm "$ESHKOL_TEST_OUT" "$ESHKOL_TEST_BIN" "$ESHKOL_TEST_BIN.tmp.o"

# Exit with appropriate code. The canary is checked independently of FAIL: a
# harness that cannot prove it can fail must not report success regardless of
# how many real tests passed — see VERDICT CONTRACT item 4.
if [ "$CANARY_HARD_FAIL" -eq 1 ]; then
    echo -e "${RED}FATAL: gate canary did not fail — refusing to report this run as trustworthy.${NC}"
    exit 1
fi

if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
