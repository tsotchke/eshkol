#!/usr/bin/env bash

# Eshkol Generated-Code Optimization-Level Test Suite
#
# Regression guard for the "sleeper" O0 default (ADR 0007, Phase 0:
# "Make every performance build record and assert generated-code O3").
#
# A CMake Release build of the compiler does NOT imply optimized *emitted*
# Eshkol code -- the backend optimization plane is independent of how the
# compiler was built. Historically the default was O0, so even
# `eshkol-run file.esk -o bin` shipped unoptimized binaries. These assertions
# pin the intended behaviour so a regression is caught:
#
#   * artifact-producing paths (-o / -c / --shared-lib) default to O2;
#   * ephemeral run paths (plain run, no -o) stay at O0 for fast turnaround;
#   * -g (debug) stays unoptimized unless -O is explicit;
#   * an explicit -O<n> always wins.
#
# The backend prints its applied level via the -d info log
# ("Applied LLVM optimization passes at -O<n>" / "Applied LLVM O0 cleanup
# passes"), which this suite parses as the observable.

# NOTE: no `set -e` -- we tally failures and report a summary like the
# other suites so run_all_tests.sh can aggregate our Passed/Failed counts.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
declare -a FAILED_TESTS

echo "========================================="
echo "  Eshkol Generated-Code Opt-Level Tests"
echo "========================================="
echo ""

BUILD_DIR="${BUILD_DIR:-build}"
if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/eshkol-run" ]; then
    echo -e "${RED}Error: Build directory not found or eshkol-run missing.${NC}"
    echo "Please build first: cd build && cmake .. && make -j8"
    exit 1
fi
RUN="./$BUILD_DIR/eshkol-run"
# Absolute forms, so each check can run with its cwd inside this run's own
# scratch directory. The "plain run (no -o)" case is *defined* by having no
# -o, so eshkol-run writes an implicit a.out next to the cwd — which used to
# be the repository root, i.e. the shared artifact every other suite was
# fighting over. Same assertion, private cwd.
BUILD_DIR_ABS="$(cd "$BUILD_DIR" && pwd)"
RUN_ABS="$BUILD_DIR_ABS/eshkol-run"
echo -e "${GREEN}Using build directory: $BUILD_DIR${NC}"
echo ""

# Per-run, per-repo-root isolation and the shared honest-detection helpers.
ESHKOL_TEST_ISOLATION_NO_TRAP=1
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
eshkol_test_isolation_init "codegen-optlevel"
WORK="$ESHKOL_TEST_TMPDIR/work"
mkdir -p "$WORK"
trap eshkol_test_isolation_cleanup EXIT

PROBE="$WORK/optlevel_probe.esk"
printf '%s\n' \
    '(define (sq x) (* x x))' \
    '(display (sq 21))' \
    '(newline)' | tee "$PROBE" >/dev/null

# Assert that the applied backend opt level matches an expected marker.
#   $1 = human-readable case name
#   $2 = expected marker (extended regex) in the -d log
#   $3.. = eshkol-run arguments
check_level() {
    local name="$1"; shift
    local expected="$1"; shift
    printf "Testing %-52s " "$name"

    local log="$WORK/log.txt"
    ( cd "$WORK" && "$RUN_ABS" -d -L"$BUILD_DIR_ABS" "$@" ) > "$log" 2>&1

    if grep -qE "$expected" "$log"; then
        echo -e "${GREEN}PASS${NC}"
        ((PASS++)) || true
    else
        echo -e "${RED}ASSERTION FAIL${NC}"
        echo "    expected marker: $expected"
        echo "    got:"
        grep -iE "Applied LLVM|cleanup passes" "$log" | head -3 | sed 's/^/      /'
        FAILED_TESTS+=("$name")
        ((FAIL++)) || true
    fi
}

# Assert that NO optimizing pipeline was applied (O0 / debug path). The O0
# path prints "Applied LLVM O0 cleanup passes" (or nothing on targets that
# skip cleanup), so the invariant is: never an "at -O2/-O3" line.
check_not_optimized() {
    local name="$1"; shift
    printf "Testing %-52s " "$name"

    local log="$WORK/log.txt"
    local rc=0
    ( cd "$WORK" && "$RUN_ABS" -d -L"$BUILD_DIR_ABS" "$@" ) > "$log" 2>&1 || rc=$?

    # A compiler that crashed produces no "-O2/-O3" line either, so the plain
    # "marker absent ⇒ PASS" test scored a failed compile as an assertion pass.
    # Require the command to have succeeded and to have said *something*.
    if [ "$rc" -ne 0 ]; then
        echo -e "${RED}COMPILE FAIL (exit $rc)${NC}"
        tail -5 "$log" | sed 's/^/      /'
        FAILED_TESTS+=("$name")
        ((FAIL++)) || true
        return
    fi
    if eshkol_test_output_is_silent "$log"; then
        echo -e "${RED}NO DIAGNOSTIC OUTPUT${NC}"
        echo "    -d produced no pass diagnostics; absence of the -O2/-O3 line"
        echo "    proves nothing when nothing was printed at all."
        FAILED_TESTS+=("$name")
        ((FAIL++)) || true
        return
    fi

    if grep -qE "optimization passes at -O[123]" "$log"; then
        echo -e "${RED}ASSERTION FAIL${NC}"
        echo "    expected O0 / unoptimized, but got:"
        grep -iE "Applied LLVM" "$log" | head -3 | sed 's/^/      /'
        FAILED_TESTS+=("$name")
        ((FAIL++)) || true
    else
        echo -e "${GREEN}PASS${NC}"
        ((PASS++)) || true
    fi
}

OUT="$WORK/out.bin"
OBJ="$WORK/out.o"

# --- Default opt level: the sleeper-bug guard ------------------------------
check_level  "AOT -o default is O2"          "optimization passes at -O2"  "$PROBE" -o "$OUT"
check_level  "-c compile-only default is O2" "optimization passes at -O2"  -c "$PROBE" -o "$OBJ"
check_not_optimized "plain run (no -o) stays O0"                            "$PROBE"

# --- Explicit -O always wins ----------------------------------------------
check_level  "-o -O3 is O3"                  "optimization passes at -O3"  -O3 "$PROBE" -o "$OUT"
check_level  "-o -O1 is O1"                  "optimization passes at -O1"  -O1 "$PROBE" -o "$OUT"
check_not_optimized "-o -O0 opts out"                                      -O0 "$PROBE" -o "$OUT"

# --- Debug stays unoptimized ----------------------------------------------
check_not_optimized "-o -g stays unoptimized"                              -g "$PROBE" -o "$OUT"

echo ""
echo "========================================="
echo "  Opt-Level Test Results Summary"
echo "========================================="
TOTAL=$((PASS + FAIL))
echo "Total Tests:    $TOTAL"
echo -e "${GREEN}Passed:         $PASS${NC}"
echo -e "${RED}Failed:         $FAIL${NC}"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "Failed Tests:"
    for t in "${FAILED_TESTS[@]}"; do
        echo "  - $t"
    done
    echo ""
    echo -e "${RED}Some opt-level assertions failed.${NC}"
    exit 1
fi

echo -e "${GREEN}All opt-level assertions passed!${NC}"
exit 0
