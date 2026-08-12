#!/usr/bin/env bash

# Eshkol Type System Test Suite
# Checks that the type checker detects the faults it claims to detect AND that
# --strict-types enforces them the way --help and
# docs/reference/runtime/eshkol-run.md say it does: "Type errors are fatal."
#
# Each .esk fixture declares its expectations in header comments:
#   ;; EXPECT-MODE: strict-types | unsafe | default
#   ;; EXPECT-COMPILE: fail | ok      (REQUIRED — see below)
#   ;; EXPECT-STDERR: <pattern>       (this pattern must appear in stderr)
#   ;; EXPECT-NO-STDERR: <pattern>    (this pattern must NOT appear in stderr)
#
# EXPECT-COMPILE is the assertion this suite used to be missing, and its absence
# is why "20/20" coexisted with a --strict-types flag that only changed the
# WORDING of the diagnostic. Every reject fixture printed its error and then
# exited 0, and the AOT compile wrote a finished binary for the program it had
# just rejected. The old harness assigned `compile_exit=$?` and then never read
# it, so a suite that verified only stderr text reported full marks while the
# documented contract was broken. Verdicts now come from BOTH the stderr
# patterns and the compile outcome:
#
#   EXPECT-COMPILE: fail -> the compile must exit NONZERO and write NO binary
#   EXPECT-COMPILE: ok   -> the compile must exit ZERO and write a binary
#
# A fixture with no EXPECT-COMPILE line FAILS. That is deliberate: an
# unasserted exit status is exactly the hole this suite fell into, so a new
# fixture cannot quietly opt out of the check.
#
# The suite finishes with a generated bare-type-name coverage cell derived
# straight from the parser's source of truth — see the last section.

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
eshkol_test_isolation_init "typesystem"

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

# Honour $BUILD_DIR (CI passes it via the matrix); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: build directory not found. Run cmake first.${NC}"
    exit 1
fi

if [ ! -f "$BUILD_DIR/eshkol-run" ]; then
    echo -e "${RED}Error: eshkol-run not found. Run make first.${NC}"
    exit 1
fi

# Scratch paths come from the shared isolation prelude, not from a second
# mktemp/trap pair of this suite's own.  Installing another `trap ... EXIT` here
# would REPLACE the trap eshkol_test_isolation_init set, leaking that run's
# private tmpdir; and a repo-root `rm -f a.out` can delete a concurrent suite's
# binary.  These are aliases so the assertions below read the way they were
# written while the paths stay per-run private.
WORK_DIR="$ESHKOL_TEST_TMPDIR"
TEST_BIN="$ESHKOL_TEST_BIN"
TEST_ERR="$ESHKOL_TEST_COMPILE_LOG"

echo "Testing type checker enforcement..."
echo ""

for test_file in tests/typesystem/*.esk; do
    [ -e "$test_file" ] || continue

    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    # Clean up stale temp files, including any binary a previous fixture wrote:
    # "the compile produced no binary" is only meaningful if nothing else could
    # have left one behind.  eshkol_test_reset_bin drops THIS run's binary and
    # its .tmp.o sidecar out of the private scratch dir, so it cannot delete a
    # concurrent suite's artifact the way the old repo-root `rm -f a.out` did.
    eshkol_test_reset_bin

    # Extract metadata from header comments
    mode=$(grep '^;; EXPECT-MODE:' "$test_file" | head -1 | sed 's/;; EXPECT-MODE: *//')
    expect_compile=$(grep '^;; EXPECT-COMPILE:' "$test_file" | head -1 | sed 's/;; EXPECT-COMPILE: *//')

    # Build compiler flags
    flags="-L./$BUILD_DIR"
    case "$mode" in
        strict-types) flags="$flags --strict-types" ;;
        unsafe)       flags="$flags --unsafe" ;;
        default)      ;; # no extra flags
        *)            ;; # no extra flags
    esac

    # Compile, capturing stderr separately.
    #
    # The `|| compile_exit=$?` form is load-bearing twice over. This is largely
    # a NEGATIVE suite, so a nonzero compile exit is the EXPECTED outcome for
    # most fixtures, and `set -e` at the top of this script would otherwise
    # abort the whole run at the first one — leaving no PASS/FAIL line and no
    # summary, which looks nothing like a failing suite. That went unnoticed
    # only because those fixtures used to compile successfully anyway. And
    # unlike a trailing `|| true`, this preserves the real status: the exit code
    # IS the assertion below, so it must survive.
    compile_exit=0
    ./$BUILD_DIR/eshkol-run "$test_file" $flags -o "$ESHKOL_TEST_BIN" \
        > /dev/null 2>"$ESHKOL_TEST_COMPILE_LOG" || compile_exit=$?

    test_passed=true
    declare -a reasons=()

    # ---- Compile-outcome assertion (exit status + binary) ----
    case "$expect_compile" in
        fail)
            if [ "$compile_exit" -eq 0 ]; then
                test_passed=false
                reasons+=("expected a NONZERO compile exit, got 0")
            fi
            if [ -e "$ESHKOL_TEST_BIN" ]; then
                test_passed=false
                reasons+=("compile was expected to fail but still wrote a binary")
            fi
            ;;
        ok)
            if [ "$compile_exit" -ne 0 ]; then
                test_passed=false
                reasons+=("expected a ZERO compile exit, got $compile_exit")
            fi
            if [ ! -e "$ESHKOL_TEST_BIN" ]; then
                test_passed=false
                reasons+=("compile succeeded but produced no binary")
            fi
            ;;
        *)
            test_passed=false
            reasons+=("no ';; EXPECT-COMPILE: fail|ok' declared in the fixture header")
            ;;
    esac

    # Check all EXPECT-STDERR patterns
    while IFS= read -r line; do
        pattern=$(echo "$line" | sed 's/;; EXPECT-STDERR: *//')
        if [ -n "$pattern" ]; then
            if ! grep -qF "$pattern" "$ESHKOL_TEST_COMPILE_LOG" 2>/dev/null; then
                test_passed=false
                reasons+=("missing expected stderr pattern: $pattern")
            fi
        fi
    done < <(grep '^;; EXPECT-STDERR:' "$test_file")

    # Check all EXPECT-NO-STDERR patterns
    while IFS= read -r line; do
        pattern=$(echo "$line" | sed 's/;; EXPECT-NO-STDERR: *//')
        if [ -n "$pattern" ]; then
            if grep -qF "$pattern" "$ESHKOL_TEST_COMPILE_LOG" 2>/dev/null; then
                test_passed=false
                reasons+=("forbidden stderr pattern present: $pattern")
            fi
        fi
    done < <(grep '^;; EXPECT-NO-STDERR:' "$test_file")

    # ---- `-r` must not publish a cached binary for a rejected program ----
    # A rejected program must not leave a runnable artifact behind on the JIT
    # path either: `-r` builds the cached run binary by spawning the compiler,
    # and that child compile now fails, so the cache must stay empty. Pointed at
    # a private cache dir so the check reads this run and not the developer's
    # warm ~/.cache/eshkol/jit.
    if [ "$expect_compile" = "fail" ]; then
        jit_cache="$WORK_DIR/jitcache/$test_name"
        rm -rf "$jit_cache"
        ESHKOL_JIT_CACHE_DIR="$jit_cache" ./$BUILD_DIR/eshkol-run "$test_file" $flags -r \
            > /dev/null 2>&1 || true
        if [ -d "$jit_cache" ] && [ -n "$(find "$jit_cache" -name 'run-*' -type f -print -quit 2>/dev/null)" ]; then
            test_passed=false
            reasons+=("-r published a cached run binary for a program that must not compile")
        fi
    fi

    if $test_passed; then
        echo -e "${GREEN}PASS${NC}"
        ((PASS++)) || true
    else
        echo -e "${RED}FAIL${NC}"
        FAILED_TESTS+=("$test_name")
        ((FAIL++)) || true
        for reason in "${reasons[@]}"; do
            echo "    reason: $reason"
        done
        # Show stderr for debugging
        if [ -s "$ESHKOL_TEST_COMPILE_LOG" ]; then
            echo "    stderr: $(head -3 "$ESHKOL_TEST_COMPILE_LOG")"
        else
            echo "    stderr: (empty)"
        fi
    fi
    unset reasons
done

# ===========================================================================
# Generated coverage: every bare type name the parser accepts
# ===========================================================================
#
# `(the <type> <expr>)` and the `x : <type>` annotations accept a fixed set of
# bare type-name spellings. That set lives in ONE place —
# eshkol::hott::builtinTypeSpellings() in lib/types/hott_types.cpp — and both
# the parser and the type environment are driven from it. This cell reads that
# table directly and compiles an ascription for every entry, so adding a
# spelling there cannot outrun its coverage: the previous split between the
# parser's private allow-list and the type system's name table is exactly how
# `(the number 3)` shipped as the parse error "Unknown function: the" while
# `(the string s)` worked and `number?` was an advertised narrowing predicate.
printf "Testing %-50s " "generated:every-bare-type-name"

SPELLING_SOURCE="lib/types/hott_types.cpp"
coverage_passed=true
declare -a coverage_reasons=()

if [ ! -f "$SPELLING_SOURCE" ]; then
    coverage_passed=false
    coverage_reasons+=("$SPELLING_SOURCE not found (run from the repository root)")
else
    type_names=$(awk '/builtinTypeSpellings\(\) \{/,/return table;/' "$SPELLING_SOURCE" \
        | grep -oE '^[[:space:]]*\{"[A-Za-z0-9_-]+"' \
        | sed 's/[^"]*"//; s/"//')
    name_count=$(printf '%s\n' "$type_names" | grep -c . || true)

    # A silently empty extraction would make this cell vacuously green, which is
    # the same failure mode as never reading compile_exit. Demand a plausible
    # table (the registry has dozens of entries).
    if [ "${name_count:-0}" -lt 20 ]; then
        coverage_passed=false
        coverage_reasons+=("extracted only ${name_count:-0} type names from $SPELLING_SOURCE — the registry moved or the parse broke")
    else
        coverage_src="$WORK_DIR/bare_type_name_coverage.esk"
        {
            echo ";; GENERATED by scripts/run_typesystem_tests.sh — do not commit."
            echo ";; One ascription per entry of eshkol::hott::builtinTypeSpellings()."
            echo "(define (coverage-identity x) x)"
            while IFS= read -r type_name; do
                [ -n "$type_name" ] || continue
                echo "(display (the $type_name (coverage-identity 1)))"
                echo "(newline)"
            done <<< "$type_names"
        } > "$coverage_src"

        rm -f "$TEST_BIN"
        coverage_exit=0
        ./$BUILD_DIR/eshkol-run "$coverage_src" -L./"$BUILD_DIR" --strict-types \
            -o "$TEST_BIN" > /dev/null 2>"$TEST_ERR" || coverage_exit=$?

        if [ "$coverage_exit" -ne 0 ]; then
            coverage_passed=false
            coverage_reasons+=("compiling $name_count bare type names exited $coverage_exit")
        fi
        if [ ! -e "$TEST_BIN" ]; then
            coverage_passed=false
            coverage_reasons+=("compile produced no binary")
        fi
        # An unrecognised type name does not fail the compile — the form decays
        # into a call to an undefined procedure named `the`, which is reported
        # and then tolerated. Assert on the diagnostics too.
        for forbidden in "Unknown function: the" "Undefined variable" "[ERROR]" "[WARN]"; do
            if grep -qF "$forbidden" "$TEST_ERR" 2>/dev/null; then
                coverage_passed=false
                coverage_reasons+=("forbidden diagnostic while ascribing bare type names: $forbidden")
            fi
        done
    fi
fi

if $coverage_passed; then
    echo -e "${GREEN}PASS${NC} (${name_count:-0} type names)"
    ((PASS++)) || true
else
    echo -e "${RED}FAIL${NC}"
    FAILED_TESTS+=("generated:every-bare-type-name")
    ((FAIL++)) || true
    for reason in "${coverage_reasons[@]}"; do
        echo "    reason: $reason"
    done
    if [ -s "$TEST_ERR" ]; then
        echo "    stderr: $(head -3 "$TEST_ERR")"
    fi
fi

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
rm -f "$ESHKOL_TEST_COMPILE_LOG" "$ESHKOL_TEST_BIN"

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    exit 1
fi
