#!/bin/bash
# run_v1_2_edge_cases.sh — proper runner for tests/v1_2_edge_cases/
#
# Each .esk test may declare a mode header on its first line:
#   ;; mode: jit       — run with `eshkol-run -r FILE`
#   ;; mode: compile   — compile with `-o BIN` then run BIN  (default)
#   ;; mode: shell     — for *.sh tests; just exec the script
#
# Module files (named *_module.esk) are NEVER run directly — they're
# imported by sibling test files. They're skipped with no failure.
#
# Exit 0 iff every non-skipped test passes.

set -u

# Per-run, per-repo-root isolation for temp files and build artifacts.
# Two suites (two worktrees, two agents, CI plus a local run) must never share
# a scratch path or a build artifact — see scripts/lib/test_isolation.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test_isolation.sh"
eshkol_test_isolation_init "v12-edge-runner"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/$BUILD_DIR/eshkol-run"
DIR="$ROOT/tests/v1_2_edge_cases"

# Several edge tests intentionally use project-root-relative `(load ...)`
# paths. Add the repo root to ESHKOL_PATH so both quiet and verbose runners
# resolve those modules the same way.
export ESHKOL_PATH="${ESHKOL_PATH:-.}"

if [ ! -x "$RUN" ]; then
    echo "ERROR: $RUN not built. Run 'make -C build -j8' first."
    exit 1
fi

PASS=0
FAIL=0
SKIP=0
FAILED_NAMES=()

# Detect mode from first 3 lines of file. Default = compile.
detect_mode() {
    local f="$1"
    local hdr
    hdr=$(head -3 "$f" 2>/dev/null | grep -iE "^;;[[:space:]]*mode:" | head -1)
    case "$hdr" in
        *jit*)     echo "jit" ;;
        *compile*) echo "compile" ;;
        *)         echo "compile" ;;
    esac
}

run_one() {
    local esk="$1"
    local name; name=$(basename "$esk" .esk)

    # Skip module files — imported, not standalone tests.
    if [[ "$name" == *"_module" ]]; then
        SKIP=$((SKIP+1))
        return
    fi

    local mode; mode=$(detect_mode "$esk")
    local rc=0
    # Capture the program output instead of discarding it: these tests print
    # their own PASS/FAIL lines and exit 0 either way, so a verdict taken from
    # the exit status alone certified printed failures as passes.
    local out="$ESHKOL_TEST_TMPDIR/v1_2_edge_${name}.out"

    case "$mode" in
        jit)
            "$RUN" -r "$esk" > "$out" 2>&1
            rc=$?
            ;;
        compile|*)
            local bin="$ESHKOL_TEST_TMPDIR/v1_2_edge_${name}_bin"
            rm -f "$bin"
            "$RUN" "$esk" -o "$bin" > "$out" 2>&1 || { rc=$?; }
            if [ "$rc" -eq 0 ] && [ -x "$bin" ]; then
                "$bin" > "$out" 2>&1
                rc=$?
            elif [ "$rc" -eq 0 ]; then
                # compile reported success but no binary — treat as fail
                rc=99
            fi
            ;;
    esac

    if [ "$rc" -eq 0 ] && eshkol_test_output_has_failure "$out"; then
        FAIL=$((FAIL+1))
        FAILED_NAMES+=("$name [mode=$mode rc=0, output reports failures]")
        eshkol_test_output_failures "$out" "" 6 | sed 's/^/    /'
        rm -f "$out"
        return
    fi
    rm -f "$out"

    if [ "$rc" -eq 0 ]; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        FAILED_NAMES+=("$name [mode=$mode rc=$rc]")
    fi
}

for esk in "$DIR"/*.esk; do
    [ -f "$esk" ] || continue
    run_one "$esk"
done

# Shell-based regression tests in the same directory.
for sh in "$DIR"/*.sh; do
    [ -f "$sh" ] || continue
    name=$(basename "$sh" .sh)
    if bash "$sh" >/dev/null 2>&1; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        FAILED_NAMES+=("$name [mode=shell]")
    fi
done

echo "============================================================"
echo "v1.2 edge-cases: $PASS pass, $FAIL fail, $SKIP module-files-skipped"
echo "============================================================"
if [ "$FAIL" -gt 0 ]; then
    echo "Failures:"
    for n in "${FAILED_NAMES[@]}"; do
        echo "  - $n"
    done
    exit 1
fi
exit 0
