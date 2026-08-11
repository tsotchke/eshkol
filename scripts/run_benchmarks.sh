#!/usr/bin/env bash
# Run Eshkol performance benchmarks
# Usage: ./scripts/run_benchmarks.sh [benchmark_name]

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
eshkol_test_isolation_init "benchmarks"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
BENCH_DIR="${PROJECT_DIR}/benchmarks"

# Ensure build is up to date
echo "Building Eshkol..."
cmake --build "$BUILD_DIR" --target eshkol-run 2>&1 | tail -5

echo ""
echo "================================================"
echo "  Eshkol Performance Benchmarks"
echo "================================================"
echo ""

run_benchmark() {
    local bench_file=$1
    local bench_name=$(basename "$bench_file" .esk)

    echo "Running $bench_name..."
    echo "----------------------------------------"

    # Compile and link (using same mode as test scripts)
    "$BUILD_DIR/eshkol-run" "$bench_file" -L"$BUILD_DIR" \
        -o "$ESHKOL_TEST_TMPDIR/${bench_name}" 2>&1

    # Run
    "$ESHKOL_TEST_TMPDIR/${bench_name}"

    echo ""
}

if [ -n "$1" ]; then
    # Run specific benchmark
    bench_file="$BENCH_DIR/${1}.esk"
    if [ -f "$bench_file" ]; then
        run_benchmark "$bench_file"
    else
        echo "Benchmark not found: $1"
        echo "Available benchmarks:"
        ls -1 "$BENCH_DIR"/*.esk 2>/dev/null | xargs -I{} basename {} .esk
        exit 1
    fi
else
    # Run all benchmarks
    for bench_file in "$BENCH_DIR"/*.esk; do
        if [ -f "$bench_file" ]; then
            run_benchmark "$bench_file"
        fi
    done
fi

echo "================================================"
echo "  All benchmarks complete"
echo "================================================"
