#!/bin/bash
# Run Eshkol performance benchmarks
# Usage: ./scripts/run_benchmarks.sh [benchmark_name]

set -e

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

    # Compile
    "$BUILD_DIR/eshkol-run" -c "$bench_file" -o "/tmp/${bench_name}.o" 2>&1

    # Link
    clang "/tmp/${bench_name}.o" "$BUILD_DIR/stdlib.o" \
        -L/opt/homebrew/opt/llvm/lib \
        -lc -lm \
        -o "/tmp/${bench_name}" 2>&1

    # Run
    "/tmp/${bench_name}"

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
