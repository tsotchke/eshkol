#!/bin/bash
# Extreme Matmul Benchmark — MAX OUT SYSTEM LIMITS
# Each size runs in its own process so the arena allocator starts fresh.
# This avoids OOM from accumulated tensors.
#
# Memory per test: 3 × N² × 8 bytes (A, B, C matrices)
#   N=25000 → 15 GB     N=40000 → 38.4 GB    N=55000 → 72.6 GB
#   N=30000 → 21.6 GB   N=45000 → 48.6 GB    N=60000 → 86.4 GB
#   N=35000 → 29.4 GB   N=50000 → 60 GB      N=65000 → 101.4 GB

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
TMPDIR="${TMPDIR:-/tmp}"

echo "====================================================================="
echo "  EXTREME MATMUL BENCHMARK — SYSTEM LIMIT STRESS TEST"
echo "  Each size runs in a fresh process (arena reset between tests)"
echo "  System: $(sysctl -n hw.memsize | awk '{printf "%.0f GB", $1/1073741824}') unified memory"
echo "====================================================================="
echo ""

# Sizes to test — from large to absolute system limit
SIZES=(
    # Tier 8: Extreme (11.6-29.4 GB working set)
    22000 24000 25000 26000 28000 30000 32000 35000
    # Tier 9: Beyond (34.7-48.6 GB working set)
    38000 40000 42000 45000
    # Tier 10: System Limit (55.3-86.4 GB working set)
    48000 50000 55000 60000
)

for N in "${SIZES[@]}"; do
    # Calculate memory requirement
    MEM_BYTES=$((3 * N * N * 8))
    MEM_GB=$(echo "scale=1; $MEM_BYTES / 1073741824" | bc)

    # Check if we have enough system memory (leave 16 GB for OS)
    TOTAL_MEM=$(sysctl -n hw.memsize)
    AVAIL=$((TOTAL_MEM - 17179869184))  # Reserve 16 GB
    if [ "$MEM_BYTES" -gt "$AVAIL" ]; then
        echo "  SKIP ${N}x${N} (needs ${MEM_GB} GB, exceeds safe limit)"
        continue
    fi

    # Generate single-size benchmark using time-it from stdlib
    BENCH_SRC="$TMPDIR/eshkol_bench_${N}.esk"
    BENCH_BIN="$TMPDIR/eshkol_bench_${N}"

    cat > "$BENCH_SRC" << 'HEREDOC_END'
(require stdlib)
HEREDOC_END

    cat >> "$BENCH_SRC" << HEREDOC_END
(define n $N)
(define a (random-tensor (list n n)))
(define b (random-tensor (list n n)))
(define elapsed (time-it (lambda () (matmul a b)) 1))
(define flops (* 2.0 n n n))
(define gflops (/ flops (* (/ elapsed 1e9) 1e9)))
(display "  ${N}x${N}  ${MEM_GB} GB  ")
(cond
  ((>= elapsed 1e12)
   (display (number->string (round (/ elapsed 1e9) 0.01)))
   (display " s"))
  ((>= elapsed 1e9)
   (display (number->string (round (/ elapsed 1e6) 0.1)))
   (display " ms"))
  (else
   (display (number->string (round (/ elapsed 1e3) 0.1)))
   (display " us")))
(display "  @ ")
(if (>= gflops 1000)
    (begin (display (number->string (round (/ gflops 1000.0) 0.001)))
           (display " TFLOPS"))
    (begin (display (number->string (round gflops 0.1)))
           (display " GFLOPS")))
(newline)
HEREDOC_END

    # Compile
    if ! "$ESHKOL_RUN" "$BENCH_SRC" -L"$BUILD_DIR" -o "$BENCH_BIN" 2>/dev/null; then
        echo "  FAIL ${N}x${N} (compilation failed)"
        rm -f "$BENCH_SRC"
        continue
    fi

    # Run (no timeout — let it take as long as needed)
    "$BENCH_BIN"
    EXIT_CODE=$?

    if [ "$EXIT_CODE" -ne 0 ]; then
        if [ "$EXIT_CODE" -eq 137 ]; then
            echo "  OOM  ${N}x${N} (killed — out of memory at ${MEM_GB} GB)"
        else
            echo "  FAIL ${N}x${N} (exit code $EXIT_CODE)"
        fi
    fi

    # Cleanup
    rm -f "$BENCH_SRC" "$BENCH_BIN"
done

echo ""
echo "====================================================================="
echo "  EXTREME BENCHMARK COMPLETE"
echo "====================================================================="
