#!/bin/bash
# GPU Matmul Benchmark — All 3 tiers + v1/v2 comparison
# Forces GPU dispatch via ESHKOL_GPU_MATMUL_THRESHOLD=1

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
TMPDIR="${TMPDIR:-/tmp}"

SIZES=(4096 8000 12000 16000 20000 24000 28000 32000 36000 40000)

echo "====================================================================="
echo "  GPU MATMUL BENCHMARK — All Tiers"
echo "  M2 Ultra · 76 GPU cores · 128 ALUs/core"
echo "  All dispatched to GPU via ESHKOL_GPU_MATMUL_THRESHOLD=1"
echo "====================================================================="
echo ""

run_bench() {
    local PRECISION="$1"
    local KERNEL="$2"
    local LABEL="$3"

    echo "--- $LABEL ---"

    for N in "${SIZES[@]}"; do
        BENCH_SRC="$TMPDIR/eshkol_gpu_bench_${N}.esk"
        BENCH_BIN="$TMPDIR/eshkol_gpu_bench_${N}"

        cat > "$BENCH_SRC" << 'HEREDOC_END'
(require stdlib)
HEREDOC_END

        cat >> "$BENCH_SRC" << HEREDOC_END
(define n $N)
(define a (random-tensor (list n n)))
(define b (random-tensor (list n n)))
;; GPU shader warmup — first matmul compiles/caches the pipeline
(matmul a b)
(define elapsed (time-it (lambda () (matmul a b)) 1))
(define flops (* 2.0 n n n))
(define gflops (/ flops (* (/ elapsed 1e9) 1e9)))
(display "  ")
(if (< n 1000) (display " ") #t)
(display n) (display "x") (display n)
(display "  ")
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

        if ! "$ESHKOL_RUN" "$BENCH_SRC" -L"$BUILD_DIR" -o "$BENCH_BIN" 2>/dev/null; then
            echo "  FAIL ${N}x${N} (compilation failed)"
            rm -f "$BENCH_SRC"
            continue
        fi

        ESHKOL_GPU_MATMUL_THRESHOLD=1 ESHKOL_GPU_PRECISION="$PRECISION" ESHKOL_SF64_KERNEL="$KERNEL" "$BENCH_BIN"
        EXIT_CODE=$?

        if [ "$EXIT_CODE" -ne 0 ]; then
            echo "  FAIL ${N}x${N} (exit code $EXIT_CODE)"
        fi

        rm -f "$BENCH_SRC" "$BENCH_BIN"
    done
    echo ""
}

# Tier 1: Exact sf64 (53-bit IEEE f64 precision)
run_bench "exact" "v1" "TIER 1a: sf64 v1 (original) — 53-bit exact"
run_bench "exact" "v2" "TIER 1b: sf64 v2 (deferred rounding) — 53-bit exact"

# Tier 2: df64 hybrid (~48-bit via f32 FMA hardware)
run_bench "high" "" "TIER 2: df64 hybrid — ~48-bit via f32 FMA"

# Tier 3: Pure f32 (24-bit native Metal)
run_bench "fast" "" "TIER 3: f32 native — 24-bit fast"

echo "====================================================================="
echo "  GPU BENCHMARK COMPLETE"
echo "====================================================================="
