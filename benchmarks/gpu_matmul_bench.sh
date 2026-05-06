#!/bin/bash
# GPU Matmul Benchmark — All 3 tiers + v1/v2 comparison
# Forces GPU dispatch via ESHKOL_GPU_MATMUL_THRESHOLD=1

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
TMPDIR="${TMPDIR:-/tmp}"
TMPDIR="${TMPDIR%/}"
if [ -z "$TMPDIR" ]; then
    TMPDIR="/"
fi

fail_benchmark_path() {
    local label="$1"
    local value="$2"

    echo "unsupported GPU benchmark $label: $value" >&2
    exit 1
}

require_benchmark_tmpdir() {
    local path="$1"

    case "$path" in
        /*)
            ;;
        *)
            fail_benchmark_path "temporary directory" "$path"
            ;;
    esac

    if [ "$path" = "/" ]; then
        fail_benchmark_path "temporary directory" "$path"
    fi

    case "$path" in
        *//*|*/..|*/../*|*/.|*/./*|*[!A-Za-z0-9._/-]*)
            fail_benchmark_path "temporary directory" "$path"
            ;;
    esac

    if [ -L "$path" ] || [ ! -d "$path" ] || [ ! -w "$path" ]; then
        echo "GPU benchmark temporary directory missing, symlinked, or not writable: $path" >&2
        exit 1
    fi
}

require_benchmark_size() {
    local value="$1"

    case "$value" in
        ""|*[!0-9]*)
            fail_benchmark_path "matrix size" "$value"
            ;;
    esac

    if [ "$value" -le 0 ]; then
        fail_benchmark_path "matrix size" "$value"
    fi
}

require_benchmark_work_dir() {
    local path="$1"

    case "$path" in
        "$TMPDIR"/eshkol_gpu_bench.*)
            ;;
        *)
            fail_benchmark_path "work directory" "$path"
            ;;
    esac

    if [ -L "$path" ] || [ ! -d "$path" ] || [ ! -w "$path" ]; then
        echo "GPU benchmark work directory missing, symlinked, or not writable: $path" >&2
        exit 1
    fi
}

require_benchmark_artifact_path() {
    local label="$1"
    local path="$2"

    require_benchmark_work_dir "$BENCH_WORK_DIR"

    case "$path" in
        "$BENCH_WORK_DIR"/*)
            ;;
        *)
            fail_benchmark_path "$label" "$path"
            ;;
    esac

    case "$path" in
        */|*//*|*/..|*/../*|*/.|*/./*|*[!A-Za-z0-9._/-]*)
            fail_benchmark_path "$label" "$path"
            ;;
    esac

    if [ -L "$path" ] || { [ -e "$path" ] && [ ! -f "$path" ]; }; then
        echo "GPU benchmark $label exists but is not a regular non-symlinked file: $path" >&2
        exit 1
    fi
}

remove_benchmark_artifact() {
    local path="$1"

    require_benchmark_artifact_path "artifact" "$path"
    if [ -e "$path" ]; then
        rm -f -- "$path"
    fi
}

cleanup_benchmark_work_dir() {
    if [ -z "${BENCH_WORK_DIR:-}" ]; then
        return
    fi

    require_benchmark_work_dir "$BENCH_WORK_DIR"
    find "$BENCH_WORK_DIR" -mindepth 1 -maxdepth 1 -type f -delete
    rmdir "$BENCH_WORK_DIR" 2>/dev/null || true
}

write_benchmark_source() {
    local n="$1"
    local path="$2"

    require_benchmark_size "$n"
    require_benchmark_artifact_path "source file" "$path"
    if [ -e "$path" ]; then
        echo "GPU benchmark source file already exists: $path" >&2
        exit 1
    fi

    cat > "$path" << HEREDOC_END
(require stdlib)
(define n $n)
(define a (random-tensor (list n n)))
(define b (random-tensor (list n n)))
;; GPU shader warmup - first matmul compiles/caches the pipeline
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

    if [ -L "$path" ] || [ ! -f "$path" ] || [ ! -s "$path" ]; then
        echo "GPU benchmark source file was not produced as a regular non-symlinked file: $path" >&2
        exit 1
    fi
}

require_benchmark_tmpdir "$TMPDIR"
BENCH_WORK_DIR="$(mktemp -d "$TMPDIR/eshkol_gpu_bench.XXXXXX")"
require_benchmark_work_dir "$BENCH_WORK_DIR"
trap cleanup_benchmark_work_dir EXIT

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
        require_benchmark_size "$N"
        BENCH_SRC="$BENCH_WORK_DIR/eshkol_gpu_bench_${N}.esk"
        BENCH_BIN="$BENCH_WORK_DIR/eshkol_gpu_bench_${N}"
        write_benchmark_source "$N" "$BENCH_SRC"

        if ! "$ESHKOL_RUN" "$BENCH_SRC" -L"$BUILD_DIR" -o "$BENCH_BIN" 2>/dev/null; then
            echo "  FAIL ${N}x${N} (compilation failed)"
            remove_benchmark_artifact "$BENCH_SRC"
            continue
        fi

        require_benchmark_artifact_path "compiled binary" "$BENCH_BIN"
        set +e
        ESHKOL_GPU_MATMUL_THRESHOLD=1 ESHKOL_GPU_PRECISION="$PRECISION" ESHKOL_SF64_KERNEL="$KERNEL" "$BENCH_BIN"
        EXIT_CODE=$?
        set -e

        if [ "$EXIT_CODE" -ne 0 ]; then
            echo "  FAIL ${N}x${N} (exit code $EXIT_CODE)"
        fi

        remove_benchmark_artifact "$BENCH_SRC"
        remove_benchmark_artifact "$BENCH_BIN"
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
