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
TMPDIR="${TMPDIR%/}"
if [ -z "$TMPDIR" ]; then
    TMPDIR="/"
fi

fail_benchmark_path() {
    local label="$1"
    local value="$2"

    echo "unsupported extreme benchmark $label: $value" >&2
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
        echo "Extreme benchmark temporary directory missing, symlinked, or not writable: $path" >&2
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

require_benchmark_decimal() {
    local label="$1"
    local value="$2"

    case "$value" in
        ""|.|*[!0-9.]*|*.*.*)
            fail_benchmark_path "$label" "$value"
            ;;
    esac
}

require_benchmark_work_dir() {
    local path="$1"

    case "$path" in
        "$TMPDIR"/eshkol_extreme_bench.*)
            ;;
        *)
            fail_benchmark_path "work directory" "$path"
            ;;
    esac

    if [ -L "$path" ] || [ ! -d "$path" ] || [ ! -w "$path" ]; then
        echo "Extreme benchmark work directory missing, symlinked, or not writable: $path" >&2
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
        echo "Extreme benchmark $label exists but is not a regular non-symlinked file: $path" >&2
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
    local mem_gb="$2"
    local path="$3"

    require_benchmark_size "$n"
    require_benchmark_decimal "memory size" "$mem_gb"
    require_benchmark_artifact_path "source file" "$path"
    if [ -e "$path" ]; then
        echo "Extreme benchmark source file already exists: $path" >&2
        exit 1
    fi

    cat > "$path" << HEREDOC_END
(require stdlib)
(define n $n)
(define a (random-tensor (list n n)))
(define b (random-tensor (list n n)))
(define elapsed (time-it (lambda () (matmul a b)) 1))
(define flops (* 2.0 n n n))
(define gflops (/ flops (* (/ elapsed 1e9) 1e9)))
(display "  ${n}x${n}  ${mem_gb} GB  ")
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
        echo "Extreme benchmark source file was not produced as a regular non-symlinked file: $path" >&2
        exit 1
    fi
}

require_benchmark_tmpdir "$TMPDIR"
BENCH_WORK_DIR="$(mktemp -d "$TMPDIR/eshkol_extreme_bench.XXXXXX")"
require_benchmark_work_dir "$BENCH_WORK_DIR"
trap cleanup_benchmark_work_dir EXIT

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
    require_benchmark_size "$N"
    require_benchmark_decimal "memory size" "$MEM_GB"
    BENCH_SRC="$BENCH_WORK_DIR/eshkol_bench_${N}.esk"
    BENCH_BIN="$BENCH_WORK_DIR/eshkol_bench_${N}"
    write_benchmark_source "$N" "$MEM_GB" "$BENCH_SRC"

    # Compile
    if ! "$ESHKOL_RUN" "$BENCH_SRC" -L"$BUILD_DIR" -o "$BENCH_BIN" 2>/dev/null; then
        echo "  FAIL ${N}x${N} (compilation failed)"
        remove_benchmark_artifact "$BENCH_SRC"
        continue
    fi

    # Run (no timeout — let it take as long as needed)
    require_benchmark_artifact_path "compiled binary" "$BENCH_BIN"
    set +e
    "$BENCH_BIN"
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -ne 0 ]; then
        if [ "$EXIT_CODE" -eq 137 ]; then
            echo "  OOM  ${N}x${N} (killed — out of memory at ${MEM_GB} GB)"
        else
            echo "  FAIL ${N}x${N} (exit code $EXIT_CODE)"
        fi
    fi

    # Cleanup
    remove_benchmark_artifact "$BENCH_SRC"
    remove_benchmark_artifact "$BENCH_BIN"
done

echo ""
echo "====================================================================="
echo "  EXTREME BENCHMARK COMPLETE"
echo "====================================================================="
