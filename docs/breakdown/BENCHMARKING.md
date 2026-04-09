# Benchmarking

**Status:** Production (v1.1.13)
**Applies to:** Eshkol compiler v1.1-accelerate and later

---

## Overview

The `benchmarks/` directory contains programs for measuring Eshkol's computational performance across matrix multiplication, neural network activation functions, convolution operations, and GPU vs CPU dispatch. Benchmarks are written in Eshkol (`.esk`) or as shell scripts (`.sh`) that generate and run Eshkol programs.

---

## Available Benchmarks

### 1. Matrix Multiplication (`matmul_bench.esk`)

**What it measures:** End-to-end matrix multiplication performance from 2x2 to 20000x20000 matrices, covering all dispatch tiers in the Eshkol matmul pipeline.

**Dispatch tiers:**

| Tier | Matrix Size | Method | Description |
|------|------------|--------|-------------|
| 1 | N <= 4 | Scalar | Direct loop multiplication |
| 2 | N = 8-64 | cBLAS | Small BLAS calls (Accelerate/OpenBLAS) |
| 3 | N = 96-512 | cBLAS | Medium BLAS, AMX hardware on Apple Silicon |
| 4 | N = 768-2048 | cBLAS | Large BLAS |
| 5 | N = 3072-6000 | cBLAS | Massive BLAS |
| 6 | N = 7000-10000 | cBLAS | Giant, 1.2-2.4 GB working set |
| 7 | N = 12000-20000 | cBLAS | Colossal, 3.4-9.6 GB per test |

**How to run:**

```bash
cd build
./eshkol-run ../benchmarks/matmul_bench.esk -o /tmp/matmul_bench
/tmp/matmul_bench
```

**Output:** For each size, reports dimensions, memory usage, wall-clock time, and throughput in GFLOPS or TFLOPS.

**Notes:** The arena allocator accumulates all tensors (no garbage collection). Tier 7 tests may consume 30-40 GB of accumulated arena memory from prior tiers. For sizes above N=20000, use `matmul_extreme.sh` instead.

---

### 2. Extreme Matrix Multiplication (`matmul_extreme.sh`)

**What it measures:** Matrix multiplication at system memory limits (N=22000 to N=60000). Each size runs in a separate process to avoid arena accumulation.

**Memory requirements per test:**

| Matrix Size | Working Set |
|------------|-------------|
| N = 25000 | 15 GB |
| N = 30000 | 21.6 GB |
| N = 40000 | 38.4 GB |
| N = 50000 | 60 GB |
| N = 60000 | 86.4 GB |

Formula: `3 * N^2 * 8 bytes` (three NxN matrices of float64).

**How to run:**

```bash
bash benchmarks/matmul_extreme.sh
```

The script automatically detects system memory and skips sizes that would exceed available RAM (with 16 GB reserved for the OS). Each test compiles a fresh Eshkol program, runs it in an isolated process, and cleans up temporary files.

**Output:** For each size, reports dimensions, memory usage, wall-clock time, and throughput.

---

### 3. GPU Matrix Multiplication (`gpu_matmul_bench.sh`)

**What it measures:** GPU-accelerated matrix multiplication using Metal softfloat64 (SF64) on Apple Silicon, comparing three precision tiers.

**Precision tiers:**

| Tier | Precision | Method | Bits |
|------|-----------|--------|------|
| 1a | Exact | sf64 v1 (original) | 53-bit IEEE f64 |
| 1b | Exact | sf64 v2 (deferred rounding) | 53-bit IEEE f64 |
| 2 | High | df64 hybrid (f32 FMA) | ~48-bit |
| 3 | Fast | Native f32 | 24-bit |

**How to run:**

```bash
bash benchmarks/gpu_matmul_bench.sh
```

Forces GPU dispatch for all sizes by setting `ESHKOL_GPU_MATMUL_THRESHOLD=1`. Sizes range from 4096x4096 to 40000x40000.

**Environment variables used:**
- `ESHKOL_GPU_MATMUL_THRESHOLD` -- Element count threshold for GPU dispatch
- `ESHKOL_GPU_PRECISION` -- Precision mode (`exact`, `high`, `fast`)
- `ESHKOL_SF64_KERNEL` -- Kernel version (`v1`, `v2`)

---

### 4. GPU vs CPU Comparison (`gpu_vs_cpu_bench.esk`)

**What it measures:** Direct comparison of GPU (Metal softfloat f64) vs CPU (cBLAS/Accelerate) matmul performance across the dispatch threshold boundary.

**Dispatch thresholds:**

| Path | Element Count | Example Size |
|------|--------------|--------------|
| Scalar | < 64 | 4x4, 8x8 |
| SIMD | 64 - 4095 | 16x16, 32x32 |
| cBLAS | 4096 - 99999 | 64x64 through 300x300 |
| GPU softfloat | >= 100000 | 316x316 and above |

**How to run:**

```bash
cd build
./eshkol-run ../benchmarks/gpu_vs_cpu_bench.esk -o /tmp/gpu_cpu_bench
/tmp/gpu_cpu_bench
```

**Output:** For each size, shows the dispatch path taken, wall-clock time, and throughput.

---

### 5. Activation Functions (`activation_bench.esk`)

**What it measures:** Elementwise neural network activation function throughput across three tensor sizes (small, medium, large).

**Functions benchmarked:**
- `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `leaky-relu`, `softmax`
- Scalar math ops: `exp`, `log`, `sin`, `cos`, `sqrt`

**Tensor sizes:**
- Small: n = 1,000
- Medium: n = 100,000
- Large: n = 1,000,000 (XLA threshold territory)

**How to run:**

```bash
cd build
./eshkol-run ../benchmarks/activation_bench.esk -o /tmp/act_bench
/tmp/act_bench
```

**Output:** For each function and size, reports average time in milliseconds and throughput in millions of elements per second.

---

### 6. Convolution and Pooling (`conv2d_bench.esk`)

**What it measures:** 2D convolution and pooling operation performance at various input sizes.

**Operations benchmarked:**
- `conv2d` with various input sizes (28x28 through 224x224) and kernel sizes (3x3, 7x7)
- `max-pool2d` with 2x2 pooling and stride 2
- `avg-pool2d` with 2x2 pooling and stride 2

**Input sizes:**
- 28x28 (MNIST-like)
- 64x64, 128x128
- 224x224 (ImageNet input size)

**How to run:**

```bash
cd build
./eshkol-run ../benchmarks/conv2d_bench.esk -o /tmp/conv_bench
/tmp/conv_bench
```

---

## Running All Benchmarks

There is no unified "run all benchmarks" script. Run each benchmark individually as shown above. For automated benchmarking, compile all `.esk` benchmarks first:

```bash
cd build

# Compile all Eshkol benchmarks
for f in ../benchmarks/*.esk; do
    name=$(basename "$f" .esk)
    ./eshkol-run "$f" -o "/tmp/bench_${name}" 2>/dev/null && echo "Built: $name"
done

# Run each
for f in /tmp/bench_*; do
    echo "=== $(basename $f) ==="
    "$f"
    echo ""
done
```

---

## Writing Custom Benchmarks

Use `time-it` from the standard library for precise timing:

```scheme
(require stdlib)

(define elapsed-ns
  (time-it (lambda () (matmul a b)) iterations))
```

`time-it` returns the average elapsed time per iteration in nanoseconds. It handles warmup internally.

For manual timing:

```scheme
(define start (current-time-ms))
;; ... operation ...
(define elapsed (- (current-time-ms) start))
```

---

## Performance Expectations

The following are approximate throughput ranges on Apple Silicon (M1/M2/M3):

| Operation | Typical Throughput |
|-----------|--------------------|
| Matmul (1024x1024, cBLAS) | 100-400 GFLOPS |
| Matmul (4096x4096, cBLAS) | 200-600 GFLOPS |
| ReLU (1M elements) | 500+ M elem/s |
| Sigmoid (1M elements) | 100-300 M elem/s |
| Conv2D (224x224, 3x3) | 10-50 ms |

Actual performance depends on the specific chip, memory bandwidth, thermal state, and whether the AMX coprocessor or GPU is engaged.

---

## See Also

- [GPU Acceleration](GPU_ACCELERATION.md) -- Metal SF64, CUDA, dispatch cost model
- [Machine Learning](MACHINE_LEARNING.md) -- Activation functions, convolution, training
- [Runtime Configuration](RUNTIME_CONFIGURATION.md) -- GPU thresholds, optimization levels
- [XLA Backend](XLA_BACKEND.md) -- Tensor runtime operations
