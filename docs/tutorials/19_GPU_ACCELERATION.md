# Tutorial 19: GPU Acceleration

Eshkol automatically dispatches tensor operations to the fastest available
backend: SIMD (SSE/AVX/NEON), cBLAS, Metal (macOS), or CUDA (Linux/Windows).

---

## The Cost-Model Dispatcher

When you call `matmul`, the compiler chooses the backend based on tensor
size:

| Tensor size | Backend | Why |
|---|---|---|
| Small (< 64 elements) | SIMD intrinsics | Overhead of GPU launch > computation |
| Medium (64-4096) | cBLAS (Accelerate/OpenBLAS) | Optimised CPU BLAS |
| Large (> 4096) | Metal or CUDA | GPU parallelism wins |

```scheme
;; Same API regardless of backend
(define A (rand 100 100))
(define B (rand 100 100))
(define C (matmul A B))     ;; auto-dispatched to best backend
```

---

## Metal (macOS)

On Apple Silicon, Eshkol uses Metal compute shaders with multiple
precision tiers:

| Tier | Precision | Speed | Use case |
|---|---|---|---|
| SF64 | Simulated float64 | Moderate | When doubles are required |
| DF64 | Double-float64 | Moderate | High precision |
| F32 | Native float32 | Fast | ML inference |
| FP24 | 24-bit float | Fastest | Approximate computation |
| FP53 | 53-bit mantissa | Moderate | Double-precision compatible |

The Ozaki-II CRT-based algorithm provides exact matrix multiplication
by splitting doubles into float32 components and accumulating with
compensated summation.

---

## CUDA (Linux / Windows)

On NVIDIA GPUs, Eshkol uses cuBLAS for matrix operations with
occupancy-aware kernel configuration:

```scheme
(define A #(#(1.0 2.0) #(3.0 4.0)))
(define B #(#(5.0 6.0) #(7.0 8.0)))
(define M A)
(define logits #(1.0 2.0 3.0))

;; Explicit GPU dispatch (the same calls fall back to the CPU when the
;; build or the machine has no GPU)
(define result (gpu-matmul A B))            ;; => #((19 22) (43 50))

;; Element-wise GPU operations
(define scaled (gpu-elementwise * A B))     ;; => #((5 12) (21 32))
(define reduced (gpu-reduce + M))           ;; => 10
(define soft (gpu-softmax logits))          ;; => #(0.0900... 0.2447... 0.6652...)
(define transposed (gpu-transpose M))       ;; => #((1 3) (2 4))
```

`gpu-elementwise` accepts the binary operators `+`, `-`, `*`, and `/`.
`gpu-reduce` accepts `+`, `mean`, `max`, and `min` and reduces across the full
tensor. These forms use GPU dispatch when profitable and fall back to the CPU
tensor implementation otherwise.

---

## SIMD Vectorisation

For small tensors, the LLVM backend generates platform-specific SIMD
instructions:

- **x86_64**: SSE4.2, AVX2, AVX-512 (when available)
- **ARM64**: NEON

Loop vectorisation is automatic — the compiler detects vectorisable
patterns in tensor operations and emits wide instructions.

```scheme
(define large-vector (rand 4096))
(define a #(1.0 2.0 3.0 4.0))
(define b #(5.0 6.0 7.0 8.0))
(define v #(1.0 2.0 3.0 4.0))

;; These are all SIMD-vectorised on supported hardware
(define sum (tensor-sum large-vector))
(define product (tensor-mul a b))
(define scaled (tensor-scale v 3.14))
```

---

## Build Configurations

The release ships three tiers per platform:

| Tier | Flag | Includes |
|---|---|---|
| **Lite** | default | SIMD + cBLAS |
| **XLA** | `-DESHKOL_XLA_ENABLED=ON` | + StableHLO/MLIR backend |
| **CUDA** | `-DESHKOL_GPU_ENABLED=ON` | + Metal/CUDA acceleration |

```bash
# Build with GPU support
cmake .. -DCMAKE_BUILD_TYPE=Release -DESHKOL_GPU_ENABLED=ON
make -j$(nproc)
```
