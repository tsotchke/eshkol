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
;; Explicit GPU dispatch
(define result (gpu-matmul A B))

;; Element-wise GPU operations
(define scaled (gpu-elementwise (lambda (x) (* x 2.0)) A))
(define reduced (gpu-reduce + M))
(define soft (gpu-softmax logits))
(define transposed (gpu-transpose M))
```

---

## SIMD Vectorisation

For small tensors, the LLVM backend generates platform-specific SIMD
instructions:

- **x86_64**: SSE4.2, AVX2, AVX-512 (when available)
- **ARM64**: NEON

Loop vectorisation is automatic — the compiler detects vectorisable
patterns in tensor operations and emits wide instructions.

```scheme
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
