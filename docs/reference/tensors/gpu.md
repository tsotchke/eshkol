# GPU dispatch

Tensor operations use the GPU backend selected for the build: Metal, CUDA, or
WebGPU in a browser. The native and compiled-WASM paths share the
`eshkol_matmul_dispatch` and `eshkol_gpu_*` interfaces. The browser bytecode VM
uses the same WebGPU backend through `EshkolWebGPU.attachVm()` and JSPI. A
missing device or unsupported operation takes the CPU path; browser callers can
inspect the reason in the backend diagnostics.

## Operations and precision

`gpu-matmul`, `gpu-elementwise`, `gpu-softmax`, `gpu-transpose`, and
`gpu-reduce` resolve in native JIT and AOT builds. Their names do not bypass
backend admission: operation support, size thresholds, and device availability
still decide whether a GPU kernel runs.

```scheme
(define A (reshape (tensor 1.0 2.0 3.0 4.0) (list 2 2)))
(gpu-matmul A A)             ;; => #((7 10) (15 22))
(gpu-elementwise + A A)      ;; => #((2 4) (6 8))
(gpu-reduce + (tensor 1.0 2.0 3.0 4.0))  ;; => 10
```

`gpu-elementwise` accepts a bare `+`, `-`, `*`, or `/` operator (also the
`add` and `tensor-add` spellings). `gpu-reduce` accepts a bare `+`, `mean`,
`max`, or `min` operator and returns a scalar for a full reduction. A quoted
operator is not accepted by these forms.

In the browser, the default `exact` tier uses sf64 kernels that perform
binary64 arithmetic on integer words. Matmul and supported elementwise results
match the CPU path bit for bit; reductions may differ because of block
reassociation and are checked within `1e-9`. The `high` tier uses the same
sf64 kernels. The f32 `fast` tier requires an explicit `precision: "fast"` and
`gateTolerance >= 1e-6`.

WebGPU kernels cover matmul, same-shape elementwise add/subtract/multiply/divide,
and full sum/mean/max/min reductions for the browser VM. The compiled-WASM
backend also has sf64 negation, absolute value, relu, reciprocal, product and
additional reductions. Softmax, transpose, axis reductions, normalization,
transcendental elementwise operations, batched matmul and backward kernels use
the CPU path when the browser backend cannot serve them. Equal element counts
alone do not make two operand shapes compatible.

## Dispatch controls

On native backends, `ESHKOL_GPU_MATMUL_THRESHOLD` controls the BLAS dispatch
path by output element count (default `1000000000`; `0` forces the GPU decision
for smaller matmuls). `ESHKOL_GPU_THRESHOLD` is a separate backend threshold
for Metal/CUDA (default `100000`; values greater than zero apply). See
[environment variables](../runtime/environment-variables.md) for the other
runtime controls. Browser callers set `threshold` (default `100000`) and
`precision` through `initWebGPU()` instead.

The browser requires WebGPU, a device, and JSPI for GPU dispatch. When any is
missing, `initWebGPU()` or `attachVm()` reports a reason and evaluation runs on
the CPU. With a device, `dispatchCount`, `fallbackCount`, `lastPath`, and
`diagnostics` expose what the backend served or declined. VM evaluations that
may suspend must use `EshkolWebGPU.vmCall()`; it serializes calls per module.
See [GPU acceleration](../../breakdown/GPU_ACCELERATION.md#enabling-webgpu-in-a-page)
for both browser setup examples.

## CUDA build architectures

The portable `ESHKOL_CUDA_ARCHITECTURES` defaults are filtered against the
installed toolkit. Configuration first asks `nvcc --list-gpu-arch`; when that
is unavailable, it uses the toolkit version's supported range. Unsupported
defaults are reported and dropped, and configuration fails if none remain.
For example, CUDA 13 drops SM72 while CUDA 12 can keep it. An explicit
`CMAKE_CUDA_ARCHITECTURES` is used as supplied and is not filtered; nvcc reports
an unsupported explicit choice. The policy is implemented in
`cmake/EshkolCudaArchitectures.cmake`.

## See also

- [Tensor operations](operations.md)
- [Tensor creation and shapes](creation.md)
- [GPU acceleration and backend details](../../breakdown/GPU_ACCELERATION.md)
