# Eshkol on Jetson AGX Xavier (NixOS, CUDA)

Build + run the Eshkol CUDA GPU backend on a Jetson AGX Xavier running NixOS
(L4T R35.6.4, driver CUDA 11.4, GPU compute capability sm_72 / Volta).

## TL;DR

```sh
export NIXPKGS_ALLOW_UNFREE=1
nix-shell nix/jetson/shell.nix --run 'bash nix/jetson/build.sh'
```

Produces `build-cuda/eshkol-run` with `ESHKOL_GPU_CUDA_ENABLED`.

## Why this is non-trivial

The Jetson driver supports **CUDA 11.4 maximum** (`cuDriverGetVersion` = 11040),
but the NixOS 25.11 system channel has **removed CUDA 11 and every gcc <= 11**,
and its CUDA 12.8 runtime is rejected by the driver. The leftover 11.4 nvcc in
the store cannot parse glibc-2.40 / gcc-14 C++ headers.

Resolution (see `shell.nix`): build the C++ host with **clang-21 (25.11)** and the
`.cu` kernels with **nvcc 11.8 from pinned nixpkgs 24.05** (`cudaPackages_11`),
wiring **gcc11** only as nvcc's host compiler. CUDA 11.8's cudart runs on the
11.4 driver (minor-version compatibility).

Three gotchas the build/run scripts handle:

1. **libstdc++ split.** The 24.05 CUDA imported targets inject gcc-11 lib dirs
   ahead of clang-21's gcc-14 libstdc++ on the link line, so `-lstdc++` resolves
   to the old one (missing `_M_replace_cold`, `GLIBCXX_3.4.30` that libLLVM-21
   needs). Fix: force `$GCC14_LIBDIR` to the front via `CMAKE_*_LINKER_FLAGS`.

2. **Stub libcuda.** nix `cuda_cudart` ships a *stub* `libcuda.so.1` in its main
   `lib/`. If it wins the runtime search, `cudaGetDeviceCount` returns error 34
   (`cudaErrorStubLibrary`). Fix: put `/run/opengl-driver/lib` (the real L4T
   driver) first on `LD_LIBRARY_PATH`.

3. **cuBLAS version.** nix cuBLAS 11.8 fails `cublasCreate` with status 3 on the
   11.4 driver. The **L4T-native cuBLAS 11.6** (`cuda-merged-11.4/lib`) initializes
   cleanly (ABI-stable within CUDA 11.x). Put it ahead of nix cuBLAS at runtime.

## Verifying a real GPU GEMM

`nix/jetson/jetson_gemm_bench.esk` runs an NxN matmul and reports GFLOPS. Force GPU
dispatch with `ESHKOL_GPU_THRESHOLD=1`; `ESHKOL_GPU_VERBOSE=1` prints
`[GPU] matmul ... -> CUDA cuBLAS` lines.

Measured on this Xavier (1024x1024x1024 f64, end-to-end through `(matmul ...)`):

| path | per-GEMM | GFLOPS |
|------|----------|--------|
| GPU (cuBLAS, sm_72) | ~103 ms | ~21 |
| CPU (SIMD/scalar)   | ~1258 ms | ~1.7 |

~12x speedup; GR3D engine at 99% during the run (tegrastats). Results are
numerically identical to the CPU reference. The full `tests/gpu/*` suite
(matmul / scale-correctness up to 4096x4096, transpose, reduce, elementwise)
passes on the GPU.

## Relationship to the CI GPU execution gate

This directory was the only place Eshkol's GPU backend was ever actually
*run* — CI's `*-cuda` lanes only compile it (GitHub-hosted runners have no
GPU). `tests/gpu/gpu_correctness_gate.sh` generalizes the differential
check this benchmark did by hand (GPU matmul vs. CPU reference, same
input, numeric diff) into a self-contained script that builds Eshkol with
and without GPU acceleration, runs a shared workload through both, and
skips cleanly on any host without a real GPU. It is wired into
`.github/workflows/gpu-execution-gate.yml` for a self-hosted GPU
runner (this Jetson is one candidate) or a scheduled job; see that
workflow file and `docs/breakdown/GPU_ACCELERATION.md` section 9 for the
compilation-vs-execution distinction. `jetson_gemm_bench.esk` stays as-is
for hand-run GFLOPS benchmarking — the gate script is the correctness
signal, this file is the performance one.
