# Per-Platform Build Notes

Eshkol builds with CMake (3.14+), a C17 / C++20 toolchain (GCC 11+ or Clang 14+),
and **LLVM 21**. LLVM discovery is handled by `cmake/LLVMToolchain.cmake`
(`eshkol_find_lite_llvm`), which probes Homebrew `llvm@21` prefixes and Windows
SDK paths and validates the major version (`eshkol_validate_llvm_major` errors on
mismatch).

Baseline build:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Useful targets: `eshkol-run` (compiler/JIT driver), `eshkol-repl`,
`eshkol-vm-standalone`, and `stdlib` (precompiled standard library object).

> **Discrepancy (report only):** the top-level `README.md` Prerequisites section
> still lists "LLVM 17" in one place while the rest of the repo (CI, other README
> sections, `cmake/LLVMToolchain.cmake`) requires **LLVM 21**. The authoritative
> requirement is LLVM 21.

## macOS (Apple Silicon + x86_64)

```sh
brew install llvm@21 cmake ninja readline pcre2 sqlite
export LLVM_CONFIG="$(brew --prefix llvm@21)/bin/llvm-config"
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

- Apple Silicon is CI runner `macos-14`; Intel is `macos-15-intel`.
- BLAS uses the Apple **Accelerate** framework (AMX path, ~1100 GFLOPS),
  auto-detected by CMake.
- GPU uses **Metal** + MetalPerformanceShaders (`ESHKOL_GPU_METAL_ENABLED`),
  with the SF64 Metal shader embedded from `lib/backend/gpu/metal_softfloat.h`.

## Linux (x86_64 + arm64)

Install LLVM 21 from apt.llvm.org plus dev libraries:

```sh
# llvm-21 llvm-21-dev lld-21 libreadline-dev pkg-config libssl-dev
# libncurses-dev libpcre2-dev libsqlite3-dev libpng/jpeg/webp-dev
sudo ln -sf "$(command -v ld.lld-21)" /usr/local/bin/ld.lld   # AArch64 links use lld
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
```

- CUDA is auto-detected via `find_package(CUDAToolkit)`; when present, CMake
  selects a compatible `nvcc` host compiler and defines `ESHKOL_GPU_CUDA_ENABLED`.

## Windows x86_64 (MSYS2 / MinGW64)

Native MinGW64 build (originally contributed in PR #9). Not a hosted CI lane;
validated via `scripts/remote_windows_verify.sh --suite-only` against a cached
MSYS/UCRT build (`windows-lite` mode). The MinGW CMake path differs from the MSVC
path (`WIN32 AND NOT MINGW` uses `find_package(LLVM CONFIG REQUIRED)` + the MSVC
static runtime).

## Windows arm64 (VS2022 + ClangCL)

Built natively on `windows-11-arm`:

```
cmake -B build -G "Visual Studio 17 2022" -A ARM64 -T ClangCL
```

Downloads the LLVM 21.1.8 aarch64 Windows SDK. Three layered fixes made this
platform build and run (culminating in **PR #77**,
`fix/windows-arm64-sincos-portable`):

1. A **portable `sincos` shim** (MSVC has no `sincos`).
2. **AOT codegen**: use the *small* code model with `FunctionSections` +
   `DataSections` and `/OPT:REF,/OPT:ICF` dead-stripping instead of the Large
   code model — Large corrupts aarch64-COFF SEH `.pdata`/`.xdata` and breaks
   `_setjmpex`/longjmp unwinding used by `raise`/`guard`/`dynamic-wind`.
3. **JIT**: scope the windows-arm64 executable triple correctly.

These were verified on CI (no local arm64-Windows machine).

## NixOS on Jetson (CUDA)

See `nix/jetson/README.md`. Target: Jetson AGX Xavier, NixOS, L4T R35.6.4
(driver CUDA 11.4 max, GPU sm_72 / Volta):

```sh
NIXPKGS_ALLOW_UNFREE=1 nix-shell nix/jetson/shell.nix \
  --run 'bash nix/jetson/build.sh'   # -> build-cuda/eshkol-run
```

Non-trivial because NixOS 25.11 dropped CUDA 11 / gcc ≤ 11. Resolution: build
host C++ with clang-21, compile `.cu` kernels with nvcc 11.8 from pinned nixpkgs
24.05 (`cudaPackages_11`), and use gcc11 only as the nvcc host compiler. Three
runtime gotchas: force the gcc14 libdir to the front of `libstdc++` search; put
`/run/opengl-driver/lib` first on `LD_LIBRARY_PATH` (else `cudaErrorStubLibrary`
/ error 34 from the stub `libcuda.so.1`); use L4T-native cuBLAS 11.6. Verified
GEMM ~21 GFLOPS on GPU vs ~1.7 on CPU (~12×) via
`nix/jetson/jetson_gemm_bench.esk`.

## GPU / backend selection at runtime

The compiled binary contains **all** code paths (scalar / NEON / AVX / cBLAS /
GPU); backend selection is a runtime cost-model decision in
`lib/backend/blas_backend.cpp`. Key gates:

| Env var | Effect | Default |
|---------|--------|---------|
| `ESHKOL_GPU_ENABLED` (CMake) | Compile in Metal (macOS) / CUDA (Linux/Win) | ON |
| `ESHKOL_GPU_MATMUL_THRESHOLD` | Element count above which matmul goes to GPU (set `1` to force) | 100000 |
| `ESHKOL_GPU_PRECISION` | `exact` (sf64), `high` (df64), `fast` (f32) | `exact` |
| `ESHKOL_GPU_VERBOSE` | Log GPU dispatch decisions | off |
| `ESHKOL_BLAS_THRESHOLD` / `ESHKOL_XLA_THRESHOLD` | CPU BLAS / XLA dispatch thresholds | see [env vars](../reference/runtime/environment-variables.md) |

`ESHKOL_XLA_ENABLED` is a CMake option (default OFF) requiring a StableHLO/LLVM
bundle (`-DSTABLEHLO_ROOT`).
