# CI Lane Matrix

Continuous integration for Eshkol runs from `.github/workflows/ci.yml`. All
**14 lanes** live in two matrix jobs: `unix-matrix` (11 lanes) and
`windows-matrix` (3 lanes). Global env pins `LLVM_MAJOR=21` and the Windows SDK
at `WINDOWS_LLVM_SDK_VERSION=21.1.8`.

| # | Lane | Runner (OS/arch) | build dir | XLA | GPU | What it does |
|---|------|------------------|-----------|-----|-----|--------------|
| 1 | `linux-x64-lite` | ubuntu-22.04 (x64) | `build` | off | off | Full AOT+JIT suite (`run_all_tests.sh`), WASM-import check, uploads `eshkol-linux-x64` |
| 2 | `linux-arm64-lite` | ubuntu-22.04-arm | `build` | off | off | `run_all_tests.sh`, WASM check, uploads `eshkol-linux-arm64` |
| 3 | `linux-x64-xla` | ubuntu-22.04 | `build-xla` | on | off | `run_xla_tests.sh` |
| 4 | `linux-arm64-xla` | ubuntu-22.04-arm | `build-xla` | on | off | `run_xla_tests.sh` |
| 5 | `linux-x64-cuda` | ubuntu-22.04 | `build-cuda` | off | on | `run_gpu_tests.sh` (CUDA) |
| 6 | `linux-arm64-cuda` | ubuntu-22.04-arm | `build-cuda` | off | on | `run_gpu_tests.sh` (CUDA) |
| 7 | `macos-arm64-lite` | macos-14 (Apple Silicon) | `build` | off | off | `run_all_tests.sh`, WASM check, uploads `eshkol-macos-arm64` |
| 8 | `macos-x64-lite` | macos-15-intel (x86_64) | `build` | off | off | `run_all_tests.sh`, WASM check, uploads `eshkol-macos-x64` |
| 9 | `macos-arm64-xla` | macos-14 | `build-xla` | on | off | `run_xla_tests.sh` |
| 10 | `macos-x64-xla` | macos-15-intel | `build-xla` | on | off | `run_xla_tests.sh` |
| 11 | `linux-x64-asan-ubsan` | ubuntu-22.04 | `build-asan` | off | off | ASan+UBSan build, `run_v1_2_edge_cases_tests.sh` (TSan/MSan deferred) |
| 12 | `windows-arm64-lite` | windows-11-arm | `build` | off | off | VS2022 `-A ARM64 -T ClangCL`, `run_all_tests.ps1 -Mode windows-lite`, uploads `eshkol-windows-arm64` |
| 13 | `windows-arm64-xla` | windows-11-arm | `build-xla` | on | off | builds `xla_codegen_test`, `-Mode xla` |
| 14 | `windows-x64-cuda` | windows-2022 (x64) | `build-cuda` | off | on | installs CUDA 12.4, verifies real CUDA sources, then runs `-Mode gpu` |

## Lane groups

- **lite** (1,2,7,8,12) — the baseline AOT + JIT test suite per OS/arch; these
  are the lanes that upload release binaries.
- **xla** (3,4,9,10,13) — build with `-DESHKOL_XLA_ENABLED=ON` and run the XLA
  codegen/runtime tests.
- **cuda** (5,6,14) — build with the CUDA backend and run GPU tests.
- **asan-ubsan** (11) — sanitizer build over the v1.2 edge-case suite.

## Notes and gotchas

- Windows ARM64 covers lite/XLA. NVIDIA's native Windows toolkit is x86-64, so
  the CUDA compile lane uses hosted Windows x64 and no Windows ARM64 artifact is
  labeled CUDA.
- The `windows-matrix` runs `max-parallel: 2`; every lane downloads the matching
  official LLVM 21.1.8 SDK and builds with `ESHKOL_BUILD_TESTS=OFF`.
- CUDA lanes pin CUDA 12.4, require `ESHKOL_GPU_BACKEND=CUDA`, and reject any
  generated graph containing `gpu_memory_stub.cpp` or omitting the CUDA kernel
  sources. Linux artifacts target SM 72/75/80/86/89/90.
- Linux lanes symlink `ld.lld-21` → `ld.lld` because codegen passes
  `-fuse-ld=lld` for AArch64 Linux links.
- Concurrency **does not cancel on `master`** — the release gate wants the full
  matrix signal even when commits land quickly.
- Per the project rule, treat the non-required xla/asan/cuda lanes as
  first-class: a green "required" set can still hide a regression that only one
  of these lanes catches.

Other workflows: `pages.yml` (docs site) and `release.yml` (release packaging).
