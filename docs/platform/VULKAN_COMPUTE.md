# Vulkan Compute Backend

Eshkol's GPU dispatch (`inc/eshkol/backend/gpu/gpu_memory.h`) runs on Apple
Metal, NVIDIA CUDA, browser WebGPU, and — on Linux — Vulkan compute
(`ESHKOL_GPU_VULKAN`). Vulkan gives the same tensor operations a device path on
any GPU with a Vulkan driver: AMD, Intel, NVIDIA without a CUDA toolkit, and
virtual or software devices. This page covers how the backend is selected,
what runs on the device, the exactness contract, and how it is verified.
Design record: ADR-0053.

## Where it sits

| Build | Host backend file | Vulkan role |
|-------|-------------------|-------------|
| Linux with CUDA | `gpu_memory_cuda.cpp` | fallback device backend when no CUDA device initialises, or when `ESHKOL_GPU_BACKEND=vulkan` |
| Linux without CUDA | `gpu_memory_stub.cpp` | the device backend |
| macOS / iOS | `gpu_memory.mm` (Metal) | not built |
| WebAssembly | `gpu_memory_webgpu.cpp` | not built |

`lib/backend/gpu/gpu_vulkan.c` opens the Vulkan loader with `dlopen`
(`libvulkan.so.1`, or `ESHKOL_VULKAN_LIBRARY`). Nothing links against Vulkan:
a build host needs neither the loader nor the Khronos headers, and binaries
produced by Eshkol run unchanged on machines without Vulkan (the dispatch then
reports the CPU). When `<vulkan/vulkan.h>` is present it is used; otherwise
`lib/backend/gpu/vulkan_abi.h` declares the Vulkan 1.0 subset the backend uses,
and `scripts/check_vulkan_abi.sh` proves that subset identical to the Khronos
header (every size, field offset and enumerant: 451 entries).

CMake: the backend is part of every Linux build with `ESHKOL_GPU_ENABLED=ON`;
`-DESHKOL_VULKAN_COMPUTE=OFF` leaves it out. In a build without CUDA the
configure line reports `GPU: Vulkan compute (loader opened at run time; CPU
when no device)` and the cache records `ESHKOL_GPU_BACKEND=VULKAN`.
`ESHKOL_REQUIRE_GPU_BACKEND=ON` still requires Metal or CUDA.

## Selection

`eshkol_gpu_init()` reads `ESHKOL_GPU_BACKEND`:

| Value | Behaviour |
|-------|-----------|
| unset / `auto` | CUDA when present; otherwise the best GPU-type Vulkan device (discrete, then integrated, then virtual) |
| `vulkan` | Vulkan only; CPU-type devices (Mesa lavapipe) are accepted too |
| `cuda` | CUDA only |
| `cpu` / `none` | no device backend |

`ESHKOL_VULKAN_DEVICE=<index>` opens one physical device by index, of any
type. The loader's own `VK_DRIVER_FILES` (or `VK_ICD_FILENAMES`) restricts
which drivers are visible. A device is usable when it has a compute queue and
`shaderFloat64`. Automatic selection skips CPU-type devices so a software
implementation never displaces the native CPU kernels.

## What runs on the device

| Operation | Kernel (`lib/backend/gpu/vulkan/`) |
|-----------|------------------------------------|
| `eshkol_gpu_elementwise_f64` add, sub, mul, div, neg, abs, relu, sqrt, reciprocal | `elementwise_f64.comp` |
| `eshkol_gpu_reduce_f64` sum, prod, min, max, mean | `reduce_f64.comp` |
| `eshkol_gpu_matmul_f64` | `matmul_f64.comp` |
| `eshkol_gpu_matmul_f32` | `matmul_f32.comp` |
| `eshkol_matmul_dispatch` (generated code, above the threshold) | `matmul_f64.comp` |

Every other entry point — f64 exp/log/sin/cos/tanh/sigmoid (GLSL.std.450
defines those only for 16- and 32-bit floats), axis reductions, softmax,
transpose, normalize, and the backward kernels — continues on the CPU with the
same `EshkolGPUBuffer` handles, because a Vulkan buffer's `host_ptr` always
holds the caller-visible data. Shapes beyond the kernels' 32-bit indexing or the
device's `maxStorageBufferRange` also take the CPU path.

The shaders are GLSL 4.50 sources compiled for the Vulkan 1.0 environment and
embedded as SPIR-V words in `lib/backend/gpu/vulkan_spirv.h`. That header is
generated only by `scripts/gen_vulkan_spirv.sh` (glslc from shaderc);
`scripts/gen_vulkan_spirv.sh --check` regenerates it and requires an identical
file.

## Exactness contract

- **Elementwise**: each shader expression is the CPU reference expression
  (same operand order and comparisons, e.g. `abs` is `x < 0 ? -x : x`), marked
  `precise` so no fused multiply-add is formed. On IEEE-754 round-to-nearest
  hardware the result is bit-identical to the CPU.
- **Matmul**: one invocation per output element sums `a[i,k]*b[k,j]` in index
  order without contraction; the result equals the sequential non-fused sum
  bit for bit. It can differ in the last bits from a blocked CPU BLAS, whose
  summation order is different.
- **Reductions**: a fixed tree. Invocation `g` folds elements
  `[256g, 256g+256)` in order from the identity; partials are reduced the same
  way until one value remains; `mean` divides by `n` on the host. The tree
  depends only on `n`, so results are reproducible run to run and across
  devices. `min`/`max` equal the CPU's sequential scan exactly; `sum` lies
  within `2(n-1)u·Σ|x|` of the sequential sum.
- **Admission**: when a device opens, every kernel is run on probe data —
  all pairs of 20 edge values (signed zeros, subnormals, `DBL_MIN`, `DBL_MAX`,
  infinities, NaN), 4096 random pairs that include raw bit patterns, a
  70000-element three-pass reduction, and 19×23×17 matmuls — and compared
  bit for bit with the host. A kernel that differs anywhere is not admitted on
  that device; its calls return `ESHKOL_VK_UNSUPPORTED` and the dispatch
  computes them on the CPU. The admitted set appears in `eshkol_vk_status()`
  and in the `GPU initialized: Vulkan compute (...)` log line.

## Memory

Every buffer owns a host-visible, host-coherent staging allocation (cached
when the driver offers it). When the device has separate device-local memory,
a device-local twin holds the kernel operands and each operation records the
copies in its command buffer: staging to device for inputs, device to staging
for outputs, with the barriers that make the host writes and reads visible.
Devices whose host-visible memory is already device-local (integrated GPUs,
software devices) use one allocation.

- `eshkol_gpu_wrap_host(p, ...)` keeps `host_ptr == p`. Each operation copies
  its inputs from `p` and writes its outputs back to `p` before returning.
- `eshkol_gpu_alloc(...)` returns `host_ptr` pointing at the mapped staging
  memory, valid for direct reads and writes.
- `eshkol_gpu_sync` moves data between the caller memory, the staging
  allocation and the device-local twin in the requested direction.
- Operations are synchronous: each submits one command buffer and waits on a
  fence, so `eshkol_gpu_wait` has nothing to wait for.

## Verification

CTest (registered when the backend is built; each exits 77, reported as
skipped, where its device or tool is absent):

| Test | What it checks |
|------|----------------|
| `gpu_vulkan_compute_crosscheck` | every kernel on every reachable f64 device against the CPU reference, bitwise: elementwise on sizes 1–300007 including edge values, reductions on sizes 1–1000003 against the mirrored tree and the sequential bound, f64/f32 matmul on five shapes, wrapped/allocated/in-place buffers, sync round trip, and that unimplemented or non-admitted operations defer to the CPU |
| `gpu_vulkan_dispatch` | the ordinary `eshkol_gpu_*` API with `ESHKOL_GPU_BACKEND=vulkan` through the production backend objects: selection queries, matmul/elementwise/reduce on the device, exp/tanh/transpose/softmax continuing on the CPU, allocated buffers with sync, `eshkol_matmul_dispatch`, shutdown |
| `gpu_vulkan_abi_layout` | `vulkan_abi.h` against the Khronos `vulkan_core.h` (needs the Khronos headers) |
| `gpu_vulkan_spirv_current` | `vulkan_spirv.h` against the GLSL sources (needs glslc) |

Measured on a Linux x86-64 node with an NVIDIA GPU (driver ICD, device-local
memory with host staging) and the Mesa lavapipe ICD (CPU-type device, shared
memory): all sixteen kernels (nine elementwise operations, five reductions,
f64 and f32 matmul) admitted on both devices;
`gpu_vulkan_compute_crosscheck` 334/334 checks per device (668 with both
drivers visible); `gpu_vulkan_dispatch` 46/46 checks per device through
both the CUDA host file and the non-CUDA host file. With both CUDA and Vulkan
present, automatic selection chose CUDA; with CUDA hidden
(`CUDA_VISIBLE_DEVICES=`) it chose the NVIDIA Vulkan device and the dispatch
test passed.

Manual commands:

```sh
# one driver at a time
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json    build/vulkan_compute_test
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json build/vulkan_compute_test
ESHKOL_GPU_BACKEND=vulkan build/vulkan_dispatch_test

# whole-language GPU gates on the Vulkan device
ESHKOL_GPU_BACKEND=vulkan tests/gpu/gpu_correctness_gate.sh

# ABI and SPIR-V provenance (on a host with Khronos headers and glslc)
scripts/check_vulkan_abi.sh
scripts/gen_vulkan_spirv.sh --check
```

## Planned for v1.8.1

- Windows: the loader path (`vulkan-1.dll`) is in `gpu_vulkan.c`, but the
  Windows build does not compile the backend yet.
- Apple: Vulkan through MoltenVK (Metal remains the Apple device backend).
- f64 transcendentals on the device through a software f64 library in SPIR-V,
  in the manner of the Metal SF64 kernels.
- Axis reductions, softmax, transpose, normalize and the backward kernels as
  Vulkan shaders.
- A tiled shared-memory matmul, a buffer pool and asynchronous submission
  (the current kernels favour exactness and simplicity over throughput).
- f16/bf16 storage paths and shapes beyond 32-bit element indexing.
