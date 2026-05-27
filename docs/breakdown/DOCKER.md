# Docker Images

**Status:** Production (v1.2.1-scale)
**Applies to:** Eshkol compiler v1.2.0-scale and later

---

## Overview

Eshkol provides five Docker images for building the compiler in different configurations. These images are primarily used by the release CI pipeline to produce platform-specific artifacts, but they are also useful for local development and testing.

All images are based on Debian or Ubuntu and include LLVM 21, CMake, Ninja, and the necessary system libraries, including libpng/libjpeg/libwebp for native Linux image I/O.

---

## Available Images

### 1. CUDA (`docker/cuda/Dockerfile`)

**Base image:** `nvidia/cuda:12.4.1-devel-ubuntu22.04`

Builds Eshkol with native float64 GPU acceleration via CUDA. This image includes the full CUDA 12.4 development toolkit.

**When to use:** When deploying on Linux systems with NVIDIA GPUs and you want native double-precision GPU matmul without the software float64 overhead of Metal SF64.

**Build:**
```bash
docker build -f docker/cuda/Dockerfile -t eshkol-cuda .
```

**GPU requirements:** NVIDIA GPU with compute capability 7.0+ and CUDA 12.4 compatible drivers.

**Additional packages:** `libopenblas-dev` (CPU BLAS fallback), `libpng-dev`, `libjpeg-dev`, `libwebp-dev`, `curl`, `dpkg-dev`, `file`.

---

### 2. Debian Debug (`docker/debian/debug/Dockerfile`)

**Base image:** `debian:12-slim`

Debug build of Eshkol on Debian 12 (Bookworm). Includes `gdb` for debugging.

**When to use:** For diagnosing compiler bugs, reproducing issues in a clean Debian environment, or running the compiler under a debugger.

**Build:**
```bash
docker build -f docker/debian/debug/Dockerfile -t eshkol-debug .
```

**Additional packages:** `gdb`, `libopenblas-dev`, `libpng-dev`, `libjpeg-dev`, `libwebp-dev`, `pkg-config`.

**Notes:** LLVM 21 is installed from the `bookworm` LLVM repository (not `jammy`).

---

### 3. Debian Release (`docker/debian/release/Dockerfile`)

**Base image:** `ubuntu:22.04`

Release build matching the GitHub Actions CI environment. Despite the directory name `debian/release`, this uses Ubuntu 22.04 for consistency with CI.

**When to use:** For producing release-quality Linux x86-64 binaries, or when you want to verify that a build matches CI exactly.

**Build:**
```bash
docker build -f docker/debian/release/Dockerfile -t eshkol-release .
```

**Additional packages:** `libopenblas-dev`, `libpng-dev`, `libjpeg-dev`, `libwebp-dev`, `curl`, `pkg-config`, `dpkg-dev`, `file`.

---

### 4. Ubuntu Release (`docker/ubuntu/release/Dockerfile`)

**Base image:** `ubuntu:22.04`

Release build that matches the GitHub Actions `ubuntu-22.04` runner. Functionally identical to the Debian release image.

**When to use:** For producing release binaries or when you want a minimal Ubuntu-based build environment.

**Build:**
```bash
docker build -f docker/ubuntu/release/Dockerfile -t eshkol-ubuntu .
```

**Additional packages:** Same as Debian release.

---

### 5. XLA (`docker/xla/Dockerfile`)

**Base image:** `ubuntu:22.04`

Builds Eshkol with StableHLO/MLIR support for XLA tensor operations. Includes MLIR 21 tools and development libraries.

**When to use:** When deploying tensor-heavy workloads that benefit from the XLA compilation pipeline, or when you need StableHLO interoperability.

**Build:**
```bash
docker build -f docker/xla/Dockerfile -t eshkol-xla .
```

**Additional packages:** `mlir-21-tools`, `libmlir-21-dev`, `libopenblas-dev`, `libpng-dev`, `libjpeg-dev`, `libwebp-dev`, `curl`, `pkg-config`.

---

### 6. Paper Reproducibility (`docker/Dockerfile.paper`)

**Base image:** `ubuntu:22.04`

Pinned reproducibility environment for the artifact accompanying *The Self-Differentiating Neural Computer: Computable Transformers via Analytical Weight Construction* (Atkins, 2026). The image fixes Clang/LLVM at 21, installs `numpy==1.26.4` for the trace comparator and table generator, and defaults its entrypoint to `scripts/paper/run_paper_suite.sh`, which drives the bit-identical agreement gate between the reference C interpreter and the matrix forward pass.

**When to use:** When reproducing the SDNC paper's numerical results, regenerating its tables, or running the agreement oracle on a fresh host.

**Build and run:**
```bash
docker build -f docker/Dockerfile.paper -t eshkol-paper .

# Default: run the full paper suite against the mounted source tree.
docker run --rm -v "$(pwd):/work" -w /work eshkol-paper

# Or drop into the same pinned environment for ad-hoc work.
docker run --rm -it -v "$(pwd):/work" -w /work eshkol-paper bash
```

**Additional packages:** `clang-21`, `llvm-21`, `llvm-21-dev`, `cmake`, `ninja-build`, `pkg-config`, `libpng-dev`, `libjpeg-dev`, `libwebp-dev`, `python3`, `python3-pip`, `numpy==1.26.4`.

---

## Common Build Pattern

All Dockerfiles follow the same structure:

1. Add the LLVM 21 apt repository from `apt.llvm.org`
2. Install build dependencies (CMake, Ninja, LLVM 21, g++, readline)
3. Copy the source tree into the container
4. Configure with CMake (Ninja generator, Release mode)
5. Build in parallel

To build and extract artifacts from any image:

```bash
# Build the image
docker build -f docker/<variant>/Dockerfile -t eshkol-<variant> .

# Create a container and extract binaries
docker create --name extract eshkol-<variant>
docker cp extract:/app/build/eshkol-run ./eshkol-run
docker cp extract:/app/build/eshkol-repl ./eshkol-repl
docker cp extract:/app/build/stdlib.o ./stdlib.o
docker rm extract
```

---

## Running the Compiler in Docker

To compile and run an Eshkol program inside a Docker container:

```bash
# Build the image
docker build -f docker/ubuntu/release/Dockerfile -t eshkol .

# Compile a program
docker run --rm -v $(pwd):/work -w /work eshkol \
    /app/build/eshkol-run hello.esk -o hello

# Run it
docker run --rm -v $(pwd):/work -w /work eshkol /work/hello
```

For GPU-accelerated Docker runs with CUDA:

```bash
docker run --gpus all --rm -v $(pwd):/work -w /work eshkol-cuda \
    /app/build/eshkol-run program.esk -o program
```

---

## Image Comparison

| Image | Base | Size (approx.) | GPU Support | Debug Tools | Use Case |
|-------|------|----------------|-------------|-------------|----------|
| CUDA | nvidia/cuda:12.4.1-devel-ubuntu22.04 | ~6 GB | NVIDIA CUDA | No | GPU acceleration |
| Debian Debug | debian:12-slim | ~2 GB | No | gdb | Debugging |
| Debian Release | ubuntu:22.04 | ~2 GB | No | No | CI-matched builds |
| Ubuntu Release | ubuntu:22.04 | ~2 GB | No | No | CI-matched builds |
| XLA | ubuntu:22.04 | ~3 GB | XLA/MLIR | No | Tensor operations |
| Paper reproducibility | ubuntu:22.04 | ~2 GB | No | No | SDNC paper artefact suite |

---

## See Also

- [CI/CD Pipelines](CI_CD.md) -- How Docker images are used in release builds
- [GPU Acceleration](GPU_ACCELERATION.md) -- Metal SF64, CUDA runtime
- [XLA Backend](XLA_BACKEND.md) -- StableHLO/MLIR tensor operations
- [Getting Started](GETTING_STARTED.md) -- Building from source without Docker
