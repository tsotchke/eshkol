#!/usr/bin/env bash
# nix/jetson/build.sh — configure + build Eshkol with CUDA on a Jetson AGX Xavier
# (NixOS, L4T R35.6.4). Run from the repo root:
#
#     export NIXPKGS_ALLOW_UNFREE=1
#     nix-shell nix/jetson/shell.nix --run 'bash nix/jetson/build.sh'
#
# Produces build-cuda/eshkol-run with the CUDA GPU backend enabled (sm_72).
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export CC=clang CXX=clang++

echo "nvcc:      $($NVCC --version | tail -2 | head -1)"
echo "host gcc:  $($GCC11/g++ --version | head -1)"
echo "clang:     $(clang++ --version | head -1)"

cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DESHKOL_REQUIRED_LLVM_MAJOR=21 \
  -DLLVM_CONFIG_EXECUTABLE="$(command -v llvm-config)" \
  -DESHKOL_XLA_ENABLED=OFF \
  -DESHKOL_GPU_ENABLED=ON \
  -DESHKOL_BUILD_TESTS=OFF \
  -DCUDAToolkit_ROOT="$CUDA_NVCC" \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_HOST_COMPILER="$GCC11/g++" \
  -DCMAKE_CUDA_ARCHITECTURES=72 \
  -DCMAKE_PREFIX_PATH="$CUDA_CUDART;$CUDA_CUBLAS;$CUDA_CCCL;$CUDA_NVCC" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$GCC14_LIBDIR -Wl,-rpath,$GCC14_LIBDIR" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$GCC14_LIBDIR -Wl,-rpath,$GCC14_LIBDIR"

ninja -C build-cuda
echo "Built: $REPO/build-cuda/eshkol-run"
echo
echo "=== RUNTIME NOTE ==="
echo "The Jetson Nix shell sets LD_LIBRARY_PATH so the REAL L4T driver libcuda and"
echo "L4T-native cuBLAS 11.6 win over nix's stub libcuda and nix's cuBLAS 11.8"
echo "(cublasCreate status=3 on the 11.4 driver). Keep execution inside this shell."
echo "Force GPU matmul dispatch with ESHKOL_GPU_THRESHOLD=1;"
echo "ESHKOL_GPU_VERBOSE=1 prints [GPU] dispatch lines."
