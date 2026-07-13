# nix/jetson/shell.nix
#
# Reproducible dev shell for building Eshkol with CUDA on a Jetson AGX Xavier
# (NixOS, L4T R35.6.4, driver CUDA 11.4, GPU sm_72).
#
# Why two nixpkgs generations:
#   * nixpkgs 25.11 (system channel) has LLVM 21 / clang 21 but REMOVED CUDA 11
#     and every gcc <= 11. Its CUDA 12.8 runtime is rejected by the R35 driver
#     ("CUDA driver version is insufficient"), and the leftover 11.4 nvcc cannot
#     parse glibc-2.40 / gcc-14 headers.
#   * nixpkgs 24.05 still ships cudaPackages_11 (CUDA 11.8 nvcc) + gcc11, a fully
#     supported nvcc/host-compiler pair. CUDA 11.8 cudart runs on the 11.4 driver
#     (minor-version compat).
#
# So: build the C++ host with clang-21 (25.11) and the .cu kernels with nvcc 11.8
# (24.05), wiring gcc11 ONLY as nvcc's host compiler. See build.sh for the exact
# CMake flags and the two runtime gotchas (stub libcuda, cuBLAS 11.8 vs driver).
{ }:
let
  pkgs = import <nixpkgs> { system = "aarch64-linux"; config.allowUnfree = true; };
  oldpkgs = import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/refs/tags/24.05.tar.gz";
  }) { system = "aarch64-linux"; config.allowUnfree = true; config.cudaSupport = false; };
  cp = oldpkgs.cudaPackages_11;
in pkgs.mkShell {
  buildInputs = [
    pkgs.llvmPackages_21.clang pkgs.llvmPackages_21.llvm pkgs.llvmPackages_21.lld
    pkgs.cmake pkgs.ninja pkgs.git pkgs.pkg-config
    pkgs.openssl pkgs.curl pkgs.sqlite pkgs.zlib pkgs.libffi pkgs.libxml2 pkgs.ncurses
    cp.cuda_nvcc cp.cuda_cudart cp.libcublas cp.cuda_cccl
    # gcc11 is deliberately NOT in buildInputs — it must only be nvcc's host
    # compiler (CMAKE_CUDA_HOST_COMPILER), never on the general C++ link path, or
    # its older libstdc++ shadows clang-21's libstdc++ and breaks the link
    # (_M_replace_cold / GLIBCXX_3.4.30 undefined refs).
  ];
  shellHook = ''
    # gcc-14 libstdc++ that clang-21 targets. The 24.05 CUDA imported targets
    # drag gcc-11's lib dirs onto the link line ahead of this; build.sh forces
    # this dir to the FRONT of the linker search path so -lstdc++ resolves to
    # the new libstdc++ (has _M_replace_cold + GLIBCXX_3.4.30 libLLVM-21 needs).
    export GCC14_LIBDIR=${pkgs.gcc.cc.lib}/lib
    export NVCC=${cp.cuda_nvcc}/bin/nvcc
    export GCC11=${oldpkgs.gcc11}/bin
    export CUDA_NVCC=${cp.cuda_nvcc}
    export CUDA_CUDART=${cp.cuda_cudart}
    export CUDA_CUBLAS=${cp.libcublas}
    export CUDA_CCCL=${cp.cuda_cccl}

    # Runtime closure for generated CUDA executables.  Jetson/L4T does not
    # expose its integrated GPU through nvidia-smi, and Nix's CUDA package can
    # otherwise put a stub libcuda plus a driver-incompatible cuBLAS ahead of
    # the board's native libraries.  Prefer the real L4T driver and the
    # installed CUDA-11.4 aggregate; retain the pinned 11.8 cudart/cuBLAS as a
    # portable fallback when the aggregate is absent.
    export JETSON_L4T_CUDA="$(find /nix/store -maxdepth 1 -type d -name '*-cuda-merged-11.4' -print -quit 2>/dev/null || true)"
    if [ -n "$JETSON_L4T_CUDA" ]; then
      _eshkol_cuda_runtime="$JETSON_L4T_CUDA/lib"
    else
      _eshkol_cuda_runtime="$CUDA_CUDART/lib:$CUDA_CUBLAS/lib"
    fi
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:$_eshkol_cuda_runtime:$GCC14_LIBDIR''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    unset _eshkol_cuda_runtime
  '';
}
