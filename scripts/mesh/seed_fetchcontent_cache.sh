#!/usr/bin/env bash
# Seed the shared FetchContent source tree used by self-hosted Linux runners.
# Run this from a checked-out Eshkol revision, preferably origin/master.
set -euo pipefail

source_dir="${ESHKOL_SOURCE_DIR:-$PWD}"
cache_dir="${FETCHCONTENT_CACHE_DIR:-$HOME/lanes/_deps}"
build_dir="${FETCHCONTENT_SEED_BUILD_DIR:-$PWD/build-fetchcontent-cache}"
marker="$cache_dir/.eshkol-fetchcontent-cache-ready"

if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
  echo "error: source directory does not contain CMakeLists.txt: $source_dir" >&2
  exit 1
fi

mkdir -p "$cache_dir"
cmake_args=(
  -S "$source_dir"
  -B "$build_dir"
  -G Ninja
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DFETCHCONTENT_BASE_DIR="$cache_dir"
  -DESHKOL_REQUIRED_LLVM_MAJOR=21
  -DLLVM_CONFIG_EXECUTABLE=/usr/lib/llvm-21/bin/llvm-config
  -DESHKOL_XLA_ENABLED=OFF
  -DESHKOL_GPU_ENABLED=OFF
  -DESHKOL_REQUIRE_GPU_BACKEND=OFF
  -DESHKOL_BUILD_TESTS=ON
  -DESHKOL_BUILD_AGENT_FFI=ON
  -DESHKOL_QUANTUM_ENABLED=ON
)

if [[ -f "$marker" ]]; then
  cmake_args+=("-DFETCHCONTENT_FULLY_DISCONNECTED=ON")
  echo "Revalidating existing FetchContent cache: $cache_dir"
else
  echo "Seeding FetchContent cache: $cache_dir"
fi

cmake "${cmake_args[@]}"
touch "$marker"
echo "FetchContent cache ready: $cache_dir"
