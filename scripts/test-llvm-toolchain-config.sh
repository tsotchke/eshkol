#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURE_ROOT="${PROJECT_ROOT}/tests/toolchain/fake-llvm-root"
PROBE_ROOT="${PROJECT_ROOT}/build-toolchain-probe"

mkdir -p "${FIXTURE_ROOT}/include/llvm/IR" "${FIXTURE_ROOT}/lib"
touch "${FIXTURE_ROOT}/include/llvm/IR/LLVMContext.h"
mkdir -p "${PROBE_ROOT}"

rm -rf \
  "${PROJECT_ROOT}/build-toolchain-fake18" \
  "${PROJECT_ROOT}/build-toolchain-fake21"

cat > "${PROBE_ROOT}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.14)
project(EshkolLLVMToolchainProbe LANGUAGES C CXX)
set(LLVM_CONFIG_EXECUTABLE "" CACHE FILEPATH "llvm-config override")
include("${PROJECT_ROOT}/cmake/LLVMToolchain.cmake")
eshkol_find_lite_llvm()
message(STATUS "Probe LLVM version: \${ESHKOL_LLVM_VERSION}")
EOF

if cmake -S "${PROBE_ROOT}" -B "${PROJECT_ROOT}/build-toolchain-fake18" -G Ninja \
    -DLLVM_CONFIG_EXECUTABLE="${PROJECT_ROOT}/tests/toolchain/fake-llvm-config-18.sh" \
    >/tmp/eshkol-cmake-fake18.log 2>&1; then
  echo "fake LLVM 18 configure unexpectedly succeeded" >&2
  exit 1
fi

cmake -S "${PROBE_ROOT}" -B "${PROJECT_ROOT}/build-toolchain-fake21" -G Ninja \
  -DLLVM_CONFIG_EXECUTABLE="${PROJECT_ROOT}/tests/toolchain/fake-llvm-config-21.sh" \
  >/tmp/eshkol-cmake-fake21.log 2>&1
