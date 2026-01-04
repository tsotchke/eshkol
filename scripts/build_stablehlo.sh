#!/bin/bash
# Build StableHLO with LLVM for Eshkol XLA backend
#
# This script builds LLVM/MLIR and StableHLO with all necessary targets
# including WebAssembly and X86 for full Eshkol compatibility.
#
# Usage: ./scripts/build_stablehlo.sh
# Time: ~10-15 minutes on Apple Silicon

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESHKOL_DIR="$(dirname "$SCRIPT_DIR")"
STABLEHLO_DIR="$ESHKOL_DIR/deps/stablehlo"

echo "========================================="
echo "  StableHLO Build for Eshkol XLA"
echo "========================================="
echo ""
echo "Eshkol dir: $ESHKOL_DIR"
echo "StableHLO dir: $STABLEHLO_DIR"
echo ""

# Check if StableHLO is cloned
if [ ! -f "$STABLEHLO_DIR/.git/config" ]; then
    echo "Cloning StableHLO..."
    mkdir -p "$STABLEHLO_DIR"
    git clone https://github.com/openxla/stablehlo.git "$STABLEHLO_DIR"
fi

cd "$STABLEHLO_DIR"

# Initialize submodules (LLVM)
echo "Initializing submodules (this may take a few minutes)..."
git submodule update --init --recursive

LLVM_SRC_DIR="$STABLEHLO_DIR/llvm-project"
LLVM_BUILD_DIR="$STABLEHLO_DIR/llvm-build"
STABLEHLO_BUILD_DIR="$STABLEHLO_DIR/build"

# Build LLVM/MLIR with additional targets
echo ""
echo "Building LLVM/MLIR..."
echo "Build directory: $LLVM_BUILD_DIR"
echo ""

mkdir -p "$LLVM_BUILD_DIR"

# Check if ccache is available
CMAKE_LAUNCHER=""
if command -v ccache &>/dev/null; then
    echo "Using ccache for faster builds"
    CMAKE_LAUNCHER="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache"
fi

# Determine LLD availability (not on macOS)
LLVM_ENABLE_LLD="OFF"
if [[ "$(uname)" != "Darwin" ]]; then
    LLVM_ENABLE_LLD="ON"
fi

# Configure LLVM with host + WebAssembly + X86 targets
cmake -GNinja \
  "-H$LLVM_SRC_DIR/llvm" \
  "-B$LLVM_BUILD_DIR" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_LLD="$LLVM_ENABLE_LLD" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="host;WebAssembly;X86" \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_VERSION_SUFFIX="" \
  -DCMAKE_PLATFORM_NO_VERSIONED_SONAME:BOOL=ON \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_USE_SPLIT_DWARF=ON \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  $CMAKE_LAUNCHER

echo ""
echo "Building LLVM/MLIR (this takes ~8-10 minutes)..."
cmake --build "$LLVM_BUILD_DIR" --target all --parallel

# Build StableHLO
echo ""
echo "Building StableHLO..."
echo "Build directory: $STABLEHLO_BUILD_DIR"
echo ""

mkdir -p "$STABLEHLO_BUILD_DIR"

cmake -GNinja \
  "-H$STABLEHLO_DIR" \
  "-B$STABLEHLO_BUILD_DIR" \
  "-DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir" \
  "-DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  $CMAKE_LAUNCHER

cmake --build "$STABLEHLO_BUILD_DIR" --target all --parallel

echo ""
echo "========================================="
echo "  StableHLO Build Complete!"
echo "========================================="
echo ""
echo "LLVM/MLIR built at: $LLVM_BUILD_DIR"
echo "StableHLO built at: $STABLEHLO_BUILD_DIR"
echo ""
echo "To build Eshkol with XLA support:"
echo ""
echo "  cmake -B build-xla -G Ninja \\"
echo "    -DCMAKE_BUILD_TYPE=Release \\"
echo "    -DESHKOL_XLA_ENABLED=ON \\"
echo "    -DSTABLEHLO_ROOT=$STABLEHLO_DIR"
echo ""
echo "  cmake --build build-xla --parallel"
echo ""
echo "Then run XLA tests:"
echo "  ./scripts/run_xla_tests.sh"
echo ""

# Verify targets were built
echo "Verifying LLVM targets..."
if [ -f "$LLVM_BUILD_DIR/include/llvm/Config/Targets.def" ]; then
    echo "Available targets:"
    grep "LLVM_TARGET" "$LLVM_BUILD_DIR/include/llvm/Config/Targets.def" | grep -v "^#"
fi
