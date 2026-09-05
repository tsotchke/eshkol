#!/usr/bin/env bash
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

# LLVM IS NOT A SUBMODULE OF STABLEHLO, and has not been for some time.
#
# This script used to run `git submodule update --init --recursive` here and
# then point cmake at $STABLEHLO_DIR/llvm-project/llvm. StableHLO ships no
# .gitmodules at all, so the submodule command silently did nothing and cmake
# died on the next line with:
#
#     CMake Error: The source directory ".../stablehlo/llvm-project/llvm"
#     does not exist.
#
# It failed in about ten seconds, which is why nobody noticed it was broken:
# the XLA lanes have read `awaiting-toolchain` and the gate has built with
# ESHKOL_XLA_ENABLED=OFF, so nothing ever ran this script to completion.
#
# StableHLO instead PINS an exact LLVM commit in build_tools/llvm_version.txt
# and expects you to supply the source yourself (its own build_tools/build_mlir.sh
# takes <path/to/llvm> <build_dir> as arguments). We fetch that exact commit
# here. We keep our OWN cmake invocation below rather than calling
# build_mlir.sh, because that script hardcodes LLVM_TARGETS_TO_BUILD=host and
# Eshkol needs WebAssembly and X86 as well for the wasm lane.
LLVM_PIN_FILE="$STABLEHLO_DIR/build_tools/llvm_version.txt"
if [ ! -f "$LLVM_PIN_FILE" ]; then
    echo "ERROR: $LLVM_PIN_FILE missing — StableHLO layout changed again." >&2
    echo "Find how this StableHLO revision pins LLVM before proceeding." >&2
    exit 1
fi
LLVM_COMMIT="$(tr -d '[:space:]' < "$LLVM_PIN_FILE")"
LLVM_SRC_DIR="$STABLEHLO_DIR/llvm-project"

echo "StableHLO pins LLVM at: $LLVM_COMMIT"
if [ ! -d "$LLVM_SRC_DIR/.git" ]; then
    echo "Fetching LLVM (shallow, single commit — the full history is ~3 GB)..."
    mkdir -p "$LLVM_SRC_DIR"
    git -C "$LLVM_SRC_DIR" init -q
    git -C "$LLVM_SRC_DIR" remote add origin https://github.com/llvm/llvm-project.git
fi
if ! git -C "$LLVM_SRC_DIR" cat-file -e "$LLVM_COMMIT^{commit}" 2>/dev/null; then
    git -C "$LLVM_SRC_DIR" fetch -q --depth 1 origin "$LLVM_COMMIT"
fi
git -C "$LLVM_SRC_DIR" checkout -q --detach "$LLVM_COMMIT"
echo "LLVM source at: $(git -C "$LLVM_SRC_DIR" rev-parse --short HEAD)"

if [ ! -f "$LLVM_SRC_DIR/llvm/CMakeLists.txt" ]; then
    echo "ERROR: $LLVM_SRC_DIR/llvm/CMakeLists.txt still missing after fetch." >&2
    exit 1
fi
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
