#!/bin/bash
#
# macOS Build Script - Build Eshkol natively on macOS
#
# Supports:
#   - Native builds (arm64 on Silicon, x86_64 on Intel)
#   - Universal binaries (arm64 + x86_64) via cross-compilation
#
# Usage: ./scripts/build-macos.sh [options]
#   --version VER     Version string (default: 1.0.0)
#   --output DIR      Output directory for artifacts
#   --run-tests       Run tests after build
#   --universal       Build universal binary (arm64 + x86_64)
#   --arch ARCH       Specific architecture: arm64, x86_64, or native
#   --clean           Clean build directory first
#
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
#

set -e

# Defaults
VERSION="1.0.0"
OUTPUT_DIR="./dist"
RUN_TESTS=false
UNIVERSAL=false
TARGET_ARCH="native"
CLEAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --run-tests)
            RUN_TESTS=true
            shift
            ;;
        --universal)
            UNIVERSAL=true
            shift
            ;;
        --arch)
            TARGET_ARCH="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Detect host architecture
HOST_ARCH=$(uname -m)
echo "Host architecture: $HOST_ARCH"

# Set up LLVM path (Homebrew)
if [ "$HOST_ARCH" = "arm64" ]; then
    LLVM_PATH="/opt/homebrew/opt/llvm@17"
else
    LLVM_PATH="/usr/local/opt/llvm@17"
fi

if [ ! -d "$LLVM_PATH" ]; then
    echo "Error: LLVM 17 not found at $LLVM_PATH"
    echo "Install with: brew install llvm@17"
    exit 1
fi

export PATH="$LLVM_PATH/bin:$PATH"
export LDFLAGS="-L$LLVM_PATH/lib"
export CPPFLAGS="-I$LLVM_PATH/include"

mkdir -p "$OUTPUT_DIR"

# Function to build for a specific architecture
build_arch() {
    local arch=$1
    local build_dir="build-$arch"

    echo "Building for architecture: $arch"

    if [ "$CLEAN" = true ] && [ -d "$build_dir" ]; then
        rm -rf "$build_dir"
    fi

    # Set architecture-specific flags
    local cmake_arch_flags=""
    if [ "$arch" = "arm64" ]; then
        cmake_arch_flags="-DCMAKE_OSX_ARCHITECTURES=arm64"
    elif [ "$arch" = "x86_64" ]; then
        cmake_arch_flags="-DCMAKE_OSX_ARCHITECTURES=x86_64"
        # For cross-compilation on arm64, we need to ensure LLVM can cross-compile
        if [ "$HOST_ARCH" = "arm64" ]; then
            echo "Cross-compiling x86_64 on arm64 host"
        fi
    fi

    # Configure
    cmake -B "$build_dir" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        $cmake_arch_flags

    # Build
    cmake --build "$build_dir" --parallel

    echo "Build complete for $arch"
}

# Function to run tests
run_tests() {
    local build_dir=$1
    echo "Running tests..."

    # Only run tests if architecture matches or Rosetta is available
    if [ "$HOST_ARCH" = "arm64" ] && [ "$TARGET_ARCH" = "x86_64" ]; then
        echo "Running x86_64 tests via Rosetta..."
        arch -x86_64 ./scripts/run_types_tests.sh || true
    else
        ./scripts/run_types_tests.sh
        ./scripts/run_list_tests.sh
        ./scripts/run_autodiff_tests.sh
    fi
}

# Function to create package
create_package() {
    local arch=$1
    local build_dir=$2
    local output_name=$3

    echo "Creating package: $output_name"

    local pkg_dir="$OUTPUT_DIR/${output_name}"
    mkdir -p "$pkg_dir"

    # Copy binaries
    cp "$build_dir/eshkol-run" "$pkg_dir/"
    cp "$build_dir/eshkol-repl" "$pkg_dir/"
    cp "$build_dir/stdlib.o" "$pkg_dir/"

    # Create tarball
    local tarball="$OUTPUT_DIR/eshkol-${VERSION}-${output_name}.tar.gz"
    (
        mkdir -p "$pkg_dir/pkg/bin" "$pkg_dir/pkg/lib" "$pkg_dir/pkg/share/eshkol"
        cp "$build_dir/eshkol-run" "$pkg_dir/pkg/bin/"
        cp "$build_dir/eshkol-repl" "$pkg_dir/pkg/bin/"
        cp "$build_dir/stdlib.o" "$pkg_dir/pkg/lib/"
        cp lib/stdlib.esk "$pkg_dir/pkg/share/eshkol/"
        [ -d lib/core ] && cp -r lib/core "$pkg_dir/pkg/share/eshkol/"
        cp README.md LICENSE "$pkg_dir/pkg/" 2>/dev/null || true
        tar -czvf "$tarball" -C "$pkg_dir/pkg" .
        rm -rf "$pkg_dir/pkg"
    )

    echo "Created: $tarball"
}

# Function to create universal binary
create_universal() {
    echo "Creating universal binary..."

    local universal_dir="$OUTPUT_DIR/macos-universal"
    mkdir -p "$universal_dir"

    # Use lipo to combine arm64 and x86_64 binaries
    for binary in eshkol-run eshkol-repl; do
        if [ -f "build-arm64/$binary" ] && [ -f "build-x86_64/$binary" ]; then
            lipo -create \
                "build-arm64/$binary" \
                "build-x86_64/$binary" \
                -output "$universal_dir/$binary"
            echo "Created universal binary: $binary"
            file "$universal_dir/$binary"
        else
            echo "Warning: Cannot create universal $binary - missing architecture"
        fi
    done

    # stdlib.o needs to be built separately for each arch - copy native version
    cp "build-arm64/stdlib.o" "$universal_dir/stdlib-arm64.o" 2>/dev/null || true
    cp "build-x86_64/stdlib.o" "$universal_dir/stdlib-x86_64.o" 2>/dev/null || true

    # Create tarball
    local tarball="$OUTPUT_DIR/eshkol-${VERSION}-macos-universal.tar.gz"
    (
        mkdir -p "$universal_dir/pkg/bin" "$universal_dir/pkg/lib" "$universal_dir/pkg/share/eshkol"
        cp "$universal_dir/eshkol-run" "$universal_dir/pkg/bin/"
        cp "$universal_dir/eshkol-repl" "$universal_dir/pkg/bin/"
        # Include both arch-specific stdlib.o files
        cp "$universal_dir/stdlib-arm64.o" "$universal_dir/pkg/lib/" 2>/dev/null || true
        cp "$universal_dir/stdlib-x86_64.o" "$universal_dir/pkg/lib/" 2>/dev/null || true
        cp lib/stdlib.esk "$universal_dir/pkg/share/eshkol/"
        [ -d lib/core ] && cp -r lib/core "$universal_dir/pkg/share/eshkol/"
        cp README.md LICENSE "$universal_dir/pkg/" 2>/dev/null || true
        tar -czvf "$tarball" -C "$universal_dir/pkg" .
        rm -rf "$universal_dir/pkg"
    )

    echo "Created: $tarball"
}

# Main execution
if [ "$UNIVERSAL" = true ]; then
    echo "Building universal binary (arm64 + x86_64)..."

    # Build for both architectures
    build_arch "arm64"
    build_arch "x86_64"

    # Create universal binary
    create_universal

    # Create separate packages too
    create_package "arm64" "build-arm64" "macos-arm64"
    create_package "x86_64" "build-x86_64" "macos-x64"

    if [ "$RUN_TESTS" = true ]; then
        run_tests "build-arm64"
    fi

elif [ "$TARGET_ARCH" = "native" ]; then
    # Build for host architecture
    if [ "$HOST_ARCH" = "arm64" ]; then
        ARCH_NAME="arm64"
        OUTPUT_NAME="macos-arm64"
    else
        ARCH_NAME="x86_64"
        OUTPUT_NAME="macos-x64"
    fi

    build_arch "$ARCH_NAME"

    if [ "$RUN_TESTS" = true ]; then
        run_tests "build-$ARCH_NAME"
    fi

    create_package "$ARCH_NAME" "build-$ARCH_NAME" "$OUTPUT_NAME"

else
    # Build for specific architecture
    build_arch "$TARGET_ARCH"

    if [ "$TARGET_ARCH" = "arm64" ]; then
        OUTPUT_NAME="macos-arm64"
    else
        OUTPUT_NAME="macos-x64"
    fi

    if [ "$RUN_TESTS" = true ]; then
        run_tests "build-$TARGET_ARCH"
    fi

    create_package "$TARGET_ARCH" "build-$TARGET_ARCH" "$OUTPUT_NAME"
fi

echo ""
echo "macOS build complete!"
echo "Artifacts in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
