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

fail() {
    echo "Error: $1" >&2
    exit 1
}

validate_package_version() {
    local version="$1"

    case "$version" in
        ""|*/*|*\\*|*..*|*[!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.+:~-]*)
            fail "unsafe macOS artifact version: $version"
            ;;
    esac

    if [ "${#version}" -gt 128 ]; then
        fail "macOS artifact version is too long: $version"
    fi
}

validate_target_arch() {
    local arch="$1"

    case "$arch" in
        native|arm64|x86_64)
            ;;
        *)
            fail "unsupported macOS target architecture: $arch"
            ;;
    esac
}

require_output_directory() {
    local label="$1"
    local path="$2"

    if [ -L "$path" ]; then
        fail "$label must not be a symlink: $path"
    fi

    mkdir -p "$path"

    if [ -L "$path" ] || [ ! -d "$path" ]; then
        fail "$label missing or symlinked after creation: $path"
    fi
}

remove_build_directory() {
    local path="$1"

    case "$path" in
        build-arm64|build-x86_64)
            ;;
        *)
            fail "refusing to remove unexpected macOS build directory: $path"
            ;;
    esac

    if [ -L "$path" ] || { [ -e "$path" ] && [ ! -d "$path" ]; }; then
        fail "refusing to remove non-directory or symlinked macOS build path: $path"
    fi

    rm -rf -- "$path"
}

remove_package_stage() {
    local stage="$1"
    local root="$2"

    case "$stage" in
        "$root"/.pkg.*)
            ;;
        *)
            fail "refusing to remove unexpected macOS package stage path: $stage"
            ;;
    esac

    if [ -L "$stage" ] || { [ -e "$stage" ] && [ ! -d "$stage" ]; }; then
        fail "refusing to remove non-directory or symlinked macOS package stage: $stage"
    fi

    rm -rf -- "$stage"
}

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

validate_package_version "$VERSION"
validate_target_arch "$TARGET_ARCH"

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
source "${SCRIPT_DIR}/lib/llvm21-env.sh"

# Detect host architecture
HOST_ARCH=$(uname -m)
echo "Host architecture: $HOST_ARCH"

eshkol_activate_llvm_toolchain
LLVM_PATH="${ESHKOL_LLVM_ROOT}"

require_output_directory "macOS artifact output directory" "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"

# Function to build for a specific architecture
build_arch() {
    local arch=$1
    local build_dir="build-$arch"

    echo "Building for architecture: $arch"

    if [ "$CLEAN" = true ] && [ -d "$build_dir" ]; then
        remove_build_directory "$build_dir"
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
        -DESHKOL_REQUIRED_LLVM_MAJOR="${ESHKOL_REQUIRED_LLVM_MAJOR}" \
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
    require_output_directory "macOS package output directory" "$pkg_dir"
    pkg_dir="$(cd "$pkg_dir" && pwd -P)"

    # Copy binaries
    cp "$build_dir/eshkol-run" "$pkg_dir/"
    cp "$build_dir/eshkol-repl" "$pkg_dir/"
    cp "$build_dir/stdlib.o" "$pkg_dir/"
    cp "$build_dir/stdlib.bc" "$pkg_dir/"

    # Create tarball
    local tarball="$OUTPUT_DIR/eshkol-${VERSION}-${output_name}.tar.gz"
    local pkg_stage
    pkg_stage="$(mktemp -d "$pkg_dir/.pkg.XXXXXX")"
    (
        mkdir -p "$pkg_stage/bin" "$pkg_stage/lib" "$pkg_stage/share/eshkol"
        cp "$build_dir/eshkol-run" "$pkg_stage/bin/"
        cp "$build_dir/eshkol-repl" "$pkg_stage/bin/"
        cp "$build_dir/stdlib.o" "$pkg_stage/lib/"
        cp "$build_dir/stdlib.bc" "$pkg_stage/lib/"
        cp lib/stdlib.esk "$pkg_stage/share/eshkol/"
        [ -d lib/core ] && cp -r lib/core "$pkg_stage/share/eshkol/"
        cp README.md LICENSE "$pkg_stage/" 2>/dev/null || true
        tar -czvf "$tarball" -C "$pkg_stage" .
    )
    remove_package_stage "$pkg_stage" "$pkg_dir"

    echo "Created: $tarball"
}

# Function to create universal binary
create_universal() {
    echo "Creating universal binary..."

    local universal_dir="$OUTPUT_DIR/macos-universal"
    require_output_directory "macOS universal artifact directory" "$universal_dir"
    universal_dir="$(cd "$universal_dir" && pwd -P)"

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

    # stdlib artifacts need to be built separately for each arch - copy native versions
    cp "build-arm64/stdlib.o" "$universal_dir/stdlib-arm64.o" 2>/dev/null || true
    cp "build-x86_64/stdlib.o" "$universal_dir/stdlib-x86_64.o" 2>/dev/null || true
    cp "build-arm64/stdlib.bc" "$universal_dir/stdlib-arm64.bc" 2>/dev/null || true
    cp "build-x86_64/stdlib.bc" "$universal_dir/stdlib-x86_64.bc" 2>/dev/null || true

    # Create tarball
    local tarball="$OUTPUT_DIR/eshkol-${VERSION}-macos-universal.tar.gz"
    local pkg_stage
    pkg_stage="$(mktemp -d "$universal_dir/.pkg.XXXXXX")"
    (
        mkdir -p "$pkg_stage/bin" "$pkg_stage/lib" "$pkg_stage/share/eshkol"
        cp "$universal_dir/eshkol-run" "$pkg_stage/bin/"
        cp "$universal_dir/eshkol-repl" "$pkg_stage/bin/"
        # Include arch-specific stdlib artifacts.
        cp "$universal_dir/stdlib-arm64.o" "$pkg_stage/lib/" 2>/dev/null || true
        cp "$universal_dir/stdlib-x86_64.o" "$pkg_stage/lib/" 2>/dev/null || true
        cp "$universal_dir/stdlib-arm64.bc" "$pkg_stage/lib/" 2>/dev/null || true
        cp "$universal_dir/stdlib-x86_64.bc" "$pkg_stage/lib/" 2>/dev/null || true
        cp lib/stdlib.esk "$pkg_stage/share/eshkol/"
        [ -d lib/core ] && cp -r lib/core "$pkg_stage/share/eshkol/"
        cp README.md LICENSE "$pkg_stage/" 2>/dev/null || true
        tar -czvf "$tarball" -C "$pkg_stage" .
    )
    remove_package_stage "$pkg_stage" "$universal_dir"

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
