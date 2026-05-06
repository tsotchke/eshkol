#!/bin/bash
#
# Docker Build Script - Build Eshkol for Linux via Docker
#
# Supports multi-platform builds (amd64/arm64)
#
# Usage: ./scripts/build-docker.sh [options]
#   --platform PLAT   Target platform: linux/amd64 or linux/arm64
#   --version VER     Version string (default: 1.0.0)
#   --output DIR      Output directory for artifacts
#   --no-cache        Build without Docker cache
#   --debug           Build debug instead of release
#
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
#

set -e

# Defaults
PLATFORM="linux/amd64"
VERSION="1.0.0"
OUTPUT_DIR="./dist"
NO_CACHE=""
BUILD_TYPE="release"

fail() {
    echo "Error: $1" >&2
    exit 1
}

validate_package_version() {
    local version="$1"

    case "$version" in
        ""|*/*|*\\*|*..*|*[!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.+:~-]*)
            fail "unsafe Docker artifact version: $version"
            ;;
    esac

    if [ "${#version}" -gt 128 ]; then
        fail "Docker artifact version is too long: $version"
    fi
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

remove_package_stage() {
    local stage="$1"
    local root="$2"

    case "$stage" in
        "$root"/.pkg.*)
            ;;
        *)
            fail "refusing to remove unexpected Docker package stage path: $stage"
            ;;
    esac

    if [ -L "$stage" ] || { [ -e "$stage" ] && [ ! -d "$stage" ]; }; then
        fail "refusing to remove non-directory or symlinked Docker package stage: $stage"
    fi

    rm -rf -- "$stage"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --debug)
            BUILD_TYPE="debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

validate_package_version "$VERSION"

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Determine architecture from platform
case "$PLATFORM" in
    linux/amd64)
        ARCH="amd64"
        ARCH_NAME="x64"
        ;;
    linux/arm64)
        ARCH="arm64"
        ARCH_NAME="arm64"
        ;;
    *)
        echo "Unsupported platform: $PLATFORM"
        exit 1
        ;;
esac

echo "Building for platform: $PLATFORM (arch: $ARCH)"
echo "Version: $VERSION"
echo "Build type: $BUILD_TYPE"

# Create output directory
require_output_directory "Docker artifact output directory" "$OUTPUT_DIR"

require_docker_container_name() {
    local value="$1"

    case "$value" in
        ""|.*|*/*|*[!A-Za-z0-9_.-]*)
            echo "Unsafe Docker container name: $value" >&2
            exit 1
            ;;
    esac
}

remove_build_container() {
    local container_name="$1"

    require_docker_container_name "$container_name"
    docker container rm --force "$container_name" 2>/dev/null || true
}

# Docker image name
IMAGE_NAME="eshkol-builder-debian-${ARCH}"
CONTAINER_NAME="eshkol-build-${ARCH}-$$"
require_docker_container_name "$CONTAINER_NAME"

# Check if we need to set up QEMU for cross-platform builds
HOST_ARCH=$(uname -m)
if [ "$HOST_ARCH" = "arm64" ] && [ "$ARCH" = "amd64" ]; then
    echo "Setting up QEMU for amd64 emulation on arm64 host..."
    docker run --rm --privileged tonistiigi/binfmt --install amd64 2>/dev/null || true
elif [ "$HOST_ARCH" = "x86_64" ] && [ "$ARCH" = "arm64" ]; then
    echo "Setting up QEMU for arm64 emulation on x86_64 host..."
    docker run --rm --privileged tonistiigi/binfmt --install arm64 2>/dev/null || true
fi

# Build the Docker image
echo "Building Docker image..."
docker build \
    --platform "$PLATFORM" \
    $NO_CACHE \
    -t "$IMAGE_NAME" \
    -f docker/debian/${BUILD_TYPE}/Dockerfile \
    --build-arg VERSION="$VERSION" \
    .

# Create and run container
echo "Running build in container..."
remove_build_container "$CONTAINER_NAME"
docker create --platform "$PLATFORM" --name "$CONTAINER_NAME" "$IMAGE_NAME"

# Extract artifacts
echo "Extracting artifacts..."

# Create arch-specific output directory
ARCH_OUTPUT="$OUTPUT_DIR/linux-${ARCH_NAME}"
require_output_directory "Docker architecture artifact directory" "$ARCH_OUTPUT"
ARCH_OUTPUT="$(cd "$ARCH_OUTPUT" && pwd -P)"

# Copy binaries
docker cp "$CONTAINER_NAME:/app/build/eshkol-run" "$ARCH_OUTPUT/" 2>/dev/null || echo "Warning: eshkol-run not found"
docker cp "$CONTAINER_NAME:/app/build/eshkol-repl" "$ARCH_OUTPUT/" 2>/dev/null || echo "Warning: eshkol-repl not found"
docker cp "$CONTAINER_NAME:/app/build/stdlib.o" "$ARCH_OUTPUT/" 2>/dev/null || echo "Warning: stdlib.o not found"
docker cp "$CONTAINER_NAME:/app/build/stdlib.bc" "$ARCH_OUTPUT/" 2>/dev/null || echo "Warning: stdlib.bc not found"
docker cp "$CONTAINER_NAME:/app/build/libeshkol-static.a" "$ARCH_OUTPUT/" 2>/dev/null || echo "Warning: libeshkol-static.a not found"

# Copy .deb package if it exists
DEB_FILE=$(docker run --rm --platform "$PLATFORM" "$IMAGE_NAME" find /app/build/_packages -name "*.deb" 2>/dev/null | head -1)
if [ -n "$DEB_FILE" ]; then
    docker cp "$CONTAINER_NAME:$DEB_FILE" "$ARCH_OUTPUT/eshkol_${VERSION}_${ARCH}.deb"
    echo "Copied: eshkol_${VERSION}_${ARCH}.deb"
fi

# Create tarball
echo "Creating tarball..."
TARBALL_NAME="eshkol-${VERSION}-linux-${ARCH_NAME}.tar.gz"
PKG_STAGE="$(mktemp -d "$ARCH_OUTPUT/.pkg.XXXXXX")"
(
    mkdir -p "$PKG_STAGE/bin" "$PKG_STAGE/lib" "$PKG_STAGE/share/eshkol"
    cp "$ARCH_OUTPUT/eshkol-run" "$PKG_STAGE/bin/" 2>/dev/null || true
    cp "$ARCH_OUTPUT/eshkol-repl" "$PKG_STAGE/bin/" 2>/dev/null || true
    cp "$ARCH_OUTPUT/stdlib.o" "$PKG_STAGE/lib/" 2>/dev/null || true
    cp "$ARCH_OUTPUT/stdlib.bc" "$PKG_STAGE/lib/" 2>/dev/null || true
    cp "$ARCH_OUTPUT/libeshkol-static.a" "$PKG_STAGE/lib/" 2>/dev/null || true
    cp "$PROJECT_ROOT/lib/stdlib.esk" "$PKG_STAGE/share/eshkol/" 2>/dev/null || true
    if [ -d "$PROJECT_ROOT/lib/core" ]; then
        cp -r "$PROJECT_ROOT/lib/core" "$PKG_STAGE/share/eshkol/"
    fi
    cp "$PROJECT_ROOT/README.md" "$PKG_STAGE/" 2>/dev/null || true
    cp "$PROJECT_ROOT/LICENSE" "$PKG_STAGE/" 2>/dev/null || true
    tar -czvf "$ARCH_OUTPUT/$TARBALL_NAME" -C "$PKG_STAGE" .
)
remove_package_stage "$PKG_STAGE" "$ARCH_OUTPUT"

# Cleanup container
remove_build_container "$CONTAINER_NAME"

echo ""
echo "Build complete! Artifacts in: $ARCH_OUTPUT"
ls -la "$ARCH_OUTPUT"
