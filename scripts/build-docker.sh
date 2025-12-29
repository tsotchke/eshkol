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
mkdir -p "$OUTPUT_DIR"

# Docker image name
IMAGE_NAME="eshkol-builder-debian-${ARCH}"
CONTAINER_NAME="eshkol-build-${ARCH}-$$"

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
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
docker create --platform "$PLATFORM" --name "$CONTAINER_NAME" "$IMAGE_NAME"

# Extract artifacts
echo "Extracting artifacts..."

# Create arch-specific output directory
ARCH_OUTPUT="$OUTPUT_DIR/linux-${ARCH_NAME}"
mkdir -p "$ARCH_OUTPUT"

# Copy binaries
docker cp "$CONTAINER_NAME:/app/build/eshkol-run" "$ARCH_OUTPUT/" 2>/dev/null || echo "Warning: eshkol-run not found"
docker cp "$CONTAINER_NAME:/app/build/eshkol-repl" "$ARCH_OUTPUT/" 2>/dev/null || echo "Warning: eshkol-repl not found"
docker cp "$CONTAINER_NAME:/app/build/stdlib.o" "$ARCH_OUTPUT/" 2>/dev/null || echo "Warning: stdlib.o not found"
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
(
    cd "$ARCH_OUTPUT"
    mkdir -p pkg/bin pkg/lib pkg/share/eshkol
    cp eshkol-run pkg/bin/ 2>/dev/null || true
    cp eshkol-repl pkg/bin/ 2>/dev/null || true
    cp stdlib.o pkg/lib/ 2>/dev/null || true
    cp libeshkol-static.a pkg/lib/ 2>/dev/null || true
    cp "$PROJECT_ROOT/lib/stdlib.esk" pkg/share/eshkol/ 2>/dev/null || true
    if [ -d "$PROJECT_ROOT/lib/core" ]; then
        cp -r "$PROJECT_ROOT/lib/core" pkg/share/eshkol/
    fi
    cp "$PROJECT_ROOT/README.md" pkg/ 2>/dev/null || true
    cp "$PROJECT_ROOT/LICENSE" pkg/ 2>/dev/null || true
    tar -czvf "$TARBALL_NAME" -C pkg .
    rm -rf pkg
)

# Cleanup container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo ""
echo "Build complete! Artifacts in: $ARCH_OUTPUT"
ls -la "$ARCH_OUTPUT"
