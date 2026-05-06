#!/bin/bash
#
# Build Eshkol packages using Docker
#
# Usage: ./scripts/make-docker.sh [BUILD_DIR] [VERSION] [TYPE]
#   BUILD_DIR  - Output directory (default: ./dist)
#   VERSION    - Version string (default: 1.0.0)
#   TYPE       - Build type: release or debug (default: release)
#
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
#

set -e

BUILD_DIR="${1:-./dist}"
VERSION="${2:-1.0.0}"
TYPE="${3:-release}"

case "$TYPE" in
    release|debug)
        ;;
    *)
        echo "Unsupported Docker build type: $TYPE" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Building Eshkol packages via Docker"
echo "  Output: $BUILD_DIR"
echo "  Version: $VERSION"
echo "  Type: $TYPE"
echo ""

mkdir -p "$BUILD_DIR"

require_docker_name() {
    local value="$1"
    local label="$2"

    case "$value" in
        ""|.*|*/*|*[!A-Za-z0-9_.-]*)
            echo "Unsafe Docker $label: $value" >&2
            exit 1
            ;;
    esac
}

remove_build_container() {
    local name="$1"

    require_docker_name "$name" "container name"
    docker container rm --force "$name" 2>/dev/null || true
}

# Determine host architecture for platform selection
HOST_ARCH=$(uname -m)
if [ "$HOST_ARCH" = "arm64" ]; then
    # On Apple Silicon, build for both amd64 (via emulation) and arm64
    PLATFORMS=("linux/amd64" "linux/arm64")
else
    PLATFORMS=("linux/amd64")
fi

# Setup QEMU for cross-platform builds if needed
if [ "$HOST_ARCH" = "arm64" ]; then
    echo "Setting up QEMU for cross-platform builds..."
    docker run --rm --privileged tonistiigi/binfmt --install all 2>/dev/null || true
fi

for os_dir in docker/*/; do
    os=$(basename "$os_dir")

    # Skip if no Dockerfile for this type
    if [ ! -f "docker/$os/$TYPE/Dockerfile" ]; then
        echo "Skipping $os (no $TYPE Dockerfile)"
        continue
    fi

    for platform in "${PLATFORMS[@]}"; do
        arch="${platform#linux/}"
        image_name="eshkol_${os}_${TYPE}_${arch}"
        container_name="eshkol_build_${os}_${TYPE}_${arch}_$$"
        require_docker_name "$image_name" "image name"
        require_docker_name "$container_name" "container name"

        echo ""
        echo "=== Building $os ($arch) ==="

        # Build image
        echo "Building Docker image: $image_name"
        if ! docker build \
            --platform "$platform" \
            --build-arg VERSION="$VERSION" \
            -t "$image_name" \
            -f "docker/$os/$TYPE/Dockerfile" \
            .; then
            echo "ERROR: Build failed for $os ($arch)"
            continue
        fi

        # Create container and extract artifacts
        echo "Extracting artifacts..."
        remove_build_container "$container_name"
        container_id=$(docker create --platform "$platform" --name "$container_name" "$image_name")

        # Create output directory
        out_dir="$BUILD_DIR/$os/$TYPE/$arch"
        mkdir -p "$out_dir"

        # Copy binaries
        docker cp "$container_id:/app/build/eshkol-run" "$out_dir/" 2>/dev/null || echo "  Warning: eshkol-run not found"
        docker cp "$container_id:/app/build/eshkol-repl" "$out_dir/" 2>/dev/null || echo "  Warning: eshkol-repl not found"
        docker cp "$container_id:/app/build/stdlib.o" "$out_dir/" 2>/dev/null || echo "  Warning: stdlib.o not found"
        docker cp "$container_id:/app/build/stdlib.bc" "$out_dir/" 2>/dev/null || echo "  Warning: stdlib.bc not found"

        # Find and copy .deb package
        deb_file=$(docker run --rm --platform "$platform" "$image_name" find /app/build -name "*.deb" 2>/dev/null | head -1)
        if [ -n "$deb_file" ]; then
            docker cp "$container_id:$deb_file" "$out_dir/eshkol_${VERSION}_${arch}.deb"
            echo "  Copied: eshkol_${VERSION}_${arch}.deb"
        fi

        # Cleanup
        remove_build_container "$container_name"

        # List outputs
        echo "  Artifacts in $out_dir:"
        ls -la "$out_dir"
    done
done

echo ""
echo "=== Build Complete ==="
echo "Artifacts in: $BUILD_DIR"
find "$BUILD_DIR" -type f -name "*.deb" -o -name "eshkol-run" -o -name "eshkol-repl" | head -20
