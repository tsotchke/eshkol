#!/bin/bash
#
# Build Debian package for Eshkol
#
# Usage: ./scripts/build-deb.sh <version>
#
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
#

set -e

VERSION="${1:-1.0.0}"
BUILD_DIR="build"
PACKAGE_NAME="eshkol"

echo "Building Debian package for ${PACKAGE_NAME} v${VERSION}"

# Ensure we're in the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Build if not already built
if [ ! -f "${BUILD_DIR}/eshkol-run" ]; then
    echo "Building project..."
    cmake -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --parallel
fi

# Generate .deb package using CPack
echo "Generating Debian package..."
cd "${BUILD_DIR}"
cpack -G DEB \
    -D CPACK_PACKAGE_VERSION="${VERSION}" \
    -D CPACK_DEBIAN_PACKAGE_VERSION="${VERSION}"

# Find the generated package (CPack outputs to _packages/ subdirectory)
DEB_FILE=$(find . -name "*.deb" -type f 2>/dev/null | head -1)
if [ -z "$DEB_FILE" ]; then
    echo "Error: No .deb file generated"
    exit 1
fi

# Move to project root with standardized name
OUTPUT_NAME="${PACKAGE_NAME}_${VERSION}_amd64.deb"
mv "$DEB_FILE" "../${OUTPUT_NAME}"

echo "Successfully created: ${OUTPUT_NAME}"
