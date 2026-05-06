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

VERSION="${1:-1.1.0}"
BUILD_DIR="build"
PACKAGE_NAME="eshkol"

fail() {
    echo "Error: $1" >&2
    exit 1
}

validate_package_version() {
    local version="$1"

    case "$version" in
        ""|*/*|*\\*|*..*|*[!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.+:~-]*)
            fail "unsafe Debian package version: $version"
            ;;
    esac

    if [ "${#version}" -gt 128 ]; then
        fail "Debian package version is too long: $version"
    fi
}

require_regular_executable() {
    local label="$1"
    local path="$2"

    if [ -L "$path" ] || ! test -f "$path" || ! test -s "$path" || ! test -x "$path"; then
        fail "$label missing, empty, symlinked, or not executable: $path"
    fi
}

require_regular_package_file() {
    local path="$1"

    if [ -L "$path" ] || ! test -f "$path" || ! test -s "$path"; then
        fail "generated Debian package missing, empty, symlinked, or not a regular file: $path"
    fi
}

validate_package_version "$VERSION"

echo "Building Debian package for ${PACKAGE_NAME} v${VERSION}"

# Ensure we're in the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Build if not already built
ESHKOL_RUN="${BUILD_DIR}/eshkol-run"
if ! test -e "$ESHKOL_RUN"; then
    echo "Building project..."
    cmake -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --parallel
fi
require_regular_executable "eshkol-run" "$ESHKOL_RUN"

# Generate .deb package using CPack
echo "Generating Debian package..."
cd "${BUILD_DIR}"
cpack -G DEB \
    -D CPACK_PACKAGE_VERSION="${VERSION}" \
    -D CPACK_DEBIAN_PACKAGE_VERSION="${VERSION}"

# Find the generated package (CPack outputs to _packages/ subdirectory)
DEB_FILE=$(find . -name "*.deb" -type f 2>/dev/null | head -1)
if [ -z "$DEB_FILE" ]; then
    fail "No .deb file generated"
fi
require_regular_package_file "$DEB_FILE"

# Move to project root with standardized name
ARCH=$(dpkg --print-architecture 2>/dev/null || echo "amd64")
OUTPUT_NAME="${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
mv -- "$DEB_FILE" "../${OUTPUT_NAME}"

echo "Successfully created: ${OUTPUT_NAME}"
