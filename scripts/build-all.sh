#!/bin/bash
#
# Master Build Script - Build all targets locally via Docker + native
#
# This script builds:
#   - Linux amd64 (Docker)
#   - Linux arm64 (Docker)
#   - macOS arm64 (native - only on Apple Silicon)
#   - Homebrew formula test
#
# Usage: ./scripts/build-all.sh [options]
#   --linux-only      Only build Linux targets
#   --macos-only      Only build macOS target
#   --homebrew-only   Only test Homebrew formula
#   --skip-tests      Skip running tests
#   --version VER     Set version (default: 1.0.0)
#
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERSION="${ESHKOL_VERSION:-1.0.0}"
BUILD_LINUX=true
BUILD_MACOS=true
TEST_HOMEBREW=true
RUN_TESTS=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --linux-only)
            BUILD_MACOS=false
            TEST_HOMEBREW=false
            shift
            ;;
        --macos-only)
            BUILD_LINUX=false
            shift
            ;;
        --homebrew-only)
            BUILD_LINUX=false
            BUILD_MACOS=false
            shift
            ;;
        --skip-tests)
            RUN_TESTS=false
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Output directory
OUTPUT_DIR="$PROJECT_ROOT/dist"
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Eshkol Build System v${VERSION}${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Detect platform
ARCH=$(uname -m)
OS=$(uname -s)

echo -e "${YELLOW}Platform: ${OS} ${ARCH}${NC}"
echo -e "${YELLOW}Output directory: ${OUTPUT_DIR}${NC}"
echo ""

# Track results (Bash 3.2 compatible - no associative arrays)
BUILD_RESULTS_PASS=""
BUILD_RESULTS_FAIL=""

# Function to log status
log_status() {
    local target=$1
    local status=$2
    if [ "$status" = "success" ]; then
        echo -e "${GREEN}[PASS]${NC} $target"
        BUILD_RESULTS_PASS="$BUILD_RESULTS_PASS $target"
    else
        echo -e "${RED}[FAIL]${NC} $target"
        BUILD_RESULTS_FAIL="$BUILD_RESULTS_FAIL $target"
    fi
}

# =============================================================================
# Linux Builds (Docker)
# =============================================================================
build_linux() {
    echo -e "${BLUE}--- Building Linux targets via Docker ---${NC}"

    # Check Docker availability
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker not found. Skipping Linux builds.${NC}"
        return 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        echo -e "${RED}Docker daemon not running. Skipping Linux builds.${NC}"
        return 1
    fi

    # Build for amd64 (x86_64)
    echo -e "${YELLOW}Building Linux amd64...${NC}"
    if ./scripts/build-docker.sh --platform linux/amd64 --version "$VERSION" --output "$OUTPUT_DIR"; then
        log_status "linux-amd64" "success"
    else
        log_status "linux-amd64" "failed"
    fi

    # Build for arm64 (only if on arm64 host or with QEMU)
    echo -e "${YELLOW}Building Linux arm64...${NC}"
    if ./scripts/build-docker.sh --platform linux/arm64 --version "$VERSION" --output "$OUTPUT_DIR"; then
        log_status "linux-arm64" "success"
    else
        log_status "linux-arm64" "failed"
    fi
}

# =============================================================================
# macOS Build (Native)
# =============================================================================
build_macos() {
    echo -e "${BLUE}--- Building macOS target (native) ---${NC}"

    if [ "$OS" != "Darwin" ]; then
        echo -e "${YELLOW}Not on macOS. Skipping macOS build.${NC}"
        return 0
    fi

    echo -e "${YELLOW}Building macOS ${ARCH}...${NC}"
    if ./scripts/build-macos.sh --version "$VERSION" --output "$OUTPUT_DIR" ${RUN_TESTS:+--run-tests}; then
        log_status "macos-${ARCH}" "success"
    else
        log_status "macos-${ARCH}" "failed"
    fi
}

# =============================================================================
# Homebrew Test
# =============================================================================
test_homebrew() {
    echo -e "${BLUE}--- Testing Homebrew formula ---${NC}"

    if [ "$OS" != "Darwin" ]; then
        echo -e "${YELLOW}Not on macOS. Skipping Homebrew test.${NC}"
        return 0
    fi

    if ! command -v brew &> /dev/null; then
        echo -e "${RED}Homebrew not found. Skipping Homebrew test.${NC}"
        return 1
    fi

    echo -e "${YELLOW}Testing Homebrew formula...${NC}"
    if ./scripts/test-homebrew.sh; then
        log_status "homebrew" "success"
    else
        log_status "homebrew" "failed"
    fi
}

# =============================================================================
# Main execution
# =============================================================================
OVERALL_SUCCESS=true

if [ "$BUILD_LINUX" = true ]; then
    build_linux || OVERALL_SUCCESS=false
fi

if [ "$BUILD_MACOS" = true ]; then
    build_macos || OVERALL_SUCCESS=false
fi

if [ "$TEST_HOMEBREW" = true ]; then
    test_homebrew || OVERALL_SUCCESS=false
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Build Summary${NC}"
echo -e "${BLUE}============================================${NC}"

if [ -n "$BUILD_RESULTS_PASS" ]; then
    for target in $BUILD_RESULTS_PASS; do
        echo -e "  ${GREEN}[PASS]${NC} $target"
    done
fi

if [ -n "$BUILD_RESULTS_FAIL" ]; then
    for target in $BUILD_RESULTS_FAIL; do
        echo -e "  ${RED}[FAIL]${NC} $target"
    done
fi

echo ""
echo -e "Artifacts in: ${OUTPUT_DIR}"
ls -la "$OUTPUT_DIR" 2>/dev/null || true

echo ""
if [ "$OVERALL_SUCCESS" = true ]; then
    echo -e "${GREEN}All builds completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}Some builds failed. Check output above.${NC}"
    exit 1
fi
