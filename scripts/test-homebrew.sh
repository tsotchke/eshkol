#!/bin/bash
#
# Homebrew Formula Test Script
#
# Tests the Homebrew formula locally before publishing
#
# Usage: ./scripts/test-homebrew.sh [options]
#   --install       Actually install via brew (requires tap setup)
#   --audit         Run brew audit on formula
#   --local         Test from local build instead of GitHub
#
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DO_INSTALL=false
DO_AUDIT=false
LOCAL_TEST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --install)
            DO_INSTALL=true
            shift
            ;;
        --audit)
            DO_AUDIT=true
            shift
            ;;
        --local)
            LOCAL_TEST=true
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

echo -e "${YELLOW}=== Homebrew Formula Test ===${NC}"

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo -e "${RED}Homebrew not installed${NC}"
    exit 1
fi

# Check formula exists
FORMULA_PATH="packaging/homebrew/eshkol.rb"
if [ ! -f "$FORMULA_PATH" ]; then
    echo -e "${RED}Formula not found: $FORMULA_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Formula found: $FORMULA_PATH${NC}"

# Check dependencies are available
echo -e "${YELLOW}Checking dependencies...${NC}"
DEPS=("llvm@17" "cmake" "ninja" "readline")
for dep in "${DEPS[@]}"; do
    if brew list "$dep" &>/dev/null; then
        echo -e "  ${GREEN}[OK]${NC} $dep"
    else
        echo -e "  ${YELLOW}[MISSING]${NC} $dep - will be installed"
    fi
done

# Syntax check the formula
echo -e "${YELLOW}Checking formula syntax...${NC}"
if ruby -c "$FORMULA_PATH" &>/dev/null; then
    echo -e "  ${GREEN}[OK]${NC} Ruby syntax valid"
else
    echo -e "  ${RED}[FAIL]${NC} Ruby syntax error"
    ruby -c "$FORMULA_PATH"
    exit 1
fi

# Test local build (simulating what Homebrew does)
if [ "$LOCAL_TEST" = true ]; then
    echo -e "${YELLOW}Testing local build (simulating Homebrew)...${NC}"

    # Get LLVM path
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        LLVM_PATH="/opt/homebrew/opt/llvm@17"
    else
        LLVM_PATH="/usr/local/opt/llvm@17"
    fi

    if [ ! -d "$LLVM_PATH" ]; then
        echo -e "${RED}LLVM 17 not found. Install with: brew install llvm@17${NC}"
        exit 1
    fi

    export PATH="$LLVM_PATH/bin:$PATH"

    # Clean and build
    BUILD_DIR="build-homebrew-test"
    rm -rf "$BUILD_DIR"

    echo "Configuring..."
    cmake -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release

    echo "Building..."
    cmake --build "$BUILD_DIR" --parallel

    # Verify outputs exist
    echo -e "${YELLOW}Verifying build outputs...${NC}"
    OUTPUTS=("$BUILD_DIR/eshkol-run" "$BUILD_DIR/eshkol-repl" "$BUILD_DIR/stdlib.o")
    ALL_OK=true
    for output in "${OUTPUTS[@]}"; do
        if [ -f "$output" ]; then
            echo -e "  ${GREEN}[OK]${NC} $output"
        else
            echo -e "  ${RED}[MISSING]${NC} $output"
            ALL_OK=false
        fi
    done

    if [ "$ALL_OK" = false ]; then
        echo -e "${RED}Build incomplete${NC}"
        exit 1
    fi

    # Run a quick test
    echo -e "${YELLOW}Running quick test...${NC}"
    TEST_FILE=$(mktemp /tmp/eshkol-test.XXXXXX.esk)
    echo '(display "Hello from Homebrew test!")' > "$TEST_FILE"

    if "$BUILD_DIR/eshkol-run" "$TEST_FILE" -L"$BUILD_DIR"; then
        if [ -f "a.out" ]; then
            echo "Running compiled output..."
            ./a.out
            rm -f a.out
            echo -e "  ${GREEN}[OK]${NC} Compilation and execution successful"
        fi
    else
        echo -e "  ${RED}[FAIL]${NC} Compilation failed"
    fi

    rm -f "$TEST_FILE"

    # Cleanup
    rm -rf "$BUILD_DIR"
fi

# Run brew audit if requested
if [ "$DO_AUDIT" = true ]; then
    echo -e "${YELLOW}Running brew audit...${NC}"

    # Create a temporary tap directory
    TAP_DIR=$(mktemp -d)
    mkdir -p "$TAP_DIR/Formula"
    cp "$FORMULA_PATH" "$TAP_DIR/Formula/eshkol.rb"

    # Run audit (may fail on URL check if not published yet)
    brew audit --strict "$TAP_DIR/Formula/eshkol.rb" 2>&1 || true

    rm -rf "$TAP_DIR"
fi

# Test actual installation if requested
if [ "$DO_INSTALL" = true ]; then
    echo -e "${YELLOW}Testing actual Homebrew installation...${NC}"

    # Check if tap exists
    if ! brew tap | grep -q "tsotchke/eshkol"; then
        echo "Tap not found. Creating local tap..."

        # Create a local tap for testing
        TAP_PATH="$(brew --repository)/Library/Taps/tsotchke/homebrew-eshkol"
        mkdir -p "$TAP_PATH/Formula"
        cp "$FORMULA_PATH" "$TAP_PATH/Formula/eshkol.rb"

        echo "Installing from local tap..."
        brew install --build-from-source tsotchke/eshkol/eshkol || {
            echo -e "${RED}Installation failed${NC}"
            rm -rf "$TAP_PATH"
            exit 1
        }

        # Test the installed version
        echo "Testing installed binaries..."
        which eshkol-run && eshkol-run --version
        which eshkol-repl

        # Cleanup
        brew uninstall eshkol 2>/dev/null || true
        rm -rf "$TAP_PATH"
    else
        echo "Using existing tap..."
        brew reinstall --build-from-source eshkol
    fi
fi

echo ""
echo -e "${GREEN}=== Homebrew test complete ===${NC}"
