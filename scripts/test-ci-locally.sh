#!/bin/bash
#
# Test GitHub CI Locally
#
# This script replicates the exact GitHub Actions CI workflow
# so you can test locally before pushing.
#
# Usage: ./scripts/test-ci-locally.sh [target]
#   linux         Test Linux build via Docker (matches ubuntu-22.04)
#   macos         Test macOS build natively
#   release       Test full release workflow
#   homebrew      Test Homebrew formula update process
#   all           Test everything
#
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TARGET="${1:-all}"
VERSION="${ESHKOL_VERSION:-1.0.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Results tracking (Bash 3.2 compatible - no associative arrays)
RESULTS_PASS=""
RESULTS_FAIL=""
OVERALL_PASS=true

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    RESULTS_PASS="$RESULTS_PASS $1"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    RESULTS_FAIL="$RESULTS_FAIL $1"
    OVERALL_PASS=false
}

# =============================================================================
# Linux CI Test (matches .github/workflows/ci.yml build-linux job)
# =============================================================================
test_linux_ci() {
    echo -e "${BLUE}=== Testing Linux CI (ubuntu-22.04) ===${NC}"

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker not found - cannot test Linux CI${NC}"
        log_fail "linux-ci"
        return 1
    fi

    # Build Docker image that exactly matches CI
    DOCKER_IMAGE="eshkol-ci-test"
    CONTAINER_NAME="eshkol-ci-test-$$"

    echo "Building CI test image..."

    # Create a temporary Dockerfile that exactly matches CI
    cat > /tmp/Dockerfile.ci-test << 'DOCKERFILE'
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Add LLVM 17 repository (required - not in Ubuntu 22.04 default repos)
RUN apt-get update && apt-get install -y \
    wget gnupg software-properties-common \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" >> /etc/apt/sources.list.d/llvm.list \
    && apt-get update

# Install build dependencies - matching GitHub Actions CI
RUN apt-get install -y \
    cmake ninja-build \
    llvm-17-dev llvm-17 \
    libreadline-dev \
    g++ \
    file \
    pkg-config

# Create symlinks for LLVM tools
RUN ln -sf /usr/bin/llvm-config-17 /usr/bin/llvm-config

WORKDIR /app
COPY . .

# Configure
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build
RUN cmake --build build --parallel

# Verify binaries exist
RUN test -f build/eshkol-run && echo "eshkol-run: OK" || exit 1
RUN test -f build/eshkol-repl && echo "eshkol-repl: OK" || exit 1
RUN test -f build/stdlib.o && echo "stdlib.o: OK" || exit 1

# Run all tests exactly as CI does
RUN ./scripts/run_types_tests.sh
RUN ./scripts/run_cpp_type_tests.sh
RUN ./scripts/run_list_tests.sh
RUN ./scripts/run_autodiff_tests.sh
RUN ./scripts/run_memory_tests.sh
RUN ./scripts/run_modules_tests.sh
RUN ./scripts/run_stdlib_tests.sh
RUN ./scripts/run_features_tests.sh
RUN ./scripts/run_ml_tests.sh
RUN ./scripts/run_neural_tests.sh

CMD ["echo", "All CI tests passed!"]
DOCKERFILE

    # Build and run
    if docker build --platform linux/amd64 -t "$DOCKER_IMAGE" -f /tmp/Dockerfile.ci-test .; then
        log_pass "linux-ci-build"
    else
        log_fail "linux-ci-build"
        rm /tmp/Dockerfile.ci-test
        return 1
    fi

    rm /tmp/Dockerfile.ci-test
    echo -e "${GREEN}Linux CI test passed!${NC}"
    log_pass "linux-ci"
}

# =============================================================================
# macOS CI Test (matches .github/workflows/ci.yml build-macos-arm job)
# =============================================================================
test_macos_ci() {
    echo -e "${BLUE}=== Testing macOS CI (arm64) ===${NC}"

    if [ "$(uname)" != "Darwin" ]; then
        echo -e "${YELLOW}Not on macOS - skipping${NC}"
        return 0
    fi

    ARCH=$(uname -m)
    echo "Architecture: $ARCH"

    # Check dependencies (matching CI: brew install llvm@17 cmake ninja readline)
    echo "Checking dependencies..."
    DEPS_OK=true
    for dep in llvm@17 cmake ninja readline; do
        if brew list "$dep" &>/dev/null; then
            echo -e "  ${GREEN}[OK]${NC} $dep"
        else
            echo -e "  ${RED}[MISSING]${NC} $dep"
            DEPS_OK=false
        fi
    done

    if [ "$DEPS_OK" = false ]; then
        echo "Install missing dependencies: brew install llvm@17 cmake ninja readline"
        log_fail "macos-ci-deps"
        return 1
    fi
    log_pass "macos-ci-deps"

    # Set PATH exactly as CI does
    if [ "$ARCH" = "arm64" ]; then
        export PATH="/opt/homebrew/opt/llvm@17/bin:$PATH"
    else
        export PATH="/usr/local/opt/llvm@17/bin:$PATH"
    fi

    # Clean build
    BUILD_DIR="build-ci-test"
    rm -rf "$BUILD_DIR"

    # Configure - exactly as CI does
    echo "Configuring..."
    if cmake -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release; then
        log_pass "macos-ci-configure"
    else
        log_fail "macos-ci-configure"
        return 1
    fi

    # Build
    echo "Building..."
    if cmake --build "$BUILD_DIR" --parallel; then
        log_pass "macos-ci-build"
    else
        log_fail "macos-ci-build"
        return 1
    fi

    # Verify outputs
    echo "Verifying outputs..."
    for f in eshkol-run eshkol-repl stdlib.o; do
        if [ -f "$BUILD_DIR/$f" ]; then
            echo -e "  ${GREEN}[OK]${NC} $f"
        else
            echo -e "  ${RED}[MISSING]${NC} $f"
            log_fail "macos-ci-outputs"
            return 1
        fi
    done
    log_pass "macos-ci-outputs"

    # Run tests (matching CI)
    echo "Running tests..."
    if ./scripts/run_types_tests.sh; then
        log_pass "macos-ci-types-test"
    else
        log_fail "macos-ci-types-test"
    fi

    if ./scripts/run_list_tests.sh; then
        log_pass "macos-ci-list-test"
    else
        log_fail "macos-ci-list-test"
    fi

    if ./scripts/run_autodiff_tests.sh; then
        log_pass "macos-ci-autodiff-test"
    else
        log_fail "macos-ci-autodiff-test"
    fi

    # Cleanup
    rm -rf "$BUILD_DIR"

    echo -e "${GREEN}macOS CI test completed!${NC}"
    log_pass "macos-ci"
}

# =============================================================================
# Release Workflow Test
# =============================================================================
test_release() {
    echo -e "${BLUE}=== Testing Release Workflow ===${NC}"

    # Test Linux release build
    echo "Testing Linux release package..."

    cat > /tmp/Dockerfile.release-test << DOCKERFILE
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Add LLVM 17 repository
RUN apt-get update && apt-get install -y wget gnupg software-properties-common \\
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \\
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" >> /etc/apt/sources.list.d/llvm.list \\
    && apt-get update

RUN apt-get install -y \\
    cmake ninja-build llvm-17 llvm-17-dev libreadline-dev dpkg-dev g++ file pkg-config

# Create symlinks for LLVM tools
RUN ln -sf /usr/bin/llvm-config-17 /usr/bin/llvm-config

WORKDIR /app
COPY . .

# Build
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --parallel

# Run subset of tests
RUN ./scripts/run_types_tests.sh
RUN ./scripts/run_list_tests.sh
RUN ./scripts/run_autodiff_tests.sh

# Package (matching release workflow)
RUN mkdir -p pkg/bin pkg/lib pkg/share/eshkol
RUN cp build/eshkol-run pkg/bin/
RUN cp build/eshkol-repl pkg/bin/
RUN cp build/stdlib.o pkg/lib/
RUN cp lib/stdlib.esk pkg/share/eshkol/
RUN if [ -d lib/core ]; then cp -r lib/core pkg/share/eshkol/; fi
RUN cp README.md LICENSE pkg/ 2>/dev/null || true
RUN cd pkg && tar -czvf /app/eshkol-v${VERSION}-linux-x64.tar.gz .

# Build Debian package
RUN ./scripts/build-deb.sh "${VERSION}"

# Verify packages
RUN ls -la /app/*.tar.gz /app/*.deb

CMD ["echo", "Release build complete!"]
DOCKERFILE

    if docker build --platform linux/amd64 -t "eshkol-release-test" -f /tmp/Dockerfile.release-test .; then
        log_pass "release-linux-build"

        # Extract and verify artifacts
        CONTAINER_ID=$(docker create eshkol-release-test)
        mkdir -p dist/release-test
        docker cp "$CONTAINER_ID:/app/eshkol-v${VERSION}-linux-x64.tar.gz" dist/release-test/ 2>/dev/null || true
        docker cp "$CONTAINER_ID:/app/eshkol_${VERSION}_amd64.deb" dist/release-test/ 2>/dev/null || true
        docker rm "$CONTAINER_ID"

        if [ -f "dist/release-test/eshkol-v${VERSION}-linux-x64.tar.gz" ]; then
            log_pass "release-linux-tarball"
        else
            log_fail "release-linux-tarball"
        fi

        if [ -f "dist/release-test/eshkol_${VERSION}_amd64.deb" ]; then
            log_pass "release-linux-deb"
        else
            log_fail "release-linux-deb"
        fi
    else
        log_fail "release-linux-build"
    fi

    rm /tmp/Dockerfile.release-test

    log_pass "release"
}

# =============================================================================
# Homebrew Test
# =============================================================================
test_homebrew() {
    echo -e "${BLUE}=== Testing Homebrew Workflow ===${NC}"

    if [ "$(uname)" != "Darwin" ]; then
        echo -e "${YELLOW}Not on macOS - skipping Homebrew test${NC}"
        return 0
    fi

    # Test formula syntax
    if [ -f "packaging/homebrew/eshkol.rb" ]; then
        echo "Checking formula syntax..."
        if ruby -c packaging/homebrew/eshkol.rb; then
            log_pass "homebrew-syntax"
        else
            log_fail "homebrew-syntax"
        fi

        # Test that the build process works (what Homebrew will do)
        echo "Testing Homebrew-style build..."
        ./scripts/test-homebrew.sh --local && log_pass "homebrew-local-build" || log_fail "homebrew-local-build"
    else
        echo -e "${RED}Formula not found${NC}"
        log_fail "homebrew"
        return 1
    fi

    log_pass "homebrew"
}

# =============================================================================
# Main
# =============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  GitHub CI Local Test - v${VERSION}${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

case "$TARGET" in
    linux)
        test_linux_ci
        ;;
    macos)
        test_macos_ci
        ;;
    release)
        test_release
        ;;
    homebrew)
        test_homebrew
        ;;
    all)
        test_linux_ci
        test_macos_ci
        test_release
        test_homebrew
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [linux|macos|release|homebrew|all]"
        exit 1
        ;;
esac

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}============================================${NC}"

if [ -n "$RESULTS_PASS" ]; then
    for test in $RESULTS_PASS; do
        echo -e "  ${GREEN}[PASS]${NC} $test"
    done
fi

if [ -n "$RESULTS_FAIL" ]; then
    for test in $RESULTS_FAIL; do
        echo -e "  ${RED}[FAIL]${NC} $test"
    done
fi

echo ""
if [ "$OVERALL_PASS" = true ]; then
    echo -e "${GREEN}All tests passed! CI should work on GitHub.${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Fix issues before pushing.${NC}"
    exit 1
fi
