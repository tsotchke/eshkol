#!/bin/bash
#
# Test GitHub CI Locally
#
# This script replicates the exact GitHub Actions CI workflow
# so you can test locally before pushing.
#
# Usage: ./scripts/test-ci-locally.sh [target]
#   linux         Test Linux lite build via Docker (matches ubuntu-22.04)
#   linux-xla     Test Linux XLA lane via Docker
#   linux-cuda    Test Linux CUDA lane via Docker
#   matrix        Validate workflow matrix metadata and Linux matrix lanes
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
LLVM_MAJOR="${ESHKOL_REQUIRED_LLVM_MAJOR:-21}"

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
    cat > /tmp/Dockerfile.ci-test << DOCKERFILE
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Add the requested LLVM repository (required - not in Ubuntu 22.04 default repos)
RUN apt-get update && apt-get install -y \
    wget gnupg software-properties-common \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_MAJOR} main" >> /etc/apt/sources.list.d/llvm.list \
    && apt-get update

# Install build dependencies - matching GitHub Actions CI
RUN apt-get install -y \
    cmake ninja-build \
    llvm-${LLVM_MAJOR}-dev llvm-${LLVM_MAJOR} \
    libreadline-dev \
    g++ \
    file \
    pkg-config \
    libssl-dev \
    libncurses-dev

WORKDIR /app
COPY . .

# Configure
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DESHKOL_REQUIRED_LLVM_MAJOR=${LLVM_MAJOR} -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-${LLVM_MAJOR}

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

test_linux_backend_ci() {
    local backend_name="$1"
    local build_dir="$2"
    local xla_enabled="$3"
    local gpu_enabled="$4"
    local suite_command="$5"

    echo -e "${BLUE}=== Testing Linux ${backend_name} lane (ubuntu-22.04) ===${NC}"

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker not found - cannot test Linux ${backend_name} lane${NC}"
        log_fail "linux-${backend_name}"
        return 1
    fi

    local docker_image="eshkol-${backend_name}-ci-test"
    local dockerfile="/tmp/Dockerfile.${backend_name}.ci-test"

    cat > "$dockerfile" << DOCKERFILE
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \\
    wget gnupg software-properties-common \\
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \\
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_MAJOR} main" >> /etc/apt/sources.list.d/llvm.list \\
    && apt-get update

RUN apt-get install -y \\
    cmake ninja-build \\
    llvm-${LLVM_MAJOR} llvm-${LLVM_MAJOR}-dev \\
    libreadline-dev pkg-config \\
    libssl-dev libncurses-dev \\
    git python3

WORKDIR /app
COPY . .

RUN cmake -S . -B ${build_dir} -G Ninja \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DESHKOL_REQUIRED_LLVM_MAJOR=${LLVM_MAJOR} \\
    -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-${LLVM_MAJOR} \\
    -DESHKOL_XLA_ENABLED=${xla_enabled} \\
    -DESHKOL_GPU_ENABLED=${gpu_enabled}

RUN cmake --build ${build_dir} --parallel
ENV BUILD_DIR=${build_dir}
RUN ${suite_command}
DOCKERFILE

    if docker build --platform linux/amd64 -t "$docker_image" -f "$dockerfile" .; then
        log_pass "linux-${backend_name}"
    else
        log_fail "linux-${backend_name}"
        rm -f "$dockerfile"
        return 1
    fi

    rm -f "$dockerfile"
}

test_workflow_matrix() {
    echo -e "${BLUE}=== Validating workflow matrix metadata ===${NC}"

    if ruby -e 'require "yaml"; YAML.load_file(".github/workflows/ci.yml"); YAML.load_file(".github/workflows/release.yml")'; then
        log_pass "workflow-yaml"
    else
        log_fail "workflow-yaml"
        return 1
    fi

    if command -v act &> /dev/null; then
        if act -W .github/workflows/ci.yml -l; then
            log_pass "workflow-act-list"
        else
            log_fail "workflow-act-list"
        fi
    else
        echo -e "${YELLOW}act not found - skipping workflow listing${NC}"
    fi
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

    # Check dependencies (matching CI: brew install llvm@${LLVM_MAJOR} cmake ninja readline)
    echo "Checking dependencies..."
    DEPS_OK=true
    for dep in llvm@"$LLVM_MAJOR" cmake ninja readline; do
        if brew list "$dep" &>/dev/null; then
            echo -e "  ${GREEN}[OK]${NC} $dep"
        else
            echo -e "  ${RED}[MISSING]${NC} $dep"
            DEPS_OK=false
        fi
    done

    if [ "$DEPS_OK" = false ]; then
        echo "Install missing dependencies: brew install llvm@${LLVM_MAJOR} cmake ninja readline"
        log_fail "macos-ci-deps"
        return 1
    fi
    log_pass "macos-ci-deps"

    # Set PATH exactly as CI does
    if [ "$ARCH" = "arm64" ]; then
        export PATH="/opt/homebrew/opt/llvm@${LLVM_MAJOR}/bin:$PATH"
    else
        export PATH="/usr/local/opt/llvm@${LLVM_MAJOR}/bin:$PATH"
    fi

    # Clean build
    BUILD_DIR="build-ci-test"
    rm -rf "$BUILD_DIR"

    # Configure - exactly as CI does
    echo "Configuring..."
    if cmake -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DESHKOL_REQUIRED_LLVM_MAJOR="${LLVM_MAJOR}" -DLLVM_CONFIG_EXECUTABLE="$(command -v llvm-config)"; then
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

    # Test a representative release lane: linux-x64-lite
    echo "Testing linux-x64-lite release package..."

    cat > /tmp/Dockerfile.release-test << DOCKERFILE
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Add the requested LLVM repository
RUN apt-get update && apt-get install -y wget gnupg software-properties-common \\
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \\
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_MAJOR} main" >> /etc/apt/sources.list.d/llvm.list \\
    && apt-get update

RUN apt-get install -y \\
    cmake ninja-build llvm-${LLVM_MAJOR} llvm-${LLVM_MAJOR}-dev libreadline-dev g++ file pkg-config libssl-dev libncurses-dev git python3

WORKDIR /app
COPY . .

# Configure and build
RUN cmake -S . -B build -G Ninja \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DESHKOL_REQUIRED_LLVM_MAJOR=${LLVM_MAJOR} \\
    -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-${LLVM_MAJOR} \\
    -DESHKOL_XLA_ENABLED=OFF \\
    -DESHKOL_GPU_ENABLED=OFF
RUN cmake --build build --parallel

# Run the lite test suite
ENV BUILD_DIR=build
RUN ./scripts/run_all_tests.sh

# Package (matching release workflow)
RUN archive_root="eshkol-v${VERSION}-linux-x64-lite" \\
    && pkg_dir="/tmp/\${archive_root}" \\
    && mkdir -p "\${pkg_dir}/bin" "\${pkg_dir}/lib/eshkol" "\${pkg_dir}/share/eshkol/lib" \\
    && cp build/eshkol-run "\${pkg_dir}/bin/" \\
    && cp build/eshkol-repl "\${pkg_dir}/bin/" \\
    && cp build/stdlib.o "\${pkg_dir}/lib/" \\
    && cp build/stdlib.o "\${pkg_dir}/lib/eshkol/" \\
    && cp lib/stdlib.esk "\${pkg_dir}/share/eshkol/" \\
    && [ ! -f lib/math.esk ] || cp lib/math.esk "\${pkg_dir}/share/eshkol/" \\
    && find lib -type f -name '*.esk' -print0 | while IFS= read -r -d '' source_file; do \\
         rel_path="\${source_file#lib/}"; \\
         dest_path="\${pkg_dir}/share/eshkol/lib/\${rel_path}"; \\
         mkdir -p "$(dirname "\${dest_path}")"; \\
         cp "\${source_file}" "\${dest_path}"; \\
       done \\
    && for doc in README.md LICENSE CHANGELOG.md; do [ ! -f "\${doc}" ] || cp "\${doc}" "\${pkg_dir}/"; done \\
    && tar -czvf "/app/\${archive_root}.tar.gz" -C /tmp "\${archive_root}"

# Verify package
RUN ls -la /app/*.tar.gz

CMD ["echo", "Release build complete!"]
DOCKERFILE

    if docker build --platform linux/amd64 -t "eshkol-release-test" -f /tmp/Dockerfile.release-test .; then
        log_pass "release-linux-build"

        # Extract and verify artifacts
        CONTAINER_ID=$(docker create eshkol-release-test)
        mkdir -p dist/release-test
        docker cp "$CONTAINER_ID:/app/eshkol-v${VERSION}-linux-x64-lite.tar.gz" dist/release-test/ 2>/dev/null || true
        docker rm "$CONTAINER_ID"

        if [ -f "dist/release-test/eshkol-v${VERSION}-linux-x64-lite.tar.gz" ]; then
            log_pass "release-linux-tarball"
        else
            log_fail "release-linux-tarball"
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
    linux-xla)
        test_linux_backend_ci "xla" "build-xla" "ON" "OFF" "./scripts/run_xla_tests.sh"
        ;;
    linux-cuda)
        test_linux_backend_ci "cuda" "build-cuda" "OFF" "ON" "./scripts/run_gpu_tests.sh"
        ;;
    matrix)
        test_workflow_matrix
        test_linux_ci
        test_linux_backend_ci "xla" "build-xla" "ON" "OFF" "./scripts/run_xla_tests.sh"
        test_linux_backend_ci "cuda" "build-cuda" "OFF" "ON" "./scripts/run_gpu_tests.sh"
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
        test_workflow_matrix
        test_linux_ci
        test_linux_backend_ci "xla" "build-xla" "ON" "OFF" "./scripts/run_xla_tests.sh"
        test_linux_backend_ci "cuda" "build-cuda" "OFF" "ON" "./scripts/run_gpu_tests.sh"
        test_macos_ci
        test_release
        test_homebrew
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [linux|linux-xla|linux-cuda|matrix|macos|release|homebrew|all]"
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
