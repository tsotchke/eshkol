#!/usr/bin/env bash
# Prepare and optionally build Eshkol on a Debian/Ubuntu Linux host.
#
# Defaults are intentionally CI-friendly:
#   - install apt build dependencies when passwordless sudo is available
#   - configure a Release build in ./build
#   - build with all available CPUs
#
# Set LLVM_VERSION=21 to prefer LLVM/Clang 21 packages. If those packages are
# unavailable, the script falls back to distro-default clang/llvm packages.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
LLVM_VERSION="${LLVM_VERSION:-21}"
INSTALL_DEPS=1
CONFIGURE=1
BUILD=1
RUN_CTEST=0

usage() {
    cat <<'USAGE'
Usage: scripts/bootstrap_linux_build_host.sh [options]

Options:
  --build-dir DIR       Build directory relative to the repo (default: build)
  --build-type TYPE     CMake build type (default: Release)
  --llvm-version N      Preferred LLVM/Clang major version (default: 21)
  --no-install-deps     Skip apt dependency installation
  --no-configure        Skip CMake configure
  --no-build            Skip CMake build
  --ctest               Run CTest after build
  -h, --help            Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --llvm-version)
            LLVM_VERSION="$2"
            shift 2
            ;;
        --no-install-deps)
            INSTALL_DEPS=0
            shift
            ;;
        --no-configure)
            CONFIGURE=0
            shift
            ;;
        --no-build)
            BUILD=0
            shift
            ;;
        --ctest)
            RUN_CTEST=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "$PROJECT_DIR"

sudo_cmd() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
        return
    fi

    if ! sudo -n true 2>/dev/null; then
        cat >&2 <<'SUDO_HELP'
passwordless sudo is not available.

To enable this build host, run once on the host:

  sudo visudo -f /etc/sudoers.d/90-tyr-build

and add:

  tyr ALL=(ALL) NOPASSWD:ALL

Then verify with:

  sudo -n true && echo sudo-ok
SUDO_HELP
        exit 3
    fi

    sudo -n "$@"
}

apt_has_package() {
    apt-cache show "$1" >/dev/null 2>&1
}

install_deps() {
    if ! test -r /etc/debian_version; then
        echo "dependency installation currently supports Debian/Ubuntu hosts only" >&2
        exit 4
    fi

    sudo_cmd apt-get update

    local packages=(
        build-essential
        ca-certificates
        cmake
        git
        libcurl4-openssl-dev
        libncurses-dev
        libopenblas-dev
        libpcre2-dev
        libreadline-dev
        libsqlite3-dev
        libssl-dev
        ninja-build
        pkg-config
        python3
        zlib1g-dev
    )

    if apt_has_package "clang-${LLVM_VERSION}" &&
       apt_has_package "llvm-${LLVM_VERSION}-dev" &&
       apt_has_package "lld-${LLVM_VERSION}"; then
        packages+=("clang-${LLVM_VERSION}" "llvm-${LLVM_VERSION}-dev" "lld-${LLVM_VERSION}")
    else
        packages+=(clang llvm-dev lld)
    fi

    sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
}

first_executable() {
    local candidate
    for candidate in "$@"; do
        if test -n "$candidate" && test -x "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
        if type -P "$candidate" >/dev/null 2>&1; then
            type -P "$candidate"
            return 0
        fi
    done
    return 1
}

configure_build() {
    local llvm_config
    local cc
    local cxx
    local llvm_dir
    local cmake_args

    if ! llvm_config="$(first_executable \
        "llvm-config-${LLVM_VERSION}" \
        "/usr/lib/llvm-${LLVM_VERSION}/bin/llvm-config" \
        llvm-config)"; then
        echo "unable to find llvm-config; rerun with --install-deps or install llvm-dev" >&2
        exit 5
    fi
    if ! cc="$(first_executable "clang-${LLVM_VERSION}" clang cc)"; then
        echo "unable to find a C compiler; rerun with --install-deps or install clang" >&2
        exit 5
    fi
    if ! cxx="$(first_executable "clang++-${LLVM_VERSION}" clang++ c++)"; then
        echo "unable to find a C++ compiler; rerun with --install-deps or install clang++" >&2
        exit 5
    fi

    llvm_dir="$("$llvm_config" --cmakedir 2>/dev/null || true)"
    cmake_args=(
        -S .
        -B "$BUILD_DIR"
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DCMAKE_C_COMPILER="$cc"
        -DCMAKE_CXX_COMPILER="$cxx"
    )

    if [ -n "$llvm_dir" ] && test -d "$llvm_dir"; then
        cmake_args+=("-DLLVM_DIR=$llvm_dir")
    else
        cmake_args+=("-DLLVM_CONFIG_EXECUTABLE=$llvm_config")
    fi

    if type -P ninja >/dev/null 2>&1; then
        cmake_args=(-G Ninja "${cmake_args[@]}")
    fi

    cmake "${cmake_args[@]}"
}

if [ "$INSTALL_DEPS" -eq 1 ]; then
    install_deps
fi

if [ "$CONFIGURE" -eq 1 ]; then
    configure_build
fi

if [ "$BUILD" -eq 1 ]; then
    cmake --build "$BUILD_DIR" --parallel "$(getconf _NPROCESSORS_ONLN)"
fi

if [ "$RUN_CTEST" -eq 1 ]; then
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi
