#!/usr/bin/env bash
# Build Eshkol from source on any mainstream Linux distribution.
#
# Eshkol requires LLVM 21 (CMake enforces the major via
# ESHKOL_REQUIRED_LLVM_MAJOR in cmake/LLVMToolchain.cmake). What differs
# between distributions is only how LLVM 21 and the build dependencies are
# obtained; the configure/build logic itself is distro-agnostic and lives in
# scripts/bootstrap_linux_build_host.sh. This script detects the distro
# family from /etc/os-release, provisions LLVM 21 plus dependencies for that
# family, and then delegates.
#
# Supported families:
#   debian   Debian, Ubuntu, Linux Mint, LMDE, Pop!_OS, elementary, Kali, ...
#            (apt + apt.llvm.org, with upstream base-codename resolution for
#            derivatives whose own codename apt.llvm.org does not serve)
#   arch     Arch, Manjaro, EndeavourOS, CachyOS, ... (pacman; rolling LLVM
#            must currently be major 21 or the CMake gate fails closed)
#   fedora   Fedora, RHEL, CentOS Stream, Rocky, Alma (dnf; prefers the
#            distro's default LLVM when it is major 21, otherwise the
#            versioned llvm21 compat packages)
#   suse     openSUSE Tumbleweed/Leap (zypper; versioned llvm21 packages)
#   alpine   Alpine (apk; musl libc — experimental, not a release platform)
#
# Usage:
#   scripts/build-linux.sh                 # deps + configure + build
#   scripts/build-linux.sh --ctest         # also run the test suite
#   scripts/build-linux.sh --no-install-deps ...
#
# Options other than dependency installation are forwarded verbatim to
# bootstrap_linux_build_host.sh (see --help there).
#
# Environment:
#   LLVM_VERSION           LLVM major to provision (default: 21). Must match
#                          ESHKOL_REQUIRED_LLVM_MAJOR if you override that.
#   ESHKOL_BASE_CODENAME   debian family only: override the resolved
#                          apt.llvm.org suite (jammy, noble, bookworm, ...).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_VERSION="${LLVM_VERSION:-21}"

if [ ! -r /etc/os-release ]; then
    echo "build-linux.sh: /etc/os-release not found; is this Linux?" >&2
    exit 4
fi
# shellcheck disable=SC1091
. /etc/os-release

INSTALL_DEPS=1
FORWARD_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-install-deps) INSTALL_DEPS=0 ;;
    esac
    FORWARD_ARGS+=("$arg")
done

sudo_cmd() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
        return
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        echo "build-linux.sh: not root and sudo is unavailable; rerun as root or install sudo" >&2
        exit 3
    fi
    if sudo -n true 2>/dev/null; then
        sudo -n "$@"
    else
        sudo "$@"
    fi
}

# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------

family=""
for candidate in "${ID:-}" ${ID_LIKE:-}; do
    case "$candidate" in
        debian|ubuntu|linuxmint) family=debian; break ;;
        arch) family=arch; break ;;
        fedora|rhel|centos) family=fedora; break ;;
        suse|opensuse|opensuse-tumbleweed|opensuse-leap|sles) family=suse; break ;;
        alpine) family=alpine; break ;;
    esac
done
# Mint container images sometimes carry upstream os-release; the Mint info
# file is definitive.
if [ -z "$family" ] && [ -r /etc/linuxmint/info ]; then
    family=debian
fi

if [ -z "$family" ]; then
    cat >&2 <<EOF
build-linux.sh: unrecognized distribution ID='${ID:-unknown}' ID_LIKE='${ID_LIKE:-}'.

Manual path: install LLVM ${LLVM_VERSION} (clang, llvm devel headers, lld),
plus: gcc/g++ or clang, cmake, ninja, git, pkg-config, python3, and devel
packages for openblas, pcre2, readline, sqlite3, openssl, libpng, libjpeg,
libwebp, zlib, libcurl, and ncurses. Then run:

  scripts/bootstrap_linux_build_host.sh --no-install-deps
EOF
    exit 4
fi

echo "build-linux.sh: ${PRETTY_NAME:-${ID:-linux}} -> family '${family}', LLVM ${LLVM_VERSION}"

# ---------------------------------------------------------------------------
# Fail-closed LLVM major verification (used by the non-debian families; on
# the debian family the versioned clang-NN packages pin the major, and the
# CMake gate in cmake/LLVMToolchain.cmake is the final authority everywhere).
# ---------------------------------------------------------------------------

verify_llvm_major() {
    local llvm_config="$1"
    local found_major
    if ! command -v "$llvm_config" >/dev/null 2>&1 && [ ! -x "$llvm_config" ]; then
        return 1
    fi
    found_major="$("$llvm_config" --version | cut -d. -f1)"
    if [ "$found_major" != "$LLVM_VERSION" ]; then
        cat >&2 <<EOF
build-linux.sh: found LLVM major ${found_major}, but Eshkol requires LLVM
${LLVM_VERSION} (cmake/LLVMToolchain.cmake fails closed on any other major).
If your distribution has moved past ${LLVM_VERSION}, install its versioned
llvm${LLVM_VERSION} compatibility packages, or build a pinned LLVM.
EOF
        return 2
    fi
}

# ---------------------------------------------------------------------------
# debian family: apt + apt.llvm.org with base-codename resolution
# ---------------------------------------------------------------------------

debian_base_codename() {
    if [ -n "${ESHKOL_BASE_CODENAME:-}" ]; then
        printf '%s\n' "$ESHKOL_BASE_CODENAME"
        return
    fi
    # Ubuntu derivatives (Mint, Pop, elementary, ...) carry the base here.
    if [ -n "${UBUNTU_CODENAME:-}" ]; then
        printf '%s\n' "$UBUNTU_CODENAME"
        return
    fi
    # Debian proper and close derivatives.
    if [ "${ID:-}" = "debian" ] && [ -n "${VERSION_CODENAME:-}" ]; then
        printf '%s\n' "$VERSION_CODENAME"
        return
    fi
    # Debian derivatives without a usable codename (LMDE, Kali, ...): map the
    # Debian release number.
    if [ -r /etc/debian_version ]; then
        case "$(cut -d. -f1 /etc/debian_version)" in
            12) printf 'bookworm\n'; return ;;
            13) printf 'trixie\n'; return ;;
        esac
    fi
    if [ -n "${VERSION_CODENAME:-}" ]; then
        printf '%s\n' "$VERSION_CODENAME"
        return
    fi
    cat >&2 <<EOF
build-linux.sh: could not resolve the apt.llvm.org suite for this host.
Set ESHKOL_BASE_CODENAME to the Ubuntu/Debian codename this distribution is
based on (for example jammy, noble, bookworm) and rerun.
EOF
    exit 4
}

llvm_apt_available() {
    apt-cache show "clang-${LLVM_VERSION}" >/dev/null 2>&1 &&
    apt-cache show "llvm-${LLVM_VERSION}-dev" >/dev/null 2>&1 &&
    apt-cache show "lld-${LLVM_VERSION}" >/dev/null 2>&1
}

install_debian() {
    local codename
    if ! llvm_apt_available; then
        codename="$(debian_base_codename)"
        echo "build-linux.sh: configuring apt.llvm.org for '${codename}'"
        if ! command -v wget >/dev/null 2>&1; then
            sudo_cmd apt-get update -o Acquire::Retries=5
            sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y wget ca-certificates
        fi
        wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key |
            sudo_cmd tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null
        printf 'deb http://apt.llvm.org/%s/ llvm-toolchain-%s-%s main\n' \
            "$codename" "$codename" "$LLVM_VERSION" |
            sudo_cmd tee /etc/apt/sources.list.d/eshkol-llvm.list >/dev/null
        sudo_cmd apt-get update -o Acquire::Retries=5
        if ! llvm_apt_available; then
            cat >&2 <<EOF
build-linux.sh: apt.llvm.org for '${codename}' does not provide
clang-${LLVM_VERSION}/llvm-${LLVM_VERSION}-dev/lld-${LLVM_VERSION} on this
architecture. Check https://apt.llvm.org for supported suites, or set
ESHKOL_BASE_CODENAME if the resolved suite is wrong.
EOF
            exit 6
        fi
    fi
    # bootstrap_linux_build_host.sh installs the full apt dependency list and
    # picks the versioned clang now that it is available.
}

# ---------------------------------------------------------------------------
# arch family: pacman (rolling; single current LLVM major)
# ---------------------------------------------------------------------------

install_arch() {
    sudo_cmd pacman -Sy --needed --noconfirm \
        base-devel cmake ninja git pkgconf python \
        openblas pcre2 readline sqlite openssl \
        libpng libjpeg-turbo libwebp zlib curl ncurses
    # Rolling release: use the current llvm when it is the required major,
    # otherwise the versioned legacy packages (llvm21/clang21/lld21) that
    # Arch keeps in [extra] across major transitions.
    local current_major=""
    current_major="$(pacman -Si llvm 2>/dev/null |
        awk '/^Version/{print $3}' | cut -d. -f1)"
    if [ "$current_major" = "$LLVM_VERSION" ]; then
        sudo_cmd pacman -S --needed --noconfirm llvm clang lld
        verify_llvm_major llvm-config
    else
        echo "build-linux.sh: rolling llvm is major ${current_major:-unknown}; installing versioned llvm${LLVM_VERSION} packages"
        sudo_cmd pacman -S --needed --noconfirm \
            "llvm${LLVM_VERSION}" "clang${LLVM_VERSION}" "lld${LLVM_VERSION}"
        verify_llvm_major "llvm-config-${LLVM_VERSION}" ||
            verify_llvm_major "/usr/lib/llvm${LLVM_VERSION}/bin/llvm-config"
    fi
}

# ---------------------------------------------------------------------------
# fedora family: dnf (default LLVM when it is major 21, else llvm21 compat)
# ---------------------------------------------------------------------------

install_fedora() {
    local common=(
        gcc gcc-c++ make cmake ninja-build git pkgconf-pkg-config python3
        openblas-devel pcre2-devel readline-devel sqlite-devel openssl-devel
        libpng-devel libjpeg-turbo-devel libwebp-devel zlib-devel
        libcurl-devel ncurses-devel
    )
    local default_major=""
    default_major="$(dnf repoquery --queryformat '%{VERSION}' llvm 2>/dev/null |
        sort -V | tail -1 | cut -d. -f1 || true)"
    if [ "$default_major" = "$LLVM_VERSION" ]; then
        sudo_cmd dnf install -y "${common[@]}" llvm llvm-devel clang lld
        verify_llvm_major llvm-config
    else
        sudo_cmd dnf install -y "${common[@]}" \
            "llvm${LLVM_VERSION}" "llvm${LLVM_VERSION}-devel" \
            "clang${LLVM_VERSION}" "lld${LLVM_VERSION}"
        # Fedora versioned toolchains live under /usr/lib64/llvmNN;
        # bootstrap_linux_build_host.sh searches that prefix.
        verify_llvm_major "/usr/lib64/llvm${LLVM_VERSION}/bin/llvm-config" ||
            verify_llvm_major "llvm-config-${LLVM_VERSION}"
    fi
}

# ---------------------------------------------------------------------------
# suse family: zypper (versioned llvm21 packages)
# ---------------------------------------------------------------------------

install_suse() {
    sudo_cmd zypper --non-interactive install \
        gcc gcc-c++ cmake ninja git pkg-config python3 \
        openblas-devel pcre2-devel readline-devel sqlite3-devel \
        libopenssl-devel libpng16-devel libjpeg8-devel libwebp-devel \
        zlib-devel libcurl-devel ncurses-devel \
        "llvm${LLVM_VERSION}" "llvm${LLVM_VERSION}-devel" \
        "clang${LLVM_VERSION}" "lld${LLVM_VERSION}"
    verify_llvm_major "llvm-config-${LLVM_VERSION}" ||
        verify_llvm_major "/usr/lib64/llvm${LLVM_VERSION}/bin/llvm-config" ||
        verify_llvm_major llvm-config
}

# ---------------------------------------------------------------------------
# alpine family: apk (musl libc; experimental)
# ---------------------------------------------------------------------------

install_alpine() {
    echo "build-linux.sh: NOTE: Alpine/musl is experimental and not a release platform" >&2
    sudo_cmd apk add --no-cache \
        build-base cmake ninja git pkgconf python3 \
        openblas-dev pcre2-dev readline-dev sqlite-dev openssl-dev \
        libpng-dev libjpeg-turbo-dev libwebp-dev zlib-dev curl-dev \
        ncurses-dev \
        "llvm${LLVM_VERSION}-dev" "clang${LLVM_VERSION}" lld
    verify_llvm_major "llvm-config-${LLVM_VERSION}" ||
        verify_llvm_major "/usr/lib/llvm${LLVM_VERSION}/bin/llvm-config"
}

# ---------------------------------------------------------------------------
# Provision, then delegate to the distro-agnostic configure/build
# ---------------------------------------------------------------------------

if [ "$INSTALL_DEPS" -eq 1 ]; then
    case "$family" in
        debian) install_debian ;;
        arch) install_arch ;;
        fedora) install_fedora ;;
        suse) install_suse ;;
        alpine) install_alpine ;;
    esac
fi

# bootstrap_linux_build_host.sh installs apt packages itself on the debian
# family (single authoritative list); other families installed above.
BOOTSTRAP_ARGS=(--llvm-version "$LLVM_VERSION")
if [ "$family" != "debian" ] && [ "$INSTALL_DEPS" -eq 1 ]; then
    BOOTSTRAP_ARGS+=(--no-install-deps)
fi

# bootstrap uses non-interactive sudo (CI convention); prime the timestamp on
# desktops with password sudo so its apt calls succeed.
if [ "$family" = "debian" ] && [ "$INSTALL_DEPS" -eq 1 ] && [ "$(id -u)" -ne 0 ]; then
    if ! sudo -n true 2>/dev/null; then
        sudo -v
    fi
fi

exec "$SCRIPT_DIR/bootstrap_linux_build_host.sh" "${BOOTSTRAP_ARGS[@]}" "${FORWARD_ARGS[@]}"
