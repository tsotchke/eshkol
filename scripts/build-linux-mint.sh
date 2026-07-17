#!/usr/bin/env bash
# Build Eshkol on Linux Mint (Ubuntu-based 20/21/22 series) and LMDE.
#
# Mint is handled by the universal Linux build script: Eshkol requires
# LLVM 21, which no Mint release ships, and the standard apt.llvm.org setup
# fails on Mint because `lsb_release -cs` reports Mint's own codename
# (virginia, wilma, xia, ...) that apt.llvm.org does not serve. The
# universal script resolves the upstream base codename (UBUNTU_CODENAME
# from /etc/os-release, or the Debian release for LMDE), configures
# apt.llvm.org fail-closed, and delegates configure/build to
# scripts/bootstrap_linux_build_host.sh. This entry point exists so Mint
# users find a script with their distro's name on it.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build-linux.sh" "$@"
