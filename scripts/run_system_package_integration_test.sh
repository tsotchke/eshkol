#!/usr/bin/env bash
#
# System-package integration test (ADR-0010 gap A10 follow-on).
#
# Verifies the documented downstream "SYSTEM" CMake integration path end to
# end, against an INSTALLED Eshkol prefix rather than an in-tree build: a
# from-scratch consumer project (tests/integration/system_package/) adds the
# packaged cmake/ directory to CMAKE_MODULE_PATH, calls find_package(Eshkol),
# includes EshkolCompile.cmake, compiles a bare-top-level-expression .esk
# program via eshkol_compile_executable(), links it, and runs it.
#
# This is the class of gap a 2026-08 consumer audit found broken: a packaged
# release's find_package(Eshkol) path had no canonical Find module to
# discover the real library name, and a naive guess at "the Eshkol library"
# could resolve to the compiler/tool aggregate (eshkol-static) instead of the
# lean runtime a compiled program actually needs (eshkol-runtime) — see
# cmake/FindEshkol.cmake's header comment for the full contract this pins.
#
# Usage:
#   scripts/run_system_package_integration_test.sh --prefix /path/to/installed/eshkol
#   scripts/run_system_package_integration_test.sh --build-dir build   # stage from a local build
#
# --prefix points at an ALREADY-INSTALLED Eshkol (e.g. `brew --prefix eshkol`,
# or a GitHub release tarball extracted somewhere) that already carries
# share/eshkol/cmake/{FindEshkol,EshkolCompile}.cmake.
#
# --build-dir stages a homebrew-shaped prefix from a local CMake build
# directory (mirroring packaging/homebrew/eshkol.rb's install list) into a
# private scratch directory, for CI lanes that build from source rather than
# installing a package. This script's staging list and the homebrew formula
# are two independent copies of the same layout on purpose (same reason
# .icc/package-manifest.yaml exists): if the formula's list and this script's
# list drift, that is exactly the kind of divergence this test is meant to
# surface, not paper over by sourcing one from the other.

set -euo pipefail

ESHKOL_TEST_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test_isolation.sh"
if [ ! -r "$ESHKOL_TEST_LIB" ]; then
    echo "FATAL: cannot read $ESHKOL_TEST_LIB" >&2
    exit 2
fi
source "$ESHKOL_TEST_LIB"
eshkol_test_isolation_init "system_package_integration"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX=""
BUILD_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$PREFIX" ] && [ -z "$BUILD_DIR" ]; then
    echo "usage: $0 (--prefix DIR | --build-dir DIR)" >&2
    exit 2
fi

STAGE_DIR="$ESHKOL_TEST_TMPDIR/staged-prefix"

if [ -z "$PREFIX" ]; then
    echo "Staging a homebrew-shaped prefix from build dir: $BUILD_DIR"
    PREFIX="$STAGE_DIR"
    mkdir -p "$PREFIX/bin" "$PREFIX/lib/eshkol" "$PREFIX/share/eshkol/lib"

    cp "$BUILD_DIR/eshkol-run" "$PREFIX/bin/"
    [ -f "$BUILD_DIR/eshkol-repl" ] && cp "$BUILD_DIR/eshkol-repl" "$PREFIX/bin/"

    for f in stdlib.o stdlib.bc libeshkol-runtime.a libeshkol-static.a; do
        [ -f "$BUILD_DIR/$f" ] && cp "$BUILD_DIR/$f" "$PREFIX/lib/eshkol/"
    done
    [ -f "$BUILD_DIR/libeshkol-agent-ffi.a" ] && cp "$BUILD_DIR/libeshkol-agent-ffi.a" "$PREFIX/lib/eshkol/"
    for f in "$BUILD_DIR"/eshkol-agent-*.a; do
        [ -f "$f" ] && cp "$f" "$PREFIX/lib/eshkol/"
    done
    for f in stdlib.o stdlib.bc libeshkol-runtime.a libeshkol-static.a; do
        [ -f "$PREFIX/lib/eshkol/$f" ] && ln -sf "../lib/eshkol/$f" "$PREFIX/lib/$f"
    done

    cp "$REPO_ROOT/lib/stdlib.esk" "$PREFIX/share/eshkol/lib/"
    [ -f "$REPO_ROOT/lib/math.esk" ] && cp "$REPO_ROOT/lib/math.esk" "$PREFIX/share/eshkol/lib/"
    [ -f "$REPO_ROOT/lib/tensorcore.esk" ] && cp "$REPO_ROOT/lib/tensorcore.esk" "$PREFIX/share/eshkol/lib/"
    for mod in core math signal ml random web tensor quantum; do
        [ -d "$REPO_ROOT/lib/$mod" ] && cp -R "$REPO_ROOT/lib/$mod" "$PREFIX/share/eshkol/lib/"
    done
    mkdir -p "$PREFIX/share/eshkol/lib/agent"
    for f in "$REPO_ROOT"/lib/agent/*.esk; do
        [ -f "$f" ] && cp "$f" "$PREFIX/share/eshkol/lib/agent/"
    done
fi

# The cmake integration modules: an already-installed --prefix from a package
# built with this same change already carries them; a from-build-dir stage
# never has, so always (re)install this repo's copies into the prefix we are
# about to test against. This keeps the test meaningful even before a given
# packaging pipeline has picked up the change that ships them.
mkdir -p "$PREFIX/share/eshkol/cmake"
cp "$REPO_ROOT/cmake/FindEshkol.cmake" "$PREFIX/share/eshkol/cmake/"
cp "$REPO_ROOT/cmake/EshkolCompile.cmake" "$PREFIX/share/eshkol/cmake/"

echo "Testing find_package(Eshkol) discovery + compile + link + run against: $PREFIX"

CONSUMER_BUILD="$ESHKOL_TEST_TMPDIR/consumer-build"
mkdir -p "$CONSUMER_BUILD"

cmake -S "$REPO_ROOT/tests/integration/system_package" -B "$CONSUMER_BUILD" -G Ninja \
    -DESHKOL_CMAKE_DIR="$PREFIX/share/eshkol/cmake" \
    -DEshkol_ROOT="$PREFIX"

cmake --build "$CONSUMER_BUILD"

OUT="$("$CONSUMER_BUILD/system_package_hello")"
if [ "$OUT" != "system-package-integration-ok" ]; then
    echo "FAIL: expected 'system-package-integration-ok', got: $OUT" >&2
    exit 1
fi

echo "PASS: system-package integration test (prefix: $PREFIX)"
