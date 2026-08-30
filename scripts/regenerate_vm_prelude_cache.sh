#!/usr/bin/env bash
# regenerate_vm_prelude_cache.sh — SW-49: build lib/backend/vm_prelude_cache.h
# from the current committed VM sources, or verify (--check) that the
# committed copy already matches without touching it.
#
# BACKGROUND
#   lib/backend/vm_prelude_cache.c is a generator: compiled with
#   -DGENERATE_PRELUDE_CACHE it #includes lib/backend/eshkol_vm.c (the whole
#   bytecode-VM unity build) so it can reach the reader/compiler internals
#   (parse_sexp, compile_expr, emit_builtin_preamble, ...) that are `static`
#   to that one translation unit, compiles the canonical prelude
#   (ESHKOL_VM_PRELUDE_SOURCE, lib/backend/vm_prelude_source.h) to bytecode,
#   and prints the result as vm_prelude_cache.h C array literals on stdout.
#   emit_builtin_preamble() turns every entry of eshkol_vm.c's BUILTINS[]
#   dispatch table into one prelude local, so ANY addition, removal or
#   rename there invalidates the committed header — and nothing before
#   SW-49 ever re-ran this to notice. The committed cache drifted 28 (later
#   30) builtins stale; its only consumer is the WASM REPL
#   (lib/backend/vm_wasm_repl.c, the one site that defines
#   ESHKOL_VM_NO_DISASM as a macro), so no native ctest ever touched it.
#
#   The recipe this replaces was a hand-copied shell command recorded as a
#   comment in vm_prelude_source.h (cc ... vm_prelude_cache.c -o gen_prelude
#   build/libeshkol-runtime.a -lm -lc++ -framework ...). It is not
#   CMake-driven, so it silently drifts out of sync with whatever
#   eshkol-runtime/eshkol-static actually need to link on a given platform
#   or build configuration (BLAS backend, GPU backend, quantum/tensorcore
#   opt-ins). This script instead builds the generator through the real
#   CMake target (eshkol-vm-prelude-cache-gen, defined in CMakeLists.txt
#   right after eshkol-vm-standalone-test), which inherits eshkol-static's
#   actual, current link requirements instead of a frozen guess at them.
#
# USAGE
#   scripts/regenerate_vm_prelude_cache.sh                  # clean-tree regen
#   scripts/regenerate_vm_prelude_cache.sh --build-dir DIR  # reuse a build tree
#   scripts/regenerate_vm_prelude_cache.sh --gen-exe PATH   # generator already built
#   scripts/regenerate_vm_prelude_cache.sh --check [--gen-exe PATH]
#
#   --check       Do not modify lib/backend/vm_prelude_cache.h. Diff the
#                 generator's fresh output against the committed copy and
#                 exit nonzero (with the diff) if they differ. This is what
#                 the `vm_prelude_cache_is_current` ctest runs.
#   --build-dir   An existing (or to-be-created) CMake build directory to
#                 configure/build the generator in. Default: a fresh
#                 directory created with mktemp -d, removed on exit unless
#                 --keep-build-dir is also given.
#   --gen-exe     Path to an already-built eshkol-vm-prelude-cache-gen
#                 binary. Skips configure/build entirely.
#   --keep-build-dir  Do not remove a script-created build directory on exit.
#
# EXIT
#   0  regenerated (default mode) or already current (--check)
#   1  --check found a diff, or the generator produced no output
#   2  misuse / configure or build failure
#
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE_HEADER="$REPO_ROOT/lib/backend/vm_prelude_cache.h"

CHECK=0
GEN_EXE=""
BUILD_DIR=""
KEEP_BUILD_DIR=0
OWN_BUILD_DIR=0

usage() {
    sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK=1; shift ;;
        --gen-exe) GEN_EXE="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --keep-build-dir) KEEP_BUILD_DIR=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "FAIL: unrecognized argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

cleanup() {
    if [ "$OWN_BUILD_DIR" -eq 1 ] && [ "$KEEP_BUILD_DIR" -eq 0 ] && [ -n "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
    fi
}
trap cleanup EXIT

if [ -z "$GEN_EXE" ]; then
    if [ -z "$BUILD_DIR" ]; then
        mkdir -p "$REPO_ROOT/.scratch"
        BUILD_DIR="$(mktemp -d "$REPO_ROOT/.scratch/eshkol-vm-prelude-cache-gen.XXXXXX")" \
            || { echo "FAIL: mktemp -d failed" >&2; exit 2; }
        OWN_BUILD_DIR=1
    fi

    if [ ! -f "$BUILD_DIR/build.ninja" ] && [ ! -f "$BUILD_DIR/Makefile" ]; then
        echo "regenerate_vm_prelude_cache: configuring $BUILD_DIR" >&2
        cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -G Ninja \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DESHKOL_BUILD_AGENT_FFI=OFF \
            -DESHKOL_BUILD_EXAMPLES=OFF \
            -DESHKOL_BUILD_INTEGRATION_TESTS=OFF \
            >&2 || { echo "FAIL: cmake configure failed" >&2; exit 2; }
    fi

    echo "regenerate_vm_prelude_cache: building eshkol-vm-prelude-cache-gen" >&2
    cmake --build "$BUILD_DIR" --target eshkol-vm-prelude-cache-gen >&2 \
        || { echo "FAIL: build of eshkol-vm-prelude-cache-gen failed" >&2; exit 2; }

    GEN_EXE="$BUILD_DIR/eshkol-vm-prelude-cache-gen"
fi

if [ ! -x "$GEN_EXE" ]; then
    echo "FAIL: generator binary not found or not executable: $GEN_EXE" >&2
    exit 2
fi

FRESH_OUTPUT="$(dirname "$GEN_EXE")/vm_prelude_cache.generated.h"
"$GEN_EXE" > "$FRESH_OUTPUT" 2>/dev/null
if [ ! -s "$FRESH_OUTPUT" ]; then
    echo "FAIL: generator produced no output" >&2
    exit 1
fi

if [ "$CHECK" -eq 1 ]; then
    if [ ! -f "$CACHE_HEADER" ]; then
        echo "FAIL: committed cache is missing: $CACHE_HEADER" >&2
        exit 1
    fi
    if diff -q "$CACHE_HEADER" "$FRESH_OUTPUT" >/dev/null 2>&1; then
        echo "PASS: committed vm_prelude_cache.h matches the generator's current output"
        exit 0
    fi
    echo "FAIL: committed lib/backend/vm_prelude_cache.h is stale — it no longer matches" >&2
    echo "what the generator produces from the current VM sources. Regenerate it with:" >&2
    echo "  scripts/regenerate_vm_prelude_cache.sh" >&2
    echo "and commit the result." >&2
    echo "names in the generator's output but not the committed cache:" >&2
    comm -13 \
        <(grep -o '"[^"]*"' "$CACHE_HEADER" | sort -u) \
        <(grep -o '"[^"]*"' "$FRESH_OUTPUT" | sort -u) >&2
    exit 1
fi

cp "$FRESH_OUTPUT" "$CACHE_HEADER"
echo "regenerate_vm_prelude_cache: wrote $CACHE_HEADER" >&2
exit 0
