#!/usr/bin/env bash
# build-site.sh — Build the Eshkol website bundle for GitHub Pages deploy.
#
# Produces (relative to the repo root):
#   site/static/eshkol-site.wasm    — site/src/main.esk compiled to WebAssembly
#   site/static/content/*.html      — rendered fragments of root/docs markdown
#                                     (only if `pandoc` is available)
#
# What this script DOES NOT build:
#   site/static/eshkol-vm.{js,wasm} — these are produced by the Emscripten SDK
#       (`emcc`) from lib/backend/vm_wasm_repl.c.  They are the bytecode-VM
#       browser REPL bundle, independent of the LLVM `--wasm` path used here,
#       and changing the host compiler does not change them.  When the VM C
#       source or the prelude cache changes, rebuild manually per the
#       instructions in CONTRIBUTING.md (Building the Website section).
#
# Usage:
#   scripts/build-site.sh                 # use ./build (or $BUILD_DIR) and
#                                         # build eshkol-run if it's missing
#   BUILD_DIR=build-site scripts/build-site.sh
#   ESHKOL_RUN=/path/to/eshkol-run scripts/build-site.sh   # use a prebuilt binary
#
# This script is also the canonical build invocation used by the GitHub
# Pages workflow (.github/workflows/pages.yml).  Keep them in lock-step.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
ESHKOL_RUN="${ESHKOL_RUN:-${BUILD_DIR}/eshkol-run}"

SITE_SRC="site/src/main.esk"
SITE_OUT="site/static/eshkol-site.wasm"

if [[ ! -f "$SITE_SRC" ]]; then
    echo "build-site.sh: $SITE_SRC not found" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 1. Ensure eshkol-run exists.  If the caller hasn't built it, configure +
#    build a minimal (lite, tests off) tree just for the compiler binary.
# ----------------------------------------------------------------------------

if [[ ! -x "$ESHKOL_RUN" ]]; then
    echo "build-site.sh: $ESHKOL_RUN not found, configuring + building it"

    if command -v ninja >/dev/null 2>&1; then
        CMAKE_GEN=(-G Ninja)
    else
        CMAKE_GEN=()
    fi

    # Match the lite lane of ci.yml: no XLA, no GPU, no tests.  Tests are
    # off because the workflow only needs the eshkol-run binary; building
    # the test tree drags in extra dependencies and minutes.
    cmake -S . -B "$BUILD_DIR" "${CMAKE_GEN[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DESHKOL_BUILD_TESTS=OFF \
        -DESHKOL_XLA_ENABLED=OFF \
        -DESHKOL_GPU_ENABLED=OFF \
        "$@"

    cmake --build "$BUILD_DIR" --target eshkol-run --parallel
fi

if [[ ! -x "$ESHKOL_RUN" ]]; then
    echo "build-site.sh: $ESHKOL_RUN still missing after build" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 2. Compile site/src/main.esk to WebAssembly.
#
#    The `--wasm` (a.k.a. `-w`) flag in exe/eshkol-run.cpp sets the LLVM
#    target to wasm32-unknown-unknown and routes through
#    eshkol_compile_llvm_ir_to_wasm_file(), which writes the .wasm without
#    invoking the host linker.  No stdlib.o is required because main.esk
#    has no `(require ...)` and uses only `extern` declarations resolved
#    by the JS glue (site/static/eshkol-runtime.js).
# ----------------------------------------------------------------------------

echo "build-site.sh: compiling $SITE_SRC -> $SITE_OUT"
mkdir -p "$(dirname "$SITE_OUT")"
"$ESHKOL_RUN" --wasm "$SITE_SRC" -o "$SITE_OUT"

if [[ ! -s "$SITE_OUT" ]]; then
    echo "build-site.sh: $SITE_OUT was not produced" >&2
    exit 1
fi

ls -la "$SITE_OUT"

# ----------------------------------------------------------------------------
# 3. Regenerate the markdown-rendered content fragments.  These are fetched
#    by the WASM app at runtime via `web-load-content`.  We reuse the
#    existing build-site-content.sh so the local-dev and CI paths produce
#    identical output.  Skipped (with a warning) if pandoc is not on PATH.
# ----------------------------------------------------------------------------

if command -v pandoc >/dev/null 2>&1; then
    echo "build-site.sh: regenerating site content via build-site-content.sh"
    "$REPO_ROOT/scripts/build-site-content.sh"
else
    echo "build-site.sh: pandoc not found, skipping site content regeneration" >&2
    echo "                 (existing committed site/static/content/*.html will be used)" >&2
fi

echo "build-site.sh: done"
