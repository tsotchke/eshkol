#!/usr/bin/env bash
# build-wasm-repl.sh — build the browser REPL bundle,
# site/static/eshkol-vm.{js,wasm}, from lib/backend/vm_wasm_repl.c with the
# Emscripten SDK.
#
# This is the CANONICAL invocation.  It used to live as a copied-out `emcc`
# line in CONTRIBUTING.md, and that line drifted until it could no longer
# produce the artifact at all: it was missing `-I inc` (so the compile failed
# outright on eshkol/core/arity_contract.h), missing
# `-s ERROR_ON_UNDEFINED_SYMBOLS=0` (so the link failed), and — most damaging
# — it named only the VM's own translation unit, so the lib/core helpers the
# VM had come to depend on were replaced by aborting stubs and the shipped
# bundle died in the browser on `(char-alphabetic? #\a)`.  Prose cannot be
# gated; a script can, so the recipe lives here and CONTRIBUTING.md points at
# it.
#
# What this bundle is: the bytecode VM compiled to WebAssembly, exposing
# `repl_init` / `repl_reset` / `repl_eval` (the persistent browser REPL) and
# `run_program` (batch execution).  site/static/index.html loads it as
# `EshkolVM` and drives `repl_eval` for the REPL pane and for every runnable
# code block on the docs pages.  It is independent of the LLVM `--wasm` path
# that scripts/build-site.sh uses for eshkol-site.wasm.
#
# Usage:
#   scripts/build-wasm-repl.sh                  # -> site/static/eshkol-vm.js
#   OUT=/tmp/x/eshkol-vm.js scripts/build-wasm-repl.sh
#   BUILD_DIR=build-site scripts/build-wasm-repl.sh
#
# Requires: an activated emsdk (`. $EMSDK/emsdk_env.sh`) and a CONFIGURED
# CMake build dir, whose generated/ holds eshkol/build_config.h.  Configuring
# is enough — nothing from the native build is linked.
#
# Exit: 0 built, 1 build failed or an unexpected undefined symbol, 2 misuse.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
REPO_ROOT="$(pwd)"

# shellcheck source=./scripts/lib/wasm_vm_sources.sh
# shellcheck disable=SC1091
. "$REPO_ROOT/scripts/lib/wasm_vm_sources.sh"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) : ;;
    *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;;
esac
OUT="${OUT:-$REPO_ROOT/site/static/eshkol-vm.js}"

if ! command -v emcc >/dev/null 2>&1; then
    if [ -n "${EMSDK:-}" ] && [ -f "$EMSDK/emsdk_env.sh" ]; then
        # shellcheck disable=SC1091
        . "$EMSDK/emsdk_env.sh" >/dev/null 2>&1 || true
    fi
fi
if ! command -v emcc >/dev/null 2>&1; then
    echo "build-wasm-repl.sh: emcc not on PATH — activate emsdk first:" >&2
    echo "  . \$EMSDK/emsdk_env.sh" >&2
    exit 2
fi

if [ ! -f "$BUILD_DIR/generated/eshkol/build_config.h" ]; then
    echo "build-wasm-repl.sh: $BUILD_DIR/generated/eshkol/build_config.h not found." >&2
    echo "  lib/core/platform_runtime.cpp includes it; configure a build dir first:" >&2
    echo "  cmake -S . -B ${BUILD_DIR#$REPO_ROOT/}" >&2
    exit 2
fi

LOG="$(mktemp "${TMPDIR:-/var/tmp}/eshkol-wasm-repl-emcc.XXXXXX")"
trap 'rm -f "$LOG"' EXIT

echo "== building browser REPL bundle (emcc) -> $OUT"

# -ffp-contract=off mirrors the project-wide setting in CMakeLists.txt and the
#   execute-and-diff lane: native must not fuse `a*b + c` into a singly-rounded
#   multiply-add that WebAssembly cannot reproduce, and this side states the
#   rule rather than relying on the wasm backend's inability to contract.
# -s STACK_SIZE=8MB because Emscripten's default C stack is 64KB while the
#   VM's recursive parser/compiler expects the ~8MB an OS stack gives it
#   natively.  Under-provisioning does not fail cleanly — it overflows into
#   linear memory and corrupts results — so provision to parity.  Same value,
#   same reason, as scripts/run_wasm_differential.sh.
# -s ALLOW_MEMORY_GROWTH so a REPL session that outgrows the initial heap
#   grows instead of aborting the page.
if ! emcc -O2 -ffp-contract=off \
        -s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME='EshkolVM' \
        -s ERROR_ON_UNDEFINED_SYMBOLS=0 \
        -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
        -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=67108864 -s STACK_SIZE=8388608 \
        -DESHKOL_VM_WASM -DESHKOL_VM_NO_DISASM \
        -I "$REPO_ROOT/inc" -I "$BUILD_DIR/generated" -I "$REPO_ROOT/lib/backend" \
        "${ESHKOL_WASM_VM_SOURCES[@]}" \
        -o "$OUT" -lm 2> "$LOG"; then
    echo "build-wasm-repl.sh: emcc FAILED:" >&2
    tail -30 "$LOG" >&2
    exit 1
fi

# ── undefined-symbol allowlist gate ──────────────────────────────────────
# ERROR_ON_UNDEFINED_SYMBOLS=0 is what lets the documented leaf dependencies
# become aborting stubs.  It also means a NEW unlinked dependency ships as an
# abort with nothing but a warning to say so — the exact way the unicode
# helpers got into the bundle.  Diff the warning set against the allowlist so
# that can only ever happen on purpose.
unexpected=""
while IFS= read -r sym; do
    known=0
    for a in "${ESHKOL_WASM_ALLOWED_UNDEFINED[@]}"; do
        [ "$sym" = "$a" ] && { known=1; break; }
    done
    [ "$known" -eq 1 ] || unexpected="$unexpected $sym"
done < <(sed -n 's/^warning: undefined symbol: \([A-Za-z0-9_]*\).*/\1/p' "$LOG" | sort -u)

if [ -n "$unexpected" ]; then
    echo "build-wasm-repl.sh: UNEXPECTED undefined symbol(s):$unexpected" >&2
    echo "  Each one is an aborting stub in the shipped bundle: the browser REPL" >&2
    echo "  dies the first time a program reaches it.  Either add the defining" >&2
    echo "  translation unit to ESHKOL_WASM_VM_SOURCES in" >&2
    echo "  scripts/lib/wasm_vm_sources.sh, or — if it genuinely has no WASM" >&2
    echo "  implementation — add it to ESHKOL_WASM_ALLOWED_UNDEFINED there with" >&2
    echo "  a reason." >&2
    exit 1
fi

echo "   built: $OUT ($(wc -c < "${OUT%.js}.wasm" | tr -d ' ') bytes of wasm)"
echo "   aborting-stub leaf deps (documented):${ESHKOL_WASM_ALLOWED_UNDEFINED[*]/#/ }"
