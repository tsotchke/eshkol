#!/usr/bin/env bash
# tests/memory/arena_block_pool_test.sh — the large-block pool is invisible to
# results and to heap accounting.
#
# Region teardown hands blocks of >= 1 MiB to a pool (ESHKOL_ARENA_BLOCK_POOL_MB,
# default 1024) instead of back to the OS. The pool must change nothing a program
# can observe:
#   1. results are identical with the pool on, off (=0), and under
#      ESHKOL_ARENA_POISON=1 (which disables it);
#   2. heap accounting stays balanced: a pooled block is deallocated on entry to
#      the pool and allocated again on reuse, so a loop whose working set is a
#      few blocks runs to completion under ESHKOL_MAX_HEAP=256M even though it
#      allocates 4.8 GiB in total. If pooled blocks were still charged, or
#      charged twice on reuse, the tracked heap would grow every iteration and
#      the fail-closed limit would stop the run.
#
# Usage: tests/memory/arena_block_pool_test.sh   (BUILD_DIR selects the build, default: build)
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in /*) B="$BUILD_DIR" ;; *) B="$REPO_ROOT/$BUILD_DIR" ;; esac
ESHKOL_RUN="${ESHKOL_RUN:-$B/eshkol-run}"
[ -x "$ESHKOL_RUN" ] || { echo "arena_block_pool_test.sh: $ESHKOL_RUN not found" >&2; exit 2; }
SRC="$REPO_ROOT/tests/memory/arena_block_pool_test.esk"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
if ! "$ESHKOL_RUN" "$SRC" -L"$B" -o "$WORK/prog" > "$WORK/compile.log" 2>&1; then
    echo "FAIL: arena_block_pool_test: compile failed"; cat "$WORK/compile.log"; exit 1
fi

fail=0
check() {  # label env...
    local label=$1; shift
    if env "$@" timeout 600 "$WORK/prog" > "$WORK/$label.out" 2>&1 && grep -qx "PASS" "$WORK/$label.out"; then
        echo "  ok: $label"
    else
        echo "  FAIL: $label"; sed 's/^/    /' "$WORK/$label.out" | head -20; fail=1
    fi
}
check pool-default    ESHKOL_ARENA_BLOCK_POOL_MB=1024
check pool-off        ESHKOL_ARENA_BLOCK_POOL_MB=0
check poison          ESHKOL_ARENA_POISON=1
check heap-limit-pool ESHKOL_MAX_HEAP=256M ESHKOL_ARENA_BLOCK_POOL_MB=1024
check heap-limit-off  ESHKOL_MAX_HEAP=256M ESHKOL_ARENA_BLOCK_POOL_MB=0

if [ $fail -ne 0 ]; then echo "FAIL: arena_block_pool_test"; exit 1; fi
echo "PASS: arena_block_pool_test"
