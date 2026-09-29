#!/usr/bin/env bash
# tests/xla/xla_region_reclaim_test.sh — XLA results are reclaimed by with-region.
#
# In an ESHKOL_XLA_ENABLED build, tensor arithmetic on >= ESHKOL_XLA_THRESHOLD
# elements is emitted as a call into the XLA runtime (which dispatches to the
# GPU when one is present). Those call sites used to pass the raw
# __global_arena slot as the result's arena; with-region no longer redirects
# that slot (it sets the thread-local allocation domain eshkol_current_arena()
# returns), so every such result landed in the process arena and was never
# reclaimed. xla_region_reclaim_test.esk makes 800 MiB of region-scoped
# temporaries: fixed, the process arena ends near 4 MiB; broken, it holds them
# all. The gate reads ESHKOL_ARENA_REPORT (deterministic to the byte, unlike
# peak RSS) and fails above a 256 MiB ceiling.
#
# In a build without XLA the same tensors take the inline CPU path, which
# already honoured regions, so the gate passes there too — it is a regression
# test for XLA builds and a no-op check elsewhere.
#
# Usage: tests/xla/xla_region_reclaim_test.sh   (BUILD_DIR selects the build, default: build)
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in /*) B="$BUILD_DIR" ;; *) B="$REPO_ROOT/$BUILD_DIR" ;; esac
ESHKOL_RUN="${ESHKOL_RUN:-$B/eshkol-run}"
[ -x "$ESHKOL_RUN" ] || { echo "xla_region_reclaim_test.sh: $ESHKOL_RUN not found" >&2; exit 2; }
SRC="$REPO_ROOT/tests/xla/xla_region_reclaim_test.esk"
CEILING=$((256 * 1024 * 1024))

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
# Compile ahead of time so the report comes from the program alone, and so no
# run-cache entry from another configuration can stand in for this one.
if ! ESHKOL_XLA_THRESHOLD=100000 "$ESHKOL_RUN" "$SRC" -L"$B" -o "$WORK/prog" > "$WORK/compile.log" 2>&1; then
    echo "FAIL: xla_region_reclaim_test: compile failed"; cat "$WORK/compile.log"; exit 1
fi
ESHKOL_ARENA_REPORT=1 timeout 300 "$WORK/prog" > "$WORK/out.log" 2>&1
status=$?
if [ $status -ne 0 ] || ! grep -qx "PASS" "$WORK/out.log"; then
    echo "FAIL: xla_region_reclaim_test: program exited $status or computed a wrong result"; cat "$WORK/out.log"; exit 1
fi
bytes=$(sed -n 's/.*global_total_allocated_bytes=\([0-9]*\).*/\1/p' "$WORK/out.log" | sort -n | tail -1)
if [ -z "$bytes" ]; then
    echo "FAIL: xla_region_reclaim_test: no ESHKOL_ARENA_REPORT line"; cat "$WORK/out.log"; exit 1
fi
if [ "$bytes" -gt "$CEILING" ]; then
    echo "FAIL: xla_region_reclaim_test: process arena holds $bytes bytes after 800 MiB of region-scoped temporaries (ceiling $CEILING)"
    exit 1
fi
echo "PASS: xla_region_reclaim_test: process arena $bytes bytes (ceiling $CEILING)"
