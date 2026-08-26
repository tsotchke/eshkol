#!/usr/bin/env bash
# bench/large_single_file_compile_bench.sh — continuous measurement of the
# large-single-file AOT compile-time shape a 2026-08-26 downstream-consumer
# audit found: a single hand-written file with many top-level `define`s
# (as opposed to the same code spread across many separately-compiled
# files) compiled disproportionately slower than a multi-file bundle of
# comparable total size — 4x+ slower for a file the audit measured, and it
# never finished within that audit's time window. Eshkol's own tests/bench
# corpus had nothing of this shape before this file, so a regression here
# would ship undetected.
#
# This does NOT attempt to fix or optimize the compile-time cost — that is
# a v1.3.6+ item. It measures it, continuously, against a generous ceiling,
# using a SYNTHETIC fixture (bench/generate_large_single_file.py) so the
# corpus never depends on, or leaks anything about, the consumer whose
# usage motivated it.
#
# Usage:
#   bench/large_single_file_compile_bench.sh <build_dir> [defines] [ceiling_s]
#
#   build_dir   an already-built Eshkol tree (needs build_dir/eshkol-run)
#   defines     number of top-level defines to generate (default: 1600 —
#               calibrated to land in the single-digit-minutes range on a
#               typical CI runner, well under the ceiling, while still deep
#               enough into the super-linear region to be regression-
#               sensitive; see bench/generate_large_single_file.py)
#   ceiling_s   fail if the compile exceeds this many seconds (default: 900
#               = 15 minutes, a deliberately generous ceiling per the
#               finding this measures rather than fixes)
#
# Exits 0 and prints a one-line JSON result on success (compile finished
# under the ceiling). Exits 1 if the compile fails outright or exceeds the
# ceiling. Also captures a coarse eshkol-run phase-timing breakdown
# (ESHKOL_PHASE_TIME=1) alongside the timed run, written to
# <work_dir>/phase_timing.log, so a regression investigation starts with
# "which phase grew" rather than a bare wall-clock number.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

BUILD_DIR="${1:?usage: large_single_file_compile_bench.sh <build_dir> [defines] [ceiling_s]}"
DEFINES="${2:-1600}"
CEILING_S="${3:-900}"

case "$BUILD_DIR" in
    /*) ;;
    *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;;
esac
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
[ -x "$ESHKOL_RUN" ] || bench_die "large_single_file_compile_bench: $ESHKOL_RUN not found or not executable"

bench_pin_single_thread
WORK_DIR="$(bench_make_work_dir "large-single-file")"
trap 'bench_cleanup_work_dir "$WORK_DIR"' EXIT

SRC="$WORK_DIR/generated_large_single_file.esk"
OBJ="$WORK_DIR/generated_large_single_file.o"
PHASE_LOG="$WORK_DIR/phase_timing.log"

bench_log "large_single_file_compile_bench: generating defines=$DEFINES -> $SRC"
python3 "$SCRIPT_DIR/generate_large_single_file.py" --defines "$DEFINES" --out "$SRC" \
    || bench_die "generator failed"
bench_disk_cap_check "$WORK_DIR"

bench_log "large_single_file_compile_bench: compiling (ceiling ${CEILING_S}s)..."
# Whole-second resolution (bash's builtin $SECONDS) rather than `date +%s%N`:
# %N is a GNU-date extension BSD/macOS date does not support, and this run
# is measured in tens of seconds to minutes, where sub-second precision buys
# nothing.
SECONDS=0

# ESHKOL_PHASE_TIME=1 makes eshkol-run print [PHASE] lines to stderr (see
# exe/eshkol-run.cpp's ESH-0103 phase timing) on the SAME timed/gated
# compile — one compile serves both the pass/fail measurement and the
# diagnostic breakdown, rather than paying for the pathology twice.
# NOTE: eshkol-run does not exit promptly on SIGTERM while deep in codegen
# (measured directly: SIGTERM delivered mid-compile left it running), so
# every path below sends SIGKILL on timeout, never TERM — a ceiling that can
# be ignored is not a ceiling. GNU `timeout`'s default signal is TERM, hence
# `--signal=KILL` rather than a bare `timeout Ns`.
COMPILE_EXIT=0
if command -v timeout >/dev/null 2>&1; then
    ESHKOL_PHASE_TIME=1 timeout --signal=KILL "${CEILING_S}s" "$ESHKOL_RUN" --emit-object -o "$OBJ" "$SRC" \
        > "$PHASE_LOG" 2>&1 || COMPILE_EXIT=$?
elif command -v gtimeout >/dev/null 2>&1; then
    ESHKOL_PHASE_TIME=1 gtimeout --signal=KILL "${CEILING_S}s" "$ESHKOL_RUN" --emit-object -o "$OBJ" "$SRC" \
        > "$PHASE_LOG" 2>&1 || COMPILE_EXIT=$?
else
    # No timeout(1) available (e.g. plain macOS without coreutils): fall back
    # to a background job + watchdog. Correctness over elegance — this path
    # only exists so the gate still fails closed rather than hanging forever.
    ESHKOL_PHASE_TIME=1 "$ESHKOL_RUN" --emit-object -o "$OBJ" "$SRC" > "$PHASE_LOG" 2>&1 &
    COMPILE_PID=$!
    (sleep "$CEILING_S" && kill -KILL "$COMPILE_PID" 2>/dev/null) &
    WATCHDOG_PID=$!
    wait "$COMPILE_PID"
    COMPILE_EXIT=$?
    kill "$WATCHDOG_PID" 2>/dev/null
    wait "$WATCHDOG_PID" 2>/dev/null
fi

ELAPSED_S="$SECONDS"

if [ "$COMPILE_EXIT" -eq 124 ] || [ "$COMPILE_EXIT" -eq 137 ]; then
    bench_log "large_single_file_compile_bench: FAIL — did not finish within ${CEILING_S}s ceiling (defines=$DEFINES)"
    tail -n 40 "$PHASE_LOG" >&2 2>/dev/null
    echo "{\"status\":\"fail\",\"reason\":\"ceiling_exceeded\",\"defines\":$DEFINES,\"ceiling_s\":$CEILING_S}"
    exit 1
fi
if [ "$COMPILE_EXIT" -ne 0 ] || [ ! -s "$OBJ" ]; then
    bench_log "large_single_file_compile_bench: FAIL — compile exited $COMPILE_EXIT or produced no object (defines=$DEFINES)"
    tail -n 40 "$PHASE_LOG" >&2 2>/dev/null
    echo "{\"status\":\"fail\",\"reason\":\"compile_error\",\"exit\":$COMPILE_EXIT,\"defines\":$DEFINES,\"ceiling_s\":$CEILING_S}"
    exit 1
fi

bench_log "large_single_file_compile_bench: PASS — ${ELAPSED_S}s (ceiling ${CEILING_S}s, defines=$DEFINES)"
bench_log "large_single_file_compile_bench: compile + phase-timing log at $PHASE_LOG"
echo "{\"status\":\"pass\",\"elapsed_s\":$ELAPSED_S,\"ceiling_s\":$CEILING_S,\"defines\":$DEFINES}"
exit 0
