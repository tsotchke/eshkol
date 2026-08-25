#!/usr/bin/env bash
# bench/lib/common.sh — shared helpers for the public benchmark suite
# (bench/run_public_benchmarks.sh and bench/axes/*.sh).
#
# Conventions this file establishes, followed by every axis script:
#   * every axis writes ONE JSON-Lines-free JSON object per invocation to
#     $BENCH_AXIS_JSON, plus a markdown fragment to $BENCH_AXIS_MD;
#   * every axis gets its own scratch directory under $BENCH_WORK_ROOT,
#     which is disk-capped (bench_disk_cap_check) and removed on exit
#     unless $BENCH_KEEP_WORK=1;
#   * timing noise controls follow docs/design/adr/0007-performance-pgo-wpo.md
#     ("Result protocol" / "Noise controls"): raw samples are kept, warmup is
#     explicit, iteration counts are chosen so each sample takes a
#     floor amount of wall time, and thread/JIT-cache state is pinned and
#     recorded rather than left ambient.
#
# This file only defines functions/vars; it does not run anything itself.

if [ -n "${ESHKOL_BENCH_COMMON_SH_LOADED:-}" ]; then
    return 0 2>/dev/null || true
fi
ESHKOL_BENCH_COMMON_SH_LOADED=1

bench_log() { printf '%s\n' "$*" >&2; }
bench_die() { bench_log "FATAL: $*"; exit 1; }

# ── noise-control pins (ADR-0007 "Noise controls") ──────────────────────────
# Single-thread primary lane: pin every thread pool this process tree can see
# to 1 so BLAS/OpenMP/GCD fan-out does not turn a "cost of one call" number
# into "cost of one call spread over N cores". Axis scripts that deliberately
# want multi-threaded throughput (none in wave 1) must override explicitly
# and record the override in their JSON.
bench_pin_single_thread() {
    export OMP_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    # JIT cache state must be pinned and recorded, not ambient (ADR-0007).
    export ESHKOL_JIT_CACHE="${ESHKOL_JIT_CACHE:-0}"
}

# ── disk cap (mandatory per-harness budget, matches scripts/run_p8_escape.sh
#    and the fuzz/harness disk-budget policy in project memory) ─────────────
BENCH_DISK_CAP_MB="${BENCH_DISK_CAP_MB:-2048}"
bench_disk_cap_check() { # <dir>
    local dir="$1" mb
    [ -d "$dir" ] || return 0
    mb="$(du -sm "$dir" 2>/dev/null | awk '{print $1}')"
    [ -n "$mb" ] || return 0
    if [ "$mb" -gt "$BENCH_DISK_CAP_MB" ]; then
        bench_die "$dir exceeded the ${BENCH_DISK_CAP_MB}MB benchmark disk cap (now ${mb}MB) — aborting rather than filling the disk."
    fi
}

# One scratch dir per axis run, cleaned up on exit unless BENCH_KEEP_WORK=1.
# Never under /tmp by default when a durable root is available; falls back to
# mktemp under TMPDIR (matching every other harness in this repo) only when
# the caller has not pointed BENCH_WORK_ROOT somewhere durable.
bench_make_work_dir() { # <label> -> prints path
    local label="$1" base
    if [ -n "${BENCH_WORK_ROOT:-}" ]; then
        base="${BENCH_WORK_ROOT%/}/${label}"
        mkdir -p "$base" || bench_die "could not create $base"
        printf '%s\n' "$base"
    else
        mktemp -d "${TMPDIR:-/tmp}/eshkol-bench-${label}.XXXXXX"
    fi
}

bench_cleanup_work_dir() { # <dir>
    local dir="$1"
    [ -n "$dir" ] || return 0
    [ -d "$dir" ] || return 0
    if [ "${BENCH_KEEP_WORK:-0}" = "1" ]; then
        bench_log "BENCH_KEEP_WORK=1 — leaving $dir in place"
        return 0
    fi
    rm -rf -- "$dir"
}

# ── timing ───────────────────────────────────────────────────────────────
# Nanosecond wall clock from the shell (used only for process-level spans;
# in-language timing inside .esk fixtures uses (current-time-ns) so the
# measured span never includes process startup/compile time).
bench_now_ns() {
    python3 -c 'import time; print(time.time_ns())' 2>/dev/null \
        || perl -MTime::HiRes=time -e 'printf("%d\n", time()*1e9)'
}

# ── peak-RSS measurement (macOS BSD `time -l` / Linux GNU `time -v`) ───────
# Detected once per process; callers source this file then call
# bench_detect_time_mode before bench_run_rss.
BENCH_TIME_MODE=""
bench_detect_time_mode() {
    local probe
    probe="$(bench_make_work_dir time-probe)"
    if /usr/bin/time -l true >/dev/null 2>"$probe/probe" && grep -q "maximum resident set size" "$probe/probe" 2>/dev/null; then
        BENCH_TIME_MODE="bsd"
    elif /usr/bin/time -v true >"$probe/probe" 2>&1 && grep -qi "Maximum resident set size" "$probe/probe" 2>/dev/null; then
        BENCH_TIME_MODE="gnu"
    fi
    rm -rf "$probe"
    [ -n "$BENCH_TIME_MODE" ] || bench_die "neither /usr/bin/time -l (macOS) nor -v (Linux) reports peak RSS on this host"
}

# bench_run_rss <bin> <out> <timelog> <timeout_s> [args...] -> sets
# BENCH_RSS_MB BENCH_RUN_RC
bench_run_rss() {
    local bin="$1" out="$2" timelog="$3" timeout_s="$4"; shift 4
    if [ "$BENCH_TIME_MODE" = "bsd" ]; then
        /usr/bin/time -l perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
            "$timeout_s" "$bin" "$@" >"$out" 2>"$timelog"
        BENCH_RUN_RC=$?
        BENCH_RSS_MB=$(awk '/maximum resident set size/{printf "%d", $1/1048576}' "$timelog")
    else
        /usr/bin/time -v perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
            "$timeout_s" "$bin" "$@" >"$out" 2>"$timelog"
        BENCH_RUN_RC=$?
        BENCH_RSS_MB=$(awk -F: '/Maximum resident set size/{printf "%d", $2/1024}' "$timelog")
    fi
    [ -n "$BENCH_RSS_MB" ] || BENCH_RSS_MB=0
}

# ── JSON helpers (python3 is required; every macOS/Linux CI/dev host has it) ─
bench_require_python3() {
    command -v python3 >/dev/null 2>&1 || bench_die "python3 is required by the benchmark harness (JSON emission) and was not found on PATH"
}

# bench_json_escape <string> -> prints a JSON string literal (with quotes)
bench_json_escape() {
    python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}
