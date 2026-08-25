#!/usr/bin/env bash
# bench/run_public_benchmarks.sh — the ONE public, reproducible entry point
# for Eshkol's benchmark-on-our-axes suite (v1.3.5 "benchmarks wave 1").
#
# WHY THIS EXISTS: every performance/rigor claim Eshkol has published was,
# until this suite, self-reported with no way for a stranger to reproduce
# it. This script measures ONLY the axes where the project claims something
# distinctive — it deliberately does not benchmark Eshkol against XLA/PyTorch
# on their own turf (ResNet training, large dense float64 GEMM throughput as
# a horse race, etc.) — see bench/README.md "What this suite does NOT
# benchmark, and why" for the explicit list.
#
# THE FOUR AXES (wave 1):
#   1. exact-AD cost curves        — bench/axes/01_exact_ad.sh
#   2. Ozaki-II CRT exact f64 GEMM — bench/axes/02_ozaki_gemm.sh
#   3. flat-RSS under resident load — bench/axes/03_flat_rss.sh
#   4. differentiable quantum kernels — bench/axes/04_quantum_kernels.sh
#
# Usage:
#   bench/run_public_benchmarks.sh --build-dir build [--smoke] [--only 1,3]
#       [--out-dir DIR] [--work-root DIR] [--keep-work]
#
#   --build-dir DIR   an already-configured-and-built Eshkol build directory.
#                      Axis 4's VQE half needs -DESHKOL_QUANTUM_ENABLED=ON;
#                      every other axis works on an ordinary build (it is
#                      skipped, honestly, not faked, otherwise).
#   --smoke           fast harness-correctness subset (tiny sweeps, seconds
#                      not minutes) — NOT a measurement run. Used by CI.
#   --only 1,2,3,4     comma-separated axis numbers to run (default: all).
#   --out-dir DIR      where results.json / results.md land
#                      (default: bench/results/<UTC timestamp>/).
#   --work-root DIR    scratch root for generated .esk/binaries (default:
#                      mktemp under TMPDIR, per-axis disk-capped and cleaned
#                      up — see bench/lib/common.sh). Point this somewhere
#                      durable (never /tmp) if you want to keep the raw
#                      per-run artifacts.
#   --keep-work        do not delete --work-root contents on exit.
#
# Noise controls and result-protocol fields follow
# docs/design/adr/0007-performance-pgo-wpo.md ("Result protocol" / "Noise
# controls"); see bench/README.md for the full mapping.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"
# shellcheck source=lib/fingerprint.sh
. "$SCRIPT_DIR/lib/fingerprint.sh"

bench_require_python3

BUILD_DIR=""
SMOKE=0
ONLY="1,2,3,4"
OUT_DIR=""
KEEP_WORK=0

while [ $# -gt 0 ]; do
    case "$1" in
        --build-dir) BUILD_DIR="${2:?}"; shift 2 ;;
        --smoke) SMOKE=1; shift ;;
        --only) ONLY="${2:?}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
        --work-root) export BENCH_WORK_ROOT="${2:?}"; shift 2 ;;
        --keep-work) KEEP_WORK=1; shift ;;
        -h|--help)
            sed -n '2,40p' "$0"; exit 0 ;;
        *) bench_die "unknown argument: $1 (see --help)" ;;
    esac
done

[ -n "$BUILD_DIR" ] || bench_die "--build-dir is required, e.g. --build-dir build"
case "$BUILD_DIR" in
    /*) ;;
    *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;;
esac
[ -x "$BUILD_DIR/eshkol-run" ] || bench_die "$BUILD_DIR/eshkol-run not found or not executable — build first (see bench/README.md)"

export BENCH_KEEP_WORK="$KEEP_WORK"

STARTED_AT="$(python3 -c 'import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat())')"
RUN_ID="$(python3 -c 'import datetime; print(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))')"
[ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/bench/results/$RUN_ID"
mkdir -p "$OUT_DIR" || bench_die "could not create $OUT_DIR"

WORK_ROOT="$(bench_make_work_dir "run-$RUN_ID")"
bench_log "run_public_benchmarks: build_dir=$BUILD_DIR out_dir=$OUT_DIR work_root=$WORK_ROOT smoke=$SMOKE only=$ONLY"
trap 'bench_cleanup_work_dir "$WORK_ROOT"' EXIT

run_axis() { # <n> <script> <label>
    local n="$1" script="$2" label="$3"
    case ",$ONLY," in
        *",$n,"*) ;;
        *) bench_log "skipping axis $n ($label) — not in --only $ONLY"; return 0 ;;
    esac
    local axis_work="$WORK_ROOT/axis$n"
    mkdir -p "$axis_work"
    bench_log "=== axis $n: $label ==="
    if ! bash "$SCRIPT_DIR/axes/$script" "$BUILD_DIR" "$axis_work" \
            "$OUT_DIR/axis${n}.json" "$OUT_DIR/axis${n}.md" "$SMOKE"; then
        bench_log "axis $n ($label) FAILED — see above; continuing with remaining axes"
        return 1
    fi
}

OVERALL_RC=0
run_axis 1 01_exact_ad.sh "exact-AD cost curves" || OVERALL_RC=1
run_axis 2 02_ozaki_gemm.sh "Ozaki-II CRT exact f64 GEMM" || OVERALL_RC=1
run_axis 3 03_flat_rss.sh "flat-RSS under resident load" || OVERALL_RC=1
run_axis 4 04_quantum_kernels.sh "differentiable quantum kernels" || OVERALL_RC=1

FINISHED_AT="$(python3 -c 'import datetime; print(datetime.datetime.now(datetime.timezone.utc).isoformat())')"

bench_capture_fingerprint "$BUILD_DIR" "eshkol-public-benchmarks" > "$OUT_DIR/fingerprint.json"

combine_args=(--fingerprint "$OUT_DIR/fingerprint.json" --json-out "$OUT_DIR/results.json" \
    --md-out "$OUT_DIR/results.md" --smoke "$SMOKE" --started-at "$STARTED_AT" --finished-at "$FINISHED_AT")
for n in 1 2 3 4; do
    case ",$ONLY," in
        *",$n,"*)
            [ -f "$OUT_DIR/axis${n}.json" ] && combine_args+=(--axis${n} "$OUT_DIR/axis${n}.json")
            [ -f "$OUT_DIR/axis${n}.md" ] && combine_args+=(--axis${n}-md "$OUT_DIR/axis${n}.md")
            ;;
    esac
done
python3 "$SCRIPT_DIR/combine_results.py" "${combine_args[@]}" \
    || bench_die "failed to combine per-axis results"

bench_log ""
bench_log "wrote $OUT_DIR/results.json"
bench_log "wrote $OUT_DIR/results.md"
if [ "$OVERALL_RC" -ne 0 ]; then
    bench_log "one or more axes failed — see log above and per-axis logs under $WORK_ROOT (or rerun with --keep-work)"
fi
exit "$OVERALL_RC"
