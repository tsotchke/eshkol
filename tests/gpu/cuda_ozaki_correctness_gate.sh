#!/usr/bin/env bash
# tests/gpu/cuda_ozaki_correctness_gate.sh — INT8 tensor-core Ozaki f64 GEMM gate.
#
# Drives the opt-in CUDA INT8-Ozaki matmul (ESHKOL_CUDA_F64_KERNEL=ozaki-int8)
# and asserts it is numerically correct vs an independent in-process CPU f64
# reference across integer / fractional / pi-e / wide-magnitude regimes at K up
# to 4096, plus:
#   * default T=6 (full f64): normwise Frobenius error is machine-precision
#     (< 1e-9) on every regime — proving the INT8 tensor-core path recovers f64,
#   * fast T=4 (~1e-11): still passes the correctness gate (normwise < 1e-6),
#   * an out-of-range accuracy knob (ESHKOL_OZAKI_CUDA_T=99) FAILS LOUDLY (a
#     clamp warning on stderr) and still returns correct results (clamped to the
#     default T=6) instead of garbage — mirrors #307's loud-clamp philosophy.
#
# The K<133000 int32-exactness guard (loud fallback to cublasDgemm) is verified
# behaviorally at unit level on real 3090 hardware by the standalone harness
# (scratchpad ozaki_cuda_proto / cuda_ozaki_unit_harness.cu — a K>133000 skinny
# GEMM is infeasible to build as an in-language tensor here). The guard itself is
# a single K-bound check by inspection.
#
# INT8-Ozaki is a CUDA (IMMA tensor-core) kernel. On a Metal-only or GPU-less
# host it SKIPS cleanly (exit 0) and leaves no ICC trace, so the oracle reports
# "no evidence yet" rather than a false PASS.
#
# Usage:
#   ESHKOL_RUN=/path/to/eshkol-run tests/gpu/cuda_ozaki_correctness_gate.sh
# If ESHKOL_RUN is unset it searches build*/eshkol-run under the repo root.
set -u
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
TEST_ESK="$REPO_ROOT/tests/gpu/cuda_ozaki_correctness_test.esk"

# ICC evidence trace (.icc/completion-oracles.yaml::cuda-ozaki-int8-correctness).
# Written only on an actual PASS/FAIL verdict; a SKIP leaves no trace.
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/cuda_ozaki_correctness.jsonl"
emit_trace() {  # emit_trace <value> <snippet>
    local value="$1" snippet="$2"
    mkdir -p "$TRACE_DIR"
    local esnip
    if command -v python3 >/dev/null 2>&1; then
        esnip="$(printf '%s' "$snippet" | python3 -c 'import json,sys; sys.stdout.write(json.dumps(sys.stdin.read()))')"
    else
        esnip="$(printf '%s' "$snippet" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"
        esnip="\"$esnip\""
    fi
    printf '{"kind":"cuda_ozaki_correctness","name":"cuda_ozaki_correctness_gate","value":"%s","snippet":%s,"confidence":0.95}\n' \
        "$value" "$esnip" >> "${TRACE_FILE:?}"
}

log()  { printf '%s\n' "$*"; }
skip() { log "SKIP: $*"; exit 0; }
fail() { log "FAIL: $*"; emit_trace FAIL "$*"; exit 1; }
pass() { log "PASS: $*"; emit_trace PASS "$*"; exit 0; }

# ── GPU presence (CUDA / IMMA only) ─────────────────────────────────────────
UNAME_S="$(uname -s)"
if [ "$UNAME_S" = "Darwin" ]; then
    skip "INT8-Ozaki is a CUDA (IMMA tensor-core) kernel; no CUDA GPU on macOS"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    log "note: CUDA host detected — driving INT8-Ozaki f64 GEMM"
    nvidia-smi -L 2>/dev/null | head -1
else
    skip "no CUDA GPU detected (nvidia-smi absent or no device)"
fi

# ── locate eshkol-run ───────────────────────────────────────────────────────
BIN="${ESHKOL_RUN:-}"
if [ -z "$BIN" ]; then
    for d in build-ozaki build build-gpu-gate build-*; do
        if [ -x "$REPO_ROOT/$d/eshkol-run" ]; then BIN="$REPO_ROOT/$d/eshkol-run"; break; fi
    done
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || skip "no eshkol-run binary found (set ESHKOL_RUN)"
log "using eshkol-run: $BIN"

# Force the GPU INT8-Ozaki path for every matmul regardless of size.
FORCE_GPU=(
  ESHKOL_GPU_THRESHOLD=1
  ESHKOL_GPU_MATMUL_THRESHOLD=1
  ESHKOL_GPU_WAIT_TIMEOUT=300
  ESHKOL_CUDA_F64_KERNEL=ozaki-int8
)

OUT_F="$(mktemp)"
ERR_F="$(mktemp)"
run_case() {  # run_case <extra-env...>; stdout -> $OUT_F, stderr -> $ERR_F
    env "${FORCE_GPU[@]}" "$@" "$BIN" -r "$TEST_ESK" >"$OUT_F" 2>"$ERR_F"
}

# max NORMWISE value printed across all REGIME lines (awk float compare)
max_normwise() { awk '
    { for (i=1;i<=NF;i++) if ($i ~ /^NORMWISE=/) { v=substr($i,10)+0; if (v>m) m=v } }
    END { printf "%.3e", m }' ; }

ALL_OUT=""
overall=0

# ── Case 1: DEFAULT T=6 (full f64). HARD: normwise machine-precision everywhere ─
log "=== Case 1: default INT8-Ozaki T=6 (full f64) across all regimes, K<=4096 [HARD] ==="
run_case; OUT1="$(cat "$OUT_F")"; ALL_OUT="$OUT1"
printf '%s\n' "$OUT1"
printf '%s\n' "$OUT1" | grep -q "SUMMARY total_fail=0" || { overall=1; log "  -> T=6 produced failures (normwise >= 1e-6)"; }
printf '%s\n' "$OUT1" | grep -q "VERDICT=FAIL" && { overall=1; log "  -> T=6 has a FAIL verdict"; }
MAXNW1="$(printf '%s\n' "$OUT1" | max_normwise)"
if awk "BEGIN{exit !($MAXNW1 < 1e-9)}"; then
    log "  -> T=6 full-f64 confirmed: max normwise error $MAXNW1 < 1e-9"
else
    overall=1; log "  -> T=6 NOT full-f64: max normwise error $MAXNW1 >= 1e-9 (INT8 tensor-core path did not recover f64)"
fi

# ── Case 2: FAST T=4 (~1e-11). Correctness gate must still pass (normwise<1e-6). ─
log "=== Case 2: fast INT8-Ozaki T=4 (~1e-11, ~2x faster) — correctness gate [HARD] ==="
run_case ESHKOL_OZAKI_CUDA_T=4; OUT2="$(cat "$OUT_F")"
printf '%s\n' "$OUT2"
printf '%s\n' "$OUT2" | grep -q "SUMMARY total_fail=0" || { overall=1; log "  -> T=4 produced failures (normwise >= 1e-6)"; }
MAXNW2="$(printf '%s\n' "$OUT2" | max_normwise)"
log "  -> T=4 max normwise error $MAXNW2 (expected ~1e-11, gate is 1e-6)"

# ── Case 3: out-of-range knob must FAIL LOUDLY (warn) and stay correct [HARD] ──
log "=== Case 3: out-of-range ESHKOL_OZAKI_CUDA_T=99 -> loud clamp to T=6, still correct [HARD] ==="
run_case ESHKOL_OZAKI_CUDA_T=99; OUT3="$(cat "$OUT_F")"
printf '%s\n' "$OUT3"
grep -qi "out of range" "$ERR_F" || { overall=1; log "  -> expected a loud out-of-range warning on stderr, none seen"; }
printf '%s\n' "$OUT3" | grep -q "SUMMARY total_fail=0" || { overall=1; log "  -> clamped run still incorrect"; }

# ── verdict ─────────────────────────────────────────────────────────────────
SNIP="$(printf '%s\n' "$ALL_OUT" | grep -E 'REGIME=|SUMMARY' | head -20)"
rm -f "$OUT_F" "$ERR_F"
if [ "$overall" -eq 0 ]; then
    pass "INT8-Ozaki f64 GEMM: T=6 machine-precision (normwise $MAXNW1) and T=4 correct across integer/fractional/pi_e/wide regimes at K<=4096; out-of-range knob clamps loudly and stays correct"
else
    fail "INT8-Ozaki f64 GEMM correctness gate failed; output:\n$SNIP"
fi
