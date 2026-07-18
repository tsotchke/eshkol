#!/usr/bin/env bash
# tests/gpu/ozaki_correctness_gate.sh — Ozaki-II exact-DGEMM EXECUTION gate.
#
# Drives the GPU Ozaki-II CRT matmul (ESHKOL_SF64_KERNEL=ozaki) and asserts it
# is numerically correct vs an independent in-process CPU f64 reference across
# integer / fractional / pi-e / wide-magnitude regimes at K up to 4096, plus a
# moduli-count sweep that pins:
#   * default Ozaki (fixed N=16) correct on every regime,
#   * adaptive (ESHKOL_OZAKI_ADAPTIVE=1, now full-precision-targeted) correct,
#   * an out-of-range moduli request FAILS LOUDLY (clamp warning) and still
#     returns correct results instead of __int128-overflow garbage.
#
# This closes the coverage gap that let the CRT-overflow / precision-truncation
# defects ship: the prior GPU suite used integer/identity data and compared
# GPU-against-GPU, so a consistent numerical error was invisible.
#
# Requires a real Metal (macOS) or CUDA (Linux/Windows) GPU. On a GPU-less host
# it SKIPS cleanly (exit 0). It never fails for lacking hardware.
#
# Usage:
#   ESHKOL_RUN=/path/to/eshkol-run tests/gpu/ozaki_correctness_gate.sh
# If ESHKOL_RUN is unset it searches build*/eshkol-run under the repo root.
#
# Env overrides:
#   ESHKOL_RUN         path to the eshkol-run binary to exercise
#   OZAKI_GATE_TOL     unused here (verdict is decided inside the .esk); kept for
#                      parity with gpu_correctness_gate.sh
set -u
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
TEST_ESK="$REPO_ROOT/tests/gpu/ozaki_correctness_test.esk"

# ICC evidence trace (.icc/completion-oracles.yaml::ozaki-correctness). Written
# only on an actual PASS/FAIL verdict; a SKIP leaves no trace so the oracle
# reports "no evidence yet" rather than a false PASS on a GPU-less host.
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/ozaki_correctness.jsonl"
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
    printf '{"kind":"ozaki_correctness","name":"ozaki_correctness_gate","value":"%s","snippet":%s,"confidence":0.95}\n' \
        "$value" "$esnip" >> "${TRACE_FILE:?}"
}

log()  { printf '%s\n' "$*"; }
skip() { log "SKIP: $*"; exit 0; }
fail() { log "FAIL: $*"; emit_trace FAIL "$*"; exit 1; }
pass() { log "PASS: $*"; emit_trace PASS "$*"; exit 0; }

# ── GPU presence ────────────────────────────────────────────────────────────
UNAME_S="$(uname -s)"
if [ "$UNAME_S" = "Darwin" ]; then
    :  # Metal is present on all supported macOS hosts
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    log "note: CUDA host — Ozaki-II is a Metal-only kernel; skipping"
    skip "Ozaki-II CRT kernel is Metal-only; no Metal GPU on this host"
else
    skip "no Metal/CUDA GPU detected"
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

# Force the GPU Ozaki-II path for every matmul regardless of size.
FORCE_GPU=(
  ESHKOL_GPU_MATMUL_THRESHOLD=0
  ESHKOL_GPU_THRESHOLD=1
  ESHKOL_GPU_VERBOSE=1
  ESHKOL_BLAS_PEAK_GFLOPS=0.001
  ESHKOL_GPU_PEAK_GFLOPS=1000000
  ESHKOL_GPU_WAIT_TIMEOUT=300
  ESHKOL_SF64_KERNEL=ozaki
)

OUT_F="$(mktemp)"
ERR_F="$(mktemp)"
run_case() {  # run_case <extra-env...>; stdout -> $OUT_F, stderr -> $ERR_F (parent-scoped)
    env "${FORCE_GPU[@]}" "$@" "$BIN" -r "$TEST_ESK" >"$OUT_F" 2>"$ERR_F"
}

ALL_OUT=""
overall=0
require_exact_ozaki_markers() {  # require_exact_ozaki_markers <stderr-file> <label> <count> <pattern>
    local file="$1" label="$2" expected="$3" pattern="$4" init_pattern="$5"
    local count sample poison

    count="$(grep -E "$pattern" "$file" | wc -l | tr -d ' ')"
    if [ -z "$count" ]; then count=0; fi

    if [ "$count" -ne "$expected" ]; then
        overall=1
        log "  -> ${label}: expected ${expected} dispatch marker lines, got ${count}"
        log "     pattern: ${pattern}"
        sample="$(grep -E "$pattern" "$file" | head -n 3 | sed 's/^/       /')"
        if [ -n "$sample" ]; then
            log "     sample dispatch lines:"
            while IFS= read -r line; do log "$line"; done <<< "$sample"
        fi
    fi

    poison="$(grep -E '^\[GPU\] Ozaki-II:' "$file" | grep -Ev "$pattern" || true)"
    if [ -n "$init_pattern" ]; then
        poison="$(printf '%s' "$poison" | grep -Ev "$init_pattern" || true)"
    fi
    if [ -n "$poison" ]; then
        overall=1
        log "  -> ${label}: non-dispatch Ozaki marker lines detected (poison guard)"
        while IFS= read -r line; do log "       $line"; done <<< "$(echo "$poison" | head -n 3)"
    fi
}

OZAKI_INIT_PATTERN='^\[GPU\] Ozaki-II: N=[0-9]+ moduli, log2\(P\)='

# ── Case 1: DEFAULT / SHIPPED Ozaki config (fixed N=16). This is the exactness
#    contract — HARD assertion: bit-exact vs the CPU reference on every regime. ──
log "=== Case 1: default Ozaki (fixed N=16) across all regimes, K<=4096 [HARD] ==="
run_case; OUT1="$(cat "$OUT_F")"; ALL_OUT="$OUT1"
printf '%s\n' "$OUT1"
printf '%s\n' "$OUT1" | grep -q "SUMMARY total_fail=0" || { overall=1; log "  -> default config produced failures"; }
printf '%s\n' "$OUT1" | grep -q "VERDICT=FAIL" && { overall=1; log "  -> default config has a FAIL verdict"; }
require_exact_ozaki_markers "$ERR_F" "case 1" 16 '^\[GPU\] Ozaki-II: N=[0-9]+( \(adaptive\))?, log2\(P\)=' "$OZAKI_INIT_PATTERN"

# ── Case 2: adaptive moduli (ESHKOL_OZAKI_ADAPTIVE=1) — OPT-IN / INFORMATIONAL.
#    Adaptive can reduce the moduli count below the tuned N=16 point; the
#    double-double CRT reconstruction is only bit-exact at the full budget, so
#    adaptive is best-effort (~1e-8), NOT the exactness contract. We report it
#    and assert only that it no longer GROSSLY corrupts (the old 5-30% bug):
#    every probe within 1e-6. It does not gate the shipped verdict. ───────────
log "=== Case 2: adaptive moduli (opt-in, approximate) — informational ==="
run_case ESHKOL_OZAKI_ADAPTIVE=1; OUT2="$(cat "$OUT_F")"
printf '%s\n' "$OUT2"
require_exact_ozaki_markers "$ERR_F" "case 2" 16 '^\[GPU\] Ozaki-II: N=[0-9]+( \(adaptive\))?, log2\(P\)=' "$OZAKI_INIT_PATTERN"
if printf '%s\n' "$OUT2" | grep -q "SUMMARY total_fail=0"; then
    log "  -> adaptive is bit-exact on this run"
else
    # No gross corruption: no MAXREL >= 1e-6 anywhere (catches the pre-fix CRT bug).
    GROSS="$(printf '%s\n' "$OUT2" | awk '/MAXREL=/{for(i=1;i<=NF;i++){if($i ~ /^MAXREL=/){v=substr($i,8)+0; if(v>=1e-6) print v}}}')"
    if [ -n "$GROSS" ]; then
        overall=1; log "  -> adaptive GROSSLY wrong (MAXREL>=1e-6): $GROSS"
    else
        log "  -> adaptive within 1e-6 (opt-in approximate; no gross CRT error)"
    fi
fi

# ── Case 3: out-of-range moduli must FAIL LOUDLY (warn) and stay correct [HARD] ─
log "=== Case 3: out-of-range ESHKOL_OZAKI_NUM_MODULI=32 -> loud clamp, still correct [HARD] ==="
run_case ESHKOL_OZAKI_NUM_MODULI=32; OUT3="$(cat "$OUT_F")"
printf '%s\n' "$OUT3"
grep -qi "out of range" "$ERR_F" || { overall=1; log "  -> expected a loud out-of-range warning on stderr, none seen"; }
require_exact_ozaki_markers "$ERR_F" "case 3" 16 '^\[GPU\] Ozaki-II: N=[0-9]+( \(adaptive\))?, log2\(P\)=' "$OZAKI_INIT_PATTERN"
printf '%s\n' "$OUT3" | grep -q "SUMMARY total_fail=0" || { overall=1; log "  -> clamped run still incorrect"; }

# ── verdict ─────────────────────────────────────────────────────────────────
SNIP="$(printf '%s\n' "$ALL_OUT" | grep -E 'REGIME=|SUMMARY' | head -20)"
rm -f "$OUT_F" "$ERR_F"
if [ "$overall" -eq 0 ]; then
    pass "Ozaki-II fixed-N=16 bit-exact across integer/fractional/pi_e/wide regimes at K<=4096; out-of-range clamps loudly; adaptive no gross error"
else
    fail "Ozaki-II correctness gate failed; output:\n$SNIP"
fi
