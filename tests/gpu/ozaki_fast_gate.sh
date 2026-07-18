#!/usr/bin/env bash
# tests/gpu/ozaki_fast_gate.sh — Ozaki-II REDUCED-PRECISION fast-tier gate.
#
# Drives the opt-in fully-GPU Ozaki-II fast tier (ESHKOL_SF64_KERNEL=ozaki-fast:
# GPU residue split + df32 CRT reconstruction, fewer moduli) and asserts:
#
#   1. ACCURACY — every probe across the four data regimes (integer, fractional,
#      pi/e, wide-magnitude) at K up to 4096 is within the documented ~1e-8 tier
#      bound (rel err <= 1e-7 vs an independent in-process CPU f64 reference).
#   2. THE FAST PATH ACTUALLY ENGAGES — the dispatch prints its "[GPU] Ozaki-II
#      FAST" marker AND the accuracy is the reduced-precision signature (NOT the
#      exact tier's ~1e-14): a run that silently fell back to the exact/serial
#      tier would be near-bit-exact, so we assert the marker is present and the
#      reconstruction never fell back (no "falling back to exact tier" on stderr).
#
# This is a companion to ozaki_correctness_gate.sh, which pins the DEFAULT
# (exact, bit-exact N=16) path and MUST keep passing untouched. This gate governs
# only the opt-in fast tier and never affects the default.
#
# Requires a real Metal (macOS) GPU. On a GPU-less host it SKIPS cleanly (exit 0).
#
# Usage:  ESHKOL_RUN=/path/to/eshkol-run tests/gpu/ozaki_fast_gate.sh
set -u
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
TEST_ESK="$REPO_ROOT/tests/gpu/ozaki_fast_test.esk"

TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/ozaki_fast.jsonl"
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
    printf '{"kind":"ozaki_fast","name":"ozaki_fast_gate","value":"%s","snippet":%s,"confidence":0.95}\n' \
        "$value" "$esnip" >> "${TRACE_FILE:?}"
}

log()  { printf '%s\n' "$*"; }
skip() { log "SKIP: $*"; exit 0; }
fail() { log "FAIL: $*"; emit_trace FAIL "$*"; exit 1; }
pass() { log "PASS: $*"; emit_trace PASS "$*"; exit 0; }

# ── GPU presence ────────────────────────────────────────────────────────────
if [ "$(uname -s)" = "Darwin" ]; then
    :  # Metal present on all supported macOS hosts
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    skip "Ozaki-II fast tier is a Metal-only kernel; no Metal GPU on this host"
else
    skip "no Metal GPU detected"
fi

# ── locate eshkol-run ───────────────────────────────────────────────────────
BIN="${ESHKOL_RUN:-}"
if [ -z "$BIN" ]; then
    for d in build-metalfast build-ozaki build build-gpu-gate build-*; do
        if [ -x "$REPO_ROOT/$d/eshkol-run" ]; then BIN="$REPO_ROOT/$d/eshkol-run"; break; fi
    done
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || skip "no eshkol-run binary found (set ESHKOL_RUN)"
log "using eshkol-run: $BIN"

# Force every matmul through the GPU + select the opt-in fast tier (default 10
# moduli = the documented ~1e-8 tier).
FORCE_FAST=(
  ESHKOL_GPU_MATMUL_THRESHOLD=0
  ESHKOL_GPU_THRESHOLD=1
  ESHKOL_GPU_VERBOSE=1
  ESHKOL_BLAS_PEAK_GFLOPS=0.001
  ESHKOL_GPU_PEAK_GFLOPS=1000000
  ESHKOL_GPU_WAIT_TIMEOUT=300
  ESHKOL_SF64_KERNEL=ozaki-fast
)

OUT_F="$(mktemp)"; ERR_F="$(mktemp)"
env "${FORCE_FAST[@]}" "$BIN" -r "$TEST_ESK" >"$OUT_F" 2>"$ERR_F"
OUT="$(cat "$OUT_F")"
printf '%s\n' "$OUT"

overall=0

# (1) accuracy: no FAIL verdicts, total_fail=0 (all probes <= 1e-7)
printf '%s\n' "$OUT" | grep -q "SUMMARY total_fail=0" || { overall=1; log "  -> fast tier exceeded the 1e-7 bound on some regime"; }
printf '%s\n' "$OUT" | grep -q "VERDICT=FAIL" && { overall=1; log "  -> fast tier has a FAIL verdict"; }

# (2) the fast path actually engaged (marker present)…
if ! grep -q "\[GPU\] Ozaki-II FAST" "$ERR_F"; then
    overall=1; log "  -> fast-tier marker '[GPU] Ozaki-II FAST' NOT seen — path did not engage"
fi
# …and never silently fell back to the exact tier (which would be near-bit-exact)
if grep -q "falling back to exact tier" "$ERR_F"; then
    overall=1; log "  -> fast dispatch fell back to the exact tier (not a genuine fast-path run)"
fi

MODULI="$(grep -o "Ozaki-II FAST tier ENABLED: N=[0-9]* moduli" "$ERR_F" | head -1)"
SNIP="$(printf '%s\n' "$OUT" | grep -E 'REGIME=|SUMMARY' | head -20)"
rm -f "$OUT_F" "$ERR_F"

if [ "$overall" -eq 0 ]; then
    pass "Ozaki-II fast tier (${MODULI:-fast}) within 1e-7 across integer/fractional/pi_e/wide at K<=4096; GPU split + df32 CRT reconstruction engaged"
else
    fail "Ozaki-II fast-tier gate failed; output:\n$SNIP"
fi
