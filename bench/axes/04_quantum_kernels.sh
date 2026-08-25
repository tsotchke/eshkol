#!/usr/bin/env bash
# bench/axes/04_quantum_kernels.sh — AXIS 4: differentiable quantum kernels.
#
# Two shipped examples/tests, run as-is (not regenerated — these are already
# the exact fixtures the public claims point at):
#   * examples/h2_vibrational.esk — H2 vibrational frequency via
#     ARBITRARY-ORDER AD (derivative-n to 2nd order gives the exact Hessian
#     d^2E/dR^2, no finite differences). Prints equilibrium R*, the harmonic
#     force constant, and the vibrational frequency. Pure Eshkol AD — no
#     quantum build required.
#   * tests/quantum/vqe_test.esk — H2 VQE energy + native adjoint gradient
#     against Moonlab's exact ground-energy oracle, printing
#     "H2 |VQE - exact| (Ha)". REQUIRES a quantum-enabled build
#     (-DESHKOL_QUANTUM_ENABLED=ON -DESHKOL_BUILD_AGENT_FFI=ON), and is
#     SKIPPED (not faked) when the build directory was not configured that
#     way — detected via CMakeCache.txt, same convention as
#     bench/lib/fingerprint.sh.
#
# Reports wall-clock (each fixture compiled once AOT, then run ROUNDS times
# as fresh processes — wall-clock necessarily includes program startup, since
# these are one-shot computations rather than tight loops timed in-language)
# plus the achieved scientific values themselves (R*, force constant,
# frequency; VQE energies and |VQE-exact|), so a reader can see both cost and
# correctness together.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
. "$REPO_ROOT/bench/lib/common.sh"

BUILD_DIR="${1:?usage: 04_quantum_kernels.sh <build_dir> <work_dir> <json_out> <md_out> [smoke]}"
WORK_DIR="${2:?}"
JSON_OUT="${3:?}"
MD_OUT="${4:?}"
SMOKE="${5:-0}"

ESHKOL_RUN="$BUILD_DIR/eshkol-run"
[ -x "$ESHKOL_RUN" ] || bench_die "04_quantum_kernels: $ESHKOL_RUN not found or not executable"

mkdir -p "$WORK_DIR"
bench_pin_single_thread

if [ "$SMOKE" = "1" ]; then
    ROUNDS=1
    TIMEOUT_S=60
else
    ROUNDS=5
    TIMEOUT_S=300
fi

QUANTUM_ENABLED=0
if [ -f "$BUILD_DIR/CMakeCache.txt" ] && grep -q '^ESHKOL_QUANTUM_ENABLED:BOOL=ON' "$BUILD_DIR/CMakeCache.txt"; then
    QUANTUM_ENABLED=1
fi

bench_disk_cap_check "$WORK_DIR"

# ── H2 vibrational Hessian (no quantum build needed) ────────────────────────
H2_SRC="$REPO_ROOT/examples/h2_vibrational.esk"
H2_BIN="$WORK_DIR/h2_vibrational"
H2_OK=0
H2_WALL_NS_LIST=""
H2_OUT="$WORK_DIR/h2.out"
if [ -f "$H2_SRC" ]; then
    if "$ESHKOL_RUN" "$H2_SRC" -o "$H2_BIN" -L"$BUILD_DIR" >"$WORK_DIR/h2.compile.log" 2>&1; then
        for i in $(seq 1 "$ROUNDS"); do
            t0="$(bench_now_ns)"
            perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
                "$TIMEOUT_S" "$H2_BIN" >"$H2_OUT" 2>"$WORK_DIR/h2.stderr.$i"
            rc=$?
            t1="$(bench_now_ns)"
            if [ "$rc" -eq 0 ]; then
                H2_OK=1
                H2_WALL_NS_LIST="$H2_WALL_NS_LIST $((t1 - t0))"
            fi
        done
    else
        bench_log "04_quantum_kernels: h2_vibrational.esk failed to compile (see $WORK_DIR/h2.compile.log)"
    fi
else
    bench_log "04_quantum_kernels: $H2_SRC not found"
fi

# ── VQE H2 energy + gradient (requires quantum build) ───────────────────────
VQE_SRC="$REPO_ROOT/tests/quantum/vqe_test.esk"
VQE_BIN="$WORK_DIR/vqe_test"
VQE_OK=0
VQE_WALL_NS_LIST=""
VQE_OUT="$WORK_DIR/vqe.out"
if [ "$QUANTUM_ENABLED" = "1" ] && [ -f "$VQE_SRC" ]; then
    if "$ESHKOL_RUN" "$VQE_SRC" -o "$VQE_BIN" -L"$BUILD_DIR" >"$WORK_DIR/vqe.compile.log" 2>&1; then
        for i in $(seq 1 "$ROUNDS"); do
            t0="$(bench_now_ns)"
            perl -e 'my $s=shift; alarm $s; exec @ARGV; die "exec failed: $!\n"' \
                "$TIMEOUT_S" "$VQE_BIN" >"$VQE_OUT" 2>"$WORK_DIR/vqe.stderr.$i"
            rc=$?
            t1="$(bench_now_ns)"
            if [ "$rc" -eq 0 ] && grep -q "VQE H2: PASS" "$VQE_OUT"; then
                VQE_OK=1
                VQE_WALL_NS_LIST="$VQE_WALL_NS_LIST $((t1 - t0))"
            fi
        done
    else
        bench_log "04_quantum_kernels: vqe_test.esk failed to compile (see $WORK_DIR/vqe.compile.log)"
    fi
elif [ "$QUANTUM_ENABLED" != "1" ]; then
    bench_log "04_quantum_kernels: $BUILD_DIR is not a quantum-enabled build (ESHKOL_QUANTUM_ENABLED != ON) — VQE row skipped, not faked"
fi

bench_disk_cap_check "$WORK_DIR"

python3 "$SCRIPT_DIR/04_quantum_kernels_reduce.py" \
    --h2-ok "$H2_OK" --h2-out "$H2_OUT" --h2-wall-ns "${H2_WALL_NS_LIST# }" \
    --quantum-enabled "$QUANTUM_ENABLED" --vqe-ok "$VQE_OK" --vqe-out "$VQE_OUT" --vqe-wall-ns "${VQE_WALL_NS_LIST# }" \
    --json-out "$JSON_OUT" --md-out "$MD_OUT" \
    || bench_die "04_quantum_kernels: result reduction failed"

bench_log "04_quantum_kernels: wrote $JSON_OUT and $MD_OUT"
