#!/usr/bin/env bash
# run_xla_gate.sh — gate script for .icc/completion-oracles.yaml's
# `xla-tpu-ready` target. Each stage below corresponds to exactly one
# `requires:` criterion in that target; the action: line for each criterion
# names one of this script's flags.
#
#   Stage 0 (--baseline)       -> xla_backend_builds_and_baseline_recorded
#   Stage 1 (--pjrt-cpu)       -> xla_pjrt_cpu_roundtrip
#   Stage 2 (--op-parity)      -> xla_op_surface_parity
#   Stage 3 (--geometric-sweep)-> xla_geometric_parity
#   Stage 4 (--training-step)  -> xla_training_step_parity
#   Stage 5 (--multidevice)    -> xla_multidevice_step
#   Stage 6 (--numerics)       -> xla_bf16_numerics_bounded
#   Stage 7 (--production)     -> xla_tpu_production_ready
#
# HONESTY CONTRACT, non-negotiable:
#   A stage this script cannot yet genuinely exercise emits FAIL with reason
#   "stage not implemented", never PASS and never a silent skip. A gate that
#   cannot fail is worthless — several were found vacuous in this codebase
#   the week this script was written, and this file exists specifically not
#   to repeat that. Passing a stage here is a claim someone will build on;
#   only make it when this script actually ran something that could have
#   come back FAIL.
#
# Evidence: one JSON-L record per stage, appended (never truncated — the
# oracle calls this script once per criterion, at different times, and each
# call must not erase every other stage's already-recorded evidence) to
# scripts/icc_traces/xla.jsonl via eshkol_outcome_emit_event (see
# scripts/lib/harness_outcome.sh), the same emitter scripts/run_language_coverage.sh
# uses. Record shape: {"kind","name","value","snippet","confidence"} — kind
# is always "xla" here, name is the event name table above, value is the
# literal string PASS or FAIL.
#
# Every working/build directory this script creates lives under
# <repo>/.scratch/xla_gate/ — never /tmp or /private/tmp.
set -u

usage() {
    cat <<'EOF'
Usage: scripts/run_xla_gate.sh [STAGE...]

Stages (at least one required; each maps to one xla-tpu-ready oracle criterion):
  --baseline         Build with ESHKOL_XLA_ENABLED=ON, run xla_codegen_test.
                      -> xla_backend_builds_and_baseline_recorded
  --pjrt-cpu         Run pjrt_smoke_test against a CPU PJRT plugin if one can
                      be located.
                      -> xla_pjrt_cpu_roundtrip
  --op-parity        StableHLO op differential coverage vs CPU/CUDA.
                      -> xla_op_surface_parity
  --geometric-sweep  Hyperbolic/spherical/euclidean ops vs qllm_manifold_*.
                      -> xla_geometric_parity
  --training-step    Full training step (fwd/bwd/optimizer) vs CUDA path.
                      -> xla_training_step_parity
  --multidevice      Sharded training step across >=2 devices.
                      -> xla_multidevice_step
  --numerics         bf16 error bounds across a dimension sweep.
                      -> xla_bf16_numerics_bounded
  --production       TPU production deploy/preemption/checkpoint survival.
                      -> xla_tpu_production_ready
  --all              Run every stage above, in order.

Environment overrides:
  STABLEHLO_ROOT           Default: <repo>/deps/stablehlo
  XLA_GATE_BUILD_DIR       Default: <repo>/.scratch/xla_gate/build
  ESHKOL_PJRT_PLUGIN_PATH  Forces the exact PJRT plugin pjrt_smoke_test loads
                           (see inc/eshkol/backend/xla/pjrt_client.h).

Exit status: 0 only if every requested stage emitted PASS. Any FAIL, or an
unrecognized/missing argument, exits non-zero.
EOF
}

if [ "$#" -eq 0 ]; then
    usage >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 2

# shellcheck source=lib/harness_outcome.sh
. "$REPO_ROOT/scripts/lib/harness_outcome.sh"

TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/xla.jsonl"
mkdir -p "$TRACE_DIR"
: >> "$TRACE_FILE"  # create if absent; never truncate — see header.

# Durable working root. NEVER /tmp: everything this script creates lives
# under the repo, matching the convention scripts/run_icc_smoke.sh and
# scripts/run_wasm_differential.sh already use for `.scratch/`.
SCRATCH_ROOT="$REPO_ROOT/.scratch/xla_gate"
mkdir -p "$SCRATCH_ROOT"

STABLEHLO_ROOT="${STABLEHLO_ROOT:-$REPO_ROOT/deps/stablehlo}"
BUILD_DIR="${XLA_GATE_BUILD_DIR:-$SCRATCH_ROOT/build}"

GATE_FAILED=0

# emit_stage <event_name> <PASS|FAIL> <snippet>
emit_stage() {
    local name="$1" value="$2" snippet="$3"
    eshkol_outcome_emit_event "$TRACE_FILE" "xla" "$name" "$value" "$snippet"
    if [ "$value" = "PASS" ]; then
        printf '  PASS %-40s %s\n' "$name" "$snippet"
    else
        GATE_FAILED=1
        printf '  FAIL %-40s %s\n' "$name" "$snippet"
    fi
}

# tail_for_snippet <file> — bounded, single-line-safe excerpt for a snippet.
tail_for_snippet() {
    tr '\n' ' ' < "$1" 2>/dev/null | tail -c 400
}

# ─────────────────────────────────────────────────────────────────────────
# Stage 0 — baseline
# ─────────────────────────────────────────────────────────────────────────
stage_baseline() {
    local name="xla_backend_builds_and_baseline_recorded"
    local log="$SCRATCH_ROOT/baseline-configure.log"

    if [ ! -d "$STABLEHLO_ROOT" ]; then
        emit_stage "$name" FAIL \
            "STABLEHLO_ROOT ($STABLEHLO_ROOT) does not exist; run scripts/build_stablehlo.sh first"
        return
    fi

    mkdir -p "$BUILD_DIR"

    if ! cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
            -DCMAKE_BUILD_TYPE=Release \
            -DESHKOL_XLA_ENABLED=ON \
            -DSTABLEHLO_ROOT="$STABLEHLO_ROOT" \
            > "$log" 2>&1; then
        emit_stage "$name" FAIL "cmake configure failed: $(tail_for_snippet "$log")"
        return
    fi

    local build_log="$SCRATCH_ROOT/baseline-build.log"
    if ! cmake --build "$BUILD_DIR" \
            --target eshkol-run stdlib xla_codegen_test pjrt_smoke_test \
            --parallel \
            > "$build_log" 2>&1; then
        emit_stage "$name" FAIL "cmake --build failed: $(tail_for_snippet "$build_log")"
        return
    fi

    if [ ! -x "$BUILD_DIR/xla_codegen_test" ]; then
        emit_stage "$name" FAIL \
            "build reported success but $BUILD_DIR/xla_codegen_test was not produced"
        return
    fi

    local test_log="$SCRATCH_ROOT/baseline-xla_codegen_test.log"
    if ! "$BUILD_DIR/xla_codegen_test" > "$test_log" 2>&1; then
        emit_stage "$name" FAIL "xla_codegen_test exited non-zero: $(tail_for_snippet "$test_log")"
        return
    fi

    local pjrt_note="pjrt_smoke_test not built"
    if [ -x "$BUILD_DIR/pjrt_smoke_test" ]; then
        pjrt_note="pjrt_smoke_test built"
    fi

    local passed
    passed="$(grep -o 'Passed: [0-9]*' "$test_log" | tail -1)"
    emit_stage "$name" PASS \
        "ESHKOL_XLA_ENABLED=ON build OK; xla_codegen_test exit 0 ($passed); $pjrt_note"
}

# ─────────────────────────────────────────────────────────────────────────
# Stage 1 — PJRT-CPU roundtrip
#
# HONEST SCOPE: pjrt_smoke_test (tests/xla/pjrt_smoke_test.cpp) proves
# connectivity only — plugin discovery/load, client creation, platform name,
# device enumeration. The actual criterion asks for a compiled StableHLO
# module EXECUTING through PJRT-CPU and matching the LLVM-direct path
# bit-identically. No such compile()+execute()+diff harness exists yet
# (PjrtClient::compile()/execute() are wired in pjrt_client.h/.cpp, but
# nothing in this repo calls them end-to-end against a real CPU plugin and
# compares to xla_runtime.cpp's LLVM-direct execution path). So this stage
# can, at best, report FAIL with real connectivity evidence in the snippet —
# it must never report PASS until that comparison harness exists.
# ─────────────────────────────────────────────────────────────────────────
stage_pjrt_cpu() {
    local name="xla_pjrt_cpu_roundtrip"

    if [ ! -x "$BUILD_DIR/pjrt_smoke_test" ]; then
        emit_stage "$name" FAIL \
            "$BUILD_DIR/pjrt_smoke_test not built — run --baseline first"
        return
    fi

    # Best-effort CPU PJRT plugin discovery. findPjrtPlugin() in
    # pjrt_client.cpp only searches for a TPU plugin by default (or an exact
    # ESHKOL_PJRT_PLUGIN_PATH override, which wins regardless of backend) —
    # there is no CPU-plugin search path in that function today. These
    # candidates are best-effort locations a CPU PJRT plugin might have been
    # installed to (e.g. via a jax/jaxlib wheel); none are verified to exist
    # on any particular host.
    local plugin_path="${ESHKOL_PJRT_PLUGIN_PATH:-}"
    if [ -z "$plugin_path" ]; then
        local candidate
        for candidate in \
            "$HOME"/.local/lib/python3.1[0-9]/site-packages/jaxlib/cpu_plugin.so \
            "$HOME"/.local/lib/python3.1[0-9]/site-packages/jax_plugins/xla_cpu/xla_cpu_pjrt_plugin.so \
            /usr/lib/pjrt/pjrt_c_api_cpu_plugin.so; do
            if [ -f "$candidate" ]; then
                plugin_path="$candidate"
                break
            fi
        done
    fi

    local log="$SCRATCH_ROOT/pjrt-cpu.log"
    if [ -z "$plugin_path" ]; then
        ESHKOL_XLA_GATE_SCRATCH_DIR="$SCRATCH_ROOT/pjrt_smoke_scratch" \
            "$BUILD_DIR/pjrt_smoke_test" > "$log" 2>&1
        local rc=$?
        if [ "$rc" -eq 77 ]; then
            emit_stage "$name" FAIL \
                "stage not implemented: no CPU PJRT plugin found on this host (searched jaxlib/jax_plugins well-known locations; set ESHKOL_PJRT_PLUGIN_PATH to force one), AND no compile()/execute()/parity harness exists yet even when one is found"
        else
            emit_stage "$name" FAIL \
                "no CPU PJRT plugin found, and pjrt_smoke_test's own negative-control/other checks did not cleanly SKIP either (exit $rc): $(tail_for_snippet "$log")"
        fi
        return
    fi

    ESHKOL_PJRT_PLUGIN_PATH="$plugin_path" \
        ESHKOL_XLA_GATE_SCRATCH_DIR="$SCRATCH_ROOT/pjrt_smoke_scratch" \
        "$BUILD_DIR/pjrt_smoke_test" > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        emit_stage "$name" FAIL \
            "connectivity check against $plugin_path failed (pjrt_smoke_test exit $rc): $(tail_for_snippet "$log")"
        return
    fi

    # Connectivity succeeded — the full criterion (compile+execute,
    # bit-identical to the LLVM-direct path) still is not. Honest FAIL.
    emit_stage "$name" FAIL \
        "stage not implemented: PJRT-CPU connectivity verified via pjrt_smoke_test against $plugin_path ($(tail_for_snippet "$log")), but compiling a StableHLO module through PJRT_Client_Compile/Execute and diffing against the LLVM-direct path is not implemented anywhere in this repo yet"
}

# ─────────────────────────────────────────────────────────────────────────
# Stages 2-7 — none of these have any implementation to exercise yet: no
# differential harness against qllm_manifold_*, no training-step harness, no
# sharding/GSPMD wiring, no bf16 numerics sweep, no production deploy check.
# Each emits FAIL with a specific reason naming exactly what is missing, per
# the honesty contract at the top of this file.
# ─────────────────────────────────────────────────────────────────────────
stage_not_implemented() {
    local name="$1" reason="$2"
    emit_stage "$name" FAIL "stage not implemented: $reason"
}

stage_op_parity() {
    stage_not_implemented "xla_op_surface_parity" \
        "no differential harness exists comparing StableHLO op execution (Gather/Scatter, DynamicSlice/DynamicUpdateSlice, masking, assembly, sampling) against the CPU/CUDA path"
}

stage_geometric_sweep() {
    stage_not_implemented "xla_geometric_parity" \
        "no dimension-swept comparison exists between StableHLO-lowered hyperbolic/spherical/euclidean ops and qllm_manifold_*"
}

stage_training_step() {
    stage_not_implemented "xla_training_step_parity" \
        "no full training-step (forward+backward+optimizer) harness exists comparing the PJRT path against the CUDA path"
}

stage_multidevice() {
    stage_not_implemented "xla_multidevice_step" \
        "no GSPMD/sharding wiring or multi-device execution exists to compare against a single-device result"
}

stage_numerics() {
    stage_not_implemented "xla_bf16_numerics_bounded" \
        "no bf16 error-bound sweep exists, including the hyperbolic-boundary-near-precision-loss case called out in the oracle label"
}

stage_production() {
    stage_not_implemented "xla_tpu_production_ready" \
        "no GKE/Vertex TPU deployment, preemption-survival, or checkpoint-authority verification exists"
}

# ─────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────
STAGES=()
for arg in "$@"; do
    case "$arg" in
        --baseline)        STAGES+=(stage_baseline) ;;
        --pjrt-cpu)         STAGES+=(stage_pjrt_cpu) ;;
        --op-parity)        STAGES+=(stage_op_parity) ;;
        --geometric-sweep)  STAGES+=(stage_geometric_sweep) ;;
        --training-step)    STAGES+=(stage_training_step) ;;
        --multidevice)      STAGES+=(stage_multidevice) ;;
        --numerics)         STAGES+=(stage_numerics) ;;
        --production)       STAGES+=(stage_production) ;;
        --all)
            STAGES+=(stage_baseline stage_pjrt_cpu stage_op_parity \
                      stage_geometric_sweep stage_training_step \
                      stage_multidevice stage_numerics stage_production)
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "run_xla_gate.sh: unrecognized argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

echo "Running XLA gate stages -> $TRACE_FILE"
echo

for stage_fn in "${STAGES[@]}"; do
    "$stage_fn"
done

echo
if [ "$GATE_FAILED" -ne 0 ]; then
    echo "run_xla_gate.sh: one or more stages FAILED" >&2
    exit 1
fi
echo "run_xla_gate.sh: all requested stages PASSED"
exit 0
