#!/usr/bin/env bash
# run_xla_gate.sh — gate script for .icc/completion-oracles.yaml's
# `xla-tpu-ready` target. Each stage below corresponds to exactly one
# `requires:` criterion in that target; the action: line for each criterion
# names one of this script's flags.
#
#   Stage 0 (--baseline)       -> xla_backend_builds_and_baseline_recorded
#   Stage 1 (--pjrt-cpu)       -> xla_pjrt_cpu_roundtrip
#   Stage 2 (--op-parity)      -> xla_op_surface_parity
#   Stage 2b (--gradients)     -> xla_device_gradient_parity
#   Stage 3 (--geometric-sweep)-> xla_geometric_parity
#   Stage 4 (--training-step)  -> xla_training_step_parity
#   Stage 5 (--multidevice)    -> xla_multidevice_step
#   Stage 6 (--numerics)       -> xla_bf16_numerics_bounded
#   Stage 7 (--production)     -> xla_tpu_production_ready
#   Fragment (--fragment-coverage) -> stablehlo_fragment_contract_present,
#                                     stablehlo_builtin_classification_complete,
#                                     stablehlo_device_builtin_parity
#   Regions  (--region-formation)  -> xla_region_formation_outlines_maximal,
#                                     xla_region_formation_breaks_reported,
#                                     xla_region_formation_result_parity
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
  --pjrt-cpu         Compile and execute a StableHLO module through
                      pjrt_roundtrip_test against whatever PJRT plugin this
                      host provides (a CPU plugin where one is installed, the
                      TPU plugin on the dev node).
                      -> xla_pjrt_cpu_roundtrip
  --op-parity        Run tests/xla/op_parity_test: every lowered tensor op
                      computed on the PJRT device and on the host runtime from
                      the same inputs, compared against the per-dtype tolerance
                      in docs/design/ESHKOL_S_FRAGMENT.md.
                      -> xla_op_surface_parity
  --gradients        Run tests/xla/gradient_parity_test: the reverse-mode VJP
                      of every lowered op, emitted into StableHLO and executed
                      on the PJRT device, against the host's own reverse-mode
                      AD entry points and a central finite difference over the
                      host forward; plus a two-layer composite and the
                      sphere_project golden Jacobian. Also runs
                      tests/xla/host_max_tie_convention.esk and requires the
                      host max/min tie convention it measures to be the one the
                      device VJP implements.
                      -> xla_device_gradient_parity
  --geometric-sweep  Run tests/xla/geometric_parity_test: every hyperbolic,
                      spherical and Euclidean primitive lowered as a StableHLO
                      composition and executed on the PJRT device, forward and
                      reverse, at d in {2,4,16,64}, against the same
                      decomposition on the host tensor runtime, a finite
                      difference over it, the committed golden Jacobians in
                      tests/qllm_oracle/golden, the host AD tape, and manifold
                      identities computed from device outputs alone.
                      -> xla_geometric_parity
  --training-step    Full training step (fwd/bwd/optimizer) vs CUDA path.
                      -> xla_training_step_parity
  --multidevice      Sharded training step across >=2 devices.
                      -> xla_multidevice_step
  --numerics         bf16 error bounds across a dimension sweep.
                      -> xla_bf16_numerics_bounded
  --production       TPU production deploy/preemption/checkpoint survival.
  --fragment-coverage  Eshkol-S contract present, all 204 builtins classified, device builtins at parity.
  --region-formation   Maximal Eshkol-S subgraphs outlined, graph breaks reported, result parity.
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
    # geometric_parity_test is built here so that --geometric-sweep has a
    # binary to run without a second configure, exactly as the other parity
    # harnesses are.
    #
    # eshkol-vm-standalone-test is built here for a different reason: the AD
    # exactness gate's VM leg SKIPS when that binary is absent, and a skip in a
    # gate reads as a pass to everything downstream. Building it in the XLA
    # baseline is what stops that leg from silently not running on this node.
    if ! cmake --build "$BUILD_DIR" \
            --target eshkol-run stdlib xla_codegen_test pjrt_smoke_test \
                     pjrt_roundtrip_test op_parity_test gradient_parity_test \
                     geometric_parity_test builtin_parity_test \
                     region_formation_test eshkol-vm-standalone-test \
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
    local geo_note="geometric_parity_test not built"
    if [ -x "$BUILD_DIR/geometric_parity_test" ]; then
        geo_note="geometric_parity_test built"
    fi
    local vm_note="eshkol-vm-standalone-test not built"
    if [ -x "$BUILD_DIR/eshkol-vm-standalone-test" ]; then
        vm_note="eshkol-vm-standalone-test built"
    fi

    local passed
    passed="$(grep -o 'Passed: [0-9]*' "$test_log" | tail -1)"
    emit_stage "$name" PASS \
        "ESHKOL_XLA_ENABLED=ON build OK; xla_codegen_test exit 0 ($passed); $pjrt_note; $geo_note; $vm_note"
}

# ─────────────────────────────────────────────────────────────────────────
# Stage 1 — PJRT round trip
#
# Runs tests/xla/pjrt_roundtrip_test.cpp: it builds a StableHLO module with
# StableHLOEmitter, compiles it through PjrtClient::compile(), transfers real
# inputs to the device, executes, reads the result back, and checks it
# element-by-element against a hand-computed answer — for an elementwise add
# and a matmul — then runs a negative control that asserts compile() refuses
# a deliberately malformed module. This is a genuine compile-and-execute
# round trip, not connectivity-only (that remains pjrt_smoke_test's job).
#
# The criterion name (xla_pjrt_cpu_roundtrip) predates this rewrite and is
# kept unchanged because the oracle, roadmap goals, and design record
# reference it, but the round trip itself runs against whatever PJRT plugin
# the host actually provides: a CPU plugin where one is installed (discovery
# below), and the TPU plugin on the dev node (pjrt_roundtrip_test's own
# findPjrtPlugin("tpu") search, which pjrt_client.cpp implements).
# ─────────────────────────────────────────────────────────────────────────
stage_pjrt_cpu() {
    local name="xla_pjrt_cpu_roundtrip"

    if [ ! -x "$BUILD_DIR/pjrt_roundtrip_test" ]; then
        emit_stage "$name" FAIL \
            "$BUILD_DIR/pjrt_roundtrip_test not built — build the pjrt_roundtrip_test target first"
        return
    fi

    # Best-effort CPU PJRT plugin discovery, kept as one of the ways a plugin
    # is found. findPjrtPlugin() in pjrt_client.cpp only searches for a TPU
    # plugin by default (or an exact ESHKOL_PJRT_PLUGIN_PATH override, which
    # wins regardless of backend) — there is no CPU-plugin search path in
    # that function today. These candidates are best-effort locations a CPU
    # PJRT plugin might have been installed to (e.g. via a jax/jaxlib wheel);
    # none are verified to exist on any particular host. When none is found
    # here, pjrt_roundtrip_test still runs and performs its own TPU search —
    # this is what makes the same stage do the right thing on a CPU-only host
    # and on the TPU dev node without a flag to tell it which.
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

    local log="$SCRATCH_ROOT/pjrt-roundtrip.log"
    if [ -n "$plugin_path" ]; then
        ESHKOL_PJRT_PLUGIN_PATH="$plugin_path" \
            "$BUILD_DIR/pjrt_roundtrip_test" > "$log" 2>&1
    else
        "$BUILD_DIR/pjrt_roundtrip_test" > "$log" 2>&1
    fi
    local rc=$?

    if [ "$rc" -eq 77 ]; then
        emit_stage "$name" FAIL "no PJRT plugin found on this host"
        return
    fi
    if [ "$rc" -ne 0 ]; then
        emit_stage "$name" FAIL "pjrt_roundtrip_test exited $rc: $(tail_for_snippet "$log")"
        return
    fi

    local summary_line
    summary_line="$(grep -o 'SUMMARY: .*' "$log" | tail -1)"
    emit_stage "$name" PASS \
        "pjrt_roundtrip_test compiled and executed a StableHLO add and matmul module through PJRT, verified both results, and confirmed compile() refuses a malformed module; $summary_line"
}

# ─────────────────────────────────────────────────────────────────────────
# Stages 3-7 — none of these has any implementation to exercise yet: no
# differential harness against qllm_manifold_*, no training-step harness, no
# sharding/GSPMD wiring, no bf16 numerics sweep, no production deploy check.
# Each emits FAIL with a specific reason naming exactly what is missing, per
# the honesty contract at the top of this file. (Stage 2, --op-parity, is
# implemented below and runs a real differential.)
# ─────────────────────────────────────────────────────────────────────────
stage_not_implemented() {
    local name="$1" reason="$2"
    emit_stage "$name" FAIL "stage not implemented: $reason"
}

# ─────────────────────────────────────────────────────────────────────────
# Shared: locate a PJRT plugin the same way stage_pjrt_cpu does.
#
# Factored out because two criteria now need it (xla_op_surface_parity and
# stablehlo_device_builtin_parity), and two copies of a search order is how
# the two stages would eventually end up testing different devices.
# ─────────────────────────────────────────────────────────────────────────
discover_pjrt_plugin() {
    if [ -n "${ESHKOL_PJRT_PLUGIN_PATH:-}" ]; then
        printf '%s' "$ESHKOL_PJRT_PLUGIN_PATH"
        return
    fi
    local candidate
    for candidate in \
        "$HOME"/.local/lib/python3.1[0-9]/site-packages/jaxlib/cpu_plugin.so \
        "$HOME"/.local/lib/python3.1[0-9]/site-packages/jax_plugins/xla_cpu/xla_cpu_pjrt_plugin.so \
        /usr/lib/pjrt/pjrt_c_api_cpu_plugin.so; do
        if [ -f "$candidate" ]; then
            printf '%s' "$candidate"
            return
        fi
    done
    printf ''
}

# run_op_parity_test <log-path>
#
# Runs tests/xla/op_parity_test, which computes every lowered op on the device
# and on the host runtime from the same inputs and compares them against the
# per-dtype tolerance in docs/design/ESHKOL_S_FRAGMENT.md. Echoes the exit
# status; the caller decides what each status means for its criterion.
#
#   0  every row agreed
#   1  a row disagreed, or the harness's own comparator control failed
#   2  the binary was not built
#   77 no PJRT device was reachable
run_op_parity_test() {
    local log="$1"
    if [ ! -x "$BUILD_DIR/op_parity_test" ]; then
        return 2
    fi
    local plugin_path
    plugin_path="$(discover_pjrt_plugin)"
    if [ -n "$plugin_path" ]; then
        ESHKOL_PJRT_PLUGIN_PATH="$plugin_path" nice -n 19 "$BUILD_DIR/op_parity_test" > "$log" 2>&1
    else
        nice -n 19 "$BUILD_DIR/op_parity_test" > "$log" 2>&1
    fi
    return $?
}

# run_builtin_parity_test <log-path>
#
# Two steps, in this order and not the other:
#   1. scripts/gen_builtin_parity_plan.py runs every builtin in
#      lib/backend/xla/device_lowering_table.yaml through eshkol-run with the
#      device switch OFF and records what the language printed;
#   2. tests/xla/builtin_parity_test executes each builtin's StableHLO lowering
#      on the device and compares against those recorded values.
#
# The reference therefore comes from the language and the device value from a
# StableHLO graph -- two independent code paths. Generating the plan inside the
# harness would have let one program produce both, which is the arrangement
# under which a parity row proves nothing.
#
#   0  every row agreed        1  a row disagreed or a control failed
#   2  the binary was not built or the plan could not be generated
#   77 no PJRT device was reachable
run_builtin_parity_test() {
    local log="$1"
    if [ ! -x "$BUILD_DIR/builtin_parity_test" ]; then
        return 2
    fi
    local plan="$SCRATCH_ROOT/builtin_parity_plan.txt"
    if ! nice -n 19 python3 "$REPO_ROOT/scripts/gen_builtin_parity_plan.py" \
            --eshkol-run "$BUILD_DIR/eshkol-run" \
            --out "$plan" \
            --work "$SCRATCH_ROOT" > "$log" 2>&1; then
        return 2
    fi
    local plugin_path
    plugin_path="$(discover_pjrt_plugin)"
    if [ -n "$plugin_path" ]; then
        ESHKOL_PJRT_PLUGIN_PATH="$plugin_path" nice -n 19 \
            "$BUILD_DIR/builtin_parity_test" "$plan" >> "$log" 2>&1
    else
        nice -n 19 "$BUILD_DIR/builtin_parity_test" "$plan" >> "$log" 2>&1
    fi
    return $?
}

# ─────────────────────────────────────────────────────────────────────────
# Stage 2 — op surface parity
#
# PASS requires an actual device, an actual StableHLO compile per op, and
# every row within tolerance. No device is FAIL, not a skip: "the ops agree
# on device and host" is not a claim that can be made without a device.
# ─────────────────────────────────────────────────────────────────────────
stage_op_parity() {
    local name="xla_op_surface_parity"
    local log="$SCRATCH_ROOT/op-parity.log"

    run_op_parity_test "$log"
    local rc=$?

    case "$rc" in
        2)
            emit_stage "$name" FAIL \
                "$BUILD_DIR/op_parity_test not built — run --baseline first"
            return
            ;;
        77)
            emit_stage "$name" FAIL \
                "no PJRT device reachable on this host, so no device/host differential could be measured: $(tail_for_snippet "$log")"
            return
            ;;
        0) ;;
        *)
            emit_stage "$name" FAIL "op_parity_test exited $rc: $(tail_for_snippet "$log")"
            return
            ;;
    esac

    local summary
    summary="$(grep -o 'SUMMARY: .*' "$log" | tail -1)"
    if [ -z "$summary" ]; then
        emit_stage "$name" FAIL \
            "op_parity_test exited 0 but emitted no SUMMARY line, so nothing was measured"
        return
    fi
    emit_stage "$name" PASS \
        "device/host differential over every lowered StableHLO op, within the per-dtype tolerance of docs/design/ESHKOL_S_FRAGMENT.md; $summary"
}

# ─────────────────────────────────────────────────────────────────────────
# Stage 2b — device gradient parity
#
# PASS requires an actual device, an actual StableHLO compile of a module whose
# BACKWARD pass is device code, and every row within tolerance — including the
# two-layer composite and the sphere_project golden Jacobian rows.
#
# It also requires the tie convention to agree across the two AD paths. The
# host language AD is asked directly, through eshkol-run on
# tests/xla/host_max_tie_convention.esk, and the answer it prints must be the
# convention gradient_parity_test requires of the device. A gradient that
# depends on which of the two evaluated it is exactly the failure a device
# backward introduces, and it is invisible in every row without a tie, so it is
# checked here rather than assumed.
#
# No device is FAIL, not a skip: "the device gradients agree with the host" is
# not a claim that can be made without a device.
# ─────────────────────────────────────────────────────────────────────────
stage_gradients() {
    local name="xla_device_gradient_parity"
    local log="$SCRATCH_ROOT/gradient-parity.log"
    local tie_log="$SCRATCH_ROOT/gradient-tie-convention.log"

    if [ ! -x "$BUILD_DIR/gradient_parity_test" ]; then
        emit_stage "$name" FAIL \
            "$BUILD_DIR/gradient_parity_test not built — run --baseline first"
        return
    fi

    # The tie convention, measured through the host language AD. Required, not
    # advisory: without it the device convention is only this repository's
    # reading of lib/backend/autodiff_codegen.cpp.
    local tie_convention=""
    if [ ! -x "$BUILD_DIR/eshkol-run" ]; then
        emit_stage "$name" FAIL \
            "$BUILD_DIR/eshkol-run not built, so the host max/min tie convention could not be measured"
        return
    fi
    # -r JIT-executes. Without it eshkol-run COMPILES the file to ./a.out and
    # exits 0 without running anything, which would leave this stage grading an
    # empty log — a gate that measures nothing and says PASS.
    ( cd "$REPO_ROOT" && nice -n 19 "$BUILD_DIR/eshkol-run" -r \
        "$REPO_ROOT/tests/xla/host_max_tie_convention.esk" ) > "$tie_log" 2>&1
    local tie_rc=$?
    if [ "$tie_rc" -ne 0 ]; then
        emit_stage "$name" FAIL \
            "host_max_tie_convention.esk exited $tie_rc: $(tail_for_snippet "$tie_log")"
        return
    fi
    if ! grep -q "host_max_tie_convention: ALL PASS" "$tie_log"; then
        emit_stage "$name" FAIL \
            "host_max_tie_convention.esk did not report ALL PASS, so the convention it printed is not trustworthy: $(tail_for_snippet "$tie_log")"
        return
    fi
    tie_convention="$(grep -o 'HOST_MAX_TIE_CONVENTION: .*' "$tie_log" | tail -1 | awk '{print $2}')"
    local tie_convention_min
    tie_convention_min="$(grep -o 'HOST_MIN_TIE_CONVENTION: .*' "$tie_log" | tail -1 | awk '{print $2}')"
    if [ "$tie_convention" != "last-index-wins" ] || [ "$tie_convention_min" != "last-index-wins" ]; then
        emit_stage "$name" FAIL \
            "the host AD tie convention is max='$tie_convention' min='$tie_convention_min', but the device VJP in lib/backend/xla/stablehlo_emitter.cpp implements last-index-wins; they must be the same convention or a gradient depends on where it ran"
        return
    fi

    local plugin_path
    plugin_path="$(discover_pjrt_plugin)"
    if [ -n "$plugin_path" ]; then
        ESHKOL_PJRT_PLUGIN_PATH="$plugin_path" nice -n 19 "$BUILD_DIR/gradient_parity_test" \
            > "$log" 2>&1
    else
        nice -n 19 "$BUILD_DIR/gradient_parity_test" > "$log" 2>&1
    fi
    local rc=$?

    case "$rc" in
        77)
            emit_stage "$name" FAIL \
                "no PJRT device reachable on this host, so no device/host gradient differential could be measured: $(tail_for_snippet "$log")"
            return
            ;;
        0) ;;
        *)
            emit_stage "$name" FAIL "gradient_parity_test exited $rc: $(tail_for_snippet "$log")"
            return
            ;;
    esac

    local summary
    summary="$(grep -o 'SUMMARY: .*' "$log" | tail -1)"
    if [ -z "$summary" ]; then
        emit_stage "$name" FAIL \
            "gradient_parity_test exited 0 but emitted no SUMMARY line, so nothing was measured"
        return
    fi
    # A summary that reports a failed composite or golden leg must not pass,
    # even if the process exited 0 — the exit status and the summary have to
    # agree, and a gate that trusts only one of them can be satisfied by a
    # harness that stopped grading.
    case "$summary" in
        *composite=FAIL*|*golden_sphere_project=FAIL*|*rows_passed=0*)
            emit_stage "$name" FAIL "gradient_parity_test exited 0 but its summary reports a failure: $summary"
            return
            ;;
    esac

    emit_stage "$name" PASS \
        "reverse-mode VJPs of every lowered op executed on the PJRT device and agreed with the host AD reference and its finite-difference cross-check; two-layer composite and sphere_project golden Jacobian rows agreed; host tie convention measured as $tie_convention; $summary"
}

# ─────────────────────────────────────────────────────────────────────────
# Stage 3 — geometric parity
#
# Runs tests/xla/geometric_parity_test: every hyperbolic, spherical and
# Euclidean primitive lowered as a StableHLO composition and executed on the
# PJRT device, forward and reverse, at d in {2,4,16,64}, against
#
#   - the same decomposition evaluated by the HOST tensor runtime,
#   - a central finite difference over that host composition,
#   - the committed golden Jacobians in tests/qllm_oracle/golden (read at run
#     time, so a regenerated corpus is picked up rather than drifting away
#     from a transcription),
#   - the live host AD tape for the three primitives that have bridge nodes,
#   - and manifold identities computed only from device outputs.
#
# PASS requires a device, every row and every identity within tolerance, and a
# non-zero number of golden cases actually graded. The last of those matters:
# every golden case can be excluded for f32 representability, and a row that
# graded nothing must not report that it agreed.
#
# The test is run from the repository root because it reads the golden corpus
# by relative path (ESHKOL_GOLDEN_DIR overrides it).
# ─────────────────────────────────────────────────────────────────────────
stage_geometric_sweep() {
    local name="xla_geometric_parity"
    local log="$SCRATCH_ROOT/geometric-parity.log"

    if [ ! -x "$BUILD_DIR/geometric_parity_test" ]; then
        emit_stage "$name" FAIL \
            "$BUILD_DIR/geometric_parity_test not built — run --baseline first"
        return
    fi

    local plugin_path
    plugin_path="$(discover_pjrt_plugin)"
    if [ -n "$plugin_path" ]; then
        ( cd "$REPO_ROOT" && ESHKOL_PJRT_PLUGIN_PATH="$plugin_path" \
            nice -n 19 "$BUILD_DIR/geometric_parity_test" ) > "$log" 2>&1
    else
        ( cd "$REPO_ROOT" && nice -n 19 "$BUILD_DIR/geometric_parity_test" ) > "$log" 2>&1
    fi
    local rc=$?

    case "$rc" in
        77)
            emit_stage "$name" FAIL \
                "no PJRT device reachable on this host, so no geometric differential could be measured: $(tail_for_snippet "$log")"
            return
            ;;
        0) ;;
        *)
            emit_stage "$name" FAIL "geometric_parity_test exited $rc: $(tail_for_snippet "$log")"
            return
            ;;
    esac

    local summary
    summary="$(grep -o 'SUMMARY: .*' "$log" | tail -1)"
    if [ -z "$summary" ]; then
        emit_stage "$name" FAIL \
            "geometric_parity_test exited 0 but emitted no SUMMARY line, so nothing was measured"
        return
    fi
    # The exit status and the summary must agree. A harness that stopped
    # grading would exit 0 with a summary that says so, and a gate that reads
    # only the exit status would call that a pass.
    case "$summary" in
        *rows_failed=0*) ;;
        *)
            emit_stage "$name" FAIL "geometric_parity_test exited 0 but rows failed: $summary"
            return
            ;;
    esac
    case "$summary" in
        *invariants_failed=0*) ;;
        *)
            emit_stage "$name" FAIL \
                "a manifold identity failed, so a lowering is wrong in a way the host comparison did not see: $summary"
            return
            ;;
    esac
    case "$summary" in
        *rows_passed=0*|*golden_cases=0*|*invariants_passed=0*)
            emit_stage "$name" FAIL \
                "geometric_parity_test graded nothing (no rows, no golden cases, or no identities): $summary"
            return
            ;;
    esac
    if ! grep -q "GEOMETRIC PARITY: PASS" "$log"; then
        emit_stage "$name" FAIL \
            "geometric_parity_test did not report GEOMETRIC PARITY: PASS: $(tail_for_snippet "$log")"
        return
    fi

    emit_stage "$name" PASS \
        "hyperbolic, spherical and Euclidean primitives lowered as StableHLO compositions and executed on the PJRT device, forward and reverse, at d in {2,4,16,64} and three curvatures through one executable per shape; graded against the host composition, a finite difference over it, the tests/qllm_oracle/golden Jacobians, the host AD tape, and manifold identities computed from device outputs alone; $summary"
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
# Whole-language stages. These grade the stablehlo-fragment-coverage and
# xla-region-formation oracles rather than xla-tpu-ready. Same honesty
# contract: each criterion emits FAIL naming exactly what is missing until
# the thing exists. Nothing here may emit PASS on the strength of a plan.
# ─────────────────────────────────────────────────────────────────────────
stage_fragment_coverage() {
    local contract_name="stablehlo_fragment_contract_present"
    local classify_name="stablehlo_builtin_classification_complete"
    local parity_name="stablehlo_device_builtin_parity"
    local contract_file="$REPO_ROOT/docs/design/ESHKOL_S_FRAGMENT.md"

    # ── stablehlo_fragment_contract_present ──
    # PASS requires the file to exist AND to actually name all three labels
    # and state the parity rule (with a per-dtype tolerance) — presence of
    # the file alone is not checked as sufficient, per the S2a brief.
    if [ ! -f "$contract_file" ]; then
        emit_stage "$contract_name" FAIL \
            "contract file not found: docs/design/ESHKOL_S_FRAGMENT.md"
    else
        local missing=()
        grep -qF '`device`' "$contract_file" || missing+=("label 'device'")
        grep -qF '`host`' "$contract_file" || missing+=("label 'host'")
        grep -qF '`host-with-device-inner`' "$contract_file" || \
            missing+=("label 'host-with-device-inner'")
        grep -qi "parity rule" "$contract_file" || missing+=("a parity rule section")
        grep -qi "tolerance" "$contract_file" || missing+=("a stated tolerance")
        if [ "${#missing[@]}" -eq 0 ]; then
            emit_stage "$contract_name" PASS \
                "docs/design/ESHKOL_S_FRAGMENT.md names all three labels (device/host/host-with-device-inner) and states the parity rule with a per-dtype tolerance"
        else
            local joined
            joined="$(IFS=', '; echo "${missing[*]}")"
            emit_stage "$contract_name" FAIL \
                "docs/design/ESHKOL_S_FRAGMENT.md exists but is missing: $joined"
        fi
    fi

    # ── stablehlo_builtin_classification_complete ──
    # PASS only if the checker (which re-derives the registry from
    # tests/coverage/language_surface.json and diffs it against
    # lib/backend/xla/builtin_classification.yaml) exits 0. No build
    # required for this criterion.
    local checker_log="$SCRATCH_ROOT/builtin_classification_check.log"
    if nice -n 19 python3 "$REPO_ROOT/scripts/check_builtin_classification.py" \
            > "$checker_log" 2>&1; then
        emit_stage "$classify_name" PASS "$(tail_for_snippet "$checker_log")"
    else
        emit_stage "$classify_name" FAIL \
            "check_builtin_classification.py exited non-zero: $(tail_for_snippet "$checker_log")"
    fi

    # ── stablehlo_device_builtin_parity ──
    #
    # The join: op_parity_test prints a COVERED_BUILTINS line naming the
    # builtins its PASSING rows exercised; lib/backend/xla/builtin_classification.yaml
    # says which builtins are labelled device. PASS requires every
    # device-labelled builtin to be covered by a passing parity row. Anything
    # less is FAIL with the count, because "parity for the device builtins"
    # is a claim about all of them — a partial number is progress, not a pass.
    local parity_log="$SCRATCH_ROOT/device-builtin-parity.log"
    local yaml="$REPO_ROOT/lib/backend/xla/builtin_classification.yaml"

    run_op_parity_test "$parity_log"
    local parity_rc=$?

    if [ ! -f "$yaml" ]; then
        emit_stage "$parity_name" FAIL \
            "lib/backend/xla/builtin_classification.yaml not found; there is no device-builtin set to be at parity with"
        return
    fi

    local total_device
    total_device="$(nice -n 19 python3 - "$yaml" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
print(sum(1 for _ in re.finditer(r'^  ".*?":\n    label: device$', text, re.M)))
PY
)"

    if [ "$parity_rc" -eq 2 ]; then
        emit_stage "$parity_name" FAIL \
            "$BUILD_DIR/op_parity_test not built — run --baseline first; 0 of $total_device device-labelled builtins measured"
        return
    fi
    if [ "$parity_rc" -eq 77 ]; then
        emit_stage "$parity_name" FAIL \
            "no PJRT device reachable on this host; 0 of $total_device device-labelled builtins measured"
        return
    fi

    # The per-builtin harness covers what the eshkol_xla_* entry points cannot
    # reach; its passing rows join the op harness's.
    local builtin_log="$SCRATCH_ROOT/builtin-parity.log"
    run_builtin_parity_test "$builtin_log"
    local builtin_rc=$?

    local covered_line covered_count matched
    covered_line="$(grep -o '^COVERED_BUILTINS:.*' "$parity_log" | tail -1)"
    covered_line="$covered_line $(grep -o '^COVERED_BUILTINS:.*' "$builtin_log" | tail -1 | sed 's/^COVERED_BUILTINS://')"
    matched="$(COVERED="${covered_line#COVERED_BUILTINS:}" nice -n 19 python3 - "$yaml" <<'PY'
import os, re, sys
text = open(sys.argv[1]).read()
device = {m.group(1) for m in re.finditer(r'^  "(.*?)":\n    label: device$', text, re.M)}
covered = set(os.environ.get("COVERED", "").split())
print(len(covered & device))
PY
)"
    covered_count="${matched:-0}"

    if [ "$parity_rc" -ne 0 ]; then
        emit_stage "$parity_name" FAIL \
            "op_parity_test exited $parity_rc (a device/host row disagreed); $covered_count of $total_device device-labelled builtins covered by a passing row: $(tail_for_snippet "$parity_log")"
        return
    fi

    if [ "$covered_count" -ge "$total_device" ] && [ "$total_device" -gt 0 ]; then
        emit_stage "$parity_name" PASS \
            "every device-labelled builtin ($covered_count of $total_device) has a passing device/host parity row"
    else
        # Name what is still missing. A bare count says how far there is to go
        # but not what to do next; the list is what someone picks up.
        local uncovered builtin_note
        uncovered="$(COVERED="${covered_line#COVERED_BUILTINS:}" nice -n 19 python3 - "$yaml" <<'PY'
import os, re, sys
text = open(sys.argv[1]).read()
device = {m.group(1) for m in re.finditer(r'^  "(.*?)":\n    label: device$', text, re.M)}
covered = set(os.environ.get("COVERED", "").split())
missing = sorted(device - covered)
head = " ".join(missing[:20])
print(head + (" ... (+%d more)" % (len(missing) - 20) if len(missing) > 20 else ""))
PY
)"
        builtin_note=""
        if [ "$builtin_rc" -eq 2 ]; then
            builtin_note=" (the per-builtin harness did not run: not built, or the plan could not be generated)"
        elif [ "$builtin_rc" -eq 77 ]; then
            builtin_note=" (the per-builtin harness found no device)"
        elif [ "$builtin_rc" -ne 0 ]; then
            builtin_note=" (the per-builtin harness reported a disagreeing or degenerate row)"
        fi
        emit_stage "$parity_name" FAIL \
            "$covered_count of $total_device device-labelled builtins have a passing device/host parity row$builtin_note; still uncovered: $uncovered"
    fi
}

# Region formation (S5). Three criteria, three separate things that can be
# wrong, so three records:
#
#   outlines_maximal  the regions the pass formed are the ones the CONTRACT
#                     says are there, program by program, in
#                     tests/xla/regions/*.expected.json — files written from
#                     docs/design/ESHKOL_S_FRAGMENT.md, not from this pass's
#                     output. A file regenerated from the implementation
#                     grades nothing.
#   breaks_reported   every break the contract predicts is reported with the
#                     construct that caused it, and no break is reported that
#                     it does not predict.
#   result_parity     each corpus program run with regions ON produces the
#                     same numbers as the same program run entirely on the
#                     host.
#
# region_formation_test prints one line per criterion and exits non-zero when
# either of the first two disagrees. It is a HOST test: what it measures is
# decided before anything executes, so it runs on any machine.
stage_region_formation() {
    local outlines="xla_region_formation_outlines_maximal"
    local breaks="xla_region_formation_breaks_reported"
    local parity="xla_region_formation_result_parity"

    local bin="$BUILD_DIR/region_formation_test"
    if [ ! -x "$bin" ]; then
        emit_stage "$outlines" FAIL \
            "$bin not built -- run --baseline first"
        emit_stage "$breaks" FAIL \
            "$bin not built -- run --baseline first"
        emit_stage "$parity" FAIL \
            "$bin not built -- run --baseline first"
        return
    fi

    # A compiled-in eligibility table that has drifted from the YAML it was
    # generated from does not crash: it forms the wrong regions from stale
    # labels and reports breaks that are no longer real. So the staleness
    # check runs before the corpus, and a stale table fails the criterion
    # rather than being measured against.
    local tables_log="$SCRATCH_ROOT/region-tables-check.log"
    if ! python3 "$REPO_ROOT/scripts/check_region_tables.py" > "$tables_log" 2>&1; then
        emit_stage "$outlines" FAIL "$(tail_for_snippet "$tables_log")"
        emit_stage "$breaks" FAIL "region_tables.inc is stale; eligibility was decided from labels that no longer match the classification table"
        emit_stage "$parity" FAIL "region_tables.inc is stale; nothing measured"
        return
    fi

    local corpus="$REPO_ROOT/tests/xla/regions"
    local n_programs
    n_programs=$(ls -1 "$corpus"/*.esk 2>/dev/null | wc -l | tr -d ' ')
    if [ "${n_programs:-0}" -lt 12 ]; then
        # Twelve is the corpus size this stage was specified with. A gate that
        # accepted a corpus of one would pass by being asked nothing.
        emit_stage "$outlines" FAIL \
            "corpus has $n_programs programs; at least 12 of increasing host/device mixing are required"
        emit_stage "$breaks" FAIL "corpus has $n_programs programs; at least 12 are required"
        emit_stage "$parity" FAIL "corpus has $n_programs programs; at least 12 are required"
        return
    fi

    local report_dir="$SCRATCH_ROOT/region_reports"
    mkdir -p "$report_dir"
    local log="$SCRATCH_ROOT/region-formation.log"
    "$bin" --corpus "$corpus" --report-dir "$report_dir" > "$log" 2>&1
    local rc=$?

    local outlines_line breaks_line
    outlines_line=$(grep '^outlines_maximal:' "$log" | head -1)
    breaks_line=$(grep '^breaks_reported:' "$log" | head -1)

    if [ -z "$outlines_line" ] || [ -z "$breaks_line" ]; then
        emit_stage "$outlines" FAIL \
            "region_formation_test exited $rc without printing a verdict: $(tail_for_snippet "$log")"
        emit_stage "$breaks" FAIL \
            "region_formation_test exited $rc without printing a verdict: $(tail_for_snippet "$log")"
        emit_stage "$parity" FAIL \
            "region formation did not produce a verdict, so nothing was executed to compare"
        return
    fi

    case "$outlines_line" in
        *PASS*) emit_stage "$outlines" PASS \
                    "$n_programs corpus programs: $outlines_line" ;;
        *)      emit_stage "$outlines" FAIL \
                    "$outlines_line; $(grep -c '      outline:' "$log") disagreements, see $log" ;;
    esac
    case "$breaks_line" in
        *PASS*) emit_stage "$breaks" PASS \
                    "$n_programs corpus programs: $breaks_line" ;;
        *)      emit_stage "$breaks" FAIL \
                    "$breaks_line; $(grep -c '      break:' "$log") disagreements, see $log" ;;
    esac

    # ── result parity ──
    #
    # Two measurements, and the record needs both.
    #
    # First, region by region: every region of the corpus whose input shapes
    # are static, executed as one fused StableHLO module on the device and
    # compared against the host runtime's own entry points over the same
    # inputs. That covers everything about a region that can be numerically
    # wrong — the fusion, the marshalling, the read-back, every op inside it —
    # and it is how the executable-cache collision that returned one region's
    # numbers for another was found. It also reaches the regions the
    # whole-program run leaves on the host.
    local parity_log="$SCRATCH_ROOT/region-parity.log"
    local plugin_path
    plugin_path="$(discover_pjrt_plugin)"
    if [ -n "$plugin_path" ]; then
        ESHKOL_PJRT_PLUGIN_PATH="$plugin_path" nice -n 19 \
            "$bin" --corpus "$corpus" --parity > "$parity_log" 2>&1
    else
        nice -n 19 "$bin" --corpus "$corpus" --parity > "$parity_log" 2>&1
    fi
    local measured
    measured=$(grep '^region_device_host_parity:' "$parity_log" | head -1)
    if [ -z "$measured" ]; then
        measured="no region was executed on a device (no PJRT plugin reachable)"
    fi

    # ── whole-program parity ──
    #
    # The wiring the paragraph above said did not exist now does: generated
    # code calls eshkol_xla_region() in place of the subtree it outlined
    # (codegenRegionCall in lib/backend/llvm_codegen.cpp). So the criterion
    # is answered as asked: scripts/run_region_parity.sh runs every corpus
    # program twice, regions off and regions on, and compares the two
    # outputs numerically under the contract's f32 bounds with the
    # non-numeric skeleton required to match exactly. The record is PASS
    # only when BOTH measurements pass — the whole-program comparison, and
    # the region-by-region one above, which still covers the regions the
    # whole-program run leaves on the host (inside a gradient, or with
    # shapes only known at run time).
    local whole_log="$SCRATCH_ROOT/region-whole-program.log"
    XLA_GATE_BUILD_DIR="$BUILD_DIR" nice -n 19 \
        bash "$REPO_ROOT/scripts/run_region_parity.sh" "$corpus" > "$whole_log" 2>&1
    local whole
    whole=$(grep '^whole_program_region_parity:' "$whole_log" | head -1)
    if [ -z "$whole" ]; then
        whole="whole_program_region_parity: FAIL (no verdict: $(tail_for_snippet "$whole_log"))"
    fi

    case "$whole|$measured" in
        *"whole_program_region_parity: PASS"*"region_device_host_parity: PASS"*)
            emit_stage "$parity" PASS \
                "every corpus program run with regions on agrees with its regions-off run: $whole; and region by region on the device against the host runtime: $measured" ;;
        *)
            emit_stage "$parity" FAIL \
                "whole program: $whole; region by region on the device against the host runtime: $measured; see $whole_log and $parity_log" ;;
    esac
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
        --gradients)        STAGES+=(stage_gradients) ;;
        --geometric-sweep)  STAGES+=(stage_geometric_sweep) ;;
        --training-step)    STAGES+=(stage_training_step) ;;
        --multidevice)      STAGES+=(stage_multidevice) ;;
        --numerics)         STAGES+=(stage_numerics) ;;
        --production)       STAGES+=(stage_production) ;;
        --fragment-coverage) STAGES+=(stage_fragment_coverage) ;;
        --region-formation)  STAGES+=(stage_region_formation) ;;
        --all)
            STAGES+=(stage_baseline stage_pjrt_cpu stage_op_parity \
                      stage_gradients stage_geometric_sweep stage_training_step \
                      stage_multidevice stage_numerics stage_production \
                      stage_fragment_coverage stage_region_formation)
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
