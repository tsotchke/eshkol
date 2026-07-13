#!/usr/bin/env bash
# tests/gpu/gpu_correctness_gate.sh — GPU EXECUTION correctness gate.
#
# What this closes: the *-cuda lanes in .github/workflows/ci.yml run on
# GitHub-hosted GPU-less runners, so ESHKOL_GPU_ENABLED=ON only proves
# `nvcc`/Metal frameworks *compile* — no GPU kernel ever actually runs in
# CI. This script is the missing EXECUTION gate: on a machine with a real
# GPU (Metal on macOS, CUDA on Linux/Windows), it builds Eshkol twice —
# once with GPU acceleration on, once with it off — runs the identical
# differentiable workload (tests/gpu/gpu_correctness_gate.esk) through
# both, and diffs the numeric output within a tolerance. A mismatch means
# the GPU kernel path is producing wrong answers: a real bug, not a CI
# infra gap.
#
# On a machine with no GPU (the common case — every GitHub-hosted runner,
# a laptop with no discrete/integrated GPU framework support), this
# SKIPS cleanly: exit 0 with a "SKIP:" message. It never fails a run for
# lacking hardware it was never meant to require.
#
# Usage:
#   tests/gpu/gpu_correctness_gate.sh
#
# Env overrides:
#   BUILD_DIR_GPU   build dir for the GPU-enabled binary (default: build-gpu-gate)
#   BUILD_DIR_CPU   build dir for the GPU-disabled reference binary (default: build-gpu-gate-cpuref)
#   REUSE_BUILDS=1  skip (re)configuring/building if eshkol-run already exists
#                   in both dirs — for CI jobs that cache the build step
#   GPU_GATE_TOL    relative tolerance for the numeric diff (default: 1e-4;
#                   loose because eshkol's `display` prints floats with
#                   ~6 significant digits, not full round-trip precision —
#                   see the comment in gpu_correctness_gate.esk)
#   LLVM_CONFIG     path to llvm-config (auto-detected via `brew --prefix
#                   llvm@21` on macOS, or llvm-config-21/llvm-config on
#                   Linux and Git Bash/MSYS2 Windows, if unset)
#   LLVM_DIR        Windows LLVM SDK CMake directory containing LLVMConfig.cmake.
#   ESHKOL_HOST_CXX_COMPILER
#                   Windows LLVM SDK clang++.exe used for generated AOT links.
#   GPU_GATE_CMAKE_GENERATOR / GPU_GATE_CMAKE_PLATFORM / GPU_GATE_CMAKE_TOOLSET
#                   Optional Windows generator overrides. Defaults select the
#                   newest installed VS 2022+ generator, x64, and ClangCL.
set -u
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

BUILD_DIR_GPU="${BUILD_DIR_GPU:-build-gpu-gate}"
BUILD_DIR_CPU="${BUILD_DIR_CPU:-build-gpu-gate-cpuref}"
REUSE_BUILDS="${REUSE_BUILDS:-0}"
GPU_GATE_TOL="${GPU_GATE_TOL:-1e-4}"
GATE_ESK="$REPO_ROOT/tests/gpu/gpu_correctness_gate.esk"

# ICC evidence trace (.icc/completion-oracles.yaml::gpu-execution). Only
# written on an actual PASS/FAIL verdict — a SKIP (no GPU on this host)
# leaves no trace line, so the oracle correctly reports "no evidence yet"
# rather than a false PASS, on every GPU-less dev machine and hosted
# runner. See scripts/run_reference_differential.sh for the same
# kind/name/value/snippet JSON-L convention this reuses.
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/gpu_execution.jsonl"
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
    printf '{"kind":"gpu_execution","name":"gpu_execution_gate","value":"%s","snippet":%s,"confidence":0.95}\n' \
        "$value" "$esnip" >> "$TRACE_FILE"
}

log()  { printf '%s\n' "$*"; }
skip() { log "SKIP: $*"; exit 0; }
fail() { log "FAIL: $*"; emit_trace FAIL "$*"; exit 1; }

# ─────────────────────────────────────────────────────────────────
# Step 1: build-time GPU framework detection. No point configuring a
# full build if neither Metal nor a CUDA toolchain is present at all.
# ─────────────────────────────────────────────────────────────────
UNAME_S="$(uname -s)"
WINDOWS_POSIX=0
HAVE_GPU_FRAMEWORK=0
case "$UNAME_S" in
    Darwin)
        # Every Mac Eshkol supports has a Metal-capable GPU (integrated
        # or discrete); the runtime check in step 4 is what actually
        # proves a device answers, which is what matters on virtualized
        # hosted runners that may not pass a GPU through to Metal.
        if xcrun -sdk macosx --show-sdk-path >/dev/null 2>&1; then
            HAVE_GPU_FRAMEWORK=1
        fi
        ;;
    Linux|MINGW*|MSYS*|CYGWIN*)
        case "$UNAME_S" in MINGW*|MSYS*|CYGWIN*) WINDOWS_POSIX=1 ;; esac
        if command -v nvidia-smi >/dev/null 2>&1 || command -v nvcc >/dev/null 2>&1; then
            HAVE_GPU_FRAMEWORK=1
        fi
        ;;
    *)
        HAVE_GPU_FRAMEWORK=0
        ;;
esac

if [ "$HAVE_GPU_FRAMEWORK" -ne 1 ]; then
    skip "no GPU build framework detected on $UNAME_S (no Metal SDK / no nvidia-smi or nvcc) — nothing to execute"
fi

case "$UNAME_S" in
Linux|MINGW*|MSYS*|CYGWIN*)
    if command -v nvidia-smi >/dev/null 2>&1; then
        if ! nvidia-smi -L >/dev/null 2>&1 || [ -z "$(nvidia-smi -L 2>/dev/null)" ]; then
            skip "nvidia-smi present but reports no GPU device"
        fi
    elif [ -e /dev/nvhost-gpu ] || [ -e /dev/nvidiactl ] || [ -e /dev/nvidia0 ]; then
        # Jetson/L4T exposes the integrated GPU through nvhost device nodes
        # and intentionally does not ship the datacenter-oriented nvidia-smi
        # utility.  Treat this only as a coarse device hint: step 4 still
        # requires an actual Eshkol [GPU] dispatch record before issuing PASS.
        log "CUDA device node detected without nvidia-smi (Jetson/L4T path); runtime dispatch proof remains required"
    else
        skip "nvcc present but no NVIDIA device node or nvidia-smi GPU was found — CUDA toolchain without a runtime device (e.g. hosted compile-only CI)"
    fi
    ;;
esac

# ─────────────────────────────────────────────────────────────────
# Step 2: resolve LLVM (mirrors CMakeLists.txt / nix/jetson/build.sh's
# lite-build LLVM discovery).
# ─────────────────────────────────────────────────────────────────
if [ "$WINDOWS_POSIX" -eq 1 ] && [ -n "${LLVM_DIR:-}" ]; then
    [ -f "$LLVM_DIR/LLVMConfig.cmake" ] \
        || fail "LLVM_DIR=$LLVM_DIR does not contain LLVMConfig.cmake"
elif [ -z "${LLVM_CONFIG:-}" ]; then
    if [ "$UNAME_S" = "Darwin" ] && command -v brew >/dev/null 2>&1 && brew --prefix llvm@21 >/dev/null 2>&1; then
        LLVM_CONFIG="$(brew --prefix llvm@21)/bin/llvm-config"
    elif command -v llvm-config-21 >/dev/null 2>&1; then
        LLVM_CONFIG="$(command -v llvm-config-21)"
    elif command -v llvm-config >/dev/null 2>&1; then
        LLVM_CONFIG="$(command -v llvm-config)"
    else
        fail "could not locate llvm-config (LLVM 21) — set LLVM_CONFIG explicitly"
    fi
fi
if [ -n "${LLVM_CONFIG:-}" ]; then
    [ -x "$LLVM_CONFIG" ] || fail "LLVM_CONFIG=$LLVM_CONFIG is not executable"
fi

runner_path() {
    local build_dir="$1" candidate
    for candidate in \
        "$build_dir/eshkol-run" \
        "$build_dir/eshkol-run.exe" \
        "$build_dir/Release/eshkol-run.exe"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

configure_and_build() {
    local build_dir="$1" gpu_flag="$2"
    local runner=""
    runner="$(runner_path "$build_dir" 2>/dev/null || true)"
    if [ "$REUSE_BUILDS" = "1" ] && [ -n "$runner" ]; then
        log "  reusing $runner (REUSE_BUILDS=1)"
        return 0
    fi
    log "  configuring $build_dir (ESHKOL_GPU_ENABLED=$gpu_flag)..."
    local -a cmake_args=(
        -S "$REPO_ROOT"
        -B "$build_dir"
        -DCMAKE_BUILD_TYPE=Release
        -DESHKOL_REQUIRED_LLVM_MAJOR=21
        -DESHKOL_XLA_ENABLED=OFF
        -DESHKOL_GPU_ENABLED="$gpu_flag"
        -DESHKOL_BUILD_TESTS=OFF
    )

    if [ "$WINDOWS_POSIX" -eq 1 ]; then
        local generator="${GPU_GATE_CMAKE_GENERATOR:-}"
        if [ -z "$generator" ]; then
            if cmake --help 2>/dev/null | grep -q 'Visual Studio 18 2026'; then
                generator='Visual Studio 18 2026'
            elif cmake --help 2>/dev/null | grep -q 'Visual Studio 17 2022'; then
                generator='Visual Studio 17 2022'
            else
                fail "no supported Visual Studio 2022+ CMake generator found for Windows CUDA"
            fi
        fi
        cmake_args+=(
            -G "$generator"
            -A "${GPU_GATE_CMAKE_PLATFORM:-x64}"
            -T "${GPU_GATE_CMAKE_TOOLSET:-ClangCL}"
        )
        [ -n "${LLVM_DIR:-}" ] && cmake_args+=(-DLLVM_DIR="$LLVM_DIR")
        [ -n "${ESHKOL_HOST_CXX_COMPILER:-}" ] \
            && cmake_args+=(-DESHKOL_HOST_CXX_COMPILER="$ESHKOL_HOST_CXX_COMPILER")
        [ -n "${LLVM_CONFIG:-}" ] \
            && cmake_args+=(-DLLVM_CONFIG_EXECUTABLE="$LLVM_CONFIG")
    else
        cmake_args+=(-G Ninja -DLLVM_CONFIG_EXECUTABLE="$LLVM_CONFIG")
    fi

    cmake "${cmake_args[@]}" > "$build_dir.configure.log" 2>&1 \
        || { tail -n 60 "$build_dir.configure.log"; return 1; }

    if [ "$gpu_flag" = "ON" ] && ! grep -q "GPU acceleration: ENABLED" "$build_dir.configure.log"; then
        # Framework check in step 1 said yes, but CMake's own probe (which
        # actually tries to find_package(Metal)/find_package(CUDAToolkit))
        # disagrees — trust CMake and skip rather than fail the gate.
        return 2
    fi

    log "  building $build_dir..."
    cmake --build "$build_dir" --config Release --target eshkol-run --parallel \
        > "$build_dir.build.log" 2>&1 \
        || { tail -n 80 "$build_dir.build.log"; return 1; }
}

log "=== Eshkol GPU execution correctness gate ==="
log "Platform: $UNAME_S"
log ""
log "Building GPU-enabled reference ($BUILD_DIR_GPU)..."
configure_and_build "$BUILD_DIR_GPU" ON
rc=$?
if [ "$rc" -eq 2 ]; then
    skip "CMake could not find a usable GPU framework (Metal/CUDA) despite the coarse OS-level probe — nothing to execute"
fi
[ "$rc" -eq 0 ] || fail "GPU-enabled build failed — see $BUILD_DIR_GPU.build.log"

log "Building CPU-only reference ($BUILD_DIR_CPU)..."
configure_and_build "$BUILD_DIR_CPU" OFF || fail "CPU-only reference build failed — see $BUILD_DIR_CPU.build.log"

GPU_RUN="$(runner_path "$BUILD_DIR_GPU" 2>/dev/null || true)"
CPU_RUN="$(runner_path "$BUILD_DIR_CPU" 2>/dev/null || true)"
[ -n "$GPU_RUN" ] && GPU_RUN="$REPO_ROOT/$GPU_RUN"
[ -n "$CPU_RUN" ] && CPU_RUN="$REPO_ROOT/$CPU_RUN"
[ -x "$GPU_RUN" ] || fail "$GPU_RUN missing after build"
[ -x "$CPU_RUN" ] || fail "$CPU_RUN missing after build"

# ─────────────────────────────────────────────────────────────────
# Step 3: compile the shared payload with each binary.
# ─────────────────────────────────────────────────────────────────
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-gpu-gate.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

log ""
log "Compiling gate payload with both binaries..."
"$GPU_RUN" "$GATE_ESK" -o "$WORK_DIR/gate_gpu.bin" > "$WORK_DIR/gpu_compile.log" 2>&1 \
    || fail "GPU binary failed to compile $GATE_ESK — $(tail -n 20 "$WORK_DIR/gpu_compile.log")"
"$CPU_RUN" "$GATE_ESK" -o "$WORK_DIR/gate_cpu.bin" > "$WORK_DIR/cpu_compile.log" 2>&1 \
    || fail "CPU-reference binary failed to compile $GATE_ESK — $(tail -n 20 "$WORK_DIR/cpu_compile.log")"

GPU_PAYLOAD="$WORK_DIR/gate_gpu.bin"
CPU_PAYLOAD="$WORK_DIR/gate_cpu.bin"
[ -x "$GPU_PAYLOAD" ] || GPU_PAYLOAD="$GPU_PAYLOAD.exe"
[ -x "$CPU_PAYLOAD" ] || CPU_PAYLOAD="$CPU_PAYLOAD.exe"
[ -x "$GPU_PAYLOAD" ] || fail "GPU gate payload missing after successful compile"
[ -x "$CPU_PAYLOAD" ] || fail "CPU-reference gate payload missing after successful compile"

# ─────────────────────────────────────────────────────────────────
# Step 4: runtime GPU-presence check + forced-dispatch run. Verbose GPU
# logging is opt-in (ESHKOL_VERBOSE on Metal, ESHKOL_GPU_VERBOSE on
# CUDA) and prints an init banner ("[GPU] Metal: ..." / a CUDA device
# line) the moment eshkol_gpu_init() finds a real device, plus (on the
# forced sf64 dispatch kernel) an explicit per-call
# "[GPU] sf64 dispatch: ... completed: ...ms ...GFLOPS" line and (on
# CUDA) "-> CUDA cuBLAS" per matmul call. ESHKOL_GPU_THRESHOLD=1 forces
# every tensor op in the payload through the GPU path regardless of the
# VM's default dispatch-size heuristic.
# ─────────────────────────────────────────────────────────────────
log ""
log "Running GPU-enabled binary with dispatch forced..."
GPU_STDOUT="$WORK_DIR/gpu_stdout.txt"
GPU_STDERR="$WORK_DIR/gpu_stderr.txt"
ESHKOL_VERBOSE=1 ESHKOL_GPU_VERBOSE=1 ESHKOL_GPU_THRESHOLD=1 ESHKOL_SF64_KERNEL=legacy \
    "$GPU_PAYLOAD" > "$GPU_STDOUT" 2> "$GPU_STDERR"
gpu_run_rc=$?
[ "$gpu_run_rc" -eq 0 ] || fail "GPU binary crashed/exited $gpu_run_rc — stderr: $(tail -n 40 "$GPU_STDERR")"
grep -q "^GATE-DONE$" "$GPU_STDOUT" || fail "GPU run did not reach GATE-DONE — output: $(cat "$GPU_STDOUT")"

if grep -qE '\[GPU\] Metal: |\[GPU\] .*-> CUDA cuBLAS|\[GPU\] sf64 (dispatch|completed):' "$GPU_STDERR"; then
    log "  GPU device confirmed live (init banner / per-call dispatch log present):"
    grep -E '\[GPU\] Metal: |\[GPU\] .*-> CUDA cuBLAS|\[GPU\] sf64 (dispatch|completed):' "$GPU_STDERR" | sed 's/^/    /'
else
    skip "GPU-enabled binary built and ran, but no GPU device announced itself at runtime (no [GPU] init/dispatch log) — likely a GPU-less/virtualized host despite compile-time framework detection"
fi

log ""
log "Running CPU-only reference binary..."
CPU_STDOUT="$WORK_DIR/cpu_stdout.txt"
"$CPU_PAYLOAD" > "$CPU_STDOUT" 2>"$WORK_DIR/cpu_stderr.txt"
cpu_run_rc=$?
[ "$cpu_run_rc" -eq 0 ] || fail "CPU-reference binary crashed/exited $cpu_run_rc"
grep -q "^GATE-DONE$" "$CPU_STDOUT" || fail "CPU run did not reach GATE-DONE"

# ─────────────────────────────────────────────────────────────────
# Step 5: differential comparison, tolerance-based.
# ─────────────────────────────────────────────────────────────────
log ""
log "Diffing GPU vs CPU RESULT lines (relative tolerance $GPU_GATE_TOL)..."

grep '^RESULT ' "$GPU_STDOUT" > "$WORK_DIR/gpu_results.txt"
grep '^RESULT ' "$CPU_STDOUT" > "$WORK_DIR/cpu_results.txt"

gpu_labels="$(awk '{print $2}' "$WORK_DIR/gpu_results.txt")"
cpu_labels="$(awk '{print $2}' "$WORK_DIR/cpu_results.txt")"
[ "$gpu_labels" = "$cpu_labels" ] || fail "RESULT label sets differ between GPU and CPU runs — payload divergence, not a numeric issue"

mismatch=0
max_rel_diff=0
while read -r _ label gval; do
    cval="$(awk -v l="$label" '$2==l {print $3}' "$WORK_DIR/cpu_results.txt")"
    rel_diff="$(awk -v g="$gval" -v c="$cval" -v tol="$GPU_GATE_TOL" '
        BEGIN {
            diff = g - c; if (diff < 0) diff = -diff;
            denom = c; if (denom < 0) denom = -denom;
            if (denom < 1e-12) denom = 1e-12;
            rel = diff / denom;
            printf "%.10g %d", rel, (rel > tol) ? 1 : 0;
        }')"
    rel="$(echo "$rel_diff" | awk '{print $1}')"
    bad="$(echo "$rel_diff" | awk '{print $2}')"
    is_greater="$(awk -v a="$rel" -v b="$max_rel_diff" 'BEGIN{print (a>b)?1:0}')"
    [ "$is_greater" = "1" ] && max_rel_diff="$rel"
    if [ "$bad" = "1" ]; then
        mismatch=1
        log "  MISMATCH $label: GPU=$gval CPU=$cval rel_diff=$rel (> $GPU_GATE_TOL)"
    else
        log "  ok       $label: GPU=$gval CPU=$cval rel_diff=$rel"
    fi
done < "$WORK_DIR/gpu_results.txt"

log ""
log "Max relative diff across all probes: $max_rel_diff (tolerance $GPU_GATE_TOL)"

if [ "$mismatch" -eq 1 ]; then
    fail "GPU-vs-CPU differential mismatch exceeded tolerance — this is a GPU kernel correctness bug, not an infra flake"
fi

log ""
log "PASS: GPU execution matches CPU reference within tolerance ($GPU_GATE_TOL) on $UNAME_S"
emit_trace PASS "platform=$UNAME_S max_rel_diff=$max_rel_diff tol=$GPU_GATE_TOL probes=$(wc -l < "$WORK_DIR/gpu_results.txt" | tr -d ' ')"
exit 0
