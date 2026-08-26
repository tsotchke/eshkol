#!/usr/bin/env bash
# bench/lib/fingerprint.sh — capture the environment a benchmark ran in.
#
# WHY THIS EXISTS
#
# The single sharpest external criticism the project has received is that
# every performance/rigor claim is self-reported, with no way for a stranger
# to reproduce it. Half of "reproduce" is the harness; the other half is
# knowing EXACTLY what machine, compiler, and build produced a number. This
# file writes that half.
#
# NOTE ON scripts/lib/build_fingerprint.sh: at the time this suite was
# written, PR #465 (which adds scripts/lib/build_fingerprint.sh) had not
# merged. That file solves a different, narrower problem — proving a test
# harness's evidence talks about the binary actually on disk, appending one
# JSON-L event per harness run keyed on a binary's sha256/mtime. This file
# solves the provenance problem: a full description of the machine/toolchain/
# build-flags a benchmark ran under, embedded once per results.json. If/when
# scripts/lib/build_fingerprint.sh lands, the two are complementary (that one
# answers "is this still the binary that ran"; this one answers "what was
# that binary built with, on what machine") and this file should start
# calling eshkol_emit_build_fingerprint_event() for the staleness half rather
# than reimplementing it.
#
# Usage:
#   source bench/lib/fingerprint.sh
#   bench_capture_fingerprint <build_dir> <label>   # prints one JSON object

bench_capture_fingerprint() { # <build_dir> <label> -> JSON object on stdout
    local build_dir="$1" label="$2" repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

    local os_name os_version kernel_release
    os_name="$(uname -s)"
    kernel_release="$(uname -r)"
    if [ "$os_name" = "Darwin" ]; then
        os_version="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
    elif [ -r /etc/os-release ]; then
        os_version="$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")"
    else
        os_version="unknown"
    fi

    local cpu_model cpu_logical cpu_physical mem_bytes
    if [ "$os_name" = "Darwin" ]; then
        cpu_model="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
        cpu_logical="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 0)"
        cpu_physical="$(sysctl -n hw.physicalcpu 2>/dev/null || echo 0)"
        mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    else
        cpu_model="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | sed 's/^[^:]*: //' || echo unknown)"
        cpu_logical="$(nproc 2>/dev/null || echo 0)"
        cpu_physical="$cpu_logical"
        mem_bytes="$(awk '/MemTotal/{print $2*1024}' /proc/meminfo 2>/dev/null || echo 0)"
    fi

    local gpu_model="unknown"
    if [ "$os_name" = "Darwin" ]; then
        gpu_model="$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model/{print $2; exit}')"
        [ -n "$gpu_model" ] || gpu_model="unknown (Apple integrated GPU via Metal — see cpu_model)"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        gpu_model="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
        [ -n "$gpu_model" ] || gpu_model="unknown"
    fi

    local cc_version="unknown" cxx_version="unknown"
    if command -v clang >/dev/null 2>&1; then
        cc_version="$(clang --version 2>/dev/null | head -1)"
    elif command -v cc >/dev/null 2>&1; then
        cc_version="$(cc --version 2>/dev/null | head -1)"
    fi
    if command -v clang++ >/dev/null 2>&1; then
        cxx_version="$(clang++ --version 2>/dev/null | head -1)"
    fi

    local llvm_version="unknown"
    if command -v llvm-config >/dev/null 2>&1; then
        llvm_version="$(llvm-config --version 2>/dev/null)"
    fi

    local blas_kind="unknown"
    if [ "$os_name" = "Darwin" ]; then
        blas_kind="Apple Accelerate (vecLib/AMX)"
    elif [ -n "${ESHKOL_BLAS_KIND:-}" ]; then
        blas_kind="$ESHKOL_BLAS_KIND"
    fi

    local git_sha git_branch git_dirty
    git_sha="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)"
    git_branch="$(git -C "$repo_root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    if git -C "$repo_root" diff --quiet 2>/dev/null && git -C "$repo_root" diff --cached --quiet 2>/dev/null; then
        git_dirty="false"
    else
        git_dirty="true"
    fi

    local eshkol_version="unknown"
    if [ -f "$repo_root/inc/eshkol/eshkol.h" ]; then
        eshkol_version="$(grep '^#define ESHKOL_VERSION_STRING' "$repo_root/inc/eshkol/eshkol.h" | sed -E 's/.*"([^"]+)".*/\1/' | head -1)"
        [ -n "$eshkol_version" ] || eshkol_version="unknown"
    fi

    local quantum_enabled="unknown" gpu_enabled="unknown"
    if [ -f "$build_dir/CMakeCache.txt" ]; then
        quantum_enabled="$(awk -F= '/^ESHKOL_QUANTUM_ENABLED:/{print $2}' "$build_dir/CMakeCache.txt")"
        gpu_enabled="$(awk -F= '/^ESHKOL_GPU_ENABLED:/{print $2}' "$build_dir/CMakeCache.txt")"
        [ -n "$quantum_enabled" ] || quantum_enabled="unknown"
        [ -n "$gpu_enabled" ] || gpu_enabled="unknown"
    fi

    local build_type="unknown"
    if [ -f "$build_dir/CMakeCache.txt" ]; then
        build_type="$(awk -F= '/^CMAKE_BUILD_TYPE:/{print $2}' "$build_dir/CMakeCache.txt")"
        [ -n "$build_type" ] || build_type="unknown"
    fi

    local load_avg
    load_avg="$(uptime 2>/dev/null | sed -E 's/.*load averages?:? *//')"

    python3 - "$label" "$os_name" "$os_version" "$kernel_release" "$cpu_model" "$cpu_physical" \
        "$cpu_logical" "$mem_bytes" "$gpu_model" "$cc_version" "$cxx_version" "$llvm_version" \
        "$blas_kind" "$git_sha" "$git_branch" "$git_dirty" "$eshkol_version" "$quantum_enabled" \
        "$gpu_enabled" "$build_type" "$build_dir" "$load_avg" <<'PYEOF'
import json, sys
(label, os_name, os_version, kernel_release, cpu_model, cpu_physical, cpu_logical,
 mem_bytes, gpu_model, cc_version, cxx_version, llvm_version, blas_kind, git_sha,
 git_branch, git_dirty, eshkol_version, quantum_enabled, gpu_enabled, build_type,
 build_dir, load_avg) = sys.argv[1:23]
print(json.dumps({
    "label": label,
    "os": {"name": os_name, "version": os_version, "kernel": kernel_release},
    "cpu": {"model": cpu_model, "physical_cores": int(cpu_physical or 0),
            "logical_cores": int(cpu_logical or 0)},
    "memory_bytes": int(mem_bytes or 0),
    "gpu_model": gpu_model,
    "compiler": {"cc": cc_version, "cxx": cxx_version, "llvm": llvm_version},
    "blas": blas_kind,
    "git": {"sha": git_sha, "branch": git_branch, "dirty": git_dirty == "true"},
    "eshkol_version": eshkol_version,
    "build": {"dir": build_dir, "type": build_type,
              "quantum_enabled": quantum_enabled, "gpu_enabled": gpu_enabled},
    "load_average_at_capture": load_avg,
}, indent=2, ensure_ascii=False))
PYEOF
}
