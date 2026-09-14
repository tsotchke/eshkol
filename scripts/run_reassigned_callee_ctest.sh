#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_root/build}"
task_cache="$repo_root/.scratch/reassigned-callee-ctest"

export ESHKOL_LIB_DIR="${ESHKOL_LIB_DIR:-$build_dir}"
export ESHKOL_PATH="${ESHKOL_PATH:-$repo_root/lib}"
export ESHKOL_JIT_CACHE_DIR="${ESHKOL_JIT_CACHE_DIR:-$task_cache/jit}"
export ESHKOL_AOT_MODULE_CACHE_DIR="${ESHKOL_AOT_MODULE_CACHE_DIR:-$task_cache/aot}"

ctest --test-dir "$build_dir" --output-on-failure -R '^reassigned_callee_'
