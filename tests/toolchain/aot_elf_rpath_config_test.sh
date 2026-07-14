#!/usr/bin/env bash
# Verify that every ELF library search directory emitted for an AOT-generated
# program is also preserved in that program's DT_RUNPATH.
set -euo pipefail

BUILD_DIR="${1:-}"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

[ -n "$BUILD_DIR" ] || fail "usage: $0 <build-dir>"
CONFIG_HEADER="$BUILD_DIR/generated/eshkol/build_config.h"
[ -f "$CONFIG_HEADER" ] || fail "generated build config not found: $CONFIG_HEADER"

config_value() {
    local name="$1"
    sed -n "s/^#define ${name} \"\\(.*\\)\"$/\\1/p" "$CONFIG_HEADER" | head -1
}

assert_runtime_paths() {
    local label="$1"
    local raw_args="$2"
    local normalized="${raw_args//;/ }"
    local -a args=()
    local -a expected_dirs=()
    local arg dir expected found

    read -r -a args <<< "$normalized"
    for arg in "${args[@]}"; do
        dir=""
        case "$arg" in
            -L?*) dir="${arg#-L}" ;;
            /*.so|/*.so.[0-9]*) dir="${arg%/*}" ;;
        esac
        [ -n "$dir" ] && expected_dirs+=("$dir")
    done

    for expected in "${expected_dirs[@]}"; do
        found=0
        for arg in "${args[@]}"; do
            if [ "$arg" = "-Wl,-rpath,$expected" ]; then
                found=1
                break
            fi
        done
        [ "$found" -eq 1 ] ||
            fail "$label omits -Wl,-rpath,$expected for an AOT shared-library search directory"
    done
}

assert_runtime_paths "LLVM link args" "$(config_value ESHKOL_HOST_LLVM_LINK_ARGS)"
assert_runtime_paths "runtime link args" "$(config_value ESHKOL_HOST_RUNTIME_LINK_ARGS)"
assert_runtime_paths "agent FFI link args" "$(config_value ESHKOL_HOST_AGENT_FFI_LINK_ARGS)"

HOST_CXX="$(config_value ESHKOL_HOST_CXX_COMPILER)"
LLVM_ARGS="$(config_value ESHKOL_HOST_LLVM_LINK_ARGS)"
if [ -n "$HOST_CXX" ] && [ -x "$HOST_CXX" ]; then
    for runtime_name in libstdc++.so.6 libc++.so.1 libgcc_s.so.1; do
        runtime_file="$($HOST_CXX "-print-file-name=$runtime_name" 2>/dev/null || true)"
        case "$runtime_file" in
            /*)
                [ -f "$runtime_file" ] || continue
                runtime_dir="${runtime_file%/*}"
                case ";${LLVM_ARGS// /;};" in
                    *";-Wl,-rpath,$runtime_dir;"*) ;;
                    *) fail "LLVM link args omit host C++ runtime path $runtime_dir" ;;
                esac
                ;;
        esac
    done
fi

echo "PASS: aot_elf_rpath_config_test"
