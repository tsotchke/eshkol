#!/usr/bin/env bash
# Linux native regression test. Supply a matching Eshkol runtime archive,
# or build the eshkol-runtime CMake target in build/ first.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
runtime=${1:-"$root/build/libeshkol-runtime.a"}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
"${CXX:-c++}" -std=c++17 -ffunction-sections -fdata-sections -Wl,--gc-sections \
    -I "$root/inc" -I "$root/lib/core" "$root/tests/ffi/tensor_shape_roundtrip.cpp" \
    "$root/lib/ffi/eshkol_ffi.cpp" "$root/lib/core/tensor_validation.cpp" \
    "$runtime" -ldl -pthread -lm -o "$work/test"
"$work/test"
