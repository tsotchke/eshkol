#!/usr/bin/env bash
# Compare lib/backend/gpu/vulkan_abi.h with the Khronos Vulkan headers.
#
# gpu_vulkan.c compiles against <vulkan/vulkan.h> when it exists and against
# the self-contained lib/backend/gpu/vulkan_abi.h otherwise. This script builds
# tests/gpu/vulkan_abi_layout.c both ways and requires byte-identical output:
# every struct size, field offset, field size and enumerant the backend uses.
#
# Usage: scripts/check_vulkan_abi.sh [VULKAN_INCLUDE_DIR]
#   VULKAN_INCLUDE_DIR defaults to $VULKAN_SDK/include, then the usual system
#   and Homebrew locations. Exit 0 = identical, 1 = differs, 77 = no headers.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
inc="${1:-}"
if [ -z "$inc" ]; then
    for cand in "${VULKAN_SDK:-}/include" /usr/include /usr/local/include \
                /opt/homebrew/opt/vulkan-headers/include; do
        if [ -f "$cand/vulkan/vulkan_core.h" ]; then inc="$cand"; break; fi
    done
fi
if [ -z "$inc" ] || [ ! -f "$inc/vulkan/vulkan_core.h" ]; then
    echo "SKIP: no Khronos Vulkan headers found"
    exit 77
fi

work="${ESHKOL_VK_ABI_WORKDIR:-$root/.scratch/vulkan-abi-check}"
mkdir -p "$work"
cc="${CC:-cc}"
src="$root/tests/gpu/vulkan_abi_layout.c"

"$cc" -std=c11 -O0 -I"$root/lib/backend/gpu" -o "$work/layout_local" "$src"
"$cc" -std=c11 -O0 -DESHKOL_VK_USE_KHRONOS -I"$inc" -o "$work/layout_khronos" "$src"
"$work/layout_local" > "$work/local.txt"
"$work/layout_khronos" > "$work/khronos.txt"

if cmp -s "$work/local.txt" "$work/khronos.txt"; then
    echo "PASS: vulkan_abi.h matches $inc/vulkan/vulkan_core.h ($(wc -l < "$work/local.txt" | tr -d ' ') entries)"
    exit 0
fi
echo "FAIL: vulkan_abi.h differs from $inc/vulkan/vulkan_core.h"
diff "$work/khronos.txt" "$work/local.txt" || true
exit 1
