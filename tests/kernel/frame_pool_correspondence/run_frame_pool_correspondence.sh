#!/usr/bin/env bash
# Correspondence between the native frame allocator (lib/kernel/k2_mmu_frames.c) and
# FramePool in proofs/lean/EshkolKernelK2.lean. For each seed, runs traces without and
# with k2_frame_alloc_kernel; both are compared operation by operation and checked for
# the invariants.
# Exit status 0 if and only if no comparison fails.
# Usage: run_frame_pool_correspondence.sh [seed ...]   (default seeds: 1 20261126 777)
# Requires a C11 compiler (CC, default cc) and Lean 4.12.0 (LEAN_BIN, default lean).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
lean_bin="${LEAN_BIN:-lean}"
cc="${CC:-cc}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$cc" -std=c11 -O1 -Wall -I"$root/inc" "$here/generate_frame_pool_traces.c" \
  "$root/lib/kernel/k2_mmu_frames.c" -o "$work/generate_frame_pool_traces"
cp "$root/proofs/lean/EshkolKernelK2.lean" "$work/"
(cd "$work" && "$lean_bin" -o EshkolKernelK2.olean EshkolKernelK2.lean)
seeds=("$@")
if [ "${#seeds[@]}" -eq 0 ]; then seeds=(1 20261126 777); fi
status=0
for seed in "${seeds[@]}"; do
  for kernel_alloc in 0 1; do
    "$work/generate_frame_pool_traces" "$seed" 2000 60 "$kernel_alloc" > "$work/trace"
    echo "== seed $seed, alloc_kernel $kernel_alloc"
    LEAN_PATH="$work" "$lean_bin" --run "$here/FramePoolCorrespondence.lean" "$work/trace" || status=1
  done
done
exit "$status"
