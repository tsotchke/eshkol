#!/usr/bin/env bash
# Full-trace correspondence between the native compositor
# (lib/core/os_k5_k6_services.c) and proofs/lean/EshkolCompositorAligned.lean.
# Exit status 0 if and only if every trace agrees after every operation.
# Usage: run_compositor_correspondence.sh [seed ...]   (default seeds: 1 20261126 777)
# Requires a C11 compiler (CC, default cc) and Lean 4.12.0 (LEAN_BIN, default lean).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
lean_bin="${LEAN_BIN:-lean}"
cc="${CC:-cc}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$cc" -std=c11 -O1 -Wall -I"$root/inc" "$here/generate_compositor_full_traces.c" \
  "$root/lib/core/os_k5_k6_services.c" -o "$work/generate_compositor_full_traces"
cp "$root/proofs/lean/EshkolCompositorAligned.lean" "$work/"
(cd "$work" && "$lean_bin" -o EshkolCompositorAligned.olean EshkolCompositorAligned.lean)
seeds=("$@")
if [ "${#seeds[@]}" -eq 0 ]; then seeds=(1 20261126 777); fi
status=0
for seed in "${seeds[@]}"; do
  "$work/generate_compositor_full_traces" "$seed" 2000 60 > "$work/trace-$seed"
  echo "== seed $seed"
  LEAN_PATH="$work" "$lean_bin" --run "$here/CompositorFullTrace.lean" "$work/trace-$seed" || status=1
done
exit "$status"
