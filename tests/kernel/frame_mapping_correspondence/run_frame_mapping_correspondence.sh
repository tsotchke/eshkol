#!/usr/bin/env bash
# Correspondence between native frames and address spaces (lib/kernel/k2_mmu_frames.c)
# and the composed model proofs/lean/EshkolKernelFrameMappings.lean. For each seed and
# each ASID configuration (unique, collide), replays native traces on the model and
# compares the status and the complete state after every operation. Also reports the
# axioms of the headline theorems, the native-side invariant tallies and the
# classification of PERM_ALIAS refusals, and runs the two directed reproducers.
# Exit status 0 if and only if every trace agrees, the axiom report contains no sorryAx,
# and both reproducers reproduce.
# Usage: run_frame_mapping_correspondence.sh [seed ...]   (default seeds: 1 20261126 777)
# Requires a C11 compiler (CC, default cc) and Lean 4.12.0 (LEAN_BIN, default lean).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
lean_bin="${LEAN_BIN:-lean}"
cc="${CC:-cc}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
for program in generate_frame_mapping_traces reproduce_kernel_last_asid_refusal \
    reproduce_asid_collision; do
  "$cc" -std=c11 -O1 -Wall -I"$root/inc" "$here/$program.c" \
    "$root/lib/kernel/k2_mmu_frames.c" -o "$work/$program"
done
cp "$root/proofs/lean/EshkolKernelFrameMappings.lean" "$work/"
(cd "$work" && "$lean_bin" -o EshkolKernelFrameMappings.olean EshkolKernelFrameMappings.lean)
status=0
echo "== axioms"
LEAN_PATH="$work" "$lean_bin" "$here/check_axioms.lean" | tee "$work/axioms"
if grep -q sorryAx "$work/axioms"; then status=1; fi
seeds=("$@")
if [ "${#seeds[@]}" -eq 0 ]; then seeds=(1 20261126 777); fi
for configuration in unique collide; do
  for seed in "${seeds[@]}"; do
    "$work/generate_frame_mapping_traces" "$seed" 2000 80 "$configuration" > "$work/$configuration-$seed.trace"
    echo "== $configuration, seed $seed"
    LEAN_PATH="$work" "$lean_bin" --run "$here/FrameMappingCorrespondence.lean" \
      "$work/$configuration-$seed.trace" || status=1
  done
  echo "== $configuration: native invariant bits (RefcountExact ReferentialIntegrity"
  echo "   KernelFramesNotInUser PermUnionCovers NoWriteExecuteAlias MutableIsolation SingleUserAsid)"
  cat "$work/$configuration"-*.trace | grep -o 'inv [01]*' | sort | uniq -c
  echo "== $configuration: PERM_ALIAS classification"
  cat "$work/$configuration"-*.trace | grep -o 'precision [a-z-]*' | grep -v 'precision -$' | sort | uniq -c
done
echo "== directed reproducer: kernel last ASID refuses a read-only user mapping"
"$work/reproduce_kernel_last_asid_refusal" || status=1
echo "== directed reproducer: a shared ASID admits writable cross-space sharing"
"$work/reproduce_asid_collision" || status=1
exit "$status"
