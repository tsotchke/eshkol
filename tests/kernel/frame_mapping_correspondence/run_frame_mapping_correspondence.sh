#!/usr/bin/env bash
# Correspondence between native frames and address spaces (lib/kernel/k2_mmu_frames.c)
# and the composed model proofs/lean/EshkolKernelFrameMappings.lean. For each seed and
# each ASID configuration (distinct, shared), replays native traces on the model and
# compares the status and the complete state after every operation, and native's
# verdict on each probed initialization. Also reports the axioms of the headline
# theorems, the native-side invariant tallies and the classification of PERM_ALIAS
# refusals, and runs the two directed checks.
# Exit status 0 if and only if every trace and probe agrees, the axiom report contains
# no sorryAx, no refusal is unexplained, and both directed checks pass.
# Usage: run_frame_mapping_correspondence.sh [seed ...]   (default seeds: 1 20261126 777)
# Requires a C11 compiler (CC, default cc) and Lean 4.12.0 (LEAN_BIN, default lean).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
lean_bin="${LEAN_BIN:-lean}"
cc="${CC:-cc}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
for program in generate_frame_mapping_traces check_kernel_mapping_admits_user \
    check_asid_collision_refused; do
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
for configuration in distinct shared; do
  for seed in "${seeds[@]}"; do
    "$work/generate_frame_mapping_traces" "$seed" 2000 80 "$configuration" > "$work/$configuration-$seed.trace"
    echo "== $configuration, seed $seed"
    LEAN_PATH="$work" "$lean_bin" --run "$here/FrameMappingCorrespondence.lean" \
      "$work/$configuration-$seed.trace" || status=1
  done
  echo "== $configuration: native invariant bits (RefcountExact UserRefcountExact ReadCountExact"
  echo "   WriteCountExact ExecCountExact PermsExact OwnerExact UserEntriesOwned ReferentialIntegrity"
  echo "   KernelFramesNotInUser NoWriteExecuteAlias SingleUserSpace MutableIsolation)"
  cat "$work/$configuration"-*.trace | grep -o 'inv [01]*' | sort | uniq -c
  echo "== $configuration: PERM_ALIAS classification"
  cat "$work/$configuration"-*.trace | grep -o 'precision [a-z-]*' | grep -v 'precision -$' | sort | uniq -c
  if grep -q 'precision unexplained' "$work/$configuration"-*.trace; then status=1; fi
  if grep -o 'inv [01]*' "$work/$configuration"-*.trace | grep -q 0; then status=1; fi
done
echo "== directed check: a frame mapped only by the kernel is admitted to a user space"
"$work/check_kernel_mapping_admits_user" || status=1
echo "== directed check: the ASID collision is refused at initialization and at map"
"$work/check_asid_collision_refused" || status=1
exit "$status"
