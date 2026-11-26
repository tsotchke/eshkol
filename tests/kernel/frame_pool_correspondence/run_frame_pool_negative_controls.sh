#!/usr/bin/env bash
# Negative controls for the frame-pool correspondence: each mutant in
# proofs/lean/mutation/manifests/EshkolKernelK2.mutants.json whose "differential"
# field is DIVERGES or AGREES is applied to the model's definitions and replayed
# against native traces (without alloc_kernel). Exit status is non-zero if any control
# does otherwise. Usage: run_frame_pool_negative_controls.sh [seed]   (default 20261126)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
lean_bin="${LEAN_BIN:-lean}"
cc="${CC:-cc}"
seed="${1:-20261126}"
manifest="$root/proofs/lean/mutation/manifests/EshkolKernelK2.mutants.json"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$cc" -std=c11 -O1 -Wall -I"$root/inc" "$here/generate_frame_pool_traces.c" \
  "$root/lib/kernel/k2_mmu_frames.c" -o "$work/generate_frame_pool_traces"
"$work/generate_frame_pool_traces" "$seed" 2000 60 0 > "$work/trace"
# Write one definitions-only model per applicable mutant (the theorems would not compile).
python3 -I - "$root/proofs/lean/EshkolKernelK2.lean" "$manifest" "$work" <<'PY'
import json, os, sys
model, manifest, work = sys.argv[1:4]
definitions = open(model).read().split("theorem pool_init_conserved")[0] + "end EshkolKernelK2\n"
with open(os.path.join(work, "controls"), "w") as listing:
    for mutant in json.load(open(manifest))["mutants"]:
        if mutant.get("differential") not in ("DIVERGES", "AGREES"):
            continue
        if definitions.count(mutant["old"]) != 1:
            sys.exit(f"mutant {mutant['id']}: old text does not occur exactly once in the definitions")
        directory = os.path.join(work, mutant["id"])
        os.makedirs(directory)
        with open(os.path.join(directory, "EshkolKernelK2.lean"), "w") as f:
            f.write(definitions.replace(mutant["old"], mutant["new"]))
        listing.write(f"{mutant['id']} {mutant['differential']}\n")
PY
status=0
while read -r id expected; do
  (cd "$work/$id" && "$lean_bin" -o EshkolKernelK2.olean EshkolKernelK2.lean > /dev/null)
  if LEAN_PATH="$work/$id" "$lean_bin" --run "$here/FramePoolCorrespondence.lean" "$work/trace" > "$work/$id.log"; then
    observed=AGREES
  else
    observed=DIVERGES
  fi
  echo "$observed	$id	$(head -1 "$work/$id.log")"
  if [ "$observed" != "$expected" ]; then
    echo "UNEXPECTED	$id	expected $expected"
    status=1
  fi
done < "$work/controls"
exit "$status"
