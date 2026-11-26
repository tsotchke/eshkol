#!/usr/bin/env bash
# Negative controls for the frame-mapping correspondence: each mutant in
# proofs/lean/mutation/manifests/EshkolKernelFrameMappings.mutants.json whose
# "differential" field is DIVERGES or AGREES is applied to the model's definitions and
# replayed against native traces (seed 1, both ASID configurations, distinct and shared). Exit status is
# non-zero if any control does otherwise.
# Usage: run_frame_mapping_negative_controls.sh [seed]   (default 1)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
lean_bin="${LEAN_BIN:-lean}"
cc="${CC:-cc}"
seed="${1:-1}"
manifest="$root/proofs/lean/mutation/manifests/EshkolKernelFrameMappings.mutants.json"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$cc" -std=c11 -O1 -Wall -I"$root/inc" "$here/generate_frame_mapping_traces.c" \
  "$root/lib/kernel/k2_mmu_frames.c" -o "$work/generate_frame_mapping_traces"
"$work/generate_frame_mapping_traces" "$seed" 2000 80 distinct > "$work/distinct.trace"
"$work/generate_frame_mapping_traces" "$seed" 2000 80 shared > "$work/shared.trace"
# Write one definitions-only model per applicable mutant (the theorems would not compile).
python3 -I - "$root/proofs/lean/EshkolKernelFrameMappings.lean" "$manifest" "$work" <<'PY'
import json, os, sys
model, manifest, work = sys.argv[1:4]
source = open(model).read()
definitions = source[:source.index("/-! ## The actual relation and native's summaries -/")] + \
    "end EshkolKernelFrameMappings\n"
with open(os.path.join(work, "controls"), "w") as listing:
    for mutant in json.load(open(manifest))["mutants"]:
        if mutant.get("differential") not in ("DIVERGES", "AGREES"):
            continue
        if definitions.count(mutant["old"]) != 1:
            sys.exit(f"mutant {mutant['id']}: old text does not occur exactly once in the definitions")
        directory = os.path.join(work, mutant["id"])
        os.makedirs(directory)
        with open(os.path.join(directory, "EshkolKernelFrameMappings.lean"), "w") as out:
            out.write(definitions.replace(mutant["old"], mutant["new"]))
        listing.write(f"{mutant['id']} {mutant['differential']}\n")
PY
status=0
while read -r id expected; do
  directory="$work/$id"
  (cd "$directory" && "$lean_bin" -o EshkolKernelFrameMappings.olean EshkolKernelFrameMappings.lean)
  outcome=AGREES
  for trace in "$work/distinct.trace" "$work/shared.trace"; do
    if ! LEAN_PATH="$directory" "$lean_bin" --run "$here/FrameMappingCorrespondence.lean" "$trace" \
        > "$directory/replay.log" 2>&1; then
      outcome=DIVERGES
      break
    fi
  done
  if [ "$outcome" = "$expected" ]; then
    printf '%-40s %s\n' "$id" "$outcome"
  else
    printf '%-40s %s  UNEXPECTED (expected %s)\n' "$id" "$outcome" "$expected"
    status=1
  fi
done < "$work/controls"
exit "$status"
