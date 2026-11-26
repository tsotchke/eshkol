#!/usr/bin/env bash
# Negative controls for the correspondence test: each mutant in
# proofs/lean/mutation/manifests/EshkolCompositorAligned.mutants.json is applied to the
# model's definitions, and the result is replayed against the native traces. Each
# mutant's "differential" field records whether it must DIVERGE from native or AGREE
# with it (a check that is redundant on every reachable state). Exit status is
# non-zero if any control does otherwise.
# Usage: run_correspondence_negative_controls.sh [seed]   (default 20261126)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
lean_bin="${LEAN_BIN:-lean}"
cc="${CC:-cc}"
seed="${1:-20261126}"
manifest="$root/proofs/lean/mutation/manifests/EshkolCompositorAligned.mutants.json"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$cc" -std=c11 -O1 -Wall -I"$root/inc" "$here/generate_compositor_full_traces.c" \
  "$root/lib/core/os_k5_k6_services.c" -o "$work/generate_compositor_full_traces"
"$work/generate_compositor_full_traces" "$seed" 2000 60 > "$work/trace"
# Write one definitions-only model per mutant (the theorems would not compile).
python3 -I - "$root/proofs/lean/EshkolCompositorAligned.lean" "$manifest" "$work" <<'PY'
import json, os, sys
model, manifest, work = sys.argv[1:4]
definitions = open(model).read().split("/-! ## Write guarantees -/")[0] + "end EshkolCompositorAligned\n"
with open(os.path.join(work, "controls"), "w") as listing:
    for mutant in json.load(open(manifest))["mutants"]:
        if definitions.count(mutant["old"]) != 1:
            sys.exit(f"mutant {mutant['id']}: old text does not occur exactly once in the definitions")
        directory = os.path.join(work, mutant["id"])
        os.makedirs(directory)
        with open(os.path.join(directory, "EshkolCompositorAligned.lean"), "w") as f:
            f.write(definitions.replace(mutant["old"], mutant["new"]))
        listing.write(f"{mutant['id']} {mutant['differential']}\n")
PY
status=0
while read -r id expected; do
  (cd "$work/$id" && "$lean_bin" -o EshkolCompositorAligned.olean EshkolCompositorAligned.lean > /dev/null)
  if LEAN_PATH="$work/$id" "$lean_bin" --run "$here/CompositorFullTrace.lean" "$work/trace" > "$work/$id.log"; then
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
