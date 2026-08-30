#!/usr/bin/env bash
set -euo pipefail

root="${1:?usage: prepare_computer_mesh.sh <destination> [inventory]}"
inventory="${2:-$root/nodes.json}"
version_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/computer_mesh.version"
repository="$(sed -n 's/^repository=//p' "$version_file")"
commit="$(sed -n 's/^commit=//p' "$version_file")"
test -n "$repository" -a -n "$commit"
mkdir -p "$root"
if [ ! -d "$root/.git" ]; then
  git clone --no-checkout "$repository" "$root"
fi
git -C "$root" fetch --no-tags --depth=1 origin "$commit"
git -C "$root" checkout --detach --force "$commit"
test "$(git -C "$root" rev-parse HEAD)" = "$commit"
test -f "$inventory"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
python3 -m computer_mesh.mesh_inventory "$inventory" audit --json
python3 -m computer_mesh.mesh_inventory "$inventory" graph --json
if [ -f "$root/gates/eshkol/lanes.json" ]; then
  python3 -m computer_mesh.mesh_inventory "$inventory" gates plan --lane eshkol --ref "${GITHUB_SHA:-HEAD}" --json
fi
