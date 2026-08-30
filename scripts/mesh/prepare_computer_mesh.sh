#!/usr/bin/env bash
set -euo pipefail

# This helper is also safe when a workflow reuses it from a hosted matrix.
# GitHub-hosted and Windows runners do not carry the self-hosted mesh
# environment this check is intended to inspect, and Windows does not provide
# the Bash/Python/Git contract below. The workflow condition is the primary
# gate; this guard keeps the script harmless when invoked directly.
if [[ "${GITHUB_ACTIONS:-}" == "true" ]] && {
  [[ "${RUNNER_ENVIRONMENT:-}" != "self-hosted" ]] ||
  [[ "${RUNNER_OS:-}" == "Windows" ]];
}; then
  echo "computer_mesh preflight skipped: not a supported self-hosted runner"
  exit 0
fi

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
