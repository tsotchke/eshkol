#!/usr/bin/env bash
# Run the mutation runner's self-test and every manifest under manifests/.
# Exit status is non-zero if any baseline fails to compile clean or any mutant's
# outcome differs from the "expect" recorded in its manifest.
# Requires Lean 4.12.0; set LEAN_BIN if it is not "lean" on PATH.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
lean_bin="${LEAN_BIN:-lean}"
status=0
for manifest in "$here/selftest/manifest.json" "$here"/manifests/*.mutants.json; do
  echo "== ${manifest#"$here"/}"
  LEAN_BIN="$lean_bin" "$lean_bin" --run "$here/MutationRunner.lean" "$manifest" || status=1
done
exit "$status"
