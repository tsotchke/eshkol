#!/usr/bin/env bash
# Falsifier F1 for ADR-0011: a guest tracing collector hosted in an Eshkol region.
#
# Self-contained: no CMake, no LLVM, no configure step. It compiles five
# UNMODIFIED runtime translation units so the arena under test is the real one,
# plus harness_stubs.cpp for the process-accounting symbols the experiment does
# not exercise (see that file for why they are orthogonal).
#
#   usage:  docs/design/adr/0011-gc-adapter-falsifier/build-and-run.sh
#
# Exits non-zero if any check fails.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
out="${TMPDIR_OVERRIDE:-$here/.build}"
mkdir -p "$out"

CXX="${CXX:-c++}"
inc=(-I"$root/lib/core" -I"$root/inc")

# Unmodified runtime sources: the arena allocator under test, its mutex shim,
# its poison-diagnostics shim, the tagged-cons helpers, and the logger.
units=(
  lib/core/runtime_arena_core.cpp
  lib/core/runtime_arena_sync_hosted.cpp
  lib/core/runtime_arena_diagnostics_hosted.cpp
  lib/core/runtime_tagged_cons.cpp
  lib/core/logger.cpp
)

objs=()
for u in "${units[@]}"; do
  o="$out/$(basename "${u%.cpp}").o"
  "$CXX" -std=c++17 -O2 "${inc[@]}" -c "$root/$u" -o "$o"
  objs+=("$o")
done

"$CXX" -std=c++17 -O2 "${inc[@]}" \
  "$here/gc_adapter_falsifier.cpp" "$here/harness_stubs.cpp" "${objs[@]}" -o "$out/falsifier"

exec "$out/falsifier"
