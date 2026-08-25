#!/usr/bin/env bash
# Native JIT+AOT OALR regression for Scheme-visible builtin/agent results.
set -euo pipefail

ESHKOL_RUN="${1:-${ESHKOL_RUN:-}}"
if [ -z "$ESHKOL_RUN" ]; then
    ESHKOL_RUN="./build-offline-src/eshkol-run"
fi
[ -x "$ESHKOL_RUN" ] || { echo "FAIL: missing eshkol-run" >&2; exit 1; }

build_dir="$(cd "$(dirname "$ESHKOL_RUN")" && pwd)"
root="$(cd "$(dirname "$0")/../.." && pwd)"
src="$root/tests/memory/region_builtin_pipeline_stress.esk"
rows="${ESHKOL_REGION_BUILTIN_ROWS:-50000}"
ceiling_mb="${ESHKOL_REGION_BUILTIN_CEIL_MB:-512}"
work="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-region-builtin.XXXXXX")"
trap 'rm -rf "$work"' EXIT

# Static ownership regression: these helpers return Scheme-visible values, so
# a direct global arena load would silently bypass native with-region scopes.
if rg -q 'CreateLoad\(ctx_\.ptrType\(\), ctx_\.globalArena\(\)\)' \
      "$root/lib/backend/string_io_codegen.cpp"; then
    echo "FAIL: StringIO Scheme-visible allocator bypasses currentArena" >&2
    exit 1
fi

peak_mb() {
    local label="$1"; shift
    local timing="$work/$label.time" raw
    if [ "$(uname -s)" = Darwin ]; then
        /usr/bin/time -l "$@" >"$work/$label.out" 2>"$timing"
        raw="$(awk '/maximum resident set size/{print $1}' "$timing")"
        [ -n "$raw" ] && [ "$raw" -gt 0 ] || {
            echo "FAIL: macOS time did not report peak RSS" >&2; return 1;
        }
        echo $(( raw / 1024 / 1024 ))
    else
        /usr/bin/time -v "$@" >"$work/$label.out" 2>"$timing"
        raw="$(awk -F': ' '/Maximum resident set size/{print $2}' "$timing")"
        [ -n "$raw" ] && [ "$raw" -gt 0 ] || {
            echo "FAIL: GNU time did not report peak RSS" >&2; return 1;
        }
        echo $(( raw / 1024 ))
    fi
}

run_env=(env "ESHKOL_PATH=$root/lib" "ESHKOL_LIB_DIR=$build_dir"
         "ESHKOL_JIT_CACHE_DIR=$work/jit-cache" "ESHKOL_REGION_BUILTIN_ROWS=$rows")
jit_mb="$(peak_mb jit "${run_env[@]}" "ESHKOL_REGION_BUILTIN_FIXTURE=$work/jit.fixture" "$ESHKOL_RUN" --strict-types -r "$src" -L"$build_dir")"
grep -q "PASS: region builtin pipeline rows=$rows" "$work/jit.out"
"${run_env[@]}" "$ESHKOL_RUN" --strict-types -o "$work/pipeline-aot" "$src" -L"$build_dir" >"$work/aot.compile" 2>&1
aot_mb="$(peak_mb aot "${run_env[@]}" "ESHKOL_REGION_BUILTIN_FIXTURE=$work/aot.fixture" "$work/pipeline-aot")"
grep -q "PASS: region builtin pipeline rows=$rows" "$work/aot.out"

echo "region-builtin-pipeline rows=$rows jit=${jit_mb}MB aot=${aot_mb}MB ceiling=${ceiling_mb}MB"
test "$jit_mb" -lt "$ceiling_mb"
test "$aot_mb" -lt "$ceiling_mb"
! grep -Eqi 'heap (usage|limit)|error:' "$work/jit.out" "$work/jit.time" "$work/aot.out" "$work/aot.time"
echo "PASS: native JIT/AOT region builtin pipeline stays bounded"
