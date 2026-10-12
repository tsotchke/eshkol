#!/bin/sh
# Run one unsupported nested-reverse case and accept only its documented refusal.
set -eu

if [ "$#" -ne 5 ]; then
    echo "FAIL: usage: $0 RUNNER SOURCE BUILD_DIR MODE EXPECTED_DIAGNOSTIC" >&2
    exit 2
fi
runner=$1
source_file=$2
build_dir=$3
mode=$4
expected=$5
repo_root=$(CDPATH= cd -- "$(dirname -- "$source_file")/../.." && pwd)
runner_dir=$(dirname -- "$runner")
work=$(mktemp -d "$build_dir/nested-ad-refusal-${mode}.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
mkdir -p "$work/jit-cache"
export ESHKOL_PATH="$repo_root/lib"
export ESHKOL_LIB_DIR="$runner_dir"
export ESHKOL_JIT_CACHE_DIR="$work/jit-cache"

case "$mode" in
    jit)
        export ESHKOL_JIT_CACHE=0
        if "$runner" -r "$source_file" -L"$build_dir" >"$work/result.log" 2>&1; then
            result_code=0
        else
            result_code=$?
        fi
        ;;
    aot)
        executable="$work/nested_refusal"
        if ! "$runner" -o "$executable" "$source_file" -L"$build_dir" >"$work/compile.log" 2>&1; then
            echo "FAIL: AOT compile failed before the refusal case" >&2
            cat "$work/compile.log" >&2
            exit 1
        fi
        if "$executable" >"$work/result.log" 2>&1; then
            result_code=0
        else
            result_code=$?
        fi
        ;;
    *)
        echo "FAIL: unknown mode: $mode" >&2
        exit 2
        ;;
esac

if [ "$result_code" -eq 0 ]; then
    echo "FAIL: $mode succeeded; expected refusal: $expected" >&2
    cat "$work/result.log" >&2
    exit 1
fi
if [ "$result_code" -ge 128 ]; then
    echo "FAIL: $mode exited with signal-style code $result_code, not a catchable refusal" >&2
    cat "$work/result.log" >&2
    exit 1
fi
if ! grep -F -- "$expected" "$work/result.log" >/dev/null; then
    echo "FAIL: $mode exited $result_code without the intended refusal: $expected" >&2
    cat "$work/result.log" >&2
    exit 1
fi
printf 'PASS: nested reverse refusal (%s, %s)\n' "$(basename -- "$source_file" .esk)" "$mode"
