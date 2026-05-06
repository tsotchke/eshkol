#!/bin/bash

test_output_fail() {
    echo "$1" >&2
    exit 1
}

test_output_reject_unsafe_path() {
    local label="$1"
    local path="$2"

    case "$path" in
        ""|/*|..|../*|*/../*|*/..|*$'\n'*|*$'\r'*)
            test_output_fail "unsafe $label path: $path"
            ;;
    esac
}

test_output_prepare_dir() {
    local dir="$1"

    test_output_reject_unsafe_path "test output directory" "$dir"

    if [ -L "$dir" ]; then
        test_output_fail "test output directory must not be a symlink: $dir"
    fi

    mkdir -p -- "$dir" || test_output_fail "failed to create test output directory: $dir"

    if [ -L "$dir" ] || [ ! -d "$dir" ]; then
        test_output_fail "test output directory missing or symlinked after creation: $dir"
    fi
}

test_output_require_file() {
    local file="$1"
    local dir="$2"
    local label="${3:-test output file}"

    test_output_reject_unsafe_path "$label" "$file"

    case "$file" in
        "$dir"/*)
            ;;
        *)
            test_output_fail "$label must stay under test output directory: $file"
            ;;
    esac

    if [ -L "$file" ]; then
        test_output_fail "$label must not be a symlink: $file"
    fi
}

test_output_reset_file() {
    local file="$1"
    local dir="$2"
    local label="${3:-test output file}"

    test_output_require_file "$file" "$dir" "$label"
    : > "$file" || test_output_fail "failed to reset $label: $file"

    if [ -L "$file" ] || [ ! -f "$file" ]; then
        test_output_fail "$label missing or symlinked after reset: $file"
    fi
}

test_output_append_line() {
    local file="$1"
    local dir="$2"
    local line="$3"

    test_output_require_file "$file" "$dir" "test results file"
    printf '%s\n' "$line" >> "$file" || test_output_fail "failed to append test result: $file"
}
