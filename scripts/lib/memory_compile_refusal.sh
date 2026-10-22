#!/usr/bin/env bash
# Helpers for the memory suite's declared compile-refusal fixtures.

memory_expected_compile_error() {
    sed -n 's/^;;; Expected: Error - "\(.*\)"$/\1/p' "$1" | head -1
}

memory_compile_refusal_matches() {
    local status="$1"
    local expected="$2"
    local diagnostic_file="$3"
    local output_file="$4"

    [ "$status" -eq 1 ] || return 1
    [ -n "$expected" ] || return 1
    [ -s "$diagnostic_file" ] || return 1
    grep -Fq -- "$expected" "$diagnostic_file" || return 1
    [ ! -e "$output_file" ]
}
