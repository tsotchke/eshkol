#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/scripts/lib/memory_compile_refusal.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fixture="$tmp/refusal.esk"
diagnostic="$tmp/compiler.log"
artifact="$tmp/program"
printf '%s\n' ';;; Expected: Error - "specific ownership diagnostic"' > "$fixture"
expected="$(memory_expected_compile_error "$fixture")"
[ "$expected" = "specific ownership diagnostic" ]

printf '%s\n' "ERROR: $expected" > "$diagnostic"
memory_compile_refusal_matches 1 "$expected" "$diagnostic" "$artifact"

# A generic failure, a crash-like status, and a produced artifact must not pass.
printf '%s\n' 'ERROR: unrelated parse error' > "$diagnostic"
if memory_compile_refusal_matches 1 "$expected" "$diagnostic" "$artifact"; then exit 1; fi
printf '%s\n' "ERROR: $expected" > "$diagnostic"
if memory_compile_refusal_matches 139 "$expected" "$diagnostic" "$artifact"; then exit 1; fi
: > "$artifact"
if memory_compile_refusal_matches 1 "$expected" "$diagnostic" "$artifact"; then exit 1; fi

echo "PASS: memory compile-refusal contract rejects generic failures, crashes, and emitted artifacts"
