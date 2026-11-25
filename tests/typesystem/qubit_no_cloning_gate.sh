#!/usr/bin/env bash
# Qualify the strict-types refusal against a single-use control.
set -eu
runner="${1:?compiler required}"
work="$(mktemp -d "${2:?work directory required}/qubit.XXXXXX")"
fixtures="$(cd "$(dirname "$0")" && pwd)"
flags=(--strict-types "-L$(dirname "$runner")")
"$runner" "${flags[@]}" "$fixtures/qubit_linear_valid_test.esk" -o "$work/control"
test -s "$work/control" && test -x "$work/control" || exit 1
"$work/control"
if "$runner" "${flags[@]}" "$fixtures/qubit_no_cloning_test.esk" -o "$work/clone" 2>"$work/stderr"; then
    echo "expected strict-types to refuse the cloned qubit" >&2
    exit 1
else
    status=$?
fi
expected="$(sed -n 's/^;; EXPECT-STDERR: //p' "$fixtures/qubit_no_cloning_test.esk")"
cat "$work/stderr"
test "$status" -eq 1 && test ! -e "$work/clone" && test -n "$expected" && grep -Fq -- "$expected" "$work/stderr"
