#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
VM="$ROOT_DIR/$BUILD_DIR/eshkol-vm-standalone-test"

if [ ! -x "$VM" ]; then
    echo "FAIL: VM executable not found at $VM" >&2
    exit 2
fi

failures=0
for source in \
    tests/integration/fatal/first_class_builtin_wrong_arity.esk \
    tests/integration/fatal/first_class_string_ref_wrong_arity.esk \
    tests/integration/fatal/first_class_vref_wrong_arity.esk
do
    output="$("$VM" "$ROOT_DIR/$source" 2>&1)"
    status=$?
    if [ "$status" -eq 0 ] || ! printf '%s\n' "$output" | grep -q 'arity mismatch'; then
        echo "FAIL: $source did not fail with an arity diagnostic"
        failures=$((failures + 1))
    else
        echo "PASS: $source"
    fi
done

if [ "$failures" -ne 0 ]; then
    exit 1
fi
echo "PASS: first-class builtin arity enforcement"
