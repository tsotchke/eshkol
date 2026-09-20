#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUILD_DIR=${BUILD_DIR:-"$ROOT/build"}
NATIVE=${ESHKOL_RUN:-"$BUILD_DIR/eshkol-run"}
VM=${ESHKOL_VM:-"$BUILD_DIR/eshkol-vm-standalone-test"}

for binary in "$NATIVE" "$VM"; do
    if [[ ! -x "$binary" ]]; then
        echo "INFRA: missing executable $binary" >&2
        exit 125
    fi
done

for fixture in \
    reject_define_syntax_special_form.esk \
    reject_let_syntax_special_form.esk \
    reject_letrec_syntax_special_form.esk; do
    for binary in "$NATIVE" "$VM"; do
        output=$("$binary" "$ROOT/tests/macros/$fixture" 2>&1) && {
            echo "FAIL: $binary accepted $fixture" >&2
            exit 1
        }
        grep -q "unsupported macro binding" <<<"$output" || {
            echo "FAIL: $binary did not explain $fixture" >&2
            printf '%s\n' "$output" >&2
            exit 1
        }
    done
done

for binary in "$NATIVE" "$VM"; do
    output=$("$binary" "$ROOT/tests/macros/basic_macro_test.esk" 2>&1) || {
        echo "FAIL: $binary rejected ordinary macro controls" >&2
        printf '%s\n' "$output" >&2
        exit 1
    }
    if grep -q '^FAIL:' <<<"$output"; then
        echo "FAIL: $binary reported a failed ordinary macro assertion" >&2
        printf '%s\n' "$output" >&2
        exit 1
    fi
    grep -q "PASS: forward ordinary macro reference" <<<"$output" || {
        echo "FAIL: $binary lost forward ordinary macro support" >&2
        exit 1
    }
done

echo "PASS: SW-192 rejects parser-lowered macro shadowing on native and VM"
