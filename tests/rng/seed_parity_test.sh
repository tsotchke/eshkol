#!/usr/bin/env bash
set -euo pipefail

RUN=${1:?missing eshkol-run}
VM=${2:?missing VM executable}
REPO=${3:?missing repository root}
mkdir -p "$REPO/.scratch"
WORK=$(mktemp -d "$REPO/.scratch/rng-seed-parity.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

export ESHKOL_PATH="$REPO/lib"

normalize() {
    tr -d '\r\n' < "$1" |
        sed -E 's/^\[ESKB\] Loaded .* constants=== Eshkol VM — running [^=]*===//; s/=== Execution complete ===$//'
}

run_case() {
    local name=$1
    local source=$2

    "$RUN" -n -r "$source" > "$WORK/$name.jit.out" 2> "$WORK/$name.jit.err"
    "$RUN" -n -o "$WORK/$name.aot" "$source" > "$WORK/$name.aot.compile.out" \
        2> "$WORK/$name.aot.compile.err"
    "$WORK/$name.aot" > "$WORK/$name.aot.out" 2> "$WORK/$name.aot.err"

    "$RUN" --profile hosted-vm --emit-eskb "$WORK/$name.eskb" "$source" \
        > "$WORK/$name.vm.compile.out" 2> "$WORK/$name.vm.compile.err"
    ESHKOL_VM_NO_DISASM=1 "$VM" "$WORK/$name.eskb" \
        > "$WORK/$name.vm.out" 2> "$WORK/$name.vm.err"

    local jit aot vm
    jit=$(normalize "$WORK/$name.jit.out")
    aot=$(normalize "$WORK/$name.aot.out")
    vm=$(normalize "$WORK/$name.vm.out")
    if [[ "$jit" != "$aot" || "$jit" != "$vm" ]]; then
        echo "FAIL: $name differs across JIT, AOT, and VM" >&2
        printf 'JIT: %s\nAOT: %s\nVM:  %s\n' "$jit" "$aot" "$vm" >&2
        exit 1
    fi
    printf '%s\n' "$jit"
}

scalar=$(run_case scalar "$REPO/tests/rng/seed_scalar_parity.esk")
expected_scalar='0.9687044341853870.95638303397888080.62642566141779450.79286341331584960.063807073410391270.26798134342242010.9449023081891390.9406623079385881'
if [[ "$scalar" != "$expected_scalar" ]]; then
    echo "FAIL: seeded scalar sequence changed" >&2
    printf 'expected: %s\nactual:   %s\n' "$expected_scalar" "$scalar" >&2
    exit 1
fi

adversarial=$(run_case adversarial "$REPO/tests/rng/seed_adversarial.esk")
expected_adversarial='0.170828036106289720.170828036106289720.30002572744070120.17082803610628972'
if [[ "$adversarial" != "$expected_adversarial" ]]; then
    echo "FAIL: reseeding or low-32-bit seed handling changed" >&2
    printf 'expected: %s\nactual:   %s\n' "$expected_adversarial" "$adversarial" >&2
    exit 1
fi

run_case issue_553 "$REPO/tests/vm_parity/corpus/58_rng_seed_parity.esk" > "$WORK/issue_553.sequence"
echo "PASS: SW-113 seeded PRNG sequence agrees across JIT, AOT, and VM"
