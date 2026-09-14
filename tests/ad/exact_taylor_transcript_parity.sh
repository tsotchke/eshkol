#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: $0 NATIVE VM SOURCE WORK" >&2
    exit 2
fi

native="$1"
vm="$2"
source_file="$3"
work="$4"
mkdir -p "$work"

"$native" -r "$source_file" -L"$(dirname "$native")" >"$work/native.raw" 2>&1
ESHKOL_VM_NO_DISASM=1 ESHKOL_PATH="$(cd "$(dirname "$source_file")/../.." && pwd)/lib" \
    "$vm" "$source_file" >"$work/vm.raw" 2>&1

# Engine banners and compile diagnostics are transport framing.  The remaining
# stream is the program's external transcript and must compare byte-for-byte.
sed -E '/NOTICE:|remark:|warning:|^\x1b|^=== Eshkol|^=== Execution|^$/d' \
    "$work/native.raw" >"$work/native.transcript"
sed -E '/^=== Eshkol|^=== Execution|^$/d' \
    "$work/vm.raw" >"$work/vm.transcript"

if ! cmp -s "$work/native.transcript" "$work/vm.transcript"; then
    echo "FAIL: exact Taylor native/VM transcripts differ" >&2
    diff -u "$work/native.transcript" "$work/vm.transcript" >&2 || true
    exit 1
fi

echo "PASS: exact Taylor native/VM transcripts byte-identical"
