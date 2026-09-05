#!/usr/bin/env bash
# Exercise atomic ESKT tensor and ESKM model saves through all four engines.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${3:-${BUILD_DIR:-$ROOT/build}}"
ESHKOL_RUN="${1:-$BUILD_DIR/eshkol-run}"
VM="${2:-$BUILD_DIR/eshkol-vm-standalone-test}"
SOURCE="$ROOT/tests/core/atomic_checkpoint_save_test.esk"
export ESHKOL_PATH="$ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"
. "$ROOT/scripts/lib/durable_work_root.sh"
WORK_DURABLE=0
if [ -n "${ESHKOL_TEST_TMPDIR:-}" ]; then
    WORK="$(mktemp -d "$ESHKOL_TEST_TMPDIR/eshkol-atomic-save.XXXXXX")" || exit 125
    if eshkol_durable_enabled; then WORK_DURABLE=1; fi
elif eshkol_durable_enabled; then
    WORK="$(eshkol_durable_prepare_dir atomic-checkpoint-save)" || exit $?
    WORK_DURABLE=1
else
    TMP_BASE="${ESHKOL_TEST_TMP_ROOT:-${TMPDIR:-/tmp}}"
    WORK="$(mktemp -d "$TMP_BASE/eshkol-atomic-save.XXXXXX")" || exit 125
fi
KEEP="${ESHKOL_TEST_KEEP_TMPDIR:-}"

cleanup() {
    local rc=$?
    trap - EXIT
    if [ "$WORK_DURABLE" -eq 0 ] && [ -z "$KEEP" ]; then rm -rf -- "$WORK"; fi
    return "$rc"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [ ! -x "$ESHKOL_RUN" ] || [ ! -x "$VM" ]; then
    echo "INFRA: expected executable eshkol-run and VM paths" >&2
    exit 127
fi

# Every axis runs from an isolated output directory. Resolve caller-supplied
# relative executable paths before changing directories.
ESHKOL_RUN="$(cd "$(dirname "$ESHKOL_RUN")" && pwd -P)/$(basename "$ESHKOL_RUN")"
VM="$(cd "$(dirname "$VM")" && pwd -P)/$(basename "$VM")"

python3 - "$WORK" <<'PY'
import pathlib, struct, sys, zlib

root = pathlib.Path(sys.argv[1])

def checkpoint(records):
    data = bytearray(b"ESKM")
    data += struct.pack("<III", 1, len(records), 0)
    for name, shape, values in records:
        encoded = name.encode()
        data += struct.pack("<I", len(encoded)) + encoded
        data += struct.pack("<I", len(shape))
        data += b"".join(struct.pack("<Q", dim) for dim in shape)
        data += b"\0"
        data += b"".join(struct.pack("<d", value) for value in values)
    data += struct.pack("<I", zlib.crc32(data))
    return data

def tensor_checkpoint(values):
    return (b"TKSE" + struct.pack("<IIQ", 1, 1, len(values)) +
            b"".join(struct.pack("<d", value) for value in values))

for variant, tensor, model in (
    ("a", [1.5, -2.0], [5.0, 6.0, 7.0]),
    ("b", [9.0, 10.0], [50.0, 60.0, 70.0]),
):
    (root / f"tensor-{variant}.expected").write_bytes(tensor_checkpoint(tensor))
    (root / f"model-{variant}.expected").write_bytes(
        checkpoint([("weights", [2], model[:2]), ("bias", [1], model[2:])]))
PY

AOT="$WORK/atomic-save-aot"
ESKB="$WORK/atomic-save.eskb"
if ! "$ESHKOL_RUN" "$SOURCE" -o "$AOT" -L"$BUILD_DIR" \
        >"$WORK/aot-build.out" 2>"$WORK/aot-build.err"; then
    echo "INFRA: AOT fixture compilation failed" >&2
    sed -n '1,80p' "$WORK/aot-build.err" >&2
    exit 125
fi
if ! "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$ESKB" "$SOURCE" \
        >"$WORK/eskb-build.out" 2>"$WORK/eskb-build.err"; then
    echo "INFRA: VM bytecode fixture compilation failed" >&2
    sed -n '1,80p' "$WORK/eskb-build.err" >&2
    exit 125
fi

run_program() {
    local axis="$1" directory="$2" variant="$3" save_case="${4:-success}"
    local -a command=()
    case "$axis" in
        jit) command=("$ESHKOL_RUN" -r "$SOURCE" "-L$BUILD_DIR") ;;
        aot) command=("$AOT") ;;
        vm-source) command=("$VM" "$SOURCE") ;;
        vm-bytecode) command=("$VM" "$ESKB") ;;
        *) return 1 ;;
    esac
    (
        cd "$directory" || exit 125
        ESHKOL_VM_NO_DISASM=1 \
        ESHKOL_ATOMIC_SAVE_VARIANT="$variant" \
        ESHKOL_ATOMIC_SAVE_CASE="$save_case" \
        "${command[@]}"
    ) >"$directory/$axis.stdout" 2>"$directory/$axis.stderr"
}

assert_no_temp() {
    local directory="$1"
    local temporary
    for temporary in "$directory"/.eshkol.*; do
        if [ -e "$temporary" ] || [ -L "$temporary" ]; then
            echo "FAIL: orphan checkpoint temp: $temporary" >&2
            return 1
        fi
    done
}

mode_of() {
    python3 -c 'import os, stat, sys; print(oct(stat.S_IMODE(os.stat(sys.argv[1]).st_mode))[2:])' "$1"
}

prepare_existing() {
    local directory="$1"
    mkdir -p "$directory" || return 1
    printf 'original tensor checkpoint\n' >"$directory/tensor.eskm"
    printf 'original model checkpoint\n' >"$directory/model.eskm"
    chmod 604 "$directory/tensor.eskm"
    chmod 640 "$directory/model.eskm"
}

check_missing_parent_failure() {
    local axis="$1" directory="$WORK/$axis-missing-parent"
    mkdir -p "$directory" || return 1
    if ! run_program "$axis" "$directory" a missing-parent; then return 1; fi
    grep -qx 'ATOMIC-TENSOR:#f' "$directory/$axis.stdout" || return 1
    grep -qx 'ATOMIC-MODEL:#f' "$directory/$axis.stdout" || return 1
    grep -qx 'ATOMIC-TENSOR-LOAD:#f' "$directory/$axis.stdout" || return 1
    [ ! -e "$directory/missing-parent" ] || return 1
    assert_no_temp "$directory"
}

check_directory_destination_failure() {
    local axis="$1" directory="$WORK/$axis-directory-destination"
    mkdir -p "$directory/tensor.eskm" "$directory/model.eskm" || return 1
    printf 'tensor directory canary\n' >"$directory/tensor.eskm/canary"
    printf 'model directory canary\n' >"$directory/model.eskm/canary"
    if ! run_program "$axis" "$directory" a directory-destination; then return 1; fi
    grep -qx 'ATOMIC-TENSOR:#f' "$directory/$axis.stdout" || return 1
    grep -qx 'ATOMIC-MODEL:#f' "$directory/$axis.stdout" || return 1
    grep -qx 'ATOMIC-TENSOR-LOAD:#f' "$directory/$axis.stdout" || return 1
    grep -qx 'tensor directory canary' "$directory/tensor.eskm/canary" || return 1
    grep -qx 'model directory canary' "$directory/model.eskm/canary" || return 1
    assert_no_temp "$directory"
}

check_success_case() {
    local axis="$1" directory="$WORK/$axis-success"
    prepare_existing "$directory" || return 1
    if ! run_program "$axis" "$directory" a; then return 1; fi
    grep -qx 'ATOMIC-TENSOR:#t' "$directory/$axis.stdout" || return 1
    grep -qx 'ATOMIC-MODEL:#t' "$directory/$axis.stdout" || return 1
    grep -qx 'ATOMIC-TENSOR-LOAD:#t' "$directory/$axis.stdout" || return 1
    cmp -s "$WORK/tensor-a.expected" "$directory/tensor.eskm" || return 1
    cmp -s "$WORK/model-a.expected" "$directory/model.eskm" || return 1
    [ "$(mode_of "$directory/tensor.eskm")" = 604 ] || return 1
    [ "$(mode_of "$directory/model.eskm")" = 640 ] || return 1
    assert_no_temp "$directory"
}

FAILED=0
for axis in jit aot vm-source vm-bytecode; do
    if check_success_case "$axis"; then
        echo "PASS: $axis exact ESKT/ESKM atomic replacement"
    else
        echo "FAIL: $axis successful replacement contract" >&2
        FAILED=1
    fi
    if ! check_missing_parent_failure "$axis"; then
        echo "FAIL: $axis missing-parent failure contract" >&2
        FAILED=1
    fi
    if ! check_directory_destination_failure "$axis"; then
        echo "FAIL: $axis destination-directory failure contract" >&2
        FAILED=1
    fi
done

for axis in jit vm-source; do
    directory="$WORK/$axis-symlink"
    mkdir -p "$directory"
    printf 'tensor referent\n' >"$directory/tensor-target"
    printf 'model referent\n' >"$directory/model-target"
    ln -s tensor-target "$directory/tensor.eskm"
    ln -s model-target "$directory/model.eskm"
    if run_program "$axis" "$directory" a &&
       [ ! -L "$directory/tensor.eskm" ] && [ ! -L "$directory/model.eskm" ] &&
       grep -qx 'tensor referent' "$directory/tensor-target" &&
       grep -qx 'model referent' "$directory/model-target" &&
       cmp -s "$WORK/tensor-a.expected" "$directory/tensor.eskm" &&
       cmp -s "$WORK/model-a.expected" "$directory/model.eskm" &&
       [ "$(mode_of "$directory/tensor.eskm")" = 600 ] &&
       [ "$(mode_of "$directory/model.eskm")" = 600 ] &&
       assert_no_temp "$directory"; then
        echo "PASS: $axis replaces symlinks without touching referents"
    else
        echo "FAIL: $axis symlink contract" >&2
        FAILED=1
    fi
done

CONCURRENT="$WORK/concurrent"
mkdir -p "$CONCURRENT"
pids=()
index=0
for axis in jit aot vm-source vm-bytecode; do
    variant=a
    [ $((index % 2)) -eq 1 ] && variant=b
    run_program "$axis" "$CONCURRENT" "$variant" concurrent &
    pids+=("$!")
    index=$((index + 1))
done
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then FAILED=1; fi
done
for axis in jit aot vm-source vm-bytecode; do
    if ! grep -qx 'ATOMIC-TENSOR:#t' "$CONCURRENT/$axis.stdout" ||
       ! grep -qx 'ATOMIC-MODEL:#t' "$CONCURRENT/$axis.stdout" ||
       ! grep -qx 'ATOMIC-TENSOR-LOAD:#t' "$CONCURRENT/$axis.stdout"; then
        echo "FAIL: $axis concurrent writer reported save failure" >&2
        FAILED=1
    fi
done
if { cmp -s "$WORK/tensor-a.expected" "$CONCURRENT/tensor.eskm" ||
     cmp -s "$WORK/tensor-b.expected" "$CONCURRENT/tensor.eskm"; } &&
   { cmp -s "$WORK/model-a.expected" "$CONCURRENT/model.eskm" ||
     cmp -s "$WORK/model-b.expected" "$CONCURRENT/model.eskm"; } &&
   [ "$(mode_of "$CONCURRENT/tensor.eskm")" = 600 ] &&
   [ "$(mode_of "$CONCURRENT/model.eskm")" = 600 ] &&
   assert_no_temp "$CONCURRENT"; then
    echo "PASS: concurrent writers publish only complete checkpoints"
else
    echo "FAIL: concurrent writer contract" >&2
    FAILED=1
fi

if [ "$FAILED" -ne 0 ]; then exit 1; fi
echo "PASS: atomic checkpoint saves (jit/aot/vm-source/vm-bytecode)"
