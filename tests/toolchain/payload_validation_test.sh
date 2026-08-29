#!/usr/bin/env bash
# payload_validation_test.sh — issue #549's malformed checkpoint and missing
# file reproducer across native JIT, native AOT, and the hosted bytecode VM.

set -u

ESHKOL_RUN="${1:-}"
VM_BIN="${2:-}"
SOURCE="$(cd "$(dirname "$0")/.." && pwd)/v1_3_edge_cases/payload_validation_test.esk"

if [ -z "$ESHKOL_RUN" ] || [ ! -x "$ESHKOL_RUN" ]; then
    echo "FAIL: payload_validation_test could not locate eshkol-run" >&2
    exit 1
fi
if [ -z "$VM_BIN" ] || [ ! -x "$VM_BIN" ]; then
    echo "FAIL: payload_validation_test could not locate the hosted VM" >&2
    exit 1
fi

case "$ESHKOL_RUN" in /*) ;; *) ESHKOL_RUN="$(pwd)/$ESHKOL_RUN" ;; esac
case "$VM_BIN" in /*) ;; *) VM_BIN="$(pwd)/$VM_BIN" ;; esac

WORK="$PWD/.scratch/issue-549-payload-validation"
rm -rf "$WORK"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

assert_output() {
    local label="$1" output="$2"
    # The standalone VM's display primitive appends a newline, whereas native
    # display does not. Compare semantic markers across that documented output
    # formatting difference.
    local compact
    compact="$(printf '%s' "$output" | tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g')"
    for marker in \
        'tensor-load rejected' \
        'model-load rejected' \
        'truncated rejected' \
        'missing-file caught' \
        'invalid-port caught' \
        'payload-validation-test complete'; do
        case "$compact" in
            *"$marker"*) ;;
            *) echo "FAIL: $label did not print '$marker'" >&2; return 1 ;;
        esac
    done
}

run_native_jit() {
    local dir="$WORK/native-jit"
    mkdir -p "$dir"
    (cd "$dir" && ESHKOL_JIT_CACHE_DIR="$dir/jit-cache" "$ESHKOL_RUN" -r "$SOURCE")
}

run_native_aot() {
    local dir="$WORK/native-aot" bin="$WORK/native-aot/payload-validation"
    mkdir -p "$dir"
    "$ESHKOL_RUN" "$SOURCE" -o "$bin" >/dev/null
    (cd "$dir" && "$bin")
}

run_vm() {
    local dir="$WORK/vm" eskb="$WORK/vm/payload-validation.eskb"
    mkdir -p "$dir"
    "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$eskb" "$SOURCE" >/dev/null
    (cd "$dir" && "$VM_BIN" "$eskb")
}

for engine in native-jit native-aot vm; do
    case "$engine" in
        native-jit) output="$(run_native_jit 2>&1)"; rc=$? ;;
        native-aot) output="$(run_native_aot 2>&1)"; rc=$? ;;
        vm)         output="$(run_vm 2>&1)"; rc=$? ;;
    esac
    if [ "$rc" -ne 0 ]; then
        echo "FAIL: $engine exited $rc: $output" >&2
        exit 1
    fi
    if ! assert_output "$engine" "$output"; then
        echo "--- $engine output ---" >&2
        echo "$output" >&2
        exit 1
    fi
    echo "  ok: $engine rejected malformed payloads and caught missing/invalid ports"
done

echo "PASS: payload_validation_test"
