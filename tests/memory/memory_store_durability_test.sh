#!/usr/bin/env bash
# Cross-engine gate for the core.memory event-log and core.memory_store journal.
#
# The durable journal uses native FFI and is therefore exercised through native
# JIT and AOT. The core.memory CRDT faculty is also run by the standalone VM;
# its result summary must agree with both native modes.

set -u
export LC_ALL=C LC_CTYPE=C LANG=C

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
case "$BUILD_DIR" in
    /*) ;;
    *) BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;;
esac

ESHKOL_RUN="${ESHKOL_RUN:-$BUILD_DIR/eshkol-run}"
VM_BIN="${ESHKOL_VM:-$BUILD_DIR/eshkol-vm-standalone-test}"
if [ "$#" -ge 1 ]; then ESHKOL_RUN="$1"; fi
if [ "$#" -ge 2 ]; then VM_BIN="$2"; fi
# Keep this fixture outside tests/memory/*.esk: it requires this wrapper's
# isolated scratch root and memory-store environment.
DURABILITY_SRC="$REPO_ROOT/tests/memory/fixtures/memory_store_durability_test.esk"
MEMORY_SRC="$REPO_ROOT/tests/memory/memory_test.esk"
VM_MEMORY_SRC="$REPO_ROOT/tests/memory/memory_vm_parity_test.esk"
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
TRACE_FILE="$TRACE_DIR/memory_store_durability.jsonl"

for required in "$ESHKOL_RUN" "$VM_BIN" "$DURABILITY_SRC" "$MEMORY_SRC" "$VM_MEMORY_SRC"; do
    if [ ! -e "$required" ]; then
        echo "memory_store_durability_test.sh: missing $required" >&2
        exit 2
    fi
done
if [ ! -x "$ESHKOL_RUN" ] || [ ! -x "$VM_BIN" ]; then
    echo "memory_store_durability_test.sh: native runner and standalone VM must be executable" >&2
    exit 2
fi

mkdir -p "$REPO_ROOT/.scratch"
mkdir -p "$TRACE_DIR"
: >"$TRACE_FILE"
WORK="$(mktemp -d "$REPO_ROOT/.scratch/memory-store-durability-gate.XXXXXX")" || exit 2
trap 'rm -rf -- "$WORK"' EXIT INT TERM

run_and_check() {
    label="$1"
    expected="$2"
    shift 2
    output="$WORK/$label.out"
    if "$@" >"$output" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    if [ "$rc" -ne 0 ] || ! grep -q "$expected" "$output" || grep -q '^FAIL:' "$output"; then
        printf '{"kind":"test_result","name":"memory_store.%s","value":{"passed":false,"summary":"exit=%s"},"timestamp":%s}\n' "$label" "$rc" "$(date +%s)" >>"$TRACE_FILE"
        echo "FAIL: $label (exit=$rc)"
        sed -n '1,240p' "$output"
        return 1
    fi
    printf '{"kind":"test_result","name":"memory_store.%s","value":{"passed":true,"summary":"%s"},"timestamp":%s}\n' "$label" "$expected" "$(date +%s)" >>"$TRACE_FILE"
    echo "PASS: $label"
    grep -E '^(Passed:|Failed:|  \[PASS\]|  \[FAIL\])' "$output" | tail -8
    return 0
}

export ESHKOL_PATH="$REPO_ROOT/lib"
export ESHKOL_MEMORY_STORE_TEST_ROOT="$WORK/journal"
mkdir -p "$ESHKOL_MEMORY_STORE_TEST_ROOT"
export ESHKOL_JIT_CACHE_DIR="$WORK/jit-cache"
mkdir -p "$ESHKOL_JIT_CACHE_DIR"

failed=0

run_and_check native_jit_durability 'Failed: 0' \
    "$ESHKOL_RUN" --strict-types -r "$DURABILITY_SRC" || failed=1

AOT_BIN="$WORK/memory-store-durability-aot"
AOT_LOG="$WORK/aot-compile.out"
if "$ESHKOL_RUN" --strict-types "$DURABILITY_SRC" -o "$AOT_BIN" >"$AOT_LOG" 2>&1; then
    chmod +x "$AOT_BIN"
    run_and_check native_aot_durability 'Failed: 0' "$AOT_BIN" || failed=1
else
    echo "FAIL: native_aot_durability_compile"
    sed -n '1,240p' "$AOT_LOG"
    failed=1
fi

run_and_check native_jit_memory 'Failed: 0' \
    "$ESHKOL_RUN" --strict-types -r "$MEMORY_SRC" || failed=1

run_and_check native_jit_compatibility 'Failed: 0' \
    "$ESHKOL_RUN" --strict-types -r "$REPO_ROOT/tests/memory/memory_store_test.esk" || failed=1

run_and_check vm_memory_dependency 'PASS: VM vector-clock substrate for core.memory' \
    "$VM_BIN" "$VM_MEMORY_SRC" || failed=1

if [ "$failed" -ne 0 ]; then
    echo "memory_store_durability_test.sh: FAIL"
    exit 1
fi
echo "memory_store_durability_test.sh: PASS"
