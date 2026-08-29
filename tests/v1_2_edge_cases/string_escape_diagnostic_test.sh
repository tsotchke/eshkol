#!/usr/bin/env bash
# Malformed string escapes must fail with a source location on native and VM
# source readers; they must never become literal text with exit status zero.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"
VM="$ROOT/${BUILD_DIR:-build}/eshkol-vm-standalone-test"
VALID="$ROOT/tests/v1_2_edge_cases/string_semantics_regression_test.esk"
PARITY="$ROOT/tests/vm_parity/corpus/73_string_escape_nul_search.esk"
WORK="$ROOT/.scratch/string-escape-diagnostic.$$"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/malformed.esk" <<'EOF'
(display "bad \xZZ")
EOF

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

if "$RUN" -n "$WORK/malformed.esk" -o "$WORK/native" >"$WORK/native.out" 2>"$WORK/native.err"; then
    echo "FAIL: native accepted malformed \\x escape"
    exit 1
fi
if ! grep -Eq 'malformed string escape|incomplete string escape' "$WORK/native.err" ||
   ! grep -q 'malformed.esk:' "$WORK/native.err"; then
    echo "FAIL: native diagnostic was not sourceful"
    sed -n '1,20p' "$WORK/native.err"
    exit 1
fi

if ! "$RUN" -r "$VALID" >"$WORK/jit.out" 2>"$WORK/jit.err"; then
    echo "FAIL: native JIT string regression"
    sed -n '1,30p' "$WORK/jit.err"
    exit 1
fi
if ! grep -Eq 'FAIL: [1-9][0-9]*' "$WORK/jit.out" && grep -q 'Total:' "$WORK/jit.out"; then
    :
else
    echo "FAIL: native JIT string regression assertions"
    sed -n '1,40p' "$WORK/jit.out"
    exit 1
fi

if ! "$RUN" "$VALID" -o "$WORK/valid-aot" >"$WORK/aot-compile.out" 2>"$WORK/aot-compile.err" ||
   ! "$WORK/valid-aot" >"$WORK/aot.out" 2>"$WORK/aot.err"; then
    echo "FAIL: native AOT string regression"
    sed -n '1,30p' "$WORK/aot-compile.err" "$WORK/aot.err"
    exit 1
fi
if grep -Eq 'FAIL: [1-9][0-9]*' "$WORK/aot.out" || ! grep -q 'Total:' "$WORK/aot.out"; then
    echo "FAIL: native AOT string regression assertions"
    sed -n '1,40p' "$WORK/aot.out"
    exit 1
fi

if [ -x "$VM" ]; then
    if ESHKOL_VM_NO_DISASM=1 "$VM" "$WORK/malformed.esk" >"$WORK/vm.out" 2>"$WORK/vm.err"; then
        echo "FAIL: VM accepted malformed \\x escape"
        exit 1
    fi
    if ! grep -Eq 'malformed string escape|incomplete string escape' "$WORK/vm.err"; then
        echo "FAIL: VM diagnostic did not identify malformed string escape"
        sed -n '1,20p' "$WORK/vm.err"
        exit 1
    fi
    if ! ESHKOL_VM_NO_DISASM=1 "$VM" "$PARITY" >"$WORK/valid-vm.out" 2>"$WORK/valid-vm.err" ||
       grep -Eq 'FAIL: [1-9][0-9]*' "$WORK/valid-vm.out" ||
       ! grep -q 'OK string-escape-nul-search' "$WORK/valid-vm.out"; then
        echo "FAIL: VM string regression assertions"
        sed -n '1,40p' "$WORK/valid-vm.out" "$WORK/valid-vm.err"
        exit 1
    fi
fi

echo "PASS: malformed string escapes are rejected with diagnostics"
