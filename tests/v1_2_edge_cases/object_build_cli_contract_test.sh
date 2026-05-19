#!/bin/bash
# object_build_cli_contract_test.sh - Noesis Bug LL positive regression.
#
# Build systems need an exact object-output contract. This verifies the
# compatibility surface Noesis expects:
#   --emit-object -o <exact.o> [--shared-lib] [-fPIC] [-I dir] [-D name=value]
#
# The important regression points are:
#   - --emit-object is accepted as an alias for --compile-only
#   - -o requested.o creates exactly requested.o, not requested.o.o
#   - -I contributes to module/load resolution
#   - -D and -fPIC are accepted for build-system compatibility

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-object-cli.XXXXXX") || exit 1
cleanup() {
    [ -n "${WORK:-}" ] && [ -d "$WORK" ] && rm -rf -- "$WORK"
}
trap cleanup EXIT

mkdir -p "$WORK/src" "$WORK/include"

cat > "$WORK/include/object_dep.esk" <<'ESK'
(define (object-contract-value) 42)
ESK

cat > "$WORK/src/object_contract.esk" <<'ESK'
(load "object_dep.esk")
(define (object-contract-entry) (object-contract-value))
ESK

PASS=0
FAIL=0

check() {
    label="$1"
    condition="$2"
    if [ "$condition" = "yes" ]; then
        echo "PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $label"
        FAIL=$((FAIL + 1))
    fi
}

requested="$WORK/requested.o"
output="$WORK/emit.out"

"$RUN" \
    --emit-object \
    -o "$requested" \
    --shared-lib \
    -fPIC \
    -I "$WORK/include" \
    -D NOESIS_OBJECT_CONTRACT=1 \
    "$WORK/src/object_contract.esk" >"$output" 2>&1
rc=$?

check "--emit-object command exits 0" "$([ "$rc" -eq 0 ] && echo yes || echo no)"
check "requested object exists exactly" "$([ -f "$requested" ] && echo yes || echo no)"
check "stale .o.o output is not created" "$([ ! -e "$requested.o" ] && echo yes || echo no)"

compile_only="$WORK/compile-only.o"
"$RUN" --compile-only -o "$compile_only" "$WORK/src/object_contract.esk" \
    -I "$WORK/include" >"$WORK/compile-only.out" 2>&1
rc=$?

check "--compile-only exact .o exits 0" "$([ "$rc" -eq 0 ] && echo yes || echo no)"
check "--compile-only exact .o exists" "$([ -f "$compile_only" ] && echo yes || echo no)"
check "--compile-only does not append .o" "$([ ! -e "$compile_only.o" ] && echo yes || echo no)"

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -gt 0 ] && {
    echo "--- emit-object output ---"
    sed -n '1,40p' "$output"
    echo "--- compile-only output ---"
    sed -n '1,40p' "$WORK/compile-only.out"
    exit 1
}
exit 0
