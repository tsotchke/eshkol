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

WORK_TMP=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-object-cli.XXXXXX") || exit 1
cleanup() {
    [ -n "${WORK_TMP:-}" ] && [ -d "$WORK_TMP" ] && rm -rf -- "$WORK_TMP"
}
trap cleanup EXIT

mkdir -p "$WORK_TMP/src" "$WORK_TMP/include"

cat > "$WORK_TMP/include/object_dep.esk" <<'ESK'
(define (object-contract-value) 42)
ESK

cat > "$WORK_TMP/src/object_contract.esk" <<'ESK'
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

REQUESTED_TMP="$WORK_TMP/requested.o"
OUTPUT_TMP="$WORK_TMP/emit.out"

"$RUN" \
    --emit-object \
    -o "$REQUESTED_TMP" \
    --shared-lib \
    -fPIC \
    -I "$WORK_TMP/include" \
    -D NOESIS_OBJECT_CONTRACT=1 \
    "$WORK_TMP/src/object_contract.esk" >"$OUTPUT_TMP" 2>&1
rc=$?

check "--emit-object command exits 0" "$([ "$rc" -eq 0 ] && echo yes || echo no)"
check "requested object exists exactly" "$([ -f "$REQUESTED_TMP" ] && echo yes || echo no)"
check "stale .o.o output is not created" "$([ ! -e "$REQUESTED_TMP.o" ] && echo yes || echo no)"

COMPILE_ONLY_TMP="$WORK_TMP/compile-only.o"
COMPILE_ONLY_OUT_TMP="$WORK_TMP/compile-only.out"
"$RUN" --compile-only -o "$COMPILE_ONLY_TMP" "$WORK_TMP/src/object_contract.esk" \
    -I "$WORK_TMP/include" >"$COMPILE_ONLY_OUT_TMP" 2>&1
rc=$?

check "--compile-only exact .o exits 0" "$([ "$rc" -eq 0 ] && echo yes || echo no)"
check "--compile-only exact .o exists" "$([ -f "$COMPILE_ONLY_TMP" ] && echo yes || echo no)"
check "--compile-only does not append .o" "$([ ! -e "$COMPILE_ONLY_TMP.o" ] && echo yes || echo no)"

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -gt 0 ] && {
    echo "--- emit-object output ---"
    sed -n '1,40p' "$OUTPUT_TMP"
    echo "--- compile-only output ---"
    sed -n '1,40p' "$COMPILE_ONLY_OUT_TMP"
    exit 1
}
exit 0
