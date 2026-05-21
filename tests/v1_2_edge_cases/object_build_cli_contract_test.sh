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
#   - --profile freestanding-kernel-native requires --target and emits
#     an exact object path when the target is explicit

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

native_target_triple() {
    if [ -n "${ESHKOL_TEST_TARGET_TRIPLE:-}" ]; then
        printf '%s\n' "$ESHKOL_TEST_TARGET_TRIPLE"
        return
    fi

    case "$(uname -s):$(uname -m)" in
        Darwin:arm64) printf '%s\n' "arm64-apple-darwin" ;;
        Darwin:x86_64) printf '%s\n' "x86_64-apple-darwin" ;;
        Linux:x86_64) printf '%s\n' "x86_64-unknown-linux-gnu" ;;
        Linux:aarch64|Linux:arm64) printf '%s\n' "aarch64-unknown-linux-gnu" ;;
        *) printf '%s\n' "$(uname -m)-unknown-unknown" ;;
    esac
}

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

PROFILE_MISSING_TARGET_TMP="$WORK_TMP/profile-missing-target.out"
"$RUN" --profile freestanding-kernel-native \
    -o "$WORK_TMP/profile-missing-target.o" \
    "$WORK_TMP/src/object_contract.esk" \
    -I "$WORK_TMP/include" >"$PROFILE_MISSING_TARGET_TMP" 2>&1
rc=$?

check "--profile freestanding-kernel-native requires --target" \
    "$([ "$rc" -ne 0 ] && grep -q "requires --target" "$PROFILE_MISSING_TARGET_TMP" && echo yes || echo no)"

PROFILE_OBJECT_TMP="$WORK_TMP/profile-contract.o"
PROFILE_OUT_TMP="$WORK_TMP/profile-contract.out"
PROFILE_TARGET="$(native_target_triple)"
"$RUN" --profile freestanding-kernel-native \
    --target "$PROFILE_TARGET" \
    -o "$PROFILE_OBJECT_TMP" \
    "$WORK_TMP/src/object_contract.esk" \
    -I "$WORK_TMP/include" >"$PROFILE_OUT_TMP" 2>&1
rc=$?

check "--profile freestanding-kernel-native exits 0 with --target" \
    "$([ "$rc" -eq 0 ] && echo yes || echo no)"
check "--profile freestanding-kernel-native exact .o exists" \
    "$([ -f "$PROFILE_OBJECT_TMP" ] && echo yes || echo no)"
check "--profile freestanding-kernel-native does not append .o" \
    "$([ ! -e "$PROFILE_OBJECT_TMP.o" ] && echo yes || echo no)"

UNKNOWN_PROFILE_TMP="$WORK_TMP/unknown-profile.out"
"$RUN" --profile not-a-profile "$WORK_TMP/src/object_contract.esk" \
    -I "$WORK_TMP/include" >"$UNKNOWN_PROFILE_TMP" 2>&1
rc=$?

check "--profile rejects unknown names" \
    "$([ "$rc" -ne 0 ] && grep -q "Unknown execution profile" "$UNKNOWN_PROFILE_TMP" && echo yes || echo no)"

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -gt 0 ] && {
    echo "--- emit-object output ---"
    sed -n '1,40p' "$OUTPUT_TMP"
    echo "--- compile-only output ---"
    sed -n '1,40p' "$COMPILE_ONLY_OUT_TMP"
    echo "--- profile missing target output ---"
    sed -n '1,40p' "$PROFILE_MISSING_TARGET_TMP"
    echo "--- profile object output ---"
    sed -n '1,40p' "$PROFILE_OUT_TMP"
    echo "--- unknown profile output ---"
    sed -n '1,40p' "$UNKNOWN_PROFILE_TMP"
    exit 1
}
exit 0
