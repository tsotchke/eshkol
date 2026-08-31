#!/usr/bin/env bash
# The host shell is intentional: this gate orchestrates the Eshkol compiler,
# the platform linker driver, and the produced native executable. The semantic
# fixture itself is Eshkol and exercises the smallest no-stdlib result surface.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
. "$REPO_ROOT/scripts/lib/durable_work_root.sh"

ESHKOL_RUN="${1:-${ESHKOL_RUN:-}}"
if [[ -z "$ESHKOL_RUN" || ! -x "$ESHKOL_RUN" ]]; then
    echo "FAIL: no_stdlib_aot_driver_test could not locate an executable eshkol-run" >&2
    exit 1
fi
ESHKOL_RUN="$(cd "$(dirname "$ESHKOL_RUN")" && pwd)/$(basename "$ESHKOL_RUN")"

if eshkol_durable_enabled; then
    WORK="$(eshkol_durable_prepare_dir no-stdlib-aot-driver)" || exit $?
else
    WORK="$REPO_ROOT/.scratch/no-stdlib-aot-driver-$$"
    mkdir -p "$WORK"
    trap 'rm -rf "$WORK"' EXIT
fi

fail() {
    echo "FAIL: no_stdlib_aot_driver_test - $1" >&2
    exit 1
}

SOURCE="$WORK/zero.esk"
cat > "$SOURCE" <<'EOF'
(display 0)
(newline)
EOF

if ! env -u ESHKOL_CXX_COMPILER "$ESHKOL_RUN" -n -r "$SOURCE" \
        >"$WORK/jit.out" 2>"$WORK/jit.err"; then
    fail "no-stdlib JIT execution failed"
fi
[[ "$(tr -d '\r\n' < "$WORK/jit.out")" == "0" ]] ||
    fail "no-stdlib JIT did not print the expected zero"

AOT_BIN="$WORK/zero-aot"
if ! env -u ESHKOL_CXX_COMPILER "$ESHKOL_RUN" -d -n "$SOURCE" -o "$AOT_BIN" \
        >"$WORK/aot-build.log" 2>&1; then
    sed -n '1,160p' "$WORK/aot-build.log" >&2
    fail "no-stdlib AOT compile/link failed"
fi
[[ -x "$AOT_BIN" ]] || fail "no-stdlib AOT produced no executable"
if ! "$AOT_BIN" >"$WORK/aot.out" 2>"$WORK/aot.err"; then
    fail "no-stdlib AOT executable failed"
fi
[[ "$(tr -d '\r\n' < "$WORK/aot.out")" == "0" ]] ||
    fail "no-stdlib AOT did not print the expected zero"

# Negative control: an explicitly unusable host C++ driver must make the link
# fail and must leave no executable. If this unexpectedly passes, the positive
# result above did not prove the configured driver-selection path at all.
BROKEN_BIN="$WORK/zero-broken-driver"
if ESHKOL_CXX_COMPILER=/usr/bin/false "$ESHKOL_RUN" -n "$SOURCE" -o "$BROKEN_BIN" \
        >"$WORK/broken-driver.log" 2>&1; then
    fail "AOT ignored the explicit broken host C++ driver"
fi
[[ ! -e "$BROKEN_BIN" ]] || fail "failed link left a native executable behind"

echo "PASS: no_stdlib_aot_driver_test"
