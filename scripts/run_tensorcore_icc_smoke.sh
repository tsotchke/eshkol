#!/usr/bin/env bash
# Campaign-scoped ICC evidence for Eshkol's canonical TensorCore adapter.
set -u

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)
TRACE_DIR="$REPO_ROOT/scripts/icc_traces"
TRACE_FILE="$TRACE_DIR/tensorcore_adapter.jsonl"
ENABLED_BUILD=${ESHKOL_TENSORCORE_ENABLED_BUILD:-/private/tmp/eshkol-tensorcore-enabled}
DISABLED_BUILD=${ESHKOL_TENSORCORE_DISABLED_BUILD:-/private/tmp/eshkol-tensorcore-disabled}
HARDWARE_BUILD=${ESHKOL_TENSORCORE_HARDWARE_BUILD:-/private/tmp/eshkol-tensorcore-metal}
TENSORCORE_PREFIX=${ESHKOL_TENSORCORE_PREFIX:-/private/tmp/tensorcore-sdk-campaign-0.1.22}
LEGACY_TENSORCORE_PREFIX=${ESHKOL_TENSORCORE_LEGACY_PREFIX:-/private/tmp/tensorcore-sdk-0.1.22}
ESHKOL_INSTALL_PREFIX=${ESHKOL_TENSORCORE_INSTALL_PREFIX:-/private/tmp/eshkol-sdk-tensorcore}
FAILURES=0

mkdir -p "$TRACE_DIR"
: > "$TRACE_FILE"

emit_event() {
    local name=$1 status=$2 snippet=$3 escaped
    escaped=$(printf '%s' "$snippet" |
        tr '\r\n' '  ' |
        sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')
    printf '{"kind":"eshkol_tensorcore","name":"%s","value":"%s","snippet":"%s","confidence":0.99}\n' \
        "$name" "$status" "$escaped" >> "$TRACE_FILE"
}

probe() {
    local name=$1 label=$2 command=$3 output status snippet
    output=$(eval "$command" 2>&1)
    status=$?
    snippet=$(printf '%s' "$output" | tail -c 500)
    if [ "$status" -eq 0 ]; then
        emit_event "$name" PASS "$label: $snippet"
        printf '  ✓ %-38s %s\n' "$name" "$label"
    else
        emit_event "$name" FAIL "$label: $snippet"
        printf '  ✗ %-38s %s (exit %d)\n' "$name" "$label" "$status"
        FAILURES=$((FAILURES + 1))
    fi
}

echo "Running TensorCore ICC probes → $TRACE_FILE"
echo

probe tensorcore_explicit_unavailable \
    "disabled build returns the stable unavailable protocol" \
    'ctest --test-dir "$DISABLED_BUILD" --output-on-failure -R "^tensorcore_adapter_test$"'

probe tensorcore_codegen_verify_module \
    "canonical 34-symbol lowering passes verifyModule and rejects mixed signatures" \
    'ctest --test-dir "$ENABLED_BUILD" --output-on-failure -R "^tensorcore_codegen_test$"'

probe tensorcore_portable_runtime \
    "installed public capability ABI fails closed and lifecycle, buffers, GEMM, and attention execute" \
    'ctest --test-dir "$ENABLED_BUILD" --output-on-failure -R "^tensorcore_adapter_test$"'

probe tensorcore_compatibility_parity \
    "canonical and compatibility shims return identical GEMM/diagnostics" \
    'ctest --test-dir "$ENABLED_BUILD" --output-on-failure -R "^tensorcore_compatibility_test$"'

probe tensorcore_language_aot \
    "(require tensorcore) compiles, links, and runs through the installed package" \
    'ctest --test-dir "$ENABLED_BUILD" --output-on-failure -R "^tensorcore_language_aot_smoke$"'

probe tensorcore_installed_package \
    "installed Eshkol headers, module, compiler, and dependency rpath are usable" \
    'test -f "$ESHKOL_INSTALL_PREFIX/include/eshkol/tensorcore_adapter.h" &&
     test -f "$ESHKOL_INSTALL_PREFIX/include/eshkol/backend/tensorcore_codegen.h" &&
     test -f "$ESHKOL_INSTALL_PREFIX/share/eshkol/lib/tensorcore.esk" &&
     ESHKOL_PATH="$ESHKOL_INSTALL_PREFIX/share/eshkol/lib" \
     ESHKOL_JIT_CACHE_DIR="${TMPDIR:-/tmp}/eshkol-installed-jit-cache" \
       "$ESHKOL_INSTALL_PREFIX/bin/eshkol-run" -r \
       "$REPO_ROOT/tests/backend/tensorcore_language_smoke.esk" | grep -q "#t"'

probe tensorcore_hardware_runtime \
    "installed hardware package executes through a real serving backend" \
    'TC_TRACE=1 ctest --test-dir "$HARDWARE_BUILD" -V -R "^tensorcore_adapter_test$" 2>&1 | grep -q "backend=mps"'

probe tensorcore_capability_abi_required \
    "packages without tc_runtime_capabilities_get ABI v1 are rejected" \
    'reject_dir=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-tc-capability-reject.XXXXXX");
     if cmake -S "$REPO_ROOT" -B "$reject_dir" \
          -DESHKOL_TENSORCORE_ENABLED=ON \
          -Dtensorcore_DIR="$LEGACY_TENSORCORE_PREFIX/lib/cmake/tensorcore" \
          >"$reject_dir/configure.log" 2>&1; then
       exit 1;
     fi;
     grep -q "does not provide the required tc_runtime_capabilities_get" \
       "$reject_dir/configure.log"'

probe tensorcore_version_rejection \
    "unsupported installed ABI versions fail configuration deterministically" \
    'reject_dir=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-tc-version-reject.XXXXXX");
     if cmake -S "$REPO_ROOT" -B "$reject_dir" \
          -DESHKOL_TENSORCORE_ENABLED=ON \
          -DESHKOL_TENSORCORE_MIN_VERSION=0.1.23 \
          -DCMAKE_PREFIX_PATH="$TENSORCORE_PREFIX" >"$reject_dir/configure.log" 2>&1; then
       exit 1;
     fi;
     grep -q "TensorCore 0.1.22 is too old" "$reject_dir/configure.log"'

echo
echo "Trace written: $TRACE_FILE"

if [ "$FAILURES" -ne 0 ]; then
    echo "TensorCore ICC campaign failed: $FAILURES probe(s) did not pass" >&2
    exit 1
fi
