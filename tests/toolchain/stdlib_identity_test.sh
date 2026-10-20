#!/usr/bin/env bash
set -euo pipefail

ESHKOL_RUN="$1"
EXPECTED_IDENTITY="$2"
RELEASE="$3"
OBJECT_ABI="$4"
SOURCE_FINGERPRINT="$5"
WORK_ROOT="$6"
LLVM_AS="${7:-}"
LLC="${8:-}"
WORK="$(mktemp -d "$WORK_ROOT/stdlib-identity.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
# Exercise all stamped/unstamped controls through one writable cache. Sharing
# the source while changing the embedded marker must never reuse the old IR.
export ESHKOL_AOT_MODULE_CACHE_DIR="$WORK/module-cache"

mkdir -p "$WORK/source" "$WORK/matching" "$WORK/wrong-release" \
    "$WORK/wrong-source" "$WORK/wrong-abi" "$WORK/unstamped" \
    "$WORK/declaration-only" "$WORK/mixed-markers"
cat > "$WORK/source/stdlib.esk" <<'ESK'
(define (identity-probe) 17)
ESK
cat > "$WORK/source/main.esk" <<'ESK'
(require stdlib)
(display (identity-probe))
(newline)
ESK

build_control() {
    local root="$1" identity="$2"
    if [[ -n "$identity" ]]; then
        "$ESHKOL_RUN" --shared-lib -c \
            --embed-stdlib-identity="$identity" \
            -o "$root/stdlib" "$WORK/source/stdlib.esk" >/dev/null
    else
        "$ESHKOL_RUN" --shared-lib -c \
            -o "$root/stdlib" "$WORK/source/stdlib.esk" >/dev/null
    fi
}

OTHER_ABI=object_abi_v1
[[ "$OBJECT_ABI" == object_abi_v1 ]] && OTHER_ABI=object_abi_v2
ZERO_SOURCE="$(printf '%064d' 0)"
build_control "$WORK/matching" "$EXPECTED_IDENTITY"
build_control "$WORK/wrong-release" \
    "eshkol_stdlib_identity_release_0_0_0_abi_${OBJECT_ABI}_source_${SOURCE_FINGERPRINT}"
build_control "$WORK/wrong-source" \
    "eshkol_stdlib_identity_release_${RELEASE}_abi_${OBJECT_ABI}_source_${ZERO_SOURCE}"
build_control "$WORK/wrong-abi" \
    "eshkol_stdlib_identity_release_${RELEASE}_abi_${OTHER_ABI}_source_${SOURCE_FINGERPRINT}"
"$ESHKOL_RUN" --shared-lib -c \
    -o "$WORK/unstamped/stdlib" "$WORK/source/stdlib.esk" >/dev/null

compile_probe() {
    local root="$1" expected="$2" label="$3" explicit_object="${4:-}"
    local output_path="$WORK/$label.o" output status
    # A nonempty argv array also works with macOS Bash 3.2 and nounset.
    local -a compile_args=("$ESHKOL_RUN" -c -o "$output_path" "$WORK/source/main.esk")
    [[ -n "$explicit_object" ]] && compile_args+=("$explicit_object")
    if output="$(ESHKOL_LIB_DIR="$root" ESHKOL_PATH="$WORK/source" \
        "${compile_args[@]}" 2>&1)"; then
        status=0
    else
        status=$?
    fi
    if [[ "$expected" == match ]]; then
        [[ "$status" == 0 && -f "$output_path" ]]
        # INFO logging is optional; successful admission is verified by output.
        ! grep -Fq "rejecting precompiled stdlib object" <<<"$output"
    else
        [[ "$status" != 0 && ! -e "$output_path" ]]
        grep -Fq "rejecting precompiled stdlib object $root/stdlib.o" <<<"$output"
        ! grep -Fq "Auto-linking pre-compiled stdlib: $root/stdlib.o" <<<"$output"
    fi
}

compile_probe "$WORK/wrong-release" reject aot-explicit-object \
    "$WORK/wrong-release/stdlib.o"

compile_probe "$WORK/matching" match aot-matching
compile_probe "$WORK/wrong-release" reject aot-wrong-release
compile_probe "$WORK/wrong-source" reject aot-wrong-source
compile_probe "$WORK/wrong-abi" reject aot-wrong-abi
compile_probe "$WORK/unstamped" reject aot-unstamped

# Exercise an admitted real object through the full native link and execution.
matching_aot_output="$(ESHKOL_LIB_DIR="$WORK/matching" \
    ESHKOL_PATH="$WORK/source" "$ESHKOL_RUN" -o "$WORK/matching-executable" \
    "$WORK/source/main.esk" 2>&1)"
[[ "$("$WORK/matching-executable")" == 17 ]]

if aot_output="$(ESHKOL_LIB_DIR="$WORK/wrong-release" \
    ESHKOL_PATH="$WORK/source" "$ESHKOL_RUN" -o "$WORK/rejected-executable" \
    "$WORK/source/main.esk" 2>&1)"; then
    aot_status=0
else
    aot_status=$?
fi
[[ "$aot_status" != 0 && ! -e "$WORK/rejected-executable" ]]
grep -Fq "rejecting precompiled stdlib object $WORK/wrong-release/stdlib.o" \
    <<<"$aot_output"
! grep -Fxq "17" <<<"$aot_output"

run_probe() {
    local root="$1" expected="$2" reason="${3:-}" output status
    if output="$(ESHKOL_LIB_DIR="$root" ESHKOL_PATH="$WORK/source" \
        ESHKOL_JIT_CACHE=0 "$ESHKOL_RUN" -r "$WORK/source/main.esk" 2>&1)"; then
        status=0
    else
        status=$?
    fi
    if [[ "$expected" == match ]]; then
        [[ "$status" == 0 ]]
        { grep -Fq "[REPL] Loaded stdlib bitcode from: $root/stdlib.bc" <<<"$output" ||
          grep -Fq "[REPL] Loaded stdlib from cached object:" <<<"$output"; }
        grep -Fxq "17" <<<"$output"
    else
        [[ "$status" != 0 ]]
        { grep -Fq "rejecting stdlib.bc $root/stdlib.bc" <<<"$output" ||
          grep -Fq "rejecting precompiled stdlib bitcode $root/stdlib.bc" <<<"$output"; }
        ! grep -Fq "[REPL] Loaded stdlib bitcode from: $root/stdlib.bc" <<<"$output"
        ! grep -Fxq "17" <<<"$output"
        if [[ -n "$reason" ]]; then
            grep -Fq "$reason" <<<"$output"
        fi
    fi
}

run_probe "$WORK/matching" match
run_probe "$WORK/wrong-release" reject
run_probe "$WORK/wrong-source" reject
run_probe "$WORK/wrong-abi" reject
run_probe "$WORK/unstamped" reject

if eval_output="$(ESHKOL_LIB_DIR="$WORK/wrong-release" \
    ESHKOL_PATH="$WORK/source" "$ESHKOL_RUN" -e '(display 17)' 2>&1)"; then
    eval_status=0
else
    eval_status=$?
fi
[[ "$eval_status" != 0 ]]
grep -Fq "rejecting stdlib.bc $WORK/wrong-release/stdlib.bc" <<<"$eval_output"
! grep -Fxq "17" <<<"$eval_output"

# The persistent executable cache is checked only after stdlib admission; a
# stale cached run must not hide an incompatible currently selected package.
if cached_run_output="$(ESHKOL_LIB_DIR="$WORK/wrong-release" \
    ESHKOL_PATH="$WORK/source" ESHKOL_JIT_CACHE_DIR="$WORK/run-cache" \
    "$ESHKOL_RUN" -r "$WORK/source/main.esk" 2>&1)"; then
    cached_run_status=0
else
    cached_run_status=$?
fi
[[ "$cached_run_status" != 0 ]]
grep -Fq "rejecting precompiled stdlib bitcode $WORK/wrong-release/stdlib.bc" \
    <<<"$cached_run_output"
! grep -Fxq "17" <<<"$cached_run_output"

build_marker_controls() {
    local expected_ir="$WORK/identity-declaration.ll"
    local mixed_ir="$WORK/identity-mixed.ll"
    cat > "$expected_ir" <<IR
@$EXPECTED_IDENTITY = external constant i8
define ptr @identity_marker_reference() {
    ret ptr @$EXPECTED_IDENTITY
}
IR
    cat > "$mixed_ir" <<IR
@$EXPECTED_IDENTITY = constant i8 0
@eshkol_stdlib_identity_foreign_control = constant i8 0
IR
    "$LLVM_AS" "$expected_ir" -o "$WORK/declaration-only/stdlib.bc"
    "$LLC" -filetype=obj "$WORK/declaration-only/stdlib.bc" \
        -o "$WORK/declaration-only/stdlib.o"
    "$LLVM_AS" "$mixed_ir" -o "$WORK/mixed-markers/stdlib.bc"
    "$LLC" -filetype=obj "$WORK/mixed-markers/stdlib.bc" \
        -o "$WORK/mixed-markers/stdlib.o"

    compile_probe "$WORK/declaration-only" reject aot-declaration-only
    compile_probe "$WORK/mixed-markers" reject aot-mixed-markers
    run_probe "$WORK/declaration-only" reject "declaration-only identity marker"
    run_probe "$WORK/mixed-markers" reject "multiple embedded identity markers"
}

if [[ -x "$LLVM_AS" && -x "$LLC" ]]; then
    build_marker_controls
else
    echo "SKIP: LLVM assembler/codegen unavailable for declaration/mixed-marker controls"
fi

# Seed the JIT object cache from matching bitcode, then replace the discovered
# cache entry with a real object carrying a foreign marker. Admission must
# reject it before any user code runs.
if cache_output="$(ESHKOL_LIB_DIR="$WORK/matching" ESHKOL_PATH="$WORK/source" \
    ESHKOL_JIT_CACHE=0 "$ESHKOL_RUN" -r "$WORK/source/main.esk" 2>&1)"; then
    cache_status=0
else
    cache_status=$?
fi
[[ "$cache_status" == 0 ]]
cached_object="$(sed -n 's/.*Loaded stdlib from cached object: //p' \
    <<<"$cache_output" | head -n 1)"
[[ -n "$cached_object" && -f "$cached_object" ]]
cp "$WORK/wrong-release/stdlib.o" "$cached_object"
if cache_output="$(ESHKOL_LIB_DIR="$WORK/matching" ESHKOL_PATH="$WORK/source" \
    ESHKOL_JIT_CACHE=0 "$ESHKOL_RUN" -r "$WORK/source/main.esk" 2>&1)"; then
    cache_status=0
else
    cache_status=$?
fi
[[ "$cache_status" != 0 ]]
grep -Fq "rejecting cached stdlib object $cached_object" <<<"$cache_output"
! grep -Fxq "17" <<<"$cache_output"

echo "PASS: stdlib identity matching and available artifact/cache controls"
