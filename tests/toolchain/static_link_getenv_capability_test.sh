#!/usr/bin/env bash
# ESH-0077: exercise the downstream manual static-link path.
#
# The contract under test is:
#   eshkol-run --emit-object probe.esk -> probe.o
#   c++ probe.o stdlib.o libeshkol-static.a <configured host link args> -> exe
#
# This intentionally does not use eshkol-run's built-in final AOT link step.
set -euo pipefail

RUN="${1:-}"
SOURCE_ROOT="${2:-}"
BUILD_DIR="${3:-}"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

if [ -z "$RUN" ] || [ ! -x "$RUN" ]; then
    fail "eshkol-run is not executable: ${RUN:-<empty>}"
fi
if [ -z "$SOURCE_ROOT" ] || [ ! -d "$SOURCE_ROOT/lib" ]; then
    fail "source root does not contain lib/: ${SOURCE_ROOT:-<empty>}"
fi
if [ -z "$BUILD_DIR" ] || [ ! -d "$BUILD_DIR" ]; then
    fail "build dir does not exist: ${BUILD_DIR:-<empty>}"
fi

SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd)"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
CONFIG_HEADER="$BUILD_DIR/generated/eshkol/build_config.h"

if [ ! -f "$CONFIG_HEADER" ]; then
    fail "generated build config not found: $CONFIG_HEADER"
fi

config_value() {
    local name="$1"
    sed -n "s/^#define ${name} \"\\(.*\\)\"$/\\1/p" "$CONFIG_HEADER" | head -1
}

HOST_CXX="$(config_value ESHKOL_HOST_CXX_COMPILER)"
EXE_SUFFIX="$(config_value ESHKOL_HOST_EXECUTABLE_SUFFIX)"
STATIC_PREFIX="$(config_value ESHKOL_HOST_STATIC_LIBRARY_PREFIX)"
STATIC_SUFFIX="$(config_value ESHKOL_HOST_STATIC_LIBRARY_SUFFIX)"

if [ -z "$HOST_CXX" ]; then
    HOST_CXX="${CXX:-c++}"
fi
if [ -z "$STATIC_SUFFIX" ]; then
    STATIC_SUFFIX=".a"
fi

STDLIB_OBJ="$BUILD_DIR/stdlib.o"
STATIC_LIB="$BUILD_DIR/${STATIC_PREFIX}eshkol-static${STATIC_SUFFIX}"

[ -f "$STDLIB_OBJ" ] || fail "stdlib object is missing: $STDLIB_OBJ"
[ -f "$STATIC_LIB" ] || fail "static library is missing: $STATIC_LIB"

WORK_TMP="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-static-link.XXXXXX")"
cleanup() {
    [ -n "${WORK_TMP:-}" ] && [ -d "$WORK_TMP" ] && rm -rf -- "$WORK_TMP"
}
trap cleanup EXIT

cat > "$WORK_TMP/getenv_allowed.esk" <<'ESK'
(require stdlib)

(define env-policy-name "ESHKOL_STATIC_LINK_GETENV_PROBE")
(define got (getenv env-policy-name))

(if got
    (if (string=? got "static-link-ok")
        (begin
          (display "PASS: static link getenv reads set var") (newline)
          (exit 0))
        (begin
          (display "FAIL: static link getenv read wrong value") (newline)
          (exit 1)))
    (begin
      (display "FAIL: static link getenv returned false") (newline)
      (exit 1)))
ESK

cat > "$WORK_TMP/getenv_denied.esk" <<'ESK'
(require stdlib)
(require core.capabilities)

(define env-policy-name "ESHKOL_STATIC_LINK_GETENV_PROBE")
(capability-install-policy! '())
(define got (getenv env-policy-name))

(if got
    (begin
      (display "FAIL: static link capability policy allowed getenv") (newline)
      (exit 1))
    (begin
      (display "PASS: static link capability policy denies getenv") (newline)
      (exit 0)))
ESK

cat > "$WORK_TMP/getenv_missing_allowed.esk" <<'ESK'
(require stdlib)
(require core.capabilities)

(capability-install-policy! (list 'env-read))
(define got (getenv "ESHKOL_STATIC_LINK_GETENV_MISSING_PROBE"))

(if got
    (begin
      (display "FAIL: static link missing getenv returned value") (newline)
      (exit 1))
    (begin
      (display "PASS: static link missing getenv returns false") (newline)
      (exit 0)))
ESK

path_separator() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) printf ';' ;;
        *) printf ':' ;;
    esac
}

compile_probe() {
    local src="$1"
    local obj="$2"
    local log="$3"
    local sep
    sep="$(path_separator)"

    if ! env "ESHKOL_PATH=$SOURCE_ROOT/lib${ESHKOL_PATH:+$sep$ESHKOL_PATH}" \
        "$RUN" --emit-object -o "$obj" -I "$SOURCE_ROOT/lib" "$src" >"$log" 2>&1; then
        echo "--- compile log: $src ---" >&2
        sed -n '1,120p' "$log" >&2
        fail "--emit-object failed for $src"
    fi

    [ -f "$obj" ] || fail "--emit-object did not create expected object: $obj"
}

add_split_args() {
    local raw="${1:-}"
    local parts=()

    raw="${raw//;/ }"
    if [ -z "$raw" ]; then
        return
    fi

    # build_config.h carries whitespace-separated link args; this matches the
    # eshkol-run split contract and intentionally does not support spaced paths.
    # shellcheck disable=SC2206
    parts=( $raw )
    for arg in "${parts[@]}"; do
        [ -n "$arg" ] && LINK_ARGS+=("$arg")
    done
}

add_platform_args() {
    case "$(uname -s)" in
        Darwin)
            LINK_ARGS+=(
                "-framework" "Accelerate"
                "-framework" "Metal"
                "-framework" "MetalPerformanceShaders"
                "-framework" "Foundation"
                "-framework" "ImageIO"
                "-framework" "CoreGraphics"
                "-framework" "CoreFoundation"
            )
            sdk_path="$(xcrun --show-sdk-path 2>/dev/null || true)"
            if [ -n "${sdk_path:-}" ] && [ -d "$sdk_path/usr/lib" ]; then
                LINK_ARGS+=("-L$sdk_path/usr/lib")
            fi
            LINK_ARGS+=("-lobjc" "-Wl,-stack_size,0x20000000")
            ;;
        Linux)
            LINK_ARGS+=("-Wl,-z,stack-size=536870912" "-Wl,--export-dynamic")
            case "$(uname -m)" in
                aarch64|arm64) LINK_ARGS+=("-fuse-ld=lld") ;;
            esac
            LINK_ARGS+=("-lm" "-ldl")
            ;;
        MINGW*|MSYS*|CYGWIN*)
            LINK_ARGS+=("-fuse-ld=lld" "-Wl,--stack,536870912")
            ;;
    esac
}

link_probe() {
    local obj="$1"
    local exe="$2"
    local log="$3"

    LINK_ARGS=("$HOST_CXX")
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) ;;
        *) LINK_ARGS+=("-fPIE") ;;
    esac

    LINK_ARGS+=("$obj" "$STDLIB_OBJ" "$STATIC_LIB")
    add_split_args "$(config_value ESHKOL_HOST_RUNTIME_LINK_ARGS)"
    add_split_args "$(config_value ESHKOL_HOST_LLVM_LINK_ARGS)"
    add_platform_args
    LINK_ARGS+=("-o" "$exe")

    if ! "${LINK_ARGS[@]}" >"$log" 2>&1; then
        echo "--- link command ---" >&2
        printf '%q ' "${LINK_ARGS[@]}" >&2
        echo >&2
        echo "--- link log ---" >&2
        sed -n '1,160p' "$log" >&2
        fail "manual static link failed for $obj"
    fi

    [ -x "$exe" ] || fail "manual link did not create executable: $exe"
}

run_probe() {
    local exe="$1"
    local pass_line="$2"
    local log="$3"

    if ! ESHKOL_STATIC_LINK_GETENV_PROBE=static-link-ok "$exe" >"$log" 2>&1; then
        echo "--- run log ---" >&2
        sed -n '1,160p' "$log" >&2
        fail "manual static-link executable failed: $exe"
    fi
    if ! grep -q "$pass_line" "$log"; then
        echo "--- run log ---" >&2
        sed -n '1,160p' "$log" >&2
        fail "missing expected output: $pass_line"
    fi
    if grep -q '^FAIL:' "$log"; then
        echo "--- run log ---" >&2
        sed -n '1,160p' "$log" >&2
        fail "probe reported failure"
    fi
}

ALLOWED_OBJ="$WORK_TMP/getenv_allowed.o"
DENIED_OBJ="$WORK_TMP/getenv_denied.o"
MISSING_OBJ="$WORK_TMP/getenv_missing_allowed.o"
ALLOWED_EXE="$WORK_TMP/getenv_allowed${EXE_SUFFIX}"
DENIED_EXE="$WORK_TMP/getenv_denied${EXE_SUFFIX}"
MISSING_EXE="$WORK_TMP/getenv_missing_allowed${EXE_SUFFIX}"

compile_probe "$WORK_TMP/getenv_allowed.esk" "$ALLOWED_OBJ" "$WORK_TMP/compile_allowed.log"
compile_probe "$WORK_TMP/getenv_denied.esk" "$DENIED_OBJ" "$WORK_TMP/compile_denied.log"
compile_probe "$WORK_TMP/getenv_missing_allowed.esk" "$MISSING_OBJ" "$WORK_TMP/compile_missing.log"

link_probe "$ALLOWED_OBJ" "$ALLOWED_EXE" "$WORK_TMP/link_allowed.log"
link_probe "$DENIED_OBJ" "$DENIED_EXE" "$WORK_TMP/link_denied.log"
link_probe "$MISSING_OBJ" "$MISSING_EXE" "$WORK_TMP/link_missing.log"

run_probe "$ALLOWED_EXE" "PASS: static link getenv reads set var" "$WORK_TMP/run_allowed.log"
run_probe "$DENIED_EXE" "PASS: static link capability policy denies getenv" "$WORK_TMP/run_denied.log"
run_probe "$MISSING_EXE" "PASS: static link missing getenv returns false" "$WORK_TMP/run_missing.log"

if ! grep -q "capability denied: env-read" "$WORK_TMP/run_denied.log"; then
    echo "--- denied run log ---" >&2
    sed -n '1,160p' "$WORK_TMP/run_denied.log" >&2
    fail "denied getenv did not emit env-read diagnostic"
fi
if grep -q "capability denied:" "$WORK_TMP/run_missing.log"; then
    echo "--- missing run log ---" >&2
    sed -n '1,160p' "$WORK_TMP/run_missing.log" >&2
    fail "allowed missing getenv emitted a denial diagnostic"
fi

cat "$WORK_TMP/run_allowed.log"
cat "$WORK_TMP/run_denied.log"
cat "$WORK_TMP/run_missing.log"
echo "PASS: static_link_getenv_capability_test"
