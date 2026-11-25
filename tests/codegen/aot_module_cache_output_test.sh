#!/usr/bin/env bash
# Compile-only AOT cache must reproduce both the object and its companion
# bitcode at the same paths on a cold miss and a warm hit.
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
export ESHKOL_TEST_TMP_ROOT="${ESHKOL_TEST_TMP_ROOT:-$REPO_ROOT/.scratch}"
source "$REPO_ROOT/scripts/lib/test_isolation.sh" || {
    echo "FAIL: cannot source test isolation helper" >&2
    exit 1
}
eshkol_test_isolation_init "aot-module-cache-output" || exit 1

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in /*) B="$BUILD_DIR" ;; *) B="$REPO_ROOT/$BUILD_DIR" ;; esac
ESHKOL_RUN="${ESHKOL_RUN:-$B/eshkol-run}"
[ -x "$ESHKOL_RUN" ] || { echo "FAIL: $ESHKOL_RUN not found" >&2; exit 2; }
WORK="$ESHKOL_TEST_TMPDIR"
eshkol_test_toolchain_snapshot "$B" || { echo "FAIL: toolchain snapshot failed" >&2; exit 1; }

cat > "$WORK/explicit_source.esk" <<'ESK'
(define aot-cache-value (+ 20 22))
ESK
mkdir -p "$WORK/default"
cat > "$WORK/default/default_source.esk" <<'ESK'
(define aot-cache-value (+ 20 22))
ESK

run_compile() {
    local label=$1 cache_dir=$2 source=$3 cwd=$4 output=${5:-}
    if [ -n "$output" ]; then
        (cd "$cwd" && ESHKOL_AOT_MODULE_CACHE_DIR="$cache_dir" \
            ESHKOL_AOT_MODULE_CACHE_TRACE=1 eshkol_test_timeout 300 \
            "$ESHKOL_RUN" -c "$source" -o "$output" -L"$B") \
            > "$WORK/$label.log" 2>&1 || {
                echo "FAIL: $label compile failed" >&2
                sed 's/^/  /' "$WORK/$label.log" >&2
                return 1
            }
    else
        (cd "$cwd" && ESHKOL_AOT_MODULE_CACHE_DIR="$cache_dir" \
            ESHKOL_AOT_MODULE_CACHE_TRACE=1 eshkol_test_timeout 300 \
            "$ESHKOL_RUN" -c "$source" -L"$B") \
            > "$WORK/$label.log" 2>&1 || {
                echo "FAIL: $label compile failed" >&2
                sed 's/^/  /' "$WORK/$label.log" >&2
                return 1
            }
    fi
}

assert_trace() {
    local label=$1 expected=$2 count status
    count=$(grep -c '^ESH-0089: AOT module cache ' "$WORK/$label.log" || true)
    status=$(grep '^ESH-0089: AOT module cache ' "$WORK/$label.log" | \
        sed -E 's/^ESH-0089: AOT module cache ([^ ]+).*/\1/')
    if [ "$count" -ne 1 ] || [ "$status" != "$expected" ]; then
        echo "FAIL: $label expected one AOT cache $expected trace (got count=$count status='$status')" >&2
        sed 's/^/  /' "$WORK/$label.log" >&2
        return 1
    fi
}

check_outputs() {
    local label=$1 object=$2 bitcode=$3
    if [ ! -s "$object" ] || [ ! -s "$bitcode" ]; then
        echo "FAIL: $label did not produce nonempty object and bitcode at expected paths" >&2
        echo "  object: $object ($(wc -c < "$object" 2>/dev/null || echo missing) bytes)" >&2
        echo "  bitcode: $bitcode ($(wc -c < "$bitcode" 2>/dev/null || echo missing) bytes)" >&2
        return 1
    fi
}

explicit_output="$WORK/explicit-output"
explicit_object="$explicit_output.o"
explicit_bitcode="$explicit_output.bc"
run_compile explicit-cold "$WORK/explicit-cache" "$WORK/explicit_source.esk" "$WORK" "$explicit_output" || exit 1
assert_trace explicit-cold miss || exit 1
check_outputs explicit-cold "$explicit_object" "$explicit_bitcode" || exit 1
rm -f "$explicit_object" "$explicit_bitcode" "$explicit_object.bc"
run_compile explicit-warm "$WORK/explicit-cache" "$WORK/explicit_source.esk" "$WORK" "$explicit_output" || exit 1
assert_trace explicit-warm hit || exit 1
check_outputs explicit-warm "$explicit_object" "$explicit_bitcode" || exit 1
[ ! -e "$explicit_object.bc" ] || {
    echo "FAIL: warm explicit-output cache wrote the wrong bitcode path: $explicit_object.bc" >&2
    exit 1
}

default_object="$WORK/default/default_source.o"
default_bitcode="$WORK/default/default_source.bc"
run_compile default-cold "$WORK/default-cache" "$WORK/default/default_source.esk" "$WORK/default" || exit 1
assert_trace default-cold miss || exit 1
check_outputs default-cold "$default_object" "$default_bitcode" || exit 1
rm -f "$default_object" "$default_bitcode" "$default_object.bc"
run_compile default-warm "$WORK/default-cache" "$WORK/default/default_source.esk" "$WORK/default" || exit 1
assert_trace default-warm hit || exit 1
check_outputs default-warm "$default_object" "$default_bitcode" || exit 1
[ ! -e "$default_object.bc" ] || {
    echo "FAIL: warm default-output cache wrote the wrong bitcode path: $default_object.bc" >&2
    exit 1
}

# Shared-library and executable compilation of the same source have distinct
# codegen modes. A cache entry from either mode must not contaminate the other.
cat > "$WORK/mode_source.esk" <<'ESK'
(define aot-mode-value (+ 20 22))
ESK
find_nm() {
    local candidate
    for candidate in "${LLVM_NM:-}" llvm-nm nm; do
        [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1 && {
            command -v "$candidate"
            return 0
        }
    done
    return 1
}
NM_TOOL=$(find_nm) || { echo "FAIL: nm tool unavailable for mode cache regression" >&2; exit 1; }
has_main_symbol() {
    awk '($NF == "main" || $NF == "_main") && $(NF-1) != "U" { found=1 } END { exit !found }' <<< "$1"
}
mode_compile() {
    local label=$1 mode=$2 expected_main=$3 expected_trace=$4 out="$WORK/$1.o"
    local symbols
    local -a args=("$ESHKOL_RUN" -c "$WORK/mode_source.esk" -o "$out" -L"$B")
    [ "$mode" = library ] && args+=(--shared-lib)
    (cd "$WORK" && ESHKOL_AOT_MODULE_CACHE_DIR="$WORK/mode-cache" \
        ESHKOL_AOT_MODULE_CACHE_TRACE=1 eshkol_test_timeout 300 "${args[@]}") \
        > "$WORK/$label.log" 2>&1 || { sed 's/^/  /' "$WORK/$label.log" >&2; exit 1; }
    assert_trace "$label" "$expected_trace" || exit 1
    symbols=$("$NM_TOOL" "$out") || { echo "FAIL: $label symbol inspection failed" >&2; exit 1; }
    if [ "$expected_main" = yes ]; then
        has_main_symbol "$symbols" || { echo "FAIL: $label object lacks main symbol" >&2; exit 1; }
    else
        ! has_main_symbol "$symbols" || { echo "FAIL: $label library object contains main symbol" >&2; exit 1; }
    fi
}
mode_compile library-first-library library no miss
mode_compile library-first-library-warm library no hit
mode_compile library-first-executable executable yes miss
rm -rf "$WORK/mode-cache"
mode_compile executable-first-executable executable yes miss
mode_compile executable-first-executable-warm executable yes hit
mode_compile executable-first-library library no miss

if ! eshkol_test_toolchain_verify "$B"; then
    echo "FAIL: AOT cache toolchain changed during run" >&2
    exit 1
fi
echo "PASS: aot_module_cache_output_test: explicit miss/hit and default miss/hit emit matching .o/.bc paths"
