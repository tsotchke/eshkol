#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: sanitizer_build_dir_contract_test.sh <source-root>" >&2
    exit 2
fi

SOURCE_ROOT="$(cd "$1" && pwd)"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-sanitizer-build-dir.XXXXXX")"
trap 'rm -rf "$TMP_ROOT"' EXIT

FAKE_BIN="$TMP_ROOT/bin"
CUSTOM_BUILD="$TMP_ROOT/custom/nested/sanitizer-build"
CMAKE_LOG="$TMP_ROOT/cmake.log"
mkdir -p "$FAKE_BIN" "$TMP_ROOT/unrelated-cwd"

cat > "$FAKE_BIN/cmake" <<'FAKE_CMAKE'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$ESHKOL_TEST_CMAKE_LOG"
FAKE_CMAKE
chmod +x "$FAKE_BIN/cmake"

(
    cd "$TMP_ROOT/unrelated-cwd"
    PATH="$FAKE_BIN:$PATH" \
        ESHKOL_TEST_CMAKE_LOG="$CMAKE_LOG" \
        ESHKOL_BUILD_JOBS=3 \
        BUILD_DIR="$CUSTOM_BUILD" \
        CMAKE_BUILD_TYPE=RelWithDebInfo \
        "$SOURCE_ROOT/scripts/build-sanitizer.sh" asan+ubsan >/dev/null
)

call_count="$(wc -l < "$CMAKE_LOG" | tr -d ' ')"
if [ "$call_count" -ne 2 ]; then
    echo "expected exactly two cmake calls, got $call_count" >&2
    cat "$CMAKE_LOG" >&2
    exit 1
fi

configure="$(sed -n '1p' "$CMAKE_LOG")"
build="$(sed -n '2p' "$CMAKE_LOG")"
CANONICAL_BUILD="$(cd "$CUSTOM_BUILD" && pwd)"

case "$configure" in
    *"-S $SOURCE_ROOT -B $CANONICAL_BUILD"*) ;;
    *) echo "configure did not preserve source/build roots: $configure" >&2; exit 1 ;;
esac
case "$configure" in
    *"-DESHKOL_ENABLE_ASAN=ON"*"-DESHKOL_ENABLE_UBSAN=ON"*) ;;
    *) echo "configure lost sanitizer selection: $configure" >&2; exit 1 ;;
esac
case "$build" in
    *"--build $CANONICAL_BUILD --target eshkol-run stdlib --parallel 3"*) ;;
    *) echo "build did not use the requested directory/targets: $build" >&2; exit 1 ;;
esac

test -d "$CUSTOM_BUILD"

if PATH="$FAKE_BIN:$PATH" \
    ESHKOL_TEST_CMAKE_LOG="$CMAKE_LOG" \
    ESHKOL_BUILD_JOBS=0 \
    BUILD_DIR="$TMP_ROOT/invalid-jobs" \
    "$SOURCE_ROOT/scripts/build-sanitizer.sh" asan >/dev/null 2>"$TMP_ROOT/invalid.err"; then
    echo "invalid ESHKOL_BUILD_JOBS unexpectedly succeeded" >&2
    exit 1
fi
grep -F "ESHKOL_BUILD_JOBS must be a positive integer: 0" "$TMP_ROOT/invalid.err" >/dev/null
test ! -e "$TMP_ROOT/invalid-jobs"

if PATH="$FAKE_BIN:$PATH" \
    ESHKOL_TEST_CMAKE_LOG="$CMAKE_LOG" \
    ESHKOL_BUILD_JOBS=1 \
    BUILD_DIR="$SOURCE_ROOT" \
    "$SOURCE_ROOT/scripts/build-sanitizer.sh" asan >/dev/null 2>"$TMP_ROOT/in-source.err"; then
    echo "in-source sanitizer build unexpectedly succeeded" >&2
    exit 1
fi
grep -F "refusing in-source sanitizer build: $SOURCE_ROOT" "$TMP_ROOT/in-source.err" >/dev/null

if [ "$(wc -l < "$CMAKE_LOG" | tr -d ' ')" -ne "$call_count" ]; then
    echo "invalid contract inputs reached cmake" >&2
    cat "$CMAKE_LOG" >&2
    exit 1
fi

echo "sanitizer custom build-directory contract: PASS"
