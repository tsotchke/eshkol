#!/usr/bin/env bash
# A sibling build directory must fingerprint the source configured by CMake,
# not a different Git repository that happens to contain both directories.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-build-fingerprint-root.XXXXXX")"
trap 'rm -rf -- "$TEMP_ROOT"' EXIT

commit_file() {
    local repo="$1" name="$2" contents="$3"
    printf '%s\n' "$contents" > "$repo/$name"
    git -C "$repo" add "$name"
    git -C "$repo" -c user.name='Fingerprint Test' \
        -c user.email='fingerprint-test@example.invalid' \
        commit -qm "add $name"
}

OUTER="$TEMP_ROOT/outer"
SOURCE="$OUTER/nested-source"
BUILD="$OUTER/sibling-build"
mkdir -p "$OUTER" "$SOURCE" "$BUILD"
git -C "$OUTER" init -q
commit_file "$OUTER" outer.txt outer-head

git -C "$SOURCE" init -q
commit_file "$SOURCE" source.txt configured-source-head
OUTER_HEAD="$(git -C "$OUTER" rev-parse HEAD)"
SOURCE_HEAD="$(git -C "$SOURCE" rev-parse HEAD)"
printf 'CMAKE_HOME_DIRECTORY:INTERNAL=%s\n' "$SOURCE" > "$BUILD/CMakeCache.txt"
printf 'binary fixture\n' > "$BUILD/eshkol-run"

# The parent directory is an unrelated committed checkout; CMake's configured
# source root must win when the shared fingerprint producer records git_sha.
source "$ROOT/scripts/lib/build_fingerprint.sh"
TRACE="$TEMP_ROOT/trace"
eshkol_emit_build_fingerprint_event "$TRACE" source-root-regression "$BUILD" eshkol-run >/dev/null 2>&1
RECORDED="$(python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).readline())["git_sha"])' "$TRACE/build_fingerprint.jsonl")"
if [ "$RECORDED" != "$SOURCE_HEAD" ] || [ "$RECORDED" = "$OUTER_HEAD" ]; then
    echo "FAIL: expected configured source HEAD $SOURCE_HEAD, got $RECORDED (outer HEAD $OUTER_HEAD)"
    exit 1
fi

# A configured but unversioned source tree must not inherit OUTER_HEAD either.
UNVERSIONED_SOURCE="$OUTER/unversioned-source"
mkdir -p "$UNVERSIONED_SOURCE"
printf 'CMAKE_HOME_DIRECTORY:INTERNAL=%s\n' "$UNVERSIONED_SOURCE" > "$BUILD/CMakeCache.txt"
UNVERSIONED_TRACE="$TEMP_ROOT/unversioned-trace"
eshkol_emit_build_fingerprint_event "$UNVERSIONED_TRACE" source-root-regression "$BUILD" eshkol-run >/dev/null 2>&1
UNVERSIONED_RECORDED="$(python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).readline())["git_sha"])' "$UNVERSIONED_TRACE/build_fingerprint.jsonl")"
if [ "$UNVERSIONED_RECORDED" != "unknown" ]; then
    echo "FAIL: unversioned configured source inherited enclosing HEAD $UNVERSIONED_RECORDED"
    exit 1
fi

# With no CMake source root and no checkout at the build directory itself,
# fail closed instead of walking upward and recording OUTER_HEAD.
NO_CACHE_BUILD="$OUTER/no-cache-build"
mkdir -p "$NO_CACHE_BUILD"
printf 'binary fixture\n' > "$NO_CACHE_BUILD/eshkol-run"
NO_CACHE_TRACE="$TEMP_ROOT/no-cache-trace"
eshkol_emit_build_fingerprint_event "$NO_CACHE_TRACE" source-root-regression "$NO_CACHE_BUILD" eshkol-run >/dev/null 2>&1
NO_CACHE_RECORDED="$(python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).readline())["git_sha"])' "$NO_CACHE_TRACE/build_fingerprint.jsonl")"
if [ "$NO_CACHE_RECORDED" != "unknown" ]; then
    echo "FAIL: missing source configuration fell through to enclosing HEAD $NO_CACHE_RECORDED"
    exit 1
fi

echo "PASS: build fingerprint uses CMAKE_HOME_DIRECTORY and never falls through to an enclosing repository"
