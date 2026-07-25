#!/usr/bin/env bash
# Runtime-archive / install-artifact resolution precedence (RC B3).
#
# The defect this pins down: find_runtime_library() searched name-major — every
# location for `libeshkol-runtime.a`, INCLUDING the system prefixes, before
# `libeshkol-static.a` was tried anywhere, including right next to the running
# compiler. An install that ships only the legacy aggregate name (the Homebrew
# keg) therefore linked a stale /usr/local/lib archive from a different Eshkol
# in preference to its own, and $ESHKOL_LIB_DIR could not override it because
# the env directory was consulted once per NAME rather than ahead of every
# location. A stale archive whose symbol set still matches links silently, so
# this is a wrong-runtime class defect, not merely a build failure.
#
# The contract asserted here, in precedence order:
#   1. $ESHKOL_LIB_DIR wins absolutely.
#   2. Otherwise the archive co-located with the compiler's REAL path wins —
#      including when the compiler was launched through a symlink, and
#      including when the co-located archive only has the legacy name.
#   3. Only then does a system location get used.
#   4. A system-resolved artifact is announced on stderr, and a version
#      disagreement is announced loudly, whatever tier it came from.
#
# Every case is hermetic: $ESHKOL_SYSTEM_PREFIXES replaces the built-in system
# prefixes with a directory this test owns, so a real /usr/local install on the
# host can neither rescue nor break it.
set -uo pipefail

RUN="${1:-}"
BUILD_DIR="${2:-}"

STATUS=0
CASE=""

fail() {
    echo "FAIL: ${CASE:+$CASE: }$*" >&2
    STATUS=1
}

note() { echo "  - $*"; }

if [ -z "$RUN" ] || [ ! -x "$RUN" ]; then
    echo "FAIL: eshkol-run is not executable: ${RUN:-<empty>}" >&2
    exit 1
fi
if [ -z "$BUILD_DIR" ] || [ ! -d "$BUILD_DIR" ]; then
    echo "FAIL: build dir does not exist: ${BUILD_DIR:-<empty>}" >&2
    exit 1
fi
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"

REAL_RUNTIME_ARCHIVE="$BUILD_DIR/libeshkol-runtime.a"
REAL_STDLIB_OBJECT="$BUILD_DIR/stdlib.o"
[ -f "$REAL_RUNTIME_ARCHIVE" ] || { echo "FAIL: missing $REAL_RUNTIME_ARCHIVE" >&2; exit 1; }
[ -f "$REAL_STDLIB_OBJECT" ] || { echo "FAIL: missing $REAL_STDLIB_OBJECT" >&2; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-archive-resolution.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
# The driver reports canonical paths (macOS $TMPDIR is itself a symlink), so
# compare against the canonical form.
WORK="$(cd "$WORK" && pwd -P)"

MARKER="ARCHIVE-RESOLUTION-OK"
printf '(display "%s")\n(newline)\n' "$MARKER" > "$WORK/hello.esk"

# A file that is not an archive at all: if resolution picks it, the link fails
# loudly instead of producing a plausible-looking binary.
make_bogus_archive() {
    local path="$1"
    local stamped_version="${2:-}"
    mkdir -p "$(dirname "$path")"
    printf 'not a real archive\n' > "$path"
    if [ -n "$stamped_version" ]; then
        # Same stamp layout lib/core/platform_runtime.cpp embeds, so the driver
        # reads this file's "version" exactly as it would a real archive's.
        printf 'ESHKOL-RUNTIME-BUILD-STAMP:%s;\n' "$stamped_version" >> "$path"
    fi
}

# An install prefix: bin/eshkol-run plus, optionally, its own lib/eshkol.
make_install() {
    local root="$1"      # prefix to create
    local with_lib="$2"  # "colocated" to install the runtime beside it
    mkdir -p "$root/bin"
    cp "$RUN" "$root/bin/eshkol-run"
    if [ "$with_lib" = "colocated" ]; then
        mkdir -p "$root/lib/eshkol"
        # Deliberately the LEGACY aggregate name: the keg layout that lost to a
        # system libeshkol-runtime.a before this fix.
        cp "$REAL_RUNTIME_ARCHIVE" "$root/lib/eshkol/libeshkol-static.a"
        cp "$REAL_STDLIB_OBJECT" "$root/lib/eshkol/stdlib.o"
    fi
}

# Run a compile+execute in a controlled cwd. Writes stderr to $WORK/stderr.txt
# and the program's stdout to $WORK/stdout.txt.
# Any further arguments are VAR=VALUE settings for this compile only — passed
# through `env` so no case can leak a resolution variable into the next one,
# whatever the shell's rules for assignments preceding a function call are.
compile_and_run() {
    local driver="$1" cwd="$2"
    shift 2
    ( cd "$cwd" && env "$@" "$driver" hello.esk -o program ) \
        > "$WORK/compile.txt" 2> "$WORK/stderr.txt"
    local compile_status=$?
    : > "$WORK/stdout.txt"
    if [ "$compile_status" -eq 0 ] && [ -x "$cwd/program" ]; then
        ( cd "$cwd" && ./program ) > "$WORK/stdout.txt" 2>&1
    fi
    return "$compile_status"
}

expect_marker() {
    grep -q "$MARKER" "$WORK/stdout.txt" ||
        fail "the linked program did not print $MARKER (stdout: $(tr '\n' ' ' < "$WORK/stdout.txt" | head -c 200))"
}

expect_stderr() {
    grep -q "$1" "$WORK/stderr.txt" ||
        fail "expected stderr to mention '$1'; got: $(tr '\n' ' ' < "$WORK/stderr.txt" | head -c 400)"
}

refute_stderr() {
    if grep -q "$1" "$WORK/stderr.txt"; then
        fail "stderr should not mention '$1'; got: $(tr '\n' ' ' < "$WORK/stderr.txt" | head -c 400)"
    fi
}

# Nothing inherited from the caller may influence resolution.
unset ESHKOL_LIB_DIR ESHKOL_PATH ESHKOL_SYSTEM_PREFIXES

# ---------------------------------------------------------------------------
# 1. The compiler's own archive beats one found through a lower-precedence
#    root, even when the compiler's copy only has the legacy name.
# ---------------------------------------------------------------------------
CASE="colocated_archive_wins"
make_install "$WORK/case1/prefix" colocated
mkdir -p "$WORK/case1/cwd"
cp "$WORK/hello.esk" "$WORK/case1/cwd/"
# A stale archive reachable from the working directory AND from a system
# prefix. Neither may outrank the install the compiler belongs to.
make_bogus_archive "$WORK/case1/cwd/libeshkol-runtime.a"
make_bogus_archive "$WORK/case1/sysprefix/lib/libeshkol-runtime.a"
compile_and_run "$WORK/case1/prefix/bin/eshkol-run" "$WORK/case1/cwd" \
    "ESHKOL_SYSTEM_PREFIXES=$WORK/case1/sysprefix" ||
    fail "compiling against the co-located archive failed (stderr: $(tr '\n' ' ' < "$WORK/stderr.txt" | head -c 400))"
expect_marker
refute_stderr "system location"
refute_stderr "was built from Eshkol"
note "case 1: co-located libeshkol-static.a used; cwd/system libeshkol-runtime.a ignored"

# ---------------------------------------------------------------------------
# 2. $ESHKOL_LIB_DIR is the escape hatch and must win over everything.
# ---------------------------------------------------------------------------
CASE="env_lib_dir_wins"
make_install "$WORK/case2/prefix" bare
mkdir -p "$WORK/case2/envdir" "$WORK/case2/cwd"
cp "$REAL_RUNTIME_ARCHIVE" "$WORK/case2/envdir/libeshkol-static.a"
cp "$REAL_STDLIB_OBJECT" "$WORK/case2/envdir/stdlib.o"
cp "$WORK/hello.esk" "$WORK/case2/cwd/"
make_bogus_archive "$WORK/case2/cwd/libeshkol-runtime.a"
make_bogus_archive "$WORK/case2/sysprefix/lib/libeshkol-runtime.a"
compile_and_run "$WORK/case2/prefix/bin/eshkol-run" "$WORK/case2/cwd" \
    "ESHKOL_LIB_DIR=$WORK/case2/envdir" \
    "ESHKOL_SYSTEM_PREFIXES=$WORK/case2/sysprefix" ||
    fail "ESHKOL_LIB_DIR did not take effect (stderr: $(tr '\n' ' ' < "$WORK/stderr.txt" | head -c 400))"
expect_marker
refute_stderr "system location"
note "case 2: \$ESHKOL_LIB_DIR/libeshkol-static.a used ahead of every other location"

# ---------------------------------------------------------------------------
# 3. Launching through a symlink still resolves the real install (the Homebrew
#    <prefix>/bin/eshkol-run -> ../Cellar/... layout).
# ---------------------------------------------------------------------------
CASE="symlinked_launch_path"
make_install "$WORK/case3/prefix" colocated
mkdir -p "$WORK/case3/linkdir" "$WORK/case3/cwd"
ln -s "$WORK/case3/prefix/bin/eshkol-run" "$WORK/case3/linkdir/eshkol-run"
cp "$WORK/hello.esk" "$WORK/case3/cwd/"
make_bogus_archive "$WORK/case3/cwd/libeshkol-runtime.a"
make_bogus_archive "$WORK/case3/sysprefix/lib/libeshkol-runtime.a"
compile_and_run "$WORK/case3/linkdir/eshkol-run" "$WORK/case3/cwd" \
    "ESHKOL_SYSTEM_PREFIXES=$WORK/case3/sysprefix" ||
    fail "resolution through a symlinked launch path failed (stderr: $(tr '\n' ' ' < "$WORK/stderr.txt" | head -c 400))"
expect_marker
refute_stderr "system location"
note "case 3: real executable path resolved; archive found in the install behind the symlink"

# ---------------------------------------------------------------------------
# 4. With nothing co-located, a system archive is still used — and said out
#    loud rather than resolved silently.
# ---------------------------------------------------------------------------
CASE="system_fallback_is_announced"
make_install "$WORK/case4/prefix" bare
mkdir -p "$WORK/case4/sysprefix/lib" "$WORK/case4/cwd"
cp "$REAL_RUNTIME_ARCHIVE" "$WORK/case4/sysprefix/lib/libeshkol-runtime.a"
cp "$REAL_STDLIB_OBJECT" "$WORK/case4/sysprefix/lib/stdlib.o"
cp "$WORK/hello.esk" "$WORK/case4/cwd/"
compile_and_run "$WORK/case4/prefix/bin/eshkol-run" "$WORK/case4/cwd" \
    "ESHKOL_SYSTEM_PREFIXES=$WORK/case4/sysprefix" ||
    fail "the system-location fallback stopped working (stderr: $(tr '\n' ' ' < "$WORK/stderr.txt" | head -c 400))"
expect_marker
expect_stderr "runtime archive from a system location"
expect_stderr "$WORK/case4/sysprefix/lib/libeshkol-runtime.a"
refute_stderr "was built from Eshkol"
note "case 4: system fallback works and is reported; versions agree so it is only a note"

# ---------------------------------------------------------------------------
# 5. A system archive built from a different Eshkol is reported loudly. This is
#    the silent-miscompile case: same symbols, different runtime layout.
# ---------------------------------------------------------------------------
CASE="version_skew_is_loud"
make_install "$WORK/case5/prefix" bare
mkdir -p "$WORK/case5/cwd"
cp "$WORK/hello.esk" "$WORK/case5/cwd/"
make_bogus_archive "$WORK/case5/sysprefix/lib/libeshkol-runtime.a" "0.0.1-resolution-test"
# The precompiled stdlib object has to be reachable or compilation stops before
# the link, and the archive would never be examined.
cp "$REAL_STDLIB_OBJECT" "$WORK/case5/sysprefix/lib/stdlib.o"
compile_and_run "$WORK/case5/prefix/bin/eshkol-run" "$WORK/case5/cwd" \
    "ESHKOL_SYSTEM_PREFIXES=$WORK/case5/sysprefix"
expect_stderr "was built from Eshkol 0.0.1-resolution-test"
expect_stderr "WARNING"
if [ -x "$WORK/case5/cwd/program" ]; then
    fail "a program was produced from a version-mismatched archive"
fi
note "case 5: version skew warned about by name, and the bad archive did not yield a binary"

# ---------------------------------------------------------------------------
# 6. The skew warning is not limited to system locations: an archive named by
#    $ESHKOL_LIB_DIR is checked too (and, being tier 1, is what gets picked
#    even though a good archive sits beside the compiler).
# ---------------------------------------------------------------------------
CASE="env_lib_dir_skew_is_loud"
make_install "$WORK/case6/prefix" colocated
mkdir -p "$WORK/case6/cwd" "$WORK/case6/sysprefix/lib"
cp "$WORK/hello.esk" "$WORK/case6/cwd/"
make_bogus_archive "$WORK/case6/envdir/libeshkol-runtime.a" "0.0.2-resolution-test"
compile_and_run "$WORK/case6/prefix/bin/eshkol-run" "$WORK/case6/cwd" \
    "ESHKOL_LIB_DIR=$WORK/case6/envdir" \
    "ESHKOL_SYSTEM_PREFIXES=$WORK/case6/sysprefix"
expect_stderr "was built from Eshkol 0.0.2-resolution-test"
note "case 6: \$ESHKOL_LIB_DIR is authoritative, and a mismatched archive there is still called out"

if [ "$STATUS" -ne 0 ]; then
    exit "$STATUS"
fi
echo "PASS: runtime_archive_resolution_test"
