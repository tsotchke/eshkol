#!/bin/bash
# test_isolation.sh — per-run, per-repo-root isolation for the shell test suites.
#
# WHY THIS EXISTS
#
# The suites under scripts/run_*_tests.sh used to write their scratch files to
# hardcoded machine-global paths (`/tmp/test_output.txt`, `/tmp/<suite>_compile.log`,
# `-o /tmp/<suite>_test_bin`) and to compile their test programs to a bare
# repo-root `./a.out`.  Every one of those names is shared by every checkout on
# the machine, so two suites running at the same time — two git worktrees, two
# agents, CI plus a local run, or even the same suite invoked twice — silently
# clobbered each other:
#
#   * Suite A's `./a.out` was overwritten by suite B between A's compile and
#     A's run, so A executed *B's* program and reported the result against A's
#     test name.
#   * `rm -f a.out` in one suite deleted the binary another suite was about to
#     run, producing spurious "COMPILE FAIL"/"RUNTIME FAIL" verdicts.
#   * `/tmp/test_output.txt` was read back by whichever suite got there last,
#     so a PASS could be certified from a *different run's* output.
#
# Failures in the wrong direction (false PASS) are the dangerous ones: a
# harness that a concurrent run can corrupt cannot certify a release.
#
# THE CONTRACT
#
# Source this file and call `eshkol_test_isolation_init <label>` once, early:
#
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test_isolation.sh"
#     eshkol_test_isolation_init "autodiff"
#
# You then get, all unique to this invocation:
#
#     ESHKOL_TEST_TMPDIR       private scratch directory (removed on exit)
#     ESHKOL_TEST_BIN          path to compile test programs to (pass -o "$ESHKOL_TEST_BIN")
#     ESHKOL_TEST_OUT          captured program stdout/stderr
#     ESHKOL_TEST_COMPILE_LOG  captured compiler stdout/stderr
#     ESHKOL_TEST_REPO_ROOT    resolved repo root of *this* checkout
#     ESHKOL_TEST_REPO_TAG     stable short digest of the repo root (for lock names)
#
# ESHKOL_TEST_BIN deliberately keeps the basename `a.out` so eshkol-run's
# "compiled to 'a.out'" default-output notice is emitted exactly as before —
# this change is isolation plumbing only and must not alter suite semantics.
#
# DISK BUDGET (project rule: no unbounded temp growth)
#
# The scratch directory is removed by an EXIT trap.  Because a suite can be
# SIGKILLed (`Killed: 9` under memory pressure) before its trap runs, init also
# prunes leftover directories from previous runs — see
# eshkol_test_isolation_prune_stale.

# Guard against double-sourcing (run_all_tests.sh -> suite -> shared helper).
if [ -n "${ESHKOL_TEST_ISOLATION_SH_LOADED:-}" ]; then
    return 0 2>/dev/null || true
fi
ESHKOL_TEST_ISOLATION_SH_LOADED=1

# Where per-run scratch directories live.  Override with ESHKOL_TEST_TMP_ROOT
# when the default temp filesystem has no disk budget to spare.
eshkol_test_tmp_root() {
    printf '%s' "${ESHKOL_TEST_TMP_ROOT:-${TMPDIR:-/tmp}}" | sed 's:/*$::'
}

eshkol_test_isolation_fail() {
    echo "test_isolation: $1" >&2
    exit 1
}

# Resolve the repo root from this file's own location (scripts/lib/), not from
# $PWD — suites are invoked from the repo root, from ctest's build directory,
# and from run_all_tests.sh, and must all agree on which checkout they are in.
eshkol_test_repo_root() {
    local lib_dir
    lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
    (cd "$lib_dir/../.." && pwd)
}

# Stable short digest of a string, used to key locks and scratch names to a
# specific checkout.  cksum is POSIX and present on macOS, Linux and MSYS —
# no shasum/md5sum divergence to paper over.
eshkol_test_digest() {
    printf '%s' "$1" | cksum | awk '{print $1}'
}

# Remove scratch directories orphaned by killed runs.  Two bounds, both cheap:
# anything older than ESHKOL_TEST_TMP_MAX_AGE_MIN (default 720 = 12h) goes, and
# if more than ESHKOL_TEST_TMP_MAX_DIRS (default 64) remain we drop the oldest
# down to that cap.  Never touches a directory belonging to a live run within
# the age window, so concurrent suites do not prune each other.
eshkol_test_isolation_prune_stale() {
    local root max_age max_dirs
    root="$(eshkol_test_tmp_root)"
    max_age="${ESHKOL_TEST_TMP_MAX_AGE_MIN:-720}"
    max_dirs="${ESHKOL_TEST_TMP_MAX_DIRS:-64}"

    [ -d "$root" ] || return 0

    # Age-based sweep.
    find "$root" -maxdepth 1 -type d -name 'eshkol-test.*' \
        -mmin "+$max_age" -exec rm -rf -- {} + 2>/dev/null || true

    # Count-based sweep: oldest first, delete the overflow.
    local count
    count=$(find "$root" -maxdepth 1 -type d -name 'eshkol-test.*' 2>/dev/null | wc -l | tr -d ' ')
    [ -n "$count" ] || return 0
    if [ "$count" -gt "$max_dirs" ]; then
        # ls -dt sorts newest first; tail past the cap is the overflow.
        # shellcheck disable=SC2012
        ls -dt "$root"/eshkol-test.* 2>/dev/null \
            | tail -n "+$((max_dirs + 1))" \
            | while IFS= read -r stale; do
                case "$stale" in
                    "$root"/eshkol-test.*) rm -rf -- "$stale" 2>/dev/null || true ;;
                esac
            done
    fi
}

eshkol_test_isolation_cleanup() {
    local dir="${ESHKOL_TEST_TMPDIR:-}"
    [ -n "$dir" ] || return 0

    # Debugging escape hatch: keep this run's logs and binaries for inspection.
    # Off by default so the disk budget holds in CI.
    if [ -n "${ESHKOL_TEST_KEEP_TMPDIR:-}" ]; then
        echo "test_isolation: keeping scratch directory: $dir" >&2
        return 0
    fi

    # Only ever remove a directory we created: it must sit directly under the
    # temp root and carry our prefix.  Refuse symlinks outright.
    local root
    root="$(eshkol_test_tmp_root)"
    case "$dir" in
        "$root"/eshkol-test.*) ;;
        *)
            echo "test_isolation: refusing to remove unexpected scratch dir: $dir" >&2
            return 0
            ;;
    esac
    if [ -L "$dir" ]; then
        echo "test_isolation: refusing to remove symlinked scratch dir: $dir" >&2
        return 0
    fi

    rm -rf -- "$dir" 2>/dev/null || true
}

# Create this invocation's scratch directory and export the standard paths.
# $1 — short suite label (used in the directory name for debuggability).
eshkol_test_isolation_init() {
    local label="${1:-suite}"
    local root repo_root

    # Keep the label filesystem-safe.
    label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')"

    repo_root="$(eshkol_test_repo_root)" \
        || eshkol_test_isolation_fail "cannot resolve repo root"
    root="$(eshkol_test_tmp_root)"
    [ -d "$root" ] || mkdir -p -- "$root" 2>/dev/null || true
    [ -d "$root" ] || eshkol_test_isolation_fail "temp root is not a directory: $root"

    ESHKOL_TEST_REPO_ROOT="$repo_root"
    ESHKOL_TEST_REPO_TAG="$(eshkol_test_digest "$repo_root")"

    eshkol_test_isolation_prune_stale

    # mktemp -d gives the per-invocation uniqueness; the repo tag makes the name
    # say which checkout it belongs to, so a stray directory is attributable.
    ESHKOL_TEST_TMPDIR="$(mktemp -d "$root/eshkol-test.$label.$ESHKOL_TEST_REPO_TAG.XXXXXX")" \
        || eshkol_test_isolation_fail "failed to create scratch directory under $root"

    # Basename stays a.out on purpose — see the header note.
    ESHKOL_TEST_BIN="$ESHKOL_TEST_TMPDIR/a.out"
    ESHKOL_TEST_OUT="$ESHKOL_TEST_TMPDIR/run.out"
    ESHKOL_TEST_COMPILE_LOG="$ESHKOL_TEST_TMPDIR/compile.log"

    export ESHKOL_TEST_TMPDIR ESHKOL_TEST_BIN ESHKOL_TEST_OUT \
           ESHKOL_TEST_COMPILE_LOG ESHKOL_TEST_REPO_ROOT ESHKOL_TEST_REPO_TAG

    # Suites that need their own EXIT handler call eshkol_test_isolation_cleanup
    # from it and set ESHKOL_TEST_ISOLATION_NO_TRAP=1 to keep this from
    # overwriting theirs.
    if [ -z "${ESHKOL_TEST_ISOLATION_NO_TRAP:-}" ]; then
        trap eshkol_test_isolation_cleanup EXIT
    fi
}

# Drop this invocation's test binary and its .tmp.o sidecar between tests.
# Replaces the old `rm -f a.out a.out.tmp.o`, which reached into the repo root
# and could delete a *concurrent* suite's binary.
eshkol_test_reset_bin() {
    local bin="${1:-${ESHKOL_TEST_BIN:-}}"
    [ -n "$bin" ] || return 0
    rm -f -- "$bin" "$bin.tmp.o" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Pinning the binary under test
#
# A long suite shells out to $BUILD_DIR/eshkol-run for tens of minutes.  Nothing
# used to pin that binary, so a rebuild in the same worktree *during* the run
# swapped the compiler underneath the suite.  A 40-minute coverage run that
# straddled two relinks reported "93% pass, 6 failures including SEGFAULT in
# examples/autodiff.esk"; all six pass on a stable build.  The segfault was an
# artifact of the binary changing mid-run — the harness inventing a crash.
#
# Two tools, use whichever fits the suite:
#
#   eshkol_test_pin_toolchain <build_dir>
#       Copy the toolchain into this run's scratch dir and echo the pinned
#       directory.  Point the suite's ESHKOL_RUN at it and a concurrent rebuild
#       cannot touch what is under test.  Only usable by suites that address the
#       compiler through an absolute path variable — suites hardcoding
#       `./$BUILD_DIR/eshkol-run` need the repo-relative build dir and must use
#       the fingerprint guard instead.
#
#   eshkol_test_toolchain_snapshot <build_dir> / eshkol_test_toolchain_verify
#       Record size+mtime of every relinkable artifact at start and re-check
#       later.  Cheap, works for every suite, and converts a false verdict into
#       a loud, explicit invalidation.
# ---------------------------------------------------------------------------

# Artifacts a rebuild can swap under a running suite.
#
# The static archives belong here as much as the executables do. eshkol-run's
# AOT link resolves libeshkol-runtime.a through find_runtime_library(), whose
# candidate order is: $ESHKOL_LIB_DIR, then -L paths, then *the directory of the
# running executable*, then ./ and ./build. So copying the archives alongside a
# pinned eshkol-run is what makes the pin real for AOT — without them the link
# falls back to the live ./build tree and a concurrent relink is back in play.
ESHKOL_TEST_TOOLCHAIN_ARTIFACTS="eshkol-run stdlib.o stdlib.bc eshkol-repl eshkol-vm-standalone-test libeshkol-runtime.a libeshkol-static.a libeshkol-agent-ffi.a"

# Portable "size mtime" stamp for one file. BSD stat and GNU stat disagree on
# flags, so try both; a missing file stamps as "-" (absent is a stable state).
eshkol_test_file_stamp() {
    local path="$1"
    [ -e "$path" ] || { printf '%s' '-'; return 0; }
    stat -f '%z %m' "$path" 2>/dev/null && return 0
    stat -c '%s %Y' "$path" 2>/dev/null && return 0
    # Last resort: content digest.
    cksum < "$path" 2>/dev/null || printf '%s' '?'
}

eshkol_test_toolchain_fingerprint() {
    local build_dir="$1"
    local name
    for name in $ESHKOL_TEST_TOOLCHAIN_ARTIFACTS; do
        printf '%s=%s\n' "$name" "$(eshkol_test_file_stamp "$build_dir/$name")"
    done
}

# Record the toolchain state at the start of a run.
# $1 — build dir. $2 — optional slot name, so a suite that drives more than one
# build tree (core plus a quantum-enabled tree, say) can guard each of them.
eshkol_test_toolchain_snapshot() {
    local build_dir="$1"
    local slot="${2:-default}"
    [ -n "${ESHKOL_TEST_TMPDIR:-}" ] || return 0
    if [ "$slot" = default ]; then
        ESHKOL_TEST_PINNED_SOURCE_DIR="$build_dir"
        export ESHKOL_TEST_PINNED_SOURCE_DIR
    fi
    eshkol_test_toolchain_fingerprint "$build_dir" \
        > "$ESHKOL_TEST_TMPDIR/toolchain.$slot.fingerprint" 2>/dev/null || true
}

# Re-check the toolchain. Returns 0 if unchanged, 1 if it moved under us — in
# which case every result the suite gathered is suspect and the caller must say
# so instead of reporting pass/fail. $1 — build dir (defaults to the snapshot's).
# $2 — slot name, matching the snapshot call.
eshkol_test_toolchain_verify() {
    local build_dir="${1:-${ESHKOL_TEST_PINNED_SOURCE_DIR:-}}"
    local slot="${2:-default}"
    local snap="${ESHKOL_TEST_TMPDIR:-}/toolchain.$slot.fingerprint"

    [ -n "$build_dir" ] || return 0
    [ -f "$snap" ] || return 0

    local now
    now="$(eshkol_test_toolchain_fingerprint "$build_dir")"
    if [ "$now" = "$(cat "$snap" 2>/dev/null)" ]; then
        return 0
    fi

    echo "" >&2
    echo "===========================================================" >&2
    echo "INVALID RUN: the compiler binary changed during this run." >&2
    echo "Results are invalid and are NOT being reported." >&2
    echo "" >&2
    echo "  build dir: $build_dir" >&2
    echo "  changed artifacts:" >&2
    local line name
    while IFS= read -r line; do
        name="${line%%=*}"
        case "$now" in
            *"$line"*) ;;
            *) echo "    - $name" >&2 ;;
        esac
    done < "$snap"
    echo "" >&2
    echo "Something rebuilt $build_dir while the suite was running, so the" >&2
    echo "tests were not all run against the same compiler. Re-run against a" >&2
    echo "stable build before drawing any conclusion — in particular, do not" >&2
    echo "trust crash or failure verdicts from this run." >&2
    echo "===========================================================" >&2
    echo "" >&2
    return 1
}

# Copy the toolchain into this run's scratch dir; echo the pinned directory.
# A real copy, not a hardlink: a hardlink shares an inode with the build tree,
# so an in-place relink would still mutate what we are holding.
# $1 — source build dir.
eshkol_test_pin_toolchain() {
    local build_dir="$1"
    local pinned name

    [ -n "${ESHKOL_TEST_TMPDIR:-}" ] \
        || eshkol_test_isolation_fail "eshkol_test_pin_toolchain before eshkol_test_isolation_init"
    [ -d "$build_dir" ] \
        || eshkol_test_isolation_fail "cannot pin toolchain, no such build dir: $build_dir"

    pinned="$ESHKOL_TEST_TMPDIR/toolchain"
    mkdir -p -- "$pinned" \
        || eshkol_test_isolation_fail "cannot create pinned toolchain dir: $pinned"

    # eshkol-run discovers stdlib.o/stdlib.bc relative to its own location, so
    # copying them alongside keeps the pinned compiler self-sufficient.
    for name in $ESHKOL_TEST_TOOLCHAIN_ARTIFACTS; do
        [ -f "$build_dir/$name" ] || continue
        cp -p -- "$build_dir/$name" "$pinned/$name" 2>/dev/null || true
    done

    [ -x "$pinned/eshkol-run" ] \
        || eshkol_test_isolation_fail "failed to pin eshkol-run from $build_dir"

    # Record the source state too, so a suite can still report that the tree it
    # pinned from was rebuilt (useful signal even though results stay valid).
    eshkol_test_toolchain_snapshot "$build_dir"

    ESHKOL_TEST_PINNED_BUILD_DIR="$pinned"
    export ESHKOL_TEST_PINNED_BUILD_DIR
    printf '%s' "$pinned"
}

# Path of a lock directory scoped to *this checkout* rather than the machine, so
# two worktrees never serialize on — or falsely block — each other.
# $1 — lock label, e.g. "v12-edge".
eshkol_test_lock_path() {
    local label="${1:-suite}"
    local repo_root tag
    repo_root="${ESHKOL_TEST_REPO_ROOT:-$(eshkol_test_repo_root)}"
    tag="${ESHKOL_TEST_REPO_TAG:-$(eshkol_test_digest "$repo_root")}"
    label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')"
    printf '%s/eshkol-lock.%s.%s.d' "$(eshkol_test_tmp_root)" "$label" "$tag"
}

# Acquire the per-repo-root lock for a genuinely non-reentrant suite.
# Returns 0 on success; 1 if another run *of this same checkout* holds it.
# Stale locks (holder gone) are reclaimed so a killed run cannot wedge CI.
# $1 — lock label. Sets ESHKOL_TEST_LOCK_PATH on success.
eshkol_test_acquire_lock() {
    local label="${1:-suite}"
    local lock
    lock="$(eshkol_test_lock_path "$label")"

    if ! mkdir "$lock" 2>/dev/null; then
        # Reclaim if the recorded holder is no longer running.
        local holder=""
        [ -f "$lock/pid" ] && holder="$(cat "$lock/pid" 2>/dev/null)"
        if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
            rm -rf -- "$lock" 2>/dev/null || true
            mkdir "$lock" 2>/dev/null || return 1
        else
            return 1
        fi
    fi

    printf '%s\n' "$$" > "$lock/pid" 2>/dev/null || true
    ESHKOL_TEST_LOCK_PATH="$lock"
    export ESHKOL_TEST_LOCK_PATH
    return 0
}

eshkol_test_release_lock() {
    local lock="${ESHKOL_TEST_LOCK_PATH:-}"
    [ -n "$lock" ] || return 0
    case "$lock" in
        */eshkol-lock.*.d) rm -rf -- "$lock" 2>/dev/null || true ;;
    esac
}

# ---------------------------------------------------------------------------
# Honest failure detection
#
# An Eshkol .esk test program prints its own verdicts and then exits 0 — the
# process status says nothing about whether the assertions held.  A suite that
# decides PASS from the exit status, or that greps for a failure marker with a
# `^`-anchored pattern, therefore certifies broken code.
#
# The concrete case that motivated this: tests/gpu/sf64_primitives_test.esk
# prints every failure as `  <case>: FAIL` — two leading spaces, bare `FAIL`,
# no trailing colon, and no `Failed: N` summary anywhere.  run_gpu_tests.sh
# matched `^FAIL:`, so four genuinely failing cases were invisible and the gate
# reported PASS.
#
# Rules encoded here:
#   1. A failure marker anywhere in the output — any column, any line — fails.
#   2. Absence of an expected success marker fails.  A test that crashed
#      before printing anything, or printed nothing at all, is NOT a pass.
#
# Case matters and is load-bearing: the markers are the upper-case tokens test
# programs print, so summary prose like "Failed Tests:" or "Failed: 0" does not
# trip rule 1 while "Failed: 3" does.
# ---------------------------------------------------------------------------

# Unanchored failure markers. Deliberately NOT including bare `ERROR:` —
# several error-handling tests print that as their *expected* output. Suites
# that want it pass it as the extra pattern argument.
ESHKOL_TEST_FAILURE_REGEX='(^|[^A-Za-z0-9_])(FAIL|FAILED|FAILURE|FAILS)([^A-Za-z0-9_]|$)|Failed:[[:space:]]*[1-9]|Failures:[[:space:]]*[1-9]|^[[:space:]]*✗|Assertion failed|Segmentation fault|Bus error|Abort trap|fatal signal'

# True when $1 (a file) contains a failure marker. $2 — optional extra ERE,
# OR-ed in for suites with their own markers (e.g. 'RESULT: FAIL|error:').
eshkol_test_output_has_failure() {
    local file="$1"
    local extra="${2:-}"
    local pattern="$ESHKOL_TEST_FAILURE_REGEX"

    [ -f "$file" ] || return 1
    [ -n "$extra" ] && pattern="$pattern|$extra"
    LC_ALL=C grep -Eq -- "$pattern" "$file" 2>/dev/null
}

# Same, for output already captured in a shell variable.
eshkol_test_text_has_failure() {
    local text="$1"
    local extra="${2:-}"
    local pattern="$ESHKOL_TEST_FAILURE_REGEX"

    [ -n "$extra" ] && pattern="$pattern|$extra"
    printf '%s\n' "$text" | LC_ALL=C grep -Eq -- "$pattern" 2>/dev/null
}

# Print the offending lines so a suite can show *why* it failed.
# $1 — file, $2 — optional extra ERE, $3 — max lines (default 10).
eshkol_test_output_failures() {
    local file="$1"
    local extra="${2:-}"
    local limit="${3:-10}"
    local pattern="$ESHKOL_TEST_FAILURE_REGEX"

    [ -f "$file" ] || return 0
    [ -n "$extra" ] && pattern="$pattern|$extra"
    LC_ALL=C grep -En -- "$pattern" "$file" 2>/dev/null | head -n "$limit"
}

# True when the file holds nothing but whitespace — "the test printed nothing".
# Rule 2's cheapest form, usable by suites that have no single success marker.
eshkol_test_output_is_silent() {
    local file="$1"
    [ -f "$file" ] || return 0
    ! LC_ALL=C grep -q '[^[:space:]]' "$file" 2>/dev/null
}

# True when the file contains the required success marker $2 (an ERE).
# Absence is a failure — see rule 2.
eshkol_test_output_has_marker() {
    local file="$1"
    local marker="$2"
    [ -f "$file" ] || return 1
    [ -n "$marker" ] || return 0
    LC_ALL=C grep -Eq -- "$marker" "$file" 2>/dev/null
}
