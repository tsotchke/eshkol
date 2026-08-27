#!/usr/bin/env bash
# build_fingerprint.sh — record which exact binary a harness ran against.
#
# WHY THIS EXISTS
#
# D-11 in docs/design/FLAW_DETECTION_ROADMAP.md: a harness was run against a
# STALE binary after a rebase, and the result was believed. Two false verdicts
# in the v1.3.4 campaign trace back to this — the sharpest was #418's new
# corpus file failing `engine_semantic_parity` because the binary under test
# predated the source that had just been rebased in.
#
# `scripts/lib/test_isolation.sh` already solves the SIBLING problem — a
# rebuild swapping the binary out from under a suite MID-RUN
# (`eshkol_test_pin_toolchain` / `eshkol_test_toolchain_verify`). This file
# solves a different one: proving, AFTER the fact, that the binary a harness's
# evidence talks about is (a) still the binary on disk when a gate reads that
# evidence, and (b) was not already stale — built before a source change that
# should have triggered a rebuild — at the moment the harness started.
#
# THE CONTRACT
#
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/build_fingerprint.sh"
#   eshkol_emit_build_fingerprint_event "$TRACE_DIR" "run_icc_smoke" "$BUILD_DIR" eshkol-run
#
# appends one JSON-L record to `<trace_dir>/build_fingerprint.jsonl` (created,
# never truncated — many harnesses share this file across one CI run) naming:
#
#   harness           which script recorded this fingerprint
#   binary / path     which file, and where
#   size / mtime      the file's stat() at record time
#   sha256            a real content digest (mtime+size can lie across
#                     filesystems with coarse mtime resolution; the digest
#                     cannot)
#   git_sha           `git rev-parse HEAD` in this checkout, if available
#   recorded_epoch    when this event was written
#
# `scripts/check_build_fingerprint.py` reads that file at gate time and fails
# when a recorded fingerprint no longer matches the binary now on disk, or
# when the binary on disk predates the source tree it was supposedly built
# from.

if [ -n "${ESHKOL_BUILD_FINGERPRINT_SH_LOADED:-}" ]; then
    return 0 2>/dev/null || true
fi
ESHKOL_BUILD_FINGERPRINT_SH_LOADED=1

# Portable "size mtime" stamp — see test_isolation.sh's eshkol_test_file_stamp
# for why GNU stat must be tried before BSD stat's `-f` (which is a totally
# different, filesystem-level flag on BSD/macOS).
eshkol_fp_stat() {
    local path="$1"
    stat -c '%s %Y' "$path" 2>/dev/null && return 0
    stat -f '%z %m' "$path" 2>/dev/null && return 0
    return 1
}

# Content digest of one file. Tries sha256sum (GNU/Linux), then shasum -a 256
# (macOS/BSD), then openssl as a last resort. Prints "unavailable" rather than
# failing outright — a harness recording a fingerprint must not itself die
# because no digest tool exists on this host.
eshkol_fp_sha256() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" 2>/dev/null | awk '{print $1}' && return 0
    fi
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$path" 2>/dev/null | awk '{print $1}' && return 0
    fi
    if command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$path" 2>/dev/null | awk '{print $NF}' && return 0
    fi
    printf '%s' "unavailable"
}

# HEAD sha of the checkout containing $1 (a file or directory), or "unknown".
eshkol_fp_git_sha() {
    local at="$1"
    local dir
    dir="$at"
    [ -d "$dir" ] || dir="$(dirname "$dir")"
    git -C "$dir" rev-parse HEAD 2>/dev/null || printf '%s' "unknown"
}

# Append one build-fingerprint event.
#   $1 trace_dir   directory the JSON-L file lives in (created if missing)
#   $2 harness     short name of the calling harness (e.g. "run_icc_smoke")
#   $3 build_dir   build directory containing the binary
#   $4 binary      binary filename, default "eshkol-run"
#
# Never fails the caller: a harness that cannot record a fingerprint should
# still be able to run its own probes. Prints a warning to stderr instead.
eshkol_emit_build_fingerprint_event() {
    local trace_dir="$1" harness="$2" build_dir="$3" binary="${4:-eshkol-run}"
    local path="$build_dir/$binary"

    if [ ! -e "$path" ]; then
        echo "build_fingerprint: $path does not exist — not recording a fingerprint for $harness" >&2
        return 0
    fi

    mkdir -p "$trace_dir" 2>/dev/null || true
    local out="$trace_dir/build_fingerprint.jsonl"

    local stamp size mtime sha git_sha
    stamp="$(eshkol_fp_stat "$path")" || stamp="0 0"
    size="${stamp%% *}"
    mtime="${stamp##* }"
    sha="$(eshkol_fp_sha256 "$path")"
    git_sha="$(eshkol_fp_git_sha "$build_dir")"

    python3 -c '
import json, sys, time
harness, binary, path, size, mtime, sha, git_sha = sys.argv[1:8]
print(json.dumps({
    "kind": "build_fingerprint",
    "harness": harness,
    "binary": binary,
    "path": path,
    "size": int(size),
    "mtime": int(float(mtime)),
    "sha256": sha,
    "git_sha": git_sha,
    "recorded_epoch": int(time.time()),
}, ensure_ascii=False))
' "$harness" "$binary" "$path" "$size" "$mtime" "$sha" "$git_sha" >> "$out" 2>/dev/null \
        || echo "build_fingerprint: failed to write $out" >&2

    echo "build_fingerprint: $harness -> $binary size=$size mtime=$mtime sha256=${sha:0:12}... git=$git_sha" >&2
}
