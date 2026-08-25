#!/usr/bin/env bash
#
# ADR-0010 gap A12 self-test: proves that ASan leak detection is actually
# ARMED under the exact ASAN_OPTIONS/LSAN_OPTIONS configuration the
# linux-x64-asan-ubsan CI lane uses, not merely configured on paper.
#
# "A gate that cannot fail is not a gate": this script compiles TWO tiny,
# throwaway probes under -fsanitize=address —
#
#   1. a program that deliberately leaks a uniquely-sized, uniquely-tagged
#      allocation and never frees or stores it anywhere reachable. Run under
#      this repo's real ASAN_OPTIONS/LSAN_OPTIONS (including the checked-in
#      suppression file, .icc/lsan-suppressions.txt), LeakSanitizer MUST
#      report it — proving detect_leaks=1 is actually in effect and the
#      suppression file has not silently grown broad enough to swallow an
#      ordinary leak.
#   2. a program that allocates and frees cleanly. Under the same
#      configuration it MUST exit 0 with no LeakSanitizer report — proving
#      the gate does not simply fail every run regardless of content.
#
# Both probes are generated into a private, repo-local scratch directory
# (never /tmp) and removed before this script exits either way.
#
# Usage
#   ./scripts/check_leak_detection_selftest.sh
#   ./scripts/check_leak_detection_selftest.sh --no-trace
#
# Exit status is 0 when both probes behave as expected, 1 otherwise.
#
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/scripts/icc_traces}"
TRACE_FILE="$TRACE_DIR/leak_detection_gate.jsonl"
SUPPRESSIONS="$REPO_ROOT/.icc/lsan-suppressions.txt"
PROBE_ID="leak_detection_gate"

NO_TRACE=0
for arg in "$@"; do
    case "$arg" in
        --no-trace) NO_TRACE=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# Find a compiler whose ASan runtime actually supports LeakSanitizer at
# RUNTIME, not merely one that accepts -fsanitize=address at compile time.
# Apple's shipped Xcode clang compiles -fsanitize=address fine but its ASan
# runtime refuses detect_leaks outright ("not supported on this platform"),
# which is a silent-pass failure mode indistinguishable from "no leak" if
# this script only checked compile success. Homebrew's llvm@N clang ships a
# compiler-rt build that does support it; on Linux CI, the apt.llvm.org
# clang used by the ASan lane supports it natively.
find_leak_capable_cc() {
    local candidates=()
    [ -n "${CC:-}" ] && candidates+=("$CC")
    for prefix in /opt/homebrew/opt/llvm@21 /opt/homebrew/opt/llvm /usr/local/opt/llvm@21 /usr/local/opt/llvm; do
        [ -x "$prefix/bin/clang" ] && candidates+=("$prefix/bin/clang")
    done
    candidates+=("clang" "cc")

    local probe_dir candidate
    probe_dir="$(mktemp -d "$REPO_ROOT/.selftest-leak-cc-probe-XXXXXX")"
    trap 'rm -rf "$probe_dir"' RETURN
    cat > "$probe_dir/probe.c" <<'PROBE_EOF'
#include <stdlib.h>
int main(void) { void* p = malloc(16); (void)p; return 0; }
PROBE_EOF

    for candidate in "${candidates[@]}"; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        if ! "$candidate" -fsanitize=address -g -O0 "$probe_dir/probe.c" -o "$probe_dir/probe" >/dev/null 2>&1; then
            continue
        fi
        local out
        out="$(ASAN_OPTIONS=detect_leaks=1 "$probe_dir/probe" 2>&1 || true)"
        if echo "$out" | grep -q "not supported on this platform"; then
            continue
        fi
        echo "$candidate"
        return 0
    done
    return 1
}

CC="$(find_leak_capable_cc)" || {
    echo "FAIL: no compiler on this host has an ASan runtime that supports LeakSanitizer" >&2
    echo "      (tried \$CC, Homebrew llvm@21/llvm, clang, cc). This self-test cannot" >&2
    echo "      prove anything on a host where leak detection is not runtime-capable." >&2
    exit 1
}

emit_event() {
    local status="$1" snippet="$2"
    if [ "$NO_TRACE" -eq 1 ]; then return; fi
    mkdir -p "$TRACE_DIR"
    python3 -c '
import json, sys
print(json.dumps({"kind": "eshkol_smoke", "name": sys.argv[1],
                  "value": sys.argv[2], "snippet": sys.argv[3],
                  "confidence": 1.0}, ensure_ascii=False))
' "$PROBE_ID" "$status" "$snippet" >> "$TRACE_FILE"
}

if [ ! -f "$SUPPRESSIONS" ]; then
    echo "FAIL: suppression file not found at $SUPPRESSIONS (the gate fails closed — a" >&2
    echo "      missing suppression file must not silently fall back to no suppressions" >&2
    echo "      or, worse, to an LSAN_OPTIONS parse error that disables checking)." >&2
    emit_event "FAIL" "suppression file not found at $SUPPRESSIONS"
    exit 1
fi

SCRATCH="$REPO_ROOT/.selftest-leak-detection-$$"
mkdir -p "$SCRATCH"
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

# A unique tag per run avoids ever matching a real suppression rule by
# accident (a rule aimed at a real Eshkol/LLVM allocation site cannot also
# happen to match this synthetic marker).
TAG="selftest_leak_marker_$$"

cat > "$SCRATCH/leaky.c" <<EOF
#include <stdlib.h>
/* ${TAG}: deliberately leaked, never freed, never stored anywhere reachable. */
int main(void) {
    void* p = malloc(4096);
    (void)p;
    return 0;
}
EOF

cat > "$SCRATCH/clean.c" <<'EOF'
#include <stdlib.h>
int main(void) {
    void* p = malloc(4096);
    free(p);
    return 0;
}
EOF

"$CC" -fsanitize=address -g -O0 "$SCRATCH/leaky.c" -o "$SCRATCH/leaky"
"$CC" -fsanitize=address -g -O0 "$SCRATCH/clean.c" -o "$SCRATCH/clean"

EFFECTIVE_SUPPRESSIONS="$SUPPRESSIONS"

# The CI lane this proves is Linux (linux-x64-asan-ubsan, ubuntu-22.04),
# where LSan is native. On macOS, the objc runtime / libxpc's own one-time
# process-init allocations are independently known to read as LSan "leaks"
# regardless of anything this repo does (verified: they appear even for a
# probe with no Eshkol code at all). Layering a macOS-only, non-shipped
# suppression file in for local verification keeps that platform noise out
# of the checked-in .icc/lsan-suppressions.txt (which must stay Linux-CI-
# accurate) while still letting this self-test run meaningfully on a
# developer's Mac.
if [ "$(uname -s)" = "Darwin" ]; then
    DARWIN_NOISE="$SCRATCH/darwin-init-noise-suppressions.txt"
    cat > "$DARWIN_NOISE" <<'EOF'
# macOS-only process-init allocator noise (objc/xpc runtime), NOT shipped —
# this file exists only for local self-test runs on Darwin. The CI lane this
# gate protects (linux-x64-asan-ubsan) never sees these frames.
leak:_fetchInitializingClassList
leak:_libxpc_initializer
leak:libSystem_initializer
EOF
    EFFECTIVE_SUPPRESSIONS="$SCRATCH/merged-suppressions.txt"
    cat "$SUPPRESSIONS" "$DARWIN_NOISE" > "$EFFECTIVE_SUPPRESSIONS"
fi

export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:allocator_may_return_null=1"
export LSAN_OPTIONS="suppressions=$EFFECTIVE_SUPPRESSIONS:print_suppressions=0"

leaky_out="$SCRATCH/leaky.out"
leaky_exit=0
"$SCRATCH/leaky" > "$leaky_out" 2>&1 || leaky_exit=$?

clean_out="$SCRATCH/clean.out"
clean_exit=0
"$SCRATCH/clean" > "$clean_out" 2>&1 || clean_exit=$?

failures=0
reasons=""

if [ "$leaky_exit" -eq 0 ]; then
    failures=$((failures + 1))
    reasons="$reasons; the deliberately-leaking probe exited 0 (LeakSanitizer did not fire under this repo's ASAN_OPTIONS/LSAN_OPTIONS)"
fi
if ! grep -q "LeakSanitizer" "$leaky_out"; then
    failures=$((failures + 1))
    reasons="$reasons; the deliberately-leaking probe's output has no LeakSanitizer report"
fi

if [ "$clean_exit" -ne 0 ]; then
    failures=$((failures + 1))
    reasons="$reasons; the clean (leak-free) probe exited $clean_exit — a suppression or option is misconfigured"
fi
if grep -q "LeakSanitizer" "$clean_out"; then
    failures=$((failures + 1))
    reasons="$reasons; the clean (leak-free) probe was reported as leaking (false positive)"
fi

echo "$PROBE_ID: leaky probe exit=$leaky_exit, clean probe exit=$clean_exit"
if [ -s "$leaky_out" ]; then
    echo "--- leaky probe output (head) ---"
    head -5 "$leaky_out"
fi

if [ "$failures" -eq 0 ]; then
    echo "$PROBE_ID: PASS — leak detection is armed (catches a real leak) and not over-broad (does not flag a clean program)"
    emit_event "PASS" "leaky probe exit=$leaky_exit with LeakSanitizer report; clean probe exit=$clean_exit with none"
    exit 0
else
    echo "$PROBE_ID: FAIL$reasons" >&2
    emit_event "FAIL" "$failures failure(s):$reasons"
    exit 1
fi
