#!/usr/bin/env bash
#
# Continuation re-entry gate.
#
# Every fixture in tests/continuations/ is run on all three engines — native
# JIT (-r), native AOT, and the bytecode VM — and its transcript compared
# against a committed expected file. Comparing the exact transcript on all
# three, rather than looking for a PASS marker, is the point: these programs
# measure WHERE control resumes, and the classic way to get that wrong is to
# produce plausible output in the wrong order or from the wrong extent.
#
# These fixtures used to live outside CI because they crashed (native SIGILL /
# SIGSEGV, ledger SW-60) or hung (bytecode VM, SW-61) by design. Both defects
# are fixed, so they are gates now.
#
# Output is normalised exactly as scripts/run_vm_parity.sh does: strip banner
# and compiler-noise lines, then remove ALL newlines. The VM emits a newline
# after every `display` where native emits none, so per-line comparison cannot
# align the two; the newline-free byte stream is the strongest comparison that
# quirk permits, and it preserves spaces, so list spelling stays significant.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

BUILD_DIR=${BUILD_DIR:-build}
RUN="$BUILD_DIR/eshkol-run"
VM="$BUILD_DIR/eshkol-vm-standalone-test"
EXPECTED_DIR=tests/continuations/expected
WORK=${ESHKOL_CONT_WORK:-$BUILD_DIR/continuation-tests}
TIMEOUT_SECS=${ESHKOL_CONT_TIMEOUT:-60}

if [ ! -x "$RUN" ]; then
    echo "Missing $RUN — build first." >&2
    exit 2
fi
if [ ! -x "$VM" ]; then
    echo "Missing $VM — build first." >&2
    exit 2
fi

mkdir -p "$WORK"
PASS=0
FAIL=0

# Normalise a transcript (mirrors scripts/run_vm_parity.sh's normalize).
normalise() {
    perl -ne 'next if
        /^WARN/ or /^INFO:/ or /^DEBUG/ or
        /^\[ESKB\]/ or /^\[GPU\]/ or /^\s*\[compiled:/ or
        /^=== Eshkol VM/ or /^=== Execution complete ===/ or
        /^remark:/ or /^warning: <unknown>/;
        print' | tr -d '\n'
}

# Run a command with a timeout, portably (macOS has no timeout(1)).
run_bounded() {
    python3 - "$TIMEOUT_SECS" "$@" <<'PY'
import subprocess, sys
secs = int(sys.argv[1])
try:
    p = subprocess.run(sys.argv[2:], capture_output=True, text=True,
                       timeout=secs, errors="replace")
    sys.stdout.write(p.stdout or "")
    sys.exit(p.returncode)
except subprocess.TimeoutExpired:
    sys.stderr.write("TIMEOUT after %ds\n" % secs)
    sys.exit(124)
PY
}

check() {
    local label="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo "PASSED $label"
        PASS=$((PASS + 1))
    else
        echo "FAILED $label"
        echo "  expected: $expected"
        echo "  actual  : $actual"
        FAIL=$((FAIL + 1))
    fi
}

for test_file in tests/continuations/*.esk; do
    [ -f "$test_file" ] || continue
    name=$(basename "$test_file" .esk)
    exp_file="$EXPECTED_DIR/$name.txt"
    if [ ! -f "$exp_file" ]; then
        echo "FAILED $test_file::expected-file-present"
        echo "  missing $exp_file — a continuation fixture must pin its transcript"
        FAIL=$((FAIL + 1))
        continue
    fi
    want=$(normalise < "$exp_file")

    got=$(run_bounded "$RUN" -r "$test_file" 2>/dev/null | normalise)
    check "$test_file::native-jit" "$want" "$got"

    exe="$WORK/$name"
    if run_bounded "$RUN" -o "$exe" "$test_file" >/dev/null 2>&1 && [ -x "$exe" ]; then
        got=$(run_bounded "$exe" 2>/dev/null | normalise)
        check "$test_file::native-aot" "$want" "$got"
    else
        echo "FAILED $test_file::native-aot"
        echo "  AOT compile failed"
        FAIL=$((FAIL + 1))
    fi

    got=$(ESHKOL_VM_NO_DISASM=1 ESHKOL_VM_REGION_QUIET=1 \
          run_bounded "$VM" "$test_file" 2>/dev/null | normalise)
    check "$test_file::vm" "$want" "$got"
done

echo ""
echo "continuations: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
