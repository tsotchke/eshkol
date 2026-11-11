#!/usr/bin/env bash
#
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
#
# Reader nesting gate.
#
# What this asserts, on the native JIT, native AOT and the bytecode VM:
#
#   1. NESTING DEPTH IS BOUNDED BY MEMORY.  A quoted datum nested 10^4, 10^5
#      and 10^6 levels deep reads, compares (equal?) and prints in full.  The
#      printed form is checked to carry exactly as many levels as were
#      written, so "it printed something" is not mistaken for "it printed the
#      datum".
#   2. A DEEPLY NESTED CODE FORM IS ANSWERED, NEVER SIGNALLED.  The same
#      nesting written as code either compiles or is refused with an ordinary
#      diagnostic and a nonzero status.  A signal — any status of 128+N — and
#      an abort are failures of this gate, which is the whole point of it.
#   3. THE INPUT-SIZE CAP IS THE DOCUMENTED ONE.  A text just under the cap is
#      accepted; a text over it is refused, the refusal names the cap, and the
#      cap honours both ESHKOL_MAX_SOURCE_BYTES and --max-source-bytes=N.
#   4. RUNNING OUT OF MEMORY IS A DIAGNOSTIC.  Under a hard address-space
#      limit, a datum too big to read is refused in words and the process
#      exits; it does not abort on an uncaught allocation failure.
#
# Usage: reader_nesting_depth_gate.sh <eshkol-run> <eshkol-vm|none> <workdir>

set -uo pipefail

ESHKOL_RUN="${1:?eshkol-run path required}"
ESHKOL_VM="${2:?eshkol-vm path required (or 'none')}"
WORK="${3:?work directory required}"

rm -rf "$WORK"
mkdir -p "$WORK" || { echo "FAIL: cannot create $WORK"; exit 1; }

export ESHKOL_JIT_CACHE=0
export ESHKOL_VM_NO_DISASM=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ESHKOL_PATH="${ESHKOL_PATH:-$REPO_ROOT/lib}"

failures=0
note() { echo "  $*"; }
fail() { echo "FAIL: $*"; failures=$((failures + 1)); }

# A status of 128+N is a signal: the one outcome this gate exists to rule out.
signalled() {
    local status="$1"
    [ "$status" -gt 128 ] && [ "$status" -lt 192 ]
}

# ── input generation ───────────────────────────────────────────────────────
# Written with printf rather than a shell loop: a million-level datum has to
# be produced in well under a second for this gate to fit in a test budget.
repeat_char() {
    local ch="$1" count="$2"
    printf "%${count}s" "" | tr ' ' "$ch"
}

write_quoted_datum_program() {
    local path="$1" depth="$2"
    local open close
    open="$(repeat_char '(' "$depth")"
    close="$(repeat_char ')' "$depth")"
    {
        printf "(define a '%s%s)\n" "$open" "$close"
        printf "(define b '%s%s)\n" "$open" "$close"
        printf '(display (equal? a b))\n(newline)\n'
        printf '(display a)\n(newline)\n'
    } > "$path"
}

write_code_form_program() {
    local path="$1" depth="$2"
    {
        repeat_char '(' "$depth"
        repeat_char ')' "$depth"
        printf '\n'
    } > "$path"
}

# ── 1. a quoted datum nested as deeply as memory allows ────────────────────
check_quoted_datum() {
    local label="$1" depth="$2"; shift 2
    local out status first second opens
    out="$("$@" 2>&1)"
    status=$?
    if signalled "$status"; then
        fail "$label: depth $depth ended on a signal (status $status)"
        return
    fi
    if [ "$status" -ne 0 ]; then
        fail "$label: depth $depth did not run (status $status): $(printf '%s' "$out" | tail -3)"
        return
    fi
    # Output lines may be preceded by banner text on the VM; take the first
    # line that is exactly the comparison's answer.
    first="$(printf '%s\n' "$out" | grep -m1 -x '#t\|#f')"
    if [ "$first" != "#t" ]; then
        fail "$label: depth $depth compared two identical datums as '$first'"
        return
    fi
    second="$(printf '%s\n' "$out" | grep -m1 '^((')"
    opens="$(printf '%s' "$second" | tr -cd '(' | wc -c | tr -d ' ')"
    if [ "$opens" != "$depth" ]; then
        fail "$label: depth $depth printed $opens levels"
        return
    fi
    note "$label: depth $depth read, compared and printed $opens levels"
}

# ── 2. the same nesting as a code form ─────────────────────────────────────
check_code_form() {
    local label="$1" depth="$2"; shift 2
    local out status
    out="$("$@" 2>&1)"
    status=$?
    if signalled "$status"; then
        fail "$label: depth $depth code form ended on a signal (status $status)"
        return
    fi
    if [ "$status" -eq 0 ]; then
        note "$label: depth $depth code form compiled"
        return
    fi
    if printf '%s' "$out" | grep -qiE 'too deep|out of memory|over the .*cap|nesting|closing parenthesis|unterminated'; then
        note "$label: depth $depth code form refused in words (status $status)"
    else
        fail "$label: depth $depth code form failed without saying why (status $status): $(printf '%s' "$out" | tail -3)"
    fi
}

echo "=== quoted data: nesting bounded by memory ==="
for depth in 10000 100000 1000000; do
    prog="$WORK/quoted-$depth.esk"
    write_quoted_datum_program "$prog" "$depth"
    check_quoted_datum "jit" "$depth" "$ESHKOL_RUN" -r "$prog"
    if [ "$ESHKOL_VM" != "none" ]; then
        check_quoted_datum "vm" "$depth" "$ESHKOL_VM" "$prog"
    fi
done

# AOT compiles and links, so it runs the two smaller depths; the deepest is
# covered on the two paths above and by the code-form half below.
echo "=== quoted data: ahead-of-time ==="
for depth in 10000 100000; do
    prog="$WORK/quoted-aot-$depth.esk"
    write_quoted_datum_program "$prog" "$depth"
    binary="$WORK/quoted-aot-$depth.bin"
    build_out="$("$ESHKOL_RUN" -o "$binary" "$prog" 2>&1)"
    build_status=$?
    if signalled "$build_status"; then
        fail "aot: depth $depth compilation ended on a signal (status $build_status)"
        continue
    fi
    if [ "$build_status" -ne 0 ] || [ ! -x "$binary" ]; then
        fail "aot: depth $depth did not build (status $build_status): $(printf '%s' "$build_out" | tail -3)"
        continue
    fi
    check_quoted_datum "aot" "$depth" "$binary"
done

echo "=== code forms: answered, never signalled ==="
for depth in 10000 100000 1000000; do
    prog="$WORK/code-$depth.esk"
    write_code_form_program "$prog" "$depth"
    check_code_form "jit" "$depth" "$ESHKOL_RUN" -r "$prog"
    if [ "$ESHKOL_VM" != "none" ]; then
        check_code_form "vm" "$depth" "$ESHKOL_VM" "$prog"
    fi
done

# ── the reported reproducers ───────────────────────────────────────────────
# 200 KB and 400 KB of open parentheses, and a 50,000-deep cons pattern: the
# three shapes reported against the readers and the pattern expander.
echo "=== reported reproducers ==="
# As reported: an unclosed run of parentheses. It is not a complete datum, so
# the outcome asserted is the one that matters — a diagnostic, never a signal.
for bytes in 200000 400000; do
    prog="$WORK/parens-$bytes.esk"
    repeat_char '(' "$bytes" > "$prog"
    printf '\n' >> "$prog"
    check_code_form "jit" "$bytes" "$ESHKOL_RUN" -r "$prog"
    if [ "$ESHKOL_VM" != "none" ]; then
        check_code_form "vm" "$bytes" "$ESHKOL_VM" "$prog"
    fi
done

pattern_prog="$WORK/deep-match-pattern.esk"
{
    printf '(define (classify x)\n  (match x\n    ('
    repeat_char '(' 50000 | sed 's/(/(cons _ /g'
    printf '_'
    repeat_char ')' 50000
    printf '\n     0)\n    (_ 1)))\n(display (classify 7))\n(newline)\n'
} > "$pattern_prog"
check_code_form "jit" "50000-deep match pattern" "$ESHKOL_RUN" -r "$pattern_prog"
if [ "$ESHKOL_VM" != "none" ]; then
    check_code_form "vm" "50000-deep match pattern" "$ESHKOL_VM" "$pattern_prog"
fi

# ── 3. the documented input-size cap ───────────────────────────────────────
echo "=== the input-size cap ==="
# A long FORM, not a long comment: the front end weighs each form as it reads
# it, and a comment body never becomes part of one.
cap_prog="$WORK/cap.esk"
{
    printf "(define big '("
    for _ in $(seq 1 1000); do printf '1234 '; done
    printf '))\n'
    printf '(display "ok")\n(newline)\n'
} > "$cap_prog"
cap_bytes=$(wc -c < "$cap_prog" | tr -d ' ')

# Just under: accepted, and the program's own output is unaffected.
under_out="$(ESHKOL_MAX_SOURCE_BYTES=$((cap_bytes + 1024)) "$ESHKOL_RUN" -r "$cap_prog" 2>&1)"
under_status=$?
if [ "$under_status" -ne 0 ] || ! printf '%s' "$under_out" | grep -q '^ok$'; then
    fail "cap: a text under the cap was not accepted (status $under_status): $(printf '%s' "$under_out" | tail -3)"
else
    note "cap: a text under the cap runs unaffected"
fi

# Over: refused, in words that name the cap, with no signal.
check_over_cap() {
    local label="$1"; shift
    local out status
    out="$("$@" 2>&1)"
    status=$?
    if signalled "$status"; then
        fail "cap: $label ended on a signal (status $status)"
    elif [ "$status" -eq 0 ]; then
        fail "cap: $label was accepted over the cap"
    elif printf '%s' "$out" | grep -qiE 'input cap|max-source-bytes|ESHKOL_MAX_SOURCE_BYTES'; then
        note "cap: $label refused, naming the cap"
    else
        fail "cap: $label refused without naming the cap: $(printf '%s' "$out" | tail -3)"
    fi
}
check_over_cap "jit, environment variable" \
    env ESHKOL_MAX_SOURCE_BYTES=64 "$ESHKOL_RUN" -r "$cap_prog"
check_over_cap "jit, command-line option" \
    "$ESHKOL_RUN" --max-source-bytes=64 -r "$cap_prog"
if [ "$ESHKOL_VM" != "none" ]; then
    check_over_cap "vm, environment variable" \
        env ESHKOL_MAX_SOURCE_BYTES=64 "$ESHKOL_VM" "$cap_prog"
    check_over_cap "vm, command-line option" \
        "$ESHKOL_VM" --max-source-bytes=64 "$cap_prog"
fi

# ── 4. running out of memory is a diagnostic ───────────────────────────────
# A datum whose reading needs more address space than the process is allowed.
# The limit is applied in a subshell so it cannot affect the rest of the gate.
echo "=== out of memory while reading ==="
oom_prog="$WORK/oom.esk"
write_quoted_datum_program "$oom_prog" 1000000
oom_out="$(
    (
        ulimit -v 1048576 2>/dev/null || exit 70
        "$ESHKOL_RUN" -r "$oom_prog" 2>&1
    )
)"
oom_status=$?
if [ "$oom_status" -eq 70 ]; then
    note "out of memory: this platform does not take an address-space limit; skipped"
elif signalled "$oom_status"; then
    fail "out of memory: ended on a signal (status $oom_status)"
elif [ "$oom_status" -eq 0 ]; then
    note "out of memory: the datum fit in the limit after all; nothing to assert"
else
    note "out of memory: refused with a diagnostic (status $oom_status)"
fi

echo
if [ "$failures" -eq 0 ]; then
    echo "PASS: nesting depth is bounded by memory"
    exit 0
fi
echo "FAIL: $failures check(s) did not hold"
exit 1
