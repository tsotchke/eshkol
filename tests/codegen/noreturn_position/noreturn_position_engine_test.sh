#!/usr/bin/env bash
# noreturn_position_engine_test.sh — forms that never return, on every engine.
#
# WHAT THIS HOLDS
#   `raise`, `error`, `exit` and an operand whose static type can never satisfy
#   its operation all lower to a call that does not return. Whatever the
#   enclosing form still has to emit after such a call is dead code, and the
#   program must build and behave identically wherever the call sits: first in
#   a top-level form, under `guard`, after definitions, inside `begin`, as an
#   argument, an initialiser or a test.
#
#   Each row of cases.tsv is one whole program (top-level forms separated by
#   " @@ ", written one per line) with its expected exit status and standard
#   output. The row is run four ways:
#
#     jit    eshkol-run -r prog.esk
#     aot    eshkol-run prog.esk -o prog && ./prog
#     vm     eshkol-run --profile hosted-vm --emit-eskb prog.eskb prog.esk
#            && eshkol-vm-standalone-test prog.eskb
#     repl   the same forms fed to eshkol-repl on standard input
#
#   jit, aot and vm must each produce the expected output and status, so the
#   three agree with each other by construction. On all four, the generated
#   module must be accepted by the IR verifier.
#
# USAGE
#   bash noreturn_position_engine_test.sh <eshkol-run> <eshkol-vm-standalone-test> <eshkol-repl> <work dir>
#   An empty <eshkol-vm-standalone-test> or <eshkol-repl> skips that engine.
set -uo pipefail

ESHKOL_RUN="${1:?eshkol-run path}"
ESHKOL_VM="${2:-}"
ESHKOL_REPL="${3:-}"
WORK="${4:?work directory}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES="$HERE/cases.tsv"
LIBDIR="$(cd "$(dirname "$ESHKOL_RUN")" && pwd)"

mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd)"
cd "$WORK" || exit 1

IR_REJECTED='verification failed|Terminator found|does not dominate|Invalid LLVM module|Failed to generate LLVM IR|fatal signal'

failures=0
rows=0
fail() { echo "FAIL: $*"; failures=$((failures + 1)); }

# transcript <file>: program output with the VM's framing lines removed and
# line breaks dropped, so the three engines are compared on what the program
# itself wrote.
transcript() {
    sed -e 's/=== Execution complete ===//' "$1" \
        | grep -v -e '^\[ESKB\]' -e '^=== Eshkol VM' \
        | tr -d '\n'
}

status_ok() {  # status_ok <expected> <actual>
    if [ "$1" = "nonzero" ]; then [ "$2" -ne 0 ]; else [ "$2" -eq "$1" ]; fi
}

check_engine() {  # check_engine <id> <engine> <expected status> <expected stdout> <actual status>
    local id="$1" engine="$2" want_status="$3" want_out="$4" got_status="$5"
    local got_out
    got_out="$(transcript "$id.$engine.out")"
    if grep -Eq "$IR_REJECTED" "$id.$engine.out" "$id.$engine.err" 2>/dev/null; then
        fail "$id [$engine] generated module was rejected: $(grep -Eh -m1 "$IR_REJECTED" "$id.$engine.out" "$id.$engine.err" | head -1)"
        return
    fi
    if [ "$got_out" != "$want_out" ]; then
        fail "$id [$engine] stdout '$got_out', expected '$want_out'"
    fi
    if ! status_ok "$want_status" "$got_status"; then
        fail "$id [$engine] exit status $got_status, expected $want_status"
    fi
}

while IFS=$'\t' read -r want_status want_out program; do
    case "$want_status" in ''|'#'*) continue ;; esac
    rows=$((rows + 1))
    id="$(printf 'case%03d' "$rows")"
    # One top-level form per line: the REPL reads a form per input line.
    printf '%s\n' "${program// @@ /$'\n'}" > "$id.esk"

    "$ESHKOL_RUN" -r "$id.esk" -L"$LIBDIR" < /dev/null > "$id.jit.out" 2> "$id.jit.err"
    check_engine "$id" jit "$want_status" "$want_out" $?

    if "$ESHKOL_RUN" "$id.esk" -o "$id.bin" -L"$LIBDIR" < /dev/null > "$id.aot.out" 2> "$id.aot.err" && [ -x "$id.bin" ]; then
        "./$id.bin" < /dev/null > "$id.aot.out" 2>> "$id.aot.err"
        check_engine "$id" aot "$want_status" "$want_out" $?
    else
        fail "$id [aot] did not build: $(head -c 300 "$id.aot.err" | tr '\n' ' ')"
    fi

    if [ -n "$ESHKOL_VM" ]; then
        if "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$id.eskb" "$id.esk" -L"$LIBDIR" < /dev/null > "$id.vmc.out" 2>&1 && [ -f "$id.eskb" ]; then
            "$ESHKOL_VM" "$id.eskb" < /dev/null > "$id.vm.out" 2> "$id.vm.err"
            check_engine "$id" vm "$want_status" "$want_out" $?
        else
            fail "$id [vm] did not compile: $(head -c 300 "$id.vmc.out" | tr '\n' ' ')"
        fi
    fi

    if [ -n "$ESHKOL_REPL" ]; then
        # The REPL session survives an unhandled condition and goes on to the
        # next line, so its transcript is compared only for programs that
        # handle theirs; every row must still produce an accepted module.
        "$ESHKOL_REPL" < "$id.esk" > "$id.repl.out" 2> "$id.repl.err"
        if grep -Eq "$IR_REJECTED" "$id.repl.out" "$id.repl.err"; then
            fail "$id [repl] generated module was rejected: $(grep -Eh -m1 "$IR_REJECTED" "$id.repl.out" "$id.repl.err" | head -1)"
        elif [ "$want_status" = "0" ]; then
            repl_out="$(tr -d '\n' < "$id.repl.out")"
            if [ "$repl_out" != "$want_out" ]; then
                fail "$id [repl] stdout '$repl_out', expected '$want_out'"
            fi
        elif ! grep -q 'before|' "$id.repl.out"; then
            fail "$id [repl] did not run the form ahead of the no-return call"
        fi
    fi
done < "$CASES"

echo "$rows programs, $failures failures"
if [ "$failures" -eq 0 ] && [ "$rows" -gt 0 ]; then
    echo "PASS: noreturn_position_engine_test"
    exit 0
fi
echo "FAIL: noreturn_position_engine_test"
exit 1
