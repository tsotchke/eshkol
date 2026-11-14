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
#   three agree with each other by construction. The REPL column states what
#   an interactive session does with the same forms: the same transcript when
#   the program handles its conditions, the reported diagnostic when it does
#   not, the exit status when it exits. On all four, and in the VM compile
#   step, the generated module must be accepted by the IR checks.
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

IR_REJECTED='verification failed|Terminator found|emitted after .unreachable.|does not dominate|Invalid LLVM module|Failed to generate LLVM IR|fatal signal'

failures=0
rows=0
fail() { echo "FAIL: $*"; failures=$((failures + 1)); }

# transcript <file>: program output with the VM's framing lines removed and
# line breaks dropped, so the three engines are compared on what the program
# itself wrote.
transcript() {
    sed -E -e 's/=== Execution complete ===//' \
           -e 's/[^ |]*\.esk:[0-9]+:[0-9]+: //g' "$1" \
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

while IFS=$'\t' read -r want_status want_out want_repl program; do
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
            if grep -Eq "$IR_REJECTED" "$id.vmc.out"; then
                fail "$id [vm] compile step reported a rejected module: $(grep -Eh -m1 "$IR_REJECTED" "$id.vmc.out" | head -1)"
            fi
            "$ESHKOL_VM" "$id.eskb" < /dev/null > "$id.vm.out" 2> "$id.vm.err"
            check_engine "$id" vm "$want_status" "$want_out" $?
        else
            fail "$id [vm] did not compile: $(head -c 300 "$id.vmc.out" | tr '\n' ' ')"
        fi
    fi

    if [ -n "$ESHKOL_REPL" ]; then
        # A REPL session reports an unhandled condition and goes on to the
        # next line; `exit` ends it with the program's status.
        "$ESHKOL_REPL" < "$id.esk" > "$id.repl.out" 2> "$id.repl.err"
        repl_status=$?
        repl_out="$(transcript "$id.repl.out")"
        if grep -Eq "$IR_REJECTED" "$id.repl.out" "$id.repl.err"; then
            fail "$id [repl] generated module was rejected: $(grep -Eh -m1 "$IR_REJECTED" "$id.repl.out" "$id.repl.err" | head -1)"
        elif [ "$want_repl" = "-" ]; then
            if [ "$repl_out" != "$want_out" ]; then
                fail "$id [repl] stdout '$repl_out', expected '$want_out'"
            fi
        elif [ "$want_repl" = "exit" ]; then
            if [ "$repl_out" != "$want_out" ]; then
                fail "$id [repl] stdout '$repl_out', expected '$want_out'"
            fi
            if ! status_ok "$want_status" "$repl_status"; then
                fail "$id [repl] exit status $repl_status, expected $want_status"
            fi
        else
            if ! grep -Fq -- "$want_repl" "$id.repl.err" "$id.repl.out"; then
                fail "$id [repl] no diagnostic '$want_repl' (stderr: $(head -c 200 "$id.repl.err" | tr '\n' ' '))"
            fi
            case "$repl_out" in
                "$want_out"*) : ;;
                *) fail "$id [repl] stdout '$repl_out' does not begin with '$want_out'" ;;
            esac
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
