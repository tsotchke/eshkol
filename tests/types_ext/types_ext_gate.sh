#!/usr/bin/env bash
# Structural types gate (ADR-0046): row-polymorphic records, arbitrary-rank
# types and session types, on every engine a fixture names.
#
# Each fixture's header declares:
#   ;; ENGINES: jit aot vm           engines that must run (or must refuse) it
#   ;; EXPECT-OUTPUT: <line>         (positive fixtures) stdout, line by line
#   ;; EXPECT-SCHEME: <name : type>  (positive) the principal type the checker
#                                    reports under ESHKOL_DUMP_STRUCTURAL_TYPES=1
#   ;; EXPECT-STDERR: <text>         (reject/ fixtures) the diagnostic
#
# Positive fixtures must compile, run and print exactly the expected lines.
# Reject fixtures must be refused: a nonzero compile exit with the diagnostic
# and NO binary on the native engine, and a refusal with no output on the VM.
#
# Usage: types_ext_gate.sh <eshkol-run> <vm-standalone|none> <workdir> <lib-dir>
set -u
ESHKOL_RUN="${1:?usage}"; VM="${2:?usage}"; WORK="${3:?usage}"; LIBDIR="${4:?usage}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$WORK" || { echo "FAIL: cannot create $WORK"; exit 1; }
export ESHKOL_VM_NO_DISASM=1
PASS=0; FAILED=0
fail() { echo "FAIL: $1"; FAILED=$((FAILED + 1)); }
engines_of() { sed -n 's/^;; ENGINES: *//p' "$1" | head -1; }
has_engine() { case " $(engines_of "$1") " in *" $2 "*) return 0 ;; esac; return 1; }

check_output() {   # fixture actual-output-file label
    local want="$WORK/want.txt"
    sed -n 's/^;; EXPECT-OUTPUT: //p' "$1" > "$want"
    if ! diff -u "$want" "$2" > "$WORK/diff.txt"; then
        fail "$(basename "$1") [$3]: output differs: $(head -c 600 "$WORK/diff.txt")"
        return 1
    fi
    return 0
}

for f in "$HERE"/*_ok.esk; do
    name=$(basename "$f" .esk); ok=1
    bin="$WORK/$name.bin"; rm -f "$bin"
    ESHKOL_DUMP_STRUCTURAL_TYPES=1 "$ESHKOL_RUN" "$f" -L "$LIBDIR" -o "$bin" \
        > "$WORK/$name.cout" 2> "$WORK/$name.cerr"; rc=$?
    if [ $rc -ne 0 ] || [ ! -x "$bin" ]; then
        fail "$name: compile failed (exit $rc): $(grep -m3 -i 'error' "$WORK/$name.cerr")"; continue
    fi
    while IFS= read -r scheme; do
        [ -z "$scheme" ] && continue
        grep -qxF "[types] $scheme" "$WORK/$name.cerr" || {
            fail "$name: principal type not reported: $scheme (got: $(grep -m1 "\[types\] ${scheme%% :*} :" "$WORK/$name.cerr"))"; ok=0; }
    done < <(sed -n 's/^;; EXPECT-SCHEME: //p' "$f")
    if has_engine "$f" aot; then
        timeout 120 "$bin" > "$WORK/$name.aot" 2> "$WORK/$name.aot.err" || { fail "$name [aot]: exit $?"; ok=0; }
        check_output "$f" "$WORK/$name.aot" aot || ok=0
    fi
    if has_engine "$f" jit; then
        timeout 300 "$ESHKOL_RUN" -r "$f" -L "$LIBDIR" > "$WORK/$name.jit" 2> "$WORK/$name.jit.err" || { fail "$name [jit]: exit $?"; ok=0; }
        check_output "$f" "$WORK/$name.jit" jit || ok=0
    fi
    if has_engine "$f" vm && [ "$VM" != none ] && [ -x "$VM" ]; then
        (cd "$(dirname "$LIBDIR")" && timeout 300 "$VM" "$f") > "$WORK/$name.vm" 2> "$WORK/$name.vm.err" || { fail "$name [vm]: exit $?"; ok=0; }
        check_output "$f" "$WORK/$name.vm" vm || ok=0
    fi
    [ $ok -eq 1 ] && { PASS=$((PASS + 1)); echo "ok   $name"; }
done

for f in "$HERE"/reject/*.esk; do
    name=$(basename "$f" .esk); ok=1
    want=$(sed -n 's/^;; EXPECT-STDERR: //p' "$f" | head -1)
    bin="$WORK/$name.bin"; rm -f "$bin"
    "$ESHKOL_RUN" "$f" -L "$LIBDIR" -o "$bin" > "$WORK/$name.cout" 2> "$WORK/$name.cerr"; rc=$?
    if [ $rc -eq 0 ]; then fail "$name: compiled (exit 0) but must be refused"; ok=0; fi
    if [ -e "$bin" ]; then fail "$name: a binary was written for a refused program"; ok=0; fi
    grep -qF -- "$want" "$WORK/$name.cerr" || { fail "$name: missing diagnostic '$want': $(grep -m2 -i error "$WORK/$name.cerr")"; ok=0; }
    if has_engine "$f" vm && [ "$VM" != none ] && [ -x "$VM" ]; then
        (cd "$(dirname "$LIBDIR")" && timeout 120 "$VM" "$f") > "$WORK/$name.vm" 2> "$WORK/$name.vm.err"; vrc=$?
        [ $vrc -ne 0 ] || { fail "$name [vm]: the VM ran a program the checker refuses"; ok=0; }
        grep -qF -- "$want" "$WORK/$name.vm.err" || { fail "$name [vm]: missing diagnostic '$want'"; ok=0; }
    fi
    [ $ok -eq 1 ] && { PASS=$((PASS + 1)); echo "ok   $name (refused)"; }
done

echo "types_ext: $PASS passed, $FAILED failed"
[ $FAILED -eq 0 ] && echo "PASS: structural types gate" && exit 0
exit 1
