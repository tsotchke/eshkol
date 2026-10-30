#!/usr/bin/env bash
# Refinement types (ADR-0044) on every engine: the programs in this directory
# are accepted or rejected identically by the native JIT and by the bytecode
# VM's source path, with the diagnostic each one pins.
#
#   accept.esk           - every obligation proved; output matches accept.expected
#   reject_*.esk         - compilation refused; the first line names the expected
#                          diagnostic (";; expect: <text>")
#   boundary.esk         - a gradual-boundary finding: a warning that still runs,
#                          and a refusal under --strict-types
#
# Usage: refinement_gate.sh <eshkol-run> <vm-standalone|none> <workdir>
set -u
ESHKOL_RUN="${1:?usage: $0 <eshkol-run> <vm-standalone|none> <workdir>}"
VM="${2:?usage}"
WORKDIR="${3:?usage}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$WORKDIR" || { echo "FAIL: cannot create $WORKDIR"; exit 1; }
export ESHKOL_JIT_CACHE=0

fail=0
checks=0
note() { checks=$((checks + 1)); }
bad() { echo "FAIL: $*"; fail=$((fail + 1)); }

engines=("native")
if [ "$VM" != "none" ] && [ -x "$VM" ]; then engines+=("vm"); fi

run() {  # engine file [flags...]
    local engine="$1" file="$2"; shift 2
    if [ "$engine" = native ]; then "$ESHKOL_RUN" "$@" -r "$file" 2>&1
    else "$VM" "$file" 2>&1; fi
}

for engine in "${engines[@]}"; do
    err="$WORKDIR/accept.$engine.err"
    if [ "$engine" = native ]; then out=$("$ESHKOL_RUN" -r "$HERE/accept.esk" 2>"$err"); rc=$?
    else "$VM" "$HERE/accept.esk" > "$WORKDIR/accept.vm.out" 2>"$err"; rc=$?
         out=$(grep -v '^=== ' "$WORKDIR/accept.vm.out"); fi
    note
    if [ $rc -ne 0 ]; then bad "[$engine] accept.esk exited $rc: $out $(cat "$err")"
    elif ! diff <(printf '%s\n' "$out") "$HERE/accept.expected" > "$WORKDIR/accept.$engine.diff"; then
        bad "[$engine] accept.esk output differs: $(cat "$WORKDIR/accept.$engine.diff")"
    elif grep -qE "Refinement error|refinement (not proved|violated|undecided)" "$err"; then
        bad "[$engine] accept.esk reported a refinement finding: $(grep -E 'Refinement error|refinement (not|viol|undec)' "$err")"
    fi

    for f in "$HERE"/reject_*.esk; do
        want=$(sed -n '1s/^;; expect: //p' "$f")
        out=$(run "$engine" "$f"); rc=$?
        note
        if [ $rc -eq 0 ]; then bad "[$engine] $(basename "$f") was accepted"
        elif ! printf '%s' "$out" | grep -qF -- "$want"; then
            bad "[$engine] $(basename "$f") lacks '$want': $out"
        fi
    done
done

# Gradual boundary: native only (the VM draws only the fatal verdict).
out=$(run native "$HERE/boundary.esk"); rc=$?
note
if [ $rc -ne 0 ] || ! printf '%s' "$out" | grep -q "^2$"; then bad "boundary.esk did not run: $out"
elif ! printf '%s' "$out" | grep -q "not proved at a gradual boundary"; then bad "boundary.esk gave no boundary finding: $out"
fi
out=$(run native "$HERE/boundary.esk" --strict-types); rc=$?
note
if [ $rc -eq 0 ]; then bad "boundary.esk was accepted under --strict-types"; fi

if [ $fail -ne 0 ]; then echo "FAIL: $fail of $checks refinement checks"; exit 1; fi
echo "PASS: refinement types on ${engines[*]} ($checks checks)"
