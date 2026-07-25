#!/usr/bin/env bash
# driver_fault_exit_code_test.sh — the eshkol-run driver must never MASK a fault
# with a zero exit. Every fault input yields a NONZERO exit; a missing/unreadable
# entry file also writes NO output artifact. Retro-guards #334 (link failures
# fatal under -r) and pins ESH-0361 (exit-0 masking residue found by the P8
# escape-closure axis-7 fault-injection matrix):
#
#   (a) `eshkol-run MISSING.esk -o out` (AOT) exited 0 AND wrote a ~5 MB binary
#       from an EMPTY compilation unit — the most harmful cell: a build step sees
#       exit 0 + a fresh binary and ships an executable that never contained the
#       user program. Must exit nonzero and write nothing.
#   (b) `eshkol-run -r SYNTAX-ERR.esk` exited 0 after printing the parse error —
#       the -r loop stopped at the ESHKOL_INVALID sentinel (also the EOF marker)
#       without consulting the parser error flag. Must exit nonzero.
#   (c) `eshkol-run -r UNREADABLE.esk` and its AOT form must exit nonzero.
#
# Self-contained: synthesizes its own fixtures; asserts only exit codes and the
# absence of a written binary (no host toolchain assumptions).

set -uo pipefail

ESHKOL_RUN="${1:-${ESHKOL_RUN:-}}"
if [ -z "$ESHKOL_RUN" ]; then
    if [ -x "./build/eshkol-run" ]; then
        ESHKOL_RUN="./build/eshkol-run"
    else
        echo "FAIL: driver_fault_exit_code_test could not locate eshkol-run" >&2
        exit 1
    fi
fi
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "FAIL: driver_fault_exit_code_test eshkol-run is not executable: $ESHKOL_RUN" >&2
    exit 1
fi

tmp="$(mktemp -d)"
trap 'chmod -R u+rwx "$tmp" 2>/dev/null; rm -rf "$tmp"' EXIT
# Isolate the persistent -r JIT cache so a prior run cannot serve a fault input.
export ESHKOL_JIT_CACHE_DIR="$tmp/jit"; mkdir -p "$ESHKOL_JIT_CACHE_DIR"

fail() { echo "FAIL: driver_fault_exit_code_test — $1" >&2; exit 1; }

# assert_nonzero <label> <cmd...> : run cmd, require a nonzero exit.
assert_nonzero() {
    local label="$1"; shift
    "$@" >/dev/null 2>&1
    local ec=$?
    if [ "$ec" -eq 0 ]; then
        fail "$label exited 0 (masked the fault; expected nonzero)"
    fi
    echo "  ok: $label exited $ec (nonzero)"
}

# ── fixtures ────────────────────────────────────────────────────────────────
ok="$tmp/ok.esk";        printf '(display 1)(newline)\n' > "$ok"
malformed="$tmp/mal.esk"; printf '(display 1\n' > "$malformed"          # unbalanced
undef="$tmp/undef.esk";  printf '(display (totally-undefined-fn 3))\n' > "$undef"
noperm="$tmp/noperm.esk"; printf '(display 1)(newline)\n' > "$noperm"; chmod 000 "$noperm"
badreq="$tmp/badreq.esk"; printf '(require nonexistent.module.xyz)(display 1)(newline)\n' > "$badreq"
missing="$tmp/does-not-exist-$$.esk"                                    # never created

# ── (a) AOT missing input: nonzero AND no binary written ────────────────────
out_a="$tmp/out_a.bin"
rm -f "$out_a"
"$ESHKOL_RUN" "$missing" -o "$out_a" >/dev/null 2>&1
ec_a=$?
[ "$ec_a" -ne 0 ] || fail "AOT missing input exited 0 (ESH-0361 cell a)"
[ ! -e "$out_a" ] || fail "AOT missing input wrote a binary $out_a (ESH-0361 cell a — must write nothing)"
echo "  ok: AOT missing input exited $ec_a and wrote no binary"

# ── (b) -r malformed source: nonzero after the parse diagnostic ─────────────
assert_nonzero "-r malformed source (ESH-0361 cell b)" "$ESHKOL_RUN" -r "$malformed"

# ── (c) unreadable input under -r and AOT ───────────────────────────────────
# Some CI runs as root, where mode 000 is still readable; only assert when the
# process genuinely cannot read the file (the fault the cell targets).
if ! { : < "$noperm"; } 2>/dev/null; then
    assert_nonzero "-r unreadable input (ESH-0361 cell c)" "$ESHKOL_RUN" -r "$noperm"
    out_c="$tmp/out_c.bin"; rm -f "$out_c"
    "$ESHKOL_RUN" "$noperm" -o "$out_c" >/dev/null 2>&1
    ec_c=$?
    [ "$ec_c" -ne 0 ] || fail "AOT unreadable input exited 0 (ESH-0361 cell c)"
    [ ! -e "$out_c" ] || fail "AOT unreadable input wrote a binary (ESH-0361 cell c)"
    echo "  ok: AOT unreadable input exited $ec_c and wrote no binary"
else
    echo "  skip: mode-000 file still readable here (running as root?) — cell c not exercisable"
fi

# ── (d)/(e) unresolved (require …): a missing hard dependency is fatal, and
#    the AOT form writes no binary (was: printed "Module not found" then ran and
#    exited 0, cell e compiling a ~5 MB program that silently dropped the module).
assert_nonzero "-r unresolved require (ESH-0361 cell d)" "$ESHKOL_RUN" -r "$badreq"
out_e="$tmp/out_e.bin"; rm -f "$out_e"
"$ESHKOL_RUN" "$badreq" -o "$out_e" >/dev/null 2>&1
ec_e=$?
[ "$ec_e" -ne 0 ] || fail "AOT unresolved require exited 0 (ESH-0361 cell e)"
[ ! -e "$out_e" ] || fail "AOT unresolved require wrote a binary (ESH-0361 cell e)"
echo "  ok: AOT unresolved require exited $ec_e and wrote no binary"

# ── retro-guards: faults that were ALREADY fatal must stay fatal ────────────
assert_nonzero "-r missing input"              "$ESHKOL_RUN" -r "$missing"
assert_nonzero "AOT malformed source (#334)"   "$ESHKOL_RUN" "$malformed" -o "$tmp/mal.bin"
assert_nonzero "-r undefined symbol"           "$ESHKOL_RUN" -r "$undef"
assert_nonzero "AOT undefined symbol"          "$ESHKOL_RUN" "$undef" -o "$tmp/undef.bin"
assert_nonzero "AOT output into missing dir"   "$ESHKOL_RUN" "$ok" -o "/nonexistent-dir-esh0361/out.bin"

echo "PASS: driver_fault_exit_code_test"
exit 0
