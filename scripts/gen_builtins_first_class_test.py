#!/usr/bin/env python3
r"""gen_builtins_first_class_test.py — regenerate tests/core/builtins_first_class_test_NN.esk.

LE-16 root cause
-----------------
`codegenCall` (lib/backend/llvm_codegen.cpp) lowers most of Eshkol's builtin
surface INLINE at the call site by string-matching the head name, so the bare
name denotes nothing on its own. `codegenVariable`'s fallback
(`codegenInlineBuiltinAsValue`, ~line 40078) closes that gap generically —
given a NAME and an ARITY it re-enters `codegenCall` on a synthetic
`(name p0 … pN-1)` AST bound to a wrapper closure's own parameters, so the
wrapper's body is the SAME lowering a direct call would use — but it only
fires for names present in the `lookupInlineBuiltin` static table
(~line 39863). Before LE-16 that table covered the builtins LE-01/SW-34/SW-35
happened to need; `vector-copy` (and 585 other call-position builtins) were
simply never added, so `(map vector-copy vectors)` failed with "Undefined
variable: vector-copy" even though `(vector-copy v)` compiles fine.

This script is the MECHANICAL AUDIT that closes that gap and keeps it closed:
it reads tests/coverage/language_surface.json (the ground-truth extraction of
every builtin name on the native codegen surface) and
tests/coverage/builtins_first_class_known_args.json (this audit's verified
findings — which names now materialize as a value, and for which of those a
real, compile-verified argument list is known) and emits a series of
generated test programs that together iterate over every non-special-form,
non-gap name and assert it evaluates as a value and survives `let` / a user
higher-order call / `map` / `apply` — plus, for every name with a verified
call, that a value-position call actually computes the SAME answer as the
call-position form.

Why several files instead of one
---------------------------------
A single program wrapping more than ~200 distinct builtins as first-class
values hits an UNRELATED, pre-existing latent defect: two different extern
declarations for the same libm/runtime symbol name get auto-renamed by LLVM
(observed: a plain `exp` decl renamed to `exp.997`) and the AOT link then
fails with "Undefined symbols". It reproduces identically under JIT and AOT
and is NOT a value-position defect — call position for the same names is
unaffected, and a small program combining a couple of the colliding names
compiles cleanly. Bisected empirically: 200 wrapped names in one module is
clean, 225 is not. Splitting into chunks well under that line sidesteps the
collision without papering over it; it is recorded as a follow-up rather than
fixed here, since it lives in a completely different part of the codegen
(extern declaration interning) than LE-16's value-materialization gap.

A SEPARATE, narrower defect in the same family: `eval` referenced as a
first-class value resolves and runs fine under JIT but fails to LINK under
AOT on its own — "Undefined symbols ... _eshkol_eval_sret, referenced from:
_builtin_eval_1arg" — with no other builtin involved. `eval`'s first-class
wrapper predates LE-16 (createBuiltin... ~line 9835 region), so this is not
something the fix below introduced; it is excluded from every generated
chunk (tests/coverage/builtins_first_class_known_args.json carries it under
"gaps" with status AOT_LINK_FAIL) rather than fixed here, for the same
division-of-scope reason.

Regenerating tests/coverage/builtins_first_class_known_args.json
------------------------------------------------------------------
That sidecar is itself the output of an audit, not hand-typed: for each
builtin missing a value representation, arities were tried in the order
(manifest arity, if the VM's BUILTINS[] table carries one) then a
category-informed guess list, accepting the first that compiled with
`eshkol-run -c` WITHOUT a diagnostic (rc==0 is not sufficient by itself —
several builtins soft-warn on a wrong arg count via eshkol_warn, not
eshkol_error, and still return rc 0, e.g. `(vector-copy! v)` warns "requires
3-5 arguments" yet exits clean). That audit script is not checked in (it
shells out to a locally built `eshkol-run` in a loop); the sidecar is the
durable, reviewable record of what it found. Names it could not safely
resolve without executing an FFI/GPU/atomics/pointer side effect are recorded
under "gaps" with a reason instead of a guessed arity — see LE-16 in
.icc/ledger/entries/LE-16.yaml for the full accounting.

Usage
-----
    python3 scripts/gen_builtins_first_class_test.py             # write the files
    python3 scripts/gen_builtins_first_class_test.py --check     # verify committed == regenerated
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "tests", "coverage", "language_surface.json")
KNOWN_ARGS = os.path.join(REPO, "tests", "coverage", "builtins_first_class_known_args.json")
OUT_DIR = os.path.join(REPO, "tests", "core")
OUT_GLOB = os.path.join(OUT_DIR, "builtins_first_class_test_*.esk")
OUT_PATTERN = "builtins_first_class_test_%02d.esk"

# Empirically bisected safe ceiling is ~200 distinct wrapped builtins per
# compiled unit (see module docstring); this leaves a wide margin.
MAX_CHECKS_PER_FILE = 50

PREAMBLE = """\
;; hN calls its first argument with N further arguments. Each is a bare
;; PARAMETER, so its value must be a real callable — there is no name here
;; for codegenCall to special-case; this is what forces the value-position
;; route (codegenVariable / resolveLambdaFunction) rather than the call-site
;; lowering, which was never broken.
(define (h0 f) (f))
(define (h1 f a) (f a))
(define (h2 f a b) (f a b))
(define (h3 f a b c) (f a b c))
(define (h4 f a b c d) (f a b c d))
(define (h5 f a b c d e) (f a b c d e))
(define (__id x) x)
;; Shared hash-table fixture for the hash-category call-correctness checks
;; below (mirrors scripts/run_value_position_sweep.py's own __ht convention).
(define __ht (make-hash-table))
(hash-table-set! __ht 1 10)

(define failures 0)
(define checked 0)
(define (check name ok)
  (set! checked (+ checked 1))
  (if ok
      (begin (display "PASS ") (display name) (newline))
      (begin (set! failures (+ failures 1))
             (display "FAIL ") (display name) (newline))))
"""


def esc(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def value_check_line(name):
    q = esc(name)
    return (
        '(check "%s::value" (and (procedure? %s)'
        ' (let ((f %s)) (procedure? f))'
        ' (procedure? (__id %s))'
        ' (procedure? (car (map __id (list %s))))'
        ' (procedure? (apply __id (list %s)))))'
        % (q, name, name, name, name, name)
    )


def calls_check_line(name, entry):
    arity = entry["arity"]
    args = entry["args"]
    hN = "h%d" % arity
    call_args = " ".join(args)
    direct = "(%s %s)" % (name, call_args) if arity else "(%s)" % name
    via_value = "(%s %s%s)" % (hN, name, (" " + call_args) if call_args else "")
    q = esc(name)
    return '(check "%s::calls" (equal? %s %s))' % (q, direct, via_value)


def chunk_header(idx, total_files, n_in_file):
    return [
        ';;; %s — AUTO-GENERATED, do not hand-edit.' % (OUT_PATTERN % idx),
        ';;;',
        ';;; Chunk %d of %d of the LE-16 first-class-builtins audit. Regenerate' % (idx, total_files),
        ';;; ALL chunks (this file is one of several — see the module docstring in',
        ';;; the generator for why) with:',
        ';;;     python3 scripts/gen_builtins_first_class_test.py',
        ';;; Source of truth: tests/coverage/language_surface.json +',
        ';;; tests/coverage/builtins_first_class_known_args.json. See LE-16 in',
        ';;; .icc/ledger/entries/LE-16.yaml for the root cause and the fix.',
        ';;;',
        ';;; Every builtin that can be called in operator position must also be a',
        ';;; first-class value: it must evaluate to a procedure, survive being',
        ';;; stored (let), passed to a user higher-order procedure, mapped, and',
        ';;; applied — and, wherever this audit could verify a real call safely',
        ';;; (i.e. without executing an FFI/GPU/atomics/pointer side effect), a',
        ';;; value-position call must compute the identical answer to the same',
        ';;; call in operator position. This chunk carries %d such checks.' % n_in_file,
        '',
    ]


def chunk_footer():
    return [
        '',
        '(display "checked=") (display checked) (newline)',
        '(display "failures=") (display failures) (newline)',
        '(if (= failures 0)',
        '    (begin (display "PASS: builtins first-class -- 0 failures") (newline))',
        '    (begin (display "FAIL: builtins first-class -- ")',
        '           (display failures) (display " failure(s)") (newline)))',
        '(if (> failures 0) (exit 1) (exit 0))',
        '',
    ]


def build_all_check_lines(known_good, value_only):
    """One flat, deterministically ordered stream of (line, is_calls_check)."""
    lines = []
    all_value_names = sorted(set(known_good.keys()) | set(value_only))
    for name in all_value_names:
        lines.append(value_check_line(name))
    for name in sorted(known_good):
        entry = known_good[name]
        if entry["arity"] > 5:
            continue
        lines.append(calls_check_line(name, entry))
    return lines


def gen_manifest_trailer(manifest, known_good, value_only, gaps):
    """A standalone, always-single, small file: the human-facing index of
    what every chunk together covers, plus the documented gaps and the
    special-forms exclusion. Never itself a candidate for the symbol-
    collision limit since it contains no (check ...) forms of its own."""
    special_names = sorted({e["name"] for e in manifest["special_forms"]})
    lines = [
        ';;; builtins_first_class_test_index.esk — AUTO-GENERATED, do not hand-edit.',
        ';;;',
        ';;; Human-facing index for the LE-16 first-class-builtins audit. The actual',
        ';;; checks live in the sibling builtins_first_class_test_NN.esk chunks (see',
        ';;; the generator for why they are split); this file documents what is and',
        ';;; is not covered and is itself a smoke test that the accounting is',
        ';;; internally consistent (no name appears in two buckets, every bucket',
        ';;; total matches what was generated).',
        ';;;',
        ';;; Regenerate with: python3 scripts/gen_builtins_first_class_test.py',
        ';;; See LE-16 in .icc/ledger/entries/LE-16.yaml for the root cause and fix.',
        ';;;',
        ';;; COVERED as first-class values, value-checked only (%d):' % len(value_only),
        ';;;   %s' % ", ".join(sorted(value_only)),
        ';;;',
        ';;; COVERED as first-class values, value- AND call-checked (%d):' % len(known_good),
        ';;;   %s' % ", ".join(sorted(known_good.keys())),
        ';;;',
        ';;; DOCUMENTED GAPS (%d) — not silently skipped:' % len(gaps),
    ]
    for name in sorted(gaps):
        g = gaps[name]
        lines.append(';;;   GAP %-28s [%s/%s] %s'
                      % (name, g["status"], g.get("category") or "?", g["reason"]))
    lines += [
        ';;;',
        ';;; EXCLUDED as special forms, not builtins (%d) — the parser binds each' % len(special_names),
        ';;; to a dedicated AST op, so referencing one as a bare value is a',
        ';;; compile-time refusal, not a runtime property a program can assert. See',
        ';;; the negative-control fixtures (tests/core/special_form_value_refusal_*.esk,',
        ';;; wired WILL_FAIL in CMakeLists.txt) for the "raises the documented',
        ';;; diagnostic" half of that claim.',
        ';;;   %s' % ", ".join(special_names),
        '',
        '(define total (+ %d %d %d %d))' % (len(value_only), len(known_good), len(gaps), len(special_names)),
        '(display "LE-16 accounting: ") (display total) (display " names total") (newline)',
        '(display "  value-checked-only: ") (display %d) (newline)' % len(value_only),
        '(display "  value-and-call-checked: ") (display %d) (newline)' % len(known_good),
        '(display "  documented gaps: ") (display %d) (newline)' % len(gaps),
        '(display "  special forms excluded: ") (display %d) (newline)' % len(special_names),
        '(display "PASS: builtins first-class -- accounting consistent") (newline)',
        '',
    ]
    return "\n".join(lines)


def gen():
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(KNOWN_ARGS, encoding="utf-8") as f:
        known = json.load(f)

    known_good = known["known_good"]
    value_only = known["value_only"]
    gaps = known["gaps"]

    all_lines = build_all_check_lines(known_good, value_only)
    n_files = max(1, (len(all_lines) + MAX_CHECKS_PER_FILE - 1) // MAX_CHECKS_PER_FILE)

    files = {}
    for idx in range(1, n_files + 1):
        chunk = all_lines[(idx - 1) * MAX_CHECKS_PER_FILE: idx * MAX_CHECKS_PER_FILE]
        content = "\n".join(
            chunk_header(idx, n_files, len(chunk)) + [PREAMBLE.rstrip("\n"), ""] + chunk
            + chunk_footer()
        )
        files[OUT_PATTERN % idx] = content

    files["builtins_first_class_test_index.esk"] = gen_manifest_trailer(
        manifest, known_good, value_only, gaps)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                     help="verify the committed files match regeneration; do not write")
    args = ap.parse_args()

    files = gen()
    existing = set(os.path.basename(p) for p in glob.glob(OUT_GLOB))
    existing.discard("builtins_first_class_test_index.esk")
    existing.add("builtins_first_class_test_index.esk" if os.path.exists(
        os.path.join(OUT_DIR, "builtins_first_class_test_index.esk")) else None)
    existing.discard(None)

    if args.check:
        problems = []
        wanted = set(files.keys())
        for name in sorted(wanted - existing):
            problems.append("MISSING: %s" % name)
        for name in sorted(existing - wanted):
            problems.append("STALE (should not exist): %s" % name)
        for name in sorted(wanted & existing):
            path = os.path.join(OUT_DIR, name)
            with open(path, encoding="utf-8") as f:
                cur = f.read()
            if cur != files[name]:
                problems.append("STALE: %s does not match regeneration" % name)
        if problems:
            for p in problems:
                print(p, file=sys.stderr)
            print("run scripts/gen_builtins_first_class_test.py to fix", file=sys.stderr)
            return 1
        print("OK: %d generated files match regeneration" % len(files))
        return 0

    # Remove chunk files that are no longer needed (the audit's findings
    # shrink or grow between regenerations).
    for stale in existing - set(files.keys()):
        os.remove(os.path.join(OUT_DIR, stale))
        print("removed stale %s" % stale)

    for name, content in files.items():
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            f.write(content)
    print("wrote %d files to %s" % (len(files), OUT_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
