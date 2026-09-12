#!/usr/bin/env python3
r"""five_way_surface.py — P8 escape-closure axis 6: five-way surface-agreement
gate (static).

Originating escape (see .swarm/P8_ESCAPE_ANALYSIS.md): a low-level AD builtin
(ad-pow / ad-tape-length) was documented and dispatched on ONE backend but not
registered on the other. No gate cross-checked a builtin's presence across all
the places it must agree, so the asymmetry was invisible until a differential
test happened to exercise it.

For every builtin the project documents or exports, this gate cross-checks FIVE
independent surfaces and reports any DISAGREEMENT:
  1. doc mention        docs/reference/stdlib/*.md  (### `(name ...` headers)
  2. manifest entry     tests/coverage/language_surface.json builtins/prelude/
                        special_forms
  3. native registration  manifest backends intersect {native, native_llvm},
                        or any other vehicle the native engine reaches the
                        name by: a declared special form (parser + codegen),
                        the compiled-in prelude, a core/stdlib module
                        definition, or — via a row's `mirrors` annotation —
                        the public construct this row is one engine's private
                        spelling of. See native_reaches() below: reading the
                        builtin tables alone reports operators as absent from
                        an engine that runs them.
  4. VM dispatch          manifest backends intersect {vm}
  5. module provide list  (provide name ...) across lib/**/*.esk, with a
                        matching (define name .../(define (name ...) somewhere

Disagreement classes (key = "class::name"):
  doc_orphan                    documented but in NEITHER the manifest NOR any
                                provide list — a doc referencing a builtin that
                                no longer exists / was renamed.
  native_missing                a manifest builtin dispatched on the VM but not
                                registered natively (the ad-pow class).
  vm_missing                    a manifest builtin registered natively but not
                                dispatched on the VM.
  provide_orphan                a name in a (provide ...) list with no visible
                                definition and no manifest entry (export drift).

The gate is a shrink-only ratchet against
tests/escape_matrix/five_way_baseline.json: every disagreement that exists today
is a legitimate, grandfathered gap; the gate fails only on a NEW key not in the
baseline. A key that no longer disagrees may be dropped from the baseline
(shrink-only), never silently added.

Usage:
  five_way_surface.py [--baseline FILE] [--update-baseline] [--trace FILE]
                      [--repo-root DIR] [--report]
Exit 0 iff no NEW disagreement.
"""

import argparse
import glob
import json
import os
import re
import sys

DOC_HDR = re.compile(r"^#+\s*`\(([a-z][a-zA-Z0-9!?*+<>=./_%-]*)")
PROVIDE = re.compile(r"\(provide\s+([^)]*)\)", re.S)
# Public module bindings include ordinary defines, uppercase/asterisk
# constants, and FFI declarations. Keep the token grammar aligned with the
# reader instead of silently dropping valid names because they do not begin
# with a lowercase letter.
VERBATIM_NAME = r"\|(?:\\.|[^|])*\|"
NAME = r"((?:[A-Za-z_*][A-Za-z0-9!?*+<>=./_%-]*|" + VERBATIM_NAME + r"))"
DEFINE = re.compile(r"\(define(?:-syntax)?\s+\(?\s*" + NAME)
EXTERN = re.compile(r"\(extern\s+\S+\s+" + NAME)


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def strip_scheme_comments(text):
    """Remove Scheme line comments without changing string contents.

    The module collector is a source-surface check, not a prose search. A
    `provide` example in a comment must not create an exported name, and a
    semicolon inside a string must remain data. Keeping this normalization at
    the collector boundary makes all five surfaces use the same source view.
    """
    out = []
    in_string = False
    escaped = False
    in_comment = False
    in_verbatim = False
    for ch in text:
        if in_comment:
            if ch == "\n":
                in_comment = False
                out.append(ch)
            continue
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if in_verbatim:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "|":
                in_verbatim = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
        elif ch == "|":
            in_verbatim = True
            out.append(ch)
        elif ch == ";":
            in_comment = True
        else:
            out.append(ch)
    return "".join(out)


def scheme_tokens(text):
    """Split a comment-free Scheme surface while preserving |...| names."""
    return re.findall(r"\|(?:\\.|[^|])*\||[^\s]+", text)


def collect_docs(root):
    names = set()
    for md in glob.glob(os.path.join(root, "docs/reference/stdlib/*.md")):
        for ln in read(md).splitlines():
            m = DOC_HDR.match(ln)
            if m:
                names.add(m.group(1))
    return names


def collect_manifest(root):
    d = json.load(open(os.path.join(root, "tests/coverage/language_surface.json")))
    backends = {}
    mirrors = {}
    present = set()
    for e in d.get("builtins", []):
        present.add(e["name"])
        backends[e["name"]] = set(e.get("backends") or [])
        if e.get("mirrors"):
            mirrors[e["name"]] = e["mirrors"]
    # The native engine reaches a special form through the parser and codegen,
    # not through a builtin dispatch table, so a construct declared as syntax
    # has a native surface no builtin table can witness. Recorded separately
    # from `present` because native availability, not mere presence, is what
    # the backend-asymmetry check needs.
    syntax = {e["name"] if isinstance(e, dict) else e
              for e in d.get("special_forms", [])}
    prelude = {e["name"] if isinstance(e, dict) else e
               for e in d.get("prelude", [])}
    present |= syntax | prelude
    return present, backends, mirrors, syntax, prelude


def collect_modules(root):
    provided = set()
    defined = set()
    for esk in glob.glob(os.path.join(root, "lib/**/*.esk"), recursive=True):
        txt = strip_scheme_comments(read(esk))
        for m in PROVIDE.finditer(txt):
            for tok in scheme_tokens(m.group(1)):
                tok = tok.strip()
                if tok and not tok.startswith(";"):
                    provided.add(tok)
        for m in DEFINE.finditer(txt):
            defined.add(m.group(1))
        for m in EXTERN.finditer(txt):
            defined.add(m.group(1))
    return provided, defined


def compute_disagreements(root):
    docs = collect_docs(root)
    manifest, backends, mirrors, syntax, prelude = collect_manifest(root)
    provided, defined = collect_modules(root)
    return cross_check(docs, manifest, backends, mirrors, syntax, prelude,
                       provided, defined)


def cross_check(docs, manifest, backends, mirrors, syntax, prelude,
                provided, defined):
    """The five-way comparison itself, over already-collected surfaces.

    Separated from collection so self_test() can drive it with synthetic
    surfaces and assert the resolution rules directly, rather than inferring
    them from whatever the repository happens to contain today.
    """

    def native_reaches(nm, bk=None):
        """Can a program running on the NATIVE engine call `nm`?

        Four vehicles, all first-class — a builtin dispatch table is only one
        of them, and reading it alone is what made core AD operators look
        absent from an engine that runs them:
          * the native/AOT builtin tables;
          * the parser + codegen syntax path (a declared special form);
          * the Scheme prelude compiled into every program;
          * a core/stdlib module definition (`defined`), which is how a VM
            opcode that mirrors a module function is reached natively.
        """
        if bk is None:
            bk = backends.get(nm, set())
        return (bool(bk & {"native", "native_llvm"}) or nm in defined
                or nm in syntax or nm in prelude)

    dis = set()
    # doc_orphan: documented but nowhere implemented/exported.
    for nm in docs:
        if nm not in manifest and nm not in provided and nm not in defined:
            dis.add("doc_orphan::" + nm)
    # backend asymmetry over manifest builtins.
    for nm, bk in backends.items():
        # A VM opcode may intentionally mirror a native stdlib/prelude
        # definition rather than a native compiler builtin.  Treat that module
        # definition as native availability; otherwise promoting compound list
        # accessors to first-class VM builtins manufactures native_missing
        # findings even though eshkol-run executes the same public names.
        #
        # `mirrors` carries that same relation for the rows whose two spellings
        # differ: a VM-private arity split, or a lower-level handle form, of a
        # public construct the native engine reaches by another vehicle. The
        # annotation is not taken on trust — the public name it names has to be
        # natively reachable in its own right, so annotating a genuine gap
        # leaves the disagreement standing.
        has_native = native_reaches(nm, bk)
        if not has_native and nm in mirrors:
            has_native = native_reaches(mirrors[nm])
        has_vm = "vm" in bk
        # agent_ffi-only builtins are intentionally native-only host bridges.
        if bk == {"agent_ffi"}:
            continue
        if has_vm and not has_native:
            dis.add("native_missing::" + nm)
        if has_native and not has_vm:
            dis.add("vm_missing::" + nm)
    # provide_orphan: exported but not defined and not a manifest builtin.
    for nm in provided:
        if nm not in defined and nm not in manifest:
            dis.add("provide_orphan::" + nm)
    return sorted(dis), {"docs": len(docs), "manifest": len(manifest),
                         "provided": len(provided), "defined": len(defined)}


def self_test():
    """Assert the native-reachability rules on synthetic surfaces.

    The rules this pins down are the ones a future edit could quietly weaken
    into a rubber stamp: a `mirrors` annotation must BUY nothing unless the
    public name it points at is itself natively reachable, and the vehicles
    other than the builtin tables (syntax, prelude, module definition) must
    each count on their own.
    """
    def run(**kw):
        args = dict(docs=set(), manifest=set(), backends={}, mirrors={},
                    syntax=set(), prelude=set(), provided=set(), defined=set())
        args.update(kw)
        return set(cross_check(**args)[0])

    vm_only = {"widget": {"vm"}}
    cases = [
        ("a vm-only builtin with no native vehicle disagrees",
         run(backends=vm_only), {"native_missing::widget"}),
        ("a declared special form is natively reachable",
         run(backends=vm_only, syntax={"widget"}), set()),
        ("a prelude definition is natively reachable",
         run(backends=vm_only, prelude={"widget"}), set()),
        ("a module definition is natively reachable",
         run(backends=vm_only, defined={"widget"}), set()),
        ("mirrors resolves to a natively reachable public name",
         run(backends=vm_only, mirrors={"widget": "gadget"},
             defined={"gadget"}), set()),
        ("mirrors at a name the native engine also lacks resolves nothing",
         run(backends=vm_only, mirrors={"widget": "gadget"}),
         {"native_missing::widget"}),
        ("mirrors at a VM-only name resolves nothing",
         run(backends={"widget": {"vm"}, "gadget": {"vm"}},
             mirrors={"widget": "gadget"}),
         {"native_missing::widget", "native_missing::gadget"}),
        ("a native-only builtin still reports the VM side",
         run(backends={"widget": {"native"}}), {"vm_missing::widget"}),
    ]
    failures = 0
    for label, got, want in cases:
        if got != want:
            failures += 1
            print("FAIL self-test: %s\n      got  %s\n      want %s"
                  % (label, sorted(got), sorted(want)))
        else:
            print("ok  %s" % label)
    if failures:
        print("five-way self-test: %d FAILED" % failures)
        return 1
    print("five-way self-test: %d checks OK" % len(cases))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--baseline",
                    default="tests/escape_matrix/five_way_baseline.json")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--trace")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="check the resolution rules on synthetic surfaces "
                         "and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    dis, counts = compute_disagreements(args.repo_root)
    baseline_path = os.path.join(args.repo_root, args.baseline) \
        if not os.path.isabs(args.baseline) else args.baseline

    if args.update_baseline:
        with open(baseline_path, "w") as fh:
            json.dump({"_comment": "P8 axis-6 five-way surface-agreement known "
                                   "gaps; the gate fails on any disagreement key "
                                   "NOT listed here. Shrink-only ratchet. "
                                   "Regenerate with five_way_surface.py "
                                   "--update-baseline.",
                       "counts": counts,
                       "known_disagreements": dis}, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print("wrote five-way baseline (%d disagreements) -> %s" % (len(dis), baseline_path))
        return 0

    baseline = set()
    if os.path.exists(baseline_path):
        baseline = set(json.load(open(baseline_path)).get("known_disagreements", []))
    new = [k for k in dis if k not in baseline]
    resolved = [k for k in baseline if k not in dis]

    if args.report:
        from collections import Counter
        c = Counter(k.split("::", 1)[0] for k in dis)
        print("surfaces:", counts)
        print("disagreements by class:", dict(c))

    status = "PASS" if not new else "FAIL"
    if args.trace:
        os.makedirs(os.path.dirname(args.trace) or ".", exist_ok=True)
        with open(args.trace, "a") as fh:
            fh.write(json.dumps({
                "kind": "escape_matrix", "name": "five_way_surface_agreement",
                "value": status, "total_disagreements": len(dis),
                "new_disagreements": new, "resolved_vs_baseline": len(resolved),
                "confidence": 0.97}) + "\n")

    print("axis-6 five-way surface: %d disagreements (baseline=%d), NEW=%d, resolved=%d"
          % (len(dis), len(baseline), len(new), len(resolved)))
    if new:
        print("NEW surface disagreements (not in baseline):")
        for k in new[:60]:
            print("   ", k)
    print("axis-6 gate: %s" % status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
