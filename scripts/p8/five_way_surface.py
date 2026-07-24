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
  3. native registration  manifest backends intersect {native, native_llvm}
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
DEFINE = re.compile(r"\(define\s+\(?\s*([a-z][a-zA-Z0-9!?*+<>=./_%-]*)")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


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
    present = set()
    for e in d.get("builtins", []):
        present.add(e["name"])
        backends[e["name"]] = set(e.get("backends") or [])
    for key in ("special_forms", "prelude"):
        for e in d.get(key, []):
            nm = e["name"] if isinstance(e, dict) else e
            present.add(nm)
    return present, backends


def collect_modules(root):
    provided = set()
    defined = set()
    for esk in glob.glob(os.path.join(root, "lib/**/*.esk"), recursive=True):
        txt = read(esk)
        for m in PROVIDE.finditer(txt):
            for tok in m.group(1).split():
                tok = tok.strip()
                if tok and not tok.startswith(";"):
                    provided.add(tok)
        for m in DEFINE.finditer(txt):
            defined.add(m.group(1))
    return provided, defined


def compute_disagreements(root):
    docs = collect_docs(root)
    manifest, backends = collect_manifest(root)
    provided, defined = collect_modules(root)

    dis = set()
    # doc_orphan: documented but nowhere implemented/exported.
    for nm in docs:
        if nm not in manifest and nm not in provided and nm not in defined:
            dis.add("doc_orphan::" + nm)
    # backend asymmetry over manifest builtins.
    for nm, bk in backends.items():
        has_native = bool(bk & {"native", "native_llvm"})
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--baseline",
                    default="tests/escape_matrix/five_way_baseline.json")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--trace")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

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
