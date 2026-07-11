#!/usr/bin/env python3
"""language_coverage.py — dynamic language-surface coverage for the exposure engines.

Runs the two generative exposure engines, collects the set of BUILTINS and
SPECIAL FORMS their generated (and executed) programs actually contain, diffs
that against the ground-truth manifest (tests/coverage/language_surface.json),
and reports covered% plus the categorised list of UNCOVERED constructs — the
constructs no engine exercises today.

This is the "ICC tracks every part of the language dynamically" mechanism: each
run emits a coverage sidecar (tests/coverage/coverage_run.json) whose
`covered_fraction` and `uncovered` fields are consumable as an ICC
completion-oracle runtime_event (see tests/coverage/README.md).

Engines measured:
  * scripts/gen_generative_corpus.py        (run_generative_differential.py)
  * tests/ad_adversarial/gen_ad_adversarial.py (run_ad_adversarial.sh)

Usage:
  python3 scripts/language_coverage.py [--json OUT] [--emit-runtime-event]
"""

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests", "ad_adversarial"))

MANIFEST = os.path.join(REPO, "tests", "coverage", "language_surface.json")

# reader macros -> the special form they expand to
READER_MACROS = {
    "'": "quote",
    "`": "quasiquote",
    ",@": "unquote-splicing",
    ",": "unquote",
}


def collect_heads(text):
    """Return the set of symbols that appear as an application/operator head, plus
    the special forms introduced by reader macros (', `, , ,@) and #( vectors."""
    heads = set()
    # reader-macro forms actually present in the source
    if "#(" in text:
        heads.add("vector")           # vector literal reader syntax
    for rm, form in READER_MACROS.items():
        if rm in text:
            heads.add(form)
    # walk tokens; a symbol immediately following '(' is a head
    i, n = 0, len(text)
    expect_head = False
    while i < n:
        c = text[i]
        if c == ";":                  # line comment
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == '"':                  # string literal
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    i += 1
                i += 1
            i += 1
            expect_head = False
            continue
        if c == "#" and i + 1 < n and text[i + 1] == "\\":  # char literal
            i += 2
            while i < n and (text[i].isalnum()):
                i += 1
            expect_head = False
            continue
        if c == "(":
            expect_head = True
            i += 1
            continue
        if c in ") \t\n\r'`,":
            i += 1
            continue
        # read an atom
        j = i
        while j < n and text[j] not in "() \t\n\r\";":
            j += 1
        atom = text[i:j]
        if expect_head and atom:
            heads.add(atom)
        expect_head = False
        i = j
    return heads


def gen_generative_corpus_text():
    import gen_generative_corpus as g
    progs = g.generate_programs()
    return "\n".join(body for _, _, body in progs)


def gen_ad_adversarial_text():
    import gen_ad_adversarial as a
    probes = a.Gen().generate()
    parts = [a.PRELUDE, a.SUMMARY]
    for p in probes:
        parts.append("\n".join(p["lines"]))
    return "\n".join(parts)


def load_manifest():
    with open(MANIFEST) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=os.path.join(REPO, "tests", "coverage",
                                                   "coverage_run.json"))
    ap.add_argument("--emit-runtime-event", action="store_true",
                    help="print a one-line ICC runtime_event JSON to stdout")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="fail (exit 1) if covered_fraction < threshold")
    args = ap.parse_args()

    manifest = load_manifest()
    # surface = builtins + special forms + prelude higher-order functions.
    surface = {}
    for b in manifest["builtins"]:
        surface[b["name"]] = b["category"]
    for s in manifest["special_forms"]:
        surface.setdefault(s["name"], s["category"])
    for p in manifest["prelude"]:
        surface.setdefault(p["name"], p["category"])
    # internal-only helpers (leading underscore) are not user-facing surface.
    surface = {k: v for k, v in surface.items() if not k.startswith("_")}

    text = gen_generative_corpus_text() + "\n" + gen_ad_adversarial_text()
    heads = collect_heads(text)

    covered = {k: v for k, v in surface.items() if k in heads}
    uncovered = {k: v for k, v in surface.items() if k not in heads}

    total = len(surface)
    frac = len(covered) / total if total else 0.0

    # categorised uncovered, ranked by silent-wrong-risk then size
    RISK_ORDER = ["numeric", "tensor_ad", "geometry", "control_flow",
                  "consciousness", "higher_order", "list_pair", "vector",
                  "string_char", "hash", "predicate", "io_port", "binding_form",
                  "macro_syntax", "module", "memory_region", "misc_core",
                  "ffi_system", "misc"]
    by_cat = {}
    for k, v in uncovered.items():
        by_cat.setdefault(v, []).append(k)
    cov_by_cat = {}
    for k, v in covered.items():
        cov_by_cat.setdefault(v, []).append(k)

    ranked = sorted(by_cat.items(),
                    key=lambda kv: (RISK_ORDER.index(kv[0])
                                    if kv[0] in RISK_ORDER else 99,
                                    -len(kv[1])))

    out = {
        "surface_total": total,
        "covered": len(covered),
        "uncovered": len(uncovered),
        "covered_fraction": round(frac, 4),
        "covered_by_category": {k: len(v) for k, v in sorted(cov_by_cat.items())},
        "uncovered_by_category": {k: sorted(v) for k, v in ranked},
        "covered_names": sorted(covered),
        "engines": ["gen_generative_corpus.py", "gen_ad_adversarial.py"],
    }
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")

    print("Language-surface coverage (exposure engines)")
    print("  surface constructs : %d" % total)
    print("  covered            : %d (%.1f%%)" % (len(covered), 100 * frac))
    print("  uncovered          : %d" % len(uncovered))
    print("  sidecar            : %s" % args.json)
    print("\nUncovered by category (highest silent-wrong risk first):")
    for cat, names in ranked:
        cov_n = len(cov_by_cat.get(cat, []))
        tot = cov_n + len(names)
        print("  %-14s %3d/%-3d uncovered  (e.g. %s)"
              % (cat, len(names), tot, ", ".join(sorted(names)[:6])))

    if args.emit_runtime_event:
        event = {
            "kind": "runtime_event",
            "event": "language_surface_coverage",
            "covered_fraction": round(frac, 4),
            "covered": len(covered),
            "surface_total": total,
            "status": "PASSED" if frac >= args.threshold else "FAILED",
        }
        print(json.dumps(event))

    if frac < args.threshold:
        print("\nFAIL: covered_fraction %.4f < threshold %.4f"
              % (frac, args.threshold), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
