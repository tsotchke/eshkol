#!/usr/bin/env python3
"""Rewrite doc output blocks whose ONLY divergence from the build is that the
doc pasted a truncated float.

Safety rule: a block is rewritten only when the expected and actual text have
the identical non-numeric skeleton AND every numeric token agrees to within the
precision the doc actually printed (i.e. the doc's number is a correct rounding
of the build's number). Anything else is left alone and reported.

Usage: patch_float_outputs.py <repo> <results.json> [--apply]
Reads `expected`/`got` (or `got_tail`) plus `out_line` from the results file.
"""

import json
import math
import re
import sys

NUM = re.compile(r"[-+]?(?:\d+\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\d+)")


def split(text):
    nums = []
    skel = []
    last = 0
    for m in NUM.finditer(text):
        skel.append(text[last:m.start()])
        nums.append(m.group(0))
        last = m.end()
    skel.append(text[last:])
    return skel, nums


def sig_digits(s):
    """Significant decimal digits in a literal like -0.00842293 or 1.38e-17."""
    mant = re.split(r"[eE]", s)[0].lstrip("-+")
    mant = mant.replace(".", "").lstrip("0")
    return len(mant.rstrip("0")) or 1


def rounds_to(exp_s, got_s):
    """True if exp_s is a correct rounding of got_s at exp_s's own precision.

    Integers must match EXACTLY: a doc that says `3` where the build says `7` is
    a wrong answer, not a truncated one, and must never be auto-rewritten.
    """
    if exp_s == got_s:
        return True
    is_float = lambda s: ("." in s) or ("e" in s) or ("E" in s)
    if not (is_float(exp_s) and is_float(got_s)):
        return False
    try:
        e = float(exp_s)
        g = float(got_s)
    except ValueError:
        return False
    if g == 0:
        return abs(e) < 1e-12
    tol = abs(g) * 10 ** (-(sig_digits(exp_s) - 1)) * 0.75 + 1e-300
    return math.isclose(e, g, rel_tol=0, abs_tol=tol)


def main():
    root = sys.argv[1]
    results = json.load(open(sys.argv[2]))
    apply = "--apply" in sys.argv
    patched, skipped = [], []
    edits = {}
    for r in results:
        if r.get("verdict") != "MISMATCH":
            continue
        exp = r["expected"]
        got = r.get("got_tail", r.get("got", ""))
        se, ne = split(exp)
        sg, ng = split(got)
        why = None
        if se != sg:
            why = "skeleton differs"
        elif len(ne) != len(ng):
            why = "different number count"
        else:
            for a, b in zip(ne, ng):
                if not rounds_to(a, b):
                    why = "value %s is not a rounding of %s" % (a, b)
                    break
        if why:
            skipped.append((r["file"], r.get("code_line"), why))
            continue
        edits.setdefault(r["file"], []).append((r["out_line"], exp, got))
        patched.append((r["file"], r.get("code_line")))

    for rel, items in sorted(edits.items()):
        path = "%s/%s" % (root, rel)
        lines = open(path, encoding="utf-8").read().split("\n")
        # apply from the bottom so earlier line numbers stay valid
        for out_line, exp, got in sorted(items, reverse=True):
            start = out_line  # 0-based index of the first content line
            n = len(exp.split("\n"))
            assert "\n".join(lines[start:start + n]) == exp, (rel, out_line)
            lines[start:start + n] = got.split("\n")
        if apply:
            open(path, "w", encoding="utf-8").write("\n".join(lines))

    print("patched %d block(s) across %d file(s)%s"
          % (len(patched), len(edits), "" if apply else "  [DRY RUN]"))
    for rel in sorted(edits):
        print("  %s: %d" % (rel, len(edits[rel])))
    print("skipped %d block(s) — need a human decision:" % len(skipped))
    for f, l, why in skipped:
        print("  %s:%s  %s" % (f, l, why))


if __name__ == "__main__":
    main()
