#!/usr/bin/env python3
"""Third pass: compare `;; => value` annotations against what the build prints.

For every doc block, each source line of the shape

    <one complete form>   ;; => <expected>

is rewritten to `(display <form>)(newline)` and the block is executed. The
printed sequence is matched against the annotation sequence.

Verdicts per pair:
  MATCH      printed text equals the annotation (after normalisation)
  MISMATCH   both are values and they differ  <- the finding class
  PROSE      the annotation is not a value literal (a comment), skipped
  NORUN      the block did not execute cleanly, so no comparison is possible
"""

import json
import os
import re
import subprocess
import sys
import tempfile

TIMEOUT = 60
ANN = re.compile(r"^(\s*)(\(.*)\s;+\s*=>\s*(.+?)\s*$")
SENTINEL = "@@DOCAUDIT@@"


def balanced_prefix(text):
    """Return the length of the shortest balanced prefix of `text`, or None."""
    depth = 0
    in_str = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == ";":
            return None
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "#" and i + 1 < n and text[i + 1] == "\\":
            i += 3
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0:
                return i + 1
            if depth < 0:
                return None
        i += 1
    return None


VALUE_RE = re.compile(
    r"^(#t|#f|#true|#false|'?\(.*\)|#\(.*\)|\"[^\"]*\"|#\\.|[-+]?[0-9][0-9a-zA-Z.eE+/_-]*|[-+]?\.[0-9]+)$"
)


def is_value(text):
    t = text.strip()
    # strip a trailing explanatory clause after two spaces or a comma
    return bool(VALUE_RE.match(t))


def normalise(text):
    t = text.strip()
    if t.startswith("'"):
        t = t[1:]
    return t


def instrument(code):
    """Return (new_code, [expected...]) with annotated forms wrapped in display."""
    out = []
    expects = []
    for ln in code.splitlines():
        m = ANN.match(ln)
        if not m:
            out.append(ln)
            continue
        indent, rest, exp = m.group(1), m.group(2), m.group(3)
        k = balanced_prefix(rest)
        if k is None or rest[k:].strip():
            out.append(ln)
            continue
        form = rest[:k]
        out.append(
            '%s(display "%s")(display %s)(newline)' % (indent, SENTINEL, form)
        )
        expects.append(exp)
    return "\n".join(out), expects


def main():
    examples = json.load(open(sys.argv[1]))
    eshkol_run = sys.argv[2]
    out_path = sys.argv[3]
    work = tempfile.mkdtemp(prefix="docaudit-exp-")
    findings = []
    stats = {"blocks_with_annotations": 0, "pairs": 0, "MATCH": 0, "MISMATCH": 0,
             "PROSE": 0, "NORUN": 0}
    for n, e in enumerate(examples):
        code, expects = instrument(e["code"])
        if not expects:
            continue
        stats["blocks_with_annotations"] += 1
        d = os.path.join(work, "%d" % n)
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "ex.esk")
        with open(src, "w") as fh:
            fh.write(code + "\n")
        env = dict(os.environ)
        try:
            p = subprocess.run([eshkol_run, "-r", src], cwd=d, env=env,
                               capture_output=True, text=True, timeout=TIMEOUT)
            rc, so, se = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            rc, so, se = -9, "", "TIMEOUT"
        printed = []
        for chunk in so.split(SENTINEL)[1:]:
            printed.append(chunk.split("\n")[0])
        for i, exp in enumerate(expects):
            stats["pairs"] += 1
            got = printed[i] if i < len(printed) else None
            if not is_value(exp):
                verdict = "PROSE"
            elif rc != 0 or got is None:
                verdict = "NORUN"
            elif normalise(exp) == normalise(got):
                verdict = "MATCH"
            else:
                verdict = "MISMATCH"
            stats[verdict] += 1
            if verdict in ("MISMATCH", "NORUN"):
                findings.append({
                    "file": e["file"], "start_line": e["start_line"],
                    "expected": exp, "got": got, "verdict": verdict,
                    "exit": rc, "stderr": se[-600:],
                })
        if (n + 1) % 100 == 0:
            print("... %d/%d" % (n + 1, len(examples)), file=sys.stderr)
    with open(out_path, "w") as fh:
        json.dump({"stats": stats, "findings": findings}, fh, indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
