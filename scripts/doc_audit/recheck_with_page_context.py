#!/usr/bin/env python3
"""Re-run the output-block pairs that failed standalone, this time with the
page's own context: the `**Require**:` line from the page header plus every
earlier scheme block in the same file.

Reference pages state their `(require …)` once in the header and omit it from
each snippet, and later snippets reuse names bound by earlier ones. A block that
only fails for one of those reasons is `needs-context`, not a defect — but its
*output* is still checkable once the context is supplied, which is what this
pass does.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_output_blocks import collect  # noqa: E402

TIMEOUT = 90
REQ = re.compile(r"`\(require ([a-z0-9_.-]+)\)`")


def page_requires(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for i, ln in enumerate(fh):
            if i > 40:
                break
            if "**Require**" in ln or ln.startswith("**Require"):
                out += REQ.findall(ln)
    return out


def main():
    root, eshkol_run, prev_path, out_path = sys.argv[1:5]
    prev = json.load(open(prev_path))
    failed = {(r["file"], r["code_line"]) for r in prev if r["verdict"] != "MATCH"}
    pairs = collect(root)
    by_file = {}
    for p in pairs:
        by_file.setdefault(p["file"], []).append(p)
    work = tempfile.mkdtemp(prefix="docaudit-ctx2-")
    results = []
    todo = [p for p in pairs if (p["file"], p["code_line"]) in failed]
    for n, p in enumerate(todo):
        reqs = page_requires(os.path.join(root, p["file"]))
        prelude = ["(require %s)" % r for r in reqs]
        for q in by_file[p["file"]]:
            if q["code_line"] >= p["code_line"]:
                break
            prelude.append(q["code"])
        d = os.path.join(work, str(n))
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "ex.esk")
        open(src, "w").write("\n".join(prelude) + "\n" + p["code"] + "\n")
        try:
            r = subprocess.run([eshkol_run, "-r", src], cwd=d,
                               capture_output=True, text=True, timeout=TIMEOUT)
            rc, so, se = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            rc, so, se = -9, "", "TIMEOUT"
        # the block's own output is the tail of stdout
        exp = p["expected"].rstrip("\n")
        got_all = so.rstrip("\n")
        tail = "\n".join(got_all.splitlines()[-len(exp.splitlines()):]) if exp else got_all
        verdict = "MATCH" if tail == exp else ("NORUN" if rc != 0 else "MISMATCH")
        results.append({"file": p["file"], "code_line": p["code_line"],
                        "out_line": p["out_line"], "expected": exp,
                        "got_tail": tail[:1500], "exit": rc, "verdict": verdict,
                        "prelude_blocks": len(prelude),
                        "stderr": se[-400:] if verdict != "MATCH" else ""})
        if (n + 1) % 25 == 0:
            print("... %d/%d" % (n + 1, len(todo)), file=sys.stderr)
    json.dump(results, open(out_path, "w"), indent=1)
    from collections import Counter
    print(json.dumps(Counter(r["verdict"] for r in results)))


if __name__ == "__main__":
    main()
