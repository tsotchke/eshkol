#!/usr/bin/env python3
"""Second pass: re-run failing blocks with the file's earlier blocks as a prelude.

A block that only fails because it references a helper defined in an earlier
block of the same document is `needs-context`, not a defect. Anything still
failing with the whole document in scope is a candidate finding.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

TIMEOUT = 60
MISSING = re.compile(
    r"Undefined variable: ([^\s]+)|called undefined function '([^']+)'|Unbound variable: ([^\s(]+)"
)


def main():
    examples = json.load(open(sys.argv[1]))
    results = json.load(open(sys.argv[2]))
    eshkol_run = sys.argv[3]
    out_path = sys.argv[4]

    by_file = {}
    for e in examples:
        by_file.setdefault(e["file"], []).append(e)
    for v in by_file.values():
        v.sort(key=lambda x: x["start_line"])

    ok_keys = {(r["file"], r["start_line"]) for r in results if r["exit"] == 0}

    out = []
    work = tempfile.mkdtemp(prefix="docaudit-ctx-")
    todo = [r for r in results if r["exit"] != 0]
    for n, r in enumerate(todo):
        blob = (r["stderr"] or "") + (r["stdout"] or "")
        if not MISSING.search(blob):
            r2 = dict(r)
            r2["ctx_exit"] = None
            r2["ctx_note"] = "not-a-missing-name failure"
            out.append(r2)
            continue
        prelude = []
        for e in by_file[r["file"]]:
            if e["start_line"] >= r["start_line"]:
                break
            if (e["file"], e["start_line"]) in ok_keys and "(define" in e["code"]:
                prelude.append(e["code"])
        d = os.path.join(work, "%d" % n)
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "ex.esk")
        with open(src, "w") as fh:
            fh.write("\n".join(prelude))
            fh.write("\n")
            fh.write(r"" if not prelude else "")
            fh.write(next(
                e["code"] for e in by_file[r["file"]] if e["start_line"] == r["start_line"]
            ))
            fh.write("\n")
        env = dict(os.environ)
        try:
            p = subprocess.run(
                [eshkol_run, "-r", src], cwd=d, env=env,
                capture_output=True, text=True, timeout=TIMEOUT,
            )
            rc, se = p.returncode, p.stderr
        except subprocess.TimeoutExpired:
            rc, se = -9, "TIMEOUT"
        r2 = dict(r)
        r2["ctx_exit"] = rc
        r2["ctx_stderr"] = se[:4000]
        r2["ctx_prelude_blocks"] = len(prelude)
        out.append(r2)
        if (n + 1) % 25 == 0:
            print("... %d/%d" % (n + 1, len(todo)), file=sys.stderr)

    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    healed = sum(1 for r in out if r.get("ctx_exit") == 0)
    print("retried=%d healed_by_context=%d still_failing=%d" % (len(out), healed, len(out) - healed))


if __name__ == "__main__":
    main()
