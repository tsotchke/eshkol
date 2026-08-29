#!/usr/bin/env python3
"""Fourth pass: a scheme block immediately followed by a bare/`text` fence is a
program plus the exact stdout the doc promises. Run it and diff the two.

Usage: check_output_blocks.py <repo> <eshkol-run> <out.json> [--mode jit|vm|aot]
"""

import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_examples import FENCE_RE, LANGS, closing_fence, iter_files  # noqa: E402

TIMEOUT = 90


def collect(root):
    pairs = []
    for rel in iter_files(root):
        path = os.path.join(root, rel)
        lines = open(path, encoding="utf-8").read().splitlines()
        i, n = 0, len(lines)
        while i < n:
            m = FENCE_RE.match(lines[i])
            if not m or m.group(3).lower() not in LANGS:
                i += 1
                continue
            j = closing_fence(lines, i, m)
            if j is None:
                break
            code = "\n".join(lines[i + 1 : j])
            # look ahead: optional blank lines, then a bare or ```text fence
            k = j + 1
            while k < n and not lines[k].strip():
                k += 1
            m2 = FENCE_RE.match(lines[k]) if k < n else None
            if m2 and m2.group(3).lower() in ("", "text", "output"):
                e = closing_fence(lines, k, m2)
                if e is not None:
                    expected = "\n".join(lines[k + 1 : e])
                    pairs.append({
                        "file": rel, "code_line": i + 1, "out_line": k + 1,
                        "code": code, "expected": expected,
                    })
            i = j + 1
    return pairs


def main():
    root, eshkol_run, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    mode = "jit"
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
    pairs = collect(root)
    scratch = os.path.join(root, ".scratch")
    os.makedirs(scratch, exist_ok=True)
    work = tempfile.mkdtemp(prefix="docaudit-out-", dir=scratch)
    results = []
    for n, p in enumerate(pairs):
        d = os.path.join(work, str(n))
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "ex.esk")
        open(src, "w").write(p["code"] + "\n")
        cmd = [eshkol_run, "-r", src] if mode == "jit" else [eshkol_run, src]
        env = dict(os.environ)
        env["ESHKOL_JIT_CACHE"] = "0"
        try:
            r = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, timeout=TIMEOUT)
            rc, so = r.returncode, r.stdout
            se = r.stderr
        except subprocess.TimeoutExpired:
            rc, so, se = -9, "", "TIMEOUT"
        got = so.rstrip("\n")
        exp = p["expected"].rstrip("\n")
        verdict = "MATCH" if got == exp else ("NORUN" if rc != 0 else "MISMATCH")
        results.append({**{k: p[k] for k in ("file", "code_line", "out_line", "expected")},
                        "got": got[:2000], "exit": rc, "verdict": verdict,
                        "stderr": se[-500:] if verdict != "MATCH" else ""})
        if (n + 1) % 50 == 0:
            print("... %d/%d" % (n + 1, len(pairs)), file=sys.stderr)
    json.dump(results, open(out_path, "w"), indent=1)
    from collections import Counter
    print(json.dumps(Counter(r["verdict"] for r in results)))


if __name__ == "__main__":
    main()
