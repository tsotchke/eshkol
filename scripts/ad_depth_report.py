#!/usr/bin/env python3
"""Depth-parametric AD oracle reporter.

Reads the raw run log produced by scripts/run_ad_depth.sh, the cell registry
(tests/ad_depth/generated/cells.tsv), and produces:

  * AD_DEPTH_REPORT.md            per-cell depth table + MAX-CORRECT-DEPTH
  * scripts/icc_traces/ad_depth.jsonl   ICC events (kind ad_depth)

Verdict per (cell, mode, depth):
  PASS   RESULT ... PASS present
  FAIL   RESULT ... FAIL present  (silent wrong value at that depth)
  LIMIT  no RESULT line for that depth AND the run crashed/timed-out
         (clean error/exception boundary)

MAX-CORRECT-DEPTH(cell, mode) = largest d such that depths 1..d are all PASS
(0 if depth 1 is not PASS). A cell whose max-depth is below its tracked
baseline is a REGRESSION (gate red); at-or-above baseline is expected.

Raw log line grammar (tab-separated):
  RUN\t<mode>\t<file>\t<rc>\t<crashed 0|1>
  OUT\t<mode>\t<file>\t<raw stdout line>
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
GEN = os.path.join(ROOT, "tests", "ad_depth", "generated")
TRACE = os.path.join(ROOT, "scripts", "icc_traces", "ad_depth.jsonl")
REPORT = os.path.join(ROOT, "AD_DEPTH_REPORT.md")

# Tracked baseline max-correct-depth per (comp, cap) — the KNOWN capability
# boundary as measured by this oracle on master. A result >= baseline is
# expected (green); below baseline is a regression (red). Fixes that raise a
# boundary stay green and are flagged as improvements in the report.
BASELINE = {
    # ESH-0186 (P1 Taylor tower): nested `derivative` chains of depth>=3 route
    # through the arbitrary-order tower, so derivative^d is now exact to d=8 for
    # every capture form (this also closes the ESH-0122 localparam/vecref cases,
    # whose captures flow through the tower call unchanged). Was 2/2/1/1.
    ("deriv", "capnone"): 8,
    ("deriv", "glob"): 8,
    ("deriv", "localparam"): 8,
    ("deriv", "vecref"): 8,
    ("gradn", "capnone"): 2,
    ("gradn", "vecref"): 1,       # d2 returns garbage (ESH-0122 capture bug)
    ("gofd", "vecref"): 1,        # d>=2 returns 0 (ESH-0117 family)
    ("jacod", "vecref"): 0,       # d1 already returns 0 (ESH-0120)
    ("hessod", "vecref"): 0,      # d1 SIGSEGV (ESH-0121)
}
TRACK = {
    ("deriv", "localparam"): "ESH-0122",
    ("deriv", "vecref"): "ESH-0122",
    ("deriv", "capnone"): "ESH-0118",   # d>=3 returns 0
    ("deriv", "glob"): "ESH-0118",
    ("gradn", "capnone"): "ESH-0118",
    ("gradn", "vecref"): "ESH-0122",
    ("gofd", "vecref"): "ESH-0117",
    ("jacod", "vecref"): "ESH-0120",
    ("hessod", "vecref"): "ESH-0121",
}


def load_cells():
    cells = {}
    with open(os.path.join(GEN, "cells.tsv")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cid, comp, shape, point, binding, cap, fil, maxd = \
                line.rstrip("\n").split("\t")
            cells[cid] = dict(comp=comp, shape=shape, point=point,
                              binding=binding, cap=cap, file=fil,
                              maxdepth=int(maxd))
    return cells


def baseline_for(c):
    return BASELINE.get((c["comp"], c["cap"]), 0)


def track_for(c):
    return TRACK.get((c["comp"], c["cap"]), "")


RESULT_RE = re.compile(r"RESULT (\S+) d(\d+) (PASS|FAIL)")


def parse_log(path):
    """Return (verdicts, crashed) where
       verdicts[(cid, mode, depth)] = 'PASS'|'FAIL'
       crashed[(mode, file)] = bool."""
    verdicts = {}
    crashed = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "RUN" and len(parts) >= 5:
                _, mode, fil, rc, cr = parts[:5]
                crashed[(mode, fil)] = (cr == "1")
            elif parts[0] == "OUT" and len(parts) >= 4:
                mode, fil, raw = parts[1], parts[2], parts[3]
                m = RESULT_RE.search(raw)
                if m:
                    cid, d, verd = m.group(1), int(m.group(2)), m.group(3)
                    verdicts[(cid, mode, d)] = verd
    return verdicts, crashed


def max_correct_depth(cid, mode, maxd, verdicts):
    d = 0
    for k in range(1, maxd + 1):
        if verdicts.get((cid, mode, k)) == "PASS":
            d = k
        else:
            break
    return d


def was_run(mode, c, crashed):
    return (mode, c["file"]) in crashed


def cell_depth_row(cid, mode, c, verdicts, crashed):
    """Return (cells list, maxdepth). maxdepth is None if the cell's file was
    not executed in this mode (e.g. --quick subset)."""
    if not was_run(mode, c, crashed):
        return (["-"] * c["maxdepth"], None)
    cr = crashed.get((mode, c["file"]), False)
    row = []
    for k in range(1, c["maxdepth"] + 1):
        v = verdicts.get((cid, mode, k))
        if v == "PASS":
            row.append("P")
        elif v == "FAIL":
            row.append("F")
        elif cr:
            row.append("L")     # crashed/timed-out => clean limit boundary
        else:
            row.append("?")
    return row, max_correct_depth(cid, mode, c["maxdepth"], verdicts)


def esc(s):
    return (str(s).replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))


def main():
    if len(sys.argv) < 2:
        print("usage: ad_depth_report.py <raw-log> [modes...]", file=sys.stderr)
        return 2
    log = sys.argv[1]
    modes = sys.argv[2:] or ["jit", "aot"]
    cells = load_cells()
    verdicts, crashed = parse_log(log)

    rows = []          # (cid, c, per-mode {mode: (row, maxd)})
    regressions = []
    improvements = []
    for cid in sorted(cells):
        c = cells[cid]
        permode = {}
        for mode in modes:
            permode[mode] = cell_depth_row(cid, mode, c, verdicts, crashed)
        rows.append((cid, c, permode))
        base = baseline_for(c)
        run_mcds = [permode[m][1] for m in modes if permode[m][1] is not None]
        if not run_mcds:
            continue                       # cell not executed (e.g. --quick)
        obs = min(run_mcds)                # worst executed lane
        if obs < base:
            regressions.append((cid, obs, base))
        elif obs > base:
            improvements.append((cid, obs, base))

    # ---- ICC trace events -------------------------------------------------
    os.makedirs(os.path.dirname(TRACE), exist_ok=True)
    with open(TRACE, "w") as tf:
        for cid, c, permode in rows:
            for mode in modes:
                row, maxd = permode[mode]
                if maxd is None:
                    continue               # not executed in this mode
                base = baseline_for(c)
                val = "PASS" if maxd >= base else "FAIL"
                snip = f"{cid} {mode} max-correct-depth={maxd} baseline={base} " \
                       f"[{''.join(row)}]"
                tf.write('{"kind":"ad_depth","name":"%s","value":"%s",'
                         '"snippet":"%s","confidence":0.95}\n'
                         % (esc(f"{cid}::{mode}"), val, esc(snip)))
        gate = "PASS" if not regressions else "FAIL"
        tf.write('{"kind":"ad_depth","name":"ad_depth_gate","value":"%s",'
                 '"snippet":"%s","confidence":0.95}\n'
                 % (gate,
                    esc(f"{len(rows)} cells swept d1..8 x {len(modes)} lanes; "
                        f"{len(regressions)} regressions, "
                        f"{len(improvements)} improvements vs baseline")))

    # ---- markdown report --------------------------------------------------
    write_report(rows, modes, regressions, improvements)

    print(f"ad-depth: {len(rows)} cells x {len(modes)} lanes -> {TRACE}")
    print(f"          gate={'PASS' if not regressions else 'FAIL'}  "
          f"regressions={len(regressions)} improvements={len(improvements)}")
    for cid, obs, base in regressions:
        print(f"  REGRESSION {cid}: max-depth {obs} < baseline {base}")
    for cid, obs, base in improvements:
        print(f"  IMPROVEMENT {cid}: max-depth {obs} > baseline {base}")
    return 1 if regressions else 0


def write_report(rows, modes, regressions, improvements):
    L = []
    L.append("# Depth-parametric AD oracle report (P6a)\n")
    L.append("Generated by `scripts/run_ad_depth.sh` "
             "(`scripts/gen_ad_depth.py` + `scripts/ad_depth_report.py`).\n")
    L.append("Every AD value is swept at nesting depth **d = 1..8** and checked "
             "against a **closed-form analytic** ground truth (n-th derivative "
             "of the base shape), corroborated at shallow depth by an "
             "in-language n-th central-difference stencil. "
             "`P`=PASS `F`=FAIL(silent wrong) `L`=LIMIT(clean crash/limit).\n")
    L.append("**MCD** = max-correct-depth (largest d with 1..d all PASS, worst "
             "lane). Lanes: " + ", ".join(modes) + ".\n")

    # headline
    L.append("\n## Max-correct-depth by composition\n")
    L.append("| composition | capture | MCD | tracked |")
    L.append("|---|---|---|---|")
    seen = {}
    for cid, c, permode in rows:
        key = (c["comp"], c["cap"])
        run_mcds = [permode[m][1] for m in modes if permode[m][1] is not None]
        if not run_mcds:
            continue
        seen.setdefault(key, []).append(min(run_mcds))
    for key in sorted(seen):
        comp, cap = key
        mcds = seen[key]
        mn, mx = min(mcds), max(mcds)
        rng = f"{mn}" if mn == mx else f"{mn}–{mx}"
        tr = TRACK.get(key, "")
        L.append(f"| {comp} | {cap} | {rng} | {tr} |")

    if regressions:
        L.append("\n## ⚠️ REGRESSIONS (gate RED)\n")
        for cid, obs, base in regressions:
            L.append(f"- `{cid}`: max-depth **{obs}** < baseline **{base}**")
    if improvements:
        L.append("\n## ✅ Improvements over baseline (fix landed)\n")
        for cid, obs, base in improvements:
            L.append(f"- `{cid}`: max-depth **{obs}** > baseline **{base}** "
                     "(update BASELINE in ad_depth_report.py once confirmed)")

    L.append("\n## Per-cell depth tables\n")
    cur_comp = None
    for cid, c, permode in rows:
        if c["comp"] != cur_comp:
            cur_comp = c["comp"]
            L.append(f"\n### {cur_comp}\n")
            hdr = "| cell | lane | " + " | ".join(
                f"d{k}" for k in range(1, c["maxdepth"] + 1)) + " | MCD |"
            L.append(hdr)
            L.append("|" + "---|" * (c["maxdepth"] + 3))
        for mode in modes:
            row, mcd = permode[mode]
            cells_md = " | ".join(row)
            mcd_s = "—" if mcd is None else f"**{mcd}**"
            L.append(f"| `{cid}` | {mode} | {cells_md} | {mcd_s} |")

    with open(REPORT, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
