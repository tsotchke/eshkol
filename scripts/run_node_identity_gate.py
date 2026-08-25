#!/usr/bin/env python3
"""ADR-0000 Stage 1 gate: measure the frontend node-identity substrate.

ADR-0000 makes each of its fourteen stages carry a *falsifiable* gate, and
§7 risk 5 predicts the program fails by scope compression if a stage is
declared attained on a description rather than a number. Stage 1's identity
half is the `NodeId -> {SourceSpan, BindingId, TypedExprInfo}` substrate;
this gate measures the first column of it.

What it measures
----------------
`ESHKOL_NODE_IDENTITY_STATS=1` makes every Eshkol frontend process print one
line to stderr as it exits:

    eshkol-node-identity: allocated=N queried=N resolved=N located=N extent=N

  allocated  NodeIds the parser handed out.
  queried    Nodes a substrate CONSUMER asked about. The consumer today is
             the LLVM codegen dispatcher, which resolves the location of
             every node it visits through the substrate.
  resolved   Of those, the ones whose NodeId named a real span.
  located    Of those resolved, the ones whose span names a real line.
  extent     Of those resolved, the ones whose span end is measured rather
             than mirroring the start.

The gated number is `resolved / queried`: the fraction of the AST that the
substrate actually covers on a real compile. It is deliberately measured at
a consumer and not at the parser, because a parser-side count can only say
how many ids were minted, not whether the answer arrived where it was
needed. `located` and `extent` are reported alongside it so "has an
identity", "has a location" and "has an extent" stay three separate numbers
and none of them can be mistaken for another.

Ratchet
-------
The floor lives in tests/coverage/NODE_IDENTITY_BASELINE.json and may never
fall. `--update-baseline` rewrites it; CI never passes that flag.

Usage:
  scripts/run_node_identity_gate.py [--update-baseline] [--corpus GLOB]...
                                    [--limit N] [--json OUT]
Exit: 0 green, 1 red, 2 misuse/environment.
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(REPO, "tests", "coverage", "NODE_IDENTITY_BASELINE.json")
TRACE_DIR = os.environ.get("TRACE_DIR", os.path.join(REPO, "scripts", "icc_traces"))
TRACE = os.path.join(TRACE_DIR, "node_identity.jsonl")

BUILD = os.environ.get("BUILD_DIR", os.path.join(REPO, "build"))
if not os.path.isabs(BUILD):
    BUILD = os.path.join(REPO, BUILD)
ESHKOL_RUN = os.path.join(BUILD, "eshkol-run")

# A deliberately mixed corpus: plain expressions, control flow, macros,
# modules and tensor/AD forms, so the measurement is not dominated by one
# node shape. These are compile-only runs; program behaviour is irrelevant
# here and is gated everywhere else.
DEFAULT_CORPUS = [
    "tests/parser/*.esk",
    "tests/control_flow/*.esk",
    "tests/macros/*.esk",
    "tests/lists/*.esk",
    "tests/v1_2_edge_cases/*.esk",
]

REPORT_RE = re.compile(
    r"^eshkol-node-identity: allocated=(\d+) queried=(\d+) resolved=(\d+) "
    r"located=(\d+) extent=(\d+)\s*$")


def die(msg):
    sys.stderr.write("run_node_identity_gate: %s\n" % msg)
    sys.exit(2)


def parse_report(stderr_text):
    """Return the LAST substrate report line in `stderr_text`, or None.

    The last one, not the first: a driver that forks a cache-build child
    emits the child's line first, and the parent's totals are the ones that
    describe the compile the caller asked for.
    """
    found = None
    for line in stderr_text.splitlines():
        match = REPORT_RE.match(line.strip())
        if match:
            found = tuple(int(g) for g in match.groups())
    return found


def measure(path, workdir):
    """Compile one source with substrate stats on; return its counters."""
    env = dict(os.environ)
    env["ESHKOL_NODE_IDENTITY_STATS"] = "1"
    # Compile only. The gate is about the frontend; running the program
    # would add its runtime's exit behaviour to what we are measuring.
    out = os.path.join(workdir, "out.o")
    try:
        proc = subprocess.run(
            [ESHKOL_RUN, "-c", path, "-o", out],
            # Generous on purpose. A source dropped for taking too long would
            # change the denominator and make the ratchet jitter in its last
            # digit, which is worse than a slow gate: a floor that moves on its
            # own cannot tell a regression from a loaded machine.
            env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None
    return parse_report(proc.stderr or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--corpus", action="append", default=None)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if not os.path.exists(ESHKOL_RUN):
        die("eshkol-run not found at %s (build first, or set BUILD_DIR)"
            % ESHKOL_RUN)

    patterns = args.corpus or DEFAULT_CORPUS
    sources = []
    for pattern in patterns:
        sources.extend(sorted(glob.glob(os.path.join(REPO, pattern))))
    sources = sources[:args.limit]
    if not sources:
        die("corpus matched no sources: %s" % ", ".join(patterns))

    allocated = queried = resolved = located = extent = 0
    measured = 0
    silent = []

    # Scratch objects go under the build tree, not the system temp dir: the
    # build tree is where the rest of this gate's inputs already live, and it
    # survives whatever the machine does to /tmp between runs.
    scratch_root = os.path.join(BUILD, "node-identity-gate")
    os.makedirs(scratch_root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run-", dir=scratch_root) as workdir:
        for path in sources:
            counters = measure(path, workdir)
            if counters is None:
                silent.append(os.path.relpath(path, REPO))
                continue
            measured += 1
            allocated += counters[0]
            queried += counters[1]
            resolved += counters[2]
            located += counters[3]
            extent += counters[4]

    fraction = (resolved / queried) if queried else 0.0
    located_fraction = (located / queried) if queried else 0.0

    print("ADR-0000 Stage 1 — node-identity substrate coverage")
    print("  sources compiled with stats on  : %d / %d" % (measured, len(sources)))
    print("  NodeIds allocated by the parser : %d" % allocated)
    print("  consumer lookups                : %d" % queried)
    print("  resolved to a span              : %d  (%.2f%%)  [the gated number]"
          % (resolved, 100.0 * fraction))
    print("  span names a real line          : %d  (%.2f%%)"
          % (located, 100.0 * located_fraction))
    print("  span end is measured (extent)   : %d" % extent)

    baseline = {"span_coverage_floor": 0.0}
    if os.path.exists(BASELINE):
        try:
            with open(BASELINE, encoding="utf-8") as handle:
                baseline = json.load(handle)
        except (OSError, ValueError) as exc:
            die("baseline %s is unreadable: %s" % (BASELINE, exc))
    floor = float(baseline.get("span_coverage_floor", 0.0))

    if args.update_baseline:
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w", encoding="utf-8") as handle:
            json.dump({
                "_comment": (
                    "Ratchet for scripts/run_node_identity_gate.py "
                    "(ADR-0000 Stage 1, node-identity substrate). "
                    "span_coverage_floor is resolved/queried at the LLVM "
                    "codegen dispatcher and may never fall."),
                # Truncated, never rounded. `round()` can land ABOVE the value
                # it came from (0.994858 -> 0.9949), and a floor recorded above
                # the measurement it was taken from fails the very run that
                # wrote it — a ratchet that no unchanged tree can satisfy.
                "span_coverage_floor": math.floor(fraction * 10000) / 10000.0,
            }, handle, indent=2)
            handle.write("\n")
        print("  baseline written: floor %.4f" % fraction)
        return 0

    # A substrate that minted no ids, or one no consumer ever asked, is not
    # a substrate. Both are separate failures from "coverage fell".
    substrate_present = allocated > 0 and queried > 0 and resolved > 0
    coverage_ok = fraction >= floor
    passed = substrate_present and coverage_ok and not silent

    timestamp = time.time()
    events = [
        {
            "kind": "runtime_event",
            "event": "node_identity_substrate_present",
            "name": "node_identity_substrate_present",
            "value": "PASS" if substrate_present else "FAIL",
            "status": "PASSED" if substrate_present else "FAILED",
            "allocated": allocated,
            "queried": queried,
            "resolved": resolved,
            "sources_measured": measured,
            "sources_silent": silent[:20],
            "timestamp": timestamp,
            "confidence": 1.0,
        },
        {
            "kind": "runtime_event",
            "event": "node_identity_span_coverage",
            "name": "node_identity_span_coverage",
            "value": "PASS" if (coverage_ok and substrate_present) else "FAIL",
            "status": "PASSED" if (coverage_ok and substrate_present) else "FAILED",
            "covered_fraction": round(fraction, 4),
            "located_fraction": round(located_fraction, 4),
            "resolved": resolved,
            "queried": queried,
            "located": located,
            "extent": extent,
            "threshold": floor,
            "timestamp": timestamp,
            "confidence": 1.0,
        },
    ]

    os.makedirs(TRACE_DIR, exist_ok=True)
    with open(TRACE, "w", encoding="utf-8") as handle:
        for event in events:
            json.dump(event, handle, sort_keys=True)
            handle.write("\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({
                "allocated": allocated,
                "queried": queried,
                "resolved": resolved,
                "located": located,
                "extent": extent,
                "span_coverage": fraction,
                "floor": floor,
                "silent_sources": silent,
            }, handle, indent=2)
            handle.write("\n")

    rc = 0
    if not substrate_present:
        rc = 1
        print()
        print("FAIL: no substrate evidence. allocated=%d queried=%d resolved=%d"
              % (allocated, queried, resolved))
        print("      A NodeId nobody mints, or nobody asks for, is not an "
              "identity substrate.")
    if silent:
        rc = 1
        print()
        print("FAIL: %d source(s) compiled without emitting a substrate report."
              % len(silent))
        print("      Either the frontend did not run, or the report hook is "
              "not on that path.")
        for name in silent[:20]:
            print("  %s" % name)
    if not coverage_ok:
        rc = 1
        print()
        print("FAIL: span coverage %.2f%% fell below the recorded floor %.2f%%."
              % (100.0 * fraction, 100.0 * floor))
        print("      Regenerate deliberately with --update-baseline only when "
              "the drop is understood.")
    if rc == 0:
        print()
        print("PASS: substrate present; span coverage %.2f%% >= floor %.2f%%."
              % (100.0 * fraction, 100.0 * floor))
    print("  trace: %s" % os.path.relpath(TRACE, REPO))
    return rc


if __name__ == "__main__":
    sys.exit(main())
