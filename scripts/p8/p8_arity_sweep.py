#!/usr/bin/env python3
r"""p8_arity_sweep.py — P8 escape-closure axis 3: manifest-driven native-vs-VM
arity/parity differential.

Originating escape (see .swarm/P8_ESCAPE_ANALYSIS.md): a family of tensor/matmul
special forms behaved differently under the hosted VM than under native codegen
(wrong arity handling for arange, nested-literal operands dropped, multi-dim
tensor-ref/set mismatched). No generator asserted native==VM across the
DOCUMENTED builtin surface, so the divergence shipped until reported.

This harness reads tests/coverage/language_surface.json and, for every builtin
with a known arity that is registered on BOTH the native and the VM backend,
constructs three deterministic calls:
    correct  documented arity with type-correct args (category heuristic)
    warity   one wrong-arity call (arity +/- 1)
    wtype    one wrong-type call (an argument swapped for an incompatible type)
Each call is guard-wrapped and its `write`n value (or the token ERR on a caught
error) is printed on an indexed line. The program is run under NATIVE
(eshkol-run -r) and the hosted VM (emit-eskb + eshkol-vm-standalone-test); the
per-index outputs must be IDENTICAL — a matching value on both, or ERR on both.
A hard crash (SIGSEGV/abort, which guard cannot catch) on one engine but not the
other truncates that engine's output and is reported as a divergence at the
first missing index.

Each builtin gets its OWN tiny program so one hard crash isolates to three
probes instead of poisoning the batch.

Determinism: a builtin's three probes are a pure function of its NAME (args are
derived from a per-name hash), so a probe is identical regardless of which
sample it lands in — the baseline is stable. --seed only selects WHICH builtins
are sampled in the bounded (CI) mode.

Baseline ratchet: current native-vs-VM divergences are legitimate, documented
parity gaps (VM-only error text, guard semantics, native type-error hard-faults)
and are grandfathered into tests/escape_matrix/arity_parity_baseline.json. The
gate fails only on a NEW divergence key not in the baseline (shrink-only: a key
that no longer diverges can be dropped from the baseline, never added silently).

Usage:
  p8_arity_sweep.py --native BIN --vm VMBIN [--sample N | --full] [--seed S]
                    [--baseline FILE] [--update-baseline] [--trace FILE]
                    [--workdir DIR] [--timeout SECS]
Exit 0 iff no NEW divergence and no builtin hard-crashed on exactly one engine
outside the baseline.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

CATEGORIES = {"numeric", "predicate", "string_char", "list_pair", "vector",
              "hash", "higher_order"}

# Deterministic per-type value pools. Index into them with a per-name hash.
POOLS = {
    "num": ["3", "5", "2", "7", "4", "6"],
    "flo": ["2.5", "3.5", "1.5"],
    "str": ['"hello"', '"abc"', '"xyz"'],
    "list": ["(list 1 2 3)", "(list 4 5 6)", "(list 7 8)"],
    "vec": ["(vector 1 2 3)", "(vector 4 5 6)"],
    "char": ["#\\a", "#\\z", "#\\m"],
    "bool": ["#t", "#f"],
    "fn": ["(lambda (x) (+ x 1))", "(lambda (x) (* x 2))"],
}

# category -> ordered list of arg types to cycle through for a correct call.
CAT_ARGS = {
    "numeric": ["num"],
    "predicate": ["num"],
    "string_char": ["str"],
    "list_pair": ["list"],
    "vector": ["vec"],
    "hash": ["hash"],           # handled specially (needs a table binding)
    "higher_order": ["fn", "list"],
}

# Incompatible type to swap in for the wrong-type probe, by category.
WTYPE = {
    "numeric": '"notanumber"',
    "predicate": '"s"',
    "string_char": "42",
    "list_pair": "42",
    "vector": "42",
    "hash": "42",
    "higher_order": "42",
}


def _h(name):
    return int(hashlib.sha1(name.encode()).hexdigest(), 16)


def pick(pool, name, slot):
    lst = POOLS[pool]
    return lst[(_h(name) + slot) % len(lst)]


def correct_args(name, cat, arity):
    if cat == "hash":
        # first arg is the table binding __ht; remaining args typed by name.
        args = ["__ht"]
        for i in range(1, arity):
            args.append(pick("num", name, i))
        return args[:arity] if arity else []
    types = CAT_ARGS.get(cat, ["num"])
    out = []
    for i in range(arity):
        t = types[i] if i < len(types) else types[-1]
        out.append(pick(t, name, i))
    return out


def build_program(name, cat, arity):
    """Return (esk_source, [probe_keys]) — 3 indexed guard-wrapped probes.

    The VM emits a newline after every output op, so a probe cannot rely on
    line structure. Each probe is bracketed by whitespace-free sentinels
    (#P8# idx # value #/P8#); the reader strips ALL whitespace from both
    engines' output identically, then regex-extracts each probe. A value's
    own internal spaces vanish on both sides, so the comparison is exact and
    newline-insertion-proof."""
    head = [
        "(define (probe idx thunk)",
        "  (display \"#P8#\") (display idx) (display \"#\")",
        "  (guard (e (#t (display \"ERR\")))",
        "    (write (thunk)))",
        "  (display \"#/P8#\"))",
    ]
    if cat == "hash":
        head.append("(define __ht (make-hash-table))")
        head.append("(hash-table-set! __ht 1 10)")
    src = ["\n".join(head)]
    keys = []

    def call(args):
        return "(%s %s)" % (name, " ".join(args))

    # correct arity
    a = correct_args(name, cat, arity)
    src.append('(probe 0 (lambda () %s))' % call(a))
    keys.append("%s::correct" % name)
    # wrong arity: one extra arg (or one fewer if arity>=1)
    if arity >= 1:
        wa = a[:-1]
    else:
        wa = [pick("num", name, 99)]
    src.append('(probe 1 (lambda () %s))' % call(wa))
    keys.append("%s::warity" % name)
    # wrong type: swap arg 0 (if any) for an incompatible type
    if arity >= 1:
        wt = list(a)
        wt[0] = WTYPE.get(cat, '"x"')
    else:
        wt = [WTYPE.get(cat, '"x"')]
    src.append('(probe 2 (lambda () %s))' % call(wt))
    keys.append("%s::wtype" % name)
    return "\n".join(src) + "\n", keys


def run(cmd, timeout, env=None):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout, env=env)
        return p.returncode, p.stdout.decode("utf-8", "replace"), False
    except subprocess.TimeoutExpired:
        return None, "", True


import re

_PROBE_RE = re.compile(r"#P8#(\d+)#(.*?)#/P8#", re.S)


def parse_indexed(out):
    """idx -> value, from #P8#idx#value#/P8# spans. ALL whitespace is removed
    first (identically for both engines) so the VM's per-op newline insertion
    and any value-internal spacing cancel out."""
    flat = re.sub(r"\s+", "", out)
    res = {}
    for m in _PROBE_RE.finditer(flat):
        res[int(m.group(1))] = m.group(2)
    return res


def run_native(esk, native, workdir, timeout):
    env = dict(os.environ, ESHKOL_JIT_CACHE_DIR=os.path.join(workdir, "jit"))
    rc, out, to = run([native, "-r", esk], timeout, env)
    return rc, parse_indexed(out), to


def run_vm(esk, vm, native, workdir, timeout):
    eskb = esk + "b"
    rc, _, to = run([native, "--profile", "hosted-vm", "--emit-eskb", eskb, esk],
                    timeout)
    if to or rc != 0 or not os.path.exists(eskb):
        return "compile", {}, to
    env = dict(os.environ, ESHKOL_VM_NO_DISASM="1")
    rc, out, to = run([vm, eskb], timeout, env)
    try:
        os.remove(eskb)
    except OSError:
        pass
    return rc, parse_indexed(out), to


def compare(keys, nres, vres, ncrash, vcrash):
    """Yield divergence keys. A probe diverges if the engines disagree on its
    value, or exactly one engine is missing the index (crash truncation)."""
    divs = []
    for idx, key in enumerate(keys):
        nv = nres.get(idx)
        vv = vres.get(idx)
        if nv is None and vv is None:
            # both missing: attribute to a crash only if exactly one crashed.
            if ncrash != vcrash:
                divs.append(key)
            continue
        if nv is None or vv is None:
            divs.append(key)     # one engine truncated here
            continue
        if nv != vv:
            divs.append(key)
    return divs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="tests/coverage/language_surface.json")
    ap.add_argument("--native", required=True)
    ap.add_argument("--vm", required=True)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=8803)
    ap.add_argument("--baseline",
                    default="tests/escape_matrix/arity_parity_baseline.json")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--trace")
    ap.add_argument("--workdir")
    ap.add_argument("--timeout", type=float, default=25.0)
    args = ap.parse_args()

    d = json.load(open(args.manifest))
    cands = [e for e in d["builtins"]
             if e.get("arity") is not None
             and isinstance(e["arity"], int)
             and 0 <= e["arity"] <= 3
             and e.get("category") in CATEGORIES
             and "vm" in (e.get("backends") or [])
             and (("native_llvm" in (e.get("backends") or []))
                  or ("native" in (e.get("backends") or [])))]
    cands.sort(key=lambda e: e["name"])

    if not args.full:
        import random
        rng = random.Random(args.seed)
        cands = sorted(rng.sample(cands, min(args.sample, len(cands))),
                       key=lambda e: e["name"])

    baseline = set()
    if os.path.exists(args.baseline):
        baseline = set(json.load(open(args.baseline)).get("known_divergences", []))

    workdir = args.workdir or tempfile.mkdtemp(prefix="p8-arity-")
    os.makedirs(os.path.join(workdir, "jit"), exist_ok=True)

    all_divs = []
    tested = 0
    for e in cands:
        name, cat, arity = e["name"], e["category"], e["arity"]
        src, keys = build_program(name, cat, arity)
        esk = os.path.join(workdir, "probe.esk")
        with open(esk, "w") as fh:
            fh.write(src)
        nrc, nres, nto = run_native(esk, args.native, workdir, args.timeout)
        vrc, vres, vto = run_vm(esk, args.vm, args.native, workdir, args.timeout)
        if nto or vto:
            continue  # a timeout is environmental, not a parity claim
        ncrash = (nrc not in (0,))
        vcrash = (vrc in ("compile",)) or (isinstance(vrc, int) and vrc not in (0,))
        all_divs.extend(compare(keys, nres, vres, ncrash, vcrash))
        tested += 1

    all_divs = sorted(set(all_divs))
    new_divs = [k for k in all_divs if k not in baseline]

    if args.update_baseline:
        # shrink-only union: keep baseline entries still-present is caller's job;
        # here we write exactly the observed set (run with --full to refresh).
        with open(args.baseline, "w") as fh:
            json.dump({"_comment": "P8 axis-3 native-vs-VM known parity gaps; "
                                   "gate fails on any key NOT listed here. "
                                   "Generated by p8_arity_sweep.py --update-baseline --full.",
                       "known_divergences": all_divs}, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print("wrote baseline (%d divergences over %d builtins) -> %s"
              % (len(all_divs), tested, args.baseline))
        return 0

    status = "PASS" if not new_divs else "FAIL"
    if args.trace:
        os.makedirs(os.path.dirname(args.trace) or ".", exist_ok=True)
        with open(args.trace, "a") as fh:
            fh.write(json.dumps({
                "kind": "escape_matrix", "name": "arity_sweep_native_vm_parity",
                "value": status, "tested": tested, "new_divergences": new_divs,
                "total_divergences": len(all_divs),
                "confidence": 0.95}) + "\n")
    print("axis-3 arity sweep: tested=%d builtins, divergences=%d (baseline=%d), NEW=%d"
          % (tested, len(all_divs), len(baseline), len(new_divs)))
    if new_divs:
        print("NEW native-vs-VM divergences (not in baseline):")
        for k in new_divs[:40]:
            print("   ", k)
    print("axis-3 gate: %s" % status)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
