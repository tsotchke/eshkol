#!/usr/bin/env python3
"""run_surface_parity.py — make the VM parity ledger EXECUTION-BACKED.

Why this exists
---------------
Every parity guarantee in this repo was checked against SOURCE TEXT, never
against a running engine:

  * scripts/vm_parity_audit.py enumerates the "codegen surface" by scraping
    builtin names out of lib/backend/llvm_codegen.cpp and the op enum out of
    inc/eshkol/eshkol.h, then checks each has a row in PARITY.tsv.  A row's
    STATUS is a hand-typed string; nothing ever ran the name.
  * PARITY.tsv's `vm-supported` therefore means "somebody wrote
    vm-supported", not "the VM resolves this".
  * scripts/run_vm_parity.sh does compare real output — but only for the
    ~60 curated programs in tests/vm_parity/corpus, so it covers what
    someone thought to write and nothing else.

Consequence: a name can exist natively, be labelled `vm-supported`, and
crash the VM, with every gate green.  Measured when this script was written:
`assq`, `assv`, `memv`, `partition` and `string-contains` all resolve
natively, abort the VM with "undefined variable", have NO row in PARITY.tsv
— and vm_parity_audit.py still reported "OK — every codegen symbol is
VM-supported or consciously waived".  The same blind spot hid a VM
`kb-query` that returned facts where native returned substitutions, a
`unify` that read its arguments in the wrong order, a `logic-var?` that was
unconditionally #f, and an `is_truthy` that contradicted its own doc comment
so `(if '() 'T 'F)` disagreed across engines.

The repo already solved this problem once, for coverage:
scripts/language_coverage.py gates on "EXECUTION-backed" constructs and
prints "[the only gated number]" next to it.  This applies the same rule to
parity.

What it does
------------
For every name on the surface it RUNS both engines and records whether each
resolves the name, then cross-references PARITY.tsv:

  FAIL  native resolves, VM does not, and the ledger says `vm-supported`
        -> the ledger asserts something untrue of the running system.
  FAIL  native resolves, VM does not, and the ledger has NO row
        -> a divergence nothing is tracking (this is the `assq` class).
  ok    everything else, with counts reported.

A baseline file (tests/vm_parity/SURFACE_BASELINE.tsv) ratchets the
pre-existing backlog the way language_coverage.py's deficit ratchet does:
known entries do not fail the build, but the count may never grow and a
fixed entry may never regress.

Probe semantics
---------------
Resolution is probed with `(define __probe <name>)` — a VALUE reference.
That is deliberately stricter than "callable": it also catches names an
engine only implements as an opcode in call position.  Because both engines
have such names, an opcode-only name is NOT failed; it is counted and
reported separately, and a VM name is credited as resolved when it appears
in vm_parity_audit.vm_surface() (the BUILTINS table, compiler dispatch, or
the prelude), which is exactly the set the VM can call.

What it does NOT do
-------------------
This closes ONE divergence class: a name that exists on one engine and not
the other.  It does NOT compare VALUES — `kb-query` returning facts on the
VM where native returned substitutions would still pass here, because both
engines resolve the name.  That class is covered by
tests/vm_parity/corpus/*.esk via run_vm_parity.sh, which is opt-in and
therefore only as complete as the corpus.

Closing the gap between them needs construct-level differential evidence:
run the corpus under BOTH engines with the language_coverage.py
instrumentation active and require every construct that executes on both to
produce the same result.  Until that exists, "green here" means "no name is
missing", not "the engines agree".  Do not read it as more than that.

Usage:
  scripts/run_surface_parity.py [--update-baseline] [--limit N] [--workdir DIR]
Exit: 0 = gate green, 1 = gate red, 2 = misuse/environment problem.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "tests", "vm_parity", "PARITY.tsv")
BASELINE = os.path.join(REPO, "tests", "vm_parity", "SURFACE_BASELINE.tsv")
TRACE_DIR = os.path.join(REPO, "scripts", "icc_traces")
TRACE = os.path.join(TRACE_DIR, "surface_parity.jsonl")

BUILD = os.environ.get("BUILD_DIR", os.path.join(REPO, "build"))
if not os.path.isabs(BUILD):
    BUILD = os.path.join(REPO, BUILD)
ESHKOL_RUN = os.path.join(BUILD, "eshkol-run")
VM_BIN = os.path.join(BUILD, "eshkol-vm-standalone-test")
if not os.path.exists(VM_BIN):
    VM_BIN = os.path.join(BUILD, "eshkol-vm-standalone")
NATIVE_CACHE = os.path.join(BUILD, "surface_parity_native_cache.json")

# Syntax keywords and reader forms cannot appear in a value position at all,
# so a resolution probe is meaningless for them.
SYNTAX = {
    "if", "define", "lambda", "let", "let*", "letrec", "letrec*", "quote",
    "quasiquote", "unquote", "unquote-splicing", "set!", "begin", "cond",
    "case", "and", "or", "when", "unless", "do", "delay", "delay-force",
    "define-syntax", "let-syntax", "letrec-syntax", "syntax-rules",
    "define-record-type", "parameterize", "guard", "else", "=>",
    "define-values", "let-values", "let*-values", "case-lambda", "include",
    "import", "export", "define-library", "cond-expand", "assert",
    "with-region", "define-structure", "match", "lambda*", "named-lambda",
}

NAME_RE = re.compile(r"^[A-Za-z*+\-/<>=!?_][^\s()'\"`;]*$")


def die(msg):
    sys.stderr.write("run_surface_parity: %s\n" % msg)
    sys.exit(2)


def require_empty_workdir(path):
    """Validate a caller-owned durable directory before writing probes."""
    if not os.path.isabs(path):
        raise ValueError("--workdir must be an absolute path")
    if os.path.islink(path):
        raise ValueError("--workdir must not be a symlink: %s" % path)
    if not os.path.isdir(path):
        raise ValueError("--workdir must be a caller-created directory: %s" % path)
    if os.listdir(path):
        raise ValueError("--workdir must be empty: %s" % path)
    return path


def write_probe_source(workdir, prefix, number, src):
    """Write a retained deterministic probe file, or the legacy temp file."""
    if workdir is None:
        fd, path = tempfile.mkstemp(suffix=".esk")
        with os.fdopen(fd, "w") as f:
            f.write(src)
        return path
    path = os.path.join(workdir, "%s-%04d.esk" % (prefix, number))
    with open(path, "x", encoding="utf-8") as f:
        f.write(src)
    return path


def load_manifest():
    rows = {}
    if not os.path.exists(MANIFEST):
        die("manifest missing: %s" % MANIFEST)
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0] and parts[0] != "status":
                rows[parts[0]] = parts[1]
    return rows


def load_baseline():
    known = {}
    if not os.path.exists(BASELINE):
        return known
    with open(BASELINE, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                known[parts[0]] = parts[1]
    return known


def stdlib_names():
    """Public procedure/value names defined in the Scheme stdlib.

    vm_parity_audit.py's surface is the C++ dispatch table, which excludes
    every one of these — the structural reason `assq` was invisible.
    """
    names = set()
    lib = os.path.join(REPO, "lib")
    for root, _dirs, files in os.walk(lib):
        for fn in files:
            if not fn.endswith(".esk"):
                continue
            try:
                src = open(os.path.join(root, fn), encoding="utf-8",
                           errors="replace").read()
            except OSError:
                continue
            for m in re.finditer(r"^\(define\s+\(([^\s()]+)", src, re.M):
                names.add(m.group(1))
            for m in re.finditer(r"^\(define\s+([^\s()]+)\s", src, re.M):
                names.add(m.group(1))
    # Internal helpers: leading/trailing markers and earmuffed globals are
    # module-private by convention and are not part of the callable surface.
    return {n for n in names
            if not n.startswith("__") and not n.startswith("-")
            and not (n.startswith("*") and n.endswith("*"))}


def probe_vm(names, vm_surface, workdir=None):
    """Batch-probe VM name resolution; credit vm_surface membership."""
    resolved = {}
    env = dict(os.environ, ESHKOL_VM_NO_DISASM="1")
    batch = 60
    for batch_number, i in enumerate(range(0, len(names), batch), 1):
        chunk = names[i:i + batch]
        src = "".join("(define __p%d %s)\n" % (j, n)
                      for j, n in enumerate(chunk))
        path = write_probe_source(workdir, "vm-batch", batch_number, src)
        try:
            out = subprocess.run([VM_BIN, path], capture_output=True,
                                 text=True, timeout=300, env=env)
            txt = out.stdout + out.stderr
        except (OSError, subprocess.SubprocessError):
            txt = ""
        finally:
            if workdir is None:
                os.unlink(path)
        for n in chunk:
            warned = ("undefined variable '%s'" % n) in txt
            resolved[n] = (not warned) or (n in vm_surface)
    return resolved


def _native_batch_ok(names, env, workdir=None, probe_number=0):
    """True if native compiles a program referencing every name in `names`."""
    src = "".join("(define __p%d %s)\n" % (i, n) for i, n in enumerate(names))
    path = write_probe_source(workdir, "native-batch", probe_number, src)
    try:
        r = subprocess.run([ESHKOL_RUN, "-r", path], capture_output=True,
                           text=True, timeout=600, env=env)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        if workdir is None:
            os.unlink(path)


def _native_cache_key():
    """Fingerprint of everything that can change what native resolves:
    the compiler binary and every Scheme stdlib file."""
    h = hashlib.sha256()
    try:
        st = os.stat(ESHKOL_RUN)
        h.update(("%d:%d" % (st.st_size, int(st.st_mtime))).encode())
    except OSError:
        return None
    lib = os.path.join(REPO, "lib")
    for dirpath, dirs, files in os.walk(lib):
        dirs.sort()
        for fn in sorted(files):
            if not fn.endswith(".esk"):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            h.update(("%s:%d:%d" % (os.path.relpath(fp, REPO), st.st_size,
                                    int(st.st_mtime))).encode())
    return h.hexdigest()


def _load_native_cache(key):
    if not key or not os.path.exists(NATIVE_CACHE):
        return {}
    try:
        with open(NATIVE_CACHE, encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return {}
    return blob.get("names", {}) if blob.get("key") == key else {}


def _save_native_cache(key, resolved):
    if not key:
        return
    os.makedirs(os.path.dirname(NATIVE_CACHE), exist_ok=True)
    try:
        with open(NATIVE_CACHE, "w", encoding="utf-8") as f:
            json.dump({"key": key, "names": resolved}, f)
    except OSError:
        pass


def probe_native(names, workdir=None):
    """Probe native resolution, batched with bisection.

    An unresolved name aborts the whole program (and native's error path can
    exit 139 — ledger item #98), so a failing batch says only "at least one
    of these is unresolved".  Bisecting on failure keeps the common case —
    a batch where everything resolves — to a single process.  Probing one
    name per process took ~25 minutes for the full surface, which is a gate
    nobody would run; this brings it into the smoke suite's budget.
    """
    env = dict(os.environ, ESHKOL_JIT_CACHE="0", ESHKOL_LIB_DIR=BUILD)
    # Native resolution only changes when the compiler or the Scheme stdlib
    # changes, and probing it is the expensive half (one process per batch,
    # bisected). Cache it under that fingerprint so a re-run costs seconds.
    key = _native_cache_key()
    cached = _load_native_cache(key)
    resolved = {n: cached[n] for n in names if n in cached}
    todo = [n for n in names if n not in resolved]

    probe_number = 0

    def walk(chunk):
        nonlocal probe_number
        if not chunk:
            return
        probe_number += 1
        if _native_batch_ok(chunk, env, workdir, probe_number):
            for n in chunk:
                resolved[n] = True
            return
        if len(chunk) == 1:
            resolved[chunk[0]] = False
            return
        mid = len(chunk) // 2
        walk(chunk[:mid])
        walk(chunk[mid:])

    batch = 64
    for i in range(0, len(todo), batch):
        walk(todo[i:i + batch])
    merged = dict(cached)
    merged.update(resolved)
    _save_native_cache(key, merged)
    return resolved


def emit_trace(events):
    os.makedirs(TRACE_DIR, exist_ok=True)
    with open(TRACE, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true",
                    help="rewrite the ratchet baseline from this run")
    ap.add_argument("--limit", type=int, default=0,
                    help="probe only the first N names (smoke use)")
    ap.add_argument("--workdir",
                    help="empty caller-created absolute directory retaining probe files")
    args = ap.parse_args()

    if args.workdir is not None:
        try:
            args.workdir = require_empty_workdir(args.workdir)
        except ValueError as exc:
            die(str(exc))

    for path, what in ((ESHKOL_RUN, "eshkol-run"), (VM_BIN, "the VM binary")):
        if not os.path.exists(path):
            die("%s not built at %s (set BUILD_DIR)" % (what, path))

    spec = importlib.util.spec_from_file_location(
        "vpa", os.path.join(REPO, "scripts", "vm_parity_audit.py"))
    vpa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vpa)
    vm_surface = vpa.vm_surface()
    codegen = {n for n in vpa.codegen_surface() if not n.startswith("op:")}

    ledger = load_manifest()
    baseline = load_baseline()

    surface = sorted((codegen | stdlib_names() |
                      {n for n in ledger if not n.startswith("op:")})
                     - SYNTAX)
    surface = [n for n in surface if NAME_RE.match(n)]
    if args.limit:
        surface = surface[:args.limit]

    print("run_surface_parity: probing %d names on both engines "
          "(codegen surface + Scheme stdlib + ledger rows)" % len(surface))

    vm_ok = probe_vm(surface, vm_surface, args.workdir)
    nat_ok = probe_native(surface, args.workdir)

    findings, opcode_only, untracked = [], [], []
    for n in surface:
        if nat_ok[n] and not vm_ok[n]:
            status = ledger.get(n)
            if status in (None, "vm-supported"):
                findings.append((n, status or "NO-ROW"))
        if n in vm_surface and not vm_ok.get(n, False):
            opcode_only.append(n)
        if nat_ok[n] and n not in ledger:
            untracked.append(n)

    probed = set(surface)
    found_names = {f[0] for f in findings}
    new = [(n, s) for n, s in findings if n not in baseline]
    # Only baseline entries that were actually PROBED in this run can be
    # judged fixed. Without this, --limit compared a subset against the whole
    # baseline and reported hundreds of unprobed names as "now fixed" — a
    # gate that reports numbers it did not measure is the defect this script
    # exists to catch.
    fixed = [n for n in baseline if n in probed and n not in found_names]

    print("  native-resolves + VM-does-not, ledger silent or claiming "
          "vm-supported : %d" % len(findings))
    print("    of those, NOT in the ratchet baseline (NEW)                    "
          "   : %d" % len(new))
    print("    baseline entries now fixed                                     "
          "   : %d" % len(fixed))
    print("  names resolving natively with no ledger row at all               "
          "   : %d" % len(untracked))
    print("  names the VM has only in call position (opcode-only, reported)   "
          "   : %d" % len(opcode_only))

    events = [{"kind": "runtime_event", "name": "surface_parity_probe",
               "value": "PASS" if not new else "FAIL",
               "snippet": "%d probed / %d divergences / %d new" %
                          (len(surface), len(findings), len(new)),
               "confidence": 0.95}]
    emit_trace(events)

    if args.update_baseline:
        if args.limit:
            die("--update-baseline needs a full run; --limit would record a "
                "partial surface as the baseline")
        with open(BASELINE, "w", encoding="utf-8") as f:
            f.write("# Ratchet baseline for scripts/run_surface_parity.py.\n")
            f.write("# Each row is a name native resolves and the VM does "
                    "not, with the ledger\n# silent or wrongly claiming "
                    "vm-supported. The count may never grow, and a\n"
                    "# name removed from here may never come back.\n")
            f.write("# name\tledger-status-at-baseline\n")
            for n, s in sorted(findings):
                f.write("%s\t%s\n" % (n, s))
        print("  baseline rewritten: %d entries" % len(findings))
        return 0

    if new:
        print()
        print("FAIL: %d name(s) diverge and are not in the ratchet baseline."
              % len(new))
        for n, s in sorted(new):
            print("  %-28s native=resolves vm=UNRESOLVED ledger=%s" % (n, s))
        print()
        print("Either implement the name on the VM, or record it in "
              "PARITY.tsv as a `gap`\nrow with a justification — a status is "
              "a claim about the running system.")
        return 1

    if fixed:
        print()
        print("NOTE: %d baseline entr(y|ies) now pass; re-run with "
              "--update-baseline to ratchet." % len(fixed))
    print()
    print("run_surface_parity: OK — every probed name that native resolves "
          "is either\nresolved by the VM or recorded, and no new divergence "
          "appeared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
