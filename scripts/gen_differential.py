#!/usr/bin/env python3
"""gen_differential.py — seeded random Eshkol program generator + differential
runner + shrinker (adversarial testing campaign, pillar P1).

Generates bounded-depth, always-terminating, deterministic programs over the
core forms (mixed exact/inexact arithmetic, let/let*/letrec, lambda +
application, if/cond, quote, list/vector ops, string ops, set! on locals,
bounded named-let loops), runs each on every native execution axis
(jit, jit-nocache, aot-o0, aot-o2), and diffs (exit code, normalized stdout)
pairwise. Identical => PASS. Any divergence is a compiler bug by definition.

On divergence the program is SHRUNK: subtrees are iteratively deleted or
replaced with same-type literals while the divergence persists; the minimal
repro is saved to tests/differential/found/NNN_shrunk.esk with a header
documenting seed, program index, and the per-axis outputs.

Usage:
  python3 scripts/gen_differential.py --seed 42 --count 200 \
      [--build-dir build] [--found-dir tests/differential/found] \
      [--timeout 60] [--max-findings 10] [--emit-only DIR]

Normalization here MUST stay in sync with normalize() in
scripts/run_differential.sh.
"""

import argparse
import hashlib
import os
import random
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# S-expression AST: nested python lists, leaves are strings.
# ---------------------------------------------------------------------------

def render(node):
    if isinstance(node, str):
        return node
    return "(" + " ".join(render(c) for c in node) + ")"

def render_program(forms):
    return "\n".join(render(f) for f in forms) + "\n"

# ---------------------------------------------------------------------------
# Typed random generator.
# Types: 'int', 'float', 'bool', 'str', 'list' (list of ints), 'vec'
# (vector of ints, meta = static length), plus static-length metadata for
# safe substring / vector-ref / car access.
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, rng, env=None, depth=0, max_depth=5):
        self.rng = rng
        self.env = env if env is not None else []  # (name, type, meta)
        self.depth = depth
        self.max_depth = max_depth
        self.counter = [0]

    def child(self):
        c = Ctx(self.rng, list(self.env), self.depth + 1, self.max_depth)
        c.counter = self.counter
        return c

    def fresh(self, prefix="v"):
        self.counter[0] += 1
        return "%s%d" % (prefix, self.counter[0])

    def vars_of(self, ty):
        return [(n, m) for (n, t, m) in self.env if t == ty]

def gen_int_literal(ctx):
    return str(ctx.rng.randint(-50, 50)), None

def gen_float_literal(ctx):
    v = ctx.rng.choice([0.5, 1.5, -2.5, 0.25, 3.0, -0.125, 10.0, 0.1])
    return repr(v), None

STR_ALPHABET = ["ab", "xyz", "q", "hello", "Zw", ""]

def gen_str_literal(ctx):
    s = ctx.rng.choice(STR_ALPHABET)
    return '"%s"' % s, len(s)

def gen(ctx, ty):
    """Generate an expression of type ty; returns (node, meta)."""
    rng = ctx.rng
    leafy = ctx.depth >= ctx.max_depth or rng.random() < 0.18
    vs = ctx.vars_of(ty)
    if vs and rng.random() < 0.35:
        n, m = rng.choice(vs)
        return n, m
    if ty == "int":
        if leafy:
            return gen_int_literal(ctx)
        return gen_int_expr(ctx)
    if ty == "float":
        if leafy:
            return gen_float_literal(ctx)
        return gen_float_expr(ctx)
    if ty == "bool":
        if leafy:
            return rng.choice(["#t", "#f"]), None
        return gen_bool_expr(ctx)
    if ty == "str":
        if leafy:
            return gen_str_literal(ctx)
        return gen_str_expr(ctx)
    if ty == "list":
        if leafy:
            k = rng.randint(0, 3)
            items = [gen_int_literal(ctx)[0] for _ in range(k)]
            return ["list"] + items, k
        return gen_list_expr(ctx)
    if ty == "vec":
        k = rng.randint(1, 4)
        items = [gen(ctx.child(), "int")[0] for _ in range(k)]
        return ["vector"] + items, k
    raise AssertionError(ty)

def wrap_binding_forms(ctx, ty, body_gen):
    """let / let* / letrec / lambda-app / set!-sequence around a body of type ty."""
    rng = ctx.rng
    kind = rng.choice(["let", "let*", "letrec", "lambda-app", "set-seq"])
    c = ctx.child()
    nb = rng.randint(1, 2)
    binds = []
    names = []
    for _ in range(nb):
        bty = rng.choice(["int", "int", "float", "bool", "str", "list"])
        val, meta = gen(c, bty)
        name = c.fresh()
        binds.append([name, val])
        names.append((name, bty, meta))
    if kind == "lambda-app":
        for (n, t, m) in names:
            c.env.append((n, t, m))
        body, bmeta = body_gen(c)
        lam = ["lambda", [n for (n, _, _) in names], body]
        return [lam] + [b[1] for b in binds], bmeta
    if kind == "set-seq":
        # (let ((x e)) (begin (set! x e2) body))
        (n, t, m) = names[0]
        c.env.append((n, t, None))  # meta invalid after set!
        val2, _ = gen(c, t)
        body, bmeta = body_gen(c)
        return ["let", [binds[0]], ["begin", ["set!", n, val2], body]], bmeta
    if kind in ("let", "let*", "letrec"):
        for (n, t, m) in names:
            c.env.append((n, t, m))
        body, bmeta = body_gen(c)
        return [kind, binds, body], bmeta
    raise AssertionError(kind)

def gen_int_expr(ctx):
    rng = ctx.rng
    choice = rng.randint(0, 11)
    c = ctx.child()
    if choice <= 2:
        op = rng.choice(["+", "-", "*"])
        a, _ = gen(c, "int"); b, _ = gen(c, "int")
        return [op, a, b], None
    if choice == 3:
        op = rng.choice(["quotient", "remainder", "modulo"])
        a, _ = gen(c, "int")
        d = str(rng.choice([2, 3, 5, 7, -4]))
        return [op, a, d], None
    if choice == 4:
        op = rng.choice(["min", "max"])
        a, _ = gen(c, "int"); b, _ = gen(c, "int")
        return [op, a, b], None
    if choice == 5:
        a, _ = gen(c, "int")
        return ["abs", a], None
    if choice == 6:
        b, _ = gen(c, "bool")
        t, _ = gen(c, "int"); e, _ = gen(c, "int")
        return ["if", b, t, e], None
    if choice == 7:
        lst, _ = gen(c, "list")
        return ["length", lst], None
    if choice == 8:
        s, _ = gen(c, "str")
        return ["string-length", s], None
    if choice == 9:
        v, k = gen(c, "vec")
        idx = rng.randint(0, (k or 1) - 1)
        return ["vector-ref", v, str(idx)], None
    if choice == 10:
        # bounded named-let accumulator loop
        n = rng.randint(1, 12)
        loop = c.fresh("loop")
        i = c.fresh("i")
        acc = c.fresh("acc")
        c2 = c.child()
        c2.env.append((i, "int", None))
        c2.env.append((acc, "int", None))
        step, _ = gen(c2, "int")
        init, _ = gen(c, "int")
        return ["let", loop, [[i, "0"], [acc, init]],
                ["if", [">=", i, str(n)], acc,
                 [loop, ["+", i, "1"], ["+", acc, step]]]], None
    return wrap_binding_forms(ctx, "int", lambda c2: gen(c2, "int"))

def gen_float_expr(ctx):
    rng = ctx.rng
    choice = rng.randint(0, 5)
    c = ctx.child()
    if choice <= 1:
        op = rng.choice(["+", "-", "*"])
        a, _ = gen(c, "float"); b, _ = gen(c, "float")
        return [op, a, b], None
    if choice == 2:
        op = rng.choice(["+", "*", "-"])
        a, _ = gen(c, "int"); b, _ = gen(c, "float")
        pair = [a, b] if rng.random() < 0.5 else [b, a]
        return [op] + pair, None
    if choice == 3:
        a, _ = gen(c, "float")
        return ["*", "0.5", ["abs", a]], None
    if choice == 4:
        b, _ = gen(c, "bool")
        t, _ = gen(c, "float"); e, _ = gen(c, "float")
        return ["if", b, t, e], None
    return wrap_binding_forms(ctx, "float", lambda c2: gen(c2, "float"))

def gen_bool_expr(ctx):
    rng = ctx.rng
    choice = rng.randint(0, 5)
    c = ctx.child()
    if choice <= 1:
        op = rng.choice(["<", ">", "<=", ">=", "="])
        a, _ = gen(c, "int"); b, _ = gen(c, "int")
        return [op, a, b], None
    if choice == 2:
        op = rng.choice(["and", "or"])
        a, _ = gen(c, "bool"); b, _ = gen(c, "bool")
        return [op, a, b], None
    if choice == 3:
        a, _ = gen(c, "bool")
        return ["not", a], None
    if choice == 4:
        lst, _ = gen(c, "list")
        return ["null?", lst], None
    s1, _ = gen(c, "str"); s2, _ = gen(c, "str")
    return ["string=?", s1, s2], None

def gen_str_expr(ctx):
    rng = ctx.rng
    choice = rng.randint(0, 3)
    c = ctx.child()
    if choice == 0:
        a, la = gen(c, "str"); b, lb = gen(c, "str")
        meta = (la + lb) if (la is not None and lb is not None) else None
        return ["string-append", a, b], meta
    if choice == 1:
        a, _ = gen(c, "int")
        return ["number->string", a], None
    if choice == 2:
        a, la = gen(c, "str")
        if la is not None and la >= 1:
            j = rng.randint(1, la)
            i = rng.randint(0, j)
            return ["substring", a, str(i), str(j)], j - i
        return ["string-append", a, '"z"'], None
    b, _ = gen(c, "bool")
    t, lt = gen(c, "str"); e, le = gen(c, "str")
    return ["if", b, t, e], None

def gen_list_expr(ctx):
    rng = ctx.rng
    choice = rng.randint(0, 5)
    c = ctx.child()
    if choice == 0:
        a, la = gen(c, "list"); b, lb = gen(c, "list")
        meta = (la + lb) if (la is not None and lb is not None) else None
        return ["append", a, b], meta
    if choice == 1:
        a, la = gen(c, "list")
        return ["reverse", a], la
    if choice == 2:
        x, _ = gen(c, "int")
        a, la = gen(c, "list")
        return ["cons", x, a], (la + 1) if la is not None else None
    if choice == 3:
        a, la = gen(c, "list")
        if la is not None and la >= 1:
            return ["cdr", a], la - 1
        x, _ = gen(c, "int")
        return ["cdr", ["cons", x, a]], la
    if choice == 4:
        k = rng.randint(0, 4)
        items = [gen(c, "int")[0] for _ in range(k)]
        return ["list"] + items, k
    q = [str(rng.randint(-9, 9)) for _ in range(rng.randint(0, 3))]
    return "'(" + " ".join(q) + ")", len(q)

DISPLAY_TYPES = ["int", "int", "float", "bool", "str", "list", "vec"]

def gen_program(seed, index, max_depth=5):
    rng = random.Random((seed << 20) ^ index)
    ctx = Ctx(rng, max_depth=max_depth)
    forms = []
    # 0-3 global defines (values and single-arg functions)
    for _ in range(rng.randint(0, 3)):
        if rng.random() < 0.4:
            # function define: int -> int
            fname = ctx.fresh("f")
            c = ctx.child()
            pname = c.fresh("p")
            c.env.append((pname, "int", None))
            body, _ = gen(c, "int")
            forms.append(["define", [fname, pname], body])
            ctx.env.append((fname, "fn-int", None))
        else:
            ty = rng.choice(["int", "float", "str", "list", "bool"])
            val, meta = gen(ctx.child(), ty)
            name = ctx.fresh("g")
            forms.append(["define", name, val])
            ctx.env.append((name, ty, meta))
    # 1-4 display statements
    for _ in range(rng.randint(1, 4)):
        ty = rng.choice(DISPLAY_TYPES)
        c = ctx.child()
        # let function-typed globals participate
        fns = [n for (n, t, _) in ctx.env if t == "fn-int"]
        if ty == "int" and fns and rng.random() < 0.5:
            arg, _ = gen(c, "int")
            expr = [rng.choice(fns), arg]
        else:
            expr, _ = gen(c, ty)
        forms.append(["display", expr])
        forms.append(["newline"])
    return forms

# ---------------------------------------------------------------------------
# Differential runner (mirrors scripts/run_differential.sh).
# ---------------------------------------------------------------------------

NOISE = re.compile(
    r"^(WARN|INFO:|DEBUG|\[ESKB\]|\s*\[compiled:|=== Eshkol VM"
    r"|=== Execution complete ===|remark:|warning: <unknown>)")

def normalize(text):
    lines = [l for l in text.splitlines() if not NOISE.match(l)]
    while lines and not lines[0].strip():
        lines.pop(0)  # keep in sync with run_differential.sh normalize()
    return "\n".join(lines)

class Runner:
    AXES = ["jit", "jit-nocache", "aot-o0", "aot-o2"]

    def __init__(self, build_dir, workdir, timeout):
        self.eshkol_run = os.path.join(build_dir, "eshkol-run")
        self.workdir = workdir
        self.timeout = timeout
        self.cache_dir = os.path.join(workdir, "jit-cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.evals = 0

    def _run(self, cmd, env_extra=None):
        env = dict(os.environ)
        env["ESHKOL_JIT_CACHE_DIR"] = self.cache_dir
        if env_extra:
            env.update(env_extra)
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=self.timeout, env=env)
            return p.returncode, p.stdout
        except subprocess.TimeoutExpired:
            return 124, "<TIMEOUT>"
        except UnicodeDecodeError:
            return 125, "<UNDECODABLE>"

    def run_axes(self, source):
        """Returns {axis: (rc, normalized_stdout)}."""
        self.evals += 1
        path = os.path.join(self.workdir, "prog.esk")
        with open(path, "w") as f:
            f.write(source)
        res = {}
        rc, out = self._run([self.eshkol_run, "-r", path])
        res["jit"] = (rc, normalize(out))
        rc, out = self._run([self.eshkol_run, "-r", path],
                            {"ESHKOL_JIT_CACHE": "0"})
        res["jit-nocache"] = (rc, normalize(out))
        for axis, olvl in (("aot-o0", "-O0"), ("aot-o2", "-O2")):
            binpath = os.path.join(self.workdir, "prog-" + axis + ".bin")
            if os.path.exists(binpath):
                os.unlink(binpath)
            crc, _ = self._run([self.eshkol_run, olvl, path, "-o", binpath])
            if crc != 0 or not os.path.exists(binpath):
                res[axis] = (125, "<COMPILE-FAIL rc=%d>" % crc)
            else:
                rc, out = self._run([binpath])
                res[axis] = (rc, normalize(out))
        return res

def make_divergence_predicate(runner, orig_res):
    """Predicate for the shrinker. A candidate must still DIVERGE, and must
    not have drifted into a merely-invalid program: if the original program
    ran cleanly (rc==0) on at least one axis, every accepted candidate must
    too. Otherwise shrinking converges on e.g. an undefined-variable snippet
    whose per-path error reporting trivially 'diverges' (JIT: runtime error
    rc=1; AOT: compile fail rc=125) — a different, uninteresting program."""
    orig_had_ok = any(rc == 0 for (rc, _) in orig_res.values())

    def is_divergent(source):
        r = runner.run_axes(source)
        if classify(r)[0] != "DIVERGE":
            return False
        if orig_had_ok and not any(rc == 0 for (rc, _) in r.values()):
            return False
        return True

    return is_divergent

def classify(res):
    """Returns (status, detail): status in {AGREE, DIVERGE, ALL-FAIL}."""
    vals = list(res.values())
    if all(v == vals[0] for v in vals):
        if vals[0][0] != 0:
            return "ALL-FAIL", "all axes rc=%d" % vals[0][0]
        return "AGREE", ""
    axes = list(res.keys())
    pairs = []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            if res[axes[i]] != res[axes[j]]:
                pairs.append("%s-vs-%s" % (axes[i], axes[j]))
    return "DIVERGE", ",".join(pairs)

# ---------------------------------------------------------------------------
# Shrinker: greedy subtree deletion / literal replacement while the program
# still diverges.
# ---------------------------------------------------------------------------

LITERAL_CANDIDATES = ["0", "1", "#t", "#f", '""', "'()", "0.5"]

def paths(node, prefix=()):
    """Yield all paths to subnodes (lists only are recursed)."""
    yield prefix, node
    if isinstance(node, list):
        for i, c in enumerate(node):
            yield from paths(c, prefix + (i,))

def get_at(node, path):
    for i in path:
        node = node[i]
    return node

def replace_at(root, path, new):
    if not path:
        return new
    root = list(root)
    root[path[0]] = replace_at(root[path[0]], path[1:], new)
    return root

def shrink(forms, runner, is_divergent, budget=300):
    """Greedy shrink; is_divergent(source)->bool. Returns minimal forms."""
    def still_diverges(fs):
        if not fs:
            return False
        return is_divergent(render_program(fs))

    changed = True
    while changed and runner.evals < budget:
        changed = False
        # pass 1: drop whole top-level forms
        i = 0
        while i < len(forms) and runner.evals < budget:
            trial = forms[:i] + forms[i + 1:]
            if trial and still_diverges(trial):
                forms = trial
                changed = True
            else:
                i += 1
        # pass 2: replace subexpressions with literals (deepest-first
        # tends to lock in too early; go largest-first: shallow paths first)
        f = 0
        while f < len(forms) and runner.evals < budget:
            all_paths = [p for p, n in paths(forms[f])
                         if p and isinstance(n, list)]
            all_paths.sort(key=len)
            replaced = False
            for p in all_paths:
                if runner.evals >= budget:
                    break
                cur = get_at(forms[f], p)
                # try hoisting a child first (keeps types), then literals
                candidates = [c for c in (cur[1:] if isinstance(cur, list) else [])
                              if isinstance(c, (str, list))][:3]
                candidates += LITERAL_CANDIDATES
                for cand in candidates:
                    if cand == cur:
                        continue
                    trial_form = replace_at(forms[f], p, cand)
                    trial = forms[:f] + [trial_form] + forms[f + 1:]
                    if still_diverges(trial):
                        forms = trial
                        changed = True
                        replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                f += 1
    return forms

# ---------------------------------------------------------------------------
# Findings output.
# ---------------------------------------------------------------------------

def next_finding_number(found_dir):
    best = 0
    if os.path.isdir(found_dir):
        for name in os.listdir(found_dir):
            m = re.match(r"^(\d{3})_", name)
            if m:
                best = max(best, int(m.group(1)))
    return best + 1

def save_finding(found_dir, num, seed, index, source, res, orig_source):
    os.makedirs(found_dir, exist_ok=True)
    path = os.path.join(found_dir, "%03d_shrunk.esk" % num)
    with open(path, "w") as f:
        f.write(";; FINDING %03d — auto-shrunk divergent program\n" % num)
        f.write(";; generator: scripts/gen_differential.py --seed %d (program index %d)\n"
                % (seed, index))
        f.write(";; Per-axis (exit code, normalized stdout):\n")
        for axis, (rc, out) in res.items():
            snippet = out.replace("\n", "\\n")
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            f.write(";;   %-12s rc=%-3d stdout=%s\n" % (axis, rc, snippet))
        f.write(";; Original (unshrunk) program was %d bytes.\n" % len(orig_source))
        f.write(source)
    return path

# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--found-dir", default="tests/differential/found")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--max-findings", type=int, default=10)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--shrink-budget", type=int, default=300,
                    help="max axis-evaluations per shrink")
    ap.add_argument("--emit-only", metavar="DIR",
                    help="only write generated programs to DIR, don't run")
    ap.add_argument("--trace-file",
                    default="scripts/icc_traces/differential_fuzz.jsonl")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    build_dir = args.build_dir if os.path.isabs(args.build_dir) \
        else os.path.join(repo_root, args.build_dir)

    if args.emit_only:
        os.makedirs(args.emit_only, exist_ok=True)
        for i in range(args.count):
            src = render_program(gen_program(args.seed, i, args.max_depth))
            with open(os.path.join(args.emit_only,
                                   "fuzz_s%d_%04d.esk" % (args.seed, i)), "w") as f:
                f.write(src)
        print("Emitted %d programs to %s" % (args.count, args.emit_only))
        return 0

    if not os.access(os.path.join(build_dir, "eshkol-run"), os.X_OK):
        print("gen_differential.py: %s/eshkol-run not found" % build_dir,
              file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.trace_file), exist_ok=True)
    trace = open(args.trace_file, "w")

    def emit_event(name, value, snippet):
        import json
        trace.write(json.dumps({"kind": "differential_smoke", "name": name,
                                "value": value, "snippet": snippet,
                                "confidence": 0.95}) + "\n")
        trace.flush()

    stats = {"AGREE": 0, "DIVERGE": 0, "ALL-FAIL": 0}
    findings = []
    seen_signatures = set()

    with tempfile.TemporaryDirectory(prefix="eshkol-fuzz.") as workdir:
        runner = Runner(build_dir, workdir, args.timeout)
        for i in range(args.count):
            forms = gen_program(args.seed, i, args.max_depth)
            source = render_program(forms)
            res = runner.run_axes(source)
            status, detail = classify(res)
            stats[status] += 1
            nodeid = "tests/differential/fuzz::seed%d-prog%04d" % (args.seed, i)
            if status == "AGREE":
                print("PASSED %s" % nodeid)
            elif status == "ALL-FAIL":
                # program itself fails identically everywhere: generator gap,
                # not a divergence — reported but not a finding.
                print("SKIPPED %s (%s)" % (nodeid, detail))
            else:
                print("FAILED %s (%s)" % (nodeid, detail))
                if len(findings) < args.max_findings:
                    base_evals = runner.evals
                    is_div = make_divergence_predicate(runner, res)
                    runner.evals = 0
                    shrunk = shrink(forms, runner, is_div,
                                    budget=args.shrink_budget)
                    runner.evals = base_evals
                    shrunk_src = render_program(shrunk)
                    shrunk_res = runner.run_axes(shrunk_src)
                    if classify(shrunk_res)[0] != "DIVERGE":
                        shrunk_src, shrunk_res = source, res  # paranoia
                    sig = hashlib.sha256(shrunk_src.encode()).hexdigest()
                    if sig in seen_signatures:
                        print("  (duplicate of an earlier shrunk finding — not re-saved)")
                    else:
                        seen_signatures.add(sig)
                        num = next_finding_number(args.found_dir)
                        path = save_finding(args.found_dir, num, args.seed, i,
                                            shrunk_src, shrunk_res, source)
                        findings.append((num, i, detail, path))
                        print("  shrunk repro saved: %s" % path)
                        emit_event("differential_fuzz_finding_%03d" % num,
                                   "FAIL", "seed=%d prog=%d %s"
                                   % (args.seed, i, detail))
            emit_event("differential_fuzz_seed%d_prog%04d" % (args.seed, i),
                       "PASS" if status == "AGREE" else
                       ("SKIP" if status == "ALL-FAIL" else "FAIL"), detail)

    print()
    print("Fuzz summary: seed=%d count=%d — %d agree, %d diverge, %d all-fail(skipped)"
          % (args.seed, args.count, stats["AGREE"], stats["DIVERGE"],
             stats["ALL-FAIL"]))
    for (num, i, detail, path) in findings:
        print("  FINDING %03d (prog %d): %s -> %s" % (num, i, detail, path))
    emit_event("differential_fuzz_clean",
               "PASS" if stats["DIVERGE"] == 0 else "FAIL",
               "seed=%d count=%d diverge=%d allfail=%d"
               % (args.seed, args.count, stats["DIVERGE"], stats["ALL-FAIL"]))
    trace.close()
    return 1 if stats["DIVERGE"] else 0

if __name__ == "__main__":
    sys.exit(main())
