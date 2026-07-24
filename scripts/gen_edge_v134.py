#!/usr/bin/env python3
r"""gen_edge_v134.py — dynamic, seeded, bounded edge-case generator for the
v1.3.4 language surface (nursery iter-scope, capturing parallel-map, exact
gradient through callable/curried params, native i128, VM matmul parity,
low-level ad-tape/ad-pow, and the number<->string round-trip property).

This is the P2 (edge-matrix) + P6 (depth-parametric) extension for the surface
that the v1.3.4 wave added. It emits SELF-CHECKING closed programs: each file
carries a `;; CHECKS: N` header and prints exactly N lines beginning `PASS: `
(one per assertion) plus `FAIL: ` on any wrong value. The runner
(scripts/run_edge_coverage_v134.sh) classifies each file+mode by the PASS/FAIL
count under JIT / AOT-O0 / AOT-O2 / VM, and — for the matmul + round-trip
families — cross-checks native vs VM (differential oracle).

Design rules (mirror tests/edge_matrix/gen_matrix.py):
  * Deterministic: a pure function of (seed, --counts, --max-depth). Same inputs
    => byte-identical corpus, so a CI divergence reproduces locally.
  * Ground truth is computed IN this generator (Python) and embedded as a
    literal the Eshkol program is checked against — a single broken builtin
    cannot make both sides wrong the same way. Where an external analytic value
    is unavailable, a metamorphic property (identity / involution) is asserted.
  * Bounded: every family caps its case count; depth sweeps are 1..max_depth.
  * Each file names its family via the filename prefix so the runner can gate
    per-family (matches the one-oracle-per-surface-family requirement).

Families / filename prefixes:
  nursery_    mutating define-loop / named-let ticks — 6 barrier channels,
              escape-set size, nested-loop depth (P6b depth axis)
  parallel_   parallel-map / parallel-execute capturing closures returning
              collections — n around the pool threshold, closure shapes,
              scope-op-heavy closures, nesting depth
  gradient_   exact gradient through a callable parameter + curried form —
              arity 1..5, list vs vector point, composition depth (P6a)
  i128_       native 128-bit integer boundaries — +-2^127, wraparound,
              conversions, arithmetic-chain length (P6d depth axis)
  matmul_     VM matmul parity — arange arities, nested literals, multi-dim
              ref/set (differential native-vs-VM in the runner)
  adtape_     low-level ad-tape / ad-pow — fractional/negative/zero exponent,
              tape reuse, 1024-node growth boundary
  roundtrip_  number->string . string->number == identity over generated
              doubles: subnormals, +-0.0, powers of two, 1e+-308

Usage:
  python3 scripts/gen_edge_v134.py --out DIR [--seed N] [--max-depth D]
                                   [--family NAME] [--list-families]
"""

import argparse
import os
import random
import struct
import sys

FAMILIES = ("nursery", "parallel", "gradient", "i128",
            "matmul", "adtape", "roundtrip")

# --------------------------------------------------------------------------
# Self-checking harness embedded at the top of every generated program.
# `chk` records a PASS/FAIL line; `apx` is an absolute-tolerance float compare.
# --------------------------------------------------------------------------
HARNESS = (
    "(define __pass 0)\n"
    "(define __fail 0)\n"
    "(define (chk name ok)\n"
    "  (if ok (begin (set! __pass (+ __pass 1)) (display \"PASS: \") (display name) (newline))\n"
    "         (begin (set! __fail (+ __fail 1)) (display \"FAIL: \") (display name) (newline))))\n"
    "(define (apx a b tol) (< (abs (- a b)) tol))\n"
)


def _fmt_double(x):
    """Emit a double literal that reads back bit-exact in Eshkol (17 sig digits)."""
    if x != x:
        return "+nan.0"
    if x == float("inf"):
        return "+inf.0"
    if x == float("-inf"):
        return "-inf.0"
    s = repr(x)
    if "." not in s and "e" not in s and "E" not in s and "n" not in s:
        s += ".0"
    return s


class Program:
    """Accumulates check lines and renders a complete self-checking file."""

    def __init__(self, doc):
        self.doc = doc
        self.top = []       # top-level definitions
        self.checks = []    # (name, eshkol-bool-expr)
        self.vm_skip = False  # if set, the runner skips this file on the VM

    def define(self, text):
        self.top.append(text)

    def check(self, name, expr):
        # name must be a bare token (no spaces) so runner grep stays simple.
        self.checks.append((name.replace(" ", "-"), expr))

    def render(self):
        out = [";; %s" % self.doc, ";; CHECKS: %d" % len(self.checks)]
        if self.vm_skip:
            out.append(";; VM-SKIP")
        out.append(HARNESS)
        out.extend(self.top)
        for name, expr in self.checks:
            out.append("(chk \"%s\" %s)" % (name, expr))
        out.append(
            "(display \"SUMMARY \") (display __pass) (display \"/\") "
            "(display __fail) (newline)")
        return "\n".join(out) + "\n"


# ==========================================================================
# Family: nursery iter-scope (mutating define-loops / named-let ticks)
# ==========================================================================
# The six structural mutation channels whose write barriers ESH-0214e relies
# on to evacuate persistent-mutation escapees before an iter-scope reset.
_BARRIER_CHANNELS = [
    ("hash-table-set", "(define __kb (make-hash-table))",
     "(hash-table-set! __kb (modulo {i} 64) (list {i} (* {i} 2)))"),
    ("vector-set", "(define __ws (make-vector 64 0))",
     "(vector-set! __ws (modulo {i} 64) (+ {i} 1))"),
    ("set-cdr", "(define __trail (cons 'head '()))",
     "(if (= (modulo {i} 100) 0) (set-cdr! __trail (cons {i} (cdr __trail))) #t)"),
    ("set-car", "(define __cell (cons 0 'tag))",
     "(set-car! __cell {i})"),
    ("string-set", "(define __buf (make-string 8 #\\a))",
     "(string-set! __buf (modulo {i} 8) #\\z)"),
    ("bytevector-u8-set", "(define __bv (make-bytevector 8 0))",
     "(bytevector-u8-set! __bv (modulo {i} 8) (modulo {i} 256))"),
]


def gen_nursery(rng, max_depth, counts):
    files = {}
    n = counts

    # (a) one probe per barrier channel: a guard-wrapped self-tail define loop
    # that mutates persistent state EVERY tick and allocates transient garbage
    # that must die with the tick. Correctness = the loop returns the exact
    # checksum; the reclamation itself is RSS-gated elsewhere, here we assert
    # the mutating loop still computes the right values.
    for name, decl, mut in _BARRIER_CHANNELS:
        p = Program("nursery iter-scope, barrier channel: %s" % name)
        p.define(decl)
        ticks = 3000
        body = (
            "(define (loop i acc)\n"
            "  (guard (e (#t acc))\n"
            "    (if (>= i %d) acc\n"
            "        (let ((scratch (list i (number->string i) (make-vector 16 i))))\n"
            "          %s\n"
            "          (loop (+ i 1) (+ acc (length scratch)))))))\n"
            % (ticks, mut.format(i="i"))
        )
        p.define(body)
        # length of scratch is always 3 -> checksum is 3*ticks.
        p.check("channel-%s-checksum" % name, "(= (loop 0 0) %d)" % (3 * ticks))
        files["nursery_channel_%s" % name] = p.render()

    # (b) all six channels mixed in ONE loop (max mutation-channel mix).
    p = Program("nursery iter-scope, all six barrier channels mixed")
    for _, decl, _ in _BARRIER_CHANNELS:
        p.define(decl)
    ticks = 2000
    muts = "\n          ".join(m.format(i="i") for _, _, m in _BARRIER_CHANNELS)
    p.define(
        "(define (loop i acc)\n"
        "  (guard (e (#t acc))\n"
        "    (if (>= i %d) acc\n"
        "        (let ((scratch (make-vector 32 i)))\n"
        "          %s\n"
        "          (loop (+ i 1) (+ acc (vector-length scratch)))))))\n"
        % (ticks, muts))
    p.check("all-channels-mixed-checksum", "(= (loop 0 0) %d)" % (32 * ticks))
    # persistent state must read back correct after the loop (escapees promoted).
    p.check("kb-escapee-survives",
            "(equal? (hash-table-ref __kb (modulo %d 64) (lambda () #f)) "
            "(list %d %d))" % (ticks - 1, ticks - 1, 2 * (ticks - 1)))
    p.check("vector-escapee-survives",
            "(= (vector-ref __ws (modulo %d 64)) %d)" % (ticks - 1, ticks))
    files["nursery_all_channels_mixed"] = p.render()

    # (c) escape-set-size sweep: named-let carrying an N-element persistent
    # accumulator whose live set grows — the carried out-values must all be
    # promoted at each back edge.
    for esize in (1, 4, 16, 64):
        p = Program("nursery named-let, escape-set size %d" % esize)
        ticks = 1500
        # accumulate an escaping vector of length esize, updated each tick.
        p.define(
            "(define (run)\n"
            "  (let loop ((i 0) (acc (make-vector %d 0)))\n"
            "    (if (>= i %d) acc\n"
            "        (let ((tmp (make-vector 64 i)))\n"
            "          (vector-set! acc (modulo i %d) (+ (vector-ref acc (modulo i %d)) 1))\n"
            "          (loop (+ i 1) acc)))))\n"
            % (esize, ticks, esize, esize))
        # each slot incremented ticks/esize or +1 times; sum of all slots = ticks.
        p.define("(define __r (run))")
        p.define(
            "(define (vsum v) (let s ((i 0) (a 0)) "
            "(if (>= i (vector-length v)) a (s (+ i 1) (+ a (vector-ref v i))))))")
        p.check("escape-set-%d-total" % esize, "(= (vsum __r) %d)" % ticks)
        files["nursery_escapeset_%d" % esize] = p.render()

    # (d) DEPTH axis: nested mutating loops depth 1..max_depth. A depth-d nest
    # runs d loops each of T ticks incrementing a shared persistent counter;
    # ground truth is the closed-form product/sum.
    for d in range(1, max_depth + 1):
        p = Program("nursery nested mutating loops, depth %d (P6b)" % d)
        p.define("(define __ctr (make-vector 1 0))")
        # per-level tick count chosen so the TOTAL iteration budget t^d stays
        # bounded (~4096) regardless of depth.
        t = max(2, int(round(4096 ** (1.0 / d))))
        # build d nested self-tail loops.
        inner = "(vector-set! __ctr 0 (+ (vector-ref __ctr 0) 1))"
        body = inner
        for lvl in range(d):
            body = (
                "(let l%d ((i%d 0)) (if (>= i%d %d) #t "
                "(let ((junk (make-vector 8 i%d))) %s (l%d (+ i%d 1)))))"
                % (lvl, lvl, lvl, t, lvl, body, lvl, lvl))
        p.define("(define (run) %s)" % body)
        p.define("(run)")
        p.check("nested-depth-%d-count" % d,
                "(= (vector-ref __ctr 0) %d)" % (t ** d))
        files["nursery_depth_%02d" % d] = p.render()

    return files


# ==========================================================================
# Family: parallel-map / parallel-execute with capturing closures
# ==========================================================================
def gen_parallel(rng, max_depth, counts):
    files = {}
    # n values around the worker-pool threshold (documented ~16).
    THRESH = (1, 4, 15, 16, 17, 64, 500)

    # (a) capturing closure returning a COLLECTION per element, swept over n.
    for n in THRESH:
        p = Program("parallel-map capturing closure returns list, n=%d" % n)
        cap = 7
        p.define("(define __cap %d)" % cap)
        p.define("(define __in (let b ((i 0) (a '())) "
                 "(if (>= i %d) (reverse a) (b (+ i 1) (cons i a)))))" % n)
        # each element -> (list x (+ x cap) (* x x)); check element-wise sum.
        p.define("(define __out (parallel-map "
                 "(lambda (x) (list x (+ x __cap) (* x x))) __in))")
        p.check("plist-n%d-length" % n, "(= (length __out) %d)" % n)
        p.define(
            "(define (flat-sum lst) (let s ((l lst) (a 0)) "
            "(if (null? l) a (s (cdr l) (+ a (apply + (car l)))))))")
        # sum over x of (x + (x+cap) + x*x) = sum(2x + cap + x^2)
        exp = sum(2 * x + cap + x * x for x in range(n))
        p.check("plist-n%d-flatsum" % n, "(= (flat-sum __out) %d)" % exp)
        files["parallel_list_n%03d" % n] = p.render()

    # (b) closure allocation shapes: nested lists / strings / #f mixes.
    p = Program("parallel-map closure shapes: nested list/string/#f mix")
    n = 32
    p.define("(define __in (let b ((i 0) (a '())) "
             "(if (>= i %d) (reverse a) (b (+ i 1) (cons i a)))))" % n)
    p.define(
        "(define __out (parallel-map (lambda (x)\n"
        "  (if (even? x)\n"
        "      (list (list x x) (number->string x) #f)\n"
        "      (list #f (string-append \"s\" (number->string x)) (list x)))) __in))")
    p.check("shapes-length", "(= (length __out) %d)" % n)
    # element 0 is even -> ((0 0) "0" #f); element 1 odd -> (#f "s1" (1))
    p.check("shapes-even0", "(equal? (car __out) (list (list 0 0) \"0\" #f))")
    p.check("shapes-odd1", "(equal? (cadr __out) (list #f \"s1\" (list 1)))")
    files["parallel_shapes_mix"] = p.render()

    # (c) scope-op-heavy closures: memv + internal named-let inside the worker.
    p = Program("parallel-map scope-op-heavy closure: memv + internal named-let")
    n = 40
    p.define("(define __in (let b ((i 0) (a '())) "
             "(if (>= i %d) (reverse a) (b (+ i 1) (cons i a)))))" % n)
    p.define(
        "(define __out (parallel-map (lambda (x)\n"
        "  (let dloop ((k 0) (acc 0))\n"
        "    (if (>= k x) (if (memv acc (list acc)) acc -1)\n"
        "        (dloop (+ k 1) (+ acc k))))) __in))")
    # worker computes sum 0..x-1 = x*(x-1)/2
    exp = sum(x * (x - 1) // 2 for x in range(n))
    p.check("scopeheavy-sum", "(= (apply + __out) %d)" % exp)
    files["parallel_scopeheavy"] = p.render()

    # (d) parallel-execute with capturing thunks (if the builtin exists).
    p = Program("parallel-execute capturing thunks returning collections")
    p.define("(define __g 100)")
    p.define("(define __res (parallel-execute\n"
             "  (lambda () (list 1 (+ __g 1)))\n"
             "  (lambda () (list 2 (+ __g 2)))\n"
             "  (lambda () (list 3 (+ __g 3)))))")
    # parallel-execute returns a list of the thunk results (order preserved).
    p.check("pexec-count", "(= (length __res) 3)")
    p.check("pexec-values",
            "(equal? __res (list (list 1 101) (list 2 102) (list 3 103)))")
    files["parallel_execute_thunks"] = p.render()

    # (e) DEPTH axis: nested parallel-map (parallel-map inside the worker).
    for d in range(1, min(max_depth, 4) + 1):
        p = Program("parallel-map nesting depth %d (P6f)" % d)
        base = "x"
        expr = base
        for lvl in range(d):
            expr = ("(apply + (parallel-map (lambda (y%d) (+ y%d %s)) "
                    "(list 1 2 3)))" % (lvl, lvl, expr))
        p.define("(define __out (parallel-map (lambda (x) %s) (list 0 1 2 3)))" % expr)
        # compute expected in python: f_0(x)=x; f_{k}(x)=sum_{y in 1..3}(y+f_{k-1}(x))
        def fk(depth, x):
            if depth == 0:
                return x
            return sum(y + fk(depth - 1, x) for y in (1, 2, 3))
        exp = sum(fk(d, x) for x in range(4))
        p.check("pmap-nest-d%d-sum" % d, "(= (apply + __out) %d)" % exp)
        files["parallel_depth_%02d" % d] = p.render()

    return files


# ==========================================================================
# Family: exact gradient through callable parameter + curried form
# ==========================================================================
def gen_gradient(rng, max_depth, counts):
    files = {}

    # (a) arity 1..5, callable reached through a wrapper, point as vector.
    # loss_a(x1..xa) = sum_i (x_i - c_i)^2 ; grad_i = 2(x_i - c_i).
    # NOTE: an arity-1 callable receives the WHOLE point as one vector/tensor
    # argument (the #330 tensor-loss path), so its body must read the point
    # through vector-ref; arity>=2 callables get the point unpacked into N
    # scalar params.
    for arity in range(1, 6):
        cs = [float(i + 1) for i in range(arity)]     # centers c_i
        xs = [0.0] * arity                              # evaluate at origin
        p = Program("gradient through wrapper, arity %d, vector point" % arity)
        if arity == 1:
            # arity-1 tensor loss: single vector argument, body via vector-ref.
            p.define("(define (loss v) (* (- (vector-ref v 0) %s) "
                     "(- (vector-ref v 0) %s)))"
                     % (_fmt_double(cs[0]), _fmt_double(cs[0])))
        else:
            params = " ".join("x%d" % i for i in range(arity))
            body = " ".join(
                "(* (- x%d %s) (- x%d %s))"
                % (i, _fmt_double(cs[i]), i, _fmt_double(cs[i]))
                for i in range(arity))
            p.define("(define (loss %s) (+ %s))" % (params, body))
        p.define("(define (wrap f pt) (gradient f pt))")
        pt = "(vector %s)" % " ".join(_fmt_double(v) for v in xs)
        p.define("(define __d (gradient loss %s))" % pt)
        p.define("(define __w (wrap loss %s))" % pt)
        for i in range(arity):
            g = 2.0 * (xs[i] - cs[i])
            p.check("arity%d-direct-g%d" % (arity, i),
                    "(apx (vector-ref __d %d) %s 1e-9)" % (i, _fmt_double(g)))
            # wrapped must be byte-identical to direct (exact AD).
            p.check("arity%d-wrapped-eq-direct-g%d" % (arity, i),
                    "(= (vector-ref __w %d) (vector-ref __d %d))" % (i, i))
        files["gradient_arity%d_vec" % arity] = p.render()

    # (b) point as LIST vs vector must agree.
    p = Program("gradient point as list vs vector agree")
    p.define("(define (loss x y z) (+ (* (- x 3.0) (- x 3.0)) "
             "(+ (* (- y 5.0) (- y 5.0)) (* (- z 7.0) (- z 7.0)))))")
    p.define("(define __gv (gradient loss (vector 0.0 0.0 0.0)))")
    p.define("(define __gl (gradient loss (list 0.0 0.0 0.0)))")
    for i, c in enumerate((3.0, 5.0, 7.0)):
        p.check("listvec-g%d" % i,
                "(= (vector-ref __gv %d) (vector-ref __gl %d))" % (i, i))
        p.check("listvec-value-g%d" % i,
                "(apx (vector-ref __gv %d) %s 1e-9)" % (i, _fmt_double(-2.0 * c)))
    files["gradient_list_vs_vector"] = p.render()

    # (c) non-polynomial composition through a wrapper.
    # loss(x,y) = (sin x)^2 + (exp (- y)) ; grad = (2 sin x cos x, -exp(-y))
    import math
    p = Program("gradient non-polynomial composition through wrapper")
    p.define("(define (loss x y) (+ (* (sin x) (sin x)) (exp (- y))))")
    p.define("(define (wrap f pt) (gradient f pt))")
    x0, y0 = 0.5, 0.3
    p.define("(define __w (wrap loss (vector %s %s)))"
             % (_fmt_double(x0), _fmt_double(y0)))
    gx = 2 * math.sin(x0) * math.cos(x0)
    gy = -math.exp(-y0)
    p.check("nonpoly-gx", "(apx (vector-ref __w 0) %s 1e-7)" % _fmt_double(gx))
    p.check("nonpoly-gy", "(apx (vector-ref __w 1) %s 1e-7)" % _fmt_double(gy))
    files["gradient_nonpoly"] = p.render()

    # (d) curried form ((gradient f) point) — finite-difference path (~1e-6).
    p = Program("gradient curried form ((gradient f) point)")
    p.define("(define (loss x y) (+ (* x x) (* 3.0 (* y y))))")
    p.define("(define gf (gradient loss))")
    p.define("(define __c (gf (vector 2.0 4.0)))")
    p.check("curried-gx", "(apx (vector-ref __c 0) 4.0 1e-4)")   # d/dx 2x = 4
    p.check("curried-gy", "(apx (vector-ref __c 1) 24.0 1e-4)")  # d/dy 6y = 24
    files["gradient_curried"] = p.render()

    # (e) gradient-in-loop repetition: same gradient recomputed T times must be
    # stable (guards tape/arena reuse across repeated gradient calls).
    p = Program("gradient recomputed in a loop stays stable")
    p.define("(define (loss x y) (+ (* (- x 1.0) (- x 1.0)) (* (- y 2.0) (- y 2.0))))")
    p.define(
        "(define (spin n) (let l ((i 0) (ok #t))\n"
        "  (if (>= i n) ok\n"
        "    (let ((g (gradient loss (vector 0.0 0.0))))\n"
        "      (l (+ i 1) (and ok (apx (vector-ref g 0) -2.0 1e-9)\n"
        "                          (apx (vector-ref g 1) -4.0 1e-9)))))))")
    p.check("grad-loop-stable", "(spin 200)")
    files["gradient_loop_repeat"] = p.render()

    # (f) DEPTH axis: gradient of a function that calls a function, depth 1..N.
    # g_d(x) = f_d(x,y) where f_1(x,y)=x*y, f_{k}=f_{k-1}(x,y)+x*y  => k*x*y.
    # grad wrt (x,y) at (x0,y0) = (k*y0, k*x0).
    for d in range(1, max_depth + 1):
        p = Program("gradient composition depth %d (P6a)" % d)
        # define f_1..f_d each calling the previous.
        p.define("(define (f1 x y) (* x y))")
        for k in range(2, d + 1):
            p.define("(define (f%d x y) (+ (f%d x y) (* x y)))" % (k, k - 1))
        p.define("(define (top v) (f%d (vector-ref v 0) (vector-ref v 1)))" % d)
        p.define("(define __g (gradient top (vector 2.0 3.0)))")
        p.check("gradcomp-d%d-gx" % d,
                "(apx (vector-ref __g 0) %s 1e-7)" % _fmt_double(float(d) * 3.0))
        p.check("gradcomp-d%d-gy" % d,
                "(apx (vector-ref __g 1) %s 1e-7)" % _fmt_double(float(d) * 2.0))
        files["gradient_depth_%02d" % d] = p.render()

    return files


# ==========================================================================
# Family: native i128 boundaries
# ==========================================================================
_I128_MAX = (1 << 127) - 1
_I128_MIN = -(1 << 127)
_WRAP = 1 << 128


def _i128_wrap(v):
    """Two's-complement wrap into [-2^127, 2^127-1]."""
    v &= (_WRAP - 1)
    if v >= (1 << 127):
        v -= _WRAP
    return v


def gen_i128(rng, max_depth, counts):
    files = {}
    MAXS = str(_I128_MAX)
    MINS = str(_I128_MIN)

    # (a) boundary constructors + round-trip.
    p = Program("i128 boundary constructors and string round-trip")
    p.define("(define MAXI (string->i128 \"%s\"))" % MAXS)
    p.define("(define MINI (string->i128 \"%s\"))" % MINS)
    p.check("max-roundtrip", "(string=? (i128->string MAXI) \"%s\")" % MAXS)
    p.check("min-roundtrip", "(string=? (i128->string MINI) \"%s\")" % MINS)
    p.check("zero-roundtrip", "(string=? (i128->string (i128 0)) \"0\")")
    p.check("pred-true", "(i128? (i128 5))")
    p.check("pred-false", "(not (i128? 5))")
    files["i128_boundary_roundtrip"] = p.render()

    # (b) wraparound arithmetic at the boundary (ground truth via python wrap).
    p = Program("i128 wraparound arithmetic at +-2^127")
    p.define("(define MAXI (string->i128 \"%s\"))" % MAXS)
    p.define("(define MINI (string->i128 \"%s\"))" % MINS)
    cases = [
        ("max-plus-1", "(i128-add MAXI (i128 1))", _i128_wrap(_I128_MAX + 1)),
        ("min-minus-1", "(i128-sub MINI (i128 1))", _i128_wrap(_I128_MIN - 1)),
        ("max-times-2", "(i128-mul MAXI (i128 2))", _i128_wrap(_I128_MAX * 2)),
        ("neg-min", "(i128-neg MINI)", _i128_wrap(-_I128_MIN)),
        ("min-times-min", "(i128-mul MINI MINI)", _i128_wrap(_I128_MIN * _I128_MIN)),
    ]
    for name, expr, exp in cases:
        p.check("wrap-%s" % name,
                "(string=? (i128->string %s) \"%d\")" % (expr, exp))
    files["i128_wraparound"] = p.render()

    # (c) conversion edges: int->i128 / i128->int within and out of int range.
    # VM-SKIP: the out-of-range i128->int check relies on a catchable `guard`;
    # on the bytecode VM that condition is FATAL (matches the VM's integer
    # div-by-zero convention, per tests/types/i128_test.esk), so this file is
    # native-only. The wrapping/boundary i128 files ARE VM-differential.
    p = Program("i128 conversion edges")
    p.vm_skip = True
    p.define("(define big (i128-mul (i128 1000000000000) (i128 1000000000000)))")
    p.check("int-roundtrip-small", "(= (i128->int (i128 123456789)) 123456789)")
    p.check("int-roundtrip-neg", "(= (i128->int (i128 -987654321)) -987654321)")
    # big (10^24) is out of int64 range -> i128->int must raise (guarded).
    p.check("out-of-range-raises",
            "(guard (e (#t #t)) (i128->int big) #f)")
    files["i128_conversions"] = p.render()

    # (d) DEPTH axis: arithmetic chain length 1..N (P6d). Repeatedly multiply/
    # add and wrap; compare to python-computed wrapped result.
    for d in range(1, max_depth + 1):
        p = Program("i128 arithmetic chain length %d (P6d)" % d)
        # start from a large base and apply d fused mul-by-3-add-7 steps.
        base = 1234567890123456789
        acc = base
        expr = "(i128 %d)" % base
        for _ in range(d):
            expr = "(i128-add (i128-mul %s (i128 3)) (i128 7))" % expr
            acc = _i128_wrap(_i128_wrap(acc * 3) + 7)
        p.define("(define __r %s)" % expr)
        p.check("chain-d%d" % d, "(string=? (i128->string __r) \"%d\")" % acc)
        files["i128_depth_%02d" % d] = p.render()

    return files


# ==========================================================================
# Family: VM matmul parity (arange arities, nested literals, multi-dim ref/set)
# ==========================================================================
def gen_matmul(rng, max_depth, counts):
    files = {}

    # (a) arange arities: (arange n) / (arange lo hi) / (arange lo hi step).
    p = Program("arange arities produce expected tensors")
    p.define("(define t1 (arange 5))")
    p.check("arange1-elem0", "(apx (tensor-ref t1 0) 0.0 1e-9)")
    p.check("arange1-elem4", "(apx (tensor-ref t1 4) 4.0 1e-9)")
    p.define("(define t2 (arange 2 7))")
    p.check("arange2-elem0", "(apx (tensor-ref t2 0) 2.0 1e-9)")
    p.check("arange2-elem4", "(apx (tensor-ref t2 4) 6.0 1e-9)")
    p.define("(define t3 (arange 0 10 2))")
    p.check("arange3-elem0", "(apx (tensor-ref t3 0) 0.0 1e-9)")
    p.check("arange3-elem4", "(apx (tensor-ref t3 4) 8.0 1e-9)")
    files["matmul_arange_arities"] = p.render()

    # (b) matmul from reshape/arange literals — ground truth computed in python.
    # A = reshape(arange 6, 2, 3) = [[0,1,2],[3,4,5]]
    # B = reshape(arange 6, 3, 2) = [[0,1],[2,3],[4,5]]  => C = [[10,13],[28,40]]
    A = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    B = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    C = [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(2)]
         for i in range(2)]
    p = Program("matmul from reshape/arange matches reference product")
    p.define("(define A (reshape (arange 6) 2 3))")
    p.define("(define B (reshape (arange 6) 3 2))")
    p.define("(define M (matmul A B))")
    for i in range(2):
        for j in range(2):
            p.check("matmul-%d%d" % (i, j),
                    "(apx (tensor-ref M %d %d) %s 1e-6)"
                    % (i, j, _fmt_double(C[i][j])))
    files["matmul_nested_literals"] = p.render()

    # (c) multi-dim ref/set round-trip.
    p = Program("multi-dim tensor ref/set round-trip")
    p.define("(define T (reshape (arange 6) 2 3))")   # [[0,1,2],[3,4,5]]
    p.define("(tensor-set! T 0 2 9.0)")
    p.define("(tensor-set! T 1 0 5.0)")
    p.check("set-get-02", "(apx (tensor-ref T 0 2) 9.0 1e-9)")
    p.check("set-get-10", "(apx (tensor-ref T 1 0) 5.0 1e-9)")
    p.check("untouched-11", "(apx (tensor-ref T 1 1) 4.0 1e-9)")
    files["matmul_multidim_refset"] = p.render()

    return files


# ==========================================================================
# Family: low-level ad-tape / ad-pow
# ==========================================================================
def gen_adtape(rng, max_depth, counts):
    files = {}
    import math

    # (a) ad-pow with fractional / negative / zero exponents.
    # y = x^p ; dy/dx = p * x^(p-1).
    cases = [
        ("frac", 4.0, 0.5),      # sqrt
        ("neg", 2.0, -1.0),      # reciprocal
        ("zero", 3.0, 0.0),      # constant 1
        ("frac2", 8.0, 1.0 / 3.0),
        ("int3", 2.0, 3.0),
    ]
    p = Program("ad-pow fractional/negative/zero exponents")
    for name, x, pw in cases:
        val = x ** pw
        grad = pw * (x ** (pw - 1.0)) if pw != 0.0 else 0.0
        p.define("(define tp-%s (ad-tape-new))" % name)
        p.define("(define xv-%s (ad-var tp-%s %s))" % (name, name, _fmt_double(x)))
        p.define("(define pe-%s (ad-const tp-%s %s))" % (name, name, _fmt_double(pw)))
        p.define("(define yn-%s (ad-pow tp-%s xv-%s pe-%s))"
                 % (name, name, name, name))
        p.define("(ad-backward tp-%s yn-%s)" % (name, name))
        p.check("adpow-%s-value" % name,
                "(apx (ad-node-value tp-%s yn-%s) %s 1e-9)"
                % (name, name, _fmt_double(val)))
        p.check("adpow-%s-grad" % name,
                "(apx (ad-gradient tp-%s xv-%s) %s 1e-7)"
                % (name, name, _fmt_double(grad)))
    files["adtape_pow_exponents"] = p.render()

    # (b) tape reuse: build/backward/release, then a fresh tape gives same value.
    p = Program("ad-tape reuse across new/release cycles")
    p.define(
        "(define (run-once) (let ((tp (ad-tape-new)))\n"
        "  (let ((x (ad-var tp 3.0)) (c (ad-const tp 2.0)))\n"
        "    (let ((y (ad-mul tp x c)))\n"
        "      (ad-backward tp y)\n"
        "      (let ((g (ad-gradient tp x))) (ad-tape-release tp) g)))))")
    p.define(
        "(define (spin n) (let l ((i 0) (ok #t))\n"
        "  (if (>= i n) ok (l (+ i 1) (and ok (apx (run-once) 2.0 1e-12))))))")
    p.check("tape-reuse-stable", "(spin 100)")
    files["adtape_reuse"] = p.render()

    # (c) 1024-node growth boundary: build a chain of ~1200 ad ops on one tape
    # and verify the forward value + a known gradient survive the growth.
    p = Program("ad-tape 1024-node growth boundary")
    nnodes = 1200
    # y = x + 1 applied nnodes times via ad-add with a const 1: y = x + nnodes.
    p.define("(define tp (ad-tape-new))")
    p.define("(define x (ad-var tp 0.0))")
    p.define("(define one (ad-const tp 1.0))")
    p.define(
        "(define y (let l ((i 0) (acc x)) "
        "(if (>= i %d) acc (l (+ i 1) (ad-add tp acc one)))))" % nnodes)
    p.define("(ad-backward tp y)")
    p.check("growth-value", "(apx (ad-node-value tp y) %s 1e-9)"
            % _fmt_double(float(nnodes)))
    p.check("growth-grad", "(apx (ad-gradient tp x) 1.0 1e-12)")
    p.check("growth-tape-len", "(> (ad-tape-length tp) 1024)")
    files["adtape_growth_boundary"] = p.render()

    return files


# ==========================================================================
# Family: number->string . string->number round-trip property
# ==========================================================================
def _interesting_doubles(rng, n):
    vals = [
        0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 0.1, 0.2, 0.3,
        1e308, 1e-308, -1e308, -1e-308,
        2.2250738585072014e-308,        # smallest normal
        5e-324,                          # smallest subnormal
        4.9e-324, 1.5e-323,
        1.7976931348623157e308,         # DBL_MAX
        3.141592653589793, 2.718281828459045,
        123456789.0, -987654321.0,
        1024.0, -2048.0, 65536.0,
    ]
    # powers of two across the exponent range.
    for e in range(-40, 41, 8):
        vals.append(2.0 ** e)
    # deterministic random doubles from raw bit patterns (finite only).
    while len(vals) < len(vals) + n:
        if n <= 0:
            break
        bits = rng.getrandbits(64)
        d = struct.unpack("<d", struct.pack("<Q", bits))[0]
        if d == d and d not in (float("inf"), float("-inf")):
            vals.append(d)
        n -= 1
    return vals


def gen_roundtrip(rng, max_depth, counts):
    files = {}
    vals = _interesting_doubles(rng, counts)
    # Split into chunks so no single file is huge.
    CHUNK = 40
    chunks = [vals[i:i + CHUNK] for i in range(0, len(vals), CHUNK)]
    for ci, chunk in enumerate(chunks):
        p = Program("number->string . string->number identity, chunk %d" % ci)
        for vi, v in enumerate(chunk):
            lit = _fmt_double(v)
            # property: (string->number (number->string v)) == v exactly.
            # -0.0 compares = to 0.0 in R7RS, so this is a value-identity check.
            p.check("rt-%d-%d" % (ci, vi),
                    "(let ((v %s)) (= (string->number (number->string v)) v))"
                    % lit)
        files["roundtrip_chunk_%02d" % ci] = p.render()
    return files


GENERATORS = {
    "nursery": gen_nursery,
    "parallel": gen_parallel,
    "gradient": gen_gradient,
    "i128": gen_i128,
    "matmul": gen_matmul,
    "adtape": gen_adtape,
    "roundtrip": gen_roundtrip,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="output directory for generated .esk files")
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--max-depth", type=int, default=6,
                    help="max depth for the depth-parametric families")
    ap.add_argument("--counts", type=int, default=24,
                    help="per-family bounded random case count")
    ap.add_argument("--family", action="append",
                    help="restrict to one or more families (repeatable)")
    ap.add_argument("--list-families", action="store_true")
    args = ap.parse_args()

    if args.list_families:
        print("\n".join(FAMILIES))
        return 0

    fams = args.family or list(FAMILIES)
    bad = [f for f in fams if f not in GENERATORS]
    if bad:
        sys.exit("unknown family: %s (valid: %s)" % (", ".join(bad),
                                                     ", ".join(FAMILIES)))

    all_files = {}
    for fam in fams:
        rng = random.Random(args.seed ^ hash(fam) & 0xFFFFFFFF)
        all_files.update(GENERATORS[fam](rng, args.max_depth, args.counts))

    if not args.out:
        # dry run: just report.
        print("would generate %d files for families: %s"
              % (len(all_files), ", ".join(fams)))
        for name in sorted(all_files):
            print("  %s.esk" % name)
        return 0

    os.makedirs(args.out, exist_ok=True)
    for name in sorted(all_files):
        with open(os.path.join(args.out, name + ".esk"), "w") as f:
            f.write(all_files[name])
    print("wrote %d files to %s (seed=%d max_depth=%d)"
          % (len(all_files), args.out, args.seed, args.max_depth))
    return 0


if __name__ == "__main__":
    sys.exit(main())
