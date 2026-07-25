#!/usr/bin/env python3
r"""gen_ad_escape.py — P8 escape-closure axes 1 (BINDING-FORM) and 2
(INDIRECTION) for the automatic-differentiation operator family.

Originating escapes (see .swarm/P8_ESCAPE_ANALYSIS.md):
  * A gradient/hessian/laplacian differentiation POINT was classified from its
    AST node kind. A point that was a VARIABLE bound to a vector, a (the ...)
    wrapper, or a general call took a scalar/tensor mispath the moment its
    concrete value diverged from the node kind — SIGSEGV or a silent all-zero
    gradient. The identical value written as a #(...) literal worked, so a
    literal-only test never saw it. (binding-form class)
  * (gradient f point) misbehaved when f was reached through a function
    PARAMETER / wrapper / curried form instead of named directly — the runtime
    -closure branch ignored the callable's arity. Byte-identical to the direct
    call only after the fix. (indirection class)

This generator makes BOTH the point-construction form and the call-indirection
form first-class, swept exhaustively across the AD operator family with an
analytic (closed-form) ground truth. The invariant is doubly strong:
  (G) every form agrees with the CLOSED-FORM ground truth, and
  (A) every form agrees with EACH OTHER (a divergence between two forms is a
      bug even if you cannot say which one is "right").

AXIS 1 — binding-form. Point is built in each of:
    veclit  #(...)          vector  (vector ...)   list  (list ...)
    tensor  (tensor n ...)  var (top-level define) let (let-bound)
    fnret   ((lambda()...)) the (the (vector any) ...)
  (scalar-input operators use numlit / var / let / fnret / the(double)).

AXIS 2 — indirection. The operator CALL is wrapped as:
    direct      (op f pt)
    param       (op reached with f passed through a function parameter)  <- #330
    curried     ((op f) pt)                                              <- #330
    letfun      (let ((g f)) (op g pt))
    twolevel    f threaded through two nested wrapper frames             <- #330

Operators: derivative gradient hessian laplacian jacobian divergence curl
directional-derivative, across arity/dimension 1..3 where each applies.

Deterministic: pure function of (seed). Ground truth is computed HERE in Python
and embedded as a literal. Output: self-checking programs in the shared
scripts/p8/harness.py format.

Usage: python3 scripts/p8/gen_ad_escape.py --out DIR [--seed N]
                 [--axis binding|indirection] [--list]
"""

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import Program, fmt_double            # noqa: E402

# Tolerances. Polynomial losses are exact up to float noise; transcendental
# ones use a looser bar. Second-order (hessian) is looser than first-order.
RTOL1, ATOL1 = "1e-6", "1e-9"
RTOL2, ATOL2 = "1e-4", "1e-7"

# Known-open (operator, point-form) cells discovered BY this generator. They are
# emitted into their own ;; P8-XCRASH file so the runner tolerates the crash
# (XKNOWN) while the tracked task is open, and reports XPASS (stale — promote to
# gate) the moment the fix lands. EMPTY = every generated cell is a hard gate.
#
# ESH-0360-field-list-point: jacobian / divergence / curl SIGSEGV when the
#   differentiation POINT is built with (list ...). The identical point as a
#   #(...) literal / (vector ...) / (tensor ...) / VAR-bound / let-bound /
#   fnret / (the (vector any) ...) is correct, and gradient/hessian/laplacian
#   at a (list ...) point are correct — the #343 cons->svec point normalization
#   was applied to the scalar-output operators but NOT to the vector-field
#   operators. Found on master 5cb02c8a. Reported, unfixed.
KNOWN_CRASH = {
    ("jacobian", "list"): "ESH-0360-field-list-point",
    ("divergence", "list"): "ESH-0360-field-list-point",
    ("curl", "list"): "ESH-0360-field-list-point",
}


def fnum(rng, lo, hi):
    # A "nice" point away from zero so the analytic value is well-conditioned.
    return round(rng.uniform(lo, hi), 4)


# --------------------------------------------------------------------------
# Loss registry. Each scalar-output loss L: R^n -> R carries closed-form
# grad/hessian/laplacian evaluated in Python. Bodies index a whole-vector
# parameter with vector-ref (the exact shape the binding-form escape hit).
# --------------------------------------------------------------------------
def vref(i):
    return "(vector-ref __v %d)" % i


def loss_quad_diag(a):
    """f(v)=sum a_i v_i^2 ; grad_i=2 a_i v_i ; hess=diag(2a_i); lap=2 sum a_i."""
    n = len(a)
    terms = " ".join("(* %s (* %s %s))" % (fmt_double(a[i]), vref(i), vref(i))
                     for i in range(n))
    body = "(+ 0.0 %s)" % terms if n else "0.0"

    def grad(v):
        return [2 * a[i] * v[i] for i in range(n)]

    def hess(v):
        return [[(2 * a[i] if i == j else 0.0) for j in range(n)]
                for i in range(n)]

    def lap(v):
        return 2 * sum(a)
    return body, grad, hess, lap, False


def loss_quad_cross(a, b):
    """f(v)=sum a_i v_i^2 + b v_0 v_1 (n>=2). Off-diagonal hessian term."""
    n = len(a)
    terms = " ".join("(* %s (* %s %s))" % (fmt_double(a[i]), vref(i), vref(i))
                     for i in range(n))
    body = "(+ (+ 0.0 %s) (* %s (* %s %s)))" % (terms, fmt_double(b),
                                                vref(0), vref(1))

    def grad(v):
        g = [2 * a[i] * v[i] for i in range(n)]
        g[0] += b * v[1]
        g[1] += b * v[0]
        return g

    def hess(v):
        h = [[(2 * a[i] if i == j else 0.0) for j in range(n)]
             for i in range(n)]
        h[0][1] = h[1][0] = b
        return h

    def lap(v):
        return 2 * sum(a)
    return body, grad, hess, lap, False


def loss_sincos(a):
    """f(v)=sum a_i sin(v_i) ; grad_i=a_i cos(v_i); hess=diag(-a_i sin v_i);
    lap=-sum a_i sin(v_i). Transcendental -> looser tol."""
    n = len(a)
    terms = " ".join("(* %s (sin %s))" % (fmt_double(a[i]), vref(i))
                     for i in range(n))
    body = "(+ 0.0 %s)" % terms

    def grad(v):
        return [a[i] * math.cos(v[i]) for i in range(n)]

    def hess(v):
        return [[(-a[i] * math.sin(v[i]) if i == j else 0.0)
                 for j in range(n)] for i in range(n)]

    def lap(v):
        return sum(-a[i] * math.sin(v[i]) for i in range(n))
    return body, grad, hess, lap, True


# --------------------------------------------------------------------------
# point construction forms (axis 1)
# --------------------------------------------------------------------------
def vec_forms(v):
    """Return list of (formname, prelude, expr) for a vector point v."""
    elems = " ".join(fmt_double(x) for x in v)
    n = len(v)
    return [
        ("veclit", "", "#(%s)" % elems),
        ("vector", "", "(vector %s)" % elems),
        ("list", "", "(list %s)" % elems),
        ("tensor", "", "(tensor %d %s)" % (n, elems)),
        ("var", "(define __pt (vector %s))" % elems, "__pt"),
        ("let", "", "(let ((__p (vector %s))) __p)" % elems),
        ("fnret", "(define (__mkpt) (vector %s))" % elems, "(__mkpt)"),
        ("the", "", "(the (vector any) (vector %s))" % elems),
    ]


def scalar_forms(x):
    xs = fmt_double(x)
    return [
        ("numlit", "", xs),
        ("var", "(define __ps %s)" % xs, "__ps"),
        ("let", "", "(let ((__q %s)) __q)" % xs),
        ("fnret", "(define (__mks) %s)" % xs, "(__mks)"),
        ("the", "", "(the double %s)" % xs),
    ]


# --------------------------------------------------------------------------
# indirection wrappers (axis 2). Each returns (formname, prelude, callexpr)
# given the operator name, the loss symbol, and the point expression.
# --------------------------------------------------------------------------
def indir_forms(op, f, pt):
    return [
        ("direct", "", "(%s %s %s)" % (op, f, pt)),
        ("param",
         "(define (__call_%s g pt) (%s g pt))" % (op, op),
         "(__call_%s %s %s)" % (op, f, pt)),
        ("curried", "", "((%s %s) %s)" % (op, f, pt)),
        ("letfun", "", "(let ((__g %s)) (%s __g %s))" % (f, op, pt)),
        ("twolevel",
         "(define (__l2_%s h pt) (%s h pt))\n"
         "(define (__l1_%s h pt) (__l2_%s h pt))" % (op, op, op, op),
         "(__l1_%s %s %s)" % (op, f, pt)),
    ]


def approx(name, got, ref, second=False):
    r, a = (RTOL2, ATOL2) if second else (RTOL1, ATOL1)
    return "(chk \"%s\" (close? %s %s %s %s))" % (name, got, fmt_double(ref),
                                                  r, a)


# --------------------------------------------------------------------------
# BINDING-FORM family. One file per (operator, loss, n): sweep every point
# form, checking each against the closed form.
# --------------------------------------------------------------------------
def emit_binding(rng, files):
    losses = []
    for n in (1, 2, 3):
        a = [fnum(rng, 0.5, 2.0) for _ in range(n)]
        losses.append(("quaddiag", n, loss_quad_diag(a),
                       [fnum(rng, -1.5, 1.5) for _ in range(n)]))
        if n >= 2:
            b = fnum(rng, 0.4, 1.2)
            losses.append(("quadcross", n, loss_quad_cross(a, b),
                           [fnum(rng, -1.2, 1.2) for _ in range(n)]))
        losses.append(("sincos", n, loss_sincos(a),
                       [fnum(rng, -1.0, 1.0) for _ in range(n)]))

    for lname, n, (body, grad, hess, lap, trans), pt in losses:
        gtruth = grad(pt)
        htruth = hess(pt)
        ltruth = lap(pt)

        # ---- gradient at every vector-point form -------------------------
        p = Program("binding-form: gradient %s n=%d, all point forms" % (lname, n))
        p.tag("P8-AXIS binding-form")
        p.tag("P8-OP gradient")
        p.define("(define (__v_loss __v) %s)" % body)
        for fn, pre, expr in vec_forms(pt):
            if pre:
                p.define(pre)
            gsym = "__g_%s" % fn
            p.define("(define %s (gradient __v_loss %s))" % (gsym, expr))
            for i in range(n):
                p.check("grad-%s-c%d" % (fn, i),
                        "(close? (vector-ref %s %d) %s %s %s)"
                        % (gsym, i, fmt_double(gtruth[i]), RTOL1, ATOL1))
        # cross-form agreement: every form == the veclit form componentwise
        for fn, _, _ in vec_forms(pt)[1:]:
            for i in range(n):
                p.check("grad-agree-%s-c%d" % (fn, i),
                        "(close? (vector-ref __g_%s %d) (vector-ref __g_veclit %d) 1e-12 1e-12)"
                        % (fn, i, i))
        files["ad_bind_gradient_%s_n%d" % (lname, n)] = p.render()

        # ---- hessian at every vector-point form (the SIGSEGV escape) ------
        p = Program("binding-form: hessian %s n=%d, all point forms" % (lname, n))
        p.tag("P8-AXIS binding-form")
        p.tag("P8-OP hessian")
        p.define("(define (__v_loss __v) %s)" % body)
        for fn, pre, expr in vec_forms(pt):
            if pre:
                p.define(pre)
            hsym = "__h_%s" % fn
            p.define("(define %s (hessian __v_loss %s))" % (hsym, expr))
            for i in range(n):
                for j in range(n):
                    p.check("hess-%s-%d-%d" % (fn, i, j),
                            "(close? (tensor-ref %s %d %d) %s %s %s)"
                            % (hsym, i, j, fmt_double(htruth[i][j]), RTOL2, ATOL2))
        files["ad_bind_hessian_%s_n%d" % (lname, n)] = p.render()

        # ---- laplacian at every vector-point form ------------------------
        p = Program("binding-form: laplacian %s n=%d, all point forms" % (lname, n))
        p.tag("P8-AXIS binding-form")
        p.tag("P8-OP laplacian")
        p.define("(define (__v_loss __v) %s)" % body)
        for fn, pre, expr in vec_forms(pt):
            if pre:
                p.define(pre)
            p.check("lap-%s" % fn,
                    "(close? (laplacian __v_loss %s) %s %s %s)"
                    % (expr, fmt_double(ltruth), RTOL2, ATOL2))
        files["ad_bind_laplacian_%s_n%d" % (lname, n)] = p.render()

    # ---- derivative: scalar-input forms ---------------------------------
    # f(x)=c3 x^3 + c2 x^2 + c1 x ; f'(x)=3c3 x^2 + 2c2 x + c1.
    for k in range(3):
        c3, c2, c1 = (fnum(rng, 0.5, 2.0), fnum(rng, -1.5, 1.5),
                      fnum(rng, -1.0, 1.0))
        x = fnum(rng, -1.3, 1.3)
        d1 = 3 * c3 * x * x + 2 * c2 * x + c1
        d2 = 6 * c3 * x + 2 * c2
        body = ("(+ (+ (* %s (* __x (* __x __x))) (* %s (* __x __x))) (* %s __x))"
                % (fmt_double(c3), fmt_double(c2), fmt_double(c1)))
        p = Program("binding-form: derivative cubic k=%d, scalar-point forms" % k)
        p.tag("P8-AXIS binding-form")
        p.tag("P8-OP derivative")
        p.define("(define (__poly __x) %s)" % body)
        for fn, pre, expr in scalar_forms(x):
            if pre:
                p.define(pre)
            p.check("deriv-%s" % fn,
                    "(close? (derivative __poly %s) %s %s %s)"
                    % (expr, fmt_double(d1), RTOL1, ATOL1))
            p.check("hess-scalar-%s" % fn,
                    "(close? (hessian __poly %s) %s %s %s)"
                    % (expr, fmt_double(d2), RTOL2, ATOL2))
        files["ad_bind_derivative_cubic_%d" % k] = p.render()

    # ---- vector field ops: jacobian / divergence / curl -----------------
    # F(v) = (a0 v0, a1 v1, a2 v2)  (diagonal linear field, R^3->R^3):
    #   jacobian = diag(a_i) ; divergence = sum a_i ; curl = 0.
    a = [fnum(rng, 0.5, 2.5) for _ in range(3)]
    pt = [fnum(rng, -1.0, 1.0) for _ in range(3)]
    fbody = ("(vector (* %s (vector-ref __v 0)) (* %s (vector-ref __v 1)) "
             "(* %s (vector-ref __v 2)))"
             % (fmt_double(a[0]), fmt_double(a[1]), fmt_double(a[2])))
    fielddef = "(define (__F __v) %s)" % fbody

    def field_checks(p, fn, expr):
        jsym = "__J_%s" % fn
        p.define("(define %s (jacobian __F %s))" % (jsym, expr))
        for i in range(3):
            for j in range(3):
                ref = a[i] if i == j else 0.0
                p.check("jac-%s-%d-%d" % (fn, i, j),
                        "(close? (tensor-ref %s %d %d) %s %s %s)"
                        % (jsym, i, j, fmt_double(ref), RTOL1, ATOL1))
        p.check("div-%s" % fn,
                "(close? (divergence __F %s) %s %s %s)"
                % (expr, fmt_double(sum(a)), RTOL1, ATOL1))
        for i in range(3):
            p.check("curl-%s-c%d" % (fn, i),
                    "(close? (vector-ref (curl __F %s) %d) 0.0 1e-6 1e-6)"
                    % (expr, i))

    # Gated file: every point form whose (op, form) is NOT known-crashing.
    p = Program("binding-form: jacobian/divergence/curl linear field, point forms")
    p.tag("P8-AXIS binding-form")
    p.tag("P8-OP field")
    p.define(fielddef)
    quarantined = {}          # task -> list of (fn, pre, expr)
    for fn, pre, expr in vec_forms(pt):
        # A form is quarantined for the whole field file if ANY field operator
        # is known to crash on it (jacobian/divergence/curl share the point).
        tasks = {KNOWN_CRASH.get((op, fn)) for op in ("jacobian", "divergence", "curl")}
        tasks.discard(None)
        if tasks:
            quarantined.setdefault(sorted(tasks)[0], []).append((fn, pre, expr))
            continue
        if pre:
            p.define(pre)
        field_checks(p, fn, expr)
    files["ad_bind_field_linear"] = p.render()

    # Quarantined (known-open) forms -> one ;; P8-XCRASH file per task.
    for task, cells in quarantined.items():
        p = Program("binding-form (known-open %s): field ops at crashing point form" % task)
        p.tag("P8-AXIS binding-form")
        p.tag("P8-OP field")
        p.tag("P8-XCRASH %s" % task)
        p.define(fielddef)
        for fn, pre, expr in cells:
            if pre:
                p.define(pre)
            field_checks(p, fn, expr)
        files["ad_bind_field_xc_%s" % task.replace("-", "_")] = p.render()


# --------------------------------------------------------------------------
# INDIRECTION family (axis 2). One file per (operator, loss): sweep every
# call-indirection wrapper, all against the closed form AND each other.
# --------------------------------------------------------------------------
def emit_indirection(rng, files):
    # gradient / hessian / laplacian through wrappers.
    for n in (1, 2, 3):
        a = [fnum(rng, 0.5, 2.0) for _ in range(n)]
        body, grad, hess, lap, _ = loss_quad_diag(a)
        pt = [fnum(rng, -1.3, 1.3) for _ in range(n)]
        ptexpr = "(vector %s)" % " ".join(fmt_double(x) for x in pt)
        gtruth, htruth, ltruth = grad(pt), hess(pt), lap(pt)

        p = Program("indirection: gradient quaddiag n=%d, all call forms" % n)
        p.tag("P8-AXIS indirection")
        p.tag("P8-OP gradient")
        p.define("(define (__v_loss __v) %s)" % body)
        for fn, pre, call in indir_forms("gradient", "__v_loss", ptexpr):
            if pre:
                p.define(pre)
            gsym = "__gi_%s" % fn
            p.define("(define %s %s)" % (gsym, call))
            for i in range(n):
                p.check("grad-%s-c%d" % (fn, i),
                        "(close? (vector-ref %s %d) %s %s %s)"
                        % (gsym, i, fmt_double(gtruth[i]), RTOL1, ATOL1))
        # agreement with direct
        for fn, _, _ in indir_forms("gradient", "__v_loss", ptexpr)[1:]:
            for i in range(n):
                p.check("grad-agree-%s-c%d" % (fn, i),
                        "(close? (vector-ref __gi_%s %d) (vector-ref __gi_direct %d) 1e-12 1e-12)"
                        % (fn, i, i))
        files["ad_indir_gradient_n%d" % n] = p.render()

        # hessian through wrappers (curried hessian may be unsupported; guard)
        p = Program("indirection: hessian quaddiag n=%d, param/letfun/twolevel" % n)
        p.tag("P8-AXIS indirection")
        p.tag("P8-OP hessian")
        p.define("(define (__v_loss __v) %s)" % body)
        for fn, pre, call in indir_forms("hessian", "__v_loss", ptexpr):
            if fn == "curried":
                continue  # ((hessian f) pt) is not a documented curried form
            if pre:
                p.define(pre)
            hsym = "__hi_%s" % fn
            p.define("(define %s %s)" % (hsym, call))
            for i in range(n):
                for j in range(n):
                    p.check("hess-%s-%d-%d" % (fn, i, j),
                            "(close? (tensor-ref %s %d %d) %s %s %s)"
                            % (hsym, i, j, fmt_double(htruth[i][j]), RTOL2, ATOL2))
        files["ad_indir_hessian_n%d" % n] = p.render()

    # derivative through wrappers (scalar).
    c3, c2, c1 = fnum(rng, 0.5, 2.0), fnum(rng, -1.0, 1.0), fnum(rng, -1.0, 1.0)
    x = fnum(rng, -1.2, 1.2)
    d1 = 3 * c3 * x * x + 2 * c2 * x + c1
    body = ("(+ (+ (* %s (* __x (* __x __x))) (* %s (* __x __x))) (* %s __x))"
            % (fmt_double(c3), fmt_double(c2), fmt_double(c1)))
    p = Program("indirection: derivative cubic, all call forms")
    p.tag("P8-AXIS indirection")
    p.tag("P8-OP derivative")
    p.define("(define (__poly __x) %s)" % body)
    for fn, pre, call in indir_forms("derivative", "__poly", fmt_double(x)):
        if pre:
            p.define(pre)
        p.define("(define __di_%s %s)" % (fn, call))
        p.check("deriv-%s" % fn,
                "(close? __di_%s %s %s %s)" % (fn, fmt_double(d1), RTOL1, ATOL1))
    for fn, _, _ in indir_forms("derivative", "__poly", fmt_double(x))[1:]:
        p.check("deriv-agree-%s" % fn,
                "(close? __di_%s __di_direct 1e-12 1e-12)" % fn)
    files["ad_indir_derivative_cubic"] = p.render()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=False, help="output directory")
    ap.add_argument("--seed", type=int, default=8801)
    ap.add_argument("--axis", choices=("binding", "indirection", "both"),
                    default="both")
    ap.add_argument("--list", action="store_true",
                    help="print file names that would be generated, then exit")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = {}
    if args.axis in ("binding", "both"):
        emit_binding(rng, files)
    if args.axis in ("indirection", "both"):
        emit_indirection(rng, files)

    if args.list:
        for k in sorted(files):
            print(k)
        return 0
    if not args.out:
        sys.exit("--out DIR required (or use --list)")
    os.makedirs(args.out, exist_ok=True)
    for name, text in sorted(files.items()):
        with open(os.path.join(args.out, name + ".esk"), "w") as fh:
            fh.write(text)
    print("wrote %d files to %s" % (len(files), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
