#!/usr/bin/env python3
"""AD composition oracle generator (adversarial testing campaign, pillar P3).

Deterministically enumerates the automatic-differentiation surface of Eshkol as
a composition matrix:

  operators : derivative gradient jacobian hessian divergence curl laplacian
  points    : scalar, 2-vector, 3-vector, 2/3-tensor, multi-param + list
  shapes    : polynomial, product-of-linears, with-subtraction, rational,
              exp/sin composite, let-bound intermediate reused twice,
              named-let accumulation loop
  binding   : inline lambda, named define, lambda-in-variable
  captures  : none, global scalar, LOCAL param scalar, vector-ref of outer param
  nesting   : none, derivative-of-derivative (pure 2nd + perturbation
              confusion), gradient-of-derivative (scalar and vector param),
              gradient-of-gradient (scalar and vector param), AD-in-loop reuse

For every valid cell it emits a probe that computes the AD value AND an
in-language central finite-difference approximation of the same quantity
(ground truth needs no hand computation), then checks

    |ad - fd| <= atol + rtol*|fd|      (rtol 1e-4)

First-order stencils use h=1e-5 (atol 1e-6); second-order stencils (hessian
diagonals/cross terms, laplacian) use h=1e-4 (atol 1e-5) because the eps/h^2
round-off term dominates at h=1e-5.

Probes print "PASS: <id>" / "FAIL: <id> ad=<v> fd=<v>". Cells that are
known-open compiler bugs print "XKNOWN: <id> (<ESH-task>) ..." instead of
FAIL so the gate stays green while the bug is tracked:

  ESH-0078  2nd-order gradient through a NAMED-function inner gradient
  ESH-0093  vector-param gradient over an inner derivative (mixed mode)
  ESH-0095  hessian/laplacian SIGSEGV on tensor-literal points  (found by
            this oracle; crash cells live in their own ad_oracle_xc_* files
            so one crash cannot mask other probes)
  ESH-0096  vector-param gradient-of-gradient returns zeros (found by this
            oracle)
  ESH-0097  vector-param AD op (gradient/jacobian/hessian/divergence/curl/
            laplacian) whose lambda captures a LOCAL function parameter
            fails LLVM verification: "PtrToInt source must be pointer
            (ptrtoint %eshkol_tagged_value %a to i64)" on BOTH -r and AOT
            (found by this oracle; compile-time failure kills the whole
            translation unit, so these cells also live in xc_* files)

Output: tests/ad_oracle/generated/ad_oracle_<section>_<NN>.esk (~15 probes
per file), plus ad_oracle_xc_<task>_<NN>.esk single-probe expected-crash
files, plus MANIFEST.txt. Deterministic: no RNG, stable ordering; rerunning
the generator reproduces the corpus byte-for-byte.

Usage: python3 tests/ad_oracle/gen_ad_oracle.py [--outdir DIR]
"""

import argparse
import os
import sys

H1 = "1e-5"        # first-order central-difference step
H2 = "1e-4"        # second-order stencil step
ATOL1, ATOL2 = "1e-6", "1e-5"
RTOL = "1e-4"

# xknown tasks whose failure mode kills the whole translation unit
# (SIGSEGV or compile-time IR verification error): these probes are
# emitted one-per-file as ad_oracle_xc_<task>_<NN>.esk.
CRASH_TASKS = ()  # ESH-0095/0097 fixed in sweep C (were tensor-point / local-capture crashes)

X0 = "1.3"                      # scalar evaluation point
P2 = ["1.3", "-0.7"]            # 2-d evaluation point
P3 = ["1.3", "-0.7", "0.6"]     # 3-d evaluation point

MAX_CHKS_PER_FILE = 20          # ~15 probes/file for 1-chk probes

# ---------------------------------------------------------------------------
# point helpers
# ---------------------------------------------------------------------------

POINT_CTOR = {"v": "vector", "t": "tensor", "l": "list"}


def point(kind, comps):
    """kind: v2 v3 t2 t3 l2 -> literal point expression."""
    return "({} {})".format(POINT_CTOR[kind[0]], " ".join(comps))


def shifted(comps, i, sign, h):
    out = list(comps)
    out[i] = "({} {} {})".format(sign, comps[i], h)
    return out


# ---------------------------------------------------------------------------
# function shapes — scalar argument (x is an expression string)
# ---------------------------------------------------------------------------

def s_poly(x):
    return f"(+ (- (* 1.5 {x} {x} {x}) (* 2.0 {x} {x})) (* 3.0 {x}))"


def s_prodlin(x):
    return f"(* (+ (* 2.0 {x}) 1.0) (- {x} 3.0) (+ (* 0.5 {x}) 2.0))"


def s_sub(x):
    return f"(- (* (- {x} 0.7) (- {x} 0.7)) {x})"


def s_rational(x):
    return f"(/ 1.0 (+ 1.0 (* {x} {x})))"


def s_expsin(x):
    return f"(* (exp (* 0.3 {x})) (sin {x}))"


def s_letreuse(x):
    return f"(let ((t (* {x} {x}))) (+ (* t t) t))"


def s_namedlet(x):
    return (f"(let loop ((i 0) (acc 0.0)) (if (>= i 3) acc "
            f"(loop (+ i 1) (+ acc (* {x} {x} (+ 1.0 (* 0.5 i)))))))")


SCALAR_SHAPES = [
    ("poly", s_poly), ("prodlin", s_prodlin), ("sub", s_sub),
    ("rational", s_rational), ("expsin", s_expsin),
    ("letreuse", s_letreuse), ("namedlet", s_namedlet),
]

# ---------------------------------------------------------------------------
# scalar fields R^n -> R  (xs = component accessor expressions; vname for the
# named-let shape, which iterates (vref v i))
# ---------------------------------------------------------------------------

def sf_poly(xs, vname=None):
    e = (f"(+ (- (* 1.5 {xs[0]} {xs[0]} {xs[0]}) (* 2.0 {xs[1]} {xs[1]}))"
         f" (* 3.0 {xs[0]} {xs[1]}))")
    if len(xs) == 3:
        e = f"(+ {e} (* 0.7 {xs[2]} {xs[2]} {xs[2]}))"
    return e


def sf_prodlin(xs, vname=None):
    e = f"(* (+ (* 2.0 {xs[0]}) 1.0) (- {xs[1]} 3.0) (+ (* 0.5 {xs[0]}) 2.0))"
    if len(xs) == 3:
        e = f"(* {e} (+ {xs[2]} 1.5))"
    return e


def sf_sub(xs, vname=None):
    e = f"(- (* (- {xs[0]} {xs[1]}) (- {xs[0]} {xs[1]})) {xs[0]})"
    if len(xs) == 3:
        e = f"(- {e} (* {xs[2]} {xs[1]}))"
    return e


def sf_rational(xs, vname=None):
    den = " ".join(f"(* {x} {x})" for x in xs)
    return f"(/ 1.0 (+ 1.0 {den}))"


def sf_expsin(xs, vname=None):
    e = f"(* (exp (* 0.3 {xs[0]})) (sin {xs[1]}))"
    if len(xs) == 3:
        e = f"(+ {e} (cos {xs[2]}))"
    return e


def sf_letreuse(xs, vname=None):
    if len(xs) == 3:
        return (f"(let ((t (* {xs[0]} {xs[1]}))) "
                f"(+ (* t t) t (* t {xs[2]})))")
    return f"(let ((t (* {xs[0]} {xs[1]}))) (+ (* t t) t))"


def sf_namedlet(xs, vname):
    n = len(xs)
    return (f"(let loop ((i 0) (acc 0.0)) (if (>= i {n}) acc (loop (+ i 1) "
            f"(+ acc (* (vref {vname} i) (vref {vname} i) "
            f"(+ 1.0 (* 0.5 i)))))))")


FIELD_SHAPES = [
    ("poly", sf_poly), ("prodlin", sf_prodlin), ("sub", sf_sub),
    ("rational", sf_rational), ("expsin", sf_expsin),
    ("letreuse", sf_letreuse), ("namedlet", sf_namedlet),
]

# ---------------------------------------------------------------------------
# vector fields R^n -> R^n (component list; each comp i depends on x_i so the
# jacobian diagonal / divergence are nontrivial)
# ---------------------------------------------------------------------------

def vf_poly(xs, vname=None):
    c = [f"(* 1.5 {xs[0]} {xs[0]} {xs[1]})",
         f"(- (* {xs[0]} {xs[1]} {xs[1]}) (* 2.0 {xs[1]}))"]
    if len(xs) == 3:
        c.append(f"(* {xs[2]} {xs[2]} {xs[0]})")
    return c


def vf_prodlin(xs, vname=None):
    c = [f"(* (+ (* 2.0 {xs[0]}) 1.0) (- {xs[1]} 3.0))",
         f"(* (+ {xs[1]} 2.0) (+ (* 0.5 {xs[0]}) 1.0))"]
    if len(xs) == 3:
        c.append(f"(* (+ {xs[2]} 0.5) (+ {xs[0]} 1.0))")
    return c


def vf_sub(xs, vname=None):
    c = [f"(- (* {xs[0]} {xs[0]}) {xs[1]})",
         f"(- (* {xs[1]} {xs[1]}) {xs[0]})"]
    if len(xs) == 3:
        c.append(f"(- (* {xs[2]} {xs[2]}) {xs[0]})")
    return c


def vf_rational(xs, vname=None):
    c = [f"(/ {xs[0]} (+ 1.0 (* {xs[1]} {xs[1]})))",
         f"(/ {xs[1]} (+ 1.0 (* {xs[0]} {xs[0]})))"]
    if len(xs) == 3:
        c.append(f"(/ {xs[2]} (+ 1.0 (* {xs[0]} {xs[0]})))")
    return c


def vf_expsin(xs, vname=None):
    c = [f"(* (exp (* 0.2 {xs[0]})) (sin {xs[1]}))",
         f"(* (cos {xs[0]}) {xs[1]})"]
    if len(xs) == 3:
        c.append(f"(* (sin {xs[2]}) (exp (* 0.1 {xs[0]})))")
    return c


def vf_letreuse(xs, vname=None):
    c = [f"(let ((t (* {xs[0]} {xs[1]}))) (+ (* t t) {xs[0]}))",
         f"(let ((t (- {xs[0]} {xs[1]}))) (* t t {xs[1]}))"]
    if len(xs) == 3:
        c.append(f"(let ((t (* {xs[2]} {xs[0]}))) (+ t (* t t)))")
    return c


def vf_namedlet(xs, vname):
    n = len(xs)
    loop = (f"(let loop ((i 0) (acc 0.0)) (if (>= i {n}) acc (loop (+ i 1) "
            f"(+ acc (* (vref {vname} i) (vref {vname} i))))))")
    c = [loop, f"(* {xs[0]} {xs[1]})"]
    if len(xs) == 3:
        c.append(f"(- {xs[2]} {xs[0]})")
    return c


VFIELD_SHAPES = [
    ("poly", vf_poly), ("prodlin", vf_prodlin), ("sub", vf_sub),
    ("rational", vf_rational), ("expsin", vf_expsin),
    ("letreuse", vf_letreuse), ("namedlet", vf_namedlet),
]

VF_BY_NAME = dict(VFIELD_SHAPES)
SF_BY_NAME = dict(FIELD_SHAPES)
SS_BY_NAME = dict(SCALAR_SHAPES)

# ---------------------------------------------------------------------------
# probe emission
# ---------------------------------------------------------------------------

class Gen:
    def __init__(self):
        self.n = 0
        self.probes = []   # dicts: section,id,lines,nchk,xc

    def uid(self):
        self.n += 1
        return self.n

    def add(self, section, pid, lines, nchk, xc=None):
        self.probes.append(dict(section=section, id=pid,
                                lines=[f";; probe {pid}"] + lines + [""],
                                nchk=nchk, xc=xc))

    # -- capture plumbing -------------------------------------------------
    # capture kinds: none | glob | local | vrefout
    # Returns (setup_lines, body_wrapper, ad_maker, fd_fun_name_or_None)
    def cap_mul(self, cap, body):
        if cap == "none":
            return body
        if cap == "glob":
            return f"(* cap-g {body})"
        if cap == "local":
            return f"(* a {body})"
        if cap == "vrefout":
            return f"(* (vector-ref w 0) {body})"
        raise ValueError(cap)

    CAP_ARG = {"local": "1.7", "vrefout": "(vector 1.7 0.3)"}
    CAP_PARAM = {"local": "a", "vrefout": "w"}

    def wrap_capture(self, cap, u, expr_of_body_kind, params, body):
        """Return (lines, ad_value_expr, fd_fun_name).

        expr_of_body_kind(fn_expr) builds the AD expression given a function
        expression; params is the lambda parameter list string; body is the
        (already capture-wrapped) function body.
        """
        lam = f"(lambda ({params}) {body})"
        lines = []
        if cap in ("local", "vrefout"):
            p = self.CAP_PARAM[cap]
            arg = self.CAP_ARG[cap]
            lines.append(f"(define (mk{u} {p}) {expr_of_body_kind(lam)})")
            lines.append(f"(define ad{u} (mk{u} {arg}))")
            lines.append(f"(define (mkf{u} {p}) {lam})")
            lines.append(f"(define fdf{u} (mkf{u} {arg}))")
            return lines, f"ad{u}", f"fdf{u}"
        # none/glob: plain defines
        lines.append(f"(define (fdf{u} {params}) {body})")
        lines.append(f"(define ad{u} {expr_of_body_kind(lam)})")
        return lines, f"ad{u}", f"fdf{u}"

    def bind_fn(self, binding, u, params, body):
        """Return (lines, fn_expr_for_AD, fd_fun_name)."""
        lines = [f"(define (fdf{u} {params}) {body})"]
        if binding == "inline":
            return lines, f"(lambda ({params}) {body})", f"fdf{u}"
        if binding == "named":
            return lines, f"fdf{u}", f"fdf{u}"
        if binding == "lamvar":
            lines.append(f"(define fv{u} (lambda ({params}) {body}))")
            return lines, f"fv{u}", f"fdf{u}"
        raise ValueError(binding)

    # -- FD builders -------------------------------------------------------
    def fd_call(self, f, kind, comps):
        return f"({f} {point(kind, comps)})" if kind != "s" \
            else f"({f} {comps[0]})"

    def fd1_scalar(self, f, x):
        return (f"(/ (- ({f} (+ {x} {H1})) ({f} (- {x} {H1})))"
                f" (* 2.0 {H1}))")

    def fd1_comp(self, f, kind, comps, i, wrap=None):
        """d f / d x_i by central difference; wrap extracts a component of
        f's return value (for vector-valued f), e.g. wrap='(vref {} 0)'."""
        pp = f"({f} {point(kind, shifted(comps, i, '+', H1))})"
        pm = f"({f} {point(kind, shifted(comps, i, '-', H1))})"
        if wrap:
            pp, pm = wrap.format(pp), wrap.format(pm)
        return f"(/ (- {pp} {pm}) (* 2.0 {H1}))"

    def fd1_multi(self, f, comps, i):
        ap = " ".join(shifted(comps, i, "+", H1))
        am = " ".join(shifted(comps, i, "-", H1))
        return f"(/ (- ({f} {ap}) ({f} {am})) (* 2.0 {H1}))"

    def fd2_diag(self, f, kind, comps, i, multi=False):
        if multi:
            pp = f"({f} {' '.join(shifted(comps, i, '+', H2))})"
            pm = f"({f} {' '.join(shifted(comps, i, '-', H2))})"
            p0 = f"({f} {' '.join(comps)})"
        else:
            pp = f"({f} {point(kind, shifted(comps, i, '+', H2))})"
            pm = f"({f} {point(kind, shifted(comps, i, '-', H2))})"
            p0 = f"({f} {point(kind, comps)})"
        return f"(/ (+ (- {pp} (* 2.0 {p0})) {pm}) (* {H2} {H2}))"

    def fd2_cross(self, f, kind, comps, i, j, multi=False):
        def at(si, sj):
            c = shifted(shifted(comps, i, si, H2), j, sj, H2)
            return (f"({f} {' '.join(c)})" if multi
                    else f"({f} {point(kind, c)})")
        return (f"(/ (- (+ {at('+', '+')} {at('-', '-')})"
                f" (+ {at('+', '-')} {at('-', '+')}))"
                f" (* 4.0 {H2} {H2}))")

    # -- chk emission -------------------------------------------------------
    def chk(self, pid, ad, fd, second=False, xc=None):
        atol = ATOL2 if second else ATOL1
        if xc:
            return f'(chk-x "{pid}" "{xc}" {ad} {fd} {atol} {RTOL})'
        return f'(chk "{pid}" {ad} {fd} {atol} {RTOL})'

    # ======================================================================
    # sections
    # ======================================================================

    def gen_deriv(self):
        # shapes x bindings (capture none)
        for sh, fn in SCALAR_SHAPES:
            for b in ("inline", "named", "lamvar"):
                u = self.uid()
                body = fn("x")
                lines, fexpr, fdn = self.bind_fn(b, u, "x", body)
                lines.append(f"(define ad{u} (derivative {fexpr} {X0}))")
                lines.append(f"(define fd{u} {self.fd1_scalar(fdn, X0)})")
                pid = f"deriv.{sh}.s.{b}.capnone"
                lines.append(self.chk(pid, f"ad{u}", f"fd{u}"))
                self.add("deriv", pid, lines, 1)
        # shapes x captures (inline binding)
        for sh, fn in SCALAR_SHAPES:
            for cap in ("glob", "local", "vrefout"):
                u = self.uid()
                body = self.cap_mul(cap, fn("x"))
                lines, adv, fdn = self.wrap_capture(
                    cap, u, lambda lam: f"(derivative {lam} {X0})",
                    "x", body)
                lines.append(f"(define fd{u} {self.fd1_scalar(fdn, X0)})")
                pid = f"deriv.{sh}.s.inline.cap{cap}"
                lines.append(self.chk(pid, adv, f"fd{u}"))
                self.add("deriv", pid, lines, 1)

    def grad_probe(self, sh, fn, kind, comps, binding="inline", cap="none",
                   xc=None):
        u = self.uid()
        n = len(comps)
        if kind == "s":
            body = self.cap_mul(cap, fn(comps[0] and "x"))
            params, xs = "x", None
        elif kind == "l":
            params = " ".join(f"x{i}" for i in range(n))
            xs = params.split()
            body = self.cap_mul(cap, fn(xs, None))
        else:
            params, xs = "v", [f"(vref v {i})" for i in range(n)]
            body = self.cap_mul(cap, fn(xs, "v"))
        pt = comps[0] if kind == "s" else point(kind, comps)
        adexpr = lambda lam: f"(gradient {lam} {pt})"
        if cap in ("local", "vrefout"):
            lines, adv, fdn = self.wrap_capture(cap, u, adexpr, params, body)
        else:
            lines, fexpr, fdn = self.bind_fn(binding, u, params, body)
            lines.append(f"(define ad{u} {adexpr(fexpr)})")
            adv = f"ad{u}"
        pid = f"grad.{sh}.{kind if kind == 's' else kind + str(n)}" \
              f".{binding}.cap{cap}"
        nchk = 0
        if kind == "s":
            lines.append(f"(define fd{u} {self.fd1_scalar(fdn, X0)})")
            lines.append(self.chk(pid, adv, f"fd{u}", xc=xc))
            nchk = 1
        else:
            for i in range(n):
                fd = (self.fd1_multi(fdn, comps, i) if kind == "l"
                      else self.fd1_comp(fdn, kind, comps, i))
                lines.append(f"(define fd{u}_{i} {fd})")
                lines.append(self.chk(f"{pid}[{i}]", f"(vref {adv} {i})",
                                      f"fd{u}_{i}", xc=xc))
                nchk += 1
        self.add("grad", pid, lines, nchk, xc=xc)

    def gen_grad(self):
        for sh, fn in SCALAR_SHAPES:
            self.grad_probe(sh, lambda x, _=None, f=fn: f("x"), "s", [X0])
        for kind, comps in (("v", P2), ("v", P3), ("t", P2), ("t", P3)):
            for sh, fn in FIELD_SHAPES:
                self.grad_probe(sh, fn, kind, comps)
        for sh in ("poly", "prodlin", "sub", "rational", "expsin",
                   "letreuse"):
            self.grad_probe(sh, SF_BY_NAME[sh], "l", P2)
        # binding axis
        for b in ("named", "lamvar"):
            for sh in ("poly", "rational", "namedlet"):
                self.grad_probe(sh, SF_BY_NAME[sh], "v", P2, binding=b)
            self.grad_probe(sh := "poly",
                            lambda x, _=None: s_poly("x"), "s", [X0],
                            binding=b)
        # capture axis (local/vrefout: ESH-0097 compile-time IR failure)
        for cap in ("glob", "local", "vrefout"):
            xc = None  # ESH-0097 fixed (sweep C)
            for sh in ("poly", "expsin"):
                self.grad_probe(sh, SF_BY_NAME[sh], "v", P2, cap=cap, xc=xc)

    def hess_probe(self, sh, kind, comps, binding="inline", cap="none",
                   xc=None):
        u = self.uid()
        n = len(comps)
        fn = SF_BY_NAME[sh]
        if kind == "l":
            params = " ".join(f"x{i}" for i in range(n))
            body = self.cap_mul(cap, fn(params.split(), None))
            multi = True
        else:
            params = "v"
            body = self.cap_mul(cap, fn([f"(vref v {i})" for i in range(n)],
                                        "v"))
            multi = False
        pt = point("l" if kind == "l" else kind, comps)
        adexpr = lambda lam: f"(hessian {lam} {pt})"
        if cap in ("local", "vrefout"):
            lines, adv, fdn = self.wrap_capture(cap, u, adexpr, params, body)
        else:
            lines, fexpr, fdn = self.bind_fn(binding, u, params, body)
            lines.append(f"(define ad{u} {adexpr(fexpr)})")
            adv = f"ad{u}"
        pid = f"hess.{sh}.{kind}{n}.{binding}.cap{cap}"
        nchk = 0
        for i in range(n):
            for j in range(n):
                fd = (self.fd2_diag(fdn, kind, comps, i, multi) if i == j
                      else self.fd2_cross(fdn, kind, comps, i, j, multi))
                lines.append(f"(define fd{u}_{i}{j} {fd})")
                lines.append(self.chk(f"{pid}[{i}][{j}]",
                                      f"(tensor-ref {adv} {i} {j})",
                                      f"fd{u}_{i}{j}", second=True, xc=xc))
                nchk += 1
        self.add("hess", pid, lines, nchk, xc=xc)

    def gen_hess(self):
        for sh, _ in FIELD_SHAPES:
            self.hess_probe(sh, "v", P2)
        for sh in ("poly", "rational", "expsin"):
            self.hess_probe(sh, "v", P3)
        for sh in ("poly", "prodlin"):
            self.hess_probe(sh, "l", P2)
        for b in ("named", "lamvar"):
            self.hess_probe("poly", "v", P2, binding=b)
        for cap in ("glob", "local", "vrefout"):
            self.hess_probe("poly", "v", P2, cap=cap,
                            xc=None)
        # tensor-literal points: SIGSEGV — ESH-0095 (own xc files)
        self.hess_probe("poly", "t", P2, xc=None)
        self.hess_probe("poly", "t", P3, xc=None)

    def jac_probe(self, sh, kind, comps, binding="inline", cap="none",
                  xc=None):
        u = self.uid()
        n = len(comps)
        fn = VF_BY_NAME[sh]
        xs = [f"(vref v {i})" for i in range(n)]
        cs = fn(xs, "v")
        body = "(vector {})".format(" ".join(
            self.cap_mul(cap, c) for c in cs))
        pt = point(kind, comps)
        adexpr = lambda lam: f"(jacobian {lam} {pt})"
        if cap in ("local", "vrefout"):
            lines, adv, fdn = self.wrap_capture(cap, u, adexpr, "v", body)
        else:
            lines, fexpr, fdn = self.bind_fn(binding, u, "v", body)
            lines.append(f"(define ad{u} {adexpr(fexpr)})")
            adv = f"ad{u}"
        pid = f"jac.{sh}.{kind}{n}.{binding}.cap{cap}"
        nchk = 0
        for i in range(n):
            for j in range(n):
                fd = self.fd1_comp(fdn, kind, comps, j,
                                   wrap=f"(vref {{}} {i})")
                lines.append(f"(define fd{u}_{i}{j} {fd})")
                lines.append(self.chk(f"{pid}[{i}][{j}]",
                                      f"(tensor-ref {adv} {i} {j})",
                                      f"fd{u}_{i}{j}", xc=xc))
                nchk += 1
        self.add("jac", pid, lines, nchk, xc=xc)

    def gen_jac(self):
        for sh, _ in VFIELD_SHAPES:
            self.jac_probe(sh, "v", P2)
        for sh in ("poly", "rational", "namedlet"):
            self.jac_probe(sh, "v", P3)
        for sh in ("poly", "expsin"):
            self.jac_probe(sh, "t", P2)
        for b in ("named", "lamvar"):
            self.jac_probe("poly", "v", P2, binding=b)
        for cap in ("glob", "local", "vrefout"):
            self.jac_probe("poly", "v", P2, cap=cap,
                           xc=None)

    def div_probe(self, sh, kind, comps, binding="inline", cap="none",
                  xc=None):
        u = self.uid()
        n = len(comps)
        cs = VF_BY_NAME[sh]([f"(vref v {i})" for i in range(n)], "v")
        body = "(vector {})".format(" ".join(
            self.cap_mul(cap, c) for c in cs))
        pt = point(kind, comps)
        adexpr = lambda lam: f"(divergence {lam} {pt})"
        if cap in ("local", "vrefout"):
            lines, adv, fdn = self.wrap_capture(cap, u, adexpr, "v", body)
        else:
            lines, fexpr, fdn = self.bind_fn(binding, u, "v", body)
            lines.append(f"(define ad{u} {adexpr(fexpr)})")
            adv = f"ad{u}"
        parts = " ".join(self.fd1_comp(fdn, kind, comps, i,
                                       wrap=f"(vref {{}} {i})")
                         for i in range(n))
        lines.append(f"(define fd{u} (+ {parts}))")
        pid = f"div.{sh}.{kind}{n}.{binding}.cap{cap}"
        lines.append(self.chk(pid, adv, f"fd{u}", xc=xc))
        self.add("div", pid, lines, 1, xc=xc)

    def gen_div(self):
        for sh, _ in VFIELD_SHAPES:
            self.div_probe(sh, "v", P2)
        for sh in ("poly", "rational", "namedlet"):
            self.div_probe(sh, "v", P3)
        self.div_probe("poly", "t", P2)
        self.div_probe("poly", "t", P3)
        for b in ("named", "lamvar"):
            self.div_probe("poly", "v", P2, binding=b)
        for cap in ("glob", "local"):
            self.div_probe("poly", "v", P2, cap=cap,
                           xc=None)

    def curl_probe(self, sh, kind, binding="inline", cap="none", xc=None):
        u = self.uid()
        comps = P3
        cs = VF_BY_NAME[sh]([f"(vref v {i})" for i in range(3)], "v")
        body = "(vector {})".format(" ".join(
            self.cap_mul(cap, c) for c in cs))
        pt = point(kind, comps)
        adexpr = lambda lam: f"(curl {lam} {pt})"
        if cap in ("local", "vrefout"):
            lines, adv, fdn = self.wrap_capture(cap, u, adexpr, "v", body)
        else:
            lines, fexpr, fdn = self.bind_fn(binding, u, "v", body)
            lines.append(f"(define ad{u} {adexpr(fexpr)})")
            adv = f"ad{u}"

        def d(i, j):  # dF_i/dx_j
            return self.fd1_comp(fdn, kind, comps, j, wrap=f"(vref {{}} {i})")
        fds = [f"(- {d(2, 1)} {d(1, 2)})",
               f"(- {d(0, 2)} {d(2, 0)})",
               f"(- {d(1, 0)} {d(0, 1)})"]
        pid = f"curl.{sh}.{kind}3.{binding}.cap{cap}"
        nchk = 0
        for k in range(3):
            lines.append(f"(define fd{u}_{k} {fds[k]})")
            lines.append(self.chk(f"{pid}[{k}]", f"(vref {adv} {k})",
                                  f"fd{u}_{k}", xc=xc))
            nchk += 1
        self.add("curl", pid, lines, nchk, xc=xc)

    def gen_curl(self):
        for sh, _ in VFIELD_SHAPES:
            self.curl_probe(sh, "v")
        self.curl_probe("poly", "t")
        self.curl_probe("poly", "v", binding="named")
        self.curl_probe("poly", "v", cap="local", xc=None)

    def lap_probe(self, sh, kind, comps, binding="inline", cap="none",
                  xc=None):
        u = self.uid()
        n = len(comps)
        body = self.cap_mul(cap, SF_BY_NAME[sh](
            [f"(vref v {i})" for i in range(n)], "v"))
        pt = point(kind, comps)
        adexpr = lambda lam: f"(laplacian {lam} {pt})"
        if cap in ("local", "vrefout"):
            lines, adv, fdn = self.wrap_capture(cap, u, adexpr, "v", body)
        else:
            lines, fexpr, fdn = self.bind_fn(binding, u, "v", body)
            lines.append(f"(define ad{u} {adexpr(fexpr)})")
            adv = f"ad{u}"
        parts = " ".join(self.fd2_diag(fdn, kind, comps, i)
                         for i in range(n))
        lines.append(f"(define fd{u} (+ {parts}))")
        pid = f"lap.{sh}.{kind}{n}.{binding}.cap{cap}"
        lines.append(self.chk(pid, adv, f"fd{u}", second=True, xc=xc))
        self.add("lap", pid, lines, 1, xc=xc)

    def gen_lap(self):
        for sh, _ in FIELD_SHAPES:
            self.lap_probe(sh, "v", P2)
        for sh in ("poly", "rational", "expsin"):
            self.lap_probe(sh, "v", P3)
        for b in ("named", "lamvar"):
            self.lap_probe("poly", "v", P2, binding=b)
        for cap in ("glob", "local", "vrefout"):
            self.lap_probe("poly", "v", P2, cap=cap,
                           xc=None)
        # tensor-literal points: SIGSEGV — ESH-0095 (own xc files)
        self.lap_probe("poly", "t", P2, xc=None)
        self.lap_probe("poly", "t", P3, xc=None)

    # -- nesting ------------------------------------------------------------
    def nest_scalar(self, outer_op, sh, binding, pid_nest, x0="1.1",
                    xc=None):
        """outer_op in {derivative, gradient}; inner is (derivative|gradient
        f y) where f per binding; here inner op mirrors pid_nest."""
        u = self.uid()
        # pid_nest: dofd / gofd -> inner derivative; gofg -> inner gradient
        inner_op = "derivative" if pid_nest.endswith("d") else "gradient"
        body = SS_BY_NAME[sh]("y")
        lines = []
        if binding == "inline":
            inner_fn = f"(lambda (y) {body})"
        elif binding == "named":
            lines.append(f"(define (inr{u} y) {body})")
            inner_fn = f"inr{u}"
        else:  # lamvar
            lines.append(f"(define inr{u} (lambda (y) {body}))")
            inner_fn = f"inr{u}"
        ad = (f"({outer_op} (lambda (x) ({inner_op} {inner_fn} x)) {x0})")
        lines.append(f"(define ad{u} {ad})")
        # FD baseline: central diff of the (separately validated) inner AD
        lines.append(f"(define (g{u} x) ({inner_op} (lambda (y) {body}) x))")
        lines.append(f"(define fd{u} {self.fd1_scalar(f'g{u}', x0)})")
        pid = f"nest.{pid_nest}.{sh}.s.{binding}.capnone"
        lines.append(self.chk(pid, f"ad{u}", f"fd{u}", xc=xc))
        self.add("nest", pid, lines, 1, xc=xc)

    def gen_nest(self):
        # derivative-of-derivative (pure second derivative)
        for sh in ("poly", "prodlin", "rational", "expsin", "letreuse"):
            for b in ("inline", "named", "lamvar"):
                self.nest_scalar("derivative", sh, b, "dofd")
        # perturbation-confusion patterns
        u = self.uid()
        lines = [
            f"(define ad{u} (derivative (lambda (x) (* x (derivative "
            f"(lambda (y) (* x y y)) 2.0))) 3.0))",
            f"(define (g{u} x) (* x (derivative (lambda (y) (* x y y)) "
            f"2.0)))",
            f"(define fd{u} {self.fd1_scalar(f'g{u}', '3.0')})",
            self.chk("nest.pertconf.a.s.inline.capnone",
                     f"ad{u}", f"fd{u}"),
        ]
        self.add("nest", "nest.pertconf.a.s.inline.capnone", lines, 1)
        u = self.uid()
        lines = [
            f"(define ad{u} (derivative (lambda (x) (derivative "
            f"(lambda (y) (* x y)) x)) 1.7))",
            f"(define (g{u} x) (derivative (lambda (y) (* x y)) x))",
            f"(define fd{u} {self.fd1_scalar(f'g{u}', '1.7')})",
            self.chk("nest.pertconf.b.s.inline.capnone",
                     f"ad{u}", f"fd{u}"),
        ]
        self.add("nest", "nest.pertconf.b.s.inline.capnone", lines, 1)
        u = self.uid()
        lines = [
            f"(define ad{u} (derivative (lambda (x) (* x (derivative "
            f"(lambda (y) (* y y x)) x))) 1.3))",
            f"(define (g{u} x) (* x (derivative (lambda (y) (* y y x)) x)))",
            f"(define fd{u} {self.fd1_scalar(f'g{u}', '1.3')})",
            self.chk("nest.pertconf.c.s.inline.capnone",
                     f"ad{u}", f"fd{u}"),
        ]
        self.add("nest", "nest.pertconf.c.s.inline.capnone", lines, 1)
        # gradient-of-derivative, scalar outer point
        for sh in ("poly", "rational", "expsin"):
            for b in ("inline", "named"):
                self.nest_scalar("gradient", sh, b, "gofd")
        # gradient-of-gradient, scalar: inline OK; named/lamvar = ESH-0078
        for sh in ("poly", "rational", "expsin"):
            self.nest_scalar("gradient", sh, "inline", "gofg")
        for sh in ("poly", "expsin"):
            self.nest_scalar("gradient", sh, "named", "gofg",
                             xc=None)
        self.nest_scalar("gradient", "poly", "lamvar", "gofg",
                         xc=None)
        # gradient over inner derivative, VECTOR outer param — ESH-0093
        u = self.uid()
        comps = ["3.0", "4.0"]
        lines = [
            f"(define ad{u} (gradient (lambda (v) (derivative (lambda (z) "
            f"(* z (vref v 0))) (vref v 1))) (vector 3.0 4.0)))",
            f"(define (g{u} v) (derivative (lambda (z) (* z (vref v 0))) "
            f"(vref v 1)))",
        ]
        pid = "nest.gofd.a.v2.inline.capnone"
        for i in range(2):
            lines.append(f"(define fd{u}_{i} "
                         f"{self.fd1_comp(f'g{u}', 'v', comps, i)})")
            lines.append(self.chk(f"{pid}[{i}]", f"(vref ad{u} {i})",
                                  f"fd{u}_{i}", xc="ESH-0093"))
        self.add("nest", pid, lines, 2, xc="ESH-0093")
        u = self.uid()
        comps = P2
        lines = [
            f"(define ad{u} (gradient (lambda (v) (derivative (lambda (z) "
            f"(* z z (vref v 1))) (vref v 0))) (vector {P2[0]} {P2[1]})))",
            f"(define (g{u} v) (derivative (lambda (z) (* z z (vref v 1))) "
            f"(vref v 0)))",
        ]
        pid = "nest.gofd.b.v2.inline.capnone"
        for i in range(2):
            lines.append(f"(define fd{u}_{i} "
                         f"{self.fd1_comp(f'g{u}', 'v', comps, i)})")
            lines.append(self.chk(f"{pid}[{i}]", f"(vref ad{u} {i})",
                                  f"fd{u}_{i}", xc="ESH-0093"))
        self.add("nest", pid, lines, 2, xc="ESH-0093")
        # gradient-of-gradient, VECTOR param — found by this oracle: ESH-0096
        u = self.uid()
        lines = [
            f"(define ad{u} (gradient (lambda (v) (vref (gradient "
            f"(lambda (w) (* (vref w 0) (vref w 0) (vref w 0))) v) 0)) "
            f"(vector 2.0)))",
            f"(define (g{u} v) (vref (gradient (lambda (w) (* (vref w 0) "
            f"(vref w 0) (vref w 0))) v) 0))",
            f"(define fd{u}_0 "
            f"{self.fd1_comp(f'g{u}', 'v', ['2.0'], 0)})",
            self.chk("nest.gofg.a.v1.inline.capnone[0]",
                     f"(vref ad{u} 0)", f"fd{u}_0", xc=None),
        ]
        self.add("nest", "nest.gofg.a.v1.inline.capnone", lines, 1,
                 xc=None)
        u = self.uid()
        comps = ["3.0", "4.0"]
        lines = [
            f"(define ad{u} (gradient (lambda (v) (vref (gradient "
            f"(lambda (w) (* (vref w 0) (vref w 0) (vref w 1))) v) 0)) "
            f"(vector 3.0 4.0)))",
            f"(define (g{u} v) (vref (gradient (lambda (w) (* (vref w 0) "
            f"(vref w 0) (vref w 1))) v) 0))",
        ]
        pid = "nest.gofg.b.v2.inline.capnone"
        for i in range(2):
            lines.append(f"(define fd{u}_{i} "
                         f"{self.fd1_comp(f'g{u}', 'v', comps, i)})")
            lines.append(self.chk(f"{pid}[{i}]", f"(vref ad{u} {i})",
                                  f"fd{u}_{i}", xc=None))
        self.add("nest", pid, lines, 2, xc=None)

        # ESH-0117: gradient OVER derivative-of-derivative (gofdofd) — the outer
        # gradient must differentiate the d12 (mixed second-order) coefficient of
        # a 2-level-nested inner forward derivative w.r.t. a CAPTURED parameter.
        # Scalar, vector (forward level protocol) and tensor (reverse tape) param
        # points all exercise the fix. Ground truth: central-difference the outer
        # gradient over the (separately dofd-validated) inner derivative-of-
        # derivative — the same in-language pattern every other nest cell uses.
        # A fully-nested pure FD (2nd-diff stencil + outer diff) is catastrophically
        # cancellation-noisy (~5e-4 rel), so the outer link is what we FD here; the
        # inner second derivative is independently gated by the dofd/pertconf cells.
        for sh in ("poly", "prodlin", "sub", "expsin"):
            self.gofdofd_probe(sh, "s", [X0])
        for sh in ("poly", "prodlin", "sub"):
            self.gofdofd_probe(sh, "v", P2)
        # reverse-tape path (tensor param) — the #113 mixed-mode mechanism at
        # 2-level nesting (d12 tape linkage).
        for sh in ("poly", "prodlin"):
            self.gofdofd_probe(sh, "t", P2)

    def gofdofd_probe(self, sh, kind, comps, x0="1.1"):
        """gradient over (derivative of derivative) with a captured param.

        body(param, z) = param * shape(z). The inner
          d/dx ( d/dz body |_{z=x} ) |_{x=x0}  =  param * shape''(x0)
        is a 2nd derivative; the outer gradient d/dparam = shape''(x0). The FD
        baseline central-differences the outer gradient variable over that inner
        derivative-of-derivative (`g` below), validating the ESH-0117 outer link
        (the inner 2nd derivative is gated independently by the dofd cells).
        """
        u = self.uid()
        shp = SS_BY_NAME[sh]
        if kind == "s":
            params, pt = "p", comps[0]
            cap = "p"
        else:
            params, pt = "v", point(kind, comps)
            cap = "(vector-ref v 0)"
        body = f"(* {cap} {shp('z')})"
        inner = (f"(derivative (lambda (x) (derivative (lambda (z) {body}) x)) "
                 f"{x0})")
        lines = [f"(define (g{u} {params}) {inner})"]
        lines.append(f"(define ad{u} (gradient (lambda ({params}) {inner}) {pt}))")
        knd = "s" if kind == "s" else kind + str(len(comps))
        pid = f"nest.gofdofd.{sh}.{knd}.inline.capnone"
        if kind == "s":
            lines.append(f"(define fd{u} {self.fd1_scalar(f'g{u}', comps[0])})")
            lines.append(self.chk(pid, f"ad{u}", f"fd{u}"))
            self.add("nest", pid, lines, 1)
        else:
            nchk = 0
            for i in range(len(comps)):
                lines.append(f"(define fd{u}_{i} "
                             f"{self.fd1_comp(f'g{u}', kind, comps, i)})")
                lines.append(self.chk(f"{pid}[{i}]", f"(vref ad{u} {i})",
                                      f"fd{u}_{i}"))
                nchk += 1
            self.add("nest", pid, lines, nchk)

    # -- loop-iterated reuse --------------------------------------------
    def gen_loop(self):
        # 50x fixed-point accumulation
        u = self.uid()
        body = s_poly("x")
        lines = [
            f"(define (f{u} x) {body})",
            f"(define ad{u} (let loop ((i 0) (acc 0.0)) (if (>= i 50) acc "
            f"(loop (+ i 1) (+ acc (derivative f{u} {X0}))))))",
            f"(define fd{u} (* 50.0 {self.fd1_scalar(f'f{u}', X0)}))",
            self.chk("loop.fixed50.poly.s.named.capnone",
                     f"ad{u}", f"fd{u}", second=True),
        ]
        self.add("loop", "loop.fixed50.poly.s.named.capnone", lines, 1)
        # 20x varying-point accumulation
        u = self.uid()
        body = s_expsin("x")
        lines = [
            f"(define (f{u} x) {body})",
            f"(define (fdc{u} x) (/ (- (f{u} (+ x {H1})) (f{u} (- x {H1})))"
            f" (* 2.0 {H1})))",
            f"(define ad{u} (let loop ((i 0) (acc 0.0)) (if (>= i 20) acc "
            f"(loop (+ i 1) (+ acc (derivative f{u} (+ 1.0 (* 0.05 i))))))))",
            f"(define fd{u} (let loop ((i 0) (acc 0.0)) (if (>= i 20) acc "
            f"(loop (+ i 1) (+ acc (fdc{u} (+ 1.0 (* 0.05 i))))))))",
            self.chk("loop.vary20.expsin.s.named.capnone",
                     f"ad{u}", f"fd{u}", second=True),
        ]
        self.add("loop", "loop.vary20.expsin.s.named.capnone", lines, 1)
        # gradient inside a named-let loop (regression guard for PR #84)
        u = self.uid()
        lines = [
            f"(define (f{u} v) (* (vref v 0) (vref v 0) (vref v 1)))",
            f"(define (gc{u} x) (vref (gradient f{u} (vector x 2.0)) 0))",
            f"(define (fdc{u} x) (/ (- (f{u} (vector (+ x {H1}) 2.0)) "
            f"(f{u} (vector (- x {H1}) 2.0))) (* 2.0 {H1})))",
            f"(define ad{u} (let loop ((i 0) (acc 0.0)) (if (>= i 10) acc "
            f"(loop (+ i 1) (+ acc (gc{u} (+ 1.0 (* 0.1 i))))))))",
            f"(define fd{u} (let loop ((i 0) (acc 0.0)) (if (>= i 10) acc "
            f"(loop (+ i 1) (+ acc (fdc{u} (+ 1.0 (* 0.1 i))))))))",
            self.chk("loop.gradloop10.poly.v2.named.capnone",
                     f"ad{u}", f"fd{u}", second=True),
        ]
        self.add("loop", "loop.gradloop10.poly.v2.named.capnone", lines, 1)
        # newton-style: derivative result feeds the next loop point
        u = self.uid()
        lines = [
            f"(define (f{u} x) (- (* x x) 2.0))",
            f"(define (fdc{u} x) (/ (- (f{u} (+ x {H1})) (f{u} (- x {H1})))"
            f" (* 2.0 {H1})))",
            f"(define ad{u} (let loop ((i 0) (x 1.5)) (if (>= i 5) x "
            f"(loop (+ i 1) (- x (/ (f{u} x) (derivative f{u} x)))))))",
            f"(define fd{u} (let loop ((i 0) (x 1.5)) (if (>= i 5) x "
            f"(loop (+ i 1) (- x (/ (f{u} x) (fdc{u} x)))))))",
            self.chk("loop.newton5.poly.s.named.capnone",
                     f"ad{u}", f"fd{u}"),
        ]
        self.add("loop", "loop.newton5.poly.s.named.capnone", lines, 1)

    def generate(self):
        self.gen_deriv()
        self.gen_grad()
        self.gen_hess()
        self.gen_jac()
        self.gen_div()
        self.gen_curl()
        self.gen_lap()
        self.gen_nest()
        self.gen_loop()
        return self.probes


PRELUDE = """\
;; GENERATED by tests/ad_oracle/gen_ad_oracle.py — DO NOT EDIT BY HAND.
;; AD composition oracle (adversarial pillar P3). Every AD value below is
;; checked against an in-language central finite difference.
(define op-pass 0)
(define op-fail 0)
(define op-xknown 0)
(define cap-g 1.7)
(define (close? ad fd atol rtol)
  (<= (abs (- ad fd)) (+ atol (* rtol (abs fd)))))
(define (chk id ad fd atol rtol)
  (if (close? ad fd atol rtol)
      (begin (set! op-pass (+ op-pass 1))
             (display "PASS: ") (display id) (newline))
      (begin (set! op-fail (+ op-fail 1))
             (display "FAIL: ") (display id)
             (display " ad=") (display ad)
             (display " fd=") (display fd) (newline))))
(define (chk-x id task ad fd atol rtol)
  (if (close? ad fd atol rtol)
      (begin (set! op-pass (+ op-pass 1))
             (display "PASS: ") (display id)
             (display " (fixed: ") (display task) (display ")") (newline))
      (begin (set! op-xknown (+ op-xknown 1))
             (display "XKNOWN: ") (display id)
             (display " (") (display task)
             (display ") ad=") (display ad)
             (display " fd=") (display fd) (newline))))
"""

SUMMARY = """\
(display "Passed: ") (display op-pass) (newline)
(display "Failed: ") (display op-fail) (newline)
(display "Xknown: ") (display op-xknown) (newline)
"""


def write_files(probes, outdir):
    os.makedirs(outdir, exist_ok=True)
    for old in sorted(os.listdir(outdir)):
        if old.startswith("ad_oracle_") and old.endswith(".esk"):
            os.remove(os.path.join(outdir, old))

    manifest = []

    def emit(fname, chunk):
        path = os.path.join(outdir, fname)
        with open(path, "w") as f:
            f.write(PRELUDE + "\n")
            for p in chunk:
                f.write("\n".join(p["lines"]) + "\n")
            f.write(SUMMARY)
        manifest.append((fname, len(chunk), sum(p["nchk"] for p in chunk),
                         chunk[0]["xc"] if chunk[0]["xc"] and
                         "xc_" in fname else ""))

    # expected-crash / expected-compile-fail probes get one file each so a
    # crash or IR-verification failure masks nothing else
    xc_crash = [p for p in probes if p["xc"] in CRASH_TASKS]
    normal = [p for p in probes if p["xc"] not in CRASH_TASKS]

    by_section = {}
    for p in normal:
        by_section.setdefault(p["section"], []).append(p)

    for section in ("deriv", "grad", "hess", "jac", "div", "curl", "lap",
                    "nest", "loop"):
        plist = by_section.get(section, [])
        chunk, nchk, idx = [], 0, 0
        for p in plist:
            if chunk and nchk + p["nchk"] > MAX_CHKS_PER_FILE:
                idx += 1
                emit(f"ad_oracle_{section}_{idx:02d}.esk", chunk)
                chunk, nchk = [], 0
            chunk.append(p)
            nchk += p["nchk"]
        if chunk:
            idx += 1
            emit(f"ad_oracle_{section}_{idx:02d}.esk", chunk)

    counters = {}
    for p in xc_crash:
        k = counters[p["xc"]] = counters.get(p["xc"], 0) + 1
        emit(f"ad_oracle_xc_{p['xc']}_{k:02d}.esk", [p])

    with open(os.path.join(outdir, "MANIFEST.txt"), "w") as f:
        f.write("# file probes checks expected-crash-task\n")
        for fname, np_, nc, xc in manifest:
            f.write(f"{fname} {np_} {nc} {xc or '-'}\n")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "generated")
    ap.add_argument("--outdir", default=default_out)
    args = ap.parse_args()
    probes = Gen().generate()
    manifest = write_files(probes, args.outdir)
    nprobe = sum(m[1] for m in manifest)
    nchk = sum(m[2] for m in manifest)
    nx = sum(1 for p in probes if p["xc"])
    print(f"ad_oracle: {len(manifest)} files, {nprobe} probes, "
          f"{nchk} checks ({nx} probes marked xknown/xcrash)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
