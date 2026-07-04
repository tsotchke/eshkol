#!/usr/bin/env python3
"""Depth-parametric AD oracle generator (adversarial pillar P6a).

Where the P3 AD oracle (tests/ad_oracle) tests a WIDE matrix at SHALLOW,
fixed nesting (nesting depth <= 2), this generator sweeps the nesting DEPTH
itself.  For every composable AD construct it emits the composition
PARAMETRICALLY at depth d = 1..MAX_DEPTH and checks each depth against a
ground-truth oracle that scales with depth, so we can record the MAXIMUM
depth at which each construct stays correct (and whether it FAILS = silent
wrong value, or hits a clean LIMIT = error/exception).

Compositions swept
------------------
  deriv   derivative^d of a scalar function            (sweep 1)
  gradn   gradient^d nested-reverse on a scalar point   (sweep 3)
  gofd    gradient (reverse) OVER derivative^d, vector  (sweep 2, ESH-0117
          param captured via vector-ref                 family; d>=2 tracked)
  jacod   jacobian OVER derivative^d, vector field      (sweep 4)
  hessod  hessian  OVER derivative^d, scalar field      (sweep 4)

Ground truth  (NO hand computation, NO reliance on the AD path)
---------------------------------------------------------------
Every shape used here has a CLOSED-FORM n-th derivative, so the ground truth
at ANY depth is an analytic literal computed in Python:

    mono  t^K          f^(n) = K!/(K-n)! * t^(K-n)     (0 for n>K)
    expc  exp(A t)      f^(n) = A^n * exp(A t)
    sinc  sin(A t)      f^(n) = A^n * sin(A t + n*pi/2)

Captures multiply the base function by a constant c (global / local-param /
vector-ref), which scales the n-th derivative by c.

As a SECOND, AD-independent anchor for the low-depth viable range, each deriv
probe also computes an in-language n-th central-difference stencil of the base
function for d <= FD_MAX_DEPTH and checks it agrees.

Tolerance schedule  (documented; failures return an exact 0 so they are far
outside any of these bands)
-----------------------------------------------------------------------------
    analytic vs AD :   d<=2  rtol 1e-6  atol 1e-6
                       d<=4  rtol 1e-5  atol 1e-5
                       d>=5  rtol 1e-4  atol 1e-4   (nested fp accumulation)
    fd stencil     :   only emitted for d <= 4 (order-n central difference is
                       numerically dead beyond that: round-off ~ eps/h^n).
                       h = 1e-2, rtol 5e-2 * 2^(d-1), atol 1e-2  (diagnostic,
                       corroborates analytic; not the sole gate).

Output
------
  tests/ad_depth/generated/ad_depth_<comp>_<NN>.esk   probe files
  tests/ad_depth/generated/MANIFEST.txt

Each check prints a machine-parseable line consumed by scripts/run_ad_depth.sh:
    RESULT <cellid> d<d> PASS
    RESULT <cellid> d<d> FAIL ad=<v> gt=<v>
    FDCHK  <cellid> d<d> OK|OFF ad=<v> fd=<v>
where cellid = <comp>.<shape>.<point>.<binding>.<cap>.

Deterministic: no RNG, stable ordering; rerunning reproduces byte-for-byte.

Usage: python3 scripts/gen_ad_depth.py [--outdir DIR] [--max-depth N]
"""

import argparse
import math
import os

MAX_DEPTH = 8
FD_MAX_DEPTH = 4
PROBES_PER_FILE = 6

# scalar differentiation-variable evaluation point per shape (exact-ish)
T0 = {"mono": 2.0, "poly": 2.0, "expc": 0.5, "sinc": 0.7}
K = 8            # monomial / poly leading degree (covers depths 1..8)
A_EXP = 0.5      # exp(A t)
A_SIN = 1.3      # sin(A t)
CAPV = 1.7       # capture constant

# vector evaluation points for reverse-outer compositions
VP2 = [3.0, 4.0]
VP3 = [3.0, 4.0, 5.0]


# ---------------------------------------------------------------------------
# analytic n-th derivative of the base scalar function (no AD, no FD)
# ---------------------------------------------------------------------------
def analytic_nth(shape, n):
    t = T0[shape]
    if shape == "mono":
        if n > K:
            return 0.0
        return math.factorial(K) / math.factorial(K - n) * t ** (K - n)
    if shape == "poly":
        # p(t) = t^K - 2 t^5 + 3 t^2
        def term(coef, k):
            if n > k:
                return 0.0
            return coef * math.factorial(k) / math.factorial(k - n) * t ** (k - n)
        return term(1.0, K) + term(-2.0, 5) + term(3.0, 2)
    if shape == "expc":
        return A_EXP ** n * math.exp(A_EXP * t)
    if shape == "sinc":
        return A_SIN ** n * math.sin(A_SIN * t + n * math.pi / 2.0)
    raise ValueError(shape)


# ---------------------------------------------------------------------------
# base scalar function body in Eshkol, over differentiation variable `var`,
# optionally scaled by capture expression `cap` (a string or None)
# ---------------------------------------------------------------------------
def base_body(shape, var):
    if shape == "mono":
        return "(* " + " ".join([var] * K) + ")"
    if shape == "poly":
        m = lambda k: "(* " + " ".join([var] * k) + ")"
        return f"(+ (- {m(K)} (* 2.0 {m(5)})) (* 3.0 {m(2)}))"
    if shape == "expc":
        return f"(exp (* {A_EXP} {var}))"
    if shape == "sinc":
        return f"(sin (* {A_SIN} {var}))"
    raise ValueError(shape)


def scaled_body(shape, var, cap_expr):
    b = base_body(shape, var)
    if cap_expr is None:
        return b
    return f"(* {cap_expr} {b})"


# ---------------------------------------------------------------------------
# tolerance schedule
# ---------------------------------------------------------------------------
def tol(d):
    if d <= 2:
        return "1e-6", "1e-6"
    if d <= 4:
        return "1e-5", "1e-5"
    return "1e-4", "1e-4"


def fd_tol(d):
    # order-n central difference, h=1e-2: round-off grows ~ eps/h^n
    rtol = 5e-2 * (2 ** (d - 1))
    return f"{rtol:.4g}", "1e-2"


# ---------------------------------------------------------------------------
# nested-derivative expression builder
# ---------------------------------------------------------------------------
def nest_deriv(op, d, shape, cap_expr, point_expr, innermost_callee=None):
    """Depth-d nesting of `op` (derivative|gradient) around base scalar body.

    Returns an expression evaluating the d-th derivative (w.r.t. one scalar
    variable) of scaled_body at `point_expr`.  Uses distinct lambda vars.

    If `innermost_callee` is given (a function NAME), the innermost op call
    differentiates that named function directly instead of an inline lambda
    (the `named`/`lamvar` binding forms).
    """
    if innermost_callee is not None:
        inner_point = f"z{d-1}" if d > 1 else point_expr
        expr = f"({op} {innermost_callee} {inner_point})"
    else:
        var = f"z{d}"
        body = scaled_body(shape, var, cap_expr)
        inner_point = f"z{d-1}" if d > 1 else point_expr
        expr = f"({op} (lambda ({var}) {body}) {inner_point})"
    for lvl in range(d - 1, 0, -1):
        var = f"z{lvl}"
        pt = f"z{lvl-1}" if lvl > 1 else point_expr
        expr = f"({op} (lambda ({var}) {expr}) {pt})"
    return expr


def fd_stencil(bfname, n, point, h="1e-2"):
    """In-language n-th central-difference of the base scalar function.

    f^(n)(x) ~ h^-n * sum_{k=0}^n (-1)^k C(n,k) f(x + (n/2 - k) h)
    Emits it as a fully expanded arithmetic expression referencing the named
    base-function helper `bfname` defined by the caller as (bfname t)."""
    terms = []
    for k in range(n + 1):
        coef = ((-1) ** k) * math.comb(n, k)
        off = (n / 2.0 - k)
        if off == 0.0:
            arg = point
        elif off > 0:
            arg = f"(+ {point} (* {off} {h}))"
        else:
            arg = f"(- {point} (* {abs(off)} {h}))"
        terms.append(f"(* {coef:.1f} ({bfname} {arg}))")
    summ = terms[0]
    for t in terms[1:]:
        summ = f"(+ {summ} {t})"
    hn = float(h) ** n
    return f"(/ {summ} {hn!r})"


# ---------------------------------------------------------------------------
# probe emitters
# ---------------------------------------------------------------------------
PRELUDE = """;; GENERATED by scripts/gen_ad_depth.py — DO NOT EDIT BY HAND.
;; Depth-parametric AD oracle (adversarial pillar P6a). Each AD value is
;; checked against an ANALYTIC closed-form ground truth (and, for shallow
;; depths, an in-language n-th central-difference stencil).
(define n-pass 0)
(define n-fail 0)
(define cap-g 1.7)
(define (close? ad gt atol rtol)
  (<= (abs (- ad gt)) (+ atol (* rtol (abs gt)))))
(define (chk id d ad gt atol rtol)
  (if (close? ad gt atol rtol)
      (begin (set! n-pass (+ n-pass 1))
             (display "RESULT ") (display id) (display " d") (display d)
             (display " PASS") (newline))
      (begin (set! n-fail (+ n-fail 1))
             (display "RESULT ") (display id) (display " d") (display d)
             (display " FAIL ad=") (display ad)
             (display " gt=") (display gt) (newline))))
(define (fdchk id d ad fd atol rtol)
  (if (close? ad fd atol rtol)
      (begin (display "FDCHK ") (display id) (display " d") (display d)
             (display " OK ad=") (display ad) (display " fd=") (display fd)
             (newline))
      (begin (display "FDCHK ") (display id) (display " d") (display d)
             (display " OFF ad=") (display ad) (display " fd=") (display fd)
             (newline))))
"""


class Gen:
    def __init__(self, outdir, max_depth):
        self.outdir = outdir
        self.max_depth = max_depth
        self.files = []
        self.total_checks = 0
        self.cells = {}   # cid -> dict(comp, shape, point, binding, cap, file)

    def register(self, cid, comp, shape, point, binding, cap):
        self.cells[cid] = dict(comp=comp, shape=shape, point=point,
                               binding=binding, cap=cap, file=None)
        return cid

    def write_file(self, comp, idx, items):
        """items: list of (cid, text)."""
        name = f"ad_depth_{comp}_{idx:02d}.esk"
        path = os.path.join(self.outdir, name)
        body = PRELUDE + "\n" + "\n".join(t for _, t in items) + "\n"
        body += ('(display "SUMMARY pass=") (display n-pass)'
                 ' (display " fail=") (display n-fail) (newline)\n')
        with open(path, "w") as f:
            f.write(body)
        self.files.append(name)
        for cid, _ in items:
            if cid in self.cells:
                self.cells[cid]["file"] = name

    # -- sweep 1 & 3: derivative^d / gradient^d on a scalar -----------------
    def probe_scalar(self, comp, op, shape, binding, cap):
        """One cell: depths 1..max for op^d of scaled_body, scalar point."""
        cid = self.register(f"{comp}.{shape}.s.{binding}.{cap}",
                            comp, shape, "s", binding, cap)
        lines = [f";; cell {cid}"]
        # capture wiring
        if cap == "capnone":
            cap_expr = None
            wrap = None
        elif cap == "glob":
            cap_expr = "cap-g"
            wrap = None
        elif cap == "localparam":
            cap_expr = "c"          # a local function param
            wrap = "local"
        elif cap == "vecref":
            cap_expr = "(vref pv 0)"
            wrap = "vecref"
        else:
            raise ValueError(cap)

        pt = f"{T0[shape]}"
        # named/lamvar bindings define the innermost base function once.
        callee = None
        if binding in ("named", "lamvar"):
            bfn = f"bf_{comp}_{shape}"
            if binding == "named":
                lines.append(f"(define ({bfn} zz) {base_body(shape, 'zz')})")
            else:  # lamvar
                lines.append(f"(define {bfn} (lambda (zz) {base_body(shape, 'zz')}))")
            callee = bfn
        # single fd base-function helper for the shallow numeric anchor
        emit_fd = (cap == "capnone" and binding == "inline")
        bffn = f"bff_{comp}_{shape}"
        if emit_fd:
            lines.append(f"(define ({bffn} t) {base_body(shape, 't')})")
        for d in range(1, self.max_depth + 1):
            adv = nest_deriv(op, d, shape, cap_expr, pt, innermost_callee=callee)
            if wrap == "local":
                adexpr = f"((lambda (c) {adv}) {CAPV})"
            elif wrap == "vecref":
                adexpr = f"((lambda (pv) {adv}) (vector {CAPV} 1.0))"
            else:
                adexpr = adv
            gt = analytic_nth(shape, d)
            capfac = CAPV if cap in ("glob", "localparam", "vecref") else 1.0
            gt *= capfac
            rtol, atol = tol(d)
            lines.append(f"(chk \"{cid}\" {d} {adexpr} {gt!r} {atol} {rtol})")
            self.total_checks += 1
            # fd anchor for shallow, capnone only (independent numeric truth)
            if d <= FD_MAX_DEPTH and emit_fd:
                sten = fd_stencil(bffn, d, pt)
                frtol, fatol = fd_tol(d)
                lines.append(
                    f"(fdchk \"{cid}\" {d} {adexpr} {sten} {fatol} {frtol})")
        return cid, "\n".join(lines)

    # -- sweep 2: gradient (reverse) OVER derivative^d, vector param --------
    def probe_gofd(self, shape, dim, binding):
        vp = VP2 if dim == 2 else VP3
        cid = self.register(f"gofd.{shape}.v{dim}.{binding}.vecref",
                            "gofd", shape, f"v{dim}", binding, "vecref")
        lines = [f";; cell {cid}  (reverse-outer over forward^d; ESH-0117 family)"]
        pt = f"{T0[shape]}"
        for d in range(1, self.max_depth + 1):
            # inner body: (vref v 0) * base(z)
            var = f"z{d}"
            inner = f"(* (vref v 0) {base_body(shape, var)})"
            expr = f"(derivative (lambda ({var}) {inner}) " \
                   f"{'z'+str(d-1) if d>1 else pt})"
            for lvl in range(d - 1, 0, -1):
                v = f"z{lvl}"
                p = f"z{lvl-1}" if lvl > 1 else pt
                expr = f"(derivative (lambda ({v}) {expr}) {p})"
            vpt = "(vector " + " ".join(f"{x}" for x in vp) + ")"
            if binding == "named":
                fn = f"gf{shape}{dim}{d}"
                lines.append(f"(define ({fn} v) {expr})")
                adfull = f"(gradient {fn} {vpt})"
            else:
                adfull = f"(gradient (lambda (v) {expr}) {vpt})"
            adexpr = f"(vref {adfull} 0)"     # component 0 (the nonzero one)
            gt = analytic_nth(shape, d) * 1.0  # coeff wrt (vref v 0) is 1
            rtol, atol = tol(d)
            lines.append(f"(chk \"{cid}\" {d} {adexpr} {gt!r} {atol} {rtol})")
            self.total_checks += 1
        return cid, "\n".join(lines)

    # -- sweep 4: jacobian OVER derivative^d (vector field) -----------------
    def probe_jacod(self, shape, dim):
        vp = VP2 if dim == 2 else VP3
        cid = self.register(f"jacod.{shape}.v{dim}.inline.vecref",
                            "jacod", shape, f"v{dim}", "inline", "vecref")
        lines = [f";; cell {cid}  (jacobian-outer over forward^d)"]
        pt = f"{T0[shape]}"
        vpt = "(vector " + " ".join(f"{x}" for x in vp) + ")"
        for d in range(1, self.max_depth + 1):
            # F_i(v) = (vref v i) * base^(d)(t0); jacobian = diag(coeff)
            comps = []
            for i in range(dim):
                var = f"z{d}"
                inner = f"(* (vref v {i}) {base_body(shape, var)})"
                expr = f"(derivative (lambda ({var}) {inner}) " \
                       f"{'z'+str(d-1) if d>1 else pt})"
                for lvl in range(d - 1, 0, -1):
                    v = f"z{lvl}"
                    p = f"z{lvl-1}" if lvl > 1 else pt
                    expr = f"(derivative (lambda ({v}) {expr}) {p})"
                comps.append(expr)
            field = "(vector " + " ".join(comps) + ")"
            adfull = f"(jacobian (lambda (v) {field}) {vpt})"
            adexpr = f"(vref (vref {adfull} 0) 0)"   # J[0][0]
            gt = analytic_nth(shape, d)
            rtol, atol = tol(d)
            lines.append(f"(chk \"{cid}\" {d} {adexpr} {gt!r} {atol} {rtol})")
            self.total_checks += 1
        return cid, "\n".join(lines)

    # -- sweep 4: hessian OVER derivative^d (scalar field) ------------------
    def probe_hessod(self, shape, dim):
        vp = VP2 if dim == 2 else VP3
        cid = self.register(f"hessod.{shape}.v{dim}.inline.vecref",
                            "hessod", shape, f"v{dim}", "inline", "vecref")
        lines = [f";; cell {cid}  (hessian-outer over forward^d)"]
        pt = f"{T0[shape]}"
        vpt = "(vector " + " ".join(f"{x}" for x in vp) + ")"
        for d in range(1, self.max_depth + 1):
            # G(v) = 0.5*(vref v 0)^2 * base^(d)(t0); hessian[0][0] = coeff
            var = f"z{d}"
            inner = f"(* 0.5 (vref v 0) (vref v 0) {base_body(shape, var)})"
            expr = f"(derivative (lambda ({var}) {inner}) " \
                   f"{'z'+str(d-1) if d>1 else pt})"
            for lvl in range(d - 1, 0, -1):
                v = f"z{lvl}"
                p = f"z{lvl-1}" if lvl > 1 else pt
                expr = f"(derivative (lambda ({v}) {expr}) {p})"
            adfull = f"(hessian (lambda (v) {expr}) {vpt})"
            adexpr = f"(vref (vref {adfull} 0) 0)"   # H[0][0]
            gt = analytic_nth(shape, d)
            rtol, atol = tol(d)
            lines.append(f"(chk \"{cid}\" {d} {adexpr} {gt!r} {atol} {rtol})")
            self.total_checks += 1
        return cid, "\n".join(lines)

    # -- orchestration ------------------------------------------------------
    def generate(self):
        # sweep 1: derivative^d
        cells = []
        for shape in ("mono", "poly", "expc", "sinc"):
            for binding in ("inline", "named", "lamvar"):
                cells.append(("deriv", "derivative", shape, binding, "capnone"))
        for shape in ("mono", "expc"):
            for cap in ("glob", "localparam", "vecref"):
                cells.append(("deriv", "derivative", shape, "inline", cap))
        self.emit_scalar_cells("deriv", cells)

        # sweep 3: gradient^d (nested reverse, scalar)
        cells = []
        for shape in ("mono", "expc"):
            for binding in ("inline", "named"):
                cells.append(("gradn", "gradient", shape, binding, "capnone"))
        cells.append(("gradn", "gradient", "mono", "inline", "vecref"))
        self.emit_scalar_cells("gradn", cells)

        # sweep 2: gradient over derivative^d, vector param (ESH-0117 family)
        items = []
        for shape in ("mono", "poly"):
            for dim in (2, 3):
                for binding in ("inline", "named"):
                    items.append(self.probe_gofd(shape, dim, binding))
        self.flush_chunks("gofd", items)

        # sweep 4a: jacobian over derivative^d (returns 0; no crash -> chunk)
        items = []
        for shape in ("mono", "expc"):
            for dim in (2, 3):
                items.append(self.probe_jacod(shape, dim))
        self.flush_chunks("compose", items)

        # sweep 4b: hessian over derivative^d — SIGSEGVs at d1, so isolate ONE
        # cell per file (an in-file crash must not mask any other cell).
        idx = 1
        for shape in ("mono", "expc"):
            for dim in (2, 3):
                self.write_file("hessod_xc", idx, [self.probe_hessod(shape, dim)])
                idx += 1

        self.write_manifest()

    def flush_chunks(self, comp, items):
        idx = 1
        for i in range(0, len(items), PROBES_PER_FILE):
            self.write_file(comp, idx, items[i:i + PROBES_PER_FILE])
            idx += 1

    def emit_scalar_cells(self, comp, cells):
        items = [self.probe_scalar(c, op, shape, binding, cap)
                 for (c, op, shape, binding, cap) in cells]
        self.flush_chunks(comp, items)

    def write_manifest(self):
        with open(os.path.join(self.outdir, "MANIFEST.txt"), "w") as f:
            f.write(f"# depth-parametric AD oracle — {len(self.files)} files, "
                    f"{self.total_checks} depth-checks, max-depth {self.max_depth}\n")
            for n in sorted(self.files):
                f.write(n + "\n")
        # cell registry consumed by scripts/ad_depth_report.py
        with open(os.path.join(self.outdir, "cells.tsv"), "w") as f:
            f.write("#cellid\tcomp\tshape\tpoint\tbinding\tcap\tfile\tmaxdepth\n")
            for cid in sorted(self.cells):
                c = self.cells[cid]
                f.write("\t".join([cid, c["comp"], c["shape"], c["point"],
                                   c["binding"], c["cap"], c["file"] or "",
                                   str(self.max_depth)]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.join(
        here, "..", "tests", "ad_depth", "generated")
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    for fn in os.listdir(outdir):
        if fn.startswith("ad_depth_") and fn.endswith(".esk"):
            os.remove(os.path.join(outdir, fn))
    g = Gen(outdir, args.max_depth)
    g.generate()
    print(f"wrote {len(g.files)} files, {g.total_checks} depth-checks -> {outdir}")


if __name__ == "__main__":
    main()
