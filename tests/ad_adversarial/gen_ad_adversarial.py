#!/usr/bin/env python3
"""Generative adversarial AD-vs-finite-difference oracle (pillar P3-gen).

Where tests/ad_oracle/ enumerates a FIXED composition matrix, this harness is a
GENERATIVE exposure engine: it grows random-but-seeded differentiable programs
out of the AD-supported primitives (+ - * / exp log sin cos tanh sqrt pow) AND
the tensor/ML operators (tensor-sum tensor-mean tensor-dot tensor-add
tensor-mul tensor-scale tensor-matmul conv2d batch-norm layer-norm
scaled-dot-attention softmax), differentiates each with Eshkol AD, and checks
every value against an in-language CENTRAL finite difference. Its job is to keep
finding NEW silent-wrong gradients, not to guard a fixed set of known repros.

Guiding principle (maintainer): "if our system does not constantly expose every
single hidden bug then it has no coverage." A zero AD gradient where the finite
difference is non-zero is the worst class of defect (silently-wrong training
gradients) and is a hard FAIL here.

Determinism: every random choice comes from `random.Random(seed)` over a FIXED
seed list, so regenerating reproduces the corpus byte-for-byte and CI is stable
even though the compiler/runtime has no RNG. Rerun with
`python3 tests/ad_adversarial/gen_ad_adversarial.py`.

Families (see README):
  scalar   random scalar expression trees -> derivative + 2nd derivative vs FD
  field    random R^n->R fields -> gradient (per component), laplacian,
           hessian diagonal, at BOTH vector and tensor-literal (#(...)) points
  gofg     vector-parameter gradient-of-gradient (the ESH-0096 silent-zero
           shape) at scalar-lifted and multi-component points
  tensor   random ML-op loss compositions -> gradient of a flattened operand vs
           FD, across literal / first-class / higher-order-wrapper loss forms,
           first AND second operand, scalar AND per-feature (vector) gamma
  htensor  higher-order over tensor ops: hessian of a tensor loss at vector and
           tensor-literal points (the ESH-0095 shape)

Verdict tokens per probe: PASS / FAIL / XKNOWN(<task>). A cell whose failure
mode kills the translation unit (SIGSEGV / IR-verify) is emitted ONE PER FILE as
ad_adv_xc_<task>_<NN>.esk and gated by KNOWN_CRASHERS so a crasher can stay
tracked without turning the whole harness red. Non-crash known-open cells are
marked chk-x with their task id. Both flip to PASS automatically once fixed.

Usage: python3 tests/ad_adversarial/gen_ad_adversarial.py [--outdir DIR]
"""

import argparse
import math
import os
import random
import sys

RTOL1 = "1e-4"      # first-order scalar/field rel tol
ATOL1 = "1e-6"
RTOL2 = "2e-3"      # second-order stencils (hessian/laplacian/2nd deriv)
ATOL2 = "1e-4"
RTOLT = "2e-3"      # nonlinear tensor ops (norm/attention/softmax/conv)
ATOLT = "1e-4"
H1 = "1e-5"
H2 = "1e-4"

# Known-open cells whose task is tracked. Non-crash cells list their id-prefix
# here to be emitted as chk-x (XKNOWN, tolerated) instead of FAIL. Crash/compile
# -fail cells list their task in KNOWN_CRASHERS so they get their own file and
# are skipped by the runner's crash isolation. Both are EMPTY on master: every
# generated shape currently matches finite differences and none crash. When the
# generator surfaces a real defect, add its id here with an ESH task so the gate
# stays green-able while a fix agent works, and it FAILs loudly if it regresses
# in a NEW way.
# ESH-0235 (found by this harness): reverse-mode AD through a tensor op used to
# return a silently-WRONG all-zero gradient when the differentiation point was
# built with the (vector ...) constructor (the identical #(...) literal /
# (tensor ...) point was correct). FIXED — a single-arg tensor-op function's
# (vector ...) point is now converted to a 1-D tensor and reverse-seeded exactly
# like the #(...)/(tensor ...) point, in every calling form (direct lambda,
# first-class named loss, named indirection, higher-order wrapper). The
# `vecpoint` family therefore drops out of XKNOWN and is a hard PASS/FAIL gate:
# any recurrence FAILs loudly.
XKNOWN = {}                          # id-prefix -> "ESH-NNNN"
KNOWN_CRASHERS = {}                  # id-prefix -> "ESH-NNNN"

SEEDS_SCALAR = list(range(0, 48))
SEEDS_FIELD = list(range(100, 132))
SEEDS_TENSOR = list(range(200, 230))


# ---------------------------------------------------------------------------
# safe random expression grammar (values stay well-conditioned near the eval
# point so the central difference is an accurate ground truth)
# ---------------------------------------------------------------------------

def fnum(rng, lo, hi):
    return f"{rng.uniform(lo, hi):.4f}"


def bounded(rng, leaf, depth):
    """Expression whose value is in [-1, 1] (sin/cos/tanh of anything)."""
    op = rng.choice(["sin", "cos", "tanh"])
    return f"({op} {sub(rng, leaf, depth - 1)})"


def safe_pos(rng, leaf, depth):
    """Strictly-positive, away-from-zero expression for log/sqrt/pow/div."""
    c = rng.uniform(1.6, 2.6)
    a = rng.uniform(0.1, 0.3)
    return f"(+ {c:.4f} (* {a:.4f} {bounded(rng, leaf, depth)}))"


def sub(rng, leaf, depth):
    """A moderate-magnitude differentiable expression in the given leaf(s)."""
    if depth <= 0:
        return leaf(rng) if rng.random() < 0.75 else fnum(rng, -1.5, 1.5)
    kind = rng.choice([
        "leaf", "add", "sub", "mul", "div",
        "exp", "log", "sin", "cos", "tanh", "sqrt", "pow",
    ])
    if kind == "leaf":
        return leaf(rng) if rng.random() < 0.8 else fnum(rng, -1.5, 1.5)
    if kind in ("add", "sub"):
        s = {"add": "+", "sub": "-"}[kind]
        return (f"({s} (* {fnum(rng, -1.2, 1.2)} {sub(rng, leaf, depth - 1)})"
                f" (* {fnum(rng, -1.2, 1.2)} {sub(rng, leaf, depth - 1)}))")
    if kind == "mul":
        return (f"(* {fnum(rng, -1.0, 1.0)} {sub(rng, leaf, depth - 1)}"
                f" {sub(rng, leaf, depth - 1)})")
    if kind == "div":
        return f"(/ {sub(rng, leaf, depth - 1)} {safe_pos(rng, leaf, depth)})"
    if kind == "exp":
        return f"(exp (* 0.3 {bounded(rng, leaf, depth)}))"
    if kind in ("sin", "cos", "tanh"):
        return f"({kind} {sub(rng, leaf, depth - 1)})"
    if kind == "log":
        return f"(log {safe_pos(rng, leaf, depth)})"
    if kind == "sqrt":
        return f"(sqrt {safe_pos(rng, leaf, depth)})"
    if kind == "pow":
        return f"(pow {safe_pos(rng, leaf, depth)} {fnum(rng, 0.5, 3.0)})"
    raise AssertionError(kind)


# ---------------------------------------------------------------------------
# probe container
# ---------------------------------------------------------------------------

class Gen:
    def __init__(self):
        self.probes = []
        self.n = 0

    def uid(self):
        self.n += 1
        return self.n

    def add(self, section, pid, lines, nchk, xc=None, crash=None):
        self.probes.append(dict(section=section, id=pid,
                                lines=[f";; probe {pid}"] + lines + [""],
                                nchk=nchk, xc=xc, crash=crash))

    def xc_for(self, pid):
        for pref, task in XKNOWN.items():
            if pid.startswith(pref):
                return task
        return None

    def crash_for(self, pid):
        for pref, task in KNOWN_CRASHERS.items():
            if pid.startswith(pref):
                return task
        return None

    def chk(self, pid, ad, fd, rtol, atol, xc=None):
        if xc:
            return f'(chk-x "{pid}" "{xc}" {ad} {fd} {atol} {rtol})'
        return f'(chk "{pid}" {ad} {fd} {atol} {rtol})'

    # -- FD stencils (scalar / component) ---------------------------------
    def fd1_scalar(self, f, x):
        return (f"(/ (- ({f} (+ {x} {H1})) ({f} (- {x} {H1}))) (* 2.0 {H1}))")

    def fd2_scalar(self, f, x):
        return (f"(/ (+ (- ({f} (+ {x} {H2})) (* 2.0 ({f} {x}))) "
                f"({f} (- {x} {H2}))) (* {H2} {H2}))")

    # ======================================================================
    # family: scalar
    # ======================================================================
    def gen_scalar(self):
        def leaf(rng):
            return "x"
        for seed in SEEDS_SCALAR:
            rng = random.Random(seed)
            depth = rng.randint(3, 6)
            x0 = fnum(rng, 0.4, 1.6)
            # guarantee a genuine x-dependence so a silent zero is caught
            body = (f"(+ (* {fnum(rng, 0.5, 1.5)} x) "
                    f"{sub(rng, leaf, depth)})")
            u = self.uid()
            lines = [f"(define (f{u} x) {body})"]
            lines.append(f"(define ad{u} (derivative f{u} {x0}))")
            lines.append(f"(define fd{u} {self.fd1_scalar(f'f{u}', x0)})")
            pid = f"scalar.s{seed}.d1"
            lines.append(self.chk(pid, f"ad{u}", f"fd{u}", RTOL1, ATOL1,
                                  xc=self.xc_for(pid)))
            # second derivative (derivative-of-derivative), FD of AD-derivative
            lines.append(f"(define (df{u} x) (derivative f{u} x))")
            lines.append(f"(define ad2{u} (derivative df{u} {x0}))")
            lines.append(f"(define fd2{u} {self.fd1_scalar(f'df{u}', x0)})")
            pid2 = f"scalar.s{seed}.d2"
            lines.append(self.chk(pid2, f"ad2{u}", f"fd2{u}", RTOL2, ATOL2,
                                  xc=self.xc_for(pid2)))
            self.add("scalar", pid, lines, 2, xc=self.xc_for(pid))

    # ======================================================================
    # family: field  (R^n -> R)
    # ======================================================================
    def field_body(self, rng, n, depth):
        def leaf(r):
            return f"(vref v {r.randint(0, n - 1)})"
        lin = " ".join(
            f"(* {fnum(rng, 0.4, 1.4)} (vref v {i}))" for i in range(n))
        return f"(+ (+ {lin}) {sub(rng, leaf, depth)})"

    def gen_field(self):
        for seed in SEEDS_FIELD:
            rng = random.Random(seed)
            n = rng.choice([2, 3])
            depth = rng.randint(3, 5)
            pt = [fnum(rng, 0.4, 1.5) for _ in range(n)]
            body = self.field_body(rng, n, depth)
            u = self.uid()
            lines = [f"(define (f{u} v) {body})"]
            # points: a vector AND a tensor-literal (exercises the reverse-tape
            # tensor-point path that ESH-0095 crashed on).
            for tag, ctor in (("v", "vector"), ("t", "tensor")):
                if ctor == "tensor":
                    pexpr = f"(tensor {n} {' '.join(pt)})"
                else:
                    pexpr = f"(vector {' '.join(pt)})"
                # gradient, per-component vs central FD
                lines.append(f"(define g{u}{tag} (gradient f{u} {pexpr}))")
                pid = f"field.s{seed}.{tag}{n}.grad"
                lines.append(
                    f"(chk-grad \"{pid}\" g{u}{tag} f{u} "
                    f"(vector {' '.join(pt)}) {RTOL2} {H2} "
                    f"{self._xc_arg(pid)})")
                # laplacian vs sum of FD second diagonals
                lap_pid = f"field.s{seed}.{tag}{n}.lap"
                lines.append(f"(define lap{u}{tag} (laplacian f{u} {pexpr}))")
                lines.append(
                    f"(define flap{u}{tag} "
                    f"(fd-laplacian f{u} (vector {' '.join(pt)}) {n} {H2}))")
                lines.append(self.chk(lap_pid, f"lap{u}{tag}",
                                      f"flap{u}{tag}", RTOL2, ATOL2,
                                      xc=self.xc_for(lap_pid)))
            self.add("field", f"field.s{seed}", lines, 4)

    def _xc_arg(self, pid):
        t = self.xc_for(pid)
        return f'"{t}"' if t else "#f"

    # ======================================================================
    # family: gofg  (vector-parameter gradient-of-gradient, ESH-0096 shape)
    # ======================================================================
    def gen_gofg(self):
        cases = [
            # (id, inner-body over w, point comps)
            ("cube1", "(* (vref w 0) (vref w 0) (vref w 0))", ["2.0"]),
            ("mix2", "(* (vref w 0) (vref w 0) (vref w 1))", ["1.3", "-0.7"]),
            ("rat2", "(/ 1.0 (+ 1.0 (* (vref w 0) (vref w 0)) "
                     "(* (vref w 1) (vref w 1))))", ["1.1", "0.6"]),
            ("es2", "(* (exp (* 0.3 (vref w 0))) (sin (vref w 1)))",
             ["0.7", "1.2"]),
        ]
        for cid, ibody, comps in cases:
            n = len(comps)
            u = self.uid()
            pv = f"(vector {' '.join(comps)})"
            inner = f"(gradient (lambda (w) {ibody}) v)"
            # outer differentiates component 0 of the inner gradient
            g = f"(lambda (v) (vref {inner} 0))"
            lines = [f"(define (g{u} v) (vref {inner} 0))"]
            lines.append(f"(define ad{u} (gradient {g} {pv}))")
            pid = f"gofg.{cid}.v{n}"
            lines.append(
                f"(chk-grad \"{pid}\" ad{u} g{u} {pv} {RTOL2} {H2} "
                f"{self._xc_arg(pid)})")
            self.add("gofg", pid, lines, n)

    # ======================================================================
    # family: tensor  (ML-op loss compositions)
    # ======================================================================
    def _tensor_probe(self, pid, setup, loss_expr, x0_expr, x0_vec,
                      rtol=RTOLT, section="tensor"):
        """Emit literal / first-class / wrapper gradient checks for a loss.

        loss_expr is a lambda-parameterised body string using `z` as the
        operand.  x0_expr is the AD evaluation point (a tensor);
        x0_vec is the equivalent Scheme vector used by the FD oracle.
        """
        u = self.uid()
        lines = list(setup)
        lines.append(f"(define (loss{u} z) {loss_expr})")
        xc = self._xc_arg(pid)
        # literal, first-class variable, higher-order wrapper — all vs FD.
        lines.append(f"(define lit{u} (gradient (lambda (z) (loss{u} z)) "
                     f"{x0_expr}))")
        lines.append(f"(define var{u} (gradient loss{u} {x0_expr}))")
        lines.append(f"(define wrp{u} (wg loss{u} {x0_expr}))")
        lines.append(f"(chk-grad \"{pid}.literal\" lit{u} loss{u} "
                     f"{x0_vec} {rtol} {H2} {xc})")
        lines.append(f"(chk-grad \"{pid}.firstclass\" var{u} loss{u} "
                     f"{x0_vec} {rtol} {H2} {xc})")
        lines.append(f"(chk-grad \"{pid}.wrapper\" wrp{u} loss{u} "
                     f"{x0_vec} {rtol} {H2} {xc})")
        self.add(section, pid, lines, 3, xc=self.xc_for(pid))

    def gen_tensor(self):
        for seed in SEEDS_TENSOR:
            rng = random.Random(seed)
            op = SEEDS_TENSOR.index(seed) % len(TENSOR_BUILDERS)
            TENSOR_BUILDERS[op](self, seed, rng)

    # ---- individual ML-op builders (each guarantees a non-zero gradient by
    #      folding the op result against a random weight tensor) -----------
    def _rand_vec(self, rng, k, lo=-1.5, hi=1.5):
        return [f"{rng.uniform(lo, hi):.4f}" for _ in range(k)]

    def t_matmul(self, seed, rng):
        A = self._rand_vec(rng, 4)
        W = self._rand_vec(rng, 4)
        B = self._rand_vec(rng, 4)
        setup = [f"(define Am{seed} (reshape (vector {' '.join(A)}) 2 2))",
                 f"(define Wm{seed} (vector {' '.join(W)}))"]
        # differentiate the SECOND operand (B) of A @ B
        loss = (f"(tensor-sum (tensor-mul (tensor-matmul Am{seed} "
                f"(reshape z 2 2)) (reshape Wm{seed} 2 2)))")
        self._tensor_probe(f"tensor.s{seed}.matmulB", setup, loss,
                           f"(tensor 4 {' '.join(B)})",
                           f"(vector {' '.join(B)})", rtol=RTOL1)
        # differentiate the FIRST operand (A) of z @ Bfix
        u = self.uid()
        setup2 = [f"(define Bf{seed} (reshape (vector {' '.join(B)}) 2 2))",
                  f"(define Wm2{seed} (vector {' '.join(W)}))"]
        loss2 = (f"(tensor-sum (tensor-mul (tensor-matmul (reshape z 2 2) "
                 f"Bf{seed}) (reshape Wm2{seed} 2 2)))")
        self._tensor_probe(f"tensor.s{seed}.matmulA", setup2, loss2,
                           f"(tensor 4 {' '.join(A)})",
                           f"(vector {' '.join(A)})", rtol=RTOL1)

    def t_conv(self, seed, rng):
        img = self._rand_vec(rng, 9)
        ker = self._rand_vec(rng, 4)
        W = self._rand_vec(rng, 4)   # output is 2x2 for 3x3 img, 2x2 kernel
        setup = [f"(define Im{seed} (reshape (vector {' '.join(img)}) 3 3))",
                 f"(define Wc{seed} (reshape (vector {' '.join(W)}) 2 2))"]
        loss = (f"(tensor-sum (tensor-mul (conv2d Im{seed} (reshape z 2 2) 1) "
                f"Wc{seed}))")
        self._tensor_probe(f"tensor.s{seed}.conv2dKernel", setup, loss,
                           f"(tensor 4 {' '.join(ker)})",
                           f"(vector {' '.join(ker)})", rtol=RTOL1)

    def t_attention(self, seed, rng):
        Q = self._rand_vec(rng, 4)
        K = self._rand_vec(rng, 4)
        V = self._rand_vec(rng, 4)
        W = self._rand_vec(rng, 4)
        setup = [f"(define Qa{seed} (reshape (vector {' '.join(Q)}) 2 2))",
                 f"(define Ka{seed} (reshape (vector {' '.join(K)}) 2 2))",
                 f"(define Wa{seed} (reshape (vector {' '.join(W)}) 2 2))"]
        # differentiate V (linear operand -> well-conditioned) and K (softmax)
        lossV = (f"(tensor-sum (tensor-mul (scaled-dot-attention Qa{seed} "
                 f"Ka{seed} (reshape z 2 2)) Wa{seed}))")
        self._tensor_probe(f"tensor.s{seed}.attnV", setup, lossV,
                           f"(tensor 4 {' '.join(V)})",
                           f"(vector {' '.join(V)})")
        setup2 = [f"(define Qb{seed} (reshape (vector {' '.join(Q)}) 2 2))",
                  f"(define Vb{seed} (reshape (vector {' '.join(V)}) 2 2))",
                  f"(define Wb{seed} (reshape (vector {' '.join(W)}) 2 2))"]
        lossK = (f"(tensor-sum (tensor-mul (scaled-dot-attention Qb{seed} "
                 f"(reshape z 2 2) Vb{seed}) Wb{seed}))")
        self._tensor_probe(f"tensor.s{seed}.attnK", setup2, lossK,
                           f"(tensor 4 {' '.join(K)})",
                           f"(vector {' '.join(K)})")

    def t_norm(self, seed, rng):
        xin = self._rand_vec(rng, 3, 0.5, 3.0)
        gam = self._rand_vec(rng, 3, 0.5, 2.0)
        W = self._rand_vec(rng, 3)
        for opname, tag in (("batch-norm", "bn"), ("layer-norm", "ln")):
            setup = [f"(define Xn{seed}{tag} (tensor 3 {' '.join(xin)}))",
                     f"(define Wn{seed}{tag} (tensor 3 {' '.join(W)}))"]
            # differentiate the per-feature (vector) gamma — the silent-zero
            # shape from ESH-0212.
            loss = (f"(tensor-sum (tensor-mul ({opname} Xn{seed}{tag} "
                    f"(reshape z 3) 0.0 0.00001) Wn{seed}{tag}))")
            self._tensor_probe(f"tensor.s{seed}.{tag}VecGamma", setup, loss,
                               f"(tensor 3 {' '.join(gam)})",
                               f"(vector {' '.join(gam)})")

    def t_reduce(self, seed, rng):
        vals = self._rand_vec(rng, 4, 0.4, 2.5)
        W = self._rand_vec(rng, 4)
        setup = [f"(define Wr{seed} (tensor 4 {' '.join(W)}))"]
        # softmax then weighted mean — nonlinear, coupled across all entries.
        loss = (f"(tensor-mean (tensor-mul (softmax z) Wr{seed}))")
        self._tensor_probe(f"tensor.s{seed}.softmaxMean", setup, loss,
                           f"(tensor 4 {' '.join(vals)})",
                           f"(vector {' '.join(vals)})")
        # dot with a random constant vector (pure linear -> exact). Point given
        # as a (tensor ...) value, which the tensor reverse path seeds
        # correctly; the (vector ...)-point silent-zero defect is covered
        # deterministically (and tracked) by the vecpoint family below.
        setup2 = [f"(define Cr{seed} (vector {' '.join(W)}))"]
        loss2 = f"(tensor-dot z Cr{seed})"
        self._tensor_probe(f"tensor.s{seed}.dot", setup2, loss2,
                           f"(tensor 4 {' '.join(vals)})",
                           f"(vector {' '.join(vals)})", rtol=RTOL1)

    # ======================================================================
    # family: vecpoint  (ESH-0235 — tensor-op gradient at a (vector ...) point)
    # ======================================================================
    def gen_vecpoint(self):
        """Differentiate tensor-op losses at a point built by (vector ...).

        The identical point as a #(...) literal or a (tensor ...) value gives
        the correct gradient; the (vector ...) constructor silently yields a
        zero gradient (ESH-0235). All probes here are XKNOWN via the "vecpoint"
        prefix and flip to PASS once the tensor reverse path seeds a
        (vector ...) point like a (tensor ...) one.
        """
        cases = [
            ("dot", "(tensor-dot z Cvp)",
             ["1.2335", "0.5795", "1.6262", "0.6787"], 4,
             ["(define Cvp (vector 2.0 -1.0 0.5 -0.75))"]),
            ("summul", "(tensor-sum (tensor-mul z Cvp2))",
             ["1.0", "2.0", "4.0"], 3,
             ["(define Cvp2 (vector 0.5 -1.5 0.25))"]),
            ("matmul", "(tensor-sum (tensor-matmul (reshape z 2 2) Kvp))",
             ["1.0", "2.0", "3.0", "4.0"], 4,
             ["(define Kvp (reshape (vector 1.0 0.0 0.0 1.0) 2 2))"]),
        ]
        for cid, loss, comps, n, setup in cases:
            pid = f"vecpoint.{cid}"
            self._tensor_probe(pid, setup, loss,
                               f"(vector {' '.join(comps)})",
                               f"(vector {' '.join(comps)})", rtol=RTOL1,
                               section="vecpoint")

    # ======================================================================
    # family: exactpoint  (ESH-0393 — EXACT evaluation points at the jet/tape
    #                      entry points)
    # ======================================================================
    def gen_exactpoint(self):
        """Differentiate at an EXACT point and check against FD at the double.

        The jet and the tape carry a raw double per component, so every AD entry
        point coerces its point at the boundary. That coercion used to
        REINTERPRET the tagged data field, which is correct only for an
        immediate: an exact rational or bignum is HEAP-tagged, so its data field
        is a POINTER and the substrate differentiated at the object's ADDRESS
        (`(derivative (lambda (x) (* x x)) 1/3)` => 1.0e10, heap magnitude) or at
        a ~5e-314 denormal, or the point was misclassified as a COLLECTION and
        its object read as [dims][rank][elems].

        The oracle is the whole point of this family: AD is evaluated at the
        EXACT point, FD at `(exact->inexact <same point>)`. A fabricated seed --
        an address, a denormal or a silent zero -- cannot agree with a finite
        difference taken at the value the point actually denotes, whatever its
        magnitude. Exact and inexact seeds must name the same derivative.
        """
        # Exact rationals with well-conditioned doubles, plus one bignum and one
        # small exact integer (the immediate case, which must not regress).
        #
        # `fd` marks the points a finite difference can actually resolve. At a
        # bignum point (~1.2e20) every representable step satisfies x + h == x in
        # double precision, so the FD stencil is identically 0 and would fail a
        # CORRECT derivative. The exact-vs-inexact-seed comparison below is the
        # controlling oracle at every magnitude and is applied to all points; FD
        # is added only where it is meaningful.
        points = [
            ("rat13",  "1/3",                     True),
            ("rat25",  "2/5",                     True),
            ("rat34",  "3/4",                     True),
            ("rat78",  "7/8",                     True),
            ("big",    "123456789012345678901",   False),
            ("int3",   "3",                        True),
        ]

        for tag, p, use_fd in points:
            u = self.uid()
            px = f"(exact->inexact {p})"
            lines = [f"(define (fx{u} x) (+ (* 1.5 x) (* x x) (* 0.5 x x x)))"]

            # -- derivative at a scalar exact point ------------------------
            # The controlling oracle: the SAME operator at the inexact seed.
            # A fabricated seed (heap address, ~5e-314 denormal, silent zero)
            # cannot agree with it, at any magnitude.
            lines.append(f"(define ad{u} (derivative fx{u} {p}))")
            lines.append(f"(define ai{u} (derivative fx{u} {px}))")
            pid = f"exactpoint.{tag}.derivative"
            lines.append(self.chk(pid, f"ad{u}", f"ai{u}", ATOL1, RTOL1,
                                  xc=self.xc_for(pid)))
            nchk = 1
            if use_fd:
                lines.append(f"(define fd{u} {self.fd1_scalar(f'fx{u}', px)})")
                pidf = f"exactpoint.{tag}.derivative-fd"
                lines.append(self.chk(pidf, f"ad{u}", f"fd{u}", ATOL1, RTOL1,
                                      xc=self.xc_for(pidf)))
                nchk += 1

            # An exact point reached through a VARIABLE: exact only at runtime,
            # so this cannot be answered by constant folding.
            lines.append(f"(define pv{u} {p})")
            lines.append(f"(define adv{u} (derivative fx{u} pv{u}))")
            pidv = f"exactpoint.{tag}.derivative-var"
            lines.append(self.chk(pidv, f"adv{u}", f"ai{u}", ATOL1, RTOL1,
                                  xc=self.xc_for(pidv)))
            nchk += 1

            # -- gradient at a bare exact scalar point ---------------------
            # `is_scalar` enumerated INT64/DOUBLE, so an exact heap scalar was
            # routed down the collection path; the ENTRY and EXIT classifications
            # must also agree on the return SHAPE (a bare scalar, not #(x)) —
            # comparing against the inexact seed's result asserts both.
            lines.append(f"(define gs{u} (gradient fx{u} {p}))")
            lines.append(f"(define gi{u} (gradient fx{u} {px}))")
            pidg = f"exactpoint.{tag}.gradient-scalar"
            lines.append(self.chk(pidg, f"gs{u}", f"gi{u}", ATOL1, RTOL1,
                                  xc=self.xc_for(pidg)))
            nchk += 1

            self.add("exactpoint", pid, lines, nchk, xc=self.xc_for(pid))

            # -- collection point forms, one probe per constructor ----------
            # (vector …) and (list …) both reach the tape. #(…) and (tensor …)
            # are NOT covered: those literal constructors drop an exact rational
            # to 0 before AD ever sees the point (`(display (tensor 1/3))` =>
            # #(0)), which is a tensor-literal defect, not an AD one.
            for ctor in ("vector", "list"):
                u2 = self.uid()
                lines2 = [f"(define (fv{u2} v) "
                          f"(+ (* 1.5 (vref v 0)) (* (vref v 0) (vref v 0))))"]
                lines2.append(f"(define g{u2} (gradient fv{u2} ({ctor} {p})))")
                lines2.append(f"(define gr{u2} (gradient fv{u2} ({ctor} {px})))")
                pid2 = f"exactpoint.{tag}.gradient-{ctor}"
                lines2.append(self.chk(pid2, f"(vector-ref g{u2} 0)",
                                       f"(vector-ref gr{u2} 0)", ATOL1, RTOL1,
                                       xc=self.xc_for(pid2)))
                n2 = 1
                if use_fd:
                    pid2f = f"exactpoint.{tag}.gradient-{ctor}-fd"
                    lines2.append(f"(chk-grad \"{pid2f}\" g{u2} fv{u2} "
                                  f"(vector {px}) {RTOL1} {H2} "
                                  f"{self._xc_arg(pid2f)})")
                    n2 += 1
                self.add("exactpoint", pid2, lines2, n2, xc=self.xc_for(pid2))

            # -- MIXED vector: exact rational, double, exact integer ---------
            # Each slot is coerced independently, so one misread slot cannot
            # hide behind its neighbours.
            u3 = self.uid()
            lines3 = [f"(define (fm{u3} v) (+ (* (vref v 0) (vref v 0)) "
                      f"(* 2.0 (vref v 1) (vref v 1)) "
                      f"(* 0.5 (vref v 2) (vref v 2))))"]
            lines3.append(f"(define gm{u3} (gradient fm{u3} (vector {p} 0.5 2)))")
            lines3.append(f"(define gn{u3} (gradient fm{u3} (vector {px} 0.5 2.0)))")
            pid3 = f"exactpoint.{tag}.gradient-mixed"
            n3 = 0
            for slot in (0, 1, 2):
                pid3s = f"{pid3}-slot{slot}"
                lines3.append(self.chk(pid3s, f"(vector-ref gm{u3} {slot})",
                                       f"(vector-ref gn{u3} {slot})",
                                       ATOL1, RTOL1, xc=self.xc_for(pid3s)))
                n3 += 1
            if use_fd:
                pid3f = f"{pid3}-fd"
                lines3.append(f"(chk-grad \"{pid3f}\" gm{u3} fm{u3} "
                              f"(vector {px} 0.5 2.0) {RTOL1} {H2} "
                              f"{self._xc_arg(pid3f)})")
                n3 += 1
            self.add("exactpoint", pid3, lines3, n3, xc=self.xc_for(pid3))

            # -- second-order and the remaining multivariate operators ------
            # x^4 so the second derivative DEPENDS on the point (12x^2); x^2
            # would give the constant 2 and pass even with a fabricated seed.
            u4 = self.uid()
            lines4 = [f"(define (fq{u4} v) (* (vref v 0) (vref v 0) "
                      f"(vref v 0) (vref v 0)))"]
            lines4.append(f"(define hs{u4} "
                          f"(vector-ref (vector-ref (hessian fq{u4} (vector {p})) 0) 0))")
            lines4.append(f"(define hi{u4} "
                          f"(vector-ref (vector-ref (hessian fq{u4} (vector {px})) 0) 0))")
            pid4 = f"exactpoint.{tag}.hessian"
            lines4.append(self.chk(pid4, f"hs{u4}", f"hi{u4}", ATOL2, RTOL2,
                                   xc=self.xc_for(pid4)))
            lines4.append(f"(define lp{u4} (laplacian fq{u4} (vector {p})))")
            lines4.append(f"(define li{u4} (laplacian fq{u4} (vector {px})))")
            pid5 = f"exactpoint.{tag}.laplacian"
            lines4.append(self.chk(pid5, f"lp{u4}", f"li{u4}", ATOL2, RTOL2,
                                   xc=self.xc_for(pid5)))
            lines4.append(f"(define dd{u4} "
                          f"(directional-derivative fq{u4} (vector {p}) (vector 1)))")
            lines4.append(f"(define di{u4} "
                          f"(directional-derivative fq{u4} (vector {px}) (vector 1)))")
            pid6 = f"exactpoint.{tag}.directional-derivative"
            lines4.append(self.chk(pid6, f"dd{u4}", f"di{u4}", ATOL1, RTOL1,
                                   xc=self.xc_for(pid6)))
            n4 = 3
            if use_fd:
                lines4.append(f"(define hfd{u4} (fd-diag fq{u4} (vector {px}) 0 {H2}))")
                pid4f = f"exactpoint.{tag}.hessian-fd"
                lines4.append(self.chk(pid4f, f"hs{u4}", f"hfd{u4}", ATOL2, RTOL2,
                                       xc=self.xc_for(pid4f)))
                lines4.append(f"(define dfd{u4} (fd-comp fq{u4} (vector {px}) 0 {H2}))")
                pid6f = f"exactpoint.{tag}.directional-derivative-fd"
                lines4.append(self.chk(pid6f, f"dd{u4}", f"dfd{u4}", ATOL1, RTOL1,
                                       xc=self.xc_for(pid6f)))
                n4 += 2
            self.add("exactpoint", pid4, lines4, n4, xc=self.xc_for(pid4))

            # -- derivative-n / taylor keep the EXACT tier ------------------
            # Not an FD comparison: these must come back EXACT, so assert the
            # exactness tag and the exact value. This is the tier
            # docs/guide/AUTOMATIC_DIFFERENTIATION.md section 3 documents, and
            # it must not degrade when the entry-point coercion changes.
            u5 = self.uid()
            lines5 = [f"(define (fs{u5} x) (* x x))"]
            lines5.append(f"(define dn{u5} (derivative-n fs{u5} {p} 1))")
            pid7 = f"exactpoint.{tag}.derivative-n-exact"
            lines5.append(f"(chk-exact \"{pid7}\" dn{u5} (* 2 {p}) "
                          f"{self._xc_arg(pid7)})")
            self.add("exactpoint", pid7, lines5, 1, xc=self.xc_for(pid7))

    # ======================================================================
    # family: htensor  (higher-order over tensor ops, ESH-0095 shape)
    # ======================================================================
    def gen_htensor(self):
        # hessian of a scalar tensor loss at BOTH a vector and a tensor point
        cases = [
            ("quad", "(tensor-sum (tensor-mul v v))",
             ["1.3", "-0.7", "0.6"]),
            ("rat", "(/ 1.0 (+ 1.0 (tensor-dot v v)))",
             ["0.8", "1.1"]),
            ("es", "(+ (* (exp (* 0.3 (vref v 0))) (sin (vref v 1))) "
                   "(* (vref v 0) (vref v 1)))", ["0.7", "1.2"]),
            # ESH-0121: reshape a vector into a 2x2 tensor and tensor-matmul.
            # reshape used to flatten the forward-mode dual jets to plain doubles
            # and the 2-D matmul was reverse-mode-only, so this Hessian silently
            # collapsed to all-zero. Now exact via the dual-tensor matmul path.
            ("mmsq", "(tensor-sum (tensor-matmul (reshape v 2 2) (reshape v 2 2)))",
             ["1.0", "2.0", "3.0", "4.0"]),
        ]
        for cid, body, comps in cases:
            n = len(comps)
            for tag, ctor in (("v", "vector"), ("t", "tensor")):
                u = self.uid()
                if ctor == "tensor":
                    pexpr = f"(tensor {n} {' '.join(comps)})"
                else:
                    pexpr = f"(vector {' '.join(comps)})"
                lines = [f"(define (h{u} v) {body})"]
                lines.append(f"(define H{u} (hessian h{u} {pexpr}))")
                pid = f"htensor.{cid}.{tag}{n}.hess"
                lines.append(
                    f"(chk-hess \"{pid}\" H{u} h{u} "
                    f"(vector {' '.join(comps)}) {n} {RTOL2} {H2} "
                    f"{self._xc_arg(pid)})")
                self.add("htensor", pid, lines, n * n)

    def generate(self):
        self.gen_scalar()
        self.gen_field()
        self.gen_gofg()
        self.gen_tensor()
        self.gen_vecpoint()
        self.gen_exactpoint()
        self.gen_htensor()
        return self.probes


# builders indexed round-robin across the tensor seeds
TENSOR_BUILDERS = [
    Gen.t_matmul, Gen.t_conv, Gen.t_attention, Gen.t_norm, Gen.t_reduce,
]


# ---------------------------------------------------------------------------
# in-language prelude: chk/chk-x plus vector/hessian FD harnesses
# ---------------------------------------------------------------------------

PRELUDE = r""";; GENERATED by tests/ad_adversarial/gen_ad_adversarial.py — DO NOT EDIT.
;; Generative adversarial AD-vs-finite-difference oracle (pillar P3-gen).
;; Every AD value below is checked against an in-language central finite
;; difference; a zero AD gradient where FD is non-zero is a hard FAIL.
(require stdlib)
(define op-pass 0)
(define op-fail 0)
(define op-xknown 0)
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

;; ESH-0393: the exact tier is not an FD comparison — derivative-n/taylor at an
;; exact seed must come back EXACT, so assert both the exactness tag and the
;; value. task=#f -> PASS/FAIL; task=string -> XKNOWN on mismatch.
(define (chk-exact id got expect task)
  (if (and (exact? got) (= got expect))
      (begin (set! op-pass (+ op-pass 1))
             (display "PASS: ") (display id) (newline))
      (if task
          (begin (set! op-xknown (+ op-xknown 1))
                 (display "XKNOWN: ") (display id)
                 (display " (") (display task)
                 (display ") got=") (display got)
                 (display " exact?=") (display (exact? got))
                 (display " expect=") (display expect) (newline))
          (begin (set! op-fail (+ op-fail 1))
                 (display "FAIL: ") (display id)
                 (display " got=") (display got)
                 (display " exact?=") (display (exact? got))
                 (display " expect=") (display expect) (newline)))))

;; perturb component i of a Scheme vector by d (fresh copy).
(define (perturb-vector v i d)
  (let ((n (vector-length v)) (out (make-vector (vector-length v) 0.0)))
    (letrec ((lp (lambda (k)
                   (if (< k n)
                       (begin (vector-set! out k
                                (if (= k i) (+ (vector-ref v k) d)
                                    (vector-ref v k)))
                              (lp (+ k 1)))
                       out))))
      (lp 0))))

;; central first difference of scalar-valued f at component i.
(define (fd-comp f v i h)
  (/ (- (f (perturb-vector v i h)) (f (perturb-vector v i (- h))))
     (* 2.0 h)))

;; central second difference (diagonal) of f at component i.
(define (fd-diag f v i h)
  (/ (+ (- (f (perturb-vector v i h)) (* 2.0 (f v)))
        (f (perturb-vector v i (- h))))
     (* h h)))

;; mixed second difference d2 f / dxi dxj.
(define (fd-cross f v i j h)
  (let ((vpp (perturb-vector (perturb-vector v i h) j h))
        (vpm (perturb-vector (perturb-vector v i h) j (- h)))
        (vmp (perturb-vector (perturb-vector v i (- h)) j h))
        (vmm (perturb-vector (perturb-vector v i (- h)) j (- h))))
    (/ (- (+ (f vpp) (f vmm)) (+ (f vpm) (f vmp))) (* 4.0 h h))))

;; sum of second diagonals = laplacian ground truth.
(define (fd-laplacian f v n h)
  (letrec ((lp (lambda (i acc)
                 (if (< i n) (lp (+ i 1) (+ acc (fd-diag f v i h))) acc))))
    (lp 0 0.0)))

;; higher-order wrapper: forces the loss to arrive as a runtime function value.
(define (wg f x) (gradient f x))

(define (any-nonzero? g)
  (let ((n (vector-length g)))
    (letrec ((lp (lambda (i)
                   (if (< i n)
                       (if (> (abs (vector-ref g i)) 1e-9) #t (lp (+ i 1)))
                       #f))))
      (lp 0))))

;; Check a whole AD gradient vector g of loss f at point v (a Scheme vector)
;; against central finite differences.  task=#f -> PASS/FAIL; task=string ->
;; XKNOWN on mismatch (tracked open bug).  A gradient that is identically zero
;; where FD is non-zero is the silent-wrong class and fails here.
(define (chk-grad id g f v rtol h task)
  (let* ((n (vector-length g)))
    (letrec ((lp (lambda (i bad firstad firstfd)
       (if (< i n)
           (let* ((fd (fd-comp f v i h))
                  (gi (vector-ref g i))
                  (ok (<= (abs (- gi fd)) (+ 1e-4 (* rtol (+ 1.0 (abs fd)))))))
             (if ok
                 (lp (+ i 1) bad firstad firstfd)
                 (lp (+ i 1) #t (if bad firstad gi) (if bad firstfd fd))))
           (begin
             ;; a genuine all-zero AD gradient where FD moves is a silent bug
             (if (and (not bad) (not (any-nonzero? g))
                      (> (abs (fd-comp f v 0 h)) 1e-6))
                 (set! bad #t))
             (if bad
                 (if task
                     (begin (set! op-xknown (+ op-xknown 1))
                            (display "XKNOWN: ") (display id)
                            (display " (") (display task) (display ") ad=")
                            (display g) (display " fd0=") (display firstfd)
                            (newline))
                     (begin (set! op-fail (+ op-fail 1))
                            (display "FAIL: ") (display id)
                            (display " ad=") (display g)
                            (display " fd0=") (display firstfd) (newline)))
                 (begin (set! op-pass (+ op-pass 1))
                        (display "PASS: ") (display id) (newline))))))))
      (lp 0 #f 0.0 0.0))))

;; Check a full AD hessian H of f at v against central second differences.
(define (chk-hess id H f v n rtol h task)
  (letrec ((row (lambda (i bad)
     (if (< i n)
         (letrec ((col (lambda (j b)
            (if (< j n)
                (let* ((fd (if (= i j) (fd-diag f v i h)
                               (fd-cross f v i j h)))
                       (hij (tensor-ref H i j))
                       (ok (<= (abs (- hij fd))
                               (+ 1e-3 (* rtol (+ 1.0 (abs fd)))))))
                  (col (+ j 1) (or b (not ok))))
                b))))
           (row (+ i 1) (col 0 bad)))
         (if bad
             (if task
                 (begin (set! op-xknown (+ op-xknown 1))
                        (display "XKNOWN: ") (display id)
                        (display " (") (display task) (display ")") (newline))
                 (begin (set! op-fail (+ op-fail 1))
                        (display "FAIL: ") (display id)
                        (display " H=") (display H) (newline)))
             (begin (set! op-pass (+ op-pass 1))
                    (display "PASS: ") (display id) (newline)))))))
    (row 0 #f)))
"""

SUMMARY = """\
(display "Passed: ") (display op-pass) (newline)
(display "Failed: ") (display op-fail) (newline)
(display "Xknown: ") (display op-xknown) (newline)
"""

MAX_CHKS_PER_FILE = 24


def write_files(probes, outdir):
    os.makedirs(outdir, exist_ok=True)
    for old in sorted(os.listdir(outdir)):
        if old.startswith("ad_adv_") and old.endswith(".esk"):
            os.remove(os.path.join(outdir, old))

    manifest = []

    def emit(fname, chunk):
        with open(os.path.join(outdir, fname), "w") as f:
            f.write(PRELUDE + "\n")
            for p in chunk:
                f.write("\n".join(p["lines"]) + "\n")
            f.write(SUMMARY)
        manifest.append((fname, len(chunk), sum(p["nchk"] for p in chunk)))

    crash = [p for p in probes if p["crash"]]
    normal = [p for p in probes if not p["crash"]]

    by_section = {}
    for p in normal:
        by_section.setdefault(p["section"], []).append(p)

    for section in ("scalar", "field", "gofg", "tensor", "vecpoint",
                    "exactpoint", "htensor"):
        plist = by_section.get(section, [])
        chunk, nchk, idx = [], 0, 0
        for p in plist:
            if chunk and nchk + p["nchk"] > MAX_CHKS_PER_FILE:
                idx += 1
                emit(f"ad_adv_{section}_{idx:02d}.esk", chunk)
                chunk, nchk = [], 0
            chunk.append(p)
            nchk += p["nchk"]
        if chunk:
            idx += 1
            emit(f"ad_adv_{section}_{idx:02d}.esk", chunk)

    counters = {}
    for p in crash:
        k = counters[p["crash"]] = counters.get(p["crash"], 0) + 1
        emit(f"ad_adv_xc_{p['crash']}_{k:02d}.esk", [p])

    with open(os.path.join(outdir, "MANIFEST.txt"), "w") as f:
        f.write("# file probes checks\n")
        for fname, np_, nc in manifest:
            f.write(f"{fname} {np_} {nc}\n")
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
    print(f"ad_adversarial: {len(manifest)} files, {nprobe} probes, "
          f"{nchk} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
