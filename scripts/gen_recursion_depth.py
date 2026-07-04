#!/usr/bin/env python3
"""Depth-parametric recursion/control sweep generator (pillar P6b).

For EACH recursion/control KIND, emit a ladder of self-checking probes at
increasing depth so scripts/run_recursion_depth.sh can find and gate the
maximum safe depth of that construct under BOTH the JIT (-r) and AOT.

Kinds (see .swarm/DEPTH_PARAMETRIC_TESTING.md):
  self_tail      self tail recursion            -> must be O(1) stack (1e8 ok)
  mutual_tail2   mutual tail recursion, 2-cycle -> ESH-0102 (not TCO'd)
  mutual_tail3   mutual tail recursion, 3-cycle -> ESH-0102
  non_tail       non-tail recursion (stack acc) -> ESH-0112
  cps            CPS / explicit continuation    -> clean 100000 guard (ESH-0080 fixed #93)
  through_map    recursion through map (h.o.)    -> clean 100000 guard
  metacircular   interpreted recursion (eval)   -> ESH-0119 (silent SIGILL ~30k)
  dynamic_wind   deep dynamic-wind nesting      -> clean SIGBUS diagnostic ~100k
  callcc         deep call/cc nesting           -> clean 100000 guard
  guard          deep guard nesting             -> ESH-0119 (silent SIGILL ~200k)
  stdlib_length  stdlib (length) on long list   -> ESH-0108 (non-tail length)

Every probe computes a value with a KNOWN CLOSED FORM and checks it in-language:
  tri   sum_{k=1}^{N} k = N*(N+1)/2   (the recursion sums; the oracle multiplies)
  count length N                       (stdlib_length)
so a WRONG value at any depth is detectable, not just a crash. On success a
probe prints "PASS: <kind> d=<N> v=<got>"; on a mismatch "WRONG: <kind> ...".
A crash prints neither and is classified by the runner from the exit signal
and whether any diagnostic was emitted.

Each file carries directive comments the runner reads:
  ; KIND: <kind>
  ; DEPTH: <N>
  ; EXPECT: pass | limit | xknown <ESH-task>
    pass    -> must PASS (value correct, clean exit); anything else is a gate fail
    limit   -> a documented CLEAN boundary (diagnostic + nonzero exit); a SILENT
               crash or wrong value here is a gate fail (the limit went silent)
    xknown  -> a tracked SILENT crash / wrong value tied to an ESH task; tolerated

Deterministic: no RNG, stable ordering; rerunning reproduces the corpus
byte-for-byte.

Usage: python3 scripts/gen_recursion_depth.py [--outdir DIR]
"""

import argparse
import os

# ── kind definitions ────────────────────────────────────────────────────
# defs : function definitions ({N} may be used but usually not)
# call : expression that computes the swept quantity for depth {N}
# oracle: "tri" -> N*(N+1)/2 ; "count" -> N
# stdlib: needs (require stdlib)
# ladder: list of (depth, expect) ; expect in {"pass","limit",("xknown","ESH-...")}

KINDS = {
    "self_tail": {
        "doc": "self tail recursion -- must be O(1) stack",
        "defs": "(define (rec n acc) (if (= n 0) acc (rec (- n 1) (+ acc n))))",
        "call": "(rec {N} 0)",
        "oracle": "tri",
        "stdlib": False,
        "ladder": [
            (1000000, "pass"),
            (10000000, "pass"),
            (100000000, "pass"),
        ],
    },
    "mutual_tail2": {
        "doc": "mutual tail recursion, 2-cycle (ping/pong) -- not TCO'd (ESH-0102)",
        "defs": ("(define (ping n acc) (if (= n 0) acc (pong (- n 1) (+ acc n))))\n"
                 "(define (pong n acc) (if (= n 0) acc (ping (- n 1) (+ acc n))))"),
        "call": "(ping {N} 0)",
        "oracle": "tri",
        "stdlib": False,
        "ladder": [
            (10000, "pass"),
            (100000, "pass"),
            (200000, "pass"),
            (300000, ("xknown", "ESH-0102")),
            (500000, ("xknown", "ESH-0102")),
        ],
    },
    "mutual_tail3": {
        "doc": "mutual tail recursion, 3-cycle (a/b/c) -- not TCO'd (ESH-0102)",
        "defs": ("(define (a n acc) (if (= n 0) acc (b (- n 1) (+ acc n))))\n"
                 "(define (b n acc) (if (= n 0) acc (c (- n 1) (+ acc n))))\n"
                 "(define (c n acc) (if (= n 0) acc (a (- n 1) (+ acc n))))"),
        "call": "(a {N} 0)",
        "oracle": "tri",
        "stdlib": False,
        "ladder": [
            (10000, "pass"),
            (100000, "pass"),
            (200000, "pass"),
            (300000, ("xknown", "ESH-0102")),
            (500000, ("xknown", "ESH-0102")),
        ],
    },
    "non_tail": {
        "doc": "non-tail recursion (accumulator on the C stack) -- ESH-0112",
        "defs": "(define (down n) (if (= n 0) 0 (+ n (down (- n 1)))))",
        "call": "(down {N})",
        "oracle": "tri",
        "stdlib": False,
        "ladder": [
            (10000, "pass"),
            (100000, "pass"),
            (200000, "pass"),
            (250000, ("xknown", "ESH-0112")),
            (300000, ("xknown", "ESH-0112")),
        ],
    },
    "cps": {
        "doc": "CPS / explicit continuation chain -- clean 100000 guard (ESH-0080 fixed #93)",
        "defs": "(define (sum-cps n k) (if (= n 0) (k 0) (sum-cps (- n 1) (lambda (r) (k (+ r n))))))",
        "call": "(sum-cps {N} (lambda (x) x))",
        "oracle": "tri",
        "stdlib": False,
        "ladder": [
            (1000, "pass"),
            (10000, "pass"),
            (50000, "pass"),
            (100000, "limit"),
            (200000, "limit"),
        ],
    },
    "through_map": {
        "doc": "recursion routed through map (higher-order) -- clean 100000 guard",
        "defs": "(define (rec n) (if (= n 0) 0 (+ n (car (map (lambda (_) (rec (- n 1))) (list 0))))))",
        "call": "(rec {N})",
        "oracle": "tri",
        "stdlib": True,
        "ladder": [
            (1000, "pass"),
            (10000, "pass"),
            (50000, "pass"),
            (100000, "pass"),
            (200000, "limit"),
        ],
    },
    "metacircular": {
        "doc": "recursion through a metacircular evaluator -- silent SIGILL ~30k (ESH-0119)",
        "defs": None,  # emitted specially (multi-line evaluator, embeds {N} in prog)
        "call": None,
        "oracle": "tri",
        "stdlib": True,
        "ladder": [
            (100, "pass"),
            (1000, "pass"),
            (10000, "pass"),
            (30000, ("xknown", "ESH-0119")),
            (50000, ("xknown", "ESH-0119")),
        ],
    },
    "dynamic_wind": {
        # ~100k overflows nondeterministically as SIGBUS (caught -> diagnostic)
        # OR SIGILL (not caught -> silent) -- the silent variant is ESH-0119.
        "doc": "deep dynamic-wind nesting -- overflow ~100k; SIGILL variant silent (ESH-0119)",
        "defs": ("(define (dw n)\n"
                 "  (if (= n 0) 0\n"
                 "      (dynamic-wind (lambda () #f)\n"
                 "                    (lambda () (+ n (dw (- n 1))))\n"
                 "                    (lambda () #f))))"),
        "call": "(dw {N})",
        "oracle": "tri",
        "stdlib": False,
        "ladder": [
            (1000, "pass"),
            (10000, "pass"),
            (50000, "pass"),
            (90000, "pass"),
            (100000, ("xknown", "ESH-0119")),
        ],
    },
    "callcc": {
        "doc": "deep call/cc nesting -- clean 100000 guard",
        "defs": "(define (cc n) (if (= n 0) 0 (call/cc (lambda (k) (+ n (cc (- n 1)))))))",
        "call": "(cc {N})",
        "oracle": "tri",
        "stdlib": False,
        "ladder": [
            (1000, "pass"),
            (10000, "pass"),
            (50000, "pass"),
            (100000, "pass"),
            (200000, "limit"),
        ],
    },
    "guard": {
        "doc": "deep guard nesting -- silent SIGILL ~200k (ESH-0119)",
        "defs": "(define (g n) (if (= n 0) 0 (guard (e (#t -1)) (+ n (g (- n 1))))))",
        "call": "(g {N})",
        "oracle": "tri",
        "stdlib": False,
        "ladder": [
            (1000, "pass"),
            (10000, "pass"),
            (100000, "pass"),
            (150000, "pass"),
            (200000, ("xknown", "ESH-0119")),
            (300000, ("xknown", "ESH-0119")),
        ],
    },
    "stdlib_length": {
        "doc": "stdlib (length) on a long list -- non-tail length SIGILLs (ESH-0108)",
        "defs": "(define (build n acc) (if (= n 0) acc (build (- n 1) (cons n acc))))",
        "call": "(length (build {N} (quote ())))",
        "oracle": "count",
        "stdlib": True,
        "ladder": [
            (10000, "pass"),
            (100000, "pass"),
            (300000, "pass"),
            (500000, ("xknown", "ESH-0108")),
            (1000000, ("xknown", "ESH-0108")),
        ],
    },
}

# order files deterministically
KIND_ORDER = [
    "self_tail", "mutual_tail2", "mutual_tail3", "non_tail",
    "cps", "through_map", "metacircular",
    "dynamic_wind", "callcc", "guard", "stdlib_length",
]

METACIRCULAR_EVAL = """\
(define (tagged? e t) (and (pair? e) (eq? (car e) t)))
(define (lookup v env)
  (if (null? env) (error "unbound" v)
      (let loop ((vs (caar env)) (xs (cdar env)))
        (cond ((null? vs) (lookup v (cdr env)))
              ((eq? (car vs) v) (car xs))
              (else (loop (cdr vs) (cdr xs)))))))
(define (meval exp env)
  (cond ((number? exp) exp)
        ((symbol? exp) (lookup exp env))
        ((tagged? exp 'quote) (cadr exp))
        ((tagged? exp 'if)
         (if (not (eq? (meval (cadr exp) env) #f))
             (meval (caddr exp) env) (meval (cadddr exp) env)))
        ((tagged? exp 'lambda) (list 'closure (cadr exp) (caddr exp) env))
        (else (mapply (meval (car exp) env)
                      (map (lambda (a) (meval a env)) (cdr exp))))))
(define (mapply proc args)
  (cond ((procedure? proc) (apply proc args))
        ((tagged? proc 'closure)
         (meval (caddr proc) (cons (cons (cadr proc) args) (cadddr proc))))
        (else (error "not-applicable" proc))))
(define meta-env (list (cons (list '+ '- '* '= '<) (list + - * = <))))
;; interpreted (self-application) tail-recursive triangular sum to depth {N}
(define meta-prog
  '((lambda (sum) (sum sum {N} 0))
    (lambda (self n acc) (if (= n 0) acc (self self (- n 1) (+ acc n))))))
"""


def want_expr(oracle, n):
    if oracle == "count":
        return str(n)
    # tri: N*(N+1)/2, integer-exact via quotient
    return "(quotient (* {n} (+ {n} 1)) 2)".format(n=n)


def render(kind, n, expect):
    spec = KINDS[kind]
    if isinstance(expect, tuple):
        exp_line = "xknown " + expect[1]
    else:
        exp_line = expect
    lines = []
    lines.append("; recursion-depth probe -- generated by scripts/gen_recursion_depth.py")
    lines.append("; {}".format(spec["doc"]))
    lines.append("; KIND: {}".format(kind))
    lines.append("; DEPTH: {}".format(n))
    lines.append("; EXPECT: {}".format(exp_line))
    if spec["stdlib"]:
        lines.append("(require stdlib)")
    if kind == "metacircular":
        lines.append(METACIRCULAR_EVAL.replace("{N}", str(n)).rstrip())
        got = "(meval meta-prog meta-env)"
    else:
        lines.append(spec["defs"])
        got = spec["call"].replace("{N}", str(n))
    lines.append("(define __want {})".format(want_expr(spec["oracle"], n)))
    lines.append("(define __got {})".format(got))
    lines.append("(if (= __got __want)")
    lines.append('    (begin (display "PASS: {} d={} v=") (display __got) (newline))'.format(kind, n))
    lines.append('    (begin (display "WRONG: {} d={} got=") (display __got)'.format(kind, n))
    lines.append('           (display " want=") (display __want) (newline)))')
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "tests", "recursion_depth", "generated")
    ap.add_argument("--outdir", default=default_out)
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # clear any stale generated probes
    for fn in os.listdir(outdir):
        if fn.startswith("rec_") and fn.endswith(".esk"):
            os.remove(os.path.join(outdir, fn))

    manifest = []
    for kind in KIND_ORDER:
        spec = KINDS[kind]
        for n, expect in spec["ladder"]:
            fname = "rec_{}_d{}.esk".format(kind, n)
            with open(os.path.join(outdir, fname), "w") as f:
                f.write(render(kind, n, expect))
            exp = expect[1] if isinstance(expect, tuple) else expect
            manifest.append("{}\t{}\t{}\t{}".format(fname, kind, n, exp))

    with open(os.path.join(outdir, "MANIFEST.txt"), "w") as f:
        f.write("# file\tkind\tdepth\texpect\n")
        f.write("\n".join(manifest) + "\n")
    print("wrote {} probes to {}".format(len(manifest), outdir))


if __name__ == "__main__":
    main()
