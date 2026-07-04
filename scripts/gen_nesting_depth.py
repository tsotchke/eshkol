#!/usr/bin/env python3
"""Depth-parametric syntax/data NESTING sweep generator (pillar P6c).

For EACH nestable syntactic / data construct, emit a ladder of self-contained
probes at increasing NESTING depth d = 1..N so scripts/run_nesting_depth.sh can
find and gate the maximum depth at which the construct still produces the
CORRECT value under every execution axis.

Meta-lesson (see .swarm/DEPTH_PARAMETRIC_TESTING.md): our earlier harnesses
tested fixed, shallow shapes. A construct correct at nesting depth 1 can be
miscompiled — or silently crash — at depth d>=2. This sweep nests every
composable construct parametrically and verifies each depth against a CLOSED
FORM oracle AND across the -r / AOT-O0 / AOT-O2 axes, so a silent wrong answer
or an axis divergence at any depth is caught, not just a crash.

Constructs (each nested d levels, closed-form result known at generation time):
  quote_nest        deeply nested quoted list literal, walker-summed
  quasiquote_nest   nested quasiquote with unquoted numeric elements
  unquote_splice    nested quasiquote + unquote-splicing chain
  let_nest          nested let    accumulator
  let_star_nest     nested let*   accumulator
  letrec_nest       nested letrec accumulator
  letrec_star_nest  nested letrec* accumulator
  lambda_chain      curried lambda chain, applied
  closure_capture   nested lets each closing over the previous closure
  vector_nest       nested (vector ...) built structure, walker-summed
  list_nest         nested (list ...) built structure, walker-summed
  if_nest           nested if
  cond_nest         nested cond
  case_nest         nested case
  begin_nest        nested begin (set! accumulator)
  guard_nest        nested guard, each handler re-raises accumulating
  dynamic_wind_nest nested dynamic-wind, before-thunks accumulate
  app_chain         deep application chain (f (f (f ... x)))

Oracle (no hand computation):
  tri  -> d*(d+1)/2   (accumulate-by-level constructs)
  dep  -> d           (application chain: +1 per level)

Each probe prints EXACTLY one line: "NVAL <got>". The runner compares that
value across all three axes and against the closed form, classifying the cell:
  PASS             every axis == the closed form
  WRONG            axes agree with each other but the value is wrong (silent
                   miscompile that only the closed form catches)
  AXIS-DIVERGENCE  axes disagree (-r vs AOT-O0 vs AOT-O2), or one axis crashes
                   while another returns a value
  LIMIT            a clean, consistent error/crash-with-diagnostic on every
                   axis (a documented capability boundary)
  SILENT-CRASH     a crash with no diagnostic / no output (a bug)

Each file carries directive comments the runner reads:
  ; CONSTRUCT: <name>
  ; DEPTH: <d>
  ; VALUE: <closed-form integer>

Deterministic: no RNG, stable ordering; rerunning reproduces the corpus
byte-for-byte.

Usage: python3 scripts/gen_nesting_depth.py [--outdir DIR] [--depths d,d,...]
"""

import argparse
import os

# ── construct builders ──────────────────────────────────────────────────
# Each builder(d) returns (body_expr_string, needs_stdlib).  The rendered
# probe wraps it as (display <body>)(newline); the value must be an integer.


def _tri(d):
    return d * (d + 1) // 2


def c_quote_nest(d):
    # '(1 (2 (3 ... (d) ...)))  -- literal numbers inside quote, walker sums.
    def node(k):
        return "(%d)" % k if k == d else "(%d %s)" % (k, node(k + 1))
    walk = ("(define (s x) (cond ((null? x) 0) ((number? x) x) "
            "((pair? x) (+ (s (car x)) (s (cdr x)))) (else 0)))")
    return "%s\n(s (quote %s))" % (walk, node(1)), False


def c_quasiquote_nest(d):
    # `(,1 (,2 (,3 ... (,d) ...)))  -- unquoted numbers at every nesting level.
    def node(k):
        return "(,%d)" % k if k == d else "(,%d %s)" % (k, node(k + 1))
    walk = ("(define (s x) (cond ((null? x) 0) ((number? x) x) "
            "((pair? x) (+ (s (car x)) (s (cdr x)))) (else 0)))")
    return "%s\n(s `%s)" % (walk, node(1)), False


def c_unquote_splice(d):
    # `(,1 ,@`(,2 ,@`(,3 ... ,@'()))) -- nested quasiquote + unquote-splicing.
    walk = "(define (s x) (if (pair? x) (+ (car x) (s (cdr x))) 0))"
    e = "'()"
    for k in range(d, 0, -1):
        e = "`(,%d ,@%s)" % (k, e)
    return "%s\n(s %s)" % (walk, e), False


def c_let_nest(d):
    e = "a"
    for k in range(d, 0, -1):
        e = "(let ((a (+ a %d))) %s)" % (k, e)
    return "(let ((a 0)) %s)" % e, False


def c_let_star_nest(d):
    e = "a"
    for k in range(d, 0, -1):
        e = "(let* ((a (+ a %d))) %s)" % (k, e)
    return "(let* ((a 0)) %s)" % e, False


def c_letrec_nest(d):
    # letrec with a constant binding is legal; nest the accumulator.
    e = "acc"
    for k in range(d, 0, -1):
        e = "(letrec ((v %d)) (let ((acc (+ acc v))) %s))" % (k, e)
    return "(let ((acc 0)) %s)" % e, False


def c_letrec_star_nest(d):
    e = "acc"
    for k in range(d, 0, -1):
        e = "(letrec* ((v %d)) (let ((acc (+ acc v))) %s))" % (k, e)
    return "(let ((acc 0)) %s)" % e, False


def c_lambda_chain(d):
    hdr = "".join("(lambda (x%d) " % i for i in range(1, d + 1))
    body = "(+ " + " ".join("x%d" % i for i in range(1, d + 1)) + ")"
    calls = "".join(" %d)" % i for i in range(1, d + 1))
    return "(" * d + hdr + body + ")" * d + calls, False


def c_closure_capture(d):
    # nested lets, each binding a thunk that closes over the previous thunk.
    e = "(f%d)" % d
    for k in range(d, 0, -1):
        prev = "(f%d)" % (k - 1) if k > 1 else "0"
        e = "(let ((f%d (lambda () (+ %d %s)))) %s)" % (k, k, prev, e)
    return e, False


def c_vector_nest(d):
    def node(k):
        return "(vector %d)" % k if k == d else "(vector %d %s)" % (k, node(k + 1))
    walk = ("(define (s v) (if (= (vector-length v) 2) "
            "(+ (vector-ref v 0) (s (vector-ref v 1))) (vector-ref v 0)))")
    return "%s\n(s %s)" % (walk, node(1)), False


def c_list_nest(d):
    def node(k):
        return "(list %d)" % k if k == d else "(list %d %s)" % (k, node(k + 1))
    walk = ("(define (s x) (cond ((null? x) 0) ((number? x) x) "
            "((pair? x) (+ (s (car x)) (s (cdr x)))) (else 0)))")
    return "%s\n(s %s)" % (walk, node(1)), False


def c_if_nest(d):
    def node(k):
        return "(if #t %d 0)" % k if k == d else "(if #t (+ %d %s) 0)" % (k, node(k + 1))
    return node(1), False


def c_cond_nest(d):
    def node(k):
        inner = "%d" % k if k == d else "(+ %d %s)" % (k, node(k + 1))
        return "(cond (#t %s) (else 0))" % inner
    return node(1), False


def c_case_nest(d):
    def node(k):
        inner = "%d" % k if k == d else "(+ %d %s)" % (k, node(k + 1))
        return "(case 1 ((1) %s) (else 0))" % inner
    return node(1), False


def c_begin_nest(d):
    e = "acc"
    for k in range(d, 0, -1):
        e = "(begin (set! acc (+ acc %d)) %s)" % (k, e)
    return "(let ((acc 0)) %s)" % e, False


def c_guard_nest(d):
    # innermost raises 0; each level re-raises accumulating; outermost returns.
    inner = "(raise 0)"
    for k in range(d, 1, -1):
        inner = "(guard (e (#t (raise (+ %d e)))) %s)" % (k, inner)
    return "(guard (e (#t (+ 1 e))) %s)" % inner, False


def c_dynamic_wind_nest(d):
    # each before-thunk adds its level to acc; innermost body returns acc, so
    # the value observed is the sum of all before-thunks = d*(d+1)/2.
    e = "acc"
    for k in range(d, 0, -1):
        e = ("(dynamic-wind (lambda () (set! acc (+ acc %d))) "
             "(lambda () %s) (lambda () #f))" % (k, e))
    return "(let ((acc 0)) %s)" % e, False


def c_app_chain(d):
    e = "0"
    for _ in range(d):
        e = "(f %s)" % e
    return "(define (f x) (+ x 1))\n%s" % e, False


CONSTRUCTS = {
    "quote_nest":        (c_quote_nest,        "tri"),
    "quasiquote_nest":   (c_quasiquote_nest,   "tri"),
    "unquote_splice":    (c_unquote_splice,    "tri"),
    "let_nest":          (c_let_nest,          "tri"),
    "let_star_nest":     (c_let_star_nest,     "tri"),
    "letrec_nest":       (c_letrec_nest,       "tri"),
    "letrec_star_nest":  (c_letrec_star_nest,  "tri"),
    "lambda_chain":      (c_lambda_chain,      "tri"),
    "closure_capture":   (c_closure_capture,   "tri"),
    "vector_nest":       (c_vector_nest,       "tri"),
    "list_nest":         (c_list_nest,         "tri"),
    "if_nest":           (c_if_nest,           "tri"),
    "cond_nest":         (c_cond_nest,         "tri"),
    "case_nest":         (c_case_nest,         "tri"),
    "begin_nest":        (c_begin_nest,        "tri"),
    "guard_nest":        (c_guard_nest,        "tri"),
    "dynamic_wind_nest": (c_dynamic_wind_nest, "tri"),
    "app_chain":         (c_app_chain,         "dep"),
}

# deterministic file ordering
CONSTRUCT_ORDER = [
    "quote_nest", "quasiquote_nest", "unquote_splice",
    "let_nest", "let_star_nest", "letrec_nest", "letrec_star_nest",
    "lambda_chain", "closure_capture",
    "vector_nest", "list_nest",
    "if_nest", "cond_nest", "case_nest", "begin_nest",
    "guard_nest", "dynamic_wind_nest",
    "app_chain",
]

DEFAULT_DEPTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def closed_form(oracle, d):
    return d if oracle == "dep" else _tri(d)


def render(name, d):
    builder, oracle = CONSTRUCTS[name]
    body, _stdlib = builder(d)
    value = closed_form(oracle, d)
    lines = [
        "; nesting-depth probe -- generated by scripts/gen_nesting_depth.py",
        "; CONSTRUCT: {}".format(name),
        "; DEPTH: {}".format(d),
        "; VALUE: {}".format(value),
    ]
    # the body may contain leading (define ...) helper lines then a final expr.
    parts = body.rsplit("\n", 1)
    if len(parts) == 2:
        lines.append(parts[0])
        expr = parts[1]
    else:
        expr = parts[0]
    lines.append("(display {})".format(expr))
    lines.append("(newline)")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "tests", "nesting_depth", "generated")
    ap.add_argument("--outdir", default=default_out)
    ap.add_argument("--depths", default=None,
                    help="comma-separated depth ladder (default: %s)"
                         % ",".join(map(str, DEFAULT_DEPTHS)))
    args = ap.parse_args()
    depths = DEFAULT_DEPTHS
    if args.depths:
        depths = [int(x) for x in args.depths.split(",") if x.strip()]
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    for fn in os.listdir(outdir):
        if fn.startswith("nest_") and fn.endswith(".esk"):
            os.remove(os.path.join(outdir, fn))

    manifest = []
    for name in CONSTRUCT_ORDER:
        _b, oracle = CONSTRUCTS[name]
        for d in depths:
            fname = "nest_{}_d{}.esk".format(name, d)
            with open(os.path.join(outdir, fname), "w") as f:
                f.write(render(name, d))
            manifest.append("{}\t{}\t{}\t{}".format(
                fname, name, d, closed_form(oracle, d)))

    with open(os.path.join(outdir, "MANIFEST.txt"), "w") as f:
        f.write("# file\tconstruct\tdepth\tvalue\n")
        f.write("\n".join(manifest) + "\n")
    print("wrote {} probes ({} constructs x {} depths) to {}".format(
        len(manifest), len(CONSTRUCT_ORDER), len(depths), outdir))


if __name__ == "__main__":
    main()
