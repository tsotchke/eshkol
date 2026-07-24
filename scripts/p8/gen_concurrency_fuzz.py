#!/usr/bin/env python3
r"""gen_concurrency_fuzz.py — P8 escape-closure axis 5: generative concurrency
fuzz for parallel-map / parallel-execute.

Originating escape (see .swarm/P8_ESCAPE_ANALYSIS.md): parallel-map runs its
mapped closure on work-stealing pool workers, all pinned to the single shared
thread-safe process arena. A bump-arena's SCOPE STACK is intrinsically
single-threaded — a pop rewinds the shared bump pointer. A closure whose body
drives scope-based reclamation (an internal named-let's per-iteration scope, or
a builtin such as memv that brackets scratch allocation in a push/pop) made
workers concurrently push/pop that one stack, so worker A's pop freed memory
worker B was still using: "car/cdr: not a pair", SIGSEGV/SIGBUS, or a hang —
NONDETERMINISTICALLY, and only once the input crossed the parallel threshold.
Serial map over the same closure was always correct. A fixed single-shape
regression fixture existed; the ESCAPE is that the trigger space (closure body
shape × threshold-straddling n × repetition) was not swept.

This generator grows seeded worker closures out of the trigger primitives
{memv, internal named-let, allocating collections, string ops, mixed #f
returns} and, for each, runs parallel-map REPEAT times at every n that straddles
the pool threshold {4,15,16,17,64}, asserting EVERY parallel result deep-equals
the SERIAL map reference (an in-program oracle — no second engine needed). A
deep checksum after completion trips on dangling interior structure. The race is
~50% per run pre-fix, so REPEAT (default 20) drives detection to ~1.

Deterministic generator (the RUNTIME race is not, but the correctness verdict
is: correct == always equal to serial). Output: self-checking programs in the
shared scripts/p8/harness.py format.

Usage: python3 scripts/p8/gen_concurrency_fuzz.py --out DIR [--seed N]
                 [--repeats R] [--ns 4,15,16,17,64] [--shapes K] [--list]
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import Program            # noqa: E402

PRELUDE = (
    "(define (iota n)\n"
    "  (let loop ((i (- n 1)) (acc '())) (if (< i 0) acc (loop (- i 1) (cons i acc)))))\n"
    ";; internal named-let loop + memv over freshly-allocated scratch lists:\n"
    ";; each call drives per-iteration arena scope push/pop on the shared stack.\n"
    "(define (overlap a b)\n"
    "  (let loop ((xs a) (c 0))\n"
    "    (if (pair? xs) (loop (cdr xs) (if (memv (car xs) b) (+ c 1) c)) c)))\n"
    "(define (build-list-mod base len m)\n"
    "  (let loop ((k 0) (acc '()))\n"
    "    (if (< k len) (loop (+ k 1) (cons (modulo (+ base k) m) acc)) acc)))\n"
    ";; deep checksum: full traversal trips (crash) on dangling interior structure.\n"
    "(define (deepsum x)\n"
    "  (cond ((pair? x) (+ (deepsum (car x)) (deepsum (cdr x))))\n"
    "        ((vector? x) (let loop ((i 0) (s 0))\n"
    "                       (if (< i (vector-length x)) (loop (+ i 1) (+ s (deepsum (vector-ref x i)))) s)))\n"
    "        ((string? x) (string-length x))\n"
    "        ((number? x) (if (exact? x) x 0))\n"
    "        ((char? x) 1)\n"
    "        ((null? x) 0)\n"
    "        ((eq? x #f) 0)\n"
    "        (else 1)))\n"
    ";; run parallel-map R times; #t iff every run deep-equals the serial ref AND\n"
    ";; the deep checksum matches (catches dangling structure equal? might skip).\n"
    "(define (pm-stable f input R)\n"
    "  (let* ((ref (map f input)) (refsum (deepsum ref)))\n"
    "    (let loop ((k 0))\n"
    "      (if (>= k R) #t\n"
    "          (let ((got (parallel-map f input)))\n"
    "            (if (and (equal? got ref) (= (deepsum got) refsum)) (loop (+ k 1)) #f))))))\n"
)

# Trigger fragments. Each returns an Eshkol expression (string) computing a
# per-element sub-result from the loop variable `p`; the generator wires a
# random subset into the worker body. All are PURE functions of p so the serial
# map is a sound reference.
TRIGGERS = {
    "memv": "(overlap (build-list-mod p 12 7) (list 0 1 2 3 4 5))",
    "namedlet": "(let loop ((k 0) (s 0)) (if (< k 20) (loop (+ k 1) (+ s (modulo (+ p k) 5))) s))",
    "alloc": "(vector-length (list->vector (build-list-mod p 16 11)))",
    "string": "(string-length (string-append \"w\" (number->string p) \"-\" (number->string (* p p))))",
    "nestlist": "(length (list (build-list-mod p 4 3) (build-list-mod (+ p 1) 4 3) (number->string p)))",
}


def make_worker(rng, shape_id):
    """Build a worker closure using a random non-empty subset of triggers,
    returning either a nested structure or #f (mixed returns)."""
    keys = list(TRIGGERS)
    rng.shuffle(keys)
    chosen = keys[:rng.randint(2, len(keys))]
    parts = [TRIGGERS[k] for k in chosen]
    # Compose the parts into a sum used both to branch on #f and as payload.
    sum_expr = "(+ 0 %s)" % " ".join(parts)
    # A nested payload built from allocating operations so the result carries
    # interior heap structure a dangling pop would corrupt.
    payload = ("(list p (build-list-mod p 6 5) "
               "(vector (number->string p) (build-list-mod (+ p 3) 5 4)) __s)")
    body = ("(lambda (p)\n"
            "  (let ((__s %s))\n"
            "    (if (= (modulo __s 3) 0) #f %s)))" % (sum_expr, payload))
    return body, chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=8805)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--ns", default="4,15,16,17,64")
    ap.add_argument("--shapes", type=int, default=6,
                    help="number of distinct closure shapes to generate")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    rng = random.Random(args.seed)
    files = {}
    for s in range(args.shapes):
        body, chosen = make_worker(rng, s)
        p = Program("concurrency fuzz: parallel-map worker shape %d {%s}"
                    % (s, ",".join(chosen)))
        p.tag("P8-AXIS concurrency-fuzz")
        p.tag("P8-TRIGGERS %s" % ",".join(chosen))
        p.define(PRELUDE)
        p.define("(define __worker\n%s)" % body)
        for n in ns:
            p.define("(define __in_%d (iota %d))" % (n, n))
            p.check("pm-shape%d-n%d-r%d" % (s, n, args.repeats),
                    "(pm-stable __worker __in_%d %d)" % (n, args.repeats))
        files["conc_shape_%02d" % s] = p.render()

    if args.list:
        for k in sorted(files):
            print(k)
        return 0
    if not args.out:
        sys.exit("--out DIR required (or --list)")
    os.makedirs(args.out, exist_ok=True)
    for name, text in sorted(files.items()):
        with open(os.path.join(args.out, name + ".esk"), "w") as fh:
            fh.write(text)
    print("wrote %d files to %s" % (len(files), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
