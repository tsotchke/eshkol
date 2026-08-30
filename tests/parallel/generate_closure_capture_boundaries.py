#!/usr/bin/env python3
"""Generate native parallel closure-environment boundary programs."""

from __future__ import annotations

import argparse
from pathlib import Path


COUNTS = (31, 32, 33, 200, 4096)
INPUTS = tuple(range(1, 9))


def make_program(count: int) -> str:
    names = [f"capture_{index:04d}" for index in range(count)]
    bindings = "\n".join(
        f"(define {name} {index})" for index, name in enumerate(names)
    )
    sum_expression = " ".join(names)
    capture_sequence = f"(begin {sum_expression})"
    result_expression = f"(+ x {capture_sequence})"
    thunk_definition = "" if count == 4096 else f"""
(define (make-thunk)
  (lambda () {capture_sequence}))
"""
    extra_checks = "" if count == 4096 else f"""
(define thunk (make-thunk))
(display (if (equal? (parallel-execute thunk thunk)
                    (list {count - 1} {count - 1}))
             "PASS: parallel-execute"
             "FAIL: parallel-execute"))
(newline)
(parallel-for-each f inputs)
(display "PASS: parallel-for-each")
(newline)
"""
    inputs = " ".join(str(value) for value in INPUTS)
    return f""";; Parallel closure environment regression: {count} captures.
;; The serial and parallel map lines must be identical. The execute and
;; for-each calls exercise the nullary and unary worker entry points too.
(define (map f xs)
  (if (null? xs) '()
      (cons (f (car xs)) (map f (cdr xs)))))
{bindings}
(define (make-capture)
  (lambda (x) {result_expression}))
{thunk_definition}
(define inputs '({inputs}))
(define f (make-capture))
(define serial (map f inputs))
(define parallel (parallel-map f inputs))
(display "RESULT {count} SERIAL=")
(display serial)
(display " PARALLEL=")
(display parallel)
(newline)
(display (if (equal? serial parallel)
             "PASS: parallel-map"
             "FAIL: parallel-map"))
(newline)
{extra_checks}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for count in COUNTS:
        (args.output_dir / f"closure_capture_{count}.esk").write_text(
            make_program(count), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
