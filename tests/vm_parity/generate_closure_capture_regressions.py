#!/usr/bin/env python3
"""Generate deterministic high-capture map/for-each parity programs."""

from __future__ import annotations

import argparse
from pathlib import Path


COUNTS = (255, 256, 257, 4096, 65536, 65537)


def make_program(count: int) -> str:
    names = [f"c{i:05d}" for i in range(count)]
    bindings = "\n".join(f"(define {name} {i})" for i, name in enumerate(names))
    # Refer to every binding while keeping the body linear in the number of
    # captures. The last value is the high-index capture and proves that the
    # complete environment survived construction and invocation.
    body = " ".join(names)
    expected = count - 1
    return f""";; Generated closure-capture boundary regression: {count} captures.
;; Keep the fixture self-contained so the native high-count probe does not
;; spend its measurement budget compiling the unrelated standard library.
(define (map f xs) (cons (f (car xs)) '()))
(define (for-each f xs) (f (car xs)))
{bindings}
(define (make-capture-{count})
  (lambda (x) (begin {body})))
(display (map (make-capture-{count}) '(1)))
(newline)
(for-each (lambda (f) (display (f 1))) (list (make-capture-{count})))
(newline)
;; Expected output is {expected} twice; the second call exercises for-each.
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
