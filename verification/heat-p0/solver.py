"""P0 lowering for the unit rod, kappa=1, r=1/4, fixed zero endpoints."""


def lower(intervals):
    rows = [
        f"    (vector-set! fresh {i} (/ (+ (vector-ref old {i-1}) "
        f"(* 2 (vector-ref old {i})) (vector-ref old {i+1})) 4))"
        for i in range(1, intervals)
    ]
    return "\n".join([
        "(define (heat-step old)",
        f"  (let ((fresh (make-vector {intervals+1} 0)))",
        *rows,
        "    fresh))",
    ])


def program(kernel, samples, steps):
    """The CPU entry point emits one exact scalar for every grid point."""
    values = " ".join(map(str, samples))
    return f"""{kernel}
(define (advance grid steps)
  (if (= steps 0) grid (advance (heat-step grid) (- steps 1))))
(define field (advance (vector {values}) {steps}))
(let print-field ((i 0))
  (if (< i {len(samples)})
      (begin (display (vector-ref field i)) (newline) (print-field (+ i 1)))
      0))
"""
