;; numeric / division_signs  (R7RS-small portable; reference-differential corpus)
;;
;; R7RS 6.2.6: `modulo` / `floor-remainder` take the sign of the DIVISOR,
;; `remainder` / `truncate-remainder` take the sign of the DIVIDEND, and
;; `quotient` truncates toward zero. A lowering that emits a bare machine
;; remainder (`srem`, C semantics) passes every positive-divisor case and
;; fails only the rows below — which is why the negative-divisor rows are
;; here rather than one representative case per operator.
(display (modulo 7 3))(newline)
(display (modulo -7 3))(newline)
(display (modulo 7 -3))(newline)
(display (modulo -7 -3))(newline)
(display (modulo 1 -3))(newline)
(display (modulo -1 3))(newline)
(display (modulo -6 3))(newline)
(display (remainder 7 3))(newline)
(display (remainder -7 3))(newline)
(display (remainder 7 -3))(newline)
(display (remainder -7 -3))(newline)
(display (quotient 7 3))(newline)
(display (quotient -7 3))(newline)
(display (quotient 7 -3))(newline)
(display (quotient -7 -3))(newline)
(display (floor-quotient 7 -3))(newline)
(display (floor-remainder 7 -3))(newline)
(display (floor-quotient -7 3))(newline)
(display (floor-remainder -7 3))(newline)
(display (truncate-quotient 7 -3))(newline)
(display (truncate-remainder 7 -3))(newline)
(display (= -7 (+ (* 3 (quotient -7 3)) (remainder -7 3))))(newline)
(display (= 7 (+ (* -3 (quotient 7 -3)) (remainder 7 -3))))(newline)
(display (modulo (- (expt 10 20)) 3))(newline)
(display (modulo (expt 10 20) -3))(newline)
(display (remainder (- (expt 10 20)) 3))(newline)
