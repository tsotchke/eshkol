;; control / arrow_clauses  (R7RS-small portable; reference-differential corpus)
(display (cond ((assv 2 '((1 . "a") (2 . "b"))) => cdr) (else "none")))(newline)
(display (cond ((memv 3 '(1 2 3 4)) => car) (else 'no)))(newline)
(display (case 5 ((1 2 3) 'lo) ((4 5 6) => (lambda (x) (* x 10))) (else 'no)))(newline)
