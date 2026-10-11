;; exceptions / raise_continuable  (R7RS-small portable; reference-differential corpus)
(display (with-exception-handler
           (lambda (con) (cond ((string? con) 1) (else 42)))
           (lambda () (+ (raise-continuable 'oops) 23))))
(newline)
(display (with-exception-handler
           (lambda (c) (list 'outer c))
           (lambda ()
             (with-exception-handler
               (lambda (c) (list 'inner c (raise-continuable 'again)))
               (lambda () (raise-continuable 'first))))))
(newline)
(display (guard (e (#t (list 'caught e))) (+ 1 (raise-continuable 'g))))
(newline)
