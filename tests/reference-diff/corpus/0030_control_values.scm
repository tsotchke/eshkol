;; control / values  (R7RS-small portable; reference-differential corpus)
(call-with-values (lambda () (values 1 2 3)) (lambda (a b c) (display (+ a b c)) (newline)))
(display (call-with-values (lambda () (values 10 20)) +))(newline)
(display (let-values (((a b) (values 3 4)) ((c) (values 5))) (+ a b c)))(newline)
