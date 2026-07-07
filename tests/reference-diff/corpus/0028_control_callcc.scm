;; control / callcc  (R7RS-small portable; reference-differential corpus)
(display (call-with-current-continuation (lambda (k) (+ 1 (k 42)))))(newline)
(display (+ 1 (call-with-current-continuation (lambda (k) 10))))(newline)
(display (call/cc (lambda (k) (* 2 (+ 3 (k 99))))))(newline)
