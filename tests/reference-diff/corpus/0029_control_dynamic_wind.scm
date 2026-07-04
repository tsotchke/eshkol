;; control / dynamic_wind  (R7RS-small portable; reference-differential corpus)
(dynamic-wind
  (lambda () (display "in "))
  (lambda () (display "body "))
  (lambda () (display "out ")))
(newline)
(display (call/cc (lambda (k)
  (dynamic-wind
    (lambda () (display "["))
    (lambda () (k 'escaped))
    (lambda () (display "]"))))))
(newline)
