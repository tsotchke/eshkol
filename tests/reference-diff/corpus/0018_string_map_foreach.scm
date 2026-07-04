;; string / map_foreach  (R7RS-small portable; reference-differential corpus)
(display (string-map char-upcase "hello"))(newline)
(string-for-each (lambda (c) (display c) (display ".")) "abc")(newline)
