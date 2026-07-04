;; vector / vector_map_single  (R7RS-small portable; reference-differential corpus)
(display (vector-map (lambda (x) (* x x)) #(1 2 3 4)))(newline)
(vector-for-each display #(1 2 3))(newline)
