# Tutorial 13: Strings and I/O

---

## String Operations

```scheme
(require core.strings)

(string-length "hello")               ;; => 5
(string-ref "hello" 1)                ;; => #\e
(substring "hello world" 0 5)         ;; => "hello"
(string-append "hello" " " "world")   ;; => "hello world"
(string-upcase "hello")               ;; => "HELLO"
(string-downcase "HELLO")             ;; => "hello"
(string-contains "hello world" "world")  ;; => 6 (index)
(string-reverse "abcde")              ;; => "edcba"
(string-repeat "ab" 3)                ;; => "ababab"
(string-starts-with "hello" "hel")    ;; => #t
(string-ends-with "hello" "llo")      ;; => #t
(string-count "banana" #\a)           ;; => 3
```

---

## String Predicates

```scheme
(string? "hello")       ;; => #t
(string=? "abc" "abc")  ;; => #t
(string<? "abc" "abd")  ;; => #t
(string-ci=? "Hello" "hello")  ;; => #t (case-insensitive)
```

---

## Character Operations

```scheme
(char? #\a)              ;; => #t
(char-alphabetic? #\a)   ;; => #t
(char-numeric? #\5)      ;; => #t
(char-whitespace? #\space) ;; => #t
(char-uppercase? #\A)    ;; => #t
(char-lowercase? #\a)    ;; => #t
(char-upcase #\a)        ;; => #\A
(char-downcase #\A)      ;; => #\a
```

---

## Display and Output

```scheme
(display "Hello, world!")  ;; prints: Hello, world!
(newline)                  ;; prints a newline
(display 42)               ;; prints: 42
(display #t)               ;; prints: #t
(display '(1 2 3))         ;; prints: (1 2 3)

;; write outputs in read-back format (strings quoted, etc.)
(write "hello")            ;; prints: "hello" (with quotes)
(write '(1 "two" 3))       ;; prints: (1 "two" 3)
```

---

## File I/O

```scheme
;; Write to a file
(define out (open-output-file "data.txt"))
(write-string "line 1\n" out)
(write-string "line 2\n" out)
(close-port out)

;; Read from a file
(define in (open-input-file "data.txt"))
(display (read-line in))   ;; => line 1
(display (read-line in))   ;; => line 2
(close-port in)

;; Read entire file character by character
(define in (open-input-file "data.txt"))
(let loop ()
  (let ((ch (read-char in)))
    (if (eof-object? ch)
        (close-port in)
        (begin (display ch) (loop)))))
```

---

## String Ports

String ports let you build strings incrementally or parse from strings:

```scheme
;; Output string port — build a string
(define sp (open-output-string))
(write-string "hello " sp)
(write-string "world" sp)
(display (get-output-string sp))  ;; => "hello world"

;; Input string port — parse from a string
(define ip (open-input-string "(+ 1 2)"))
(define expr (read ip))
(display expr)  ;; => (+ 1 2)
```

---

## Number-String Conversion

```scheme
(number->string 42)          ;; => "42"
(number->string 3.14)        ;; => "3.14"
(string->number "42")        ;; => 42
(string->number "3.14")      ;; => 3.14
(string->number "not-a-num") ;; => #f
```
