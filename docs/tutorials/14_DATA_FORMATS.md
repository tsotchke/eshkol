# Tutorial 14: JSON, CSV, and Base64

Eshkol includes built-in support for common data interchange formats.

---

## JSON

```scheme
(require core.json)

;; Parse JSON string to Eshkol data
(define data (json-parse "{\"name\":\"Alice\",\"age\":30}"))
(display data)
(newline)

;; Convert Eshkol data to JSON string
(define json-str (json-stringify '((name . "Bob") (age . 25))))
(display json-str)
(newline)

;; File I/O
(define port (open-input-file "data.json"))
(define parsed (json-read port))
(close-port port)

(define out (open-output-file "output.json"))
(json-write parsed out)
(close-port out)
```

---

## CSV

```scheme
(require core.data.csv)

;; Parse CSV string
(define rows (csv-parse "name,age\nAlice,30\nBob,25"))
(display rows)
(newline)
;; => list of rows, each row is a list of fields

;; Write CSV
(define csv-str (csv-stringify '(("name" "age") ("Alice" "30") ("Bob" "25"))))
(display csv-str)
(newline)

;; File I/O
(csv-write-file "output.csv" '(("x" "y") ("1" "2") ("3" "4")))
```

---

## Base64

```scheme
(require core.data.base64)

;; Encode string to Base64
(define encoded (base64-encode "Hello, world!"))
(display encoded)
(newline)
;; => "SGVsbG8sIHdvcmxkIQ=="

;; Decode Base64 to string
(define decoded (base64-decode encoded))
(display decoded)
(newline)
;; => "Hello, world!"
```
