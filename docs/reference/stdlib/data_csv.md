# `core.data.csv` — CSV parsing and generation

**Source**: [`lib/core/data/csv.esk`](../../../lib/core/data/csv.esk)
**Require**: `(require core.data.csv)` — auto-loaded via `(require stdlib)`.

Pure-Eshkol CSV reader/writer. A parsed CSV is a **list of rows**, each row a
**list of field values** (strings; an empty field is represented as `()`).
Quoting follows the common convention: fields may be `"…"`-wrapped, and an
embedded quote is escaped by doubling (`""`).

> **Row-order bug:** `csv-parse` currently returns rows in **reverse** order.
> This also corrupts [`core.data.dataframe`](data_dataframe.md). See
> [Known issues](#known-issues).

## Functions

### `(csv-split-fields line)` / `(csv-parse-line line)`
Split one line into a list of field values, honouring quotes. `csv-parse-line`
is an alias. An empty line gives `()`; an empty field becomes `()`.

```scheme
(require core.data.csv)
(display (csv-split-fields "a,b,c")) (newline)
(display (csv-split-fields "a,\"b,c\",d")) (newline)      ; quoted comma
(display (csv-split-fields "\"he said \"\"hi\"\"\",x")) (newline)  ; escaped quote
(display (csv-split-fields "a,,c")) (newline)             ; empty field -> ()
(display (csv-split-fields "")) (newline)                 ; empty line -> ()
```
```
(a b c)
(a b,c d)
(he said "hi" x)
(a () c)
()
```

### `(csv-parse str)` — **returns rows reversed** (see [Known issues](#known-issues))
Parse a whole CSV string into a list of rows. Blank lines are skipped.

```scheme
(display (csv-parse "a,b\n1,2\n3,4")) (newline)
```
```
((3 4) (1 2) (a b))
```
The expected result is `((a b) (1 2) (3 4))`; the rows come out in reverse
because of a stale internal `reverse`.

### `(csv-parse-lines lines)`
Parse an already-split list of line strings into rows (blank lines skipped).
Preserves the order of the `lines` argument.

```scheme
(display (csv-parse-lines '("a,b" "1,2"))) (newline)
```
```
((a b) (1 2))
```

### `(csv-parse-file filename)`
Read a file with `read-file` and parse it. Returns `()` if the file cannot be
read. Inherits the reversed-row behaviour of `csv-parse`.

```scheme
;; after (csv-write-file f '(("a" "b") ("1" "2")))
(display (csv-parse-file f)) (newline)
```
```
((1 2) (a b))
```

### `(csv-stringify-row row)`
Convert one row (list of fields) to a CSV line. Fields containing a comma,
quote, or newline are quoted and inner quotes doubled. `()` fields become
empty; numbers are rendered with `number->string`.

```scheme
(display (csv-stringify-row '("a" "b" "c"))) (newline)
(display (csv-stringify-row '("has,comma" "plain"))) (newline)
(display (csv-stringify-row '(1 2 3))) (newline)
```
```
a,b,c
"has,comma",plain
1,2,3
```

### `(csv-stringify rows)`
Convert a list of rows to a multi-line CSV string (rows joined with `\n`). This
is the inverse of `csv-parse` **only if you account for the parse reversal** —
`csv-stringify` itself preserves the order you give it.

```scheme
(display (csv-stringify '(("a" "b") ("1" "2")))) (newline)
```
```
a,b
1,2
```

### `(csv-write-file filename rows)`
Write rows to a file via `csv-stringify` + `write-file`.

```scheme
(csv-write-file "/tmp/out.csv" '(("a" "b") ("1" "2")))
```

## Known issues

### `csv-parse` returns rows in reverse order
`csv-parse` is implemented as
`(csv-parse-lines (reverse (string-split str #\newline)))`. The `reverse` was
needed when `string-split` returned fields right-to-left, but `string-split` now
returns them **left-to-right** (confirmed: `(string-split "a\nb\nc" #\newline)`
⇒ `(a b c)`). The leftover `reverse` therefore flips the row order.

```scheme
(require core.data.csv)
(display (csv-parse "a,b\n1,2\n3,4")) (newline)
;; => ((3 4) (1 2) (a b))   expected ((a b) (1 2) (3 4))
```
Impact: any consumer that treats the first row as a header (notably
`csv-parse-typed` in [`core.data.dataframe`](data_dataframe.md)) gets the wrong
header. Workaround: `(reverse (csv-parse s))`.
