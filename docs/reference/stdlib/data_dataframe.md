# `core.data.dataframe` — columnar operations over CSV data

**Source**: [`lib/core/data/dataframe.esk`](../../../lib/core/data/dataframe.esk)
**Require**: `(require core.data.dataframe)` — auto-loaded via `(require stdlib)`.

DataFrame layer on top of [`core.data.csv`](data_csv.md). A **DataFrame** is a
pair `(header . rows)`:

- `header` — a list of column-name strings
- `rows` — a list of rows, each a list of typed values

`csv-parse-typed` adds type inference: numeric-looking fields become numbers,
everything else stays a string.

## Functions

### `(csv-parse-typed str)`
Parse a CSV string into a `(header . typed-rows)` DataFrame; the first row is the
header, remaining rows get type inference.

```scheme
(require core.data.dataframe)
(display (csv-parse-typed "col\n42")) (newline)
```
```
((col) (42))
```
(Previously the header/row came out swapped because the underlying `csv-parse`
returned rows reversed; that has been fixed.)

The remaining accessors are illustrated on a directly-built frame:

```scheme
(require core.data.dataframe)
(define df (cons '("name" "age")
                 '(("Alice" 30) ("Bob" 25) ("Cy" 40))))
```

### `(csv-header df)` / `(csv-rows df)`
`car` and `cdr` of the frame: the column-name list and the row list.

```scheme
(display (csv-header df)) (newline)
(display (csv-rows df)) (newline)
```
```
(name age)
((Alice 30) (Bob 25) (Cy 40))
```

### `(csv-row-count df)` / `(csv-col-count df)`
Number of data rows / number of columns.

```scheme
(display (csv-row-count df)) (newline)   ; 3
(display (csv-col-count df)) (newline)   ; 2
```
```
3
2
```

### `(csv-column-idx df name)`
0-based index of the named column, or `-1` if not present.

```scheme
(display (csv-column-idx df "age")) (newline)   ; 1
(display (csv-column-idx df "zzz")) (newline)   ; -1
```
```
1
-1
```

### `(csv-column df name)`
Extract a column by name as a list of values. **Errors** (`csv-column: unknown
column`) if the name is absent.

```scheme
(display (csv-column df "age")) (newline)   ; (30 25 40)
```
```
(30 25 40)
```

### `(csv-select df names)`
Project the named columns into a **new DataFrame** whose header is `names`.
Errors on any unknown column name.

```scheme
(display (csv-select df '("age"))) (newline)
```
```
((age) (30) (25) (40))
```

### `(csv-filter df pred)`
Keep rows for which `(pred row)` is true; returns a new DataFrame (same header).
`pred` receives the row as a list of values.

```scheme
(display (csv-filter df (lambda (row) (> (cadr row) 25)))) (newline)
```
```
((name age) (Alice 30) (Cy 40))
```

### `(csv-map-column df name f)`
Apply `f` to every value in the named column, returning a new DataFrame. If the
column is unknown the frame is returned unchanged (no error).

```scheme
(display (csv-map-column df "age" (lambda (a) (+ a 1)))) (newline)
```
```
((name age) (Alice 31) (Bob 26) (Cy 41))
```

### `(csv-describe df)`
Print a one-line shape summary and the column names to stdout. Returns an
unspecified value; used for its side effect.

```scheme
(csv-describe df)
```
```
DataFrame: 3 rows x 2 columns
Columns: name, age
```

## Known issues

None. (Historically `csv-parse-typed` picked the **last** CSV line as the header
because the underlying `csv-parse` returned rows reversed. That reversal has been
fixed in [`core.data.csv`](data_csv.md), so the header is now the first line:

```scheme
(require core.data.dataframe)
(define df (csv-parse-typed "name,age\nAlice,30\nBob,25"))
(display (csv-header df)) (newline)          ; (name age)
(display (csv-rows df)) (newline)            ; ((Alice 30) (Bob 25))
```
)
