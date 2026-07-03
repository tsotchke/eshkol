# `core.data.dataframe` — columnar operations over CSV data

**Source**: [`lib/core/data/dataframe.esk`](../../../lib/core/data/dataframe.esk)
**Require**: `(require core.data.dataframe)` — auto-loaded via `(require stdlib)`.

DataFrame layer on top of [`core.data.csv`](data_csv.md). A **DataFrame** is a
pair `(header . rows)`:

- `header` — a list of column-name strings
- `rows` — a list of rows, each a list of typed values

`csv-parse-typed` adds type inference: numeric-looking fields become numbers,
everything else stays a string.

> **Broken through `csv-parse-typed`.** Because [`csv-parse`](data_csv.md)
> returns rows reversed, `csv-parse-typed` picks the **last** CSV line as the
> header, which breaks the accessors below. The accessors themselves are correct
> when given a well-formed `(header . rows)` pair. See
> [Known issues](#known-issues). The examples below build the frame by hand to
> document intended behaviour.

## Functions

### `(csv-parse-typed str)` — **header comes out wrong** (see [Known issues](#known-issues))
Parse a CSV string into a `(header . typed-rows)` DataFrame; first row is meant
to be the header, remaining rows get type inference.

```scheme
(require core.data.dataframe)
(display (csv-parse-typed "col\n42")) (newline)
```
```
((42) (col))
```
Expected `((col) (42))`; the header/row are swapped by the underlying
`csv-parse` reversal.

For the remaining accessors, a correctly-shaped frame is built directly:

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

### `csv-parse-typed` picks the last CSV line as the header
`csv-parse-typed` calls `csv-parse`, which returns rows reversed (see
[`core.data.csv`](data_csv.md)), then takes `(car rows)` as the header. The
header therefore becomes the last data line and subsequent lookups fail:

```scheme
(require core.data.dataframe)
(define df (csv-parse-typed "name,age\nAlice,30\nBob,25"))
(display (csv-header df)) (newline)          ; (Bob 25)   -- should be (name age)
(display (csv-rows df)) (newline)            ; ((Alice 30) (name age))
(csv-column df "age")                          ; ERROR: csv-column: unknown column
```
Root cause is entirely in `csv-parse`; the DataFrame accessors are correct on a
well-formed frame. Workaround until `csv-parse` is fixed: reconstruct the frame
from a reversed parse, e.g. build `(cons header data)` yourself from
`(reverse (csv-parse str))`.
