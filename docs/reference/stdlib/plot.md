# `core.plot` — terminal plotting (sparklines, bar charts, histograms)

**Source**: [`lib/core/plot.esk`](../../../lib/core/plot.esk)
**Require**: `(require core.plot)` — **auto-loaded** via `stdlib` (listed in `lib/stdlib.esk`), so `(require stdlib)` is enough; requiring it directly also works.

Pure-Eshkol terminal visualisations using Unicode block characters `▁▂▃▄▅▆▇█`. `sparkline` returns a string; `bar-chart` and `histogram` print directly to standard output.

## Functions

### `(sparkline data)`
Map a list of numbers to a compact string of block characters (8 levels, scaled between the data's min and max). Returns `""` for the empty list; a constant series maps to the mid-level block `▄`.

```scheme
;; plot.esk
(require core.plot)
(display (sparkline (list 1 4 2 8 5 3 7))) (newline)
(display (sparkline (list))) (newline)       ; empty -> ""
(display (sparkline (list 5 5 5))) (newline)  ; constant -> mid-level
```
```
▁▄▂█▅▃▇

▄▄▄
```

### `(bar-chart entries)`
Print a horizontal bar chart. `entries` is a list of `(label value)` 2-lists. Bars are 30 cells wide, scaled to the max value, filled with `█` and padded with `░`; the numeric value is shown at the right. Labels are left-padded to 8 columns. Returns `'()`.

```scheme
;; plot.esk
(require core.plot)
(bar-chart (list (list "Alpha" 10) (list "Beta" 25) (list "Gamma" 15)))
```
```
Alpha    ████████████░░░░░░░░░░░░░░░░░░  10
Beta     ██████████████████████████████  25
Gamma    ██████████████████░░░░░░░░░░░░  15
```

### `(histogram data n-bins)`
Print a frequency histogram of `data` split into `n-bins` equal-width buckets between min and max. Each row shows the `[lo, hi)` range (last bin is inclusive of the max), a run of `█`, and the count. Returns `'()` for empty data or `n-bins <= 0`.

```scheme
;; plot.esk
(require core.plot)
(histogram (list 1 2 2 3 3 3 4 4 5) 5)
```
```
[1, 1.8) █ 1
[1.8, 2.6) ██ 2
[2.6, 3.4) ███ 3
[3.4, 4.2) ██ 2
[4.2, 5) █ 1
```

Edge cases: `sparkline` on `'()` → `""`. `bar-chart`/`histogram` on `'()` → `'()` (nothing printed). `histogram` with `n-bins <= 0` → `'()`. Ranges are formatted to ~6 characters via truncation, so very precise bounds are shortened. Min/max use non-tail recursion (safe up to ~10K elements per the source note).
