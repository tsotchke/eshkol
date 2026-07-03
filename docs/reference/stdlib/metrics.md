# `core.metrics` — Prometheus-compatible counters, gauges, histograms

**Source**: [`lib/core/metrics.esk`](../../../lib/core/metrics.esk)
**Require**: `(require core.metrics)` — **NOT** auto-loaded via `stdlib`; require it explicitly.

Prometheus-style metric primitives with a global registry that renders to the Prometheus text exposition format. A metric is a vector `#(name help label-keys kind state)`; `state` is a hash-table keyed by the joined label-value string. Every metric registers itself on creation, so `(metrics-render)` walks all of them. Designed to feed a `/metrics` endpoint (see `core.http_server`).

> REPL note: requiring this module prints a few harmless `error: Unknown function: length` diagnostics to **stderr** during compile. They are a REPL forward-reference artifact only — `length` resolves at runtime, all functions execute correctly, and no exception is raised. Ignore them (they do not appear in AOT builds' program output).

Labels: pass label **keys** at creation (a list of strings) and matching label **values** on each update (in the same order). Use `'()` for an unlabelled metric.

## Constructors & accessors

### `(make-counter name help label-keys)`
Create a monotonic counter. `name` must match `[a-zA-Z_:][a-zA-Z0-9_:]*`; each label key must match `[a-zA-Z_][a-zA-Z0-9_]*`.

### `(make-gauge name help label-keys)`
Create a gauge (value may rise or fall).

### `(make-histogram name help label-keys buckets)`
Create a histogram. `buckets` is a list of strictly-increasing numeric upper bounds; a `+Inf` bucket is added at render time.

### `(metric-name m)` / `(metric-help m)` / `(metric-kind m)`
Accessors. `metric-kind` returns the symbol `'counter` or `'gauge` for those; for a histogram it returns the **list** `(histogram (bucket …))`, not a bare symbol.

## Counter

### `(counter-inc! m label-vals)`
Increment by 1 for the given label values.

### `(counter-add! m label-vals n)`
Add `n` (must be `>= 0`). Negative → error.

## Gauge

### `(gauge-set! m label-vals v)` — set to `v`.
### `(gauge-inc! m label-vals)` / `(gauge-dec! m label-vals)` — ±1.

## Histogram

### `(histogram-observe! m label-vals value)`
Record `value`: increments every bucket whose upper bound `>= value`, plus the count and sum. `value` must be numeric.

### `(histogram-buckets m)`
Return the bucket bounds list.

## Registry

### `(metrics-register! m)`
Add a metric to the global registry (constructors call this automatically).

### `(metrics-render)`
Return the full Prometheus text-format string of all registered metrics.

### `(metrics-reset!)`
Zero counters and histograms; **gauges retain** their value (a gauge represents live state, not a rate).

```scheme
;; metrics.esk
(require core.metrics)
(define reqs (make-counter "eshkol_requests_total" "Total requests" (list "path")))
(counter-inc! reqs (list "/a"))
(counter-add! reqs (list "/a") 4)
(counter-inc! reqs (list "/b"))
(define g (make-gauge "eshkol_inflight" "In flight" (list)))
(gauge-set! g (list) 10) (gauge-inc! g (list)) (gauge-dec! g (list))
(define h (make-histogram "eshkol_latency" "Latency" (list "path") (list 0.1 0.5 1.0)))
(histogram-observe! h (list "/a") 0.07)
(histogram-observe! h (list "/a") 0.3)
(histogram-observe! h (list "/a") 2.0)
(display (metric-kind reqs)) (newline)
(display (metric-kind h)) (newline)
(display (metrics-render))
```
```
counter
(histogram (0.1 0.5 1))
# HELP eshkol_requests_total Total requests
# TYPE eshkol_requests_total counter
eshkol_requests_total{path="/b"} 1
eshkol_requests_total{path="/a"} 5
# HELP eshkol_inflight In flight
# TYPE eshkol_inflight gauge
eshkol_inflight 10
# HELP eshkol_latency Latency
# TYPE eshkol_latency histogram
eshkol_latency_bucket{path="/a",le="0.1"} 1
eshkol_latency_bucket{path="/a",le="0.5"} 2
eshkol_latency_bucket{path="/a",le="1"} 2
eshkol_latency_bucket{path="/a",le="+Inf"} 3
eshkol_latency_count{path="/a"} 3
eshkol_latency_sum{path="/a"} 2.37
```

After `(metrics-reset!)` the counter and histogram series drop out of the render output while the gauge line `eshkol_inflight 10` remains.

Edge cases (all raise `Unhandled exception`):
- label value count ≠ declared label keys → `metric label value count does not match label keys`
- `counter-add!` with `n < 0` → `counter-add!: counters only increment`
- invalid metric name (e.g. `"9bad"`) → `metric name must match [a-zA-Z_:][a-zA-Z0-9_:]*`
- non-increasing histogram buckets → `histogram buckets must be strictly increasing`

(Timestamps and label ordering in output are deterministic; the example above is exact verbatim output.)
