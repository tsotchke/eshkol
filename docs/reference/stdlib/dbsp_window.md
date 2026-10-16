# Explicit event-time windows

`(require core.dbsp.window)` maintains an event-time view over signed Z-set
updates. Watermarks are explicit exact rational inputs, never wall-clock reads.

```scheme
(define window (make-dbsp-window car 10 3 'reject 0))
(define step (dbsp-window-step window
  (zset-from-weighted '((1 (7 trade-a)) (1 (12 trade-b)))) 7))
(dbsp-window-output (vector-ref step 0))
```

Constructor arguments are pure/stable timestamp function, positive width,
nonnegative allowed lateness, `reject` or `drop`, and initial watermark.
The active interval is `(watermark-width, watermark]`. Future events are
buffered; advancing the watermark activates them and expires the lower
boundary. Rows newer than the lower boundary are retained, including future
rows. Expired rows are forgotten.

Each tick consolidates the supplied Z-set, admits positive insertions against
`timestamp >= previous-watermark-lateness`, then applies the new watermark.
An insertion outside that admission either refuses the complete tick or is
returned explicitly as dropped. Retractions of retained rows are allowed even
when late; absent/excess retractions always refuse. Integrated multiplicities
must stay nonnegative. A consolidated zero update needs no late admission.
There is no transport deduplication or implicit correction for forgotten events.

Step returns `#(successor signed-view-delta dropped-insertions work)`.
Insertions, deletions, activation and expiry share the same signed output
algebra. Feed this delta to `dbsp-ordered-step` for rolling exact quantiles/top-k,
or to another standing circuit. Snapshots/forks remain valid. Observations own
their outer containers; row payloads and the timestamp callback obey the
existing immutable-row/pure-function contract.

`dbsp-window-info` returns `#(watermark width lateness retained-distinct work)`.
Work is `#(changed-distinct candidate-scanned expired-distinct dropped-distinct)`.
This reference implementation scans the full candidate relation and pays for
Z-set consolidation/differencing. It is not an indexed expiration engine or
sparse-total-time claim. At most 65536 changed rows and 65536 transient candidate
rows are admitted; a large tick can refuse before expiration reduces storage.
Future buffering and retained history consume memory. Checkpoint/transport,
temporal joins, automatic watermarks and approximate quantiles are outside scope.
