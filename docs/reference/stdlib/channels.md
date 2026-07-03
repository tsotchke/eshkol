# `core.channels` — Go-style bounded CSP channels

**Source**: [`lib/core/channels.esk`](../../../lib/core/channels.esk)
**Require**: `(require core.channels)` — **NOT** auto-loaded via `stdlib`; require it explicitly. (It internally pulls in `stdlib` and `core.threads`.)

Bounded, blocking channels built on `core.threads` mutex + condvar. A channel is a fixed-capacity ring buffer; blocking `channel-send!`/`channel-recv!` are the default, with non-blocking `channel-try-send!`/`channel-try-recv!` for poll-style use. A channel is a length-10 vector tagged `'eshkol-channel`.

Once **closed**: `channel-send!` errors, `channel-recv!` drains any buffered items then returns the closed-sentinel forever, and all waiters are woken.

> This pure-Scheme module is the precursor to the planned native channel primitive tracked in **ESH-0016** ("Channel API on the work-stealing thread pool", v1.4 concurrency foundation, status: active), which will add `ESHKOL_VALUE_CHANNEL` as a tagged-value subtype with lock-free SPSC/bounded/unbounded backends. Until then, use this module and drive concurrency via `parallel-map` / `parallel-execute` (see `core.threads`).

## Functions

### `(make-channel cap)`
Create an open, empty channel with integer capacity `cap` (must be `>= 1`). You are responsible for eventually calling `channel-close!` so blocked receivers can drain.

### `(channel? ch)`
Predicate: length-10 vector tagged `'eshkol-channel`.

### `(channel-capacity ch)`
The configured capacity (no locking).

### `(channel-len ch)`
Current number of buffered items (takes the lock).

### `(channel-closed? ch)`
`#t` if the channel has been closed (takes the lock).

### `(channel-send! ch val)`
Blocking send. Blocks while the buffer is full; **errors** if the channel is closed.

### `(channel-recv! ch)`
Blocking receive. Blocks while empty and open. Returns `(channel-closed-sentinel)` once the channel is closed AND drained — test with `eq?`.

### `(channel-try-send! ch val)`
Non-blocking send. Returns `#t` if delivered, `#f` if the buffer is full or the channel is closed.

### `(channel-try-recv! ch)`
Non-blocking receive. Returns the value if available; `(channel-closed-sentinel)` if closed AND drained; `#f` if open but empty.

### `(channel-close! ch)`
Mark closed and wake all waiters. Idempotent.

### `(channel-closed-sentinel)`
Returns the unique sentinel object that `channel-recv!`/`channel-try-recv!` yield on a closed+drained channel. Compare with `eq?`.

```scheme
;; channels.esk
(require core.channels)
(define ch (make-channel 2))
(display (channel? ch)) (newline)
(display (channel-capacity ch)) (newline)
(channel-send! ch 'a)
(channel-send! ch 'b)
(display (channel-try-send! ch 'c)) (newline)   ; buffer full -> #f
(display (channel-len ch)) (newline)
(display (channel-recv! ch)) (newline)           ; a
(display (channel-try-recv! ch)) (newline)       ; b
(display (channel-try-recv! ch)) (newline)       ; open + empty -> #f
(channel-close! ch)
(display (eq? (channel-recv! ch) (channel-closed-sentinel))) (newline) ; #t
(display (channel-try-send! ch 'x)) (newline)    ; closed -> #f
```
```
#t
2
#f
2
a
b
#f
#t
#f
```

Edge cases: `(make-channel 0)` → `Unhandled exception: make-channel: capacity must be >= 1`. `channel-send!` on a closed channel → `Unhandled exception: channel-send!: send on closed channel`. Keep examples bounded — `channel-recv!` on an open, empty channel blocks forever without another thread sending.
