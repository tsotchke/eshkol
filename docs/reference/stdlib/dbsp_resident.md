---
kind: reference
status: current
owner-area: runtime
since: v1.4.0-connection
sources:
  - lib/core/dbsp/resident.esk
---

# `core.dbsp.resident`

`(require core.dbsp.resident)` adds explicit native ownership to ordinary standing
DBSP circuits, reusing transactional graph evacuation.

```scheme
(call-with-dbsp-resident circuit 1048576
  (lambda (owner initial)
    (let ((next (dbsp-resident-step owner initial changes)))
      (dbsp-resident-release! owner initial)
      (dbsp-resident-output owner next))))
```

The owner exists only inside this dynamic extent. Generation-checked integer
snapshot handles retain complete historical graphs until release or owner close.
They support forks and reject stale/foreign/cross-thread use. Output/delta/work
observations copy reachable owner and nursery data into the caller's domain.
Mutable input payloads are copied even when they live in an older caller arena
or in a static literal; mutating an observation cannot change a retained row.
Owner and snapshot handles must be positive exact signed-word integers.

Each operation uses a tick region. Inputs are imported so caller nursery rows
cannot become dangling retained data. Failed pure callbacks publish no successor;
exceptions are exported before leaving the borrow. Callbacks may not publish
internal objects through mutation or escape a borrow with nonlocal control.

At an idle boundary, owner used bytes above an explicit threshold trigger one
transaction copying all retained roots, preserving cycles/sharing and releasing
the old arena. Failure retains previous roots. Opaque runtime-side aliases,
parallel borrows and enclosing AD tapes refuse. Limits: 32 owners per thread,
4095 roots per owner, threshold 64KiB–1GiB. The threshold triggers compaction;
it is not a cap on legitimate live state.

`dbsp-resident-compact!` forces compaction; `dbsp-resident-stats` returns
`#(live-roots used-arena-bytes compactions copied-live-bytes threshold)`.
Compaction scans retained roots and reports copying. Keeping every historical
snapshot, growing logical data or retaining exported caller observations grows
memory. There is no background collector, VM reclamation, durable owner format
or whole-program bound. RSS claims require an executed memory-family receipt.

Gates cover forks, nursery imports, copied nested observations, failures and
handles; C++ checks graph cycles/sharing and allocation rollback. The sustained
family holds 16 versioned rows and one snapshot while increasing update count,
with a disabled-compaction growth control.
