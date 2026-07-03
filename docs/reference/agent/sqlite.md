# `agent.sqlite` — Embedded SQLite

Thin, safe bindings over the bundled SQLite C library. Handles are opaque `i64`
integers (a database connection or a prepared statement).

```scheme
(require agent.sqlite)
```

Source: `lib/agent/sqlite.esk`. C symbols: `eshkol_sqlite_*`.

## Connection lifecycle

| Procedure | Signature | Notes |
|-----------|-----------|-------|
| `sqlite-open` | `(sqlite-open path)` | Returns handle `>= 0`, or negative on error |
| `sqlite-close` | `(sqlite-close handle)` | |
| `with-db` | `(with-db path fn)` | Opens, calls `(fn db)`, closes on exit via `dynamic-wind`; `error`s if open fails |

## Statement lifecycle

| Procedure | Signature |
|-----------|-----------|
| `sqlite-prepare` | `(sqlite-prepare db sql)` → stmt handle (negative on error) |
| `sqlite-step` | `(sqlite-step stmt)` → `#t` if a row is available (`SQLITE_ROW`) |
| `sqlite-reset` | `(sqlite-reset stmt)` |
| `sqlite-finalize` | `(sqlite-finalize stmt)` |
| `with-statement` | `(with-statement db sql fn)` — prepares, `(fn stmt)`, finalizes on exit |

## Direct execution

| Procedure | Signature | Notes |
|-----------|-----------|-------|
| `sqlite-exec` | `(sqlite-exec handle sql)` | Execute SQL without a result cursor |
| `sqlite-exec-safe` | `(sqlite-exec-safe handle sql)` | Guarded variant |

## Binding parameters (1-indexed)

`sqlite-bind-text`, `sqlite-bind-int`, `sqlite-bind-double`, `sqlite-bind-null`
— each `(… stmt idx value)`.

## Reading columns (0-indexed)

`sqlite-column-text`, `sqlite-column-int`, `sqlite-column-double`,
`sqlite-column-count`, `sqlite-column-name`, `sqlite-column-type`.

`sqlite-column-text` allocates a buffer sized from `sqlite-column-bytes-raw`, so
dynamically-sized TEXT/BLOB columns are read in full (this fixed an earlier
fixed-buffer truncation bug — see session-persistence work).

## Metadata

| Procedure | Returns |
|-----------|---------|
| `sqlite-last-error` | error string (via a 1024-byte scratch buffer) |
| `sqlite-last-insert-rowid` | rowid `i64` |
| `sqlite-changes` | rows affected |

## Constants

`SQLITE_ROW`, `SQLITE_DONE`.

## Verified example

```scheme
(require agent.sqlite)
(with-db "/tmp/demo.db"
  (lambda (db)
    (sqlite-exec db "CREATE TABLE t(id INTEGER, name TEXT)")
    (with-statement db "INSERT INTO t VALUES(?,?)"
      (lambda (s)
        (sqlite-bind-int s 1 1)
        (sqlite-bind-text s 2 "alice")
        (sqlite-step s)))
    (with-statement db "SELECT name FROM t WHERE id=?"
      (lambda (s)
        (sqlite-bind-int s 1 1)
        (when (sqlite-step s)
          (display (sqlite-column-text s 0)) (newline))))))
```

AOT binaries link the SQLite objects automatically when a required module uses
`agent.sqlite` (see [FFI & AOT linking](ffi.md)).
