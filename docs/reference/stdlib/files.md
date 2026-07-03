# `core.files` — file-system convenience helpers

**Source**: [`lib/core/files.esk`](../../../lib/core/files.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.files)`

Path helpers and atomic file writes. Atomic writes go through a temp file in the destination's directory that is renamed over the target only after the writer succeeds, so a reader never observes a partially written file. The examples below write to `/tmp` and clean up.

## Functions

### `(path-directory path)`
Returns the directory component of `path`. Handles both `/` and `\` separators. Special cases: a path with no separator returns `"."`; a root-anchored path returns `"/"`; a Windows drive prefix like `C:\...` returns the `"C:\"` root.

```scheme
;; files.esk
(require stdlib)
(display (path-directory "/tmp/foo/bar.txt")) (newline)
(display (path-directory "bar.txt")) (newline)
(display (path-directory "/etc")) (newline)
```
```
/tmp/foo
.
/
```

### `(atomic-write-file path contents)`
Convenience wrapper: atomically writes the string `contents` to `path`, returning `#t` on success and `#f` on setup/rename failure. Internally calls `with-atomic-output-file`.

```scheme
(require stdlib)
(display (atomic-write-file "/tmp/eshkol-doc-test.txt" "hello atomic\n")) (newline)
(display (read-file "/tmp/eshkol-doc-test.txt"))
(delete-file "/tmp/eshkol-doc-test.txt")
```
```
#t
hello atomic
```

### `(with-atomic-output-file path proc)`
Opens a temp file in `path`'s directory, calls `(proc port)`, closes the port, then renames the temp over `path`. Returns `proc`'s value when the rename succeeds; returns `#f` if the temp file could not be created or the rename failed. If `proc` raises, the temp file is deleted and the original exception is re-raised.

```scheme
(require stdlib)
(with-atomic-output-file "/tmp/eshkol-doc-test2.txt"
  (lambda (port) (write-string "line1\n" port) 'ok))
(display (read-file "/tmp/eshkol-doc-test2.txt"))
(delete-file "/tmp/eshkol-doc-test2.txt")
```
```
line1
```

Edge cases: the atomic guarantee relies on the temp file living in the same directory as the destination (same filesystem) so the rename is atomic — `path-directory` is used to place it there. On writer exceptions the partial temp file is removed before the exception propagates. Capability-gated file operations may cause the underlying open/rename to fail (see `core.capabilities`); the relevant denial behavior is tracked in `.swarm/tasks/ESH-0076` ("Capability-denied getenv/file ops return #f silently — make denial distinguishable", status: done).
