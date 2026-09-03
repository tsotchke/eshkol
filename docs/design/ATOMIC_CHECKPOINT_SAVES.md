# Atomic Checkpoint Save Contract

Status: normative for hosted `tensor-save` and `model-save`.

## Guarantee

A successful save publishes one complete, valid checkpoint at the requested
path. `tensor-save` retains its existing ESKT bytes; `model-save` retains its
ESKM v1 bytes. A failure before publication leaves an existing destination
byte-for-byte unchanged. Success and handled failure leave no transaction
temporary file behind.

This contract changes neither the public API nor either byte format. The native
JIT/AOT runtime and the source/bytecode VM use the same publication primitive.

## Format and dispatch compatibility

The public `tensor-save`/`tensor-load` routes remain on their established ESKT
implementations, including native file capability checks. This milestone wraps
those writers rather than rerouting the API to the separate ESKM single-record
helpers. The focused gate compares every tensor output with an independently
constructed ESKT fixture and every model output with an independently
constructed ESKM v1 fixture under JIT, AOT, VM source, and VM bytecode.

## Publication protocol

1. Create a unique file with exclusive-create semantics in the destination's
   directory. Its fixed-size basename is an implementation detail with the
   form `.eshkol.<unique>` and does not grow with the destination basename.
   POSIX uses `mkostemp(O_CLOEXEC)` where available and a checked
   `fcntl(FD_CLOEXEC)` fallback otherwise; Windows uses `_O_NOINHERIT`.
2. Serialize the existing ESKT (`tensor-save`) or ESKM v1 (`model-save`) bytes
   to that file. Every write is checked.
3. Flush the C stream and close it, checking both operations.
4. Rename the temporary file over the destination. Same-directory rename is
   the only commit point.
5. On any earlier error, close and unlink the temporary file and report save
   failure.

No code path truncates or writes through the destination before commit.

## Filesystem behavior

- A new checkpoint has exactly mode `0600` on POSIX. The implementation applies
  that mode explicitly after exclusive creation, so even a restrictive process
  umask cannot remove the owner read/write bits.
- Replacing an existing regular file preserves its POSIX permission bits.
  Ownership, ACLs, extended attributes, and special mode bits are not copied.
- A destination symlink is replaced as a directory entry. Its referent is not
  opened, truncated, or modified.
- A destination directory or a path whose parent cannot create files causes a
  clean failure.
- Concurrent writers do not coordinate or merge. Each successful writer
  publishes a complete checkpoint atomically; the last successful rename wins.
  Readers may observe either complete version, never a partially serialized
  version.

## Interruption and durability boundary

On POSIX, the saving thread defers `SIGHUP`, `SIGINT`, `SIGQUIT`, and `SIGTERM`
from temporary-file creation through commit or cleanup. The original signal
mask is restored immediately after that boundary. This prevents a catchable
interruption delivered to the saving thread from abandoning a named temporary
file halfway through the transaction. Normal error returns also always clean
up.

There is no claim for `SIGKILL`, abrupt machine loss, kernel failure, or storage
failure. The implementation does not `fsync` the checkpoint and its parent
directory, so it does not claim power-loss durability. A future durability
milestone must add and test both file and directory synchronization before
making that stronger promise.

## Test-only failure injection

The dedicated `atomic_checkpoint_file_test` executable compiles a private copy
of the transaction helper with `ESHKOL_MODEL_IO_TEST_HOOKS` and may set
`ESHKOL_TEST_MODEL_IO_FAIL` to one of:

- `open`
- `write` or `write:N` (fail the first or Nth write call)
- `flush`
- `close`
- `interrupt` (a controlled pre-commit cancellation after close)
- `signal` (raise `SIGTERM` before commit while the transaction mask is active)
- `rename`

Neither `eshkol-runtime` nor `eshkol-static` ever compiles that definition,
including in a default `ESHKOL_BUILD_TESTS=ON` build. Their helper object has no
environment-hook string or `getenv` reference. The variable is not a supported
application interface.

Every injected failure must return false, preserve a pre-existing destination,
and leave no `.eshkol.*` file. Separately, the production-binary acceptance
gate exercises tensor and model saves in JIT, AOT, VM source, and VM bytecode
modes. It verifies exact ESKT and ESKM v1 bytes, missing-parent and
destination-directory failures, permissions and symlink behavior, and
concurrent writers without using the artificial hook.
