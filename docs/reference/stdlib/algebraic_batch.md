---
kind: reference
status: current
owner-area: stdlib
since: v1.4.0-connection
sources:
  - lib/core/algebraic_batch.esk
  - tests/stdlib/algebraic_batch_test.esk
  - examples/algebraic_batch.esk
---

# Packed exact algebraic multiplication

`core.algebraic_batch` compiles a certified field and any admitted nested
Taylor/hyperdual coefficient domain into a sparse integer multiplication table.
`(make-algebraic-multiplication-plan owner [component-budget])` admits at most
64 flattened rational coordinates and 8192 nonzero table terms. The optional
component budget is an exact integer from 1 through 64. Plans/owners are
immutable by contract and retain explicit field/context identity.

`(algebraic-batch-mul plan left right)` accepts equally sized vectors of at most
4096 values and returns `(vector outputs receipt)`. Each input is packed with
its own common rational denominator. Preflight checks input integers, table
constants, denominator products and per-output absolute-term sums against
signed-32-bit limits before the entire batch enters the integer kernel.
Absolute sums conservatively bound every accumulation; cancellation cannot
hide an overflowing term. Exact rational coordinates are restored per pair.

The hot loop is ordinary native-compiled Eshkol integer multiply/accumulate,
with runtime scalar integer storage. This is not an alternate LLVM generator,
an i32 buffer ABI, floating tensor computation or arbitrary-precision native
backend. The receipt vector contains backend `native-scheme-integer`, batch
length, component count, table-term count, maximum absolute sum, executed term
products and kernel calls. Empty batches make zero kernel calls. No hidden
generic fallback or approximate conversion occurs; out-of-range cases refuse.

The generic field/AD multiplication remains the independent oracle. Existing
exact scalar limits still apply, with arithmetic demotion refused. The native
path accelerates multiplication only; inverse/root operations remain owned by
the existing exact carriers.

Run `ctest --test-dir build --output-on-failure -R algebraic_batch`.
`bench/algebraic_batch.esk` compares four passes of a dense cubic-field
hyperdual batch, including packing, preflight and reconstruction. Set
`ESHKOL_BATCH_MODE=packed` or `generic` and `ESHKOL_BATCH_COUNT=1`, `32` or
`256`. Plan creation is outside the inner timing; every output is checked after
timing against a generic product. Report measured crossover and RSS rather
than inferring speed from operation counts.
