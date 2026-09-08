# Parser continuation stack

The syntax parser now suspends child parses in C++20 coroutine frames and
resumes them through an explicit linked continuation stack. `await_suspend`
only records a child; `final_suspend` only suspends. Neither resumes another
frame. Native stack use therefore stays independent of grammar nesting even
when the compiler is built without optimization.

The grammar bodies retain their dispatch, token consumption, diagnostics,
AST construction and source stamping. Expressions, list special forms,
quoted/quasiquoted data, vectors, interpolation, types, match patterns,
syntax-rules patterns, import sets and feature requirements all use the same
driver. Exceptions propagate back through suspended parents, destroying each
completed child before resuming the next parent.

Two synchronous helpers also needed attention: lambda capture reference
collection uses an explicit pending-node vector, and type parsing attaches
fresh arena-owned child trees to shallow constructor results. The public type
constructors continue to copy borrowed trees. Avoiding those copies during
parsing removes recursive subtree copying and quadratic type allocation.

The Linux stack guard now measures remaining space from the low stack bound;
the previous expression inverted remaining and consumed space. Its 65,536-byte
safety margin and diagnostic text are unchanged. No production stack limit,
build stack setting, ESH-0103 threshold or skip behavior was changed.

## Regression gates

- `parser_explicit_stack`: an actual 8 MiB pthread stack and an 8 MiB process
  resource limit; 16,000 levels of arithmetic, reader data, interpolation,
  vectors, types, special forms, patterns and import/feature requirements.
  Checks arithmetic structure, operand order, source locations, node identity,
  dotted-tail structure, vector shape, type structure, malformed syntax,
  recovery, exception propagation and continuation destruction.
- `parser_stack_compile`: fixes both soft and hard process stack limits at
  8 MiB, compiles the shipped stdlib to nonempty object/bitcode files, then
  executes 16,000 nested additions through JIT and AOT and requires 16000.
- CI runs both gates on Linux x64 lite and macOS ARM64 lite.

macOS fixes its main stack mapping at executable load time; the dedicated
pthread test independently enforces an actual 8 MiB mapping there. Linux's
process-stack gate checks the normal limited process path.

## Local evidence (macOS ARM64, September 8, 2026)

- Both new CTests and the unchanged ESH-0103 checker self-test: 3/3 pass.
- Same stress harness linked against the original parser at `55c72c18`: fails
  with `stack space exhausted during parsing — expression nesting too deep`.
- New parser compiled separately with `-O0`: stress harness passes.
- Existing parser suite: 31 passed, 0 failed.
- Comprehensive type suite: 37 passed, 0 failed.
- Unchanged ESH-0103 gate under an inherited 8 MiB hard limit: all six samples
  pass. At 16k, JIT: 2.663 s / 522.2 MiB; AOT: 1.470 s / 426.3 MiB.
  Largest time ratio: 4.377; largest RSS ratio: 2.941 (limit: 8).

ICC context retrieval was attempted first and refused because the registered
`eshkol_lang` store was BLIND (missing index and memory). Findings above use
live source and measured executable results; no stale ICC verdict is claimed.
