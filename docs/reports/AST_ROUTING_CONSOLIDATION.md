# AST and callable routing consolidation

The compiler architecture gate previously found 32 partial AST switches (64
omission/default findings) and seven callable consumers outside the canonical
invocation path. The gate now reports one exhaustive operation dispatcher,
32 complete consumer policies, one callable dispatcher, and zero findings.

`inc/eshkol/core/ast_routing.h` owns the runtime switch over all 113
`eshkol_op_t` members. Each consumer supplies explicit operation groups whose
results belong to its local route enum. Template instantiation requires every
operation exactly once. The central switch enables exhaustive-switch errors;
the structural gate separately checks the operation domain, group uniqueness,
route-arm equality, defaults, and bypasses across the compiler sources and
headers. A new operation therefore requires a deliberate decision in every
policy. Policies preserve the existing passes' distinct behavior and union
payload assumptions; an explicitly listed no-effect operation is not claimed
to have acquired new traversal semantics.

All discovered callable invocation paths enter `codegenClosureCall`. Apply
stages and initializes arguments before canonical spread dispatch, with a
separate empty-argument path for parameter getters. Parallel workers receive
the canonical callback before their bodies are generated. Higher-order
derivatives capture the resolved procedure value. Known ABI signatures for
checked REPL forward references and local self-calls are explicit inputs to
the same invocation function. The callable scanner was not weakened.

## Reproducible checks

```sh
python3 scripts/gate_compiler_architecture.py
python3 scripts/gate_compiler_architecture.py --self-test
python3 tests/backend/ast_routing_compile_mutations.py --cxx c++
python3 tests/backend/run_ast_callable_routing.py --build BUILD
ctest --test-dir BUILD -R '^(compiler_architecture_gate|compiler_architecture_gate_selftest|ast_routing_compile_mutations|ast_callable_routing|semantic_identity_test)$' --output-on-failure
```

Validation used macOS arm64, LLVM 21, a Release CPU build with Agent FFI and
REPL enabled, and a standard library rebuilt by the modified compiler. The
build used cached pinned FetchContent sources in a separate build directory.

- Architecture gate: PASS, zero findings.
- Structural mutation controls: 14 passed, including omitted/duplicate/unknown
  operations, missing/default/nested route arms, bypasses, and rogue callables.
- Real compiler probes: complete-domain dispatch passed for all 113 operations;
  omitted policy, duplicate policy, and new enum mutations failed compilation.
  A compilable but incorrect route assignment failed the runtime oracle.
- New callable regression: 14 assertions each under native JIT and AOT,
  covering derivatives, captures, variadics, parameter procedures, arithmetic,
  macros, parallel workers, mutable captures, and apply tail recursion.
- Existing curried derivative, shared mutable capture, and tensor callable
  regression suites passed under JIT (53, 24, and 31 assertion lines).
- Existing first-class variadics suite passed under JIT against the rebuilt
  standard library (26 PASS lines, including its final zero-failures summary).
- Parallel capture boundaries 33, 65, 200, and 4096 passed under JIT and AOT.
- All five targeted CTest checks passed, including semantic identity.

The source gate remains a structural lexical policy, not a C++ data-flow
proof. Runtime and compilation mutation checks provide independent evidence
for the routing mechanism and invocation behavior.

## Continuation-parser integration

Rebased onto `cf7fe34a` (explicit parser continuations). The reference collector
retains its `pending` work stack; operation routing selects which children to
queue. All 17 enqueue expressions are preserved in their original order, and
there is no recursive collector call. The parser continuation implementation
and its grammar changes remain intact.

Post-rebase validation: all seven targeted CTests pass (the five routing and
identity checks above plus `parser_explicit_stack` and `parser_stack_compile`).
The parser tests preserve 16,000-level syntax on an actual 8 MiB worker stack,
compile the shipped standard library with an 8 MiB process limit, and execute
16,000 nested additions through JIT and AOT. The four existing callable suites
also pass again (53/24/31/26 passing output lines as described above).
