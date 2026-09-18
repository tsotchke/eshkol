---
kind: project
status: current
owner-area: project
since: v1.3.5
sources:
  - CHANGELOG.md
  - tests/coverage/release_record.json
  - lib/types/type_checker.cpp
  - lib/types/type_relation.cpp
  - inc/eshkol/backend/static_callee_binding.h
---
# Upgrading to v1.3.5-evolve

What a program, a build or a contributor workflow written against
v1.3.4-evolve meets on v1.3.5-evolve. Each section states what is true now and
what, if anything, to do about it. The complete list of changes is the
`[1.3.5-evolve]` section of [CHANGELOG.md](../CHANGELOG.md); the release
summary is [RELEASE_NOTES.md](../RELEASE_NOTES.md).

Every program on this page was run on the release compiler, on the JIT
(`eshkol-run -r`) and as an AOT binary, with identical output.

## Contents

1. [The type checker sees more of your program](#1-the-type-checker-sees-more-of-your-program)
2. [Loop accumulators are typed by what they carry](#2-loop-accumulators-are-typed-by-what-they-carry)
3. [A procedure variable means its current value](#3-a-procedure-variable-means-its-current-value)
4. [Differentiation](#4-differentiation)
5. [Library behaviour](#5-library-behaviour)
6. [Building Eshkol](#6-building-eshkol)
7. [Environment variables](#7-environment-variables)
8. [Contributors](#8-contributors)

## 1. The type checker sees more of your program

The optional static checker examines every evaluated position of every control
form: the tests and bodies of `cond`, `case`, `match`, `when`, `unless` and
`do`, `guard` bodies and handlers, `and`/`or` operands, `set!` values,
quasiquote escapes, the thunks of `dynamic-wind` and `parameterize`, the parts of
`call/cc`, `values`, `call-with-values` and `let-values`, `with-region` bodies,
computed callees, the point of a calculus operator, and every expression of a
body rather than only the last. A call in any of those places is checked
against the callee's annotations exactly as the same call at top level.

A program that compiled silently on v1.3.4 may therefore print type warnings on
v1.3.5. Each one is a call whose argument contradicts an annotation:

```scheme
(define (area (w : number) (h : number)) (* w h))

(define (report flag)
  (when flag
    (area "wide" 3)))

(report #f)
(display "still runs")
(newline)
```

```text
still runs
```

```console
[WARN] Type warning: argument 1 of 'area': expected Number, got String (line 5:11)
   WARNING: HoTT: 1 type warnings detected (gradual typing continues)
```

**What to do.** A warning does not stop compilation or change what the program
computes. Under `--strict-types` every such diagnostic is an error and no code
is generated, so a strict build that passed on v1.3.4 can stop on v1.3.5 at a
mistake it did not see before. Fix the call, narrow the value with a type
predicate, or ascribe it with `(the type expr)` where you know more than the
checker. The rules, and how to read a diagnostic, are in
[the gradual typing guide](guide/GRADUAL_TYPING.md).

Three related rules apply everywhere:

- A multi-branch form has the **join** of its branch types, and `if` and the
  equivalent `cond` have the same type.
- A **return annotation** accepts a body the checker types as `Value` (unknown),
  as an argument position does.
- **Function types** print as arrows in diagnostics, for example
  `expected (-> Number Number), got (-> Int64 Number)`, and are checked
  contravariantly in their parameters.

## 2. Loop accumulators are typed by what they carry

An unannotated named-`let` parameter has the join of its initial value and of
every argument the loop passes back to it. Accumulators that start as a
literal and grow, such as a pair of flonums fed the results of arithmetic, are
accepted without warnings, including the ones in the standard library's
interval and Taylor-model modules:

```scheme
(define (mean-pair xs)
  (let loop ((rest xs) (acc (cons 0.0 0)))
    (if (null? rest)
        (/ (car acc) (cdr acc))
        (loop (cdr rest)
              (cons (+ (car acc) (car rest)) (+ (cdr acc) 1))))))

(display (mean-pair (list 1.0 2.0 3.0 6.0)))
(newline)
```

```text
3
```

An argument with nothing in common with the seed, such as a string passed to a
parameter seeded with `0`, is still reported. Nothing to do, unless you had
added an annotation or a `the` only to silence one of these warnings; it can go.

## 3. A procedure variable means its current value

A call through a variable reaches the procedure the variable holds at that
moment, on every path: a direct call, `apply`, `map`, `vector-map`, `reduce`,
`remove` and the differentiation operators. A variable reassigned with `set!`,
a top-level name defined twice, and a parameter or `let` binding that reuses a
procedure's name all behave that way.

```scheme
(define twice (lambda (x) (* 2 x)))
(define before (twice 5))
(set! twice (lambda (x) (* 3 x)))
(display (list before (twice 5) (apply twice (list 5)) (map twice (list 1 2 3))))
(newline)

(define G (lambda (x) (* x x)))
(set! G (lambda (x) (* x x x)))
(display (derivative G 2.0))
(newline)
```

```text
(10 15 15 (3 6 9))
12
```

**What to do.** A program that relied on `map`, `apply` or `derivative` using a
procedure's first binding after a reassignment gets the current one; bind the
first procedure to its own name if that is what you meant. Bindings that are
never reassigned keep direct calls, so this costs nothing where nothing
changes. See [the binding reference](reference/language/binding-mutation-and-scope.md#procedure-bindings-that-change).

A user procedure may share its name with a C math library routine (`exp`,
`log`, `tanh`, ...) and with the tensor activations that use one; both work in
the same program, in either order.

## 4. Differentiation

- **Exact at exact points.** Differentiating a polynomial or rational function
  at an exact point answers exactly, and the exact tower stays closed through
  `sqrt` of a perfect square and `expt` with an exact exponent:

  ```scheme
  (define (f x) (* x x x))
  (display (derivative f 2))
  (newline)
  (display (exact? (derivative f 2)))
  (newline)
  (display (derivative f 1/3))
  (newline)
  (display (derivative-n f 2 2))
  (newline)
  (display (sqrt 4/9))
  (newline)
  (display (expt 2/3 -3))
  (newline)
  ```

  ```text
  12
  #t
  1/3
  12
  2/3
  27/8
  ```

  Code that tested a derivative with `inexact?`, or compared it with `eqv?` to a
  flonum, sees an exact number where it saw a flonum. Use `=` or `exact->inexact`.

- **Captures are resolved where the closure was made.** A let-bound closure that
  captures a variable can be differentiated from inside a nested lambda, and a
  binding at the call site that shares a captured variable's name does not
  disturb it. See
  [the AD capture rules](reference/ad/operators.md#where-a-capture-is-resolved).

The complete AD support matrix, including the nesting ceiling, is
[reference/ad/support-matrix.md](reference/ad/support-matrix.md).

## 5. Library behaviour

- `json-get-in` takes an optional default, returned when the path is absent:
  `(json-get-in obj path default)`. See
  [the JSON reference](reference/stdlib/json.md).
- `tensor-apply` resolves its callable lexically: a local binding whose name is
  also a builtin's is the one called. Pass the procedure you mean.
- The legacy `riemannian-adam-step` form refuses on the VM rather than sharing
  optimizer moments between parameters; allocate a state per parameter.
- Tensor and model saves write validated ESKM v1, and loading validates the
  payload.
- `manifold-dim` returns the manifold's dimension, and the VM's raw manifold
  operations are the `-handle` family, so they no longer shadow `core.manifold`.
- `make-vector` has one capacity limit and one diagnostic on every engine.
- The size variables (`ESHKOL_STACK_SIZE` and its siblings) accept `K`/`M`/`G`
  and `KiB`/`MiB`/`GiB`, and report a value they cannot parse.

## 6. Building Eshkol

- **Host compilers.** v1.3.5-evolve is built and verified with GCC 13 and with
  Clang/LLVM 21. GCC 15 is not a supported host compiler in this release: where
  it is the system default, select a supported one (`CC=gcc-13 CXX=g++-13`, or
  Clang 21). See
  [Supported host compilers](platform/BUILD_NOTES.md#supported-host-compilers).
- **`roundeven`.** Scheme `round` is ties-to-even. Where the platform C library
  lacks the C23 `roundeven`, the runtime supplies it; the configure step prints
  which one is used.
- **Python bindings.** The extension requires the Python *development*
  component (headers and the embedding library), not only the interpreter, and
  on ELF platforms the build makes the archives it links position-independent
  automatically. See [Python bindings](platform/BUILD_NOTES.md#python-bindings).
- **Deeply nested programs** compile ahead of time within the default 8 MiB
  stack.

Problems and their fixes are collected in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## 7. Environment variables

Read by the compiler and runtime since v1.3.5. The full table, with defaults,
is [the environment-variable reference](reference/runtime/environment-variables.md).

| Variable | Purpose |
|---|---|
| `ESHKOL_AOT_MODULE_CACHE_DIR` | Directory of the content-addressed AOT module cache |
| `ESHKOL_AOT_MODULE_CACHE_TRACE` | Trace AOT module-cache hits and misses |
| `ESHKOL_ARENA_REPORT` | `1` prints the global arena's byte total at exit |
| `ESHKOL_SUBPROC_MAX_CONCURRENT` | Maximum concurrently running spawned children (default 64) |
| `ESHKOL_VM_REGION_EVAC`, `ESHKOL_VM_REGION_COMPACT`, `ESHKOL_VM_REGION_RECYCLE`, `ESHKOL_VM_REGION_VERIFY`, `ESHKOL_VM_REGION_VERIFY_FATAL` | Bytecode-VM region reclamation controls |
| `ESHKOL_DUMP_IR_ON_VERIFY_FAIL` | Print the LLVM module when verification fails |
| `ESHKOL_LLVM_REMARKS` | Let LLVM optimization remarks reach stderr |
| `ESHKOL_TAIL_TRANSFER_ONLY` | Force mutual tail calls onto the portable tail-transfer dispatcher |
| `ESHKOL_LANGUAGE_COVERAGE_HOOK_STATS`, `ESHKOL_NODE_IDENTITY_STATS` | Coverage and node-identity diagnostics for the gates |
| `ESHKOL_DENSE_TENSOR_AD_NODES`, `ESHKOL_AD_STRICT` | AD engine switches; see the reference for their current effect |

For evidence-producing scripts, a relative `TRACE_DIR` or `ICC_TRACE_DIR`
means a path relative to the repository root.

## 8. Contributors

- **Tutorial and guide examples are executed** on the JIT and as AOT binaries
  by the documentation example gate. A page change that breaks an example fails
  CI; an example that cannot run carries an explicit marker. See
  [DOCUMENTATION.md](DOCUMENTATION.md#5-examples-are-executed).
- **Every merged pull request needs a home**: a `CHANGELOG.md` reference, or a
  reasoned entry in `tests/coverage/changelog_no_user_facing_change.json`.
- **Release facts have one source**, `tests/coverage/release_record.json`; edit
  the record and run `python3 scripts/check_surface_counts.py --sync`.
- **Scripts that write generated files** use the helpers in
  `scripts/lib/checked_write.sh`, and scripts that read an evidence path use
  `scripts/lib/evidence_paths.sh`.
- **Architecture Decision Records** have unique numbers; three were renumbered
  to 0016, 0017 and 0018. See [the ADR index](design/adr/README.md).

The contributor workflow is in [CONTRIBUTING.md](../CONTRIBUTING.md), the
documentation system in [DOCUMENTATION.md](DOCUMENTATION.md), and the release
process in [platform/RELEASE_PROCESS.md](platform/RELEASE_PROCESS.md).
