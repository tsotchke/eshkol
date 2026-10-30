# HoTT: identity types, univalence and higher inductive types

Eshkol's type system contains a cubical type theory (ADR-0043): identity
types as path types, path induction, transport, function extensionality,
univalence and higher inductive types, all of which compute. Proofs are
checked when the program is compiled; values they compute can be brought
into the running program. The same kernel runs for the JIT, AOT binaries and
the bytecode VM.

A guided introduction is [docs/guide/HOTT.md](../../guide/HOTT.md).

## The two forms

| Form | Meaning |
|------|---------|
| `(hott decl ...)` | Check declarations in the kernel at compile time. No runtime code. |
| `(hott-eval term)` | Normalize `term`; the form is replaced by the normal form as a quoted datum. |

A form the kernel rejects is a compile-time error in every mode (also
without `--strict-types`): no binary is written and nothing runs. Kernel
declarations accumulate over a program, in order; in the REPL they persist
across inputs.

```scheme
(hott
  (import numbers)
  (define three : Nat (add 1 2)))

(display (hott-eval three)) (newline)
(display (hott-eval (sucInt (negsuc 0)))) (newline)
```

Output:

```text
3
(pos 0)
```

Natural numbers print as numerals; other inductive values print as
constructor applications.

## Declarations

| Declaration | Meaning |
|-------------|---------|
| `(import module)` | Load a library module (below). Idempotent. |
| `(data Name ctor ...)` | An inductive type in `U`. |
| `(data (Name (l : Level) (A : (U l)) ...) : (U l) ctor ...)` | A parameterized inductive type. |
| `(define name : T body)` | A definition. |
| `(define (name (x : A) ...) : T body)` | A definition with parameters; level parameters `(l : Level)` come first. |
| `(example T body)` | Check `body : T` without naming it. |
| `(builtin natural Nat)` | Numerals denote `Nat` (the `numbers` module does this). |

A constructor is `name`, or `(name (x : A) ... (dims i ...) (boundary (face t) ...))`.
Constructors with `dims` are path constructors: `(loop (dims i) (boundary ((= i 0) base) ((= i 1) base)))`
declares `loop` as a path from `base` to `base`.

## Terms

| Term | Meaning |
|------|---------|
| `U`, `(U l)` | Universes; `U` is `(U 0)`, and `(U l) : (U (lsuc l))`. Levels: numerals, level parameters, `(lsuc l)`, `(lmax l k)`. |
| `(Pi ((x : A) ...) B)`, `(-> A ... B)` | Dependent and plain function types. |
| `(Sigma ((x : A) ...) B)`, `(pair a b)`, `(fst p)`, `(snd p)` | Dependent pairs. |
| `(lambda (x ...) body)` | A function, or a path when checked against a path type. |
| `(f a ...)` | Application; a path applied to an interval expression is path application. |
| `(Path A a b)`, `(PathP (i) A a b)` | Paths in `A`; dependent paths over the line `(i) A`. |
| `(transp (i) A phi u)` | Transport `u : A[i0/i]` along the line `(i) A`, constant where `phi` holds. |
| `(hcomp A (i) ((face side) ...) base)` | Homogeneous composition. |
| `(comp (i) A ((face side) ...) base)` | Heterogeneous composition. |
| `(Glue A ((face T e) ...))`, `(glue ((face t) ...) a)`, `(unglue x)` | Glue types (`e : Equiv T A`). |
| `(split (pattern body) ...)` | A function out of an inductive type, by cases on constructors (path constructors bind their dimensions). |
| `(let ((x : T e) ...) body)`, `(the T e)` | Local definitions and ascriptions. |

Interval expressions are `i0`, `i1` (also `0`, `1`), dimension variables,
`(~ r)`, `(min r s ...)` and `(max r s ...)`. Faces are `(= r 0)`, `(= r 1)`,
`(and face ...)` and `(or face ...)`.

The kernel is predicative and total: data types must be strictly positive, a
definition may call itself only on a structurally smaller pattern variable,
a `split` must cover every constructor and agree with the boundary of every
path constructor, composition sides must meet the base and agree on
overlapping faces, and `transp` must be along a line constant where its
formula holds.

## What computes

Definitionally (by evaluation):

- beta, eta (functions, pairs, paths, Glue elements) and unfolding of definitions;
- the De Morgan interval laws and path endpoints (`(p i0)` is the start of `p`);
- `transp` and `hcomp` in every type former; `hcomp` in a universe is a Glue type;
- transport along `(ua e)` applies `(fst e)` (exactly, when the target is an inductive type);
- `(happly (funExt h))` is `h`; eliminators of higher inductive types on points, paths and compositions;
- J on `refl` when the motive is an inductive type.

As terms (proofs): `J-refl` in general, `ua-beta` in general, the groupoid laws,
`iso-to-equiv`, `decodeEncode`, and the equivalence of the loop space of the
circle with the integers.

## Library modules

| Module | Provides |
|--------|----------|
| `foundations` | `refl sym trans ap apd transport subst J J-refl rUnit lUnit rCancel lCancel symInvol funExt happly isProp isSet isGroupoid isProp->isSet isContr->isProp ua ua-beta equiv-fun iso-lemma iso-to-equiv iso-to-path`; with the kernel prelude `isContr fiber isEquiv Equiv idEquiv path-to-equiv` |
| `numbers` | `Nat zero suc add Int pos negsuc sucInt predInt sucPred predSuc sucEquiv sucPathInt` |
| `circle` | `S1 base loop OmegaS1 helix winding encode decode intLoop decodeSquare decodeEncode windingIntLoop OmegaS1-equiv-Int OmegaS1=Int` |
| `truncation` | `isOfHLevel Trunc inc squash isProp-Trunc Trunc-rec SetTrunc sinc squash0 isSet-SetTrunc` |
| `suspension` | `Susp north south merid Bool true false S1->SuspBool SuspBool->S1` |

Most library definitions take level arguments first: `(refl 0 Int (pos 1))`,
`(trans 0 S1 base base base loop loop)`.

## Engines

`hott-eval` is replaced before code generation on every engine, so a program
prints the same values under `eshkol-run -r`, an AOT binary and the bytecode
VM (`--profile hosted-vm`). See `tests/hott/` and `scripts/run_hott_gate.sh`.
