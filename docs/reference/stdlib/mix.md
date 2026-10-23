# `core.compiler.mix`

An optional, MIT-licensed compiler workbench for the versioned `mix-v1`
first-order language. Eshkol hosts a strict reference evaluator and an ordinary
quoted Mix specializer. Specializing that specializer actually constructs
compilers and a compiler-generator. No host compiler constructs those outputs.

```scheme
(require core.compiler.mix)
(define limits (mix-limits 20000000 4096 2000000 300000))
(define options (mix-options 256 512))
(define interpreter (mix-bytecode-interpreter))
(define residual (mix-specialize interpreter '((code . 768) (n . 1)) options limits))
(mix-result-status residual) ; normal
(mix-inputs residual)        ; (x)
(mix-result-value (mix-run (mix-result-value residual) '((x . 7)) limits)) ; 7
```

## Public operations

| Operation | Meaning |
|---|---|
| `mix-limits work depth allocations nodes` | Construct explicit execution limits. |
| `mix-options generalization polyvariance` | Configure the existing specializer policy; memoization is enabled. |
| `mix-run program environment limits` | Strict reference execution. |
| `mix-specialize program statics options limits` | Execute the quoted specializer on the supplied source and static inputs. |
| `mix-apply-compiler compiler target statics options limits` | Execute a generated compiler with the same static-input ABI. `target` supplies the separately declared source input contract. |
| `mix-compile source options limits` | Execute the combined specializer on the same-IR self-interpreter and source; normalization and cleanup execute inside Mix. Requires an admitted scalar result. |
| `mix-emit-eshkol result name` | Emit an admitted scalar specialization as ordinary Eshkol source forms. |
| `mix-source`, `mix-combined-source`, `mix-self-interpreter` | Return fresh quoted object programs. |
| `mix-bytecode-interpreter`, `mix-bytecode-interpreter-alt` | Return the two corpus interpreters. |

Results are vectors:

```text
#(mix-result-v1 status value expression-visits accounted-pairs inputs origin)
#(mix-emission-v1 status source-forms inputs)
```

Use `mix-result-status`, `mix-result-value`, `mix-result-work`,
`mix-result-allocations`, and `mix-inputs` for execution results. Only `normal`
publishes a value. Refusals publish no partial program. Input snapshots detach
the result from subsequent caller mutation and preserve pair sharing and
distinct equal cells. Inputs must remain immutable during a call. Treat the
published artifact as immutable as well.

`origin` records producer arguments, not a cryptographic attestation or proof.
Recipients must independently check source expectations and execute semantic
tests. In particular, the `target` supplied to `mix-apply-compiler` is a caller
contract; the library does not prove an arbitrary compiler implements it.

## Language and primitive contract

Programs have shape `(program ((name (parameters ...) body) ...) main)`.
Expressions are exact integers, `(quote datum)`, `(var symbol)`, sequential
`(let ((name expression) ...) body)`, `if`, `prim`, and named `call`. Functions
are first-order and closed over their parameters. Main free variables come
from a complete symbol-key alist; the first binding wins.

Values are exact integers, symbols, nil and immutable pairs. The port maps
the selected source's keyword tags to ordinary Scheme symbols; nil becomes
`()`. It does not admit arbitrary Common Lisp package identity. Both `()` and
the symbol `false` are false; numeric zero is true. Predicate results are the
symbols `true` and `false`. Arguments evaluate strictly from left to right.

The primitive inventory is `cons car cdr first second third fourth list append
reverse null pairp atomp symbolp integerp booleanp not truthy eq equal + - * = <
<= mod shr inline-ok apply-prim`. Selectors on nil return nil. Nil is a symbol
for `symbolp`. `mod` uses floor modulo and returns zero for a zero divisor.
`shr` is signed arithmetic right shift with its count reduced modulo 64.
Nested `apply-prim` stays within this closed inventory and has a dispatch-depth
guard. `inline-ok` retains the reference policy and is not user evaluation.

`eq` preserves pair identity and symbol identity. Comparing two integers with
`eq` refuses as `unsupported-identity`: the reference host leaves that identity
implementation-dependent. Use `=` for integer equality and `equal` for
structural equality. Pair mutation, effects, stochastic choice, floats, foreign
functions and arbitrary Eshkol source are outside this language.

## Bounds and refusal

Limits separately bound expression visits, evaluation depth (at most 4096),
accounted execution pair allocations and validation/output tree occurrences.
The data validator uses an explicit worklist. Fixed profile bounds also admit
at most 256 car-nesting levels, 2048 cells per list spine, 256 function
definitions and 64 arguments, parameters or sequential bindings per form.
Cycles consume a data bound and refuse. Long quoted name lists do not consume
the evaluation-depth budget.

Counters do not measure RSS, validation/snapshot temporary allocation, hash
table traversal, bignum work, compiler allocation or every host instruction.
The shell gate separately imposes time and Linux address-space limits. These
limits are not an allocation-free or real-time execution theorem.

Statuses distinguish `work-bound`, `depth-bound`, `allocation-bound`,
`data-bound`, `data-depth-bound`, `data-spine-bound`, invalid syntax/data/limits,
unbound inputs, primitive type/arity errors and unsupported identity. The scalar
compiler returns `unsupported-residual` when object normalization keeps its raw
fallback; it never publishes that fallback under the normalized input ABI.

## Source emission and qualification

The emitter admits exact integer parameters, sequential integer bindings,
integer-valued named functions with acyclic calls, conditionals and arithmetic/comparison
primitives. It prunes unreachable helpers, assigns fresh local identifiers and
checks every declared input at the generated entry. Inputs remain required
even when normalization removes their reads. General pair-valued compilers and
cogens and recursive residuals continue to execute through the bounded reference
evaluator. Reference fuel is not transported into emitted native code; the
emitter's acyclic-call admission excludes unbounded recursive native execution.
Host arithmetic costs and external process limits still apply.

Call discovery follows expression positions in the Mix grammar. A variable or
local binding named `call` is not a call, and a function named `quote` does not
hide calls in its arguments. Source identifiers receive fresh lexical names
independently of these spellings. Export names are lowercase ASCII identifiers;
syntax names and the generated wrapper's runtime names (`not`, `error`,
`modulo`, `quotient`, `expt`) refuse before emission to avoid capture.

The output is ordinary source: it can be compiled with JIT or standalone AOT
without linking a runtime JIT into the resulting executable. No new LLVM pass,
AD transform, compiler startup hook or persistent cache is installed.

Run `ESHKOL_BIN=/path/to/eshkol-run bash scripts/run_mix_gate.sh`. The gate
executes actual F1/F2/F3 production for two interpreters and seven bytecode
programs, comparing all dynamic results with a separately generated integer
oracle. It checks producer origins against fake-generator controls, bounded
refusals, exact arithmetic and input ownership. It then separately compiles
and executes fourteen emitted F3 residuals on JIT and AOT, and executes the
self-hosted cleanup example and erased-input guard. JIT caches are disabled;
the complete native assertion count is checked before execution, and a control
with one dropped assertion must refuse. Identifier regressions exercise source
bindings named `call`/`quote` and conflicting native export names. All
the source libraries come from the tested checkout. A CTest `mix_gate` invokes
the same script on Unix builds.

These are finite semantic and structural witnesses. General Jones optimality,
whole-compiler correctness, native throughput, VM/WASM parity and arbitrary
Scheme specialization are not claimed. BLC/SBLC retain their own evaluation
contracts and are not dependencies. The selected source donation is licensed
under MIT; no surrounding source repository or runtime service is required.
