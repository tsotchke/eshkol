# ADR-0015: Static callee binding identity

**Status:** Accepted — implemented in v1.3.5-evolve (`inc/eshkol/backend/static_callee_binding.h`)
**Ledger:** LE-28
**Scope:** Native LLVM code generation, JIT and AOT.

## Context

The backend keeps `<name>_func` aliases so calls, `apply`, list higher-order
operations and differentiation can call an LLVM function directly. A name is
only a valid shortcut while it still denotes the binding that created the
alias. A later `set!`, top-level redefinition, sibling binding or parameter can
make the runtime value a different procedure. Before LE-28, those paths could
silently call the original lambda. Dynamic `remove` also treated predicate
results as integer bits and rebuilt list elements without preserving their
tagged type.

## Decision

Static callee aliases are binding facts. One shared abstraction records the
alias and the LLVM storage that owns it. A reassigned binding gets no alias; a
same-name runtime binding hides an alias unless its storage is the recorded
owner. Top-level reassignment and redefinition facts are computed before code
generation from the compiler's existing lexical mutation analysis. This uses
the existing source binding names and storage locations; it creates no second
AST identifier namespace.

Every static callee fast path applies the same facts. When a name can change,
the backend evaluates its current value and dispatches through the closure ABI.
Dynamic `remove` tests the tagged predicate result with Scheme truthiness,
compares non-procedure items with the operation's `eq?`, `eqv?` or `equal?`
function, and copies retained elements with their original tags.

The mutation scan treats an `extern` declaration as signature metadata, not a
call expression. Compiler assurance compares the duplicated public and
implementation `EshkolLLVMCodeGen` member layouts across their real translation
units so a drift cannot silently corrupt LLVM state.

## Consequences

Calls through a mutable binding take the runtime closure path. Unmodified
bindings retain direct-call resolution. The shared owner fact also rejects a
stale alias after a sibling `let` or parameter shadows its name. These
constraints trade a fast path only where static resolution would violate the
program's current binding.

## Verification

The JIT and AOT regression gates cover reassigned differentiands and callees,
redefinition, capture and shadowing, `apply`, `map`, `vector-map`, `reduce`,
dynamic predicate removal, tagged inexact list elements, and `eq?`/`eqv?`/
`equal?` item removal. See `.icc/ledger/entries/LE-28.yaml` for the defect and
measured evidence.
