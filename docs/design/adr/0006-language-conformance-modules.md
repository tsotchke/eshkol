# ADR-0006: Binding-resolved libraries and proper tail invocation

- Status: Proposed
- Date: 2026-07-09
- Owners: frontend, module loader, type system, native/VM backends
- Scope: R7RS-small language profile, Eshkol `require`/`provide`, AOT, ORC JIT/REPL, precompiled libraries, and proper tail calls

## Decision summary

Eshkol will stop representing a library import as source-file inclusion plus
new top-level `define` aliases. The frontend will retain `define-library` and
recursive import sets as declarations, resolve them against a versioned
`LibraryInterface`, and bind every identifier use to an opaque `BindingId`.
`only`, `except`, `rename`, and `prefix` will transform an interface map in
source order; they will not copy values or create new Scheme bindings. A
library's private bindings will exist only in its own resolved environment and
will receive collision-free physical symbols. Export status and LLVM linkage
will be consequences of resolution, not substitutes for it.

The same library resolver, interface artifacts, and initialization graph will
be used by AOT, JIT, the REPL, the type checker, the macro expander, and the VM.
R7RS libraries are strict immediately. Legacy `require`/`provide` files enter
strict mode when they contain `provide`; export-all remains a temporary,
diagnosed compatibility rule only for legacy files with no `provide`.

Proper tail calls will likewise become an explicit property of resolved Core
IR. One tail-context pass will replace the several syntax-shape walkers in the
backend. Self calls retain the in-function loop transform; compatible known
calls use LLVM `musttail` through a scalar internal ABI; every other tail call,
including dynamically dispatched closures and `apply`, uses a bounded-stack
tail-transfer dispatcher. An annotated tail call may never silently lower to
an ordinary stack-growing call.

This is one architectural change because both problem families have the same
root cause: semantic identity and context are currently erased too early and
then reconstructed from strings and incidental AST shapes in downstream code.

## Normative target and profiles

The normative module behavior is the corrected R7RS-small report's
[library syntax](https://standards.scheme.org/r7rs-html5/index.html#true-libraries),
[import-set semantics](https://standards.scheme.org/r7rs-html5/index.html#true-import-declarations),
and [proper-tail-recursion rules](https://standards.scheme.org/r7rs-html5/index.html#true-proper-tail-recursion).
In particular:

- library names contain identifiers and exact non-negative integers;
- library declarations include `export`, `import`, `begin`, `include`,
  `include-ci`, `include-library-declarations`, and declaration-level
  `cond-expand`;
- export renaming and arbitrarily nested import sets operate on bindings;
- an imported binding cannot be redefined or mutated in a program or library,
  while a REPL may install a later interactive shadow;
- imported syntax keeps its transformer and definition environment;
- the target of `apply` and `call/cc`, the consumer of `call-with-values`, and
  evaluation performed by `eval` preserve the specified tail continuation.

Two explicit language profiles prevent compatibility policy from weakening
the standard:

1. `eshkol` is the default dialect. It retains implicit Eshkol facilities and
   the legacy module adapter described below.
2. `r7rs-small` starts a program with an empty import environment, exposes the
   exact standard-library interfaces, rejects Eshkol-only syntax unless it is
   imported from an Eshkol library, and enables the `r7rs` feature only after
   the conformance gates in this ADR pass.

“Full R7RS module semantics” in this ADR means the complete R7RS library and
import behavior, including syntax exports and initialization. It does not mean
R7RS-large, a package/version solver, or a security sandbox.

## Current-state evidence

### Library identity and import semantics are erased in the parser

The current parser maps only `(scheme base)` specially, to the much broader
`stdlib` module, and otherwise joins name parts with dots
(`lib/frontend/parser.cpp:3584-3601`). Library-name parsers accept only symbol
tokens, so integer name components are rejected
(`lib/frontend/parser.cpp:3760-3791`, `lib/frontend/parser.cpp:3952-3969`).

`R7rsImportSpec` flattens a recursive import set into four unrelated fields:
one `only` vector, one accumulated `except` vector, a rename vector, and one
prefix string (`lib/frontend/parser.cpp:3604-3615`). The recursive parser then
overwrites `only`, appends exclusions and renames, and concatenates prefixes
(`lib/frontend/parser.cpp:3881-3948`). This loses the order-sensitive name set
on which an outer modifier must operate. For example, the names mentioned by
an outer `only` or `rename` are names *after* any inner renaming, not necessarily
the original export spellings.

The lowering is split between metadata for one narrow “prefix all” case and
synthetic `(define alias source)` forms. Prefix metadata is recorded only when
there is no `only` and no rename (`lib/frontend/parser.cpp:3656-3681`); other
rename/prefix cases become value aliases (`lib/frontend/parser.cpp:3685-3720`,
`lib/frontend/parser.cpp:4033-4061`). An `only` without a prefix and an
unprefixed `except` create no filtering metadata at all. The original module is
still required wholesale. A rename leaves the old spelling reachable, and a
`define` copies a value rather than naming the same imported binding. This is
incorrect for locations, syntax, duplicate-import checks, and imported-binding
immutability even when the happy-path procedure call appears to work.

`define-library` currently validates and then discards its library name,
rejects export rename specs, accepts only `export`/`import`/`begin`, and lowers
the entire declaration to an ordinary sequence
(`lib/frontend/parser.cpp:4080-4181`). Consequently the declared library is not
a namespace, compilation unit, or registry entry. The AST can carry only
module-name strings plus prefix/except side arrays, and `provide` can carry only
export-name strings (`inc/eshkol/eshkol.h:2213-2226`). There is no place for a
library identity, recursive import-set tree, binding origin, syntax export,
export rename, or source span for a modifier.

The existing modifier test checks only that desired aliases happen to call;
it never asserts that excluded or pre-rename names are absent
(`tests/modules/r7rs_import_modifiers_test.esk:1-31`). Its `except` case even
uses a name that the source library does not export, which R7RS requires to be
an error (`tests/modules/r7rs_import_modifiers_test.esk:7`).

### Visibility is lane-dependent and not lexical

The AOT driver flattens top-level sequences and physically prepends every
required module's ASTs to one compilation unit
(`exe/eshkol-run.cpp:2170-2185`, `exe/eshkol-run.cpp:3396-3412`,
`exe/eshkol-run.cpp:3634-3641`). Its symbol table treats a missing or empty
export set as export-all (`exe/eshkol-run.cpp:1094-1124`). A private-name
rewriter exists, but it finds definitions and recursively rewrites matching
strings (`exe/eshkol-run.cpp:3348-3391`); the active load path deliberately
disables it because `provide` is informational
(`exe/eshkol-run.cpp:3577-3598`). `provide` nodes are then removed before final
code generation (`exe/eshkol-run.cpp:3616-3629`). The positive visibility test
explicitly requires a non-provided helper to remain externally callable
(`tests/modules/visibility_fail_test.esk:11-29`,
`tests/modules/visibility_fail_test.esk:37-45`).

The REPL implements a different rule. It computes private names after parsing a
module (`lib/repl/repl_jit.cpp:2583-2627`), compiles the whole batch, and then
registers each bare spelling in a process-global private-name set
(`lib/repl/repl_jit.cpp:2664-2676`). Codegen rejects later lookups by testing
only that spelling (`lib/backend/llvm_codegen.cpp:6377-6391`,
`lib/backend/llvm_codegen.cpp:9487-9499`). The set records neither the owning
library nor the importing environment, so two libraries cannot independently
own the same private spelling. It is a temporal REPL deny-list, not module
scope.

The AOT and REPL also duplicate module-path algorithms and already differ in
search behavior (`exe/eshkol-run.cpp:2437-2545`,
`lib/repl/repl_jit.cpp:2868-2976`). A missing AOT module logs errors and then
continues processing (`exe/eshkol-run.cpp:3514-3527`). Precompiled modules are
reparsed from source to rediscover declarations and exports
(`exe/eshkol-run.cpp:3421-3450`), so an installed object is not a self-contained
library artifact. The type checker does not consume module interfaces; it
hard-codes a few known exports for selected module-name strings
(`lib/types/type_checker.cpp:869-899`, `lib/types/type_checker.cpp:1226-1228`).
LLVM codegen treats every module operation as a no-op because all semantics are
expected to have happened elsewhere (`lib/backend/llvm_codegen.cpp:10532-10537`).

### Tail context is reconstructed several times and incompletely

`TailCallCodegen` has one recursive self-call analysis. It knows several
surface constructs, but its default case returns “safe” without traversing
unknown operation children (`lib/backend/tail_call_codegen.cpp:253-342`,
`lib/backend/tail_call_codegen.cpp:540-560`). The main backend independently
implements `isInTailPosition`, `countAllRecursiveCalls`, `findTailCalls`, and
`collectMutualTailCallSites` (`lib/backend/llvm_codegen.cpp:23282-23392`,
`lib/backend/llvm_codegen.cpp:23394-23596`,
`lib/backend/llvm_codegen.cpp:23598-23682`). Those walkers recognize different
subsets. The mutual-call collector, for example, descends through `if`, the
last sequence expression, and let bodies, but not `cond`, `case`, `and`, `or`,
`when`, `unless`, `do`, multiple-value forms, or `case-lambda`
(`lib/backend/llvm_codegen.cpp:23608-23679`). Parser quirks such as representing
`if` as a call named `"if"` require repeated string special cases
(`lib/backend/tail_call_codegen.cpp:272-307`).

Known direct mutual tail calls are emitted as real `musttail` only on AArch64.
Other targets receive an optional `tail` hint because the 16-byte aggregate
return is not accepted for `musttail`; calls forwarding pointer arguments also
fall back to an ordinary call (`lib/backend/llvm_codegen.cpp:17739-17807`).
That leaves x86/arm32/riscv64 and higher-order closure-forwarding paths without
the R7RS constant-space guarantee.

In the checked-out snapshot, `codegenApply` delegates directly to
`CallApplyCodegen` (`lib/backend/llvm_codegen.cpp:34180-34185`), whose known
function and dynamic closure paths emit ordinary LLVM calls
(`lib/backend/call_apply_codegen.cpp:664-773`,
`lib/backend/call_apply_codegen.cpp:1012-1050`,
`lib/backend/call_apply_codegen.cpp:1129-1153`). The implementation also caps
dynamic apply dispatch at eight arguments and eight captures
(`inc/eshkol/backend/call_apply_codegen.h:46-50`).

ESH-0227's just-completed incoming change is accepted as baseline: a statically
spelled self `(apply f leading ... (list ...))` can reuse the existing loop
back-edge. This worktree predates that integration, as its task still describes
the pre-fix seam (`.swarm/tasks/ESH-0227.json:19-28`). The fix is a valuable
fast path, but it intentionally leaves a runtime-built final list or dynamic
callee on the ordinary apply path. R7RS requires the procedure passed to
`apply` to be invoked tail-wise regardless of whether the compiler can prove a
self target, so ESH-0227 is not the end-state architecture.

The same gap appears in other required tail-calling procedures. `call/cc`
calls its procedure through the general closure path and then branches to a
merge block (`lib/backend/llvm_codegen.cpp:20161-20175`). `call-with-values`
does the same for its consumer and supports at most eight produced values
(`lib/backend/llvm_codegen.cpp:20469-20525`,
`lib/backend/llvm_codegen.cpp:20538-20575`). Current continuation state stores
a pointer to a `jmp_buf` in the capturing native stack frame
(`inc/eshkol/eshkol.h:1385-1397`), and codegen allocates that buffer on the
stack (`lib/backend/llvm_codegen.cpp:20132-20156`). Unlimited-extent,
multi-shot continuations therefore also need an explicit continuation
representation before the `r7rs-small` profile can claim full conformance.

Finally, `cond-expand` currently advertises `r7rs` unconditionally and uses a
hard-coded feature predicate (`lib/frontend/parser.cpp:6736-6758`); its compound
feature parser skips unrecognized requirements rather than asking a library
registry (`lib/frontend/parser.cpp:6787-6822`). A conformance claim must be a
tested profile property, not a parser constant.

## Architectural invariants

The implementation must maintain all of the following:

1. **Binding identity precedes spelling.** A resolved use names a `BindingId`.
   Local, exported, renamed, and prefixed spellings are environment keys for
   the same binding, not distinct variables.
2. **Interfaces are closed sets.** An importer can resolve only entries in the
   provider's `LibraryInterface`. A private-name index may improve diagnostics
   but may never participate in successful lookup.
3. **Privacy is enforced before codegen.** LLVM linkage and symbol visibility
   reinforce the decision; they do not decide what source code can name.
4. **Import modifiers are compositional.** Each modifier transforms the exact
   result of its nested import set, with simultaneous rename semantics and
   validation at that step.
5. **One source of truth serves every lane.** AOT, JIT, REPL, VM, type checking,
   macro expansion, documentation, and precompiled-library loading consume the
   same interface and resolved graph.
6. **Library initialization is explicit.** Dependencies initialize first;
   each library instance executes its body in textual order at most once per
   program or REPL session.
7. **Tailness cannot be an optimization hint.** Every Core IR call marked
   `Tail` lowers to a constant-native-stack mechanism on every supported
   target, or compilation fails with an internal error.
8. **No caller-frame pointer crosses a tail transfer.** Arguments and escaping
   closure environments are copied or promoted into invocation-owned storage
   before the current activation is discarded.
9. **Compatibility is named.** Legacy open-module behavior is a profile or
   transition flag, never an accidental fallback inside strict R7RS resolution.

## Library architecture

### 1. Preserve declarations and recursive import sets

The parser will produce declaration IR instead of lowering library forms into
ordinary expressions:

```text
LibraryName  = [NamePart]
NamePart     = Identifier | ExactNonNegativeInteger

ImportSet    = Library(LibraryName)
             | Only(ImportSet, [Identifier])
             | Except(ImportSet, [Identifier])
             | Prefix(ImportSet, Identifier)
             | Rename(ImportSet, [(Identifier, Identifier)])

ExportSpec   = Direct(local-name)
             | Rename(local-name, external-name)

LibraryDecl  = Export([ExportSpec])
             | Import([ImportSet])
             | Begin([Form])
             | Include(paths, case-fold?)
             | IncludeLibraryDeclarations(paths)
             | CondExpand([DeclarationClause])
```

Every node retains a source span. Empty export/import lists are represented and
validated according to the grammar rather than rejected by a generic “at least
one symbol” helper. Included paths resolve relative to the file containing the
declaration, enter the dependency manifest, and are cycle-checked. `include-ci`
uses the reader's case-folding mode only for the included source. Declaration
`cond-expand` selects and splices declarations before interface construction;
`(library <name>)` queries the registry without instantiating the library.

A strict R7RS library source must declare exactly one `LibraryName` matching
the name by which it was resolved. Eshkol will document a deterministic search
mapping (`.sld` before `.esk`) but keep name identity separate from path
identity. Two different sources claiming the same `LibraryName` in one build
are an error. This replaces the current dot-joined string with an unambiguous,
length-delimited canonical encoding.

### 2. Resolve names to bindings

The resolver introduces these logical records; their concrete C++ ownership
can use indices/arenas rather than nested heap objects:

```text
LibraryId       { canonical_name, provider_id }
BindingId       { library_id, ordinal }
Binding         { id, declared_name, kind, mutability, type, definition_span }
ExportedBinding { binding_id, external_name, kind, type, link_symbol }
LibraryInterface {
  schema_version, library_id, abi_id, feature_fingerprint,
  exports: OrderedMap<Identifier, ExportedBinding>,
  syntax_payloads, initializer_symbol, dependency_interface_hashes
}
LibraryUnit {
  id, source_path, declarations, local_environment, body,
  interface, value_dependencies, syntax_dependencies
}
```

`Binding.kind` distinguishes value and syntax bindings while the environment
remains a single Scheme identifier map, so conflicting uses follow Scheme's
binding rules rather than separate language namespaces. AST identifier nodes become
`ResolvedRef { BindingId, use_span }` after expansion and name resolution.
Types, arities, mutability, and syntax transformers attach to `BindingId`, not
to raw strings. Diagnostics retain the source spelling and import path.

The macro expander must be parameterized by a library syntax environment.
Today it starts with one empty global scope and registers macros by string name
(`lib/frontend/macro_expander.cpp:16-22`,
`lib/frontend/macro_expander.cpp:52-80`). Under this ADR, an exported syntax
binding serializes its transformer plus the defining syntax environment (or a
reconstructible environment reference). Import rename/prefix changes only the
use-site name. Hygiene continues to refer to definition-site `BindingId`s.
Syntax dependencies are expanded before value code, matching the existing
two-pass intent for top-level macros
(`lib/frontend/macro_expander.cpp:90-123`) without leaking all macros into one
global scope.

### 3. Evaluate import-set algebra over an interface

Resolving `Library(name)` yields a fresh ordered view of the provider's export
map. Each enclosing modifier transforms the current view:

```text
only(S, names):
  require every name in S; retain exactly names

except(S, names):
  require every name in S; remove exactly names

prefix(S, p):
  map every (name -> binding) to (concat(p, name) -> binding)

rename(S, pairs):
  require each old name in S and each old name only once;
  simultaneously replace old keys with new keys;
  reject duplicate final keys
```

The transformations are recursive, not normalized into independent fields.
For a provider exporting `x` and `y`, composition must produce:

| Import set | Local map |
|---|---|
| `(prefix (rename (m) (x z)) p-)` | `p-z -> m:x`, `p-y -> m:y` |
| `(rename (prefix (m) p-) (p-x z))` | `z -> m:x`, `p-y -> m:y` |
| `(only (rename (m) (x z)) z)` | `z -> m:x` |
| `(except (prefix (m) p-) p-x)` | `p-y -> m:y` |

Merging import sets is also binding-aware. Importing the same local name more
than once is allowed only when every occurrence denotes the same `BindingId`;
different bindings are an ambiguity error with both import paths shown. No
operation above emits a `define`, LLVM global, wrapper, or runtime assignment.

### 4. Build and validate the export interface

After declaration splicing, imported syntax setup, macro expansion, and local
binding collection, every export spec resolves in the library environment:

- a direct export maps its external name to the resolved local or imported
  `BindingId`;
- an export rename maps the requested external name to that same `BindingId`;
- an unbound export, duplicate external name for different bindings, or export
  of an inaccessible binding is a compile-time error;
- re-exporting an imported binding is permitted and retains binding identity;
- a local binding absent from this map is private, including helper functions,
  variables, record bindings, and syntax definitions.

The compiler rejects `define` or `set!` targeting an imported binding in a
program or library. A lexical binding may shadow an import where R7RS permits
it. The REPL implements the report's exception by installing a new interactive
binding in an overlay environment; it never mutates or changes the interface of
the already-loaded library.

### 5. Separate source visibility from physical linkage

Every non-FFI Scheme binding receives a collision-free physical name derived
from its `LibraryId` and `BindingId`, for example
`__eshkol_L<stable-hash>_B<ordinal>`. Source spellings such as `free`, `helper`,
or two libraries' independent `state` bindings never become raw host symbols.
An `extern` or explicit `:export-symbol` declaration remains the only route to
a requested C ABI name.

Private functions and globals use LLVM internal linkage when emitted in the
same library object. Exported Scheme bindings use hidden external linkage so
other Eshkol library objects can link to their canonical symbols without
polluting the process ABI. Explicit foreign/public ABI exports get platform
default visibility through generated wrappers. Imported source names are
resolved directly to the provider's canonical symbol; prefix and rename do not
change linkage names.

This model also closes reflection back doors. `interaction-environment`,
`environment`, `eval`, documentation, and completion enumerate the active
`LibraryInterface` view, not the backend global symbol table. The current
interaction environment walks all global symbol strings
(`lib/backend/llvm_codegen.cpp:23218-23248`); that must be replaced before
privacy is considered complete. A diagnostic-only private-name index may say
“`helper` is private to `(m)`”, but it cannot return its `BindingId`.

Module privacy is encapsulation, not a security boundary: unsafe FFI, object
inspection, or native debuggers can still observe machine code and memory.

### 6. Use one library graph and one initialization protocol

The shared `LibraryResolver` performs this pipeline for all execution lanes:

```text
parse declarations
  -> resolve source/library identities and declaration cond-expand
  -> build syntax/value dependency graphs
  -> load dependency interfaces
  -> evaluate import-set maps
  -> expand syntax in the library environment
  -> collect local bindings and validate exports
  -> resolve every identifier to BindingId
  -> type-check and lower each LibraryUnit
  -> emit/load object + LibraryInterface + initializer
```

Missing libraries, identity mismatches, invalid import modifiers, and dependency
cycles are fatal diagnostics. Syntax-phase cycles are rejected with the full
cycle. Value-phase cycles are initially rejected as well; supporting a cyclic
initialization semantics is a separate proposal, not an accidental consequence
of an “already loaded” set.

Each library emits `__eshkol_init_<library-id>(Runtime*)` with an idempotence
guard. The program initializer invokes dependencies in deterministic
topological order, then runs each library's top-level commands in textual
order. Multiple imports, including separately filtered imports of one library,
refer to the same instantiation and locations. The REPL keeps the same graph
and instances for the session; explicit developer reload creates a new
generation and is outside R7RS semantics.

### 7. Make precompiled interfaces first-class artifacts

Every `.o`, `.bc`, or JIT cache entry for a library is accompanied by a
versioned `.eshkoli` interface artifact. It contains no executable private
implementation, but does contain the exact exports, binding kinds, type/arity
metadata, syntax payloads, initializer symbol, ABI/profile ID, source and
included-file digests, feature selection, and dependency-interface hashes.

Consumers load `.eshkoli` without reparsing provider source. A stale or
incompatible interface is a cache miss, never a best-effort load. The same
artifact drives external declarations, type checking, macro imports,
documentation, and REPL registration. This removes the current precompiled
path's dependency on installed `.esk` sources and prevents AOT/JIT interface
drift.

### 8. Adapt native `require`/`provide` deliberately

The Eshkol syntax is an adapter into the same model:

| Source form | Interface rule | Foreign visibility |
|---|---|---|
| `define-library` + `export` | Exact export map | Strict immediately |
| Legacy file with one or more `provide` forms | Union of provided bindings | Strict after migration gate |
| Legacy file with no `provide` | Implicit export-all adapter | Allowed only in `eshkol`; warning and manifest marker |
| Main program | No export interface unless building a library | Program environment only |
| `load` | Textual evaluation in the requested environment | Deliberately not a module boundary |

`require m` imports the complete interface of the native module `m`. It cannot
see private definitions. Multiple `provide` forms are order-independent and
their union is validated after expansion. A compatibility option
`--legacy-open-modules` temporarily reproduces informational `provide`; it is
for migration and cannot be enabled in `r7rs-small`.

Before strict native visibility becomes the default, the stdlib and downstream
corpus must be compiled in an audit mode that reports every cross-module use of
a non-provided binding. Those dependencies are either exported intentionally
or replaced by a proper internal/shared module. Silent private-name mangling of
the flattened AST is not an acceptable migration mechanism.

### 9. Provide exact standard-library interfaces

The library registry maps each standard `(scheme ...)` name to its own exact
interface rather than mapping `(scheme base)` to all of `stdlib`. At minimum it
contains the R7RS-small libraries `base`, `case-lambda`, `char`, `complex`,
`cxr`, `eval`, `file`, `inexact`, `lazy`, `load`, `process-context`, `read`,
`repl`, `time`, `write`, and `r5rs`. Implementations may share object code, but
their exported binding maps remain distinct. Eshkol extensions live under
reserved Eshkol interfaces and do not leak through `(scheme base)`.

## Tail-call and conformance architecture

### 1. Normalize first, annotate tail context once

After macro expansion and binding resolution, surface control forms lower to a
small Core IR with explicit `If`, `Sequence`, `Let`, `Call`, `Values`, and
handler/dynamic-context nodes. No control form is encoded as a call whose
callee happens to be the string `"if"`. A single inductive pass annotates calls
with `CallPosition::Tail` or `NonTail`, relative to a particular lambda.

The pass implements every R7RS tail context:

| Construct | Children inheriting tail context |
|---|---|
| `lambda`, `case-lambda` | Last expression of every body |
| `if` | Consequent and alternate |
| `cond`, `case` | Last expression of every selected clause |
| `cond`/`case` `=>` | Implied receiver invocation, not receiver expression |
| `and`, `or` | Last operand |
| `when`, `unless` | Last body expression |
| `let`, named `let`, `let*`, `letrec`, `letrec*` | Last body expression |
| `let-values`, `let*-values`, `let-syntax`, `letrec-syntax` | Last body expression |
| `begin` | Last expression |
| `do` | Last expression of its termination result sequence |

Eshkol forms such as `match` and `guard` state their tail-child rules in the
same Core IR visitor. Adding an expression node requires implementing a
`childrenWithContext` method; there is no permissive default. Debug builds fail
if an executable Core node has no policy. Self-recursion discovery, mutual-call
lowering, escape analysis, and diagnostics consume the annotation instead of
walking surface ASTs again.

Tail-calling library procedures are represented explicitly:

- `apply` evaluates the procedure and effective argument sequence, then invokes
  it with `Tail` relative to `apply`'s continuation;
- `call/cc` invokes its procedure with the continuation in `Tail` position;
- `call-with-values` invokes the producer normally and the consumer with the
  continuation of `call-with-values`;
- `eval` evaluates its input with the continuation of `eval`.

This also makes `cond`/`case` `=>`, arbitrary `define-values`, and multiple
values part of normal Core lowering rather than backend exceptions. Dynamic
argument and value counts are not capped at eight.

### 2. Use a hybrid lowering with a semantic fallback

Each annotated call chooses the first applicable lowering:

| Call shape | Lowering | Stack guarantee |
|---|---|---|
| Direct self call | Existing parameter-update branch to loop header | O(1) |
| Known direct callee, compatible internal signature | LLVM `musttail` + immediate `ret` | O(1) |
| Dynamic closure, differing arity, captures, dynamic `apply`, or target without legal `musttail` | Tail-transfer dispatcher | O(1) |
| Non-tail call | Specialized direct call or ordinary invocation driver | One frame per semantically active non-tail call |

ESH-0227's static self-`apply` recognition remains as the first-row
optimization. Correctness no longer depends on recognizing `(list ...)` in the
source: a runtime list and a computed callable take the third row.

The internal direct-call ABI returns the 16-byte tagged value as an LLVM
`i128`, with explicit pack/unpack helpers. This permits `musttail` on the
currently excluded native targets and addresses the tracked aggregate-return
gap (`.swarm/tasks/ESH-0171.json:1-21`). C runtime and FFI boundaries use
generated wrappers that marshal between the stable C
`eshkol_tagged_value_t` struct and the scalar internal form; the compiler does
not assume that two source-level ABI types are interchangeable merely because
they have the same size.

### 3. Define the tail-transfer protocol

Every first-class callable has a universal invoke entry in addition to any
specialized direct entry:

```text
invoke_entry(Invocation*, ClosureEnv*, Value* argv, size_t argc) -> void

Invocation {
  state: Returned | TailTransfer | Raised | ContinuationJump,
  result: Value,
  next_callable: Callable,
  argv: growable invocation-owned Value buffer,
  argc: size_t,
  continuation: ContinuationRef,
  dynamic_context: DynamicContextRef
}
```

An ordinary invocation creates or borrows an `Invocation` and runs:

```text
while true:
  current.invoke_entry(invocation, current.env, argv, argc)
  if state == Returned: return result
  if state == TailTransfer:
    current = next_callable
    (argv, argc) = transferred arguments
    continue
  dispatch raise/continuation states without growing the native call stack
```

A general tail call evaluates all operands left-to-right, copies tagged values
into the invocation-owned buffer, promotes an escaping closure environment if
needed, replaces `next_callable`, and returns `TailTransfer` to the loop. It
does not call the callee from the current native activation. Double-buffered
small inline storage handles common arities; a growable arena/heap buffer
handles arbitrary `apply` and multiple-value counts. The buffer belongs to the
invocation driver, so no pointer into the discarded caller frame reaches the
callee.

Before a self branch, `musttail`, or tail transfer, one shared tail epilogue
performs the existing per-iteration arena and exception-handler bookkeeping.
The current loop back-edge already needs bespoke scope release and handler
draining (`lib/backend/llvm_codegen.cpp:24434-24469`); centralizing that logic
prevents each new tail mechanism from forgetting a dynamic resource. A
non-tail dynamic construct such as `dynamic-wind` retains one driver frame, so
the entire tail chain in its thunk completes before its `after` action runs.

The existing zero-argument bounce trampoline proves the codebase already has a
looping call substrate, but it carries only a packed thunk pointer
(`lib/backend/tail_call_codegen.cpp:185-241`). It is replaced, not extended:
the tagged-pointer mask cannot safely encode arbitrary arguments, closure
environments, multiple values, exceptions, and continuations.

### 4. Make continuations explicit and heap-owned

The invocation's continuation is an immutable/persistent chain of semantic
continuation frames. A tail call reuses the same `ContinuationRef`; a non-tail
call pushes one frame. `call/cc` captures this chain plus the dynamic-wind mark
in heap/arena-promoted storage, so invoking it after the native function that
created it has returned is defined. Multi-shot invocation shares immutable
frames and copies mutable delivery state. Dynamic-wind transitions compare the
current and target dynamic-context chains and run `after` then `before` thunks
in the required order.

This replaces the stack-address `jmp_buf` as the semantic continuation. A
platform `setjmp` may remain an implementation detail for escaping to the
nearest dispatcher, but no captured Scheme continuation may retain its address.
The tail-call conformance gate includes continuations because R7RS counts calls
that can return through a later continuation as active calls.

### 5. Verify the guarantee in IR and at runtime

After lowering, a `ProperTailVerifier` examines every Core `Tail` call and its
LLVM lowering record. Accepted records are `SelfBranch`, `MustTail`, or
`TailTransfer`. `llvm.tail` hints and ordinary calls are rejected for semantic
tail sites. For `MustTail`, the verifier checks identical LLVM signatures,
calling convention, immediate return, and absence of caller-frame pointer
arguments. For transfers, it checks that argument storage is invocation-owned.

Optimized and unoptimized builds run the same verifier. LLVM optimization may
improve non-tail calls, but language correctness never depends on it.

## Conformance program

The `r7rs-small` feature is earned by executable evidence:

### Module gates

- Parse and round-trip every library declaration, integer name component,
  direct/renamed export, and recursively nested import-set shape.
- Table-test import algebra in both modifier orders, including unknown-name,
  duplicate rename, post-prefix rename, collision, and redundant-same-binding
  cases.
- Prove negative visibility: a private value and private macro are unbound to
  importers in AOT, JIT, REPL, and VM; two modules may use the same private
  spelling; each may still call its own helper.
- Prove binding identity: two aliases of one exported mutable location observe
  one location; redefining or mutating an import is rejected; re-exporting
  preserves identity.
- Prove syntax behavior: exported `syntax-rules` macros work through direct,
  renamed, prefixed, and re-exported imports with definition-site hygiene.
- Prove initialization: diamond imports initialize a stateful library once and
  in dependency/text order; missing libraries and cycles fail before codegen.
- Run from source, precompiled `.o`/`.bc` plus `.eshkoli`, persistent cache, and
  installed layouts without provider source; results and diagnostics agree.
- Verify exact `(scheme ...)` export lists and ensure Eshkol extensions do not
  appear in the R7RS profile.

### Tail-call gates

- For every row in the R7RS tail-context table, run self and mutual recursion
  for at least 10,000,000 transitions under a 512 KiB native stack in AOT and
  JIT. Include `cond`/`case` receiver clauses, multiple-value bodies, and `do`
  termination results.
- Repeat mutual tests on AArch64, x86-64, and every supported arm32/riscv64
  builder. RSS must remain bounded after warm-up.
- Exercise `apply` with a runtime-computed callable, leading arguments, a
  runtime-built list, differing arities, more than eight arguments, captures,
  and mutual recursion. None may depend on the static ESH-0227 fast path.
- Exercise a higher-order tail call forwarding a freshly created capturing
  closure, the current pointer-argument fallback case.
- Exercise `call-with-values` with zero, one, eight, and more than eight values;
  the consumer runs in the inherited continuation.
- Exercise `call/cc`'s procedure call, stored multi-shot continuations invoked
  after the creator returns, and `dynamic-wind` crossings.
- Pair every positive case with a non-tail control whose result is consumed, to
  catch an unsafe branch rewrite.
- Require the `ProperTailVerifier` report to contain zero ordinary lowerings for
  annotated tail sites.

### Broader language gates

The existing reference-differential design is the right oracle: it executes the
same portable program on Eshkol and an external strict R7RS implementation
(`docs/reports/REFERENCE_DIFFERENTIAL_REPORT.md:1-8`,
`docs/reports/REFERENCE_DIFFERENTIAL_REPORT.md:25-41`). Extend it so Eshkol and
the reference receive the same real import prologue; do not strip imports from
the Eshkol lane. Add library/private/syntax suites and all tail-procedure cases,
then require 100% agreement in both AOT and JIT.

In addition:

- generate the `(scheme ...)` interface inventory from the normative catalog
  and diff it against compiled `.eshkoli` artifacts;
- run the same portable corpus through the VM and make native/VM disagreement
  a failure, not a documentation caveat;
- keep unsupported Eshkol extensions outside the portable corpus;
- do not advertise `r7rs` from `cond-expand` until parser, library, tail,
  continuation, standard-interface, and differential gates are green.

## Delivery sequence

1. **Declaration IR and diagnostics.** Add `LibraryName`, recursive
   `ImportSet`, all declaration forms, source spans, and parser-only tests. Stop
   lowering strict R7RS forms to `require`/`provide`/`define` aliases.
2. **Interface resolver.** Add `LibraryUnit`, `BindingId`, import algebra,
   export validation, standard-library catalog, and fatal graph diagnostics.
   Resolve values and syntax before type checking.
3. **Unified loading and artifacts.** Move resolution out of
   `eshkol-run.cpp`/`repl_jit.cpp` into a shared frontend service; emit
   `.eshkoli` and init functions; make AOT, JIT, REPL, type checking, docs, and
   VM consume them.
4. **Strict visibility.** Enable strict R7RS immediately. Audit stdlib and
   downstream native modules, then make `provide` normative with a temporary
   `--legacy-open-modules` escape hatch. Add physical symbol isolation and
   reflection filtering.
5. **Core IR tail annotation.** Normalize control constructs, land the one
   context pass and verifier, and delete the duplicate backend walkers only
   after differential tests cover every old path. Integrate ESH-0227 as an
   optimization consuming the annotation.
6. **Portable direct ABI.** Introduce internal `i128` tagged returns and C/FFI
   wrappers; enable verified `musttail` on all target backends.
7. **General tail dispatcher and continuations.** Land invocation-owned argv,
   dynamic `apply`/multiple values, tail transfer, heap continuation chains,
   and dynamic-wind integration. Remove fixed argument/capture dispatch caps.
8. **Profile graduation.** Run the full module, proper-tail, continuation,
   standard-library, differential, VM-parity, SICP, AD, sanitizer, and cache
   suites. Only then expose `--language=r7rs-small` as conforming and set its
   `r7rs` feature.

Slices 1-3 can ship behind a resolver flag without changing legacy source.
Slice 4 is the compatibility boundary. Slices 5-7 may land incrementally, but
the verifier must distinguish “not yet supported” from “lowered properly”; no
partial target may claim the R7RS profile.

## Rejected alternatives

### Defer more alias generation to `process_requires`

Knowing the export list later fixes bare prefix enumeration but not semantics.
Synthetic `define` aliases still copy values, leak originals from the flattened
module, cannot import syntax hygienically, cannot enforce imported-binding
immutability, and duplicate logic across AOT and JIT. ESH-0014's current task
sketch proposes this incremental route (`.swarm/tasks/ESH-0014.json:20-32`);
this ADR supersedes that mechanism while retaining its user-visible acceptance
cases.

### Reactivate whole-AST private string rewriting

String rewriting confuses lexical shadowing, imported bindings, quoted data,
macros, generated identifiers, and two modules with the same spelling. It also
cannot produce a principled error at the import boundary. Binding resolution
provides privacy without guessing which strings are identifiers.

### Treat linker visibility as module privacy

Hidden/internal linkage prevents some external symbol lookup but does not stop
the compiler's flattened global environment from resolving a private source
name. Language visibility must be decided before LLVM. Linkage is defense in
depth and an optimization.

### Rely on LLVM `tail` or optimization passes

`tail` is a hint and already fails the constant-space requirement on supported
targets. Proper tail recursion is observable language behavior, including at
`-O0`; only a verified self branch, `musttail`, or dispatcher transfer is
acceptable.

### Use only the `i128` ABI change

Scalar returns unlock direct `musttail`, but differing arities, dynamic
callables, capture environments, runtime `apply` lists, `call-with-values`, and
continuation delivery still need a universal path. The dispatcher is the
semantic fallback; `i128` preserves the fast direct path.

### Trampoline every call

A universal interpreter-style trampoline would simplify the guarantee but
would impose argument-buffer and dispatch overhead on ordinary known non-tail
calls and obscure LLVM optimization. The hybrid keeps specialized direct calls
while making the slow path semantically complete.

### Turn strict `provide` on globally without migration

The current stdlib and downstream code were developed under informational
`provide`, and the AOT source documents that compatibility constraint
(`exe/eshkol-run.cpp:3577-3596`). Immediate breakage would encourage export-all
workarounds. Audit diagnostics, exact interface fixes, and a time-bounded
compatibility flag make strictness adoptable without weakening R7RS libraries.

## Consequences

Positive consequences:

- rename/prefix/only/except compose correctly for values and syntax;
- private internals, duplicate-name diagnostics, and imported-binding rules are
  deterministic and identical in every execution lane;
- two libraries can safely use the same source names and cannot collide with C
  runtime symbols;
- precompiled libraries become self-describing and cache invalidation follows
  interface dependencies;
- tail-call correctness covers dynamic higher-order Scheme, not only syntactic
  self recursion on one target;
- the `r7rs` feature becomes an evidence-backed profile claim.

Costs and risks:

- binding-aware AST/Core IR and syntax interfaces are a substantial frontend
  change;
- serialized macro transformers and interface schema evolution require strict
  versioning;
- stdlib/downstream module audits will reveal undeclared dependencies;
- the internal scalar ABI touches every native call boundary and needs
  cross-target ABI tests;
- invocation-owned buffers and heap continuations add runtime machinery to
  dynamic tail paths;
- debugger and stack-trace tooling must reconstruct logical tail transitions,
  because native frames intentionally disappear;
- cycles are diagnosed rather than tolerated until a separate cyclic-library
  initialization design is accepted.

## Non-goals

- package download, semantic-version selection, lockfiles, or multiple versions
  of one `LibraryName` in a program;
- R7RS-large libraries;
- treating private bindings as a security capability;
- changing textual `load` into a module import;
- guaranteeing tail-call optimization for calls that are not in an R7RS tail
  context;
- specifying REPL hot-reload generation semantics beyond keeping standard
  imports and already-instantiated libraries correct.

## Acceptance decision

This ADR is implemented only when all of the following are true:

1. strict R7RS libraries never lower import modifiers to Scheme `define`s;
2. every successful identifier lookup after expansion carries a `BindingId`;
3. private value and syntax bindings are unreachable outside their library in
   AOT, JIT, REPL, and VM;
4. source and precompiled libraries expose byte-for-byte equivalent versioned
   interfaces and initialization behavior;
5. all R7RS tail contexts and required tail-calling procedures pass the bounded
   stack gates on every supported native target;
6. no annotated tail site lowers to an ordinary call;
7. stored multi-shot continuations do not retain native stack addresses;
8. the external R7RS differential and native/VM parity gates are green; and
9. only then does `cond-expand` report the `r7rs` feature for the
   `r7rs-small` profile.
