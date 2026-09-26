---
kind: explanation
status: current
owner-area: language
since: v1.4.0
sources:
  - docs/design/adr/0006-language-conformance-modules.md
  - inc/eshkol/frontend/semantic_identity.h
  - lib/core/module_visibility.cpp
  - lib/frontend/parser.cpp
  - lib/backend/llvm_codegen.cpp
  - lib/backend/binding_codegen.cpp
  - lib/backend/system_codegen.cpp
  - lib/repl/repl_jit.cpp
  - lib/backend/eshkol_vm.c
  - lib/backend/vm_compiler.c
  - lib/backend/vm_native.c
  - exe/eshkol-run.cpp
  - docs/BUILD_INTEGRATION.md
---
# ADR-0030: One program-entry rule and Eshkol-owned physical names

**Status:** Accepted (design, 2026-09-26); implementation planned for v1.4
**Issue:** #724
**Implements:** ADR-0006 section 5 (collision-free physical names), which was
specified but never built.
**Scope:** Every engine that runs Eshkol (native AOT and executables, the
cached and in-process JIT behind `eshkol-run -r`, shared libraries,
WebAssembly, the bytecode VM, the REPL), the precompiled standard library,
and the C boundary.

## Context

Issue #724: `(define (main a b) ...)` stops the native compiler with an LLVM
verifier error. It is the visible end of two design gaps.

### Gap 1: programs have no single entry rule

Each engine decides for itself what a program is and what `main` means.

| Program | Native AOT / cached `-r` | In-process JIT | VM |
|---|---|---|---|
| top-level forms plus `(define (main) ... 7)` | runs only `main`; top-level forms and library init are skipped; exit 7 | refuses the program | runs only the top-level forms; `main` is never called; exit 0 |
| `(define (main args) ...)` or any arity but zero | verifier error | verifier error | defined, never called |
| `main` returns `#f`, a string, a float, `'()` | exit 0, 168, 154, 0 (raw bits truncated) | - | - |
| `(command-line)` | the path of an internal cache binary | `()` | the source path and arguments |
| `emergency-exit` | exits | exits | undefined |

`eshkol-run -r` has no way to pass arguments to the script at all.

The native wrapper that handles a user `main` is a second code path beside
the normal one, and it disagrees with it on what runs.

### Gap 2: Eshkol names and C names share one namespace

Native codegen names a user function's object symbol with its raw Scheme
identifier, with public linkage. A definition therefore can collide with or
replace a C or runtime symbol, and `main` has to be renamed to `scheme_main`
as a special case. Variables already avoid this through a mangling helper,
but that helper is duplicated in two files, and there are three separate
builders for module-private names. ADR-0006 section 5 already decided the
fix: physical names derived from the binding's identity, with only `extern`
imports and `:export-symbol` exports carrying C names. It was not
implemented.

## Decision

Three rules, each owned by exactly one facility.

### Rule 1: one physical-name facility

Every object-level name derived from an Eshkol binding comes from one
function: `eshkol::physicalName(const BindingId&)` in `lib/frontend`, built on
the `BindingId` / `LibraryId` types in `semantic_identity.h`.

- **Encoding.** `_ESK` followed by the library-path components and the
  identifier, each length-prefixed, with every byte outside `[A-Za-z0-9_]`
  written as `_xHH` (lower-case hex). Length prefixes make the encoding
  injective: different bindings never share a physical name, within a program
  or across libraries. A top-level program's bindings use the program's
  `LibraryId`.
- **Derived names use the same facility.** Lambdas, captured variables,
  homoiconic `_sexpr` records, REPL versions (`__rv<N>`), REPL forwarding and
  storage slots, module-private bindings, shared-library implementation
  bodies and library-init chunks all derive from a binding's physical name
  plus a kind suffix. The facility appends the suffix; no caller concatenates
  strings.
- **Retired.** The raw-name function path, `userGlobalStorageName` and its
  copy in `binding_codegen.cpp`, the three module-private name builders, and
  every name check against `main`, `scheme_main` or `_start` outside the
  entry stub.
- **Debug information** keeps the Scheme identifier as the display name and
  the physical name as the linkage name, as it already does for defines.
- **Diagnostics** show Scheme identifiers. A demangler in the same facility
  maps physical names back, and the backtrace and profiler paths use it.

### Rule 2: the export list is the C contract

Internal definitions and the C-visible interface are separate things. Every
definition gets a physical name (Rule 1); only a declared interface reaches C,
and that interface is an explicit, versioned, validated contract.

- **The contract is the export list.** For a library, it is the
  `define-library` / `provide` export list. For a `--shared-lib` build of a
  plain source file, it is the file's public top-level definitions, as today.
  Nothing else is visible to C.
- **Exported C names do not change.** Each exported binding keeps the C name
  it has today: a C-ABI entry under its Scheme identifier that forwards to
  the physical definition. Existing C hosts need no migration.
  `:export-symbol <name>` chooses a different C name, and `:export-symbol` on
  a program definition is how a program exposes a binding to C or to a
  WebAssembly host.
- **Every exported C name is checked at build time.** A name that is
  reserved for the runtime (`eshkol_*`, `__eshkol_*`, `_ESK*`, `main`,
  `_start`), or that equals a symbol defined by anything the artifact links
  (the runtime, the standard library, libc, libm), is a build error naming
  both definitions. An export can therefore never interpose a C or runtime
  function.
- **Each shared library emits an interface manifest** beside the binary:
  every exported binding's C name, physical name, arity and, where known,
  types, with an interface version hash that changes whenever the contract
  changes. The C header is generated from the manifest, so the two cannot
  drift. The same manifest can later describe the interface for calls that
  cross process or machine boundaries.
- `extern` declarations keep importing C functions by their C names.
- **The standard library** is compiled with physical names. Programs declare
  stdlib bindings by their physical names and link to `stdlib.o` or the JIT
  by those names. The JIT discovers stdlib bindings from a binding manifest
  emitted beside `stdlib.bc`, rather than assuming symbol equals identifier.
  Precompiled-artifact cache keys include the physical-name scheme version.

### Rule 3: one program-entry plan

A *program* is what `eshkol-run -r`, an AOT executable build or a WebAssembly
application build compiles. Libraries, `--shared-lib` builds and REPL
sessions are not programs and never get an entry call.

- **One decision.** A C function `eshkol_program_entry_plan` in `lib/core`,
  used by both the native front end and the VM compiler, inspects the
  program's top-level definitions and returns one of: no `main`; `main` with
  zero parameters; `main` with one required parameter; or an error. The
  error says: "`main` is the program entry point; it takes no parameters, or
  one parameter that receives `(command-line)`", and shows that form. It
  covers any other arity, required-plus-rest and rest-only parameter lists
  (`(define (main . args) ...)`), optional parameters, a multi-arity `main`,
  and a `main` that is not a procedure. Every accepted shape has exactly one
  meaning, known at compile time. Two argument conventions exist elsewhere
  (SRFI-22's `(main args)` receives the full command line; some Schemes
  spread arguments into `(main . args)` without the program name), and
  accepting both would make the program name depend on how `main` is
  written. Rejecting is also the reversible choice: a form can be admitted
  later without breaking anyone.
- **One lowering.** Both compilers append the planned call to the program as
  ordinary code after every top-level form:

  ```scheme
  <every top-level form, in source order>
  (exit (main))                  ; plan: zero parameters
  (exit (main (command-line)))   ; plan: one parameter
  ```

  `main` is an ordinary binding with an ordinary physical name, and the call
  is an ordinary call. No engine treats the name specially.
- **Exit status** comes from the standard `exit` procedure: an exact integer
  is the status, `#t` or an unspecified value is success, `#f` is failure,
  anything else raises, as `exit` does today. The raw-bit truncation goes.
- **One native entry stub.** The runtime's C `main` initialises the runtime,
  records argc/argv, runs library init, then runs the lowered program. The
  second wrapper for a user `main` is deleted, so top-level forms and library
  init always run.
- **In-process JIT** evaluates the same lowered program, so a program with
  `main` runs instead of being refused.
- **VM.** The compiled program chunk gets a reserved internal name that is
  not an Eshkol identifier, so a user `main` no longer collides with it, and
  the planned call is compiled like any other call.
- **WebAssembly** exports the entry stub and the program's `:export-symbol`
  bindings. The website marks the procedure it re-invokes on navigation with
  `:export-symbol`, and the site checks move with it.

### Rule 4: `command-line` and exit procedures mean the same thing everywhere

- `(command-line)` is the program as the user named it (the source path
  given to `eshkol-run`, or an executable's `argv[0]`) followed by the
  program's arguments, on every engine.
- `eshkol-run -r prog.esk -- a b` passes `a b` to the program. The cached
  `-r` path hands the child the user-facing program name and the arguments,
  and the in-process JIT passes the real argc/argv to the program instead of
  overwriting them.
- The VM gains `emergency-exit`. On both engines it uses the same
  status mapping as `exit` and skips exit handlers and `dynamic-wind`.

## Consequences

- #724 is fixed at the root, together with every symptom in the tables
  above: the verifier error, skipped top-level forms and library init,
  engine-dependent exit statuses, `command-line`, script arguments, and user
  definitions colliding with C or runtime symbols.
- The special cases for `main`, `scheme_main` and `_start`, the duplicate
  mangling helpers and the private-name builders are removed. Each concern
  has one owner.
- **No migration for C hosts.** Shared-library exports keep their C names.
  Hosts gain a generated header and a versioned interface manifest, and a
  build now fails where an export would have interposed a C or runtime
  function.
- **Programs that defined `main`** now also run their top-level forms and
  library init on native, and exit with the `exit` mapping of `main`'s
  result. That behaviour was the defect. The 34 test programs that define
  `main` are re-run under the new rule, and any that change behaviour are
  reviewed as findings, not rebaselined.
- Precompiled artifacts (`stdlib.o`, JIT object and AOT caches) are rebuilt
  once, keyed by the scheme version.

## Implementation lanes

Each lane has an ICC brief, a ledger entry where it fixes observable
behaviour, and parity tests, and merges only after the ICC exam. They land in
this order, since each builds on the last.

1. **Physical-name facility.** `physicalName`, the demangler and unit tests.
   Variables move onto it with no behaviour change; the duplicates are
   deleted.
2. **Functions and derived names.** User and nested functions, lambdas,
   captures, `_sexpr` records, REPL names, module-private names. Stdlib
   binding manifest and JIT discovery.
3. **The C contract.** Export-list C entries forwarding to physical
   definitions, `:export-symbol` naming, the build-time clash check against
   reserved names and linked libraries, the interface manifest with its
   version hash, and header generation from it.
4. **Entry plan.** `eshkol_program_entry_plan`, the native lowering and the
   single entry stub, the in-process JIT, the VM chunk name and lowering,
   WebAssembly exports and the site.
5. **Command line and exit.** `--` for `-r`, program name and arguments on
   the cached and in-process paths, VM `emergency-exit`.
6. **Gates and docs.** The verification below, a "Program structure and
   entry point" language-reference page, and updates to
   `docs/BUILD_INTEGRATION.md` and the release process.

## Verification

- **Ledger.** Entries for the engine divergence in entry behaviour and exit
  status, for `command-line`, and for C-symbol collisions, each with its
  repro.
- **Parity corpus**, run on native AOT, cached and in-process `-r`,
  WebAssembly and the VM, with identical output and exit status:
  - no `main`;
  - `main` with zero parameters and each kind of return value;
  - `main` with one parameter, run with and without script arguments;
  - top-level forms before and after `main`;
  - a library that defines `main`, which must never be called;
  - `emergency-exit`.
- **Negative corpus.** Every rejected `main` shape gives the entry diagnostic
  on every engine, and no program reaches the LLVM verifier with a
  mismatched entry call.
- **Symbol-table gate** over `stdlib.o`, an AOT object, an executable and a
  shared library: every symbol is a physical name, a runtime name, an
  `extern` import or an `:export-symbol` alias. No symbol equals a Scheme
  identifier by accident.
- **Collision corpus.** User definitions named after C library and runtime
  functions, including `main`, run correctly on every engine and match the
  VM.
- **Contract checks.** Existing shared-library harnesses keep working
  unchanged; exporting a reserved or linked-library name fails the build;
  the manifest's version hash changes exactly when the export list or a
  signature changes; the generated header matches the manifest.
- **ICC.** This ADR is registered; `trace-callers` and `impact` cover every
  site that builds an object name from a binding; `duplicate-implementations`
  shows one physical-name facility and one entry plan.

## Alternatives rejected

- **Fixing only the verifier error** leaves the engines disagreeing and the
  collision class open.
- **Treating a non-zero-arity `main` as an ordinary function** makes one name
  mean different things by arity and leaves top-level forms skipped.
- **Making the VM imitate the native wrapper** doubles the special case.
- **A deny-list of names users may not define** is incomplete by
  construction: any C or runtime symbol can collide.
- **A second mangling scheme beside ADR-0006's** would give the same concept
  two owners.
- **Only `:export-symbol` bindings reach C** would force every existing C host
  to change for no gain in safety: the build-time clash check already stops
  an export from interposing a C or runtime function.

## Decisions taken with the owner

1. **C exports:** the export list is the contract; exported names keep their
   current C names, are validated against reserved and linked symbols, and
   are described by a versioned interface manifest. No migration for hosts.
2. **`(define (main . args) ...)`** is rejected with the entry diagnostic.
3. **The one-parameter `main`** receives the full `(command-line)`, program
   name first.
