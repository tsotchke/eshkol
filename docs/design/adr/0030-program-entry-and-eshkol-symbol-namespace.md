---
kind: explanation
status: current
owner-area: language
since: v1.4.0
sources:
  - lib/frontend/parser.cpp
  - lib/backend/llvm_codegen.cpp
  - lib/backend/vm_compiler.c
  - exe/eshkol-run.cpp
  - docs/BUILD_INTEGRATION.md
---
# ADR-0030: One program-entry rule and an Eshkol-owned symbol namespace

**Status:** Proposed
**Issue:** #724
**Scope:** Every engine that runs an Eshkol program (native JIT, AOT objects and
executables, shared libraries, WebAssembly, the bytecode VM, the REPL), the
precompiled standard library, and C interoperation.

## Context

Issue #724 reports that `(define (main a b) ...)` fails with an internal LLVM
verifier error. The cause is two independent special cases that contradict each
other:

1. **Native renames `main`.** Codegen renames a user function called `main` to
   `scheme_main` and has the C entry wrapper call it with no arguments, whatever
   its arity. The wrapper also uses its return value as the exit status and does
   not run top-level expressions: `(define (main) (display "hi") 0)` followed by
   a top-level `(display "top")` prints only `hi`.
2. **The VM does not.** It treats `main` as an ordinary function and never calls
   it: the same program prints only `top`, and `(define (main) 7)` exits 0
   where native exits 7.

So one program has two meanings, and the native one crashes for any arity but
zero.

The rename exists because of a deeper defect: native codegen emits every
user definition under its raw Scheme name with public linkage. A Scheme
identifier and a C symbol share one namespace, so a user definition can
replace a runtime or C library function. Measured on 1.3.5 (native; the VM is
correct in every case):

| User defines | Native result |
|---|---|
| `main` of any arity but zero | LLVM verifier error |
| `malloc` | program prints nothing; the runtime's allocator was replaced |
| `memcpy` | SIGSEGV |
| `printf`, `exit`, `strlen`, `abs`, `write`, `free` | correct today by luck of link order |

`main` is one symptom. Any fix that special-cases more names leaves the class
open.

## Decision

Two rules, each implemented once.

### Rule 1: Eshkol definitions live in an Eshkol-owned symbol namespace

Every top-level definition compiled from Eshkol source gets a mangled object
symbol that no C or runtime symbol can equal. Scheme names never appear raw in
an object file's symbol table.

- **Encoding.** `_ESK` + the library path components + the identifier, each
  length-prefixed and with every byte outside `[A-Za-z0-9_]` escaped as `_xHH`
  (lower-case hex). Length prefixes make the encoding injective, so distinct
  Scheme names never collide with each other either, including across
  libraries. Example: `list->vector` in `(eshkol core)` becomes
  `_ESK6eshkol4core12list_x2d_x3evector`.
- **One facility.** A single `eshkolSymbolName(library, identifier)` function
  in the backend produces every definition symbol. Codegen, the REPL's
  versioned hot-reload names (`<mangled>__rv<N>`), debug information
  (`DISubprogram` name = the Scheme identifier, linkage name = the mangled
  symbol), the WebAssembly dead-strip, the homoiconic display registry and the
  API/ABI scanners all call it. No other code builds a symbol from an
  identifier.
- **C interoperation is explicit.** A definition is visible to C under a chosen
  name only through the existing `:export-symbol [<name>]` modifier, which
  emits a C-linkage alias for the mangled definition. `extern` declarations
  keep importing C functions by their C names; they are the other direction and
  are unaffected. Shared libraries export exactly their `:export-symbol`
  definitions to C, and the generated library header lists them.
- **The standard library follows the same rule.** `stdlib.o` is compiled with
  the same mangling, so user programs link to it by mangled names and a user
  definition can never shadow or replace a standard-library symbol at link
  time. Lexical shadowing in Scheme is unchanged.

### Rule 2: One program-entry rule for every engine

The shared front end lowers a *program* (a `--run`, AOT or WebAssembly
application compile, never a library, `--shared-lib` build or REPL session)
into one explicit form before any backend sees it:

```scheme
<every top-level form, in source order>
(exit (main))                  ; when main is defined with zero parameters
(exit (main (command-line)))   ; when main is defined with one parameter
```

- Top-level forms always run, in order, on every engine.
- `main` is an ordinary function. Its call is explicit in the lowered program,
  so every backend executes the same call and none of them treats the name
  specially.
- The exit status goes through the standard `exit` procedure: an exact integer
  is the status, `#t` or an unspecified value is success, `#f` is failure. There
  is no second exit-status path in the C entry stub.
- `(command-line)` returns the same list on every engine: the program as the
  user named it (the source path given to `eshkol-run`, or the executable's
  `argv[0]`), then the program's own arguments. Measured on 1.3.5 it does not:
  the VM returns `("prog.esk")`, while the native JIT returns the path of an
  internal JIT cache file. The runtime receives the user-facing program name
  from the driver, so the one-parameter `main` sees identical input everywhere.
- `main` with any other arity, including a rest parameter, is a compile-time
  diagnostic at its definition: "`main` is the program entry point and takes
  zero parameters, or one parameter receiving the command-line list". A program
  that wants a two-argument helper names it something else.
- The only C-level `main` is the runtime's entry stub. It initializes the
  runtime and runs the lowered program. With Rule 1 there is no user `main`
  symbol for it to collide with, so the rename and every `main` / `scheme_main`
  special case in codegen are deleted.

## Consequences

- **Fixes #724 completely**: the verifier error, the native/VM divergence in
  exit status, top-level execution and `(command-line)`, and the whole class of user definitions
  replacing runtime or C symbols.
- **Removes special cases**: the `scheme_main` rename, the name checks in the
  WebAssembly dead-strip and the entry wrapper's implicit call all go.
- **Breaking change for C consumers of shared libraries** that call Eshkol
  functions by their Scheme names. They must mark those definitions
  `:export-symbol`. The release notes and `docs/BUILD_INTEGRATION.md` state
  this, and the shared-library build lists every public definition without
  `:export-symbol` as a note, so the migration is mechanical.
- **Precompiled artifacts** (`stdlib.o`, the JIT object cache, AOT caches)
  change their symbol names, so their cache keys include the mangling version.
- **Programs relying on native's old behaviour** (top-level forms skipped when
  `main` is defined) change behaviour. That behaviour was the defect.

## Verification

- A silent-wrong ledger entry for the native/VM divergence and one for runtime
  symbol replacement, each with the repro above.
- Parity-corpus programs run on native JIT, AOT, WebAssembly and the VM with
  identical output and exit status: `main` of arity zero with an exit code;
  arity one with command-line arguments; top-level forms before and after
  `main`; no `main`; a library defining `main`, which must not be called.
- `(command-line)` parity: identical lists on every engine for a program run
  with and without arguments.
- Negative tests: arities two and rest give the diagnostic on every engine; no
  program reaches the LLVM verifier with a mismatched entry call.
- A symbol-table gate over `stdlib.o`, an AOT object and a shared library: no
  symbol equals a Scheme identifier except `:export-symbol` aliases and
  `extern` imports. User definitions of `malloc`, `memcpy`, `printf`, `exit`,
  `free` and `main` run correctly on native and match the VM.
- ICC: this ADR registered, `trace-callers` and `impact` over every site that
  builds a symbol from an identifier, and `duplicate-implementations` showing a
  single mangling facility.

## Alternatives rejected

- **Only fixing the native crash** leaves the silent divergence and the symbol
  class open.
- **Treating a non-zero-arity `main` as an ordinary function** makes one name
  mean different things by arity, and still leaves top-level forms skipped.
- **Making the VM copy native's implicit call** doubles the special case.
- **A deny-list of reserved names** (`main`, `malloc`, ...) is incomplete by
  construction: any C or runtime symbol can collide.
