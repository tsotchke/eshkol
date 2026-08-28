# Modules

Eshkol supports two complementary module styles: a lightweight `provide`/`require`
pair, and R7RS `define-library`/`import`. Both resolve module names to source
files on a search path. `load` includes a file inline.

Search path for module resolution:
1. the directory of the file doing the `require`/`import`/`load`,
2. the project root (the current working directory),
3. directories passed with `-I DIR`, then entries in the `$ESHKOL_PATH`
   environment variable (`-I` directories are appended to it),
4. the compiler's bundled `lib/` directory — itself located beside the running
   compiler, and only then in the system prefixes.

A path you name explicitly therefore overrides a module that ships with the
compiler, while a project's own sources always come first. See
[environment variables](../runtime/environment-variables.md#resolution-precedence)
for the same rule applied to the native artifacts.

The search runs top to bottom and the first match wins, so the two rules that
matter most are independent: a program that lives beside its helpers keeps
finding them wherever you run it from (step 1), and a project-rooted spelling
like `(load "tests/fixtures/dep.esk")` keeps working from the project root
(step 2).

This order is a property of the **language**, not of how you happen to run the
program. `eshkol-run` chooses among several execution engines — a persistent
JIT run cache, an in-process JIT (used whenever the cache is bypassed: `-d`,
`--dump-ast`, `--dump-ir`, several input files, or an active
`$ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR`), an AOT compile, and the VM — and every
one of them resolves `require`, `import` and `load` through the same resolver,
`eshkol::platform::resolve_module_source_path`. A program cannot load different
files because of which engine ran it. The `load_path_engine_parity_test` CTest
gate holds every engine to identical output on the same source, with a
same-named decoy planted in the working directory so a lane that regressed to
cwd-rooted resolution fails on a different answer rather than a missing file.

## `provide` / `require`

```
(provide name …)     ; in the providing file: export these names
(require module)      ; in the consuming file: load module and import its provides
```
A module name maps to a file path: `core.list.transform` → `core/list/transform.esk`,
`greet` → `greet.esk`.

`greet.esk`:
```scheme
(provide greet)
(define (greet who) (string-append "Hello, " who))
```
`main.esk`:
```scheme
(require greet)
(display (greet "World")) (newline)
```
Run with the directory on the search path:
```sh
eshkol-run -r main.esk -I .
```
```
Hello, World
```

The standard library is required the same way:
```scheme
(require stdlib)                 ; whole standard library
(require core.list.transform)   ; a single stdlib submodule (e.g. filter, map)
(require core.capabilities)      ; capability policy API
```

## `define-library` / `export` / `import`

```
(define-library (name …)
  (export name …)
  (import lib …)     ; optional
  (begin definition …))
```
`import` resolves a library name to a file the same way `require` does. A library
`(geo)` is looked up as `geo.esk`; `(my math)` is looked up as `my/math.esk`.

`geo.esk`:
```scheme
(define-library (geo)
  (export area)
  (begin (define (area r) (* 3 r r))))
```
`main.esk` (in the same directory):
```scheme
(import (geo))
(display (area 2)) (newline)
```
```
12
```

### `define-library` and `import` in the same file

A library defined in the file being compiled is importable by the forms written
below it, with no filesystem search (R7RS-small 5.6.1). Resolution order is:

1. libraries this compilation unit already defined with `define-library`,
2. precompiled stdlib modules,
3. the module search path above.

```scheme
(define-library (geo)
  (export area)
  (begin (define (area r) (* 3 r r))))

(import (geo))
(display (area 2)) (newline)
```
```
12
```

A library is established by its **whole** `define-library` form, so it cannot
satisfy an `import` written above it — that is a forward reference, and the
compiler says so and names the line the library is defined on:

```
Module 'geo' not found: its define-library form is at line 7, below this import
  A library must be defined before it is imported (R7RS-small 5.6.1); move the
  define-library above the import, or put the library in its own file.
```

`only`, `except`, `rename` and `prefix` work over a same-unit library exactly as
they do over a file-backed one. The two module spellings share one namespace:
`(require geo)` resolves a same-unit `(define-library (geo) …)` too.

Same-unit libraries do not yet hide their non-exported names — the library body
is spliced into the unit's top level, so every name it defines is visible.
Strict library isolation is tracked as module-privacy work.

## `load`

```
(load "path.esk")
```
Reads and evaluates a file inline in the current top-level environment — no
export/import boundary. Handy for scripts and REPL-style composition.
The path literal is resolved by the same resolver on native JIT/AOT, VM source,
and ESKB execution: the requiring file's directory is preferred, followed by
the project search path. Nested loads therefore resolve relative to the file
that contains the nested `load`.

`lib.esk`:
```scheme
(provide double)
(define (double x) (* 2 x))
```
`main.esk`:
```scheme
(load "lib.esk")
(display (double 21)) (newline)
```
```
42
```
