# Modules

Eshkol supports two complementary module styles: a lightweight `provide`/`require`
pair, and R7RS `define-library`/`import`. Both resolve module names to source
files on a search path. `load` includes a file inline.

The compiler reserves every module shipped in its standard library. The build
records the installed relative paths and their SHA-256 digests in the stdlib
identity. For example, `core.list.transform` and `./core/list/transform.esk`
select the same installed module. The resolver checks the installed source
against its recorded digest. An unavailable installed module is an error.
Precompiled artifacts must carry the compiler's stdlib identity.

For project modules the resolver searches, in order,

1. the directory containing the referring file,
2. the current working directory,
3. the configured project paths in `$ESHKOL_PATH`, including `-I` entries,
4. `$ESHKOL_LIB_DIR` and its `lib` subdirectory, an explicit workspace
   module root, and the module trees under explicitly configured
   `$ESHKOL_SYSTEM_PREFIXES`.

Environment directories can supply additional project modules. Reserved
source modules come from the compiler's installation. The browser VM serves
its supported modules from its embedded image.

Module references must be relative. An absolute reference or a reference
with a `..` component raises a condition naming the requested module.
A drive-qualified spelling is rooted on every host.
For example, `(guard (e (else (error-object-message e))) (load "../data.esk"))`
returns the condition's message. A relative path may include `.` components
and directory names containing dots. The interactive `:load` command uses
the same resolver. A command-line entry file may be named by its absolute path.

## `provide` / `require`

```
(provide name …)     ; in the providing file: export these names
(require module)      ; in the consuming file: load module and import its provides
```
When a module has a `provide` form, only the listed names are visible to
importers; definitions omitted from it remain private on native and VM engines.
If no `provide` form is present, all definitions retain the legacy importable
surface.
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

### Built-in libraries

Some R7RS library names have no source file to find because Eshkol provides
them itself. Those names are mapped through **one** table,
[`inc/eshkol/builtin_libraries.h`](../../../inc/eshkol/builtin_libraries.h),
consulted by the native front end's `join_r7rs_library_name()` and by the
bytecode VM's `vm_library_name_from_datum()` — so neither engine can drift from
the other's idea of which libraries exist, and adding one is a single row.

| Library name | Provided by |
|---|---|
| `(scheme base)` | the built-in standard library |

Any other library name is resolved as a source file like any other module; a
name that resolves to nothing is refused rather than silently ignored.

Because the mapping happens where the library-name datum is joined, **every**
R7RS import modifier reaches it — `only`, `except`, `prefix` and `rename` — on
both engines:

```scheme
(import (scheme base))
(import (only (scheme base) car))
(import (except (scheme base) vector-fill!))
(display (list (car (list 1 2)) (cdr (list 1 2)))) (newline)
```
```
(1 (2))
```

Identical under `eshkol-run -r`, under AOT and under the bytecode VM; pinned by
`tests/vm_parity/corpus/82_scheme_base_builtin_import.esk` and
`tests/parser/r7rs_import_modifier_test.esk`.

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

1. reserved names from the installed stdlib,
2. project libraries this compilation unit already defined with `define-library`,
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
and ESKB compilation. A project load searches the referring file's directory, followed by
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
