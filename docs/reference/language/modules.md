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

### Note — `define-library` and `import` in the same file

`import` resolves from the **filesystem**, not from a `define-library` written
earlier in the same file. Put a `define-library` in its own file and `import` it
from another. (If you write both in one file, the `begin` body still executes and
defines its names at top level, but the `import` line will report the module as not
found on the path.)

## `load`

```
(load "path.esk")
```
Reads and evaluates a file inline in the current top-level environment — no
export/import boundary. Handy for scripts and REPL-style composition.

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
