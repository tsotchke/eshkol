# Eshkol Module System

**Status:** Production (v1.1-accelerate)
**Primary implementation:** `exe/eshkol-run.cpp`
**Related files:** `lib/backend/llvm_codegen.cpp`, `lib/frontend/parser.cpp`, `lib/backend/binding_codegen.cpp`

---

## 1. Overview

Eshkol's module system is a compile-time, file-based dependency resolution mechanism. Modules are ordinary `.esk` source files that declare exported bindings via `provide` forms and consume other modules via `require` forms. There is no runtime module registry; the compiler resolves, loads, and prepends all transitive dependencies into a single compilation unit before code generation begins.

This design has three significant consequences:

1. **No dynamic linking of user modules.** All user code is statically compiled into a single binary unless the `--shared-lib` flag is used.
2. **Topological ordering is enforced by the compiler.** Sub-modules are always prepended before the requiring module, guaranteeing that all definitions are available at the point of use.
3. **The standard library (`stdlib.o`) is a special case.** It is pre-compiled to an object file and linked externally; the compiler emits only `extern` declarations for its symbols rather than re-compiling the source.

---

## 2. Module Syntax

### 2.1 Declaring Exports: `provide`

A module file declares its public interface using one or more `provide` forms at the top level:

```scheme
;; lib/core/list/transform.esk
(provide map filter fold-left fold-right)

(define (map f lst)
  (if (null? lst)
      '()
      (cons (f (car lst)) (map f (cdr lst)))))
```

`provide` is a declaration-level directive processed by the compiler. It does not generate any runtime code. Its purpose is to make the linker-visible symbol name predictable and to support tooling such as the LSP server.

### 2.2 Consuming Modules: `require`

A module or top-level program consumes another module with `require`:

```scheme
(require core.list.transform)
(require core.functional.compose)

(define result (map (compose add1 square) '(1 2 3 4 5)))
```

The argument to `require` is a dot-separated module path. The compiler resolves this path to a filesystem location, loads the source, recursively resolves its own dependencies, and prepends the resulting ASTs.

Multiple `require` forms are allowed and are processed in declaration order.

---

## 3. Path Resolution Algorithm

### 3.1 Dot Notation to Filesystem Path

Given a module identifier such as `core.list.transform`, the compiler performs the following transformation:

1. Replace every `.` with the platform directory separator `/`.
2. Append the `.esk` file extension.
3. Prepend each candidate search directory in turn.

So `core.list.transform` resolves to the candidate path `<search_dir>/core/list/transform.esk`.

The function responsible for this translation is inside `process_requires` in `exe/eshkol-run.cpp` at approximately line 1986.

### 3.2 Search Order

The compiler searches for a module file in the following order, stopping at the first match:

1. Each directory listed in the `ESHKOL_PATH` environment variable (colon-separated on POSIX, semicolon-separated on Windows).
2. The `lib/` directory relative to the compiler executable (`g_lib_dir`).
3. System-level paths (platform-dependent, typically `/usr/lib/eshkol/` or `/usr/local/lib/eshkol/`).

`g_lib_dir` is determined at startup from the path of the `eshkol-run` executable itself, making the compiler relocatable without requiring environment configuration for the standard library.

### 3.3 Example Resolution

```
require core.list.transform
  → candidate: $ESHKOL_PATH/core/list/transform.esk    (checked first)
  → candidate: <exe>/../lib/core/list/transform.esk     (g_lib_dir, typical match)
  → candidate: /usr/local/lib/eshkol/core/list/transform.esk
```

---

## 4. Cycle Detection

### 4.1 The `imported_files` Set

Circular imports are detected using a static set of canonical (fully-resolved, symlink-expanded) file paths:

```cpp
// exe/eshkol-run.cpp, lines 60-64
static std::set<std::string> imported_files;
```

The function `load_file_asts` (line 1182) performs the following check before loading any file:

1. Resolve the file path to its canonical form (via `std::filesystem::canonical` or equivalent).
2. Check whether the canonical path already exists in `imported_files`.
3. If it does, return immediately — this file has already been loaded or is currently being loaded (the latter case catches true cycles).
4. If it does not, insert the canonical path into `imported_files`, then parse the file.

Because the canonical path is inserted **before** recursive processing of that file's own `require` forms, any cycle back to this file will find it already in `imported_files` and terminate cleanly.

### 4.2 Why Canonical Paths

Using canonical paths rather than raw string paths prevents false negatives from path aliasing. Two different strings such as `./lib/../lib/core/list/transform.esk` and `lib/core/list/transform.esk` both resolve to the same canonical path and are correctly identified as the same module.

### 4.3 `collect_all_submodules` Must Not Pollute `imported_files`

There is a discovery-only function, `collect_all_submodules` (line 1543), that recursively walks the stdlib source tree to enumerate all sub-modules. Crucially, this function parses source files only to find nested `require` statements — it does **not** call `load_file_asts`, and therefore does **not** insert paths into `imported_files`.

If it did insert paths, those stdlib modules would appear already-imported when a user program later requires them, causing their definitions to be silently omitted from the compilation unit.

```cpp
// exe/eshkol-run.cpp, ~line 1543
// Discovery-only: parses source to find sub-requires, but never calls
// load_file_asts, preserving imported_files integrity.
void collect_all_submodules(
    const std::string& module_name,
    std::set<std::string>& precompiled_modules,
    const std::string& lib_dir);
```

---

## 5. Precompiled Standard Library

### 5.1 Motivation

The Eshkol standard library comprises dozens of source files. Re-compiling all of them for every user program would add hundreds of milliseconds to each build. Instead, the stdlib is compiled once to a single object file (`stdlib.o`) using the `--shared-lib` flag, then linked into every user binary.

### 5.2 The `--shared-lib` Flag

Passing `--shared-lib` (or `-s`) to the compiler sets `shared_lib=1` and `compile_only=1`. The compiler invokes `eshkol_generate_llvm_ir_library()` instead of the normal program-generation path. This path applies **LinkOnceODRLinkage** to all emitted symbols (see Section 6).

The stdlib is built as:

```sh
eshkol-run lib/stdlib.esk --shared-lib -o build/stdlib.o
```

### 5.3 Marking Modules as Precompiled

At compiler startup, before processing any user file, the compiler calls:

```cpp
// exe/eshkol-run.cpp, ~line 2508
collect_all_submodules("stdlib", precompiled_modules, g_lib_dir);
```

This recursively walks `lib/stdlib.esk` and all transitively required sub-modules, adding each module name to the static set:

```cpp
static std::set<std::string> precompiled_modules;
```

### 5.4 Handling Precompiled Modules in `process_requires`

When `process_requires` encounters a `ESHKOL_REQUIRE_OP` node, it first checks:

```cpp
// exe/eshkol-run.cpp, ~line 1990
if (precompiled_modules.count(module_name)) {
    // Emit extern declarations only — do not load source
    emit_extern_declarations(module_name);
    return;
}
```

If the module is precompiled, the compiler emits only `declare` (extern) forms in the LLVM IR, relying on the linker to resolve the actual definitions from `stdlib.o`. The source file is never opened, parsed, or added to `imported_files`.

If the module is not precompiled, the normal load-and-prepend path is taken.

---

## 6. LinkOnceODRLinkage and User Override Semantics

### 6.1 The Problem

When a user program does:

```scheme
(require stdlib)
(define (gradient-descent ...) ...)  ; user's custom implementation
```

Both `stdlib.o` and the user's object file would contain a definition for the symbol `_gradient-descent_sexpr`. Without special handling, the linker would report a duplicate symbol error and fail.

### 6.2 LinkOnceODRLinkage as the Solution

All symbols emitted in library mode (`--shared-lib`) use `llvm::GlobalValue::LinkOnceODRLinkage`. The semantics of this linkage are:

> "If multiple translation units define this symbol, keep exactly one copy. Prefer a definition with `ExternalLinkage` (the user's definition) over a `LinkOnceODR` definition (the stdlib's definition)."

This means the user's `(define (gradient-descent ...))`, which compiles to `ExternalLinkage`, silently wins over the stdlib's `LinkOnceODR` version. No linker error. No silent data corruption. The user gets exactly what they defined.

### 6.3 Affected Symbol Sites

LinkOnceODRLinkage is applied to three categories of symbols in library mode:

**Function definitions** — in `createFunctionDeclaration`, approximately line 2654 of `lib/backend/llvm_codegen.cpp`:

```cpp
if (shared_lib_mode) {
    fn->setLinkage(GlobalValue::LinkOnceODRLinkage);
}
```

**Homoiconic metadata globals** (`_sexpr` symbols) — 12 separate creation sites throughout `llvm_codegen.cpp`. Each site checks `shared_lib_mode` before setting linkage.

**Global variable definitions** — in `lib/backend/binding_codegen.cpp`, line 334:

```cpp
gvar->setLinkage(shared_lib_mode
    ? GlobalValue::LinkOnceODRLinkage
    : GlobalValue::ExternalLinkage);
```

### 6.4 Interaction with Parallel Worker Functions

The same `LinkOnceODRLinkage` mechanism is used for functions generated for parallel constructs (`parallel-map`, `pfor`, etc.) to prevent duplicate symbols when these appear in both stdlib code and user programs that include the stdlib.

---

## 7. Standard Library Organization

The stdlib source tree under `lib/` has the following structure. All modules under `lib/` are transitively discovered by `collect_all_submodules` and treated as precompiled when `stdlib.o` is present.

```
lib/
  stdlib.esk                     -- top-level aggregator; requires all sub-modules
  core/
    list/
      transform.esk              -- map, filter, for-each
      query.esk                  -- member, assoc, find
      sort.esk                   -- sort, merge-sort
      higher_order.esk           -- fold-left, fold-right, reduce
      compound.esk               -- append, flatten, zip
      generate.esk               -- iota, make-list, list-tabulate
      search.esk                 -- any, every, list-index
      convert.esk                -- list->vector, vector->list, list->string
    functional/
      compose.esk                -- compose, pipe
      curry.esk                  -- curry, partial
      flip.esk                   -- flip, swap-args
    control/
      trampoline.esk             -- trampoline, thunk helpers for deep recursion
    operators/
      arithmetic.esk             -- generic +, -, *, / with numeric tower dispatch
      compare.esk                -- generic <, >, <=, >=, = across types
    logic/
      predicates.esk             -- null?, pair?, procedure?, etc.
      types.esk                  -- type-of, type predicates
      boolean.esk                -- and, or, not, xor
    strings.esk                  -- string-split, string-join, string-trim, etc.
    json.esk                     -- json-parse, json-stringify
    io.esk                       -- read-line, write-line, file I/O utilities
    data/
      csv.esk                    -- csv-parse, csv-write
      base64.esk                 -- base64-encode, base64-decode
  math/
    constants.esk                -- pi, e, phi, euler-gamma
    special.esk                  -- gamma, beta, erf, bessel functions
    ode.esk                      -- runge-kutta-4, adaptive-ode
    statistics.esk               -- mean, variance, std-dev, percentile
  random/
    random.esk                   -- random, random-seed!, mt19937
  signal/
    fft.esk                      -- fft, ifft, power-spectrum
    filters.esk                  -- butterworth, fir-filter, moving-average
  web/
    http.esk                     -- http-get, http-post (libcurl binding)
  ml/                            -- machine learning primitives
  quantum/                       -- quantum circuit simulation
```

`lib/stdlib.esk` is the top-level aggregator. Its sole content is a sequence of `require` forms for the sub-modules the user is most likely to want without explicit targeting:

```scheme
;; lib/stdlib.esk
(require core.list.transform)
(require core.list.higher_order)
(require core.functional.compose)
(require core.functional.curry)
;; ... (all sub-modules)
```

---

## 8. Code Examples

### 8.1 Basic Module Require

```scheme
;; my_program.esk
(require core.list.transform)
(require core.list.higher_order)

(define numbers '(1 2 3 4 5 6 7 8 9 10))

(define evens (filter even? numbers))
(define sum   (fold-left + 0 evens))

(display sum)  ;; => 30
```

### 8.2 Targeting a Specific Sub-Module

Rather than requiring all of stdlib, a program can require only what it needs, reducing compilation time and binary size:

```scheme
;; numeric_analysis.esk
(require math.special)
(require math.ode)
(require random.random)

;; Use Runge-Kutta on a system using Bessel functions
(define solution
  (runge-kutta-4
    (lambda (t y) (list (bessel-j0 t) (bessel-j1 t)))
    0.0 '(1.0 0.0) 0.01 100))
```

### 8.3 User Override of a Stdlib Function

LinkOnceODRLinkage allows the user to replace any stdlib function:

```scheme
;; custom_ml.esk
(require stdlib)

;; Override stdlib gradient-descent with a custom Adam optimizer
(define (gradient-descent loss-fn params learning-rate)
  ;; Adam optimizer implementation
  (let loop ((params params) (m 0.0) (v 0.0) (t 1))
    (let* ((grad    (numerical-gradient loss-fn params))
           (m-new   (+ (* 0.9 m) (* 0.1 grad)))
           (v-new   (+ (* 0.999 v) (* 0.001 (* grad grad))))
           (m-hat   (/ m-new (- 1.0 (expt 0.9 t))))
           (v-hat   (/ v-new (- 1.0 (expt 0.999 t))))
           (update  (/ (* learning-rate m-hat) (+ (sqrt v-hat) 1e-8))))
      (if (< (abs update) 1e-7)
          params
          (loop (- params update) m-new v-new (+ t 1))))))
```

The compiler emits `ExternalLinkage` for the user's `gradient-descent`. At link time the linker discards the `LinkOnceODR` definition from `stdlib.o` in favour of the user's.

### 8.4 Building a User Library

A user can build their own pre-compiled library with the same mechanism:

```sh
# Compile user library to object file
eshkol-run my_utils.esk --shared-lib -o my_utils.o

# Link against both stdlib and user library
eshkol-run main.esk -o main my_utils.o
```

### 8.5 Inspecting Module Loading

The `--dump-ast` flag prints the AST after all `require` forms have been resolved and sub-module ASTs prepended, making it straightforward to verify the topological ordering:

```sh
eshkol-run my_program.esk --dump-ast 2>&1 | head -80
```

---

## 9. Process Flow Summary

The following pseudocode summarises the full module loading pipeline executed by `process_requires` and `load_file_asts`:

```
startup:
  collect_all_submodules("stdlib", precompiled_modules, g_lib_dir)
  # precompiled_modules now contains every stdlib sub-module name

compile(user_file):
  asts = load_file_asts(user_file)
  process_requires(asts, result_asts)
  codegen(result_asts)

load_file_asts(path):
  canonical = canonical_path(path)
  if canonical in imported_files: return []   # already loaded or cycle
  imported_files.add(canonical)
  return parse(read(path))

process_requires(asts, result):
  for each ast in asts:
    if ast.op == ESHKOL_REQUIRE_OP:
      module_name = ast.arg
      if module_name in precompiled_modules:
        emit_extern_declarations(module_name)  # no source load
      else:
        path = resolve(module_name)            # dots → slashes + .esk
        sub_asts = load_file_asts(path)
        process_requires(sub_asts, result)     # depth-first, prepend
    else:
      result.append(ast)
```

This guarantees that every definition appears in `result` before any use of it, regardless of how deeply nested the dependency graph is.

---

## 10. Implementation References

| Concept | File | Approx. Line |
|---|---|---|
| `imported_files` declaration | `exe/eshkol-run.cpp` | 60 |
| `precompiled_modules` declaration | `exe/eshkol-run.cpp` | 61 |
| `load_file_asts` | `exe/eshkol-run.cpp` | 1182 |
| `collect_all_submodules` | `exe/eshkol-run.cpp` | 1543 |
| `process_requires` | `exe/eshkol-run.cpp` | 1986 |
| `--shared-lib` flag handling | `exe/eshkol-run.cpp` | 2251 |
| Stdlib precompilation detection | `exe/eshkol-run.cpp` | 2508 |
| `LinkOnceODRLinkage` for functions | `lib/backend/llvm_codegen.cpp` | ~2654 |
| `LinkOnceODRLinkage` for global vars | `lib/backend/binding_codegen.cpp` | 334 |
| `ESHKOL_REQUIRE_OP` parser node | `lib/frontend/parser.cpp` | — |

---

## 11. Edge Cases and Known Constraints

**Circular imports terminate cleanly** but silently omit the second occurrence. If module A requires B and B requires A, the second encounter of A (or B) during DFS is silently skipped. This is intentional but can mask real design problems; the compiler does not currently emit a warning for detected cycles.

**`ESHKOL_PATH` directories are not recursively scanned** for module discovery. Only explicit `require` forms drive the discovery process. A module that exists on the filesystem but is never transitively required is never compiled.

**Module names are flat strings, not hierarchical objects.** `core.list.transform` is simply the string `"core.list.transform"` in `precompiled_modules`. There is no namespace object or module-level environment at runtime.

**`provide` is not enforced at compile time.** A module can `define` a symbol without `provide`-ing it, and other modules can still use it if they know the name. `provide` is advisory for tooling; the compiler does not generate linkage restrictions based on it.

**Relative requires are not supported.** All module paths are resolved relative to the search path, never relative to the requiring file's location. This avoids ambiguity in deeply nested module trees.

---

## 12. See Also

- `docs/breakdown/README.md` — Project-wide architecture overview
- `ESHKOL_V1_LANGUAGE_REFERENCE.md` — Language-level `require`/`provide` syntax
- `docs/ESHKOL_V1_ARCHITECTURE.md` — Compiler pipeline overview
- `lib/backend/llvm_codegen.cpp` — Symbol linkage and code generation
- `lib/backend/binding_codegen.cpp` — Global variable definition and linkage
- `exe/eshkol-run.cpp` — Driver: CLI parsing, module loading, compilation orchestration
