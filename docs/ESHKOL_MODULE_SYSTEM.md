# Eshkol Module System

## Overview

The Eshkol module system is designed to be **gradual** and **minimal**, following the language's philosophy. Module names are inferred from file paths, and files use simple top-level directives rather than wrapping forms.

---

## Syntax

### Primary Syntax (Eshkol Style)

```scheme
;;; lib/data/json.esk
;;; Module: data.json (inferred from path)

(require core.strings)                    ; Import dependency
(provide json-parse json-stringify)       ; Export public API

(define (json-parse str) ...)
(define (json-stringify val) ...)
(define (_internal-helper x) ...)         ; Private (underscore convention)
```

### Alternative Syntax (Scheme Compatibility)

```scheme
;;; Supported for R7RS compatibility
(define-library (data json)
  (import (core strings))
  (export json-parse json-stringify)
  (begin
    (define (json-parse str) ...)
    (define (json-stringify val) ...)))
```

Both syntaxes compile to the same internal representation.

---

## Directives

### `(require module-name ...)`

Import dependencies. Multiple modules can be listed.

```scheme
(require core.strings)              ; Import single module
(require data.json web.http)        ; Import multiple modules

;; Advanced forms (Scheme-compatible)
(require (only data.json json-parse json-get))     ; Selective import
(require (prefix net.tcp tcp:))                    ; Prefixed import
(require (rename data.csv (csv-parse parse-csv)))  ; Rename on import
```

### `(provide name ...)`

Declare the module's public API. Only listed names are exported.

```scheme
(provide json-parse json-stringify json-get)
```

If no `(provide ...)` is present:
- For library files (`lib/`): Nothing is exported (fully private)
- For scripts: No exports needed (just runs)

### Privacy Convention

Functions prefixed with `_` are considered private by convention:

```scheme
(provide json-parse)           ; Exported

(define (json-parse str)       ; Public
  (_skip-whitespace str 0))

(define (_skip-whitespace s i) ; Private (not in provide)
  ...)
```

---

## Module Resolution

### Path Resolution Order

1. **Current directory**: `./data/json.esk`
2. **Library path**: `$ESHKOL_LIB/data/json.esk` (default: `lib/`)
3. **User paths**: `$ESHKOL_PATH` entries (colon-separated)

### Module Name to File Path

| Module Name | File Path |
|-------------|-----------|
| `data.json` | `lib/data/json.esk` |
| `core.strings` | `lib/core/strings.esk` |
| `ext.ml.nn` | `lib/ext/ml/nn.esk` |
| `my-app.server` | `./my-app/server.esk` |

### Special Cases

- `stdlib` → `lib/stdlib.esk` (core library, auto-loaded)
- `math` → `lib/math.esk` (legacy, prefer `math.*` modules)

---

## Compilation Process

### Phase 1: Dependency Scan

```
main.esk
  └── (require data.json)
        └── (require core.strings)
  └── (require web.http)
        └── (require net.tcp)
        └── (require web.url)
              └── (require core.strings)  ; Already loaded
```

### Phase 2: Build Dependency Graph

```
                 main.esk
                /        \
          data.json    web.http
              |       /        \
       core.strings  net.tcp   web.url
                         |         |
                    core.strings (dedup)
```

### Phase 3: Topological Sort

Compile order: `core.strings → data.json → net.tcp → web.url → web.http → main.esk`

### Phase 4: Macro Expansion

For each file in order:
1. Load macro definitions from required modules
2. Expand macros in current file
3. Cache expanded AST

### Phase 5: Compilation

For each expanded AST:
1. Compile to LLVM IR
2. Cache compiled module

### Phase 6: Link

Combine all modules into final executable.

---

## Macro Handling

Macros are automatically detected by the compiler.

```scheme
;;; lib/macros/control.esk
(provide when unless)

(define-syntax when
  (syntax-rules ()
    ((when test body ...)
     (if test (begin body ...)))))

(define-syntax unless
  (syntax-rules ()
    ((unless test body ...)
     (if (not test) (begin body ...)))))
```

```scheme
;;; app.esk
(require macros.control)

(when (> x 0)        ; Macro expanded at parse time
  (display "positive"))
```

The compiler:
1. Detects `define-syntax` forms during parsing
2. Loads macros before expanding importing modules
3. Maintains proper hygiene across module boundaries

---

## Module Cache

Modules are cached to prevent duplicate loading:

```c
typedef struct eshkol_module {
    const char* name;              // "data.json"
    const char* path;              // "lib/data/json.esk"
    eshkol_ast_t* ast;            // Parsed AST
    eshkol_ast_t* expanded_ast;   // After macro expansion
    symbol_table_t* exports;      // Exported symbols
    macro_table_t* macros;        // Exported macros
    llvm::Module* ir;             // Compiled LLVM IR
    bool fully_compiled;          // Compilation complete?
} eshkol_module_t;
```

---

## Implementation Requirements

### Parser Changes

1. Recognize `require` as a top-level form
2. Recognize `provide` as a top-level form
3. Support `define-library` for Scheme compatibility
4. Track which forms are directives vs. code

### AST Changes

```c
typedef enum {
    AST_REQUIRE,      // (require module ...)
    AST_PROVIDE,      // (provide name ...)
    AST_DEFINE_LIB,   // (define-library ...)
    // ... existing types
} ast_type_t;

typedef struct {
    char** module_names;    // Required module names
    size_t num_modules;
    // For advanced forms:
    char** only_names;      // (only mod name ...)
    char* prefix;           // (prefix mod p)
    // etc.
} ast_require_t;

typedef struct {
    char** export_names;    // Exported names
    size_t num_exports;
} ast_provide_t;
```

### Compiler Changes

1. **Module loader**: Resolve paths, check cache, load files
2. **Dependency resolver**: Build graph, detect cycles, topological sort
3. **Symbol resolver**: Track exports, check visibility
4. **Linker**: Combine compiled modules

### New Files

```
lib/core/module_loader.cpp      // Module loading and caching
lib/core/dependency_resolver.cpp // Dependency graph
lib/core/symbol_resolver.cpp    // Export/import resolution
```

---

## Library Reorganization Plan

### Current Structure (Flat)

```
lib/
├── stdlib.esk      # 217 lines, everything in one file
└── math.esk        # 442 lines, everything in one file
```

### Target Structure (Modular)

```
lib/
├── stdlib.esk              # Minimal bootstrap, auto-loaded
│
├── core/
│   ├── operators.esk       # add, sub, mul, div, etc.
│   ├── predicates.esk      # is-zero?, is-null?, etc.
│   ├── combinators.esk     # compose, flip, identity
│   ├── currying.esk        # curry2, curry3, partial2
│   ├── lists.esk           # sort, partition, iota
│   └── strings.esk         # [NEW] string-split, string-join
│
├── math/
│   ├── constants.esk       # pi, e, epsilon
│   ├── linalg.esk          # det, inv, solve, dot, cross
│   ├── stats.esk           # variance, std, covariance
│   ├── calculus.esk        # integrate, newton
│   └── eigen.esk           # power-iteration
│
├── data/                   # [NEW PHASE 2]
│   ├── json.esk
│   ├── csv.esk
│   └── base64.esk
│
├── io/                     # [NEW PHASE 4]
│   ├── files.esk
│   └── paths.esk
│
├── net/                    # [NEW PHASE 5]
│   ├── tcp.esk
│   └── udp.esk
│
├── web/                    # [NEW PHASE 6]
│   ├── http.esk
│   └── url.esk
│
├── system/                 # [NEW PHASE 7]
│   ├── env.esk
│   ├── process.esk
│   └── time.esk
│
└── concurrent/             # [NEW PHASE 8]
    ├── threads.esk
    ├── channels.esk
    └── sync.esk
```

### Migration Examples

**Before (stdlib.esk):**
```scheme
;; Everything in one file
(define (compose f g) (lambda (x) (f (g x))))
(define (curry2 f) (lambda (x) (lambda (y) (f x y))))
(define (sort lst less?) ...)
```

**After (core/combinators.esk):**
```scheme
;;; lib/core/combinators.esk

(provide compose compose3 identity constantly flip)

(define (compose f g)
  (lambda (x) (f (g x))))

(define (compose3 f g h)
  (lambda (x) (f (g (h x)))))

(define (identity x) x)

(define (constantly x)
  (lambda (y) x))

(define (flip f)
  (lambda (x y) (f y x)))
```

**After (core/currying.esk):**
```scheme
;;; lib/core/currying.esk

(provide curry2 curry3 uncurry2 partial1 partial2 partial3 negate)

(define (curry2 f)
  (lambda (x)
    (lambda (y)
      (f x y))))

(define (curry3 f)
  (lambda (x)
    (lambda (y)
      (lambda (z)
        (f x y z)))))

;; ... etc
```

---

## Auto-Loading Behavior

When `eshkol-run` starts, these are automatically available:

```scheme
;; From builtins (C/LLVM):
;; +, -, *, /, <, >, =, cons, car, cdr, list, vector, etc.

;; From lib/stdlib.esk (bootstrap):
;; Minimal utilities that require/provide other core modules

;; From lib/core/* (auto-required by stdlib):
;; compose, curry2, sort, iota, partition, etc.

;; From lib/math/* (auto-required):
;; det, inv, solve, integrate, newton, etc.
```

User code can then require additional modules:

```scheme
(require data.json)
(require web.http)

(define response (http-get "https://api.example.com/data"))
(display (json-parse (http-response-body response)))
```

---

## Error Handling

### Module Not Found
```
Error: Module 'data.xml' not found
  Searched:
    - ./data/xml.esk
    - /usr/local/lib/eshkol/data/xml.esk
    - $ESHKOL_PATH entries
```

### Circular Dependency
```
Error: Circular dependency detected
  a.esk requires b.esk
  b.esk requires c.esk
  c.esk requires a.esk
```

### Undefined Export
```
Error: In module 'data.json'
  'json-parse' is provided but not defined
```

### Import Not Exported
```
Error: Cannot import 'internal-helper' from 'data.json'
  'internal-helper' is not exported by this module
```

---

## Implementation Phases

### Phase 0: Module System Infrastructure
1. Implement `require` and `provide` parsing
2. Implement module loader and path resolution
3. Implement dependency resolver
4. Implement symbol visibility checks

### Phase 1: Library Reorganization
1. Split stdlib.esk into core/* modules
2. Split math.esk into math/* modules
3. Update stdlib.esk to require core modules
4. Test all existing functionality

### Phase 2-9: New Features
Follow the extension implementation plan, with each new feature as a proper module.

---

*Document Version: 1.0*
*Last Updated: December 2025*
