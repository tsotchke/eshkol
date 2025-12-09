# Eshkol Extension Implementation Plan
## Complete System for AI, Networking, Text, HTTP, and General Applications

**Version**: 1.0
**Date**: December 2025
**Status**: Planning Phase

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Architecture Overview](#3-architecture-overview)
4. [Implementation Phases](#4-implementation-phases)
   - [Phase 1: Enhanced String Processing](#phase-1-enhanced-string-processing)
   - [Phase 2: Data Formats (JSON/CSV/Base64)](#phase-2-data-formats)
   - [Phase 3: Hash Tables](#phase-3-hash-tables)
   - [Phase 4: File System Operations](#phase-4-file-system-operations)
   - [Phase 5: TCP/UDP Networking](#phase-5-tcpudp-networking)
   - [Phase 6: HTTP Client/Server](#phase-6-http-clientserver)
   - [Phase 7: System & Environment](#phase-7-system--environment)
   - [Phase 8: Concurrency](#phase-8-concurrency)
   - [Phase 9: Error Handling](#phase-9-error-handling)
5. [Implementation Strategy](#5-implementation-strategy)
6. [File Structure](#6-file-structure)
7. [Testing Strategy](#7-testing-strategy)
8. [Dependencies](#8-dependencies)
9. [Risk Assessment](#9-risk-assessment)
10. [Success Metrics](#10-success-metrics)

---

## 1. Executive Summary

### Goal
Extend Eshkol from a scientific computing language with autodiff capabilities into a full-stack programming language suitable for:
- AI/ML application development
- Web services and APIs
- Network programming
- Data processing pipelines
- General-purpose scripting

### Scope
- **108 new functions** across 9 implementation phases
- **72 functions** implemented in C/LLVM (system primitives)
- **36 functions** implemented in pure Eshkol (high-level APIs)
- **~7,100 lines** of new code estimated

### Key Principles
1. **Minimal C Core**: Only implement in C what absolutely requires it
2. **Eshkol-First**: Build high-level APIs in pure Eshkol for maintainability
3. **Incremental Delivery**: Each phase delivers usable functionality
4. **Test-Driven**: Write tests before implementation
5. **Backward Compatible**: No breaking changes to existing code

---

## 2. Current State Analysis

### Existing Capabilities (266 functions/operators)

| Category | Count | Status |
|----------|-------|--------|
| Arithmetic & Math | 39 | Complete |
| List Operations | 61 | Complete |
| Higher-Order Functions | 10 | Complete |
| Tensor/Linear Algebra | 26 | Complete |
| Automatic Differentiation | 9 | Complete |
| String Operations | 15 | Basic |
| I/O Operations | 10 | Basic |
| Type System | 14 value types | Complete |

### Gaps Identified

| Category | Current | Needed |
|----------|---------|--------|
| String Processing | **Complete** (split, join, trim, case) | Regex support |
| Data Formats | None | JSON, CSV, Base64 |
| Key-Value Storage | Association lists O(n) | Hash tables O(1) |
| File System | Basic file I/O | Full directory operations |
| Networking | None | TCP/UDP sockets |
| HTTP | None | Client and server |
| Concurrency | None | Threads, channels |
| Error Handling | Basic `error` | Structured exceptions |

**Note**: String processing was implemented in Phase 1 (see `lib/core/strings.esk` and
builtin functions `string-split`, `string-contains?`, `string-index`, `string-upcase`,
`string-downcase` in llvm_codegen.cpp).

### Codebase Statistics

```
Component                    Lines    Purpose
─────────────────────────────────────────────────
lib/backend/llvm_codegen.cpp 29,352   LLVM IR generation
lib/frontend/parser.cpp       3,293   Tokenizer & parser
lib/core/arena_memory.cpp       934   Memory management
lib/stdlib.esk                  217   Standard library
lib/math.esk                    442   Math library
─────────────────────────────────────────────────
Total                        34,238
```

---

## 3. Architecture Overview

### Implementation Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    USER APPLICATION CODE                      │
│                         (*.esk files)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PURE ESHKOL LIBRARIES                        │
│  lib/http.esk  lib/json.esk  lib/csv.esk  lib/url.esk        │
│  lib/strings.esk  lib/base64.esk  lib/error.esk              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   BUILTIN FUNCTIONS                           │
│              (llvm_codegen.cpp dispatch)                      │
│  string-split  tcp-connect  hash-ref  file-exists?           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    RUNTIME SUPPORT                            │
│                  (arena_memory.cpp)                           │
│  Hash table allocation  Socket wrappers  Thread structures    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    C STANDARD LIBRARY                         │
│  string.h  stdio.h  stdlib.h  sys/socket.h  pthread.h        │
└─────────────────────────────────────────────────────────────┘
```

### Value Type Extensions

Current types (14):
```c
ESHKOL_VALUE_NULL        = 0   // Empty/null value
ESHKOL_VALUE_INT64       = 1   // 64-bit signed integer
ESHKOL_VALUE_DOUBLE      = 2   // Double-precision floating point
ESHKOL_VALUE_CONS_PTR    = 3   // Pointer to cons cell (list)
ESHKOL_VALUE_DUAL_NUMBER = 4   // Dual number for forward-mode AD
ESHKOL_VALUE_AD_NODE_PTR = 5   // Pointer to AD computation graph node
ESHKOL_VALUE_TENSOR_PTR  = 6   // Pointer to tensor structure
ESHKOL_VALUE_LAMBDA_SEXPR = 7  // Lambda S-expression (homoiconicity)
ESHKOL_VALUE_STRING_PTR  = 8   // Pointer to string
ESHKOL_VALUE_CHAR        = 9   // Character (Unicode codepoint)
ESHKOL_VALUE_VECTOR_PTR  = 10  // Scheme vector (heterogeneous array)
ESHKOL_VALUE_SYMBOL      = 11  // Symbol (interned string for identifiers)
ESHKOL_VALUE_CLOSURE_PTR = 12  // Closure (function + environment)
ESHKOL_VALUE_BOOL        = 13  // Boolean (#t or #f)
// ESHKOL_VALUE_MAX      = 15  // 4-bit type field limit
```

New types to add (2):
```c
ESHKOL_VALUE_HASH_PTR    = 14  // Hash table pointer
ESHKOL_VALUE_SOCKET_PTR  = 15  // Socket descriptor wrapper (uses MAX slot)
// Note: THREAD_PTR would require expanding beyond 4-bit type field
```

---

## 4. Implementation Phases

---

### Phase 1: Enhanced String Processing

**Status**: ✅ **COMPLETE**
**Dependencies**: None
**Implemented LOC**: ~500

#### Functions to Implement

##### C/LLVM Implementation (4 functions)

| Function | Signature | C Function Used |
|----------|-----------|-----------------|
| `string-split` | `(string-split str delim) → list` | Custom loop with `strstr` |
| `string-contains?` | `(string-contains? str substr) → bool` | `strstr` |
| `string-index` | `(string-index str substr) → int` | `strstr` |
| `string-upcase` | `(string-upcase str) → string` | `toupper` loop |
| `string-downcase` | `(string-downcase str) → string` | `tolower` loop |

##### Pure Eshkol Implementation (lib/core/strings.esk) (7 functions)

```scheme
;;; lib/core/strings.esk - String utilities
;;; Part of the core library (auto-loaded)

(define-module core.strings
  (export string-join string-trim string-trim-left string-trim-right
          string-replace string-reverse string-copy string-repeat
          string-starts-with? string-ends-with?)

;; string-join: Join list of strings with delimiter
;; (string-join '("a" "b" "c") ",") => "a,b,c"
(define (string-join lst delim)
  (if (null? lst)
      ""
      (fold (lambda (s acc)
              (string-append acc delim s))
            (car lst)
            (cdr lst))))

;; string-trim: Remove leading and trailing whitespace
(define (string-trim str)
  (string-trim-right (string-trim-left str)))

;; string-trim-left: Remove leading whitespace
(define (string-trim-left str)
  (define (find-start i)
    (if (>= i (string-length str))
        ""
        (let ((c (string-ref str i)))
          (if (or (= c 32) (= c 9) (= c 10) (= c 13))
              (find-start (+ i 1))
              (substring str i (string-length str))))))
  (find-start 0))

;; string-trim-right: Remove trailing whitespace
(define (string-trim-right str)
  (define len (string-length str))
  (define (find-end i)
    (if (< i 0)
        ""
        (let ((c (string-ref str i)))
          (if (or (= c 32) (= c 9) (= c 10) (= c 13))
              (find-end (- i 1))
              (substring str 0 (+ i 1))))))
  (find-end (- len 1)))

;; string-replace: Replace all occurrences of old with new
;; (string-replace "hello world" "o" "0") => "hell0 w0rld"
(define (string-replace str old new)
  (string-join (string-split str old) new))

;; string-reverse: Reverse a string
(define (string-reverse str)
  (list->string (reverse (string->list str))))

;; string-copy: Create a copy of string
(define (string-copy str)
  (substring str 0 (string-length str)))

;; string-repeat: Repeat string n times
(define (string-repeat str n)
  (if (<= n 0)
      ""
      (string-append str (string-repeat str (- n 1)))))

;; string-starts-with?: Check if string starts with prefix
(define (string-starts-with? str prefix)
  (and (>= (string-length str) (string-length prefix))
       (string=? (substring str 0 (string-length prefix)) prefix)))

;; string-ends-with?: Check if string ends with suffix
(define (string-ends-with? str suffix)
  (let ((str-len (string-length str))
        (suf-len (string-length suffix)))
    (and (>= str-len suf-len)
         (string=? (substring str (- str-len suf-len) str-len) suffix))))

) ; end define-module core.strings
```

##### C/LLVM Implementation Details

**string-split** (most complex):
```cpp
Value* codegenStringSplit(const eshkol_operations_t* op) {
    // 1. Get string and delimiter
    Value* str = extractStringPtr(codegenAST(&op->call_op.variables[0]));
    Value* delim = extractStringPtr(codegenAST(&op->call_op.variables[1]));

    // 2. Get delimiter length
    Value* delim_len = builder->CreateCall(strlen_func, {delim});

    // 3. Create result list (build in reverse, then reverse)
    // 4. Loop: find next delimiter with strstr
    // 5. Extract substring between current position and delimiter
    // 6. Add to result list
    // 7. Move position past delimiter
    // 8. Repeat until no more delimiters
    // 9. Add final segment
    // 10. Reverse list and return
}
```

**string-upcase/string-downcase**:
```cpp
Value* codegenStringUpcase(const eshkol_operations_t* op) {
    Value* str = extractStringPtr(codegenAST(&op->call_op.variables[0]));
    Value* len = builder->CreateCall(strlen_func, {str});
    Value* alloc_len = builder->CreateAdd(len, ConstantInt::get(i64, 1));

    // Allocate new string
    Value* arena_ptr = builder->CreateLoad(ptr_type, global_arena);
    Value* new_str = builder->CreateCall(arena_allocate_func, {arena_ptr, alloc_len});

    // Loop through characters, calling toupper on each
    // (Create loop with PHI nodes)

    return packPtrToTaggedValue(new_str, ESHKOL_VALUE_STRING_PTR);
}
```

---

### Phase 2: Data Formats

**Priority**: High
**Dependencies**: Phase 1 (string-split)
**Estimated LOC**: 800

#### All Pure Eshkol Implementation

##### JSON (lib/data/json.esk)

```scheme
;;; lib/data/json.esk - JSON parsing and generation

(define-module data.json
  (import core.strings)
  (export json-parse json-stringify json-get json-get-in json-escape-string)

;; JSON value types:
;; - object: association list ((key . value) ...)
;; - array: Eshkol list (value ...)
;; - string: Eshkol string
;; - number: int64 or double
;; - boolean: #t or #f
;; - null: '()

;; json-parse: Parse JSON string to Eshkol value
(define (json-parse str)
  (define pos 0)
  (define len (string-length str))

  ;; Skip whitespace
  (define (skip-ws)
    (define (loop)
      (when (< pos len)
        (let ((c (string-ref str pos)))
          (when (or (= c 32) (= c 9) (= c 10) (= c 13))
            (set! pos (+ pos 1))
            (loop)))))
    (loop))

  ;; Parse value based on first character
  (define (parse-value)
    (skip-ws)
    (if (>= pos len)
        (error "Unexpected end of JSON")
        (let ((c (string-ref str pos)))
          (cond
            ((= c 123) (parse-object))      ; {
            ((= c 91) (parse-array))        ; [
            ((= c 34) (parse-string))       ; "
            ((= c 116) (parse-true))        ; t
            ((= c 102) (parse-false))       ; f
            ((= c 110) (parse-null))        ; n
            ((or (= c 45) (and (>= c 48) (<= c 57)))
             (parse-number))
            (else (error "Invalid JSON character"))))))

  ;; Parse object {...}
  (define (parse-object)
    (set! pos (+ pos 1))  ; skip {
    (skip-ws)
    (if (= (string-ref str pos) 125)  ; }
        (begin (set! pos (+ pos 1)) '())
        (let loop ((result '()))
          (skip-ws)
          (let* ((key (parse-string))
                 (_ (skip-ws))
                 (_ (set! pos (+ pos 1)))  ; skip :
                 (value (parse-value)))
            (skip-ws)
            (let ((c (string-ref str pos)))
              (set! pos (+ pos 1))
              (if (= c 125)  ; }
                  (reverse (cons (cons key value) result))
                  (loop (cons (cons key value) result))))))))

  ;; Parse array [...]
  (define (parse-array)
    (set! pos (+ pos 1))  ; skip [
    (skip-ws)
    (if (= (string-ref str pos) 93)  ; ]
        (begin (set! pos (+ pos 1)) '())
        (let loop ((result '()))
          (let ((value (parse-value)))
            (skip-ws)
            (let ((c (string-ref str pos)))
              (set! pos (+ pos 1))
              (if (= c 93)  ; ]
                  (reverse (cons value result))
                  (loop (cons value result))))))))

  ;; Parse string "..."
  (define (parse-string)
    (set! pos (+ pos 1))  ; skip opening "
    (let loop ((chars '()))
      (let ((c (string-ref str pos)))
        (set! pos (+ pos 1))
        (cond
          ((= c 34) (list->string (reverse chars)))  ; "
          ((= c 92)  ; backslash escape
           (let ((next (string-ref str pos)))
             (set! pos (+ pos 1))
             (loop (cons (cond ((= next 110) 10)   ; \n
                              ((= next 116) 9)    ; \t
                              ((= next 114) 13)   ; \r
                              (else next))
                        chars))))
          (else (loop (cons c chars)))))))

  ;; Parse number
  (define (parse-number)
    (let loop ((chars '()))
      (if (>= pos len)
          (string->number (list->string (reverse chars)))
          (let ((c (string-ref str pos)))
            (if (or (and (>= c 48) (<= c 57))
                    (= c 45) (= c 46) (= c 101) (= c 69) (= c 43))
                (begin
                  (set! pos (+ pos 1))
                  (loop (cons c chars)))
                (string->number (list->string (reverse chars))))))))

  ;; Parse true
  (define (parse-true)
    (set! pos (+ pos 4))
    #t)

  ;; Parse false
  (define (parse-false)
    (set! pos (+ pos 5))
    #f)

  ;; Parse null
  (define (parse-null)
    (set! pos (+ pos 4))
    '())

  ;; Main parse
  (parse-value))

;; json-stringify: Convert Eshkol value to JSON string
(define (json-stringify value)
  (cond
    ((null? value) "null")
    ((boolean? value) (if value "true" "false"))
    ((number? value) (number->string value))
    ((string? value) (string-append "\"" (json-escape-string value) "\""))
    ((and (pair? value) (pair? (car value)) (string? (caar value)))
     ;; Object (association list with string keys)
     (string-append "{"
                   (string-join
                    (map (lambda (pair)
                           (string-append (json-stringify (car pair))
                                        ":"
                                        (json-stringify (cdr pair))))
                         value)
                    ",")
                   "}"))
    ((list? value)
     ;; Array
     (string-append "["
                   (string-join (map json-stringify value) ",")
                   "]"))
    (else (error "Cannot convert to JSON"))))

;; json-escape-string: Escape special characters in string
(define (json-escape-string str)
  (let loop ((i 0) (result '()))
    (if (>= i (string-length str))
        (list->string (reverse result))
        (let ((c (string-ref str i)))
          (loop (+ i 1)
                (cond
                  ((= c 34) (append '(34 92) result))   ; " -> \"
                  ((= c 92) (append '(92 92) result))   ; \ -> \\
                  ((= c 10) (append '(110 92) result))  ; newline -> \n
                  ((= c 13) (append '(114 92) result))  ; return -> \r
                  ((= c 9) (append '(116 92) result))   ; tab -> \t
                  (else (cons c result))))))))

;; json-get: Get value from JSON object by key
(define (json-get obj key)
  (let ((pair (assoc key obj)))
    (if pair (cdr pair) #f)))

;; json-get-in: Get nested value by path
(define (json-get-in obj path)
  (if (null? path)
      obj
      (let ((next (if (string? (car path))
                      (json-get obj (car path))
                      (list-ref obj (car path)))))
        (if next
            (json-get-in next (cdr path))
            #f))))

) ; end define-module data.json
```

##### CSV (lib/data/csv.esk)

```scheme
;;; lib/data/csv.esk - CSV parsing and generation

(define-module data.csv
  (import core.strings)
  (import io.files)
  (export csv-parse csv-parse-with-headers csv-stringify csv-escape-field
          csv-read-file csv-write-file)

;; csv-parse: Parse CSV string to list of lists
;; Handles quoted fields with embedded commas and quotes
(define (csv-parse str)
  (define (parse-row row-str)
    ;; Simple split - TODO: handle quoted fields
    (string-split row-str ","))

  (map parse-row (string-split str "\n")))

;; csv-parse-with-headers: Parse CSV with first row as headers
;; Returns list of association lists
(define (csv-parse-with-headers str)
  (let* ((rows (csv-parse str))
         (headers (car rows))
         (data (cdr rows)))
    (map (lambda (row)
           (map cons headers row))
         data)))

;; csv-stringify: Convert list of lists to CSV string
(define (csv-stringify data)
  (string-join
   (map (lambda (row)
          (string-join (map csv-escape-field row) ","))
        data)
   "\n"))

;; csv-escape-field: Escape field if necessary
(define (csv-escape-field field)
  (let ((str (if (string? field) field (number->string field))))
    (if (or (string-contains? str ",")
            (string-contains? str "\"")
            (string-contains? str "\n"))
        (string-append "\"" (string-replace str "\"" "\"\"") "\"")
        str)))

;; csv-read-file: Read and parse CSV file
(define (csv-read-file path)
  (csv-parse (read-file path)))

;; csv-write-file: Write data to CSV file
(define (csv-write-file path data)
  (write-file path (csv-stringify data)))

) ; end define-module data.csv
```

##### Base64 (lib/data/base64.esk)

```scheme
;;; lib/data/base64.esk - Base64 encoding/decoding

(define-module data.base64
  (import core.strings)
  (export base64-encode base64-decode)

(define base64-chars
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")

;; base64-encode: Encode string to base64
(define (base64-encode str)
  (define len (string-length str))
  (define (get-byte i)
    (if (< i len) (string-ref str i) 0))

  (let loop ((i 0) (result '()))
    (if (>= i len)
        (let* ((padding (modulo (- 3 (modulo len 3)) 3))
               (encoded (list->string (reverse result))))
          (string-append
           (if (> padding 0)
               (substring encoded 0 (- (string-length encoded) padding))
               encoded)
           (string-repeat "=" padding)))
        (let* ((b1 (get-byte i))
               (b2 (get-byte (+ i 1)))
               (b3 (get-byte (+ i 2)))
               (c1 (string-ref base64-chars (quotient b1 4)))
               (c2 (string-ref base64-chars
                              (+ (* (modulo b1 4) 16) (quotient b2 16))))
               (c3 (string-ref base64-chars
                              (+ (* (modulo b2 16) 4) (quotient b3 64))))
               (c4 (string-ref base64-chars (modulo b3 64))))
          (loop (+ i 3) (cons c4 (cons c3 (cons c2 (cons c1 result)))))))))

;; base64-decode: Decode base64 string
(define (base64-decode str)
  ;; Remove padding and decode
  (define (char-index c)
    (string-index base64-chars (list->string (list c))))

  (let* ((clean (string-replace str "=" ""))
         (len (string-length clean)))
    (let loop ((i 0) (result '()))
      (if (>= i len)
          (list->string (reverse result))
          (let* ((c1 (char-index (string-ref clean i)))
                 (c2 (if (< (+ i 1) len) (char-index (string-ref clean (+ i 1))) 0))
                 (c3 (if (< (+ i 2) len) (char-index (string-ref clean (+ i 2))) 0))
                 (c4 (if (< (+ i 3) len) (char-index (string-ref clean (+ i 3))) 0))
                 (b1 (+ (* c1 4) (quotient c2 16)))
                 (b2 (+ (* (modulo c2 16) 16) (quotient c3 4)))
                 (b3 (+ (* (modulo c3 4) 64) c4)))
            (loop (+ i 4)
                  (cons b3 (cons b2 (cons b1 result)))))))))

) ; end define-module data.base64
```

---

### Phase 3: Hash Tables

**Priority**: High
**Dependencies**: None
**Estimated LOC**: 600

#### All C/LLVM Implementation

##### New Type Definition (eshkol.h)

```c
// Add to eshkol_value_type_t enum:
ESHKOL_VALUE_HASH_PTR = 14,  // Pointer to hash table
```

##### Runtime Structures (arena_memory.h)

```c
// Hash table entry (linked list for collision handling)
typedef struct eshkol_hash_entry {
    eshkol_tagged_value_t key;
    eshkol_tagged_value_t value;
    struct eshkol_hash_entry* next;
} eshkol_hash_entry_t;

// Hash table structure
typedef struct eshkol_hash_table {
    size_t capacity;          // Number of buckets
    size_t size;              // Number of entries
    eshkol_hash_entry_t** buckets;  // Array of bucket pointers
} eshkol_hash_table_t;

// Arena allocation functions
eshkol_hash_table_t* arena_allocate_hash_table(arena_t* arena, size_t initial_capacity);
eshkol_hash_entry_t* arena_allocate_hash_entry(arena_t* arena);

// Hash table operations (C implementations for performance)
uint64_t eshkol_hash_value(const eshkol_tagged_value_t* key);
bool eshkol_hash_values_equal(const eshkol_tagged_value_t* a, const eshkol_tagged_value_t* b);
```

##### Codegen Functions (llvm_codegen.cpp)

| Function | Implementation |
|----------|----------------|
| `make-hash-table` | Allocate hash table struct, initialize buckets |
| `hash-table?` | Check type tag == HASH_PTR |
| `hash-ref` | Hash key, walk bucket chain, return value or default |
| `hash-set!` | Hash key, find/create entry, set value |
| `hash-remove!` | Hash key, unlink entry from chain |
| `hash-has-key?` | Hash key, walk chain, return bool |
| `hash-keys` | Walk all buckets, collect keys into list |
| `hash-values` | Walk all buckets, collect values into list |
| `hash-count` | Return size field |
| `hash->alist` | Walk all buckets, build alist |
| `alist->hash` | Create table, iterate alist, set each pair |

##### Hash Function

```c
uint64_t eshkol_hash_value(const eshkol_tagged_value_t* key) {
    uint8_t type = key->type & 0x0F;

    switch (type) {
        case ESHKOL_VALUE_INT64:
            // FNV-1a hash for integers
            return key->data.int_val * 1099511628211ULL;

        case ESHKOL_VALUE_DOUBLE:
            // Hash the bits of the double
            return (uint64_t)(key->data.double_val * 1e10) * 1099511628211ULL;

        case ESHKOL_VALUE_STRING_PTR: {
            // FNV-1a hash for strings
            const char* str = (const char*)key->data.ptr_val;
            uint64_t hash = 14695981039346656037ULL;
            while (*str) {
                hash ^= (uint8_t)*str++;
                hash *= 1099511628211ULL;
            }
            return hash;
        }

        case ESHKOL_VALUE_SYMBOL:
            // Symbols are interned, so hash the pointer
            return key->data.ptr_val * 1099511628211ULL;

        default:
            // Hash the raw pointer/value
            return key->data.raw_val * 1099511628211ULL;
    }
}
```

---

### Phase 4: File System Operations

**Priority**: Medium-High
**Dependencies**: None
**Estimated LOC**: 400

#### All C/LLVM Implementation

##### C Library Functions to Declare

```cpp
// In createBuiltinFunctions():

// File operations
declareExternFunction("access", i32, {ptr_type, i32});
declareExternFunction("remove", i32, {ptr_type});
declareExternFunction("rename", i32, {ptr_type, ptr_type});

// Stat structure access
declareExternFunction("stat", i32, {ptr_type, ptr_type});

// Directory operations
declareExternFunction("opendir", ptr_type, {ptr_type});
declareExternFunction("readdir", ptr_type, {ptr_type});
declareExternFunction("closedir", i32, {ptr_type});
declareExternFunction("mkdir", i32, {ptr_type, i32});
declareExternFunction("rmdir", i32, {ptr_type});

// Working directory
declareExternFunction("getcwd", ptr_type, {ptr_type, i64});
declareExternFunction("chdir", i32, {ptr_type});
```

##### Functions to Implement

| Function | C Function | Notes |
|----------|------------|-------|
| `file-exists?` | `access(path, F_OK)` | Return bool |
| `file-readable?` | `access(path, R_OK)` | Return bool |
| `file-writable?` | `access(path, W_OK)` | Return bool |
| `file-delete` | `remove(path)` | Return bool success |
| `file-rename` | `rename(old, new)` | Return bool success |
| `file-size` | `stat(path, &st); st.st_size` | Return int64 |
| `directory-exists?` | `stat` + `S_ISDIR` check | Return bool |
| `make-directory` | `mkdir(path, 0755)` | Return bool success |
| `delete-directory` | `rmdir(path)` | Return bool success |
| `directory-list` | `opendir`, loop `readdir`, `closedir` | Return list of strings |
| `current-directory` | `getcwd(buf, size)` | Return string |
| `set-current-directory!` | `chdir(path)` | Return bool success |
| `read-file` | `fopen`, `fseek`, `ftell`, `fread`, `fclose` | Return string |
| `write-file` | `fopen`, `fwrite`, `fclose` | Return bool success |
| `append-file` | `fopen` with "a" mode | Return bool success |

##### Path Utilities (lib/io/paths.esk - Pure Eshkol)

```scheme
;;; lib/io/paths.esk - Path manipulation utilities

(define-module io.paths
  (import core.strings)
  (export path-separator path-join path-dirname path-basename
          path-extension path-stem path-absolute?)

;; Detect path separator based on OS
(define path-separator "/")  ; TODO: detect Windows

;; path-join: Join path components
(define (path-join . parts)
  (string-join (filter (lambda (s) (> (string-length s) 0)) parts)
               path-separator))

;; path-dirname: Get directory portion of path
(define (path-dirname path)
  (let ((idx (string-last-index path path-separator)))
    (if (< idx 0)
        "."
        (substring path 0 idx))))

;; path-basename: Get filename portion of path
(define (path-basename path)
  (let ((idx (string-last-index path path-separator)))
    (if (< idx 0)
        path
        (substring path (+ idx 1) (string-length path)))))

;; path-extension: Get file extension
(define (path-extension path)
  (let* ((base (path-basename path))
         (idx (string-last-index base ".")))
    (if (< idx 0)
        ""
        (substring base idx (string-length base)))))

;; path-stem: Get filename without extension
(define (path-stem path)
  (let* ((base (path-basename path))
         (idx (string-last-index base ".")))
    (if (< idx 0)
        base
        (substring base 0 idx))))

;; path-absolute?: Check if path is absolute
(define (path-absolute? path)
  (and (> (string-length path) 0)
       (= (string-ref path 0) (string-ref "/" 0))))

) ; end define-module io.paths
```

---

### Phase 5: TCP/UDP Networking

**Priority**: High
**Dependencies**: Phase 4 (for helper functions)
**Estimated LOC**: 1000

#### All C/LLVM Implementation

##### New Type Definition

```c
// Add to eshkol_value_type_t:
ESHKOL_VALUE_SOCKET_PTR = 15,  // Uses last available slot in 4-bit type field

// Socket wrapper structure (arena_memory.h)
typedef struct eshkol_socket {
    int fd;                    // File descriptor
    int type;                  // SOCK_STREAM or SOCK_DGRAM
    int family;               // AF_INET or AF_INET6
    bool connected;           // Connection state
    bool listening;           // Listening state
} eshkol_socket_t;
```

##### C Library Functions to Declare

```cpp
// Socket creation and connection
declareExternFunction("socket", i32, {i32, i32, i32});
declareExternFunction("connect", i32, {i32, ptr_type, i32});
declareExternFunction("bind", i32, {i32, ptr_type, i32});
declareExternFunction("listen", i32, {i32, i32});
declareExternFunction("accept", i32, {i32, ptr_type, ptr_type});

// Data transfer
declareExternFunction("send", i64, {i32, ptr_type, i64, i32});
declareExternFunction("recv", i64, {i32, ptr_type, i64, i32});
declareExternFunction("sendto", i64, {i32, ptr_type, i64, i32, ptr_type, i32});
declareExternFunction("recvfrom", i64, {i32, ptr_type, i64, i32, ptr_type, ptr_type});

// Socket control
declareExternFunction("close", i32, {i32});
declareExternFunction("shutdown", i32, {i32, i32});
declareExternFunction("setsockopt", i32, {i32, i32, i32, ptr_type, i32});
declareExternFunction("getsockopt", i32, {i32, i32, i32, ptr_type, ptr_type});

// Address resolution
declareExternFunction("getaddrinfo", i32, {ptr_type, ptr_type, ptr_type, ptr_type});
declareExternFunction("freeaddrinfo", void_type, {ptr_type});
declareExternFunction("inet_ntop", ptr_type, {i32, ptr_type, ptr_type, i32});
declareExternFunction("inet_pton", i32, {i32, ptr_type, ptr_type});
declareExternFunction("htons", i16, {i16});
declareExternFunction("ntohs", i16, {i16});
declareExternFunction("htonl", i32, {i32});
declareExternFunction("ntohl", i32, {i32});
```

##### Functions to Implement

| Function | Description |
|----------|-------------|
| `tcp-connect` | Create socket, resolve host, connect |
| `tcp-listen` | Create socket, bind, listen |
| `tcp-accept` | Accept connection, return (socket . address) |
| `tcp-send` | Send data, return bytes sent |
| `tcp-recv` | Receive data, return string |
| `tcp-close` | Close socket |
| `udp-socket` | Create UDP socket |
| `udp-bind` | Bind UDP socket to port |
| `udp-send-to` | Send datagram to address |
| `udp-recv-from` | Receive datagram, return (data . address) |
| `socket-set-timeout` | Set SO_RCVTIMEO/SO_SNDTIMEO |
| `socket-set-nonblocking` | Set O_NONBLOCK via fcntl |
| `socket-set-reuseaddr` | Set SO_REUSEADDR |
| `resolve-hostname` | getaddrinfo, return IP string |
| `socket-local-address` | getsockname, return address string |
| `socket-peer-address` | getpeername, return address string |

##### Implementation Example: tcp-connect

```cpp
Value* codegenTcpConnect(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) {
        eshkol_error("tcp-connect requires 2 arguments (host port)");
        return nullptr;
    }

    Value* host = extractStringPtr(codegenAST(&op->call_op.variables[0]));
    Value* port = extractInt64(codegenAST(&op->call_op.variables[1]));

    // 1. Create socket
    Value* sock_fd = builder->CreateCall(socket_func, {
        ConstantInt::get(i32, AF_INET),
        ConstantInt::get(i32, SOCK_STREAM),
        ConstantInt::get(i32, 0)
    });

    // 2. Check for error
    Value* sock_valid = builder->CreateICmpSGE(sock_fd, ConstantInt::get(i32, 0));
    // ... error handling ...

    // 3. Resolve hostname with getaddrinfo
    // 4. Connect to resolved address
    // 5. Create socket wrapper structure
    // 6. Return as SOCKET_PTR tagged value

    Value* arena_ptr = builder->CreateLoad(ptr_type, global_arena);
    Value* socket_struct = builder->CreateCall(arena_allocate_socket_func, {arena_ptr});

    // Store fd, type, connected flag
    // ...

    return packPtrToTaggedValue(socket_struct, ESHKOL_VALUE_SOCKET_PTR);
}
```

---

### Phase 6: HTTP Client/Server

**Priority**: High
**Dependencies**: Phase 5 (sockets), Phase 1 (strings), Phase 2 (JSON)
**Estimated LOC**: 1200

#### All Pure Eshkol Implementation (lib/web/http.esk)

```scheme
;;; lib/web/http.esk - HTTP/1.1 client and server

(define-module web.http
  (import core.strings)
  (import net.tcp)
  (import web.url)
  (export http-request http-get http-post http-post-json
          http-response-status http-response-headers http-response-body http-response-json
          http-serve http-request-method http-request-path http-request-headers
          http-request-body http-request-header http-status-text)

;;; ============================================================
;;; HTTP Client
;;; ============================================================

;; http-request: Make HTTP request
;; method: "GET", "POST", "PUT", "DELETE", etc.
;; url: Full URL or just path
;; headers: Association list of headers
;; body: Request body (string or #f)
;; Returns: (status-code headers body)
(define (http-request method url headers body)
  (let* ((parsed (url-parse url))
         (host (cdr (assoc "host" parsed)))
         (port (let ((p (assoc "port" parsed)))
                 (if p (string->number (cdr p)) 80)))
         (path (let ((p (assoc "path" parsed)))
                 (if p (cdr p) "/")))
         (socket (tcp-connect host port)))

    ;; Build request
    (define request-line
      (string-append method " " path " HTTP/1.1\r\n"))

    (define header-lines
      (string-join
       (map (lambda (h)
              (string-append (car h) ": " (cdr h)))
            (cons (cons "Host" host)
                  (cons (cons "Connection" "close")
                        (if body
                            (cons (cons "Content-Length"
                                       (number->string (string-length body)))
                                  headers)
                            headers))))
       "\r\n"))

    (define full-request
      (string-append request-line header-lines "\r\n\r\n"
                    (if body body "")))

    ;; Send request
    (tcp-send socket full-request)

    ;; Receive response
    (define response (http-receive-response socket))

    ;; Close connection
    (tcp-close socket)

    response))

;; http-receive-response: Read and parse HTTP response
(define (http-receive-response socket)
  (define (read-until-crlf-crlf)
    (let loop ((data ""))
      (let ((chunk (tcp-recv socket 4096)))
        (if (= (string-length chunk) 0)
            data
            (let ((new-data (string-append data chunk)))
              (if (string-contains? new-data "\r\n\r\n")
                  new-data
                  (loop new-data)))))))

  (let* ((response (read-until-crlf-crlf))
         (header-end (string-index response "\r\n\r\n"))
         (header-part (substring response 0 header-end))
         (body-start (+ header-end 4))
         (initial-body (substring response body-start (string-length response)))
         (header-lines (string-split header-part "\r\n"))
         (status-line (car header-lines))
         (status-code (string->number
                       (cadr (string-split status-line " "))))
         (headers (map (lambda (line)
                        (let ((parts (string-split line ": ")))
                          (cons (car parts) (cadr parts))))
                      (cdr header-lines))))

    ;; Read remaining body if Content-Length specified
    (let* ((content-length-header (assoc "Content-Length" headers))
           (content-length (if content-length-header
                              (string->number (cdr content-length-header))
                              0))
           (remaining (- content-length (string-length initial-body)))
           (full-body (if (> remaining 0)
                         (string-append initial-body
                                       (tcp-recv socket remaining))
                         initial-body)))

      (list status-code headers full-body))))

;; Convenience functions
(define (http-get url)
  (http-request "GET" url '() #f))

(define (http-post url body)
  (http-request "POST" url
               '(("Content-Type" . "application/json"))
               body))

(define (http-post-json url data)
  (http-post url (json-stringify data)))

;; Response accessors
(define (http-response-status response)
  (car response))

(define (http-response-headers response)
  (cadr response))

(define (http-response-body response)
  (caddr response))

(define (http-response-json response)
  (json-parse (http-response-body response)))

;;; ============================================================
;;; HTTP Server
;;; ============================================================

;; http-serve: Start HTTP server
;; port: Port number to listen on
;; handler: Function (request) -> response
;;   request: (method path headers body)
;;   response: (status-code headers body) or just body string
(define (http-serve port handler)
  (let ((server-socket (tcp-listen port 128)))
    (display "HTTP server listening on port ")
    (display port)
    (newline)

    (let loop ()
      (let* ((client-info (tcp-accept server-socket))
             (client-socket (car client-info)))

        ;; Handle request in same thread (TODO: spawn thread)
        (http-handle-client client-socket handler)

        ;; Continue accepting
        (loop)))))

;; http-handle-client: Handle single client connection
(define (http-handle-client socket handler)
  (let* ((request-data (tcp-recv socket 8192))
         (request (http-parse-request request-data))
         (response (handler request))
         (response-str (http-format-response response)))

    (tcp-send socket response-str)
    (tcp-close socket)))

;; http-parse-request: Parse HTTP request
(define (http-parse-request data)
  (let* ((header-end (string-index data "\r\n\r\n"))
         (header-part (substring data 0 header-end))
         (body (substring data (+ header-end 4) (string-length data)))
         (lines (string-split header-part "\r\n"))
         (request-line (car lines))
         (request-parts (string-split request-line " "))
         (method (car request-parts))
         (path (cadr request-parts))
         (headers (map (lambda (line)
                        (let ((parts (string-split line ": ")))
                          (cons (car parts) (cadr parts))))
                      (cdr lines))))

    (list method path headers body)))

;; http-format-response: Format HTTP response
(define (http-format-response response)
  (let* ((status (if (pair? response) (car response) 200))
         (headers (if (and (pair? response) (pair? (cdr response)))
                     (cadr response)
                     '()))
         (body (cond
                ((string? response) response)
                ((and (pair? response) (pair? (cdr response)) (pair? (cddr response)))
                 (caddr response))
                (else "")))
         (status-text (http-status-text status))
         (all-headers (cons (cons "Content-Length"
                                 (number->string (string-length body)))
                           (cons (cons "Connection" "close")
                                 headers))))

    (string-append
     "HTTP/1.1 " (number->string status) " " status-text "\r\n"
     (string-join
      (map (lambda (h) (string-append (car h) ": " (cdr h)))
           all-headers)
      "\r\n")
     "\r\n\r\n"
     body)))

;; http-status-text: Get status text for code
(define (http-status-text code)
  (cond
    ((= code 200) "OK")
    ((= code 201) "Created")
    ((= code 204) "No Content")
    ((= code 301) "Moved Permanently")
    ((= code 302) "Found")
    ((= code 304) "Not Modified")
    ((= code 400) "Bad Request")
    ((= code 401) "Unauthorized")
    ((= code 403) "Forbidden")
    ((= code 404) "Not Found")
    ((= code 405) "Method Not Allowed")
    ((= code 500) "Internal Server Error")
    ((= code 502) "Bad Gateway")
    ((= code 503) "Service Unavailable")
    (else "Unknown")))

;; Request accessors
(define (http-request-method request) (car request))
(define (http-request-path request) (cadr request))
(define (http-request-headers request) (caddr request))
(define (http-request-body request) (cadddr request))
(define (http-request-header request name)
  (let ((h (assoc name (http-request-headers request))))
    (if h (cdr h) #f)))

) ; end define-module web.http
```

##### URL Utilities (lib/web/url.esk)

```scheme
;;; lib/web/url.esk - URL parsing and encoding

(define-module web.url
  (import core.strings)
  (export url-parse url-encode url-decode query-parse query-stringify)

;; url-parse: Parse URL into components
;; Returns alist: ((scheme . "http") (host . "example.com") (port . "80") ...)
(define (url-parse url)
  (let* ((scheme-end (string-index url "://"))
         (scheme (if (>= scheme-end 0)
                    (substring url 0 scheme-end)
                    "http"))
         (rest (if (>= scheme-end 0)
                  (substring url (+ scheme-end 3) (string-length url))
                  url))
         (path-start (string-index rest "/"))
         (host-port (if (>= path-start 0)
                       (substring rest 0 path-start)
                       rest))
         (path (if (>= path-start 0)
                  (substring rest path-start (string-length rest))
                  "/"))
         (port-start (string-index host-port ":"))
         (host (if (>= port-start 0)
                  (substring host-port 0 port-start)
                  host-port))
         (port (if (>= port-start 0)
                  (substring host-port (+ port-start 1) (string-length host-port))
                  (if (string=? scheme "https") "443" "80")))
         (query-start (string-index path "?"))
         (path-only (if (>= query-start 0)
                       (substring path 0 query-start)
                       path))
         (query (if (>= query-start 0)
                   (substring path (+ query-start 1) (string-length path))
                   "")))

    (list (cons "scheme" scheme)
          (cons "host" host)
          (cons "port" port)
          (cons "path" path-only)
          (cons "query" query))))

;; url-encode: Percent-encode string
(define (url-encode str)
  (define (encode-char c)
    (cond
      ;; Unreserved characters (RFC 3986)
      ((and (>= c 65) (<= c 90)) (list->string (list c)))   ; A-Z
      ((and (>= c 97) (<= c 122)) (list->string (list c)))  ; a-z
      ((and (>= c 48) (<= c 57)) (list->string (list c)))   ; 0-9
      ((= c 45) "-")   ; -
      ((= c 95) "_")   ; _
      ((= c 46) ".")   ; .
      ((= c 126) "~")  ; ~
      (else
       (string-append "%"
                     (number->string (quotient c 16) 16)
                     (number->string (remainder c 16) 16)))))

  (let loop ((i 0) (result '()))
    (if (>= i (string-length str))
        (apply string-append (reverse result))
        (loop (+ i 1)
              (cons (encode-char (string-ref str i)) result)))))

;; url-decode: Decode percent-encoded string
(define (url-decode str)
  (let loop ((i 0) (result '()))
    (if (>= i (string-length str))
        (list->string (reverse result))
        (let ((c (string-ref str i)))
          (if (= c 37)  ; %
              (let ((hex (substring str (+ i 1) (+ i 3))))
                (loop (+ i 3)
                      (cons (string->number hex 16) result)))
              (loop (+ i 1)
                    (cons (if (= c 43) 32 c) result)))))))  ; + -> space

;; query-parse: Parse query string to alist
(define (query-parse query)
  (map (lambda (pair)
         (let ((parts (string-split pair "=")))
           (cons (url-decode (car parts))
                 (url-decode (if (pair? (cdr parts))
                                (cadr parts)
                                "")))))
       (string-split query "&")))

;; query-stringify: Build query string from alist
(define (query-stringify params)
  (string-join
   (map (lambda (p)
          (string-append (url-encode (car p)) "=" (url-encode (cdr p))))
        params)
   "&"))

) ; end define-module web.url
```

---

### Phase 7: System & Environment

**Priority**: Medium
**Dependencies**: None
**Estimated LOC**: 300

#### All C/LLVM Implementation

##### C Library Functions to Declare

```cpp
declareExternFunction("getenv", ptr_type, {ptr_type});
declareExternFunction("setenv", i32, {ptr_type, ptr_type, i32});
declareExternFunction("unsetenv", i32, {ptr_type});
declareExternFunction("exit", void_type, {i32});
declareExternFunction("system", i32, {ptr_type});
declareExternFunction("usleep", i32, {i32});
declareExternFunction("time", i64, {ptr_type});
declareExternFunction("localtime", ptr_type, {ptr_type});
declareExternFunction("strftime", i64, {ptr_type, i64, ptr_type, ptr_type});
```

##### Functions to Implement

| Function | Implementation |
|----------|----------------|
| `getenv` | Call C `getenv`, return string or #f |
| `setenv` | Call C `setenv(name, value, 1)`, return bool |
| `unsetenv` | Call C `unsetenv`, return bool |
| `command-line` | Return cached list from main's argc/argv |
| `program-name` | Return argv[0] |
| `exit` | Call C `exit` |
| `system` | Call C `system`, return exit code |
| `sleep` | Call `usleep(seconds * 1000000)` |
| `current-seconds` | Call `time(NULL)` |
| `format-time` | Call `strftime` with format string |

##### Command Line Argument Handling

Need to capture argc/argv at program start:

```cpp
// In main wrapper generation:
GlobalVariable* g_argc = new GlobalVariable(*module, i32, false,
    GlobalValue::ExternalLinkage, nullptr, "__eshkol_argc");
GlobalVariable* g_argv = new GlobalVariable(*module, ptr_type, false,
    GlobalValue::ExternalLinkage, nullptr, "__eshkol_argv");

// In main function:
// Store argc and argv to globals
builder->CreateStore(argc_arg, g_argc);
builder->CreateStore(argv_arg, g_argv);
```

---

### Phase 8: Concurrency

**Priority**: Medium
**Dependencies**: Phase 9 (error handling for thread safety)
**Estimated LOC**: 1500

#### All C/LLVM Implementation

##### New Type Definition

**Note**: With the current 4-bit type field (max value 15), all slots are used once
HASH_PTR (14) and SOCKET_PTR (15) are added. Thread support has two options:

1. **Expand type field to 5 bits** (max 31 types) - requires runtime struct change
2. **Encode threads as closures** - threads store a CLOSURE_PTR with thread metadata

For option 1, add to eshkol_value_type_t:
```c
ESHKOL_VALUE_THREAD_PTR = 16,  // Requires expanding type field beyond 4 bits

// Thread structure
typedef struct eshkol_thread {
    pthread_t handle;
    eshkol_tagged_value_t result;
    bool completed;
    bool joined;
} eshkol_thread_t;

// Mutex structure
typedef struct eshkol_mutex {
    pthread_mutex_t handle;
} eshkol_mutex_t;

// Channel structure (bounded queue)
typedef struct eshkol_channel {
    eshkol_tagged_value_t* buffer;
    size_t capacity;
    size_t head;
    size_t tail;
    size_t count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    bool closed;
} eshkol_channel_t;
```

##### Thread Safety Considerations

1. **Arena Memory**: Each thread needs its own arena or global arena needs locking
2. **Global Variables**: REPL mode globals need mutex protection
3. **Symbol Tables**: Read-mostly, may need RW locks

##### Implementation Approach

```cpp
// Thread wrapper function that invokes Eshkol closure
void* eshkol_thread_wrapper(void* arg) {
    eshkol_thread_context_t* ctx = (eshkol_thread_context_t*)arg;

    // Set up thread-local arena
    arena_t* thread_arena = arena_create(8192);

    // Call the thunk (closure with no arguments)
    eshkol_tagged_value_t result = ctx->thunk();

    // Store result
    ctx->thread->result = result;
    ctx->thread->completed = true;

    // Cleanup
    arena_destroy(thread_arena);

    return NULL;
}
```

##### Functions to Implement

| Function | Implementation |
|----------|----------------|
| `thread-spawn` | Create pthread, pass closure |
| `thread-join` | pthread_join, return result |
| `thread-current` | pthread_self wrapper |
| `thread-yield` | sched_yield |
| `thread-sleep` | usleep |
| `make-mutex` | pthread_mutex_init |
| `mutex-lock` | pthread_mutex_lock |
| `mutex-unlock` | pthread_mutex_unlock |
| `mutex-try-lock` | pthread_mutex_trylock |
| `make-channel` | Allocate channel structure |
| `channel-send` | Lock, wait if full, enqueue, signal |
| `channel-recv` | Lock, wait if empty, dequeue, signal |
| `channel-try-recv` | Non-blocking receive |
| `channel-close` | Set closed flag, signal waiters |
| `atomic-box` | Create atomic container |
| `atomic-ref` | Atomic load |
| `atomic-set!` | Atomic store |
| `atomic-cas!` | Compare-and-swap |

---

### Phase 9: Error Handling

**Priority**: Medium
**Dependencies**: None (but threads need this)
**Estimated LOC**: 800

#### Mixed Implementation

##### C/LLVM Core (setjmp/longjmp)

```c
// Exception handler stack
typedef struct eshkol_exception_handler {
    jmp_buf jump_buffer;
    eshkol_tagged_value_t condition;
    bool has_condition;
    struct eshkol_exception_handler* next;
} eshkol_exception_handler_t;

// Thread-local handler stack
__thread eshkol_exception_handler_t* current_handler = NULL;
```

##### Codegen for try/raise

```cpp
// try: (try expr handler)
Value* codegenTry(const eshkol_operations_t* op) {
    // 1. Push exception handler onto stack
    Value* handler = builder->CreateCall(push_exception_handler_func, {});

    // 2. Call setjmp
    Value* jmp_result = builder->CreateCall(setjmp_func, {handler_jmp_buf});

    // 3. Branch: if jmp_result == 0, execute expr
    //           else, execute handler with condition

    BasicBlock* normal_block = BasicBlock::Create(*context, "try_normal", current_func);
    BasicBlock* exception_block = BasicBlock::Create(*context, "try_exception", current_func);
    BasicBlock* merge_block = BasicBlock::Create(*context, "try_merge", current_func);

    Value* is_normal = builder->CreateICmpEQ(jmp_result, ConstantInt::get(i32, 0));
    builder->CreateCondBr(is_normal, normal_block, exception_block);

    // Normal execution
    builder->SetInsertPoint(normal_block);
    Value* expr_result = codegenAST(&op->call_op.variables[0]);
    builder->CreateCall(pop_exception_handler_func, {});
    builder->CreateBr(merge_block);

    // Exception handling
    builder->SetInsertPoint(exception_block);
    Value* condition = builder->CreateCall(get_current_condition_func, {});
    // Call handler with condition
    Value* handler_result = ...; // Apply handler closure to condition
    builder->CreateBr(merge_block);

    // Merge
    builder->SetInsertPoint(merge_block);
    PHINode* result = builder->CreatePHI(tagged_value_type, 2);
    result->addIncoming(expr_result, normal_block);
    result->addIncoming(handler_result, exception_block);

    return result;
}

// raise: (raise condition)
Value* codegenRaise(const eshkol_operations_t* op) {
    Value* condition = codegenAST(&op->call_op.variables[0]);

    // Store condition in handler
    builder->CreateCall(set_current_condition_func, {condition});

    // longjmp to handler
    Value* handler = builder->CreateCall(get_current_handler_func, {});
    builder->CreateCall(longjmp_func, {handler_jmp_buf, ConstantInt::get(i32, 1)});

    // Unreachable (longjmp doesn't return)
    builder->CreateUnreachable();
    return nullptr;
}
```

##### Pure Eshkol Utilities (lib/core/errors.esk)

```scheme
;;; lib/core/errors.esk - Error handling utilities
;;; Part of the core library (auto-loaded)

(define-module core.errors
  (export make-condition condition? condition-type condition-message
          condition-irritants make-error-condition make-type-error-condition
          make-io-error-condition error assert guard with-exception-handler
          ignore-errors type-name)

;; Condition types (represented as tagged lists)
(define (make-condition type message irritants)
  (list 'condition type message irritants))

(define (condition? obj)
  (and (pair? obj) (eq? (car obj) 'condition)))

(define (condition-type cond)
  (cadr cond))

(define (condition-message cond)
  (caddr cond))

(define (condition-irritants cond)
  (cadddr cond))

;; Common condition constructors
(define (make-error-condition message . irritants)
  (make-condition 'error message irritants))

(define (make-type-error-condition expected got)
  (make-condition 'type-error
                 (string-append "Expected " expected ", got "
                               (type-name got))
                 (list got)))

(define (make-io-error-condition message path)
  (make-condition 'io-error message (list path)))

;; Error helper (raises error condition)
(define (error message . irritants)
  (raise (apply make-error-condition message irritants)))

;; Assert helper
(define (assert condition message)
  (unless condition
    (error message)))

;; Guard macro-like function
;; (guard handler body)
(define (guard handler thunk)
  (try (thunk)
       (lambda (condition)
         (handler condition))))

;; with-exception-handler (R7RS style)
(define (with-exception-handler handler thunk)
  (guard handler thunk))

;; ignore-errors: Execute, return #f on error
(define (ignore-errors thunk)
  (try (thunk)
       (lambda (c) #f)))

;; type-name: Get type name string
(define (type-name value)
  (cond
    ((null? value) "null")
    ((boolean? value) "boolean")
    ((integer? value) "integer")
    ((real? value) "real")
    ((string? value) "string")
    ((symbol? value) "symbol")
    ((pair? value) "pair")
    ((vector? value) "vector")
    ((procedure? value) "procedure")
    (else "unknown")))

) ; end define-module core.errors
```

---

## 5. Implementation Strategy

### Development Workflow

```
1. Design
   └── Define function signatures
   └── Write documentation
   └── Create test cases

2. Implement
   └── C/LLVM primitives first
   └── Pure Eshkol libraries second
   └── Follow existing code patterns

3. Test
   └── Run unit tests
   └── Run integration tests
   └── Test in REPL

4. Document
   └── Update function reference
   └── Add examples
   └── Update this plan

5. Release
   └── Merge to main branch
   └── Tag version
   └── Update changelog
```

### Code Organization

```
lib/backend/llvm_codegen.cpp:
  - Add dispatch entries in codegenFunctionCall()
  - Add codegen methods grouped by category
  - Follow naming: codegen<Category><Function>

lib/core/arena_memory.h/.cpp:
  - Add new structures at end of file
  - Add allocation functions
  - Keep 16-byte alignment for tagged values

inc/eshkol/eshkol.h:
  - Add new value types to enum
  - Add helper macros

lib/*.esk:
  - One file per category
  - Export all public functions at top
  - Internal helpers with leading underscore
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| C codegen function | `codegen<Category><Name>` | `codegenStringSplit` |
| C struct | `eshkol_<name>_t` | `eshkol_hash_table_t` |
| C arena function | `arena_<action>_<type>` | `arena_allocate_hash_table` |
| Eshkol function | `kebab-case` | `string-split` |
| Eshkol predicate | `name?` | `hash-table?` |
| Eshkol mutator | `name!` | `hash-set!` |

---

## 6. File Structure

### Library Organization Philosophy

The Eshkol library is organized into three tiers:

1. **Core** (`lib/core/`): Fundamental utilities loaded by default
2. **Standard** (`lib/`): Common functionality, explicitly imported
3. **Extended** (`lib/ext/`): Domain-specific libraries

### Complete Library Structure

The library uses a hierarchical module naming convention that maps directly
to file paths. Module `data.json` corresponds to file `lib/data/json.esk`.

```
lib/
├── stdlib.esk                 # [EXISTING] Core standard library
├── math.esk                   # [EXISTING] Mathematical functions
│
├── core/                      # TIER 1: Auto-loaded (no import needed)
│   ├── strings.esk           # Module: core.strings
│   ├── collections.esk       # Module: core.collections
│   └── errors.esk            # Module: core.errors
│
├── data/                      # TIER 2: Data formats
│   ├── json.esk              # Module: data.json
│   ├── csv.esk               # Module: data.csv
│   ├── base64.esk            # Module: data.base64
│   └── xml.esk               # Module: data.xml [FUTURE]
│
├── io/                        # TIER 2: Input/Output
│   ├── files.esk             # Module: io.files
│   ├── paths.esk             # Module: io.paths
│   └── streams.esk           # Module: io.streams [FUTURE]
│
├── net/                       # TIER 2: Networking
│   ├── sockets.esk           # Module: net.sockets
│   ├── tcp.esk               # Module: net.tcp
│   ├── udp.esk               # Module: net.udp
│   └── dns.esk               # Module: net.dns
│
├── web/                       # TIER 2: Web protocols
│   ├── http.esk              # Module: web.http
│   ├── url.esk               # Module: web.url
│   ├── websocket.esk         # Module: web.websocket [FUTURE]
│   └── router.esk            # Module: web.router
│
├── system/                    # TIER 2: System interaction
│   ├── env.esk               # Module: system.env
│   ├── process.esk           # Module: system.process
│   └── time.esk              # Module: system.time
│
├── concurrent/                # TIER 2: Concurrency
│   ├── threads.esk           # Module: concurrent.threads
│   ├── channels.esk          # Module: concurrent.channels
│   ├── sync.esk              # Module: concurrent.sync
│   └── async.esk             # Module: concurrent.async [FUTURE]
│
└── ext/                       # TIER 3: Extended/Optional
    ├── ml/                   # Machine learning
    │   ├── nn.esk            # Module: ext.ml.nn
    │   ├── optim.esk         # Module: ext.ml.optim
    │   └── data.esk          # Module: ext.ml.data
    │
    ├── crypto/               # Cryptography [FUTURE]
    │   ├── hash.esk          # Module: ext.crypto.hash
    │   └── cipher.esk        # Module: ext.crypto.cipher
    │
    └── test/                  # Testing framework
        ├── assert.esk        # Module: ext.test.assert
        └── runner.esk        # Module: ext.test.runner
```

### Import Conventions (Per Language Specification Section 13)

Eshkol uses symbolic module names with dot notation, following the R7RS-inspired
module system defined in the language specification:

```scheme
;;; Tier 1: Auto-loaded (no import needed)
;; Core functions are available immediately:
;; string-split, string-join, hash-table, error, etc.

;;; Tier 2: Explicit import with symbolic module names
(import data.json)              ; json-parse, json-stringify
(import web.http)               ; http-get, http-serve
(import net.tcp)                ; tcp-connect, tcp-listen

;;; Tier 3: Extended libraries
(import ext.ml.nn)              ; nn-dense, nn-relu
(import ext.test.assert)        ; assert-equal, assert-true

;;; Selective imports (R7RS style)
(import (only data.json json-parse json-stringify))
(import (prefix net.tcp tcp:))  ; tcp:connect, tcp:listen
(import (rename data.csv (csv-parse parse-csv)))

;;; Multiple imports
(import data.json data.csv data.base64)
```

### Module Naming Convention

| Module Name | Prefix | Exported Functions |
|-------------|--------|---------|
| `data.json` | `json-` | `json-parse`, `json-stringify`, `json-get` |
| `data.csv` | `csv-` | `csv-parse`, `csv-stringify`, `csv-read-file` |
| `data.base64` | `base64-` | `base64-encode`, `base64-decode` |
| `web.http` | `http-` | `http-get`, `http-post`, `http-serve` |
| `web.url` | `url-` | `url-parse`, `url-encode`, `url-decode` |
| `web.router` | `router-` | `router-create`, `router-add-route` |
| `net.tcp` | `tcp-` | `tcp-connect`, `tcp-listen`, `tcp-send` |
| `net.udp` | `udp-` | `udp-socket`, `udp-bind`, `udp-send-to` |
| `io.files` | `file-` | `file-exists?`, `read-file`, `write-file` |
| `io.paths` | `path-` | `path-join`, `path-dirname`, `path-basename` |
| `system.env` | None | `getenv`, `setenv`, `unsetenv` |
| `system.process` | None | `system`, `exit`, `command-line` |
| `system.time` | None | `current-seconds`, `format-time`, `sleep` |
| `concurrent.threads` | `thread-` | `thread-spawn`, `thread-join` |
| `concurrent.channels` | `channel-` | `channel-send`, `channel-recv` |
| `concurrent.sync` | `mutex-` | `make-mutex`, `mutex-lock`, `mutex-unlock` |

### Test Structure

```
tests/
├── core/
│   ├── test_strings.esk
│   ├── test_collections.esk
│   └── test_errors.esk
│
├── data/
│   ├── test_json.esk
│   ├── test_csv.esk
│   └── test_base64.esk
│
├── io/
│   ├── test_files.esk
│   └── test_paths.esk
│
├── net/
│   ├── test_tcp.esk
│   └── test_udp.esk
│
├── web/
│   ├── test_http_client.esk
│   ├── test_http_server.esk
│   └── test_url.esk
│
├── system/
│   ├── test_env.esk
│   └── test_process.esk
│
├── concurrent/
│   ├── test_threads.esk
│   └── test_channels.esk
│
└── integration/
    ├── test_http_json.esk       # HTTP + JSON integration
    ├── test_file_csv.esk        # File I/O + CSV
    └── test_concurrent_net.esk  # Threads + Networking
```

### Auto-Loading Behavior

When `eshkol-run` starts, it automatically loads the core modules:

```scheme
;; These modules are loaded implicitly (no import needed):

;; 1. Core standard library
;;    compose, curry, sort, iota, partition, etc.

;; 2. Core extensions
;;    string-split, string-join, string-trim, etc.
;;    hash-table?, hash-ref, hash-set!, etc.
;;    error, make-condition, condition?, etc.

;; 3. Math library
;;    det, inv, solve, integrate, newton, etc.
```

User code imports additional modules explicitly:

```scheme
;;; Example: Web API application
(define-module my-app.server
  (import web.http)
  (import data.json)
  (import io.files)
  (export start-server)

  (define (handle-request req)
    (let* ((path (http-request-path req))
           (data (json-parse (read-file "data.json"))))
      (json-stringify (json-get data path))))

  (define (start-server port)
    (http-serve port handle-request)))

;; Start the server
(start-server 8080)
```

Simple scripts can also use imports without defining a module:

```scheme
;;; Example: Simple script
(import data.json)
(import web.http)

(define response (http-get "https://api.example.com/data"))
(define data (json-parse (http-response-body response)))
(display (json-get data "name"))
```

### Files to Modify

```
inc/eshkol/eshkol.h
  - Add ESHKOL_VALUE_HASH_PTR, SOCKET_PTR, THREAD_PTR
  - Add structure forward declarations

lib/core/arena_memory.h
  - Add hash table structures
  - Add socket structures
  - Add thread structures
  - Add allocation function declarations

lib/core/arena_memory.cpp
  - Implement allocation functions

lib/backend/llvm_codegen.cpp
  - Add ~72 new codegen functions
  - Add dispatch entries
  - Add C library function declarations

exe/eshkol-run.cpp
  - Load new library files in stdlib
  - Store argc/argv for command-line access
```

---

## 7. Testing Strategy

### Test Categories

1. **Unit Tests**: Individual function tests
2. **Integration Tests**: Feature combination tests
3. **Regression Tests**: Prevent breakage
4. **Performance Tests**: Benchmark critical paths

### Test File Format

```scheme
;;; test_<category>_<feature>.esk

(define (test-<name>)
  (let ((result (<function> <args>)))
    (if (equal? result <expected>)
        (begin (display "PASS: <name>\n") #t)
        (begin (display "FAIL: <name>\n")
               (display "  Expected: ") (display <expected>) (newline)
               (display "  Got: ") (display result) (newline)
               #f))))

(define (run-all-tests)
  (let ((results (list
                  (test-<name1>)
                  (test-<name2>)
                  ...)))
    (let ((passed (length (filter identity results)))
          (total (length results)))
      (display "Results: ")
      (display passed)
      (display "/")
      (display total)
      (display " passed\n")
      (= passed total))))

(run-all-tests)
```

### Example Test: string-split

```scheme
;;; tests/strings/test_string_split.esk

(define (test-split-basic)
  (equal? (string-split "a,b,c" ",") '("a" "b" "c")))

(define (test-split-no-delimiter)
  (equal? (string-split "abc" ",") '("abc")))

(define (test-split-empty)
  (equal? (string-split "" ",") '("")))

(define (test-split-consecutive)
  (equal? (string-split "a,,b" ",") '("a" "" "b")))

(define (test-split-multi-char)
  (equal? (string-split "a<>b<>c" "<>") '("a" "b" "c")))

(define (run-all-tests)
  (and (test-split-basic)
       (test-split-no-delimiter)
       (test-split-empty)
       (test-split-consecutive)
       (test-split-multi-char)))
```

---

## 8. Dependencies

### External Libraries Required

| Library | Purpose | Platform |
|---------|---------|----------|
| LLVM | Code generation | All |
| pthreads | Concurrency | POSIX |
| BSD sockets | Networking | POSIX |
| C stdlib | Various | All |

### No External Dependencies Added

All new features use:
- POSIX standard APIs (available on Linux, macOS, BSD)
- C standard library
- LLVM (already required)

Windows support would need:
- Winsock2 instead of BSD sockets
- Windows threads instead of pthreads
- Different path handling

---

## 9. Risk Assessment

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Thread safety in arena | High | Per-thread arenas or careful locking |
| Complex HTTP parsing | Medium | Start simple, iterate |
| setjmp/longjmp portability | Low | Standard C, widely supported |
| Performance of pure Eshkol | Medium | Profile, move hot paths to C if needed |

### Schedule Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scope creep | High | Strict phase boundaries |
| Testing overhead | Medium | Automate tests early |
| Integration issues | Medium | Continuous integration |

### Dependency Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Phase dependencies | Medium | Implement in order |
| Breaking changes | Low | Maintain backward compatibility |

---

## 10. Success Metrics

### Functional Requirements

After all phases complete, Eshkol must be able to:

- [ ] Split and join strings
- [ ] Parse and generate JSON
- [ ] Parse and generate CSV
- [ ] Use hash tables with O(1) operations
- [ ] Read and write files
- [ ] List and navigate directories
- [ ] Open TCP and UDP sockets
- [ ] Make HTTP requests
- [ ] Serve HTTP responses
- [ ] Access environment variables
- [ ] Spawn and join threads
- [ ] Use channels for message passing
- [ ] Handle exceptions with try/raise

### Performance Requirements

| Operation | Target |
|-----------|--------|
| Hash table lookup | < 1μs |
| JSON parse 1KB | < 1ms |
| HTTP request | < 100ms (network dependent) |
| Thread spawn | < 1ms |
| Channel send/recv | < 10μs |

### Quality Requirements

- All existing tests continue to pass
- New features have >90% test coverage
- No memory leaks in new code
- Documentation for all new functions

---

## Appendix A: Function Quick Reference

### Phase 1: Strings (12 functions)
```
string-split string-join string-trim string-trim-left string-trim-right
string-replace string-contains? string-index string-upcase string-downcase
string-reverse string-copy
```

### Phase 2: Data Formats (10 functions)
```
json-parse json-stringify json-get json-get-in
csv-parse csv-stringify csv-read-file csv-write-file
base64-encode base64-decode
```

### Phase 3: Hash Tables (10 functions)
```
make-hash-table hash-table? hash-ref hash-set! hash-remove!
hash-has-key? hash-keys hash-values hash-count hash->alist alist->hash
```

### Phase 4: File System (15 functions)
```
file-exists? file-readable? file-writable? file-delete file-rename file-size
directory-exists? make-directory delete-directory directory-list
current-directory set-current-directory!
read-file write-file append-file
```

### Phase 5: Networking (16 functions)
```
tcp-connect tcp-listen tcp-accept tcp-send tcp-recv tcp-close
udp-socket udp-bind udp-send-to udp-recv-from
socket-set-timeout socket-set-nonblocking socket-set-reuseaddr
resolve-hostname socket-local-address socket-peer-address
```

### Phase 6: HTTP (15 functions)
```
http-get http-post http-request
http-response-status http-response-headers http-response-body http-response-json
http-serve http-request-method http-request-path http-request-headers http-request-body
url-parse url-encode url-decode
```

### Phase 7: System (10 functions)
```
getenv setenv unsetenv
command-line program-name
exit system sleep current-seconds format-time
```

### Phase 8: Concurrency (15 functions)
```
thread-spawn thread-join thread-current thread-yield thread-sleep
make-mutex mutex-lock mutex-unlock mutex-try-lock
make-channel channel-send channel-recv channel-try-recv channel-close
atomic-box atomic-ref atomic-set! atomic-cas!
```

### Phase 9: Error Handling (8 functions)
```
try raise error
make-condition condition? condition-type condition-message condition-irritants
```

**Total: 108 new functions**

---

*Document Version: 1.0*
*Last Updated: December 2025*
