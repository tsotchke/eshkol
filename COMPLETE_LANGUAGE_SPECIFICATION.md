# Eshkol Language - Complete Technical Specification

**Version:** 1.0.0-foundation  
**Generated:** 2025-12-12  
**Status:** Comprehensive implementation documentation from source code

---

## Table of Contents

1. [Language Overview](#1-language-overview)
2. [Data Types](#2-data-types)
3. [Syntax and Special Forms](#3-syntax-and-special-forms)
4. [Built-in Operators and Functions](#4-built-in-operators-and-functions)
5. [Standard Library](#5-standard-library)
6. [Memory Management System](#6-memory-management-system)
7. [Type System](#7-type-system)
8. [Automatic Differentiation](#8-automatic-differentiation)
9. [Module System](#9-module-system)
10. [Compilation Model](#10-compilation-model)
11. [REPL and JIT](#11-repl-and-jit)
12. [Advanced Features](#12-advanced-features)
13. [Runtime Architecture](#13-runtime-architecture)

---

## 1. Language Overview

### 1.1 What is Eshkol?

Eshkol is a **Scheme dialect** with extensions for scientific computing, automatic differentiation, and neural network development. It combines:

- **Scheme R7RS compatibility** (subset)
- **Automatic Differentiation** (forward-mode and reverse-mode)
- **HoTT-inspired Type System** (Homotopy Type Theory)
- **Arena Memory Management** (OALR - Ownership-Aware Lexical Regions)
- **LLVM Backend** for native code generation
- **JIT REPL** for interactive development
- **Quantum-inspired RNG** for high-quality randomness

### 1.2 Core Philosophy

1. **Code as Data** (Homoiconicity) - Lambdas can be displayed as their source S-expressions
2. **Gradual Typing** - Optional type annotations with inference
3. **Memory Safety** - Arena allocation with ownership tracking
4. **Performance** - Native compilation via LLVM
5. **Scientific Computing** - First-class support for tensors and automatic differentiation

---

## 2. Data Types

### 2.1 Immediate Value Types

These types store data directly in the `eshkol_tagged_value_t` struct (no heap allocation):

#### 2.1.1 `NULL` (Empty List)
- **Type Tag:** `ESHKOL_VALUE_NULL` (0)
- **Syntax:** `'()` or `()`
- **Description:** Represents the empty list / nil value
- **Example:** `(null? '())` => `#t`

#### 2.1.2 `INT64` (Exact Integer)
- **Type Tag:** `ESHKOL_VALUE_INT64` (1)
- **Syntax:** `42`, `-17`, `0`
- **Range:** -2^63 to 2^63-1
- **Exactness:** Exact numbers (preserves exactness in Scheme numeric tower)
- **Example:** `(+ 5 10)` => `15`

#### 2.1.3 `DOUBLE` (Inexact Real)
- **Type Tag:** `ESHKOL_VALUE_DOUBLE` (2)
- **Syntax:** `3.14`, `1.0`, `2.5e-10`
- **Precision:** IEEE 754 double-precision (64-bit)
- **Exactness:** Inexact numbers
- **Scientific Notation:** Supports `e` or `E` notation
- **Example:** `(+ 1.5 2.5)` => `4.0`

#### 2.1.4 `BOOL` (Boolean)
- **Type Tag:** `ESHKOL_VALUE_BOOL` (3)
- **Syntax:** `#t` (true), `#f` (false)
- **Truthiness:** Only `#f` is false; all other values are truthy
- **Example:** `(if #t 1 2)` => `1`

#### 2.1.5 `CHAR` (Character)
- **Type Tag:** `ESHKOL_VALUE_CHAR` (4)
- **Syntax:** `#\a`, `#\space`, `#\newline`, `#\tab`, `#\xHHHH`
- **Encoding:** Unicode codepoint (stored as int64)
- **Named Characters:** `#\space`, `#\newline`, `#\tab`, `#\return`
- **Example:** `(char->integer #\A)` => `65`

#### 2.1.6 `SYMBOL` (Interned Symbol)
- **Type Tag:** `ESHKOL_VALUE_SYMBOL` (5)
- **Syntax:** `'foo`, `'hello-world`, `'+`
- **Description:** Interned string used for identifiers
- **Comparison:** Uses pointer equality (fast)
- **Example:** `(symbol? 'x)` => `#t`

#### 2.1.7 `DUAL_NUMBER` (Forward-mode AD)
- **Type Tag:** `ESHKOL_VALUE_DUAL_NUMBER` (6)
- **Structure:** `{ double value; double derivative; }`
- **Size:** 16 bytes
- **Usage:** Internal representation for forward-mode automatic differentiation
- **Example:** Used internally by `derivative` operator

### 2.2 Heap Pointer Types (Consolidated)

These types use the consolidated `HEAP_PTR` (8) or `CALLABLE` (9) tags with subtype information in object headers.

#### 2.2.1 `HEAP_PTR` - Consolidated Heap Objects (Type 8)

All heap-allocated data structures share type tag 8 with subtype stored in 8-byte header:

**Header Structure** (`eshkol_object_header_t`):
```c
struct {
    uint8_t  subtype;      // Distinguishes cons, string, vector, etc.
    uint8_t  flags;        // GC marks, linear status, etc.
    uint16_t ref_count;    // Reference count (0 = not ref-counted)
    uint32_t size;         // Object size in bytes
}
```

**Subtypes:**
- `HEAP_SUBTYPE_CONS` (0) - Cons cell / pair
- `HEAP_SUBTYPE_STRING` (1) - UTF-8 string
- `HEAP_SUBTYPE_VECTOR` (2) - Heterogeneous vector
- `HEAP_SUBTYPE_TENSOR` (3) - N-dimensional numeric tensor
- `HEAP_SUBTYPE_MULTI_VALUE` (4) - Multiple return values container
- `HEAP_SUBTYPE_HASH` (5) - Hash table
- `HEAP_SUBTYPE_EXCEPTION` (6) - Exception object
- `HEAP_SUBTYPE_RECORD` (7) - User-defined record
- `HEAP_SUBTYPE_BYTEVECTOR` (8) - Raw byte vector (R7RS)
- `HEAP_SUBTYPE_PORT` (9) - I/O port

##### CONS (Pair/List Node)
- **Subtype:** `HEAP_SUBTYPE_CONS` (0)
- **Structure:** `arena_tagged_cons_cell_t` (32 bytes)
  - `car`: Complete `eshkol_tagged_value_t` (16 bytes)
  - `cdr`: Complete `eshkol_tagged_value_t` (16 bytes)
- **Syntax:** `(cons 1 2)`, `'(1 2 3)`
- **Operations:** `cons`, `car`, `cdr`, `set-car!`, `set-cdr!`
- **Features:**
  - Mixed-type lists (can hold any combination of types)
  - Proper and improper lists supported
  - Dotted pairs: `(1 . 2)`
- **Example:** `(cons 1 (cons 2 '()))` => `(1 2)`

##### STRING
- **Subtype:** `HEAP_SUBTYPE_STRING` (1)
- **Encoding:** UTF-8 with null terminator
- **Syntax:** `"hello"`, `"multi\nline"`
- **Escape Sequences:** `\n`, `\t`, `\r`, `\\`, `\"`
- **Operations:** `string-length`, `string-ref`, `string-set!`, `string-append`, `substring`
- **Example:** `(string-append "hello" " " "world")` => `"hello world"`

##### VECTOR
- **Subtype:** `HEAP_SUBTYPE_VECTOR` (2)
- **Structure:** 
  - Offset 0: `int64_t length` (8 bytes)
  - Offset 8: Array of `eshkol_tagged_value_t` elements (16 bytes each)
- **Syntax:** `#(1 2 3)`, `(vector 1 2 3)`, `(make-vector 5 0)`
- **Features:**
  - Heterogeneous (can hold any type mix)
  - O(1) random access
  - Mutable with `vector-set!`
- **Operations:** `vector`, `make-vector`, `vector-length`, `vector-ref`, `vector-set!`
- **Example:** `(vector-ref #(10 20 30) 1)` => `20`

##### TENSOR
- **Subtype:** `HEAP_SUBTYPE_TENSOR` (3)
- **Structure:** `eshkol_tensor_t` (32 bytes)
  ```c
  struct {
      uint64_t* dimensions;      // Dimension sizes array
      uint64_t  num_dimensions;  // Rank (number of dimensions)
      int64_t*  elements;        // Element data (doubles as int64 bits)
      uint64_t  total_elements;  // Product of all dimensions
  }
  ```
- **Syntax:** 
  - 1D: `#(1.0 2.0 3.0)` or `(vector 1.0 2.0 3.0)`
  - Matrix: `(matrix 2 3 1.0 2.0 3.0 4.0 5.0 6.0)`
  - Generic: `(tensor dims... elements...)`
- **Features:**
  - Homogeneous numeric storage (doubles stored as int64 bit patterns)
  - N-dimensional (vectors, matrices, 3D+ arrays)
  - Efficient element-wise operations
  - Support for autodiff
- **Operations:** `tensor`, `matrix`, `vector`, `vref`, `tensor-get`, `tensor-set`, `tensor-add`, `tensor-mul`, `tensor-dot`, `matmul`, `transpose`, `reshape`
- **Example:** `(matmul W1 W2)` multiplies two matrices

##### HASH_TABLE
- **Subtype:** `HEAP_SUBTYPE_HASH` (5)
- **Structure:** `eshkol_hash_table_t`
  ```c
  struct {
      size_t capacity;                // Bucket count
      size_t size;                    // Entry count
      size_t tombstones;              // Deleted entries
      eshkol_tagged_value_t* keys;    // Key array
      eshkol_tagged_value_t* values;  // Value array
      uint8_t* status;                // Entry status (empty/occupied/deleted)
  }
  ```
- **Implementation:** Open addressing with linear probing
- **Load Factor:** 0.75 (rehashes when exceeded)
- **Initial Capacity:** 16 buckets
- **Syntax:** `(make-hash-table)`, `(hash key1 val1 key2 val2 ...)`
- **Operations:** `hash-set!`, `hash-ref`, `hash-has-key?`, `hash-remove!`, `hash-keys`, `hash-values`, `hash-count`, `hash-clear!`
- **Example:** 
  ```scheme
  (define h (hash 'a 1 'b 2))
  (hash-ref h 'a)  ; => 1
  ```

##### EXCEPTION
- **Subtype:** `HEAP_SUBTYPE_EXCEPTION` (6)
- **Structure:** `eshkol_exception_t`
  ```c
  struct {
      eshkol_exception_type_t type;   // Exception type code
      char* message;                   // Error message
      eshkol_tagged_value_t* irritants; // Irritant values
      uint32_t num_irritants;
      uint32_t line;                   // Source location
      uint32_t column;
      char* filename;
  }
  ```
- **Types:**
  - `ESHKOL_EXCEPTION_ERROR` - Generic error
  - `ESHKOL_EXCEPTION_TYPE_ERROR` - Type mismatch
  - `ESHKOL_EXCEPTION_FILE_ERROR` - File operation failed
  - `ESHKOL_EXCEPTION_READ_ERROR` - Read/parse error
  - `ESHKOL_EXCEPTION_SYNTAX_ERROR` - Syntax error
  - `ESHKOL_EXCEPTION_RANGE_ERROR` - Index out of bounds
  - `ESHKOL_EXCEPTION_ARITY_ERROR` - Wrong argument count
  - `ESHKOL_EXCEPTION_DIVIDE_BY_ZERO` - Division by zero
  - `ESHKOL_EXCEPTION_USER_DEFINED` - User-defined exception
- **Operations:** `guard`, `raise`
- **Example:**
  ```scheme
  (guard (e ((error? e) (display "Caught error")))
    (error "Something went wrong"))
  ```

#### 2.2.2 `CALLABLE` - Consolidated Callable Objects (Type 9)

All callable objects share type tag 9 with subtype in header:

**Subtypes:**
- `CALLABLE_SUBTYPE_CLOSURE` (0) - Compiled closure with environment
- `CALLABLE_SUBTYPE_LAMBDA_SEXPR` (1) - Lambda as data (homoiconic)
- `CALLABLE_SUBTYPE_AD_NODE` (2) - Autodiff computation node
- `CALLABLE_SUBTYPE_PRIMITIVE` (3) - Built-in primitive
- `CALLABLE_SUBTYPE_CONTINUATION` (4) - First-class continuation

##### CLOSURE
- **Subtype:** `CALLABLE_SUBTYPE_CLOSURE` (0)
- **Structure:** `eshkol_closure_t`
  ```c
  struct {
      uint64_t func_ptr;              // Pointer to lambda function
      eshkol_closure_env_t* env;      // Captured environment
      uint64_t sexpr_ptr;             // S-expression for display
      uint8_t return_type;            // Return type category
      uint8_t input_arity;            // Expected argument count
      uint8_t flags;                  // Variadic flag, etc.
      uint8_t reserved;
      uint32_t hott_type_id;          // HoTT type ID
  }
  ```
- **Environment Structure:** `eshkol_closure_env_t`
  ```c
  struct {
      size_t num_captures;            // Packed: captures | (fixed_params << 16) | (is_variadic << 63)
      eshkol_tagged_value_t captures[]; // Flexible array of captured values
  }
  ```
- **Features:**
  - Lexical scoping with variable capture
  - First-class functions
  - Homoiconic display (shows lambda source)
  - Variadic support
- **Example:**
  ```scheme
  (define (make-adder n)
    (lambda (x) (+ x n)))  ; Captures n
  ((make-adder 5) 10)  ; => 15
  ```

##### AD_NODE (Computational Graph Node)
- **Subtype:** `CALLABLE_SUBTYPE_AD_NODE` (2)
- **Structure:** `ad_node_t`
  ```c
  struct {
      ad_node_type_t type;   // Operation type (ADD, MUL, SIN, etc.)
      double value;          // Forward pass result
      double gradient;       // Backward pass gradient
      ad_node* input1;       // First parent node
      ad_node* input2;       // Second parent node
      size_t id;             // Node ID for topological sort
  }
  ```
- **Node Types:**
  - `AD_NODE_CONSTANT` - Leaf node (constant value)
  - `AD_NODE_VARIABLE` - Input variable
  - `AD_NODE_ADD`, `AD_NODE_SUB`, `AD_NODE_MUL`, `AD_NODE_DIV`
  - `AD_NODE_SIN`, `AD_NODE_COS`, `AD_NODE_EXP`, `AD_NODE_LOG`
  - `AD_NODE_POW`, `AD_NODE_NEG`
- **Usage:** Internal representation for reverse-mode automatic differentiation
- **Associated Structure:** `ad_tape_t` for recording computation graphs

### 2.3 Legacy Pointer Types (Backward Compatibility)

These types (32+) are deprecated but maintained for display system compatibility:

- `ESHKOL_VALUE_CONS_PTR` (32) - Use `HEAP_PTR` + `HEAP_SUBTYPE_CONS`
- `ESHKOL_VALUE_STRING_PTR` (33) - Use `HEAP_PTR` + `HEAP_SUBTYPE_STRING`
- `ESHKOL_VALUE_VECTOR_PTR` (34) - Use `HEAP_PTR` + `HEAP_SUBTYPE_VECTOR`
- `ESHKOL_VALUE_TENSOR_PTR` (35) - Use `HEAP_PTR` + `HEAP_SUBTYPE_TENSOR`
- `ESHKOL_VALUE_HASH_PTR` (36) - Use `HEAP_PTR` + `HEAP_SUBTYPE_HASH`
- `ESHKOL_VALUE_CLOSURE_PTR` (38) - Use `CALLABLE` + `CALLABLE_SUBTYPE_CLOSURE`
- `ESHKOL_VALUE_LAMBDA_SEXPR` (39) - Use `CALLABLE` + `CALLABLE_SUBTYPE_LAMBDA_SEXPR`
- `ESHKOL_VALUE_AD_NODE_PTR` (40) - Use `CALLABLE` + `CALLABLE_SUBTYPE_AD_NODE`

### 2.4 Future Multimedia Types (Reserved)

These types (16-19) are reserved for future multimedia extensions:

#### HANDLE (Type 16)
- Managed resource handles (windows, contexts, devices)
- Subtypes include: window, GL context, audio device, MIDI port, camera, socket, etc.

#### BUFFER (Type 17)
- Typed data buffers for zero-copy transfer
- Subtypes include: raw bytes, image pixels, audio samples, GPU buffers

#### STREAM (Type 18)
- Lazy data streams
- Subtypes include: byte stream, audio, video, MIDI, network, transform pipeline

#### EVENT (Type 19)
- Real-time event handling
- Subtypes include: input, window, audio, MIDI, timer, network, custom

### 2.5 Tagged Value Runtime Representation

All values at runtime are represented as `eshkol_tagged_value_t` (16 bytes):

```c
struct eshkol_tagged_value {
    uint8_t type;        // Value type tag (0-255)
    uint8_t flags;       // Exactness, special flags
    uint16_t reserved;   // Future use
    union {
        int64_t int_val;     // Integer, bool, char, symbol, pointers
        double double_val;   // Floating-point values
        uint64_t ptr_val;    // Heap pointers
        uint64_t raw_val;    // Raw manipulation
    } data;
}
```

---

## 3. Syntax and Special Forms

### 3.1 Comments

```scheme
; Line comment - from semicolon to end of line
;; Convention: double semicolon for important comments
;;; Convention: triple semicolon for section headers
```

### 3.2 Literals

#### Numeric Literals
- **Integer:** `42`, `-17`, `0`, `+99`
- **Floating-point:** `3.14`, `1.0`, `-2.5`
- **Scientific notation:** `1.5e10`, `2.3E-5`, `1e4`

#### String Literals
- **Basic:** `"hello world"`
- **Escape sequences:**
  - `\n` - newline
  - `\t` - tab
  - `\r` - carriage return
  - `\\` - backslash
  - `\"` - double quote

#### Character Literals
- **Simple:** `#\a`, `#\Z`, `#\0`
- **Named:** `#\space`, `#\newline`, `#\tab`, `#\return`
- **Unicode:** `#\xHHHH` (hexadecimal codepoint)

#### Boolean Literals
- **True:** `#t`
- **False:** `#f`

#### List Literals
- **Quoted list:** `'(1 2 3)`
- **Empty list:** `'()` or `()`
- **Dotted pair:** `'(1 . 2)`
- **Quote shorthand:** `'x` equivalent to `(quote x)`

#### Vector Literals
- **Vector syntax:** `#(1 2 3)`
- **Mixed types:** `#(1 "two" #t)`

### 3.3 Variable Definition and Binding

#### 3.3.1 `define` - Variable Definition
**Syntax:**
```scheme
(define name value)                    ; Variable
(define (name param...) body...)       ; Function
(define (name param... . rest) body)   ; Variadic function
```

**Type Annotations:**
```scheme
(define (f (x : int) (y : real)) : real body)  ; Typed parameters and return
(define (g x y) : int body)                     ; Return type only
```

**Examples:**
```scheme
(define x 42)
(define (square n) (* n n))
(define (variadic-sum . args) (fold + 0 args))
```

#### 3.3.2 `lambda` - Anonymous Functions
**Syntax:**
```scheme
(lambda (params...) body...)           ; Regular
(lambda (params... . rest) body)       ; Variadic
(lambda args body)                     ; All args as list
(lambda ((x : int) (y : real)) body)   ; Typed parameters
(lambda (params) : type body)          ; Return type annotation
```

**Examples:**
```scheme
(lambda (x) (* x x))
(lambda (x y) (+ x y))
(lambda args (fold + 0 args))
```

**Features:**
- Lexical scoping
- Automatic closure creation for captured variables
- Homoiconic (can be displayed as source)
- First-class (can be passed, returned, stored)

#### 3.3.3 `let` - Local Bindings (Parallel)
**Syntax:**
```scheme
(let ((var1 val1) (var2 val2) ...) body...)
(let name ((var init) ...) body...)  ; Named let (loop construct)
```

**Semantics:** Bindings evaluated in parallel (cannot reference each other)

**Examples:**
```scheme
(let ((x 5) (y 10))
  (+ x y))  ; => 15

; Named let for looping
(let loop ((n 10) (acc 0))
  (if (= n 0)
      acc
      (loop (- n 1) (+ acc n))))  ; => 55
```

#### 3.3.4 `let*` - Sequential Bindings
**Syntax:**
```scheme
(let* ((var1 val1) (var2 val2) ...) body...)
```

**Semantics:** Bindings evaluated left-to-right (later bindings can reference earlier ones)

**Example:**
```scheme
(let* ((x 5)
       (y (+ x 1)))  ; y can use x
  (+ x y))  ; => 11
```

#### 3.3.5 `letrec` - Recursive Bindings
**Syntax:**
```scheme
(letrec ((var1 val1) (var2 val2) ...) body...)
```

**Semantics:** All bindings visible to all values (supports mutual recursion)

**Example:**
```scheme
(letrec ((even? (lambda (n)
                  (if (= n 0) #t (odd? (- n 1)))))
         (odd? (lambda (n)
                 (if (= n 0) #f (even? (- n 1))))))
  (even? 42))  ; => #t
```

#### 3.3.6 `set!` - Mutation
**Syntax:**
```scheme
(set! variable new-value)
```

**Example:**
```scheme
(define x 10)
(set! x 20)
x  ; => 20
```

### 3.4 Control Flow

#### 3.4.1 `if` - Conditional
**Syntax:**
```scheme
(if condition then-expr else-expr)
(if condition then-expr)  ; else defaults to unspecified
```

**Examples:**
```scheme
(if (> x 0) "positive" "non-positive")
(if (null? lst) (display "empty"))
```

#### 3.4.2 `cond` - Multi-way Conditional
**Syntax:**
```scheme
(cond (test1 expr1...)
      (test2 expr2...)
      ...
      (else exprN...))
```

**Example:**
```scheme
(cond ((< x 0) "negative")
      ((= x 0) "zero")
      ((> x 0) "positive"))
```

#### 3.4.3 `case` - Switch on Value
**Syntax:**
```scheme
(case key
  ((datum1 datum2 ...) expr1...)
  ((datum3) expr2...)
  (else exprN...))
```

**Comparison:** Uses `eqv?` semantics

**Example:**
```scheme
(case (+ 1 1)
  ((1) "one")
  ((2) "two")
  (else "other"))  ; => "two"
```

#### 3.4.4 `match` - Pattern Matching
**Syntax:**
```scheme
(match expr
  (pattern1 body1...)
  (pattern2 body2...)
  (_ default-body...))
```

**Patterns:**
- **Literal:** `42`, `"hello"`, `#t`
- **Variable:** `x`, `y` (binds the value)
- **Wildcard:** `_` (matches anything, doesn't bind)
- **Cons:** `(cons car-pat cdr-pat)`
- **List:** `(list pat1 pat2 ...)`
- **Predicate:** `(? pred-expr)`
- **Or:** `(or pat1 pat2 ...)`

**Example:**
```scheme
(match (list 1 2)
  ((list x y) (+ x y))  ; => 3
  (_ 0))
```

#### 3.4.5 `do` - Iteration Construct
**Syntax:**
```scheme
(do ((var1 init1 step1)
     (var2 init2 step2)
     ...)
    ((test) result...)
  body...)
```

**Example:**
```scheme
(do ((i 0 (+ i 1))
     (sum 0 (+ sum i)))
    ((= i 10) sum))  ; => 45
```

#### 3.4.6 `when` / `unless` - One-Armed Conditionals
**Syntax:**
```scheme
(when test expr...)    ; Execute if test is true
(unless test expr...)  ; Execute if test is false
```

**Examples:**
```scheme
(when (> x 0) (display "positive"))
(unless (null? lst) (display "not empty"))
```

#### 3.4.7 `and` / `or` - Short-Circuit Logic
**Syntax:**
```scheme
(and expr1 expr2 ...)  ; Returns first false or last value
(or expr1 expr2 ...)   ; Returns first true or last value
```

**Examples:**
```scheme
(and #t #t)      ; => #t
(and #t #f #t)   ; => #f (short-circuits)
(or #f #f #t)    ; => #t
(or 1 2 3)       ; => 1 (first truthy)
```

#### 3.4.8 `begin` - Sequencing
**Syntax:**
```scheme
(begin expr1 expr2 ... exprN)
```

**Returns:** Value of last expression

**Example:**
```scheme
(begin
  (display "Hello")
  (newline)
  42)  ; => 42
```

### 3.5 Quotation and Homoiconicity

#### 3.5.1 `quote` - Literal Data
**Syntax:**
```scheme
(quote datum)
'datum  ; Shorthand
```

**Example:**
```scheme
'(1 2 3)  ; => (1 2 3) as data, not function call
'x        ; => x as symbol
```

#### 3.5.2 `quasiquote` - Template with Unquotes
**Syntax:**
```scheme
(quasiquote template)
`template  ; Shorthand
```

**Unquote operators:**
- `,expr` - Evaluate and substitute
- `,@expr` - Evaluate and splice into list

**Example:**
```scheme
(define x 5)
`(1 2 ,x 4)    ; => (1 2 5 4)
`(1 ,@(list 2 3) 4)  ; => (1 2 3 4)
```

### 3.6 Type Annotations

#### 3.6.1 Type Annotation Syntax
```scheme
(: name type)  ; Standalone type declaration
```

#### 3.6.2 Inline Parameter Types
```scheme
(define (f (x : int) (y : float)) body)
(lambda ((x : int)) body)
```

#### 3.6.3 Return Type Annotations
```scheme
(define (f x y) : int body)
(lambda (x) : real body)
```

#### 3.6.4 Type Expressions
**Primitive Types:**
- `integer`, `int`, `int64`
- `real`, `float`, `double`, `float64`
- `boolean`, `bool`
- `string`, `str`
- `char`, `character`
- `symbol`
- `null`, `nil`
- `any` (top type)
- `nothing`, `never` (bottom type)

**Compound Types:**
- `(list element-type)` - Homogeneous list
- `(vector element-type)` - Homogeneous vector
- `(tensor element-type)` - Homogeneous tensor
- `(pair left-type right-type)` - Pair type
- `(-> param-types... return-type)` - Function type
- `(* type1 type2)` - Product type
- `(+ type1 type2)` - Sum type

**Polymorphic Types:**
- `(forall (a b ...) type-expr)` - Universal quantification

**Example:**
```scheme
(define (map : (forall (a b) (-> (-> a b) (list a) (list b)))
         f lst) ...)
```

#### 3.6.5 `define-type` - Type Aliases
**Syntax:**
```scheme
(define-type Name type-expr)
(define-type (Name params...) type-expr)  ; Parameterized
```

**Examples:**
```scheme
(define-type Point (pair real real))
(define-type (Maybe a) (+ a null))
(define-type (List a) (list a))
```

### 3.7 Module System

#### 3.7.1 `require` - Import Modules
**Syntax:**
```scheme
(require module.name ...)
```

**Module Resolution:**
1. Current directory
2. `lib/` directory
3. `$ESHKOL_PATH` directories
4. System library paths

**Examples:**
```scheme
(require stdlib)
(require core.functional.compose)
(require core.list.higher_order)
```

#### 3.7.2 `provide` - Export Symbols
**Syntax:**
```scheme
(provide name1 name2 ...)
```

**Semantics:** Declares which symbols are public (all others are private)

**Example:**
```scheme
(provide add-squared multiply-squared)
(define (helper x) (* x 2))  ; Private
(define (add-squared x y) (+ (* x x) (* y y)))  ; Public
```

#### 3.7.3 `import` - Legacy File Import
**Syntax:**
```scheme
(import "path/to/file.esk")
```

**Example:**
```scheme
(import "lib/utils.esk")
```

### 3.8 Exception Handling

#### 3.8.1 `guard` - Exception Handler
**Syntax:**
```scheme
(guard (var
        (test1 handler1...)
        (test2 handler2...)
        (else default...))
  body...)
```

**Example:**
```scheme
(guard (e
        ((error? e) (display "Error occurred"))
        (else (display "Unknown exception")))
  (/ 1 0))  ; Division by zero
```

#### 3.8.2 `raise` - Raise Exception
**Syntax:**
```scheme
(raise exception-object)
```

**Example:**
```scheme
(raise (error "Something went wrong"))
```

### 3.9 Multiple Return Values

#### 3.9.1 `values` - Return Multiple Values
**Syntax:**
```scheme
(values expr1 expr2 ...)
```

**Example:**
```scheme
(values 1 2 3)  ; Returns 3 values
```

#### 3.9.2 `call-with-values` - Consume Multiple Values
**Syntax:**
```scheme
(call-with-values producer consumer)
```

**Example:**
```scheme
(call-with-values
  (lambda () (values 1 2))
  +)  ; => 3
```

#### 3.9.3 `let-values` - Bind Multiple Values
**Syntax:**
```scheme
(let-values (((var1 var2 ...) producer1)
             ((var3 var4 ...) producer2)
             ...)
  body...)
```

**Example:**
```scheme
(let-values (((x y) (values 1 2)))
  (+ x y))  ; => 3
```

#### 3.9.4 `let*-values` - Sequential Multiple Value Bindings
**Syntax:**
```scheme
(let*-values (((vars...) producer) ...)
  body...)
```

### 3.10 Macros

#### 3.10.1 `define-syntax` - Hygienic Macros
**Syntax:**
```scheme
(define-syntax name
  (syntax-rules (literal1 literal2 ...)
    ((pattern1) template1)
    ((pattern2) template2)
    ...))
```

**Pattern Elements:**
- Literals - Must match exactly
- Pattern variables - Capture values
- `...` - Ellipsis for repetition
- Nested lists

**Example:**
```scheme
(define-syntax when
  (syntax-rules ()
    ((when test expr ...)
     (if test (begin expr ...)))))
```

### 3.11 External FFI

#### 3.11.1 `extern` - Declare External Function
**Syntax:**
```scheme
(extern return-type function-name param-type...)
(extern return-type function-name :real c-function-name param-type...)
```

**Example:**
```scheme
(extern void printf char* ...)
(extern int strcmp char* char*)
```

#### 3.11.2 `extern-var` - Declare External Variable
**Syntax:**
```scheme
(extern-var type variable-name)
```

---

## 4. Built-in Operators and Functions

### 4.1 Arithmetic Operators

All arithmetic operators are polymorphic (work on integers, floats, dual numbers, AD nodes, and tensors).

#### 4.1.1 `+` - Addition
**Signatures:**
```scheme
(+ num)           ; Unary plus (identity)
(+ num1 num2 ...) ; Variadic addition
```

**Type Promotion:**
- `int + int` => `int`
- `int + float` => `float`
- `float + float` => `float`

**Tensor Support:** Element-wise for vectors/tensors

**Examples:**
```scheme
(+ 1 2)        ; => 3
(+ 1.5 2.5)    ; => 4.0
(+ 1 2 3 4 5)  ; => 15
```

#### 4.1.2 `-` - Subtraction/Negation
**Signatures:**
```scheme
(- num)           ; Negation
(- num1 num2 ...) ; Subtraction (left-associative)
```

**Examples:**
```scheme
(- 5)      ; => -5
(- 10 3)   ; => 7
(- 10 3 2) ; => 5
```

#### 4.1.3 `*` - Multiplication
**Signatures:**
```scheme
(* num1 num2 ...) ; Variadic multiplication
```

**Examples:**
```scheme
(* 3 4)     ; => 12
(* 2 3 4)   ; => 24
```

#### 4.1.4 `/` - Division
**Signatures:**
```scheme
(/ num1 num2 ...) ; Left-associative division
```

**Examples:**
```scheme
(/ 10 2)    ; => 5
(/ 20 2 2)  ; => 5
```

#### 4.1.5 `quotient` - Integer Division
**Signature:** `(quotient n1 n2)`

**Returns:** Truncated quotient

**Example:** `(quotient 10 3)` => `3`

#### 4.1.6 `remainder` - Integer Remainder
**Signature:** `(remainder n1 n2)`

**Returns:** Remainder (Scheme semantics)

**Example:** `(remainder 10 3)` => `1`

#### 4.1.7 `modulo` - Modulo Operation
**Signature:** `(modulo n1 n2)`

**Example:** `(modulo -10 3)` => `2`

#### 4.1.8 `abs` - Absolute Value
**Signature:** `(abs num)`

**Examples:**
```scheme
(abs -5)   ; => 5
(abs 3.14) ; => 3.14
```

#### 4.1.9 `expt` / `pow` - Exponentiation
**Signature:** `(expt base exponent)`

**Example:** `(expt 2 10)` => `1024`

#### 4.1.10 `min` / `max` - Minimum/Maximum
**Signatures:**
```scheme
(min num1 num2 ...)
(max num1 num2 ...)
```

**Examples:**
```scheme
(min 3 1 4 1 5)  ; => 1
(max 3 1 4 1 5)  ; => 5
```

### 4.2 Math Functions

All math functions support dual numbers and AD nodes for automatic differentiation.

#### Trigonometric Functions
- `(sin x)` - Sine (radians)
- `(cos x)` - Cosine (radians)
- `(tan x)` - Tangent (radians)
- `(asin x)` - Arc sine (returns radians)
- `(acos x)` - Arc cosine (returns radians)
- `(atan x)` - Arc tangent (returns radians)

#### Hyperbolic Functions
- `(sinh x)` - Hyperbolic sine
- `(cosh x)` - Hyperbolic cosine
- `(tanh x)` - Hyperbolic tangent

#### Exponential and Logarithmic
- `(exp x)` - e^x
- `(log x)` - Natural logarithm (base e)
- `(sqrt x)` - Square root

#### Rounding Functions
- `(floor x)` - Round down to integer
- `(ceiling x)` - Round up to integer
- `(truncate x)` - Round toward zero
- `(round x)` - Round to nearest integer

### 4.3 Comparison Operators

All comparison operators return booleans and support numeric type promotion.

#### 4.3.1 Numeric Equality
- `(= n1 n2 ...)` - Numeric equality (exact or approximate)

#### 4.3.2 Ordering
- `(< n1 n2 ...)` - Less than (monotonically increasing)
- `(> n1 n2 ...)` - Greater than (monotonically decreasing)
- `(<= n1 n2 ...)` - Less than or equal
- `(>= n1 n2 ...)` - Greater than or equal

#### 4.3.3 Numeric Predicates
- `(zero? n)` - Test if n is zero
- `(positive? n)` - Test if n > 0
- `(negative? n)` - Test if n < 0
- `(odd? n)` - Test if integer is odd
- `(even? n)` - Test if integer is even
- `(exact? n)` - Test if number is exact
- `(inexact? n)` - Test if number is inexact

### 4.4 Equality Predicates

#### 4.4.1 `eq?` - Pointer Equality
**Signature:** `(eq? obj1 obj2)`

**Semantics:** Compares pointer addresses (fastest)

**Use for:** Symbols, booleans, `'()`

#### 4.4.2 `eqv?` - Value Equality
**Signature:** `(eqv? obj1 obj2)`

**Semantics:** Compares values for numbers/chars, pointers for others

**Use for:** Numbers, characters, symbols

#### 4.4.3 `equal?` - Structural Equality
**Signature:** `(equal? obj1 obj2)`

**Semantics:** Deep recursive comparison

**Use for:** Lists, strings, compound structures

**Implementation:** Uses `eshkol_deep_equal()` C runtime function

**Example:**
```scheme
(equal? '(1 2 3) '(1 2 3))  ; => #t (different pointers, same structure)
```

### 4.5 Pair and List Operations

#### 4.5.1 Core Pair Operations
- `(cons car cdr)` - Create pair
- `(car pair)` - Get first element
- `(cdr pair)` - Get second element / rest of list
- `(set-car! pair value)` - Mutate first element
- `(set-cdr! pair value)` - Mutate second element / tail

#### 4.5.2 List Construction
- `(list elem...)` - Create list from elements
- `(list* elem... tail)` - Create list with custom tail
- `(make-list n fill)` - Create n-element list with fill value

#### 4.5.3 List Accessors
**Compound car/cdr (up to 4 levels):**
- 2-level: `caar`, `cadr`, `cdar`, `cddr`
- 3-level: `caaar`, `caadr`, `cadar`, `caddr`, `cdaar`, `cdadr`, `cddar`, `cdddr`
- 4-level: `caaaar` through `cddddr` (16 combinations)

**Positional accessors:**
- `(first lst)` through `(tenth lst)`

#### 4.5.4 List Predicates
- `(null? obj)` - Test if empty list
- `(pair? obj)` - Test if cons cell
- `(list? obj)` - Test if proper list

#### 4.5.5 List Queries
- `(length lst)` - Count elements
- `(list-ref lst n)` - Get nth element (0-indexed)
- `(list-tail lst n)` - Get sublist from position n

#### 4.5.6 List Transformations
- `(append lst...)` - Concatenate lists
- `(reverse lst)` - Reverse list
- `(take lst n)` - First n elements
- `(drop lst n)` - Skip first n elements

#### 4.5.7 List Search
- `(member x lst)` - Find element with `equal?`, return sublist or `#f`
- `(memq x lst)` - Find with `eq?`
- `(memv x lst)` - Find with `eqv?`
- `(assoc key alist)` - Find key in association list with `equal?`
- `(assq key alist)` - Find with `eq?`
- `(assv key alist)` - Find with `eqv?`

### 4.6 Higher-Order Functions

#### 4.6.1 `map` - Apply to Each Element
**Signatures:**
```scheme
(map proc list)              ; Single list
(map proc list1 list2 ...)   ; Multiple lists (parallel)
```

**Examples:**
```scheme
(map square '(1 2 3))  ; => (1 4 9)
(map + '(1 2) '(10 20))  ; => (11 22)
```

#### 4.6.2 `filter` - Select Elements
**Signature:** `(filter predicate list)`

**Example:**
```scheme
(filter even? '(1 2 3 4 5))  ; => (2 4)
```

#### 4.6.3 `fold` / `foldl` - Left Fold
**Signature:** `(fold proc init list)`

**Example:**
```scheme
(fold + 0 '(1 2 3 4))  ; => 10
(fold cons '() '(1 2 3))  ; => (3 2 1)
```

#### 4.6.4 `fold-right` / `foldr` - Right Fold
**Signature:** `(fold-right proc init list)`

**Example:**
```scheme
(fold-right cons '() '(1 2 3))  ; => (1 2 3)
```

#### 4.6.5 `for-each` - Side Effects
**Signature:** `(for-each proc list)`

**Returns:** Unspecified

**Example:**
```scheme
(for-each display '(1 2 3))  ; Prints: 123
```

#### 4.6.6 `apply` - Apply Function to List
**Signature:** `(apply proc arg... list)`

**Example:**
```scheme
(apply + '(1 2 3))  ; => 6
(apply + 1 2 '(3 4))  ; => 10
```

### 4.7 String Operations

#### 4.7.1 String Construction
- `(string char...)` - Build string from characters
- `(make-string k char)` - String of k copies of char
- `(string-append str...)` - Concatenate strings

#### 4.7.2 String Access
- `(string-length str)` - Character count
- `(string-ref str k)` - Get character at index k
- `(string-set! str k char)` - Set character at index k
- `(substring str start end)` - Extract substring

#### 4.7.3 String Predicates
- `(string? obj)` - Test if string
- `(string=? str1 str2)` - String equality
- `(string<? str1 str2)` - Lexicographic less than
- `(string>? str1 str2)` - Lexicographic greater than
- `(string<=? str1 str2)` - Less than or equal
- `(string>=? str1 str2)` - Greater than or equal

#### 4.7.4 String Utilities (from stdlib)
- `(string-join lst delim)` - Join list with delimiter
- `(string-split str delim)` - Split by delimiter
- `(string-trim str)` - Remove leading/trailing whitespace
- `(string-upcase str)` - Convert to uppercase
- `(string-downcase str)` - Convert to lowercase
- `(string-replace str old new)` - Replace all occurrences
- `(string-reverse str)` - Reverse string
- `(string-contains? str substr)` - Test for substring
- `(string-starts-with? str prefix)` - Test for prefix
- `(string-ends-with? str suffix)` - Test for suffix
- `(string-index str substr)` - Find first occurrence index
- `(string-count str substr)` - Count occurrences

#### 4.7.5 Conversions
- `(string->number str)` - Parse number from string
- `(number->string num)` - Convert number to string
- `(string->list str)` - Convert to character list
- `(list->string chars)` - Build from character list

### 4.8 Character Operations

- `(char? obj)` - Test if character
- `(char->integer char)` - Get Unicode codepoint
- `(integer->char n)` - Create character from codepoint
- `(char=? c1 c2)` - Character equality
- `(char<? c1 c2)` - Character less than
- `(char>? c1 c2)` - Character greater than

### 4.9 Vector Operations

#### 4.9.1 Vector Construction
- `(vector elem...)` - Create from elements
- `(make-vector k fill)` - Create k-element vector with fill value

#### 4.9.2 Vector Access
- `(vector-length vec)` - Element count
- `(vector-ref vec k)` - Get element at index k
- `(vector-set! vec k val)` - Set element at index k

#### 4.9.3 Vector Conversions
- `(vector->list vec)` - Convert to list
- `(list->vector lst)` - Convert from list

### 4.10 Tensor Operations

#### 4.10.1 Tensor Creation
- `(vector elem...)` - 1D tensor
- `(matrix rows cols elem...)` - 2D tensor
- `(tensor dim... elem...)` - N-D tensor
- `#(elem...)` - Vector literal syntax

#### 4.10.2 Tensor Access
- `(vref tensor idx)` - 1D access (vector reference)
- `(tensor-get tensor idx...)` - N-D access
- `(tensor-set tensor val idx...)` - N-D mutation

#### 4.10.3 Tensor Arithmetic
- `(tensor-add t1 t2)` - Element-wise addition
- `(tensor-sub t1 t2)` - Element-wise subtraction
- `(tensor-mul t1 t2)` - Element-wise multiplication
- `(tensor-div t1 t2)` - Element-wise division
- `(tensor-dot t1 t2)` - Dot product / matrix multiply
- `(matmul m1 m2)` - Matrix multiplication (alias for tensor-dot)

#### 4.10.4 Tensor Reductions
- `(tensor-sum tensor)` - Sum all elements
- `(tensor-mean tensor)` - Mean of all elements
- `(tensor-reduce-all tensor func init)` - Custom reduction

#### 4.10.5 Tensor Transformations
- `(transpose tensor)` - Transpose (swap dimensions)
- `(reshape tensor dims)` - Change shape (shares data)
- `(tensor-shape tensor)` - Get dimension list
- `(tensor-apply tensor func)` - Apply function element-wise

#### 4.10.6 Tensor Generators
- `(zeros dim...)` - Zero-filled tensor
- `(ones dim...)` - One-filled tensor
- `(eye n)` - Identity matrix
- `(arange start end [step])` - Range tensor
- `(linspace start end num)` - Evenly spaced values

### 4.11 Hash Table Operations

#### 4.11.1 Hash Table Construction
- `(make-hash-table)` - Create empty hash table
- `(hash key1 val1 key2 val2 ...)` - Create from key-value pairs

#### 4.11.2 Hash Table Access
- `(hash-ref table key [default])` - Get value by key
- `(hash-set! table key value)` - Set key-value pair
- `(hash-has-key? table key)` - Check if key exists
- `(hash-remove! table key)` - Remove key

#### 4.11.3 Hash Table Queries
- `(hash-keys table)` - Get all keys as list
- `(hash-values table)` - Get all values as list
- `(hash-count table)` - Count of entries
- `(hash-clear! table)` - Remove all entries

#### 4.11.4 Hash Table Predicates
- `(hash-table? obj)` - Test if hash table

### 4.12 Type Predicates

#### 4.12.1 Basic Type Tests
- `(null? obj)` - Test if empty list
- `(boolean? obj)` - Test if boolean
- `(char? obj)` - Test if character
- `(string? obj)` - Test if string
- `(symbol? obj)` - Test if symbol
- `(number? obj)` - Test if number
- `(integer? obj)` - Test if integer
- `(real? obj)` - Test if real number
- `(pair? obj)` - Test if cons cell
- `(list? obj)` - Test if proper list
- `(vector? obj)` - Test if vector
- `(procedure? obj)` - Test if callable
- `(hash-table? obj)` - Test if hash table
- `(tensor? obj)` - Test if tensor

#### 4.12.2 Port Predicates
- `(input-port? obj)` - Test if input port
- `(output-port? obj)` - Test if output port
- `(port? obj)` - Test if any port
- `(eof-object? obj)` - Test if EOF object

### 4.13 I/O Operations

#### 4.13.1 Output
- `(display obj [port])` - Write object without quotes
- `(write obj [port])` - Write object with quotes (Scheme write semantics)
- `(newline [port])` - Write newline
- `(write-char char [port])` - Write single character
- `(write-string str [port])` - Write string to port
- `(write-line str [port])` - Write string with newline

#### 4.13.2 Input
- `(read [port])` - Read S-expression
- `(read-line [port])` - Read line as string
- `(read-char [port])` - Read single character
- `(peek-char [port])` - Peek at next character without consuming

#### 4.13.3 Ports
- `(open-input-file filename)` - Open file for reading
- `(open-output-file filename)` - Open file for writing
- `(close-port port)` - Close port
- `(flush-output-port port)` - Flush output buffer
- `(current-input-port)` - Get stdin
- `(current-output-port)` - Get stdout

#### 4.13.4 File Operations
- `(read-file filename)` - Read entire file as string
- `(write-file filename content)` - Write string to file
- `(append-file filename content)` - Append to file
- `(file-exists? filename)` - Test if file exists
- `(file-readable? filename)` - Test if readable
- `(file-writable? filename)` - Test if writable
- `(file-delete filename)` - Delete file
- `(file-rename old new)` - Rename/move file
- `(file-size filename)` - Get file size in bytes

#### 4.13.5 Directory Operations
- `(directory-exists? path)` - Test if directory exists
- `(make-directory path)` - Create directory
- `(delete-directory path)` - Delete directory
- `(directory-list path)` - List directory contents
- `(current-directory)` - Get current working directory
- `(set-current-directory! path)` - Change working directory

### 4.14 System Operations

#### 4.14.1 Environment Variables
- `(getenv name)` - Get environment variable
- `(setenv name value)` - Set environment variable
- `(unsetenv name)` - Unset environment variable

#### 4.14.2 System Calls
- `(system command)` - Execute shell command
- `(exit [code])` - Exit program with code
- `(sleep seconds)` - Sleep for duration
- `(current-seconds)` - Get Unix timestamp
- `(command-line)` - Get command-line arguments as list

### 4.15 Type Conversions

- `(exact->inexact num)` - Convert to inexact
- `(inexact->exact num)` - Convert to exact
- `(string->symbol str)` - Create symbol from string
- `(symbol->string sym)` - Get symbol name as string

---

## 5. Standard Library

The standard library is organized into modules under `lib/core/`:

### 5.1 core.functional

#### 5.1.1 compose.esk
- `(compose f g)` - Function composition: `(lambda (x) (f (g x)))`
- `(compose3 f g h)` - Three-function composition
- `(identity x)` - Identity function
- `(constantly x)` - Function that always returns x

#### 5.1.2 curry.esk
- `(curry2 f)` - Curry 2-argument function
- `(curry3 f)` - Curry 3-argument function
- `(uncurry2 f)` - Uncurry curried function
- `(partial f arg...)` - Partial application
- `(partial1 f x)` - Partial (unary thunk)
- `(partial2 f x)` - Partial (returns unary function)
- `(partial3 f x)` - Partial (returns binary function)

#### 5.1.3 flip.esk
- `(flip f)` - Swap arguments of binary function

### 5.2 core.list

#### 5.2.1 compound.esk
All compound car/cdr operations (caar through cddddr)

#### 5.2.2 generate.esk
- `(iota n)` - Generate list [0, 1, ..., n-1]
- `(iota-from n start)` - Generate n numbers from start
- `(iota-step n start step)` - Generate with custom step
- `(repeat n x)` - Create list of n copies of x
- `(make-list n fill)` - N-element list with fill
- `(range start end)` - Generate [start, start+1, ..., end-1]
- `(zip lst1 lst2)` - Combine into list of pairs

#### 5.2.3 higher_order.esk
- `(map1 proc lst)` - Map over single list
- `(map2 proc lst1 lst2)` - Map over two lists
- `(map3 proc lst1 lst2 lst3)` - Map over three lists
- `(fold proc init lst)` - Left fold
- `(fold-right proc init lst)` - Right fold
- `(for-each proc lst)` - For side effects
- `(any pred lst)` - True if any element satisfies pred
- `(every pred lst)` - True if all elements satisfy pred

#### 5.2.4 query.esk
- `(length lst)` - List length
- `(count-if pred lst)` - Count elements satisfying predicate
- `(find pred lst)` - First element satisfying predicate or #f

#### 5.2.5 search.esk
- `(member x lst)` - Find with equal?
- `(member? x lst)` - Test membership (returns boolean)
- `(memq x lst)` - Find with eq?
- `(memv x lst)` - Find with eqv?
- `(assoc key alist)` - Association list lookup
- `(assq key alist)` - With eq?
- `(assv key alist)` - With eqv?
- `(list-ref lst n)` - Nth element
- `(list-tail lst n)` - Sublist from position n

#### 5.2.6 sort.esk
- `(sort lst less?)` - Sort using merge sort algorithm

#### 5.2.7 transform.esk
- `(take lst n)` - First n elements
- `(drop lst n)` - Skip first n elements
- `(append lst...)` - Concatenate
- `(reverse lst)` - Reverse
- `(filter pred lst)` - Select elements
- `(unzip pairs)` - Split list of pairs into two lists
- `(partition pred lst)` - Split into (passing, failing)

#### 5.2.8 convert.esk
- `(list->vector lst)` - Convert to vector
- `(vector->list vec)` - Convert to list

### 5.3 core.logic

#### 5.3.1 boolean.esk
- `(negate pred)` - Negate predicate function
- `(all? pred lst)` - All elements satisfy
- `(none? pred lst)` - No elements satisfy

#### 5.3.2 predicates.esk
- `(is-zero? x)` - Wrapper for zero?
- `(is-positive? x)` - Wrapper for positive?
- `(is-negative? x)` - Wrapper for negative?
- `(is-even? x)` - Wrapper for even?
- `(is-odd? x)` - Wrapper for odd?

#### 5.3.3 types.esk
- `(is-null? x)` - Wrapper for null?
- `(is-pair? x)` - Wrapper for pair?

### 5.4 core.operators

#### 5.4.1 arithmetic.esk
First-class wrappers for operators:
- `(add x y)` - Wrapper for +
- `(sub x y)` - Wrapper for -
- `(mul x y)` - Wrapper for *
- `(div x y)` - Wrapper for /

#### 5.4.2 compare.esk
- `(lt x y)` - Wrapper for <
- `(gt x y)` - Wrapper for >
- `(le x y)` - Wrapper for <=
- `(ge x y)` - Wrapper for >=
- `(eq x y)` - Wrapper for =

### 5.5 core.strings
Extended string utilities (see §4.7.4 for complete list)

### 5.6 core.io
- `(print x)` - Wrapper for display (for higher-order use)
- `(println x)` - Display with newline

### 5.7 core.json
JSON parsing and serialization:
- `(json-parse str)` - Parse JSON string to hash tables/lists
- `(json-stringify val)` - Convert to JSON string
- `(json-get obj key [default])` - Get field from object
- `(json-array-ref arr idx)` - Get array element
- `(hash-table->alist ht)` - Convert to association list
- `(alist->hash-table alist)` - Convert from association list
- `(json-read-file filename)` - Read JSON file
- `(json-write-file filename data)` - Write JSON file
- `(alist->json alist)` - Convert alist to JSON string
- `(alist-write-json filename alist)` - Write alist as JSON

**JSON Mapping:**
- JSON object → hash-table
- JSON array → list
- JSON string → string
- JSON number → integer or real
- JSON true → #t
- JSON false → #f
- JSON null → '()

### 5.8 core.data

#### 5.8.1 base64.esk
- `(base64-encode bytes)` - Encode byte list to base64 string
- `(base64-decode str)` - Decode base64 string to byte list
- `(base64-encode-string str)` - Encode string to base64
- `(base64-decode-string str)` - Decode base64 to string
- `(string->bytes str)` - Convert string to byte list
- `(bytes->string bytes)` - Convert byte list to string

#### 5.8.2 csv.esk
- `(csv-parse str)` - Parse CSV string to list of rows
- `(csv-parse-file filename)` - Read and parse CSV file
- `(csv-stringify rows)` - Convert rows to CSV string
- `(csv-write-file filename rows)` - Write rows to CSV file
- `(csv-parse-line line)` - Parse single CSV line
- `(csv-stringify-row row)` - Convert row to CSV string

### 5.9 core.control

#### 5.9.1 trampoline.esk
- `(trampoline thunk)` - Evaluate thunks in constant stack space
- `(bounce thunk)` - Create continuation thunk
- `(done value)` - Signal completion

### 5.10 math.esk - Advanced Mathematics

#### 5.10.1 Constants
- `pi` - π (3.141592653589793)
- `e` - Euler's number (2.718281828459045)
- `epsilon` - Machine epsilon (1e-15)

#### 5.10.2 Linear Algebra
- `(mat-ref M cols row col)` - Get matrix element
- `(tensor-copy T)` - Create mutable copy
- `(mat-vec-mul A x rows cols)` - Matrix-vector multiplication
- `(dot u v)` - Dot product of vectors
- `(cross u v)` - Cross product (3D only)
- `(normalize v)` - Unit vector in same direction

#### 5.10.3 Matrix Operations
- `(det M n)` - Determinant (LU decomposition with partial pivoting)
- `(inv M n)` - Matrix inverse (Gauss-Jordan elimination)
- `(solve A b n)` - Solve linear system Ax = b

#### 5.10.4 Eigenvalues
- `(power-iteration A n max-iters tolerance)` - Dominant eigenvalue estimation

#### 5.10.5 Numerical Integration
- `(integrate f a b n)` - Numerical integration (Simpson's rule)

#### 5.10.6 Root Finding
- `(newton f df x0 tolerance max-iters)` - Newton-Raphson method

#### 5.10.7 Statistics
- `(variance v)` - Variance of vector
- `(std v)` - Standard deviation
- `(covariance u v)` - Covariance of two vectors

---

## 6. Memory Management System

### 6.1 Arena Memory Management

#### 6.1.1 Core Concepts
- **Arena Allocation:** Bump-pointer allocation in large blocks
- **Scope-based Cleanup:** Memory freed when scope exits
- **No Garbage Collection:** Deterministic, predictable performance
- **Cache-Friendly:** Linear allocation pattern

#### 6.1.2 Arena Structure
```c
struct arena {
    arena_block_t* current_block;   // Current allocation block
    arena_scope_t* current_scope;   // Current scope
    size_t default_block_size;      // Default size for new blocks
    size_t total_allocated;         // Total memory allocated
    size_t alignment;               // Memory alignment
}
```

#### 6.1.3 Global Arena
- **Variable:** `__global_arena`
- **Default Block Size:** 64KB
- **Usage:** Default allocation target outside regions

### 6.2 OALR (Ownership-Aware Lexical Regions)

#### 6.2.1 `with-region` - Lexical Memory Regions
**Syntax:**
```scheme
(with-region body...)
(with-region 'name body...)
(with-region ('name size-hint) body...)
```

**Semantics:**
- Creates a dedicated arena for the body
- Memory freed when region exits
- Supports nesting (stack of regions)

**Example:**
```scheme
(with-region 'temp
  (define data (iota 1000))
  (process data))
; data's memory freed here
```

#### 6.2.2 `owned` - Mark Ownership
**Syntax:** `(owned expr)`

**Semantics:** Value must be consumed before scope exit

**Example:**
```scheme
(define resource (owned (allocate-something)))
(use-and-consume resource)  ; Must use exactly once
```

#### 6.2.3 `move` - Transfer Ownership
**Syntax:** `(move variable)`

**Semantics:** Transfers ownership, marks original as moved

**Compile-Time Checks:**
- Cannot use after move
- Cannot move while borrowed

**Example:**
```scheme
(define x (owned (list 1 2 3)))
(define y (move x))
; x is now invalid, y owns the list
```

#### 6.2.4 `borrow` - Temporary Access
**Syntax:**
```scheme
(borrow value body...)
```

**Semantics:** Temporary read-only access during body

**Example:**
```scheme
(define x (owned (list 1 2 3)))
(borrow x
  (display x)  ; Can read but not move
  (length x))
; x still owned after borrow
```

#### 6.2.5 `shared` - Reference Counting
**Syntax:** `(shared expr)`

**Semantics:** Create reference-counted value

**Structure:** Prepends `eshkol_shared_header_t` (24 bytes):
```c
struct {
    void (*destructor)(void*);  // Custom cleanup
    uint32_t ref_count;         // Strong references
    uint32_t weak_count;        // Weak references
    uint8_t flags;
    uint8_t value_type;
}
```

**Operations:**
- `shared_retain(ptr)` - Increment ref count
- `shared_release(ptr)` - Decrement ref count (frees at 0)

#### 6.2.6 `weak-ref` - Weak References
**Syntax:** `(weak-ref shared-value)`

**Semantics:** Reference that doesn't prevent deallocation

**Operations:**
- `weak_ref_create(ptr)` - Create weak ref
- `weak_ref_upgrade(weak)` - Try to get strong ref (NULL if freed)
- `weak_ref_is_alive(weak)` - Check if target exists

### 6.3 Memory Statistics

Runtime functions for memory tracking:
- `arena_get_used_memory(arena)` - Bytes used
- `arena_get_total_memory(arena)` - Total allocated
- `arena_get_block_count(arena)` - Number of blocks
- `region_get_used_memory(region)` - Region bytes used
- `region_get_total_memory(region)` - Region total
- `region_get_depth()` - Current region nesting depth

---

## 7. Type System

### 7.1 HoTT (Homotopy Type Theory) Foundation

Eshkol implements a gradual type system inspired by Homotopy Type Theory with:

- **Universe Levels:** Type₀, Type₁, Type₂, Typeω
- **Subtype Hierarchy:** Transitive subtyping with caching
- **Type Inference:** Bidirectional type checking
- **Runtime Types:** Type information preserved in tagged values

### 7.2 Universe Hierarchy

#### Universe Levels
- **U0 (Type₀):** Ground types (integers, floats, strings, chars, booleans)
- **U1 (Type₁):** Type constructors (List, Vector, ->, Pair, Tensor)
- **U2 (Type₂):** Propositions (Eq, <:, Bounded, Linear) - erased at runtime
- **UOmega (Typeω):** Universe polymorphic

### 7.3 Type Hierarchy

```
Value (root)
├── Number
│   ├── Integer
│   │   ├── Int64
│   │   └── Natural
│   ├── Real
│   │   ├── Float64
│   │   └── Float32
│   └── Complex
│       ├── Complex64
│       └── Complex128
├── Text
│   ├── String
│   └── Char
├── Boolean
├── Null
└── Symbol
```

### 7.4 Type Constructors

#### Parameterized Types
- `List<A>` - Homogeneous list
- `Vector<A>` - Homogeneous vector  
- `Tensor<A>` - Homogeneous tensor
- `Pair<A,B>` - Typed pair
- `Function<A,B>` or `A -> B` - Function type
- `HashTable<K,V>` - Typed hash table

#### Type Families (U1)
- `List` - List type constructor
- `Vector` - Vector type constructor
- `Pair` - Pair type constructor
- `->` - Function type constructor
- `Tensor` - Tensor type constructor
- `Closure` - Closure type
- `DualNumber<T>` - Dual number for AD
- `ADNode<T>` - AD computation node
- `Handle<K>` - Resource handle
- `Buffer<A,n>` - Typed buffer with size
- `Stream<A>` - Data stream

### 7.5 Type Flags

- `TYPE_FLAG_EXACT` - Scheme exactness (integers)
- `TYPE_FLAG_LINEAR` - Must use exactly once (quantum no-cloning)
- `TYPE_FLAG_PROOF` - Compile-time only (erased at runtime)
- `TYPE_FLAG_ABSTRACT` - Cannot instantiate directly

### 7.6 Type Checking

#### 7.6.1 Bidirectional Type Checking

**Synthesis Mode (⇒):** Infer type from expression
```
Γ ⊢ e ⇒ τ
```

**Checking Mode (⇐):** Verify expression has expected type
```
Γ ⊢ e ⇐ τ
```

#### 7.6.2 Type Inference

The type checker infers types for:
- Literals (int64, float64, string, boolean, etc.)
- Variables (from context)
- Applications (from function type)
- Lambda expressions (from parameter annotations and body)
- Let bindings (from value expressions)

#### 7.6.3 Subtyping

Transitive subtype relations with caching:
- Int64 <: Integer <: Number <: Value
- Float64 <: Real <: Number <: Value
- String <: Text <: Value

Type promotion for arithmetic:
- int + int => int
- int + float => float
- float + float => float

### 7.7 Dependent Types (Phase 5)

#### 7.7.1 Compile-Time Values
```cpp
class CTValue {
    Kind: Nat | Bool | Expr | Unknown
    
    // Operations
    tryEvalNat() -> Optional<uint64_t>
    tryEvalFloat() -> Optional<double>
    add(other) -> CTValue
    mul(other) -> CTValue
}
```

#### 7.7.2 Dependent Type Expressions
```cpp
class DependentType {
    TypeId base;
    vector<TypeId> type_indices;    // Type parameters
    vector<CTValue> value_indices;  // Dimension parameters
}
```

**Examples:**
- `Vector<Float64, 100>` - 100-element float vector
- `Matrix<Int64, 3, 4>` - 3×4 integer matrix
- `Buffer<Float64, n>` - Buffer with symbolic dimension

#### 7.7.3 Dimension Checking
Static verification of array bounds and dimension compatibility:
- Index bounds checking
- Matrix multiplication dimension matching
- Dot product dimension equality

### 7.8 Linear Types (Phase 6)

#### 7.8.1 Linear Type Constraints
Values with `TYPE_FLAG_LINEAR` must be used exactly once:
- Quantum states (no-cloning theorem)
- File handles (must close)
- Network connections

#### 7.8.2 Borrow Checking
**States:**
- `Owned` - Value is owned, can be moved or borrowed
- `Moved` - Value has been moved, cannot use
- `BorrowedShared` - Immutably borrowed (multiple readers)
- `BorrowedMut` - Mutably borrowed (exclusive access)
- `Dropped` - Explicitly dropped

**Rules:**
1. Can move only once
2. Cannot move while borrowed
3. Mutable borrows are exclusive
4. Shared borrows allow multiple readers
5. Borrows must not outlive borrowed value

---

## 8. Automatic Differentiation

Eshkol provides first-class automatic differentiation for scientific computing and machine learning.

### 8.1 Forward-Mode AD (Dual Numbers)

#### 8.1.1 Dual Number Structure
```c
struct eshkol_dual_number {
    double value;       // f(x)
    double derivative;  // f'(x)
}
```

#### 8.1.2 Dual Arithmetic Rules
- `(a, a') + (b, b')` = `(a+b, a'+b')`
- `(a, a') - (b, b')` = `(a-b, a'-b')`
- `(a, a') * (b, b')` = `(a*b, a*b' + a'*b)`
- `(a, a') / (b, b')` = `(a/b, (a'*b - a*b')/b²)`

#### 8.1.3 `derivative` - Compute Derivative
**Syntax:**
```scheme
(derivative function point)       ; Evaluate at point
(derivative function)             ; Return derivative function
```

**Examples:**
```scheme
(derivative (lambda (x) (* x x)) 5.0)  ; => 10.0 (2x at x=5)

(define df (derivative (lambda (x) (sin x))))
(df 0.0)  ; => 1.0 (cos(0))
```

**Implementation:** Forward-mode AD with dual numbers

### 8.2 Reverse-Mode AD (Computational Graphs)

#### 8.2.1 AD Tape Structure
```c
struct ad_tape {
    ad_node_t** nodes;         // Array of nodes in eval order
    size_t num_nodes;          // Current node count
    size_t capacity;           // Allocated capacity
    ad_node_t** variables;     // Input variable nodes
    size_t num_variables;      // Number of inputs
}
```

#### 8.2.2 Tape Stack (for nested gradients)
- **Global:** `__ad_tape_stack[32]` - Stack of tapes
- **Depth:** `__ad_tape_depth` - Current nesting level
- **Max Nesting:** 32 levels

#### 8.2.3 `gradient` - Compute Gradient
**Syntax:**
```scheme
(gradient function point)         ; Scalar field: ℝⁿ → ℝ
(gradient function)               ; Return gradient function
```

**Returns:** Vector of partial derivatives

**Example:**
```scheme
(define (f v)
  (+ (* (vref v 0) (vref v 0))    ; x²
     (* (vref v 1) (vref v 1))))  ; + y²

(gradient f (vector 3.0 4.0))  ; => #(6.0 8.0)  [∂f/∂x, ∂f/∂y]
```

**Implementation:**
1. Forward pass: Record computation graph on tape
2. Backward pass: Backpropagate gradients from output to inputs

### 8.3 Vector Calculus Operations

#### 8.3.1 `jacobian` - Jacobian Matrix
**Syntax:** `(jacobian function point)`

**Domain:** Vector function ℝⁿ → ℝᵐ

**Returns:** m×n matrix of partial derivatives

**Example:**
```scheme
(define (polar-to-cartesian v)
  (vector (* (vref v 0) (cos (vref v 1)))
          (* (vref v 0) (sin (vref v 1)))))

(jacobian polar-to-cartesian (vector 1.0 0.0))
; => Jacobian at (r=1, θ=0)
```

#### 8.3.2 `hessian` - Hessian Matrix
**Syntax:** `(hessian function point)`

**Domain:** Scalar field ℝⁿ → ℝ

**Returns:** n×n matrix of second partial derivatives

**Example:**
```scheme
(define (f v)
  (+ (* (vref v 0) (vref v 0))
     (* (vref v 1) (vref v 1))))

(hessian f (vector 1.0 1.0))
; => #((2 0) (0 2))  [constant Hessian for quadratic]
```

#### 8.3.3 `divergence` - Vector Field Divergence
**Syntax:** `(divergence function point)`

**Domain:** Vector field ℝⁿ → ℝⁿ

**Returns:** Scalar (trace of Jacobian)

**Formula:** ∇·F = ∂F₁/∂x₁ + ∂F₂/∂x₂ + ... + ∂Fₙ/∂xₙ

**Example:**
```scheme
(define (radial v)
  (vector (vref v 0) (vref v 1)))  ; F(x,y) = (x, y)

(divergence radial (vector 1.0 1.0))  ; => 2.0
```

#### 8.3.4 `curl` - Vector Field Curl
**Syntax:** `(curl function point)`

**Domain:** 3D vector field ℝ³ → ℝ³

**Returns:** 3D vector

**Formula:** ∇×F = (∂F₃/∂y - ∂F₂/∂z, ∂F₁/∂z - ∂F₃/∂x, ∂F₂/∂x - ∂F₁/∂y)

**Example:**
```scheme
(define (rotating v)
  (vector (- 0.0 (vref v 1))   ; -y
          (vref v 0)             ; x
          0.0))                  ; 0

(curl rotating (vector 1.0 1.0 0.0))  ; => #(0 0 2)
```

#### 8.3.5 `laplacian` - Scalar Field Laplacian
**Syntax:** `(laplacian function point)`

**Domain:** Scalar field ℝⁿ → ℝ

**Returns:** Scalar (trace of Hessian)

**Formula:** ∇²f = ∂²f/∂x₁² + ∂²f/∂x₂² + ... + ∂²f/∂xₙ²

**Example:**
```scheme
(define (harmonic v)
  (+ (* (vref v 0) (vref v 0))
     (* (vref v 1) (vref v 1))))  ; x² + y²

(laplacian harmonic (vector 1.0 1.0))  ; => 4.0
```

#### 8.3.6 `directional-derivative` - Directional Derivative
**Syntax:** `(directional-derivative function point direction)`

**Formula:** D_v f(p) = ∇f(p) · v

**Example:**
```scheme
(define (f v) (+ (* (vref v 0) (vref v 0))
                 (* (vref v 1) (vref v 1))))
(define point (vector 1.0 1.0))
(define direction (vector 1.0 0.0))  ; x-direction

(directional-derivative f point direction)
```

### 8.4 AD Mode Tracking

#### Global State
- `__ad_mode_active` - Boolean flag indicating AD context
- `__current_ad_tape` - Current tape for graph recording

#### Nested Gradients
Supports arbitrary nesting depth via tape stack:
```scheme
(gradient 
  (lambda (x)
    (gradient 
      (lambda (y) (* x y y))
      (vector 1.0)))
  (vector 2.0))
; Computes ∂/∂x[∂/∂y(xy²)]
```

---

## 9. Module System

### 9.1 Module Resolution

#### Search Order:
1. Current directory (relative to source file)
2. `lib/` directory
3. `$ESHKOL_PATH` environment variable (colon-separated)
4. System library paths

#### Path Conversion:
- Module name `core.functional.compose`
- Becomes file path `lib/core/functional/compose.esk`

### 9.2 Symbol Visibility

#### Private Symbols
Symbols not in `provide` are renamed with module prefix:
```
Module: test.utils
Private symbol: helper
Mangled name: __test_utils__helper
```

#### Public Symbols
Symbols in `provide` keep their original names and are visible to importers.

### 9.3 Dependency Management

#### Circular Dependency Detection
- Uses DFS coloring (UNVISITED, VISITING, VISITED)
- Reports complete cycle path on error

#### Topological Sorting
Modules loaded in dependency order (dependencies before dependents)

### 9.4 Pre-compiled Modules

Modules can be pre-compiled to `.o` files for faster loading:
- `stdlib.o` - Pre-compiled standard library
- Linked at final executable creation
- `require` statements recognize pre-compiled modules

---

## 10. Compilation Model

### 10.1 Compilation Pipeline

```
Source (.esk) → Parser → AST → Macro Expansion → Type Checking → 
LLVM IR → Object File (.o) → Executable
```

### 10.2 Parser

#### 10.2.1 Tokenization
**Token Types:**
- `TOKEN_LPAREN`, `TOKEN_RPAREN` - Parentheses
- `TOKEN_QUOTE` - `'`
- `TOKEN_BACKQUOTE` - `` ` ``
- `TOKEN_COMMA` - `,`
- `TOKEN_COMMA_AT` - `,@`
- `TOKEN_SYMBOL` - Identifiers
- `TOKEN_STRING` - String literals
- `TOKEN_NUMBER` - Numeric literals
- `TOKEN_BOOLEAN` - `#t` / `#f`
- `TOKEN_CHAR` - Character literals
- `TOKEN_VECTOR_START` - `#(`
- `TOKEN_COLON` - `:` (type annotations)
- `TOKEN_ARROW` - `->` (function types)

#### 10.2.2 AST Structure
```c
struct eshkol_ast {
    eshkol_type_t type;     // AST node type
    union {
        int64_t int64_val;
        double double_val;
        struct { char* ptr; uint64_t size; } str_val;
        struct { char* id; ... } variable;
        struct { ... } eshkol_func;
        struct { car; cdr; } cons_cell;
        eshkol_operations_t operation;
    };
    uint32_t inferred_hott_type;  // Type checker result
    uint32_t line;                // Source location
    uint32_t column;
}
```

### 10.3 LLVM Backend

#### 10.3.1 Code Generation Architecture

**Modular Design:**
- `CodegenContext` - Shared state and infrastructure
- `TypeSystem` - LLVM type management
- `TaggedValueCodegen` - Pack/unpack tagged values
- `ArithmeticCodegen` - Polymorphic arithmetic
- `AutodiffCodegen` - AD operators
- `BindingCodegen` - Variable definitions
- `CollectionCodegen` - Lists and vectors
- `ControlFlowCodegen` - Conditionals and loops
- `FunctionCodegen` - Lambdas and closures
- `TensorCodegen` - Tensor operations
- `HashCodegen` - Hash tables
- `StringIOCodegen` - Strings and I/O
- `SystemCodegen` - System operations
- `HomoiconicCodegen` - Quote and S-expressions
- `TailCallCodegen` - Tail call optimization

#### 10.3.2 Closure Compilation

**Steps:**
1. Analyze for captured variables (static AST analysis)
2. Generate lambda function with captures as parameters
3. Allocate closure struct with environment
4. Store captures in environment
5. Register S-expression for homoiconic display

**Calling Convention:**
```c
eshkol_tagged_value func(param1, param2, ..., capture1, capture2, ...)
```

### 10.4 Optimization Strategies

#### Tail Call Optimization
- Direct tail calls marked with `musttail` attribute
- Self-recursive functions transformed to loops
- Trampoline for closure tail calls

#### Inline Caching
- Function table for fast symbol lookup
- Type specialization for hot paths
- Interned strings for fast equality

---

## 11. REPL and JIT

### 11.1 Interactive REPL

#### 11.1.1 Features
- **JIT Compilation:** Instant execution via LLVM ORC JIT
- **Persistent State:** Variables and functions persist across evaluations
- **Tab Completion:** Context-aware completion for builtins and user symbols
- **History:** Command history with readline support
- **Multi-line Input:** Auto-continuation for incomplete expressions
- **Colored Output:** Syntax highlighting for results

#### 11.1.2 REPL Commands
- `:help`, `:h` - Show help
- `:quit`, `:q` - Exit REPL
- `:clear` - Clear screen
- `:env`, `:e` - Show defined symbols
- `:type <expr>` - Show type of expression
- `:doc <name>` - Show function documentation
- `:ast <expr>` - Show AST structure
- `:time <expr>` - Time execution
- `:load <file>` - Load and execute file
- `:reload` - Reload last file
- `:stdlib` - Load standard library
- `:examples` - Show example expressions
- `:version`, `:v` - Show version

### 11.2 JIT Architecture

#### 11.2.1 LLVM ORC JIT
- **Engine:** LLJIT (LLVM On-Request Compilation)
- **Threading:** Single-threaded compilation
- **Symbol Resolution:** Dynamic library search for runtime functions

#### 11.2.2 Shared Arena
- **Global:** `__repl_shared_arena`
- **Persistence:** Single arena shared across all evaluations
- **Purpose:** Data from one evaluation accessible in later ones

#### 11.2.3 Symbol Persistence
- Function names registered in global REPL context
- Previous definitions injectable as external declarations
- Cross-evaluation function calls supported

---

## 12. Advanced Features

### 12.1 Closures

#### 12.1.1 Lexical Scoping
Variables captured from enclosing scopes:
```scheme
(define (make-adder n)
  (lambda (x) (+ x n)))  ; Captures n

(define add5 (make-adder 5))
(add5 10)  ; => 15
```

#### 12.1.2 Variadic Functions
**Fixed + Rest Parameters:**
```scheme
(define (f x y . rest)
  (cons x (cons y rest)))

(f 1 2 3 4)  ; => (1 2 3 4)
```

**All Parameters as List:**
```scheme
(lambda args (fold + 0 args))
```

#### 12.1.3 Closure Environment Encoding
```c
num_captures field encodes:
- Bits 0-15:  actual capture count
- Bits 16-31: fixed parameter count
- Bit 63:     is_variadic flag
```

### 12.2 Homoiconicity (Code as Data)

#### 12.2.1 Lambda Registry
Runtime table mapping function pointers to S-expressions:
```c
struct eshkol_lambda_entry {
    uint64_t func_ptr;   // Function pointer
    uint64_t sexpr_ptr;  // Pointer to S-expression cons cell
    const char* name;    // Function name (optional)
}
```

#### 12.2.2 S-Expression Storage
Each lambda stores its source as a cons-cell chain:
```scheme
(define square (lambda (x) (* x x)))
(display square)  ; => (lambda (x) (* x x))
```

#### 12.2.3 Quote Implementation
Compile-time AST converted to runtime cons cells:
```scheme
'(1 2 (+ 3 4))  ; Stored as cons cells, not evaluated
```

### 12.3 Pattern Matching

#### Pattern Types
- `PATTERN_LITERAL` - Match exact value
- `PATTERN_VARIABLE` - Bind to variable
- `PATTERN_WILDCARD` - Match anything (no binding)
- `PATTERN_CONS` - Match pair structure
- `PATTERN_LIST` - Match list of patterns
- `PATTERN_PREDICATE` - Match if predicate returns true
- `PATTERN_OR` - Match any of several patterns

#### Match Compilation
Patterns compiled to decision trees with binding extraction.

### 12.4 Tail Call Optimization

#### 12.4.1 Direct Tail Calls
Functions in tail position marked with LLVM `musttail` attribute:
```scheme
(define (sum-to n acc)
  (if (= n 0)
      acc
      (sum-to (- n 1) (+ acc n))))  ; Tail call optimized
```

#### 12.4.2 Named Let Loops
Transformed to iteration:
```scheme
(let loop ((n 10) (acc 0))
  (if (= n 0)
      acc
      (loop (- n 1) (+ acc n))))
; Compiles to while loop, not recursion
```

#### 12.4.3 Trampoline
For closures with captures, use bounce/trampoline:
```scheme
(define (t-fact n acc)
  (if (= n 0)
      acc
      (lambda () (t-fact (- n 1) (* acc n)))))

(trampoline (lambda () (t-fact 1000 1)))  ; Constant stack
```

### 12.5 Internal Defines

Function bodies can contain internal definitions, which are automatically transformed to `letrec`:

```scheme
(define (outer x)
  (define (helper y) (+ y 1))
  (define z 10)
  (+ x (helper z)))

; Transformed to:
(define (outer x)
  (letrec ((helper (lambda (y) (+ y 1)))
           (z 10))
    (+ x (helper z))))
```

---

## 13. Runtime Architecture

### 13.1 Tagged Value System

#### 13.1.1 Runtime Representation
Every value at runtime is a 16-byte `eshkol_tagged_value_t`:
- Offset 0: `uint8_t type` - Type tag (0-255)
- Offset 1: `uint8_t flags` - Exactness, special flags
- Offset 2: `uint16_t reserved`
- Offset 4: `uint32_t padding` - Alignment
- Offset 8: `union data` - int64/double/ptr (8 bytes)

#### 13.1.2 Type Tag Encoding

**Immediate Types (0-7):** Data stored directly
- 0: NULL
- 1: INT64
- 2: DOUBLE
- 3: BOOL
- 4: CHAR
- 5: SYMBOL
- 6: DUAL_NUMBER

**Consolidated Types (8-9):** Subtype in header
- 8: HEAP_PTR (cons, string, vector, tensor, hash, exception)
- 9: CALLABLE (closure, lambda-sexpr, ad-node)

**Multimedia Types (16-19):** Reserved for future
- 16: HANDLE
- 17: BUFFER
- 18: STREAM
- 19: EVENT

**Legacy Types (32+):** Backward compatibility
- 32: CONS_PTR (deprecated)
- 33: STRING_PTR (deprecated)
- 34-40: Other legacy pointer types

### 13.2 Object Header System

All heap objects have an 8-byte header prepended:
```c
struct eshkol_object_header {
    uint8_t  subtype;      // Type within HEAP_PTR/CALLABLE
    uint8_t  flags;        // GC marks, linear status
    uint16_t ref_count;    // Reference count
    uint32_t size;         // Object size
}
```

**Flags:**
- `ESHKOL_OBJ_FLAG_MARKED` (0x01) - GC mark bit
- `ESHKOL_OBJ_FLAG_LINEAR` (0x02) - Linear type
- `ESHKOL_OBJ_FLAG_BORROWED` (0x04) - Currently borrowed
- `ESHKOL_OBJ_FLAG_CONSUMED` (0x08) - Linear value consumed
- `ESHKOL_OBJ_FLAG_SHARED` (0x10) - Reference-counted
- `ESHKOL_OBJ_FLAG_WEAK` (0x20) - Weak reference
- `ESHKOL_OBJ_FLAG_PINNED` (0x40) - Pinned (no relocation)
- `ESHKOL_OBJ_FLAG_EXTERNAL` (0x80) - External resource

**Header Access Macros:**
```c
ESHKOL_GET_HEADER(data_ptr)     // Get header from data
ESHKOL_GET_SUBTYPE(data_ptr)    // Get subtype byte
ESHKOL_GET_FLAGS(data_ptr)      // Get flags byte
ESHKOL_HAS_FLAG(data_ptr, flag) // Check specific flag
```

### 13.3 Cons Cell Implementation

#### 13.3.1 Tagged Cons Cell Structure
```c
struct arena_tagged_cons_cell {
    eshkol_tagged_value_t car;  // 16 bytes - Complete tagged value
    eshkol_tagged_value_t cdr;  // 16 bytes - Complete tagged value
}  // Total: 32 bytes (cache-aligned)
```

#### 13.3.2 Mixed-Type List Support
Each cons cell element stores complete type information, enabling:
- `(1 "two" #t)` - Integer, string, boolean
- `((lambda (x) x) 42 (list 1 2))` - Function, number, list
- Full type preservation through list operations

### 13.4 Display System

#### 13.4.1 Unified Display
Single `eshkol_display_value()` function handles all types:
- Immediate values: Direct display
- Cons cells: Recursive list display with dotted pair support
- Vectors: `#(...)` notation
- Tensors: Nested parentheses for N-D structure
- Strings: With or without quotes (display vs write)
- Lambdas: Show source S-expression via registry lookup
- Closures: Show embedded S-expression
- Hash tables: `#<hash:N>` format

#### 13.4.2 Display Options
```c
struct eshkol_display_opts {
    int max_depth;         // Recursion limit
    uint8_t quote_strings; // Display vs write semantics
    uint8_t show_types;    // Debug: show type tags
    void* output;          // Output stream
}
```

---

## 14. Quantum RNG

### 14.1 Quantum-Inspired Random Number Generator

#### 14.1.1 Core Principles
- **Quantum Circuit Simulation:** 8-qubit quantum state
- **Hadamard Gates:** Create superposition
- **Phase Gates:** Quantum entanglement
- **Measurement Collapse:** Extract classical randomness
- **Entropy Pooling:** 16-element entropy pool
- **Runtime Entropy:** Continuous entropy injection

#### 14.1.2 API Functions

**Initialization:**
```c
qrng_error qrng_init(qrng_ctx **ctx, const uint8_t *seed, size_t seed_len)
```

**Random Generation:**
```c
uint64_t qrng_uint64(qrng_ctx *ctx)           // Random 64-bit integer
double qrng_double(qrng_ctx *ctx)             // Random [0,1)
int32_t qrng_range32(qrng_ctx *ctx, min, max) // Random in range
qrng_error qrng_bytes(qrng_ctx *ctx, uint8_t *out, size_t len)
```

**Quantum Operations:**
```c
qrng_error qrng_entangle_states(ctx, state1, state2, len)  // Create entanglement
qrng_error qrng_measure_state(ctx, state, len)             // Collapse superposition
```

#### 14.1.3 Eshkol Wrapper Functions
```c
eshkol_qrng_init()           // Initialize global context
eshkol_qrng_double()         // Get random [0,1)
eshkol_qrng_uint64()         // Get random uint64
eshkol_qrng_range(min, max)  // Get random in range
eshkol_qrng_bytes(buf, len)  // Fill buffer with random bytes
```

---

## 15. Complete Operator and Function Reference

### 15.1 All Special Forms (38 Total)

1. `define` - Variable/function definition
2. `lambda` - Anonymous function
3. `if` - Conditional
4. `cond` - Multi-way conditional
5. `case` - Switch on value
6. `match` - Pattern matching
7. `let` - Local bindings (parallel)
8. `let*` - Sequential bindings
9. `letrec` - Recursive bindings
10. `let-values` - Multiple value bindings
11. `let*-values` - Sequential multiple value bindings
12. `do` - Iteration construct
13. `begin` - Sequencing
14. `and` - Short-circuit logical AND
15. `or` - Short-circuit logical OR
16. `when` - One-armed if (true case)
17. `unless` - One-armed if (false case)
18. `quote` - Literal data
19. `quasiquote` - Template with unquotes
20. `unquote` - Escape from quasiquote
21. `unquote-splicing` - Splice into quasiquote
22. `set!` - Mutation
23. `define-type` - Type alias
24. `define-syntax` - Macro definition
25. `import` - File import (legacy)
26. `require` - Module import
27. `provide` - Symbol export
28. `extern` - External function declaration
29. `extern-var` - External variable declaration
30. `with-region` - Memory region
31. `owned` - Ownership marker
32. `move` - Transfer ownership
33. `borrow` - Temporary access
34. `shared` - Reference counting
35. `weak-ref` - Weak reference
36. `guard` - Exception handler
37. `raise` - Raise exception
38. `values` - Multiple return values
39. `call-with-values` - Consume multiple values

### 15.2 All Arithmetic/Math Functions (50+)

**Arithmetic:**
`+`, `-`, `*`, `/`, `quotient`, `remainder`, `modulo`, `abs`, `min`, `max`, `expt`

**Trigonometric:**
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`

**Hyperbolic:**
`sinh`, `cosh`, `tanh`

**Exponential/Logarithmic:**
`exp`, `log`, `sqrt`

**Rounding:**
`floor`, `ceiling`, `truncate`, `round`

**Comparison:**
`=`, `<`, `>`, `<=`, `>=`

**Numeric Predicates:**
`zero?`, `positive?`, `negative?`, `odd?`, `even?`, `exact?`, `inexact?`

### 15.3 All List Functions (60+)

**Core:**
`cons`, `car`, `cdr`, `set-car!`, `set-cdr!`, `list`, `list*`, `make-list`

**Compound Accessors:**
`caar`, `cadr`, `cdar`, `cddr`, `caaar` through `cddddr`, `first` through `tenth`

**Predicates:**
`null?`, `pair?`, `list?`

**Queries:**
`length`, `list-ref`, `list-tail`, `count-if`, `find`

**Transformations:**
`append`, `reverse`, `take`, `drop`, `filter`, `unzip`, `partition`

**Search:**
`member`, `member?`, `memq`, `memv`, `assoc`, `assq`, `assv`

**Higher-Order:**
`map`, `map1`, `map2`, `map3`, `fold`, `fold-right`, `for-each`, `any`, `every`, `apply`

**Generation:**
`iota`, `iota-from`, `iota-step`, `repeat`, `range`, `zip`

**Sorting:**
`sort`

**Conversion:**
`list->vector`, `vector->list`, `string->list`, `list->string`

### 15.4 All String Functions (30+)

**Construction:**
`string`, `make-string`, `string-append`, `substring`

**Access:**
`string-length`, `string-ref`, `string-set!`

**Comparison:**
`string=?`, `string<?`, `string>?`, `string<=?`, `string>=?`

**Utilities:**
`string-join`, `string-split`, `string-trim`, `string-upcase`, `string-downcase`, `string-replace`, `string-reverse`, `string-copy`, `string-repeat`, `string-starts-with?`, `string-ends-with?`, `string-contains?`, `string-index`, `string-count`

**Conversions:**
`string->number`, `number->string`, `string->symbol`, `symbol->string`, `string->list`, `list->string`

### 15.5 All Tensor Operations (25+)

**Creation:**
`vector`, `matrix`, `tensor`, `make-vector`, `zeros`, `ones`, `eye`, `arange`, `linspace`

**Access:**
`vref`, `tensor-get`, `tensor-set`, `vector-ref`, `vector-set!`, `vector-length`

**Arithmetic:**
`tensor-add`, `tensor-sub`, `tensor-mul`, `tensor-div`, `tensor-dot`, `matmul`

**Transformations:**
`transpose`, `reshape`, `tensor-apply`

**Reductions:**
`tensor-sum`, `tensor-mean`, `tensor-reduce-all`

**Queries:**
`tensor-shape`, `tensor-rank`, `tensor-size`

### 15.6 All Hash Table Operations (10+)

`make-hash-table`, `hash`, `hash-ref`, `hash-set!`, `hash-has-key?`, `hash-remove!`, `hash-keys`, `hash-values`, `hash-count`, `hash-clear!`, `hash-table?`

### 15.7 All Autodiff Operators (8)

**Univariate:**
`derivative`

**Multivariate:**
`gradient`, `jacobian`, `hessian`

**Vector Calculus:**
`divergence`, `curl`, `laplacian`, `directional-derivative`

### 15.8 All Type Predicates (20+)

`null?`, `boolean?`, `char?`, `string?`, `symbol?`, `number?`, `integer?`, `real?`, `complex?`, `pair?`, `list?`, `vector?`, `procedure?`, `hash-table?`, `tensor?`, `input-port?`, `output-port?`, `port?`, `eof-object?`

### 15.9 All Equality/Comparison (3)

`eq?`, `eqv?`, `equal?`

### 15.10 All I/O Functions (20+)

**Output:**
`display`, `write`, `newline`, `write-char`, `write-string`, `write-line`, `flush-output-port`

**Input:**
`read`, `read-line`, `read-char`, `peek-char`

**Ports:**
`open-input-file`, `open-output-file`, `close-port`, `current-input-port`, `current-output-port`

**Files:**
`read-file`, `write-file`, `append-file`, `file-exists?`, `file-readable?`, `file-writable?`, `file-delete`, `file-rename`, `file-size`

**Directories:**
`directory-exists?`, `make-directory`, `delete-directory`, `directory-list`, `current-directory`, `set-current-directory!`

### 15.11 All System Functions (5+)

`getenv`, `setenv`, `unsetenv`, `system`, `exit`, `sleep`, `current-seconds`, `command-line`

### 15.12 All Functional Programming (10+)

`compose`, `compose3`, `curry2`, `curry3`, `uncurry2`, `partial`, `partial1`, `partial2`, `partial3`, `flip`, `identity`, `constantly`, `negate`

### 15.13 All JSON Operations (10)

`json-parse`, `json-stringify`, `json-get`, `json-array-ref`, `hash-table->alist`, `alist->hash-table`, `json-read-file`, `json-write-file`, `alist->json`, `alist-write-json`

### 15.14 All CSV Operations (6)

`csv-parse`, `csv-parse-file`, `csv-stringify`, `csv-write-file`, `csv-parse-line`, `csv-stringify-row`

### 15.15 All Base64 Operations (6)

`base64-encode`, `base64-decode`, `base64-encode-string`, `base64-decode-string`, `string->bytes`, `bytes->string`

### 15.16 All Conversion Functions (12)

`exact->inexact`, `inexact->exact`, `string->number`, `number->string`, `string->symbol`, `symbol->string`, `char->integer`, `integer->char`, `list->vector`, `vector->list`, `string->list`, `list->string`

---

## 16. Complete Language Capabilities Summary

### Total Language Elements:
- **Special Forms:** 39
- **Built-in Operators:** 15 (arithmetic/logical)
- **Math Functions:** 20
- **List Functions:** 60+
- **String Functions:** 30+
- **Tensor Operations:** 25+
- **Hash Table Operations:** 10
- **Autodiff Operators:** 8
- **I/O Functions:** 20+
- **System Functions:** 8
- **Type Predicates:** 20+
- **Equality/Comparison:** 9
- **Functional Programming:** 15
- **JSON Operations:** 10
- **CSV Operations:** 6
- **Base64 Operations:** 6
- **Memory Management:** 6
- **REPL Commands:** 14

### **GRAND TOTAL: 300+ Language Features**

---

## 17. Compiler Capabilities

### 17.1 Supported Optimizations
1. **Tail Call Elimination** - Self-recursive functions → loops
2. **Closure Optimization** - Static capture analysis
3. **Inline Expansion** - Small functions inlined
4. **Type Specialization** - Monomorphic fast paths
5. **Constant Folding** - Compile-time evaluation
6. **Dead Code Elimination** - LLVM optimization passes

### 17.2 Compilation Modes

#### Standard Compilation
```bash
eshkol-run input.esk -o output
```

#### Library Compilation
```bash
eshkol-run --shared-lib input.esk -o library.o
```

#### IR Dump
```bash
eshkol-run --dump-ir input.esk  # Produces input.ll
```

#### AST Dump
```bash
eshkol-run --dump-ast input.esk  # Produces input.ast
```

### 17.3 Linking

**Auto-linking stdlib:**
If source requires stdlib, `stdlib.o` is automatically linked.

**Manual linking:**
```bash
eshkol-run input.esk stdlib.o utils.o -o output
```

---

## 18. Implementation Details

### 18.1 Numeric Tower Promotion

**Rules:**
- `int + int` → `int`
- `int + float` → `float`
- `float + float` → `float`
- Operations preserve exactness when possible

### 18.2 Function Calling Convention

**Standard Functions:**
```c
eshkol_tagged_value_t func(eshkol_tagged_value_t arg1, ...)
```

**Closures:**
```c
eshkol_tagged_value_t func(args..., captures...)
```

**Variadic Functions:**
- Fixed parameters first
- Rest parameter as tagged cons list

### 18.3 Exception Handling Mechanism

**Implementation:**
- C `setjmp`/`longjmp` for non-local exits
- Exception handler stack
- Global `g_current_exception` pointer
- Automatic handler cleanup

### 18.4 Module Name Mangling

**Private Symbols:**
- Module: `foo.bar.baz`
- Symbol: `helper`
- Mangled: `__foo_bar_baz__helper`

**Public Symbols:**
Keep original name (exported via `provide`)

---

## 19. Version Information

**Current Version:** 1.0.0-foundation

**Version History:**
- v1.0.0-foundation - Initial stable release
  - Core Scheme compatibility
  - Automatic differentiation system
  - HoTT type system foundation
  - Arena memory management
  - Module system
  - REPL with JIT

---

## 20. File Organization

### Source Code Structure
```
inc/eshkol/              # Public headers
  eshkol.h               # Main API header
  llvm_backend.h         # LLVM backend API
  logger.h               # Logging utilities
  backend/               # Backend codegen headers
  frontend/              # Frontend headers
  types/                 # Type system headers

lib/                     # Implementation
  core/                  # Core runtime
  frontend/              # Parser
  backend/               # LLVM codegen
  types/                 # Type checker
  repl/                  # REPL/JIT
  quantum/               # Quantum RNG
  *.esk                  # Standard library modules

exe/                     # Executables
  eshkol-run.cpp         # Compiler
  eshkol-repl.cpp        # REPL

tests/                   # Test suites
  autodiff/              # AD tests
  lists/                 # List operation tests
  features/              # Language feature tests
  ml/                    # Machine learning tests
  neural/                # Neural network tests
  types/                 # Type system tests
```

---

## Conclusion

This document provides a **complete** specification of the Eshkol programming language version 1.0.0-foundation, documenting **every** feature, function, operator, and capability found in the implementation.

**Total Coverage:**
- ✅ All 39 special forms
- ✅ All 300+ built-in functions and operators
- ✅ Complete type system (15+ types with subtypes)
- ✅ Full memory management system
- ✅ Entire standard library
- ✅ Complete autodiff system
- ✅ Module system
- ✅ REPL features
- ✅ Quantum RNG
- ✅ Compilation pipeline
- ✅ Runtime architecture
