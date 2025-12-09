# Eshkol v1.0 Implementation Plan
## Roadmap to Production-Ready Scientific Computing Language

**Document Version**: 1.0
**Created**: December 2025
**Target Completion**: 14 weeks from start

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [Gap Analysis](#3-gap-analysis)
4. [Implementation Phases](#4-implementation-phases)
5. [Detailed Implementation Specifications](#5-detailed-implementation-specifications)
6. [File-by-File Modification Guide](#6-file-by-file-modification-guide)
7. [Testing Strategy](#7-testing-strategy)
8. [Quick Wins](#8-quick-wins)
9. [Post-v1.0 Roadmap](#9-post-v10-roadmap)
10. [Appendices](#10-appendices)

---

## 1. Executive Summary

### 1.1 Current Completion: ~90%

Eshkol is a remarkably complete scientific computing language with:
- **25,219 lines** of LLVM codegen
- **253 test files** (all passing)
- **360+ functions** (200 builtin + 160 stdlib)
- Full automatic differentiation
- HoTT type system with dependent types
- Module system with cycle detection
- Comprehensive I/O and data format support

### 1.2 Remaining Work: ~10%

| Category | Gap | Effort |
|----------|-----|--------|
| Error Handling | No exceptions | 2 weeks |
| Pattern Matching | No `match` | 3 weeks |
| Metaprogramming | No macros/quasiquote | 5 weeks |
| Multi-values | No `values`/`call-with-values` | 1 week |
| Tooling | Basic only | 3 weeks |
| **Total** | | **14 weeks** |

### 1.3 What's NOT Needed for v1.0

- `call/cc` (continuations) — rarely used in scientific computing
- GPU support — defer to v1.1
- Full R7RS compliance — focus on scientific use cases
- Hygenic macro system — simple `syntax-rules` sufficient

---

## 2. Current State Assessment

### 2.1 Codebase Statistics

```
Component                    Lines of Code    Status
─────────────────────────────────────────────────────
lib/backend/llvm_codegen.cpp    25,219       Production
lib/frontend/parser.cpp          4,461       Production
lib/core/arena_memory.cpp        2,325       Production
lib/types/*.cpp                  ~3,000      Production
lib/core/**/*.esk                ~2,500      Production
tests/**/*.esk                     253 files All passing
─────────────────────────────────────────────────────
Total C/C++                     ~50,000+
Total Eshkol stdlib              ~2,500
```

### 2.2 Complete Feature Inventory

#### 2.2.1 Special Forms (18 implemented)

| Form | Status | Location |
|------|--------|----------|
| `if` | ✅ Complete | parser.cpp:356 |
| `lambda` | ✅ Complete | parser.cpp:357 |
| `let` | ✅ Complete | parser.cpp:358 |
| `let*` | ✅ Complete | parser.cpp:359 |
| `letrec` | ✅ Complete | parser.cpp:360 |
| `and` | ✅ Complete | parser.cpp:361 |
| `or` | ✅ Complete | parser.cpp:362 |
| `cond` | ✅ Complete | parser.cpp:363 |
| `case` | ✅ Complete | parser.cpp:364 |
| `do` | ✅ Complete | parser.cpp:365 |
| `when` | ✅ Complete | parser.cpp:366 |
| `unless` | ✅ Complete | parser.cpp:367 |
| `quote` | ✅ Complete | parser.cpp:368 |
| `define` | ✅ Complete | parser.cpp:370 |
| `define-type` | ✅ Complete | parser.cpp:371 |
| `set!` | ✅ Complete | parser.cpp:372 |
| `begin` | ✅ Complete | (implicit in parser) |
| `extern` | ✅ Complete | parser.cpp:383 |

#### 2.2.2 Builtin Functions (200+)

**Arithmetic (15)**
```
+ - * / modulo mod % remainder quotient gcd lcm min max pow expt abs
```

**Math Functions (25)**
```
sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh
exp exp2 log log10 log2 sqrt cbrt floor ceiling ceil round truncate trunc fabs
```

**Comparison & Logic (10)**
```
< > = <= >= eq? eqv? equal? and or not
```

**Type Predicates (18)**
```
number? integer? real? boolean? string? symbol? char? null? pair? list?
vector? procedure? hash-table? positive? negative? zero? even? odd?
```

**List Operations (30+)**
```
cons car cdr list null? pair? length
cadr caddr cadddr caar cdar cddr (all 28 c*r variants)
set-car! set-cdr! map apply reduce
list* acons split-at remove remq remv last last-pair
```

**Vector/Tensor Operations (25+)**
```
vector make-vector vector-ref vector-set! vector-length vector?
vref tensor-get tensor-set tensor-add tensor-sub tensor-mul tensor-div
tensor-dot tensor-shape tensor-apply tensor-reduce tensor-reduce-all
zeros ones eye arange linspace reshape transpose flatten
matmul trace norm outer tensor-sum tensor-mean
```

**Automatic Differentiation (8)**
```
derivative gradient jacobian hessian divergence curl laplacian directional-derivative
```

**String Operations (20+)**
```
string-length string-ref string-append substring
string=? string<? string>? string<=? string>=?
number->string string->number make-string string-set!
string->list list->string string-split string-contains?
string-index string-upcase string-downcase
```

**Character Operations (7)**
```
char? char->integer integer->char char=? char<? char>? char<=? char>=?
```

**I/O Operations (15)**
```
display newline error
open-input-file read-line close-port eof-object?
open-output-file write-string write-line write-char flush-output-port
read-file write-file append-file
```

**File System (12)**
```
file-exists? file-readable? file-writable? file-delete file-rename file-size
directory-exists? make-directory delete-directory directory-list
current-directory set-current-directory!
```

**System Operations (8)**
```
getenv setenv unsetenv system sleep current-seconds exit command-line
```

**Hash Table Operations (10)**
```
make-hash-table hash-table? hash-ref hash-set!
hash-has-key? hash-remove! hash-keys hash-values hash-count hash-clear!
```

**Random (4)**
```
random quantum-random quantum-random-int quantum-random-range
```

#### 2.2.3 Standard Library Modules (160+ functions)

| Module | Functions | Purpose |
|--------|-----------|---------|
| `core.io` | 2 | print, println |
| `core.operators.arithmetic` | 4 | add, sub, mul, div |
| `core.operators.compare` | 5 | lt, gt, le, ge, eq |
| `core.logic.predicates` | 5 | is-zero?, is-even?, etc. |
| `core.logic.boolean` | 3 | negate, all?, none? |
| `core.logic.types` | 2 | is-null?, is-pair? |
| `core.functional.compose` | 4 | compose, identity, constantly |
| `core.functional.curry` | 6 | curry2/3, partial1/2/3 |
| `core.functional.flip` | 1 | flip |
| `core.control.trampoline` | 3 | trampoline, bounce, done |
| `core.list.compound` | 38 | caar..cddddr, first..tenth |
| `core.list.generate` | 7 | iota, range, zip, repeat |
| `core.list.transform` | 8 | take, drop, filter, partition |
| `core.list.query` | 3 | count-if, find, length |
| `core.list.higher_order` | 8 | map1/2/3, fold, any, every |
| `core.list.search` | 9 | member, assoc, list-ref |
| `core.list.sort` | 1 | sort (merge sort) |
| `core.list.convert` | 2 | list->vector, vector->list |
| `core.strings` | 17 | trim, replace, split, etc. |
| `core.json` | 15 | parse, stringify, file I/O |
| `core.data.csv` | 10 | RFC 4180 CSV |
| `core.data.base64` | 10 | encode, decode |
| `lib/math.esk` | 20+ | det, inv, solve, integrate |

#### 2.2.4 Type System (HoTT)

**Implemented**:
- Universe levels (U0, U1, U2, UOmega)
- Primitive types (Int64, Float64, String, Boolean, Char, Symbol, Null)
- Type constructors (List, Vector, Tensor, Function, Pair, HashTable)
- Arrow types `(-> a b c)`
- Container types `(list a)`, `(vector a)`, `(tensor a)`
- Product types `(* a b)`, Sum types `(+ a b)`
- Forall types `(forall (a) body)`
- Dependent types with CTValue
- Linear types (use-exactly-once)
- Borrow checking (Owned, Moved, Borrowed)
- Bidirectional type checking
- Type aliases with parameters

**Type Annotations Syntax**:
```scheme
(define (add (x : int) (y : int)) : int
  (+ x y))

(lambda ((x : real)) : real (* x x))

(define-type IntList (list int))
(define-type (Pair a b) (* a b))
```

#### 2.2.5 Memory Management (OALR)

**Implemented**:
```scheme
(with-region body ...)      ; Lexical region allocation
(owned value)               ; Linear ownership
(move value)                ; Transfer ownership
(borrow value body ...)     ; Temporary access
(shared value)              ; Reference counted
(weak-ref value)            ; Weak reference
```

#### 2.2.6 Module System

**Implemented**:
```scheme
(require core.json core.list.sort)   ; Import modules
(provide func1 func2 var1)           ; Export symbols
(import "path/to/file.esk")          ; File import (legacy)
```

Features:
- Cycle detection via DFS
- Topological sorting for load order
- Symbol visibility per module
- Qualified names (e.g., `core.list.transform`)

---

## 3. Gap Analysis

### 3.1 Critical Gaps (Must Fix)

#### Gap 1: Exception Handling
**Current State**: No structured error handling
**Impact**: Cannot recover from errors gracefully
**Needed**:
```scheme
(guard (e ((type-error? e) (handle-type-error e))
          ((file-error? e) (handle-file-error e))
          (else (re-raise e)))
  (risky-operation))

(raise (make-error 'type-error "Expected integer"))
(error "message" irritant1 irritant2)
```

#### Gap 2: Pattern Matching
**Current State**: Only `cond` and `case`
**Impact**: Verbose destructuring code
**Needed**:
```scheme
(match value
  ((list 'point x y) (sqrt (+ (* x x) (* y y))))
  ((list 'circle _ _ r) (* pi r r))
  ((? number? n) (* n 2))
  ((cons h t) (process h t))
  (_ 'default))
```

#### Gap 3: Quasiquotation
**Current State**: Only `quote`
**Impact**: No code generation capability
**Needed**:
```scheme
`(1 2 ,(+ 1 2) ,@(list 4 5))  ; => (1 2 3 4 5)
```

#### Gap 4: Multiple Return Values
**Current State**: Not implemented
**Impact**: Cannot return multiple values efficiently
**Needed**:
```scheme
(define (quotient-remainder a b)
  (values (quotient a b) (remainder a b)))

(call-with-values
  (lambda () (quotient-remainder 10 3))
  (lambda (q r) (list q r)))

(let-values (((q r) (quotient-remainder 10 3)))
  (+ q r))
```

### 3.2 Important Gaps (Should Fix)

#### Gap 5: Basic Macros
**Current State**: None
**Impact**: No syntactic abstraction
**Needed** (minimal):
```scheme
(define-syntax my-when
  (syntax-rules ()
    ((my-when test body ...)
     (if test (begin body ...) #f))))
```

#### Gap 6: `apply` with Fixed Arguments
**Current State**: `(apply + '(1 2 3))` works
**Needed**: `(apply + 1 2 '(3 4 5))` — prepend fixed args
**Impact**: Standard Scheme compatibility

#### Gap 7: Improved Error Messages
**Current State**: Basic error text
**Needed**:
```
Error at line 42, column 15 in file.esk:
  Type mismatch in function application
  Expected: (-> integer integer integer)
  Got:      (-> string integer integer)

  41 | (define (add x y)
  42 |   (+ x "hello"))
             ^^^^^^^
  Hint: The second argument should be an integer, not a string.
```

### 3.3 Nice-to-Have Gaps (Can Defer)

| Gap | Impact | Effort | Priority |
|-----|--------|--------|----------|
| `call/cc` | Rare use case | 4 weeks | Low |
| SIMD vectorization | Performance | 3 weeks | Medium |
| Parallel primitives | Performance | 4 weeks | Medium |
| GPU backend | Scale | 3 months | Low |
| Package manager | Distribution | 3 weeks | Medium |
| LSP server | Dev experience | 4 weeks | Medium |
| Profiler | Debugging | 2 weeks | Low |

---

## 4. Implementation Phases

### Phase 1: Error Handling (Weeks 1-2)

**Goal**: Implement R7RS-compatible exception system

**Week 1**:
- Add `error` function (already partially exists)
- Add `guard` special form to parser
- Implement exception value type
- Add `raise` and `raise-continuable`

**Week 2**:
- Implement condition types
- Add error predicates (`error-object?`, `file-error?`, etc.)
- Add `with-exception-handler`
- Add `error-object-message`, `error-object-irritants`
- Write tests

**Deliverables**:
- New AST node: `ESHKOL_GUARD_OP`
- New runtime functions: `eshkol_raise`, `eshkol_guard`
- Tests: `tests/exceptions/`

### Phase 2: Multiple Values & Apply Fix (Week 3)

**Goal**: Implement `values` and `call-with-values`, fix `apply`

**Tasks**:
- Add new tagged value type: `ESHKOL_VALUE_MULTI`
- Implement `values` (pack multiple values)
- Implement `call-with-values` (unpack and apply)
- Implement `let-values` and `let*-values`
- Fix `apply` to handle `(apply proc arg1 arg2 ... arglist)`

**Deliverables**:
- New value type in `eshkol.h`
- New codegen in `llvm_codegen.cpp`
- Tests: `tests/multivalue/`

### Phase 3: Quasiquotation (Weeks 4-5)

**Goal**: Implement quasiquote, unquote, unquote-splicing

**Week 4**:
- Add lexer tokens: BACKQUOTE, COMMA, COMMA_AT
- Add parser rules for quasiquote nesting
- Implement basic quasiquote expansion

**Week 5**:
- Handle nested quasiquotes correctly
- Implement `unquote-splicing` (`,@`)
- Optimize constant quasiquotes
- Write tests

**Deliverables**:
- New tokens in parser
- New AST nodes: `ESHKOL_QUASIQUOTE_OP`, `ESHKOL_UNQUOTE_OP`, `ESHKOL_UNQUOTE_SPLICING_OP`
- Expansion logic in codegen
- Tests: `tests/quasiquote/`

### Phase 4: Pattern Matching (Weeks 6-8)

**Goal**: Implement `match` expression

**Week 6**:
- Design pattern AST representation
- Add `match` to parser
- Implement literal patterns
- Implement variable patterns

**Week 7**:
- Implement constructor patterns (cons, list, vector)
- Implement predicate patterns (`(? pred)`)
- Implement `_` wildcard
- Implement `...` repetition (basic)

**Week 8**:
- Implement pattern guards (`(pat when expr)`)
- Implement `match-let` and `match-lambda`
- Optimize pattern compilation
- Write comprehensive tests

**Deliverables**:
- New AST node: `ESHKOL_MATCH_OP`
- Pattern compilation to decision tree
- Tests: `tests/pattern/`

### Phase 5: Basic Macros (Weeks 9-12)

**Goal**: Implement `syntax-rules` macro system

**Week 9**:
- Design macro representation
- Implement `define-syntax` parser
- Implement pattern matching for macro patterns
- Store macro definitions in environment

**Week 10**:
- Implement template instantiation
- Implement `...` ellipsis handling
- Handle literal identifiers
- Basic expansion

**Week 11**:
- Implement macro expansion phase (before codegen)
- Handle nested macro definitions
- Implement `let-syntax` and `letrec-syntax`

**Week 12**:
- Debug edge cases
- Optimize expansion
- Write standard macro library
- Comprehensive testing

**Deliverables**:
- New AST node: `ESHKOL_DEFINE_SYNTAX_OP`
- Macro expansion pass
- Standard macros: `when`, `unless`, `case`, `and`, `or` as macros
- Tests: `tests/macros/`

### Phase 6: Error Messages & Polish (Weeks 13-14)

**Goal**: Production-quality error messages and final polish

**Week 13**:
- Improve type error messages
- Add source context to all errors
- Implement error hints
- Color output for terminal

**Week 14**:
- Final integration testing
- Performance benchmarks
- Documentation updates
- Release preparation

**Deliverables**:
- Enhanced error formatting
- Updated documentation
- Benchmark suite
- Release notes

---

## 5. Detailed Implementation Specifications

### 5.1 Exception Handling

#### 5.1.1 New Types (`eshkol.h`)

```c
// Add to eshkol_value_type_t
ESHKOL_VALUE_EXCEPTION = 16,  // Exception object

// Exception structure
typedef struct eshkol_exception {
    char* type;                    // Exception type symbol
    char* message;                 // Error message
    eshkol_tagged_value_t* irritants;  // List of irritants
    size_t num_irritants;
    uint32_t line;                 // Source location
    uint32_t column;
    char* filename;
} eshkol_exception_t;
```

#### 5.1.2 New Operations (`eshkol.h`)

```c
// Add to eshkol_op_t
ESHKOL_GUARD_OP,                  // (guard (var clause ...) body)
ESHKOL_RAISE_OP,                  // (raise exception)
ESHKOL_WITH_EXCEPTION_HANDLER_OP, // (with-exception-handler handler thunk)
```

#### 5.1.3 Parser Changes (`parser.cpp`)

Add to `get_operator_type`:
```cpp
if (op == "guard") return ESHKOL_GUARD_OP;
if (op == "raise") return ESHKOL_RAISE_OP;
if (op == "with-exception-handler") return ESHKOL_WITH_EXCEPTION_HANDLER_OP;
```

Add guard parsing (after line ~2800):
```cpp
case ESHKOL_GUARD_OP: {
    // (guard (var clause ...) body ...)
    // clause = (test expr ...) | (test => proc)
    Token var_token = tokenizer.nextToken();
    if (var_token.type != TOKEN_LPAREN) {
        eshkol_error("guard requires (var clause ...) form");
        return ast;
    }
    // Parse variable name
    Token var_name = tokenizer.nextToken();
    ast.operation.guard_op.var_name = strdup(var_name.value.c_str());

    // Parse clauses until )
    std::vector<guard_clause_t> clauses;
    while (true) {
        Token peek = tokenizer.peekToken();
        if (peek.type == TOKEN_RPAREN) {
            tokenizer.nextToken();
            break;
        }
        // Parse (test expr ...) clause
        guard_clause_t clause = parse_guard_clause(tokenizer);
        clauses.push_back(clause);
    }
    // ... store clauses and parse body
    break;
}
```

#### 5.1.4 Codegen Changes (`llvm_codegen.cpp`)

```cpp
// Add exception handling
Value* codegenGuard(const eshkol_operations_t* op) {
    // Create exception landing pad
    BasicBlock* try_block = BasicBlock::Create(ctx_.context(), "try", current_func);
    BasicBlock* catch_block = BasicBlock::Create(ctx_.context(), "catch", current_func);
    BasicBlock* cont_block = BasicBlock::Create(ctx_.context(), "cont", current_func);

    // Set up exception variable in scope
    ctx_.pushScope();
    Value* exception_var = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
    ctx_.addLocal(op->guard_op.var_name, exception_var);

    // Try block
    ctx_.builder().SetInsertPoint(try_block);
    Value* body_result = codegenSequence(op->guard_op.body, op->guard_op.num_body);
    ctx_.builder().CreateBr(cont_block);

    // Catch block - match clauses
    ctx_.builder().SetInsertPoint(catch_block);
    // Load exception into variable
    ctx_.builder().CreateStore(/* current exception */, exception_var);

    // Generate clause matching
    for (auto& clause : clauses) {
        // Test condition
        Value* test = codegenAST(&clause.test);
        BasicBlock* then_block = BasicBlock::Create(ctx_.context(), "then", current_func);
        BasicBlock* else_block = BasicBlock::Create(ctx_.context(), "else", current_func);
        ctx_.builder().CreateCondBr(test, then_block, else_block);

        ctx_.builder().SetInsertPoint(then_block);
        Value* result = codegenSequence(clause.exprs, clause.num_exprs);
        ctx_.builder().CreateBr(cont_block);

        ctx_.builder().SetInsertPoint(else_block);
    }
    // If no clause matches, re-raise
    ctx_.builder().CreateCall(getRaiseFunc(), {exception_var});
    ctx_.builder().CreateUnreachable();

    // Continuation
    ctx_.builder().SetInsertPoint(cont_block);
    PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
    // ... add incoming values from try and catch blocks

    ctx_.popScope();
    return phi;
}
```

#### 5.1.5 Runtime Support (`arena_memory.cpp`)

```cpp
// Global exception handler stack
thread_local std::vector<std::function<void(eshkol_exception_t*)>> g_exception_handlers;

extern "C" void eshkol_raise(eshkol_exception_t* exception) {
    if (g_exception_handlers.empty()) {
        // Unhandled exception - print and exit
        fprintf(stderr, "Unhandled exception: %s\n", exception->message);
        if (exception->filename) {
            fprintf(stderr, "  at %s:%d:%d\n",
                    exception->filename, exception->line, exception->column);
        }
        exit(1);
    }
    // Call topmost handler
    auto handler = g_exception_handlers.back();
    handler(exception);
}

extern "C" void eshkol_push_exception_handler(void (*handler)(eshkol_exception_t*)) {
    g_exception_handlers.push_back(handler);
}

extern "C" void eshkol_pop_exception_handler() {
    g_exception_handlers.pop_back();
}
```

### 5.2 Multiple Return Values

#### 5.2.1 New Types (`eshkol.h`)

```c
// Add to eshkol_value_type_t
ESHKOL_VALUE_MULTI = 17,  // Multiple values wrapper

// Multiple values structure
typedef struct eshkol_multi_value {
    size_t count;
    eshkol_tagged_value_t values[];  // Flexible array
} eshkol_multi_value_t;
```

#### 5.2.2 Codegen (`llvm_codegen.cpp`)

```cpp
// (values v1 v2 ...)
Value* codegenValues(const eshkol_operations_t* op) {
    size_t count = op->call_op.num_vars;

    // Allocate multi-value struct
    Value* size = ConstantInt::get(ctx_.int64Type(),
        sizeof(size_t) + count * sizeof(eshkol_tagged_value_t));
    Value* ptr = mem_.arenaAllocate(size);

    // Store count
    Value* count_ptr = ctx_.builder().CreateBitCast(ptr,
        PointerType::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), count), count_ptr);

    // Store each value
    for (size_t i = 0; i < count; i++) {
        Value* val = codegenAST(&op->call_op.variables[i]);
        Value* slot = ctx_.builder().CreateGEP(/* compute offset */);
        ctx_.builder().CreateStore(val, slot);
    }

    // Return tagged pointer
    return tagged_.packPtr(ptr, ESHKOL_VALUE_MULTI);
}

// (call-with-values producer consumer)
Value* codegenCallWithValues(const eshkol_operations_t* op) {
    // Call producer (should return multi-value or single value)
    Value* producer = codegenAST(&op->call_op.variables[0]);
    Value* produced = ctx_.builder().CreateCall(producer, {});

    // Check if multi-value
    Value* is_multi = /* check type tag */;

    BasicBlock* multi_block = BasicBlock::Create(ctx_.context(), "multi");
    BasicBlock* single_block = BasicBlock::Create(ctx_.context(), "single");
    BasicBlock* cont_block = BasicBlock::Create(ctx_.context(), "cont");

    ctx_.builder().CreateCondBr(is_multi, multi_block, single_block);

    // Multi-value case: unpack and call consumer with all values
    ctx_.builder().SetInsertPoint(multi_block);
    // ... unpack values and call consumer

    // Single-value case: call consumer with one argument
    ctx_.builder().SetInsertPoint(single_block);
    Value* consumer = codegenAST(&op->call_op.variables[1]);
    Value* single_result = ctx_.builder().CreateCall(consumer, {produced});
    ctx_.builder().CreateBr(cont_block);

    ctx_.builder().SetInsertPoint(cont_block);
    PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
    // ... add incoming values
    return phi;
}
```

### 5.3 Quasiquotation

#### 5.3.1 Lexer Changes (`parser.cpp`)

```cpp
// Add to TokenType enum
TOKEN_BACKQUOTE,      // `
TOKEN_COMMA,          // ,
TOKEN_COMMA_AT,       // ,@

// Add to nextToken() switch
case '`':
    pos++;
    column_++;
    return {TOKEN_BACKQUOTE, "`", pos - 1, tok_line, tok_col};
case ',':
    if (pos + 1 < length && input[pos + 1] == '@') {
        pos += 2;
        column_ += 2;
        return {TOKEN_COMMA_AT, ",@", pos - 2, tok_line, tok_col};
    }
    pos++;
    column_++;
    return {TOKEN_COMMA, ",", pos - 1, tok_line, tok_col};
```

#### 5.3.2 Parser Changes (`parser.cpp`)

```cpp
// Add to eshkol_op_t
ESHKOL_QUASIQUOTE_OP,
ESHKOL_UNQUOTE_OP,
ESHKOL_UNQUOTE_SPLICING_OP,

// Parse quasiquote
eshkol_ast_t parse_quasiquote(SchemeTokenizer& tokenizer, int depth) {
    Token token = tokenizer.nextToken();

    if (token.type == TOKEN_BACKQUOTE) {
        // Nested quasiquote - increase depth
        eshkol_ast_t inner = parse_quasiquote(tokenizer, depth + 1);
        // Wrap in quasiquote node
        return make_quasiquote_node(inner, depth);
    }

    if (token.type == TOKEN_COMMA) {
        if (depth == 0) {
            eshkol_error("unquote outside of quasiquote");
        }
        // Unquote - decrease depth
        eshkol_ast_t inner = parse_quasiquote(tokenizer, depth - 1);
        return make_unquote_node(inner);
    }

    if (token.type == TOKEN_COMMA_AT) {
        if (depth == 0) {
            eshkol_error("unquote-splicing outside of quasiquote");
        }
        eshkol_ast_t inner = parse_quasiquote(tokenizer, depth - 1);
        return make_unquote_splicing_node(inner);
    }

    if (token.type == TOKEN_LPAREN) {
        // Parse list, recursively handling quasiquote elements
        std::vector<eshkol_ast_t> elements;
        while (true) {
            Token peek = tokenizer.peekToken();
            if (peek.type == TOKEN_RPAREN) {
                tokenizer.nextToken();
                break;
            }
            elements.push_back(parse_quasiquote(tokenizer, depth));
        }
        return make_quasiquote_list(elements, depth);
    }

    // Atom - return as quoted data
    return parse_atom(token);
}
```

#### 5.3.3 Codegen (`llvm_codegen.cpp`)

```cpp
Value* codegenQuasiquote(const eshkol_ast_t* ast, int depth) {
    if (ast->type == ESHKOL_OP && ast->operation.op == ESHKOL_UNQUOTE_OP) {
        if (depth == 1) {
            // Evaluate the unquoted expression
            return codegenAST(ast->operation.unquote_op.expr);
        } else {
            // Nested unquote - keep as syntax
            return buildUnquoteForm(codegenQuasiquote(ast->operation.unquote_op.expr, depth - 1));
        }
    }

    if (ast->type == ESHKOL_OP && ast->operation.op == ESHKOL_UNQUOTE_SPLICING_OP) {
        if (depth == 1) {
            // Evaluate and splice
            return codegenAST(ast->operation.unquote_op.expr);
            // Caller must handle splicing into list
        } else {
            return buildUnquoteSplicingForm(codegenQuasiquote(ast->operation.unquote_op.expr, depth - 1));
        }
    }

    if (ast->type == ESHKOL_CONS) {
        // Build list, handling splicing
        std::vector<Value*> elements;
        eshkol_ast_t* current = ast;

        while (current->type == ESHKOL_CONS) {
            eshkol_ast_t* elem = current->cons_cell.car;

            if (elem->type == ESHKOL_OP &&
                elem->operation.op == ESHKOL_UNQUOTE_SPLICING_OP &&
                depth == 1) {
                // Splice: append the evaluated list
                Value* splice_list = codegenAST(elem->operation.unquote_op.expr);
                // Append splice_list elements to elements
                // ... (use list-append at runtime)
            } else {
                elements.push_back(codegenQuasiquote(elem, depth));
            }
            current = current->cons_cell.cdr;
        }

        return buildList(elements);
    }

    // Atom - quote it
    return codegenQuotedAtom(ast);
}
```

### 5.4 Pattern Matching

#### 5.4.1 Pattern AST (`eshkol.h`)

```c
typedef enum {
    PATTERN_LITERAL,      // 42, "hello", #t
    PATTERN_VARIABLE,     // x, y, z (binds)
    PATTERN_WILDCARD,     // _
    PATTERN_CONS,         // (cons h t)
    PATTERN_LIST,         // (list a b c) or (a b c)
    PATTERN_VECTOR,       // #(a b c)
    PATTERN_PREDICATE,    // (? pred) or (? pred var)
    PATTERN_AND,          // (and p1 p2 ...)
    PATTERN_OR,           // (or p1 p2 ...)
    PATTERN_NOT,          // (not p)
    PATTERN_QUOTE,        // 'symbol or '(data)
    PATTERN_QUASIQUOTE,   // `(,x ,@rest)
} pattern_type_t;

typedef struct pattern {
    pattern_type_t type;
    union {
        eshkol_ast_t literal;           // For LITERAL
        char* var_name;                  // For VARIABLE
        struct {
            struct pattern* car;
            struct pattern* cdr;
        } cons_pat;                      // For CONS
        struct {
            struct pattern** elements;
            size_t count;
        } list_pat;                      // For LIST
        struct {
            eshkol_ast_t* predicate;
            char* var_name;              // Optional binding
        } pred_pat;                      // For PREDICATE
        struct {
            struct pattern** patterns;
            size_t count;
        } compound_pat;                  // For AND, OR
        struct pattern* negated;         // For NOT
    };
} pattern_t;

// Match clause
typedef struct match_clause {
    pattern_t* pattern;
    eshkol_ast_t* guard;                 // Optional (when expr)
    eshkol_ast_t* body;
    size_t num_body;
} match_clause_t;

// Match operation
struct {
    eshkol_ast_t* expr;                  // Expression to match
    match_clause_t* clauses;
    size_t num_clauses;
} match_op;
```

#### 5.4.2 Pattern Compilation (`llvm_codegen.cpp`)

```cpp
// Compile pattern to decision tree
Value* codegenMatch(const eshkol_operations_t* op) {
    // Evaluate the scrutinee
    Value* scrutinee = codegenAST(op->match_op.expr);
    Value* scrutinee_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
    ctx_.builder().CreateStore(scrutinee, scrutinee_ptr);

    // Create blocks for each clause
    BasicBlock* fail_block = BasicBlock::Create(ctx_.context(), "match_fail", current_func);
    BasicBlock* cont_block = BasicBlock::Create(ctx_.context(), "match_cont", current_func);

    std::vector<std::pair<BasicBlock*, Value*>> clause_results;

    for (size_t i = 0; i < op->match_op.num_clauses; i++) {
        match_clause_t& clause = op->match_op.clauses[i];

        BasicBlock* clause_block = BasicBlock::Create(ctx_.context(),
            "clause_" + std::to_string(i), current_func);
        BasicBlock* next_block = (i + 1 < op->match_op.num_clauses)
            ? BasicBlock::Create(ctx_.context(), "try_" + std::to_string(i + 1), current_func)
            : fail_block;

        ctx_.builder().SetInsertPoint(clause_block);

        // Push scope for pattern bindings
        ctx_.pushScope();

        // Compile pattern match
        Value* matched = compilePattern(clause.pattern, scrutinee_ptr, next_block);

        // Check guard if present
        if (clause.guard) {
            Value* guard_result = codegenAST(clause.guard);
            Value* guard_true = /* check if truthy */;
            ctx_.builder().CreateCondBr(guard_true, /* body block */, next_block);
        }

        // Compile body
        Value* body_result = codegenSequence(clause.body, clause.num_body);
        clause_results.push_back({ctx_.builder().GetInsertBlock(), body_result});
        ctx_.builder().CreateBr(cont_block);

        ctx_.popScope();
    }

    // Fail block - raise match error
    ctx_.builder().SetInsertPoint(fail_block);
    ctx_.builder().CreateCall(getErrorFunc(),
        {makeString("no matching pattern")});
    ctx_.builder().CreateUnreachable();

    // Continuation with PHI
    ctx_.builder().SetInsertPoint(cont_block);
    PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), clause_results.size());
    for (auto& [block, value] : clause_results) {
        phi->addIncoming(value, block);
    }

    return phi;
}

// Compile a single pattern
Value* compilePattern(const pattern_t* pat, Value* scrutinee_ptr, BasicBlock* fail_block) {
    switch (pat->type) {
        case PATTERN_LITERAL: {
            Value* scrutinee = ctx_.builder().CreateLoad(ctx_.taggedValueType(), scrutinee_ptr);
            Value* literal = codegenQuotedData(&pat->literal);
            Value* eq = ctx_.builder().CreateCall(getEqualFunc(), {scrutinee, literal});
            BasicBlock* cont = BasicBlock::Create(ctx_.context(), "lit_ok", current_func);
            ctx_.builder().CreateCondBr(eq, cont, fail_block);
            ctx_.builder().SetInsertPoint(cont);
            return eq;
        }

        case PATTERN_VARIABLE: {
            // Bind variable to scrutinee
            Value* scrutinee = ctx_.builder().CreateLoad(ctx_.taggedValueType(), scrutinee_ptr);
            Value* var_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
            ctx_.builder().CreateStore(scrutinee, var_ptr);
            ctx_.addLocal(pat->var_name, var_ptr);
            return ConstantInt::getTrue(ctx_.context());
        }

        case PATTERN_WILDCARD: {
            return ConstantInt::getTrue(ctx_.context());
        }

        case PATTERN_CONS: {
            // Check if pair
            Value* scrutinee = ctx_.builder().CreateLoad(ctx_.taggedValueType(), scrutinee_ptr);
            Value* is_pair = codegenTypePredicate(scrutinee, ESHKOL_VALUE_CONS_PTR);
            BasicBlock* pair_block = BasicBlock::Create(ctx_.context(), "is_pair", current_func);
            ctx_.builder().CreateCondBr(is_pair, pair_block, fail_block);
            ctx_.builder().SetInsertPoint(pair_block);

            // Extract car and cdr
            Value* car_val = codegenCarPrimitive(scrutinee);
            Value* cdr_val = codegenCdrPrimitive(scrutinee);

            Value* car_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
            Value* cdr_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
            ctx_.builder().CreateStore(car_val, car_ptr);
            ctx_.builder().CreateStore(cdr_val, cdr_ptr);

            // Recursively match car and cdr patterns
            compilePattern(pat->cons_pat.car, car_ptr, fail_block);
            compilePattern(pat->cons_pat.cdr, cdr_ptr, fail_block);
            return ConstantInt::getTrue(ctx_.context());
        }

        case PATTERN_PREDICATE: {
            Value* scrutinee = ctx_.builder().CreateLoad(ctx_.taggedValueType(), scrutinee_ptr);
            Value* pred_func = codegenAST(pat->pred_pat.predicate);
            Value* result = ctx_.builder().CreateCall(pred_func, {scrutinee});
            Value* is_true = /* check if truthy */;
            BasicBlock* cont = BasicBlock::Create(ctx_.context(), "pred_ok", current_func);
            ctx_.builder().CreateCondBr(is_true, cont, fail_block);
            ctx_.builder().SetInsertPoint(cont);

            // Optionally bind variable
            if (pat->pred_pat.var_name) {
                Value* var_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
                ctx_.builder().CreateStore(scrutinee, var_ptr);
                ctx_.addLocal(pat->pred_pat.var_name, var_ptr);
            }
            return is_true;
        }

        // ... other pattern types
    }
}
```

### 5.5 Basic Macros

#### 5.5.1 Macro Representation (`eshkol.h`)

```c
typedef struct syntax_rule {
    pattern_t* pattern;              // Macro pattern
    eshkol_ast_t* template_expr;     // Template to instantiate
} syntax_rule_t;

typedef struct macro_def {
    char* name;
    char** literals;                  // Literal identifiers
    size_t num_literals;
    syntax_rule_t* rules;
    size_t num_rules;
} macro_def_t;

// Add to eshkol_op_t
ESHKOL_DEFINE_SYNTAX_OP,
ESHKOL_LET_SYNTAX_OP,
ESHKOL_LETREC_SYNTAX_OP,
```

#### 5.5.2 Macro Expansion Phase

```cpp
// New pass before codegen
class MacroExpander {
public:
    MacroExpander(std::map<std::string, macro_def_t>& macros)
        : macros_(macros) {}

    eshkol_ast_t expand(const eshkol_ast_t& ast) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_DEFINE_SYNTAX_OP) {
            // Register macro
            registerMacro(ast.operation.define_syntax_op);
            return makeVoid();
        }

        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_CALL_OP) {
            // Check if calling a macro
            std::string name = getFunctionName(ast);
            if (macros_.count(name)) {
                return expandMacroCall(macros_[name], ast);
            }
        }

        // Recursively expand subexpressions
        return expandChildren(ast);
    }

private:
    eshkol_ast_t expandMacroCall(const macro_def_t& macro, const eshkol_ast_t& call) {
        // Try each rule
        for (size_t i = 0; i < macro.num_rules; i++) {
            std::map<std::string, eshkol_ast_t*> bindings;
            if (matchPattern(macro.rules[i].pattern, call, bindings)) {
                eshkol_ast_t expanded = instantiateTemplate(
                    macro.rules[i].template_expr, bindings);
                // Recursively expand result
                return expand(expanded);
            }
        }
        eshkol_error("No matching macro rule for %s", macro.name);
    }

    bool matchPattern(const pattern_t* pat, const eshkol_ast_t& input,
                      std::map<std::string, eshkol_ast_t*>& bindings) {
        // Pattern matching logic
        // Handle ellipsis (...) for repetition
        // Handle literal matching
        // Collect bindings for pattern variables
    }

    eshkol_ast_t instantiateTemplate(const eshkol_ast_t* tmpl,
                                     const std::map<std::string, eshkol_ast_t*>& bindings) {
        // Replace pattern variables with bound values
        // Expand ellipsis patterns
    }

    std::map<std::string, macro_def_t>& macros_;
};
```

#### 5.5.3 Integration Point (`eshkol-run.cpp`)

```cpp
// Before codegen, run macro expansion
MacroExpander expander(global_macros);
std::vector<eshkol_ast_t> expanded_asts;
for (const auto& ast : asts) {
    expanded_asts.push_back(expander.expand(ast));
}

// Then generate code for expanded ASTs
LLVMModuleRef module = eshkol_generate_llvm_ir(
    expanded_asts.data(), expanded_asts.size(), module_name);
```

---

## 6. File-by-File Modification Guide

### 6.1 Header Files

| File | Changes |
|------|---------|
| `inc/eshkol/eshkol.h` | Add exception types, multi-value type, pattern types, macro types, new ops |
| `inc/eshkol/types/hott_types.h` | Add exception type to type hierarchy |
| `inc/eshkol/backend/codegen_context.h` | Add exception handler stack, macro environment |

### 6.2 Implementation Files

| File | Changes |
|------|---------|
| `lib/frontend/parser.cpp` | Add quasiquote tokens, guard/match/syntax parsing |
| `lib/backend/llvm_codegen.cpp` | Add codegen for all new forms |
| `lib/core/arena_memory.cpp` | Add exception runtime support |
| `lib/types/type_checker.cpp` | Add type rules for new forms |

### 6.3 New Files to Create

| File | Purpose |
|------|---------|
| `lib/frontend/macro_expander.cpp` | Macro expansion pass |
| `inc/eshkol/frontend/macro_expander.h` | Macro expander interface |
| `lib/core/exception.esk` | Standard exception types |
| `lib/core/match.esk` | Match helper functions |

### 6.4 Test Files to Create

| Directory | Tests |
|-----------|-------|
| `tests/exceptions/` | guard, raise, error handling |
| `tests/multivalue/` | values, call-with-values, let-values |
| `tests/quasiquote/` | Quasiquote expansion |
| `tests/pattern/` | Pattern matching |
| `tests/macros/` | Macro definition and expansion |

---

## 7. Testing Strategy

### 7.1 Unit Tests

Each new feature requires:
1. **Positive tests**: Feature works as expected
2. **Negative tests**: Proper errors for invalid input
3. **Edge cases**: Boundary conditions, empty inputs
4. **Integration**: Interacts correctly with existing features

### 7.2 Test Templates

#### Exception Tests (`tests/exceptions/basic_test.esk`)
```scheme
(require stdlib)

(define tests-passed 0)
(define tests-failed 0)

(define (check name expected actual)
  (if (equal? expected actual)
      (set! tests-passed (+ tests-passed 1))
      (begin
        (set! tests-failed (+ tests-failed 1))
        (display "FAIL: ") (display name) (newline))))

;; Test 1: Simple error
(check "error raises"
  'caught
  (guard (e (else 'caught))
    (error "test error")))

;; Test 2: Error with irritants
(check "error with irritants"
  '("test" 1 2 3)
  (guard (e (else (list (error-object-message e)
                        (error-object-irritants e))))
    (error "test" 1 2 3)))

;; Test 3: Guard with multiple clauses
(check "guard multiple clauses"
  'file-error
  (guard (e ((file-error? e) 'file-error)
            ((type-error? e) 'type-error)
            (else 'other))
    (open-input-file "/nonexistent/path")))

;; Test 4: No exception
(check "guard no exception"
  42
  (guard (e (else 'error))
    (* 6 7)))

;; Test 5: Re-raise
(check "re-raise"
  'outer
  (guard (e (else 'outer))
    (guard (e ((type-error? e) 'inner))
      (error "generic"))))

(display "Tests passed: ") (display tests-passed) (newline)
(display "Tests failed: ") (display tests-failed) (newline)
```

#### Pattern Matching Tests (`tests/pattern/basic_test.esk`)
```scheme
(require stdlib)

(define (check name expected actual)
  (if (equal? expected actual)
      (display "PASS: ")
      (display "FAIL: "))
  (display name)
  (newline))

;; Test 1: Literal matching
(check "literal match"
  'one
  (match 1
    (1 'one)
    (2 'two)
    (_ 'other)))

;; Test 2: Variable binding
(check "variable binding"
  10
  (match 5
    (x (* x 2))))

;; Test 3: List destructuring
(check "list destructuring"
  6
  (match '(1 2 3)
    ((list a b c) (+ a b c))))

;; Test 4: Cons pattern
(check "cons pattern"
  '(2 3)
  (match '(1 2 3)
    ((cons h t) t)))

;; Test 5: Predicate pattern
(check "predicate pattern"
  'even
  (match 4
    ((? odd?) 'odd)
    ((? even?) 'even)))

;; Test 6: Nested patterns
(check "nested patterns"
  3
  (match '((1 2) (3 4))
    ((list (list a b) (list c d)) c)))

;; Test 7: Wildcard
(check "wildcard"
  'matched
  (match '(1 2 3)
    ((list _ _ _) 'matched)))

;; Test 8: Guard
(check "pattern with guard"
  'big
  (match 100
    ((? number? n) (when (> n 50)) 'big)
    ((? number? n) 'small)))
```

#### Quasiquote Tests (`tests/quasiquote/basic_test.esk`)
```scheme
(require stdlib)

(define (check name expected actual)
  (if (equal? expected actual)
      (display "PASS: ")
      (display "FAIL: "))
  (display name)
  (newline))

;; Test 1: Simple quasiquote (no unquotes)
(check "simple quasiquote"
  '(a b c)
  `(a b c))

;; Test 2: Unquote
(check "unquote"
  '(1 2 3)
  `(1 ,(+ 1 1) 3))

;; Test 3: Unquote-splicing
(check "unquote-splicing"
  '(1 2 3 4 5)
  `(1 ,@(list 2 3 4) 5))

;; Test 4: Nested quasiquote
(check "nested quasiquote"
  '`(a ,(+ 1 2))
  ``(a ,(+ 1 2)))

;; Test 5: Mixed unquote and splice
(check "mixed"
  '(a 1 2 3 b 4 5 6 c)
  `(a ,@(list 1 2 3) b ,@(list 4 5 6) c))

;; Test 6: Quasiquote in function
(define (make-adder n)
  `(lambda (x) (+ x ,n)))

(check "quasiquote in function"
  '(lambda (x) (+ x 5))
  (make-adder 5))
```

### 7.3 Regression Testing

After each phase, run full test suite:
```bash
./scripts/run_tests_with_output.sh
./scripts/run_stdlib_tests.sh
./scripts/run_features_tests.sh
./scripts/run_autodiff_tests.sh
```

---

## 8. Quick Wins

These can be implemented in < 1 day each:

### 8.1 `dynamic-wind` (4 hours)
```scheme
(dynamic-wind before thunk after)
;; Call before, then thunk, then after
;; Useful for cleanup with exceptions
```

### 8.2 `string-map` and `string-for-each` (2 hours)
```scheme
(define (string-map proc str)
  (list->string (map proc (string->list str))))

(define (string-for-each proc str)
  (for-each proc (string->list str)))
```

### 8.3 `vector-map` and `vector-for-each` (2 hours)
```scheme
(define (vector-map proc vec)
  (list->vector (map proc (vector->list vec))))

(define (vector-for-each proc vec)
  (for-each proc (vector->list vec)))
```

### 8.4 `make-parameter` and `parameterize` (4 hours)
```scheme
(define current-output-port (make-parameter stdout))

(parameterize ((current-output-port file-port))
  (display "goes to file"))
```

### 8.5 `call-with-port` (1 hour)
```scheme
(define (call-with-port port proc)
  (dynamic-wind
    (lambda () #f)
    (lambda () (proc port))
    (lambda () (close-port port))))
```

### 8.6 `call-with-input-file` / `call-with-output-file` (1 hour)
```scheme
(define (call-with-input-file filename proc)
  (call-with-port (open-input-file filename) proc))

(define (call-with-output-file filename proc)
  (call-with-port (open-output-file filename) proc))
```

---

## 9. Post-v1.0 Roadmap

### 9.1 Version 1.1 (Performance)

| Feature | Effort | Impact |
|---------|--------|--------|
| SIMD vectorization | 3 weeks | 2-10x tensor ops |
| Parallel primitives | 4 weeks | Multi-core scaling |
| JIT optimization | 2 weeks | Faster REPL |
| Tail call optimization audit | 1 week | Deep recursion |

### 9.2 Version 1.2 (Tooling)

| Feature | Effort | Impact |
|---------|--------|--------|
| Package manager | 3 weeks | Distribution |
| LSP server | 4 weeks | IDE support |
| Profiler | 2 weeks | Performance debugging |
| Debugger | 3 weeks | Development |

### 9.3 Version 2.0 (Scale)

| Feature | Effort | Impact |
|---------|--------|--------|
| GPU backend (CUDA) | 2 months | 100x for large tensors |
| Distributed computing | 2 months | Cluster support |
| `call/cc` | 4 weeks | Scheme compatibility |
| Full R7RS | 2 months | Standard compliance |

---

## 10. Appendices

### 10.1 Current File Structure

```
eshkol/
├── exe/
│   ├── eshkol-run.cpp          # Main driver (2000 lines)
│   └── eshkol-repl.cpp         # REPL with JIT
├── inc/eshkol/
│   ├── eshkol.h                # Main types (850 lines)
│   ├── backend/
│   │   ├── codegen_context.h   # Codegen state
│   │   ├── llvm_backend.h      # Public API
│   │   ├── arithmetic_codegen.h
│   │   ├── autodiff_codegen.h
│   │   ├── tensor_codegen.h
│   │   ├── function_codegen.h
│   │   ├── binding_codegen.h
│   │   ├── control_flow_codegen.h
│   │   ├── collection_codegen.h
│   │   ├── tagged_value_codegen.h
│   │   ├── memory_codegen.h
│   │   ├── string_io_codegen.h
│   │   ├── system_codegen.h
│   │   ├── hash_codegen.h
│   │   └── ... (20+ modules)
│   └── types/
│       ├── hott_types.h        # Type system
│       ├── type_checker.h      # Type checking
│       └── dependent.h         # Dependent types
├── lib/
│   ├── frontend/
│   │   └── parser.cpp          # Lexer + Parser (4461 lines)
│   ├── backend/
│   │   ├── llvm_codegen.cpp    # Main codegen (25219 lines)
│   │   ├── tensor_codegen.cpp
│   │   ├── autodiff_codegen.cpp
│   │   └── ... (implementations)
│   ├── types/
│   │   ├── hott_types.cpp
│   │   ├── type_checker.cpp
│   │   └── dependent.cpp
│   ├── core/
│   │   ├── arena_memory.cpp    # Memory management
│   │   └── *.esk               # Stdlib modules
│   ├── stdlib.esk              # Stdlib re-exports
│   └── math.esk                # Math library
├── tests/
│   ├── autodiff/               # 51 tests
│   ├── lists/                  # 131 tests
│   ├── types/                  # 13 tests
│   ├── features/               # 9 tests
│   └── ...                     # 253 total
└── docs/
    ├── MASTER_DEVELOPMENT_PLAN.md
    └── IMPLEMENTATION_PLAN_V1.md  # This document
```

### 10.2 Build Commands

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Test
./eshkol-run ../tests/features/extreme_stress_test.esk
./scripts/run_tests_with_output.sh

# REPL
./eshkol-repl
```

### 10.3 Key Data Structures

```c
// Tagged value (16 bytes) - universal value representation
typedef struct eshkol_tagged_value {
    uint8_t type;        // eshkol_value_type_t
    uint8_t flags;       // Exactness, etc.
    uint16_t reserved;
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
    } data;
} eshkol_tagged_value_t;

// Closure (function + captured environment)
typedef struct eshkol_closure {
    uint64_t func_ptr;
    eshkol_closure_env_t* env;
    uint64_t sexpr_ptr;          // For homoiconicity
    uint8_t return_type;
    uint8_t input_arity;
    uint32_t hott_type_id;
} eshkol_closure_t;

// Dual number (forward-mode AD)
typedef struct eshkol_dual_number {
    double value;
    double derivative;
} eshkol_dual_number_t;

// AD node (reverse-mode AD)
typedef struct ad_node {
    ad_node_type_t type;
    double value;
    double gradient;
    struct ad_node* input1;
    struct ad_node* input2;
    size_t id;
} ad_node_t;
```

### 10.4 References

- R7RS-small: https://small.r7rs.org/
- SRFI-1 (List Library): https://srfi.schemers.org/srfi-1/
- SRFI-9 (Records): https://srfi.schemers.org/srfi-9/
- LLVM LangRef: https://llvm.org/docs/LangRef.html
- HoTT Book: https://homotopytypetheory.org/book/

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | December 2024 | Initial comprehensive plan |

---

**End of Implementation Plan**
