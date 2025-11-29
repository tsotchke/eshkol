# Eshkol Built-in Functions Development Guide

## Executive Summary

This document provides a comprehensive analysis of Eshkol's built-in functions, comparing the current implementation against the complete operations specification. It serves as a development roadmap for achieving language completeness.

**Legend:**
- ✅ = Implemented and working
- ⚠️ = Partially implemented or has known issues
- ❌ = Not implemented

**Summary Statistics (Updated):**
- **Implemented:** ~123 operations
- **Not Implemented:** ~37 operations
- **Priority for eval:** quote, apply, set!

---

## 1. Core Special Forms

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `define` | `(define name value)` | Variable definition | ✅ |
| `define` (function) | `(define (f x) body)` | Function definition | ✅ |
| `define` (nested) | Inside function bodies | Nested definitions | ✅ |
| `lambda` | `(lambda (x) body)` | Anonymous functions | ✅ |
| `let` | `(let ((x 1)) body)` | Parallel binding | ✅ |
| `let*` | `(let* ((x 1) (y x)) body)` | Sequential binding | ✅ |
| `letrec` | `(letrec ((f ...)) body)` | Recursive binding | ❌ |
| `if` | `(if cond then else)` | Conditional | ✅ |
| `cond` | `(cond (test expr) ...)` | Multi-branch conditional | ❌ |
| `case` | `(case key ((val) expr) ...)` | Pattern matching | ❌ |
| `begin` / `sequence` | `(begin expr1 expr2 ...)` | Sequential evaluation | ✅ |
| `quote` | `'expr` / `(quote expr)` | Literal data | ❌ |
| `quasiquote` | `` `expr `` | Template with unquoting | ❌ |
| `unquote` | `,expr` | Evaluate in quasiquote | ❌ |
| `unquote-splicing` | `,@expr` | Splice in quasiquote | ❌ |
| `set!` | `(set! var value)` | Variable mutation | ❌ |

---

## 2. Arithmetic Operations

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `+` | `(+ a b ...)` | Addition (variadic) | ✅ |
| `-` | `(- a b ...)` | Subtraction (variadic) | ✅ |
| `*` | `(* a b ...)` | Multiplication (variadic) | ✅ |
| `/` | `(/ a b ...)` | Division (variadic) | ✅ |
| `%` / `mod` / `modulo` | `(% a b)` | Modulo (floor division) | ⚠️ (single call works) |
| `remainder` | `(remainder a b)` | Remainder (truncate div) | ⚠️ (single call works) |
| `quotient` | `(quotient a b)` | Integer division | ✅ |
| `abs` | `(abs x)` | Absolute value | ✅ |
| `floor` | `(floor x)` | Round down | ✅ |
| `ceiling` | `(ceiling x)` | Round up | ✅ |
| `round` | `(round x)` | Round to nearest | ✅ |
| `truncate` | `(truncate x)` | Truncate toward zero | ✅ |
| `min` | `(min a b ...)` | Minimum (variadic) | ✅ |
| `max` | `(max a b ...)` | Maximum (variadic) | ✅ |
| `gcd` | `(gcd a b)` | Greatest common divisor | ✅ |
| `lcm` | `(lcm a b)` | Least common multiple | ✅ |
| `expt` | `(expt base exp)` | Exponentiation | ✅ |

**Known Issues:**
- `modulo` and `remainder` can crash when called multiple times in sequence due to raw int64 return values being misinterpreted as cons pointers. Single calls work correctly.

---

## 3. Comparison & Logic

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `=` | `(= a b)` | Numeric equality | ✅ |
| `>` | `(> a b)` | Greater than | ✅ |
| `<` | `(< a b)` | Less than | ✅ |
| `>=` | `(>= a b)` | Greater or equal | ✅ |
| `<=` | `(<= a b)` | Less or equal | ✅ |
| `and` | `(and a b ...)` | Logical AND (short-circuit) | ✅ |
| `or` | `(or a b ...)` | Logical OR (short-circuit) | ✅ |
| `not` | `(not x)` | Logical NOT | ✅ |
| `eq?` | `(eq? a b)` | Identity (pointer) equality | ✅ |
| `eqv?` | `(eqv? a b)` | Value equality (primitives) | ✅ |
| `equal?` | `(equal? a b)` | Deep structural equality | ✅ |

---

## 4. List Operations (Core)

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `cons` | `(cons a b)` | Construct pair | ✅ |
| `car` | `(car pair)` | First element | ✅ |
| `cdr` | `(cdr pair)` | Rest of list | ✅ |
| `set-car!` | `(set-car! pair val)` | Mutate car | ❌ |
| `set-cdr!` | `(set-cdr! pair val)` | Mutate cdr | ❌ |
| `list` | `(list a b ...)` | Create list | ✅ |
| `list?` | `(list? x)` | Test if proper list | ✅ |
| `null?` | `(null? x)` | Test if empty list | ✅ |
| `pair?` | `(pair? x)` | Test if pair | ✅ |
| `length` | `(length lst)` | List length | ✅ |
| `append` | `(append lst1 lst2 ...)` | Concatenate (variadic) | ✅ |
| `reverse` | `(reverse lst)` | Reverse list | ✅ |

---

## 5. List Operations (Advanced)

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `list-ref` | `(list-ref lst n)` | Get nth element | ✅ |
| `list-tail` | `(list-tail lst n)` | Drop n elements | ✅ |
| `last` | `(last lst)` | Last element | ✅ |
| `last-pair` | `(last-pair lst)` | Last cons cell | ✅ |
| `make-list` | `(make-list n [fill])` | Create n-element list | ❌ |
| `iota` | `(iota count [start step])` | Generate sequence | ❌ |

---

## 6. List Accessors (c[ad]+r Family)

All compound car/cdr operations are **✅ Implemented**:

### 2-Level Accessors
| `caar` | `cadr` | `cdar` | `cddr` |
|--------|--------|--------|--------|
| ✅ | ✅ | ✅ | ✅ |

### 3-Level Accessors
| `caaar` | `caadr` | `cadar` | `caddr` | `cdaar` | `cdadr` | `cddar` | `cdddr` |
|---------|---------|---------|---------|---------|---------|---------|---------|
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### 4-Level Accessors
All 16 four-level accessors (`caaaar` through `cddddr`) are **✅ Implemented**.

---

## 7. Higher-Order Functions

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `map` | `(map f lst ...)` | Map over lists | ✅ |
| `filter` | `(filter pred lst)` | Filter by predicate | ✅ |
| `fold` / `foldl` | `(fold f init lst)` | Left fold | ✅ |
| `fold-right` / `foldr` | `(fold-right f init lst)` | Right fold | ✅ |
| `reduce` | `(reduce f lst)` | Reduce (no init) | ✅ |
| `for-each` | `(for-each f lst)` | Iteration (side effects) | ✅ |
| `apply` | `(apply f args)` | Apply to arg list | ❌ |
| `compose` | `(compose f g)` | Function composition | ✅ |
| `curry` | `(curry f)` | Currying | ❌ |

---

## 8. List Search & Membership

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `member` | `(member x lst)` | Find using `equal?` | ❌ |
| `memq` | `(memq x lst)` | Find using `eq?` | ❌ |
| `memv` | `(memv x lst)` | Find using `eqv?` | ❌ |
| `assoc` | `(assoc key alist)` | Alist lookup (`equal?`) | ❌ |
| `assq` | `(assq key alist)` | Alist lookup (`eq?`) | ❌ |
| `assv` | `(assv key alist)` | Alist lookup (`eqv?`) | ❌ |
| `find` | `(find pred lst)` | First matching element | ✅ |
| `any` / `some` | `(any pred lst)` | Any element matches? | ❌ |
| `every` / `all` | `(every pred lst)` | All elements match? | ❌ |

---

## 9. List Manipulation

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `take` | `(take n lst)` | First n elements | ✅ |
| `drop` | `(drop n lst)` | Remove first n | ✅ |
| `split-at` | `(split-at n lst)` | Split at index | ✅ |
| `partition` | `(partition pred lst)` | Split by predicate | ✅ |
| `remove` | `(remove pred lst)` | Remove matching | ✅ |
| `remq` | `(remq x lst)` | Remove by identity | ❌ |
| `delete` | `(delete x lst)` | Remove specific element | ❌ |
| `sort` | `(sort lst less?)` | Sort list | ❌ |
| `zip` | `(zip lst1 lst2 ...)` | Zip lists together | ❌ |
| `unzip` | `(unzip lst)` | Unzip list | ❌ |

---

## 10. Mathematical Functions

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `sin` | `(sin x)` | Sine | ✅ |
| `cos` | `(cos x)` | Cosine | ✅ |
| `tan` | `(tan x)` | Tangent | ✅ |
| `asin` | `(asin x)` | Arc sine | ✅ |
| `acos` | `(acos x)` | Arc cosine | ✅ |
| `atan` | `(atan x)` | Arc tangent | ✅ |
| `atan2` | `(atan2 y x)` | Two-arg arctangent | ✅ |
| `sinh` | `(sinh x)` | Hyperbolic sine | ✅ |
| `cosh` | `(cosh x)` | Hyperbolic cosine | ✅ |
| `tanh` | `(tanh x)` | Hyperbolic tangent | ✅ |
| `asinh` | `(asinh x)` | Inverse hyperbolic sine | ✅ |
| `acosh` | `(acosh x)` | Inverse hyperbolic cosine | ✅ |
| `atanh` | `(atanh x)` | Inverse hyperbolic tangent | ✅ |
| `sqrt` | `(sqrt x)` | Square root | ✅ |
| `cbrt` | `(cbrt x)` | Cube root | ✅ |
| `pow` | `(pow base exp)` | Power | ✅ |
| `exp` | `(exp x)` | e^x | ✅ |
| `log` | `(log x)` | Natural logarithm | ✅ |
| `log10` | `(log10 x)` | Base-10 logarithm | ✅ |
| `log2` | `(log2 x)` | Base-2 logarithm | ✅ |

---

## 11. Vector/Tensor Operations

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `vector` | `(vector a b ...)` | Create 1D tensor | ✅ |
| `vector?` | `(vector? x)` | Test if vector | ❌ |
| `make-vector` | `(make-vector n [fill])` | Create n-element vector | ❌ |
| `vref` / `vector-ref` | `(vref v i)` | Get element | ✅ |
| `vset!` / `vector-set!` | `(vset! v i val)` | Set element | ⚠️ (`tensor-set`) |
| `vector-length` | `(vector-length v)` | Vector length | ❌ |
| `vector->list` | `(vector->list v)` | Convert to list | ❌ |
| `list->vector` | `(list->vector lst)` | Convert from list | ❌ |
| `tensor` | `(tensor dims... elems...)` | N-D tensor | ✅ |
| `matrix` | `(matrix r c elems...)` | 2D tensor | ✅ |
| `tensor-ref` | `(tensor-ref t indices...)` | Tensor element | ⚠️ (`tensor-get`) |
| `tensor-shape` | `(tensor-shape t)` | Get dimensions | ❌ |
| `reshape` | `(reshape t dims...)` | Reshape tensor | ❌ |
| `transpose` | `(transpose m)` | Matrix transpose | ❌ |
| `dot` | `(dot a b)` | Dot product | ✅ |
| `matmul` | `(matmul a b)` | Matrix multiply | ❌ |
| `tensor-add` | `(tensor-add a b)` | Element-wise add | ✅ |
| `tensor-sub` | `(tensor-sub a b)` | Element-wise sub | ✅ |
| `tensor-mul` | `(tensor-mul a b)` | Element-wise mul | ✅ |
| `tensor-div` | `(tensor-div a b)` | Element-wise div | ✅ |

---

## 12. Automatic Differentiation

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `derivative` | `(derivative f x)` | Scalar derivative | ✅ |
| `gradient` | `(gradient f v)` | Vector gradient | ✅ |
| `jacobian` | `(jacobian f v)` | Jacobian matrix | ✅ |
| `hessian` | `(hessian f v)` | Hessian matrix | ✅ |
| `divergence` | `(divergence F v)` | Vector divergence | ✅ |
| `curl` | `(curl F v)` | Vector curl (3D) | ✅ |
| `laplacian` | `(laplacian f v)` | Laplacian operator | ✅ |
| `directional-derivative` | `(directional-derivative f v dir)` | Directional deriv | ✅ |
| `diff` | `(diff expr var)` | Symbolic differentiation | ✅ |

---

## 13. String Operations

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `string?` | `(string? x)` | Test if string | ❌ |
| `string-length` | `(string-length s)` | String length | ❌ |
| `string-ref` | `(string-ref s k)` | Get character | ❌ |
| `string-append` | `(string-append s1 s2 ...)` | Concatenate | ❌ |
| `substring` | `(substring s start end)` | Extract substring | ❌ |
| `string->list` | `(string->list s)` | To char list | ❌ |
| `list->string` | `(list->string lst)` | From char list | ❌ |
| `string=?` | `(string=? s1 s2)` | String equality | ❌ |
| `string<?` | `(string<? s1 s2)` | Lexicographic < | ❌ |
| `string>?` | `(string>? s1 s2)` | Lexicographic > | ❌ |
| `number->string` | `(number->string n)` | Convert to string | ❌ |
| `string->number` | `(string->number s)` | Parse number | ❌ |
| `symbol->string` | `(symbol->string sym)` | Symbol to string | ❌ |
| `string->symbol` | `(string->symbol s)` | String to symbol | ❌ |

---

## 14. Type Predicates

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `number?` | `(number? x)` | Test if number | ✅ |
| `integer?` | `(integer? x)` | Test if integer | ✅ |
| `real?` | `(real? x)` | Test if real | ✅ |
| `exact?` | `(exact? x)` | Test if exact | ❌ |
| `inexact?` | `(inexact? x)` | Test if inexact | ❌ |
| `positive?` | `(positive? x)` | x > 0 | ✅ |
| `negative?` | `(negative? x)` | x < 0 | ✅ |
| `zero?` | `(zero? x)` | x = 0 | ✅ |
| `odd?` | `(odd? x)` | Is odd | ✅ |
| `even?` | `(even? x)` | Is even | ✅ |
| `procedure?` | `(procedure? x)` | Test if function | ❌ |
| `boolean?` | `(boolean? x)` | Test if boolean | ❌ |
| `symbol?` | `(symbol? x)` | Test if symbol | ❌ |

---

## 15. I/O Operations

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `display` | `(display x)` | Print value | ✅ |
| `newline` | `(newline)` | Print newline | ✅ |
| `print` | `(print x)` | Display with newline | ❌ |
| `write` | `(write x)` | Write S-expression | ❌ |
| `read` | `(read)` | Read input | ❌ |
| `read-line` | `(read-line)` | Read line | ❌ |

---

## 16. Control Flow

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `when` | `(when test expr ...)` | One-armed if | ❌ |
| `unless` | `(unless test expr ...)` | Negated when | ❌ |
| `do` | `(do ((var init step) ...) (test result) body)` | Iteration | ❌ |
| `while` | `(while test body ...)` | While loop | ❌ |
| `break` | `(break)` | Loop break | ❌ |
| `continue` | `(continue)` | Loop continue | ❌ |

---

## 17. Function Introspection

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `arity` | `(arity f)` | Get function arity | ❌ |
| `procedure-arity` | `(procedure-arity f)` | Parameter count | ❌ |

---

## 18. Error Handling

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `error` | `(error msg args...)` | Raise error | ❌ |
| `raise` | `(raise condition)` | Raise condition | ❌ |
| `guard` / `try-catch` | `(guard (var clause...) body)` | Exception handling | ❌ |
| `with-exception-handler` | `(with-exception-handler h thunk)` | Handle exceptions | ❌ |

---

## 19. FFI (Foreign Function Interface)

| Operation | Syntax | Description | Status |
|-----------|--------|-------------|--------|
| `extern` | `(extern ret-type name types...)` | C function decl | ✅ |
| `extern-var` | `(extern-var type name)` | C variable decl | ✅ |

---

## Remaining Work - Priority List

### Priority 1: Critical for `eval`
| Operation | Effort | Notes |
|-----------|--------|-------|
| `quote` | Medium | Parser token exists |
| `apply` | Medium | Variadic calls |
| `set!` | Medium | GlobalVariable mutation |
| ~~`eq?`~~ | ~~Low~~ | ✅ Implemented |
| ~~`eqv?`~~ | ~~Low~~ | ✅ Implemented |
| ~~`equal?`~~ | ~~Medium~~ | ✅ Implemented |

### Priority 2: Core Scheme Compatibility
| Operation | Effort | Notes |
|-----------|--------|-------|
| `letrec` | Medium | Forward declaration |
| `cond` | Low | Transform to nested `if` |
| `case` | Medium | Pattern matching |
| `when`/`unless` | Low | Simple macros |

### Priority 3: List Operations
| Operation | Effort | Notes |
|-----------|--------|-------|
| `member`/`memq`/`memv` | Low | List search |
| `assoc`/`assq`/`assv` | Low | Alist operations |
| `any`/`every` | Low | Predicate checks |
| `sort` | Medium | Merge sort |
| `zip`/`unzip` | Low | List manipulation |

### Priority 4: Advanced Features
| Operation | Effort | Notes |
|-----------|--------|-------|
| String operations | Medium | Need runtime string type |
| `matmul`/`transpose` | Medium | Tensor operations |
| Error handling | High | Exception infrastructure |
| `call/cc` | High | Continuation capture |

---

## Appendix A: Full Implementation Status Summary

### Fully Implemented (✅)
```
+, -, *, /, quotient
=, <, >, <=, >=
and, or, not
eq?, eqv?, equal?
abs, floor, ceiling, round, truncate, min, max, gcd, lcm, expt
sin, cos, tan, asin, acos, atan, atan2
sinh, cosh, tanh, asinh, acosh, atanh
sqrt, cbrt, pow, exp, log, log10, log2
cons, car, cdr, list, null?, pair?, list?, length, append, reverse
list-ref, list-tail, last, last-pair
take, drop, split-at, partition, remove, find
All c[ad]+r accessors (28 total)
map, filter, fold, fold-right, for-each, reduce, compose
vector, matrix, tensor, vref, dot
tensor-add, tensor-sub, tensor-mul, tensor-div
derivative, gradient, jacobian, hessian
divergence, curl, laplacian, directional-derivative, diff
display, newline
define, lambda, if, let, let*, begin
extern, extern-var
number?, integer?, real?, positive?, negative?, zero?, odd?, even?
```

### Partially Working (⚠️)
```
modulo, remainder       (Single call works, multiple calls may crash)
tensor-set, tensor-get  (Different naming)
```

### Not Implemented (❌)
```
quote, apply, set!               (Critical for eval)
letrec, cond, case               (Control flow)
set-car!, set-cdr!               (Mutation)
member, memq, memv               (List search)
assoc, assq, assv                (Alist operations)
any, every                       (Predicate checks)
make-list, iota                  (List generation)
remq, delete, sort, zip, unzip   (List manipulation)
String operations                (All)
vector?, make-vector, vector-length, vector->list, list->vector
tensor-shape, reshape, transpose, matmul
when, unless, do, while, break, continue
exact?, inexact?, procedure?, boolean?, symbol?
print, write, read, read-line
error, raise, guard
arity, procedure-arity
curry, call/cc
```

---

*Document updated: Current session*
*Source files: llvm_codegen.cpp, parser.cpp, eshkol.h*
