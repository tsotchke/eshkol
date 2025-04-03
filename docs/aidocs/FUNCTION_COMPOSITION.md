# Function Composition in Eshkol

## Overview of Function Composition

Function composition is a core feature of Eshkol, allowing developers to build complex operations by combining simpler functions. This functional programming approach enables cleaner, more modular code with improved reusability.

```mermaid
graph TD
    A[Function f] --> C[Composition f ∘ g]
    B[Function g] --> C
    C --> D[Result: f(g(x))]
    
    style A fill:#f9d5e5,stroke:#333,stroke-width:2px
    style B fill:#f9d5e5,stroke:#333,stroke-width:2px
    style C fill:#d0f0c0,stroke:#333,stroke-width:2px
    style D fill:#d0f0c0,stroke:#333,stroke-width:2px
```

## Closure Implementation

Closures in Eshkol are implemented using a combination of function pointers and environment structures. This allows functions to capture and retain access to variables from their defining scope.

### Core Components

1. **EshkolClosure Structure**: Contains a function pointer and an environment
2. **EshkolEnvironment**: Stores captured variables and their values
3. **Environment Chain**: Linked environments for nested scopes

```c
typedef struct {
    void* (*function)(void**, EshkolEnvironment*);  // Function pointer
    EshkolEnvironment* environment;                // Captured environment
} EshkolClosure;

typedef struct EshkolEnvironment {
    struct EshkolEnvironment* parent;  // Parent environment
    void** values;                     // Captured values
    size_t value_count;                // Number of values
    uint64_t ref_count;                // Reference count for memory management
    // Additional fields omitted for brevity
} EshkolEnvironment;
```

## Function Composition Examples

### Basic Composition

```scheme
;; Define simple functions
(define square (lambda (x) (* x x)))
(define add1 (lambda (x) (+ x 1)))
(define double-value (lambda (x) (* x 2)))

;; Compose functions
(define (compose f g)
  (lambda (x) (f (g x))))

;; Create composed functions
(define square-then-add1 (compose add1 square))
(define add1-then-square (compose square add1))

;; Usage
(square-then-add1 5)  ;; add1(square(5)) = add1(25) = 26
(add1-then-square 5)  ;; square(add1(5)) = square(6) = 36
```

### Multi-Function Composition

```scheme
;; Compose multiple functions
(define (compose-all . funcs)
  (lambda (x)
    (foldr (lambda (f result) (f result)) 
           x 
           funcs)))

;; Create a pipeline of functions
(define pipeline (compose-all double-value add1 square))

;; Usage
(pipeline 5)  ;; double-value(add1(square(5))) = double-value(add1(25)) = double-value(26) = 52
```

## Relationship Between Closures and Environments

When a closure is created, it captures its lexical environment, including any variables referenced in the function body but defined in an outer scope.

```scheme
(define (make-counter start)
  (let ((count start))
    (lambda ()
      (set! count (+ count 1))
      count)))

(define counter1 (make-counter 0))
(define counter2 (make-counter 10))

(counter1)  ;; 1
(counter1)  ;; 2
(counter2)  ;; 11
```

In this example:
1. `make-counter` returns a closure that captures the `count` variable
2. Each call to `make-counter` creates a new environment with its own `count`
3. `counter1` and `counter2` are separate closures with different environments

## Optimization Techniques

Eshkol employs several optimization techniques for function composition:

### 1. Inlining

Small functions are often inlined to eliminate function call overhead:

```scheme
;; Original code
(define (add1 x) (+ x 1))
(define (square x) (* x x))
(define result (square (add1 5)))

;; After inlining optimization
;; Effectively becomes:
(define result (* (+ 5 1) (+ 5 1)))
(define result 36)
```

### 2. Tail Call Optimization

Eshkol implements proper tail call optimization to prevent stack overflow in recursive compositions:

```scheme
;; Recursive function with tail call
(define (map f lst)
  (if (null? lst)
      '()
      (cons (f (car lst))
            (map f (cdr lst)))))

;; This compiles to a loop in C, avoiding stack growth
```

### 3. Environment Pruning

The compiler analyzes which variables are actually used by closures and only captures those, reducing memory overhead:

```scheme
;; Before optimization
(let ((a 1) (b 2) (c 3))
  (lambda (x) (+ x a)))  ;; Captures a, b, c

;; After environment pruning
(let ((a 1) (b 2) (c 3))
  (lambda (x) (+ x a)))  ;; Only captures a
```

### 4. Closure Specialization

When a closure is called with constant arguments, the compiler can create specialized versions:

```scheme
;; Original higher-order function
(define (multiplier n)
  (lambda (x) (* x n)))

;; When used with a constant
(define double (multiplier 2))

;; Can be specialized to:
(define double (lambda (x) (* x 2)))
```

## Implementation Details

The function composition system is implemented in the following files:

- `src/core/utils/closure.c` - Core closure implementation
- `src/core/utils/closure_environment.c` - Environment management
- `src/core/utils/closure_management.c` - Closure lifecycle functions

## Advanced Usage: Higher-Order Functions

Eshkol supports a rich set of higher-order functions for working with collections:

```scheme
;; Map: Apply a function to each element
(define (map f lst)
  (if (null? lst)
      '()
      (cons (f (car lst))
            (map f (cdr lst)))))

;; Filter: Keep elements that satisfy a predicate
(define (filter pred lst)
  (cond ((null? lst) '())
        ((pred (car lst))
         (cons (car lst) (filter pred (cdr lst))))
        (else (filter pred (cdr lst)))))

;; Reduce: Combine elements using a function
(define (reduce f init lst)
  (if (null? lst)
      init
      (reduce f (f init (car lst)) (cdr lst))))

;; Usage example
(define numbers '(1 2 3 4 5))
(define squares (map square numbers))
(define even-squares (filter even? squares))
(define sum (reduce + 0 even-squares))
```

## Best Practices

1. **Favor composition over complex functions** - Build complex operations from simple, reusable functions
2. **Use higher-order functions** - Leverage map, filter, reduce for cleaner code
3. **Be mindful of closure capture** - Only capture what you need to minimize memory usage
4. **Consider performance implications** - Use composition judiciously in performance-critical code
5. **Leverage type annotations** - Add types to composed functions for better error checking and optimization
