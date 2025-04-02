# Function Composition in Eshkol

This document provides a comprehensive guide to function composition in Eshkol, including both binary and n-ary composition approaches, as well as array-based composition.

## Introduction

Function composition is a fundamental concept in functional programming that allows combining multiple functions to create a new function. In mathematical notation, if we have functions f and g, their composition (f ∘ g)(x) is equivalent to f(g(x)).

Eshkol supports several approaches to function composition:

1. Binary composition: Combining two functions
2. N-ary composition: Combining any number of functions
3. Array-based composition: Using arrays/vectors to store and apply functions

## Binary Function Composition

The simplest form of function composition combines two functions:

```scheme
(define (compose f g)
  (lambda (x) (f (g x))))
```

This creates a new function that applies g first, then applies f to the result.

Example usage:

```scheme
(define square-then-add1 (compose add1 square))
(square-then-add1 5)  ; => 26 (add1(square(5)) = add1(25) = 26)
```

## N-ary Function Composition

To compose any number of functions, we can use a variadic function with rest parameters:

```scheme
(define (apply-functions fs x)
  (if (null? fs)
      x
      (apply-functions (cdr fs) ((car fs) x))))

(define (compose-n . fns)
  (lambda (x)
    (apply-functions (reverse fns) x)))
```

The `compose-n` function takes any number of functions and returns a new function that applies them in right-to-left order (mathematical convention).

Example usage:

```scheme
(define f1 (compose-n square add1 double))
(f1 5)  ; => 121 (square(add1(double(5))) = square(add1(10)) = square(11) = 121)
```

## Array-based Function Composition

For more dynamic composition, we can use arrays (vectors) to store functions:

```scheme
(define (eval fs size x)
  (let loop ((i 0) (result x))
    (if (= i size)
        result
        (loop (+ i 1) ((vector-ref fs i) result)))))
```

This approach allows for runtime modification of the function sequence:

```scheme
(define fs (vector square add1 double))
(eval fs 3 5)  ; => 52 (double(add1(square(5))) = double(add1(25)) = double(26) = 52)

;; Reorder functions
(vector-set! fs 0 double)
(vector-set! fs 2 square)
(eval fs 3 5)  ; => 121 (square(add1(double(5))) = square(add1(10)) = square(11) = 121)
```

We can also create a function from an array:

```scheme
(define (make-composed-function fs size)
  (lambda (x)
    (eval fs size x)))

(define composed-function (make-composed-function fs 3))
(composed-function 5)  ; => 121
```

## Implementation in C

The array-based approach is similar to the C implementation provided in the original example:

```c
typedef int (*FXN)(int);

int eval(FXN fs[], int size, int x)
{
   for (int i = 0; i < size; i++) {
       x = fs[i](x);
   }
   return x;
}
```

## Closures and Environment Capture

When implementing function composition in Eshkol, it's important to properly handle closures and environment capture. The lambda function created by `compose` needs to capture both `f` and `g` in its environment.

In the C code generation, this requires:

1. Creating an environment that includes both functions
2. Properly passing the captured functions to the lambda
3. Ensuring the types are handled correctly (functions should be treated as closures or function pointers, not as float values)

## Recommendations for Eshkol Implementation

1. Fix the environment capture in the code generation for closures
2. Ensure proper type handling for function arguments
3. Enhance the special handling for nested compositions
4. Implement the array-based approach for more dynamic composition

## Example Files

The following example files demonstrate different approaches to function composition:

1. `function_composition_n.esk`: Demonstrates n-ary function composition using lists
2. `function_composition_array.esk`: Demonstrates array-based function composition
3. `function_composition_complete.esk`: Comprehensive example with all approaches
4. `function_composition_array.c`: C implementation of array-based function composition

## Conclusion

Function composition is a powerful technique in functional programming that enables building complex functions from simpler ones. Eshkol's support for closures and higher-order functions makes it well-suited for implementing various function composition approaches.

By providing both binary and n-ary composition, as well as array-based composition, Eshkol offers flexibility in how functions can be combined and manipulated at runtime.
