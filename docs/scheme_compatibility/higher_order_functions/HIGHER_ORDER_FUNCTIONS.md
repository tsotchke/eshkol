# Higher-Order Functions in Eshkol

This document provides a comprehensive guide to higher-order functions in Eshkol, including map, filter, reduce, and their relationship to function composition.

## Introduction

Higher-order functions are functions that take other functions as arguments or return functions as results. They are a fundamental concept in functional programming and enable powerful abstractions and code reuse.

The three most common higher-order functions are:

1. **Map**: Apply a function to each element of a collection
2. **Filter**: Select elements from a collection that satisfy a predicate
3. **Reduce**: Combine elements of a collection using a binary function

## Map Implementation

The `map` function applies a given function to each element of a list, returning a new list with the results:

```scheme
(define (map f lst)
  (if (null? lst)
      '()
      (cons (f (car lst)) (map f (cdr lst)))))
```

Example usage:

```scheme
(map square '(1 2 3 4 5))  ; => (1 4 9 16 25)
(map add1 '(1 2 3 4 5))    ; => (2 3 4 5 6)
```

## Filter Implementation

The `filter` function selects elements from a list that satisfy a given predicate:

```scheme
(define (filter pred lst)
  (cond ((null? lst) '())
        ((pred (car lst)) (cons (car lst) (filter pred (cdr lst))))
        (else (filter pred (cdr lst)))))
```

Example usage:

```scheme
(filter even? '(1 2 3 4 5 6))  ; => (2 4 6)
(filter odd? '(1 2 3 4 5 6))   ; => (1 3 5)
```

## Reduce Implementation

The `reduce` function combines elements of a list using a binary function:

```scheme
(define (reduce f lst . initial)
  (cond ((null? lst) (if (null? initial) 
                         (error "reduce: empty list with no initial value")
                         (car initial)))
        ((null? initial) (reduce f (cdr lst) (car lst)))
        (else (reduce f (cdr lst) (f (car initial) (car lst))))))
```

Example usage:

```scheme
(reduce + '(1 2 3 4 5))      ; => 15
(reduce * '(1 2 3 4 5))      ; => 120
(reduce + '(1 2 3 4 5) 10)   ; => 25
```

## Combining Higher-Order Functions

Higher-order functions can be combined to create powerful data processing pipelines:

```scheme
;; Sum of squares of even numbers
(reduce + (map square (filter even? '(1 2 3 4 5 6 7 8 9 10))))  ; => 220

;; Product of incremented odd numbers
(reduce * (map add1 (filter odd? '(1 2 3 4 5))))  ; => 48 (2*4*6)
```

## Implementing Function Composition with Reduce

The `reduce` function can be used to implement function composition:

```scheme
(define (compose-with-reduce . fns)
  (lambda (x)
    (reduce (lambda (result f) (f result))
            (reverse fns)
            x)))
```

This implementation takes any number of functions and returns a new function that applies them in right-to-left order (mathematical convention).

Example usage:

```scheme
(define f1 (compose-with-reduce square add1 double))
(f1 5)  ; => 121 (square(add1(double(5))) = square(add1(10)) = square(11) = 121)
```

## Relationship to Function Composition

Higher-order functions and function composition are closely related concepts:

1. **Function Composition** combines functions to create new functions
2. **Higher-Order Functions** operate on functions and data

Together, they form the foundation of functional programming and enable powerful abstractions.

## Tail Recursion Considerations

The implementations of `map`, `filter`, and `reduce` shown above are not tail-recursive, which means they can cause stack overflow for large lists. For production use, these functions should be implemented with proper tail recursion:

```scheme
;; Tail-recursive map
(define (map f lst)
  (let loop ((lst lst) (result '()))
    (if (null? lst)
        (reverse result)
        (loop (cdr lst) (cons (f (car lst)) result)))))

;; Tail-recursive filter
(define (filter pred lst)
  (let loop ((lst lst) (result '()))
    (cond ((null? lst) (reverse result))
          ((pred (car lst)) (loop (cdr lst) (cons (car lst) result)))
          (else (loop (cdr lst) result)))))

;; Tail-recursive reduce
(define (reduce f lst . initial)
  (let ((init (if (null? initial) (car lst) (car initial)))
        (rest (if (null? initial) (cdr lst) lst)))
    (let loop ((lst rest) (result init))
      (if (null? lst)
          result
          (loop (cdr lst) (f result (car lst)))))))
```

## Example Files

The following example files demonstrate higher-order functions:

1. `higher_order_functions.esk`: Demonstrates map, filter, reduce, and their combinations
2. `function_composition_complete.esk`: Comprehensive example with function composition
3. `function_composition_n.esk`: Demonstrates n-ary function composition using lists

## Conclusion

Higher-order functions are a powerful tool in functional programming that enable concise and expressive code. By implementing `map`, `filter`, and `reduce`, Eshkol gains the ability to process collections of data in a functional style.

Combined with function composition, higher-order functions provide a solid foundation for functional programming in Eshkol.
