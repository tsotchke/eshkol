# Eshkol Quick Reference Card

**v1.1-accelerate** -- 555+ built-in functions

## Basics

```scheme
;; Variables
(define x 42)
(define pi 3.14159)

;; Functions
(define (square x) (* x x))
(define (add a b) (+ a b))

;; Lambdas
(lambda (x) (* x 2))

;; Let bindings
(let ((x 5) (y 10)) (+ x y))
(let* ((x 5) (y (* x 2))) y)    ;; sequential binding
(letrec ((even? (lambda (n) (if (= n 0) #t (odd? (- n 1)))))
         (odd? (lambda (n) (if (= n 0) #f (even? (- n 1))))))
  (even? 10))

;; Module system
(require stdlib)
(require core.list.transform)
(require signal.fft)
```

## Control Flow

```scheme
(if condition then-expr else-expr)

(cond ((test1) result1)
      ((test2) result2)
      (else default))

(when test expr...)
(unless test expr...)

(and a b c)  ;; short-circuit
(or a b c)   ;; short-circuit

(case expr
  ((val1) result1)
  ((val2 val3) result2)
  (else default))
```

## Lists

```scheme
(list 1 2 3)              ;; create list
(cons 1 (list 2 3))       ;; prepend
(car lst)                 ;; first element
(cdr lst)                 ;; rest of list
(cadr lst)                ;; second element (car (cdr lst))
(caddr lst)               ;; third element
(length lst)              ;; list length
(append lst1 lst2)        ;; concatenate
(reverse lst)             ;; reverse
(list-ref lst n)          ;; nth element
(list-copy lst)           ;; shallow copy
(member x lst)            ;; find element
(assoc key alist)         ;; lookup in assoc list
(take n lst)              ;; first n elements
(drop n lst)              ;; skip first n elements
```

## Higher-Order Functions

```scheme
(map f lst)               ;; apply f to each element
(filter pred lst)         ;; select matching elements
(fold f init lst)         ;; reduce left
(fold-right f init lst)   ;; reduce right
(for-each f lst)          ;; side effects
(any pred lst)            ;; any match?
(every pred lst)          ;; all match?
(find pred lst)           ;; first match
(reduce f init lst)       ;; general reduction
(string-for-each f str)   ;; iterate chars
(string-map f str)        ;; map over chars
(vector-for-each f vec)   ;; iterate vector elements
(vector-map f vec)        ;; map over vector elements
```

## Vectors & Tensors

```scheme
(vector 1 2 3)            ;; create vector (heterogeneous, 16-byte tagged)
(vref v i)                ;; get element
(vector-set! v i x)       ;; set element
(make-vector n val)       ;; n copies of val
(vector-length v)         ;; length

;; Tensor creation (homogeneous doubles, 8-byte)
#(1.0 2.0 3.0)            ;; tensor literal
(zeros m n)               ;; m*n zeros
(ones m n)                ;; m*n ones
(eye n)                   ;; n*n identity
(arange n)                ;; 0 to n-1
(linspace a b n)          ;; n values from a to b
(reshape v m n)           ;; reshape to m*n
(make-tensor (list m n) val)  ;; m*n filled with val
(rand m n)                ;; m*n random values

;; Operations
(tensor-add a b)          ;; element-wise add
(tensor-sub a b)          ;; element-wise subtract
(tensor-mul a b)          ;; element-wise multiply
(tensor-div a b)          ;; element-wise divide
(tensor-dot a b)          ;; dot product
(matmul A B)              ;; matrix multiply (auto-dispatches SIMD/BLAS/GPU)
(transpose M)             ;; transpose
(norm v)                  ;; L2 norm
(trace M)                 ;; sum of diagonal
(outer u v)               ;; outer product
(tensor-sum v)            ;; sum all
(tensor-mean v)           ;; mean
(tensor-reduce v f init)  ;; custom reduction
(tensor-reduce-all M f init)  ;; reduce entire tensor
(flatten M)               ;; flatten to 1D
(tensor-shape M)          ;; get dimensions
```

## Automatic Differentiation

```scheme
;; Symbolic (compile-time)
(diff (* x x) x)          ;; -> (* 2 x)

;; Forward-mode (dual numbers)
(derivative f x)          ;; df/dx at x

;; Reverse-mode (computation graphs)
(gradient f v)            ;; nabla f at v
(jacobian f v)            ;; Jacobian matrix
(hessian f v)             ;; Hessian matrix

;; Vector calculus
(divergence F v)          ;; div(F) at v
(curl F v)                ;; curl(F) at v (3D)
(laplacian f v)           ;; laplacian(f) at v
(directional-derivative f v dir)
```

## Exact Arithmetic

```scheme
;; Bignums (automatic promotion beyond 64-bit)
(expt 2 256)              ;; arbitrary precision integer
(* 99999999999999999 99999999999999999)  ;; exact result

;; Rational numbers
1/3                        ;; rational literal
(+ 1/3 1/6)               ;; -> 1/2 (exact)
(* 2/3 3/4)               ;; -> 1/2 (auto-reduces)
(numerator 3/7)            ;; -> 3
(denominator 3/7)          ;; -> 7

;; Exactness predicates
(exact? 42)                ;; -> #t
(exact? 1/3)               ;; -> #t
(inexact? 3.14)            ;; -> #t

;; Conversion
(exact->inexact 1/3)       ;; -> 0.3333333333333333
(inexact->exact 0.5)       ;; -> 1/2
(rationalize 3.14 0.01)    ;; -> 22/7

;; number->string / string->number with bignums
(number->string (expt 2 128))
(string->number "999999999999999999999")  ;; -> bignum
```

## Complex Numbers

```scheme
;; Creation
(make-rectangular 3.0 4.0)    ;; 3+4i
(make-polar 5.0 0.927)        ;; r*e^(i*theta)

;; Accessors
(real-part z)              ;; real component
(imag-part z)              ;; imaginary component
(magnitude z)              ;; |z| = sqrt(a^2 + b^2)
(angle z)                  ;; arg(z) in radians

;; Arithmetic (all standard ops work)
(+ z1 z2)                 ;; complex addition
(* z1 z2)                 ;; complex multiplication
(/ z1 z2)                 ;; complex division (Smith's formula)
(sqrt (make-rectangular -1.0 0.0))  ;; -> 0+1i
(exp (make-rectangular 0.0 pi))     ;; -> -1+0i (Euler's identity)

;; Predicates
(complex? z)               ;; -> #t
(number? z)                ;; -> #t
```

## Continuations & Exception Handling

```scheme
;; First-class continuations
(call/cc (lambda (k)       ;; capture continuation
  (+ 1 (k 42))))           ;; -> 42

(call-with-current-continuation  ;; long form
  (lambda (k) (k 99)))

;; Dynamic wind (resource management)
(dynamic-wind
  before-thunk              ;; called on entry
  body-thunk                ;; main computation
  after-thunk)              ;; called on exit (even via continuation)

;; Exception handling
(guard (exn                 ;; try/catch equivalent
        ((string? exn) (string-append "caught: " exn))
        (#t "unknown error"))
  (raise "error message"))

(with-exception-handler     ;; low-level handler
  handler-proc
  thunk)

;; Raise exceptions
(raise value)               ;; raise any value as exception

;; Multiple values
(values 1 2 3)              ;; return multiple values
(call-with-values
  (lambda () (values 1 2))
  +)                        ;; -> 3
```

## Parallel Primitives

```scheme
;; Parallel versions of standard HOFs
(parallel-map f lst)           ;; map across cores
(parallel-filter pred lst)     ;; filter across cores
(parallel-fold f init lst)     ;; parallel reduction
(parallel-for-each f lst)      ;; parallel side effects

;; Futures (explicit concurrency)
(define f (future thunk))      ;; spawn computation
(force f)                      ;; wait for result

;; Parallel execute (multiple thunks)
(parallel-execute thunk1 thunk2 thunk3)

;; Auto falls back to sequential for small inputs
```

## GPU Acceleration

```scheme
;; Automatic dispatch (matmul selects SIMD/BLAS/GPU by cost model)
(matmul A B)                   ;; fastest backend auto-selected

;; Explicit GPU operations
(gpu-matmul A B)               ;; force GPU matrix multiply
(gpu-elementwise f A B)        ;; element-wise operation on GPU
(gpu-reduce f M)               ;; GPU reduction
(gpu-softmax v)                ;; softmax on GPU
(gpu-transpose M)              ;; GPU transpose

;; Backends: SIMD | Apple Accelerate AMX (~1.1 TFLOPS)
;;           | Metal (macOS, sf64) | CUDA (Linux, native f64)
```

## Consciousness Engine

```scheme
;; Logic variables and unification
(define s (make-substitution))     ;; empty substitution
(define s1 (unify ?x 42 s))       ;; bind ?x to 42
(walk ?x s1)                      ;; -> 42
(unify ?x ?y s)                   ;; bind ?x and ?y together

;; Predicates
(logic-var? ?x)                ;; -> #t
(substitution? s)              ;; -> #t

;; Knowledge bases
(define kb (make-kb))
(kb-assert! kb (make-fact 'parent (list 'alice 'bob)))
(kb-query kb 'parent)          ;; -> list of matching facts
(kb? kb)                       ;; -> #t
(fact? f)                      ;; -> #t

;; Factor graphs (probabilistic inference)
(define fg (make-factor-graph n))      ;; n variables
(fg-add-factor! fg vars cpt)          ;; add factor with CPT
(fg-infer! fg iterations)             ;; belief propagation
(fg-update-cpt! fg factor-idx new-cpt) ;; update CPT (learning)
(factor-graph? fg)                     ;; -> #t

;; Free energy
(free-energy fg observations)          ;; surprise measure
;; observations: #(var_idx observed_state) pairs
(expected-free-energy fg action)       ;; EFE for action selection

;; Global workspace (consciousness)
(define ws (make-workspace))
(ws-register! ws "name" module-closure)  ;; register cognitive module
(ws-step! ws input-tensor)              ;; competition + broadcast
(workspace? ws)                         ;; -> #t
```

## Signal Processing

```scheme
;; FFT / Inverse FFT
(fft signal)                   ;; Fast Fourier Transform
(ifft spectrum)                ;; Inverse FFT

;; Window functions
(window-hamming n)             ;; Hamming window of size n
(window-hann n)                ;; Hann window of size n

;; Convolution
(convolve signal kernel)       ;; linear convolution

;; FIR filter
(fir-filter coeffs signal)     ;; finite impulse response

;; IIR filter
(iir-filter b-coeffs a-coeffs signal)  ;; infinite impulse response

;; Filter design
(butterworth-lowpass order cutoff)     ;; Butterworth lowpass
```

## Web Platform (WASM)

```scheme
;; Compile: eshkol-run prog.esk --target wasm -o prog.wasm

;; DOM manipulation
(web-create-element tag)           ;; create element
(web-set-text elem text)           ;; set text content
(web-add-event-listener elem event handler)  ;; attach handler

;; Canvas API
(web-canvas-create w h)            ;; create canvas
(web-canvas-clear canvas)          ;; clear canvas
(web-canvas-fill-rect canvas x y w h color)
(web-canvas-stroke-circle canvas cx cy r color)
(web-canvas-draw-text canvas text x y color)
(web-request-animation-frame callback)
```

## Math Functions

```scheme
;; Arithmetic
+ - * / abs floor ceiling round truncate
modulo remainder quotient gcd lcm min max expt

;; Trigonometric
sin cos tan asin acos atan atan2
sinh cosh tanh asinh acosh atanh

;; Exponential
exp log log10 log2 sqrt pow
```

## Type Predicates

```scheme
(null? x)    (pair? x)     (list? x)
(number? x)  (integer? x)  (real? x)
(complex? x) (rational? x)
(exact? x)   (inexact? x)
(string? x)  (char? x)     (symbol? x)
(vector? x)  (procedure? x)
(zero? x)    (positive? x) (negative? x)
(even? x)    (odd? x)
(boolean? x) (port? x)

;; Consciousness engine types
(logic-var? x)     (substitution? x)
(kb? x)            (fact? x)
(factor-graph? x)  (workspace? x)
```

## I/O

```scheme
(display x)               ;; print value
(newline)                 ;; print newline
(printf fmt args...)      ;; formatted print

;; Files
(open-input-file path)
(open-output-file path)
(read-line port)
(read-char port)
(peek-char port)
(write-string str port)
(close-port port)
```

## C Interop

```scheme
(extern return-type name param-types...)
(extern void printf char* ...)
(extern double sin double)

;; With aliasing
(extern void log :real printf char* ...)
```

## Standard Library (require stdlib)

```scheme
;; Combinators
(compose f g)             ;; (f (g x))
(identity x)              ;; x
(constantly x)            ;; (lambda (_) x)
(flip f)                  ;; swap arguments
(negate pred)             ;; (not (pred x))

;; Currying
(curry2 f)                ;; curry 2-arg function
(curry3 f)                ;; curry 3-arg function
(partial2 f x)            ;; fix first arg

;; List utilities
(iota n)                  ;; (0 1 ... n-1)
(iota-from n start)       ;; (start start+1 ... start+n-1)
(repeat n x)              ;; n copies of x
(sort lst less?)          ;; merge sort
(partition pred lst)      ;; split by predicate
(all? pred lst)           ;; all satisfy?
(none? pred lst)          ;; none satisfy?
(count-if pred lst)       ;; count matches

;; ML utilities
(gradient-descent loss params lr steps)
```

## Math Library

```scheme
;; Linear algebra
(det M n)                 ;; determinant
(inv M n)                 ;; inverse
(solve A b n)             ;; solve Ax=b
(cross u v)               ;; 3D cross product
(dot u v)                 ;; dot product
(normalize v)             ;; unit vector

;; Numerical methods
(power-iteration A n max tol)  ;; eigenvalue
(integrate f a b n)            ;; Simpson's rule
(newton f df x0 tol max)       ;; root finding

;; Statistics
(variance v)
(std v)
(covariance u v)
```

## REPL Commands

```
:help        Show help
:quit        Exit REPL
:env         Show symbols
:load FILE   Load file
:reload      Reload last
:time EXPR   Time execution
:ast EXPR    Show AST
:clear       Clear screen
:history     Show history
```

## CLI Options

```bash
eshkol-run [options] file.esk

-h, --help          Help
-d, --debug         Debug mode
-a, --dump-ast      Dump AST
-i, --dump-ir       Dump LLVM IR
-o, --output FILE   Output binary name
-c, --compile-only  Object file only
-l LIB              Link library
-L PATH             Library path
-n, --no-stdlib     Skip stdlib
--shared-lib        Compile as library (LinkOnceODR)
--target wasm       WebAssembly output
```
