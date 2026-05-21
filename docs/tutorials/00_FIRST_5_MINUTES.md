# Your First 5 Minutes with Eshkol

No theory, no background — just type and see what happens.

---

## Install (30 seconds)

**macOS:**
```bash
brew tap tsotchke/eshkol && brew install eshkol
```

**Linux:**
```bash
# Download from GitHub Releases
curl -LO https://github.com/tsotchke/eshkol/releases/latest
# Extract and add to PATH
```

**No install — browser:**
Go to [eshkol.ai](https://eshkol.ai) and type in the REPL.

---

## Hello World (30 seconds)

```bash
$ eshkol-repl
> (display "Hello, world!")
Hello, world!
> (+ 1 2 3 4 5)
15
> (define (square x) (* x x))
> (square 7)
49
```

---

## 5 Things That Will Surprise You

### 1. The compiler does calculus

```scheme
> (derivative (lambda (x) (* x x x)) 2.0)
12.0
```

That's `d/dx(x^3)` at `x=2`. The answer is `3x^2 = 12`. No numerical
approximation — the compiler computed the exact derivative.

### 2. Numbers never overflow

```scheme
> (expt 2 256)
115792089237316195423570985008687907853269984665640564039457584007913129639936
> (* 999999999999999999 999999999999999999)
999999999999999998000000000000000001
```

Integers automatically promote to arbitrary precision. No overflow, no
truncation, no surprises.

### 3. Fractions stay exact

```scheme
> (+ 1/3 1/6)
1/2
> (* 2/3 3/4)
1/2
> (+ 1/7 1/7 1/7 1/7 1/7 1/7 1/7)
1
```

No floating-point rounding. Rationals are first-class values.

### 4. Functions are values

```scheme
> (map (lambda (x) (* x x)) '(1 2 3 4 5))
(1 4 9 16 25)
> (filter even? '(1 2 3 4 5 6 7 8))
(2 4 6 8)
> (fold-left + 0 '(1 2 3 4 5))
15
```

Pass functions to functions. Transform lists in one line.

### 5. Redefine anything, anytime

```scheme
> (define (greet) (display "Hello"))
> (greet)
Hello
> (define (greet) (display "Hey!"))
> (greet)
Hey!
```

The REPL hot-reloads every definition. Functions, variables, lambdas —
redefine and the change takes effect immediately.

---

## Your First Program (2 minutes)

Create a file called `hello.esk`:

```scheme
(define (factorial n)
  (if (= n 0) 1
      (* n (factorial (- n 1)))))

(display "10! = ")
(display (factorial 10))
(newline)

(display "100! = ")
(display (factorial 100))
(newline)
```

Compile and run:

```bash
$ eshkol-run hello.esk -o hello
$ ./hello
10! = 3628800
100! = 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
```

---

## What Next?

| I want to... | Start here |
|---|---|
| Learn the basics | [Tutorial 12: Lists](12_LISTS.md), [Tutorial 17: Functional Programming](17_FUNCTIONAL_PROGRAMMING.md) |
| Do machine learning | [Tutorial 1: Autodiff and ML](01_AUTODIFF_AND_ML.md), [Project: Neural Network](21_PROJECT_NEURAL_NETWORK.md) |
| Build something for the web | [Tutorial 18: Web Platform](18_WEB_PLATFORM.md) |
| Understand the unique features | [Tutorial 3: Weight Transformer](03_WEIGHT_MATRIX_TRANSFORMER.md), [Tutorial 4: Consciousness Engine](04_CONSCIOUSNESS_ENGINE.md) |
| See complete working programs | [Project tutorials](README.md#project-tutorials) |
| Just explore | Paste code into the REPL at [eshkol.ai](https://eshkol.ai) |
