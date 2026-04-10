# Why Eshkol?

A side-by-side comparison showing what Eshkol does that other languages
can't — or make much harder.

---

## 1. Derivatives without a framework

**Python (requires JAX):**
```python
import jax
import jax.numpy as jnp

def f(x):
    return x ** 3

grad_f = jax.grad(f)
print(grad_f(2.0))  # 12.0
# Requires: pip install jax, GPU drivers, 500MB+ dependencies
```

**Eshkol:**
```scheme
(derivative (lambda (x) (* x x x)) 2.0)  ;; => 12.0
;; Built into the compiler. Zero dependencies.
```

---

## 2. No overflow — ever

**Python:**
```python
2 ** 64       # 18446744073709551616 (Python handles this)
1/3 + 1/6    # 0.49999999999999994 (floating point error!)
```

**JavaScript:**
```javascript
2 ** 64       // 18446744073709552000 (WRONG — float64 precision lost)
1/3 + 1/6    // 0.49999999999999994
```

**Eshkol:**
```scheme
(expt 2 64)    ;; => 18446744073709551616 (exact)
(+ 1/3 1/6)   ;; => 1/2 (exact rational — not 0.4999...)
```

---

## 3. Hot-reload in the REPL

**Python:**
```python
# Define a function
def f(x): return x * 2
f(5)  # 10

# Redefine — but callers may have cached the old version
def f(x): return x * 3
f(5)  # 15 (works here, but not always in complex programs)
```

**Eshkol:**
```scheme
(define (f x) (* x 2))
(f 5)                    ;; => 10
(define (f x) (* x 3))
(f 5)                    ;; => 15
;; ALL callers see the new definition — even closures that
;; captured f by name. R7RS-correct dynamic name resolution.
```

---

## 4. Programs as neural network weights

No other language does this. Eshkol compiles programs to bytecode,
encodes the bytecode VM into a 5-layer transformer's weight matrices,
and executes programs as matrix multiplications. 55/55 test programs
verified across three execution paths.

This means:
- Programs are differentiable (gradient descent on program behaviour)
- The interpreter itself is a neural network
- You can learn to improve the interpreter via backpropagation

---

## 5. Built-in AI reasoning

**Python (requires multiple libraries):**
```python
# Logic programming → need pyDatalog or kanren
# Factor graphs → need pgmpy or pomegranate
# Global workspace → need custom implementation
# All three together → research project
```

**Eshkol (22 builtins, zero dependencies):**
```scheme
(define kb (make-kb))
(kb-assert! kb (make-fact 'parent 'alice 'bob))
(kb-query kb (make-fact 'parent ?p 'bob))
;; => ((parent alice bob))

(define fg (make-factor-graph 3))
(fg-add-factor! fg 0 1 #(0.9 0.1 0.1 0.9))
(fg-infer! fg 10)
(free-energy fg #(0 1))
;; Bayesian inference in one line
```

---

## 6. Zero garbage collection

Eshkol uses arena allocation with ownership-aware lexical regions (OALR).
Memory is freed deterministically at scope exit — no GC pauses, no
unpredictable latency spikes, no "GC took 200ms" in your ML training loop.

---

## 7. One language, every platform

| Target | How |
|---|---|
| macOS native (ARM64 + x64) | `eshkol-run program.esk -o binary` |
| Linux native (x64 + ARM64) | Same command |
| Windows native (x64 + ARM64) | Same command |
| WebAssembly (browser) | `eshkol-run program.esk -w -o program.wasm` |
| GPU (Metal + CUDA) | Automatic dispatch for large tensors |
| Portable bytecode | `eshkol-run program.esk -B program.eskb` |

---

## The Eshkol Stack

```
┌──────────────────────────────────────────┐
│  Your Code (.esk)                        │
├──────────────────────────────────────────┤
│  555+ builtins: ML, signal, logic, math  │
├──────────────────────────────────────────┤
│  Autodiff (forward + reverse mode)       │
├──────────────────────────────────────────┤
│  Arena Memory (zero GC, OALR)            │
├──────────────────────────────────────────┤
│  LLVM 21 backend ──── Bytecode VM        │
│  (AOT native)         (portable, WASM)   │
├──────────────────────────────────────────┤
│  Metal / CUDA / SIMD / XLA               │
└──────────────────────────────────────────┘
```

---

## Get Started

```bash
brew tap tsotchke/eshkol && brew install eshkol
```

Or just go to [eshkol.ai](https://eshkol.ai) and start typing.
