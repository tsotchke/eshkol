# Tutorial 2: The Eshkol Bytecode VM

Eshkol has two execution backends: the **LLVM native compiler** (AOT, fast,
produces machine code) and the **bytecode VM** (interpreted, portable,
instant startup). This tutorial shows you when and how to use each.

---

## When to Use What

| Feature | LLVM Backend | Bytecode VM |
|---|---|---|
| Startup time | ~100ms (compile first) | Instant |
| Execution speed | Native (fast) | Interpreted (~10-50x slower) |
| Output format | Platform binary | `.eskb` portable bytecode |
| Autodiff | Full (forward + reverse) | Forward-mode only (dual numbers) |
| Platform | Requires LLVM 21 | Runs anywhere (including WASM) |
| Best for | Production, performance | REPL, prototyping, web, scripting |

**Rule of thumb:** Use the LLVM backend for anything performance-sensitive.
Use the VM for quick iteration, the REPL, and the browser.

---

## Part 1: Running Code in the REPL

The REPL uses the LLVM JIT backend by default:

```bash
$ eshkol-repl
Eshkol REPL v1.1.13-accelerate
> (+ 1 2 3)
6
> (define (fib n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))
> (fib 10)
55
> :quit
```

Every expression is JIT-compiled to native code and executed. Definitions
persist across evaluations — you can incrementally build up a program.

### Hot Reload

The REPL supports live redefinition. Redefine any function or variable and
all subsequent calls use the new definition:

```scheme
> (define (greet name) (display "Hello, ") (display name) (newline))
> (greet "Alice")
Hello, Alice
> (define (greet name) (display "Hey ") (display name) (display "!") (newline))
> (greet "Alice")
Hey Alice!
```

This works for functions, variables, lambdas, and even across closures
that reference the redefined name.

---

## Part 2: Compiling to Native Code

For production, compile to a native binary:

```bash
# Compile
$ eshkol-run program.esk -o program

# Run
$ ./program
```

Add optimization:

```bash
$ eshkol-run program.esk -o program -O 2
```

Optimization levels: 0 (none), 1 (basic), 2 (full), 3 (aggressive).

---

## Part 3: Compiling to Bytecode

Emit portable ESKB bytecode alongside native compilation:

```bash
# Compile to both native AND bytecode
$ eshkol-run program.esk -o program -B program.eskb
```

The `.eskb` file is a section-based binary container:
- Header with magic bytes and version
- Instruction stream (63 opcodes)
- Constant pool (numbers, strings, symbols)
- Local variable table
- CRC32 integrity check

---

## Part 4: The VM Architecture

The bytecode VM is a hybrid register+stack machine:

- **63 opcodes** — arithmetic, control flow, function calls, data structures
- **555+ native call IDs** — builtins accessible from bytecode via `CALL_BUILTIN`
- **Arena memory** — same OALR system as the native backend, zero GC
- **Closure support** — first-class functions with captured environments
- **Tail call optimization** — proper tail calls for unbounded recursion

### Key Opcodes

| Opcode | Description |
|---|---|
| `PUSH` | Push constant onto stack |
| `POP` | Pop top of stack |
| `ADD`, `SUB`, `MUL`, `DIV` | Arithmetic |
| `CALL` | Call function (push frame, jump) |
| `CALL_BUILTIN` | Call native builtin by ID |
| `RET` | Return from function |
| `JMP`, `JZ` | Unconditional/conditional jump |
| `MAKE_CLOSURE` | Create closure with captured variables |
| `GET_LOCAL`, `SET_LOCAL` | Access local variables |
| `CONS`, `CAR`, `CDR` | List operations |
| `TAIL_CALL` | Tail-recursive call (reuse frame) |

---

## Part 5: The Browser REPL

The VM compiles to WebAssembly, powering the browser REPL at
[eshkol.ai](https://eshkol.ai). Everything you type in the browser runs
through the same 63-opcode ISA:

```scheme
;; These all work in the browser REPL:
(define (factorial n)
  (if (= n 0) 1
      (* n (factorial (- n 1)))))

(display (factorial 20))
;; => 2432902008176640000

;; Exact arithmetic (bignums)
(display (expt 2 64))
;; => 18446744073709551616

;; Higher-order functions
(display (map (lambda (x) (* x x)) '(1 2 3 4 5)))
;; => (1 4 9 16 25)

;; Autodiff (forward-mode dual numbers)
(define (f x) (* x x x))
(display (derivative f 2.0))
;; => 12.0
```

---

## Part 6: VM vs LLVM Feature Parity

Most features work in both backends. Key differences:

| Feature | LLVM | VM |
|---|---|---|
| Forward-mode AD | Yes | Yes (dual numbers) |
| Reverse-mode AD | Yes | No |
| Gradient/Jacobian/Hessian | Yes | No (forward-mode only) |
| Tensors | Yes (N-dimensional) | Yes (1D) |
| Complex numbers | Yes | Yes |
| Bignums + Rationals | Yes | Yes |
| call/cc | Yes | Yes |
| dynamic-wind | Yes | Yes |
| Parallel primitives | Yes | No |
| GPU acceleration | Yes (Metal/CUDA) | No |
| XLA backend | Yes | No |

---

*Next: Tutorial 3 — The Weight Matrix Transformer*
