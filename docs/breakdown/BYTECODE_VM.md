# Bytecode VM Architecture

## Overview

Eshkol has a dual backend: the primary LLVM compiler generates native machine code, and a secondary bytecode VM interprets programs through a 63-opcode instruction set. The VM serves three purposes:

1. **WebAssembly execution** — the browser REPL at eshkol.ai
2. **Weight matrix compilation** — programs compile to transformer neural network weights (see [Computable Transformer](COMPUTABLE_TRANSFORMER.md))
3. **Portable bytecode** — ESKB binary format for cross-platform distribution

## Architecture

The VM is a unity-build C system: `eshkol_vm.c` includes all modules via `#include`:

```
eshkol_vm.c (hub)
 ├─ vm_core.c       — VM struct, stack, heap, value types
 ├─ vm_run.c        — 63-opcode dispatch loop (computed-goto + switch fallback)
 ├─ vm_native.c     — 400+ native function implementations
 ├─ vm_compiler.c   — bytecode compiler (S-expression → opcodes)
 ├─ vm_parser.c     — S-expression parser, FuncChunk, locals, upvalues
 ├─ vm_macro.c      — hygienic macro expansion (syntax-rules)
 ├─ vm_tests.c      — 50 built-in tests with verified output capture
 ├─ vm_peephole.c   — peephole optimizer
 ├─ vm_symbolic_ad.c — symbolic automatic differentiation
 ├─ vm_parallel.c   — POSIX thread pool (work-stealing)
 ├─ vm_geometric.c  — Riemannian manifold operations (804-843)
 ├─ vm_gpu_dispatch.h — GPU tensor dispatch (Metal/CUDA)
 └─ Type modules:
    ├─ vm_complex.c, vm_rational.c, vm_bignum.c, vm_dual.c
    ├─ vm_autodiff.c, vm_tensor.c, vm_tensor_ops.c
    ├─ vm_logic.c, vm_inference.c, vm_workspace.c
    ├─ vm_string.c, vm_io.c, vm_hashtable.c, vm_bytevector.c
    ├─ vm_multivalue.c, vm_error.c, vm_parameter.c
    └─ vm_prelude_cache.h (pre-compiled stdlib bytecode)
```

## Instruction Set (63 Opcodes)

| Opcode | Name | Description |
|--------|------|-------------|
| 0 | NOP | No operation |
| 1 | CONST | Push constant from pool |
| 2 | NIL | Push nil |
| 3 | TRUE | Push #t |
| 4 | FALSE | Push #f |
| 5 | POP | Discard TOS |
| 6 | DUP | Duplicate TOS |
| 7-13 | ADD, SUB, MUL, DIV, MOD, NEG, ABS | Arithmetic (dual-number aware) |
| 14-19 | EQ, LT, GT, LE, GE, NOT | Comparison |
| 20-23 | GET_LOCAL, SET_LOCAL, GET_UPVALUE, SET_UPVALUE | Variable access |
| 24 | CLOSURE | Create closure with upvalue captures |
| 25 | CALL | Function call |
| 26 | TAIL_CALL | Tail call (reuses frame) |
| 27 | RETURN | Return from function |
| 28-30 | JUMP, JUMP_IF_FALSE, LOOP | Control flow |
| 31-34 | CONS, CAR, CDR, NULL_P | List operations |
| 35 | PRINT | Display value |
| 36 | HALT | Stop execution |
| 37 | NATIVE_CALL | Dispatch to native function by ID |
| 38 | CLOSE_UPVALUE | Patch closure self-reference |
| 39-44 | VEC_CREATE, VEC_REF, VEC_SET, VEC_LEN, STR_REF, STR_LEN | Data access |
| 45-50 | PAIR_P, NUM_P, STR_P, BOOL_P, PROC_P, VEC_P | Type predicates |
| 51-53 | SET_CAR, SET_CDR, POPN | Mutation |
| 54 | OPEN_CLOSURE | (reserved) |
| 55 | CALLCC | Capture continuation |
| 56 | INVOKE_CC | Invoke continuation |
| 57-58 | PUSH_HANDLER, POP_HANDLER | Exception handling |
| 59 | GET_EXN | Get current exception |
| 60 | PACK_REST | Pack variadic args into list |
| 61-62 | WIND_PUSH, WIND_POP | Dynamic-wind stack |

## Automatic Differentiation

The arithmetic opcodes (ADD, SUB, MUL, DIV) and math functions (sin, cos, exp, log, sqrt, pow) detect `VAL_DUAL` operands and dispatch to dual number arithmetic:

- ADD: `(a+b, a'+b')`
- MUL: `(a*b, a'*b + a*b')` (product rule)
- DIV: `(a/b, (a'*b - a*b')/b²)` (quotient rule)
- sin: `(sin(a), a'*cos(a))` (chain rule)

This means `(derivative (lambda (x) (* x x)) 3.0)` returns `6.0` through the bytecode VM.

## Building and Testing

```bash
# Build the standalone VM
gcc -O2 -std=c11 -w lib/backend/eshkol_vm.c -o test_vm -lm -lpthread

# Run built-in tests
ESHKOL_VM_NO_DISASM=1 ./test_vm

# Run an Eshkol program
./test_vm program.esk

# Build for WebAssembly (browser REPL)
emcc -O2 -s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME='EshkolVM' \
  -DESHKOL_VM_WASM -DESHKOL_VM_NO_DISASM \
  -I lib/backend lib/backend/vm_wasm_repl.c \
  -o site/static/eshkol-vm.js -lm
```

## ESKB Binary Format

Programs compile to ESKB (Eshkol Bytecode) binary format:

```bash
eshkol-run program.esk -B program.eskb
```

ESKB files contain:
- Magic number: `ESKB`
- Instruction count + constant count
- Opcode/operand pairs
- Constant pool (integers, floats, strings)
- CRC32 checksums

## Web Compilation Server

`exe/eshkol-server.cpp` (490 lines) provides an HTTP API for on-demand compilation:

```
POST /compile  — compile Eshkol code to WASM
  Request:  { "code": "(display 42)", "session_id": "..." }
  Response: { "wasm": "...", "session_id": "..." }

GET /health    — server health check
GET /*         — serve static files (website)
```

Build and run:
```bash
# Not built by default — build manually:
g++ -O2 -std=c++20 exe/eshkol-server.cpp -o eshkol-server \
  -I inc $(llvm-config-21 --cxxflags --ldflags --libs) -lm

# Run on port 8080
./eshkol-server --port 8080 --web-dir site/static/
```
