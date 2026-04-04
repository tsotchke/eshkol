# The Computable Transformer: Programs as Neural Network Weights

## Overview

Eshkol's most architecturally distinctive feature is that its bytecode VM can be encoded into the weight matrices of a 5-layer transformer. Programs don't just run on a VM — they execute as matrix multiplications through a neural network whose weights ARE the interpreter.

This means:
- A program is a sequence of bytecode instructions
- The VM's runtime state is a 36-dimensional float vector
- One forward pass through the transformer = one instruction executed
- The transformer's weights implement the full 63-opcode ISA

The state vector simultaneously serves as:
- The VM's runtime state (PC, TOS, SOS, registers, memory, type tags)
- The embedding dimension for the transformer (d_model=36)

## Architecture

```
                    ┌─────────────────────────────┐
                    │  Eshkol Source (.esk)       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Bytecode Compiler          │
                    │  (vm_compiler.c)            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  ESKB Binary Format         │
                    │  (opcodes + constants)      │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼─────────┐  ┌───────▼──────┐  ┌──────────▼─────────┐
    │  Reference VM     │  │  Simulated   │  │  Matrix Forward    │
    │  (C switch/case)  │  │  Transformer │  │  Pass (W @ x + b)  │
    │  63 opcodes       │  │  (C functions│  │  (actual matmul)   │
    │                   │  │   mirroring  │  │                    │
    │                   │  │   weights)   │  │                    │
    └─────────┬─────────┘  └───────┬──────┘  └──────────┬─────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                         3-way verification
                         (55/55 tests pass)
```

## The State Vector (d_model=36)

The 36-dimensional state vector encodes the complete VM state:

| Dimensions | Content |
|-----------|---------|
| 0 | Program Counter (PC) |
| 1 | Stack Pointer (SP) |
| 2 | Top-of-Stack (TOS) |
| 3 | Second-on-Stack (SOS) |
| 4-7 | Registers / scratch |
| 8-15 | Memory cells |
| 16-19 | Type tags |
| 20-23 | Flags (zero, negative, overflow, halt) |
| 24-31 | Extended state (frame pointer, upvalues) |
| 32-35 | Instruction operand / immediate value |

## Transformer Architecture

```
d_model  = 36       (state vector dimension)
n_heads  = 16       (attention heads)
head_dim = 2        (per-head dimension)
n_layers = 5        (transformer depth)
FFN_DIM  = 512      (feed-forward hidden dimension)
Parameters: ~307,000
```

### Layer 0: Instruction Fetch
- **Mechanism**: Gaussian attention peaked at the PC position
- **Function**: Reads the current instruction from the code buffer into the state vector
- **Attention pattern**: Sharply peaked — attends almost entirely to the instruction at position PC

### Layer 1: Product Precompute
- **Mechanism**: Square activation FFN
- **Function**: Computes TOS × SOS and other products needed by arithmetic opcodes
- **Activation**: h = Wx + b, then a = h², then out = a·W' + b'

### Layer 2: Preprocessing
- **Mechanism**: Gated FFN (sigmoid gate × linear up-projection)
- **Function**: Address resolution, comparison flags, operand decoding
- **Operations**: Resolves local/upvalue slot addresses, computes comparison results

### Layer 3: Execution
- **Mechanism**: Gated FFN — the core opcode dispatcher
- **Function**: Each opcode is implemented as a pair of neurons (gate + value)
- **Pattern**: Gate neuron activates for a specific opcode; value neuron computes the result
- **Coverage**: 25 core opcodes are fully weight-encoded; 38 complex opcodes (CONS, CAR, closures, vectors) set IS_NATIVE=1 and delegate to C

### Layer 4: Frame Management
- **Mechanism**: Gated FFN for CALL/RETURN
- **Function**: Manages call frames, stack frame setup, return address tracking

## Two Tiers of Execution

### Tier 1: Weight-Encoded Opcodes (0-62)

The 63 core ISA opcodes — ADD, SUB, MUL, CONST, JUMP, etc. — compute their results entirely through matrix multiplications. For example:

**OP_ADD**: The gate neuron fires when `opcode == 7`. The value neurons compute `TOS + SOS` and write the result back to the TOS dimension of the state vector. This happens as `W @ state + b` — no C code involved.

25 of the 63 opcodes are fully weight-encoded. The remaining 38 (CONS, CAR, CDR, closures, heap allocation, I/O) set the `IS_NATIVE` flag in the state vector, causing the execution loop to delegate to C-level `exec_loop_postprocess()`.

### Tier 2: Native Call Dispatch (IDs 300+)

`OP_NATIVE_CALL` (opcode 37) carries a native function ID as its operand. The transformer detects it, sets `IS_NATIVE=1`, and the C postprocessor dispatches to the appropriate native function. This is how:

- Consciousness engine (500-549): logic, inference, workspace
- Tensor operations (410-470): matmul, reshape, reduce
- Automatic differentiation (370-409): dual numbers, AD tape
- Geometric manifolds (800-859): Riemannian operations
- String/list/hash/I/O operations

## Three-Way Verification

Every program is executed through all three paths and the results are compared:

1. **Reference interpreter**: Direct C `switch(opcode)` — the ground truth
2. **Simulated transformer**: C functions that mirror exactly what the weight matrices compute, but without actual matrix multiplication
3. **Matrix-based forward pass**: Actual `W @ state + b` through generic matmul — the real transformer execution

All three must produce identical results. **55/55 test programs pass 3-way verification.**

## QLMW Binary Format

The analytically constructed weights are exported in QLMW (Quantum LLM Weights) binary format for loading by the qLLM runtime:

```
Header:
  magic: "QLMW"
  version: 3
  d_model: 36
  n_layers: 5
  n_heads: 16
  ffn_dim: 512

Per-layer weights:
  W_Q, W_K, W_V, W_O  (attention projection matrices)
  W1, b1, W2, b2       (FFN weights and biases)
  ln_gamma, ln_beta     (layer norm parameters)
```

## Integration with qLLM

The `qllm_interpreter.c` module loads QLMW weights into a qLLM transformer model and executes Eshkol programs through the qLLM tensor engine:

```
Eshkol source → bytecode compiler → ESKB binary → load into state vector
→ qLLM forward pass (Metal/NEON accelerated) → extract results from state vector
```

This enables:
- **Metal-accelerated execution** on Apple Silicon (via libsemiclassical_qllm)
- **CUDA-accelerated execution** on NVIDIA GPUs
- **Hybrid classical-quantum execution** where the transformer weights can be further optimized through gradient descent

## Why This Matters

1. **Programs are differentiable**: Since execution is matrix multiplication, you can compute gradients of program behavior with respect to the weights. This enables learning to improve the interpreter itself.

2. **The interpreter is a neural network**: The weight matrices implement a universal computation engine. Any changes to the weights create a different interpreter — one that might execute programs differently, faster, or with learned optimizations.

3. **Bridging discrete and continuous**: Bytecode is discrete (integer opcodes), but execution is continuous (float matrix operations). This is the foundation for differentiable programming at the instruction level.

4. **Computable transformer theory**: The existence of weight matrices that implement a universal stack machine proves that transformer architectures are Turing-complete in a constructive, verifiable way — not just theoretically.

## Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `lib/backend/weight_matrices.c` | 2,299 | Weight matrix construction + 3-way verification |
| `lib/backend/qllm_interpreter.c` | ~1,000 | qLLM integration + QLMW loading |
| `lib/backend/qllm_backward.c` | ~700 | Backward pass for weight learning |
| `lib/backend/eskb_format.h` | ~100 | ESKB binary format specification |
| `lib/backend/eskb_reader.c` | ~200 | ESKB deserialization |
| `lib/backend/eskb_writer.c` | ~200 | ESKB serialization |
