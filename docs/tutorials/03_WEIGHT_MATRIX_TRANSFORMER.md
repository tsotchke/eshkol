# Tutorial 3: The Weight Matrix Transformer

Eshkol's most architecturally distinctive feature: bytecode programs execute
as matrix multiplications through a 5-layer transformer whose weights ARE
the interpreter. Programs are literally neural network weights.

---

## The Concept

```
Eshkol source → bytecode (63 opcodes) → state vector (36 floats)
→ transformer forward pass (W @ x + b) → next state → repeat
```

One forward pass = one instruction executed. The transformer's 307,000
parameters implement a universal stack machine. This is verified by running
every program three ways and comparing results:

1. **Reference VM** — C `switch(opcode)` interpreter
2. **Simulated transformer** — C functions mirroring the weight matrices
3. **Matrix forward pass** — actual `W @ state + b` matmul

All three produce identical results. 55/55 test programs pass 3-way
verification.

---

## The State Vector (d_model = 36)

The VM's entire runtime state fits in 36 floats:

| Dimensions | Content |
|---|---|
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

---

## The 5 Layers

| Layer | Mechanism | What it does |
|---|---|---|
| 0 | Gaussian attention | Instruction fetch — reads opcode at PC position |
| 1 | Square-activation FFN | Product precompute — computes TOS * SOS for arithmetic |
| 2 | Gated FFN | Preprocessing — address resolution, comparison flags |
| 3 | Gated FFN | Execution — each opcode is a (gate, value) neuron pair |
| 4 | Gated FFN | Frame management — CALL/RETURN stack frame handling |

### How OP_ADD Works in Weights

The gate neuron fires when `opcode == 7`. The value neurons compute
`TOS + SOS` and write the result back. This is pure `W @ state + b` —
no C code, no branching, just linear algebra.

25 of 63 opcodes are fully weight-encoded. The remaining 38 (CONS, CAR,
closures, heap operations) set an `IS_NATIVE` flag and delegate to C.

---

## Two Tiers of Execution

**Tier 1: Weight-encoded opcodes** — arithmetic, comparisons, jumps, stack
manipulation. Computed entirely through matrix multiplication.

**Tier 2: Native dispatch (IDs 300+)** — `OP_NATIVE_CALL` carries a native
function ID. The transformer detects it and delegates to C:

| ID Range | Subsystem |
|---|---|
| 300-369 | Core (cons, car, cdr, display, I/O) |
| 370-409 | Automatic differentiation (dual numbers, AD tape) |
| 410-470 | Tensor operations (matmul, reshape, reduce) |
| 500-549 | Consciousness engine (logic, inference, workspace) |
| 800-859 | Geometric manifolds (Riemannian operations) |

---

## Using It

### Compile to Bytecode

```bash
$ eshkol-run program.esk -o program -B program.eskb
```

The `-B` flag emits an ESKB binary (section-based bytecode container with
CRC32 integrity check) alongside the native binary.

### Export as QLMW Weights

The weight matrices are exported in QLMW (Quantum LLM Weights) binary
format:

```
Header: magic "QLMW", version 3, d_model 36, n_layers 5
Per-layer: W_Q, W_K, W_V, W_O, W1, b1, W2, b2, ln_gamma, ln_beta
```

These can be loaded by the qLLM runtime for Metal/CUDA-accelerated
execution of Eshkol programs as neural network inference.

---

## Why This Matters

1. **Programs are differentiable** — execution is matrix multiplication, so
   you can compute gradients of program behavior with respect to the
   weights. This enables learning to improve the interpreter itself.

2. **The interpreter is a neural network** — weight changes create a
   different interpreter. One that might execute programs faster or with
   learned optimizations.

3. **Bridging discrete and continuous** — bytecode is discrete (integer
   opcodes), execution is continuous (float matmul). This is the foundation
   for differentiable programming at the instruction level.

4. **Constructive Turing-completeness proof** — the weight matrices
   implement a universal stack machine, proving transformer architectures
   are Turing-complete in a verifiable, constructive way.

---

## Source Files

| File | Purpose |
|---|---|
| `lib/backend/weight_matrices.c` | Weight construction + 3-way verification |
| `lib/backend/weight_compiler.c` | Bytecode → weight matrix compilation |
| `lib/backend/qllm_interpreter.c` | qLLM integration + QLMW loading |
| `lib/backend/qllm_backward.c` | Backward pass for weight learning |
| `lib/backend/eskb_writer.c` | ESKB bytecode serialisation |
| `lib/backend/eskb_reader.c` | ESKB bytecode deserialisation |

---

*Next: Tutorial 4 — The Consciousness Engine*
