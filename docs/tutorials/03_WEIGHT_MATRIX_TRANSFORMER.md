# Tutorial 3: The Weight Matrix Transformer

Eshkol's most architecturally distinctive feature: bytecode programs execute
as matrix multiplications through a 6-layer transformer whose weights ARE
the interpreter. Programs are literally neural network weights.

---

## The Concept

```
Eshkol source → canonical bytecode/AD ISA (83 opcodes) → state vector (256 floats)
→ transformer forward pass (W @ x + b) → next state → repeat
```

One forward pass = one instruction executed. The transformer's 12.22 million
analytically constructed parameters implement a bounded, arena-backed stack
machine with an in-state reverse-mode AD tape. This is verified by running
programs three ways and comparing results:

1. **Reference VM** — C `switch(opcode)` interpreter
2. **Simulated transformer** — C functions mirroring the weight matrices
3. **Matrix forward pass** — actual `W @ state + b` matmul

All three produce identical results. The current repository artifact verifies
126/126 inline programs and 123/123 traced programs.

---

## The State Vector (d_model = 256)

The VM's bounded runtime state fits in 256 floats:

| Dimensions | Content |
|---|---|
| 0-15 | Persistent VM state: PC, stack top values, output, halt, locals, SP/FP, closure |
| 16-31 | Per-cycle transients: opcode, operand, products, loads, comparisons, dispatch flags |
| 32-35 | Type tags for TOS/SOS/R2/R3 |
| 36-47 | AD control state: tape length, cursor, mode, current node fields |
| 48-111 | AD tape storage: 8 nodes x 8 fields |
| 112-127 | AD/type-predicate transient indicators |
| 128-207 | Bounded arena bank: 16 cells x 5 fields |
| 208-255 | Arena operation transients for cons/list/vector/string/closure/continuation slices |

The arena bank is the important memory model point: stack values hold small
cell indices, not host pointers. There is no GC heap in the weight artifact.

---

## The 6 Layers

| Layer | Mechanism | What it does |
|---|---|---|
| 0 | Gaussian attention | Instruction fetch — reads opcode at PC position |
| 1 | Gated preprocessing | Address resolution, comparisons, type predicates, AD parent loads |
| 2 | Square-activation FFN | Product precompute — computes TOS * SOS and AD products |
| 3 | Gated FFN | Execution — opcode dispatch, arithmetic, control flow, arena reads/writes |
| 4 | Gated FFN | Tape and arena writeback, continuation/closure state, AD parent load |
| 5 | Gated FFN | AD gradient writeback and backward-pass updates |

### How OP_ADD Works in Weights

The gate neuron fires when `opcode == 7`. The value neurons compute
`TOS + SOS` and write the result back. This is pure `W @ state + b` —
no C code, no branching, just linear algebra.

82 of 83 canonical opcodes are weight-encoded in the current bounded artifact.
The only external boundary is `OP_NATIVE_CALL`.

---

## Two Tiers of Execution

**Tier 1: Weight-encoded opcodes** — arithmetic, comparisons, jumps, stack
manipulation, bounded arena memory, closures, continuations, type predicates,
and the verifier-covered reverse-mode AD tape. Computed entirely through matrix
multiplication.

**Tier 2: Native dispatch boundary (IDs 300+)** — `OP_NATIVE_CALL` carries a native
function ID. The transformer detects it and delegates to C:

| ID Range | Subsystem |
|---|---|
| 300-369 | Core (cons, car, cdr, display, I/O) |
| 370-409 | Automatic differentiation (dual numbers, AD tape) |
| 410-470 | Tensor operations (matmul, reshape, reduce) |
| 500-549 | Consciousness engine (logic, inference, workspace) |
| 800-803 | Model I/O (tensor/model checkpoint save-load) |
| 804-843 | Geometric manifolds (Riemannian operations) |

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
Header: magic "QLMW", version 3, d_model 256, n_layers 6
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
