# Eshkol Competitive Implementation Guide

## Outperforming JAX and Establishing Production Dominance

**Date:** December 7, 2025
**Classification:** Strategic Implementation Specification
**Objective:** Make Eshkol the definitively superior choice over JAX and all other ML systems

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [JAX Core Strengths Analysis](#2-jax-core-strengths-analysis)
3. [Phase 1: Matching JAX Feature Parity](#3-phase-1-matching-jax-feature-parity)
4. [Phase 2: Exceeding JAX Performance](#4-phase-2-exceeding-jax-performance)
5. [Phase 3: Unique Differentiators](#5-phase-3-unique-differentiators)
6. [Phase 4: Production Viability](#6-phase-4-production-viability)
7. [Implementation Specifications](#7-implementation-specifications)
8. [Benchmark Suite](#8-benchmark-suite)
9. [Migration Path](#9-migration-path)
10. [Success Criteria](#10-success-criteria)

---

## 1. Executive Summary

### What Eshkol Already Has (Existing Advantages)

**Eshkol's REPL already uses JIT and AOT compilation for real-time computing:**

| Capability | Implementation | Status |
|------------|----------------|--------|
| **JIT Compilation** | LLVM ORC LLJIT | ✅ Complete |
| **AOT Compilation** | LLVM TargetMachine → Native object files | ✅ Complete |
| **Incremental Compilation** | Symbol persistence across REPL evaluations | ✅ Complete |
| **Native Execution** | No interpreter, direct machine code | ✅ Complete |
| **Shared Memory Arena** | Real-time memory management | ✅ Complete |

```cpp
// REPL JIT (lib/repl/repl_jit.cpp)
auto jit_or_err = LLJITBuilder()
    .setNumCompileThreads(1)
    .create();
// Compiles Eshkol → LLVM IR → Native code in milliseconds

// AOT Compilation (lib/backend/llvm_codegen.cpp)
std::unique_ptr<TargetMachine> target_machine(
    target->createTargetMachine(target_triple, cpu_name, features, ...));
// Generates standalone native executables
```

**This means Eshkol already has zero Python overhead** - the REPL compiles to native code, not interpretation. This is a fundamental architectural advantage over JAX.

### Existing Autodiff Architecture (Deep Dive)

Eshkol's autodiff system is already fully implemented for scalar operations. Understanding its architecture is critical for extension to tensors.

**Dual Number Forward-Mode (`lib/backend/autodiff_codegen.cpp:33-201`)**

```cpp
// Dual number creation: (primal, tangent)
llvm::Value* AutodiffCodegen::createDualNumber(llvm::Value* primal, llvm::Value* tangent) {
    llvm::Value* dual_ptr = ctx_.builder().CreateAlloca(ctx_.dualNumberType());
    // Store primal in field 0
    llvm::Value* primal_ptr = ctx_.builder().CreateStructGEP(ctx_.dualNumberType(), dual_ptr, 0);
    ctx_.builder().CreateStore(primal, primal_ptr);
    // Store tangent in field 1
    llvm::Value* tangent_ptr = ctx_.builder().CreateStructGEP(ctx_.dualNumberType(), dual_ptr, 1);
    ctx_.builder().CreateStore(tangent, tangent_ptr);
    return ctx_.builder().CreateLoad(ctx_.dualNumberType(), dual_ptr);
}

// Product rule: (a, a') * (b, b') = (a*b, a'*b + a*b')
llvm::Value* AutodiffCodegen::dualMul(llvm::Value* dual_a, llvm::Value* dual_b) {
    llvm::Value* a = getDualPrimal(dual_a);
    llvm::Value* a_prime = getDualTangent(dual_a);
    llvm::Value* b = getDualPrimal(dual_b);
    llvm::Value* b_prime = getDualTangent(dual_b);

    llvm::Value* value = ctx_.builder().CreateFMul(a, b);
    llvm::Value* term1 = ctx_.builder().CreateFMul(a_prime, b);
    llvm::Value* term2 = ctx_.builder().CreateFMul(a, b_prime);
    llvm::Value* deriv = ctx_.builder().CreateFAdd(term1, term2);

    return createDualNumber(value, deriv);
}
```

**Reverse-Mode Backpropagation (`lib/backend/autodiff_codegen.cpp:1141-1250`)**

```cpp
void AutodiffCodegen::backpropagate(llvm::Value* tape, llvm::Value* output_node) {
    // Initialize output gradient = 1.0 (seed)
    storeNodeGradient(output_node, llvm::ConstantFP::get(ctx_.doubleType(), 1.0));

    // Get number of nodes in tape
    llvm::Value* num_nodes = ctx_.builder().CreateCall(get_count_func, {tape});

    // Iterate tape in REVERSE order (topological sort)
    // counter = num_nodes; while (counter > 0) { counter--; process(node[counter]); }
    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type());
    ctx_.builder().CreateStore(num_nodes, counter);

    // Loop: for each node in reverse order
    // ... (creates loop_cond, loop_body, loop_exit basic blocks)

    // For each node: load gradient, propagate to inputs based on operation type
    propagateGradient(node_ptr);
}

void AutodiffCodegen::propagateGradient(llvm::Value* node_ptr) {
    llvm::Value* node_type = ctx_.builder().CreateLoad(ctx_.int32Type(), type_ptr);
    llvm::Value* node_grad = loadNodeGradient(node_ptr);

    // Switch on operation type for chain rule application
    // AD_NODE_ADD (2): dL/dx = dL/dz, dL/dy = dL/dz
    // AD_NODE_MUL (4): dL/dx = dL/dz * y, dL/dy = dL/dz * x
    // AD_NODE_SIN (6): dL/dx = dL/dz * cos(x)
    // ...

    // MUL gradient propagation (product rule in reverse):
    ctx_.builder().SetInsertPoint(mul_block);
    llvm::Value* input1_val = loadNodeValue(input1);
    llvm::Value* input2_val = loadNodeValue(input2);
    llvm::Value* grad_input1 = ctx_.builder().CreateFMul(node_grad, input2_val);
    llvm::Value* grad_input2 = ctx_.builder().CreateFMul(node_grad, input1_val);
    accumulateGradient(input1, grad_input1);
    accumulateGradient(input2, grad_input2);
}
```

**Nested Gradient Support (`lib/backend/autodiff_codegen.cpp:1454-1550`)**

```cpp
// Push current tape to stack, switch to new tape
void AutodiffCodegen::pushTapeContext(llvm::Value* new_tape) {
    // Load current depth
    llvm::Value* depth = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_tape_depth);

    // Save current tape to stack[depth]
    llvm::Value* current_tape = ctx_.builder().CreateLoad(ctx_.ptrType(), current_ad_tape);
    llvm::Value* slot_ptr = ctx_.builder().CreateGEP(stack_type, ad_tape_stack,
        {llvm::ConstantInt::get(ctx_.int64Type(), 0), depth});
    ctx_.builder().CreateStore(current_tape, slot_ptr);

    // Increment depth and set new tape
    llvm::Value* new_depth = ctx_.builder().CreateAdd(depth, 1);
    ctx_.builder().CreateStore(new_depth, ad_tape_depth);
    ctx_.builder().CreateStore(new_tape, current_ad_tape);
    ctx_.builder().CreateStore(1, ad_mode_active);  // Enable AD mode
}
```

**Global AD State (shared across JIT modules)**

```cpp
// lib/core/arena_memory.cpp:24-47
ad_tape_t* __current_ad_tape = nullptr;           // Current active tape
bool __ad_mode_active = false;                     // AD mode flag
ad_tape_t* __ad_tape_stack[32] = {nullptr};       // Stack for nested gradients
uint64_t __ad_tape_depth = 0;                      // Current nesting depth
void* __inner_var_node_ptr = nullptr;              // For double backward
uint64_t __gradient_x_degree = 0;                  // Tracks derivative order
```

**Key Insight**: The architecture already supports:
- ✅ Forward-mode AD (dual numbers with chain rules)
- ✅ Reverse-mode AD (tape-based backpropagation)
- ✅ Nested gradients (32-level stack)
- ✅ Double backward (second derivatives)
- ✅ JIT symbol sharing (AD state persists across REPL evaluations)

**What's needed for tensor extension**:
- Extend `ad_node_t` with tensor data/gradient pointers
- Add tensor operation types (MATMUL, SOFTMAX, LAYERNORM, etc.)
- Implement tensor chain rules in `propagateGradient`
- Integrate with XLA for GPU/TPU execution

### The Goal

Transform Eshkol into the **unambiguously superior** choice for ML development by:

1. **Matching** JAX's composable transformations (`jit`, `grad`, `vmap`, `pmap`)
2. **Exceeding** JAX's performance through native compilation
3. **Differentiating** with capabilities no other system offers
4. **Proving** production viability through benchmarks and real-world deployment

### The Thesis

JAX succeeded because it made functional transformations composable. Eshkol can win by making those same transformations **type-safe**, **natively compiled**, and **formally verifiable** - while adding capabilities JAX fundamentally cannot have.

### Key Metrics for Success

| Metric | Target | Measurement |
|--------|--------|-------------|
| Shape error detection | 100% at compile time | Test suite |
| Small op latency | <50% of JAX | Benchmark |
| Large op throughput | ≥100% of JAX | Benchmark |
| Control flow support | 100% (no tracing limits) | Test suite |
| Compilation time | <2x JAX JIT | Benchmark |
| Memory efficiency | ≥100% of JAX | Profiler |

---

## 2. JAX Core Strengths Analysis

### 2.1 The Four Transformations

JAX's power comes from four composable function transformations:

#### 2.1.1 `jit` - Just-In-Time Compilation

```python
@jax.jit
def f(x):
    return x @ W + b

# First call: traces function, compiles via XLA
# Subsequent calls: runs compiled code
```

**What makes it good:**
- Transparent compilation
- Automatic optimization
- XLA's powerful compiler

**What's wrong with it:**
- Tracing overhead on first call
- Python dispatch on every call
- Shape must be known at trace time
- Control flow limitations

#### 2.1.2 `grad` - Automatic Differentiation

```python
def loss(params, x, y):
    return jnp.mean((model(params, x) - y) ** 2)

grads = jax.grad(loss)(params, x, y)
```

**What makes it good:**
- Reverse-mode AD
- Composes with other transformations
- Higher-order derivatives via nesting

**What's wrong with it:**
- Tracing-based (same limitations as jit)
- No compile-time shape checking
- Limited control flow in differentiated code

#### 2.1.3 `vmap` - Automatic Batching

```python
def single_example(x):
    return x @ W

batched = jax.vmap(single_example)
batched(xs)  # Automatically batched
```

**What makes it good:**
- Write single-example code, get batched execution
- Composable with grad (per-example gradients)
- Efficient implementation

**What's wrong with it:**
- Tracing limitations
- Can't express complex batching patterns easily
- No compile-time batch dimension checking

#### 2.1.4 `pmap` - Parallel Mapping

```python
@jax.pmap
def parallel_step(params, batch):
    grads = jax.grad(loss)(params, batch)
    return jax.lax.pmean(grads, axis_name='devices')

# Runs on all available devices
```

**What makes it good:**
- Simple data parallelism
- Collective operations built-in
- Scales to TPU pods

**What's wrong with it:**
- All devices must run same code
- Limited flexibility for model parallelism
- Python coordination overhead

### 2.2 JAX's Architectural Weaknesses

| Weakness | Impact | Eshkol Opportunity |
|----------|--------|-------------------|
| Python interpreter | Dispatch latency | Native compilation |
| Tracing model | Control flow limits | Source transformation |
| Dynamic typing | Runtime shape errors | Dependent types |
| No effect tracking | Hidden side effects | Effect system |
| No verification | Can't prove properties | HoTT foundations |
| Numerical only | No symbolic | Homoiconic design |

---

## 3. Phase 1: Matching JAX Feature Parity

### 3.1 Implementing `jit` Equivalent

#### 3.1.1 Design

Eshkol's JIT is fundamentally different from JAX's - we compile at definition time, not first call:

```scheme
;; Eshkol: Compiled when defined, not when first called
(define (f x)
  (+ (matmul x W) b))

;; Already compiled to native code via LLVM
;; No tracing, no first-call overhead
```

For XLA integration, we need lazy compilation to XLA when targeting GPU/TPU:

```scheme
;; Explicit XLA compilation for hardware acceleration
(define f-xla (xla/compile f :device "gpu:0"))

;; Or automatic based on context
(with-device "gpu:0"
  (f x))  ; Automatically uses XLA-compiled version
```

#### 3.1.2 Implementation Specification

**File:** `lib/backend/xla_compiler.cpp`

```cpp
class XLACompiler {
public:
    // Compile Eshkol function to XLA executable
    xla::XlaComputation compile(const eshkol_ast_t* func) {
        xla::XlaBuilder builder(func->name);

        // Lower AST to HLO
        std::vector<xla::XlaOp> params;
        for (size_t i = 0; i < func->param_count; i++) {
            auto shape = inferShape(func->params[i]);
            params.push_back(xla::Parameter(&builder, i, shape,
                                            func->params[i]->name));
        }

        // Recursively lower body
        xla::XlaOp result = lowerExpr(func->body, &builder, params);

        return builder.Build(result).value();
    }

    xla::XlaOp lowerExpr(const eshkol_ast_t* expr,
                         xla::XlaBuilder* builder,
                         const std::vector<xla::XlaOp>& env) {
        switch (expr->type) {
            case AST_LITERAL:
                return lowerLiteral(expr, builder);

            case AST_VAR_REF:
                return env[expr->var_index];

            case AST_BINARY_OP:
                return lowerBinaryOp(expr, builder, env);

            case AST_CALL:
                if (isBuiltinOp(expr->callee)) {
                    return lowerBuiltinCall(expr, builder, env);
                } else {
                    return lowerUserCall(expr, builder, env);
                }

            case AST_IF:
                return lowerConditional(expr, builder, env);

            case AST_LOOP:
                return lowerLoop(expr, builder, env);

            // ... more cases
        }
    }

    xla::XlaOp lowerBuiltinCall(const eshkol_ast_t* expr,
                                xla::XlaBuilder* builder,
                                const std::vector<xla::XlaOp>& env) {
        std::string op = expr->callee->name;

        if (op == "matmul") {
            auto lhs = lowerExpr(expr->args[0], builder, env);
            auto rhs = lowerExpr(expr->args[1], builder, env);

            xla::DotDimensionNumbers dims;
            dims.add_lhs_contracting_dimensions(-1);
            dims.add_rhs_contracting_dimensions(-2);

            return xla::DotGeneral(lhs, rhs, dims);
        }
        else if (op == "softmax") {
            auto input = lowerExpr(expr->args[0], builder, env);
            auto dim = expr->args[1]->int_value;

            // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
            auto max_val = xla::Reduce(input,
                xla::MinValue(builder, xla::F32),
                xla::CreateScalarMaxComputation(xla::F32, builder),
                {dim});
            auto shifted = xla::Sub(input, max_val);
            auto exp_shifted = xla::Exp(shifted);
            auto sum_exp = xla::Reduce(exp_shifted,
                xla::ConstantR0<float>(builder, 0.0f),
                xla::CreateScalarAddComputation(xla::F32, builder),
                {dim});
            return xla::Div(exp_shifted, sum_exp);
        }
        // ... more builtins
    }

    // Handle control flow - NOT tracing, actual conditionals
    xla::XlaOp lowerConditional(const eshkol_ast_t* expr,
                                xla::XlaBuilder* builder,
                                const std::vector<xla::XlaOp>& env) {
        auto pred = lowerExpr(expr->condition, builder, env);

        // Build true branch computation
        xla::XlaBuilder true_builder("true_branch");
        auto true_result = lowerExpr(expr->then_branch, &true_builder, env);
        auto true_comp = true_builder.Build(true_result).value();

        // Build false branch computation
        xla::XlaBuilder false_builder("false_branch");
        auto false_result = lowerExpr(expr->else_branch, &false_builder, env);
        auto false_comp = false_builder.Build(false_result).value();

        // XLA conditional - REAL control flow, not traced
        return xla::Conditional(pred, {}, true_comp, {}, false_comp);
    }

    // Handle loops - REAL loops, not unrolled traces
    xla::XlaOp lowerLoop(const eshkol_ast_t* expr,
                         xla::XlaBuilder* builder,
                         const std::vector<xla::XlaOp>& env) {
        // Build loop body computation
        xla::XlaBuilder body_builder("loop_body");
        // ... body lowering
        auto body_comp = body_builder.Build().value();

        // Build condition computation
        xla::XlaBuilder cond_builder("loop_cond");
        // ... condition lowering
        auto cond_comp = cond_builder.Build().value();

        // XLA while loop
        return xla::While(cond_comp, body_comp, initial_state);
    }
};
```

**File:** `lib/runtime/xla_runtime.cpp`

```cpp
class XLARuntime {
    std::unordered_map<std::string, std::unique_ptr<xla::LocalExecutable>> cache_;
    xla::LocalClient* client_;

public:
    XLARuntime() {
        // Initialize XLA client for available devices
        client_ = xla::ClientLibrary::LocalClientOrDie();
    }

    xla::Literal execute(const std::string& func_name,
                         const std::vector<xla::Literal>& args) {
        auto it = cache_.find(func_name);
        if (it == cache_.end()) {
            // Compile on first use (but NOT tracing - full compilation)
            auto computation = compiler_.compile(getFunction(func_name));
            auto executable = client_->Compile(computation, {}, {}).value();
            cache_[func_name] = std::move(executable);
            it = cache_.find(func_name);
        }

        // Execute
        std::vector<const xla::ShapedBuffer*> arg_buffers;
        for (const auto& arg : args) {
            arg_buffers.push_back(client_->LiteralToShapedBuffer(arg, 0).value());
        }

        auto result = it->second->Run(arg_buffers, {}).value();
        return client_->ShapedBufferToLiteral(result).value();
    }
};
```

#### 3.1.3 Key Differences from JAX

| Aspect | JAX `jit` | Eshkol JIT |
|--------|-----------|------------|
| When compiled | First call (tracing) | Definition time or first call (full compile) |
| Control flow | Traced (concrete values) | Real XLA conditionals/loops |
| Dispatch | Python → C++ → XLA | Native → XLA |
| Shape changes | Re-trace required | Type error or handled |
| Overhead | Python on every call | Zero after compilation |

### 3.2 Implementing `grad` Equivalent

#### 3.2.1 Design

Eshkol already has reverse-mode AD. The key improvements needed:

1. **Tensor-level operations** (matmul, softmax, etc.)
2. **XLA-aware AD** (gradients as XLA computations)
3. **Composability** with vmap/pmap

```scheme
;; Current Eshkol (scalar-level AD)
(gradient (lambda (x) (* x x)) 3.0)  ; => 6.0

;; Required (tensor-level AD with XLA)
(gradient (lambda (W) (sum (matmul x W))) W)  ; => Tensor gradient
```

#### 3.2.2 Implementation Specification

**File:** `lib/backend/tensor_autodiff.cpp`

```cpp
class TensorAutodiff {
public:
    // Transform function to compute gradients
    // Returns: (value, gradient_function)
    std::pair<xla::XlaComputation, xla::XlaComputation>
    valueAndGrad(const eshkol_ast_t* func, size_t wrt_arg) {

        // Forward pass computation
        xla::XlaBuilder fwd_builder(std::string(func->name) + "_fwd");
        auto fwd_result = buildForwardPass(func, &fwd_builder);
        auto fwd_comp = fwd_builder.Build(fwd_result.output).value();

        // Backward pass computation
        xla::XlaBuilder bwd_builder(std::string(func->name) + "_bwd");
        auto bwd_result = buildBackwardPass(func, fwd_result, &bwd_builder, wrt_arg);
        auto bwd_comp = bwd_builder.Build(bwd_result).value();

        return {fwd_comp, bwd_comp};
    }

private:
    struct ForwardResult {
        xla::XlaOp output;
        std::vector<xla::XlaOp> intermediates;  // For backward pass
        std::vector<ADNode> tape;               // Computation graph
    };

    struct ADNode {
        enum Type {
            INPUT, CONST,
            ADD, SUB, MUL, DIV,
            MATMUL, SOFTMAX, LAYERNORM, ATTENTION,
            EXP, LOG, SIN, COS, TANH,
            REDUCE_SUM, REDUCE_MEAN,
            RESHAPE, TRANSPOSE, BROADCAST
        };
        Type type;
        std::vector<size_t> inputs;  // Indices into tape
        xla::XlaOp value;            // Forward pass value
        xla::Shape shape;            // Output shape
        std::any op_data;            // Operation-specific data
    };

    ForwardResult buildForwardPass(const eshkol_ast_t* expr,
                                   xla::XlaBuilder* builder) {
        ForwardResult result;
        buildForwardRecursive(expr, builder, &result);
        return result;
    }

    size_t buildForwardRecursive(const eshkol_ast_t* expr,
                                 xla::XlaBuilder* builder,
                                 ForwardResult* result) {
        switch (expr->type) {
            case AST_VAR_REF: {
                ADNode node;
                node.type = ADNode::INPUT;
                node.value = /* get from env */;
                node.shape = inferShape(expr);
                result->tape.push_back(node);
                return result->tape.size() - 1;
            }

            case AST_CALL: {
                if (expr->callee->name == "matmul") {
                    size_t lhs_idx = buildForwardRecursive(expr->args[0], builder, result);
                    size_t rhs_idx = buildForwardRecursive(expr->args[1], builder, result);

                    auto lhs = result->tape[lhs_idx].value;
                    auto rhs = result->tape[rhs_idx].value;

                    xla::DotDimensionNumbers dims;
                    dims.add_lhs_contracting_dimensions(-1);
                    dims.add_rhs_contracting_dimensions(-2);

                    ADNode node;
                    node.type = ADNode::MATMUL;
                    node.inputs = {lhs_idx, rhs_idx};
                    node.value = xla::DotGeneral(lhs, rhs, dims);
                    node.shape = inferMatmulShape(result->tape[lhs_idx].shape,
                                                  result->tape[rhs_idx].shape);

                    result->tape.push_back(node);
                    return result->tape.size() - 1;
                }
                else if (expr->callee->name == "softmax") {
                    size_t input_idx = buildForwardRecursive(expr->args[0], builder, result);
                    int dim = expr->args[1]->int_value;

                    auto input = result->tape[input_idx].value;

                    // Forward: softmax
                    auto max_val = xla::Reduce(input, /*...*/);
                    auto shifted = xla::Sub(input, max_val);
                    auto exp_shifted = xla::Exp(shifted);
                    auto sum_exp = xla::Reduce(exp_shifted, /*...*/);
                    auto output = xla::Div(exp_shifted, sum_exp);

                    ADNode node;
                    node.type = ADNode::SOFTMAX;
                    node.inputs = {input_idx};
                    node.value = output;
                    node.shape = result->tape[input_idx].shape;
                    node.op_data = dim;  // Store dim for backward

                    // Cache output for backward pass
                    result->intermediates.push_back(output);

                    result->tape.push_back(node);
                    return result->tape.size() - 1;
                }
                // ... more operations
            }
        }
    }

    xla::XlaOp buildBackwardPass(const eshkol_ast_t* func,
                                 const ForwardResult& fwd,
                                 xla::XlaBuilder* builder,
                                 size_t wrt_arg) {
        size_t n = fwd.tape.size();
        std::vector<xla::XlaOp> grads(n);

        // Seed gradient = 1 for output
        grads[n - 1] = xla::ConstantR0<float>(builder, 1.0f);
        // Or ones_like for tensor output

        // Backward pass through tape
        for (size_t i = n - 1; i > 0; i--) {
            const ADNode& node = fwd.tape[i];
            xla::XlaOp grad = grads[i];

            if (!grad.valid()) continue;  // No gradient flows here

            switch (node.type) {
                case ADNode::MATMUL: {
                    // C = A @ B
                    // dA = dC @ B^T
                    // dB = A^T @ dC
                    size_t a_idx = node.inputs[0];
                    size_t b_idx = node.inputs[1];

                    auto A = fwd.tape[a_idx].value;
                    auto B = fwd.tape[b_idx].value;

                    // dA = dC @ B^T
                    auto B_T = xla::Transpose(B, {-1, -2});
                    auto grad_A = xla::DotGeneral(grad, B_T, /*dims*/);
                    accumulateGrad(&grads[a_idx], grad_A, builder);

                    // dB = A^T @ dC
                    auto A_T = xla::Transpose(A, {-1, -2});
                    auto grad_B = xla::DotGeneral(A_T, grad, /*dims*/);
                    accumulateGrad(&grads[b_idx], grad_B, builder);
                    break;
                }

                case ADNode::SOFTMAX: {
                    // y = softmax(x)
                    // dx = y * (dy - sum(dy * y, dim))
                    size_t x_idx = node.inputs[0];
                    int dim = std::any_cast<int>(node.op_data);

                    auto y = node.value;  // Cached softmax output
                    auto dy = grad;

                    auto dy_y = xla::Mul(dy, y);
                    auto sum_dy_y = xla::Reduce(dy_y, /*sum over dim*/);
                    auto grad_x = xla::Mul(y, xla::Sub(dy, sum_dy_y));

                    accumulateGrad(&grads[x_idx], grad_x, builder);
                    break;
                }

                case ADNode::LAYERNORM: {
                    // Complex gradient - see implementation
                    layernormBackward(node, grad, fwd, &grads, builder);
                    break;
                }

                // ... more operations
            }
        }

        // Return gradient w.r.t. requested argument
        return grads[wrt_arg];
    }

    void accumulateGrad(xla::XlaOp* existing, xla::XlaOp new_grad,
                        xla::XlaBuilder* builder) {
        if (!existing->valid()) {
            *existing = new_grad;
        } else {
            *existing = xla::Add(*existing, new_grad);
        }
    }
};
```

#### 3.2.3 Key Differences from JAX

| Aspect | JAX `grad` | Eshkol `gradient` |
|--------|------------|-------------------|
| Mechanism | Tracing + transform | Source-to-source + XLA |
| Control flow | Limited (traced) | Full support |
| Type checking | None | Compile-time shape verification |
| Higher-order | Nested tracing | Native composition |
| Custom rules | `custom_vjp` decorator | `defvjp` macro |

### 3.3 Implementing `vmap` Equivalent

#### 3.3.1 Design

`vmap` transforms a function over single examples to work on batches:

```scheme
;; Single example function
(define (predict x)
  (softmax (matmul x W)))

;; Batched version
(define predict-batch (vmap predict))

;; Type transformation:
;; predict : (Tensor f32 [d]) -> (Tensor f32 [c])
;; predict-batch : (Tensor f32 [batch d]) -> (Tensor f32 [batch c])
```

#### 3.3.2 Implementation Specification

**File:** `lib/transforms/vmap.cpp`

```cpp
class VmapTransform {
public:
    // Transform function to batched version
    eshkol_ast_t* transform(const eshkol_ast_t* func,
                            const std::vector<int>& in_axes,
                            int out_axis) {
        // Create new function with batched signature
        auto batched_func = cloneAST(func);

        // Transform parameter types
        for (size_t i = 0; i < func->param_count; i++) {
            if (in_axes[i] >= 0) {
                batched_func->params[i]->type =
                    addBatchDim(func->params[i]->type, in_axes[i]);
            }
        }

        // Transform body
        VmapContext ctx{in_axes, out_axis};
        batched_func->body = transformExpr(func->body, &ctx);

        // Transform return type
        batched_func->return_type = addBatchDim(func->return_type, out_axis);

        return batched_func;
    }

private:
    struct VmapContext {
        std::vector<int> in_axes;
        int out_axis;
        std::unordered_map<std::string, int> var_batch_dims;
    };

    eshkol_ast_t* transformExpr(const eshkol_ast_t* expr, VmapContext* ctx) {
        switch (expr->type) {
            case AST_VAR_REF: {
                // Variable reference - track batch dimension
                auto it = ctx->var_batch_dims.find(expr->name);
                if (it != ctx->var_batch_dims.end()) {
                    // This variable has a batch dimension
                    auto result = cloneAST(expr);
                    result->batch_dim = it->second;
                    return result;
                }
                return cloneAST(expr);
            }

            case AST_CALL: {
                if (expr->callee->name == "matmul") {
                    return transformMatmul(expr, ctx);
                }
                else if (expr->callee->name == "softmax") {
                    return transformSoftmax(expr, ctx);
                }
                else if (isElementwise(expr->callee->name)) {
                    return transformElementwise(expr, ctx);
                }
                else if (isReduction(expr->callee->name)) {
                    return transformReduction(expr, ctx);
                }
                else {
                    // User function - recursively vmap it
                    return transformUserCall(expr, ctx);
                }
            }

            case AST_LAMBDA: {
                // Lambda inside vmap - close over batch dimension
                return transformLambda(expr, ctx);
            }

            // ... more cases
        }
    }

    eshkol_ast_t* transformMatmul(const eshkol_ast_t* expr, VmapContext* ctx) {
        auto lhs = transformExpr(expr->args[0], ctx);
        auto rhs = transformExpr(expr->args[1], ctx);

        int lhs_batch = getBatchDim(lhs);
        int rhs_batch = getBatchDim(rhs);

        if (lhs_batch >= 0 && rhs_batch >= 0) {
            // Both batched: batched matmul
            // (batch, m, k) @ (batch, k, n) -> (batch, m, n)
            return makeBatchedMatmul(lhs, rhs, lhs_batch);
        }
        else if (lhs_batch >= 0) {
            // LHS batched: broadcast RHS
            // (batch, m, k) @ (k, n) -> (batch, m, n)
            return makeBroadcastMatmul(lhs, rhs, lhs_batch, /*rhs_broadcast=*/true);
        }
        else if (rhs_batch >= 0) {
            // RHS batched: broadcast LHS
            // (m, k) @ (batch, k, n) -> (batch, m, n)
            return makeBroadcastMatmul(lhs, rhs, rhs_batch, /*lhs_broadcast=*/true);
        }
        else {
            // Neither batched: normal matmul
            return makeCall("matmul", {lhs, rhs});
        }
    }

    eshkol_ast_t* transformSoftmax(const eshkol_ast_t* expr, VmapContext* ctx) {
        auto input = transformExpr(expr->args[0], ctx);
        int dim = expr->args[1]->int_value;
        int batch_dim = getBatchDim(input);

        if (batch_dim >= 0) {
            // Adjust softmax dim to account for batch dimension
            int adjusted_dim = (dim >= batch_dim) ? dim + 1 : dim;
            return makeCall("softmax", {input, makeInt(adjusted_dim)});
        }
        else {
            return makeCall("softmax", {input, makeInt(dim)});
        }
    }

    eshkol_ast_t* transformReduction(const eshkol_ast_t* expr, VmapContext* ctx) {
        auto input = transformExpr(expr->args[0], ctx);
        int reduce_dim = expr->args[1]->int_value;
        int batch_dim = getBatchDim(input);

        if (batch_dim >= 0) {
            // Reduce over non-batch dimension
            // Adjust dimension index
            int adjusted_dim = (reduce_dim >= batch_dim) ? reduce_dim + 1 : reduce_dim;

            // Result keeps batch dimension
            auto result = makeCall(expr->callee->name, {input, makeInt(adjusted_dim)});
            result->batch_dim = batch_dim;
            return result;
        }
        else {
            return makeCall(expr->callee->name, {input, makeInt(reduce_dim)});
        }
    }
};
```

**File:** `lib/backend/xla_vmap.cpp`

```cpp
// XLA-level vmap for maximum performance
class XLAVmap {
public:
    xla::XlaComputation vmapComputation(const xla::XlaComputation& comp,
                                        const std::vector<int>& in_axes,
                                        int out_axis,
                                        int batch_size) {
        // Use XLA's native batching where possible
        xla::XlaBuilder builder("vmapped_" + comp.name());

        // For simple cases, use xla::Map
        if (canUseXlaMap(comp, in_axes)) {
            return buildXlaMap(comp, in_axes, out_axis, batch_size, &builder);
        }

        // For complex cases, transform the HLO directly
        return transformHLO(comp, in_axes, out_axis, batch_size);
    }

private:
    xla::XlaComputation buildXlaMap(const xla::XlaComputation& comp,
                                    const std::vector<int>& in_axes,
                                    int out_axis,
                                    int batch_size,
                                    xla::XlaBuilder* builder) {
        // Build input parameters with batch dimension
        std::vector<xla::XlaOp> params;
        // ...

        // Use xla::Map for parallel execution
        auto mapped = xla::Map(builder, params, comp, {0});  // Map over dim 0

        return builder->Build(mapped).value();
    }
};
```

#### 3.3.3 Composability

```scheme
;; vmap composes with grad
(define per-example-grads
  (vmap (lambda (x y)
          (gradient (lambda (p) (loss (model p x) y)) params))
        :in-axes [0 0]    ; Batch over examples
        :out-axes 0))     ; Stack gradients

;; vmap composes with itself (nested batching)
(define doubly-batched
  (vmap (vmap f :in-axes 0) :in-axes 1))

;; vmap composes with pmap
(define distributed-batched
  (pmap (vmap f)))
```

### 3.4 Implementing `pmap` Equivalent

#### 3.4.1 Design

```scheme
;; Data parallelism across devices
(define (parallel-train-step params batches)
  (pmap (lambda (batch)
          (let ([grads (gradient loss params batch)])
            (pmean grads)))  ; All-reduce
        batches
        :devices (available-devices)))
```

#### 3.4.2 Implementation Specification

**File:** `lib/transforms/pmap.cpp`

```cpp
class PmapTransform {
public:
    // Transform function for parallel execution
    eshkol_ast_t* transform(const eshkol_ast_t* func,
                            const std::vector<std::string>& devices) {
        auto parallel_func = cloneAST(func);

        // Add device placement annotations
        parallel_func->annotations["devices"] = devices;
        parallel_func->annotations["parallel"] = true;

        // Transform collective operations
        parallel_func->body = transformCollectives(func->body);

        return parallel_func;
    }

private:
    eshkol_ast_t* transformCollectives(const eshkol_ast_t* expr) {
        switch (expr->type) {
            case AST_CALL: {
                if (expr->callee->name == "pmean") {
                    // Transform to XLA all-reduce
                    return makeXlaAllReduce(expr->args[0], "mean");
                }
                else if (expr->callee->name == "psum") {
                    return makeXlaAllReduce(expr->args[0], "sum");
                }
                else if (expr->callee->name == "pmax") {
                    return makeXlaAllReduce(expr->args[0], "max");
                }
                else if (expr->callee->name == "all-gather") {
                    return makeXlaAllGather(expr->args[0]);
                }
                else if (expr->callee->name == "all-to-all") {
                    return makeXlaAllToAll(expr->args[0], expr->args[1]->int_value);
                }
                // ... recurse for other calls
            }
            // ... other cases
        }
    }
};
```

**File:** `lib/runtime/distributed_runtime.cpp`

```cpp
class DistributedRuntime {
    std::vector<xla::LocalClient*> device_clients_;

public:
    DistributedRuntime(const std::vector<std::string>& devices) {
        for (const auto& device : devices) {
            device_clients_.push_back(getClientForDevice(device));
        }
    }

    std::vector<xla::Literal> executePmap(
            const xla::XlaComputation& comp,
            const std::vector<std::vector<xla::Literal>>& inputs) {

        // Launch on all devices in parallel
        std::vector<std::future<xla::Literal>> futures;

        for (size_t i = 0; i < device_clients_.size(); i++) {
            futures.push_back(std::async(std::launch::async, [&, i]() {
                return device_clients_[i]->Execute(comp, inputs[i]);
            }));
        }

        // Gather results
        std::vector<xla::Literal> results;
        for (auto& future : futures) {
            results.push_back(future.get());
        }

        return results;
    }
};
```

---

## 4. Phase 2: Exceeding JAX Performance

### 4.1 Zero Python Overhead

**The Problem:** JAX has Python dispatch overhead on every call, even for JIT-compiled functions.

**The Solution:** Eshkol compiles to native code with zero interpreter overhead.

#### 4.1.1 Performance Comparison

```
JAX call path:
Python → PyBind11 → C++ → XLA Dispatch → Kernel Execution
        ↑
        ~1-10μs overhead per call

Eshkol call path:
Native Code → XLA Dispatch → Kernel Execution
             ↑
             ~0.1-0.5μs overhead per call
```

#### 4.1.2 Implementation

**File:** `lib/runtime/native_dispatch.cpp`

```cpp
// Direct native dispatch with minimal overhead
class NativeDispatch {
    // Pre-compiled function pointers
    std::unordered_map<std::string, void*> compiled_functions_;

    // XLA executable cache
    std::unordered_map<std::string, xla::LocalExecutable*> xla_cache_;

public:
    // Call compiled function directly
    template<typename Ret, typename... Args>
    Ret call(const std::string& func_name, Args... args) {
        auto fn = reinterpret_cast<Ret(*)(Args...)>(compiled_functions_[func_name]);
        return fn(args...);  // Direct function pointer call
    }

    // Call XLA computation with minimal dispatch
    xla::Literal callXLA(const std::string& func_name,
                         const std::vector<xla::Literal>& args) {
        auto exec = xla_cache_[func_name];

        // Direct execution - no Python in the path
        std::vector<const xla::ShapedBuffer*> buffers;
        for (const auto& arg : args) {
            buffers.push_back(literalToBuffer(arg));
        }

        return exec->Run(buffers).value();
    }
};
```

#### 4.1.3 Benchmark Target

| Operation | JAX Latency | Eshkol Target | Speedup |
|-----------|-------------|---------------|---------|
| Small matmul (32x32) | 15μs | 5μs | 3x |
| Softmax (1024) | 12μs | 4μs | 3x |
| Full forward pass | 100μs | 40μs | 2.5x |
| Gradient computation | 200μs | 80μs | 2.5x |

### 4.2 No Tracing Overhead

**The Problem:** JAX traces functions on first call, which can be slow for complex functions.

**The Solution:** Eshkol compiles directly from source, no tracing needed.

#### 4.2.1 Implementation

```cpp
// Source-to-XLA compilation (no tracing)
class DirectCompiler {
public:
    xla::XlaComputation compile(const eshkol_ast_t* func) {
        // Direct AST → XLA translation
        // No Python tracing
        // No concrete value propagation
        // Full support for control flow

        xla::XlaBuilder builder(func->name);

        // Build parameters
        auto params = buildParameters(func, &builder);

        // Compile body directly
        auto result = compileExpr(func->body, &builder, params);

        return builder.Build(result).value();
    }

private:
    xla::XlaOp compileExpr(const eshkol_ast_t* expr,
                           xla::XlaBuilder* builder,
                           const Environment& env) {
        // Direct compilation - NOT tracing
        // Control flow becomes real XLA control flow
        // No limitations on what values can be

        switch (expr->type) {
            case AST_IF:
                // Real conditional, not traced
                return compileConditional(expr, builder, env);

            case AST_LOOP:
                // Real loop, not unrolled
                return compileLoop(expr, builder, env);

            // ...
        }
    }
};
```

#### 4.2.2 Control Flow Comparison

**JAX (tracing limitations):**
```python
@jit
def f(x, n):
    result = 0
    for i in range(n):  # ERROR: n must be concrete at trace time!
        result += x[i]
    return result
```

**Eshkol (no limitations):**
```scheme
(define (f x n)
  (for/fold ([result 0]) ([i (range n)])  ; n can be dynamic!
    (+ result (vref x i))))

;; Compiles to real XLA while loop
;; No tracing limitations
```

### 4.3 Compile-Time Optimization

**The Problem:** JAX optimizes at trace time with limited information.

**The Solution:** Eshkol has full type information at compile time for better optimization.

#### 4.3.1 Shape-Based Optimization

```scheme
;; Eshkol knows shapes at compile time
(define (f [x : (Tensor f32 [batch 784])]
           [W1 : (Tensor f32 [784 256])]
           [W2 : (Tensor f32 [256 10])])
  (matmul (relu (matmul x W1)) W2))

;; Compiler can:
;; 1. Choose optimal matmul algorithm based on shapes
;; 2. Fuse operations knowing intermediate shapes
;; 3. Allocate exact memory needed
;; 4. Generate specialized kernels
```

#### 4.3.2 Implementation

**File:** `lib/optimizer/shape_optimizer.cpp`

```cpp
class ShapeOptimizer {
public:
    eshkol_ast_t* optimize(const eshkol_ast_t* func) {
        // All shapes known at compile time
        auto shapes = inferAllShapes(func);

        // Choose optimal algorithms
        auto with_algorithms = selectAlgorithms(func, shapes);

        // Fuse operations
        auto fused = fuseOperations(with_algorithms, shapes);

        // Generate specialized code
        auto specialized = specializeForShapes(fused, shapes);

        return specialized;
    }

private:
    eshkol_ast_t* selectAlgorithms(const eshkol_ast_t* func,
                                   const ShapeMap& shapes) {
        // For matmul, choose between:
        // - Direct multiplication (small matrices)
        // - Tiled multiplication (medium matrices)
        // - cuBLAS/oneDNN (large matrices)

        for (auto& node : func->nodes) {
            if (node->type == AST_CALL && node->callee->name == "matmul") {
                auto lhs_shape = shapes[node->args[0]];
                auto rhs_shape = shapes[node->args[1]];

                size_t m = lhs_shape[0], k = lhs_shape[1], n = rhs_shape[1];

                if (m * k * n < 1000) {
                    node->algorithm = "direct";
                } else if (m * k * n < 100000) {
                    node->algorithm = "tiled";
                } else {
                    node->algorithm = "blas";
                }
            }
        }
    }

    eshkol_ast_t* fuseOperations(const eshkol_ast_t* func,
                                 const ShapeMap& shapes) {
        // Fuse patterns:
        // - matmul + bias → fused_matmul_bias
        // - matmul + relu → fused_matmul_relu
        // - layernorm components → single layernorm
        // - softmax + matmul → flash_attention pattern

        return FusionPass(shapes).run(func);
    }
};
```

---

## 5. Phase 3: Unique Differentiators

### 5.1 Compile-Time Shape Checking

**What JAX cannot do:**

```python
# JAX: This crashes at runtime
x = jnp.zeros((32, 784))
W = jnp.zeros((100, 256))  # Wrong shape!
jnp.dot(x, W)  # RuntimeError: shapes (32,784) and (100,256) not aligned
```

**What Eshkol does:**

```scheme
;; Eshkol: Compile-time error with helpful message
(define x : (Tensor f32 [32 784]))
(define W : (Tensor f32 [100 256]))
(matmul x W)

;; Compiler error:
;; Error at line 3: Shape mismatch in matmul
;;   Left operand:  (Tensor f32 [32 784])
;;   Right operand: (Tensor f32 [100 256])
;;   Cannot contract dimension 784 with 100
;;
;;   Did you mean to use a weight matrix of shape [784 256]?
```

#### 5.1.1 Implementation

**File:** `lib/types/tensor_types.cpp`

```cpp
class TensorTypeChecker {
public:
    TypeCheckResult checkMatmul(const Type* lhs, const Type* rhs) {
        auto lhs_tensor = dynamic_cast<const TensorType*>(lhs);
        auto rhs_tensor = dynamic_cast<const TensorType*>(rhs);

        if (!lhs_tensor || !rhs_tensor) {
            return error("matmul requires tensor arguments");
        }

        auto lhs_shape = lhs_tensor->shape();
        auto rhs_shape = rhs_tensor->shape();

        if (lhs_shape.size() < 1 || rhs_shape.size() < 1) {
            return error("matmul requires at least 1D tensors");
        }

        size_t lhs_k = lhs_shape.back();
        size_t rhs_k = rhs_shape[rhs_shape.size() - 2];

        if (!unify(lhs_k, rhs_k)) {
            return error(formatMatmulError(lhs_shape, rhs_shape, lhs_k, rhs_k));
        }

        // Compute result shape
        auto result_shape = computeMatmulShape(lhs_shape, rhs_shape);
        return ok(makeTensorType(lhs_tensor->dtype(), result_shape));
    }

private:
    std::string formatMatmulError(const Shape& lhs, const Shape& rhs,
                                  size_t lhs_k, size_t rhs_k) {
        std::stringstream ss;
        ss << "Shape mismatch in matmul\n";
        ss << "  Left operand:  (Tensor f32 " << formatShape(lhs) << ")\n";
        ss << "  Right operand: (Tensor f32 " << formatShape(rhs) << ")\n";
        ss << "  Cannot contract dimension " << lhs_k << " with " << rhs_k << "\n";

        // Suggest fix
        if (lhs_k != rhs_k) {
            Shape suggested = rhs;
            suggested[suggested.size() - 2] = lhs_k;
            ss << "\n  Did you mean to use a weight matrix of shape "
               << formatShape(suggested) << "?\n";
        }

        return ss.str();
    }
};
```

### 5.2 Effect System

**What JAX cannot track:**

```python
# JAX has no idea what effects this function has
def mystery(x):
    if random.random() > 0.5:  # Hidden randomness!
        return x * 2
    return x

# JAX just hopes it's pure for JIT/grad to work
```

**What Eshkol tracks:**

```scheme
;; Effect is in the type signature
(define (dropout x p)
  : {Random Gradient} (-> (Tensor f32 shape) Float (Tensor f32 shape))
  (if (sample (bernoulli (- 1 p)))
      (* x (/ 1 (- 1 p)))
      (zeros-like x)))

;; Compiler enforces:
;; - Can't use in pure context without handler
;; - Can't accidentally forget randomness
;; - Gradient computation knows about effects
```

#### 5.2.1 Implementation

**File:** `lib/types/effects.cpp`

```cpp
class EffectSystem {
public:
    // Built-in effects
    static const Effect Random;      // Uses random sampling
    static const Effect Gradient;    // Needs gradient tracking
    static const Effect IO;          // Performs I/O
    static const Effect State;       // Mutates state
    static const Effect Device;      // Device placement

    // Check effect compatibility
    TypeCheckResult checkEffects(const EffectRow& required,
                                 const EffectRow& provided) {
        for (const auto& eff : required.effects) {
            if (!provided.contains(eff)) {
                return error(formatMissingEffect(eff, provided));
            }
        }
        return ok();
    }

    // Infer effects for expression
    EffectRow inferEffects(const eshkol_ast_t* expr) {
        switch (expr->type) {
            case AST_CALL:
                if (expr->callee->name == "sample") {
                    return EffectRow({Random});
                }
                else if (expr->callee->name == "gradient") {
                    return EffectRow({Gradient});
                }
                else {
                    // Combine effects of callee and arguments
                    auto callee_effects = getEffects(expr->callee);
                    auto arg_effects = combineEffects(expr->args);
                    return callee_effects.union_with(arg_effects);
                }

            // ... more cases
        }
    }
};

// Effect handlers
class EffectHandler {
public:
    // Handle Random effect
    eshkol_ast_t* handleRandom(const eshkol_ast_t* expr,
                                RandomMode mode) {
        switch (mode) {
            case RandomMode::Sample:
                // Normal random sampling
                return expr;

            case RandomMode::Deterministic:
                // Replace samples with expected values
                return transformToDeterministic(expr);

            case RandomMode::Reparameterized:
                // Reparameterization trick for gradients
                return transformToReparameterized(expr);
        }
    }
};
```

### 5.3 Formal Verification

**What JAX cannot prove:**

```python
# JAX cannot verify any properties about your code
# You just have to trust it works
```

**What Eshkol can prove:**

```scheme
;; Prove output is bounded
(define/contract (bounded-softmax x)
  #:requires (tensor? x)
  #:ensures (lambda (y) (and (all (>= y 0))
                              (all (<= y 1))
                              (≈ (sum y) 1.0)))
  (softmax x))

;; Prove Lipschitz constant
(define/lipschitz (safe-layer x)
  #:constant 1.0
  (matmul (spectral-normalize W) x))

;; Prove gradient correctness
(define/gradient-correct (my-softmax x)
  #:numerical-check #t
  #:tolerance 1e-5
  (/ (exp x) (sum (exp x))))
```

#### 5.3.1 Implementation

**File:** `lib/verification/contracts.cpp`

```cpp
class ContractVerifier {
public:
    VerificationResult verify(const eshkol_ast_t* func,
                              const Contract& contract) {
        // Static verification where possible
        if (auto static_result = tryStaticVerification(func, contract)) {
            return *static_result;
        }

        // Generate runtime checks
        auto checked_func = insertRuntimeChecks(func, contract);

        // Symbolic execution for path coverage
        auto paths = enumeratePaths(func);
        for (const auto& path : paths) {
            if (!verifyPath(path, contract)) {
                return VerificationResult::Failed(path);
            }
        }

        return VerificationResult::Verified(checked_func);
    }

private:
    std::optional<VerificationResult> tryStaticVerification(
            const eshkol_ast_t* func,
            const Contract& contract) {
        // Use SMT solver for simple contracts
        z3::context ctx;
        z3::solver solver(ctx);

        // Encode function as logical formula
        auto formula = encodeAsFormula(func, ctx);

        // Encode contract
        auto precondition = encodeContract(contract.requires, ctx);
        auto postcondition = encodeContract(contract.ensures, ctx);

        // Check: precondition ∧ function ⇒ postcondition
        solver.add(precondition);
        solver.add(formula);
        solver.add(!postcondition);  // Try to find counterexample

        if (solver.check() == z3::unsat) {
            return VerificationResult::StaticallyVerified();
        }

        return std::nullopt;  // Need runtime verification
    }
};
```

### 5.4 Symbolic Integration

**What JAX cannot do:**

```python
# JAX is purely numerical
# No symbolic differentiation
# No term rewriting
# No code generation
```

**What Eshkol can do:**

```scheme
;; Symbolic differentiation
(define expr '(+ (* x x) (* 2 x) 1))
(define deriv (symbolic-diff expr 'x))
;; => '(+ (* 2 x) 2)

;; Simplify
(define simplified (simplify deriv))
;; => '(+ (* 2 x) 2)

;; Compile to efficient code
(define f (eval `(lambda (x) ,simplified)))

;; Combine symbolic and numerical
(define (hybrid-gradient f x)
  (let ([symbolic-part (try-symbolic-diff f)]
        [numerical-part (gradient f x)])
    (if symbolic-part
        (eval symbolic-part x)  ; Use symbolic if possible
        numerical-part)))        ; Fall back to numerical

;; Neural-guided symbolic search
(define (learn-symbolic-function examples)
  (neural-symbolic-search
    (lambda (candidate)
      (let ([f (eval candidate)])
        (mse (map f (map first examples))
             (map second examples))))
    :grammar mathematical-expressions
    :max-depth 10))
```

#### 5.4.1 Implementation

**File:** `lib/symbolic/symbolic_diff.cpp`

```cpp
class SymbolicDifferentiator {
public:
    eshkol_ast_t* differentiate(const eshkol_ast_t* expr,
                                const std::string& var) {
        switch (expr->type) {
            case AST_LITERAL:
                return makeInt(0);

            case AST_VAR_REF:
                return (expr->name == var) ? makeInt(1) : makeInt(0);

            case AST_CALL: {
                std::string op = expr->callee->name;

                if (op == "+") {
                    // d/dx (a + b) = da/dx + db/dx
                    return makeAdd(differentiate(expr->args[0], var),
                                   differentiate(expr->args[1], var));
                }
                else if (op == "*") {
                    // d/dx (a * b) = a * db/dx + da/dx * b
                    auto a = expr->args[0];
                    auto b = expr->args[1];
                    return makeAdd(
                        makeMul(a, differentiate(b, var)),
                        makeMul(differentiate(a, var), b));
                }
                else if (op == "sin") {
                    // d/dx sin(a) = cos(a) * da/dx
                    auto a = expr->args[0];
                    return makeMul(makeCos(a), differentiate(a, var));
                }
                else if (op == "exp") {
                    // d/dx exp(a) = exp(a) * da/dx
                    auto a = expr->args[0];
                    return makeMul(makeExp(a), differentiate(a, var));
                }
                // ... more rules
            }
        }
    }
};

class Simplifier {
public:
    eshkol_ast_t* simplify(const eshkol_ast_t* expr) {
        // Apply rewrite rules until fixed point
        auto current = expr;
        while (true) {
            auto simplified = applyRules(current);
            if (equal(simplified, current)) break;
            current = simplified;
        }
        return current;
    }

private:
    eshkol_ast_t* applyRules(const eshkol_ast_t* expr) {
        // Algebraic rules
        // a + 0 → a
        // a * 1 → a
        // a * 0 → 0
        // a - a → 0
        // a / a → 1 (if a ≠ 0)
        // ...

        switch (expr->type) {
            case AST_CALL: {
                std::string op = expr->callee->name;
                auto a = simplify(expr->args[0]);

                if (op == "+" && expr->args.size() == 2) {
                    auto b = simplify(expr->args[1]);
                    if (isZero(a)) return b;
                    if (isZero(b)) return a;
                    if (equal(a, b)) return makeMul(makeInt(2), a);
                    return makeAdd(a, b);
                }
                // ... more rules
            }
        }
    }
};
```

---

## 6. Phase 4: Production Viability

### 6.1 Reliability Requirements

| Requirement | Implementation |
|-------------|----------------|
| No silent failures | Comprehensive error handling |
| Deterministic | Reproducible random seeds |
| Debuggable | Source maps, stack traces |
| Testable | Property-based testing |
| Monitorable | Metrics, logging |

#### 6.1.1 Error Handling

```scheme
;; Comprehensive error types
(define-type EshkolError
  (Union
    (ShapeError source expected actual)
    (TypeError source expected actual)
    (DeviceError device operation message)
    (NumericalError operation value message)
    (OutOfMemoryError device requested available)))

;; Errors include source location and context
;; Error at /path/to/file.esk:42:15
;;   in function 'train_step'
;;   in expression (matmul x W)
;;
;; ShapeError: Incompatible shapes for matmul
;;   Expected: [batch, 784] @ [784, 256]
;;   Actual:   [batch, 784] @ [100, 256]
;;
;; Context:
;;   x : (Tensor f32 [32 784])  ; defined at line 10
;;   W : (Tensor f32 [100 256]) ; defined at line 15
```

### 6.2 Performance Requirements

| Metric | Requirement | Measurement |
|--------|-------------|-------------|
| Compilation | <10s for 10K LOC | CI benchmark |
| Small ops | <10μs latency | Micro-benchmark |
| Large ops | Within 5% of cuBLAS | Benchmark suite |
| Memory | No leaks, predictable | Valgrind, profiler |
| Scaling | Linear to 1000 GPUs | Distributed tests |

### 6.3 Ecosystem Requirements

#### 6.3.1 Python Interoperability

```scheme
;; Import Python libraries
(import-python torch)
(import-python transformers :only [AutoModel])

;; Use Python objects
(define model (AutoModel.from_pretrained "bert-base"))

;; Export Eshkol to Python
(export-to-python my-model :name "EshkolModel")
```

```python
# From Python
import eshkol
model = eshkol.EshkolModel()
output = model(input)  # Calls Eshkol
grads = model.gradient(input)  # Uses Eshkol AD
```

#### 6.3.2 Model Serialization

```scheme
;; Save model
(save-model model "/path/to/model.eshkol"
  :format 'native      ; or 'onnx, 'safetensors
  :include-optimizer #t
  :include-metadata #t)

;; Load model
(define loaded (load-model "/path/to/model.eshkol"))

;; Export to ONNX
(export-onnx model "/path/to/model.onnx"
  :opset 17
  :dynamic-axes {"input" [0]})  ; Dynamic batch dimension
```

### 6.4 Documentation Requirements

| Component | Status | Priority |
|-----------|--------|----------|
| Language reference | Need | Critical |
| API documentation | Need | Critical |
| Tutorial (basics) | Need | Critical |
| Tutorial (AD) | Need | High |
| Tutorial (XLA) | Need | High |
| Migration guide (JAX) | Need | High |
| Migration guide (PyTorch) | Need | Medium |
| Performance guide | Need | Medium |

---

## 7. Implementation Specifications

### 7.1 Core Transformations API

```scheme
;; === JIT COMPILATION ===

;; Compile function for specific device
(define f-gpu (jit f :device "cuda:0"))

;; Compile with specific options
(define f-optimized (jit f
  :device "cuda:0"
  :precision 'mixed    ; bf16 compute, f32 accumulate
  :optimize #t         ; Enable all optimizations
  :debug #f))          ; Disable debug info

;; === AUTOMATIC DIFFERENTIATION ===

;; Basic gradient
(define grad-f (grad f))              ; Gradient of f w.r.t. first arg
(define grad-f (grad f :argnums 1))   ; Gradient w.r.t. second arg
(define grad-f (grad f :argnums [0 1])) ; Gradient w.r.t. multiple args

;; Value and gradient together
(define-values (value grads) (value-and-grad f x))

;; Higher-order derivatives
(define hess-f (hessian f))
(define jac-f (jacobian f))

;; Custom gradient rules
(defvjp my-op
  (lambda (primals tangents)
    ;; Custom backward pass
    ...))

;; === AUTOMATIC BATCHING ===

;; Basic vmap
(define batched-f (vmap f))

;; Specify axes
(define batched-f (vmap f
  :in-axes [0 #f]      ; Batch first arg, broadcast second
  :out-axis 0))        ; Output batch in first dimension

;; Nested vmap
(define doubly-batched (vmap (vmap f)))

;; === PARALLELISM ===

;; Data parallelism
(define parallel-f (pmap f :devices ["cuda:0" "cuda:1"]))

;; With collective operations
(define (parallel-step params batch)
  (pmap (lambda (local-batch)
          (let ([grads (grad loss params local-batch)])
            (pmean grads)))  ; All-reduce mean
        (shard batch :axis 0)))

;; === COMPOSITION ===

;; All transformations compose
(define per-example-grads
  (jit
    (vmap (grad loss :argnums 0))
    :device "cuda:0"))

(define distributed-training
  (pmap
    (jit
      (vmap (grad loss)))))
```

### 7.2 Type System API

```scheme
;; === TENSOR TYPES ===

;; Basic tensor types
(Tensor f32 [batch seq dim])    ; 3D float32 tensor
(Tensor bf16 [m n])             ; 2D bfloat16 tensor
(Tensor i32 [n])                ; 1D int32 tensor

;; Shape variables (dependent types)
(define (matmul [A : (Tensor f32 [m k])]
                [B : (Tensor f32 [k n])])
  : (Tensor f32 [m n])
  ...)

;; Shape constraints
(define (valid-attention [Q : (Tensor f32 [batch heads seq d_k])]
                         [K : (Tensor f32 [batch heads seq d_k])]  ; Same d_k!
                         [V : (Tensor f32 [batch heads seq d_v])])
  : (Tensor f32 [batch heads seq d_v])
  ...)

;; === EFFECT TYPES ===

;; Effect annotations
(define (dropout x p)
  : {Random} (-> (Tensor f32 s) Float (Tensor f32 s))
  ...)

(define (train-step params batch)
  : {Random Gradient Device} ...
  ...)

;; Effect handlers
(with-handler deterministic
  (dropout x 0.5))  ; Uses expected value instead of sampling

;; === CONTRACTS ===

(define/contract (bounded-output f x)
  #:requires (tensor? x)
  #:ensures (lambda (y) (all (<= (abs y) 1.0)))
  (tanh (f x)))
```

### 7.3 XLA Backend API

```scheme
;; === DEVICE MANAGEMENT ===

(available-devices)  ; => ["cpu:0" "cuda:0" "cuda:1"]
(current-device)     ; => "cuda:0"

(with-device "cuda:1"
  (f x))  ; Runs on cuda:1

;; === MEMORY MANAGEMENT ===

(device-memory-info "cuda:0")
;; => {:total 16GB :used 4GB :free 12GB}

(with-memory-limit (* 8 (expt 2 30))  ; 8GB
  (train model data))

;; === COMPILATION OPTIONS ===

(jit f
  :device "cuda:0"
  :precision 'mixed
  :memory-optimization 'recompute  ; or 'store
  :fusion #t
  :layout 'optimal)

;; === PROFILING ===

(with-profiling
  (train-step model batch))
;; => {:time 45ms
;;     :memory {:peak 2.3GB :allocated 1.8GB}
;;     :operations [{:name "matmul" :time 12ms :flops 1.2e9} ...]}
```

---

## 8. Benchmark Suite

### 8.1 Micro-Benchmarks

```scheme
;; Benchmark small operations
(benchmark "small_matmul"
  :warmup 100
  :iterations 1000
  :operation (lambda ()
               (matmul (randn [32 32]) (randn [32 32]))))

(benchmark "softmax"
  :warmup 100
  :iterations 1000
  :operation (lambda ()
               (softmax (randn [1024]) -1)))

(benchmark "gradient_small"
  :warmup 100
  :iterations 1000
  :operation (lambda ()
               (grad (lambda (x) (sum (square x))) (randn [100]))))
```

### 8.2 Model Benchmarks

```scheme
;; Benchmark full models
(benchmark "transformer_forward"
  :model (transformer-block :d-model 512 :n-heads 8)
  :input (randn [32 128 512])
  :warmup 10
  :iterations 100)

(benchmark "transformer_backward"
  :operation (lambda ()
               (grad transformer-loss params batch))
  :warmup 10
  :iterations 100)

(benchmark "training_step"
  :operation (lambda ()
               (train-step model batch optimizer))
  :warmup 10
  :iterations 100)
```

### 8.3 Comparison Framework

```scheme
;; Compare against JAX
(compare-benchmark "matmul_comparison"
  :eshkol (lambda () (matmul A B))
  :jax "jnp.dot(A, B)"
  :sizes [[32 32] [256 256] [1024 1024] [4096 4096]]
  :devices ["cpu" "cuda"])

;; Generate comparison report
(generate-benchmark-report
  :benchmarks [small-ops model-ops training-ops]
  :baselines ["jax" "pytorch"]
  :output "benchmark_report.html")
```

### 8.4 Target Metrics

| Benchmark | JAX | Eshkol Target | Notes |
|-----------|-----|---------------|-------|
| matmul 32x32 | 15μs | 5μs | Python overhead dominates |
| matmul 4096x4096 | 2.1ms | 2.0ms | Compute dominates |
| softmax 1024 | 12μs | 4μs | Python overhead |
| softmax 1M | 0.8ms | 0.8ms | Compute dominates |
| transformer fwd | 45ms | 40ms | Less dispatch overhead |
| transformer bwd | 95ms | 80ms | Better fusion |
| full train step | 150ms | 120ms | Cumulative gains |

---

## 9. Migration Path

### 9.1 From JAX

#### 9.1.1 Syntax Mapping

| JAX | Eshkol |
|-----|--------|
| `jax.jit` | `jit` |
| `jax.grad` | `grad` or `gradient` |
| `jax.vmap` | `vmap` |
| `jax.pmap` | `pmap` |
| `jnp.dot` | `matmul` |
| `jnp.sum` | `sum` |
| `jax.random.PRNGKey` | Implicit (effect system) |
| `jax.lax.scan` | `for/fold` |
| `jax.lax.cond` | `if` |

#### 9.1.2 Code Translation Example

**JAX:**
```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

@jax.jit
def train_step(params, x, y):
    def loss_fn(p):
        logits = model.apply(p, x)
        return jnp.mean((logits - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_map(lambda p, g: p - 0.01 * g, params, grads)
    return params, loss
```

**Eshkol:**
```scheme
(define-type MLPParams
  (Record [W1 : (Tensor f32 [784 256])]
          [b1 : (Tensor f32 [256])]
          [W2 : (Tensor f32 [256 10])]
          [b2 : (Tensor f32 [10])]))

(define (mlp params x)
  (let* ([h (relu (+ (matmul x params.W1) params.b1))]
         [out (+ (matmul h params.W2) params.b2)])
    out))

(define (train-step params x y)
  (let* ([loss-fn (lambda (p) (mean (square (- (mlp p x) y))))]
         [(loss grads) (value-and-grad loss-fn params)]
         [new-params (tree-map (lambda (p g) (- p (* 0.01 g)))
                               params grads)])
    (values new-params loss)))

;; JIT is automatic, or explicit:
(define train-step-jit (jit train-step :device "cuda:0"))
```

### 9.2 Migration Tools

```scheme
;; Automatic JAX to Eshkol translation
(define eshkol-code
  (translate-from-jax "
    @jax.jit
    def f(x):
        return jnp.sum(x ** 2)
  "))

;; Type inference from JAX shapes
(define typed-func
  (infer-types-from-jax f sample-input))

;; Gradual migration: call JAX from Eshkol
(define jax-model (import-jax-function "my_model" "module.py"))
(define output (jax-model input))  ; Calls JAX
(define grads (grad jax-model input))  ; Uses Eshkol AD
```

---

## 10. Success Criteria

### 10.1 Technical Metrics

| Metric | Target | Measurement | Timeline |
|--------|--------|-------------|----------|
| Shape errors at compile time | 100% | Test suite | Month 3 |
| Small op latency vs JAX | <50% | Benchmark | Month 4 |
| Large op throughput vs JAX | ≥100% | Benchmark | Month 4 |
| Control flow support | 100% | Test suite | Month 2 |
| Compilation time | <2x JAX JIT | Benchmark | Month 5 |
| Memory efficiency | ≥100% JAX | Profiler | Month 5 |

### 10.2 Adoption Metrics

| Metric | Target | Timeline |
|--------|--------|----------|
| GitHub stars | 1,000 | Month 6 |
| GitHub stars | 10,000 | Month 12 |
| Active contributors | 20 | Month 6 |
| Production users | 5 | Month 9 |
| Production users | 50 | Month 12 |
| Published papers using Eshkol | 10 | Month 12 |

### 10.3 Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test coverage | >90% | CI |
| Documentation coverage | 100% public API | CI |
| Zero known critical bugs | 0 | Issue tracker |
| Response time to issues | <48 hours | Issue tracker |

### 10.4 Competitive Position

**Success looks like:**

1. **Researchers choose Eshkol** for new projects because of type safety and verification
2. **Companies migrate from JAX** for performance and reliability
3. **JAX users acknowledge** Eshkol is better for production
4. **New ML frameworks** build on Eshkol instead of Python

**The ultimate validation:**
> "We rewrote our model in Eshkol and found 3 shape bugs that had been causing silent accuracy degradation in production for months. Training is now 20% faster and we can prove our model satisfies safety constraints."

---

## Appendix: Complete Implementation Timeline

### Month 1-2: Foundation
- [ ] XLA integration (basic ops)
- [ ] Tensor types with dependent shapes
- [ ] Shape checking at compile time
- [ ] Basic benchmark suite

### Month 3-4: Core Transformations
- [ ] `grad` for tensor operations
- [ ] `vmap` implementation
- [ ] `pmap` implementation
- [ ] Composability of transformations

### Month 5-6: Performance
- [ ] Zero-overhead dispatch
- [ ] Compilation optimization
- [ ] Memory optimization
- [ ] Full benchmark suite vs JAX

### Month 7-8: Differentiation
- [ ] Effect system
- [ ] Symbolic differentiation
- [ ] Symbolic-numerical integration
- [ ] Custom gradient rules

### Month 9-10: Verification
- [ ] Contract system
- [ ] Static verification (SMT)
- [ ] Runtime checking
- [ ] Gradient correctness verification

### Month 11-12: Ecosystem
- [ ] Python interoperability
- [ ] ONNX import/export
- [ ] Model hub integration
- [ ] Complete documentation

---

*This document serves as the definitive guide for making Eshkol the superior choice over JAX and all other ML systems. Every feature, implementation detail, and metric has been specified to enable focused, measurable progress toward production dominance.*
