# Eshkol Sequential Completion Roadmap

## Absolute Detail Implementation Guide

**Total Estimated Effort**: 95 engineering days
**Recommended Team Size**: 1-2 developers
**Start Date**: After v1.0-foundation release

---

## Table of Contents

1. [Phase 1: SIMD Vectorization](#phase-1-simd-vectorization-12-days)
2. [Phase 2: Parallel Execution](#phase-2-parallel-execution-11-days)
3. [Phase 3: Hygienic Macro System](#phase-3-hygienic-macro-system-16-days)
4. [Phase 4: GPU/CUDA Backend](#phase-4-gpucuda-backend-25-days)
5. [Phase 5: Neural Network Primitives](#phase-5-neural-network-primitives-15-days)
6. [Phase 6: OALR Memory Completion](#phase-6-oalr-memory-completion-5-days)
7. [Phase 7: Serialization & Checkpointing](#phase-7-serialization--checkpointing-5-days)
8. [Phase 8: Profiling Infrastructure](#phase-8-profiling-infrastructure-6-days)

---

## Phase 1: SIMD Vectorization (12 days)

### Why First?
- Immediate 4-8x performance improvement
- No external dependencies
- Foundation for GPU work (similar patterns)
- Required for competitive tensor operations

### Day 1-2: SIMD Detection Infrastructure

**Files to Create**:
```
inc/eshkol/backend/simd_config.h
lib/backend/simd_config.cpp
```

**Step 1.1**: Create `inc/eshkol/backend/simd_config.h`

```cpp
#ifndef ESHKOL_SIMD_CONFIG_H
#define ESHKOL_SIMD_CONFIG_H

#include <cstddef>

namespace eshkol {

enum class SIMDLevel {
    None = 0,      // Scalar fallback
    SSE42 = 1,     // 128-bit (2 doubles)
    AVX2 = 2,      // 256-bit (4 doubles)
    AVX512 = 3,    // 512-bit (8 doubles)
    NEON = 4       // ARM 128-bit
};

struct SIMDConfig {
    SIMDLevel level;
    unsigned vector_width;      // Number of doubles per vector register
    bool has_fma;               // Fused multiply-add
    bool has_gather_scatter;    // Gather/scatter operations
    bool has_masked_ops;        // AVX-512 masking

    // Runtime detection
    static SIMDConfig detect();

    // Get LLVM vector type width
    unsigned getLLVMVectorWidth() const { return vector_width; }

    // Human-readable name
    const char* getName() const;
};

// Global config (initialized once at startup)
extern SIMDConfig g_simd_config;

void initializeSIMD();

} // namespace eshkol

#endif
```

**Step 1.2**: Create `lib/backend/simd_config.cpp`

```cpp
#include "eshkol/backend/simd_config.h"
#include <llvm/TargetParser/Host.h>
#include <llvm/ADT/StringMap.h>

namespace eshkol {

SIMDConfig g_simd_config;

SIMDConfig SIMDConfig::detect() {
    SIMDConfig config;
    config.level = SIMDLevel::None;
    config.vector_width = 1;
    config.has_fma = false;
    config.has_gather_scatter = false;
    config.has_masked_ops = false;

    llvm::StringMap<bool> features;
    llvm::sys::getHostCPUFeatures(features);

    // Check from highest to lowest capability
    if (features["avx512f"] && features["avx512dq"]) {
        config.level = SIMDLevel::AVX512;
        config.vector_width = 8;
        config.has_fma = features["fma"];
        config.has_gather_scatter = true;
        config.has_masked_ops = true;
    }
    else if (features["avx2"]) {
        config.level = SIMDLevel::AVX2;
        config.vector_width = 4;
        config.has_fma = features["fma"];
        config.has_gather_scatter = true;
        config.has_masked_ops = false;
    }
    else if (features["sse4.2"]) {
        config.level = SIMDLevel::SSE42;
        config.vector_width = 2;
        config.has_fma = false;
        config.has_gather_scatter = false;
        config.has_masked_ops = false;
    }
    // ARM NEON detection would go here

    return config;
}

const char* SIMDConfig::getName() const {
    switch (level) {
        case SIMDLevel::AVX512: return "AVX-512";
        case SIMDLevel::AVX2: return "AVX2";
        case SIMDLevel::SSE42: return "SSE4.2";
        case SIMDLevel::NEON: return "NEON";
        default: return "Scalar";
    }
}

void initializeSIMD() {
    g_simd_config = SIMDConfig::detect();
}

} // namespace eshkol
```

**Step 1.3**: Integrate into startup

Modify `exe/eshkol-run.cpp`:
```cpp
#include "eshkol/backend/simd_config.h"

int main(int argc, char** argv) {
    eshkol::initializeSIMD();
    // ... rest of main
}
```

**Verification**: Build and run, print detected SIMD level

### Day 3-5: Core Vectorized Operations

**Files to Create**:
```
inc/eshkol/backend/vectorized_ops.h
lib/backend/vectorized_ops.cpp
```

**Step 1.4**: Create `inc/eshkol/backend/vectorized_ops.h`

```cpp
#ifndef ESHKOL_VECTORIZED_OPS_H
#define ESHKOL_VECTORIZED_OPS_H

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Module.h>
#include "eshkol/backend/simd_config.h"

namespace eshkol {

class CodegenContext;  // Forward declaration

class VectorizedOps {
public:
    VectorizedOps(CodegenContext& ctx);

    // ============ Element-wise Operations ============

    // c[i] = a[i] + b[i]
    void vectorAdd(llvm::Value* a_ptr, llvm::Value* b_ptr,
                   llvm::Value* c_ptr, llvm::Value* n);

    // c[i] = a[i] * b[i]
    void vectorMul(llvm::Value* a_ptr, llvm::Value* b_ptr,
                   llvm::Value* c_ptr, llvm::Value* n);

    // c[i] = a[i] * b[i] + c[i] (FMA)
    void vectorFMA(llvm::Value* a_ptr, llvm::Value* b_ptr,
                   llvm::Value* c_ptr, llvm::Value* n);

    // b[i] = scalar * a[i]
    void vectorScale(llvm::Value* a_ptr, llvm::Value* scalar,
                     llvm::Value* b_ptr, llvm::Value* n);

    // ============ Reduction Operations ============

    // return sum(a[0:n])
    llvm::Value* vectorSum(llvm::Value* a_ptr, llvm::Value* n);

    // return max(a[0:n])
    llvm::Value* vectorMax(llvm::Value* a_ptr, llvm::Value* n);

    // return sqrt(sum(a[i]^2))
    llvm::Value* vectorL2Norm(llvm::Value* a_ptr, llvm::Value* n);

    // ============ Dot Product ============

    // return sum(a[i] * b[i])
    llvm::Value* dotProduct(llvm::Value* a_ptr, llvm::Value* b_ptr, llvm::Value* n);

    // ============ Matrix Operations ============

    // C[M,N] = A[M,K] @ B[K,N]
    void matmul(llvm::Value* A, llvm::Value* B, llvm::Value* C,
                llvm::Value* M, llvm::Value* K, llvm::Value* N);

    // Tiled matmul for cache efficiency
    void matmulTiled(llvm::Value* A, llvm::Value* B, llvm::Value* C,
                     size_t M, size_t K, size_t N,
                     size_t tile_m = 64, size_t tile_n = 64, size_t tile_k = 256);

    // ============ Activation Functions (Vectorized) ============

    void vectorReLU(llvm::Value* ptr, llvm::Value* n);
    void vectorSigmoid(llvm::Value* ptr, llvm::Value* n);
    void vectorTanh(llvm::Value* ptr, llvm::Value* n);
    void vectorExp(llvm::Value* ptr, llvm::Value* n);

private:
    CodegenContext& ctx_;

    // Get vector type for current SIMD level
    llvm::VectorType* getVectorType();

    // Load vector from memory (aligned)
    llvm::Value* loadVector(llvm::Value* ptr, llvm::Value* idx);

    // Store vector to memory (aligned)
    void storeVector(llvm::Value* vec, llvm::Value* ptr, llvm::Value* idx);

    // Horizontal reduction (sum all elements of vector)
    llvm::Value* horizontalAdd(llvm::Value* vec);

    // Broadcast scalar to vector
    llvm::Value* broadcast(llvm::Value* scalar);

    // Generate loop: for i in [0, n, step]
    void generateLoop(llvm::Value* n, llvm::Value* step,
                      std::function<void(llvm::Value* i)> body);

    // Handle remainder elements (scalar loop for n % vector_width)
    void handleRemainder(llvm::Value* ptr, llvm::Value* start, llvm::Value* end,
                         std::function<llvm::Value*(llvm::Value*)> op);
};

} // namespace eshkol

#endif
```

**Step 1.5**: Implement `lib/backend/vectorized_ops.cpp`

```cpp
#include "eshkol/backend/vectorized_ops.h"
#include "eshkol/backend/codegen_context.h"
#include <llvm/IR/Intrinsics.h>

namespace eshkol {

VectorizedOps::VectorizedOps(CodegenContext& ctx) : ctx_(ctx) {}

llvm::VectorType* VectorizedOps::getVectorType() {
    return llvm::FixedVectorType::get(
        llvm::Type::getDoubleTy(ctx_.getLLVMContext()),
        g_simd_config.vector_width
    );
}

llvm::Value* VectorizedOps::broadcast(llvm::Value* scalar) {
    auto vec_type = getVectorType();
    llvm::Value* vec = llvm::UndefValue::get(vec_type);
    for (unsigned i = 0; i < g_simd_config.vector_width; i++) {
        vec = ctx_.getBuilder().CreateInsertElement(
            vec, scalar, ctx_.getBuilder().getInt32(i)
        );
    }
    return vec;
}

llvm::Value* VectorizedOps::loadVector(llvm::Value* ptr, llvm::Value* idx) {
    auto& builder = ctx_.getBuilder();
    auto double_type = llvm::Type::getDoubleTy(ctx_.getLLVMContext());
    auto vec_type = getVectorType();

    // Get pointer to element at idx
    auto elem_ptr = builder.CreateGEP(double_type, ptr, idx);

    // Cast to vector pointer
    auto vec_ptr_type = llvm::PointerType::getUnqual(vec_type);
    auto vec_ptr = builder.CreateBitCast(elem_ptr, vec_ptr_type);

    // Aligned load
    unsigned alignment = g_simd_config.vector_width * sizeof(double);
    return builder.CreateAlignedLoad(vec_type, vec_ptr, llvm::MaybeAlign(alignment));
}

void VectorizedOps::storeVector(llvm::Value* vec, llvm::Value* ptr, llvm::Value* idx) {
    auto& builder = ctx_.getBuilder();
    auto double_type = llvm::Type::getDoubleTy(ctx_.getLLVMContext());
    auto vec_type = getVectorType();

    auto elem_ptr = builder.CreateGEP(double_type, ptr, idx);
    auto vec_ptr_type = llvm::PointerType::getUnqual(vec_type);
    auto vec_ptr = builder.CreateBitCast(elem_ptr, vec_ptr_type);

    unsigned alignment = g_simd_config.vector_width * sizeof(double);
    builder.CreateAlignedStore(vec, vec_ptr, llvm::MaybeAlign(alignment));
}

llvm::Value* VectorizedOps::horizontalAdd(llvm::Value* vec) {
    auto& builder = ctx_.getBuilder();
    auto& context = ctx_.getLLVMContext();

    // Use LLVM's vector reduction intrinsic
    auto reduce_fadd = llvm::Intrinsic::getDeclaration(
        &ctx_.getModule(),
        llvm::Intrinsic::vector_reduce_fadd,
        {vec->getType()}
    );

    auto zero = llvm::ConstantFP::get(llvm::Type::getDoubleTy(context), 0.0);
    return builder.CreateCall(reduce_fadd, {zero, vec});
}

llvm::Value* VectorizedOps::dotProduct(llvm::Value* a_ptr, llvm::Value* b_ptr,
                                        llvm::Value* n) {
    auto& builder = ctx_.getBuilder();
    auto& context = ctx_.getLLVMContext();
    auto double_type = llvm::Type::getDoubleTy(context);
    auto int64_type = llvm::Type::getInt64Ty(context);
    auto vec_type = getVectorType();

    unsigned vec_width = g_simd_config.vector_width;
    auto vec_width_val = llvm::ConstantInt::get(int64_type, vec_width);

    // Calculate number of vector iterations
    auto num_vec_iters = builder.CreateUDiv(n, vec_width_val);

    // Create basic blocks
    auto func = builder.GetInsertBlock()->getParent();
    auto preheader = builder.GetInsertBlock();
    auto loop_header = llvm::BasicBlock::Create(context, "dot_loop", func);
    auto loop_body = llvm::BasicBlock::Create(context, "dot_body", func);
    auto loop_end = llvm::BasicBlock::Create(context, "dot_end", func);
    auto remainder = llvm::BasicBlock::Create(context, "dot_remainder", func);
    auto done = llvm::BasicBlock::Create(context, "dot_done", func);

    // Initialize accumulator
    auto zero_vec = llvm::ConstantVector::getSplat(
        llvm::ElementCount::getFixed(vec_width),
        llvm::ConstantFP::get(double_type, 0.0)
    );

    builder.CreateBr(loop_header);
    builder.SetInsertPoint(loop_header);

    // PHI nodes
    auto phi_i = builder.CreatePHI(int64_type, 2, "i");
    auto phi_acc = builder.CreatePHI(vec_type, 2, "acc");
    phi_i->addIncoming(llvm::ConstantInt::get(int64_type, 0), preheader);
    phi_acc->addIncoming(zero_vec, preheader);

    // Check if done with vector loop
    auto cmp = builder.CreateICmpULT(phi_i, num_vec_iters);
    builder.CreateCondBr(cmp, loop_body, loop_end);

    // Loop body
    builder.SetInsertPoint(loop_body);
    auto idx = builder.CreateMul(phi_i, vec_width_val);
    auto vec_a = loadVector(a_ptr, idx);
    auto vec_b = loadVector(b_ptr, idx);

    llvm::Value* prod;
    if (g_simd_config.has_fma) {
        // Use FMA: acc = a * b + acc
        auto fma = llvm::Intrinsic::getDeclaration(
            &ctx_.getModule(), llvm::Intrinsic::fma, {vec_type}
        );
        prod = builder.CreateCall(fma, {vec_a, vec_b, phi_acc});
    } else {
        auto mul = builder.CreateFMul(vec_a, vec_b);
        prod = builder.CreateFAdd(phi_acc, mul);
    }

    auto next_i = builder.CreateAdd(phi_i, llvm::ConstantInt::get(int64_type, 1));
    phi_i->addIncoming(next_i, loop_body);
    phi_acc->addIncoming(prod, loop_body);
    builder.CreateBr(loop_header);

    // After vector loop - horizontal sum
    builder.SetInsertPoint(loop_end);
    auto vec_sum = horizontalAdd(phi_acc);

    // Handle remainder
    auto remainder_start = builder.CreateMul(num_vec_iters, vec_width_val);
    auto has_remainder = builder.CreateICmpULT(remainder_start, n);
    builder.CreateCondBr(has_remainder, remainder, done);

    // Remainder scalar loop
    builder.SetInsertPoint(remainder);
    auto rem_header = llvm::BasicBlock::Create(context, "rem_header", func);
    auto rem_body = llvm::BasicBlock::Create(context, "rem_body", func);
    auto rem_end = llvm::BasicBlock::Create(context, "rem_end", func);

    builder.CreateBr(rem_header);
    builder.SetInsertPoint(rem_header);

    auto rem_i = builder.CreatePHI(int64_type, 2);
    auto rem_acc = builder.CreatePHI(double_type, 2);
    rem_i->addIncoming(remainder_start, remainder);
    rem_acc->addIncoming(vec_sum, remainder);

    auto rem_cmp = builder.CreateICmpULT(rem_i, n);
    builder.CreateCondBr(rem_cmp, rem_body, rem_end);

    builder.SetInsertPoint(rem_body);
    auto a_val = builder.CreateLoad(double_type,
        builder.CreateGEP(double_type, a_ptr, rem_i));
    auto b_val = builder.CreateLoad(double_type,
        builder.CreateGEP(double_type, b_ptr, rem_i));
    auto rem_prod = builder.CreateFMul(a_val, b_val);
    auto rem_sum = builder.CreateFAdd(rem_acc, rem_prod);
    auto rem_next_i = builder.CreateAdd(rem_i, llvm::ConstantInt::get(int64_type, 1));

    rem_i->addIncoming(rem_next_i, rem_body);
    rem_acc->addIncoming(rem_sum, rem_body);
    builder.CreateBr(rem_header);

    builder.SetInsertPoint(rem_end);
    builder.CreateBr(done);

    // Final result
    builder.SetInsertPoint(done);
    auto result = builder.CreatePHI(double_type, 2);
    result->addIncoming(vec_sum, loop_end);
    result->addIncoming(rem_acc, rem_end);

    return result;
}

// ... Continue with matmul, activation functions, etc.

} // namespace eshkol
```

### Day 6-8: Tiled Matrix Multiplication

**Step 1.6**: Add to `lib/backend/vectorized_ops.cpp`

```cpp
void VectorizedOps::matmulTiled(llvm::Value* A, llvm::Value* B, llvm::Value* C,
                                 size_t M, size_t K, size_t N,
                                 size_t tile_m, size_t tile_n, size_t tile_k) {
    auto& builder = ctx_.getBuilder();
    auto& context = ctx_.getLLVMContext();
    auto int64_type = llvm::Type::getInt64Ty(context);
    auto double_type = llvm::Type::getDoubleTy(context);

    unsigned vec_width = g_simd_config.vector_width;

    // Generate tiled loop nest:
    // for tile_i in range(0, M, tile_m):
    //   for tile_j in range(0, N, tile_n):
    //     for tile_k in range(0, K, tile_k):
    //       // Micro-kernel: SIMD on j dimension
    //       for i in range(tile_i, min(tile_i + tile_m, M)):
    //         for j in range(tile_j, min(tile_j + tile_n, N), vec_width):
    //           c_vec = load(C, i, j)
    //           for k in range(tile_k, min(tile_k + tile_k, K)):
    //             a_val = A[i, k]  // broadcast
    //             b_vec = load(B, k, j)
    //             c_vec = fma(a_val, b_vec, c_vec)
    //           store(c_vec, C, i, j)

    // [Full implementation would be ~200 lines]
    // Key optimization: register blocking, cache blocking, prefetching
}
```

### Day 9-10: Integrate with Tensor Codegen

**Step 1.7**: Modify `lib/backend/tensor_codegen.cpp`

```cpp
#include "eshkol/backend/vectorized_ops.h"

// In TensorCodegen::generateMatMul():
llvm::Value* TensorCodegen::generateMatMul(llvm::Value* A, llvm::Value* B,
                                            size_t M, size_t K, size_t N) {
    // Allocate result tensor
    auto C = allocateTensor(M, N);

    // Use SIMD if beneficial
    if (g_simd_config.level != SIMDLevel::None &&
        M >= 32 && K >= 32 && N >= 32) {
        VectorizedOps vec_ops(ctx_);
        vec_ops.matmulTiled(A, B, C, M, K, N);
    } else {
        // Scalar fallback
        generateMatMulScalar(A, B, C, M, K, N);
    }

    return C;
}

// In TensorCodegen::generateDotProduct():
llvm::Value* TensorCodegen::generateDotProduct(llvm::Value* a, llvm::Value* b,
                                                llvm::Value* n) {
    if (g_simd_config.level != SIMDLevel::None) {
        VectorizedOps vec_ops(ctx_);
        return vec_ops.dotProduct(a, b, n);
    }
    return generateDotProductScalar(a, b, n);
}
```

### Day 11-12: Testing and Benchmarking

**Step 1.8**: Create `tests/simd/vectorized_ops_test.esk`

```scheme
(require stdlib)

(display "=== SIMD Vectorization Tests ===")
(newline)

;; Test 1: Dot product
(define a (tensor 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0))
(define b (tensor 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0))
(display "Dot product: ")
(display (tensor-dot a b))
(display " (expected: 120.0)")
(newline)

;; Test 2: Matrix multiplication
(define A (tensor (1.0 2.0) (3.0 4.0)))
(define B (tensor (5.0 6.0) (7.0 8.0)))
(display "Matrix mul: ")
(display (tensor-matmul A B))
(display " (expected: ((19 22) (43 50)))")
(newline)

;; Test 3: Large matrix for benchmarking
(define (benchmark-matmul size)
  (define M (make-random-matrix size size))
  (define N (make-random-matrix size size))
  (define start (current-milliseconds))
  (define result (tensor-matmul M N))
  (define end (current-milliseconds))
  (display "Matrix ")
  (display size)
  (display "x")
  (display size)
  (display " took ")
  (display (- end start))
  (display " ms")
  (newline))

(benchmark-matmul 100)
(benchmark-matmul 500)
(benchmark-matmul 1000)
```

**Step 1.9**: Create benchmark script `scripts/benchmark_simd.sh`

```bash
#!/bin/bash
echo "SIMD Benchmark Results"
echo "====================="
echo ""

# Build with optimizations
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run benchmarks
./build/eshkol-run tests/simd/vectorized_ops_test.esk
```

**Verification Checklist**:
- [ ] SIMD detection works on target hardware
- [ ] Dot product matches scalar implementation
- [ ] Matrix multiplication produces correct results
- [ ] 4-8x speedup on large matrices
- [ ] All existing tests still pass

---

## Phase 2: Parallel Execution (11 days)

### Why Second?
- Complements SIMD for multi-core utilization
- Enables parallel-map, parallel-reduce
- Foundation for distributed computing later
- No external dependencies

### Day 13-14: Work-Stealing Scheduler Design

**Files to Create**:
```
inc/eshkol/parallel/scheduler.h
inc/eshkol/parallel/work_stealing_deque.h
lib/parallel/scheduler.cpp
lib/parallel/work_stealing_deque.cpp
```

**Step 2.1**: Create `inc/eshkol/parallel/work_stealing_deque.h`

```cpp
#ifndef ESHKOL_WORK_STEALING_DEQUE_H
#define ESHKOL_WORK_STEALING_DEQUE_H

#include <atomic>
#include <memory>
#include <optional>

namespace eshkol::parallel {

// Lock-free Chase-Lev work-stealing deque
// Paper: "Dynamic Circular Work-Stealing Deque" (Chase & Lev, 2005)
template<typename T>
class WorkStealingDeque {
public:
    explicit WorkStealingDeque(size_t initial_capacity = 1024);
    ~WorkStealingDeque();

    // Owner operations (single-threaded, bottom of deque)
    void push(T item);
    std::optional<T> pop();

    // Thief operations (multi-threaded, top of deque)
    std::optional<T> steal();

    size_t size() const;
    bool empty() const;

private:
    struct CircularArray {
        std::unique_ptr<std::atomic<T>[]> buffer;
        size_t capacity;
        size_t mask;

        explicit CircularArray(size_t cap);
        T get(size_t idx) const;
        void put(size_t idx, T val);
        CircularArray* grow(size_t bottom, size_t top);
    };

    std::atomic<size_t> top_;
    std::atomic<size_t> bottom_;
    std::atomic<CircularArray*> array_;
};

} // namespace eshkol::parallel

#endif
```

**Step 2.2**: Create `inc/eshkol/parallel/scheduler.h`

```cpp
#ifndef ESHKOL_SCHEDULER_H
#define ESHKOL_SCHEDULER_H

#include <functional>
#include <vector>
#include <thread>
#include <atomic>
#include <memory>
#include <random>
#include "work_stealing_deque.h"

namespace eshkol::parallel {

using Task = std::function<void()>;

class Worker {
public:
    Worker(size_t id, class Scheduler& scheduler);
    ~Worker();

    void start();
    void stop();
    void submit(Task task);
    std::optional<Task> steal();

    size_t id() const { return id_; }

private:
    void run();
    std::optional<Task> findWork();

    size_t id_;
    Scheduler& scheduler_;
    WorkStealingDeque<Task> local_queue_;
    std::thread thread_;
    std::atomic<bool> running_{false};
};

class Scheduler {
public:
    static Scheduler& instance();

    void initialize(size_t num_workers = 0);  // 0 = hardware_concurrency
    void shutdown();

    // Submit single task
    void submit(Task task);

    // Parallel for: body(i) for i in [start, end)
    template<typename Body>
    void parallelFor(size_t start, size_t end, Body body, size_t grain_size = 1);

    // Parallel map: result[i] = f(input[i])
    template<typename T, typename F>
    std::vector<T> parallelMap(const std::vector<T>& input, F f);

    // Parallel reduce: fold over input with op
    template<typename T, typename BinOp>
    T parallelReduce(const std::vector<T>& input, T identity, BinOp op);

    // Get random worker for stealing
    Worker* getVictim(size_t exclude_id);

    size_t numWorkers() const { return workers_.size(); }
    bool isInitialized() const { return initialized_; }

private:
    Scheduler() = default;
    ~Scheduler();

    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<bool> initialized_{false};
    std::mt19937 rng_;
    std::mutex rng_mutex_;
};

// Implementation of templates
template<typename Body>
void Scheduler::parallelFor(size_t start, size_t end, Body body, size_t grain_size) {
    if (!initialized_ || end <= start) return;

    size_t n = end - start;
    if (n <= grain_size) {
        // Sequential
        for (size_t i = start; i < end; i++) body(i);
        return;
    }

    std::atomic<size_t> completed{0};
    size_t num_tasks = (n + grain_size - 1) / grain_size;

    for (size_t t = 0; t < num_tasks; t++) {
        size_t task_start = start + t * grain_size;
        size_t task_end = std::min(task_start + grain_size, end);

        submit([&body, task_start, task_end, &completed]() {
            for (size_t i = task_start; i < task_end; i++) {
                body(i);
            }
            completed++;
        });
    }

    // Wait for completion
    while (completed < num_tasks) {
        std::this_thread::yield();
    }
}

} // namespace eshkol::parallel

#endif
```

### Day 15-17: Scheduler Implementation

**Step 2.3**: Create `lib/parallel/work_stealing_deque.cpp`

```cpp
#include "eshkol/parallel/work_stealing_deque.h"

namespace eshkol::parallel {

template<typename T>
WorkStealingDeque<T>::CircularArray::CircularArray(size_t cap)
    : capacity(cap), mask(cap - 1) {
    buffer = std::make_unique<std::atomic<T>[]>(cap);
}

template<typename T>
T WorkStealingDeque<T>::CircularArray::get(size_t idx) const {
    return buffer[idx & mask].load(std::memory_order_relaxed);
}

template<typename T>
void WorkStealingDeque<T>::CircularArray::put(size_t idx, T val) {
    buffer[idx & mask].store(val, std::memory_order_relaxed);
}

template<typename T>
typename WorkStealingDeque<T>::CircularArray*
WorkStealingDeque<T>::CircularArray::grow(size_t bottom, size_t top) {
    auto new_array = new CircularArray(capacity * 2);
    for (size_t i = top; i < bottom; i++) {
        new_array->put(i, get(i));
    }
    return new_array;
}

template<typename T>
WorkStealingDeque<T>::WorkStealingDeque(size_t initial_capacity)
    : top_(0), bottom_(0) {
    array_ = new CircularArray(initial_capacity);
}

template<typename T>
WorkStealingDeque<T>::~WorkStealingDeque() {
    delete array_.load();
}

template<typename T>
void WorkStealingDeque<T>::push(T item) {
    size_t b = bottom_.load(std::memory_order_relaxed);
    size_t t = top_.load(std::memory_order_acquire);
    auto* a = array_.load(std::memory_order_relaxed);

    if (b - t >= a->capacity - 1) {
        // Grow
        a = a->grow(b, t);
        array_.store(a, std::memory_order_release);
    }

    a->put(b, item);
    std::atomic_thread_fence(std::memory_order_release);
    bottom_.store(b + 1, std::memory_order_relaxed);
}

template<typename T>
std::optional<T> WorkStealingDeque<T>::pop() {
    size_t b = bottom_.load(std::memory_order_relaxed) - 1;
    auto* a = array_.load(std::memory_order_relaxed);
    bottom_.store(b, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);

    size_t t = top_.load(std::memory_order_relaxed);

    if (t <= b) {
        // Non-empty
        T item = a->get(b);
        if (t == b) {
            // Last element, compete with thieves
            if (!top_.compare_exchange_strong(t, t + 1,
                    std::memory_order_seq_cst, std::memory_order_relaxed)) {
                // Lost race
                bottom_.store(b + 1, std::memory_order_relaxed);
                return std::nullopt;
            }
            bottom_.store(b + 1, std::memory_order_relaxed);
        }
        return item;
    } else {
        // Empty
        bottom_.store(b + 1, std::memory_order_relaxed);
        return std::nullopt;
    }
}

template<typename T>
std::optional<T> WorkStealingDeque<T>::steal() {
    size_t t = top_.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    size_t b = bottom_.load(std::memory_order_acquire);

    if (t < b) {
        auto* a = array_.load(std::memory_order_consume);
        T item = a->get(t);
        if (!top_.compare_exchange_strong(t, t + 1,
                std::memory_order_seq_cst, std::memory_order_relaxed)) {
            return std::nullopt;  // Lost race
        }
        return item;
    }
    return std::nullopt;
}

// Explicit instantiation
template class WorkStealingDeque<std::function<void()>>;

} // namespace eshkol::parallel
```

**Step 2.4**: Create `lib/parallel/scheduler.cpp`

```cpp
#include "eshkol/parallel/scheduler.h"
#include <random>

namespace eshkol::parallel {

Worker::Worker(size_t id, Scheduler& scheduler)
    : id_(id), scheduler_(scheduler) {}

Worker::~Worker() {
    stop();
}

void Worker::start() {
    running_ = true;
    thread_ = std::thread(&Worker::run, this);
}

void Worker::stop() {
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void Worker::submit(Task task) {
    local_queue_.push(std::move(task));
}

std::optional<Task> Worker::steal() {
    return local_queue_.steal();
}

void Worker::run() {
    while (running_) {
        auto task = findWork();
        if (task) {
            (*task)();
        } else {
            std::this_thread::yield();
        }
    }
}

std::optional<Task> Worker::findWork() {
    // Try local queue first
    auto task = local_queue_.pop();
    if (task) return task;

    // Try stealing from others
    Worker* victim = scheduler_.getVictim(id_);
    if (victim) {
        return victim->steal();
    }

    return std::nullopt;
}

Scheduler& Scheduler::instance() {
    static Scheduler instance;
    return instance;
}

void Scheduler::initialize(size_t num_workers) {
    if (initialized_) return;

    if (num_workers == 0) {
        num_workers = std::thread::hardware_concurrency();
    }

    workers_.reserve(num_workers);
    for (size_t i = 0; i < num_workers; i++) {
        workers_.push_back(std::make_unique<Worker>(i, *this));
        workers_.back()->start();
    }

    initialized_ = true;
}

void Scheduler::shutdown() {
    for (auto& worker : workers_) {
        worker->stop();
    }
    workers_.clear();
    initialized_ = false;
}

Scheduler::~Scheduler() {
    shutdown();
}

void Scheduler::submit(Task task) {
    if (!initialized_ || workers_.empty()) {
        task();  // Run synchronously
        return;
    }

    // Round-robin to workers
    static std::atomic<size_t> next_worker{0};
    size_t idx = next_worker++ % workers_.size();
    workers_[idx]->submit(std::move(task));
}

Worker* Scheduler::getVictim(size_t exclude_id) {
    if (workers_.size() <= 1) return nullptr;

    std::lock_guard<std::mutex> lock(rng_mutex_);
    std::uniform_int_distribution<size_t> dist(0, workers_.size() - 1);

    size_t attempts = 3;
    while (attempts-- > 0) {
        size_t victim_id = dist(rng_);
        if (victim_id != exclude_id) {
            return workers_[victim_id].get();
        }
    }
    return nullptr;
}

} // namespace eshkol::parallel
```

### Day 18-20: Eshkol Language Integration

**Step 2.5**: Add parallel primitives to parser

Modify `lib/frontend/parser.cpp`:
```cpp
// Add to get_operator_type():
if (op == "parallel-map") return ESHKOL_PARALLEL_MAP_OP;
if (op == "parallel-for") return ESHKOL_PARALLEL_FOR_OP;
if (op == "parallel-reduce") return ESHKOL_PARALLEL_REDUCE_OP;
```

**Step 2.6**: Add to `inc/eshkol/eshkol.h`:
```cpp
ESHKOL_PARALLEL_MAP_OP,
ESHKOL_PARALLEL_FOR_OP,
ESHKOL_PARALLEL_REDUCE_OP,
```

**Step 2.7**: Implement codegen in `lib/backend/llvm_codegen.cpp`:

```cpp
llvm::Value* LLVMCodegen::codegenParallelMap(const eshkol_ast_t& ast) {
    // (parallel-map f list)
    auto func = codegenExpr(ast.operation.call_op.func);
    auto list = codegenExpr(ast.operation.call_op.variables[0]);

    // Convert list to vector
    auto vec = listToVector(list);
    auto n = vectorLength(vec);

    // Allocate result vector
    auto result = allocateVector(n);

    // Generate parallel loop
    auto& scheduler = eshkol::parallel::Scheduler::instance();

    // Create task function
    auto task_fn = createParallelMapTask(func, vec, result);

    // Submit to scheduler
    // [implementation details]

    return vectorToList(result);
}
```

### Day 21-23: Testing

**Step 2.8**: Create `tests/parallel/parallel_test.esk`

```scheme
(require stdlib)

(display "=== Parallel Execution Tests ===")
(newline)

;; Test parallel-map
(define data (range 1 1000001))
(define (square x) (* x x))

(display "Sequential map... ")
(define start1 (current-milliseconds))
(define result1 (map square data))
(define end1 (current-milliseconds))
(display (- end1 start1))
(display " ms")
(newline)

(display "Parallel map... ")
(define start2 (current-milliseconds))
(define result2 (parallel-map square data))
(define end2 (current-milliseconds))
(display (- end2 start2))
(display " ms")
(newline)

(display "Speedup: ")
(display (/ (- end1 start1) (- end2 start2)))
(display "x")
(newline)
```

---

## Phase 3: Hygienic Macro System (16 days)

### Why Third?
- Enables user-defined control structures
- Unlocks DSL creation
- Makes the language extensible
- Required for full Scheme compatibility

### Day 24-26: Syntax Object Design

**Files to Create**:
```
inc/eshkol/macro/syntax.h
lib/macro/syntax.cpp
```

**Step 3.1**: Create `inc/eshkol/macro/syntax.h`

```cpp
#ifndef ESHKOL_SYNTAX_H
#define ESHKOL_SYNTAX_H

#include <string>
#include <vector>
#include <set>
#include <memory>
#include <variant>

namespace eshkol::macro {

// Scope identifier for hygiene
using ScopeId = uint64_t;
using ScopeSet = std::set<ScopeId>;

// Source location for error messages
struct SourceLocation {
    std::string file;
    size_t line;
    size_t column;
};

// Forward declaration
class SyntaxObject;
using SyntaxPtr = std::shared_ptr<SyntaxObject>;

// Datum types that can appear in syntax objects
struct SyntaxSymbol {
    std::string name;
};

struct SyntaxList {
    std::vector<SyntaxPtr> elements;
};

struct SyntaxNumber {
    double value;
    bool is_integer;
};

struct SyntaxString {
    std::string value;
};

struct SyntaxBoolean {
    bool value;
};

using Datum = std::variant<
    SyntaxSymbol,
    SyntaxList,
    SyntaxNumber,
    SyntaxString,
    SyntaxBoolean
>;

// Syntax object: datum + scopes + location
class SyntaxObject {
public:
    SyntaxObject(Datum datum, ScopeSet scopes, SourceLocation loc);

    const Datum& datum() const { return datum_; }
    const ScopeSet& scopes() const { return scopes_; }
    const SourceLocation& location() const { return location_; }

    // Add a scope (for macro expansion)
    SyntaxPtr addScope(ScopeId scope) const;

    // Flip a scope (for macro definition vs use)
    SyntaxPtr flipScope(ScopeId scope) const;

    // Check if this is a specific type
    bool isSymbol() const;
    bool isList() const;
    bool isNumber() const;
    bool isString() const;
    bool isBoolean() const;

    // Get as specific type (throws if wrong type)
    const SyntaxSymbol& asSymbol() const;
    const SyntaxList& asList() const;
    const SyntaxNumber& asNumber() const;
    const SyntaxString& asString() const;
    const SyntaxBoolean& asBoolean() const;

    // Create syntax objects
    static SyntaxPtr makeSymbol(const std::string& name, ScopeSet scopes, SourceLocation loc);
    static SyntaxPtr makeList(std::vector<SyntaxPtr> elements, ScopeSet scopes, SourceLocation loc);
    static SyntaxPtr makeNumber(double value, bool is_integer, ScopeSet scopes, SourceLocation loc);
    static SyntaxPtr makeString(const std::string& value, ScopeSet scopes, SourceLocation loc);
    static SyntaxPtr makeBoolean(bool value, ScopeSet scopes, SourceLocation loc);

private:
    Datum datum_;
    ScopeSet scopes_;
    SourceLocation location_;
};

// Global scope counter
ScopeId freshScope();

} // namespace eshkol::macro

#endif
```

### Day 27-30: Pattern Matching

**Files to Create**:
```
inc/eshkol/macro/pattern.h
lib/macro/pattern.cpp
```

**Step 3.2**: Create `inc/eshkol/macro/pattern.h`

```cpp
#ifndef ESHKOL_PATTERN_H
#define ESHKOL_PATTERN_H

#include "syntax.h"
#include <unordered_map>
#include <unordered_set>

namespace eshkol::macro {

// Pattern types for syntax-rules
enum class PatternKind {
    Literal,      // Must match exactly
    Variable,     // Binds to value
    Wildcard,     // _ matches anything, no binding
    List,         // (pattern ...)
    Ellipsis,     // pattern ... (zero or more)
    Improper      // (pattern . pattern)
};

class Pattern;
using PatternPtr = std::shared_ptr<Pattern>;

class Pattern {
public:
    PatternKind kind;

    // For Literal/Variable
    std::string name;

    // For List/Ellipsis
    std::vector<PatternPtr> elements;
    PatternPtr tail;  // For improper lists

    // For Ellipsis
    PatternPtr repeated;  // The pattern that repeats

    static PatternPtr parseLiteral(const std::string& name);
    static PatternPtr parseVariable(const std::string& name);
    static PatternPtr parseWildcard();
    static PatternPtr parseList(std::vector<PatternPtr> elements);
    static PatternPtr parseEllipsis(PatternPtr repeated);
};

// Result of pattern matching
struct MatchBindings {
    // variable_name -> list of matched values
    // For non-ellipsis: list has one element
    // For ellipsis: list has zero or more elements
    std::unordered_map<std::string, std::vector<SyntaxPtr>> bindings;

    bool success = false;
    std::string error;
};

// Pattern matcher
class PatternMatcher {
public:
    PatternMatcher(const std::unordered_set<std::string>& literals);

    // Parse a pattern from syntax object
    PatternPtr parsePattern(SyntaxPtr stx);

    // Match pattern against input
    MatchBindings match(PatternPtr pattern, SyntaxPtr input);

private:
    std::unordered_set<std::string> literals_;

    bool matchImpl(PatternPtr pattern, SyntaxPtr input, MatchBindings& bindings);
    bool matchEllipsis(PatternPtr pattern, const std::vector<SyntaxPtr>& inputs,
                       size_t start, MatchBindings& bindings);
};

} // namespace eshkol::macro

#endif
```

### Day 31-35: Template Expansion

**Files to Create**:
```
inc/eshkol/macro/template.h
lib/macro/template.cpp
```

**Step 3.3**: Create `inc/eshkol/macro/template.h`

```cpp
#ifndef ESHKOL_TEMPLATE_H
#define ESHKOL_TEMPLATE_H

#include "syntax.h"
#include "pattern.h"

namespace eshkol::macro {

// Template types for syntax-rules
enum class TemplateKind {
    Literal,      // Literal symbol/datum
    Variable,     // Substitute from bindings
    List,         // (template ...)
    Ellipsis,     // template ... (expand repeated)
    Subtemplate   // Nested template
};

class Template;
using TemplatePtr = std::shared_ptr<Template>;

class Template {
public:
    TemplateKind kind;

    // For Literal
    SyntaxPtr literal;

    // For Variable
    std::string name;

    // For List
    std::vector<TemplatePtr> elements;

    // For Ellipsis
    TemplatePtr repeated;
    std::vector<std::string> ellipsis_vars;  // Variables inside ellipsis
};

// Template expander
class TemplateExpander {
public:
    // Parse a template from syntax object
    TemplatePtr parseTemplate(SyntaxPtr stx,
                              const std::unordered_set<std::string>& pattern_vars);

    // Expand template with bindings
    SyntaxPtr expand(TemplatePtr tmpl, const MatchBindings& bindings,
                     ScopeId intro_scope);

private:
    SyntaxPtr expandImpl(TemplatePtr tmpl, const MatchBindings& bindings,
                         ScopeId intro_scope, size_t ellipsis_index = 0);

    // Count how many times an ellipsis should expand
    size_t countEllipsisIterations(const std::vector<std::string>& vars,
                                   const MatchBindings& bindings);
};

} // namespace eshkol::macro

#endif
```

### Day 36-39: Macro Expander

**Files to Create**:
```
inc/eshkol/macro/expander.h
lib/macro/expander.cpp
```

**Step 3.4**: Create `inc/eshkol/macro/expander.h`

```cpp
#ifndef ESHKOL_EXPANDER_H
#define ESHKOL_EXPANDER_H

#include "syntax.h"
#include "pattern.h"
#include "template.h"
#include <unordered_map>

namespace eshkol::macro {

// A syntax transformer (macro)
struct Transformer {
    std::vector<std::string> literals;
    std::vector<std::pair<PatternPtr, TemplatePtr>> rules;
    ScopeId definition_scope;
};

// Macro expander
class MacroExpander {
public:
    MacroExpander();

    // Define a macro
    void defineSyntax(const std::string& name, Transformer transformer);

    // Check if name is a macro
    bool isMacro(const std::string& name) const;

    // Expand all macros in syntax object (recursive)
    SyntaxPtr expand(SyntaxPtr stx);

    // Expand a single macro application
    SyntaxPtr expandMacro(const std::string& name, SyntaxPtr stx);

private:
    std::unordered_map<std::string, Transformer> macros_;

    // Scope-based identifier resolution
    std::string resolveIdentifier(SyntaxPtr id);

    // Check if identifier refers to a macro
    bool refersMacro(SyntaxPtr id);

    // Expand application form
    SyntaxPtr expandApplication(SyntaxPtr stx);

    // Expand define form
    SyntaxPtr expandDefine(SyntaxPtr stx);

    // Expand let/let*/letrec
    SyntaxPtr expandLet(SyntaxPtr stx);
};

// Parse define-syntax form
Transformer parseDefineSyntax(SyntaxPtr stx);

// Parse syntax-rules form
Transformer parseSyntaxRules(SyntaxPtr stx);

} // namespace eshkol::macro

#endif
```

**Step 3.5**: Implement `lib/macro/expander.cpp`

```cpp
#include "eshkol/macro/expander.h"

namespace eshkol::macro {

MacroExpander::MacroExpander() {}

void MacroExpander::defineSyntax(const std::string& name, Transformer transformer) {
    macros_[name] = std::move(transformer);
}

bool MacroExpander::isMacro(const std::string& name) const {
    return macros_.find(name) != macros_.end();
}

SyntaxPtr MacroExpander::expand(SyntaxPtr stx) {
    if (!stx) return nullptr;

    if (stx->isSymbol()) {
        // Check if it's a macro identifier
        return stx;  // Symbols don't expand on their own
    }

    if (stx->isList()) {
        const auto& list = stx->asList();
        if (list.elements.empty()) {
            return stx;
        }

        auto first = list.elements[0];

        // Check for define-syntax
        if (first->isSymbol() && first->asSymbol().name == "define-syntax") {
            // Parse and register the macro
            auto transformer = parseDefineSyntax(stx);
            auto name_stx = list.elements[1];
            if (name_stx->isSymbol()) {
                defineSyntax(name_stx->asSymbol().name, transformer);
            }
            // define-syntax expands to nothing
            return SyntaxObject::makeList({}, stx->scopes(), stx->location());
        }

        // Check if first element is a macro
        if (first->isSymbol()) {
            const auto& name = first->asSymbol().name;
            if (isMacro(name)) {
                auto expanded = expandMacro(name, stx);
                // Recursively expand
                return expand(expanded);
            }
        }

        // Not a macro, expand subforms
        std::vector<SyntaxPtr> expanded_elements;
        for (const auto& elem : list.elements) {
            expanded_elements.push_back(expand(elem));
        }
        return SyntaxObject::makeList(expanded_elements, stx->scopes(), stx->location());
    }

    return stx;
}

SyntaxPtr MacroExpander::expandMacro(const std::string& name, SyntaxPtr stx) {
    const auto& transformer = macros_.at(name);
    PatternMatcher matcher(
        std::unordered_set<std::string>(transformer.literals.begin(),
                                         transformer.literals.end())
    );
    TemplateExpander expander;

    // Try each rule
    for (const auto& [pattern, tmpl] : transformer.rules) {
        auto bindings = matcher.match(pattern, stx);
        if (bindings.success) {
            // Create introduction scope for hygiene
            auto intro_scope = freshScope();
            return expander.expand(tmpl, bindings, intro_scope);
        }
    }

    // No rule matched - error
    throw std::runtime_error("No matching syntax-rules clause for: " + name);
}

Transformer parseDefineSyntax(SyntaxPtr stx) {
    // (define-syntax name (syntax-rules (literals ...) rules ...))
    const auto& list = stx->asList();

    if (list.elements.size() < 3) {
        throw std::runtime_error("define-syntax requires name and syntax-rules");
    }

    auto syntax_rules = list.elements[2];
    return parseSyntaxRules(syntax_rules);
}

Transformer parseSyntaxRules(SyntaxPtr stx) {
    // (syntax-rules (literal ...) (pattern template) ...)
    Transformer transformer;
    transformer.definition_scope = freshScope();

    const auto& list = stx->asList();

    if (list.elements.size() < 2) {
        throw std::runtime_error("syntax-rules requires at least literals list");
    }

    // Parse literals
    auto literals_stx = list.elements[1];
    if (literals_stx->isList()) {
        for (const auto& lit : literals_stx->asList().elements) {
            if (lit->isSymbol()) {
                transformer.literals.push_back(lit->asSymbol().name);
            }
        }
    }

    // Parse rules
    std::unordered_set<std::string> literals_set(
        transformer.literals.begin(), transformer.literals.end()
    );
    PatternMatcher matcher(literals_set);

    for (size_t i = 2; i < list.elements.size(); i++) {
        auto rule_stx = list.elements[i];
        if (!rule_stx->isList()) continue;

        const auto& rule = rule_stx->asList();
        if (rule.elements.size() < 2) continue;

        auto pattern = matcher.parsePattern(rule.elements[0]);

        // Collect pattern variables
        std::unordered_set<std::string> pattern_vars;
        // [collect vars from pattern]

        TemplateExpander expander;
        auto tmpl = expander.parseTemplate(rule.elements[1], pattern_vars);

        transformer.rules.emplace_back(pattern, tmpl);
    }

    return transformer;
}

} // namespace eshkol::macro
```

### Day 40: Parser Integration

**Step 3.6**: Modify `lib/frontend/parser.cpp`

Add to get_operator_type():
```cpp
if (op == "define-syntax") return ESHKOL_DEFINE_SYNTAX_OP;
if (op == "syntax-rules") return ESHKOL_SYNTAX_RULES_OP;
if (op == "quasiquote" || op == "`") return ESHKOL_QUASIQUOTE_OP;
if (op == "unquote" || op == ",") return ESHKOL_UNQUOTE_OP;
if (op == "unquote-splicing" || op == ",@") return ESHKOL_UNQUOTE_SPLICING_OP;
```

**Step 3.7**: Add macro expansion pass before codegen

In `lib/backend/llvm_codegen.cpp`:
```cpp
#include "eshkol/macro/expander.h"

llvm::Value* LLVMCodegen::codegen(eshkol_ast_t& ast) {
    // Step 1: Convert AST to syntax objects
    auto stx = astToSyntax(ast);

    // Step 2: Macro expansion
    macro::MacroExpander expander;
    registerStandardMacros(expander);  // when, unless, etc.
    auto expanded = expander.expand(stx);

    // Step 3: Convert back to AST
    auto expanded_ast = syntaxToAst(expanded);

    // Step 4: Original codegen
    return codegenImpl(*expanded_ast);
}
```

### Day 41-42: Standard Macros

**Step 3.8**: Create `lib/stdlib/macros.esk`

```scheme
;;; Standard Macros

;; when - one-armed if
(define-syntax when
  (syntax-rules ()
    ((when test body ...)
     (if test (begin body ...) #f))))

;; unless - negated when
(define-syntax unless
  (syntax-rules ()
    ((unless test body ...)
     (if test #f (begin body ...)))))

;; cond (if not already special form)
(define-syntax cond
  (syntax-rules (else =>)
    ((cond (else result ...))
     (begin result ...))
    ((cond (test => func) rest ...)
     (let ((temp test))
       (if temp (func temp) (cond rest ...))))
    ((cond (test result ...) rest ...)
     (if test (begin result ...) (cond rest ...)))))

;; case
(define-syntax case
  (syntax-rules (else)
    ((case expr (else result ...))
     (begin result ...))
    ((case expr ((datum ...) result ...) rest ...)
     (let ((temp expr))
       (if (memv temp '(datum ...))
           (begin result ...)
           (case temp rest ...))))))

;; let* in terms of nested lets
(define-syntax let*
  (syntax-rules ()
    ((let* () body ...)
     (begin body ...))
    ((let* ((var val) rest ...) body ...)
     (let ((var val))
       (let* (rest ...) body ...)))))

;; letrec using set!
(define-syntax letrec
  (syntax-rules ()
    ((letrec ((var val) ...) body ...)
     (let ((var #f) ...)
       (set! var val) ...
       body ...))))

;; do loop
(define-syntax do
  (syntax-rules ()
    ((do ((var init step ...) ...)
         (test result ...)
         body ...)
     (letrec ((loop
               (lambda (var ...)
                 (if test
                     (begin result ...)
                     (begin
                       body ...
                       (loop (do-step var step ...) ...))))))
       (loop init ...)))))

(define-syntax do-step
  (syntax-rules ()
    ((do-step var) var)
    ((do-step var step) step)))
```

### Day 43: Testing

**Step 3.9**: Create `tests/macro/macro_test.esk`

```scheme
(require stdlib)

(display "=== Macro System Tests ===")
(newline)

;; Test when
(display "when (true): ")
(display (when #t 42))
(newline)

(display "when (false): ")
(display (when #f 42))
(newline)

;; Test unless
(display "unless (false): ")
(display (unless #f 42))
(newline)

;; Test let*
(display "let* sequencing: ")
(display (let* ((x 1)
                (y (+ x 1))
                (z (+ y 1)))
           z))
(display " (expected: 3)")
(newline)

;; Test hygiene
(display "Hygiene test: ")
(define-syntax swap
  (syntax-rules ()
    ((swap a b)
     (let ((temp a))
       (set! a b)
       (set! b temp)))))

(define temp 1)  ;; Outer temp shouldn't conflict
(define x 10)
(define y 20)
(swap x y)
(display (list x y temp))
(display " (expected: (20 10 1))")
(newline)
```

---

## Phase 4: GPU/CUDA Backend (25 days)

### Why Fourth?
- SIMD and parallel done first for CPU optimization
- GPU requires significant infrastructure
- 10-100x performance for ML workloads

### Day 44-46: GPU Runtime Abstraction

**Files to Create**:
```
inc/eshkol/gpu/gpu_types.h
inc/eshkol/gpu/gpu_runtime.h
lib/gpu/gpu_runtime.cpp
```

**Step 4.1**: Create `inc/eshkol/gpu/gpu_types.h`

```cpp
#ifndef ESHKOL_GPU_TYPES_H
#define ESHKOL_GPU_TYPES_H

#include <cstddef>
#include <cstdint>

namespace eshkol::gpu {

// GPU backend enumeration
enum class GPUBackend {
    None = 0,
    CUDA = 1,
    HIP = 2,      // AMD ROCm
    Metal = 3,    // Apple
    Vulkan = 4    // Cross-platform compute
};

// Device information
struct DeviceInfo {
    int device_id;
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_shared_memory_per_block;
    int warp_size;
    bool has_tensor_cores;
    bool has_fp16;
    bool has_bf16;
};

// Memory types
enum class MemoryType {
    Device = 0,       // GPU global memory
    Pinned = 1,       // CPU pinned (page-locked) for fast transfers
    Unified = 2,      // Managed memory (CUDA Unified Memory)
    HostMapped = 3    // Host memory mapped to device
};

// Memory allocation handle
struct GPUBuffer {
    void* device_ptr;
    void* host_ptr;          // For pinned/unified
    size_t size;
    MemoryType type;
    int device_id;
    uint64_t allocation_id;  // For tracking
};

// Stream for async operations
struct GPUStream {
    void* native_handle;     // cudaStream_t or equivalent
    int device_id;
    bool is_default;
};

// Event for synchronization
struct GPUEvent {
    void* native_handle;     // cudaEvent_t or equivalent
    int device_id;
};

// Kernel launch configuration
struct LaunchConfig {
    unsigned int grid_x, grid_y, grid_z;
    unsigned int block_x, block_y, block_z;
    size_t shared_memory_bytes;
    GPUStream* stream;
};

} // namespace eshkol::gpu

#endif
```

**Step 4.2**: Create `inc/eshkol/gpu/gpu_runtime.h`

```cpp
#ifndef ESHKOL_GPU_RUNTIME_H
#define ESHKOL_GPU_RUNTIME_H

#include "gpu_types.h"
#include <vector>
#include <string>
#include <functional>

namespace eshkol::gpu {

// Error handling
struct GPUError {
    int code;
    std::string message;
    std::string file;
    int line;

    bool ok() const { return code == 0; }
    static GPUError success() { return {0, "success", "", 0}; }
};

#define GPU_CHECK(call) \
    do { \
        auto err = (call); \
        if (!err.ok()) { \
            err.file = __FILE__; \
            err.line = __LINE__; \
            return err; \
        } \
    } while(0)

// GPU Runtime singleton
class GPURuntime {
public:
    static GPURuntime& instance();

    // Initialization
    GPUError initialize(GPUBackend preferred = GPUBackend::CUDA);
    void shutdown();
    bool isInitialized() const { return initialized_; }

    // Device management
    int deviceCount() const;
    GPUError getDeviceInfo(int device_id, DeviceInfo& info);
    GPUError setDevice(int device_id);
    int currentDevice() const;

    // Memory management
    GPUError allocate(GPUBuffer& buffer, size_t size, MemoryType type = MemoryType::Device);
    GPUError free(GPUBuffer& buffer);
    GPUError memcpyHostToDevice(GPUBuffer& dst, const void* src, size_t size, GPUStream* stream = nullptr);
    GPUError memcpyDeviceToHost(void* dst, const GPUBuffer& src, size_t size, GPUStream* stream = nullptr);
    GPUError memcpyDeviceToDevice(GPUBuffer& dst, const GPUBuffer& src, size_t size, GPUStream* stream = nullptr);
    GPUError memset(GPUBuffer& buffer, int value, size_t size, GPUStream* stream = nullptr);

    // Stream management
    GPUError createStream(GPUStream& stream, int device_id = -1);
    GPUError destroyStream(GPUStream& stream);
    GPUError synchronizeStream(GPUStream& stream);
    GPUStream& defaultStream();

    // Event management
    GPUError createEvent(GPUEvent& event);
    GPUError destroyEvent(GPUEvent& event);
    GPUError recordEvent(GPUEvent& event, GPUStream& stream);
    GPUError waitEvent(GPUStream& stream, GPUEvent& event);
    GPUError synchronizeEvent(GPUEvent& event);
    float elapsedTime(GPUEvent& start, GPUEvent& end);

    // Kernel launching (for pre-compiled kernels)
    GPUError launchKernel(void* kernel_func, LaunchConfig& config, void** args);

    // Device synchronization
    GPUError synchronizeDevice();

    // Query
    GPUBackend backend() const { return backend_; }
    const DeviceInfo& currentDeviceInfo() const;

private:
    GPURuntime() = default;
    ~GPURuntime();

    bool initialized_ = false;
    GPUBackend backend_ = GPUBackend::None;
    int current_device_ = 0;
    std::vector<DeviceInfo> devices_;
    GPUStream default_stream_;
};

// Memory pool for efficient allocation
class GPUMemoryPool {
public:
    GPUMemoryPool(size_t initial_size = 256 * 1024 * 1024);  // 256 MB default
    ~GPUMemoryPool();

    GPUError allocate(GPUBuffer& buffer, size_t size);
    void free(GPUBuffer& buffer);
    void reset();  // Free all allocations

    size_t totalSize() const { return total_size_; }
    size_t usedSize() const { return used_size_; }
    size_t freeSize() const { return total_size_ - used_size_; }

private:
    struct Block {
        size_t offset;
        size_t size;
        bool free;
    };

    GPUBuffer pool_buffer_;
    std::vector<Block> blocks_;
    size_t total_size_;
    size_t used_size_;

    void coalesce();  // Merge adjacent free blocks
};

} // namespace eshkol::gpu

#endif
```

**Step 4.3**: Create `lib/gpu/gpu_runtime.cpp`

```cpp
#include "eshkol/gpu/gpu_runtime.h"

#ifdef ESHKOL_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace eshkol::gpu {

GPURuntime& GPURuntime::instance() {
    static GPURuntime instance;
    return instance;
}

GPUError GPURuntime::initialize(GPUBackend preferred) {
    if (initialized_) return GPUError::success();

#ifdef ESHKOL_CUDA_ENABLED
    if (preferred == GPUBackend::CUDA || preferred == GPUBackend::None) {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);

        if (err == cudaSuccess && device_count > 0) {
            backend_ = GPUBackend::CUDA;

            // Enumerate devices
            devices_.resize(device_count);
            for (int i = 0; i < device_count; i++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);

                devices_[i].device_id = i;
                strncpy(devices_[i].name, prop.name, 255);
                devices_[i].total_memory = prop.totalGlobalMem;
                devices_[i].compute_capability_major = prop.major;
                devices_[i].compute_capability_minor = prop.minor;
                devices_[i].multiprocessor_count = prop.multiProcessorCount;
                devices_[i].max_threads_per_block = prop.maxThreadsPerBlock;
                devices_[i].max_shared_memory_per_block = prop.sharedMemPerBlock;
                devices_[i].warp_size = prop.warpSize;
                devices_[i].has_tensor_cores = (prop.major >= 7);
                devices_[i].has_fp16 = (prop.major >= 5);
                devices_[i].has_bf16 = (prop.major >= 8);
            }

            // Create default stream
            default_stream_.is_default = true;
            default_stream_.device_id = 0;
            default_stream_.native_handle = nullptr;  // CUDA default stream

            initialized_ = true;
            return GPUError::success();
        }
    }
#endif

    return {-1, "No GPU backend available", "", 0};
}

GPUError GPURuntime::allocate(GPUBuffer& buffer, size_t size, MemoryType type) {
#ifdef ESHKOL_CUDA_ENABLED
    buffer.size = size;
    buffer.type = type;
    buffer.device_id = current_device_;
    buffer.host_ptr = nullptr;

    cudaError_t err;
    switch (type) {
        case MemoryType::Device:
            err = cudaMalloc(&buffer.device_ptr, size);
            break;
        case MemoryType::Pinned:
            err = cudaMallocHost(&buffer.host_ptr, size);
            buffer.device_ptr = buffer.host_ptr;
            break;
        case MemoryType::Unified:
            err = cudaMallocManaged(&buffer.device_ptr, size);
            buffer.host_ptr = buffer.device_ptr;
            break;
        default:
            return {-1, "Unknown memory type", "", 0};
    }

    if (err != cudaSuccess) {
        return {err, cudaGetErrorString(err), "", 0};
    }

    return GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

GPUError GPURuntime::free(GPUBuffer& buffer) {
#ifdef ESHKOL_CUDA_ENABLED
    cudaError_t err;
    switch (buffer.type) {
        case MemoryType::Device:
            err = cudaFree(buffer.device_ptr);
            break;
        case MemoryType::Pinned:
            err = cudaFreeHost(buffer.host_ptr);
            break;
        case MemoryType::Unified:
            err = cudaFree(buffer.device_ptr);
            break;
        default:
            return {-1, "Unknown memory type", "", 0};
    }

    if (err != cudaSuccess) {
        return {err, cudaGetErrorString(err), "", 0};
    }

    buffer.device_ptr = nullptr;
    buffer.host_ptr = nullptr;
    buffer.size = 0;
    return GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

GPUError GPURuntime::memcpyHostToDevice(GPUBuffer& dst, const void* src,
                                         size_t size, GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    cudaError_t err;
    if (stream && stream->native_handle) {
        err = cudaMemcpyAsync(dst.device_ptr, src, size,
                              cudaMemcpyHostToDevice,
                              (cudaStream_t)stream->native_handle);
    } else {
        err = cudaMemcpy(dst.device_ptr, src, size, cudaMemcpyHostToDevice);
    }

    if (err != cudaSuccess) {
        return {err, cudaGetErrorString(err), "", 0};
    }
    return GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

GPUError GPURuntime::memcpyDeviceToHost(void* dst, const GPUBuffer& src,
                                         size_t size, GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    cudaError_t err;
    if (stream && stream->native_handle) {
        err = cudaMemcpyAsync(dst, src.device_ptr, size,
                              cudaMemcpyDeviceToHost,
                              (cudaStream_t)stream->native_handle);
    } else {
        err = cudaMemcpy(dst, src.device_ptr, size, cudaMemcpyDeviceToHost);
    }

    if (err != cudaSuccess) {
        return {err, cudaGetErrorString(err), "", 0};
    }
    return GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

GPUError GPURuntime::createStream(GPUStream& stream, int device_id) {
#ifdef ESHKOL_CUDA_ENABLED
    if (device_id < 0) device_id = current_device_;

    cudaStream_t cuda_stream;
    cudaError_t err = cudaStreamCreate(&cuda_stream);
    if (err != cudaSuccess) {
        return {err, cudaGetErrorString(err), "", 0};
    }

    stream.native_handle = cuda_stream;
    stream.device_id = device_id;
    stream.is_default = false;
    return GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

GPUError GPURuntime::synchronizeDevice() {
#ifdef ESHKOL_CUDA_ENABLED
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return {err, cudaGetErrorString(err), "", 0};
    }
    return GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

void GPURuntime::shutdown() {
#ifdef ESHKOL_CUDA_ENABLED
    if (initialized_) {
        cudaDeviceReset();
        initialized_ = false;
    }
#endif
}

GPURuntime::~GPURuntime() {
    shutdown();
}

// Memory Pool Implementation
GPUMemoryPool::GPUMemoryPool(size_t initial_size)
    : total_size_(initial_size), used_size_(0) {
    GPURuntime::instance().allocate(pool_buffer_, initial_size, MemoryType::Device);
    blocks_.push_back({0, initial_size, true});
}

GPUMemoryPool::~GPUMemoryPool() {
    GPURuntime::instance().free(pool_buffer_);
}

GPUError GPUMemoryPool::allocate(GPUBuffer& buffer, size_t size) {
    // Align to 256 bytes
    size = (size + 255) & ~255;

    // First-fit allocation
    for (auto& block : blocks_) {
        if (block.free && block.size >= size) {
            buffer.device_ptr = static_cast<char*>(pool_buffer_.device_ptr) + block.offset;
            buffer.size = size;
            buffer.type = MemoryType::Device;
            buffer.device_id = pool_buffer_.device_id;

            if (block.size > size) {
                // Split block
                blocks_.push_back({block.offset + size, block.size - size, true});
            }
            block.size = size;
            block.free = false;
            used_size_ += size;

            return GPUError::success();
        }
    }

    return {-1, "Out of GPU memory pool", "", 0};
}

void GPUMemoryPool::free(GPUBuffer& buffer) {
    size_t offset = static_cast<char*>(buffer.device_ptr) -
                    static_cast<char*>(pool_buffer_.device_ptr);

    for (auto& block : blocks_) {
        if (block.offset == offset) {
            block.free = true;
            used_size_ -= block.size;
            coalesce();
            return;
        }
    }
}

void GPUMemoryPool::coalesce() {
    // Sort blocks by offset
    std::sort(blocks_.begin(), blocks_.end(),
              [](const Block& a, const Block& b) { return a.offset < b.offset; });

    // Merge adjacent free blocks
    for (size_t i = 0; i + 1 < blocks_.size(); ) {
        if (blocks_[i].free && blocks_[i+1].free) {
            blocks_[i].size += blocks_[i+1].size;
            blocks_.erase(blocks_.begin() + i + 1);
        } else {
            i++;
        }
    }
}

void GPUMemoryPool::reset() {
    blocks_.clear();
    blocks_.push_back({0, total_size_, true});
    used_size_ = 0;
}

} // namespace eshkol::gpu
```

### Day 47-52: NVPTX Code Generator

**Files to Create**:
```
inc/eshkol/gpu/gpu_codegen.h
lib/gpu/gpu_codegen.cpp
```

**Step 4.4**: Create `inc/eshkol/gpu/gpu_codegen.h`

```cpp
#ifndef ESHKOL_GPU_CODEGEN_H
#define ESHKOL_GPU_CODEGEN_H

#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Target/TargetMachine.h>
#include <string>
#include <vector>
#include <memory>

namespace eshkol::gpu {

// Kernel parameter descriptor
struct KernelParam {
    std::string name;
    llvm::Type* type;
    bool is_pointer;
    bool is_readonly;
};

// Compiled kernel
struct CompiledKernel {
    std::string name;
    std::string ptx_code;
    std::vector<KernelParam> params;
    unsigned int shared_memory_size;
    unsigned int max_threads_per_block;
    void* cuda_function;  // CUfunction after loading
};

// GPU code generator
class GPUCodegen {
public:
    GPUCodegen();
    ~GPUCodegen();

    // Initialize for target GPU
    bool initialize(int compute_capability_major, int compute_capability_minor);

    // Create a new kernel
    void beginKernel(const std::string& name, const std::vector<KernelParam>& params);

    // Get builder for kernel body
    llvm::IRBuilder<>& builder() { return *builder_; }
    llvm::LLVMContext& context() { return *context_; }

    // Built-in functions for kernels
    llvm::Value* getThreadIdX();
    llvm::Value* getThreadIdY();
    llvm::Value* getThreadIdZ();
    llvm::Value* getBlockIdX();
    llvm::Value* getBlockIdY();
    llvm::Value* getBlockIdZ();
    llvm::Value* getBlockDimX();
    llvm::Value* getBlockDimY();
    llvm::Value* getBlockDimZ();
    llvm::Value* getGridDimX();
    llvm::Value* getGridDimY();
    llvm::Value* getGridDimZ();

    // Global thread index helpers
    llvm::Value* getGlobalIdX();
    llvm::Value* getGlobalIdY();
    llvm::Value* getGlobalIdZ();

    // Synchronization
    void syncThreads();
    void memoryFenceBlock();
    void memoryFenceDevice();

    // Shared memory allocation
    llvm::Value* allocateSharedMemory(llvm::Type* element_type, size_t count,
                                       const std::string& name);

    // Atomic operations
    llvm::Value* atomicAdd(llvm::Value* ptr, llvm::Value* val);
    llvm::Value* atomicMax(llvm::Value* ptr, llvm::Value* val);
    llvm::Value* atomicMin(llvm::Value* ptr, llvm::Value* val);
    llvm::Value* atomicCAS(llvm::Value* ptr, llvm::Value* cmp, llvm::Value* val);

    // Warp-level primitives (CUDA 9.0+)
    llvm::Value* warpShuffle(llvm::Value* val, llvm::Value* src_lane);
    llvm::Value* warpShuffleUp(llvm::Value* val, unsigned int delta);
    llvm::Value* warpShuffleDown(llvm::Value* val, unsigned int delta);
    llvm::Value* warpShuffleXor(llvm::Value* val, unsigned int mask);
    llvm::Value* warpVote(llvm::Value* predicate);
    llvm::Value* warpBallot(llvm::Value* predicate);

    // Math intrinsics (fast GPU versions)
    llvm::Value* fastExp(llvm::Value* x);
    llvm::Value* fastLog(llvm::Value* x);
    llvm::Value* fastSin(llvm::Value* x);
    llvm::Value* fastCos(llvm::Value* x);
    llvm::Value* fastRsqrt(llvm::Value* x);
    llvm::Value* fma(llvm::Value* a, llvm::Value* b, llvm::Value* c);

    // Finish kernel and compile to PTX
    CompiledKernel endKernel();

    // Compile entire module
    std::string compileToPTX();

    // High-level kernel generators
    CompiledKernel generateVectorAdd();
    CompiledKernel generateMatMul(bool use_shared_memory = true);
    CompiledKernel generateSoftmax();
    CompiledKernel generateAttention();
    CompiledKernel generateConv2d();

private:
    std::unique_ptr<llvm::LLVMContext> context_;
    std::unique_ptr<llvm::Module> module_;
    std::unique_ptr<llvm::IRBuilder<>> builder_;
    std::unique_ptr<llvm::TargetMachine> target_machine_;

    llvm::Function* current_kernel_;
    std::string current_kernel_name_;
    std::vector<KernelParam> current_params_;
    unsigned int shared_memory_used_;

    int sm_major_;
    int sm_minor_;

    // Initialize NVPTX target
    bool initializeNVPTX();

    // Get NVPTX intrinsic
    llvm::Function* getNVPTXIntrinsic(llvm::Intrinsic::ID id);
};

} // namespace eshkol::gpu

#endif
```

**Step 4.5**: Create `lib/gpu/gpu_codegen.cpp`

```cpp
#include "eshkol/gpu/gpu_codegen.h"
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils.h>

namespace eshkol::gpu {

GPUCodegen::GPUCodegen()
    : current_kernel_(nullptr), shared_memory_used_(0), sm_major_(7), sm_minor_(0) {
    context_ = std::make_unique<llvm::LLVMContext>();
    module_ = std::make_unique<llvm::Module>("eshkol_gpu", *context_);
    builder_ = std::make_unique<llvm::IRBuilder<>>(*context_);
}

GPUCodegen::~GPUCodegen() = default;

bool GPUCodegen::initialize(int compute_capability_major, int compute_capability_minor) {
    sm_major_ = compute_capability_major;
    sm_minor_ = compute_capability_minor;
    return initializeNVPTX();
}

bool GPUCodegen::initializeNVPTX() {
    // Initialize NVPTX target
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget("nvptx64-nvidia-cuda", error);
    if (!target) {
        return false;
    }

    // Create target machine
    std::string cpu = "sm_" + std::to_string(sm_major_) + std::to_string(sm_minor_);
    std::string features = "+ptx75";  // PTX 7.5 for CUDA 11.x

    llvm::TargetOptions opt;
    target_machine_.reset(target->createTargetMachine(
        "nvptx64-nvidia-cuda", cpu, features, opt, llvm::Reloc::Model::PIC_));

    module_->setDataLayout(target_machine_->createDataLayout());
    module_->setTargetTriple("nvptx64-nvidia-cuda");

    return true;
}

void GPUCodegen::beginKernel(const std::string& name, const std::vector<KernelParam>& params) {
    current_kernel_name_ = name;
    current_params_ = params;
    shared_memory_used_ = 0;

    // Build function type
    std::vector<llvm::Type*> param_types;
    for (const auto& p : params) {
        param_types.push_back(p.type);
    }

    auto func_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(*context_), param_types, false);

    current_kernel_ = llvm::Function::Create(
        func_type, llvm::Function::ExternalLinkage, name, module_.get());

    // Set kernel calling convention
    current_kernel_->setCallingConv(llvm::CallingConv::PTX_Kernel);

    // Add parameter attributes
    size_t i = 0;
    for (auto& arg : current_kernel_->args()) {
        arg.setName(params[i].name);
        if (params[i].is_readonly && params[i].is_pointer) {
            arg.addAttr(llvm::Attribute::ReadOnly);
            arg.addAttr(llvm::Attribute::NoCapture);
        }
        i++;
    }

    // Create entry block
    auto entry = llvm::BasicBlock::Create(*context_, "entry", current_kernel_);
    builder_->SetInsertPoint(entry);
}

llvm::Value* GPUCodegen::getThreadIdX() {
    auto intrinsic = llvm::Intrinsic::getDeclaration(
        module_.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);
    return builder_->CreateCall(intrinsic);
}

llvm::Value* GPUCodegen::getThreadIdY() {
    auto intrinsic = llvm::Intrinsic::getDeclaration(
        module_.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y);
    return builder_->CreateCall(intrinsic);
}

llvm::Value* GPUCodegen::getBlockIdX() {
    auto intrinsic = llvm::Intrinsic::getDeclaration(
        module_.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
    return builder_->CreateCall(intrinsic);
}

llvm::Value* GPUCodegen::getBlockDimX() {
    auto intrinsic = llvm::Intrinsic::getDeclaration(
        module_.get(), llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x);
    return builder_->CreateCall(intrinsic);
}

llvm::Value* GPUCodegen::getGlobalIdX() {
    auto block_id = getBlockIdX();
    auto block_dim = getBlockDimX();
    auto thread_id = getThreadIdX();
    auto offset = builder_->CreateMul(block_id, block_dim);
    return builder_->CreateAdd(offset, thread_id);
}

void GPUCodegen::syncThreads() {
    auto intrinsic = llvm::Intrinsic::getDeclaration(
        module_.get(), llvm::Intrinsic::nvvm_barrier0);
    builder_->CreateCall(intrinsic);
}

llvm::Value* GPUCodegen::allocateSharedMemory(llvm::Type* element_type, size_t count,
                                               const std::string& name) {
    auto array_type = llvm::ArrayType::get(element_type, count);

    // Shared memory is address space 3 in NVPTX
    auto shared_mem = new llvm::GlobalVariable(
        *module_, array_type, false,
        llvm::GlobalValue::InternalLinkage,
        llvm::UndefValue::get(array_type),
        name, nullptr,
        llvm::GlobalValue::NotThreadLocal,
        3  // Address space 3 = shared memory
    );

    shared_mem->setAlignment(llvm::MaybeAlign(16));

    // Calculate size
    auto& layout = module_->getDataLayout();
    shared_memory_used_ += layout.getTypeAllocSize(array_type);

    // Return pointer to first element
    auto zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context_), 0);
    return builder_->CreateGEP(array_type, shared_mem, {zero, zero});
}

llvm::Value* GPUCodegen::atomicAdd(llvm::Value* ptr, llvm::Value* val) {
    return builder_->CreateAtomicRMW(
        llvm::AtomicRMWInst::FAdd, ptr, val, llvm::MaybeAlign(4),
        llvm::AtomicOrdering::Monotonic);
}

llvm::Value* GPUCodegen::warpShuffleDown(llvm::Value* val, unsigned int delta) {
    auto intrinsic = llvm::Intrinsic::getDeclaration(
        module_.get(), llvm::Intrinsic::nvvm_shfl_sync_down_f32);

    auto mask = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context_), 0xFFFFFFFF);
    auto delta_val = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context_), delta);
    auto width = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context_), 32);

    return builder_->CreateCall(intrinsic, {mask, val, delta_val, width});
}

llvm::Value* GPUCodegen::fastExp(llvm::Value* x) {
    // Use NVPTX fast exp approximation
    auto intrinsic = llvm::Intrinsic::getDeclaration(
        module_.get(), llvm::Intrinsic::nvvm_ex2_approx_f);

    // exp(x) = 2^(x * log2(e))
    auto log2e = llvm::ConstantFP::get(llvm::Type::getFloatTy(*context_), 1.4426950408889634);
    auto scaled = builder_->CreateFMul(x, log2e);
    return builder_->CreateCall(intrinsic, {scaled});
}

llvm::Value* GPUCodegen::fma(llvm::Value* a, llvm::Value* b, llvm::Value* c) {
    auto intrinsic = llvm::Intrinsic::getDeclaration(
        module_.get(), llvm::Intrinsic::fma, {a->getType()});
    return builder_->CreateCall(intrinsic, {a, b, c});
}

CompiledKernel GPUCodegen::endKernel() {
    // Add return
    builder_->CreateRetVoid();

    // Verify function
    std::string errors;
    llvm::raw_string_ostream error_stream(errors);
    if (llvm::verifyFunction(*current_kernel_, &error_stream)) {
        throw std::runtime_error("Kernel verification failed: " + errors);
    }

    CompiledKernel result;
    result.name = current_kernel_name_;
    result.params = current_params_;
    result.shared_memory_size = shared_memory_used_;
    result.max_threads_per_block = 1024;  // Default, can be tuned
    result.cuda_function = nullptr;

    // Add metadata for CUDA
    auto one = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context_), 1);
    auto md_node = llvm::MDNode::get(*context_, {
        llvm::ValueAsMetadata::get(current_kernel_),
        llvm::MDString::get(*context_, "kernel"),
        llvm::ConstantAsMetadata::get(one)
    });
    module_->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md_node);

    current_kernel_ = nullptr;
    return result;
}

std::string GPUCodegen::compileToPTX() {
    std::string ptx;
    llvm::raw_string_ostream ptx_stream(ptx);

    llvm::legacy::PassManager pass_manager;
    target_machine_->addPassesToEmitFile(
        pass_manager, ptx_stream, nullptr,
        llvm::CodeGenFileType::AssemblyFile);

    pass_manager.run(*module_);

    return ptx;
}

// High-level kernel: Vector Addition
CompiledKernel GPUCodegen::generateVectorAdd() {
    auto float_ptr = llvm::PointerType::get(llvm::Type::getFloatTy(*context_), 0);
    auto int_type = llvm::Type::getInt32Ty(*context_);

    std::vector<KernelParam> params = {
        {"a", float_ptr, true, true},
        {"b", float_ptr, true, true},
        {"c", float_ptr, true, false},
        {"n", int_type, false, true}
    };

    beginKernel("vector_add", params);

    auto a = current_kernel_->getArg(0);
    auto b = current_kernel_->getArg(1);
    auto c = current_kernel_->getArg(2);
    auto n = current_kernel_->getArg(3);

    // i = blockIdx.x * blockDim.x + threadIdx.x
    auto i = getGlobalIdX();

    // Bounds check
    auto in_bounds = builder_->CreateICmpULT(i, n);
    auto then_block = llvm::BasicBlock::Create(*context_, "then", current_kernel_);
    auto end_block = llvm::BasicBlock::Create(*context_, "end", current_kernel_);
    builder_->CreateCondBr(in_bounds, then_block, end_block);

    builder_->SetInsertPoint(then_block);

    // c[i] = a[i] + b[i]
    auto float_type = llvm::Type::getFloatTy(*context_);
    auto a_ptr = builder_->CreateGEP(float_type, a, i);
    auto b_ptr = builder_->CreateGEP(float_type, b, i);
    auto c_ptr = builder_->CreateGEP(float_type, c, i);

    auto a_val = builder_->CreateLoad(float_type, a_ptr);
    auto b_val = builder_->CreateLoad(float_type, b_ptr);
    auto sum = builder_->CreateFAdd(a_val, b_val);
    builder_->CreateStore(sum, c_ptr);

    builder_->CreateBr(end_block);
    builder_->SetInsertPoint(end_block);

    return endKernel();
}

// High-level kernel: Tiled Matrix Multiplication
CompiledKernel GPUCodegen::generateMatMul(bool use_shared_memory) {
    constexpr int TILE_SIZE = 16;

    auto float_ptr = llvm::PointerType::get(llvm::Type::getFloatTy(*context_), 0);
    auto int_type = llvm::Type::getInt32Ty(*context_);

    std::vector<KernelParam> params = {
        {"A", float_ptr, true, true},
        {"B", float_ptr, true, true},
        {"C", float_ptr, true, false},
        {"M", int_type, false, true},
        {"K", int_type, false, true},
        {"N", int_type, false, true}
    };

    beginKernel("matmul_tiled", params);

    auto A = current_kernel_->getArg(0);
    auto B = current_kernel_->getArg(1);
    auto C = current_kernel_->getArg(2);
    auto M = current_kernel_->getArg(3);
    auto K = current_kernel_->getArg(4);
    auto N = current_kernel_->getArg(5);

    auto float_type = llvm::Type::getFloatTy(*context_);

    if (use_shared_memory) {
        // Allocate shared memory tiles
        auto tile_A = allocateSharedMemory(float_type, TILE_SIZE * TILE_SIZE, "tile_A");
        auto tile_B = allocateSharedMemory(float_type, TILE_SIZE * TILE_SIZE, "tile_B");

        // Thread indices
        auto tx = getThreadIdX();
        auto ty = getThreadIdY();
        auto bx = getBlockIdX();
        auto by = getBlockIdY();

        // Global row and column
        auto row = builder_->CreateAdd(builder_->CreateMul(by,
            llvm::ConstantInt::get(int_type, TILE_SIZE)), ty);
        auto col = builder_->CreateAdd(builder_->CreateMul(bx,
            llvm::ConstantInt::get(int_type, TILE_SIZE)), tx);

        // Accumulator
        auto acc = builder_->CreateAlloca(float_type, nullptr, "acc");
        builder_->CreateStore(llvm::ConstantFP::get(float_type, 0.0f), acc);

        // Tile loop
        auto num_tiles = builder_->CreateUDiv(
            builder_->CreateAdd(K, llvm::ConstantInt::get(int_type, TILE_SIZE - 1)),
            llvm::ConstantInt::get(int_type, TILE_SIZE));

        // [Full tiled matmul implementation with shared memory loading,
        //  synchronization barriers, and tile accumulation - ~100 more lines]
    }

    return endKernel();
}

// High-level kernel: Softmax
CompiledKernel GPUCodegen::generateSoftmax() {
    auto float_ptr = llvm::PointerType::get(llvm::Type::getFloatTy(*context_), 0);
    auto int_type = llvm::Type::getInt32Ty(*context_);

    std::vector<KernelParam> params = {
        {"input", float_ptr, true, true},
        {"output", float_ptr, true, false},
        {"batch_size", int_type, false, true},
        {"dim_size", int_type, false, true}
    };

    beginKernel("softmax", params);

    auto input = current_kernel_->getArg(0);
    auto output = current_kernel_->getArg(1);
    auto batch_size = current_kernel_->getArg(2);
    auto dim_size = current_kernel_->getArg(3);

    auto float_type = llvm::Type::getFloatTy(*context_);

    // Each block handles one row
    auto row = getBlockIdX();
    auto tid = getThreadIdX();

    // Allocate shared memory for reduction
    auto shared_max = allocateSharedMemory(float_type, 256, "shared_max");
    auto shared_sum = allocateSharedMemory(float_type, 256, "shared_sum");

    // Step 1: Find max (for numerical stability)
    // Each thread finds max of its elements
    auto thread_max = builder_->CreateAlloca(float_type);
    builder_->CreateStore(llvm::ConstantFP::get(float_type, -INFINITY), thread_max);

    // [Loop through elements assigned to this thread, find max]

    // Parallel reduction for max
    // [Warp shuffle reduction + shared memory reduction]

    syncThreads();

    // Step 2: Compute exp(x - max) and sum
    // [Similar parallel reduction pattern]

    syncThreads();

    // Step 3: Normalize by sum
    // [Each thread writes output[i] = exp(input[i] - max) / sum]

    return endKernel();
}

} // namespace eshkol::gpu
```

### Day 53-57: Tensor GPU Integration

**Files to Create**:
```
inc/eshkol/gpu/gpu_tensor.h
lib/gpu/gpu_tensor.cpp
```

**Step 4.6**: Create `inc/eshkol/gpu/gpu_tensor.h`

```cpp
#ifndef ESHKOL_GPU_TENSOR_H
#define ESHKOL_GPU_TENSOR_H

#include "gpu_runtime.h"
#include "gpu_codegen.h"
#include <vector>

namespace eshkol::gpu {

// GPU Tensor with automatic memory management
class GPUTensor {
public:
    GPUTensor();
    GPUTensor(const std::vector<size_t>& shape, bool on_device = true);
    GPUTensor(const GPUTensor& other);
    GPUTensor(GPUTensor&& other) noexcept;
    ~GPUTensor();

    GPUTensor& operator=(const GPUTensor& other);
    GPUTensor& operator=(GPUTensor&& other) noexcept;

    // Properties
    const std::vector<size_t>& shape() const { return shape_; }
    size_t rank() const { return shape_.size(); }
    size_t size() const;  // Total number of elements
    size_t sizeInBytes() const { return size() * sizeof(float); }
    bool isOnDevice() const { return on_device_; }

    // Data access
    float* devicePtr() { return static_cast<float*>(buffer_.device_ptr); }
    const float* devicePtr() const { return static_cast<const float*>(buffer_.device_ptr); }

    // Memory transfer
    GPUError copyFromHost(const float* data);
    GPUError copyToHost(float* data) const;
    GPUError copyFromHostAsync(const float* data, GPUStream& stream);
    GPUError copyToHostAsync(float* data, GPUStream& stream) const;

    // Move between device/host
    GPUError toDevice();
    GPUError toHost();

    // Reshape (no data copy)
    void reshape(const std::vector<size_t>& new_shape);

    // Factory methods
    static GPUTensor zeros(const std::vector<size_t>& shape);
    static GPUTensor ones(const std::vector<size_t>& shape);
    static GPUTensor random(const std::vector<size_t>& shape, float min = 0.0f, float max = 1.0f);
    static GPUTensor fromHost(const float* data, const std::vector<size_t>& shape);

private:
    std::vector<size_t> shape_;
    GPUBuffer buffer_;
    bool on_device_;
    bool owns_memory_;
};

// GPU Tensor operations
class GPUTensorOps {
public:
    static GPUTensorOps& instance();

    void initialize();
    void shutdown();

    // Element-wise operations
    GPUError add(const GPUTensor& a, const GPUTensor& b, GPUTensor& c, GPUStream* stream = nullptr);
    GPUError sub(const GPUTensor& a, const GPUTensor& b, GPUTensor& c, GPUStream* stream = nullptr);
    GPUError mul(const GPUTensor& a, const GPUTensor& b, GPUTensor& c, GPUStream* stream = nullptr);
    GPUError div(const GPUTensor& a, const GPUTensor& b, GPUTensor& c, GPUStream* stream = nullptr);

    // Scalar operations
    GPUError scale(const GPUTensor& a, float scalar, GPUTensor& b, GPUStream* stream = nullptr);
    GPUError addScalar(const GPUTensor& a, float scalar, GPUTensor& b, GPUStream* stream = nullptr);

    // Matrix operations
    GPUError matmul(const GPUTensor& A, const GPUTensor& B, GPUTensor& C, GPUStream* stream = nullptr);
    GPUError transpose(const GPUTensor& a, GPUTensor& b, GPUStream* stream = nullptr);

    // Reductions
    GPUError sum(const GPUTensor& a, GPUTensor& result, int axis = -1, GPUStream* stream = nullptr);
    GPUError max(const GPUTensor& a, GPUTensor& result, int axis = -1, GPUStream* stream = nullptr);
    GPUError mean(const GPUTensor& a, GPUTensor& result, int axis = -1, GPUStream* stream = nullptr);

    // Activation functions
    GPUError relu(const GPUTensor& a, GPUTensor& b, GPUStream* stream = nullptr);
    GPUError sigmoid(const GPUTensor& a, GPUTensor& b, GPUStream* stream = nullptr);
    GPUError tanh(const GPUTensor& a, GPUTensor& b, GPUStream* stream = nullptr);
    GPUError softmax(const GPUTensor& a, GPUTensor& b, int axis = -1, GPUStream* stream = nullptr);
    GPUError gelu(const GPUTensor& a, GPUTensor& b, GPUStream* stream = nullptr);

    // Neural network operations
    GPUError layerNorm(const GPUTensor& a, const GPUTensor& gamma, const GPUTensor& beta,
                       GPUTensor& output, float eps = 1e-5, GPUStream* stream = nullptr);
    GPUError batchNorm(const GPUTensor& a, const GPUTensor& gamma, const GPUTensor& beta,
                       const GPUTensor& running_mean, const GPUTensor& running_var,
                       GPUTensor& output, float eps = 1e-5, float momentum = 0.1,
                       bool training = true, GPUStream* stream = nullptr);
    GPUError attention(const GPUTensor& Q, const GPUTensor& K, const GPUTensor& V,
                       GPUTensor& output, const GPUTensor* mask = nullptr,
                       GPUStream* stream = nullptr);
    GPUError conv2d(const GPUTensor& input, const GPUTensor& kernel, GPUTensor& output,
                    int stride = 1, int padding = 0, GPUStream* stream = nullptr);

private:
    GPUTensorOps() = default;

    // Compiled kernels
    std::unordered_map<std::string, CompiledKernel> kernels_;
    GPUCodegen codegen_;
    bool initialized_ = false;

    // Launch helpers
    LaunchConfig computeLaunchConfig(size_t total_threads, size_t threads_per_block = 256);
    void loadKernel(CompiledKernel& kernel);
};

} // namespace eshkol::gpu

#endif
```

**Step 4.7**: Create `lib/gpu/gpu_tensor.cpp`

```cpp
#include "eshkol/gpu/gpu_tensor.h"
#include <algorithm>
#include <numeric>
#include <random>

#ifdef ESHKOL_CUDA_ENABLED
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn.h>
#endif

namespace eshkol::gpu {

GPUTensor::GPUTensor() : on_device_(false), owns_memory_(false) {}

GPUTensor::GPUTensor(const std::vector<size_t>& shape, bool on_device)
    : shape_(shape), on_device_(on_device), owns_memory_(true) {
    GPURuntime::instance().allocate(buffer_, sizeInBytes(),
        on_device ? MemoryType::Device : MemoryType::Pinned);
}

GPUTensor::~GPUTensor() {
    if (owns_memory_ && buffer_.device_ptr) {
        GPURuntime::instance().free(buffer_);
    }
}

size_t GPUTensor::size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
}

GPUError GPUTensor::copyFromHost(const float* data) {
    return GPURuntime::instance().memcpyHostToDevice(buffer_, data, sizeInBytes());
}

GPUError GPUTensor::copyToHost(float* data) const {
    return GPURuntime::instance().memcpyDeviceToHost(data, buffer_, sizeInBytes());
}

GPUTensor GPUTensor::zeros(const std::vector<size_t>& shape) {
    GPUTensor t(shape, true);
    GPURuntime::instance().memset(t.buffer_, 0, t.sizeInBytes());
    return t;
}

GPUTensor GPUTensor::fromHost(const float* data, const std::vector<size_t>& shape) {
    GPUTensor t(shape, true);
    t.copyFromHost(data);
    return t;
}

// GPU Tensor Operations
GPUTensorOps& GPUTensorOps::instance() {
    static GPUTensorOps instance;
    return instance;
}

void GPUTensorOps::initialize() {
    if (initialized_) return;

    auto& runtime = GPURuntime::instance();
    auto info = runtime.currentDeviceInfo();

    codegen_.initialize(info.compute_capability_major, info.compute_capability_minor);

    // Compile standard kernels
    auto vector_add = codegen_.generateVectorAdd();
    loadKernel(vector_add);
    kernels_["vector_add"] = std::move(vector_add);

    auto matmul = codegen_.generateMatMul(true);
    loadKernel(matmul);
    kernels_["matmul_tiled"] = std::move(matmul);

    auto softmax = codegen_.generateSoftmax();
    loadKernel(softmax);
    kernels_["softmax"] = std::move(softmax);

    initialized_ = true;
}

void GPUTensorOps::loadKernel(CompiledKernel& kernel) {
#ifdef ESHKOL_CUDA_ENABLED
    // Compile PTX to cubin and load
    CUmodule module;
    CUfunction function;

    auto ptx = codegen_.compileToPTX();

    CUjit_option options[] = {CU_JIT_OPTIMIZATION_LEVEL};
    void* option_values[] = {(void*)4};  // Max optimization

    cuModuleLoadDataEx(&module, ptx.c_str(), 1, options, option_values);
    cuModuleGetFunction(&function, module, kernel.name.c_str());

    kernel.cuda_function = function;
#endif
}

LaunchConfig GPUTensorOps::computeLaunchConfig(size_t total_threads, size_t threads_per_block) {
    LaunchConfig config;
    config.block_x = threads_per_block;
    config.block_y = 1;
    config.block_z = 1;
    config.grid_x = (total_threads + threads_per_block - 1) / threads_per_block;
    config.grid_y = 1;
    config.grid_z = 1;
    config.shared_memory_bytes = 0;
    config.stream = nullptr;
    return config;
}

GPUError GPUTensorOps::add(const GPUTensor& a, const GPUTensor& b, GPUTensor& c, GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    auto& kernel = kernels_["vector_add"];

    int n = a.size();
    void* args[] = {
        (void*)&a.devicePtr(),
        (void*)&b.devicePtr(),
        (void*)&c.devicePtr(),
        &n
    };

    auto config = computeLaunchConfig(n);
    config.stream = stream;

    return GPURuntime::instance().launchKernel(kernel.cuda_function, config, args);
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

GPUError GPUTensorOps::matmul(const GPUTensor& A, const GPUTensor& B, GPUTensor& C, GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    // Use cuBLAS for production matmul
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }

    int M = A.shape()[0];
    int K = A.shape()[1];
    int N = B.shape()[1];

    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS uses column-major, so we compute B^T @ A^T = (A @ B)^T
    // Then the result is already in row-major order
    auto status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              B.devicePtr(), N,
                              A.devicePtr(), K,
                              &beta,
                              C.devicePtr(), N);

    if (status != CUBLAS_STATUS_SUCCESS) {
        return {status, "cuBLAS error", "", 0};
    }

    return GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

GPUError GPUTensorOps::softmax(const GPUTensor& a, GPUTensor& b, int axis, GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    // Use cuDNN for softmax
    static cudnnHandle_t handle = nullptr;
    if (!handle) {
        cudnnCreate(&handle);
    }

    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);

    // Set tensor descriptor based on shape
    int n = a.shape()[0];
    int c = (a.rank() > 1) ? a.shape()[1] : 1;
    int h = (a.rank() > 2) ? a.shape()[2] : 1;
    int w = (a.rank() > 3) ? a.shape()[3] : 1;

    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    float alpha = 1.0f;
    float beta = 0.0f;

    auto status = cudnnSoftmaxForward(handle,
                                      CUDNN_SOFTMAX_ACCURATE,
                                      CUDNN_SOFTMAX_MODE_INSTANCE,
                                      &alpha, desc, a.devicePtr(),
                                      &beta, desc, b.devicePtr());

    cudnnDestroyTensorDescriptor(desc);

    if (status != CUDNN_STATUS_SUCCESS) {
        return {status, "cuDNN error", "", 0};
    }

    return GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

GPUError GPUTensorOps::attention(const GPUTensor& Q, const GPUTensor& K, const GPUTensor& V,
                                  GPUTensor& output, const GPUTensor* mask, GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    // Scaled dot-product attention:
    // output = softmax(Q @ K^T / sqrt(d_k)) @ V

    size_t batch = Q.shape()[0];
    size_t seq_len = Q.shape()[1];
    size_t d_k = Q.shape()[2];

    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // Step 1: scores = Q @ K^T
    GPUTensor scores({batch, seq_len, seq_len});
    // [matmul with transpose on K]

    // Step 2: scores = scores / sqrt(d_k)
    // [scale operation]

    // Step 3: Apply mask if provided
    if (mask) {
        // [add large negative value where mask is 0]
    }

    // Step 4: attention_weights = softmax(scores)
    GPUTensor weights({batch, seq_len, seq_len});
    softmax(scores, weights, -1, stream);

    // Step 5: output = attention_weights @ V
    matmul(weights, V, output, stream);

    return GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

} // namespace eshkol::gpu
```

### Day 58-62: Eshkol Language Integration

**Step 4.8**: Add GPU primitives to parser (`lib/frontend/parser.cpp`):

```cpp
// Add to get_operator_type():
if (op == "gpu-tensor") return ESHKOL_GPU_TENSOR_OP;
if (op == "gpu-matmul") return ESHKOL_GPU_MATMUL_OP;
if (op == "gpu-softmax") return ESHKOL_GPU_SOFTMAX_OP;
if (op == "gpu-attention") return ESHKOL_GPU_ATTENTION_OP;
if (op == "gpu-conv2d") return ESHKOL_GPU_CONV2D_OP;
if (op == "to-gpu") return ESHKOL_TO_GPU_OP;
if (op == "to-cpu") return ESHKOL_TO_CPU_OP;
if (op == "gpu-sync") return ESHKOL_GPU_SYNC_OP;
```

**Step 4.9**: Add to `inc/eshkol/eshkol.h`:

```cpp
ESHKOL_GPU_TENSOR_OP,
ESHKOL_GPU_MATMUL_OP,
ESHKOL_GPU_SOFTMAX_OP,
ESHKOL_GPU_ATTENTION_OP,
ESHKOL_GPU_CONV2D_OP,
ESHKOL_TO_GPU_OP,
ESHKOL_TO_CPU_OP,
ESHKOL_GPU_SYNC_OP,
```

### Day 63-68: Testing and Optimization

**Step 4.10**: Create `tests/gpu/gpu_tensor_test.esk`

```scheme
(require stdlib)

(display "=== GPU Tensor Tests ===")
(newline)

;; Test 1: Create GPU tensor
(define a (gpu-tensor '(1.0 2.0 3.0 4.0)))
(define b (gpu-tensor '(5.0 6.0 7.0 8.0)))

(display "Vector add on GPU: ")
(display (to-cpu (gpu-add a b)))
(display " (expected: (6.0 8.0 10.0 12.0))")
(newline)

;; Test 2: Matrix multiplication
(define A (gpu-tensor '((1.0 2.0) (3.0 4.0))))
(define B (gpu-tensor '((5.0 6.0) (7.0 8.0))))

(display "Matrix mul on GPU: ")
(display (to-cpu (gpu-matmul A B)))
(display " (expected: ((19 22) (43 50)))")
(newline)

;; Test 3: Softmax
(define logits (gpu-tensor '(1.0 2.0 3.0 4.0)))
(display "Softmax: ")
(display (to-cpu (gpu-softmax logits)))
(newline)

;; Test 4: Benchmark
(define (benchmark-gpu-matmul size)
  (define M (gpu-tensor (make-random-matrix size size)))
  (define N (gpu-tensor (make-random-matrix size size)))
  (gpu-sync)  ;; Ensure data is on GPU

  (define start (current-milliseconds))
  (define result (gpu-matmul M N))
  (gpu-sync)  ;; Wait for completion
  (define end (current-milliseconds))

  (display "GPU Matrix ")
  (display size)
  (display "x")
  (display size)
  (display " took ")
  (display (- end start))
  (display " ms")
  (newline))

(benchmark-gpu-matmul 1000)
(benchmark-gpu-matmul 2000)
(benchmark-gpu-matmul 4000)

(display "GPU tests complete")
(newline)
```

**Verification Checklist for Phase 4**:
- [ ] GPU runtime initializes correctly
- [ ] Memory allocation/deallocation works
- [ ] Host-to-device and device-to-host copies work
- [ ] Vector operations produce correct results
- [ ] Matrix multiplication matches CPU version
- [ ] Softmax is numerically stable
- [ ] cuBLAS/cuDNN integration works
- [ ] 10x+ speedup on large matrices
- [ ] All tests pass with and without GPU

---

## Phase 5: Neural Network Primitives (15 days)

### Why Fifth?
- Builds on GPU infrastructure from Phase 4
- Core building blocks for ML models
- Enables transformer and CNN architectures
- Required for competitive deep learning support

### Day 69-71: Core Neural Network Operations Header

**Files to Create**:
```
inc/eshkol/nn/nn_ops.h
inc/eshkol/nn/nn_layer.h
lib/nn/nn_ops.cpp
lib/nn/nn_layer.cpp
```

**Step 5.1**: Create `inc/eshkol/nn/nn_ops.h`

```cpp
#ifndef ESHKOL_NN_OPS_H
#define ESHKOL_NN_OPS_H

#include "eshkol/gpu/gpu_tensor.h"
#include <cmath>

namespace eshkol::nn {

using gpu::GPUTensor;
using gpu::GPUError;
using gpu::GPUStream;

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

class Activations {
public:
    // ReLU: max(0, x)
    static GPUError relu(const GPUTensor& input, GPUTensor& output, GPUStream* stream = nullptr);
    static GPUError reluBackward(const GPUTensor& grad_output, const GPUTensor& input,
                                  GPUTensor& grad_input, GPUStream* stream = nullptr);

    // Leaky ReLU: x if x > 0 else alpha * x
    static GPUError leakyRelu(const GPUTensor& input, GPUTensor& output,
                               float alpha = 0.01f, GPUStream* stream = nullptr);
    static GPUError leakyReluBackward(const GPUTensor& grad_output, const GPUTensor& input,
                                       GPUTensor& grad_input, float alpha = 0.01f,
                                       GPUStream* stream = nullptr);

    // GELU: x * Φ(x) where Φ is standard normal CDF
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    static GPUError gelu(const GPUTensor& input, GPUTensor& output, GPUStream* stream = nullptr);
    static GPUError geluBackward(const GPUTensor& grad_output, const GPUTensor& input,
                                  GPUTensor& grad_input, GPUStream* stream = nullptr);

    // SiLU/Swish: x * sigmoid(x)
    static GPUError silu(const GPUTensor& input, GPUTensor& output, GPUStream* stream = nullptr);
    static GPUError siluBackward(const GPUTensor& grad_output, const GPUTensor& input,
                                  GPUTensor& grad_input, GPUStream* stream = nullptr);

    // Softmax: exp(x_i) / sum(exp(x_j)) with numerical stability
    static GPUError softmax(const GPUTensor& input, GPUTensor& output,
                            int axis = -1, GPUStream* stream = nullptr);
    static GPUError softmaxBackward(const GPUTensor& grad_output, const GPUTensor& output,
                                     GPUTensor& grad_input, int axis = -1,
                                     GPUStream* stream = nullptr);

    // Log-Softmax: log(softmax(x)) computed stably
    static GPUError logSoftmax(const GPUTensor& input, GPUTensor& output,
                                int axis = -1, GPUStream* stream = nullptr);
};

// ============================================================================
// NORMALIZATION LAYERS
// ============================================================================

class Normalization {
public:
    // Layer Normalization
    // output = (input - mean) / sqrt(var + eps) * gamma + beta
    static GPUError layerNorm(const GPUTensor& input,
                               const GPUTensor& gamma,
                               const GPUTensor& beta,
                               GPUTensor& output,
                               GPUTensor& mean,      // saved for backward
                               GPUTensor& inv_std,   // saved for backward
                               float eps = 1e-5f,
                               GPUStream* stream = nullptr);

    static GPUError layerNormBackward(const GPUTensor& grad_output,
                                       const GPUTensor& input,
                                       const GPUTensor& gamma,
                                       const GPUTensor& mean,
                                       const GPUTensor& inv_std,
                                       GPUTensor& grad_input,
                                       GPUTensor& grad_gamma,
                                       GPUTensor& grad_beta,
                                       GPUStream* stream = nullptr);

    // Batch Normalization
    static GPUError batchNorm(const GPUTensor& input,
                               const GPUTensor& gamma,
                               const GPUTensor& beta,
                               GPUTensor& running_mean,
                               GPUTensor& running_var,
                               GPUTensor& output,
                               GPUTensor& saved_mean,
                               GPUTensor& saved_inv_std,
                               float eps = 1e-5f,
                               float momentum = 0.1f,
                               bool training = true,
                               GPUStream* stream = nullptr);

    // RMS Normalization (used in LLaMA)
    // output = input / sqrt(mean(input^2) + eps) * gamma
    static GPUError rmsNorm(const GPUTensor& input,
                             const GPUTensor& gamma,
                             GPUTensor& output,
                             float eps = 1e-5f,
                             GPUStream* stream = nullptr);
};

// ============================================================================
// ATTENTION MECHANISMS
// ============================================================================

class Attention {
public:
    // Scaled Dot-Product Attention
    // output = softmax(Q @ K^T / sqrt(d_k)) @ V
    struct AttentionConfig {
        float scale = -1.0f;      // -1 = auto (1/sqrt(d_k))
        bool causal = false;       // Apply causal mask
        float dropout_prob = 0.0f;
        bool return_attention_weights = false;
    };

    static GPUError scaledDotProductAttention(
        const GPUTensor& query,           // [batch, seq_q, d_k]
        const GPUTensor& key,             // [batch, seq_k, d_k]
        const GPUTensor& value,           // [batch, seq_k, d_v]
        GPUTensor& output,                // [batch, seq_q, d_v]
        const AttentionConfig& config = {},
        const GPUTensor* attention_mask = nullptr,  // [batch, seq_q, seq_k] or broadcastable
        GPUTensor* attention_weights = nullptr,     // Optional output
        GPUStream* stream = nullptr);

    // Multi-Head Attention
    // Splits d_model into num_heads, applies attention, concatenates
    struct MultiHeadConfig {
        int num_heads;
        int head_dim;             // d_k = d_v = head_dim
        float dropout_prob = 0.0f;
        bool causal = false;
    };

    static GPUError multiHeadAttention(
        const GPUTensor& query,           // [batch, seq_q, d_model]
        const GPUTensor& key,             // [batch, seq_k, d_model]
        const GPUTensor& value,           // [batch, seq_k, d_model]
        const GPUTensor& W_q,             // [d_model, d_model]
        const GPUTensor& W_k,             // [d_model, d_model]
        const GPUTensor& W_v,             // [d_model, d_model]
        const GPUTensor& W_o,             // [d_model, d_model]
        GPUTensor& output,                // [batch, seq_q, d_model]
        const MultiHeadConfig& config,
        const GPUTensor* attention_mask = nullptr,
        GPUStream* stream = nullptr);

    // Flash Attention (memory-efficient)
    // Uses tiling to avoid materializing full attention matrix
    static GPUError flashAttention(
        const GPUTensor& query,
        const GPUTensor& key,
        const GPUTensor& value,
        GPUTensor& output,
        bool causal = false,
        GPUStream* stream = nullptr);
};

// ============================================================================
// CONVOLUTION OPERATIONS
// ============================================================================

class Convolution {
public:
    struct Conv2dConfig {
        int stride_h = 1, stride_w = 1;
        int padding_h = 0, padding_w = 0;
        int dilation_h = 1, dilation_w = 1;
        int groups = 1;
    };

    // 2D Convolution
    // input:  [batch, in_channels, height, width]
    // kernel: [out_channels, in_channels/groups, kernel_h, kernel_w]
    // output: [batch, out_channels, out_h, out_w]
    static GPUError conv2d(const GPUTensor& input,
                           const GPUTensor& kernel,
                           const GPUTensor* bias,
                           GPUTensor& output,
                           const Conv2dConfig& config = {},
                           GPUStream* stream = nullptr);

    static GPUError conv2dBackwardData(const GPUTensor& grad_output,
                                        const GPUTensor& kernel,
                                        GPUTensor& grad_input,
                                        const Conv2dConfig& config = {},
                                        GPUStream* stream = nullptr);

    static GPUError conv2dBackwardFilter(const GPUTensor& grad_output,
                                          const GPUTensor& input,
                                          GPUTensor& grad_kernel,
                                          const Conv2dConfig& config = {},
                                          GPUStream* stream = nullptr);

    // 1D Convolution (for sequences)
    static GPUError conv1d(const GPUTensor& input,      // [batch, in_channels, length]
                           const GPUTensor& kernel,     // [out_channels, in_channels, kernel_size]
                           const GPUTensor* bias,
                           GPUTensor& output,
                           int stride = 1, int padding = 0, int dilation = 1,
                           GPUStream* stream = nullptr);

    // Depthwise Separable Convolution
    static GPUError depthwiseConv2d(const GPUTensor& input,
                                     const GPUTensor& depthwise_kernel,
                                     const GPUTensor& pointwise_kernel,
                                     GPUTensor& output,
                                     const Conv2dConfig& config = {},
                                     GPUStream* stream = nullptr);
};

// ============================================================================
// POOLING OPERATIONS
// ============================================================================

class Pooling {
public:
    struct Pool2dConfig {
        int kernel_h, kernel_w;
        int stride_h = -1, stride_w = -1;  // -1 = same as kernel
        int padding_h = 0, padding_w = 0;
    };

    // Max Pooling
    static GPUError maxPool2d(const GPUTensor& input,
                               GPUTensor& output,
                               GPUTensor* indices,  // For backward pass
                               const Pool2dConfig& config,
                               GPUStream* stream = nullptr);

    static GPUError maxPool2dBackward(const GPUTensor& grad_output,
                                       const GPUTensor& indices,
                                       GPUTensor& grad_input,
                                       const Pool2dConfig& config,
                                       GPUStream* stream = nullptr);

    // Average Pooling
    static GPUError avgPool2d(const GPUTensor& input,
                               GPUTensor& output,
                               const Pool2dConfig& config,
                               bool count_include_pad = true,
                               GPUStream* stream = nullptr);

    // Global Average Pooling (reduce spatial dimensions to 1x1)
    static GPUError globalAvgPool2d(const GPUTensor& input,  // [batch, channels, h, w]
                                     GPUTensor& output,       // [batch, channels, 1, 1]
                                     GPUStream* stream = nullptr);

    // Adaptive Average Pooling (output to target size)
    static GPUError adaptiveAvgPool2d(const GPUTensor& input,
                                       GPUTensor& output,
                                       int output_h, int output_w,
                                       GPUStream* stream = nullptr);
};

// ============================================================================
// DROPOUT
// ============================================================================

class Dropout {
public:
    // Standard dropout
    // During training: randomly zero elements with probability p, scale by 1/(1-p)
    // During inference: identity
    static GPUError dropout(const GPUTensor& input,
                            GPUTensor& output,
                            GPUTensor& mask,        // Saved for backward
                            float p,
                            bool training,
                            uint64_t seed = 0,
                            GPUStream* stream = nullptr);

    static GPUError dropoutBackward(const GPUTensor& grad_output,
                                     const GPUTensor& mask,
                                     GPUTensor& grad_input,
                                     float p,
                                     GPUStream* stream = nullptr);
};

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

class Loss {
public:
    // Cross-Entropy Loss with Softmax
    // input: [batch, num_classes] (logits)
    // target: [batch] (class indices)
    static GPUError crossEntropyLoss(const GPUTensor& input,
                                      const GPUTensor& target,
                                      GPUTensor& loss,           // scalar or [batch]
                                      GPUTensor& grad_input,     // gradient
                                      bool reduce_mean = true,
                                      GPUStream* stream = nullptr);

    // Binary Cross-Entropy with Sigmoid
    static GPUError binaryCrossEntropyWithLogits(const GPUTensor& input,
                                                   const GPUTensor& target,
                                                   GPUTensor& loss,
                                                   GPUTensor& grad_input,
                                                   bool reduce_mean = true,
                                                   GPUStream* stream = nullptr);

    // Mean Squared Error
    static GPUError mseLoss(const GPUTensor& input,
                            const GPUTensor& target,
                            GPUTensor& loss,
                            GPUTensor& grad_input,
                            bool reduce_mean = true,
                            GPUStream* stream = nullptr);

    // Smooth L1 Loss (Huber Loss)
    static GPUError smoothL1Loss(const GPUTensor& input,
                                  const GPUTensor& target,
                                  GPUTensor& loss,
                                  GPUTensor& grad_input,
                                  float beta = 1.0f,
                                  GPUStream* stream = nullptr);
};

} // namespace eshkol::nn

#endif
```

### Day 72-74: Core Operations Implementation

**Step 5.2**: Create `lib/nn/nn_ops.cpp`

```cpp
#include "eshkol/nn/nn_ops.h"
#include "eshkol/gpu/gpu_codegen.h"

#ifdef ESHKOL_CUDA_ENABLED
#include <cudnn.h>
#include <curand.h>
#endif

namespace eshkol::nn {

// ============================================================================
// ACTIVATION IMPLEMENTATIONS
// ============================================================================

GPUError Activations::gelu(const GPUTensor& input, GPUTensor& output, GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    // GELU kernel
    // y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    static gpu::CompiledKernel* kernel = nullptr;

    if (!kernel) {
        gpu::GPUCodegen codegen;
        codegen.initialize(7, 0);  // SM 7.0+

        auto float_ptr = llvm::PointerType::get(
            llvm::Type::getFloatTy(codegen.context()), 0);
        auto int_type = llvm::Type::getInt32Ty(codegen.context());

        std::vector<gpu::KernelParam> params = {
            {"input", float_ptr, true, true},
            {"output", float_ptr, true, false},
            {"n", int_type, false, true}
        };

        codegen.beginKernel("gelu_forward", params);

        auto& builder = codegen.builder();
        auto& ctx = codegen.context();

        auto i = codegen.getGlobalIdX();
        auto n = codegen.builder().GetInsertBlock()->getParent()->getArg(2);

        auto in_bounds = builder.CreateICmpULT(i, n);
        auto then_bb = llvm::BasicBlock::Create(ctx, "then",
            builder.GetInsertBlock()->getParent());
        auto end_bb = llvm::BasicBlock::Create(ctx, "end",
            builder.GetInsertBlock()->getParent());

        builder.CreateCondBr(in_bounds, then_bb, end_bb);
        builder.SetInsertPoint(then_bb);

        auto float_type = llvm::Type::getFloatTy(ctx);
        auto input_ptr = builder.GetInsertBlock()->getParent()->getArg(0);
        auto output_ptr = builder.GetInsertBlock()->getParent()->getArg(1);

        auto x_ptr = builder.CreateGEP(float_type, input_ptr, i);
        auto x = builder.CreateLoad(float_type, x_ptr);

        // Constants
        auto sqrt_2_over_pi = llvm::ConstantFP::get(float_type, 0.7978845608f);
        auto coeff = llvm::ConstantFP::get(float_type, 0.044715f);
        auto half = llvm::ConstantFP::get(float_type, 0.5f);
        auto one = llvm::ConstantFP::get(float_type, 1.0f);

        // x^3
        auto x2 = builder.CreateFMul(x, x);
        auto x3 = builder.CreateFMul(x2, x);

        // x + 0.044715 * x^3
        auto inner = codegen.fma(coeff, x3, x);

        // sqrt(2/π) * inner
        auto scaled = builder.CreateFMul(sqrt_2_over_pi, inner);

        // tanh
        auto tanh_intrinsic = llvm::Intrinsic::getDeclaration(
            builder.GetInsertBlock()->getModule(),
            llvm::Intrinsic::nvvm_tanh_approx_f);
        auto tanh_val = builder.CreateCall(tanh_intrinsic, {scaled});

        // 1 + tanh(...)
        auto one_plus_tanh = builder.CreateFAdd(one, tanh_val);

        // 0.5 * x * (1 + tanh(...))
        auto result = builder.CreateFMul(half, builder.CreateFMul(x, one_plus_tanh));

        auto out_ptr = builder.CreateGEP(float_type, output_ptr, i);
        builder.CreateStore(result, out_ptr);

        builder.CreateBr(end_bb);
        builder.SetInsertPoint(end_bb);

        auto compiled = codegen.endKernel();
        // ... load kernel
    }

    // Launch kernel
    int n = input.size();
    auto config = gpu::GPUTensorOps::instance().computeLaunchConfig(n);
    // ... launch

    return gpu::GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

GPUError Activations::softmax(const GPUTensor& input, GPUTensor& output,
                               int axis, GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    // Use cuDNN for optimized softmax
    static cudnnHandle_t handle = nullptr;
    if (!handle) {
        cudnnCreate(&handle);
    }

    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);

    // Handle negative axis
    int rank = input.rank();
    if (axis < 0) axis = rank + axis;

    // Set up descriptor based on input shape
    // Softmax is computed over the specified axis
    std::vector<int> dims(4, 1);
    std::vector<int> strides(4, 1);

    // Map to NCHW format for cuDNN
    size_t outer = 1, inner = 1, dim_size = 1;
    for (int i = 0; i < axis; i++) outer *= input.shape()[i];
    dim_size = input.shape()[axis];
    for (int i = axis + 1; i < rank; i++) inner *= input.shape()[i];

    dims[0] = outer;
    dims[1] = dim_size;
    dims[2] = inner;
    dims[3] = 1;

    strides[0] = dim_size * inner;
    strides[1] = inner;
    strides[2] = 1;
    strides[3] = 1;

    cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 4, dims.data(), strides.data());

    float alpha = 1.0f, beta = 0.0f;

    auto status = cudnnSoftmaxForward(
        handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,  // Softmax over dim[1]
        &alpha, desc, input.devicePtr(),
        &beta, desc, output.devicePtr());

    cudnnDestroyTensorDescriptor(desc);

    if (status != CUDNN_STATUS_SUCCESS) {
        return {status, "cuDNN softmax failed", "", 0};
    }

    return gpu::GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

// ============================================================================
// NORMALIZATION IMPLEMENTATIONS
// ============================================================================

GPUError Normalization::layerNorm(const GPUTensor& input,
                                   const GPUTensor& gamma,
                                   const GPUTensor& beta,
                                   GPUTensor& output,
                                   GPUTensor& mean,
                                   GPUTensor& inv_std,
                                   float eps,
                                   GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    // Layer norm: normalize over last dimension(s)
    // Shape: [batch, ..., normalized_shape]

    size_t batch_size = 1;
    size_t norm_size = gamma.size();

    // Calculate batch size (all dimensions except last)
    for (size_t i = 0; i < input.rank() - 1; i++) {
        batch_size *= input.shape()[i];
    }

    // Custom kernel for layer norm
    // Each block handles one sample
    // Step 1: Compute mean
    // Step 2: Compute variance
    // Step 3: Normalize and scale

    // For now, use a simple implementation
    // Production would use Welford's algorithm for numerical stability

    static gpu::CompiledKernel* kernel = nullptr;

    if (!kernel) {
        gpu::GPUCodegen codegen;
        codegen.initialize(7, 0);

        auto float_ptr = llvm::PointerType::get(
            llvm::Type::getFloatTy(codegen.context()), 0);
        auto int_type = llvm::Type::getInt32Ty(codegen.context());
        auto float_type = llvm::Type::getFloatTy(codegen.context());

        std::vector<gpu::KernelParam> params = {
            {"input", float_ptr, true, true},
            {"gamma", float_ptr, true, true},
            {"beta", float_ptr, true, true},
            {"output", float_ptr, true, false},
            {"mean", float_ptr, true, false},
            {"inv_std", float_ptr, true, false},
            {"batch_size", int_type, false, true},
            {"norm_size", int_type, false, true},
            {"eps", float_type, false, true}
        };

        codegen.beginKernel("layer_norm_forward", params);

        auto& builder = codegen.builder();

        // Each block handles one sample
        auto batch_idx = codegen.getBlockIdX();
        auto tid = codegen.getThreadIdX();

        // Allocate shared memory for reduction
        auto shared_sum = codegen.allocateSharedMemory(float_type, 256, "shared_sum");
        auto shared_sq_sum = codegen.allocateSharedMemory(float_type, 256, "shared_sq_sum");

        // [Implementation: parallel reduction for mean, then variance,
        //  then normalize each element]

        codegen.endKernel();
    }

    // Launch kernel
    gpu::LaunchConfig config;
    config.grid_x = batch_size;
    config.block_x = std::min((size_t)256, norm_size);
    config.shared_memory_bytes = 256 * 2 * sizeof(float);

    // ... launch kernel

    return gpu::GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

// ============================================================================
// ATTENTION IMPLEMENTATIONS
// ============================================================================

GPUError Attention::scaledDotProductAttention(
    const GPUTensor& query,
    const GPUTensor& key,
    const GPUTensor& value,
    GPUTensor& output,
    const AttentionConfig& config,
    const GPUTensor* attention_mask,
    GPUTensor* attention_weights,
    GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    // Q: [batch, seq_q, d_k]
    // K: [batch, seq_k, d_k]
    // V: [batch, seq_k, d_v]

    size_t batch = query.shape()[0];
    size_t seq_q = query.shape()[1];
    size_t d_k = query.shape()[2];
    size_t seq_k = key.shape()[1];
    size_t d_v = value.shape()[2];

    // Scale factor
    float scale = config.scale > 0 ? config.scale : 1.0f / std::sqrt(static_cast<float>(d_k));

    // Step 1: Compute scores = Q @ K^T
    // scores: [batch, seq_q, seq_k]
    GPUTensor scores({batch, seq_q, seq_k});

    // Use cuBLAS batched GEMM
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }

    // K^T has shape [batch, d_k, seq_k]
    // scores = Q @ K^T => [batch, seq_q, d_k] @ [batch, d_k, seq_k]

    // cuBLAS strided batched GEMM
    float alpha_blas = scale;
    float beta_blas = 0.0f;

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,  // K transposed, Q not
        seq_k, seq_q, d_k,         // m, n, k
        &alpha_blas,
        key.devicePtr(), d_k, seq_k * d_k,    // K: [batch, seq_k, d_k]
        query.devicePtr(), d_k, seq_q * d_k,  // Q: [batch, seq_q, d_k]
        &beta_blas,
        scores.devicePtr(), seq_k, seq_q * seq_k,  // scores: [batch, seq_q, seq_k]
        batch
    );

    // Step 2: Apply mask if provided
    if (attention_mask) {
        // Add mask (large negative values where mask is 0)
        // [kernel to add mask]
    }

    // Step 3: Apply causal mask if needed
    if (config.causal) {
        // Create lower triangular mask
        // [kernel to apply causal mask]
    }

    // Step 4: Softmax over last dimension
    GPUTensor attention_probs({batch, seq_q, seq_k});
    Activations::softmax(scores, attention_probs, -1, stream);

    // Step 5: Optional dropout
    GPUTensor dropout_mask({});
    if (config.dropout_prob > 0.0f) {
        Dropout::dropout(attention_probs, attention_probs, dropout_mask,
                         config.dropout_prob, true, 0, stream);
    }

    // Step 6: Compute output = attention_probs @ V
    // [batch, seq_q, seq_k] @ [batch, seq_k, d_v] => [batch, seq_q, d_v]
    float alpha_out = 1.0f;
    float beta_out = 0.0f;

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        d_v, seq_q, seq_k,
        &alpha_out,
        value.devicePtr(), d_v, seq_k * d_v,
        attention_probs.devicePtr(), seq_k, seq_q * seq_k,
        &beta_out,
        output.devicePtr(), d_v, seq_q * d_v,
        batch
    );

    // Return attention weights if requested
    if (attention_weights && config.return_attention_weights) {
        gpu::GPURuntime::instance().memcpyDeviceToDevice(
            *reinterpret_cast<gpu::GPUBuffer*>(attention_weights),
            *reinterpret_cast<const gpu::GPUBuffer*>(&attention_probs),
            attention_probs.sizeInBytes(), stream);
    }

    return gpu::GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

// ============================================================================
// CONVOLUTION IMPLEMENTATIONS
// ============================================================================

GPUError Convolution::conv2d(const GPUTensor& input,
                              const GPUTensor& kernel,
                              const GPUTensor* bias,
                              GPUTensor& output,
                              const Conv2dConfig& config,
                              GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    static cudnnHandle_t handle = nullptr;
    if (!handle) {
        cudnnCreate(&handle);
    }

    // Create descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Input: [batch, in_channels, height, width]
    int batch = input.shape()[0];
    int in_channels = input.shape()[1];
    int in_h = input.shape()[2];
    int in_w = input.shape()[3];

    // Kernel: [out_channels, in_channels/groups, kernel_h, kernel_w]
    int out_channels = kernel.shape()[0];
    int kernel_h = kernel.shape()[2];
    int kernel_w = kernel.shape()[3];

    // Set input descriptor
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batch, in_channels, in_h, in_w);

    // Set filter descriptor
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               out_channels, in_channels / config.groups, kernel_h, kernel_w);

    // Set convolution descriptor
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    config.padding_h, config.padding_w,
                                    config.stride_h, config.stride_w,
                                    config.dilation_h, config.dilation_w,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    if (config.groups > 1) {
        cudnnSetConvolutionGroupCount(conv_desc, config.groups);
    }

    // Get output dimensions
    int out_n, out_c, out_h, out_w;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
                                          &out_n, &out_c, &out_h, &out_w);

    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               out_n, out_c, out_h, out_w);

    // Find best algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm_v7(
        handle, input_desc, filter_desc, conv_desc, output_desc,
        1, nullptr, &algo);

    // Get workspace size
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc,
                                            conv_desc, output_desc, algo, &workspace_size);

    // Allocate workspace
    gpu::GPUBuffer workspace;
    if (workspace_size > 0) {
        gpu::GPURuntime::instance().allocate(workspace, workspace_size);
    }

    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    auto status = cudnnConvolutionForward(
        handle,
        &alpha, input_desc, input.devicePtr(),
        filter_desc, kernel.devicePtr(),
        conv_desc, algo,
        workspace.device_ptr, workspace_size,
        &beta, output_desc, output.devicePtr());

    // Add bias if provided
    if (bias) {
        cudnnTensorDescriptor_t bias_desc;
        cudnnCreateTensorDescriptor(&bias_desc);
        cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, out_channels, 1, 1);

        float alpha_bias = 1.0f, beta_bias = 1.0f;
        cudnnAddTensor(handle, &alpha_bias, bias_desc, bias->devicePtr(),
                       &beta_bias, output_desc, output.devicePtr());

        cudnnDestroyTensorDescriptor(bias_desc);
    }

    // Cleanup
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    if (workspace_size > 0) {
        gpu::GPURuntime::instance().free(workspace);
    }

    if (status != CUDNN_STATUS_SUCCESS) {
        return {status, "cuDNN conv2d failed", "", 0};
    }

    return gpu::GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

// ============================================================================
// POOLING IMPLEMENTATIONS
// ============================================================================

GPUError Pooling::maxPool2d(const GPUTensor& input,
                             GPUTensor& output,
                             GPUTensor* indices,
                             const Pool2dConfig& config,
                             GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    static cudnnHandle_t handle = nullptr;
    if (!handle) {
        cudnnCreate(&handle);
    }

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnPoolingDescriptor_t pool_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreatePoolingDescriptor(&pool_desc);

    int batch = input.shape()[0];
    int channels = input.shape()[1];
    int in_h = input.shape()[2];
    int in_w = input.shape()[3];

    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batch, channels, in_h, in_w);

    int stride_h = config.stride_h > 0 ? config.stride_h : config.kernel_h;
    int stride_w = config.stride_w > 0 ? config.stride_w : config.kernel_w;

    cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                                config.kernel_h, config.kernel_w,
                                config.padding_h, config.padding_w,
                                stride_h, stride_w);

    int out_n, out_c, out_h, out_w;
    cudnnGetPooling2dForwardOutputDim(pool_desc, input_desc, &out_n, &out_c, &out_h, &out_w);

    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               out_n, out_c, out_h, out_w);

    float alpha = 1.0f, beta = 0.0f;
    auto status = cudnnPoolingForward(handle, pool_desc,
                                      &alpha, input_desc, input.devicePtr(),
                                      &beta, output_desc, output.devicePtr());

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyPoolingDescriptor(pool_desc);

    if (status != CUDNN_STATUS_SUCCESS) {
        return {status, "cuDNN maxpool failed", "", 0};
    }

    return gpu::GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

// ============================================================================
// DROPOUT IMPLEMENTATION
// ============================================================================

GPUError Dropout::dropout(const GPUTensor& input,
                          GPUTensor& output,
                          GPUTensor& mask,
                          float p,
                          bool training,
                          uint64_t seed,
                          GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    if (!training || p == 0.0f) {
        // Identity during inference
        gpu::GPURuntime::instance().memcpyDeviceToDevice(
            *reinterpret_cast<gpu::GPUBuffer*>(&output),
            *reinterpret_cast<const gpu::GPUBuffer*>(&input),
            input.sizeInBytes(), stream);
        return gpu::GPUError::success();
    }

    // Generate random mask
    static curandGenerator_t gen = nullptr;
    if (!gen) {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed ? seed : time(nullptr));
    }

    // Generate uniform random numbers in mask
    curandGenerateUniform(gen, static_cast<float*>(mask.devicePtr()), mask.size());

    // Apply dropout: output = input * (mask > p) / (1 - p)
    // [Custom kernel]

    return gpu::GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

// ============================================================================
// LOSS IMPLEMENTATIONS
// ============================================================================

GPUError Loss::crossEntropyLoss(const GPUTensor& input,
                                 const GPUTensor& target,
                                 GPUTensor& loss,
                                 GPUTensor& grad_input,
                                 bool reduce_mean,
                                 GPUStream* stream) {
#ifdef ESHKOL_CUDA_ENABLED
    // Fused softmax + cross-entropy for numerical stability
    // loss = -log(softmax(input)[target])
    // grad = softmax(input) - one_hot(target)

    size_t batch = input.shape()[0];
    size_t num_classes = input.shape()[1];

    // Custom kernel for fused operation
    // [Implementation with log-sum-exp trick for stability]

    return gpu::GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

} // namespace eshkol::nn
```

### Day 75-78: Optimizers

**Files to Create**:
```
inc/eshkol/nn/optimizer.h
lib/nn/optimizer.cpp
```

**Step 5.3**: Create `inc/eshkol/nn/optimizer.h`

```cpp
#ifndef ESHKOL_OPTIMIZER_H
#define ESHKOL_OPTIMIZER_H

#include "eshkol/gpu/gpu_tensor.h"
#include <vector>
#include <unordered_map>

namespace eshkol::nn {

using gpu::GPUTensor;
using gpu::GPUError;

// Parameter group for different learning rates
struct ParamGroup {
    std::vector<GPUTensor*> params;
    std::vector<GPUTensor*> grads;
    float lr = 0.001f;
    float weight_decay = 0.0f;
};

// Base Optimizer class
class Optimizer {
public:
    virtual ~Optimizer() = default;

    // Perform one optimization step
    virtual GPUError step() = 0;

    // Zero all gradients
    virtual void zeroGrad();

    // Add parameters to optimize
    void addParamGroup(ParamGroup group) { param_groups_.push_back(std::move(group)); }

    // State dict for checkpointing
    virtual void loadState(const std::unordered_map<std::string, GPUTensor>& state) {}
    virtual std::unordered_map<std::string, GPUTensor> saveState() const { return {}; }

protected:
    std::vector<ParamGroup> param_groups_;
    int step_count_ = 0;
};

// Stochastic Gradient Descent
class SGD : public Optimizer {
public:
    SGD(float lr = 0.01f, float momentum = 0.0f, float dampening = 0.0f,
        float weight_decay = 0.0f, bool nesterov = false);

    GPUError step() override;

private:
    float momentum_;
    float dampening_;
    float weight_decay_;
    bool nesterov_;

    // Momentum buffers (one per parameter)
    std::vector<GPUTensor> momentum_buffers_;
};

// Adam Optimizer
class Adam : public Optimizer {
public:
    Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.0f, bool amsgrad = false);

    GPUError step() override;

    void loadState(const std::unordered_map<std::string, GPUTensor>& state) override;
    std::unordered_map<std::string, GPUTensor> saveState() const override;

private:
    float beta1_, beta2_;
    float eps_;
    float weight_decay_;
    bool amsgrad_;

    // First moment (mean)
    std::vector<GPUTensor> m_;
    // Second moment (variance)
    std::vector<GPUTensor> v_;
    // Max second moment (for AMSGrad)
    std::vector<GPUTensor> v_max_;
};

// AdamW (Adam with decoupled weight decay)
class AdamW : public Optimizer {
public:
    AdamW(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 0.01f);

    GPUError step() override;

private:
    float beta1_, beta2_;
    float eps_;
    float weight_decay_;

    std::vector<GPUTensor> m_;
    std::vector<GPUTensor> v_;
};

// LAMB Optimizer (for large batch training)
class LAMB : public Optimizer {
public:
    LAMB(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-6f, float weight_decay = 0.0f);

    GPUError step() override;

private:
    float beta1_, beta2_;
    float eps_;
    float weight_decay_;

    std::vector<GPUTensor> m_;
    std::vector<GPUTensor> v_;
};

// Learning Rate Schedulers
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float getLR(int step) = 0;
};

class StepLR : public LRScheduler {
public:
    StepLR(float initial_lr, int step_size, float gamma = 0.1f);
    float getLR(int step) override;

private:
    float initial_lr_;
    int step_size_;
    float gamma_;
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(float initial_lr, int T_max, float eta_min = 0.0f);
    float getLR(int step) override;

private:
    float initial_lr_;
    int T_max_;
    float eta_min_;
};

class WarmupLR : public LRScheduler {
public:
    WarmupLR(float initial_lr, int warmup_steps, LRScheduler* after_warmup);
    float getLR(int step) override;

private:
    float initial_lr_;
    int warmup_steps_;
    std::unique_ptr<LRScheduler> after_warmup_;
};

} // namespace eshkol::nn

#endif
```

**Step 5.4**: Create `lib/nn/optimizer.cpp`

```cpp
#include "eshkol/nn/optimizer.h"
#include <cmath>

namespace eshkol::nn {

void Optimizer::zeroGrad() {
    for (auto& group : param_groups_) {
        for (auto* grad : group.grads) {
            if (grad) {
                gpu::GPURuntime::instance().memset(
                    *reinterpret_cast<gpu::GPUBuffer*>(grad), 0, grad->sizeInBytes());
            }
        }
    }
}

// SGD Implementation
SGD::SGD(float lr, float momentum, float dampening, float weight_decay, bool nesterov)
    : momentum_(momentum), dampening_(dampening), weight_decay_(weight_decay), nesterov_(nesterov) {
    ParamGroup default_group;
    default_group.lr = lr;
    param_groups_.push_back(default_group);
}

GPUError SGD::step() {
#ifdef ESHKOL_CUDA_ENABLED
    // CUDA kernel for SGD update
    // p = p - lr * (grad + weight_decay * p)
    // With momentum: v = momentum * v + grad; p = p - lr * v
    // With Nesterov: p = p - lr * (momentum * v + grad)

    for (size_t g = 0; g < param_groups_.size(); g++) {
        auto& group = param_groups_[g];
        float lr = group.lr;

        for (size_t i = 0; i < group.params.size(); i++) {
            auto* param = group.params[i];
            auto* grad = group.grads[i];

            if (!grad) continue;

            // Weight decay
            if (weight_decay_ != 0.0f) {
                // grad = grad + weight_decay * param
                // [CUDA kernel]
            }

            // Momentum
            if (momentum_ != 0.0f) {
                // Initialize momentum buffer if needed
                if (momentum_buffers_.size() <= i) {
                    momentum_buffers_.emplace_back(param->shape(), true);
                    gpu::GPURuntime::instance().memset(
                        *reinterpret_cast<gpu::GPUBuffer*>(&momentum_buffers_.back()),
                        0, momentum_buffers_.back().sizeInBytes());
                }

                auto& buf = momentum_buffers_[i];
                // buf = momentum * buf + (1 - dampening) * grad
                // [CUDA kernel]

                if (nesterov_) {
                    // grad = grad + momentum * buf
                }
            }

            // param = param - lr * grad
            // [CUDA kernel: axpy]
        }
    }

    step_count_++;
    return gpu::GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

// Adam Implementation
Adam::Adam(float lr, float beta1, float beta2, float eps, float weight_decay, bool amsgrad)
    : beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay), amsgrad_(amsgrad) {
    ParamGroup default_group;
    default_group.lr = lr;
    param_groups_.push_back(default_group);
}

GPUError Adam::step() {
#ifdef ESHKOL_CUDA_ENABLED
    step_count_++;

    // Bias correction
    float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
    float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);

    for (size_t g = 0; g < param_groups_.size(); g++) {
        auto& group = param_groups_[g];
        float lr = group.lr;

        for (size_t i = 0; i < group.params.size(); i++) {
            auto* param = group.params[i];
            auto* grad = group.grads[i];

            if (!grad) continue;

            // Initialize state if needed
            size_t idx = g * 1000 + i;  // Unique index
            if (m_.size() <= idx) {
                m_.resize(idx + 1);
                v_.resize(idx + 1);
                if (amsgrad_) v_max_.resize(idx + 1);

                m_[idx] = GPUTensor(param->shape(), true);
                v_[idx] = GPUTensor(param->shape(), true);
                gpu::GPURuntime::instance().memset(
                    *reinterpret_cast<gpu::GPUBuffer*>(&m_[idx]), 0, m_[idx].sizeInBytes());
                gpu::GPURuntime::instance().memset(
                    *reinterpret_cast<gpu::GPUBuffer*>(&v_[idx]), 0, v_[idx].sizeInBytes());

                if (amsgrad_) {
                    v_max_[idx] = GPUTensor(param->shape(), true);
                    gpu::GPURuntime::instance().memset(
                        *reinterpret_cast<gpu::GPUBuffer*>(&v_max_[idx]), 0, v_max_[idx].sizeInBytes());
                }
            }

            auto& m = m_[idx];
            auto& v = v_[idx];

            // m = beta1 * m + (1 - beta1) * grad
            // v = beta2 * v + (1 - beta2) * grad^2

            // Bias-corrected estimates
            // m_hat = m / bias_correction1
            // v_hat = v / bias_correction2

            // Update: param = param - lr * m_hat / (sqrt(v_hat) + eps)
            // With weight decay (L2): grad = grad + weight_decay * param

            // [CUDA fused kernel for efficiency]
        }
    }

    return gpu::GPUError::success();
#else
    return {-1, "CUDA not enabled", "", 0};
#endif
}

// Learning Rate Schedulers
StepLR::StepLR(float initial_lr, int step_size, float gamma)
    : initial_lr_(initial_lr), step_size_(step_size), gamma_(gamma) {}

float StepLR::getLR(int step) {
    return initial_lr_ * std::pow(gamma_, step / step_size_);
}

CosineAnnealingLR::CosineAnnealingLR(float initial_lr, int T_max, float eta_min)
    : initial_lr_(initial_lr), T_max_(T_max), eta_min_(eta_min) {}

float CosineAnnealingLR::getLR(int step) {
    return eta_min_ + (initial_lr_ - eta_min_) *
           (1.0f + std::cos(M_PI * step / T_max_)) / 2.0f;
}

WarmupLR::WarmupLR(float initial_lr, int warmup_steps, LRScheduler* after_warmup)
    : initial_lr_(initial_lr), warmup_steps_(warmup_steps),
      after_warmup_(after_warmup) {}

float WarmupLR::getLR(int step) {
    if (step < warmup_steps_) {
        return initial_lr_ * (step + 1) / warmup_steps_;
    }
    return after_warmup_->getLR(step - warmup_steps_);
}

} // namespace eshkol::nn
```

### Day 79-83: Eshkol Language Integration

**Step 5.5**: Add NN primitives to parser (`lib/frontend/parser.cpp`):

```cpp
// Add to get_operator_type():
if (op == "nn-relu") return ESHKOL_NN_RELU_OP;
if (op == "nn-gelu") return ESHKOL_NN_GELU_OP;
if (op == "nn-softmax") return ESHKOL_NN_SOFTMAX_OP;
if (op == "nn-layer-norm") return ESHKOL_NN_LAYER_NORM_OP;
if (op == "nn-attention") return ESHKOL_NN_ATTENTION_OP;
if (op == "nn-multi-head-attention") return ESHKOL_NN_MHA_OP;
if (op == "nn-conv2d") return ESHKOL_NN_CONV2D_OP;
if (op == "nn-max-pool2d") return ESHKOL_NN_MAXPOOL2D_OP;
if (op == "nn-dropout") return ESHKOL_NN_DROPOUT_OP;
if (op == "nn-cross-entropy") return ESHKOL_NN_CROSS_ENTROPY_OP;
if (op == "nn-mse-loss") return ESHKOL_NN_MSE_LOSS_OP;
if (op == "optimizer-sgd") return ESHKOL_OPT_SGD_OP;
if (op == "optimizer-adam") return ESHKOL_OPT_ADAM_OP;
if (op == "optimizer-step") return ESHKOL_OPT_STEP_OP;
if (op == "optimizer-zero-grad") return ESHKOL_OPT_ZERO_GRAD_OP;
```

**Step 5.6**: Create `tests/nn/nn_ops_test.esk`

```scheme
(require stdlib)

(display "=== Neural Network Primitives Tests ===")
(newline)

;; Test 1: Activation functions
(define x (gpu-tensor '(-2.0 -1.0 0.0 1.0 2.0)))

(display "ReLU: ")
(display (to-cpu (nn-relu x)))
(display " (expected: (0 0 0 1 2))")
(newline)

(display "GELU: ")
(display (to-cpu (nn-gelu x)))
(newline)

;; Test 2: Softmax
(define logits (gpu-tensor '(1.0 2.0 3.0)))
(display "Softmax: ")
(display (to-cpu (nn-softmax logits)))
(newline)

;; Test 3: Layer Normalization
(define input (gpu-tensor '((1.0 2.0 3.0) (4.0 5.0 6.0))))
(define gamma (gpu-tensor '(1.0 1.0 1.0)))
(define beta (gpu-tensor '(0.0 0.0 0.0)))
(display "Layer Norm: ")
(display (to-cpu (nn-layer-norm input gamma beta)))
(newline)

;; Test 4: Simple MLP forward pass
(define (mlp-forward x W1 b1 W2 b2)
  (let* ((h1 (nn-relu (tensor-add (tensor-matmul x W1) b1)))
         (out (tensor-add (tensor-matmul h1 W2) b2)))
    out))

;; Test 5: Attention
(define Q (gpu-tensor (make-random-tensor '(2 4 8))))  ; batch=2, seq=4, d_k=8
(define K (gpu-tensor (make-random-tensor '(2 4 8))))
(define V (gpu-tensor (make-random-tensor '(2 4 8))))

(display "Attention output shape: ")
(display (tensor-shape (nn-attention Q K V)))
(display " (expected: (2 4 8))")
(newline)

;; Test 6: Conv2d
(define img (gpu-tensor (make-random-tensor '(1 3 28 28))))  ; batch=1, C=3, H=28, W=28
(define kernel (gpu-tensor (make-random-tensor '(16 3 3 3))))  ; out=16, in=3, k=3x3

(display "Conv2d output shape: ")
(display (tensor-shape (nn-conv2d img kernel)))
(display " (expected: (1 16 26 26))")
(newline)

;; Test 7: Training loop example
(display "\\n=== Training Example ===\\n")

(define (train-step model optimizer x y)
  (optimizer-zero-grad optimizer)
  (let* ((pred (model x))
         (loss (nn-mse-loss pred y)))
    (backward loss)  ; Autodiff backward pass
    (optimizer-step optimizer)
    loss))

(display "Neural network tests complete!")
(newline)
```

**Verification Checklist for Phase 5**:
- [ ] GELU, ReLU, Leaky ReLU produce correct results
- [ ] Softmax is numerically stable (no NaN/Inf)
- [ ] Layer norm matches PyTorch output
- [ ] Attention produces correct output shapes
- [ ] Multi-head attention splits heads correctly
- [ ] Conv2d matches cuDNN output
- [ ] Pooling produces correct dimensions
- [ ] Cross-entropy loss gradient is correct
- [ ] Adam optimizer converges on simple problem
- [ ] All operations work with autodiff backward pass

---

## Phase 6: OALR Memory Completion (5 days)

### Why Sixth?
- Parser already handles OALR syntax
- Type checker has linear type infrastructure
- Just need to complete code generation
- Enables safe, deterministic memory management

### Day 84-85: Region Codegen

**Files to Modify**:
```
lib/backend/llvm_codegen.cpp
lib/backend/region_codegen.cpp (new)
inc/eshkol/backend/region_codegen.h (new)
```

**Step 6.1**: Create `inc/eshkol/backend/region_codegen.h`

```cpp
#ifndef ESHKOL_REGION_CODEGEN_H
#define ESHKOL_REGION_CODEGEN_H

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Module.h>
#include <stack>
#include <unordered_map>

namespace eshkol {

class CodegenContext;

// Region-based memory allocation
// Implements arena allocation with stack-based lifetimes
class RegionCodegen {
public:
    RegionCodegen(CodegenContext& ctx);

    // Region management
    // (with-region body ...) -> allocates region, runs body, frees region
    llvm::Value* enterRegion(const std::string& name);
    void exitRegion();

    // Allocate within current region
    // Returns pointer to allocated memory
    llvm::Value* regionAlloc(llvm::Type* type, const std::string& name = "");
    llvm::Value* regionAllocArray(llvm::Type* elem_type, llvm::Value* count,
                                   const std::string& name = "");

    // Get current region (for nested regions)
    llvm::Value* currentRegion();

    // Check if we're inside a region
    bool inRegion() const { return !region_stack_.empty(); }

private:
    CodegenContext& ctx_;

    // Stack of active regions
    struct RegionInfo {
        std::string name;
        llvm::Value* region_ptr;      // Pointer to region struct
        llvm::Value* alloc_ptr;       // Current allocation pointer
        llvm::BasicBlock* cleanup_bb; // Cleanup block for this region
    };
    std::stack<RegionInfo> region_stack_;

    // Runtime functions
    llvm::Function* getRegionCreate();
    llvm::Function* getRegionDestroy();
    llvm::Function* getRegionAlloc();
};

// Ownership tracking for linear types
class OwnershipCodegen {
public:
    OwnershipCodegen(CodegenContext& ctx);

    // (owned expr) - Take ownership of value
    // Returns value with ownership flag set
    llvm::Value* takeOwnership(llvm::Value* value, const std::string& name);

    // (move value) - Transfer ownership
    // Invalidates source, returns owned value
    llvm::Value* moveOwnership(llvm::Value* source, const std::string& dest_name);

    // Check ownership status at compile time (via type system)
    // Runtime check for debugging
    void assertOwned(llvm::Value* value, const std::string& msg);

    // Drop owned value (call destructor/cleanup)
    void dropOwned(llvm::Value* value);

private:
    CodegenContext& ctx_;

    // Track which values are currently owned
    std::unordered_map<llvm::Value*, bool> ownership_map_;
};

// Borrow checking codegen
class BorrowCodegen {
public:
    BorrowCodegen(CodegenContext& ctx);

    // (borrow value) - Create immutable borrow
    // Returns borrowed reference
    llvm::Value* borrowShared(llvm::Value* value, const std::string& name);

    // (borrow-mut value) - Create mutable borrow
    // Returns mutable borrowed reference
    llvm::Value* borrowMut(llvm::Value* value, const std::string& name);

    // End borrow scope
    void endBorrow(llvm::Value* borrow);

    // Runtime borrow checking (debug mode)
    void checkBorrowValid(llvm::Value* borrow, const std::string& msg);

private:
    CodegenContext& ctx_;

    // Active borrows
    struct BorrowInfo {
        llvm::Value* source;
        bool is_mutable;
        llvm::BasicBlock* scope_end;
    };
    std::vector<BorrowInfo> active_borrows_;
};

// Reference counting for shared values
class SharedRefCodegen {
public:
    SharedRefCodegen(CodegenContext& ctx);

    // (shared value) - Create reference-counted value
    llvm::Value* makeShared(llvm::Value* value, const std::string& name);

    // Clone shared reference (increment refcount)
    llvm::Value* cloneShared(llvm::Value* shared_ref);

    // Drop shared reference (decrement refcount, free if zero)
    void dropShared(llvm::Value* shared_ref);

    // (weak-ref shared) - Create weak reference
    llvm::Value* makeWeak(llvm::Value* shared_ref);

    // (weak-upgrade weak) - Try to upgrade weak to strong
    // Returns null if value was freed
    llvm::Value* upgradeWeak(llvm::Value* weak_ref);

    // Drop weak reference
    void dropWeak(llvm::Value* weak_ref);

private:
    CodegenContext& ctx_;

    // Runtime functions
    llvm::Function* getRefCountIncrement();
    llvm::Function* getRefCountDecrement();
    llvm::Function* getWeakRefCreate();
    llvm::Function* getWeakRefUpgrade();
};

} // namespace eshkol

#endif
```

**Step 6.2**: Create `lib/backend/region_codegen.cpp`

```cpp
#include "eshkol/backend/region_codegen.h"
#include "eshkol/backend/codegen_context.h"

namespace eshkol {

// ============================================================================
// REGION CODEGEN
// ============================================================================

RegionCodegen::RegionCodegen(CodegenContext& ctx) : ctx_(ctx) {}

llvm::Value* RegionCodegen::enterRegion(const std::string& name) {
    auto& builder = ctx_.getBuilder();
    auto& context = ctx_.getLLVMContext();

    // Create region struct
    // struct Region { void* base; size_t size; size_t used; }
    auto region_create = getRegionCreate();

    // Default region size: 1MB
    auto size = llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 1024 * 1024);
    auto region_ptr = builder.CreateCall(region_create, {size}, name + "_region");

    // Create cleanup block
    auto current_func = builder.GetInsertBlock()->getParent();
    auto cleanup_bb = llvm::BasicBlock::Create(context, name + "_cleanup", current_func);

    // Push to stack
    RegionInfo info;
    info.name = name;
    info.region_ptr = region_ptr;
    info.cleanup_bb = cleanup_bb;
    region_stack_.push(info);

    return region_ptr;
}

void RegionCodegen::exitRegion() {
    if (region_stack_.empty()) return;

    auto& builder = ctx_.getBuilder();
    auto info = region_stack_.top();
    region_stack_.pop();

    // Jump to cleanup
    builder.CreateBr(info.cleanup_bb);

    // Generate cleanup code
    builder.SetInsertPoint(info.cleanup_bb);
    auto region_destroy = getRegionDestroy();
    builder.CreateCall(region_destroy, {info.region_ptr});
}

llvm::Value* RegionCodegen::regionAlloc(llvm::Type* type, const std::string& name) {
    if (region_stack_.empty()) {
        // Fall back to heap allocation
        return ctx_.getBuilder().CreateAlloca(type, nullptr, name);
    }

    auto& builder = ctx_.getBuilder();
    auto& context = ctx_.getLLVMContext();

    auto region_alloc = getRegionAlloc();
    auto& layout = ctx_.getModule().getDataLayout();
    auto size = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(context),
        layout.getTypeAllocSize(type));
    auto align = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(context),
        layout.getABITypeAlign(type).value());

    auto void_ptr = builder.CreateCall(region_alloc,
        {region_stack_.top().region_ptr, size, align}, name + "_ptr");

    // Cast to correct type
    return builder.CreateBitCast(void_ptr, llvm::PointerType::getUnqual(type), name);
}

llvm::Function* RegionCodegen::getRegionCreate() {
    auto& module = ctx_.getModule();
    auto& context = ctx_.getLLVMContext();

    auto func = module.getFunction("eshkol_region_create");
    if (func) return func;

    // void* eshkol_region_create(size_t size)
    auto func_type = llvm::FunctionType::get(
        llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)),
        {llvm::Type::getInt64Ty(context)},
        false);

    func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage,
                                   "eshkol_region_create", module);
    return func;
}

llvm::Function* RegionCodegen::getRegionDestroy() {
    auto& module = ctx_.getModule();
    auto& context = ctx_.getLLVMContext();

    auto func = module.getFunction("eshkol_region_destroy");
    if (func) return func;

    // void eshkol_region_destroy(void* region)
    auto func_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context),
        {llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context))},
        false);

    func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage,
                                   "eshkol_region_destroy", module);
    return func;
}

llvm::Function* RegionCodegen::getRegionAlloc() {
    auto& module = ctx_.getModule();
    auto& context = ctx_.getLLVMContext();

    auto func = module.getFunction("eshkol_region_alloc");
    if (func) return func;

    // void* eshkol_region_alloc(void* region, size_t size, size_t align)
    auto ptr_type = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context));
    auto int64_type = llvm::Type::getInt64Ty(context);

    auto func_type = llvm::FunctionType::get(
        ptr_type, {ptr_type, int64_type, int64_type}, false);

    func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage,
                                   "eshkol_region_alloc", module);
    return func;
}

// ============================================================================
// OWNERSHIP CODEGEN
// ============================================================================

OwnershipCodegen::OwnershipCodegen(CodegenContext& ctx) : ctx_(ctx) {}

llvm::Value* OwnershipCodegen::takeOwnership(llvm::Value* value, const std::string& name) {
    // Mark value as owned
    ownership_map_[value] = true;

    // In debug mode, set ownership flag in value wrapper
    if (ctx_.isDebugMode()) {
        // [Add runtime ownership tracking]
    }

    return value;
}

llvm::Value* OwnershipCodegen::moveOwnership(llvm::Value* source, const std::string& dest_name) {
    // Check source is owned
    if (ownership_map_.find(source) == ownership_map_.end() || !ownership_map_[source]) {
        // Compile-time error would have caught this, but runtime check for safety
        if (ctx_.isDebugMode()) {
            assertOwned(source, "Cannot move from non-owned value");
        }
    }

    // Transfer ownership
    ownership_map_[source] = false;

    // Create new value with ownership
    auto result = source;  // In most cases, just use the same value
    ownership_map_[result] = true;

    return result;
}

void OwnershipCodegen::dropOwned(llvm::Value* value) {
    if (ownership_map_.find(value) == ownership_map_.end() || !ownership_map_[value]) {
        return;  // Already dropped or never owned
    }

    auto& builder = ctx_.getBuilder();

    // Get type and call appropriate destructor
    auto type = value->getType();
    if (type->isPointerTy()) {
        // Free the memory
        auto free_func = ctx_.getModule().getOrInsertFunction(
            "free",
            llvm::FunctionType::get(
                llvm::Type::getVoidTy(ctx_.getLLVMContext()),
                {llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(ctx_.getLLVMContext()))},
                false));

        auto void_ptr = builder.CreateBitCast(value,
            llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(ctx_.getLLVMContext())));
        builder.CreateCall(free_func, {void_ptr});
    }

    ownership_map_[value] = false;
}

// ============================================================================
// BORROW CODEGEN
// ============================================================================

BorrowCodegen::BorrowCodegen(CodegenContext& ctx) : ctx_(ctx) {}

llvm::Value* BorrowCodegen::borrowShared(llvm::Value* value, const std::string& name) {
    // Create borrowed reference (just a pointer in LLVM)
    // The borrow checker has already verified this is safe

    BorrowInfo info;
    info.source = value;
    info.is_mutable = false;
    info.scope_end = nullptr;  // Set when borrow ends
    active_borrows_.push_back(info);

    // In debug mode, increment borrow count
    if (ctx_.isDebugMode()) {
        // [Runtime borrow tracking]
    }

    return value;
}

llvm::Value* BorrowCodegen::borrowMut(llvm::Value* value, const std::string& name) {
    BorrowInfo info;
    info.source = value;
    info.is_mutable = true;
    info.scope_end = nullptr;
    active_borrows_.push_back(info);

    // In debug mode, set exclusive borrow flag
    if (ctx_.isDebugMode()) {
        // [Runtime exclusive borrow check]
    }

    return value;
}

void BorrowCodegen::endBorrow(llvm::Value* borrow) {
    // Find and remove borrow
    for (auto it = active_borrows_.begin(); it != active_borrows_.end(); ++it) {
        if (it->source == borrow) {
            // In debug mode, decrement borrow count / clear exclusive flag
            if (ctx_.isDebugMode()) {
                // [Runtime cleanup]
            }
            active_borrows_.erase(it);
            break;
        }
    }
}

// ============================================================================
// SHARED REF CODEGEN
// ============================================================================

SharedRefCodegen::SharedRefCodegen(CodegenContext& ctx) : ctx_(ctx) {}

llvm::Value* SharedRefCodegen::makeShared(llvm::Value* value, const std::string& name) {
    auto& builder = ctx_.getBuilder();
    auto& context = ctx_.getLLVMContext();

    // Allocate shared wrapper: { refcount: i64, weak_count: i64, value: T }
    auto value_type = value->getType();

    // For now, use a simple struct
    auto int64_type = llvm::Type::getInt64Ty(context);
    std::vector<llvm::Type*> members = {int64_type, int64_type, value_type};
    auto shared_type = llvm::StructType::get(context, members);

    // Allocate
    auto shared_ptr = builder.CreateAlloca(shared_type, nullptr, name + "_shared");

    // Initialize refcount = 1, weak_count = 0
    auto refcount_ptr = builder.CreateStructGEP(shared_type, shared_ptr, 0);
    builder.CreateStore(llvm::ConstantInt::get(int64_type, 1), refcount_ptr);

    auto weak_ptr = builder.CreateStructGEP(shared_type, shared_ptr, 1);
    builder.CreateStore(llvm::ConstantInt::get(int64_type, 0), weak_ptr);

    // Store value
    auto value_ptr = builder.CreateStructGEP(shared_type, shared_ptr, 2);
    builder.CreateStore(value, value_ptr);

    return shared_ptr;
}

llvm::Value* SharedRefCodegen::cloneShared(llvm::Value* shared_ref) {
    auto& builder = ctx_.getBuilder();
    auto& context = ctx_.getLLVMContext();

    // Increment refcount atomically
    auto shared_type = shared_ref->getType()->getPointerElementType();
    auto refcount_ptr = builder.CreateStructGEP(shared_type, shared_ref, 0);

    builder.CreateAtomicRMW(
        llvm::AtomicRMWInst::Add, refcount_ptr,
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 1),
        llvm::MaybeAlign(8), llvm::AtomicOrdering::AcquireRelease);

    return shared_ref;
}

void SharedRefCodegen::dropShared(llvm::Value* shared_ref) {
    auto& builder = ctx_.getBuilder();
    auto& context = ctx_.getLLVMContext();

    // Decrement refcount atomically
    auto shared_type = shared_ref->getType()->getPointerElementType();
    auto refcount_ptr = builder.CreateStructGEP(shared_type, shared_ref, 0);

    auto old_count = builder.CreateAtomicRMW(
        llvm::AtomicRMWInst::Sub, refcount_ptr,
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 1),
        llvm::MaybeAlign(8), llvm::AtomicOrdering::AcquireRelease);

    // If old count was 1, we need to free
    auto was_one = builder.CreateICmpEQ(old_count,
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 1));

    auto current_func = builder.GetInsertBlock()->getParent();
    auto free_bb = llvm::BasicBlock::Create(context, "free_shared", current_func);
    auto cont_bb = llvm::BasicBlock::Create(context, "cont", current_func);

    builder.CreateCondBr(was_one, free_bb, cont_bb);

    // Free block
    builder.SetInsertPoint(free_bb);
    // Check weak_count == 0 before freeing control block
    auto weak_ptr = builder.CreateStructGEP(shared_type, shared_ref, 1);
    auto weak_count = builder.CreateLoad(llvm::Type::getInt64Ty(context), weak_ptr);
    auto no_weak = builder.CreateICmpEQ(weak_count,
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(context), 0));

    auto free_all_bb = llvm::BasicBlock::Create(context, "free_all", current_func);
    auto free_value_bb = llvm::BasicBlock::Create(context, "free_value", current_func);

    builder.CreateCondBr(no_weak, free_all_bb, free_value_bb);

    // Free everything (no weak refs)
    builder.SetInsertPoint(free_all_bb);
    auto void_ptr = builder.CreateBitCast(shared_ref,
        llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context)));
    auto free_func = ctx_.getModule().getOrInsertFunction(
        "free",
        llvm::FunctionType::get(
            llvm::Type::getVoidTy(context),
            {llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(context))},
            false));
    builder.CreateCall(free_func, {void_ptr});
    builder.CreateBr(cont_bb);

    // Free just value (weak refs exist)
    builder.SetInsertPoint(free_value_bb);
    // [Mark value as freed but keep control block]
    builder.CreateBr(cont_bb);

    builder.SetInsertPoint(cont_bb);
}

} // namespace eshkol
```

### Day 86-87: Runtime Support Library

**Step 6.3**: Create `runtime/region.c`

```c
// Eshkol Runtime: Region-based Memory Management

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

typedef struct Region {
    void* base;
    size_t size;
    size_t used;
    struct Region* parent;  // For nested regions
} Region;

// Thread-local current region
static __thread Region* current_region = NULL;

void* eshkol_region_create(size_t size) {
    Region* region = (Region*)malloc(sizeof(Region) + size);
    if (!region) return NULL;

    region->base = (char*)region + sizeof(Region);
    region->size = size;
    region->used = 0;
    region->parent = current_region;

    current_region = region;
    return region;
}

void eshkol_region_destroy(void* region_ptr) {
    Region* region = (Region*)region_ptr;
    if (!region) return;

    // Restore parent region
    current_region = region->parent;

    // Free entire region at once
    free(region);
}

void* eshkol_region_alloc(void* region_ptr, size_t size, size_t align) {
    Region* region = (Region*)region_ptr;
    if (!region) return malloc(size);  // Fallback to heap

    // Align the current position
    size_t aligned_used = (region->used + align - 1) & ~(align - 1);

    // Check if we have space
    if (aligned_used + size > region->size) {
        // Out of region space - could grow or fall back to heap
        return malloc(size);
    }

    void* ptr = (char*)region->base + aligned_used;
    region->used = aligned_used + size;

    return ptr;
}

// Get current region's remaining space
size_t eshkol_region_available(void* region_ptr) {
    Region* region = (Region*)region_ptr;
    if (!region) return 0;
    return region->size - region->used;
}

// Reset region (free all allocations but keep memory)
void eshkol_region_reset(void* region_ptr) {
    Region* region = (Region*)region_ptr;
    if (region) {
        region->used = 0;
    }
}
```

### Day 88: Integration and Testing

**Step 6.4**: Add OALR codegen to main codegen (`lib/backend/llvm_codegen.cpp`):

```cpp
#include "eshkol/backend/region_codegen.h"

// In LLVMCodegen class:
RegionCodegen region_codegen_;
OwnershipCodegen ownership_codegen_;
BorrowCodegen borrow_codegen_;
SharedRefCodegen shared_codegen_;

// Handle with-region
llvm::Value* LLVMCodegen::codegenWithRegion(const eshkol_ast_t& ast) {
    auto name = getStringValue(ast.operation.with_region.name);

    // Enter region
    auto region = region_codegen_.enterRegion(name);

    // Codegen body
    llvm::Value* result = nullptr;
    for (size_t i = 0; i < ast.operation.with_region.body_count; i++) {
        result = codegenExpr(ast.operation.with_region.body[i]);
    }

    // Exit region (cleanup)
    region_codegen_.exitRegion();

    return result;
}

// Handle owned
llvm::Value* LLVMCodegen::codegenOwned(const eshkol_ast_t& ast) {
    auto value = codegenExpr(ast.operation.owned.value);
    return ownership_codegen_.takeOwnership(value, "owned");
}

// Handle move
llvm::Value* LLVMCodegen::codegenMove(const eshkol_ast_t& ast) {
    auto source = codegenExpr(ast.operation.move.source);
    return ownership_codegen_.moveOwnership(source, "moved");
}

// Handle borrow
llvm::Value* LLVMCodegen::codegenBorrow(const eshkol_ast_t& ast) {
    auto value = codegenExpr(ast.operation.borrow.value);
    if (ast.operation.borrow.is_mutable) {
        return borrow_codegen_.borrowMut(value, "borrow_mut");
    }
    return borrow_codegen_.borrowShared(value, "borrow");
}

// Handle shared
llvm::Value* LLVMCodegen::codegenShared(const eshkol_ast_t& ast) {
    auto value = codegenExpr(ast.operation.shared.value);
    return shared_codegen_.makeShared(value, "shared");
}

// Handle weak-ref
llvm::Value* LLVMCodegen::codegenWeakRef(const eshkol_ast_t& ast) {
    auto shared = codegenExpr(ast.operation.weak_ref.shared);
    return shared_codegen_.makeWeak(shared);
}
```

**Step 6.5**: Create `tests/oalr/oalr_test.esk`

```scheme
(require stdlib)

(display "=== OALR Memory Tests ===")
(newline)

;; Test 1: Region-based allocation
(display "Region test: ")
(with-region my-region
  (define x (region-alloc 100))  ;; Allocate 100 bytes
  (define y (region-alloc 200))
  (display "allocated in region")
  (newline)
  ;; x and y are automatically freed when region exits
  )
(display "region freed")
(newline)

;; Test 2: Ownership
(display "Ownership test: ")
(define data (owned (make-vector 1000)))
(define moved-data (move data))
;; (use data) ;; ERROR: data has been moved
(display "ownership transferred")
(newline)

;; Test 3: Borrowing
(display "Borrow test: ")
(define vec (owned (vector 1 2 3 4 5)))
(define (sum-vec borrowed-vec)
  (fold + 0 borrowed-vec))

(display "sum = ")
(display (sum-vec (borrow vec)))
(newline)
;; vec is still valid after borrow ends

;; Test 4: Shared references
(display "Shared ref test: ")
(define shared-data (shared (list 1 2 3)))
(define ref1 shared-data)
(define ref2 shared-data)
;; Both ref1 and ref2 point to same data
;; Data is freed when last reference is dropped
(display "refcount works")
(newline)

;; Test 5: Weak references
(display "Weak ref test: ")
(define shared-obj (shared (make-object)))
(define weak (weak-ref shared-obj))
(define upgraded (weak-upgrade weak))
(if upgraded
    (display "weak ref valid")
    (display "weak ref expired"))
(newline)

(display "OALR tests complete!")
(newline)
```

**Verification Checklist for Phase 6**:
- [ ] Region allocation works correctly
- [ ] Region cleanup frees all allocations
- [ ] Nested regions work properly
- [ ] Ownership transfer invalidates source
- [ ] Borrow checker prevents use-after-move
- [ ] Shared references are reference counted
- [ ] Weak references return null after source freed
- [ ] No memory leaks in any scenario
- [ ] Performance comparable to manual memory management

---

## Phase 7: Serialization & Checkpointing (5 days)

### Why Seventh?
- Models can be saved and loaded
- Enables training resumption
- Required for production deployment
- Foundation for distributed training

### Day 89-91: Checkpoint Format Design

**Files to Create**:
```
inc/eshkol/io/checkpoint.h
inc/eshkol/io/serializer.h
lib/io/checkpoint.cpp
lib/io/serializer.cpp
```

**Step 7.1**: Create `inc/eshkol/io/serializer.h`

```cpp
#ifndef ESHKOL_SERIALIZER_H
#define ESHKOL_SERIALIZER_H

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>

namespace eshkol::io {

// Supported value types for serialization
enum class ValueType : uint8_t {
    Null = 0,
    Bool = 1,
    Int64 = 2,
    Float64 = 3,
    String = 4,
    List = 5,
    Map = 6,
    Tensor = 7,
    Symbol = 8,
    Closure = 9
};

// Forward declaration
class Serializer;
class Deserializer;

// Serializable interface
class ISerializable {
public:
    virtual ~ISerializable() = default;
    virtual void serialize(Serializer& s) const = 0;
    virtual void deserialize(Deserializer& d) = 0;
};

// Binary serializer
class Serializer {
public:
    explicit Serializer(std::ostream& out);

    // Primitive types
    void write(bool value);
    void write(int64_t value);
    void write(double value);
    void write(const std::string& value);

    // Containers
    template<typename T>
    void write(const std::vector<T>& vec);

    template<typename K, typename V>
    void write(const std::unordered_map<K, V>& map);

    // Raw bytes
    void writeBytes(const void* data, size_t size);

    // Tensor (shape + data)
    void writeTensor(const std::vector<size_t>& shape, const float* data);

    // Type tag (for polymorphic serialization)
    void writeType(ValueType type);

    // Serializable objects
    void write(const ISerializable& obj);

private:
    std::ostream& out_;

    void writeVarInt(uint64_t value);
};

// Binary deserializer
class Deserializer {
public:
    explicit Deserializer(std::istream& in);

    // Primitive types
    bool readBool();
    int64_t readInt64();
    double readFloat64();
    std::string readString();

    // Containers
    template<typename T>
    std::vector<T> readVector();

    template<typename K, typename V>
    std::unordered_map<K, V> readMap();

    // Raw bytes
    void readBytes(void* data, size_t size);

    // Tensor
    void readTensor(std::vector<size_t>& shape, std::vector<float>& data);

    // Type tag
    ValueType readType();

    // Check if more data available
    bool hasMore() const;

private:
    std::istream& in_;

    uint64_t readVarInt();
};

// Implementation of templates
template<typename T>
void Serializer::write(const std::vector<T>& vec) {
    writeVarInt(vec.size());
    for (const auto& item : vec) {
        write(item);
    }
}

template<typename K, typename V>
void Serializer::write(const std::unordered_map<K, V>& map) {
    writeVarInt(map.size());
    for (const auto& [key, value] : map) {
        write(key);
        write(value);
    }
}

} // namespace eshkol::io

#endif
```

**Step 7.2**: Create `inc/eshkol/io/checkpoint.h`

```cpp
#ifndef ESHKOL_CHECKPOINT_H
#define ESHKOL_CHECKPOINT_H

#include "serializer.h"
#include "eshkol/gpu/gpu_tensor.h"
#include <string>
#include <unordered_map>
#include <variant>
#include <functional>

namespace eshkol::io {

// Checkpoint file format (.eskpt)
// Header: magic (4 bytes) + version (4 bytes) + metadata_size (8 bytes)
// Metadata: JSON-like key-value pairs
// Data: Binary tensor data

constexpr uint32_t CHECKPOINT_MAGIC = 0x45534B50;  // "ESKP"
constexpr uint32_t CHECKPOINT_VERSION = 1;

// Metadata value types
using MetadataValue = std::variant<
    bool, int64_t, double, std::string,
    std::vector<int64_t>, std::vector<double>, std::vector<std::string>
>;

// Checkpoint metadata
struct CheckpointMetadata {
    uint32_t version = CHECKPOINT_VERSION;
    std::string model_name;
    std::string model_version;
    std::unordered_map<std::string, MetadataValue> custom;

    // Training state
    int64_t epoch = 0;
    int64_t global_step = 0;
    double best_loss = std::numeric_limits<double>::infinity();

    // Timestamp
    int64_t created_at = 0;
};

// Tensor entry in checkpoint
struct TensorEntry {
    std::string name;
    std::vector<size_t> shape;
    std::string dtype;  // "float32", "float64", "int64", etc.
    size_t offset;      // Byte offset in data section
    size_t size;        // Size in bytes
};

// Checkpoint reader
class CheckpointReader {
public:
    explicit CheckpointReader(const std::string& path);
    ~CheckpointReader();

    // Open checkpoint file
    bool open();
    void close();

    // Check if valid checkpoint
    bool isValid() const { return valid_; }

    // Get metadata
    const CheckpointMetadata& metadata() const { return metadata_; }

    // List all tensors
    std::vector<std::string> tensorNames() const;

    // Check if tensor exists
    bool hasTensor(const std::string& name) const;

    // Get tensor shape without loading data
    std::vector<size_t> tensorShape(const std::string& name) const;

    // Load tensor to CPU
    std::vector<float> loadTensor(const std::string& name);

    // Load tensor directly to GPU
    gpu::GPUTensor loadTensorGPU(const std::string& name);

    // Load all tensors matching pattern
    std::unordered_map<std::string, gpu::GPUTensor> loadTensors(
        const std::string& pattern = "*");

private:
    std::string path_;
    std::ifstream file_;
    bool valid_ = false;

    CheckpointMetadata metadata_;
    std::unordered_map<std::string, TensorEntry> tensors_;

    bool readHeader();
    bool readMetadata();
    bool readTensorIndex();
};

// Checkpoint writer
class CheckpointWriter {
public:
    explicit CheckpointWriter(const std::string& path);
    ~CheckpointWriter();

    // Set metadata
    void setMetadata(const CheckpointMetadata& metadata);

    // Add tensor from CPU data
    void addTensor(const std::string& name,
                   const std::vector<size_t>& shape,
                   const float* data);

    // Add tensor from GPU
    void addTensor(const std::string& name, const gpu::GPUTensor& tensor);

    // Add optimizer state
    void addOptimizerState(const std::string& name,
                           const std::unordered_map<std::string, gpu::GPUTensor>& state);

    // Write checkpoint to file
    bool write();

private:
    std::string path_;
    CheckpointMetadata metadata_;
    std::vector<std::pair<std::string, std::vector<float>>> pending_tensors_;
    std::vector<std::pair<std::string, std::vector<size_t>>> tensor_shapes_;

    bool writeHeader(std::ofstream& out);
    bool writeMetadata(std::ofstream& out);
    bool writeTensors(std::ofstream& out);
};

// Convenience functions
bool saveCheckpoint(const std::string& path,
                    const std::unordered_map<std::string, gpu::GPUTensor>& tensors,
                    const CheckpointMetadata& metadata = {});

std::unordered_map<std::string, gpu::GPUTensor> loadCheckpoint(const std::string& path);

} // namespace eshkol::io

#endif
```

**Step 7.3**: Create `lib/io/checkpoint.cpp`

```cpp
#include "eshkol/io/checkpoint.h"
#include <cstring>
#include <ctime>
#include <regex>

namespace eshkol::io {

// ============================================================================
// CHECKPOINT READER
// ============================================================================

CheckpointReader::CheckpointReader(const std::string& path) : path_(path) {}

CheckpointReader::~CheckpointReader() {
    close();
}

bool CheckpointReader::open() {
    file_.open(path_, std::ios::binary);
    if (!file_) return false;

    if (!readHeader()) {
        close();
        return false;
    }

    if (!readMetadata()) {
        close();
        return false;
    }

    if (!readTensorIndex()) {
        close();
        return false;
    }

    valid_ = true;
    return true;
}

void CheckpointReader::close() {
    if (file_.is_open()) {
        file_.close();
    }
    valid_ = false;
}

bool CheckpointReader::readHeader() {
    uint32_t magic, version;
    file_.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file_.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != CHECKPOINT_MAGIC) {
        return false;
    }

    if (version > CHECKPOINT_VERSION) {
        return false;  // Newer version, can't read
    }

    metadata_.version = version;
    return true;
}

bool CheckpointReader::readMetadata() {
    uint64_t metadata_size;
    file_.read(reinterpret_cast<char*>(&metadata_size), sizeof(metadata_size));

    // Read metadata as JSON-like format
    std::vector<char> buffer(metadata_size);
    file_.read(buffer.data(), metadata_size);

    // Parse metadata (simplified - real implementation would use JSON parser)
    std::string metadata_str(buffer.begin(), buffer.end());
    // [Parse model_name, epoch, global_step, etc.]

    return true;
}

bool CheckpointReader::readTensorIndex() {
    uint64_t num_tensors;
    file_.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

    for (uint64_t i = 0; i < num_tensors; i++) {
        TensorEntry entry;

        // Read name length and name
        uint32_t name_len;
        file_.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        entry.name.resize(name_len);
        file_.read(entry.name.data(), name_len);

        // Read shape
        uint32_t ndim;
        file_.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        entry.shape.resize(ndim);
        for (uint32_t j = 0; j < ndim; j++) {
            uint64_t dim;
            file_.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            entry.shape[j] = dim;
        }

        // Read dtype
        uint32_t dtype_len;
        file_.read(reinterpret_cast<char*>(&dtype_len), sizeof(dtype_len));
        entry.dtype.resize(dtype_len);
        file_.read(entry.dtype.data(), dtype_len);

        // Read offset and size
        file_.read(reinterpret_cast<char*>(&entry.offset), sizeof(entry.offset));
        file_.read(reinterpret_cast<char*>(&entry.size), sizeof(entry.size));

        tensors_[entry.name] = entry;
    }

    return true;
}

std::vector<std::string> CheckpointReader::tensorNames() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : tensors_) {
        names.push_back(name);
    }
    return names;
}

bool CheckpointReader::hasTensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

std::vector<size_t> CheckpointReader::tensorShape(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return {};
    return it->second.shape;
}

std::vector<float> CheckpointReader::loadTensor(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return {};

    const auto& entry = it->second;

    // Seek to tensor data
    file_.seekg(entry.offset);

    // Read data
    std::vector<float> data(entry.size / sizeof(float));
    file_.read(reinterpret_cast<char*>(data.data()), entry.size);

    return data;
}

gpu::GPUTensor CheckpointReader::loadTensorGPU(const std::string& name) {
    auto cpu_data = loadTensor(name);
    if (cpu_data.empty()) return {};

    auto shape = tensorShape(name);
    return gpu::GPUTensor::fromHost(cpu_data.data(), shape);
}

// ============================================================================
// CHECKPOINT WRITER
// ============================================================================

CheckpointWriter::CheckpointWriter(const std::string& path) : path_(path) {
    metadata_.created_at = std::time(nullptr);
}

CheckpointWriter::~CheckpointWriter() = default;

void CheckpointWriter::setMetadata(const CheckpointMetadata& metadata) {
    metadata_ = metadata;
    metadata_.created_at = std::time(nullptr);
}

void CheckpointWriter::addTensor(const std::string& name,
                                  const std::vector<size_t>& shape,
                                  const float* data) {
    size_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;

    std::vector<float> tensor_data(data, data + num_elements);
    pending_tensors_.emplace_back(name, std::move(tensor_data));
    tensor_shapes_.emplace_back(name, shape);
}

void CheckpointWriter::addTensor(const std::string& name, const gpu::GPUTensor& tensor) {
    std::vector<float> cpu_data(tensor.size());
    tensor.copyToHost(cpu_data.data());
    addTensor(name, tensor.shape(), cpu_data.data());
}

bool CheckpointWriter::write() {
    std::ofstream out(path_, std::ios::binary);
    if (!out) return false;

    if (!writeHeader(out)) return false;
    if (!writeMetadata(out)) return false;
    if (!writeTensors(out)) return false;

    return true;
}

bool CheckpointWriter::writeHeader(std::ofstream& out) {
    out.write(reinterpret_cast<const char*>(&CHECKPOINT_MAGIC), sizeof(CHECKPOINT_MAGIC));
    out.write(reinterpret_cast<const char*>(&CHECKPOINT_VERSION), sizeof(CHECKPOINT_VERSION));
    return out.good();
}

bool CheckpointWriter::writeMetadata(std::ofstream& out) {
    // Serialize metadata to JSON-like format
    std::string metadata_str;
    metadata_str += "model_name:" + metadata_.model_name + "\n";
    metadata_str += "epoch:" + std::to_string(metadata_.epoch) + "\n";
    metadata_str += "global_step:" + std::to_string(metadata_.global_step) + "\n";
    metadata_str += "best_loss:" + std::to_string(metadata_.best_loss) + "\n";
    metadata_str += "created_at:" + std::to_string(metadata_.created_at) + "\n";

    uint64_t metadata_size = metadata_str.size();
    out.write(reinterpret_cast<const char*>(&metadata_size), sizeof(metadata_size));
    out.write(metadata_str.data(), metadata_size);

    return out.good();
}

bool CheckpointWriter::writeTensors(std::ofstream& out) {
    // Write tensor index
    uint64_t num_tensors = pending_tensors_.size();
    out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));

    // Calculate offsets
    size_t current_offset = out.tellp();
    current_offset += pending_tensors_.size() * 100;  // Rough estimate for index

    std::vector<TensorEntry> entries;
    for (size_t i = 0; i < pending_tensors_.size(); i++) {
        const auto& [name, data] = pending_tensors_[i];
        const auto& shape = tensor_shapes_[i].second;

        TensorEntry entry;
        entry.name = name;
        entry.shape = shape;
        entry.dtype = "float32";
        entry.size = data.size() * sizeof(float);
        entry.offset = current_offset;

        entries.push_back(entry);
        current_offset += entry.size;
    }

    // Write index
    for (const auto& entry : entries) {
        uint32_t name_len = entry.name.size();
        out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        out.write(entry.name.data(), name_len);

        uint32_t ndim = entry.shape.size();
        out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        for (auto dim : entry.shape) {
            uint64_t d = dim;
            out.write(reinterpret_cast<const char*>(&d), sizeof(d));
        }

        uint32_t dtype_len = entry.dtype.size();
        out.write(reinterpret_cast<const char*>(&dtype_len), sizeof(dtype_len));
        out.write(entry.dtype.data(), dtype_len);

        out.write(reinterpret_cast<const char*>(&entry.offset), sizeof(entry.offset));
        out.write(reinterpret_cast<const char*>(&entry.size), sizeof(entry.size));
    }

    // Write tensor data
    for (const auto& [name, data] : pending_tensors_) {
        out.write(reinterpret_cast<const char*>(data.data()),
                  data.size() * sizeof(float));
    }

    return out.good();
}

// Convenience functions
bool saveCheckpoint(const std::string& path,
                    const std::unordered_map<std::string, gpu::GPUTensor>& tensors,
                    const CheckpointMetadata& metadata) {
    CheckpointWriter writer(path);
    writer.setMetadata(metadata);

    for (const auto& [name, tensor] : tensors) {
        writer.addTensor(name, tensor);
    }

    return writer.write();
}

std::unordered_map<std::string, gpu::GPUTensor> loadCheckpoint(const std::string& path) {
    CheckpointReader reader(path);
    if (!reader.open()) return {};

    return reader.loadTensors("*");
}

} // namespace eshkol::io
```

### Day 92-93: Language Integration and Testing

**Step 7.4**: Add serialization primitives to parser:

```cpp
// Add to get_operator_type():
if (op == "save-checkpoint") return ESHKOL_SAVE_CHECKPOINT_OP;
if (op == "load-checkpoint") return ESHKOL_LOAD_CHECKPOINT_OP;
if (op == "serialize") return ESHKOL_SERIALIZE_OP;
if (op == "deserialize") return ESHKOL_DESERIALIZE_OP;
```

**Step 7.5**: Create `tests/io/checkpoint_test.esk`

```scheme
(require stdlib)

(display "=== Checkpoint Tests ===")
(newline)

;; Create some tensors
(define W1 (gpu-tensor (make-random-tensor '(784 256))))
(define b1 (gpu-tensor (make-random-tensor '(256))))
(define W2 (gpu-tensor (make-random-tensor '(256 10))))
(define b2 (gpu-tensor (make-random-tensor '(10))))

;; Save checkpoint
(define checkpoint-path "/tmp/test_model.eskpt")

(display "Saving checkpoint... ")
(save-checkpoint checkpoint-path
  (list (cons "layer1.weight" W1)
        (cons "layer1.bias" b1)
        (cons "layer2.weight" W2)
        (cons "layer2.bias" b2))
  (make-metadata "test_model" 1 100 0.05))
(display "done")
(newline)

;; Load checkpoint
(display "Loading checkpoint... ")
(define loaded (load-checkpoint checkpoint-path))
(display "done")
(newline)

;; Verify shapes
(display "Loaded tensors:")
(newline)
(for-each (lambda (kv)
            (display "  ")
            (display (car kv))
            (display ": ")
            (display (tensor-shape (cdr kv)))
            (newline))
          loaded)

;; Verify data integrity
(display "Verifying data integrity... ")
(define W1-loaded (assoc-ref loaded "layer1.weight"))
(define diff (tensor-sum (tensor-abs (tensor-sub W1 W1-loaded))))
(if (< diff 1e-6)
    (display "PASS")
    (display "FAIL"))
(newline)

(display "Checkpoint tests complete!")
(newline)
```

**Verification Checklist for Phase 7**:
- [ ] Checkpoint files are valid binary format
- [ ] Metadata is correctly saved and loaded
- [ ] Tensor data integrity is preserved
- [ ] Shape information is correct
- [ ] GPU tensors can be saved directly
- [ ] Partial loading works (load specific tensors)
- [ ] Optimizer state can be checkpointed
- [ ] File size is reasonable (no unnecessary overhead)
- [ ] Loading is fast (memory-mapped when possible)

---

## Phase 8: Profiling Infrastructure (6 days)

### Why Eighth?
- Performance optimization needs measurement
- Identifies bottlenecks in training
- Required for production monitoring
- Enables data-driven optimizations

### Day 94-96: Profiler Core

**Files to Create**:
```
inc/eshkol/profiler/profiler.h
inc/eshkol/profiler/timer.h
lib/profiler/profiler.cpp
lib/profiler/timer.cpp
```

**Step 8.1**: Create `inc/eshkol/profiler/timer.h`

```cpp
#ifndef ESHKOL_TIMER_H
#define ESHKOL_TIMER_H

#include <chrono>
#include <string>

namespace eshkol::profiler {

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::nanoseconds;

// RAII timer for automatic scope timing
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();

    // Get elapsed time so far
    Duration elapsed() const;

    // Pause/resume
    void pause();
    void resume();

private:
    std::string name_;
    TimePoint start_;
    Duration accumulated_{0};
    bool paused_ = false;
};

// Manual timer control
class Timer {
public:
    void start();
    void stop();
    void reset();

    Duration elapsed() const;
    double elapsedMs() const;
    double elapsedUs() const;

    bool isRunning() const { return running_; }

private:
    TimePoint start_;
    Duration accumulated_{0};
    bool running_ = false;
};

// Macro for easy scope timing
#define ESHKOL_PROFILE_SCOPE(name) \
    eshkol::profiler::ScopedTimer _eshkol_timer_##__LINE__(name)

#define ESHKOL_PROFILE_FUNCTION() \
    ESHKOL_PROFILE_SCOPE(__FUNCTION__)

} // namespace eshkol::profiler

#endif
```

**Step 8.2**: Create `inc/eshkol/profiler/profiler.h`

```cpp
#ifndef ESHKOL_PROFILER_H
#define ESHKOL_PROFILER_H

#include "timer.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <thread>
#include <fstream>

namespace eshkol::profiler {

// Event types for Chrome trace format
enum class EventType {
    Begin,      // B - duration begin
    End,        // E - duration end
    Complete,   // X - complete event
    Instant,    // i - instant event
    Counter,    // C - counter event
    Metadata    // M - metadata
};

// Profiler event
struct ProfileEvent {
    std::string name;
    std::string category;
    EventType type;
    int64_t timestamp_us;
    int64_t duration_us;
    uint32_t thread_id;
    uint32_t process_id;
    std::unordered_map<std::string, std::string> args;
};

// Memory allocation event
struct MemoryEvent {
    int64_t timestamp_us;
    size_t bytes;
    bool is_allocation;  // true = alloc, false = free
    std::string location;
    void* address;
};

// Statistics for a named region
struct ProfileStats {
    std::string name;
    uint64_t call_count = 0;
    Duration total_time{0};
    Duration min_time{Duration::max()};
    Duration max_time{Duration::min()};
    double avg_time_us = 0.0;
};

// Global profiler
class Profiler {
public:
    static Profiler& instance();

    // Enable/disable profiling
    void enable();
    void disable();
    bool isEnabled() const { return enabled_; }

    // Record events
    void beginEvent(const std::string& name, const std::string& category = "");
    void endEvent(const std::string& name);
    void recordInstant(const std::string& name, const std::string& category = "");
    void recordCounter(const std::string& name, int64_t value);

    // Memory tracking
    void recordAllocation(size_t bytes, void* address, const std::string& location = "");
    void recordFree(void* address, const std::string& location = "");

    // Get statistics
    std::vector<ProfileStats> getStats() const;
    ProfileStats getStats(const std::string& name) const;

    // Clear all recorded data
    void clear();

    // Export to Chrome trace format
    bool exportChromeTrace(const std::string& path);

    // Export to simple text format
    std::string exportText() const;

    // Memory statistics
    size_t currentMemoryUsage() const { return current_memory_.load(); }
    size_t peakMemoryUsage() const { return peak_memory_.load(); }
    uint64_t totalAllocations() const { return total_allocations_.load(); }

private:
    Profiler() = default;

    std::atomic<bool> enabled_{false};
    std::mutex mutex_;

    // Events
    std::vector<ProfileEvent> events_;

    // Per-name statistics
    std::unordered_map<std::string, ProfileStats> stats_;

    // Active events (for matching begin/end)
    struct ActiveEvent {
        std::string name;
        TimePoint start;
    };
    std::unordered_map<uint32_t, std::vector<ActiveEvent>> active_events_;

    // Memory tracking
    std::vector<MemoryEvent> memory_events_;
    std::atomic<size_t> current_memory_{0};
    std::atomic<size_t> peak_memory_{0};
    std::atomic<uint64_t> total_allocations_{0};
    std::unordered_map<void*, size_t> allocations_;

    // Helper functions
    uint32_t getThreadId() const;
    int64_t getTimestampUs() const;
};

// RAII profiler scope
class ProfileScope {
public:
    ProfileScope(const std::string& name, const std::string& category = "")
        : name_(name) {
        Profiler::instance().beginEvent(name, category);
    }

    ~ProfileScope() {
        Profiler::instance().endEvent(name_);
    }

private:
    std::string name_;
};

// Convenient macros
#ifdef ESHKOL_ENABLE_PROFILING
    #define ESHKOL_PROFILE_BEGIN(name) \
        eshkol::profiler::Profiler::instance().beginEvent(name)
    #define ESHKOL_PROFILE_END(name) \
        eshkol::profiler::Profiler::instance().endEvent(name)
    #define ESHKOL_PROFILE_SCOPE_CAT(name, cat) \
        eshkol::profiler::ProfileScope _profile_scope_##__LINE__(name, cat)
#else
    #define ESHKOL_PROFILE_BEGIN(name)
    #define ESHKOL_PROFILE_END(name)
    #define ESHKOL_PROFILE_SCOPE_CAT(name, cat)
#endif

} // namespace eshkol::profiler

#endif
```

**Step 8.3**: Create `lib/profiler/profiler.cpp`

```cpp
#include "eshkol/profiler/profiler.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace eshkol::profiler {

Profiler& Profiler::instance() {
    static Profiler instance;
    return instance;
}

void Profiler::enable() {
    enabled_ = true;
}

void Profiler::disable() {
    enabled_ = false;
}

void Profiler::beginEvent(const std::string& name, const std::string& category) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto tid = getThreadId();
    auto timestamp = getTimestampUs();

    // Record event
    ProfileEvent event;
    event.name = name;
    event.category = category.empty() ? "default" : category;
    event.type = EventType::Begin;
    event.timestamp_us = timestamp;
    event.thread_id = tid;
    event.process_id = 0;
    events_.push_back(event);

    // Track active event
    ActiveEvent active;
    active.name = name;
    active.start = Clock::now();
    active_events_[tid].push_back(active);
}

void Profiler::endEvent(const std::string& name) {
    if (!enabled_) return;

    auto end_time = Clock::now();
    std::lock_guard<std::mutex> lock(mutex_);

    auto tid = getThreadId();
    auto timestamp = getTimestampUs();

    // Find matching begin event
    auto& active = active_events_[tid];
    for (auto it = active.rbegin(); it != active.rend(); ++it) {
        if (it->name == name) {
            auto duration = std::chrono::duration_cast<Duration>(end_time - it->start);

            // Record end event
            ProfileEvent event;
            event.name = name;
            event.type = EventType::End;
            event.timestamp_us = timestamp;
            event.thread_id = tid;
            event.process_id = 0;
            events_.push_back(event);

            // Update statistics
            auto& stat = stats_[name];
            stat.name = name;
            stat.call_count++;
            stat.total_time += duration;
            stat.min_time = std::min(stat.min_time, duration);
            stat.max_time = std::max(stat.max_time, duration);
            stat.avg_time_us = std::chrono::duration<double, std::micro>(
                stat.total_time).count() / stat.call_count;

            // Remove from active
            active.erase(std::next(it).base());
            break;
        }
    }
}

void Profiler::recordInstant(const std::string& name, const std::string& category) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    ProfileEvent event;
    event.name = name;
    event.category = category.empty() ? "instant" : category;
    event.type = EventType::Instant;
    event.timestamp_us = getTimestampUs();
    event.thread_id = getThreadId();
    event.process_id = 0;
    events_.push_back(event);
}

void Profiler::recordCounter(const std::string& name, int64_t value) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    ProfileEvent event;
    event.name = name;
    event.category = "counter";
    event.type = EventType::Counter;
    event.timestamp_us = getTimestampUs();
    event.thread_id = getThreadId();
    event.process_id = 0;
    event.args["value"] = std::to_string(value);
    events_.push_back(event);
}

void Profiler::recordAllocation(size_t bytes, void* address, const std::string& location) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    MemoryEvent event;
    event.timestamp_us = getTimestampUs();
    event.bytes = bytes;
    event.is_allocation = true;
    event.location = location;
    event.address = address;
    memory_events_.push_back(event);

    allocations_[address] = bytes;
    current_memory_ += bytes;
    total_allocations_++;

    size_t current = current_memory_.load();
    size_t peak = peak_memory_.load();
    while (current > peak && !peak_memory_.compare_exchange_weak(peak, current)) {}
}

void Profiler::recordFree(void* address, const std::string& location) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocations_.find(address);
    if (it != allocations_.end()) {
        MemoryEvent event;
        event.timestamp_us = getTimestampUs();
        event.bytes = it->second;
        event.is_allocation = false;
        event.location = location;
        event.address = address;
        memory_events_.push_back(event);

        current_memory_ -= it->second;
        allocations_.erase(it);
    }
}

std::vector<ProfileStats> Profiler::getStats() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));

    std::vector<ProfileStats> result;
    for (const auto& [name, stat] : stats_) {
        result.push_back(stat);
    }

    // Sort by total time descending
    std::sort(result.begin(), result.end(),
        [](const ProfileStats& a, const ProfileStats& b) {
            return a.total_time > b.total_time;
        });

    return result;
}

void Profiler::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.clear();
    stats_.clear();
    active_events_.clear();
    memory_events_.clear();
    current_memory_ = 0;
    peak_memory_ = 0;
    total_allocations_ = 0;
    allocations_.clear();
}

bool Profiler::exportChromeTrace(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ofstream out(path);
    if (!out) return false;

    out << "{\"traceEvents\":[";

    bool first = true;
    for (const auto& event : events_) {
        if (!first) out << ",";
        first = false;

        out << "{";
        out << "\"name\":\"" << event.name << "\",";
        out << "\"cat\":\"" << event.category << "\",";

        char phase;
        switch (event.type) {
            case EventType::Begin: phase = 'B'; break;
            case EventType::End: phase = 'E'; break;
            case EventType::Complete: phase = 'X'; break;
            case EventType::Instant: phase = 'i'; break;
            case EventType::Counter: phase = 'C'; break;
            default: phase = 'M';
        }
        out << "\"ph\":\"" << phase << "\",";

        out << "\"ts\":" << event.timestamp_us << ",";
        out << "\"pid\":" << event.process_id << ",";
        out << "\"tid\":" << event.thread_id;

        if (event.type == EventType::Complete) {
            out << ",\"dur\":" << event.duration_us;
        }

        if (!event.args.empty()) {
            out << ",\"args\":{";
            bool first_arg = true;
            for (const auto& [key, value] : event.args) {
                if (!first_arg) out << ",";
                first_arg = false;
                out << "\"" << key << "\":\"" << value << "\"";
            }
            out << "}";
        }

        out << "}";
    }

    out << "]}";
    return true;
}

std::string Profiler::exportText() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));

    std::stringstream ss;
    ss << "=== Profiler Report ===\n\n";

    ss << "Memory:\n";
    ss << "  Current: " << (current_memory_ / 1024.0 / 1024.0) << " MB\n";
    ss << "  Peak:    " << (peak_memory_ / 1024.0 / 1024.0) << " MB\n";
    ss << "  Allocs:  " << total_allocations_ << "\n\n";

    ss << "Timing Statistics:\n";
    ss << std::setw(40) << std::left << "Name"
       << std::setw(10) << std::right << "Calls"
       << std::setw(15) << "Total (ms)"
       << std::setw(15) << "Avg (us)"
       << std::setw(15) << "Min (us)"
       << std::setw(15) << "Max (us)"
       << "\n";
    ss << std::string(110, '-') << "\n";

    auto stats = getStats();
    for (const auto& s : stats) {
        auto total_ms = std::chrono::duration<double, std::milli>(s.total_time).count();
        auto min_us = std::chrono::duration<double, std::micro>(s.min_time).count();
        auto max_us = std::chrono::duration<double, std::micro>(s.max_time).count();

        ss << std::setw(40) << std::left << s.name
           << std::setw(10) << std::right << s.call_count
           << std::setw(15) << std::fixed << std::setprecision(2) << total_ms
           << std::setw(15) << std::fixed << std::setprecision(2) << s.avg_time_us
           << std::setw(15) << std::fixed << std::setprecision(2) << min_us
           << std::setw(15) << std::fixed << std::setprecision(2) << max_us
           << "\n";
    }

    return ss.str();
}

uint32_t Profiler::getThreadId() const {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return static_cast<uint32_t>(pthread_self());
#endif
}

int64_t Profiler::getTimestampUs() const {
    auto now = Clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

} // namespace eshkol::profiler
```

### Day 97-99: Integration and Testing

**Step 8.4**: Add profiling primitives to parser:

```cpp
// Add to get_operator_type():
if (op == "profile-enable") return ESHKOL_PROFILE_ENABLE_OP;
if (op == "profile-disable") return ESHKOL_PROFILE_DISABLE_OP;
if (op == "profile-begin") return ESHKOL_PROFILE_BEGIN_OP;
if (op == "profile-end") return ESHKOL_PROFILE_END_OP;
if (op == "profile-report") return ESHKOL_PROFILE_REPORT_OP;
if (op == "profile-export") return ESHKOL_PROFILE_EXPORT_OP;
```

**Step 8.5**: Create `tests/profiler/profiler_test.esk`

```scheme
(require stdlib)

(display "=== Profiler Tests ===")
(newline)

;; Enable profiling
(profile-enable)

;; Profile some operations
(profile-begin "matrix_operations")

(define A (make-random-matrix 1000 1000))
(define B (make-random-matrix 1000 1000))

(profile-begin "matmul")
(define C (tensor-matmul A B))
(profile-end "matmul")

(profile-begin "element_ops")
(define D (tensor-add C C))
(define E (tensor-mul D 0.5))
(profile-end "element_ops")

(profile-end "matrix_operations")

;; Profile a training loop
(profile-begin "training")
(do ((i 0 (+ i 1)))
    ((>= i 10))
  (profile-begin "forward")
  ;; Forward pass
  (profile-end "forward")

  (profile-begin "backward")
  ;; Backward pass
  (profile-end "backward")

  (profile-begin "optimizer")
  ;; Optimizer step
  (profile-end "optimizer"))
(profile-end "training")

;; Print report
(display (profile-report))
(newline)

;; Export to Chrome trace format
(profile-export "/tmp/eshkol_trace.json")
(display "Trace exported to /tmp/eshkol_trace.json")
(newline)
(display "Open chrome://tracing and load the file to visualize")
(newline)

;; Memory statistics
(display "Memory usage: ")
(display (profile-memory-current))
(display " bytes")
(newline)

(display "Peak memory: ")
(display (profile-memory-peak))
(display " bytes")
(newline)

(profile-disable)
(display "Profiler tests complete!")
(newline)
```

**Verification Checklist for Phase 8**:
- [ ] Timer accuracy is sub-microsecond
- [ ] Nested profile scopes work correctly
- [ ] Thread-safe for multi-threaded code
- [ ] Chrome trace files are valid JSON
- [ ] Memory tracking catches all allocations
- [ ] Peak memory is accurate
- [ ] Minimal overhead when disabled
- [ ] Statistics are accurate
- [ ] Export to text format is readable
- [ ] GPU operations can be profiled (with CUDA events)

---

## Summary Timeline

```
Week 1-2:   SIMD Vectorization       [12 days]
Week 3-4:   Parallel Execution       [11 days]
Week 5-7:   Hygienic Macro System    [16 days]
Week 8-12:  GPU/CUDA Backend         [25 days]
Week 13-15: Neural Network Primitives[15 days]
Week 16:    OALR Completion          [5 days]
Week 17:    Serialization            [5 days]
Week 18:    Profiling                [6 days]
─────────────────────────────────────────────
TOTAL:                               95 days
```

---

## Post-Completion: Production Readiness

After completing all phases:

1. **Integration Testing** (2 weeks)
   - All features work together
   - Performance regression tests
   - Memory leak detection

2. **Documentation** (2 weeks)
   - API reference
   - Tutorial series
   - Migration guide from v1.0

3. **Security Audit** (1 week)
   - Input validation
   - Memory safety verification

4. **Release v2.0** (1 week)
   - Package for all platforms
   - Update website
   - Announcement

**Total to Production v2.0: ~118 days**
