/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * ParallelCodegen - LLVM code generation for parallel primitives
 *
 * This module generates:
 * 1. Declarations for C runtime parallel functions (parallel_codegen.cpp)
 * 2. LLVM dispatcher functions that handle closure calling conventions:
 *    - __eshkol_call_unary_closure(item, closure) -> result
 *    - __eshkol_call_binary_closure(arg1, arg2, closure) -> result
 *
 * The dispatchers handle the complexity of Eshkol's closure calling convention
 * where captures are passed as pointers with dynamic count (0-32 captures).
 * This allows the C runtime to call closures without knowing capture counts.
 *
 * Parallel primitives:
 * - parallel-map: Apply function in parallel across list elements
 * - parallel-fold: Reduction (sequential, as fold is inherently sequential)
 * - parallel-filter: Filter list elements in parallel
 * - parallel-for-each: Apply side-effecting function in parallel
 *
 * Integration with thread pool for work distribution and thread-local arenas.
 */
#ifndef ESHKOL_BACKEND_PARALLEL_CODEGEN_H
#define ESHKOL_BACKEND_PARALLEL_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

namespace eshkol {

/**
 * ParallelCodegen generates LLVM IR for parallel operations.
 *
 * The parallel primitives call into C runtime functions that use a thread pool
 * for parallel execution. Each worker thread has a thread-local arena for
 * memory allocation.
 *
 * Usage in Scheme:
 *   (parallel-map fn list)         -> Apply fn to each element in parallel
 *   (parallel-fold fn init list)   -> Reduce list with fn (map phase parallel)
 *   (parallel-filter pred list)    -> Keep elements where pred returns true
 *   (parallel-for-each fn list)    -> Apply fn for side effects
 *   (thread-pool-info)             -> Get number of worker threads
 */
class ParallelCodegen {
public:
    /**
     * Construct ParallelCodegen for the given context.
     * Declares all runtime function signatures.
     */
    explicit ParallelCodegen(CodegenContext& ctx);

    // === Parallel Operations ===

    /**
     * Generate code for (parallel-map fn list)
     * @param op The operation AST node
     * @return Result tagged value (new list)
     */
    llvm::Value* parallelMap(const eshkol_operations_t* op);

    /**
     * Generate code for (parallel-fold fn init list)
     * @param op The operation AST node
     * @return Result tagged value (accumulated result)
     */
    llvm::Value* parallelFold(const eshkol_operations_t* op);

    /**
     * Generate code for (parallel-filter pred list)
     * @param op The operation AST node
     * @return Result tagged value (filtered list)
     */
    llvm::Value* parallelFilter(const eshkol_operations_t* op);

    /**
     * Generate code for (parallel-for-each fn list)
     * @param op The operation AST node
     * @return Null value (side-effect only)
     */
    llvm::Value* parallelForEach(const eshkol_operations_t* op);

    /**
     * Generate code for (thread-pool-info)
     * @param op The operation AST node
     * @return Int64 number of worker threads
     */
    llvm::Value* threadPoolInfo(const eshkol_operations_t* op);

    /**
     * Generate code for (thread-pool-stats)
     * Prints thread pool statistics (side effect)
     */
    llvm::Value* threadPoolStats(const eshkol_operations_t* op);

    // === Future Primitives ===

    /**
     * Generate code for (future expr)
     * Wraps expr in a thunk, submits to thread pool, returns future handle
     * @return Tagged value containing future pointer
     */
    llvm::Value* future(const eshkol_operations_t* op);

    /**
     * Generate code for (force fut)
     * Blocks until future completes and returns its result
     * @return The result of the future's computation
     */
    llvm::Value* force(const eshkol_operations_t* op);

    /**
     * Generate code for (future-ready? fut)
     * Non-blocking check if future has completed
     * @return Boolean tagged value
     */
    llvm::Value* futureReady(const eshkol_operations_t* op);

    // === Runtime Function Accessors ===

    llvm::Function* getParallelMapFunc() const { return parallel_map_func_; }
    llvm::Function* getParallelFoldFunc() const { return parallel_fold_func_; }
    llvm::Function* getParallelFilterFunc() const { return parallel_filter_func_; }
    llvm::Function* getParallelForEachFunc() const { return parallel_for_each_func_; }
    llvm::Function* getThreadPoolNumThreadsFunc() const { return thread_pool_num_threads_func_; }
    llvm::Function* getThreadPoolPrintStatsFunc() const { return thread_pool_print_stats_func_; }

private:
    CodegenContext& ctx_;

    // Cached function declarations
    llvm::Function* parallel_map_func_;
    llvm::Function* parallel_fold_func_;
    llvm::Function* parallel_filter_func_;
    llvm::Function* parallel_for_each_func_;
    llvm::Function* thread_pool_num_threads_func_;
    llvm::Function* thread_pool_print_stats_func_;

    // C runtime function declarations
    void declareParallelMap();
    void declareParallelFold();
    void declareParallelFilter();
    void declareParallelForEach();
    void declareThreadPoolInfo();

    // Dispatcher function generation
    // These generate LLVM functions that handle closure calling conventions
    // and are called by the C runtime for parallel operations
    void generateNullaryClosureDispatcher(); // __eshkol_call_nullary_closure (thunks)
    void generateUnaryClosureDispatcher();   // __eshkol_call_unary_closure
    void generateBinaryClosureDispatcher();  // __eshkol_call_binary_closure

    // === Pure LLVM Worker Functions ===
    // These worker functions stay entirely in LLVM IR, eliminating ABI boundaries.
    // They are JIT compiled to native function pointers for the thread pool.

    // Task struct types (mirrors C-side structs for thread pool tasks)
    llvm::StructType* getParallelMapTaskType();
    llvm::StructType* getParallelFoldTaskType();

    // Generate LLVM worker functions: void* worker(void* task_data)
    // These call the dispatchers directly (LLVM→LLVM, no ABI crossing)
    void generateMapWorker();    // __parallel_map_worker
    void generateFoldWorker();   // __parallel_fold_worker
    void generateFilterWorker(); // __parallel_filter_worker

    // Generate module initializer that registers workers with C runtime
    void generateWorkerRegistration();

    // Cached task types and worker functions
    llvm::StructType* parallel_map_task_type_ = nullptr;
    llvm::StructType* parallel_fold_task_type_ = nullptr;
    llvm::Function* map_worker_func_ = nullptr;
    llvm::Function* fold_worker_func_ = nullptr;
    llvm::Function* filter_worker_func_ = nullptr;

    // Dispatcher function references (for workers to call)
    llvm::Function* nullary_dispatcher_func_ = nullptr;  // For thunks (0-arg closures)
    llvm::Function* unary_dispatcher_func_ = nullptr;
    llvm::Function* binary_dispatcher_func_ = nullptr;

    // Helper to get global arena pointer
    llvm::Value* getArenaPtr();

    // Helper to wrap raw numeric types in tagged values
    llvm::Value* ensureTaggedValue(llvm::Value* val);

    // Helper to reverse a list (fixes reversed order from prepend-based building)
    llvm::Value* generateListReversal(llvm::Value* list_val);

    // Callback for AST code generation (uses void* for compatibility with wrapper)
    using CodegenASTCallback = llvm::Value* (*)(const void*, void*);
    CodegenASTCallback codegen_ast_callback_ = nullptr;
    void* callback_context_ = nullptr;

public:
    // === Callback Interface ===
    // These allow the main codegen to inject AST processing callbacks
    void setCodegenASTCallback(CodegenASTCallback callback, void* context);
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_PARALLEL_CODEGEN_H
