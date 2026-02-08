/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * ParallelCodegen implementation - LLVM code generation for parallel primitives
 *
 * This module generates:
 * 1. Declarations for C runtime parallel functions (parallel_codegen.cpp)
 * 2. Dispatcher functions that handle closure calling conventions:
 *    - __eshkol_call_unary_closure(item, closure) -> result
 *    - __eshkol_call_binary_closure(arg1, arg2, closure) -> result
 *
 * The dispatchers use the same logic as codegenClosureCall in llvm_codegen.cpp
 * but as standalone functions callable from C runtime.
 */

#include <eshkol/backend/parallel_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <vector>

namespace eshkol {

// Maximum captures supported (matches llvm_codegen.cpp)
static const int MAX_CAPTURES = 32;

// Track if workers have been generated globally (for JIT mode)
// Once generated in the first module, subsequent modules only need declarations
static bool g_workers_generated = false;

// Helper to check if we should generate full worker definitions or just declarations.
// With pure LLVM approach, we always generate full dispatcher bodies.
// The dispatchers are pure LLVM IR with no external dependencies.
// Workers are no longer used - parallel primitives generate inline loops.
static bool shouldGenerateWorkerBodies(const std::string& module_name) {
    (void)module_name;  // Unused - always generate
    return true;  // Always generate full bodies for pure LLVM approach
}

ParallelCodegen::ParallelCodegen(CodegenContext& ctx)
    : ctx_(ctx)
    , parallel_map_func_(nullptr)
    , parallel_fold_func_(nullptr)
    , parallel_filter_func_(nullptr)
    , parallel_for_each_func_(nullptr)
    , parallel_execute_func_(nullptr)
    , thread_pool_num_threads_func_(nullptr)
    , thread_pool_print_stats_func_(nullptr) {

    // Declare C runtime functions
    declareParallelMap();
    declareParallelFold();
    declareParallelFilter();
    declareParallelForEach();
    declareParallelExecute();
    declareThreadPoolInfo();

    // Generate dispatcher functions that handle closure calling conventions
    generateNullaryClosureDispatcher();  // For thunks (0-arg closures like futures)
    generateUnaryClosureDispatcher();
    generateBinaryClosureDispatcher();

    // Generate pure LLVM worker functions (call dispatchers directly)
    generateMapWorker();
    generateFoldWorker();
    generateFilterWorker();
    generateExecuteWorker();

    // Generate module initializer that registers workers with C runtime
    // This runs at module load time and sets up function pointers
    generateWorkerRegistration();

    // Mark as generated for subsequent JIT modules (they only need declarations)
    if (!g_workers_generated) {
        g_workers_generated = true;
        eshkol_debug("ParallelCodegen: generated full definitions (first module)");
    } else {
        eshkol_debug("ParallelCodegen: created declarations (subsequent JIT module)");
    }
}

// ============================================================================
// C Runtime Function Declarations
// ============================================================================

void ParallelCodegen::declareParallelMap() {
    std::vector<llvm::Type*> args;
    args.push_back(ctx_.taggedValueType());  // fn
    args.push_back(ctx_.taggedValueType());  // list
    args.push_back(ctx_.ptrType());          // arena

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.taggedValueType(), args, false);

    parallel_map_func_ = llvm::Function::Create(
        func_type, llvm::Function::ExternalLinkage,
        "eshkol_parallel_map", &ctx_.module());

    ctx_.defineFunction("eshkol_parallel_map", parallel_map_func_);
}

void ParallelCodegen::declareParallelFold() {
    std::vector<llvm::Type*> args;
    args.push_back(ctx_.taggedValueType());  // fn
    args.push_back(ctx_.taggedValueType());  // init
    args.push_back(ctx_.taggedValueType());  // list
    args.push_back(ctx_.ptrType());          // arena

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.taggedValueType(), args, false);

    parallel_fold_func_ = llvm::Function::Create(
        func_type, llvm::Function::ExternalLinkage,
        "eshkol_parallel_fold", &ctx_.module());

    ctx_.defineFunction("eshkol_parallel_fold", parallel_fold_func_);
}

void ParallelCodegen::declareParallelFilter() {
    std::vector<llvm::Type*> args;
    args.push_back(ctx_.taggedValueType());  // pred
    args.push_back(ctx_.taggedValueType());  // list
    args.push_back(ctx_.ptrType());          // arena

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.taggedValueType(), args, false);

    parallel_filter_func_ = llvm::Function::Create(
        func_type, llvm::Function::ExternalLinkage,
        "eshkol_parallel_filter", &ctx_.module());

    ctx_.defineFunction("eshkol_parallel_filter", parallel_filter_func_);
}

void ParallelCodegen::declareParallelForEach() {
    std::vector<llvm::Type*> args;
    args.push_back(ctx_.taggedValueType());  // fn
    args.push_back(ctx_.taggedValueType());  // list
    args.push_back(ctx_.ptrType());          // arena

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.voidType(), args, false);

    parallel_for_each_func_ = llvm::Function::Create(
        func_type, llvm::Function::ExternalLinkage,
        "eshkol_parallel_for_each", &ctx_.module());

    ctx_.defineFunction("eshkol_parallel_for_each", parallel_for_each_func_);
}

void ParallelCodegen::declareParallelExecute() {
    // eshkol_parallel_execute(thunks_ptr, num_thunks, arena) -> tagged_value (list)
    std::vector<llvm::Type*> args;
    args.push_back(ctx_.ptrType());          // thunks array pointer
    args.push_back(ctx_.int64Type());        // num_thunks
    args.push_back(ctx_.ptrType());          // arena

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.taggedValueType(), args, false);

    parallel_execute_func_ = llvm::Function::Create(
        func_type, llvm::Function::ExternalLinkage,
        "eshkol_parallel_execute", &ctx_.module());

    ctx_.defineFunction("eshkol_parallel_execute", parallel_execute_func_);
}

void ParallelCodegen::declareThreadPoolInfo() {
    // eshkol_thread_pool_num_threads: int64_t(void)
    llvm::FunctionType* num_threads_type = llvm::FunctionType::get(
        ctx_.int64Type(), {}, false);

    thread_pool_num_threads_func_ = llvm::Function::Create(
        num_threads_type, llvm::Function::ExternalLinkage,
        "eshkol_thread_pool_num_threads", &ctx_.module());

    ctx_.defineFunction("eshkol_thread_pool_num_threads", thread_pool_num_threads_func_);

    // eshkol_thread_pool_print_stats: void(void)
    llvm::FunctionType* print_stats_type = llvm::FunctionType::get(
        ctx_.voidType(), {}, false);

    thread_pool_print_stats_func_ = llvm::Function::Create(
        print_stats_type, llvm::Function::ExternalLinkage,
        "eshkol_thread_pool_print_stats", &ctx_.module());

    ctx_.defineFunction("eshkol_thread_pool_print_stats", thread_pool_print_stats_func_);
}

// ============================================================================
// Dispatcher Function Generation
// ============================================================================

/**
 * Generate __eshkol_call_nullary_closure(closure) -> result
 *
 * This function handles calling a thunk (zero-argument closure) with 0-32 captures.
 * Used for futures and delayed evaluation.
 *
 * Calling convention:
 *   result = func_ptr(&cap[0], &cap[1], ..., &cap[N-1])
 */
void ParallelCodegen::generateNullaryClosureDispatcher() {
    // Check if dispatcher already exists in this module
    if (llvm::Function* existing = ctx_.module().getFunction("__eshkol_call_nullary_closure")) {
        nullary_dispatcher_func_ = existing;
        ctx_.defineFunction("__eshkol_call_nullary_closure", existing);
        eshkol_debug("__eshkol_call_nullary_closure already exists in module, reusing");
        return;
    }

    // Function type: (tagged_value) -> tagged_value
    std::vector<llvm::Type*> params = {
        ctx_.taggedValueType()   // closure only (no item for thunks)
    };
    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.taggedValueType(), params, false);

    // Check if we should generate full body or just declaration
    std::string module_name = ctx_.module().getName().str();
    if (!shouldGenerateWorkerBodies(module_name)) {
        llvm::Function* dispatcher = llvm::Function::Create(
            func_type, llvm::Function::ExternalLinkage,
            "__eshkol_call_nullary_closure", &ctx_.module());
        nullary_dispatcher_func_ = dispatcher;
        ctx_.defineFunction("__eshkol_call_nullary_closure", dispatcher);
        eshkol_debug("__eshkol_call_nullary_closure: created declaration (resolved from stdlib.o)");
        return;
    }

    llvm::LLVMContext& llvm_ctx = ctx_.context();
    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Function* dispatcher = llvm::Function::Create(
        func_type, llvm::Function::LinkOnceODRLinkage,
        "__eshkol_call_nullary_closure", &ctx_.module());

    // Get function argument (just closure)
    auto arg_iter = dispatcher->arg_begin();
    llvm::Value* closure_arg = &*arg_iter;
    closure_arg->setName("closure");

    // Create entry block
    llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(llvm_ctx, "entry", dispatcher);
    builder.SetInsertPoint(entry_bb);

    // Extract closure pointer from tagged value (data field at index 4)
    llvm::Value* closure_ptr_i64 = builder.CreateExtractValue(closure_arg, {4}, "closure_ptr_i64");
    llvm::Value* closure_ptr = builder.CreateIntToPtr(closure_ptr_i64,
        llvm::PointerType::getUnqual(llvm_ctx), "closure_ptr");

    // Load func_ptr from closure (offset 0)
    llvm::Value* func_ptr_i64 = builder.CreateLoad(ctx_.int64Type(), closure_ptr, "func_ptr_i64");
    llvm::Value* func_ptr = builder.CreateIntToPtr(func_ptr_i64,
        llvm::PointerType::getUnqual(llvm_ctx), "func_ptr");

    // Load env pointer from closure (offset 8)
    llvm::Value* env_ptr_addr = builder.CreateGEP(ctx_.int8Type(), closure_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8), "env_ptr_addr");
    llvm::Value* env_ptr = builder.CreateLoad(llvm::PointerType::getUnqual(llvm_ctx),
        env_ptr_addr, "env_ptr");

    // Check if env is null (0 captures)
    llvm::Value* env_is_null = builder.CreateICmpEQ(env_ptr,
        llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(llvm_ctx)), "env_is_null");

    llvm::BasicBlock* env_null_bb = llvm::BasicBlock::Create(llvm_ctx, "env_null", dispatcher);
    llvm::BasicBlock* env_valid_bb = llvm::BasicBlock::Create(llvm_ctx, "env_valid", dispatcher);
    llvm::BasicBlock* dispatch_bb = llvm::BasicBlock::Create(llvm_ctx, "dispatch", dispatcher);

    builder.CreateCondBr(env_is_null, env_null_bb, env_valid_bb);

    // Env null path: 0 captures
    builder.SetInsertPoint(env_null_bb);
    llvm::Value* zero_captures = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    builder.CreateBr(dispatch_bb);

    // Env valid path: read num_captures from packed_info
    builder.SetInsertPoint(env_valid_bb);
    llvm::Value* packed_info = builder.CreateLoad(ctx_.int64Type(), env_ptr, "packed_info");
    llvm::Value* num_captures = builder.CreateAnd(packed_info,
        llvm::ConstantInt::get(ctx_.int64Type(), 0xFFFF), "num_captures");
    builder.CreateBr(dispatch_bb);

    // Dispatch block: PHI for capture count, then switch
    builder.SetInsertPoint(dispatch_bb);
    llvm::PHINode* capture_count = builder.CreatePHI(ctx_.int64Type(), 2, "capture_count");
    capture_count->addIncoming(zero_captures, env_null_bb);
    capture_count->addIncoming(num_captures, env_valid_bb);

    // Captures base address (offset 8 from env, after packed_info)
    llvm::Value* captures_base = builder.CreateGEP(ctx_.int8Type(), env_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8), "captures_base");

    // Create switch for dispatch by capture count
    llvm::BasicBlock* default_bb = llvm::BasicBlock::Create(llvm_ctx, "default", dispatcher);
    llvm::SwitchInst* sw = builder.CreateSwitch(capture_count, default_bb, MAX_CAPTURES + 1);

    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(llvm_ctx, "merge", dispatcher);
    std::vector<std::pair<llvm::BasicBlock*, llvm::Value*>> results;

    // Generate cases for 0 to MAX_CAPTURES
    for (int cap = 0; cap <= MAX_CAPTURES; cap++) {
        llvm::BasicBlock* case_bb = llvm::BasicBlock::Create(llvm_ctx,
            "cap_" + std::to_string(cap), dispatcher);
        sw->addCase(llvm::ConstantInt::get(ctx_.int64Type(), cap), case_bb);

        builder.SetInsertPoint(case_bb);

        // Build argument list: (&cap[0], &cap[1], ..., &cap[cap-1]) - NO item for thunks
        std::vector<llvm::Value*> call_args;

        for (int i = 0; i < cap; i++) {
            llvm::Value* cap_ptr = builder.CreateGEP(ctx_.taggedValueType(), captures_base,
                llvm::ConstantInt::get(ctx_.int64Type(), i), "cap_ptr_" + std::to_string(i));
            call_args.push_back(cap_ptr);
        }

        // Build function type: (ptr, ptr, ...) -> tagged_value (no item arg)
        std::vector<llvm::Type*> param_types;
        for (int i = 0; i < cap; i++) {
            param_types.push_back(llvm::PointerType::getUnqual(llvm_ctx));  // capture ptr
        }
        llvm::FunctionType* call_type = llvm::FunctionType::get(
            ctx_.taggedValueType(), param_types, false);

        llvm::Value* result = builder.CreateCall(call_type, func_ptr, call_args, "result");
        builder.CreateBr(merge_bb);
        results.push_back({builder.GetInsertBlock(), result});
    }

    // Default case: return null
    builder.SetInsertPoint(default_bb);
    llvm::Value* null_result = llvm::UndefValue::get(ctx_.taggedValueType());
    null_result = builder.CreateInsertValue(null_result,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), {0});
    null_result = builder.CreateInsertValue(null_result,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    null_result = builder.CreateInsertValue(null_result,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), {4});
    builder.CreateBr(merge_bb);
    results.push_back({default_bb, null_result});

    // Merge block: PHI for result
    builder.SetInsertPoint(merge_bb);
    llvm::PHINode* final_result = builder.CreatePHI(ctx_.taggedValueType(),
        results.size(), "final_result");
    for (auto& [bb, val] : results) {
        final_result->addIncoming(val, bb);
    }

    builder.CreateRet(final_result);

    ctx_.defineFunction("__eshkol_call_nullary_closure", dispatcher);
    nullary_dispatcher_func_ = dispatcher;
    eshkol_debug("Generated __eshkol_call_nullary_closure dispatcher");
}

/**
 * Generate __eshkol_call_unary_closure(item, closure) -> result
 *
 * This function handles calling a closure with 1 argument and 0-32 captures.
 * It extracts the capture count from the closure environment and dispatches
 * to the correct function signature.
 *
 * Calling convention:
 *   result = func_ptr(arg, &cap[0], &cap[1], ..., &cap[N-1])
 */
void ParallelCodegen::generateUnaryClosureDispatcher() {
    // Check if dispatcher already exists in this module
    if (llvm::Function* existing = ctx_.module().getFunction("__eshkol_call_unary_closure")) {
        unary_dispatcher_func_ = existing;
        ctx_.defineFunction("__eshkol_call_unary_closure", existing);
        eshkol_debug("__eshkol_call_unary_closure already exists in module, reusing");
        return;
    }

    // Function type: (tagged_value, tagged_value) -> tagged_value
    std::vector<llvm::Type*> params = {
        ctx_.taggedValueType(),  // item
        ctx_.taggedValueType()   // closure
    };
    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.taggedValueType(), params, false);

    // Check if we should generate full body or just declaration
    std::string module_name = ctx_.module().getName().str();
    if (!shouldGenerateWorkerBodies(module_name)) {
        llvm::Function* dispatcher = llvm::Function::Create(
            func_type, llvm::Function::ExternalLinkage,
            "__eshkol_call_unary_closure", &ctx_.module());
        unary_dispatcher_func_ = dispatcher;
        ctx_.defineFunction("__eshkol_call_unary_closure", dispatcher);
        eshkol_debug("__eshkol_call_unary_closure: created declaration (resolved from stdlib.o)");
        return;
    }

    llvm::LLVMContext& llvm_ctx = ctx_.context();
    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Function* dispatcher = llvm::Function::Create(
        func_type, llvm::Function::LinkOnceODRLinkage,
        "__eshkol_call_unary_closure", &ctx_.module());

    // Get function arguments
    auto arg_iter = dispatcher->arg_begin();
    llvm::Value* item_arg = &*arg_iter++;
    llvm::Value* closure_arg = &*arg_iter;
    item_arg->setName("item");
    closure_arg->setName("closure");

    // Create entry block
    llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(llvm_ctx, "entry", dispatcher);
    builder.SetInsertPoint(entry_bb);

    // Extract closure pointer from tagged value (data field at index 4)
    llvm::Value* closure_ptr_i64 = builder.CreateExtractValue(closure_arg, {4}, "closure_ptr_i64");
    llvm::Value* closure_ptr = builder.CreateIntToPtr(closure_ptr_i64,
        llvm::PointerType::getUnqual(llvm_ctx), "closure_ptr");

    // Load func_ptr from closure (offset 0)
    llvm::Value* func_ptr_i64 = builder.CreateLoad(ctx_.int64Type(), closure_ptr, "func_ptr_i64");
    llvm::Value* func_ptr = builder.CreateIntToPtr(func_ptr_i64,
        llvm::PointerType::getUnqual(llvm_ctx), "func_ptr");

    // Load env pointer from closure (offset 8)
    llvm::Value* env_ptr_addr = builder.CreateGEP(ctx_.int8Type(), closure_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8), "env_ptr_addr");
    llvm::Value* env_ptr = builder.CreateLoad(llvm::PointerType::getUnqual(llvm_ctx),
        env_ptr_addr, "env_ptr");

    // Check if env is null (0 captures)
    llvm::Value* env_is_null = builder.CreateICmpEQ(env_ptr,
        llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(llvm_ctx)), "env_is_null");

    llvm::BasicBlock* env_null_bb = llvm::BasicBlock::Create(llvm_ctx, "env_null", dispatcher);
    llvm::BasicBlock* env_valid_bb = llvm::BasicBlock::Create(llvm_ctx, "env_valid", dispatcher);
    llvm::BasicBlock* dispatch_bb = llvm::BasicBlock::Create(llvm_ctx, "dispatch", dispatcher);

    builder.CreateCondBr(env_is_null, env_null_bb, env_valid_bb);

    // Env null path: 0 captures
    builder.SetInsertPoint(env_null_bb);
    llvm::Value* zero_captures = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    builder.CreateBr(dispatch_bb);

    // Env valid path: read num_captures from packed_info
    builder.SetInsertPoint(env_valid_bb);
    llvm::Value* packed_info = builder.CreateLoad(ctx_.int64Type(), env_ptr, "packed_info");
    llvm::Value* num_captures = builder.CreateAnd(packed_info,
        llvm::ConstantInt::get(ctx_.int64Type(), 0xFFFF), "num_captures");
    builder.CreateBr(dispatch_bb);

    // Dispatch block: PHI for capture count, then switch
    builder.SetInsertPoint(dispatch_bb);
    llvm::PHINode* capture_count = builder.CreatePHI(ctx_.int64Type(), 2, "capture_count");
    capture_count->addIncoming(zero_captures, env_null_bb);
    capture_count->addIncoming(num_captures, env_valid_bb);

    // Captures base address (offset 8 from env, after packed_info)
    llvm::Value* captures_base = builder.CreateGEP(ctx_.int8Type(), env_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8), "captures_base");

    // Create switch for dispatch by capture count
    llvm::BasicBlock* default_bb = llvm::BasicBlock::Create(llvm_ctx, "default", dispatcher);
    llvm::SwitchInst* sw = builder.CreateSwitch(capture_count, default_bb, MAX_CAPTURES + 1);

    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(llvm_ctx, "merge", dispatcher);
    std::vector<std::pair<llvm::BasicBlock*, llvm::Value*>> results;

    // Generate cases for 0 to MAX_CAPTURES
    for (int cap = 0; cap <= MAX_CAPTURES; cap++) {
        llvm::BasicBlock* case_bb = llvm::BasicBlock::Create(llvm_ctx,
            "cap_" + std::to_string(cap), dispatcher);
        sw->addCase(llvm::ConstantInt::get(ctx_.int64Type(), cap), case_bb);

        builder.SetInsertPoint(case_bb);

        // Build argument list: (item, &cap[0], &cap[1], ..., &cap[cap-1])
        std::vector<llvm::Value*> call_args;
        call_args.push_back(item_arg);

        for (int i = 0; i < cap; i++) {
            llvm::Value* cap_ptr = builder.CreateGEP(ctx_.taggedValueType(), captures_base,
                llvm::ConstantInt::get(ctx_.int64Type(), i), "cap_ptr_" + std::to_string(i));
            call_args.push_back(cap_ptr);
        }

        // Build function type: (tagged_value, ptr, ptr, ...) -> tagged_value
        std::vector<llvm::Type*> param_types;
        param_types.push_back(ctx_.taggedValueType());  // item
        for (int i = 0; i < cap; i++) {
            param_types.push_back(llvm::PointerType::getUnqual(llvm_ctx));  // capture ptr
        }
        llvm::FunctionType* call_type = llvm::FunctionType::get(
            ctx_.taggedValueType(), param_types, false);

        llvm::Value* result = builder.CreateCall(call_type, func_ptr, call_args, "result");
        builder.CreateBr(merge_bb);
        results.push_back({builder.GetInsertBlock(), result});
    }

    // Default case: return null
    builder.SetInsertPoint(default_bb);
    llvm::Value* null_result = llvm::UndefValue::get(ctx_.taggedValueType());
    null_result = builder.CreateInsertValue(null_result,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), {0});
    null_result = builder.CreateInsertValue(null_result,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    null_result = builder.CreateInsertValue(null_result,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), {4});
    builder.CreateBr(merge_bb);
    results.push_back({default_bb, null_result});

    // Merge block: PHI for result
    builder.SetInsertPoint(merge_bb);
    llvm::PHINode* final_result = builder.CreatePHI(ctx_.taggedValueType(),
        results.size(), "final_result");
    for (auto& [bb, val] : results) {
        final_result->addIncoming(val, bb);
    }

    builder.CreateRet(final_result);

    ctx_.defineFunction("__eshkol_call_unary_closure", dispatcher);
    unary_dispatcher_func_ = dispatcher;
    eshkol_debug("Generated __eshkol_call_unary_closure dispatcher");
}

/**
 * Generate __eshkol_call_binary_closure(arg1, arg2, closure) -> result
 *
 * This function handles calling a closure with 2 arguments and 0-32 captures.
 * Used for fold operations.
 *
 * Calling convention:
 *   result = func_ptr(arg1, arg2, &cap[0], &cap[1], ..., &cap[N-1])
 */
void ParallelCodegen::generateBinaryClosureDispatcher() {
    // Check if dispatcher already exists in this module
    if (llvm::Function* existing = ctx_.module().getFunction("__eshkol_call_binary_closure")) {
        binary_dispatcher_func_ = existing;
        ctx_.defineFunction("__eshkol_call_binary_closure", existing);
        eshkol_debug("__eshkol_call_binary_closure already exists in module, reusing");
        return;
    }

    // Function type: (tagged_value, tagged_value, tagged_value) -> tagged_value
    std::vector<llvm::Type*> params = {
        ctx_.taggedValueType(),  // arg1
        ctx_.taggedValueType(),  // arg2
        ctx_.taggedValueType()   // closure
    };
    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.taggedValueType(), params, false);

    // Check if we should generate full body or just declaration
    std::string module_name = ctx_.module().getName().str();
    if (!shouldGenerateWorkerBodies(module_name)) {
        llvm::Function* dispatcher = llvm::Function::Create(
            func_type, llvm::Function::ExternalLinkage,
            "__eshkol_call_binary_closure", &ctx_.module());
        binary_dispatcher_func_ = dispatcher;
        ctx_.defineFunction("__eshkol_call_binary_closure", dispatcher);
        eshkol_debug("__eshkol_call_binary_closure: created declaration (resolved from stdlib.o)");
        return;
    }

    llvm::LLVMContext& llvm_ctx = ctx_.context();
    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Function* dispatcher = llvm::Function::Create(
        func_type, llvm::Function::LinkOnceODRLinkage,
        "__eshkol_call_binary_closure", &ctx_.module());

    // Get function arguments
    auto arg_iter = dispatcher->arg_begin();
    llvm::Value* arg1 = &*arg_iter++;
    llvm::Value* arg2 = &*arg_iter++;
    llvm::Value* closure_arg = &*arg_iter;
    arg1->setName("arg1");
    arg2->setName("arg2");
    closure_arg->setName("closure");

    // Create entry block
    llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(llvm_ctx, "entry", dispatcher);
    builder.SetInsertPoint(entry_bb);

    // Extract closure pointer (data field at index 4)
    llvm::Value* closure_ptr_i64 = builder.CreateExtractValue(closure_arg, {4}, "closure_ptr_i64");
    llvm::Value* closure_ptr = builder.CreateIntToPtr(closure_ptr_i64,
        llvm::PointerType::getUnqual(llvm_ctx), "closure_ptr");

    // Load func_ptr (offset 0)
    llvm::Value* func_ptr_i64 = builder.CreateLoad(ctx_.int64Type(), closure_ptr, "func_ptr_i64");
    llvm::Value* func_ptr = builder.CreateIntToPtr(func_ptr_i64,
        llvm::PointerType::getUnqual(llvm_ctx), "func_ptr");

    // Load env pointer (offset 8)
    llvm::Value* env_ptr_addr = builder.CreateGEP(ctx_.int8Type(), closure_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8), "env_ptr_addr");
    llvm::Value* env_ptr = builder.CreateLoad(llvm::PointerType::getUnqual(llvm_ctx),
        env_ptr_addr, "env_ptr");

    // Check if env is null
    llvm::Value* env_is_null = builder.CreateICmpEQ(env_ptr,
        llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(llvm_ctx)), "env_is_null");

    llvm::BasicBlock* env_null_bb = llvm::BasicBlock::Create(llvm_ctx, "env_null", dispatcher);
    llvm::BasicBlock* env_valid_bb = llvm::BasicBlock::Create(llvm_ctx, "env_valid", dispatcher);
    llvm::BasicBlock* dispatch_bb = llvm::BasicBlock::Create(llvm_ctx, "dispatch", dispatcher);

    builder.CreateCondBr(env_is_null, env_null_bb, env_valid_bb);

    // Env null path
    builder.SetInsertPoint(env_null_bb);
    llvm::Value* zero_captures = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    builder.CreateBr(dispatch_bb);

    // Env valid path
    builder.SetInsertPoint(env_valid_bb);
    llvm::Value* packed_info = builder.CreateLoad(ctx_.int64Type(), env_ptr, "packed_info");
    llvm::Value* num_captures = builder.CreateAnd(packed_info,
        llvm::ConstantInt::get(ctx_.int64Type(), 0xFFFF), "num_captures");
    builder.CreateBr(dispatch_bb);

    // Dispatch block
    builder.SetInsertPoint(dispatch_bb);
    llvm::PHINode* capture_count = builder.CreatePHI(ctx_.int64Type(), 2, "capture_count");
    capture_count->addIncoming(zero_captures, env_null_bb);
    capture_count->addIncoming(num_captures, env_valid_bb);

    llvm::Value* captures_base = builder.CreateGEP(ctx_.int8Type(), env_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8), "captures_base");

    // Switch for dispatch
    llvm::BasicBlock* default_bb = llvm::BasicBlock::Create(llvm_ctx, "default", dispatcher);
    llvm::SwitchInst* sw = builder.CreateSwitch(capture_count, default_bb, MAX_CAPTURES + 1);

    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(llvm_ctx, "merge", dispatcher);
    std::vector<std::pair<llvm::BasicBlock*, llvm::Value*>> results;

    // Generate cases for 0 to MAX_CAPTURES
    for (int cap = 0; cap <= MAX_CAPTURES; cap++) {
        llvm::BasicBlock* case_bb = llvm::BasicBlock::Create(llvm_ctx,
            "cap_" + std::to_string(cap), dispatcher);
        sw->addCase(llvm::ConstantInt::get(ctx_.int64Type(), cap), case_bb);

        builder.SetInsertPoint(case_bb);

        // Build argument list: (arg1, arg2, &cap[0], ..., &cap[cap-1])
        std::vector<llvm::Value*> call_args;
        call_args.push_back(arg1);
        call_args.push_back(arg2);

        for (int i = 0; i < cap; i++) {
            llvm::Value* cap_ptr = builder.CreateGEP(ctx_.taggedValueType(), captures_base,
                llvm::ConstantInt::get(ctx_.int64Type(), i), "cap_ptr_" + std::to_string(i));
            call_args.push_back(cap_ptr);
        }

        // Build function type
        std::vector<llvm::Type*> param_types;
        param_types.push_back(ctx_.taggedValueType());  // arg1
        param_types.push_back(ctx_.taggedValueType());  // arg2
        for (int i = 0; i < cap; i++) {
            param_types.push_back(llvm::PointerType::getUnqual(llvm_ctx));
        }
        llvm::FunctionType* call_type = llvm::FunctionType::get(
            ctx_.taggedValueType(), param_types, false);

        llvm::Value* result = builder.CreateCall(call_type, func_ptr, call_args, "result");
        builder.CreateBr(merge_bb);
        results.push_back({builder.GetInsertBlock(), result});
    }

    // Default case
    builder.SetInsertPoint(default_bb);
    llvm::Value* null_result = llvm::UndefValue::get(ctx_.taggedValueType());
    null_result = builder.CreateInsertValue(null_result,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), {0});
    null_result = builder.CreateInsertValue(null_result,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    null_result = builder.CreateInsertValue(null_result,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), {4});
    builder.CreateBr(merge_bb);
    results.push_back({default_bb, null_result});

    // Merge block
    builder.SetInsertPoint(merge_bb);
    llvm::PHINode* final_result = builder.CreatePHI(ctx_.taggedValueType(),
        results.size(), "final_result");
    for (auto& [bb, val] : results) {
        final_result->addIncoming(val, bb);
    }

    builder.CreateRet(final_result);

    ctx_.defineFunction("__eshkol_call_binary_closure", dispatcher);
    binary_dispatcher_func_ = dispatcher;
    eshkol_debug("Generated __eshkol_call_binary_closure dispatcher");
}

// ============================================================================
// Pure LLVM Worker Functions
// ============================================================================

/**
 * Get or create the parallel_map_task struct type.
 *
 * struct parallel_map_task {
 *   i64 closure_ptr;   // offset 0: pointer to closure struct
 *   i64 item_type;     // offset 8: tagged value type field
 *   i64 item_data;     // offset 16: tagged value data field
 *   i64 result_ptr;    // offset 24: pointer to store result (eshkol_tagged_value_t*)
 * }
 */
llvm::StructType* ParallelCodegen::getParallelMapTaskType() {
    if (parallel_map_task_type_) return parallel_map_task_type_;

    parallel_map_task_type_ = llvm::StructType::create(ctx_.context(), {
        ctx_.int64Type(),  // closure_ptr
        ctx_.int64Type(),  // item_type
        ctx_.int64Type(),  // item_data
        ctx_.int64Type()   // result_ptr
    }, "parallel_map_task");

    return parallel_map_task_type_;
}

/**
 * Get or create the parallel_fold_task struct type.
 *
 * struct parallel_fold_task {
 *   i64 closure_ptr;   // offset 0: pointer to closure struct
 *   i64 arg1_type;     // offset 8: first arg type
 *   i64 arg1_data;     // offset 16: first arg data
 *   i64 arg2_type;     // offset 24: second arg type
 *   i64 arg2_data;     // offset 32: second arg data
 *   i64 result_ptr;    // offset 40: pointer to store result
 * }
 */
llvm::StructType* ParallelCodegen::getParallelFoldTaskType() {
    if (parallel_fold_task_type_) return parallel_fold_task_type_;

    parallel_fold_task_type_ = llvm::StructType::create(ctx_.context(), {
        ctx_.int64Type(),  // closure_ptr
        ctx_.int64Type(),  // arg1_type
        ctx_.int64Type(),  // arg1_data
        ctx_.int64Type(),  // arg2_type
        ctx_.int64Type(),  // arg2_data
        ctx_.int64Type()   // result_ptr
    }, "parallel_fold_task");

    return parallel_fold_task_type_;
}

/**
 * Generate __parallel_map_worker(void* arg) -> void*
 *
 * This worker function:
 * 1. Unpacks task data from the void* arg
 * 2. Reconstructs item and closure as tagged values
 * 3. Calls __eshkol_call_unary_closure directly (LLVM→LLVM, no ABI crossing!)
 * 4. Stores result via the result_ptr
 * 5. Returns nullptr
 */
void ParallelCodegen::generateMapWorker() {
    // Check if worker already exists in this module
    if (llvm::Function* existing = ctx_.module().getFunction("__parallel_map_worker")) {
        map_worker_func_ = existing;
        eshkol_debug("__parallel_map_worker already exists in module, reusing");
        return;
    }

    // Function type: void* (void*)
    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.ptrType(), {ctx_.ptrType()}, false);

    // Check if we should generate full body or just declaration
    std::string module_name = ctx_.module().getName().str();
    if (!shouldGenerateWorkerBodies(module_name)) {
        llvm::Function* worker = llvm::Function::Create(
            func_type, llvm::Function::ExternalLinkage,
            "__parallel_map_worker", &ctx_.module());
        map_worker_func_ = worker;
        ctx_.defineFunction("__parallel_map_worker", worker);
        eshkol_debug("__parallel_map_worker: created declaration (resolved from stdlib.o)");
        return;
    }

    llvm::LLVMContext& llvm_ctx = ctx_.context();

    // Save current insert point
    llvm::BasicBlock* saved_bb = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock::iterator saved_pt;
    if (saved_bb) {
        saved_pt = ctx_.builder().GetInsertPoint();
    }

    // Get task type
    llvm::StructType* task_type = getParallelMapTaskType();

    llvm::Function* worker = llvm::Function::Create(
        func_type, llvm::Function::LinkOnceODRLinkage,
        "__parallel_map_worker", &ctx_.module());

    // Get argument
    llvm::Value* arg = worker->arg_begin();
    arg->setName("task_arg");

    // Create entry block
    llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(llvm_ctx, "entry", worker);
    ctx_.builder().SetInsertPoint(entry_bb);

    // Cast void* to task struct pointer
    llvm::Value* task = ctx_.builder().CreateBitCast(arg,
        llvm::PointerType::getUnqual(task_type), "task");

    // Load task fields
    llvm::Value* closure_ptr_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 0), "closure_ptr_i64");
    llvm::Value* item_type = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 1), "item_type");
    llvm::Value* item_data = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 2), "item_data");
    llvm::Value* result_ptr_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 3), "result_ptr_i64");

    // Build item tagged value
    llvm::Value* item = llvm::UndefValue::get(ctx_.taggedValueType());
    item = ctx_.builder().CreateInsertValue(item,
        ctx_.builder().CreateTrunc(item_type, ctx_.int8Type()), {0});  // type
    item = ctx_.builder().CreateInsertValue(item,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});  // flags
    item = ctx_.builder().CreateInsertValue(item, item_data, {4});  // data

    // Build closure tagged value
    llvm::Value* closure = llvm::UndefValue::get(ctx_.taggedValueType());
    closure = ctx_.builder().CreateInsertValue(closure,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE), {0});  // type
    closure = ctx_.builder().CreateInsertValue(closure,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});  // flags
    closure = ctx_.builder().CreateInsertValue(closure, closure_ptr_i64, {4});  // data (ptr as i64)

    // Get the unary dispatcher function
    llvm::Function* dispatcher = ctx_.module().getFunction("__eshkol_call_unary_closure");
    if (!dispatcher) {
        eshkol_error("__parallel_map_worker: dispatcher not found");
        ctx_.builder().CreateRet(llvm::ConstantPointerNull::get(ctx_.ptrType()));

        // Restore insert point
        if (saved_bb) ctx_.builder().SetInsertPoint(saved_bb, saved_pt);
        return;
    }

    // Call dispatcher directly (LLVM→LLVM, no ABI crossing!)
    llvm::Value* result = ctx_.builder().CreateCall(dispatcher, {item, closure}, "result");

    // Store result via pointer
    llvm::Value* result_dest = ctx_.builder().CreateIntToPtr(result_ptr_i64,
        llvm::PointerType::getUnqual(ctx_.taggedValueType()), "result_dest");
    ctx_.builder().CreateStore(result, result_dest);

    // Return nullptr
    ctx_.builder().CreateRet(llvm::ConstantPointerNull::get(ctx_.ptrType()));

    map_worker_func_ = worker;
    ctx_.defineFunction("__parallel_map_worker", worker);
    eshkol_debug("Generated __parallel_map_worker");

    // Restore insert point
    if (saved_bb) ctx_.builder().SetInsertPoint(saved_bb, saved_pt);
}

/**
 * Generate __parallel_fold_worker(void* arg) -> void*
 *
 * Similar to map worker but for binary closure calls.
 */
void ParallelCodegen::generateFoldWorker() {
    // Check if worker already exists in this module
    if (llvm::Function* existing = ctx_.module().getFunction("__parallel_fold_worker")) {
        fold_worker_func_ = existing;
        eshkol_debug("__parallel_fold_worker already exists in module, reusing");
        return;
    }

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.ptrType(), {ctx_.ptrType()}, false);

    // Check if we should generate full body or just declaration
    std::string module_name = ctx_.module().getName().str();
    if (!shouldGenerateWorkerBodies(module_name)) {
        llvm::Function* worker = llvm::Function::Create(
            func_type, llvm::Function::ExternalLinkage,
            "__parallel_fold_worker", &ctx_.module());
        fold_worker_func_ = worker;
        ctx_.defineFunction("__parallel_fold_worker", worker);
        eshkol_debug("__parallel_fold_worker: created declaration (resolved from stdlib.o)");
        return;
    }

    llvm::LLVMContext& llvm_ctx = ctx_.context();

    // Save current insert point
    llvm::BasicBlock* saved_bb = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock::iterator saved_pt;
    if (saved_bb) {
        saved_pt = ctx_.builder().GetInsertPoint();
    }

    llvm::StructType* task_type = getParallelFoldTaskType();

    llvm::Function* worker = llvm::Function::Create(
        func_type, llvm::Function::LinkOnceODRLinkage,
        "__parallel_fold_worker", &ctx_.module());

    llvm::Value* arg = worker->arg_begin();
    arg->setName("task_arg");

    llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(llvm_ctx, "entry", worker);
    ctx_.builder().SetInsertPoint(entry_bb);

    llvm::Value* task = ctx_.builder().CreateBitCast(arg,
        llvm::PointerType::getUnqual(task_type), "task");

    // Load task fields
    llvm::Value* closure_ptr_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 0), "closure_ptr_i64");
    llvm::Value* arg1_type = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 1), "arg1_type");
    llvm::Value* arg1_data = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 2), "arg1_data");
    llvm::Value* arg2_type = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 3), "arg2_type");
    llvm::Value* arg2_data = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 4), "arg2_data");
    llvm::Value* result_ptr_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 5), "result_ptr_i64");

    // Build arg1 tagged value
    llvm::Value* arg1 = llvm::UndefValue::get(ctx_.taggedValueType());
    arg1 = ctx_.builder().CreateInsertValue(arg1,
        ctx_.builder().CreateTrunc(arg1_type, ctx_.int8Type()), {0});
    arg1 = ctx_.builder().CreateInsertValue(arg1,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    arg1 = ctx_.builder().CreateInsertValue(arg1, arg1_data, {4});

    // Build arg2 tagged value
    llvm::Value* arg2 = llvm::UndefValue::get(ctx_.taggedValueType());
    arg2 = ctx_.builder().CreateInsertValue(arg2,
        ctx_.builder().CreateTrunc(arg2_type, ctx_.int8Type()), {0});
    arg2 = ctx_.builder().CreateInsertValue(arg2,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    arg2 = ctx_.builder().CreateInsertValue(arg2, arg2_data, {4});

    // Build closure tagged value
    llvm::Value* closure = llvm::UndefValue::get(ctx_.taggedValueType());
    closure = ctx_.builder().CreateInsertValue(closure,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE), {0});
    closure = ctx_.builder().CreateInsertValue(closure,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    closure = ctx_.builder().CreateInsertValue(closure, closure_ptr_i64, {4});

    // Get binary dispatcher
    llvm::Function* dispatcher = ctx_.module().getFunction("__eshkol_call_binary_closure");
    if (!dispatcher) {
        eshkol_error("__parallel_fold_worker: dispatcher not found");
        ctx_.builder().CreateRet(llvm::ConstantPointerNull::get(ctx_.ptrType()));
        if (saved_bb) ctx_.builder().SetInsertPoint(saved_bb, saved_pt);
        return;
    }

    // Call dispatcher
    llvm::Value* result = ctx_.builder().CreateCall(dispatcher, {arg1, arg2, closure}, "result");

    // Store result
    llvm::Value* result_dest = ctx_.builder().CreateIntToPtr(result_ptr_i64,
        llvm::PointerType::getUnqual(ctx_.taggedValueType()), "result_dest");
    ctx_.builder().CreateStore(result, result_dest);

    ctx_.builder().CreateRet(llvm::ConstantPointerNull::get(ctx_.ptrType()));

    fold_worker_func_ = worker;
    ctx_.defineFunction("__parallel_fold_worker", worker);
    eshkol_debug("Generated __parallel_fold_worker");

    if (saved_bb) ctx_.builder().SetInsertPoint(saved_bb, saved_pt);
}

/**
 * Generate __parallel_filter_worker(void* arg) -> void*
 *
 * Same as map worker (predicate returns boolean).
 */
void ParallelCodegen::generateFilterWorker() {
    // Check if worker already exists in this module
    if (llvm::Function* existing = ctx_.module().getFunction("__parallel_filter_worker")) {
        filter_worker_func_ = existing;
        eshkol_debug("__parallel_filter_worker already exists in module, reusing");
        return;
    }

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.ptrType(), {ctx_.ptrType()}, false);

    // Check if we should generate full body or just declaration
    std::string module_name = ctx_.module().getName().str();
    if (!shouldGenerateWorkerBodies(module_name)) {
        llvm::Function* worker = llvm::Function::Create(
            func_type, llvm::Function::ExternalLinkage,
            "__parallel_filter_worker", &ctx_.module());
        filter_worker_func_ = worker;
        ctx_.defineFunction("__parallel_filter_worker", worker);
        eshkol_debug("__parallel_filter_worker: created declaration (resolved from stdlib.o)");
        return;
    }

    // Filter uses unary closure (predicate), same task structure as map
    llvm::LLVMContext& llvm_ctx = ctx_.context();

    llvm::BasicBlock* saved_bb = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock::iterator saved_pt;
    if (saved_bb) {
        saved_pt = ctx_.builder().GetInsertPoint();
    }

    llvm::StructType* task_type = getParallelMapTaskType();  // Same struct as map

    llvm::Function* worker = llvm::Function::Create(
        func_type, llvm::Function::LinkOnceODRLinkage,
        "__parallel_filter_worker", &ctx_.module());

    llvm::Value* arg = worker->arg_begin();
    arg->setName("task_arg");

    llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(llvm_ctx, "entry", worker);
    ctx_.builder().SetInsertPoint(entry_bb);

    llvm::Value* task = ctx_.builder().CreateBitCast(arg,
        llvm::PointerType::getUnqual(task_type), "task");

    llvm::Value* closure_ptr_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 0), "closure_ptr_i64");
    llvm::Value* item_type = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 1), "item_type");
    llvm::Value* item_data = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 2), "item_data");
    llvm::Value* result_ptr_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(task_type, task, 3), "result_ptr_i64");

    llvm::Value* item = llvm::UndefValue::get(ctx_.taggedValueType());
    item = ctx_.builder().CreateInsertValue(item,
        ctx_.builder().CreateTrunc(item_type, ctx_.int8Type()), {0});
    item = ctx_.builder().CreateInsertValue(item,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    item = ctx_.builder().CreateInsertValue(item, item_data, {4});

    llvm::Value* closure = llvm::UndefValue::get(ctx_.taggedValueType());
    closure = ctx_.builder().CreateInsertValue(closure,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE), {0});
    closure = ctx_.builder().CreateInsertValue(closure,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    closure = ctx_.builder().CreateInsertValue(closure, closure_ptr_i64, {4});

    llvm::Function* dispatcher = ctx_.module().getFunction("__eshkol_call_unary_closure");
    if (!dispatcher) {
        eshkol_error("__parallel_filter_worker: dispatcher not found");
        ctx_.builder().CreateRet(llvm::ConstantPointerNull::get(ctx_.ptrType()));
        if (saved_bb) ctx_.builder().SetInsertPoint(saved_bb, saved_pt);
        return;
    }

    llvm::Value* result = ctx_.builder().CreateCall(dispatcher, {item, closure}, "result");

    llvm::Value* result_dest = ctx_.builder().CreateIntToPtr(result_ptr_i64,
        llvm::PointerType::getUnqual(ctx_.taggedValueType()), "result_dest");
    ctx_.builder().CreateStore(result, result_dest);

    ctx_.builder().CreateRet(llvm::ConstantPointerNull::get(ctx_.ptrType()));

    filter_worker_func_ = worker;
    ctx_.defineFunction("__parallel_filter_worker", worker);
    eshkol_debug("Generated __parallel_filter_worker");

    if (saved_bb) ctx_.builder().SetInsertPoint(saved_bb, saved_pt);
}

// ============================================================================
// Execute Worker (Nullary Thunk Worker)
// ============================================================================

/**
 * Generate __parallel_execute_worker(void* task) -> void*
 *
 * Worker for parallel-execute that calls a nullary closure (thunk).
 * Task struct: { closure_ptr_i64, closure_type_i64, closure_flags_i64, result_ptr_i64 }
 * All fields are i64 to avoid struct-by-value ABI issues.
 */
void ParallelCodegen::generateExecuteWorker() {
    // Check if worker already exists in this module
    if (llvm::Function* existing = ctx_.module().getFunction("__parallel_execute_worker")) {
        execute_worker_func_ = existing;
        eshkol_debug("__parallel_execute_worker already exists in module, reusing");
        return;
    }

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.ptrType(), {ctx_.ptrType()}, false);

    // Check if we should generate full body or just declaration
    std::string module_name = ctx_.module().getName().str();
    if (!shouldGenerateWorkerBodies(module_name)) {
        llvm::Function* worker = llvm::Function::Create(
            func_type, llvm::Function::ExternalLinkage,
            "__parallel_execute_worker", &ctx_.module());
        execute_worker_func_ = worker;
        ctx_.defineFunction("__parallel_execute_worker", worker);
        eshkol_debug("__parallel_execute_worker: created declaration (resolved from stdlib.o)");
        return;
    }

    llvm::LLVMContext& llvm_ctx = ctx_.context();

    llvm::BasicBlock* saved_bb = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock::iterator saved_pt;
    if (saved_bb) {
        saved_pt = ctx_.builder().GetInsertPoint();
    }

    // Task struct: { i64 closure_ptr, i64 closure_type, i64 closure_flags, i64 result_ptr }
    llvm::StructType* exec_task_type = llvm::StructType::create(llvm_ctx,
        {ctx_.int64Type(), ctx_.int64Type(), ctx_.int64Type(), ctx_.int64Type()},
        "parallel_execute_task");

    llvm::Function* worker = llvm::Function::Create(
        func_type, llvm::Function::LinkOnceODRLinkage,
        "__parallel_execute_worker", &ctx_.module());

    llvm::Value* arg = worker->arg_begin();
    arg->setName("task_arg");

    llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(llvm_ctx, "entry", worker);
    ctx_.builder().SetInsertPoint(entry_bb);

    // Cast void* to task struct pointer
    llvm::Value* task = ctx_.builder().CreateBitCast(arg,
        llvm::PointerType::getUnqual(exec_task_type), "task");

    // Load task fields
    llvm::Value* closure_ptr_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(exec_task_type, task, 0), "closure_ptr_i64");
    llvm::Value* closure_type_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(exec_task_type, task, 1), "closure_type_i64");
    llvm::Value* closure_flags_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(exec_task_type, task, 2), "closure_flags_i64");
    llvm::Value* result_ptr_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(),
        ctx_.builder().CreateStructGEP(exec_task_type, task, 3), "result_ptr_i64");

    // Reconstruct closure as tagged value
    llvm::Value* closure = llvm::UndefValue::get(ctx_.taggedValueType());
    closure = ctx_.builder().CreateInsertValue(closure,
        ctx_.builder().CreateTrunc(closure_type_i64, ctx_.int8Type()), {0});
    closure = ctx_.builder().CreateInsertValue(closure,
        ctx_.builder().CreateTrunc(closure_flags_i64, ctx_.int8Type()), {1});
    closure = ctx_.builder().CreateInsertValue(closure,
        llvm::ConstantInt::get(ctx_.int16Type(), 0), {2});  // reserved
    closure = ctx_.builder().CreateInsertValue(closure,
        llvm::ConstantInt::get(ctx_.int32Type(), 0), {3});  // padding
    closure = ctx_.builder().CreateInsertValue(closure, closure_ptr_i64, {4});

    // Call nullary dispatcher (thunks take no args)
    llvm::Function* dispatcher = ctx_.module().getFunction("__eshkol_call_nullary_closure");
    if (!dispatcher) {
        eshkol_error("__parallel_execute_worker: nullary dispatcher not found");
        ctx_.builder().CreateRet(llvm::ConstantPointerNull::get(ctx_.ptrType()));
        if (saved_bb) ctx_.builder().SetInsertPoint(saved_bb, saved_pt);
        return;
    }

    llvm::Value* result = ctx_.builder().CreateCall(dispatcher, {closure}, "result");

    // Store result at result_ptr
    llvm::Value* result_dest = ctx_.builder().CreateIntToPtr(result_ptr_i64,
        llvm::PointerType::getUnqual(ctx_.taggedValueType()), "result_dest");
    ctx_.builder().CreateStore(result, result_dest);

    ctx_.builder().CreateRet(llvm::ConstantPointerNull::get(ctx_.ptrType()));

    execute_worker_func_ = worker;
    ctx_.defineFunction("__parallel_execute_worker", worker);
    eshkol_debug("Generated __parallel_execute_worker");

    if (saved_bb) ctx_.builder().SetInsertPoint(saved_bb, saved_pt);
}

// ============================================================================
// Parallel Primitive Codegen
// ============================================================================

/**
 * Ensure a value is a tagged value.
 * If the input is a raw i64 or double, wrap it in a tagged value struct.
 * If it's already a tagged value struct, return it unchanged.
 */
llvm::Value* ParallelCodegen::ensureTaggedValue(llvm::Value* val) {
    if (!val) return nullptr;

    llvm::Type* val_type = val->getType();

    // Debug: print what type we got
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    val_type->print(rso);
    eshkol_debug("ensureTaggedValue: val_type = %s", rso.str().c_str());

    // Already a tagged value struct
    if (val_type == ctx_.taggedValueType()) {
        eshkol_debug("ensureTaggedValue: already tagged value");
        return val;
    }

    // Raw i64 - wrap as INT64
    if (val_type == ctx_.int64Type()) {
        eshkol_debug("ensureTaggedValue: wrapping i64 as INT64");
        llvm::Value* result = llvm::UndefValue::get(ctx_.taggedValueType());
        result = ctx_.builder().CreateInsertValue(result,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64), {0});
        result = ctx_.builder().CreateInsertValue(result,
            llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
        result = ctx_.builder().CreateInsertValue(result, val, {4});
        return result;
    }

    // Raw double - wrap as DOUBLE
    if (val_type == ctx_.doubleType()) {
        llvm::Value* double_as_i64 = ctx_.builder().CreateBitCast(val, ctx_.int64Type());
        llvm::Value* result = llvm::UndefValue::get(ctx_.taggedValueType());
        result = ctx_.builder().CreateInsertValue(result,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE), {0});
        result = ctx_.builder().CreateInsertValue(result,
            llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
        result = ctx_.builder().CreateInsertValue(result, double_as_i64, {4});
        return result;
    }

    // Pointer type - might be a callable or other heap object
    if (val_type->isPointerTy()) {
        llvm::Value* ptr_as_i64 = ctx_.builder().CreatePtrToInt(val, ctx_.int64Type());
        llvm::Value* result = llvm::UndefValue::get(ctx_.taggedValueType());
        result = ctx_.builder().CreateInsertValue(result,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE), {0});
        result = ctx_.builder().CreateInsertValue(result,
            llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
        result = ctx_.builder().CreateInsertValue(result, ptr_as_i64, {4});
        return result;
    }

    // Unknown type - return as-is and hope for the best
    eshkol_warn("ensureTaggedValue: unknown value type, returning unchanged");
    return val;
}

llvm::Value* ParallelCodegen::getArenaPtr() {
    llvm::GlobalVariable* arena_global = ctx_.module().getNamedGlobal("__global_arena");
    if (!arena_global) {
        eshkol_error("ParallelCodegen: __global_arena not found");
        return nullptr;
    }
    return ctx_.builder().CreateLoad(ctx_.ptrType(), arena_global, "arena");
}

// Generate LLVM IR to reverse a list (tagged value -> tagged value)
// This is used by parallel-map and parallel-filter to correct the reversed order
llvm::Value* ParallelCodegen::generateListReversal(llvm::Value* list_val) {
    llvm::LLVMContext& llvm_ctx = ctx_.context();
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Create basic blocks
    llvm::BasicBlock* null_check_bb = llvm::BasicBlock::Create(llvm_ctx, "rev_null_check", current_func);
    llvm::BasicBlock* loop_bb = llvm::BasicBlock::Create(llvm_ctx, "rev_loop", current_func);
    llvm::BasicBlock* loop_body_bb = llvm::BasicBlock::Create(llvm_ctx, "rev_body", current_func);
    llvm::BasicBlock* done_bb = llvm::BasicBlock::Create(llvm_ctx, "rev_done", current_func);

    // Jump to null check
    ctx_.builder().CreateBr(null_check_bb);

    // === NULL CHECK ===
    ctx_.builder().SetInsertPoint(null_check_bb);
    llvm::Value* list_type = ctx_.builder().CreateExtractValue(list_val, {0}, "list_type");
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(list_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), "is_null");
    ctx_.builder().CreateCondBr(is_null, done_bb, loop_bb);

    // Create null tagged value for empty result
    llvm::Value* null_tagged = llvm::UndefValue::get(ctx_.taggedValueType());
    null_tagged = ctx_.builder().CreateInsertValue(null_tagged,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), {0});
    null_tagged = ctx_.builder().CreateInsertValue(null_tagged,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    null_tagged = ctx_.builder().CreateInsertValue(null_tagged,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), {4});

    // === LOOP ===
    ctx_.builder().SetInsertPoint(loop_bb);

    // PHI for current list element
    llvm::PHINode* current_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "rev_current");
    current_phi->addIncoming(list_val, null_check_bb);

    // PHI for result (reversed list)
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "rev_result");
    result_phi->addIncoming(null_tagged, null_check_bb);

    // Check if current is null (end of list)
    llvm::Value* curr_type = ctx_.builder().CreateExtractValue(current_phi, {0}, "curr_type");
    llvm::Value* curr_is_null = ctx_.builder().CreateICmpEQ(curr_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), "curr_is_null");
    ctx_.builder().CreateCondBr(curr_is_null, done_bb, loop_body_bb);

    // === LOOP BODY ===
    ctx_.builder().SetInsertPoint(loop_body_bb);

    // Get cons cell structure type
    llvm::StructType* cons_type = llvm::StructType::get(llvm_ctx, {
        ctx_.taggedValueType(),  // car
        ctx_.taggedValueType()   // cdr
    });

    // Extract pointer to current cons cell
    llvm::Value* curr_ptr_i64 = ctx_.builder().CreateExtractValue(current_phi, {4}, "curr_ptr_i64");
    llvm::Value* curr_ptr = ctx_.builder().CreateIntToPtr(curr_ptr_i64, ctx_.ptrType(), "curr_ptr");

    // Get car and cdr
    llvm::Value* car_ptr = ctx_.builder().CreateStructGEP(cons_type, curr_ptr, 0, "car_ptr");
    llvm::Value* cdr_ptr = ctx_.builder().CreateStructGEP(cons_type, curr_ptr, 1, "cdr_ptr");
    llvm::Value* car_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), car_ptr, "car_val");
    llvm::Value* cdr_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), cdr_ptr, "cdr_val");

    // Allocate new cons cell: (cons car result)
    llvm::Value* arena_ptr = getArenaPtr();
    llvm::Function* alloc_cons_func = ctx_.module().getFunction("arena_allocate_cons_with_header");
    if (!alloc_cons_func) {
        llvm::FunctionType* alloc_type = llvm::FunctionType::get(ctx_.ptrType(), {ctx_.ptrType()}, false);
        alloc_cons_func = llvm::Function::Create(alloc_type, llvm::Function::ExternalLinkage,
            "arena_allocate_cons_with_header", &ctx_.module());
    }
    llvm::Value* new_cell = ctx_.builder().CreateCall(alloc_cons_func, {arena_ptr}, "rev_new_cell");

    // Store car
    llvm::Value* new_car_ptr = ctx_.builder().CreateStructGEP(cons_type, new_cell, 0, "new_car_ptr");
    ctx_.builder().CreateStore(car_val, new_car_ptr);

    // Store current result as cdr
    llvm::Value* new_cdr_ptr = ctx_.builder().CreateStructGEP(cons_type, new_cell, 1, "new_cdr_ptr");
    ctx_.builder().CreateStore(result_phi, new_cdr_ptr);

    // Create new result tagged value
    llvm::Value* new_cell_i64 = ctx_.builder().CreatePtrToInt(new_cell, ctx_.int64Type(), "new_cell_i64");
    llvm::Value* new_result = llvm::UndefValue::get(ctx_.taggedValueType());
    new_result = ctx_.builder().CreateInsertValue(new_result,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR), {0});
    new_result = ctx_.builder().CreateInsertValue(new_result,
        llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_CONS), {1});
    new_result = ctx_.builder().CreateInsertValue(new_result, new_cell_i64, {4});

    // Update PHIs and loop back
    current_phi->addIncoming(cdr_val, loop_body_bb);
    result_phi->addIncoming(new_result, loop_body_bb);
    ctx_.builder().CreateBr(loop_bb);

    // === DONE ===
    ctx_.builder().SetInsertPoint(done_bb);

    // Final result PHI
    llvm::PHINode* final_result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "rev_final");
    final_result->addIncoming(null_tagged, null_check_bb);
    final_result->addIncoming(result_phi, loop_bb);

    return final_result;
}

llvm::Value* ParallelCodegen::parallelMap(const eshkol_operations_t* op) {
    if (!op || op->call_op.num_vars < 2) {
        eshkol_error("parallel-map requires 2 arguments: fn and list");
        return nullptr;
    }

    if (!codegen_ast_callback_) {
        eshkol_error("ParallelCodegen: codegen callback not set");
        return nullptr;
    }

    // Get fn and list arguments
    llvm::Value* fn_raw = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!fn_raw) {
        eshkol_error("parallel-map: failed to generate fn argument");
        return nullptr;
    }
    llvm::Value* fn_val = ensureTaggedValue(fn_raw);

    llvm::Value* list_raw = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!list_raw) {
        eshkol_error("parallel-map: failed to generate list argument");
        return nullptr;
    }
    llvm::Value* list_val = ensureTaggedValue(list_raw);

    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    // =========================================================================
    // PURE LLVM PARALLEL-MAP IMPLEMENTATION
    // =========================================================================
    // No C runtime - everything generated in LLVM IR
    //
    // Algorithm:
    // 1. Check if list is null → return null
    // 2. Loop over list elements:
    //    - Extract car from cons cell
    //    - Call unary dispatcher (fn car)
    //    - Build result list
    // 3. Return result list

    llvm::LLVMContext& llvm_ctx = ctx_.context();
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Create basic blocks
    llvm::BasicBlock* entry_bb = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock* null_check_bb = llvm::BasicBlock::Create(llvm_ctx, "pmap_null_check", current_func);
    llvm::BasicBlock* loop_bb = llvm::BasicBlock::Create(llvm_ctx, "pmap_loop", current_func);
    llvm::BasicBlock* loop_body_bb = llvm::BasicBlock::Create(llvm_ctx, "pmap_body", current_func);
    llvm::BasicBlock* done_bb = llvm::BasicBlock::Create(llvm_ctx, "pmap_done", current_func);

    // Jump to null check
    ctx_.builder().CreateBr(null_check_bb);

    // === NULL CHECK ===
    ctx_.builder().SetInsertPoint(null_check_bb);
    llvm::Value* list_type = ctx_.builder().CreateExtractValue(list_val, {0}, "list_type");
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(list_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), "is_null");
    ctx_.builder().CreateCondBr(is_null, done_bb, loop_bb);

    // === LOOP HEADER (PHI nodes) ===
    ctx_.builder().SetInsertPoint(loop_bb);

    // PHI for current list element
    llvm::PHINode* current_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "current");
    current_phi->addIncoming(list_val, null_check_bb);

    // PHI for result list head (builds in reverse, we'll handle this)
    llvm::PHINode* result_head_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "result_head");
    llvm::Value* null_tagged = llvm::UndefValue::get(ctx_.taggedValueType());
    null_tagged = ctx_.builder().CreateInsertValue(null_tagged,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), {0});
    null_tagged = ctx_.builder().CreateInsertValue(null_tagged,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    null_tagged = ctx_.builder().CreateInsertValue(null_tagged,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), {4});
    result_head_phi->addIncoming(null_tagged, null_check_bb);

    // Check if current is null (loop exit condition)
    llvm::Value* curr_type = ctx_.builder().CreateExtractValue(current_phi, {0}, "curr_type");
    llvm::Value* curr_is_null = ctx_.builder().CreateICmpEQ(curr_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), "curr_is_null");
    ctx_.builder().CreateCondBr(curr_is_null, done_bb, loop_body_bb);

    // === LOOP BODY ===
    ctx_.builder().SetInsertPoint(loop_body_bb);

    // Get cons cell pointer from current tagged value
    llvm::Value* cell_ptr_i64 = ctx_.builder().CreateExtractValue(current_phi, {4}, "cell_ptr_i64");
    llvm::Value* cell_ptr = ctx_.builder().CreateIntToPtr(cell_ptr_i64, ctx_.ptrType(), "cell_ptr");

    // Define tagged cons cell type: { tagged_value, tagged_value }
    llvm::StructType* cons_type = llvm::StructType::get(llvm_ctx, {
        ctx_.taggedValueType(),  // car
        ctx_.taggedValueType()   // cdr
    });

    // Load car (first 16 bytes of cons cell)
    llvm::Value* car_ptr = ctx_.builder().CreateStructGEP(cons_type, cell_ptr, 0, "car_ptr");
    llvm::Value* car_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), car_ptr, "car_val");

    // Load cdr for next iteration
    llvm::Value* cdr_ptr = ctx_.builder().CreateStructGEP(cons_type, cell_ptr, 1, "cdr_ptr");
    llvm::Value* cdr_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), cdr_ptr, "cdr_val");

    // Call unary dispatcher: result = dispatcher(car, fn)
    llvm::Value* mapped_val = ctx_.builder().CreateCall(unary_dispatcher_func_,
        {car_val, fn_val}, "mapped_val");

    // Allocate new cons cell with header for result
    // Using arena_allocate_cons_with_header so subtype is in header for display
    llvm::Function* alloc_cons_func = ctx_.module().getFunction("arena_allocate_cons_with_header");
    if (!alloc_cons_func) {
        llvm::FunctionType* alloc_type = llvm::FunctionType::get(ctx_.ptrType(), {ctx_.ptrType()}, false);
        alloc_cons_func = llvm::Function::Create(alloc_type, llvm::Function::ExternalLinkage,
            "arena_allocate_cons_with_header", &ctx_.module());
    }

    llvm::Value* new_cell = ctx_.builder().CreateCall(alloc_cons_func, {arena_ptr}, "new_cell");

    // Store mapped value as car
    llvm::Value* new_car_ptr = ctx_.builder().CreateStructGEP(cons_type, new_cell, 0, "new_car_ptr");
    ctx_.builder().CreateStore(mapped_val, new_car_ptr);

    // Store current result_head as cdr (building list in reverse)
    llvm::Value* new_cdr_ptr = ctx_.builder().CreateStructGEP(cons_type, new_cell, 1, "new_cdr_ptr");
    ctx_.builder().CreateStore(result_head_phi, new_cdr_ptr);

    // Create new result head tagged value pointing to new cell
    llvm::Value* new_cell_i64 = ctx_.builder().CreatePtrToInt(new_cell, ctx_.int64Type(), "new_cell_i64");
    llvm::Value* new_result_head = llvm::UndefValue::get(ctx_.taggedValueType());
    new_result_head = ctx_.builder().CreateInsertValue(new_result_head,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR), {0});
    new_result_head = ctx_.builder().CreateInsertValue(new_result_head,
        llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_CONS), {1});
    new_result_head = ctx_.builder().CreateInsertValue(new_result_head, new_cell_i64, {4});

    // Update PHI nodes and loop back
    current_phi->addIncoming(cdr_val, loop_body_bb);
    result_head_phi->addIncoming(new_result_head, loop_body_bb);
    ctx_.builder().CreateBr(loop_bb);

    // === DONE ===
    ctx_.builder().SetInsertPoint(done_bb);

    // Result PHI: either null (empty list) or the built result
    llvm::PHINode* final_result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "final_result");
    final_result->addIncoming(null_tagged, null_check_bb);
    final_result->addIncoming(result_head_phi, loop_bb);

    // The list was built in reverse order - reverse it to get correct order
    llvm::Value* reversed_result = generateListReversal(final_result);

    eshkol_debug("Generated pure LLVM parallel-map");
    return reversed_result;
}

llvm::Value* ParallelCodegen::parallelFold(const eshkol_operations_t* op) {
    if (!op || op->call_op.num_vars < 3) {
        eshkol_error("parallel-fold requires 3 arguments: fn, init, and list");
        return nullptr;
    }

    if (!codegen_ast_callback_) {
        eshkol_error("ParallelCodegen: codegen callback not set");
        return nullptr;
    }

    // Get fn, init, and list arguments
    llvm::Value* fn_raw = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!fn_raw) {
        eshkol_error("parallel-fold: failed to generate fn argument");
        return nullptr;
    }
    llvm::Value* fn_val = ensureTaggedValue(fn_raw);

    llvm::Value* init_raw = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!init_raw) {
        eshkol_error("parallel-fold: failed to generate init argument");
        return nullptr;
    }
    llvm::Value* init_val = ensureTaggedValue(init_raw);

    llvm::Value* list_raw = codegen_ast_callback_(&op->call_op.variables[2], callback_context_);
    if (!list_raw) {
        eshkol_error("parallel-fold: failed to generate list argument");
        return nullptr;
    }
    llvm::Value* list_val = ensureTaggedValue(list_raw);

    // =========================================================================
    // PURE LLVM PARALLEL-FOLD IMPLEMENTATION
    // =========================================================================
    // Sequential fold: acc = fn(acc, item) for each item
    // Fold is inherently sequential for non-associative operations

    llvm::LLVMContext& llvm_ctx = ctx_.context();
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Create basic blocks
    llvm::BasicBlock* null_check_bb = llvm::BasicBlock::Create(llvm_ctx, "pfold_null_check", current_func);
    llvm::BasicBlock* loop_bb = llvm::BasicBlock::Create(llvm_ctx, "pfold_loop", current_func);
    llvm::BasicBlock* loop_body_bb = llvm::BasicBlock::Create(llvm_ctx, "pfold_body", current_func);
    llvm::BasicBlock* done_bb = llvm::BasicBlock::Create(llvm_ctx, "pfold_done", current_func);

    // Jump to null check
    ctx_.builder().CreateBr(null_check_bb);

    // === NULL CHECK ===
    ctx_.builder().SetInsertPoint(null_check_bb);
    llvm::Value* list_type = ctx_.builder().CreateExtractValue(list_val, {0}, "list_type");
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(list_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), "is_null");
    ctx_.builder().CreateCondBr(is_null, done_bb, loop_bb);

    // === LOOP HEADER (PHI nodes) ===
    ctx_.builder().SetInsertPoint(loop_bb);

    // PHI for current list element
    llvm::PHINode* current_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "current");
    current_phi->addIncoming(list_val, null_check_bb);

    // PHI for accumulator
    llvm::PHINode* acc_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "acc");
    acc_phi->addIncoming(init_val, null_check_bb);

    // Check if current is null (loop exit condition)
    llvm::Value* curr_type = ctx_.builder().CreateExtractValue(current_phi, {0}, "curr_type");
    llvm::Value* curr_is_null = ctx_.builder().CreateICmpEQ(curr_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), "curr_is_null");
    ctx_.builder().CreateCondBr(curr_is_null, done_bb, loop_body_bb);

    // === LOOP BODY ===
    ctx_.builder().SetInsertPoint(loop_body_bb);

    // Get cons cell pointer from current tagged value
    llvm::Value* cell_ptr_i64 = ctx_.builder().CreateExtractValue(current_phi, {4}, "cell_ptr_i64");
    llvm::Value* cell_ptr = ctx_.builder().CreateIntToPtr(cell_ptr_i64, ctx_.ptrType(), "cell_ptr");

    // Define tagged cons cell type
    llvm::StructType* cons_type = llvm::StructType::get(llvm_ctx, {
        ctx_.taggedValueType(),  // car
        ctx_.taggedValueType()   // cdr
    });

    // Load car
    llvm::Value* car_ptr = ctx_.builder().CreateStructGEP(cons_type, cell_ptr, 0, "car_ptr");
    llvm::Value* car_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), car_ptr, "car_val");

    // Load cdr for next iteration
    llvm::Value* cdr_ptr = ctx_.builder().CreateStructGEP(cons_type, cell_ptr, 1, "cdr_ptr");
    llvm::Value* cdr_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), cdr_ptr, "cdr_val");

    // Call binary dispatcher: new_acc = fn(acc, item)
    llvm::Value* new_acc = ctx_.builder().CreateCall(binary_dispatcher_func_,
        {acc_phi, car_val, fn_val}, "new_acc");

    // Update PHI nodes and loop back
    current_phi->addIncoming(cdr_val, loop_body_bb);
    acc_phi->addIncoming(new_acc, loop_body_bb);
    ctx_.builder().CreateBr(loop_bb);

    // === DONE ===
    ctx_.builder().SetInsertPoint(done_bb);

    // Final accumulator value
    llvm::PHINode* final_result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "final_result");
    final_result->addIncoming(init_val, null_check_bb);
    final_result->addIncoming(acc_phi, loop_bb);

    eshkol_debug("Generated pure LLVM parallel-fold");
    return final_result;
}

llvm::Value* ParallelCodegen::parallelFilter(const eshkol_operations_t* op) {
    if (!op || op->call_op.num_vars < 2) {
        eshkol_error("parallel-filter requires 2 arguments: pred and list");
        return nullptr;
    }

    if (!codegen_ast_callback_) {
        eshkol_error("ParallelCodegen: codegen callback not set");
        return nullptr;
    }

    // Get pred and list arguments
    llvm::Value* pred_raw = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!pred_raw) {
        eshkol_error("parallel-filter: failed to generate pred argument");
        return nullptr;
    }
    llvm::Value* pred_val = ensureTaggedValue(pred_raw);

    llvm::Value* list_raw = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!list_raw) {
        eshkol_error("parallel-filter: failed to generate list argument");
        return nullptr;
    }
    llvm::Value* list_val = ensureTaggedValue(list_raw);

    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    // =========================================================================
    // PURE LLVM PARALLEL-FILTER IMPLEMENTATION
    // =========================================================================
    // Filter: keep elements where pred returns truthy value

    llvm::LLVMContext& llvm_ctx = ctx_.context();
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Create basic blocks
    llvm::BasicBlock* null_check_bb = llvm::BasicBlock::Create(llvm_ctx, "pfilter_null_check", current_func);
    llvm::BasicBlock* loop_bb = llvm::BasicBlock::Create(llvm_ctx, "pfilter_loop", current_func);
    llvm::BasicBlock* loop_body_bb = llvm::BasicBlock::Create(llvm_ctx, "pfilter_body", current_func);
    llvm::BasicBlock* keep_bb = llvm::BasicBlock::Create(llvm_ctx, "pfilter_keep", current_func);
    llvm::BasicBlock* skip_bb = llvm::BasicBlock::Create(llvm_ctx, "pfilter_skip", current_func);
    llvm::BasicBlock* done_bb = llvm::BasicBlock::Create(llvm_ctx, "pfilter_done", current_func);

    // Jump to null check
    ctx_.builder().CreateBr(null_check_bb);

    // === NULL CHECK ===
    ctx_.builder().SetInsertPoint(null_check_bb);
    llvm::Value* list_type = ctx_.builder().CreateExtractValue(list_val, {0}, "list_type");
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(list_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), "is_null");

    // Create null tagged value for use in multiple places
    llvm::Value* null_tagged = llvm::UndefValue::get(ctx_.taggedValueType());
    null_tagged = ctx_.builder().CreateInsertValue(null_tagged,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), {0});
    null_tagged = ctx_.builder().CreateInsertValue(null_tagged,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    null_tagged = ctx_.builder().CreateInsertValue(null_tagged,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), {4});

    ctx_.builder().CreateCondBr(is_null, done_bb, loop_bb);

    // === LOOP HEADER (PHI nodes) ===
    ctx_.builder().SetInsertPoint(loop_bb);

    // PHI for current list element
    llvm::PHINode* current_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3, "current");
    current_phi->addIncoming(list_val, null_check_bb);

    // PHI for result list head (builds in reverse)
    llvm::PHINode* result_head_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3, "result_head");
    result_head_phi->addIncoming(null_tagged, null_check_bb);

    // Check if current is null (loop exit condition)
    llvm::Value* curr_type = ctx_.builder().CreateExtractValue(current_phi, {0}, "curr_type");
    llvm::Value* curr_is_null = ctx_.builder().CreateICmpEQ(curr_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), "curr_is_null");
    ctx_.builder().CreateCondBr(curr_is_null, done_bb, loop_body_bb);

    // === LOOP BODY ===
    ctx_.builder().SetInsertPoint(loop_body_bb);

    // Get cons cell pointer
    llvm::Value* cell_ptr_i64 = ctx_.builder().CreateExtractValue(current_phi, {4}, "cell_ptr_i64");
    llvm::Value* cell_ptr = ctx_.builder().CreateIntToPtr(cell_ptr_i64, ctx_.ptrType(), "cell_ptr");

    // Define tagged cons cell type
    llvm::StructType* cons_type = llvm::StructType::get(llvm_ctx, {
        ctx_.taggedValueType(),  // car
        ctx_.taggedValueType()   // cdr
    });

    // Load car and cdr
    llvm::Value* car_ptr = ctx_.builder().CreateStructGEP(cons_type, cell_ptr, 0, "car_ptr");
    llvm::Value* car_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), car_ptr, "car_val");
    llvm::Value* cdr_ptr = ctx_.builder().CreateStructGEP(cons_type, cell_ptr, 1, "cdr_ptr");
    llvm::Value* cdr_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), cdr_ptr, "cdr_val");

    // Call predicate: result = pred(item)
    llvm::Value* pred_result = ctx_.builder().CreateCall(unary_dispatcher_func_,
        {car_val, pred_val}, "pred_result");

    // Check if result is truthy (not null and not boolean false)
    llvm::Value* pred_type = ctx_.builder().CreateExtractValue(pred_result, {0}, "pred_type");
    llvm::Value* pred_data = ctx_.builder().CreateExtractValue(pred_result, {4}, "pred_data");

    // Is null?
    llvm::Value* pred_is_null = ctx_.builder().CreateICmpEQ(pred_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), "pred_is_null");
    // Is boolean false?
    llvm::Value* is_bool = ctx_.builder().CreateICmpEQ(pred_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_BOOL), "is_bool");
    llvm::Value* is_false = ctx_.builder().CreateICmpEQ(pred_data,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), "is_false");
    llvm::Value* is_bool_false = ctx_.builder().CreateAnd(is_bool, is_false, "is_bool_false");

    // Truthy = not null AND not boolean false
    llvm::Value* is_falsy = ctx_.builder().CreateOr(pred_is_null, is_bool_false, "is_falsy");
    ctx_.builder().CreateCondBr(is_falsy, skip_bb, keep_bb);

    // === KEEP ELEMENT ===
    ctx_.builder().SetInsertPoint(keep_bb);

    // Allocate new cons cell with header
    llvm::Function* alloc_cons_func = ctx_.module().getFunction("arena_allocate_cons_with_header");
    if (!alloc_cons_func) {
        llvm::FunctionType* alloc_type = llvm::FunctionType::get(ctx_.ptrType(), {ctx_.ptrType()}, false);
        alloc_cons_func = llvm::Function::Create(alloc_type, llvm::Function::ExternalLinkage,
            "arena_allocate_cons_with_header", &ctx_.module());
    }

    llvm::Value* new_cell = ctx_.builder().CreateCall(alloc_cons_func, {arena_ptr}, "new_cell");

    // Store car (the kept element)
    llvm::Value* new_car_ptr = ctx_.builder().CreateStructGEP(cons_type, new_cell, 0, "new_car_ptr");
    ctx_.builder().CreateStore(car_val, new_car_ptr);

    // Store cdr (current result head)
    llvm::Value* new_cdr_ptr = ctx_.builder().CreateStructGEP(cons_type, new_cell, 1, "new_cdr_ptr");
    ctx_.builder().CreateStore(result_head_phi, new_cdr_ptr);

    // Create new result head
    llvm::Value* new_cell_i64 = ctx_.builder().CreatePtrToInt(new_cell, ctx_.int64Type(), "new_cell_i64");
    llvm::Value* new_result_head = llvm::UndefValue::get(ctx_.taggedValueType());
    new_result_head = ctx_.builder().CreateInsertValue(new_result_head,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR), {0});
    new_result_head = ctx_.builder().CreateInsertValue(new_result_head,
        llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_CONS), {1});
    new_result_head = ctx_.builder().CreateInsertValue(new_result_head, new_cell_i64, {4});

    ctx_.builder().CreateBr(loop_bb);

    // === SKIP ELEMENT ===
    ctx_.builder().SetInsertPoint(skip_bb);
    ctx_.builder().CreateBr(loop_bb);

    // Update PHI nodes
    current_phi->addIncoming(cdr_val, keep_bb);
    current_phi->addIncoming(cdr_val, skip_bb);
    result_head_phi->addIncoming(new_result_head, keep_bb);
    result_head_phi->addIncoming(result_head_phi, skip_bb);

    // === DONE ===
    ctx_.builder().SetInsertPoint(done_bb);

    // Final result
    llvm::PHINode* final_result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "final_result");
    final_result->addIncoming(null_tagged, null_check_bb);
    final_result->addIncoming(result_head_phi, loop_bb);

    // The list was built in reverse order - reverse it to get correct order
    llvm::Value* reversed_result = generateListReversal(final_result);

    eshkol_debug("Generated pure LLVM parallel-filter");
    return reversed_result;
}

llvm::Value* ParallelCodegen::parallelForEach(const eshkol_operations_t* op) {
    if (!op || op->call_op.num_vars < 2) {
        eshkol_error("parallel-for-each requires 2 arguments: fn and list");
        return nullptr;
    }

    if (!codegen_ast_callback_) {
        eshkol_error("ParallelCodegen: codegen callback not set");
        return nullptr;
    }

    llvm::Value* fn_raw = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!fn_raw) {
        eshkol_error("parallel-for-each: failed to generate fn argument");
        return nullptr;
    }
    llvm::Value* fn_val = ensureTaggedValue(fn_raw);

    llvm::Value* list_raw = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!list_raw) {
        eshkol_error("parallel-for-each: failed to generate list argument");
        return nullptr;
    }
    llvm::Value* list_val = ensureTaggedValue(list_raw);

    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    ctx_.builder().CreateCall(parallel_for_each_func_, {fn_val, list_val, arena_ptr});

    eshkol_debug("Generated parallel-for-each call");

    // Return null value
    llvm::Value* null_val = llvm::UndefValue::get(ctx_.taggedValueType());
    null_val = ctx_.builder().CreateInsertValue(null_val,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), {0});
    null_val = ctx_.builder().CreateInsertValue(null_val,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    null_val = ctx_.builder().CreateInsertValue(null_val,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), {4});

    return null_val;
}

llvm::Value* ParallelCodegen::parallelExecute(const eshkol_operations_t* op) {
    if (!op || op->call_op.num_vars < 1) {
        eshkol_error("parallel-execute requires at least 1 argument: thunk(s)");
        return nullptr;
    }

    if (!codegen_ast_callback_) {
        eshkol_error("ParallelCodegen: codegen callback not set");
        return nullptr;
    }

    int num_thunks = op->call_op.num_vars;
    eshkol_debug("parallel-execute: generating code for %d thunks", num_thunks);

    // Evaluate all thunk arguments and store in a stack-allocated array
    llvm::Value* array_alloca = ctx_.builder().CreateAlloca(
        ctx_.taggedValueType(),
        llvm::ConstantInt::get(ctx_.int64Type(), num_thunks),
        "thunks_array");

    for (int i = 0; i < num_thunks; ++i) {
        llvm::Value* thunk_raw = codegen_ast_callback_(&op->call_op.variables[i], callback_context_);
        if (!thunk_raw) {
            eshkol_error("parallel-execute: failed to generate thunk argument %d", i);
            return nullptr;
        }
        llvm::Value* thunk_val = ensureTaggedValue(thunk_raw);

        // Store in array
        llvm::Value* slot = ctx_.builder().CreateGEP(
            ctx_.taggedValueType(), array_alloca,
            llvm::ConstantInt::get(ctx_.int64Type(), i),
            "thunk_slot_" + std::to_string(i));
        ctx_.builder().CreateStore(thunk_val, slot);
    }

    // Get arena pointer
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    // Call C runtime: eshkol_parallel_execute(thunks_ptr, num_thunks, arena)
    llvm::Value* result = ctx_.builder().CreateCall(parallel_execute_func_, {
        array_alloca,
        llvm::ConstantInt::get(ctx_.int64Type(), num_thunks),
        arena_ptr
    }, "pexec_result");

    eshkol_debug("Generated parallel-execute call for %d thunks", num_thunks);
    return result;
}

llvm::Value* ParallelCodegen::threadPoolInfo(const eshkol_operations_t* op) {
    (void)op;

    llvm::Value* num_threads = ctx_.builder().CreateCall(
        thread_pool_num_threads_func_, {}, "num_threads");

    // Pack as tagged int64
    llvm::Value* result = llvm::UndefValue::get(ctx_.taggedValueType());
    result = ctx_.builder().CreateInsertValue(result,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64), {0});
    result = ctx_.builder().CreateInsertValue(result,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    result = ctx_.builder().CreateInsertValue(result, num_threads, {4});

    eshkol_debug("Generated thread-pool-info call");
    return result;
}

llvm::Value* ParallelCodegen::threadPoolStats(const eshkol_operations_t* op) {
    (void)op;

    ctx_.builder().CreateCall(thread_pool_print_stats_func_, {});

    // Return null
    llvm::Value* null_val = llvm::UndefValue::get(ctx_.taggedValueType());
    null_val = ctx_.builder().CreateInsertValue(null_val,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), {0});
    null_val = ctx_.builder().CreateInsertValue(null_val,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    null_val = ctx_.builder().CreateInsertValue(null_val,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), {4});

    eshkol_debug("Generated thread-pool-stats call");
    return null_val;
}

void ParallelCodegen::setCodegenASTCallback(CodegenASTCallback callback, void* context) {
    codegen_ast_callback_ = callback;
    callback_context_ = context;
}

// ============================================================================
// Worker Registration (Module Initializer)
// ============================================================================

/**
 * Generate a module initializer that registers the LLVM-generated workers
 * with the C runtime. This allows the C runtime to call workers via function
 * pointers without creating a circular build dependency.
 *
 * Creates:
 *   1. Declaration for __eshkol_register_parallel_workers(void*, void*, void*, void*, void*)
 *   2. Init function __eshkol_init_parallel_workers that passes worker addresses
 *   3. Adds init function to llvm.global_ctors for automatic execution at module load
 */
void ParallelCodegen::generateWorkerRegistration() {
    std::string module_name = ctx_.module().getName().str();

    // Only generate for stdlib - user code resolves symbols from linked stdlib.o
    if (!shouldGenerateWorkerBodies(module_name)) {
        eshkol_debug("Worker registration skipped for module '%s' (not stdlib)", module_name.c_str());
        return;
    }

    // Check that workers were actually generated
    if (!map_worker_func_ || !filter_worker_func_ || !unary_dispatcher_func_ || !binary_dispatcher_func_) {
        eshkol_warn("Worker registration skipped - workers not fully generated");
        return;
    }

    llvm::LLVMContext& llvm_ctx = ctx_.context();

    // Save current insertion point
    llvm::BasicBlock* saved_bb = ctx_.builder().GetInsertBlock();
    auto saved_pt = saved_bb ? ctx_.builder().GetInsertPoint() : llvm::BasicBlock::iterator();

    // 1. Declare __eshkol_register_parallel_workers(void*, void*, void*, void*, void*, void*)
    std::vector<llvm::Type*> reg_args(6, ctx_.ptrType());
    llvm::FunctionType* reg_type = llvm::FunctionType::get(ctx_.voidType(), reg_args, false);
    llvm::Function* register_func = llvm::Function::Create(
        reg_type, llvm::Function::ExternalLinkage,
        "__eshkol_register_parallel_workers", &ctx_.module());

    // 2. Create init function __eshkol_init_parallel_workers
    llvm::FunctionType* init_type = llvm::FunctionType::get(ctx_.voidType(), {}, false);
    llvm::Function* init_func = llvm::Function::Create(
        init_type, llvm::Function::InternalLinkage,
        "__eshkol_init_parallel_workers", &ctx_.module());

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(llvm_ctx, "entry", init_func);
    ctx_.builder().SetInsertPoint(entry);

    // Get worker function pointers (cast to void*)
    llvm::Value* map_ptr = ctx_.builder().CreateBitCast(map_worker_func_, ctx_.ptrType());
    llvm::Value* fold_ptr = fold_worker_func_
        ? ctx_.builder().CreateBitCast(fold_worker_func_, ctx_.ptrType())
        : llvm::ConstantPointerNull::get(ctx_.ptrType());
    llvm::Value* filter_ptr = ctx_.builder().CreateBitCast(filter_worker_func_, ctx_.ptrType());
    llvm::Value* unary_ptr = ctx_.builder().CreateBitCast(unary_dispatcher_func_, ctx_.ptrType());
    llvm::Value* binary_ptr = ctx_.builder().CreateBitCast(binary_dispatcher_func_, ctx_.ptrType());
    llvm::Value* execute_ptr = execute_worker_func_
        ? ctx_.builder().CreateBitCast(execute_worker_func_, ctx_.ptrType())
        : llvm::ConstantPointerNull::get(ctx_.ptrType());

    // Call registration function
    ctx_.builder().CreateCall(register_func, {map_ptr, fold_ptr, filter_ptr, unary_ptr, binary_ptr, execute_ptr});
    ctx_.builder().CreateRetVoid();

    // 3. Add to llvm.global_ctors
    // Structure: { i32 priority, void ()* function, i8* data }
    llvm::StructType* ctor_type = llvm::StructType::get(llvm_ctx, {
        ctx_.builder().getInt32Ty(),
        llvm::PointerType::getUnqual(llvm_ctx),
        ctx_.ptrType()
    });

    // Create constructor entry with priority 65535 (highest, runs first)
    llvm::Constant* ctor_entry = llvm::ConstantStruct::get(ctor_type, {
        llvm::ConstantInt::get(ctx_.builder().getInt32Ty(), 65535),
        init_func,
        llvm::ConstantPointerNull::get(ctx_.ptrType())
    });

    // Get or create llvm.global_ctors array
    llvm::GlobalVariable* ctors = ctx_.module().getNamedGlobal("llvm.global_ctors");
    if (ctors) {
        // Append to existing array
        llvm::Constant* old_init = ctors->getInitializer();
        llvm::ArrayType* old_type = llvm::cast<llvm::ArrayType>(old_init->getType());
        unsigned old_size = old_type->getNumElements();

        std::vector<llvm::Constant*> new_entries;
        for (unsigned i = 0; i < old_size; ++i) {
            new_entries.push_back(old_init->getAggregateElement(i));
        }
        new_entries.push_back(ctor_entry);

        llvm::ArrayType* new_type = llvm::ArrayType::get(ctor_type, new_entries.size());
        llvm::Constant* new_init = llvm::ConstantArray::get(new_type, new_entries);

        ctors->eraseFromParent();
        ctors = new llvm::GlobalVariable(ctx_.module(), new_type, false,
            llvm::GlobalValue::AppendingLinkage, new_init, "llvm.global_ctors");
    } else {
        // Create new array
        llvm::ArrayType* ctors_type = llvm::ArrayType::get(ctor_type, 1);
        llvm::Constant* ctors_init = llvm::ConstantArray::get(ctors_type, {ctor_entry});
        ctors = new llvm::GlobalVariable(ctx_.module(), ctors_type, false,
            llvm::GlobalValue::AppendingLinkage, ctors_init, "llvm.global_ctors");
    }

    eshkol_debug("Generated worker registration in llvm.global_ctors");

    // Restore insertion point
    if (saved_bb) ctx_.builder().SetInsertPoint(saved_bb, saved_pt);
}

// ============================================================================
// Future Primitives Implementation
// ============================================================================

llvm::Value* ParallelCodegen::future(const eshkol_operations_t* op) {
    if (!op || op->call_op.num_vars < 1) {
        eshkol_error("future requires 1 argument: thunk (zero-argument procedure)");
        return nullptr;
    }

    if (!codegen_ast_callback_) {
        eshkol_error("ParallelCodegen: codegen callback not set");
        return nullptr;
    }

    // Get the thunk argument (should be a zero-argument procedure/closure)
    llvm::Value* thunk_raw = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!thunk_raw) {
        eshkol_error("future: failed to generate thunk argument");
        return nullptr;
    }
    llvm::Value* thunk_val = ensureTaggedValue(thunk_raw);

    // Extract the components from tagged value to pass as simple types
    llvm::Value* thunk_ptr = ctx_.builder().CreateExtractValue(thunk_val, {4}, "thunk_ptr");
    llvm::Value* thunk_type = ctx_.builder().CreateExtractValue(thunk_val, {0}, "thunk_type");
    llvm::Value* thunk_flags = ctx_.builder().CreateExtractValue(thunk_val, {1}, "thunk_flags");

    // Get or declare eshkol_lazy_future_create_ptr(ptr, type, flags) -> lazy_future*
    llvm::Function* create_func = ctx_.module().getFunction("eshkol_lazy_future_create_ptr");
    if (!create_func) {
        llvm::FunctionType* create_type = llvm::FunctionType::get(
            ctx_.ptrType(),
            {ctx_.int64Type(), ctx_.int8Type(), ctx_.int8Type()},
            false);
        create_func = llvm::Function::Create(create_type, llvm::Function::ExternalLinkage,
            "eshkol_lazy_future_create_ptr", &ctx_.module());
    }

    // Create lazy future storing the thunk components
    llvm::Value* future_ptr = ctx_.builder().CreateCall(create_func,
        {thunk_ptr, thunk_type, thunk_flags}, "future_ptr");

    // Wrap future pointer in tagged value (HEAP_PTR with HEAP_SUBTYPE_FUTURE)
    llvm::Value* future_i64 = ctx_.builder().CreatePtrToInt(future_ptr, ctx_.int64Type(), "future_i64");
    llvm::Value* result = llvm::UndefValue::get(ctx_.taggedValueType());
    result = ctx_.builder().CreateInsertValue(result,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR), {0});
    result = ctx_.builder().CreateInsertValue(result,
        llvm::ConstantInt::get(ctx_.int8Type(), 10), {1});  // HEAP_SUBTYPE_FUTURE = 10
    result = ctx_.builder().CreateInsertValue(result, future_i64, {4});

    eshkol_debug("Generated lazy future");
    return result;
}

llvm::Value* ParallelCodegen::force(const eshkol_operations_t* op) {
    if (!op || op->call_op.num_vars < 1) {
        eshkol_error("force requires 1 argument: future");
        return nullptr;
    }

    if (!codegen_ast_callback_) {
        eshkol_error("ParallelCodegen: codegen callback not set");
        return nullptr;
    }

    // Get the future argument
    llvm::Value* future_raw = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!future_raw) {
        eshkol_error("force: failed to generate future argument");
        return nullptr;
    }
    llvm::Value* future_val = ensureTaggedValue(future_raw);

    // Extract pointer from tagged value
    llvm::Value* future_i64 = ctx_.builder().CreateExtractValue(future_val, {4}, "future_i64");
    llvm::Value* future_ptr = ctx_.builder().CreateIntToPtr(future_i64, ctx_.ptrType(), "future_ptr");

    llvm::LLVMContext& llvm_ctx = ctx_.context();
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Declare simple helper functions (return primitive types, not structs)
    llvm::Function* is_ready_func = ctx_.module().getFunction("eshkol_lazy_future_is_ready");
    if (!is_ready_func) {
        llvm::FunctionType* ready_type = llvm::FunctionType::get(ctx_.int8Type(), {ctx_.ptrType()}, false);
        is_ready_func = llvm::Function::Create(ready_type, llvm::Function::ExternalLinkage,
            "eshkol_lazy_future_is_ready", &ctx_.module());
    }

    llvm::Function* get_thunk_ptr_func = ctx_.module().getFunction("eshkol_lazy_future_get_thunk_ptr");
    if (!get_thunk_ptr_func) {
        llvm::FunctionType* get_type = llvm::FunctionType::get(ctx_.int64Type(), {ctx_.ptrType()}, false);
        get_thunk_ptr_func = llvm::Function::Create(get_type, llvm::Function::ExternalLinkage,
            "eshkol_lazy_future_get_thunk_ptr", &ctx_.module());
    }

    llvm::Function* get_thunk_type_func = ctx_.module().getFunction("eshkol_lazy_future_get_thunk_type");
    if (!get_thunk_type_func) {
        llvm::FunctionType* get_type = llvm::FunctionType::get(ctx_.int8Type(), {ctx_.ptrType()}, false);
        get_thunk_type_func = llvm::Function::Create(get_type, llvm::Function::ExternalLinkage,
            "eshkol_lazy_future_get_thunk_type", &ctx_.module());
    }

    llvm::Function* get_thunk_flags_func = ctx_.module().getFunction("eshkol_lazy_future_get_thunk_flags");
    if (!get_thunk_flags_func) {
        llvm::FunctionType* get_type = llvm::FunctionType::get(ctx_.int8Type(), {ctx_.ptrType()}, false);
        get_thunk_flags_func = llvm::Function::Create(get_type, llvm::Function::ExternalLinkage,
            "eshkol_lazy_future_get_thunk_flags", &ctx_.module());
    }

    llvm::Function* get_result_ptr_func = ctx_.module().getFunction("eshkol_lazy_future_get_result_ptr");
    if (!get_result_ptr_func) {
        llvm::FunctionType* get_type = llvm::FunctionType::get(ctx_.int64Type(), {ctx_.ptrType()}, false);
        get_result_ptr_func = llvm::Function::Create(get_type, llvm::Function::ExternalLinkage,
            "eshkol_lazy_future_get_result_ptr", &ctx_.module());
    }

    llvm::Function* get_result_type_func = ctx_.module().getFunction("eshkol_lazy_future_get_result_type");
    if (!get_result_type_func) {
        llvm::FunctionType* get_type = llvm::FunctionType::get(ctx_.int8Type(), {ctx_.ptrType()}, false);
        get_result_type_func = llvm::Function::Create(get_type, llvm::Function::ExternalLinkage,
            "eshkol_lazy_future_get_result_type", &ctx_.module());
    }

    llvm::Function* get_result_flags_func = ctx_.module().getFunction("eshkol_lazy_future_get_result_flags");
    if (!get_result_flags_func) {
        llvm::FunctionType* get_type = llvm::FunctionType::get(ctx_.int8Type(), {ctx_.ptrType()}, false);
        get_result_flags_func = llvm::Function::Create(get_type, llvm::Function::ExternalLinkage,
            "eshkol_lazy_future_get_result_flags", &ctx_.module());
    }

    llvm::Function* set_result_func = ctx_.module().getFunction("eshkol_lazy_future_set_result_ptr");
    if (!set_result_func) {
        llvm::FunctionType* set_type = llvm::FunctionType::get(
            ctx_.builder().getVoidTy(),
            {ctx_.ptrType(), ctx_.int64Type(), ctx_.int8Type(), ctx_.int8Type()},
            false);
        set_result_func = llvm::Function::Create(set_type, llvm::Function::ExternalLinkage,
            "eshkol_lazy_future_set_result_ptr", &ctx_.module());
    }

    // Create basic blocks for branching
    llvm::BasicBlock* check_bb = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock* eval_bb = llvm::BasicBlock::Create(llvm_ctx, "force_eval", current_func);
    llvm::BasicBlock* cached_bb = llvm::BasicBlock::Create(llvm_ctx, "force_cached", current_func);
    llvm::BasicBlock* done_bb = llvm::BasicBlock::Create(llvm_ctx, "force_done", current_func);

    // Check if already forced
    llvm::Value* is_ready = ctx_.builder().CreateCall(is_ready_func, {future_ptr}, "is_ready");
    llvm::Value* is_forced = ctx_.builder().CreateICmpNE(is_ready,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), "is_forced");
    ctx_.builder().CreateCondBr(is_forced, cached_bb, eval_bb);

    // === EVAL BLOCK: need to call the thunk ===
    ctx_.builder().SetInsertPoint(eval_bb);

    // Get thunk components and reconstruct tagged value
    llvm::Value* thunk_ptr = ctx_.builder().CreateCall(get_thunk_ptr_func, {future_ptr}, "thunk_ptr");
    llvm::Value* thunk_type = ctx_.builder().CreateCall(get_thunk_type_func, {future_ptr}, "thunk_type");
    llvm::Value* thunk_flags = ctx_.builder().CreateCall(get_thunk_flags_func, {future_ptr}, "thunk_flags");

    // Reconstruct thunk as tagged value
    llvm::Value* thunk = llvm::UndefValue::get(ctx_.taggedValueType());
    thunk = ctx_.builder().CreateInsertValue(thunk, thunk_type, {0});
    thunk = ctx_.builder().CreateInsertValue(thunk, thunk_flags, {1});
    thunk = ctx_.builder().CreateInsertValue(thunk,
        llvm::ConstantInt::get(ctx_.int16Type(), 0), {2});  // reserved
    thunk = ctx_.builder().CreateInsertValue(thunk,
        llvm::ConstantInt::get(ctx_.int32Type(), 0), {3});  // padding
    thunk = ctx_.builder().CreateInsertValue(thunk, thunk_ptr, {4});

    // Call thunk via nullary dispatcher (thunks take no args, only captures)
    llvm::Value* computed_result = ctx_.builder().CreateCall(nullary_dispatcher_func_,
        {thunk}, "computed_result");

    // Extract result components and store
    llvm::Value* result_ptr = ctx_.builder().CreateExtractValue(computed_result, {4}, "result_ptr");
    llvm::Value* result_type = ctx_.builder().CreateExtractValue(computed_result, {0}, "result_type");
    llvm::Value* result_flags = ctx_.builder().CreateExtractValue(computed_result, {1}, "result_flags");
    ctx_.builder().CreateCall(set_result_func, {future_ptr, result_ptr, result_type, result_flags});
    ctx_.builder().CreateBr(done_bb);

    // === CACHED BLOCK: already forced, get cached result ===
    ctx_.builder().SetInsertPoint(cached_bb);

    // Get cached result components and reconstruct tagged value
    llvm::Value* cached_ptr = ctx_.builder().CreateCall(get_result_ptr_func, {future_ptr}, "cached_ptr");
    llvm::Value* cached_type = ctx_.builder().CreateCall(get_result_type_func, {future_ptr}, "cached_type");
    llvm::Value* cached_flags = ctx_.builder().CreateCall(get_result_flags_func, {future_ptr}, "cached_flags");

    llvm::Value* cached_result = llvm::UndefValue::get(ctx_.taggedValueType());
    cached_result = ctx_.builder().CreateInsertValue(cached_result, cached_type, {0});
    cached_result = ctx_.builder().CreateInsertValue(cached_result, cached_flags, {1});
    cached_result = ctx_.builder().CreateInsertValue(cached_result,
        llvm::ConstantInt::get(ctx_.int16Type(), 0), {2});
    cached_result = ctx_.builder().CreateInsertValue(cached_result,
        llvm::ConstantInt::get(ctx_.int32Type(), 0), {3});
    cached_result = ctx_.builder().CreateInsertValue(cached_result, cached_ptr, {4});
    ctx_.builder().CreateBr(done_bb);

    // === DONE BLOCK: merge results ===
    ctx_.builder().SetInsertPoint(done_bb);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "force_result");
    result->addIncoming(computed_result, eval_bb);
    result->addIncoming(cached_result, cached_bb);

    eshkol_debug("Generated force (lazy evaluation)");
    return result;
}

llvm::Value* ParallelCodegen::futureReady(const eshkol_operations_t* op) {
    if (!op || op->call_op.num_vars < 1) {
        eshkol_error("future-ready? requires 1 argument: future");
        return nullptr;
    }

    if (!codegen_ast_callback_) {
        eshkol_error("ParallelCodegen: codegen callback not set");
        return nullptr;
    }

    // Get the future argument
    llvm::Value* future_raw = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!future_raw) {
        eshkol_error("future-ready?: failed to generate future argument");
        return nullptr;
    }
    llvm::Value* future_val = ensureTaggedValue(future_raw);

    // Extract pointer from tagged value
    llvm::Value* future_i64 = ctx_.builder().CreateExtractValue(future_val, {4}, "future_i64");
    llvm::Value* future_ptr = ctx_.builder().CreateIntToPtr(future_i64, ctx_.ptrType(), "future_ptr");

    // Get or declare eshkol_lazy_future_is_ready(future_ptr) -> uint8
    llvm::Function* ready_func = ctx_.module().getFunction("eshkol_lazy_future_is_ready");
    if (!ready_func) {
        llvm::FunctionType* ready_type = llvm::FunctionType::get(
            ctx_.int8Type(),
            {ctx_.ptrType()},
            false);
        ready_func = llvm::Function::Create(ready_type, llvm::Function::ExternalLinkage,
            "eshkol_lazy_future_is_ready", &ctx_.module());
    }

    // Call is_ready - returns 0 or 1
    llvm::Value* is_ready = ctx_.builder().CreateCall(ready_func, {future_ptr}, "is_ready");

    // Convert to boolean (i1)
    llvm::Value* is_ready_bool = ctx_.builder().CreateICmpNE(is_ready,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), "is_ready_bool");

    // Convert boolean to tagged value
    llvm::Value* result = llvm::UndefValue::get(ctx_.taggedValueType());
    result = ctx_.builder().CreateInsertValue(result,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_BOOL), {0});
    result = ctx_.builder().CreateInsertValue(result,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});

    // Store boolean (1 for true, 0 for false) in data field
    llvm::Value* bool_as_i64 = ctx_.builder().CreateZExt(is_ready_bool, ctx_.int64Type(), "bool_i64");
    result = ctx_.builder().CreateInsertValue(result, bool_as_i64, {4});

    eshkol_debug("Generated future-ready? check");
    return result;
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
