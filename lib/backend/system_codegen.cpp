/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * SystemCodegen implementation
 *
 * This module implements system, environment, and file operations.
 */

#include <eshkol/backend/system_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Intrinsics.h>

#if LLVM_VERSION_MAJOR >= 21
#define SYS_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getOrInsertDeclaration(mod, id, types)
#else
#define SYS_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getDeclaration(mod, id, types)
#endif
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>

namespace eshkol {

SystemCodegen::SystemCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem,
                             std::unordered_map<std::string, llvm::Function*>& function_table)
    : ctx_(ctx)
    , tagged_(tagged)
    , mem_(mem)
    , function_table_(function_table) {
    eshkol_debug("SystemCodegen initialized");
}

llvm::Value* SystemCodegen::extractStringPtr(llvm::Value* tagged_val) {
    llvm::Value* ptr_int = tagged_.unpackInt64(tagged_val);
    return ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());
}

// =========================================================================
// ENVIRONMENT OPERATIONS
// =========================================================================

llvm::Value* SystemCodegen::getenv(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("getenv requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* getenv_func = function_table_["getenv"];
    if (!getenv_func) return tagged_.packNull();

    if (!codegen_ast_callback_) {
        eshkol_warn("SystemCodegen::getenv - callback not set");
        return tagged_.packNull();
    }

    // Get name argument
    llvm::Value* name_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!name_arg) return nullptr;

    // Extract string pointer
    llvm::Value* name_ptr = extractStringPtr(name_arg);

    // Call getenv
    llvm::Value* result = ctx_.builder().CreateCall(getenv_func, {name_ptr});

    // Check if result is NULL (variable not set)
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(result,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* null_block = llvm::BasicBlock::Create(ctx_.context(), "getenv_null", current_func);
    llvm::BasicBlock* valid_block = llvm::BasicBlock::Create(ctx_.context(), "getenv_valid", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "getenv_merge", current_func);

    ctx_.builder().CreateCondBr(is_null, null_block, valid_block);

    // NULL case: return #f
    ctx_.builder().SetInsertPoint(null_block);
    llvm::Value* false_val = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    ctx_.builder().CreateBr(merge_block);

    // Valid case: copy libc string into an arena-allocated Scheme string
    // (Bug GG, 2026-05-06). Previously this packed `result` directly as a
    // HEAP_PTR — but `result` points into libc's environ block, which
    // doesn't carry the 8-byte ESHKOL_OBJECT_HEADER our predicates check.
    // `string?` reads the subtype byte at ptr-8 and got garbage from the
    // page above environ, returning #f for legitimately-set vars. The
    // bytes were valid (string-length walked them correctly), but every
    // type-tested code path went down the wrong branch.
    //
    // Copy into arena_allocate_string_with_header(arena, len) so the
    // returned pointer carries HEAP_SUBTYPE_STRING and string? returns #t.
    ctx_.builder().SetInsertPoint(valid_block);
    llvm::Function* getenv_strlen_func = function_table_["strlen"];
    // strcpy isn't always pre-declared in function_table_ (only readFile et al
    // currently force it), so declare on-demand. Declaration is module-scoped
    // and idempotent — getOrInsertFunction reuses an existing decl.
    auto* strcpy_ty = llvm::FunctionType::get(
        ctx_.ptrType(), {ctx_.ptrType(), ctx_.ptrType()}, false);
    llvm::FunctionCallee strcpy_callee =
        ctx_.module().getOrInsertFunction("strcpy", strcpy_ty);
    llvm::Value* string_val;
    if (getenv_strlen_func) {
        llvm::Value* env_len = ctx_.builder().CreateCall(getenv_strlen_func, {result});
        llvm::Value* arena_ptr = ctx_.builder().CreateLoad(
            ctx_.ptrType(), ctx_.globalArena());
        llvm::Value* env_buf = ctx_.builder().CreateCall(
            mem_.getArenaAllocateStringWithHeader(), {arena_ptr, env_len});
        ctx_.builder().CreateCall(strcpy_callee, {env_buf, result});
        string_val = tagged_.packPtr(env_buf, ESHKOL_VALUE_HEAP_PTR);
    } else {
        // strlen unavailable — fall back to raw pack so string-length / etc.
        // still work via byte walk; only string? will continue to lie.
        string_val = tagged_.packPtr(result, ESHKOL_VALUE_HEAP_PTR);
    }
    llvm::BasicBlock* valid_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(merge_block);

    // Merge. valid_exit is captured AFTER the strlen+alloc+strcpy block
    // because those calls created intermediate basic blocks (each
    // arena helper / strlen call may insert blocks for OOM checks etc.),
    // shifting the builder away from the original valid_block.
    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
    phi->addIncoming(false_val, null_block);
    phi->addIncoming(string_val, valid_exit);

    return phi;
}

llvm::Value* SystemCodegen::setenv(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) {
        eshkol_warn("setenv requires exactly 2 arguments (name value)");
        return nullptr;
    }

    llvm::Function* setenv_func = function_table_["setenv"];
    if (!setenv_func) return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));

    if (!codegen_ast_callback_) {
        eshkol_warn("SystemCodegen::setenv - callback not set");
        return tagged_.packNull();
    }

    // Get name and value arguments
    llvm::Value* name_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    llvm::Value* value_arg = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!name_arg || !value_arg) return nullptr;

    // Extract string pointers
    llvm::Value* name_ptr = extractStringPtr(name_arg);
    llvm::Value* value_ptr = extractStringPtr(value_arg);

    // Call setenv(name, value, 1) - overwrite=1
    llvm::Value* result = ctx_.builder().CreateCall(setenv_func, {
        name_ptr, value_ptr, llvm::ConstantInt::get(ctx_.int32Type(), 1)
    });

    // Return #t if result == 0, #f otherwise
    llvm::Value* success = ctx_.builder().CreateICmpEQ(result, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    return tagged_.packBool(success);
}

llvm::Value* SystemCodegen::unsetenv(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("unsetenv requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* unsetenv_func = function_table_["unsetenv"];
    if (!unsetenv_func) return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));

    if (!codegen_ast_callback_) {
        eshkol_warn("SystemCodegen::unsetenv - callback not set");
        return tagged_.packNull();
    }

    // Get name argument
    llvm::Value* name_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!name_arg) return nullptr;

    // Extract string pointer
    llvm::Value* name_ptr = extractStringPtr(name_arg);

    // Call unsetenv
    llvm::Value* result = ctx_.builder().CreateCall(unsetenv_func, {name_ptr});

    // Return #t if result == 0, #f otherwise
    llvm::Value* success = ctx_.builder().CreateICmpEQ(result, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    return tagged_.packBool(success);
}

// =========================================================================
// SYSTEM OPERATIONS
// =========================================================================

llvm::Value* SystemCodegen::systemCall(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("system requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* system_func = function_table_["system"];
    if (!system_func) return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), -1), true);

    if (!codegen_ast_callback_) {
        eshkol_warn("SystemCodegen::systemCall - callback not set");
        return tagged_.packNull();
    }

    // Get command argument
    llvm::Value* cmd_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!cmd_arg) return nullptr;

    // Extract string pointer
    llvm::Value* cmd_ptr = extractStringPtr(cmd_arg);

    // Call system
    llvm::Value* result = ctx_.builder().CreateCall(system_func, {cmd_ptr});

    // Extend to int64 and return
    llvm::Value* result_i64 = ctx_.builder().CreateSExt(result, ctx_.int64Type());
    return tagged_.packInt64(result_i64, true);
}

llvm::Value* SystemCodegen::sleep(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("sleep requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* usleep_func = function_table_["usleep"];
    if (!usleep_func) return tagged_.packNull();

    if (!codegen_typed_ast_callback_) {
        eshkol_warn("SystemCodegen::sleep - callback not set");
        return tagged_.packNull();
    }

    // Get seconds argument via typed AST
    void* seconds_tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!seconds_tv_ptr) return nullptr;

    // Extract raw LLVM value from TypedValue
    llvm::Value* seconds_raw = *reinterpret_cast<llvm::Value**>(seconds_tv_ptr);
    if (!seconds_raw) return nullptr;

    // Convert to double — sleep always works in floating-point seconds
    llvm::Value* seconds_double;
    if (seconds_raw->getType() == ctx_.taggedValueType()) {
        // Tagged value: extract as double (handles both int and double tagged values)
        seconds_double = tagged_.unpackDouble(seconds_raw);
    } else if (seconds_raw->getType()->isDoubleTy()) {
        seconds_double = seconds_raw;
    } else if (seconds_raw->getType()->isIntegerTy()) {
        seconds_double = ctx_.builder().CreateSIToFP(seconds_raw, ctx_.builder().getDoubleTy());
    } else {
        seconds_double = llvm::ConstantFP::get(ctx_.builder().getDoubleTy(), 0.0);
    }

    // Convert seconds to microseconds
    llvm::Value* usec_double = ctx_.builder().CreateFMul(seconds_double,
        llvm::ConstantFP::get(ctx_.builder().getDoubleTy(), 1000000.0));
    // Clamp to valid int32 range before conversion to prevent UB
    llvm::Value* clamped = ctx_.builder().CreateCall(
        SYS_GET_INTRINSIC(&ctx_.module(),
            llvm::Intrinsic::minnum, {ctx_.doubleType()}),
        {usec_double, llvm::ConstantFP::get(ctx_.doubleType(), 2147483647.0)});
    clamped = ctx_.builder().CreateCall(
        SYS_GET_INTRINSIC(&ctx_.module(),
            llvm::Intrinsic::maxnum, {ctx_.doubleType()}),
        {clamped, llvm::ConstantFP::get(ctx_.doubleType(), 0.0)});
    llvm::Value* usec_i64 = ctx_.builder().CreateFPToSI(clamped, ctx_.int64Type());
    llvm::Value* usec = ctx_.builder().CreateTrunc(usec_i64, ctx_.int32Type());

    // Call usleep
    ctx_.builder().CreateCall(usleep_func, {usec});

    return tagged_.packNull();
}

llvm::Value* SystemCodegen::currentSeconds(const eshkol_operations_t* op) {
    llvm::Function* time_func = function_table_["time"];
    if (!time_func) return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);

    // Call time(NULL)
    llvm::Value* null_ptr = llvm::ConstantPointerNull::get(ctx_.ptrType());
    llvm::Value* result = ctx_.builder().CreateCall(time_func, {null_ptr});

    return tagged_.packInt64(result, true);
}

llvm::Value* SystemCodegen::currentTime(const eshkol_operations_t* op) {
    (void)op;

    llvm::Function* gettimeofday_func = function_table_["gettimeofday"];
    if (!gettimeofday_func) {
        // Fallback: integer seconds
        llvm::Function* time_func = function_table_["time"];
        if (time_func) {
            llvm::Value* null_ptr = llvm::ConstantPointerNull::get(ctx_.ptrType());
            llvm::Value* secs = ctx_.builder().CreateCall(time_func, {null_ptr});
            llvm::Value* secs_dbl = ctx_.builder().CreateSIToFP(secs, ctx_.doubleType());
            return tagged_.packDouble(secs_dbl);
        }
        return tagged_.packDouble(llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    }

    // struct timeval { int64 tv_sec; int32 tv_usec; int32 padding; }
    llvm::StructType* timeval_type = llvm::StructType::get(
        ctx_.context(), {ctx_.int64Type(), ctx_.int32Type(), ctx_.int32Type()});

    llvm::Value* tv_alloca = ctx_.builder().CreateAlloca(timeval_type, nullptr, "timeval");
    llvm::Value* null_tz = llvm::ConstantPointerNull::get(ctx_.ptrType());
    ctx_.builder().CreateCall(gettimeofday_func, {tv_alloca, null_tz});

    llvm::Value* sec_ptr = ctx_.builder().CreateStructGEP(timeval_type, tv_alloca, 0);
    llvm::Value* usec_ptr = ctx_.builder().CreateStructGEP(timeval_type, tv_alloca, 1);
    llvm::Value* tv_sec = ctx_.builder().CreateLoad(ctx_.int64Type(), sec_ptr);
    llvm::Value* tv_usec_32 = ctx_.builder().CreateLoad(ctx_.int32Type(), usec_ptr);
    llvm::Value* tv_usec = ctx_.builder().CreateSExt(tv_usec_32, ctx_.int64Type());

    // seconds = tv_sec + tv_usec / 1000000.0
    llvm::Value* sec_dbl = ctx_.builder().CreateSIToFP(tv_sec, ctx_.doubleType());
    llvm::Value* usec_dbl = ctx_.builder().CreateSIToFP(tv_usec, ctx_.doubleType());
    llvm::Value* frac = ctx_.builder().CreateFDiv(usec_dbl,
        llvm::ConstantFP::get(ctx_.doubleType(), 1000000.0));
    llvm::Value* result = ctx_.builder().CreateFAdd(sec_dbl, frac);

    return tagged_.packDouble(result);
}

llvm::Value* SystemCodegen::currentTimeMs(const eshkol_operations_t* op) {
    (void)op;  // No arguments needed

    llvm::Function* gettimeofday_func = function_table_["gettimeofday"];
    if (!gettimeofday_func) {
        // Fallback: return current seconds * 1000
        llvm::Function* time_func = function_table_["time"];
        if (time_func) {
            llvm::Value* null_ptr = llvm::ConstantPointerNull::get(ctx_.ptrType());
            llvm::Value* secs = ctx_.builder().CreateCall(time_func, {null_ptr});
            llvm::Value* ms = ctx_.builder().CreateMul(secs, llvm::ConstantInt::get(ctx_.int64Type(), 1000));
            llvm::Value* ms_double = ctx_.builder().CreateSIToFP(ms, ctx_.doubleType());
            return tagged_.packDouble(ms_double);
        }
        return tagged_.packDouble(llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    }

    // Create struct timeval { int64 tv_sec; int32 tv_usec; int32 padding; }
    // On macOS/Darwin: time_t is long (64-bit), suseconds_t is int32_t
    llvm::StructType* timeval_type = llvm::StructType::get(
        ctx_.context(), {ctx_.int64Type(), ctx_.int32Type(), ctx_.int32Type()});

    // Allocate timeval on stack
    llvm::Value* tv_alloca = ctx_.builder().CreateAlloca(timeval_type, nullptr, "timeval");

    // Call gettimeofday(tv, NULL)
    llvm::Value* null_tz = llvm::ConstantPointerNull::get(ctx_.ptrType());
    ctx_.builder().CreateCall(gettimeofday_func, {tv_alloca, null_tz});

    // Load tv_sec (index 0) and tv_usec (index 1)
    llvm::Value* sec_ptr = ctx_.builder().CreateStructGEP(timeval_type, tv_alloca, 0, "tv_sec_ptr");
    llvm::Value* usec_ptr = ctx_.builder().CreateStructGEP(timeval_type, tv_alloca, 1, "tv_usec_ptr");
    llvm::Value* tv_sec = ctx_.builder().CreateLoad(ctx_.int64Type(), sec_ptr, "tv_sec");
    llvm::Value* tv_usec_32 = ctx_.builder().CreateLoad(ctx_.int32Type(), usec_ptr, "tv_usec_32");
    // Sign-extend to int64 for arithmetic
    llvm::Value* tv_usec = ctx_.builder().CreateSExt(tv_usec_32, ctx_.int64Type(), "tv_usec");

    // Compute milliseconds: tv_sec * 1000 + tv_usec / 1000
    llvm::Value* sec_ms = ctx_.builder().CreateMul(tv_sec,
        llvm::ConstantInt::get(ctx_.int64Type(), 1000), "sec_ms");
    llvm::Value* usec_ms = ctx_.builder().CreateSDiv(tv_usec,
        llvm::ConstantInt::get(ctx_.int64Type(), 1000), "usec_ms");
    llvm::Value* total_ms = ctx_.builder().CreateAdd(sec_ms, usec_ms, "total_ms");

    // Convert to double for return
    llvm::Value* result = ctx_.builder().CreateSIToFP(total_ms, ctx_.doubleType(), "ms_double");

    return tagged_.packDouble(result);
}

llvm::Value* SystemCodegen::currentTimeNs(const eshkol_operations_t* op) {
    (void)op;  // No arguments needed

    // Use clock_gettime for nanosecond precision
    // struct timespec { time_t tv_sec; long tv_nsec; }
    // On macOS arm64: time_t is long (8 bytes), tv_nsec is long (8 bytes)
    llvm::StructType* timespec_type = llvm::StructType::get(
        ctx_.context(), {ctx_.int64Type(), ctx_.int64Type()});

    // Get or declare clock_gettime
    llvm::Function* clock_gettime_func = ctx_.module().getFunction("clock_gettime");
    if (!clock_gettime_func) {
        llvm::FunctionType* cgt_type = llvm::FunctionType::get(
            ctx_.int32Type(),
            {ctx_.int32Type(), ctx_.ptrType()},
            false);
        clock_gettime_func = llvm::Function::Create(
            cgt_type, llvm::Function::ExternalLinkage, "clock_gettime", &ctx_.module());
    }

    llvm::Value* ts_alloca = ctx_.builder().CreateAlloca(timespec_type, nullptr, "timespec");

    // CLOCK_REALTIME = 0 on both macOS and Linux - Unix time
    llvm::Value* clock_id = llvm::ConstantInt::get(ctx_.int32Type(), 0);  // CLOCK_REALTIME

    ctx_.builder().CreateCall(clock_gettime_func, {clock_id, ts_alloca});

    llvm::Value* sec_ptr = ctx_.builder().CreateStructGEP(timespec_type, ts_alloca, 0, "ts_sec_ptr");
    llvm::Value* nsec_ptr = ctx_.builder().CreateStructGEP(timespec_type, ts_alloca, 1, "ts_nsec_ptr");
    llvm::Value* tv_sec = ctx_.builder().CreateLoad(ctx_.int64Type(), sec_ptr, "tv_sec");
    llvm::Value* tv_nsec = ctx_.builder().CreateLoad(ctx_.int64Type(), nsec_ptr, "tv_nsec");

    // Compute nanoseconds: tv_sec * 1000000000 + tv_nsec
    llvm::Value* sec_ns = ctx_.builder().CreateMul(tv_sec,
        llvm::ConstantInt::get(ctx_.int64Type(), 1000000000LL), "sec_ns");
    llvm::Value* total_ns = ctx_.builder().CreateAdd(sec_ns, tv_nsec, "total_ns");

    llvm::Value* result = ctx_.builder().CreateSIToFP(total_ns, ctx_.doubleType(), "ns_double");
    return tagged_.packDouble(result);
}

llvm::Value* SystemCodegen::exitProgram(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("exit requires exactly 1 argument (exit code)");
        return nullptr;
    }

    llvm::Function* exit_func = function_table_["exit"];
    if (!exit_func) return nullptr;

    if (!codegen_typed_ast_callback_) {
        eshkol_warn("SystemCodegen::exitProgram - callback not set");
        return nullptr;
    }

    // Get exit code argument via typed AST
    void* code_tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!code_tv_ptr) return nullptr;

    // Extract raw LLVM value from TypedValue
    llvm::Value* code = *reinterpret_cast<llvm::Value**>(code_tv_ptr);
    if (!code) return nullptr;

    // Convert to i32 if needed
    llvm::Value* code_i32;
    if (code->getType()->isDoubleTy()) {
        // Clamp to valid exit code range [0, 255]
        llvm::Value* clamped_code = ctx_.builder().CreateCall(
            SYS_GET_INTRINSIC(&ctx_.module(),
                llvm::Intrinsic::minnum, {ctx_.doubleType()}),
            {code, llvm::ConstantFP::get(ctx_.doubleType(), 255.0)});
        clamped_code = ctx_.builder().CreateCall(
            SYS_GET_INTRINSIC(&ctx_.module(),
                llvm::Intrinsic::maxnum, {ctx_.doubleType()}),
            {clamped_code, llvm::ConstantFP::get(ctx_.doubleType(), 0.0)});
        code_i32 = ctx_.builder().CreateFPToSI(clamped_code, ctx_.int32Type());
    } else if (code->getType()->isIntegerTy(64)) {
        code_i32 = ctx_.builder().CreateTrunc(code, ctx_.int32Type());
    } else {
        code_i32 = code;
    }

    // Call exit(code)
    ctx_.builder().CreateCall(exit_func, {code_i32});

    // exit() doesn't return, but we need something for the compiler
    ctx_.builder().CreateUnreachable();
    return nullptr;
}

llvm::Value* SystemCodegen::commandLine(const eshkol_operations_t* op) {
    // The argc/argv globals are owned by the host process and exposed to
    // JIT via ADD_DATA_SYMBOL (lib/repl/repl_jit.cpp:558-559). The
    // top-level main-emitting path declares them, but each REPL `-e`
    // expression is its own module — so on a bare `(command-line)` from
    // the REPL we have to declare the externs ourselves. Without this,
    // commandLine returned NULL and the user saw an empty list even when
    // argc was clearly non-zero.
    llvm::Module* module = ctx_.builder().GetInsertBlock()->getParent()->getParent();

    llvm::GlobalVariable* g_argc = module->getGlobalVariable("__eshkol_argc");
    if (!g_argc) {
        g_argc = new llvm::GlobalVariable(
            *module, ctx_.int32Type(), false,
            llvm::GlobalValue::ExternalLinkage, nullptr,
            "__eshkol_argc");
    }
    llvm::GlobalVariable* g_argv = module->getGlobalVariable("__eshkol_argv");
    if (!g_argv) {
        g_argv = new llvm::GlobalVariable(
            *module, ctx_.ptrType(), false,
            llvm::GlobalValue::ExternalLinkage, nullptr,
            "__eshkol_argv");
    }

    llvm::Function* strlen_func = function_table_["strlen"];
    llvm::Function* strcpy_func = ctx_.funcs().getStrcpy();
    if (!strlen_func || !strcpy_func) {
        return tagged_.packNull();
    }

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Load argc and argv
    llvm::Value* argc = ctx_.builder().CreateLoad(ctx_.int32Type(), g_argc);
    llvm::Value* argv = ctx_.builder().CreateLoad(ctx_.ptrType(), g_argv);

    // Create allocas at function entry
    llvm::IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
    llvm::BasicBlock& entry = current_func->getEntryBlock();
    ctx_.builder().SetInsertPoint(&entry, entry.begin());
    llvm::Value* result_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "cmdline_result");
    llvm::Value* idx_ptr = ctx_.builder().CreateAlloca(ctx_.int32Type(), nullptr, "cmdline_idx");
    ctx_.builder().restoreIP(saved_ip);

    // Initialize result to null (empty list), idx to argc-1 (build in reverse)
    ctx_.builder().CreateStore(tagged_.packNull(), result_ptr);
    llvm::Value* start_idx = ctx_.builder().CreateSub(argc, llvm::ConstantInt::get(ctx_.int32Type(), 1));
    ctx_.builder().CreateStore(start_idx, idx_ptr);

    // Create loop blocks
    llvm::BasicBlock* loop_block = llvm::BasicBlock::Create(ctx_.context(), "cmdline_loop", current_func);
    llvm::BasicBlock* body_block = llvm::BasicBlock::Create(ctx_.context(), "cmdline_body", current_func);
    llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "cmdline_done", current_func);

    ctx_.builder().CreateBr(loop_block);

    // Loop condition: while (idx >= 0)
    ctx_.builder().SetInsertPoint(loop_block);
    llvm::Value* idx = ctx_.builder().CreateLoad(ctx_.int32Type(), idx_ptr);
    llvm::Value* cond = ctx_.builder().CreateICmpSGE(idx, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    ctx_.builder().CreateCondBr(cond, body_block, done_block);

    // Loop body: cons argv[idx] onto result
    ctx_.builder().SetInsertPoint(body_block);
    llvm::Value* current_idx = ctx_.builder().CreateLoad(ctx_.int32Type(), idx_ptr);

    // Get argv[idx]
    llvm::Value* idx_64 = ctx_.builder().CreateSExt(current_idx, ctx_.int64Type());
    llvm::Value* arg_ptr_ptr = ctx_.builder().CreateGEP(ctx_.ptrType(), argv, idx_64);
    llvm::Value* arg_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), arg_ptr_ptr);

    // Copy argv[idx] into an arena-allocated Scheme string. Must use
    // `arena_allocate_string_with_header` (NOT raw `arena_allocate`) so
    // the string carries the 8-byte ESHKOL_OBJECT_HEADER that
    // (display), (string-length), (string-ref), and the rest of the
    // string toolkit rely on. Without the header the user sees garbage
    // (`#<unknown>`) when displaying argv. The allocator reserves the
    // +1 NUL byte itself, so we pass the bare strlen.
    llvm::Value* arg_len = ctx_.builder().CreateCall(strlen_func, {arg_ptr});
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    llvm::Value* new_str = ctx_.builder().CreateCall(
        mem_.getArenaAllocateStringWithHeader(), {arena_ptr, arg_len});
    ctx_.builder().CreateCall(strcpy_func, {new_str, arg_ptr});

    // Pack string and cons onto list
    llvm::Value* str_tagged = tagged_.packPtr(new_str, ESHKOL_VALUE_HEAP_PTR);
    llvm::Value* current_list = ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_ptr);

    // Allocate cons cell with object header (consolidated pointer format)
    llvm::Value* cons = ctx_.builder().CreateCall(mem_.getArenaAllocateConsWithHeader(), {arena_ptr});

    // Set car to string, cdr to current list
    // Note: is_cdr parameter is i1 (bool), not i32
    llvm::Value* is_car = llvm::ConstantInt::getFalse(ctx_.context());  // false = car
    llvm::Value* is_cdr = llvm::ConstantInt::getTrue(ctx_.context());   // true = cdr

    llvm::Value* str_ptr_alloca = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
    llvm::Value* list_ptr_alloca = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
    ctx_.builder().CreateStore(str_tagged, str_ptr_alloca);
    ctx_.builder().CreateStore(current_list, list_ptr_alloca);

    ctx_.builder().CreateCall(mem_.getTaggedConsSetTaggedValue(), {cons, is_car, str_ptr_alloca});
    ctx_.builder().CreateCall(mem_.getTaggedConsSetTaggedValue(), {cons, is_cdr, list_ptr_alloca});

    // Update result (HEAP_PTR - consolidated pointer format)
    llvm::Value* new_list = tagged_.packHeapPtr(cons);
    ctx_.builder().CreateStore(new_list, result_ptr);

    // Decrement index
    llvm::Value* new_idx = ctx_.builder().CreateSub(current_idx, llvm::ConstantInt::get(ctx_.int32Type(), 1));
    ctx_.builder().CreateStore(new_idx, idx_ptr);
    ctx_.builder().CreateBr(loop_block);

    // Done
    ctx_.builder().SetInsertPoint(done_block);
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_ptr);
}

// =========================================================================
// FILE OPERATIONS
// =========================================================================

llvm::Value* SystemCodegen::fileExists(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("file-exists? requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* access_func = function_table_["access"];
    if (!access_func) return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));

    if (!codegen_ast_callback_) {
        eshkol_warn("SystemCodegen::fileExists - callback not set");
        return tagged_.packNull();
    }

    // Get path argument
    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!path_arg) return nullptr;

    // Extract string pointer
    llvm::Value* path_ptr = extractStringPtr(path_arg);

    // Call access(path, F_OK) - F_OK = 0
    llvm::Value* result = ctx_.builder().CreateCall(access_func, {
        path_ptr, llvm::ConstantInt::get(ctx_.int32Type(), 0)  // F_OK
    });

    // Return #t if result == 0
    llvm::Value* exists = ctx_.builder().CreateICmpEQ(result, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    return tagged_.packBool(exists);
}

llvm::Value* SystemCodegen::fileReadable(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("file-readable? requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* access_func = function_table_["access"];
    if (!access_func) return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!path_arg) return nullptr;

    llvm::Value* path_ptr = extractStringPtr(path_arg);

    // Call access(path, R_OK) - R_OK = 4
    llvm::Value* result = ctx_.builder().CreateCall(access_func, {
        path_ptr, llvm::ConstantInt::get(ctx_.int32Type(), 4)  // R_OK
    });

    llvm::Value* readable = ctx_.builder().CreateICmpEQ(result, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    return tagged_.packBool(readable);
}

llvm::Value* SystemCodegen::fileWritable(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("file-writable? requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* access_func = function_table_["access"];
    if (!access_func) return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!path_arg) return nullptr;

    llvm::Value* path_ptr = extractStringPtr(path_arg);

    // Call access(path, W_OK) - W_OK = 2
    llvm::Value* result = ctx_.builder().CreateCall(access_func, {
        path_ptr, llvm::ConstantInt::get(ctx_.int32Type(), 2)  // W_OK
    });

    llvm::Value* writable = ctx_.builder().CreateICmpEQ(result, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    return tagged_.packBool(writable);
}

llvm::Value* SystemCodegen::fileDelete(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("file-delete requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* remove_func = function_table_["remove"];
    if (!remove_func) return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!path_arg) return nullptr;

    llvm::Value* path_ptr = extractStringPtr(path_arg);

    // Call remove(path)
    llvm::Value* result = ctx_.builder().CreateCall(remove_func, {path_ptr});

    llvm::Value* success = ctx_.builder().CreateICmpEQ(result, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    return tagged_.packBool(success);
}

llvm::Value* SystemCodegen::fileRename(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) {
        eshkol_warn("file-rename requires exactly 2 arguments");
        return nullptr;
    }

    llvm::Function* rename_func = function_table_["rename"];
    if (!rename_func) return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* old_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    llvm::Value* new_arg = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!old_arg || !new_arg) return nullptr;

    llvm::Value* old_ptr = extractStringPtr(old_arg);
    llvm::Value* new_ptr = extractStringPtr(new_arg);

    // Call rename(old, new)
    llvm::Value* result = ctx_.builder().CreateCall(rename_func, {old_ptr, new_ptr});

    llvm::Value* success = ctx_.builder().CreateICmpEQ(result, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    return tagged_.packBool(success);
}

llvm::Value* SystemCodegen::fileSize(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("file-size requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* fopen_func = function_table_["fopen"];
    llvm::Function* fseek_func = function_table_["fseek"];
    llvm::Function* ftell_func = function_table_["ftell"];
    llvm::Function* fclose_func = function_table_["fclose"];
    if (!fopen_func || !fseek_func || !ftell_func || !fclose_func) {
        return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    }

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!path_arg) return nullptr;

    llvm::Value* path_ptr = extractStringPtr(path_arg);

    // Open file in read mode
    llvm::GlobalVariable* mode_str = ctx_.internString("rb");
    llvm::Value* mode = ctx_.builder().CreatePointerCast(mode_str, ctx_.ptrType());
    llvm::Value* file = ctx_.builder().CreateCall(fopen_func, {path_ptr, mode});

    // Check if open succeeded
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(file,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* error_block = llvm::BasicBlock::Create(ctx_.context(), "size_error", current_func);
    llvm::BasicBlock* success_block = llvm::BasicBlock::Create(ctx_.context(), "size_success", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "size_merge", current_func);

    ctx_.builder().CreateCondBr(is_null, error_block, success_block);

    // Error case: return #f
    ctx_.builder().SetInsertPoint(error_block);
    llvm::Value* error_val = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    ctx_.builder().CreateBr(merge_block);

    // Success case: seek to end, get position, close
    ctx_.builder().SetInsertPoint(success_block);
    // fseek(file, 0, SEEK_END) - SEEK_END = 2
    ctx_.builder().CreateCall(fseek_func, {
        file, llvm::ConstantInt::get(ctx_.int64Type(), 0), llvm::ConstantInt::get(ctx_.int32Type(), 2)
    });
    llvm::Value* size = ctx_.builder().CreateCall(ftell_func, {file});
    ctx_.builder().CreateCall(fclose_func, {file});
    llvm::Value* size_val = tagged_.packInt64(size, true);
    ctx_.builder().CreateBr(merge_block);

    // Merge
    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
    phi->addIncoming(error_val, error_block);
    phi->addIncoming(size_val, success_block);

    return phi;
}

llvm::Value* SystemCodegen::readFile(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("read-file requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* fopen_func = function_table_["fopen"];
    llvm::Function* fseek_func = function_table_["fseek"];
    llvm::Function* ftell_func = function_table_["ftell"];
    llvm::Function* fread_func = function_table_["fread"];
    llvm::Function* fclose_func = function_table_["fclose"];
    if (!fopen_func || !fseek_func || !ftell_func || !fread_func || !fclose_func) {
        return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    }

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!path_arg) return nullptr;

    llvm::Value* path_ptr = extractStringPtr(path_arg);

    // Open file
    llvm::GlobalVariable* mode_str = ctx_.internString("rb");
    llvm::Value* mode = ctx_.builder().CreatePointerCast(mode_str, ctx_.ptrType());
    llvm::Value* file = ctx_.builder().CreateCall(fopen_func, {path_ptr, mode});

    // Check if NULL
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(file,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* error_block = llvm::BasicBlock::Create(ctx_.context(), "readfile_error", current_func);
    llvm::BasicBlock* read_block = llvm::BasicBlock::Create(ctx_.context(), "readfile_read", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "readfile_merge", current_func);

    ctx_.builder().CreateCondBr(is_null, error_block, read_block);

    // Error case
    ctx_.builder().SetInsertPoint(error_block);
    llvm::Value* error_val = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    ctx_.builder().CreateBr(merge_block);

    // Read file
    ctx_.builder().SetInsertPoint(read_block);
    // Seek to end to get size
    ctx_.builder().CreateCall(fseek_func, {
        file, llvm::ConstantInt::get(ctx_.int64Type(), 0), llvm::ConstantInt::get(ctx_.int32Type(), 2)
    });
    llvm::Value* size = ctx_.builder().CreateCall(ftell_func, {file});
    // Seek back to beginning
    ctx_.builder().CreateCall(fseek_func, {
        file, llvm::ConstantInt::get(ctx_.int64Type(), 0), llvm::ConstantInt::get(ctx_.int32Type(), 0)
    });

    // Allocate buffer with HEAP_SUBTYPE_STRING header so `string?` /
    // `string-length` / `substring` recognise the result.  Previously this
    // used raw `arena_allocate`, leaving the bytes at offset -8 as
    // arbitrary memory; the (string? (read-file …)) check then returned
    // #f because the subtype byte was garbage, and the agent project
    // worked around it via `(run-argv-capture (cat path))`.
    // arena_allocate_string_with_header(arena, size) reserves size+1
    // bytes for the trailing NUL and stamps header.size = size+1.
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    llvm::Value* buf = ctx_.builder().CreateCall(
        mem_.getArenaAllocateStringWithHeader(), {arena_ptr, size});

    // Read entire file
    ctx_.builder().CreateCall(fread_func, {
        buf, llvm::ConstantInt::get(ctx_.int64Type(), 1), size, file
    });

    // Null terminate at offset `size` (the allocator reserved +1 for it).
    llvm::Value* term_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), buf, size);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int8Type(), 0), term_ptr);

    // Close file
    ctx_.builder().CreateCall(fclose_func, {file});

    // Return string
    llvm::Value* success_val = tagged_.packPtr(buf, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(merge_block);

    // Merge
    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
    phi->addIncoming(error_val, error_block);
    phi->addIncoming(success_val, read_block);

    return phi;
}

llvm::Value* SystemCodegen::writeFile(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) {
        eshkol_warn("write-file requires exactly 2 arguments");
        return nullptr;
    }

    llvm::Function* fopen_func = function_table_["fopen"];
    llvm::Function* fwrite_func = function_table_["fwrite"];
    llvm::Function* fclose_func = function_table_["fclose"];
    llvm::Function* strlen_func = function_table_["strlen"];
    if (!fopen_func || !fwrite_func || !fclose_func || !strlen_func) {
        return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    }

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    llvm::Value* content_arg = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!path_arg || !content_arg) return nullptr;

    llvm::Value* path_ptr = extractStringPtr(path_arg);
    llvm::Value* content_ptr = extractStringPtr(content_arg);

    // Open file for writing
    llvm::GlobalVariable* mode_str = ctx_.internString("wb");
    llvm::Value* mode = ctx_.builder().CreatePointerCast(mode_str, ctx_.ptrType());
    llvm::Value* file = ctx_.builder().CreateCall(fopen_func, {path_ptr, mode});

    // Check if NULL
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(file,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* error_block = llvm::BasicBlock::Create(ctx_.context(), "writefile_error", current_func);
    llvm::BasicBlock* write_block = llvm::BasicBlock::Create(ctx_.context(), "writefile_write", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "writefile_merge", current_func);

    ctx_.builder().CreateCondBr(is_null, error_block, write_block);

    // Error case
    ctx_.builder().SetInsertPoint(error_block);
    llvm::Value* false_val = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    ctx_.builder().CreateBr(merge_block);

    // Write case
    ctx_.builder().SetInsertPoint(write_block);
    llvm::Value* content_len = ctx_.builder().CreateCall(strlen_func, {content_ptr});
    ctx_.builder().CreateCall(fwrite_func, {
        content_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 1), content_len, file
    });
    ctx_.builder().CreateCall(fclose_func, {file});
    llvm::Value* true_val = tagged_.packBool(llvm::ConstantInt::getTrue(ctx_.context()));
    ctx_.builder().CreateBr(merge_block);

    // Merge
    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
    phi->addIncoming(false_val, error_block);
    phi->addIncoming(true_val, write_block);

    return phi;
}

llvm::Value* SystemCodegen::appendFile(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) {
        eshkol_warn("append-file requires exactly 2 arguments");
        return nullptr;
    }

    llvm::Function* fopen_func = function_table_["fopen"];
    llvm::Function* fwrite_func = function_table_["fwrite"];
    llvm::Function* fclose_func = function_table_["fclose"];
    llvm::Function* strlen_func = function_table_["strlen"];
    if (!fopen_func || !fwrite_func || !fclose_func || !strlen_func) {
        return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    }

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    llvm::Value* content_arg = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!path_arg || !content_arg) return nullptr;

    llvm::Value* path_ptr = extractStringPtr(path_arg);
    llvm::Value* content_ptr = extractStringPtr(content_arg);

    // Open file for appending
    llvm::GlobalVariable* mode_str = ctx_.internString("ab");
    llvm::Value* mode = ctx_.builder().CreatePointerCast(mode_str, ctx_.ptrType());
    llvm::Value* file = ctx_.builder().CreateCall(fopen_func, {path_ptr, mode});

    // Check if NULL
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(file,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* error_block = llvm::BasicBlock::Create(ctx_.context(), "appendfile_error", current_func);
    llvm::BasicBlock* write_block = llvm::BasicBlock::Create(ctx_.context(), "appendfile_write", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "appendfile_merge", current_func);

    ctx_.builder().CreateCondBr(is_null, error_block, write_block);

    // Error case
    ctx_.builder().SetInsertPoint(error_block);
    llvm::Value* false_val = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    ctx_.builder().CreateBr(merge_block);

    // Write case
    ctx_.builder().SetInsertPoint(write_block);
    llvm::Value* content_len = ctx_.builder().CreateCall(strlen_func, {content_ptr});
    ctx_.builder().CreateCall(fwrite_func, {
        content_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 1), content_len, file
    });
    ctx_.builder().CreateCall(fclose_func, {file});
    llvm::Value* true_val = tagged_.packBool(llvm::ConstantInt::getTrue(ctx_.context()));
    ctx_.builder().CreateBr(merge_block);

    // Merge
    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
    phi->addIncoming(false_val, error_block);
    phi->addIncoming(true_val, write_block);

    return phi;
}

// =========================================================================
// DIRECTORY OPERATIONS
// =========================================================================

llvm::Value* SystemCodegen::directoryExists(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("directory-exists? requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* opendir_func = function_table_["opendir"];
    llvm::Function* closedir_func = function_table_["closedir"];
    if (!opendir_func || !closedir_func) {
        return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    }

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!path_arg) return nullptr;

    llvm::Value* path_ptr = extractStringPtr(path_arg);

    // Try to open as directory
    llvm::Value* dir = ctx_.builder().CreateCall(opendir_func, {path_ptr});

    // Check if NULL
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(dir,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* not_dir_block = llvm::BasicBlock::Create(ctx_.context(), "not_dir", current_func);
    llvm::BasicBlock* is_dir_block = llvm::BasicBlock::Create(ctx_.context(), "is_dir", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "dir_merge", current_func);

    ctx_.builder().CreateCondBr(is_null, not_dir_block, is_dir_block);

    // Not a directory
    ctx_.builder().SetInsertPoint(not_dir_block);
    llvm::Value* false_val = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    ctx_.builder().CreateBr(merge_block);

    // Is a directory - close it first
    ctx_.builder().SetInsertPoint(is_dir_block);
    ctx_.builder().CreateCall(closedir_func, {dir});
    llvm::Value* true_val = tagged_.packBool(llvm::ConstantInt::getTrue(ctx_.context()));
    ctx_.builder().CreateBr(merge_block);

    // Merge
    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
    phi->addIncoming(false_val, not_dir_block);
    phi->addIncoming(true_val, is_dir_block);

    return phi;
}

llvm::Value* SystemCodegen::makeDirectory(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("make-directory requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* mkdir_func = function_table_["mkdir"];
    if (!mkdir_func) return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!path_arg) return nullptr;

    llvm::Value* path_ptr = extractStringPtr(path_arg);

    // Call mkdir(path, 0755)
    llvm::Value* result = ctx_.builder().CreateCall(mkdir_func, {
        path_ptr, llvm::ConstantInt::get(ctx_.int32Type(), 0755)
    });

    llvm::Value* success = ctx_.builder().CreateICmpEQ(result, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    return tagged_.packBool(success);
}

llvm::Value* SystemCodegen::deleteDirectory(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("delete-directory requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* rmdir_func = function_table_["rmdir"];
    if (!rmdir_func) return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!path_arg) return nullptr;

    llvm::Value* path_ptr = extractStringPtr(path_arg);

    // Call rmdir(path)
    llvm::Value* result = ctx_.builder().CreateCall(rmdir_func, {path_ptr});

    llvm::Value* success = ctx_.builder().CreateICmpEQ(result, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    return tagged_.packBool(success);
}

llvm::Value* SystemCodegen::directoryList(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("directory-list requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* opendir_func = function_table_["opendir"];
    llvm::Function* readdir_func = function_table_["readdir"];
    llvm::Function* closedir_func = function_table_["closedir"];
    llvm::Function* strlen_func = function_table_["strlen"];
    if (!opendir_func || !readdir_func || !closedir_func || !strlen_func) {
        return tagged_.packNull();
    }

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!path_arg) return nullptr;

    llvm::Value* path_ptr = extractStringPtr(path_arg);

    // Open directory
    llvm::Value* dir = ctx_.builder().CreateCall(opendir_func, {path_ptr});

    // Check if NULL
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(dir,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* error_block = llvm::BasicBlock::Create(ctx_.context(), "dirlist_error", current_func);
    llvm::BasicBlock* loop_block = llvm::BasicBlock::Create(ctx_.context(), "dirlist_loop", current_func);
    llvm::BasicBlock* check_block = llvm::BasicBlock::Create(ctx_.context(), "dirlist_check", current_func);
    llvm::BasicBlock* add_block = llvm::BasicBlock::Create(ctx_.context(), "dirlist_add", current_func);
    llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "dirlist_done", current_func);

    // Create allocas at function entry
    llvm::IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
    llvm::BasicBlock& entry = current_func->getEntryBlock();
    ctx_.builder().SetInsertPoint(&entry, entry.begin());
    llvm::Value* result_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "dirlist_result");
    ctx_.builder().restoreIP(saved_ip);

    // Initialize result to null (empty list)
    ctx_.builder().CreateStore(tagged_.packNull(), result_ptr);

    ctx_.builder().CreateCondBr(is_null, error_block, loop_block);

    // Error case
    ctx_.builder().SetInsertPoint(error_block);
    ctx_.builder().CreateBr(done_block);

    // Loop: read entries
    ctx_.builder().SetInsertPoint(loop_block);
    llvm::Value* entry_ptr = ctx_.builder().CreateCall(readdir_func, {dir});
    llvm::Value* entry_null = ctx_.builder().CreateICmpEQ(entry_ptr,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));
    ctx_.builder().CreateCondBr(entry_null, check_block, add_block);

    // Check and close
    ctx_.builder().SetInsertPoint(check_block);
    ctx_.builder().CreateCall(closedir_func, {dir});
    ctx_.builder().CreateBr(done_block);

    // Add entry to list
    ctx_.builder().SetInsertPoint(add_block);
    // Get d_name field - offset 21 on macOS
    llvm::Value* name_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), entry_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 21));

    // Copy the name to arena-allocated string
    llvm::Value* name_len = ctx_.builder().CreateCall(strlen_func, {name_ptr});
    llvm::Value* alloc_len = ctx_.builder().CreateAdd(name_len, llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    llvm::Value* new_str = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, alloc_len});

    llvm::Function* strcpy_func = ctx_.funcs().getStrcpy();
    ctx_.builder().CreateCall(strcpy_func, {new_str, name_ptr});

    // Pack string and cons onto list
    llvm::Value* str_tagged = tagged_.packPtr(new_str, ESHKOL_VALUE_HEAP_PTR);
    llvm::Value* current_list = ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_ptr);

    // Allocate cons cell with object header (consolidated pointer format)
    llvm::Value* cons = ctx_.builder().CreateCall(mem_.getArenaAllocateConsWithHeader(), {arena_ptr});
    // Set car to string (is_cdr = false for car, true for cdr)
    llvm::Value* is_car = llvm::ConstantInt::getFalse(ctx_.context());
    llvm::Value* is_cdr = llvm::ConstantInt::getTrue(ctx_.context());

    // Create alloca for tagged values
    llvm::Value* str_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
    llvm::Value* list_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
    ctx_.builder().CreateStore(str_tagged, str_ptr);
    ctx_.builder().CreateStore(current_list, list_ptr);

    ctx_.builder().CreateCall(mem_.getTaggedConsSetTaggedValue(), {cons, is_car, str_ptr});
    ctx_.builder().CreateCall(mem_.getTaggedConsSetTaggedValue(), {cons, is_cdr, list_ptr});

    // Update result (HEAP_PTR - consolidated pointer format)
    llvm::Value* new_list = tagged_.packHeapPtr(cons);
    ctx_.builder().CreateStore(new_list, result_ptr);

    ctx_.builder().CreateBr(loop_block);

    // Done
    ctx_.builder().SetInsertPoint(done_block);
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_ptr);
}

llvm::Value* SystemCodegen::currentDirectory(const eshkol_operations_t* op) {
    llvm::Function* getcwd_func = function_table_["getcwd"];
    if (!getcwd_func) return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));

    // Allocate buffer for path (PATH_MAX is typically 4096)
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    llvm::Value* buf = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {
        arena_ptr, llvm::ConstantInt::get(ctx_.sizeType(), 4096)
    });

    // Call getcwd(buf, 4096)
    llvm::Value* result = ctx_.builder().CreateCall(getcwd_func, {
        buf, llvm::ConstantInt::get(ctx_.sizeType(), 4096)
    });

    // Check if NULL (error)
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(result,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* error_block = llvm::BasicBlock::Create(ctx_.context(), "cwd_error", current_func);
    llvm::BasicBlock* success_block = llvm::BasicBlock::Create(ctx_.context(), "cwd_success", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "cwd_merge", current_func);

    ctx_.builder().CreateCondBr(is_null, error_block, success_block);

    ctx_.builder().SetInsertPoint(error_block);
    llvm::Value* error_val = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    ctx_.builder().CreateBr(merge_block);

    ctx_.builder().SetInsertPoint(success_block);
    llvm::Value* success_val = tagged_.packPtr(result, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(merge_block);

    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
    phi->addIncoming(error_val, error_block);
    phi->addIncoming(success_val, success_block);

    return phi;
}

llvm::Value* SystemCodegen::setCurrentDirectory(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) {
        eshkol_warn("set-current-directory! requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* chdir_func = function_table_["chdir"];
    if (!chdir_func) return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));

    if (!codegen_ast_callback_) return tagged_.packNull();

    llvm::Value* path_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!path_arg) return nullptr;

    llvm::Value* path_ptr = extractStringPtr(path_arg);

    // Call chdir(path)
    llvm::Value* result = ctx_.builder().CreateCall(chdir_func, {path_ptr});

    llvm::Value* success = ctx_.builder().CreateICmpEQ(result, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    return tagged_.packBool(success);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * v1.2 System Builtins — delegate to C runtime (system_builtins.c)
 *
 * These functions are declared extern "C" in system_builtins.c and return
 * eshkol_sysbuiltin_value_t which is layout-identical to eshkol_tagged_value_t.
 * We declare them as returning the LLVM tagged value type and call directly.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* All-pointer calling convention for C runtime builtins.
 * Both the result and all arguments are passed through pointers to avoid
 * ALL struct passing ABI mismatches between LLVM IR and C on ARM64/etc.
 * Signature: void c_func(tagged_value_t* out, tagged_value_t* arg1, ...) */
llvm::Function* getOrDeclareRuntimeFuncAllPtr(CodegenContext& ctx, llvm::Module* mod,
                                               const std::string& name, int nargs) {
    llvm::Function* f = mod->getFunction(name);
    if (f) return f;
    std::vector<llvm::Type*> arg_types(1 + nargs, ctx.ptrType());
    llvm::FunctionType* ft = llvm::FunctionType::get(
        llvm::Type::getVoidTy(ctx.context()), arg_types, false);
    f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, mod);
    return f;
}

/* Store tagged value to alloca, call C function with pointers, load result.
 *
 * Architectural note: the AST callback may return a raw LLVM value (double,
 * i64, i1, or pointer) rather than a tagged_value struct. Storing a raw
 * scalar directly into the 16-byte tagged_value slot would leave the upper
 * bytes uninitialized, which the sret callee then reads as garbage type/data
 * fields. ensureTagged normalises every argument into a full tagged_value
 * before the store, eliminating that whole class of ABI corruption. */
static llvm::Value* callPtrRuntime(CodegenContext& ctx, llvm::Function* f,
                                    llvm::ArrayRef<llvm::Value*> args) {
    llvm::IRBuilder<>& builder = ctx.builder();
    TaggedValueCodegen tagged(ctx);
    llvm::Value* result_ptr = builder.CreateAlloca(ctx.taggedValueType(), nullptr, "rt_out");
    std::vector<llvm::Value*> call_args;
    call_args.push_back(result_ptr);
    for (auto* a : args) {
        llvm::Value* tagged_arg = tagged.ensureTagged(a);
        llvm::Value* slot = builder.CreateAlloca(ctx.taggedValueType(), nullptr, "rt_arg");
        builder.CreateStore(tagged_arg, slot);
        call_args.push_back(slot);
    }
    builder.CreateCall(f, call_args);
    return builder.CreateLoad(ctx.taggedValueType(), result_ptr);
}

/* Zero-arg builtin */
#define ZERO_ARG_BUILTIN(method_name, c_func_name) \
llvm::Value* SystemCodegen::method_name(const eshkol_operations_t* op) { \
    (void)op; \
    llvm::Module* mod = ctx_.builder().GetInsertBlock()->getParent()->getParent(); \
    llvm::Function* f = getOrDeclareRuntimeFuncAllPtr(ctx_, mod, c_func_name, 0); \
    return callPtrRuntime(ctx_, f, {}); \
}

/* One-arg builtin */
#define ONE_ARG_BUILTIN(method_name, c_func_name) \
llvm::Value* SystemCodegen::method_name(const eshkol_operations_t* op) { \
    if (op->call_op.num_vars != 1) return tagged_.packNull(); \
    if (!codegen_ast_callback_) return tagged_.packNull(); \
    llvm::Value* arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_); \
    if (!arg) return tagged_.packNull(); \
    llvm::Module* mod = ctx_.builder().GetInsertBlock()->getParent()->getParent(); \
    llvm::Function* f = getOrDeclareRuntimeFuncAllPtr(ctx_, mod, c_func_name, 1); \
    return callPtrRuntime(ctx_, f, {arg}); \
}

/* Two-arg builtin */
#define TWO_ARG_BUILTIN(method_name, c_func_name) \
llvm::Value* SystemCodegen::method_name(const eshkol_operations_t* op) { \
    if (op->call_op.num_vars != 2) return tagged_.packNull(); \
    if (!codegen_ast_callback_) return tagged_.packNull(); \
    llvm::Value* a = codegen_ast_callback_(&op->call_op.variables[0], callback_context_); \
    llvm::Value* b = codegen_ast_callback_(&op->call_op.variables[1], callback_context_); \
    if (!a || !b) return tagged_.packNull(); \
    llvm::Module* mod = ctx_.builder().GetInsertBlock()->getParent()->getParent(); \
    llvm::Function* f = getOrDeclareRuntimeFuncAllPtr(ctx_, mod, c_func_name, 2); \
    return callPtrRuntime(ctx_, f, {a, b}); \
}

/* System info */
ZERO_ARG_BUILTIN(osType, "eshkol_builtin_os_type")
ZERO_ARG_BUILTIN(osArch, "eshkol_builtin_os_arch")
ZERO_ARG_BUILTIN(hostnameBuiltin, "eshkol_builtin_hostname")
ZERO_ARG_BUILTIN(usernameBuiltin, "eshkol_builtin_username")
ZERO_ARG_BUILTIN(cpuCount, "eshkol_builtin_cpu_count")
ZERO_ARG_BUILTIN(getpidBuiltin, "eshkol_builtin_getpid")
ZERO_ARG_BUILTIN(homeDirectory, "eshkol_builtin_home_directory")
/* Time API (#168) */
ZERO_ARG_BUILTIN(currentTimestamp, "eshkol_builtin_current_timestamp")
ONE_ARG_BUILTIN(formatIso8601, "eshkol_builtin_format_iso8601")
ONE_ARG_BUILTIN(parseIso8601, "eshkol_builtin_parse_iso8601")

/* System ops with args */
ONE_ARG_BUILTIN(sleepMs, "eshkol_builtin_sleep_ms")
ONE_ARG_BUILTIN(executableExists, "eshkol_builtin_executable_exists")

/* Path manipulation */
TWO_ARG_BUILTIN(pathJoin, "eshkol_builtin_path_join")
ONE_ARG_BUILTIN(pathDirname, "eshkol_builtin_path_dirname")
ONE_ARG_BUILTIN(pathBasename, "eshkol_builtin_path_basename")
ONE_ARG_BUILTIN(pathExtname, "eshkol_builtin_path_extname")
ONE_ARG_BUILTIN(pathIsAbsolute, "eshkol_builtin_path_is_absolute")
ONE_ARG_BUILTIN(pathNormalize, "eshkol_builtin_path_normalize")
ONE_ARG_BUILTIN(realpathBuiltin, "eshkol_builtin_realpath")

/* Filesystem */
ONE_ARG_BUILTIN(fileStat, "eshkol_builtin_file_stat")
TWO_ARG_BUILTIN(fileCopy, "eshkol_builtin_file_copy")
ONE_ARG_BUILTIN(mkdirRecursive, "eshkol_builtin_mkdir_recursive")
ONE_ARG_BUILTIN(mkdtempBuiltin, "eshkol_builtin_mkdtemp")
ONE_ARG_BUILTIN(directoryDeleteRecursive, "eshkol_builtin_directory_delete_recursive")

/* Shell */
ONE_ARG_BUILTIN(shellQuote, "eshkol_builtin_shell_quote")

/* Process */
TWO_ARG_BUILTIN(processSpawn, "eshkol_builtin_process_spawn")
ONE_ARG_BUILTIN(processWait, "eshkol_builtin_process_wait")

/* IO multiplexing */
TWO_ARG_BUILTIN(pollFd, "eshkol_builtin_poll_fd")

/* Tensor persistence */
TWO_ARG_BUILTIN(tensorSave, "eshkol_builtin_tensor_save")
ONE_ARG_BUILTIN(tensorLoad, "eshkol_builtin_tensor_load")

/* v1.2 batch 2: VM-parity + new builtins */
TWO_ARG_BUILTIN(fileChmod, "eshkol_builtin_file_chmod")
TWO_ARG_BUILTIN(symlinkCreate, "eshkol_builtin_symlink_create")
ONE_ARG_BUILTIN(symlinkRead, "eshkol_builtin_symlink_read")
ONE_ARG_BUILTIN(directoryWalk, "eshkol_builtin_directory_walk")
ONE_ARG_BUILTIN(mkstempBuiltin, "eshkol_builtin_mkstemp")
TWO_ARG_BUILTIN(processKill, "eshkol_builtin_process_kill")
ONE_ARG_BUILTIN(fileMtime, "eshkol_builtin_file_mtime")
ONE_ARG_BUILTIN(fileAtime, "eshkol_builtin_file_atime")
ONE_ARG_BUILTIN(fileLock, "eshkol_builtin_file_lock")
ONE_ARG_BUILTIN(fileUnlock, "eshkol_builtin_file_unlock")
TWO_ARG_BUILTIN(pathRelative, "eshkol_builtin_path_relative")
TWO_ARG_BUILTIN(pathResolve, "eshkol_builtin_path_resolve")
ONE_ARG_BUILTIN(globExpand, "eshkol_builtin_glob_expand")
TWO_ARG_BUILTIN(globMatch, "eshkol_builtin_glob_match")

/* v1.2 batch 3: advanced process management */
TWO_ARG_BUILTIN(processSetpgid, "eshkol_builtin_process_setpgid")
TWO_ARG_BUILTIN(processKillTree, "eshkol_builtin_process_kill_tree")
ONE_ARG_BUILTIN(processSpawnPty, "eshkol_builtin_process_spawn_pty")
TWO_ARG_BUILTIN(processReadNonblocking, "eshkol_builtin_process_read_nonblocking")

/* Three-arg builtin */
#define THREE_ARG_BUILTIN(method_name, c_func_name) \
llvm::Value* SystemCodegen::method_name(const eshkol_operations_t* op) { \
    if (op->call_op.num_vars != 3) return tagged_.packNull(); \
    if (!codegen_ast_callback_) return tagged_.packNull(); \
    llvm::Value* a = codegen_ast_callback_(&op->call_op.variables[0], callback_context_); \
    llvm::Value* b = codegen_ast_callback_(&op->call_op.variables[1], callback_context_); \
    llvm::Value* c = codegen_ast_callback_(&op->call_op.variables[2], callback_context_); \
    if (!a || !b || !c) return tagged_.packNull(); \
    llvm::Module* mod = ctx_.builder().GetInsertBlock()->getParent()->getParent(); \
    llvm::Function* f = getOrDeclareRuntimeFuncAllPtr(ctx_, mod, c_func_name, 3); \
    return callPtrRuntime(ctx_, f, {a, b, c}); \
}

/* Noesis requirements */
TWO_ARG_BUILTIN(fgMarginal, "eshkol_builtin_fg_marginal")
TWO_ARG_BUILTIN(fgEntropy, "eshkol_builtin_fg_entropy")
TWO_ARG_BUILTIN(kbRetract, "eshkol_builtin_kb_retract")

/* Consciousness engine — uses existing tagged functions from logic.cpp/inference.cpp/workspace.cpp.
 * These have signature: void func(arena_t* arena, const tv* arg1, ..., tv* result)
 * Arena comes first, result comes last. Custom dispatch needed. */

/* Helper: call arena-first tagged function with N args */
static llvm::Value* callArenaTaggedFunc(CodegenContext& ctx, TaggedValueCodegen& tagged,
                                         const std::string& name, int nargs,
                                         llvm::ArrayRef<llvm::Value*> args) {
    llvm::Module* mod = ctx.builder().GetInsertBlock()->getParent()->getParent();
    llvm::Function* f = mod->getFunction(name);
    if (!f) {
        /* Signature: void func(ptr arena, [ptr arg1, ..., ptr argN,] ptr result) */
        std::vector<llvm::Type*> param_types(2 + nargs, ctx.ptrType());
        llvm::FunctionType* ft = llvm::FunctionType::get(
            llvm::Type::getVoidTy(ctx.context()), param_types, false);
        f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, mod);
    }

    llvm::IRBuilder<>& builder = ctx.builder();
    llvm::Value* result_ptr = builder.CreateAlloca(ctx.taggedValueType(), nullptr, "ce_result");
    llvm::Value* arena = builder.CreateLoad(ctx.ptrType(), ctx.globalArena());

    std::vector<llvm::Value*> call_args;
    call_args.push_back(arena);
    for (auto* a : args) {
        llvm::Value* slot = builder.CreateAlloca(ctx.taggedValueType());
        builder.CreateStore(a, slot);
        call_args.push_back(slot);
    }
    call_args.push_back(result_ptr);

    builder.CreateCall(f, call_args);
    return builder.CreateLoad(ctx.taggedValueType(), result_ptr);
}

#define CE_ZERO_ARG(method_name, c_func_name) \
llvm::Value* SystemCodegen::method_name(const eshkol_operations_t* op) { \
    (void)op; \
    return callArenaTaggedFunc(ctx_, tagged_, c_func_name, 0, {}); \
}

#define CE_TWO_ARG(method_name, c_func_name) \
llvm::Value* SystemCodegen::method_name(const eshkol_operations_t* op) { \
    if (op->call_op.num_vars != 2) return tagged_.packNull(); \
    if (!codegen_ast_callback_) return tagged_.packNull(); \
    llvm::Value* a = codegen_ast_callback_(&op->call_op.variables[0], callback_context_); \
    llvm::Value* b = codegen_ast_callback_(&op->call_op.variables[1], callback_context_); \
    if (!a || !b) return tagged_.packNull(); \
    return callArenaTaggedFunc(ctx_, tagged_, c_func_name, 2, {a, b}); \
}

#define CE_THREE_ARG(method_name, c_func_name) \
llvm::Value* SystemCodegen::method_name(const eshkol_operations_t* op) { \
    if (op->call_op.num_vars != 3) return tagged_.packNull(); \
    if (!codegen_ast_callback_) return tagged_.packNull(); \
    llvm::Value* a = codegen_ast_callback_(&op->call_op.variables[0], callback_context_); \
    llvm::Value* b = codegen_ast_callback_(&op->call_op.variables[1], callback_context_); \
    llvm::Value* c = codegen_ast_callback_(&op->call_op.variables[2], callback_context_); \
    if (!a || !b || !c) return tagged_.packNull(); \
    return callArenaTaggedFunc(ctx_, tagged_, c_func_name, 3, {a, b, c}); \
}

#define CE_ONE_ARG(method_name, c_func_name) \
llvm::Value* SystemCodegen::method_name(const eshkol_operations_t* op) { \
    if (op->call_op.num_vars != 1) return tagged_.packNull(); \
    if (!codegen_ast_callback_) return tagged_.packNull(); \
    llvm::Value* a = codegen_ast_callback_(&op->call_op.variables[0], callback_context_); \
    if (!a) return tagged_.packNull(); \
    return callArenaTaggedFunc(ctx_, tagged_, c_func_name, 1, {a}); \
}

/* Logic engine */
CE_ZERO_ARG(makeSubstitution, "eshkol_make_substitution_tagged")
CE_ZERO_ARG(makeKbBuiltin, "eshkol_make_kb_tagged")
CE_THREE_ARG(unifyBuiltin, "eshkol_unify_tagged")
CE_TWO_ARG(walkBuiltin, "eshkol_walk_tagged")
CE_TWO_ARG(makeFactBuiltin, "eshkol_make_fact_tagged")
CE_TWO_ARG(kbAssertBuiltin, "eshkol_kb_assert_tagged")
CE_TWO_ARG(kbQueryBuiltin, "eshkol_kb_query_tagged")

/* Inference engine */
CE_TWO_ARG(makeFactorGraphBuiltin, "eshkol_make_factor_graph_tagged")
CE_THREE_ARG(fgAddFactorBuiltin, "eshkol_fg_add_factor_tagged")
CE_THREE_ARG(fgInferBuiltin, "eshkol_fg_infer_tagged")
CE_TWO_ARG(freeEnergyBuiltin, "eshkol_free_energy_tagged")
CE_THREE_ARG(expectedFreeEnergyBuiltin, "eshkol_efe_tagged")

/* Workspace */
CE_TWO_ARG(makeWorkspaceBuiltin, "eshkol_make_workspace_tagged")
CE_THREE_ARG(wsRegisterBuiltin, "eshkol_ws_register_tagged")
CE_ONE_ARG(wsStepBuiltin, "eshkol_ws_step_tagged")

#undef CE_ZERO_ARG
#undef CE_ONE_ARG
#undef CE_TWO_ARG
#undef CE_THREE_ARG

/* Reverse-mode AD tape — all use sret pattern from system_builtins */
ZERO_ARG_BUILTIN(adTapeNew, "eshkol_ad_tape_new_sret")
/* Bug I (2026-04-20): release the tape's owned sub-arena so iterative
 * fit loops don't leak. One-arg: the tape to release. */
ONE_ARG_BUILTIN(adTapeRelease, "eshkol_ad_tape_release_sret")
TWO_ARG_BUILTIN(adConst, "eshkol_ad_const_sret")
TWO_ARG_BUILTIN(adVar, "eshkol_ad_var_sret")

llvm::Value* SystemCodegen::adBinaryOp(const eshkol_operations_t* op, const char* func_name) {
    if (op->call_op.num_vars != 3) return tagged_.packNull();
    if (!codegen_ast_callback_) return tagged_.packNull();
    llvm::Value* tape = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    llvm::Value* left = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    llvm::Value* right = codegen_ast_callback_(&op->call_op.variables[2], callback_context_);
    if (!tape || !left || !right) return tagged_.packNull();
    llvm::Module* mod = ctx_.builder().GetInsertBlock()->getParent()->getParent();
    llvm::Function* f = getOrDeclareRuntimeFuncAllPtr(ctx_, mod, func_name, 3);
    return callPtrRuntime(ctx_, f, {tape, left, right});
}

llvm::Value* SystemCodegen::adUnaryOp(const eshkol_operations_t* op, const char* func_name) {
    if (op->call_op.num_vars != 2) return tagged_.packNull();
    if (!codegen_ast_callback_) return tagged_.packNull();
    llvm::Value* tape = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    llvm::Value* node = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!tape || !node) return tagged_.packNull();
    llvm::Module* mod = ctx_.builder().GetInsertBlock()->getParent()->getParent();
    llvm::Function* f = getOrDeclareRuntimeFuncAllPtr(ctx_, mod, func_name, 2);
    return callPtrRuntime(ctx_, f, {tape, node});
}

TWO_ARG_BUILTIN(adBackward, "eshkol_ad_backward_sret")
TWO_ARG_BUILTIN(adGradient, "eshkol_ad_gradient_sret")
TWO_ARG_BUILTIN(adNodeValue, "eshkol_ad_node_value_sret")
TWO_ARG_BUILTIN(onnxExportTensor, "eshkol_builtin_onnx_export_tensor")

/* Type predicates */
ONE_ARG_BUILTIN(logicVarPred, "eshkol_builtin_logic_var_p")
ONE_ARG_BUILTIN(substitutionPred, "eshkol_builtin_substitution_p")
ONE_ARG_BUILTIN(factPred, "eshkol_builtin_fact_p")
ONE_ARG_BUILTIN(kbPred, "eshkol_builtin_kb_p")
ONE_ARG_BUILTIN(factorGraphPred, "eshkol_builtin_factor_graph_p")
ONE_ARG_BUILTIN(workspacePred, "eshkol_builtin_workspace_p")
ONE_ARG_BUILTIN(tensorPred, "eshkol_builtin_tensor_p")
ONE_ARG_BUILTIN(dualPred, "eshkol_builtin_dual_p")
THREE_ARG_BUILTIN(fgUpdateCpt, "eshkol_builtin_fg_update_cpt")
ONE_ARG_BUILTIN(kbCount, "eshkol_builtin_kb_count")

/* Image I/O */
ONE_ARG_BUILTIN(imageRead, "eshkol_builtin_image_read_sret")
THREE_ARG_BUILTIN(imageWrite, "eshkol_builtin_image_write_sret")
ONE_ARG_BUILTIN(imageGrayscale, "eshkol_builtin_image_grayscale_sret")

/* v1.2 batch 4 */
ZERO_ARG_BUILTIN(processPid, "eshkol_builtin_process_pid")
ONE_ARG_BUILTIN(fileMmap, "eshkol_builtin_file_mmap")
ONE_ARG_BUILTIN(fileMunmap, "eshkol_builtin_file_munmap")
ONE_ARG_BUILTIN(unixSocketConnect, "eshkol_builtin_unix_socket_connect")
TWO_ARG_BUILTIN(socketSend, "eshkol_builtin_socket_send")
TWO_ARG_BUILTIN(socketRecv, "eshkol_builtin_socket_recv")
ONE_ARG_BUILTIN(socketClose, "eshkol_builtin_socket_close")
TWO_ARG_BUILTIN(termSetScrollRegion, "eshkol_builtin_term_set_scroll_region")
ZERO_ARG_BUILTIN(termResetScrollRegion, "eshkol_builtin_term_reset_scroll_region")
ZERO_ARG_BUILTIN(termEnableMouse, "eshkol_builtin_term_enable_mouse")
ZERO_ARG_BUILTIN(termDisableMouse, "eshkol_builtin_term_disable_mouse")
ONE_ARG_BUILTIN(termReadMouseEvent, "eshkol_builtin_term_read_mouse_event")
ZERO_ARG_BUILTIN(termEnableAlternateScreen, "eshkol_builtin_term_enable_alternate_screen")
ZERO_ARG_BUILTIN(termDisableAlternateScreen, "eshkol_builtin_term_disable_alternate_screen")
ONE_ARG_BUILTIN(termClipboardWrite, "eshkol_builtin_term_clipboard_write")
ZERO_ARG_BUILTIN(termClipboardRead, "eshkol_builtin_term_clipboard_read")
TWO_ARG_BUILTIN(termHyperlink, "eshkol_builtin_term_hyperlink")
ZERO_ARG_BUILTIN(termDetectCapabilities, "eshkol_builtin_term_detect_capabilities")
ZERO_ARG_BUILTIN(termBell, "eshkol_builtin_term_bell")
TWO_ARG_BUILTIN(fsWatchNative, "eshkol_builtin_fs_watch_native")
TWO_ARG_BUILTIN(fsWatchRecursive, "eshkol_builtin_fs_watch_recursive")
ONE_ARG_BUILTIN(fsWatchPoll, "eshkol_builtin_fs_watch_poll")
ONE_ARG_BUILTIN(fsUnwatch, "eshkol_builtin_fs_unwatch")
ONE_ARG_BUILTIN(ansiStrip, "eshkol_builtin_ansi_strip")
ONE_ARG_BUILTIN(stringDisplayWidth, "eshkol_builtin_string_display_width")
TWO_ARG_BUILTIN(kbSave, "eshkol_builtin_kb_save")
ONE_ARG_BUILTIN(kbLoad, "eshkol_builtin_kb_load")
ONE_ARG_BUILTIN(tensorTokenEstimate, "eshkol_builtin_tensor_token_estimate")

#undef ZERO_ARG_BUILTIN
#undef ONE_ARG_BUILTIN
#undef TWO_ARG_BUILTIN

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
