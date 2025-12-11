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

    // Valid case: return string
    ctx_.builder().SetInsertPoint(valid_block);
    llvm::Value* string_val = tagged_.packPtr(result, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(merge_block);

    // Merge
    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
    phi->addIncoming(false_val, null_block);
    phi->addIncoming(string_val, valid_block);

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
    llvm::Value* seconds = *reinterpret_cast<llvm::Value**>(seconds_tv_ptr);
    if (!seconds) return nullptr;

    // Convert seconds to microseconds (seconds * 1000000)
    if (seconds->getType()->isDoubleTy()) {
        seconds = ctx_.builder().CreateFPToSI(seconds, ctx_.int64Type());
    }
    llvm::Value* microseconds = ctx_.builder().CreateMul(seconds, llvm::ConstantInt::get(ctx_.int64Type(), 1000000));
    llvm::Value* usec = ctx_.builder().CreateTrunc(microseconds, ctx_.int32Type());

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
        code_i32 = ctx_.builder().CreateFPToSI(code, ctx_.int32Type());
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
    // Get global argc and argv
    llvm::Module* module = ctx_.builder().GetInsertBlock()->getParent()->getParent();

    llvm::GlobalVariable* g_argc = module->getGlobalVariable("__eshkol_argc");
    llvm::GlobalVariable* g_argv = module->getGlobalVariable("__eshkol_argv");

    if (!g_argc || !g_argv) {
        eshkol_warn("command-line: argc/argv globals not found");
        return tagged_.packNull();
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

    // Copy string to arena
    llvm::Value* arg_len = ctx_.builder().CreateCall(strlen_func, {arg_ptr});
    llvm::Value* alloc_len = ctx_.builder().CreateAdd(arg_len, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    llvm::Value* new_str = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, alloc_len});
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

    // Allocate buffer (+1 for null terminator)
    llvm::Value* alloc_size = ctx_.builder().CreateAdd(size, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    llvm::Value* buf = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, alloc_size});

    // Read entire file
    ctx_.builder().CreateCall(fread_func, {
        buf, llvm::ConstantInt::get(ctx_.int64Type(), 1), size, file
    });

    // Null terminate
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
    // Set car to string (is_cdr = 0 for car)
    llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int32Type(), 0);
    llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int32Type(), 1);

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
        arena_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 4096)
    });

    // Call getcwd(buf, 4096)
    llvm::Value* result = ctx_.builder().CreateCall(getcwd_func, {
        buf, llvm::ConstantInt::get(ctx_.int64Type(), 4096)
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

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
