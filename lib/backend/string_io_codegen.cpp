/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * StringIOCodegen implementation
 *
 * Note: The complex string and I/O implementations remain in llvm_codegen.cpp
 * for now due to dependencies on AST codegen methods. This module provides
 * the basic string creation and printf utilities which are self-contained.
 */

#include <eshkol/backend/string_io_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Function.h>

namespace eshkol {

StringIOCodegen::StringIOCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged)
    : ctx_(ctx)
    , tagged_(tagged)
    , printf_func_(nullptr) {
    eshkol_debug("StringIOCodegen initialized");
}

llvm::Value* StringIOCodegen::ensureRawInt64(llvm::Value* val, const std::string& name) {
    if (!val) return nullptr;

    // If already raw i64, return as-is
    if (val->getType()->isIntegerTy(64)) {
        return val;
    }

    // If it's a tagged_value struct, extract the int64 data field
    if (val->getType() == ctx_.taggedValueType()) {
        return tagged_.unpackInt64(val);
    }

    // For other integer types, extend to i64
    if (val->getType()->isIntegerTy()) {
        return ctx_.builder().CreateSExt(val, ctx_.int64Type(), name);
    }

    // Fallback: try to extract from tagged value anyway
    eshkol_warn("ensureRawInt64: unexpected type, attempting unpack");
    return tagged_.unpackInt64(val);
}

llvm::Value* StringIOCodegen::createString(const char* str) {
    if (!str) return nullptr;

    // Use context's string interning
    llvm::GlobalVariable* global_str = ctx_.internString(str);

    // Cast to generic pointer for use with printf, etc.
    return ctx_.builder().CreatePointerCast(
        global_str,
        llvm::PointerType::getUnqual(ctx_.context())
    );
}

llvm::Value* StringIOCodegen::packString(const char* str) {
    llvm::Value* str_ptr = createString(str);
    if (!str_ptr) return tagged_.packNull();
    return tagged_.packPtr(str_ptr, ESHKOL_VALUE_STRING_PTR);
}

llvm::Function* StringIOCodegen::getPrintf() {
    if (printf_func_) return printf_func_;

    // Declare printf if not already declared
    printf_func_ = ctx_.module().getFunction("printf");
    if (!printf_func_) {
        llvm::FunctionType* printf_type = llvm::FunctionType::get(
            ctx_.int32Type(),
            {ctx_.ptrType()},
            true  // variadic
        );
        printf_func_ = llvm::Function::Create(
            printf_type,
            llvm::Function::ExternalLinkage,
            "printf",
            &ctx_.module()
        );
    }
    return printf_func_;
}

llvm::Value* StringIOCodegen::newline(const eshkol_operations_t* op) {
    // Simple newline - just print \n
    llvm::Value* newline_str = createString("\n");
    ctx_.builder().CreateCall(getPrintf(), {newline_str});
    return tagged_.packNull();
}

// Note: The following implementations are complex and depend on:
// - codegenAST, codegenTypedAST (AST code generation)
// - Memory allocation for string manipulation
// - Runtime library functions
//
// These implementations remain in llvm_codegen.cpp until those modules are extracted.
// The functions below provide stubs that warn when called.

llvm::Value* StringIOCodegen::stringLength(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("StringIOCodegen::stringLength - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("string-length requires exactly 1 argument");
        return nullptr;
    }

    // Generate code for argument
    llvm::Value* arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!arg) return nullptr;

    // Extract string pointer from tagged value
    llvm::Value* ptr_int = tagged_.unpackInt64(arg);
    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());

    // Get or declare strlen
    llvm::Function* strlen_func = ctx_.module().getFunction("strlen");
    if (!strlen_func) {
        llvm::FunctionType* strlen_type = llvm::FunctionType::get(
            ctx_.int64Type(), {ctx_.ptrType()}, false);
        strlen_func = llvm::Function::Create(
            strlen_type, llvm::Function::ExternalLinkage, "strlen", &ctx_.module());
    }

    // Call strlen and pack result
    llvm::Value* len = ctx_.builder().CreateCall(strlen_func, {str_ptr});
    return tagged_.packInt64(len, true);
}

llvm::Value* StringIOCodegen::stringRef(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_ || !codegen_typed_ast_callback_) {
        eshkol_warn("StringIOCodegen::stringRef - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 2) {
        eshkol_warn("string-ref requires exactly 2 arguments");
        return nullptr;
    }

    // Get string argument
    llvm::Value* str_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!str_arg) return nullptr;

    // Get index via typed AST
    void* idx_tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!idx_tv_ptr) return nullptr;

    // Extract the LLVM value from TypedValue - first field is llvm_value
    llvm::Value* idx_raw = *reinterpret_cast<llvm::Value**>(idx_tv_ptr);
    if (!idx_raw) return nullptr;

    // CRITICAL: Ensure index is raw i64, not tagged_value struct
    // GEP indices must be integers - if we have a tagged_value from a variable,
    // we need to extract the actual integer value
    llvm::Value* idx = ensureRawInt64(idx_raw, "string_ref_idx");
    if (!idx) return nullptr;

    // Extract string pointer
    llvm::Value* ptr_int = tagged_.unpackInt64(str_arg);
    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());

    // Get character at index
    llvm::Value* char_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), str_ptr, idx);
    llvm::Value* char_val = ctx_.builder().CreateLoad(ctx_.int8Type(), char_ptr);

    // HoTT TYPE SYSTEM: string-ref returns CHAR type, not INT64
    return tagged_.packChar(char_val);
}

llvm::Value* StringIOCodegen::stringAppend(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("StringIOCodegen::stringAppend - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars < 1) {
        eshkol_warn("string-append requires at least 1 argument");
        return nullptr;
    }

    // Get C library functions from FunctionCache
    llvm::Function* strlen_func = ctx_.funcs().getStrlen();
    llvm::Function* strcpy_func = ctx_.funcs().getStrcpy();
    llvm::Function* strcat_func = ctx_.funcs().getStrcat();

    // Calculate total length needed
    std::vector<llvm::Value*> str_ptrs;
    llvm::Value* total_len = llvm::ConstantInt::get(ctx_.int64Type(), 1); // +1 for null terminator

    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
        llvm::Value* arg = codegen_ast_callback_(&op->call_op.variables[i], callback_context_);
        if (!arg) return nullptr;

        // Extract string pointer from tagged value
        llvm::Value* ptr_int = tagged_.unpackInt64(arg);
        llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());
        str_ptrs.push_back(str_ptr);

        // Add length to total
        llvm::Value* len = ctx_.builder().CreateCall(strlen_func, {str_ptr});
        total_len = ctx_.builder().CreateAdd(total_len, len);
    }

    // Allocate new string using arena_allocate
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    llvm::Value* new_str = ctx_.builder().CreateCall(
        ctx_.memory().getArenaAllocate(), {arena_ptr, total_len});

    // Copy first string
    ctx_.builder().CreateCall(strcpy_func, {new_str, str_ptrs[0]});

    // Concatenate remaining strings
    for (size_t i = 1; i < str_ptrs.size(); i++) {
        ctx_.builder().CreateCall(strcat_func, {new_str, str_ptrs[i]});
    }

    return tagged_.packPtr(new_str, ESHKOL_VALUE_STRING_PTR);
}

llvm::Value* StringIOCodegen::substring(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_ || !codegen_typed_ast_callback_) {
        eshkol_warn("StringIOCodegen::substring - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 3) {
        eshkol_warn("substring requires exactly 3 arguments");
        return nullptr;
    }

    // Get string argument
    llvm::Value* str_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!str_arg) return nullptr;

    // Get start and end indices via typed AST
    void* start_tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[1], callback_context_);
    void* end_tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[2], callback_context_);
    if (!start_tv_ptr || !end_tv_ptr) return nullptr;

    // Extract LLVM values from TypedValue
    llvm::Value* start_raw = *reinterpret_cast<llvm::Value**>(start_tv_ptr);
    llvm::Value* end_raw = *reinterpret_cast<llvm::Value**>(end_tv_ptr);
    if (!start_raw || !end_raw) return nullptr;

    // CRITICAL: Ensure indices are raw i64, not tagged_value struct
    // All operations below require raw integers (GEP, sub, memcpy size)
    llvm::Value* start = ensureRawInt64(start_raw, "substring_start");
    llvm::Value* end = ensureRawInt64(end_raw, "substring_end");
    if (!start || !end) return nullptr;

    // Extract string pointer from tagged value
    llvm::Value* ptr_int = tagged_.unpackInt64(str_arg);
    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());

    // Calculate length
    llvm::Value* len = ctx_.builder().CreateSub(end, start);
    llvm::Value* alloc_len = ctx_.builder().CreateAdd(len, llvm::ConstantInt::get(ctx_.int64Type(), 1));

    // Allocate new string
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    llvm::Value* new_str = ctx_.builder().CreateCall(
        ctx_.memory().getArenaAllocate(), {arena_ptr, alloc_len});

    // Get memcpy function
    llvm::Function* memcpy_func = ctx_.funcs().getMemcpy();

    // Copy substring
    llvm::Value* src_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), str_ptr, start);
    ctx_.builder().CreateCall(memcpy_func, {new_str, src_ptr, len});

    // Null terminate
    llvm::Value* null_pos = ctx_.builder().CreateGEP(ctx_.int8Type(), new_str, len);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int8Type(), 0), null_pos);

    return tagged_.packPtr(new_str, ESHKOL_VALUE_STRING_PTR);
}

llvm::Value* StringIOCodegen::stringCompare(const eshkol_operations_t* op, const std::string& cmp_type) {
    if (!codegen_ast_callback_) {
        eshkol_warn("StringIOCodegen::stringCompare - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 2) {
        eshkol_warn("String comparison requires exactly 2 arguments");
        return nullptr;
    }

    // Get string arguments
    llvm::Value* str1_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    llvm::Value* str2_arg = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!str1_arg || !str2_arg) return nullptr;

    // Extract string pointers from tagged values
    llvm::Value* ptr1_int = tagged_.unpackInt64(str1_arg);
    llvm::Value* str1_ptr = ctx_.builder().CreateIntToPtr(ptr1_int, ctx_.ptrType());
    llvm::Value* ptr2_int = tagged_.unpackInt64(str2_arg);
    llvm::Value* str2_ptr = ctx_.builder().CreateIntToPtr(ptr2_int, ctx_.ptrType());

    // Get strcmp function
    llvm::Function* strcmp_func = ctx_.funcs().getStrcmp();

    // Call strcmp
    llvm::Value* cmp_result = ctx_.builder().CreateCall(strcmp_func, {str1_ptr, str2_ptr});
    llvm::Value* zero = llvm::ConstantInt::get(ctx_.int32Type(), 0);
    llvm::Value* result;

    if (cmp_type == "eq") {
        result = ctx_.builder().CreateICmpEQ(cmp_result, zero);
    } else if (cmp_type == "lt") {
        result = ctx_.builder().CreateICmpSLT(cmp_result, zero);
    } else if (cmp_type == "gt") {
        result = ctx_.builder().CreateICmpSGT(cmp_result, zero);
    } else if (cmp_type == "le") {
        result = ctx_.builder().CreateICmpSLE(cmp_result, zero);
    } else if (cmp_type == "ge") {
        result = ctx_.builder().CreateICmpSGE(cmp_result, zero);
    } else {
        result = llvm::ConstantInt::getFalse(ctx_.context());
    }

    return tagged_.packBool(result);
}

llvm::Value* StringIOCodegen::stringToNumber(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("StringIOCodegen::stringToNumber - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("string->number requires exactly 1 argument");
        return nullptr;
    }

    // Get string argument
    llvm::Value* str_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!str_arg) return nullptr;

    // Extract string pointer from tagged value
    llvm::Value* ptr_int = tagged_.unpackInt64(str_arg);
    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());

    // Get strtod function (handles both integers and floats)
    llvm::Function* strtod_func = ctx_.funcs().getStrtod();

    // Call strtod with null end pointer
    llvm::Value* null_ptr = llvm::ConstantPointerNull::get(ctx_.ptrType());
    llvm::Value* result = ctx_.builder().CreateCall(strtod_func, {str_ptr, null_ptr});

    return tagged_.packDouble(result);
}

// Forward declaration of TypedValue layout for callback access
// Must match the actual TypedValue struct in llvm_codegen.cpp
namespace {
    struct TypedValueLayout {
        llvm::Value* llvm_value;
        eshkol_value_type_t type;
        bool is_exact;
        uint8_t flags;
    };
}

llvm::Value* StringIOCodegen::numberToString(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_) {
        eshkol_warn("StringIOCodegen::numberToString - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("number->string requires exactly 1 argument");
        return nullptr;
    }

    // Get typed value via callback
    void* tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!tv_ptr) return nullptr;

    // Access TypedValue fields
    auto* tv = reinterpret_cast<TypedValueLayout*>(tv_ptr);
    if (!tv->llvm_value) return nullptr;

    // Allocate buffer for string (64 bytes should be enough for any number)
    llvm::Value* buf_size = llvm::ConstantInt::get(ctx_.int64Type(), 64);
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    llvm::Value* buf = ctx_.builder().CreateCall(
        ctx_.memory().getArenaAllocate(), {arena_ptr, buf_size});

    // Get snprintf function
    llvm::Function* snprintf_func = ctx_.funcs().getSnprintf();

    llvm::Value* raw_val = tv->llvm_value;

    // For tagged values, we need runtime type checking since the value
    // might come from a hash table or other dynamic source
    if (raw_val->getType() == ctx_.taggedValueType()) {
        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

        // Get runtime type from tagged value
        // Use getBaseType() to properly handle legacy types (>=32)
        // DO NOT use 0x0F mask - 34 & 0x0F = 2 (DOUBLE) which is WRONG!
        llvm::Value* runtime_type = tagged_.getType(raw_val);
        llvm::Value* base_type = tagged_.getBaseType(runtime_type);
        llvm::Value* is_runtime_double = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));

        // Extract raw i64 data
        llvm::Value* data_i64 = tagged_.unpackInt64(raw_val);

        // Create blocks for double vs integer formatting
        llvm::BasicBlock* double_block = llvm::BasicBlock::Create(ctx_.context(), "n2s_double", current_func);
        llvm::BasicBlock* int_block = llvm::BasicBlock::Create(ctx_.context(), "n2s_int", current_func);
        llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "n2s_merge", current_func);

        ctx_.builder().CreateCondBr(is_runtime_double, double_block, int_block);

        // Format as double
        ctx_.builder().SetInsertPoint(double_block);
        llvm::Value* double_val = ctx_.builder().CreateBitCast(data_i64, ctx_.doubleType());
        llvm::Value* fmt_double = createString("%g");
        ctx_.builder().CreateCall(snprintf_func, {buf, buf_size, fmt_double, double_val});
        ctx_.builder().CreateBr(merge_block);

        // Format as integer
        ctx_.builder().SetInsertPoint(int_block);
        llvm::Value* fmt_int = createString("%lld");
        ctx_.builder().CreateCall(snprintf_func, {buf, buf_size, fmt_int, data_i64});
        ctx_.builder().CreateBr(merge_block);

        ctx_.builder().SetInsertPoint(merge_block);
    } else {
        // Non-tagged value: use static type info
        if (tv->type == ESHKOL_VALUE_DOUBLE || !tv->is_exact) {
            // Format as double
            llvm::Value* double_val = raw_val;
            if (raw_val->getType()->isIntegerTy(64)) {
                double_val = ctx_.builder().CreateBitCast(raw_val, ctx_.doubleType());
            } else if (!raw_val->getType()->isDoubleTy()) {
                double_val = ctx_.builder().CreateSIToFP(raw_val, ctx_.doubleType());
            }
            llvm::Value* fmt = createString("%g");
            ctx_.builder().CreateCall(snprintf_func, {buf, buf_size, fmt, double_val});
        } else {
            // Format as integer
            llvm::Value* int_val = raw_val;
            if (raw_val->getType()->isDoubleTy()) {
                int_val = ctx_.builder().CreateFPToSI(raw_val, ctx_.int64Type());
            } else if (!raw_val->getType()->isIntegerTy(64)) {
                int_val = ctx_.builder().CreateSExt(raw_val, ctx_.int64Type());
            }
            llvm::Value* fmt = createString("%lld");
            ctx_.builder().CreateCall(snprintf_func, {buf, buf_size, fmt, int_val});
        }
    }

    return tagged_.packPtr(buf, ESHKOL_VALUE_STRING_PTR);
}

llvm::Value* StringIOCodegen::makeString(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !codegen_ast_callback_) {
        eshkol_warn("StringIOCodegen::makeString - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_warn("make-string requires 1 or 2 arguments");
        return nullptr;
    }

    // Get length via typed AST
    void* len_tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!len_tv_ptr) return nullptr;

    llvm::Value* len = *reinterpret_cast<llvm::Value**>(len_tv_ptr);
    if (!len) return nullptr;

    // Ensure length is i64
    if (!len->getType()->isIntegerTy(64)) {
        len = ctx_.builder().CreateZExt(len, ctx_.int64Type());
    }

    // Get the fill character (default to space, ASCII 32)
    llvm::Value* fill_char;
    if (op->call_op.num_vars == 2) {
        llvm::Value* char_arg = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
        if (!char_arg) return nullptr;
        fill_char = tagged_.unpackInt64(char_arg);
        fill_char = ctx_.builder().CreateTrunc(fill_char, ctx_.int8Type());
    } else {
        fill_char = llvm::ConstantInt::get(ctx_.int8Type(), ' ');
    }

    // Allocate buffer: len + 1 for null terminator
    llvm::Value* buf_size = ctx_.builder().CreateAdd(len, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Function* malloc_func = ctx_.funcs().getMalloc();
    llvm::Value* buf = ctx_.builder().CreateCall(malloc_func, {buf_size});

    // Fill with the character using memset
    llvm::Function* memset_func = ctx_.funcs().getMemset();
    llvm::Value* fill_char_i32 = ctx_.builder().CreateZExt(fill_char, ctx_.int32Type());
    ctx_.builder().CreateCall(memset_func, {buf, fill_char_i32, len});

    // Add null terminator
    llvm::Value* term_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), buf, len);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int8Type(), 0), term_ptr);

    return tagged_.packPtr(buf, ESHKOL_VALUE_STRING_PTR);
}

llvm::Value* StringIOCodegen::stringSet(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_ || !codegen_typed_ast_callback_) {
        eshkol_warn("StringIOCodegen::stringSet - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 3) {
        eshkol_warn("string-set! requires exactly 3 arguments");
        return nullptr;
    }

    // Get string argument
    llvm::Value* str_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!str_arg) return nullptr;

    // Get index via typed AST
    void* idx_tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!idx_tv_ptr) return nullptr;

    llvm::Value* idx_raw = *reinterpret_cast<llvm::Value**>(idx_tv_ptr);
    if (!idx_raw) return nullptr;

    // CRITICAL: Ensure index is raw i64, not tagged_value struct
    llvm::Value* idx = ensureRawInt64(idx_raw, "string_set_idx");
    if (!idx) return nullptr;

    // Get character argument
    llvm::Value* char_arg = codegen_ast_callback_(&op->call_op.variables[2], callback_context_);
    if (!char_arg) return nullptr;

    // Extract string pointer from tagged value
    llvm::Value* ptr_int = tagged_.unpackInt64(str_arg);
    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());

    // Get character value
    llvm::Value* char_val = tagged_.unpackInt64(char_arg);
    char_val = ctx_.builder().CreateTrunc(char_val, ctx_.int8Type());

    // Store character at index
    llvm::Value* char_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), str_ptr, idx);
    ctx_.builder().CreateStore(char_val, char_ptr);

    // Return the string (unspecified in Scheme)
    return str_arg;
}

llvm::Value* StringIOCodegen::stringSplit(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_ || !cons_create_callback_) {
        eshkol_warn("StringIOCodegen::stringSplit - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 2) {
        eshkol_warn("string-split requires exactly 2 arguments (string delimiter)");
        return nullptr;
    }

    llvm::Value* str_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    llvm::Value* delim_arg = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!str_arg || !delim_arg) return nullptr;

    llvm::Function* parent_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::Function* strlen_func = ctx_.funcs().getStrlen();
    llvm::Function* malloc_func = ctx_.funcs().getMalloc();
    llvm::Function* memcpy_func = ctx_.funcs().getMemcpy();

    // Extract string pointer
    llvm::Value* str_ptr_int = tagged_.unpackInt64(str_arg);
    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(str_ptr_int, ctx_.ptrType());

    // Handle delimiter: could be string or char
    // Check if delimiter is a char (type == ESHKOL_VALUE_CHAR)
    llvm::Value* delim_type = tagged_.getType(delim_arg);
    llvm::Value* is_char = ctx_.builder().CreateICmpEQ(delim_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CHAR));

    // Create blocks for char vs string delimiter handling
    llvm::BasicBlock* char_delim_block = llvm::BasicBlock::Create(ctx_.context(), "split_char_delim", parent_func);
    llvm::BasicBlock* str_delim_block = llvm::BasicBlock::Create(ctx_.context(), "split_str_delim", parent_func);
    llvm::BasicBlock* delim_ready_block = llvm::BasicBlock::Create(ctx_.context(), "split_delim_ready", parent_func);

    // Create alloca for delimiter pointer at function entry
    llvm::IRBuilderBase::InsertPoint saved_ip_delim = ctx_.builder().saveIP();
    llvm::BasicBlock& entry_block = parent_func->getEntryBlock();
    ctx_.builder().SetInsertPoint(&entry_block, entry_block.begin());
    llvm::Value* delim_ptr_storage = ctx_.builder().CreateAlloca(ctx_.ptrType(), nullptr, "delim_ptr_storage");
    ctx_.builder().restoreIP(saved_ip_delim);

    ctx_.builder().CreateCondBr(is_char, char_delim_block, str_delim_block);

    // Handle char delimiter: convert to single-char string
    ctx_.builder().SetInsertPoint(char_delim_block);
    llvm::Value* char_val = tagged_.unpackInt64(delim_arg);
    // Allocate 2 bytes for single char + null terminator
    llvm::Value* char_buf = ctx_.builder().CreateCall(malloc_func,
        {llvm::ConstantInt::get(ctx_.int64Type(), 2)});
    // Store char and null terminator
    llvm::Value* char_i8 = ctx_.builder().CreateTrunc(char_val, ctx_.int8Type());
    ctx_.builder().CreateStore(char_i8, char_buf);
    llvm::Value* term_char = ctx_.builder().CreateGEP(ctx_.int8Type(), char_buf,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int8Type(), 0), term_char);
    ctx_.builder().CreateStore(char_buf, delim_ptr_storage);
    ctx_.builder().CreateBr(delim_ready_block);

    // Handle string delimiter: use directly
    ctx_.builder().SetInsertPoint(str_delim_block);
    llvm::Value* delim_ptr_int = tagged_.unpackInt64(delim_arg);
    llvm::Value* str_delim_ptr = ctx_.builder().CreateIntToPtr(delim_ptr_int, ctx_.ptrType());
    ctx_.builder().CreateStore(str_delim_ptr, delim_ptr_storage);
    ctx_.builder().CreateBr(delim_ready_block);

    // Continue with unified delimiter handling
    ctx_.builder().SetInsertPoint(delim_ready_block);
    llvm::Value* delim_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), delim_ptr_storage);

    // Get delimiter length and string length
    llvm::Value* delim_len = ctx_.builder().CreateCall(strlen_func, {delim_ptr});
    llvm::Value* str_len = ctx_.builder().CreateCall(strlen_func, {str_ptr});

    // Create basic blocks
    llvm::BasicBlock* search_cond = llvm::BasicBlock::Create(ctx_.context(), "split_search", parent_func);
    llvm::BasicBlock* search_body = llvm::BasicBlock::Create(ctx_.context(), "split_body", parent_func);
    llvm::BasicBlock* found_delim = llvm::BasicBlock::Create(ctx_.context(), "split_found", parent_func);
    llvm::BasicBlock* not_found = llvm::BasicBlock::Create(ctx_.context(), "split_not_found", parent_func);
    llvm::BasicBlock* add_segment = llvm::BasicBlock::Create(ctx_.context(), "split_add", parent_func);
    llvm::BasicBlock* loop_end = llvm::BasicBlock::Create(ctx_.context(), "split_end", parent_func);

    // Create allocas at function entry
    llvm::IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
    llvm::BasicBlock& entry = parent_func->getEntryBlock();
    ctx_.builder().SetInsertPoint(&entry, entry.begin());
    llvm::Value* result_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "split_result");
    llvm::Value* pos_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "split_pos");
    llvm::Value* segment_start_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "split_start");
    ctx_.builder().restoreIP(saved_ip);

    // Initialize: result = '(), pos = 0, segment_start = 0
    llvm::Value* null_val = tagged_.packNull();
    ctx_.builder().CreateStore(null_val, result_ptr);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), pos_ptr);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), segment_start_ptr);
    ctx_.builder().CreateBr(search_cond);

    // Search condition: pos <= str_len
    ctx_.builder().SetInsertPoint(search_cond);
    llvm::Value* curr_pos = ctx_.builder().CreateLoad(ctx_.int64Type(), pos_ptr);
    llvm::Value* still_searching = ctx_.builder().CreateICmpULE(curr_pos, str_len);
    ctx_.builder().CreateCondBr(still_searching, search_body, loop_end);

    // Search body: check if delimiter starts at current position
    ctx_.builder().SetInsertPoint(search_body);
    // Check if we're at end of string
    llvm::Value* at_end = ctx_.builder().CreateICmpEQ(curr_pos, str_len);

    // Create intermediate block for checking if we should try matching
    llvm::BasicBlock* check_match = llvm::BasicBlock::Create(ctx_.context(), "split_check_match", parent_func);

    // If at end, add the final segment; otherwise check if we can match
    ctx_.builder().CreateCondBr(at_end, add_segment, check_match);

    // Check if we can match delimiter at current position
    ctx_.builder().SetInsertPoint(check_match);
    llvm::Value* remaining = ctx_.builder().CreateSub(str_len, curr_pos);
    llvm::Value* can_match = ctx_.builder().CreateICmpUGE(remaining, delim_len);

    // Check if delimiter length is 0 (edge case)
    llvm::Value* delim_nonzero = ctx_.builder().CreateICmpUGT(delim_len,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* should_check = ctx_.builder().CreateAnd(can_match, delim_nonzero);

    // If we can match, try to match; otherwise just advance
    ctx_.builder().CreateCondBr(should_check, found_delim, not_found);

    // Found possible delimiter location, compare using strncmp
    ctx_.builder().SetInsertPoint(found_delim);
    llvm::Value* curr_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), str_ptr, curr_pos);

    // Use strncmp to compare delimiter with current position
    llvm::Function* strncmp_func = ctx_.funcs().getStrncmp();
    llvm::Value* cmp_result = ctx_.builder().CreateCall(strncmp_func,
        {curr_ptr, delim_ptr, delim_len});
    llvm::Value* is_match = ctx_.builder().CreateICmpEQ(cmp_result,
        llvm::ConstantInt::get(ctx_.int32Type(), 0));

    ctx_.builder().CreateCondBr(is_match, add_segment, not_found);

    // Not found at this position, advance
    ctx_.builder().SetInsertPoint(not_found);
    llvm::Value* next_pos = ctx_.builder().CreateAdd(curr_pos, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_pos, pos_ptr);
    ctx_.builder().CreateBr(search_cond);

    // Add segment: extract substring from segment_start to pos
    ctx_.builder().SetInsertPoint(add_segment);
    llvm::Value* seg_start = ctx_.builder().CreateLoad(ctx_.int64Type(), segment_start_ptr);
    llvm::Value* seg_len = ctx_.builder().CreateSub(curr_pos, seg_start);

    // Allocate new string for segment
    llvm::Value* seg_buf_size = ctx_.builder().CreateAdd(seg_len, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* seg_buf = ctx_.builder().CreateCall(malloc_func, {seg_buf_size});

    // Copy characters using memcpy
    llvm::Value* src_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), str_ptr, seg_start);
    ctx_.builder().CreateCall(memcpy_func, {seg_buf, src_ptr, seg_len});

    // Null-terminate
    llvm::Value* term_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), seg_buf, seg_len);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int8Type(), 0), term_ptr);

    // Create tagged string value
    llvm::Value* seg_tagged = tagged_.packPtr(seg_buf, ESHKOL_VALUE_STRING_PTR);

    // Cons this segment onto result using callback
    llvm::Value* curr_result = ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_ptr);
    llvm::Value* cons_ptr_int = cons_create_callback_(seg_tagged, curr_result, callback_context_);

    // Pack new cons cell
    llvm::Value* new_result = tagged_.packPtr(
        ctx_.builder().CreateIntToPtr(cons_ptr_int, ctx_.ptrType()),
        ESHKOL_VALUE_CONS_PTR);
    ctx_.builder().CreateStore(new_result, result_ptr);

    // Update segment_start to pos + delim_len
    llvm::Value* check_at_end = ctx_.builder().CreateICmpEQ(curr_pos, str_len);
    llvm::Value* new_seg_start = ctx_.builder().CreateSelect(check_at_end,
        str_len,
        ctx_.builder().CreateAdd(curr_pos, delim_len));
    ctx_.builder().CreateStore(new_seg_start, segment_start_ptr);

    // Advance pos past delimiter (or past end)
    llvm::Value* new_pos = ctx_.builder().CreateSelect(check_at_end,
        ctx_.builder().CreateAdd(str_len, llvm::ConstantInt::get(ctx_.int64Type(), 1)),
        ctx_.builder().CreateAdd(curr_pos, delim_len));
    ctx_.builder().CreateStore(new_pos, pos_ptr);
    ctx_.builder().CreateBr(search_cond);

    // Loop end: return result (segments in reverse order - use Eshkol's reverse if needed)
    ctx_.builder().SetInsertPoint(loop_end);
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_ptr);
}

llvm::Value* StringIOCodegen::stringContains(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("StringIOCodegen::stringContains - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 2) {
        eshkol_warn("string-contains? requires exactly 2 arguments");
        return nullptr;
    }

    // Get string arguments
    llvm::Value* str_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    llvm::Value* substr_arg = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!str_arg || !substr_arg) return nullptr;

    // Extract string pointers
    llvm::Value* ptr1_int = tagged_.unpackInt64(str_arg);
    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(ptr1_int, ctx_.ptrType());
    llvm::Value* ptr2_int = tagged_.unpackInt64(substr_arg);
    llvm::Value* substr_ptr = ctx_.builder().CreateIntToPtr(ptr2_int, ctx_.ptrType());

    // Get strstr function
    llvm::Function* strstr_func = ctx_.funcs().getStrstr();

    // Call strstr and check if result is not null
    llvm::Value* result = ctx_.builder().CreateCall(strstr_func, {str_ptr, substr_ptr});
    llvm::Value* found = ctx_.builder().CreateICmpNE(result,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));

    return tagged_.packBool(found);
}

llvm::Value* StringIOCodegen::stringIndex(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("StringIOCodegen::stringIndex - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 2) {
        eshkol_warn("string-index requires exactly 2 arguments");
        return nullptr;
    }

    // Get string arguments
    llvm::Value* str_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    llvm::Value* substr_arg = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!str_arg || !substr_arg) return nullptr;

    // Extract string pointers
    llvm::Value* ptr1_int = tagged_.unpackInt64(str_arg);
    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(ptr1_int, ctx_.ptrType());
    llvm::Value* ptr2_int = tagged_.unpackInt64(substr_arg);
    llvm::Value* substr_ptr = ctx_.builder().CreateIntToPtr(ptr2_int, ctx_.ptrType());

    // Get strstr function
    llvm::Function* strstr_func = ctx_.funcs().getStrstr();

    // Call strstr
    llvm::Value* result = ctx_.builder().CreateCall(strstr_func, {str_ptr, substr_ptr});
    llvm::Value* found = ctx_.builder().CreateICmpNE(result,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));

    // Calculate index: result - str_ptr, or -1 if not found
    llvm::Value* str_int = ctx_.builder().CreatePtrToInt(str_ptr, ctx_.int64Type());
    llvm::Value* result_int = ctx_.builder().CreatePtrToInt(result, ctx_.int64Type());
    llvm::Value* index = ctx_.builder().CreateSub(result_int, str_int);
    llvm::Value* neg_one = llvm::ConstantInt::get(ctx_.int64Type(), -1);
    llvm::Value* final_index = ctx_.builder().CreateSelect(found, index, neg_one);

    return tagged_.packInt64(final_index, true);
}

llvm::Value* StringIOCodegen::stringUpcase(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("StringIOCodegen::stringUpcase - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("string-upcase requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* str_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!str_arg) return nullptr;

    // Extract string pointer
    llvm::Value* ptr_int = tagged_.unpackInt64(str_arg);
    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());

    llvm::Function* strlen_func = ctx_.funcs().getStrlen();
    llvm::Function* malloc_func = ctx_.funcs().getMalloc();
    llvm::Function* parent_func = ctx_.builder().GetInsertBlock()->getParent();

    // Get string length
    llvm::Value* str_len = ctx_.builder().CreateCall(strlen_func, {str_ptr});
    llvm::Value* buf_size = ctx_.builder().CreateAdd(str_len,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    // Allocate new string
    llvm::Value* new_str = ctx_.builder().CreateCall(malloc_func, {buf_size});

    // Create loop to convert each character
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "upcase_cond", parent_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "upcase_body", parent_func);
    llvm::BasicBlock* loop_end = llvm::BasicBlock::Create(ctx_.context(), "upcase_end", parent_func);

    // Create alloca for index at function entry
    llvm::IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
    llvm::BasicBlock& entry = parent_func->getEntryBlock();
    ctx_.builder().SetInsertPoint(&entry, entry.begin());
    llvm::Value* idx_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "upcase_idx");
    ctx_.builder().restoreIP(saved_ip);

    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx_ptr);
    ctx_.builder().CreateBr(loop_cond);

    // Loop condition
    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* curr_idx = ctx_.builder().CreateLoad(ctx_.int64Type(), idx_ptr);
    llvm::Value* in_bounds = ctx_.builder().CreateICmpULT(curr_idx, str_len);
    ctx_.builder().CreateCondBr(in_bounds, loop_body, loop_end);

    // Loop body: convert character to uppercase
    ctx_.builder().SetInsertPoint(loop_body);
    llvm::Value* src_char_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), str_ptr, curr_idx);
    llvm::Value* src_char = ctx_.builder().CreateLoad(ctx_.int8Type(), src_char_ptr);

    // Check if lowercase: 'a' (97) <= c <= 'z' (122)
    llvm::Value* is_lower_a = ctx_.builder().CreateICmpUGE(src_char,
        llvm::ConstantInt::get(ctx_.int8Type(), 97));
    llvm::Value* is_lower_z = ctx_.builder().CreateICmpULE(src_char,
        llvm::ConstantInt::get(ctx_.int8Type(), 122));
    llvm::Value* is_lower = ctx_.builder().CreateAnd(is_lower_a, is_lower_z);

    // Convert: c - 32 ('a' - 'A' = 32)
    llvm::Value* upper_char = ctx_.builder().CreateSub(src_char,
        llvm::ConstantInt::get(ctx_.int8Type(), 32));
    llvm::Value* result_char = ctx_.builder().CreateSelect(is_lower, upper_char, src_char);

    // Store result
    llvm::Value* dest_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), new_str, curr_idx);
    ctx_.builder().CreateStore(result_char, dest_ptr);

    // Increment index
    llvm::Value* next_idx = ctx_.builder().CreateAdd(curr_idx,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_idx, idx_ptr);
    ctx_.builder().CreateBr(loop_cond);

    // Loop end: null-terminate and return
    ctx_.builder().SetInsertPoint(loop_end);
    llvm::Value* term_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), new_str, str_len);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int8Type(), 0), term_ptr);

    return tagged_.packPtr(new_str, ESHKOL_VALUE_STRING_PTR);
}

llvm::Value* StringIOCodegen::stringDowncase(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("StringIOCodegen::stringDowncase - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("string-downcase requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* str_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!str_arg) return nullptr;

    // Extract string pointer
    llvm::Value* ptr_int = tagged_.unpackInt64(str_arg);
    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());

    llvm::Function* strlen_func = ctx_.funcs().getStrlen();
    llvm::Function* malloc_func = ctx_.funcs().getMalloc();
    llvm::Function* parent_func = ctx_.builder().GetInsertBlock()->getParent();

    // Get string length
    llvm::Value* str_len = ctx_.builder().CreateCall(strlen_func, {str_ptr});
    llvm::Value* buf_size = ctx_.builder().CreateAdd(str_len,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    // Allocate new string
    llvm::Value* new_str = ctx_.builder().CreateCall(malloc_func, {buf_size});

    // Create loop to convert each character
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "downcase_cond", parent_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "downcase_body", parent_func);
    llvm::BasicBlock* loop_end = llvm::BasicBlock::Create(ctx_.context(), "downcase_end", parent_func);

    // Create alloca for index at function entry
    llvm::IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
    llvm::BasicBlock& entry = parent_func->getEntryBlock();
    ctx_.builder().SetInsertPoint(&entry, entry.begin());
    llvm::Value* idx_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "downcase_idx");
    ctx_.builder().restoreIP(saved_ip);

    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx_ptr);
    ctx_.builder().CreateBr(loop_cond);

    // Loop condition
    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* curr_idx = ctx_.builder().CreateLoad(ctx_.int64Type(), idx_ptr);
    llvm::Value* in_bounds = ctx_.builder().CreateICmpULT(curr_idx, str_len);
    ctx_.builder().CreateCondBr(in_bounds, loop_body, loop_end);

    // Loop body: convert character to lowercase
    ctx_.builder().SetInsertPoint(loop_body);
    llvm::Value* src_char_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), str_ptr, curr_idx);
    llvm::Value* src_char = ctx_.builder().CreateLoad(ctx_.int8Type(), src_char_ptr);

    // Check if uppercase: 'A' (65) <= c <= 'Z' (90)
    llvm::Value* is_upper_a = ctx_.builder().CreateICmpUGE(src_char,
        llvm::ConstantInt::get(ctx_.int8Type(), 65));
    llvm::Value* is_upper_z = ctx_.builder().CreateICmpULE(src_char,
        llvm::ConstantInt::get(ctx_.int8Type(), 90));
    llvm::Value* is_upper = ctx_.builder().CreateAnd(is_upper_a, is_upper_z);

    // Convert: c + 32
    llvm::Value* lower_char = ctx_.builder().CreateAdd(src_char,
        llvm::ConstantInt::get(ctx_.int8Type(), 32));
    llvm::Value* result_char = ctx_.builder().CreateSelect(is_upper, lower_char, src_char);

    // Store result
    llvm::Value* dest_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), new_str, curr_idx);
    ctx_.builder().CreateStore(result_char, dest_ptr);

    // Increment index
    llvm::Value* next_idx = ctx_.builder().CreateAdd(curr_idx,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_idx, idx_ptr);
    ctx_.builder().CreateBr(loop_cond);

    // Loop end: null-terminate and return
    ctx_.builder().SetInsertPoint(loop_end);
    llvm::Value* term_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), new_str, str_len);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int8Type(), 0), term_ptr);

    return tagged_.packPtr(new_str, ESHKOL_VALUE_STRING_PTR);
}

llvm::Value* StringIOCodegen::stringToList(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_ || !cons_create_callback_) {
        eshkol_warn("StringIOCodegen::stringToList - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("string->list requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* str_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!str_arg) return nullptr;

    // Extract string pointer
    llvm::Value* ptr_int = tagged_.unpackInt64(str_arg);
    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());

    // Get string length using strlen
    llvm::Function* strlen_func = ctx_.funcs().getStrlen();
    llvm::Value* len = ctx_.builder().CreateCall(strlen_func, {str_ptr});

    // We'll build the list backwards (from end to beginning), which produces
    // correct order without needing reverse

    llvm::Function* parent_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "strtolist_cond", parent_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "strtolist_body", parent_func);
    llvm::BasicBlock* loop_end = llvm::BasicBlock::Create(ctx_.context(), "strtolist_end", parent_func);

    // Create alloca for index and result list at function entry
    llvm::IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
    llvm::BasicBlock& entry = parent_func->getEntryBlock();
    ctx_.builder().SetInsertPoint(&entry, entry.begin());
    llvm::Value* idx_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "stl_idx");
    llvm::Value* list_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "stl_list");
    ctx_.builder().restoreIP(saved_ip);

    // Start from the end of string: idx = len - 1
    llvm::Value* start_idx = ctx_.builder().CreateSub(len, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(start_idx, idx_ptr);

    // Start with empty list (null)
    llvm::Value* nil_val = tagged_.packNull();
    ctx_.builder().CreateStore(nil_val, list_ptr);

    ctx_.builder().CreateBr(loop_cond);

    // Loop condition: idx >= 0
    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* current_idx = ctx_.builder().CreateLoad(ctx_.int64Type(), idx_ptr);
    llvm::Value* cond = ctx_.builder().CreateICmpSGE(current_idx, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(cond, loop_body, loop_end);

    // Loop body: cons char at idx with current list
    ctx_.builder().SetInsertPoint(loop_body);
    llvm::Value* body_idx = ctx_.builder().CreateLoad(ctx_.int64Type(), idx_ptr);
    llvm::Value* char_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), str_ptr, body_idx);
    llvm::Value* char_byte = ctx_.builder().CreateLoad(ctx_.int8Type(), char_ptr);
    llvm::Value* char_i64 = ctx_.builder().CreateZExt(char_byte, ctx_.int64Type());
    llvm::Value* char_tagged = tagged_.packChar(char_i64);

    // Cons this character with current list using callback
    llvm::Value* current_list = ctx_.builder().CreateLoad(ctx_.taggedValueType(), list_ptr);
    llvm::Value* cons_ptr_int = cons_create_callback_(char_tagged, current_list, callback_context_);

    // Pack cons cell pointer as tagged value with CONS_PTR type
    llvm::Value* new_list = tagged_.packPtr(
        ctx_.builder().CreateIntToPtr(cons_ptr_int, ctx_.ptrType()),
        ESHKOL_VALUE_CONS_PTR);
    ctx_.builder().CreateStore(new_list, list_ptr);

    // Decrement index
    llvm::Value* next_idx = ctx_.builder().CreateSub(body_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_idx, idx_ptr);
    ctx_.builder().CreateBr(loop_cond);

    // Loop end
    ctx_.builder().SetInsertPoint(loop_end);
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), list_ptr);
}

llvm::Value* StringIOCodegen::listToString(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("StringIOCodegen::listToString - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("list->string requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* list_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!list_arg) return nullptr;

    // Ensure list_arg is a tagged value (codegenList returns i64 cons pointer)
    if (list_arg->getType() != ctx_.taggedValueType()) {
        // It's an i64 cons pointer, pack it as tagged value
        list_arg = tagged_.packPtr(
            ctx_.builder().CreateIntToPtr(list_arg, ctx_.ptrType()),
            ESHKOL_VALUE_CONS_PTR);
    }

    llvm::Function* parent_func = ctx_.builder().GetInsertBlock()->getParent();

    // First, count the length of the list
    llvm::BasicBlock* count_cond = llvm::BasicBlock::Create(ctx_.context(), "lts_count_cond", parent_func);
    llvm::BasicBlock* count_body = llvm::BasicBlock::Create(ctx_.context(), "lts_count_body", parent_func);
    llvm::BasicBlock* count_end = llvm::BasicBlock::Create(ctx_.context(), "lts_count_end", parent_func);
    llvm::BasicBlock* fill_cond = llvm::BasicBlock::Create(ctx_.context(), "lts_fill_cond", parent_func);
    llvm::BasicBlock* fill_body = llvm::BasicBlock::Create(ctx_.context(), "lts_fill_body", parent_func);
    llvm::BasicBlock* fill_end = llvm::BasicBlock::Create(ctx_.context(), "lts_fill_end", parent_func);

    // Create allocas at function entry
    llvm::IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
    llvm::BasicBlock& entry = parent_func->getEntryBlock();
    ctx_.builder().SetInsertPoint(&entry, entry.begin());
    llvm::Value* count_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "lts_count");
    llvm::Value* list_iter_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "lts_iter");
    llvm::Value* idx_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "lts_idx");
    ctx_.builder().restoreIP(saved_ip);

    // Initialize count = 0, list_iter = list_arg
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), count_ptr);
    ctx_.builder().CreateStore(list_arg, list_iter_ptr);
    ctx_.builder().CreateBr(count_cond);

    // Count loop condition: while list is not nil
    ctx_.builder().SetInsertPoint(count_cond);
    llvm::Value* iter_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), list_iter_ptr);
    llvm::Value* type_val = ctx_.builder().CreateExtractValue(iter_val, {0});
    llvm::Value* is_nil = ctx_.builder().CreateICmpEQ(type_val,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL));
    ctx_.builder().CreateCondBr(is_nil, count_end, count_body);

    // Count loop body: increment count, move to cdr
    ctx_.builder().SetInsertPoint(count_body);
    llvm::Value* curr_count = ctx_.builder().CreateLoad(ctx_.int64Type(), count_ptr);
    llvm::Value* new_count = ctx_.builder().CreateAdd(curr_count, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(new_count, count_ptr);

    // Get cdr of current cons cell
    llvm::Value* curr_iter = ctx_.builder().CreateLoad(ctx_.taggedValueType(), list_iter_ptr);
    llvm::Value* cons_data = ctx_.builder().CreateExtractValue(curr_iter, {4});
    llvm::Value* cons_ptr = ctx_.builder().CreateIntToPtr(cons_data, ctx_.ptrType());
    // cdr is at offset 16 (second tagged value in cons cell)
    llvm::Value* cdr_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), cons_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 16));
    llvm::Value* cdr_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), cdr_ptr);
    ctx_.builder().CreateStore(cdr_val, list_iter_ptr);
    ctx_.builder().CreateBr(count_cond);

    // Count end: allocate string buffer
    ctx_.builder().SetInsertPoint(count_end);
    llvm::Value* final_count = ctx_.builder().CreateLoad(ctx_.int64Type(), count_ptr);
    llvm::Value* buf_size = ctx_.builder().CreateAdd(final_count, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Function* malloc_func = ctx_.funcs().getMalloc();
    llvm::Value* str_buf = ctx_.builder().CreateCall(malloc_func, {buf_size});

    // Reset iterator and index for filling
    ctx_.builder().CreateStore(list_arg, list_iter_ptr);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx_ptr);
    ctx_.builder().CreateBr(fill_cond);

    // Fill loop condition
    ctx_.builder().SetInsertPoint(fill_cond);
    llvm::Value* fill_iter = ctx_.builder().CreateLoad(ctx_.taggedValueType(), list_iter_ptr);
    llvm::Value* fill_type = ctx_.builder().CreateExtractValue(fill_iter, {0});
    llvm::Value* fill_is_nil = ctx_.builder().CreateICmpEQ(fill_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL));
    ctx_.builder().CreateCondBr(fill_is_nil, fill_end, fill_body);

    // Fill loop body: get car (character), store in buffer
    ctx_.builder().SetInsertPoint(fill_body);
    llvm::Value* fill_curr = ctx_.builder().CreateLoad(ctx_.taggedValueType(), list_iter_ptr);
    llvm::Value* fill_cons_data = ctx_.builder().CreateExtractValue(fill_curr, {4});
    llvm::Value* fill_cons_ptr = ctx_.builder().CreateIntToPtr(fill_cons_data, ctx_.ptrType());

    // car is at offset 0
    llvm::Value* car_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), fill_cons_ptr);
    llvm::Value* char_data = ctx_.builder().CreateExtractValue(car_val, {4});
    llvm::Value* char_byte = ctx_.builder().CreateTrunc(char_data, ctx_.int8Type());

    // Store in buffer at current index
    llvm::Value* curr_idx = ctx_.builder().CreateLoad(ctx_.int64Type(), idx_ptr);
    llvm::Value* dest_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), str_buf, curr_idx);
    ctx_.builder().CreateStore(char_byte, dest_ptr);

    // Increment index
    llvm::Value* next_idx = ctx_.builder().CreateAdd(curr_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_idx, idx_ptr);

    // Move to cdr
    llvm::Value* fill_cdr_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), fill_cons_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 16));
    llvm::Value* fill_cdr = ctx_.builder().CreateLoad(ctx_.taggedValueType(), fill_cdr_ptr);
    ctx_.builder().CreateStore(fill_cdr, list_iter_ptr);
    ctx_.builder().CreateBr(fill_cond);

    // Fill end: add null terminator and return
    ctx_.builder().SetInsertPoint(fill_end);
    llvm::Value* term_idx = ctx_.builder().CreateLoad(ctx_.int64Type(), idx_ptr);
    llvm::Value* term_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), str_buf, term_idx);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int8Type(), 0), term_ptr);

    return tagged_.packPtr(str_buf, ESHKOL_VALUE_STRING_PTR);
}

llvm::Value* StringIOCodegen::display(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("StringIOCodegen::display - callbacks not set");
        return tagged_.packNull();
    }

    if (!display_value_func_) {
        eshkol_warn("StringIOCodegen::display - display_value_func not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("display requires exactly 1 argument");
        return nullptr;
    }

    // Check for S-expression lookup (for displaying lambdas/procedures)
    if (op->call_op.variables[0].type == ESHKOL_VAR &&
        op->call_op.variables[0].variable.id) {
        std::string var_name = op->call_op.variables[0].variable.id;

        // Try scoped name first (e.g., "enclosing.var_sexpr")
        llvm::GlobalVariable* sexpr_global = nullptr;
        llvm::Function* enclosing_func = ctx_.builder().GetInsertBlock()->getParent();
        if (enclosing_func) {
            std::string scoped_key = enclosing_func->getName().str() + "." + var_name + "_sexpr";
            sexpr_global = ctx_.module().getNamedGlobal(scoped_key);
            if (sexpr_global) {
                eshkol_debug("[DISPLAY] Found scoped S-expression: %s", scoped_key.c_str());
            }
        }

        // Fall back to unscoped name
        if (!sexpr_global) {
            std::string unscoped_key = var_name + "_sexpr";
            sexpr_global = ctx_.module().getNamedGlobal(unscoped_key);
            if (sexpr_global) {
                eshkol_debug("[DISPLAY] Found unscoped S-expression: %s", unscoped_key.c_str());
            }
        }

        if (sexpr_global) {
            // S-expression global exists - generate runtime load and display
            llvm::Value* sexpr_list = ctx_.builder().CreateLoad(ctx_.int64Type(), sexpr_global);

            // Runtime null check
            llvm::Value* is_not_null = ctx_.builder().CreateICmpNE(sexpr_list,
                llvm::ConstantInt::get(ctx_.int64Type(), 0));

            llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
            llvm::BasicBlock* sexpr_ok = llvm::BasicBlock::Create(ctx_.context(), "var_sexpr_ok", current_func);
            llvm::BasicBlock* sexpr_fallback = llvm::BasicBlock::Create(ctx_.context(), "var_sexpr_fallback", current_func);
            llvm::BasicBlock* display_done = llvm::BasicBlock::Create(ctx_.context(), "display_done", current_func);

            ctx_.builder().CreateCondBr(is_not_null, sexpr_ok, sexpr_fallback);

            // Display S-expression if valid
            ctx_.builder().SetInsertPoint(sexpr_ok);
            {
                // Create tagged value with type CONS_PTR for the S-expression
                llvm::Value* sexpr_tagged = llvm::UndefValue::get(ctx_.taggedValueType());
                llvm::Value* cons_type = llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CONS_PTR);
                sexpr_tagged = ctx_.builder().CreateInsertValue(sexpr_tagged, cons_type, {0});
                sexpr_tagged = ctx_.builder().CreateInsertValue(sexpr_tagged, sexpr_list, {4});

                llvm::Value* sexpr_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "display_sexpr");
                ctx_.builder().CreateStore(sexpr_tagged, sexpr_ptr);
                ctx_.builder().CreateCall(display_value_func_, {sexpr_ptr});
            }
            ctx_.builder().CreateBr(display_done);

            // Fallback: use typed AST codegen
            ctx_.builder().SetInsertPoint(sexpr_fallback);
            {
                void* tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
                if (tv_ptr) {
                    llvm::Value* fallback_tagged = typed_to_tagged_callback_(tv_ptr, callback_context_);
                    if (fallback_tagged) {
                        llvm::Value* fallback_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "display_fallback");
                        ctx_.builder().CreateStore(fallback_tagged, fallback_ptr);
                        ctx_.builder().CreateCall(display_value_func_, {fallback_ptr});
                    }
                }
            }
            ctx_.builder().CreateBr(display_done);

            ctx_.builder().SetInsertPoint(display_done);
            return llvm::ConstantInt::get(ctx_.int32Type(), 0);
        }
    }

    // Standard display: use codegenTypedAST to get typed value, then call C display function
    void* tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!tv_ptr) return nullptr;

    llvm::Value* tagged = typed_to_tagged_callback_(tv_ptr, callback_context_);
    if (!tagged) return nullptr;

    // Store on stack and call unified C display function
    llvm::Value* display_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "display_arg");
    ctx_.builder().CreateStore(tagged, display_ptr);
    ctx_.builder().CreateCall(display_value_func_, {display_ptr});

    return llvm::ConstantInt::get(ctx_.int32Type(), 0);
}

// === File I/O Operations ===

// Helper to get or declare fopen
static llvm::Function* getOrDeclareFopen(CodegenContext& ctx) {
    if (auto* existing = ctx.module().getFunction("fopen")) return existing;
    auto* ft = llvm::FunctionType::get(ctx.ptrType(),
        {ctx.ptrType(), ctx.ptrType()}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "fopen", ctx.module());
}

// Helper to get or declare fgets
static llvm::Function* getOrDeclareFgets(CodegenContext& ctx) {
    if (auto* existing = ctx.module().getFunction("fgets")) return existing;
    auto* ft = llvm::FunctionType::get(ctx.ptrType(),
        {ctx.ptrType(), ctx.int32Type(), ctx.ptrType()}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "fgets", ctx.module());
}

// Helper to get or declare fclose
static llvm::Function* getOrDeclareFclose(CodegenContext& ctx) {
    if (auto* existing = ctx.module().getFunction("fclose")) return existing;
    auto* ft = llvm::FunctionType::get(ctx.int32Type(), {ctx.ptrType()}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "fclose", ctx.module());
}

// Helper to get or declare fputs
static llvm::Function* getOrDeclareFputs(CodegenContext& ctx) {
    if (auto* existing = ctx.module().getFunction("fputs")) return existing;
    auto* ft = llvm::FunctionType::get(ctx.int32Type(),
        {ctx.ptrType(), ctx.ptrType()}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "fputs", ctx.module());
}

// Helper to get or declare fputc
static llvm::Function* getOrDeclareFputc(CodegenContext& ctx) {
    if (auto* existing = ctx.module().getFunction("fputc")) return existing;
    auto* ft = llvm::FunctionType::get(ctx.int32Type(),
        {ctx.int32Type(), ctx.ptrType()}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "fputc", ctx.module());
}

// Helper to get or declare fflush
static llvm::Function* getOrDeclareFflush(CodegenContext& ctx) {
    if (auto* existing = ctx.module().getFunction("fflush")) return existing;
    auto* ft = llvm::FunctionType::get(ctx.int32Type(), {ctx.ptrType()}, false);
    return llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "fflush", ctx.module());
}

// Helper to get stdout global variable
// On macOS/Darwin, stdout is __stdoutp; on Linux it's stdout
static llvm::Value* getStdout(CodegenContext& ctx) {
#ifdef __APPLE__
    const char* stdout_name = "__stdoutp";
#else
    const char* stdout_name = "stdout";
#endif
    llvm::GlobalVariable* stdout_var = ctx.module().getGlobalVariable(stdout_name);
    if (!stdout_var) {
        stdout_var = new llvm::GlobalVariable(
            ctx.module(), ctx.ptrType(), false,
            llvm::GlobalVariable::ExternalLinkage, nullptr, stdout_name);
    }
    return ctx.builder().CreateLoad(ctx.ptrType(), stdout_var);
}

llvm::Value* StringIOCodegen::openInputFile(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("StringIOCodegen::openInputFile - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("open-input-file requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* fopen_func = getOrDeclareFopen(ctx_);
    if (!fopen_func) return nullptr;

    // Get filename argument (should be a string)
    void* tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!tv_ptr) return nullptr;

    llvm::Value* tagged = typed_to_tagged_callback_(tv_ptr, callback_context_);
    if (!tagged) return nullptr;

    llvm::Value* filename_ptr = ctx_.builder().CreateIntToPtr(
        ctx_.builder().CreateExtractValue(tagged, {4}),
        ctx_.ptrType());

    // Call fopen with "r" mode
    llvm::Value* mode = createString("r");
    llvm::Value* file_ptr = ctx_.builder().CreateCall(fopen_func, {filename_ptr, mode});

    // Convert FILE* to i64 for storage in tagged value
    llvm::Value* file_ptr_int = ctx_.builder().CreatePtrToInt(file_ptr, ctx_.int64Type());

    // Pack as a tagged value with a special "port" type
    // We'll use ESHKOL_VALUE_CONS_PTR + 0x10 flag to indicate it's a port
    llvm::Value* result = llvm::UndefValue::get(ctx_.taggedValueType());
    result = ctx_.builder().CreateInsertValue(result,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CONS_PTR | 0x10), {0}); // type = port
    result = ctx_.builder().CreateInsertValue(result,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1}); // flags
    result = ctx_.builder().CreateInsertValue(result,
        llvm::ConstantInt::get(ctx_.int16Type(), 0), {2}); // reserved
    result = ctx_.builder().CreateInsertValue(result, file_ptr_int, {4}); // data = FILE* at index 4

    return result;
}

llvm::Value* StringIOCodegen::openOutputFile(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("StringIOCodegen::openOutputFile - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("open-output-file requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* fopen_func = getOrDeclareFopen(ctx_);
    if (!fopen_func) return nullptr;

    // Get filename argument (should be a string)
    void* tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!tv_ptr) return nullptr;

    llvm::Value* tagged = typed_to_tagged_callback_(tv_ptr, callback_context_);
    if (!tagged) return nullptr;

    llvm::Value* filename_ptr = ctx_.builder().CreateIntToPtr(
        ctx_.builder().CreateExtractValue(tagged, {4}),
        ctx_.ptrType());

    // Call fopen with "w" mode
    llvm::Value* mode = createString("w");
    llvm::Value* file_ptr = ctx_.builder().CreateCall(fopen_func, {filename_ptr, mode});

    // Convert FILE* to i64 for storage in tagged value
    llvm::Value* file_ptr_int = ctx_.builder().CreatePtrToInt(file_ptr, ctx_.int64Type());

    // Pack as a tagged value with output port type
    // NOTE: Use 0x40 instead of 0x20 because CONS_PTR=32=0x20, so 32|0x20=32!
    llvm::Value* result = llvm::UndefValue::get(ctx_.taggedValueType());
    result = ctx_.builder().CreateInsertValue(result,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CONS_PTR | 0x40), {0}); // type = output port
    result = ctx_.builder().CreateInsertValue(result,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1}); // flags
    result = ctx_.builder().CreateInsertValue(result,
        llvm::ConstantInt::get(ctx_.int16Type(), 0), {2}); // reserved
    result = ctx_.builder().CreateInsertValue(result, file_ptr_int, {4}); // data = FILE*

    return result;
}

llvm::Value* StringIOCodegen::readLine(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("StringIOCodegen::readLine - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("read-line requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* fgets_func = getOrDeclareFgets(ctx_);
    llvm::Function* strlen_func = ctx_.funcs().getStrlen();
    llvm::Function* malloc_func = ctx_.funcs().getMalloc();
    if (!fgets_func || !strlen_func || !malloc_func) return nullptr;

    // Get port argument
    void* tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!tv_ptr) return nullptr;

    llvm::Value* tagged = typed_to_tagged_callback_(tv_ptr, callback_context_);
    if (!tagged) return nullptr;

    llvm::Value* file_ptr_int = ctx_.builder().CreateExtractValue(tagged, {4});
    llvm::Value* file_ptr = ctx_.builder().CreateIntToPtr(file_ptr_int, ctx_.ptrType());

    // Allocate a buffer for reading (1024 bytes)
    llvm::Value* buffer_size = llvm::ConstantInt::get(ctx_.int64Type(), 1024);
    llvm::Value* buffer = ctx_.builder().CreateCall(malloc_func, {buffer_size});

    // Call fgets
    llvm::Value* result_ptr = ctx_.builder().CreateCall(fgets_func, {
        buffer,
        llvm::ConstantInt::get(ctx_.int32Type(), 1024),
        file_ptr
    });

    // Check if fgets returned NULL (EOF or error)
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* eof_block = llvm::BasicBlock::Create(ctx_.context(), "read_eof", current_func);
    llvm::BasicBlock* success_block = llvm::BasicBlock::Create(ctx_.context(), "read_success", current_func);
    llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "read_done", current_func);

    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(result_ptr,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));
    ctx_.builder().CreateCondBr(is_null, eof_block, success_block);

    // EOF case: return eof-object (tagged value with type 0xFF)
    ctx_.builder().SetInsertPoint(eof_block);
    llvm::Value* eof_result = llvm::UndefValue::get(ctx_.taggedValueType());
    eof_result = ctx_.builder().CreateInsertValue(eof_result,
        llvm::ConstantInt::get(ctx_.int8Type(), 0xFF), {0}); // type = eof-object
    eof_result = ctx_.builder().CreateInsertValue(eof_result,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    eof_result = ctx_.builder().CreateInsertValue(eof_result,
        llvm::ConstantInt::get(ctx_.int16Type(), 0), {2});
    eof_result = ctx_.builder().CreateInsertValue(eof_result,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), {4}); // data at index 4
    ctx_.builder().CreateBr(done_block);

    // Success case: strip newline and return string
    ctx_.builder().SetInsertPoint(success_block);

    // Get string length
    llvm::Value* len = ctx_.builder().CreateCall(strlen_func, {buffer});

    // Strip trailing newline if present
    llvm::Value* last_idx = ctx_.builder().CreateSub(len, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* last_char_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), buffer, last_idx);
    llvm::Value* last_char = ctx_.builder().CreateLoad(ctx_.int8Type(), last_char_ptr);
    llvm::Value* is_newline = ctx_.builder().CreateICmpEQ(last_char, llvm::ConstantInt::get(ctx_.int8Type(), '\n'));

    llvm::BasicBlock* strip_block = llvm::BasicBlock::Create(ctx_.context(), "strip_newline", current_func);
    llvm::BasicBlock* no_strip_block = llvm::BasicBlock::Create(ctx_.context(), "no_strip", current_func);
    ctx_.builder().CreateCondBr(is_newline, strip_block, no_strip_block);

    ctx_.builder().SetInsertPoint(strip_block);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int8Type(), 0), last_char_ptr);
    ctx_.builder().CreateBr(no_strip_block);

    ctx_.builder().SetInsertPoint(no_strip_block);

    // Pack string as tagged value
    llvm::Value* buffer_int = ctx_.builder().CreatePtrToInt(buffer, ctx_.int64Type());
    llvm::Value* str_result = llvm::UndefValue::get(ctx_.taggedValueType());
    str_result = ctx_.builder().CreateInsertValue(str_result,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_STRING_PTR), {0});
    str_result = ctx_.builder().CreateInsertValue(str_result,
        llvm::ConstantInt::get(ctx_.int8Type(), 0), {1});
    str_result = ctx_.builder().CreateInsertValue(str_result,
        llvm::ConstantInt::get(ctx_.int16Type(), 0), {2});
    str_result = ctx_.builder().CreateInsertValue(str_result, buffer_int, {4}); // data at index 4
    ctx_.builder().CreateBr(done_block);

    // Done block with PHI
    ctx_.builder().SetInsertPoint(done_block);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "read_result");
    phi->addIncoming(eof_result, eof_block);
    phi->addIncoming(str_result, no_strip_block);

    return phi;
}

llvm::Value* StringIOCodegen::closePort(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("StringIOCodegen::closePort - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("close-port requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* fclose_func = getOrDeclareFclose(ctx_);
    if (!fclose_func) return nullptr;

    // Get port argument
    void* tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!tv_ptr) return nullptr;

    llvm::Value* tagged = typed_to_tagged_callback_(tv_ptr, callback_context_);
    if (!tagged) return nullptr;

    llvm::Value* file_ptr_int = ctx_.builder().CreateExtractValue(tagged, {4});
    llvm::Value* file_ptr = ctx_.builder().CreateIntToPtr(file_ptr_int, ctx_.ptrType());

    // Call fclose
    ctx_.builder().CreateCall(fclose_func, {file_ptr});

    // Return void as tagged value (0)
    return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
}

llvm::Value* StringIOCodegen::eofObject(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("StringIOCodegen::eofObject - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("eof-object? requires exactly 1 argument");
        return nullptr;
    }

    // Get argument
    void* tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!tv_ptr) return nullptr;

    llvm::Value* tagged = typed_to_tagged_callback_(tv_ptr, callback_context_);
    if (!tagged) return nullptr;

    llvm::Value* type_byte = ctx_.builder().CreateExtractValue(tagged, {0});

    // Check if type is 0xFF (eof-object)
    llvm::Value* is_eof = ctx_.builder().CreateICmpEQ(type_byte,
        llvm::ConstantInt::get(ctx_.int8Type(), 0xFF));

    return tagged_.packBool(is_eof);
}

llvm::Value* StringIOCodegen::writeString(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("StringIOCodegen::writeString - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_warn("write-string requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Function* fputs_func = getOrDeclareFputs(ctx_);
    if (!fputs_func) return nullptr;

    // Get string argument
    void* str_tv = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!str_tv) return nullptr;
    llvm::Value* str_tagged = typed_to_tagged_callback_(str_tv, callback_context_);
    if (!str_tagged) return nullptr;

    // Extract string pointer
    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(
        ctx_.builder().CreateExtractValue(str_tagged, {4}), ctx_.ptrType());

    llvm::Value* file_ptr;
    if (op->call_op.num_vars == 2) {
        // Get port argument
        void* port_tv = codegen_typed_ast_callback_(&op->call_op.variables[1], callback_context_);
        if (!port_tv) return nullptr;
        llvm::Value* port_tagged = typed_to_tagged_callback_(port_tv, callback_context_);
        if (!port_tagged) return nullptr;

        llvm::Value* file_ptr_int = ctx_.builder().CreateExtractValue(port_tagged, {4});
        file_ptr = ctx_.builder().CreateIntToPtr(file_ptr_int, ctx_.ptrType());
    } else {
        // Write to stdout
        file_ptr = getStdout(ctx_);
    }

    // Call fputs
    ctx_.builder().CreateCall(fputs_func, {str_ptr, file_ptr});

    // Return void as tagged value (0)
    return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
}

llvm::Value* StringIOCodegen::writeLine(const eshkol_operations_t* op) {
    // (write-line str [port]) -> unspecified
    // Writes string followed by newline
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("StringIOCodegen::writeLine - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_warn("write-line requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Function* fputs_func = getOrDeclareFputs(ctx_);
    llvm::Function* fputc_func = getOrDeclareFputc(ctx_);
    if (!fputs_func || !fputc_func) return nullptr;

    // Get string argument
    void* str_tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!str_tv_ptr) return nullptr;

    llvm::Value* tagged_str = typed_to_tagged_callback_(str_tv_ptr, callback_context_);
    if (!tagged_str) return nullptr;

    llvm::Value* str_ptr = ctx_.builder().CreateIntToPtr(
        tagged_.unpackInt64(tagged_str),
        ctx_.ptrType());

    llvm::Value* file_ptr;
    if (op->call_op.num_vars == 2) {
        // Get port argument
        void* port_tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[1], callback_context_);
        if (!port_tv_ptr) return nullptr;

        llvm::Value* tagged_port = typed_to_tagged_callback_(port_tv_ptr, callback_context_);
        if (!tagged_port) return nullptr;

        file_ptr = ctx_.builder().CreateIntToPtr(
            tagged_.unpackInt64(tagged_port),
            ctx_.ptrType());
    } else {
        // Write to stdout
        file_ptr = getStdout(ctx_);
    }

    // Call fputs for string, then fputc for newline
    ctx_.builder().CreateCall(fputs_func, {str_ptr, file_ptr});
    ctx_.builder().CreateCall(fputc_func, {
        llvm::ConstantInt::get(ctx_.int32Type(), '\n'),
        file_ptr
    });

    // Return void as tagged value (0)
    return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
}

llvm::Value* StringIOCodegen::writeChar(const eshkol_operations_t* op) {
    // (write-char char [port]) -> unspecified
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("StringIOCodegen::writeChar - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_warn("write-char requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Function* fputc_func = getOrDeclareFputc(ctx_);
    if (!fputc_func) return nullptr;

    // Get char argument
    void* char_tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!char_tv_ptr) return nullptr;

    // Extract raw LLVM value from TypedValue
    llvm::Value* char_llvm_val = *reinterpret_cast<llvm::Value**>(char_tv_ptr);
    if (!char_llvm_val) return nullptr;

    llvm::Value* char_val;
    if (char_llvm_val->getType()->isIntegerTy(64)) {
        char_val = ctx_.builder().CreateTrunc(char_llvm_val, ctx_.int32Type());
    } else if (char_llvm_val->getType()->isIntegerTy(32)) {
        char_val = char_llvm_val;
    } else {
        // Handle tagged value
        llvm::Value* tagged = typed_to_tagged_callback_(char_tv_ptr, callback_context_);
        if (!tagged) return nullptr;
        llvm::Value* data = tagged_.unpackInt64(tagged);
        char_val = ctx_.builder().CreateTrunc(data, ctx_.int32Type());
    }

    llvm::Value* file_ptr;
    if (op->call_op.num_vars == 2) {
        // Get port argument
        void* port_tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[1], callback_context_);
        if (!port_tv_ptr) return nullptr;

        llvm::Value* tagged_port = typed_to_tagged_callback_(port_tv_ptr, callback_context_);
        if (!tagged_port) return nullptr;

        file_ptr = ctx_.builder().CreateIntToPtr(
            tagged_.unpackInt64(tagged_port),
            ctx_.ptrType());
    } else {
        // Write to stdout
        file_ptr = getStdout(ctx_);
    }

    // Call fputc
    ctx_.builder().CreateCall(fputc_func, {char_val, file_ptr});

    // Return void as tagged value (0)
    return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
}

llvm::Value* StringIOCodegen::flushOutputPort(const eshkol_operations_t* op) {
    // (flush-output-port port) -> unspecified
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("StringIOCodegen::flushOutputPort - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("flush-output-port requires exactly 1 argument");
        return nullptr;
    }

    llvm::Function* fflush_func = getOrDeclareFflush(ctx_);
    if (!fflush_func) return nullptr;

    // Get port argument
    void* port_tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!port_tv_ptr) return nullptr;

    llvm::Value* tagged = typed_to_tagged_callback_(port_tv_ptr, callback_context_);
    if (!tagged) return nullptr;

    llvm::Value* file_ptr = ctx_.builder().CreateIntToPtr(
        tagged_.unpackInt64(tagged),
        ctx_.ptrType());

    // Call fflush
    ctx_.builder().CreateCall(fflush_func, {file_ptr});

    // Return void as tagged value (0)
    return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
}

// === Character Operations ===

llvm::Value* StringIOCodegen::charToInteger(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("StringIOCodegen::charToInteger - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("char->integer requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!arg) return nullptr;

    // Extract the integer value from the char's data field
    llvm::Value* char_val = tagged_.unpackInt64(arg);
    return tagged_.packInt64(char_val, true);
}

llvm::Value* StringIOCodegen::integerToChar(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_) {
        eshkol_warn("StringIOCodegen::integerToChar - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("integer->char requires exactly 1 argument");
        return nullptr;
    }

    // Get typed value via callback to get raw integer
    void* tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!tv_ptr) return nullptr;

    // Extract raw LLVM value from TypedValue
    llvm::Value* int_val = *reinterpret_cast<llvm::Value**>(tv_ptr);
    if (!int_val) return nullptr;

    // Pack as character
    return tagged_.packChar(int_val);
}

llvm::Value* StringIOCodegen::charCompare(const eshkol_operations_t* op, const std::string& cmp_type) {
    if (!codegen_ast_callback_) {
        eshkol_warn("StringIOCodegen::charCompare - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 2) {
        eshkol_warn("Character comparison requires exactly 2 arguments");
        return nullptr;
    }

    llvm::Value* char1_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    llvm::Value* char2_arg = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!char1_arg || !char2_arg) return nullptr;

    llvm::Value* char1_val = tagged_.unpackInt64(char1_arg);
    llvm::Value* char2_val = tagged_.unpackInt64(char2_arg);

    llvm::Value* result;
    if (cmp_type == "eq") {
        result = ctx_.builder().CreateICmpEQ(char1_val, char2_val);
    } else if (cmp_type == "lt") {
        result = ctx_.builder().CreateICmpSLT(char1_val, char2_val);
    } else if (cmp_type == "gt") {
        result = ctx_.builder().CreateICmpSGT(char1_val, char2_val);
    } else if (cmp_type == "le") {
        result = ctx_.builder().CreateICmpSLE(char1_val, char2_val);
    } else if (cmp_type == "ge") {
        result = ctx_.builder().CreateICmpSGE(char1_val, char2_val);
    } else {
        result = llvm::ConstantInt::getFalse(ctx_.context());
    }

    return tagged_.packBool(result);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
