/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TaggedValueCodegen implementation - Pack/unpack operations for tagged values
 */

#include <eshkol/backend/tagged_value_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>

namespace eshkol {

TaggedValueCodegen::TaggedValueCodegen(CodegenContext& ctx)
    : ctx_(ctx) {
}

// === Helper ===

llvm::Value* TaggedValueCodegen::createEntryAlloca(const char* name) {
    llvm::IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    if (func && !func->empty()) {
        llvm::BasicBlock& entry = func->getEntryBlock();
        ctx_.builder().SetInsertPoint(&entry, entry.begin());
    }

    llvm::Value* alloca = ctx_.builder().CreateAlloca(
        ctx_.taggedValueType(), nullptr, name);

    ctx_.builder().restoreIP(saved_ip);
    return alloca;
}

// === Pack Functions ===

llvm::Value* TaggedValueCodegen::packInt64(llvm::Value* int64_val, bool is_exact) {
    llvm::Value* tagged_val_ptr = createEntryAlloca("tagged_val");

    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 0);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64), type_ptr);

    llvm::Value* flags_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 1);
    uint8_t flags = is_exact ? ESHKOL_VALUE_EXACT_FLAG : 0;
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), flags), flags_ptr);

    llvm::Value* reserved_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 2);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int16Type(), 0), reserved_ptr);

    llvm::Value* data_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 4);
    ctx_.builder().CreateStore(int64_val, data_ptr);

    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), tagged_val_ptr);
}

llvm::Value* TaggedValueCodegen::packInt64WithType(
    llvm::Value* int64_val,
    eshkol_value_type_t type,
    uint8_t flags) {

    llvm::Value* tagged_val_ptr = createEntryAlloca("tagged_val_typed");

    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 0);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), type), type_ptr);

    llvm::Value* flags_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 1);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), flags), flags_ptr);

    llvm::Value* reserved_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 2);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int16Type(), 0), reserved_ptr);

    llvm::Value* data_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 4);

    // Ensure value is i64
    llvm::Value* val_as_i64;
    if (int64_val->getType()->isIntegerTy(64)) {
        val_as_i64 = int64_val;
    } else if (int64_val->getType()->isPointerTy()) {
        val_as_i64 = ctx_.builder().CreatePtrToInt(int64_val, ctx_.int64Type());
    } else if (int64_val->getType()->isIntegerTy()) {
        val_as_i64 = ctx_.builder().CreateZExtOrTrunc(int64_val, ctx_.int64Type());
    } else {
        eshkol_warn("packInt64WithType: unexpected type, defaulting to 0");
        val_as_i64 = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    }

    ctx_.builder().CreateStore(val_as_i64, data_ptr);
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), tagged_val_ptr);
}

llvm::Value* TaggedValueCodegen::packBool(llvm::Value* bool_val) {
    llvm::Value* tagged_val_ptr = createEntryAlloca("tagged_bool");

    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 0);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_BOOL), type_ptr);

    llvm::Value* flags_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 1);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), 0), flags_ptr);

    llvm::Value* reserved_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 2);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int16Type(), 0), reserved_ptr);

    llvm::Value* data_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 4);
    // Extend boolean to int64 for storage
    llvm::Value* int64_val = ctx_.builder().CreateZExt(bool_val, ctx_.int64Type());
    ctx_.builder().CreateStore(int64_val, data_ptr);

    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), tagged_val_ptr);
}

llvm::Value* TaggedValueCodegen::packInt64WithTypeAndFlags(
    llvm::Value* int64_val,
    llvm::Value* type_val,
    llvm::Value* flags_val) {

    llvm::Value* tagged_val_ptr = createEntryAlloca("tagged_val");

    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 0);
    ctx_.builder().CreateStore(type_val, type_ptr);

    llvm::Value* flags_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 1);
    ctx_.builder().CreateStore(flags_val, flags_ptr);

    llvm::Value* reserved_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 2);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int16Type(), 0), reserved_ptr);

    llvm::Value* data_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 4);
    ctx_.builder().CreateStore(int64_val, data_ptr);

    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), tagged_val_ptr);
}

llvm::Value* TaggedValueCodegen::packDouble(llvm::Value* double_val) {
    llvm::Value* tagged_val_ptr = createEntryAlloca("tagged_val");

    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 0);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE), type_ptr);

    llvm::Value* flags_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 1);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INEXACT_FLAG), flags_ptr);

    llvm::Value* reserved_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 2);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int16Type(), 0), reserved_ptr);

    llvm::Value* data_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 4);
    llvm::Value* double_as_int64 = ctx_.builder().CreateBitCast(
        double_val, ctx_.int64Type());
    ctx_.builder().CreateStore(double_as_int64, data_ptr);

    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), tagged_val_ptr);
}

llvm::Value* TaggedValueCodegen::packPtr(
    llvm::Value* ptr_val,
    eshkol_value_type_t type,
    uint8_t flags) {

    llvm::Value* tagged_val_ptr = createEntryAlloca("tagged_val");

    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 0);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), type), type_ptr);

    llvm::Value* flags_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 1);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), flags), flags_ptr);

    llvm::Value* reserved_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 2);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int16Type(), 0), reserved_ptr);

    llvm::Value* data_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 4);

    // Handle ptr_val that might already be i64 (from PtrToInt elsewhere)
    llvm::Value* ptr_as_int64;
    if (ptr_val->getType()->isIntegerTy(64)) {
        ptr_as_int64 = ptr_val;
    } else if (ptr_val->getType()->isPointerTy()) {
        ptr_as_int64 = ctx_.builder().CreatePtrToInt(ptr_val, ctx_.int64Type());
    } else {
        eshkol_warn("packPtr: unexpected type, defaulting to 0");
        ptr_as_int64 = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    }

    ctx_.builder().CreateStore(ptr_as_int64, data_ptr);
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), tagged_val_ptr);
}

llvm::Value* TaggedValueCodegen::packPtrWithFlags(
    llvm::Value* ptr_val,
    llvm::Value* type_val,
    llvm::Value* flags_val) {

    llvm::Value* tagged_val_ptr = createEntryAlloca("tagged_val_dyn");

    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 0);
    ctx_.builder().CreateStore(type_val, type_ptr);

    llvm::Value* flags_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 1);
    ctx_.builder().CreateStore(flags_val, flags_ptr);

    llvm::Value* reserved_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 2);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int16Type(), 0), reserved_ptr);

    llvm::Value* data_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 4);

    // Convert pointer to i64 if needed
    llvm::Value* ptr_as_int64;
    if (ptr_val->getType()->isIntegerTy(64)) {
        ptr_as_int64 = ptr_val;
    } else if (ptr_val->getType()->isPointerTy()) {
        ptr_as_int64 = ctx_.builder().CreatePtrToInt(ptr_val, ctx_.int64Type());
    } else {
        eshkol_warn("packPtrWithFlags: unexpected type, defaulting to 0");
        ptr_as_int64 = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    }

    ctx_.builder().CreateStore(ptr_as_int64, data_ptr);
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), tagged_val_ptr);
}

llvm::Value* TaggedValueCodegen::packNull() {
    llvm::Value* tagged_val_ptr = createEntryAlloca("tagged_null");

    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 0);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL), type_ptr);

    llvm::Value* flags_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 1);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), 0), flags_ptr);

    llvm::Value* reserved_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 2);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int16Type(), 0), reserved_ptr);

    llvm::Value* data_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 4);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int64Type(), 0), data_ptr);

    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), tagged_val_ptr);
}

llvm::Value* TaggedValueCodegen::packChar(llvm::Value* char_val) {
    llvm::Value* tagged_val_ptr = createEntryAlloca("char_tagged");

    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 0);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CHAR), type_ptr);

    llvm::Value* flags_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 1);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int8Type(), 0), flags_ptr);

    llvm::Value* reserved_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 2);
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int16Type(), 0), reserved_ptr);

    llvm::Value* data_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), tagged_val_ptr, 4);

    // Ensure char_val is i64
    llvm::Value* char_as_i64;
    if (char_val->getType()->isIntegerTy(64)) {
        char_as_i64 = char_val;
    } else if (char_val->getType()->isIntegerTy()) {
        // Only ZExt if it's an integer type (not struct)
        char_as_i64 = ctx_.builder().CreateZExt(char_val, ctx_.int64Type());
    } else if (char_val->getType() == ctx_.taggedValueType()) {
        // If it's already a tagged value, unpack it first
        char_as_i64 = unpackInt64(char_val);
    } else {
        // Fallback: try to interpret as pointer and convert
        eshkol_warn("packChar: unexpected type, attempting pointer cast");
        char_as_i64 = ctx_.builder().CreatePtrToInt(char_val, ctx_.int64Type());
    }
    ctx_.builder().CreateStore(char_as_i64, data_ptr);

    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), tagged_val_ptr);
}

// === Unpack Functions ===

llvm::Value* TaggedValueCodegen::getType(llvm::Value* tagged_val) {
    llvm::Value* temp_ptr = createEntryAlloca("temp_tagged");
    ctx_.builder().CreateStore(tagged_val, temp_ptr);

    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), temp_ptr, 0);
    return ctx_.builder().CreateLoad(ctx_.int8Type(), type_ptr);
}

llvm::Value* TaggedValueCodegen::getFlags(llvm::Value* tagged_val) {
    llvm::Value* temp_ptr = createEntryAlloca("temp_tagged_flags");
    ctx_.builder().CreateStore(tagged_val, temp_ptr);

    llvm::Value* flags_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), temp_ptr, 1);
    return ctx_.builder().CreateLoad(ctx_.int8Type(), flags_ptr);
}

llvm::Value* TaggedValueCodegen::unpackInt64(llvm::Value* tagged_val) {
    llvm::Value* temp_ptr = createEntryAlloca("temp_tagged");
    ctx_.builder().CreateStore(tagged_val, temp_ptr);

    llvm::Value* data_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), temp_ptr, 4);
    return ctx_.builder().CreateLoad(ctx_.int64Type(), data_ptr);
}

llvm::Value* TaggedValueCodegen::unpackDouble(llvm::Value* tagged_val) {
    llvm::Value* temp_ptr = createEntryAlloca("temp_tagged");
    ctx_.builder().CreateStore(tagged_val, temp_ptr);

    llvm::Value* data_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), temp_ptr, 4);
    llvm::Value* data_as_int64 = ctx_.builder().CreateLoad(ctx_.int64Type(), data_ptr);
    return ctx_.builder().CreateBitCast(data_as_int64, ctx_.doubleType());
}

llvm::Value* TaggedValueCodegen::unpackPtr(llvm::Value* tagged_val) {
    llvm::Value* temp_ptr = createEntryAlloca("temp_tagged");
    ctx_.builder().CreateStore(tagged_val, temp_ptr);

    llvm::Value* data_ptr = ctx_.builder().CreateStructGEP(
        ctx_.taggedValueType(), temp_ptr, 4);
    llvm::Value* data_as_int64 = ctx_.builder().CreateLoad(ctx_.int64Type(), data_ptr);
    return ctx_.builder().CreateIntToPtr(data_as_int64, ctx_.ptrType());
}

// === Utility Functions ===

llvm::Value* TaggedValueCodegen::safeExtractInt64(llvm::Value* val) {
    if (!val) {
        return llvm::ConstantInt::get(ctx_.int64Type(), 0);
    }

    llvm::Type* val_type = val->getType();

    // Already i64 - return as-is
    if (val_type->isIntegerTy(64)) {
        return val;
    }

    // If it's a tagged_value struct, unpack the i64 data
    if (val_type == ctx_.taggedValueType()) {
        return unpackInt64(val);
    }

    // Other integer types - extend/truncate to i64
    if (val_type->isIntegerTy()) {
        return ctx_.builder().CreateSExtOrTrunc(val, ctx_.int64Type());
    }

    // Pointer types - convert to i64
    if (val_type->isPointerTy()) {
        return ctx_.builder().CreatePtrToInt(val, ctx_.int64Type());
    }

    // Float types - convert to i64
    if (val_type->isFloatingPointTy()) {
        return ctx_.builder().CreateFPToSI(val, ctx_.int64Type());
    }

    // Fallback - return 0
    return llvm::ConstantInt::get(ctx_.int64Type(), 0);
}

bool TaggedValueCodegen::isTaggedValue(llvm::Value* val) const {
    return val && val->getType() == ctx_.taggedValueType();
}

// === Type Introspection ===

llvm::Value* TaggedValueCodegen::typeOf(llvm::Value* tagged_val) {
    // Get the type tag from the tagged value
    llvm::Value* type_tag = getType(tagged_val);
    // Use getBaseType() to properly handle legacy types (>=32) and exactness flags
    llvm::Value* base_type = getBaseType(type_tag);

    // We need to return a symbol. For now, return the type tag as an integer
    // wrapped in a tagged value with SYMBOL type.
    // A proper implementation would return interned symbol strings like 'integer, 'float, etc.
    // For the HoTT type system, returning the numeric type ID is sufficient for type tests.

    // Create a result based on type tag - return as INT64 for now
    // This allows tests like (= (type-of 42) 1) where 1 is ESHKOL_VALUE_INT64
    return packInt64(ctx_.builder().CreateZExt(base_type, ctx_.int64Type()), true);
}

// === Type Compatibility Checks (M1 Migration) ===
// These generate IR to check if a tagged value is a specific type.
// Currently supports old format. Will be extended to support consolidated types.

llvm::Value* TaggedValueCodegen::isNull(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);
    // Use getBaseType() to properly handle legacy types (>=32) and exactness flags
    llvm::Value* base_type = getBaseType(type_tag);
    return ctx_.builder().CreateICmpEQ(
        base_type, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL));
}

llvm::Value* TaggedValueCodegen::isCons(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);

    // Check legacy format: ESHKOL_VALUE_CONS_PTR (direct comparison, no masking for types >= 32)
    llvm::Value* is_old_cons = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CONS_PTR));

    // Future: Also check new format (ESHKOL_VALUE_HEAP_PTR with HEAP_SUBTYPE_CONS)
    // For now, return legacy format check only
    return is_old_cons;
}

llvm::Value* TaggedValueCodegen::isString(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);

    // Check legacy format: ESHKOL_VALUE_STRING_PTR (direct comparison, no masking for types >= 32)
    llvm::Value* is_old_string = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_STRING_PTR));

    return is_old_string;
}

llvm::Value* TaggedValueCodegen::isVector(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);

    // Check legacy format: ESHKOL_VALUE_VECTOR_PTR (direct comparison, no masking for types >= 32)
    llvm::Value* is_old_vector = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_VECTOR_PTR));

    return is_old_vector;
}

llvm::Value* TaggedValueCodegen::isClosure(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);

    // Check legacy format: ESHKOL_VALUE_CLOSURE_PTR (direct comparison, no masking for types >= 32)
    llvm::Value* is_old_closure = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CLOSURE_PTR));

    return is_old_closure;
}

llvm::Value* TaggedValueCodegen::isLambdaSexpr(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);

    // Check legacy format: ESHKOL_VALUE_LAMBDA_SEXPR (direct comparison, no masking for types >= 32)
    llvm::Value* is_old_lambda = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_LAMBDA_SEXPR));

    return is_old_lambda;
}

llvm::Value* TaggedValueCodegen::isHeapPtr(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);

    // Check all legacy heap pointer types (direct comparison, no masking for types >= 32)
    llvm::Value* is_cons = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CONS_PTR));
    llvm::Value* is_string = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_STRING_PTR));
    llvm::Value* is_vector = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_VECTOR_PTR));
    llvm::Value* is_tensor = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_TENSOR_PTR));
    llvm::Value* is_hash = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HASH_PTR));
    llvm::Value* is_exception = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_EXCEPTION));

    // Check new consolidated HEAP_PTR type
    llvm::Value* is_new_heap = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    // OR all together
    llvm::Value* result = ctx_.builder().CreateOr(is_cons, is_string);
    result = ctx_.builder().CreateOr(result, is_vector);
    result = ctx_.builder().CreateOr(result, is_tensor);
    result = ctx_.builder().CreateOr(result, is_hash);
    result = ctx_.builder().CreateOr(result, is_exception);
    result = ctx_.builder().CreateOr(result, is_new_heap);

    return result;
}

llvm::Value* TaggedValueCodegen::isCallable(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);

    // Check legacy callable types (direct comparison, no masking for types >= 32)
    llvm::Value* is_closure = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CLOSURE_PTR));
    llvm::Value* is_lambda = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_LAMBDA_SEXPR));
    llvm::Value* is_ad_node = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_AD_NODE_PTR));

    // Check new consolidated CALLABLE type
    llvm::Value* is_new_callable = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));

    // OR all together
    llvm::Value* result = ctx_.builder().CreateOr(is_closure, is_lambda);
    result = ctx_.builder().CreateOr(result, is_ad_node);
    result = ctx_.builder().CreateOr(result, is_new_callable);

    return result;
}

llvm::Value* TaggedValueCodegen::isInt64(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);
    // Use getBaseType() to properly handle legacy types (>=32) and exactness flags
    llvm::Value* base_type = getBaseType(type_tag);

    return ctx_.builder().CreateICmpEQ(
        base_type, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
}

llvm::Value* TaggedValueCodegen::isDouble(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);
    // Use getBaseType() to properly handle legacy types (>=32) and exactness flags
    llvm::Value* base_type = getBaseType(type_tag);

    return ctx_.builder().CreateICmpEQ(
        base_type, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
}

llvm::Value* TaggedValueCodegen::isNumeric(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);
    // Use getBaseType() to properly handle legacy types (>=32) and exactness flags
    llvm::Value* base_type = getBaseType(type_tag);

    llvm::Value* is_int = ctx_.builder().CreateICmpEQ(
        base_type, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
    llvm::Value* is_double = ctx_.builder().CreateICmpEQ(
        base_type, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));

    return ctx_.builder().CreateOr(is_int, is_double);
}

llvm::Value* TaggedValueCodegen::isBool(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);
    // Use getBaseType() to properly handle legacy types (>=32) and exactness flags
    llvm::Value* base_type = getBaseType(type_tag);

    return ctx_.builder().CreateICmpEQ(
        base_type, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_BOOL));
}

llvm::Value* TaggedValueCodegen::getBaseType(llvm::Value* type_tag) {
    // Get base type from type tag, handling legacy types correctly:
    // - Immediate types (0-7): may have exactness flags in high bits, mask with 0x0F
    // - Consolidated (8-9), Multimedia (16-19), Legacy (32+): use directly
    //
    // Logic: type >= 8 ? type : (type & 0x0F)
    //
    // This is CRITICAL for pointer consolidation migration!
    // Legacy types like CONS_PTR=32, STRING_PTR=33, CLOSURE_PTR=34 must NOT be masked.
    // 32 & 0x0F = 0 (NULL) - WRONG!
    // 33 & 0x0F = 1 (INT64) - WRONG!

    llvm::Value* is_not_immediate = ctx_.builder().CreateICmpUGE(
        type_tag,
        llvm::ConstantInt::get(ctx_.int8Type(), 8));

    llvm::Value* masked = ctx_.builder().CreateAnd(
        type_tag,
        llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));

    return ctx_.builder().CreateSelect(is_not_immediate, type_tag, masked);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
