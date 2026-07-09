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

/** @brief Construct the tagged-value pack/unpack codegen helper bound to the shared codegen context. */
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

// === SSA Pack Helper ===
// Builds a tagged value using insertvalue (pure SSA, zero allocas).
// Struct: {i8 type, i8 flags, i16 reserved, i32 padding, i64 data}

llvm::Value* TaggedValueCodegen::buildTaggedValue(uint8_t type, uint8_t flags, llvm::Value* data_i64) {
    auto& B = ctx_.builder();
    llvm::Value* v = llvm::UndefValue::get(ctx_.taggedValueType());
    v = B.CreateInsertValue(v, llvm::ConstantInt::get(ctx_.int8Type(), type), {0});
    v = B.CreateInsertValue(v, llvm::ConstantInt::get(ctx_.int8Type(), flags), {1});
    v = B.CreateInsertValue(v, llvm::ConstantInt::get(ctx_.int16Type(), 0), {2});
    v = B.CreateInsertValue(v, llvm::ConstantInt::get(ctx_.int32Type(), 0), {3});
    v = B.CreateInsertValue(v, data_i64, {4});
    return v;
}

/**
 * @brief Like buildTaggedValue(), but with runtime (non-constant) type and
 *        flags values — used when the type/flags byte is itself computed in IR.
 */
llvm::Value* TaggedValueCodegen::buildTaggedValueDyn(llvm::Value* type_val, llvm::Value* flags_val, llvm::Value* data_i64) {
    auto& B = ctx_.builder();
    llvm::Value* v = llvm::UndefValue::get(ctx_.taggedValueType());
    v = B.CreateInsertValue(v, type_val, {0});
    v = B.CreateInsertValue(v, flags_val, {1});
    v = B.CreateInsertValue(v, llvm::ConstantInt::get(ctx_.int16Type(), 0), {2});
    v = B.CreateInsertValue(v, llvm::ConstantInt::get(ctx_.int32Type(), 0), {3});
    v = B.CreateInsertValue(v, data_i64, {4});
    return v;
}

// === Pack Functions (SSA form — zero allocas) ===

llvm::Value* TaggedValueCodegen::packInt64(llvm::Value* int64_val, bool is_exact) {
    uint8_t flags = is_exact ? ESHKOL_VALUE_EXACT_FLAG : 0;
    return buildTaggedValue(ESHKOL_VALUE_INT64, flags, int64_val);
}

/**
 * @brief Pack a value (i64, pointer, or narrower integer) as a tagged value
 *        with an explicit @p type and @p flags rather than the default INT64.
 */
llvm::Value* TaggedValueCodegen::packInt64WithType(
    llvm::Value* int64_val,
    eshkol_value_type_t type,
    uint8_t flags) {

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

    return buildTaggedValue(type, flags, val_as_i64);
}

/** @brief Pack an i1 boolean as a BOOL tagged value. */
llvm::Value* TaggedValueCodegen::packBool(llvm::Value* bool_val) {
    llvm::Value* int64_val = ctx_.builder().CreateZExt(bool_val, ctx_.int64Type());
    return buildTaggedValue(ESHKOL_VALUE_BOOL, 0, int64_val);
}

/** @brief Like packInt64WithType(), but with both type and flags as runtime (non-constant) values. */
llvm::Value* TaggedValueCodegen::packInt64WithTypeAndFlags(
    llvm::Value* int64_val,
    llvm::Value* type_val,
    llvm::Value* flags_val) {
    return buildTaggedValueDyn(type_val, flags_val, int64_val);
}

/** @brief Pack a double as a DOUBLE tagged value (bitcast into the i64 data slot, INEXACT flag set). */
llvm::Value* TaggedValueCodegen::packDouble(llvm::Value* double_val) {
    llvm::Value* double_as_int64 = ctx_.builder().CreateBitCast(double_val, ctx_.int64Type());
    return buildTaggedValue(ESHKOL_VALUE_DOUBLE, ESHKOL_VALUE_INEXACT_FLAG, double_as_int64);
}

/**
 * @brief Normalize any LLVM value (already-tagged struct, double, i64, i1,
 *        or pointer) into a full tagged_value struct.
 *
 * Used before storing an argument into a tagged-value ABI slot: storing a
 * raw scalar directly would leave the type/flags bytes uninitialized.
 * Returns a null tagged value for unrecognized raw types or a null input.
 */
llvm::Value* TaggedValueCodegen::ensureTagged(llvm::Value* val) {
    if (!val) {
        return buildTaggedValue(ESHKOL_VALUE_NULL, 0,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
    }
    llvm::Type* t = val->getType();
    if (t == ctx_.taggedValueType()) return val;
    if (t->isDoubleTy())             return packDouble(val);
    if (t->isIntegerTy(64))          return packInt64(val, true);
    if (t->isIntegerTy(1)) {
        // i1 boolean → BOOL tagged. Extend to i64 so the data slot is filled.
        llvm::Value* ext = ctx_.builder().CreateZExt(val, ctx_.int64Type());
        return buildTaggedValue(ESHKOL_VALUE_BOOL, 0, ext);
    }
    if (t->isPointerTy())            return packPtr(val, ESHKOL_VALUE_HEAP_PTR, 0);
    // Unknown type — warn and return a null tagged_value rather than
    // writing partial bytes into a caller's 16-byte slot.
    eshkol_warn("ensureTagged: unsupported raw LLVM type, returning null");
    return buildTaggedValue(ESHKOL_VALUE_NULL, 0,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
}

/** @brief Pack a pointer (or i64 already holding a pointer bit-pattern) as a tagged value of the given @p type/@p flags. */
llvm::Value* TaggedValueCodegen::packPtr(
    llvm::Value* ptr_val,
    eshkol_value_type_t type,
    uint8_t flags) {

    llvm::Value* ptr_as_int64;
    if (ptr_val->getType()->isIntegerTy(64)) {
        ptr_as_int64 = ptr_val;
    } else if (ptr_val->getType()->isPointerTy()) {
        ptr_as_int64 = ctx_.builder().CreatePtrToInt(ptr_val, ctx_.int64Type());
    } else {
        eshkol_warn("packPtr: unexpected type, defaulting to 0");
        ptr_as_int64 = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    }

    return buildTaggedValue(type, flags, ptr_as_int64);
}

/** @brief Like packPtr(), but with both type and flags as runtime (non-constant) values. */
llvm::Value* TaggedValueCodegen::packPtrWithFlags(
    llvm::Value* ptr_val,
    llvm::Value* type_val,
    llvm::Value* flags_val) {

    llvm::Value* ptr_as_int64;
    if (ptr_val->getType()->isIntegerTy(64)) {
        ptr_as_int64 = ptr_val;
    } else if (ptr_val->getType()->isPointerTy()) {
        ptr_as_int64 = ctx_.builder().CreatePtrToInt(ptr_val, ctx_.int64Type());
    } else {
        eshkol_warn("packPtrWithFlags: unexpected type, defaulting to 0");
        ptr_as_int64 = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    }

    return buildTaggedValueDyn(type_val, flags_val, ptr_as_int64);
}

// ════════════════════════════════════════════════════════════════════════════
// CONSOLIDATED TYPE PACKING (M1 Migration)
// These pack pointers using the new consolidated types (HEAP_PTR/CALLABLE).
// The subtype is stored in the object header (set by with_header allocators).
// ════════════════════════════════════════════════════════════════════════════

llvm::Value* TaggedValueCodegen::packHeapPtr(llvm::Value* ptr_val, uint8_t flags) {
    // Pack pointer using consolidated HEAP_PTR type.
    // The subtype (cons, string, vector, etc.) is in the object header.
    // Objects must be allocated with arena_allocate_*_with_header().
    return packPtr(ptr_val, ESHKOL_VALUE_HEAP_PTR, flags);
}

/** @brief Pack a pointer using the consolidated CALLABLE type (subtype—closure, lambda-sexpr, ad-node—lives in the object header). */
llvm::Value* TaggedValueCodegen::packCallable(llvm::Value* ptr_val, uint8_t flags) {
    // Pack pointer using consolidated CALLABLE type.
    // The subtype (closure, lambda-sexpr, ad-node) is in the object header.
    // Objects must be allocated with arena_allocate_closure_with_header().
    return packPtr(ptr_val, ESHKOL_VALUE_CALLABLE, flags);
}

/** @brief Build the NULL tagged value (via an entry alloca + field stores, then a load). */
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

/**
 * @brief Pack a codepoint value as a CHAR tagged value.
 *
 * Accepts the codepoint as i64, a narrower integer, or an already-tagged
 * value (unpacked first); falls back to a pointer-to-int cast with a
 * warning for any other type.
 */
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

// === Unpack Functions (SSA form — zero allocas) ===

llvm::Value* TaggedValueCodegen::getType(llvm::Value* tagged_val) {
    // Verify input is actually a tagged value struct, not a raw type
    if (tagged_val->getType() != ctx_.taggedValueType()) {
        // Caller passed a non-tagged value — return DOUBLE type tag for raw doubles,
        // INT64 for raw integers (backward compatibility with pre-SSA alloca behavior)
        if (tagged_val->getType()->isDoubleTy()) {
            return llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE);
        } else if (tagged_val->getType()->isIntegerTy(64)) {
            return llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64);
        } else if (tagged_val->getType()->isPointerTy()) {
            return llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR);
        }
        return llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL);
    }
    return ctx_.builder().CreateExtractValue(tagged_val, {0});
}

/** @brief Extract the flags byte (field 1) from a tagged value struct. */
llvm::Value* TaggedValueCodegen::getFlags(llvm::Value* tagged_val) {
    return ctx_.builder().CreateExtractValue(tagged_val, {1});
}

/**
 * @brief Extract the raw i64 data field from a tagged value, or coerce a
 *        raw (untagged) LLVM value of any supported type to i64.
 */
llvm::Value* TaggedValueCodegen::unpackInt64(llvm::Value* tagged_val) {
    if (tagged_val->getType() != ctx_.taggedValueType()) {
        // Raw value — convert to i64 directly
        if (tagged_val->getType()->isIntegerTy(64)) return tagged_val;
        if (tagged_val->getType()->isDoubleTy())
            return ctx_.builder().CreateBitCast(tagged_val, ctx_.int64Type());
        if (tagged_val->getType()->isPointerTy())
            return ctx_.builder().CreatePtrToInt(tagged_val, ctx_.int64Type());
        if (tagged_val->getType()->isIntegerTy())
            return ctx_.builder().CreateZExtOrTrunc(tagged_val, ctx_.int64Type());
        return llvm::ConstantInt::get(ctx_.int64Type(), 0);
    }
    return ctx_.builder().CreateExtractValue(tagged_val, {4});
}

/**
 * @brief Extract a double from a tagged value (bitcasting the i64 data
 *        field) or coerce a raw double/integer/null value to double.
 */
llvm::Value* TaggedValueCodegen::unpackDouble(llvm::Value* tagged_val) {
    if (!tagged_val) {
        return llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    }

    llvm::Type* val_type = tagged_val->getType();
    if (val_type->isDoubleTy()) {
        return tagged_val;
    }
    if (val_type->isIntegerTy()) {
        return ctx_.builder().CreateSIToFP(tagged_val, ctx_.doubleType());
    }
    if (val_type != ctx_.taggedValueType()) {
        eshkol_warn("unpackDouble: unsupported raw LLVM type, returning 0.0");
        return llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    }

    llvm::Value* data_i64 = ctx_.builder().CreateExtractValue(tagged_val, {4});
    return ctx_.builder().CreateBitCast(data_i64, ctx_.doubleType());
}

/** @brief Extract the data field of a tagged value as an LLVM pointer. */
llvm::Value* TaggedValueCodegen::unpackPtr(llvm::Value* tagged_val) {
    llvm::Value* data_i64 = ctx_.builder().CreateExtractValue(tagged_val, {4});
    return ctx_.builder().CreateIntToPtr(data_i64, ctx_.ptrType());
}

/** @brief Extract the data field of a tagged value and truncate it to i1. */
llvm::Value* TaggedValueCodegen::unpackBool(llvm::Value* tagged_val) {
    llvm::Value* data_i64 = ctx_.builder().CreateExtractValue(tagged_val, {4});
    return ctx_.builder().CreateTrunc(data_i64, ctx_.builder().getInt1Ty(), "bool_val");
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

/** @brief Compile-time (non-IR) check: does @p val have the tagged_value LLVM struct type? */
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

/**
 * @brief Emit an i1 IR check for whether @p tagged_val is a cons cell:
 *        HEAP_PTR type whose object header subtype equals HEAP_SUBTYPE_CONS.
 *        Uses branching control flow so the header is only read when the
 *        value is actually a HEAP_PTR.
 */
llvm::Value* TaggedValueCodegen::isCons(llvm::Value* tagged_val) {
    // M1 CONSOLIDATION FIX: Check HEAP_PTR type and HEAP_SUBTYPE_CONS subtype
    // Cons cells are HEAP_PTR with header subtype == HEAP_SUBTYPE_CONS (0)
    // Must use control flow to avoid reading header for non-HEAP_PTR values.

    llvm::Value* type_tag = getType(tagged_val);
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();

    // Check if HEAP_PTR type
    llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    // Create blocks for subtype check
    llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock* check_subtype = llvm::BasicBlock::Create(ctx_.context(), "is_cons.check_subtype", func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "is_cons.merge", func);

    // If HEAP_PTR, check subtype; otherwise return false
    ctx_.builder().CreateCondBr(is_heap_ptr, check_subtype, merge_block);

    // Check subtype in header
    ctx_.builder().SetInsertPoint(check_subtype);
    llvm::Value* is_cons_subtype = checkHeapSubtype(tagged_val, HEAP_SUBTYPE_CONS);
    ctx_.builder().CreateBr(merge_block);
    llvm::BasicBlock* subtype_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.int1Type(), 2, "is_cons.result");
    result->addIncoming(llvm::ConstantInt::getFalse(ctx_.context()), current_block);
    result->addIncoming(is_cons_subtype, subtype_exit);

    return result;
}

/** @brief Emit an i1 IR check for whether @p tagged_val is a string: HEAP_PTR with header subtype HEAP_SUBTYPE_STRING (see isCons()). */
llvm::Value* TaggedValueCodegen::isString(llvm::Value* tagged_val) {
    // M1 CONSOLIDATION FIX: Check HEAP_PTR type and HEAP_SUBTYPE_STRING subtype
    // Strings are HEAP_PTR with header subtype == HEAP_SUBTYPE_STRING (1)
    // Must use control flow to avoid reading header for non-HEAP_PTR values.

    llvm::Value* type_tag = getType(tagged_val);
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();

    // Check if HEAP_PTR type
    llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    // Create blocks for subtype check
    llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock* check_subtype = llvm::BasicBlock::Create(ctx_.context(), "is_string.check_subtype", func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "is_string.merge", func);

    // If HEAP_PTR, check subtype; otherwise return false
    ctx_.builder().CreateCondBr(is_heap_ptr, check_subtype, merge_block);

    // Check subtype in header
    ctx_.builder().SetInsertPoint(check_subtype);
    llvm::Value* is_string_subtype = checkHeapSubtype(tagged_val, HEAP_SUBTYPE_STRING);
    ctx_.builder().CreateBr(merge_block);
    llvm::BasicBlock* subtype_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.int1Type(), 2, "is_string.result");
    result->addIncoming(llvm::ConstantInt::getFalse(ctx_.context()), current_block);
    result->addIncoming(is_string_subtype, subtype_exit);

    return result;
}

/** @brief Emit an i1 IR check for whether @p tagged_val is a vector: HEAP_PTR with header subtype HEAP_SUBTYPE_VECTOR (see isCons()). */
llvm::Value* TaggedValueCodegen::isVector(llvm::Value* tagged_val) {
    // M1 CONSOLIDATION: Check HEAP_PTR type and HEAP_SUBTYPE_VECTOR subtype
    // Vectors are HEAP_PTR with header subtype == HEAP_SUBTYPE_VECTOR (2)

    llvm::Value* type_tag = getType(tagged_val);
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();

    // Check if HEAP_PTR type
    llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    // Create blocks for subtype check
    llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock* check_subtype = llvm::BasicBlock::Create(ctx_.context(), "is_vector.check_subtype", func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "is_vector.merge", func);

    // If HEAP_PTR, check subtype; otherwise return false
    ctx_.builder().CreateCondBr(is_heap_ptr, check_subtype, merge_block);

    // Check subtype in header
    ctx_.builder().SetInsertPoint(check_subtype);
    llvm::Value* is_vector_subtype = checkHeapSubtype(tagged_val, HEAP_SUBTYPE_VECTOR);
    ctx_.builder().CreateBr(merge_block);
    llvm::BasicBlock* subtype_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.int1Type(), 2, "is_vector.result");
    result->addIncoming(llvm::ConstantInt::getFalse(ctx_.context()), current_block);
    result->addIncoming(is_vector_subtype, subtype_exit);

    return result;
}

/** @brief Emit an i1 IR check for whether @p tagged_val is a tensor: HEAP_PTR with header subtype HEAP_SUBTYPE_TENSOR (see isCons()). */
llvm::Value* TaggedValueCodegen::isTensor(llvm::Value* tagged_val) {
    // M1 CONSOLIDATION: Check HEAP_PTR type and HEAP_SUBTYPE_TENSOR subtype
    // Tensors are HEAP_PTR with header subtype == HEAP_SUBTYPE_TENSOR (3)

    llvm::Value* type_tag = getType(tagged_val);
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();

    // Check if HEAP_PTR type
    llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    // Create blocks for subtype check
    llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock* check_subtype = llvm::BasicBlock::Create(ctx_.context(), "is_tensor.check_subtype", func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "is_tensor.merge", func);

    // If HEAP_PTR, check subtype; otherwise return false
    ctx_.builder().CreateCondBr(is_heap_ptr, check_subtype, merge_block);

    // Check subtype in header
    ctx_.builder().SetInsertPoint(check_subtype);
    llvm::Value* is_tensor_subtype = checkHeapSubtype(tagged_val, HEAP_SUBTYPE_TENSOR);
    ctx_.builder().CreateBr(merge_block);
    llvm::BasicBlock* subtype_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.int1Type(), 2, "is_tensor.result");
    result->addIncoming(llvm::ConstantInt::getFalse(ctx_.context()), current_block);
    result->addIncoming(is_tensor_subtype, subtype_exit);

    return result;
}

/**
 * @brief Emit an i1 IR check for whether @p tagged_val is a closure:
 *        CALLABLE type whose object header subtype equals
 *        CALLABLE_SUBTYPE_CLOSURE. Uses branching control flow so the
 *        header is only read when the value is actually CALLABLE.
 */
llvm::Value* TaggedValueCodegen::isClosure(llvm::Value* tagged_val) {
    // M1 CONSOLIDATION FIX: Check CALLABLE type and CALLABLE_SUBTYPE_CLOSURE subtype
    // Closures are CALLABLE with header subtype == CALLABLE_SUBTYPE_CLOSURE (0)
    // Must use control flow to avoid reading header for non-CALLABLE values.

    llvm::Value* type_tag = getType(tagged_val);
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();

    // Check if CALLABLE type
    llvm::Value* is_callable = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));

    // Create blocks for subtype check
    llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock* check_subtype = llvm::BasicBlock::Create(ctx_.context(), "is_closure.check_subtype", func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "is_closure.merge", func);

    // If CALLABLE, check subtype; otherwise return false
    ctx_.builder().CreateCondBr(is_callable, check_subtype, merge_block);

    // Check subtype in header
    ctx_.builder().SetInsertPoint(check_subtype);
    llvm::Value* is_closure_subtype = checkCallableSubtype(tagged_val, CALLABLE_SUBTYPE_CLOSURE);
    ctx_.builder().CreateBr(merge_block);
    llvm::BasicBlock* subtype_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.int1Type(), 2, "is_closure.result");
    result->addIncoming(llvm::ConstantInt::getFalse(ctx_.context()), current_block);
    result->addIncoming(is_closure_subtype, subtype_exit);

    return result;
}

/** @brief Emit an i1 IR check for whether @p tagged_val is a preserved lambda S-expression: CALLABLE with header subtype CALLABLE_SUBTYPE_LAMBDA_SEXPR (see isClosure()). */
llvm::Value* TaggedValueCodegen::isLambdaSexpr(llvm::Value* tagged_val) {
    // M1 CONSOLIDATION FIX: Check CALLABLE type and CALLABLE_SUBTYPE_LAMBDA_SEXPR subtype
    // Lambda s-exprs are CALLABLE with header subtype == CALLABLE_SUBTYPE_LAMBDA_SEXPR (1)
    // Must use control flow to avoid reading header for non-CALLABLE values.

    llvm::Value* type_tag = getType(tagged_val);
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();

    // Check if CALLABLE type
    llvm::Value* is_callable = ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));

    // Create blocks for subtype check
    llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock* check_subtype = llvm::BasicBlock::Create(ctx_.context(), "is_lambda.check_subtype", func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "is_lambda.merge", func);

    // If CALLABLE, check subtype; otherwise return false
    ctx_.builder().CreateCondBr(is_callable, check_subtype, merge_block);

    // Check subtype in header
    ctx_.builder().SetInsertPoint(check_subtype);
    llvm::Value* is_lambda_subtype = checkCallableSubtype(tagged_val, CALLABLE_SUBTYPE_LAMBDA_SEXPR);
    ctx_.builder().CreateBr(merge_block);
    llvm::BasicBlock* subtype_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.int1Type(), 2, "is_lambda.result");
    result->addIncoming(llvm::ConstantInt::getFalse(ctx_.context()), current_block);
    result->addIncoming(is_lambda_subtype, subtype_exit);

    return result;
}

/** @brief Emit an i1 IR check for whether @p tagged_val's type tag is the consolidated HEAP_PTR type (any heap data object). */
llvm::Value* TaggedValueCodegen::isHeapPtr(llvm::Value* tagged_val) {
    // M1 CONSOLIDATION FIX: All heap data objects use HEAP_PTR type (8)
    // Subtypes (cons, string, vector, tensor, hash, exception) are in the object header
    llvm::Value* type_tag = getType(tagged_val);
    return ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
}

/** @brief Emit an i1 IR check for whether @p tagged_val's type tag is the consolidated CALLABLE type (any callable object). */
llvm::Value* TaggedValueCodegen::isCallable(llvm::Value* tagged_val) {
    // M1 CONSOLIDATION FIX: All callable objects use CALLABLE type (9)
    // Subtypes (closure, lambda_sexpr, ad_node) are in the object header
    llvm::Value* type_tag = getType(tagged_val);
    return ctx_.builder().CreateICmpEQ(
        type_tag, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
}

/** @brief Emit an i1 IR check for `(exact? and integer-typed)`: base type (via getBaseType()) equals INT64. */
llvm::Value* TaggedValueCodegen::isInt64(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);
    // Use getBaseType() to properly handle legacy types (>=32) and exactness flags
    llvm::Value* base_type = getBaseType(type_tag);

    return ctx_.builder().CreateICmpEQ(
        base_type, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
}

/** @brief Emit an i1 IR check: base type (via getBaseType()) equals DOUBLE. */
llvm::Value* TaggedValueCodegen::isDouble(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);
    // Use getBaseType() to properly handle legacy types (>=32) and exactness flags
    llvm::Value* base_type = getBaseType(type_tag);

    return ctx_.builder().CreateICmpEQ(
        base_type, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
}

/** @brief Emit an i1 IR check: base type is INT64 or DOUBLE. */
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

/** @brief Emit an i1 IR check: base type (via getBaseType()) equals BOOL. */
llvm::Value* TaggedValueCodegen::isBool(llvm::Value* tagged_val) {
    llvm::Value* type_tag = getType(tagged_val);
    // Use getBaseType() to properly handle legacy types (>=32) and exactness flags
    llvm::Value* base_type = getBaseType(type_tag);

    return ctx_.builder().CreateICmpEQ(
        base_type, llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_BOOL));
}

/**
 * @brief Normalize a raw type tag to its base type for comparison,
 *        masking off exactness flag bits for immediate types (0-7) but
 *        leaving consolidated (8-9), multimedia (16-19), and legacy (32+)
 *        type tags unmasked (masking those would corrupt them — e.g.
 *        legacy CONS_PTR=32 masked with 0x0F becomes 0/NULL).
 */
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

// ════════════════════════════════════════════════════════════════════════════
// PRIVATE HELPERS FOR CONSOLIDATED TYPE CHECKING
// ════════════════════════════════════════════════════════════════════════════

llvm::Value* TaggedValueCodegen::getSubtypeFromHeader(llvm::Value* ptr_val) {
    // Object header is at (data_ptr - 8). Subtype is at offset 0 of header.
    // Header layout:
    //   offset 0: uint8_t subtype
    //   offset 1: uint8_t flags
    //   offset 2: uint16_t ref_count
    //   offset 4: uint32_t size

    // Ensure we have a pointer type
    llvm::Value* data_ptr = ptr_val;
    if (ptr_val->getType()->isIntegerTy(64)) {
        data_ptr = ctx_.builder().CreateIntToPtr(ptr_val, ctx_.ptrType(), "data_ptr");
    }

    // Compute header address: data_ptr - 8 bytes
    llvm::Value* header_ptr = ctx_.builder().CreateGEP(
        ctx_.int8Type(),
        data_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), -8),
        "header_ptr");

    // Load subtype byte from offset 0 of header
    llvm::Value* subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), header_ptr, "subtype");
    return subtype;
}

/** @brief Unpack @p tagged_val's pointer and compare its object-header subtype byte against @p expected_subtype. */
llvm::Value* TaggedValueCodegen::checkHeapSubtype(llvm::Value* tagged_val, uint8_t expected_subtype) {
    // Extract pointer from tagged value and check subtype in header
    llvm::Value* ptr_val = unpackInt64(tagged_val);
    llvm::Value* subtype = getSubtypeFromHeader(ptr_val);

    return ctx_.builder().CreateICmpEQ(
        subtype,
        llvm::ConstantInt::get(ctx_.int8Type(), expected_subtype),
        "subtype_match");
}

/** @brief Same as checkHeapSubtype(), for CALLABLE-typed values (both read the header at ptr-8). */
llvm::Value* TaggedValueCodegen::checkCallableSubtype(llvm::Value* tagged_val, uint8_t expected_subtype) {
    // Extract pointer from tagged value and check subtype in header
    // Same implementation as checkHeapSubtype - both use object header at ptr-8
    llvm::Value* ptr_val = unpackInt64(tagged_val);
    llvm::Value* subtype = getSubtypeFromHeader(ptr_val);

    return ctx_.builder().CreateICmpEQ(
        subtype,
        llvm::ConstantInt::get(ctx_.int8Type(), expected_subtype),
        "callable_subtype_match");
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
