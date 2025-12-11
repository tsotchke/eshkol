/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * CollectionCodegen implementation
 *
 * Note: The complex collection implementations remain in llvm_codegen.cpp
 * for now due to dependencies on AST codegen methods. This module provides
 * the allocConsCell helper which is self-contained.
 */

#include <eshkol/backend/collection_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>
#include <vector>

namespace eshkol {

CollectionCodegen::CollectionCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem)
    : ctx_(ctx)
    , tagged_(tagged)
    , mem_(mem) {
    eshkol_debug("CollectionCodegen initialized");
}

llvm::Value* CollectionCodegen::allocConsCell(llvm::Value* car_val, llvm::Value* cdr_val) {
    // Use arena-based tagged cons cell allocation WITH object header (consolidated type format).
    // This allocator prepends an eshkol_object_header_t with subtype HEAP_SUBTYPE_CONS.
    llvm::Function* alloc_func = mem_.getArenaAllocateConsWithHeader();
    if (!alloc_func) {
        eshkol_warn("arena_allocate_cons_with_header not available");
        return tagged_.packNull();
    }

    llvm::Function* set_func = mem_.getTaggedConsSetTaggedValue();
    if (!set_func) {
        eshkol_warn("arena_tagged_cons_set_tagged_value not available");
        return tagged_.packNull();
    }

    // Get global arena pointer (note: it's named __global_arena)
    llvm::GlobalVariable* arena_global = ctx_.module().getNamedGlobal("__global_arena");
    if (!arena_global) {
        eshkol_warn("__global_arena not found");
        return tagged_.packNull();
    }

    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), arena_global, "arena");

    // Allocate tagged cons cell with object header (takes only arena pointer).
    // Returns pointer to cons cell data; header is at (ptr - 8).
    llvm::Value* cons_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr}, "cons_cell");

    // Create allocas at function entry to ensure dominance
    llvm::IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    if (func && !func->empty()) {
        llvm::BasicBlock& entry = func->getEntryBlock();
        ctx_.builder().SetInsertPoint(&entry, entry.begin());
    }

    // Create pointers to tagged values for passing by reference
    llvm::Value* car_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "car_tagged_ptr");
    llvm::Value* cdr_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "cdr_tagged_ptr");

    // Restore insertion point for stores and calls
    ctx_.builder().restoreIP(saved_ip);

    ctx_.builder().CreateStore(car_val, car_ptr);
    ctx_.builder().CreateStore(cdr_val, cdr_ptr);

    // Set car and cdr using tagged_cons_set_tagged_value(cons_ptr, is_cdr, value_ptr)
    llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 1);
    ctx_.builder().CreateCall(set_func, {cons_ptr, is_car, car_ptr});
    ctx_.builder().CreateCall(set_func, {cons_ptr, is_cdr, cdr_ptr});

    // Return pointer to cons cell as int64, packed with HEAP_PTR type (consolidated format).
    // The object header contains HEAP_SUBTYPE_CONS to identify this as a cons cell.
    llvm::Value* cons_int = ctx_.builder().CreatePtrToInt(cons_ptr, ctx_.int64Type());
    return tagged_.packHeapPtr(cons_int);
}

// Note: The following implementations are complex and depend on:
// - codegenTypedAST, typedValueToTaggedValue (AST code generation)
// - S-expression building for proper display
//
// These implementations remain in llvm_codegen.cpp until those modules are extracted.

llvm::Value* CollectionCodegen::cons(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("CollectionCodegen::cons - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 2) {
        eshkol_warn("cons requires exactly 2 arguments");
        return nullptr;
    }

    // Generate car and cdr with type information via callbacks
    void* car_typed = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    void* cdr_typed = codegen_typed_ast_callback_(&op->call_op.variables[1], callback_context_);

    if (!car_typed || !cdr_typed) return nullptr;

    // Convert typed values to tagged values
    llvm::Value* car_tagged = typed_to_tagged_callback_(car_typed, callback_context_);
    llvm::Value* cdr_tagged = typed_to_tagged_callback_(cdr_typed, callback_context_);

    if (!car_tagged || !cdr_tagged) return nullptr;

    // Use existing allocConsCell helper
    return allocConsCell(car_tagged, cdr_tagged);
}

llvm::Value* CollectionCodegen::car(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("CollectionCodegen::car - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("car requires exactly 1 argument");
        return nullptr;
    }

    // Get current function before argument codegen (which might create new blocks)
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Generate code for argument via callback
    // Note: The argument might itself contain car/cdr calls that create blocks
    llvm::Value* pair_val = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!pair_val) return nullptr;

    // Create a continuation block to ensure we don't add instructions to
    // any block created by the argument codegen
    llvm::BasicBlock* car_start = llvm::BasicBlock::Create(ctx_.context(), "car_start", current_func);
    ctx_.builder().CreateBr(car_start);
    ctx_.builder().SetInsertPoint(car_start);

    // VECTOR/TENSOR SUPPORT: Check if input is a vector or tensor type
    // With consolidated types, CONS/VECTOR/TENSOR all use HEAP_PTR - must check subtype in header
    if (pair_val->getType() == ctx_.taggedValueType()) {
        llvm::Value* input_type = tagged_.getType(pair_val);
        llvm::Value* input_base_type = tagged_.getBaseType(input_type);

        // First check if it's a HEAP_PTR type
        llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(input_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

        // If HEAP_PTR, we need to check the subtype in the object header
        // Get pointer to the object
        llvm::Value* obj_ptr_int = tagged_.unpackInt64(pair_val);
        llvm::Value* obj_ptr = ctx_.builder().CreateIntToPtr(obj_ptr_int, ctx_.ptrType());

        // Read subtype from object header at ptr-8
        llvm::Value* header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), obj_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), -8));
        llvm::Value* subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), header_ptr, "subtype");

        // Check subtypes - HEAP_SUBTYPE_VECTOR=2, HEAP_SUBTYPE_TENSOR=3, HEAP_SUBTYPE_CONS=0
        llvm::Value* is_vector_subtype = ctx_.builder().CreateICmpEQ(subtype,
            llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
        llvm::Value* is_tensor_subtype = ctx_.builder().CreateICmpEQ(subtype,
            llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));

        // It's a vector/tensor type if HEAP_PTR AND (subtype==VECTOR OR subtype==TENSOR)
        llvm::Value* is_vector_or_tensor = ctx_.builder().CreateOr(is_vector_subtype, is_tensor_subtype);
        llvm::Value* is_vector_type = ctx_.builder().CreateAnd(is_heap_ptr, is_vector_or_tensor);

        llvm::BasicBlock* vector_block = llvm::BasicBlock::Create(ctx_.context(), "car_vector", current_func);
        llvm::BasicBlock* list_block = llvm::BasicBlock::Create(ctx_.context(), "car_list", current_func);
        llvm::BasicBlock* car_final = llvm::BasicBlock::Create(ctx_.context(), "car_final", current_func);

        ctx_.builder().CreateCondBr(is_vector_type, vector_block, list_block);

        // VECTOR/TENSOR: Extract element 0
        ctx_.builder().SetInsertPoint(vector_block);

        llvm::BasicBlock* scheme_vec_block = llvm::BasicBlock::Create(ctx_.context(), "car_scheme_vec", current_func);
        llvm::BasicBlock* tensor_block = llvm::BasicBlock::Create(ctx_.context(), "car_tensor", current_func);
        llvm::BasicBlock* vector_merge = llvm::BasicBlock::Create(ctx_.context(), "car_vector_merge", current_func);

        ctx_.builder().CreateCondBr(is_vector_subtype, scheme_vec_block, tensor_block);

        // Scheme vector: structure is [length (8 bytes), elem0, elem1, ...]
        // Layout: vec_ptr+0 = length, vec_ptr+8 = elem0, vec_ptr+24 = elem1, etc.
        ctx_.builder().SetInsertPoint(scheme_vec_block);
        llvm::Value* vec_ptr_int = tagged_.unpackInt64(pair_val);
        llvm::Value* vec_ptr = ctx_.builder().CreateIntToPtr(vec_ptr_int, ctx_.ptrType());
        // Skip 8-byte length field to get to elements base
        llvm::Value* elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), vec_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), 8));
        llvm::Value* elem0_ptr = ctx_.builder().CreatePointerCast(elem_base, ctx_.ptrType());
        llvm::Value* scheme_elem = ctx_.builder().CreateLoad(ctx_.taggedValueType(), elem0_ptr);
        ctx_.builder().CreateBr(vector_merge);
        llvm::BasicBlock* scheme_vec_exit = ctx_.builder().GetInsertBlock();

        // Tensor: structure has elements array - AD-aware like vref
        ctx_.builder().SetInsertPoint(tensor_block);
        llvm::Value* tensor_ptr_int = tagged_.unpackInt64(pair_val);
        llvm::Value* tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
        llvm::Value* elems_ptr_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 2);
        llvm::Value* elems_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), elems_ptr_ptr);

        // AD-AWARE: Load element as int64 (could be double OR AD node pointer)
        llvm::Value* elem_as_int64 = ctx_.builder().CreateLoad(ctx_.int64Type(), elems_ptr);

        // Check AD mode and use IEEE754 exponent check to distinguish types
        llvm::Value* in_ad_mode = ctx_.builder().CreateLoad(ctx_.int1Type(), ctx_.adModeActive());

        llvm::BasicBlock* car_tensor_ad = llvm::BasicBlock::Create(ctx_.context(), "car_tensor_ad", current_func);
        llvm::BasicBlock* car_tensor_normal = llvm::BasicBlock::Create(ctx_.context(), "car_tensor_normal", current_func);
        llvm::BasicBlock* car_tensor_merge = llvm::BasicBlock::Create(ctx_.context(), "car_tensor_merge", current_func);

        ctx_.builder().CreateCondBr(in_ad_mode, car_tensor_ad, car_tensor_normal);

        // AD mode: check if value is int, double (captured), or AD node pointer
        ctx_.builder().SetInsertPoint(car_tensor_ad);
        llvm::Value* is_small_ad = ctx_.builder().CreateICmpULT(elem_as_int64,
            llvm::ConstantInt::get(ctx_.int64Type(), 1000));

        llvm::BasicBlock* car_ad_small = llvm::BasicBlock::Create(ctx_.context(), "car_ad_small", current_func);
        llvm::BasicBlock* car_ad_large = llvm::BasicBlock::Create(ctx_.context(), "car_ad_large", current_func);
        ctx_.builder().CreateCondBr(is_small_ad, car_ad_small, car_ad_large);

        ctx_.builder().SetInsertPoint(car_ad_small);
        llvm::Value* car_ad_int = tagged_.packInt64(elem_as_int64, true);
        ctx_.builder().CreateBr(car_tensor_merge);
        llvm::BasicBlock* car_ad_small_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(car_ad_large);
        llvm::Value* ad_exp_mask = llvm::ConstantInt::get(ctx_.int64Type(), 0x7FF0000000000000ULL);
        llvm::Value* ad_exp_bits = ctx_.builder().CreateAnd(elem_as_int64, ad_exp_mask);
        llvm::Value* ad_has_exp = ctx_.builder().CreateICmpNE(ad_exp_bits,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::BasicBlock* car_ad_double = llvm::BasicBlock::Create(ctx_.context(), "car_ad_double", current_func);
        llvm::BasicBlock* car_ad_node = llvm::BasicBlock::Create(ctx_.context(), "car_ad_node", current_func);
        ctx_.builder().CreateCondBr(ad_has_exp, car_ad_double, car_ad_node);

        ctx_.builder().SetInsertPoint(car_ad_double);
        llvm::Value* car_ad_d = ctx_.builder().CreateBitCast(elem_as_int64, ctx_.doubleType());
        llvm::Value* car_ad_double_tagged = tagged_.packDouble(car_ad_d);
        ctx_.builder().CreateBr(car_tensor_merge);
        llvm::BasicBlock* car_ad_double_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(car_ad_node);
        llvm::Value* car_ad_ptr = ctx_.builder().CreateIntToPtr(elem_as_int64, ctx_.ptrType());
        llvm::Value* car_ad_node_tagged = tagged_.packCallable(car_ad_ptr);
        ctx_.builder().CreateBr(car_tensor_merge);
        llvm::BasicBlock* car_ad_node_exit = ctx_.builder().GetInsertBlock();

        // Normal mode: check int vs double
        ctx_.builder().SetInsertPoint(car_tensor_normal);
        llvm::Value* is_small_normal = ctx_.builder().CreateICmpULT(elem_as_int64,
            llvm::ConstantInt::get(ctx_.int64Type(), 1000));

        llvm::BasicBlock* car_normal_int = llvm::BasicBlock::Create(ctx_.context(), "car_normal_int", current_func);
        llvm::BasicBlock* car_normal_double = llvm::BasicBlock::Create(ctx_.context(), "car_normal_double", current_func);
        ctx_.builder().CreateCondBr(is_small_normal, car_normal_int, car_normal_double);

        ctx_.builder().SetInsertPoint(car_normal_int);
        llvm::Value* car_normal_int_tagged = tagged_.packInt64(elem_as_int64, true);
        ctx_.builder().CreateBr(car_tensor_merge);
        llvm::BasicBlock* car_normal_int_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(car_normal_double);
        llvm::Value* car_normal_d = ctx_.builder().CreateBitCast(elem_as_int64, ctx_.doubleType());
        llvm::Value* car_normal_double_tagged = tagged_.packDouble(car_normal_d);
        ctx_.builder().CreateBr(car_tensor_merge);
        llvm::BasicBlock* car_normal_double_exit = ctx_.builder().GetInsertBlock();

        // Merge tensor results
        ctx_.builder().SetInsertPoint(car_tensor_merge);
        llvm::PHINode* tensor_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 5);
        tensor_phi->addIncoming(car_ad_int, car_ad_small_exit);
        tensor_phi->addIncoming(car_ad_double_tagged, car_ad_double_exit);
        tensor_phi->addIncoming(car_ad_node_tagged, car_ad_node_exit);
        tensor_phi->addIncoming(car_normal_int_tagged, car_normal_int_exit);
        tensor_phi->addIncoming(car_normal_double_tagged, car_normal_double_exit);

        llvm::Value* tensor_elem = tensor_phi;
        ctx_.builder().CreateBr(vector_merge);
        llvm::BasicBlock* tensor_exit = ctx_.builder().GetInsertBlock();

        // Merge vector results
        ctx_.builder().SetInsertPoint(vector_merge);
        llvm::PHINode* vector_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
        vector_phi->addIncoming(scheme_elem, scheme_vec_exit);
        vector_phi->addIncoming(tensor_elem, tensor_exit);
        ctx_.builder().CreateBr(car_final);
        llvm::BasicBlock* vector_merge_exit = ctx_.builder().GetInsertBlock();

        // LIST: Fall through to original cons cell handling
        ctx_.builder().SetInsertPoint(list_block);
        llvm::Value* pair_int_safe = tagged_.safeExtractInt64(pair_val);

        // SAFETY CHECK: Ensure pair_int is not null (0) before dereferencing
        llvm::Value* is_null = ctx_.builder().CreateICmpEQ(pair_int_safe, llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::BasicBlock* null_block = llvm::BasicBlock::Create(ctx_.context(), "car_null", current_func);
        llvm::BasicBlock* valid_block = llvm::BasicBlock::Create(ctx_.context(), "car_valid", current_func);

        ctx_.builder().CreateCondBr(is_null, null_block, valid_block);

        // Null block: return 0 (null) for safety
        ctx_.builder().SetInsertPoint(null_block);
        llvm::Value* null_result = llvm::ConstantInt::get(ctx_.int64Type(), 0);
        llvm::Value* null_tagged = tagged_.packInt64(null_result, true);
        ctx_.builder().CreateBr(car_final);
        llvm::BasicBlock* null_exit = ctx_.builder().GetInsertBlock();

        // Valid block: use TAGGED cons cell to extract car with proper type
        ctx_.builder().SetInsertPoint(valid_block);

        llvm::Value* cons_ptr = ctx_.builder().CreateIntToPtr(pair_int_safe, ctx_.ptrType());

        // Get car type using arena_tagged_cons_get_type(cell, false)
        llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 0); // false = car
        llvm::Value* car_type = ctx_.builder().CreateCall(mem_.getTaggedConsGetType(), {cons_ptr, is_cdr});

        // Get base type (properly handles legacy types >= 32)
        llvm::Value* car_base_type = tagged_.getBaseType(car_type);

        // Type checks - handle both legacy (CONS_PTR) and consolidated (HEAP_PTR) formats
        llvm::Value* car_is_null_type = ctx_.builder().CreateICmpEQ(car_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL));
        llvm::Value* car_is_double = ctx_.builder().CreateICmpEQ(car_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* car_is_cons_legacy = ctx_.builder().CreateICmpEQ(car_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        llvm::Value* car_is_heap_ptr = ctx_.builder().CreateICmpEQ(car_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        llvm::Value* car_is_cons_ptr = ctx_.builder().CreateOr(car_is_cons_legacy, car_is_heap_ptr);
        llvm::Value* car_is_string_ptr = ctx_.builder().CreateICmpEQ(car_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        llvm::Value* car_is_lambda_sexpr = ctx_.builder().CreateICmpEQ(car_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
        // Check for both legacy CLOSURE_PTR and new CALLABLE type
        llvm::Value* car_is_closure_legacy = ctx_.builder().CreateICmpEQ(car_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
        llvm::Value* car_is_callable = ctx_.builder().CreateICmpEQ(car_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
        llvm::Value* car_is_closure_ptr = ctx_.builder().CreateOr(car_is_closure_legacy, car_is_callable);
        llvm::Value* car_is_bool = ctx_.builder().CreateICmpEQ(car_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_BOOL));
        llvm::Value* car_is_char = ctx_.builder().CreateICmpEQ(car_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CHAR));
        llvm::Value* car_is_hash_ptr = ctx_.builder().CreateICmpEQ(car_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

        // Create blocks for each type
        llvm::BasicBlock* null_car = llvm::BasicBlock::Create(ctx_.context(), "car_extract_null", current_func);
        llvm::BasicBlock* check_double = llvm::BasicBlock::Create(ctx_.context(), "car_check_double", current_func);
        llvm::BasicBlock* double_car = llvm::BasicBlock::Create(ctx_.context(), "car_extract_double", current_func);
        llvm::BasicBlock* check_cons_ptr = llvm::BasicBlock::Create(ctx_.context(), "car_check_cons_ptr", current_func);
        llvm::BasicBlock* cons_ptr_car = llvm::BasicBlock::Create(ctx_.context(), "car_extract_cons_ptr", current_func);
        llvm::BasicBlock* check_string_ptr = llvm::BasicBlock::Create(ctx_.context(), "car_check_string_ptr", current_func);
        llvm::BasicBlock* string_ptr_car = llvm::BasicBlock::Create(ctx_.context(), "car_extract_string_ptr", current_func);
        llvm::BasicBlock* check_lambda = llvm::BasicBlock::Create(ctx_.context(), "car_check_lambda", current_func);
        llvm::BasicBlock* lambda_car = llvm::BasicBlock::Create(ctx_.context(), "car_extract_lambda", current_func);
        llvm::BasicBlock* check_closure = llvm::BasicBlock::Create(ctx_.context(), "car_check_closure", current_func);
        llvm::BasicBlock* closure_car = llvm::BasicBlock::Create(ctx_.context(), "car_extract_closure", current_func);
        llvm::BasicBlock* check_bool = llvm::BasicBlock::Create(ctx_.context(), "car_check_bool", current_func);
        llvm::BasicBlock* bool_car = llvm::BasicBlock::Create(ctx_.context(), "car_extract_bool", current_func);
        llvm::BasicBlock* check_char = llvm::BasicBlock::Create(ctx_.context(), "car_check_char", current_func);
        llvm::BasicBlock* char_car = llvm::BasicBlock::Create(ctx_.context(), "car_extract_char", current_func);
        llvm::BasicBlock* check_hash = llvm::BasicBlock::Create(ctx_.context(), "car_check_hash", current_func);
        llvm::BasicBlock* hash_ptr_car = llvm::BasicBlock::Create(ctx_.context(), "car_extract_hash", current_func);
        llvm::BasicBlock* int_car = llvm::BasicBlock::Create(ctx_.context(), "car_extract_int", current_func);
        llvm::BasicBlock* merge_car = llvm::BasicBlock::Create(ctx_.context(), "car_merge", current_func);

        // Check for null first
        ctx_.builder().CreateCondBr(car_is_null_type, null_car, check_double);

        // Return NULL tagged value
        ctx_.builder().SetInsertPoint(null_car);
        llvm::Value* tagged_null_car = tagged_.packNull();
        ctx_.builder().CreateBr(merge_car);
        llvm::BasicBlock* null_car_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(check_double);
        ctx_.builder().CreateCondBr(car_is_double, double_car, check_cons_ptr);

        ctx_.builder().SetInsertPoint(double_car);
        llvm::Value* car_double = ctx_.builder().CreateCall(mem_.getTaggedConsGetDouble(), {cons_ptr, is_cdr});
        llvm::Value* tagged_double = tagged_.packDouble(car_double);
        ctx_.builder().CreateBr(merge_car);
        llvm::BasicBlock* double_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(check_cons_ptr);
        ctx_.builder().CreateCondBr(car_is_cons_ptr, cons_ptr_car, check_string_ptr);

        ctx_.builder().SetInsertPoint(cons_ptr_car);
        llvm::Value* car_cons = ctx_.builder().CreateCall(mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr});
        // Pack as HEAP_PTR (consolidated format) - type checks handle both formats
        llvm::Value* tagged_cons = tagged_.packHeapPtr(ctx_.builder().CreateIntToPtr(car_cons, ctx_.ptrType()));
        ctx_.builder().CreateBr(merge_car);
        llvm::BasicBlock* cons_exit = ctx_.builder().GetInsertBlock();

        // Handle STRING_PTR for symbols
        ctx_.builder().SetInsertPoint(check_string_ptr);
        ctx_.builder().CreateCondBr(car_is_string_ptr, string_ptr_car, check_lambda);

        ctx_.builder().SetInsertPoint(string_ptr_car);
        llvm::Value* car_string = ctx_.builder().CreateCall(mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr});
        llvm::Value* tagged_string = tagged_.packPtr(ctx_.builder().CreateIntToPtr(car_string, ctx_.ptrType()), ESHKOL_VALUE_HEAP_PTR);
        ctx_.builder().CreateBr(merge_car);
        llvm::BasicBlock* string_exit = ctx_.builder().GetInsertBlock();

        // Handle LAMBDA_SEXPR
        ctx_.builder().SetInsertPoint(check_lambda);
        ctx_.builder().CreateCondBr(car_is_lambda_sexpr, lambda_car, check_closure);

        ctx_.builder().SetInsertPoint(lambda_car);
        llvm::Value* lambda_type_val = ctx_.builder().CreateCall(mem_.getTaggedConsGetType(), {cons_ptr, is_cdr});
        llvm::Value* lambda_flags_val = ctx_.builder().CreateCall(mem_.getTaggedConsGetFlags(), {cons_ptr, is_cdr});
        llvm::Value* lambda_ptr_val = ctx_.builder().CreateCall(mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr});
        llvm::Value* tagged_lambda = tagged_.packPtrWithFlags(
            ctx_.builder().CreateIntToPtr(lambda_ptr_val, ctx_.ptrType()),
            lambda_type_val, lambda_flags_val);
        ctx_.builder().CreateBr(merge_car);
        llvm::BasicBlock* lambda_exit = ctx_.builder().GetInsertBlock();

        // Handle CLOSURE_PTR
        ctx_.builder().SetInsertPoint(check_closure);
        ctx_.builder().CreateCondBr(car_is_closure_ptr, closure_car, check_bool);

        ctx_.builder().SetInsertPoint(closure_car);
        llvm::Value* closure_type_val = ctx_.builder().CreateCall(mem_.getTaggedConsGetType(), {cons_ptr, is_cdr});
        llvm::Value* closure_flags_val = ctx_.builder().CreateCall(mem_.getTaggedConsGetFlags(), {cons_ptr, is_cdr});
        llvm::Value* closure_ptr_val = ctx_.builder().CreateCall(mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr});
        llvm::Value* tagged_closure = tagged_.packPtrWithFlags(
            ctx_.builder().CreateIntToPtr(closure_ptr_val, ctx_.ptrType()),
            closure_type_val, closure_flags_val);
        ctx_.builder().CreateBr(merge_car);
        llvm::BasicBlock* closure_exit = ctx_.builder().GetInsertBlock();

        // Handle boolean values
        ctx_.builder().SetInsertPoint(check_bool);
        ctx_.builder().CreateCondBr(car_is_bool, bool_car, check_char);

        ctx_.builder().SetInsertPoint(bool_car);
        llvm::Value* car_bool_int = ctx_.builder().CreateCall(mem_.getTaggedConsGetInt64(), {cons_ptr, is_cdr});
        llvm::Value* car_bool_i1 = ctx_.builder().CreateICmpNE(car_bool_int, llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* tagged_bool = tagged_.packBool(car_bool_i1);
        ctx_.builder().CreateBr(merge_car);
        llvm::BasicBlock* bool_exit = ctx_.builder().GetInsertBlock();

        // Handle CHAR values
        ctx_.builder().SetInsertPoint(check_char);
        ctx_.builder().CreateCondBr(car_is_char, char_car, check_hash);

        ctx_.builder().SetInsertPoint(char_car);
        llvm::Value* car_char_int = ctx_.builder().CreateCall(mem_.getTaggedConsGetInt64(), {cons_ptr, is_cdr});
        llvm::Value* tagged_char = tagged_.packChar(car_char_int);
        ctx_.builder().CreateBr(merge_car);
        llvm::BasicBlock* char_exit = ctx_.builder().GetInsertBlock();

        // Handle HASH_PTR for hash tables (legacy HASH_PTR - repacks as HEAP_PTR)
        ctx_.builder().SetInsertPoint(check_hash);
        ctx_.builder().CreateCondBr(car_is_hash_ptr, hash_ptr_car, int_car);

        ctx_.builder().SetInsertPoint(hash_ptr_car);
        llvm::Value* car_hash = ctx_.builder().CreateCall(mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr});
        // Repack as HEAP_PTR (consolidated format) - subtype is in object header
        llvm::Value* tagged_hash = tagged_.packHeapPtr(ctx_.builder().CreateIntToPtr(car_hash, ctx_.ptrType()));
        ctx_.builder().CreateBr(merge_car);
        llvm::BasicBlock* hash_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(int_car);
        llvm::Value* car_int64 = ctx_.builder().CreateCall(mem_.getTaggedConsGetInt64(), {cons_ptr, is_cdr});
        llvm::Value* tagged_int64 = tagged_.packInt64(car_int64, true);
        ctx_.builder().CreateBr(merge_car);
        llvm::BasicBlock* int_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(merge_car);
        llvm::PHINode* car_tagged_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 10);
        car_tagged_phi->addIncoming(tagged_null_car, null_car_exit);
        car_tagged_phi->addIncoming(tagged_double, double_exit);
        car_tagged_phi->addIncoming(tagged_cons, cons_exit);
        car_tagged_phi->addIncoming(tagged_string, string_exit);
        car_tagged_phi->addIncoming(tagged_lambda, lambda_exit);
        car_tagged_phi->addIncoming(tagged_closure, closure_exit);
        car_tagged_phi->addIncoming(tagged_bool, bool_exit);
        car_tagged_phi->addIncoming(tagged_char, char_exit);
        car_tagged_phi->addIncoming(tagged_hash, hash_exit);
        car_tagged_phi->addIncoming(tagged_int64, int_exit);
        ctx_.builder().CreateBr(car_final);
        llvm::BasicBlock* merge_exit = ctx_.builder().GetInsertBlock();

        // Final merge of all paths
        ctx_.builder().SetInsertPoint(car_final);
        llvm::PHINode* final_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3);
        final_phi->addIncoming(vector_phi, vector_merge_exit);
        final_phi->addIncoming(null_tagged, null_exit);
        final_phi->addIncoming(car_tagged_phi, merge_exit);

        // Create continuation block so car_final stays clean with just the PHI
        llvm::BasicBlock* car_done = llvm::BasicBlock::Create(ctx_.context(), "car_done", current_func);
        ctx_.builder().CreateBr(car_done);
        ctx_.builder().SetInsertPoint(car_done);

        return final_phi;
    }

    // Fallback for non-tagged values
    llvm::Value* pair_int_safe = tagged_.safeExtractInt64(pair_val);
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(pair_int_safe, llvm::ConstantInt::get(ctx_.int64Type(), 0));

    llvm::BasicBlock* null_block = llvm::BasicBlock::Create(ctx_.context(), "car_null", current_func);
    llvm::BasicBlock* valid_block = llvm::BasicBlock::Create(ctx_.context(), "car_valid", current_func);
    llvm::BasicBlock* continue_block = llvm::BasicBlock::Create(ctx_.context(), "car_continue", current_func);

    ctx_.builder().CreateCondBr(is_null, null_block, valid_block);

    ctx_.builder().SetInsertPoint(null_block);
    llvm::Value* null_result = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    llvm::Value* null_tagged = tagged_.packInt64(null_result, true);
    ctx_.builder().CreateBr(continue_block);
    llvm::BasicBlock* null_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(valid_block);
    llvm::Value* cons_ptr = ctx_.builder().CreateIntToPtr(pair_int_safe, ctx_.ptrType());
    llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    llvm::Value* car_int64 = ctx_.builder().CreateCall(mem_.getTaggedConsGetInt64(), {cons_ptr, is_cdr});
    llvm::Value* valid_tagged = tagged_.packInt64(car_int64, true);
    ctx_.builder().CreateBr(continue_block);
    llvm::BasicBlock* valid_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(continue_block);
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
    result_phi->addIncoming(null_tagged, null_exit);
    result_phi->addIncoming(valid_tagged, valid_exit);

    // Create done block to not pollute continue_block
    llvm::BasicBlock* car_fallback_done = llvm::BasicBlock::Create(ctx_.context(), "car_fallback_done", current_func);
    ctx_.builder().CreateBr(car_fallback_done);
    ctx_.builder().SetInsertPoint(car_fallback_done);

    return result_phi;
}

llvm::Value* CollectionCodegen::cdr(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("CollectionCodegen::cdr - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("cdr requires exactly 1 argument");
        return nullptr;
    }

    // Get current function before argument codegen (which might create new blocks)
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Generate code for argument via callback
    // Note: The argument might itself contain cdr calls that create blocks
    llvm::Value* pair_val = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!pair_val) return nullptr;

    // Create a continuation block to ensure we don't add instructions to
    // any block created by the argument codegen
    llvm::BasicBlock* cdr_start = llvm::BasicBlock::Create(ctx_.context(), "cdr_start", current_func);
    ctx_.builder().CreateBr(cdr_start);
    ctx_.builder().SetInsertPoint(cdr_start);

    // VECTOR/TENSOR SUPPORT: Check if input is a vector or tensor type
    // With consolidated types, CONS/VECTOR/TENSOR all use HEAP_PTR - must check subtype in header
    if (pair_val->getType() == ctx_.taggedValueType()) {
        llvm::Value* input_type = tagged_.getType(pair_val);
        llvm::Value* input_base_type = tagged_.getBaseType(input_type);

        // First check if it's a HEAP_PTR type
        llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(input_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

        // If HEAP_PTR, we need to check the subtype in the object header
        llvm::Value* obj_ptr_int = tagged_.unpackInt64(pair_val);
        llvm::Value* obj_ptr = ctx_.builder().CreateIntToPtr(obj_ptr_int, ctx_.ptrType());

        // Read subtype from object header at ptr-8
        llvm::Value* header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), obj_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), -8));
        llvm::Value* subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), header_ptr, "subtype");

        // Check subtypes - HEAP_SUBTYPE_VECTOR=2, HEAP_SUBTYPE_TENSOR=3
        llvm::Value* is_vector_subtype = ctx_.builder().CreateICmpEQ(subtype,
            llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
        llvm::Value* is_tensor_subtype = ctx_.builder().CreateICmpEQ(subtype,
            llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));

        // It's a vector/tensor type if HEAP_PTR AND (subtype==VECTOR OR subtype==TENSOR)
        llvm::Value* is_vector_or_tensor = ctx_.builder().CreateOr(is_vector_subtype, is_tensor_subtype);
        llvm::Value* is_vector_type = ctx_.builder().CreateAnd(is_heap_ptr, is_vector_or_tensor);

        llvm::BasicBlock* vector_block = llvm::BasicBlock::Create(ctx_.context(), "cdr_vector", current_func);
        llvm::BasicBlock* list_block = llvm::BasicBlock::Create(ctx_.context(), "cdr_list", current_func);
        llvm::BasicBlock* cdr_final = llvm::BasicBlock::Create(ctx_.context(), "cdr_final", current_func);

        ctx_.builder().CreateCondBr(is_vector_type, vector_block, list_block);

        // VECTOR/TENSOR: Create new vector with elements 1 through n-1
        ctx_.builder().SetInsertPoint(vector_block);

        llvm::BasicBlock* scheme_vec_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_scheme_vec", current_func);
        llvm::BasicBlock* tensor_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_tensor", current_func);
        llvm::BasicBlock* vector_merge = llvm::BasicBlock::Create(ctx_.context(), "cdr_vector_merge", current_func);

        ctx_.builder().CreateCondBr(is_vector_subtype, scheme_vec_cdr, tensor_cdr);

        // Scheme vector cdr: create new vector with elements 1..n-1
        // Layout: [header (8 bytes)][length (8 bytes)][elem0 (16 bytes)][elem1 (16 bytes)]...
        ctx_.builder().SetInsertPoint(scheme_vec_cdr);
        llvm::Value* vec_ptr_int = tagged_.unpackInt64(pair_val);
        llvm::Value* vec_ptr = ctx_.builder().CreateIntToPtr(vec_ptr_int, ctx_.ptrType());
        llvm::Value* length = ctx_.builder().CreateLoad(ctx_.int64Type(), vec_ptr);
        llvm::Value* new_length = ctx_.builder().CreateSub(length, llvm::ConstantInt::get(ctx_.int64Type(), 1));

        // Allocate new vector with header using arena
        llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        llvm::Value* typed_new_vec = ctx_.builder().CreateCall(
            mem_.getArenaAllocateVectorWithHeader(), {arena_ptr, new_length});

        // Store new length
        ctx_.builder().CreateStore(new_length, typed_new_vec);

        // Get pointers to element bases (after 8-byte length field)
        llvm::Value* src_elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), vec_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), 8));
        llvm::Value* dst_elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), typed_new_vec,
            llvm::ConstantInt::get(ctx_.int64Type(), 8));

        // Copy loop: copy elements 1..n-1 from source to 0..n-2 in destination
        llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "cdr_copy_cond", current_func);
        llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "cdr_copy_body", current_func);
        llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "cdr_copy_done", current_func);

        llvm::Value* copy_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "copy_idx");
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_idx);
        ctx_.builder().CreateBr(copy_cond);

        ctx_.builder().SetInsertPoint(copy_cond);
        llvm::Value* idx = ctx_.builder().CreateLoad(ctx_.int64Type(), copy_idx);
        llvm::Value* idx_less_new_len = ctx_.builder().CreateICmpULT(idx, new_length);
        ctx_.builder().CreateCondBr(idx_less_new_len, copy_body, copy_done);

        ctx_.builder().SetInsertPoint(copy_body);
        // Source: element at index (idx + 1) from original vector
        llvm::Value* src_elem_idx = ctx_.builder().CreateAdd(idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        llvm::Value* src_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), src_elem_base, src_elem_idx);
        llvm::Value* elem = ctx_.builder().CreateLoad(ctx_.taggedValueType(), src_ptr);
        // Destination: element at index idx in new vector
        llvm::Value* dst_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), dst_elem_base, idx);
        ctx_.builder().CreateStore(elem, dst_ptr);
        ctx_.builder().CreateStore(ctx_.builder().CreateAdd(idx, llvm::ConstantInt::get(ctx_.int64Type(), 1)), copy_idx);
        ctx_.builder().CreateBr(copy_cond);

        ctx_.builder().SetInsertPoint(copy_done);
        llvm::Value* scheme_cdr_result = tagged_.packHeapPtr(typed_new_vec);
        ctx_.builder().CreateBr(vector_merge);
        llvm::BasicBlock* scheme_vec_exit = ctx_.builder().GetInsertBlock();

        // Tensor cdr: create new tensor with elements 1..n-1
        ctx_.builder().SetInsertPoint(tensor_cdr);
        llvm::Value* tensor_ptr_int = tagged_.unpackInt64(pair_val);
        llvm::Value* tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
        llvm::Value* dims_ptr_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 0);
        llvm::Value* dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), dims_ptr_ptr);
        llvm::Value* tensor_len = ctx_.builder().CreateLoad(ctx_.int64Type(), dims_ptr);
        llvm::Value* tensor_new_len = ctx_.builder().CreateSub(tensor_len, llvm::ConstantInt::get(ctx_.int64Type(), 1));

        // Get arena for OALR-compliant allocation
        llvm::Value* tensor_arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());

        // Allocate new tensor structure via arena (OALR compliant - no malloc)
        llvm::Value* new_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {tensor_arena_ptr});

        // Allocate dims array via arena
        llvm::Value* new_dims = ctx_.builder().CreateCall(mem_.getArenaAllocate(),
            {tensor_arena_ptr, llvm::ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t))});
        ctx_.builder().CreateStore(tensor_new_len, new_dims);
        ctx_.builder().CreateStore(new_dims, ctx_.builder().CreateStructGEP(ctx_.tensorType(), new_tensor, 0));
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1),
            ctx_.builder().CreateStructGEP(ctx_.tensorType(), new_tensor, 1));
        ctx_.builder().CreateStore(tensor_new_len, ctx_.builder().CreateStructGEP(ctx_.tensorType(), new_tensor, 3));

        // Allocate and copy elements via arena
        llvm::Value* new_elems_size = ctx_.builder().CreateMul(tensor_new_len,
            llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
        llvm::Value* new_elems = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {tensor_arena_ptr, new_elems_size});
        ctx_.builder().CreateStore(new_elems, ctx_.builder().CreateStructGEP(ctx_.tensorType(), new_tensor, 2));

        // Get old elements
        llvm::Value* old_elems_ptr_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 2);
        llvm::Value* old_elems = ctx_.builder().CreateLoad(ctx_.ptrType(), old_elems_ptr_ptr);

        // Copy loop
        llvm::BasicBlock* tcopy_cond = llvm::BasicBlock::Create(ctx_.context(), "tensor_cdr_copy_cond", current_func);
        llvm::BasicBlock* tcopy_body = llvm::BasicBlock::Create(ctx_.context(), "tensor_cdr_copy_body", current_func);
        llvm::BasicBlock* tcopy_done = llvm::BasicBlock::Create(ctx_.context(), "tensor_cdr_copy_done", current_func);

        llvm::Value* tcopy_idx = ctx_.builder().CreateAlloca(ctx_.int64Type());
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), tcopy_idx);
        ctx_.builder().CreateBr(tcopy_cond);

        ctx_.builder().SetInsertPoint(tcopy_cond);
        llvm::Value* tidx = ctx_.builder().CreateLoad(ctx_.int64Type(), tcopy_idx);
        llvm::Value* tidx_less = ctx_.builder().CreateICmpULT(tidx, tensor_new_len);
        ctx_.builder().CreateCondBr(tidx_less, tcopy_body, tcopy_done);

        ctx_.builder().SetInsertPoint(tcopy_body);
        llvm::Value* tsrc_idx = ctx_.builder().CreateAdd(tidx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        llvm::Value* tsrc_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), old_elems, tsrc_idx);
        llvm::Value* telem = ctx_.builder().CreateLoad(ctx_.int64Type(), tsrc_ptr);
        llvm::Value* tdst_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), new_elems, tidx);
        ctx_.builder().CreateStore(telem, tdst_ptr);
        ctx_.builder().CreateStore(ctx_.builder().CreateAdd(tidx, llvm::ConstantInt::get(ctx_.int64Type(), 1)), tcopy_idx);
        ctx_.builder().CreateBr(tcopy_cond);

        ctx_.builder().SetInsertPoint(tcopy_done);
        llvm::Value* new_tensor_int = ctx_.builder().CreatePtrToInt(new_tensor, ctx_.int64Type());
        llvm::Value* tensor_cdr_result = tagged_.packPtr(
            ctx_.builder().CreateIntToPtr(new_tensor_int, ctx_.ptrType()), ESHKOL_VALUE_HEAP_PTR);
        ctx_.builder().CreateBr(vector_merge);
        llvm::BasicBlock* tensor_exit = ctx_.builder().GetInsertBlock();

        // Merge vector results
        ctx_.builder().SetInsertPoint(vector_merge);
        llvm::PHINode* vector_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
        vector_phi->addIncoming(scheme_cdr_result, scheme_vec_exit);
        vector_phi->addIncoming(tensor_cdr_result, tensor_exit);
        ctx_.builder().CreateBr(cdr_final);
        llvm::BasicBlock* vector_merge_exit = ctx_.builder().GetInsertBlock();

        // LIST: Fall through to original cons cell handling
        ctx_.builder().SetInsertPoint(list_block);
        llvm::Value* pair_int_safe = tagged_.safeExtractInt64(pair_val);

        llvm::Value* is_null = ctx_.builder().CreateICmpEQ(pair_int_safe, llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::BasicBlock* null_block = llvm::BasicBlock::Create(ctx_.context(), "cdr_null", current_func);
        llvm::BasicBlock* valid_block = llvm::BasicBlock::Create(ctx_.context(), "cdr_valid", current_func);

        ctx_.builder().CreateCondBr(is_null, null_block, valid_block);

        ctx_.builder().SetInsertPoint(null_block);
        llvm::Value* null_tagged_cdr = tagged_.packNull();
        ctx_.builder().CreateBr(cdr_final);
        llvm::BasicBlock* null_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(valid_block);

        llvm::Value* cons_ptr = ctx_.builder().CreateIntToPtr(pair_int_safe, ctx_.ptrType());

        // Get cdr type (properly handles legacy types >= 32)
        llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 1);
        llvm::Value* cdr_type = ctx_.builder().CreateCall(mem_.getTaggedConsGetType(), {cons_ptr, is_cdr});
        llvm::Value* cdr_base_type = tagged_.getBaseType(cdr_type);

        // Type checks
        llvm::Value* cdr_is_double = ctx_.builder().CreateICmpEQ(cdr_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        // Handle both legacy (CONS_PTR) and consolidated (HEAP_PTR) formats
        llvm::Value* cdr_is_cons_legacy = ctx_.builder().CreateICmpEQ(cdr_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        llvm::Value* cdr_is_heap_ptr = ctx_.builder().CreateICmpEQ(cdr_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        llvm::Value* cdr_is_cons_ptr = ctx_.builder().CreateOr(cdr_is_cons_legacy, cdr_is_heap_ptr);
        llvm::Value* cdr_is_null_type = ctx_.builder().CreateICmpEQ(cdr_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL));
        llvm::Value* cdr_is_string_ptr = ctx_.builder().CreateICmpEQ(cdr_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        llvm::Value* cdr_is_lambda_sexpr = ctx_.builder().CreateICmpEQ(cdr_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
        // Check for both legacy CLOSURE_PTR and new CALLABLE type
        llvm::Value* cdr_is_closure_legacy = ctx_.builder().CreateICmpEQ(cdr_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
        llvm::Value* cdr_is_callable = ctx_.builder().CreateICmpEQ(cdr_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
        llvm::Value* cdr_is_closure_ptr = ctx_.builder().CreateOr(cdr_is_closure_legacy, cdr_is_callable);
        llvm::Value* cdr_is_bool = ctx_.builder().CreateICmpEQ(cdr_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_BOOL));
        llvm::Value* cdr_is_char = ctx_.builder().CreateICmpEQ(cdr_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CHAR));
        llvm::Value* cdr_is_hash_ptr = ctx_.builder().CreateICmpEQ(cdr_base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

        // Create blocks for each type
        llvm::BasicBlock* double_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_extract_double", current_func);
        llvm::BasicBlock* check_cons_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_check_cons", current_func);
        llvm::BasicBlock* cons_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_extract_cons", current_func);
        llvm::BasicBlock* check_null_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_check_null", current_func);
        llvm::BasicBlock* null_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_extract_null", current_func);
        llvm::BasicBlock* check_string_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_check_string", current_func);
        llvm::BasicBlock* string_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_extract_string", current_func);
        llvm::BasicBlock* check_lambda_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_check_lambda", current_func);
        llvm::BasicBlock* lambda_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_extract_lambda", current_func);
        llvm::BasicBlock* check_closure_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_check_closure", current_func);
        llvm::BasicBlock* closure_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_extract_closure", current_func);
        llvm::BasicBlock* check_bool_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_check_bool", current_func);
        llvm::BasicBlock* bool_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_extract_bool", current_func);
        llvm::BasicBlock* check_char_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_check_char", current_func);
        llvm::BasicBlock* char_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_extract_char", current_func);
        llvm::BasicBlock* check_hash_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_check_hash", current_func);
        llvm::BasicBlock* hash_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_extract_hash", current_func);
        llvm::BasicBlock* int_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_extract_int", current_func);
        llvm::BasicBlock* merge_cdr = llvm::BasicBlock::Create(ctx_.context(), "cdr_merge", current_func);

        ctx_.builder().CreateCondBr(cdr_is_double, double_cdr, check_cons_cdr);

        ctx_.builder().SetInsertPoint(double_cdr);
        llvm::Value* cdr_double = ctx_.builder().CreateCall(mem_.getTaggedConsGetDouble(), {cons_ptr, is_cdr});
        llvm::Value* tagged_double_cdr = tagged_.packDouble(cdr_double);
        ctx_.builder().CreateBr(merge_cdr);
        llvm::BasicBlock* double_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(check_cons_cdr);
        ctx_.builder().CreateCondBr(cdr_is_cons_ptr, cons_cdr, check_null_cdr);

        ctx_.builder().SetInsertPoint(cons_cdr);
        llvm::Value* cdr_cons = ctx_.builder().CreateCall(mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr});
        // Pack as HEAP_PTR (consolidated format) - type checks handle both formats
        llvm::Value* tagged_cons_cdr = tagged_.packHeapPtr(cdr_cons);
        ctx_.builder().CreateBr(merge_cdr);
        llvm::BasicBlock* cons_exit_cdr = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(check_null_cdr);
        ctx_.builder().CreateCondBr(cdr_is_null_type, null_cdr, check_string_cdr);

        ctx_.builder().SetInsertPoint(null_cdr);
        llvm::Value* tagged_null_extract = tagged_.packNull();
        ctx_.builder().CreateBr(merge_cdr);
        llvm::BasicBlock* null_cdr_exit = ctx_.builder().GetInsertBlock();

        // Handle STRING_PTR
        ctx_.builder().SetInsertPoint(check_string_cdr);
        ctx_.builder().CreateCondBr(cdr_is_string_ptr, string_cdr, check_lambda_cdr);

        ctx_.builder().SetInsertPoint(string_cdr);
        llvm::Value* cdr_string = ctx_.builder().CreateCall(mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr});
        llvm::Value* tagged_string_cdr = tagged_.packPtr(ctx_.builder().CreateIntToPtr(cdr_string, ctx_.ptrType()), ESHKOL_VALUE_HEAP_PTR);
        ctx_.builder().CreateBr(merge_cdr);
        llvm::BasicBlock* string_exit = ctx_.builder().GetInsertBlock();

        // Handle LAMBDA_SEXPR
        ctx_.builder().SetInsertPoint(check_lambda_cdr);
        ctx_.builder().CreateCondBr(cdr_is_lambda_sexpr, lambda_cdr, check_closure_cdr);

        ctx_.builder().SetInsertPoint(lambda_cdr);
        llvm::Value* lambda_type_cdr = ctx_.builder().CreateCall(mem_.getTaggedConsGetType(), {cons_ptr, is_cdr});
        llvm::Value* lambda_flags_cdr = ctx_.builder().CreateCall(mem_.getTaggedConsGetFlags(), {cons_ptr, is_cdr});
        llvm::Value* lambda_ptr_cdr = ctx_.builder().CreateCall(mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr});
        llvm::Value* tagged_lambda_cdr = tagged_.packPtrWithFlags(
            ctx_.builder().CreateIntToPtr(lambda_ptr_cdr, ctx_.ptrType()),
            lambda_type_cdr, lambda_flags_cdr);
        ctx_.builder().CreateBr(merge_cdr);
        llvm::BasicBlock* lambda_exit = ctx_.builder().GetInsertBlock();

        // Handle CLOSURE_PTR
        ctx_.builder().SetInsertPoint(check_closure_cdr);
        ctx_.builder().CreateCondBr(cdr_is_closure_ptr, closure_cdr, check_bool_cdr);

        ctx_.builder().SetInsertPoint(closure_cdr);
        llvm::Value* closure_type_cdr = ctx_.builder().CreateCall(mem_.getTaggedConsGetType(), {cons_ptr, is_cdr});
        llvm::Value* closure_flags_cdr = ctx_.builder().CreateCall(mem_.getTaggedConsGetFlags(), {cons_ptr, is_cdr});
        llvm::Value* closure_ptr_cdr = ctx_.builder().CreateCall(mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr});
        llvm::Value* tagged_closure_cdr = tagged_.packPtrWithFlags(
            ctx_.builder().CreateIntToPtr(closure_ptr_cdr, ctx_.ptrType()),
            closure_type_cdr, closure_flags_cdr);
        ctx_.builder().CreateBr(merge_cdr);
        llvm::BasicBlock* closure_exit = ctx_.builder().GetInsertBlock();

        // Handle boolean values
        ctx_.builder().SetInsertPoint(check_bool_cdr);
        ctx_.builder().CreateCondBr(cdr_is_bool, bool_cdr, check_char_cdr);

        ctx_.builder().SetInsertPoint(bool_cdr);
        llvm::Value* cdr_bool_int = ctx_.builder().CreateCall(mem_.getTaggedConsGetInt64(), {cons_ptr, is_cdr});
        llvm::Value* cdr_bool_i1 = ctx_.builder().CreateICmpNE(cdr_bool_int, llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* tagged_bool_cdr = tagged_.packBool(cdr_bool_i1);
        ctx_.builder().CreateBr(merge_cdr);
        llvm::BasicBlock* bool_exit = ctx_.builder().GetInsertBlock();

        // Handle CHAR values
        ctx_.builder().SetInsertPoint(check_char_cdr);
        ctx_.builder().CreateCondBr(cdr_is_char, char_cdr, check_hash_cdr);

        ctx_.builder().SetInsertPoint(char_cdr);
        llvm::Value* cdr_char_int = ctx_.builder().CreateCall(mem_.getTaggedConsGetInt64(), {cons_ptr, is_cdr});
        llvm::Value* tagged_char_cdr = tagged_.packChar(cdr_char_int);
        ctx_.builder().CreateBr(merge_cdr);
        llvm::BasicBlock* char_exit = ctx_.builder().GetInsertBlock();

        // Handle HASH_PTR for hash tables (legacy HASH_PTR - repacks as HEAP_PTR)
        ctx_.builder().SetInsertPoint(check_hash_cdr);
        ctx_.builder().CreateCondBr(cdr_is_hash_ptr, hash_cdr, int_cdr);

        ctx_.builder().SetInsertPoint(hash_cdr);
        llvm::Value* cdr_hash = ctx_.builder().CreateCall(mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr});
        // Repack as HEAP_PTR (consolidated format) - subtype is in object header
        llvm::Value* tagged_hash_cdr = tagged_.packHeapPtr(ctx_.builder().CreateIntToPtr(cdr_hash, ctx_.ptrType()));
        ctx_.builder().CreateBr(merge_cdr);
        llvm::BasicBlock* hash_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(int_cdr);
        llvm::Value* cdr_int64 = ctx_.builder().CreateCall(mem_.getTaggedConsGetInt64(), {cons_ptr, is_cdr});
        llvm::Value* tagged_int64_cdr = tagged_.packInt64(cdr_int64, true);
        ctx_.builder().CreateBr(merge_cdr);
        llvm::BasicBlock* int_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(merge_cdr);
        llvm::PHINode* cdr_tagged_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 10);
        cdr_tagged_phi->addIncoming(tagged_double_cdr, double_exit);
        cdr_tagged_phi->addIncoming(tagged_cons_cdr, cons_exit_cdr);
        cdr_tagged_phi->addIncoming(tagged_null_extract, null_cdr_exit);
        cdr_tagged_phi->addIncoming(tagged_string_cdr, string_exit);
        cdr_tagged_phi->addIncoming(tagged_lambda_cdr, lambda_exit);
        cdr_tagged_phi->addIncoming(tagged_closure_cdr, closure_exit);
        cdr_tagged_phi->addIncoming(tagged_bool_cdr, bool_exit);
        cdr_tagged_phi->addIncoming(tagged_char_cdr, char_exit);
        cdr_tagged_phi->addIncoming(tagged_hash_cdr, hash_exit);
        cdr_tagged_phi->addIncoming(tagged_int64_cdr, int_exit);
        ctx_.builder().CreateBr(cdr_final);
        llvm::BasicBlock* merge_exit = ctx_.builder().GetInsertBlock();

        // Final merge
        ctx_.builder().SetInsertPoint(cdr_final);
        llvm::PHINode* final_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3);
        final_phi->addIncoming(vector_phi, vector_merge_exit);
        final_phi->addIncoming(null_tagged_cdr, null_exit);
        final_phi->addIncoming(cdr_tagged_phi, merge_exit);

        // Create continuation block so cdr_final stays clean with just the PHI
        llvm::BasicBlock* cdr_done = llvm::BasicBlock::Create(ctx_.context(), "cdr_done", current_func);
        ctx_.builder().CreateBr(cdr_done);
        ctx_.builder().SetInsertPoint(cdr_done);

        return final_phi;
    }

    // Fallback for non-tagged values
    llvm::Value* pair_int_safe = tagged_.safeExtractInt64(pair_val);
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(pair_int_safe, llvm::ConstantInt::get(ctx_.int64Type(), 0));

    llvm::BasicBlock* null_block = llvm::BasicBlock::Create(ctx_.context(), "cdr_null", current_func);
    llvm::BasicBlock* valid_block = llvm::BasicBlock::Create(ctx_.context(), "cdr_valid", current_func);
    llvm::BasicBlock* continue_block = llvm::BasicBlock::Create(ctx_.context(), "cdr_continue", current_func);

    ctx_.builder().CreateCondBr(is_null, null_block, valid_block);

    ctx_.builder().SetInsertPoint(null_block);
    llvm::Value* null_tagged = tagged_.packNull();
    ctx_.builder().CreateBr(continue_block);
    llvm::BasicBlock* null_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(valid_block);
    llvm::Value* cons_ptr = ctx_.builder().CreateIntToPtr(pair_int_safe, ctx_.ptrType());
    llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 1);
    llvm::Value* cdr_int64 = ctx_.builder().CreateCall(mem_.getTaggedConsGetInt64(), {cons_ptr, is_cdr});
    llvm::Value* valid_tagged = tagged_.packInt64(cdr_int64, true);
    ctx_.builder().CreateBr(continue_block);
    llvm::BasicBlock* valid_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(continue_block);
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2);
    result_phi->addIncoming(null_tagged, null_exit);
    result_phi->addIncoming(valid_tagged, valid_exit);

    // Create done block to not pollute continue_block
    llvm::BasicBlock* cdr_fallback_done = llvm::BasicBlock::Create(ctx_.context(), "cdr_fallback_done", current_func);
    ctx_.builder().CreateBr(cdr_fallback_done);
    ctx_.builder().SetInsertPoint(cdr_fallback_done);

    return result_phi;
}

llvm::Value* CollectionCodegen::list(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("CollectionCodegen::list - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars == 0) {
        // Empty list - return properly tagged NULL value
        return tagged_.packNull();
    }

    // SCHEME EVALUATION ORDER FIX: Evaluate arguments left-to-right FIRST
    // Then build list right-to-left using the stored results.
    // This ensures side effects happen in left-to-right order.
    std::vector<llvm::Value*> evaluated_args;
    evaluated_args.reserve(op->call_op.num_vars);

    // Phase 1: Evaluate all arguments LEFT-TO-RIGHT
    for (size_t i = 0; i < op->call_op.num_vars; i++) {
        // Generate element via callback
        void* typed_elem = codegen_typed_ast_callback_(&op->call_op.variables[i], callback_context_);
        if (!typed_elem) {
            evaluated_args.push_back(tagged_.packNull());
            continue;
        }

        // Convert to tagged value
        llvm::Value* tagged_elem = typed_to_tagged_callback_(typed_elem, callback_context_);
        if (!tagged_elem) {
            evaluated_args.push_back(tagged_.packNull());
            continue;
        }
        evaluated_args.push_back(tagged_elem);
    }

    // Phase 2: Build list from right to left using already-evaluated values
    llvm::Value* result = tagged_.packNull();
    for (int64_t i = evaluated_args.size() - 1; i >= 0; i--) {
        // Create cons cell: (element . result)
        result = allocConsCell(evaluated_args[i], result);
    }

    return result;
}

llvm::Value* CollectionCodegen::listStar(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::listStar called - using fallback");
    return tagged_.packNull();
}

llvm::Value* CollectionCodegen::isNull(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("CollectionCodegen::isNull - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("null? requires exactly 1 argument");
        return nullptr;
    }

    // Generate code for argument via callback
    void* typed_val = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!typed_val) return nullptr;

    // Convert to tagged value
    llvm::Value* tagged_arg = typed_to_tagged_callback_(typed_val, callback_context_);
    if (!tagged_arg) return nullptr;

    // Get type tag and base type (properly handles legacy types >= 32)
    llvm::Value* type_tag = tagged_.getType(tagged_arg);
    llvm::Value* base_type = tagged_.getBaseType(type_tag);

    // Check for ESHKOL_VALUE_NULL type
    llvm::Value* is_null_type = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL));

    // Also check for CONS_PTR/HEAP_PTR with null pointer (empty list representation)
    llvm::Value* is_cons_legacy = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    llvm::Value* is_cons_type = ctx_.builder().CreateOr(is_cons_legacy, is_heap_ptr);
    llvm::Value* data_val = tagged_.unpackInt64(tagged_arg);
    llvm::Value* is_null_ptr = ctx_.builder().CreateICmpEQ(data_val,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* is_cons_null = ctx_.builder().CreateAnd(is_cons_type, is_null_ptr);

    // Either null type or cons with null pointer
    llvm::Value* result = ctx_.builder().CreateOr(is_null_type, is_cons_null);
    return tagged_.packBool(result);
}

llvm::Value* CollectionCodegen::isPair(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("CollectionCodegen::isPair - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("pair? requires exactly 1 argument");
        return nullptr;
    }

    // Generate code for argument via callback
    void* typed_val = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!typed_val) return nullptr;

    // Convert to tagged value
    llvm::Value* tagged_arg = typed_to_tagged_callback_(typed_val, callback_context_);
    if (!tagged_arg) return nullptr;

    // Get type tag and base type (properly handles legacy types >= 32)
    llvm::Value* type_tag = tagged_.getType(tagged_arg);
    llvm::Value* base_type = tagged_.getBaseType(type_tag);

    // Check if type is HEAP_PTR (consolidated format)
    llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    // Also check that the pointer is not null (not empty list)
    llvm::Value* data_val = tagged_.unpackInt64(tagged_arg);
    llvm::Value* is_not_null = ctx_.builder().CreateICmpNE(data_val,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));

    // CONSOLIDATED FORMAT FIX: For HEAP_PTR, we must check the subtype in the object header
    // to distinguish cons cells from strings, vectors, tensors, etc.
    // Use checkHeapSubtype to verify subtype is HEAP_SUBTYPE_CONS (0)
    llvm::Value* is_cons_subtype = tagged_.checkHeapSubtype(tagged_arg, HEAP_SUBTYPE_CONS);

    // Pair is HEAP_PTR with non-null pointer AND subtype CONS
    llvm::Value* is_heap_cons = ctx_.builder().CreateAnd(is_heap_ptr, is_not_null);
    llvm::Value* result = ctx_.builder().CreateAnd(is_heap_cons, is_cons_subtype);
    return tagged_.packBool(result);
}

llvm::Value* CollectionCodegen::makeVector(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("CollectionCodegen::makeVector - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_warn("make-vector requires 1 or 2 arguments");
        return nullptr;
    }

    // Get length via callback
    void* len_typed = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!len_typed) return nullptr;
    llvm::Value* len_tagged = typed_to_tagged_callback_(len_typed, callback_context_);
    if (!len_tagged) return nullptr;

    // Extract length as i64
    llvm::Value* length = tagged_.unpackInt64(len_tagged);

    // Allocate from arena with header (for consolidated HEAP_PTR type)
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    llvm::Value* vec_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateVectorWithHeader(),
        {arena_ptr, length});

    // Store length at beginning (offset 0)
    llvm::Value* len_ptr = ctx_.builder().CreatePointerCast(vec_ptr, ctx_.ptrType());
    ctx_.builder().CreateStore(length, len_ptr);

    // Get fill value (default to 0 if not provided)
    llvm::Value* fill_val;
    if (op->call_op.num_vars == 2) {
        void* fill_typed = codegen_typed_ast_callback_(&op->call_op.variables[1], callback_context_);
        if (!fill_typed) {
            fill_val = tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
        } else {
            fill_val = typed_to_tagged_callback_(fill_typed, callback_context_);
            if (!fill_val) {
                fill_val = tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
            }
        }
    } else {
        fill_val = tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
    }

    // Fill loop
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_header = llvm::BasicBlock::Create(ctx_.context(), "vec_fill_header", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "vec_fill_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "vec_fill_exit", current_func);

    // Get pointer to elements (after length field)
    llvm::Value* elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), vec_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* elem_base_typed = ctx_.builder().CreatePointerCast(elem_base, ctx_.ptrType());

    ctx_.builder().CreateBr(loop_header);

    ctx_.builder().SetInsertPoint(loop_header);
    llvm::PHINode* i = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "fill_i");
    i->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 0),
        loop_header->getSinglePredecessor());
    llvm::Value* done = ctx_.builder().CreateICmpUGE(i, length);
    ctx_.builder().CreateCondBr(done, loop_exit, loop_body);

    ctx_.builder().SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), elem_base_typed, i);
    ctx_.builder().CreateStore(fill_val, elem_ptr);
    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    i->addIncoming(next_i, loop_body);
    ctx_.builder().CreateBr(loop_header);

    ctx_.builder().SetInsertPoint(loop_exit);
    return tagged_.packHeapPtr(vec_ptr);
}

llvm::Value* CollectionCodegen::vector(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("CollectionCodegen::vector - callbacks not set");
        return tagged_.packNull();
    }

    uint64_t num_elems = op->call_op.num_vars;

    // Allocate from arena with header (for consolidated HEAP_PTR type)
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    llvm::Value* vec_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateVectorWithHeader(),
        {arena_ptr, llvm::ConstantInt::get(ctx_.int64Type(), num_elems)});

    // Store length at beginning (offset 0)
    llvm::Value* len_ptr = ctx_.builder().CreatePointerCast(vec_ptr, ctx_.ptrType());
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), num_elems), len_ptr);

    // Get pointer to elements (after length field)
    llvm::Value* elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), vec_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* elem_base_typed = ctx_.builder().CreatePointerCast(elem_base, ctx_.ptrType());

    // Store each element
    for (uint64_t i = 0; i < num_elems; i++) {
        void* elem_typed = codegen_typed_ast_callback_(&op->call_op.variables[i], callback_context_);
        if (!elem_typed) return nullptr;

        llvm::Value* tagged_elem = typed_to_tagged_callback_(elem_typed, callback_context_);
        if (!tagged_elem) return nullptr;

        llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), elem_base_typed,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        ctx_.builder().CreateStore(tagged_elem, elem_ptr);
    }

    return tagged_.packHeapPtr(vec_ptr);
}

llvm::Value* CollectionCodegen::vectorLength(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("CollectionCodegen::vectorLength - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("vector-length requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* vec_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!vec_arg) return nullptr;

    llvm::Value* length = nullptr;

    if (vec_arg->getType() == ctx_.taggedValueType()) {
        // M1 CONSOLIDATION: Distinguish vector vs tensor via header subtype
        // Both are HEAP_PTR (8), so we need to check the header at ptr-8
        llvm::Value* type_tag = tagged_.getType(vec_arg);
        llvm::Value* base_type = tagged_.getBaseType(type_tag);

        llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

        llvm::Value* ptr_int = tagged_.unpackInt64(vec_arg);
        llvm::Value* ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());

        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* check_subtype = llvm::BasicBlock::Create(ctx_.context(), "check_subtype", current_func);
        llvm::BasicBlock* tensor_block = llvm::BasicBlock::Create(ctx_.context(), "tensor_len", current_func);
        llvm::BasicBlock* vector_block = llvm::BasicBlock::Create(ctx_.context(), "vector_len", current_func);
        llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "len_merge", current_func);

        // If HEAP_PTR, check subtype; otherwise assume legacy vector
        ctx_.builder().CreateCondBr(is_heap_ptr, check_subtype, vector_block);

        // Check subtype in header at ptr-8
        ctx_.builder().SetInsertPoint(check_subtype);
        llvm::Value* header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), -8), "header_ptr");
        llvm::Value* subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), header_ptr, "subtype");
        llvm::Value* is_tensor = ctx_.builder().CreateICmpEQ(subtype,
            llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));
        ctx_.builder().CreateCondBr(is_tensor, tensor_block, vector_block);

        // Tensor path: load total_elements from field 3
        ctx_.builder().SetInsertPoint(tensor_block);
        llvm::Value* total_elem_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), ptr, 3);
        llvm::Value* tensor_len = ctx_.builder().CreateLoad(ctx_.int64Type(), total_elem_ptr);
        ctx_.builder().CreateBr(merge_block);

        // Vector path: load length from beginning
        ctx_.builder().SetInsertPoint(vector_block);
        llvm::Value* vec_len_ptr = ctx_.builder().CreatePointerCast(ptr, ctx_.ptrType());
        llvm::Value* vec_len = ctx_.builder().CreateLoad(ctx_.int64Type(), vec_len_ptr);
        ctx_.builder().CreateBr(merge_block);

        // Merge - only 2 predecessors (tensor_block and vector_block)
        ctx_.builder().SetInsertPoint(merge_block);
        llvm::PHINode* len_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "length");
        len_phi->addIncoming(tensor_len, tensor_block);
        len_phi->addIncoming(vec_len, vector_block);
        length = len_phi;
    } else if (vec_arg->getType()->isIntegerTy(64)) {
        // Raw i64 pointer (likely a tensor)
        llvm::Value* ptr = ctx_.builder().CreateIntToPtr(vec_arg, ctx_.ptrType());
        llvm::Value* total_elem_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), ptr, 3);
        length = ctx_.builder().CreateLoad(ctx_.int64Type(), total_elem_ptr);
    } else {
        eshkol_warn("vector-length: unexpected argument type");
        return nullptr;
    }

    return tagged_.packInt64(length, true);
}

llvm::Value* CollectionCodegen::vectorRef(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_ || !codegen_typed_ast_callback_) {
        eshkol_warn("CollectionCodegen::vectorRef - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 2) {
        eshkol_warn("vector-ref requires exactly 2 arguments");
        return nullptr;
    }

    llvm::Value* vec_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    void* idx_typed = codegen_typed_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!vec_arg || !idx_typed) return nullptr;

    llvm::Value* idx_tagged = typed_to_tagged_callback_(idx_typed, callback_context_);
    if (!idx_tagged) return nullptr;

    // Extract index - handle both raw i64 and tagged values
    llvm::Value* idx = idx_tagged;
    if (idx->getType() == ctx_.taggedValueType()) {
        idx = tagged_.unpackInt64(idx);
    } else if (idx->getType() != ctx_.int64Type()) {
        if (idx->getType()->isIntegerTy()) {
            idx = ctx_.builder().CreateSExtOrTrunc(idx, ctx_.int64Type());
        }
    }

    // M1 CONSOLIDATION: Detect tensor vs Scheme vector via header subtype
    // Both use HEAP_PTR type, but tensors have HEAP_SUBTYPE_TENSOR (3)
    // and Scheme vectors have HEAP_SUBTYPE_VECTOR (2)
    llvm::Value* vec_type = tagged_.getType(vec_arg);
    llvm::Value* vec_base_type = tagged_.getBaseType(vec_type);
    llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(vec_base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* check_subtype = llvm::BasicBlock::Create(ctx_.context(), "vref_check_subtype", current_func);
    llvm::BasicBlock* tensor_path = llvm::BasicBlock::Create(ctx_.context(), "vref_tensor", current_func);
    llvm::BasicBlock* vector_path = llvm::BasicBlock::Create(ctx_.context(), "vref_vector", current_func);
    llvm::BasicBlock* merge_path = llvm::BasicBlock::Create(ctx_.context(), "vref_merge", current_func);

    // If HEAP_PTR, check subtype to distinguish tensor vs vector
    ctx_.builder().CreateCondBr(is_heap_ptr, check_subtype, vector_path);

    // Check subtype in header
    ctx_.builder().SetInsertPoint(check_subtype);
    llvm::Value* vec_ptr_for_sub = ctx_.builder().CreateIntToPtr(
        tagged_.unpackInt64(vec_arg), ctx_.ptrType());
    llvm::Value* header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), vec_ptr_for_sub,
        llvm::ConstantInt::get(ctx_.int64Type(), -8));
    llvm::Value* subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), header_ptr);
    llvm::Value* is_tensor = ctx_.builder().CreateICmpEQ(subtype,
        llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));
    ctx_.builder().CreateCondBr(is_tensor, tensor_path, vector_path);

    // TENSOR PATH: Extract element or row slice from tensor structure
    // Tensor layout: {dims_ptr*, num_dims, elements_ptr*, total_elements}
    // For 2D+ tensors, vector-ref returns a row slice (1D tensor)
    // For 1D tensors, vector-ref returns the scalar element
    ctx_.builder().SetInsertPoint(tensor_path);
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(vec_arg);
    llvm::Value* tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());

    // Load tensor metadata
    llvm::Value* dims_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 0);
    llvm::Value* dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), dims_field_ptr);
    llvm::Value* ndim_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 1);
    llvm::Value* ndim = ctx_.builder().CreateLoad(ctx_.int64Type(), ndim_field_ptr);
    llvm::Value* elems_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 2);
    llvm::Value* elems_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), elems_field_ptr);
    llvm::Value* total_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 3);
    llvm::Value* total_elements = ctx_.builder().CreateLoad(ctx_.int64Type(), total_field_ptr);

    // Check if tensor is multi-dimensional (ndim > 1)
    llvm::Value* is_multidim = ctx_.builder().CreateICmpUGT(ndim, llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::BasicBlock* tensor_1d_path = llvm::BasicBlock::Create(ctx_.context(), "vref_tensor_1d", current_func);
    llvm::BasicBlock* tensor_nd_path = llvm::BasicBlock::Create(ctx_.context(), "vref_tensor_nd", current_func);
    llvm::BasicBlock* tensor_merge = llvm::BasicBlock::Create(ctx_.context(), "vref_tensor_merge", current_func);

    ctx_.builder().CreateCondBr(is_multidim, tensor_nd_path, tensor_1d_path);

    // === 1D TENSOR PATH: Return single element (original behavior) ===
    ctx_.builder().SetInsertPoint(tensor_1d_path);
    llvm::Value* tensor_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), elems_ptr, idx);
    llvm::Value* tensor_elem_int64 = ctx_.builder().CreateLoad(ctx_.int64Type(), tensor_elem_ptr);

    // Check if the element is an AD node pointer (for gradient computation)
    llvm::Value* not_zero = ctx_.builder().CreateICmpNE(tensor_elem_int64,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* in_ptr_range = ctx_.builder().CreateICmpULT(tensor_elem_int64,
        llvm::ConstantInt::get(ctx_.int64Type(), 0x0001000000000000ULL));
    llvm::Value* could_be_ad_ptr = ctx_.builder().CreateAnd(not_zero, in_ptr_range);

    llvm::BasicBlock* is_ad_node = llvm::BasicBlock::Create(ctx_.context(), "vref_ad_node", current_func);
    llvm::BasicBlock* is_double = llvm::BasicBlock::Create(ctx_.context(), "vref_double", current_func);
    llvm::BasicBlock* elem_merge = llvm::BasicBlock::Create(ctx_.context(), "vref_elem_merge", current_func);

    ctx_.builder().CreateCondBr(could_be_ad_ptr, is_ad_node, is_double);

    ctx_.builder().SetInsertPoint(is_ad_node);
    // Pack as CALLABLE (AD nodes have CALLABLE_SUBTYPE_AD_NODE in header)
    llvm::Value* ad_node_result = tagged_.packPtr(tensor_elem_int64, ESHKOL_VALUE_CALLABLE);
    ctx_.builder().CreateBr(elem_merge);
    llvm::BasicBlock* ad_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(is_double);
    llvm::Value* elem_as_double = ctx_.builder().CreateBitCast(tensor_elem_int64, ctx_.doubleType());
    llvm::Value* double_result = tagged_.packDouble(elem_as_double);
    ctx_.builder().CreateBr(elem_merge);
    llvm::BasicBlock* double_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(elem_merge);
    llvm::PHINode* elem_result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "tensor_elem");
    elem_result->addIncoming(ad_node_result, ad_exit);
    elem_result->addIncoming(double_result, double_exit);
    ctx_.builder().CreateBr(tensor_merge);
    llvm::BasicBlock* tensor_1d_exit = ctx_.builder().GetInsertBlock();

    // === N-D TENSOR PATH: Return row slice as 1D tensor ===
    ctx_.builder().SetInsertPoint(tensor_nd_path);

    // Get row size from dims[1] (number of columns)
    llvm::Value* row_size_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* row_size = ctx_.builder().CreateLoad(ctx_.int64Type(), row_size_ptr);

    // Calculate row offset: idx * row_size
    llvm::Value* row_offset = ctx_.builder().CreateMul(idx, row_size);

    // Get arena for OALR-compliant allocation
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());

    // Allocate new 1D tensor struct via arena (OALR compliant - no malloc)
    llvm::Value* slice_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Allocate dims array for 1D tensor via arena: 1 * 8 bytes
    llvm::Value* slice_dims = ctx_.builder().CreateCall(mem_.getArenaAllocate(),
        {arena_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 8)});
    ctx_.builder().CreateStore(row_size, slice_dims);

    // Fill slice tensor struct
    // Field 0: dims pointer
    llvm::Value* slice_f0 = ctx_.builder().CreateStructGEP(ctx_.tensorType(), slice_tensor, 0);
    ctx_.builder().CreateStore(slice_dims, slice_f0);

    // Field 1: ndim = 1
    llvm::Value* slice_f1 = ctx_.builder().CreateStructGEP(ctx_.tensorType(), slice_tensor, 1);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), slice_f1);

    // Field 2: elements pointer (view into original at row offset)
    llvm::Value* row_start = ctx_.builder().CreateGEP(ctx_.int64Type(), elems_ptr, row_offset);
    llvm::Value* slice_f2 = ctx_.builder().CreateStructGEP(ctx_.tensorType(), slice_tensor, 2);
    ctx_.builder().CreateStore(row_start, slice_f2);

    // Field 3: total_elements = row_size
    llvm::Value* slice_f3 = ctx_.builder().CreateStructGEP(ctx_.tensorType(), slice_tensor, 3);
    ctx_.builder().CreateStore(row_size, slice_f3);

    // Pack as TENSOR_PTR
    llvm::Value* slice_int = ctx_.builder().CreatePtrToInt(slice_tensor, ctx_.int64Type());
    llvm::Value* slice_result = tagged_.packPtr(slice_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(tensor_merge);
    llvm::BasicBlock* tensor_nd_exit = ctx_.builder().GetInsertBlock();

    // === TENSOR MERGE ===
    ctx_.builder().SetInsertPoint(tensor_merge);
    llvm::PHINode* tensor_result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "tensor_result");
    tensor_result->addIncoming(elem_result, tensor_1d_exit);
    tensor_result->addIncoming(slice_result, tensor_nd_exit);
    ctx_.builder().CreateBr(merge_path);
    llvm::BasicBlock* tensor_exit = ctx_.builder().GetInsertBlock();

    // VECTOR PATH: Original Scheme vector logic
    ctx_.builder().SetInsertPoint(vector_path);
    llvm::Value* vec_ptr_int = tagged_.unpackInt64(vec_arg);
    llvm::Value* vec_ptr = ctx_.builder().CreateIntToPtr(vec_ptr_int, ctx_.ptrType());

    // Get pointer to elements (after length field)
    llvm::Value* elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), vec_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* elem_base_typed = ctx_.builder().CreatePointerCast(elem_base, ctx_.ptrType());

    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), elem_base_typed, idx);
    llvm::Value* vector_result = ctx_.builder().CreateLoad(ctx_.taggedValueType(), elem_ptr);
    ctx_.builder().CreateBr(merge_path);
    llvm::BasicBlock* vector_exit = ctx_.builder().GetInsertBlock();

    // MERGE: Return appropriate result
    ctx_.builder().SetInsertPoint(merge_path);
    llvm::PHINode* final_result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "vref_result");
    final_result->addIncoming(tensor_result, tensor_exit);
    final_result->addIncoming(vector_result, vector_exit);

    return final_result;
}

llvm::Value* CollectionCodegen::vectorSet(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_ || !codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("CollectionCodegen::vectorSet - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 3) {
        eshkol_warn("vector-set! requires exactly 3 arguments");
        return nullptr;
    }

    llvm::Value* vec_arg = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    void* idx_typed = codegen_typed_ast_callback_(&op->call_op.variables[1], callback_context_);
    void* val_typed = codegen_typed_ast_callback_(&op->call_op.variables[2], callback_context_);
    if (!vec_arg || !idx_typed || !val_typed) return nullptr;

    llvm::Value* idx_tagged = typed_to_tagged_callback_(idx_typed, callback_context_);
    llvm::Value* tagged_val = typed_to_tagged_callback_(val_typed, callback_context_);
    if (!idx_tagged || !tagged_val) return nullptr;

    // Extract vector pointer
    llvm::Value* vec_ptr_int = tagged_.unpackInt64(vec_arg);
    llvm::Value* vec_ptr = ctx_.builder().CreateIntToPtr(vec_ptr_int, ctx_.ptrType());

    // Get pointer to elements (after length field)
    llvm::Value* elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), vec_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* elem_base_typed = ctx_.builder().CreatePointerCast(elem_base, ctx_.ptrType());

    // Extract index
    llvm::Value* idx = idx_tagged;
    if (idx->getType() == ctx_.taggedValueType()) {
        idx = tagged_.unpackInt64(idx);
    } else if (idx->getType() != ctx_.int64Type()) {
        if (idx->getType()->isIntegerTy()) {
            idx = ctx_.builder().CreateSExtOrTrunc(idx, ctx_.int64Type());
        } else if (idx->getType()->isFloatingPointTy()) {
            idx = ctx_.builder().CreateFPToSI(idx, ctx_.int64Type());
        }
    }

    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), elem_base_typed, idx);
    ctx_.builder().CreateStore(tagged_val, elem_ptr);

    // Return the vector
    return vec_arg;
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
