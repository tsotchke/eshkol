/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * ArithmeticCodegen implementation
 *
 * This module implements fully polymorphic arithmetic operations that handle:
 * - Integer (exact) and floating-point (inexact) numbers
 * - Dual numbers for forward-mode automatic differentiation
 * - AD nodes for reverse-mode automatic differentiation (computational graphs)
 * - Vectors and tensors for element-wise operations
 */

#include <eshkol/backend/arithmetic_codegen.h>
#include <eshkol/eshkol.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <llvm/IR/Intrinsics.h>
#include <llvm/Config/llvm-config.h>
#include <eshkol/logger.h>

// LLVM VERSION COMPATIBILITY
#if LLVM_VERSION_MAJOR >= 18
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getOrInsertDeclaration(mod, id, types)
#else
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getDeclaration(mod, id, types)
#endif

namespace eshkol {

// Helper: Get arena pointer from __global_arena global variable
static llvm::Value* getArenaPtr(CodegenContext& ctx) {
    llvm::GlobalVariable* arena_global = ctx.module().getNamedGlobal("__global_arena");
    if (!arena_global) return nullptr;
    return ctx.builder().CreateLoad(ctx.ptrType(), arena_global);
}

// Helper: Get or declare eshkol_bignum_from_overflow(arena, a, b, op)
static llvm::Function* getBignumFromOverflowFunc(CodegenContext& ctx) {
    llvm::Function* func = ctx.module().getFunction("eshkol_bignum_from_overflow");
    if (!func) {
        llvm::FunctionType* fn_type = llvm::FunctionType::get(
            ctx.ptrType(),  // returns eshkol_bignum_t*
            {ctx.ptrType(),     // arena_t* arena
             ctx.int64Type(),   // int64_t a
             ctx.int64Type(),   // int64_t b
             ctx.int32Type()},  // int op (0=add, 1=sub, 2=mul)
            false);
        func = llvm::Function::Create(fn_type,
            llvm::Function::ExternalLinkage,
            "eshkol_bignum_from_overflow", &ctx.module());
    }
    return func;
}

// Helper: Emit bignum promotion for integer overflow
// Calls eshkol_bignum_from_overflow(arena, left, right, op) and packs as HEAP_PTR
static llvm::Value* emitBignumPromotion(CodegenContext& ctx, TaggedValueCodegen& tagged,
                                         llvm::Value* left_int, llvm::Value* right_int,
                                         int op_code) {
    llvm::Value* arena_ptr = getArenaPtr(ctx);
    if (!arena_ptr) {
        // Fallback: promote to double if arena not available
        llvm::Value* l = ctx.builder().CreateSIToFP(left_int, ctx.doubleType());
        llvm::Value* r = ctx.builder().CreateSIToFP(right_int, ctx.doubleType());
        llvm::Value* result;
        switch (op_code) {
            case 0: result = ctx.builder().CreateFAdd(l, r); break;
            case 1: result = ctx.builder().CreateFSub(l, r); break;
            case 2: result = ctx.builder().CreateFMul(l, r); break;
            default: result = ctx.builder().CreateFAdd(l, r); break;
        }
        return tagged.packDouble(result);
    }
    llvm::Function* bignum_func = getBignumFromOverflowFunc(ctx);
    llvm::Value* bignum_ptr = ctx.builder().CreateCall(bignum_func, {
        arena_ptr, left_int, right_int,
        llvm::ConstantInt::get(ctx.int32Type(), op_code)
    }, "bignum_result");
    // Pack as tagged value: type = ESHKOL_VALUE_HEAP_PTR (8)
    return tagged.packPtr(bignum_ptr, ESHKOL_VALUE_HEAP_PTR);
}

ArithmeticCodegen::ArithmeticCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged,
                                     TensorCodegen& tensor, AutodiffCodegen& autodiff,
                                     ComplexCodegen& complex)
    : ctx_(ctx)
    , tagged_(tagged)
    , tensor_(tensor)
    , autodiff_(autodiff)
    , complex_(complex) {
    eshkol_debug("ArithmeticCodegen initialized with all dependencies");
}

// === Helper Functions ===

llvm::Value* ArithmeticCodegen::convertToDual(llvm::Value* operand, llvm::Value* is_dual,
                                               llvm::Value* is_double) {
    // Create blocks for branching
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* is_dual_bb = llvm::BasicBlock::Create(ctx_.context(), "is_dual", func);
    llvm::BasicBlock* not_dual_bb = llvm::BasicBlock::Create(ctx_.context(), "not_dual", func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "dual_merge", func);

    ctx_.builder().CreateCondBr(is_dual, is_dual_bb, not_dual_bb);

    // Already a dual number - unpack it
    ctx_.builder().SetInsertPoint(is_dual_bb);
    llvm::Value* dual_value = autodiff_.unpackDualFromTagged(operand);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* is_dual_exit = ctx_.builder().GetInsertBlock();

    // Not a dual number - convert to dual with zero tangent
    ctx_.builder().SetInsertPoint(not_dual_bb);
    // Check for bignum: HEAP_PTR with BIGNUM subtype → call eshkol_bignum_to_double
    llvm::Value* type_val = tagged_.getType(operand);
    llvm::Value* base_type_val = tagged_.getBaseType(type_val);
    llvm::Value* is_heap = ctx_.builder().CreateICmpEQ(base_type_val,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    llvm::BasicBlock* bignum_check_bb = llvm::BasicBlock::Create(ctx_.context(), "bn_to_dual_check", func);
    llvm::BasicBlock* normal_convert_bb = llvm::BasicBlock::Create(ctx_.context(), "normal_to_dual", func);
    llvm::BasicBlock* convert_merge_bb = llvm::BasicBlock::Create(ctx_.context(), "to_dual_merge", func);

    ctx_.builder().CreateCondBr(is_heap, bignum_check_bb, normal_convert_bb);

    // Bignum → double path (intentionally lossy, dual numbers are floating-point)
    ctx_.builder().SetInsertPoint(bignum_check_bb);
    llvm::Value* ptr_int = tagged_.unpackInt64(operand);
    llvm::Value* bn_ptr = ctx_.builder().CreateIntToPtr(ptr_int, llvm::PointerType::get(ctx_.context(), 0));
    llvm::FunctionType* bn_to_dbl_type = llvm::FunctionType::get(ctx_.doubleType(),
        {llvm::PointerType::get(ctx_.context(), 0)}, false);
    llvm::FunctionCallee bn_to_dbl_fn = ctx_.module().getOrInsertFunction("eshkol_bignum_to_double", bn_to_dbl_type);
    llvm::Value* bn_as_double = ctx_.builder().CreateCall(bn_to_dbl_fn, {bn_ptr});
    ctx_.builder().CreateBr(convert_merge_bb);
    llvm::BasicBlock* bn_convert_exit = ctx_.builder().GetInsertBlock();

    // Normal int/double path
    ctx_.builder().SetInsertPoint(normal_convert_bb);
    llvm::Value* normal_as_double = ctx_.builder().CreateSelect(is_double,
        tagged_.unpackDouble(operand),
        ctx_.builder().CreateSIToFP(tagged_.unpackInt64(operand), ctx_.doubleType()));
    ctx_.builder().CreateBr(convert_merge_bb);
    llvm::BasicBlock* normal_convert_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(convert_merge_bb);
    llvm::PHINode* as_double = ctx_.builder().CreatePHI(ctx_.doubleType(), 2);
    as_double->addIncoming(bn_as_double, bn_convert_exit);
    as_double->addIncoming(normal_as_double, normal_convert_exit);

    llvm::Value* non_dual = autodiff_.createDualNumber(as_double,
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* not_dual_exit = ctx_.builder().GetInsertBlock();

    // Merge
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.dualNumberType(), 2, "dual_phi");
    phi->addIncoming(dual_value, is_dual_exit);
    phi->addIncoming(non_dual, not_dual_exit);

    return phi;
}

llvm::Value* ArithmeticCodegen::convertToADNode(llvm::Value* operand, llvm::Value* is_ad,
                                                 llvm::Value* base_type) {
    // Create blocks for branching
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* is_ad_bb = llvm::BasicBlock::Create(ctx_.context(), "is_ad", func);
    llvm::BasicBlock* not_ad_bb = llvm::BasicBlock::Create(ctx_.context(), "not_ad", func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "ad_merge", func);

    ctx_.builder().CreateCondBr(is_ad, is_ad_bb, not_ad_bb);

    // Already an AD node - unpack pointer
    ctx_.builder().SetInsertPoint(is_ad_bb);
    llvm::Value* ad_ptr = tagged_.unpackPtr(operand);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* is_ad_exit = ctx_.builder().GetInsertBlock();

    // Not an AD node - create constant node from any numeric type
    // Use extractAsDouble which handles DOUBLE, INT64, HEAP_PTR(bignum), CALLABLE(AD)
    ctx_.builder().SetInsertPoint(not_ad_bb);
    llvm::Value* val = extractAsDouble(operand);
    // extractAsDouble creates blocks internally, recapture exit block
    llvm::Value* ad_const = autodiff_.createADConstant(val);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* not_ad_exit = ctx_.builder().GetInsertBlock();

    // Merge
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.ptrType(), 2, "ad_phi");
    phi->addIncoming(ad_ptr, is_ad_exit);
    phi->addIncoming(ad_const, not_ad_exit);

    return phi;
}

llvm::Value* ArithmeticCodegen::isADNode(llvm::Value* operand, llvm::Value* base_type) {
    // Safely check if operand is an AD node (CALLABLE with CALLABLE_SUBTYPE_AD_NODE).
    // Uses branching to avoid dereferencing non-pointer values like DOUBLE/INT64.
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* is_callable_bb = llvm::BasicBlock::Create(ctx_.context(), "is_ad_callable", func);
    llvm::BasicBlock* not_callable_bb = llvm::BasicBlock::Create(ctx_.context(), "is_ad_not_callable", func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "is_ad_merge", func);

    // Check if type is CALLABLE
    llvm::Value* is_callable = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    ctx_.builder().CreateCondBr(is_callable, is_callable_bb, not_callable_bb);

    // CALLABLE path: Check subtype in header
    ctx_.builder().SetInsertPoint(is_callable_bb);
    llvm::Value* is_ad_subtype = tagged_.checkCallableSubtype(operand, CALLABLE_SUBTYPE_AD_NODE);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* callable_exit = ctx_.builder().GetInsertBlock();

    // NOT CALLABLE path: Return false
    ctx_.builder().SetInsertPoint(not_callable_bb);
    llvm::Value* not_ad = llvm::ConstantInt::getFalse(ctx_.context());
    ctx_.builder().CreateBr(merge_bb);

    // Merge
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.int1Type(), 2, "is_ad_phi");
    phi->addIncoming(is_ad_subtype, callable_exit);
    phi->addIncoming(not_ad, not_callable_bb);

    return phi;
}

// === Central AD Dispatch Handlers ===

llvm::Value* ArithmeticCodegen::withADBinaryDispatch(
    llvm::Value* left, llvm::Value* right,
    int ad_op_type,
    std::function<llvm::Value*()> regular_fn) {

    // Extract base types for both operands
    llvm::Value* left_type = tagged_.getType(left);
    llvm::Value* right_type = tagged_.getType(right);
    llvm::Value* left_base = tagged_.getBaseType(left_type);
    llvm::Value* right_base = tagged_.getBaseType(right_type);

    // Check if either operand has CALLABLE type (AD nodes are CALLABLE + subtype check)
    llvm::Value* left_is_callable = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    llvm::Value* right_is_callable = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    llvm::Value* any_callable = ctx_.builder().CreateOr(left_is_callable, right_is_callable);

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* check_left_sub = llvm::BasicBlock::Create(ctx_.context(), "ad_bin_check_left", func);
    llvm::BasicBlock* check_right_bb = llvm::BasicBlock::Create(ctx_.context(), "ad_bin_check_right", func);
    llvm::BasicBlock* check_right_sub = llvm::BasicBlock::Create(ctx_.context(), "ad_bin_check_right_sub", func);
    llvm::BasicBlock* neither_ad = llvm::BasicBlock::Create(ctx_.context(), "ad_bin_neither", func);
    llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), "ad_bin_path", func);
    llvm::BasicBlock* regular_entry = llvm::BasicBlock::Create(ctx_.context(), "ad_bin_regular", func);
    llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "ad_bin_merge", func);

    // If no CALLABLE operand, skip directly to regular path
    ctx_.builder().CreateCondBr(any_callable, check_left_sub, regular_entry);

    // 2-stage subtype check: left first, then right
    ctx_.builder().SetInsertPoint(check_left_sub);
    ctx_.builder().CreateCondBr(left_is_callable,
        llvm::BasicBlock::Create(ctx_.context(), "ad_bin_left_sub", func), check_right_bb);

    // Left is CALLABLE: check AD_NODE subtype
    llvm::BasicBlock* left_sub_bb = &func->back(); // the block we just created
    ctx_.builder().SetInsertPoint(left_sub_bb);
    llvm::Value* left_is_ad = tagged_.checkCallableSubtype(left, CALLABLE_SUBTYPE_AD_NODE);
    ctx_.builder().CreateCondBr(left_is_ad, ad_path, check_right_bb);

    // Check right operand
    ctx_.builder().SetInsertPoint(check_right_bb);
    ctx_.builder().CreateCondBr(right_is_callable, check_right_sub, neither_ad);

    // Right is CALLABLE: check AD_NODE subtype
    ctx_.builder().SetInsertPoint(check_right_sub);
    llvm::Value* right_is_ad = tagged_.checkCallableSubtype(right, CALLABLE_SUBTYPE_AD_NODE);
    ctx_.builder().CreateCondBr(right_is_ad, ad_path, neither_ad);

    // Neither is AD node — fall through to regular path
    ctx_.builder().SetInsertPoint(neither_ad);
    ctx_.builder().CreateBr(regular_entry);

    // AD path: convert both operands to AD nodes, record binary op
    ctx_.builder().SetInsertPoint(ad_path);
    llvm::Value* left_ad = convertToADNode(left, left_is_callable, left_base);
    llvm::Value* right_ad = convertToADNode(right, right_is_callable, right_base);
    llvm::Value* ad_result_node = autodiff_.recordADNodeBinary(ad_op_type, left_ad, right_ad);
    llvm::Value* ad_tagged = tagged_.packCallable(ad_result_node);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* ad_exit = ctx_.builder().GetInsertBlock();

    // Regular path: execute the lambda for all non-AD code paths
    ctx_.builder().SetInsertPoint(regular_entry);
    llvm::Value* regular_result = regular_fn();
    // Lambda may have created blocks — recapture exit block
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* regular_exit = ctx_.builder().GetInsertBlock();

    // Merge AD and regular results
    ctx_.builder().SetInsertPoint(merge);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "ad_bin_result");
    phi->addIncoming(ad_tagged, ad_exit);
    phi->addIncoming(regular_result, regular_exit);

    return phi;
}

llvm::Value* ArithmeticCodegen::withADUnaryDispatch(
    llvm::Value* operand,
    int ad_op_type,
    std::function<llvm::Value*()> regular_fn) {

    // Extract base type
    llvm::Value* op_type = tagged_.getType(operand);
    llvm::Value* base = tagged_.getBaseType(op_type);

    // Check if operand has CALLABLE type
    llvm::Value* is_callable = ctx_.builder().CreateICmpEQ(base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* check_sub = llvm::BasicBlock::Create(ctx_.context(), "ad_un_check_sub", func);
    llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), "ad_un_path", func);
    llvm::BasicBlock* regular_entry = llvm::BasicBlock::Create(ctx_.context(), "ad_un_regular", func);
    llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "ad_un_merge", func);

    // If not CALLABLE, skip to regular path
    ctx_.builder().CreateCondBr(is_callable, check_sub, regular_entry);

    // Check AD_NODE subtype
    ctx_.builder().SetInsertPoint(check_sub);
    llvm::Value* is_ad = tagged_.checkCallableSubtype(operand, CALLABLE_SUBTYPE_AD_NODE);
    ctx_.builder().CreateCondBr(is_ad, ad_path, regular_entry);

    // AD path: unpack, record unary op, repack
    ctx_.builder().SetInsertPoint(ad_path);
    llvm::Value* ad_ptr = tagged_.unpackPtr(operand);
    llvm::Value* ad_result_node = autodiff_.recordADNodeUnary(ad_op_type, ad_ptr);
    llvm::Value* ad_tagged = tagged_.packCallable(ad_result_node);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* ad_exit = ctx_.builder().GetInsertBlock();

    // Regular path: execute the lambda for all non-AD code paths
    ctx_.builder().SetInsertPoint(regular_entry);
    llvm::Value* regular_result = regular_fn();
    // Lambda may have created blocks — recapture exit block
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* regular_exit = ctx_.builder().GetInsertBlock();

    // Merge AD and regular results
    ctx_.builder().SetInsertPoint(merge);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "ad_un_result");
    phi->addIncoming(ad_tagged, ad_exit);
    phi->addIncoming(regular_result, regular_exit);

    return phi;
}

// === Complex Number Promotion Helper ===

llvm::Value* ArithmeticCodegen::convertToComplex(llvm::Value* operand, llvm::Value* is_complex,
                                                   llvm::Value* base_type) {
    // If already complex, unpack and return the struct {double, double}.
    // If int/double, promote to complex(value, 0.0).
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* is_complex_bb = llvm::BasicBlock::Create(ctx_.context(), "cvt_is_complex", func);
    llvm::BasicBlock* not_complex_bb = llvm::BasicBlock::Create(ctx_.context(), "cvt_not_complex", func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "cvt_complex_merge", func);

    ctx_.builder().CreateCondBr(is_complex, is_complex_bb, not_complex_bb);

    // Already complex: unpack
    ctx_.builder().SetInsertPoint(is_complex_bb);
    llvm::Value* existing_complex = complex_.unpackComplexFromTagged(operand);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* is_complex_exit = ctx_.builder().GetInsertBlock();

    // Not complex: promote real/int/bignum to complex(value, 0.0)
    ctx_.builder().SetInsertPoint(not_complex_bb);
    // Check for bignum: HEAP_PTR with BIGNUM subtype → call eshkol_bignum_to_double
    llvm::Value* is_heap = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    llvm::BasicBlock* bignum_check_bb = llvm::BasicBlock::Create(ctx_.context(), "bn_to_complex_check", func);
    llvm::BasicBlock* normal_convert_bb = llvm::BasicBlock::Create(ctx_.context(), "normal_to_complex", func);
    llvm::BasicBlock* convert_merge_bb = llvm::BasicBlock::Create(ctx_.context(), "to_complex_merge", func);

    ctx_.builder().CreateCondBr(is_heap, bignum_check_bb, normal_convert_bb);

    // Bignum → double path (intentionally lossy, complex uses floating-point)
    ctx_.builder().SetInsertPoint(bignum_check_bb);
    llvm::Value* ptr_int = tagged_.unpackInt64(operand);
    llvm::Value* bn_ptr = ctx_.builder().CreateIntToPtr(ptr_int, llvm::PointerType::get(ctx_.context(), 0));
    llvm::FunctionType* bn_to_dbl_type = llvm::FunctionType::get(ctx_.doubleType(),
        {llvm::PointerType::get(ctx_.context(), 0)}, false);
    llvm::FunctionCallee bn_to_dbl_fn = ctx_.module().getOrInsertFunction("eshkol_bignum_to_double", bn_to_dbl_type);
    llvm::Value* bn_as_double = ctx_.builder().CreateCall(bn_to_dbl_fn, {bn_ptr});
    ctx_.builder().CreateBr(convert_merge_bb);
    llvm::BasicBlock* bn_convert_exit = ctx_.builder().GetInsertBlock();

    // Normal int/double path
    ctx_.builder().SetInsertPoint(normal_convert_bb);
    llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* normal_as_double = ctx_.builder().CreateSelect(is_double,
        tagged_.unpackDouble(operand),
        ctx_.builder().CreateSIToFP(tagged_.unpackInt64(operand), ctx_.doubleType()));
    ctx_.builder().CreateBr(convert_merge_bb);
    llvm::BasicBlock* normal_convert_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(convert_merge_bb);
    llvm::PHINode* as_double = ctx_.builder().CreatePHI(ctx_.doubleType(), 2);
    as_double->addIncoming(bn_as_double, bn_convert_exit);
    as_double->addIncoming(normal_as_double, normal_convert_exit);

    llvm::Value* zero_imag = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* promoted_complex = complex_.createComplex(as_double, zero_imag);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* not_complex_exit = ctx_.builder().GetInsertBlock();

    // Merge — result is a complex struct {double, double}, not a pointer
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.complexNumberType(), 2, "complex_phi");
    phi->addIncoming(existing_complex, is_complex_exit);
    phi->addIncoming(promoted_complex, not_complex_exit);

    return phi;
}

// === Bignum Runtime Dispatch ===

// Declare eshkol_is_bignum_tagged(const tagged_value_t*) -> bool
static llvm::Function* getIsBignumTaggedFunc(CodegenContext& ctx) {
    llvm::Function* func = ctx.module().getFunction("eshkol_is_bignum_tagged");
    if (!func) {
        llvm::FunctionType* fn_type = llvm::FunctionType::get(
            ctx.int1Type(), {ctx.ptrType()}, false);
        func = llvm::Function::Create(fn_type,
            llvm::Function::ExternalLinkage, "eshkol_is_bignum_tagged", &ctx.module());
    }
    return func;
}

// Declare eshkol_bignum_binary_tagged(arena, left*, right*, op, result*)
static llvm::Function* getBignumBinaryTaggedFunc(CodegenContext& ctx) {
    llvm::Function* func = ctx.module().getFunction("eshkol_bignum_binary_tagged");
    if (!func) {
        llvm::FunctionType* fn_type = llvm::FunctionType::get(
            llvm::Type::getVoidTy(ctx.context()),
            {ctx.ptrType(), ctx.ptrType(), ctx.ptrType(),
             ctx.int32Type(), ctx.ptrType()}, false);
        func = llvm::Function::Create(fn_type,
            llvm::Function::ExternalLinkage, "eshkol_bignum_binary_tagged", &ctx.module());
    }
    return func;
}

// Declare eshkol_bignum_compare_tagged(left*, right*, op, result*)
static llvm::Function* getBignumCompareTaggedFunc(CodegenContext& ctx) {
    llvm::Function* func = ctx.module().getFunction("eshkol_bignum_compare_tagged");
    if (!func) {
        llvm::FunctionType* fn_type = llvm::FunctionType::get(
            llvm::Type::getVoidTy(ctx.context()),
            {ctx.ptrType(), ctx.ptrType(), ctx.int32Type(), ctx.ptrType()}, false);
        func = llvm::Function::Create(fn_type,
            llvm::Function::ExternalLinkage, "eshkol_bignum_compare_tagged", &ctx.module());
    }
    return func;
}

// === Bignum Codegen Helpers ===

llvm::Value* ArithmeticCodegen::emitIsBignumCheck(llvm::Value* left, llvm::Value* right) {
    // Alloca in entry block to avoid stack growth in loops (critical for TCO)
    llvm::Function* fn = ctx_.builder().GetInsertBlock()->getParent();
    llvm::IRBuilder<> entry_builder(&fn->getEntryBlock(), fn->getEntryBlock().begin());
    llvm::Value* left_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "bn_chk_l");
    llvm::Value* right_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "bn_chk_r");
    ctx_.builder().CreateStore(left, left_alloca);
    ctx_.builder().CreateStore(right, right_alloca);
    llvm::Function* is_bn = getIsBignumTaggedFunc(ctx_);
    llvm::Value* l_is = ctx_.builder().CreateCall(is_bn, {left_alloca}, "l_is_bn");
    llvm::Value* r_is = ctx_.builder().CreateCall(is_bn, {right_alloca}, "r_is_bn");
    return ctx_.builder().CreateOr(l_is, r_is, "any_bn");
}

llvm::Value* ArithmeticCodegen::emitBignumBinaryCall(llvm::Value* left, llvm::Value* right, int op_code) {
    // Alloca in entry block to avoid stack growth in loops (critical for TCO)
    llvm::Function* fn = ctx_.builder().GetInsertBlock()->getParent();
    llvm::IRBuilder<> entry_builder(&fn->getEntryBlock(), fn->getEntryBlock().begin());
    llvm::Value* left_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "bn_l");
    llvm::Value* right_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "bn_r");
    llvm::Value* result_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "bn_res");
    ctx_.builder().CreateStore(left, left_alloca);
    ctx_.builder().CreateStore(right, right_alloca);
    llvm::Value* arena_ptr = getArenaPtr(ctx_);
    ctx_.builder().CreateCall(getBignumBinaryTaggedFunc(ctx_), {
        arena_ptr, left_alloca, right_alloca,
        llvm::ConstantInt::get(ctx_.int32Type(), op_code),
        result_alloca
    });
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_alloca, "bn_result");
}

llvm::Value* ArithmeticCodegen::emitBignumCompareCall(llvm::Value* left, llvm::Value* right, int op_code) {
    // Alloca in entry block to avoid stack growth in loops (critical for TCO)
    llvm::Function* fn = ctx_.builder().GetInsertBlock()->getParent();
    llvm::IRBuilder<> entry_builder(&fn->getEntryBlock(), fn->getEntryBlock().begin());
    llvm::Value* left_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "bn_cmp_l");
    llvm::Value* right_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "bn_cmp_r");
    llvm::Value* result_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "bn_cmp_res");
    ctx_.builder().CreateStore(left, left_alloca);
    ctx_.builder().CreateStore(right, right_alloca);
    ctx_.builder().CreateCall(getBignumCompareTaggedFunc(ctx_), {
        left_alloca, right_alloca,
        llvm::ConstantInt::get(ctx_.int32Type(), op_code),
        result_alloca
    });
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_alloca, "bn_cmp_result");
}

// === Rational Codegen Helpers ===

// Declare eshkol_is_rational_tagged_ptr(val*) -> i32
static llvm::Function* getIsRationalTaggedFunc(CodegenContext& ctx) {
    llvm::Function* func = ctx.module().getFunction("eshkol_is_rational_tagged_ptr");
    if (!func) {
        llvm::FunctionType* fn_type = llvm::FunctionType::get(
            llvm::Type::getInt32Ty(ctx.context()),
            {ctx.ptrType()}, false);
        func = llvm::Function::Create(fn_type,
            llvm::Function::ExternalLinkage, "eshkol_is_rational_tagged_ptr", &ctx.module());
    }
    return func;
}

// Declare eshkol_rational_binary_tagged_ptr(arena, a*, b*, op, result*)
static llvm::Function* getRationalBinaryTaggedFunc(CodegenContext& ctx) {
    llvm::Function* func = ctx.module().getFunction("eshkol_rational_binary_tagged_ptr");
    if (!func) {
        llvm::FunctionType* fn_type = llvm::FunctionType::get(
            llvm::Type::getVoidTy(ctx.context()),
            {ctx.ptrType(), ctx.ptrType(), ctx.ptrType(),
             ctx.int32Type(), ctx.ptrType()}, false);
        func = llvm::Function::Create(fn_type,
            llvm::Function::ExternalLinkage, "eshkol_rational_binary_tagged_ptr", &ctx.module());
    }
    return func;
}

// Declare eshkol_rational_create(arena, num, denom) -> void*
static llvm::Function* getRationalCreateFunc(CodegenContext& ctx) {
    llvm::Function* func = ctx.module().getFunction("eshkol_rational_create");
    if (!func) {
        llvm::FunctionType* fn_type = llvm::FunctionType::get(
            ctx.ptrType(),
            {ctx.ptrType(), ctx.int64Type(), ctx.int64Type()}, false);
        func = llvm::Function::Create(fn_type,
            llvm::Function::ExternalLinkage, "eshkol_rational_create", &ctx.module());
    }
    return func;
}

llvm::Value* ArithmeticCodegen::emitIsRationalCheck(llvm::Value* left, llvm::Value* right) {
    llvm::Function* fn = ctx_.builder().GetInsertBlock()->getParent();
    llvm::IRBuilder<> entry_builder(&fn->getEntryBlock(), fn->getEntryBlock().begin());
    llvm::Value* left_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "rat_chk_l");
    llvm::Value* right_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "rat_chk_r");
    ctx_.builder().CreateStore(left, left_alloca);
    ctx_.builder().CreateStore(right, right_alloca);
    llvm::Function* is_rat = getIsRationalTaggedFunc(ctx_);
    llvm::Value* l_is = ctx_.builder().CreateCall(is_rat, {left_alloca}, "l_is_rat");
    llvm::Value* r_is = ctx_.builder().CreateCall(is_rat, {right_alloca}, "r_is_rat");
    llvm::Value* l_bool = ctx_.builder().CreateICmpNE(l_is, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    llvm::Value* r_bool = ctx_.builder().CreateICmpNE(r_is, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    return ctx_.builder().CreateOr(l_bool, r_bool, "any_rat");
}

// Declare eshkol_rational_compare_tagged_ptr(arena, a*, b*, op, result*)
static llvm::Function* getRationalCompareTaggedFunc(CodegenContext& ctx) {
    llvm::Function* func = ctx.module().getFunction("eshkol_rational_compare_tagged_ptr");
    if (!func) {
        llvm::FunctionType* fn_type = llvm::FunctionType::get(
            llvm::Type::getVoidTy(ctx.context()),
            {ctx.ptrType(), ctx.ptrType(), ctx.ptrType(),
             ctx.int32Type(), ctx.ptrType()}, false);
        func = llvm::Function::Create(fn_type,
            llvm::Function::ExternalLinkage, "eshkol_rational_compare_tagged_ptr", &ctx.module());
    }
    return func;
}

llvm::Value* ArithmeticCodegen::emitRationalCompareCall(llvm::Value* left, llvm::Value* right, int op_code) {
    llvm::Function* fn = ctx_.builder().GetInsertBlock()->getParent();
    llvm::IRBuilder<> entry_builder(&fn->getEntryBlock(), fn->getEntryBlock().begin());
    llvm::Value* left_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "ratcmp_l");
    llvm::Value* right_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "ratcmp_r");
    llvm::Value* result_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "ratcmp_res");
    ctx_.builder().CreateStore(left, left_alloca);
    ctx_.builder().CreateStore(right, right_alloca);
    llvm::Value* arena_ptr = getArenaPtr(ctx_);
    ctx_.builder().CreateCall(getRationalCompareTaggedFunc(ctx_), {
        arena_ptr, left_alloca, right_alloca,
        llvm::ConstantInt::get(ctx_.int32Type(), op_code),
        result_alloca
    });
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_alloca, "ratcmp_result");
}

llvm::Value* ArithmeticCodegen::emitRationalBinaryCall(llvm::Value* left, llvm::Value* right, int op_code) {
    llvm::Function* fn = ctx_.builder().GetInsertBlock()->getParent();
    llvm::IRBuilder<> entry_builder(&fn->getEntryBlock(), fn->getEntryBlock().begin());
    llvm::Value* left_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "rat_l");
    llvm::Value* right_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "rat_r");
    llvm::Value* result_alloca = entry_builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "rat_res");
    ctx_.builder().CreateStore(left, left_alloca);
    ctx_.builder().CreateStore(right, right_alloca);
    llvm::Value* arena_ptr = getArenaPtr(ctx_);
    ctx_.builder().CreateCall(getRationalBinaryTaggedFunc(ctx_), {
        arena_ptr, left_alloca, right_alloca,
        llvm::ConstantInt::get(ctx_.int32Type(), op_code),
        result_alloca
    });
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_alloca, "rat_result");
}

// === Polymorphic Addition ===

llvm::Value* ArithmeticCodegen::add(llvm::Value* left, llvm::Value* right) {
    if (!left || !right) {
        eshkol_error("arithmetic: null operand in add (left=%p, right=%p)", (void*)left, (void*)right);
        return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
    }

    return withADBinaryDispatch(left, right, 2 /*AD_NODE_ADD*/, [&]() -> llvm::Value* {
        // Re-extract types inside lambda (handler already checked AD)
        llvm::Value* left_type = tagged_.getType(left);
        llvm::Value* right_type = tagged_.getType(right);
        llvm::Value* left_base = tagged_.getBaseType(left_type);
        llvm::Value* right_base = tagged_.getBaseType(right_type);

        // Check for vector/tensor types
        llvm::Value* any_heap = ctx_.builder().CreateOr(
            ctx_.builder().CreateICmpEQ(left_base,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR)),
            ctx_.builder().CreateICmpEQ(right_base,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR)));

        llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* bignum_path = llvm::BasicBlock::Create(ctx_.context(), "add_bignum", func);
        llvm::BasicBlock* check_heap = llvm::BasicBlock::Create(ctx_.context(), "add_check_heap", func);
        llvm::BasicBlock* vector_path = llvm::BasicBlock::Create(ctx_.context(), "add_vector", func);
        llvm::BasicBlock* check_dual = llvm::BasicBlock::Create(ctx_.context(), "add_check_dual", func);
        llvm::BasicBlock* dual_path = llvm::BasicBlock::Create(ctx_.context(), "add_dual", func);
        llvm::BasicBlock* check_complex = llvm::BasicBlock::Create(ctx_.context(), "add_check_complex", func);
        llvm::BasicBlock* complex_path = llvm::BasicBlock::Create(ctx_.context(), "add_complex", func);
        llvm::BasicBlock* check_double = llvm::BasicBlock::Create(ctx_.context(), "add_check_double", func);
        llvm::BasicBlock* double_path = llvm::BasicBlock::Create(ctx_.context(), "add_double", func);
        llvm::BasicBlock* int_path = llvm::BasicBlock::Create(ctx_.context(), "add_int", func);
        llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "add_merge", func);

        // Check bignum first via safe runtime call
        llvm::Value* any_bignum = emitIsBignumCheck(left, right);
        llvm::BasicBlock* check_rational = llvm::BasicBlock::Create(ctx_.context(), "add_check_rational", func);
        ctx_.builder().CreateCondBr(any_bignum, bignum_path, check_rational);

        // Bignum path
        ctx_.builder().SetInsertPoint(bignum_path);
        llvm::Value* bn_add_tagged = emitBignumBinaryCall(left, right, 0);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* bignum_exit = ctx_.builder().GetInsertBlock();

        // Check rational
        ctx_.builder().SetInsertPoint(check_rational);
        llvm::BasicBlock* rational_path = llvm::BasicBlock::Create(ctx_.context(), "add_rational", func);
        llvm::Value* any_rational = emitIsRationalCheck(left, right);
        ctx_.builder().CreateCondBr(any_rational, rational_path, check_heap);

        // Rational path
        ctx_.builder().SetInsertPoint(rational_path);
        llvm::Value* rat_add_tagged = emitRationalBinaryCall(left, right, 0);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* rational_exit = ctx_.builder().GetInsertBlock();

        // Check for vector/tensor heap pointers
        ctx_.builder().SetInsertPoint(check_heap);
        ctx_.builder().CreateCondBr(any_heap, vector_path, check_dual);

        // Vector/tensor path
        ctx_.builder().SetInsertPoint(vector_path);
        llvm::Value* vec_result = tensor_.tensorArithmeticInternal(left, right, "add");
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* vector_exit = ctx_.builder().GetInsertBlock();

        // Check for dual numbers
        ctx_.builder().SetInsertPoint(check_dual);
        llvm::Value* left_is_dual = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
        llvm::Value* right_is_dual = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
        llvm::Value* any_dual = ctx_.builder().CreateOr(left_is_dual, right_is_dual);
        ctx_.builder().CreateCondBr(any_dual, dual_path, check_complex);

        // Dual number path
        ctx_.builder().SetInsertPoint(dual_path);
        llvm::Value* left_is_double = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* right_is_double = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* left_dual = convertToDual(left, left_is_dual, left_is_double);
        llvm::Value* right_dual = convertToDual(right, right_is_dual, right_is_double);
        llvm::Value* dual_result = autodiff_.dualAdd(left_dual, right_dual);
        llvm::Value* dual_tagged = autodiff_.packDualToTagged(dual_result);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* dual_exit = ctx_.builder().GetInsertBlock();

        // Check for complex numbers
        ctx_.builder().SetInsertPoint(check_complex);
        llvm::Value* left_is_complex = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_COMPLEX));
        llvm::Value* right_is_complex = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_COMPLEX));
        llvm::Value* any_complex = ctx_.builder().CreateOr(left_is_complex, right_is_complex);
        ctx_.builder().CreateCondBr(any_complex, complex_path, check_double);

        // Complex number path
        ctx_.builder().SetInsertPoint(complex_path);
        llvm::Value* left_z = convertToComplex(left, left_is_complex, left_base);
        llvm::Value* right_z = convertToComplex(right, right_is_complex, right_base);
        llvm::Value* complex_sum = complex_.complexAdd(left_z, right_z);
        llvm::Value* complex_tagged = complex_.packComplexToTagged(complex_sum);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* complex_exit = ctx_.builder().GetInsertBlock();

        // Check for doubles
        ctx_.builder().SetInsertPoint(check_double);
        llvm::Value* left_is_dbl = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* right_is_dbl = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* any_double = ctx_.builder().CreateOr(left_is_dbl, right_is_dbl);
        ctx_.builder().CreateCondBr(any_double, double_path, int_path);

        // Double path
        ctx_.builder().SetInsertPoint(double_path);
        llvm::Value* left_dbl = ctx_.builder().CreateSelect(left_is_dbl,
            tagged_.unpackDouble(left),
            ctx_.builder().CreateSIToFP(tagged_.unpackInt64(left), ctx_.doubleType()));
        llvm::Value* right_dbl = ctx_.builder().CreateSelect(right_is_dbl,
            tagged_.unpackDouble(right),
            ctx_.builder().CreateSIToFP(tagged_.unpackInt64(right), ctx_.doubleType()));
        llvm::Value* dbl_result = ctx_.builder().CreateFAdd(left_dbl, right_dbl);
        llvm::Value* dbl_tagged = tagged_.packDouble(dbl_result);
        ctx_.builder().CreateBr(merge);

        // Integer path with overflow detection
        ctx_.builder().SetInsertPoint(int_path);
        llvm::Value* left_int = tagged_.unpackInt64(left);
        llvm::Value* right_int = tagged_.unpackInt64(right);
        llvm::Function* sadd_ovf = ESHKOL_GET_INTRINSIC(
            &ctx_.module(), llvm::Intrinsic::sadd_with_overflow, {ctx_.int64Type()});
        llvm::Value* add_ovf_result = ctx_.builder().CreateCall(sadd_ovf, {left_int, right_int});
        llvm::Value* add_int_val = ctx_.builder().CreateExtractValue(add_ovf_result, 0);
        llvm::Value* add_overflowed = ctx_.builder().CreateExtractValue(add_ovf_result, 1);

        llvm::BasicBlock* add_ok = llvm::BasicBlock::Create(ctx_.context(), "add_int_ok", func);
        llvm::BasicBlock* add_ovf_bb = llvm::BasicBlock::Create(ctx_.context(), "add_int_ovf", func);
        ctx_.builder().CreateCondBr(add_overflowed, add_ovf_bb, add_ok);

        // Overflow: promote to bignum
        ctx_.builder().SetInsertPoint(add_ovf_bb);
        llvm::Value* add_promoted_tagged = emitBignumPromotion(ctx_, tagged_, left_int, right_int, 0);
        ctx_.builder().CreateBr(merge);

        // No overflow: pack result
        ctx_.builder().SetInsertPoint(add_ok);
        llvm::Value* int_tagged = tagged_.packInt64(add_int_val, true);
        ctx_.builder().CreateBr(merge);

        // Merge all non-AD paths
        ctx_.builder().SetInsertPoint(merge);
        llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 8, "add_result");
        phi->addIncoming(bn_add_tagged, bignum_exit);
        phi->addIncoming(rat_add_tagged, rational_exit);
        phi->addIncoming(vec_result, vector_exit);
        phi->addIncoming(dual_tagged, dual_exit);
        phi->addIncoming(complex_tagged, complex_exit);
        phi->addIncoming(dbl_tagged, double_path);
        phi->addIncoming(add_promoted_tagged, add_ovf_bb);
        phi->addIncoming(int_tagged, add_ok);

        return phi;
    });
}

// === Polymorphic Subtraction ===

llvm::Value* ArithmeticCodegen::sub(llvm::Value* left, llvm::Value* right) {
    if (!left || !right) {
        eshkol_error("arithmetic: null operand in sub (left=%p, right=%p)", (void*)left, (void*)right);
        return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
    }

    return withADBinaryDispatch(left, right, 3 /*AD_NODE_SUB*/, [&]() -> llvm::Value* {
        // Re-extract types inside lambda (handler already checked AD)
        llvm::Value* left_type = tagged_.getType(left);
        llvm::Value* right_type = tagged_.getType(right);
        llvm::Value* left_base = tagged_.getBaseType(left_type);
        llvm::Value* right_base = tagged_.getBaseType(right_type);

        // Check for vector/tensor types
        llvm::Value* any_heap = ctx_.builder().CreateOr(
            ctx_.builder().CreateICmpEQ(left_base,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR)),
            ctx_.builder().CreateICmpEQ(right_base,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR)));

        llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* bignum_path = llvm::BasicBlock::Create(ctx_.context(), "sub_bignum", func);
        llvm::BasicBlock* check_heap = llvm::BasicBlock::Create(ctx_.context(), "sub_check_heap", func);
        llvm::BasicBlock* vector_path = llvm::BasicBlock::Create(ctx_.context(), "sub_vector", func);
        llvm::BasicBlock* check_dual = llvm::BasicBlock::Create(ctx_.context(), "sub_check_dual", func);
        llvm::BasicBlock* dual_path = llvm::BasicBlock::Create(ctx_.context(), "sub_dual", func);
        llvm::BasicBlock* check_complex = llvm::BasicBlock::Create(ctx_.context(), "sub_check_complex", func);
        llvm::BasicBlock* complex_path = llvm::BasicBlock::Create(ctx_.context(), "sub_complex", func);
        llvm::BasicBlock* check_double = llvm::BasicBlock::Create(ctx_.context(), "sub_check_double", func);
        llvm::BasicBlock* double_path = llvm::BasicBlock::Create(ctx_.context(), "sub_double", func);
        llvm::BasicBlock* int_path = llvm::BasicBlock::Create(ctx_.context(), "sub_int", func);
        llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "sub_merge", func);

        // Check bignum first via safe runtime call
        llvm::Value* any_bignum = emitIsBignumCheck(left, right);
        llvm::BasicBlock* check_rational = llvm::BasicBlock::Create(ctx_.context(), "sub_check_rational", func);
        ctx_.builder().CreateCondBr(any_bignum, bignum_path, check_rational);

        // Bignum path
        ctx_.builder().SetInsertPoint(bignum_path);
        llvm::Value* bn_sub_tagged = emitBignumBinaryCall(left, right, 1);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* bignum_exit = ctx_.builder().GetInsertBlock();

        // Check rational
        ctx_.builder().SetInsertPoint(check_rational);
        llvm::BasicBlock* rational_path = llvm::BasicBlock::Create(ctx_.context(), "sub_rational", func);
        llvm::Value* any_rational = emitIsRationalCheck(left, right);
        ctx_.builder().CreateCondBr(any_rational, rational_path, check_heap);

        // Rational path
        ctx_.builder().SetInsertPoint(rational_path);
        llvm::Value* rat_sub_tagged = emitRationalBinaryCall(left, right, 1);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* rational_exit = ctx_.builder().GetInsertBlock();

        // Check for vector/tensor heap pointers
        ctx_.builder().SetInsertPoint(check_heap);
        ctx_.builder().CreateCondBr(any_heap, vector_path, check_dual);

        // Vector/tensor path
        ctx_.builder().SetInsertPoint(vector_path);
        llvm::Value* vec_result = tensor_.tensorArithmeticInternal(left, right, "sub");
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* vector_exit = ctx_.builder().GetInsertBlock();

        // Check for dual numbers
        ctx_.builder().SetInsertPoint(check_dual);
        llvm::Value* left_is_dual = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
        llvm::Value* right_is_dual = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
        llvm::Value* any_dual = ctx_.builder().CreateOr(left_is_dual, right_is_dual);
        ctx_.builder().CreateCondBr(any_dual, dual_path, check_complex);

        // Dual number path
        ctx_.builder().SetInsertPoint(dual_path);
        llvm::Value* left_is_double = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* right_is_double = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* left_dual = convertToDual(left, left_is_dual, left_is_double);
        llvm::Value* right_dual = convertToDual(right, right_is_dual, right_is_double);
        llvm::Value* dual_result = autodiff_.dualSub(left_dual, right_dual);
        llvm::Value* dual_tagged = autodiff_.packDualToTagged(dual_result);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* dual_exit = ctx_.builder().GetInsertBlock();

        // Check for complex numbers
        ctx_.builder().SetInsertPoint(check_complex);
        llvm::Value* left_is_complex = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_COMPLEX));
        llvm::Value* right_is_complex = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_COMPLEX));
        llvm::Value* any_complex = ctx_.builder().CreateOr(left_is_complex, right_is_complex);
        ctx_.builder().CreateCondBr(any_complex, complex_path, check_double);

        // Complex number path
        ctx_.builder().SetInsertPoint(complex_path);
        llvm::Value* left_z = convertToComplex(left, left_is_complex, left_base);
        llvm::Value* right_z = convertToComplex(right, right_is_complex, right_base);
        llvm::Value* complex_diff = complex_.complexSub(left_z, right_z);
        llvm::Value* complex_tagged = complex_.packComplexToTagged(complex_diff);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* complex_exit = ctx_.builder().GetInsertBlock();

        // Check for doubles
        ctx_.builder().SetInsertPoint(check_double);
        llvm::Value* left_is_dbl = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* right_is_dbl = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* any_double = ctx_.builder().CreateOr(left_is_dbl, right_is_dbl);
        ctx_.builder().CreateCondBr(any_double, double_path, int_path);

        // Double path
        ctx_.builder().SetInsertPoint(double_path);
        llvm::Value* left_dbl = ctx_.builder().CreateSelect(left_is_dbl,
            tagged_.unpackDouble(left),
            ctx_.builder().CreateSIToFP(tagged_.unpackInt64(left), ctx_.doubleType()));
        llvm::Value* right_dbl = ctx_.builder().CreateSelect(right_is_dbl,
            tagged_.unpackDouble(right),
            ctx_.builder().CreateSIToFP(tagged_.unpackInt64(right), ctx_.doubleType()));
        llvm::Value* dbl_result = ctx_.builder().CreateFSub(left_dbl, right_dbl);
        llvm::Value* dbl_tagged = tagged_.packDouble(dbl_result);
        ctx_.builder().CreateBr(merge);

        // Integer path with overflow detection
        ctx_.builder().SetInsertPoint(int_path);
        llvm::Value* left_int = tagged_.unpackInt64(left);
        llvm::Value* right_int = tagged_.unpackInt64(right);
        llvm::Function* ssub_ovf = ESHKOL_GET_INTRINSIC(
            &ctx_.module(), llvm::Intrinsic::ssub_with_overflow, {ctx_.int64Type()});
        llvm::Value* sub_ovf_result = ctx_.builder().CreateCall(ssub_ovf, {left_int, right_int});
        llvm::Value* sub_int_val = ctx_.builder().CreateExtractValue(sub_ovf_result, 0);
        llvm::Value* sub_overflowed = ctx_.builder().CreateExtractValue(sub_ovf_result, 1);

        llvm::BasicBlock* sub_ok = llvm::BasicBlock::Create(ctx_.context(), "sub_int_ok", func);
        llvm::BasicBlock* sub_ovf_bb = llvm::BasicBlock::Create(ctx_.context(), "sub_int_ovf", func);
        ctx_.builder().CreateCondBr(sub_overflowed, sub_ovf_bb, sub_ok);

        // Overflow: promote to bignum (exact) — R7RS requires arbitrary-precision integers
        ctx_.builder().SetInsertPoint(sub_ovf_bb);
        llvm::Value* sub_promoted_tagged = emitBignumPromotion(ctx_, tagged_, left_int, right_int, 1);
        ctx_.builder().CreateBr(merge);

        // No overflow: pack result
        ctx_.builder().SetInsertPoint(sub_ok);
        llvm::Value* int_tagged = tagged_.packInt64(sub_int_val, true);
        ctx_.builder().CreateBr(merge);

        // Merge all non-AD paths
        ctx_.builder().SetInsertPoint(merge);
        llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 8, "sub_result");
        phi->addIncoming(bn_sub_tagged, bignum_exit);
        phi->addIncoming(rat_sub_tagged, rational_exit);
        phi->addIncoming(vec_result, vector_exit);
        phi->addIncoming(dual_tagged, dual_exit);
        phi->addIncoming(complex_tagged, complex_exit);
        phi->addIncoming(dbl_tagged, double_path);
        phi->addIncoming(sub_promoted_tagged, sub_ovf_bb);
        phi->addIncoming(int_tagged, sub_ok);

        return phi;
    });
}

// === Polymorphic Multiplication ===

llvm::Value* ArithmeticCodegen::mul(llvm::Value* left, llvm::Value* right) {
    if (!left || !right) {
        eshkol_error("arithmetic: null operand in mul (left=%p, right=%p)", (void*)left, (void*)right);
        return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
    }

    return withADBinaryDispatch(left, right, 4 /*AD_NODE_MUL*/, [&]() -> llvm::Value* {
        // Re-extract types inside lambda (handler already checked AD)
        llvm::Value* left_type = tagged_.getType(left);
        llvm::Value* right_type = tagged_.getType(right);
        llvm::Value* left_base = tagged_.getBaseType(left_type);
        llvm::Value* right_base = tagged_.getBaseType(right_type);

        // Check for vector/tensor types
        llvm::Value* any_heap = ctx_.builder().CreateOr(
            ctx_.builder().CreateICmpEQ(left_base,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR)),
            ctx_.builder().CreateICmpEQ(right_base,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR)));

        llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* bignum_path = llvm::BasicBlock::Create(ctx_.context(), "mul_bignum", func);
        llvm::BasicBlock* check_heap = llvm::BasicBlock::Create(ctx_.context(), "mul_check_heap", func);
        llvm::BasicBlock* vector_path = llvm::BasicBlock::Create(ctx_.context(), "mul_vector", func);
        llvm::BasicBlock* check_dual = llvm::BasicBlock::Create(ctx_.context(), "mul_check_dual", func);
        llvm::BasicBlock* dual_path = llvm::BasicBlock::Create(ctx_.context(), "mul_dual", func);
        llvm::BasicBlock* check_complex = llvm::BasicBlock::Create(ctx_.context(), "mul_check_complex", func);
        llvm::BasicBlock* complex_path = llvm::BasicBlock::Create(ctx_.context(), "mul_complex", func);
        llvm::BasicBlock* check_double = llvm::BasicBlock::Create(ctx_.context(), "mul_check_double", func);
        llvm::BasicBlock* double_path = llvm::BasicBlock::Create(ctx_.context(), "mul_double", func);
        llvm::BasicBlock* int_path = llvm::BasicBlock::Create(ctx_.context(), "mul_int", func);
        llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "mul_merge", func);

        // Check bignum first via safe runtime call
        llvm::Value* any_bignum = emitIsBignumCheck(left, right);
        llvm::BasicBlock* check_rational = llvm::BasicBlock::Create(ctx_.context(), "mul_check_rational", func);
        ctx_.builder().CreateCondBr(any_bignum, bignum_path, check_rational);

        // Bignum path
        ctx_.builder().SetInsertPoint(bignum_path);
        llvm::Value* bn_mul_tagged = emitBignumBinaryCall(left, right, 2);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* bignum_exit = ctx_.builder().GetInsertBlock();

        // Check rational
        ctx_.builder().SetInsertPoint(check_rational);
        llvm::BasicBlock* rational_path = llvm::BasicBlock::Create(ctx_.context(), "mul_rational", func);
        llvm::Value* any_rational = emitIsRationalCheck(left, right);
        ctx_.builder().CreateCondBr(any_rational, rational_path, check_heap);

        // Rational path
        ctx_.builder().SetInsertPoint(rational_path);
        llvm::Value* rat_mul_tagged = emitRationalBinaryCall(left, right, 2);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* rational_exit = ctx_.builder().GetInsertBlock();

        // Check for vector/tensor heap pointers
        ctx_.builder().SetInsertPoint(check_heap);
        ctx_.builder().CreateCondBr(any_heap, vector_path, check_dual);

        // Vector/tensor path
        ctx_.builder().SetInsertPoint(vector_path);
        llvm::Value* vec_result = tensor_.tensorArithmeticInternal(left, right, "mul");
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* vector_exit = ctx_.builder().GetInsertBlock();

        // Check for dual numbers
        ctx_.builder().SetInsertPoint(check_dual);
        llvm::Value* left_is_dual = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
        llvm::Value* right_is_dual = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
        llvm::Value* any_dual = ctx_.builder().CreateOr(left_is_dual, right_is_dual);
        ctx_.builder().CreateCondBr(any_dual, dual_path, check_complex);

        // Dual number path
        ctx_.builder().SetInsertPoint(dual_path);
        llvm::Value* left_is_double = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* right_is_double = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* left_dual = convertToDual(left, left_is_dual, left_is_double);
        llvm::Value* right_dual = convertToDual(right, right_is_dual, right_is_double);
        llvm::Value* dual_result = autodiff_.dualMul(left_dual, right_dual);
        llvm::Value* dual_tagged = autodiff_.packDualToTagged(dual_result);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* dual_exit = ctx_.builder().GetInsertBlock();

        // Check for complex numbers
        ctx_.builder().SetInsertPoint(check_complex);
        llvm::Value* left_is_complex = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_COMPLEX));
        llvm::Value* right_is_complex = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_COMPLEX));
        llvm::Value* any_complex = ctx_.builder().CreateOr(left_is_complex, right_is_complex);
        ctx_.builder().CreateCondBr(any_complex, complex_path, check_double);

        // Complex number path
        ctx_.builder().SetInsertPoint(complex_path);
        llvm::Value* left_z = convertToComplex(left, left_is_complex, left_base);
        llvm::Value* right_z = convertToComplex(right, right_is_complex, right_base);
        llvm::Value* complex_prod = complex_.complexMul(left_z, right_z);
        llvm::Value* complex_tagged = complex_.packComplexToTagged(complex_prod);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* complex_exit = ctx_.builder().GetInsertBlock();

        // Check for doubles
        ctx_.builder().SetInsertPoint(check_double);
        llvm::Value* left_is_dbl = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* right_is_dbl = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* any_double = ctx_.builder().CreateOr(left_is_dbl, right_is_dbl);
        ctx_.builder().CreateCondBr(any_double, double_path, int_path);

        // Double path
        ctx_.builder().SetInsertPoint(double_path);
        llvm::Value* left_dbl = ctx_.builder().CreateSelect(left_is_dbl,
            tagged_.unpackDouble(left),
            ctx_.builder().CreateSIToFP(tagged_.unpackInt64(left), ctx_.doubleType()));
        llvm::Value* right_dbl = ctx_.builder().CreateSelect(right_is_dbl,
            tagged_.unpackDouble(right),
            ctx_.builder().CreateSIToFP(tagged_.unpackInt64(right), ctx_.doubleType()));
        llvm::Value* dbl_result = ctx_.builder().CreateFMul(left_dbl, right_dbl);
        llvm::Value* dbl_tagged = tagged_.packDouble(dbl_result);
        ctx_.builder().CreateBr(merge);

        // Integer path with overflow detection
        ctx_.builder().SetInsertPoint(int_path);
        llvm::Value* left_int = tagged_.unpackInt64(left);
        llvm::Value* right_int = tagged_.unpackInt64(right);
        llvm::Function* smul_ovf = ESHKOL_GET_INTRINSIC(
            &ctx_.module(), llvm::Intrinsic::smul_with_overflow, {ctx_.int64Type()});
        llvm::Value* mul_ovf_result = ctx_.builder().CreateCall(smul_ovf, {left_int, right_int});
        llvm::Value* mul_int_val = ctx_.builder().CreateExtractValue(mul_ovf_result, 0);
        llvm::Value* mul_overflowed = ctx_.builder().CreateExtractValue(mul_ovf_result, 1);

        llvm::BasicBlock* mul_ok = llvm::BasicBlock::Create(ctx_.context(), "mul_int_ok", func);
        llvm::BasicBlock* mul_ovf_bb = llvm::BasicBlock::Create(ctx_.context(), "mul_int_ovf", func);
        ctx_.builder().CreateCondBr(mul_overflowed, mul_ovf_bb, mul_ok);

        // Overflow: promote to bignum (exact) — R7RS requires arbitrary-precision integers
        ctx_.builder().SetInsertPoint(mul_ovf_bb);
        llvm::Value* mul_promoted_tagged = emitBignumPromotion(ctx_, tagged_, left_int, right_int, 2);
        ctx_.builder().CreateBr(merge);

        // No overflow: pack result
        ctx_.builder().SetInsertPoint(mul_ok);
        llvm::Value* int_tagged = tagged_.packInt64(mul_int_val, true);
        ctx_.builder().CreateBr(merge);

        // Merge all non-AD paths
        ctx_.builder().SetInsertPoint(merge);
        llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 8, "mul_result");
        phi->addIncoming(bn_mul_tagged, bignum_exit);
        phi->addIncoming(rat_mul_tagged, rational_exit);
        phi->addIncoming(vec_result, vector_exit);
        phi->addIncoming(dual_tagged, dual_exit);
        phi->addIncoming(complex_tagged, complex_exit);
        phi->addIncoming(dbl_tagged, double_path);
        phi->addIncoming(mul_promoted_tagged, mul_ovf_bb);
        phi->addIncoming(int_tagged, mul_ok);

        return phi;
    });
}

// === Polymorphic Division ===

llvm::Value* ArithmeticCodegen::div(llvm::Value* left, llvm::Value* right) {
    if (!left || !right) {
        eshkol_error("arithmetic: null operand in div (left=%p, right=%p)", (void*)left, (void*)right);
        return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
    }

    return withADBinaryDispatch(left, right, 5 /*AD_NODE_DIV*/, [&]() -> llvm::Value* {
        // Re-extract types inside lambda (handler already checked AD)
        llvm::Value* left_type = tagged_.getType(left);
        llvm::Value* right_type = tagged_.getType(right);
        llvm::Value* left_base = tagged_.getBaseType(left_type);
        llvm::Value* right_base = tagged_.getBaseType(right_type);

        // Check for vector/tensor types
        llvm::Value* any_heap = ctx_.builder().CreateOr(
            ctx_.builder().CreateICmpEQ(left_base,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR)),
            ctx_.builder().CreateICmpEQ(right_base,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR)));

        llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* bignum_path = llvm::BasicBlock::Create(ctx_.context(), "div_bignum", func);
        llvm::BasicBlock* check_heap = llvm::BasicBlock::Create(ctx_.context(), "div_check_heap", func);
        llvm::BasicBlock* vector_path = llvm::BasicBlock::Create(ctx_.context(), "div_vector", func);
        llvm::BasicBlock* check_dual = llvm::BasicBlock::Create(ctx_.context(), "div_check_dual", func);
        llvm::BasicBlock* dual_path = llvm::BasicBlock::Create(ctx_.context(), "div_dual", func);
        llvm::BasicBlock* check_complex = llvm::BasicBlock::Create(ctx_.context(), "div_check_complex", func);
        llvm::BasicBlock* complex_path = llvm::BasicBlock::Create(ctx_.context(), "div_complex", func);
        llvm::BasicBlock* check_double = llvm::BasicBlock::Create(ctx_.context(), "div_check_double", func);
        llvm::BasicBlock* double_path = llvm::BasicBlock::Create(ctx_.context(), "div_double", func);
        llvm::BasicBlock* int_path = llvm::BasicBlock::Create(ctx_.context(), "div_int", func);
        llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "div_merge", func);

        // Check bignum first via safe runtime call
        llvm::Value* any_bignum = emitIsBignumCheck(left, right);
        llvm::BasicBlock* check_rational = llvm::BasicBlock::Create(ctx_.context(), "div_check_rational", func);
        ctx_.builder().CreateCondBr(any_bignum, bignum_path, check_rational);

        // Bignum division: runtime handles exact/inexact logic
        ctx_.builder().SetInsertPoint(bignum_path);
        llvm::Value* bn_div_tagged = emitBignumBinaryCall(left, right, 3);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* bignum_exit = ctx_.builder().GetInsertBlock();

        // Check rational
        ctx_.builder().SetInsertPoint(check_rational);
        llvm::BasicBlock* rational_path = llvm::BasicBlock::Create(ctx_.context(), "div_rational", func);
        llvm::Value* any_rational = emitIsRationalCheck(left, right);
        ctx_.builder().CreateCondBr(any_rational, rational_path, check_heap);

        // Rational path
        ctx_.builder().SetInsertPoint(rational_path);
        llvm::Value* rat_div_tagged = emitRationalBinaryCall(left, right, 3);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* rational_exit = ctx_.builder().GetInsertBlock();

        // Check for vector/tensor heap pointers
        ctx_.builder().SetInsertPoint(check_heap);
        ctx_.builder().CreateCondBr(any_heap, vector_path, check_dual);

        // Vector/tensor path
        ctx_.builder().SetInsertPoint(vector_path);
        llvm::Value* vec_result = tensor_.tensorArithmeticInternal(left, right, "div");
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* vector_exit = ctx_.builder().GetInsertBlock();

        // Check for dual numbers
        ctx_.builder().SetInsertPoint(check_dual);
        llvm::Value* left_is_dual = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
        llvm::Value* right_is_dual = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
        llvm::Value* any_dual = ctx_.builder().CreateOr(left_is_dual, right_is_dual);
        ctx_.builder().CreateCondBr(any_dual, dual_path, check_complex);

        // Dual number path
        ctx_.builder().SetInsertPoint(dual_path);
        llvm::Value* left_is_double = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* right_is_double = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* left_dual = convertToDual(left, left_is_dual, left_is_double);
        llvm::Value* right_dual = convertToDual(right, right_is_dual, right_is_double);
        llvm::Value* dual_result = autodiff_.dualDiv(left_dual, right_dual);
        llvm::Value* dual_tagged = autodiff_.packDualToTagged(dual_result);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* dual_exit = ctx_.builder().GetInsertBlock();

        // Check for complex numbers
        ctx_.builder().SetInsertPoint(check_complex);
        llvm::Value* left_is_complex = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_COMPLEX));
        llvm::Value* right_is_complex = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_COMPLEX));
        llvm::Value* any_complex = ctx_.builder().CreateOr(left_is_complex, right_is_complex);
        ctx_.builder().CreateCondBr(any_complex, complex_path, check_double);

        // Complex number path (uses Smith's formula for overflow safety)
        ctx_.builder().SetInsertPoint(complex_path);
        llvm::Value* left_z = convertToComplex(left, left_is_complex, left_base);
        llvm::Value* right_z = convertToComplex(right, right_is_complex, right_base);
        llvm::Value* complex_quot = complex_.complexDiv(left_z, right_z);
        llvm::Value* complex_tagged = complex_.packComplexToTagged(complex_quot);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* complex_exit = ctx_.builder().GetInsertBlock();

        // Check for doubles
        ctx_.builder().SetInsertPoint(check_double);
        llvm::Value* left_is_dbl = ctx_.builder().CreateICmpEQ(left_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* right_is_dbl = ctx_.builder().CreateICmpEQ(right_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* any_double = ctx_.builder().CreateOr(left_is_dbl, right_is_dbl);
        ctx_.builder().CreateCondBr(any_double, double_path, int_path);

        // Double path
        ctx_.builder().SetInsertPoint(double_path);
        llvm::Value* left_dbl = ctx_.builder().CreateSelect(left_is_dbl,
            tagged_.unpackDouble(left),
            ctx_.builder().CreateSIToFP(tagged_.unpackInt64(left), ctx_.doubleType()));
        llvm::Value* right_dbl = ctx_.builder().CreateSelect(right_is_dbl,
            tagged_.unpackDouble(right),
            ctx_.builder().CreateSIToFP(tagged_.unpackInt64(right), ctx_.doubleType()));

        // IEEE 754: double division by zero produces +inf, -inf, or NaN — no exception
        // R7RS: (/ 1.0 0.0) → +inf.0, (/ -1.0 0.0) → -inf.0, (/ 0.0 0.0) → +nan.0
        llvm::Value* dbl_result = ctx_.builder().CreateFDiv(left_dbl, right_dbl, "div_dbl_result");
        llvm::Value* dbl_tagged = tagged_.packDouble(dbl_result);
        llvm::BasicBlock* dbl_exit_bb = ctx_.builder().GetInsertBlock();
        ctx_.builder().CreateBr(merge);

        // Integer path - Scheme uses exact division, promoting to double for non-exact
        ctx_.builder().SetInsertPoint(int_path);
        llvm::Value* left_int = tagged_.unpackInt64(left);
        llvm::Value* right_int = tagged_.unpackInt64(right);

        // Check for division by zero in integer path
        llvm::Value* int_is_zero = ctx_.builder().CreateICmpEQ(right_int,
            llvm::ConstantInt::get(ctx_.int64Type(), 0), "div_int_zero_check");

        llvm::BasicBlock* int_zero_bb = llvm::BasicBlock::Create(ctx_.context(), "div_int_zero", func);
        llvm::BasicBlock* int_safe_bb = llvm::BasicBlock::Create(ctx_.context(), "div_int_safe", func);

        ctx_.builder().CreateCondBr(int_is_zero, int_zero_bb, int_safe_bb);

        // Division by zero path - raise exception
        ctx_.builder().SetInsertPoint(int_zero_bb);
        raiseDivideByZeroException();
        ctx_.builder().CreateUnreachable();

        // Safe integer division path
        ctx_.builder().SetInsertPoint(int_safe_bb);
        // Check if division is exact
        llvm::Value* remainder = ctx_.builder().CreateSRem(left_int, right_int);
        llvm::Value* is_exact = ctx_.builder().CreateICmpEQ(remainder,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::BasicBlock* div_exact_bb = llvm::BasicBlock::Create(ctx_.context(), "div_int_exact", func);
        llvm::BasicBlock* div_inexact_bb = llvm::BasicBlock::Create(ctx_.context(), "div_int_inexact", func);
        ctx_.builder().CreateCondBr(is_exact, div_exact_bb, div_inexact_bb);

        // Exact: return integer result
        ctx_.builder().SetInsertPoint(div_exact_bb);
        llvm::Value* exact_div = ctx_.builder().CreateSDiv(left_int, right_int);
        llvm::Value* exact_tagged = tagged_.packInt64(exact_div, true);
        ctx_.builder().CreateBr(merge);

        // Inexact: produce exact rational num/denom (R7RS: exact division stays exact)
        ctx_.builder().SetInsertPoint(div_inexact_bb);
        llvm::Value* arena_ptr_rat = getArenaPtr(ctx_);
        llvm::Value* rat_ptr = ctx_.builder().CreateCall(
            getRationalCreateFunc(ctx_), {arena_ptr_rat, left_int, right_int}, "rat_div_ptr");
        llvm::Value* inexact_tagged = tagged_.packHeapPtr(rat_ptr);
        ctx_.builder().CreateBr(merge);

        // Merge all non-AD paths
        ctx_.builder().SetInsertPoint(merge);
        llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 8, "div_result");
        phi->addIncoming(bn_div_tagged, bignum_exit);
        phi->addIncoming(rat_div_tagged, rational_exit);
        phi->addIncoming(vec_result, vector_exit);
        phi->addIncoming(dual_tagged, dual_exit);
        phi->addIncoming(complex_tagged, complex_exit);
        phi->addIncoming(dbl_tagged, dbl_exit_bb);
        phi->addIncoming(exact_tagged, div_exact_bb);
        phi->addIncoming(inexact_tagged, div_inexact_bb);

        return phi;
    });
}

// === Other Operations (mod, neg, abs, type coercion) ===

llvm::Value* ArithmeticCodegen::mod(llvm::Value* left, llvm::Value* right) {
    // R7RS modulo: result has same sign as divisor
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* bn_path = llvm::BasicBlock::Create(ctx_.context(), "mod_bn", func);
    llvm::BasicBlock* int_path = llvm::BasicBlock::Create(ctx_.context(), "mod_int", func);
    llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "mod_merge", func);

    llvm::Value* any_bignum = emitIsBignumCheck(left, right);
    ctx_.builder().CreateCondBr(any_bignum, bn_path, int_path);

    ctx_.builder().SetInsertPoint(bn_path);
    llvm::Value* bn_mod_tagged = emitBignumBinaryCall(left, right, 4);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* bn_exit = ctx_.builder().GetInsertBlock();

    // Integer path
    ctx_.builder().SetInsertPoint(int_path);
    llvm::Value* left_int = tagged_.unpackInt64(left);
    llvm::Value* right_int = tagged_.unpackInt64(right);

    // Check for division by zero
    llvm::Value* is_zero = ctx_.builder().CreateICmpEQ(right_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), "mod_zero_check");

    llvm::BasicBlock* zero_bb = llvm::BasicBlock::Create(ctx_.context(), "mod_zero", func);
    llvm::BasicBlock* safe_bb = llvm::BasicBlock::Create(ctx_.context(), "mod_safe", func);

    ctx_.builder().CreateCondBr(is_zero, zero_bb, safe_bb);

    // Division by zero path - raise exception
    ctx_.builder().SetInsertPoint(zero_bb);
    raiseDivideByZeroException();
    ctx_.builder().CreateUnreachable();

    // Safe path - perform modulo
    ctx_.builder().SetInsertPoint(safe_bb);
    llvm::Value* int_result = ctx_.builder().CreateSRem(left_int, right_int, "mod_result");
    llvm::Value* int_tagged = tagged_.packInt64(int_result, true);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* int_exit = ctx_.builder().GetInsertBlock();

    // Merge
    ctx_.builder().SetInsertPoint(merge);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "mod_result_phi");
    phi->addIncoming(bn_mod_tagged, bn_exit);
    phi->addIncoming(int_tagged, int_exit);

    return phi;
}

llvm::Value* ArithmeticCodegen::neg(llvm::Value* operand) {
    return withADUnaryDispatch(operand, 11 /*AD_NODE_NEG*/, [&]() -> llvm::Value* {
        llvm::Value* type_tag = tagged_.getType(operand);
        llvm::Value* base_type = tagged_.getBaseType(type_tag);

        llvm::Value* is_heap = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        llvm::Value* is_complex = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_COMPLEX));
        llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));

        llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* bignum_bb = llvm::BasicBlock::Create(ctx_.context(), "neg_bignum", func);
        llvm::BasicBlock* check_complex = llvm::BasicBlock::Create(ctx_.context(), "neg_check_complex", func);
        llvm::BasicBlock* complex_bb = llvm::BasicBlock::Create(ctx_.context(), "neg_complex", func);
        llvm::BasicBlock* check_double_bb = llvm::BasicBlock::Create(ctx_.context(), "neg_check_double", func);
        llvm::BasicBlock* double_bb = llvm::BasicBlock::Create(ctx_.context(), "neg_double", func);
        llvm::BasicBlock* int_bb = llvm::BasicBlock::Create(ctx_.context(), "neg_int", func);
        llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "neg_merge", func);

        ctx_.builder().CreateCondBr(is_heap, bignum_bb, check_complex);

        // Bignum negation via runtime dispatch (op 7 = neg)
        ctx_.builder().SetInsertPoint(bignum_bb);
        llvm::Value* bn_neg_result = emitBignumBinaryCall(operand, operand, 7);
        ctx_.builder().CreateBr(merge_bb);
        llvm::BasicBlock* bignum_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(check_complex);
        ctx_.builder().CreateCondBr(is_complex, complex_bb, check_double_bb);

        // Complex negation
        ctx_.builder().SetInsertPoint(complex_bb);
        llvm::Value* z = complex_.unpackComplexFromTagged(operand);
        llvm::Value* neg_z = complex_.complexNeg(z);
        llvm::Value* complex_result = complex_.packComplexToTagged(neg_z);
        ctx_.builder().CreateBr(merge_bb);
        llvm::BasicBlock* complex_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(check_double_bb);
        ctx_.builder().CreateCondBr(is_double, double_bb, int_bb);

        // Double negation
        ctx_.builder().SetInsertPoint(double_bb);
        llvm::Value* dbl_val = tagged_.unpackDouble(operand);
        llvm::Value* neg_dbl = ctx_.builder().CreateFNeg(dbl_val, "neg_double");
        llvm::Value* dbl_result = tagged_.packDouble(neg_dbl);
        ctx_.builder().CreateBr(merge_bb);
        llvm::BasicBlock* double_exit = ctx_.builder().GetInsertBlock();

        // Integer negation
        ctx_.builder().SetInsertPoint(int_bb);
        llvm::Value* int_val = tagged_.unpackInt64(operand);
        llvm::Value* neg_int = ctx_.builder().CreateNeg(int_val, "neg_int");
        llvm::Value* int_result = tagged_.packInt64(neg_int, true);
        ctx_.builder().CreateBr(merge_bb);
        llvm::BasicBlock* int_exit = ctx_.builder().GetInsertBlock();

        // Merge
        ctx_.builder().SetInsertPoint(merge_bb);
        llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 4, "neg_result");
        phi->addIncoming(bn_neg_result, bignum_exit);
        phi->addIncoming(complex_result, complex_exit);
        phi->addIncoming(dbl_result, double_exit);
        phi->addIncoming(int_result, int_exit);

        return phi;
    });
}

llvm::Value* ArithmeticCodegen::abs(llvm::Value* operand) {
    return withADUnaryDispatch(operand, 42 /*AD_NODE_ABS*/, [&]() -> llvm::Value* {
        llvm::Value* type_tag = tagged_.getType(operand);
        llvm::Value* base_type = tagged_.getBaseType(type_tag);

        llvm::Value* is_heap = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));

        llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* heap_bb = llvm::BasicBlock::Create(ctx_.context(), "abs_heap", func);
        llvm::BasicBlock* check_dbl = llvm::BasicBlock::Create(ctx_.context(), "abs_check_dbl", func);
        llvm::BasicBlock* double_bb = llvm::BasicBlock::Create(ctx_.context(), "abs_double", func);
        llvm::BasicBlock* int_bb = llvm::BasicBlock::Create(ctx_.context(), "abs_int", func);
        llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "abs_merge", func);

        ctx_.builder().CreateCondBr(is_heap, heap_bb, check_dbl);

        // Bignum abs via runtime: compare to 0, negate if negative
        ctx_.builder().SetInsertPoint(heap_bb);
        llvm::Value* zero_tagged = tagged_.packInt64(
            llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
        llvm::Value* cmp_result = emitBignumCompareCall(operand, zero_tagged, 0); // lt
        llvm::Value* is_negative = tagged_.unpackBool(cmp_result);

        llvm::BasicBlock* neg_bb = llvm::BasicBlock::Create(ctx_.context(), "abs_bn_neg", func);
        llvm::BasicBlock* pos_bb = llvm::BasicBlock::Create(ctx_.context(), "abs_bn_pos", func);
        llvm::BasicBlock* bn_merge = llvm::BasicBlock::Create(ctx_.context(), "abs_bn_merge", func);
        ctx_.builder().CreateCondBr(is_negative, neg_bb, pos_bb);

        ctx_.builder().SetInsertPoint(neg_bb);
        llvm::Value* negated = emitBignumBinaryCall(operand, operand, 7); // neg
        ctx_.builder().CreateBr(bn_merge);
        llvm::BasicBlock* neg_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(pos_bb);
        ctx_.builder().CreateBr(bn_merge);
        llvm::BasicBlock* pos_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(bn_merge);
        llvm::PHINode* heap_result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "abs_bn");
        heap_result->addIncoming(negated, neg_exit);
        heap_result->addIncoming(operand, pos_exit);
        ctx_.builder().CreateBr(merge_bb);
        llvm::BasicBlock* heap_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(check_dbl);
        ctx_.builder().CreateCondBr(is_double, double_bb, int_bb);

        // Double abs
        ctx_.builder().SetInsertPoint(double_bb);
        llvm::Value* dbl_val = tagged_.unpackDouble(operand);
        llvm::Value* neg_dbl = ctx_.builder().CreateFNeg(dbl_val);
        llvm::Value* is_neg_dbl = ctx_.builder().CreateFCmpOLT(dbl_val,
            llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        llvm::Value* abs_dbl = ctx_.builder().CreateSelect(is_neg_dbl, neg_dbl, dbl_val, "abs_double");
        llvm::Value* dbl_result = tagged_.packDouble(abs_dbl);
        ctx_.builder().CreateBr(merge_bb);
        llvm::BasicBlock* double_exit = ctx_.builder().GetInsertBlock();

        // Integer abs — handle INT64_MIN overflow by promoting to bignum
        ctx_.builder().SetInsertPoint(int_bb);
        llvm::Value* int_val = tagged_.unpackInt64(operand);
        llvm::Value* is_neg_int = ctx_.builder().CreateICmpSLT(int_val,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* is_int64_min = ctx_.builder().CreateICmpEQ(int_val,
            llvm::ConstantInt::get(ctx_.int64Type(), INT64_MIN));

        llvm::BasicBlock* int_bignum_bb = llvm::BasicBlock::Create(ctx_.context(), "abs_int_bignum", func);
        llvm::BasicBlock* int_normal_bb = llvm::BasicBlock::Create(ctx_.context(), "abs_int_normal", func);
        llvm::BasicBlock* int_merge_bb = llvm::BasicBlock::Create(ctx_.context(), "abs_int_merge", func);
        ctx_.builder().CreateCondBr(is_int64_min, int_bignum_bb, int_normal_bb);

        // INT64_MIN path: promote to bignum then negate
        ctx_.builder().SetInsertPoint(int_bignum_bb);
        llvm::Value* arena = getArenaPtr(ctx_);
        llvm::FunctionType* from_i64_type = llvm::FunctionType::get(
            ctx_.ptrType(), {ctx_.ptrType(), ctx_.int64Type()}, false);
        llvm::FunctionCallee from_i64_fn = ctx_.module().getOrInsertFunction(
            "eshkol_bignum_from_int64", from_i64_type);
        llvm::Value* as_bn = ctx_.builder().CreateCall(from_i64_fn, {arena, int_val}, "abs_as_bn");
        llvm::FunctionType* bn_neg_type = llvm::FunctionType::get(
            ctx_.ptrType(), {ctx_.ptrType(), ctx_.ptrType()}, false);
        llvm::FunctionCallee bn_neg_fn = ctx_.module().getOrInsertFunction(
            "eshkol_bignum_neg", bn_neg_type);
        llvm::Value* abs_bn = ctx_.builder().CreateCall(bn_neg_fn, {arena, as_bn}, "abs_int_bn");
        llvm::Value* bn_tagged = tagged_.packPtr(abs_bn, ESHKOL_VALUE_HEAP_PTR);
        ctx_.builder().CreateBr(int_merge_bb);
        llvm::BasicBlock* int_bn_exit = ctx_.builder().GetInsertBlock();

        // Normal int path
        ctx_.builder().SetInsertPoint(int_normal_bb);
        llvm::Value* neg_int = ctx_.builder().CreateNeg(int_val);
        llvm::Value* abs_int = ctx_.builder().CreateSelect(is_neg_int, neg_int, int_val, "abs_int");
        llvm::Value* int_normal_result = tagged_.packInt64(abs_int, true);
        ctx_.builder().CreateBr(int_merge_bb);
        llvm::BasicBlock* int_normal_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(int_merge_bb);
        llvm::PHINode* int_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "abs_int_merged");
        int_phi->addIncoming(bn_tagged, int_bn_exit);
        int_phi->addIncoming(int_normal_result, int_normal_exit);
        ctx_.builder().CreateBr(merge_bb);
        llvm::BasicBlock* int_exit = ctx_.builder().GetInsertBlock();

        // Merge
        ctx_.builder().SetInsertPoint(merge_bb);
        llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3, "abs_result");
        phi->addIncoming(heap_result, heap_exit);
        phi->addIncoming(dbl_result, double_exit);
        phi->addIncoming(int_phi, int_exit);

        return phi;
    });
}

llvm::Value* ArithmeticCodegen::intToDouble(llvm::Value* int_tagged) {
    llvm::Value* int_val = tagged_.unpackInt64(int_tagged);
    llvm::Value* dbl_val = ctx_.builder().CreateSIToFP(int_val, ctx_.doubleType(), "int_to_double");
    return tagged_.packDouble(dbl_val);
}

llvm::Value* ArithmeticCodegen::doubleToInt(llvm::Value* double_tagged) {
    llvm::Value* dbl_val = tagged_.unpackDouble(double_tagged);
    llvm::Value* int_val = ctx_.builder().CreateFPToSI(dbl_val, ctx_.int64Type(), "double_to_int");
    return tagged_.packInt64(int_val, true);
}

llvm::Value* ArithmeticCodegen::extractAsDouble(llvm::Value* tagged_val) {
    if (!tagged_val) {
        eshkol_error("arithmetic: null operand in extractAsDouble");
        return nullptr;
    }

    // Handle raw double - return as-is
    if (tagged_val->getType()->isDoubleTy()) return tagged_val;

    // Handle raw int64 - convert to double
    if (tagged_val->getType()->isIntegerTy(64)) {
        return ctx_.builder().CreateSIToFP(tagged_val, ctx_.doubleType());
    }

    // Handle tagged value with 4-way dispatch: CALLABLE(AD node), DOUBLE, HEAP_PTR(bignum), INT64
    llvm::Value* type_tag = tagged_.getType(tagged_val);
    llvm::Value* base_type = tagged_.getBaseType(type_tag);

    llvm::Value* is_callable = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* ad_check_bb = llvm::BasicBlock::Create(ctx_.context(), "ead_ad_check", func);
    llvm::BasicBlock* ad_bb = llvm::BasicBlock::Create(ctx_.context(), "ead_ad", func);
    llvm::BasicBlock* dbl_check_bb = llvm::BasicBlock::Create(ctx_.context(), "ead_dbl_check", func);
    llvm::BasicBlock* dbl_bb = llvm::BasicBlock::Create(ctx_.context(), "ead_dbl", func);
    llvm::BasicBlock* heap_bb = llvm::BasicBlock::Create(ctx_.context(), "ead_heap", func);
    llvm::BasicBlock* int_bb = llvm::BasicBlock::Create(ctx_.context(), "ead_int", func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "ead_merge", func);

    // First check: CALLABLE → potential AD node (must extract primal value)
    ctx_.builder().CreateCondBr(is_callable, ad_check_bb, dbl_check_bb);

    // AD CHECK: Verify subtype is AD_NODE, extract primal double from field 1
    ctx_.builder().SetInsertPoint(ad_check_bb);
    llvm::Value* is_ad = tagged_.checkCallableSubtype(tagged_val, CALLABLE_SUBTYPE_AD_NODE);
    ctx_.builder().CreateCondBr(is_ad, ad_bb, dbl_check_bb);

    // AD NODE PATH: Load primal value (field 1 = double value)
    ctx_.builder().SetInsertPoint(ad_bb);
    llvm::Value* ad_ptr = tagged_.unpackPtr(tagged_val);
    llvm::Value* ad_val_ptr = ctx_.builder().CreateStructGEP(
        ctx_.adNodeType(), ad_ptr, 1);
    llvm::Value* ad_val = ctx_.builder().CreateLoad(ctx_.doubleType(), ad_val_ptr, "ad_primal");
    ctx_.builder().CreateBr(merge_bb);
    ad_bb = ctx_.builder().GetInsertBlock();

    // Double check
    ctx_.builder().SetInsertPoint(dbl_check_bb);
    ctx_.builder().CreateCondBr(is_double, dbl_bb, heap_bb);

    // Double path: unpack directly
    ctx_.builder().SetInsertPoint(dbl_bb);
    llvm::Value* dbl_val = tagged_.unpackDouble(tagged_val);
    ctx_.builder().CreateBr(merge_bb);
    dbl_bb = ctx_.builder().GetInsertBlock();

    // Heap pointer path: check subtype for rational or bignum, convert to double
    ctx_.builder().SetInsertPoint(heap_bb);
    llvm::BasicBlock* heap_dispatch_bb = llvm::BasicBlock::Create(ctx_.context(), "ead_heap_dispatch", func);
    ctx_.builder().CreateCondBr(is_heap_ptr, heap_dispatch_bb, int_bb);

    // Read heap subtype header at ptr-8
    ctx_.builder().SetInsertPoint(heap_dispatch_bb);
    llvm::Value* heap_ptr = tagged_.unpackPtr(tagged_val);
    llvm::Value* header_ptr = ctx_.builder().CreateGEP(
        ctx_.int8Type(), heap_ptr, llvm::ConstantInt::get(ctx_.int64Type(), -8));
    llvm::Value* subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), header_ptr, "heap_subtype");
    llvm::Value* is_rational = ctx_.builder().CreateICmpEQ(subtype,
        llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_RATIONAL));
    llvm::BasicBlock* rational_bb = llvm::BasicBlock::Create(ctx_.context(), "ead_rational", func);
    llvm::BasicBlock* bignum_bb = llvm::BasicBlock::Create(ctx_.context(), "ead_bignum", func);
    ctx_.builder().CreateCondBr(is_rational, rational_bb, bignum_bb);

    // Rational path: call eshkol_rational_to_double(ptr)
    ctx_.builder().SetInsertPoint(rational_bb);
    llvm::FunctionType* rat_to_dbl_type = llvm::FunctionType::get(
        ctx_.doubleType(), {ctx_.ptrType()}, false);
    llvm::FunctionCallee rat_to_dbl = ctx_.module().getOrInsertFunction(
        "eshkol_rational_to_double", rat_to_dbl_type);
    llvm::Value* rat_dbl = ctx_.builder().CreateCall(rat_to_dbl, {heap_ptr}, "rat_to_dbl");
    ctx_.builder().CreateBr(merge_bb);
    rational_bb = ctx_.builder().GetInsertBlock();

    // Bignum path: call eshkol_bignum_to_double(ptr)
    ctx_.builder().SetInsertPoint(bignum_bb);
    llvm::FunctionType* bn_to_dbl_type = llvm::FunctionType::get(
        ctx_.doubleType(), {ctx_.ptrType()}, false);
    llvm::FunctionCallee bn_to_dbl = ctx_.module().getOrInsertFunction(
        "eshkol_bignum_to_double", bn_to_dbl_type);
    llvm::Value* bn_dbl = ctx_.builder().CreateCall(bn_to_dbl, {heap_ptr}, "bn_to_dbl");
    ctx_.builder().CreateBr(merge_bb);
    bignum_bb = ctx_.builder().GetInsertBlock();

    // Int path: SIToFP
    ctx_.builder().SetInsertPoint(int_bb);
    llvm::Value* int_val = tagged_.unpackInt64(tagged_val);
    llvm::Value* int_as_dbl = ctx_.builder().CreateSIToFP(int_val, ctx_.doubleType());
    ctx_.builder().CreateBr(merge_bb);
    int_bb = ctx_.builder().GetInsertBlock();

    // Merge
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.doubleType(), 5, "as_double");
    phi->addIncoming(ad_val, ad_bb);
    phi->addIncoming(dbl_val, dbl_bb);
    phi->addIncoming(rat_dbl, rational_bb);
    phi->addIncoming(bn_dbl, bignum_bb);
    phi->addIncoming(int_as_dbl, int_bb);
    return phi;
}

// === Polymorphic Comparison ===
// R7RS: Numeric comparison operators (= < > <= >=) only work on numbers.
// Non-numeric operands should signal an error.

llvm::Value* ArithmeticCodegen::compare(llvm::Value* left, llvm::Value* right,
                                         const std::string& operation) {
    if (!left || !right) {
        eshkol_error("arithmetic: null operand in compare/%s (left=%p, right=%p)", operation.c_str(), (void*)left, (void*)right);
        return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
    }

    // Extract type tags
    // Use getBaseType() to properly handle legacy types (VECTOR_PTR=34, TENSOR_PTR=35, etc.)
    // DO NOT use 0x0F mask - 34 & 0x0F = 2 (DOUBLE) which is WRONG!
    llvm::Value* left_type = tagged_.getType(left);
    llvm::Value* right_type = tagged_.getType(right);

    llvm::Value* left_base = tagged_.getBaseType(left_type);
    llvm::Value* right_base = tagged_.getBaseType(right_type);

    // Check operand types
    llvm::Value* left_is_double = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* right_is_double = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* any_double = ctx_.builder().CreateOr(left_is_double, right_is_double);

    llvm::Value* left_is_int = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
    llvm::Value* right_is_int = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));

    llvm::Value* left_is_heap = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    llvm::Value* right_is_heap = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    // AD NODE AWARENESS: CALLABLE type may contain AD nodes (primal extracted by extractAsDouble)
    llvm::Value* left_is_callable = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    llvm::Value* right_is_callable = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    llvm::Value* any_callable = ctx_.builder().CreateOr(left_is_callable, right_is_callable);

    // R7RS compliance: Both operands must be numbers (int64, double, bignum, or AD node)
    llvm::Value* left_is_number = ctx_.builder().CreateOr(
        ctx_.builder().CreateOr(left_is_double, left_is_int),
        ctx_.builder().CreateOr(left_is_heap, left_is_callable));
    llvm::Value* right_is_number = ctx_.builder().CreateOr(
        ctx_.builder().CreateOr(right_is_double, right_is_int),
        ctx_.builder().CreateOr(right_is_heap, right_is_callable));
    llvm::Value* both_numbers = ctx_.builder().CreateAnd(left_is_number, right_is_number);

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* type_error_path = llvm::BasicBlock::Create(ctx_.context(), "cmp_type_error", func);
    llvm::BasicBlock* numeric_path = llvm::BasicBlock::Create(ctx_.context(), "cmp_numeric", func);
    llvm::BasicBlock* bn_cmp_path = llvm::BasicBlock::Create(ctx_.context(), "cmp_bn", func);
    llvm::BasicBlock* check_rational = llvm::BasicBlock::Create(ctx_.context(), "cmp_check_rat", func);
    llvm::BasicBlock* rational_cmp_path = llvm::BasicBlock::Create(ctx_.context(), "cmp_rat", func);
    llvm::BasicBlock* check_double = llvm::BasicBlock::Create(ctx_.context(), "cmp_check_dbl", func);
    llvm::BasicBlock* dbl_cmp_path = llvm::BasicBlock::Create(ctx_.context(), "cmp_dbl", func);
    llvm::BasicBlock* int_path = llvm::BasicBlock::Create(ctx_.context(), "cmp_int", func);
    llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "cmp_merge", func);

    // R7RS: Error if not both numbers
    ctx_.builder().CreateCondBr(both_numbers, numeric_path, type_error_path);

    // Type error path
    ctx_.builder().SetInsertPoint(type_error_path);
    llvm::Function* type_error_func = ctx_.module().getFunction("eshkol_type_error");
    if (!type_error_func) {
        llvm::FunctionType* error_type = llvm::FunctionType::get(
            ctx_.builder().getVoidTy(),
            {ctx_.builder().getPtrTy(), ctx_.builder().getPtrTy()},
            false);
        type_error_func = llvm::Function::Create(error_type, llvm::Function::ExternalLinkage,
            "eshkol_type_error", &ctx_.module());
    }
    std::string op_name = (operation == "eq") ? "=" :
                          (operation == "lt") ? "<" :
                          (operation == "gt") ? ">" :
                          (operation == "le") ? "<=" : ">=";
    llvm::Value* proc_name = ctx_.builder().CreateGlobalString(op_name, "cmp_proc_name");
    llvm::Value* expected_type = ctx_.builder().CreateGlobalString("number", "cmp_expected_type");
    ctx_.builder().CreateCall(type_error_func, {proc_name, expected_type});
    llvm::Value* error_result = tagged_.packBool(ctx_.builder().getFalse());
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* error_exit = ctx_.builder().GetInsertBlock();

    // Numeric path: check for bignum first via runtime
    ctx_.builder().SetInsertPoint(numeric_path);
    llvm::Value* any_bignum = emitIsBignumCheck(left, right);
    ctx_.builder().CreateCondBr(any_bignum, bn_cmp_path, check_rational);

    // Bignum comparison via runtime dispatch (handles all bignum combinations)
    ctx_.builder().SetInsertPoint(bn_cmp_path);
    int cmp_op = (operation == "lt") ? 0 :
                 (operation == "gt") ? 1 :
                 (operation == "eq") ? 2 :
                 (operation == "le") ? 3 : 4;
    llvm::Value* bn_cmp_result = emitBignumCompareCall(left, right, cmp_op);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* bn_cmp_exit = ctx_.builder().GetInsertBlock();

    // Check for rational operands (must come before double since rationals are HEAP_PTR)
    ctx_.builder().SetInsertPoint(check_rational);
    llvm::Value* any_rational = emitIsRationalCheck(left, right);
    ctx_.builder().CreateCondBr(any_rational, rational_cmp_path, check_double);

    // Rational comparison via runtime dispatch (handles int/rational mixed operands)
    ctx_.builder().SetInsertPoint(rational_cmp_path);
    llvm::Value* rat_cmp_result = emitRationalCompareCall(left, right, cmp_op);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* rational_cmp_exit = ctx_.builder().GetInsertBlock();

    // Check for double or AD node operands (AD nodes contain doubles, extracted by extractAsDouble)
    ctx_.builder().SetInsertPoint(check_double);
    llvm::Value* any_double_or_ad = ctx_.builder().CreateOr(any_double, any_callable);
    ctx_.builder().CreateCondBr(any_double_or_ad, dbl_cmp_path, int_path);

    // Double comparison: use extractAsDouble which handles DOUBLE, INT64, HEAP_PTR, and AD nodes
    ctx_.builder().SetInsertPoint(dbl_cmp_path);
    llvm::Value* left_double = extractAsDouble(left);
    llvm::Value* right_double = extractAsDouble(right);
    llvm::Value* double_cmp = nullptr;
    if (operation == "lt")      double_cmp = ctx_.builder().CreateFCmpOLT(left_double, right_double);
    else if (operation == "gt") double_cmp = ctx_.builder().CreateFCmpOGT(left_double, right_double);
    else if (operation == "eq") double_cmp = ctx_.builder().CreateFCmpOEQ(left_double, right_double);
    else if (operation == "le") double_cmp = ctx_.builder().CreateFCmpOLE(left_double, right_double);
    else if (operation == "ge") double_cmp = ctx_.builder().CreateFCmpOGE(left_double, right_double);
    llvm::Value* tagged_double_result = tagged_.packBool(double_cmp);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* double_exit = ctx_.builder().GetInsertBlock();

    // Int path: compare as int64
    ctx_.builder().SetInsertPoint(int_path);
    llvm::Value* left_int = tagged_.unpackInt64(left);
    llvm::Value* right_int = tagged_.unpackInt64(right);
    llvm::Value* int_cmp = nullptr;
    if (operation == "lt")      int_cmp = ctx_.builder().CreateICmpSLT(left_int, right_int);
    else if (operation == "gt") int_cmp = ctx_.builder().CreateICmpSGT(left_int, right_int);
    else if (operation == "eq") int_cmp = ctx_.builder().CreateICmpEQ(left_int, right_int);
    else if (operation == "le") int_cmp = ctx_.builder().CreateICmpSLE(left_int, right_int);
    else if (operation == "ge") int_cmp = ctx_.builder().CreateICmpSGE(left_int, right_int);
    llvm::Value* tagged_int_result = tagged_.packBool(int_cmp);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* int_exit = ctx_.builder().GetInsertBlock();

    // Merge results
    ctx_.builder().SetInsertPoint(merge);
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 5);
    result_phi->addIncoming(error_result, error_exit);
    result_phi->addIncoming(bn_cmp_result, bn_cmp_exit);
    result_phi->addIncoming(rat_cmp_result, rational_cmp_exit);
    result_phi->addIncoming(tagged_double_result, double_exit);
    result_phi->addIncoming(tagged_int_result, int_exit);

    return result_phi;
}

// === Power Function ===

llvm::Value* ArithmeticCodegen::pow(llvm::Value* base, llvm::Value* exponent) {
    if (!base || !exponent) {
        return tagged_.packDouble(llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    }

    return withADBinaryDispatch(base, exponent, 10 /*AD_NODE_POW*/, [&]() -> llvm::Value* {
        // Re-extract types inside lambda
        llvm::Value* base_type = tagged_.getType(base);
        llvm::Value* exp_type = tagged_.getType(exponent);
        llvm::Value* base_base = tagged_.getBaseType(base_type);
        llvm::Value* exp_base = tagged_.getBaseType(exp_type);

        llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* dual_path = llvm::BasicBlock::Create(ctx_.context(), "pow_dual", func);
        llvm::BasicBlock* check_exact = llvm::BasicBlock::Create(ctx_.context(), "pow_check_exact", func);
        llvm::BasicBlock* exact_path = llvm::BasicBlock::Create(ctx_.context(), "pow_exact", func);
        llvm::BasicBlock* regular_path = llvm::BasicBlock::Create(ctx_.context(), "pow_regular", func);
        llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "pow_merge", func);

        // Check for dual numbers
        llvm::Value* base_is_dual = ctx_.builder().CreateICmpEQ(base_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
        llvm::Value* exp_is_dual = ctx_.builder().CreateICmpEQ(exp_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
        llvm::Value* any_dual = ctx_.builder().CreateOr(base_is_dual, exp_is_dual);
        ctx_.builder().CreateCondBr(any_dual, dual_path, check_exact);

        // Dual number path
        ctx_.builder().SetInsertPoint(dual_path);
        llvm::Value* base_is_double = ctx_.builder().CreateICmpEQ(base_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* exp_is_double = ctx_.builder().CreateICmpEQ(exp_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* base_dual = convertToDual(base, base_is_dual, base_is_double);
        llvm::Value* exp_dual = convertToDual(exponent, exp_is_dual, exp_is_double);
        llvm::Value* dual_result = autodiff_.dualPow(base_dual, exp_dual);
        llvm::Value* dual_tagged = autodiff_.packDualToTagged(dual_result);
        llvm::BasicBlock* dual_exit = ctx_.builder().GetInsertBlock();
        ctx_.builder().CreateBr(merge);

        // Check if both operands are exact integers with non-negative exponent
        ctx_.builder().SetInsertPoint(check_exact);
        llvm::Value* base_is_int = ctx_.builder().CreateICmpEQ(base_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
        llvm::Value* base_is_heap = ctx_.builder().CreateICmpEQ(base_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        llvm::Value* base_is_exact = ctx_.builder().CreateOr(base_is_int, base_is_heap);
        llvm::Value* exp_is_int = ctx_.builder().CreateICmpEQ(exp_base,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
        llvm::Value* both_exact_int = ctx_.builder().CreateAnd(base_is_exact, exp_is_int);
        llvm::Value* exp_val = tagged_.unpackInt64(exponent);
        llvm::Value* exp_non_neg = ctx_.builder().CreateICmpSGE(exp_val,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* use_exact = ctx_.builder().CreateAnd(both_exact_int, exp_non_neg);
        ctx_.builder().CreateCondBr(use_exact, exact_path, regular_path);

        // Exact integer exponentiation via runtime
        ctx_.builder().SetInsertPoint(exact_path);
        llvm::Value* arena = getArenaPtr(ctx_);
        llvm::Value* base_alloca = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
        llvm::Value* exp_alloca = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
        llvm::Value* result_alloca = ctx_.builder().CreateAlloca(ctx_.taggedValueType());
        ctx_.builder().CreateStore(base, base_alloca);
        ctx_.builder().CreateStore(exponent, exp_alloca);

        llvm::FunctionType* pow_tagged_type = llvm::FunctionType::get(
            ctx_.builder().getVoidTy(),
            {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()},
            false);
        llvm::FunctionCallee pow_tagged_fn = ctx_.module().getOrInsertFunction(
            "eshkol_bignum_pow_tagged", pow_tagged_type);
        ctx_.builder().CreateCall(pow_tagged_fn, {arena, base_alloca, exp_alloca, result_alloca});

        llvm::Value* exact_result = ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_alloca);
        ctx_.builder().CreateBr(merge);
        llvm::BasicBlock* exact_exit = ctx_.builder().GetInsertBlock();

        // Regular path - standard pow (double)
        ctx_.builder().SetInsertPoint(regular_path);
        llvm::Value* base_dbl = extractAsDouble(base);
        llvm::Value* exp_dbl = extractAsDouble(exponent);

        llvm::Function* pow_func = ctx_.module().getFunction("pow");
        if (!pow_func) {
            llvm::FunctionType* pow_type = llvm::FunctionType::get(
                ctx_.doubleType(), {ctx_.doubleType(), ctx_.doubleType()}, false);
            pow_func = llvm::Function::Create(pow_type, llvm::Function::ExternalLinkage,
                                              "pow", &ctx_.module());
        }

        llvm::Value* result = ctx_.builder().CreateCall(pow_func, {base_dbl, exp_dbl}, "pow_result");
        llvm::Value* regular_tagged = tagged_.packDouble(result);
        llvm::BasicBlock* regular_exit = ctx_.builder().GetInsertBlock();
        ctx_.builder().CreateBr(merge);

        // Merge paths
        ctx_.builder().SetInsertPoint(merge);
        llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3, "pow_result");
        phi->addIncoming(dual_tagged, dual_exit);
        phi->addIncoming(exact_result, exact_exit);
        phi->addIncoming(regular_tagged, regular_exit);

        return phi;
    });
}

// === Min/Max Functions ===

llvm::Value* ArithmeticCodegen::min(llvm::Value* left, llvm::Value* right) {
    if (!left || !right) {
        eshkol_error("arithmetic: null operand in min (left=%p, right=%p)", (void*)left, (void*)right);
        return tagged_.packDouble(llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    }

    return withADBinaryDispatch(left, right, 45 /*AD_NODE_MIN*/, [&]() -> llvm::Value* {
        llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* bn_path = llvm::BasicBlock::Create(ctx_.context(), "min_bn", func);
        llvm::BasicBlock* dbl_path = llvm::BasicBlock::Create(ctx_.context(), "min_dbl", func);
        llvm::BasicBlock* pick_left = llvm::BasicBlock::Create(ctx_.context(), "min_left", func);
        llvm::BasicBlock* pick_right = llvm::BasicBlock::Create(ctx_.context(), "min_right", func);
        llvm::BasicBlock* min_merge = llvm::BasicBlock::Create(ctx_.context(), "min_merge", func);

        // Check if either operand is bignum for exact comparison
        llvm::Value* any_bignum = emitIsBignumCheck(left, right);
        ctx_.builder().CreateCondBr(any_bignum, bn_path, dbl_path);

        // Bignum path: use exact comparison (op 0 = lt: left < right)
        ctx_.builder().SetInsertPoint(bn_path);
        llvm::Value* bn_cmp = emitBignumCompareCall(left, right, 0); // lt
        llvm::Value* bn_is_lt = tagged_.unpackBool(bn_cmp);
        ctx_.builder().CreateCondBr(bn_is_lt, pick_left, pick_right);

        // Double path: extract as double for comparison
        ctx_.builder().SetInsertPoint(dbl_path);
        llvm::Value* left_dbl = extractAsDouble(left);
        llvm::Value* right_dbl = extractAsDouble(right);
        llvm::Value* is_le = ctx_.builder().CreateFCmpOLE(left_dbl, right_dbl, "min_le");
        ctx_.builder().CreateCondBr(is_le, pick_left, pick_right);

        ctx_.builder().SetInsertPoint(pick_left);
        ctx_.builder().CreateBr(min_merge);

        ctx_.builder().SetInsertPoint(pick_right);
        ctx_.builder().CreateBr(min_merge);

        ctx_.builder().SetInsertPoint(min_merge);
        llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "min_result");
        phi->addIncoming(left, pick_left);
        phi->addIncoming(right, pick_right);
        return phi;
    });
}

llvm::Value* ArithmeticCodegen::max(llvm::Value* left, llvm::Value* right) {
    if (!left || !right) {
        eshkol_error("arithmetic: null operand in max (left=%p, right=%p)", (void*)left, (void*)right);
        return tagged_.packDouble(llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    }

    return withADBinaryDispatch(left, right, 44 /*AD_NODE_MAX*/, [&]() -> llvm::Value* {
        llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* bn_path = llvm::BasicBlock::Create(ctx_.context(), "max_bn", func);
        llvm::BasicBlock* dbl_path = llvm::BasicBlock::Create(ctx_.context(), "max_dbl", func);
        llvm::BasicBlock* pick_left = llvm::BasicBlock::Create(ctx_.context(), "max_left", func);
        llvm::BasicBlock* pick_right = llvm::BasicBlock::Create(ctx_.context(), "max_right", func);
        llvm::BasicBlock* max_merge = llvm::BasicBlock::Create(ctx_.context(), "max_merge", func);

        // Check if either operand is bignum for exact comparison
        llvm::Value* any_bignum = emitIsBignumCheck(left, right);
        ctx_.builder().CreateCondBr(any_bignum, bn_path, dbl_path);

        // Bignum path: use exact comparison (op 1 = gt: left > right)
        ctx_.builder().SetInsertPoint(bn_path);
        llvm::Value* bn_cmp = emitBignumCompareCall(left, right, 1); // gt
        llvm::Value* bn_is_gt = tagged_.unpackBool(bn_cmp);
        ctx_.builder().CreateCondBr(bn_is_gt, pick_left, pick_right);

        // Double path: extract as double for comparison
        ctx_.builder().SetInsertPoint(dbl_path);
        llvm::Value* left_dbl = extractAsDouble(left);
        llvm::Value* right_dbl = extractAsDouble(right);
        llvm::Value* is_ge = ctx_.builder().CreateFCmpOGE(left_dbl, right_dbl, "max_ge");
        ctx_.builder().CreateCondBr(is_ge, pick_left, pick_right);

        ctx_.builder().SetInsertPoint(pick_left);
        ctx_.builder().CreateBr(max_merge);

        ctx_.builder().SetInsertPoint(pick_right);
        ctx_.builder().CreateBr(max_merge);

        ctx_.builder().SetInsertPoint(max_merge);
        llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "max_result");
        phi->addIncoming(left, pick_left);
        phi->addIncoming(right, pick_right);
        return phi;
    });
}

// === Remainder Function ===

llvm::Value* ArithmeticCodegen::remainder(llvm::Value* dividend, llvm::Value* divisor) {
    if (!dividend || !divisor) {
        return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
    }

    // Extract type information
    llvm::Value* dividend_type = tagged_.getType(dividend);
    llvm::Value* divisor_type = tagged_.getType(divisor);

    llvm::Value* dividend_base = tagged_.getBaseType(dividend_type);
    llvm::Value* divisor_base = tagged_.getBaseType(divisor_type);

    // Check operand types
    llvm::Value* dividend_is_int = ctx_.builder().CreateICmpEQ(dividend_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
    llvm::Value* divisor_is_int = ctx_.builder().CreateICmpEQ(divisor_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
    llvm::Value* both_int = ctx_.builder().CreateAnd(dividend_is_int, divisor_is_int);

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* bn_path = llvm::BasicBlock::Create(ctx_.context(), "rem_bn", func);
    llvm::BasicBlock* scalar_path = llvm::BasicBlock::Create(ctx_.context(), "rem_scalar", func);
    llvm::BasicBlock* int_path = llvm::BasicBlock::Create(ctx_.context(), "rem_int", func);
    llvm::BasicBlock* double_path = llvm::BasicBlock::Create(ctx_.context(), "rem_double", func);
    llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "rem_merge", func);

    llvm::Value* any_bignum = emitIsBignumCheck(dividend, divisor);
    ctx_.builder().CreateCondBr(any_bignum, bn_path, scalar_path);

    ctx_.builder().SetInsertPoint(bn_path);
    llvm::Value* bn_rem_tagged = emitBignumBinaryCall(dividend, divisor, 6);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* bn_exit = ctx_.builder().GetInsertBlock();

    // Scalar path: existing int64/double dispatch
    ctx_.builder().SetInsertPoint(scalar_path);
    ctx_.builder().CreateCondBr(both_int, int_path, double_path);

    // Integer path: use srem
    ctx_.builder().SetInsertPoint(int_path);
    llvm::Value* a_int = tagged_.unpackInt64(dividend);
    llvm::Value* b_int = tagged_.unpackInt64(divisor);

    // Check for division by zero in integer path
    llvm::Value* int_is_zero = ctx_.builder().CreateICmpEQ(b_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), "rem_zero_check");

    llvm::BasicBlock* int_zero_bb = llvm::BasicBlock::Create(ctx_.context(), "rem_int_zero", func);
    llvm::BasicBlock* int_safe_bb = llvm::BasicBlock::Create(ctx_.context(), "rem_int_safe", func);

    ctx_.builder().CreateCondBr(int_is_zero, int_zero_bb, int_safe_bb);

    // Division by zero path - raise exception
    ctx_.builder().SetInsertPoint(int_zero_bb);
    raiseDivideByZeroException();
    ctx_.builder().CreateUnreachable();

    // Safe integer remainder
    ctx_.builder().SetInsertPoint(int_safe_bb);
    llvm::Value* int_result = ctx_.builder().CreateSRem(a_int, b_int, "srem_result");
    llvm::Value* int_tagged = tagged_.packInt64(int_result, true);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* int_exit = ctx_.builder().GetInsertBlock();

    // Double path: use C's remainder function
    ctx_.builder().SetInsertPoint(double_path);
    llvm::Value* a_dbl = extractAsDouble(dividend);
    llvm::Value* b_dbl = extractAsDouble(divisor);

    llvm::Function* rem_func = ctx_.module().getFunction("remainder");
    if (!rem_func) {
        llvm::FunctionType* rem_type = llvm::FunctionType::get(
            ctx_.doubleType(),
            {ctx_.doubleType(), ctx_.doubleType()},
            false);
        rem_func = llvm::Function::Create(rem_type, llvm::Function::ExternalLinkage,
                                          "remainder", &ctx_.module());
    }

    llvm::Value* dbl_result = ctx_.builder().CreateCall(rem_func, {a_dbl, b_dbl}, "rem_result");
    llvm::Value* dbl_tagged = tagged_.packDouble(dbl_result);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* dbl_exit = ctx_.builder().GetInsertBlock();

    // Merge
    ctx_.builder().SetInsertPoint(merge);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3, "remainder_result");
    phi->addIncoming(bn_rem_tagged, bn_exit);
    phi->addIncoming(int_tagged, int_exit);
    phi->addIncoming(dbl_tagged, dbl_exit);

    return phi;
}

// === Quotient Function ===

llvm::Value* ArithmeticCodegen::quotient(llvm::Value* dividend, llvm::Value* divisor) {
    if (!dividend || !divisor) {
        return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
    }

    // Extract type information
    llvm::Value* dividend_type = tagged_.getType(dividend);
    llvm::Value* divisor_type = tagged_.getType(divisor);

    llvm::Value* dividend_base = tagged_.getBaseType(dividend_type);
    llvm::Value* divisor_base = tagged_.getBaseType(divisor_type);

    // Check operand types
    llvm::Value* dividend_is_int = ctx_.builder().CreateICmpEQ(dividend_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
    llvm::Value* divisor_is_int = ctx_.builder().CreateICmpEQ(divisor_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
    llvm::Value* both_int = ctx_.builder().CreateAnd(dividend_is_int, divisor_is_int);

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* bn_path = llvm::BasicBlock::Create(ctx_.context(), "quot_bn", func);
    llvm::BasicBlock* scalar_path = llvm::BasicBlock::Create(ctx_.context(), "quot_scalar", func);
    llvm::BasicBlock* int_path = llvm::BasicBlock::Create(ctx_.context(), "quot_int", func);
    llvm::BasicBlock* double_path = llvm::BasicBlock::Create(ctx_.context(), "quot_double", func);
    llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "quot_merge", func);

    llvm::Value* any_bignum = emitIsBignumCheck(dividend, divisor);
    ctx_.builder().CreateCondBr(any_bignum, bn_path, scalar_path);

    ctx_.builder().SetInsertPoint(bn_path);
    llvm::Value* bn_quot_tagged = emitBignumBinaryCall(dividend, divisor, 5);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* bn_exit = ctx_.builder().GetInsertBlock();

    // Scalar path: existing int64/double dispatch
    ctx_.builder().SetInsertPoint(scalar_path);
    ctx_.builder().CreateCondBr(both_int, int_path, double_path);

    // Integer path: use sdiv (truncates toward zero)
    ctx_.builder().SetInsertPoint(int_path);
    llvm::Value* a_int = tagged_.unpackInt64(dividend);
    llvm::Value* b_int = tagged_.unpackInt64(divisor);

    // Check for division by zero in integer path
    llvm::Value* int_is_zero = ctx_.builder().CreateICmpEQ(b_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0), "quot_zero_check");

    llvm::BasicBlock* int_zero_bb = llvm::BasicBlock::Create(ctx_.context(), "quot_int_zero", func);
    llvm::BasicBlock* int_safe_bb = llvm::BasicBlock::Create(ctx_.context(), "quot_int_safe", func);

    ctx_.builder().CreateCondBr(int_is_zero, int_zero_bb, int_safe_bb);

    // Division by zero path - raise exception
    ctx_.builder().SetInsertPoint(int_zero_bb);
    raiseDivideByZeroException();
    ctx_.builder().CreateUnreachable();

    // Safe integer division
    ctx_.builder().SetInsertPoint(int_safe_bb);
    llvm::Value* int_result = ctx_.builder().CreateSDiv(a_int, b_int, "sdiv_result");
    llvm::Value* int_tagged = tagged_.packInt64(int_result, true);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* int_exit = ctx_.builder().GetInsertBlock();

    // Double path: divide and truncate
    ctx_.builder().SetInsertPoint(double_path);
    llvm::Value* a_dbl = extractAsDouble(dividend);
    llvm::Value* b_dbl = extractAsDouble(divisor);
    llvm::Value* div_result = ctx_.builder().CreateFDiv(a_dbl, b_dbl, "fdiv_result");

    llvm::Function* trunc_func = ctx_.module().getFunction("trunc");
    if (!trunc_func) {
        llvm::FunctionType* trunc_type = llvm::FunctionType::get(
            ctx_.doubleType(),
            {ctx_.doubleType()},
            false);
        trunc_func = llvm::Function::Create(trunc_type, llvm::Function::ExternalLinkage,
                                            "trunc", &ctx_.module());
    }

    llvm::Value* truncated = ctx_.builder().CreateCall(trunc_func, {div_result}, "trunc_result");
    llvm::Value* dbl_as_int = ctx_.builder().CreateFPToSI(truncated, ctx_.int64Type(), "quot_int");
    llvm::Value* dbl_tagged = tagged_.packInt64(dbl_as_int, true);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* dbl_exit = ctx_.builder().GetInsertBlock();

    // Merge
    ctx_.builder().SetInsertPoint(merge);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3, "quotient_result");
    phi->addIncoming(bn_quot_tagged, bn_exit);
    phi->addIncoming(int_tagged, int_exit);
    phi->addIncoming(dbl_tagged, dbl_exit);

    return phi;
}

// === Unary Math Functions ===

llvm::Value* ArithmeticCodegen::mathFunc(llvm::Value* operand, const std::string& func_name) {
    if (!operand) {
        return tagged_.packDouble(llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    }

    // Extract operand as double
    llvm::Value* val = extractAsDouble(operand);

    // Get or declare the math function
    llvm::Function* math_fn = ctx_.module().getFunction(func_name);
    if (!math_fn) {
        llvm::FunctionType* fn_type = llvm::FunctionType::get(
            ctx_.doubleType(),
            {ctx_.doubleType()},
            false);
        math_fn = llvm::Function::Create(fn_type, llvm::Function::ExternalLinkage,
                                         func_name, &ctx_.module());
    }

    llvm::Value* result = ctx_.builder().CreateCall(math_fn, {val}, func_name + "_result");
    return tagged_.packDouble(result);
}

// === Exception Helpers ===

void ArithmeticCodegen::raiseDivideByZeroException() {
    // Get or declare eshkol_make_exception_with_header
    llvm::Function* make_exc_func = ctx_.module().getFunction("eshkol_make_exception_with_header");
    if (!make_exc_func) {
        llvm::FunctionType* make_type = llvm::FunctionType::get(
            llvm::PointerType::getUnqual(ctx_.context()),  // Returns exception pointer
            {ctx_.int32Type(),                              // Exception type
             llvm::PointerType::getUnqual(ctx_.context())}, // Message string
            false);
        make_exc_func = llvm::Function::Create(make_type,
            llvm::Function::ExternalLinkage, "eshkol_make_exception_with_header", &ctx_.module());
    }

    // Get or declare eshkol_raise
    llvm::Function* raise_func = ctx_.module().getFunction("eshkol_raise");
    if (!raise_func) {
        llvm::FunctionType* raise_type = llvm::FunctionType::get(
            llvm::Type::getVoidTy(ctx_.context()),
            {llvm::PointerType::getUnqual(ctx_.context())},  // Exception pointer
            false);
        raise_func = llvm::Function::Create(raise_type,
            llvm::Function::ExternalLinkage, "eshkol_raise", &ctx_.module());
        raise_func->setDoesNotReturn();
    }

    // Create error message string
    llvm::Value* error_msg = ctx_.builder().CreateGlobalString("division by zero");

    // Create exception object: eshkol_make_exception_with_header(ESHKOL_EXCEPTION_DIVIDE_BY_ZERO, msg)
    llvm::Value* exc = ctx_.builder().CreateCall(make_exc_func, {
        llvm::ConstantInt::get(ctx_.int32Type(), ESHKOL_EXCEPTION_DIVIDE_BY_ZERO),
        error_msg
    }, "div_zero_exception");

    // Raise the exception
    ctx_.builder().CreateCall(raise_func, {exc});
}

void ArithmeticCodegen::emitOverflowError(const char* message) {
    llvm::Function* make_exc_func = ctx_.module().getFunction("eshkol_make_exception_with_header");
    if (!make_exc_func) {
        llvm::FunctionType* make_type = llvm::FunctionType::get(
            llvm::PointerType::getUnqual(ctx_.context()),
            {ctx_.int32Type(), llvm::PointerType::getUnqual(ctx_.context())},
            false);
        make_exc_func = llvm::Function::Create(make_type,
            llvm::Function::ExternalLinkage, "eshkol_make_exception_with_header", &ctx_.module());
    }

    llvm::Function* raise_func = ctx_.module().getFunction("eshkol_raise");
    if (!raise_func) {
        llvm::FunctionType* raise_type = llvm::FunctionType::get(
            llvm::Type::getVoidTy(ctx_.context()),
            {llvm::PointerType::getUnqual(ctx_.context())},
            false);
        raise_func = llvm::Function::Create(raise_type,
            llvm::Function::ExternalLinkage, "eshkol_raise", &ctx_.module());
        raise_func->setDoesNotReturn();
    }

    llvm::Value* error_msg = ctx_.builder().CreateGlobalString(message);
    llvm::Value* exc = ctx_.builder().CreateCall(make_exc_func, {
        llvm::ConstantInt::get(ctx_.int32Type(), ESHKOL_EXCEPTION_ERROR),
        error_msg
    }, "overflow_exception");
    ctx_.builder().CreateCall(raise_func, {exc});
    ctx_.builder().CreateUnreachable();
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
