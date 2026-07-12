/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen — Matmul / Reduction / Apply.
 * Extracted from tensor_codegen.cpp during the v1.2 mechanical split.
 *
 * This module groups the "shared compute infrastructure" that the
 * higher-level tensor ops are built on top of:
 *   - matmulSIMD  (SIMD-accelerated matrix multiply)
 *   - tensorArithmeticInternal (general elementwise op dispatcher)
 *   - tensorDot, tensorApply
 *   - tensorReduceAll, tensorReduceWithDim, emitAxisReduce
 *   - tensorSum, tensorMean
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-reduce-extract baseline.
 */
#include <eshkol/backend/tensor_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/autodiff_codegen.h>
#include <eshkol/backend/cpu_features.h>
#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Config/llvm-config.h>

// LLVM VERSION COMPATIBILITY
#if LLVM_VERSION_MAJOR >= 21
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getOrInsertDeclaration(mod, id, types)
#else
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getDeclaration(mod, id, types)
#endif

#ifdef ESHKOL_XLA_ENABLED
#include <eshkol/backend/xla/xla_codegen.h>
#endif

namespace eshkol {

namespace {

void emitStructuralIntErrorLocation(CodegenContext& ctx) {
    uint32_t line = ctx.currentSourceLine();
    if (line == 0) {
        return;
    }

    llvm::Function* set_loc_fn = ctx.module().getFunction("eshkol_set_error_location");
    if (!set_loc_fn) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx.builder().getVoidTy(),
            {ctx.builder().getPtrTy(), ctx.int32Type(), ctx.int32Type()},
            false);
        set_loc_fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
            "eshkol_set_error_location", &ctx.module());
    }

    const std::string& file = ctx.currentSourceFile();
    llvm::Value* file_val = file.empty()
        ? static_cast<llvm::Value*>(llvm::ConstantPointerNull::get(ctx.builder().getPtrTy()))
        : static_cast<llvm::Value*>(ctx.builder().CreateGlobalString(file, "axis_int_file"));
    ctx.builder().CreateCall(set_loc_fn, {
        file_val,
        llvm::ConstantInt::get(ctx.int32Type(), line),
        llvm::ConstantInt::get(ctx.int32Type(), ctx.currentSourceColumn())});
}

llvm::Value* extractStructuralIntOrRaise(CodegenContext& ctx,
                                         TaggedValueCodegen& tagged,
                                         llvm::Value* value,
                                         const char* proc_name,
                                         const char* expected_type) {
    if (!value) {
        return llvm::ConstantInt::get(ctx.int64Type(), 0);
    }

    llvm::Type* value_type = value->getType();
    if (value_type->isIntegerTy(64)) {
        return value;
    }
    if (value_type->isIntegerTy()) {
        return ctx.builder().CreateSExtOrTrunc(value, ctx.int64Type());
    }

    if (value_type != ctx.taggedValueType()) {
        llvm::Function* func = ctx.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* err_block = llvm::BasicBlock::Create(
            ctx.context(), "axis_type_error", func);
        ctx.builder().CreateBr(err_block);
        ctx.builder().SetInsertPoint(err_block);
        emitStructuralIntErrorLocation(ctx);

        llvm::Function* type_error_fn = ctx.module().getFunction("eshkol_type_error");
        if (!type_error_fn) {
            llvm::FunctionType* ft = llvm::FunctionType::get(
                ctx.builder().getVoidTy(),
                {ctx.builder().getPtrTy(), ctx.builder().getPtrTy()},
                false);
            type_error_fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                "eshkol_type_error", &ctx.module());
            type_error_fn->setDoesNotReturn();
        }

        llvm::Value* proc = ctx.builder().CreateGlobalString(proc_name, "axis_int_proc");
        llvm::Value* expected = ctx.builder().CreateGlobalString(expected_type, "axis_int_expected");
        ctx.builder().CreateCall(type_error_fn, {proc, expected});
        ctx.builder().CreateUnreachable();
        return llvm::ConstantInt::get(ctx.int64Type(), 0);
    }

    llvm::Value* type_tag = tagged.getType(value);
    llvm::Value* base_type = tagged.getBaseType(type_tag);
    llvm::Value* is_int = ctx.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx.int8Type(), ESHKOL_VALUE_INT64));

    llvm::Function* func = ctx.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* ok_block = llvm::BasicBlock::Create(
        ctx.context(), "axis_int_ok", func);
    llvm::BasicBlock* err_block = llvm::BasicBlock::Create(
        ctx.context(), "axis_type_error", func);
    ctx.builder().CreateCondBr(is_int, ok_block, err_block);

    ctx.builder().SetInsertPoint(err_block);
    emitStructuralIntErrorLocation(ctx);
    llvm::Function* type_error_fn = ctx.module().getFunction("eshkol_type_error_with_operand");
    if (!type_error_fn) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx.builder().getVoidTy(),
            {ctx.builder().getPtrTy(), ctx.builder().getPtrTy(), ctx.builder().getPtrTy()},
            false);
        type_error_fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
            "eshkol_type_error_with_operand", &ctx.module());
        type_error_fn->setDoesNotReturn();
    }
    llvm::Value* slot = ctx.builder().CreateAlloca(ctx.taggedValueType(), nullptr, "axis_err_slot");
    ctx.builder().CreateStore(value, slot);
    llvm::Value* proc = ctx.builder().CreateGlobalString(proc_name, "axis_int_proc");
    llvm::Value* expected = ctx.builder().CreateGlobalString(expected_type, "axis_int_expected");
    ctx.builder().CreateCall(type_error_fn, {proc, expected, slot});
    ctx.builder().CreateUnreachable();

    ctx.builder().SetInsertPoint(ok_block);
    return tagged.unpackInt64(value);
}

} // namespace

llvm::Value* TensorCodegen::adNodeFromTensorElementBits(llvm::Value* elem_bits, const std::string& name) {
    if (!autodiff_ || !elem_bits) {
        return nullptr;
    }

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* check_large = llvm::BasicBlock::Create(ctx_.context(), name + "_check_large", current_func);
    llvm::BasicBlock* const_small = llvm::BasicBlock::Create(ctx_.context(), name + "_const_small", current_func);
    llvm::BasicBlock* const_double = llvm::BasicBlock::Create(ctx_.context(), name + "_const_double", current_func);
    llvm::BasicBlock* existing_node = llvm::BasicBlock::Create(ctx_.context(), name + "_existing_node", current_func);
    llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), name + "_merge", current_func);

    llvm::Value* is_small = ctx_.builder().CreateICmpULT(elem_bits,
        llvm::ConstantInt::get(ctx_.int64Type(), 1000));
    ctx_.builder().CreateCondBr(is_small, const_small, check_large);

    ctx_.builder().SetInsertPoint(const_small);
    llvm::Value* small_as_double = ctx_.builder().CreateSIToFP(elem_bits, ctx_.doubleType());
    llvm::Value* small_node = autodiff_->createADConstant(small_as_double);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* const_small_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(check_large);
    llvm::Value* exponent_mask = llvm::ConstantInt::get(ctx_.int64Type(), 0x7FF0000000000000ULL);
    llvm::Value* exponent_bits = ctx_.builder().CreateAnd(elem_bits, exponent_mask);
    llvm::Value* has_exponent = ctx_.builder().CreateICmpNE(exponent_bits,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* no_exponent = ctx_.builder().CreateNot(has_exponent);
    llvm::Value* non_zero = ctx_.builder().CreateICmpNE(elem_bits,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* below_pointer_ceiling = ctx_.builder().CreateICmpULT(elem_bits,
        llvm::ConstantInt::get(ctx_.int64Type(), 0x0001000000000000ULL));
    llvm::Value* pointer_like = ctx_.builder().CreateAnd(no_exponent,
        ctx_.builder().CreateAnd(non_zero, below_pointer_ceiling));
    ctx_.builder().CreateCondBr(pointer_like, existing_node, const_double);

    ctx_.builder().SetInsertPoint(const_double);
    llvm::Value* double_value = ctx_.builder().CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* double_node = autodiff_->createADConstant(double_value);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* const_double_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(existing_node);
    llvm::Value* ptr_node = ctx_.builder().CreateIntToPtr(elem_bits, ctx_.ptrType());
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* existing_node_exit = ctx_.builder().GetInsertBlock();

    ctx_.builder().SetInsertPoint(merge);
    llvm::PHINode* node_phi = ctx_.builder().CreatePHI(ctx_.ptrType(), 3, name + "_node");
    node_phi->addIncoming(small_node, const_small_exit);
    node_phi->addIncoming(double_node, const_double_exit);
    node_phi->addIncoming(ptr_node, existing_node_exit);
    return node_phi;
}

// ===== SIMD-ACCELERATED MATRIX MULTIPLICATION =====
// Vectorizes the j-loop (columns) to process SIMD_WIDTH columns at a time
// C[i][j:j+SIMD_WIDTH] += A[i][k] * B[k][j:j+SIMD_WIDTH]
// SIMD width is auto-detected: 2 (NEON/SSE2), 4 (AVX), or 8 (AVX-512)
// Falls back to pure scalar when SIMD_WIDTH == 1
llvm::Value* TensorCodegen::matmulSIMD(llvm::Value* ptr_a, llvm::Value* ptr_b,
                                        llvm::Value* M, llvm::Value* K, llvm::Value* N) {
    auto& builder = ctx_.builder();
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get elements pointers from tensors
    llvm::Value* a_elements_field = builder.CreateStructGEP(tensor_type, ptr_a, 2);
    llvm::Value* a_elements = builder.CreateLoad(ctx_.ptrType(), a_elements_field);

    llvm::Value* b_elements_field = builder.CreateStructGEP(tensor_type, ptr_b, 2);
    llvm::Value* b_elements = builder.CreateLoad(ctx_.ptrType(), b_elements_field);

    // Create result tensor [M x N] with zero-fill
    std::vector<llvm::Value*> result_dims = {M, N};
    llvm::Value* result_ptr = createTensorWithDims(result_dims, nullptr, true);
    if (!result_ptr) return tagged_.packNull();

    llvm::Value* result_elements_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    llvm::Value* result_elements = builder.CreateLoad(ctx_.ptrType(), result_elements_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Create basic blocks (some only used in SIMD mode)
    llvm::BasicBlock* i_cond = llvm::BasicBlock::Create(ctx_.context(), "mm_i_cond", current_func);
    llvm::BasicBlock* i_body = llvm::BasicBlock::Create(ctx_.context(), "mm_i_body", current_func);
    llvm::BasicBlock* k_cond = llvm::BasicBlock::Create(ctx_.context(), "mm_k_cond", current_func);
    llvm::BasicBlock* k_body = llvm::BasicBlock::Create(ctx_.context(), "mm_k_body", current_func);
    llvm::BasicBlock* j_scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "mm_j_scalar_cond", current_func);
    llvm::BasicBlock* j_scalar_body = llvm::BasicBlock::Create(ctx_.context(), "mm_j_scalar_body", current_func);
    llvm::BasicBlock* k_inc = llvm::BasicBlock::Create(ctx_.context(), "mm_k_inc", current_func);
    llvm::BasicBlock* i_inc = llvm::BasicBlock::Create(ctx_.context(), "mm_i_inc", current_func);
    llvm::BasicBlock* final_exit = llvm::BasicBlock::Create(ctx_.context(), "mm_exit", current_func);

    // SIMD-specific blocks (only created when using SIMD)
    llvm::BasicBlock* j_simd_cond = nullptr;
    llvm::BasicBlock* j_simd_body = nullptr;
    llvm::BasicBlock* j_simd_exit = nullptr;
    llvm::Value* N_simd_elems = nullptr;

    if (use_simd) {
        j_simd_cond = llvm::BasicBlock::Create(ctx_.context(), "mm_j_simd_cond", current_func);
        j_simd_body = llvm::BasicBlock::Create(ctx_.context(), "mm_j_simd_body", current_func);
        j_simd_exit = llvm::BasicBlock::Create(ctx_.context(), "mm_j_simd_exit", current_func);

        // Calculate N_simd_elems = (N / SIMD_WIDTH) * SIMD_WIDTH
        llvm::Value* N_simd = builder.CreateUDiv(N, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        N_simd_elems = builder.CreateMul(N_simd, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
    }

    // Allocate loop counters
    llvm::Value* i_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mm_i");
    llvm::Value* k_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mm_k");
    llvm::Value* j_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mm_j");

    // Initialize i = 0
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_counter);
    builder.CreateBr(i_cond);

    // ===== i loop (rows of A/C) =====
    builder.SetInsertPoint(i_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_counter);
    llvm::Value* i_cmp = builder.CreateICmpULT(i, M);
    builder.CreateCondBr(i_cmp, i_body, final_exit);

    builder.SetInsertPoint(i_body);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_counter);
    builder.CreateBr(k_cond);

    // ===== k loop (inner product dimension) =====
    builder.SetInsertPoint(k_cond);
    llvm::Value* k = builder.CreateLoad(ctx_.int64Type(), k_counter);
    llvm::Value* k_cmp = builder.CreateICmpULT(k, K);
    builder.CreateCondBr(k_cmp, k_body, i_inc);

    builder.SetInsertPoint(k_body);
    // Load A[i][k]
    // A[i][k] = A[i * K + k]
    llvm::Value* a_idx = builder.CreateMul(i, K);
    a_idx = builder.CreateAdd(a_idx, k);
    llvm::Value* a_ptr = builder.CreateGEP(ctx_.doubleType(), a_elements, a_idx);
    llvm::Value* a_val = builder.CreateLoad(ctx_.doubleType(), a_ptr);

    // Initialize j = 0
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_counter);

    if (use_simd) {
        // Broadcast a_val to vector for SIMD path
        llvm::Value* a_vec = llvm::UndefValue::get(vec_type);
        for (unsigned lane = 0; lane < SIMD_WIDTH; ++lane) {
            a_vec = builder.CreateInsertElement(a_vec, a_val,
                llvm::ConstantInt::get(ctx_.int32Type(), lane));
        }

        builder.CreateBr(j_simd_cond);

        // ===== j SIMD loop (process SIMD_WIDTH columns at a time) =====
        builder.SetInsertPoint(j_simd_cond);
        llvm::Value* j = builder.CreateLoad(ctx_.int64Type(), j_counter);
        llvm::Value* j_simd_cmp = builder.CreateICmpULT(j, N_simd_elems);
        builder.CreateCondBr(j_simd_cmp, j_simd_body, j_simd_exit);

        builder.SetInsertPoint(j_simd_body);
        // Load B[k][j:j+SIMD_WIDTH] - consecutive elements from row k
        // B[k][j] = B[k * N + j]
        llvm::Value* b_idx = builder.CreateMul(k, N);
        b_idx = builder.CreateAdd(b_idx, j);
        llvm::Value* b_ptr = builder.CreateGEP(ctx_.doubleType(), b_elements, b_idx);
        llvm::Value* b_vec = builder.CreateAlignedLoad(vec_type, b_ptr, llvm::MaybeAlign(8), "b_vec");

        // Load C[i][j:j+SIMD_WIDTH] - current accumulator
        // C[i][j] = C[i * N + j]
        llvm::Value* c_idx = builder.CreateMul(i, N);
        c_idx = builder.CreateAdd(c_idx, j);
        llvm::Value* c_ptr = builder.CreateGEP(ctx_.doubleType(), result_elements, c_idx);
        llvm::Value* c_vec = builder.CreateAlignedLoad(vec_type, c_ptr, llvm::MaybeAlign(8), "c_vec");

        // C[i][j:j+SIMD_WIDTH] += A[i][k] * B[k][j:j+SIMD_WIDTH]
        llvm::Value* prod = builder.CreateFMul(a_vec, b_vec, "ab_prod");
        llvm::Value* new_c = builder.CreateFAdd(c_vec, prod, "c_acc");

        // Store result
        builder.CreateAlignedStore(new_c, c_ptr, llvm::MaybeAlign(8));

        // j += SIMD_WIDTH
        llvm::Value* next_j = builder.CreateAdd(j, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        builder.CreateStore(next_j, j_counter);
        builder.CreateBr(j_simd_cond);

        // ===== j scalar loop entry from SIMD =====
        builder.SetInsertPoint(j_simd_exit);
        // j_counter already has N_simd_elems value
        builder.CreateBr(j_scalar_cond);
    } else {
        // ===== SCALAR-ONLY PATH =====
        // Skip SIMD, go directly to scalar loop
        builder.CreateBr(j_scalar_cond);
    }

    builder.SetInsertPoint(j_scalar_cond);
    llvm::Value* js = builder.CreateLoad(ctx_.int64Type(), j_counter);
    llvm::Value* j_scalar_cmp = builder.CreateICmpULT(js, N);
    builder.CreateCondBr(j_scalar_cmp, j_scalar_body, k_inc);

    builder.SetInsertPoint(j_scalar_body);
    // Scalar: C[i][j] += A[i][k] * B[k][j]
    llvm::Value* bs_idx = builder.CreateMul(k, N);
    bs_idx = builder.CreateAdd(bs_idx, js);
    llvm::Value* bs_ptr = builder.CreateGEP(ctx_.doubleType(), b_elements, bs_idx);
    llvm::Value* bs_val = builder.CreateLoad(ctx_.doubleType(), bs_ptr);

    llvm::Value* cs_idx = builder.CreateMul(i, N);
    cs_idx = builder.CreateAdd(cs_idx, js);
    llvm::Value* cs_ptr = builder.CreateGEP(ctx_.doubleType(), result_elements, cs_idx);
    llvm::Value* cs_val = builder.CreateLoad(ctx_.doubleType(), cs_ptr);

    llvm::Value* prods = builder.CreateFMul(a_val, bs_val);
    llvm::Value* new_cs = builder.CreateFAdd(cs_val, prods);
    builder.CreateStore(new_cs, cs_ptr);

    llvm::Value* next_js = builder.CreateAdd(js, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_js, j_counter);
    builder.CreateBr(j_scalar_cond);

    // ===== k increment =====
    builder.SetInsertPoint(k_inc);
    llvm::Value* k_val = builder.CreateLoad(ctx_.int64Type(), k_counter);
    llvm::Value* next_k = builder.CreateAdd(k_val, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_k, k_counter);
    builder.CreateBr(k_cond);

    // ===== i increment =====
    builder.SetInsertPoint(i_inc);
    llvm::Value* i_val = builder.CreateLoad(ctx_.int64Type(), i_counter);
    llvm::Value* next_i = builder.CreateAdd(i_val, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, i_counter);
    builder.CreateBr(i_cond);

    // ===== Final exit - return packed result =====
    builder.SetInsertPoint(final_exit);
    return tagged_.packHeapPtr(result_ptr);
}

// Main entry point: dispatches based on type (VECTOR_PTR vs TENSOR_PTR)
llvm::Value* TensorCodegen::tensorArithmeticInternal(llvm::Value* arg1, llvm::Value* arg2, const std::string& operation) {
    if (!arg1 || !arg2) return tagged_.packNull();

    // Ensure they're tagged values so we can check type at runtime
    if (arg1->getType() != ctx_.taggedValueType()) {
        arg1 = tagged_.packInt64(arg1, true);
    }
    if (arg2->getType() != ctx_.taggedValueType()) {
        arg2 = tagged_.packInt64(arg2, true);
    }

    // Check type of first argument at RUNTIME (using consolidated type check)
    llvm::Value* is_vector = tagged_.isVector(arg1);

    // Branch based on type
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* vector_path = llvm::BasicBlock::Create(ctx_.context(), "int_arith_vec_path", current_func);
    llvm::BasicBlock* tensor_path = llvm::BasicBlock::Create(ctx_.context(), "int_arith_tensor_path", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "int_arith_merge", current_func);

    // Store result in alloca for merge
    llvm::Value* result_alloca = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "int_arith_result");

    ctx_.builder().CreateCondBr(is_vector, vector_path, tensor_path);

    // === VECTOR PATH ===
    ctx_.builder().SetInsertPoint(vector_path);
    llvm::Value* vec_result = schemeVectorArithmetic(arg1, arg2, operation);
    ctx_.builder().CreateStore(vec_result, result_alloca);
    ctx_.builder().CreateBr(merge_block);

    // === TENSOR PATH ===
    ctx_.builder().SetInsertPoint(tensor_path);

    // ESH-0069: validate/coerce both operands ONCE here so every downstream
    // path (XLA, SIMD, scalar) receives a genuine tensor. A non-tensor,
    // non-numeric-vector operand raises a catchable type error instead of
    // segfaulting when rawTensorArithmetic* reinterprets it as a struct; a
    // homogeneous numeric vector is coerced to a 1-D tensor.
    {
        std::string arith_name = "tensor-" + operation;
        llvm::Value* a1_ptr = unpackTensorOperandChecked(arg1, arith_name.c_str());
        llvm::Value* a2_ptr = unpackTensorOperandChecked(arg2, arith_name.c_str());
        arg1 = tagged_.packHeapPtr(a1_ptr);
        arg2 = tagged_.packHeapPtr(a2_ptr);
    }

#ifdef ESHKOL_XLA_ENABLED
    // ===== XLA DISPATCH FOR LARGE TENSORS =====
    // Dispatch hierarchy: XLA (≥100K elements) → SIMD → scalar
    if (xla_ && xla_->isAvailable()) {
        // Map string operation to XLA ElementwiseOp
        xla::ElementwiseOp xla_op;
        bool has_xla_op = true;
        if (operation == "add") {
            xla_op = xla::ElementwiseOp::ADD;
        } else if (operation == "sub") {
            xla_op = xla::ElementwiseOp::SUB;
        } else if (operation == "mul") {
            xla_op = xla::ElementwiseOp::MUL;
        } else if (operation == "div") {
            xla_op = xla::ElementwiseOp::DIV;
        } else {
            has_xla_op = false;
        }

        if (has_xla_op) {
            // Extract tensor pointer and total elements to check threshold
            llvm::Value* t1_int = tagged_.unpackInt64(arg1);
            llvm::Value* t1_ptr = ctx_.builder().CreateIntToPtr(t1_int, ctx_.ptrType());
            llvm::StructType* tensor_type = ctx_.tensorType();
            llvm::Value* total_ptr = ctx_.builder().CreateStructGEP(tensor_type, t1_ptr,
                TypeSystem::TENSOR_TOTAL_ELEMENTS_IDX, "arith_total_ptr");
            llvm::Value* total_elements = ctx_.builder().CreateLoad(ctx_.int64Type(), total_ptr, "arith_total");

            llvm::Value* threshold = llvm::ConstantInt::get(ctx_.int64Type(), xla::xla_get_threshold());
            llvm::Value* use_xla = ctx_.builder().CreateICmpUGE(total_elements, threshold);

            llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
            llvm::BasicBlock* xla_block = llvm::BasicBlock::Create(ctx_.context(), "arith_xla", current_func);
            llvm::BasicBlock* simd_block = llvm::BasicBlock::Create(ctx_.context(), "arith_simd_fallback", current_func);

            ctx_.builder().CreateCondBr(use_xla, xla_block, simd_block);

            // XLA path: emit elementwise operation
            ctx_.builder().SetInsertPoint(xla_block);
            llvm::Value* t2_int = tagged_.unpackInt64(arg2);
            llvm::Value* t2_ptr = ctx_.builder().CreateIntToPtr(t2_int, ctx_.ptrType());
            llvm::Value* xla_result = xla_->emitElementwise(t1_ptr, t2_ptr, xla_op);
            if (xla_result) {
                llvm::Value* xla_packed = tagged_.packHeapPtr(xla_result);
                ctx_.builder().CreateStore(xla_packed, result_alloca);
                ctx_.builder().CreateBr(merge_block);
            } else {
                // XLA returned nullptr, fall back to SIMD
                ctx_.builder().CreateBr(simd_block);
            }

            // SIMD fallback
            ctx_.builder().SetInsertPoint(simd_block);
            llvm::Value* simd_result = rawTensorArithmeticSIMD(arg1, arg2, operation);
            ctx_.builder().CreateStore(simd_result, result_alloca);
            ctx_.builder().CreateBr(merge_block);
        } else {
            // Operation not mapped to XLA, use SIMD directly
            llvm::Value* tensor_result = rawTensorArithmeticSIMD(arg1, arg2, operation);
            ctx_.builder().CreateStore(tensor_result, result_alloca);
            ctx_.builder().CreateBr(merge_block);
        }
    } else {
#endif
        // No XLA available, SIMD path only
        llvm::Value* tensor_result = rawTensorArithmeticSIMD(arg1, arg2, operation);
        ctx_.builder().CreateStore(tensor_result, result_alloca);
        ctx_.builder().CreateBr(merge_block);
#ifdef ESHKOL_XLA_ENABLED
    }
#endif

    // === MERGE BLOCK ===
    ctx_.builder().SetInsertPoint(merge_block);
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_alloca);
}

llvm::Value* TensorCodegen::tensorDot(const eshkol_operations_t* op) {
    // tensor-dot: (tensor-dot A B) - Dot product for 1D vectors, matrix multiplication for 2D
    if (op->call_op.num_vars != 2) {
        eshkol_error("tensor-dot requires exactly 2 arguments: tensor A and tensor B");
        return nullptr;
    }

    llvm::Value* val_a = codegenAST(&op->call_op.variables[0]);
    llvm::Value* val_b = codegenAST(&op->call_op.variables[1]);
    if (!val_a || !val_b) return nullptr;

    // Check type of first argument: Scheme vector vs Tensor (using consolidated type checks)
    llvm::Value* is_scheme_vector = tagged_.isVector(val_a);

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* scheme_vec_block = llvm::BasicBlock::Create(ctx_.context(), "dot_scheme_vec", current_func);
    llvm::BasicBlock* tensor_block = llvm::BasicBlock::Create(ctx_.context(), "dot_tensor", current_func);
    llvm::BasicBlock* final_merge = llvm::BasicBlock::Create(ctx_.context(), "dot_final_merge", current_func);

    ctx_.builder().CreateCondBr(is_scheme_vector, scheme_vec_block, tensor_block);

    // === SCHEME VECTOR PATH ===
    ctx_.builder().SetInsertPoint(scheme_vec_block);
    llvm::Value* svec_a_ptr_int = tagged_.unpackInt64(val_a);
    llvm::Value* svec_a_ptr = ctx_.builder().CreateIntToPtr(svec_a_ptr_int, ctx_.ptrType());
    llvm::Value* svec_b_ptr_int = tagged_.unpackInt64(val_b);
    llvm::Value* svec_b_ptr = ctx_.builder().CreateIntToPtr(svec_b_ptr_int, ctx_.ptrType());

    // Scheme vector: [length:i64, element0:tagged_value, element1:tagged_value, ...]
    llvm::Value* svec_len = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_a_ptr);

    // Dot product loop for Scheme vectors
    llvm::BasicBlock* svec_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "svec_dot_cond", current_func);
    llvm::BasicBlock* svec_loop_body = llvm::BasicBlock::Create(ctx_.context(), "svec_dot_body", current_func);
    llvm::BasicBlock* svec_loop_exit = llvm::BasicBlock::Create(ctx_.context(), "svec_dot_exit", current_func);

    // ESH-0121: tagged accumulator so a Scheme vector of DUAL_NUMBER jets dots
    // to a dual (preserving the mixed e1e2 second-order term) instead of being
    // flattened to a plain double, which silently zeros the Hessian.
    llvm::Value* svec_sum = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "svec_dot_acc");
    llvm::Value* svec_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "svec_dot_i");
    ctx_.builder().CreateStore(tagged_.packDouble(llvm::ConstantFP::get(ctx_.doubleType(), 0.0)), svec_sum);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), svec_counter);
    ctx_.builder().CreateBr(svec_loop_cond);

    ctx_.builder().SetInsertPoint(svec_loop_cond);
    llvm::Value* svec_i = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_counter);
    llvm::Value* svec_cmp = ctx_.builder().CreateICmpULT(svec_i, svec_len);
    ctx_.builder().CreateCondBr(svec_cmp, svec_loop_body, svec_loop_exit);

    ctx_.builder().SetInsertPoint(svec_loop_body);
    // Elements start after 8-byte length field, then index by tagged_value size
    llvm::Value* svec_a_elems_base = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_a_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* svec_a_elems_typed = ctx_.builder().CreatePointerCast(svec_a_elems_base, ctx_.ptrType());
    llvm::Value* svec_a_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_a_elems_typed, svec_i);
    llvm::Value* svec_a_elem_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_a_elem_ptr);

    llvm::Value* svec_b_elems_base = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_b_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* svec_b_elems_typed = ctx_.builder().CreatePointerCast(svec_b_elems_base, ctx_.ptrType());
    llvm::Value* svec_b_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_b_elems_typed, svec_i);
    llvm::Value* svec_b_elem_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_b_elem_ptr);

    // Multiply and accumulate (dual-aware).
    llvm::Value* svec_product = dualAwareScalarBinOp(svec_a_elem_tagged, svec_b_elem_tagged, "mul");
    llvm::Value* svec_current_sum = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_sum);
    llvm::Value* svec_new_sum = dualAwareScalarBinOp(svec_current_sum, svec_product, "add");
    ctx_.builder().CreateStore(svec_new_sum, svec_sum);
    llvm::Value* svec_next_i = ctx_.builder().CreateAdd(svec_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(svec_next_i, svec_counter);
    ctx_.builder().CreateBr(svec_loop_cond);

    ctx_.builder().SetInsertPoint(svec_loop_exit);
    llvm::Value* svec_result = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_sum);
    // Don't branch yet - we'll branch to scalar_merge once it's created
    llvm::BasicBlock* svec_exit_block = ctx_.builder().GetInsertBlock();

    // === TENSOR PATH ===
    ctx_.builder().SetInsertPoint(tensor_block);
    llvm::Value* tensor_a_ptr = unpackTensorOperandChecked(val_a, "tensor-dot");
    llvm::Value* tensor_b_ptr = unpackTensorOperandChecked(val_b, "tensor-dot");

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get tensor A properties
    llvm::Value* a_num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_a_ptr, 1);
    llvm::Value* a_num_dims = ctx_.builder().CreateLoad(ctx_.int64Type(), a_num_dims_field_ptr);

    llvm::Value* a_total_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_a_ptr, 3);
    llvm::Value* a_total = ctx_.builder().CreateLoad(ctx_.int64Type(), a_total_field_ptr);

    llvm::Value* a_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_a_ptr, 2);
    llvm::Value* a_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), a_elements_field_ptr);
    llvm::Value* typed_a_elements_ptr = ctx_.builder().CreatePointerCast(a_elements_ptr, ctx_.ptrType());

    // Get tensor B properties
    llvm::Value* b_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_b_ptr, 2);
    llvm::Value* b_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), b_elements_field_ptr);
    llvm::Value* typed_b_elements_ptr = ctx_.builder().CreatePointerCast(b_elements_ptr, ctx_.ptrType());

    // Check if BOTH are 1D vectors - use simple dot product
    // If either operand is 2D+, use the matmul path (which handles mixed 1D×2D via PEP 465)
    llvm::Value* b_num_dims_early_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_b_ptr, 1);
    llvm::Value* b_num_dims_early = ctx_.builder().CreateLoad(ctx_.int64Type(), b_num_dims_early_ptr);
    llvm::Value* a_is_1d_early = ctx_.builder().CreateICmpEQ(a_num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* b_is_1d_early = ctx_.builder().CreateICmpEQ(b_num_dims_early, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* is_1d = ctx_.builder().CreateAnd(a_is_1d_early, b_is_1d_early);

    llvm::BasicBlock* dot_1d_block = llvm::BasicBlock::Create(ctx_.context(), "dot_1d", current_func);
    llvm::BasicBlock* dot_2d_block = llvm::BasicBlock::Create(ctx_.context(), "dot_2d", current_func);
    llvm::BasicBlock* tensor_merge = llvm::BasicBlock::Create(ctx_.context(), "tensor_dot_merge", current_func);

    ctx_.builder().CreateCondBr(is_1d, dot_1d_block, dot_2d_block);

    // 1D Vector Dot Product: sum(a[i] * b[i]) - SIMD Accelerated
    // Width is auto-detected: 2 (NEON/SSE2), 4 (AVX), or 8 (AVX-512)
    // Falls back to scalar when SIMD_WIDTH == 1
    ctx_.builder().SetInsertPoint(dot_1d_block);

    llvm::Value* ad_dot_tagged_result = nullptr;
    llvm::BasicBlock* ad_dot_exit_block = nullptr;
    if (autodiff_) {
        llvm::Value* in_ad_mode = ctx_.builder().CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
        llvm::BasicBlock* ad_dot_block = llvm::BasicBlock::Create(ctx_.context(), "dot1d_ad", current_func);
        llvm::BasicBlock* numeric_dot_block = llvm::BasicBlock::Create(ctx_.context(), "dot1d_numeric", current_func);
        ctx_.builder().CreateCondBr(in_ad_mode, ad_dot_block, numeric_dot_block);

        ctx_.builder().SetInsertPoint(ad_dot_block);
        llvm::Value* ad_acc = ctx_.builder().CreateAlloca(ctx_.ptrType(), nullptr, "dot1d_ad_acc");
        llvm::Value* ad_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "dot1d_ad_i");
        llvm::Value* zero_node = autodiff_->createADConstant(llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        ctx_.builder().CreateStore(zero_node, ad_acc);
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ad_counter);

        llvm::BasicBlock* ad_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "dot1d_ad_cond", current_func);
        llvm::BasicBlock* ad_loop_body = llvm::BasicBlock::Create(ctx_.context(), "dot1d_ad_body", current_func);
        llvm::BasicBlock* ad_loop_exit = llvm::BasicBlock::Create(ctx_.context(), "dot1d_ad_exit", current_func);
        ctx_.builder().CreateBr(ad_loop_cond);

        ctx_.builder().SetInsertPoint(ad_loop_cond);
        llvm::Value* ad_i = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_counter);
        llvm::Value* ad_has_elem = ctx_.builder().CreateICmpULT(ad_i, a_total);
        ctx_.builder().CreateCondBr(ad_has_elem, ad_loop_body, ad_loop_exit);

        ctx_.builder().SetInsertPoint(ad_loop_body);
        llvm::Value* ad_a_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_a_elements_ptr, ad_i);
        llvm::Value* ad_b_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_b_elements_ptr, ad_i);
        llvm::Value* ad_a_elem_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_a_elem_ptr);
        llvm::Value* ad_b_elem_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_b_elem_ptr);
        llvm::Value* ad_a_node = adNodeFromTensorElementBits(ad_a_elem_bits, "dot1d_ad_a");
        llvm::Value* ad_b_node = adNodeFromTensorElementBits(ad_b_elem_bits, "dot1d_ad_b");
        llvm::Value* ad_product = autodiff_->recordADNodeBinary(4, ad_a_node, ad_b_node);
        llvm::Value* ad_old_acc = ctx_.builder().CreateLoad(ctx_.ptrType(), ad_acc);
        llvm::Value* ad_new_acc = autodiff_->recordADNodeBinary(2, ad_old_acc, ad_product);
        ctx_.builder().CreateStore(ad_new_acc, ad_acc);
        llvm::Value* ad_next_i = ctx_.builder().CreateAdd(ad_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        ctx_.builder().CreateStore(ad_next_i, ad_counter);
        ctx_.builder().CreateBr(ad_loop_cond);

        ctx_.builder().SetInsertPoint(ad_loop_exit);
        llvm::Value* final_ad_dot = ctx_.builder().CreateLoad(ctx_.ptrType(), ad_acc);
        ad_dot_tagged_result = tagged_.packPtr(final_ad_dot, ESHKOL_VALUE_CALLABLE);
        ctx_.builder().CreateBr(tensor_merge);
        ad_dot_exit_block = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(numeric_dot_block);
    }

    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Scalar loop blocks (always needed)
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "dot1d_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "dot1d_scalar_body", current_func);
    llvm::BasicBlock* loop_exit_1d = llvm::BasicBlock::Create(ctx_.context(), "dot1d_exit", current_func);

    // Sum accumulator and scalar counter
    llvm::Value* sum_alloca = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "dot_sum");
    llvm::Value* scalar_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "dot1d_scalar_i");

    if (use_simd) {
        // Calculate SIMD iteration count
        llvm::Value* simd_count = ctx_.builder().CreateUDiv(a_total,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        llvm::Value* simd_elements = ctx_.builder().CreateMul(simd_count,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

        // Initialize vector accumulator
        llvm::Value* vec_acc_alloca = ctx_.builder().CreateAlloca(vec_type, nullptr, "dot_vec_acc");
        llvm::Value* zero_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH), llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        ctx_.builder().CreateStore(zero_vec, vec_acc_alloca);

        // SIMD loop blocks
        llvm::BasicBlock* simd_cond = llvm::BasicBlock::Create(ctx_.context(), "dot1d_simd_cond", current_func);
        llvm::BasicBlock* simd_body = llvm::BasicBlock::Create(ctx_.context(), "dot1d_simd_body", current_func);
        llvm::BasicBlock* simd_exit = llvm::BasicBlock::Create(ctx_.context(), "dot1d_simd_exit", current_func);

        llvm::Value* simd_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "dot1d_simd_i");
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), simd_counter);
        ctx_.builder().CreateBr(simd_cond);

        // SIMD loop condition
        ctx_.builder().SetInsertPoint(simd_cond);
        llvm::Value* simd_i = ctx_.builder().CreateLoad(ctx_.int64Type(), simd_counter);
        llvm::Value* simd_cmp = ctx_.builder().CreateICmpULT(simd_i, simd_elements);
        ctx_.builder().CreateCondBr(simd_cmp, simd_body, simd_exit);

        // SIMD loop body: load SIMD_WIDTH elements, multiply, accumulate
        ctx_.builder().SetInsertPoint(simd_body);
        llvm::Value* a_vec_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_a_elements_ptr, simd_i);
        llvm::Value* b_vec_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_b_elements_ptr, simd_i);

        llvm::Value* a_vec = ctx_.builder().CreateAlignedLoad(vec_type, a_vec_ptr, llvm::MaybeAlign(8), "a_vec");
        llvm::Value* b_vec = ctx_.builder().CreateAlignedLoad(vec_type, b_vec_ptr, llvm::MaybeAlign(8), "b_vec");
        llvm::Value* prod_vec = ctx_.builder().CreateFMul(a_vec, b_vec, "prod_vec");

        llvm::Value* old_acc = ctx_.builder().CreateLoad(vec_type, vec_acc_alloca);
        llvm::Value* new_acc = ctx_.builder().CreateFAdd(old_acc, prod_vec, "acc_vec");
        ctx_.builder().CreateStore(new_acc, vec_acc_alloca);

        llvm::Value* next_simd_i = ctx_.builder().CreateAdd(simd_i,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        ctx_.builder().CreateStore(next_simd_i, simd_counter);
        auto* dotSimdBackEdge = ctx_.builder().CreateBr(simd_cond);
        attachLoopMetadata(dotSimdBackEdge, true, SIMD_WIDTH, false, 0);

        // SIMD exit: horizontal sum of vector accumulator
        ctx_.builder().SetInsertPoint(simd_exit);
        llvm::Value* final_vec = ctx_.builder().CreateLoad(vec_type, vec_acc_alloca);

        // Horizontal sum: v[0] + v[1] + ... + v[SIMD_WIDTH-1]
        llvm::Value* simd_sum = ctx_.builder().CreateExtractElement(final_vec, (uint64_t)0);
        for (unsigned lane = 1; lane < SIMD_WIDTH; ++lane) {
            llvm::Value* elem = ctx_.builder().CreateExtractElement(final_vec, (uint64_t)lane);
            simd_sum = ctx_.builder().CreateFAdd(simd_sum, elem);
        }

        // Store partial sum for scalar tail
        ctx_.builder().CreateStore(simd_sum, sum_alloca);

        // Scalar tail loop counter starts at simd_elements
        ctx_.builder().CreateStore(simd_elements, scalar_counter);
        ctx_.builder().CreateBr(scalar_cond);
    } else {
        // ===== SCALAR-ONLY PATH =====
        // Initialize sum to 0 and counter to 0
        ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_alloca);
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), scalar_counter);
        ctx_.builder().CreateBr(scalar_cond);
    }

    // Scalar tail loop condition
    ctx_.builder().SetInsertPoint(scalar_cond);
    llvm::Value* scalar_i = ctx_.builder().CreateLoad(ctx_.int64Type(), scalar_counter);
    llvm::Value* scalar_cmp = ctx_.builder().CreateICmpULT(scalar_i, a_total);
    ctx_.builder().CreateCondBr(scalar_cmp, scalar_body, loop_exit_1d);

    // Scalar tail loop body
    ctx_.builder().SetInsertPoint(scalar_body);
    llvm::Value* a_elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_a_elements_ptr, scalar_i);
    llvm::Value* b_elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_b_elements_ptr, scalar_i);
    llvm::Value* a_elem = ctx_.builder().CreateLoad(ctx_.doubleType(), a_elem_ptr);
    llvm::Value* b_elem = ctx_.builder().CreateLoad(ctx_.doubleType(), b_elem_ptr);
    llvm::Value* product = ctx_.builder().CreateFMul(a_elem, b_elem);
    llvm::Value* old_sum = ctx_.builder().CreateLoad(ctx_.doubleType(), sum_alloca);
    llvm::Value* new_sum = ctx_.builder().CreateFAdd(old_sum, product);
    ctx_.builder().CreateStore(new_sum, sum_alloca);

    llvm::Value* next_scalar_i = ctx_.builder().CreateAdd(scalar_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_scalar_i, scalar_counter);
    auto* dotScalarBackEdge = ctx_.builder().CreateBr(scalar_cond);
    attachLoopMetadata(dotScalarBackEdge, false, 0, true, 4);

    // Final result
    ctx_.builder().SetInsertPoint(loop_exit_1d);
    llvm::Value* dot_result_1d = ctx_.builder().CreateLoad(ctx_.doubleType(), sum_alloca);
    llvm::Value* dot_tagged_result_1d = tagged_.packDouble(dot_result_1d);
    ctx_.builder().CreateBr(tensor_merge);
    llvm::BasicBlock* dot_numeric_1d_exit = ctx_.builder().GetInsertBlock();

    // 2D Matrix Multiplication: C = A @ B
    // A is (M x K), B is (K x N), C is (M x N)
    // C[i,j] = sum_k(A[i,k] * B[k,j])
    ctx_.builder().SetInsertPoint(dot_2d_block);

    // Guard: tensor-dot matmul only supports 2D tensors. Emit runtime error for ndim > 2.
    {
        llvm::Value* too_many_dims = ctx_.builder().CreateICmpUGT(a_num_dims,
            llvm::ConstantInt::get(ctx_.int64Type(), 2));
        llvm::BasicBlock* matmul_dims_ok = llvm::BasicBlock::Create(ctx_.context(), "matmul_dims_ok", current_func);
        llvm::BasicBlock* matmul_dims_err = llvm::BasicBlock::Create(ctx_.context(), "matmul_dims_err", current_func);
        ctx_.builder().CreateCondBr(too_many_dims, matmul_dims_err, matmul_dims_ok);

        ctx_.builder().SetInsertPoint(matmul_dims_err);
        llvm::Function* printf_fn_matmul = ctx_.lookupFunction("printf");
        llvm::Function* exit_fn_matmul = ctx_.lookupFunction("exit");
        if (printf_fn_matmul && exit_fn_matmul) {
            llvm::Value* fmt = ctx_.builder().CreateGlobalString(
                "Error: tensor-dot matmul only supports 1D and 2D tensors (got %lld dimensions)\n");
            ctx_.builder().CreateCall(printf_fn_matmul, {fmt, a_num_dims});
            ctx_.builder().CreateCall(exit_fn_matmul, {llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx_.context()), 1)});
        }
        ctx_.builder().CreateUnreachable();

        ctx_.builder().SetInsertPoint(matmul_dims_ok);
    }

    // === SAFE DIMENSION EXTRACTION (conditional blocks, no out-of-bounds reads) ===
    // PEP 465 semantics: 1D vectors are promoted for matmul, then contracted from result
    // A 1D → row vector (1 × K); B 1D → column vector (K × 1)

    // --- Extract A dimensions ---
    llvm::Value* a_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_a_ptr, 0);
    llvm::Value* a_dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), a_dims_field_ptr);
    llvm::Value* a_is_1d = ctx_.builder().CreateICmpEQ(a_num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::BasicBlock* a_1d_bb = llvm::BasicBlock::Create(ctx_.context(), "mm_a_1d", current_func);
    llvm::BasicBlock* a_2d_bb = llvm::BasicBlock::Create(ctx_.context(), "mm_a_2d", current_func);
    llvm::BasicBlock* a_dims_merge = llvm::BasicBlock::Create(ctx_.context(), "mm_a_merge", current_func);
    ctx_.builder().CreateCondBr(a_is_1d, a_1d_bb, a_2d_bb);

    ctx_.builder().SetInsertPoint(a_1d_bb);
    llvm::Value* a_M_1d = llvm::ConstantInt::get(ctx_.int64Type(), 1);
    llvm::Value* a_K_1d = a_total;
    ctx_.builder().CreateBr(a_dims_merge);

    ctx_.builder().SetInsertPoint(a_2d_bb);
    llvm::Value* a_M_2d = ctx_.builder().CreateLoad(ctx_.int64Type(), a_dims_ptr);
    llvm::Value* a_K_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), a_dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* a_K_2d = ctx_.builder().CreateLoad(ctx_.int64Type(), a_K_ptr);
    ctx_.builder().CreateBr(a_dims_merge);

    ctx_.builder().SetInsertPoint(a_dims_merge);
    llvm::PHINode* a_rows = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "M");
    a_rows->addIncoming(a_M_1d, a_1d_bb);
    a_rows->addIncoming(a_M_2d, a_2d_bb);
    llvm::PHINode* a_cols = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "K");
    a_cols->addIncoming(a_K_1d, a_1d_bb);
    a_cols->addIncoming(a_K_2d, a_2d_bb);

    // --- Extract B dimensions ---
    llvm::Value* b_num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_b_ptr, 1);
    llvm::Value* b_num_dims = ctx_.builder().CreateLoad(ctx_.int64Type(), b_num_dims_field_ptr);
    llvm::Value* b_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_b_ptr, 0);
    llvm::Value* b_dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), b_dims_field_ptr);
    llvm::Value* b_dim0 = ctx_.builder().CreateLoad(ctx_.int64Type(), b_dims_ptr);
    llvm::Value* b_total_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_b_ptr, 3);
    llvm::Value* b_total = ctx_.builder().CreateLoad(ctx_.int64Type(), b_total_field_ptr);
    llvm::Value* b_is_1d = ctx_.builder().CreateICmpEQ(b_num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::BasicBlock* b_1d_bb = llvm::BasicBlock::Create(ctx_.context(), "mm_b_1d", current_func);
    llvm::BasicBlock* b_2d_bb = llvm::BasicBlock::Create(ctx_.context(), "mm_b_2d", current_func);
    llvm::BasicBlock* b_dims_merge = llvm::BasicBlock::Create(ctx_.context(), "mm_b_merge", current_func);
    ctx_.builder().CreateCondBr(b_is_1d, b_1d_bb, b_2d_bb);

    ctx_.builder().SetInsertPoint(b_1d_bb);
    llvm::Value* b_N_1d = llvm::ConstantInt::get(ctx_.int64Type(), 1);
    ctx_.builder().CreateBr(b_dims_merge);

    ctx_.builder().SetInsertPoint(b_2d_bb);
    llvm::Value* b_N_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), b_dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* b_N_2d = ctx_.builder().CreateLoad(ctx_.int64Type(), b_N_ptr);
    ctx_.builder().CreateBr(b_dims_merge);

    ctx_.builder().SetInsertPoint(b_dims_merge);
    llvm::PHINode* b_cols = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "N");
    b_cols->addIncoming(b_N_1d, b_1d_bb);
    b_cols->addIncoming(b_N_2d, b_2d_bb);

    // --- K-dimension compatibility validation ---
    llvm::Value* k_match = ctx_.builder().CreateICmpEQ(a_cols, b_dim0);
    llvm::BasicBlock* shape_ok_bb = llvm::BasicBlock::Create(ctx_.context(), "mm_shape_ok", current_func);
    llvm::BasicBlock* shape_err_bb = llvm::BasicBlock::Create(ctx_.context(), "mm_shape_err", current_func);
    ctx_.builder().CreateCondBr(k_match, shape_ok_bb, shape_err_bb);

    ctx_.builder().SetInsertPoint(shape_err_bb);
    {
        llvm::Function* printf_fn_shape = ctx_.lookupFunction("printf");
        llvm::Function* exit_fn_shape = ctx_.lookupFunction("exit");
        if (printf_fn_shape && exit_fn_shape) {
            llvm::Value* fmt = ctx_.builder().CreateGlobalString(
                "Error: matmul inner dimensions mismatch (%lld vs %lld)\n");
            ctx_.builder().CreateCall(printf_fn_shape, {fmt, a_cols, b_dim0});
            ctx_.builder().CreateCall(exit_fn_shape, {llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx_.context()), 1)});
        }
        ctx_.builder().CreateUnreachable();
    }

    ctx_.builder().SetInsertPoint(shape_ok_bb);

    // Allocate the result tensor and element storage before dispatch. Numeric
    // matmul writes double bit patterns here; AD-mode matmul writes AD node
    // pointers in the same int64 element slots.
    llvm::Value* c_total = ctx_.builder().CreateMul(a_rows, b_cols);
    llvm::Value* dot_arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* c_tensor_ptr = ctx_.builder().CreateCall(alloc_tensor_func, {dot_arena_ptr}, "dot_tensor");
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* c_elements_size = ctx_.builder().CreateMul(c_total,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* c_elements_ptr = ctx_.builder().CreateCall(arena_alloc,
        {dot_arena_ptr, c_elements_size}, "dot_elems");

    llvm::BasicBlock* matmul_data_ready = nullptr;
    if (autodiff_) {
        llvm::Value* in_ad_mode = ctx_.builder().CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
        llvm::BasicBlock* ad_matmul_block = llvm::BasicBlock::Create(
            ctx_.context(), "matmul_ad", current_func);
        llvm::BasicBlock* numeric_matmul_block = llvm::BasicBlock::Create(
            ctx_.context(), "matmul_numeric", current_func);
        matmul_data_ready = llvm::BasicBlock::Create(
            ctx_.context(), "matmul_data_ready", current_func);
        ctx_.builder().CreateCondBr(in_ad_mode, ad_matmul_block, numeric_matmul_block);

        ctx_.builder().SetInsertPoint(ad_matmul_block);

        llvm::Value* row_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "matmul_ad_i");
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), row_idx);
        llvm::BasicBlock* row_cond = llvm::BasicBlock::Create(ctx_.context(), "matmul_ad_i_cond", current_func);
        llvm::BasicBlock* row_body = llvm::BasicBlock::Create(ctx_.context(), "matmul_ad_i_body", current_func);
        llvm::BasicBlock* row_exit = llvm::BasicBlock::Create(ctx_.context(), "matmul_ad_i_exit", current_func);
        ctx_.builder().CreateBr(row_cond);

        ctx_.builder().SetInsertPoint(row_cond);
        llvm::Value* row = ctx_.builder().CreateLoad(ctx_.int64Type(), row_idx);
        llvm::Value* row_has_work = ctx_.builder().CreateICmpULT(row, a_rows);
        ctx_.builder().CreateCondBr(row_has_work, row_body, row_exit);

        ctx_.builder().SetInsertPoint(row_body);
        llvm::Value* col_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "matmul_ad_j");
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), col_idx);
        llvm::BasicBlock* col_cond = llvm::BasicBlock::Create(ctx_.context(), "matmul_ad_j_cond", current_func);
        llvm::BasicBlock* col_body = llvm::BasicBlock::Create(ctx_.context(), "matmul_ad_j_body", current_func);
        llvm::BasicBlock* col_exit = llvm::BasicBlock::Create(ctx_.context(), "matmul_ad_j_exit", current_func);
        ctx_.builder().CreateBr(col_cond);

        ctx_.builder().SetInsertPoint(col_cond);
        llvm::Value* col = ctx_.builder().CreateLoad(ctx_.int64Type(), col_idx);
        llvm::Value* col_has_work = ctx_.builder().CreateICmpULT(col, b_cols);
        ctx_.builder().CreateCondBr(col_has_work, col_body, col_exit);

        ctx_.builder().SetInsertPoint(col_body);
        llvm::Value* acc = ctx_.builder().CreateAlloca(ctx_.ptrType(), nullptr, "matmul_ad_acc");
        llvm::Value* zero_node = autodiff_->createADConstant(llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        ctx_.builder().CreateStore(zero_node, acc);
        llvm::Value* k_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "matmul_ad_k");
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_idx);
        llvm::BasicBlock* k_cond = llvm::BasicBlock::Create(ctx_.context(), "matmul_ad_k_cond", current_func);
        llvm::BasicBlock* k_body = llvm::BasicBlock::Create(ctx_.context(), "matmul_ad_k_body", current_func);
        llvm::BasicBlock* k_exit = llvm::BasicBlock::Create(ctx_.context(), "matmul_ad_k_exit", current_func);
        ctx_.builder().CreateBr(k_cond);

        ctx_.builder().SetInsertPoint(k_cond);
        llvm::Value* kk = ctx_.builder().CreateLoad(ctx_.int64Type(), k_idx);
        llvm::Value* k_has_work = ctx_.builder().CreateICmpULT(kk, a_cols);
        ctx_.builder().CreateCondBr(k_has_work, k_body, k_exit);

        ctx_.builder().SetInsertPoint(k_body);
        llvm::Value* a_row_offset = ctx_.builder().CreateMul(row, a_cols);
        llvm::Value* a_index = ctx_.builder().CreateAdd(a_row_offset, kk);
        llvm::Value* b_row_offset = ctx_.builder().CreateMul(kk, b_cols);
        llvm::Value* b_index = ctx_.builder().CreateAdd(b_row_offset, col);
        llvm::Value* a_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_a_elements_ptr, a_index);
        llvm::Value* b_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_b_elements_ptr, b_index);
        llvm::Value* a_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), a_elem_ptr);
        llvm::Value* b_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), b_elem_ptr);
        llvm::Value* a_node = adNodeFromTensorElementBits(a_bits, "matmul_ad_a");
        llvm::Value* b_node = adNodeFromTensorElementBits(b_bits, "matmul_ad_b");
        llvm::Value* product_node = autodiff_->recordADNodeBinary(4, a_node, b_node);
        llvm::Value* old_acc = ctx_.builder().CreateLoad(ctx_.ptrType(), acc);
        llvm::Value* new_acc = autodiff_->recordADNodeBinary(2, old_acc, product_node);
        ctx_.builder().CreateStore(new_acc, acc);
        llvm::Value* next_k = ctx_.builder().CreateAdd(kk, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        ctx_.builder().CreateStore(next_k, k_idx);
        ctx_.builder().CreateBr(k_cond);

        ctx_.builder().SetInsertPoint(k_exit);
        llvm::Value* final_node = ctx_.builder().CreateLoad(ctx_.ptrType(), acc);
        llvm::Value* final_node_bits = ctx_.builder().CreatePtrToInt(final_node, ctx_.int64Type());
        llvm::Value* c_row_offset = ctx_.builder().CreateMul(row, b_cols);
        llvm::Value* c_index = ctx_.builder().CreateAdd(c_row_offset, col);
        llvm::Value* c_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), c_elements_ptr, c_index);
        ctx_.builder().CreateStore(final_node_bits, c_elem_ptr);
        llvm::Value* next_col = ctx_.builder().CreateAdd(col, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        ctx_.builder().CreateStore(next_col, col_idx);
        ctx_.builder().CreateBr(col_cond);

        ctx_.builder().SetInsertPoint(col_exit);
        llvm::Value* next_row = ctx_.builder().CreateAdd(row, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        ctx_.builder().CreateStore(next_row, row_idx);
        ctx_.builder().CreateBr(row_cond);

        ctx_.builder().SetInsertPoint(row_exit);
        ctx_.builder().CreateBr(matmul_data_ready);

        ctx_.builder().SetInsertPoint(numeric_matmul_block);
    }

    // Track XLA path state for PHI node construction
    llvm::Value* xla_packed_result = nullptr;
    llvm::BasicBlock* xla_exit_block = nullptr;

#ifdef ESHKOL_XLA_ENABLED
    // ===== XLA DISPATCH FOR MASSIVE TENSORS =====
    if (xla_ && xla_->isAvailable()) {
        llvm::Value* mk = ctx_.builder().CreateMul(a_rows, a_cols);
        llvm::Value* total_ops = ctx_.builder().CreateMul(mk, b_cols);
        llvm::Value* threshold = llvm::ConstantInt::get(ctx_.int64Type(), xla::xla_get_threshold());
        llvm::Value* use_xla = ctx_.builder().CreateICmpUGE(total_ops, threshold);

        llvm::BasicBlock* xla_block = llvm::BasicBlock::Create(ctx_.context(), "dot_xla", current_func);
        llvm::BasicBlock* simd_block = llvm::BasicBlock::Create(ctx_.context(), "dot_simd_fallback", current_func);
        ctx_.builder().CreateCondBr(use_xla, xla_block, simd_block);

        ctx_.builder().SetInsertPoint(xla_block);
        llvm::Value* xla_result = xla_->emitMatmul(tensor_a_ptr, tensor_b_ptr);
        if (xla_result) {
            xla_packed_result = tagged_.packHeapPtr(xla_result);
            xla_exit_block = ctx_.builder().GetInsertBlock();
            ctx_.builder().CreateBr(final_merge);
        } else {
            ctx_.builder().CreateBr(simd_block);
        }
        ctx_.builder().SetInsertPoint(simd_block);
    }
#endif

    // === MATMUL VIA BLAS RUNTIME (replaces inline triple-nested loop) ===
    // Call eshkol_matmul_f64(A_elems, B_elems, C_elems, M, K, N)
    // Tensor elements are int64 bitpatterns of doubles — same bytes, just reinterpret pointer
    auto* matmul_ft = llvm::FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(),
         ctx_.int64Type(), ctx_.int64Type(), ctx_.int64Type()}, false);
    llvm::Function* matmul_fn = ctx_.module().getFunction("eshkol_matmul_f64");
    if (!matmul_fn) {
        matmul_fn = llvm::Function::Create(matmul_ft,
            llvm::Function::ExternalLinkage, "eshkol_matmul_f64", &ctx_.module());
    }
    ctx_.builder().CreateCall(matmul_fn,
        {typed_a_elements_ptr, typed_b_elements_ptr, c_elements_ptr,
         a_rows, a_cols, b_cols});
    if (matmul_data_ready) {
        ctx_.builder().CreateBr(matmul_data_ready);
        ctx_.builder().SetInsertPoint(matmul_data_ready);
    }

    // === RESULT SHAPE CONTRACTION (PEP 465) ===
    // Determine result ndim and shape:
    //   both 2D → [M, N], ndim=2
    //   A was 1D (promoted) → [N], ndim=1
    //   B was 1D (promoted) → [M], ndim=1
    llvm::Value* both_2d = ctx_.builder().CreateAnd(
        ctx_.builder().CreateICmpEQ(a_num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2)),
        ctx_.builder().CreateICmpEQ(b_num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2)));

    llvm::BasicBlock* result_2d_bb = llvm::BasicBlock::Create(ctx_.context(), "mm_res_2d", current_func);
    llvm::BasicBlock* result_1d_bb = llvm::BasicBlock::Create(ctx_.context(), "mm_res_1d", current_func);
    llvm::BasicBlock* result_merge_bb = llvm::BasicBlock::Create(ctx_.context(), "mm_res_merge", current_func);
    ctx_.builder().CreateCondBr(both_2d, result_2d_bb, result_1d_bb);

    // 2D result: dims = [M, N], ndim = 2
    ctx_.builder().SetInsertPoint(result_2d_bb);
    llvm::Value* c_dims_2d = ctx_.builder().CreateCall(arena_alloc,
        {dot_arena_ptr, llvm::ConstantInt::get(ctx_.sizeType(), 16)}, "dot_dims_2d");
    ctx_.builder().CreateStore(a_rows,
        ctx_.builder().CreateGEP(ctx_.int64Type(), c_dims_2d, llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    ctx_.builder().CreateStore(b_cols,
        ctx_.builder().CreateGEP(ctx_.int64Type(), c_dims_2d, llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    llvm::Value* c_ndim_2d = llvm::ConstantInt::get(ctx_.int64Type(), 2);
    ctx_.builder().CreateBr(result_merge_bb);

    // 1D result: if A was 1D → dim = [N]; if B was 1D → dim = [M]
    ctx_.builder().SetInsertPoint(result_1d_bb);
    llvm::Value* c_dims_1d = ctx_.builder().CreateCall(arena_alloc,
        {dot_arena_ptr, llvm::ConstantInt::get(ctx_.sizeType(), 8)}, "dot_dims_1d");
    llvm::Value* contracted_dim = ctx_.builder().CreateSelect(a_is_1d, b_cols, a_rows);
    ctx_.builder().CreateStore(contracted_dim,
        ctx_.builder().CreateGEP(ctx_.int64Type(), c_dims_1d, llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    llvm::Value* c_ndim_1d = llvm::ConstantInt::get(ctx_.int64Type(), 1);
    ctx_.builder().CreateBr(result_merge_bb);

    // Merge result shape
    ctx_.builder().SetInsertPoint(result_merge_bb);
    llvm::PHINode* c_dims_ptr = ctx_.builder().CreatePHI(ctx_.ptrType(), 2, "c_dims");
    c_dims_ptr->addIncoming(c_dims_2d, result_2d_bb);
    c_dims_ptr->addIncoming(c_dims_1d, result_1d_bb);
    llvm::PHINode* c_ndim = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "c_ndim");
    c_ndim->addIncoming(c_ndim_2d, result_2d_bb);
    c_ndim->addIncoming(c_ndim_1d, result_1d_bb);

    // Store tensor struct fields
    ctx_.builder().CreateStore(c_dims_ptr,
        ctx_.builder().CreateStructGEP(tensor_type, c_tensor_ptr, 0));
    ctx_.builder().CreateStore(c_ndim,
        ctx_.builder().CreateStructGEP(tensor_type, c_tensor_ptr, 1));
    ctx_.builder().CreateStore(c_elements_ptr,
        ctx_.builder().CreateStructGEP(tensor_type, c_tensor_ptr, 2));
    ctx_.builder().CreateStore(c_total,
        ctx_.builder().CreateStructGEP(tensor_type, c_tensor_ptr, 3));

    llvm::Value* tensor_result_2d = tagged_.packHeapPtr(c_tensor_ptr);
    ctx_.builder().CreateBr(final_merge);
    llvm::BasicBlock* dot_2d_exit = ctx_.builder().GetInsertBlock();

    // Create scalar_merge block for 1D scalar results (from both Scheme vec and 1D tensor)
    llvm::BasicBlock* scalar_merge = llvm::BasicBlock::Create(ctx_.context(), "scalar_merge", current_func);

    // Now add the deferred branch from svec_exit_block to scalar_merge.
    // svec_result is already a tagged value (DUAL_NUMBER when the dot was over a
    // Scheme vector of forward-mode duals, DOUBLE otherwise), so no repack.
    ctx_.builder().SetInsertPoint(svec_exit_block);
    llvm::Value* svec_packed = svec_result;
    ctx_.builder().CreateBr(scalar_merge);

    // Tensor merge for 1D scalar path - pack and branch to scalar_merge
    ctx_.builder().SetInsertPoint(tensor_merge);
    llvm::PHINode* scalar_result_1d = ctx_.builder().CreatePHI(ctx_.taggedValueType(),
        ad_dot_exit_block ? 2 : 1, "dot1d_scalar_result");
    scalar_result_1d->addIncoming(dot_tagged_result_1d, dot_numeric_1d_exit);
    if (ad_dot_exit_block && ad_dot_tagged_result) {
        scalar_result_1d->addIncoming(ad_dot_tagged_result, ad_dot_exit_block);
    }
    ctx_.builder().CreateBr(scalar_merge);
    llvm::BasicBlock* tensor_1d_exit = ctx_.builder().GetInsertBlock();

    // === SCALAR MERGE (for 1D results from both Scheme vec and tensor paths) ===
    ctx_.builder().SetInsertPoint(scalar_merge);
    llvm::PHINode* scalar_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "scalar_result");
    scalar_phi->addIncoming(svec_packed, svec_exit_block);
    scalar_phi->addIncoming(scalar_result_1d, tensor_1d_exit);
    ctx_.builder().CreateBr(final_merge);
    llvm::BasicBlock* scalar_merge_exit = ctx_.builder().GetInsertBlock();

    // === FINAL MERGE ===
    ctx_.builder().SetInsertPoint(final_merge);
    unsigned phi_count = xla_exit_block ? 3 : 2;
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), phi_count, "dot_result");
    result_phi->addIncoming(scalar_phi, scalar_merge_exit);
    result_phi->addIncoming(tensor_result_2d, dot_2d_exit);
    if (xla_exit_block) {
        result_phi->addIncoming(xla_packed_result, xla_exit_block);
    }

    return result_phi;
}

llvm::Value* TensorCodegen::tensorApply(const eshkol_operations_t* op) {
    // tensor-apply: (tensor-apply tensor function)
    // Applies a function to each element of a tensor, returning a new tensor
    if (op->call_op.num_vars != 2) {
        eshkol_error("tensor-apply requires exactly 2 arguments: tensor and function");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // Get function to apply — supports named arithmetic/math functions
    eshkol_ast_t* func_ast = &op->call_op.variables[1];
    if (func_ast->type != ESHKOL_VAR) {
        eshkol_error("tensor-apply: function argument must be a named function (e.g., sin, cos, +)");
        return nullptr;
    }

    std::string func_name = func_ast->variable.id;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "tensor-apply");

    // Create result tensor with same dimensions using arena
    llvm::Value* apply_arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Allocate tensor struct with header
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* typed_result_tensor_ptr = ctx_.builder().CreateCall(alloc_tensor_func, {apply_arena_ptr}, "apply_tensor");

    // Copy tensor structure (dimensions, num_dimensions, total_elements)
    llvm::Value* dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), dims_field_ptr);
    llvm::Value* result_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 0);
    ctx_.builder().CreateStore(dims_ptr, result_dims_field_ptr);

    llvm::Value* num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = ctx_.builder().CreateLoad(ctx_.int64Type(), num_dims_field_ptr);
    llvm::Value* result_num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 1);
    ctx_.builder().CreateStore(num_dims, result_num_dims_field_ptr);

    llvm::Value* total_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = ctx_.builder().CreateLoad(ctx_.int64Type(), total_elements_field_ptr);
    llvm::Value* result_total_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 3);
    ctx_.builder().CreateStore(total_elements, result_total_elements_field_ptr);
    llvm::Value* dtype_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 4);
    llvm::Value* dtype = ctx_.builder().CreateLoad(ctx_.int64Type(), dtype_field_ptr);
    llvm::Value* result_dtype_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 4);
    ctx_.builder().CreateStore(dtype, result_dtype_field_ptr);

    // Allocate result elements array using arena
    llvm::Value* elements_size = ctx_.builder().CreateMul(total_elements,
                                            llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_elements_ptr = ctx_.builder().CreateCall(arena_alloc, {apply_arena_ptr, elements_size}, "apply_elems");
    llvm::Value* typed_result_elements_ptr = ctx_.builder().CreatePointerCast(result_elements_ptr, ctx_.ptrType());

    llvm::Value* result_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 2);
    ctx_.builder().CreateStore(typed_result_elements_ptr, result_elements_field_ptr);

    // Get source elements
    llvm::Value* src_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_elements_field_ptr);
    llvm::Value* typed_src_elements_ptr = ctx_.builder().CreatePointerCast(src_elements_ptr, ctx_.ptrType());

    // Apply function to each element (FULL implementation with loops)
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_condition = llvm::BasicBlock::Create(ctx_.context(), "apply_loop_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "apply_loop_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "apply_loop_exit", current_func);

    // Initialize loop counter
    llvm::Value* loop_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "loop_counter");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), loop_counter);

    // Jump to loop condition
    ctx_.builder().CreateBr(loop_condition);

    // Loop condition: check if counter < total_elements
    ctx_.builder().SetInsertPoint(loop_condition);
    llvm::Value* current_index = ctx_.builder().CreateLoad(ctx_.int64Type(), loop_counter);
    llvm::Value* loop_cmp = ctx_.builder().CreateICmpULT(current_index, total_elements);
    ctx_.builder().CreateCondBr(loop_cmp, loop_body, loop_exit);

    // Loop body: apply function to current element
    ctx_.builder().SetInsertPoint(loop_body);

    // Load source element at current index
    llvm::Value* src_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_src_elements_ptr, current_index);
    llvm::Value* src_elem = ctx_.builder().CreateLoad(ctx_.int64Type(), src_elem_ptr);

    // Apply function based on function name
    llvm::Value* src_double = ctx_.builder().CreateBitCast(src_elem, ctx_.doubleType());
    llvm::Value* result_double = nullptr;
    if (func_name == "double") {
        result_double = ctx_.builder().CreateFMul(
            src_double, llvm::ConstantFP::get(ctx_.doubleType(), 2.0));
    } else if (func_name == "square") {
        result_double = ctx_.builder().CreateFMul(src_double, src_double);
    } else if (func_name == "increment") {
        result_double = ctx_.builder().CreateFAdd(
            src_double, llvm::ConstantFP::get(ctx_.doubleType(), 1.0));
    } else if (func_name == "negate") {
        result_double = ctx_.builder().CreateFNeg(src_double);
    } else if (func_name == "abs") {
        llvm::Value* is_negative = ctx_.builder().CreateFCmpOLT(
            src_double, llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        result_double = ctx_.builder().CreateSelect(
            is_negative, ctx_.builder().CreateFNeg(src_double), src_double);
    } else if (func_name == "identity") {
        result_double = src_double;
    } else {
        eshkol_warn("Unknown function in tensor-apply: %s, using identity", func_name.c_str());
        result_double = src_double;
    }
    llvm::Value* result_elem = ctx_.builder().CreateBitCast(result_double, ctx_.int64Type());

    // Store result element at current index
    llvm::Value* result_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_result_elements_ptr, current_index);
    ctx_.builder().CreateStore(result_elem, result_elem_ptr);

    // Increment loop counter
    llvm::Value* next_index = ctx_.builder().CreateAdd(current_index, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_index, loop_counter);

    // Jump back to condition check
    ctx_.builder().CreateBr(loop_condition);

    // Loop exit: continue with rest of function
    ctx_.builder().SetInsertPoint(loop_exit);

    return tagged_.packHeapPtr(typed_result_tensor_ptr);
}

llvm::Value* TensorCodegen::tensorReduceAll(const eshkol_operations_t* op) {
    // tensor-reduce-all: (tensor-reduce-all tensor function initial-value)
    // Reduces entire tensor to a single value by applying a binary function
    if (op->call_op.num_vars != 3) {
        eshkol_error("tensor-reduce requires exactly 3 arguments: tensor, function, and initial value");
        return nullptr;
    }

    llvm::Value* src_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* initial_tagged = codegenAST(&op->call_op.variables[2]);
    if (!src_val || !initial_tagged) return nullptr;

    // Get function to apply
    eshkol_ast_t* func_ast = &op->call_op.variables[1];
    if (func_ast->type != ESHKOL_VAR) {
        eshkol_error("tensor-reduce currently only supports simple function names");
        return nullptr;
    }
    std::string func_name = func_ast->variable.id;

    // Check type: Scheme vector vs Tensor (using consolidated type check)
    llvm::Value* is_scheme_vector = tagged_.isVector(src_val);

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* scheme_vec_block = llvm::BasicBlock::Create(ctx_.context(), "reduce_scheme_vec", current_func);
    llvm::BasicBlock* tensor_block = llvm::BasicBlock::Create(ctx_.context(), "reduce_tensor", current_func);
    llvm::BasicBlock* reduce_merge = llvm::BasicBlock::Create(ctx_.context(), "reduce_merge", current_func);

    ctx_.builder().CreateCondBr(is_scheme_vector, scheme_vec_block, tensor_block);

    // === SCHEME VECTOR PATH ===
    ctx_.builder().SetInsertPoint(scheme_vec_block);
    llvm::Value* svec_ptr_int = tagged_.unpackInt64(src_val);
    llvm::Value* svec_ptr = ctx_.builder().CreateIntToPtr(svec_ptr_int, ctx_.ptrType());

    // Scheme vector: [length:i64, element0:tagged_value, element1:tagged_value, ...]
    llvm::Value* svec_len = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_ptr);

    llvm::BasicBlock* svec_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "svec_reduce_cond", current_func);
    llvm::BasicBlock* svec_loop_body = llvm::BasicBlock::Create(ctx_.context(), "svec_reduce_body", current_func);
    llvm::BasicBlock* svec_loop_exit = llvm::BasicBlock::Create(ctx_.context(), "svec_reduce_exit", current_func);

    // Initialize with initial_value as double
    llvm::Value* svec_initial = tagged_.unpackDouble(initial_tagged);
    llvm::Value* svec_acc = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "svec_acc");
    llvm::Value* svec_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "svec_i");
    ctx_.builder().CreateStore(svec_initial, svec_acc);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), svec_counter);
    ctx_.builder().CreateBr(svec_loop_cond);

    ctx_.builder().SetInsertPoint(svec_loop_cond);
    llvm::Value* svec_i = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_counter);
    llvm::Value* svec_cmp = ctx_.builder().CreateICmpULT(svec_i, svec_len);
    ctx_.builder().CreateCondBr(svec_cmp, svec_loop_body, svec_loop_exit);

    ctx_.builder().SetInsertPoint(svec_loop_body);
    // Elements start after 8-byte length field
    llvm::Value* svec_elems_base = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* svec_elems_typed = ctx_.builder().CreatePointerCast(svec_elems_base, ctx_.ptrType());
    llvm::Value* svec_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_elems_typed, svec_i);
    llvm::Value* svec_elem_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_elem_ptr);
    llvm::Value* svec_elem_val = extractAsDouble(svec_elem_tagged);

    llvm::Value* svec_current_acc = ctx_.builder().CreateLoad(ctx_.doubleType(), svec_acc);
    llvm::Value* svec_new_acc = nullptr;

    if (func_name == "+") {
        svec_new_acc = ctx_.builder().CreateFAdd(svec_current_acc, svec_elem_val);
    } else if (func_name == "*") {
        svec_new_acc = ctx_.builder().CreateFMul(svec_current_acc, svec_elem_val);
    } else if (func_name == "max") {
        llvm::Value* cmp = ctx_.builder().CreateFCmpOGT(svec_current_acc, svec_elem_val);
        svec_new_acc = ctx_.builder().CreateSelect(cmp, svec_current_acc, svec_elem_val);
    } else if (func_name == "min") {
        llvm::Value* cmp = ctx_.builder().CreateFCmpOLT(svec_current_acc, svec_elem_val);
        svec_new_acc = ctx_.builder().CreateSelect(cmp, svec_current_acc, svec_elem_val);
    } else {
        svec_new_acc = ctx_.builder().CreateFAdd(svec_current_acc, svec_elem_val);
    }

    ctx_.builder().CreateStore(svec_new_acc, svec_acc);
    llvm::Value* svec_next_i = ctx_.builder().CreateAdd(svec_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(svec_next_i, svec_counter);
    ctx_.builder().CreateBr(svec_loop_cond);

    ctx_.builder().SetInsertPoint(svec_loop_exit);
    llvm::Value* svec_result = ctx_.builder().CreateLoad(ctx_.doubleType(), svec_acc);
    llvm::Value* svec_result_tagged = tagged_.packDouble(svec_result);
    ctx_.builder().CreateBr(reduce_merge);
    llvm::BasicBlock* svec_exit_block = ctx_.builder().GetInsertBlock();

    // === TENSOR PATH ===
    ctx_.builder().SetInsertPoint(tensor_block);
    llvm::Value* tensor_ptr = unpackTensorOperandChecked(src_val, "tensor-reduce-all");

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), elements_field_ptr);
    llvm::Value* typed_elements_ptr = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.ptrType());

    llvm::Value* total_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = ctx_.builder().CreateLoad(ctx_.int64Type(), total_elements_field_ptr);

    llvm::BasicBlock* tensor_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "tensor_reduce_cond", current_func);
    llvm::BasicBlock* tensor_loop_body = llvm::BasicBlock::Create(ctx_.context(), "tensor_reduce_body", current_func);
    llvm::BasicBlock* tensor_loop_exit = llvm::BasicBlock::Create(ctx_.context(), "tensor_reduce_exit", current_func);

    llvm::Value* tensor_initial = tagged_.unpackDouble(initial_tagged);
    llvm::Value* tensor_acc = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "tensor_acc");
    llvm::Value* tensor_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "tensor_i");
    ctx_.builder().CreateStore(tensor_initial, tensor_acc);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), tensor_counter);

#ifdef ESHKOL_XLA_ENABLED
    // ===== XLA DISPATCH FOR LARGE TENSOR REDUCE-ALL =====
    // Dispatch hierarchy: XLA (≥100K elements) → scalar loop
    if (xla_ && xla_->isAvailable()) {
        // Map function name to XLA ReduceOp
        xla::ReduceOp xla_reduce_op;
        bool has_xla_reduce = true;
        if (func_name == "+") {
            xla_reduce_op = xla::ReduceOp::SUM;
        } else if (func_name == "*") {
            xla_reduce_op = xla::ReduceOp::PROD;
        } else if (func_name == "max") {
            xla_reduce_op = xla::ReduceOp::MAX;
        } else if (func_name == "min") {
            xla_reduce_op = xla::ReduceOp::MIN;
        } else {
            has_xla_reduce = false;
        }

        if (has_xla_reduce) {
            llvm::Value* threshold = llvm::ConstantInt::get(ctx_.int64Type(), xla::xla_get_threshold());
            llvm::Value* use_xla = ctx_.builder().CreateICmpUGE(total_elements, threshold);

            llvm::BasicBlock* xla_reduce_block = llvm::BasicBlock::Create(ctx_.context(), "treduce_xla", current_func);
            llvm::BasicBlock* scalar_reduce_block = llvm::BasicBlock::Create(ctx_.context(), "treduce_scalar_fallback", current_func);

            ctx_.builder().CreateCondBr(use_xla, xla_reduce_block, scalar_reduce_block);

            // XLA path: emit reduce (axis=-1 for reduce all)
            ctx_.builder().SetInsertPoint(xla_reduce_block);
            llvm::Value* xla_result = xla_->emitReduce(tensor_ptr, -1, xla_reduce_op);
            if (xla_result) {
                // Extract scalar from 1-element result tensor
                llvm::Value* xla_elems_ptr = ctx_.builder().CreateStructGEP(tensor_type, xla_result,
                    TypeSystem::TENSOR_ELEMENTS_IDX, "xla_reduce_elems_ptr");
                llvm::Value* xla_elems = ctx_.builder().CreateLoad(ctx_.ptrType(), xla_elems_ptr, "xla_reduce_elems");
                llvm::Value* xla_int_val = ctx_.builder().CreateLoad(ctx_.int64Type(), xla_elems, "xla_reduce_int");
                llvm::Value* xla_reduce_val = ctx_.builder().CreateBitCast(xla_int_val, ctx_.doubleType(), "xla_reduce_dbl");
                ctx_.builder().CreateStore(xla_reduce_val, tensor_acc);
                ctx_.builder().CreateBr(tensor_loop_exit);
            } else {
                // XLA returned nullptr, fall back to scalar loop
                ctx_.builder().CreateBr(scalar_reduce_block);
            }

            // Scalar fallback
            ctx_.builder().SetInsertPoint(scalar_reduce_block);
        }
    }
#endif

    ctx_.builder().CreateBr(tensor_loop_cond);

    ctx_.builder().SetInsertPoint(tensor_loop_cond);
    llvm::Value* tensor_i = ctx_.builder().CreateLoad(ctx_.int64Type(), tensor_counter);
    llvm::Value* tensor_cmp = ctx_.builder().CreateICmpULT(tensor_i, total_elements);
    ctx_.builder().CreateCondBr(tensor_cmp, tensor_loop_body, tensor_loop_exit);

    ctx_.builder().SetInsertPoint(tensor_loop_body);
    llvm::Value* tensor_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements_ptr, tensor_i);
    llvm::Value* tensor_elem_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), tensor_elem_ptr);
    llvm::Value* tensor_elem_val = ctx_.builder().CreateBitCast(tensor_elem_bits, ctx_.doubleType());

    llvm::Value* tensor_current_acc = ctx_.builder().CreateLoad(ctx_.doubleType(), tensor_acc);
    llvm::Value* tensor_new_acc = nullptr;

    if (func_name == "+") {
        tensor_new_acc = ctx_.builder().CreateFAdd(tensor_current_acc, tensor_elem_val);
    } else if (func_name == "*") {
        tensor_new_acc = ctx_.builder().CreateFMul(tensor_current_acc, tensor_elem_val);
    } else if (func_name == "max") {
        llvm::Value* cmp = ctx_.builder().CreateFCmpOGT(tensor_current_acc, tensor_elem_val);
        tensor_new_acc = ctx_.builder().CreateSelect(cmp, tensor_current_acc, tensor_elem_val);
    } else if (func_name == "min") {
        llvm::Value* cmp = ctx_.builder().CreateFCmpOLT(tensor_current_acc, tensor_elem_val);
        tensor_new_acc = ctx_.builder().CreateSelect(cmp, tensor_current_acc, tensor_elem_val);
    } else {
        tensor_new_acc = ctx_.builder().CreateFAdd(tensor_current_acc, tensor_elem_val);
    }

    ctx_.builder().CreateStore(tensor_new_acc, tensor_acc);
    llvm::Value* tensor_next_i = ctx_.builder().CreateAdd(tensor_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(tensor_next_i, tensor_counter);
    ctx_.builder().CreateBr(tensor_loop_cond);

    ctx_.builder().SetInsertPoint(tensor_loop_exit);
    llvm::Value* tensor_result = ctx_.builder().CreateLoad(ctx_.doubleType(), tensor_acc);
    llvm::Value* tensor_result_tagged = tagged_.packDouble(tensor_result);
    ctx_.builder().CreateBr(reduce_merge);
    llvm::BasicBlock* tensor_exit_block = ctx_.builder().GetInsertBlock();

    // === MERGE ===
    ctx_.builder().SetInsertPoint(reduce_merge);
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "reduce_result");
    result_phi->addIncoming(svec_result_tagged, svec_exit_block);
    result_phi->addIncoming(tensor_result_tagged, tensor_exit_block);

    return result_phi;
}

llvm::Value* TensorCodegen::tensorReduceWithDim(const eshkol_operations_t* op) {
    // tensor-reduce: (tensor-reduce tensor function initial-value dimension)
    // Reduces tensor along specified dimension, returning tensor with reduced dimensionality
    // Supports N-D tensors of any rank via eshkol_xla_reduce runtime (with GPU dispatch)
    if (op->call_op.num_vars != 4) {
        eshkol_error("tensor-reduce requires exactly 4 arguments: tensor, function, initial-value, dimension");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    // Evaluate initial_value for side effects but we don't pass it to the runtime
    // (the runtime knows the correct identity for each op)
    codegenAST(&op->call_op.variables[2]);
    llvm::Value* dimension_value = codegenAST(&op->call_op.variables[3]);
    if (!tensor_val || !dimension_value) return nullptr;

    // Get function name to determine op code
    eshkol_ast_t* func_ast = &op->call_op.variables[1];
    if (func_ast->type != ESHKOL_VAR) {
        eshkol_error("tensor-reduce currently only supports simple function names");
        return nullptr;
    }
    std::string func_name = func_ast->variable.id;

    // Map function name to XLA reduce op code: SUM=0, MEAN=1, MAX=2, MIN=3, PROD=4
    int64_t op_code;
    if (func_name == "+") op_code = 0;
    else if (func_name == "mean") op_code = 1;
    else if (func_name == "max") op_code = 2;
    else if (func_name == "min") op_code = 3;
    else if (func_name == "*") op_code = 4;
    else {
        eshkol_warn("Unknown reduction function: %s, defaulting to sum", func_name.c_str());
        op_code = 0;
    }

    // Extract tensor pointer (type-checked: ESH-0069)
    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "tensor-reduce");

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Extract tensor fields
    llvm::Value* src_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_elements_field_ptr);

    llvm::Value* src_total_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* src_total = ctx_.builder().CreateLoad(ctx_.int64Type(), src_total_field_ptr);

    llvm::Value* src_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* src_dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_dims_field_ptr);

    llvm::Value* src_num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* src_num_dims = ctx_.builder().CreateLoad(ctx_.int64Type(), src_num_dims_field_ptr);

    // Handle negative axis: if axis < 0, axis += rank
    llvm::Value* axis_val = extractStructuralIntOrRaise(
        ctx_, tagged_, dimension_value, "tensor-reduce", "integer axis");
    llvm::Value* is_negative = ctx_.builder().CreateICmpSLT(axis_val, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* adjusted_axis = ctx_.builder().CreateAdd(axis_val, src_num_dims);
    axis_val = ctx_.builder().CreateSelect(is_negative, adjusted_axis, axis_val);

    // Load arena
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());

    // Declare eshkol_xla_reduce runtime function
    auto* ptrTy = ctx_.ptrType();
    auto* i64Ty = ctx_.int64Type();
    llvm::FunctionType* reduce_fn_type = llvm::FunctionType::get(ptrTy,
        {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, i64Ty}, false);
    llvm::FunctionCallee reduce_callee = ctx_.module().getOrInsertFunction("eshkol_xla_reduce", reduce_fn_type);

    // Call eshkol_xla_reduce(arena, data, total, shape, rank, axis, op_code)
    // This handles N-D tensors of any rank with GPU dispatch for large tensors
    llvm::Value* op_code_val = llvm::ConstantInt::get(i64Ty, op_code);
    llvm::Value* result = ctx_.builder().CreateCall(reduce_callee,
        {arena_ptr, src_elements_ptr, src_total, src_dims_ptr, src_num_dims, axis_val, op_code_val},
        "reduce_dim_result");

    return tagged_.packHeapPtr(result);
}

llvm::Value* TensorCodegen::emitAxisReduce(llvm::Value* tensor_val, llvm::Value* axis_val, int64_t op_code) {
    // Emit a call to eshkol_xla_reduce runtime for axis-specific reduction.
    // Returns a tagged tensor pointer (reduced along the given axis).
    auto& builder = ctx_.builder();

    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "tensor-reduce-axis");
    llvm::StructType* tensor_type = ctx_.tensorType();

    // Extract tensor fields
    llvm::Value* src_elements_ptr = builder.CreateLoad(ctx_.ptrType(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 2));
    llvm::Value* src_total = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 3));
    llvm::Value* src_dims_ptr = builder.CreateLoad(ctx_.ptrType(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 0));
    llvm::Value* src_num_dims = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 1));

    // Handle negative axis: if axis < 0, axis += rank
    llvm::Value* axis = extractStructuralIntOrRaise(
        ctx_, tagged_, axis_val, "tensor-reduce-axis", "integer axis");
    llvm::Value* is_negative = builder.CreateICmpSLT(axis, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* adjusted = builder.CreateAdd(axis, src_num_dims);
    axis = builder.CreateSelect(is_negative, adjusted, axis);

    if (autodiff_ && op_code >= 0 && op_code <= 3) {
        llvm::Value* in_ad_mode = builder.CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
        llvm::Function* current_func = builder.GetInsertBlock()->getParent();
        llvm::BasicBlock* ad_block = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_ad", current_func);
        llvm::BasicBlock* numeric_block = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_numeric", current_func);
        llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_merge", current_func);
        builder.CreateCondBr(in_ad_mode, ad_block, numeric_block);

        builder.SetInsertPoint(ad_block);
        llvm::Value* arena_ptr = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        llvm::Function* arena_alloc = mem_.getArenaAllocate();
        llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
        llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "axis_reduce_ad_tensor");

        llvm::Value* one_i64 = llvm::ConstantInt::get(ctx_.int64Type(), 1);
        llvm::Value* out_num_dims = builder.CreateSub(src_num_dims, one_i64);
        llvm::Value* dims_bytes = builder.CreateMul(out_num_dims,
            llvm::ConstantInt::get(ctx_.int64Type(), (int64_t)sizeof(int64_t)));
        llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "axis_reduce_ad_dims");

        llvm::Value* axis_len_ptr = builder.CreateGEP(ctx_.int64Type(), src_dims_ptr, axis);
        llvm::Value* axis_len = builder.CreateLoad(ctx_.int64Type(), axis_len_ptr, "axis_reduce_len");
        llvm::Value* out_total = builder.CreateUDiv(src_total, axis_len, "axis_reduce_out_total");
        llvm::Value* elems_bytes = builder.CreateMul(out_total,
            llvm::ConstantInt::get(ctx_.int64Type(), (int64_t)sizeof(int64_t)));
        llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "axis_reduce_ad_elems");

        llvm::Value* dim_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "axis_reduce_dim_i");
        llvm::Value* dim_out = builder.CreateAlloca(ctx_.int64Type(), nullptr, "axis_reduce_dim_out");
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dim_i);
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dim_out);
        llvm::BasicBlock* dim_cond = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_dim_cond", current_func);
        llvm::BasicBlock* dim_body = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_dim_body", current_func);
        llvm::BasicBlock* dim_store = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_dim_store", current_func);
        llvm::BasicBlock* dim_next = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_dim_next", current_func);
        llvm::BasicBlock* dim_done = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_dim_done", current_func);
        builder.CreateBr(dim_cond);

        builder.SetInsertPoint(dim_cond);
        llvm::Value* d = builder.CreateLoad(ctx_.int64Type(), dim_i);
        builder.CreateCondBr(builder.CreateICmpULT(d, src_num_dims), dim_body, dim_done);

        builder.SetInsertPoint(dim_body);
        builder.CreateCondBr(builder.CreateICmpEQ(d, axis), dim_next, dim_store);

        builder.SetInsertPoint(dim_store);
        llvm::Value* src_dim_val = builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), src_dims_ptr, d));
        llvm::Value* out_d = builder.CreateLoad(ctx_.int64Type(), dim_out);
        builder.CreateStore(src_dim_val, builder.CreateGEP(ctx_.int64Type(), result_dims, out_d));
        builder.CreateStore(builder.CreateAdd(out_d, one_i64), dim_out);
        builder.CreateBr(dim_next);

        builder.SetInsertPoint(dim_next);
        llvm::Value* next_d = builder.CreateAdd(d, one_i64);
        builder.CreateStore(next_d, dim_i);
        builder.CreateBr(dim_cond);

        builder.SetInsertPoint(dim_done);

        llvm::Value* inner_stride = builder.CreateAlloca(ctx_.int64Type(), nullptr, "axis_reduce_inner_stride");
        builder.CreateStore(one_i64, inner_stride);
        llvm::Value* stride_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "axis_reduce_stride_i");
        builder.CreateStore(builder.CreateAdd(axis, one_i64), stride_i);
        llvm::BasicBlock* stride_cond = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_stride_cond", current_func);
        llvm::BasicBlock* stride_body = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_stride_body", current_func);
        llvm::BasicBlock* stride_done = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_stride_done", current_func);
        builder.CreateBr(stride_cond);

        builder.SetInsertPoint(stride_cond);
        llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), stride_i);
        builder.CreateCondBr(builder.CreateICmpULT(si, src_num_dims), stride_body, stride_done);

        builder.SetInsertPoint(stride_body);
        llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), src_dims_ptr, si));
        llvm::Value* old_stride = builder.CreateLoad(ctx_.int64Type(), inner_stride);
        builder.CreateStore(builder.CreateMul(old_stride, dim_val), inner_stride);
        builder.CreateStore(builder.CreateAdd(si, one_i64), stride_i);
        builder.CreateBr(stride_cond);

        builder.SetInsertPoint(stride_done);

        llvm::Value* out_i_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, "axis_reduce_out_i");
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), out_i_alloca);
        llvm::BasicBlock* out_cond = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_out_cond", current_func);
        llvm::BasicBlock* out_body = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_out_body", current_func);
        llvm::BasicBlock* out_done = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_out_done", current_func);
        builder.CreateBr(out_cond);

        builder.SetInsertPoint(out_cond);
        llvm::Value* out_i = builder.CreateLoad(ctx_.int64Type(), out_i_alloca);
        builder.CreateCondBr(builder.CreateICmpULT(out_i, out_total), out_body, out_done);

        builder.SetInsertPoint(out_body);
        llvm::Value* stride = builder.CreateLoad(ctx_.int64Type(), inner_stride);
        llvm::Value* inner = builder.CreateURem(out_i, stride);
        llvm::Value* outer = builder.CreateUDiv(out_i, stride);
        llvm::Value* outer_stride = builder.CreateMul(axis_len, stride);
        llvm::Value* base = builder.CreateAdd(builder.CreateMul(outer, outer_stride), inner);

        llvm::Value* first_bits = builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), src_elements_ptr, base));
        llvm::Value* acc = builder.CreateAlloca(ctx_.ptrType(), nullptr, "axis_reduce_ad_acc");
        builder.CreateStore(adNodeFromTensorElementBits(first_bits, "axis_reduce_ad_first"), acc);
        llvm::Value* k_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, "axis_reduce_k");
        builder.CreateStore(one_i64, k_alloca);
        llvm::BasicBlock* k_cond = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_k_cond", current_func);
        llvm::BasicBlock* k_body = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_k_body", current_func);
        llvm::BasicBlock* k_done = llvm::BasicBlock::Create(ctx_.context(), "axis_reduce_k_done", current_func);
        builder.CreateBr(k_cond);

        builder.SetInsertPoint(k_cond);
        llvm::Value* k = builder.CreateLoad(ctx_.int64Type(), k_alloca);
        builder.CreateCondBr(builder.CreateICmpULT(k, axis_len), k_body, k_done);

        builder.SetInsertPoint(k_body);
        llvm::Value* src_index = builder.CreateAdd(base, builder.CreateMul(k, stride));
        llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), src_elements_ptr, src_index));
        llvm::Value* elem_node = adNodeFromTensorElementBits(elem_bits, "axis_reduce_ad_elem");
        llvm::Value* old_acc = builder.CreateLoad(ctx_.ptrType(), acc);
        uint32_t reduce_ad_op = 2;
        if (op_code == 2) {
            reduce_ad_op = 44;
        } else if (op_code == 3) {
            reduce_ad_op = 45;
        }
        llvm::Value* new_acc = autodiff_->recordADNodeBinary(reduce_ad_op, old_acc, elem_node);
        builder.CreateStore(new_acc, acc);
        builder.CreateStore(builder.CreateAdd(k, one_i64), k_alloca);
        builder.CreateBr(k_cond);

        builder.SetInsertPoint(k_done);
        llvm::Value* final_node = builder.CreateLoad(ctx_.ptrType(), acc);
        if (op_code == 1) {
            llvm::Value* axis_len_fp = builder.CreateUIToFP(axis_len, ctx_.doubleType());
            llvm::Value* axis_len_node = autodiff_->createADConstant(axis_len_fp);
            final_node = autodiff_->recordADNodeBinary(5, final_node, axis_len_node);
        }
        llvm::Value* final_bits = builder.CreatePtrToInt(final_node, ctx_.int64Type());
        builder.CreateStore(final_bits, builder.CreateGEP(ctx_.int64Type(), result_elems, out_i));
        builder.CreateStore(builder.CreateAdd(out_i, one_i64), out_i_alloca);
        builder.CreateBr(out_cond);

        builder.SetInsertPoint(out_done);
        builder.CreateStore(result_dims, builder.CreateStructGEP(tensor_type, result_ptr, 0));
        builder.CreateStore(out_num_dims, builder.CreateStructGEP(tensor_type, result_ptr, 1));
        builder.CreateStore(result_elems, builder.CreateStructGEP(tensor_type, result_ptr, 2));
        builder.CreateStore(out_total, builder.CreateStructGEP(tensor_type, result_ptr, 3));
        llvm::Value* ad_packed_result = tagged_.packHeapPtr(result_ptr);
        builder.CreateBr(merge_block);
        llvm::BasicBlock* ad_exit_block = builder.GetInsertBlock();

        builder.SetInsertPoint(numeric_block);

        llvm::Value* numeric_arena_ptr = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        auto* ptrTy = ctx_.ptrType();
        auto* i64Ty = ctx_.int64Type();
        llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, i64Ty}, false);
        llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_reduce", fn_type);
        llvm::Value* numeric_result = builder.CreateCall(callee,
            {numeric_arena_ptr, src_elements_ptr, src_total, src_dims_ptr, src_num_dims,
             axis, llvm::ConstantInt::get(i64Ty, op_code)},
            "axis_reduce_result");
        llvm::Value* numeric_packed_result = tagged_.packHeapPtr(numeric_result);
        builder.CreateBr(merge_block);
        llvm::BasicBlock* numeric_exit_block = builder.GetInsertBlock();

        builder.SetInsertPoint(merge_block);
        llvm::PHINode* result_phi = builder.CreatePHI(ctx_.taggedValueType(), 2, "axis_reduce_result_phi");
        result_phi->addIncoming(ad_packed_result, ad_exit_block);
        result_phi->addIncoming(numeric_packed_result, numeric_exit_block);
        return result_phi;
    }

    // Load arena and declare runtime
    llvm::Value* arena_ptr = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
    auto* ptrTy = ctx_.ptrType();
    auto* i64Ty = ctx_.int64Type();
    llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
        {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, i64Ty}, false);
    llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_reduce", fn_type);

    llvm::Value* result = builder.CreateCall(callee,
        {arena_ptr, src_elements_ptr, src_total, src_dims_ptr, src_num_dims,
         axis, llvm::ConstantInt::get(i64Ty, op_code)},
        "axis_reduce_result");

    return tagged_.packHeapPtr(result);
}

llvm::Value* TensorCodegen::tensorSum(const eshkol_operations_t* op) {
    // tensor-sum: (tensor-sum tensor) or (tensor-sum tensor axis) - Sum elements
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("tensor-sum requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* src_val = codegenAST(&op->call_op.variables[0]);
    if (!src_val) return nullptr;

    // 2-arg case: (tensor-sum tensor axis) → reduce along axis, returns tensor
    if (op->call_op.num_vars == 2) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[1]);
        if (!axis_val) return nullptr;
        return emitAxisReduce(src_val, axis_val, 0); // SUM=0
    }

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Check type: Scheme vector vs Tensor (using consolidated type check)
    llvm::Value* is_scheme_vector = tagged_.isVector(src_val);

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* scheme_vec_block = llvm::BasicBlock::Create(ctx_.context(), "sum_scheme_vec", current_func);
    llvm::BasicBlock* tensor_block = llvm::BasicBlock::Create(ctx_.context(), "sum_tensor", current_func);
    llvm::BasicBlock* sum_merge = llvm::BasicBlock::Create(ctx_.context(), "sum_merge", current_func);

    ctx_.builder().CreateCondBr(is_scheme_vector, scheme_vec_block, tensor_block);

    // === SCHEME VECTOR PATH ===
    ctx_.builder().SetInsertPoint(scheme_vec_block);
    llvm::Value* svec_ptr_int = tagged_.unpackInt64(src_val);
    llvm::Value* svec_ptr = ctx_.builder().CreateIntToPtr(svec_ptr_int, ctx_.ptrType());

    // Scheme vector: [length:i64, element0:tagged_value, element1:tagged_value, ...]
    llvm::Value* svec_len = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_ptr);

    // Sum loop for Scheme vector
    llvm::BasicBlock* svec_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "svec_sum_cond", current_func);
    llvm::BasicBlock* svec_loop_body = llvm::BasicBlock::Create(ctx_.context(), "svec_sum_body", current_func);
    llvm::BasicBlock* svec_loop_exit = llvm::BasicBlock::Create(ctx_.context(), "svec_sum_exit", current_func);

    // ESH-0121: tagged accumulator so a Scheme vector of DUAL_NUMBER jets sums
    // to a dual (preserving the mixed e1e2 second-order term) instead of being
    // flattened to a plain double, which silently zeros the Hessian.
    llvm::Value* svec_sum = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "svec_sum_acc");
    llvm::Value* svec_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "svec_sum_i");
    ctx_.builder().CreateStore(tagged_.packDouble(llvm::ConstantFP::get(ctx_.doubleType(), 0.0)), svec_sum);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), svec_counter);
    ctx_.builder().CreateBr(svec_loop_cond);

    ctx_.builder().SetInsertPoint(svec_loop_cond);
    llvm::Value* svec_i = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_counter);
    llvm::Value* svec_cmp = ctx_.builder().CreateICmpULT(svec_i, svec_len);
    ctx_.builder().CreateCondBr(svec_cmp, svec_loop_body, svec_loop_exit);

    ctx_.builder().SetInsertPoint(svec_loop_body);
    llvm::Value* svec_elems_base = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* svec_elems_typed = ctx_.builder().CreatePointerCast(svec_elems_base, ctx_.ptrType());
    llvm::Value* svec_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_elems_typed, svec_i);
    llvm::Value* svec_elem_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_elem_ptr);
    llvm::Value* svec_current_sum = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_sum);
    llvm::Value* svec_new_sum = dualAwareScalarBinOp(svec_current_sum, svec_elem_tagged, "add");
    ctx_.builder().CreateStore(svec_new_sum, svec_sum);
    // dualAwareScalarBinOp leaves the builder at its own merge block; the
    // back-edge and increment below are correctly emitted there.
    llvm::Value* svec_next_i = ctx_.builder().CreateAdd(svec_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(svec_next_i, svec_counter);
    ctx_.builder().CreateBr(svec_loop_cond);

    ctx_.builder().SetInsertPoint(svec_loop_exit);
    llvm::Value* svec_tagged_result = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_sum);
    ctx_.builder().CreateBr(sum_merge);
    llvm::BasicBlock* svec_exit_block = ctx_.builder().GetInsertBlock();

    // === TENSOR PATH ===
    ctx_.builder().SetInsertPoint(tensor_block);
    llvm::Value* src_ptr = unpackTensorOperandChecked(src_val, "tensor-sum");

    llvm::Value* src_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 2);
    llvm::Value* src_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_elements_field_ptr);
    llvm::Value* typed_src_elements = ctx_.builder().CreatePointerCast(src_elements_ptr, ctx_.ptrType());

    llvm::Value* src_total_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 3);
    llvm::Value* src_total = ctx_.builder().CreateLoad(ctx_.int64Type(), src_total_field_ptr);

    // === ESH-0121: DUAL-TENSOR SUM ===
    // A dual tensor (dtype == DUAL, from the reshape+matmul forward-over-forward
    // Hessian path) stores 16-byte tagged DUAL_NUMBER jets, not f64 bit patterns.
    // Reduce them with the exact dual add so the mixed e1e2 second-order term
    // survives, instead of the reverse-mode/numeric paths which would misread the
    // jets and silently zero the Hessian. Gated before the AD-mode branch so it
    // applies whether or not reverse mode is active.
    llvm::Value* dsum_result = nullptr;
    llvm::BasicBlock* dsum_exit_block = nullptr;
    if (autodiff_) {
        llvm::Value* tsum_is_dual = isDualTensor(src_ptr);
        llvm::BasicBlock* dsum_bb = llvm::BasicBlock::Create(ctx_.context(), "tsum_dual", current_func);
        llvm::BasicBlock* tsum_normal_bb = llvm::BasicBlock::Create(ctx_.context(), "tsum_not_dual", current_func);
        ctx_.builder().CreateCondBr(tsum_is_dual, dsum_bb, tsum_normal_bb);

        ctx_.builder().SetInsertPoint(dsum_bb);
        llvm::Value* dsum_acc = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "dsum_acc");
        llvm::Value* dsum_i = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "dsum_i");
        ctx_.builder().CreateStore(
            tagged_.packDouble(llvm::ConstantFP::get(ctx_.doubleType(), 0.0)), dsum_acc);
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dsum_i);
        llvm::BasicBlock* dsum_cond = llvm::BasicBlock::Create(ctx_.context(), "tsum_dual_cond", current_func);
        llvm::BasicBlock* dsum_body = llvm::BasicBlock::Create(ctx_.context(), "tsum_dual_body", current_func);
        llvm::BasicBlock* dsum_done = llvm::BasicBlock::Create(ctx_.context(), "tsum_dual_done", current_func);
        ctx_.builder().CreateBr(dsum_cond);

        ctx_.builder().SetInsertPoint(dsum_cond);
        llvm::Value* dsum_iv = ctx_.builder().CreateLoad(ctx_.int64Type(), dsum_i);
        ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(dsum_iv, src_total), dsum_body, dsum_done);

        ctx_.builder().SetInsertPoint(dsum_body);
        llvm::Value* dsum_elem = ctx_.builder().CreateLoad(ctx_.taggedValueType(),
            ctx_.builder().CreateGEP(ctx_.taggedValueType(), typed_src_elements, dsum_iv));
        llvm::Value* dsum_cur = ctx_.builder().CreateLoad(ctx_.taggedValueType(), dsum_acc);
        llvm::Value* dsum_next = dualAwareScalarBinOp(dsum_cur, dsum_elem, "add");
        ctx_.builder().CreateStore(dsum_next, dsum_acc);
        // dualAwareScalarBinOp leaves the builder at its merge block; emit the
        // increment/back-edge there.
        ctx_.builder().CreateStore(
            ctx_.builder().CreateAdd(dsum_iv, llvm::ConstantInt::get(ctx_.int64Type(), 1)), dsum_i);
        ctx_.builder().CreateBr(dsum_cond);

        ctx_.builder().SetInsertPoint(dsum_done);
        dsum_result = ctx_.builder().CreateLoad(ctx_.taggedValueType(), dsum_acc);
        ctx_.builder().CreateBr(sum_merge);
        dsum_exit_block = ctx_.builder().GetInsertBlock();

        // The standard tensor sum below emits into the non-dual branch.
        ctx_.builder().SetInsertPoint(tsum_normal_bb);
    }

    // Sum all elements - SIMD Accelerated with XLA dispatch for large tensors
    // Dispatch hierarchy: XLA (≥100K elements) → SIMD → scalar
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Scalar loop blocks (always needed)
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "tsum_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "tsum_scalar_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "tsum_exit", current_func);

    // Accumulator and counter allocas - placed before XLA check so all paths can use them
    llvm::Value* sum = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "sum_acc");
    llvm::Value* scalar_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "sum_scalar_i");

    llvm::Value* ad_sum_tagged_result = nullptr;
    llvm::BasicBlock* ad_sum_exit_block = nullptr;
    if (autodiff_) {
        llvm::Value* in_ad_mode = ctx_.builder().CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
        llvm::BasicBlock* ad_sum_block = llvm::BasicBlock::Create(ctx_.context(), "tsum_ad", current_func);
        llvm::BasicBlock* numeric_sum_block = llvm::BasicBlock::Create(ctx_.context(), "tsum_numeric", current_func);
        ctx_.builder().CreateCondBr(in_ad_mode, ad_sum_block, numeric_sum_block);

        ctx_.builder().SetInsertPoint(ad_sum_block);
        llvm::Value* ad_acc = ctx_.builder().CreateAlloca(ctx_.ptrType(), nullptr, "tsum_ad_acc");
        llvm::Value* ad_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "tsum_ad_i");
        llvm::Value* zero_node = autodiff_->createADConstant(llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        ctx_.builder().CreateStore(zero_node, ad_acc);
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ad_counter);

        llvm::BasicBlock* ad_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "tsum_ad_cond", current_func);
        llvm::BasicBlock* ad_loop_body = llvm::BasicBlock::Create(ctx_.context(), "tsum_ad_body", current_func);
        llvm::BasicBlock* ad_loop_exit = llvm::BasicBlock::Create(ctx_.context(), "tsum_ad_exit", current_func);
        ctx_.builder().CreateBr(ad_loop_cond);

        ctx_.builder().SetInsertPoint(ad_loop_cond);
        llvm::Value* ad_i = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_counter);
        llvm::Value* ad_has_elem = ctx_.builder().CreateICmpULT(ad_i, src_total);
        ctx_.builder().CreateCondBr(ad_has_elem, ad_loop_body, ad_loop_exit);

        ctx_.builder().SetInsertPoint(ad_loop_body);
        llvm::Value* ad_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_src_elements, ad_i);
        llvm::Value* ad_elem_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_elem_ptr);
        llvm::Value* ad_elem_node = adNodeFromTensorElementBits(ad_elem_bits, "tsum_ad_elem");
        llvm::Value* ad_old_acc = ctx_.builder().CreateLoad(ctx_.ptrType(), ad_acc);
        llvm::Value* ad_new_acc = autodiff_->recordADNodeBinary(2, ad_old_acc, ad_elem_node);
        ctx_.builder().CreateStore(ad_new_acc, ad_acc);
        llvm::Value* ad_next_i = ctx_.builder().CreateAdd(ad_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        ctx_.builder().CreateStore(ad_next_i, ad_counter);
        ctx_.builder().CreateBr(ad_loop_cond);

        ctx_.builder().SetInsertPoint(ad_loop_exit);
        llvm::Value* final_ad_sum = ctx_.builder().CreateLoad(ctx_.ptrType(), ad_acc);
        ad_sum_tagged_result = tagged_.packPtr(final_ad_sum, ESHKOL_VALUE_CALLABLE);
        ctx_.builder().CreateBr(sum_merge);
        ad_sum_exit_block = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(numeric_sum_block);
    }

#ifdef ESHKOL_XLA_ENABLED
    // ===== XLA DISPATCH FOR LARGE TENSOR SUM =====
    if (xla_ && xla_->isAvailable()) {
        llvm::Value* threshold = llvm::ConstantInt::get(ctx_.int64Type(), xla::xla_get_threshold());
        llvm::Value* use_xla = ctx_.builder().CreateICmpUGE(src_total, threshold);

        llvm::BasicBlock* xla_sum_block = llvm::BasicBlock::Create(ctx_.context(), "tsum_xla", current_func);
        llvm::BasicBlock* simd_sum_block = llvm::BasicBlock::Create(ctx_.context(), "tsum_simd_fallback", current_func);

        ctx_.builder().CreateCondBr(use_xla, xla_sum_block, simd_sum_block);

        // XLA path: emit reduce SUM (axis=-1 for reduce all)
        ctx_.builder().SetInsertPoint(xla_sum_block);
        llvm::Value* xla_result = xla_->emitReduce(src_ptr, -1, xla::ReduceOp::SUM);
        if (xla_result) {
            // XLA reduce returns a scalar tensor; extract the single element as a double
            llvm::Value* xla_elems_ptr = ctx_.builder().CreateStructGEP(tensor_type, xla_result,
                TypeSystem::TENSOR_ELEMENTS_IDX, "xla_sum_elems_ptr");
            llvm::Value* xla_elems = ctx_.builder().CreateLoad(ctx_.ptrType(), xla_elems_ptr, "xla_sum_elems");
            // Elements are stored as int64 bitpatterns of doubles
            llvm::Value* xla_int_val = ctx_.builder().CreateLoad(ctx_.int64Type(), xla_elems, "xla_sum_int");
            llvm::Value* xla_sum_val = ctx_.builder().CreateBitCast(xla_int_val, ctx_.doubleType(), "xla_sum_dbl");
            // Store XLA result in the sum accumulator and skip to loop_exit
            ctx_.builder().CreateStore(xla_sum_val, sum);
            ctx_.builder().CreateBr(loop_exit);
        } else {
            // XLA returned nullptr, fall back to SIMD
            ctx_.builder().CreateBr(simd_sum_block);
        }

        // Continue with SIMD/scalar fallback
        ctx_.builder().SetInsertPoint(simd_sum_block);
    }
#endif

    if (use_simd) {
        // Calculate SIMD iteration count
        llvm::Value* simd_count = ctx_.builder().CreateUDiv(src_total,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        llvm::Value* simd_elements = ctx_.builder().CreateMul(simd_count,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

        // Initialize vector accumulator
        llvm::Value* vec_acc = ctx_.builder().CreateAlloca(vec_type, nullptr, "sum_vec_acc");
        llvm::Value* zero_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH), llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        ctx_.builder().CreateStore(zero_vec, vec_acc);

        // SIMD loop blocks
        llvm::BasicBlock* simd_cond = llvm::BasicBlock::Create(ctx_.context(), "tsum_simd_cond", current_func);
        llvm::BasicBlock* simd_body = llvm::BasicBlock::Create(ctx_.context(), "tsum_simd_body", current_func);
        llvm::BasicBlock* simd_exit = llvm::BasicBlock::Create(ctx_.context(), "tsum_simd_exit", current_func);

        llvm::Value* simd_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "sum_simd_i");
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), simd_counter);
        ctx_.builder().CreateBr(simd_cond);

        // SIMD loop condition
        ctx_.builder().SetInsertPoint(simd_cond);
        llvm::Value* simd_i = ctx_.builder().CreateLoad(ctx_.int64Type(), simd_counter);
        llvm::Value* simd_cmp = ctx_.builder().CreateICmpULT(simd_i, simd_elements);
        ctx_.builder().CreateCondBr(simd_cmp, simd_body, simd_exit);

        // SIMD loop body: load SIMD_WIDTH elements, accumulate
        ctx_.builder().SetInsertPoint(simd_body);
        llvm::Value* vec_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_src_elements, simd_i);
        llvm::Value* vec_val = ctx_.builder().CreateAlignedLoad(vec_type, vec_ptr, llvm::MaybeAlign(8), "sum_vec");
        llvm::Value* old_acc = ctx_.builder().CreateLoad(vec_type, vec_acc);
        llvm::Value* new_acc = ctx_.builder().CreateFAdd(old_acc, vec_val, "acc_vec");
        ctx_.builder().CreateStore(new_acc, vec_acc);

        llvm::Value* next_simd_i = ctx_.builder().CreateAdd(simd_i,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        ctx_.builder().CreateStore(next_simd_i, simd_counter);
        ctx_.builder().CreateBr(simd_cond);

        // SIMD exit: horizontal sum
        ctx_.builder().SetInsertPoint(simd_exit);
        llvm::Value* final_vec = ctx_.builder().CreateLoad(vec_type, vec_acc);
        llvm::Value* simd_sum = ctx_.builder().CreateExtractElement(final_vec, (uint64_t)0);
        for (unsigned lane = 1; lane < SIMD_WIDTH; ++lane) {
            llvm::Value* elem = ctx_.builder().CreateExtractElement(final_vec, (uint64_t)lane);
            simd_sum = ctx_.builder().CreateFAdd(simd_sum, elem);
        }

        // Store partial sum and start scalar tail from simd_elements
        ctx_.builder().CreateStore(simd_sum, sum);
        ctx_.builder().CreateStore(simd_elements, scalar_counter);
        ctx_.builder().CreateBr(scalar_cond);
    } else {
        // ===== SCALAR-ONLY PATH =====
        // Initialize sum to 0 and counter to 0
        ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), scalar_counter);
        ctx_.builder().CreateBr(scalar_cond);
    }

    // Scalar tail loop condition
    ctx_.builder().SetInsertPoint(scalar_cond);
    llvm::Value* scalar_i = ctx_.builder().CreateLoad(ctx_.int64Type(), scalar_counter);
    llvm::Value* scalar_cmp = ctx_.builder().CreateICmpULT(scalar_i, src_total);
    ctx_.builder().CreateCondBr(scalar_cmp, scalar_body, loop_exit);

    // Scalar tail loop body
    ctx_.builder().SetInsertPoint(scalar_body);
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_src_elements, scalar_i);
    llvm::Value* elem_val = ctx_.builder().CreateLoad(ctx_.doubleType(), elem_ptr);
    llvm::Value* current_sum = ctx_.builder().CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* new_sum = ctx_.builder().CreateFAdd(current_sum, elem_val);
    ctx_.builder().CreateStore(new_sum, sum);

    llvm::Value* next_scalar_i = ctx_.builder().CreateAdd(scalar_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_scalar_i, scalar_counter);
    ctx_.builder().CreateBr(scalar_cond);

    // Final result
    ctx_.builder().SetInsertPoint(loop_exit);
    llvm::Value* tensor_result = ctx_.builder().CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* tensor_tagged_result = tagged_.packDouble(tensor_result);
    ctx_.builder().CreateBr(sum_merge);
    llvm::BasicBlock* tensor_exit_block = ctx_.builder().GetInsertBlock();

    // === MERGE RESULTS ===
    ctx_.builder().SetInsertPoint(sum_merge);
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), ad_sum_exit_block ? 3 : 2, "sum_result");
    result_phi->addIncoming(svec_tagged_result, svec_exit_block);
    result_phi->addIncoming(tensor_tagged_result, tensor_exit_block);
    if (ad_sum_exit_block && ad_sum_tagged_result) {
        result_phi->addIncoming(ad_sum_tagged_result, ad_sum_exit_block);
    }
    // ESH-0121: dual-tensor sum branch (see above).
    if (dsum_exit_block && dsum_result) {
        result_phi->addIncoming(dsum_result, dsum_exit_block);
    }

    return result_phi;
}

llvm::Value* TensorCodegen::tensorMean(const eshkol_operations_t* op) {
    // tensor-mean: (tensor-mean tensor) or (tensor-mean tensor axis) - Mean of elements
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("tensor-mean requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* src_val = codegenAST(&op->call_op.variables[0]);
    if (!src_val) return nullptr;

    // 2-arg case: (tensor-mean tensor axis) → mean along axis, returns tensor
    if (op->call_op.num_vars == 2) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[1]);
        if (!axis_val) return nullptr;
        return emitAxisReduce(src_val, axis_val, 1); // MEAN=1
    }

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Check type: Scheme vector vs Tensor (using consolidated type check)
    llvm::Value* is_scheme_vector = tagged_.isVector(src_val);

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* scheme_vec_block = llvm::BasicBlock::Create(ctx_.context(), "mean_scheme_vec", current_func);
    llvm::BasicBlock* tensor_block = llvm::BasicBlock::Create(ctx_.context(), "mean_tensor", current_func);
    llvm::BasicBlock* mean_merge = llvm::BasicBlock::Create(ctx_.context(), "mean_merge", current_func);

    ctx_.builder().CreateCondBr(is_scheme_vector, scheme_vec_block, tensor_block);

    // === SCHEME VECTOR PATH ===
    ctx_.builder().SetInsertPoint(scheme_vec_block);
    llvm::Value* svec_ptr_int = tagged_.unpackInt64(src_val);
    llvm::Value* svec_ptr = ctx_.builder().CreateIntToPtr(svec_ptr_int, ctx_.ptrType());
    llvm::Value* svec_len = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_ptr);

    // Sum loop for Scheme vector
    llvm::BasicBlock* svec_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "svec_mean_cond", current_func);
    llvm::BasicBlock* svec_loop_body = llvm::BasicBlock::Create(ctx_.context(), "svec_mean_body", current_func);
    llvm::BasicBlock* svec_loop_exit = llvm::BasicBlock::Create(ctx_.context(), "svec_mean_exit", current_func);

    llvm::Value* svec_sum = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "svec_mean_acc");
    llvm::Value* svec_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "svec_mean_i");
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), svec_sum);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), svec_counter);
    ctx_.builder().CreateBr(svec_loop_cond);

    ctx_.builder().SetInsertPoint(svec_loop_cond);
    llvm::Value* svec_i = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_counter);
    llvm::Value* svec_cmp = ctx_.builder().CreateICmpULT(svec_i, svec_len);
    ctx_.builder().CreateCondBr(svec_cmp, svec_loop_body, svec_loop_exit);

    ctx_.builder().SetInsertPoint(svec_loop_body);
    llvm::Value* svec_elems_base = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* svec_elems_typed = ctx_.builder().CreatePointerCast(svec_elems_base, ctx_.ptrType());
    llvm::Value* svec_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_elems_typed, svec_i);
    llvm::Value* svec_elem_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_elem_ptr);
    llvm::Value* svec_elem_val = extractAsDouble(svec_elem_tagged);
    llvm::Value* svec_current_sum = ctx_.builder().CreateLoad(ctx_.doubleType(), svec_sum);
    llvm::Value* svec_new_sum = ctx_.builder().CreateFAdd(svec_current_sum, svec_elem_val);
    ctx_.builder().CreateStore(svec_new_sum, svec_sum);
    llvm::Value* svec_next_i = ctx_.builder().CreateAdd(svec_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(svec_next_i, svec_counter);
    ctx_.builder().CreateBr(svec_loop_cond);

    ctx_.builder().SetInsertPoint(svec_loop_exit);
    llvm::Value* svec_total = ctx_.builder().CreateLoad(ctx_.doubleType(), svec_sum);
    llvm::Value* svec_len_fp = ctx_.builder().CreateSIToFP(svec_len, ctx_.doubleType());
    llvm::Value* svec_result = ctx_.builder().CreateFDiv(svec_total, svec_len_fp);
    llvm::Value* svec_tagged_result = tagged_.packDouble(svec_result);
    ctx_.builder().CreateBr(mean_merge);
    llvm::BasicBlock* svec_exit_block = ctx_.builder().GetInsertBlock();

    // === TENSOR PATH ===
    ctx_.builder().SetInsertPoint(tensor_block);
    llvm::Value* src_ptr = unpackTensorOperandChecked(src_val, "tensor-mean");

    llvm::Value* src_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 2);
    llvm::Value* src_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_elements_field_ptr);
    llvm::Value* typed_src_elements = ctx_.builder().CreatePointerCast(src_elements_ptr, ctx_.ptrType());

    llvm::Value* src_total_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 3);
    llvm::Value* src_total = ctx_.builder().CreateLoad(ctx_.int64Type(), src_total_field_ptr);

    // Mean of all elements - SIMD Accelerated with XLA dispatch for large tensors
    // Dispatch hierarchy: XLA (≥100K elements) → SIMD → scalar
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Scalar loop blocks (always needed)
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "tmean_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "tmean_scalar_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "tmean_exit", current_func);

    // Accumulator and counter allocas - placed before XLA check so all paths can use them
    llvm::Value* sum = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "mean_acc");
    llvm::Value* scalar_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "mean_scalar_i");

    llvm::Value* ad_mean_tagged_result = nullptr;
    llvm::BasicBlock* ad_mean_exit_block = nullptr;
    if (autodiff_) {
        llvm::Value* in_ad_mode = ctx_.builder().CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
        llvm::BasicBlock* ad_mean_block = llvm::BasicBlock::Create(ctx_.context(), "tmean_ad", current_func);
        llvm::BasicBlock* numeric_mean_block = llvm::BasicBlock::Create(ctx_.context(), "tmean_numeric", current_func);
        ctx_.builder().CreateCondBr(in_ad_mode, ad_mean_block, numeric_mean_block);

        ctx_.builder().SetInsertPoint(ad_mean_block);
        llvm::Value* ad_acc = ctx_.builder().CreateAlloca(ctx_.ptrType(), nullptr, "tmean_ad_acc");
        llvm::Value* ad_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "tmean_ad_i");
        llvm::Value* zero_node = autodiff_->createADConstant(llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        ctx_.builder().CreateStore(zero_node, ad_acc);
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ad_counter);

        llvm::BasicBlock* ad_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "tmean_ad_cond", current_func);
        llvm::BasicBlock* ad_loop_body = llvm::BasicBlock::Create(ctx_.context(), "tmean_ad_body", current_func);
        llvm::BasicBlock* ad_loop_exit = llvm::BasicBlock::Create(ctx_.context(), "tmean_ad_exit", current_func);
        ctx_.builder().CreateBr(ad_loop_cond);

        ctx_.builder().SetInsertPoint(ad_loop_cond);
        llvm::Value* ad_i = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_counter);
        llvm::Value* ad_has_elem = ctx_.builder().CreateICmpULT(ad_i, src_total);
        ctx_.builder().CreateCondBr(ad_has_elem, ad_loop_body, ad_loop_exit);

        ctx_.builder().SetInsertPoint(ad_loop_body);
        llvm::Value* ad_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_src_elements, ad_i);
        llvm::Value* ad_elem_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_elem_ptr);
        llvm::Value* ad_elem_node = adNodeFromTensorElementBits(ad_elem_bits, "tmean_ad_elem");
        llvm::Value* ad_old_acc = ctx_.builder().CreateLoad(ctx_.ptrType(), ad_acc);
        llvm::Value* ad_new_acc = autodiff_->recordADNodeBinary(2, ad_old_acc, ad_elem_node);
        ctx_.builder().CreateStore(ad_new_acc, ad_acc);
        llvm::Value* ad_next_i = ctx_.builder().CreateAdd(ad_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        ctx_.builder().CreateStore(ad_next_i, ad_counter);
        ctx_.builder().CreateBr(ad_loop_cond);

        ctx_.builder().SetInsertPoint(ad_loop_exit);
        llvm::Value* final_ad_sum = ctx_.builder().CreateLoad(ctx_.ptrType(), ad_acc);
        llvm::Value* count_fp = ctx_.builder().CreateSIToFP(src_total, ctx_.doubleType());
        llvm::Value* count_node = autodiff_->createADConstant(count_fp);
        llvm::Value* final_ad_mean = autodiff_->recordADNodeBinary(5, final_ad_sum, count_node);
        ad_mean_tagged_result = tagged_.packPtr(final_ad_mean, ESHKOL_VALUE_CALLABLE);
        ctx_.builder().CreateBr(mean_merge);
        ad_mean_exit_block = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(numeric_mean_block);
    }

#ifdef ESHKOL_XLA_ENABLED
    // ===== XLA DISPATCH FOR LARGE TENSOR MEAN =====
    if (xla_ && xla_->isAvailable()) {
        llvm::Value* threshold = llvm::ConstantInt::get(ctx_.int64Type(), xla::xla_get_threshold());
        llvm::Value* use_xla = ctx_.builder().CreateICmpUGE(src_total, threshold);

        llvm::BasicBlock* xla_mean_block = llvm::BasicBlock::Create(ctx_.context(), "tmean_xla", current_func);
        llvm::BasicBlock* simd_mean_block = llvm::BasicBlock::Create(ctx_.context(), "tmean_simd_fallback", current_func);

        ctx_.builder().CreateCondBr(use_xla, xla_mean_block, simd_mean_block);

        // XLA path: emit reduce MEAN (axis=-1 for reduce all)
        ctx_.builder().SetInsertPoint(xla_mean_block);
        llvm::Value* xla_result = xla_->emitReduce(src_ptr, -1, xla::ReduceOp::MEAN);
        if (xla_result) {
            // XLA reduce returns a scalar tensor; extract the single element as a double
            llvm::Value* xla_elems_ptr = ctx_.builder().CreateStructGEP(tensor_type, xla_result,
                TypeSystem::TENSOR_ELEMENTS_IDX, "xla_mean_elems_ptr");
            llvm::Value* xla_elems = ctx_.builder().CreateLoad(ctx_.ptrType(), xla_elems_ptr, "xla_mean_elems");
            llvm::Value* xla_int_val = ctx_.builder().CreateLoad(ctx_.int64Type(), xla_elems, "xla_mean_int");
            llvm::Value* xla_mean_val = ctx_.builder().CreateBitCast(xla_int_val, ctx_.doubleType(), "xla_mean_dbl");
            // Store mean directly in sum (loop_exit divides by count, so store sum = mean * count)
            llvm::Value* count_fp_xla = ctx_.builder().CreateSIToFP(src_total, ctx_.doubleType());
            llvm::Value* xla_sum_equiv = ctx_.builder().CreateFMul(xla_mean_val, count_fp_xla, "xla_mean_as_sum");
            ctx_.builder().CreateStore(xla_sum_equiv, sum);
            ctx_.builder().CreateBr(loop_exit);
        } else {
            // XLA returned nullptr, fall back to SIMD
            ctx_.builder().CreateBr(simd_mean_block);
        }

        // Continue with SIMD/scalar fallback
        ctx_.builder().SetInsertPoint(simd_mean_block);
    }
#endif

    if (use_simd) {
        // Calculate SIMD iteration count
        llvm::Value* simd_count = ctx_.builder().CreateUDiv(src_total,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        llvm::Value* simd_elements = ctx_.builder().CreateMul(simd_count,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

        // Initialize vector accumulator
        llvm::Value* vec_acc = ctx_.builder().CreateAlloca(vec_type, nullptr, "mean_vec_acc");
        llvm::Value* zero_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH), llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        ctx_.builder().CreateStore(zero_vec, vec_acc);

        // SIMD loop blocks
        llvm::BasicBlock* simd_cond = llvm::BasicBlock::Create(ctx_.context(), "tmean_simd_cond", current_func);
        llvm::BasicBlock* simd_body = llvm::BasicBlock::Create(ctx_.context(), "tmean_simd_body", current_func);
        llvm::BasicBlock* simd_exit = llvm::BasicBlock::Create(ctx_.context(), "tmean_simd_exit", current_func);

        llvm::Value* simd_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "mean_simd_i");
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), simd_counter);
        ctx_.builder().CreateBr(simd_cond);

        // SIMD loop condition
        ctx_.builder().SetInsertPoint(simd_cond);
        llvm::Value* simd_i = ctx_.builder().CreateLoad(ctx_.int64Type(), simd_counter);
        llvm::Value* simd_cmp = ctx_.builder().CreateICmpULT(simd_i, simd_elements);
        ctx_.builder().CreateCondBr(simd_cmp, simd_body, simd_exit);

        // SIMD loop body: load SIMD_WIDTH elements, accumulate
        ctx_.builder().SetInsertPoint(simd_body);
        llvm::Value* vec_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_src_elements, simd_i);
        llvm::Value* vec_val = ctx_.builder().CreateAlignedLoad(vec_type, vec_ptr, llvm::MaybeAlign(8), "mean_vec");
        llvm::Value* old_acc = ctx_.builder().CreateLoad(vec_type, vec_acc);
        llvm::Value* new_acc = ctx_.builder().CreateFAdd(old_acc, vec_val, "acc_vec");
        ctx_.builder().CreateStore(new_acc, vec_acc);

        llvm::Value* next_simd_i = ctx_.builder().CreateAdd(simd_i,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        ctx_.builder().CreateStore(next_simd_i, simd_counter);
        ctx_.builder().CreateBr(simd_cond);

        // SIMD exit: horizontal sum
        ctx_.builder().SetInsertPoint(simd_exit);
        llvm::Value* final_vec = ctx_.builder().CreateLoad(vec_type, vec_acc);
        llvm::Value* simd_sum = ctx_.builder().CreateExtractElement(final_vec, (uint64_t)0);
        for (unsigned lane = 1; lane < SIMD_WIDTH; ++lane) {
            llvm::Value* elem = ctx_.builder().CreateExtractElement(final_vec, (uint64_t)lane);
            simd_sum = ctx_.builder().CreateFAdd(simd_sum, elem);
        }

        // Store partial sum and start scalar tail from simd_elements
        ctx_.builder().CreateStore(simd_sum, sum);
        ctx_.builder().CreateStore(simd_elements, scalar_counter);
        ctx_.builder().CreateBr(scalar_cond);
    } else {
        // ===== SCALAR-ONLY PATH =====
        // Initialize sum to 0 and counter to 0
        ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), scalar_counter);
        ctx_.builder().CreateBr(scalar_cond);
    }

    // Scalar tail loop condition
    ctx_.builder().SetInsertPoint(scalar_cond);
    llvm::Value* scalar_i = ctx_.builder().CreateLoad(ctx_.int64Type(), scalar_counter);
    llvm::Value* scalar_cmp = ctx_.builder().CreateICmpULT(scalar_i, src_total);
    ctx_.builder().CreateCondBr(scalar_cmp, scalar_body, loop_exit);

    // Scalar tail loop body
    ctx_.builder().SetInsertPoint(scalar_body);
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_src_elements, scalar_i);
    llvm::Value* elem_val = ctx_.builder().CreateLoad(ctx_.doubleType(), elem_ptr);
    llvm::Value* current_sum = ctx_.builder().CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* new_sum = ctx_.builder().CreateFAdd(current_sum, elem_val);
    ctx_.builder().CreateStore(new_sum, sum);

    llvm::Value* next_scalar_i = ctx_.builder().CreateAdd(scalar_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_scalar_i, scalar_counter);
    ctx_.builder().CreateBr(scalar_cond);

    // Final result: sum / count
    ctx_.builder().SetInsertPoint(loop_exit);
    llvm::Value* total_sum = ctx_.builder().CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* count_fp = ctx_.builder().CreateSIToFP(src_total, ctx_.doubleType());
    llvm::Value* tensor_result = ctx_.builder().CreateFDiv(total_sum, count_fp);
    llvm::Value* tensor_tagged_result = tagged_.packDouble(tensor_result);
    ctx_.builder().CreateBr(mean_merge);
    llvm::BasicBlock* tensor_exit_block = ctx_.builder().GetInsertBlock();

    // === MERGE RESULTS ===
    ctx_.builder().SetInsertPoint(mean_merge);
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), ad_mean_exit_block ? 3 : 2, "mean_result");
    result_phi->addIncoming(svec_tagged_result, svec_exit_block);
    result_phi->addIncoming(tensor_tagged_result, tensor_exit_block);
    if (ad_mean_exit_block && ad_mean_tagged_result) {
        result_phi->addIncoming(ad_mean_tagged_result, ad_mean_exit_block);
    }

    return result_phi;
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
