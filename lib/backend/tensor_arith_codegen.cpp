/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen — Internal & SIMD-Accelerated Tensor Arithmetic.
 * Extracted from tensor_codegen.cpp during the v1.2 mechanical split.
 *
 * Covers the elementwise arithmetic kernels (add, sub, mul, div, etc.)
 * and their SIMD-accelerated variants. Higher-level operations layer
 * on top via tensorArithmeticInternal in tensor_reduce_codegen.cpp.
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-arith-extract baseline.
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

// ===== INTERNAL TENSOR ARITHMETIC IMPLEMENTATIONS =====

// Scheme vector arithmetic: vectors with [length:i64][elem0:tagged][elem1:tagged]...
// ESH-0121: forward-mode-dual-aware elementwise binary op. See header.
llvm::Value* TensorCodegen::dualAwareScalarBinOp(llvm::Value* a_tagged, llvm::Value* b_tagged,
                                                 const std::string& operation) {
    // Ensure tagged form so the dual tag is inspectable.
    if (a_tagged->getType() != ctx_.taggedValueType()) a_tagged = tagged_.packDouble(extractAsDouble(a_tagged));
    if (b_tagged->getType() != ctx_.taggedValueType()) b_tagged = tagged_.packDouble(extractAsDouble(b_tagged));

    auto numericResult = [&](llvm::Value* av, llvm::Value* bv) -> llvm::Value* {
        if (operation == "add") return ctx_.builder().CreateFAdd(av, bv);
        if (operation == "sub") return ctx_.builder().CreateFSub(av, bv);
        if (operation == "mul") return ctx_.builder().CreateFMul(av, bv);
        if (operation == "div") return ctx_.builder().CreateFDiv(av, bv);
        if (operation == "pow") {
            llvm::Function* pow_fn = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::pow, {ctx_.doubleType()});
            return ctx_.builder().CreateCall(pow_fn, {av, bv});
        }
        if (operation == "max") {
            llvm::Function* max_fn = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::maxnum, {ctx_.doubleType()});
            return ctx_.builder().CreateCall(max_fn, {av, bv});
        }
        if (operation == "min") {
            llvm::Function* min_fn = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::minnum, {ctx_.doubleType()});
            return ctx_.builder().CreateCall(min_fn, {av, bv});
        }
        return llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    };

    // Without autodiff wired there are never duals in flight: plain path.
    if (!autodiff_) {
        return tagged_.packDouble(numericResult(extractAsDouble(a_tagged), extractAsDouble(b_tagged)));
    }

    llvm::Value* a_is_dual = ctx_.builder().CreateICmpEQ(
        tagged_.getBaseType(tagged_.getType(a_tagged)),
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
    llvm::Value* b_is_dual = ctx_.builder().CreateICmpEQ(
        tagged_.getBaseType(tagged_.getType(b_tagged)),
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
    llvm::Value* any_dual = ctx_.builder().CreateOr(a_is_dual, b_is_dual);

    llvm::Function* fn = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* dual_bb = llvm::BasicBlock::Create(ctx_.context(), "vbin_dual", fn);
    llvm::BasicBlock* num_bb = llvm::BasicBlock::Create(ctx_.context(), "vbin_num", fn);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "vbin_merge", fn);
    ctx_.builder().CreateCondBr(any_dual, dual_bb, num_bb);

    // Dual path: lift both operands and apply the exact dual rule.
    ctx_.builder().SetInsertPoint(dual_bb);
    llvm::Value* da = autodiff_->safeUnpackDualFromTagged(a_tagged);
    llvm::Value* db = autodiff_->safeUnpackDualFromTagged(b_tagged);
    llvm::Value* dr = nullptr;
    if (operation == "add") dr = autodiff_->dualAdd(da, db);
    else if (operation == "sub") dr = autodiff_->dualSub(da, db);
    else if (operation == "mul") dr = autodiff_->dualMul(da, db);
    else if (operation == "div") dr = autodiff_->dualDiv(da, db);
    else if (operation == "pow") dr = autodiff_->dualPow(da, db);
    else if (operation == "max") dr = autodiff_->dualMax(da, db);
    else if (operation == "min") dr = autodiff_->dualMin(da, db);
    else dr = autodiff_->dualMul(da, db);
    llvm::Value* dual_tagged = autodiff_->packDualToTagged(dr);
    llvm::BasicBlock* dual_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(merge_bb);

    // Numeric path: plain doubles.
    ctx_.builder().SetInsertPoint(num_bb);
    llvm::Value* num_tagged = tagged_.packDouble(numericResult(extractAsDouble(a_tagged), extractAsDouble(b_tagged)));
    llvm::BasicBlock* num_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(merge_bb);

    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "vbin_result");
    phi->addIncoming(dual_tagged, dual_exit);
    phi->addIncoming(num_tagged, num_exit);
    return phi;
}

// ESH-0121: dtype==DUAL predicate. See header.
llvm::Value* TensorCodegen::isDualTensor(llvm::Value* tensor_struct_ptr) {
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* dtype_field = ctx_.builder().CreateStructGEP(tensor_type, tensor_struct_ptr, 4);
    llvm::Value* dtype = ctx_.builder().CreateLoad(ctx_.int64Type(), dtype_field);
    return ctx_.builder().CreateICmpEQ(dtype,
        llvm::ConstantInt::get(ctx_.int64Type(), TENSOR_DTYPE_DUAL));
}

// ESH-0121: exact forward-mode matmul over dual tensors. See header.
llvm::Value* TensorCodegen::dualTensorMatmul(llvm::Value* a_struct_ptr, llvm::Value* b_struct_ptr) {
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::IRBuilder<>& b = ctx_.builder();
    llvm::LLVMContext& c = ctx_.context();
    llvm::Function* fn = b.GetInsertBlock()->getParent();
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* arena_ptr = b.CreateLoad(ctx_.ptrType(), ctx_.globalArena());

    const uint64_t kTagBytes = 16;  // sizeof(eshkol_tagged_value)

    auto loadDim = [&](llvm::Value* struct_ptr, unsigned which) -> llvm::Value* {
        llvm::Value* dims_field = b.CreateStructGEP(tensor_type, struct_ptr, 0);
        llvm::Value* dims_ptr = b.CreateLoad(ctx_.ptrType(), dims_field);
        llvm::Value* slot = b.CreateGEP(ctx_.int64Type(), dims_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), which));
        return b.CreateLoad(ctx_.int64Type(), slot);
    };
    auto loadNdim = [&](llvm::Value* struct_ptr) -> llvm::Value* {
        llvm::Value* nd_field = b.CreateStructGEP(tensor_type, struct_ptr, 1);
        return b.CreateLoad(ctx_.int64Type(), nd_field);
    };

    // --- Shape guard: require 2-D operands with A.cols == B.rows. Raise a
    // catchable error otherwise (never a silent wrong/zero result). ---
    llvm::Value* a_nd = loadNdim(a_struct_ptr);
    llvm::Value* b_nd = loadNdim(b_struct_ptr);
    llvm::Value* a_is2d = b.CreateICmpEQ(a_nd, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* b_is2d = b.CreateICmpEQ(b_nd, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* both2d = b.CreateAnd(a_is2d, b_is2d);
    llvm::Value* M = loadDim(a_struct_ptr, 0);
    llvm::Value* K = loadDim(a_struct_ptr, 1);
    llvm::Value* Kb = loadDim(b_struct_ptr, 0);
    llvm::Value* N = loadDim(b_struct_ptr, 1);
    llvm::Value* k_ok = b.CreateICmpEQ(K, Kb);
    llvm::Value* shape_ok = b.CreateAnd(both2d, k_ok);

    llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(c, "dmm_ok", fn);
    llvm::BasicBlock* bad_bb = llvm::BasicBlock::Create(c, "dmm_bad", fn);
    b.CreateCondBr(shape_ok, ok_bb, bad_bb);

    b.SetInsertPoint(bad_bb);
    {
        llvm::Function* raise_func = ctx_.module().getFunction("eshkol_raise");
        if (!raise_func) {
            llvm::FunctionType* rt = llvm::FunctionType::get(b.getVoidTy(), {ctx_.ptrType()}, false);
            raise_func = llvm::Function::Create(rt, llvm::Function::ExternalLinkage, "eshkol_raise", &ctx_.module());
            raise_func->setDoesNotReturn();
        }
        llvm::Function* make_exc = ctx_.module().getFunction("eshkol_make_exception_with_header");
        if (!make_exc) {
            llvm::FunctionType* mt = llvm::FunctionType::get(ctx_.ptrType(), {b.getInt32Ty(), ctx_.ptrType()}, false);
            make_exc = llvm::Function::Create(mt, llvm::Function::ExternalLinkage, "eshkol_make_exception_with_header", &ctx_.module());
        }
        llvm::Value* msg = b.CreateGlobalString(
            "matmul: dual-tensor (autodiff) matmul requires 2-D operands with A.cols == B.rows");
        llvm::Value* exc = b.CreateCall(make_exc, {llvm::ConstantInt::get(b.getInt32Ty(), 1), msg});
        b.CreateCall(raise_func, {exc});
        b.CreateUnreachable();
    }

    b.SetInsertPoint(ok_bb);

    // --- Normalize both operands to tagged (16-byte) element arrays so the
    // contraction loop is uniform. A plain (f64) operand is lifted to duals with
    // zero tangent via packDouble. ---
    auto taggedElems = [&](llvm::Value* struct_ptr) -> llvm::Value* {
        llvm::Value* is_dual = isDualTensor(struct_ptr);
        llvm::Value* elems_field = b.CreateStructGEP(tensor_type, struct_ptr, 2);
        llvm::Value* raw_elems = b.CreateLoad(ctx_.ptrType(), elems_field);
        llvm::Value* total_field = b.CreateStructGEP(tensor_type, struct_ptr, 3);
        llvm::Value* total = b.CreateLoad(ctx_.int64Type(), total_field);

        llvm::BasicBlock* use_bb = llvm::BasicBlock::Create(c, "dmm_elems_use", fn);
        llvm::BasicBlock* conv_bb = llvm::BasicBlock::Create(c, "dmm_elems_conv", fn);
        llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(c, "dmm_elems_merge", fn);
        b.CreateCondBr(is_dual, use_bb, conv_bb);

        // Already a tagged array.
        b.SetInsertPoint(use_bb);
        llvm::BasicBlock* use_exit = b.GetInsertBlock();
        b.CreateBr(merge_bb);

        // Plain f64 array -> tagged array copy.
        b.SetInsertPoint(conv_bb);
        llvm::Value* bytes = b.CreateMul(total, llvm::ConstantInt::get(ctx_.int64Type(), kTagBytes));
        llvm::Value* conv = b.CreateCall(arena_alloc, {arena_ptr, bytes}, "dmm_conv_elems");
        llvm::Value* cnt = b.CreateAlloca(ctx_.int64Type(), nullptr, "dmm_conv_i");
        b.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), cnt);
        llvm::BasicBlock* cc = llvm::BasicBlock::Create(c, "dmm_conv_cond", fn);
        llvm::BasicBlock* cb = llvm::BasicBlock::Create(c, "dmm_conv_body", fn);
        llvm::BasicBlock* ce = llvm::BasicBlock::Create(c, "dmm_conv_exit", fn);
        b.CreateBr(cc);
        b.SetInsertPoint(cc);
        llvm::Value* ci = b.CreateLoad(ctx_.int64Type(), cnt);
        b.CreateCondBr(b.CreateICmpULT(ci, total), cb, ce);
        b.SetInsertPoint(cb);
        // Raw element is an f64 bit pattern stored as i64.
        llvm::Value* src_slot = b.CreateGEP(ctx_.int64Type(), raw_elems, ci);
        llvm::Value* bits = b.CreateLoad(ctx_.int64Type(), src_slot);
        llvm::Value* dv = b.CreateBitCast(bits, ctx_.doubleType());
        llvm::Value* dst_slot = b.CreateGEP(ctx_.taggedValueType(), conv, ci);
        b.CreateStore(tagged_.packDouble(dv), dst_slot);
        b.CreateStore(b.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), cnt);
        b.CreateBr(cc);
        b.SetInsertPoint(ce);
        llvm::BasicBlock* conv_exit = b.GetInsertBlock();
        b.CreateBr(merge_bb);

        b.SetInsertPoint(merge_bb);
        llvm::PHINode* elems_phi = b.CreatePHI(ctx_.ptrType(), 2, "dmm_elems");
        elems_phi->addIncoming(raw_elems, use_exit);
        elems_phi->addIncoming(conv, conv_exit);
        return elems_phi;
    };

    llvm::Value* a_elems = taggedElems(a_struct_ptr);
    llvm::Value* b_elems = taggedElems(b_struct_ptr);

    // --- Allocate the output dual tensor: dims [M, N], MxN tagged elements. ---
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* out_ptr = b.CreateCall(alloc_tensor, {arena_ptr}, "dmm_out");

    llvm::Value* out_dims = b.CreateCall(arena_alloc,
        {arena_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 2 * sizeof(int64_t))}, "dmm_dims");
    b.CreateStore(M, out_dims);
    b.CreateStore(N, b.CreateGEP(ctx_.int64Type(), out_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1)));

    llvm::Value* MN = b.CreateMul(M, N);
    llvm::Value* out_bytes = b.CreateMul(MN, llvm::ConstantInt::get(ctx_.int64Type(), kTagBytes));
    llvm::Value* out_elems = b.CreateCall(arena_alloc, {arena_ptr, out_bytes}, "dmm_out_elems");

    b.CreateStore(out_dims, b.CreateStructGEP(tensor_type, out_ptr, 0));
    b.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 2), b.CreateStructGEP(tensor_type, out_ptr, 1));
    b.CreateStore(out_elems, b.CreateStructGEP(tensor_type, out_ptr, 2));
    b.CreateStore(MN, b.CreateStructGEP(tensor_type, out_ptr, 3));
    b.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), TENSOR_DTYPE_DUAL),
        b.CreateStructGEP(tensor_type, out_ptr, 4));

    // --- Contraction: C[i,j] = sum_k A[i,k] * B[k,j], exact dual arithmetic. ---
    llvm::Value* iv = b.CreateAlloca(ctx_.int64Type(), nullptr, "dmm_i");
    llvm::Value* jv = b.CreateAlloca(ctx_.int64Type(), nullptr, "dmm_j");
    llvm::Value* kv = b.CreateAlloca(ctx_.int64Type(), nullptr, "dmm_k");
    llvm::Value* acc = b.CreateAlloca(ctx_.taggedValueType(), nullptr, "dmm_acc");

    llvm::BasicBlock* i_cond = llvm::BasicBlock::Create(c, "dmm_i_cond", fn);
    llvm::BasicBlock* i_body = llvm::BasicBlock::Create(c, "dmm_i_body", fn);
    llvm::BasicBlock* i_exit = llvm::BasicBlock::Create(c, "dmm_i_exit", fn);
    llvm::BasicBlock* j_cond = llvm::BasicBlock::Create(c, "dmm_j_cond", fn);
    llvm::BasicBlock* j_body = llvm::BasicBlock::Create(c, "dmm_j_body", fn);
    llvm::BasicBlock* j_exit = llvm::BasicBlock::Create(c, "dmm_j_exit", fn);
    llvm::BasicBlock* k_cond = llvm::BasicBlock::Create(c, "dmm_k_cond", fn);
    llvm::BasicBlock* k_body = llvm::BasicBlock::Create(c, "dmm_k_body", fn);
    llvm::BasicBlock* k_exit = llvm::BasicBlock::Create(c, "dmm_k_exit", fn);

    b.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), iv);
    b.CreateBr(i_cond);
    b.SetInsertPoint(i_cond);
    llvm::Value* i = b.CreateLoad(ctx_.int64Type(), iv);
    b.CreateCondBr(b.CreateICmpULT(i, M), i_body, i_exit);

    b.SetInsertPoint(i_body);
    b.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), jv);
    b.CreateBr(j_cond);
    b.SetInsertPoint(j_cond);
    llvm::Value* j = b.CreateLoad(ctx_.int64Type(), jv);
    b.CreateCondBr(b.CreateICmpULT(j, N), j_body, j_exit);

    b.SetInsertPoint(j_body);
    b.CreateStore(tagged_.packDouble(llvm::ConstantFP::get(ctx_.doubleType(), 0.0)), acc);
    b.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), kv);
    b.CreateBr(k_cond);
    b.SetInsertPoint(k_cond);
    llvm::Value* k = b.CreateLoad(ctx_.int64Type(), kv);
    b.CreateCondBr(b.CreateICmpULT(k, K), k_body, k_exit);

    b.SetInsertPoint(k_body);
    // A[i,k] at index i*K + k ; B[k,j] at index k*N + j
    llvm::Value* a_idx = b.CreateAdd(b.CreateMul(i, K), k);
    llvm::Value* b_idx = b.CreateAdd(b.CreateMul(k, N), j);
    llvm::Value* a_el = b.CreateLoad(ctx_.taggedValueType(),
        b.CreateGEP(ctx_.taggedValueType(), a_elems, a_idx));
    llvm::Value* b_el = b.CreateLoad(ctx_.taggedValueType(),
        b.CreateGEP(ctx_.taggedValueType(), b_elems, b_idx));
    llvm::Value* prod = dualAwareScalarBinOp(a_el, b_el, "mul");
    llvm::Value* cur = b.CreateLoad(ctx_.taggedValueType(), acc);
    llvm::Value* nxt = dualAwareScalarBinOp(cur, prod, "add");
    b.CreateStore(nxt, acc);
    // dualAwareScalarBinOp leaves the builder at its merge block; emit the
    // increment/back-edge there.
    b.CreateStore(b.CreateAdd(k, llvm::ConstantInt::get(ctx_.int64Type(), 1)), kv);
    b.CreateBr(k_cond);

    b.SetInsertPoint(k_exit);
    llvm::Value* c_idx = b.CreateAdd(b.CreateMul(i, N), j);
    llvm::Value* c_val = b.CreateLoad(ctx_.taggedValueType(), acc);
    b.CreateStore(c_val, b.CreateGEP(ctx_.taggedValueType(), out_elems, c_idx));
    b.CreateStore(b.CreateAdd(j, llvm::ConstantInt::get(ctx_.int64Type(), 1)), jv);
    b.CreateBr(j_cond);

    b.SetInsertPoint(j_exit);
    b.CreateStore(b.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), iv);
    b.CreateBr(i_cond);

    b.SetInsertPoint(i_exit);
    return out_ptr;
}

llvm::Value* TensorCodegen::schemeVectorArithmetic(llvm::Value* vec1_tagged, llvm::Value* vec2_tagged, const std::string& operation) {
    // Extract pointers from tagged values
    llvm::Value* ptr1_int = vec1_tagged;
    llvm::Value* ptr2_int = vec2_tagged;
    if (vec1_tagged->getType() == ctx_.taggedValueType()) {
        ptr1_int = tagged_.unpackInt64(vec1_tagged);
    }
    if (vec2_tagged->getType() == ctx_.taggedValueType()) {
        ptr2_int = tagged_.unpackInt64(vec2_tagged);
    }

    llvm::Value* ptr1 = ctx_.builder().CreateIntToPtr(ptr1_int, ctx_.ptrType());
    llvm::Value* ptr2 = ctx_.builder().CreateIntToPtr(ptr2_int, ctx_.ptrType());

    // Get length from first vector
    llvm::Value* len_ptr = ctx_.builder().CreateBitCast(ptr1, ctx_.ptrType());
    llvm::Value* length = ctx_.builder().CreateLoad(ctx_.int64Type(), len_ptr);

    // Consolidated pointer system: Allocate result vector with header
    // arena_allocate_vector_with_header creates: [header(8)] + [length(8)] + [elements]
    llvm::GlobalVariable* arena_global = ctx_.globalArena();
    if (!arena_global) return tagged_.packNull();

    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), arena_global);
    llvm::Value* result_vec = ctx_.builder().CreateCall(
        mem_.getArenaAllocateVectorWithHeader(), {arena_ptr, length});

    // Store length in result (result_vec points to length field after header)
    llvm::Value* result_len_ptr = ctx_.builder().CreateBitCast(result_vec, ctx_.ptrType());
    ctx_.builder().CreateStore(length, result_len_ptr);

    // Get element bases (after 8-byte length field)
    llvm::Value* elems1_base = ctx_.builder().CreateGEP(ctx_.int8Type(), ptr1,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* elems2_base = ctx_.builder().CreateGEP(ctx_.int8Type(), ptr2,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* result_elems_base = ctx_.builder().CreateGEP(ctx_.int8Type(), result_vec,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));

    llvm::Value* elems1_typed = ctx_.builder().CreateBitCast(elems1_base, ctx_.ptrType());
    llvm::Value* elems2_typed = ctx_.builder().CreateBitCast(elems2_base, ctx_.ptrType());
    llvm::Value* result_elems_typed = ctx_.builder().CreateBitCast(result_elems_base, ctx_.ptrType());

    // Create loop
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "vec_arith_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "vec_arith_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "vec_arith_exit", current_func);

    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "vec_i");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    ctx_.builder().CreateBr(loop_cond);

    // Loop condition
    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cond = ctx_.builder().CreateICmpULT(i, length);
    ctx_.builder().CreateCondBr(cond, loop_body, loop_exit);

    // Loop body
    ctx_.builder().SetInsertPoint(loop_body);

    // Load tagged values at index i
    llvm::Value* elem1_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), elems1_typed, i);
    llvm::Value* elem2_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), elems2_typed, i);
    llvm::Value* result_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), result_elems_typed, i);

    llvm::Value* elem1_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), elem1_ptr);
    llvm::Value* elem2_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), elem2_ptr);

    // ESH-0121: dual-aware elementwise op. When the Hessian's forward-over-forward
    // sweep feeds a Scheme vector of DUAL_NUMBER jets, this propagates the exact
    // dual (including the mixed e1e2 second-order term) instead of dropping it to
    // a plain double and silently zeroing the second derivative.
    llvm::Value* result_tagged = dualAwareScalarBinOp(elem1_tagged, elem2_tagged, operation);
    ctx_.builder().CreateStore(result_tagged, result_elem_ptr);

    // Increment counter
    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, counter);
    ctx_.builder().CreateBr(loop_cond);

    // Loop exit - return result as tagged value (using consolidated HEAP_PTR)
    ctx_.builder().SetInsertPoint(loop_exit);
    return tagged_.packHeapPtr(result_vec);
}

// Raw tensor arithmetic: tensors with double elements in contiguous array
llvm::Value* TensorCodegen::rawTensorArithmetic(llvm::Value* arg1, llvm::Value* arg2, const std::string& operation) {
    auto& builder = ctx_.builder();

    // Get raw int64 values (tensor pointers)
    llvm::Value* tensor1_int = tagged_.unpackInt64(arg1);
    llvm::Value* tensor2_int = tagged_.unpackInt64(arg2);

    llvm::Value* tensor1_ptr = builder.CreateIntToPtr(tensor1_int, ctx_.ptrType());
    llvm::Value* tensor2_ptr = builder.CreateIntToPtr(tensor2_int, ctx_.ptrType());

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get arena pointer
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Create result tensor with header using arena
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* typed_result_tensor_ptr = builder.CreateCall(alloc_tensor_func, {arena_ptr}, "arith_tensor");

    // Copy dimensions from tensor1 to result
    llvm::Value* tensor1_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor1_ptr, 0);
    llvm::Value* tensor1_dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), tensor1_dims_field_ptr);

    llvm::Value* result_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 0);
    ctx_.builder().CreateStore(tensor1_dims_ptr, result_dims_field_ptr);

    // Copy num_dimensions
    llvm::Value* tensor1_num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor1_ptr, 1);
    llvm::Value* num_dims = ctx_.builder().CreateLoad(ctx_.int64Type(), tensor1_num_dims_field_ptr);

    llvm::Value* result_num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 1);
    ctx_.builder().CreateStore(num_dims, result_num_dims_field_ptr);

    // Get total elements
    llvm::Value* tensor1_total_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor1_ptr, 3);
    llvm::Value* total_elements = ctx_.builder().CreateLoad(ctx_.int64Type(), tensor1_total_elements_field_ptr);

    llvm::Value* result_total_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 3);
    ctx_.builder().CreateStore(total_elements, result_total_elements_field_ptr);

    // Allocate result elements array using arena
    llvm::Value* elements_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_elements_ptr = builder.CreateCall(arena_alloc, {arena_ptr, elements_size}, "arith_elems");
    llvm::Value* typed_result_elements_ptr = builder.CreatePointerCast(result_elements_ptr, ctx_.ptrType());

    llvm::Value* result_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 2);
    ctx_.builder().CreateStore(typed_result_elements_ptr, result_elements_field_ptr);

    // Get elements arrays
    llvm::Value* tensor1_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor1_ptr, 2);
    llvm::Value* tensor1_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), tensor1_elements_field_ptr);
    llvm::Value* typed_tensor1_elements_ptr = ctx_.builder().CreatePointerCast(tensor1_elements_ptr, ctx_.ptrType());

    llvm::Value* tensor2_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor2_ptr, 2);
    llvm::Value* tensor2_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), tensor2_elements_field_ptr);
    llvm::Value* typed_tensor2_elements_ptr = ctx_.builder().CreatePointerCast(tensor2_elements_ptr, ctx_.ptrType());

    // Create loop to iterate over all elements
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "tensor_arith_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "tensor_arith_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "tensor_arith_exit", current_func);

    // Initialize loop counter
    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "tensor_arith_i");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    ctx_.builder().CreateBr(loop_cond);

    // Loop condition: i < total_elements
    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cond = ctx_.builder().CreateICmpULT(i, total_elements);
    ctx_.builder().CreateCondBr(cond, loop_body, loop_exit);

    // Loop body: perform operation on element i
    ctx_.builder().SetInsertPoint(loop_body);

    // Get pointers to elements at index i
    llvm::Value* elem1_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_tensor1_elements_ptr, i);
    llvm::Value* elem2_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_tensor2_elements_ptr, i);
    llvm::Value* result_elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_result_elements_ptr, i);

    // Load elements as doubles
    llvm::Value* elem1 = ctx_.builder().CreateLoad(ctx_.doubleType(), elem1_ptr);
    llvm::Value* elem2 = ctx_.builder().CreateLoad(ctx_.doubleType(), elem2_ptr);

    // Perform operation
    llvm::Value* result_elem = nullptr;
    if (operation == "add") {
        result_elem = ctx_.builder().CreateFAdd(elem1, elem2);
    } else if (operation == "sub") {
        result_elem = ctx_.builder().CreateFSub(elem1, elem2);
    } else if (operation == "mul") {
        result_elem = ctx_.builder().CreateFMul(elem1, elem2);
    } else if (operation == "div") {
        result_elem = ctx_.builder().CreateFDiv(elem1, elem2);
    } else if (operation == "pow") {
        llvm::Function* pow_fn = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::pow, {ctx_.doubleType()});
        result_elem = ctx_.builder().CreateCall(pow_fn, {elem1, elem2});
    } else if (operation == "max") {
        llvm::Function* max_fn = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::maxnum, {ctx_.doubleType()});
        result_elem = ctx_.builder().CreateCall(max_fn, {elem1, elem2});
    } else if (operation == "min") {
        llvm::Function* min_fn = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::minnum, {ctx_.doubleType()});
        result_elem = ctx_.builder().CreateCall(min_fn, {elem1, elem2});
    }

    // Store result
    if (result_elem) {
        ctx_.builder().CreateStore(result_elem, result_elem_ptr);
    }

    // Increment counter
    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, counter);
    ctx_.builder().CreateBr(loop_cond);

    // Loop exit - pack tensor result as consolidated HEAP_PTR
    ctx_.builder().SetInsertPoint(loop_exit);
    // Propagate element dtype from the operands (binary promotion).
    emitDtypePropagateBinary(typed_result_tensor_ptr, tensor1_ptr, tensor2_ptr);
    return tagged_.packHeapPtr(typed_result_tensor_ptr);
}

// ===== SIMD-ACCELERATED TENSOR ARITHMETIC =====
// Processes SIMD_WIDTH doubles at a time using vector operations
// Width is auto-detected: 2 (NEON/SSE2), 4 (AVX), or 8 (AVX-512)
llvm::Value* TensorCodegen::rawTensorArithmeticSIMD(llvm::Value* arg1, llvm::Value* arg2, const std::string& operation) {
    auto& builder = ctx_.builder();
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();

    // Get raw int64 values (tensor pointers)
    llvm::Value* tensor1_int = tagged_.unpackInt64(arg1);
    llvm::Value* tensor2_int = tagged_.unpackInt64(arg2);

    llvm::Value* tensor1_ptr = builder.CreateIntToPtr(tensor1_int, ctx_.ptrType());
    llvm::Value* tensor2_ptr = builder.CreateIntToPtr(tensor2_int, ctx_.ptrType());

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Load shapes from both tensors for broadcast check
    llvm::Value* t1_dims_field = builder.CreateStructGEP(tensor_type, tensor1_ptr, 0);
    llvm::Value* t1_dims_ptr = builder.CreateLoad(ctx_.ptrType(), t1_dims_field);
    llvm::Value* t1_ndim_field = builder.CreateStructGEP(tensor_type, tensor1_ptr, 1);
    llvm::Value* t1_ndim = builder.CreateLoad(ctx_.int64Type(), t1_ndim_field);
    llvm::Value* t2_dims_field = builder.CreateStructGEP(tensor_type, tensor2_ptr, 0);
    llvm::Value* t2_dims_ptr = builder.CreateLoad(ctx_.ptrType(), t2_dims_field);
    llvm::Value* t2_ndim_field = builder.CreateStructGEP(tensor_type, tensor2_ptr, 1);
    llvm::Value* t2_ndim = builder.CreateLoad(ctx_.int64Type(), t2_ndim_field);

    // Runtime shape comparison: if shapes identical, use fast SIMD path
    auto* shapes_eq_ft = llvm::FunctionType::get(ctx_.int64Type(),
        {ctx_.ptrType(), ctx_.int64Type(), ctx_.ptrType(), ctx_.int64Type()}, false);
    llvm::Function* shapes_eq_fn = ctx_.module().getFunction("eshkol_shapes_equal");
    if (!shapes_eq_fn) {
        shapes_eq_fn = llvm::Function::Create(shapes_eq_ft,
            llvm::Function::ExternalLinkage, "eshkol_shapes_equal", &ctx_.module());
    }
    llvm::Value* shapes_match = builder.CreateCall(shapes_eq_fn,
        {t1_dims_ptr, t1_ndim, t2_dims_ptr, t2_ndim}, "shapes_match");
    llvm::Value* is_same_shape = builder.CreateICmpNE(shapes_match,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));

    // Shared result storage - declared before branch so both paths can write to it
    llvm::Value* shared_result = builder.CreateAlloca(ctx_.taggedValueType(), nullptr, "arith_shared_result");

    llvm::Function* current_top_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* fast_path = llvm::BasicBlock::Create(ctx_.context(), "arith_fast_path", current_top_func);
    llvm::BasicBlock* broadcast_path = llvm::BasicBlock::Create(ctx_.context(), "arith_broadcast_path", current_top_func);
    llvm::BasicBlock* arith_done = llvm::BasicBlock::Create(ctx_.context(), "arith_done", current_top_func);

    builder.CreateCondBr(is_same_shape, fast_path, broadcast_path);

    // ===== BROADCAST PATH: shapes differ, use runtime broadcast =====
    builder.SetInsertPoint(broadcast_path);
    {
        llvm::Value* bcast_arena = builder.CreateLoad(
            llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
        llvm::Function* arena_alloc_fn = mem_.getArenaAllocate();

        // Get elements from both tensors
        llvm::Value* t1_elems_field = builder.CreateStructGEP(tensor_type, tensor1_ptr, 2);
        llvm::Value* t1_elems = builder.CreateLoad(ctx_.ptrType(), t1_elems_field);
        llvm::Value* t2_elems_field = builder.CreateStructGEP(tensor_type, tensor2_ptr, 2);
        llvm::Value* t2_elems = builder.CreateLoad(ctx_.ptrType(), t2_elems_field);

        // Allocate output dims array (max 16 dims)
        llvm::Value* out_dims_buf = builder.CreateCall(arena_alloc_fn,
            {bcast_arena, llvm::ConstantInt::get(ctx_.int64Type(), 16 * sizeof(int64_t))}, "bcast_out_dims");
        // Allocate space for out_ndim and out_total (stack alloca)
        llvm::Value* out_ndim_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, "bcast_ndim");
        llvm::Value* out_total_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, "bcast_total");

        // Compute upper bound for output allocation: sum of totals * max expansion
        llvm::Value* t1_total_field = builder.CreateStructGEP(tensor_type, tensor1_ptr, 3);
        llvm::Value* t1_total = builder.CreateLoad(ctx_.int64Type(), t1_total_field);
        llvm::Value* t2_total_field = builder.CreateStructGEP(tensor_type, tensor2_ptr, 3);
        llvm::Value* t2_total = builder.CreateLoad(ctx_.int64Type(), t2_total_field);
        llvm::Value* max_alloc = builder.CreateMul(t1_total, t2_total);
        llvm::Value* cap = llvm::ConstantInt::get(ctx_.int64Type(), 16 * 1024 * 1024);
        llvm::Value* use_cap = builder.CreateICmpUGT(max_alloc, cap);
        llvm::Value* safe_alloc = builder.CreateSelect(use_cap, cap, max_alloc);
        llvm::Value* alloc_bytes = builder.CreateMul(safe_alloc,
            llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
        llvm::Value* out_data_buf = builder.CreateCall(arena_alloc_fn,
            {bcast_arena, alloc_bytes}, "bcast_out_data");

        // Map operation string to int: 0=add, 1=sub, 2=mul, 3=div
        int64_t op_code = 0;
        if (operation == "add") op_code = 0;
        else if (operation == "sub") op_code = 1;
        else if (operation == "mul") op_code = 2;
        else if (operation == "div") op_code = 3;

        // Call runtime: eshkol_broadcast_elementwise_f64
        auto* bcast_ft = llvm::FunctionType::get(ctx_.int64Type(),
            {ctx_.int64Type(),
             ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type(),
             ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type(),
             ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()}, false);
        llvm::Function* bcast_fn = ctx_.module().getFunction("eshkol_broadcast_elementwise_f64");
        if (!bcast_fn) {
            bcast_fn = llvm::Function::Create(bcast_ft,
                llvm::Function::ExternalLinkage, "eshkol_broadcast_elementwise_f64", &ctx_.module());
        }
        builder.CreateCall(bcast_fn,
            {llvm::ConstantInt::get(ctx_.int64Type(), op_code),
             t1_elems, t1_dims_ptr, t1_ndim,
             t2_elems, t2_dims_ptr, t2_ndim,
             out_data_buf, out_dims_buf, out_ndim_alloca, out_total_alloca});

        // Load actual ndim and total
        llvm::Value* bcast_ndim = builder.CreateLoad(ctx_.int64Type(), out_ndim_alloca);
        llvm::Value* bcast_total = builder.CreateLoad(ctx_.int64Type(), out_total_alloca);

        // Build result tensor
        llvm::Function* alloc_tensor_fn = mem_.getArenaAllocateTensorWithHeader();
        llvm::Value* bcast_tensor = builder.CreateCall(alloc_tensor_fn, {bcast_arena}, "bcast_tensor");

        builder.CreateStore(out_dims_buf,
            builder.CreateStructGEP(tensor_type, bcast_tensor, 0));
        builder.CreateStore(bcast_ndim,
            builder.CreateStructGEP(tensor_type, bcast_tensor, 1));
        builder.CreateStore(out_data_buf,
            builder.CreateStructGEP(tensor_type, bcast_tensor, 2));
        builder.CreateStore(bcast_total,
            builder.CreateStructGEP(tensor_type, bcast_tensor, 3));

        builder.CreateStore(tagged_.packHeapPtr(bcast_tensor), shared_result);
        builder.CreateBr(arith_done);
    }

    // ===== FAST PATH: shapes match, use SIMD =====
    builder.SetInsertPoint(fast_path);

    // Get arena pointer
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Create result tensor with header using arena
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* typed_result_tensor_ptr = builder.CreateCall(alloc_tensor_func, {arena_ptr}, "simd_tensor");

    // Copy dimensions from tensor1 to result
    llvm::Value* tensor1_dims_field_ptr = builder.CreateStructGEP(tensor_type, tensor1_ptr, 0);
    llvm::Value* tensor1_dims_ptr = builder.CreateLoad(ctx_.ptrType(), tensor1_dims_field_ptr);

    llvm::Value* result_dims_field_ptr = builder.CreateStructGEP(tensor_type, typed_result_tensor_ptr, 0);
    builder.CreateStore(tensor1_dims_ptr, result_dims_field_ptr);

    // Copy num_dimensions
    llvm::Value* tensor1_num_dims_field_ptr = builder.CreateStructGEP(tensor_type, tensor1_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), tensor1_num_dims_field_ptr);

    llvm::Value* result_num_dims_field_ptr = builder.CreateStructGEP(tensor_type, typed_result_tensor_ptr, 1);
    builder.CreateStore(num_dims, result_num_dims_field_ptr);

    // Get total elements
    llvm::Value* tensor1_total_elements_field_ptr = builder.CreateStructGEP(tensor_type, tensor1_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), tensor1_total_elements_field_ptr);

    llvm::Value* result_total_elements_field_ptr = builder.CreateStructGEP(tensor_type, typed_result_tensor_ptr, 3);
    builder.CreateStore(total_elements, result_total_elements_field_ptr);

    // Allocate result elements array using arena
    llvm::Value* elements_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_elements_ptr = builder.CreateCall(arena_alloc, {arena_ptr, elements_size}, "simd_elems");
    llvm::Value* typed_result_elements_ptr = builder.CreatePointerCast(result_elements_ptr, ctx_.ptrType());

    llvm::Value* result_elements_field_ptr = builder.CreateStructGEP(tensor_type, typed_result_tensor_ptr, 2);
    builder.CreateStore(typed_result_elements_ptr, result_elements_field_ptr);

    // Get elements arrays
    llvm::Value* tensor1_elements_field_ptr = builder.CreateStructGEP(tensor_type, tensor1_ptr, 2);
    llvm::Value* tensor1_elements_ptr = builder.CreateLoad(ctx_.ptrType(), tensor1_elements_field_ptr);
    llvm::Value* typed_tensor1_elements_ptr = builder.CreatePointerCast(tensor1_elements_ptr, ctx_.ptrType());

    llvm::Value* tensor2_elements_field_ptr = builder.CreateStructGEP(tensor_type, tensor2_ptr, 2);
    llvm::Value* tensor2_elements_ptr = builder.CreateLoad(ctx_.ptrType(), tensor2_elements_field_ptr);
    llvm::Value* typed_tensor2_elements_ptr = builder.CreatePointerCast(tensor2_elements_ptr, ctx_.ptrType());

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Basic blocks for control flow
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "scalar_body", current_func);
    llvm::BasicBlock* final_exit = llvm::BasicBlock::Create(ctx_.context(), "final_exit", current_func);

    uint32_t ad_op_type = 0;
    if (operation == "add") {
        ad_op_type = 2;  // AD_NODE_ADD
    } else if (operation == "sub") {
        ad_op_type = 3;  // AD_NODE_SUB
    } else if (operation == "mul") {
        ad_op_type = 4;  // AD_NODE_MUL
    } else if (operation == "div") {
        ad_op_type = 5;  // AD_NODE_DIV
    } else if (operation == "pow") {
        ad_op_type = 10; // AD_NODE_POW
    } else if (operation == "max") {
        ad_op_type = 44; // AD_NODE_MAX
    } else if (operation == "min") {
        ad_op_type = 45; // AD_NODE_MIN
    }

    if (autodiff_ && ad_op_type != 0) {
        llvm::Value* in_ad_mode = builder.CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
        llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), "arith_ad_path", current_func);
        llvm::BasicBlock* numeric_path = llvm::BasicBlock::Create(ctx_.context(), "arith_numeric_path", current_func);
        builder.CreateCondBr(in_ad_mode, ad_path, numeric_path);

        builder.SetInsertPoint(ad_path);
        llvm::Value* ad_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "arith_ad_i");
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ad_counter);
        llvm::BasicBlock* ad_cond = llvm::BasicBlock::Create(ctx_.context(), "arith_ad_cond", current_func);
        llvm::BasicBlock* ad_body = llvm::BasicBlock::Create(ctx_.context(), "arith_ad_body", current_func);
        llvm::BasicBlock* ad_exit = llvm::BasicBlock::Create(ctx_.context(), "arith_ad_exit", current_func);
        builder.CreateBr(ad_cond);

        builder.SetInsertPoint(ad_cond);
        llvm::Value* ad_i = builder.CreateLoad(ctx_.int64Type(), ad_counter);
        llvm::Value* ad_has_elem = builder.CreateICmpULT(ad_i, total_elements);
        builder.CreateCondBr(ad_has_elem, ad_body, ad_exit);

        builder.SetInsertPoint(ad_body);
        llvm::Value* elem1_ptr = builder.CreateGEP(ctx_.int64Type(), typed_tensor1_elements_ptr, ad_i);
        llvm::Value* elem2_ptr = builder.CreateGEP(ctx_.int64Type(), typed_tensor2_elements_ptr, ad_i);
        llvm::Value* result_elem_ptr = builder.CreateGEP(ctx_.int64Type(), typed_result_elements_ptr, ad_i);
        llvm::Value* elem1_bits = builder.CreateLoad(ctx_.int64Type(), elem1_ptr);
        llvm::Value* elem2_bits = builder.CreateLoad(ctx_.int64Type(), elem2_ptr);
        llvm::Value* elem1_node = adNodeFromTensorElementBits(elem1_bits, "arith_ad_lhs");
        llvm::Value* elem2_node = adNodeFromTensorElementBits(elem2_bits, "arith_ad_rhs");
        llvm::Value* result_node = autodiff_->recordADNodeBinary(ad_op_type, elem1_node, elem2_node);
        llvm::Value* result_bits = builder.CreatePtrToInt(result_node, ctx_.int64Type());
        builder.CreateStore(result_bits, result_elem_ptr);

        llvm::Value* next_ad_i = builder.CreateAdd(ad_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        builder.CreateStore(next_ad_i, ad_counter);
        builder.CreateBr(ad_cond);

        builder.SetInsertPoint(ad_exit);
        builder.CreateBr(final_exit);

        builder.SetInsertPoint(numeric_path);
    }

    // Scalar loop counter - always needed
    llvm::Value* scalar_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "scalar_i");

    // ===== SCALAR-ONLY PATH (when SIMD_WIDTH == 1 or no SIMD available) =====
    if (SIMD_WIDTH == 1 || vec_type == nullptr) {
        // Pure scalar fallback - process all elements one at a time
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), scalar_counter);
        builder.CreateBr(scalar_cond);
    } else {
        // ===== SIMD VECTOR LOOP =====
        // Calculate number of full SIMD vectors and remaining scalar elements
        llvm::Value* simd_count = builder.CreateUDiv(total_elements,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        llvm::Value* simd_elements = builder.CreateMul(simd_count,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

        llvm::BasicBlock* simd_cond = llvm::BasicBlock::Create(ctx_.context(), "simd_cond", current_func);
        llvm::BasicBlock* simd_body = llvm::BasicBlock::Create(ctx_.context(), "simd_body", current_func);
        llvm::BasicBlock* simd_exit = llvm::BasicBlock::Create(ctx_.context(), "simd_exit", current_func);

        // SIMD loop counter (counts in steps of SIMD_WIDTH)
        llvm::Value* simd_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "simd_i");
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), simd_counter);
        builder.CreateBr(simd_cond);

        // SIMD loop condition: i < simd_elements
        builder.SetInsertPoint(simd_cond);
        llvm::Value* simd_i = builder.CreateLoad(ctx_.int64Type(), simd_counter);
        llvm::Value* simd_cmp = builder.CreateICmpULT(simd_i, simd_elements);
        builder.CreateCondBr(simd_cmp, simd_body, simd_exit);

        // SIMD loop body: process SIMD_WIDTH doubles at once
        builder.SetInsertPoint(simd_body);

        // Get pointers to elements at index i (for vector load/store)
        llvm::Value* vec1_ptr = builder.CreateGEP(ctx_.doubleType(), typed_tensor1_elements_ptr, simd_i);
        llvm::Value* vec2_ptr = builder.CreateGEP(ctx_.doubleType(), typed_tensor2_elements_ptr, simd_i);
        llvm::Value* result_vec_ptr = builder.CreateGEP(ctx_.doubleType(), typed_result_elements_ptr, simd_i);

        // Load SIMD_WIDTH doubles as vectors (unaligned load for safety)
        llvm::Value* vec1 = builder.CreateAlignedLoad(vec_type, vec1_ptr, llvm::MaybeAlign(8), "vec1");
        llvm::Value* vec2 = builder.CreateAlignedLoad(vec_type, vec2_ptr, llvm::MaybeAlign(8), "vec2");

        // Perform vectorized operation
        llvm::Value* result_vec = nullptr;
        if (operation == "add") {
            result_vec = builder.CreateFAdd(vec1, vec2, "vadd");
        } else if (operation == "sub") {
            result_vec = builder.CreateFSub(vec1, vec2, "vsub");
        } else if (operation == "mul") {
            result_vec = builder.CreateFMul(vec1, vec2, "vmul");
        } else if (operation == "div") {
            result_vec = builder.CreateFDiv(vec1, vec2, "vdiv");
        } else if (operation == "pow") {
            llvm::Function* pow_vec = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::pow, {vec_type});
            result_vec = builder.CreateCall(pow_vec, {vec1, vec2}, "vpow");
        } else if (operation == "max") {
            llvm::Function* max_vec = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::maxnum, {vec_type});
            result_vec = builder.CreateCall(max_vec, {vec1, vec2}, "vmax");
        } else if (operation == "min") {
            llvm::Function* min_vec = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::minnum, {vec_type});
            result_vec = builder.CreateCall(min_vec, {vec1, vec2}, "vmin");
        }

        // Store result vector
        if (result_vec) {
            builder.CreateAlignedStore(result_vec, result_vec_ptr, llvm::MaybeAlign(8));
        }

        // Increment counter by SIMD_WIDTH
        llvm::Value* next_simd_i = builder.CreateAdd(simd_i,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        builder.CreateStore(next_simd_i, simd_counter);
        auto* simdBackEdge = builder.CreateBr(simd_cond);
        attachLoopMetadata(simdBackEdge, true, SIMD_WIDTH, false, 0);

        // ===== SCALAR TAIL LOOP =====
        // Handle remaining elements (total_elements % SIMD_WIDTH)
        builder.SetInsertPoint(simd_exit);
        builder.CreateStore(simd_elements, scalar_counter);  // Start from where SIMD ended
        builder.CreateBr(scalar_cond);
    }

    // Scalar loop condition: i < total_elements
    builder.SetInsertPoint(scalar_cond);
    llvm::Value* scalar_i = builder.CreateLoad(ctx_.int64Type(), scalar_counter);
    llvm::Value* scalar_cmp = builder.CreateICmpULT(scalar_i, total_elements);
    builder.CreateCondBr(scalar_cmp, scalar_body, final_exit);

    // Scalar loop body: process 1 double at a time
    builder.SetInsertPoint(scalar_body);

    llvm::Value* elem1_ptr = builder.CreateGEP(ctx_.doubleType(), typed_tensor1_elements_ptr, scalar_i);
    llvm::Value* elem2_ptr = builder.CreateGEP(ctx_.doubleType(), typed_tensor2_elements_ptr, scalar_i);
    llvm::Value* result_elem_ptr = builder.CreateGEP(ctx_.doubleType(), typed_result_elements_ptr, scalar_i);

    llvm::Value* elem1 = builder.CreateLoad(ctx_.doubleType(), elem1_ptr);
    llvm::Value* elem2 = builder.CreateLoad(ctx_.doubleType(), elem2_ptr);

    llvm::Value* result_elem = nullptr;
    if (operation == "add") {
        result_elem = builder.CreateFAdd(elem1, elem2);
    } else if (operation == "sub") {
        result_elem = builder.CreateFSub(elem1, elem2);
    } else if (operation == "mul") {
        result_elem = builder.CreateFMul(elem1, elem2);
    } else if (operation == "div") {
        result_elem = builder.CreateFDiv(elem1, elem2);
    } else if (operation == "pow") {
        llvm::Function* pow_fn = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::pow, {ctx_.doubleType()});
        result_elem = builder.CreateCall(pow_fn, {elem1, elem2});
    } else if (operation == "max") {
        llvm::Function* max_fn = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::maxnum, {ctx_.doubleType()});
        result_elem = builder.CreateCall(max_fn, {elem1, elem2});
    } else if (operation == "min") {
        llvm::Function* min_fn = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::minnum, {ctx_.doubleType()});
        result_elem = builder.CreateCall(min_fn, {elem1, elem2});
    }

    if (result_elem) {
        builder.CreateStore(result_elem, result_elem_ptr);
    }

    llvm::Value* next_scalar_i = builder.CreateAdd(scalar_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_scalar_i, scalar_counter);
    auto* scalarBackEdge = builder.CreateBr(scalar_cond);
    attachLoopMetadata(scalarBackEdge, false, 0, true, 4);

    // Final exit from SIMD fast path - store result and branch to merge
    builder.SetInsertPoint(final_exit);
    emitDtypePropagateBinary(typed_result_tensor_ptr, tensor1_ptr, tensor2_ptr);
    builder.CreateStore(tagged_.packHeapPtr(typed_result_tensor_ptr), shared_result);
    builder.CreateBr(arith_done);

    // ===== MERGE: return result from whichever path was taken =====
    builder.SetInsertPoint(arith_done);
    return builder.CreateLoad(ctx_.taggedValueType(), shared_result);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
