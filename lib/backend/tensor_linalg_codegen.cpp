/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen — Linear Algebra (Phase 4.4). Extracted from
 * tensor_codegen.cpp during the v1.2 mechanical split. Implements LU
 * decomposition, determinant, inverse, linear solve, Cholesky, QR,
 * SVD, and einsum.
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-linalg-extract baseline.
 */
#include <eshkol/backend/tensor_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

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

namespace eshkol {

// ===== LINEAR ALGEBRA (Phase 4.4) =====

// Helper: declare or get an extern "C" runtime function
static llvm::Function* getOrDeclareRuntimeFunc(
    llvm::Module& mod, llvm::LLVMContext& ctx,
    const char* name, llvm::FunctionType* ft) {
    llvm::Function* f = mod.getFunction(name);
    if (!f) {
        f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, &mod);
    }
    return f;
}

// Helper: emit null check after arena allocation — exits with OOM message if null
static void emitArenaAllocNullCheck(llvm::IRBuilder<>& builder, CodegenContext& ctx,
                                     llvm::Value* ptr, const char* msg) {
    llvm::Function* cur_fn = builder.GetInsertBlock()->getParent();
    llvm::Value* is_null = builder.CreateICmpEQ(ptr,
        llvm::ConstantPointerNull::get(builder.getPtrTy()));
    llvm::BasicBlock* null_bb = llvm::BasicBlock::Create(ctx.context(), "oom", cur_fn);
    llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx.context(), "alloc_ok", cur_fn);
    builder.CreateCondBr(is_null, null_bb, ok_bb);

    builder.SetInsertPoint(null_bb);
    llvm::Function* pf = ctx.lookupFunction("printf");
    llvm::Function* ef = ctx.lookupFunction("exit");
    if (pf && ef) {
        llvm::Value* fmt = builder.CreateGlobalString(msg);
        builder.CreateCall(pf, {fmt});
        builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
    }
    builder.CreateUnreachable();
    builder.SetInsertPoint(ok_bb);
}

llvm::Value* TensorCodegen::tensorLU(const eshkol_operations_t* op) {
    // tensor-lu: (tensor-lu A) -> returns list (LU-matrix, pivot-vector, sign)
    // LU decomposition with partial pivoting
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-lu requires 1 argument: square matrix");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Value* a_tagged = codegenAST(&op->call_op.variables[0]);
    if (!a_tagged) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* a_ptr = tagged_.unpackPtr(a_tagged);

    // Get dimensions - must be 2D square
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, a_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* n = builder.CreateLoad(ctx_.int64Type(), dims_ptr);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, a_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);

    // Allocate working copy of A as doubles (n*n doubles)
    llvm::Value* nn = builder.CreateMul(n, n);
    llvm::Value* byte_size = builder.CreateMul(nn, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
    llvm::Function* alloc_fn = mem_.getArenaAllocate();
    llvm::Value* lu_data = builder.CreateCall(alloc_fn, {arena_ptr, byte_size}, "lu_data");
    emitArenaAllocNullCheck(builder, ctx_, lu_data, "Error: out of memory in tensor-lu\n");

    // Copy source tensor elements (int64 bitpatterns) to double array
    // Loop: for i = 0..n*n: lu_data[i] = bitcast(src[i])
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "lu_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "lu_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "lu_copy_done", current_func);

    llvm::Value* copy_i = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), copy_i);
    builder.CreateCondBr(builder.CreateICmpULT(ci, nn), copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* src_p = builder.CreateGEP(ctx_.int64Type(), src_elems, ci);
    llvm::Value* bits = builder.CreateLoad(ctx_.int64Type(), src_p);
    llvm::Value* dval = builder.CreateBitCast(bits, ctx_.doubleType());
    llvm::Value* dst_p = builder.CreateGEP(ctx_.doubleType(), lu_data, ci);
    builder.CreateStore(dval, dst_p);
    builder.CreateStore(builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);

    // Allocate pivot array (n int64s)
    llvm::Value* piv_size = builder.CreateMul(n, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* piv_data = builder.CreateCall(alloc_fn, {arena_ptr, piv_size}, "lu_piv");
    emitArenaAllocNullCheck(builder, ctx_, piv_data, "Error: out of memory in tensor-lu (pivot)\n");

    // Call runtime LU decomposition
    llvm::FunctionType* lu_ft = llvm::FunctionType::get(
        ctx_.int64Type(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()},
        false);
    llvm::Function* lu_fn = getOrDeclareRuntimeFunc(ctx_.module(), ctx_.context(), "eshkol_lu_decompose", lu_ft);
    llvm::Value* sign = builder.CreateCall(lu_fn, {lu_data, piv_data, n}, "lu_sign");

    // Create result tensor from LU data (copy doubles back to int64 bitpatterns)
    std::vector<llvm::Value*> dims = {n, n};
    llvm::Value* result_ptr = createTensorWithDims(dims);
    if (!result_ptr) return nullptr;

    llvm::Value* res_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    llvm::Value* res_elems = builder.CreateLoad(ctx_.ptrType(), res_elems_field);

    // Copy loop: double -> int64 bitcast
    llvm::BasicBlock* cp2_cond = llvm::BasicBlock::Create(ctx_.context(), "lu_cp2_cond", current_func);
    llvm::BasicBlock* cp2_body = llvm::BasicBlock::Create(ctx_.context(), "lu_cp2_body", current_func);
    llvm::BasicBlock* cp2_done = llvm::BasicBlock::Create(ctx_.context(), "lu_cp2_done", current_func);

    llvm::Value* cp2_i = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), cp2_i);
    builder.CreateBr(cp2_cond);

    builder.SetInsertPoint(cp2_cond);
    llvm::Value* ci2 = builder.CreateLoad(ctx_.int64Type(), cp2_i);
    builder.CreateCondBr(builder.CreateICmpULT(ci2, nn), cp2_body, cp2_done);

    builder.SetInsertPoint(cp2_body);
    llvm::Value* lu_p = builder.CreateGEP(ctx_.doubleType(), lu_data, ci2);
    llvm::Value* lu_val = builder.CreateLoad(ctx_.doubleType(), lu_p);
    llvm::Value* lu_bits = builder.CreateBitCast(lu_val, ctx_.int64Type());
    llvm::Value* res_p = builder.CreateGEP(ctx_.int64Type(), res_elems, ci2);
    builder.CreateStore(lu_bits, res_p);
    builder.CreateStore(builder.CreateAdd(ci2, llvm::ConstantInt::get(ctx_.int64Type(), 1)), cp2_i);
    builder.CreateBr(cp2_cond);

    builder.SetInsertPoint(cp2_done);

    // === Create pivot result tensor (1D, n elements) ===
    std::vector<llvm::Value*> piv_dims = {n};
    llvm::Value* piv_result = createTensorWithDims(piv_dims);
    if (!piv_result) return nullptr;

    llvm::Value* piv_res_elems_field = builder.CreateStructGEP(tensor_type, piv_result, 2);
    llvm::Value* piv_res_elems = builder.CreateLoad(ctx_.ptrType(), piv_res_elems_field);

    // Copy pivot int64 indices to tensor as doubles (bitcast pattern)
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "lu_cpp_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "lu_cpp_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "lu_cpp_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, n), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* piv_src = builder.CreateGEP(ctx_.int64Type(), piv_data, i);
        llvm::Value* piv_val = builder.CreateLoad(ctx_.int64Type(), piv_src);
        // Convert pivot index (int64) to double, then bitcast to int64 for tensor storage
        llvm::Value* piv_dbl = builder.CreateSIToFP(piv_val, ctx_.doubleType());
        llvm::Value* piv_bits = builder.CreateBitCast(piv_dbl, ctx_.int64Type());
        llvm::Value* piv_dst = builder.CreateGEP(ctx_.int64Type(), piv_res_elems, i);
        builder.CreateStore(piv_bits, piv_dst);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    // === Build cons list (LU . (pivot . (sign . '()))) ===
    llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 1);

    // Cell 3: (sign . '())
    llvm::Value* cons3 = builder.CreateCall(mem_.getArenaAllocateConsWithHeader(), {arena_ptr});
    llvm::Value* sign_tagged = tagged_.packInt64(sign);
    llvm::Value* sign_alloca = builder.CreateAlloca(ctx_.taggedValueType());
    builder.CreateStore(sign_tagged, sign_alloca);
    builder.CreateCall(mem_.getTaggedConsSetTaggedValue(), {cons3, is_car, sign_alloca});
    builder.CreateCall(mem_.getTaggedConsSetNull(), {cons3, is_cdr});

    // Cell 2: (pivot . cons3)
    llvm::Value* cons2 = builder.CreateCall(mem_.getArenaAllocateConsWithHeader(), {arena_ptr});
    llvm::Value* piv_tagged = tagged_.packHeapPtr(piv_result);
    llvm::Value* piv_alloca = builder.CreateAlloca(ctx_.taggedValueType());
    builder.CreateStore(piv_tagged, piv_alloca);
    builder.CreateCall(mem_.getTaggedConsSetTaggedValue(), {cons2, is_car, piv_alloca});
    llvm::Value* cons3_int = builder.CreatePtrToInt(cons3, ctx_.int64Type());
    llvm::Value* cons_type_val = llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR);
    builder.CreateCall(mem_.getTaggedConsSetPtr(), {cons2, is_cdr, cons3_int, cons_type_val});

    // Cell 1: (LU . cons2)
    llvm::Value* cons1 = builder.CreateCall(mem_.getArenaAllocateConsWithHeader(), {arena_ptr});
    llvm::Value* lu_tagged = tagged_.packHeapPtr(result_ptr);
    llvm::Value* lu_alloca = builder.CreateAlloca(ctx_.taggedValueType());
    builder.CreateStore(lu_tagged, lu_alloca);
    builder.CreateCall(mem_.getTaggedConsSetTaggedValue(), {cons1, is_car, lu_alloca});
    llvm::Value* cons2_int = builder.CreatePtrToInt(cons2, ctx_.int64Type());
    builder.CreateCall(mem_.getTaggedConsSetPtr(), {cons1, is_cdr, cons2_int, cons_type_val});

    return tagged_.packHeapPtr(cons1);
}

llvm::Value* TensorCodegen::tensorDet(const eshkol_operations_t* op) {
    // tensor-det: (tensor-det A) -> scalar determinant
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-det requires 1 argument: square matrix");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Value* a_tagged = codegenAST(&op->call_op.variables[0]);
    if (!a_tagged) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* a_ptr = tagged_.unpackPtr(a_tagged);

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, a_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* n = builder.CreateLoad(ctx_.int64Type(), dims_ptr);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, a_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);

    // Allocate working copy + pivot array
    llvm::Value* nn = builder.CreateMul(n, n);
    llvm::Value* byte_size = builder.CreateMul(nn, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
    llvm::Function* alloc_fn = mem_.getArenaAllocate();
    llvm::Value* lu_data = builder.CreateCall(alloc_fn, {arena_ptr, byte_size}, "det_lu");
    emitArenaAllocNullCheck(builder, ctx_, lu_data, "Error: out of memory in tensor-det\n");
    llvm::Value* piv_size = builder.CreateMul(n, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* piv_data = builder.CreateCall(alloc_fn, {arena_ptr, piv_size}, "det_piv");
    emitArenaAllocNullCheck(builder, ctx_, piv_data, "Error: out of memory in tensor-det (pivot)\n");

    // Copy tensor elements to double array
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "det_cp_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "det_cp_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "det_cp_done", current_func);

    llvm::Value* copy_i = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), copy_i);
    builder.CreateCondBr(builder.CreateICmpULT(ci, nn), copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* sp = builder.CreateGEP(ctx_.int64Type(), src_elems, ci);
    llvm::Value* bits = builder.CreateLoad(ctx_.int64Type(), sp);
    llvm::Value* dval = builder.CreateBitCast(bits, ctx_.doubleType());
    llvm::Value* dp = builder.CreateGEP(ctx_.doubleType(), lu_data, ci);
    builder.CreateStore(dval, dp);
    builder.CreateStore(builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);

    // Call LU decomposition
    llvm::FunctionType* lu_ft = llvm::FunctionType::get(
        ctx_.int64Type(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()},
        false);
    llvm::Function* lu_fn = getOrDeclareRuntimeFunc(ctx_.module(), ctx_.context(), "eshkol_lu_decompose", lu_ft);
    llvm::Value* sign = builder.CreateCall(lu_fn, {lu_data, piv_data, n}, "det_sign");

    // Call det_from_lu
    llvm::FunctionType* det_ft = llvm::FunctionType::get(
        ctx_.doubleType(),
        {ctx_.ptrType(), ctx_.int64Type(), ctx_.int64Type()},
        false);
    llvm::Function* det_fn = getOrDeclareRuntimeFunc(ctx_.module(), ctx_.context(), "eshkol_det_from_lu", det_ft);
    llvm::Value* det = builder.CreateCall(det_fn, {lu_data, n, sign}, "det_val");

    return tagged_.packDouble(det);
}

llvm::Value* TensorCodegen::tensorInverse(const eshkol_operations_t* op) {
    // tensor-inverse: (tensor-inverse A) -> A^{-1}
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-inverse requires 1 argument: square matrix");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Value* a_tagged = codegenAST(&op->call_op.variables[0]);
    if (!a_tagged) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* a_ptr = tagged_.unpackPtr(a_tagged);

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, a_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* n = builder.CreateLoad(ctx_.int64Type(), dims_ptr);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, a_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);

    llvm::Value* nn = builder.CreateMul(n, n);
    llvm::Value* byte_size = builder.CreateMul(nn, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
    llvm::Function* alloc_fn = mem_.getArenaAllocate();
    llvm::Value* lu_data = builder.CreateCall(alloc_fn, {arena_ptr, byte_size}, "inv_lu");
    emitArenaAllocNullCheck(builder, ctx_, lu_data, "Error: out of memory in tensor-inverse\n");
    llvm::Value* piv_size = builder.CreateMul(n, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* piv_data = builder.CreateCall(alloc_fn, {arena_ptr, piv_size}, "inv_piv");
    emitArenaAllocNullCheck(builder, ctx_, piv_data, "Error: out of memory in tensor-inverse (pivot)\n");
    llvm::Value* inv_data = builder.CreateCall(alloc_fn, {arena_ptr, byte_size}, "inv_data");
    emitArenaAllocNullCheck(builder, ctx_, inv_data, "Error: out of memory in tensor-inverse (result)\n");

    // Copy tensor elements to double array
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "inv_cp_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "inv_cp_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "inv_cp_done", current_func);

    llvm::Value* copy_i = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), copy_i);
    builder.CreateCondBr(builder.CreateICmpULT(ci, nn), copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* sp = builder.CreateGEP(ctx_.int64Type(), src_elems, ci);
    llvm::Value* bits = builder.CreateLoad(ctx_.int64Type(), sp);
    llvm::Value* dval = builder.CreateBitCast(bits, ctx_.doubleType());
    llvm::Value* dp = builder.CreateGEP(ctx_.doubleType(), lu_data, ci);
    builder.CreateStore(dval, dp);
    builder.CreateStore(builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);

    // LU decompose
    llvm::FunctionType* lu_ft = llvm::FunctionType::get(
        ctx_.int64Type(), {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
    llvm::Function* lu_fn = getOrDeclareRuntimeFunc(ctx_.module(), ctx_.context(), "eshkol_lu_decompose", lu_ft);
    builder.CreateCall(lu_fn, {lu_data, piv_data, n});

    // Compute inverse
    llvm::FunctionType* inv_ft = llvm::FunctionType::get(
        ctx_.voidType(), {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
    llvm::Function* inv_fn = getOrDeclareRuntimeFunc(ctx_.module(), ctx_.context(), "eshkol_lu_inverse", inv_ft);
    builder.CreateCall(inv_fn, {lu_data, piv_data, inv_data, n});

    // Create result tensor and copy doubles back as int64 bitpatterns
    std::vector<llvm::Value*> dims = {n, n};
    llvm::Value* result_ptr = createTensorWithDims(dims);
    if (!result_ptr) return nullptr;

    llvm::Value* res_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    llvm::Value* res_elems = builder.CreateLoad(ctx_.ptrType(), res_elems_field);

    llvm::BasicBlock* cp2_cond = llvm::BasicBlock::Create(ctx_.context(), "inv_cp2_cond", current_func);
    llvm::BasicBlock* cp2_body = llvm::BasicBlock::Create(ctx_.context(), "inv_cp2_body", current_func);
    llvm::BasicBlock* cp2_done = llvm::BasicBlock::Create(ctx_.context(), "inv_cp2_done", current_func);

    llvm::Value* cp2_i = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), cp2_i);
    builder.CreateBr(cp2_cond);

    builder.SetInsertPoint(cp2_cond);
    llvm::Value* ci2 = builder.CreateLoad(ctx_.int64Type(), cp2_i);
    builder.CreateCondBr(builder.CreateICmpULT(ci2, nn), cp2_body, cp2_done);

    builder.SetInsertPoint(cp2_body);
    llvm::Value* inv_p = builder.CreateGEP(ctx_.doubleType(), inv_data, ci2);
    llvm::Value* inv_val = builder.CreateLoad(ctx_.doubleType(), inv_p);
    llvm::Value* inv_bits = builder.CreateBitCast(inv_val, ctx_.int64Type());
    llvm::Value* res_p = builder.CreateGEP(ctx_.int64Type(), res_elems, ci2);
    builder.CreateStore(inv_bits, res_p);
    builder.CreateStore(builder.CreateAdd(ci2, llvm::ConstantInt::get(ctx_.int64Type(), 1)), cp2_i);
    builder.CreateBr(cp2_cond);

    builder.SetInsertPoint(cp2_done);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSolve(const eshkol_operations_t* op) {
    // tensor-solve: (tensor-solve A b) -> x where Ax = b
    if (op->call_op.num_vars != 2) {
        eshkol_error("tensor-solve requires 2 arguments: matrix A, vector b");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Value* a_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* b_tagged = codegenAST(&op->call_op.variables[1]);
    if (!a_tagged || !b_tagged) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* a_ptr = tagged_.unpackPtr(a_tagged);
    llvm::Value* b_ptr = tagged_.unpackPtr(b_tagged);

    llvm::Value* a_dims_field = builder.CreateStructGEP(tensor_type, a_ptr, 0);
    llvm::Value* a_dims_ptr = builder.CreateLoad(ctx_.ptrType(), a_dims_field);
    llvm::Value* n = builder.CreateLoad(ctx_.int64Type(), a_dims_ptr);

    llvm::Value* a_elems_field = builder.CreateStructGEP(tensor_type, a_ptr, 2);
    llvm::Value* a_elems = builder.CreateLoad(ctx_.ptrType(), a_elems_field);
    llvm::Value* b_elems_field = builder.CreateStructGEP(tensor_type, b_ptr, 2);
    llvm::Value* b_elems = builder.CreateLoad(ctx_.ptrType(), b_elems_field);

    llvm::Value* nn = builder.CreateMul(n, n);
    llvm::Value* a_bytes = builder.CreateMul(nn, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* b_bytes = builder.CreateMul(n, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
    llvm::Function* alloc_fn = mem_.getArenaAllocate();
    llvm::Value* lu_data = builder.CreateCall(alloc_fn, {arena_ptr, a_bytes}, "solve_lu");
    emitArenaAllocNullCheck(builder, ctx_, lu_data, "Error: out of memory in tensor-solve\n");
    llvm::Value* piv_data = builder.CreateCall(alloc_fn, {arena_ptr, b_bytes}, "solve_piv");
    emitArenaAllocNullCheck(builder, ctx_, piv_data, "Error: out of memory in tensor-solve (pivot)\n");
    llvm::Value* b_data = builder.CreateCall(alloc_fn, {arena_ptr, b_bytes}, "solve_b");
    emitArenaAllocNullCheck(builder, ctx_, b_data, "Error: out of memory in tensor-solve (b)\n");

    // Copy A to lu_data, b to b_data
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Copy A
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "solve_cpa_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "solve_cpa_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "solve_cpa_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, nn), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* sp = builder.CreateGEP(ctx_.int64Type(), a_elems, i);
        llvm::Value* bits = builder.CreateLoad(ctx_.int64Type(), sp);
        llvm::Value* dval = builder.CreateBitCast(bits, ctx_.doubleType());
        llvm::Value* dp = builder.CreateGEP(ctx_.doubleType(), lu_data, i);
        builder.CreateStore(dval, dp);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    // Copy b
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "solve_cpb_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "solve_cpb_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "solve_cpb_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, n), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* sp = builder.CreateGEP(ctx_.int64Type(), b_elems, i);
        llvm::Value* bits = builder.CreateLoad(ctx_.int64Type(), sp);
        llvm::Value* dval = builder.CreateBitCast(bits, ctx_.doubleType());
        llvm::Value* dp = builder.CreateGEP(ctx_.doubleType(), b_data, i);
        builder.CreateStore(dval, dp);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    // LU decompose
    llvm::FunctionType* lu_ft = llvm::FunctionType::get(
        ctx_.int64Type(), {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
    llvm::Function* lu_fn = getOrDeclareRuntimeFunc(ctx_.module(), ctx_.context(), "eshkol_lu_decompose", lu_ft);
    builder.CreateCall(lu_fn, {lu_data, piv_data, n});

    // Solve
    llvm::FunctionType* solve_ft = llvm::FunctionType::get(
        ctx_.voidType(), {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
    llvm::Function* solve_fn = getOrDeclareRuntimeFunc(ctx_.module(), ctx_.context(), "eshkol_lu_solve", solve_ft);
    builder.CreateCall(solve_fn, {lu_data, piv_data, b_data, n});

    // Create result 1D tensor from solution
    std::vector<llvm::Value*> dims = {n};
    llvm::Value* result_ptr = createTensorWithDims(dims);
    if (!result_ptr) return nullptr;

    llvm::Value* res_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    llvm::Value* res_elems = builder.CreateLoad(ctx_.ptrType(), res_elems_field);

    // Copy solution back
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "solve_cp_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "solve_cp_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "solve_cp_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, n), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* sp = builder.CreateGEP(ctx_.doubleType(), b_data, i);
        llvm::Value* dval = builder.CreateLoad(ctx_.doubleType(), sp);
        llvm::Value* ibits = builder.CreateBitCast(dval, ctx_.int64Type());
        llvm::Value* rp = builder.CreateGEP(ctx_.int64Type(), res_elems, i);
        builder.CreateStore(ibits, rp);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorCholesky(const eshkol_operations_t* op) {
    // tensor-cholesky: (tensor-cholesky A) -> L where A = L @ L^T
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-cholesky requires 1 argument: SPD matrix");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Value* a_tagged = codegenAST(&op->call_op.variables[0]);
    if (!a_tagged) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* a_ptr = tagged_.unpackPtr(a_tagged);

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, a_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* n = builder.CreateLoad(ctx_.int64Type(), dims_ptr);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, a_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);

    llvm::Value* nn = builder.CreateMul(n, n);
    llvm::Value* byte_size = builder.CreateMul(nn, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
    llvm::Function* alloc_fn = mem_.getArenaAllocate();
    llvm::Value* a_data = builder.CreateCall(alloc_fn, {arena_ptr, byte_size}, "chol_a");
    emitArenaAllocNullCheck(builder, ctx_, a_data, "Error: out of memory in tensor-cholesky\n");
    llvm::Value* l_data = builder.CreateCall(alloc_fn, {arena_ptr, byte_size}, "chol_l");
    emitArenaAllocNullCheck(builder, ctx_, l_data, "Error: out of memory in tensor-cholesky (L)\n");

    // Copy tensor to double array
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "chol_cp_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "chol_cp_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "chol_cp_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, nn), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* sp = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
        llvm::Value* bits = builder.CreateLoad(ctx_.int64Type(), sp);
        llvm::Value* dval = builder.CreateBitCast(bits, ctx_.doubleType());
        llvm::Value* dp = builder.CreateGEP(ctx_.doubleType(), a_data, i);
        builder.CreateStore(dval, dp);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    // Call Cholesky
    llvm::FunctionType* chol_ft = llvm::FunctionType::get(
        ctx_.int64Type(), {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
    llvm::Function* chol_fn = getOrDeclareRuntimeFunc(ctx_.module(), ctx_.context(), "eshkol_cholesky", chol_ft);
    builder.CreateCall(chol_fn, {a_data, l_data, n});

    // Create result tensor
    std::vector<llvm::Value*> dims = {n, n};
    llvm::Value* result_ptr = createTensorWithDims(dims);
    if (!result_ptr) return nullptr;

    llvm::Value* res_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    llvm::Value* res_elems = builder.CreateLoad(ctx_.ptrType(), res_elems_field);

    // Copy L back
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "chol_cp2_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "chol_cp2_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "chol_cp2_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, nn), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* sp = builder.CreateGEP(ctx_.doubleType(), l_data, i);
        llvm::Value* dval = builder.CreateLoad(ctx_.doubleType(), sp);
        llvm::Value* ibits = builder.CreateBitCast(dval, ctx_.int64Type());
        llvm::Value* rp = builder.CreateGEP(ctx_.int64Type(), res_elems, i);
        builder.CreateStore(ibits, rp);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorQR(const eshkol_operations_t* op) {
    // tensor-qr: (tensor-qr A) -> returns list (Q R) where A = Q @ R
    // Q is orthogonal (m×m), R is upper triangular (m×n)
    // Full QR via Householder reflections
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-qr requires 1 argument: matrix");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Value* a_tagged = codegenAST(&op->call_op.variables[0]);
    if (!a_tagged) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* a_ptr = tagged_.unpackPtr(a_tagged);

    // Guard: QR decomposition requires a 2D matrix
    llvm::Value* qr_ndim_field = builder.CreateStructGEP(tensor_type, a_ptr, 1);
    llvm::Value* qr_ndim = builder.CreateLoad(ctx_.int64Type(), qr_ndim_field);
    {
        llvm::Function* cur_fn = builder.GetInsertBlock()->getParent();
        llvm::Value* dims_ok = builder.CreateICmpUGE(qr_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 2));
        llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "qr_dims_ok", cur_fn);
        llvm::BasicBlock* err_bb = llvm::BasicBlock::Create(ctx_.context(), "qr_dims_err", cur_fn);
        builder.CreateCondBr(dims_ok, ok_bb, err_bb);
        builder.SetInsertPoint(err_bb);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = builder.CreateGlobalString("Error: QR decomposition requires a 2D matrix (got %lldD)\n");
            builder.CreateCall(pf, {fmt, qr_ndim});
            builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
        }
        builder.CreateUnreachable();
        builder.SetInsertPoint(ok_bb);
    }

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, a_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* m = builder.CreateLoad(ctx_.int64Type(), dims_ptr);
    llvm::Value* dim1_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* n_val = builder.CreateLoad(ctx_.int64Type(), dim1_ptr);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, a_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);

    llvm::Value* mn = builder.CreateMul(m, n_val);
    llvm::Value* mm = builder.CreateMul(m, m);
    llvm::Value* a_bytes = builder.CreateMul(mn, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* q_bytes = builder.CreateMul(mm, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* r_bytes = a_bytes; // m×n

    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
    llvm::Function* alloc_fn = mem_.getArenaAllocate();
    llvm::Value* a_data = builder.CreateCall(alloc_fn, {arena_ptr, a_bytes}, "qr_a");
    emitArenaAllocNullCheck(builder, ctx_, a_data, "Error: out of memory in tensor-qr\n");
    llvm::Value* q_data = builder.CreateCall(alloc_fn, {arena_ptr, q_bytes}, "qr_q");
    emitArenaAllocNullCheck(builder, ctx_, q_data, "Error: out of memory in tensor-qr (Q)\n");
    llvm::Value* r_data = builder.CreateCall(alloc_fn, {arena_ptr, r_bytes}, "qr_r");
    emitArenaAllocNullCheck(builder, ctx_, r_data, "Error: out of memory in tensor-qr (R)\n");

    // Copy tensor to double array
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "qr_cp_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "qr_cp_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "qr_cp_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, mn), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* sp = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
        llvm::Value* bits = builder.CreateLoad(ctx_.int64Type(), sp);
        llvm::Value* dval = builder.CreateBitCast(bits, ctx_.doubleType());
        llvm::Value* dp = builder.CreateGEP(ctx_.doubleType(), a_data, i);
        builder.CreateStore(dval, dp);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    // Call QR decomposition
    llvm::FunctionType* qr_ft = llvm::FunctionType::get(
        ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type(), ctx_.int64Type()},
        false);
    llvm::Function* qr_fn = getOrDeclareRuntimeFunc(ctx_.module(), ctx_.context(), "eshkol_qr_decompose", qr_ft);
    builder.CreateCall(qr_fn, {a_data, q_data, r_data, m, n_val});

    // Create Q result tensor (m×m)
    std::vector<llvm::Value*> q_dims = {m, m};
    llvm::Value* q_result = createTensorWithDims(q_dims);
    if (!q_result) return nullptr;

    llvm::Value* q_res_elems_field = builder.CreateStructGEP(tensor_type, q_result, 2);
    llvm::Value* q_res_elems = builder.CreateLoad(ctx_.ptrType(), q_res_elems_field);

    // Copy Q back
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "qr_cpq_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "qr_cpq_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "qr_cpq_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, mm), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* sp = builder.CreateGEP(ctx_.doubleType(), q_data, i);
        llvm::Value* dval = builder.CreateLoad(ctx_.doubleType(), sp);
        llvm::Value* ibits = builder.CreateBitCast(dval, ctx_.int64Type());
        llvm::Value* rp = builder.CreateGEP(ctx_.int64Type(), q_res_elems, i);
        builder.CreateStore(ibits, rp);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    // === Create R result tensor (m×n) ===
    std::vector<llvm::Value*> r_dims_vec = {m, n_val};
    llvm::Value* r_result = createTensorWithDims(r_dims_vec);
    if (!r_result) return nullptr;

    llvm::Value* r_res_elems_field = builder.CreateStructGEP(tensor_type, r_result, 2);
    llvm::Value* r_res_elems = builder.CreateLoad(ctx_.ptrType(), r_res_elems_field);

    // Copy R doubles back to tensor (same bitcast loop pattern as Q)
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "qr_cpr_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "qr_cpr_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "qr_cpr_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, mn), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* sp = builder.CreateGEP(ctx_.doubleType(), r_data, i);
        llvm::Value* dval = builder.CreateLoad(ctx_.doubleType(), sp);
        llvm::Value* ibits = builder.CreateBitCast(dval, ctx_.int64Type());
        llvm::Value* rp = builder.CreateGEP(ctx_.int64Type(), r_res_elems, i);
        builder.CreateStore(ibits, rp);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    // === Build cons list (Q . (R . '())) — same pattern as tensorSVD ===
    llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 1);

    // Cell 2: (R . '())
    llvm::Value* cons2 = builder.CreateCall(
        mem_.getArenaAllocateConsWithHeader(), {arena_ptr});
    llvm::Value* r_tagged = tagged_.packHeapPtr(r_result);
    llvm::Value* r_alloca = builder.CreateAlloca(ctx_.taggedValueType());
    builder.CreateStore(r_tagged, r_alloca);
    builder.CreateCall(mem_.getTaggedConsSetTaggedValue(), {cons2, is_car, r_alloca});
    builder.CreateCall(mem_.getTaggedConsSetNull(), {cons2, is_cdr});

    // Cell 1: (Q . cons2)
    llvm::Value* cons1 = builder.CreateCall(
        mem_.getArenaAllocateConsWithHeader(), {arena_ptr});
    llvm::Value* q_tagged = tagged_.packHeapPtr(q_result);
    llvm::Value* q_alloca = builder.CreateAlloca(ctx_.taggedValueType());
    builder.CreateStore(q_tagged, q_alloca);
    builder.CreateCall(mem_.getTaggedConsSetTaggedValue(), {cons1, is_car, q_alloca});
    llvm::Value* cons2_int = builder.CreatePtrToInt(cons2, ctx_.int64Type());
    llvm::Value* cons_type_val = llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR);
    builder.CreateCall(mem_.getTaggedConsSetPtr(), {cons1, is_cdr, cons2_int, cons_type_val});

    return tagged_.packHeapPtr(cons1);
}

llvm::Value* TensorCodegen::tensorSVD(const eshkol_operations_t* op) {
    // tensor-svd: (tensor-svd A) -> returns list (U S V) where A = U @ diag(S) @ V^T
    // One-sided Jacobi SVD via runtime function
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-svd requires 1 argument: matrix");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Value* a_tagged = codegenAST(&op->call_op.variables[0]);
    if (!a_tagged) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* a_ptr = tagged_.unpackPtr(a_tagged);

    // Get dimensions: m (rows), n (cols)
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, a_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* m = builder.CreateLoad(ctx_.int64Type(), dims_ptr);
    llvm::Value* dim1_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* n_val = builder.CreateLoad(ctx_.int64Type(), dim1_ptr);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, a_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);

    // k = min(m, n)
    llvm::Value* m_lt_n = builder.CreateICmpSLT(m, n_val);
    llvm::Value* k = builder.CreateSelect(m_lt_n, m, n_val, "svd_k");

    // Allocate working buffers via arena
    llvm::Value* mn = builder.CreateMul(m, n_val);
    llvm::Value* mk = builder.CreateMul(m, k);
    llvm::Value* nn = builder.CreateMul(n_val, n_val);
    llvm::Value* eight = llvm::ConstantInt::get(ctx_.int64Type(), 8);

    llvm::Value* a_bytes = builder.CreateMul(mn, eight);
    llvm::Value* u_bytes = builder.CreateMul(mk, eight);
    llvm::Value* s_bytes = builder.CreateMul(k, eight);
    llvm::Value* v_bytes = builder.CreateMul(nn, eight);

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
    llvm::Function* alloc_fn = mem_.getArenaAllocate();

    llvm::Value* a_data = builder.CreateCall(alloc_fn, {arena_ptr, a_bytes}, "svd_a");
    emitArenaAllocNullCheck(builder, ctx_, a_data, "Error: out of memory in tensor-svd\n");
    llvm::Value* u_data = builder.CreateCall(alloc_fn, {arena_ptr, u_bytes}, "svd_u");
    emitArenaAllocNullCheck(builder, ctx_, u_data, "Error: out of memory in tensor-svd (U)\n");
    llvm::Value* s_data = builder.CreateCall(alloc_fn, {arena_ptr, s_bytes}, "svd_s");
    emitArenaAllocNullCheck(builder, ctx_, s_data, "Error: out of memory in tensor-svd (S)\n");
    llvm::Value* v_data = builder.CreateCall(alloc_fn, {arena_ptr, v_bytes}, "svd_v");
    emitArenaAllocNullCheck(builder, ctx_, v_data, "Error: out of memory in tensor-svd (V)\n");

    // Copy tensor elements (int64 bitpatterns) to double array for A
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "svd_cp_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "svd_cp_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "svd_cp_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, mn), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* sp = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
        llvm::Value* bits = builder.CreateLoad(ctx_.int64Type(), sp);
        llvm::Value* dval = builder.CreateBitCast(bits, ctx_.doubleType());
        llvm::Value* dp = builder.CreateGEP(ctx_.doubleType(), a_data, i);
        builder.CreateStore(dval, dp);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    // Call SVD runtime function
    llvm::FunctionType* svd_ft = llvm::FunctionType::get(
        ctx_.voidType(),
        {ctx_.ptrType(), ctx_.int64Type(), ctx_.int64Type(),
         ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType()},
        false);
    llvm::Function* svd_fn = getOrDeclareRuntimeFunc(
        ctx_.module(), ctx_.context(), "eshkol_tensor_svd", svd_ft);
    builder.CreateCall(svd_fn, {a_data, m, n_val, u_data, s_data, v_data});

    // Create U result tensor (m × k)
    std::vector<llvm::Value*> u_dims = {m, k};
    llvm::Value* u_result = createTensorWithDims(u_dims);
    if (!u_result) return nullptr;

    llvm::Value* u_res_elems_field = builder.CreateStructGEP(tensor_type, u_result, 2);
    llvm::Value* u_res_elems = builder.CreateLoad(ctx_.ptrType(), u_res_elems_field);

    // Copy U doubles back to int64 bitpatterns
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "svd_cpu_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "svd_cpu_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "svd_cpu_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, mk), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* sp = builder.CreateGEP(ctx_.doubleType(), u_data, i);
        llvm::Value* dval = builder.CreateLoad(ctx_.doubleType(), sp);
        llvm::Value* ibits = builder.CreateBitCast(dval, ctx_.int64Type());
        llvm::Value* rp = builder.CreateGEP(ctx_.int64Type(), u_res_elems, i);
        builder.CreateStore(ibits, rp);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    // Create S result tensor (k) - 1D vector of singular values
    std::vector<llvm::Value*> s_dims = {k};
    llvm::Value* s_result = createTensorWithDims(s_dims);
    if (!s_result) return nullptr;

    llvm::Value* s_res_elems_field = builder.CreateStructGEP(tensor_type, s_result, 2);
    llvm::Value* s_res_elems = builder.CreateLoad(ctx_.ptrType(), s_res_elems_field);

    // Copy S doubles back to int64 bitpatterns
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "svd_cps_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "svd_cps_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "svd_cps_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, k), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* sp = builder.CreateGEP(ctx_.doubleType(), s_data, i);
        llvm::Value* dval = builder.CreateLoad(ctx_.doubleType(), sp);
        llvm::Value* ibits = builder.CreateBitCast(dval, ctx_.int64Type());
        llvm::Value* rp = builder.CreateGEP(ctx_.int64Type(), s_res_elems, i);
        builder.CreateStore(ibits, rp);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    // Create V result tensor (n × n)
    std::vector<llvm::Value*> v_dims = {n_val, n_val};
    llvm::Value* v_result = createTensorWithDims(v_dims);
    if (!v_result) return nullptr;

    llvm::Value* v_res_elems_field = builder.CreateStructGEP(tensor_type, v_result, 2);
    llvm::Value* v_res_elems = builder.CreateLoad(ctx_.ptrType(), v_res_elems_field);

    // Copy V doubles back to int64 bitpatterns
    {
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "svd_cpv_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "svd_cpv_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "svd_cpv_done", current_func);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, nn), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* sp = builder.CreateGEP(ctx_.doubleType(), v_data, i);
        llvm::Value* dval = builder.CreateLoad(ctx_.doubleType(), sp);
        llvm::Value* ibits = builder.CreateBitCast(dval, ctx_.int64Type());
        llvm::Value* rp = builder.CreateGEP(ctx_.int64Type(), v_res_elems, i);
        builder.CreateStore(ibits, rp);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);
        builder.SetInsertPoint(done);
    }

    // Build result list: (U S V) using cons cells
    // Build from back to front: cons(V, '()) -> cons(S, prev) -> cons(U, prev)
    llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 1);

    // Cell 3: (V . '())
    llvm::Value* cons3 = builder.CreateCall(
        mem_.getArenaAllocateConsWithHeader(), {arena_ptr});
    llvm::Value* v_tagged = tagged_.packHeapPtr(v_result);
    llvm::Value* v_alloca = builder.CreateAlloca(ctx_.taggedValueType());
    builder.CreateStore(v_tagged, v_alloca);
    builder.CreateCall(mem_.getTaggedConsSetTaggedValue(), {cons3, is_car, v_alloca});
    builder.CreateCall(mem_.getTaggedConsSetNull(), {cons3, is_cdr});

    // Cell 2: (S . cons3)
    llvm::Value* cons2 = builder.CreateCall(
        mem_.getArenaAllocateConsWithHeader(), {arena_ptr});
    llvm::Value* s_tagged = tagged_.packHeapPtr(s_result);
    llvm::Value* s_alloca = builder.CreateAlloca(ctx_.taggedValueType());
    builder.CreateStore(s_tagged, s_alloca);
    builder.CreateCall(mem_.getTaggedConsSetTaggedValue(), {cons2, is_car, s_alloca});
    // Set cdr to cons3 pointer
    llvm::Value* cons3_int = builder.CreatePtrToInt(cons3, ctx_.int64Type());
    llvm::Value* cons_type_val = llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR);
    builder.CreateCall(mem_.getTaggedConsSetPtr(), {cons2, is_cdr, cons3_int, cons_type_val});

    // Cell 1: (U . cons2)
    llvm::Value* cons1 = builder.CreateCall(
        mem_.getArenaAllocateConsWithHeader(), {arena_ptr});
    llvm::Value* u_tagged = tagged_.packHeapPtr(u_result);
    llvm::Value* u_alloca = builder.CreateAlloca(ctx_.taggedValueType());
    builder.CreateStore(u_tagged, u_alloca);
    builder.CreateCall(mem_.getTaggedConsSetTaggedValue(), {cons1, is_car, u_alloca});
    llvm::Value* cons2_int = builder.CreatePtrToInt(cons2, ctx_.int64Type());
    builder.CreateCall(mem_.getTaggedConsSetPtr(), {cons1, is_cdr, cons2_int, cons_type_val});

    return tagged_.packHeapPtr(cons1);
}

llvm::Value* TensorCodegen::tensorEinsum(const eshkol_operations_t* op) {
    // einsum: (einsum "ij,jk->ik" A B) - Einstein summation
    // For now, support the most common cases:
    //   "ij,jk->ik" = matmul
    //   "ij->ji" = transpose
    //   "ii->" = trace
    //   "ij,ij->" = element-wise multiply and sum (Frobenius inner product)
    //   "i,i->" = dot product
    if (op->call_op.num_vars < 2) {
        eshkol_error("einsum requires at least 2 arguments: notation string and tensor(s)");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    // First argument is the notation string - we need to get it at compile time
    // For production use, einsum notation is a string literal.
    // We dispatch to existing optimized implementations based on the pattern.
    const eshkol_ast_t* notation_ast = &op->call_op.variables[0];
    if (notation_ast->type != ESHKOL_STRING) {
        eshkol_error("einsum first argument must be a string literal notation");
        return nullptr;
    }
    const char* notation = notation_ast->str_val.ptr;

    // Parse notation to determine operation
    // Format: "subscripts_input1,subscripts_input2->subscripts_output"
    std::string nota(notation);
    size_t arrow_pos = nota.find("->");

    if (nota == "ij,jk->ik" && op->call_op.num_vars == 3) {
        // Matrix multiplication - delegate to existing matmul
        llvm::Value* a_tagged = codegenAST(&op->call_op.variables[1]);
        llvm::Value* b_tagged = codegenAST(&op->call_op.variables[2]);
        if (!a_tagged || !b_tagged) return nullptr;

        llvm::StructType* tensor_type = ctx_.tensorType();
        llvm::Value* a_ptr = tagged_.unpackPtr(a_tagged);
        llvm::Value* b_ptr = tagged_.unpackPtr(b_tagged);

        // Guard: einsum matmul requires 2D tensors
        llvm::Value* a_ndim_f = builder.CreateStructGEP(tensor_type, a_ptr, 1);
        llvm::Value* a_ndim_val = builder.CreateLoad(ctx_.int64Type(), a_ndim_f);
        llvm::Value* b_ndim_f = builder.CreateStructGEP(tensor_type, b_ptr, 1);
        llvm::Value* b_ndim_val = builder.CreateLoad(ctx_.int64Type(), b_ndim_f);
        {
            llvm::Function* cur_fn = builder.GetInsertBlock()->getParent();
            llvm::Value* a_ok = builder.CreateICmpUGE(a_ndim_val, llvm::ConstantInt::get(ctx_.int64Type(), 2));
            llvm::Value* b_ok = builder.CreateICmpUGE(b_ndim_val, llvm::ConstantInt::get(ctx_.int64Type(), 2));
            llvm::Value* both_ok = builder.CreateAnd(a_ok, b_ok);
            llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "einmm_dims_ok", cur_fn);
            llvm::BasicBlock* err_bb = llvm::BasicBlock::Create(ctx_.context(), "einmm_dims_err", cur_fn);
            builder.CreateCondBr(both_ok, ok_bb, err_bb);
            builder.SetInsertPoint(err_bb);
            llvm::Function* pf = ctx_.lookupFunction("printf");
            llvm::Function* ef = ctx_.lookupFunction("exit");
            if (pf && ef) {
                llvm::Value* fmt = builder.CreateGlobalString("Error: einsum matmul requires 2D tensors (got %lldD and %lldD)\n");
                builder.CreateCall(pf, {fmt, a_ndim_val, b_ndim_val});
                builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
            }
            builder.CreateUnreachable();
            builder.SetInsertPoint(ok_bb);
        }

        // Get dimensions
        llvm::Value* a_dims_f = builder.CreateStructGEP(tensor_type, a_ptr, 0);
        llvm::Value* a_dims = builder.CreateLoad(ctx_.ptrType(), a_dims_f);
        llvm::Value* M = builder.CreateLoad(ctx_.int64Type(), a_dims);
        llvm::Value* K_ptr = builder.CreateGEP(ctx_.int64Type(), a_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        llvm::Value* K = builder.CreateLoad(ctx_.int64Type(), K_ptr);

        llvm::Value* b_dims_f = builder.CreateStructGEP(tensor_type, b_ptr, 0);
        llvm::Value* b_dims = builder.CreateLoad(ctx_.ptrType(), b_dims_f);
        llvm::Value* N_ptr = builder.CreateGEP(ctx_.int64Type(), b_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        llvm::Value* N = builder.CreateLoad(ctx_.int64Type(), N_ptr);

        llvm::Value* a_elems_f = builder.CreateStructGEP(tensor_type, a_ptr, 2);
        llvm::Value* a_elems = builder.CreateLoad(ctx_.ptrType(), a_elems_f);
        llvm::Value* b_elems_f = builder.CreateStructGEP(tensor_type, b_ptr, 2);
        llvm::Value* b_elems = builder.CreateLoad(ctx_.ptrType(), b_elems_f);

        return matmulSIMD(a_elems, b_elems, M, K, N);
    }

    if (nota == "ij->ji" && op->call_op.num_vars == 2) {
        // Transpose - delegate
        return transpose(op);
    }

    if (nota == "ii->" && op->call_op.num_vars == 2) {
        // Trace - sum of diagonal
        llvm::Value* a_tagged = codegenAST(&op->call_op.variables[1]);
        if (!a_tagged) return nullptr;

        llvm::StructType* tensor_type = ctx_.tensorType();
        llvm::Value* a_ptr = tagged_.unpackPtr(a_tagged);
        llvm::Value* dims_f = builder.CreateStructGEP(tensor_type, a_ptr, 0);
        llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_f);
        llvm::Value* n = builder.CreateLoad(ctx_.int64Type(), dims);
        llvm::Value* elems_f = builder.CreateStructGEP(tensor_type, a_ptr, 2);
        llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_f);

        // Sum diagonal: sum(A[i,i]) for i=0..n-1
        llvm::Function* current_func = builder.GetInsertBlock()->getParent();
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "trace_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "trace_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "trace_done", current_func);

        llvm::Value* acc = builder.CreateAlloca(ctx_.doubleType());
        builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), acc);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, n), body, done);

        builder.SetInsertPoint(body);
        // Index = i * n + i = i * (n + 1)
        llvm::Value* stride = builder.CreateAdd(n, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        llvm::Value* flat_idx = builder.CreateMul(i, stride);
        llvm::Value* ep = builder.CreateGEP(ctx_.int64Type(), elems, flat_idx);
        llvm::Value* bits = builder.CreateLoad(ctx_.int64Type(), ep);
        llvm::Value* dval = builder.CreateBitCast(bits, ctx_.doubleType());
        llvm::Value* cur = builder.CreateLoad(ctx_.doubleType(), acc);
        builder.CreateStore(builder.CreateFAdd(cur, dval), acc);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(done);
        return tagged_.packDouble(builder.CreateLoad(ctx_.doubleType(), acc));
    }

    if (nota == "i,i->" && op->call_op.num_vars == 3) {
        // Dot product
        llvm::Value* a_tagged = codegenAST(&op->call_op.variables[1]);
        llvm::Value* b_tagged = codegenAST(&op->call_op.variables[2]);
        if (!a_tagged || !b_tagged) return nullptr;

        llvm::StructType* tensor_type = ctx_.tensorType();
        llvm::Value* a_ptr = tagged_.unpackPtr(a_tagged);
        llvm::Value* b_ptr = tagged_.unpackPtr(b_tagged);
        llvm::Value* a_total_f = builder.CreateStructGEP(tensor_type, a_ptr, 3);
        llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), a_total_f);
        llvm::Value* a_elems_f = builder.CreateStructGEP(tensor_type, a_ptr, 2);
        llvm::Value* a_elems = builder.CreateLoad(ctx_.ptrType(), a_elems_f);
        llvm::Value* b_elems_f = builder.CreateStructGEP(tensor_type, b_ptr, 2);
        llvm::Value* b_elems = builder.CreateLoad(ctx_.ptrType(), b_elems_f);

        llvm::Function* current_func = builder.GetInsertBlock()->getParent();
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "dot_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "dot_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "dot_done", current_func);

        llvm::Value* acc = builder.CreateAlloca(ctx_.doubleType());
        builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), acc);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, total), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* ap = builder.CreateGEP(ctx_.int64Type(), a_elems, i);
        llvm::Value* a_bits = builder.CreateLoad(ctx_.int64Type(), ap);
        llvm::Value* a_val = builder.CreateBitCast(a_bits, ctx_.doubleType());
        llvm::Value* bp = builder.CreateGEP(ctx_.int64Type(), b_elems, i);
        llvm::Value* b_bits = builder.CreateLoad(ctx_.int64Type(), bp);
        llvm::Value* b_val = builder.CreateBitCast(b_bits, ctx_.doubleType());
        llvm::Value* prod = builder.CreateFMul(a_val, b_val);
        llvm::Value* cur = builder.CreateLoad(ctx_.doubleType(), acc);
        builder.CreateStore(builder.CreateFAdd(cur, prod), acc);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(done);
        return tagged_.packDouble(builder.CreateLoad(ctx_.doubleType(), acc));
    }

    if (nota == "ij,ij->" && op->call_op.num_vars == 3) {
        // Frobenius inner product: sum of element-wise products
        // Same as dot product but for 2D tensors
        llvm::Value* a_tagged = codegenAST(&op->call_op.variables[1]);
        llvm::Value* b_tagged = codegenAST(&op->call_op.variables[2]);
        if (!a_tagged || !b_tagged) return nullptr;

        llvm::StructType* tensor_type = ctx_.tensorType();
        llvm::Value* a_ptr = tagged_.unpackPtr(a_tagged);
        llvm::Value* b_ptr = tagged_.unpackPtr(b_tagged);
        llvm::Value* a_total_f = builder.CreateStructGEP(tensor_type, a_ptr, 3);
        llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), a_total_f);
        llvm::Value* a_elems_f = builder.CreateStructGEP(tensor_type, a_ptr, 2);
        llvm::Value* a_elems = builder.CreateLoad(ctx_.ptrType(), a_elems_f);
        llvm::Value* b_elems_f = builder.CreateStructGEP(tensor_type, b_ptr, 2);
        llvm::Value* b_elems = builder.CreateLoad(ctx_.ptrType(), b_elems_f);

        llvm::Function* current_func = builder.GetInsertBlock()->getParent();
        llvm::BasicBlock* cond = llvm::BasicBlock::Create(ctx_.context(), "frob_cond", current_func);
        llvm::BasicBlock* body = llvm::BasicBlock::Create(ctx_.context(), "frob_body", current_func);
        llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "frob_done", current_func);

        llvm::Value* acc = builder.CreateAlloca(ctx_.doubleType());
        builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), acc);
        llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(cond);
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
        builder.CreateCondBr(builder.CreateICmpULT(i, total), body, done);

        builder.SetInsertPoint(body);
        llvm::Value* ap = builder.CreateGEP(ctx_.int64Type(), a_elems, i);
        llvm::Value* a_bits = builder.CreateLoad(ctx_.int64Type(), ap);
        llvm::Value* a_val = builder.CreateBitCast(a_bits, ctx_.doubleType());
        llvm::Value* bp = builder.CreateGEP(ctx_.int64Type(), b_elems, i);
        llvm::Value* b_bits = builder.CreateLoad(ctx_.int64Type(), bp);
        llvm::Value* b_val = builder.CreateBitCast(b_bits, ctx_.doubleType());
        llvm::Value* prod = builder.CreateFMul(a_val, b_val);
        llvm::Value* cur = builder.CreateLoad(ctx_.doubleType(), acc);
        builder.CreateStore(builder.CreateFAdd(cur, prod), acc);
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
        builder.CreateBr(cond);

        builder.SetInsertPoint(done);
        return tagged_.packDouble(builder.CreateLoad(ctx_.doubleType(), acc));
    }

    eshkol_error("einsum: unsupported notation '%s'. Supported: 'ij,jk->ik', 'ij->ji', 'ii->', 'i,i->', 'ij,ij->'", notation);
    return nullptr;
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
