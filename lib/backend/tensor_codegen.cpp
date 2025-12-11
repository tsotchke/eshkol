/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen implementation
 *
 * Note: The complex tensor implementations remain in llvm_codegen.cpp
 * for now due to dependencies on AST codegen and autodiff operations.
 * This module provides the interface and will be populated as
 * dependencies are extracted.
 */

#include <eshkol/backend/tensor_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>

namespace eshkol {

TensorCodegen::TensorCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem)
    : ctx_(ctx)
    , tagged_(tagged)
    , mem_(mem) {
    eshkol_debug("TensorCodegen initialized");
}

// Note: All tensor implementations are complex and depend on:
// - AST code generation for nested expressions
// - Autodiff integration (dual numbers, AD nodes)
// - Arena memory allocation for results
// - Runtime library functions
//
// These implementations remain in llvm_codegen.cpp until those modules are extracted.

llvm::Value* TensorCodegen::createTensor(const eshkol_ast_t* ast) {
    eshkol_warn("TensorCodegen::createTensor called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorOperation(const eshkol_operations_t* op) {
    if (!op || op->op != ESHKOL_TENSOR_OP) return nullptr;

    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get arena pointer
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(context, 0), ctx_.globalArena());

    // Allocate tensor with header using arena_allocate_tensor_with_header
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* typed_tensor_ptr = builder.CreateCall(alloc_tensor_func, {arena_ptr}, "tensor_ptr");

    // Allocate dimensions array using arena
    llvm::Value* dims_size = llvm::ConstantInt::get(ctx_.int64Type(),
        op->tensor_op.num_dimensions * sizeof(uint64_t));
    llvm::Function* arena_alloc_func = mem_.getArenaAllocate();
    llvm::Value* dims_ptr = builder.CreateCall(arena_alloc_func, {arena_ptr, dims_size}, "dims_ptr");
    llvm::Value* typed_dims_ptr = builder.CreatePointerCast(dims_ptr, builder.getPtrTy());

    for (uint64_t i = 0; i < op->tensor_op.num_dimensions; i++) {
        llvm::Value* dim_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), op->tensor_op.dimensions[i]), dim_ptr);
    }

    // Allocate and populate elements array using arena
    llvm::Value* elements_size = llvm::ConstantInt::get(ctx_.int64Type(),
        op->tensor_op.total_elements * sizeof(int64_t));
    llvm::Value* elements_ptr = builder.CreateCall(arena_alloc_func, {arena_ptr, elements_size}, "elems_ptr");
    llvm::Value* typed_elements_ptr = builder.CreatePointerCast(elements_ptr, builder.getPtrTy());

    for (uint64_t i = 0; i < op->tensor_op.total_elements; i++) {
        llvm::Value* element_val = codegenAST(&op->tensor_op.elements[i]);
        if (element_val) {
            // Tensors store all elements as doubles (bit patterns in i64)
            // We need to convert integers to doubles first, then bitcast to i64

            llvm::Value* double_val = nullptr;

            if (element_val->getType() == ctx_.taggedValueType()) {
                // Tagged value - need to check type and extract appropriately
                // For now, use extractAsDouble which handles both int and double tagged values
                double_val = extractAsDouble(element_val);
            } else if (element_val->getType()->isIntegerTy()) {
                // Raw integer - convert to double first
                double_val = ctx_.builder().CreateSIToFP(element_val, ctx_.doubleType());
            } else if (element_val->getType()->isFloatingPointTy()) {
                // Already a double or float
                if (element_val->getType() != ctx_.doubleType()) {
                    double_val = ctx_.builder().CreateFPExt(element_val, ctx_.doubleType());
                } else {
                    double_val = element_val;
                }
            } else {
                // Unknown type - default to 0.0
                double_val = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
            }

            // Bitcast double to i64 for storage
            llvm::Value* i64_val = ctx_.builder().CreateBitCast(double_val, ctx_.int64Type());

            llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements_ptr,
                llvm::ConstantInt::get(ctx_.int64Type(), i));
            ctx_.builder().CreateStore(i64_val, elem_ptr);
        }
    }

    // Store fields in tensor structure
    llvm::Value* dims_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 0);
    ctx_.builder().CreateStore(typed_dims_ptr, dims_field_ptr);

    llvm::Value* num_dims_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 1);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), op->tensor_op.num_dimensions), num_dims_field_ptr);

    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 2);
    ctx_.builder().CreateStore(typed_elements_ptr, elements_field_ptr);

    llvm::Value* total_elements_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 3);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), op->tensor_op.total_elements), total_elements_field_ptr);

    // Return pointer to tensor as consolidated HEAP_PTR tagged value
    // (subtype HEAP_SUBTYPE_TENSOR is stored in the object header)
    return tagged_.packHeapPtr(typed_tensor_ptr);
}

llvm::Value* TensorCodegen::tensorGet(const eshkol_operations_t* op) {
    // tensor-get: (tensor-get tensor index1 index2 ...)
    //
    // N-DIMENSIONAL SLICING SUPPORT:
    // - If num_indices == ndim: return scalar element
    // - If num_indices < ndim: return view tensor (slice)
    //
    // For tensor with shape [d0, d1, ..., d(n-1)] and indices [i0, ..., i(k-1)]:
    //   linear_offset = sum(i_j * stride_j) where stride_j = product(d(j+1)..d(n-1))
    //   slice_shape = [d(k), d(k+1), ..., d(n-1)]
    //   slice_elements = elements + linear_offset
    //
    if (op->call_op.num_vars < 2) {
        eshkol_error("tensor-get requires at least tensor and one index");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    const uint64_t num_indices = op->call_op.num_vars - 1;

    // Extract tensor pointer
    llvm::Value* tensor_ptr_int = tagged_.safeExtractInt64(tensor_val);
    llvm::Value* tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());

    // Load tensor metadata
    llvm::StructType* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), dims_field);

    llvm::Value* ndim_field = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* ndim = ctx_.builder().CreateLoad(ctx_.int64Type(), ndim_field);

    llvm::Value* elements_field = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), elements_field);

    llvm::Value* total_field = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = ctx_.builder().CreateLoad(ctx_.int64Type(), total_field);

    // ===== PHASE 1: Compute strides for all dimensions =====
    // stride[i] = product of dims[i+1..n-1] (row-major order)
    // We compute these using a loop since ndim is runtime-known
    //
    // stride[n-1] = 1
    // stride[i] = stride[i+1] * dims[i+1]
    //
    // But we need strides for indices 0..num_indices-1, computed from dims
    // Since indices might be < ndim, we use total_elements/product(dims[0..i])

    // First compute stride[0] = total_elements / dims[0]
    // Then stride[i] = stride[i-1] / dims[i]

    // Collect indices and compute linear offset
    std::vector<llvm::Value*> indices;
    for (uint64_t i = 0; i < num_indices; i++) {
        llvm::Value* idx = codegenAST(&op->call_op.variables[i + 1]);
        if (!idx) return nullptr;
        indices.push_back(tagged_.safeExtractInt64(idx));
    }

    // Compute linear offset incrementally
    // offset = i0*stride0 + i1*stride1 + ... where stride_j = total / prod(d0..dj)
    llvm::Value* linear_offset = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    llvm::Value* prod_dims = llvm::ConstantInt::get(ctx_.int64Type(), 1);

    for (uint64_t i = 0; i < num_indices; i++) {
        // Load dims[i]
        llvm::Value* dim_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), dims_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        llvm::Value* dim_i = ctx_.builder().CreateLoad(ctx_.int64Type(), dim_ptr);

        // stride[i] = total_elements / (prod_dims * dim_i) = total_elements / prod_dims_including_i
        llvm::Value* prod_dims_next = ctx_.builder().CreateMul(prod_dims, dim_i);

        // stride[i] = total / prod_dims_next
        llvm::Value* stride_i = ctx_.builder().CreateUDiv(total_elements, prod_dims_next);

        // offset += indices[i] * stride[i]
        llvm::Value* contrib = ctx_.builder().CreateMul(indices[i], stride_i);
        linear_offset = ctx_.builder().CreateAdd(linear_offset, contrib);

        prod_dims = prod_dims_next;
    }

    // ===== PHASE 2: Decide scalar vs slice =====
    llvm::Value* num_indices_val = llvm::ConstantInt::get(ctx_.int64Type(), num_indices);
    llvm::Value* is_full_index = ctx_.builder().CreateICmpEQ(num_indices_val, ndim);

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* scalar_bb = llvm::BasicBlock::Create(ctx_.context(), "tget_scalar", func);
    llvm::BasicBlock* slice_bb = llvm::BasicBlock::Create(ctx_.context(), "tget_slice", func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "tget_done", func);

    ctx_.builder().CreateCondBr(is_full_index, scalar_bb, slice_bb);

    // ===== SCALAR PATH: Full indexing - return element as double =====
    ctx_.builder().SetInsertPoint(scalar_bb);

    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), elements_ptr, linear_offset);
    llvm::Value* elem_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem_double = ctx_.builder().CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* scalar_result = tagged_.packDouble(elem_double);

    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* scalar_exit = ctx_.builder().GetInsertBlock();

    // ===== SLICE PATH: Partial indexing - return view tensor =====
    ctx_.builder().SetInsertPoint(slice_bb);

    // Calculate remaining dimensions: new_ndim = ndim - num_indices
    llvm::Value* new_ndim = ctx_.builder().CreateSub(ndim, num_indices_val);

    // Calculate slice total elements: total / prod_dims
    // (prod_dims = product of indexed dimensions)
    llvm::Value* slice_total = ctx_.builder().CreateUDiv(total_elements, prod_dims);

    // Get arena pointer
    llvm::Value* arena_ptr_slice = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Allocate new tensor struct with header using arena
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* new_tensor = ctx_.builder().CreateCall(alloc_tensor_func, {arena_ptr_slice}, "slice_tensor");

    // Allocate new dims array using arena: new_ndim * 8 bytes
    llvm::Value* dims_bytes = ctx_.builder().CreateMul(new_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* new_dims = ctx_.builder().CreateCall(arena_alloc, {arena_ptr_slice, dims_bytes}, "slice_dims");

    // Copy remaining dimensions using a loop
    llvm::BasicBlock* copy_entry = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "copy_dims_cond", func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "copy_dims_body", func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "copy_dims_done", func);

    ctx_.builder().CreateBr(copy_cond);

    ctx_.builder().SetInsertPoint(copy_cond);
    llvm::PHINode* copy_i = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "i");
    copy_i->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_entry);

    llvm::Value* copy_done_cond = ctx_.builder().CreateICmpUGE(copy_i, new_ndim);
    ctx_.builder().CreateCondBr(copy_done_cond, copy_done, copy_body);

    ctx_.builder().SetInsertPoint(copy_body);
    // Source index = num_indices + i
    llvm::Value* src_idx = ctx_.builder().CreateAdd(copy_i, num_indices_val);
    llvm::Value* src_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), dims_ptr, src_idx);
    llvm::Value* dim_val = ctx_.builder().CreateLoad(ctx_.int64Type(), src_ptr);

    llvm::Value* dst_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), new_dims, copy_i);
    ctx_.builder().CreateStore(dim_val, dst_ptr);

    llvm::Value* next_i = ctx_.builder().CreateAdd(copy_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    copy_i->addIncoming(next_i, copy_body);
    ctx_.builder().CreateBr(copy_cond);

    ctx_.builder().SetInsertPoint(copy_done);

    // Fill tensor struct fields
    // Field 0: dims pointer
    llvm::Value* f0 = ctx_.builder().CreateStructGEP(tensor_type, new_tensor, 0);
    ctx_.builder().CreateStore(new_dims, f0);

    // Field 1: ndim
    llvm::Value* f1 = ctx_.builder().CreateStructGEP(tensor_type, new_tensor, 1);
    ctx_.builder().CreateStore(new_ndim, f1);

    // Field 2: elements pointer (view into original at offset)
    llvm::Value* slice_start = ctx_.builder().CreateGEP(ctx_.int64Type(), elements_ptr, linear_offset);
    llvm::Value* f2 = ctx_.builder().CreateStructGEP(tensor_type, new_tensor, 2);
    ctx_.builder().CreateStore(slice_start, f2);

    // Field 3: total_elements
    llvm::Value* f3 = ctx_.builder().CreateStructGEP(tensor_type, new_tensor, 3);
    ctx_.builder().CreateStore(slice_total, f3);

    // Pack as consolidated HEAP_PTR tagged value (subtype in header)
    llvm::Value* slice_result = tagged_.packHeapPtr(new_tensor);

    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* slice_exit = ctx_.builder().GetInsertBlock();

    // ===== MERGE =====
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "tget_result");
    result->addIncoming(scalar_result, scalar_exit);
    result->addIncoming(slice_result, slice_exit);

    return result;
}

llvm::Value* TensorCodegen::vectorRef(const eshkol_operations_t* op) {
    // vref is AD-aware and complex - remains in llvm_codegen.cpp
    eshkol_warn("TensorCodegen::vectorRef called - AD-aware vref should use codegenTensorVectorRef");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorSet(const eshkol_operations_t* op) {
    // tensor-set: (tensor-set tensor value index1 index2 ...)
    if (op->call_op.num_vars < 3) {
        eshkol_error("tensor-set requires at least tensor, value, and one index");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* new_value = codegenAST(&op->call_op.variables[1]);
    if (!tensor_val || !new_value) return nullptr;

    // Extract the tensor pointer from the tagged value
    llvm::Value* tensor_ptr_int = tagged_.safeExtractInt64(tensor_val);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());

    // Calculate linear index from multi-dimensional indices
    llvm::Value* linear_index = llvm::ConstantInt::get(ctx_.int64Type(), 0);

    llvm::Value* dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), dims_field_ptr);
    llvm::Value* typed_dims_ptr = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.ptrType());

    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), elements_field_ptr);
    llvm::Value* typed_elements_ptr = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.ptrType());

    // Calculate linear index
    llvm::Value* stride = llvm::ConstantInt::get(ctx_.int64Type(), 1);
    for (int64_t i = op->call_op.num_vars - 3; i >= 0; i--) {
        llvm::Value* index = codegenAST(&op->call_op.variables[i + 2]);
        if (index) {
            llvm::Value* contribution = ctx_.builder().CreateMul(index, stride);
            linear_index = ctx_.builder().CreateAdd(linear_index, contribution);

            if (i > 0) {
                llvm::Value* dim_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims_ptr,
                                                      llvm::ConstantInt::get(ctx_.int64Type(), i));
                llvm::Value* dim = ctx_.builder().CreateLoad(ctx_.int64Type(), dim_ptr);
                stride = ctx_.builder().CreateMul(stride, dim);
            }
        }
    }

    // Store new value at linear index
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements_ptr, linear_index);
    ctx_.builder().CreateStore(new_value, elem_ptr);

    return tensor_ptr_int; // Return the tensor
}

llvm::Value* TensorCodegen::tensorArithmetic(const eshkol_operations_t* op, const std::string& operation) {
    // tensor-add/sub/mul/div: (tensor-op arg1 arg2)
    // Supports both scheme vectors (VECTOR_PTR) and tensors (TENSOR_PTR)
    if (op->call_op.num_vars != 2) {
        eshkol_error("tensor arithmetic requires exactly 2 arguments");
        return nullptr;
    }

    // Get values - they will be tagged values containing type info at runtime
    llvm::Value* arg1 = codegenAST(&op->call_op.variables[0]);
    llvm::Value* arg2 = codegenAST(&op->call_op.variables[1]);
    if (!arg1 || !arg2) return nullptr;

    // Delegate to internal helper
    return tensorArithmeticInternal(arg1, arg2, operation);
}

// ===== INTERNAL TENSOR ARITHMETIC IMPLEMENTATIONS =====

// Scheme vector arithmetic: vectors with [length:i64][elem0:tagged][elem1:tagged]...
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

    // Extract doubles from tagged values
    llvm::Value* elem1_double = tagged_.unpackDouble(elem1_tagged);
    llvm::Value* elem2_double = tagged_.unpackDouble(elem2_tagged);

    // Perform operation
    llvm::Value* result_double = nullptr;
    if (operation == "add") {
        result_double = ctx_.builder().CreateFAdd(elem1_double, elem2_double);
    } else if (operation == "sub") {
        result_double = ctx_.builder().CreateFSub(elem1_double, elem2_double);
    } else if (operation == "mul") {
        result_double = ctx_.builder().CreateFMul(elem1_double, elem2_double);
    } else if (operation == "div") {
        result_double = ctx_.builder().CreateFDiv(elem1_double, elem2_double);
    }

    // Pack result and store
    llvm::Value* result_tagged = tagged_.packDouble(result_double);
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
    return tagged_.packHeapPtr(typed_result_tensor_ptr);
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
    llvm::Value* tensor_result = rawTensorArithmetic(arg1, arg2, operation);
    ctx_.builder().CreateStore(tensor_result, result_alloca);
    ctx_.builder().CreateBr(merge_block);

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

    llvm::Value* svec_sum = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "svec_dot_acc");
    llvm::Value* svec_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "svec_dot_i");
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), svec_sum);
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
    llvm::Value* svec_a_val = extractAsDouble(svec_a_elem_tagged);

    llvm::Value* svec_b_elems_base = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_b_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* svec_b_elems_typed = ctx_.builder().CreatePointerCast(svec_b_elems_base, ctx_.ptrType());
    llvm::Value* svec_b_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_b_elems_typed, svec_i);
    llvm::Value* svec_b_elem_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_b_elem_ptr);
    llvm::Value* svec_b_val = extractAsDouble(svec_b_elem_tagged);

    // Multiply and accumulate
    llvm::Value* svec_product = ctx_.builder().CreateFMul(svec_a_val, svec_b_val);
    llvm::Value* svec_current_sum = ctx_.builder().CreateLoad(ctx_.doubleType(), svec_sum);
    llvm::Value* svec_new_sum = ctx_.builder().CreateFAdd(svec_current_sum, svec_product);
    ctx_.builder().CreateStore(svec_new_sum, svec_sum);
    llvm::Value* svec_next_i = ctx_.builder().CreateAdd(svec_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(svec_next_i, svec_counter);
    ctx_.builder().CreateBr(svec_loop_cond);

    ctx_.builder().SetInsertPoint(svec_loop_exit);
    llvm::Value* svec_result = ctx_.builder().CreateLoad(ctx_.doubleType(), svec_sum);
    // Don't branch yet - we'll branch to scalar_merge once it's created
    llvm::BasicBlock* svec_exit_block = ctx_.builder().GetInsertBlock();

    // === TENSOR PATH ===
    ctx_.builder().SetInsertPoint(tensor_block);
    llvm::Value* tensor_a_ptr_int = tagged_.unpackInt64(val_a);
    llvm::Value* tensor_a_ptr = ctx_.builder().CreateIntToPtr(tensor_a_ptr_int, ctx_.ptrType());
    llvm::Value* tensor_b_ptr_int = tagged_.unpackInt64(val_b);
    llvm::Value* tensor_b_ptr = ctx_.builder().CreateIntToPtr(tensor_b_ptr_int, ctx_.ptrType());

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

    // Check if 1D vectors - use simple dot product
    llvm::Value* is_1d = ctx_.builder().CreateICmpEQ(a_num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::BasicBlock* dot_1d_block = llvm::BasicBlock::Create(ctx_.context(), "dot_1d", current_func);
    llvm::BasicBlock* dot_2d_block = llvm::BasicBlock::Create(ctx_.context(), "dot_2d", current_func);
    llvm::BasicBlock* tensor_merge = llvm::BasicBlock::Create(ctx_.context(), "tensor_dot_merge", current_func);

    ctx_.builder().CreateCondBr(is_1d, dot_1d_block, dot_2d_block);

    // 1D Vector Dot Product: sum(a[i] * b[i])
    ctx_.builder().SetInsertPoint(dot_1d_block);

    // Initialize accumulator
    llvm::Value* sum_alloca = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "dot_sum");
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_alloca);

    // Create loop
    llvm::BasicBlock* loop_cond_1d = llvm::BasicBlock::Create(ctx_.context(), "dot1d_cond", current_func);
    llvm::BasicBlock* loop_body_1d = llvm::BasicBlock::Create(ctx_.context(), "dot1d_body", current_func);
    llvm::BasicBlock* loop_exit_1d = llvm::BasicBlock::Create(ctx_.context(), "dot1d_exit", current_func);

    llvm::Value* counter_1d = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "dot1d_i");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter_1d);
    ctx_.builder().CreateBr(loop_cond_1d);

    ctx_.builder().SetInsertPoint(loop_cond_1d);
    llvm::Value* i_1d = ctx_.builder().CreateLoad(ctx_.int64Type(), counter_1d);
    llvm::Value* cond_1d = ctx_.builder().CreateICmpULT(i_1d, a_total);
    ctx_.builder().CreateCondBr(cond_1d, loop_body_1d, loop_exit_1d);

    ctx_.builder().SetInsertPoint(loop_body_1d);
    llvm::Value* a_elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_a_elements_ptr, i_1d);
    llvm::Value* b_elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_b_elements_ptr, i_1d);
    llvm::Value* a_elem = ctx_.builder().CreateLoad(ctx_.doubleType(), a_elem_ptr);
    llvm::Value* b_elem = ctx_.builder().CreateLoad(ctx_.doubleType(), b_elem_ptr);
    llvm::Value* product = ctx_.builder().CreateFMul(a_elem, b_elem);
    llvm::Value* old_sum = ctx_.builder().CreateLoad(ctx_.doubleType(), sum_alloca);
    llvm::Value* new_sum = ctx_.builder().CreateFAdd(old_sum, product);
    ctx_.builder().CreateStore(new_sum, sum_alloca);

    llvm::Value* next_i_1d = ctx_.builder().CreateAdd(i_1d, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i_1d, counter_1d);
    ctx_.builder().CreateBr(loop_cond_1d);

    ctx_.builder().SetInsertPoint(loop_exit_1d);
    llvm::Value* dot_result_1d = ctx_.builder().CreateLoad(ctx_.doubleType(), sum_alloca);
    ctx_.builder().CreateBr(tensor_merge);

    // 2D Matrix Multiplication: C = A @ B
    // A is (M x K), B is (K x N), C is (M x N)
    // C[i,j] = sum_k(A[i,k] * B[k,j])
    ctx_.builder().SetInsertPoint(dot_2d_block);

    // Get A dimensions: dims_a[0] = M (rows), dims_a[1] = K (cols)
    llvm::Value* a_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_a_ptr, 0);
    llvm::Value* a_dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), a_dims_field_ptr);
    llvm::Value* a_rows_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), a_dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* a_rows = ctx_.builder().CreateLoad(ctx_.int64Type(), a_rows_ptr);  // M
    llvm::Value* a_cols_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), a_dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* a_cols = ctx_.builder().CreateLoad(ctx_.int64Type(), a_cols_ptr);  // K

    // Get B dimensions: dims_b[0] = K (rows), dims_b[1] = N (cols)
    llvm::Value* b_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_b_ptr, 0);
    llvm::Value* b_dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), b_dims_field_ptr);
    llvm::Value* b_cols_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), b_dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* b_cols = ctx_.builder().CreateLoad(ctx_.int64Type(), b_cols_ptr);  // N

    // Result dimensions: M x N
    llvm::Value* c_rows = a_rows;  // M
    llvm::Value* c_cols = b_cols;  // N
    llvm::Value* c_total = ctx_.builder().CreateMul(c_rows, c_cols);

    // Allocate result tensor using arena
    llvm::Value* dot_arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Allocate tensor struct with header
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* c_tensor_ptr = ctx_.builder().CreateCall(alloc_tensor_func, {dot_arena_ptr}, "dot_tensor");

    // Allocate dims array (2 x 8 = 16 bytes)
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* c_dims_ptr = ctx_.builder().CreateCall(arena_alloc,
        {dot_arena_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 16)}, "dot_dims");
    llvm::Value* c_dims_0_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), c_dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateStore(c_rows, c_dims_0_ptr);
    llvm::Value* c_dims_1_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), c_dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(c_cols, c_dims_1_ptr);

    // Allocate elements array (c_total * 8 bytes for doubles)
    llvm::Value* c_elements_size = ctx_.builder().CreateMul(c_total,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* c_elements_ptr = ctx_.builder().CreateCall(arena_alloc, {dot_arena_ptr, c_elements_size}, "dot_elems");

    // Store tensor struct fields
    llvm::Value* c_dims_field = ctx_.builder().CreateStructGEP(tensor_type, c_tensor_ptr, 0);
    ctx_.builder().CreateStore(c_dims_ptr, c_dims_field);
    llvm::Value* c_ndims_field = ctx_.builder().CreateStructGEP(tensor_type, c_tensor_ptr, 1);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 2), c_ndims_field);
    llvm::Value* c_elems_field = ctx_.builder().CreateStructGEP(tensor_type, c_tensor_ptr, 2);
    ctx_.builder().CreateStore(c_elements_ptr, c_elems_field);
    llvm::Value* c_total_field = ctx_.builder().CreateStructGEP(tensor_type, c_tensor_ptr, 3);
    ctx_.builder().CreateStore(c_total, c_total_field);

    // Triple nested loop for matrix multiplication
    // for i in 0..M:
    //   for j in 0..N:
    //     sum = 0
    //     for k in 0..K:
    //       sum += A[i,k] * B[k,j]
    //     C[i,j] = sum

    llvm::Value* i_alloca = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "mm_i");
    llvm::Value* j_alloca = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "mm_j");
    llvm::Value* k_alloca = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "mm_k");
    llvm::Value* sum_alloca_2d = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "mm_sum");

    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_alloca);

    // Outer loop (i)
    llvm::BasicBlock* i_cond = llvm::BasicBlock::Create(ctx_.context(), "mm_i_cond", current_func);
    llvm::BasicBlock* i_body = llvm::BasicBlock::Create(ctx_.context(), "mm_i_body", current_func);
    llvm::BasicBlock* i_exit = llvm::BasicBlock::Create(ctx_.context(), "mm_i_exit", current_func);

    ctx_.builder().CreateBr(i_cond);

    ctx_.builder().SetInsertPoint(i_cond);
    llvm::Value* i_val = ctx_.builder().CreateLoad(ctx_.int64Type(), i_alloca);
    llvm::Value* i_lt = ctx_.builder().CreateICmpULT(i_val, c_rows);
    ctx_.builder().CreateCondBr(i_lt, i_body, i_exit);

    ctx_.builder().SetInsertPoint(i_body);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_alloca);

    // Middle loop (j)
    llvm::BasicBlock* j_cond = llvm::BasicBlock::Create(ctx_.context(), "mm_j_cond", current_func);
    llvm::BasicBlock* j_body = llvm::BasicBlock::Create(ctx_.context(), "mm_j_body", current_func);
    llvm::BasicBlock* j_exit = llvm::BasicBlock::Create(ctx_.context(), "mm_j_exit", current_func);

    ctx_.builder().CreateBr(j_cond);

    ctx_.builder().SetInsertPoint(j_cond);
    llvm::Value* j_val = ctx_.builder().CreateLoad(ctx_.int64Type(), j_alloca);
    llvm::Value* j_lt = ctx_.builder().CreateICmpULT(j_val, c_cols);
    ctx_.builder().CreateCondBr(j_lt, j_body, j_exit);

    ctx_.builder().SetInsertPoint(j_body);
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_alloca_2d);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_alloca);

    // Inner loop (k)
    llvm::BasicBlock* k_cond = llvm::BasicBlock::Create(ctx_.context(), "mm_k_cond", current_func);
    llvm::BasicBlock* k_body = llvm::BasicBlock::Create(ctx_.context(), "mm_k_body", current_func);
    llvm::BasicBlock* k_exit = llvm::BasicBlock::Create(ctx_.context(), "mm_k_exit", current_func);

    ctx_.builder().CreateBr(k_cond);

    ctx_.builder().SetInsertPoint(k_cond);
    llvm::Value* k_val = ctx_.builder().CreateLoad(ctx_.int64Type(), k_alloca);
    llvm::Value* k_lt = ctx_.builder().CreateICmpULT(k_val, a_cols);  // K = a_cols
    ctx_.builder().CreateCondBr(k_lt, k_body, k_exit);

    ctx_.builder().SetInsertPoint(k_body);
    // A[i,k] = A[i * K + k]
    llvm::Value* i_curr = ctx_.builder().CreateLoad(ctx_.int64Type(), i_alloca);
    llvm::Value* k_curr = ctx_.builder().CreateLoad(ctx_.int64Type(), k_alloca);
    llvm::Value* a_idx = ctx_.builder().CreateMul(i_curr, a_cols);
    a_idx = ctx_.builder().CreateAdd(a_idx, k_curr);
    llvm::Value* a_elem_ptr_2d = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_a_elements_ptr, a_idx);
    llvm::Value* a_elem_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), a_elem_ptr_2d);
    llvm::Value* a_elem_2d = ctx_.builder().CreateBitCast(a_elem_bits, ctx_.doubleType());

    // B[k,j] = B[k * N + j]
    llvm::Value* j_curr = ctx_.builder().CreateLoad(ctx_.int64Type(), j_alloca);
    llvm::Value* b_idx = ctx_.builder().CreateMul(k_curr, b_cols);
    b_idx = ctx_.builder().CreateAdd(b_idx, j_curr);
    llvm::Value* b_elem_ptr_2d = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_b_elements_ptr, b_idx);
    llvm::Value* b_elem_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), b_elem_ptr_2d);
    llvm::Value* b_elem_2d = ctx_.builder().CreateBitCast(b_elem_bits, ctx_.doubleType());

    // sum += A[i,k] * B[k,j]
    llvm::Value* prod_2d = ctx_.builder().CreateFMul(a_elem_2d, b_elem_2d);
    llvm::Value* old_sum_2d = ctx_.builder().CreateLoad(ctx_.doubleType(), sum_alloca_2d);
    llvm::Value* new_sum_2d = ctx_.builder().CreateFAdd(old_sum_2d, prod_2d);
    ctx_.builder().CreateStore(new_sum_2d, sum_alloca_2d);

    // k++
    llvm::Value* k_next = ctx_.builder().CreateAdd(k_curr, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(k_next, k_alloca);
    ctx_.builder().CreateBr(k_cond);

    // After k loop: store C[i,j] = sum
    ctx_.builder().SetInsertPoint(k_exit);
    llvm::Value* final_sum = ctx_.builder().CreateLoad(ctx_.doubleType(), sum_alloca_2d);
    // Reload i and j from allocas (k_body may not have executed if K=0)
    llvm::Value* i_for_store = ctx_.builder().CreateLoad(ctx_.int64Type(), i_alloca);
    llvm::Value* j_for_store = ctx_.builder().CreateLoad(ctx_.int64Type(), j_alloca);
    llvm::Value* c_idx = ctx_.builder().CreateMul(i_for_store, c_cols);
    c_idx = ctx_.builder().CreateAdd(c_idx, j_for_store);
    llvm::Value* c_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), c_elements_ptr, c_idx);
    llvm::Value* sum_bits = ctx_.builder().CreateBitCast(final_sum, ctx_.int64Type());
    ctx_.builder().CreateStore(sum_bits, c_elem_ptr);

    // j++
    llvm::Value* j_next = ctx_.builder().CreateAdd(j_for_store, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(j_next, j_alloca);
    ctx_.builder().CreateBr(j_cond);

    // After j loop
    ctx_.builder().SetInsertPoint(j_exit);
    // i++ - reload from alloca
    llvm::Value* i_for_inc = ctx_.builder().CreateLoad(ctx_.int64Type(), i_alloca);
    llvm::Value* i_next = ctx_.builder().CreateAdd(i_for_inc, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(i_next, i_alloca);
    ctx_.builder().CreateBr(i_cond);

    // After i loop: return tensor result as consolidated HEAP_PTR
    ctx_.builder().SetInsertPoint(i_exit);
    llvm::Value* tensor_result_2d = tagged_.packHeapPtr(c_tensor_ptr);
    ctx_.builder().CreateBr(final_merge);
    llvm::BasicBlock* dot_2d_exit = ctx_.builder().GetInsertBlock();

    // Create scalar_merge block for 1D scalar results (from both Scheme vec and 1D tensor)
    llvm::BasicBlock* scalar_merge = llvm::BasicBlock::Create(ctx_.context(), "scalar_merge", current_func);

    // Now add the deferred branch from svec_exit_block to scalar_merge
    ctx_.builder().SetInsertPoint(svec_exit_block);
    llvm::Value* svec_packed = tagged_.packDouble(svec_result);
    ctx_.builder().CreateBr(scalar_merge);

    // Tensor merge for 1D scalar path - pack and branch to scalar_merge
    ctx_.builder().SetInsertPoint(tensor_merge);
    llvm::Value* scalar_result_1d = tagged_.packDouble(dot_result_1d);
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
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "dot_result");
    result_phi->addIncoming(scalar_phi, scalar_merge_exit);
    result_phi->addIncoming(tensor_result_2d, dot_2d_exit);

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

    // Extract the tensor pointer from the tagged value
    llvm::Value* tensor_ptr_int = tagged_.safeExtractInt64(tensor_val);

    // Get function to apply - for now we'll support simple arithmetic functions
    eshkol_ast_t* func_ast = &op->call_op.variables[1];
    if (func_ast->type != ESHKOL_VAR) {
        eshkol_error("tensor-apply currently only supports simple function names");
        return nullptr;
    }

    std::string func_name = func_ast->variable.id;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());

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
    llvm::Value* result_elem = nullptr;
    if (func_name == "double") {
        result_elem = ctx_.builder().CreateMul(src_elem, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    } else if (func_name == "square") {
        result_elem = ctx_.builder().CreateMul(src_elem, src_elem);
    } else if (func_name == "increment") {
        result_elem = ctx_.builder().CreateAdd(src_elem, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    } else if (func_name == "negate") {
        result_elem = ctx_.builder().CreateNeg(src_elem);
    } else if (func_name == "abs") {
        // abs(x) = x < 0 ? -x : x
        llvm::Value* is_negative = ctx_.builder().CreateICmpSLT(src_elem, llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* negated = ctx_.builder().CreateNeg(src_elem);
        result_elem = ctx_.builder().CreateSelect(is_negative, negated, src_elem);
    } else if (func_name == "identity") {
        result_elem = src_elem;
    } else {
        eshkol_warn("Unknown function in tensor-apply: %s, using identity", func_name.c_str());
        result_elem = src_elem;
    }

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

    return ctx_.builder().CreatePtrToInt(typed_result_tensor_ptr, ctx_.int64Type());
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
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(src_val);
    llvm::Value* tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());

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
    if (op->call_op.num_vars != 4) {
        eshkol_error("tensor-reduce requires exactly 4 arguments: tensor, function, initial-value, dimension");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* initial_value = codegenAST(&op->call_op.variables[2]);
    llvm::Value* dimension_value = codegenAST(&op->call_op.variables[3]);
    if (!tensor_val || !initial_value || !dimension_value) return nullptr;

    // Extract the tensor pointer from the tagged value
    llvm::Value* tensor_ptr_int = tagged_.safeExtractInt64(tensor_val);

    // Get function to apply
    eshkol_ast_t* func_ast = &op->call_op.variables[1];
    if (func_ast->type != ESHKOL_VAR) {
        eshkol_error("tensor-reduce currently only supports simple function names");
        return nullptr;
    }

    std::string func_name = func_ast->variable.id;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());

    // Get source tensor properties
    llvm::Value* src_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* src_dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_dims_field_ptr);
    llvm::Value* typed_src_dims_ptr = ctx_.builder().CreatePointerCast(src_dims_ptr, ctx_.ptrType());

    llvm::Value* src_num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* src_num_dims = ctx_.builder().CreateLoad(ctx_.int64Type(), src_num_dims_field_ptr);

    llvm::Value* src_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_elements_field_ptr);
    llvm::Value* typed_src_elements_ptr = ctx_.builder().CreatePointerCast(src_elements_ptr, ctx_.ptrType());

    // Create result tensor with one less dimension
    // Allocate result tensor using arena
    llvm::Value* reduce_arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Allocate tensor struct with header
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* typed_result_tensor_ptr = ctx_.builder().CreateCall(alloc_tensor_func, {reduce_arena_ptr}, "reduce_tensor");

    llvm::Function* arena_alloc = mem_.getArenaAllocate();

    // Calculate result dimensions (all dimensions except the reduced one)
    llvm::Value* result_num_dims = ctx_.builder().CreateSub(src_num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));

    // Handle special case where result becomes scalar (0 dimensions)
    llvm::Value* is_scalar = ctx_.builder().CreateICmpEQ(result_num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* final_num_dims = ctx_.builder().CreateSelect(is_scalar, llvm::ConstantInt::get(ctx_.int64Type(), 1), result_num_dims);

    // Allocate result dimensions array using arena
    llvm::Value* result_dims_size = ctx_.builder().CreateMul(final_num_dims,
                                               llvm::ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t)));
    llvm::Value* result_dims_ptr = ctx_.builder().CreateCall(arena_alloc, {reduce_arena_ptr, result_dims_size}, "reduce_dims");
    llvm::Value* typed_result_dims_ptr = ctx_.builder().CreatePointerCast(result_dims_ptr, ctx_.ptrType());

    // For simplified implementation: create result with single dimension of size 1 (scalar result)
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), typed_result_dims_ptr);

    // Set result tensor properties
    llvm::Value* result_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 0);
    ctx_.builder().CreateStore(typed_result_dims_ptr, result_dims_field_ptr);

    llvm::Value* result_num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 1);
    ctx_.builder().CreateStore(final_num_dims, result_num_dims_field_ptr);

    llvm::Value* result_total_elements = llvm::ConstantInt::get(ctx_.int64Type(), 1);
    llvm::Value* result_total_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 3);
    ctx_.builder().CreateStore(result_total_elements, result_total_elements_field_ptr);

    // Allocate result elements array (single element for simplified version) using arena
    llvm::Value* result_elements_size = llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
    llvm::Value* result_elements_ptr = ctx_.builder().CreateCall(arena_alloc, {reduce_arena_ptr, result_elements_size}, "reduce_elems");
    llvm::Value* typed_result_elements_ptr = ctx_.builder().CreatePointerCast(result_elements_ptr, ctx_.ptrType());

    llvm::Value* result_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_result_tensor_ptr, 2);
    ctx_.builder().CreateStore(typed_result_elements_ptr, result_elements_field_ptr);

    // For now, implement a basic version that works for vectors (1D) and matrices (2D)
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Check which dimension we're reducing
    llvm::Value* dim_is_zero = ctx_.builder().CreateICmpEQ(dimension_value, llvm::ConstantInt::get(ctx_.int64Type(), 0));

    // Get matrix dimensions (assuming 2D for now)
    llvm::Value* dim0_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_src_dims_ptr,
                                        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* rows = ctx_.builder().CreateLoad(ctx_.int64Type(), dim0_ptr);

    llvm::Value* dim1_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_src_dims_ptr,
                                        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* cols = ctx_.builder().CreateLoad(ctx_.int64Type(), dim1_ptr);

    // Calculate result dimensions and size
    llvm::Value* result_rows = ctx_.builder().CreateSelect(dim_is_zero, llvm::ConstantInt::get(ctx_.int64Type(), 1), rows);
    llvm::Value* result_cols = ctx_.builder().CreateSelect(dim_is_zero, cols, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* result_elements = ctx_.builder().CreateMul(result_rows, result_cols);

    // Update result tensor dimensions
    ctx_.builder().CreateStore(result_rows, typed_result_dims_ptr);
    llvm::Value* result_dim1_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_result_dims_ptr,
                                                llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(result_cols, result_dim1_ptr);

    // Update result tensor total elements
    ctx_.builder().CreateStore(result_elements, result_total_elements_field_ptr);

    // Allocate result elements array using arena
    llvm::Value* result_elem_size = ctx_.builder().CreateMul(result_elements, llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* new_result_elements_ptr = ctx_.builder().CreateCall(arena_alloc, {reduce_arena_ptr, result_elem_size}, "reduce_elems2");
    llvm::Value* typed_new_result_elements_ptr = ctx_.builder().CreatePointerCast(new_result_elements_ptr, ctx_.ptrType());
    ctx_.builder().CreateStore(typed_new_result_elements_ptr, result_elements_field_ptr);

    // Create loops based on dimension
    llvm::BasicBlock* outer_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "dim_outer_cond", current_func);
    llvm::BasicBlock* outer_loop_body = llvm::BasicBlock::Create(ctx_.context(), "dim_outer_body", current_func);
    llvm::BasicBlock* inner_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "dim_inner_cond", current_func);
    llvm::BasicBlock* inner_loop_body = llvm::BasicBlock::Create(ctx_.context(), "dim_inner_body", current_func);
    llvm::BasicBlock* inner_loop_exit = llvm::BasicBlock::Create(ctx_.context(), "dim_inner_exit", current_func);
    llvm::BasicBlock* outer_loop_exit = llvm::BasicBlock::Create(ctx_.context(), "dim_outer_exit", current_func);

    // Initialize result index counter
    llvm::Value* result_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "result_idx");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), result_idx);

    // Initialize outer loop counter
    llvm::Value* outer_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "outer_counter");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), outer_counter);

    // Jump to outer loop
    ctx_.builder().CreateBr(outer_loop_cond);

    // Outer loop condition
    ctx_.builder().SetInsertPoint(outer_loop_cond);
    llvm::Value* current_outer = ctx_.builder().CreateLoad(ctx_.int64Type(), outer_counter);
    llvm::Value* outer_limit = ctx_.builder().CreateSelect(dim_is_zero, cols, rows);
    llvm::Value* outer_cmp = ctx_.builder().CreateICmpULT(current_outer, outer_limit);
    ctx_.builder().CreateCondBr(outer_cmp, outer_loop_body, outer_loop_exit);

    // Outer loop body: initialize accumulator for this dimension
    ctx_.builder().SetInsertPoint(outer_loop_body);
    llvm::Value* dim_accumulator = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "dim_acc");
    ctx_.builder().CreateStore(initial_value, dim_accumulator);

    // Initialize inner loop counter
    llvm::Value* inner_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "inner_counter");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), inner_counter);

    // Jump to inner loop
    ctx_.builder().CreateBr(inner_loop_cond);

    // Inner loop condition
    ctx_.builder().SetInsertPoint(inner_loop_cond);
    llvm::Value* current_inner = ctx_.builder().CreateLoad(ctx_.int64Type(), inner_counter);
    llvm::Value* inner_limit = ctx_.builder().CreateSelect(dim_is_zero, rows, cols);
    llvm::Value* inner_cmp = ctx_.builder().CreateICmpULT(current_inner, inner_limit);
    ctx_.builder().CreateCondBr(inner_cmp, inner_loop_body, inner_loop_exit);

    // Inner loop body: calculate element index and apply reduction
    ctx_.builder().SetInsertPoint(inner_loop_body);

    // Calculate source element index: row * cols + col
    llvm::Value* src_row = ctx_.builder().CreateSelect(dim_is_zero, current_inner, current_outer);
    llvm::Value* src_col = ctx_.builder().CreateSelect(dim_is_zero, current_outer, current_inner);
    llvm::Value* src_linear_idx = ctx_.builder().CreateMul(src_row, cols);
    src_linear_idx = ctx_.builder().CreateAdd(src_linear_idx, src_col);

    // Load source element
    llvm::Value* src_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_src_elements_ptr, src_linear_idx);
    llvm::Value* src_elem = ctx_.builder().CreateLoad(ctx_.int64Type(), src_elem_ptr);

    // Load current accumulator
    llvm::Value* current_acc = ctx_.builder().CreateLoad(ctx_.int64Type(), dim_accumulator);

    // Apply reduction function
    llvm::Value* new_acc = nullptr;
    if (func_name == "+") {
        new_acc = ctx_.builder().CreateAdd(current_acc, src_elem);
    } else if (func_name == "*") {
        new_acc = ctx_.builder().CreateMul(current_acc, src_elem);
    } else if (func_name == "max") {
        llvm::Value* cmp = ctx_.builder().CreateICmpSGT(current_acc, src_elem);
        new_acc = ctx_.builder().CreateSelect(cmp, current_acc, src_elem);
    } else if (func_name == "min") {
        llvm::Value* cmp = ctx_.builder().CreateICmpSLT(current_acc, src_elem);
        new_acc = ctx_.builder().CreateSelect(cmp, current_acc, src_elem);
    } else if (func_name == "mean") {
        new_acc = ctx_.builder().CreateAdd(current_acc, src_elem);
    } else {
        eshkol_warn("Unknown reduction function: %s, using addition", func_name.c_str());
        new_acc = ctx_.builder().CreateAdd(current_acc, src_elem);
    }

    // Store updated accumulator
    ctx_.builder().CreateStore(new_acc, dim_accumulator);

    // Increment inner counter
    llvm::Value* next_inner = ctx_.builder().CreateAdd(current_inner, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_inner, inner_counter);

    // Jump back to inner condition
    ctx_.builder().CreateBr(inner_loop_cond);

    // Inner loop exit: store result and move to next outer iteration
    ctx_.builder().SetInsertPoint(inner_loop_exit);
    llvm::Value* final_acc = ctx_.builder().CreateLoad(ctx_.int64Type(), dim_accumulator);

    // For mean, divide by the dimension size
    if (func_name == "mean") {
        final_acc = ctx_.builder().CreateSDiv(final_acc, inner_limit);
    }

    // Store result in result array
    llvm::Value* current_result_idx = ctx_.builder().CreateLoad(ctx_.int64Type(), result_idx);
    llvm::Value* result_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_new_result_elements_ptr, current_result_idx);
    ctx_.builder().CreateStore(final_acc, result_elem_ptr);

    // Increment result index and outer counter
    llvm::Value* next_result_idx = ctx_.builder().CreateAdd(current_result_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_result_idx, result_idx);

    llvm::Value* next_outer = ctx_.builder().CreateAdd(current_outer, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_outer, outer_counter);

    // Jump back to outer condition
    ctx_.builder().CreateBr(outer_loop_cond);

    // Outer loop exit
    ctx_.builder().SetInsertPoint(outer_loop_exit);

    return ctx_.builder().CreatePtrToInt(typed_result_tensor_ptr, ctx_.int64Type());
}

llvm::Value* TensorCodegen::tensorSum(const eshkol_operations_t* op) {
    // tensor-sum: (tensor-sum tensor) - Sum all elements
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("tensor-sum requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* src_val = codegenAST(&op->call_op.variables[0]);
    if (!src_val) return nullptr;

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

    llvm::Value* svec_sum = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "svec_sum_acc");
    llvm::Value* svec_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "svec_sum_i");
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
    llvm::Value* svec_result = ctx_.builder().CreateLoad(ctx_.doubleType(), svec_sum);
    ctx_.builder().CreateBr(sum_merge);
    llvm::BasicBlock* svec_exit_block = ctx_.builder().GetInsertBlock();

    // === TENSOR PATH ===
    ctx_.builder().SetInsertPoint(tensor_block);
    llvm::Value* src_ptr_int = tagged_.unpackInt64(src_val);
    llvm::Value* src_ptr = ctx_.builder().CreateIntToPtr(src_ptr_int, ctx_.ptrType());

    llvm::Value* src_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 2);
    llvm::Value* src_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_elements_field_ptr);
    llvm::Value* typed_src_elements = ctx_.builder().CreatePointerCast(src_elements_ptr, ctx_.ptrType());

    llvm::Value* src_total_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 3);
    llvm::Value* src_total = ctx_.builder().CreateLoad(ctx_.int64Type(), src_total_field_ptr);

    // Sum all elements
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "tsum_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "tsum_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "tsum_exit", current_func);

    llvm::Value* sum = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "sum_acc");
    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "sum_i");
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = ctx_.builder().CreateICmpULT(i, src_total);
    ctx_.builder().CreateCondBr(cmp, loop_body, loop_exit);

    ctx_.builder().SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_src_elements, i);
    llvm::Value* elem_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem_val = ctx_.builder().CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* current_sum = ctx_.builder().CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* new_sum = ctx_.builder().CreateFAdd(current_sum, elem_val);
    ctx_.builder().CreateStore(new_sum, sum);

    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_exit);
    llvm::Value* tensor_result = ctx_.builder().CreateLoad(ctx_.doubleType(), sum);
    ctx_.builder().CreateBr(sum_merge);
    llvm::BasicBlock* tensor_exit_block = ctx_.builder().GetInsertBlock();

    // === MERGE RESULTS ===
    ctx_.builder().SetInsertPoint(sum_merge);
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "sum_result");
    result_phi->addIncoming(svec_result, svec_exit_block);
    result_phi->addIncoming(tensor_result, tensor_exit_block);

    return tagged_.packDouble(result_phi);
}

llvm::Value* TensorCodegen::tensorMean(const eshkol_operations_t* op) {
    // tensor-mean: (tensor-mean tensor) - Mean of all elements
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("tensor-mean requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* src_val = codegenAST(&op->call_op.variables[0]);
    if (!src_val) return nullptr;

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
    ctx_.builder().CreateBr(mean_merge);
    llvm::BasicBlock* svec_exit_block = ctx_.builder().GetInsertBlock();

    // === TENSOR PATH ===
    ctx_.builder().SetInsertPoint(tensor_block);
    llvm::Value* src_ptr_int = tagged_.unpackInt64(src_val);
    llvm::Value* src_ptr = ctx_.builder().CreateIntToPtr(src_ptr_int, ctx_.ptrType());

    llvm::Value* src_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 2);
    llvm::Value* src_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_elements_field_ptr);
    llvm::Value* typed_src_elements = ctx_.builder().CreatePointerCast(src_elements_ptr, ctx_.ptrType());

    llvm::Value* src_total_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 3);
    llvm::Value* src_total = ctx_.builder().CreateLoad(ctx_.int64Type(), src_total_field_ptr);

    // Sum all elements
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "tmean_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "tmean_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "tmean_exit", current_func);

    llvm::Value* sum = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "mean_acc");
    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "mean_i");
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = ctx_.builder().CreateICmpULT(i, src_total);
    ctx_.builder().CreateCondBr(cmp, loop_body, loop_exit);

    ctx_.builder().SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_src_elements, i);
    llvm::Value* elem_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem_val = ctx_.builder().CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* current_sum = ctx_.builder().CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* new_sum = ctx_.builder().CreateFAdd(current_sum, elem_val);
    ctx_.builder().CreateStore(new_sum, sum);

    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_exit);
    llvm::Value* total_sum = ctx_.builder().CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* count_fp = ctx_.builder().CreateSIToFP(src_total, ctx_.doubleType());
    llvm::Value* tensor_result = ctx_.builder().CreateFDiv(total_sum, count_fp);
    ctx_.builder().CreateBr(mean_merge);
    llvm::BasicBlock* tensor_exit_block = ctx_.builder().GetInsertBlock();

    // === MERGE RESULTS ===
    ctx_.builder().SetInsertPoint(mean_merge);
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "mean_result");
    result_phi->addIncoming(svec_result, svec_exit_block);
    result_phi->addIncoming(tensor_result, tensor_exit_block);

    return tagged_.packDouble(result_phi);
}

llvm::Value* TensorCodegen::tensorShape(const eshkol_operations_t* op) {
    // tensor-shape: (tensor-shape tensor) -> returns dimensions as a Scheme list
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-shape requires exactly 1 tensor argument");
        return nullptr;
    }

    llvm::Value* tensor_ptr_int = codegenAST(&op->call_op.variables[0]);
    if (!tensor_ptr_int) return nullptr;

    // Extract the raw pointer value if it's a tagged value
    if (tensor_ptr_int->getType() == ctx_.taggedValueType()) {
        tensor_ptr_int = tagged_.unpackInt64(tensor_ptr_int);
    }

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());

    // Load num_dimensions
    llvm::Value* num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = ctx_.builder().CreateLoad(ctx_.int64Type(), num_dims_field_ptr);

    // Load dimensions array pointer
    llvm::Value* dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), dims_field_ptr);

    // Build a proper cons-based list from dimensions (build from end to front)
    // Start with null (empty list) and prepend each dimension
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::GlobalVariable* arena_global = ctx_.globalArena();

    if (!arena_global) {
        eshkol_error("tensor-shape requires arena for list allocation");
        return tagged_.packNull();
    }
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), arena_global);

    // Create alloca at function entry for the accumulator
    llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock& entry = current_func->getEntryBlock();
    llvm::IRBuilder<> entry_builder(&entry, entry.getFirstInsertionPt());
    llvm::Value* result_alloca = entry_builder.CreateAlloca(ctx_.int64Type(), nullptr, "shape_result");
    llvm::Value* counter_alloca = entry_builder.CreateAlloca(ctx_.int64Type(), nullptr, "shape_i");

    ctx_.builder().SetInsertPoint(current_block);

    // Initialize: result = 0 (null), counter = num_dims - 1 (iterate backwards)
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), result_alloca);
    llvm::Value* start_idx = ctx_.builder().CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(start_idx, counter_alloca);

    // Loop condition
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "shape_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "shape_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "shape_exit", current_func);

    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter_alloca);
    // Loop while i >= 0
    llvm::Value* cond = ctx_.builder().CreateICmpSGE(i, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(cond, loop_body, loop_exit);

    ctx_.builder().SetInsertPoint(loop_body);

    // Load dimension value at index i
    llvm::Value* dim_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), dims_ptr, i);
    llvm::Value* dim_val = ctx_.builder().CreateLoad(ctx_.int64Type(), dim_ptr);

    // Load current list tail
    llvm::Value* current_tail_int = ctx_.builder().CreateLoad(ctx_.int64Type(), result_alloca);

    // Allocate new cons cell with object header (consolidated pointer format)
    // M1 Migration: Use header allocator for HEAP_PTR compatibility
    llvm::Value* cons_cell = ctx_.builder().CreateCall(
        mem_.getArenaAllocateConsWithHeader(), {arena_ptr});

    // Set car to dimension value (int64)
    llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    llvm::Value* int_type = llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64);
    ctx_.builder().CreateCall(mem_.getTaggedConsSetInt64(),
        {cons_cell, is_car, dim_val, int_type});

    // Set cdr to current tail
    llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 1);
    llvm::Value* is_null_tail = ctx_.builder().CreateICmpEQ(current_tail_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));

    // Branch for null vs non-null cdr
    llvm::BasicBlock* set_null_cdr = llvm::BasicBlock::Create(ctx_.context(), "set_null_cdr", current_func);
    llvm::BasicBlock* set_cons_cdr = llvm::BasicBlock::Create(ctx_.context(), "set_cons_cdr", current_func);
    llvm::BasicBlock* cdr_done = llvm::BasicBlock::Create(ctx_.context(), "cdr_done", current_func);

    ctx_.builder().CreateCondBr(is_null_tail, set_null_cdr, set_cons_cdr);

    ctx_.builder().SetInsertPoint(set_null_cdr);
    ctx_.builder().CreateCall(mem_.getTaggedConsSetNull(), {cons_cell, is_cdr});
    ctx_.builder().CreateBr(cdr_done);

    ctx_.builder().SetInsertPoint(set_cons_cdr);
    llvm::Value* cons_type = llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateCall(mem_.getTaggedConsSetPtr(),
        {cons_cell, is_cdr, current_tail_int, cons_type});
    ctx_.builder().CreateBr(cdr_done);

    ctx_.builder().SetInsertPoint(cdr_done);

    // Update result to point to new cons cell
    llvm::Value* cons_cell_int = ctx_.builder().CreatePtrToInt(cons_cell, ctx_.int64Type());
    ctx_.builder().CreateStore(cons_cell_int, result_alloca);

    // Decrement counter
    llvm::Value* next_i = ctx_.builder().CreateSub(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, counter_alloca);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_exit);

    // Load final result
    llvm::Value* final_result_int = ctx_.builder().CreateLoad(ctx_.int64Type(), result_alloca);
    return tagged_.packHeapPtr(ctx_.builder().CreateIntToPtr(final_result_int, ctx_.ptrType()));
}

llvm::Value* TensorCodegen::transpose(const eshkol_operations_t* op) {
    // transpose: (transpose tensor) - Transpose 2D matrix (swap rows and cols)
    if (op->call_op.num_vars != 1) {
        eshkol_error("transpose requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* src_tensor = codegenAST(&op->call_op.variables[0]);
    if (!src_tensor) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Check type - transpose only works with native tensors, not Scheme vectors
    llvm::Value* is_tensor = tagged_.isTensor(src_tensor);

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* tensor_block = llvm::BasicBlock::Create(ctx_.context(), "transpose_tensor", current_func);
    llvm::BasicBlock* error_block = llvm::BasicBlock::Create(ctx_.context(), "transpose_error", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "transpose_exit", current_func);

    ctx_.builder().CreateCondBr(is_tensor, tensor_block, error_block);

    // Error path - return null for non-tensor inputs
    ctx_.builder().SetInsertPoint(error_block);
    llvm::Value* error_result = tagged_.packNull();
    ctx_.builder().CreateBr(exit_block);
    llvm::BasicBlock* error_exit = ctx_.builder().GetInsertBlock();

    // Tensor path - proceed with normal transpose
    ctx_.builder().SetInsertPoint(tensor_block);
    llvm::Value* ptr_int = tagged_.unpackInt64(src_tensor);
    llvm::Value* src_ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());

    // Get source tensor properties
    llvm::Value* src_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 0);
    llvm::Value* src_dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_dims_field_ptr);
    llvm::Value* typed_src_dims_ptr = ctx_.builder().CreatePointerCast(src_dims_ptr, ctx_.ptrType());

    llvm::Value* src_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 2);
    llvm::Value* src_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_elements_field_ptr);
    llvm::Value* typed_src_elements_ptr = ctx_.builder().CreatePointerCast(src_elements_ptr, ctx_.ptrType());

    // Get rows and cols
    llvm::Value* rows = ctx_.builder().CreateLoad(ctx_.int64Type(), typed_src_dims_ptr);
    llvm::Value* dim1_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_src_dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* cols = ctx_.builder().CreateLoad(ctx_.int64Type(), dim1_ptr);

    // Create result tensor with swapped dimensions [cols, rows]
    std::vector<llvm::Value*> new_dims = {cols, rows};
    llvm::Value* result_ptr = createTensorWithDims(new_dims, nullptr, false);
    if (!result_ptr) return nullptr;

    // Get result elements pointer
    llvm::Value* result_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, result_ptr, 2);
    llvm::Value* result_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), result_elements_field_ptr);
    llvm::Value* typed_result_elements_ptr = ctx_.builder().CreatePointerCast(result_elements_ptr, ctx_.ptrType());

    // Transpose: result[j][i] = src[i][j]
    llvm::BasicBlock* row_cond = llvm::BasicBlock::Create(ctx_.context(), "trans_row_cond", current_func);
    llvm::BasicBlock* row_body = llvm::BasicBlock::Create(ctx_.context(), "trans_row_body", current_func);
    llvm::BasicBlock* col_cond = llvm::BasicBlock::Create(ctx_.context(), "trans_col_cond", current_func);
    llvm::BasicBlock* col_body = llvm::BasicBlock::Create(ctx_.context(), "trans_col_body", current_func);
    llvm::BasicBlock* col_exit = llvm::BasicBlock::Create(ctx_.context(), "trans_col_exit", current_func);
    llvm::BasicBlock* row_exit = llvm::BasicBlock::Create(ctx_.context(), "trans_row_exit", current_func);

    llvm::Value* row_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "trans_i");
    llvm::Value* col_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "trans_j");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), row_counter);
    ctx_.builder().CreateBr(row_cond);

    ctx_.builder().SetInsertPoint(row_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), row_counter);
    llvm::Value* row_cmp = ctx_.builder().CreateICmpULT(i, rows);
    ctx_.builder().CreateCondBr(row_cmp, row_body, row_exit);

    ctx_.builder().SetInsertPoint(row_body);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), col_counter);
    ctx_.builder().CreateBr(col_cond);

    ctx_.builder().SetInsertPoint(col_cond);
    llvm::Value* j = ctx_.builder().CreateLoad(ctx_.int64Type(), col_counter);
    llvm::Value* col_cmp = ctx_.builder().CreateICmpULT(j, cols);
    ctx_.builder().CreateCondBr(col_cmp, col_body, col_exit);

    ctx_.builder().SetInsertPoint(col_body);
    // src_idx = i * cols + j
    llvm::Value* src_idx = ctx_.builder().CreateMul(i, cols);
    src_idx = ctx_.builder().CreateAdd(src_idx, j);
    // dst_idx = j * rows + i
    llvm::Value* dst_idx = ctx_.builder().CreateMul(j, rows);
    dst_idx = ctx_.builder().CreateAdd(dst_idx, i);

    llvm::Value* src_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_src_elements_ptr, src_idx);
    llvm::Value* elem = ctx_.builder().CreateLoad(ctx_.int64Type(), src_elem_ptr);
    llvm::Value* dst_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_result_elements_ptr, dst_idx);
    ctx_.builder().CreateStore(elem, dst_elem_ptr);

    llvm::Value* next_j = ctx_.builder().CreateAdd(j, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_j, col_counter);
    ctx_.builder().CreateBr(col_cond);

    ctx_.builder().SetInsertPoint(col_exit);
    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, row_counter);
    ctx_.builder().CreateBr(row_cond);

    ctx_.builder().SetInsertPoint(row_exit);
    llvm::Value* tensor_result = tagged_.packHeapPtr(result_ptr);
    ctx_.builder().CreateBr(exit_block);
    llvm::BasicBlock* tensor_exit = ctx_.builder().GetInsertBlock();

    // Merge results
    ctx_.builder().SetInsertPoint(exit_block);
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "transpose_result");
    result_phi->addIncoming(error_result, error_exit);
    result_phi->addIncoming(tensor_result, tensor_exit);

    return result_phi;
}

llvm::Value* TensorCodegen::reshape(const eshkol_operations_t* op) {
    // reshape: (reshape tensor dim1 dim2 ...) OR (reshape tensor (list d1 d2 ...))
    // Support both individual dimension args and a list of dimensions
    if (op->call_op.num_vars < 2) {
        eshkol_error("reshape requires tensor and at least 1 dimension");
        return nullptr;
    }

    // Get source tensor
    llvm::Value* src_tensor = codegenAST(&op->call_op.variables[0]);
    if (!src_tensor) return nullptr;

    // Extract tensor pointer
    llvm::Value* src_ptr;
    if (src_tensor->getType() == ctx_.taggedValueType()) {
        llvm::Value* ptr_int = tagged_.unpackInt64(src_tensor);
        src_ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());
    } else {
        src_ptr = ctx_.builder().CreateIntToPtr(src_tensor, ctx_.ptrType());
    }

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get new dimensions - need to handle two cases:
    // 1. Individual args: (reshape tensor 3 3) -> num_vars > 2
    // 2. List arg: (reshape tensor (list 3 3)) -> num_vars == 2 and arg is CONS_PTR

    std::vector<llvm::Value*> new_dims;

    if (op->call_op.num_vars == 2) {
        // Could be a single dimension OR a list of dimensions
        llvm::Value* dim_arg = codegenAST(&op->call_op.variables[1]);
        if (!dim_arg) return nullptr;

        // Check if it's a list (CONS_PTR or HEAP_PTR) at runtime
        llvm::Value* type_tag = tagged_.getType(dim_arg);
        llvm::Value* base_type = tagged_.getBaseType(type_tag);
        llvm::Value* is_cons_legacy = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        llvm::Value* is_list = ctx_.builder().CreateOr(is_cons_legacy, is_heap_ptr);

        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* list_path = llvm::BasicBlock::Create(ctx_.context(), "reshape_list", current_func);
        llvm::BasicBlock* single_path = llvm::BasicBlock::Create(ctx_.context(), "reshape_single", current_func);
        llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "reshape_merge", current_func);

        ctx_.builder().CreateCondBr(is_list, list_path, single_path);

        // LIST PATH: Extract dimensions from the list (handling 2D for now)
        ctx_.builder().SetInsertPoint(list_path);
        llvm::Value* list_ptr_int = tagged_.unpackInt64(dim_arg);

        // Get cons cell pointer
        llvm::Value* cons_ptr = ctx_.builder().CreateIntToPtr(list_ptr_int, ctx_.ptrType());
        llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);  // false = car
        llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 1);  // true = cdr

        // Extract first element (car) - use arena_tagged_cons_get_int64
        llvm::Value* dim1_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cons_ptr, is_car});

        // Get cdr (rest of list) - use arena_tagged_cons_get_ptr
        llvm::Value* cdr_ptr_int = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr});

        // Extract second element (cadr) from the cdr cons cell
        llvm::Value* cdr_cons_ptr = ctx_.builder().CreateIntToPtr(cdr_ptr_int, ctx_.ptrType());
        llvm::Value* dim2_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cdr_cons_ptr, is_car});

        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_exit = ctx_.builder().GetInsertBlock();

        // SINGLE PATH: Use the value as a single dimension
        ctx_.builder().SetInsertPoint(single_path);
        llvm::Value* single_dim = dim_arg;
        if (single_dim->getType() == ctx_.taggedValueType()) {
            single_dim = tagged_.unpackInt64(single_dim);
        }
        // For single dimension, treat as 1D reshape (dim2 = 1 as placeholder, but only 1 dim used)
        llvm::Value* dim1_from_single = single_dim;
        llvm::Value* dim2_from_single = llvm::ConstantInt::get(ctx_.int64Type(), 1);
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* single_exit = ctx_.builder().GetInsertBlock();

        // MERGE: Use phi nodes to get final dimensions
        ctx_.builder().SetInsertPoint(merge_block);

        llvm::PHINode* is_2d_phi = ctx_.builder().CreatePHI(ctx_.builder().getInt1Ty(), 2, "is_2d");
        is_2d_phi->addIncoming(ctx_.builder().getTrue(), list_exit);
        is_2d_phi->addIncoming(ctx_.builder().getFalse(), single_exit);

        llvm::PHINode* final_dim1_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "final_dim1");
        final_dim1_phi->addIncoming(dim1_from_list, list_exit);
        final_dim1_phi->addIncoming(dim1_from_single, single_exit);

        llvm::PHINode* final_dim2_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "final_dim2");
        final_dim2_phi->addIncoming(dim2_from_list, list_exit);
        final_dim2_phi->addIncoming(dim2_from_single, single_exit);

        new_dims.push_back(final_dim1_phi);
        // Always push dim2 - for single dimension case it's 1, for list case it's the actual value
        new_dims.push_back(final_dim2_phi);
    } else {
        // Multiple explicit dimension arguments
        for (uint64_t i = 1; i < op->call_op.num_vars; i++) {
            llvm::Value* dim = codegenAST(&op->call_op.variables[i]);
            if (!dim) return nullptr;
            if (dim->getType() == ctx_.taggedValueType()) {
                dim = tagged_.unpackInt64(dim);
            }
            new_dims.push_back(dim);
        }
    }

    // Get source tensor properties
    llvm::Value* src_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 2);
    llvm::Value* src_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_elements_field_ptr);
    llvm::Value* src_total_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 3);
    llvm::Value* src_total = ctx_.builder().CreateLoad(ctx_.int64Type(), src_total_field_ptr);

    // Allocate using arena
    llvm::Value* reshape_arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Create new tensor structure with header (reuse elements - no copy needed for reshape)
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* typed_new_tensor_ptr = ctx_.builder().CreateCall(alloc_tensor_func, {reshape_arena_ptr}, "reshape_tensor");

    // Allocate new dimensions array using arena
    llvm::Value* dims_size = llvm::ConstantInt::get(ctx_.int64Type(), new_dims.size() * sizeof(uint64_t));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* dims_ptr = ctx_.builder().CreateCall(arena_alloc, {reshape_arena_ptr, dims_size}, "reshape_dims");
    llvm::Value* typed_dims_ptr = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.ptrType());

    // Store new dimensions and compute new total
    llvm::Value* new_total = new_dims[0];
    for (size_t i = 0; i < new_dims.size(); i++) {
        llvm::Value* dim_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        ctx_.builder().CreateStore(new_dims[i], dim_ptr);
        if (i > 0) {
            new_total = ctx_.builder().CreateMul(new_total, new_dims[i]);
        }
    }

    // Store tensor fields (reuse source elements pointer - zero-copy reshape)
    llvm::Value* dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 0);
    ctx_.builder().CreateStore(typed_dims_ptr, dims_field_ptr);

    llvm::Value* num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 1);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), new_dims.size()), num_dims_field_ptr);

    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 2);
    ctx_.builder().CreateStore(src_elements_ptr, elements_field_ptr);  // Reuse source elements!

    llvm::Value* total_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 3);
    ctx_.builder().CreateStore(new_total, total_elements_field_ptr);

    // Pack as consolidated HEAP_PTR
    return tagged_.packHeapPtr(typed_new_tensor_ptr);
}

// ===== TENSOR CREATION HELPER =====

llvm::Value* TensorCodegen::createTensorWithDims(const std::vector<llvm::Value*>& dims,
                                                   llvm::Value* fill_value,
                                                   bool use_memset_zero) {
    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get arena pointer
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(context, 0), ctx_.globalArena());

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Calculate total elements
    llvm::Value* total_elements = dims[0];
    for (size_t i = 1; i < dims.size(); i++) {
        total_elements = builder.CreateMul(total_elements, dims[i]);
    }

    // Allocate tensor structure with header using arena
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* typed_tensor_ptr = builder.CreateCall(alloc_tensor_func, {arena_ptr}, "new_tensor");

    // Allocate dimensions array using arena
    llvm::Value* dims_size = llvm::ConstantInt::get(ctx_.int64Type(), dims.size() * sizeof(uint64_t));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* dims_ptr = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "new_dims");
    llvm::Value* typed_dims_ptr = builder.CreatePointerCast(dims_ptr, ctx_.ptrType());

    // Store dimensions
    for (size_t i = 0; i < dims.size(); i++) {
        llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), typed_dims_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        builder.CreateStore(dims[i], dim_ptr);
    }

    // Allocate elements array using arena
    llvm::Value* elements_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* elements_ptr = builder.CreateCall(arena_alloc, {arena_ptr, elements_size}, "new_elems");
    llvm::Value* typed_elements_ptr = builder.CreatePointerCast(elements_ptr, ctx_.ptrType());

    // Fill elements if requested
    if (use_memset_zero) {
        // Use memset for efficient zero-fill
        llvm::Function* memset_func = ctx_.module().getFunction("memset");
        if (!memset_func) {
            // Declare memset: void* memset(void*, int, size_t)
            llvm::FunctionType* memset_type = llvm::FunctionType::get(
                ctx_.ptrType(),
                {ctx_.ptrType(), ctx_.int32Type(), ctx_.int64Type()},
                false);
            memset_func = llvm::Function::Create(
                memset_type, llvm::Function::ExternalLinkage, "memset", &ctx_.module());
        }
        ctx_.builder().CreateCall(memset_func, {
            typed_elements_ptr,
            llvm::ConstantInt::get(ctx_.int32Type(), 0),
            elements_size
        });
    } else if (fill_value) {
        // Fill with specified value using loop
        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "fill_cond", current_func);
        llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "fill_body", current_func);
        llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "fill_exit", current_func);

        llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "fill_i");
        ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
        ctx_.builder().CreateBr(loop_cond);

        ctx_.builder().SetInsertPoint(loop_cond);
        llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
        llvm::Value* cmp = ctx_.builder().CreateICmpULT(i, total_elements);
        ctx_.builder().CreateCondBr(cmp, loop_body, loop_exit);

        ctx_.builder().SetInsertPoint(loop_body);
        llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements_ptr, i);
        ctx_.builder().CreateStore(fill_value, elem_ptr);
        llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        ctx_.builder().CreateStore(next_i, counter);
        ctx_.builder().CreateBr(loop_cond);

        ctx_.builder().SetInsertPoint(loop_exit);
    }

    // Store tensor fields
    llvm::Value* dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_tensor_ptr, 0);
    ctx_.builder().CreateStore(typed_dims_ptr, dims_field_ptr);

    llvm::Value* num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_tensor_ptr, 1);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), dims.size()), num_dims_field_ptr);

    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_tensor_ptr, 2);
    ctx_.builder().CreateStore(typed_elements_ptr, elements_field_ptr);

    llvm::Value* total_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_tensor_ptr, 3);
    ctx_.builder().CreateStore(total_elements, total_elements_field_ptr);

    return typed_tensor_ptr;
}

// ===== TENSOR CREATION FUNCTIONS =====

llvm::Value* TensorCodegen::zeros(const eshkol_operations_t* op) {
    // zeros: (zeros dim1 dim2 ...) OR (zeros '(dim1 dim2 ...)) - Create tensor filled with zeros
    // Supports both NumPy-style list syntax and separate dimension arguments
    if (op->call_op.num_vars < 1) {
        eshkol_error("zeros requires at least 1 dimension argument");
        return nullptr;
    }

    std::vector<llvm::Value*> dims;

    if (op->call_op.num_vars == 1) {
        // Could be a single dimension OR a list of dimensions
        llvm::Value* dim_arg = codegenAST(&op->call_op.variables[0]);
        if (!dim_arg) return nullptr;

        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* list_path = llvm::BasicBlock::Create(ctx_.context(), "zeros_list", current_func);
        llvm::BasicBlock* single_path = llvm::BasicBlock::Create(ctx_.context(), "zeros_single", current_func);
        llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "zeros_merge", current_func);

        // Check if it's a list (CONS_PTR or HEAP_PTR) at runtime
        // IMPORTANT: Only check type if dim_arg is actually a tagged_value struct
        // If it's a raw i64, it's definitely not a list
        if (dim_arg->getType() == ctx_.taggedValueType()) {
            llvm::Value* type_tag = tagged_.getType(dim_arg);
            // Use getBaseType() to properly handle legacy types (CONS_PTR=32)
            // DO NOT use 0x0F mask - 32 & 0x0F = 0 (NULL) which is WRONG!
            llvm::Value* base_type = tagged_.getBaseType(type_tag);
            llvm::Value* is_cons_legacy = ctx_.builder().CreateICmpEQ(base_type,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
            llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(base_type,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
            llvm::Value* is_list = ctx_.builder().CreateOr(is_cons_legacy, is_heap_ptr);
            ctx_.builder().CreateCondBr(is_list, list_path, single_path);
        } else {
            // Raw integer - definitely not a list, go directly to single_path
            ctx_.builder().CreateBr(single_path);
        }

        // LIST PATH: Extract dimensions from the list (up to 4D supported)
        ctx_.builder().SetInsertPoint(list_path);
        llvm::Value* list_ptr_int = tagged_.unpackInt64(dim_arg);
        llvm::Value* cons_ptr = ctx_.builder().CreateIntToPtr(list_ptr_int, ctx_.ptrType());
        llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
        llvm::Value* is_cdr_flag = llvm::ConstantInt::get(ctx_.int1Type(), 1);

        // Extract first element (car)
        llvm::Value* dim1_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cons_ptr, is_car});

        // Get cdr (rest of list)
        llvm::Value* cdr_ptr_int = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr_flag});

        // Check if cdr is null (single-element list)
        llvm::Value* cdr_is_null = ctx_.builder().CreateICmpEQ(cdr_ptr_int,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::BasicBlock* has_dim2 = llvm::BasicBlock::Create(ctx_.context(), "zeros_has_dim2", current_func);
        llvm::BasicBlock* list_1d = llvm::BasicBlock::Create(ctx_.context(), "zeros_list_1d", current_func);

        ctx_.builder().CreateCondBr(cdr_is_null, list_1d, has_dim2);

        // 1D list case: only one dimension
        ctx_.builder().SetInsertPoint(list_1d);
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_1d_exit = ctx_.builder().GetInsertBlock();

        // 2D+ list case: extract second dimension
        ctx_.builder().SetInsertPoint(has_dim2);
        llvm::Value* cdr_cons_ptr = ctx_.builder().CreateIntToPtr(cdr_ptr_int, ctx_.ptrType());
        llvm::Value* dim2_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cdr_cons_ptr, is_car});

        // Check for third dimension
        llvm::Value* cddr_ptr_int = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetPtr(), {cdr_cons_ptr, is_cdr_flag});
        llvm::Value* cddr_is_null = ctx_.builder().CreateICmpEQ(cddr_ptr_int,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::BasicBlock* has_dim3 = llvm::BasicBlock::Create(ctx_.context(), "zeros_has_dim3", current_func);
        llvm::BasicBlock* list_2d = llvm::BasicBlock::Create(ctx_.context(), "zeros_list_2d", current_func);

        ctx_.builder().CreateCondBr(cddr_is_null, list_2d, has_dim3);

        // 2D list case
        ctx_.builder().SetInsertPoint(list_2d);
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_2d_exit = ctx_.builder().GetInsertBlock();

        // 3D+ list case
        ctx_.builder().SetInsertPoint(has_dim3);
        llvm::Value* cddr_cons_ptr = ctx_.builder().CreateIntToPtr(cddr_ptr_int, ctx_.ptrType());
        llvm::Value* dim3_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cddr_cons_ptr, is_car});
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_3d_exit = ctx_.builder().GetInsertBlock();

        // SINGLE PATH: Use the value as a single dimension
        ctx_.builder().SetInsertPoint(single_path);
        llvm::Value* single_dim = dim_arg;
        if (single_dim->getType() == ctx_.taggedValueType()) {
            single_dim = tagged_.unpackInt64(single_dim);
        }
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* single_exit = ctx_.builder().GetInsertBlock();

        // MERGE: Use phi nodes to get final dimensions and count
        ctx_.builder().SetInsertPoint(merge_block);

        // num_dims: 1 for single/1d-list, 2 for 2d-list, 3 for 3d-list
        llvm::PHINode* num_dims_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "num_dims");
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 2), list_2d_exit);
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 3), list_3d_exit);
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), single_exit);

        llvm::PHINode* final_dim1 = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "final_dim1");
        final_dim1->addIncoming(dim1_from_list, list_1d_exit);
        final_dim1->addIncoming(dim1_from_list, list_2d_exit);
        final_dim1->addIncoming(dim1_from_list, list_3d_exit);
        final_dim1->addIncoming(single_dim, single_exit);

        llvm::PHINode* final_dim2 = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "final_dim2");
        final_dim2->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
        final_dim2->addIncoming(dim2_from_list, list_2d_exit);
        final_dim2->addIncoming(dim2_from_list, list_3d_exit);
        final_dim2->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), single_exit);

        llvm::PHINode* final_dim3 = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "final_dim3");
        final_dim3->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
        final_dim3->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_2d_exit);
        final_dim3->addIncoming(dim3_from_list, list_3d_exit);
        final_dim3->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), single_exit);

        // Build dims vector based on num_dims at runtime
        // For simplicity, we push all dims and let createTensorWithDims handle it
        // But since createTensorWithDims uses compile-time vector size, we need runtime handling

        // Compute total elements: dim1 * dim2 * dim3 (with dim2=1, dim3=1 for lower dims)
        llvm::Value* total = ctx_.builder().CreateMul(final_dim1, final_dim2);
        total = ctx_.builder().CreateMul(total, final_dim3);

        // Use dynamic tensor creation with runtime dimension count
        // For now, create based on num_dims PHI
        llvm::Value* is_1d = ctx_.builder().CreateICmpEQ(num_dims_phi,
            llvm::ConstantInt::get(ctx_.int64Type(), 1));
        llvm::Value* is_2d = ctx_.builder().CreateICmpEQ(num_dims_phi,
            llvm::ConstantInt::get(ctx_.int64Type(), 2));

        llvm::BasicBlock* create_1d = llvm::BasicBlock::Create(ctx_.context(), "create_1d", current_func);
        llvm::BasicBlock* check_2d = llvm::BasicBlock::Create(ctx_.context(), "check_2d", current_func);
        llvm::BasicBlock* create_2d = llvm::BasicBlock::Create(ctx_.context(), "create_2d", current_func);
        llvm::BasicBlock* create_3d = llvm::BasicBlock::Create(ctx_.context(), "create_3d", current_func);
        llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "zeros_done", current_func);

        ctx_.builder().CreateCondBr(is_1d, create_1d, check_2d);

        ctx_.builder().SetInsertPoint(create_1d);
        std::vector<llvm::Value*> dims_1d = {final_dim1};
        llvm::Value* tensor_1d = createTensorWithDims(dims_1d, nullptr, true);
        ctx_.builder().CreateBr(done_block);
        llvm::BasicBlock* create_1d_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(check_2d);
        ctx_.builder().CreateCondBr(is_2d, create_2d, create_3d);

        ctx_.builder().SetInsertPoint(create_2d);
        std::vector<llvm::Value*> dims_2d = {final_dim1, final_dim2};
        llvm::Value* tensor_2d = createTensorWithDims(dims_2d, nullptr, true);
        ctx_.builder().CreateBr(done_block);
        llvm::BasicBlock* create_2d_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(create_3d);
        std::vector<llvm::Value*> dims_3d = {final_dim1, final_dim2, final_dim3};
        llvm::Value* tensor_3d = createTensorWithDims(dims_3d, nullptr, true);
        ctx_.builder().CreateBr(done_block);
        llvm::BasicBlock* create_3d_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(done_block);
        llvm::PHINode* tensor_phi = ctx_.builder().CreatePHI(ctx_.ptrType(), 3, "tensor_ptr");
        tensor_phi->addIncoming(tensor_1d, create_1d_exit);
        tensor_phi->addIncoming(tensor_2d, create_2d_exit);
        tensor_phi->addIncoming(tensor_3d, create_3d_exit);

        // Pack as consolidated HEAP_PTR (subtype in header)
        return tagged_.packHeapPtr(tensor_phi);
    }

    // Multiple explicit dimension arguments: (zeros 2 3 4)
    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
        llvm::Value* dim = codegenAST(&op->call_op.variables[i]);
        if (!dim) return nullptr;
        // Extract int64 from tagged value if needed
        if (dim->getType() == ctx_.taggedValueType()) {
            dim = tagged_.unpackInt64(dim);
        }
        dims.push_back(dim);
    }

    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, true);  // use_memset_zero = true
    if (!tensor_ptr) return nullptr;

    // Pack as consolidated HEAP_PTR (subtype in header)
    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::ones(const eshkol_operations_t* op) {
    // ones: (ones dim1 dim2 ...) OR (ones '(dim1 dim2 ...)) - Create tensor filled with ones
    // Supports both NumPy-style list syntax and separate dimension arguments
    if (op->call_op.num_vars < 1) {
        eshkol_error("ones requires at least 1 dimension argument");
        return nullptr;
    }

    // Create fill value: 1.0 as double, stored as int64 bit pattern
    llvm::Value* one_double = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_bits = ctx_.builder().CreateBitCast(one_double, ctx_.int64Type());

    std::vector<llvm::Value*> dims;

    if (op->call_op.num_vars == 1) {
        // Could be a single dimension OR a list of dimensions
        llvm::Value* dim_arg = codegenAST(&op->call_op.variables[0]);
        if (!dim_arg) return nullptr;

        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* list_path = llvm::BasicBlock::Create(ctx_.context(), "ones_list", current_func);
        llvm::BasicBlock* single_path = llvm::BasicBlock::Create(ctx_.context(), "ones_single", current_func);
        llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "ones_merge", current_func);

        // Check if it's a list (CONS_PTR or HEAP_PTR) at runtime
        // IMPORTANT: Only check type if dim_arg is actually a tagged_value struct
        // If it's a raw i64, it's definitely not a list
        if (dim_arg->getType() == ctx_.taggedValueType()) {
            llvm::Value* type_tag = tagged_.getType(dim_arg);
            // Use getBaseType() to properly handle legacy types (CONS_PTR=32)
            // DO NOT use 0x0F mask - 32 & 0x0F = 0 (NULL) which is WRONG!
            llvm::Value* base_type = tagged_.getBaseType(type_tag);
            llvm::Value* is_cons_legacy = ctx_.builder().CreateICmpEQ(base_type,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
            llvm::Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(base_type,
                llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
            llvm::Value* is_list = ctx_.builder().CreateOr(is_cons_legacy, is_heap_ptr);
            ctx_.builder().CreateCondBr(is_list, list_path, single_path);
        } else {
            // Raw integer - definitely not a list, go directly to single_path
            ctx_.builder().CreateBr(single_path);
        }

        // LIST PATH: Extract dimensions from the list (up to 4D supported)
        ctx_.builder().SetInsertPoint(list_path);
        llvm::Value* list_ptr_int = tagged_.unpackInt64(dim_arg);
        llvm::Value* cons_ptr = ctx_.builder().CreateIntToPtr(list_ptr_int, ctx_.ptrType());
        llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
        llvm::Value* is_cdr_flag = llvm::ConstantInt::get(ctx_.int1Type(), 1);

        // Extract first element (car)
        llvm::Value* dim1_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cons_ptr, is_car});

        // Get cdr (rest of list)
        llvm::Value* cdr_ptr_int = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr_flag});

        // Check if cdr is null (single-element list)
        llvm::Value* cdr_is_null = ctx_.builder().CreateICmpEQ(cdr_ptr_int,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::BasicBlock* has_dim2 = llvm::BasicBlock::Create(ctx_.context(), "ones_has_dim2", current_func);
        llvm::BasicBlock* list_1d = llvm::BasicBlock::Create(ctx_.context(), "ones_list_1d", current_func);

        ctx_.builder().CreateCondBr(cdr_is_null, list_1d, has_dim2);

        // 1D list case: only one dimension
        ctx_.builder().SetInsertPoint(list_1d);
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_1d_exit = ctx_.builder().GetInsertBlock();

        // 2D+ list case: extract second dimension
        ctx_.builder().SetInsertPoint(has_dim2);
        llvm::Value* cdr_cons_ptr = ctx_.builder().CreateIntToPtr(cdr_ptr_int, ctx_.ptrType());
        llvm::Value* dim2_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cdr_cons_ptr, is_car});

        // Check for third dimension
        llvm::Value* cddr_ptr_int = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetPtr(), {cdr_cons_ptr, is_cdr_flag});
        llvm::Value* cddr_is_null = ctx_.builder().CreateICmpEQ(cddr_ptr_int,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::BasicBlock* has_dim3 = llvm::BasicBlock::Create(ctx_.context(), "ones_has_dim3", current_func);
        llvm::BasicBlock* list_2d = llvm::BasicBlock::Create(ctx_.context(), "ones_list_2d", current_func);

        ctx_.builder().CreateCondBr(cddr_is_null, list_2d, has_dim3);

        // 2D list case
        ctx_.builder().SetInsertPoint(list_2d);
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_2d_exit = ctx_.builder().GetInsertBlock();

        // 3D+ list case
        ctx_.builder().SetInsertPoint(has_dim3);
        llvm::Value* cddr_cons_ptr = ctx_.builder().CreateIntToPtr(cddr_ptr_int, ctx_.ptrType());
        llvm::Value* dim3_from_list = ctx_.builder().CreateCall(
            mem_.getTaggedConsGetInt64(), {cddr_cons_ptr, is_car});
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_3d_exit = ctx_.builder().GetInsertBlock();

        // SINGLE PATH: Use the value as a single dimension
        ctx_.builder().SetInsertPoint(single_path);
        llvm::Value* single_dim = dim_arg;
        if (single_dim->getType() == ctx_.taggedValueType()) {
            single_dim = tagged_.unpackInt64(single_dim);
        }
        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* single_exit = ctx_.builder().GetInsertBlock();

        // MERGE: Use phi nodes to get final dimensions and count
        ctx_.builder().SetInsertPoint(merge_block);

        // num_dims: 1 for single/1d-list, 2 for 2d-list, 3 for 3d-list
        llvm::PHINode* num_dims_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "num_dims");
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 2), list_2d_exit);
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 3), list_3d_exit);
        num_dims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), single_exit);

        llvm::PHINode* final_dim1 = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "final_dim1");
        final_dim1->addIncoming(dim1_from_list, list_1d_exit);
        final_dim1->addIncoming(dim1_from_list, list_2d_exit);
        final_dim1->addIncoming(dim1_from_list, list_3d_exit);
        final_dim1->addIncoming(single_dim, single_exit);

        llvm::PHINode* final_dim2 = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "final_dim2");
        final_dim2->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
        final_dim2->addIncoming(dim2_from_list, list_2d_exit);
        final_dim2->addIncoming(dim2_from_list, list_3d_exit);
        final_dim2->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), single_exit);

        llvm::PHINode* final_dim3 = ctx_.builder().CreatePHI(ctx_.int64Type(), 4, "final_dim3");
        final_dim3->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
        final_dim3->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_2d_exit);
        final_dim3->addIncoming(dim3_from_list, list_3d_exit);
        final_dim3->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), single_exit);

        // Create based on num_dims PHI
        llvm::Value* is_1d = ctx_.builder().CreateICmpEQ(num_dims_phi,
            llvm::ConstantInt::get(ctx_.int64Type(), 1));
        llvm::Value* is_2d = ctx_.builder().CreateICmpEQ(num_dims_phi,
            llvm::ConstantInt::get(ctx_.int64Type(), 2));

        llvm::BasicBlock* create_1d = llvm::BasicBlock::Create(ctx_.context(), "ones_create_1d", current_func);
        llvm::BasicBlock* check_2d = llvm::BasicBlock::Create(ctx_.context(), "ones_check_2d", current_func);
        llvm::BasicBlock* create_2d = llvm::BasicBlock::Create(ctx_.context(), "ones_create_2d", current_func);
        llvm::BasicBlock* create_3d = llvm::BasicBlock::Create(ctx_.context(), "ones_create_3d", current_func);
        llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "ones_done", current_func);

        ctx_.builder().CreateCondBr(is_1d, create_1d, check_2d);

        ctx_.builder().SetInsertPoint(create_1d);
        std::vector<llvm::Value*> dims_1d = {final_dim1};
        llvm::Value* tensor_1d = createTensorWithDims(dims_1d, one_bits, false);
        ctx_.builder().CreateBr(done_block);
        llvm::BasicBlock* create_1d_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(check_2d);
        ctx_.builder().CreateCondBr(is_2d, create_2d, create_3d);

        ctx_.builder().SetInsertPoint(create_2d);
        std::vector<llvm::Value*> dims_2d = {final_dim1, final_dim2};
        llvm::Value* tensor_2d = createTensorWithDims(dims_2d, one_bits, false);
        ctx_.builder().CreateBr(done_block);
        llvm::BasicBlock* create_2d_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(create_3d);
        std::vector<llvm::Value*> dims_3d = {final_dim1, final_dim2, final_dim3};
        llvm::Value* tensor_3d = createTensorWithDims(dims_3d, one_bits, false);
        ctx_.builder().CreateBr(done_block);
        llvm::BasicBlock* create_3d_exit = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(done_block);
        llvm::PHINode* tensor_phi = ctx_.builder().CreatePHI(ctx_.ptrType(), 3, "tensor_ptr");
        tensor_phi->addIncoming(tensor_1d, create_1d_exit);
        tensor_phi->addIncoming(tensor_2d, create_2d_exit);
        tensor_phi->addIncoming(tensor_3d, create_3d_exit);

        // Pack as consolidated HEAP_PTR
        return tagged_.packHeapPtr(tensor_phi);
    }

    // Multiple explicit dimension arguments: (ones 2 3 4)
    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
        llvm::Value* dim = codegenAST(&op->call_op.variables[i]);
        if (!dim) return nullptr;
        if (dim->getType() == ctx_.taggedValueType()) {
            dim = tagged_.unpackInt64(dim);
        }
        dims.push_back(dim);
    }

    llvm::Value* tensor_ptr = createTensorWithDims(dims, one_bits, false);
    if (!tensor_ptr) return nullptr;

    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::eye(const eshkol_operations_t* op) {
    // eye: (eye n) or (eye rows cols) - Create identity matrix
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("eye requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* rows = codegenAST(&op->call_op.variables[0]);
    if (!rows) return nullptr;
    if (rows->getType() == ctx_.taggedValueType()) {
        rows = tagged_.unpackInt64(rows);
    }

    llvm::Value* cols = rows;  // Default: square matrix
    if (op->call_op.num_vars == 2) {
        cols = codegenAST(&op->call_op.variables[1]);
        if (!cols) return nullptr;
        if (cols->getType() == ctx_.taggedValueType()) {
            cols = tagged_.unpackInt64(cols);
        }
    }

    // Create zero-filled tensor first
    std::vector<llvm::Value*> dims = {rows, cols};
    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, true);  // Zero fill
    if (!tensor_ptr) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get elements pointer
    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), elements_field_ptr);
    llvm::Value* typed_elements_ptr = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.ptrType());

    // Set diagonal to 1.0
    llvm::Value* one_double = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_bits = ctx_.builder().CreateBitCast(one_double, ctx_.int64Type());

    // Loop to set diagonal
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "eye_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "eye_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "eye_exit", current_func);

    llvm::Value* min_dim = ctx_.builder().CreateSelect(
        ctx_.builder().CreateICmpULT(rows, cols), rows, cols);
    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "eye_i");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = ctx_.builder().CreateICmpULT(i, min_dim);
    ctx_.builder().CreateCondBr(cmp, loop_body, loop_exit);

    ctx_.builder().SetInsertPoint(loop_body);
    // Diagonal index: i * cols + i
    llvm::Value* diag_idx = ctx_.builder().CreateMul(i, cols);
    diag_idx = ctx_.builder().CreateAdd(diag_idx, i);
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements_ptr, diag_idx);
    ctx_.builder().CreateStore(one_bits, elem_ptr);
    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_exit);

    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::arange(const eshkol_operations_t* op) {
    // arange: (arange n) or (arange start end) or (arange start end step)
    // Works with both integers and floats
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 3) {
        eshkol_error("arange requires 1-3 arguments");
        return nullptr;
    }

    // Helper lambda to extract double from tagged value
    auto extractDouble = [&](llvm::Value* tagged_val) -> llvm::Value* {
        llvm::Value* val_type = tagged_.getType(tagged_val);
        llvm::Value* is_int = ctx_.builder().CreateICmpEQ(val_type,
            llvm::ConstantInt::get(ctx_.int8Type(), 1)); // INT64 = 1

        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* int_bb = llvm::BasicBlock::Create(ctx_.context(), "arange_int", current_func);
        llvm::BasicBlock* dbl_bb = llvm::BasicBlock::Create(ctx_.context(), "arange_dbl", current_func);
        llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "arange_merge", current_func);

        ctx_.builder().CreateCondBr(is_int, int_bb, dbl_bb);

        ctx_.builder().SetInsertPoint(int_bb);
        llvm::Value* int_val = tagged_.unpackInt64(tagged_val);
        llvm::Value* int_as_dbl = ctx_.builder().CreateSIToFP(int_val, ctx_.doubleType());
        ctx_.builder().CreateBr(merge_bb);
        llvm::BasicBlock* int_bb_end = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(dbl_bb);
        llvm::Value* dbl_val = tagged_.unpackDouble(tagged_val);
        ctx_.builder().CreateBr(merge_bb);
        llvm::BasicBlock* dbl_bb_end = ctx_.builder().GetInsertBlock();

        ctx_.builder().SetInsertPoint(merge_bb);
        llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.doubleType(), 2);
        result->addIncoming(int_as_dbl, int_bb_end);
        result->addIncoming(dbl_val, dbl_bb_end);
        return result;
    };

    llvm::Value* start_dbl;
    llvm::Value* end_dbl;
    llvm::Value* step_dbl;

    if (op->call_op.num_vars == 1) {
        // (arange n) -> 0 to n-1
        start_dbl = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        step_dbl = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

        llvm::Value* end_tagged = codegenAST(&op->call_op.variables[0]);
        if (!end_tagged) return nullptr;
        if (end_tagged->getType() == ctx_.taggedValueType()) {
            end_dbl = extractDouble(end_tagged);
        } else if (end_tagged->getType() == ctx_.int64Type()) {
            end_dbl = ctx_.builder().CreateSIToFP(end_tagged, ctx_.doubleType());
        } else {
            end_dbl = end_tagged;
        }
    } else {
        // (arange start end) or (arange start end step)
        llvm::Value* start_tagged = codegenAST(&op->call_op.variables[0]);
        llvm::Value* end_tagged = codegenAST(&op->call_op.variables[1]);
        if (!start_tagged || !end_tagged) return nullptr;

        if (start_tagged->getType() == ctx_.taggedValueType()) {
            start_dbl = extractDouble(start_tagged);
        } else if (start_tagged->getType() == ctx_.int64Type()) {
            start_dbl = ctx_.builder().CreateSIToFP(start_tagged, ctx_.doubleType());
        } else {
            start_dbl = start_tagged;
        }

        if (end_tagged->getType() == ctx_.taggedValueType()) {
            end_dbl = extractDouble(end_tagged);
        } else if (end_tagged->getType() == ctx_.int64Type()) {
            end_dbl = ctx_.builder().CreateSIToFP(end_tagged, ctx_.doubleType());
        } else {
            end_dbl = end_tagged;
        }

        if (op->call_op.num_vars == 3) {
            llvm::Value* step_tagged = codegenAST(&op->call_op.variables[2]);
            if (!step_tagged) return nullptr;
            if (step_tagged->getType() == ctx_.taggedValueType()) {
                step_dbl = extractDouble(step_tagged);
            } else if (step_tagged->getType() == ctx_.int64Type()) {
                step_dbl = ctx_.builder().CreateSIToFP(step_tagged, ctx_.doubleType());
            } else {
                step_dbl = step_tagged;
            }
        } else {
            step_dbl = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        }
    }

    // Calculate number of elements: (end - start) / step
    llvm::Value* range_dbl = ctx_.builder().CreateFSub(end_dbl, start_dbl);
    llvm::Value* num_elements_dbl = ctx_.builder().CreateFDiv(range_dbl, step_dbl);
    llvm::Value* num_elements = ctx_.builder().CreateFPToSI(num_elements_dbl, ctx_.int64Type());

    // Create tensor
    std::vector<llvm::Value*> dims = {num_elements};
    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, false);
    if (!tensor_ptr) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get elements pointer
    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), elements_field_ptr);
    llvm::Value* typed_elements_ptr = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.ptrType());

    // Fill with range values
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "arange_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "arange_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "arange_exit", current_func);

    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "arange_i");
    llvm::Value* current_val = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "arange_val");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    ctx_.builder().CreateStore(start_dbl, current_val);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = ctx_.builder().CreateICmpULT(i, num_elements);
    ctx_.builder().CreateCondBr(cmp, loop_body, loop_exit);

    ctx_.builder().SetInsertPoint(loop_body);
    llvm::Value* val = ctx_.builder().CreateLoad(ctx_.doubleType(), current_val);
    // Store as double bit pattern for consistency
    llvm::Value* val_bits = ctx_.builder().CreateBitCast(val, ctx_.int64Type());
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements_ptr, i);
    ctx_.builder().CreateStore(val_bits, elem_ptr);

    llvm::Value* next_val = ctx_.builder().CreateFAdd(val, step_dbl);
    ctx_.builder().CreateStore(next_val, current_val);
    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_exit);

    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::linspace(const eshkol_operations_t* op) {
    // linspace: (linspace start end num) - num evenly spaced values from start to end
    if (op->call_op.num_vars != 3) {
        eshkol_error("linspace requires exactly 3 arguments: start, end, num");
        return nullptr;
    }

    llvm::Value* start = codegenAST(&op->call_op.variables[0]);
    llvm::Value* end = codegenAST(&op->call_op.variables[1]);
    llvm::Value* num = codegenAST(&op->call_op.variables[2]);
    if (!start || !end || !num) return nullptr;

    // Extract values - convert to double for computation
    if (start->getType() == ctx_.taggedValueType()) {
        start = tagged_.unpackDouble(start);
    } else if (start->getType()->isIntegerTy(64)) {
        start = ctx_.builder().CreateSIToFP(start, ctx_.doubleType());
    }
    if (end->getType() == ctx_.taggedValueType()) {
        end = tagged_.unpackDouble(end);
    } else if (end->getType()->isIntegerTy(64)) {
        end = ctx_.builder().CreateSIToFP(end, ctx_.doubleType());
    }
    if (num->getType() == ctx_.taggedValueType()) {
        num = tagged_.unpackInt64(num);
    }

    // Create tensor
    std::vector<llvm::Value*> dims = {num};
    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, false);
    if (!tensor_ptr) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get elements pointer
    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), elements_field_ptr);
    llvm::Value* typed_elements_ptr = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.ptrType());

    // Calculate step: (end - start) / (num - 1)
    llvm::Value* range = ctx_.builder().CreateFSub(end, start);
    llvm::Value* num_minus_1 = ctx_.builder().CreateSub(num, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* num_minus_1_fp = ctx_.builder().CreateSIToFP(num_minus_1, ctx_.doubleType());
    llvm::Value* step = ctx_.builder().CreateFDiv(range, num_minus_1_fp);

    // Fill with linspace values
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "linspace_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "linspace_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "linspace_exit", current_func);

    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "ls_i");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = ctx_.builder().CreateICmpULT(i, num);
    ctx_.builder().CreateCondBr(cmp, loop_body, loop_exit);

    ctx_.builder().SetInsertPoint(loop_body);
    // val = start + i * step
    llvm::Value* i_fp = ctx_.builder().CreateSIToFP(i, ctx_.doubleType());
    llvm::Value* offset = ctx_.builder().CreateFMul(i_fp, step);
    llvm::Value* val = ctx_.builder().CreateFAdd(start, offset);
    llvm::Value* val_bits = ctx_.builder().CreateBitCast(val, ctx_.int64Type());
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements_ptr, i);
    ctx_.builder().CreateStore(val_bits, elem_ptr);

    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, counter);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_exit);

    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::extractAsDouble(llvm::Value* tagged_val) {
    if (!tagged_val) return nullptr;

    // Handle raw double - return as-is
    if (tagged_val->getType()->isDoubleTy()) return tagged_val;

    // Handle raw int64 - convert to double
    if (tagged_val->getType()->isIntegerTy(64)) {
        return ctx_.builder().CreateSIToFP(tagged_val, ctx_.doubleType());
    }

    // Handle tagged value - check type and extract appropriately
    llvm::Value* type_tag = tagged_.getType(tagged_val);
    // Use getBaseType() to properly handle legacy types (>=32)
    // DO NOT use 0x0F mask - 34 & 0x0F = 2 (DOUBLE) which is WRONG!
    llvm::Value* base_type = tagged_.getBaseType(type_tag);

    llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));

    llvm::Value* dbl_val = tagged_.unpackDouble(tagged_val);
    llvm::Value* int_val = tagged_.unpackInt64(tagged_val);
    llvm::Value* int_as_dbl = ctx_.builder().CreateSIToFP(int_val, ctx_.doubleType());

    return ctx_.builder().CreateSelect(is_double, dbl_val, int_as_dbl, "as_double");
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
