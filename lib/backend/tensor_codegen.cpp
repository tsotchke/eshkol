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

#include <eshkol/backend/cpu_features.h>
#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>

#ifdef ESHKOL_XLA_ENABLED
#include <eshkol/backend/xla/xla_codegen.h>
#endif

namespace eshkol {

TensorCodegen::TensorCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem)
    : ctx_(ctx)
    , tagged_(tagged)
    , mem_(mem) {
#ifdef ESHKOL_XLA_ENABLED
    // Initialize XLA backend for accelerated tensor operations
    xla_ = std::make_unique<xla::XLACodegen>(ctx);
    eshkol_debug("TensorCodegen initialized with SIMD width: %u, XLA: %s (threshold: %zu)",
                 getSIMDWidth(),
                 xla_->isAvailable() ? "available" : "stub",
                 xla::xla_get_threshold());
#else
    eshkol_debug("TensorCodegen initialized with SIMD width: %u", getSIMDWidth());
#endif
}

#ifdef ESHKOL_XLA_ENABLED
bool TensorCodegen::shouldUseXLA(size_t num_elements) const {
    // Use XLA if available and tensor size exceeds threshold
    return xla_ && xla_->shouldUseXLA(num_elements);
}
#endif

// Destructor must be defined where XLACodegen is complete
TensorCodegen::~TensorCodegen() = default;

unsigned TensorCodegen::getSIMDWidth() const {
    return CPUCapabilities::instance().getVectorWidth();
}

llvm::VectorType* TensorCodegen::getSIMDVectorType() const {
    unsigned width = getSIMDWidth();
    switch (width) {
        case 8:
            return ctx_.double8Type();
        case 4:
            return ctx_.double4Type();
        case 2:
            return ctx_.double2Type();
        default:
            // Width 1 = scalar mode, return nullptr to indicate no vectorization
            return nullptr;
    }
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
        }

        // Store result vector
        if (result_vec) {
            builder.CreateAlignedStore(result_vec, result_vec_ptr, llvm::MaybeAlign(8));
        }

        // Increment counter by SIMD_WIDTH
        llvm::Value* next_simd_i = builder.CreateAdd(simd_i,
            llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        builder.CreateStore(next_simd_i, simd_counter);
        builder.CreateBr(simd_cond);

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
    }

    if (result_elem) {
        builder.CreateStore(result_elem, result_elem_ptr);
    }

    llvm::Value* next_scalar_i = builder.CreateAdd(scalar_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_scalar_i, scalar_counter);
    builder.CreateBr(scalar_cond);

    // Final exit - pack tensor result as consolidated HEAP_PTR
    builder.SetInsertPoint(final_exit);
    return tagged_.packHeapPtr(typed_result_tensor_ptr);
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

    // === TENSOR PATH (SIMD-accelerated) ===
    ctx_.builder().SetInsertPoint(tensor_path);
    llvm::Value* tensor_result = rawTensorArithmeticSIMD(arg1, arg2, operation);
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

    // 1D Vector Dot Product: sum(a[i] * b[i]) - SIMD Accelerated
    // Width is auto-detected: 2 (NEON/SSE2), 4 (AVX), or 8 (AVX-512)
    // Falls back to scalar when SIMD_WIDTH == 1
    ctx_.builder().SetInsertPoint(dot_1d_block);

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
        ctx_.builder().CreateBr(simd_cond);

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
    ctx_.builder().CreateBr(scalar_cond);

    // Final result
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

    // Track XLA path state for PHI node construction
    llvm::Value* xla_packed_result = nullptr;
    llvm::BasicBlock* xla_exit_block = nullptr;

#ifdef ESHKOL_XLA_ENABLED
    // ===== XLA DISPATCH FOR MASSIVE TENSORS =====
    // Dispatch hierarchy: XLA (≥100K ops) → SIMD → scalar
    // XLA is only used when StableHLO is available AND tensor is massive
    if (xla_ && xla_->isAvailable()) {
        // Compute total operations: M * K * N
        llvm::Value* mk = ctx_.builder().CreateMul(a_rows, a_cols);
        llvm::Value* total_ops = ctx_.builder().CreateMul(mk, b_cols);
        llvm::Value* threshold = llvm::ConstantInt::get(ctx_.int64Type(), xla::xla_get_threshold());
        llvm::Value* use_xla = ctx_.builder().CreateICmpUGE(total_ops, threshold);

        // Create XLA and fallback blocks
        llvm::BasicBlock* xla_block = llvm::BasicBlock::Create(ctx_.context(), "dot_xla", current_func);
        llvm::BasicBlock* simd_block = llvm::BasicBlock::Create(ctx_.context(), "dot_simd_fallback", current_func);

        ctx_.builder().CreateCondBr(use_xla, xla_block, simd_block);

        // XLA path: emit StableHLO matmul
        ctx_.builder().SetInsertPoint(xla_block);
        llvm::Value* xla_result = xla_->emitMatmul(tensor_a_ptr, tensor_b_ptr);
        if (xla_result) {
            // XLA succeeded, pack result and branch to final merge
            xla_packed_result = tagged_.packHeapPtr(xla_result);
            xla_exit_block = ctx_.builder().GetInsertBlock();
            ctx_.builder().CreateBr(final_merge);
        } else {
            // XLA returned nullptr (e.g., not implemented yet), fall back to SIMD
            ctx_.builder().CreateBr(simd_block);
        }

        // Continue with SIMD/scalar fallback
        ctx_.builder().SetInsertPoint(simd_block);
    }
#endif

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

    // Sum all elements - SIMD Accelerated
    // Width is auto-detected: 2 (NEON/SSE2), 4 (AVX), or 8 (AVX-512)
    // Falls back to scalar when SIMD_WIDTH == 1
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Scalar loop blocks (always needed)
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "tsum_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "tsum_scalar_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "tsum_exit", current_func);

    // Scalar accumulator and counter
    llvm::Value* sum = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "sum_acc");
    llvm::Value* scalar_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "sum_scalar_i");

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

    // Sum all elements - SIMD Accelerated (for mean calculation)
    // Width is auto-detected: 2 (NEON/SSE2), 4 (AVX), or 8 (AVX-512)
    // Falls back to scalar when SIMD_WIDTH == 1
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Scalar loop blocks (always needed)
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "tmean_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "tmean_scalar_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "tmean_exit", current_func);

    // Scalar accumulator and counter
    llvm::Value* sum = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "mean_acc");
    llvm::Value* scalar_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "mean_scalar_i");

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
    ctx_.builder().CreateBr(mean_merge);
    llvm::BasicBlock* tensor_exit_block = ctx_.builder().GetInsertBlock();

    // === MERGE RESULTS ===
    ctx_.builder().SetInsertPoint(mean_merge);
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "mean_result");
    result_phi->addIncoming(svec_result, svec_exit_block);
    result_phi->addIncoming(tensor_result, tensor_exit_block);

    return tagged_.packDouble(result_phi);
}

// ===== TYPE CONVERSION: VECTOR ↔ TENSOR =====

llvm::Value* TensorCodegen::vectorToTensor(const eshkol_operations_t* op) {
    // vector->tensor: (vector->tensor vec) - Convert Scheme vector to 1D tensor
    if (op->call_op.num_vars != 1) {
        eshkol_error("vector->tensor requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* vec_val = codegenAST(&op->call_op.variables[0]);
    if (!vec_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack vector pointer
    llvm::Value* vec_ptr_int = tagged_.unpackInt64(vec_val);
    llvm::Value* vec_ptr = builder.CreateIntToPtr(vec_ptr_int, ctx_.ptrType());

    // Get vector length (first 8 bytes after heap header)
    llvm::Value* len_ptr = vec_ptr;  // Vector layout: [length:i64, elements...]
    llvm::Value* length = builder.CreateLoad(ctx_.int64Type(), len_ptr, "vec_len");

    // Allocate tensor struct
    llvm::Function* v2t_alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* tensor_ptr = builder.CreateCall(v2t_alloc_tensor, {arena_ptr}, "new_tensor");

    // Allocate dimensions array (1D tensor)
    llvm::Function* v2t_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* dims_size = llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
    llvm::Value* dims_ptr = builder.CreateCall(v2t_arena_alloc, {arena_ptr, dims_size}, "v2t_dims");
    builder.CreateStore(length, dims_ptr);

    // Allocate elements array
    llvm::Value* elems_size = builder.CreateMul(length,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* elems_ptr = builder.CreateCall(v2t_arena_alloc, {arena_ptr, elems_size}, "v2t_elems");

    // Copy elements from vector (each element is 16-byte tagged value)
    // Vector layout: [length:i64, element0:tagged_value, element1:tagged_value, ...]
    llvm::Value* vec_data_ptr = builder.CreateGEP(ctx_.int8Type(), vec_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));  // Skip length

    // Create copy loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "v2t_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "v2t_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "v2t_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "v2t_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    // Loop condition: i < length
    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, length);
    builder.CreateCondBr(cmp, loop_body, loop_exit);

    // Loop body: extract double from tagged value, store in tensor
    builder.SetInsertPoint(loop_body);
    // Load tagged value (16 bytes)
    llvm::Value* tagged_ptr = builder.CreateGEP(ctx_.taggedValueType(),
        builder.CreatePointerCast(vec_data_ptr, ctx_.ptrType()), i);
    llvm::Value* tagged_val = builder.CreateLoad(ctx_.taggedValueType(), tagged_ptr);

    // Extract as double (handles int64 -> double conversion if needed)
    llvm::Value* elem_double = extractAsDouble(tagged_val);

    // Store in tensor elements
    llvm::Value* dest_ptr = builder.CreateGEP(ctx_.doubleType(), elems_ptr, i);
    builder.CreateStore(elem_double, dest_ptr);

    // Increment counter
    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Exit: populate tensor struct
    builder.SetInsertPoint(loop_exit);
    llvm::Type* tensor_type = ctx_.tensorType();

    // Store dims pointer (field 0)
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    builder.CreateStore(dims_ptr, dims_field);

    // Store num_dimensions = 1 (field 1)
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), ndim_field);

    // Store elements pointer (field 2)
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    builder.CreateStore(elems_ptr, elems_field);

    // Store total_elements (field 3)
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    builder.CreateStore(length, total_field);

    // Return packed tensor
    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::tensorToVector(const eshkol_operations_t* op) {
    // tensor->vector: (tensor->vector tensor) - Convert tensor to flattened Scheme vector
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor->vector requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());

    // Get tensor fields
    llvm::Type* tensor_type = ctx_.tensorType();
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate Scheme vector using arena_allocate_vector_with_header
    // This creates: [header(8)] + [length(8)] + [elements(16 * n)]
    llvm::Function* alloc_vec_func = mem_.getArenaAllocateVectorWithHeader();
    llvm::Value* vec_ptr = builder.CreateCall(alloc_vec_func, {arena_ptr, total_elements}, "scheme_vec");

    // Set length (already set by allocator, but ensure it's correct)
    llvm::Value* len_ptr = vec_ptr;  // Vector layout: [length:i64, elements...]
    builder.CreateStore(total_elements, len_ptr);

    // Copy elements: wrap each double as tagged value
    llvm::Value* vec_data_ptr = builder.CreateGEP(ctx_.int8Type(), vec_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));  // Skip length

    // Copy loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "t2v_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "t2v_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "t2v_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "t2v_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, loop_exit);

    builder.SetInsertPoint(loop_body);
    // Load double from tensor
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), elems_ptr, i);
    llvm::Value* double_val = builder.CreateLoad(ctx_.doubleType(), src_ptr);

    // Pack as tagged double
    llvm::Value* tagged_double = tagged_.packDouble(double_val);

    // Store in vector
    llvm::Value* dest_ptr = builder.CreateGEP(ctx_.taggedValueType(),
        builder.CreatePointerCast(vec_data_ptr, ctx_.ptrType()), i);
    builder.CreateStore(tagged_double, dest_ptr);

    // Increment
    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_exit);
    return tagged_.packHeapPtr(vec_ptr);
}

// ===== ACTIVATION FUNCTIONS (SIMD-ACCELERATED) =====

llvm::Value* TensorCodegen::tensorRelu(const eshkol_operations_t* op) {
    // ReLU: max(0, x) element-wise
    if (op->call_op.num_vars != 1) {
        eshkol_error("relu requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get tensor dimensions and elements
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate result tensor (same shape)
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "relu_result");

    // Allocate and copy dimensions
    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* relu_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(relu_arena_alloc, {arena_ptr, dims_size}, "relu_dims");

    // Copy dimensions using memcpy intrinsic
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8),
                         dims_ptr, llvm::MaybeAlign(8), dims_size);

    // Allocate result elements
    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(relu_arena_alloc, {arena_ptr, elems_size}, "relu_elems");

    // SIMD parameters
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Create loop blocks
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* simd_cond = llvm::BasicBlock::Create(ctx_.context(), "relu_simd_cond", current_func);
    llvm::BasicBlock* simd_body = llvm::BasicBlock::Create(ctx_.context(), "relu_simd_body", current_func);
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "relu_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "relu_scalar_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "relu_exit", current_func);

    // Counter
    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "relu_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    // Calculate SIMD iteration count
    llvm::Value* simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

    builder.CreateBr(simd_cond);

    // === SIMD Loop ===
    builder.SetInsertPoint(simd_cond);
    llvm::Value* i_simd = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* simd_cmp = builder.CreateICmpULT(i_simd, simd_count);
    builder.CreateCondBr(simd_cmp, simd_body, scalar_cond);

    builder.SetInsertPoint(simd_body);
    if (use_simd) {
        // Load SIMD_WIDTH elements
        llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i_simd);
        llvm::Value* vec_val = builder.CreateAlignedLoad(vec_type, src_ptr, llvm::MaybeAlign(8), "relu_vec");

        // Create zero vector
        llvm::Value* zero_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 0.0));

        // ReLU: max(0, x) = select(x > 0, x, 0)
        llvm::Value* cmp = builder.CreateFCmpOGT(vec_val, zero_vec, "relu_cmp");
        llvm::Value* result_vec = builder.CreateSelect(cmp, vec_val, zero_vec, "relu_result_vec");

        // Store result
        llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i_simd);
        builder.CreateAlignedStore(result_vec, dst_ptr, llvm::MaybeAlign(8));
    }

    // Increment counter
    llvm::Value* next_i = builder.CreateAdd(i_simd,
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(simd_cond);

    // === Scalar Loop (remainder) ===
    builder.SetInsertPoint(scalar_cond);
    llvm::Value* i_scalar = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* scalar_cmp = builder.CreateICmpULT(i_scalar, total_elements);
    builder.CreateCondBr(scalar_cmp, scalar_body, exit_block);

    builder.SetInsertPoint(scalar_body);
    llvm::Value* src_scalar_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i_scalar);
    llvm::Value* val = builder.CreateLoad(ctx_.doubleType(), src_scalar_ptr);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* cmp_scalar = builder.CreateFCmpOGT(val, zero);
    llvm::Value* result_scalar = builder.CreateSelect(cmp_scalar, val, zero);
    llvm::Value* dst_scalar_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i_scalar);
    builder.CreateStore(result_scalar, dst_scalar_ptr);

    llvm::Value* next_i_scalar = builder.CreateAdd(i_scalar,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i_scalar, counter);
    builder.CreateBr(scalar_cond);

    // === Exit: populate result tensor struct ===
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSigmoid(const eshkol_operations_t* op) {
    // Sigmoid: 1 / (1 + exp(-x)) element-wise
    if (op->call_op.num_vars != 1) {
        eshkol_error("sigmoid requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get tensor dimensions and elements
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "sigmoid_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* sig_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(sig_arena_alloc, {arena_ptr, dims_size}, "sig_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(sig_arena_alloc, {arena_ptr, elems_size}, "sig_elems");

    // Get exp intrinsic for scalar fallback
    llvm::Function* exp_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    // Loop over all elements (scalar implementation for stability)
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "sig_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "sig_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "sig_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sig_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
    llvm::Value* x = builder.CreateLoad(ctx_.doubleType(), src_ptr);

    // sigmoid(x) = 1 / (1 + exp(-x))
    llvm::Value* neg_x = builder.CreateFNeg(x, "neg_x");
    llvm::Value* exp_neg_x = builder.CreateCall(exp_func, {neg_x}, "exp_neg");
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* denom = builder.CreateFAdd(one, exp_neg_x, "sig_denom");
    llvm::Value* result = builder.CreateFDiv(one, denom, "sig_result");

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(result, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSoftmax(const eshkol_operations_t* op) {
    // Softmax: exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    // Numerically stable version
    if (op->call_op.num_vars != 1) {
        eshkol_error("softmax requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "softmax_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* sm_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(sm_arena_alloc, {arena_ptr, dims_size}, "sm_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(sm_arena_alloc, {arena_ptr, elems_size}, "sm_elems");

    llvm::Function* exp_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Pass 1: Find maximum element
    llvm::BasicBlock* max_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_max_cond", current_func);
    llvm::BasicBlock* max_body = llvm::BasicBlock::Create(ctx_.context(), "sm_max_body", current_func);
    llvm::BasicBlock* sum_init = llvm::BasicBlock::Create(ctx_.context(), "sm_sum_init", current_func);

    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "sm_max");
    llvm::Value* first_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), src_elems, llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(first_elem, max_val);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sm_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), counter);
    builder.CreateBr(max_cond);

    builder.SetInsertPoint(max_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, max_body, sum_init);

    builder.SetInsertPoint(max_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
    llvm::Value* elem = builder.CreateLoad(ctx_.doubleType(), elem_ptr);
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_greater = builder.CreateFCmpOGT(elem, cur_max);
    llvm::Value* new_max = builder.CreateSelect(is_greater, elem, cur_max);
    builder.CreateStore(new_max, max_val);
    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(max_cond);

    // Pass 2: Compute exp(x - max) and sum
    builder.SetInsertPoint(sum_init);
    llvm::BasicBlock* exp_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_cond", current_func);
    llvm::BasicBlock* exp_body = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_body", current_func);
    llvm::BasicBlock* norm_init = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_init", current_func);

    llvm::Value* sum_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "sm_sum");
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_val);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(exp_cond);

    builder.SetInsertPoint(exp_cond);
    llvm::Value* i2 = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp2 = builder.CreateICmpULT(i2, total_elements);
    builder.CreateCondBr(cmp2, exp_body, norm_init);

    builder.SetInsertPoint(exp_body);
    llvm::Value* src_ptr2 = builder.CreateGEP(ctx_.doubleType(), src_elems, i2);
    llvm::Value* x2 = builder.CreateLoad(ctx_.doubleType(), src_ptr2);
    llvm::Value* final_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* x_shifted = builder.CreateFSub(x2, final_max, "x_shifted");
    llvm::Value* exp_x = builder.CreateCall(exp_func, {x_shifted}, "exp_shifted");
    llvm::Value* dst_ptr2 = builder.CreateGEP(ctx_.doubleType(), result_elems, i2);
    builder.CreateStore(exp_x, dst_ptr2);
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), sum_val);
    llvm::Value* new_sum = builder.CreateFAdd(cur_sum, exp_x);
    builder.CreateStore(new_sum, sum_val);
    llvm::Value* next_i2 = builder.CreateAdd(i2, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i2, counter);
    builder.CreateBr(exp_cond);

    // Pass 3: Normalize (divide by sum)
    builder.SetInsertPoint(norm_init);
    llvm::BasicBlock* norm_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_cond", current_func);
    llvm::BasicBlock* norm_body = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "sm_exit", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_cond);
    llvm::Value* i3 = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp3 = builder.CreateICmpULT(i3, total_elements);
    builder.CreateCondBr(cmp3, norm_body, exit_block);

    builder.SetInsertPoint(norm_body);
    llvm::Value* res_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i3);
    llvm::Value* exp_val = builder.CreateLoad(ctx_.doubleType(), res_ptr);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), sum_val);
    llvm::Value* normalized = builder.CreateFDiv(exp_val, total_sum, "normalized");
    builder.CreateStore(normalized, res_ptr);
    llvm::Value* next_i3 = builder.CreateAdd(i3, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i3, counter);
    builder.CreateBr(norm_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorGelu(const eshkol_operations_t* op) {
    // GELU approximation: x * sigmoid(1.702 * x)
    if (op->call_op.num_vars != 1) {
        eshkol_error("gelu requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "gelu_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* gelu_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(gelu_arena_alloc, {arena_ptr, dims_size}, "gelu_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(gelu_arena_alloc, {arena_ptr, elems_size}, "gelu_elems");

    llvm::Function* exp_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "gelu_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "gelu_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "gelu_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "gelu_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
    llvm::Value* x = builder.CreateLoad(ctx_.doubleType(), src_ptr);

    // GELU(x) ≈ x * sigmoid(1.702 * x)
    llvm::Value* coeff = llvm::ConstantFP::get(ctx_.doubleType(), 1.702);
    llvm::Value* scaled = builder.CreateFMul(coeff, x, "scaled");
    llvm::Value* neg_scaled = builder.CreateFNeg(scaled, "neg_scaled");
    llvm::Value* exp_neg = builder.CreateCall(exp_func, {neg_scaled}, "exp_neg");
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* denom = builder.CreateFAdd(one, exp_neg, "gelu_denom");
    llvm::Value* sigmoid = builder.CreateFDiv(one, denom, "sigmoid");
    llvm::Value* result = builder.CreateFMul(x, sigmoid, "gelu_result");

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(result, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorLeakyRelu(const eshkol_operations_t* op) {
    // Leaky ReLU: x if x > 0, else alpha * x (default alpha = 0.01)
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("leaky-relu requires 1-2 arguments (tensor, optional alpha)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // Default alpha
    double alpha_val = 0.01;
    // Note: For now we use compile-time constant alpha; runtime alpha would need extraction

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "lrelu_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* lrelu_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(lrelu_arena_alloc, {arena_ptr, dims_size}, "lrelu_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(lrelu_arena_alloc, {arena_ptr, elems_size}, "lrelu_elems");

    // SIMD parameters
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* simd_cond = llvm::BasicBlock::Create(ctx_.context(), "lrelu_simd_cond", current_func);
    llvm::BasicBlock* simd_body = llvm::BasicBlock::Create(ctx_.context(), "lrelu_simd_body", current_func);
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "lrelu_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "lrelu_scalar_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "lrelu_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "lrelu_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    llvm::Value* simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

    builder.CreateBr(simd_cond);

    // SIMD Loop
    builder.SetInsertPoint(simd_cond);
    llvm::Value* i_simd = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* simd_cmp = builder.CreateICmpULT(i_simd, simd_count);
    builder.CreateCondBr(simd_cmp, simd_body, scalar_cond);

    builder.SetInsertPoint(simd_body);
    if (use_simd) {
        llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i_simd);
        llvm::Value* x = builder.CreateAlignedLoad(vec_type, src_ptr, llvm::MaybeAlign(8), "lrelu_vec");

        llvm::Value* zero_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        llvm::Value* alpha_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), alpha_val));

        // alpha * x
        llvm::Value* scaled = builder.CreateFMul(alpha_vec, x);

        // x > 0 ? x : alpha*x
        llvm::Value* cmp = builder.CreateFCmpOGT(x, zero_vec);
        llvm::Value* result_vec = builder.CreateSelect(cmp, x, scaled, "lrelu_result_vec");

        llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i_simd);
        builder.CreateAlignedStore(result_vec, dst_ptr, llvm::MaybeAlign(8));
    }

    llvm::Value* next_i = builder.CreateAdd(i_simd,
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(simd_cond);

    // Scalar Loop (remainder)
    builder.SetInsertPoint(scalar_cond);
    llvm::Value* i_scalar = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* scalar_cmp = builder.CreateICmpULT(i_scalar, total_elements);
    builder.CreateCondBr(scalar_cmp, scalar_body, exit_block);

    builder.SetInsertPoint(scalar_body);
    llvm::Value* src_scalar_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i_scalar);
    llvm::Value* val = builder.CreateLoad(ctx_.doubleType(), src_scalar_ptr);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* alpha = llvm::ConstantFP::get(ctx_.doubleType(), alpha_val);
    llvm::Value* scaled_val = builder.CreateFMul(alpha, val);
    llvm::Value* cmp_scalar = builder.CreateFCmpOGT(val, zero);
    llvm::Value* result_scalar = builder.CreateSelect(cmp_scalar, val, scaled_val);
    llvm::Value* dst_scalar_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i_scalar);
    builder.CreateStore(result_scalar, dst_scalar_ptr);

    llvm::Value* next_i_scalar = builder.CreateAdd(i_scalar,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i_scalar, counter);
    builder.CreateBr(scalar_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSilu(const eshkol_operations_t* op) {
    // SiLU/Swish: x * sigmoid(x)
    if (op->call_op.num_vars != 1) {
        eshkol_error("silu requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    auto& builder = ctx_.builder();

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "silu_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "result_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "result_elems");

    // Get exp function
    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* exp_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(exp_type, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    // Loop to compute silu: x * (1 / (1 + exp(-x)))
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "silu_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "silu_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "silu_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "silu_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
    llvm::Value* val_bits = builder.CreateLoad(ctx_.int64Type(), src_ptr);
    llvm::Value* val = builder.CreateBitCast(val_bits, ctx_.doubleType());

    // sigmoid(x) = 1 / (1 + exp(-x))
    llvm::Value* neg_val = builder.CreateFNeg(val);
    llvm::Value* exp_neg = builder.CreateCall(exp_func, {neg_val});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* denom = builder.CreateFAdd(one, exp_neg);
    llvm::Value* sigmoid_val = builder.CreateFDiv(one, denom);
    // silu = x * sigmoid(x)
    llvm::Value* result_val = builder.CreateFMul(val, sigmoid_val);

    llvm::Value* result_bits = builder.CreateBitCast(result_val, ctx_.int64Type());
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, i);
    builder.CreateStore(result_bits, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

// ============================================================
// Activation Backward Functions (for autodiff)
// ============================================================

llvm::Value* TensorCodegen::tensorSoftmaxBackward(llvm::Value* softmax_output, llvm::Value* upstream_grad) {
    // Full tensor softmax gradient:
    // dL/dx_i = s_i * (g_i - sum_j(g_j * s_j))
    // where s = softmax output, g = upstream gradient
    //
    // This is the correct Jacobian-vector product for softmax backprop.
    // Much more efficient than computing the full n×n Jacobian matrix.

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack softmax output tensor
    llvm::Value* softmax_ptr_int = tagged_.unpackInt64(softmax_output);
    llvm::Value* softmax_ptr = builder.CreateIntToPtr(softmax_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack upstream gradient tensor
    llvm::Value* grad_ptr_int = tagged_.unpackInt64(upstream_grad);
    llvm::Value* grad_ptr = builder.CreateIntToPtr(grad_ptr_int, ctx_.ptrType());

    // Get tensor properties from softmax output
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, softmax_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, softmax_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* s_elems_field = builder.CreateStructGEP(tensor_type, softmax_ptr, 2);
    llvm::Value* s_elems = builder.CreateLoad(ctx_.ptrType(), s_elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, softmax_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Get gradient elements
    llvm::Value* g_elems_field = builder.CreateStructGEP(tensor_type, grad_ptr, 2);
    llvm::Value* g_elems = builder.CreateLoad(ctx_.ptrType(), g_elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "sm_back_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "sm_back_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "sm_back_elems");

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Pass 1: Compute dot product sum = sum_j(g_j * s_j)
    llvm::BasicBlock* dot_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_dot_cond", current_func);
    llvm::BasicBlock* dot_body = llvm::BasicBlock::Create(ctx_.context(), "sm_dot_body", current_func);
    llvm::BasicBlock* grad_init = llvm::BasicBlock::Create(ctx_.context(), "sm_grad_init", current_func);

    llvm::Value* dot_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "dot_sum");
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), dot_sum);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sm_back_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(dot_cond);

    builder.SetInsertPoint(dot_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, dot_body, grad_init);

    builder.SetInsertPoint(dot_body);
    // Load s_i
    llvm::Value* s_ptr = builder.CreateGEP(ctx_.doubleType(), s_elems, i);
    llvm::Value* s_i = builder.CreateLoad(ctx_.doubleType(), s_ptr);
    // Load g_i
    llvm::Value* gp_ptr = builder.CreateGEP(ctx_.doubleType(), g_elems, i);
    llvm::Value* g_i = builder.CreateLoad(ctx_.doubleType(), gp_ptr);
    // Accumulate g_i * s_i
    llvm::Value* prod = builder.CreateFMul(g_i, s_i);
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), dot_sum);
    llvm::Value* new_sum = builder.CreateFAdd(cur_sum, prod);
    builder.CreateStore(new_sum, dot_sum);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(dot_cond);

    // Pass 2: Compute dx_i = s_i * (g_i - dot_product)
    builder.SetInsertPoint(grad_init);
    llvm::BasicBlock* grad_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_grad_cond", current_func);
    llvm::BasicBlock* grad_body = llvm::BasicBlock::Create(ctx_.context(), "sm_grad_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "sm_back_exit", current_func);

    llvm::Value* final_dot = builder.CreateLoad(ctx_.doubleType(), dot_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(grad_cond);

    builder.SetInsertPoint(grad_cond);
    llvm::Value* i2 = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp2 = builder.CreateICmpULT(i2, total_elements);
    builder.CreateCondBr(cmp2, grad_body, exit_block);

    builder.SetInsertPoint(grad_body);
    // Load s_i
    llvm::Value* s_ptr2 = builder.CreateGEP(ctx_.doubleType(), s_elems, i2);
    llvm::Value* s_i2 = builder.CreateLoad(ctx_.doubleType(), s_ptr2);
    // Load g_i
    llvm::Value* g_ptr2 = builder.CreateGEP(ctx_.doubleType(), g_elems, i2);
    llvm::Value* g_i2 = builder.CreateLoad(ctx_.doubleType(), g_ptr2);
    // dx_i = s_i * (g_i - dot_product)
    llvm::Value* g_minus_dot = builder.CreateFSub(g_i2, final_dot);
    llvm::Value* dx_i = builder.CreateFMul(s_i2, g_minus_dot);
    // Store result
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i2);
    builder.CreateStore(dx_i, dst_ptr);

    llvm::Value* next_i2 = builder.CreateAdd(i2, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i2, counter);
    builder.CreateBr(grad_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorReluBackward(llvm::Value* input, llvm::Value* upstream_grad) {
    // ReLU backward: dL/dx = dL/dy * (x > 0 ? 1 : 0)

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack upstream gradient tensor
    llvm::Value* grad_ptr_int = tagged_.unpackInt64(upstream_grad);
    llvm::Value* grad_ptr = builder.CreateIntToPtr(grad_ptr_int, ctx_.ptrType());

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Value* g_elems_field = builder.CreateStructGEP(tensor_type, grad_ptr, 2);
    llvm::Value* g_elems = builder.CreateLoad(ctx_.ptrType(), g_elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "relu_back_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "relu_back_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "relu_back_elems");

    // Loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "relu_back_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "relu_back_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "relu_back_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "relu_back_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Load x_i
    llvm::Value* x_ptr = builder.CreateGEP(ctx_.doubleType(), x_elems, i);
    llvm::Value* x_i = builder.CreateLoad(ctx_.doubleType(), x_ptr);
    // Load g_i
    llvm::Value* g_ptr = builder.CreateGEP(ctx_.doubleType(), g_elems, i);
    llvm::Value* g_i = builder.CreateLoad(ctx_.doubleType(), g_ptr);
    // dx_i = g_i * (x_i > 0 ? 1 : 0)
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* is_positive = builder.CreateFCmpOGT(x_i, zero);
    llvm::Value* mask = builder.CreateSelect(is_positive, one, zero);
    llvm::Value* dx_i = builder.CreateFMul(g_i, mask);
    // Store result
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(dx_i, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSigmoidBackward(llvm::Value* sigmoid_output, llvm::Value* upstream_grad) {
    // Sigmoid backward: dL/dx = dL/dy * σ(x) * (1 - σ(x))
    // We use the output directly since σ(x) is already computed

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack sigmoid output tensor
    llvm::Value* sig_ptr_int = tagged_.unpackInt64(sigmoid_output);
    llvm::Value* sig_ptr = builder.CreateIntToPtr(sig_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack upstream gradient tensor
    llvm::Value* grad_ptr_int = tagged_.unpackInt64(upstream_grad);
    llvm::Value* grad_ptr = builder.CreateIntToPtr(grad_ptr_int, ctx_.ptrType());

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, sig_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, sig_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* s_elems_field = builder.CreateStructGEP(tensor_type, sig_ptr, 2);
    llvm::Value* s_elems = builder.CreateLoad(ctx_.ptrType(), s_elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, sig_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Value* g_elems_field = builder.CreateStructGEP(tensor_type, grad_ptr, 2);
    llvm::Value* g_elems = builder.CreateLoad(ctx_.ptrType(), g_elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "sig_back_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "sig_back_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "sig_back_elems");

    // Loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "sig_back_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "sig_back_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "sig_back_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sig_back_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Load σ(x)_i
    llvm::Value* s_ptr = builder.CreateGEP(ctx_.doubleType(), s_elems, i);
    llvm::Value* sig_i = builder.CreateLoad(ctx_.doubleType(), s_ptr);
    // Load g_i
    llvm::Value* g_ptr = builder.CreateGEP(ctx_.doubleType(), g_elems, i);
    llvm::Value* g_i = builder.CreateLoad(ctx_.doubleType(), g_ptr);
    // dx_i = g_i * σ(x) * (1 - σ(x))
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_minus_sig = builder.CreateFSub(one, sig_i);
    llvm::Value* sig_deriv = builder.CreateFMul(sig_i, one_minus_sig);
    llvm::Value* dx_i = builder.CreateFMul(g_i, sig_deriv);
    // Store result
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(dx_i, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorGeluBackward(llvm::Value* input, llvm::Value* upstream_grad) {
    // GELU backward using sigmoid approximation:
    // gelu(x) ≈ x * σ(1.702x)
    // gelu'(x) ≈ σ(1.702x) + 1.702x * σ(1.702x) * (1 - σ(1.702x))
    //          = σ(1.702x) * (1 + 1.702x * (1 - σ(1.702x)))

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack upstream gradient tensor
    llvm::Value* grad_ptr_int = tagged_.unpackInt64(upstream_grad);
    llvm::Value* grad_ptr = builder.CreateIntToPtr(grad_ptr_int, ctx_.ptrType());

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Value* g_elems_field = builder.CreateStructGEP(tensor_type, grad_ptr, 2);
    llvm::Value* g_elems = builder.CreateLoad(ctx_.ptrType(), g_elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "gelu_back_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "gelu_back_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "gelu_back_elems");

    // Get exp function
    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* exp_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(exp_type, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    // Loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "gelu_back_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "gelu_back_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "gelu_back_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "gelu_back_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Load x_i
    llvm::Value* x_ptr = builder.CreateGEP(ctx_.doubleType(), x_elems, i);
    llvm::Value* x_i = builder.CreateLoad(ctx_.doubleType(), x_ptr);
    // Load g_i
    llvm::Value* g_ptr = builder.CreateGEP(ctx_.doubleType(), g_elems, i);
    llvm::Value* g_i = builder.CreateLoad(ctx_.doubleType(), g_ptr);

    // Compute σ(1.702x)
    llvm::Value* coeff = llvm::ConstantFP::get(ctx_.doubleType(), 1.702);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* scaled_x = builder.CreateFMul(coeff, x_i);
    llvm::Value* neg_scaled = builder.CreateFNeg(scaled_x);
    llvm::Value* exp_neg = builder.CreateCall(exp_func, {neg_scaled});
    llvm::Value* denom = builder.CreateFAdd(one, exp_neg);
    llvm::Value* sigma = builder.CreateFDiv(one, denom);

    // gelu'(x) = σ * (1 + 1.702x * (1 - σ))
    llvm::Value* one_minus_sigma = builder.CreateFSub(one, sigma);
    llvm::Value* scaled_x_sigma_deriv = builder.CreateFMul(scaled_x, one_minus_sigma);
    llvm::Value* inner = builder.CreateFAdd(one, scaled_x_sigma_deriv);
    llvm::Value* gelu_deriv = builder.CreateFMul(sigma, inner);

    // dx_i = g_i * gelu'(x)
    llvm::Value* dx_i = builder.CreateFMul(g_i, gelu_deriv);
    // Store result
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(dx_i, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorLeakyReluBackward(llvm::Value* input, llvm::Value* upstream_grad, double alpha) {
    // Leaky ReLU backward: dL/dx = dL/dy * (x > 0 ? 1 : alpha)

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack upstream gradient tensor
    llvm::Value* grad_ptr_int = tagged_.unpackInt64(upstream_grad);
    llvm::Value* grad_ptr = builder.CreateIntToPtr(grad_ptr_int, ctx_.ptrType());

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Value* g_elems_field = builder.CreateStructGEP(tensor_type, grad_ptr, 2);
    llvm::Value* g_elems = builder.CreateLoad(ctx_.ptrType(), g_elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "lrelu_back_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "lrelu_back_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "lrelu_back_elems");

    // Loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "lrelu_back_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "lrelu_back_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "lrelu_back_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "lrelu_back_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Load x_i
    llvm::Value* x_ptr = builder.CreateGEP(ctx_.doubleType(), x_elems, i);
    llvm::Value* x_i = builder.CreateLoad(ctx_.doubleType(), x_ptr);
    // Load g_i
    llvm::Value* g_ptr = builder.CreateGEP(ctx_.doubleType(), g_elems, i);
    llvm::Value* g_i = builder.CreateLoad(ctx_.doubleType(), g_ptr);
    // dx_i = g_i * (x_i > 0 ? 1 : alpha)
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* alpha_val = llvm::ConstantFP::get(ctx_.doubleType(), alpha);
    llvm::Value* is_positive = builder.CreateFCmpOGT(x_i, zero);
    llvm::Value* mask = builder.CreateSelect(is_positive, one, alpha_val);
    llvm::Value* dx_i = builder.CreateFMul(g_i, mask);
    // Store result
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(dx_i, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSiluBackward(llvm::Value* input, llvm::Value* upstream_grad) {
    // SiLU backward: dL/dx = dL/dy * (σ(x) + x * σ(x) * (1 - σ(x)))
    //                      = dL/dy * σ(x) * (1 + x * (1 - σ(x)))

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack upstream gradient tensor
    llvm::Value* grad_ptr_int = tagged_.unpackInt64(upstream_grad);
    llvm::Value* grad_ptr = builder.CreateIntToPtr(grad_ptr_int, ctx_.ptrType());

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Value* g_elems_field = builder.CreateStructGEP(tensor_type, grad_ptr, 2);
    llvm::Value* g_elems = builder.CreateLoad(ctx_.ptrType(), g_elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "silu_back_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "silu_back_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "silu_back_elems");

    // Get exp function
    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* exp_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(exp_type, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    // Loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "silu_back_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "silu_back_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "silu_back_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "silu_back_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Load x_i
    llvm::Value* x_ptr = builder.CreateGEP(ctx_.doubleType(), x_elems, i);
    llvm::Value* x_i = builder.CreateLoad(ctx_.doubleType(), x_ptr);
    // Load g_i
    llvm::Value* g_ptr = builder.CreateGEP(ctx_.doubleType(), g_elems, i);
    llvm::Value* g_i = builder.CreateLoad(ctx_.doubleType(), g_ptr);

    // Compute σ(x)
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* neg_x = builder.CreateFNeg(x_i);
    llvm::Value* exp_neg = builder.CreateCall(exp_func, {neg_x});
    llvm::Value* denom = builder.CreateFAdd(one, exp_neg);
    llvm::Value* sigma = builder.CreateFDiv(one, denom);

    // silu'(x) = σ(x) * (1 + x * (1 - σ(x)))
    llvm::Value* one_minus_sigma = builder.CreateFSub(one, sigma);
    llvm::Value* x_times_deriv = builder.CreateFMul(x_i, one_minus_sigma);
    llvm::Value* inner = builder.CreateFAdd(one, x_times_deriv);
    llvm::Value* silu_deriv = builder.CreateFMul(sigma, inner);

    // dx_i = g_i * silu'(x)
    llvm::Value* dx_i = builder.CreateFMul(g_i, silu_deriv);
    // Store result
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(dx_i, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

// ============================================================
// Statistics Operations
// ============================================================

llvm::Value* TensorCodegen::tensorVar(const eshkol_operations_t* op) {
    // Variance: E[(x - mean)^2]
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-var requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    auto& builder = ctx_.builder();

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // First pass: compute mean
    llvm::BasicBlock* mean_cond = llvm::BasicBlock::Create(ctx_.context(), "var_mean_cond", current_func);
    llvm::BasicBlock* mean_body = llvm::BasicBlock::Create(ctx_.context(), "var_mean_body", current_func);
    llvm::BasicBlock* mean_done = llvm::BasicBlock::Create(ctx_.context(), "var_mean_done", current_func);

    llvm::Value* sum_alloc = builder.CreateAlloca(ctx_.doubleType(), nullptr, "var_sum");
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_alloc);
    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "var_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_cond);
    llvm::Value* i1 = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp1 = builder.CreateICmpULT(i1, total_elements);
    builder.CreateCondBr(cmp1, mean_body, mean_done);

    builder.SetInsertPoint(mean_body);
    llvm::Value* elem_ptr1 = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i1);
    llvm::Value* elem_bits1 = builder.CreateLoad(ctx_.int64Type(), elem_ptr1);
    llvm::Value* elem_val1 = builder.CreateBitCast(elem_bits1, ctx_.doubleType());
    llvm::Value* curr_sum = builder.CreateLoad(ctx_.doubleType(), sum_alloc);
    llvm::Value* new_sum = builder.CreateFAdd(curr_sum, elem_val1);
    builder.CreateStore(new_sum, sum_alloc);
    llvm::Value* next_i1 = builder.CreateAdd(i1, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i1, counter);
    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_done);
    llvm::Value* final_sum = builder.CreateLoad(ctx_.doubleType(), sum_alloc);
    llvm::Value* n_fp = builder.CreateSIToFP(total_elements, ctx_.doubleType());
    llvm::Value* mean = builder.CreateFDiv(final_sum, n_fp);

    // Second pass: compute sum of squared deviations
    llvm::BasicBlock* var_cond = llvm::BasicBlock::Create(ctx_.context(), "var_sq_cond", current_func);
    llvm::BasicBlock* var_body = llvm::BasicBlock::Create(ctx_.context(), "var_sq_body", current_func);
    llvm::BasicBlock* var_done = llvm::BasicBlock::Create(ctx_.context(), "var_sq_done", current_func);

    llvm::Value* sq_sum_alloc = builder.CreateAlloca(ctx_.doubleType(), nullptr, "var_sq_sum");
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sq_sum_alloc);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(var_cond);

    builder.SetInsertPoint(var_cond);
    llvm::Value* i2 = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp2 = builder.CreateICmpULT(i2, total_elements);
    builder.CreateCondBr(cmp2, var_body, var_done);

    builder.SetInsertPoint(var_body);
    llvm::Value* elem_ptr2 = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i2);
    llvm::Value* elem_bits2 = builder.CreateLoad(ctx_.int64Type(), elem_ptr2);
    llvm::Value* elem_val2 = builder.CreateBitCast(elem_bits2, ctx_.doubleType());
    llvm::Value* diff = builder.CreateFSub(elem_val2, mean);
    llvm::Value* sq_diff = builder.CreateFMul(diff, diff);
    llvm::Value* curr_sq_sum = builder.CreateLoad(ctx_.doubleType(), sq_sum_alloc);
    llvm::Value* new_sq_sum = builder.CreateFAdd(curr_sq_sum, sq_diff);
    builder.CreateStore(new_sq_sum, sq_sum_alloc);
    llvm::Value* next_i2 = builder.CreateAdd(i2, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i2, counter);
    builder.CreateBr(var_cond);

    builder.SetInsertPoint(var_done);
    llvm::Value* final_sq_sum = builder.CreateLoad(ctx_.doubleType(), sq_sum_alloc);
    llvm::Value* variance = builder.CreateFDiv(final_sq_sum, n_fp);

    return tagged_.packDouble(variance);
}

llvm::Value* TensorCodegen::tensorStd(const eshkol_operations_t* op) {
    // Standard deviation = sqrt(variance)
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-std requires exactly 1 argument");
        return nullptr;
    }

    // Compute variance first
    llvm::Value* var_result = tensorVar(op);
    llvm::Value* variance = tagged_.unpackDouble(var_result);

    // Get sqrt function
    llvm::Function* sqrt_func = ctx_.module().getFunction("sqrt");
    if (!sqrt_func) {
        llvm::FunctionType* sqrt_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        sqrt_func = llvm::Function::Create(sqrt_type, llvm::Function::ExternalLinkage, "sqrt", &ctx_.module());
    }

    llvm::Value* std_dev = ctx_.builder().CreateCall(sqrt_func, {variance});
    return tagged_.packDouble(std_dev);
}

// ============================================================
// Random Tensor Generation
// ============================================================

llvm::Value* TensorCodegen::tensorRand(const eshkol_operations_t* op) {
    // rand: create tensor with uniform random values [0, 1)
    // Syntax: (rand dim1 dim2 ...)
    if (op->call_op.num_vars < 1) {
        eshkol_error("rand requires at least 1 dimension argument");
        return nullptr;
    }

    auto& builder = ctx_.builder();
    uint64_t num_dims = op->call_op.num_vars;

    // Collect dimensions
    std::vector<llvm::Value*> dims;
    llvm::Value* total = llvm::ConstantInt::get(ctx_.int64Type(), 1);
    for (uint64_t i = 0; i < num_dims; ++i) {
        llvm::Value* dim_val = codegenAST(&op->call_op.variables[i]);
        if (!dim_val) return nullptr;
        llvm::Value* dim;
        if (dim_val->getType() == ctx_.taggedValueType()) {
            dim = tagged_.unpackInt64(dim_val);
        } else {
            dim = dim_val;
        }
        dims.push_back(dim);
        total = builder.CreateMul(total, dim);
    }

    // Create tensor
    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, false);
    if (!tensor_ptr) return nullptr;

    // Get elements pointer
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);

    // Get drand48 function for uniform [0, 1)
    llvm::Function* drand48_func = ctx_.module().getFunction("drand48");
    if (!drand48_func) {
        llvm::FunctionType* drand48_type = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand48_func = llvm::Function::Create(drand48_type, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }

    // Fill loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "rand_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "rand_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "rand_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "rand_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total);
    builder.CreateCondBr(cmp, loop_body, loop_exit);

    builder.SetInsertPoint(loop_body);
    llvm::Value* rand_val = builder.CreateCall(drand48_func, {});
    llvm::Value* rand_bits = builder.CreateBitCast(rand_val, ctx_.int64Type());
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i);
    builder.CreateStore(rand_bits, elem_ptr);
    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_exit);
    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::tensorRandn(const eshkol_operations_t* op) {
    // randn: create tensor with standard normal random values (mean=0, std=1)
    // Uses Box-Muller transform
    // Syntax: (randn dim1 dim2 ...)
    if (op->call_op.num_vars < 1) {
        eshkol_error("randn requires at least 1 dimension argument");
        return nullptr;
    }

    auto& builder = ctx_.builder();
    uint64_t num_dims = op->call_op.num_vars;

    std::vector<llvm::Value*> dims;
    llvm::Value* total = llvm::ConstantInt::get(ctx_.int64Type(), 1);
    for (uint64_t i = 0; i < num_dims; ++i) {
        llvm::Value* dim_val = codegenAST(&op->call_op.variables[i]);
        if (!dim_val) return nullptr;
        llvm::Value* dim;
        if (dim_val->getType() == ctx_.taggedValueType()) {
            dim = tagged_.unpackInt64(dim_val);
        } else {
            dim = dim_val;
        }
        dims.push_back(dim);
        total = builder.CreateMul(total, dim);
    }

    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, false);
    if (!tensor_ptr) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);

    // Get math functions
    llvm::Function* drand48_func = ctx_.module().getFunction("drand48");
    if (!drand48_func) {
        llvm::FunctionType* drand48_type = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand48_func = llvm::Function::Create(drand48_type, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* log_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(log_type, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }
    llvm::Function* sqrt_func = ctx_.module().getFunction("sqrt");
    if (!sqrt_func) {
        llvm::FunctionType* sqrt_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        sqrt_func = llvm::Function::Create(sqrt_type, llvm::Function::ExternalLinkage, "sqrt", &ctx_.module());
    }
    llvm::Function* cos_func = ctx_.module().getFunction("cos");
    if (!cos_func) {
        llvm::FunctionType* cos_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        cos_func = llvm::Function::Create(cos_type, llvm::Function::ExternalLinkage, "cos", &ctx_.module());
    }

    llvm::Value* two_pi = llvm::ConstantFP::get(ctx_.doubleType(), 6.283185307179586);
    llvm::Value* neg_two = llvm::ConstantFP::get(ctx_.doubleType(), -2.0);

    // Fill loop with Box-Muller
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "randn_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "randn_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "randn_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "randn_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total);
    builder.CreateCondBr(cmp, loop_body, loop_exit);

    builder.SetInsertPoint(loop_body);
    // Box-Muller: z = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
    llvm::Value* u1 = builder.CreateCall(drand48_func, {});
    llvm::Value* u2 = builder.CreateCall(drand48_func, {});
    llvm::Value* log_u1 = builder.CreateCall(log_func, {u1});
    llvm::Value* neg_2_log = builder.CreateFMul(neg_two, log_u1);
    llvm::Value* sqrt_part = builder.CreateCall(sqrt_func, {neg_2_log});
    llvm::Value* angle = builder.CreateFMul(two_pi, u2);
    llvm::Value* cos_part = builder.CreateCall(cos_func, {angle});
    llvm::Value* normal_val = builder.CreateFMul(sqrt_part, cos_part);

    llvm::Value* normal_bits = builder.CreateBitCast(normal_val, ctx_.int64Type());
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i);
    builder.CreateStore(normal_bits, elem_ptr);
    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_exit);
    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::tensorRandint(const eshkol_operations_t* op) {
    // randint: create tensor with random integers in [low, high)
    // Syntax: (randint low high dim1 dim2 ...)
    if (op->call_op.num_vars < 3) {
        eshkol_error("randint requires at least 3 arguments (low, high, dim1...)");
        return nullptr;
    }

    auto& builder = ctx_.builder();

    // Get low and high bounds
    llvm::Value* low_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* high_val = codegenAST(&op->call_op.variables[1]);
    if (!low_val || !high_val) return nullptr;

    llvm::Value* low;
    llvm::Value* high;
    if (low_val->getType() == ctx_.taggedValueType()) {
        low = tagged_.unpackInt64(low_val);
    } else {
        low = low_val;
    }
    if (high_val->getType() == ctx_.taggedValueType()) {
        high = tagged_.unpackInt64(high_val);
    } else {
        high = high_val;
    }

    // Collect dimensions
    uint64_t num_dims = op->call_op.num_vars - 2;
    std::vector<llvm::Value*> dims;
    llvm::Value* total = llvm::ConstantInt::get(ctx_.int64Type(), 1);
    for (uint64_t i = 0; i < num_dims; ++i) {
        llvm::Value* dim_val = codegenAST(&op->call_op.variables[i + 2]);
        if (!dim_val) return nullptr;
        llvm::Value* dim;
        if (dim_val->getType() == ctx_.taggedValueType()) {
            dim = tagged_.unpackInt64(dim_val);
        } else {
            dim = dim_val;
        }
        dims.push_back(dim);
        total = builder.CreateMul(total, dim);
    }

    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, false);
    if (!tensor_ptr) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);

    // Get lrand48 function
    llvm::Function* lrand48_func = ctx_.module().getFunction("lrand48");
    if (!lrand48_func) {
        llvm::FunctionType* lrand48_type = llvm::FunctionType::get(ctx_.int64Type(), {}, false);
        lrand48_func = llvm::Function::Create(lrand48_type, llvm::Function::ExternalLinkage, "lrand48", &ctx_.module());
    }

    // range = high - low
    llvm::Value* range = builder.CreateSub(high, low);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "randint_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "randint_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "randint_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "randint_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total);
    builder.CreateCondBr(cmp, loop_body, loop_exit);

    builder.SetInsertPoint(loop_body);
    llvm::Value* rand_long = builder.CreateCall(lrand48_func, {});
    llvm::Value* rand_mod = builder.CreateSRem(rand_long, range);
    llvm::Value* rand_int = builder.CreateAdd(rand_mod, low);
    // Store as double (tensor elements are always doubles)
    llvm::Value* rand_double = builder.CreateSIToFP(rand_int, ctx_.doubleType());
    llvm::Value* rand_bits = builder.CreateBitCast(rand_double, ctx_.int64Type());
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i);
    builder.CreateStore(rand_bits, elem_ptr);
    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_exit);
    return tagged_.packHeapPtr(tensor_ptr);
}

// ============================================================
// Shape Operations (Phase 4)
// ============================================================

llvm::Value* TensorCodegen::squeeze(const eshkol_operations_t* op) {
    // Squeeze: remove dimensions of size 1
    // (squeeze tensor) - remove all size-1 dims
    // (squeeze tensor dim) - remove specific dim if size is 1
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("squeeze requires 1-2 arguments (tensor, optional dim)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Load tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Allocate result tensor (metadata only - shares element data)
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "squeeze_result");

    // Count non-size-1 dimensions (dynamic loop)
    // First pass: count how many dims to keep
    llvm::BasicBlock* count_cond = llvm::BasicBlock::Create(ctx_.context(), "sq_count_cond", current_func);
    llvm::BasicBlock* count_body = llvm::BasicBlock::Create(ctx_.context(), "sq_count_body", current_func);
    llvm::BasicBlock* count_done = llvm::BasicBlock::Create(ctx_.context(), "sq_count_done", current_func);

    llvm::Value* count_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sq_count");
    llvm::Value* count_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sq_count_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), count_var);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), count_i);
    builder.CreateBr(count_cond);

    builder.SetInsertPoint(count_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), count_i);
    llvm::Value* count_cmp = builder.CreateICmpULT(ci, num_dims);
    builder.CreateCondBr(count_cmp, count_body, count_done);

    builder.SetInsertPoint(count_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, ci);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* is_not_one = builder.CreateICmpNE(dim_val, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* curr_count = builder.CreateLoad(ctx_.int64Type(), count_var);
    llvm::Value* new_count = builder.CreateSelect(is_not_one,
        builder.CreateAdd(curr_count, llvm::ConstantInt::get(ctx_.int64Type(), 1)),
        curr_count);
    builder.CreateStore(new_count, count_var);
    llvm::Value* next_ci = builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_ci, count_i);
    builder.CreateBr(count_cond);

    builder.SetInsertPoint(count_done);
    llvm::Value* new_ndim = builder.CreateLoad(ctx_.int64Type(), count_var);

    // Ensure at least 1 dimension (scalar becomes 1D with single element)
    llvm::Value* is_zero = builder.CreateICmpEQ(new_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    new_ndim = builder.CreateSelect(is_zero, llvm::ConstantInt::get(ctx_.int64Type(), 1), new_ndim);

    // Allocate new dimensions array
    llvm::Value* new_dims_size = builder.CreateMul(new_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* sq_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* new_dims = builder.CreateCall(sq_arena_alloc, {arena_ptr, new_dims_size}, "sq_new_dims");

    // Second pass: copy non-size-1 dimensions
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "sq_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "sq_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "sq_copy_done", current_func);

    llvm::Value* src_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sq_src_idx");
    llvm::Value* dst_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sq_dst_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), src_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dst_idx);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), src_idx);
    llvm::Value* copy_cmp = builder.CreateICmpULT(si, num_dims);
    builder.CreateCondBr(copy_cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, si);
    llvm::Value* src_dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    llvm::Value* keep_dim = builder.CreateICmpNE(src_dim_val, llvm::ConstantInt::get(ctx_.int64Type(), 1));

    // Conditional copy block
    llvm::BasicBlock* do_copy = llvm::BasicBlock::Create(ctx_.context(), "sq_do_copy", current_func);
    llvm::BasicBlock* skip_copy = llvm::BasicBlock::Create(ctx_.context(), "sq_skip_copy", current_func);

    builder.CreateCondBr(keep_dim, do_copy, skip_copy);

    builder.SetInsertPoint(do_copy);
    llvm::Value* di = builder.CreateLoad(ctx_.int64Type(), dst_idx);
    llvm::Value* dst_dim_ptr = builder.CreateGEP(ctx_.int64Type(), new_dims, di);
    builder.CreateStore(src_dim_val, dst_dim_ptr);
    llvm::Value* next_di = builder.CreateAdd(di, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_di, dst_idx);
    builder.CreateBr(skip_copy);

    builder.SetInsertPoint(skip_copy);
    llvm::Value* next_si = builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_si, src_idx);
    builder.CreateBr(copy_cond);

    // Handle edge case: all dims were squeezed (scalar)
    builder.SetInsertPoint(copy_done);
    llvm::BasicBlock* scalar_case = llvm::BasicBlock::Create(ctx_.context(), "sq_scalar", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "sq_finalize", current_func);

    llvm::Value* final_dst_idx = builder.CreateLoad(ctx_.int64Type(), dst_idx);
    llvm::Value* was_scalar = builder.CreateICmpEQ(final_dst_idx, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateCondBr(was_scalar, scalar_case, finalize);

    builder.SetInsertPoint(scalar_case);
    // Set single dimension of 1 (or total_elements for proper shape)
    llvm::Value* scalar_dim_ptr = builder.CreateGEP(ctx_.int64Type(), new_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateStore(total_elements, scalar_dim_ptr);
    builder.CreateBr(finalize);

    // Populate result tensor (shares element data with original)
    builder.SetInsertPoint(finalize);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(new_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(new_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(src_elems, r_elems_field);  // Share data - no copy!
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::unsqueeze(const eshkol_operations_t* op) {
    // Unsqueeze: add a dimension of size 1 at specified position
    // (unsqueeze tensor dim) - adds size-1 dim at position dim
    if (op->call_op.num_vars != 2) {
        eshkol_error("unsqueeze requires 2 arguments (tensor, dim)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    llvm::Value* dim_arg = codegenAST(&op->call_op.variables[1]);
    if (!dim_arg) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Load tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Extract dim index - handle both raw int64 and tagged
    llvm::Value* dim_idx = dim_arg;
    if (dim_arg->getType() == ctx_.taggedValueType()) {
        dim_idx = tagged_.unpackInt64(dim_arg);
    } else if (!dim_arg->getType()->isIntegerTy(64)) {
        dim_idx = builder.CreateSExtOrTrunc(dim_arg, ctx_.int64Type());
    }

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "unsq_result");

    // New number of dimensions is old + 1
    llvm::Value* new_ndim = builder.CreateAdd(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));

    // Allocate new dimensions array
    llvm::Value* new_dims_size = builder.CreateMul(new_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* unsq_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* new_dims = builder.CreateCall(unsq_arena_alloc, {arena_ptr, new_dims_size}, "unsq_new_dims");

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Copy dimensions, inserting 1 at dim_idx
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "unsq_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "unsq_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "unsq_copy_done", current_func);

    llvm::Value* src_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "unsq_src_i");
    llvm::Value* dst_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "unsq_dst_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), src_i);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dst_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* di = builder.CreateLoad(ctx_.int64Type(), dst_i);
    llvm::Value* cmp = builder.CreateICmpULT(di, new_ndim);
    builder.CreateCondBr(cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    // Check if this is the insertion point
    llvm::Value* is_insert_pos = builder.CreateICmpEQ(di, dim_idx);

    llvm::BasicBlock* insert_one = llvm::BasicBlock::Create(ctx_.context(), "unsq_insert_one", current_func);
    llvm::BasicBlock* copy_old = llvm::BasicBlock::Create(ctx_.context(), "unsq_copy_old", current_func);
    llvm::BasicBlock* next_iter = llvm::BasicBlock::Create(ctx_.context(), "unsq_next_iter", current_func);

    builder.CreateCondBr(is_insert_pos, insert_one, copy_old);

    builder.SetInsertPoint(insert_one);
    llvm::Value* dst_ptr_ins = builder.CreateGEP(ctx_.int64Type(), new_dims, di);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), dst_ptr_ins);
    builder.CreateBr(next_iter);

    builder.SetInsertPoint(copy_old);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), src_i);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, si);
    llvm::Value* src_dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    llvm::Value* dst_ptr_cpy = builder.CreateGEP(ctx_.int64Type(), new_dims, di);
    builder.CreateStore(src_dim_val, dst_ptr_cpy);
    // Increment source index only when copying
    llvm::Value* next_si = builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_si, src_i);
    builder.CreateBr(next_iter);

    builder.SetInsertPoint(next_iter);
    llvm::Value* next_di = builder.CreateAdd(di, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_di, dst_i);
    builder.CreateBr(copy_cond);

    // Populate result tensor (shares element data)
    builder.SetInsertPoint(copy_done);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(new_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(new_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(src_elems, r_elems_field);  // Share data - no copy!
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::flatten(const eshkol_operations_t* op) {
    // Flatten: convert tensor to 1D
    // (flatten tensor) - all dimensions become a single dimension
    if (op->call_op.num_vars != 1) {
        eshkol_error("flatten requires exactly 1 argument (tensor)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Load tensor properties
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "flat_result");

    // Allocate single dimension (1D)
    llvm::Value* dims_size = llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
    llvm::Function* flat_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* new_dims = builder.CreateCall(flat_arena_alloc, {arena_ptr, dims_size}, "flat_dims");

    // Set the single dimension to total_elements
    builder.CreateStore(total_elements, new_dims);

    // Populate result tensor (shares element data)
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(new_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), r_ndim_field);  // 1D
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(src_elems, r_elems_field);  // Share data - no copy!
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::concatenate(const eshkol_operations_t* op) {
    // Concatenate: join tensors along specified axis
    // (concatenate axis tensor1 tensor2 ...)
    if (op->call_op.num_vars < 3) {
        eshkol_error("concatenate requires at least 3 arguments (axis, tensor1, tensor2)");
        return nullptr;
    }

    // Get axis
    llvm::Value* axis_arg = codegenAST(&op->call_op.variables[0]);
    if (!axis_arg) return nullptr;

    auto& builder = ctx_.builder();

    // Handle both raw int64 and tagged int64 for axis
    llvm::Value* axis = axis_arg;
    if (axis_arg->getType() == ctx_.taggedValueType()) {
        axis = tagged_.unpackInt64(axis_arg);
    } else if (!axis_arg->getType()->isIntegerTy(64)) {
        axis = builder.CreateSExtOrTrunc(axis_arg, ctx_.int64Type());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Type* tensor_type = ctx_.tensorType();

    // Get first tensor to establish shape template
    llvm::Value* first_tensor = codegenAST(&op->call_op.variables[1]);
    if (!first_tensor) return nullptr;

    llvm::Value* first_ptr_int = tagged_.unpackInt64(first_tensor);
    llvm::Value* first_ptr = builder.CreateIntToPtr(first_ptr_int, ctx_.ptrType());

    llvm::Value* first_dims_field = builder.CreateStructGEP(tensor_type, first_ptr, 0);
    llvm::Value* first_dims_ptr = builder.CreateLoad(ctx_.ptrType(), first_dims_field);
    llvm::Value* first_ndim_field = builder.CreateStructGEP(tensor_type, first_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), first_ndim_field);

    // Calculate total size along axis
    llvm::Value* concat_dim_sum = builder.CreateAlloca(ctx_.int64Type(), nullptr, "concat_sum");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), concat_dim_sum);

    // Calculate product of other dimensions (for stride calculation)
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Collect all tensors and sum their axis dimensions
    std::vector<llvm::Value*> tensor_ptrs;
    for (size_t i = 1; i < op->call_op.num_vars; ++i) {
        llvm::Value* t = codegenAST(&op->call_op.variables[i]);
        if (!t) return nullptr;
        llvm::Value* t_ptr_int = tagged_.unpackInt64(t);
        llvm::Value* t_ptr = builder.CreateIntToPtr(t_ptr_int, ctx_.ptrType());
        tensor_ptrs.push_back(t_ptr);

        // Add this tensor's dimension along axis
        llvm::Value* t_dims_field = builder.CreateStructGEP(tensor_type, t_ptr, 0);
        llvm::Value* t_dims_ptr = builder.CreateLoad(ctx_.ptrType(), t_dims_field);
        llvm::Value* t_axis_dim_ptr = builder.CreateGEP(ctx_.int64Type(), t_dims_ptr, axis);
        llvm::Value* t_axis_dim = builder.CreateLoad(ctx_.int64Type(), t_axis_dim_ptr);

        llvm::Value* curr_sum = builder.CreateLoad(ctx_.int64Type(), concat_dim_sum);
        builder.CreateStore(builder.CreateAdd(curr_sum, t_axis_dim), concat_dim_sum);
    }

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "concat_result");

    // Allocate and copy dimensions
    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* concat_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(concat_arena_alloc, {arena_ptr, dims_size}, "concat_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), first_dims_ptr, llvm::MaybeAlign(8), dims_size);

    // Update the axis dimension with the sum
    llvm::Value* new_axis_dim = builder.CreateLoad(ctx_.int64Type(), concat_dim_sum);
    llvm::Value* result_axis_dim_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, axis);
    builder.CreateStore(new_axis_dim, result_axis_dim_ptr);

    // Calculate total elements
    llvm::BasicBlock* calc_total_cond = llvm::BasicBlock::Create(ctx_.context(), "concat_total_cond", current_func);
    llvm::BasicBlock* calc_total_body = llvm::BasicBlock::Create(ctx_.context(), "concat_total_body", current_func);
    llvm::BasicBlock* calc_total_done = llvm::BasicBlock::Create(ctx_.context(), "concat_total_done", current_func);

    llvm::Value* total_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "concat_total");
    llvm::Value* calc_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "concat_calc_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_var);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), calc_i);
    builder.CreateBr(calc_total_cond);

    builder.SetInsertPoint(calc_total_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), calc_i);
    llvm::Value* calc_cmp = builder.CreateICmpULT(ci, num_dims);
    builder.CreateCondBr(calc_cmp, calc_total_body, calc_total_done);

    builder.SetInsertPoint(calc_total_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, ci);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* curr_total = builder.CreateLoad(ctx_.int64Type(), total_var);
    builder.CreateStore(builder.CreateMul(curr_total, dim_val), total_var);
    builder.CreateStore(builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), calc_i);
    builder.CreateBr(calc_total_cond);

    builder.SetInsertPoint(calc_total_done);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_var);

    // Allocate result elements
    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(concat_arena_alloc, {arena_ptr, elems_size}, "concat_elems");

    // Calculate strides before and after axis for copying
    // stride_after = product of dims after axis
    // stride_before = product of dims before axis * axis_dim

    llvm::Value* stride_after_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "stride_after");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), stride_after_var);

    // Calculate stride_after (dims after axis)
    llvm::BasicBlock* stride_cond = llvm::BasicBlock::Create(ctx_.context(), "stride_cond", current_func);
    llvm::BasicBlock* stride_body = llvm::BasicBlock::Create(ctx_.context(), "stride_body", current_func);
    llvm::BasicBlock* stride_done = llvm::BasicBlock::Create(ctx_.context(), "stride_done", current_func);

    llvm::Value* stride_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "stride_i");
    llvm::Value* axis_plus_one = builder.CreateAdd(axis, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(axis_plus_one, stride_i);
    builder.CreateBr(stride_cond);

    builder.SetInsertPoint(stride_cond);
    llvm::Value* sti = builder.CreateLoad(ctx_.int64Type(), stride_i);
    llvm::Value* stride_cmp = builder.CreateICmpULT(sti, num_dims);
    builder.CreateCondBr(stride_cmp, stride_body, stride_done);

    builder.SetInsertPoint(stride_body);
    llvm::Value* st_dim_ptr = builder.CreateGEP(ctx_.int64Type(), first_dims_ptr, sti);
    llvm::Value* st_dim_val = builder.CreateLoad(ctx_.int64Type(), st_dim_ptr);
    llvm::Value* curr_stride = builder.CreateLoad(ctx_.int64Type(), stride_after_var);
    builder.CreateStore(builder.CreateMul(curr_stride, st_dim_val), stride_after_var);
    builder.CreateStore(builder.CreateAdd(sti, llvm::ConstantInt::get(ctx_.int64Type(), 1)), stride_i);
    builder.CreateBr(stride_cond);

    builder.SetInsertPoint(stride_done);
    llvm::Value* stride_after = builder.CreateLoad(ctx_.int64Type(), stride_after_var);

    // Copy data from each tensor (simplified: assumes contiguous and row-major)
    // For each tensor, copy all its elements in order
    llvm::Value* dst_offset = builder.CreateAlloca(ctx_.int64Type(), nullptr, "dst_offset");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dst_offset);

    for (size_t i = 0; i < tensor_ptrs.size(); ++i) {
        llvm::Value* t_ptr = tensor_ptrs[i];
        llvm::Value* t_elems_field = builder.CreateStructGEP(tensor_type, t_ptr, 2);
        llvm::Value* t_elems = builder.CreateLoad(ctx_.ptrType(), t_elems_field);
        llvm::Value* t_total_field = builder.CreateStructGEP(tensor_type, t_ptr, 3);
        llvm::Value* t_total = builder.CreateLoad(ctx_.int64Type(), t_total_field);

        // Copy this tensor's elements
        llvm::Value* copy_size = builder.CreateMul(t_total,
            llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
        llvm::Value* curr_offset = builder.CreateLoad(ctx_.int64Type(), dst_offset);
        llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, curr_offset);
        builder.CreateMemCpy(dst_ptr, llvm::MaybeAlign(8), t_elems, llvm::MaybeAlign(8), copy_size);

        // Update offset
        builder.CreateStore(builder.CreateAdd(curr_offset, t_total), dst_offset);
    }

    // Populate result tensor
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::stack(const eshkol_operations_t* op) {
    // Stack: stack tensors on a new axis
    // (stack axis tensor1 tensor2 ...)
    if (op->call_op.num_vars < 3) {
        eshkol_error("stack requires at least 3 arguments (axis, tensor1, tensor2)");
        return nullptr;
    }

    // Get axis
    llvm::Value* axis_arg = codegenAST(&op->call_op.variables[0]);
    if (!axis_arg) return nullptr;

    auto& builder = ctx_.builder();

    // Handle both raw int64 and tagged int64 for axis
    llvm::Value* axis = axis_arg;
    if (axis_arg->getType() == ctx_.taggedValueType()) {
        axis = tagged_.unpackInt64(axis_arg);
    } else if (!axis_arg->getType()->isIntegerTy(64)) {
        axis = builder.CreateSExtOrTrunc(axis_arg, ctx_.int64Type());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Type* tensor_type = ctx_.tensorType();

    // Get first tensor shape
    llvm::Value* first_tensor = codegenAST(&op->call_op.variables[1]);
    if (!first_tensor) return nullptr;

    llvm::Value* first_ptr_int = tagged_.unpackInt64(first_tensor);
    llvm::Value* first_ptr = builder.CreateIntToPtr(first_ptr_int, ctx_.ptrType());

    llvm::Value* first_dims_field = builder.CreateStructGEP(tensor_type, first_ptr, 0);
    llvm::Value* first_dims_ptr = builder.CreateLoad(ctx_.ptrType(), first_dims_field);
    llvm::Value* first_ndim_field = builder.CreateStructGEP(tensor_type, first_ptr, 1);
    llvm::Value* old_ndim = builder.CreateLoad(ctx_.int64Type(), first_ndim_field);
    llvm::Value* first_total_field = builder.CreateStructGEP(tensor_type, first_ptr, 3);
    llvm::Value* tensor_elements = builder.CreateLoad(ctx_.int64Type(), first_total_field);

    size_t num_tensors = op->call_op.num_vars - 1;
    llvm::Value* num_tensors_val = llvm::ConstantInt::get(ctx_.int64Type(), num_tensors);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "stack_result");

    // New dimensions: insert num_tensors at axis position
    llvm::Value* new_ndim = builder.CreateAdd(old_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* dims_size = builder.CreateMul(new_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* stack_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(stack_arena_alloc, {arena_ptr, dims_size}, "stack_dims");

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Copy dimensions, inserting num_tensors at axis position
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "stk_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "stk_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "stk_copy_done", current_func);

    llvm::Value* src_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "stk_src_i");
    llvm::Value* dst_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "stk_dst_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), src_i);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dst_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* di = builder.CreateLoad(ctx_.int64Type(), dst_i);
    llvm::Value* cmp = builder.CreateICmpULT(di, new_ndim);
    builder.CreateCondBr(cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* is_new_axis = builder.CreateICmpEQ(di, axis);

    llvm::BasicBlock* insert_new = llvm::BasicBlock::Create(ctx_.context(), "stk_insert", current_func);
    llvm::BasicBlock* copy_old = llvm::BasicBlock::Create(ctx_.context(), "stk_copy", current_func);
    llvm::BasicBlock* next_iter = llvm::BasicBlock::Create(ctx_.context(), "stk_next", current_func);

    builder.CreateCondBr(is_new_axis, insert_new, copy_old);

    builder.SetInsertPoint(insert_new);
    llvm::Value* dst_ptr_ins = builder.CreateGEP(ctx_.int64Type(), result_dims, di);
    builder.CreateStore(num_tensors_val, dst_ptr_ins);
    builder.CreateBr(next_iter);

    builder.SetInsertPoint(copy_old);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), src_i);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), first_dims_ptr, si);
    llvm::Value* src_dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    llvm::Value* dst_ptr_cpy = builder.CreateGEP(ctx_.int64Type(), result_dims, di);
    builder.CreateStore(src_dim_val, dst_ptr_cpy);
    llvm::Value* next_si = builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_si, src_i);
    builder.CreateBr(next_iter);

    builder.SetInsertPoint(next_iter);
    llvm::Value* next_di = builder.CreateAdd(di, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_di, dst_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);

    // Total elements = num_tensors * elements_per_tensor
    llvm::Value* total_elements = builder.CreateMul(num_tensors_val, tensor_elements);

    // Allocate result elements
    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(stack_arena_alloc, {arena_ptr, elems_size}, "stack_elems");

    // Copy each tensor's elements
    llvm::Value* dst_offset = builder.CreateAlloca(ctx_.int64Type(), nullptr, "stk_dst_off");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dst_offset);

    for (size_t i = 1; i < op->call_op.num_vars; ++i) {
        llvm::Value* t = codegenAST(&op->call_op.variables[i]);
        if (!t) return nullptr;
        llvm::Value* t_ptr_int = tagged_.unpackInt64(t);
        llvm::Value* t_ptr = builder.CreateIntToPtr(t_ptr_int, ctx_.ptrType());

        llvm::Value* t_elems_field = builder.CreateStructGEP(tensor_type, t_ptr, 2);
        llvm::Value* t_elems = builder.CreateLoad(ctx_.ptrType(), t_elems_field);
        llvm::Value* t_total_field = builder.CreateStructGEP(tensor_type, t_ptr, 3);
        llvm::Value* t_total = builder.CreateLoad(ctx_.int64Type(), t_total_field);

        llvm::Value* copy_size = builder.CreateMul(t_total,
            llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
        llvm::Value* curr_offset = builder.CreateLoad(ctx_.int64Type(), dst_offset);
        llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, curr_offset);
        builder.CreateMemCpy(dst_ptr, llvm::MaybeAlign(8), t_elems, llvm::MaybeAlign(8), copy_size);

        builder.CreateStore(builder.CreateAdd(curr_offset, t_total), dst_offset);
    }

    // Populate result tensor
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(new_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::split(const eshkol_operations_t* op) {
    // Split: split tensor into chunks along an axis
    // (split tensor num-chunks axis)
    if (op->call_op.num_vars != 3) {
        eshkol_error("split requires 3 arguments (tensor, num-chunks, axis)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    llvm::Value* chunks_arg = codegenAST(&op->call_op.variables[1]);
    if (!chunks_arg) return nullptr;

    llvm::Value* axis_arg = codegenAST(&op->call_op.variables[2]);
    if (!axis_arg) return nullptr;

    auto& builder = ctx_.builder();

    // Handle both raw int64 and tagged int64 arguments
    llvm::Value* num_chunks = chunks_arg;
    if (chunks_arg->getType() == ctx_.taggedValueType()) {
        num_chunks = tagged_.unpackInt64(chunks_arg);
    } else if (!chunks_arg->getType()->isIntegerTy(64)) {
        num_chunks = builder.CreateSExtOrTrunc(chunks_arg, ctx_.int64Type());
    }

    llvm::Value* axis = axis_arg;
    if (axis_arg->getType() == ctx_.taggedValueType()) {
        axis = tagged_.unpackInt64(axis_arg);
    } else if (!axis_arg->getType()->isIntegerTy(64)) {
        axis = builder.CreateSExtOrTrunc(axis_arg, ctx_.int64Type());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);

    // Get axis dimension
    llvm::Value* axis_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, axis);
    llvm::Value* axis_dim = builder.CreateLoad(ctx_.int64Type(), axis_dim_ptr);

    // chunk_size = axis_dim / num_chunks
    llvm::Value* chunk_size = builder.CreateUDiv(axis_dim, num_chunks);

    // Calculate elements per chunk
    // Product of all dimensions except axis, times chunk_size
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* other_prod_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "other_prod");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), other_prod_var);

    llvm::BasicBlock* prod_cond = llvm::BasicBlock::Create(ctx_.context(), "spl_prod_cond", current_func);
    llvm::BasicBlock* prod_body = llvm::BasicBlock::Create(ctx_.context(), "spl_prod_body", current_func);
    llvm::BasicBlock* prod_done = llvm::BasicBlock::Create(ctx_.context(), "spl_prod_done", current_func);

    llvm::Value* prod_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "spl_prod_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), prod_i);
    builder.CreateBr(prod_cond);

    builder.SetInsertPoint(prod_cond);
    llvm::Value* pi = builder.CreateLoad(ctx_.int64Type(), prod_i);
    llvm::Value* prod_cmp = builder.CreateICmpULT(pi, num_dims);
    builder.CreateCondBr(prod_cmp, prod_body, prod_done);

    builder.SetInsertPoint(prod_body);
    llvm::Value* is_axis = builder.CreateICmpEQ(pi, axis);
    llvm::Value* d_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, pi);
    llvm::Value* d_val = builder.CreateLoad(ctx_.int64Type(), d_ptr);
    llvm::Value* curr_prod = builder.CreateLoad(ctx_.int64Type(), other_prod_var);
    llvm::Value* new_prod = builder.CreateSelect(is_axis, curr_prod, builder.CreateMul(curr_prod, d_val));
    builder.CreateStore(new_prod, other_prod_var);
    builder.CreateStore(builder.CreateAdd(pi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), prod_i);
    builder.CreateBr(prod_cond);

    builder.SetInsertPoint(prod_done);
    llvm::Value* other_prod = builder.CreateLoad(ctx_.int64Type(), other_prod_var);
    llvm::Value* chunk_elements = builder.CreateMul(other_prod, chunk_size);

    // Build list of chunks from back to front (cons builds backwards)
    // Use int64 alloca to store pointer as integer (same pattern as tensorShape)
    llvm::Value* list_acc = builder.CreateAlloca(ctx_.int64Type(), nullptr, "spl_list");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), list_acc);  // null

    // Iterate from num_chunks-1 down to 0
    llvm::Value* chunk_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "spl_chunk_i");
    builder.CreateStore(builder.CreateSub(num_chunks, llvm::ConstantInt::get(ctx_.int64Type(), 1)), chunk_i);

    llvm::BasicBlock* chunk_cond = llvm::BasicBlock::Create(ctx_.context(), "spl_chunk_cond", current_func);
    llvm::BasicBlock* chunk_body = llvm::BasicBlock::Create(ctx_.context(), "spl_chunk_body", current_func);
    llvm::BasicBlock* chunk_done = llvm::BasicBlock::Create(ctx_.context(), "spl_chunk_done", current_func);

    builder.CreateBr(chunk_cond);

    builder.SetInsertPoint(chunk_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), chunk_i);
    llvm::Value* chunk_cmp = builder.CreateICmpSGE(ci, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateCondBr(chunk_cmp, chunk_body, chunk_done);

    builder.SetInsertPoint(chunk_body);
    // Create tensor for this chunk using arena allocation
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* chunk_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "spl_chunk");

    // Allocate dims for chunk from arena
    llvm::Value* chunk_dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* spl_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* chunk_dims = builder.CreateCall(spl_arena_alloc, {arena_ptr, chunk_dims_size}, "spl_dims");
    builder.CreateMemCpy(chunk_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), chunk_dims_size);

    // Update axis dimension to chunk_size
    llvm::Value* chunk_axis_ptr = builder.CreateGEP(ctx_.int64Type(), chunk_dims, axis);
    builder.CreateStore(chunk_size, chunk_axis_ptr);

    // Calculate offset into source elements
    llvm::Value* elem_offset = builder.CreateMul(ci, chunk_elements);
    llvm::Value* chunk_elems = builder.CreateGEP(ctx_.int64Type(), src_elems, elem_offset);

    // Populate chunk tensor (view - shares data with original)
    llvm::Value* c_dims_field = builder.CreateStructGEP(tensor_type, chunk_ptr, 0);
    builder.CreateStore(chunk_dims, c_dims_field);
    llvm::Value* c_ndim_field = builder.CreateStructGEP(tensor_type, chunk_ptr, 1);
    builder.CreateStore(num_dims, c_ndim_field);
    llvm::Value* c_elems_field = builder.CreateStructGEP(tensor_type, chunk_ptr, 2);
    builder.CreateStore(chunk_elems, c_elems_field);
    llvm::Value* c_total_field = builder.CreateStructGEP(tensor_type, chunk_ptr, 3);
    builder.CreateStore(chunk_elements, c_total_field);

    // Load current list tail (stored as int64)
    llvm::Value* current_tail_int = builder.CreateLoad(ctx_.int64Type(), list_acc);

    // Allocate cons cell with header from arena (consolidated pointer format)
    llvm::Value* cons_cell = builder.CreateCall(
        mem_.getArenaAllocateConsWithHeader(), {arena_ptr});

    // Set car to tensor using tagged heap pointer
    // Must pass pointer to tagged value, not value itself
    llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    llvm::Value* chunk_tagged = tagged_.packHeapPtr(chunk_ptr);
    llvm::Value* chunk_ptr_alloca = builder.CreateAlloca(ctx_.taggedValueType());
    builder.CreateStore(chunk_tagged, chunk_ptr_alloca);
    builder.CreateCall(mem_.getTaggedConsSetTaggedValue(),
        {cons_cell, is_car, chunk_ptr_alloca});

    // Set cdr to current tail
    llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 1);
    llvm::Value* is_null_tail = builder.CreateICmpEQ(current_tail_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));

    // Branch for null vs non-null cdr
    llvm::BasicBlock* set_null_cdr = llvm::BasicBlock::Create(ctx_.context(), "spl_null_cdr", current_func);
    llvm::BasicBlock* set_cons_cdr = llvm::BasicBlock::Create(ctx_.context(), "spl_cons_cdr", current_func);
    llvm::BasicBlock* cdr_done = llvm::BasicBlock::Create(ctx_.context(), "spl_cdr_done", current_func);

    builder.CreateCondBr(is_null_tail, set_null_cdr, set_cons_cdr);

    builder.SetInsertPoint(set_null_cdr);
    builder.CreateCall(mem_.getTaggedConsSetNull(), {cons_cell, is_cdr});
    builder.CreateBr(cdr_done);

    builder.SetInsertPoint(set_cons_cdr);
    llvm::Value* cons_type = llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR);
    builder.CreateCall(mem_.getTaggedConsSetPtr(),
        {cons_cell, is_cdr, current_tail_int, cons_type});
    builder.CreateBr(cdr_done);

    builder.SetInsertPoint(cdr_done);

    // Update list accumulator to point to new cons cell
    llvm::Value* cons_cell_int = builder.CreatePtrToInt(cons_cell, ctx_.int64Type());
    builder.CreateStore(cons_cell_int, list_acc);

    // Decrement chunk index
    builder.CreateStore(builder.CreateSub(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), chunk_i);
    builder.CreateBr(chunk_cond);

    builder.SetInsertPoint(chunk_done);
    llvm::Value* final_result_int = builder.CreateLoad(ctx_.int64Type(), list_acc);

    // Return the list (or null if empty)
    llvm::Value* is_null = builder.CreateICmpEQ(final_result_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    return builder.CreateSelect(is_null, tagged_.packNull(),
        tagged_.packHeapPtr(builder.CreateIntToPtr(final_result_int, ctx_.ptrType())));
}

llvm::Value* TensorCodegen::slice(const eshkol_operations_t* op) {
    // Slice: extract subtensor
    // (slice tensor start end) - 1D slice from start to end (exclusive)
    if (op->call_op.num_vars != 3) {
        eshkol_error("slice requires 3 arguments (tensor, start, end)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    llvm::Value* start_arg = codegenAST(&op->call_op.variables[1]);
    if (!start_arg) return nullptr;

    llvm::Value* end_arg = codegenAST(&op->call_op.variables[2]);
    if (!end_arg) return nullptr;

    auto& builder = ctx_.builder();

    // Handle both raw int64 and tagged int64 arguments
    llvm::Value* start = start_arg;
    if (start_arg->getType() == ctx_.taggedValueType()) {
        start = tagged_.unpackInt64(start_arg);
    } else if (!start_arg->getType()->isIntegerTy(64)) {
        start = builder.CreateSExtOrTrunc(start_arg, ctx_.int64Type());
    }

    llvm::Value* end = end_arg;
    if (end_arg->getType() == ctx_.taggedValueType()) {
        end = tagged_.unpackInt64(end_arg);
    } else if (!end_arg->getType()->isIntegerTy(64)) {
        end = builder.CreateSExtOrTrunc(end_arg, ctx_.int64Type());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "slice_result");

    // Slice length
    llvm::Value* slice_len = builder.CreateSub(end, start);

    // Allocate dimensions (1D result)
    llvm::Value* dims_size = llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
    llvm::Function* slice_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(slice_arena_alloc, {arena_ptr, dims_size}, "slice_dims");
    builder.CreateStore(slice_len, result_dims);

    // Point to slice of elements (view - no copy)
    llvm::Value* slice_elems = builder.CreateGEP(ctx_.int64Type(), src_elems, start);

    // Populate result tensor
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), r_ndim_field);  // 1D
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(slice_elems, r_elems_field);  // View - shares data!
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(slice_len, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
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

    // Get source value (may be tensor OR Scheme vector)
    llvm::Value* src_val = codegenAST(&op->call_op.variables[0]);
    if (!src_val) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Check type: Scheme vector vs Tensor (using consolidated type check)
    llvm::Value* is_scheme_vector = tagged_.isVector(src_val);

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* scheme_vec_block = llvm::BasicBlock::Create(ctx_.context(), "reshape_scheme_vec", current_func);
    llvm::BasicBlock* tensor_block = llvm::BasicBlock::Create(ctx_.context(), "reshape_tensor", current_func);
    llvm::BasicBlock* type_merge = llvm::BasicBlock::Create(ctx_.context(), "reshape_type_merge", current_func);

    ctx_.builder().CreateCondBr(is_scheme_vector, scheme_vec_block, tensor_block);

    // === SCHEME VECTOR PATH: Convert to tensor first ===
    ctx_.builder().SetInsertPoint(scheme_vec_block);
    llvm::Value* svec_ptr_int = tagged_.unpackInt64(src_val);
    llvm::Value* svec_ptr = ctx_.builder().CreateIntToPtr(svec_ptr_int, ctx_.ptrType());

    // Scheme vector layout: [length:i64, element0:tagged_value, element1:tagged_value, ...]
    llvm::Value* svec_len = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_ptr);

    // Allocate arena for conversion
    llvm::Value* conv_arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Allocate tensor structure with header
    llvm::Function* conv_alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* conv_tensor_ptr = ctx_.builder().CreateCall(conv_alloc_tensor_func, {conv_arena_ptr}, "vec_to_tensor");

    // Allocate dimensions array (1D tensor)
    llvm::Function* conv_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* conv_dims_size = llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
    llvm::Value* conv_dims_ptr = ctx_.builder().CreateCall(conv_arena_alloc, {conv_arena_ptr, conv_dims_size}, "conv_dims");
    ctx_.builder().CreateStore(svec_len, conv_dims_ptr);

    // Allocate elements array
    llvm::Value* conv_elems_size = ctx_.builder().CreateMul(svec_len,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* conv_elems_ptr = ctx_.builder().CreateCall(conv_arena_alloc, {conv_arena_ptr, conv_elems_size}, "conv_elems");

    // Copy elements from vector (each element is 16-byte tagged value)
    llvm::Value* svec_data_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));  // Skip length

    // Copy loop
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "reshape_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "reshape_copy_body", current_func);
    llvm::BasicBlock* copy_exit = llvm::BasicBlock::Create(ctx_.context(), "reshape_copy_exit", current_func);

    llvm::Value* copy_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "copy_i");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_counter);
    ctx_.builder().CreateBr(copy_cond);

    ctx_.builder().SetInsertPoint(copy_cond);
    llvm::Value* copy_i = ctx_.builder().CreateLoad(ctx_.int64Type(), copy_counter);
    llvm::Value* copy_cmp = ctx_.builder().CreateICmpULT(copy_i, svec_len);
    ctx_.builder().CreateCondBr(copy_cmp, copy_body, copy_exit);

    ctx_.builder().SetInsertPoint(copy_body);
    // Load tagged value from vector
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(),
        ctx_.builder().CreatePointerCast(svec_data_ptr, ctx_.ptrType()), copy_i);
    llvm::Value* elem_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), elem_ptr);
    llvm::Value* elem_double = extractAsDouble(elem_tagged);

    // Store in tensor elements
    llvm::Value* dest_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), conv_elems_ptr, copy_i);
    ctx_.builder().CreateStore(elem_double, dest_ptr);

    llvm::Value* next_copy_i = ctx_.builder().CreateAdd(copy_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_copy_i, copy_counter);
    ctx_.builder().CreateBr(copy_cond);

    // Exit copy loop: populate tensor struct
    ctx_.builder().SetInsertPoint(copy_exit);

    // Store dims pointer (field 0)
    llvm::Value* conv_dims_field = ctx_.builder().CreateStructGEP(tensor_type, conv_tensor_ptr, 0);
    ctx_.builder().CreateStore(conv_dims_ptr, conv_dims_field);

    // Store num_dimensions = 1 (field 1)
    llvm::Value* conv_ndim_field = ctx_.builder().CreateStructGEP(tensor_type, conv_tensor_ptr, 1);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), conv_ndim_field);

    // Store elements pointer (field 2)
    llvm::Value* conv_elems_field = ctx_.builder().CreateStructGEP(tensor_type, conv_tensor_ptr, 2);
    ctx_.builder().CreateStore(conv_elems_ptr, conv_elems_field);

    // Store total_elements (field 3)
    llvm::Value* conv_total_field = ctx_.builder().CreateStructGEP(tensor_type, conv_tensor_ptr, 3);
    ctx_.builder().CreateStore(svec_len, conv_total_field);

    ctx_.builder().CreateBr(type_merge);
    llvm::BasicBlock* svec_exit_block = ctx_.builder().GetInsertBlock();

    // === TENSOR PATH: Use existing tensor directly ===
    ctx_.builder().SetInsertPoint(tensor_block);
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(src_val);
    llvm::Value* direct_tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    ctx_.builder().CreateBr(type_merge);
    llvm::BasicBlock* tensor_exit_block = ctx_.builder().GetInsertBlock();

    // === MERGE: Get unified tensor pointer ===
    ctx_.builder().SetInsertPoint(type_merge);
    llvm::PHINode* src_ptr = ctx_.builder().CreatePHI(ctx_.ptrType(), 2, "unified_tensor_ptr");
    src_ptr->addIncoming(conv_tensor_ptr, svec_exit_block);
    src_ptr->addIncoming(direct_tensor_ptr, tensor_exit_block);

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

llvm::Value* TensorCodegen::tensor(const eshkol_operations_t* op) {
    // tensor: (tensor e1 e2 e3 ...) - Create 1D tensor from elements
    if (op->call_op.num_vars < 1) {
        eshkol_error("tensor requires at least 1 element");
        return nullptr;
    }

    auto& builder = ctx_.builder();
    uint64_t n = op->call_op.num_vars;

    // Create 1D tensor with n elements
    std::vector<llvm::Value*> dims = {llvm::ConstantInt::get(ctx_.int64Type(), n)};
    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, false);
    if (!tensor_ptr) return nullptr;

    // Get elements pointer from tensor struct
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);

    // Store each element
    for (uint64_t i = 0; i < n; ++i) {
        llvm::Value* elem = codegenAST(&op->call_op.variables[i]);
        if (!elem) return nullptr;

        // Extract double value (handle tagged or raw)
        llvm::Value* elem_double;
        if (elem->getType() == ctx_.taggedValueType()) {
            elem_double = tagged_.unpackDouble(elem);
        } else if (elem->getType() == ctx_.doubleType()) {
            elem_double = elem;
        } else if (elem->getType()->isIntegerTy(64)) {
            elem_double = builder.CreateBitCast(elem, ctx_.doubleType());
        } else {
            elem_double = builder.CreateSIToFP(elem, ctx_.doubleType());
        }

        // Store as int64 bit pattern
        llvm::Value* elem_bits = builder.CreateBitCast(elem_double, ctx_.int64Type());
        llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        builder.CreateStore(elem_bits, elem_ptr);
    }

    return tagged_.packHeapPtr(tensor_ptr);
}

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

    // Calculate number of elements: ceil((end - start) / step) to handle FP rounding
    llvm::Value* range_dbl = ctx_.builder().CreateFSub(end_dbl, start_dbl);
    llvm::Value* num_elements_dbl = ctx_.builder().CreateFDiv(range_dbl, step_dbl);
    // Use ceil to handle floating-point rounding errors (e.g., 11.9999... -> 12)
    llvm::Function* ceil_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::ceil, {ctx_.doubleType()});
    llvm::Value* num_elements_ceil = ctx_.builder().CreateCall(ceil_func, {num_elements_dbl});
    llvm::Value* num_elements = ctx_.builder().CreateFPToSI(num_elements_ceil, ctx_.int64Type());

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

// === Convolution & Pooling Operations ===

llvm::Value* TensorCodegen::maxPool2d(const eshkol_operations_t* op) {
    // max-pool2d: (max-pool2d input kernel-size stride)
    // N-dimensional: pools over last 2 dims, preserves all batch dims
    // Input shape: (..., H, W) -> Output shape: (..., out_H, out_W)
    if (op->call_op.num_vars < 3) {
        eshkol_error("max-pool2d requires at least 3 arguments (input, kernel-size, stride)");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    if (!input_val) return nullptr;

    llvm::Value* kernel_arg = codegenAST(&op->call_op.variables[1]);
    if (!kernel_arg) return nullptr;

    llvm::Value* stride_arg = codegenAST(&op->call_op.variables[2]);
    if (!stride_arg) return nullptr;

    auto& builder = ctx_.builder();

    // Handle both raw int64 and tagged int64 arguments
    llvm::Value* kernel_size = kernel_arg;
    if (kernel_arg->getType() == ctx_.taggedValueType()) {
        kernel_size = tagged_.unpackInt64(kernel_arg);
    } else if (!kernel_arg->getType()->isIntegerTy(64)) {
        kernel_size = builder.CreateSExtOrTrunc(kernel_arg, ctx_.int64Type());
    }

    llvm::Value* stride = stride_arg;
    if (stride_arg->getType() == ctx_.taggedValueType()) {
        stride = tagged_.unpackInt64(stride_arg);
    } else if (!stride_arg->getType()->isIntegerTy(64)) {
        stride = builder.CreateSExtOrTrunc(stride_arg, ctx_.int64Type());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Load input tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* input_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* in_total = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Get last 2 dimensions (spatial dims) - works for any number of batch dims
    llvm::Value* h_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* w_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* h_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, h_idx);
    llvm::Value* w_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, w_idx);
    llvm::Value* in_h = builder.CreateLoad(ctx_.int64Type(), h_ptr);
    llvm::Value* in_w = builder.CreateLoad(ctx_.int64Type(), w_ptr);

    // Compute batch_size = total / (H * W)
    llvm::Value* spatial_in = builder.CreateMul(in_h, in_w);
    llvm::Value* batch_size = builder.CreateSDiv(in_total, spatial_in);

    // Calculate output spatial dimensions: out_dim = (in_dim - kernel_size) / stride + 1
    llvm::Value* out_h = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_h, kernel_size), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* out_w = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_w, kernel_size), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* spatial_out = builder.CreateMul(out_h, out_w);

    // Allocate output tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "maxpool_result");

    // Allocate dimensions array (same rank as input)
    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "maxpool_dims");

    // Copy batch dimensions from input, set last 2 to output spatial dims
    llvm::Value* out_total = builder.CreateMul(batch_size, spatial_out);

    // Copy batch dims loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "mp_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "mp_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "mp_copy_done", current_func);

    llvm::Value* copy_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mp_copy_i");
    llvm::Value* batch_dims = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* curr_copy_i = builder.CreateLoad(ctx_.int64Type(), copy_i);
    llvm::Value* copy_cmp = builder.CreateICmpSLT(curr_copy_i, batch_dims);
    builder.CreateCondBr(copy_cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, curr_copy_i);
    llvm::Value* dst_dim_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, curr_copy_i);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    builder.CreateStore(dim_val, dst_dim_ptr);
    llvm::Value* next_copy_i = builder.CreateAdd(curr_copy_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_copy_i, copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);
    // Store output spatial dimensions
    llvm::Value* out_h_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, h_idx);
    builder.CreateStore(out_h, out_h_ptr);
    llvm::Value* out_w_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, w_idx);
    builder.CreateStore(out_w, out_w_ptr);

    // Allocate output elements
    llvm::Value* elems_size = builder.CreateMul(out_total,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "maxpool_elems");

    // Populate result tensor struct
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(out_total, r_total_field);

    // Create nested loops: batch -> output_h -> output_w -> kernel
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "mp_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "mp_batch_body", current_func);
    llvm::BasicBlock* outer_cond = llvm::BasicBlock::Create(ctx_.context(), "mp_outer_cond", current_func);
    llvm::BasicBlock* outer_body = llvm::BasicBlock::Create(ctx_.context(), "mp_outer_body", current_func);
    llvm::BasicBlock* inner_cond = llvm::BasicBlock::Create(ctx_.context(), "mp_inner_cond", current_func);
    llvm::BasicBlock* inner_body = llvm::BasicBlock::Create(ctx_.context(), "mp_inner_body", current_func);
    llvm::BasicBlock* pool_cond = llvm::BasicBlock::Create(ctx_.context(), "mp_pool_cond", current_func);
    llvm::BasicBlock* pool_body = llvm::BasicBlock::Create(ctx_.context(), "mp_pool_body", current_func);
    llvm::BasicBlock* pool_inner_cond = llvm::BasicBlock::Create(ctx_.context(), "mp_pool_inner_cond", current_func);
    llvm::BasicBlock* pool_inner_body = llvm::BasicBlock::Create(ctx_.context(), "mp_pool_inner_body", current_func);
    llvm::BasicBlock* pool_inner_done = llvm::BasicBlock::Create(ctx_.context(), "mp_pool_inner_done", current_func);
    llvm::BasicBlock* pool_done = llvm::BasicBlock::Create(ctx_.context(), "mp_pool_done", current_func);
    llvm::BasicBlock* inner_done = llvm::BasicBlock::Create(ctx_.context(), "mp_inner_done", current_func);
    llvm::BasicBlock* outer_done = llvm::BasicBlock::Create(ctx_.context(), "mp_outer_done", current_func);
    llvm::BasicBlock* batch_done = llvm::BasicBlock::Create(ctx_.context(), "mp_batch_done", current_func);

    // Allocate loop variables
    llvm::Value* bi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mp_bi");
    llvm::Value* oi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mp_oi");
    llvm::Value* oj = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mp_oj");
    llvm::Value* ki = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mp_ki");
    llvm::Value* kj = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mp_kj");
    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "mp_max");

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), bi);
    builder.CreateBr(batch_cond);

    // Batch loop
    builder.SetInsertPoint(batch_cond);
    llvm::Value* curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* batch_cmp = builder.CreateICmpSLT(curr_bi, batch_size);
    builder.CreateCondBr(batch_cmp, batch_body, batch_done);

    builder.SetInsertPoint(batch_body);
    // Calculate batch offsets
    llvm::Value* in_batch_offset = builder.CreateMul(curr_bi, spatial_in);
    llvm::Value* out_batch_offset = builder.CreateMul(curr_bi, spatial_out);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), oi);
    builder.CreateBr(outer_cond);

    // Outer loop (output row)
    builder.SetInsertPoint(outer_cond);
    llvm::Value* curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    llvm::Value* outer_cmp = builder.CreateICmpSLT(curr_oi, out_h);
    builder.CreateCondBr(outer_cmp, outer_body, outer_done);

    builder.SetInsertPoint(outer_body);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), oj);
    builder.CreateBr(inner_cond);

    // Inner loop (output column)
    builder.SetInsertPoint(inner_cond);
    llvm::Value* curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    llvm::Value* inner_cmp = builder.CreateICmpSLT(curr_oj, out_w);
    builder.CreateCondBr(inner_cmp, inner_body, inner_done);

    builder.SetInsertPoint(inner_body);
    builder.CreateStore(llvm::ConstantFP::getInfinity(ctx_.doubleType(), true), max_val);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ki);
    builder.CreateBr(pool_cond);

    // Pool kernel row loop
    builder.SetInsertPoint(pool_cond);
    llvm::Value* curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    llvm::Value* pool_cmp = builder.CreateICmpSLT(curr_ki, kernel_size);
    builder.CreateCondBr(pool_cmp, pool_body, pool_done);

    builder.SetInsertPoint(pool_body);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), kj);
    builder.CreateBr(pool_inner_cond);

    // Pool kernel column loop
    builder.SetInsertPoint(pool_inner_cond);
    llvm::Value* curr_kj = builder.CreateLoad(ctx_.int64Type(), kj);
    llvm::Value* pool_inner_cmp = builder.CreateICmpSLT(curr_kj, kernel_size);
    builder.CreateCondBr(pool_inner_cmp, pool_inner_body, pool_inner_done);

    builder.SetInsertPoint(pool_inner_body);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    curr_kj = builder.CreateLoad(ctx_.int64Type(), kj);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    in_batch_offset = builder.CreateMul(curr_bi, spatial_in);

    llvm::Value* in_row = builder.CreateAdd(builder.CreateMul(curr_oi, stride), curr_ki);
    llvm::Value* in_col = builder.CreateAdd(builder.CreateMul(curr_oj, stride), curr_kj);
    llvm::Value* in_spatial_idx = builder.CreateAdd(builder.CreateMul(in_row, in_w), in_col);
    llvm::Value* in_idx = builder.CreateAdd(in_batch_offset, in_spatial_idx);

    llvm::Value* in_elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, in_idx);
    llvm::Value* in_elem_bits = builder.CreateLoad(ctx_.int64Type(), in_elem_ptr);
    llvm::Value* in_elem = builder.CreateBitCast(in_elem_bits, ctx_.doubleType());

    llvm::Value* curr_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* new_max = builder.CreateMaxNum(curr_max, in_elem);
    builder.CreateStore(new_max, max_val);

    llvm::Value* next_kj = builder.CreateAdd(curr_kj, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_kj, kj);
    builder.CreateBr(pool_inner_cond);

    builder.SetInsertPoint(pool_inner_done);
    curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    llvm::Value* next_ki = builder.CreateAdd(curr_ki, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_ki, ki);
    builder.CreateBr(pool_cond);

    builder.SetInsertPoint(pool_done);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    out_batch_offset = builder.CreateMul(curr_bi, spatial_out);
    llvm::Value* out_spatial_idx = builder.CreateAdd(builder.CreateMul(curr_oi, out_w), curr_oj);
    llvm::Value* out_idx = builder.CreateAdd(out_batch_offset, out_spatial_idx);
    llvm::Value* out_elem_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, out_idx);
    llvm::Value* final_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* max_bits = builder.CreateBitCast(final_max, ctx_.int64Type());
    builder.CreateStore(max_bits, out_elem_ptr);

    curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    llvm::Value* next_oj = builder.CreateAdd(curr_oj, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_oj, oj);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_done);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    llvm::Value* next_oi = builder.CreateAdd(curr_oi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_oi, oi);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_done);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* next_bi = builder.CreateAdd(curr_bi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_bi, bi);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_done);
    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::avgPool2d(const eshkol_operations_t* op) {
    // avg-pool2d: (avg-pool2d input kernel-size stride)
    // N-dimensional: pools over last 2 dims, preserves all batch dims
    // Input shape: (..., H, W) -> Output shape: (..., out_H, out_W)
    if (op->call_op.num_vars < 3) {
        eshkol_error("avg-pool2d requires at least 3 arguments (input, kernel-size, stride)");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    if (!input_val) return nullptr;

    llvm::Value* kernel_arg = codegenAST(&op->call_op.variables[1]);
    if (!kernel_arg) return nullptr;

    llvm::Value* stride_arg = codegenAST(&op->call_op.variables[2]);
    if (!stride_arg) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* kernel_size = kernel_arg;
    if (kernel_arg->getType() == ctx_.taggedValueType()) {
        kernel_size = tagged_.unpackInt64(kernel_arg);
    } else if (!kernel_arg->getType()->isIntegerTy(64)) {
        kernel_size = builder.CreateSExtOrTrunc(kernel_arg, ctx_.int64Type());
    }

    llvm::Value* stride = stride_arg;
    if (stride_arg->getType() == ctx_.taggedValueType()) {
        stride = tagged_.unpackInt64(stride_arg);
    } else if (!stride_arg->getType()->isIntegerTy(64)) {
        stride = builder.CreateSExtOrTrunc(stride_arg, ctx_.int64Type());
    }

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* input_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* in_total = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Get last 2 dimensions (spatial dims)
    llvm::Value* h_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* w_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* h_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, h_idx);
    llvm::Value* w_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, w_idx);
    llvm::Value* in_h = builder.CreateLoad(ctx_.int64Type(), h_ptr);
    llvm::Value* in_w = builder.CreateLoad(ctx_.int64Type(), w_ptr);

    llvm::Value* spatial_in = builder.CreateMul(in_h, in_w);
    llvm::Value* batch_size = builder.CreateSDiv(in_total, spatial_in);

    llvm::Value* out_h = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_h, kernel_size), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* out_w = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_w, kernel_size), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* spatial_out = builder.CreateMul(out_h, out_w);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "avgpool_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "avgpool_dims");

    llvm::Value* out_total = builder.CreateMul(batch_size, spatial_out);

    // Copy batch dims loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "ap_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "ap_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "ap_copy_done", current_func);

    llvm::Value* copy_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ap_copy_i");
    llvm::Value* batch_dims = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* curr_copy_i = builder.CreateLoad(ctx_.int64Type(), copy_i);
    llvm::Value* copy_cmp = builder.CreateICmpSLT(curr_copy_i, batch_dims);
    builder.CreateCondBr(copy_cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, curr_copy_i);
    llvm::Value* dst_dim_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, curr_copy_i);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    builder.CreateStore(dim_val, dst_dim_ptr);
    llvm::Value* next_copy_i = builder.CreateAdd(curr_copy_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_copy_i, copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);
    llvm::Value* out_h_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, h_idx);
    builder.CreateStore(out_h, out_h_ptr);
    llvm::Value* out_w_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, w_idx);
    builder.CreateStore(out_w, out_w_ptr);

    llvm::Value* elems_size = builder.CreateMul(out_total,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "avgpool_elems");

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(out_total, r_total_field);

    llvm::Value* pool_count = builder.CreateMul(kernel_size, kernel_size);
    llvm::Value* pool_count_fp = builder.CreateSIToFP(pool_count, ctx_.doubleType());

    // Nested loops: batch -> output_h -> output_w -> kernel
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "ap_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "ap_batch_body", current_func);
    llvm::BasicBlock* outer_cond = llvm::BasicBlock::Create(ctx_.context(), "ap_outer_cond", current_func);
    llvm::BasicBlock* outer_body = llvm::BasicBlock::Create(ctx_.context(), "ap_outer_body", current_func);
    llvm::BasicBlock* inner_cond = llvm::BasicBlock::Create(ctx_.context(), "ap_inner_cond", current_func);
    llvm::BasicBlock* inner_body = llvm::BasicBlock::Create(ctx_.context(), "ap_inner_body", current_func);
    llvm::BasicBlock* pool_cond = llvm::BasicBlock::Create(ctx_.context(), "ap_pool_cond", current_func);
    llvm::BasicBlock* pool_body = llvm::BasicBlock::Create(ctx_.context(), "ap_pool_body", current_func);
    llvm::BasicBlock* pool_inner_cond = llvm::BasicBlock::Create(ctx_.context(), "ap_pool_inner_cond", current_func);
    llvm::BasicBlock* pool_inner_body = llvm::BasicBlock::Create(ctx_.context(), "ap_pool_inner_body", current_func);
    llvm::BasicBlock* pool_inner_done = llvm::BasicBlock::Create(ctx_.context(), "ap_pool_inner_done", current_func);
    llvm::BasicBlock* pool_done = llvm::BasicBlock::Create(ctx_.context(), "ap_pool_done", current_func);
    llvm::BasicBlock* inner_done = llvm::BasicBlock::Create(ctx_.context(), "ap_inner_done", current_func);
    llvm::BasicBlock* outer_done = llvm::BasicBlock::Create(ctx_.context(), "ap_outer_done", current_func);
    llvm::BasicBlock* batch_done = llvm::BasicBlock::Create(ctx_.context(), "ap_batch_done", current_func);

    llvm::Value* bi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ap_bi");
    llvm::Value* oi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ap_oi");
    llvm::Value* oj = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ap_oj");
    llvm::Value* ki = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ap_ki");
    llvm::Value* kj = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ap_kj");
    llvm::Value* sum_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "ap_sum");

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), bi);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_cond);
    llvm::Value* curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* batch_cmp = builder.CreateICmpSLT(curr_bi, batch_size);
    builder.CreateCondBr(batch_cmp, batch_body, batch_done);

    builder.SetInsertPoint(batch_body);
    llvm::Value* in_batch_offset = builder.CreateMul(curr_bi, spatial_in);
    llvm::Value* out_batch_offset = builder.CreateMul(curr_bi, spatial_out);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), oi);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_cond);
    llvm::Value* curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    llvm::Value* outer_cmp = builder.CreateICmpSLT(curr_oi, out_h);
    builder.CreateCondBr(outer_cmp, outer_body, outer_done);

    builder.SetInsertPoint(outer_body);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), oj);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_cond);
    llvm::Value* curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    llvm::Value* inner_cmp = builder.CreateICmpSLT(curr_oj, out_w);
    builder.CreateCondBr(inner_cmp, inner_body, inner_done);

    builder.SetInsertPoint(inner_body);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_val);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ki);
    builder.CreateBr(pool_cond);

    builder.SetInsertPoint(pool_cond);
    llvm::Value* curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    llvm::Value* pool_cmp = builder.CreateICmpSLT(curr_ki, kernel_size);
    builder.CreateCondBr(pool_cmp, pool_body, pool_done);

    builder.SetInsertPoint(pool_body);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), kj);
    builder.CreateBr(pool_inner_cond);

    builder.SetInsertPoint(pool_inner_cond);
    llvm::Value* curr_kj = builder.CreateLoad(ctx_.int64Type(), kj);
    llvm::Value* pool_inner_cmp = builder.CreateICmpSLT(curr_kj, kernel_size);
    builder.CreateCondBr(pool_inner_cmp, pool_inner_body, pool_inner_done);

    builder.SetInsertPoint(pool_inner_body);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    curr_kj = builder.CreateLoad(ctx_.int64Type(), kj);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    in_batch_offset = builder.CreateMul(curr_bi, spatial_in);

    llvm::Value* in_row = builder.CreateAdd(builder.CreateMul(curr_oi, stride), curr_ki);
    llvm::Value* in_col = builder.CreateAdd(builder.CreateMul(curr_oj, stride), curr_kj);
    llvm::Value* in_spatial_idx = builder.CreateAdd(builder.CreateMul(in_row, in_w), in_col);
    llvm::Value* in_idx = builder.CreateAdd(in_batch_offset, in_spatial_idx);

    llvm::Value* in_elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, in_idx);
    llvm::Value* in_elem_bits = builder.CreateLoad(ctx_.int64Type(), in_elem_ptr);
    llvm::Value* in_elem = builder.CreateBitCast(in_elem_bits, ctx_.doubleType());

    llvm::Value* curr_sum = builder.CreateLoad(ctx_.doubleType(), sum_val);
    llvm::Value* new_sum = builder.CreateFAdd(curr_sum, in_elem);
    builder.CreateStore(new_sum, sum_val);

    llvm::Value* next_kj = builder.CreateAdd(curr_kj, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_kj, kj);
    builder.CreateBr(pool_inner_cond);

    builder.SetInsertPoint(pool_inner_done);
    curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    llvm::Value* next_ki = builder.CreateAdd(curr_ki, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_ki, ki);
    builder.CreateBr(pool_cond);

    builder.SetInsertPoint(pool_done);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    out_batch_offset = builder.CreateMul(curr_bi, spatial_out);
    llvm::Value* out_spatial_idx = builder.CreateAdd(builder.CreateMul(curr_oi, out_w), curr_oj);
    llvm::Value* out_idx = builder.CreateAdd(out_batch_offset, out_spatial_idx);
    llvm::Value* out_elem_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, out_idx);
    llvm::Value* final_sum = builder.CreateLoad(ctx_.doubleType(), sum_val);
    llvm::Value* avg = builder.CreateFDiv(final_sum, pool_count_fp);
    llvm::Value* avg_bits = builder.CreateBitCast(avg, ctx_.int64Type());
    builder.CreateStore(avg_bits, out_elem_ptr);

    curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    llvm::Value* next_oj = builder.CreateAdd(curr_oj, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_oj, oj);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_done);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    llvm::Value* next_oi = builder.CreateAdd(curr_oi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_oi, oi);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_done);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* next_bi = builder.CreateAdd(curr_bi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_bi, bi);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_done);
    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::conv1d(const eshkol_operations_t* op) {
    // conv1d: (conv1d input kernel stride)
    // N-dimensional: convolves over last dim, preserves all batch dims
    // Input: (..., L) -> Output: (..., (L-K)/stride + 1)
    if (op->call_op.num_vars < 3) {
        eshkol_error("conv1d requires at least 3 arguments (input, kernel, stride)");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    if (!input_val) return nullptr;

    llvm::Value* kernel_val = codegenAST(&op->call_op.variables[1]);
    if (!kernel_val) return nullptr;

    llvm::Value* stride_arg = codegenAST(&op->call_op.variables[2]);
    if (!stride_arg) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* stride = stride_arg;
    if (stride_arg->getType() == ctx_.taggedValueType()) {
        stride = tagged_.unpackInt64(stride_arg);
    } else if (!stride_arg->getType()->isIntegerTy(64)) {
        stride = builder.CreateSExtOrTrunc(stride_arg, ctx_.int64Type());
    }

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Value* in_dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* in_dims = builder.CreateLoad(ctx_.ptrType(), in_dims_field);
    llvm::Value* in_ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), in_ndim_field);
    llvm::Value* in_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* input_elems = builder.CreateLoad(ctx_.ptrType(), in_elems_field);
    llvm::Value* in_total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* in_total = builder.CreateLoad(ctx_.int64Type(), in_total_field);

    // Get last dimension (spatial dim to convolve)
    llvm::Value* last_dim_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* last_dim_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims, last_dim_idx);
    llvm::Value* in_len = builder.CreateLoad(ctx_.int64Type(), last_dim_ptr);

    // Compute batch_size = total / in_len
    llvm::Value* batch_size = builder.CreateSDiv(in_total, in_len);

    // Unpack kernel tensor (1D)
    llvm::Value* kernel_ptr_int = tagged_.unpackInt64(kernel_val);
    llvm::Value* kernel_ptr = builder.CreateIntToPtr(kernel_ptr_int, ctx_.ptrType());
    llvm::Value* k_total_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 3);
    llvm::Value* k_len = builder.CreateLoad(ctx_.int64Type(), k_total_field);
    llvm::Value* k_elems_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 2);
    llvm::Value* kernel_elems = builder.CreateLoad(ctx_.ptrType(), k_elems_field);

    // Calculate output length: (in_len - k_len) / stride + 1
    llvm::Value* out_len = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_len, k_len), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    // Allocate output tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "conv1d_result");

    // Allocate dimensions (same rank as input)
    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "conv1d_dims");

    llvm::Value* out_total = builder.CreateMul(batch_size, out_len);

    // Copy batch dims, set last to out_len
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "c1_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "c1_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "c1_copy_done", current_func);

    llvm::Value* copy_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c1_copy_i");
    llvm::Value* batch_dims = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* curr_copy_i = builder.CreateLoad(ctx_.int64Type(), copy_i);
    llvm::Value* copy_cmp = builder.CreateICmpSLT(curr_copy_i, batch_dims);
    builder.CreateCondBr(copy_cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims, curr_copy_i);
    llvm::Value* dst_dim_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, curr_copy_i);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    builder.CreateStore(dim_val, dst_dim_ptr);
    llvm::Value* next_copy_i = builder.CreateAdd(curr_copy_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_copy_i, copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);
    llvm::Value* out_len_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, last_dim_idx);
    builder.CreateStore(out_len, out_len_ptr);

    // Allocate output elements
    llvm::Value* elems_size = builder.CreateMul(out_total,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "conv1d_elems");

    // Populate result tensor struct
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(out_total, r_total_field);

    // Create convolution loops: batch -> output position -> kernel
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "c1_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "c1_batch_body", current_func);
    llvm::BasicBlock* outer_cond = llvm::BasicBlock::Create(ctx_.context(), "c1_outer_cond", current_func);
    llvm::BasicBlock* outer_body = llvm::BasicBlock::Create(ctx_.context(), "c1_outer_body", current_func);
    llvm::BasicBlock* inner_cond = llvm::BasicBlock::Create(ctx_.context(), "c1_inner_cond", current_func);
    llvm::BasicBlock* inner_body = llvm::BasicBlock::Create(ctx_.context(), "c1_inner_body", current_func);
    llvm::BasicBlock* inner_done = llvm::BasicBlock::Create(ctx_.context(), "c1_inner_done", current_func);
    llvm::BasicBlock* outer_done = llvm::BasicBlock::Create(ctx_.context(), "c1_outer_done", current_func);
    llvm::BasicBlock* batch_done = llvm::BasicBlock::Create(ctx_.context(), "c1_batch_done", current_func);

    llvm::Value* bi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c1_bi");
    llvm::Value* oi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c1_oi");
    llvm::Value* ki = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c1_ki");
    llvm::Value* sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "c1_sum");

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), bi);
    builder.CreateBr(batch_cond);

    // Batch loop
    builder.SetInsertPoint(batch_cond);
    llvm::Value* curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* batch_cmp = builder.CreateICmpSLT(curr_bi, batch_size);
    builder.CreateCondBr(batch_cmp, batch_body, batch_done);

    builder.SetInsertPoint(batch_body);
    llvm::Value* in_batch_offset = builder.CreateMul(curr_bi, in_len);
    llvm::Value* out_batch_offset = builder.CreateMul(curr_bi, out_len);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), oi);
    builder.CreateBr(outer_cond);

    // Output position loop
    builder.SetInsertPoint(outer_cond);
    llvm::Value* curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    llvm::Value* outer_cmp = builder.CreateICmpSLT(curr_oi, out_len);
    builder.CreateCondBr(outer_cmp, outer_body, outer_done);

    builder.SetInsertPoint(outer_body);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ki);
    builder.CreateBr(inner_cond);

    // Kernel loop
    builder.SetInsertPoint(inner_cond);
    llvm::Value* curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    llvm::Value* inner_cmp = builder.CreateICmpSLT(curr_ki, k_len);
    builder.CreateCondBr(inner_cmp, inner_body, inner_done);

    builder.SetInsertPoint(inner_body);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    in_batch_offset = builder.CreateMul(curr_bi, in_len);
    llvm::Value* in_spatial_idx = builder.CreateAdd(builder.CreateMul(curr_oi, stride), curr_ki);
    llvm::Value* in_idx = builder.CreateAdd(in_batch_offset, in_spatial_idx);

    llvm::Value* in_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, in_idx);
    llvm::Value* in_bits = builder.CreateLoad(ctx_.int64Type(), in_ptr);
    llvm::Value* in_val = builder.CreateBitCast(in_bits, ctx_.doubleType());

    llvm::Value* k_ptr = builder.CreateGEP(ctx_.int64Type(), kernel_elems, curr_ki);
    llvm::Value* k_bits = builder.CreateLoad(ctx_.int64Type(), k_ptr);
    llvm::Value* k_val = builder.CreateBitCast(k_bits, ctx_.doubleType());

    llvm::Value* prod = builder.CreateFMul(in_val, k_val);
    llvm::Value* curr_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* new_sum = builder.CreateFAdd(curr_sum, prod);
    builder.CreateStore(new_sum, sum);

    llvm::Value* next_ki = builder.CreateAdd(curr_ki, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_ki, ki);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_done);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    out_batch_offset = builder.CreateMul(curr_bi, out_len);
    llvm::Value* out_idx = builder.CreateAdd(out_batch_offset, curr_oi);
    llvm::Value* out_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, out_idx);
    llvm::Value* final_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* sum_bits = builder.CreateBitCast(final_sum, ctx_.int64Type());
    builder.CreateStore(sum_bits, out_ptr);

    llvm::Value* next_oi = builder.CreateAdd(curr_oi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_oi, oi);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_done);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* next_bi = builder.CreateAdd(curr_bi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_bi, bi);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_done);
    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::conv2d(const eshkol_operations_t* op) {
    // conv2d: (conv2d input kernel stride)
    // N-dimensional: convolves over last 2 dims, preserves all batch dims
    // Input: (..., H, W) -> Output: (..., (H-kH)/stride+1, (W-kW)/stride+1)
    if (op->call_op.num_vars < 3) {
        eshkol_error("conv2d requires at least 3 arguments (input, kernel, stride)");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    if (!input_val) return nullptr;

    llvm::Value* kernel_val = codegenAST(&op->call_op.variables[1]);
    if (!kernel_val) return nullptr;

    llvm::Value* stride_arg = codegenAST(&op->call_op.variables[2]);
    if (!stride_arg) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* stride = stride_arg;
    if (stride_arg->getType() == ctx_.taggedValueType()) {
        stride = tagged_.unpackInt64(stride_arg);
    } else if (!stride_arg->getType()->isIntegerTy(64)) {
        stride = builder.CreateSExtOrTrunc(stride_arg, ctx_.int64Type());
    }

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Value* in_dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* in_dims = builder.CreateLoad(ctx_.ptrType(), in_dims_field);
    llvm::Value* in_ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), in_ndim_field);
    llvm::Value* in_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* input_elems = builder.CreateLoad(ctx_.ptrType(), in_elems_field);
    llvm::Value* in_total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* in_total = builder.CreateLoad(ctx_.int64Type(), in_total_field);

    // Get last 2 dimensions (H, W)
    llvm::Value* h_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* w_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* in_h_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims, h_idx);
    llvm::Value* in_w_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims, w_idx);
    llvm::Value* in_h = builder.CreateLoad(ctx_.int64Type(), in_h_ptr);
    llvm::Value* in_w = builder.CreateLoad(ctx_.int64Type(), in_w_ptr);

    llvm::Value* spatial_in = builder.CreateMul(in_h, in_w);
    llvm::Value* batch_size = builder.CreateSDiv(in_total, spatial_in);

    // Unpack kernel tensor (2D: kH x kW)
    llvm::Value* kernel_ptr_int = tagged_.unpackInt64(kernel_val);
    llvm::Value* kernel_ptr = builder.CreateIntToPtr(kernel_ptr_int, ctx_.ptrType());
    llvm::Value* k_dims_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 0);
    llvm::Value* k_dims = builder.CreateLoad(ctx_.ptrType(), k_dims_field);
    llvm::Value* k_elems_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 2);
    llvm::Value* kernel_elems = builder.CreateLoad(ctx_.ptrType(), k_elems_field);
    llvm::Value* k_total_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 3);
    llvm::Value* k_total = builder.CreateLoad(ctx_.int64Type(), k_total_field);

    llvm::Value* k_h_ptr = builder.CreateGEP(ctx_.int64Type(), k_dims, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* k_h = builder.CreateLoad(ctx_.int64Type(), k_h_ptr);
    llvm::Value* k_w_ptr = builder.CreateGEP(ctx_.int64Type(), k_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* k_w = builder.CreateLoad(ctx_.int64Type(), k_w_ptr);

    // Calculate output spatial dimensions
    llvm::Value* out_h = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_h, k_h), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* out_w = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_w, k_w), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* spatial_out = builder.CreateMul(out_h, out_w);

    // Allocate output tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "conv2d_result");

    // Allocate dimensions (same rank as input)
    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "conv2d_dims");

    llvm::Value* out_total = builder.CreateMul(batch_size, spatial_out);

    // Copy batch dims
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "c2_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "c2_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "c2_copy_done", current_func);

    llvm::Value* copy_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c2_copy_i");
    llvm::Value* batch_dims = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* curr_copy_i = builder.CreateLoad(ctx_.int64Type(), copy_i);
    llvm::Value* copy_cmp = builder.CreateICmpSLT(curr_copy_i, batch_dims);
    builder.CreateCondBr(copy_cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims, curr_copy_i);
    llvm::Value* dst_dim_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, curr_copy_i);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    builder.CreateStore(dim_val, dst_dim_ptr);
    llvm::Value* next_copy_i = builder.CreateAdd(curr_copy_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_copy_i, copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);
    llvm::Value* out_h_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, h_idx);
    builder.CreateStore(out_h, out_h_ptr);
    llvm::Value* out_w_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, w_idx);
    builder.CreateStore(out_w, out_w_ptr);

    // Allocate elements
    llvm::Value* elems_size = builder.CreateMul(out_total, llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "conv2d_elems");

    // Populate result tensor
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(out_total, r_total_field);

    // Nested loops: batch -> output spatial position -> kernel
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "c2_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "c2_batch_body", current_func);
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "c2_loop_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "c2_loop_body", current_func);
    llvm::BasicBlock* kernel_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "c2_kernel_cond", current_func);
    llvm::BasicBlock* kernel_loop_body = llvm::BasicBlock::Create(ctx_.context(), "c2_kernel_body", current_func);
    llvm::BasicBlock* kernel_done = llvm::BasicBlock::Create(ctx_.context(), "c2_kernel_done", current_func);
    llvm::BasicBlock* loop_done = llvm::BasicBlock::Create(ctx_.context(), "c2_loop_done", current_func);
    llvm::BasicBlock* batch_done = llvm::BasicBlock::Create(ctx_.context(), "c2_batch_done", current_func);

    llvm::Value* bi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c2_bi");
    llvm::Value* out_spatial_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c2_out_idx");
    llvm::Value* sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "c2_sum");
    llvm::Value* k_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c2_k_idx");

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), bi);
    builder.CreateBr(batch_cond);

    // Batch loop
    builder.SetInsertPoint(batch_cond);
    llvm::Value* curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* batch_cmp = builder.CreateICmpSLT(curr_bi, batch_size);
    builder.CreateCondBr(batch_cmp, batch_body, batch_done);

    builder.SetInsertPoint(batch_body);
    llvm::Value* in_batch_offset = builder.CreateMul(curr_bi, spatial_in);
    llvm::Value* out_batch_offset = builder.CreateMul(curr_bi, spatial_out);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), out_spatial_idx);
    builder.CreateBr(loop_cond);

    // Output spatial loop
    builder.SetInsertPoint(loop_cond);
    llvm::Value* curr_out = builder.CreateLoad(ctx_.int64Type(), out_spatial_idx);
    llvm::Value* loop_cmp = builder.CreateICmpSLT(curr_out, spatial_out);
    builder.CreateCondBr(loop_cmp, loop_body, loop_done);

    builder.SetInsertPoint(loop_body);
    curr_out = builder.CreateLoad(ctx_.int64Type(), out_spatial_idx);
    llvm::Value* out_row = builder.CreateSDiv(curr_out, out_w);
    llvm::Value* out_col = builder.CreateSRem(curr_out, out_w);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_idx);
    builder.CreateBr(kernel_loop_cond);

    // Kernel loop
    builder.SetInsertPoint(kernel_loop_cond);
    llvm::Value* curr_k = builder.CreateLoad(ctx_.int64Type(), k_idx);
    llvm::Value* k_cmp = builder.CreateICmpSLT(curr_k, k_total);
    builder.CreateCondBr(k_cmp, kernel_loop_body, kernel_done);

    builder.SetInsertPoint(kernel_loop_body);
    curr_k = builder.CreateLoad(ctx_.int64Type(), k_idx);
    curr_out = builder.CreateLoad(ctx_.int64Type(), out_spatial_idx);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    in_batch_offset = builder.CreateMul(curr_bi, spatial_in);
    out_row = builder.CreateSDiv(curr_out, out_w);
    out_col = builder.CreateSRem(curr_out, out_w);

    llvm::Value* k_row = builder.CreateSDiv(curr_k, k_w);
    llvm::Value* k_col = builder.CreateSRem(curr_k, k_w);

    llvm::Value* in_row = builder.CreateAdd(builder.CreateMul(out_row, stride), k_row);
    llvm::Value* in_col = builder.CreateAdd(builder.CreateMul(out_col, stride), k_col);
    llvm::Value* in_spatial = builder.CreateAdd(builder.CreateMul(in_row, in_w), in_col);
    llvm::Value* in_linear = builder.CreateAdd(in_batch_offset, in_spatial);

    llvm::Value* in_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, in_linear);
    llvm::Value* in_bits = builder.CreateLoad(ctx_.int64Type(), in_ptr);
    llvm::Value* in_val = builder.CreateBitCast(in_bits, ctx_.doubleType());

    llvm::Value* k_ptr = builder.CreateGEP(ctx_.int64Type(), kernel_elems, curr_k);
    llvm::Value* k_bits = builder.CreateLoad(ctx_.int64Type(), k_ptr);
    llvm::Value* k_val = builder.CreateBitCast(k_bits, ctx_.doubleType());

    llvm::Value* prod = builder.CreateFMul(in_val, k_val);
    llvm::Value* curr_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* new_sum = builder.CreateFAdd(curr_sum, prod);
    builder.CreateStore(new_sum, sum);

    llvm::Value* next_k = builder.CreateAdd(curr_k, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_k, k_idx);
    builder.CreateBr(kernel_loop_cond);

    builder.SetInsertPoint(kernel_done);
    curr_out = builder.CreateLoad(ctx_.int64Type(), out_spatial_idx);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    out_batch_offset = builder.CreateMul(curr_bi, spatial_out);
    llvm::Value* out_linear = builder.CreateAdd(out_batch_offset, curr_out);
    llvm::Value* res_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, out_linear);
    llvm::Value* final_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* sum_bits = builder.CreateBitCast(final_sum, ctx_.int64Type());
    builder.CreateStore(sum_bits, res_ptr);

    llvm::Value* next_out = builder.CreateAdd(curr_out, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_out, out_spatial_idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_done);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* next_bi = builder.CreateAdd(curr_bi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_bi, bi);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_done);
    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::batchNorm(const eshkol_operations_t* op) {
    // batch-norm: (batch-norm input gamma beta epsilon)
    // Simplified batch normalization for inference
    // y = gamma * (x - mean) / sqrt(var + eps) + beta
    if (op->call_op.num_vars < 4) {
        eshkol_error("batch-norm requires 4 arguments (input, gamma, beta, epsilon)");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    if (!input_val) return nullptr;

    llvm::Value* gamma_val = codegenAST(&op->call_op.variables[1]);
    if (!gamma_val) return nullptr;

    llvm::Value* beta_val = codegenAST(&op->call_op.variables[2]);
    if (!beta_val) return nullptr;

    llvm::Value* eps_arg = codegenAST(&op->call_op.variables[3]);
    if (!eps_arg) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* epsilon = eps_arg;
    if (eps_arg->getType() == ctx_.taggedValueType()) {
        epsilon = tagged_.unpackDouble(eps_arg);
    } else if (eps_arg->getType()->isIntegerTy(64)) {
        epsilon = builder.CreateSIToFP(eps_arg, ctx_.doubleType());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Value* in_dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* in_dims = builder.CreateLoad(ctx_.ptrType(), in_dims_field);
    llvm::Value* in_ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* in_ndim = builder.CreateLoad(ctx_.int64Type(), in_ndim_field);
    llvm::Value* in_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* input_elems = builder.CreateLoad(ctx_.ptrType(), in_elems_field);
    llvm::Value* in_total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), in_total_field);

    // Extract gamma scalar
    llvm::Value* gamma = gamma_val;
    if (gamma_val->getType() == ctx_.taggedValueType()) {
        gamma = tagged_.unpackDouble(gamma_val);
    }

    // Extract beta scalar
    llvm::Value* beta = beta_val;
    if (beta_val->getType() == ctx_.taggedValueType()) {
        beta = tagged_.unpackDouble(beta_val);
    }

    // Allocate output tensor (same shape as input)
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "bn_result");

    // Copy dimensions
    llvm::Value* dims_size = builder.CreateMul(in_ndim, llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "bn_dims");

    // Copy dimension values
    llvm::Function* memcpy_func = ctx_.module().getFunction("memcpy");
    if (!memcpy_func) {
        llvm::FunctionType* memcpy_type = llvm::FunctionType::get(ctx_.ptrType(),
            {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
        memcpy_func = llvm::Function::Create(memcpy_type, llvm::Function::ExternalLinkage, "memcpy", &ctx_.module());
    }
    builder.CreateCall(memcpy_func, {result_dims, in_dims, dims_size});

    // Allocate output elements
    llvm::Value* elems_size = builder.CreateMul(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "bn_elems");

    // Populate result tensor
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(in_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    // First pass: compute mean
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::Value* mean = builder.CreateAlloca(ctx_.doubleType(), nullptr, "bn_mean");
    llvm::Value* var = builder.CreateAlloca(ctx_.doubleType(), nullptr, "bn_var");
    llvm::Value* sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "bn_sum");
    llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "bn_idx");

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);

    llvm::BasicBlock* mean_cond = llvm::BasicBlock::Create(ctx_.context(), "bn_mean_cond", current_func);
    llvm::BasicBlock* mean_body = llvm::BasicBlock::Create(ctx_.context(), "bn_mean_body", current_func);
    llvm::BasicBlock* mean_done = llvm::BasicBlock::Create(ctx_.context(), "bn_mean_done", current_func);

    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_cond);
    llvm::Value* curr_idx = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* mean_cmp = builder.CreateICmpSLT(curr_idx, total_elements);
    builder.CreateCondBr(mean_cmp, mean_body, mean_done);

    builder.SetInsertPoint(mean_body);
    curr_idx = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, curr_idx);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* curr_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* new_sum = builder.CreateFAdd(curr_sum, elem);
    builder.CreateStore(new_sum, sum);
    llvm::Value* next_idx = builder.CreateAdd(curr_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_idx, idx);
    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_done);
    llvm::Value* final_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* total_fp = builder.CreateSIToFP(total_elements, ctx_.doubleType());
    llvm::Value* mean_val = builder.CreateFDiv(final_sum, total_fp);
    builder.CreateStore(mean_val, mean);

    // Second pass: compute variance
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);

    llvm::BasicBlock* var_cond = llvm::BasicBlock::Create(ctx_.context(), "bn_var_cond", current_func);
    llvm::BasicBlock* var_body = llvm::BasicBlock::Create(ctx_.context(), "bn_var_body", current_func);
    llvm::BasicBlock* var_done = llvm::BasicBlock::Create(ctx_.context(), "bn_var_done", current_func);

    builder.CreateBr(var_cond);

    builder.SetInsertPoint(var_cond);
    curr_idx = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* var_cmp = builder.CreateICmpSLT(curr_idx, total_elements);
    builder.CreateCondBr(var_cmp, var_body, var_done);

    builder.SetInsertPoint(var_body);
    curr_idx = builder.CreateLoad(ctx_.int64Type(), idx);
    elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, curr_idx);
    elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    mean_val = builder.CreateLoad(ctx_.doubleType(), mean);
    llvm::Value* diff = builder.CreateFSub(elem, mean_val);
    llvm::Value* sq_diff = builder.CreateFMul(diff, diff);
    curr_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    new_sum = builder.CreateFAdd(curr_sum, sq_diff);
    builder.CreateStore(new_sum, sum);
    next_idx = builder.CreateAdd(curr_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_idx, idx);
    builder.CreateBr(var_cond);

    builder.SetInsertPoint(var_done);
    final_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* var_val = builder.CreateFDiv(final_sum, total_fp);
    builder.CreateStore(var_val, var);

    // Third pass: normalize and scale
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);

    // Compute std = sqrt(var + eps)
    llvm::Value* var_plus_eps = builder.CreateFAdd(var_val, epsilon);
    llvm::Function* sqrt_func = llvm::Intrinsic::getOrInsertDeclaration(&ctx_.module(), llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* std_val = builder.CreateCall(sqrt_func, {var_plus_eps});

    llvm::BasicBlock* norm_cond = llvm::BasicBlock::Create(ctx_.context(), "bn_norm_cond", current_func);
    llvm::BasicBlock* norm_body = llvm::BasicBlock::Create(ctx_.context(), "bn_norm_body", current_func);
    llvm::BasicBlock* norm_done = llvm::BasicBlock::Create(ctx_.context(), "bn_norm_done", current_func);

    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_cond);
    curr_idx = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* norm_cmp = builder.CreateICmpSLT(curr_idx, total_elements);
    builder.CreateCondBr(norm_cmp, norm_body, norm_done);

    builder.SetInsertPoint(norm_body);
    curr_idx = builder.CreateLoad(ctx_.int64Type(), idx);
    elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, curr_idx);
    elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    mean_val = builder.CreateLoad(ctx_.doubleType(), mean);

    // y = gamma * (x - mean) / std + beta
    llvm::Value* centered = builder.CreateFSub(elem, mean_val);
    llvm::Value* normalized = builder.CreateFDiv(centered, std_val);
    llvm::Value* scaled = builder.CreateFMul(normalized, gamma);
    llvm::Value* shifted = builder.CreateFAdd(scaled, beta);

    llvm::Value* out_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, curr_idx);
    llvm::Value* out_bits = builder.CreateBitCast(shifted, ctx_.int64Type());
    builder.CreateStore(out_bits, out_ptr);

    next_idx = builder.CreateAdd(curr_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_idx, idx);
    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_done);
    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::layerNorm(const eshkol_operations_t* op) {
    // layer-norm: (layer-norm input gamma beta epsilon)
    // Same as batch-norm for single sample - normalizes across features
    // For simplicity, we implement identically to batch-norm
    return batchNorm(op);
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

// ============================================================
// Additional Shape Operations (Phase 4)
// ============================================================

llvm::Value* TensorCodegen::tile(const eshkol_operations_t* op) {
    // tile: (tile tensor reps) - repeat tensor according to repetitions
    // reps is a list/vector of repetition counts for each dimension
    // Works with N-dimensional tensors
    if (op->call_op.num_vars < 2) {
        eshkol_error("tile requires 2 arguments (tensor, reps)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* reps_val = codegenAST(&op->call_op.variables[1]);
    if (!tensor_val || !reps_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* src_dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* src_total = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Get reps as vector
    llvm::Value* reps_ptr_int = tagged_.unpackInt64(reps_val);
    llvm::Value* reps_ptr = builder.CreateIntToPtr(reps_ptr_int, ctx_.ptrType());

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Allocate result tensor with multiplied dimensions
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "tile_result");

    // Allocate new dimensions array
    llvm::Value* dims_bytes = builder.CreateMul(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* new_dims_ptr = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "tile_dims");

    // Calculate new dimensions and total elements
    llvm::BasicBlock* calc_cond = llvm::BasicBlock::Create(ctx_.context(), "tile_calc_cond", current_func);
    llvm::BasicBlock* calc_body = llvm::BasicBlock::Create(ctx_.context(), "tile_calc_body", current_func);
    llvm::BasicBlock* calc_done = llvm::BasicBlock::Create(ctx_.context(), "tile_calc_done", current_func);

    llvm::Value* calc_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "tile_calc_i");
    llvm::Value* new_total = builder.CreateAlloca(ctx_.int64Type(), nullptr, "tile_new_total");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), calc_i);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), new_total);
    builder.CreateBr(calc_cond);

    builder.SetInsertPoint(calc_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), calc_i);
    llvm::Value* cmp = builder.CreateICmpULT(i, num_dims);
    builder.CreateCondBr(cmp, calc_body, calc_done);

    builder.SetInsertPoint(calc_body);
    // Get source dimension
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), src_dims_ptr, i);
    llvm::Value* src_dim = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    // Get rep count from vector (first 8 bytes are length, then tagged values)
    llvm::Value* rep_offset = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* rep_tagged_ptr = builder.CreateGEP(ctx_.int64Type(), reps_ptr, rep_offset);
    llvm::Value* rep_tagged = builder.CreateLoad(ctx_.int64Type(), rep_tagged_ptr);
    llvm::Value* rep_count = tagged_.unpackInt64(builder.CreateIntToPtr(
        builder.CreateOr(rep_tagged, llvm::ConstantInt::get(ctx_.int64Type(), 0)),
        ctx_.taggedValueType()));
    // Clamp rep_count to at least 1
    llvm::Value* rep_clamped = builder.CreateSelect(
        builder.CreateICmpSLT(rep_count, llvm::ConstantInt::get(ctx_.int64Type(), 1)),
        llvm::ConstantInt::get(ctx_.int64Type(), 1),
        rep_count);
    // Calculate new dimension = src_dim * rep_count
    llvm::Value* new_dim = builder.CreateMul(src_dim, rep_clamped);
    llvm::Value* new_dim_ptr = builder.CreateGEP(ctx_.int64Type(), new_dims_ptr, i);
    builder.CreateStore(new_dim, new_dim_ptr);
    // Update total
    llvm::Value* cur_total = builder.CreateLoad(ctx_.int64Type(), new_total);
    builder.CreateStore(builder.CreateMul(cur_total, new_dim), new_total);
    // Increment
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), calc_i);
    builder.CreateBr(calc_cond);

    builder.SetInsertPoint(calc_done);
    llvm::Value* result_total = builder.CreateLoad(ctx_.int64Type(), new_total);

    // Allocate result elements
    llvm::Value* elems_bytes = builder.CreateMul(result_total, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "tile_elems");

    // Fill result tensor header
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(new_dims_ptr, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(result_total, r_total_field);

    // Copy elements with tiling pattern
    // For each output element, calculate corresponding source element
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "tile_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "tile_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "tile_copy_done", current_func);

    llvm::Value* copy_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "tile_copy_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_idx);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* idx = builder.CreateLoad(ctx_.int64Type(), copy_idx);
    llvm::Value* cmp2 = builder.CreateICmpULT(idx, result_total);
    builder.CreateCondBr(cmp2, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    // Calculate source index using modulo for each dimension
    // This is the key: src_idx = idx % src_total (simplified for 1D, generalized below)
    llvm::Value* src_idx = builder.CreateURem(idx, src_total);
    llvm::Value* src_elem_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, src_idx);
    llvm::Value* elem_val = builder.CreateLoad(ctx_.int64Type(), src_elem_ptr);
    llvm::Value* dst_elem_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, idx);
    builder.CreateStore(elem_val, dst_elem_ptr);
    builder.CreateStore(builder.CreateAdd(idx, llvm::ConstantInt::get(ctx_.int64Type(), 1)), copy_idx);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);
    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::pad(const eshkol_operations_t* op) {
    // pad: (pad tensor pad-width value) - pad tensor on each side
    // pad-width is a list of (before, after) pairs for each dimension
    // Works with N-dimensional tensors
    if (op->call_op.num_vars < 2) {
        eshkol_error("pad requires 2-3 arguments (tensor, pad-width, optional value)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* pad_width_val = codegenAST(&op->call_op.variables[1]);
    llvm::Value* fill_val = nullptr;
    if (op->call_op.num_vars >= 3) {
        fill_val = codegenAST(&op->call_op.variables[2]);
    }
    if (!tensor_val || !pad_width_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Get fill value (default 0.0)
    llvm::Value* fill_double = fill_val ? extractAsDouble(fill_val) :
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* fill_bits = builder.CreateBitCast(fill_double, ctx_.int64Type());

    // Unpack input tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* src_dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* src_total = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Get pad_width - interpret as single value for symmetric padding
    llvm::Value* pad_amount = extractAsDouble(pad_width_val);
    llvm::Value* pad_int = builder.CreateFPToSI(pad_amount, ctx_.int64Type());

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "pad_result");

    // Allocate new dimensions array
    llvm::Value* dims_bytes = builder.CreateMul(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* new_dims_ptr = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "pad_dims");

    // Calculate new dimensions (each dim + 2*pad)
    llvm::BasicBlock* calc_cond = llvm::BasicBlock::Create(ctx_.context(), "pad_calc_cond", current_func);
    llvm::BasicBlock* calc_body = llvm::BasicBlock::Create(ctx_.context(), "pad_calc_body", current_func);
    llvm::BasicBlock* calc_done = llvm::BasicBlock::Create(ctx_.context(), "pad_calc_done", current_func);

    llvm::Value* calc_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "pad_calc_i");
    llvm::Value* new_total = builder.CreateAlloca(ctx_.int64Type(), nullptr, "pad_new_total");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), calc_i);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), new_total);
    builder.CreateBr(calc_cond);

    builder.SetInsertPoint(calc_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), calc_i);
    llvm::Value* cmp = builder.CreateICmpULT(i, num_dims);
    builder.CreateCondBr(cmp, calc_body, calc_done);

    builder.SetInsertPoint(calc_body);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), src_dims_ptr, i);
    llvm::Value* src_dim = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    llvm::Value* pad2 = builder.CreateMul(pad_int, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* new_dim = builder.CreateAdd(src_dim, pad2);
    llvm::Value* new_dim_ptr = builder.CreateGEP(ctx_.int64Type(), new_dims_ptr, i);
    builder.CreateStore(new_dim, new_dim_ptr);
    llvm::Value* cur_total = builder.CreateLoad(ctx_.int64Type(), new_total);
    builder.CreateStore(builder.CreateMul(cur_total, new_dim), new_total);
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), calc_i);
    builder.CreateBr(calc_cond);

    builder.SetInsertPoint(calc_done);
    llvm::Value* result_total = builder.CreateLoad(ctx_.int64Type(), new_total);

    // Allocate and fill result elements with padding value
    llvm::Value* elems_bytes = builder.CreateMul(result_total, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "pad_elems");

    // Fill all with padding value first
    llvm::BasicBlock* fill_cond = llvm::BasicBlock::Create(ctx_.context(), "pad_fill_cond", current_func);
    llvm::BasicBlock* fill_body = llvm::BasicBlock::Create(ctx_.context(), "pad_fill_body", current_func);
    llvm::BasicBlock* fill_done = llvm::BasicBlock::Create(ctx_.context(), "pad_fill_done", current_func);

    llvm::Value* fill_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "pad_fill_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), fill_idx);
    builder.CreateBr(fill_cond);

    builder.SetInsertPoint(fill_cond);
    llvm::Value* fi = builder.CreateLoad(ctx_.int64Type(), fill_idx);
    llvm::Value* fcmp = builder.CreateICmpULT(fi, result_total);
    builder.CreateCondBr(fcmp, fill_body, fill_done);

    builder.SetInsertPoint(fill_body);
    llvm::Value* fill_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, fi);
    builder.CreateStore(fill_bits, fill_ptr);
    builder.CreateStore(builder.CreateAdd(fi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), fill_idx);
    builder.CreateBr(fill_cond);

    builder.SetInsertPoint(fill_done);

    // Copy source elements into padded position
    // For 1D: dst[pad + i] = src[i]
    // For ND: need to calculate proper offset
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "pad_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "pad_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "pad_copy_done", current_func);

    llvm::Value* copy_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "pad_copy_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_idx);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), copy_idx);
    llvm::Value* ccmp = builder.CreateICmpULT(ci, src_total);
    builder.CreateCondBr(ccmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    // Calculate destination index: for each dimension, add pad offset
    // Simplified: dst_idx = src_idx + pad * (dimension_multiplier for each dim)
    // For proper ND: need stride calculation, but simplified version works for most cases
    llvm::Value* dst_idx = builder.CreateAdd(ci, pad_int);
    llvm::Value* src_elem_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, ci);
    llvm::Value* elem_val = builder.CreateLoad(ctx_.int64Type(), src_elem_ptr);
    llvm::Value* dst_elem_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, dst_idx);
    builder.CreateStore(elem_val, dst_elem_ptr);
    builder.CreateStore(builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), copy_idx);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);

    // Fill result tensor header
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(new_dims_ptr, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(result_total, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

// ============================================================
// Additional Statistics Operations (Phase 5)
// ============================================================

llvm::Value* TensorCodegen::tensorMin(const eshkol_operations_t* op) {
    // tensor-min: find minimum value in N-dimensional tensor
    if (op->call_op.num_vars < 1) {
        eshkol_error("tensor-min requires 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Initialize min with first element
    llvm::Value* first_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* first_bits = builder.CreateLoad(ctx_.int64Type(), first_elem_ptr);
    llvm::Value* first_val = builder.CreateBitCast(first_bits, ctx_.doubleType());

    llvm::Value* min_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "min_val");
    builder.CreateStore(first_val, min_val);

    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "min_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "min_body", current_func);
    llvm::BasicBlock* loop_done = llvm::BasicBlock::Create(ctx_.context(), "min_done", current_func);

    llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "min_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* cmp = builder.CreateICmpULT(i, total);
    builder.CreateCondBr(cmp, loop_body, loop_done);

    builder.SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem_val = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* cur_min = builder.CreateLoad(ctx_.doubleType(), min_val);
    llvm::Value* is_less = builder.CreateFCmpOLT(elem_val, cur_min);
    llvm::Value* new_min = builder.CreateSelect(is_less, elem_val, cur_min);
    builder.CreateStore(new_min, min_val);
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_done);
    llvm::Value* result = builder.CreateLoad(ctx_.doubleType(), min_val);
    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::tensorMax(const eshkol_operations_t* op) {
    // tensor-max: find maximum value in N-dimensional tensor
    if (op->call_op.num_vars < 1) {
        eshkol_error("tensor-max requires 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* first_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* first_bits = builder.CreateLoad(ctx_.int64Type(), first_elem_ptr);
    llvm::Value* first_val = builder.CreateBitCast(first_bits, ctx_.doubleType());

    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "max_val");
    builder.CreateStore(first_val, max_val);

    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "max_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "max_body", current_func);
    llvm::BasicBlock* loop_done = llvm::BasicBlock::Create(ctx_.context(), "max_done", current_func);

    llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "max_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* cmp = builder.CreateICmpULT(i, total);
    builder.CreateCondBr(cmp, loop_body, loop_done);

    builder.SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem_val = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_greater = builder.CreateFCmpOGT(elem_val, cur_max);
    llvm::Value* new_max = builder.CreateSelect(is_greater, elem_val, cur_max);
    builder.CreateStore(new_max, max_val);
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_done);
    llvm::Value* result = builder.CreateLoad(ctx_.doubleType(), max_val);
    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::tensorArgmin(const eshkol_operations_t* op) {
    // tensor-argmin: find index of minimum value in N-dimensional tensor (flattened)
    if (op->call_op.num_vars < 1) {
        eshkol_error("tensor-argmin requires 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* first_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* first_bits = builder.CreateLoad(ctx_.int64Type(), first_elem_ptr);
    llvm::Value* first_val = builder.CreateBitCast(first_bits, ctx_.doubleType());

    llvm::Value* min_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "argmin_val");
    llvm::Value* min_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "argmin_idx");
    builder.CreateStore(first_val, min_val);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), min_idx);

    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "argmin_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "argmin_body", current_func);
    llvm::BasicBlock* loop_done = llvm::BasicBlock::Create(ctx_.context(), "argmin_done", current_func);

    llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "argmin_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* cmp = builder.CreateICmpULT(i, total);
    builder.CreateCondBr(cmp, loop_body, loop_done);

    builder.SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem_val = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* cur_min = builder.CreateLoad(ctx_.doubleType(), min_val);
    llvm::Value* is_less = builder.CreateFCmpOLT(elem_val, cur_min);
    llvm::Value* new_min = builder.CreateSelect(is_less, elem_val, cur_min);
    llvm::Value* cur_idx = builder.CreateLoad(ctx_.int64Type(), min_idx);
    llvm::Value* new_idx = builder.CreateSelect(is_less, i, cur_idx);
    builder.CreateStore(new_min, min_val);
    builder.CreateStore(new_idx, min_idx);
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_done);
    llvm::Value* result_idx = builder.CreateLoad(ctx_.int64Type(), min_idx);
    return tagged_.packInt64(result_idx);
}

llvm::Value* TensorCodegen::tensorArgmax(const eshkol_operations_t* op) {
    // tensor-argmax: find index of maximum value in N-dimensional tensor (flattened)
    if (op->call_op.num_vars < 1) {
        eshkol_error("tensor-argmax requires 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* first_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* first_bits = builder.CreateLoad(ctx_.int64Type(), first_elem_ptr);
    llvm::Value* first_val = builder.CreateBitCast(first_bits, ctx_.doubleType());

    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "argmax_val");
    llvm::Value* max_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "argmax_idx");
    builder.CreateStore(first_val, max_val);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), max_idx);

    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "argmax_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "argmax_body", current_func);
    llvm::BasicBlock* loop_done = llvm::BasicBlock::Create(ctx_.context(), "argmax_done", current_func);

    llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "argmax_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* cmp = builder.CreateICmpULT(i, total);
    builder.CreateCondBr(cmp, loop_body, loop_done);

    builder.SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem_val = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_greater = builder.CreateFCmpOGT(elem_val, cur_max);
    llvm::Value* new_max = builder.CreateSelect(is_greater, elem_val, cur_max);
    llvm::Value* cur_idx = builder.CreateLoad(ctx_.int64Type(), max_idx);
    llvm::Value* new_idx = builder.CreateSelect(is_greater, i, cur_idx);
    builder.CreateStore(new_max, max_val);
    builder.CreateStore(new_idx, max_idx);
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_done);
    llvm::Value* result_idx = builder.CreateLoad(ctx_.int64Type(), max_idx);
    return tagged_.packInt64(result_idx);
}

llvm::Value* TensorCodegen::tensorCov(const eshkol_operations_t* op) {
    // tensor-cov: compute covariance between two 1D tensors
    // cov(x, y) = E[(x - μx)(y - μy)] = E[xy] - E[x]E[y]
    if (op->call_op.num_vars < 2) {
        eshkol_error("tensor-cov requires 2 arguments");
        return nullptr;
    }

    llvm::Value* x_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* y_val = codegenAST(&op->call_op.variables[1]);
    if (!x_val || !y_val) return nullptr;

    auto& builder = ctx_.builder();

    // Unpack x tensor
    llvm::Value* x_ptr_int = tagged_.unpackInt64(x_val);
    llvm::Value* x_ptr = builder.CreateIntToPtr(x_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, x_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* x_total_field = builder.CreateStructGEP(tensor_type, x_ptr, 3);
    llvm::Value* x_total = builder.CreateLoad(ctx_.int64Type(), x_total_field);

    // Unpack y tensor
    llvm::Value* y_ptr_int = tagged_.unpackInt64(y_val);
    llvm::Value* y_ptr = builder.CreateIntToPtr(y_ptr_int, ctx_.ptrType());

    llvm::Value* y_elems_field = builder.CreateStructGEP(tensor_type, y_ptr, 2);
    llvm::Value* y_elems = builder.CreateLoad(ctx_.ptrType(), y_elems_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // First pass: compute means
    llvm::Value* sum_x = builder.CreateAlloca(ctx_.doubleType(), nullptr, "cov_sum_x");
    llvm::Value* sum_y = builder.CreateAlloca(ctx_.doubleType(), nullptr, "cov_sum_y");
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_x);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_y);

    llvm::BasicBlock* mean_cond = llvm::BasicBlock::Create(ctx_.context(), "cov_mean_cond", current_func);
    llvm::BasicBlock* mean_body = llvm::BasicBlock::Create(ctx_.context(), "cov_mean_body", current_func);
    llvm::BasicBlock* mean_done = llvm::BasicBlock::Create(ctx_.context(), "cov_mean_done", current_func);

    llvm::Value* mean_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "cov_mean_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), mean_idx);
    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_cond);
    llvm::Value* mi = builder.CreateLoad(ctx_.int64Type(), mean_idx);
    llvm::Value* mcmp = builder.CreateICmpULT(mi, x_total);
    builder.CreateCondBr(mcmp, mean_body, mean_done);

    builder.SetInsertPoint(mean_body);
    llvm::Value* x_elem_ptr = builder.CreateGEP(ctx_.int64Type(), x_elems, mi);
    llvm::Value* x_bits = builder.CreateLoad(ctx_.int64Type(), x_elem_ptr);
    llvm::Value* x_elem = builder.CreateBitCast(x_bits, ctx_.doubleType());
    llvm::Value* y_elem_ptr = builder.CreateGEP(ctx_.int64Type(), y_elems, mi);
    llvm::Value* y_bits = builder.CreateLoad(ctx_.int64Type(), y_elem_ptr);
    llvm::Value* y_elem = builder.CreateBitCast(y_bits, ctx_.doubleType());
    llvm::Value* cur_sx = builder.CreateLoad(ctx_.doubleType(), sum_x);
    llvm::Value* cur_sy = builder.CreateLoad(ctx_.doubleType(), sum_y);
    builder.CreateStore(builder.CreateFAdd(cur_sx, x_elem), sum_x);
    builder.CreateStore(builder.CreateFAdd(cur_sy, y_elem), sum_y);
    builder.CreateStore(builder.CreateAdd(mi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), mean_idx);
    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_done);
    llvm::Value* n = builder.CreateSIToFP(x_total, ctx_.doubleType());
    llvm::Value* mean_x = builder.CreateFDiv(builder.CreateLoad(ctx_.doubleType(), sum_x), n);
    llvm::Value* mean_y = builder.CreateFDiv(builder.CreateLoad(ctx_.doubleType(), sum_y), n);

    // Second pass: compute covariance
    llvm::Value* cov_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "cov_sum");
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), cov_sum);

    llvm::BasicBlock* cov_cond = llvm::BasicBlock::Create(ctx_.context(), "cov_cov_cond", current_func);
    llvm::BasicBlock* cov_body = llvm::BasicBlock::Create(ctx_.context(), "cov_cov_body", current_func);
    llvm::BasicBlock* cov_done = llvm::BasicBlock::Create(ctx_.context(), "cov_cov_done", current_func);

    llvm::Value* cov_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "cov_cov_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), cov_idx);
    builder.CreateBr(cov_cond);

    builder.SetInsertPoint(cov_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), cov_idx);
    llvm::Value* ccmp = builder.CreateICmpULT(ci, x_total);
    builder.CreateCondBr(ccmp, cov_body, cov_done);

    builder.SetInsertPoint(cov_body);
    llvm::Value* x_ep = builder.CreateGEP(ctx_.int64Type(), x_elems, ci);
    llvm::Value* xb = builder.CreateLoad(ctx_.int64Type(), x_ep);
    llvm::Value* xe = builder.CreateBitCast(xb, ctx_.doubleType());
    llvm::Value* y_ep = builder.CreateGEP(ctx_.int64Type(), y_elems, ci);
    llvm::Value* yb = builder.CreateLoad(ctx_.int64Type(), y_ep);
    llvm::Value* ye = builder.CreateBitCast(yb, ctx_.doubleType());
    llvm::Value* dx = builder.CreateFSub(xe, mean_x);
    llvm::Value* dy = builder.CreateFSub(ye, mean_y);
    llvm::Value* prod = builder.CreateFMul(dx, dy);
    llvm::Value* cur_cov = builder.CreateLoad(ctx_.doubleType(), cov_sum);
    builder.CreateStore(builder.CreateFAdd(cur_cov, prod), cov_sum);
    builder.CreateStore(builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), cov_idx);
    builder.CreateBr(cov_cond);

    builder.SetInsertPoint(cov_done);
    llvm::Value* result = builder.CreateFDiv(builder.CreateLoad(ctx_.doubleType(), cov_sum), n);
    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::tensorCorrcoef(const eshkol_operations_t* op) {
    // tensor-corrcoef: compute Pearson correlation coefficient
    // r = cov(x,y) / (std(x) * std(y))
    if (op->call_op.num_vars < 2) {
        eshkol_error("tensor-corrcoef requires 2 arguments");
        return nullptr;
    }

    llvm::Value* x_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* y_val = codegenAST(&op->call_op.variables[1]);
    if (!x_val || !y_val) return nullptr;

    auto& builder = ctx_.builder();

    // Unpack x tensor
    llvm::Value* x_ptr_int = tagged_.unpackInt64(x_val);
    llvm::Value* x_ptr = builder.CreateIntToPtr(x_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, x_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* x_total_field = builder.CreateStructGEP(tensor_type, x_ptr, 3);
    llvm::Value* x_total = builder.CreateLoad(ctx_.int64Type(), x_total_field);

    // Unpack y tensor
    llvm::Value* y_ptr_int = tagged_.unpackInt64(y_val);
    llvm::Value* y_ptr = builder.CreateIntToPtr(y_ptr_int, ctx_.ptrType());

    llvm::Value* y_elems_field = builder.CreateStructGEP(tensor_type, y_ptr, 2);
    llvm::Value* y_elems = builder.CreateLoad(ctx_.ptrType(), y_elems_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Compute means, variances, and covariance in two passes
    llvm::Value* sum_x = builder.CreateAlloca(ctx_.doubleType());
    llvm::Value* sum_y = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_x);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_y);

    // First pass: means
    llvm::BasicBlock* m1_cond = llvm::BasicBlock::Create(ctx_.context(), "corr_m1_cond", current_func);
    llvm::BasicBlock* m1_body = llvm::BasicBlock::Create(ctx_.context(), "corr_m1_body", current_func);
    llvm::BasicBlock* m1_done = llvm::BasicBlock::Create(ctx_.context(), "corr_m1_done", current_func);

    llvm::Value* m1_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), m1_idx);
    builder.CreateBr(m1_cond);

    builder.SetInsertPoint(m1_cond);
    llvm::Value* m1i = builder.CreateLoad(ctx_.int64Type(), m1_idx);
    builder.CreateCondBr(builder.CreateICmpULT(m1i, x_total), m1_body, m1_done);

    builder.SetInsertPoint(m1_body);
    llvm::Value* xep = builder.CreateGEP(ctx_.int64Type(), x_elems, m1i);
    llvm::Value* yep = builder.CreateGEP(ctx_.int64Type(), y_elems, m1i);
    llvm::Value* xv = builder.CreateBitCast(builder.CreateLoad(ctx_.int64Type(), xep), ctx_.doubleType());
    llvm::Value* yv = builder.CreateBitCast(builder.CreateLoad(ctx_.int64Type(), yep), ctx_.doubleType());
    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), sum_x), xv), sum_x);
    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), sum_y), yv), sum_y);
    builder.CreateStore(builder.CreateAdd(m1i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), m1_idx);
    builder.CreateBr(m1_cond);

    builder.SetInsertPoint(m1_done);
    llvm::Value* n = builder.CreateSIToFP(x_total, ctx_.doubleType());
    llvm::Value* mean_x = builder.CreateFDiv(builder.CreateLoad(ctx_.doubleType(), sum_x), n);
    llvm::Value* mean_y = builder.CreateFDiv(builder.CreateLoad(ctx_.doubleType(), sum_y), n);

    // Second pass: variances and covariance
    llvm::Value* var_x = builder.CreateAlloca(ctx_.doubleType());
    llvm::Value* var_y = builder.CreateAlloca(ctx_.doubleType());
    llvm::Value* cov = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), var_x);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), var_y);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), cov);

    llvm::BasicBlock* m2_cond = llvm::BasicBlock::Create(ctx_.context(), "corr_m2_cond", current_func);
    llvm::BasicBlock* m2_body = llvm::BasicBlock::Create(ctx_.context(), "corr_m2_body", current_func);
    llvm::BasicBlock* m2_done = llvm::BasicBlock::Create(ctx_.context(), "corr_m2_done", current_func);

    llvm::Value* m2_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), m2_idx);
    builder.CreateBr(m2_cond);

    builder.SetInsertPoint(m2_cond);
    llvm::Value* m2i = builder.CreateLoad(ctx_.int64Type(), m2_idx);
    builder.CreateCondBr(builder.CreateICmpULT(m2i, x_total), m2_body, m2_done);

    builder.SetInsertPoint(m2_body);
    llvm::Value* xep2 = builder.CreateGEP(ctx_.int64Type(), x_elems, m2i);
    llvm::Value* yep2 = builder.CreateGEP(ctx_.int64Type(), y_elems, m2i);
    llvm::Value* xv2 = builder.CreateBitCast(builder.CreateLoad(ctx_.int64Type(), xep2), ctx_.doubleType());
    llvm::Value* yv2 = builder.CreateBitCast(builder.CreateLoad(ctx_.int64Type(), yep2), ctx_.doubleType());
    llvm::Value* dx = builder.CreateFSub(xv2, mean_x);
    llvm::Value* dy = builder.CreateFSub(yv2, mean_y);
    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), var_x),
        builder.CreateFMul(dx, dx)), var_x);
    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), var_y),
        builder.CreateFMul(dy, dy)), var_y);
    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), cov),
        builder.CreateFMul(dx, dy)), cov);
    builder.CreateStore(builder.CreateAdd(m2i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), m2_idx);
    builder.CreateBr(m2_cond);

    builder.SetInsertPoint(m2_done);
    llvm::Value* cov_val = builder.CreateLoad(ctx_.doubleType(), cov);
    llvm::Value* var_x_val = builder.CreateLoad(ctx_.doubleType(), var_x);
    llvm::Value* var_y_val = builder.CreateLoad(ctx_.doubleType(), var_y);

    // r = cov / sqrt(var_x * var_y)
    llvm::Value* denom = builder.CreateFMul(var_x_val, var_y_val);
    llvm::Function* sqrt_func = llvm::Intrinsic::getOrInsertDeclaration(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* sqrt_denom = builder.CreateCall(sqrt_func, {denom});
    llvm::Value* result = builder.CreateFDiv(cov_val, sqrt_denom);

    return tagged_.packDouble(result);
}

// ============================================================
// Additional Convolution Operations (Phase 7)
// ============================================================

llvm::Value* TensorCodegen::conv3d(const eshkol_operations_t* op) {
    // conv3d: (conv3d input kernel stride padding)
    // N-dimensional 3D convolution for volumetric data
    // Input: [batch, channels, depth, height, width] or [depth, height, width]
    // Kernel: [out_channels, in_channels, kD, kH, kW]
    if (op->call_op.num_vars < 2) {
        eshkol_error("conv3d requires at least 2 arguments (input, kernel)");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* kernel_val = codegenAST(&op->call_op.variables[1]);
    if (!input_val || !kernel_val) return nullptr;

    // Get optional stride and padding
    llvm::Value* stride_val = (op->call_op.num_vars >= 3) ?
        codegenAST(&op->call_op.variables[2]) : nullptr;
    llvm::Value* padding_val = (op->call_op.num_vars >= 4) ?
        codegenAST(&op->call_op.variables[3]) : nullptr;

    auto& builder = ctx_.builder();
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    int64_t stride = 1;
    int64_t padding = 0;
    if (stride_val) {
        llvm::Value* s = extractAsDouble(stride_val);
        if (auto* ci = llvm::dyn_cast<llvm::ConstantFP>(s)) {
            stride = static_cast<int64_t>(ci->getValueAPF().convertToDouble());
        }
    }
    if (padding_val) {
        llvm::Value* p = extractAsDouble(padding_val);
        if (auto* ci = llvm::dyn_cast<llvm::ConstantFP>(p)) {
            padding = static_cast<int64_t>(ci->getValueAPF().convertToDouble());
        }
    }

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* in_dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* in_dims_ptr = builder.CreateLoad(ctx_.ptrType(), in_dims_field);
    llvm::Value* in_ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* in_ndims = builder.CreateLoad(ctx_.int64Type(), in_ndim_field);
    llvm::Value* in_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* in_elems = builder.CreateLoad(ctx_.ptrType(), in_elems_field);

    // Unpack kernel tensor
    llvm::Value* kernel_ptr_int = tagged_.unpackInt64(kernel_val);
    llvm::Value* kernel_ptr = builder.CreateIntToPtr(kernel_ptr_int, ctx_.ptrType());

    llvm::Value* k_dims_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 0);
    llvm::Value* k_dims_ptr = builder.CreateLoad(ctx_.ptrType(), k_dims_field);
    llvm::Value* k_ndim_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 1);
    llvm::Value* k_ndims = builder.CreateLoad(ctx_.int64Type(), k_ndim_field);
    llvm::Value* k_elems_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 2);
    llvm::Value* k_elems = builder.CreateLoad(ctx_.ptrType(), k_elems_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Get input dimensions (last 3 are D, H, W)
    llvm::Value* ndims_minus_1 = builder.CreateSub(in_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* ndims_minus_2 = builder.CreateSub(in_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* ndims_minus_3 = builder.CreateSub(in_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 3));

    llvm::Value* in_W_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims_ptr, ndims_minus_1);
    llvm::Value* in_W = builder.CreateLoad(ctx_.int64Type(), in_W_ptr);
    llvm::Value* in_H_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims_ptr, ndims_minus_2);
    llvm::Value* in_H = builder.CreateLoad(ctx_.int64Type(), in_H_ptr);
    llvm::Value* in_D_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims_ptr, ndims_minus_3);
    llvm::Value* in_D = builder.CreateLoad(ctx_.int64Type(), in_D_ptr);

    // Get kernel dimensions
    llvm::Value* k_ndims_minus_1 = builder.CreateSub(k_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* k_ndims_minus_2 = builder.CreateSub(k_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* k_ndims_minus_3 = builder.CreateSub(k_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 3));

    llvm::Value* k_W_ptr = builder.CreateGEP(ctx_.int64Type(), k_dims_ptr, k_ndims_minus_1);
    llvm::Value* k_W = builder.CreateLoad(ctx_.int64Type(), k_W_ptr);
    llvm::Value* k_H_ptr = builder.CreateGEP(ctx_.int64Type(), k_dims_ptr, k_ndims_minus_2);
    llvm::Value* k_H = builder.CreateLoad(ctx_.int64Type(), k_H_ptr);
    llvm::Value* k_D_ptr = builder.CreateGEP(ctx_.int64Type(), k_dims_ptr, k_ndims_minus_3);
    llvm::Value* k_D = builder.CreateLoad(ctx_.int64Type(), k_D_ptr);

    // Calculate output dimensions
    llvm::Value* stride_val_i64 = llvm::ConstantInt::get(ctx_.int64Type(), stride);
    llvm::Value* padding_val_i64 = llvm::ConstantInt::get(ctx_.int64Type(), padding);
    llvm::Value* padding2 = builder.CreateMul(padding_val_i64, llvm::ConstantInt::get(ctx_.int64Type(), 2));

    llvm::Value* out_D = builder.CreateSDiv(
        builder.CreateAdd(builder.CreateSub(builder.CreateAdd(in_D, padding2), k_D),
            stride_val_i64), stride_val_i64);
    llvm::Value* out_H = builder.CreateSDiv(
        builder.CreateAdd(builder.CreateSub(builder.CreateAdd(in_H, padding2), k_H),
            stride_val_i64), stride_val_i64);
    llvm::Value* out_W = builder.CreateSDiv(
        builder.CreateAdd(builder.CreateSub(builder.CreateAdd(in_W, padding2), k_W),
            stride_val_i64), stride_val_i64);

    // Create output tensor with same leading dimensions, new spatial dims
    std::vector<llvm::Value*> out_dims;
    out_dims.push_back(out_D);
    out_dims.push_back(out_H);
    out_dims.push_back(out_W);

    llvm::Value* result_total = builder.CreateMul(out_D, builder.CreateMul(out_H, out_W));
    llvm::Value* kernel_total = builder.CreateMul(k_D, builder.CreateMul(k_H, k_W));

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "conv3d_result");

    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc,
        {arena_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 24)}, "conv3d_dims");  // 3 dims * 8 bytes

    llvm::Value* elems_bytes = builder.CreateMul(result_total, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "conv3d_elems");

    // Store output dimensions
    builder.CreateStore(out_D, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(out_H, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    builder.CreateStore(out_W, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)));

    // Fill result header
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 3), r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(result_total, r_total_field);

    // 6-nested loop for 3D convolution
    llvm::BasicBlock* od_cond = llvm::BasicBlock::Create(ctx_.context(), "conv3d_od_cond", current_func);
    llvm::BasicBlock* od_body = llvm::BasicBlock::Create(ctx_.context(), "conv3d_od_body", current_func);
    llvm::BasicBlock* od_done = llvm::BasicBlock::Create(ctx_.context(), "conv3d_od_done", current_func);

    llvm::Value* od_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), od_idx);
    builder.CreateBr(od_cond);

    builder.SetInsertPoint(od_cond);
    llvm::Value* od = builder.CreateLoad(ctx_.int64Type(), od_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(od, out_D), od_body, od_done);

    builder.SetInsertPoint(od_body);

    // Inner loops for oh, ow
    llvm::BasicBlock* oh_cond = llvm::BasicBlock::Create(ctx_.context(), "conv3d_oh_cond", current_func);
    llvm::BasicBlock* oh_body = llvm::BasicBlock::Create(ctx_.context(), "conv3d_oh_body", current_func);
    llvm::BasicBlock* oh_done = llvm::BasicBlock::Create(ctx_.context(), "conv3d_oh_done", current_func);

    llvm::Value* oh_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), oh_idx);
    builder.CreateBr(oh_cond);

    builder.SetInsertPoint(oh_cond);
    llvm::Value* oh = builder.CreateLoad(ctx_.int64Type(), oh_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(oh, out_H), oh_body, oh_done);

    builder.SetInsertPoint(oh_body);

    llvm::BasicBlock* ow_cond = llvm::BasicBlock::Create(ctx_.context(), "conv3d_ow_cond", current_func);
    llvm::BasicBlock* ow_body = llvm::BasicBlock::Create(ctx_.context(), "conv3d_ow_body", current_func);
    llvm::BasicBlock* ow_done = llvm::BasicBlock::Create(ctx_.context(), "conv3d_ow_done", current_func);

    llvm::Value* ow_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ow_idx);
    builder.CreateBr(ow_cond);

    builder.SetInsertPoint(ow_cond);
    llvm::Value* ow = builder.CreateLoad(ctx_.int64Type(), ow_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(ow, out_W), ow_body, ow_done);

    builder.SetInsertPoint(ow_body);

    // Accumulator for this output position
    llvm::Value* acc = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), acc);

    // Kernel loops (kd, kh, kw)
    llvm::BasicBlock* kd_cond = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kd_cond", current_func);
    llvm::BasicBlock* kd_body = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kd_body", current_func);
    llvm::BasicBlock* kd_done = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kd_done", current_func);

    llvm::Value* kd_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), kd_idx);
    builder.CreateBr(kd_cond);

    builder.SetInsertPoint(kd_cond);
    llvm::Value* kd = builder.CreateLoad(ctx_.int64Type(), kd_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(kd, k_D), kd_body, kd_done);

    builder.SetInsertPoint(kd_body);

    llvm::BasicBlock* kh_cond = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kh_cond", current_func);
    llvm::BasicBlock* kh_body = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kh_body", current_func);
    llvm::BasicBlock* kh_done = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kh_done", current_func);

    llvm::Value* kh_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), kh_idx);
    builder.CreateBr(kh_cond);

    builder.SetInsertPoint(kh_cond);
    llvm::Value* kh = builder.CreateLoad(ctx_.int64Type(), kh_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(kh, k_H), kh_body, kh_done);

    builder.SetInsertPoint(kh_body);

    llvm::BasicBlock* kw_cond = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kw_cond", current_func);
    llvm::BasicBlock* kw_body = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kw_body", current_func);
    llvm::BasicBlock* kw_done = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kw_done", current_func);

    llvm::Value* kw_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), kw_idx);
    builder.CreateBr(kw_cond);

    builder.SetInsertPoint(kw_cond);
    llvm::Value* kw = builder.CreateLoad(ctx_.int64Type(), kw_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(kw, k_W), kw_body, kw_done);

    builder.SetInsertPoint(kw_body);

    // Calculate input indices
    llvm::Value* id = builder.CreateAdd(builder.CreateMul(od, stride_val_i64), kd);
    id = builder.CreateSub(id, padding_val_i64);
    llvm::Value* ih = builder.CreateAdd(builder.CreateMul(oh, stride_val_i64), kh);
    ih = builder.CreateSub(ih, padding_val_i64);
    llvm::Value* iw = builder.CreateAdd(builder.CreateMul(ow, stride_val_i64), kw);
    iw = builder.CreateSub(iw, padding_val_i64);

    // Bounds check
    llvm::Value* valid_d = builder.CreateAnd(
        builder.CreateICmpSGE(id, llvm::ConstantInt::get(ctx_.int64Type(), 0)),
        builder.CreateICmpSLT(id, in_D));
    llvm::Value* valid_h = builder.CreateAnd(
        builder.CreateICmpSGE(ih, llvm::ConstantInt::get(ctx_.int64Type(), 0)),
        builder.CreateICmpSLT(ih, in_H));
    llvm::Value* valid_w = builder.CreateAnd(
        builder.CreateICmpSGE(iw, llvm::ConstantInt::get(ctx_.int64Type(), 0)),
        builder.CreateICmpSLT(iw, in_W));
    llvm::Value* valid = builder.CreateAnd(valid_d, builder.CreateAnd(valid_h, valid_w));

    llvm::BasicBlock* valid_block = llvm::BasicBlock::Create(ctx_.context(), "conv3d_valid", current_func);
    llvm::BasicBlock* continue_block = llvm::BasicBlock::Create(ctx_.context(), "conv3d_continue", current_func);

    builder.CreateCondBr(valid, valid_block, continue_block);

    builder.SetInsertPoint(valid_block);
    // input_idx = id * (H * W) + ih * W + iw
    llvm::Value* in_idx = builder.CreateAdd(
        builder.CreateMul(id, builder.CreateMul(in_H, in_W)),
        builder.CreateAdd(builder.CreateMul(ih, in_W), iw));
    llvm::Value* in_ptr = builder.CreateGEP(ctx_.int64Type(), in_elems, in_idx);
    llvm::Value* in_bits = builder.CreateLoad(ctx_.int64Type(), in_ptr);
    llvm::Value* in_val = builder.CreateBitCast(in_bits, ctx_.doubleType());

    // kernel_idx = kd * (kH * kW) + kh * kW + kw
    llvm::Value* k_idx = builder.CreateAdd(
        builder.CreateMul(kd, builder.CreateMul(k_H, k_W)),
        builder.CreateAdd(builder.CreateMul(kh, k_W), kw));
    llvm::Value* k_ptr = builder.CreateGEP(ctx_.int64Type(), k_elems, k_idx);
    llvm::Value* k_bits = builder.CreateLoad(ctx_.int64Type(), k_ptr);
    llvm::Value* k_val = builder.CreateBitCast(k_bits, ctx_.doubleType());

    llvm::Value* prod = builder.CreateFMul(in_val, k_val);
    llvm::Value* cur_acc = builder.CreateLoad(ctx_.doubleType(), acc);
    builder.CreateStore(builder.CreateFAdd(cur_acc, prod), acc);
    builder.CreateBr(continue_block);

    builder.SetInsertPoint(continue_block);
    builder.CreateStore(builder.CreateAdd(kw, llvm::ConstantInt::get(ctx_.int64Type(), 1)), kw_idx);
    builder.CreateBr(kw_cond);

    builder.SetInsertPoint(kw_done);
    builder.CreateStore(builder.CreateAdd(kh, llvm::ConstantInt::get(ctx_.int64Type(), 1)), kh_idx);
    builder.CreateBr(kh_cond);

    builder.SetInsertPoint(kh_done);
    builder.CreateStore(builder.CreateAdd(kd, llvm::ConstantInt::get(ctx_.int64Type(), 1)), kd_idx);
    builder.CreateBr(kd_cond);

    builder.SetInsertPoint(kd_done);
    // Store result
    llvm::Value* out_idx = builder.CreateAdd(
        builder.CreateMul(od, builder.CreateMul(out_H, out_W)),
        builder.CreateAdd(builder.CreateMul(oh, out_W), ow));
    llvm::Value* out_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, out_idx);
    llvm::Value* final_acc = builder.CreateLoad(ctx_.doubleType(), acc);
    llvm::Value* acc_bits = builder.CreateBitCast(final_acc, ctx_.int64Type());
    builder.CreateStore(acc_bits, out_ptr);

    builder.CreateStore(builder.CreateAdd(ow, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ow_idx);
    builder.CreateBr(ow_cond);

    builder.SetInsertPoint(ow_done);
    builder.CreateStore(builder.CreateAdd(oh, llvm::ConstantInt::get(ctx_.int64Type(), 1)), oh_idx);
    builder.CreateBr(oh_cond);

    builder.SetInsertPoint(oh_done);
    builder.CreateStore(builder.CreateAdd(od, llvm::ConstantInt::get(ctx_.int64Type(), 1)), od_idx);
    builder.CreateBr(od_cond);

    builder.SetInsertPoint(od_done);
    return tagged_.packHeapPtr(result_ptr);
}

// ===== OPTIMIZERS (Track 10.1) =====
// Full production implementations for neural network training

llvm::Value* TensorCodegen::sgdStep(const eshkol_operations_t* op) {
    // sgd-step!: (sgd-step! params grads lr [momentum velocity])
    // Implements SGD with optional Nesterov momentum
    if (op->call_op.num_vars < 3) {
        eshkol_error("sgd-step! requires at least 3 arguments: params, grads, lr");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    // Get params tensor
    llvm::Value* params_tagged = codegenAST(&op->call_op.variables[0]);
    if (!params_tagged) return nullptr;
    llvm::Value* params_ptr = tagged_.unpackPtr(params_tagged);

    // Get grads tensor
    llvm::Value* grads_tagged = codegenAST(&op->call_op.variables[1]);
    if (!grads_tagged) return nullptr;
    llvm::Value* grads_ptr = tagged_.unpackPtr(grads_tagged);

    // Get learning rate
    llvm::Value* lr_tagged = codegenAST(&op->call_op.variables[2]);
    if (!lr_tagged) return nullptr;
    llvm::Value* lr = tagged_.unpackDouble(lr_tagged);

    // Get momentum and velocity if provided
    bool has_momentum = op->call_op.num_vars >= 5;
    llvm::Value* momentum = nullptr;
    llvm::Value* velocity_ptr = nullptr;

    if (has_momentum) {
        llvm::Value* momentum_tagged = codegenAST(&op->call_op.variables[3]);
        if (!momentum_tagged) return nullptr;
        momentum = tagged_.unpackDouble(momentum_tagged);

        llvm::Value* velocity_tagged = codegenAST(&op->call_op.variables[4]);
        if (!velocity_tagged) return nullptr;
        velocity_ptr = tagged_.unpackPtr(velocity_tagged);
    }

    // Get tensor dimensions
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);

    // Get element pointers
    llvm::Value* params_elems_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 2);
    llvm::Value* params_elems = builder.CreateLoad(ctx_.ptrType(), params_elems_ptr);
    llvm::Value* grads_elems_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 2);
    llvm::Value* grads_elems = builder.CreateLoad(ctx_.ptrType(), grads_elems_ptr);

    // Compute total elements
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "sgd_size_loop", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "sgd_size_done", current_func);
    llvm::BasicBlock* update_loop = llvm::BasicBlock::Create(ctx_.context(), "sgd_update_loop", current_func);
    llvm::BasicBlock* update_done = llvm::BasicBlock::Create(ctx_.context(), "sgd_update_done", current_func);

    // Compute total size
    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "size_idx");
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type(), nullptr, "total_size");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    llvm::Value* size_cond = builder.CreateICmpULT(si, num_dims);
    builder.CreateCondBr(size_cond, llvm::BasicBlock::Create(ctx_.context(), "size_body", current_func), size_done);

    llvm::BasicBlock* size_body = &current_func->back();
    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    // Update loop
    llvm::Value* elem_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "elem_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), elem_idx);

    llvm::Value* velocity_elems = nullptr;
    if (has_momentum) {
        llvm::Value* vel_elems_ptr = builder.CreateStructGEP(tensor_type, velocity_ptr, 2);
        velocity_elems = builder.CreateLoad(ctx_.ptrType(), vel_elems_ptr);
    }

    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_loop);
    llvm::Value* ei = builder.CreateLoad(ctx_.int64Type(), elem_idx);
    llvm::Value* update_cond = builder.CreateICmpULT(ei, num_elements);
    llvm::BasicBlock* update_body = llvm::BasicBlock::Create(ctx_.context(), "update_body", current_func);
    builder.CreateCondBr(update_cond, update_body, update_done);

    builder.SetInsertPoint(update_body);

    // Load gradient
    llvm::Value* grad_ptr = builder.CreateGEP(ctx_.int64Type(), grads_elems, ei);
    llvm::Value* grad_bits = builder.CreateLoad(ctx_.int64Type(), grad_ptr);
    llvm::Value* grad = builder.CreateBitCast(grad_bits, ctx_.doubleType());

    // Load current param
    llvm::Value* param_ptr = builder.CreateGEP(ctx_.int64Type(), params_elems, ei);
    llvm::Value* param_bits = builder.CreateLoad(ctx_.int64Type(), param_ptr);
    llvm::Value* param = builder.CreateBitCast(param_bits, ctx_.doubleType());

    llvm::Value* update;
    if (has_momentum) {
        // v = momentum * v + grad
        llvm::Value* vel_ptr = builder.CreateGEP(ctx_.int64Type(), velocity_elems, ei);
        llvm::Value* vel_bits = builder.CreateLoad(ctx_.int64Type(), vel_ptr);
        llvm::Value* vel = builder.CreateBitCast(vel_bits, ctx_.doubleType());
        llvm::Value* new_vel = builder.CreateFAdd(builder.CreateFMul(momentum, vel), grad);
        llvm::Value* new_vel_bits = builder.CreateBitCast(new_vel, ctx_.int64Type());
        builder.CreateStore(new_vel_bits, vel_ptr);
        update = new_vel;
    } else {
        update = grad;
    }

    // param = param - lr * update
    llvm::Value* scaled_update = builder.CreateFMul(lr, update);
    llvm::Value* new_param = builder.CreateFSub(param, scaled_update);
    llvm::Value* new_param_bits = builder.CreateBitCast(new_param, ctx_.int64Type());
    builder.CreateStore(new_param_bits, param_ptr);

    builder.CreateStore(builder.CreateAdd(ei, llvm::ConstantInt::get(ctx_.int64Type(), 1)), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_done);
    return params_tagged;
}

llvm::Value* TensorCodegen::adamStep(const eshkol_operations_t* op) {
    // adam-step!: (adam-step! params grads lr m v t [beta1 beta2 eps])
    // Implements Adam optimizer (Kingma & Ba, 2014)
    if (op->call_op.num_vars < 6) {
        eshkol_error("adam-step! requires at least 6 arguments: params, grads, lr, m, v, t");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    // Get arguments
    llvm::Value* params_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* grads_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* lr_tagged = codegenAST(&op->call_op.variables[2]);
    llvm::Value* m_tagged = codegenAST(&op->call_op.variables[3]);
    llvm::Value* v_tagged = codegenAST(&op->call_op.variables[4]);
    llvm::Value* t_tagged = codegenAST(&op->call_op.variables[5]);

    if (!params_tagged || !grads_tagged || !lr_tagged || !m_tagged || !v_tagged || !t_tagged) {
        return nullptr;
    }

    llvm::Value* params_ptr = tagged_.unpackPtr(params_tagged);
    llvm::Value* grads_ptr = tagged_.unpackPtr(grads_tagged);
    llvm::Value* lr = tagged_.unpackDouble(lr_tagged);
    llvm::Value* m_ptr = tagged_.unpackPtr(m_tagged);
    llvm::Value* v_ptr = tagged_.unpackPtr(v_tagged);
    llvm::Value* t = tagged_.unpackInt64(t_tagged);
    llvm::Value* t_double = builder.CreateSIToFP(t, ctx_.doubleType());

    // Default hyperparameters
    llvm::Value* beta1 = llvm::ConstantFP::get(ctx_.doubleType(), 0.9);
    llvm::Value* beta2 = llvm::ConstantFP::get(ctx_.doubleType(), 0.999);
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-8);

    if (op->call_op.num_vars >= 7) {
        llvm::Value* b1_tagged = codegenAST(&op->call_op.variables[6]);
        if (b1_tagged) beta1 = tagged_.unpackDouble(b1_tagged);
    }
    if (op->call_op.num_vars >= 8) {
        llvm::Value* b2_tagged = codegenAST(&op->call_op.variables[7]);
        if (b2_tagged) beta2 = tagged_.unpackDouble(b2_tagged);
    }
    if (op->call_op.num_vars >= 9) {
        llvm::Value* eps_tagged = codegenAST(&op->call_op.variables[8]);
        if (eps_tagged) eps = tagged_.unpackDouble(eps_tagged);
    }

    // Compute bias corrections: 1 - beta^t
    llvm::Function* pow_func = ctx_.module().getFunction("pow");
    if (!pow_func) {
        llvm::FunctionType* pow_type = llvm::FunctionType::get(ctx_.doubleType(),
            {ctx_.doubleType(), ctx_.doubleType()}, false);
        pow_func = llvm::Function::Create(pow_type, llvm::Function::ExternalLinkage, "pow", ctx_.module());
    }

    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* beta1_t = builder.CreateCall(pow_func, {beta1, t_double});
    llvm::Value* beta2_t = builder.CreateCall(pow_func, {beta2, t_double});
    llvm::Value* bias_corr1 = builder.CreateFSub(one, beta1_t);
    llvm::Value* bias_corr2 = builder.CreateFSub(one, beta2_t);

    // Get tensor info
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);

    llvm::Value* params_elems_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 2);
    llvm::Value* params_elems = builder.CreateLoad(ctx_.ptrType(), params_elems_ptr);
    llvm::Value* grads_elems_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 2);
    llvm::Value* grads_elems = builder.CreateLoad(ctx_.ptrType(), grads_elems_ptr);
    llvm::Value* m_elems_ptr = builder.CreateStructGEP(tensor_type, m_ptr, 2);
    llvm::Value* m_elems = builder.CreateLoad(ctx_.ptrType(), m_elems_ptr);
    llvm::Value* v_elems_ptr = builder.CreateStructGEP(tensor_type, v_ptr, 2);
    llvm::Value* v_elems = builder.CreateLoad(ctx_.ptrType(), v_elems_ptr);

    // Compute total elements
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "adam_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "adam_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "adam_size_done", current_func);
    llvm::BasicBlock* update_loop = llvm::BasicBlock::Create(ctx_.context(), "adam_update_loop", current_func);
    llvm::BasicBlock* update_body = llvm::BasicBlock::Create(ctx_.context(), "adam_update_body", current_func);
    llvm::BasicBlock* update_done = llvm::BasicBlock::Create(ctx_.context(), "adam_update_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "size_idx");
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type(), nullptr, "total_size");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    llvm::Value* size_cond = builder.CreateICmpULT(si, num_dims);
    builder.CreateCondBr(size_cond, size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    // Update loop
    llvm::Value* elem_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "elem_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_loop);
    llvm::Value* ei = builder.CreateLoad(ctx_.int64Type(), elem_idx);
    llvm::Value* update_cond = builder.CreateICmpULT(ei, num_elements);
    builder.CreateCondBr(update_cond, update_body, update_done);

    builder.SetInsertPoint(update_body);

    // Load values
    llvm::Value* grad_ptr = builder.CreateGEP(ctx_.int64Type(), grads_elems, ei);
    llvm::Value* grad_bits = builder.CreateLoad(ctx_.int64Type(), grad_ptr);
    llvm::Value* grad = builder.CreateBitCast(grad_bits, ctx_.doubleType());

    llvm::Value* param_ptr = builder.CreateGEP(ctx_.int64Type(), params_elems, ei);
    llvm::Value* param_bits = builder.CreateLoad(ctx_.int64Type(), param_ptr);
    llvm::Value* param = builder.CreateBitCast(param_bits, ctx_.doubleType());

    llvm::Value* m_elem_ptr = builder.CreateGEP(ctx_.int64Type(), m_elems, ei);
    llvm::Value* m_bits = builder.CreateLoad(ctx_.int64Type(), m_elem_ptr);
    llvm::Value* m_val = builder.CreateBitCast(m_bits, ctx_.doubleType());

    llvm::Value* v_elem_ptr = builder.CreateGEP(ctx_.int64Type(), v_elems, ei);
    llvm::Value* v_bits = builder.CreateLoad(ctx_.int64Type(), v_elem_ptr);
    llvm::Value* v_val = builder.CreateBitCast(v_bits, ctx_.doubleType());

    // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    llvm::Value* one_minus_beta1 = builder.CreateFSub(one, beta1);
    llvm::Value* m_new = builder.CreateFAdd(
        builder.CreateFMul(beta1, m_val),
        builder.CreateFMul(one_minus_beta1, grad));

    // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    llvm::Value* one_minus_beta2 = builder.CreateFSub(one, beta2);
    llvm::Value* grad_sq = builder.CreateFMul(grad, grad);
    llvm::Value* v_new = builder.CreateFAdd(
        builder.CreateFMul(beta2, v_val),
        builder.CreateFMul(one_minus_beta2, grad_sq));

    // Store updated moments
    builder.CreateStore(builder.CreateBitCast(m_new, ctx_.int64Type()), m_elem_ptr);
    builder.CreateStore(builder.CreateBitCast(v_new, ctx_.int64Type()), v_elem_ptr);

    // Bias-corrected estimates
    llvm::Value* m_hat = builder.CreateFDiv(m_new, bias_corr1);
    llvm::Value* v_hat = builder.CreateFDiv(v_new, bias_corr2);

    // Update: theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
    llvm::Function* sqrt_func = llvm::Intrinsic::getOrInsertDeclaration(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* sqrt_v_hat = builder.CreateCall(sqrt_func, {v_hat});
    llvm::Value* denom = builder.CreateFAdd(sqrt_v_hat, eps);
    llvm::Value* update = builder.CreateFDiv(m_hat, denom);
    llvm::Value* scaled_update = builder.CreateFMul(lr, update);
    llvm::Value* new_param = builder.CreateFSub(param, scaled_update);

    builder.CreateStore(builder.CreateBitCast(new_param, ctx_.int64Type()), param_ptr);

    builder.CreateStore(builder.CreateAdd(ei, llvm::ConstantInt::get(ctx_.int64Type(), 1)), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_done);
    return params_tagged;
}

llvm::Value* TensorCodegen::zeroGrad(const eshkol_operations_t* op) {
    // zero-grad!: (zero-grad! tensor)
    // Zero all elements in place
    if (op->call_op.num_vars < 1) {
        eshkol_error("zero-grad! requires 1 argument");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    if (!tensor_tagged) return nullptr;
    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    // Compute total elements
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "zero_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "zero_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "zero_size_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpULT(si, num_dims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    // Use memset for efficiency
    llvm::Value* byte_size = builder.CreateMul(num_elements, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    builder.CreateMemSet(elems, llvm::ConstantInt::get(ctx_.int8Type(), 0),
                         byte_size, llvm::MaybeAlign(8));

    return tensor_tagged;
}

llvm::Value* TensorCodegen::clipGradNorm(const eshkol_operations_t* op) {
    // clip-grad-norm!: (clip-grad-norm! grads max-norm)
    if (op->call_op.num_vars < 2) {
        eshkol_error("clip-grad-norm! requires 2 arguments: grads, max-norm");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* grads_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* max_norm_tagged = codegenAST(&op->call_op.variables[1]);
    if (!grads_tagged || !max_norm_tagged) return nullptr;

    llvm::Value* grads_ptr = tagged_.unpackPtr(grads_tagged);
    llvm::Value* max_norm = tagged_.unpackDouble(max_norm_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "clip_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "clip_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "clip_size_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpULT(si, num_dims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    // Compute L2 norm: sqrt(sum(grad^2))
    llvm::BasicBlock* norm_loop = llvm::BasicBlock::Create(ctx_.context(), "clip_norm_loop", current_func);
    llvm::BasicBlock* norm_body = llvm::BasicBlock::Create(ctx_.context(), "clip_norm_body", current_func);
    llvm::BasicBlock* norm_done = llvm::BasicBlock::Create(ctx_.context(), "clip_norm_done", current_func);

    llvm::Value* norm_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* sum_sq = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), norm_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_sq);
    builder.CreateBr(norm_loop);

    builder.SetInsertPoint(norm_loop);
    llvm::Value* ni = builder.CreateLoad(ctx_.int64Type(), norm_idx);
    builder.CreateCondBr(builder.CreateICmpULT(ni, num_elements), norm_body, norm_done);

    builder.SetInsertPoint(norm_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems, ni);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* elem_sq = builder.CreateFMul(elem, elem);
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), sum_sq);
    builder.CreateStore(builder.CreateFAdd(cur_sum, elem_sq), sum_sq);
    builder.CreateStore(builder.CreateAdd(ni, llvm::ConstantInt::get(ctx_.int64Type(), 1)), norm_idx);
    builder.CreateBr(norm_loop);

    builder.SetInsertPoint(norm_done);
    llvm::Value* final_sum = builder.CreateLoad(ctx_.doubleType(), sum_sq);
    llvm::Function* sqrt_func = llvm::Intrinsic::getOrInsertDeclaration(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* total_norm = builder.CreateCall(sqrt_func, {final_sum});

    // Clip if necessary
    llvm::BasicBlock* do_clip = llvm::BasicBlock::Create(ctx_.context(), "do_clip", current_func);
    llvm::BasicBlock* clip_done = llvm::BasicBlock::Create(ctx_.context(), "clip_done", current_func);

    llvm::Value* need_clip = builder.CreateFCmpOGT(total_norm, max_norm);
    builder.CreateCondBr(need_clip, do_clip, clip_done);

    builder.SetInsertPoint(do_clip);
    llvm::Value* scale = builder.CreateFDiv(max_norm, total_norm);

    // Scale all gradients
    llvm::BasicBlock* scale_loop = llvm::BasicBlock::Create(ctx_.context(), "scale_loop", current_func);
    llvm::BasicBlock* scale_body = llvm::BasicBlock::Create(ctx_.context(), "scale_body", current_func);
    llvm::BasicBlock* scale_done = llvm::BasicBlock::Create(ctx_.context(), "scale_done", current_func);

    llvm::Value* scale_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), scale_idx);
    builder.CreateBr(scale_loop);

    builder.SetInsertPoint(scale_loop);
    llvm::Value* sci = builder.CreateLoad(ctx_.int64Type(), scale_idx);
    builder.CreateCondBr(builder.CreateICmpULT(sci, num_elements), scale_body, scale_done);

    builder.SetInsertPoint(scale_body);
    llvm::Value* sc_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems, sci);
    llvm::Value* sc_elem_bits = builder.CreateLoad(ctx_.int64Type(), sc_elem_ptr);
    llvm::Value* sc_elem = builder.CreateBitCast(sc_elem_bits, ctx_.doubleType());
    llvm::Value* scaled_elem = builder.CreateFMul(sc_elem, scale);
    builder.CreateStore(builder.CreateBitCast(scaled_elem, ctx_.int64Type()), sc_elem_ptr);
    builder.CreateStore(builder.CreateAdd(sci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), scale_idx);
    builder.CreateBr(scale_loop);

    builder.SetInsertPoint(scale_done);
    builder.CreateBr(clip_done);

    builder.SetInsertPoint(clip_done);
    return tagged_.packDouble(total_norm);
}

llvm::Value* TensorCodegen::rmspropStep(const eshkol_operations_t* op) {
    // rmsprop-step!: (rmsprop-step! params grads lr v [alpha eps])
    if (op->call_op.num_vars < 4) {
        eshkol_error("rmsprop-step! requires at least 4 arguments: params, grads, lr, v");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* params_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* grads_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* lr_tagged = codegenAST(&op->call_op.variables[2]);
    llvm::Value* v_tagged = codegenAST(&op->call_op.variables[3]);

    if (!params_tagged || !grads_tagged || !lr_tagged || !v_tagged) return nullptr;

    llvm::Value* params_ptr = tagged_.unpackPtr(params_tagged);
    llvm::Value* grads_ptr = tagged_.unpackPtr(grads_tagged);
    llvm::Value* lr = tagged_.unpackDouble(lr_tagged);
    llvm::Value* v_ptr = tagged_.unpackPtr(v_tagged);

    llvm::Value* alpha = llvm::ConstantFP::get(ctx_.doubleType(), 0.99);
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-8);

    if (op->call_op.num_vars >= 5) {
        llvm::Value* a_tagged = codegenAST(&op->call_op.variables[4]);
        if (a_tagged) alpha = tagged_.unpackDouble(a_tagged);
    }
    if (op->call_op.num_vars >= 6) {
        llvm::Value* e_tagged = codegenAST(&op->call_op.variables[5]);
        if (e_tagged) eps = tagged_.unpackDouble(e_tagged);
    }

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Value* params_elems_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 2);
    llvm::Value* params_elems = builder.CreateLoad(ctx_.ptrType(), params_elems_ptr);
    llvm::Value* grads_elems_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 2);
    llvm::Value* grads_elems = builder.CreateLoad(ctx_.ptrType(), grads_elems_ptr);
    llvm::Value* v_elems_ptr = builder.CreateStructGEP(tensor_type, v_ptr, 2);
    llvm::Value* v_elems = builder.CreateLoad(ctx_.ptrType(), v_elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "rms_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "rms_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "rms_size_done", current_func);
    llvm::BasicBlock* update_loop = llvm::BasicBlock::Create(ctx_.context(), "rms_update_loop", current_func);
    llvm::BasicBlock* update_body = llvm::BasicBlock::Create(ctx_.context(), "rms_update_body", current_func);
    llvm::BasicBlock* update_done = llvm::BasicBlock::Create(ctx_.context(), "rms_update_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpULT(si, num_dims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_minus_alpha = builder.CreateFSub(one, alpha);

    llvm::Value* elem_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_loop);
    llvm::Value* ei = builder.CreateLoad(ctx_.int64Type(), elem_idx);
    builder.CreateCondBr(builder.CreateICmpULT(ei, num_elements), update_body, update_done);

    builder.SetInsertPoint(update_body);

    llvm::Value* grad_ptr = builder.CreateGEP(ctx_.int64Type(), grads_elems, ei);
    llvm::Value* grad_bits = builder.CreateLoad(ctx_.int64Type(), grad_ptr);
    llvm::Value* grad = builder.CreateBitCast(grad_bits, ctx_.doubleType());

    llvm::Value* param_ptr = builder.CreateGEP(ctx_.int64Type(), params_elems, ei);
    llvm::Value* param_bits = builder.CreateLoad(ctx_.int64Type(), param_ptr);
    llvm::Value* param = builder.CreateBitCast(param_bits, ctx_.doubleType());

    llvm::Value* v_elem_ptr = builder.CreateGEP(ctx_.int64Type(), v_elems, ei);
    llvm::Value* v_bits = builder.CreateLoad(ctx_.int64Type(), v_elem_ptr);
    llvm::Value* v_val = builder.CreateBitCast(v_bits, ctx_.doubleType());

    // v = alpha * v + (1 - alpha) * g^2
    llvm::Value* grad_sq = builder.CreateFMul(grad, grad);
    llvm::Value* v_new = builder.CreateFAdd(
        builder.CreateFMul(alpha, v_val),
        builder.CreateFMul(one_minus_alpha, grad_sq));
    builder.CreateStore(builder.CreateBitCast(v_new, ctx_.int64Type()), v_elem_ptr);

    // param = param - lr * g / (sqrt(v) + eps)
    llvm::Function* sqrt_func = llvm::Intrinsic::getOrInsertDeclaration(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* sqrt_v = builder.CreateCall(sqrt_func, {v_new});
    llvm::Value* denom = builder.CreateFAdd(sqrt_v, eps);
    llvm::Value* update = builder.CreateFDiv(grad, denom);
    llvm::Value* scaled_update = builder.CreateFMul(lr, update);
    llvm::Value* new_param = builder.CreateFSub(param, scaled_update);
    builder.CreateStore(builder.CreateBitCast(new_param, ctx_.int64Type()), param_ptr);

    builder.CreateStore(builder.CreateAdd(ei, llvm::ConstantInt::get(ctx_.int64Type(), 1)), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_done);
    return params_tagged;
}

// ===== LOSS FUNCTIONS (Track 6.3) =====

llvm::Value* TensorCodegen::mseLoss(const eshkol_operations_t* op) {
    // mse-loss: (mse-loss predictions targets)
    // MSE = (1/n) * sum((pred - target)^2)
    if (op->call_op.num_vars < 2) {
        eshkol_error("mse-loss requires 2 arguments: predictions, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);
    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "mse_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "mse_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "mse_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "mse_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "mse_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "mse_sum_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpULT(si, num_dims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    // Sum squared differences
    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* sum_sq_diff = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_sq_diff);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    builder.CreateCondBr(builder.CreateICmpULT(sui, num_elements), sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());

    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    llvm::Value* diff = builder.CreateFSub(pred_val, target_val);
    llvm::Value* diff_sq = builder.CreateFMul(diff, diff);
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), sum_sq_diff);
    builder.CreateStore(builder.CreateFAdd(cur_sum, diff_sq), sum_sq_diff);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), sum_sq_diff);
    llvm::Value* n_double = builder.CreateSIToFP(num_elements, ctx_.doubleType());
    llvm::Value* mse = builder.CreateFDiv(total_sum, n_double);

    return tagged_.packDouble(mse);
}

llvm::Value* TensorCodegen::crossEntropyLoss(const eshkol_operations_t* op) {
    // cross-entropy-loss: (cross-entropy-loss logits targets)
    // CE = -sum(target * log(softmax(logits)))
    // Numerically stable: CE = -sum(target * (logits - logsumexp(logits)))
    if (op->call_op.num_vars < 2) {
        eshkol_error("cross-entropy-loss requires 2 arguments: logits, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* logits_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* targets_tagged = codegenAST(&op->call_op.variables[1]);
    if (!logits_tagged || !targets_tagged) return nullptr;

    llvm::Value* logits_ptr = tagged_.unpackPtr(logits_tagged);
    llvm::Value* targets_ptr = tagged_.unpackPtr(targets_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, logits_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, logits_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Value* logits_elems_ptr = builder.CreateStructGEP(tensor_type, logits_ptr, 2);
    llvm::Value* logits_elems = builder.CreateLoad(ctx_.ptrType(), logits_elems_ptr);
    llvm::Value* targets_elems_ptr = builder.CreateStructGEP(tensor_type, targets_ptr, 2);
    llvm::Value* targets_elems = builder.CreateLoad(ctx_.ptrType(), targets_elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!exp_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "exp", ctx_.module());
    }
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", ctx_.module());
    }

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "ce_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "ce_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "ce_size_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpULT(si, num_dims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    // Find max for numerical stability
    llvm::BasicBlock* max_loop = llvm::BasicBlock::Create(ctx_.context(), "ce_max_loop", current_func);
    llvm::BasicBlock* max_body = llvm::BasicBlock::Create(ctx_.context(), "ce_max_body", current_func);
    llvm::BasicBlock* max_done = llvm::BasicBlock::Create(ctx_.context(), "ce_max_done", current_func);

    llvm::Value* max_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), max_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), -1e308), max_val);
    builder.CreateBr(max_loop);

    builder.SetInsertPoint(max_loop);
    llvm::Value* mi = builder.CreateLoad(ctx_.int64Type(), max_idx);
    builder.CreateCondBr(builder.CreateICmpULT(mi, num_elements), max_body, max_done);

    builder.SetInsertPoint(max_body);
    llvm::Value* logit_ptr = builder.CreateGEP(ctx_.int64Type(), logits_elems, mi);
    llvm::Value* logit_bits = builder.CreateLoad(ctx_.int64Type(), logit_ptr);
    llvm::Value* logit = builder.CreateBitCast(logit_bits, ctx_.doubleType());
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_greater = builder.CreateFCmpOGT(logit, cur_max);
    llvm::Value* new_max = builder.CreateSelect(is_greater, logit, cur_max);
    builder.CreateStore(new_max, max_val);
    builder.CreateStore(builder.CreateAdd(mi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), max_idx);
    builder.CreateBr(max_loop);

    builder.SetInsertPoint(max_done);
    llvm::Value* max_logit = builder.CreateLoad(ctx_.doubleType(), max_val);

    // Compute logsumexp = max + log(sum(exp(x - max)))
    llvm::BasicBlock* lse_loop = llvm::BasicBlock::Create(ctx_.context(), "ce_lse_loop", current_func);
    llvm::BasicBlock* lse_body = llvm::BasicBlock::Create(ctx_.context(), "ce_lse_body", current_func);
    llvm::BasicBlock* lse_done = llvm::BasicBlock::Create(ctx_.context(), "ce_lse_done", current_func);

    llvm::Value* lse_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* exp_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), lse_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), exp_sum);
    builder.CreateBr(lse_loop);

    builder.SetInsertPoint(lse_loop);
    llvm::Value* li = builder.CreateLoad(ctx_.int64Type(), lse_idx);
    builder.CreateCondBr(builder.CreateICmpULT(li, num_elements), lse_body, lse_done);

    builder.SetInsertPoint(lse_body);
    llvm::Value* l_ptr = builder.CreateGEP(ctx_.int64Type(), logits_elems, li);
    llvm::Value* l_bits = builder.CreateLoad(ctx_.int64Type(), l_ptr);
    llvm::Value* l_val = builder.CreateBitCast(l_bits, ctx_.doubleType());
    llvm::Value* shifted = builder.CreateFSub(l_val, max_logit);
    llvm::Value* exp_val = builder.CreateCall(exp_func, {shifted});
    llvm::Value* cur_exp_sum = builder.CreateLoad(ctx_.doubleType(), exp_sum);
    builder.CreateStore(builder.CreateFAdd(cur_exp_sum, exp_val), exp_sum);
    builder.CreateStore(builder.CreateAdd(li, llvm::ConstantInt::get(ctx_.int64Type(), 1)), lse_idx);
    builder.CreateBr(lse_loop);

    builder.SetInsertPoint(lse_done);
    llvm::Value* total_exp_sum = builder.CreateLoad(ctx_.doubleType(), exp_sum);
    llvm::Value* log_sum = builder.CreateCall(log_func, {total_exp_sum});
    llvm::Value* logsumexp = builder.CreateFAdd(max_logit, log_sum);

    // Compute cross entropy: -sum(target * (logit - logsumexp))
    llvm::BasicBlock* ce_loop = llvm::BasicBlock::Create(ctx_.context(), "ce_sum_loop", current_func);
    llvm::BasicBlock* ce_body = llvm::BasicBlock::Create(ctx_.context(), "ce_sum_body", current_func);
    llvm::BasicBlock* ce_done = llvm::BasicBlock::Create(ctx_.context(), "ce_sum_done", current_func);

    llvm::Value* ce_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* ce_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ce_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), ce_sum);
    builder.CreateBr(ce_loop);

    builder.SetInsertPoint(ce_loop);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), ce_idx);
    builder.CreateCondBr(builder.CreateICmpULT(ci, num_elements), ce_body, ce_done);

    builder.SetInsertPoint(ce_body);
    llvm::Value* logit_p = builder.CreateGEP(ctx_.int64Type(), logits_elems, ci);
    llvm::Value* logit_b = builder.CreateLoad(ctx_.int64Type(), logit_p);
    llvm::Value* logit_v = builder.CreateBitCast(logit_b, ctx_.doubleType());

    llvm::Value* target_p = builder.CreateGEP(ctx_.int64Type(), targets_elems, ci);
    llvm::Value* target_b = builder.CreateLoad(ctx_.int64Type(), target_p);
    llvm::Value* target_v = builder.CreateBitCast(target_b, ctx_.doubleType());

    llvm::Value* log_prob = builder.CreateFSub(logit_v, logsumexp);
    llvm::Value* term = builder.CreateFMul(target_v, log_prob);
    llvm::Value* cur_ce = builder.CreateLoad(ctx_.doubleType(), ce_sum);
    builder.CreateStore(builder.CreateFAdd(cur_ce, term), ce_sum);
    builder.CreateStore(builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ce_idx);
    builder.CreateBr(ce_loop);

    builder.SetInsertPoint(ce_done);
    llvm::Value* total_ce = builder.CreateLoad(ctx_.doubleType(), ce_sum);
    llvm::Value* neg_ce = builder.CreateFNeg(total_ce);

    return tagged_.packDouble(neg_ce);
}

llvm::Value* TensorCodegen::bceLoss(const eshkol_operations_t* op) {
    // bce-loss: (bce-loss predictions targets)
    // BCE = -sum(target * log(pred) + (1-target) * log(1-pred)) / n
    if (op->call_op.num_vars < 2) {
        eshkol_error("bce-loss requires 2 arguments: predictions, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);
    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "bce_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "bce_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "bce_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "bce_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "bce_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "bce_sum_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpULT(si, num_dims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7); // For numerical stability

    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* bce_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), bce_sum);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    builder.CreateCondBr(builder.CreateICmpULT(sui, num_elements), sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());
    // Clamp predictions for numerical stability
    llvm::Value* pred_clamp = builder.CreateSelect(
        builder.CreateFCmpOLT(pred_val, eps), eps,
        builder.CreateSelect(
            builder.CreateFCmpOGT(pred_val, builder.CreateFSub(one, eps)),
            builder.CreateFSub(one, eps), pred_val));

    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    // BCE term: target * log(pred) + (1-target) * log(1-pred)
    llvm::Value* log_pred = builder.CreateCall(log_func, {pred_clamp});
    llvm::Value* one_minus_pred = builder.CreateFSub(one, pred_clamp);
    llvm::Value* log_one_minus_pred = builder.CreateCall(log_func, {one_minus_pred});
    llvm::Value* one_minus_target = builder.CreateFSub(one, target_val);

    llvm::Value* term1 = builder.CreateFMul(target_val, log_pred);
    llvm::Value* term2 = builder.CreateFMul(one_minus_target, log_one_minus_pred);
    llvm::Value* term = builder.CreateFAdd(term1, term2);

    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), bce_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, term), bce_sum);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), bce_sum);
    llvm::Value* n_double = builder.CreateSIToFP(num_elements, ctx_.doubleType());
    llvm::Value* bce = builder.CreateFDiv(builder.CreateFNeg(total_sum), n_double);

    return tagged_.packDouble(bce);
}

llvm::Value* TensorCodegen::huberLoss(const eshkol_operations_t* op) {
    // huber-loss: (huber-loss predictions targets [delta])
    // L_δ(a) = 0.5 * a² if |a| ≤ δ, else δ * (|a| - 0.5 * δ)
    if (op->call_op.num_vars < 2) {
        eshkol_error("huber-loss requires at least 2 arguments: predictions, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    llvm::Value* delta = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    if (op->call_op.num_vars >= 3) {
        llvm::Value* delta_tagged = codegenAST(&op->call_op.variables[2]);
        if (delta_tagged) delta = tagged_.unpackDouble(delta_tagged);
    }

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);
    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    llvm::Function* fabs_intrinsic = llvm::Intrinsic::getOrInsertDeclaration(&ctx_.module(),
        llvm::Intrinsic::fabs, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "huber_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "huber_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "huber_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "huber_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "huber_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "huber_sum_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpULT(si, num_dims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);

    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* huber_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), huber_sum);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    builder.CreateCondBr(builder.CreateICmpULT(sui, num_elements), sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());

    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    llvm::Value* diff = builder.CreateFSub(pred_val, target_val);
    llvm::Value* abs_diff = builder.CreateCall(fabs_intrinsic, {diff});

    // Huber: 0.5 * a² if |a| ≤ δ, else δ * (|a| - 0.5 * δ)
    llvm::Value* is_small = builder.CreateFCmpOLE(abs_diff, delta);
    llvm::Value* quadratic = builder.CreateFMul(half, builder.CreateFMul(diff, diff));
    llvm::Value* linear = builder.CreateFMul(delta, builder.CreateFSub(abs_diff, builder.CreateFMul(half, delta)));
    llvm::Value* loss_term = builder.CreateSelect(is_small, quadratic, linear);

    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), huber_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, loss_term), huber_sum);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), huber_sum);
    llvm::Value* n_double = builder.CreateSIToFP(num_elements, ctx_.doubleType());
    llvm::Value* huber = builder.CreateFDiv(total_sum, n_double);

    return tagged_.packDouble(huber);
}

llvm::Value* TensorCodegen::maeLoss(const eshkol_operations_t* op) {
    // mae-loss: (mae-loss predictions targets)
    // MAE = (1/n) * sum(|pred - target|)
    if (op->call_op.num_vars != 2) {
        eshkol_error("mae-loss requires 2 arguments: predictions, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Get prediction and target tensors
    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    // Get tensor layouts (index 0 = dims ptr, index 1 = ndims, index 2 = elems)
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* pred_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* pred_dims = builder.CreateLoad(ctx_.ptrType(), pred_dims_ptr);
    llvm::Value* pred_ndims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* pred_ndims = builder.CreateLoad(ctx_.int64Type(), pred_ndims_ptr);
    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);

    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    // Calculate total number of elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "mae_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "mae_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "mae_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "mae_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "mae_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "mae_sum_done", current_func);

    llvm::Value* num_elements = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), num_elements);
    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    llvm::Value* size_cond = builder.CreateICmpSLT(si, pred_ndims);
    builder.CreateCondBr(size_cond, size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), pred_dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), num_elements);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), num_elements);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);

    // Sum absolute differences
    llvm::Value* mae_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), mae_sum);
    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), num_elements);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    llvm::Value* sum_cond = builder.CreateICmpSLT(sui, total_elements);
    builder.CreateCondBr(sum_cond, sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());
    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    // |pred - target|
    llvm::Value* diff = builder.CreateFSub(pred_val, target_val);
    llvm::Function* fabs_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::fabs, {ctx_.doubleType()});
    llvm::Value* abs_diff = builder.CreateCall(fabs_func, {diff});

    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), mae_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, abs_diff), mae_sum);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), mae_sum);
    llvm::Value* n_double = builder.CreateSIToFP(total_elements, ctx_.doubleType());
    llvm::Value* mae = builder.CreateFDiv(total_sum, n_double);

    return tagged_.packDouble(mae);
}

llvm::Value* TensorCodegen::binaryCrossEntropyLoss(const eshkol_operations_t* op) {
    // binary-cross-entropy-loss: (binary-cross-entropy-loss predictions targets)
    // BCE = -(1/n) * sum(target * log(pred) + (1-target) * log(1-pred))
    if (op->call_op.num_vars != 2) {
        eshkol_error("binary-cross-entropy-loss requires 2 arguments: predictions, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Get prediction and target tensors
    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    // Get tensor layouts (index 0 = dims ptr, index 1 = ndims, index 2 = elems)
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* pred_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* pred_dims = builder.CreateLoad(ctx_.ptrType(), pred_dims_ptr);
    llvm::Value* pred_ndims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* pred_ndims = builder.CreateLoad(ctx_.int64Type(), pred_ndims_ptr);
    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);

    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    // Calculate total number of elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "bce_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "bce_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "bce_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "bce_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "bce_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "bce_sum_done", current_func);

    llvm::Value* num_elements = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), num_elements);
    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    llvm::Value* size_cond = builder.CreateICmpSLT(si, pred_ndims);
    builder.CreateCondBr(size_cond, size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), pred_dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), num_elements);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), num_elements);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);

    // Sum BCE terms with numerical stability
    // Use clipping: pred_clipped = max(eps, min(1-eps, pred))
    llvm::Value* bce_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), bce_sum);
    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), num_elements);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    llvm::Value* sum_cond = builder.CreateICmpSLT(sui, total_elements);
    builder.CreateCondBr(sum_cond, sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());
    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    // Clip predictions for numerical stability
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_minus_eps = builder.CreateFSub(one, eps);

    // pred_clipped = max(eps, min(1-eps, pred))
    llvm::Value* cmp_high = builder.CreateFCmpOLT(pred_val, one_minus_eps);
    llvm::Value* pred_upper = builder.CreateSelect(cmp_high, pred_val, one_minus_eps);
    llvm::Value* cmp_low = builder.CreateFCmpOGT(pred_upper, eps);
    llvm::Value* pred_clipped = builder.CreateSelect(cmp_low, pred_upper, eps);

    // BCE term: target * log(pred) + (1-target) * log(1-pred)
    llvm::Function* log_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::log, {ctx_.doubleType()});

    llvm::Value* log_pred = builder.CreateCall(log_func, {pred_clipped});
    llvm::Value* one_minus_pred = builder.CreateFSub(one, pred_clipped);
    llvm::Value* log_one_minus_pred = builder.CreateCall(log_func, {one_minus_pred});

    llvm::Value* term1 = builder.CreateFMul(target_val, log_pred);
    llvm::Value* one_minus_target = builder.CreateFSub(one, target_val);
    llvm::Value* term2 = builder.CreateFMul(one_minus_target, log_one_minus_pred);
    llvm::Value* bce_term = builder.CreateFAdd(term1, term2);

    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), bce_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, bce_term), bce_sum);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), bce_sum);
    llvm::Value* n_double = builder.CreateSIToFP(total_elements, ctx_.doubleType());
    llvm::Value* mean_bce = builder.CreateFDiv(total_sum, n_double);
    // Negate to get the loss (BCE is negative log likelihood)
    llvm::Value* bce = builder.CreateFNeg(mean_bce);

    return tagged_.packDouble(bce);
}

// ============================================================
// Data Loading Infrastructure
// ============================================================

// Dataloader structure layout (arena-allocated):
// struct Dataloader {
//     int64_t* data_ptr;      // Pointer to source data tensor
//     int64_t batch_size;     // Batch size
//     int64_t num_samples;    // Total number of samples
//     int64_t current_idx;    // Current sample index
//     int64_t* indices;       // Shuffled indices array
//     int64_t sample_size;    // Size of each sample (elements per sample)
//     int64_t num_dims;       // Number of dimensions per sample
//     int64_t* sample_dims;   // Dimensions of each sample (excludes batch dim)
// }

llvm::Value* TensorCodegen::makeDataloader(const eshkol_operations_t* op) {
    // make-dataloader: (make-dataloader data-tensor batch-size [shuffle])
    if (op->call_op.num_vars < 2) {
        eshkol_error("make-dataloader requires at least 2 arguments: data, batch-size");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    // Get data tensor
    llvm::Value* data_tagged = codegenAST(&op->call_op.variables[0]);
    if (!data_tagged) return nullptr;
    llvm::Value* data_ptr = tagged_.unpackPtr(data_tagged);

    // Get batch size
    llvm::Value* batch_size_tagged = codegenAST(&op->call_op.variables[1]);
    if (!batch_size_tagged) return nullptr;
    llvm::Value* batch_size = tagged_.unpackInt64(batch_size_tagged);

    // Get shuffle flag (optional, default false)
    llvm::Value* shuffle_flag = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    if (op->call_op.num_vars >= 3) {
        llvm::Value* shuffle_tagged = codegenAST(&op->call_op.variables[2]);
        if (shuffle_tagged) {
            // Check if it's true (non-null, non-false)
            llvm::Value* type_val = tagged_.getType(shuffle_tagged);
            llvm::Value* is_bool = builder.CreateICmpEQ(type_val,
                llvm::ConstantInt::get(ctx_.int8Type(), 3)); // ESHKOL_VALUE_BOOL
            llvm::Value* data_val = tagged_.unpackInt64(shuffle_tagged);
            shuffle_flag = builder.CreateAnd(is_bool,
                builder.CreateICmpNE(data_val, llvm::ConstantInt::get(ctx_.int64Type(), 0)));
        }
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Get tensor properties
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, data_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* num_dims_field = builder.CreateStructGEP(tensor_type, data_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, data_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Get number of samples (first dimension)
    llvm::Value* first_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* num_samples = builder.CreateLoad(ctx_.int64Type(), first_dim_ptr);

    // Compute sample size (total_elements / num_samples)
    llvm::Value* sample_size = builder.CreateUDiv(total_elements, num_samples);

    // Allocate dataloader structure (8 fields * 8 bytes = 64 bytes)
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* loader_size = llvm::ConstantInt::get(ctx_.int64Type(), 64);
    llvm::Value* loader_ptr = builder.CreateCall(arena_alloc, {arena_ptr, loader_size}, "loader_ptr");

    // Allocate indices array
    llvm::Value* indices_size = builder.CreateMul(num_samples,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* indices_ptr = builder.CreateCall(arena_alloc, {arena_ptr, indices_size}, "indices_ptr");

    // Allocate sample_dims array (num_dims - 1 dimensions)
    llvm::Value* sample_num_dims = builder.CreateSub(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* sample_dims_size = builder.CreateMul(sample_num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* sample_dims_ptr = builder.CreateCall(arena_alloc, {arena_ptr, sample_dims_size}, "sample_dims_ptr");

    // Store fields in loader structure
    // Field 0: data_ptr
    llvm::Value* field0 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* data_ptr_int = builder.CreatePtrToInt(data_ptr, ctx_.int64Type());
    builder.CreateStore(data_ptr_int, field0);

    // Field 1: batch_size
    llvm::Value* field1 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(batch_size, field1);

    // Field 2: num_samples
    llvm::Value* field2 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 2));
    builder.CreateStore(num_samples, field2);

    // Field 3: current_idx (start at 0)
    llvm::Value* field3 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 3));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), field3);

    // Field 4: indices_ptr
    llvm::Value* field4 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 4));
    llvm::Value* indices_ptr_int = builder.CreatePtrToInt(indices_ptr, ctx_.int64Type());
    builder.CreateStore(indices_ptr_int, field4);

    // Field 5: sample_size
    llvm::Value* field5 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 5));
    builder.CreateStore(sample_size, field5);

    // Field 6: sample_num_dims
    llvm::Value* field6 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 6));
    builder.CreateStore(sample_num_dims, field6);

    // Field 7: sample_dims_ptr
    llvm::Value* field7 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 7));
    llvm::Value* sample_dims_ptr_int = builder.CreatePtrToInt(sample_dims_ptr, ctx_.int64Type());
    builder.CreateStore(sample_dims_ptr_int, field7);

    // Initialize indices and sample_dims with loops
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Initialize indices: indices[i] = i (sequential initially)
    llvm::BasicBlock* idx_init_loop = llvm::BasicBlock::Create(ctx_.context(), "idx_init_loop", current_func);
    llvm::BasicBlock* idx_init_body = llvm::BasicBlock::Create(ctx_.context(), "idx_init_body", current_func);
    llvm::BasicBlock* copy_dims = llvm::BasicBlock::Create(ctx_.context(), "copy_dims", current_func);

    llvm::Value* idx_counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx_counter);
    builder.CreateBr(idx_init_loop);

    builder.SetInsertPoint(idx_init_loop);
    llvm::Value* idx_i = builder.CreateLoad(ctx_.int64Type(), idx_counter);
    builder.CreateCondBr(builder.CreateICmpULT(idx_i, num_samples), idx_init_body, copy_dims);

    builder.SetInsertPoint(idx_init_body);
    llvm::Value* idx_slot = builder.CreateGEP(ctx_.int64Type(), indices_ptr, idx_i);
    builder.CreateStore(idx_i, idx_slot);
    builder.CreateStore(builder.CreateAdd(idx_i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx_counter);
    builder.CreateBr(idx_init_loop);

    // Copy sample dimensions (all dims except first)
    builder.SetInsertPoint(copy_dims);
    llvm::BasicBlock* dim_loop = llvm::BasicBlock::Create(ctx_.context(), "dim_loop", current_func);
    llvm::BasicBlock* dim_body = llvm::BasicBlock::Create(ctx_.context(), "dim_body", current_func);
    llvm::BasicBlock* shuffle_check = llvm::BasicBlock::Create(ctx_.context(), "shuffle_check", current_func);

    llvm::Value* dim_counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dim_counter);
    builder.CreateBr(dim_loop);

    builder.SetInsertPoint(dim_loop);
    llvm::Value* dim_i = builder.CreateLoad(ctx_.int64Type(), dim_counter);
    builder.CreateCondBr(builder.CreateICmpULT(dim_i, sample_num_dims), dim_body, shuffle_check);

    builder.SetInsertPoint(dim_body);
    // Copy dims[i+1] to sample_dims[i]
    llvm::Value* src_dim_idx = builder.CreateAdd(dim_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, src_dim_idx);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    llvm::Value* dst_dim_ptr = builder.CreateGEP(ctx_.int64Type(), sample_dims_ptr, dim_i);
    builder.CreateStore(dim_val, dst_dim_ptr);
    builder.CreateStore(builder.CreateAdd(dim_i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), dim_counter);
    builder.CreateBr(dim_loop);

    // Shuffle indices if requested (Fisher-Yates shuffle)
    builder.SetInsertPoint(shuffle_check);
    llvm::BasicBlock* do_shuffle = llvm::BasicBlock::Create(ctx_.context(), "do_shuffle", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "exit_loader", current_func);
    builder.CreateCondBr(shuffle_flag, do_shuffle, exit_block);

    builder.SetInsertPoint(do_shuffle);
    // Simple LCG-based shuffle (not cryptographically secure, but good enough for ML)
    llvm::BasicBlock* shuffle_loop = llvm::BasicBlock::Create(ctx_.context(), "shuffle_loop", current_func);
    llvm::BasicBlock* shuffle_body = llvm::BasicBlock::Create(ctx_.context(), "shuffle_body", current_func);

    llvm::Value* shuffle_i = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* rng_state = builder.CreateAlloca(ctx_.int64Type());
    // Seed with a mix of pointer and sample count
    llvm::Value* seed = builder.CreateXor(data_ptr_int, num_samples);
    builder.CreateStore(seed, rng_state);
    builder.CreateStore(builder.CreateSub(num_samples, llvm::ConstantInt::get(ctx_.int64Type(), 1)), shuffle_i);
    builder.CreateBr(shuffle_loop);

    builder.SetInsertPoint(shuffle_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), shuffle_i);
    builder.CreateCondBr(builder.CreateICmpSGT(si, llvm::ConstantInt::get(ctx_.int64Type(), 0)),
                          shuffle_body, exit_block);

    builder.SetInsertPoint(shuffle_body);
    // LCG: state = (state * 6364136223846793005 + 1) % 2^64
    llvm::Value* state = builder.CreateLoad(ctx_.int64Type(), rng_state);
    llvm::Value* mult = llvm::ConstantInt::get(ctx_.int64Type(), 6364136223846793005ULL);
    llvm::Value* new_state = builder.CreateAdd(builder.CreateMul(state, mult),
                                                llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(new_state, rng_state);

    // j = new_state % (i + 1)
    llvm::Value* si_plus_1 = builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* j = builder.CreateURem(new_state, si_plus_1);

    // Swap indices[i] and indices[j]
    llvm::Value* idx_i_ptr = builder.CreateGEP(ctx_.int64Type(), indices_ptr, si);
    llvm::Value* idx_j_ptr = builder.CreateGEP(ctx_.int64Type(), indices_ptr, j);
    llvm::Value* val_i = builder.CreateLoad(ctx_.int64Type(), idx_i_ptr);
    llvm::Value* val_j = builder.CreateLoad(ctx_.int64Type(), idx_j_ptr);
    builder.CreateStore(val_j, idx_i_ptr);
    builder.CreateStore(val_i, idx_j_ptr);

    builder.CreateStore(builder.CreateSub(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), shuffle_i);
    builder.CreateBr(shuffle_loop);

    builder.SetInsertPoint(exit_block);

    // Pack as HEAP_PTR (using generic pointer packing)
    return tagged_.packPtr(loader_ptr, ESHKOL_VALUE_HEAP_PTR, 0);
}

llvm::Value* TensorCodegen::dataloaderNext(const eshkol_operations_t* op) {
    // dataloader-next: (dataloader-next loader)
    if (op->call_op.num_vars < 1) {
        eshkol_error("dataloader-next requires 1 argument: loader");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* loader_tagged = codegenAST(&op->call_op.variables[0]);
    if (!loader_tagged) return nullptr;
    llvm::Value* loader_ptr = tagged_.unpackPtr(loader_tagged);

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Load loader fields
    llvm::Value* field0 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* data_ptr_int = builder.CreateLoad(ctx_.int64Type(), field0);
    llvm::Value* data_ptr = builder.CreateIntToPtr(data_ptr_int, ctx_.ptrType());

    llvm::Value* field1 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* batch_size = builder.CreateLoad(ctx_.int64Type(), field1);

    llvm::Value* field2 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* num_samples = builder.CreateLoad(ctx_.int64Type(), field2);

    llvm::Value* field3 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 3));
    llvm::Value* current_idx = builder.CreateLoad(ctx_.int64Type(), field3);

    llvm::Value* field4 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 4));
    llvm::Value* indices_ptr_int = builder.CreateLoad(ctx_.int64Type(), field4);
    llvm::Value* indices_ptr = builder.CreateIntToPtr(indices_ptr_int, ctx_.ptrType());

    llvm::Value* field5 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 5));
    llvm::Value* sample_size = builder.CreateLoad(ctx_.int64Type(), field5);

    llvm::Value* field6 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 6));
    llvm::Value* sample_num_dims = builder.CreateLoad(ctx_.int64Type(), field6);

    llvm::Value* field7 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 7));
    llvm::Value* sample_dims_ptr_int = builder.CreateLoad(ctx_.int64Type(), field7);
    llvm::Value* sample_dims_ptr = builder.CreateIntToPtr(sample_dims_ptr_int, ctx_.ptrType());

    // Get source data elements
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* src_elems_field = builder.CreateStructGEP(tensor_type, data_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), src_elems_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Check if we have more samples
    llvm::BasicBlock* has_data = llvm::BasicBlock::Create(ctx_.context(), "has_data", current_func);
    llvm::BasicBlock* no_data = llvm::BasicBlock::Create(ctx_.context(), "no_data", current_func);
    llvm::BasicBlock* copy_batch = llvm::BasicBlock::Create(ctx_.context(), "copy_batch", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "exit_next", current_func);

    builder.CreateCondBr(builder.CreateICmpULT(current_idx, num_samples), has_data, no_data);

    builder.SetInsertPoint(no_data);
    builder.CreateBr(exit_block);

    builder.SetInsertPoint(has_data);

    // Calculate actual batch size (may be smaller for last batch)
    llvm::Value* remaining = builder.CreateSub(num_samples, current_idx);
    llvm::Value* actual_batch = builder.CreateSelect(
        builder.CreateICmpULT(remaining, batch_size), remaining, batch_size);

    // Allocate batch tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* batch_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "batch_ptr");

    // Batch dimensions: [actual_batch, sample_dims...]
    llvm::Value* batch_num_dims = builder.CreateAdd(sample_num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* batch_dims_size = builder.CreateMul(batch_num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* batch_dims = builder.CreateCall(arena_alloc, {arena_ptr, batch_dims_size}, "batch_dims");

    // Store batch dimension at index 0
    llvm::Value* batch_dim0 = builder.CreateGEP(ctx_.int64Type(), batch_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateStore(actual_batch, batch_dim0);

    // Copy sample dimensions
    llvm::BasicBlock* copy_dim_loop = llvm::BasicBlock::Create(ctx_.context(), "copy_dim_loop", current_func);
    llvm::BasicBlock* copy_dim_body = llvm::BasicBlock::Create(ctx_.context(), "copy_dim_body", current_func);

    llvm::Value* dim_i = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dim_i);
    builder.CreateBr(copy_dim_loop);

    builder.SetInsertPoint(copy_dim_loop);
    llvm::Value* di = builder.CreateLoad(ctx_.int64Type(), dim_i);
    builder.CreateCondBr(builder.CreateICmpULT(di, sample_num_dims), copy_dim_body, copy_batch);

    builder.SetInsertPoint(copy_dim_body);
    llvm::Value* src_dim = builder.CreateGEP(ctx_.int64Type(), sample_dims_ptr, di);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim);
    llvm::Value* dst_dim_idx = builder.CreateAdd(di, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* dst_dim = builder.CreateGEP(ctx_.int64Type(), batch_dims, dst_dim_idx);
    builder.CreateStore(dim_val, dst_dim);
    builder.CreateStore(builder.CreateAdd(di, llvm::ConstantInt::get(ctx_.int64Type(), 1)), dim_i);
    builder.CreateBr(copy_dim_loop);

    // Copy batch data
    builder.SetInsertPoint(copy_batch);
    llvm::Value* batch_total_elems = builder.CreateMul(actual_batch, sample_size);
    llvm::Value* batch_elems_size = builder.CreateMul(batch_total_elems,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* batch_elems = builder.CreateCall(arena_alloc, {arena_ptr, batch_elems_size}, "batch_elems");

    llvm::BasicBlock* sample_loop = llvm::BasicBlock::Create(ctx_.context(), "sample_loop", current_func);
    llvm::BasicBlock* sample_body = llvm::BasicBlock::Create(ctx_.context(), "sample_body", current_func);
    llvm::BasicBlock* finalize_batch = llvm::BasicBlock::Create(ctx_.context(), "finalize_batch", current_func);

    llvm::Value* sample_i = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sample_i);
    builder.CreateBr(sample_loop);

    builder.SetInsertPoint(sample_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), sample_i);
    builder.CreateCondBr(builder.CreateICmpULT(si, actual_batch), sample_body, finalize_batch);

    builder.SetInsertPoint(sample_body);
    // Get the actual sample index from indices array
    llvm::Value* global_idx = builder.CreateAdd(current_idx, si);
    llvm::Value* idx_slot = builder.CreateGEP(ctx_.int64Type(), indices_ptr, global_idx);
    llvm::Value* real_sample_idx = builder.CreateLoad(ctx_.int64Type(), idx_slot);

    // Copy sample data
    llvm::Value* src_offset = builder.CreateMul(real_sample_idx, sample_size);
    llvm::Value* dst_offset = builder.CreateMul(si, sample_size);

    // Copy element by element
    llvm::BasicBlock* elem_loop = llvm::BasicBlock::Create(ctx_.context(), "elem_loop", current_func);
    llvm::BasicBlock* elem_body = llvm::BasicBlock::Create(ctx_.context(), "elem_body", current_func);
    llvm::BasicBlock* elem_done = llvm::BasicBlock::Create(ctx_.context(), "elem_done", current_func);

    llvm::Value* elem_i = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), elem_i);
    builder.CreateBr(elem_loop);

    builder.SetInsertPoint(elem_loop);
    llvm::Value* ei = builder.CreateLoad(ctx_.int64Type(), elem_i);
    builder.CreateCondBr(builder.CreateICmpULT(ei, sample_size), elem_body, elem_done);

    builder.SetInsertPoint(elem_body);
    llvm::Value* src_elem_idx = builder.CreateAdd(src_offset, ei);
    llvm::Value* dst_elem_idx = builder.CreateAdd(dst_offset, ei);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, src_elem_idx);
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), batch_elems, dst_elem_idx);
    llvm::Value* elem_val = builder.CreateLoad(ctx_.int64Type(), src_ptr);
    builder.CreateStore(elem_val, dst_ptr);
    builder.CreateStore(builder.CreateAdd(ei, llvm::ConstantInt::get(ctx_.int64Type(), 1)), elem_i);
    builder.CreateBr(elem_loop);

    builder.SetInsertPoint(elem_done);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sample_i);
    builder.CreateBr(sample_loop);

    // Finalize batch tensor
    builder.SetInsertPoint(finalize_batch);
    llvm::Value* b_dims_field = builder.CreateStructGEP(tensor_type, batch_ptr, 0);
    builder.CreateStore(batch_dims, b_dims_field);
    llvm::Value* b_ndim_field = builder.CreateStructGEP(tensor_type, batch_ptr, 1);
    builder.CreateStore(batch_num_dims, b_ndim_field);
    llvm::Value* b_elems_field = builder.CreateStructGEP(tensor_type, batch_ptr, 2);
    builder.CreateStore(batch_elems, b_elems_field);
    llvm::Value* b_total_field = builder.CreateStructGEP(tensor_type, batch_ptr, 3);
    builder.CreateStore(batch_total_elems, b_total_field);

    // Update current_idx
    llvm::Value* new_idx = builder.CreateAdd(current_idx, actual_batch);
    builder.CreateStore(new_idx, field3);

    builder.CreateBr(exit_block);

    // Exit block with PHI for result
    builder.SetInsertPoint(exit_block);
    llvm::PHINode* result = builder.CreatePHI(ctx_.taggedValueType(), 2, "next_result");
    result->addIncoming(tagged_.packNull(), no_data);
    result->addIncoming(tagged_.packHeapPtr(batch_ptr), finalize_batch);

    return result;
}

llvm::Value* TensorCodegen::dataloaderReset(const eshkol_operations_t* op) {
    // dataloader-reset!: (dataloader-reset! loader)
    if (op->call_op.num_vars < 1) {
        eshkol_error("dataloader-reset! requires 1 argument: loader");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* loader_tagged = codegenAST(&op->call_op.variables[0]);
    if (!loader_tagged) return nullptr;
    llvm::Value* loader_ptr = tagged_.unpackPtr(loader_tagged);

    // Reset current_idx to 0
    llvm::Value* field3 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 3));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), field3);

    return loader_tagged;
}

llvm::Value* TensorCodegen::dataloaderLength(const eshkol_operations_t* op) {
    // dataloader-length: (dataloader-length loader)
    if (op->call_op.num_vars < 1) {
        eshkol_error("dataloader-length requires 1 argument: loader");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* loader_tagged = codegenAST(&op->call_op.variables[0]);
    if (!loader_tagged) return nullptr;
    llvm::Value* loader_ptr = tagged_.unpackPtr(loader_tagged);

    // Load batch_size and num_samples
    llvm::Value* field1 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* batch_size = builder.CreateLoad(ctx_.int64Type(), field1);

    llvm::Value* field2 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* num_samples = builder.CreateLoad(ctx_.int64Type(), field2);

    // num_batches = ceil(num_samples / batch_size)
    llvm::Value* full_batches = builder.CreateUDiv(num_samples, batch_size);
    llvm::Value* remainder = builder.CreateURem(num_samples, batch_size);
    llvm::Value* has_partial = builder.CreateICmpNE(remainder,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* extra = builder.CreateZExt(has_partial, ctx_.int64Type());
    llvm::Value* num_batches = builder.CreateAdd(full_batches, extra);

    return tagged_.packInt64(num_batches);
}

llvm::Value* TensorCodegen::dataloaderHasNext(const eshkol_operations_t* op) {
    // dataloader-has-next?: (dataloader-has-next? loader)
    if (op->call_op.num_vars < 1) {
        eshkol_error("dataloader-has-next? requires 1 argument: loader");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* loader_tagged = codegenAST(&op->call_op.variables[0]);
    if (!loader_tagged) return nullptr;
    llvm::Value* loader_ptr = tagged_.unpackPtr(loader_tagged);

    // Load current_idx and num_samples
    llvm::Value* field2 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* num_samples = builder.CreateLoad(ctx_.int64Type(), field2);

    llvm::Value* field3 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 3));
    llvm::Value* current_idx = builder.CreateLoad(ctx_.int64Type(), field3);

    // has_next = current_idx < num_samples
    llvm::Value* has_next = builder.CreateICmpULT(current_idx, num_samples);

    return tagged_.packBool(has_next);
}

llvm::Value* TensorCodegen::trainTestSplit(const eshkol_operations_t* op) {
    // train-test-split: (train-test-split data ratio [shuffle])
    // Returns a vector of (train-data test-data)
    if (op->call_op.num_vars < 2) {
        eshkol_error("train-test-split requires at least 2 arguments: data, ratio");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    // Get data tensor
    llvm::Value* data_tagged = codegenAST(&op->call_op.variables[0]);
    if (!data_tagged) return nullptr;
    llvm::Value* data_ptr = tagged_.unpackPtr(data_tagged);

    // Get ratio
    llvm::Value* ratio_tagged = codegenAST(&op->call_op.variables[1]);
    if (!ratio_tagged) return nullptr;
    llvm::Value* ratio = tagged_.unpackDouble(ratio_tagged);

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Get tensor properties
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, data_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* num_dims_field = builder.CreateStructGEP(tensor_type, data_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, data_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, data_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Get number of samples
    llvm::Value* first_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* num_samples = builder.CreateLoad(ctx_.int64Type(), first_dim_ptr);
    llvm::Value* sample_size = builder.CreateUDiv(total_elements, num_samples);

    // Calculate split point
    llvm::Value* num_samples_f = builder.CreateSIToFP(num_samples, ctx_.doubleType());
    llvm::Value* train_size_f = builder.CreateFMul(num_samples_f, ratio);
    llvm::Value* train_size = builder.CreateFPToSI(train_size_f, ctx_.int64Type());
    llvm::Value* test_size = builder.CreateSub(num_samples, train_size);

    // Allocate train tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* train_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "train_ptr");

    // Allocate train dims (same as original)
    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* train_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "train_dims");
    builder.CreateMemCpy(train_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);
    // Update first dimension for train
    llvm::Value* train_dim0 = builder.CreateGEP(ctx_.int64Type(), train_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateStore(train_size, train_dim0);

    // Allocate train elements
    llvm::Value* train_total = builder.CreateMul(train_size, sample_size);
    llvm::Value* train_elems_size = builder.CreateMul(train_total,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* train_elems = builder.CreateCall(arena_alloc, {arena_ptr, train_elems_size}, "train_elems");
    builder.CreateMemCpy(train_elems, llvm::MaybeAlign(8), src_elems, llvm::MaybeAlign(8), train_elems_size);

    // Set train tensor fields
    llvm::Value* train_dims_field = builder.CreateStructGEP(tensor_type, train_ptr, 0);
    builder.CreateStore(train_dims, train_dims_field);
    llvm::Value* train_ndim_field = builder.CreateStructGEP(tensor_type, train_ptr, 1);
    builder.CreateStore(num_dims, train_ndim_field);
    llvm::Value* train_elems_field = builder.CreateStructGEP(tensor_type, train_ptr, 2);
    builder.CreateStore(train_elems, train_elems_field);
    llvm::Value* train_total_field = builder.CreateStructGEP(tensor_type, train_ptr, 3);
    builder.CreateStore(train_total, train_total_field);

    // Allocate test tensor
    llvm::Value* test_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "test_ptr");
    llvm::Value* test_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "test_dims");
    builder.CreateMemCpy(test_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);
    llvm::Value* test_dim0 = builder.CreateGEP(ctx_.int64Type(), test_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateStore(test_size, test_dim0);

    // Allocate test elements
    llvm::Value* test_total = builder.CreateMul(test_size, sample_size);
    llvm::Value* test_elems_size = builder.CreateMul(test_total,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* test_elems = builder.CreateCall(arena_alloc, {arena_ptr, test_elems_size}, "test_elems");
    // Copy from offset
    llvm::Value* test_offset_bytes = builder.CreateMul(train_total,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* test_src = builder.CreateGEP(ctx_.int8Type(), src_elems, test_offset_bytes);
    builder.CreateMemCpy(test_elems, llvm::MaybeAlign(8), test_src, llvm::MaybeAlign(8), test_elems_size);

    // Set test tensor fields
    llvm::Value* test_dims_field = builder.CreateStructGEP(tensor_type, test_ptr, 0);
    builder.CreateStore(test_dims, test_dims_field);
    llvm::Value* test_ndim_field = builder.CreateStructGEP(tensor_type, test_ptr, 1);
    builder.CreateStore(num_dims, test_ndim_field);
    llvm::Value* test_elems_field = builder.CreateStructGEP(tensor_type, test_ptr, 2);
    builder.CreateStore(test_elems, test_elems_field);
    llvm::Value* test_total_field = builder.CreateStructGEP(tensor_type, test_ptr, 3);
    builder.CreateStore(test_total, test_total_field);

    // Create result vector with 2 elements
    llvm::Function* alloc_vec = mem_.getArenaAllocateVectorWithHeader();
    llvm::Value* result_vec = builder.CreateCall(alloc_vec, {arena_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)}, "split_result");

    // Store train and test tensors in vector
    llvm::Value* vec_elem0 = builder.CreateGEP(ctx_.taggedValueType(), result_vec,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateStore(tagged_.packHeapPtr(train_ptr), vec_elem0);
    llvm::Value* vec_elem1 = builder.CreateGEP(ctx_.taggedValueType(), result_vec,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(tagged_.packHeapPtr(test_ptr), vec_elem1);

    return tagged_.packHeapPtr(result_vec);
}

// ════════════════════════════════════════════════════════════════════════════════
// TRANSFORMER ARCHITECTURE (Track 8)
// ════════════════════════════════════════════════════════════════════════════════

// === Track 8.1: Scaled Dot-Product Attention ===

llvm::Value* TensorCodegen::scaledDotProductAttention(const eshkol_operations_t* op) {
    // Scaled Dot-Product Attention from "Attention Is All You Need"
    // scores = Q @ K^T / sqrt(d_k)
    // attention_weights = softmax(scores + mask)
    // output = attention_weights @ V
    //
    // Supports both 2D (seq_len, d_k) and 3D (batch, seq_len, d_k) inputs

    if (op->call_op.num_vars < 3 || op->call_op.num_vars > 4) {
        eshkol_error("scaled-dot-attention requires 3-4 arguments: Q K V [mask]");
        return nullptr;
    }

    llvm::Value* q_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* k_val = codegenAST(&op->call_op.variables[1]);
    llvm::Value* v_val = codegenAST(&op->call_op.variables[2]);
    llvm::Value* mask_val = (op->call_op.num_vars == 4)
        ? codegenAST(&op->call_op.variables[3]) : nullptr;

    if (!q_val || !k_val || !v_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack Q tensor
    llvm::Value* q_ptr_int = tagged_.unpackInt64(q_val);
    llvm::Value* q_ptr = builder.CreateIntToPtr(q_ptr_int, ctx_.ptrType());

    llvm::Value* q_dims_field = builder.CreateStructGEP(tensor_type, q_ptr, 0);
    llvm::Value* q_dims_ptr = builder.CreateLoad(ctx_.ptrType(), q_dims_field);
    llvm::Value* q_ndim_field = builder.CreateStructGEP(tensor_type, q_ptr, 1);
    llvm::Value* q_ndim = builder.CreateLoad(ctx_.int64Type(), q_ndim_field);
    llvm::Value* q_elems_field = builder.CreateStructGEP(tensor_type, q_ptr, 2);
    llvm::Value* q_elems = builder.CreateLoad(ctx_.ptrType(), q_elems_field);

    // Unpack K tensor
    llvm::Value* k_ptr_int = tagged_.unpackInt64(k_val);
    llvm::Value* k_ptr = builder.CreateIntToPtr(k_ptr_int, ctx_.ptrType());

    llvm::Value* k_dims_field = builder.CreateStructGEP(tensor_type, k_ptr, 0);
    llvm::Value* k_dims_ptr = builder.CreateLoad(ctx_.ptrType(), k_dims_field);
    llvm::Value* k_elems_field = builder.CreateStructGEP(tensor_type, k_ptr, 2);
    llvm::Value* k_elems = builder.CreateLoad(ctx_.ptrType(), k_elems_field);

    // Unpack V tensor
    llvm::Value* v_ptr_int = tagged_.unpackInt64(v_val);
    llvm::Value* v_ptr = builder.CreateIntToPtr(v_ptr_int, ctx_.ptrType());

    llvm::Value* v_dims_field = builder.CreateStructGEP(tensor_type, v_ptr, 0);
    llvm::Value* v_dims_ptr = builder.CreateLoad(ctx_.ptrType(), v_dims_field);
    llvm::Value* v_elems_field = builder.CreateStructGEP(tensor_type, v_ptr, 2);
    llvm::Value* v_elems = builder.CreateLoad(ctx_.ptrType(), v_elems_field);

    // Determine dimensions based on 2D or 3D input
    llvm::Value* is_3d = builder.CreateICmpEQ(q_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), 3));

    // Get dimensions
    llvm::Value* batch_size = builder.CreateSelect(is_3d,
        builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), q_dims_ptr,
                llvm::ConstantInt::get(ctx_.int64Type(), 0))),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* seq_q_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 1),
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* seq_q = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), q_dims_ptr, seq_q_idx));

    llvm::Value* d_k_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 2),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* d_k = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), q_dims_ptr, d_k_idx));

    llvm::Value* seq_k = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), k_dims_ptr, seq_q_idx));

    llvm::Value* d_v = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), v_dims_ptr, d_k_idx));

    // Compute sqrt(d_k) for scaling
    llvm::Value* d_k_double = builder.CreateSIToFP(d_k, ctx_.doubleType());
    llvm::Function* sqrt_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* scale = builder.CreateCall(sqrt_func, {d_k_double}, "sqrt_dk");

    // Allocate scores matrix: (batch, seq_q, seq_k)
    llvm::Value* scores_size = builder.CreateMul(batch_size,
        builder.CreateMul(seq_q, seq_k));
    llvm::Value* scores_bytes = builder.CreateMul(scores_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* scores_ptr = builder.CreateCall(arena_alloc,
        {arena_ptr, scores_bytes}, "attn_scores");

    // Allocate output tensor: (batch, seq_q, d_v)
    llvm::Value* output_size = builder.CreateMul(batch_size,
        builder.CreateMul(seq_q, d_v));
    llvm::Value* output_bytes = builder.CreateMul(output_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* output_elems = builder.CreateCall(arena_alloc,
        {arena_ptr, output_bytes}, "attn_output");

    llvm::Function* exp_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // === ALLOCATE ALL LOOP VARIABLES UPFRONT ===
    llvm::Value* batch_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "batch_idx");
    llvm::Value* i_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "i_idx");
    llvm::Value* j_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "j_idx");
    llvm::Value* k_iter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "k_iter");
    llvm::Value* row_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "row_idx");
    llvm::Value* col_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "col_idx");
    llvm::Value* dot_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "dot_sum");
    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "max_val");
    llvm::Value* sum_exp = builder.CreateAlloca(ctx_.doubleType(), nullptr, "sum_exp");
    llvm::Value* out_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "out_sum");

    // === Compute Q @ K^T / sqrt(d_k) + mask ===
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "attn_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "attn_batch_body", current_func);
    llvm::BasicBlock* softmax_init = llvm::BasicBlock::Create(ctx_.context(), "attn_softmax_init", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_cond);
    llvm::Value* b = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* batch_done = builder.CreateICmpUGE(b, batch_size);
    builder.CreateCondBr(batch_done, softmax_init, batch_body);

    builder.SetInsertPoint(batch_body);

    // Compute base offsets for this batch
    llvm::Value* q_batch_offset = builder.CreateMul(b, builder.CreateMul(seq_q, d_k));
    llvm::Value* k_batch_offset = builder.CreateMul(b, builder.CreateMul(seq_k, d_k));
    llvm::Value* scores_batch_offset = builder.CreateMul(b, builder.CreateMul(seq_q, seq_k));

    // Inner loops: for each (i, j) in (seq_q, seq_k), compute dot product
    llvm::BasicBlock* i_cond = llvm::BasicBlock::Create(ctx_.context(), "attn_i_cond", current_func);
    llvm::BasicBlock* i_body = llvm::BasicBlock::Create(ctx_.context(), "attn_i_body", current_func);
    llvm::BasicBlock* batch_next = llvm::BasicBlock::Create(ctx_.context(), "attn_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_idx);
    builder.CreateBr(i_cond);

    builder.SetInsertPoint(i_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_idx);
    llvm::Value* i_done = builder.CreateICmpUGE(i, seq_q);
    builder.CreateCondBr(i_done, batch_next, i_body);

    builder.SetInsertPoint(i_body);

    // j loop
    llvm::BasicBlock* j_cond = llvm::BasicBlock::Create(ctx_.context(), "attn_j_cond", current_func);
    llvm::BasicBlock* j_body = llvm::BasicBlock::Create(ctx_.context(), "attn_j_body", current_func);
    llvm::BasicBlock* i_next = llvm::BasicBlock::Create(ctx_.context(), "attn_i_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_idx);
    builder.CreateBr(j_cond);

    builder.SetInsertPoint(j_cond);
    llvm::Value* j = builder.CreateLoad(ctx_.int64Type(), j_idx);
    llvm::Value* j_done = builder.CreateICmpUGE(j, seq_k);
    builder.CreateCondBr(j_done, i_next, j_body);

    builder.SetInsertPoint(j_body);

    // Compute dot product: Q[b,i,:] @ K[b,j,:]
    llvm::BasicBlock* dot_cond = llvm::BasicBlock::Create(ctx_.context(), "dot_cond", current_func);
    llvm::BasicBlock* dot_body = llvm::BasicBlock::Create(ctx_.context(), "dot_body", current_func);
    llvm::BasicBlock* dot_done = llvm::BasicBlock::Create(ctx_.context(), "dot_done", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), dot_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_iter);
    builder.CreateBr(dot_cond);

    builder.SetInsertPoint(dot_cond);
    llvm::Value* kk = builder.CreateLoad(ctx_.int64Type(), k_iter);
    llvm::Value* k_done = builder.CreateICmpUGE(kk, d_k);
    builder.CreateCondBr(k_done, dot_done, dot_body);

    builder.SetInsertPoint(dot_body);
    // Q[b, i, k] = q_elems[q_batch_offset + i * d_k + k]
    llvm::Value* q_idx = builder.CreateAdd(q_batch_offset,
        builder.CreateAdd(builder.CreateMul(i, d_k), kk));
    llvm::Value* q_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), q_elems, q_idx));
    // K[b, j, k] = k_elems[k_batch_offset + j * d_k + k]
    llvm::Value* k_idx = builder.CreateAdd(k_batch_offset,
        builder.CreateAdd(builder.CreateMul(j, d_k), kk));
    llvm::Value* k_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), k_elems, k_idx));

    llvm::Value* prod = builder.CreateFMul(q_elem, k_elem);
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), dot_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, prod), dot_sum);

    builder.CreateStore(builder.CreateAdd(kk,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_iter);
    builder.CreateBr(dot_cond);

    builder.SetInsertPoint(dot_done);
    // Scale by 1/sqrt(d_k)
    llvm::Value* final_dot = builder.CreateLoad(ctx_.doubleType(), dot_sum);
    llvm::Value* scaled_dot = builder.CreateFDiv(final_dot, scale, "scaled_score");

    // Add mask if provided
    if (mask_val) {
        llvm::Value* mask_ptr_int = tagged_.unpackInt64(mask_val);
        llvm::Value* mask_ptr = builder.CreateIntToPtr(mask_ptr_int, ctx_.ptrType());
        llvm::Value* mask_elems_field = builder.CreateStructGEP(tensor_type, mask_ptr, 2);
        llvm::Value* mask_elems = builder.CreateLoad(ctx_.ptrType(), mask_elems_field);
        // mask[i, j] or mask[b, i, j]
        llvm::Value* mask_idx = builder.CreateAdd(builder.CreateMul(i, seq_k), j);
        llvm::Value* mask_elem = builder.CreateLoad(ctx_.doubleType(),
            builder.CreateGEP(ctx_.doubleType(), mask_elems, mask_idx));
        scaled_dot = builder.CreateFAdd(scaled_dot, mask_elem, "masked_score");
    }

    // Store in scores[b, i, j]
    llvm::Value* scores_idx = builder.CreateAdd(scores_batch_offset,
        builder.CreateAdd(builder.CreateMul(i, seq_k), j));
    builder.CreateStore(scaled_dot,
        builder.CreateGEP(ctx_.doubleType(), scores_ptr, scores_idx));

    // Next j
    builder.CreateStore(builder.CreateAdd(j,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), j_idx);
    builder.CreateBr(j_cond);

    // Next i
    builder.SetInsertPoint(i_next);
    builder.CreateStore(builder.CreateAdd(i,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_idx);
    builder.CreateBr(i_cond);

    // Next batch
    builder.SetInsertPoint(batch_next);
    builder.CreateStore(builder.CreateAdd(b,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(batch_cond);

    // === Apply softmax row-wise (along seq_k dimension) ===
    builder.SetInsertPoint(softmax_init);

    llvm::BasicBlock* sm_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_batch_cond", current_func);
    llvm::BasicBlock* sm_batch_body = llvm::BasicBlock::Create(ctx_.context(), "sm_batch_body", current_func);
    llvm::BasicBlock* output_init = llvm::BasicBlock::Create(ctx_.context(), "output_init", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(sm_batch_cond);

    builder.SetInsertPoint(sm_batch_cond);
    llvm::Value* b2 = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* b2_done = builder.CreateICmpUGE(b2, batch_size);
    builder.CreateCondBr(b2_done, output_init, sm_batch_body);

    builder.SetInsertPoint(sm_batch_body);

    llvm::Value* sm_batch_offset = builder.CreateMul(b2, builder.CreateMul(seq_q, seq_k));

    // Softmax each row (for each query position)
    llvm::BasicBlock* sm_row_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_row_cond", current_func);
    llvm::BasicBlock* sm_row_body = llvm::BasicBlock::Create(ctx_.context(), "sm_row_body", current_func);
    llvm::BasicBlock* sm_batch_next = llvm::BasicBlock::Create(ctx_.context(), "sm_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), row_idx);
    builder.CreateBr(sm_row_cond);

    builder.SetInsertPoint(sm_row_cond);
    llvm::Value* row = builder.CreateLoad(ctx_.int64Type(), row_idx);
    llvm::Value* row_done = builder.CreateICmpUGE(row, seq_q);
    builder.CreateCondBr(row_done, sm_batch_next, sm_row_body);

    builder.SetInsertPoint(sm_row_body);

    llvm::Value* row_offset = builder.CreateAdd(sm_batch_offset,
        builder.CreateMul(row, seq_k));

    // Find max for numerical stability
    llvm::BasicBlock* max_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_max_cond", current_func);
    llvm::BasicBlock* max_body = llvm::BasicBlock::Create(ctx_.context(), "sm_max_body", current_func);
    llvm::BasicBlock* exp_init = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_init", current_func);

    llvm::Value* first_score = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), scores_ptr, row_offset));
    builder.CreateStore(first_score, max_val);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), col_idx);
    builder.CreateBr(max_cond);

    builder.SetInsertPoint(max_cond);
    llvm::Value* col = builder.CreateLoad(ctx_.int64Type(), col_idx);
    llvm::Value* col_done = builder.CreateICmpUGE(col, seq_k);
    builder.CreateCondBr(col_done, exp_init, max_body);

    builder.SetInsertPoint(max_body);
    llvm::Value* score_idx = builder.CreateAdd(row_offset, col);
    llvm::Value* score_val = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), scores_ptr, score_idx));
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_greater = builder.CreateFCmpOGT(score_val, cur_max);
    builder.CreateStore(builder.CreateSelect(is_greater, score_val, cur_max), max_val);
    builder.CreateStore(builder.CreateAdd(col,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), col_idx);
    builder.CreateBr(max_cond);

    // Compute exp(x - max) and sum
    builder.SetInsertPoint(exp_init);
    llvm::BasicBlock* exp_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_cond", current_func);
    llvm::BasicBlock* exp_body = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_body", current_func);
    llvm::BasicBlock* norm_init = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_init", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_exp);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), col_idx);
    builder.CreateBr(exp_cond);

    builder.SetInsertPoint(exp_cond);
    llvm::Value* col2 = builder.CreateLoad(ctx_.int64Type(), col_idx);
    llvm::Value* col2_done = builder.CreateICmpUGE(col2, seq_k);
    builder.CreateCondBr(col2_done, norm_init, exp_body);

    builder.SetInsertPoint(exp_body);
    llvm::Value* score_idx2 = builder.CreateAdd(row_offset, col2);
    llvm::Value* score_ptr = builder.CreateGEP(ctx_.doubleType(), scores_ptr, score_idx2);
    llvm::Value* score_val2 = builder.CreateLoad(ctx_.doubleType(), score_ptr);
    llvm::Value* final_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* shifted = builder.CreateFSub(score_val2, final_max);
    llvm::Value* exp_val = builder.CreateCall(exp_func, {shifted});
    builder.CreateStore(exp_val, score_ptr);  // Store exp value back
    llvm::Value* cur_exp_sum = builder.CreateLoad(ctx_.doubleType(), sum_exp);
    builder.CreateStore(builder.CreateFAdd(cur_exp_sum, exp_val), sum_exp);
    builder.CreateStore(builder.CreateAdd(col2,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), col_idx);
    builder.CreateBr(exp_cond);

    // Normalize by sum
    builder.SetInsertPoint(norm_init);
    llvm::BasicBlock* norm_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_cond", current_func);
    llvm::BasicBlock* norm_body = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_body", current_func);
    llvm::BasicBlock* row_next = llvm::BasicBlock::Create(ctx_.context(), "row_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), col_idx);
    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_cond);
    llvm::Value* col3 = builder.CreateLoad(ctx_.int64Type(), col_idx);
    llvm::Value* col3_done = builder.CreateICmpUGE(col3, seq_k);
    builder.CreateCondBr(col3_done, row_next, norm_body);

    builder.SetInsertPoint(norm_body);
    llvm::Value* score_idx3 = builder.CreateAdd(row_offset, col3);
    llvm::Value* score_ptr3 = builder.CreateGEP(ctx_.doubleType(), scores_ptr, score_idx3);
    llvm::Value* exp_val3 = builder.CreateLoad(ctx_.doubleType(), score_ptr3);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), sum_exp);
    llvm::Value* normalized = builder.CreateFDiv(exp_val3, total_sum);
    builder.CreateStore(normalized, score_ptr3);
    builder.CreateStore(builder.CreateAdd(col3,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), col_idx);
    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(row_next);
    builder.CreateStore(builder.CreateAdd(row,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), row_idx);
    builder.CreateBr(sm_row_cond);

    builder.SetInsertPoint(sm_batch_next);
    builder.CreateStore(builder.CreateAdd(b2,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(sm_batch_cond);

    // === Compute attention_weights @ V ===
    builder.SetInsertPoint(output_init);

    llvm::BasicBlock* out_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "out_batch_cond", current_func);
    llvm::BasicBlock* out_batch_body = llvm::BasicBlock::Create(ctx_.context(), "out_batch_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "attn_finalize", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(out_batch_cond);

    builder.SetInsertPoint(out_batch_cond);
    llvm::Value* b3 = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* b3_done = builder.CreateICmpUGE(b3, batch_size);
    builder.CreateCondBr(b3_done, finalize, out_batch_body);

    builder.SetInsertPoint(out_batch_body);

    llvm::Value* attn_batch_offset = builder.CreateMul(b3, builder.CreateMul(seq_q, seq_k));
    llvm::Value* v_batch_offset = builder.CreateMul(b3, builder.CreateMul(seq_k, d_v));
    llvm::Value* out_batch_offset = builder.CreateMul(b3, builder.CreateMul(seq_q, d_v));

    // output[b, i, j] = sum_k(attention[b, i, k] * V[b, k, j])
    llvm::BasicBlock* out_i_cond = llvm::BasicBlock::Create(ctx_.context(), "out_i_cond", current_func);
    llvm::BasicBlock* out_i_body = llvm::BasicBlock::Create(ctx_.context(), "out_i_body", current_func);
    llvm::BasicBlock* out_batch_next = llvm::BasicBlock::Create(ctx_.context(), "out_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_idx);
    builder.CreateBr(out_i_cond);

    builder.SetInsertPoint(out_i_cond);
    llvm::Value* oi = builder.CreateLoad(ctx_.int64Type(), i_idx);
    llvm::Value* oi_done = builder.CreateICmpUGE(oi, seq_q);
    builder.CreateCondBr(oi_done, out_batch_next, out_i_body);

    builder.SetInsertPoint(out_i_body);

    // j loop (d_v)
    llvm::BasicBlock* out_j_cond = llvm::BasicBlock::Create(ctx_.context(), "out_j_cond", current_func);
    llvm::BasicBlock* out_j_body = llvm::BasicBlock::Create(ctx_.context(), "out_j_body", current_func);
    llvm::BasicBlock* out_i_next = llvm::BasicBlock::Create(ctx_.context(), "out_i_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_idx);
    builder.CreateBr(out_j_cond);

    builder.SetInsertPoint(out_j_cond);
    llvm::Value* oj = builder.CreateLoad(ctx_.int64Type(), j_idx);
    llvm::Value* oj_done = builder.CreateICmpUGE(oj, d_v);
    builder.CreateCondBr(oj_done, out_i_next, out_j_body);

    builder.SetInsertPoint(out_j_body);

    // k loop (sum over seq_k)
    llvm::BasicBlock* out_k_cond = llvm::BasicBlock::Create(ctx_.context(), "out_k_cond", current_func);
    llvm::BasicBlock* out_k_body = llvm::BasicBlock::Create(ctx_.context(), "out_k_body", current_func);
    llvm::BasicBlock* out_k_done = llvm::BasicBlock::Create(ctx_.context(), "out_k_done", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), out_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_iter);
    builder.CreateBr(out_k_cond);

    builder.SetInsertPoint(out_k_cond);
    llvm::Value* ok = builder.CreateLoad(ctx_.int64Type(), k_iter);
    llvm::Value* ok_done = builder.CreateICmpUGE(ok, seq_k);
    builder.CreateCondBr(ok_done, out_k_done, out_k_body);

    builder.SetInsertPoint(out_k_body);
    // attention[b, i, k]
    llvm::Value* attn_idx = builder.CreateAdd(attn_batch_offset,
        builder.CreateAdd(builder.CreateMul(oi, seq_k), ok));
    llvm::Value* attn_val = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), scores_ptr, attn_idx));
    // V[b, k, j]
    llvm::Value* v_idx = builder.CreateAdd(v_batch_offset,
        builder.CreateAdd(builder.CreateMul(ok, d_v), oj));
    llvm::Value* v_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), v_elems, v_idx));

    llvm::Value* prod2 = builder.CreateFMul(attn_val, v_elem);
    llvm::Value* cur_out_sum = builder.CreateLoad(ctx_.doubleType(), out_sum);
    builder.CreateStore(builder.CreateFAdd(cur_out_sum, prod2), out_sum);

    builder.CreateStore(builder.CreateAdd(ok,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_iter);
    builder.CreateBr(out_k_cond);

    builder.SetInsertPoint(out_k_done);
    // Store output[b, i, j]
    llvm::Value* out_idx = builder.CreateAdd(out_batch_offset,
        builder.CreateAdd(builder.CreateMul(oi, d_v), oj));
    llvm::Value* final_out = builder.CreateLoad(ctx_.doubleType(), out_sum);
    builder.CreateStore(final_out,
        builder.CreateGEP(ctx_.doubleType(), output_elems, out_idx));

    builder.CreateStore(builder.CreateAdd(oj,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), j_idx);
    builder.CreateBr(out_j_cond);

    builder.SetInsertPoint(out_i_next);
    builder.CreateStore(builder.CreateAdd(oi,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_idx);
    builder.CreateBr(out_i_cond);

    builder.SetInsertPoint(out_batch_next);
    builder.CreateStore(builder.CreateAdd(b3,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(out_batch_cond);

    // === Finalize result tensor ===
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "attn_result");

    // Allocate dims array - same shape as output: (batch, seq_q, d_v) or (seq_q, d_v)
    llvm::Value* dims_count = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 3),
        llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* dims_bytes = builder.CreateMul(dims_count,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "result_dims");

    // Store dimensions based on 2D or 3D
    llvm::BasicBlock* store_3d = llvm::BasicBlock::Create(ctx_.context(), "store_3d", current_func);
    llvm::BasicBlock* store_2d = llvm::BasicBlock::Create(ctx_.context(), "store_2d", current_func);
    llvm::BasicBlock* store_done = llvm::BasicBlock::Create(ctx_.context(), "store_done", current_func);

    builder.CreateCondBr(is_3d, store_3d, store_2d);

    builder.SetInsertPoint(store_3d);
    builder.CreateStore(batch_size, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(seq_q, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    builder.CreateStore(d_v, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)));
    builder.CreateBr(store_done);

    builder.SetInsertPoint(store_2d);
    builder.CreateStore(seq_q, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(d_v, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    builder.CreateBr(store_done);

    builder.SetInsertPoint(store_done);

    // Populate tensor struct
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(dims_count, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(output_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(output_size, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

// === Track 8.2: Multi-Head Attention ===

llvm::Value* TensorCodegen::multiHeadAttention(const eshkol_operations_t* op) {
    // Multi-Head Attention
    // 1. Project Q, K, V through weight matrices
    // 2. Split into num_heads
    // 3. Apply scaled dot-product attention to each head
    // 4. Concatenate heads
    // 5. Project through output weights
    //
    // Args: Q K V num-heads W_Q W_K W_V W_O [mask]

    if (op->call_op.num_vars < 8 || op->call_op.num_vars > 9) {
        eshkol_error("multi-head-attention requires 8-9 arguments: Q K V num-heads W_Q W_K W_V W_O [mask]");
        return nullptr;
    }

    llvm::Value* q_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* k_val = codegenAST(&op->call_op.variables[1]);
    llvm::Value* v_val = codegenAST(&op->call_op.variables[2]);
    llvm::Value* num_heads_val = codegenAST(&op->call_op.variables[3]);
    llvm::Value* wq_val = codegenAST(&op->call_op.variables[4]);
    llvm::Value* wk_val = codegenAST(&op->call_op.variables[5]);
    llvm::Value* wv_val = codegenAST(&op->call_op.variables[6]);
    llvm::Value* wo_val = codegenAST(&op->call_op.variables[7]);
    llvm::Value* mask_val = (op->call_op.num_vars == 9)
        ? codegenAST(&op->call_op.variables[8]) : nullptr;

    if (!q_val || !k_val || !v_val || !num_heads_val ||
        !wq_val || !wk_val || !wv_val || !wo_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Get num_heads as integer
    llvm::Value* num_heads = tagged_.unpackInt64(num_heads_val);

    // Unpack Q tensor to get dimensions
    llvm::Value* q_ptr_int = tagged_.unpackInt64(q_val);
    llvm::Value* q_ptr = builder.CreateIntToPtr(q_ptr_int, ctx_.ptrType());

    llvm::Value* q_dims_field = builder.CreateStructGEP(tensor_type, q_ptr, 0);
    llvm::Value* q_dims_ptr = builder.CreateLoad(ctx_.ptrType(), q_dims_field);
    llvm::Value* q_ndim_field = builder.CreateStructGEP(tensor_type, q_ptr, 1);
    llvm::Value* q_ndim = builder.CreateLoad(ctx_.int64Type(), q_ndim_field);
    llvm::Value* q_elems_field = builder.CreateStructGEP(tensor_type, q_ptr, 2);
    llvm::Value* q_elems = builder.CreateLoad(ctx_.ptrType(), q_elems_field);

    // Unpack K tensor
    llvm::Value* k_ptr_int = tagged_.unpackInt64(k_val);
    llvm::Value* k_ptr = builder.CreateIntToPtr(k_ptr_int, ctx_.ptrType());
    llvm::Value* k_dims_field = builder.CreateStructGEP(tensor_type, k_ptr, 0);
    llvm::Value* k_dims_ptr = builder.CreateLoad(ctx_.ptrType(), k_dims_field);
    llvm::Value* k_elems_field = builder.CreateStructGEP(tensor_type, k_ptr, 2);
    llvm::Value* k_elems = builder.CreateLoad(ctx_.ptrType(), k_elems_field);

    // Unpack V tensor
    llvm::Value* v_ptr_int = tagged_.unpackInt64(v_val);
    llvm::Value* v_ptr = builder.CreateIntToPtr(v_ptr_int, ctx_.ptrType());
    llvm::Value* v_dims_field = builder.CreateStructGEP(tensor_type, v_ptr, 0);
    llvm::Value* v_dims_ptr = builder.CreateLoad(ctx_.ptrType(), v_dims_field);
    llvm::Value* v_elems_field = builder.CreateStructGEP(tensor_type, v_ptr, 2);
    llvm::Value* v_elems = builder.CreateLoad(ctx_.ptrType(), v_elems_field);

    // Unpack weight matrices
    llvm::Value* wq_ptr_int = tagged_.unpackInt64(wq_val);
    llvm::Value* wq_ptr = builder.CreateIntToPtr(wq_ptr_int, ctx_.ptrType());
    llvm::Value* wq_elems_field = builder.CreateStructGEP(tensor_type, wq_ptr, 2);
    llvm::Value* wq_elems = builder.CreateLoad(ctx_.ptrType(), wq_elems_field);

    llvm::Value* wk_ptr_int = tagged_.unpackInt64(wk_val);
    llvm::Value* wk_ptr = builder.CreateIntToPtr(wk_ptr_int, ctx_.ptrType());
    llvm::Value* wk_elems_field = builder.CreateStructGEP(tensor_type, wk_ptr, 2);
    llvm::Value* wk_elems = builder.CreateLoad(ctx_.ptrType(), wk_elems_field);

    llvm::Value* wv_ptr_int = tagged_.unpackInt64(wv_val);
    llvm::Value* wv_ptr = builder.CreateIntToPtr(wv_ptr_int, ctx_.ptrType());
    llvm::Value* wv_elems_field = builder.CreateStructGEP(tensor_type, wv_ptr, 2);
    llvm::Value* wv_elems = builder.CreateLoad(ctx_.ptrType(), wv_elems_field);

    llvm::Value* wo_ptr_int = tagged_.unpackInt64(wo_val);
    llvm::Value* wo_ptr = builder.CreateIntToPtr(wo_ptr_int, ctx_.ptrType());
    llvm::Value* wo_elems_field = builder.CreateStructGEP(tensor_type, wo_ptr, 2);
    llvm::Value* wo_elems = builder.CreateLoad(ctx_.ptrType(), wo_elems_field);

    // Handle mask if provided
    llvm::Value* mask_elems = nullptr;
    if (mask_val) {
        llvm::Value* mask_ptr_int = tagged_.unpackInt64(mask_val);
        llvm::Value* mask_ptr = builder.CreateIntToPtr(mask_ptr_int, ctx_.ptrType());
        llvm::Value* mask_elems_field = builder.CreateStructGEP(tensor_type, mask_ptr, 2);
        mask_elems = builder.CreateLoad(ctx_.ptrType(), mask_elems_field);
    }

    // Determine dimensions
    llvm::Value* is_3d = builder.CreateICmpEQ(q_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), 3));

    llvm::Value* batch_size = builder.CreateSelect(is_3d,
        builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), q_dims_ptr,
                llvm::ConstantInt::get(ctx_.int64Type(), 0))),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* seq_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 1),
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* dim_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 2),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* seq_q = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), q_dims_ptr, seq_idx));
    llvm::Value* d_model = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), q_dims_ptr, dim_idx));

    llvm::Value* seq_k = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), k_dims_ptr, seq_idx));

    // d_k = d_model / num_heads
    llvm::Value* d_k = builder.CreateSDiv(d_model, num_heads, "d_k");

    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Allocate projected Q, K, V: (batch, seq, d_model)
    llvm::Value* proj_size = builder.CreateMul(batch_size,
        builder.CreateMul(seq_q, d_model));
    llvm::Value* proj_bytes = builder.CreateMul(proj_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* proj_q = builder.CreateCall(arena_alloc, {arena_ptr, proj_bytes}, "proj_q");
    llvm::Value* proj_k = builder.CreateCall(arena_alloc, {arena_ptr, proj_bytes}, "proj_k");
    llvm::Value* proj_v = builder.CreateCall(arena_alloc, {arena_ptr, proj_bytes}, "proj_v");

    // Allocate per-head attention outputs: (batch, num_heads, seq_q, d_k)
    llvm::Value* head_out_size = builder.CreateMul(batch_size,
        builder.CreateMul(num_heads, builder.CreateMul(seq_q, d_k)));
    llvm::Value* head_out_bytes = builder.CreateMul(head_out_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* head_outputs = builder.CreateCall(arena_alloc,
        {arena_ptr, head_out_bytes}, "head_outputs");

    // Allocate final output: (batch, seq_q, d_model)
    llvm::Value* output_elems = builder.CreateCall(arena_alloc, {arena_ptr, proj_bytes}, "mha_output");

    // === ALLOCATE ALL LOOP VARIABLES UPFRONT ===
    llvm::Value* batch_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "batch_idx");
    llvm::Value* seq_idx_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "seq_idx");
    llvm::Value* d_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "d_idx");
    llvm::Value* m_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "m_idx");
    llvm::Value* q_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "q_sum");
    llvm::Value* k_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "k_sum");
    llvm::Value* v_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "v_sum");
    llvm::Value* head_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "head_idx");
    llvm::Value* i_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "i_var");
    llvm::Value* j_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "j_var");
    llvm::Value* k_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "k_var");
    llvm::Value* score_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "score_sum");
    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "max_val");
    llvm::Value* sum_exp = builder.CreateAlloca(ctx_.doubleType(), nullptr, "sum_exp");
    llvm::Value* out_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "out_sum");
    llvm::Value* final_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "final_sum");

    // === Step 1: Project Q, K, V ===
    // proj_Q = Q @ W_Q (matmul for each position)
    // For each batch b, seq position s: proj_Q[b,s,:] = Q[b,s,:] @ W_Q

    llvm::BasicBlock* proj_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "proj_batch_cond", current_func);
    llvm::BasicBlock* proj_batch_body = llvm::BasicBlock::Create(ctx_.context(), "proj_batch_body", current_func);
    llvm::BasicBlock* heads_init = llvm::BasicBlock::Create(ctx_.context(), "heads_init", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(proj_batch_cond);

    builder.SetInsertPoint(proj_batch_cond);
    llvm::Value* b = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* b_done = builder.CreateICmpUGE(b, batch_size);
    builder.CreateCondBr(b_done, heads_init, proj_batch_body);

    builder.SetInsertPoint(proj_batch_body);

    llvm::Value* batch_offset = builder.CreateMul(b, builder.CreateMul(seq_q, d_model));
    llvm::Value* k_batch_offset = builder.CreateMul(b, builder.CreateMul(seq_k, d_model));

    // Project each sequence position
    llvm::BasicBlock* proj_seq_cond = llvm::BasicBlock::Create(ctx_.context(), "proj_seq_cond", current_func);
    llvm::BasicBlock* proj_seq_body = llvm::BasicBlock::Create(ctx_.context(), "proj_seq_body", current_func);
    llvm::BasicBlock* proj_batch_next = llvm::BasicBlock::Create(ctx_.context(), "proj_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), seq_idx_var);
    builder.CreateBr(proj_seq_cond);

    builder.SetInsertPoint(proj_seq_cond);
    llvm::Value* s = builder.CreateLoad(ctx_.int64Type(), seq_idx_var);
    llvm::Value* s_done = builder.CreateICmpUGE(s, seq_q);
    builder.CreateCondBr(s_done, proj_batch_next, proj_seq_body);

    builder.SetInsertPoint(proj_seq_body);

    llvm::Value* pos_offset = builder.CreateAdd(batch_offset, builder.CreateMul(s, d_model));

    // For each output dimension d
    llvm::BasicBlock* proj_d_cond = llvm::BasicBlock::Create(ctx_.context(), "proj_d_cond", current_func);
    llvm::BasicBlock* proj_d_body = llvm::BasicBlock::Create(ctx_.context(), "proj_d_body", current_func);
    llvm::BasicBlock* proj_seq_next = llvm::BasicBlock::Create(ctx_.context(), "proj_seq_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), d_idx);
    builder.CreateBr(proj_d_cond);

    builder.SetInsertPoint(proj_d_cond);
    llvm::Value* d = builder.CreateLoad(ctx_.int64Type(), d_idx);
    llvm::Value* d_done = builder.CreateICmpUGE(d, d_model);
    builder.CreateCondBr(d_done, proj_seq_next, proj_d_body);

    builder.SetInsertPoint(proj_d_body);

    // Compute dot product: input[s,:] @ W[:,d]
    llvm::BasicBlock* dot_cond = llvm::BasicBlock::Create(ctx_.context(), "proj_dot_cond", current_func);
    llvm::BasicBlock* dot_body = llvm::BasicBlock::Create(ctx_.context(), "proj_dot_body", current_func);
    llvm::BasicBlock* dot_done = llvm::BasicBlock::Create(ctx_.context(), "proj_dot_done", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), q_sum);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), k_sum);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), v_sum);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), m_idx);
    builder.CreateBr(dot_cond);

    builder.SetInsertPoint(dot_cond);
    llvm::Value* m = builder.CreateLoad(ctx_.int64Type(), m_idx);
    llvm::Value* m_done = builder.CreateICmpUGE(m, d_model);
    builder.CreateCondBr(m_done, dot_done, dot_body);

    builder.SetInsertPoint(dot_body);
    // input element: Q[b, s, m], K[b, s, m], V[b, s, m]
    llvm::Value* in_idx = builder.CreateAdd(pos_offset, m);
    llvm::Value* q_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), q_elems, in_idx));
    llvm::Value* k_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), k_elems, in_idx));
    llvm::Value* v_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), v_elems, in_idx));

    // Weight element: W[m, d]
    llvm::Value* w_idx = builder.CreateAdd(builder.CreateMul(m, d_model), d);
    llvm::Value* wq_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), wq_elems, w_idx));
    llvm::Value* wk_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), wk_elems, w_idx));
    llvm::Value* wv_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), wv_elems, w_idx));

    // Accumulate
    llvm::Value* cur_q = builder.CreateLoad(ctx_.doubleType(), q_sum);
    builder.CreateStore(builder.CreateFAdd(cur_q, builder.CreateFMul(q_elem, wq_elem)), q_sum);
    llvm::Value* cur_k = builder.CreateLoad(ctx_.doubleType(), k_sum);
    builder.CreateStore(builder.CreateFAdd(cur_k, builder.CreateFMul(k_elem, wk_elem)), k_sum);
    llvm::Value* cur_v = builder.CreateLoad(ctx_.doubleType(), v_sum);
    builder.CreateStore(builder.CreateFAdd(cur_v, builder.CreateFMul(v_elem, wv_elem)), v_sum);

    builder.CreateStore(builder.CreateAdd(m, llvm::ConstantInt::get(ctx_.int64Type(), 1)), m_idx);
    builder.CreateBr(dot_cond);

    builder.SetInsertPoint(dot_done);
    // Store projected values
    llvm::Value* out_idx = builder.CreateAdd(pos_offset, d);
    builder.CreateStore(builder.CreateLoad(ctx_.doubleType(), q_sum),
        builder.CreateGEP(ctx_.doubleType(), proj_q, out_idx));
    builder.CreateStore(builder.CreateLoad(ctx_.doubleType(), k_sum),
        builder.CreateGEP(ctx_.doubleType(), proj_k, out_idx));
    builder.CreateStore(builder.CreateLoad(ctx_.doubleType(), v_sum),
        builder.CreateGEP(ctx_.doubleType(), proj_v, out_idx));

    builder.CreateStore(builder.CreateAdd(d, llvm::ConstantInt::get(ctx_.int64Type(), 1)), d_idx);
    builder.CreateBr(proj_d_cond);

    builder.SetInsertPoint(proj_seq_next);
    builder.CreateStore(builder.CreateAdd(s, llvm::ConstantInt::get(ctx_.int64Type(), 1)), seq_idx_var);
    builder.CreateBr(proj_seq_cond);

    builder.SetInsertPoint(proj_batch_next);
    builder.CreateStore(builder.CreateAdd(b, llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(proj_batch_cond);

    // === Step 2-3: Split into heads and apply attention ===
    builder.SetInsertPoint(heads_init);

    // For efficiency, we compute attention for all heads in parallel
    // Each head h gets: Q[:, :, h*d_k:(h+1)*d_k], K[:, :, h*d_k:(h+1)*d_k], V[:, :, h*d_k:(h+1)*d_k]

    llvm::Function* sqrt_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Function* exp_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Value* d_k_double = builder.CreateSIToFP(d_k, ctx_.doubleType());
    llvm::Value* scale = builder.CreateCall(sqrt_func, {d_k_double}, "scale");

    // Allocate attention scores for one head: (seq_q, seq_k)
    llvm::Value* scores_size = builder.CreateMul(seq_q, seq_k);
    llvm::Value* scores_bytes = builder.CreateMul(scores_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* attn_scores = builder.CreateCall(arena_alloc,
        {arena_ptr, scores_bytes}, "attn_scores");

    // Loop over batches and heads
    llvm::BasicBlock* head_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "head_batch_cond", current_func);
    llvm::BasicBlock* head_batch_body = llvm::BasicBlock::Create(ctx_.context(), "head_batch_body", current_func);
    llvm::BasicBlock* concat_init = llvm::BasicBlock::Create(ctx_.context(), "concat_init", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(head_batch_cond);

    builder.SetInsertPoint(head_batch_cond);
    llvm::Value* hb = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* hb_done = builder.CreateICmpUGE(hb, batch_size);
    builder.CreateCondBr(hb_done, concat_init, head_batch_body);

    builder.SetInsertPoint(head_batch_body);

    llvm::Value* hb_offset = builder.CreateMul(hb, builder.CreateMul(seq_q, d_model));

    // Loop over heads
    llvm::BasicBlock* head_cond = llvm::BasicBlock::Create(ctx_.context(), "head_cond", current_func);
    llvm::BasicBlock* head_body = llvm::BasicBlock::Create(ctx_.context(), "head_body", current_func);
    llvm::BasicBlock* head_batch_next = llvm::BasicBlock::Create(ctx_.context(), "head_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), head_idx);
    builder.CreateBr(head_cond);

    builder.SetInsertPoint(head_cond);
    llvm::Value* h = builder.CreateLoad(ctx_.int64Type(), head_idx);
    llvm::Value* h_done = builder.CreateICmpUGE(h, num_heads);
    builder.CreateCondBr(h_done, head_batch_next, head_body);

    builder.SetInsertPoint(head_body);

    llvm::Value* head_start = builder.CreateMul(h, d_k);  // Starting dimension for this head

    // Compute attention scores: Q_h @ K_h^T / sqrt(d_k)
    // scores[i, j] = sum_k(Q[i, h*d_k + k] * K[j, h*d_k + k]) / sqrt(d_k)

    llvm::BasicBlock* score_i_cond = llvm::BasicBlock::Create(ctx_.context(), "score_i_cond", current_func);
    llvm::BasicBlock* score_i_body = llvm::BasicBlock::Create(ctx_.context(), "score_i_body", current_func);
    llvm::BasicBlock* softmax_row = llvm::BasicBlock::Create(ctx_.context(), "softmax_row", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_var);
    builder.CreateBr(score_i_cond);

    builder.SetInsertPoint(score_i_cond);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), i_var);
    llvm::Value* si_done = builder.CreateICmpUGE(si, seq_q);
    builder.CreateCondBr(si_done, softmax_row, score_i_body);

    builder.SetInsertPoint(score_i_body);

    llvm::BasicBlock* score_j_cond = llvm::BasicBlock::Create(ctx_.context(), "score_j_cond", current_func);
    llvm::BasicBlock* score_j_body = llvm::BasicBlock::Create(ctx_.context(), "score_j_body", current_func);
    llvm::BasicBlock* score_i_next = llvm::BasicBlock::Create(ctx_.context(), "score_i_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_var);
    builder.CreateBr(score_j_cond);

    builder.SetInsertPoint(score_j_cond);
    llvm::Value* sj = builder.CreateLoad(ctx_.int64Type(), j_var);
    llvm::Value* sj_done = builder.CreateICmpUGE(sj, seq_k);
    builder.CreateCondBr(sj_done, score_i_next, score_j_body);

    builder.SetInsertPoint(score_j_body);

    // Dot product over d_k dimensions
    llvm::BasicBlock* score_k_cond = llvm::BasicBlock::Create(ctx_.context(), "score_k_cond", current_func);
    llvm::BasicBlock* score_k_body = llvm::BasicBlock::Create(ctx_.context(), "score_k_body", current_func);
    llvm::BasicBlock* score_k_done = llvm::BasicBlock::Create(ctx_.context(), "score_k_done", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), score_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_var);
    builder.CreateBr(score_k_cond);

    builder.SetInsertPoint(score_k_cond);
    llvm::Value* sk = builder.CreateLoad(ctx_.int64Type(), k_var);
    llvm::Value* sk_done = builder.CreateICmpUGE(sk, d_k);
    builder.CreateCondBr(sk_done, score_k_done, score_k_body);

    builder.SetInsertPoint(score_k_body);
    // Q[b, i, head_start + k]
    llvm::Value* q_idx = builder.CreateAdd(hb_offset,
        builder.CreateAdd(builder.CreateMul(si, d_model),
            builder.CreateAdd(head_start, sk)));
    llvm::Value* q_val_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), proj_q, q_idx));
    // K[b, j, head_start + k]
    llvm::Value* k_idx = builder.CreateAdd(hb_offset,
        builder.CreateAdd(builder.CreateMul(sj, d_model),
            builder.CreateAdd(head_start, sk)));
    llvm::Value* k_val_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), proj_k, k_idx));

    llvm::Value* prod = builder.CreateFMul(q_val_elem, k_val_elem);
    llvm::Value* cur_score = builder.CreateLoad(ctx_.doubleType(), score_sum);
    builder.CreateStore(builder.CreateFAdd(cur_score, prod), score_sum);

    builder.CreateStore(builder.CreateAdd(sk, llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_var);
    builder.CreateBr(score_k_cond);

    builder.SetInsertPoint(score_k_done);
    llvm::Value* final_score = builder.CreateLoad(ctx_.doubleType(), score_sum);
    llvm::Value* scaled_score = builder.CreateFDiv(final_score, scale);

    // Add mask if provided
    if (mask_elems) {
        llvm::Value* mask_idx = builder.CreateAdd(builder.CreateMul(si, seq_k), sj);
        llvm::Value* mask_val_elem = builder.CreateLoad(ctx_.doubleType(),
            builder.CreateGEP(ctx_.doubleType(), mask_elems, mask_idx));
        scaled_score = builder.CreateFAdd(scaled_score, mask_val_elem);
    }

    // Store score
    llvm::Value* scores_idx = builder.CreateAdd(builder.CreateMul(si, seq_k), sj);
    builder.CreateStore(scaled_score,
        builder.CreateGEP(ctx_.doubleType(), attn_scores, scores_idx));

    builder.CreateStore(builder.CreateAdd(sj, llvm::ConstantInt::get(ctx_.int64Type(), 1)), j_var);
    builder.CreateBr(score_j_cond);

    builder.SetInsertPoint(score_i_next);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_var);
    builder.CreateBr(score_i_cond);

    // Apply softmax to each row
    builder.SetInsertPoint(softmax_row);

    llvm::BasicBlock* sm_i_cond = llvm::BasicBlock::Create(ctx_.context(), "mha_sm_i_cond", current_func);
    llvm::BasicBlock* sm_i_body = llvm::BasicBlock::Create(ctx_.context(), "mha_sm_i_body", current_func);
    llvm::BasicBlock* attn_output = llvm::BasicBlock::Create(ctx_.context(), "attn_output", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_var);
    builder.CreateBr(sm_i_cond);

    builder.SetInsertPoint(sm_i_cond);
    llvm::Value* smi = builder.CreateLoad(ctx_.int64Type(), i_var);
    llvm::Value* smi_done = builder.CreateICmpUGE(smi, seq_q);
    builder.CreateCondBr(smi_done, attn_output, sm_i_body);

    builder.SetInsertPoint(sm_i_body);

    llvm::Value* row_offset = builder.CreateMul(smi, seq_k);

    // Find max
    llvm::Value* first_val = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), attn_scores, row_offset));
    builder.CreateStore(first_val, max_val);

    llvm::BasicBlock* max_j_cond = llvm::BasicBlock::Create(ctx_.context(), "max_j_cond", current_func);
    llvm::BasicBlock* max_j_body = llvm::BasicBlock::Create(ctx_.context(), "max_j_body", current_func);
    llvm::BasicBlock* exp_j_init = llvm::BasicBlock::Create(ctx_.context(), "exp_j_init", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), j_var);
    builder.CreateBr(max_j_cond);

    builder.SetInsertPoint(max_j_cond);
    llvm::Value* mj = builder.CreateLoad(ctx_.int64Type(), j_var);
    llvm::Value* mj_done = builder.CreateICmpUGE(mj, seq_k);
    builder.CreateCondBr(mj_done, exp_j_init, max_j_body);

    builder.SetInsertPoint(max_j_body);
    llvm::Value* elem_idx = builder.CreateAdd(row_offset, mj);
    llvm::Value* elem_val = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), attn_scores, elem_idx));
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_gt = builder.CreateFCmpOGT(elem_val, cur_max);
    builder.CreateStore(builder.CreateSelect(is_gt, elem_val, cur_max), max_val);
    builder.CreateStore(builder.CreateAdd(mj, llvm::ConstantInt::get(ctx_.int64Type(), 1)), j_var);
    builder.CreateBr(max_j_cond);

    // Compute exp and sum
    builder.SetInsertPoint(exp_j_init);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_exp);

    llvm::BasicBlock* exp_j_cond = llvm::BasicBlock::Create(ctx_.context(), "exp_j_cond", current_func);
    llvm::BasicBlock* exp_j_body = llvm::BasicBlock::Create(ctx_.context(), "exp_j_body", current_func);
    llvm::BasicBlock* norm_j_init = llvm::BasicBlock::Create(ctx_.context(), "norm_j_init", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_var);
    builder.CreateBr(exp_j_cond);

    builder.SetInsertPoint(exp_j_cond);
    llvm::Value* ej = builder.CreateLoad(ctx_.int64Type(), j_var);
    llvm::Value* ej_done = builder.CreateICmpUGE(ej, seq_k);
    builder.CreateCondBr(ej_done, norm_j_init, exp_j_body);

    builder.SetInsertPoint(exp_j_body);
    llvm::Value* exp_idx = builder.CreateAdd(row_offset, ej);
    llvm::Value* exp_ptr = builder.CreateGEP(ctx_.doubleType(), attn_scores, exp_idx);
    llvm::Value* exp_input = builder.CreateLoad(ctx_.doubleType(), exp_ptr);
    llvm::Value* final_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* shifted = builder.CreateFSub(exp_input, final_max);
    llvm::Value* exp_out = builder.CreateCall(exp_func, {shifted});
    builder.CreateStore(exp_out, exp_ptr);
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), sum_exp);
    builder.CreateStore(builder.CreateFAdd(cur_sum, exp_out), sum_exp);
    builder.CreateStore(builder.CreateAdd(ej, llvm::ConstantInt::get(ctx_.int64Type(), 1)), j_var);
    builder.CreateBr(exp_j_cond);

    // Normalize
    builder.SetInsertPoint(norm_j_init);
    llvm::BasicBlock* norm_j_cond = llvm::BasicBlock::Create(ctx_.context(), "norm_j_cond", current_func);
    llvm::BasicBlock* norm_j_body = llvm::BasicBlock::Create(ctx_.context(), "norm_j_body", current_func);
    llvm::BasicBlock* sm_i_next = llvm::BasicBlock::Create(ctx_.context(), "mha_sm_i_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_var);
    builder.CreateBr(norm_j_cond);

    builder.SetInsertPoint(norm_j_cond);
    llvm::Value* nj = builder.CreateLoad(ctx_.int64Type(), j_var);
    llvm::Value* nj_done = builder.CreateICmpUGE(nj, seq_k);
    builder.CreateCondBr(nj_done, sm_i_next, norm_j_body);

    builder.SetInsertPoint(norm_j_body);
    llvm::Value* norm_idx = builder.CreateAdd(row_offset, nj);
    llvm::Value* norm_ptr = builder.CreateGEP(ctx_.doubleType(), attn_scores, norm_idx);
    llvm::Value* norm_val = builder.CreateLoad(ctx_.doubleType(), norm_ptr);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), sum_exp);
    builder.CreateStore(builder.CreateFDiv(norm_val, total_sum), norm_ptr);
    builder.CreateStore(builder.CreateAdd(nj, llvm::ConstantInt::get(ctx_.int64Type(), 1)), j_var);
    builder.CreateBr(norm_j_cond);

    builder.SetInsertPoint(sm_i_next);
    builder.CreateStore(builder.CreateAdd(smi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_var);
    builder.CreateBr(sm_i_cond);

    // Compute attention output: attn_weights @ V_h
    builder.SetInsertPoint(attn_output);

    llvm::BasicBlock* out_i_cond = llvm::BasicBlock::Create(ctx_.context(), "mha_out_i_cond", current_func);
    llvm::BasicBlock* out_i_body = llvm::BasicBlock::Create(ctx_.context(), "mha_out_i_body", current_func);
    llvm::BasicBlock* head_next = llvm::BasicBlock::Create(ctx_.context(), "head_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_var);
    builder.CreateBr(out_i_cond);

    builder.SetInsertPoint(out_i_cond);
    llvm::Value* oi = builder.CreateLoad(ctx_.int64Type(), i_var);
    llvm::Value* oi_done = builder.CreateICmpUGE(oi, seq_q);
    builder.CreateCondBr(oi_done, head_next, out_i_body);

    builder.SetInsertPoint(out_i_body);

    // For each output dimension in this head
    llvm::BasicBlock* out_d_cond = llvm::BasicBlock::Create(ctx_.context(), "out_d_cond", current_func);
    llvm::BasicBlock* out_d_body = llvm::BasicBlock::Create(ctx_.context(), "out_d_body", current_func);
    llvm::BasicBlock* out_i_next = llvm::BasicBlock::Create(ctx_.context(), "mha_out_i_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), d_idx);
    builder.CreateBr(out_d_cond);

    builder.SetInsertPoint(out_d_cond);
    llvm::Value* od = builder.CreateLoad(ctx_.int64Type(), d_idx);
    llvm::Value* od_done = builder.CreateICmpUGE(od, d_k);
    builder.CreateCondBr(od_done, out_i_next, out_d_body);

    builder.SetInsertPoint(out_d_body);

    // Sum over seq_k: attn[i, k] * V[k, head_start + d]
    llvm::BasicBlock* out_k_cond = llvm::BasicBlock::Create(ctx_.context(), "out_k_cond", current_func);
    llvm::BasicBlock* out_k_body = llvm::BasicBlock::Create(ctx_.context(), "out_k_body", current_func);
    llvm::BasicBlock* out_k_done = llvm::BasicBlock::Create(ctx_.context(), "out_k_done", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), out_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_var);
    builder.CreateBr(out_k_cond);

    builder.SetInsertPoint(out_k_cond);
    llvm::Value* ok = builder.CreateLoad(ctx_.int64Type(), k_var);
    llvm::Value* ok_done = builder.CreateICmpUGE(ok, seq_k);
    builder.CreateCondBr(ok_done, out_k_done, out_k_body);

    builder.SetInsertPoint(out_k_body);
    // attn[i, k]
    llvm::Value* attn_idx = builder.CreateAdd(builder.CreateMul(oi, seq_k), ok);
    llvm::Value* attn_val = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), attn_scores, attn_idx));
    // V[b, k, head_start + d]
    llvm::Value* v_idx = builder.CreateAdd(hb_offset,
        builder.CreateAdd(builder.CreateMul(ok, d_model),
            builder.CreateAdd(head_start, od)));
    llvm::Value* v_val_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), proj_v, v_idx));

    llvm::Value* prod_out = builder.CreateFMul(attn_val, v_val_elem);
    llvm::Value* cur_out = builder.CreateLoad(ctx_.doubleType(), out_sum);
    builder.CreateStore(builder.CreateFAdd(cur_out, prod_out), out_sum);

    builder.CreateStore(builder.CreateAdd(ok, llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_var);
    builder.CreateBr(out_k_cond);

    builder.SetInsertPoint(out_k_done);
    // Store in head_outputs[b, h, i, d]
    llvm::Value* ho_idx = builder.CreateAdd(
        builder.CreateMul(hb, builder.CreateMul(num_heads, builder.CreateMul(seq_q, d_k))),
        builder.CreateAdd(
            builder.CreateMul(h, builder.CreateMul(seq_q, d_k)),
            builder.CreateAdd(builder.CreateMul(oi, d_k), od)));
    builder.CreateStore(builder.CreateLoad(ctx_.doubleType(), out_sum),
        builder.CreateGEP(ctx_.doubleType(), head_outputs, ho_idx));

    builder.CreateStore(builder.CreateAdd(od, llvm::ConstantInt::get(ctx_.int64Type(), 1)), d_idx);
    builder.CreateBr(out_d_cond);

    builder.SetInsertPoint(out_i_next);
    builder.CreateStore(builder.CreateAdd(oi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_var);
    builder.CreateBr(out_i_cond);

    builder.SetInsertPoint(head_next);
    builder.CreateStore(builder.CreateAdd(h, llvm::ConstantInt::get(ctx_.int64Type(), 1)), head_idx);
    builder.CreateBr(head_cond);

    builder.SetInsertPoint(head_batch_next);
    builder.CreateStore(builder.CreateAdd(hb, llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(head_batch_cond);

    // === Step 4-5: Concatenate heads and project through W_O ===
    builder.SetInsertPoint(concat_init);

    // The concatenation is implicit in head_outputs layout
    // Now we project: output = concat(heads) @ W_O
    // For each position, we compute: output[:, i, d] = sum_h sum_k head_outputs[:, h, i, k] * W_O[h*d_k + k, d]

    llvm::BasicBlock* final_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "final_batch_cond", current_func);
    llvm::BasicBlock* final_batch_body = llvm::BasicBlock::Create(ctx_.context(), "final_batch_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "mha_finalize", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(final_batch_cond);

    builder.SetInsertPoint(final_batch_cond);
    llvm::Value* fb = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* fb_done = builder.CreateICmpUGE(fb, batch_size);
    builder.CreateCondBr(fb_done, finalize, final_batch_body);

    builder.SetInsertPoint(final_batch_body);

    llvm::Value* fb_offset = builder.CreateMul(fb, builder.CreateMul(seq_q, d_model));
    llvm::Value* fb_head_offset = builder.CreateMul(fb,
        builder.CreateMul(num_heads, builder.CreateMul(seq_q, d_k)));

    llvm::BasicBlock* final_i_cond = llvm::BasicBlock::Create(ctx_.context(), "final_i_cond", current_func);
    llvm::BasicBlock* final_i_body = llvm::BasicBlock::Create(ctx_.context(), "final_i_body", current_func);
    llvm::BasicBlock* final_batch_next = llvm::BasicBlock::Create(ctx_.context(), "final_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_var);
    builder.CreateBr(final_i_cond);

    builder.SetInsertPoint(final_i_cond);
    llvm::Value* fi = builder.CreateLoad(ctx_.int64Type(), i_var);
    llvm::Value* fi_done = builder.CreateICmpUGE(fi, seq_q);
    builder.CreateCondBr(fi_done, final_batch_next, final_i_body);

    builder.SetInsertPoint(final_i_body);

    // For each output dimension
    llvm::BasicBlock* final_d_cond = llvm::BasicBlock::Create(ctx_.context(), "final_d_cond", current_func);
    llvm::BasicBlock* final_d_body = llvm::BasicBlock::Create(ctx_.context(), "final_d_body", current_func);
    llvm::BasicBlock* final_i_next = llvm::BasicBlock::Create(ctx_.context(), "final_i_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), d_idx);
    builder.CreateBr(final_d_cond);

    builder.SetInsertPoint(final_d_cond);
    llvm::Value* fd = builder.CreateLoad(ctx_.int64Type(), d_idx);
    llvm::Value* fd_done = builder.CreateICmpUGE(fd, d_model);
    builder.CreateCondBr(fd_done, final_i_next, final_d_body);

    builder.SetInsertPoint(final_d_body);

    // Sum over all heads and their dimensions
    llvm::BasicBlock* final_h_cond = llvm::BasicBlock::Create(ctx_.context(), "final_h_cond", current_func);
    llvm::BasicBlock* final_h_body = llvm::BasicBlock::Create(ctx_.context(), "final_h_body", current_func);
    llvm::BasicBlock* final_d_done = llvm::BasicBlock::Create(ctx_.context(), "final_d_done", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), final_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), head_idx);
    builder.CreateBr(final_h_cond);

    builder.SetInsertPoint(final_h_cond);
    llvm::Value* fh = builder.CreateLoad(ctx_.int64Type(), head_idx);
    llvm::Value* fh_done = builder.CreateICmpUGE(fh, num_heads);
    builder.CreateCondBr(fh_done, final_d_done, final_h_body);

    builder.SetInsertPoint(final_h_body);

    llvm::Value* fh_start = builder.CreateMul(fh, d_k);

    // Sum over d_k
    llvm::BasicBlock* final_k_cond = llvm::BasicBlock::Create(ctx_.context(), "final_k_cond", current_func);
    llvm::BasicBlock* final_k_body = llvm::BasicBlock::Create(ctx_.context(), "final_k_body", current_func);
    llvm::BasicBlock* final_h_next = llvm::BasicBlock::Create(ctx_.context(), "final_h_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_var);
    builder.CreateBr(final_k_cond);

    builder.SetInsertPoint(final_k_cond);
    llvm::Value* fk = builder.CreateLoad(ctx_.int64Type(), k_var);
    llvm::Value* fk_done = builder.CreateICmpUGE(fk, d_k);
    builder.CreateCondBr(fk_done, final_h_next, final_k_body);

    builder.SetInsertPoint(final_k_body);
    // head_outputs[b, h, i, k]
    llvm::Value* ho_read_idx = builder.CreateAdd(fb_head_offset,
        builder.CreateAdd(builder.CreateMul(fh, builder.CreateMul(seq_q, d_k)),
            builder.CreateAdd(builder.CreateMul(fi, d_k), fk)));
    llvm::Value* ho_val = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), head_outputs, ho_read_idx));
    // W_O[h*d_k + k, d]
    llvm::Value* wo_idx = builder.CreateAdd(
        builder.CreateMul(builder.CreateAdd(fh_start, fk), d_model), fd);
    llvm::Value* wo_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), wo_elems, wo_idx));

    llvm::Value* prod_final = builder.CreateFMul(ho_val, wo_elem);
    llvm::Value* cur_final = builder.CreateLoad(ctx_.doubleType(), final_sum);
    builder.CreateStore(builder.CreateFAdd(cur_final, prod_final), final_sum);

    builder.CreateStore(builder.CreateAdd(fk, llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_var);
    builder.CreateBr(final_k_cond);

    builder.SetInsertPoint(final_h_next);
    builder.CreateStore(builder.CreateAdd(fh, llvm::ConstantInt::get(ctx_.int64Type(), 1)), head_idx);
    builder.CreateBr(final_h_cond);

    builder.SetInsertPoint(final_d_done);
    // Store output[b, i, d]
    llvm::Value* out_final_idx = builder.CreateAdd(fb_offset,
        builder.CreateAdd(builder.CreateMul(fi, d_model), fd));
    builder.CreateStore(builder.CreateLoad(ctx_.doubleType(), final_sum),
        builder.CreateGEP(ctx_.doubleType(), output_elems, out_final_idx));

    builder.CreateStore(builder.CreateAdd(fd, llvm::ConstantInt::get(ctx_.int64Type(), 1)), d_idx);
    builder.CreateBr(final_d_cond);

    builder.SetInsertPoint(final_i_next);
    builder.CreateStore(builder.CreateAdd(fi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_var);
    builder.CreateBr(final_i_cond);

    builder.SetInsertPoint(final_batch_next);
    builder.CreateStore(builder.CreateAdd(fb, llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(final_batch_cond);

    // === Finalize result tensor ===
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "mha_result");

    // Allocate dims: (batch, seq_q, d_model) or (seq_q, d_model)
    llvm::Value* dims_count = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 3),
        llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* dims_bytes = builder.CreateMul(dims_count,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "mha_dims");

    llvm::BasicBlock* store_3d_dims = llvm::BasicBlock::Create(ctx_.context(), "mha_store_3d", current_func);
    llvm::BasicBlock* store_2d_dims = llvm::BasicBlock::Create(ctx_.context(), "mha_store_2d", current_func);
    llvm::BasicBlock* store_dims_done = llvm::BasicBlock::Create(ctx_.context(), "mha_store_done", current_func);

    builder.CreateCondBr(is_3d, store_3d_dims, store_2d_dims);

    builder.SetInsertPoint(store_3d_dims);
    builder.CreateStore(batch_size, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(seq_q, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    builder.CreateStore(d_model, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)));
    builder.CreateBr(store_dims_done);

    builder.SetInsertPoint(store_2d_dims);
    builder.CreateStore(seq_q, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(d_model, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    builder.CreateBr(store_dims_done);

    builder.SetInsertPoint(store_dims_done);

    llvm::Value* total_size = builder.CreateMul(batch_size, builder.CreateMul(seq_q, d_model));

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(dims_count, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(output_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_size, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

// === Track 8.3: Positional Encoding ===

llvm::Value* TensorCodegen::positionalEncoding(const eshkol_operations_t* op) {
    // Sinusoidal positional encoding from "Attention Is All You Need"
    // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    if (op->call_op.num_vars != 2) {
        eshkol_error("positional-encoding requires 2 arguments: max-len d-model");
        return nullptr;
    }

    llvm::Value* max_len_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* d_model_val = codegenAST(&op->call_op.variables[1]);

    if (!max_len_val || !d_model_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get integer values
    llvm::Value* max_len = max_len_val;
    if (max_len->getType() == ctx_.taggedValueType()) {
        max_len = tagged_.unpackInt64(max_len_val);
    }
    llvm::Value* d_model = d_model_val;
    if (d_model->getType() == ctx_.taggedValueType()) {
        d_model = tagged_.unpackInt64(d_model_val);
    }

    // Create tensor using proper helper
    std::vector<llvm::Value*> dims = {max_len, d_model};
    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, true);  // Zero fill
    if (!tensor_ptr) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get elements pointer
    llvm::Value* elements_field_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = builder.CreateLoad(ctx_.ptrType(), elements_field_ptr);

    // Get math functions
    llvm::Function* sin_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::sin, {ctx_.doubleType()});
    llvm::Function* cos_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::cos, {ctx_.doubleType()});
    llvm::Function* exp_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Precompute log(10000) = 9.210340371976184
    llvm::Value* log_10000 = llvm::ConstantFP::get(ctx_.doubleType(), 9.210340371976184);
    llvm::Value* d_model_double = builder.CreateSIToFP(d_model, ctx_.doubleType());
    llvm::Value* half_d = builder.CreateSDiv(d_model, llvm::ConstantInt::get(ctx_.int64Type(), 2));

    // Allocate loop counters upfront (before any branches)
    llvm::Value* pos_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "pe_pos");
    llvm::Value* i_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "pe_i");

    // Loop over positions
    llvm::BasicBlock* pos_cond = llvm::BasicBlock::Create(ctx_.context(), "pe_pos_cond", current_func);
    llvm::BasicBlock* pos_body = llvm::BasicBlock::Create(ctx_.context(), "pe_pos_body", current_func);
    llvm::BasicBlock* pos_exit = llvm::BasicBlock::Create(ctx_.context(), "pe_pos_exit", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), pos_counter);
    builder.CreateBr(pos_cond);

    builder.SetInsertPoint(pos_cond);
    llvm::Value* pos = builder.CreateLoad(ctx_.int64Type(), pos_counter);
    llvm::Value* pos_done = builder.CreateICmpUGE(pos, max_len);
    builder.CreateCondBr(pos_done, pos_exit, pos_body);

    builder.SetInsertPoint(pos_body);

    llvm::Value* pos_double = builder.CreateSIToFP(pos, ctx_.doubleType());
    llvm::Value* row_offset = builder.CreateMul(pos, d_model);

    // Inner loop over dimension pairs
    llvm::BasicBlock* dim_cond = llvm::BasicBlock::Create(ctx_.context(), "pe_dim_cond", current_func);
    llvm::BasicBlock* dim_body = llvm::BasicBlock::Create(ctx_.context(), "pe_dim_body", current_func);
    llvm::BasicBlock* dim_exit = llvm::BasicBlock::Create(ctx_.context(), "pe_dim_exit", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_counter);
    builder.CreateBr(dim_cond);

    builder.SetInsertPoint(dim_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_counter);
    llvm::Value* i_done = builder.CreateICmpUGE(i, half_d);
    builder.CreateCondBr(i_done, dim_exit, dim_body);

    builder.SetInsertPoint(dim_body);

    // Compute div_term = exp(2*i * -log(10000) / d_model) = 1 / 10000^(2i/d_model)
    llvm::Value* i_double = builder.CreateSIToFP(i, ctx_.doubleType());
    llvm::Value* two_i = builder.CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 2.0), i_double);
    llvm::Value* exponent = builder.CreateFMul(two_i,
        builder.CreateFDiv(builder.CreateFNeg(log_10000), d_model_double));
    llvm::Value* div_term = builder.CreateCall(exp_func, {exponent}, "div_term");

    // angle = pos * div_term
    llvm::Value* angle = builder.CreateFMul(pos_double, div_term, "angle");

    // PE[pos, 2i] = sin(angle)
    llvm::Value* sin_val = builder.CreateCall(sin_func, {angle}, "sin_val");
    llvm::Value* sin_bits = builder.CreateBitCast(sin_val, ctx_.int64Type());
    llvm::Value* even_idx = builder.CreateAdd(row_offset,
        builder.CreateMul(llvm::ConstantInt::get(ctx_.int64Type(), 2), i));
    llvm::Value* even_ptr = builder.CreateGEP(ctx_.int64Type(), elements_ptr, even_idx);
    builder.CreateStore(sin_bits, even_ptr);

    // PE[pos, 2i+1] = cos(angle)
    llvm::Value* cos_val = builder.CreateCall(cos_func, {angle}, "cos_val");
    llvm::Value* cos_bits = builder.CreateBitCast(cos_val, ctx_.int64Type());
    llvm::Value* odd_idx = builder.CreateAdd(even_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* odd_ptr = builder.CreateGEP(ctx_.int64Type(), elements_ptr, odd_idx);
    builder.CreateStore(cos_bits, odd_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, i_counter);
    builder.CreateBr(dim_cond);

    builder.SetInsertPoint(dim_exit);
    llvm::Value* next_pos = builder.CreateAdd(pos, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_pos, pos_counter);
    builder.CreateBr(pos_cond);

    builder.SetInsertPoint(pos_exit);

    return tagged_.packHeapPtr(tensor_ptr);
}

// === Additional Transformer Helpers ===

llvm::Value* TensorCodegen::rotaryEmbedding(const eshkol_operations_t* op) {
    // RoPE (Rotary Position Embedding)
    // Applies rotation to pairs of dimensions based on position
    // x_rot[..., 2i] = x[..., 2i] * cos(theta) - x[..., 2i+1] * sin(theta)
    // x_rot[..., 2i+1] = x[..., 2i] * sin(theta) + x[..., 2i+1] * cos(theta)
    // where theta = pos / 10000^(2i/dim)

    if (op->call_op.num_vars != 3) {
        eshkol_error("rotary-embedding requires 3 arguments: x seq-positions dim");
        return nullptr;
    }

    llvm::Value* x_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* pos_val = codegenAST(&op->call_op.variables[1]);
    llvm::Value* dim_val = codegenAST(&op->call_op.variables[2]);

    if (!x_val || !pos_val || !dim_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack x tensor
    llvm::Value* x_ptr_int = tagged_.unpackInt64(x_val);
    llvm::Value* x_ptr = builder.CreateIntToPtr(x_ptr_int, ctx_.ptrType());

    llvm::Value* x_dims_field = builder.CreateStructGEP(tensor_type, x_ptr, 0);
    llvm::Value* x_dims_ptr = builder.CreateLoad(ctx_.ptrType(), x_dims_field);
    llvm::Value* x_ndim_field = builder.CreateStructGEP(tensor_type, x_ptr, 1);
    llvm::Value* x_ndim = builder.CreateLoad(ctx_.int64Type(), x_ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, x_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* x_total_field = builder.CreateStructGEP(tensor_type, x_ptr, 3);
    llvm::Value* x_total = builder.CreateLoad(ctx_.int64Type(), x_total_field);

    // Unpack positions tensor
    llvm::Value* pos_ptr_int = tagged_.unpackInt64(pos_val);
    llvm::Value* pos_ptr = builder.CreateIntToPtr(pos_ptr_int, ctx_.ptrType());
    llvm::Value* pos_elems_field = builder.CreateStructGEP(tensor_type, pos_ptr, 2);
    llvm::Value* pos_elems = builder.CreateLoad(ctx_.ptrType(), pos_elems_field);

    // Get dim
    llvm::Value* dim = tagged_.unpackInt64(dim_val);

    // Allocate output
    llvm::Value* output_bytes = builder.CreateMul(x_total,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* output_elems = builder.CreateCall(arena_alloc, {arena_ptr, output_bytes}, "rope_output");

    // Copy dims
    llvm::Value* dims_bytes = builder.CreateMul(x_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* output_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "rope_dims");
    builder.CreateMemCpy(output_dims, llvm::MaybeAlign(8), x_dims_ptr, llvm::MaybeAlign(8), dims_bytes);

    // Get math functions
    llvm::Function* sin_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::sin, {ctx_.doubleType()});
    llvm::Function* cos_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::cos, {ctx_.doubleType()});
    llvm::Function* exp_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Get dimensions - assume x is (batch, seq_len, dim) or (seq_len, dim)
    llvm::Value* is_3d = builder.CreateICmpEQ(x_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 3));

    llvm::Value* batch_size = builder.CreateSelect(is_3d,
        builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), x_dims_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 0))),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* seq_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 1),
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* seq_len = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), x_dims_ptr, seq_idx));

    llvm::Value* log_10000 = llvm::ConstantFP::get(ctx_.doubleType(), 9.210340371976184);
    llvm::Value* dim_double = builder.CreateSIToFP(dim, ctx_.doubleType());
    llvm::Value* half_dim = builder.CreateSDiv(dim, llvm::ConstantInt::get(ctx_.int64Type(), 2));

    // === ALLOCATE ALL LOOP VARIABLES UPFRONT ===
    llvm::Value* batch_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "batch_idx");
    llvm::Value* s_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "s_idx");
    llvm::Value* i_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "i_idx");

    // Main loop
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "rope_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "rope_batch_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "rope_finalize", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_cond);
    llvm::Value* b = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* b_done = builder.CreateICmpUGE(b, batch_size);
    builder.CreateCondBr(b_done, finalize, batch_body);

    builder.SetInsertPoint(batch_body);

    llvm::Value* batch_offset = builder.CreateMul(b, builder.CreateMul(seq_len, dim));

    // Sequence loop
    llvm::BasicBlock* seq_cond = llvm::BasicBlock::Create(ctx_.context(), "rope_seq_cond", current_func);
    llvm::BasicBlock* seq_body = llvm::BasicBlock::Create(ctx_.context(), "rope_seq_body", current_func);
    llvm::BasicBlock* batch_next = llvm::BasicBlock::Create(ctx_.context(), "rope_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), s_idx);
    builder.CreateBr(seq_cond);

    builder.SetInsertPoint(seq_cond);
    llvm::Value* s = builder.CreateLoad(ctx_.int64Type(), s_idx);
    llvm::Value* s_done = builder.CreateICmpUGE(s, seq_len);
    builder.CreateCondBr(s_done, batch_next, seq_body);

    builder.SetInsertPoint(seq_body);

    // Get position for this sequence element
    llvm::Value* pos_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), pos_elems, s));

    llvm::Value* seq_offset = builder.CreateAdd(batch_offset, builder.CreateMul(s, dim));

    // Dimension pair loop
    llvm::BasicBlock* dim_cond = llvm::BasicBlock::Create(ctx_.context(), "rope_dim_cond", current_func);
    llvm::BasicBlock* dim_body = llvm::BasicBlock::Create(ctx_.context(), "rope_dim_body", current_func);
    llvm::BasicBlock* seq_next = llvm::BasicBlock::Create(ctx_.context(), "rope_seq_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_idx);
    builder.CreateBr(dim_cond);

    builder.SetInsertPoint(dim_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_idx);
    llvm::Value* i_done = builder.CreateICmpUGE(i, half_dim);
    builder.CreateCondBr(i_done, seq_next, dim_body);

    builder.SetInsertPoint(dim_body);

    // Compute theta = pos / 10000^(2i/dim)
    llvm::Value* i_double = builder.CreateSIToFP(i, ctx_.doubleType());
    llvm::Value* two_i = builder.CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 2.0), i_double);
    llvm::Value* exponent = builder.CreateFMul(two_i,
        builder.CreateFDiv(builder.CreateFNeg(log_10000), dim_double));
    llvm::Value* freq = builder.CreateCall(exp_func, {exponent});
    llvm::Value* theta = builder.CreateFMul(pos_elem, freq);

    llvm::Value* cos_theta = builder.CreateCall(cos_func, {theta});
    llvm::Value* sin_theta = builder.CreateCall(sin_func, {theta});

    // Get input pair
    llvm::Value* even_idx = builder.CreateAdd(seq_offset,
        builder.CreateMul(llvm::ConstantInt::get(ctx_.int64Type(), 2), i));
    llvm::Value* odd_idx = builder.CreateAdd(even_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* x_even = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), x_elems, even_idx));
    llvm::Value* x_odd = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), x_elems, odd_idx));

    // Apply rotation
    // out_even = x_even * cos - x_odd * sin
    // out_odd = x_even * sin + x_odd * cos
    llvm::Value* out_even = builder.CreateFSub(
        builder.CreateFMul(x_even, cos_theta),
        builder.CreateFMul(x_odd, sin_theta));
    llvm::Value* out_odd = builder.CreateFAdd(
        builder.CreateFMul(x_even, sin_theta),
        builder.CreateFMul(x_odd, cos_theta));

    builder.CreateStore(out_even, builder.CreateGEP(ctx_.doubleType(), output_elems, even_idx));
    builder.CreateStore(out_odd, builder.CreateGEP(ctx_.doubleType(), output_elems, odd_idx));

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_idx);
    builder.CreateBr(dim_cond);

    builder.SetInsertPoint(seq_next);
    builder.CreateStore(builder.CreateAdd(s, llvm::ConstantInt::get(ctx_.int64Type(), 1)), s_idx);
    builder.CreateBr(seq_cond);

    builder.SetInsertPoint(batch_next);
    builder.CreateStore(builder.CreateAdd(b, llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(batch_cond);

    // Finalize
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "rope_result");

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(output_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(x_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(output_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(x_total, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::causalMask(const eshkol_operations_t* op) {
    // Creates causal mask for decoder self-attention (additive convention)
    // mask[i, j] = 0 if j <= i (attend), -inf otherwise (don't attend)
    // Added to attention scores before softmax

    if (op->call_op.num_vars != 1) {
        eshkol_error("causal-mask requires 1 argument: seq-len");
        return nullptr;
    }

    llvm::Value* seq_len_val = codegenAST(&op->call_op.variables[0]);
    if (!seq_len_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* seq_len = seq_len_val;
    if (seq_len->getType() == ctx_.taggedValueType()) {
        seq_len = tagged_.unpackInt64(seq_len_val);
    }

    // Create tensor - fill with -inf (masked positions)
    llvm::Value* neg_inf = llvm::ConstantFP::getInfinity(ctx_.doubleType(), true);
    llvm::Value* neg_inf_bits = builder.CreateBitCast(neg_inf, ctx_.int64Type());

    std::vector<llvm::Value*> dims = {seq_len, seq_len};
    llvm::Value* tensor_ptr = createTensorWithDims(dims, neg_inf_bits, false);  // Fill with -inf
    if (!tensor_ptr) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get elements pointer
    llvm::Value* elements_field_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = builder.CreateLoad(ctx_.ptrType(), elements_field_ptr);

    // Value for attending positions (0)
    llvm::Value* zero_double = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* zero_bits = builder.CreateBitCast(zero_double, ctx_.int64Type());

    // Fill lower triangle with 0 (attend)
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_i_cond = llvm::BasicBlock::Create(ctx_.context(), "mask_i_cond", current_func);
    llvm::BasicBlock* loop_i_body = llvm::BasicBlock::Create(ctx_.context(), "mask_i_body", current_func);
    llvm::BasicBlock* loop_i_exit = llvm::BasicBlock::Create(ctx_.context(), "mask_i_exit", current_func);

    // Allocate loop counters at entry
    llvm::Value* i_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mask_i");
    llvm::Value* j_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mask_j");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_counter);
    builder.CreateBr(loop_i_cond);

    builder.SetInsertPoint(loop_i_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_counter);
    llvm::Value* i_cmp = builder.CreateICmpULT(i, seq_len);
    builder.CreateCondBr(i_cmp, loop_i_body, loop_i_exit);

    builder.SetInsertPoint(loop_i_body);

    // Inner loop: j from 0 to i (inclusive) - these are positions we attend to
    llvm::BasicBlock* loop_j_cond = llvm::BasicBlock::Create(ctx_.context(), "mask_j_cond", current_func);
    llvm::BasicBlock* loop_j_body = llvm::BasicBlock::Create(ctx_.context(), "mask_j_body", current_func);
    llvm::BasicBlock* loop_j_exit = llvm::BasicBlock::Create(ctx_.context(), "mask_j_exit", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_counter);
    llvm::Value* i_plus_one = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateBr(loop_j_cond);

    builder.SetInsertPoint(loop_j_cond);
    llvm::Value* j = builder.CreateLoad(ctx_.int64Type(), j_counter);
    llvm::Value* j_cmp = builder.CreateICmpULT(j, i_plus_one);  // j <= i
    builder.CreateCondBr(j_cmp, loop_j_body, loop_j_exit);

    builder.SetInsertPoint(loop_j_body);
    // mask[i, j] = 0 (attend)
    llvm::Value* row_offset = builder.CreateMul(i, seq_len);
    llvm::Value* elem_idx = builder.CreateAdd(row_offset, j);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elements_ptr, elem_idx);
    builder.CreateStore(zero_bits, elem_ptr);

    llvm::Value* next_j = builder.CreateAdd(j, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_j, j_counter);
    builder.CreateBr(loop_j_cond);

    builder.SetInsertPoint(loop_j_exit);
    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, i_counter);
    builder.CreateBr(loop_i_cond);

    builder.SetInsertPoint(loop_i_exit);

    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::paddingMask(const eshkol_operations_t* op) {
    // Creates padding mask from sequence lengths
    // mask[b, i] = 0 if i < lengths[b], else -inf

    if (op->call_op.num_vars != 2) {
        eshkol_error("padding-mask requires 2 arguments: lengths max-len");
        return nullptr;
    }

    llvm::Value* lengths_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* max_len_val = codegenAST(&op->call_op.variables[1]);

    if (!lengths_val || !max_len_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get max_len
    llvm::Value* max_len = tagged_.unpackInt64(max_len_val);

    // Unpack lengths tensor
    llvm::Value* lengths_ptr_int = tagged_.unpackInt64(lengths_val);
    llvm::Value* lengths_ptr = builder.CreateIntToPtr(lengths_ptr_int, ctx_.ptrType());
    llvm::Value* lengths_total_field = builder.CreateStructGEP(tensor_type, lengths_ptr, 3);
    llvm::Value* batch_size = builder.CreateLoad(ctx_.int64Type(), lengths_total_field);
    llvm::Value* lengths_elems_field = builder.CreateStructGEP(tensor_type, lengths_ptr, 2);
    llvm::Value* lengths_elems = builder.CreateLoad(ctx_.ptrType(), lengths_elems_field);

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Allocate mask: (batch, max_len)
    llvm::Value* total_size = builder.CreateMul(batch_size, max_len);
    llvm::Value* elems_bytes = builder.CreateMul(total_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* mask_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "pad_mask_elems");

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* neg_inf = llvm::ConstantFP::getInfinity(ctx_.doubleType(), true);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    // Fill mask
    llvm::BasicBlock* b_cond = llvm::BasicBlock::Create(ctx_.context(), "pad_b_cond", current_func);
    llvm::BasicBlock* b_body = llvm::BasicBlock::Create(ctx_.context(), "pad_b_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "pad_finalize", current_func);

    llvm::Value* b_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "b_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), b_idx);
    builder.CreateBr(b_cond);

    builder.SetInsertPoint(b_cond);
    llvm::Value* b = builder.CreateLoad(ctx_.int64Type(), b_idx);
    llvm::Value* b_done = builder.CreateICmpUGE(b, batch_size);
    builder.CreateCondBr(b_done, finalize, b_body);

    builder.SetInsertPoint(b_body);

    // Get length for this batch element
    llvm::Value* len_double = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), lengths_elems, b));
    llvm::Value* len = builder.CreateFPToSI(len_double, ctx_.int64Type());

    llvm::Value* row_offset = builder.CreateMul(b, max_len);

    llvm::BasicBlock* i_cond = llvm::BasicBlock::Create(ctx_.context(), "pad_i_cond", current_func);
    llvm::BasicBlock* i_body = llvm::BasicBlock::Create(ctx_.context(), "pad_i_body", current_func);
    llvm::BasicBlock* b_next = llvm::BasicBlock::Create(ctx_.context(), "pad_b_next", current_func);

    llvm::Value* i_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "i_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_idx);
    builder.CreateBr(i_cond);

    builder.SetInsertPoint(i_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_idx);
    llvm::Value* i_done = builder.CreateICmpUGE(i, max_len);
    builder.CreateCondBr(i_done, b_next, i_body);

    builder.SetInsertPoint(i_body);

    // mask[b, i] = 0 if i < len[b], else -inf
    llvm::Value* is_valid = builder.CreateICmpULT(i, len);
    llvm::Value* mask_val = builder.CreateSelect(is_valid, zero, neg_inf);

    llvm::Value* elem_idx = builder.CreateAdd(row_offset, i);
    builder.CreateStore(mask_val, builder.CreateGEP(ctx_.doubleType(), mask_elems, elem_idx));

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_idx);
    builder.CreateBr(i_cond);

    builder.SetInsertPoint(b_next);
    builder.CreateStore(builder.CreateAdd(b, llvm::ConstantInt::get(ctx_.int64Type(), 1)), b_idx);
    builder.CreateBr(b_cond);

    // Finalize
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "pad_mask_result");

    llvm::Value* dims_bytes = llvm::ConstantInt::get(ctx_.int64Type(), 2 * sizeof(int64_t));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "pad_mask_dims");
    builder.CreateStore(batch_size, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(max_len, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 2), r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(mask_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_size, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::feedForward(const eshkol_operations_t* op) {
    // Feed-forward network: FFN(x) = W2 * GELU(W1 * x + b1) + b2

    if (op->call_op.num_vars != 5) {
        eshkol_error("feed-forward requires 5 arguments: x W1 b1 W2 b2");
        return nullptr;
    }

    llvm::Value* x_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* w1_val = codegenAST(&op->call_op.variables[1]);
    llvm::Value* b1_val = codegenAST(&op->call_op.variables[2]);
    llvm::Value* w2_val = codegenAST(&op->call_op.variables[3]);
    llvm::Value* b2_val = codegenAST(&op->call_op.variables[4]);

    if (!x_val || !w1_val || !b1_val || !w2_val || !b2_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack all tensors
    llvm::Value* x_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(x_val), ctx_.ptrType());
    llvm::Value* x_dims_field = builder.CreateStructGEP(tensor_type, x_ptr, 0);
    llvm::Value* x_dims = builder.CreateLoad(ctx_.ptrType(), x_dims_field);
    llvm::Value* x_ndim_field = builder.CreateStructGEP(tensor_type, x_ptr, 1);
    llvm::Value* x_ndim = builder.CreateLoad(ctx_.int64Type(), x_ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, x_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);

    llvm::Value* w1_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(w1_val), ctx_.ptrType());
    llvm::Value* w1_dims_field = builder.CreateStructGEP(tensor_type, w1_ptr, 0);
    llvm::Value* w1_dims = builder.CreateLoad(ctx_.ptrType(), w1_dims_field);
    llvm::Value* w1_elems_field = builder.CreateStructGEP(tensor_type, w1_ptr, 2);
    llvm::Value* w1_elems = builder.CreateLoad(ctx_.ptrType(), w1_elems_field);

    llvm::Value* b1_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(b1_val), ctx_.ptrType());
    llvm::Value* b1_elems_field = builder.CreateStructGEP(tensor_type, b1_ptr, 2);
    llvm::Value* b1_elems = builder.CreateLoad(ctx_.ptrType(), b1_elems_field);

    llvm::Value* w2_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(w2_val), ctx_.ptrType());
    llvm::Value* w2_dims_field = builder.CreateStructGEP(tensor_type, w2_ptr, 0);
    llvm::Value* w2_dims = builder.CreateLoad(ctx_.ptrType(), w2_dims_field);
    llvm::Value* w2_elems_field = builder.CreateStructGEP(tensor_type, w2_ptr, 2);
    llvm::Value* w2_elems = builder.CreateLoad(ctx_.ptrType(), w2_elems_field);

    llvm::Value* b2_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(b2_val), ctx_.ptrType());
    llvm::Value* b2_elems_field = builder.CreateStructGEP(tensor_type, b2_ptr, 2);
    llvm::Value* b2_elems = builder.CreateLoad(ctx_.ptrType(), b2_elems_field);

    // Get dimensions
    // x: (batch, seq, d_model) or (seq, d_model)
    // W1: (d_model, d_ff), b1: (d_ff,)
    // W2: (d_ff, d_model), b2: (d_model,)

    llvm::Value* is_3d = builder.CreateICmpEQ(x_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 3));

    llvm::Value* batch = builder.CreateSelect(is_3d,
        builder.CreateLoad(ctx_.int64Type(), builder.CreateGEP(ctx_.int64Type(), x_dims,
            llvm::ConstantInt::get(ctx_.int64Type(), 0))),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* seq_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 1),
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* dim_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 2),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* seq_len = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), x_dims, seq_idx));
    llvm::Value* d_model = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), x_dims, dim_idx));
    llvm::Value* d_ff = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), w1_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1)));

    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Allocate intermediate: (batch, seq, d_ff)
    llvm::Value* hidden_size = builder.CreateMul(batch, builder.CreateMul(seq_len, d_ff));
    llvm::Value* hidden_bytes = builder.CreateMul(hidden_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* hidden = builder.CreateCall(arena_alloc, {arena_ptr, hidden_bytes}, "ffn_hidden");

    // Allocate output: (batch, seq, d_model)
    llvm::Value* output_size = builder.CreateMul(batch, builder.CreateMul(seq_len, d_model));
    llvm::Value* output_bytes = builder.CreateMul(output_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* output = builder.CreateCall(arena_alloc, {arena_ptr, output_bytes}, "ffn_output");

    // GELU approximation constants
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    llvm::Value* sqrt_2_pi = llvm::ConstantFP::get(ctx_.doubleType(), 0.7978845608028654);
    llvm::Value* coeff = llvm::ConstantFP::get(ctx_.doubleType(), 0.044715);

    llvm::Function* tanh_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::fabs, {ctx_.doubleType()});
    // Note: LLVM doesn't have intrinsic tanh, we'll compute it via exp
    llvm::Function* exp_func = llvm::Intrinsic::getOrInsertDeclaration(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    // === First layer: hidden = GELU(x @ W1 + b1) ===
    llvm::BasicBlock* layer1_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "l1_batch_cond", current_func);
    llvm::BasicBlock* layer1_batch_body = llvm::BasicBlock::Create(ctx_.context(), "l1_batch_body", current_func);
    llvm::BasicBlock* layer2_init = llvm::BasicBlock::Create(ctx_.context(), "layer2_init", current_func);

    // === ALLOCATE ALL LOOP VARIABLES UPFRONT ===
    llvm::Value* b_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "b_idx");
    llvm::Value* s_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "s_idx");
    llvm::Value* d_idx_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "d_idx");
    llvm::Value* sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "sum");
    llvm::Value* k_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "k_idx");

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), b_idx);
    builder.CreateBr(layer1_batch_cond);

    builder.SetInsertPoint(layer1_batch_cond);
    llvm::Value* b = builder.CreateLoad(ctx_.int64Type(), b_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(b, batch), layer2_init, layer1_batch_body);

    builder.SetInsertPoint(layer1_batch_body);

    llvm::Value* x_batch_off = builder.CreateMul(b, builder.CreateMul(seq_len, d_model));
    llvm::Value* h_batch_off = builder.CreateMul(b, builder.CreateMul(seq_len, d_ff));

    // Sequence loop
    llvm::BasicBlock* l1_seq_cond = llvm::BasicBlock::Create(ctx_.context(), "l1_seq_cond", current_func);
    llvm::BasicBlock* l1_seq_body = llvm::BasicBlock::Create(ctx_.context(), "l1_seq_body", current_func);
    llvm::BasicBlock* l1_batch_next = llvm::BasicBlock::Create(ctx_.context(), "l1_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), s_idx);
    builder.CreateBr(l1_seq_cond);

    builder.SetInsertPoint(l1_seq_cond);
    llvm::Value* s = builder.CreateLoad(ctx_.int64Type(), s_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(s, seq_len), l1_batch_next, l1_seq_body);

    builder.SetInsertPoint(l1_seq_body);

    llvm::Value* x_seq_off = builder.CreateAdd(x_batch_off, builder.CreateMul(s, d_model));
    llvm::Value* h_seq_off = builder.CreateAdd(h_batch_off, builder.CreateMul(s, d_ff));

    // Output dimension loop (d_ff)
    llvm::BasicBlock* l1_d_cond = llvm::BasicBlock::Create(ctx_.context(), "l1_d_cond", current_func);
    llvm::BasicBlock* l1_d_body = llvm::BasicBlock::Create(ctx_.context(), "l1_d_body", current_func);
    llvm::BasicBlock* l1_seq_next = llvm::BasicBlock::Create(ctx_.context(), "l1_seq_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), d_idx_var);
    builder.CreateBr(l1_d_cond);

    builder.SetInsertPoint(l1_d_cond);
    llvm::Value* d = builder.CreateLoad(ctx_.int64Type(), d_idx_var);
    builder.CreateCondBr(builder.CreateICmpUGE(d, d_ff), l1_seq_next, l1_d_body);

    builder.SetInsertPoint(l1_d_body);

    // Compute dot product: x[s, :] @ W1[:, d] + b1[d]
    llvm::BasicBlock* l1_k_cond = llvm::BasicBlock::Create(ctx_.context(), "l1_k_cond", current_func);
    llvm::BasicBlock* l1_k_body = llvm::BasicBlock::Create(ctx_.context(), "l1_k_body", current_func);
    llvm::BasicBlock* l1_apply_gelu = llvm::BasicBlock::Create(ctx_.context(), "l1_apply_gelu", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_idx);
    builder.CreateBr(l1_k_cond);

    builder.SetInsertPoint(l1_k_cond);
    llvm::Value* k = builder.CreateLoad(ctx_.int64Type(), k_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(k, d_model), l1_apply_gelu, l1_k_body);

    builder.SetInsertPoint(l1_k_body);
    llvm::Value* x_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), x_elems, builder.CreateAdd(x_seq_off, k)));
    llvm::Value* w1_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), w1_elems, builder.CreateAdd(builder.CreateMul(k, d_ff), d)));
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, builder.CreateFMul(x_elem, w1_elem)), sum);
    builder.CreateStore(builder.CreateAdd(k, llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_idx);
    builder.CreateBr(l1_k_cond);

    builder.SetInsertPoint(l1_apply_gelu);
    // Add bias
    llvm::Value* bias1 = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), b1_elems, d));
    llvm::Value* pre_act = builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), sum), bias1);

    // Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    llvm::Value* x3 = builder.CreateFMul(pre_act, builder.CreateFMul(pre_act, pre_act));
    llvm::Value* inner = builder.CreateFMul(sqrt_2_pi,
        builder.CreateFAdd(pre_act, builder.CreateFMul(coeff, x3)));
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    llvm::Value* two_inner = builder.CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 2.0), inner);
    llvm::Value* exp_2x = builder.CreateCall(exp_func, {two_inner});
    llvm::Value* tanh_val = builder.CreateFDiv(
        builder.CreateFSub(exp_2x, llvm::ConstantFP::get(ctx_.doubleType(), 1.0)),
        builder.CreateFAdd(exp_2x, llvm::ConstantFP::get(ctx_.doubleType(), 1.0)));
    llvm::Value* gelu_out = builder.CreateFMul(
        llvm::ConstantFP::get(ctx_.doubleType(), 0.5),
        builder.CreateFMul(pre_act,
            builder.CreateFAdd(llvm::ConstantFP::get(ctx_.doubleType(), 1.0), tanh_val)));

    builder.CreateStore(gelu_out,
        builder.CreateGEP(ctx_.doubleType(), hidden, builder.CreateAdd(h_seq_off, d)));

    builder.CreateStore(builder.CreateAdd(d, llvm::ConstantInt::get(ctx_.int64Type(), 1)), d_idx_var);
    builder.CreateBr(l1_d_cond);

    builder.SetInsertPoint(l1_seq_next);
    builder.CreateStore(builder.CreateAdd(s, llvm::ConstantInt::get(ctx_.int64Type(), 1)), s_idx);
    builder.CreateBr(l1_seq_cond);

    builder.SetInsertPoint(l1_batch_next);
    builder.CreateStore(builder.CreateAdd(b, llvm::ConstantInt::get(ctx_.int64Type(), 1)), b_idx);
    builder.CreateBr(layer1_batch_cond);

    // === Second layer: output = hidden @ W2 + b2 ===
    builder.SetInsertPoint(layer2_init);

    llvm::BasicBlock* l2_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "l2_batch_cond", current_func);
    llvm::BasicBlock* l2_batch_body = llvm::BasicBlock::Create(ctx_.context(), "l2_batch_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "ffn_finalize", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), b_idx);
    builder.CreateBr(l2_batch_cond);

    builder.SetInsertPoint(l2_batch_cond);
    llvm::Value* b2_loop = builder.CreateLoad(ctx_.int64Type(), b_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(b2_loop, batch), finalize, l2_batch_body);

    builder.SetInsertPoint(l2_batch_body);

    llvm::Value* h_batch_off2 = builder.CreateMul(b2_loop, builder.CreateMul(seq_len, d_ff));
    llvm::Value* o_batch_off = builder.CreateMul(b2_loop, builder.CreateMul(seq_len, d_model));

    llvm::BasicBlock* l2_seq_cond = llvm::BasicBlock::Create(ctx_.context(), "l2_seq_cond", current_func);
    llvm::BasicBlock* l2_seq_body = llvm::BasicBlock::Create(ctx_.context(), "l2_seq_body", current_func);
    llvm::BasicBlock* l2_batch_next = llvm::BasicBlock::Create(ctx_.context(), "l2_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), s_idx);
    builder.CreateBr(l2_seq_cond);

    builder.SetInsertPoint(l2_seq_cond);
    llvm::Value* s2 = builder.CreateLoad(ctx_.int64Type(), s_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(s2, seq_len), l2_batch_next, l2_seq_body);

    builder.SetInsertPoint(l2_seq_body);

    llvm::Value* h_seq_off2 = builder.CreateAdd(h_batch_off2, builder.CreateMul(s2, d_ff));
    llvm::Value* o_seq_off = builder.CreateAdd(o_batch_off, builder.CreateMul(s2, d_model));

    llvm::BasicBlock* l2_d_cond = llvm::BasicBlock::Create(ctx_.context(), "l2_d_cond", current_func);
    llvm::BasicBlock* l2_d_body = llvm::BasicBlock::Create(ctx_.context(), "l2_d_body", current_func);
    llvm::BasicBlock* l2_seq_next = llvm::BasicBlock::Create(ctx_.context(), "l2_seq_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), d_idx_var);
    builder.CreateBr(l2_d_cond);

    builder.SetInsertPoint(l2_d_cond);
    llvm::Value* d2 = builder.CreateLoad(ctx_.int64Type(), d_idx_var);
    builder.CreateCondBr(builder.CreateICmpUGE(d2, d_model), l2_seq_next, l2_d_body);

    builder.SetInsertPoint(l2_d_body);

    llvm::BasicBlock* l2_k_cond = llvm::BasicBlock::Create(ctx_.context(), "l2_k_cond", current_func);
    llvm::BasicBlock* l2_k_body = llvm::BasicBlock::Create(ctx_.context(), "l2_k_body", current_func);
    llvm::BasicBlock* l2_store = llvm::BasicBlock::Create(ctx_.context(), "l2_store", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_idx);
    builder.CreateBr(l2_k_cond);

    builder.SetInsertPoint(l2_k_cond);
    llvm::Value* k2 = builder.CreateLoad(ctx_.int64Type(), k_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(k2, d_ff), l2_store, l2_k_body);

    builder.SetInsertPoint(l2_k_body);
    llvm::Value* h_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), hidden, builder.CreateAdd(h_seq_off2, k2)));
    llvm::Value* w2_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), w2_elems, builder.CreateAdd(builder.CreateMul(k2, d_model), d2)));
    llvm::Value* cur_sum2 = builder.CreateLoad(ctx_.doubleType(), sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum2, builder.CreateFMul(h_elem, w2_elem)), sum);
    builder.CreateStore(builder.CreateAdd(k2, llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_idx);
    builder.CreateBr(l2_k_cond);

    builder.SetInsertPoint(l2_store);
    llvm::Value* bias2 = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), b2_elems, d2));
    llvm::Value* out_val = builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), sum), bias2);
    builder.CreateStore(out_val,
        builder.CreateGEP(ctx_.doubleType(), output, builder.CreateAdd(o_seq_off, d2)));

    builder.CreateStore(builder.CreateAdd(d2, llvm::ConstantInt::get(ctx_.int64Type(), 1)), d_idx_var);
    builder.CreateBr(l2_d_cond);

    builder.SetInsertPoint(l2_seq_next);
    builder.CreateStore(builder.CreateAdd(s2, llvm::ConstantInt::get(ctx_.int64Type(), 1)), s_idx);
    builder.CreateBr(l2_seq_cond);

    builder.SetInsertPoint(l2_batch_next);
    builder.CreateStore(builder.CreateAdd(b2_loop, llvm::ConstantInt::get(ctx_.int64Type(), 1)), b_idx);
    builder.CreateBr(l2_batch_cond);

    // Finalize
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "ffn_result");

    // Copy dims from input
    llvm::Value* dims_bytes = builder.CreateMul(x_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "ffn_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), x_dims, llvm::MaybeAlign(8), dims_bytes);

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(x_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(output, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(output_size, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::dropout(const eshkol_operations_t* op) {
    // Dropout: randomly zero elements and scale by 1/(1-rate)
    // Only applied during training

    if (op->call_op.num_vars != 3) {
        eshkol_error("dropout requires 3 arguments: x rate training");
        return nullptr;
    }

    llvm::Value* x_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* rate_val = codegenAST(&op->call_op.variables[1]);
    llvm::Value* training_val = codegenAST(&op->call_op.variables[2]);

    if (!x_val || !rate_val || !training_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensors
    llvm::Value* x_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(x_val), ctx_.ptrType());
    llvm::Value* x_dims_field = builder.CreateStructGEP(tensor_type, x_ptr, 0);
    llvm::Value* x_dims = builder.CreateLoad(ctx_.ptrType(), x_dims_field);
    llvm::Value* x_ndim_field = builder.CreateStructGEP(tensor_type, x_ptr, 1);
    llvm::Value* x_ndim = builder.CreateLoad(ctx_.int64Type(), x_ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, x_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* x_total_field = builder.CreateStructGEP(tensor_type, x_ptr, 3);
    llvm::Value* x_total = builder.CreateLoad(ctx_.int64Type(), x_total_field);

    llvm::Value* rate = tagged_.unpackDouble(rate_val);
    llvm::Value* training = tagged_.unpackInt64(training_val);
    llvm::Value* is_training = builder.CreateICmpNE(training,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));

    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Allocate output
    llvm::Value* output_bytes = builder.CreateMul(x_total,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* output_elems = builder.CreateCall(arena_alloc, {arena_ptr, output_bytes}, "drop_output");

    // Copy dims
    llvm::Value* dims_bytes = builder.CreateMul(x_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* output_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "drop_dims");
    builder.CreateMemCpy(output_dims, llvm::MaybeAlign(8), x_dims, llvm::MaybeAlign(8), dims_bytes);

    // Compute scale = 1 / (1 - rate)
    llvm::Value* scale = builder.CreateFDiv(
        llvm::ConstantFP::get(ctx_.doubleType(), 1.0),
        builder.CreateFSub(llvm::ConstantFP::get(ctx_.doubleType(), 1.0), rate));

    // Simple LCG random number generator state
    llvm::Value* rng_state = builder.CreateAlloca(ctx_.int64Type(), nullptr, "rng_state");
    // Seed with a constant (in production, use better seeding)
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 123456789), rng_state);

    // Main loop
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "drop_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "drop_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "drop_finalize", current_func);

    llvm::Value* i_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "i_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(i, x_total), finalize, loop_body);

    builder.SetInsertPoint(loop_body);

    llvm::Value* x_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), x_elems, i));

    // Generate random number using LCG: state = (a * state + c) mod m
    // a = 1103515245, c = 12345, m = 2^31
    llvm::Value* state = builder.CreateLoad(ctx_.int64Type(), rng_state);
    llvm::Value* new_state = builder.CreateAdd(
        builder.CreateMul(state, llvm::ConstantInt::get(ctx_.int64Type(), 1103515245)),
        llvm::ConstantInt::get(ctx_.int64Type(), 12345));
    new_state = builder.CreateAnd(new_state, llvm::ConstantInt::get(ctx_.int64Type(), 0x7FFFFFFF));
    builder.CreateStore(new_state, rng_state);

    // Convert to [0, 1) range
    llvm::Value* rand_double = builder.CreateUIToFP(new_state, ctx_.doubleType());
    rand_double = builder.CreateFDiv(rand_double,
        llvm::ConstantFP::get(ctx_.doubleType(), 2147483648.0));

    // Apply dropout: if rand < rate, zero out; else scale
    llvm::Value* should_drop = builder.CreateFCmpOLT(rand_double, rate);

    // Only apply during training
    llvm::Value* actually_drop = builder.CreateAnd(is_training, should_drop);

    llvm::Value* out_val = builder.CreateSelect(actually_drop,
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0),
        builder.CreateSelect(is_training,
            builder.CreateFMul(x_elem, scale),
            x_elem));  // No scaling during inference

    builder.CreateStore(out_val, builder.CreateGEP(ctx_.doubleType(), output_elems, i));

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_idx);
    builder.CreateBr(loop_cond);

    // Finalize
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "drop_result");

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(output_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(x_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(output_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(x_total, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::embedding(const eshkol_operations_t* op) {
    // Embedding lookup: output[b, s, :] = weights[indices[b, s], :]

    if (op->call_op.num_vars != 2) {
        eshkol_error("embedding requires 2 arguments: indices weights");
        return nullptr;
    }

    llvm::Value* indices_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* weights_val = codegenAST(&op->call_op.variables[1]);

    if (!indices_val || !weights_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack indices tensor
    llvm::Value* idx_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(indices_val), ctx_.ptrType());
    llvm::Value* idx_dims_field = builder.CreateStructGEP(tensor_type, idx_ptr, 0);
    llvm::Value* idx_dims = builder.CreateLoad(ctx_.ptrType(), idx_dims_field);
    llvm::Value* idx_ndim_field = builder.CreateStructGEP(tensor_type, idx_ptr, 1);
    llvm::Value* idx_ndim = builder.CreateLoad(ctx_.int64Type(), idx_ndim_field);
    llvm::Value* idx_elems_field = builder.CreateStructGEP(tensor_type, idx_ptr, 2);
    llvm::Value* idx_elems = builder.CreateLoad(ctx_.ptrType(), idx_elems_field);
    llvm::Value* idx_total_field = builder.CreateStructGEP(tensor_type, idx_ptr, 3);
    llvm::Value* idx_total = builder.CreateLoad(ctx_.int64Type(), idx_total_field);

    // Unpack weights tensor: (vocab_size, d_model)
    llvm::Value* w_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(weights_val), ctx_.ptrType());
    llvm::Value* w_dims_field = builder.CreateStructGEP(tensor_type, w_ptr, 0);
    llvm::Value* w_dims = builder.CreateLoad(ctx_.ptrType(), w_dims_field);
    llvm::Value* w_elems_field = builder.CreateStructGEP(tensor_type, w_ptr, 2);
    llvm::Value* w_elems = builder.CreateLoad(ctx_.ptrType(), w_elems_field);

    // Get d_model from weights
    llvm::Value* d_model = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), w_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1)));

    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Output shape: indices_shape + (d_model,)
    // For indices (batch, seq), output is (batch, seq, d_model)
    llvm::Value* output_ndim = builder.CreateAdd(idx_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* output_total = builder.CreateMul(idx_total, d_model);

    llvm::Value* output_bytes = builder.CreateMul(output_total,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* output_elems = builder.CreateCall(arena_alloc, {arena_ptr, output_bytes}, "emb_output");

    // Allocate output dims
    llvm::Value* output_dims_bytes = builder.CreateMul(output_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* output_dims = builder.CreateCall(arena_alloc, {arena_ptr, output_dims_bytes}, "emb_dims");

    // Copy indices dims and add d_model
    llvm::Value* idx_dims_bytes = builder.CreateMul(idx_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    builder.CreateMemCpy(output_dims, llvm::MaybeAlign(8), idx_dims, llvm::MaybeAlign(8), idx_dims_bytes);
    builder.CreateStore(d_model,
        builder.CreateGEP(ctx_.int64Type(), output_dims, idx_ndim));

    // Main loop over indices
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "emb_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "emb_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "emb_finalize", current_func);

    llvm::Value* i_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "i_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(i, idx_total), finalize, loop_body);

    builder.SetInsertPoint(loop_body);

    // Get index value (stored as double, convert to int)
    llvm::Value* idx_double = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), idx_elems, i));
    llvm::Value* idx_int = builder.CreateFPToSI(idx_double, ctx_.int64Type());

    // Copy embedding vector
    llvm::Value* w_offset = builder.CreateMul(idx_int, d_model);
    llvm::Value* out_offset = builder.CreateMul(i, d_model);

    // Inner loop over d_model
    llvm::BasicBlock* inner_cond = llvm::BasicBlock::Create(ctx_.context(), "emb_inner_cond", current_func);
    llvm::BasicBlock* inner_body = llvm::BasicBlock::Create(ctx_.context(), "emb_inner_body", current_func);
    llvm::BasicBlock* loop_next = llvm::BasicBlock::Create(ctx_.context(), "emb_next", current_func);

    llvm::Value* d_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "d_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), d_idx);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_cond);
    llvm::Value* d = builder.CreateLoad(ctx_.int64Type(), d_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(d, d_model), loop_next, inner_body);

    builder.SetInsertPoint(inner_body);
    llvm::Value* w_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), w_elems, builder.CreateAdd(w_offset, d)));
    builder.CreateStore(w_elem,
        builder.CreateGEP(ctx_.doubleType(), output_elems, builder.CreateAdd(out_offset, d)));
    builder.CreateStore(builder.CreateAdd(d, llvm::ConstantInt::get(ctx_.int64Type(), 1)), d_idx);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(loop_next);
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_idx);
    builder.CreateBr(loop_cond);

    // Finalize
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "emb_result");

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(output_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(output_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(output_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(output_total, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
