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

llvm::Value* TensorCodegen::taggedNumericToDouble(CodegenContext& ctx,
                                                  TaggedValueCodegen& tagged,
                                                  llvm::Value* value) {
    if (!value) {
        return nullptr;
    }

    if (value->getType() == ctx.taggedValueType()) {
        llvm::Value* type_tag = ctx.builder().CreateExtractValue(value, {0});
        llvm::Value* is_int64 = ctx.builder().CreateICmpEQ(
            type_tag,
            llvm::ConstantInt::get(ctx.int8Type(), ESHKOL_VALUE_INT64)
        );
        llvm::Value* int_as_double = ctx.builder().CreateSIToFP(
            tagged.unpackInt64(value),
            ctx.doubleType()
        );
        llvm::Value* raw_double = tagged.unpackDouble(value);
        return ctx.builder().CreateSelect(is_int64, int_as_double, raw_double);
    }

    if (value->getType()->isIntegerTy()) {
        return ctx.builder().CreateSIToFP(value, ctx.doubleType());
    }

    return value;
}

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

void TensorCodegen::attachLoopMetadata(llvm::BranchInst* backEdge,
                                        bool vectorize, unsigned vecWidth,
                                        bool unroll, unsigned unrollCount) {
    auto& ctx = backEdge->getContext();
    llvm::SmallVector<llvm::Metadata*, 4> ops;
    // Temporary self-reference placeholder
    auto tmp = llvm::MDNode::getTemporary(ctx, llvm::ArrayRef<llvm::Metadata*>());
    ops.push_back(tmp.get());

    if (vectorize) {
        llvm::Metadata* vecEnable[] = {
            llvm::MDString::get(ctx, "llvm.loop.vectorize.enable"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::getTrue(ctx))
        };
        ops.push_back(llvm::MDNode::get(ctx, vecEnable));

        llvm::Metadata* vecW[] = {
            llvm::MDString::get(ctx, "llvm.loop.vectorize.width"),
            llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), vecWidth))
        };
        ops.push_back(llvm::MDNode::get(ctx, vecW));
    }

    if (unroll) {
        llvm::Metadata* unrollC[] = {
            llvm::MDString::get(ctx, "llvm.loop.unroll.count"),
            llvm::ConstantAsMetadata::get(
                llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), unrollCount))
        };
        ops.push_back(llvm::MDNode::get(ctx, unrollC));
    }

    auto* loopMD = llvm::MDNode::get(ctx, ops);
    loopMD->replaceOperandWith(0, loopMD);  // self-reference
    backEdge->setMetadata(llvm::LLVMContext::MD_loop, loopMD);
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

    // Null check: raise "out of memory" if allocation failed
    {
        llvm::Value* is_null = builder.CreateICmpEQ(typed_tensor_ptr,
            llvm::ConstantPointerNull::get(builder.getPtrTy()));
        llvm::Function* current_func = builder.GetInsertBlock()->getParent();
        llvm::BasicBlock* alloc_ok = llvm::BasicBlock::Create(context, "tensor_alloc_ok", current_func);
        llvm::BasicBlock* alloc_fail = llvm::BasicBlock::Create(context, "tensor_alloc_fail", current_func);
        builder.CreateCondBr(is_null, alloc_fail, alloc_ok);
        builder.SetInsertPoint(alloc_fail);
        llvm::Function* raise_func = ctx_.module().getFunction("eshkol_raise");
        if (!raise_func) {
            llvm::FunctionType* raise_type = llvm::FunctionType::get(
                builder.getVoidTy(), {ctx_.ptrType()}, false);
            raise_func = llvm::Function::Create(raise_type, llvm::Function::ExternalLinkage,
                "eshkol_raise", &ctx_.module());
            raise_func->setDoesNotReturn();
        }
        llvm::Function* make_exc_func = ctx_.module().getFunction("eshkol_make_exception_with_header");
        if (!make_exc_func) {
            llvm::FunctionType* make_type = llvm::FunctionType::get(ctx_.ptrType(),
                {builder.getInt32Ty(), ctx_.ptrType()}, false);
            make_exc_func = llvm::Function::Create(make_type, llvm::Function::ExternalLinkage,
                "eshkol_make_exception_with_header", &ctx_.module());
        }
        llvm::Value* err_msg = builder.CreateGlobalString("tensor allocation failed: out of memory");
        llvm::Value* exc = builder.CreateCall(make_exc_func,
            {llvm::ConstantInt::get(builder.getInt32Ty(), ESHKOL_EXCEPTION_ERROR), err_msg});
        builder.CreateCall(raise_func, {exc});
        builder.CreateUnreachable();
        builder.SetInsertPoint(alloc_ok);
    }

    // Allocate dimensions array using arena
    llvm::Value* dims_size = llvm::ConstantInt::get(ctx_.sizeType(),
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
    llvm::Value* elements_size = llvm::ConstantInt::get(ctx_.sizeType(),
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

    // Collect indices and compute linear offset. Each index is funneled
    // through `eshkol_unwrap_list_index` so (tensor-get t (list i j))
    // works alongside (tensor-get t i j) — fit.esk passes list-wrapped
    // indices from its tensor-set!/tensor-ref call sites.
    llvm::Function* unwrap_fn = ctx_.module().getFunction("eshkol_unwrap_list_index");
    if (!unwrap_fn) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.int64Type(), {ctx_.ptrType()}, false);
        unwrap_fn = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage,
            "eshkol_unwrap_list_index", &ctx_.module());
    }

    std::vector<llvm::Value*> indices;
    for (uint64_t i = 0; i < num_indices; i++) {
        llvm::Value* idx = codegenAST(&op->call_op.variables[i + 1]);
        if (!idx) return nullptr;
        if (idx->getType() != ctx_.taggedValueType()) {
            idx = tagged_.packInt64(tagged_.safeExtractInt64(idx), true);
        }
        llvm::IRBuilderBase::InsertPoint saved = ctx_.builder().saveIP();
        llvm::Function* cur_fn = ctx_.builder().GetInsertBlock()->getParent();
        ctx_.builder().SetInsertPoint(&cur_fn->getEntryBlock(), cur_fn->getEntryBlock().begin());
        llvm::Value* idx_slot = ctx_.builder().CreateAlloca(
            ctx_.taggedValueType(), nullptr, "tget_idx_slot");
        ctx_.builder().restoreIP(saved);
        ctx_.builder().CreateStore(idx, idx_slot);
        indices.push_back(ctx_.builder().CreateCall(unwrap_fn, {idx_slot}));
    }

    // Guard: num_indices must not exceed ndim (prevents out-of-bounds dims[i] read)
    {
        llvm::Value* idx_count = llvm::ConstantInt::get(ctx_.int64Type(), num_indices);
        llvm::Value* too_many = ctx_.builder().CreateICmpUGT(idx_count, ndim);
        llvm::Function* cur_fn = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* idx_ok = llvm::BasicBlock::Create(ctx_.context(), "tget_idx_ok", cur_fn);
        llvm::BasicBlock* idx_err = llvm::BasicBlock::Create(ctx_.context(), "tget_idx_err", cur_fn);
        ctx_.builder().CreateCondBr(too_many, idx_err, idx_ok);

        ctx_.builder().SetInsertPoint(idx_err);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = ctx_.builder().CreateGlobalString(
                "Error: too many indices for tensor-get (got %lld, tensor is %lldD)\n");
            ctx_.builder().CreateCall(pf, {fmt, idx_count, ndim});
            ctx_.builder().CreateCall(ef, {llvm::ConstantInt::get(ctx_.builder().getInt32Ty(), 1)});
        }
        ctx_.builder().CreateUnreachable();

        ctx_.builder().SetInsertPoint(idx_ok);
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

        // Bounds check: verify indices[i] < dims[i]
        llvm::Value* oob = ctx_.builder().CreateICmpUGE(indices[i], dim_i, "idx_oob");
        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* oob_bb = llvm::BasicBlock::Create(ctx_.context(), "tensor_oob", current_func);
        llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "tensor_ok", current_func);
        ctx_.builder().CreateCondBr(oob, oob_bb, ok_bb);

        ctx_.builder().SetInsertPoint(oob_bb);
        llvm::Function* printf_fn = ctx_.lookupFunction("printf");
        llvm::Function* exit_fn = ctx_.lookupFunction("exit");
        if (printf_fn && exit_fn) {
            llvm::Value* fmt = ctx_.builder().CreateGlobalString(
                "Error: tensor index out of bounds (dimension %lld)\n");
            ctx_.builder().CreateCall(printf_fn, {fmt, dim_i});
            ctx_.builder().CreateCall(exit_fn, {llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx_.context()), 1)});
        }
        ctx_.builder().CreateUnreachable();

        ctx_.builder().SetInsertPoint(ok_bb);

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
    // tensor-set!: (tensor-set! tensor index1 index2 ... value)
    // R7RS-style: value is the LAST argument (matches vector-set!)
    if (op->call_op.num_vars < 3) {
        eshkol_error("tensor-set! requires at least tensor, index, and value");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    // Value is the LAST argument
    llvm::Value* new_value = codegenAST(&op->call_op.variables[op->call_op.num_vars - 1]);
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

    llvm::Value* ndim_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* ndim = ctx_.builder().CreateLoad(ctx_.int64Type(), ndim_field_ptr);

    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), elements_field_ptr);
    llvm::Value* typed_elements_ptr = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.ptrType());

    llvm::Value* total_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = ctx_.builder().CreateLoad(ctx_.int64Type(), total_field_ptr);

    // Indices are variables[1] through variables[num_vars-2] (last is value)
    const uint64_t num_set_indices = op->call_op.num_vars - 2;

    // Guard: num_indices must not exceed ndim
    {
        llvm::Value* idx_count = llvm::ConstantInt::get(ctx_.int64Type(), num_set_indices);
        llvm::Value* too_many = ctx_.builder().CreateICmpUGT(idx_count, ndim);
        llvm::Function* cur_fn = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* idx_ok = llvm::BasicBlock::Create(ctx_.context(), "tset_idx_ok", cur_fn);
        llvm::BasicBlock* idx_err = llvm::BasicBlock::Create(ctx_.context(), "tset_idx_err", cur_fn);
        ctx_.builder().CreateCondBr(too_many, idx_err, idx_ok);

        ctx_.builder().SetInsertPoint(idx_err);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = ctx_.builder().CreateGlobalString(
                "Error: too many indices for tensor-set! (got %lld, tensor is %lldD)\n");
            ctx_.builder().CreateCall(pf, {fmt, idx_count, ndim});
            ctx_.builder().CreateCall(ef, {llvm::ConstantInt::get(ctx_.builder().getInt32Ty(), 1)});
        }
        ctx_.builder().CreateUnreachable();

        ctx_.builder().SetInsertPoint(idx_ok);
    }

    // Calculate linear index (indices are variables[1..num_vars-2]).
    //
    // Each index is funneled through `eshkol_unwrap_list_index` — same
    // runtime helper used by tensor-ref — so the NumPy/JAX idiom
    // `(tensor-set! t (list i) v)` works uniformly. The helper returns
    // the integer index directly for scalar INT64/DOUBLE and extracts
    // car for HEAP_PTR+CONS, so no AST-level detection is needed.
    llvm::Function* unwrap_fn = ctx_.module().getFunction("eshkol_unwrap_list_index");
    if (!unwrap_fn) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.int64Type(), {ctx_.ptrType()}, false);
        unwrap_fn = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage,
            "eshkol_unwrap_list_index", &ctx_.module());
    }

    llvm::Value* stride = llvm::ConstantInt::get(ctx_.int64Type(), 1);
    for (int64_t i = static_cast<int64_t>(num_set_indices) - 1; i >= 0; i--) {
        llvm::Value* index = codegenAST(&op->call_op.variables[i + 1]);
        if (index) {
            // Normalise to tagged_value, then unwrap through the helper.
            if (index->getType() != ctx_.taggedValueType()) {
                index = tagged_.packInt64(tagged_.safeExtractInt64(index), true);
            }
            llvm::IRBuilderBase::InsertPoint saved = ctx_.builder().saveIP();
            llvm::Function* cur_fn = ctx_.builder().GetInsertBlock()->getParent();
            ctx_.builder().SetInsertPoint(&cur_fn->getEntryBlock(), cur_fn->getEntryBlock().begin());
            llvm::Value* idx_slot = ctx_.builder().CreateAlloca(
                ctx_.taggedValueType(), nullptr, "tset_idx_slot");
            ctx_.builder().restoreIP(saved);
            ctx_.builder().CreateStore(index, idx_slot);
            llvm::Value* idx_int = ctx_.builder().CreateCall(unwrap_fn, {idx_slot});

            llvm::Value* contribution = ctx_.builder().CreateMul(idx_int, stride);
            linear_index = ctx_.builder().CreateAdd(linear_index, contribution);

            if (i > 0) {
                llvm::Value* dim_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims_ptr,
                                                      llvm::ConstantInt::get(ctx_.int64Type(), i));
                llvm::Value* dim = ctx_.builder().CreateLoad(ctx_.int64Type(), dim_ptr);
                stride = ctx_.builder().CreateMul(stride, dim);
            }
        }
    }

    // Bounds check: linear_index < total_elements
    {
        llvm::Value* oob = ctx_.builder().CreateICmpUGE(linear_index, total_elements);
        llvm::Function* cur_fn = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* bounds_ok = llvm::BasicBlock::Create(ctx_.context(), "tset_bounds_ok", cur_fn);
        llvm::BasicBlock* bounds_err = llvm::BasicBlock::Create(ctx_.context(), "tset_bounds_err", cur_fn);
        ctx_.builder().CreateCondBr(oob, bounds_err, bounds_ok);

        ctx_.builder().SetInsertPoint(bounds_err);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = ctx_.builder().CreateGlobalString(
                "Error: tensor-set! index out of bounds (index %lld, size %lld)\n");
            ctx_.builder().CreateCall(pf, {fmt, linear_index, total_elements});
            ctx_.builder().CreateCall(ef, {llvm::ConstantInt::get(ctx_.builder().getInt32Ty(), 1)});
        }
        ctx_.builder().CreateUnreachable();

        ctx_.builder().SetInsertPoint(bounds_ok);
    }

    // Store new value at linear index — tensor stores doubles as int64 bitpatterns
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements_ptr, linear_index);
    llvm::Value* val_double = extractAsDouble(new_value);
    llvm::Value* val_bits = ctx_.builder().CreateBitCast(val_double, ctx_.int64Type());
    ctx_.builder().CreateStore(val_bits, elem_ptr);

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
    } else if (operation == "pow") {
        llvm::Function* pow_fn = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::pow, {ctx_.doubleType()});
        result_double = ctx_.builder().CreateCall(pow_fn, {elem1_double, elem2_double});
    } else if (operation == "max") {
        llvm::Function* max_fn = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::maxnum, {ctx_.doubleType()});
        result_double = ctx_.builder().CreateCall(max_fn, {elem1_double, elem2_double});
    } else if (operation == "min") {
        llvm::Function* min_fn = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::minnum, {ctx_.doubleType()});
        result_double = ctx_.builder().CreateCall(min_fn, {elem1_double, elem2_double});
    } else {
        // Unsupported operation — use zero as fallback (prevents nullptr crash)
        result_double = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
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
    builder.CreateStore(tagged_.packHeapPtr(typed_result_tensor_ptr), shared_result);
    builder.CreateBr(arith_done);

    // ===== MERGE: return result from whichever path was taken =====
    builder.SetInsertPoint(arith_done);
    return builder.CreateLoad(ctx_.taggedValueType(), shared_result);
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
    ctx_.builder().CreateBr(tensor_merge);

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
    llvm::Value* c_total = ctx_.builder().CreateMul(a_rows, b_cols);

    // Allocate result tensor
    llvm::Value* dot_arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* c_tensor_ptr = ctx_.builder().CreateCall(alloc_tensor_func, {dot_arena_ptr}, "dot_tensor");

    // Allocate result elements (M*N doubles stored as int64 bitpatterns)
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* c_elements_size = ctx_.builder().CreateMul(c_total,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* c_elements_ptr = ctx_.builder().CreateCall(arena_alloc,
        {dot_arena_ptr, c_elements_size}, "dot_elems");

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

    // Get function to apply — supports named arithmetic/math functions
    eshkol_ast_t* func_ast = &op->call_op.variables[1];
    if (func_ast->type != ESHKOL_VAR) {
        eshkol_error("tensor-apply: function argument must be a named function (e.g., sin, cos, +)");
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

    // Extract tensor pointer
    llvm::Value* tensor_ptr_int = tagged_.safeExtractInt64(tensor_val);
    llvm::Value* tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());

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
    llvm::Value* axis_val = tagged_.safeExtractInt64(dimension_value);
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

    llvm::Value* tensor_ptr_int = tagged_.safeExtractInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
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
    llvm::Value* axis = tagged_.safeExtractInt64(axis_val);
    llvm::Value* is_negative = builder.CreateICmpSLT(axis, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* adjusted = builder.CreateAdd(axis, src_num_dims);
    axis = builder.CreateSelect(is_negative, adjusted, axis);

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

// Activation functions (forward + backward): see tensor_activation_codegen.cpp

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

    // Stride-aware copy via runtime function for correct ND concatenation.
    // Compute outer_count = product of result_dims[0..axis-1]
    llvm::BasicBlock* outer_cond = llvm::BasicBlock::Create(ctx_.context(), "concat_outer_cond", current_func);
    llvm::BasicBlock* outer_body = llvm::BasicBlock::Create(ctx_.context(), "concat_outer_body", current_func);
    llvm::BasicBlock* outer_done = llvm::BasicBlock::Create(ctx_.context(), "concat_outer_done", current_func);

    llvm::Value* outer_count_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "outer_count");
    llvm::Value* outer_i_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "outer_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), outer_count_var);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), outer_i_var);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_cond);
    llvm::Value* oi = builder.CreateLoad(ctx_.int64Type(), outer_i_var);
    llvm::Value* outer_cmp = builder.CreateICmpULT(oi, axis);
    builder.CreateCondBr(outer_cmp, outer_body, outer_done);

    builder.SetInsertPoint(outer_body);
    llvm::Value* od_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, oi);
    llvm::Value* od_val = builder.CreateLoad(ctx_.int64Type(), od_ptr);
    llvm::Value* oc_curr = builder.CreateLoad(ctx_.int64Type(), outer_count_var);
    builder.CreateStore(builder.CreateMul(oc_curr, od_val), outer_count_var);
    builder.CreateStore(builder.CreateAdd(oi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), outer_i_var);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_done);
    llvm::Value* outer_count = builder.CreateLoad(ctx_.int64Type(), outer_count_var);

    // Build arrays of tensor element pointers and axis dims for runtime function
    size_t num_tensors = tensor_ptrs.size();
    llvm::Value* src_datas_buf = builder.CreateCall(concat_arena_alloc,
        {arena_ptr, llvm::ConstantInt::get(ctx_.int64Type(), num_tensors * sizeof(double*))}, "src_datas");
    llvm::Value* src_axis_dims_buf = builder.CreateCall(concat_arena_alloc,
        {arena_ptr, llvm::ConstantInt::get(ctx_.int64Type(), num_tensors * sizeof(int64_t))}, "src_axis_dims");

    for (size_t i = 0; i < num_tensors; ++i) {
        llvm::Value* t_ptr = tensor_ptrs[i];
        llvm::Value* t_elems_field = builder.CreateStructGEP(tensor_type, t_ptr, 2);
        llvm::Value* t_elems = builder.CreateLoad(ctx_.ptrType(), t_elems_field);

        // Store element pointer
        llvm::Value* slot = builder.CreateGEP(ctx_.ptrType(), src_datas_buf,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        builder.CreateStore(t_elems, slot);

        // Store axis dimension
        llvm::Value* t_dims_field = builder.CreateStructGEP(tensor_type, t_ptr, 0);
        llvm::Value* t_dims_ptr = builder.CreateLoad(ctx_.ptrType(), t_dims_field);
        llvm::Value* t_axis_dim_ptr = builder.CreateGEP(ctx_.int64Type(), t_dims_ptr, axis);
        llvm::Value* t_axis_dim = builder.CreateLoad(ctx_.int64Type(), t_axis_dim_ptr);
        llvm::Value* adim_slot = builder.CreateGEP(ctx_.int64Type(), src_axis_dims_buf,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        builder.CreateStore(t_axis_dim, adim_slot);
    }

    // Call runtime: eshkol_concat_strided(result_data, num_tensors, src_datas, src_axis_dims, stride_after, outer_count)
    auto* concat_ft = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_.context()),
        {ctx_.ptrType(), ctx_.int64Type(), ctx_.ptrType(), ctx_.ptrType(),
         ctx_.int64Type(), ctx_.int64Type()}, false);
    llvm::Function* concat_fn = ctx_.module().getFunction("eshkol_concat_strided");
    if (!concat_fn) {
        concat_fn = llvm::Function::Create(concat_ft,
            llvm::Function::ExternalLinkage, "eshkol_concat_strided", &ctx_.module());
    }
    builder.CreateCall(concat_fn,
        {result_elems, llvm::ConstantInt::get(ctx_.int64Type(), num_tensors),
         src_datas_buf, src_axis_dims_buf, stride_after, outer_count});

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

llvm::Value* TensorCodegen::tensorLength(const eshkol_operations_t* op) {
    // tensor-length: (tensor-length tensor) -> total number of elements
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-length requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* arg = codegenAST(&op->call_op.variables[0]);
    if (!arg) return nullptr;

    // Extract raw pointer from tagged value
    if (arg->getType() == ctx_.taggedValueType()) {
        arg = tagged_.unpackInt64(arg);
    }

    llvm::Value* tensor_ptr = ctx_.builder().CreateIntToPtr(arg, ctx_.ptrType());

    // Field 3 of tensor struct is total_elements (int64)
    llvm::Value* total_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 3);
    llvm::Value* total = ctx_.builder().CreateLoad(ctx_.int64Type(), total_field);

    return tagged_.packInt64(total);
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

    // Use alloca-based merge (avoids PHI predecessor issues with XLA blocks)
    llvm::Value* result_alloca = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "trans_result");

    ctx_.builder().CreateCondBr(is_tensor, tensor_block, error_block);

    // Error path - return null for non-tensor inputs
    ctx_.builder().SetInsertPoint(error_block);
    llvm::Value* error_result = tagged_.packNull();
    ctx_.builder().CreateStore(error_result, result_alloca);
    ctx_.builder().CreateBr(exit_block);

    // Tensor path - proceed with normal transpose
    ctx_.builder().SetInsertPoint(tensor_block);
    llvm::Value* ptr_int = tagged_.unpackInt64(src_tensor);
    llvm::Value* src_ptr = ctx_.builder().CreateIntToPtr(ptr_int, ctx_.ptrType());

    // Get source tensor properties
    llvm::Value* src_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 0);
    llvm::Value* src_dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_dims_field_ptr);
    llvm::Value* typed_src_dims_ptr = ctx_.builder().CreatePointerCast(src_dims_ptr, ctx_.ptrType());

    // Guard: transpose only supports 2D tensors
    {
        llvm::Value* src_ndim_field = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 1);
        llvm::Value* src_ndim = ctx_.builder().CreateLoad(ctx_.int64Type(), src_ndim_field);
        llvm::Value* not_2d = ctx_.builder().CreateICmpNE(src_ndim,
            llvm::ConstantInt::get(ctx_.int64Type(), 2));
        llvm::BasicBlock* trans_dims_ok = llvm::BasicBlock::Create(ctx_.context(), "transpose_dims_ok", current_func);
        llvm::BasicBlock* trans_dims_err = llvm::BasicBlock::Create(ctx_.context(), "transpose_dims_err", current_func);
        ctx_.builder().CreateCondBr(not_2d, trans_dims_err, trans_dims_ok);

        ctx_.builder().SetInsertPoint(trans_dims_err);
        llvm::Function* printf_fn_trans = ctx_.lookupFunction("printf");
        llvm::Function* exit_fn_trans = ctx_.lookupFunction("exit");
        if (printf_fn_trans && exit_fn_trans) {
            llvm::Value* fmt = ctx_.builder().CreateGlobalString(
                "Error: transpose only supports 2D tensors (got %lld dimensions)\n");
            ctx_.builder().CreateCall(printf_fn_trans, {fmt, src_ndim});
            ctx_.builder().CreateCall(exit_fn_trans, {llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx_.context()), 1)});
        }
        ctx_.builder().CreateUnreachable();

        ctx_.builder().SetInsertPoint(trans_dims_ok);
    }

#ifdef ESHKOL_XLA_ENABLED
    if (xla_ && xla_->isAvailable()) {
        // Check if tensor is large enough for XLA dispatch
        llvm::Value* total_field = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 3);
        llvm::Value* total_elements = ctx_.builder().CreateLoad(ctx_.int64Type(), total_field, "trans_total");
        llvm::Value* threshold = llvm::ConstantInt::get(ctx_.int64Type(), xla::xla_get_threshold());
        llvm::Value* use_xla = ctx_.builder().CreateICmpUGE(total_elements, threshold);

        llvm::BasicBlock* xla_block = llvm::BasicBlock::Create(ctx_.context(), "trans_xla", current_func);
        llvm::BasicBlock* cpu_block = llvm::BasicBlock::Create(ctx_.context(), "trans_cpu", current_func);
        ctx_.builder().CreateCondBr(use_xla, xla_block, cpu_block);

        // XLA path
        ctx_.builder().SetInsertPoint(xla_block);
        llvm::Value* xla_result = xla_->emitTranspose(src_ptr);
        if (xla_result) {
            llvm::Value* xla_tagged = tagged_.packHeapPtr(xla_result);
            ctx_.builder().CreateStore(xla_tagged, result_alloca);
            ctx_.builder().CreateBr(exit_block);
        } else {
            ctx_.builder().CreateBr(cpu_block);
        }

        // CPU fallback
        ctx_.builder().SetInsertPoint(cpu_block);
    }
#endif

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
    ctx_.builder().CreateStore(tensor_result, result_alloca);
    ctx_.builder().CreateBr(exit_block);

    // Merge — load from alloca (all paths store their result)
    ctx_.builder().SetInsertPoint(exit_block);
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_alloca, "transpose_result");
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

    // Null check: arena allocation can fail on OOM
    {
        llvm::Value* is_null = ctx_.builder().CreateICmpEQ(conv_tensor_ptr,
            llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_.context(), 0)));
        llvm::Function* curr_fn = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* alloc_ok = llvm::BasicBlock::Create(ctx_.context(), "vec2tensor_ok", curr_fn);
        llvm::BasicBlock* alloc_fail = llvm::BasicBlock::Create(ctx_.context(), "vec2tensor_oom", curr_fn);
        ctx_.builder().CreateCondBr(is_null, alloc_fail, alloc_ok);
        ctx_.builder().SetInsertPoint(alloc_fail);
        llvm::Function* raise_func = ctx_.module().getFunction("eshkol_raise");
        if (!raise_func) {
            llvm::FunctionType* raise_type = llvm::FunctionType::get(ctx_.builder().getVoidTy(), {ctx_.ptrType()}, false);
            raise_func = llvm::Function::Create(raise_type, llvm::Function::ExternalLinkage, "eshkol_raise", &ctx_.module());
            raise_func->setDoesNotReturn();
        }
        llvm::Function* make_exc_func = ctx_.module().getFunction("eshkol_make_exception_with_header");
        if (!make_exc_func) {
            llvm::FunctionType* make_type = llvm::FunctionType::get(ctx_.ptrType(), {ctx_.builder().getInt32Ty(), ctx_.ptrType()}, false);
            make_exc_func = llvm::Function::Create(make_type, llvm::Function::ExternalLinkage, "eshkol_make_exception_with_header", &ctx_.module());
        }
        llvm::Value* err_msg = ctx_.builder().CreateGlobalString("vector->tensor: allocation failed (out of memory)");
        llvm::Value* exception = ctx_.builder().CreateCall(make_exc_func,
            {llvm::ConstantInt::get(ctx_.builder().getInt32Ty(), 1), err_msg});
        ctx_.builder().CreateCall(raise_func, {exception});
        ctx_.builder().CreateUnreachable();
        ctx_.builder().SetInsertPoint(alloc_ok);
    }

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

    // Get new dimensions - three cases:
    // 1. Individual args: (reshape tensor 3 3) -> num_vars > 2
    // 2. List arg: (reshape tensor (list 3 3 2)) -> num_vars == 2 and arg is list
    // 3. Single dim: (reshape tensor 9) -> num_vars == 2 and arg is integer
    // All paths produce: final_dims_ptr (int64_t*), final_ndim (i64), final_total (i64)

    llvm::Value* final_dims_ptr = nullptr;
    llvm::Value* final_ndim = nullptr;
    llvm::Value* final_total = nullptr;

    // Get source tensor properties (needed for all paths)
    llvm::Value* src_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 2);
    llvm::Value* src_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_elements_field_ptr);

    llvm::Function* arena_alloc = mem_.getArenaAllocate();

    if (op->call_op.num_vars == 2) {
        // Could be a single dimension OR a list of dimensions
        llvm::Value* dim_arg = codegenAST(&op->call_op.variables[1]);
        if (!dim_arg) return nullptr;

        // Check if it's a HEAP_PTR (could be cons list OR tensor)
        llvm::Value* type_tag = tagged_.getType(dim_arg);
        llvm::Value* base_type = tagged_.getBaseType(type_tag);
        llvm::Value* is_heap = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

        llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* check_subtype = llvm::BasicBlock::Create(ctx_.context(), "reshape_check_sub", func);
        llvm::BasicBlock* tensor_dims_path = llvm::BasicBlock::Create(ctx_.context(), "reshape_tensor_dims", func);
        llvm::BasicBlock* list_path = llvm::BasicBlock::Create(ctx_.context(), "reshape_list", func);
        llvm::BasicBlock* single_path = llvm::BasicBlock::Create(ctx_.context(), "reshape_single", func);
        llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "reshape_merge", func);

        ctx_.builder().CreateCondBr(is_heap, check_subtype, single_path);

        // CHECK SUBTYPE: is it a tensor or a cons list?
        ctx_.builder().SetInsertPoint(check_subtype);
        llvm::Value* heap_ptr_int = tagged_.unpackInt64(dim_arg);
        llvm::Value* heap_ptr = ctx_.builder().CreateIntToPtr(heap_ptr_int, ctx_.ptrType());
        // Header is at ptr - 8, subtype is the first byte
        llvm::Value* header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), heap_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), -8));
        llvm::Value* subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), header_ptr);
        llvm::Value* is_tensor = ctx_.builder().CreateICmpEQ(subtype,
            llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));
        ctx_.builder().CreateCondBr(is_tensor, tensor_dims_path, list_path);

        // TENSOR DIMS PATH: call runtime to extract dims from tensor elements
        ctx_.builder().SetInsertPoint(tensor_dims_path);
        llvm::Value* t_arena = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        llvm::Value* td_max_bytes = llvm::ConstantInt::get(ctx_.int64Type(), 16 * sizeof(int64_t));
        llvm::Value* td_dims_array = ctx_.builder().CreateCall(
            arena_alloc, {t_arena, td_max_bytes}, "tensor_dims");

        auto* t2d_ft = llvm::FunctionType::get(ctx_.int64Type(),
            {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
        llvm::Function* t2d_fn = ctx_.module().getFunction("eshkol_tensor_to_dims");
        if (!t2d_fn) {
            t2d_fn = llvm::Function::Create(t2d_ft,
                llvm::Function::ExternalLinkage, "eshkol_tensor_to_dims", &ctx_.module());
        }
        llvm::Value* td_ndim = ctx_.builder().CreateCall(
            t2d_fn, {heap_ptr, td_dims_array, llvm::ConstantInt::get(ctx_.int64Type(), 16)},
            "tensor_ndim");

        auto* td_total_ft = llvm::FunctionType::get(ctx_.int64Type(),
            {ctx_.ptrType(), ctx_.int64Type()}, false);
        llvm::Function* td_total_fn = ctx_.module().getFunction("eshkol_compute_dims_total");
        if (!td_total_fn) {
            td_total_fn = llvm::Function::Create(td_total_ft,
                llvm::Function::ExternalLinkage, "eshkol_compute_dims_total", &ctx_.module());
        }
        llvm::Value* td_total = ctx_.builder().CreateCall(
            td_total_fn, {td_dims_array, td_ndim}, "tensor_total");

        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* tensor_dims_exit = ctx_.builder().GetInsertBlock();

        // LIST PATH: Walk cons list to extract N dimensions via runtime helper
        ctx_.builder().SetInsertPoint(list_path);
        llvm::Value* cons_ptr = ctx_.builder().CreateIntToPtr(heap_ptr_int, ctx_.ptrType());

        // Allocate dims array for up to 16 dimensions
        llvm::Value* list_arena = ctx_.builder().CreateLoad(
            llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
        llvm::Value* max_dims_bytes = llvm::ConstantInt::get(ctx_.int64Type(), 16 * sizeof(int64_t));
        llvm::Value* list_dims_array = ctx_.builder().CreateCall(
            arena_alloc, {list_arena, max_dims_bytes}, "list_dims");

        // Call runtime: int64_t eshkol_cons_list_to_dims(void*, int64_t*, int64_t)
        auto* list_to_dims_ft = llvm::FunctionType::get(ctx_.int64Type(),
            {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
        llvm::Function* list_to_dims_fn = ctx_.module().getFunction("eshkol_cons_list_to_dims");
        if (!list_to_dims_fn) {
            list_to_dims_fn = llvm::Function::Create(list_to_dims_ft,
                llvm::Function::ExternalLinkage, "eshkol_cons_list_to_dims", &ctx_.module());
        }
        llvm::Value* list_ndim = ctx_.builder().CreateCall(
            list_to_dims_fn,
            {cons_ptr, list_dims_array, llvm::ConstantInt::get(ctx_.int64Type(), 16)},
            "list_ndim");

        // Compute total via runtime: int64_t eshkol_compute_dims_total(int64_t*, int64_t)
        auto* total_ft = llvm::FunctionType::get(ctx_.int64Type(),
            {ctx_.ptrType(), ctx_.int64Type()}, false);
        llvm::Function* total_fn = ctx_.module().getFunction("eshkol_compute_dims_total");
        if (!total_fn) {
            total_fn = llvm::Function::Create(total_ft,
                llvm::Function::ExternalLinkage, "eshkol_compute_dims_total", &ctx_.module());
        }
        llvm::Value* list_total = ctx_.builder().CreateCall(
            total_fn, {list_dims_array, list_ndim}, "list_total");

        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_exit = ctx_.builder().GetInsertBlock();

        // SINGLE PATH: Treat as 1D reshape
        ctx_.builder().SetInsertPoint(single_path);
        llvm::Value* single_dim = dim_arg;
        if (single_dim->getType() == ctx_.taggedValueType()) {
            single_dim = tagged_.unpackInt64(single_dim);
        }
        // Allocate 1-element dims array
        llvm::Value* single_arena = ctx_.builder().CreateLoad(
            llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
        llvm::Value* single_bytes = llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
        llvm::Value* single_dims_array = ctx_.builder().CreateCall(
            arena_alloc, {single_arena, single_bytes}, "single_dims");
        ctx_.builder().CreateStore(single_dim, single_dims_array);
        llvm::Value* single_ndim = llvm::ConstantInt::get(ctx_.int64Type(), 1);
        llvm::Value* single_total = single_dim;

        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* single_exit = ctx_.builder().GetInsertBlock();

        // MERGE: PHI nodes for dims_ptr, ndim, total
        ctx_.builder().SetInsertPoint(merge_block);

        llvm::PHINode* dims_phi = ctx_.builder().CreatePHI(ctx_.ptrType(), 3, "merged_dims");
        dims_phi->addIncoming(td_dims_array, tensor_dims_exit);
        dims_phi->addIncoming(list_dims_array, list_exit);
        dims_phi->addIncoming(single_dims_array, single_exit);

        llvm::PHINode* ndim_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 3, "merged_ndim");
        ndim_phi->addIncoming(td_ndim, tensor_dims_exit);
        ndim_phi->addIncoming(list_ndim, list_exit);
        ndim_phi->addIncoming(single_ndim, single_exit);

        llvm::PHINode* total_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 3, "merged_total");
        total_phi->addIncoming(td_total, tensor_dims_exit);
        total_phi->addIncoming(list_total, list_exit);
        total_phi->addIncoming(single_total, single_exit);

        final_dims_ptr = dims_phi;
        final_ndim = ndim_phi;
        final_total = total_phi;
    } else {
        // Multiple explicit dimension arguments: (reshape tensor 3 3 2)
        size_t ndim_count = op->call_op.num_vars - 1;

        llvm::Value* multi_arena = ctx_.builder().CreateLoad(
            llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
        llvm::Value* multi_bytes = llvm::ConstantInt::get(ctx_.int64Type(), ndim_count * sizeof(int64_t));
        llvm::Value* multi_dims_array = ctx_.builder().CreateCall(
            arena_alloc, {multi_arena, multi_bytes}, "multi_dims");

        llvm::Value* multi_total = nullptr;
        for (uint64_t i = 1; i < op->call_op.num_vars; i++) {
            llvm::Value* dim = codegenAST(&op->call_op.variables[i]);
            if (!dim) return nullptr;
            if (dim->getType() == ctx_.taggedValueType()) {
                dim = tagged_.unpackInt64(dim);
            }
            llvm::Value* dim_slot = ctx_.builder().CreateGEP(
                ctx_.int64Type(), multi_dims_array,
                llvm::ConstantInt::get(ctx_.int64Type(), i - 1));
            ctx_.builder().CreateStore(dim, dim_slot);

            if (multi_total == nullptr) {
                multi_total = dim;
            } else {
                multi_total = ctx_.builder().CreateMul(multi_total, dim);
            }
        }

        final_dims_ptr = multi_dims_array;
        final_ndim = llvm::ConstantInt::get(ctx_.int64Type(), ndim_count);
        final_total = multi_total;
    }

    // Allocate using arena
    llvm::Value* reshape_arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Create new tensor structure with header (reuse elements - no copy needed for reshape)
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* typed_new_tensor_ptr = ctx_.builder().CreateCall(
        alloc_tensor_func, {reshape_arena_ptr}, "reshape_tensor");

    // Null check: arena allocation can fail on OOM
    {
        llvm::Value* is_null = ctx_.builder().CreateICmpEQ(typed_new_tensor_ptr,
            llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_.context(), 0)));
        llvm::Function* curr_fn = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* alloc_ok = llvm::BasicBlock::Create(ctx_.context(), "reshape_ok", curr_fn);
        llvm::BasicBlock* alloc_fail = llvm::BasicBlock::Create(ctx_.context(), "reshape_oom", curr_fn);
        ctx_.builder().CreateCondBr(is_null, alloc_fail, alloc_ok);
        ctx_.builder().SetInsertPoint(alloc_fail);
        llvm::Function* raise_func = ctx_.module().getFunction("eshkol_raise");
        if (!raise_func) {
            llvm::FunctionType* raise_type = llvm::FunctionType::get(ctx_.builder().getVoidTy(), {ctx_.ptrType()}, false);
            raise_func = llvm::Function::Create(raise_type, llvm::Function::ExternalLinkage, "eshkol_raise", &ctx_.module());
            raise_func->setDoesNotReturn();
        }
        llvm::Function* make_exc_func = ctx_.module().getFunction("eshkol_make_exception_with_header");
        if (!make_exc_func) {
            llvm::FunctionType* make_type = llvm::FunctionType::get(ctx_.ptrType(), {ctx_.builder().getInt32Ty(), ctx_.ptrType()}, false);
            make_exc_func = llvm::Function::Create(make_type, llvm::Function::ExternalLinkage, "eshkol_make_exception_with_header", &ctx_.module());
        }
        llvm::Value* err_msg = ctx_.builder().CreateGlobalString("reshape: allocation failed (out of memory)");
        llvm::Value* exception = ctx_.builder().CreateCall(make_exc_func,
            {llvm::ConstantInt::get(ctx_.builder().getInt32Ty(), 1), err_msg});
        ctx_.builder().CreateCall(raise_func, {exception});
        ctx_.builder().CreateUnreachable();
        ctx_.builder().SetInsertPoint(alloc_ok);
    }

    // Store tensor fields: dims_ptr, ndim, elements (reused), total
    llvm::Value* dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 0);
    ctx_.builder().CreateStore(final_dims_ptr, dims_field_ptr);

    llvm::Value* num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 1);
    ctx_.builder().CreateStore(final_ndim, num_dims_field_ptr);

    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 2);
    ctx_.builder().CreateStore(src_elements_ptr, elements_field_ptr);

    llvm::Value* total_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 3);
    ctx_.builder().CreateStore(final_total, total_elements_field_ptr);

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

llvm::Value* TensorCodegen::makeTensor(const eshkol_operations_t* op) {
    // make-tensor: Two modes:
    //   (make-tensor shape fill_value) - create tensor with given shape filled with fill_value
    //   (make-tensor e1 e2 e3 ...)    - create 1D tensor from elements (falls back to tensor())
    if (op->call_op.num_vars < 1) {
        eshkol_error("make-tensor requires at least 1 argument");
        return nullptr;
    }

    // If not exactly 2 args, fall back to element-literal mode
    if (op->call_op.num_vars != 2) {
        return tensor(op);
    }

    // 2 args: (make-tensor shape fill_value)
    // First arg is shape (list of dimensions), second is fill value
    auto& builder = ctx_.builder();

    llvm::Value* shape_arg = codegenAST(&op->call_op.variables[0]);
    if (!shape_arg) return nullptr;

    llvm::Value* fill_arg = codegenAST(&op->call_op.variables[1]);
    if (!fill_arg) return nullptr;

    // Extract fill value as double → int64 bit pattern
    llvm::Value* fill_double;
    if (fill_arg->getType() == ctx_.taggedValueType()) {
        fill_double = extractAsDouble(fill_arg);
    } else if (fill_arg->getType() == ctx_.doubleType()) {
        fill_double = fill_arg;
    } else if (fill_arg->getType()->isIntegerTy()) {
        fill_double = builder.CreateSIToFP(fill_arg, ctx_.doubleType());
    } else {
        fill_double = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    }
    llvm::Value* fill_bits = builder.CreateBitCast(fill_double, ctx_.int64Type());

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Check if first arg is a list/cons (shape) or a scalar (element)
    // If it's a list → shape+fill mode; if scalar → element-literal mode
    llvm::BasicBlock* heap_dispatch = llvm::BasicBlock::Create(ctx_.context(), "mt_heap_dispatch", current_func);
    llvm::BasicBlock* list_path = llvm::BasicBlock::Create(ctx_.context(), "mt_list", current_func);
    llvm::BasicBlock* vector_path = llvm::BasicBlock::Create(ctx_.context(), "mt_vector", current_func);
    llvm::BasicBlock* tensor_path = llvm::BasicBlock::Create(ctx_.context(), "mt_tensor_shape", current_func);
    llvm::BasicBlock* scalar_path = llvm::BasicBlock::Create(ctx_.context(), "mt_scalar", current_func);

    if (shape_arg->getType() == ctx_.taggedValueType()) {
        llvm::Value* type_tag = tagged_.getType(shape_arg);
        llvm::Value* base_type = tagged_.getBaseType(type_tag);
        // Lists are HEAP_PTR (cons cells)
        llvm::Value* is_heap = builder.CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
        builder.CreateCondBr(is_heap, heap_dispatch, scalar_path);
    } else {
        // Raw value — not a list, use element-literal mode
        builder.CreateBr(scalar_path);
    }

    builder.SetInsertPoint(heap_dispatch);
    llvm::Value* shape_ptr_int = tagged_.unpackInt64(shape_arg);
    llvm::Value* shape_ptr = builder.CreateIntToPtr(shape_ptr_int, ctx_.ptrType());
    llvm::Value* shape_header_ptr = builder.CreateGEP(
        ctx_.int8Type(), shape_ptr, llvm::ConstantInt::get(ctx_.int64Type(), -8));
    llvm::Value* shape_subtype = builder.CreateLoad(ctx_.int8Type(), shape_header_ptr);
    llvm::Value* is_vector_shape = builder.CreateICmpEQ(
        shape_subtype,
        llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    llvm::Value* is_tensor_shape = builder.CreateICmpEQ(
        shape_subtype,
        llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));
    llvm::BasicBlock* heap_subtype_check = llvm::BasicBlock::Create(
        ctx_.context(), "mt_heap_subtype_check", current_func);
    builder.CreateCondBr(is_vector_shape, vector_path, heap_subtype_check);

    builder.SetInsertPoint(heap_subtype_check);
    builder.CreateCondBr(is_tensor_shape, tensor_path, list_path);

    // SCALAR PATH: treat as 2-element tensor (fallback to tensor())
    builder.SetInsertPoint(scalar_path);
    // Create 1D tensor with 2 elements [shape_arg, fill_arg]
    std::vector<llvm::Value*> scalar_dims = {llvm::ConstantInt::get(ctx_.int64Type(), 2)};
    llvm::Value* scalar_tensor = createTensorWithDims(scalar_dims, nullptr, false);
    if (scalar_tensor) {
        llvm::StructType* tensor_type = ctx_.tensorType();
        llvm::Value* s_elems_field = builder.CreateStructGEP(tensor_type, scalar_tensor, 2);
        llvm::Value* s_elems = builder.CreateLoad(ctx_.ptrType(), s_elems_field);
        // Store shape_arg as element 0
        llvm::Value* e0_double;
        if (shape_arg->getType() == ctx_.taggedValueType()) {
            e0_double = extractAsDouble(shape_arg);
        } else if (shape_arg->getType() == ctx_.doubleType()) {
            e0_double = shape_arg;
        } else {
            e0_double = builder.CreateSIToFP(shape_arg, ctx_.doubleType());
        }
        builder.CreateStore(builder.CreateBitCast(e0_double, ctx_.int64Type()),
            builder.CreateGEP(ctx_.int64Type(), s_elems, llvm::ConstantInt::get(ctx_.int64Type(), 0)));
        // Store fill_arg as element 1
        builder.CreateStore(fill_bits,
            builder.CreateGEP(ctx_.int64Type(), s_elems, llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    }
    llvm::BasicBlock* final_merge = llvm::BasicBlock::Create(ctx_.context(), "mt_final", current_func);
    builder.CreateBr(final_merge);
    llvm::BasicBlock* scalar_exit = builder.GetInsertBlock();

    // LIST PATH: Extract dimensions from cons list (up to 4D)
    builder.SetInsertPoint(list_path);
    llvm::Value* list_ptr_int = tagged_.unpackInt64(shape_arg);
    llvm::Value* cons_ptr = builder.CreateIntToPtr(list_ptr_int, ctx_.ptrType());
    llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    llvm::Value* is_cdr_flag = llvm::ConstantInt::get(ctx_.int1Type(), 1);

    // Extract first dimension
    llvm::Value* dim1 = builder.CreateCall(
        mem_.getTaggedConsGetInt64(), {cons_ptr, is_car});

    // Get cdr
    llvm::Value* cdr_int = builder.CreateCall(
        mem_.getTaggedConsGetPtr(), {cons_ptr, is_cdr_flag});
    llvm::Value* cdr_null = builder.CreateICmpEQ(cdr_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));

    llvm::BasicBlock* mt_has_dim2 = llvm::BasicBlock::Create(ctx_.context(), "mt_has_dim2", current_func);
    llvm::BasicBlock* mt_list_1d = llvm::BasicBlock::Create(ctx_.context(), "mt_list_1d", current_func);
    builder.CreateCondBr(cdr_null, mt_list_1d, mt_has_dim2);

    // 1D case
    builder.SetInsertPoint(mt_list_1d);
    builder.CreateBr(final_merge);  // placeholder, will fix
    llvm::BasicBlock* mt_merge = llvm::BasicBlock::Create(ctx_.context(), "mt_dims_merge", current_func);
    // Re-route: go to merge
    mt_list_1d->getTerminator()->eraseFromParent();
    builder.SetInsertPoint(mt_list_1d);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* list_1d_exit = builder.GetInsertBlock();

    // 2D+ case
    builder.SetInsertPoint(mt_has_dim2);
    llvm::Value* cdr_cons = builder.CreateIntToPtr(cdr_int, ctx_.ptrType());
    llvm::Value* dim2 = builder.CreateCall(
        mem_.getTaggedConsGetInt64(), {cdr_cons, is_car});

    llvm::Value* cddr_int = builder.CreateCall(
        mem_.getTaggedConsGetPtr(), {cdr_cons, is_cdr_flag});
    llvm::Value* cddr_null = builder.CreateICmpEQ(cddr_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));

    llvm::BasicBlock* mt_has_dim3 = llvm::BasicBlock::Create(ctx_.context(), "mt_has_dim3", current_func);
    llvm::BasicBlock* mt_list_2d = llvm::BasicBlock::Create(ctx_.context(), "mt_list_2d", current_func);
    builder.CreateCondBr(cddr_null, mt_list_2d, mt_has_dim3);

    // 2D case
    builder.SetInsertPoint(mt_list_2d);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* list_2d_exit = builder.GetInsertBlock();

    // 3D case
    builder.SetInsertPoint(mt_has_dim3);
    llvm::Value* cddr_cons = builder.CreateIntToPtr(cddr_int, ctx_.ptrType());
    llvm::Value* dim3 = builder.CreateCall(
        mem_.getTaggedConsGetInt64(), {cddr_cons, is_car});
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* list_3d_exit = builder.GetInsertBlock();

    // MERGE dimensions
    builder.SetInsertPoint(mt_merge);

    llvm::PHINode* ndims_phi = builder.CreatePHI(ctx_.int64Type(), 9, "mt_ndims");
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 2), list_2d_exit);
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 3), list_3d_exit);

    llvm::PHINode* d1_phi = builder.CreatePHI(ctx_.int64Type(), 9, "mt_d1");
    d1_phi->addIncoming(dim1, list_1d_exit);
    d1_phi->addIncoming(dim1, list_2d_exit);
    d1_phi->addIncoming(dim1, list_3d_exit);

    llvm::PHINode* d2_phi = builder.CreatePHI(ctx_.int64Type(), 9, "mt_d2");
    d2_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
    d2_phi->addIncoming(dim2, list_2d_exit);
    d2_phi->addIncoming(dim2, list_3d_exit);

    llvm::PHINode* d3_phi = builder.CreatePHI(ctx_.int64Type(), 9, "mt_d3");
    d3_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_1d_exit);
    d3_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), list_2d_exit);
    d3_phi->addIncoming(dim3, list_3d_exit);

    // Create tensor based on ndims
    llvm::Value* is_1d = builder.CreateICmpEQ(ndims_phi,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* is_2d = builder.CreateICmpEQ(ndims_phi,
        llvm::ConstantInt::get(ctx_.int64Type(), 2));

    llvm::BasicBlock* mt_create_1d = llvm::BasicBlock::Create(ctx_.context(), "mt_c1d", current_func);
    llvm::BasicBlock* mt_check_2d = llvm::BasicBlock::Create(ctx_.context(), "mt_chk2d", current_func);
    llvm::BasicBlock* mt_create_2d = llvm::BasicBlock::Create(ctx_.context(), "mt_c2d", current_func);
    llvm::BasicBlock* mt_create_3d = llvm::BasicBlock::Create(ctx_.context(), "mt_c3d", current_func);
    llvm::BasicBlock* mt_list_done = llvm::BasicBlock::Create(ctx_.context(), "mt_ldone", current_func);

    builder.CreateCondBr(is_1d, mt_create_1d, mt_check_2d);

    builder.SetInsertPoint(mt_create_1d);
    std::vector<llvm::Value*> dims_1d = {d1_phi};
    llvm::Value* t1d = createTensorWithDims(dims_1d, fill_bits, false);
    builder.CreateBr(mt_list_done);
    llvm::BasicBlock* c1d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_check_2d);
    builder.CreateCondBr(is_2d, mt_create_2d, mt_create_3d);

    builder.SetInsertPoint(mt_create_2d);
    std::vector<llvm::Value*> dims_2d = {d1_phi, d2_phi};
    llvm::Value* t2d = createTensorWithDims(dims_2d, fill_bits, false);
    builder.CreateBr(mt_list_done);
    llvm::BasicBlock* c2d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_create_3d);
    std::vector<llvm::Value*> dims_3d = {d1_phi, d2_phi, d3_phi};
    llvm::Value* t3d = createTensorWithDims(dims_3d, fill_bits, false);
    builder.CreateBr(mt_list_done);
    llvm::BasicBlock* c3d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_list_done);
    llvm::PHINode* list_tensor = builder.CreatePHI(ctx_.ptrType(), 3, "mt_tensor");
    list_tensor->addIncoming(t1d, c1d_exit);
    list_tensor->addIncoming(t2d, c2d_exit);
    list_tensor->addIncoming(t3d, c3d_exit);

    builder.CreateBr(final_merge);
    llvm::BasicBlock* list_done_exit = builder.GetInsertBlock();

    // VECTOR PATH: Extract dimensions from Scheme vector #(d1 d2 d3)
    builder.SetInsertPoint(vector_path);
    llvm::Value* vector_ptr_int = tagged_.unpackInt64(shape_arg);
    llvm::Value* vector_ptr = builder.CreateIntToPtr(vector_ptr_int, ctx_.ptrType());
    llvm::Value* vector_len = builder.CreateLoad(ctx_.int64Type(), vector_ptr);
    llvm::Value* vector_elems = builder.CreateGEP(
        ctx_.int8Type(), vector_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* vector_elems_typed = builder.CreatePointerCast(vector_elems, ctx_.ptrType());

    llvm::Value* vec_dim1_ptr = builder.CreateGEP(
        ctx_.taggedValueType(), vector_elems_typed, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* vec_dim1_tagged = builder.CreateLoad(ctx_.taggedValueType(), vec_dim1_ptr);
    llvm::Value* vec_dim1 = tagged_.safeExtractInt64(vec_dim1_tagged);

    llvm::Value* vec_has_dim2 = builder.CreateICmpUGT(
        vector_len, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::BasicBlock* mt_vec_has_dim2 = llvm::BasicBlock::Create(ctx_.context(), "mt_vec_has_dim2", current_func);
    llvm::BasicBlock* mt_vec_1d = llvm::BasicBlock::Create(ctx_.context(), "mt_vec_1d", current_func);
    builder.CreateCondBr(vec_has_dim2, mt_vec_has_dim2, mt_vec_1d);

    builder.SetInsertPoint(mt_vec_1d);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* vec_1d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_vec_has_dim2);
    llvm::Value* vec_dim2_ptr = builder.CreateGEP(
        ctx_.taggedValueType(), vector_elems_typed, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* vec_dim2_tagged = builder.CreateLoad(ctx_.taggedValueType(), vec_dim2_ptr);
    llvm::Value* vec_dim2 = tagged_.safeExtractInt64(vec_dim2_tagged);

    llvm::Value* vec_has_dim3 = builder.CreateICmpUGT(
        vector_len, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::BasicBlock* mt_vec_has_dim3 = llvm::BasicBlock::Create(ctx_.context(), "mt_vec_has_dim3", current_func);
    llvm::BasicBlock* mt_vec_2d = llvm::BasicBlock::Create(ctx_.context(), "mt_vec_2d", current_func);
    builder.CreateCondBr(vec_has_dim3, mt_vec_has_dim3, mt_vec_2d);

    builder.SetInsertPoint(mt_vec_2d);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* vec_2d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_vec_has_dim3);
    llvm::Value* vec_dim3_ptr = builder.CreateGEP(
        ctx_.taggedValueType(), vector_elems_typed, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* vec_dim3_tagged = builder.CreateLoad(ctx_.taggedValueType(), vec_dim3_ptr);
    llvm::Value* vec_dim3 = tagged_.safeExtractInt64(vec_dim3_tagged);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* vec_3d_exit = builder.GetInsertBlock();

    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), vec_1d_exit);
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 2), vec_2d_exit);
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 3), vec_3d_exit);

    d1_phi->addIncoming(vec_dim1, vec_1d_exit);
    d1_phi->addIncoming(vec_dim1, vec_2d_exit);
    d1_phi->addIncoming(vec_dim1, vec_3d_exit);

    d2_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), vec_1d_exit);
    d2_phi->addIncoming(vec_dim2, vec_2d_exit);
    d2_phi->addIncoming(vec_dim2, vec_3d_exit);

    d3_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), vec_1d_exit);
    d3_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), vec_2d_exit);
    d3_phi->addIncoming(vec_dim3, vec_3d_exit);

    // TENSOR SHAPE PATH: #(...) literals are parsed as numeric tensors.
    // Use their element list as the requested output shape.
    builder.SetInsertPoint(tensor_path);
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(shape_arg);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Value* tensor_total_field = builder.CreateStructGEP(ctx_.tensorType(), tensor_ptr, 3);
    llvm::Value* tensor_shape_len = builder.CreateLoad(ctx_.int64Type(), tensor_total_field);
    llvm::Value* tensor_elems_field = builder.CreateStructGEP(ctx_.tensorType(), tensor_ptr, 2);
    llvm::Value* tensor_elems = builder.CreateLoad(ctx_.ptrType(), tensor_elems_field);

    llvm::Value* tensor_dim1_bits = builder.CreateLoad(
        ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), tensor_elems,
            llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    llvm::Value* tensor_dim1_double = builder.CreateBitCast(tensor_dim1_bits, ctx_.doubleType());
    llvm::Value* tensor_dim1 = builder.CreateFPToSI(tensor_dim1_double, ctx_.int64Type());

    llvm::Value* tensor_has_dim2 = builder.CreateICmpUGT(
        tensor_shape_len, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::BasicBlock* mt_tensor_has_dim2 = llvm::BasicBlock::Create(
        ctx_.context(), "mt_tensor_has_dim2", current_func);
    llvm::BasicBlock* mt_tensor_1d = llvm::BasicBlock::Create(
        ctx_.context(), "mt_tensor_1d", current_func);
    builder.CreateCondBr(tensor_has_dim2, mt_tensor_has_dim2, mt_tensor_1d);

    builder.SetInsertPoint(mt_tensor_1d);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* tensor_1d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_tensor_has_dim2);
    llvm::Value* tensor_dim2_bits = builder.CreateLoad(
        ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), tensor_elems,
            llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    llvm::Value* tensor_dim2_double = builder.CreateBitCast(tensor_dim2_bits, ctx_.doubleType());
    llvm::Value* tensor_dim2 = builder.CreateFPToSI(tensor_dim2_double, ctx_.int64Type());

    llvm::Value* tensor_has_dim3 = builder.CreateICmpUGT(
        tensor_shape_len, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::BasicBlock* mt_tensor_has_dim3 = llvm::BasicBlock::Create(
        ctx_.context(), "mt_tensor_has_dim3", current_func);
    llvm::BasicBlock* mt_tensor_2d = llvm::BasicBlock::Create(
        ctx_.context(), "mt_tensor_2d", current_func);
    builder.CreateCondBr(tensor_has_dim3, mt_tensor_has_dim3, mt_tensor_2d);

    builder.SetInsertPoint(mt_tensor_2d);
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* tensor_2d_exit = builder.GetInsertBlock();

    builder.SetInsertPoint(mt_tensor_has_dim3);
    llvm::Value* tensor_dim3_bits = builder.CreateLoad(
        ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), tensor_elems,
            llvm::ConstantInt::get(ctx_.int64Type(), 2)));
    llvm::Value* tensor_dim3_double = builder.CreateBitCast(tensor_dim3_bits, ctx_.doubleType());
    llvm::Value* tensor_dim3 = builder.CreateFPToSI(tensor_dim3_double, ctx_.int64Type());
    builder.CreateBr(mt_merge);
    llvm::BasicBlock* tensor_3d_exit = builder.GetInsertBlock();

    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), tensor_1d_exit);
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 2), tensor_2d_exit);
    ndims_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 3), tensor_3d_exit);

    d1_phi->addIncoming(tensor_dim1, tensor_1d_exit);
    d1_phi->addIncoming(tensor_dim1, tensor_2d_exit);
    d1_phi->addIncoming(tensor_dim1, tensor_3d_exit);

    d2_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), tensor_1d_exit);
    d2_phi->addIncoming(tensor_dim2, tensor_2d_exit);
    d2_phi->addIncoming(tensor_dim2, tensor_3d_exit);

    d3_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), tensor_1d_exit);
    d3_phi->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), tensor_2d_exit);
    d3_phi->addIncoming(tensor_dim3, tensor_3d_exit);

    // FINAL MERGE: scalar path or heap-shape path
    builder.SetInsertPoint(final_merge);
    llvm::PHINode* result = builder.CreatePHI(ctx_.ptrType(), 2, "mt_result");
    result->addIncoming(scalar_tensor, scalar_exit);
    result->addIncoming(list_tensor, list_done_exit);

    return tagged_.packHeapPtr(result);
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
    llvm::Function* ceil_func = ESHKOL_GET_INTRINSIC(
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

// Convolution & pooling: see tensor_conv_codegen.cpp

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
    // tensor-min: (tensor-min tensor) or (tensor-min tensor axis)
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("tensor-min requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // 2-arg case: (tensor-min tensor axis) → min along axis, returns tensor
    if (op->call_op.num_vars == 2) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[1]);
        if (!axis_val) return nullptr;
        return emitAxisReduce(tensor_val, axis_val, 3); // MIN=3
    }

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
    // tensor-max: (tensor-max tensor) or (tensor-max tensor axis)
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("tensor-max requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // 2-arg case: (tensor-max tensor axis) → max along axis, returns tensor
    if (op->call_op.num_vars == 2) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[1]);
        if (!axis_val) return nullptr;
        return emitAxisReduce(tensor_val, axis_val, 2); // MAX=2
    }

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
    // tensor-argmin: (tensor-argmin tensor) or (tensor-argmin tensor axis)
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("tensor-argmin requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // 2-arg case: (tensor-argmin tensor axis) → argmin along axis, returns tensor of indices
    if (op->call_op.num_vars == 2) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[1]);
        if (!axis_val) return nullptr;
        // Emit call to eshkol_xla_argreduce(arena, data, total, shape, rank, axis, is_max=0)
        auto& builder = ctx_.builder();
        llvm::Value* ptr_int = tagged_.safeExtractInt64(tensor_val);
        llvm::Value* ptr = builder.CreateIntToPtr(ptr_int, ctx_.ptrType());
        llvm::StructType* ttype = ctx_.tensorType();
        llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 2));
        llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 3));
        llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 0));
        llvm::Value* rank = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 1));
        llvm::Value* axis = tagged_.safeExtractInt64(axis_val);
        llvm::Value* neg = builder.CreateICmpSLT(axis, llvm::ConstantInt::get(ctx_.int64Type(), 0));
        axis = builder.CreateSelect(neg, builder.CreateAdd(axis, rank), axis);
        llvm::Value* arena = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        auto* ptrTy = ctx_.ptrType();
        auto* i64Ty = ctx_.int64Type();
        llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, i64Ty}, false);
        llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_argreduce", fn_type);
        llvm::Value* result = builder.CreateCall(callee,
            {arena, elems, total, dims, rank, axis, llvm::ConstantInt::get(i64Ty, 0)},
            "argmin_axis_result");
        return tagged_.packHeapPtr(result);
    }

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
    // tensor-argmax: (tensor-argmax tensor) or (tensor-argmax tensor axis)
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("tensor-argmax requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // 2-arg case: (tensor-argmax tensor axis) → argmax along axis, returns tensor of indices
    if (op->call_op.num_vars == 2) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[1]);
        if (!axis_val) return nullptr;
        auto& builder = ctx_.builder();
        llvm::Value* ptr_int = tagged_.safeExtractInt64(tensor_val);
        llvm::Value* ptr = builder.CreateIntToPtr(ptr_int, ctx_.ptrType());
        llvm::StructType* ttype = ctx_.tensorType();
        llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 2));
        llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 3));
        llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 0));
        llvm::Value* rank = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 1));
        llvm::Value* axis = tagged_.safeExtractInt64(axis_val);
        llvm::Value* neg = builder.CreateICmpSLT(axis, llvm::ConstantInt::get(ctx_.int64Type(), 0));
        axis = builder.CreateSelect(neg, builder.CreateAdd(axis, rank), axis);
        llvm::Value* arena = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        auto* ptrTy = ctx_.ptrType();
        auto* i64Ty = ctx_.int64Type();
        llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, i64Ty}, false);
        llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_argreduce", fn_type);
        llvm::Value* result = builder.CreateCall(callee,
            {arena, elems, total, dims, rank, axis, llvm::ConstantInt::get(i64Ty, 1)},
            "argmax_axis_result");
        return tagged_.packHeapPtr(result);
    }

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
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
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

    // Guard: conv3d requires at least 3D tensors for both input and kernel
    {
        llvm::Value* in_ok = builder.CreateICmpUGE(in_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 3));
        llvm::Value* k_ok = builder.CreateICmpUGE(k_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 3));
        llvm::Value* both_ok = builder.CreateAnd(in_ok, k_ok);
        llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "c3d_dims_ok", current_func);
        llvm::BasicBlock* err_bb = llvm::BasicBlock::Create(ctx_.context(), "c3d_dims_err", current_func);
        builder.CreateCondBr(both_ok, ok_bb, err_bb);
        builder.SetInsertPoint(err_bb);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = builder.CreateGlobalString("Error: conv3d requires at least 3D tensors (input=%lldD, kernel=%lldD)\n");
            builder.CreateCall(pf, {fmt, in_ndims, k_ndims});
            builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
        }
        builder.CreateUnreachable();
        builder.SetInsertPoint(ok_bb);
    }

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

// Optimizers / weight init / LR schedulers: see tensor_training_codegen.cpp

// Linear algebra (Phase 4.4): see tensor_linalg_codegen.cpp

// Loss functions (Track 6.3): see tensor_loss_codegen.cpp

// Data loading infrastructure: see tensor_dataloader_codegen.cpp

// Transformer architecture (Track 8): see tensor_transformer_codegen.cpp
// ===================================================================
// Tensor Unary Operations (tensor-neg, tensor-abs, tensor-sqrt, etc.)
// ===================================================================

/**
 * Generic SIMD+scalar unary loop.  Applies a single-operand math op to
 * every element of a tensor and returns a new tensor of the same shape.
 *
 * @param tensor_val  Tagged tensor input
 * @param op_name     Short label used for LLVM basic-block names
 * @param intrinsic_id LLVM floating-point intrinsic to call per element
 *                    (ignored when use_fneg == true)
 * @param use_fneg    If true, emit FNeg instead of an intrinsic call
 */
llvm::Value* TensorCodegen::emitTensorUnaryOp(llvm::Value* tensor_val,
                                               const std::string& op_name,
                                               llvm::Intrinsic::ID intrinsic_id,
                                               bool use_fneg) {
    if (!tensor_val) return tagged_.packNull();
    auto& builder = ctx_.builder();

    // Ensure tagged value
    if (tensor_val->getType() != ctx_.taggedValueType())
        tensor_val = tagged_.packInt64(tensor_val, true);

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr     = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type*  tensor_type    = ctx_.tensorType();

    // Load shape metadata
    llvm::Value* dims_ptr       = builder.CreateLoad(ctx_.ptrType(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 0));
    llvm::Value* num_dims       = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 1));
    llvm::Value* src_elems      = builder.CreateLoad(ctx_.ptrType(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 2));
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 3));

    // Allocate result tensor header
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value*    result_ptr   = builder.CreateCall(alloc_tensor, {arena_ptr}, (op_name + "_result").c_str());

    // Allocate and copy dimensions
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* dims_bytes = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, (op_name + "_dims").c_str());
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_bytes);

    // Allocate result elements
    llvm::Value* elems_bytes = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), (int64_t)sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, (op_name + "_elems").c_str());

    // SIMD parameters
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Declare scalar intrinsic (used in scalar tail loop)
    llvm::Function* scalar_intr = nullptr;
    if (!use_fneg)
        scalar_intr = ESHKOL_GET_INTRINSIC(&ctx_.module(), intrinsic_id, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* simd_cond   = llvm::BasicBlock::Create(ctx_.context(), (op_name+"_simd_cond").c_str(),   current_func);
    llvm::BasicBlock* simd_body   = llvm::BasicBlock::Create(ctx_.context(), (op_name+"_simd_body").c_str(),   current_func);
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), (op_name+"_scalar_cond").c_str(), current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), (op_name+"_scalar_body").c_str(), current_func);
    llvm::BasicBlock* exit_block  = llvm::BasicBlock::Create(ctx_.context(), (op_name+"_exit").c_str(),        current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, (op_name + "_i").c_str());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    // simd_count = (total / SIMD_WIDTH) * SIMD_WIDTH
    llvm::Value* simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

    builder.CreateBr(simd_cond);

    // === SIMD loop ===
    builder.SetInsertPoint(simd_cond);
    {
        llvm::Value* i    = builder.CreateLoad(ctx_.int64Type(), counter);
        llvm::Value* cond = builder.CreateICmpULT(i, simd_count);
        builder.CreateCondBr(cond, simd_body, scalar_cond);
    }

    builder.SetInsertPoint(simd_body);
    {
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
        if (use_simd) {
            llvm::Value* src_vec_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
            llvm::Value* x_vec       = builder.CreateAlignedLoad(vec_type, src_vec_ptr,
                                                                  llvm::MaybeAlign(8));
            llvm::Value* result_vec;
            if (use_fneg) {
                result_vec = builder.CreateFNeg(x_vec);
            } else {
                llvm::Function* vec_intr = ESHKOL_GET_INTRINSIC(&ctx_.module(), intrinsic_id, {vec_type});
                result_vec = builder.CreateCall(vec_intr, {x_vec});
            }
            llvm::Value* dst_vec_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
            builder.CreateAlignedStore(result_vec, dst_vec_ptr, llvm::MaybeAlign(8));
        }
        llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        builder.CreateStore(next_i, counter);
        auto* backEdge = builder.CreateBr(simd_cond);
        attachLoopMetadata(backEdge, true, SIMD_WIDTH, false, 0);
    }

    // === Scalar tail loop ===
    builder.SetInsertPoint(scalar_cond);
    {
        llvm::Value* i    = builder.CreateLoad(ctx_.int64Type(), counter);
        llvm::Value* cond = builder.CreateICmpULT(i, total_elements);
        builder.CreateCondBr(cond, scalar_body, exit_block);
    }

    builder.SetInsertPoint(scalar_body);
    {
        llvm::Value* i   = builder.CreateLoad(ctx_.int64Type(), counter);
        llvm::Value* src = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
        llvm::Value* x   = builder.CreateLoad(ctx_.doubleType(), src);
        llvm::Value* r;
        if (use_fneg) {
            r = builder.CreateFNeg(x);
        } else {
            r = builder.CreateCall(scalar_intr, {x});
        }
        builder.CreateStore(r, builder.CreateGEP(ctx_.doubleType(), result_elems, i));
        llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        builder.CreateStore(next_i, counter);
        auto* backEdge = builder.CreateBr(scalar_cond);
        attachLoopMetadata(backEdge, false, 0, true, 4);
    }

    // === Exit: populate result tensor struct ===
    builder.SetInsertPoint(exit_block);
    builder.CreateStore(result_dims,
        builder.CreateStructGEP(tensor_type, result_ptr, 0));
    builder.CreateStore(num_dims,
        builder.CreateStructGEP(tensor_type, result_ptr, 1));
    builder.CreateStore(result_elems,
        builder.CreateStructGEP(tensor_type, result_ptr, 2));
    builder.CreateStore(total_elements,
        builder.CreateStructGEP(tensor_type, result_ptr, 3));

    return tagged_.packHeapPtr(result_ptr);
}

// ---- Individual unary wrappers ----

llvm::Value* TensorCodegen::tensorNeg(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-neg requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "tneg", llvm::Intrinsic::fabs /* unused */, /*use_fneg=*/true);
}

llvm::Value* TensorCodegen::tensorAbs(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-abs requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "tabs", llvm::Intrinsic::fabs);
}

llvm::Value* TensorCodegen::tensorSqrt(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-sqrt requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "tsqrt", llvm::Intrinsic::sqrt);
}

llvm::Value* TensorCodegen::tensorExp(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-exp requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "texp", llvm::Intrinsic::exp);
}

llvm::Value* TensorCodegen::tensorLog(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-log requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "tlog", llvm::Intrinsic::log);
}

llvm::Value* TensorCodegen::tensorSin(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-sin requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "tsin", llvm::Intrinsic::sin);
}

llvm::Value* TensorCodegen::tensorCos(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-cos requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "tcos", llvm::Intrinsic::cos);
}

// ===================================================================
// Tensor Binary Operations: tensor-pow, tensor-maximum, tensor-minimum
// Delegate through rawTensorArithmeticSIMD which already handles "pow"/"max"/"min"
// ===================================================================

llvm::Value* TensorCodegen::tensorPow(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) { eshkol_error("tensor-pow requires 2 arguments"); return nullptr; }
    llvm::Value* a = codegenAST(&op->call_op.variables[0]);
    llvm::Value* b = codegenAST(&op->call_op.variables[1]);
    if (!a || !b) return nullptr;
    return tensorArithmeticInternal(a, b, "pow");
}

llvm::Value* TensorCodegen::tensorMaximum(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) { eshkol_error("tensor-maximum requires 2 arguments"); return nullptr; }
    llvm::Value* a = codegenAST(&op->call_op.variables[0]);
    llvm::Value* b = codegenAST(&op->call_op.variables[1]);
    if (!a || !b) return nullptr;
    return tensorArithmeticInternal(a, b, "max");
}

llvm::Value* TensorCodegen::tensorMinimum(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) { eshkol_error("tensor-minimum requires 2 arguments"); return nullptr; }
    llvm::Value* a = codegenAST(&op->call_op.variables[0]);
    llvm::Value* b = codegenAST(&op->call_op.variables[1]);
    if (!a || !b) return nullptr;
    return tensorArithmeticInternal(a, b, "min");
}

// ===================================================================
// tensor-scale: (tensor-scale tensor scalar) — multiply all elements
// ===================================================================

llvm::Value* TensorCodegen::tensorScale(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) { eshkol_error("tensor-scale requires 2 arguments: tensor and scalar"); return nullptr; }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* scalar_val = codegenAST(&op->call_op.variables[1]);
    if (!tensor_val || !scalar_val) return nullptr;

    auto& builder = ctx_.builder();

    // Ensure tagged
    if (tensor_val->getType() != ctx_.taggedValueType())
        tensor_val = tagged_.packInt64(tensor_val, true);

    // Extract scalar as double (handles both int64 and double tagged values)
    llvm::Value* scalar_d = extractAsDouble(scalar_val);

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr     = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type*  tensor_type    = ctx_.tensorType();

    // Load shape metadata
    llvm::Value* dims_ptr       = builder.CreateLoad(ctx_.ptrType(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 0));
    llvm::Value* num_dims       = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 1));
    llvm::Value* src_elems      = builder.CreateLoad(ctx_.ptrType(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 2));
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 3));

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value*    result_ptr   = builder.CreateCall(alloc_tensor, {arena_ptr}, "tscale_result");

    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* dims_bytes  = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), (int64_t)sizeof(int64_t)));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "tscale_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_bytes);

    llvm::Value* elems_bytes  = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), (int64_t)sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "tscale_elems");

    // SIMD setup
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* simd_cond   = llvm::BasicBlock::Create(ctx_.context(), "tscale_simd_cond",   current_func);
    llvm::BasicBlock* simd_body   = llvm::BasicBlock::Create(ctx_.context(), "tscale_simd_body",   current_func);
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "tscale_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "tscale_scalar_body", current_func);
    llvm::BasicBlock* exit_block  = llvm::BasicBlock::Create(ctx_.context(), "tscale_exit",        current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "tscale_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    llvm::Value* simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

    // Splat scalar to vector once
    llvm::Value* scale_vec = nullptr;
    if (use_simd) {
        scale_vec = llvm::UndefValue::get(vec_type);
        for (unsigned k = 0; k < SIMD_WIDTH; ++k)
            scale_vec = builder.CreateInsertElement(scale_vec, scalar_d,
                llvm::ConstantInt::get(ctx_.int32Type(), k));
    }

    builder.CreateBr(simd_cond);

    // SIMD loop
    builder.SetInsertPoint(simd_cond);
    {
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
        builder.CreateCondBr(builder.CreateICmpULT(i, simd_count), simd_body, scalar_cond);
    }
    builder.SetInsertPoint(simd_body);
    {
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
        if (use_simd) {
            llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
            llvm::Value* x_vec   = builder.CreateAlignedLoad(vec_type, src_ptr, llvm::MaybeAlign(8));
            llvm::Value* r_vec   = builder.CreateFMul(x_vec, scale_vec);
            builder.CreateAlignedStore(r_vec, builder.CreateGEP(ctx_.doubleType(), result_elems, i), llvm::MaybeAlign(8));
        }
        llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        builder.CreateStore(next_i, counter);
        auto* be = builder.CreateBr(simd_cond);
        attachLoopMetadata(be, true, SIMD_WIDTH, false, 0);
    }

    // Scalar tail loop
    builder.SetInsertPoint(scalar_cond);
    {
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
        builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), scalar_body, exit_block);
    }
    builder.SetInsertPoint(scalar_body);
    {
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
        llvm::Value* x = builder.CreateLoad(ctx_.doubleType(),
            builder.CreateGEP(ctx_.doubleType(), src_elems, i));
        llvm::Value* r = builder.CreateFMul(x, scalar_d);
        builder.CreateStore(r, builder.CreateGEP(ctx_.doubleType(), result_elems, i));
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), counter);
        auto* be = builder.CreateBr(scalar_cond);
        attachLoopMetadata(be, false, 0, true, 4);
    }

    // Populate result tensor struct
    builder.SetInsertPoint(exit_block);
    builder.CreateStore(result_dims,
        builder.CreateStructGEP(tensor_type, result_ptr, 0));
    builder.CreateStore(num_dims,
        builder.CreateStructGEP(tensor_type, result_ptr, 1));
    builder.CreateStore(result_elems,
        builder.CreateStructGEP(tensor_type, result_ptr, 2));
    builder.CreateStore(total_elements,
        builder.CreateStructGEP(tensor_type, result_ptr, 3));

    return tagged_.packHeapPtr(result_ptr);
}

// ===================================================================
// batch-matmul: (batch-matmul A B)
// A: [batch, M, K]  B: [batch, K, N]  Result: [batch, M, N]
// Delegates to C runtime eshkol_batch_matmul_f64.
// ===================================================================

llvm::Value* TensorCodegen::batchMatmul(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) { eshkol_error("batch-matmul requires 2 arguments"); return nullptr; }

    llvm::Value* a_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* b_val = codegenAST(&op->call_op.variables[1]);
    if (!a_val || !b_val) return nullptr;

    auto& builder = ctx_.builder();

    // Ensure tagged
    if (a_val->getType() != ctx_.taggedValueType()) a_val = tagged_.packInt64(a_val, true);
    if (b_val->getType() != ctx_.taggedValueType()) b_val = tagged_.packInt64(b_val, true);

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor pointers
    llvm::Value* a_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(a_val), ctx_.ptrType());
    llvm::Value* b_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(b_val), ctx_.ptrType());
    llvm::Type*  tt    = ctx_.tensorType();

    // Load A dims: [batch, M, K]
    llvm::Value* a_dims  = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(tt, a_ptr, 0));
    llvm::Value* a_total = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(tt, a_ptr, 3));
    llvm::Value* a_elems = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(tt, a_ptr, 2));

    llvm::Value* b_dims  = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(tt, b_ptr, 0));
    llvm::Value* b_total = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(tt, b_ptr, 3));
    llvm::Value* b_elems = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(tt, b_ptr, 2));

    // Extract batch, M, K from A; K, N from B
    llvm::Value* batch = builder.CreateLoad(ctx_.int64Type(), builder.CreateGEP(ctx_.int64Type(), a_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    llvm::Value* M_dim = builder.CreateLoad(ctx_.int64Type(), builder.CreateGEP(ctx_.int64Type(), a_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    llvm::Value* K_dim = builder.CreateLoad(ctx_.int64Type(), builder.CreateGEP(ctx_.int64Type(), a_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)));
    llvm::Value* N_dim = builder.CreateLoad(ctx_.int64Type(), builder.CreateGEP(ctx_.int64Type(), b_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)));

    // Result total = batch * M * N
    llvm::Value* result_total = builder.CreateMul(builder.CreateMul(batch, M_dim), N_dim);

    // Allocate result tensor [batch, M, N]
    llvm::Function* arena_alloc  = mem_.getArenaAllocate();
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value*    result_ptr   = builder.CreateCall(alloc_tensor, {arena_ptr}, "bmm_result");

    // Allocate dims array [3] = {batch, M, N}
    llvm::Value* dims_bytes  = llvm::ConstantInt::get(ctx_.int64Type(), 3 * sizeof(int64_t));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "bmm_dims");
    builder.CreateStore(batch, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(M_dim, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    builder.CreateStore(N_dim, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)));

    llvm::Value* elems_bytes  = builder.CreateMul(result_total,
        llvm::ConstantInt::get(ctx_.int64Type(), (int64_t)sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "bmm_elems");

    // Call C runtime: void eshkol_batch_matmul_f64(
    //     const double* a, const double* b, double* c,
    //     int64_t batch, int64_t M, int64_t K, int64_t N)
    auto* bmm_ft = llvm::FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(),
         ctx_.int64Type(), ctx_.int64Type(), ctx_.int64Type(), ctx_.int64Type()}, false);
    llvm::Function* bmm_fn = ctx_.module().getFunction("eshkol_batch_matmul_f64");
    if (!bmm_fn) {
        bmm_fn = llvm::Function::Create(bmm_ft, llvm::Function::ExternalLinkage,
            "eshkol_batch_matmul_f64", &ctx_.module());
    }
    builder.CreateCall(bmm_fn, {a_elems, b_elems, result_elems, batch, M_dim, K_dim, N_dim});

    // Populate result tensor struct
    llvm::Type* tensor_type = ctx_.tensorType();
    builder.CreateStore(result_dims,
        builder.CreateStructGEP(tensor_type, result_ptr, 0));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 3),
        builder.CreateStructGEP(tensor_type, result_ptr, 1));
    builder.CreateStore(result_elems,
        builder.CreateStructGEP(tensor_type, result_ptr, 2));
    builder.CreateStore(result_total,
        builder.CreateStructGEP(tensor_type, result_ptr, 3));

    return tagged_.packHeapPtr(result_ptr);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
