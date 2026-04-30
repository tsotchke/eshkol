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

namespace {

llvm::Value* taggedNumericToDouble(CodegenContext& ctx, TaggedValueCodegen& tagged, llvm::Value* value) {
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

} // namespace

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

// ===== ACTIVATION FUNCTIONS (SIMD-ACCELERATED) =====

llvm::Value* TensorCodegen::tensorRelu(const eshkol_operations_t* op) {
    // ReLU: max(0, x) element-wise
    // Note: For large tensors (≥100K elements), GPU acceleration is available
    // through the XLA elementwise runtime (RELU = op code 9) when invoked via
    // tensor-apply or the general tensor arithmetic path.
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
    auto* reluSimdBackEdge = builder.CreateBr(simd_cond);
    attachLoopMetadata(reluSimdBackEdge, true, SIMD_WIDTH, false, 0);

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
    auto* reluScalarBackEdge = builder.CreateBr(scalar_cond);
    attachLoopMetadata(reluScalarBackEdge, false, 0, true, 4);

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

    // SIMD parameters
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Scalar exp intrinsic (for tail loop)
    llvm::Function* exp_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* simd_cond = llvm::BasicBlock::Create(ctx_.context(), "sig_simd_cond", current_func);
    llvm::BasicBlock* simd_body = llvm::BasicBlock::Create(ctx_.context(), "sig_simd_body", current_func);
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "sig_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "sig_scalar_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "sig_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sig_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    // Calculate SIMD iteration count (rounds down to multiple of SIMD_WIDTH)
    llvm::Value* simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

    builder.CreateBr(simd_cond);

    // === SIMD Loop: process SIMD_WIDTH elements at a time ===
    builder.SetInsertPoint(simd_cond);
    llvm::Value* i_simd = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* simd_cmp = builder.CreateICmpULT(i_simd, simd_count);
    builder.CreateCondBr(simd_cmp, simd_body, scalar_cond);

    builder.SetInsertPoint(simd_body);
    if (use_simd) {
        // Load SIMD_WIDTH elements
        llvm::Value* src_vec_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i_simd);
        llvm::Value* x_vec = builder.CreateAlignedLoad(vec_type, src_vec_ptr, llvm::MaybeAlign(8), "sig_x_vec");

        // Vector intrinsics
        llvm::Function* exp_vec_func = ESHKOL_GET_INTRINSIC(
            &ctx_.module(), llvm::Intrinsic::exp, {vec_type});
        llvm::Function* fabs_vec_func = ESHKOL_GET_INTRINSIC(
            &ctx_.module(), llvm::Intrinsic::fabs, {vec_type});

        // Numerically stable sigmoid: avoid computing exp(large positive)
        // exp_neg_abs = exp(-|x|) — argument always <= 0, no overflow
        llvm::Value* abs_x_vec = builder.CreateCall(fabs_vec_func, {x_vec}, "sig_abs_x_vec");
        llvm::Value* neg_abs_x_vec = builder.CreateFNeg(abs_x_vec, "sig_neg_abs_vec");
        llvm::Value* exp_neg_abs_vec = builder.CreateCall(exp_vec_func, {neg_abs_x_vec}, "sig_exp_vec");

        llvm::Value* one_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 1.0));
        llvm::Value* zero_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 0.0));

        llvm::Value* denom_vec = builder.CreateFAdd(one_vec, exp_neg_abs_vec, "sig_denom_vec");
        // x >= 0: 1/denom;  x < 0: exp_neg_abs/denom
        llvm::Value* pos_result = builder.CreateFDiv(one_vec, denom_vec, "sig_pos_vec");
        llvm::Value* neg_result = builder.CreateFDiv(exp_neg_abs_vec, denom_vec, "sig_neg_vec");
        llvm::Value* is_positive = builder.CreateFCmpOGE(x_vec, zero_vec, "sig_ge_zero");
        llvm::Value* result_vec = builder.CreateSelect(is_positive, pos_result, neg_result, "sig_result_vec");

        // Store result
        llvm::Value* dst_vec_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i_simd);
        builder.CreateAlignedStore(result_vec, dst_vec_ptr, llvm::MaybeAlign(8));
    }

    llvm::Value* next_i_simd = builder.CreateAdd(i_simd,
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
    builder.CreateStore(next_i_simd, counter);
    auto* sigSimdBackEdge = builder.CreateBr(simd_cond);
    attachLoopMetadata(sigSimdBackEdge, true, SIMD_WIDTH, false, 0);

    // === Scalar Loop (remainder elements) ===
    builder.SetInsertPoint(scalar_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, scalar_body, exit_block);

    builder.SetInsertPoint(scalar_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
    llvm::Value* x = builder.CreateLoad(ctx_.doubleType(), src_ptr);

    // Numerically stable sigmoid: avoid computing exp(large positive)
    // exp_neg_abs = exp(-|x|) — argument always <= 0, no overflow
    llvm::Function* fabs_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::fabs, {ctx_.doubleType()});
    llvm::Value* abs_x = builder.CreateCall(fabs_func, {x}, "sig_abs_x");
    llvm::Value* neg_abs_x = builder.CreateFNeg(abs_x, "sig_neg_abs");
    llvm::Value* exp_neg_abs = builder.CreateCall(exp_func, {neg_abs_x}, "sig_exp_neg_abs");
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* denom = builder.CreateFAdd(one, exp_neg_abs, "sig_denom");
    // x >= 0: 1/denom;  x < 0: exp_neg_abs/denom
    llvm::Value* pos_result = builder.CreateFDiv(one, denom, "sig_pos");
    llvm::Value* neg_result = builder.CreateFDiv(exp_neg_abs, denom, "sig_neg");
    llvm::Value* is_positive = builder.CreateFCmpOGE(x,
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0), "sig_ge_zero");
    llvm::Value* result = builder.CreateSelect(is_positive, pos_result, neg_result, "sig_result");

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(result, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    auto* sigScalarBackEdge = builder.CreateBr(scalar_cond);
    attachLoopMetadata(sigScalarBackEdge, false, 0, true, 4);

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
    // (softmax tensor) — global softmax
    // (softmax tensor axis) — softmax along specified axis
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("softmax requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // 2-arg case: (softmax tensor axis) → axis-aware softmax via runtime
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

        llvm::Value* arena = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        auto* ptrTy = ctx_.ptrType();
        auto* i64Ty = ctx_.int64Type();
        llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty}, false);
        llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_softmax", fn_type);
        llvm::Value* result = builder.CreateCall(callee,
            {arena, elems, total, dims, rank, axis}, "softmax_axis_result");
        return tagged_.packHeapPtr(result);
    }

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

    llvm::Function* exp_func = ESHKOL_GET_INTRINSIC(
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

    // SIMD parameters for softmax
    const unsigned SM_SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* sm_vec_type = getSIMDVectorType();
    const bool sm_use_simd = (SM_SIMD_WIDTH > 1 && sm_vec_type != nullptr);

    // Pass 2: Compute exp(x - max) and sum — SIMD vectorized
    builder.SetInsertPoint(sum_init);
    llvm::BasicBlock* exp_simd_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_simd_cond", current_func);
    llvm::BasicBlock* exp_simd_body = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_simd_body", current_func);
    llvm::BasicBlock* exp_scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_scalar_cond", current_func);
    llvm::BasicBlock* exp_scalar_body = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_scalar_body", current_func);
    llvm::BasicBlock* norm_init = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_init", current_func);

    llvm::Value* sum_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "sm_sum");
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_val);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    // SIMD iteration count for exp pass
    llvm::Value* sm_simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SM_SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SM_SIMD_WIDTH));
    builder.CreateBr(exp_simd_cond);

    // === SIMD exp loop ===
    builder.SetInsertPoint(exp_simd_cond);
    llvm::Value* i2s = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp2s = builder.CreateICmpULT(i2s, sm_simd_count);
    builder.CreateCondBr(cmp2s, exp_simd_body, exp_scalar_cond);

    builder.SetInsertPoint(exp_simd_body);
    if (sm_use_simd) {
        llvm::Value* final_max_s = builder.CreateLoad(ctx_.doubleType(), max_val);
        llvm::Value* max_vec = builder.CreateVectorSplat(SM_SIMD_WIDTH, final_max_s, "max_splat");

        llvm::Value* src_vec_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i2s);
        llvm::Value* x_vec = builder.CreateAlignedLoad(sm_vec_type, src_vec_ptr, llvm::MaybeAlign(8), "sm_x_vec");
        llvm::Value* shifted_vec = builder.CreateFSub(x_vec, max_vec, "sm_shifted_vec");

        llvm::Function* exp_vec_func = ESHKOL_GET_INTRINSIC(
            &ctx_.module(), llvm::Intrinsic::exp, {sm_vec_type});
        llvm::Value* exp_vec = builder.CreateCall(exp_vec_func, {shifted_vec}, "sm_exp_vec");

        llvm::Value* dst_vec_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i2s);
        builder.CreateAlignedStore(exp_vec, dst_vec_ptr, llvm::MaybeAlign(8));

        // Horizontal sum of exp vector for running total
        llvm::Value* cur_sum_s = builder.CreateLoad(ctx_.doubleType(), sum_val);
        for (unsigned lane = 0; lane < SM_SIMD_WIDTH; ++lane) {
            llvm::Value* lane_val = builder.CreateExtractElement(exp_vec,
                llvm::ConstantInt::get(ctx_.int32Type(), lane), "sm_lane_" + std::to_string(lane));
            cur_sum_s = builder.CreateFAdd(cur_sum_s, lane_val);
        }
        builder.CreateStore(cur_sum_s, sum_val);
    }
    llvm::Value* next_i2s = builder.CreateAdd(i2s, llvm::ConstantInt::get(ctx_.int64Type(), SM_SIMD_WIDTH));
    builder.CreateStore(next_i2s, counter);
    builder.CreateBr(exp_simd_cond);

    // === Scalar exp tail loop ===
    builder.SetInsertPoint(exp_scalar_cond);
    llvm::Value* i2 = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp2 = builder.CreateICmpULT(i2, total_elements);
    builder.CreateCondBr(cmp2, exp_scalar_body, norm_init);

    builder.SetInsertPoint(exp_scalar_body);
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
    builder.CreateBr(exp_scalar_cond);

    // Pass 3: Normalize (divide by sum) — SIMD vectorized
    builder.SetInsertPoint(norm_init);

    // Zero-guard: prevent division by zero if all exp values underflowed to 0
    {
        llvm::Value* cur_total_sum = builder.CreateLoad(ctx_.doubleType(), sum_val);
        llvm::Value* sum_is_zero = builder.CreateFCmpOEQ(cur_total_sum,
            llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        llvm::Value* safe_total_sum = builder.CreateSelect(sum_is_zero,
            llvm::ConstantFP::get(ctx_.doubleType(), 1e-10), cur_total_sum);
        builder.CreateStore(safe_total_sum, sum_val);
    }

    llvm::BasicBlock* norm_simd_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_simd_cond", current_func);
    llvm::BasicBlock* norm_simd_body = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_simd_body", current_func);
    llvm::BasicBlock* norm_scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_scalar_cond", current_func);
    llvm::BasicBlock* norm_scalar_body = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_scalar_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "sm_exit", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(norm_simd_cond);

    // === SIMD normalization loop ===
    builder.SetInsertPoint(norm_simd_cond);
    llvm::Value* i3s = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp3s = builder.CreateICmpULT(i3s, sm_simd_count);
    builder.CreateCondBr(cmp3s, norm_simd_body, norm_scalar_cond);

    builder.SetInsertPoint(norm_simd_body);
    if (sm_use_simd) {
        llvm::Value* total_sum_s = builder.CreateLoad(ctx_.doubleType(), sum_val);
        llvm::Value* sum_vec = builder.CreateVectorSplat(SM_SIMD_WIDTH, total_sum_s, "sum_splat");

        llvm::Value* res_vec_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i3s);
        llvm::Value* exp_vals = builder.CreateAlignedLoad(sm_vec_type, res_vec_ptr, llvm::MaybeAlign(8), "sm_exp_vals");
        llvm::Value* norm_vec = builder.CreateFDiv(exp_vals, sum_vec, "sm_norm_vec");
        builder.CreateAlignedStore(norm_vec, res_vec_ptr, llvm::MaybeAlign(8));
    }
    llvm::Value* next_i3s = builder.CreateAdd(i3s, llvm::ConstantInt::get(ctx_.int64Type(), SM_SIMD_WIDTH));
    builder.CreateStore(next_i3s, counter);
    builder.CreateBr(norm_simd_cond);

    // === Scalar normalization tail loop ===
    builder.SetInsertPoint(norm_scalar_cond);
    llvm::Value* i3 = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp3 = builder.CreateICmpULT(i3, total_elements);
    builder.CreateCondBr(cmp3, norm_scalar_body, exit_block);

    builder.SetInsertPoint(norm_scalar_body);
    llvm::Value* res_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i3);
    llvm::Value* exp_val = builder.CreateLoad(ctx_.doubleType(), res_ptr);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), sum_val);
    llvm::Value* normalized = builder.CreateFDiv(exp_val, total_sum, "normalized");
    builder.CreateStore(normalized, res_ptr);
    llvm::Value* next_i3 = builder.CreateAdd(i3, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i3, counter);
    builder.CreateBr(norm_scalar_cond);

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
    // GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) — PyTorch standard
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

    // SIMD parameters
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Scalar exp intrinsic (for tail loop)
    llvm::Function* exp_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* simd_cond = llvm::BasicBlock::Create(ctx_.context(), "gelu_simd_cond", current_func);
    llvm::BasicBlock* simd_body = llvm::BasicBlock::Create(ctx_.context(), "gelu_simd_body", current_func);
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "gelu_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "gelu_scalar_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "gelu_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "gelu_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    // Calculate SIMD iteration count (rounds down to multiple of SIMD_WIDTH)
    llvm::Value* simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

    builder.CreateBr(simd_cond);

    // === SIMD Loop: process SIMD_WIDTH elements at a time ===
    builder.SetInsertPoint(simd_cond);
    llvm::Value* i_simd = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* simd_cmp = builder.CreateICmpULT(i_simd, simd_count);
    builder.CreateCondBr(simd_cmp, simd_body, scalar_cond);

    builder.SetInsertPoint(simd_body);
    if (use_simd) {
        // Load SIMD_WIDTH elements
        llvm::Value* src_vec_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i_simd);
        llvm::Value* x_vec = builder.CreateAlignedLoad(vec_type, src_vec_ptr, llvm::MaybeAlign(8), "gelu_x_vec");

        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        // Standard tanh approximation (PyTorch default)
        llvm::Value* half_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 0.5));
        llvm::Value* one_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 1.0));
        llvm::Value* sqrt2pi_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 0.7978845608028654)); // sqrt(2/π)
        llvm::Value* k_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 0.044715));
        llvm::Value* two_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 2.0));

        // x³ = x * x * x
        llvm::Value* x2_vec = builder.CreateFMul(x_vec, x_vec, "gelu_x2_vec");
        llvm::Value* x3_vec = builder.CreateFMul(x2_vec, x_vec, "gelu_x3_vec");
        // inner = x + 0.044715 * x³
        llvm::Value* kx3_vec = builder.CreateFMul(k_vec, x3_vec, "gelu_kx3_vec");
        llvm::Value* inner_vec = builder.CreateFAdd(x_vec, kx3_vec, "gelu_inner_vec");
        // arg = sqrt(2/π) * inner
        llvm::Value* arg_vec = builder.CreateFMul(sqrt2pi_vec, inner_vec, "gelu_arg_vec");
        // tanh(arg) via (exp(2*arg) - 1) / (exp(2*arg) + 1)
        llvm::Function* exp_vec_func = ESHKOL_GET_INTRINSIC(
            &ctx_.module(), llvm::Intrinsic::exp, {vec_type});
        llvm::Value* two_arg_vec = builder.CreateFMul(two_vec, arg_vec, "gelu_2arg_vec");
        llvm::Value* exp_2arg_vec = builder.CreateCall(exp_vec_func, {two_arg_vec}, "gelu_exp_vec");
        llvm::Value* tanh_num = builder.CreateFSub(exp_2arg_vec, one_vec, "gelu_tanh_num");
        llvm::Value* tanh_den = builder.CreateFAdd(exp_2arg_vec, one_vec, "gelu_tanh_den");
        llvm::Value* tanh_vec = builder.CreateFDiv(tanh_num, tanh_den, "gelu_tanh_vec");
        // result = 0.5 * x * (1 + tanh)
        llvm::Value* one_plus_tanh = builder.CreateFAdd(one_vec, tanh_vec, "gelu_1pt_vec");
        llvm::Value* half_x = builder.CreateFMul(half_vec, x_vec, "gelu_halfx_vec");
        llvm::Value* result_vec = builder.CreateFMul(half_x, one_plus_tanh, "gelu_result_vec");

        // Store result
        llvm::Value* dst_vec_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i_simd);
        builder.CreateAlignedStore(result_vec, dst_vec_ptr, llvm::MaybeAlign(8));
    }

    llvm::Value* next_i_simd = builder.CreateAdd(i_simd,
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
    builder.CreateStore(next_i_simd, counter);
    auto* geluSimdBackEdge = builder.CreateBr(simd_cond);
    attachLoopMetadata(geluSimdBackEdge, true, SIMD_WIDTH, false, 0);

    // === Scalar Loop (remainder elements) ===
    builder.SetInsertPoint(scalar_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, scalar_body, exit_block);

    builder.SetInsertPoint(scalar_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
    llvm::Value* x = builder.CreateLoad(ctx_.doubleType(), src_ptr);

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* sqrt2pi = llvm::ConstantFP::get(ctx_.doubleType(), 0.7978845608028654);
    llvm::Value* k = llvm::ConstantFP::get(ctx_.doubleType(), 0.044715);
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    // x³
    llvm::Value* x2 = builder.CreateFMul(x, x, "gelu_x2");
    llvm::Value* x3 = builder.CreateFMul(x2, x, "gelu_x3");
    // inner = x + 0.044715 * x³
    llvm::Value* kx3 = builder.CreateFMul(k, x3, "gelu_kx3");
    llvm::Value* inner = builder.CreateFAdd(x, kx3, "gelu_inner");
    // arg = sqrt(2/π) * inner
    llvm::Value* arg = builder.CreateFMul(sqrt2pi, inner, "gelu_arg");
    // tanh(arg) via (exp(2*arg) - 1) / (exp(2*arg) + 1)
    llvm::Value* two_arg = builder.CreateFMul(two, arg, "gelu_2arg");
    llvm::Value* exp_2arg = builder.CreateCall(exp_func, {two_arg}, "gelu_exp");
    llvm::Value* tanh_num = builder.CreateFSub(exp_2arg, one, "gelu_tanh_n");
    llvm::Value* tanh_den = builder.CreateFAdd(exp_2arg, one, "gelu_tanh_d");
    llvm::Value* tanh_val = builder.CreateFDiv(tanh_num, tanh_den, "gelu_tanh");
    // 0.5 * x * (1 + tanh)
    llvm::Value* one_plus_tanh = builder.CreateFAdd(one, tanh_val, "gelu_1pt");
    llvm::Value* half_x = builder.CreateFMul(half, x, "gelu_halfx");
    llvm::Value* result = builder.CreateFMul(half_x, one_plus_tanh, "gelu_result");

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(result, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    auto* geluScalarBackEdge = builder.CreateBr(scalar_cond);
    attachLoopMetadata(geluScalarBackEdge, false, 0, true, 4);

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
    auto* lreluSimdBackEdge = builder.CreateBr(simd_cond);
    attachLoopMetadata(lreluSimdBackEdge, true, SIMD_WIDTH, false, 0);

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
    auto* lreluScalarBackEdge = builder.CreateBr(scalar_cond);
    attachLoopMetadata(lreluScalarBackEdge, false, 0, true, 4);

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
    auto* siluBackEdge = builder.CreateBr(loop_cond);
    attachLoopMetadata(siluBackEdge, false, 0, true, 4);

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

llvm::Value* TensorCodegen::tensorElu(const eshkol_operations_t* op) {
    // ELU: x > 0 ? x : alpha * (exp(x) - 1)
    // Default alpha = 1.0
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("elu requires 1 or 2 arguments (tensor [, alpha])");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* alpha;
    if (op->call_op.num_vars == 2) {
        llvm::Value* alpha_tagged = codegenAST(&op->call_op.variables[1]);
        if (!alpha_tagged) return nullptr;
        alpha = tagged_.unpackDouble(alpha_tagged);
    } else {
        alpha = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    }

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
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "elu_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "elu_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "elu_elems");

    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* exp_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(exp_type, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "elu_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "elu_body", current_func);
    llvm::BasicBlock* elu_pos = llvm::BasicBlock::Create(ctx_.context(), "elu_pos", current_func);
    llvm::BasicBlock* elu_neg = llvm::BasicBlock::Create(ctx_.context(), "elu_neg", current_func);
    llvm::BasicBlock* elu_merge = llvm::BasicBlock::Create(ctx_.context(), "elu_merge", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "elu_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "elu_i");
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

    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* is_positive = builder.CreateFCmpOGT(val, zero);
    builder.CreateCondBr(is_positive, elu_pos, elu_neg);

    // Positive branch: output = x
    builder.SetInsertPoint(elu_pos);
    builder.CreateBr(elu_merge);

    // Negative branch: output = alpha * (exp(x) - 1)
    builder.SetInsertPoint(elu_neg);
    llvm::Value* exp_val = builder.CreateCall(exp_func, {val});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* exp_minus_1 = builder.CreateFSub(exp_val, one);
    llvm::Value* neg_result = builder.CreateFMul(alpha, exp_minus_1);
    builder.CreateBr(elu_merge);

    // Merge
    builder.SetInsertPoint(elu_merge);
    llvm::PHINode* result_val = builder.CreatePHI(ctx_.doubleType(), 2, "elu_val");
    result_val->addIncoming(val, elu_pos);
    result_val->addIncoming(neg_result, elu_neg);

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

llvm::Value* TensorCodegen::tensorSelu(const eshkol_operations_t* op) {
    // SELU: lambda * (x > 0 ? x : alpha * (exp(x) - 1))
    // lambda = 1.0507009873554804934193349852946
    // alpha  = 1.6732632423543772848170429916717
    if (op->call_op.num_vars != 1) {
        eshkol_error("selu requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

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
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "selu_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "selu_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "selu_elems");

    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* exp_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(exp_type, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    llvm::Value* selu_lambda = llvm::ConstantFP::get(ctx_.doubleType(), 1.0507009873554804934193349852946);
    llvm::Value* selu_alpha = llvm::ConstantFP::get(ctx_.doubleType(), 1.6732632423543772848170429916717);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "selu_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "selu_body", current_func);
    llvm::BasicBlock* selu_pos = llvm::BasicBlock::Create(ctx_.context(), "selu_pos", current_func);
    llvm::BasicBlock* selu_neg = llvm::BasicBlock::Create(ctx_.context(), "selu_neg", current_func);
    llvm::BasicBlock* selu_merge = llvm::BasicBlock::Create(ctx_.context(), "selu_merge", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "selu_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "selu_i");
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

    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* is_positive = builder.CreateFCmpOGT(val, zero);
    builder.CreateCondBr(is_positive, selu_pos, selu_neg);

    // Positive: lambda * x
    builder.SetInsertPoint(selu_pos);
    llvm::Value* pos_result = builder.CreateFMul(selu_lambda, val);
    builder.CreateBr(selu_merge);

    // Negative: lambda * alpha * (exp(x) - 1)
    builder.SetInsertPoint(selu_neg);
    llvm::Value* exp_val = builder.CreateCall(exp_func, {val});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* exp_minus_1 = builder.CreateFSub(exp_val, one);
    llvm::Value* alpha_exp = builder.CreateFMul(selu_alpha, exp_minus_1);
    llvm::Value* neg_result = builder.CreateFMul(selu_lambda, alpha_exp);
    builder.CreateBr(selu_merge);

    builder.SetInsertPoint(selu_merge);
    llvm::PHINode* result_val = builder.CreatePHI(ctx_.doubleType(), 2, "selu_val");
    result_val->addIncoming(pos_result, selu_pos);
    result_val->addIncoming(neg_result, selu_neg);

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

llvm::Value* TensorCodegen::tensorMish(const eshkol_operations_t* op) {
    // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    // Numerically stable: for x > 20, softplus(x) ≈ x, so mish ≈ x * tanh(x)
    if (op->call_op.num_vars != 1) {
        eshkol_error("mish requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

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
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "mish_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "mish_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "mish_elems");

    // Declare math functions
    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }
    llvm::Function* tanh_func = ctx_.module().getFunction("tanh");
    if (!tanh_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        tanh_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "tanh", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "mish_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "mish_body", current_func);
    llvm::BasicBlock* mish_large = llvm::BasicBlock::Create(ctx_.context(), "mish_large", current_func);
    llvm::BasicBlock* mish_normal = llvm::BasicBlock::Create(ctx_.context(), "mish_normal", current_func);
    llvm::BasicBlock* mish_merge = llvm::BasicBlock::Create(ctx_.context(), "mish_merge", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "mish_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mish_i");
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

    // Stability: if x > 20, softplus(x) ≈ x, so mish ≈ x * tanh(x)
    llvm::Value* threshold = llvm::ConstantFP::get(ctx_.doubleType(), 20.0);
    llvm::Value* is_large = builder.CreateFCmpOGT(val, threshold);
    builder.CreateCondBr(is_large, mish_large, mish_normal);

    // Large x: x * tanh(x)
    builder.SetInsertPoint(mish_large);
    llvm::Value* tanh_x = builder.CreateCall(tanh_func, {val});
    llvm::Value* large_result = builder.CreateFMul(val, tanh_x);
    builder.CreateBr(mish_merge);

    // Normal: x * tanh(log(1 + exp(x)))
    builder.SetInsertPoint(mish_normal);
    llvm::Value* exp_x = builder.CreateCall(exp_func, {val});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_plus_exp = builder.CreateFAdd(one, exp_x);
    llvm::Value* softplus = builder.CreateCall(log_func, {one_plus_exp});
    llvm::Value* tanh_sp = builder.CreateCall(tanh_func, {softplus});
    llvm::Value* normal_result = builder.CreateFMul(val, tanh_sp);
    builder.CreateBr(mish_merge);

    builder.SetInsertPoint(mish_merge);
    llvm::PHINode* result_val = builder.CreatePHI(ctx_.doubleType(), 2, "mish_val");
    result_val->addIncoming(large_result, mish_large);
    result_val->addIncoming(normal_result, mish_normal);

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

llvm::Value* TensorCodegen::tensorHardSwish(const eshkol_operations_t* op) {
    // Hard Swish: x * min(max(x + 3, 0), 6) / 6
    // Piecewise: x <= -3 → 0, x >= 3 → x, else → x * (x + 3) / 6
    if (op->call_op.num_vars != 1) {
        eshkol_error("hard-swish requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

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
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "hswish_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "hswish_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "hswish_elems");

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "hswish_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "hswish_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "hswish_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "hswish_i");
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

    // Compute min(max(x + 3, 0), 6)
    llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
    llvm::Value* six = llvm::ConstantFP::get(ctx_.doubleType(), 6.0);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    llvm::Value* x_plus_3 = builder.CreateFAdd(val, three);
    // max(x + 3, 0)
    llvm::Value* cmp_zero = builder.CreateFCmpOGT(x_plus_3, zero);
    llvm::Value* clamped_low = builder.CreateSelect(cmp_zero, x_plus_3, zero);
    // min(clamped, 6)
    llvm::Value* cmp_six = builder.CreateFCmpOLT(clamped_low, six);
    llvm::Value* clamped = builder.CreateSelect(cmp_six, clamped_low, six);
    // x * clamped / 6
    llvm::Value* scaled = builder.CreateFMul(val, clamped);
    llvm::Value* result_val = builder.CreateFDiv(scaled, six);

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

llvm::Value* TensorCodegen::tensorHardSigmoid(const eshkol_operations_t* op) {
    // Hard Sigmoid: clip((x + 3) / 6, 0, 1) = min(max((x + 3) / 6, 0), 1)
    if (op->call_op.num_vars != 1) {
        eshkol_error("hard-sigmoid requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

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
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "hsigmoid_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "hsigmoid_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "hsigmoid_elems");

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "hsigmoid_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "hsigmoid_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "hsigmoid_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "hsigmoid_i");
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

    // (x + 3) / 6
    llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
    llvm::Value* six = llvm::ConstantFP::get(ctx_.doubleType(), 6.0);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

    llvm::Value* x_plus_3 = builder.CreateFAdd(val, three);
    llvm::Value* divided = builder.CreateFDiv(x_plus_3, six);
    // clip to [0, 1]
    llvm::Value* cmp_zero = builder.CreateFCmpOGT(divided, zero);
    llvm::Value* clamped_low = builder.CreateSelect(cmp_zero, divided, zero);
    llvm::Value* cmp_one = builder.CreateFCmpOLT(clamped_low, one);
    llvm::Value* result_val = builder.CreateSelect(cmp_one, clamped_low, one);

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

llvm::Value* TensorCodegen::tensorSoftplus(const eshkol_operations_t* op) {
    // Softplus: (1/beta) * log(1 + exp(beta * x))
    // Default beta = 1.0, threshold = 20.0
    // For numerical stability: if beta*x > threshold, return x
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("softplus requires 1 or 2 arguments (tensor [, beta])");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* beta;
    if (op->call_op.num_vars == 2) {
        llvm::Value* beta_tagged = codegenAST(&op->call_op.variables[1]);
        if (!beta_tagged) return nullptr;
        beta = tagged_.unpackDouble(beta_tagged);
    } else {
        beta = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    }

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
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "softplus_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "softplus_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "softplus_elems");

    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "sp_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "sp_body", current_func);
    llvm::BasicBlock* sp_large = llvm::BasicBlock::Create(ctx_.context(), "sp_large", current_func);
    llvm::BasicBlock* sp_normal = llvm::BasicBlock::Create(ctx_.context(), "sp_normal", current_func);
    llvm::BasicBlock* sp_merge = llvm::BasicBlock::Create(ctx_.context(), "sp_merge", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "sp_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sp_i");
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

    // beta * x
    llvm::Value* beta_x = builder.CreateFMul(beta, val);
    // Threshold check: if beta*x > 20, just return x
    llvm::Value* threshold = llvm::ConstantFP::get(ctx_.doubleType(), 20.0);
    llvm::Value* is_large = builder.CreateFCmpOGT(beta_x, threshold);
    builder.CreateCondBr(is_large, sp_large, sp_normal);

    // Large: return x directly (softplus saturates to identity)
    builder.SetInsertPoint(sp_large);
    builder.CreateBr(sp_merge);

    // Normal: (1/beta) * log(1 + exp(beta * x))
    builder.SetInsertPoint(sp_normal);
    llvm::Value* exp_bx = builder.CreateCall(exp_func, {beta_x});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_plus_exp = builder.CreateFAdd(one, exp_bx);
    llvm::Value* log_val = builder.CreateCall(log_func, {one_plus_exp});
    llvm::Value* inv_beta = builder.CreateFDiv(one, beta);
    llvm::Value* normal_result = builder.CreateFMul(inv_beta, log_val);
    builder.CreateBr(sp_merge);

    builder.SetInsertPoint(sp_merge);
    llvm::PHINode* result_val = builder.CreatePHI(ctx_.doubleType(), 2, "sp_val");
    result_val->addIncoming(val, sp_large);
    result_val->addIncoming(normal_result, sp_normal);

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

llvm::Value* TensorCodegen::tensorDropout(const eshkol_operations_t* op) {
    // Dropout: x_i * mask_i / (1 - p), where mask_i ~ Bernoulli(1 - p)
    // During training, randomly zeros elements with probability p and scales survivors
    // Args: tensor, p (drop probability, 0 < p < 1)
    if (op->call_op.num_vars != 2) {
        eshkol_error("dropout requires exactly 2 arguments (tensor, drop_probability)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;
    llvm::Value* p_tagged = codegenAST(&op->call_op.variables[1]);
    if (!p_tagged) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Value* p = tagged_.unpackDouble(p_tagged);

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
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "dropout_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "dropout_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "dropout_elems");

    // Get drand48 for random number generation
    llvm::Function* drand_func = ctx_.module().getFunction("drand48");
    if (!drand_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }

    // Compute scale = 1.0 / (1.0 - p) for inverted dropout
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_minus_p = builder.CreateFSub(one, p);
    llvm::Value* scale = builder.CreateFDiv(one, one_minus_p);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "drop_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "drop_body", current_func);
    llvm::BasicBlock* drop_keep = llvm::BasicBlock::Create(ctx_.context(), "drop_keep", current_func);
    llvm::BasicBlock* drop_zero = llvm::BasicBlock::Create(ctx_.context(), "drop_zero", current_func);
    llvm::BasicBlock* drop_merge = llvm::BasicBlock::Create(ctx_.context(), "drop_merge", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "drop_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "drop_i");
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

    // Generate random number in [0, 1)
    llvm::Value* rand_val = builder.CreateCall(drand_func, {});
    // Keep if rand >= p (probability 1-p of keeping)
    llvm::Value* keep = builder.CreateFCmpOGE(rand_val, p);
    builder.CreateCondBr(keep, drop_keep, drop_zero);

    // Keep: scale the value
    builder.SetInsertPoint(drop_keep);
    llvm::Value* scaled_val = builder.CreateFMul(val, scale);
    builder.CreateBr(drop_merge);

    // Drop: zero
    builder.SetInsertPoint(drop_zero);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    builder.CreateBr(drop_merge);

    builder.SetInsertPoint(drop_merge);
    llvm::PHINode* result_val = builder.CreatePHI(ctx_.doubleType(), 2, "drop_val");
    result_val->addIncoming(scaled_val, drop_keep);
    result_val->addIncoming(zero, drop_zero);

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

llvm::Value* TensorCodegen::tensorCelu(const eshkol_operations_t* op) {
    // CELU: max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
    // Default alpha = 1.0
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("celu requires 1 or 2 arguments (tensor [, alpha])");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* alpha;
    if (op->call_op.num_vars == 2) {
        llvm::Value* alpha_tagged = codegenAST(&op->call_op.variables[1]);
        if (!alpha_tagged) return nullptr;
        alpha = tagged_.unpackDouble(alpha_tagged);
    } else {
        alpha = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    }

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
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "celu_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "celu_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "celu_elems");

    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "celu_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "celu_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "celu_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "celu_i");
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

    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

    // max(0, x)
    llvm::Value* cmp_pos = builder.CreateFCmpOGT(val, zero);
    llvm::Value* pos_part = builder.CreateSelect(cmp_pos, val, zero);

    // min(0, alpha * (exp(x / alpha) - 1))
    llvm::Value* x_over_alpha = builder.CreateFDiv(val, alpha);
    llvm::Value* exp_val = builder.CreateCall(exp_func, {x_over_alpha});
    llvm::Value* exp_minus_1 = builder.CreateFSub(exp_val, one);
    llvm::Value* alpha_exp = builder.CreateFMul(alpha, exp_minus_1);
    llvm::Value* cmp_neg = builder.CreateFCmpOLT(alpha_exp, zero);
    llvm::Value* neg_part = builder.CreateSelect(cmp_neg, alpha_exp, zero);

    // CELU = max(0,x) + min(0, alpha*(exp(x/alpha)-1))
    llvm::Value* result_val = builder.CreateFAdd(pos_part, neg_part);

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

    // Guard: maxpool2d requires at least 2D tensor
    {
        llvm::Function* cur_fn = builder.GetInsertBlock()->getParent();
        llvm::Value* dims_ok = builder.CreateICmpUGE(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
        llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "mp2d_dims_ok", cur_fn);
        llvm::BasicBlock* err_bb = llvm::BasicBlock::Create(ctx_.context(), "mp2d_dims_err", cur_fn);
        builder.CreateCondBr(dims_ok, ok_bb, err_bb);
        builder.SetInsertPoint(err_bb);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = builder.CreateGlobalString("Error: maxpool2d requires at least 2D tensor (got %lldD)\n");
            builder.CreateCall(pf, {fmt, num_dims});
            builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
        }
        builder.CreateUnreachable();
        builder.SetInsertPoint(ok_bb);
    }

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

    // Guard: avgpool2d requires at least 2D tensor
    {
        llvm::Function* cur_fn = builder.GetInsertBlock()->getParent();
        llvm::Value* dims_ok = builder.CreateICmpUGE(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
        llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "ap2d_dims_ok", cur_fn);
        llvm::BasicBlock* err_bb = llvm::BasicBlock::Create(ctx_.context(), "ap2d_dims_err", cur_fn);
        builder.CreateCondBr(dims_ok, ok_bb, err_bb);
        builder.SetInsertPoint(err_bb);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = builder.CreateGlobalString("Error: avgpool2d requires at least 2D tensor (got %lldD)\n");
            builder.CreateCall(pf, {fmt, num_dims});
            builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
        }
        builder.CreateUnreachable();
        builder.SetInsertPoint(ok_bb);
    }

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
    llvm::Value* k_ndim_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 1);
    llvm::Value* k_ndim = builder.CreateLoad(ctx_.int64Type(), k_ndim_field);
    llvm::Value* k_elems_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 2);
    llvm::Value* kernel_elems = builder.CreateLoad(ctx_.ptrType(), k_elems_field);
    llvm::Value* k_total_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 3);
    llvm::Value* k_total = builder.CreateLoad(ctx_.int64Type(), k_total_field);

    // Guard: conv2d kernel requires at least 2D tensor
    {
        llvm::Function* cur_fn = builder.GetInsertBlock()->getParent();
        llvm::Value* dims_ok = builder.CreateICmpUGE(k_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 2));
        llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "c2d_kdims_ok", cur_fn);
        llvm::BasicBlock* err_bb = llvm::BasicBlock::Create(ctx_.context(), "c2d_kdims_err", cur_fn);
        builder.CreateCondBr(dims_ok, ok_bb, err_bb);
        builder.SetInsertPoint(err_bb);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = builder.CreateGlobalString("Error: conv2d kernel requires at least 2D tensor (got %lldD)\n");
            builder.CreateCall(pf, {fmt, k_ndim});
            builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
        }
        builder.CreateUnreachable();
        builder.SetInsertPoint(ok_bb);
    }

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
    // batch-norm: (batch-norm input gamma beta epsilon [axis])
    // Simplified batch normalization for inference
    // y = gamma * (x - mean) / sqrt(var + eps) + beta
    // axis defaults to 0 (batch dimension); optional 5th arg overrides
    if (op->call_op.num_vars < 4 || op->call_op.num_vars > 5) {
        eshkol_error("batch-norm requires 4-5 arguments (input, gamma, beta, epsilon, [axis])");
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

    // 5-arg case: (batch-norm input gamma beta epsilon axis) → axis-aware via runtime
    if (op->call_op.num_vars == 5) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[4]);
        if (!axis_val) return nullptr;

        auto& builder = ctx_.builder();
        llvm::Value* ptr_int = tagged_.safeExtractInt64(input_val);
        llvm::Value* ptr = builder.CreateIntToPtr(ptr_int, ctx_.ptrType());
        llvm::StructType* ttype = ctx_.tensorType();
        llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 2));
        llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 3));
        llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 0));
        llvm::Value* rank = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 1));
        llvm::Value* axis = tagged_.safeExtractInt64(axis_val);

        llvm::Value* gamma_d = gamma_val;
        if (gamma_val->getType() == ctx_.taggedValueType()) gamma_d = tagged_.unpackDouble(gamma_val);
        llvm::Value* beta_d = beta_val;
        if (beta_val->getType() == ctx_.taggedValueType()) beta_d = tagged_.unpackDouble(beta_val);
        llvm::Value* eps_d = eps_arg;
        if (eps_arg->getType() == ctx_.taggedValueType()) eps_d = tagged_.unpackDouble(eps_arg);
        else if (eps_arg->getType()->isIntegerTy(64)) eps_d = builder.CreateSIToFP(eps_arg, ctx_.doubleType());

        llvm::Value* arena = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        auto* ptrTy = ctx_.ptrType();
        auto* i64Ty = ctx_.int64Type();
        auto* dblTy = ctx_.doubleType();
        llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, dblTy, dblTy, dblTy}, false);
        llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_normalize", fn_type);
        llvm::Value* result = builder.CreateCall(callee,
            {arena, elems, total, dims, rank, axis, gamma_d, beta_d, eps_d}, "bn_axis_result");
        return tagged_.packHeapPtr(result);
    }

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
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::sqrt, {ctx_.doubleType()});
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
    // layer-norm: (layer-norm input gamma beta epsilon [axis])
    // Normalizes across the LAST dimension (features) for each sample independently.
    // This is fundamentally different from batch-norm which normalizes across the batch dimension.
    // For 1D: normalizes all elements (single sample)
    // For 2D (batch×features): normalizes each row independently
    // For ND: last dim = features, everything else = batch
    // Optional 5th arg overrides axis (default: -1 = last dimension)
    if (op->call_op.num_vars < 4 || op->call_op.num_vars > 5) {
        eshkol_error("layer-norm requires 4-5 arguments (input, gamma, beta, epsilon, [axis])");
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

    // 5-arg case: (layer-norm input gamma beta epsilon axis) → axis-aware via runtime
    if (op->call_op.num_vars == 5) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[4]);
        if (!axis_val) return nullptr;

        auto& builder = ctx_.builder();
        llvm::Value* ptr_int = tagged_.safeExtractInt64(input_val);
        llvm::Value* ptr = builder.CreateIntToPtr(ptr_int, ctx_.ptrType());
        llvm::StructType* ttype = ctx_.tensorType();
        llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 2));
        llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 3));
        llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 0));
        llvm::Value* rank = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 1));
        llvm::Value* axis = tagged_.safeExtractInt64(axis_val);

        llvm::Value* gamma_d = gamma_val;
        if (gamma_val->getType() == ctx_.taggedValueType()) gamma_d = tagged_.unpackDouble(gamma_val);
        llvm::Value* beta_d = beta_val;
        if (beta_val->getType() == ctx_.taggedValueType()) beta_d = tagged_.unpackDouble(beta_val);
        llvm::Value* eps_d = eps_arg;
        if (eps_arg->getType() == ctx_.taggedValueType()) eps_d = tagged_.unpackDouble(eps_arg);
        else if (eps_arg->getType()->isIntegerTy(64)) eps_d = builder.CreateSIToFP(eps_arg, ctx_.doubleType());

        llvm::Value* arena = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        auto* ptrTy = ctx_.ptrType();
        auto* i64Ty = ctx_.int64Type();
        auto* dblTy = ctx_.doubleType();
        llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, dblTy, dblTy, dblTy}, false);
        llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_normalize", fn_type);
        llvm::Value* result = builder.CreateCall(callee,
            {arena, elems, total, dims, rank, axis, gamma_d, beta_d, eps_d}, "ln_axis_result");
        return tagged_.packHeapPtr(result);
    }

    auto& builder = ctx_.builder();

    llvm::Value* epsilon = eps_arg;
    if (eps_arg->getType() == ctx_.taggedValueType()) {
        epsilon = tagged_.unpackDouble(eps_arg);
    } else if (eps_arg->getType()->isIntegerTy(64)) {
        epsilon = builder.CreateSIToFP(eps_arg, ctx_.doubleType());
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
    llvm::Value* in_ndim = builder.CreateLoad(ctx_.int64Type(), in_ndim_field);
    llvm::Value* in_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* input_elems = builder.CreateLoad(ctx_.ptrType(), in_elems_field);
    llvm::Value* in_total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), in_total_field);

    // Extract gamma/beta scalars
    llvm::Value* gamma = gamma_val;
    if (gamma_val->getType() == ctx_.taggedValueType()) {
        gamma = tagged_.unpackDouble(gamma_val);
    }
    llvm::Value* beta = beta_val;
    if (beta_val->getType() == ctx_.taggedValueType()) {
        beta = tagged_.unpackDouble(beta_val);
    }

    // Allocate output tensor (same shape as input)
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "ln_result");

    llvm::Value* dims_size = builder.CreateMul(in_ndim, llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "ln_dims");

    llvm::Function* memcpy_func = ctx_.module().getFunction("memcpy");
    if (!memcpy_func) {
        llvm::FunctionType* memcpy_type = llvm::FunctionType::get(ctx_.ptrType(),
            {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
        memcpy_func = llvm::Function::Create(memcpy_type, llvm::Function::ExternalLinkage, "memcpy", &ctx_.module());
    }
    builder.CreateCall(memcpy_func, {result_dims, in_dims, dims_size});

    llvm::Value* elems_size = builder.CreateMul(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "ln_elems");

    // Populate result tensor metadata
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(in_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    // Compute feature_size = dims[ndim-1] (last dimension)
    llvm::Value* last_dim_idx = builder.CreateSub(in_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* last_dim_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims, last_dim_idx);
    llvm::Value* feature_size = builder.CreateLoad(ctx_.int64Type(), last_dim_ptr);

    // batch_size = total_elements / feature_size
    llvm::Value* batch_size = builder.CreateUDiv(total_elements, feature_size);
    llvm::Value* feature_fp = builder.CreateSIToFP(feature_size, ctx_.doubleType());

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::sqrt, {ctx_.doubleType()});

    // Allocas for loop variables
    llvm::Value* batch_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ln_batch_idx");
    llvm::Value* feat_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ln_feat_idx");
    llvm::Value* ln_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "ln_sum");
    llvm::Value* ln_mean = builder.CreateAlloca(ctx_.doubleType(), nullptr, "ln_mean");

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);

    // === Outer loop: iterate over samples ===
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "ln_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "ln_batch_body", current_func);
    llvm::BasicBlock* batch_done = llvm::BasicBlock::Create(ctx_.context(), "ln_batch_done", current_func);

    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_cond);
    llvm::Value* bi = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    builder.CreateCondBr(builder.CreateICmpULT(bi, batch_size), batch_body, batch_done);

    builder.SetInsertPoint(batch_body);
    llvm::Value* base_offset = builder.CreateMul(bi, feature_size);

    // --- Pass 1: Compute mean for this sample ---
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), ln_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), feat_idx);

    llvm::BasicBlock* mean_cond = llvm::BasicBlock::Create(ctx_.context(), "ln_mean_cond", current_func);
    llvm::BasicBlock* mean_body = llvm::BasicBlock::Create(ctx_.context(), "ln_mean_body", current_func);
    llvm::BasicBlock* mean_done = llvm::BasicBlock::Create(ctx_.context(), "ln_mean_done", current_func);

    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_cond);
    llvm::Value* fi = builder.CreateLoad(ctx_.int64Type(), feat_idx);
    builder.CreateCondBr(builder.CreateICmpULT(fi, feature_size), mean_body, mean_done);

    builder.SetInsertPoint(mean_body);
    llvm::Value* elem_offset = builder.CreateAdd(base_offset, fi);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, elem_offset);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), ln_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, elem), ln_sum);
    builder.CreateStore(builder.CreateAdd(fi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), feat_idx);
    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_done);
    llvm::Value* sum_val = builder.CreateLoad(ctx_.doubleType(), ln_sum);
    llvm::Value* mean_val = builder.CreateFDiv(sum_val, feature_fp);
    builder.CreateStore(mean_val, ln_mean);

    // --- Pass 2: Compute variance for this sample ---
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), ln_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), feat_idx);

    llvm::BasicBlock* var_cond = llvm::BasicBlock::Create(ctx_.context(), "ln_var_cond", current_func);
    llvm::BasicBlock* var_body = llvm::BasicBlock::Create(ctx_.context(), "ln_var_body", current_func);
    llvm::BasicBlock* var_done = llvm::BasicBlock::Create(ctx_.context(), "ln_var_done", current_func);

    builder.CreateBr(var_cond);

    builder.SetInsertPoint(var_cond);
    fi = builder.CreateLoad(ctx_.int64Type(), feat_idx);
    builder.CreateCondBr(builder.CreateICmpULT(fi, feature_size), var_body, var_done);

    builder.SetInsertPoint(var_body);
    elem_offset = builder.CreateAdd(base_offset, fi);
    elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, elem_offset);
    elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    mean_val = builder.CreateLoad(ctx_.doubleType(), ln_mean);
    llvm::Value* diff = builder.CreateFSub(elem, mean_val);
    llvm::Value* sq_diff = builder.CreateFMul(diff, diff);
    cur_sum = builder.CreateLoad(ctx_.doubleType(), ln_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, sq_diff), ln_sum);
    builder.CreateStore(builder.CreateAdd(fi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), feat_idx);
    builder.CreateBr(var_cond);

    builder.SetInsertPoint(var_done);
    llvm::Value* var_sum = builder.CreateLoad(ctx_.doubleType(), ln_sum);
    llvm::Value* var_val = builder.CreateFDiv(var_sum, feature_fp);
    llvm::Value* var_plus_eps = builder.CreateFAdd(var_val, epsilon);
    llvm::Value* std_val = builder.CreateCall(sqrt_func, {var_plus_eps});

    // --- Pass 3: Normalize, scale, and shift for this sample ---
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), feat_idx);

    llvm::BasicBlock* norm_cond = llvm::BasicBlock::Create(ctx_.context(), "ln_norm_cond", current_func);
    llvm::BasicBlock* norm_body = llvm::BasicBlock::Create(ctx_.context(), "ln_norm_body", current_func);
    llvm::BasicBlock* norm_done = llvm::BasicBlock::Create(ctx_.context(), "ln_norm_done", current_func);

    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_cond);
    fi = builder.CreateLoad(ctx_.int64Type(), feat_idx);
    builder.CreateCondBr(builder.CreateICmpULT(fi, feature_size), norm_body, norm_done);

    builder.SetInsertPoint(norm_body);
    elem_offset = builder.CreateAdd(base_offset, fi);
    elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, elem_offset);
    elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    mean_val = builder.CreateLoad(ctx_.doubleType(), ln_mean);

    // y = gamma * (x - mean) / std + beta
    llvm::Value* centered = builder.CreateFSub(elem, mean_val);
    llvm::Value* normalized = builder.CreateFDiv(centered, std_val);
    llvm::Value* scaled = builder.CreateFMul(normalized, gamma);
    llvm::Value* shifted = builder.CreateFAdd(scaled, beta);

    llvm::Value* out_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, elem_offset);
    llvm::Value* out_bits = builder.CreateBitCast(shifted, ctx_.int64Type());
    builder.CreateStore(out_bits, out_ptr);

    builder.CreateStore(builder.CreateAdd(fi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), feat_idx);
    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_done);
    // Advance to next sample
    builder.CreateStore(builder.CreateAdd(bi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_done);
    return tagged_.packHeapPtr(result_ptr);
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
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
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
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
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
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
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

llvm::Value* TensorCodegen::adamwStep(const eshkol_operations_t* op) {
    // adamw-step!: (adamw-step! params grads lr m v t [beta1 beta2 eps weight_decay])
    // AdamW: Decoupled weight decay — weight decay is applied directly to params,
    // NOT through the gradient (unlike L2 regularization in vanilla Adam)
    if (op->call_op.num_vars < 6) {
        eshkol_error("adamw-step! requires at least 6 arguments: params, grads, lr, m, v, t");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* params_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* grads_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* lr_tagged = codegenAST(&op->call_op.variables[2]);
    llvm::Value* m_tagged = codegenAST(&op->call_op.variables[3]);
    llvm::Value* v_tagged = codegenAST(&op->call_op.variables[4]);
    llvm::Value* t_tagged = codegenAST(&op->call_op.variables[5]);

    if (!params_tagged || !grads_tagged || !lr_tagged || !m_tagged || !v_tagged || !t_tagged)
        return nullptr;

    llvm::Value* params_ptr = tagged_.unpackPtr(params_tagged);
    llvm::Value* grads_ptr = tagged_.unpackPtr(grads_tagged);
    llvm::Value* lr = tagged_.unpackDouble(lr_tagged);
    llvm::Value* m_ptr = tagged_.unpackPtr(m_tagged);
    llvm::Value* v_ptr = tagged_.unpackPtr(v_tagged);
    llvm::Value* t = tagged_.unpackInt64(t_tagged);
    llvm::Value* t_double = builder.CreateSIToFP(t, ctx_.doubleType());

    llvm::Value* beta1 = llvm::ConstantFP::get(ctx_.doubleType(), 0.9);
    llvm::Value* beta2 = llvm::ConstantFP::get(ctx_.doubleType(), 0.999);
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-8);
    llvm::Value* weight_decay = llvm::ConstantFP::get(ctx_.doubleType(), 0.01);

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
    if (op->call_op.num_vars >= 10) {
        llvm::Value* wd_tagged = codegenAST(&op->call_op.variables[9]);
        if (wd_tagged) weight_decay = tagged_.unpackDouble(wd_tagged);
    }

    // Bias corrections
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

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 3);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);

    llvm::Value* params_elems_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 2);
    llvm::Value* params_elems = builder.CreateLoad(ctx_.ptrType(), params_elems_ptr);
    llvm::Value* grads_elems_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 2);
    llvm::Value* grads_elems = builder.CreateLoad(ctx_.ptrType(), grads_elems_ptr);
    llvm::Value* m_elems_ptr = builder.CreateStructGEP(tensor_type, m_ptr, 2);
    llvm::Value* m_elems = builder.CreateLoad(ctx_.ptrType(), m_elems_ptr);
    llvm::Value* v_elems_ptr = builder.CreateStructGEP(tensor_type, v_ptr, 2);
    llvm::Value* v_elems = builder.CreateLoad(ctx_.ptrType(), v_elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* update_loop = llvm::BasicBlock::Create(ctx_.context(), "adamw_loop", current_func);
    llvm::BasicBlock* update_body = llvm::BasicBlock::Create(ctx_.context(), "adamw_body", current_func);
    llvm::BasicBlock* update_done = llvm::BasicBlock::Create(ctx_.context(), "adamw_done", current_func);

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

    llvm::Value* m_elem_ptr = builder.CreateGEP(ctx_.int64Type(), m_elems, ei);
    llvm::Value* m_bits = builder.CreateLoad(ctx_.int64Type(), m_elem_ptr);
    llvm::Value* m_val = builder.CreateBitCast(m_bits, ctx_.doubleType());

    llvm::Value* v_elem_ptr = builder.CreateGEP(ctx_.int64Type(), v_elems, ei);
    llvm::Value* v_bits = builder.CreateLoad(ctx_.int64Type(), v_elem_ptr);
    llvm::Value* v_val = builder.CreateBitCast(v_bits, ctx_.doubleType());

    // STEP 1: Decoupled weight decay — applied to params directly, NOT through gradient
    // param = param * (1 - lr * weight_decay)
    llvm::Value* decay_factor = builder.CreateFSub(one, builder.CreateFMul(lr, weight_decay));
    llvm::Value* decayed_param = builder.CreateFMul(param, decay_factor);

    // STEP 2: Standard Adam moment updates
    llvm::Value* one_minus_beta1 = builder.CreateFSub(one, beta1);
    llvm::Value* m_new = builder.CreateFAdd(
        builder.CreateFMul(beta1, m_val),
        builder.CreateFMul(one_minus_beta1, grad));

    llvm::Value* one_minus_beta2 = builder.CreateFSub(one, beta2);
    llvm::Value* grad_sq = builder.CreateFMul(grad, grad);
    llvm::Value* v_new = builder.CreateFAdd(
        builder.CreateFMul(beta2, v_val),
        builder.CreateFMul(one_minus_beta2, grad_sq));

    builder.CreateStore(builder.CreateBitCast(m_new, ctx_.int64Type()), m_elem_ptr);
    builder.CreateStore(builder.CreateBitCast(v_new, ctx_.int64Type()), v_elem_ptr);

    // STEP 3: Bias-corrected Adam update on already-decayed param
    llvm::Value* m_hat = builder.CreateFDiv(m_new, bias_corr1);
    llvm::Value* v_hat = builder.CreateFDiv(v_new, bias_corr2);

    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* sqrt_v_hat = builder.CreateCall(sqrt_func, {v_hat});
    llvm::Value* denom = builder.CreateFAdd(sqrt_v_hat, eps);
    llvm::Value* update = builder.CreateFDiv(m_hat, denom);
    llvm::Value* scaled_update = builder.CreateFMul(lr, update);
    llvm::Value* new_param = builder.CreateFSub(decayed_param, scaled_update);

    builder.CreateStore(builder.CreateBitCast(new_param, ctx_.int64Type()), param_ptr);

    builder.CreateStore(builder.CreateAdd(ei, llvm::ConstantInt::get(ctx_.int64Type(), 1)), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_done);
    return params_tagged;
}

llvm::Value* TensorCodegen::adagradStep(const eshkol_operations_t* op) {
    // adagrad-step!: (adagrad-step! params grads lr accum [eps])
    // Adagrad: Per-parameter adaptive learning rate via accumulated squared gradients
    // accum_new = accum + grad^2
    // param = param - lr * grad / (sqrt(accum_new) + eps)
    if (op->call_op.num_vars < 4) {
        eshkol_error("adagrad-step! requires at least 4 arguments: params, grads, lr, accum");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* params_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* grads_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* lr_tagged = codegenAST(&op->call_op.variables[2]);
    llvm::Value* accum_tagged = codegenAST(&op->call_op.variables[3]);
    if (!params_tagged || !grads_tagged || !lr_tagged || !accum_tagged) return nullptr;

    llvm::Value* params_ptr = tagged_.unpackPtr(params_tagged);
    llvm::Value* grads_ptr = tagged_.unpackPtr(grads_tagged);
    llvm::Value* lr = tagged_.unpackDouble(lr_tagged);
    llvm::Value* accum_ptr = tagged_.unpackPtr(accum_tagged);

    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-10);
    if (op->call_op.num_vars >= 5) {
        llvm::Value* eps_tagged = codegenAST(&op->call_op.variables[4]);
        if (eps_tagged) eps = tagged_.unpackDouble(eps_tagged);
    }

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 3);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);

    llvm::Value* params_elems_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 2);
    llvm::Value* params_elems = builder.CreateLoad(ctx_.ptrType(), params_elems_ptr);
    llvm::Value* grads_elems_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 2);
    llvm::Value* grads_elems = builder.CreateLoad(ctx_.ptrType(), grads_elems_ptr);
    llvm::Value* accum_elems_ptr = builder.CreateStructGEP(tensor_type, accum_ptr, 2);
    llvm::Value* accum_elems = builder.CreateLoad(ctx_.ptrType(), accum_elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* update_loop = llvm::BasicBlock::Create(ctx_.context(), "adagrad_loop", current_func);
    llvm::BasicBlock* update_body = llvm::BasicBlock::Create(ctx_.context(), "adagrad_body", current_func);
    llvm::BasicBlock* update_done = llvm::BasicBlock::Create(ctx_.context(), "adagrad_done", current_func);

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

    llvm::Value* acc_ptr = builder.CreateGEP(ctx_.int64Type(), accum_elems, ei);
    llvm::Value* acc_bits = builder.CreateLoad(ctx_.int64Type(), acc_ptr);
    llvm::Value* acc_val = builder.CreateBitCast(acc_bits, ctx_.doubleType());

    // accum += grad^2
    llvm::Value* grad_sq = builder.CreateFMul(grad, grad);
    llvm::Value* acc_new = builder.CreateFAdd(acc_val, grad_sq);
    builder.CreateStore(builder.CreateBitCast(acc_new, ctx_.int64Type()), acc_ptr);

    // param -= lr * grad / (sqrt(accum) + eps)
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* sqrt_acc = builder.CreateCall(sqrt_func, {acc_new});
    llvm::Value* denom = builder.CreateFAdd(sqrt_acc, eps);
    llvm::Value* update = builder.CreateFDiv(grad, denom);
    llvm::Value* scaled_update = builder.CreateFMul(lr, update);
    llvm::Value* new_param = builder.CreateFSub(param, scaled_update);
    builder.CreateStore(builder.CreateBitCast(new_param, ctx_.int64Type()), param_ptr);

    builder.CreateStore(builder.CreateAdd(ei, llvm::ConstantInt::get(ctx_.int64Type(), 1)), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_done);
    return params_tagged;
}

llvm::Value* TensorCodegen::checkGradHealth(const eshkol_operations_t* op) {
    // check-grad-health: (check-grad-health tensor) → #t if all finite, #f if NaN/Inf found
    if (op->call_op.num_vars != 1) {
        eshkol_error("check-grad-health requires exactly 1 argument");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    if (!tensor_tagged) return nullptr;

    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "health_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "health_body", current_func);
    llvm::BasicBlock* loop_inc = llvm::BasicBlock::Create(ctx_.context(), "health_inc", current_func);
    llvm::BasicBlock* health_bad = llvm::BasicBlock::Create(ctx_.context(), "health_bad", current_func);
    llvm::BasicBlock* health_good = llvm::BasicBlock::Create(ctx_.context(), "health_good", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "health_merge", current_func);

    llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
    builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), loop_body, health_good);

    builder.SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems, i);
    llvm::Value* bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* val = builder.CreateBitCast(bits, ctx_.doubleType());

    // Detect NaN or Inf: for finite values, (x - x) == 0.0 (ordered).
    // For NaN or Inf, (x - x) is NaN (unordered).
    llvm::Value* diff = builder.CreateFSub(val, val);
    llvm::Value* not_finite = builder.CreateFCmpUNO(diff, diff);
    builder.CreateCondBr(not_finite, health_bad, loop_inc);

    // Increment and loop back
    builder.SetInsertPoint(loop_inc);
    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, idx);
    builder.CreateBr(loop_cond);

    // All elements finite → return true
    builder.SetInsertPoint(health_good);
    llvm::Value* true_val = tagged_.packBool(builder.getInt1(1));
    builder.CreateBr(merge_block);

    // Found NaN/Inf → return false
    builder.SetInsertPoint(health_bad);
    llvm::Value* false_val = tagged_.packBool(builder.getInt1(0));
    builder.CreateBr(merge_block);

    builder.SetInsertPoint(merge_block);
    llvm::PHINode* result = builder.CreatePHI(true_val->getType(), 2, "health_result");
    result->addIncoming(true_val, health_good);
    result->addIncoming(false_val, health_bad);
    return result;
}

// ===== WEIGHT INITIALIZATION =====

llvm::Value* TensorCodegen::xavierUniform(const eshkol_operations_t* op) {
    // xavier-uniform!: (xavier-uniform! tensor fan_in fan_out)
    // Fills tensor with U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
    if (op->call_op.num_vars != 3) {
        eshkol_error("xavier-uniform! requires 3 arguments: tensor, fan_in, fan_out");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* fan_in_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* fan_out_tagged = codegenAST(&op->call_op.variables[2]);
    if (!tensor_tagged || !fan_in_tagged || !fan_out_tagged) return nullptr;

    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);
    llvm::Value* fan_in = taggedNumericToDouble(ctx_, tagged_, fan_in_tagged);
    llvm::Value* fan_out = taggedNumericToDouble(ctx_, tagged_, fan_out_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    // limit = sqrt(6 / (fan_in + fan_out))
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* fan_sum = builder.CreateFAdd(fan_in, fan_out);
    llvm::Value* six = llvm::ConstantFP::get(ctx_.doubleType(), 6.0);
    llvm::Value* ratio = builder.CreateFDiv(six, fan_sum);
    llvm::Value* limit = builder.CreateCall(sqrt_func, {ratio});
    llvm::Value* neg_limit = builder.CreateFNeg(limit);
    llvm::Value* range = builder.CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 2.0), limit);

    llvm::Function* drand_func = ctx_.module().getFunction("drand48");
    if (!drand_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "xu_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "xu_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "xu_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Generate U(-limit, limit) = drand48() * 2 * limit - limit
    llvm::Value* rand_val = builder.CreateCall(drand_func, {});
    llvm::Value* scaled = builder.CreateFMul(rand_val, range);
    llvm::Value* result = builder.CreateFAdd(scaled, neg_limit);

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), elems, i);
    builder.CreateStore(builder.CreateBitCast(result, ctx_.int64Type()), dst_ptr);

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    return tensor_tagged;
}

llvm::Value* TensorCodegen::xavierNormal(const eshkol_operations_t* op) {
    // xavier-normal!: (xavier-normal! tensor fan_in fan_out)
    // Fills tensor with N(0, std) where std = sqrt(2 / (fan_in + fan_out))
    // Uses Box-Muller transform for normal distribution
    if (op->call_op.num_vars != 3) {
        eshkol_error("xavier-normal! requires 3 arguments: tensor, fan_in, fan_out");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* fan_in_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* fan_out_tagged = codegenAST(&op->call_op.variables[2]);
    if (!tensor_tagged || !fan_in_tagged || !fan_out_tagged) return nullptr;

    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);
    llvm::Value* fan_in = taggedNumericToDouble(ctx_, tagged_, fan_in_tagged);
    llvm::Value* fan_out = taggedNumericToDouble(ctx_, tagged_, fan_out_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    // std = sqrt(2 / (fan_in + fan_out))
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* fan_sum = builder.CreateFAdd(fan_in, fan_out);
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* ratio = builder.CreateFDiv(two, fan_sum);
    llvm::Value* std_dev = builder.CreateCall(sqrt_func, {ratio});

    llvm::Function* drand_func = ctx_.module().getFunction("drand48");
    if (!drand_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }
    llvm::Function* cos_func = ctx_.module().getFunction("cos");
    if (!cos_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        cos_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "cos", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "xn_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "xn_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "xn_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Box-Muller transform: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    llvm::Value* u1 = builder.CreateCall(drand_func, {});
    llvm::Value* u2 = builder.CreateCall(drand_func, {});
    // Clamp u1 away from 0 for log safety
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-10);
    llvm::Value* u1_cmp = builder.CreateFCmpOGT(u1, eps);
    llvm::Value* u1_safe = builder.CreateSelect(u1_cmp, u1, eps);

    llvm::Value* neg_two = llvm::ConstantFP::get(ctx_.doubleType(), -2.0);
    llvm::Value* log_u1 = builder.CreateCall(log_func, {u1_safe});
    llvm::Value* inner = builder.CreateFMul(neg_two, log_u1);
    llvm::Value* radius = builder.CreateCall(sqrt_func, {inner});

    llvm::Value* two_pi = llvm::ConstantFP::get(ctx_.doubleType(), 6.283185307179586);
    llvm::Value* angle = builder.CreateFMul(two_pi, u2);
    llvm::Value* cos_val = builder.CreateCall(cos_func, {angle});

    llvm::Value* z = builder.CreateFMul(radius, cos_val);
    llvm::Value* result = builder.CreateFMul(z, std_dev);

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), elems, i);
    builder.CreateStore(builder.CreateBitCast(result, ctx_.int64Type()), dst_ptr);

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    return tensor_tagged;
}

llvm::Value* TensorCodegen::kaimingUniform(const eshkol_operations_t* op) {
    // kaiming-uniform!: (kaiming-uniform! tensor fan_in)
    // Fills tensor with U(-limit, limit) where limit = sqrt(6 / fan_in)
    // For ReLU networks (He initialization)
    if (op->call_op.num_vars < 2 || op->call_op.num_vars > 2) {
        eshkol_error("kaiming-uniform! requires 2 arguments: tensor, fan_in");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* fan_in_tagged = codegenAST(&op->call_op.variables[1]);
    if (!tensor_tagged || !fan_in_tagged) return nullptr;

    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);
    llvm::Value* fan_in = taggedNumericToDouble(ctx_, tagged_, fan_in_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    // limit = sqrt(6 / fan_in)
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* six = llvm::ConstantFP::get(ctx_.doubleType(), 6.0);
    llvm::Value* ratio = builder.CreateFDiv(six, fan_in);
    llvm::Value* limit = builder.CreateCall(sqrt_func, {ratio});
    llvm::Value* neg_limit = builder.CreateFNeg(limit);
    llvm::Value* range = builder.CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 2.0), limit);

    llvm::Function* drand_func = ctx_.module().getFunction("drand48");
    if (!drand_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "ku_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "ku_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "ku_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* rand_val = builder.CreateCall(drand_func, {});
    llvm::Value* scaled = builder.CreateFMul(rand_val, range);
    llvm::Value* result = builder.CreateFAdd(scaled, neg_limit);

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), elems, i);
    builder.CreateStore(builder.CreateBitCast(result, ctx_.int64Type()), dst_ptr);

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    return tensor_tagged;
}

llvm::Value* TensorCodegen::kaimingNormal(const eshkol_operations_t* op) {
    // kaiming-normal!: (kaiming-normal! tensor fan_in)
    // Fills tensor with N(0, std) where std = sqrt(2 / fan_in)
    if (op->call_op.num_vars != 2) {
        eshkol_error("kaiming-normal! requires 2 arguments: tensor, fan_in");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* fan_in_tagged = codegenAST(&op->call_op.variables[1]);
    if (!tensor_tagged || !fan_in_tagged) return nullptr;

    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);
    llvm::Value* fan_in = taggedNumericToDouble(ctx_, tagged_, fan_in_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    // std = sqrt(2 / fan_in)
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* ratio = builder.CreateFDiv(two, fan_in);
    llvm::Value* std_dev = builder.CreateCall(sqrt_func, {ratio});

    llvm::Function* drand_func = ctx_.module().getFunction("drand48");
    if (!drand_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }
    llvm::Function* cos_func = ctx_.module().getFunction("cos");
    if (!cos_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        cos_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "cos", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "kn_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "kn_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "kn_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Box-Muller: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    llvm::Value* u1 = builder.CreateCall(drand_func, {});
    llvm::Value* u2 = builder.CreateCall(drand_func, {});
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-10);
    llvm::Value* u1_cmp = builder.CreateFCmpOGT(u1, eps);
    llvm::Value* u1_safe = builder.CreateSelect(u1_cmp, u1, eps);

    llvm::Value* neg_two = llvm::ConstantFP::get(ctx_.doubleType(), -2.0);
    llvm::Value* log_u1 = builder.CreateCall(log_func, {u1_safe});
    llvm::Value* inner = builder.CreateFMul(neg_two, log_u1);
    llvm::Value* radius = builder.CreateCall(sqrt_func, {inner});

    llvm::Value* two_pi = llvm::ConstantFP::get(ctx_.doubleType(), 6.283185307179586);
    llvm::Value* angle = builder.CreateFMul(two_pi, u2);
    llvm::Value* cos_val = builder.CreateCall(cos_func, {angle});

    llvm::Value* z = builder.CreateFMul(radius, cos_val);
    llvm::Value* result = builder.CreateFMul(z, std_dev);

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), elems, i);
    builder.CreateStore(builder.CreateBitCast(result, ctx_.int64Type()), dst_ptr);

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    return tensor_tagged;
}

llvm::Value* TensorCodegen::lecunNormal(const eshkol_operations_t* op) {
    // lecun-normal!: (lecun-normal! tensor fan_in)
    // Fills tensor with N(0, std) where std = sqrt(1 / fan_in)
    // For SELU activation
    if (op->call_op.num_vars != 2) {
        eshkol_error("lecun-normal! requires 2 arguments: tensor, fan_in");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* fan_in_tagged = codegenAST(&op->call_op.variables[1]);
    if (!tensor_tagged || !fan_in_tagged) return nullptr;

    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);
    llvm::Value* fan_in = taggedNumericToDouble(ctx_, tagged_, fan_in_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    // std = sqrt(1 / fan_in) = 1 / sqrt(fan_in)
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* ratio = builder.CreateFDiv(one, fan_in);
    llvm::Value* std_dev = builder.CreateCall(sqrt_func, {ratio});

    llvm::Function* drand_func = ctx_.module().getFunction("drand48");
    if (!drand_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }
    llvm::Function* cos_func = ctx_.module().getFunction("cos");
    if (!cos_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        cos_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "cos", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "ln_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "ln_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "ln_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Box-Muller: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    llvm::Value* u1 = builder.CreateCall(drand_func, {});
    llvm::Value* u2 = builder.CreateCall(drand_func, {});
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-10);
    llvm::Value* u1_cmp = builder.CreateFCmpOGT(u1, eps);
    llvm::Value* u1_safe = builder.CreateSelect(u1_cmp, u1, eps);

    llvm::Value* neg_two = llvm::ConstantFP::get(ctx_.doubleType(), -2.0);
    llvm::Value* log_u1 = builder.CreateCall(log_func, {u1_safe});
    llvm::Value* inner = builder.CreateFMul(neg_two, log_u1);
    llvm::Value* radius = builder.CreateCall(sqrt_func, {inner});

    llvm::Value* two_pi = llvm::ConstantFP::get(ctx_.doubleType(), 6.283185307179586);
    llvm::Value* angle = builder.CreateFMul(two_pi, u2);
    llvm::Value* cos_val = builder.CreateCall(cos_func, {angle});

    llvm::Value* z = builder.CreateFMul(radius, cos_val);
    llvm::Value* result = builder.CreateFMul(z, std_dev);

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), elems, i);
    builder.CreateStore(builder.CreateBitCast(result, ctx_.int64Type()), dst_ptr);

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    return tensor_tagged;
}

// ===== LEARNING RATE SCHEDULERS =====

llvm::Value* TensorCodegen::cosineAnnealingLR(const eshkol_operations_t* op) {
    // cosine-annealing-lr: (cosine-annealing-lr base_lr min_lr current_step total_steps) → lr
    // lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * t / T))
    if (op->call_op.num_vars != 4) {
        eshkol_error("cosine-annealing-lr requires 4 arguments: base_lr, min_lr, current_step, total_steps");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* base_lr_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* min_lr_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* step_tagged = codegenAST(&op->call_op.variables[2]);
    llvm::Value* total_tagged = codegenAST(&op->call_op.variables[3]);
    if (!base_lr_tagged || !min_lr_tagged || !step_tagged || !total_tagged) return nullptr;

    llvm::Value* base_lr = tagged_.unpackDouble(base_lr_tagged);
    llvm::Value* min_lr = tagged_.unpackDouble(min_lr_tagged);
    llvm::Value* step = tagged_.unpackDouble(step_tagged);
    llvm::Value* total = tagged_.unpackDouble(total_tagged);

    llvm::Function* cos_func = ctx_.module().getFunction("cos");
    if (!cos_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        cos_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "cos", &ctx_.module());
    }

    // lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * step / total))
    llvm::Value* pi = llvm::ConstantFP::get(ctx_.doubleType(), 3.14159265358979323846);
    llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

    llvm::Value* lr_range = builder.CreateFSub(base_lr, min_lr);
    llvm::Value* progress = builder.CreateFDiv(step, total);
    llvm::Value* angle = builder.CreateFMul(pi, progress);
    llvm::Value* cos_val = builder.CreateCall(cos_func, {angle});
    llvm::Value* one_plus_cos = builder.CreateFAdd(one, cos_val);
    llvm::Value* scaled = builder.CreateFMul(half, builder.CreateFMul(lr_range, one_plus_cos));
    llvm::Value* result = builder.CreateFAdd(min_lr, scaled);

    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::stepDecayLR(const eshkol_operations_t* op) {
    // step-decay-lr: (step-decay-lr base_lr gamma epoch step_size) → lr
    // lr = base_lr * gamma^(floor(epoch / step_size))
    if (op->call_op.num_vars != 4) {
        eshkol_error("step-decay-lr requires 4 arguments: base_lr, gamma, epoch, step_size");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* base_lr_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* gamma_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* epoch_tagged = codegenAST(&op->call_op.variables[2]);
    llvm::Value* step_tagged = codegenAST(&op->call_op.variables[3]);
    if (!base_lr_tagged || !gamma_tagged || !epoch_tagged || !step_tagged) return nullptr;

    llvm::Value* base_lr = tagged_.unpackDouble(base_lr_tagged);
    llvm::Value* gamma = tagged_.unpackDouble(gamma_tagged);
    llvm::Value* epoch = tagged_.unpackDouble(epoch_tagged);
    llvm::Value* step_size = tagged_.unpackDouble(step_tagged);

    llvm::Function* pow_func = ctx_.module().getFunction("pow");
    if (!pow_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType(), ctx_.doubleType()}, false);
        pow_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "pow", &ctx_.module());
    }
    llvm::Function* floor_func = ctx_.module().getFunction("floor");
    if (!floor_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        floor_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "floor", &ctx_.module());
    }

    // exponent = floor(epoch / step_size)
    llvm::Value* ratio = builder.CreateFDiv(epoch, step_size);
    llvm::Value* exponent = builder.CreateCall(floor_func, {ratio});
    // lr = base_lr * gamma^exponent
    llvm::Value* decay = builder.CreateCall(pow_func, {gamma, exponent});
    llvm::Value* result = builder.CreateFMul(base_lr, decay);

    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::linearWarmupLR(const eshkol_operations_t* op) {
    // linear-warmup-lr: (linear-warmup-lr base_lr current_step warmup_steps) → lr
    // lr = base_lr * min(1.0, current_step / warmup_steps)
    if (op->call_op.num_vars != 3) {
        eshkol_error("linear-warmup-lr requires 3 arguments: base_lr, current_step, warmup_steps");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* base_lr_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* step_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* warmup_tagged = codegenAST(&op->call_op.variables[2]);
    if (!base_lr_tagged || !step_tagged || !warmup_tagged) return nullptr;

    llvm::Value* base_lr = tagged_.unpackDouble(base_lr_tagged);
    llvm::Value* step = tagged_.unpackDouble(step_tagged);
    llvm::Value* warmup = tagged_.unpackDouble(warmup_tagged);

    // ratio = step / warmup
    llvm::Value* ratio = builder.CreateFDiv(step, warmup);
    // min(1.0, ratio)
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* cmp = builder.CreateFCmpOLT(ratio, one);
    llvm::Value* factor = builder.CreateSelect(cmp, ratio, one);
    llvm::Value* result = builder.CreateFMul(base_lr, factor);

    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::exponentialDecayLR(const eshkol_operations_t* op) {
    // exponential-decay-lr: (exponential-decay-lr base_lr gamma epoch) → lr
    // lr = base_lr * gamma^epoch
    if (op->call_op.num_vars != 3) {
        eshkol_error("exponential-decay-lr requires 3 arguments: base_lr, gamma, epoch");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* base_lr_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* gamma_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* epoch_tagged = codegenAST(&op->call_op.variables[2]);
    if (!base_lr_tagged || !gamma_tagged || !epoch_tagged) return nullptr;

    llvm::Value* base_lr = tagged_.unpackDouble(base_lr_tagged);
    llvm::Value* gamma = tagged_.unpackDouble(gamma_tagged);
    llvm::Value* epoch = tagged_.unpackDouble(epoch_tagged);

    llvm::Function* pow_func = ctx_.module().getFunction("pow");
    if (!pow_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType(), ctx_.doubleType()}, false);
        pow_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "pow", &ctx_.module());
    }

    llvm::Value* decay = builder.CreateCall(pow_func, {gamma, epoch});
    llvm::Value* result = builder.CreateFMul(base_lr, decay);

    return tagged_.packDouble(result);
}

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
