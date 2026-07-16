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

#include <cstdlib>
#include <cstring>

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

/**
 * @brief Coerce a value to a raw double for tensor element storage.
 *
 * Tagged values are branched on their type tag (INT64 is converted via
 * SIToFP, DOUBLE is unpacked directly); raw integer values are converted
 * via SIToFP; anything else (already a double) is returned unchanged.
 */
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

/**
 * @brief Construct the tensor codegen helper, optionally initializing the
 *        XLA backend (when ESHKOL_XLA_ENABLED) for accelerated tensor ops.
 */
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
/** @brief Whether a tensor op of @p num_elements elements should be routed
 *         through XLA rather than the scalar/SIMD codegen path. */
bool TensorCodegen::shouldUseXLA(size_t num_elements) const {
    // Use XLA if available and tensor size exceeds threshold
    return xla_ && xla_->shouldUseXLA(num_elements);
}
#endif

// Destructor must be defined where XLACodegen is complete
TensorCodegen::~TensorCodegen() = default;

/** @brief Report the SIMD vector width (in doubles) for the active target.
 *
 * Normal compiler/JIT invocations retain host specialization. Release stdlib
 * generation sets ESHKOL_TARGET_CPU=generic; in that mode the generated IR
 * must use the shared x86-64/AArch64 128-bit baseline rather than whichever
 * AVX width happens to be available on the hosted builder.
 */
unsigned TensorCodegen::getSIMDWidth() const {
    const unsigned host_width =
        CPUCapabilities::instance().getVectorWidth();
    const char* target_cpu = std::getenv("ESHKOL_TARGET_CPU");
    if (target_cpu && std::strcmp(target_cpu, "generic") == 0) {
        // x86-64 and AArch64 both guarantee 128-bit SIMD (SSE2/NEON), which
        // makes two doubles the deterministic baseline for every release lane.
        return 2;
    }
    return host_width;
}

/** @brief Return the LLVM vector type matching getSIMDWidth() (double2/4/8),
 *         or nullptr when the width is 1 (scalar-only, no vectorization). */
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

/**
 * @brief Attach `llvm.loop` metadata (vectorize.enable/width and/or
 *        unroll.count hints) to a loop's back-edge branch instruction, so
 *        LLVM's optimizer vectorizes/unrolls tensor loops as requested.
 */
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

// ─────────────────────────────────────────────────────────────────────────────
// ESH-0069: Centralized, type-checked tensor-operand unpack.
//
// Single choke point that every tensor op routes its primary operand through,
// replacing the historical `IntToPtr(unpackInt64(v))` that blindly reinterpreted
// any operand as an eshkol_tensor_t* (segfaulting on a vector/int/string). The
// actual validation/coercion/error logic lives in the runtime helper
// `eshkol_tensor_operand_checked`, so there is exactly one source of truth and
// the per-op codegen stays a single call.
// ─────────────────────────────────────────────────────────────────────────────
/** @brief Type-checked unpack of a tensor operand to an `eshkol_tensor_t*`
 *         (see ESH-0069 note above): records the current source location
 *         for error reporting, coerces non-tagged operands to a tagged
 *         INT64 so the runtime helper can report a clean type error instead
 *         of dereferencing garbage, and calls
 *         `eshkol_tensor_operand_checked` to validate/coerce the operand. */
llvm::Value* TensorCodegen::unpackTensorOperandChecked(llvm::Value* tensor_val,
                                                       const char* op_name) {
    auto& b = ctx_.builder();

    // Record the current source location so the runtime error formatter can
    // prefix the message with "file:line:col:" (matches the arithmetic path).
    uint32_t line = ctx_.currentSourceLine();
    if (line != 0) {
        llvm::Function* set_loc = ctx_.module().getFunction("eshkol_set_error_location");
        if (!set_loc) {
            llvm::FunctionType* ft = llvm::FunctionType::get(
                b.getVoidTy(),
                {ctx_.ptrType(), ctx_.int32Type(), ctx_.int32Type()},
                false);
            set_loc = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                             "eshkol_set_error_location", &ctx_.module());
        }
        const std::string& file = ctx_.currentSourceFile();
        llvm::Value* file_str = file.empty()
            ? static_cast<llvm::Value*>(llvm::ConstantPointerNull::get(ctx_.ptrType()))
            : ctx_.internCString(file);
        b.CreateCall(set_loc, {
            file_str,
            llvm::ConstantInt::get(ctx_.int32Type(), line),
            llvm::ConstantInt::get(ctx_.int32Type(), ctx_.currentSourceColumn())});
    }

    // The operand must arrive as a 16-byte tagged value; store it to an alloca
    // and hand the runtime helper its address (a by-value tagged-value struct
    // zeroes across the IR→C ABI boundary).
    if (tensor_val->getType() != ctx_.taggedValueType()) {
        // A raw (untagged) scalar — e.g. an integer literal operand. Tag it as
        // an INT64 (matching tensorArithmeticInternal / emitOperandTypeError);
        // the runtime helper then reports a clean type error rather than
        // dereferencing the value as a heap pointer. Never pack as a heap ptr:
        // ESHKOL_GET_HEADER on a small integer would read out of bounds.
        tensor_val = tagged_.packInt64(tensor_val, true);
    }
    llvm::Value* slot = b.CreateAlloca(ctx_.taggedValueType(), nullptr, "tensor_operand_slot");
    b.CreateStore(tensor_val, slot);

    llvm::Function* fn = ctx_.module().getFunction("eshkol_tensor_operand_checked");
    if (!fn) {
        // void* eshkol_tensor_operand_checked(const eshkol_tagged_value_t*, const char*)
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.ptrType(), {ctx_.ptrType(), ctx_.ptrType()}, false);
        fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                    "eshkol_tensor_operand_checked", &ctx_.module());
    }
    llvm::Value* name = ctx_.internCString(op_name ? op_name : "tensor-op");
    return b.CreateCall(fn, {slot, name}, "tensor_ptr_checked");
}

// Note: All tensor implementations are complex and depend on:
// - AST code generation for nested expressions
// - Autodiff integration (dual numbers, AD nodes)
// - Arena memory allocation for results
// - Runtime library functions
//
// These implementations remain in llvm_codegen.cpp until those modules are extracted.

/** @brief Unimplemented fallback for AST-driven tensor creation; logs a
 *         warning and returns a packed null (real tensor-literal codegen is
 *         handled by tensorOperation()). */
llvm::Value* TensorCodegen::createTensor(const eshkol_ast_t* ast) {
    eshkol_warn("TensorCodegen::createTensor called - using fallback");
    return tagged_.packNull();
}

/**
 * @brief Codegen for the `(tensor ...)` literal/constructor form.
 *
 * A single-element form whose argument is a runtime list/vector/scalar is
 * routed through `eshkol_tensor_from_collection` to build a 1-D tensor
 * (numpy-like unpacking). Otherwise allocates a tensor object (with header),
 * its dimensions array, and its elements array from the arena, evaluates
 * and stores each element (converted to a double bit pattern), and returns
 * the tensor as a packed HEAP_PTR tagged value. Raises an "out of memory"
 * Eshkol exception if arena allocation fails.
 */
llvm::Value* TensorCodegen::tensorOperation(const eshkol_operations_t* op) {
    if (!op || op->op != ESHKOL_TENSOR_OP) return nullptr;

    auto& builder = ctx_.builder();
    auto& context = ctx_.context();

    // Get arena pointer
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(context, 0), ctx_.globalArena());

    // (tensor X) with a single argument: if X evaluates to a list or vector at
    // runtime, unpack it element-by-element into a 1-D tensor (numpy-like).
    // Otherwise this builds a 1-element tensor whose sole element is the
    // collection pointer reinterpreted as a double (garbage, e.g.
    // (tensor (list 1.0 2.0 3.0)) -> #(4.96e9)). Scalars still yield a 1-element
    // tensor; an existing tensor passes through. eshkol_tensor_from_collection
    // handles all three at runtime.
    if (op->tensor_op.num_dimensions == 1 && op->tensor_op.total_elements == 1) {
        llvm::Value* arg = codegenAST(&op->tensor_op.elements[0]);
        if (arg) {
            llvm::Value* arg_tagged;
            if (arg->getType() == ctx_.taggedValueType()) arg_tagged = arg;
            else if (arg->getType() == ctx_.doubleType()) arg_tagged = tagged_.packDouble(arg);
            else if (arg->getType()->isIntegerTy(64)) arg_tagged = tagged_.packInt64(arg, true);
            else arg_tagged = tagged_.packInt64(builder.CreateZExtOrTrunc(arg, ctx_.int64Type()), true);
            llvm::Function* cur_fn = builder.GetInsertBlock()->getParent();
            llvm::IRBuilder<> entryB(&cur_fn->getEntryBlock(), cur_fn->getEntryBlock().begin());
            llvm::Value* slot = entryB.CreateAlloca(ctx_.taggedValueType(), nullptr, "tensorop_in");
            builder.CreateStore(arg_tagged, slot);
            llvm::Function* fn = ctx_.module().getFunction("eshkol_tensor_from_collection");
            if (!fn) {
                llvm::FunctionType* ft = llvm::FunctionType::get(
                    ctx_.ptrType(), {ctx_.ptrType(), ctx_.ptrType()}, false);
                fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                    "eshkol_tensor_from_collection", &ctx_.module());
            }
            llvm::Value* tptr = builder.CreateCall(fn, {arena_ptr, slot});
            return tagged_.packHeapPtr(tptr);
        }
    }

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

    // Extract tensor pointer (type-checked: ESH-0069)
    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "tensor-get");

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

    // Bug II (2026-05-07): when num_indices == 1, the single arg may be
    // a cons list packing all dims (the NumPy-style `t[(i,j,k)]` idiom
    // Noesis uses for image-write framebuffers). The previous codegen
    // funneled it through `eshkol_unwrap_list_index`, which returns
    // only the *car* — collapsing every (0 _ _) read on a 3D tensor
    // to flat[0]. Fix: in the 1-arg case, route through
    // `eshkol_tensor_linear_from_index_arg` to compute the full
    // row-major offset, and use `eshkol_tensor_index_arg_count` to
    // decide scalar-vs-slice the same way the multi-arg path does.
    // The multi-arg path keeps the original per-arg unwrap.
    llvm::Function* unwrap_fn = ctx_.module().getFunction("eshkol_unwrap_list_index");
    if (!unwrap_fn) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.int64Type(), {ctx_.ptrType()}, false);
        unwrap_fn = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage,
            "eshkol_unwrap_list_index", &ctx_.module());
    }

    std::vector<llvm::Value*> indices;
    llvm::Value* runtime_linear = nullptr;
    llvm::Value* runtime_index_count = nullptr;

    if (num_indices == 1) {
        llvm::Function* lin_fn = ctx_.module().getFunction("eshkol_tensor_linear_from_index_arg");
        if (!lin_fn) {
            llvm::FunctionType* ft = llvm::FunctionType::get(
                ctx_.int64Type(),
                {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
            lin_fn = llvm::Function::Create(
                ft, llvm::Function::ExternalLinkage,
                "eshkol_tensor_linear_from_index_arg", &ctx_.module());
        }
        llvm::Function* count_fn = ctx_.module().getFunction("eshkol_tensor_index_arg_count");
        if (!count_fn) {
            llvm::FunctionType* ft = llvm::FunctionType::get(
                ctx_.int64Type(), {ctx_.ptrType()}, false);
            count_fn = llvm::Function::Create(
                ft, llvm::Function::ExternalLinkage,
                "eshkol_tensor_index_arg_count", &ctx_.module());
        }

        llvm::Value* idx = codegenAST(&op->call_op.variables[1]);
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
        runtime_linear = ctx_.builder().CreateCall(lin_fn, {idx_slot, dims_ptr, ndim});
        runtime_index_count = ctx_.builder().CreateCall(count_fn, {idx_slot});
    } else {
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
    }

    llvm::Value* linear_offset = nullptr;
    llvm::Value* prod_dims = nullptr;
    llvm::Value* num_indices_val = nullptr;

    if (runtime_linear) {
        // Bug II fix path: linear and count come from the runtime helper
        // that walked the cons list. Skip per-arg bounds check (the
        // runtime helper trusts the user — bounds violations would
        // produce a wrong but non-crashing offset; could harden later).
        // Compute prod_dims = product(dims[0..count-1]) with a small
        // IR loop, since we can't constant-fold it at codegen time.
        linear_offset = runtime_linear;
        num_indices_val = runtime_index_count;

        llvm::Function* fn = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* pd_entry = ctx_.builder().GetInsertBlock();
        llvm::BasicBlock* pd_cond = llvm::BasicBlock::Create(ctx_.context(), "pd_cond", fn);
        llvm::BasicBlock* pd_body = llvm::BasicBlock::Create(ctx_.context(), "pd_body", fn);
        llvm::BasicBlock* pd_done = llvm::BasicBlock::Create(ctx_.context(), "pd_done", fn);

        ctx_.builder().CreateBr(pd_cond);
        ctx_.builder().SetInsertPoint(pd_cond);
        llvm::PHINode* pd_i = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "pd_i");
        llvm::PHINode* pd_acc = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "pd_acc");
        pd_i->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 0), pd_entry);
        pd_acc->addIncoming(llvm::ConstantInt::get(ctx_.int64Type(), 1), pd_entry);

        llvm::Value* pd_done_cond = ctx_.builder().CreateICmpUGE(pd_i, runtime_index_count);
        ctx_.builder().CreateCondBr(pd_done_cond, pd_done, pd_body);

        ctx_.builder().SetInsertPoint(pd_body);
        llvm::Value* pd_dim_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), dims_ptr, pd_i);
        llvm::Value* pd_dim_v = ctx_.builder().CreateLoad(ctx_.int64Type(), pd_dim_ptr);
        llvm::Value* pd_acc_next = ctx_.builder().CreateMul(pd_acc, pd_dim_v);
        llvm::Value* pd_i_next = ctx_.builder().CreateAdd(pd_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        pd_i->addIncoming(pd_i_next, ctx_.builder().GetInsertBlock());
        pd_acc->addIncoming(pd_acc_next, ctx_.builder().GetInsertBlock());
        ctx_.builder().CreateBr(pd_cond);

        ctx_.builder().SetInsertPoint(pd_done);
        prod_dims = pd_acc;
    } else {
        // Multi-arg path: indices vector is populated. Static bounds
        // check per-arg; static prod_dims accumulates dims[i] in IR.
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
        linear_offset = llvm::ConstantInt::get(ctx_.int64Type(), 0);
        prod_dims = llvm::ConstantInt::get(ctx_.int64Type(), 1);

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

        num_indices_val = llvm::ConstantInt::get(ctx_.int64Type(), num_indices);
    }

    // ===== BOUNDS CHECK: linear_offset in [0, total_elements) =====
    // The single-index (num_indices == 1) path routes through
    // eshkol_tensor_linear_from_index_arg, which trusts the caller and does
    // NOT bounds-check the resulting offset. Without this guard a negative or
    // out-of-range scalar/list index (e.g. (tensor-get t 900000000000)) walked
    // straight into an out-of-bounds load below (ASAN: SEGV on READ). An
    // unsigned compare catches both negatives (wrap to huge) and overshoot,
    // and — since any valid slice start is < total_elements — it is safe for
    // the partial-index slice path too. Mirrors the guard tensor-set! already
    // performs on its own linear index.
    {
        llvm::Value* oob = ctx_.builder().CreateICmpUGE(linear_offset, total_elements);
        llvm::Function* cur_fn = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* tget_bounds_ok = llvm::BasicBlock::Create(ctx_.context(), "tget_bounds_ok", cur_fn);
        llvm::BasicBlock* tget_bounds_err = llvm::BasicBlock::Create(ctx_.context(), "tget_bounds_err", cur_fn);
        ctx_.builder().CreateCondBr(oob, tget_bounds_err, tget_bounds_ok);

        ctx_.builder().SetInsertPoint(tget_bounds_err);
        {
            llvm::Function* raise_func = ctx_.module().getFunction("eshkol_raise");
            if (!raise_func) {
                llvm::FunctionType* raise_type = llvm::FunctionType::get(
                    ctx_.builder().getVoidTy(), {ctx_.ptrType()}, false);
                raise_func = llvm::Function::Create(raise_type, llvm::Function::ExternalLinkage,
                    "eshkol_raise", &ctx_.module());
                raise_func->setDoesNotReturn();
            }
            llvm::Function* make_exc_func = ctx_.module().getFunction("eshkol_make_exception_with_header");
            if (!make_exc_func) {
                llvm::FunctionType* make_type = llvm::FunctionType::get(ctx_.ptrType(),
                    {ctx_.builder().getInt32Ty(), ctx_.ptrType()}, false);
                make_exc_func = llvm::Function::Create(make_type, llvm::Function::ExternalLinkage,
                    "eshkol_make_exception_with_header", &ctx_.module());
            }
            llvm::Value* msg = ctx_.builder().CreateGlobalString(
                "tensor-get: index out of bounds");
            llvm::Value* exc_type = llvm::ConstantInt::get(ctx_.builder().getInt32Ty(), ESHKOL_EXCEPTION_ERROR);
            llvm::Value* exc = ctx_.builder().CreateCall(make_exc_func, {exc_type, msg});
            ctx_.builder().CreateCall(raise_func, {exc});
            ctx_.builder().CreateUnreachable();
        }

        ctx_.builder().SetInsertPoint(tget_bounds_ok);
    }

    // ===== PHASE 2: Decide scalar vs slice =====
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

/** @brief Unimplemented fallback for `vref`; the real AD-aware
 *         implementation lives in codegenTensorVectorRef() in
 *         llvm_codegen.cpp. Logs a warning and returns a packed null. */
llvm::Value* TensorCodegen::vectorRef(const eshkol_operations_t* op) {
    // vref is AD-aware and complex - remains in llvm_codegen.cpp
    eshkol_warn("TensorCodegen::vectorRef called - AD-aware vref should use codegenTensorVectorRef");
    return tagged_.packNull();
}

/**
 * @brief Codegen for `(tensor-set! tensor index... value)` (R7RS-style,
 *        value is the last argument).
 *
 * Supports both a flat list of scalar per-dimension indices and a single
 * list/cons argument packing all dimensions (dispatched via
 * `eshkol_tensor_linear_from_index_arg`), computes and bounds-checks the
 * row-major linear offset, then stores the new value (converted to its
 * double bit pattern) into the tensor's backing element array. Emits a
 * runtime error and traps (unreachable) on too many indices or an
 * out-of-bounds index; returns the (validated) tensor pointer on success.
 */
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

    // Extract the tensor pointer from the tagged value (type-checked: ESH-0069)
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "tensor-set!");

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
    // Two shapes are supported:
    //   (A) (tensor-set! t i j k v)         — N scalar args, one per dim
    //   (B) (tensor-set! t (list i j k) v)  — single list arg holding all dims
    //
    // Bug II (2026-05-07): the previous codegen funneled each arg through
    // `eshkol_unwrap_list_index`, which returns just the *car* of a cons
    // list. That made (B) silently collapse to `flat[car]` — a 3D
    // (list 0 1 0) write hit flat[0] instead of flat[3], and any
    // (tensor-ref t (list 0 _ _)) read came back flat[0]. Symptom: an
    // image-write framebuffer rendered nearly-black with a smear in the
    // top-left.  Fix: when there's exactly one index argument, dispatch
    // to `eshkol_tensor_linear_from_index_arg`, which walks the cons
    // chain and computes the row-major linear offset from the tensor's
    // own dims — covering both list and scalar cases in one call. The
    // multi-arg path keeps the per-arg unwrap (each arg is one dim).
    if (num_set_indices == 1) {
        llvm::Function* lin_fn = ctx_.module().getFunction("eshkol_tensor_linear_from_index_arg");
        if (!lin_fn) {
            llvm::FunctionType* ft = llvm::FunctionType::get(
                ctx_.int64Type(),
                {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
            lin_fn = llvm::Function::Create(
                ft, llvm::Function::ExternalLinkage,
                "eshkol_tensor_linear_from_index_arg", &ctx_.module());
        }

        llvm::Value* index = codegenAST(&op->call_op.variables[1]);
        if (index) {
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
            linear_index = ctx_.builder().CreateCall(lin_fn,
                {idx_slot, typed_dims_ptr, ndim});
        }
    } else {
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

    return tagged_.packHeapPtr(tensor_ptr); // Return the (validated) tensor
}

// Shared codegen for tensor-rect-fill! / tensor-disk-fill!. Both
// builtins take a tensor + a fixed prefix of integer args + 1..C
// numeric "channel value" args, and dispatch to a single C runtime
// helper that does the heavy lifting per-pixel. We allocate a stack
// buffer of `num_channels` doubles, populate it from the trailing
// args, and pass (tensor_tv*, ints..., chan_ptr, num_channels) to
// the runtime — so per-pixel cost in C is one memcpy + one store
// per channel, not a tagged-value dispatch.
//
// Why ptr-to-tagged-value rather than struct-by-value: same backend
// convention used everywhere else in tensor_codegen.cpp — direct
// struct-by-value calls to C cause ABI divergence on ARM64 between
// the LLVM-emitted call and the C runtime. Always alloca + pointer.
llvm::Value* TensorCodegen::tensorRectFill(const eshkol_operations_t* op) {
    // (tensor-rect-fill! t row0 col0 row1 col1 v0 [v1 ...])
    //   6 fixed args + N channel values, N >= 1
    if (op->call_op.num_vars < 6) {
        eshkol_error("tensor-rect-fill! requires tensor, row0, col0, row1, col1, value, ...");
        return nullptr;
    }
    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;
    if (tensor_val->getType() != ctx_.taggedValueType()) {
        tensor_val = tagged_.packInt64(tagged_.safeExtractInt64(tensor_val), false);
    }

    llvm::Value* row0 = tagged_.safeExtractInt64(codegenAST(&op->call_op.variables[1]));
    llvm::Value* col0 = tagged_.safeExtractInt64(codegenAST(&op->call_op.variables[2]));
    llvm::Value* row1 = tagged_.safeExtractInt64(codegenAST(&op->call_op.variables[3]));
    llvm::Value* col1 = tagged_.safeExtractInt64(codegenAST(&op->call_op.variables[4]));

    const uint64_t num_channels = op->call_op.num_vars - 5;

    // Materialise tensor + chan buffer in entry block so call sees
    // stable pointers regardless of where this codegen runs.
    llvm::IRBuilderBase::InsertPoint saved = ctx_.builder().saveIP();
    llvm::Function* cur_fn = ctx_.builder().GetInsertBlock()->getParent();
    ctx_.builder().SetInsertPoint(&cur_fn->getEntryBlock(), cur_fn->getEntryBlock().begin());
    llvm::Value* t_slot = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "rect_t_slot");
    llvm::Value* chan_buf = ctx_.builder().CreateAlloca(
        ctx_.doubleType(),
        llvm::ConstantInt::get(ctx_.int64Type(), num_channels),
        "rect_chans");
    ctx_.builder().restoreIP(saved);
    ctx_.builder().CreateStore(tensor_val, t_slot);

    for (uint64_t i = 0; i < num_channels; i++) {
        llvm::Value* v = codegenAST(&op->call_op.variables[5 + i]);
        if (!v) return nullptr;
        llvm::Value* d = extractAsDouble(v);
        llvm::Value* slot = ctx_.builder().CreateGEP(
            ctx_.doubleType(), chan_buf,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        ctx_.builder().CreateStore(d, slot);
    }

    llvm::Function* fn = ctx_.module().getFunction("eshkol_tensor_rect_fill");
    if (!fn) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.builder().getVoidTy(),
            {ctx_.ptrType(),                     // tensor tagged-value*
             ctx_.int64Type(), ctx_.int64Type(), // row0, col0
             ctx_.int64Type(), ctx_.int64Type(), // row1, col1
             ctx_.ptrType(),                     // double* channels
             ctx_.int64Type()},                  // num_channels
            false);
        fn = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage,
            "eshkol_tensor_rect_fill", &ctx_.module());
    }
    ctx_.builder().CreateCall(fn, {
        t_slot, row0, col0, row1, col1, chan_buf,
        llvm::ConstantInt::get(ctx_.int64Type(), num_channels)});

    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorDiskFill(const eshkol_operations_t* op) {
    // (tensor-disk-fill! t cy cx radius v0 [v1 ...])
    //   4 fixed args + N channel values, N >= 1
    if (op->call_op.num_vars < 5) {
        eshkol_error("tensor-disk-fill! requires tensor, cy, cx, radius, value, ...");
        return nullptr;
    }
    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;
    if (tensor_val->getType() != ctx_.taggedValueType()) {
        tensor_val = tagged_.packInt64(tagged_.safeExtractInt64(tensor_val), false);
    }

    llvm::Value* cy = tagged_.safeExtractInt64(codegenAST(&op->call_op.variables[1]));
    llvm::Value* cx = tagged_.safeExtractInt64(codegenAST(&op->call_op.variables[2]));
    llvm::Value* radius = tagged_.safeExtractInt64(codegenAST(&op->call_op.variables[3]));

    const uint64_t num_channels = op->call_op.num_vars - 4;

    llvm::IRBuilderBase::InsertPoint saved = ctx_.builder().saveIP();
    llvm::Function* cur_fn = ctx_.builder().GetInsertBlock()->getParent();
    ctx_.builder().SetInsertPoint(&cur_fn->getEntryBlock(), cur_fn->getEntryBlock().begin());
    llvm::Value* t_slot = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "disk_t_slot");
    llvm::Value* chan_buf = ctx_.builder().CreateAlloca(
        ctx_.doubleType(),
        llvm::ConstantInt::get(ctx_.int64Type(), num_channels),
        "disk_chans");
    ctx_.builder().restoreIP(saved);
    ctx_.builder().CreateStore(tensor_val, t_slot);

    for (uint64_t i = 0; i < num_channels; i++) {
        llvm::Value* v = codegenAST(&op->call_op.variables[4 + i]);
        if (!v) return nullptr;
        llvm::Value* d = extractAsDouble(v);
        llvm::Value* slot = ctx_.builder().CreateGEP(
            ctx_.doubleType(), chan_buf,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        ctx_.builder().CreateStore(d, slot);
    }

    llvm::Function* fn = ctx_.module().getFunction("eshkol_tensor_disk_fill");
    if (!fn) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.builder().getVoidTy(),
            {ctx_.ptrType(),                                          // tensor tagged-value*
             ctx_.int64Type(), ctx_.int64Type(), ctx_.int64Type(),    // cy, cx, radius
             ctx_.ptrType(), ctx_.int64Type()},                       // chans, num
            false);
        fn = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage,
            "eshkol_tensor_disk_fill", &ctx_.module());
    }
    ctx_.builder().CreateCall(fn, {
        t_slot, cy, cx, radius, chan_buf,
        llvm::ConstantInt::get(ctx_.int64Type(), num_channels)});

    return tagged_.packNull();
}

/** @brief Codegen entry point for binary tensor arithmetic
 *         (`tensor-add`/`tensor-sub`/`tensor-mul`/`tensor-div`): evaluates
 *         both operands and delegates to tensorArithmeticInternal(), which
 *         handles both Scheme vectors and tensors plus SIMD dispatch. */
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

// Internal & SIMD tensor arithmetic: see tensor_arith_codegen.cpp

// Matmul / reduction / apply: see tensor_reduce_codegen.cpp

// ===== TYPE CONVERSION: VECTOR ↔ TENSOR =====

/**
 * @brief Codegen for `(vector->tensor vec)`: allocates a 1-D tensor with the
 *        same length as the Scheme vector, then emits a copy loop that
 *        unpacks each 16-byte tagged vector element to a double and stores
 *        it into the tensor's element array.
 */
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

/**
 * @brief Codegen for `(tensor->vector tensor)`: allocates a Scheme vector
 *        sized to the tensor's total element count, then emits a copy loop
 *        that wraps each tensor double as a tagged value and stores it into
 *        the vector's element array (flattening any shape to 1-D).
 */
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

    // Unpack tensor (type-checked: ESH-0069)
    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "tensor->vector");

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

/** @brief Codegen for `(tensor-var t)`: computes the variance E[(x-mean)^2]
 *         of a tensor's elements via two generated loops (one to accumulate
 *         the mean, one to accumulate squared deviations from it) and
 *         returns the result as a packed double. */
llvm::Value* TensorCodegen::tensorVar(const eshkol_operations_t* op) {
    // Variance: E[(x - mean)^2]
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-var requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    auto& builder = ctx_.builder();

    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "tensor-var");
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

/** @brief Codegen for `(tensor-std t)`: standard deviation, computed as
 *         sqrt(tensorVar(t)). */
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

/** @brief Codegen for `(rand dim1 dim2 ...)`: allocates a tensor of the
 *         given shape and fills it with uniform random doubles in [0, 1)
 *         generated via `drand48` in an emitted fill loop. */
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

/** @brief Codegen for `(randn dim1 dim2 ...)`: allocates a tensor of the
 *         given shape and fills it with standard-normal random doubles
 *         (mean 0, std 1) using the Box-Muller transform on `drand48`
 *         samples, in an emitted fill loop. */
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

/** @brief Codegen for `(randint low high dim1 dim2 ...)`: allocates a
 *         tensor of the given shape and fills it with random integers in
 *         [low, high), stored as doubles, in an emitted fill loop. */
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

// Shape operations (Phase 4): see tensor_shape_codegen.cpp

// Tensor creation: see tensor_creation_codegen.cpp

// Convolution & pooling: see tensor_conv_codegen.cpp

// ============================================================
// Phase 4/5/7 extras + tensor unary/binary/scale/batch-matmul: see tensor_extras_codegen.cpp

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
