/*
 * XLA Memory Integration Implementation for Eshkol
 *
 * Bridges Eshkol's OALR (Ownership-Aware Lexical Regions) arena memory
 * model with XLA buffer management. CPU path uses zero-copy arena access;
 * GPU paths will use pinned or unified memory when those targets are active.
 *
 * In LLVM-direct mode (v1.1), allocation functions emit IR calls to the
 * arena runtime (arena_allocate_aligned) with 64-byte alignment for
 * AVX-512/cache-line compatibility.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/xla_memory.h"
#include "eshkol/backend/xla/xla_codegen.h"
#include "eshkol/backend/xla/xla_types.h"
#include "eshkol/backend/codegen_context.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Constants.h>

// GPU memory functions for device transfer
#include "eshkol/backend/gpu/gpu_memory.h"

namespace eshkol {
namespace xla {

// ===== XLAMemoryIntegration Implementation =====

class XLAMemoryIntegration::Impl {
public:
    CodegenContext* ctx_ = nullptr;
    MemoryStrategy cpu_strategy_ = MemoryStrategy::ZERO_COPY;
    MemoryStrategy gpu_strategy_ = MemoryStrategy::PINNED;

    // Allocation statistics for diagnostics
    size_t arena_allocated_ = 0;
    size_t device_allocated_ = 0;
    size_t transfers_ = 0;

    explicit Impl(CodegenContext& ctx) : ctx_(&ctx) {}

    /// Get or declare the arena_allocate_aligned runtime function.
    /// Signature: void* arena_allocate_aligned(void* arena, int64_t size, int64_t alignment)
    llvm::Function* getOrDeclareArenaAllocateAligned() {
        auto& mod = ctx_->module();
        auto* existing = mod.getFunction("arena_allocate_aligned");
        if (existing) return existing;

        auto& llvm_ctx = ctx_->context();
        auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
        auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);
        auto* funcTy = llvm::FunctionType::get(ptrTy, {ptrTy, i64Ty, i64Ty}, false);
        return llvm::Function::Create(funcTy, llvm::GlobalValue::ExternalLinkage,
                                      "arena_allocate_aligned", &mod);
    }
};

XLAMemoryIntegration::XLAMemoryIntegration(CodegenContext& ctx)
    : impl_(std::make_unique<Impl>(ctx)) {}

XLAMemoryIntegration::~XLAMemoryIntegration() = default;

// ===== Buffer Wrapping =====

/// Wrap an existing arena tensor for XLA use.
/// CPU path: zero-copy -- the arena tensor is already in host memory, so we
/// simply return the tensor pointer as the XLA buffer handle.
llvm::Value* XLAMemoryIntegration::wrapArenaTensor(llvm::Value* arena_ptr,
                                                    llvm::Value* tensor_ptr,
                                                    Target target) {
    (void)arena_ptr;
    (void)target;
    // CPU: zero-copy -- arena tensor IS the XLA buffer.
    // GPU targets would pin or transfer here; handled at runtime dispatch level.
    return tensor_ptr;
}

/// Wrap a raw data pointer as an XLA-compatible buffer.
/// For CPU/LLVM-direct mode, the raw data pointer is returned directly --
/// the caller (xla_codegen.cpp) handles tensor struct creation when needed.
llvm::Value* XLAMemoryIntegration::wrapRawBuffer(llvm::Value* data_ptr,
                                                   const std::vector<int64_t>& shape,
                                                   size_t element_size,
                                                   Target target) {
    (void)shape;
    (void)element_size;
    (void)target;
    // CPU/LLVM-direct: return the data pointer; caller wraps into tensor struct.
    return data_ptr;
}

// ===== Buffer Allocation =====

/// Allocate an XLA buffer in the arena with 64-byte alignment.
/// Emits LLVM IR that calls arena_allocate_aligned at runtime.
/// The arena manages lifetime via OALR regions, so no explicit free is needed.
llvm::Value* XLAMemoryIntegration::allocateXLABuffer(llvm::Value* arena_ptr,
                                                       const TensorType& type,
                                                       BufferScope scope) {
    (void)scope;  // Arena scoping handles lifetime; BufferScope is advisory.

    size_t size_bytes = type.sizeInBytes();
    if (size_bytes == 0) {
        // Dynamic-sized tensor -- caller must handle dynamic allocation.
        // Return nullptr to signal that static allocation is not possible.
        return nullptr;
    }

    // Track allocation for diagnostics
    impl_->arena_allocated_ += size_bytes;

    auto& builder = impl_->ctx_->builder();
    auto& llvm_ctx = impl_->ctx_->context();
    auto* ptrTy = llvm::PointerType::get(llvm_ctx, 0);
    auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

    // Get or declare arena_allocate_aligned
    auto* allocFunc = impl_->getOrDeclareArenaAllocateAligned();

    // Load the arena pointer from the global
    auto* arenaPtr = arena_ptr;
    auto* sizeVal = llvm::ConstantInt::get(i64Ty, size_bytes);
    auto* alignVal = llvm::ConstantInt::get(i64Ty, 64);  // 64-byte alignment for AVX-512/XLA
    return builder.CreateCall(allocFunc, {arenaPtr, sizeVal, alignVal}, "xla_buffer");
}

/// Allocate a raw aligned buffer in the arena with caller-specified alignment.
/// Emits LLVM IR that calls arena_allocate_aligned at runtime.
llvm::Value* XLAMemoryIntegration::allocateAligned(llvm::Value* arena_ptr,
                                                     size_t size_bytes,
                                                     size_t alignment) {
    // Track allocation for diagnostics
    impl_->arena_allocated_ += size_bytes;

    auto& builder = impl_->ctx_->builder();
    auto& llvm_ctx = impl_->ctx_->context();
    auto* i64Ty = llvm::Type::getInt64Ty(llvm_ctx);

    // Get or declare arena_allocate_aligned
    auto* allocFunc = impl_->getOrDeclareArenaAllocateAligned();

    auto* sizeVal = llvm::ConstantInt::get(i64Ty, size_bytes);
    auto* alignVal = llvm::ConstantInt::get(i64Ty, alignment);
    return builder.CreateCall(allocFunc, {arena_ptr, sizeVal, alignVal}, "xla_aligned_buffer");
}

// ===== Device Transfer =====

/// Ensure tensor is on the specified device.
/// CPU path: no-op -- data is already in host memory.
/// GPU targets: emit IR to call eshkol_gpu_wrap_host + eshkol_gpu_sync(HOST_TO_DEVICE).
/// Note: In v1.1, actual GPU dispatch happens at the runtime level (xla_runtime.cpp),
/// so this function primarily serves as a target-aware passthrough. The runtime functions
/// handle GPU buffer wrapping/sync internally per-operation.
llvm::Value* XLAMemoryIntegration::ensureDevice(llvm::Value* tensor_ptr, Target target) {
    if (target == Target::CPU) {
        return tensor_ptr;
    }

    // For GPU targets, track the transfer for diagnostics
    impl_->transfers_++;

    // In v1.1, GPU dispatch is handled per-operation in xla_runtime.cpp
    // (eshkol_xla_matmul, eshkol_xla_elementwise, etc. call eshkol_gpu_wrap_host
    // internally). The tensor pointer is returned as-is since the runtime dispatch
    // layer handles the actual device transfer transparently.
    return tensor_ptr;
}

/// Synchronize device tensor back to host memory.
/// CPU path: no-op -- host and device are the same.
/// GPU targets: runtime dispatch handles sync internally per-operation.
void XLAMemoryIntegration::syncToHost(llvm::Value* tensor_ptr) {
    (void)tensor_ptr;
    // In v1.1, GPU sync is handled per-operation in xla_runtime.cpp.
    // Each XLA runtime function wraps/syncs GPU buffers internally.
}

// ===== Region Integration =====

/// Register buffer with OALR region for automatic cleanup.
/// CPU path: no-op -- the arena itself tracks all allocations and frees them
/// when the region exits. GPU targets would register device-side cleanup callbacks.
void XLAMemoryIntegration::registerWithRegion(llvm::Value* buffer,
                                                llvm::Value* region_ptr,
                                                Target target) {
    (void)buffer;
    (void)region_ptr;
    (void)target;
    // Arena handles all cleanup for CPU allocations.
}

/// Begin an XLA computation scope. Enables memory reuse optimization.
/// CPU path: no-op -- arena scoping is handled by the OALR region system.
void XLAMemoryIntegration::beginComputationScope(llvm::Value* region_ptr) {
    (void)region_ptr;
    // Arena region scoping manages computation-local memory.
}

/// End an XLA computation scope. Releases temporary buffers.
/// CPU path: no-op -- arena handles deallocation at region exit.
void XLAMemoryIntegration::endComputationScope() {
    // Arena region exit handles cleanup.
}

// ===== Memory Strategy =====

/// Set the memory strategy for a given target backend.
void XLAMemoryIntegration::setStrategy(Target target, MemoryStrategy strategy) {
    switch (target) {
        case Target::CPU:
            impl_->cpu_strategy_ = strategy;
            break;
        case Target::CUDA:
        case Target::Metal:
        case Target::Vulkan:
            impl_->gpu_strategy_ = strategy;
            break;
    }
}

/// Get the current memory strategy for a given target backend.
MemoryStrategy XLAMemoryIntegration::getStrategy(Target target) const {
    switch (target) {
        case Target::CPU:
            return impl_->cpu_strategy_;
        case Target::CUDA:
        case Target::Metal:
        case Target::Vulkan:
            return impl_->gpu_strategy_;
    }
    return MemoryStrategy::ZERO_COPY;
}

/// Return the recommended memory strategy for the given target.
/// CPU uses zero-copy (arena memory directly), CUDA/Vulkan use pinned memory
/// for efficient DMA transfer, Metal uses Apple unified memory.
MemoryStrategy XLAMemoryIntegration::recommendedStrategy(Target target) {
    switch (target) {
        case Target::CPU:
            return MemoryStrategy::ZERO_COPY;
        case Target::CUDA:
        case Target::Vulkan:
            return MemoryStrategy::PINNED;
        case Target::Metal:
            return MemoryStrategy::UNIFIED;
    }
    return MemoryStrategy::ZERO_COPY;
}

// ===== Diagnostics =====

/// Retrieve memory allocation statistics.
void XLAMemoryIntegration::getStats(size_t& arena_allocated,
                                      size_t& device_allocated,
                                      size_t& transfers) const {
    arena_allocated = impl_->arena_allocated_;
    device_allocated = impl_->device_allocated_;
    transfers = impl_->transfers_;
}

/// Reset all statistics counters to zero.
void XLAMemoryIntegration::resetStats() {
    impl_->arena_allocated_ = 0;
    impl_->device_allocated_ = 0;
    impl_->transfers_ = 0;
}

} // namespace xla
} // namespace eshkol
