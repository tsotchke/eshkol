/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_BACKEND_TENSOR_SHAPE_GUARD_H
#define ESHKOL_BACKEND_TENSOR_SHAPE_GUARD_H

#include <eshkol/backend/codegen_context.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/ADT/ArrayRef.h>
#include <string>

namespace eshkol {

inline void emitCatchableTensorContractGuard(CodegenContext& ctx,
                                             llvm::Value* condition,
                                             const std::string& message,
                                             const char* block_prefix) {
    llvm::IRBuilder<>& builder = ctx.builder();
    llvm::Function* function = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* ok = llvm::BasicBlock::Create(ctx.context(),
        std::string(block_prefix) + "_ok", function);
    llvm::BasicBlock* error = llvm::BasicBlock::Create(ctx.context(),
        std::string(block_prefix) + "_error", function);
    builder.CreateCondBr(condition, ok, error);
    builder.SetInsertPoint(error);
    ctx.emitRaiseFmt(message.c_str(), {});
    builder.SetInsertPoint(ok);
}

// Emit one catchable guard for operands that must have identical ranks and
// extents. The runtime comparison is also used by tensor arithmetic and
// tensor-dot, keeping shape semantics centralized and exact (same products do
// not make differently shaped tensors equal).
inline void emitMatchingTensorShapes(CodegenContext& ctx,
                                     llvm::ArrayRef<llvm::Value*> tensors,
                                     const char* operation) {
    if (tensors.size() < 2) return;

    llvm::IRBuilder<>& builder = ctx.builder();
    llvm::StructType* tensor_type = ctx.tensorType();
    llvm::FunctionCallee shapes_equal = ctx.module().getOrInsertFunction(
        "eshkol_shapes_equal",
        llvm::FunctionType::get(ctx.int64Type(),
            {ctx.ptrType(), ctx.int64Type(), ctx.ptrType(), ctx.int64Type()}, false));

    llvm::Value* all_equal = llvm::ConstantInt::getTrue(ctx.context());
    llvm::Value* first_dims = builder.CreateLoad(ctx.ptrType(),
        builder.CreateStructGEP(tensor_type, tensors[0], 0));
    llvm::Value* first_rank = builder.CreateLoad(ctx.int64Type(),
        builder.CreateStructGEP(tensor_type, tensors[0], 1));
    for (size_t i = 1; i < tensors.size(); ++i) {
        llvm::Value* dims = builder.CreateLoad(ctx.ptrType(),
            builder.CreateStructGEP(tensor_type, tensors[i], 0));
        llvm::Value* rank = builder.CreateLoad(ctx.int64Type(),
            builder.CreateStructGEP(tensor_type, tensors[i], 1));
        llvm::Value* equal = builder.CreateICmpNE(
            builder.CreateCall(shapes_equal, {first_dims, first_rank, dims, rank}),
            llvm::ConstantInt::get(ctx.int64Type(), 0));
        all_equal = builder.CreateAnd(all_equal, equal);
    }

    const std::string message = std::string(operation) +
        ": tensor ranks and extents must match";
    emitCatchableTensorContractGuard(ctx, all_equal, message,
                                     (std::string(operation) + "_shape").c_str());
}

} // namespace eshkol

#endif
