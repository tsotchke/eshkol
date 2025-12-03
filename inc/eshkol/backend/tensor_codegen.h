/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen - Tensor operations code generation
 *
 * This module handles:
 * - Tensor creation and access
 * - Tensor arithmetic (add, sub, mul, div, dot)
 * - Tensor transformations (apply, reduce, transpose)
 * - Shape operations
 */
#ifndef ESHKOL_BACKEND_TENSOR_CODEGEN_H
#define ESHKOL_BACKEND_TENSOR_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/memory_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <string>

namespace eshkol {

/**
 * TensorCodegen handles tensor operations.
 *
 * Tensors in Eshkol are n-dimensional arrays with:
 * - Arbitrary dimensions (1D vectors, 2D matrices, etc.)
 * - Element-wise arithmetic with broadcasting
 * - Support for autodiff operations
 */
class TensorCodegen {
public:
    /**
     * Construct TensorCodegen with context and helpers.
     */
    TensorCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem);

    // === Tensor Creation ===

    /**
     * Create a tensor from a literal: #[1 2 3] or #[[1 2] [3 4]]
     * @param ast The tensor AST node
     * @return Tagged tensor pointer
     */
    llvm::Value* createTensor(const eshkol_ast_t* ast);

    /**
     * Create a tensor via operation: (tensor ...)
     * @param op The tensor operation AST node
     * @return Tagged tensor pointer
     */
    llvm::Value* tensorOperation(const eshkol_operations_t* op);

    // === Tensor Access ===

    /**
     * Get element at indices: (tensor-get tensor idx1 idx2 ...)
     * @param op The operation AST node
     * @return Tagged value at the specified indices
     */
    llvm::Value* tensorGet(const eshkol_operations_t* op);

    /**
     * Vector reference (1D): (vref tensor idx)
     * @param op The operation AST node
     * @return Tagged value at index
     */
    llvm::Value* vectorRef(const eshkol_operations_t* op);

    /**
     * Set element at indices: (tensor-set tensor val idx1 idx2 ...)
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* tensorSet(const eshkol_operations_t* op);

    // === Tensor Arithmetic ===

    /**
     * Element-wise arithmetic: tensor-add, tensor-sub, tensor-mul, tensor-div
     * @param op The operation AST node
     * @param operation One of "add", "sub", "mul", "div"
     * @return Result tensor
     */
    llvm::Value* tensorArithmetic(const eshkol_operations_t* op, const std::string& operation);

    /**
     * Internal arithmetic implementation for two tagged values.
     * @param left Left operand (tagged)
     * @param right Right operand (tagged)
     * @param operation One of "add", "sub", "mul", "div"
     * @return Result tensor
     */
    llvm::Value* tensorArithmeticInternal(llvm::Value* left, llvm::Value* right, const std::string& operation);

    /**
     * Dot product / matrix multiplication: (tensor-dot A B)
     * @param op The operation AST node
     * @return Result tensor or scalar
     */
    llvm::Value* tensorDot(const eshkol_operations_t* op);

    // === Tensor Transformations ===

    /**
     * Apply function element-wise: (tensor-apply tensor func)
     * @param op The operation AST node
     * @return Result tensor
     */
    llvm::Value* tensorApply(const eshkol_operations_t* op);

    /**
     * Reduce all elements: (tensor-reduce-all tensor func init)
     * @param op The operation AST node
     * @return Scalar result
     */
    llvm::Value* tensorReduceAll(const eshkol_operations_t* op);

    /**
     * Reduce along dimension: (tensor-reduce tensor func init dim)
     * @param op The operation AST node
     * @return Result tensor
     */
    llvm::Value* tensorReduceWithDim(const eshkol_operations_t* op);

    /**
     * Sum all elements: (tensor-sum tensor)
     * @param op The operation AST node
     * @return Scalar sum
     */
    llvm::Value* tensorSum(const eshkol_operations_t* op);

    /**
     * Mean of all elements: (tensor-mean tensor)
     * @param op The operation AST node
     * @return Scalar mean
     */
    llvm::Value* tensorMean(const eshkol_operations_t* op);

    // === Shape Operations ===

    /**
     * Get tensor shape: (tensor-shape tensor)
     * @param op The operation AST node
     * @return Vector of dimensions
     */
    llvm::Value* tensorShape(const eshkol_operations_t* op);

    /**
     * Transpose tensor: (transpose tensor)
     * @param op The operation AST node
     * @return Transposed tensor
     */
    llvm::Value* transpose(const eshkol_operations_t* op);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    MemoryCodegen& mem_;

    // Callback for AST code generation
    using CodegenASTFunc = llvm::Value* (*)(const void* ast, void* context);
    CodegenASTFunc codegen_ast_callback_ = nullptr;
    void* callback_context_ = nullptr;

public:
    /**
     * Set callback for AST code generation.
     */
    void setCodegenCallback(CodegenASTFunc callback, void* context) {
        codegen_ast_callback_ = callback;
        callback_context_ = context;
    }
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_TENSOR_CODEGEN_H
