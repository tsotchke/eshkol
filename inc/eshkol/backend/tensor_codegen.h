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

    /**
     * Reshape tensor: (reshape tensor new-dims...)
     * @param op The operation AST node
     * @return Reshaped tensor (shares data with original)
     */
    llvm::Value* reshape(const eshkol_operations_t* op);

    // === Tensor Creation Functions ===

    /**
     * Create zero-filled tensor: (zeros dim1 dim2 ...)
     * @param op The operation AST node
     * @return Tensor filled with zeros
     */
    llvm::Value* zeros(const eshkol_operations_t* op);

    /**
     * Create one-filled tensor: (ones dim1 dim2 ...)
     * @param op The operation AST node
     * @return Tensor filled with ones
     */
    llvm::Value* ones(const eshkol_operations_t* op);

    /**
     * Create identity matrix: (eye n) or (eye rows cols)
     * @param op The operation AST node
     * @return Identity matrix
     */
    llvm::Value* eye(const eshkol_operations_t* op);

    /**
     * Create range tensor: (arange stop) or (arange start stop) or (arange start stop step)
     * @param op The operation AST node
     * @return Range tensor
     */
    llvm::Value* arange(const eshkol_operations_t* op);

    /**
     * Create linspace tensor: (linspace start stop num)
     * @param op The operation AST node
     * @return Evenly spaced tensor
     */
    llvm::Value* linspace(const eshkol_operations_t* op);

    // === Tensor Utility (Public for use by other codegen modules) ===

    /**
     * Create a tensor with given dimensions.
     * @param dims Vector of dimension values
     * @param fill_value Optional fill value (as i64 bit pattern)
     * @param use_memset_zero If true, use memset for efficient zero-fill
     * @return Pointer to tensor struct
     */
    llvm::Value* createTensorWithDims(const std::vector<llvm::Value*>& dims,
                                       llvm::Value* fill_value = nullptr,
                                       bool use_memset_zero = false);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    MemoryCodegen& mem_;

    // Callback for AST code generation (matches other codegen modules)
    using CodegenASTFunc = llvm::Value* (*)(const void* ast, void* context);
    using CodegenTypedASTFunc = void* (*)(const void* ast, void* context);
    using TypedToTaggedFunc = llvm::Value* (*)(void* typed_value, void* context);

    CodegenASTFunc codegen_ast_callback_ = nullptr;
    CodegenTypedASTFunc codegen_typed_ast_callback_ = nullptr;
    TypedToTaggedFunc typed_to_tagged_callback_ = nullptr;
    void* callback_context_ = nullptr;

    // === Internal Helpers ===

    /**
     * Call codegenAST via callback.
     */
    llvm::Value* codegenAST(const eshkol_ast_t* ast) {
        if (codegen_ast_callback_ && ast) {
            return codegen_ast_callback_(ast, callback_context_);
        }
        return nullptr;
    }

    /**
     * Scheme vector arithmetic (VECTOR_PTR type).
     * Vectors use tagged_value elements after an 8-byte length field.
     * @param vec1 First vector (tagged)
     * @param vec2 Second vector (tagged)
     * @param operation One of "add", "sub", "mul", "div"
     * @return Result vector (tagged)
     */
    llvm::Value* schemeVectorArithmetic(llvm::Value* vec1, llvm::Value* vec2, const std::string& operation);

    /**
     * Tensor arithmetic for TENSOR_PTR type.
     * Tensors use double elements in a contiguous array.
     * @param tensor1 First tensor (tagged)
     * @param tensor2 Second tensor (tagged)
     * @param operation One of "add", "sub", "mul", "div"
     * @return Result tensor (tagged)
     */
    llvm::Value* rawTensorArithmetic(llvm::Value* tensor1, llvm::Value* tensor2, const std::string& operation);

    /**
     * Extract a tagged value as double, handling both int64 and double types.
     * @param tagged_val The tagged value
     * @return The extracted double value
     */
    llvm::Value* extractAsDouble(llvm::Value* tagged_val);

public:
    /**
     * Set callbacks for AST code generation.
     */
    void setCodegenCallbacks(
        CodegenASTFunc codegen_ast,
        CodegenTypedASTFunc codegen_typed_ast,
        TypedToTaggedFunc typed_to_tagged,
        void* context
    ) {
        codegen_ast_callback_ = codegen_ast;
        codegen_typed_ast_callback_ = codegen_typed_ast;
        typed_to_tagged_callback_ = typed_to_tagged;
        callback_context_ = context;
    }
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_TENSOR_CODEGEN_H
