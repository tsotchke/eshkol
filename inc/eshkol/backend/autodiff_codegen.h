/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * AutodiffCodegen - Automatic differentiation code generation
 *
 * This module handles:
 * - Forward-mode AD (dual numbers)
 * - Reverse-mode AD (tape recording, backpropagation)
 * - Gradient computation
 * - Jacobian matrix computation
 */
#ifndef ESHKOL_BACKEND_AUTODIFF_CODEGEN_H
#define ESHKOL_BACKEND_AUTODIFF_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/memory_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <cstdint>

namespace eshkol {

/**
 * AutodiffCodegen handles automatic differentiation operations.
 *
 * Eshkol supports two modes of automatic differentiation:
 * 1. Forward-mode AD using dual numbers (efficient for few inputs, many outputs)
 * 2. Reverse-mode AD using tape recording (efficient for many inputs, few outputs)
 */
class AutodiffCodegen {
public:
    /**
     * Construct AutodiffCodegen with context and helpers.
     */
    AutodiffCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem);

    // === Dual Number Operations (Forward-mode AD) ===

    /**
     * Create a dual number: (primal, tangent)
     * @param primal The primal value
     * @param tangent The tangent (derivative) value
     * @return Tagged dual number
     */
    llvm::Value* createDualNumber(llvm::Value* primal, llvm::Value* tangent);

    /**
     * Extract primal from dual number.
     * @param dual The dual number
     * @return Primal value as double
     */
    llvm::Value* getDualPrimal(llvm::Value* dual);

    /**
     * Extract tangent from dual number.
     * @param dual The dual number
     * @return Tangent value as double
     */
    llvm::Value* getDualTangent(llvm::Value* dual);

    /**
     * Add two dual numbers: (a + b, a' + b')
     * @param left Left dual number
     * @param right Right dual number
     * @return Result dual number
     */
    llvm::Value* dualAdd(llvm::Value* left, llvm::Value* right);

    /**
     * Subtract dual numbers: (a - b, a' - b')
     */
    llvm::Value* dualSub(llvm::Value* left, llvm::Value* right);

    /**
     * Multiply dual numbers: (a * b, a * b' + a' * b)
     */
    llvm::Value* dualMul(llvm::Value* left, llvm::Value* right);

    /**
     * Divide dual numbers: (a / b, (a' * b - a * b') / b^2)
     */
    llvm::Value* dualDiv(llvm::Value* left, llvm::Value* right);

    // === AD Node Operations (Reverse-mode AD) ===

    /**
     * Record a binary operation on the AD tape.
     * @param op_type Operation type (add=2, sub=3, mul=4, div=5)
     * @param left Left operand AD node
     * @param right Right operand AD node
     * @return Result AD node
     */
    llvm::Value* recordADNodeBinary(uint32_t op_type, llvm::Value* left, llvm::Value* right);

    /**
     * Record a unary operation on the AD tape.
     * @param op_type Operation type (sin=6, cos=7, exp=8, log=9, etc.)
     * @param input Input AD node
     * @return Result AD node
     */
    llvm::Value* recordADNodeUnary(uint32_t op_type, llvm::Value* input);

    /**
     * Create a constant AD node (leaf node with no dependencies).
     * @param value The constant value
     * @return AD node
     */
    llvm::Value* createADConstant(llvm::Value* value);

    // === High-level AD Operations ===

    /**
     * Compute gradient: (grad func args...)
     * @param op The gradient operation AST node
     * @return Gradient value or vector
     */
    llvm::Value* gradient(const eshkol_operations_t* op);

    /**
     * Compute Jacobian matrix: (jacobian func args...)
     * @param op The Jacobian operation AST node
     * @return Jacobian matrix as tensor
     */
    llvm::Value* jacobian(const eshkol_operations_t* op);

    // === Tape Management ===

    /**
     * Create a new AD tape.
     * @return Tape pointer
     */
    llvm::Value* createTape();

    /**
     * Run backpropagation on tape.
     * @param tape The tape to backpropagate
     * @param output_node The output node to differentiate from
     */
    void backpropagate(llvm::Value* tape, llvm::Value* output_node);

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
#endif // ESHKOL_BACKEND_AUTODIFF_CODEGEN_H
