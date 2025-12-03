/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * ArithmeticCodegen - Polymorphic arithmetic code generation
 *
 * This module handles arithmetic operations on tagged values, supporting:
 * - Integer arithmetic (exact)
 * - Floating-point arithmetic (inexact)
 * - Mixed integer/float operations (promoting to float)
 * - Dual number arithmetic (forward-mode AD)
 * - AD node arithmetic (reverse-mode AD graph construction)
 * - Vector/tensor element-wise arithmetic
 */
#ifndef ESHKOL_BACKEND_ARITHMETIC_CODEGEN_H
#define ESHKOL_BACKEND_ARITHMETIC_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <llvm/IR/Value.h>

namespace eshkol {

/**
 * ArithmeticCodegen handles polymorphic arithmetic on tagged values.
 *
 * All operations take tagged_value inputs and return tagged_value results.
 * The type of the result depends on the input types:
 * - int + int = int
 * - int + double = double
 * - double + double = double
 * - dual + any = dual (forward-mode AD)
 * - ad_node + any = ad_node (reverse-mode AD graph)
 * - vector/tensor + vector/tensor = vector/tensor (element-wise)
 */
class ArithmeticCodegen {
public:
    /**
     * Construct ArithmeticCodegen with context and tagged value helper.
     */
    ArithmeticCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged);

    // === Binary Arithmetic Operations ===

    /**
     * Polymorphic addition: a + b
     * @param left Left operand (tagged_value)
     * @param right Right operand (tagged_value)
     * @return Result as tagged_value
     */
    llvm::Value* add(llvm::Value* left, llvm::Value* right);

    /**
     * Polymorphic subtraction: a - b
     * @param left Left operand (tagged_value)
     * @param right Right operand (tagged_value)
     * @return Result as tagged_value
     */
    llvm::Value* sub(llvm::Value* left, llvm::Value* right);

    /**
     * Polymorphic multiplication: a * b
     * @param left Left operand (tagged_value)
     * @param right Right operand (tagged_value)
     * @return Result as tagged_value
     */
    llvm::Value* mul(llvm::Value* left, llvm::Value* right);

    /**
     * Polymorphic division: a / b
     * @param left Left operand (tagged_value)
     * @param right Right operand (tagged_value)
     * @return Result as tagged_value
     */
    llvm::Value* div(llvm::Value* left, llvm::Value* right);

    /**
     * Polymorphic modulo: a % b (integer only)
     * @param left Left operand (tagged_value)
     * @param right Right operand (tagged_value)
     * @return Result as tagged_value
     */
    llvm::Value* mod(llvm::Value* left, llvm::Value* right);

    // === Unary Arithmetic Operations ===

    /**
     * Polymorphic negation: -a
     * @param operand Operand (tagged_value)
     * @return Result as tagged_value
     */
    llvm::Value* neg(llvm::Value* operand);

    /**
     * Polymorphic absolute value: |a|
     * @param operand Operand (tagged_value)
     * @return Result as tagged_value
     */
    llvm::Value* abs(llvm::Value* operand);

    // === Type Coercion ===

    /**
     * Convert integer to double.
     * @param int_tagged Integer as tagged_value
     * @return Double as tagged_value
     */
    llvm::Value* intToDouble(llvm::Value* int_tagged);

    /**
     * Convert double to integer (truncation).
     * @param double_tagged Double as tagged_value
     * @return Integer as tagged_value
     */
    llvm::Value* doubleToInt(llvm::Value* double_tagged);

    // === Helper: Extract numeric value as double ===

    /**
     * Extract the numeric value from a tagged value as a double.
     * Works for INT64, DOUBLE, and DUAL_NUMBER types.
     * @param tagged Tagged value
     * @return LLVM double value
     */
    llvm::Value* extractAsDouble(llvm::Value* tagged);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;

    // Implementation callback type for tensor/AD operations
    // These are set by the main codegen to avoid circular dependencies
    using TensorArithFunc = llvm::Value* (*)(llvm::Value*, llvm::Value*, const std::string&, void*);
    using ADNodeCreateFunc = llvm::Value* (*)(llvm::Value*, void*);

    TensorArithFunc tensor_arith_callback_ = nullptr;
    ADNodeCreateFunc ad_const_callback_ = nullptr;
    void* callback_context_ = nullptr;

public:
    /**
     * Set callbacks for tensor and AD operations.
     * Called by EshkolLLVMCodeGen to inject dependencies.
     */
    void setTensorArithCallback(TensorArithFunc func, void* context) {
        tensor_arith_callback_ = func;
        callback_context_ = context;
    }

    void setADConstCallback(ADNodeCreateFunc func) {
        ad_const_callback_ = func;
    }
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_ARITHMETIC_CODEGEN_H
