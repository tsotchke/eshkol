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
#include <eshkol/backend/tensor_codegen.h>
#include <eshkol/backend/autodiff_codegen.h>
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
     * Construct ArithmeticCodegen with all dependencies.
     * @param ctx The shared codegen context
     * @param tagged Tagged value operations helper
     * @param tensor Tensor operations helper
     * @param autodiff Automatic differentiation helper
     */
    ArithmeticCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged,
                      TensorCodegen& tensor, AutodiffCodegen& autodiff);

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

    // === Comparison Operations ===

    /**
     * Polymorphic comparison: a <op> b
     * Handles int/double/string comparisons with type promotion.
     * @param left Left operand (tagged_value)
     * @param right Right operand (tagged_value)
     * @param operation One of: "lt", "gt", "eq", "le", "ge"
     * @return Boolean result as tagged_value
     */
    llvm::Value* compare(llvm::Value* left, llvm::Value* right, const std::string& operation);

    // === Math Functions ===

    /**
     * Unary math function with dual number and AD node support.
     * @param operand Operand (tagged_value)
     * @param func_name Function name (sin, cos, exp, log, etc.)
     * @return Result as tagged_value
     */
    llvm::Value* mathFunc(llvm::Value* operand, const std::string& func_name);

    /**
     * Power function: base^exponent
     * @param base Base value (tagged_value)
     * @param exponent Exponent value (tagged_value)
     * @return Result as tagged_value
     */
    llvm::Value* pow(llvm::Value* base, llvm::Value* exponent);

    /**
     * Minimum of two values.
     * @param left Left operand (tagged_value)
     * @param right Right operand (tagged_value)
     * @return Minimum as tagged_value
     */
    llvm::Value* min(llvm::Value* left, llvm::Value* right);

    /**
     * Maximum of two values.
     * @param left Left operand (tagged_value)
     * @param right Right operand (tagged_value)
     * @return Maximum as tagged_value
     */
    llvm::Value* max(llvm::Value* left, llvm::Value* right);

    /**
     * Integer remainder (Scheme remainder semantics).
     * @param dividend Dividend (tagged_value)
     * @param divisor Divisor (tagged_value)
     * @return Remainder as tagged_value
     */
    llvm::Value* remainder(llvm::Value* dividend, llvm::Value* divisor);

    /**
     * Integer quotient (truncated division).
     * @param dividend Dividend (tagged_value)
     * @param divisor Divisor (tagged_value)
     * @return Quotient as tagged_value
     */
    llvm::Value* quotient(llvm::Value* dividend, llvm::Value* divisor);

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
    TensorCodegen& tensor_;
    AutodiffCodegen& autodiff_;

    // === Internal Helpers ===

    /**
     * Convert operand to dual number (promote constants to dual with zero tangent).
     * @param operand Tagged value operand
     * @param is_dual Whether operand is already a dual number
     * @param is_double Whether operand is a double
     * @return Dual number struct
     */
    llvm::Value* convertToDual(llvm::Value* operand, llvm::Value* is_dual, llvm::Value* is_double);

    /**
     * Convert operand to AD node (promote constants to constant AD nodes).
     * @param operand Tagged value operand
     * @param is_ad Whether operand is already an AD node
     * @param base_type The base type of the operand
     * @return AD node pointer
     */
    llvm::Value* convertToADNode(llvm::Value* operand, llvm::Value* is_ad, llvm::Value* base_type);

    /**
     * Check if a tagged value is specifically an AD node.
     * Safely checks both CALLABLE type AND CALLABLE_SUBTYPE_AD_NODE subtype.
     * Uses proper branching to avoid dereferencing non-pointer values.
     * @param operand Tagged value operand
     * @param base_type The base type of the operand
     * @return Boolean i1 value: true if AD node, false otherwise
     */
    llvm::Value* isADNode(llvm::Value* operand, llvm::Value* base_type);
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_ARITHMETIC_CODEGEN_H
