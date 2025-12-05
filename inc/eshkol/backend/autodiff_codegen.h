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
#include <llvm/IR/GlobalVariable.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

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
     * Pack dual number to tagged value (heap-allocated).
     * @param dual The dual number struct
     * @return Tagged dual number value
     */
    llvm::Value* packDualToTagged(llvm::Value* dual);

    /**
     * Unpack dual number from tagged value.
     * @param tagged The tagged dual number
     * @return Dual number struct
     */
    llvm::Value* unpackDualFromTagged(llvm::Value* tagged);

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

    /**
     * Create a variable AD node (input to gradient computation).
     * @param value The variable value
     * @param var_index Index of this variable
     * @return AD node
     */
    llvm::Value* createADVariable(llvm::Value* value, size_t var_index);

    /**
     * Load input node 1 from an AD node.
     * @param node_ptr Pointer to AD node
     * @return Input node 1 pointer
     */
    llvm::Value* loadNodeInput1(llvm::Value* node_ptr);

    /**
     * Load input node 2 from an AD node.
     * @param node_ptr Pointer to AD node
     * @return Input node 2 pointer
     */
    llvm::Value* loadNodeInput2(llvm::Value* node_ptr);

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

    /**
     * Compute derivative (forward-mode AD): (derivative func point)
     * @param op The derivative operation AST node
     * @return Derivative value as double
     */
    llvm::Value* derivative(const eshkol_operations_t* op);

    /**
     * Compute Hessian matrix (second derivatives): (hessian func args...)
     * @param op The Hessian operation AST node
     * @return Hessian matrix as tensor
     */
    llvm::Value* hessian(const eshkol_operations_t* op);

    /**
     * Compute divergence (trace of Jacobian): (divergence func args...)
     * @param op The divergence operation AST node
     * @return Scalar divergence value
     */
    llvm::Value* divergence(const eshkol_operations_t* op);

    /**
     * Compute curl (3D vector field rotation): (curl func args...)
     * @param op The curl operation AST node
     * @return 3D vector as tensor
     */
    llvm::Value* curl(const eshkol_operations_t* op);

    /**
     * Compute Laplacian (trace of Hessian): (laplacian func args...)
     * @param op The Laplacian operation AST node
     * @return Scalar Laplacian value
     */
    llvm::Value* laplacian(const eshkol_operations_t* op);

    /**
     * Compute directional derivative: (directional-derivative func point direction)
     * @param op The directional derivative operation AST node
     * @return Scalar directional derivative
     */
    llvm::Value* directionalDerivative(const eshkol_operations_t* op);

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

    // === Dual Number Math Operations ===
    // These implement chain rule for various math functions

    /**
     * Sine of dual number: (sin(a), a' * cos(a))
     */
    llvm::Value* dualSin(llvm::Value* dual);

    /**
     * Cosine of dual number: (cos(a), -a' * sin(a))
     */
    llvm::Value* dualCos(llvm::Value* dual);

    /**
     * Exponential of dual number: (exp(a), a' * exp(a))
     */
    llvm::Value* dualExp(llvm::Value* dual);

    /**
     * Natural log of dual number: (log(a), a' / a)
     */
    llvm::Value* dualLog(llvm::Value* dual);

    /**
     * Square root of dual number: (sqrt(a), a' / (2 * sqrt(a)))
     */
    llvm::Value* dualSqrt(llvm::Value* dual);

    /**
     * Power of dual number: (a^b, a^b * (b' * ln(a) + b * a' / a))
     * For constant exponent: (a^n, n * a^(n-1) * a')
     */
    llvm::Value* dualPow(llvm::Value* dual_base, llvm::Value* dual_exp);

    /**
     * Tangent of dual number: (tan(a), a' / cos²(a))
     */
    llvm::Value* dualTan(llvm::Value* dual);

    /**
     * Arc sine of dual number: (asin(a), a' / sqrt(1 - a²))
     */
    llvm::Value* dualAsin(llvm::Value* dual);

    /**
     * Arc cosine of dual number: (acos(a), -a' / sqrt(1 - a²))
     */
    llvm::Value* dualAcos(llvm::Value* dual);

    /**
     * Arc tangent of dual number: (atan(a), a' / (1 + a²))
     */
    llvm::Value* dualAtan(llvm::Value* dual);

    /**
     * Absolute value of dual number: (|a|, a' * sign(a))
     */
    llvm::Value* dualAbs(llvm::Value* dual);

    /**
     * Negation of dual number: (-a, -a')
     */
    llvm::Value* dualNeg(llvm::Value* dual);

    /**
     * Hyperbolic sine: (sinh(a), a' * cosh(a))
     */
    llvm::Value* dualSinh(llvm::Value* dual);

    /**
     * Hyperbolic cosine: (cosh(a), a' * sinh(a))
     */
    llvm::Value* dualCosh(llvm::Value* dual);

    /**
     * Hyperbolic tangent: (tanh(a), a' * sech²(a))
     */
    llvm::Value* dualTanh(llvm::Value* dual);

    // === Tape Management ===
    // These enable nested gradient computations via a tape stack

    /**
     * Push current tape context and activate a new tape.
     * @param new_tape The new tape to activate
     */
    void pushTapeContext(llvm::Value* new_tape);

    /**
     * Pop tape context, restoring the previous tape.
     */
    void popTapeContext();

    /**
     * Get the outer tape (for nested gradient access).
     * @return Outer tape pointer, or null if not nested
     */
    llvm::Value* getOuterTape();

    /**
     * Check if currently in a nested gradient context.
     * @return i1 value (true if nested)
     */
    llvm::Value* isNested();

    /**
     * Create AD constant node on a specific tape.
     * @param tape_ptr The tape to add to
     * @param value The constant value
     * @return AD node pointer
     */
    llvm::Value* createADConstantOnTape(llvm::Value* tape_ptr, llvm::Value* value);

    /**
     * Record binary operation on a specific tape.
     * Used for double backward computations.
     */
    llvm::Value* recordADNodeBinaryOnTape(llvm::Value* tape_ptr, uint32_t op_type,
                                           llvm::Value* left_node, llvm::Value* right_node);

    // === AD Node Helpers ===

    /**
     * Load the value field from an AD node.
     */
    llvm::Value* loadNodeValue(llvm::Value* node_ptr);

    /**
     * Load the gradient field from an AD node.
     */
    llvm::Value* loadNodeGradient(llvm::Value* node_ptr);

    /**
     * Store a gradient value to an AD node.
     */
    void storeNodeGradient(llvm::Value* node_ptr, llvm::Value* gradient);

    /**
     * Accumulate gradient (add to existing gradient).
     */
    void accumulateGradient(llvm::Value* node_ptr, llvm::Value* gradient_to_add);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    MemoryCodegen& mem_;

    // Node ID counter for AD graph nodes
    uint64_t next_node_id_ = 0;

    // Helper: Get arena pointer from global
    llvm::Value* getArenaPtr();

    // Helper: Get or declare math function
    llvm::Function* getMathFunc(const std::string& name);

    // Helper: Propagate gradient from a node to its inputs based on operation type
    void propagateGradient(llvm::Value* node_ptr);

    // References to function table (for sin, cos, exp, etc.)
    std::unordered_map<std::string, llvm::Function*>* function_table_ = nullptr;

    // Symbol tables for variable/capture lookup
    std::unordered_map<std::string, llvm::Value*>* symbol_table_ = nullptr;
    std::unordered_map<std::string, llvm::Value*>* global_symbol_table_ = nullptr;

    // REPL mode flag pointer
    bool* repl_mode_enabled_ = nullptr;

    // Callbacks for AST codegen and function resolution
    void* callback_context_ = nullptr;

    // Callback types
    using CodegenASTCallback = llvm::Value* (*)(const eshkol_ast_t*, void*);
    CodegenASTCallback codegen_ast_callback_ = nullptr;

    using ResolveLambdaCallback = llvm::Value* (*)(const eshkol_ast_t*, size_t, void*);
    ResolveLambdaCallback resolve_lambda_callback_ = nullptr;

    // Pack/unpack helpers that delegate to main codegen
    using PackDualCallback = llvm::Value* (*)(llvm::Value*, llvm::Value*, void*);
    PackDualCallback pack_dual_callback_ = nullptr;

    using UnpackDualCallback = std::pair<llvm::Value*, llvm::Value*> (*)(llvm::Value*, void*);
    UnpackDualCallback unpack_dual_callback_ = nullptr;

    using PackDualToTaggedCallback = llvm::Value* (*)(llvm::Value*, void*);
    PackDualToTaggedCallback pack_dual_to_tagged_callback_ = nullptr;

    using UnpackDualFromTaggedCallback = llvm::Value* (*)(llvm::Value*, void*);
    UnpackDualFromTaggedCallback unpack_dual_from_tagged_callback_ = nullptr;

    using PackInt64Callback = llvm::Value* (*)(llvm::Value*, bool, void*);
    PackInt64Callback pack_int64_callback_ = nullptr;

public:
    /**
     * Set function table reference (for math functions).
     */
    void setFunctionTable(std::unordered_map<std::string, llvm::Function*>* table) {
        function_table_ = table;
    }

    /**
     * Set symbol table references for variable/capture lookup.
     */
    void setSymbolTables(
        std::unordered_map<std::string, llvm::Value*>* symbol_table,
        std::unordered_map<std::string, llvm::Value*>* global_symbol_table
    ) {
        symbol_table_ = symbol_table;
        global_symbol_table_ = global_symbol_table;
    }

    /**
     * Set REPL mode flag pointer.
     */
    void setReplMode(bool* repl_mode) {
        repl_mode_enabled_ = repl_mode;
    }

    /**
     * Set callback for general AST code generation.
     */
    void setCodegenASTCallback(CodegenASTCallback callback, void* context) {
        codegen_ast_callback_ = callback;
        callback_context_ = context;
    }

    /**
     * Set callback for resolving lambda functions.
     */
    void setResolveLambdaCallback(ResolveLambdaCallback callback) {
        resolve_lambda_callback_ = callback;
    }

    /**
     * Set dual number pack/unpack callbacks.
     */
    void setDualCallbacks(
        PackDualCallback pack_dual,
        UnpackDualCallback unpack_dual,
        PackDualToTaggedCallback pack_dual_to_tagged,
        UnpackDualFromTaggedCallback unpack_dual_from_tagged,
        PackInt64Callback pack_int64
    ) {
        pack_dual_callback_ = pack_dual;
        unpack_dual_callback_ = unpack_dual;
        pack_dual_to_tagged_callback_ = pack_dual_to_tagged;
        unpack_dual_from_tagged_callback_ = unpack_dual_from_tagged;
        pack_int64_callback_ = pack_int64;
    }
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_AUTODIFF_CODEGEN_H
