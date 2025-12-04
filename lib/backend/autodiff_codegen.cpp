/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * AutodiffCodegen implementation
 *
 * Note: The complex autodiff implementations remain in llvm_codegen.cpp
 * for now due to extensive dependencies on AST codegen, tape management,
 * and runtime library functions. This module provides the interface.
 */

#include <eshkol/backend/autodiff_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>

namespace eshkol {

AutodiffCodegen::AutodiffCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem)
    : ctx_(ctx)
    , tagged_(tagged)
    , mem_(mem) {
    eshkol_debug("AutodiffCodegen initialized");
}

// ===== DUAL NUMBER OPERATIONS (Forward-mode AD) =====
// Fully implemented - these are self-contained and don't depend on AST codegen

llvm::Value* AutodiffCodegen::createDualNumber(llvm::Value* primal, llvm::Value* tangent) {
    if (!primal || !tangent) return nullptr;

    // Create alloca for dual number at function entry
    llvm::IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    if (func && !func->empty()) {
        llvm::BasicBlock& entry = func->getEntryBlock();
        ctx_.builder().SetInsertPoint(&entry, entry.begin());
    }

    llvm::Value* dual_ptr = ctx_.builder().CreateAlloca(ctx_.dualNumberType(), nullptr, "dual");

    // Restore insertion point for the actual stores
    ctx_.builder().restoreIP(saved_ip);

    // Store primal in field 0
    llvm::Value* primal_ptr = ctx_.builder().CreateStructGEP(ctx_.dualNumberType(), dual_ptr, 0);
    ctx_.builder().CreateStore(primal, primal_ptr);

    // Store tangent in field 1
    llvm::Value* tangent_ptr = ctx_.builder().CreateStructGEP(ctx_.dualNumberType(), dual_ptr, 1);
    ctx_.builder().CreateStore(tangent, tangent_ptr);

    // Load and return the dual number struct
    return ctx_.builder().CreateLoad(ctx_.dualNumberType(), dual_ptr);
}

llvm::Value* AutodiffCodegen::getDualPrimal(llvm::Value* dual) {
    if (!dual) return llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    // Store dual to temporary alloca
    llvm::Value* dual_ptr = ctx_.builder().CreateAlloca(ctx_.dualNumberType(), nullptr, "temp_dual");
    ctx_.builder().CreateStore(dual, dual_ptr);

    // Extract primal (field 0)
    llvm::Value* primal_ptr = ctx_.builder().CreateStructGEP(ctx_.dualNumberType(), dual_ptr, 0);
    return ctx_.builder().CreateLoad(ctx_.doubleType(), primal_ptr);
}

llvm::Value* AutodiffCodegen::getDualTangent(llvm::Value* dual) {
    if (!dual) return llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    // Store dual to temporary alloca
    llvm::Value* dual_ptr = ctx_.builder().CreateAlloca(ctx_.dualNumberType(), nullptr, "temp_dual");
    ctx_.builder().CreateStore(dual, dual_ptr);

    // Extract tangent (field 1)
    llvm::Value* tangent_ptr = ctx_.builder().CreateStructGEP(ctx_.dualNumberType(), dual_ptr, 1);
    return ctx_.builder().CreateLoad(ctx_.doubleType(), tangent_ptr);
}

llvm::Value* AutodiffCodegen::packDualToTagged(llvm::Value* dual) {
    if (!dual) return nullptr;

    // Get global arena pointer
    llvm::GlobalVariable* arena_global = ctx_.module().getNamedGlobal("__global_arena");
    if (!arena_global) {
        eshkol_warn("packDualToTagged: __global_arena not found");
        return tagged_.packNull();
    }

    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), arena_global);

    // Allocate space for dual number on the heap (arena)
    // dual_number is 16 bytes (two doubles)
    llvm::Value* size = llvm::ConstantInt::get(ctx_.int64Type(), 16);
    llvm::Function* alloc_func = mem_.getArenaAllocate();
    if (!alloc_func) {
        eshkol_warn("packDualToTagged: arena_allocate not found");
        return tagged_.packNull();
    }

    llvm::Value* dual_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr, size});

    // Store dual number to heap-allocated memory
    ctx_.builder().CreateStore(dual, dual_ptr);

    // Pack as pointer type tagged value with DUAL_NUMBER type
    return tagged_.packPtr(dual_ptr, ESHKOL_VALUE_DUAL_NUMBER);
}

llvm::Value* AutodiffCodegen::unpackDualFromTagged(llvm::Value* tagged_val) {
    if (!tagged_val) return nullptr;

    // Extract pointer from tagged value
    llvm::Value* ptr_val = tagged_.unpackPtr(tagged_val);

    // Load and return dual number
    return ctx_.builder().CreateLoad(ctx_.dualNumberType(), ptr_val);
}

// Dual arithmetic: (a, a') + (b, b') = (a+b, a'+b')
llvm::Value* AutodiffCodegen::dualAdd(llvm::Value* dual_a, llvm::Value* dual_b) {
    if (!dual_a || !dual_b) return nullptr;

    llvm::Value* a = getDualPrimal(dual_a);
    llvm::Value* a_prime = getDualTangent(dual_a);
    llvm::Value* b = getDualPrimal(dual_b);
    llvm::Value* b_prime = getDualTangent(dual_b);

    // Value: a + b
    llvm::Value* value = ctx_.builder().CreateFAdd(a, b);

    // Derivative: a' + b'
    llvm::Value* deriv = ctx_.builder().CreateFAdd(a_prime, b_prime);

    return createDualNumber(value, deriv);
}

// Dual arithmetic: (a, a') - (b, b') = (a-b, a'-b')
llvm::Value* AutodiffCodegen::dualSub(llvm::Value* dual_a, llvm::Value* dual_b) {
    if (!dual_a || !dual_b) return nullptr;

    llvm::Value* a = getDualPrimal(dual_a);
    llvm::Value* a_prime = getDualTangent(dual_a);
    llvm::Value* b = getDualPrimal(dual_b);
    llvm::Value* b_prime = getDualTangent(dual_b);

    // Value: a - b
    llvm::Value* value = ctx_.builder().CreateFSub(a, b);

    // Derivative: a' - b'
    llvm::Value* deriv = ctx_.builder().CreateFSub(a_prime, b_prime);

    return createDualNumber(value, deriv);
}

// Dual arithmetic: (a, a') * (b, b') = (a*b, a'*b + a*b') - product rule
llvm::Value* AutodiffCodegen::dualMul(llvm::Value* dual_a, llvm::Value* dual_b) {
    if (!dual_a || !dual_b) return nullptr;

    llvm::Value* a = getDualPrimal(dual_a);
    llvm::Value* a_prime = getDualTangent(dual_a);
    llvm::Value* b = getDualPrimal(dual_b);
    llvm::Value* b_prime = getDualTangent(dual_b);

    // Value: a * b
    llvm::Value* value = ctx_.builder().CreateFMul(a, b);

    // Derivative: a' * b + a * b' (product rule)
    llvm::Value* term1 = ctx_.builder().CreateFMul(a_prime, b);
    llvm::Value* term2 = ctx_.builder().CreateFMul(a, b_prime);
    llvm::Value* deriv = ctx_.builder().CreateFAdd(term1, term2);

    return createDualNumber(value, deriv);
}

// Dual arithmetic: (a, a') / (b, b') = (a/b, (a'*b - a*b')/b²) - quotient rule
llvm::Value* AutodiffCodegen::dualDiv(llvm::Value* dual_a, llvm::Value* dual_b) {
    if (!dual_a || !dual_b) return nullptr;

    llvm::Value* a = getDualPrimal(dual_a);
    llvm::Value* a_prime = getDualTangent(dual_a);
    llvm::Value* b = getDualPrimal(dual_b);
    llvm::Value* b_prime = getDualTangent(dual_b);

    // Value: a / b
    llvm::Value* value = ctx_.builder().CreateFDiv(a, b);

    // Derivative: (a' * b - a * b') / b²
    llvm::Value* numerator_term1 = ctx_.builder().CreateFMul(a_prime, b);
    llvm::Value* numerator_term2 = ctx_.builder().CreateFMul(a, b_prime);
    llvm::Value* numerator = ctx_.builder().CreateFSub(numerator_term1, numerator_term2);
    llvm::Value* denominator = ctx_.builder().CreateFMul(b, b);
    llvm::Value* deriv = ctx_.builder().CreateFDiv(numerator, denominator);

    return createDualNumber(value, deriv);
}

// Helper to get arena pointer from global
llvm::Value* AutodiffCodegen::getArenaPtr() {
    llvm::GlobalVariable* arena_global = ctx_.module().getNamedGlobal("__global_arena");
    if (!arena_global) return nullptr;
    return ctx_.builder().CreateLoad(ctx_.ptrType(), arena_global);
}

// Create AD node for a constant value (gradient = 0)
llvm::Value* AutodiffCodegen::createADConstant(llvm::Value* value) {
    if (!value) return nullptr;

    // Convert value to double if needed
    if (value->getType()->isIntegerTy()) {
        value = ctx_.builder().CreateSIToFP(value, ctx_.doubleType());
    }

    // Allocate AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNode();
    if (!alloc_func) return nullptr;

    llvm::Value* node_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr});

    llvm::StructType* ad_type = ctx_.adNodeType();

    // Set type = AD_NODE_CONSTANT (0)
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 0);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int32Type(), 0), type_ptr);

    // Set value
    llvm::Value* value_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1);
    ctx_.builder().CreateStore(value, value_ptr);

    // Initialize gradient = 0.0
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), grad_ptr);

    // Set input pointers to null
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input1_ptr);

    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input2_ptr);

    // Set node ID
    llvm::Value* id_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++), id_ptr);

    // Add node to tape
    llvm::GlobalVariable* tape_global = ctx_.currentAdTape();
    if (tape_global) {
        llvm::Value* tape_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), tape_global);
        llvm::Function* add_node_func = mem_.getArenaTapeAddNode();
        if (add_node_func) {
            ctx_.builder().CreateCall(add_node_func, {tape_ptr, node_ptr});
        }
    }

    return node_ptr;
}

// Record binary operation node (add, sub, mul, div) in computational graph
llvm::Value* AutodiffCodegen::recordADNodeBinary(uint32_t op_type, llvm::Value* left_node, llvm::Value* right_node) {
    if (!left_node || !right_node) return nullptr;

    llvm::StructType* ad_type = ctx_.adNodeType();

    // Load values from input nodes
    llvm::Value* left_value_ptr = ctx_.builder().CreateStructGEP(ad_type, left_node, 1);
    llvm::Value* left_value = ctx_.builder().CreateLoad(ctx_.doubleType(), left_value_ptr);

    llvm::Value* right_value_ptr = ctx_.builder().CreateStructGEP(ad_type, right_node, 1);
    llvm::Value* right_value = ctx_.builder().CreateLoad(ctx_.doubleType(), right_value_ptr);

    // Compute result value based on operation
    llvm::Value* result_value = nullptr;
    switch (op_type) {
        case 2: // AD_NODE_ADD
            result_value = ctx_.builder().CreateFAdd(left_value, right_value);
            break;
        case 3: // AD_NODE_SUB
            result_value = ctx_.builder().CreateFSub(left_value, right_value);
            break;
        case 4: // AD_NODE_MUL
            result_value = ctx_.builder().CreateFMul(left_value, right_value);
            break;
        case 5: // AD_NODE_DIV
            result_value = ctx_.builder().CreateFDiv(left_value, right_value);
            break;
        default:
            eshkol_warn("Unknown binary AD operation type: %u", op_type);
            return nullptr;
    }

    // Allocate new AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNode();
    if (!alloc_func) return nullptr;

    llvm::Value* node_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr});

    // Set operation type
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 0);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int32Type(), op_type), type_ptr);

    // Set computed value
    llvm::Value* value_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1);
    ctx_.builder().CreateStore(result_value, value_ptr);

    // Initialize gradient = 0.0
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), grad_ptr);

    // Set input pointers
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    ctx_.builder().CreateStore(left_node, input1_ptr);

    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    ctx_.builder().CreateStore(right_node, input2_ptr);

    // Set node ID
    llvm::Value* id_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++), id_ptr);

    // Add to tape
    llvm::GlobalVariable* tape_global = ctx_.currentAdTape();
    if (tape_global) {
        llvm::Value* tape_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), tape_global);
        llvm::Function* add_node_func = mem_.getArenaTapeAddNode();
        if (add_node_func) {
            ctx_.builder().CreateCall(add_node_func, {tape_ptr, node_ptr});
        }
    }

    return node_ptr;
}

llvm::Value* AutodiffCodegen::recordADNodeUnary(uint32_t op_type, llvm::Value* input) {
    // TODO: Implement when math functions are migrated
    eshkol_warn("AutodiffCodegen::recordADNodeUnary not yet fully implemented");
    return nullptr;
}

llvm::Value* AutodiffCodegen::gradient(const eshkol_operations_t* op) {
    // This requires AST codegen callback - use fallback for now
    eshkol_warn("AutodiffCodegen::gradient requires AST codegen - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::jacobian(const eshkol_operations_t* op) {
    // This requires AST codegen callback - use fallback for now
    eshkol_warn("AutodiffCodegen::jacobian requires AST codegen - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::createTape() {
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return llvm::ConstantPointerNull::get(ctx_.ptrType());

    llvm::Function* alloc_tape = mem_.getArenaAllocateTape();
    if (!alloc_tape) return llvm::ConstantPointerNull::get(ctx_.ptrType());

    return ctx_.builder().CreateCall(alloc_tape, {arena_ptr});
}

void AutodiffCodegen::backpropagate(llvm::Value* tape, llvm::Value* output_node) {
    // TODO: Full backprop implementation is complex - keep in llvm_codegen for now
    eshkol_warn("AutodiffCodegen::backpropagate not yet fully implemented");
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
