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
#include <llvm/IR/Intrinsics.h>

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

// ===== DUAL NUMBER MATH OPERATIONS =====
// These implement chain rule for various math functions

// Helper: Get or declare math function
llvm::Function* AutodiffCodegen::getMathFunc(const std::string& name) {
    // Check function table first
    if (function_table_) {
        auto it = function_table_->find(name);
        if (it != function_table_->end()) {
            return it->second;
        }
    }

    // Check if already declared in module
    llvm::Function* func = ctx_.module().getFunction(name);
    if (func) return func;

    // Declare the function
    std::vector<llvm::Type*> args = {ctx_.doubleType()};
    // pow takes 2 args
    if (name == "pow") {
        args.push_back(ctx_.doubleType());
    }

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.doubleType(), args, false);
    func = llvm::Function::Create(
        func_type, llvm::Function::ExternalLinkage, name, &ctx_.module());

    // Add to function table if available
    if (function_table_) {
        (*function_table_)[name] = func;
    }

    return func;
}

// Sine: sin(a, a') = (sin(a), a' * cos(a))
// Chain rule: d/dx[sin(f(x))] = cos(f(x)) * f'(x)
llvm::Value* AutodiffCodegen::dualSin(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* sin_func = getMathFunc("sin");
    llvm::Function* cos_func = getMathFunc("cos");
    if (!sin_func || !cos_func) return nullptr;

    // Value: sin(a)
    llvm::Value* value = ctx_.builder().CreateCall(sin_func, {a});

    // Derivative: a' * cos(a)
    llvm::Value* cos_a = ctx_.builder().CreateCall(cos_func, {a});
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, cos_a);

    return createDualNumber(value, deriv);
}

// Cosine: cos(a, a') = (cos(a), -a' * sin(a))
// Chain rule: d/dx[cos(f(x))] = -sin(f(x)) * f'(x)
llvm::Value* AutodiffCodegen::dualCos(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* sin_func = getMathFunc("sin");
    llvm::Function* cos_func = getMathFunc("cos");
    if (!sin_func || !cos_func) return nullptr;

    // Value: cos(a)
    llvm::Value* value = ctx_.builder().CreateCall(cos_func, {a});

    // Derivative: -a' * sin(a)
    llvm::Value* sin_a = ctx_.builder().CreateCall(sin_func, {a});
    llvm::Value* neg_sin_a = ctx_.builder().CreateFNeg(sin_a);
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, neg_sin_a);

    return createDualNumber(value, deriv);
}

// Exponential: exp(a, a') = (exp(a), a' * exp(a))
// Chain rule: d/dx[exp(f(x))] = exp(f(x)) * f'(x)
llvm::Value* AutodiffCodegen::dualExp(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* exp_func = getMathFunc("exp");
    if (!exp_func) return nullptr;

    // Value: exp(a)
    llvm::Value* exp_a = ctx_.builder().CreateCall(exp_func, {a});

    // Derivative: a' * exp(a)
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, exp_a);

    return createDualNumber(exp_a, deriv);
}

// Logarithm: log(a, a') = (log(a), a'/a)
// Chain rule: d/dx[log(f(x))] = f'(x)/f(x)
llvm::Value* AutodiffCodegen::dualLog(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* log_func = getMathFunc("log");
    if (!log_func) return nullptr;

    // Value: log(a)
    llvm::Value* value = ctx_.builder().CreateCall(log_func, {a});

    // Derivative: a' / a
    llvm::Value* deriv = ctx_.builder().CreateFDiv(a_prime, a);

    return createDualNumber(value, deriv);
}

// Tangent: tan(a, a') = (tan(a), a' * sec²(a)) = (tan(a), a' * (1 + tan²(a)))
// Chain rule: d/dx[tan(f(x))] = sec²(f(x)) * f'(x)
llvm::Value* AutodiffCodegen::dualTan(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* tan_func = getMathFunc("tan");
    if (!tan_func) return nullptr;

    // Value: tan(a)
    llvm::Value* tan_a = ctx_.builder().CreateCall(tan_func, {a});

    // Derivative: a' * (1 + tan²(a)) = a' * sec²(a)
    llvm::Value* tan_sq = ctx_.builder().CreateFMul(tan_a, tan_a);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* sec_sq = ctx_.builder().CreateFAdd(one, tan_sq);
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, sec_sq);

    return createDualNumber(tan_a, deriv);
}

// Square root: sqrt(a, a') = (sqrt(a), a' / (2 * sqrt(a)))
// Chain rule: d/dx[sqrt(f(x))] = f'(x) / (2 * sqrt(f(x)))
llvm::Value* AutodiffCodegen::dualSqrt(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* sqrt_func = getMathFunc("sqrt");
    if (!sqrt_func) return nullptr;

    // Value: sqrt(a)
    llvm::Value* sqrt_a = ctx_.builder().CreateCall(sqrt_func, {a});

    // Derivative: a' / (2 * sqrt(a))
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* two_sqrt_a = ctx_.builder().CreateFMul(two, sqrt_a);
    llvm::Value* deriv = ctx_.builder().CreateFDiv(a_prime, two_sqrt_a);

    return createDualNumber(sqrt_a, deriv);
}

// Power: (a, a')^(b, b') = (a^b, a^b * (b' * log(a) + b * a'/a))
// General power rule for both base and exponent being functions
llvm::Value* AutodiffCodegen::dualPow(llvm::Value* dual_base, llvm::Value* dual_exp) {
    if (!dual_base || !dual_exp) return nullptr;

    llvm::Value* a = getDualPrimal(dual_base);
    llvm::Value* a_prime = getDualTangent(dual_base);
    llvm::Value* b = getDualPrimal(dual_exp);
    llvm::Value* b_prime = getDualTangent(dual_exp);

    llvm::Function* pow_func = getMathFunc("pow");
    llvm::Function* log_func = getMathFunc("log");
    if (!pow_func || !log_func) return nullptr;

    // Value: a^b
    llvm::Value* value = ctx_.builder().CreateCall(pow_func, {a, b});

    // Derivative: a^b * (b' * log(a) + b * a'/a)
    llvm::Value* log_a = ctx_.builder().CreateCall(log_func, {a});
    llvm::Value* term1 = ctx_.builder().CreateFMul(b_prime, log_a);
    llvm::Value* term2 = ctx_.builder().CreateFMul(b, ctx_.builder().CreateFDiv(a_prime, a));
    llvm::Value* sum = ctx_.builder().CreateFAdd(term1, term2);
    llvm::Value* deriv = ctx_.builder().CreateFMul(value, sum);

    return createDualNumber(value, deriv);
}

// Arc sine: asin(a, a') = (asin(a), a' / sqrt(1 - a²))
// Chain rule: d/dx[asin(f(x))] = f'(x) / sqrt(1 - f(x)²)
llvm::Value* AutodiffCodegen::dualAsin(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* asin_func = getMathFunc("asin");
    llvm::Function* sqrt_func = getMathFunc("sqrt");
    if (!asin_func || !sqrt_func) return nullptr;

    // Value: asin(a)
    llvm::Value* value = ctx_.builder().CreateCall(asin_func, {a});

    // Derivative: a' / sqrt(1 - a²)
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* a_sq = ctx_.builder().CreateFMul(a, a);
    llvm::Value* one_minus_a_sq = ctx_.builder().CreateFSub(one, a_sq);
    llvm::Value* sqrt_term = ctx_.builder().CreateCall(sqrt_func, {one_minus_a_sq});
    llvm::Value* deriv = ctx_.builder().CreateFDiv(a_prime, sqrt_term);

    return createDualNumber(value, deriv);
}

// Arc cosine: acos(a, a') = (acos(a), -a' / sqrt(1 - a²))
// Chain rule: d/dx[acos(f(x))] = -f'(x) / sqrt(1 - f(x)²)
llvm::Value* AutodiffCodegen::dualAcos(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* acos_func = getMathFunc("acos");
    llvm::Function* sqrt_func = getMathFunc("sqrt");
    if (!acos_func || !sqrt_func) return nullptr;

    // Value: acos(a)
    llvm::Value* value = ctx_.builder().CreateCall(acos_func, {a});

    // Derivative: -a' / sqrt(1 - a²)
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* a_sq = ctx_.builder().CreateFMul(a, a);
    llvm::Value* one_minus_a_sq = ctx_.builder().CreateFSub(one, a_sq);
    llvm::Value* sqrt_term = ctx_.builder().CreateCall(sqrt_func, {one_minus_a_sq});
    llvm::Value* neg_a_prime = ctx_.builder().CreateFNeg(a_prime);
    llvm::Value* deriv = ctx_.builder().CreateFDiv(neg_a_prime, sqrt_term);

    return createDualNumber(value, deriv);
}

// Arc tangent: atan(a, a') = (atan(a), a' / (1 + a²))
// Chain rule: d/dx[atan(f(x))] = f'(x) / (1 + f(x)²)
llvm::Value* AutodiffCodegen::dualAtan(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* atan_func = getMathFunc("atan");
    if (!atan_func) return nullptr;

    // Value: atan(a)
    llvm::Value* value = ctx_.builder().CreateCall(atan_func, {a});

    // Derivative: a' / (1 + a²)
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* a_sq = ctx_.builder().CreateFMul(a, a);
    llvm::Value* one_plus_a_sq = ctx_.builder().CreateFAdd(one, a_sq);
    llvm::Value* deriv = ctx_.builder().CreateFDiv(a_prime, one_plus_a_sq);

    return createDualNumber(value, deriv);
}

// Absolute value: abs(a, a') = (|a|, a' * sign(a))
// Note: derivative is undefined at 0, we use 0 there
llvm::Value* AutodiffCodegen::dualAbs(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* fabs_func = getMathFunc("fabs");
    if (!fabs_func) return nullptr;

    // Value: |a|
    llvm::Value* abs_a = ctx_.builder().CreateCall(fabs_func, {a});

    // Sign: sign(a) = a / |a| (handles a=0 by returning 0)
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* is_zero = ctx_.builder().CreateFCmpOEQ(a, zero);
    llvm::Value* sign = ctx_.builder().CreateSelect(
        is_zero, zero, ctx_.builder().CreateFDiv(a, abs_a));

    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, sign);

    return createDualNumber(abs_a, deriv);
}

// Negation: -(a, a') = (-a, -a')
llvm::Value* AutodiffCodegen::dualNeg(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    // Value: -a
    llvm::Value* value = ctx_.builder().CreateFNeg(a);

    // Derivative: -a'
    llvm::Value* deriv = ctx_.builder().CreateFNeg(a_prime);

    return createDualNumber(value, deriv);
}

// Hyperbolic sine: sinh(a, a') = (sinh(a), a' * cosh(a))
// Chain rule: d/dx[sinh(f(x))] = cosh(f(x)) * f'(x)
llvm::Value* AutodiffCodegen::dualSinh(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* sinh_func = getMathFunc("sinh");
    llvm::Function* cosh_func = getMathFunc("cosh");
    if (!sinh_func || !cosh_func) return nullptr;

    llvm::Value* sinh_a = ctx_.builder().CreateCall(sinh_func, {a});
    llvm::Value* cosh_a = ctx_.builder().CreateCall(cosh_func, {a});
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, cosh_a);

    return createDualNumber(sinh_a, deriv);
}

// Hyperbolic cosine: cosh(a, a') = (cosh(a), a' * sinh(a))
// Chain rule: d/dx[cosh(f(x))] = sinh(f(x)) * f'(x)
llvm::Value* AutodiffCodegen::dualCosh(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* sinh_func = getMathFunc("sinh");
    llvm::Function* cosh_func = getMathFunc("cosh");
    if (!sinh_func || !cosh_func) return nullptr;

    llvm::Value* cosh_a = ctx_.builder().CreateCall(cosh_func, {a});
    llvm::Value* sinh_a = ctx_.builder().CreateCall(sinh_func, {a});
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, sinh_a);

    return createDualNumber(cosh_a, deriv);
}

// Hyperbolic tangent: tanh(a, a') = (tanh(a), a' * (1 - tanh²(a))) = (tanh(a), a' * sech²(a))
// Chain rule: d/dx[tanh(f(x))] = sech²(f(x)) * f'(x)
llvm::Value* AutodiffCodegen::dualTanh(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* tanh_func = getMathFunc("tanh");
    if (!tanh_func) return nullptr;

    // Value: tanh(a)
    llvm::Value* tanh_a = ctx_.builder().CreateCall(tanh_func, {a});

    // Derivative: a' * (1 - tanh²(a))
    llvm::Value* tanh_sq = ctx_.builder().CreateFMul(tanh_a, tanh_a);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* sech_sq = ctx_.builder().CreateFSub(one, tanh_sq);
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, sech_sq);

    return createDualNumber(tanh_a, deriv);
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

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
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

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
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

llvm::Value* AutodiffCodegen::recordADNodeUnary(uint32_t op_type, llvm::Value* input_node) {
    if (!input_node) return nullptr;

    llvm::StructType* ad_type = ctx_.adNodeType();

    // Load value from input node
    llvm::Value* input_value_ptr = ctx_.builder().CreateStructGEP(ad_type, input_node, 1);
    llvm::Value* input_value = ctx_.builder().CreateLoad(ctx_.doubleType(), input_value_ptr);

    // Compute result value based on operation
    llvm::Value* result_value = nullptr;
    switch (op_type) {
        case 6: // AD_NODE_SIN
            {
                llvm::Function* sin_func = getMathFunc("sin");
                if (!sin_func) return nullptr;
                result_value = ctx_.builder().CreateCall(sin_func, {input_value});
            }
            break;
        case 7: // AD_NODE_COS
            {
                llvm::Function* cos_func = getMathFunc("cos");
                if (!cos_func) return nullptr;
                result_value = ctx_.builder().CreateCall(cos_func, {input_value});
            }
            break;
        case 8: // AD_NODE_EXP
            {
                llvm::Function* exp_func = getMathFunc("exp");
                if (!exp_func) return nullptr;
                result_value = ctx_.builder().CreateCall(exp_func, {input_value});
            }
            break;
        case 9: // AD_NODE_LOG
            {
                llvm::Function* log_func = getMathFunc("log");
                if (!log_func) return nullptr;
                result_value = ctx_.builder().CreateCall(log_func, {input_value});
            }
            break;
        case 11: // AD_NODE_NEG
            result_value = ctx_.builder().CreateFNeg(input_value);
            break;
        case 12: // AD_NODE_ABS
            {
                llvm::Function* fabs_func = getMathFunc("fabs");
                if (!fabs_func) return nullptr;
                result_value = ctx_.builder().CreateCall(fabs_func, {input_value});
            }
            break;
        default:
            eshkol_warn("Unknown unary AD operation type: %u", op_type);
            return nullptr;
    }

    // Allocate new AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
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

    // Set input1 pointer (for unary operations)
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    ctx_.builder().CreateStore(input_node, input1_ptr);

    // Set input2 to null (unary operation has only one input)
    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input2_ptr);

    // Set node ID
    llvm::Value* id_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++), id_ptr);

    // Add to tape - use global runtime tape pointer with null check
    llvm::GlobalVariable* tape_global = ctx_.currentAdTape();
    if (tape_global) {
        llvm::Value* tape_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), tape_global);
        llvm::Value* tape_not_null = ctx_.builder().CreateICmpNE(
            ctx_.builder().CreatePtrToInt(tape_ptr, ctx_.int64Type()),
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* add_to_tape = llvm::BasicBlock::Create(ctx_.context(), "add_unary_to_tape", current_func);
        llvm::BasicBlock* skip_tape = llvm::BasicBlock::Create(ctx_.context(), "skip_unary_tape", current_func);

        ctx_.builder().CreateCondBr(tape_not_null, add_to_tape, skip_tape);

        ctx_.builder().SetInsertPoint(add_to_tape);
        llvm::Function* add_node_func = mem_.getArenaTapeAddNode();
        if (add_node_func) {
            ctx_.builder().CreateCall(add_node_func, {tape_ptr, node_ptr});
        }
        ctx_.builder().CreateBr(skip_tape);

        ctx_.builder().SetInsertPoint(skip_tape);
    }

    return node_ptr;
}

llvm::Value* AutodiffCodegen::createADVariable(llvm::Value* value, size_t var_index) {
    if (!value) return nullptr;

    // Convert value to double if needed
    if (value->getType()->isIntegerTy()) {
        value = ctx_.builder().CreateSIToFP(value, ctx_.doubleType());
    }

    // Allocate AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
    if (!alloc_func) return nullptr;

    llvm::Value* node_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr});

    llvm::StructType* ad_type = ctx_.adNodeType();

    // Set type = AD_NODE_VARIABLE (1)
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 0);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int32Type(), 1), type_ptr);

    // Set value
    llvm::Value* value_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1);
    ctx_.builder().CreateStore(value, value_ptr);

    // Initialize gradient = 0.0 (will be set during backward pass)
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    ctx_.builder().CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), grad_ptr);

    // Set input pointers to null (variables have no inputs)
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input1_ptr);

    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input2_ptr);

    // Set node ID
    llvm::Value* id_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++), id_ptr);

    // Variables are NOT added to tape (they're stored separately)

    return node_ptr;
}

llvm::Value* AutodiffCodegen::loadNodeInput1(llvm::Value* node_ptr) {
    if (!node_ptr) return nullptr;
    llvm::StructType* ad_type = ctx_.adNodeType();
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    return ctx_.builder().CreateLoad(ctx_.ptrType(), input1_ptr);
}

llvm::Value* AutodiffCodegen::loadNodeInput2(llvm::Value* node_ptr) {
    if (!node_ptr) return nullptr;
    llvm::StructType* ad_type = ctx_.adNodeType();
    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    return ctx_.builder().CreateLoad(ctx_.ptrType(), input2_ptr);
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

llvm::Value* AutodiffCodegen::derivative(const eshkol_operations_t* op) {
    using namespace llvm;

    if (!op->derivative_op.function) {
        eshkol_error("Invalid derivative operation - missing function");
        return nullptr;
    }

    // Higher-order form: (derivative f) - delegate back to main codegen
    if (!op->derivative_op.point) {
        // Return nullptr to signal that main codegen should handle this
        return nullptr;
    }

    if (!resolve_lambda_callback_ || !codegen_ast_callback_) {
        eshkol_error("derivative: Required callbacks not set");
        return tagged_.packNull();
    }

    eshkol_info("Computing derivative using forward-mode AD (dual numbers)");

    // Get the function to differentiate
    Value* func = resolve_lambda_callback_(op->derivative_op.function, 1, callback_context_);
    if (!func) {
        // RUNTIME FUNCTION PARAMETER FIX: Return nullptr without error
        // to let the fallback codegenDerivative handle runtime function parameters
        eshkol_debug("AutodiffCodegen::derivative: function not resolved, falling back to main codegen");
        return nullptr;
    }

    Function* func_ptr = dyn_cast<Function>(func);
    if (!func_ptr) {
        // RUNTIME FUNCTION PARAMETER FIX: Return nullptr without error
        // to let the fallback handle runtime closures
        eshkol_debug("AutodiffCodegen::derivative: not a Function*, falling back to main codegen");
        return nullptr;
    }

    // Get evaluation point - must be a scalar double
    Value* x = codegen_ast_callback_(op->derivative_op.point, callback_context_);
    if (!x) {
        eshkol_error("Failed to evaluate derivative point");
        return nullptr;
    }

    // Convert x to double if it's an integer or tagged_value
    if (x->getType()->isIntegerTy()) {
        x = ctx_.builder().CreateSIToFP(x, ctx_.doubleType());
    } else if (x->getType() == ctx_.taggedValueType()) {
        // Handle computed values that return tagged_value_t
        x = tagged_.unpackDouble(x);
    } else if (!x->getType()->isDoubleTy()) {
        eshkol_error("derivative point must be numeric (int64 or double)");
        return nullptr;
    }

    // Create dual number with seed derivative = 1.0
    Value* one = ConstantFP::get(ctx_.doubleType(), 1.0);
    Value* x_dual = createDualNumber(x, one);

    // Pack dual number into tagged_value for function call
    Value* x_dual_tagged = packDualToTagged(x_dual);

    // Build arguments for derivative lambda call
    std::vector<Value*> deriv_call_args = {x_dual_tagged};

    // CLOSURE FIX: Load captures from STORAGE
    FunctionType* deriv_func_type = func_ptr->getFunctionType();
    if (deriv_func_type->getNumParams() > 1) {
        size_t num_captures = deriv_func_type->getNumParams() - 1;
        std::string lambda_name = func_ptr->getName().str();

        // REPL MODE: Get capture names from registry
        std::vector<std::string> capture_names;
        if (repl_mode_enabled_ && *repl_mode_enabled_ && repl_mutex_ && repl_lambda_captures_) {
            std::lock_guard<std::mutex> lock(*repl_mutex_);
            auto captures_it = repl_lambda_captures_->find(lambda_name);
            if (captures_it != repl_lambda_captures_->end()) {
                capture_names = captures_it->second;
            }
        }

        for (size_t i = 0; i < num_captures; i++) {
            std::string var_name;
            if (i < capture_names.size()) {
                var_name = capture_names[i];
            } else {
                // Fallback to LLVM parameter names (for non-REPL mode)
                auto arg_it = func_ptr->arg_begin();
                std::advance(arg_it, i + 1);  // Skip first parameter
                if (arg_it != func_ptr->arg_end()) {
                    var_name = arg_it->getName().str();
                    if (var_name.find("captured_") == 0) {
                        var_name = var_name.substr(9);
                    }
                }
            }

            std::string capture_key = lambda_name + "_capture_" + var_name;

            // First try local symbol tables with capture_key
            bool found = false;
            Value* storage = nullptr;

            if (global_symbol_table_) {
                auto it = global_symbol_table_->find(capture_key);
                if (it != global_symbol_table_->end()) {
                    found = true;
                    storage = it->second;
                }
            }
            if (!found && symbol_table_) {
                auto it = symbol_table_->find(capture_key);
                if (it != symbol_table_->end()) {
                    found = true;
                    storage = it->second;
                }
            }

            // INNER FUNCTION FIX: If capture_key not found, try plain variable name
            // This handles lambdas inside functions where captures are function parameters
            // (not stored as GlobalVariables with _capture_ keys)
            // Also handles top-level global variables that are captured by lambdas
            if (!found && global_symbol_table_) {
                auto it = global_symbol_table_->find(var_name);
                if (it != global_symbol_table_->end()) {
                    found = true;
                    storage = it->second;
                    eshkol_debug("Derivative: found capture '%s' via global variable name", var_name.c_str());
                }
            }
            if (!found && symbol_table_) {
                auto it = symbol_table_->find(var_name);
                if (it != symbol_table_->end()) {
                    found = true;
                    storage = it->second;
                    eshkol_debug("Derivative: found capture '%s' via plain variable name", var_name.c_str());
                }
            }

            // REPL MODE: Try creating external declaration for capture global
            if (!found && repl_mode_enabled_ && *repl_mode_enabled_ &&
                repl_mutex_ && repl_symbol_addresses_) {
                std::lock_guard<std::mutex> lock(*repl_mutex_);
                auto sym_it = repl_symbol_addresses_->find(capture_key);
                if (sym_it != repl_symbol_addresses_->end()) {
                    GlobalVariable* capture_global = ctx_.module().getGlobalVariable(capture_key);
                    if (!capture_global) {
                        capture_global = new GlobalVariable(
                            ctx_.module(),
                            ctx_.taggedValueType(),
                            false,
                            GlobalValue::ExternalLinkage,
                            nullptr,
                            capture_key
                        );
                    }
                    // MUTABLE CAPTURE FIX: Pass pointer instead of loaded value
                    deriv_call_args.push_back(capture_global);
                    continue;
                }
            }

            if (found && storage) {
                // MUTABLE CAPTURE FIX: Pass pointer instead of loaded value
                // But first check if we need to create a temporary storage for pass-by-value
                if (storage->getType()->isPointerTy()) {
                    deriv_call_args.push_back(storage);
                } else {
                    // Value is not a pointer - need to create temporary storage
                    // This happens when capturing function parameters (pass-by-value)
                    Value* temp_storage = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "capture_temp");
                    ctx_.builder().CreateStore(storage, temp_storage);
                    deriv_call_args.push_back(temp_storage);
                    eshkol_debug("Derivative: created temp storage for capture '%s'", var_name.c_str());
                }
            } else {
                // MUTABLE CAPTURE FIX: Push null pointer instead of packed zero
                deriv_call_args.push_back(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())));
                eshkol_warn("Derivative: capture '%s' not found, using null pointer", var_name.c_str());
            }
        }
    }

    // Call function with dual number input and captures
    Value* result_tagged = ctx_.builder().CreateCall(func_ptr, deriv_call_args);

    // Unpack result from tagged_value
    Value* result_dual = unpackDualFromTagged(result_tagged);

    // Extract derivative component from result
    Value* derivative_val = getDualTangent(result_dual);

    eshkol_debug("Derivative operator: extracted derivative component");

    // Return derivative as tagged_value for consistent handling in arithmetic
    return tagged_.packDouble(derivative_val);
}

llvm::Value* AutodiffCodegen::hessian(const eshkol_operations_t* op) {
    // Second-order derivatives - requires jacobian and gradient
    eshkol_warn("AutodiffCodegen::hessian requires AST codegen - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::divergence(const eshkol_operations_t* op) {
    // Trace of Jacobian - requires jacobian
    eshkol_warn("AutodiffCodegen::divergence requires AST codegen - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::curl(const eshkol_operations_t* op) {
    // Cross product of partials - requires jacobian
    eshkol_warn("AutodiffCodegen::curl requires AST codegen - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::laplacian(const eshkol_operations_t* op) {
    // Trace of Hessian - requires hessian
    eshkol_warn("AutodiffCodegen::laplacian requires AST codegen - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::directionalDerivative(const eshkol_operations_t* op) {
    // Gradient dot direction - requires gradient
    eshkol_warn("AutodiffCodegen::directionalDerivative requires AST codegen - using fallback");
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
    // CRITICAL: Add runtime null checks for placeholder functions
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    if (!current_func) {
        eshkol_error("Backward pass requires active function context");
        return;
    }

    // Create safety check blocks
    llvm::BasicBlock* check_validity = llvm::BasicBlock::Create(ctx_.context(), "backward_check_valid", current_func);
    llvm::BasicBlock* backward_valid = llvm::BasicBlock::Create(ctx_.context(), "backward_valid", current_func);
    llvm::BasicBlock* backward_skip = llvm::BasicBlock::Create(ctx_.context(), "backward_skip", current_func);

    ctx_.builder().CreateBr(check_validity);

    // Check if output node and tape are valid (not null)
    ctx_.builder().SetInsertPoint(check_validity);
    llvm::Value* output_int = ctx_.builder().CreatePtrToInt(output_node, ctx_.int64Type());
    llvm::Value* tape_int = ctx_.builder().CreatePtrToInt(tape, ctx_.int64Type());

    llvm::Value* output_valid = ctx_.builder().CreateICmpNE(output_int, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* tape_valid = ctx_.builder().CreateICmpNE(tape_int, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* both_valid = ctx_.builder().CreateAnd(output_valid, tape_valid);

    ctx_.builder().CreateCondBr(both_valid, backward_valid, backward_skip);

    ctx_.builder().SetInsertPoint(backward_valid);

    // Initialize output gradient = 1.0 (seed for backpropagation)
    storeNodeGradient(output_node, llvm::ConstantFP::get(ctx_.doubleType(), 1.0));

    // Get number of nodes in tape (runtime value, not compile-time constant)
    llvm::Function* get_count_func = mem_.getArenaTapeGetNodeCount();
    if (!get_count_func) {
        eshkol_error("Backward pass: arena_tape_get_node_count not available");
        ctx_.builder().CreateBr(backward_skip);
        ctx_.builder().SetInsertPoint(backward_skip);
        return;
    }
    llvm::Value* num_nodes = ctx_.builder().CreateCall(get_count_func, {tape});

    // Allocate loop counter for backward traversal (MUST iterate in reverse order)
    llvm::Value* counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "backward_counter");
    if (!counter) {
        eshkol_error("Failed to allocate backward pass counter");
        ctx_.builder().CreateBr(backward_skip);
        ctx_.builder().SetInsertPoint(backward_skip);
        return;
    }

    // Initialize counter = num_nodes (start at end, decrement to 0)
    ctx_.builder().CreateStore(num_nodes, counter);

    // Create loop basic blocks (REQUIRED for LLVM IR structure)
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "backward_loop_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "backward_loop_body", current_func);
    llvm::BasicBlock* check_node = llvm::BasicBlock::Create(ctx_.context(), "backward_check_node", current_func);
    llvm::BasicBlock* propagate_block = llvm::BasicBlock::Create(ctx_.context(), "backward_propagate", current_func);
    llvm::BasicBlock* skip_node = llvm::BasicBlock::Create(ctx_.context(), "backward_skip_node", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "backward_loop_exit", current_func);

    // Jump to loop condition
    ctx_.builder().CreateBr(loop_cond);

    // Loop condition: while (counter > 0)
    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* counter_val = ctx_.builder().CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* counter_gt_zero = ctx_.builder().CreateICmpUGT(counter_val,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(counter_gt_zero, loop_body, loop_exit);

    // Loop body: Process node at index (counter - 1)
    ctx_.builder().SetInsertPoint(loop_body);

    // Decrement counter FIRST to get 0-based index
    llvm::Value* counter_minus_1 = ctx_.builder().CreateSub(counter_val,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(counter_minus_1, counter);

    // Get node at index using arena_tape_get_node (may return nullptr)
    llvm::Function* get_node_func = mem_.getArenaTapeGetNode();
    llvm::Value* node_ptr = ctx_.builder().CreateCall(get_node_func,
        {tape, counter_minus_1});

    // Null check before propagation (defensive programming)
    ctx_.builder().CreateBr(check_node);

    ctx_.builder().SetInsertPoint(check_node);
    llvm::Value* node_is_null = ctx_.builder().CreateICmpEQ(node_ptr,
        llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx_.context())));
    ctx_.builder().CreateCondBr(node_is_null, skip_node, propagate_block);

    // Propagate gradient for this node
    ctx_.builder().SetInsertPoint(propagate_block);
    propagateGradient(node_ptr);
    ctx_.builder().CreateBr(skip_node);

    // Skip or continue to next iteration
    ctx_.builder().SetInsertPoint(skip_node);
    ctx_.builder().CreateBr(loop_cond);

    // Loop exit: backward pass complete
    ctx_.builder().SetInsertPoint(loop_exit);
    ctx_.builder().CreateBr(backward_skip);

    // Skip block: exit point for null/invalid inputs
    ctx_.builder().SetInsertPoint(backward_skip);

    eshkol_debug("Completed backward pass through computational graph");
}

void AutodiffCodegen::propagateGradient(llvm::Value* node_ptr) {
    if (!node_ptr) return;

    llvm::StructType* ad_node_type = ctx_.adNodeType();

    // Load node type
    llvm::Value* type_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 0);
    llvm::Value* node_type = ctx_.builder().CreateLoad(ctx_.int32Type(), type_ptr);

    // Load node gradient
    llvm::Value* node_grad = loadNodeGradient(node_ptr);

    // Load input pointers
    llvm::Value* input1 = loadNodeInput1(node_ptr);
    llvm::Value* input2 = loadNodeInput2(node_ptr);

    // Branch on operation type to apply correct gradient rules
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Create blocks for each operation type
    llvm::BasicBlock* add_block = llvm::BasicBlock::Create(ctx_.context(), "grad_add", current_func);
    llvm::BasicBlock* sub_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sub", current_func);
    llvm::BasicBlock* mul_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mul", current_func);
    llvm::BasicBlock* div_block = llvm::BasicBlock::Create(ctx_.context(), "grad_div", current_func);
    llvm::BasicBlock* sin_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sin", current_func);
    llvm::BasicBlock* cos_block = llvm::BasicBlock::Create(ctx_.context(), "grad_cos", current_func);
    llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "grad_done", current_func);

    // Switch on node type
    // For ADD (type=2): gradient flows equally to both inputs
    llvm::Value* is_add = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 2));

    llvm::BasicBlock* check_sub = llvm::BasicBlock::Create(ctx_.context(), "check_sub", current_func);
    ctx_.builder().CreateCondBr(is_add, add_block, check_sub);

    // ADD: dL/dx = dL/dz * 1, dL/dy = dL/dz * 1
    ctx_.builder().SetInsertPoint(add_block);
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad);
    ctx_.builder().CreateBr(done_block);

    // Check for SUB
    ctx_.builder().SetInsertPoint(check_sub);
    llvm::Value* is_sub = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 3));
    llvm::BasicBlock* check_mul = llvm::BasicBlock::Create(ctx_.context(), "check_mul", current_func);
    ctx_.builder().CreateCondBr(is_sub, sub_block, check_mul);

    // SUB: dL/dx = dL/dz * 1, dL/dy = dL/dz * (-1)
    ctx_.builder().SetInsertPoint(sub_block);
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) {
        llvm::Value* neg_grad = ctx_.builder().CreateFNeg(node_grad);
        accumulateGradient(input2, neg_grad);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for MUL
    ctx_.builder().SetInsertPoint(check_mul);
    llvm::Value* is_mul = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 4));
    llvm::BasicBlock* check_div = llvm::BasicBlock::Create(ctx_.context(), "check_div", current_func);
    ctx_.builder().CreateCondBr(is_mul, mul_block, check_div);

    // MUL: dL/dx = dL/dz * y, dL/dy = dL/dz * x
    ctx_.builder().SetInsertPoint(mul_block);
    if (input1 && input2) {
        llvm::Value* input1_val = loadNodeValue(input1);
        llvm::Value* input2_val = loadNodeValue(input2);

        llvm::Value* grad_input1 = ctx_.builder().CreateFMul(node_grad, input2_val);
        llvm::Value* grad_input2 = ctx_.builder().CreateFMul(node_grad, input1_val);

        // DOUBLE BACKWARD: Track degree when multiplying by variable value
        llvm::GlobalVariable* inner_var_node_ptr = ctx_.innerVarNodePtr();
        llvm::GlobalVariable* gradient_x_degree = ctx_.gradientXDegree();

        if (inner_var_node_ptr && gradient_x_degree) {
            // Load stored variable node and its value for comparison
            llvm::Value* stored_var_node = ctx_.builder().CreateLoad(llvm::PointerType::getUnqual(ctx_.context()), inner_var_node_ptr);
            llvm::Value* stored_var_is_valid = ctx_.builder().CreateICmpNE(stored_var_node,
                llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx_.context())));

            // Only track degree if we have a stored variable node
            llvm::BasicBlock* track_degree_bb = llvm::BasicBlock::Create(ctx_.context(), "track_degree", current_func);
            llvm::BasicBlock* skip_degree_bb = llvm::BasicBlock::Create(ctx_.context(), "skip_degree", current_func);
            llvm::BasicBlock* after_degree_bb = llvm::BasicBlock::Create(ctx_.context(), "after_degree", current_func);

            ctx_.builder().CreateCondBr(stored_var_is_valid, track_degree_bb, skip_degree_bb);

            ctx_.builder().SetInsertPoint(track_degree_bb);
            llvm::Value* var_val = loadNodeValue(stored_var_node);

            // Check node TYPE as well as value to avoid false positives
            llvm::Value* input1_type_ptr = ctx_.builder().CreateStructGEP(ad_node_type, input1, 0);
            llvm::Value* input1_type = ctx_.builder().CreateLoad(ctx_.int32Type(), input1_type_ptr);
            llvm::Value* input1_is_var_type = ctx_.builder().CreateICmpEQ(input1_type, llvm::ConstantInt::get(ctx_.int32Type(), 1));

            llvm::Value* input2_type_ptr = ctx_.builder().CreateStructGEP(ad_node_type, input2, 0);
            llvm::Value* input2_type = ctx_.builder().CreateLoad(ctx_.int32Type(), input2_type_ptr);
            llvm::Value* input2_is_var_type = ctx_.builder().CreateICmpEQ(input2_type, llvm::ConstantInt::get(ctx_.int32Type(), 1));

            // Check if input2 is the variable (by value comparison with tolerance AND type check)
            llvm::Value* diff2 = ctx_.builder().CreateFSub(input2_val, var_val);
            llvm::Function* fabs_intrinsic = llvm::Intrinsic::getDeclaration(&ctx_.module(), llvm::Intrinsic::fabs, {ctx_.doubleType()});
            llvm::Value* abs_diff2 = ctx_.builder().CreateCall(fabs_intrinsic, {diff2});
            llvm::Value* val_matches_2 = ctx_.builder().CreateFCmpOLT(abs_diff2, llvm::ConstantFP::get(ctx_.doubleType(), 1e-10));
            llvm::Value* is_var2 = ctx_.builder().CreateAnd(val_matches_2, input2_is_var_type);

            // Check if input1 is the variable
            llvm::Value* diff1 = ctx_.builder().CreateFSub(input1_val, var_val);
            llvm::Value* abs_diff1 = ctx_.builder().CreateCall(fabs_intrinsic, {diff1});
            llvm::Value* val_matches_1 = ctx_.builder().CreateFCmpOLT(abs_diff1, llvm::ConstantFP::get(ctx_.doubleType(), 1e-10));
            llvm::Value* is_var1 = ctx_.builder().CreateAnd(val_matches_1, input1_is_var_type);

            // Count how many times we multiply by variable value
            llvm::Value* current_degree = ctx_.builder().CreateLoad(ctx_.int64Type(), gradient_x_degree);
            llvm::Value* inc2 = ctx_.builder().CreateSelect(is_var2,
                llvm::ConstantInt::get(ctx_.int64Type(), 1),
                llvm::ConstantInt::get(ctx_.int64Type(), 0));
            llvm::Value* inc1 = ctx_.builder().CreateSelect(is_var1,
                llvm::ConstantInt::get(ctx_.int64Type(), 1),
                llvm::ConstantInt::get(ctx_.int64Type(), 0));
            llvm::Value* total_inc = ctx_.builder().CreateAdd(inc1, inc2);
            llvm::Value* new_degree = ctx_.builder().CreateAdd(current_degree, total_inc);
            ctx_.builder().CreateStore(new_degree, gradient_x_degree);
            ctx_.builder().CreateBr(after_degree_bb);

            ctx_.builder().SetInsertPoint(skip_degree_bb);
            ctx_.builder().CreateBr(after_degree_bb);

            ctx_.builder().SetInsertPoint(after_degree_bb);
        }

        accumulateGradient(input1, grad_input1);
        accumulateGradient(input2, grad_input2);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for DIV
    ctx_.builder().SetInsertPoint(check_div);
    llvm::Value* is_div = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 5));
    llvm::BasicBlock* check_sin = llvm::BasicBlock::Create(ctx_.context(), "check_sin", current_func);
    ctx_.builder().CreateCondBr(is_div, div_block, check_sin);

    // DIV: dL/dx = dL/dz / y, dL/dy = dL/dz * (-x/y²)
    ctx_.builder().SetInsertPoint(div_block);
    if (input1 && input2) {
        llvm::Value* input1_val = loadNodeValue(input1);
        llvm::Value* input2_val = loadNodeValue(input2);

        llvm::Value* grad_input1 = ctx_.builder().CreateFDiv(node_grad, input2_val);

        llvm::Value* y_squared = ctx_.builder().CreateFMul(input2_val, input2_val);
        llvm::Value* neg_x_over_y2 = ctx_.builder().CreateFDiv(ctx_.builder().CreateFNeg(input1_val), y_squared);
        llvm::Value* grad_input2 = ctx_.builder().CreateFMul(node_grad, neg_x_over_y2);

        accumulateGradient(input1, grad_input1);
        accumulateGradient(input2, grad_input2);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SIN
    ctx_.builder().SetInsertPoint(check_sin);
    llvm::Value* is_sin = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 6));
    llvm::BasicBlock* check_cos = llvm::BasicBlock::Create(ctx_.context(), "check_cos", current_func);
    ctx_.builder().CreateCondBr(is_sin, sin_block, check_cos);

    // SIN: dL/dx = dL/dz * cos(x)
    ctx_.builder().SetInsertPoint(sin_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* cos_func = getMathFunc("cos");
        if (cos_func) {
            llvm::Value* cos_val = ctx_.builder().CreateCall(cos_func, {input_val});
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, cos_val);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Check for COS
    ctx_.builder().SetInsertPoint(check_cos);
    llvm::Value* is_cos = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 7));
    ctx_.builder().CreateCondBr(is_cos, cos_block, done_block); // Default to done if unknown

    // COS: dL/dx = dL/dz * (-sin(x))
    ctx_.builder().SetInsertPoint(cos_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* sin_func = getMathFunc("sin");
        if (sin_func) {
            llvm::Value* sin_val = ctx_.builder().CreateCall(sin_func, {input_val});
            llvm::Value* neg_sin = ctx_.builder().CreateFNeg(sin_val);
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, neg_sin);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Done
    ctx_.builder().SetInsertPoint(done_block);
}

// ===== TAPE MANAGEMENT (Nested Gradient Support) =====
// These functions enable arbitrary-depth nested gradient computations
// by saving/restoring the tape context on a stack.

void AutodiffCodegen::pushTapeContext(llvm::Value* new_tape) {
    if (!new_tape) return;

    llvm::GlobalVariable* ad_tape_depth = ctx_.adTapeDepth();
    llvm::GlobalVariable* ad_tape_stack = ctx_.adTapeStack();
    llvm::GlobalVariable* current_ad_tape = ctx_.currentAdTape();
    llvm::GlobalVariable* ad_mode_active = ctx_.adModeActive();

    if (!ad_tape_depth || !ad_tape_stack || !current_ad_tape || !ad_mode_active) {
        eshkol_warn("pushTapeContext: AD globals not initialized");
        return;
    }

    // Load current depth
    llvm::Value* depth = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_tape_depth);

    // Save current tape to stack[depth]
    llvm::Value* current_tape = ctx_.builder().CreateLoad(ctx_.ptrType(), current_ad_tape);
    llvm::ArrayType* stack_type = llvm::ArrayType::get(ctx_.ptrType(), CodegenContext::MAX_TAPE_DEPTH);
    llvm::Value* slot_ptr = ctx_.builder().CreateGEP(stack_type, ad_tape_stack,
        {llvm::ConstantInt::get(ctx_.int64Type(), 0), depth});
    ctx_.builder().CreateStore(current_tape, slot_ptr);

    // Increment depth
    llvm::Value* new_depth = ctx_.builder().CreateAdd(depth, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(new_depth, ad_tape_depth);

    // Set new tape as current
    ctx_.builder().CreateStore(new_tape, current_ad_tape);

    // Set AD mode active
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int1Type(), 1), ad_mode_active);
}

void AutodiffCodegen::popTapeContext() {
    llvm::GlobalVariable* ad_tape_depth = ctx_.adTapeDepth();
    llvm::GlobalVariable* ad_tape_stack = ctx_.adTapeStack();
    llvm::GlobalVariable* current_ad_tape = ctx_.currentAdTape();
    llvm::GlobalVariable* ad_mode_active = ctx_.adModeActive();

    if (!ad_tape_depth || !ad_tape_stack || !current_ad_tape || !ad_mode_active) {
        eshkol_warn("popTapeContext: AD globals not initialized");
        return;
    }

    // Load current depth
    llvm::Value* depth = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_tape_depth);

    // Decrement depth
    llvm::Value* new_depth = ctx_.builder().CreateSub(depth, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(new_depth, ad_tape_depth);

    // Restore tape from stack[new_depth]
    llvm::ArrayType* stack_type = llvm::ArrayType::get(ctx_.ptrType(), CodegenContext::MAX_TAPE_DEPTH);
    llvm::Value* slot_ptr = ctx_.builder().CreateGEP(stack_type, ad_tape_stack,
        {llvm::ConstantInt::get(ctx_.int64Type(), 0), new_depth});
    llvm::Value* saved_tape = ctx_.builder().CreateLoad(ctx_.ptrType(), slot_ptr);

    // Set restored tape as current
    ctx_.builder().CreateStore(saved_tape, current_ad_tape);

    // Set AD mode based on whether we still have active tapes
    // If new_depth == 0, we're exiting the outermost gradient
    llvm::Value* still_active = ctx_.builder().CreateICmpNE(new_depth,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateStore(still_active, ad_mode_active);
}

llvm::Value* AutodiffCodegen::getOuterTape() {
    llvm::GlobalVariable* ad_tape_depth = ctx_.adTapeDepth();
    llvm::GlobalVariable* ad_tape_stack = ctx_.adTapeStack();

    if (!ad_tape_depth || !ad_tape_stack) {
        return llvm::ConstantPointerNull::get(ctx_.ptrType());
    }

    llvm::Value* depth = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_tape_depth);

    // Check if nested (depth > 0)
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* nested_bb = llvm::BasicBlock::Create(ctx_.context(), "outer_tape_nested", current_func);
    llvm::BasicBlock* not_nested_bb = llvm::BasicBlock::Create(ctx_.context(), "outer_tape_not_nested", current_func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "outer_tape_merge", current_func);

    llvm::Value* is_nested = ctx_.builder().CreateICmpUGT(depth, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(is_nested, nested_bb, not_nested_bb);

    // Nested: get tape from stack[depth-1]
    ctx_.builder().SetInsertPoint(nested_bb);
    llvm::Value* outer_idx = ctx_.builder().CreateSub(depth, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::ArrayType* stack_type = llvm::ArrayType::get(ctx_.ptrType(), CodegenContext::MAX_TAPE_DEPTH);
    llvm::Value* outer_slot = ctx_.builder().CreateGEP(stack_type, ad_tape_stack,
        {llvm::ConstantInt::get(ctx_.int64Type(), 0), outer_idx});
    llvm::Value* outer_tape = ctx_.builder().CreateLoad(ctx_.ptrType(), outer_slot);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* nested_exit = ctx_.builder().GetInsertBlock();

    // Not nested: return null
    ctx_.builder().SetInsertPoint(not_nested_bb);
    llvm::Value* null_tape = llvm::ConstantPointerNull::get(ctx_.ptrType());
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* not_nested_exit = ctx_.builder().GetInsertBlock();

    // Merge
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.ptrType(), 2, "outer_tape_result");
    result->addIncoming(outer_tape, nested_exit);
    result->addIncoming(null_tape, not_nested_exit);

    return result;
}

llvm::Value* AutodiffCodegen::isNested() {
    llvm::GlobalVariable* ad_tape_depth = ctx_.adTapeDepth();
    if (!ad_tape_depth) {
        return llvm::ConstantInt::get(ctx_.int1Type(), 0);
    }

    llvm::Value* depth = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_tape_depth);
    return ctx_.builder().CreateICmpUGT(depth, llvm::ConstantInt::get(ctx_.int64Type(), 0));
}

// ===== TAPE-SPECIFIC AD NODE OPERATIONS =====
// Used for double backward - record operations on outer tape

llvm::Value* AutodiffCodegen::createADConstantOnTape(llvm::Value* tape_ptr, llvm::Value* value) {
    if (!tape_ptr || !value) return nullptr;

    // Convert value to double if needed
    if (value->getType()->isIntegerTy()) {
        value = ctx_.builder().CreateSIToFP(value, ctx_.doubleType());
    }

    // Allocate AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
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

    // Set input pointers to null (constant has no inputs)
    llvm::Value* input1_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input1_ptr);

    llvm::Value* input2_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4);
    ctx_.builder().CreateStore(llvm::ConstantPointerNull::get(ctx_.ptrType()), input2_ptr);

    // Set node ID
    llvm::Value* id_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++), id_ptr);

    // Add to specified tape
    llvm::Function* add_node_func = mem_.getArenaTapeAddNode();
    if (add_node_func) {
        ctx_.builder().CreateCall(add_node_func, {tape_ptr, node_ptr});
    }

    return node_ptr;
}

llvm::Value* AutodiffCodegen::recordADNodeBinaryOnTape(llvm::Value* tape_ptr, uint32_t op_type,
                                                        llvm::Value* left_node, llvm::Value* right_node) {
    if (!tape_ptr || !left_node || !right_node) return nullptr;

    // Load values from input nodes
    llvm::Value* left_value = loadNodeValue(left_node);
    llvm::Value* right_value = loadNodeValue(right_node);

    if (!left_value || !right_value) return nullptr;

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

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
    if (!alloc_func) return nullptr;

    llvm::Value* node_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr});

    llvm::StructType* ad_type = ctx_.adNodeType();

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

    // Add to specified tape
    llvm::Function* add_node_func = mem_.getArenaTapeAddNode();
    if (add_node_func) {
        ctx_.builder().CreateCall(add_node_func, {tape_ptr, node_ptr});
    }

    return node_ptr;
}

// ===== AD NODE HELPERS =====
// Access fields of AD nodes

llvm::Value* AutodiffCodegen::loadNodeValue(llvm::Value* node_ptr) {
    if (!node_ptr) return nullptr;
    llvm::StructType* ad_type = ctx_.adNodeType();
    llvm::Value* value_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1);
    return ctx_.builder().CreateLoad(ctx_.doubleType(), value_ptr);
}

llvm::Value* AutodiffCodegen::loadNodeGradient(llvm::Value* node_ptr) {
    if (!node_ptr) return nullptr;
    llvm::StructType* ad_type = ctx_.adNodeType();
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    return ctx_.builder().CreateLoad(ctx_.doubleType(), grad_ptr);
}

void AutodiffCodegen::storeNodeGradient(llvm::Value* node_ptr, llvm::Value* gradient) {
    if (!node_ptr || !gradient) return;
    llvm::StructType* ad_type = ctx_.adNodeType();
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    ctx_.builder().CreateStore(gradient, grad_ptr);
}

void AutodiffCodegen::accumulateGradient(llvm::Value* node_ptr, llvm::Value* gradient_to_add) {
    if (!node_ptr || !gradient_to_add) return;  // Compile-time check

    // RUNTIME NULL CHECK: Generate LLVM IR to check if node_ptr is null at runtime
    // This is critical because AD constant/variable nodes have null input pointers
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    llvm::BasicBlock* accumulate_block = llvm::BasicBlock::Create(ctx_.context(), "accumulate_grad", current_func);
    llvm::BasicBlock* skip_accumulate = llvm::BasicBlock::Create(ctx_.context(), "skip_accumulate", current_func);
    llvm::BasicBlock* merge_accumulate = llvm::BasicBlock::Create(ctx_.context(), "merge_accumulate", current_func);

    // Check if node_ptr is null at runtime
    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(node_ptr,
        llvm::ConstantPointerNull::get(ctx_.ptrType()));
    ctx_.builder().CreateCondBr(is_null, skip_accumulate, accumulate_block);

    // Non-null path: perform gradient accumulation
    ctx_.builder().SetInsertPoint(accumulate_block);
    llvm::StructType* ad_type = ctx_.adNodeType();
    llvm::Value* grad_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2);
    llvm::Value* current_grad = ctx_.builder().CreateLoad(ctx_.doubleType(), grad_ptr);
    llvm::Value* new_grad = ctx_.builder().CreateFAdd(current_grad, gradient_to_add);
    ctx_.builder().CreateStore(new_grad, grad_ptr);
    ctx_.builder().CreateBr(merge_accumulate);

    // Null path: skip accumulation
    ctx_.builder().SetInsertPoint(skip_accumulate);
    ctx_.builder().CreateBr(merge_accumulate);

    // Merge point: continue from here
    ctx_.builder().SetInsertPoint(merge_accumulate);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
