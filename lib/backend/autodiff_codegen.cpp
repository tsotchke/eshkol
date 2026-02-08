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
#include <llvm/Config/llvm-config.h>

// LLVM VERSION COMPATIBILITY
#if LLVM_VERSION_MAJOR >= 18
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getOrInsertDeclaration(mod, id, types)
#else
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getDeclaration(mod, id, types)
#endif

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
        case 10: // AD_NODE_POW
            {
                llvm::Function* pow_func = getMathFunc("pow");
                if (!pow_func) return nullptr;
                result_value = ctx_.builder().CreateCall(pow_func, {left_value, right_value});
            }
            break;
        case 44: // AD_NODE_MAX
            {
                // max(a, b) = a if a > b else b
                llvm::Value* cmp = ctx_.builder().CreateFCmpOGT(left_value, right_value);
                result_value = ctx_.builder().CreateSelect(cmp, left_value, right_value);
            }
            break;
        case 45: // AD_NODE_MIN
            {
                // min(a, b) = a if a < b else b
                llvm::Value* cmp = ctx_.builder().CreateFCmpOLT(left_value, right_value);
                result_value = ctx_.builder().CreateSelect(cmp, left_value, right_value);
            }
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

        // === Activation functions (12-18) ===
        case 12: // AD_NODE_RELU
            {
                // relu(x) = max(0, x)
                llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
                llvm::Value* cmp = ctx_.builder().CreateFCmpOGT(input_value, zero);
                result_value = ctx_.builder().CreateSelect(cmp, input_value, zero);
            }
            break;
        case 13: // AD_NODE_SIGMOID
            {
                // sigmoid(x) = 1 / (1 + exp(-x))
                llvm::Function* exp_func = getMathFunc("exp");
                if (!exp_func) return nullptr;
                llvm::Value* neg_x = ctx_.builder().CreateFNeg(input_value);
                llvm::Value* exp_neg_x = ctx_.builder().CreateCall(exp_func, {neg_x});
                llvm::Value* one_plus = ctx_.builder().CreateFAdd(
                    llvm::ConstantFP::get(ctx_.doubleType(), 1.0), exp_neg_x);
                result_value = ctx_.builder().CreateFDiv(
                    llvm::ConstantFP::get(ctx_.doubleType(), 1.0), one_plus);
            }
            break;
        case 15: // AD_NODE_TANH
            {
                // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
                llvm::Function* exp_func = getMathFunc("exp");
                if (!exp_func) return nullptr;
                llvm::Value* two_x = ctx_.builder().CreateFMul(
                    llvm::ConstantFP::get(ctx_.doubleType(), 2.0), input_value);
                llvm::Value* exp_2x = ctx_.builder().CreateCall(exp_func, {two_x});
                llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
                llvm::Value* numer = ctx_.builder().CreateFSub(exp_2x, one);
                llvm::Value* denom = ctx_.builder().CreateFAdd(exp_2x, one);
                result_value = ctx_.builder().CreateFDiv(numer, denom);
            }
            break;
        case 16: // AD_NODE_GELU
            {
                // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                llvm::Function* exp_func = getMathFunc("exp");
                if (!exp_func) return nullptr;
                llvm::Value* sqrt_2_pi = llvm::ConstantFP::get(ctx_.doubleType(), 0.7978845608028654);
                llvm::Value* coeff = llvm::ConstantFP::get(ctx_.doubleType(), 0.044715);
                llvm::Value* x_cubed = ctx_.builder().CreateFMul(input_value,
                    ctx_.builder().CreateFMul(input_value, input_value));
                llvm::Value* inner = ctx_.builder().CreateFMul(sqrt_2_pi,
                    ctx_.builder().CreateFAdd(input_value,
                        ctx_.builder().CreateFMul(coeff, x_cubed)));
                // tanh via exp
                llvm::Value* two_inner = ctx_.builder().CreateFMul(
                    llvm::ConstantFP::get(ctx_.doubleType(), 2.0), inner);
                llvm::Value* exp_2x = ctx_.builder().CreateCall(exp_func, {two_inner});
                llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
                llvm::Value* tanh_val = ctx_.builder().CreateFDiv(
                    ctx_.builder().CreateFSub(exp_2x, one),
                    ctx_.builder().CreateFAdd(exp_2x, one));
                result_value = ctx_.builder().CreateFMul(
                    llvm::ConstantFP::get(ctx_.doubleType(), 0.5),
                    ctx_.builder().CreateFMul(input_value,
                        ctx_.builder().CreateFAdd(one, tanh_val)));
            }
            break;
        case 17: // AD_NODE_LEAKY_RELU
            {
                // leaky_relu(x) = x > 0 ? x : 0.01 * x
                llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
                llvm::Value* leak = llvm::ConstantFP::get(ctx_.doubleType(), 0.01);
                llvm::Value* cmp = ctx_.builder().CreateFCmpOGT(input_value, zero);
                llvm::Value* leaked = ctx_.builder().CreateFMul(leak, input_value);
                result_value = ctx_.builder().CreateSelect(cmp, input_value, leaked);
            }
            break;
        case 18: // AD_NODE_SILU (Swish)
            {
                // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                llvm::Function* exp_func = getMathFunc("exp");
                if (!exp_func) return nullptr;
                llvm::Value* neg_x = ctx_.builder().CreateFNeg(input_value);
                llvm::Value* exp_neg_x = ctx_.builder().CreateCall(exp_func, {neg_x});
                llvm::Value* one_plus = ctx_.builder().CreateFAdd(
                    llvm::ConstantFP::get(ctx_.doubleType(), 1.0), exp_neg_x);
                result_value = ctx_.builder().CreateFDiv(input_value, one_plus);
            }
            break;

        // === Additional math operations (41-44) ===
        case 41: // AD_NODE_SQRT
            {
                llvm::Function* sqrt_func = getMathFunc("sqrt");
                if (!sqrt_func) return nullptr;
                result_value = ctx_.builder().CreateCall(sqrt_func, {input_value});
            }
            break;
        case 42: // AD_NODE_ABS
            {
                llvm::Function* fabs_func = getMathFunc("fabs");
                if (!fabs_func) return nullptr;
                result_value = ctx_.builder().CreateCall(fabs_func, {input_value});
            }
            break;
        case 43: // AD_NODE_SQUARE
            {
                result_value = ctx_.builder().CreateFMul(input_value, input_value);
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

// === Tensor AD Node Recording ===

llvm::Value* AutodiffCodegen::recordADNodeTensor(
    uint32_t op_type,
    llvm::Value* input1, llvm::Value* input2,
    llvm::Value* input3, llvm::Value* input4,
    llvm::Value* tensor_result,
    llvm::Value* saved_tensors, llvm::Value* num_saved,
    llvm::Value* shape, llvm::Value* ndim)
{
    llvm::StructType* ad_type = ctx_.adNodeType();
    auto null_ptr = llvm::ConstantPointerNull::get(ctx_.ptrType());
    auto zero_i64 = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    auto zero_f64 = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    // Allocate new AD node
    llvm::Value* arena_ptr = getArenaPtr();
    if (!arena_ptr) return nullptr;

    llvm::Function* alloc_func = mem_.getArenaAllocateAdNodeWithHeader();
    if (!alloc_func) return nullptr;

    llvm::Value* node_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr});

    // Field 0: type
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int32Type(), op_type),
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 0));

    // Field 1: value = 0.0 (tensor ops use tensor_value instead)
    ctx_.builder().CreateStore(zero_f64,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 1));

    // Field 2: gradient = 0.0
    ctx_.builder().CreateStore(zero_f64,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 2));

    // Field 3: input1
    ctx_.builder().CreateStore(input1 ? input1 : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 3));

    // Field 4: input2
    ctx_.builder().CreateStore(input2 ? input2 : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 4));

    // Field 5: id
    ctx_.builder().CreateStore(
        llvm::ConstantInt::get(ctx_.int64Type(), next_node_id_++),
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 5));

    // Field 6: tensor_value
    ctx_.builder().CreateStore(tensor_result ? tensor_result : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 6));

    // Field 7: tensor_gradient = null (allocated during backward)
    ctx_.builder().CreateStore(null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 7));

    // Field 8: input3
    ctx_.builder().CreateStore(input3 ? input3 : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 8));

    // Field 9: input4
    ctx_.builder().CreateStore(input4 ? input4 : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 9));

    // Field 10: saved_tensors
    ctx_.builder().CreateStore(saved_tensors ? saved_tensors : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 10));

    // Field 11: num_saved
    ctx_.builder().CreateStore(num_saved ? num_saved : zero_i64,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 11));

    // Field 12: params (zero-initialized array, caller sets specific values)
    llvm::ArrayType* params_type = llvm::ArrayType::get(ctx_.int64Type(), 6);
    llvm::Value* params_ptr = ctx_.builder().CreateStructGEP(ad_type, node_ptr, 12);
    for (unsigned i = 0; i < 6; i++) {
        llvm::Value* elem_ptr = ctx_.builder().CreateConstGEP2_32(params_type, params_ptr, 0, i);
        ctx_.builder().CreateStore(zero_i64, elem_ptr);
    }

    // Field 13: shape
    ctx_.builder().CreateStore(shape ? shape : null_ptr,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 13));

    // Field 14: ndim
    ctx_.builder().CreateStore(ndim ? ndim : zero_i64,
        ctx_.builder().CreateStructGEP(ad_type, node_ptr, 14));

    // Add to tape
    llvm::GlobalVariable* tape_global = ctx_.currentAdTape();
    if (tape_global) {
        llvm::Value* tape_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), tape_global);
        llvm::Value* tape_not_null = ctx_.builder().CreateICmpNE(
            ctx_.builder().CreatePtrToInt(tape_ptr, ctx_.int64Type()),
            llvm::ConstantInt::get(ctx_.int64Type(), 0));

        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* add_block = llvm::BasicBlock::Create(
            ctx_.context(), "add_tensor_to_tape", current_func);
        llvm::BasicBlock* skip_block = llvm::BasicBlock::Create(
            ctx_.context(), "skip_tensor_tape", current_func);

        ctx_.builder().CreateCondBr(tape_not_null, add_block, skip_block);

        ctx_.builder().SetInsertPoint(add_block);
        llvm::Function* add_node_func = mem_.getArenaTapeAddNode();
        if (add_node_func) {
            ctx_.builder().CreateCall(add_node_func, {tape_ptr, node_ptr});
        }
        ctx_.builder().CreateBr(skip_block);

        ctx_.builder().SetInsertPoint(skip_block);
    }

    return node_ptr;
}

// === Tensor Gradient Accumulation ===

void AutodiffCodegen::accumulateTensorGradient(
    llvm::Value* node_ptr, llvm::Value* grad_tensor, llvm::Value* num_elements)
{
    if (!node_ptr || !grad_tensor || !num_elements) return;

    // Declare the runtime function if not already declared
    llvm::Module* mod = ctx_.builder().GetInsertBlock()->getModule();
    llvm::Function* accum_func = mod->getFunction("eshkol_accumulate_tensor_grad");
    if (!accum_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(
            ctx_.voidType(),
            {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()},
            false);
        accum_func = llvm::Function::Create(
            ft, llvm::Function::ExternalLinkage,
            "eshkol_accumulate_tensor_grad", mod);
    }

    // Null check on node_ptr
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* accum_block = llvm::BasicBlock::Create(
        ctx_.context(), "accum_tensor_grad", current_func);
    llvm::BasicBlock* skip_block = llvm::BasicBlock::Create(
        ctx_.context(), "skip_tensor_grad", current_func);

    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(
        node_ptr, llvm::ConstantPointerNull::get(ctx_.ptrType()));
    ctx_.builder().CreateCondBr(is_null, skip_block, accum_block);

    ctx_.builder().SetInsertPoint(accum_block);
    ctx_.builder().CreateCall(accum_func, {node_ptr, grad_tensor, num_elements});
    ctx_.builder().CreateBr(skip_block);

    ctx_.builder().SetInsertPoint(skip_block);
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
    // Gradient computation is handled directly in llvm_codegen.cpp via the gradient operator.
    // This method exists for API completeness but is not used in the current codegen pipeline.
    eshkol_error("AutodiffCodegen::gradient called directly - use gradient operator in main codegen instead");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::jacobian(const eshkol_operations_t* op) {
    // Jacobian computation is handled directly in llvm_codegen.cpp via the jacobian operator.
    // This method exists for API completeness but is not used in the current codegen pipeline.
    eshkol_error("AutodiffCodegen::jacobian called directly - use jacobian operator in main codegen instead");
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

    // Seed tensor gradient: if the output node has tensor_value set,
    // allocate an all-ones tensor gradient (dL/dL = 1 for every element).
    // This is a no-op for scalar nodes (tensor_value == NULL).
    {
        llvm::FunctionType* seed_type = llvm::FunctionType::get(
            ctx_.voidType(), {ctx_.ptrType()}, false);
        llvm::FunctionCallee seed_fn = ctx_.module().getOrInsertFunction(
            "eshkol_seed_tensor_gradient", seed_type);
        ctx_.builder().CreateCall(seed_fn, {output_node});
    }

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

    // Create done block first (referenced by tensor dispatch path below)
    llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "grad_done", current_func);

    // === TENSOR GRADIENT FAST PATH ===
    // If tensor_gradient (field 7) is non-null, the node was recorded as a tensor
    // operation via recordADNodeTensor. Dispatch to the C runtime backward function
    // which reads saved_tensors, params, shape/ndim and calls the appropriate
    // eshkol_backward_* function (conv2d, matmul, attention, etc.)
    {
        llvm::Value* tg_field_ptr = ctx_.builder().CreateStructGEP(
            ad_node_type, node_ptr, TypeSystem::AD_NODE_TENSOR_GRADIENT_IDX);
        llvm::Value* tg_val = ctx_.builder().CreateLoad(ctx_.ptrType(), tg_field_ptr);
        llvm::Value* has_tensor = ctx_.builder().CreateICmpNE(tg_val,
            llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(ctx_.context())));

        llvm::BasicBlock* tensor_dispatch_bb = llvm::BasicBlock::Create(
            ctx_.context(), "tensor_backward_dispatch", current_func);
        llvm::BasicBlock* scalar_dispatch_bb = llvm::BasicBlock::Create(
            ctx_.context(), "scalar_backward_dispatch", current_func);

        ctx_.builder().CreateCondBr(has_tensor, tensor_dispatch_bb, scalar_dispatch_bb);

        // Tensor path: call runtime dispatcher that handles all tensor ops
        ctx_.builder().SetInsertPoint(tensor_dispatch_bb);
        llvm::FunctionType* dispatch_type = llvm::FunctionType::get(
            ctx_.voidType(), {ctx_.ptrType()}, false);
        llvm::FunctionCallee dispatch_fn = ctx_.module().getOrInsertFunction(
            "eshkol_tensor_backward_dispatch", dispatch_type);
        ctx_.builder().CreateCall(dispatch_fn, {node_ptr});
        ctx_.builder().CreateBr(done_block);

        // Continue with scalar dispatch for nodes without tensor gradients
        ctx_.builder().SetInsertPoint(scalar_dispatch_bb);
    }

    // Create blocks for each scalar operation type
    llvm::BasicBlock* add_block = llvm::BasicBlock::Create(ctx_.context(), "grad_add", current_func);
    llvm::BasicBlock* sub_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sub", current_func);
    llvm::BasicBlock* mul_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mul", current_func);
    llvm::BasicBlock* div_block = llvm::BasicBlock::Create(ctx_.context(), "grad_div", current_func);
    llvm::BasicBlock* sin_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sin", current_func);
    llvm::BasicBlock* cos_block = llvm::BasicBlock::Create(ctx_.context(), "grad_cos", current_func);

    // Switch on node type (scalar backward passes)
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
            llvm::Function* fabs_intrinsic = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::fabs, {ctx_.doubleType()});
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
    llvm::BasicBlock* check_exp = llvm::BasicBlock::Create(ctx_.context(), "check_exp", current_func);
    ctx_.builder().CreateCondBr(is_cos, cos_block, check_exp);

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

    // Create blocks for additional operations
    llvm::BasicBlock* exp_block = llvm::BasicBlock::Create(ctx_.context(), "grad_exp", current_func);
    llvm::BasicBlock* log_block = llvm::BasicBlock::Create(ctx_.context(), "grad_log", current_func);
    llvm::BasicBlock* pow_block = llvm::BasicBlock::Create(ctx_.context(), "grad_pow", current_func);
    llvm::BasicBlock* neg_block = llvm::BasicBlock::Create(ctx_.context(), "grad_neg", current_func);
    llvm::BasicBlock* relu_block = llvm::BasicBlock::Create(ctx_.context(), "grad_relu", current_func);
    llvm::BasicBlock* sigmoid_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sigmoid", current_func);
    llvm::BasicBlock* softmax_block = llvm::BasicBlock::Create(ctx_.context(), "grad_softmax", current_func);
    llvm::BasicBlock* tanh_block = llvm::BasicBlock::Create(ctx_.context(), "grad_tanh", current_func);
    llvm::BasicBlock* gelu_block = llvm::BasicBlock::Create(ctx_.context(), "grad_gelu", current_func);
    llvm::BasicBlock* leaky_relu_block = llvm::BasicBlock::Create(ctx_.context(), "grad_leaky_relu", current_func);
    llvm::BasicBlock* silu_block = llvm::BasicBlock::Create(ctx_.context(), "grad_silu", current_func);
    llvm::BasicBlock* matmul_block = llvm::BasicBlock::Create(ctx_.context(), "grad_matmul", current_func);
    llvm::BasicBlock* sum_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sum", current_func);
    llvm::BasicBlock* mean_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mean", current_func);
    llvm::BasicBlock* sqrt_block = llvm::BasicBlock::Create(ctx_.context(), "grad_sqrt", current_func);
    llvm::BasicBlock* abs_block = llvm::BasicBlock::Create(ctx_.context(), "grad_abs", current_func);
    llvm::BasicBlock* square_block = llvm::BasicBlock::Create(ctx_.context(), "grad_square", current_func);
    llvm::BasicBlock* max_block = llvm::BasicBlock::Create(ctx_.context(), "grad_max", current_func);
    llvm::BasicBlock* min_block = llvm::BasicBlock::Create(ctx_.context(), "grad_min", current_func);

    // Check for EXP (type=8)
    ctx_.builder().SetInsertPoint(check_exp);
    llvm::Value* is_exp = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 8));
    llvm::BasicBlock* check_log = llvm::BasicBlock::Create(ctx_.context(), "check_log", current_func);
    ctx_.builder().CreateCondBr(is_exp, exp_block, check_log);

    // EXP: dL/dx = dL/dz * exp(x)
    ctx_.builder().SetInsertPoint(exp_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* exp_func = getMathFunc("exp");
        if (exp_func) {
            llvm::Value* exp_val = ctx_.builder().CreateCall(exp_func, {input_val});
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, exp_val);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Check for LOG (type=9)
    ctx_.builder().SetInsertPoint(check_log);
    llvm::Value* is_log = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 9));
    llvm::BasicBlock* check_pow = llvm::BasicBlock::Create(ctx_.context(), "check_pow", current_func);
    ctx_.builder().CreateCondBr(is_log, log_block, check_pow);

    // LOG: dL/dx = dL/dz / x
    ctx_.builder().SetInsertPoint(log_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, input_val);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for POW (type=10)
    ctx_.builder().SetInsertPoint(check_pow);
    llvm::Value* is_pow = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 10));
    llvm::BasicBlock* check_neg = llvm::BasicBlock::Create(ctx_.context(), "check_neg", current_func);
    ctx_.builder().CreateCondBr(is_pow, pow_block, check_neg);

    // POW: dL/dx = dL/dz * y * x^(y-1), dL/dy = dL/dz * x^y * ln(x)
    ctx_.builder().SetInsertPoint(pow_block);
    if (input1 && input2) {
        llvm::Value* base_val = loadNodeValue(input1);
        llvm::Value* exp_val = loadNodeValue(input2);

        llvm::Function* pow_func = getMathFunc("pow");
        llvm::Function* log_func = getMathFunc("log");

        if (pow_func && log_func) {
            // Gradient w.r.t. base: y * x^(y-1) = y * x^y / x
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* exp_minus_1 = ctx_.builder().CreateFSub(exp_val, one);
            llvm::Value* pow_val = ctx_.builder().CreateCall(pow_func, {base_val, exp_minus_1});
            llvm::Value* base_deriv = ctx_.builder().CreateFMul(exp_val, pow_val);
            llvm::Value* grad_base = ctx_.builder().CreateFMul(node_grad, base_deriv);
            accumulateGradient(input1, grad_base);

            // Gradient w.r.t. exponent: x^y * ln(x)
            llvm::Value* pow_full = ctx_.builder().CreateCall(pow_func, {base_val, exp_val});
            llvm::Value* log_base = ctx_.builder().CreateCall(log_func, {base_val});
            llvm::Value* exp_deriv = ctx_.builder().CreateFMul(pow_full, log_base);
            llvm::Value* grad_exp = ctx_.builder().CreateFMul(node_grad, exp_deriv);
            accumulateGradient(input2, grad_exp);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Check for NEG (type=11)
    ctx_.builder().SetInsertPoint(check_neg);
    llvm::Value* is_neg = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 11));
    llvm::BasicBlock* check_relu = llvm::BasicBlock::Create(ctx_.context(), "check_relu", current_func);
    ctx_.builder().CreateCondBr(is_neg, neg_block, check_relu);

    // NEG: dL/dx = -dL/dz
    ctx_.builder().SetInsertPoint(neg_block);
    if (input1) {
        llvm::Value* neg_grad = ctx_.builder().CreateFNeg(node_grad);
        accumulateGradient(input1, neg_grad);
    }
    ctx_.builder().CreateBr(done_block);

    // === ML ACTIVATION GRADIENTS ===

    // Check for RELU (type=12)
    ctx_.builder().SetInsertPoint(check_relu);
    llvm::Value* is_relu = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 12));
    llvm::BasicBlock* check_sigmoid = llvm::BasicBlock::Create(ctx_.context(), "check_sigmoid", current_func);
    ctx_.builder().CreateCondBr(is_relu, relu_block, check_sigmoid);

    // RELU: dL/dx = dL/dz * (x > 0 ? 1 : 0)
    ctx_.builder().SetInsertPoint(relu_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(input_val, zero);
        llvm::Value* local_grad = ctx_.builder().CreateSelect(is_positive, one, zero);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, local_grad);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SIGMOID (type=13)
    ctx_.builder().SetInsertPoint(check_sigmoid);
    llvm::Value* is_sigmoid = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 13));
    llvm::BasicBlock* check_softmax = llvm::BasicBlock::Create(ctx_.context(), "check_softmax", current_func);
    ctx_.builder().CreateCondBr(is_sigmoid, sigmoid_block, check_softmax);

    // SIGMOID: dL/dx = dL/dz * σ(x) * (1 - σ(x))
    // Note: We can use the node's output value which is σ(x)
    ctx_.builder().SetInsertPoint(sigmoid_block);
    if (input1) {
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* sigma_x = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* one_minus_sigma = ctx_.builder().CreateFSub(one, sigma_x);
        llvm::Value* sigma_deriv = ctx_.builder().CreateFMul(sigma_x, one_minus_sigma);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, sigma_deriv);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SOFTMAX (type=14) - Softmax Jacobian: diag(s) - s*s^T
    // Backward: grad_input = s * (grad - dot(grad, s))
    ctx_.builder().SetInsertPoint(check_softmax);
    llvm::Value* is_softmax = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 14));
    llvm::BasicBlock* check_tanh = llvm::BasicBlock::Create(ctx_.context(), "check_tanh", current_func);
    ctx_.builder().CreateCondBr(is_softmax, softmax_block, check_tanh);

    // SOFTMAX backward: grad_input = s * (grad - dot(grad, s))
    // where s = softmax(x) is the forward output stored in the node value.
    // Derivation: Jacobian of softmax is diag(s) - s*s^T,
    // so grad_input_i = sum_j (s_i * delta_ij - s_i * s_j) * grad_j
    //                 = s_i * grad_i - s_i * sum_j(s_j * grad_j)
    //                 = s_i * (grad_i - dot(grad, s))
    ctx_.builder().SetInsertPoint(softmax_block);
    if (input1) {
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* s = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        // dot(grad, s) - for scalar case this is just grad * s
        llvm::Value* dot_grad_s = ctx_.builder().CreateFMul(node_grad, s);
        // grad - dot(grad, s)
        llvm::Value* grad_minus_dot = ctx_.builder().CreateFSub(node_grad, dot_grad_s);
        // s * (grad - dot(grad, s))
        llvm::Value* grad_input = ctx_.builder().CreateFMul(s, grad_minus_dot);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for TANH (type=15)
    ctx_.builder().SetInsertPoint(check_tanh);
    llvm::Value* is_tanh = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 15));
    llvm::BasicBlock* check_gelu = llvm::BasicBlock::Create(ctx_.context(), "check_gelu", current_func);
    ctx_.builder().CreateCondBr(is_tanh, tanh_block, check_gelu);

    // TANH: dL/dx = dL/dz * (1 - tanh²(x))
    ctx_.builder().SetInsertPoint(tanh_block);
    if (input1) {
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* tanh_x = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        llvm::Value* tanh_sq = ctx_.builder().CreateFMul(tanh_x, tanh_x);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* tanh_deriv = ctx_.builder().CreateFSub(one, tanh_sq);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, tanh_deriv);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for GELU (type=16)
    ctx_.builder().SetInsertPoint(check_gelu);
    llvm::Value* is_gelu = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 16));
    llvm::BasicBlock* check_leaky_relu = llvm::BasicBlock::Create(ctx_.context(), "check_leaky_relu", current_func);
    ctx_.builder().CreateCondBr(is_gelu, gelu_block, check_leaky_relu);

    // GELU: approximate derivative using sigmoid approximation
    // gelu(x) ≈ x * σ(1.702x), so gelu'(x) ≈ σ(1.702x) + 1.702x * σ(1.702x) * (1 - σ(1.702x))
    ctx_.builder().SetInsertPoint(gelu_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* exp_func = getMathFunc("exp");
        if (exp_func) {
            llvm::Value* coeff = llvm::ConstantFP::get(ctx_.doubleType(), 1.702);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

            llvm::Value* scaled_x = ctx_.builder().CreateFMul(coeff, input_val);
            llvm::Value* neg_scaled = ctx_.builder().CreateFNeg(scaled_x);
            llvm::Value* exp_neg = ctx_.builder().CreateCall(exp_func, {neg_scaled});
            llvm::Value* denom = ctx_.builder().CreateFAdd(one, exp_neg);
            llvm::Value* sigma = ctx_.builder().CreateFDiv(one, denom);

            llvm::Value* one_minus_sigma = ctx_.builder().CreateFSub(one, sigma);
            llvm::Value* sigma_deriv = ctx_.builder().CreateFMul(sigma, one_minus_sigma);
            llvm::Value* scaled_sigma_deriv = ctx_.builder().CreateFMul(coeff, sigma_deriv);
            llvm::Value* x_times_deriv = ctx_.builder().CreateFMul(input_val, scaled_sigma_deriv);
            llvm::Value* gelu_deriv = ctx_.builder().CreateFAdd(sigma, x_times_deriv);
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, gelu_deriv);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Check for LEAKY_RELU (type=17)
    ctx_.builder().SetInsertPoint(check_leaky_relu);
    llvm::Value* is_leaky_relu = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 17));
    llvm::BasicBlock* check_silu = llvm::BasicBlock::Create(ctx_.context(), "check_silu", current_func);
    ctx_.builder().CreateCondBr(is_leaky_relu, leaky_relu_block, check_silu);

    // LEAKY_RELU: dL/dx = dL/dz * (x > 0 ? 1 : α)
    // Default α = 0.01
    ctx_.builder().SetInsertPoint(leaky_relu_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* alpha = llvm::ConstantFP::get(ctx_.doubleType(), 0.01);
        llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(input_val, zero);
        llvm::Value* local_grad = ctx_.builder().CreateSelect(is_positive, one, alpha);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, local_grad);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SILU (type=18)
    ctx_.builder().SetInsertPoint(check_silu);
    llvm::Value* is_silu = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 18));
    llvm::BasicBlock* check_matmul = llvm::BasicBlock::Create(ctx_.context(), "check_matmul", current_func);
    ctx_.builder().CreateCondBr(is_silu, silu_block, check_matmul);

    // SILU (Swish): dL/dx = dL/dz * σ(x) * (1 + x * (1 - σ(x)))
    ctx_.builder().SetInsertPoint(silu_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* exp_func = getMathFunc("exp");
        if (exp_func) {
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* neg_x = ctx_.builder().CreateFNeg(input_val);
            llvm::Value* exp_neg = ctx_.builder().CreateCall(exp_func, {neg_x});
            llvm::Value* denom = ctx_.builder().CreateFAdd(one, exp_neg);
            llvm::Value* sigma = ctx_.builder().CreateFDiv(one, denom);

            llvm::Value* one_minus_sigma = ctx_.builder().CreateFSub(one, sigma);
            llvm::Value* x_times_one_minus = ctx_.builder().CreateFMul(input_val, one_minus_sigma);
            llvm::Value* one_plus_term = ctx_.builder().CreateFAdd(one, x_times_one_minus);
            llvm::Value* silu_deriv = ctx_.builder().CreateFMul(sigma, one_plus_term);
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, silu_deriv);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Check for MATMUL (type=24) - Tensor operation gradients are more complex
    // For now, we'll add a placeholder that can be extended for tensor autodiff
    ctx_.builder().SetInsertPoint(check_matmul);
    llvm::Value* is_matmul = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 24));
    llvm::BasicBlock* check_sum = llvm::BasicBlock::Create(ctx_.context(), "check_sum", current_func);
    ctx_.builder().CreateCondBr(is_matmul, matmul_block, check_sum);

    // MATMUL: For scalar case, acts like MUL
    // Full tensor matmul: dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
    // Placeholder for tensor autodiff integration
    ctx_.builder().SetInsertPoint(matmul_block);
    if (input1 && input2) {
        // Scalar approximation: treat as multiply
        llvm::Value* input1_val = loadNodeValue(input1);
        llvm::Value* input2_val = loadNodeValue(input2);
        llvm::Value* grad_input1 = ctx_.builder().CreateFMul(node_grad, input2_val);
        llvm::Value* grad_input2 = ctx_.builder().CreateFMul(node_grad, input1_val);
        accumulateGradient(input1, grad_input1);
        accumulateGradient(input2, grad_input2);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SUM (type=27)
    ctx_.builder().SetInsertPoint(check_sum);
    llvm::Value* is_sum = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 27));
    llvm::BasicBlock* check_mean = llvm::BasicBlock::Create(ctx_.context(), "check_mean", current_func);
    ctx_.builder().CreateCondBr(is_sum, sum_block, check_mean);

    // SUM: dL/dx_i = dL/dz for all i (gradient broadcasts to all elements)
    ctx_.builder().SetInsertPoint(sum_block);
    if (input1) {
        // For scalar, gradient passes through unchanged
        accumulateGradient(input1, node_grad);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for MEAN (type=28)
    ctx_.builder().SetInsertPoint(check_mean);
    llvm::Value* is_mean = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 28));
    llvm::BasicBlock* check_sqrt = llvm::BasicBlock::Create(ctx_.context(), "check_sqrt", current_func);
    ctx_.builder().CreateCondBr(is_mean, mean_block, check_sqrt);

    // MEAN: dL/dx_i = dL/dz / n for all i
    // For scalar, gradient passes through unchanged (n=1)
    ctx_.builder().SetInsertPoint(mean_block);
    if (input1) {
        accumulateGradient(input1, node_grad);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SQRT (type=41)
    ctx_.builder().SetInsertPoint(check_sqrt);
    llvm::Value* is_sqrt = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 41));
    llvm::BasicBlock* check_abs = llvm::BasicBlock::Create(ctx_.context(), "check_abs", current_func);
    ctx_.builder().CreateCondBr(is_sqrt, sqrt_block, check_abs);

    // SQRT: dL/dx = dL/dz * 0.5 / sqrt(x)
    // Note: We can use the node's output value which is sqrt(x)
    ctx_.builder().SetInsertPoint(sqrt_block);
    if (input1) {
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* sqrt_x = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);
        llvm::Value* sqrt_deriv = ctx_.builder().CreateFDiv(half, sqrt_x);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, sqrt_deriv);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for ABS (type=42)
    ctx_.builder().SetInsertPoint(check_abs);
    llvm::Value* is_abs = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 42));
    llvm::BasicBlock* check_square = llvm::BasicBlock::Create(ctx_.context(), "check_square", current_func);
    ctx_.builder().CreateCondBr(is_abs, abs_block, check_square);

    // ABS: dL/dx = dL/dz * sign(x)
    // sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
    ctx_.builder().SetInsertPoint(abs_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* pos_one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* neg_one = llvm::ConstantFP::get(ctx_.doubleType(), -1.0);
        llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(input_val, zero);
        llvm::Value* is_negative = ctx_.builder().CreateFCmpOLT(input_val, zero);
        // sign = is_positive ? 1.0 : (is_negative ? -1.0 : 0.0)
        llvm::Value* neg_or_zero = ctx_.builder().CreateSelect(is_negative, neg_one, zero);
        llvm::Value* sign_val = ctx_.builder().CreateSelect(is_positive, pos_one, neg_or_zero);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, sign_val);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for SQUARE (type=43)
    ctx_.builder().SetInsertPoint(check_square);
    llvm::Value* is_square = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 43));
    llvm::BasicBlock* check_max = llvm::BasicBlock::Create(ctx_.context(), "check_max", current_func);
    ctx_.builder().CreateCondBr(is_square, square_block, check_max);

    // SQUARE: dL/dx = dL/dz * 2x
    ctx_.builder().SetInsertPoint(square_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
        llvm::Value* two_x = ctx_.builder().CreateFMul(two, input_val);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, two_x);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for MAX (type=44)
    ctx_.builder().SetInsertPoint(check_max);
    llvm::Value* is_max = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 44));
    llvm::BasicBlock* check_min = llvm::BasicBlock::Create(ctx_.context(), "check_min", current_func);
    ctx_.builder().CreateCondBr(is_max, max_block, check_min);

    // MAX: dL/dx = dL/dz if x > y, dL/dy = dL/dz if y >= x
    // Gradient goes entirely to the larger input
    ctx_.builder().SetInsertPoint(max_block);
    if (input1 && input2) {
        llvm::Value* input1_val = loadNodeValue(input1);
        llvm::Value* input2_val = loadNodeValue(input2);
        llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* cmp = ctx_.builder().CreateFCmpOGT(input1_val, input2_val);
        llvm::Value* grad_input1 = ctx_.builder().CreateSelect(cmp, node_grad, zero);
        llvm::Value* grad_input2 = ctx_.builder().CreateSelect(cmp, zero, node_grad);
        accumulateGradient(input1, grad_input1);
        accumulateGradient(input2, grad_input2);
    }
    ctx_.builder().CreateBr(done_block);

    // Check for MIN (type=45)
    ctx_.builder().SetInsertPoint(check_min);
    llvm::Value* is_min = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 45));
    llvm::BasicBlock* check_conv2d = llvm::BasicBlock::Create(ctx_.context(), "check_conv2d", current_func);
    ctx_.builder().CreateCondBr(is_min, min_block, check_conv2d);

    // MIN: dL/dx = dL/dz if x < y, dL/dy = dL/dz if y <= x
    ctx_.builder().SetInsertPoint(min_block);
    if (input1 && input2) {
        llvm::Value* input1_val = loadNodeValue(input1);
        llvm::Value* input2_val = loadNodeValue(input2);
        llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
        llvm::Value* cmp = ctx_.builder().CreateFCmpOLT(input1_val, input2_val);
        llvm::Value* grad_input1 = ctx_.builder().CreateSelect(cmp, node_grad, zero);
        llvm::Value* grad_input2 = ctx_.builder().CreateSelect(cmp, zero, node_grad);
        accumulateGradient(input1, grad_input1);
        accumulateGradient(input2, grad_input2);
    }
    ctx_.builder().CreateBr(done_block);

    // ===== TENSOR OPERATION SCALAR FALLBACKS =====
    // These scalar approximations handle tensor op types (19-32) when
    // tensor_gradient is NULL (scalar mode). When tensor_gradient is set,
    // the tensor fast path above dispatches to eshkol_tensor_backward_dispatch()
    // which calls the proper runtime backward functions with full tensor data.

    // Create blocks for all new operation types
    llvm::BasicBlock* conv2d_block = llvm::BasicBlock::Create(ctx_.context(), "grad_conv2d", current_func);
    llvm::BasicBlock* maxpool_block = llvm::BasicBlock::Create(ctx_.context(), "grad_maxpool", current_func);
    llvm::BasicBlock* avgpool_block = llvm::BasicBlock::Create(ctx_.context(), "grad_avgpool", current_func);
    llvm::BasicBlock* batchnorm_block = llvm::BasicBlock::Create(ctx_.context(), "grad_batchnorm", current_func);
    llvm::BasicBlock* layernorm_block = llvm::BasicBlock::Create(ctx_.context(), "grad_layernorm", current_func);
    llvm::BasicBlock* transpose_block = llvm::BasicBlock::Create(ctx_.context(), "grad_transpose", current_func);
    llvm::BasicBlock* reshape_block = llvm::BasicBlock::Create(ctx_.context(), "grad_reshape", current_func);
    llvm::BasicBlock* attention_block = llvm::BasicBlock::Create(ctx_.context(), "grad_attention", current_func);
    llvm::BasicBlock* mha_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mha", current_func);
    llvm::BasicBlock* posenc_block = llvm::BasicBlock::Create(ctx_.context(), "grad_posenc", current_func);
    llvm::BasicBlock* embedding_block = llvm::BasicBlock::Create(ctx_.context(), "grad_embedding", current_func);
    llvm::BasicBlock* hyp_dist_block = llvm::BasicBlock::Create(ctx_.context(), "grad_hyp_dist", current_func);
    llvm::BasicBlock* poincare_exp_block = llvm::BasicBlock::Create(ctx_.context(), "grad_poincare_exp", current_func);
    llvm::BasicBlock* poincare_log_block = llvm::BasicBlock::Create(ctx_.context(), "grad_poincare_log", current_func);
    llvm::BasicBlock* tangent_proj_block = llvm::BasicBlock::Create(ctx_.context(), "grad_tangent_proj", current_func);
    llvm::BasicBlock* geodesic_attn_block = llvm::BasicBlock::Create(ctx_.context(), "grad_geodesic_attn", current_func);
    llvm::BasicBlock* mobius_add_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mobius_add", current_func);
    llvm::BasicBlock* mobius_matmul_block = llvm::BasicBlock::Create(ctx_.context(), "grad_mobius_matmul", current_func);
    llvm::BasicBlock* gyrovector_block = llvm::BasicBlock::Create(ctx_.context(), "grad_gyrovector", current_func);

    // --- CONV2D (type=19): dL/d_input ≈ grad (scalar approx) ---
    ctx_.builder().SetInsertPoint(check_conv2d);
    llvm::Value* is_conv2d = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 19));
    llvm::BasicBlock* check_maxpool = llvm::BasicBlock::Create(ctx_.context(), "check_maxpool", current_func);
    ctx_.builder().CreateCondBr(is_conv2d, conv2d_block, check_maxpool);

    ctx_.builder().SetInsertPoint(conv2d_block);
    // Conv2D backward: dL/d_input = conv_transpose(grad, kernel), dL/d_kernel = conv(input, grad)
    // Scalar approximation: pass gradient through to input (identity in scalar case)
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- MAXPOOL2D (type=20): gradient through max index ---
    ctx_.builder().SetInsertPoint(check_maxpool);
    llvm::Value* is_maxpool = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 20));
    llvm::BasicBlock* check_avgpool = llvm::BasicBlock::Create(ctx_.context(), "check_avgpool", current_func);
    ctx_.builder().CreateCondBr(is_maxpool, maxpool_block, check_avgpool);

    ctx_.builder().SetInsertPoint(maxpool_block);
    // MaxPool backward: gradient flows only through saved max indices
    // Scalar approximation: pass gradient to input (the max value was the input)
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- AVGPOOL2D (type=21): gradient divided by pool size ---
    ctx_.builder().SetInsertPoint(check_avgpool);
    llvm::Value* is_avgpool = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 21));
    llvm::BasicBlock* check_batchnorm = llvm::BasicBlock::Create(ctx_.context(), "check_batchnorm", current_func);
    ctx_.builder().CreateCondBr(is_avgpool, avgpool_block, check_batchnorm);

    ctx_.builder().SetInsertPoint(avgpool_block);
    // AvgPool backward: grad / pool_window_size (scalar case: pool_size=1)
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- BATCHNORM (type=22): standard 3-gradient backward ---
    ctx_.builder().SetInsertPoint(check_batchnorm);
    llvm::Value* is_batchnorm = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 22));
    llvm::BasicBlock* check_layernorm = llvm::BasicBlock::Create(ctx_.context(), "check_layernorm", current_func);
    ctx_.builder().CreateCondBr(is_batchnorm, batchnorm_block, check_layernorm);

    ctx_.builder().SetInsertPoint(batchnorm_block);
    // BatchNorm backward: dL/d_input = gamma * grad / sqrt(var + eps)
    // dL/d_gamma = grad * (x - mean) / sqrt(var + eps), dL/d_beta = grad
    // Scalar approximation: gamma=1, var=1, eps=1e-5 → grad ≈ grad * 1/sqrt(1+eps) ≈ grad
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad); // gamma/beta params
    ctx_.builder().CreateBr(done_block);

    // --- LAYERNORM (type=23): same structure as batchnorm ---
    ctx_.builder().SetInsertPoint(check_layernorm);
    llvm::Value* is_layernorm = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 23));
    llvm::BasicBlock* check_transpose = llvm::BasicBlock::Create(ctx_.context(), "check_transpose", current_func);
    ctx_.builder().CreateCondBr(is_layernorm, layernorm_block, check_transpose);

    ctx_.builder().SetInsertPoint(layernorm_block);
    // LayerNorm backward: same as BatchNorm but along feature dim
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- TRANSPOSE (type=25): grad = transpose(upstream_grad) ---
    ctx_.builder().SetInsertPoint(check_transpose);
    llvm::Value* is_transpose = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 25));
    llvm::BasicBlock* check_reshape = llvm::BasicBlock::Create(ctx_.context(), "check_reshape", current_func);
    ctx_.builder().CreateCondBr(is_transpose, transpose_block, check_reshape);

    ctx_.builder().SetInsertPoint(transpose_block);
    // Transpose backward: grad_input = transpose(upstream_grad)
    // Scalar case: identity (transpose of scalar is scalar)
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- RESHAPE (type=26): grad = reshape(upstream_grad, original_shape) ---
    ctx_.builder().SetInsertPoint(check_reshape);
    llvm::Value* is_reshape = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 26));
    llvm::BasicBlock* check_attention = llvm::BasicBlock::Create(ctx_.context(), "check_attention", current_func);
    ctx_.builder().CreateCondBr(is_reshape, reshape_block, check_attention);

    ctx_.builder().SetInsertPoint(reshape_block);
    // Reshape backward: reshape gradient back to input shape (scalar: passthrough)
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- ATTENTION (type=29): dV=attn^T@grad, dQ/dK through softmax backward ---
    ctx_.builder().SetInsertPoint(check_attention);
    llvm::Value* is_attention = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 29));
    llvm::BasicBlock* check_mha = llvm::BasicBlock::Create(ctx_.context(), "check_mha", current_func);
    ctx_.builder().CreateCondBr(is_attention, attention_block, check_mha);

    ctx_.builder().SetInsertPoint(attention_block);
    // Attention backward: dV = attn_weights^T @ grad_output
    // dS = grad_output @ V^T, then through softmax backward, then:
    // dQ = dS_softmax @ K / sqrt(d_k), dK = dS_softmax^T @ Q / sqrt(d_k)
    // Scalar approximation: gradient flows to Q, K, V inputs
    if (input1) accumulateGradient(input1, node_grad); // Q
    if (input2) accumulateGradient(input2, node_grad); // K (V would be input3)
    ctx_.builder().CreateBr(done_block);

    // --- MULTIHEAD_ATTENTION (type=30) ---
    ctx_.builder().SetInsertPoint(check_mha);
    llvm::Value* is_mha = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 30));
    llvm::BasicBlock* check_posenc = llvm::BasicBlock::Create(ctx_.context(), "check_posenc", current_func);
    ctx_.builder().CreateCondBr(is_mha, mha_block, check_posenc);

    ctx_.builder().SetInsertPoint(mha_block);
    // Multi-head attention backward: split across heads, per-head attention backward,
    // then backprop through W_Q, W_K, W_V, W_O projection matrices
    // Scalar approximation: gradient flows through
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- POSITIONAL_ENCODING (type=31): additive constant, gradient passes through ---
    ctx_.builder().SetInsertPoint(check_posenc);
    llvm::Value* is_posenc = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 31));
    llvm::BasicBlock* check_embedding = llvm::BasicBlock::Create(ctx_.context(), "check_embedding", current_func);
    ctx_.builder().CreateCondBr(is_posenc, posenc_block, check_embedding);

    ctx_.builder().SetInsertPoint(posenc_block);
    // Positional encoding is additive: y = x + PE (PE is constant)
    // dL/dx = dL/dy (gradient passes through unchanged)
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- EMBEDDING (type=32): scatter-add ---
    ctx_.builder().SetInsertPoint(check_embedding);
    llvm::Value* is_embedding = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 32));
    llvm::BasicBlock* check_hyp_dist = llvm::BasicBlock::Create(ctx_.context(), "check_hyp_dist", current_func);
    ctx_.builder().CreateCondBr(is_embedding, embedding_block, check_hyp_dist);

    ctx_.builder().SetInsertPoint(embedding_block);
    // Embedding backward: weight_grad[indices[i]] += upstream_grad[i]
    // Scalar approximation: gradient flows to the embedding weight input
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // ===== GEOMETRIC/HYPERBOLIC OPERATION BACKWARD PASSES =====
    // All use Poincaré ball model with conformal factor λ_x = 2/(1-||x||²)

    // --- HYPERBOLIC_DISTANCE (type=33) ---
    ctx_.builder().SetInsertPoint(check_hyp_dist);
    llvm::Value* is_hyp_dist = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 33));
    llvm::BasicBlock* check_poincare_exp = llvm::BasicBlock::Create(ctx_.context(), "check_poincare_exp", current_func);
    ctx_.builder().CreateCondBr(is_hyp_dist, hyp_dist_block, check_poincare_exp);

    ctx_.builder().SetInsertPoint(hyp_dist_block);
    {
        // d(x,y) = acosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))
        // dL/dx = dL/dd * dd/dx, where dd/dx involves conformal factors
        // Scalar approximation: use Euclidean gradient scaled by conformal factor
        if (input1 && input2) {
            llvm::Value* x_val = loadNodeValue(input1);
            llvm::Value* y_val = loadNodeValue(input2);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
            // Conformal factor λ_x = 2/(1-x²)
            llvm::Value* x_sq = ctx_.builder().CreateFMul(x_val, x_val);
            llvm::Value* y_sq = ctx_.builder().CreateFMul(y_val, y_val);
            llvm::Value* denom_x = ctx_.builder().CreateFSub(one, x_sq);
            llvm::Value* denom_y = ctx_.builder().CreateFSub(one, y_sq);
            // Clamp to avoid division by zero at boundary
            llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
            llvm::Value* safe_dx = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(denom_x, eps), eps, denom_x);
            llvm::Value* safe_dy = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(denom_y, eps), eps, denom_y);
            llvm::Value* lambda_x = ctx_.builder().CreateFDiv(two, safe_dx);
            llvm::Value* lambda_y = ctx_.builder().CreateFDiv(two, safe_dy);
            // diff = x - y
            llvm::Value* diff = ctx_.builder().CreateFSub(x_val, y_val);
            // grad_x = grad * lambda_x² * diff / dist_factor
            llvm::Value* lx_sq = ctx_.builder().CreateFMul(lambda_x, lambda_x);
            llvm::Value* grad_x = ctx_.builder().CreateFMul(node_grad, ctx_.builder().CreateFMul(lx_sq, diff));
            // grad_y = -grad * lambda_y² * diff / dist_factor
            llvm::Value* ly_sq = ctx_.builder().CreateFMul(lambda_y, lambda_y);
            llvm::Value* neg_diff = ctx_.builder().CreateFNeg(diff);
            llvm::Value* grad_y = ctx_.builder().CreateFMul(node_grad, ctx_.builder().CreateFMul(ly_sq, neg_diff));
            accumulateGradient(input1, grad_x);
            accumulateGradient(input2, grad_y);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- POINCARE_EXP_MAP (type=34) ---
    ctx_.builder().SetInsertPoint(check_poincare_exp);
    llvm::Value* is_poincare_exp = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 34));
    llvm::BasicBlock* check_poincare_log = llvm::BasicBlock::Create(ctx_.context(), "check_poincare_log", current_func);
    ctx_.builder().CreateCondBr(is_poincare_exp, poincare_exp_block, check_poincare_log);

    ctx_.builder().SetInsertPoint(poincare_exp_block);
    {
        // exp_p(v) = p ⊕ tanh(λ_p * ||v|| / 2) * v / ||v||
        // Scalar approximation: gradient scaled by conformal factor
        if (input1) {
            llvm::Value* p_val = loadNodeValue(input1);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
            llvm::Value* p_sq = ctx_.builder().CreateFMul(p_val, p_val);
            llvm::Value* denom = ctx_.builder().CreateFSub(one, p_sq);
            llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
            llvm::Value* safe_denom = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(denom, eps), eps, denom);
            llvm::Value* lambda_p = ctx_.builder().CreateFDiv(two, safe_denom);
            llvm::Value* grad_p = ctx_.builder().CreateFMul(node_grad, lambda_p);
            accumulateGradient(input1, grad_p);
        }
        if (input2) {
            // Tangent vector gradient: scaled by 1/λ_p
            accumulateGradient(input2, node_grad);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- POINCARE_LOG_MAP (type=35) ---
    ctx_.builder().SetInsertPoint(check_poincare_log);
    llvm::Value* is_poincare_log = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 35));
    llvm::BasicBlock* check_tangent = llvm::BasicBlock::Create(ctx_.context(), "check_tangent", current_func);
    ctx_.builder().CreateCondBr(is_poincare_log, poincare_log_block, check_tangent);

    ctx_.builder().SetInsertPoint(poincare_log_block);
    {
        // log_p(q) = (2/λ_p) * atanh(||(-p)⊕q||) * ((-p)⊕q) / ||(-p)⊕q||
        // Scalar approximation: inverse of exp map, gradient scaled by 1/λ_p
        if (input1) {
            llvm::Value* p_val = loadNodeValue(input1);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
            llvm::Value* p_sq = ctx_.builder().CreateFMul(p_val, p_val);
            llvm::Value* denom = ctx_.builder().CreateFSub(one, p_sq);
            llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
            llvm::Value* safe_denom = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(denom, eps), eps, denom);
            // 1/λ_p = (1-||p||²)/2
            llvm::Value* inv_lambda = ctx_.builder().CreateFDiv(safe_denom, two);
            llvm::Value* grad_p = ctx_.builder().CreateFMul(node_grad, inv_lambda);
            accumulateGradient(input1, grad_p);
        }
        if (input2) accumulateGradient(input2, node_grad);
    }
    ctx_.builder().CreateBr(done_block);

    // --- TANGENT_PROJECT (type=36) ---
    ctx_.builder().SetInsertPoint(check_tangent);
    llvm::Value* is_tangent = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 36));
    llvm::BasicBlock* check_geodesic = llvm::BasicBlock::Create(ctx_.context(), "check_geodesic", current_func);
    ctx_.builder().CreateCondBr(is_tangent, tangent_proj_block, check_geodesic);

    ctx_.builder().SetInsertPoint(tangent_proj_block);
    // Tangent space projection: projects vector onto tangent plane
    // Gradient passes through (projection is linear)
    if (input1) accumulateGradient(input1, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- GEODESIC_ATTENTION (type=37) ---
    ctx_.builder().SetInsertPoint(check_geodesic);
    llvm::Value* is_geodesic = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 37));
    llvm::BasicBlock* check_mobius_add = llvm::BasicBlock::Create(ctx_.context(), "check_mobius_add", current_func);
    ctx_.builder().CreateCondBr(is_geodesic, geodesic_attn_block, check_mobius_add);

    ctx_.builder().SetInsertPoint(geodesic_attn_block);
    // Geodesic attention: attention in hyperbolic space using geodesic distances
    // Gradient flows to Q, K inputs
    if (input1) accumulateGradient(input1, node_grad);
    if (input2) accumulateGradient(input2, node_grad);
    ctx_.builder().CreateBr(done_block);

    // --- MOBIUS_ADD (type=38) ---
    ctx_.builder().SetInsertPoint(check_mobius_add);
    llvm::Value* is_mobius_add = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 38));
    llvm::BasicBlock* check_mobius_matmul = llvm::BasicBlock::Create(ctx_.context(), "check_mobius_matmul", current_func);
    ctx_.builder().CreateCondBr(is_mobius_add, mobius_add_block, check_mobius_matmul);

    ctx_.builder().SetInsertPoint(mobius_add_block);
    {
        // Möbius addition: x ⊕ y = ((1+2<x,y>+||y||²)x + (1-||x||²)y) / (1+2<x,y>+||x||²||y||²)
        // Scalar 1D case: simplified derivative involves conformal factors
        if (input1 && input2) {
            llvm::Value* x_val = loadNodeValue(input1);
            llvm::Value* y_val = loadNodeValue(input2);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* x_sq = ctx_.builder().CreateFMul(x_val, x_val);
            llvm::Value* y_sq = ctx_.builder().CreateFMul(y_val, y_val);
            // d(x⊕y)/dx at scalar level: (1+y²)/(1+2xy+x²y²)² * (1+2xy+y²-x²(1+2xy+y²)+...)
            // Simplified: gradient ~ (1-||y||²)/(1+2xy+||x||²||y||²)² * (denominator terms)
            // Use conformal scaling: λ_{x⊕y}/λ_x for grad_x, λ_{x⊕y}/λ_y for grad_y
            llvm::Value* xy = ctx_.builder().CreateFMul(x_val, y_val);
            llvm::Value* two_xy = ctx_.builder().CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 2.0), xy);
            llvm::Value* xsq_ysq = ctx_.builder().CreateFMul(x_sq, y_sq);
            llvm::Value* denom = ctx_.builder().CreateFAdd(one, ctx_.builder().CreateFAdd(two_xy, xsq_ysq));
            llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
            llvm::Value* safe_denom = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(denom, eps), eps, denom);
            llvm::Value* denom_sq = ctx_.builder().CreateFMul(safe_denom, safe_denom);
            llvm::Value* inv_denom_sq = ctx_.builder().CreateFDiv(one, denom_sq);
            // dx: (1 + 2xy + y²) * safe_denom - that simplifies, use gyration-based formula
            llvm::Value* factor_x = ctx_.builder().CreateFDiv(
                ctx_.builder().CreateFSub(one, y_sq), safe_denom);
            llvm::Value* factor_y = ctx_.builder().CreateFDiv(
                ctx_.builder().CreateFSub(one, x_sq), safe_denom);
            accumulateGradient(input1, ctx_.builder().CreateFMul(node_grad, factor_x));
            accumulateGradient(input2, ctx_.builder().CreateFMul(node_grad, factor_y));
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- MOBIUS_MATMUL (type=39) ---
    ctx_.builder().SetInsertPoint(check_mobius_matmul);
    llvm::Value* is_mobius_matmul = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 39));
    llvm::BasicBlock* check_gyrovector = llvm::BasicBlock::Create(ctx_.context(), "check_gyrovector", current_func);
    ctx_.builder().CreateCondBr(is_mobius_matmul, mobius_matmul_block, check_gyrovector);

    ctx_.builder().SetInsertPoint(mobius_matmul_block);
    {
        // Möbius matrix multiplication: M ⊗ x = exp_0(M * log_0(x))
        // Gradient: dL/dM = dL/d(M⊗x) * (log_0(x))^T, dL/dx via chain rule through exp/log
        // Scalar approximation: gradient scaled by conformal factor
        if (input1) accumulateGradient(input1, node_grad); // M
        if (input2) {
            llvm::Value* x_val = loadNodeValue(input2);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
            llvm::Value* x_sq = ctx_.builder().CreateFMul(x_val, x_val);
            llvm::Value* denom = ctx_.builder().CreateFSub(one, x_sq);
            llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
            llvm::Value* safe_denom = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(denom, eps), eps, denom);
            llvm::Value* lambda_x = ctx_.builder().CreateFDiv(two, safe_denom);
            llvm::Value* grad_x = ctx_.builder().CreateFMul(node_grad, lambda_x);
            accumulateGradient(input2, grad_x);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- GYROVECTOR_SPACE (type=40) ---
    ctx_.builder().SetInsertPoint(check_gyrovector);
    llvm::Value* is_gyrovector = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 40));
    llvm::BasicBlock* unknown_type_block = llvm::BasicBlock::Create(ctx_.context(), "grad_unknown_type", current_func);
    ctx_.builder().CreateCondBr(is_gyrovector, gyrovector_block, unknown_type_block);

    ctx_.builder().SetInsertPoint(gyrovector_block);
    {
        // Gyrovector space operation: general operation in the Poincaré ball
        // Gradient uses conformal factor scaling
        if (input1 && input2) {
            llvm::Value* x_val = loadNodeValue(input1);
            llvm::Value* y_val = loadNodeValue(input2);
            llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
            llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
            llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
            // λ_x = 2/(1-||x||²)
            llvm::Value* x_sq = ctx_.builder().CreateFMul(x_val, x_val);
            llvm::Value* y_sq = ctx_.builder().CreateFMul(y_val, y_val);
            llvm::Value* dx = ctx_.builder().CreateFSub(one, x_sq);
            llvm::Value* dy = ctx_.builder().CreateFSub(one, y_sq);
            llvm::Value* safe_dx = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(dx, eps), eps, dx);
            llvm::Value* safe_dy = ctx_.builder().CreateSelect(
                ctx_.builder().CreateFCmpOLT(dy, eps), eps, dy);
            llvm::Value* lambda_x = ctx_.builder().CreateFDiv(two, safe_dx);
            llvm::Value* lambda_y = ctx_.builder().CreateFDiv(two, safe_dy);
            accumulateGradient(input1, ctx_.builder().CreateFMul(node_grad, lambda_x));
            accumulateGradient(input2, ctx_.builder().CreateFMul(node_grad, lambda_y));
        }
    }
    ctx_.builder().CreateBr(done_block);

    // Unknown type: emit runtime warning for unhandled AD_NODE types
    ctx_.builder().SetInsertPoint(unknown_type_block);
    {
        // Call fprintf(stderr, "Warning: Unknown AD node type %d in backward pass\n", type)
        llvm::FunctionType* fprintf_type = llvm::FunctionType::get(
            ctx_.int32Type(), {ctx_.ptrType(), ctx_.ptrType()}, true);
        llvm::FunctionCallee fprintf_fn = ctx_.module().getOrInsertFunction("fprintf", fprintf_type);

        // Get stderr via __stderrp (macOS) or stderr global
        llvm::GlobalVariable* stderr_var = ctx_.module().getGlobalVariable("__stderrp");
        if (!stderr_var) {
            stderr_var = new llvm::GlobalVariable(ctx_.module(), ctx_.ptrType(), false,
                llvm::GlobalValue::ExternalLinkage, nullptr, "__stderrp");
        }
        llvm::Value* stderr_val = ctx_.builder().CreateLoad(ctx_.ptrType(), stderr_var);

        llvm::Value* fmt_str = ctx_.builder().CreateGlobalStringPtr(
            "Warning: Unknown AD node type %d in backward pass\n");
        ctx_.builder().CreateCall(fprintf_fn, {stderr_val, fmt_str, node_type});
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

    // Overflow check: abort if depth >= MAX_TAPE_DEPTH
    {
        llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* overflow_bb = llvm::BasicBlock::Create(ctx_.context(), "tape_overflow", current_func);
        llvm::BasicBlock* safe_bb = llvm::BasicBlock::Create(ctx_.context(), "tape_safe", current_func);
        llvm::Value* is_overflow = ctx_.builder().CreateICmpUGE(depth,
            llvm::ConstantInt::get(ctx_.int64Type(), CodegenContext::MAX_TAPE_DEPTH));
        ctx_.builder().CreateCondBr(is_overflow, overflow_bb, safe_bb);

        ctx_.builder().SetInsertPoint(overflow_bb);
        // Print error and abort
        llvm::FunctionType* fprintf_type = llvm::FunctionType::get(ctx_.int32Type(),
            {ctx_.ptrType(), ctx_.ptrType()}, true);
        llvm::FunctionCallee fprintf_func = ctx_.module().getOrInsertFunction("fprintf", fprintf_type);
        llvm::FunctionType* fdopen_type = llvm::FunctionType::get(ctx_.ptrType(),
            {ctx_.int32Type(), ctx_.ptrType()}, false);
        // Use stderr via global
        llvm::FunctionCallee abort_func = ctx_.module().getOrInsertFunction("abort",
            llvm::FunctionType::get(ctx_.voidType(), false));
        ctx_.builder().CreateCall(abort_func);
        ctx_.builder().CreateUnreachable();

        ctx_.builder().SetInsertPoint(safe_bb);
    }

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
        case 10: // AD_NODE_POW
            {
                llvm::Function* pow_func = getMathFunc("pow");
                if (!pow_func) return nullptr;
                result_value = ctx_.builder().CreateCall(pow_func, {left_value, right_value});
            }
            break;
        case 44: // AD_NODE_MAX
            {
                // max(a, b) = a if a > b else b
                llvm::Value* cmp = ctx_.builder().CreateFCmpOGT(left_value, right_value);
                result_value = ctx_.builder().CreateSelect(cmp, left_value, right_value);
            }
            break;
        case 45: // AD_NODE_MIN
            {
                // min(a, b) = a if a < b else b
                llvm::Value* cmp = ctx_.builder().CreateFCmpOLT(left_value, right_value);
                result_value = ctx_.builder().CreateSelect(cmp, left_value, right_value);
            }
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

// ===== ML ACTIVATION FUNCTION DUAL NUMBER OPERATIONS =====
// These implement chain rule for ML activation functions

// ReLU: relu(a, a') = (max(0, a), a > 0 ? a' : 0)
// Derivative: d/dx[relu(f(x))] = f'(x) if f(x) > 0, else 0
llvm::Value* AutodiffCodegen::dualRelu(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    // Value: max(0, a)
    llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(a, zero);
    llvm::Value* value = ctx_.builder().CreateSelect(is_positive, a, zero);

    // Derivative: a > 0 ? a' : 0
    llvm::Value* deriv = ctx_.builder().CreateSelect(is_positive, a_prime, zero);

    return createDualNumber(value, deriv);
}

// Sigmoid: σ(a, a') = (σ(a), a' * σ(a) * (1 - σ(a)))
// Chain rule: d/dx[σ(f(x))] = σ(f(x)) * (1 - σ(f(x))) * f'(x)
llvm::Value* AutodiffCodegen::dualSigmoid(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* exp_func = getMathFunc("exp");
    if (!exp_func) return nullptr;

    // σ(a) = 1 / (1 + exp(-a))
    llvm::Value* neg_a = ctx_.builder().CreateFNeg(a);
    llvm::Value* exp_neg_a = ctx_.builder().CreateCall(exp_func, {neg_a});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* denom = ctx_.builder().CreateFAdd(one, exp_neg_a);
    llvm::Value* sigma_a = ctx_.builder().CreateFDiv(one, denom);

    // Derivative: a' * σ(a) * (1 - σ(a))
    llvm::Value* one_minus_sigma = ctx_.builder().CreateFSub(one, sigma_a);
    llvm::Value* sigma_deriv = ctx_.builder().CreateFMul(sigma_a, one_minus_sigma);
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, sigma_deriv);

    return createDualNumber(sigma_a, deriv);
}

// GELU: Gaussian Error Linear Unit
// Approximation: gelu(x) ≈ x * σ(1.702 * x)
// d/dx[x * σ(1.702*x)] = σ(1.702*x) + x * 1.702 * σ(1.702*x) * (1 - σ(1.702*x))
llvm::Value* AutodiffCodegen::dualGelu(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* exp_func = getMathFunc("exp");
    if (!exp_func) return nullptr;

    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* coeff = llvm::ConstantFP::get(ctx_.doubleType(), 1.702);

    // σ(1.702 * a)
    llvm::Value* scaled_a = ctx_.builder().CreateFMul(coeff, a);
    llvm::Value* neg_scaled_a = ctx_.builder().CreateFNeg(scaled_a);
    llvm::Value* exp_neg = ctx_.builder().CreateCall(exp_func, {neg_scaled_a});
    llvm::Value* denom = ctx_.builder().CreateFAdd(one, exp_neg);
    llvm::Value* sigma = ctx_.builder().CreateFDiv(one, denom);

    // Value: a * σ(1.702 * a)
    llvm::Value* value = ctx_.builder().CreateFMul(a, sigma);

    // Derivative: a' * (σ(1.702*a) + a * 1.702 * σ(1.702*a) * (1 - σ(1.702*a)))
    llvm::Value* one_minus_sigma = ctx_.builder().CreateFSub(one, sigma);
    llvm::Value* sigma_deriv = ctx_.builder().CreateFMul(sigma, one_minus_sigma);
    llvm::Value* scaled_sigma_deriv = ctx_.builder().CreateFMul(coeff, sigma_deriv);
    llvm::Value* a_times_scaled = ctx_.builder().CreateFMul(a, scaled_sigma_deriv);
    llvm::Value* total_deriv = ctx_.builder().CreateFAdd(sigma, a_times_scaled);
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, total_deriv);

    return createDualNumber(value, deriv);
}

// Leaky ReLU: leaky_relu(a, a') = (a > 0 ? a : α*a, a > 0 ? a' : α*a')
// Derivative: d/dx[leaky_relu(f(x))] = f'(x) if f(x) > 0, else α * f'(x)
llvm::Value* AutodiffCodegen::dualLeakyRelu(llvm::Value* dual, double alpha) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* alpha_val = llvm::ConstantFP::get(ctx_.doubleType(), alpha);

    // Value: a > 0 ? a : α * a
    llvm::Value* is_positive = ctx_.builder().CreateFCmpOGT(a, zero);
    llvm::Value* alpha_a = ctx_.builder().CreateFMul(alpha_val, a);
    llvm::Value* value = ctx_.builder().CreateSelect(is_positive, a, alpha_a);

    // Derivative: a > 0 ? a' : α * a'
    llvm::Value* alpha_a_prime = ctx_.builder().CreateFMul(alpha_val, a_prime);
    llvm::Value* deriv = ctx_.builder().CreateSelect(is_positive, a_prime, alpha_a_prime);

    return createDualNumber(value, deriv);
}

// SiLU (Swish): silu(a) = a * σ(a)
// d/dx[x * σ(x)] = σ(x) + x * σ(x) * (1 - σ(x)) = σ(x) * (1 + x * (1 - σ(x)))
llvm::Value* AutodiffCodegen::dualSilu(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* exp_func = getMathFunc("exp");
    if (!exp_func) return nullptr;

    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

    // σ(a) = 1 / (1 + exp(-a))
    llvm::Value* neg_a = ctx_.builder().CreateFNeg(a);
    llvm::Value* exp_neg_a = ctx_.builder().CreateCall(exp_func, {neg_a});
    llvm::Value* denom = ctx_.builder().CreateFAdd(one, exp_neg_a);
    llvm::Value* sigma_a = ctx_.builder().CreateFDiv(one, denom);

    // Value: a * σ(a)
    llvm::Value* value = ctx_.builder().CreateFMul(a, sigma_a);

    // Derivative: a' * σ(a) * (1 + a * (1 - σ(a)))
    llvm::Value* one_minus_sigma = ctx_.builder().CreateFSub(one, sigma_a);
    llvm::Value* a_times_one_minus = ctx_.builder().CreateFMul(a, one_minus_sigma);
    llvm::Value* one_plus_term = ctx_.builder().CreateFAdd(one, a_times_one_minus);
    llvm::Value* sigma_times_term = ctx_.builder().CreateFMul(sigma_a, one_plus_term);
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, sigma_times_term);

    return createDualNumber(value, deriv);
}

// Square: square(a, a') = (a², 2 * a * a')
llvm::Value* AutodiffCodegen::dualSquare(llvm::Value* dual) {
    if (!dual) return nullptr;

    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);

    // Value: a²
    llvm::Value* value = ctx_.builder().CreateFMul(a, a);

    // Derivative: 2 * a * a'
    llvm::Value* two_a = ctx_.builder().CreateFMul(two, a);
    llvm::Value* deriv = ctx_.builder().CreateFMul(two_a, a_prime);

    return createDualNumber(value, deriv);
}

// Max: max(a, b, a', b') = (max(a, b), a > b ? a' : b')
// Subgradient selection: when a == b, we arbitrarily choose a'
llvm::Value* AutodiffCodegen::dualMax(llvm::Value* dual_a, llvm::Value* dual_b) {
    if (!dual_a || !dual_b) return nullptr;

    llvm::Value* a = getDualPrimal(dual_a);
    llvm::Value* a_prime = getDualTangent(dual_a);
    llvm::Value* b = getDualPrimal(dual_b);
    llvm::Value* b_prime = getDualTangent(dual_b);

    // Value: max(a, b)
    llvm::Value* a_gt_b = ctx_.builder().CreateFCmpOGT(a, b);
    llvm::Value* value = ctx_.builder().CreateSelect(a_gt_b, a, b);

    // Derivative: a > b ? a' : b'
    llvm::Value* deriv = ctx_.builder().CreateSelect(a_gt_b, a_prime, b_prime);

    return createDualNumber(value, deriv);
}

// Min: min(a, b, a', b') = (min(a, b), a < b ? a' : b')
// Subgradient selection: when a == b, we arbitrarily choose a'
llvm::Value* AutodiffCodegen::dualMin(llvm::Value* dual_a, llvm::Value* dual_b) {
    if (!dual_a || !dual_b) return nullptr;

    llvm::Value* a = getDualPrimal(dual_a);
    llvm::Value* a_prime = getDualTangent(dual_a);
    llvm::Value* b = getDualPrimal(dual_b);
    llvm::Value* b_prime = getDualTangent(dual_b);

    // Value: min(a, b)
    llvm::Value* a_lt_b = ctx_.builder().CreateFCmpOLT(a, b);
    llvm::Value* value = ctx_.builder().CreateSelect(a_lt_b, a, b);

    // Derivative: a < b ? a' : b'
    llvm::Value* deriv = ctx_.builder().CreateSelect(a_lt_b, a_prime, b_prime);

    return createDualNumber(value, deriv);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
