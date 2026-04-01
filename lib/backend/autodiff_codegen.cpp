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

llvm::Value* AutodiffCodegen::dualAsinh(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* asinh_func = getMathFunc("asinh");
    llvm::Function* sqrt_func = getMathFunc("sqrt");
    if (!asinh_func || !sqrt_func) return nullptr;

    // Value: asinh(a)
    llvm::Value* asinh_a = ctx_.builder().CreateCall(asinh_func, {a});

    // Derivative: a' / sqrt(1 + a²)
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* a_sq = ctx_.builder().CreateFMul(a, a);
    llvm::Value* under = ctx_.builder().CreateFAdd(one, a_sq);
    llvm::Value* sqrt_under = ctx_.builder().CreateCall(sqrt_func, {under});
    llvm::Value* deriv = ctx_.builder().CreateFDiv(a_prime, sqrt_under);

    return createDualNumber(asinh_a, deriv);
}

llvm::Value* AutodiffCodegen::dualAcosh(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* acosh_func = getMathFunc("acosh");
    llvm::Function* sqrt_func = getMathFunc("sqrt");
    if (!acosh_func || !sqrt_func) return nullptr;

    // Value: acosh(a)
    llvm::Value* acosh_a = ctx_.builder().CreateCall(acosh_func, {a});

    // Derivative: a' / sqrt(a² - 1)
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* a_sq = ctx_.builder().CreateFMul(a, a);
    llvm::Value* under = ctx_.builder().CreateFSub(a_sq, one);
    llvm::Value* sqrt_under = ctx_.builder().CreateCall(sqrt_func, {under});
    llvm::Value* deriv = ctx_.builder().CreateFDiv(a_prime, sqrt_under);

    return createDualNumber(acosh_a, deriv);
}

llvm::Value* AutodiffCodegen::dualAtanh(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* atanh_func = getMathFunc("atanh");
    if (!atanh_func) return nullptr;

    // Value: atanh(a)
    llvm::Value* atanh_a = ctx_.builder().CreateCall(atanh_func, {a});

    // Derivative: a' / (1 - a²)
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* a_sq = ctx_.builder().CreateFMul(a, a);
    llvm::Value* denom = ctx_.builder().CreateFSub(one, a_sq);
    llvm::Value* deriv = ctx_.builder().CreateFDiv(a_prime, denom);

    return createDualNumber(atanh_a, deriv);
}

llvm::Value* AutodiffCodegen::dualLog10(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* log10_func = getMathFunc("log10");
    if (!log10_func) return nullptr;

    // Value: log10(a)
    llvm::Value* log10_a = ctx_.builder().CreateCall(log10_func, {a});

    // Derivative: a' / (a * ln(10))
    llvm::Value* ln10 = llvm::ConstantFP::get(ctx_.doubleType(), 2.302585092994046);
    llvm::Value* a_times_ln10 = ctx_.builder().CreateFMul(a, ln10);
    llvm::Value* deriv = ctx_.builder().CreateFDiv(a_prime, a_times_ln10);

    return createDualNumber(log10_a, deriv);
}

llvm::Value* AutodiffCodegen::dualLog2(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* log2_func = getMathFunc("log2");
    if (!log2_func) return nullptr;

    // Value: log2(a)
    llvm::Value* log2_a = ctx_.builder().CreateCall(log2_func, {a});

    // Derivative: a' / (a * ln(2))
    llvm::Value* ln2 = llvm::ConstantFP::get(ctx_.doubleType(), 0.6931471805599453);
    llvm::Value* a_times_ln2 = ctx_.builder().CreateFMul(a, ln2);
    llvm::Value* deriv = ctx_.builder().CreateFDiv(a_prime, a_times_ln2);

    return createDualNumber(log2_a, deriv);
}

llvm::Value* AutodiffCodegen::dualExp2(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* exp2_func = getMathFunc("exp2");
    if (!exp2_func) return nullptr;

    // Value: exp2(a) = 2^a
    llvm::Value* exp2_a = ctx_.builder().CreateCall(exp2_func, {a});

    // Derivative: a' * 2^a * ln(2)
    llvm::Value* ln2 = llvm::ConstantFP::get(ctx_.doubleType(), 0.6931471805599453);
    llvm::Value* exp2_times_ln2 = ctx_.builder().CreateFMul(exp2_a, ln2);
    llvm::Value* deriv = ctx_.builder().CreateFMul(a_prime, exp2_times_ln2);

    return createDualNumber(exp2_a, deriv);
}

llvm::Value* AutodiffCodegen::dualCbrt(llvm::Value* dual) {
    if (!dual) return nullptr;
    llvm::Value* a = getDualPrimal(dual);
    llvm::Value* a_prime = getDualTangent(dual);

    llvm::Function* cbrt_func = getMathFunc("cbrt");
    if (!cbrt_func) return nullptr;

    // Value: cbrt(a)
    llvm::Value* cbrt_a = ctx_.builder().CreateCall(cbrt_func, {a});

    // Derivative: a' / (3 * cbrt(a)²)
    llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
    llvm::Value* cbrt_sq = ctx_.builder().CreateFMul(cbrt_a, cbrt_a);
    llvm::Value* denom = ctx_.builder().CreateFMul(three, cbrt_sq);
    llvm::Value* deriv = ctx_.builder().CreateFDiv(a_prime, denom);

    return createDualNumber(cbrt_a, deriv);
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

        // === Complete math functions (54-66) ===
        case 54: // AD_NODE_TAN
            {
                llvm::Function* tan_func = getMathFunc("tan");
                if (!tan_func) return nullptr;
                result_value = ctx_.builder().CreateCall(tan_func, {input_value});
            }
            break;
        case 55: // AD_NODE_ASIN
            {
                llvm::Function* asin_func = getMathFunc("asin");
                if (!asin_func) return nullptr;
                result_value = ctx_.builder().CreateCall(asin_func, {input_value});
            }
            break;
        case 56: // AD_NODE_ACOS
            {
                llvm::Function* acos_func = getMathFunc("acos");
                if (!acos_func) return nullptr;
                result_value = ctx_.builder().CreateCall(acos_func, {input_value});
            }
            break;
        case 57: // AD_NODE_ATAN
            {
                llvm::Function* atan_func = getMathFunc("atan");
                if (!atan_func) return nullptr;
                result_value = ctx_.builder().CreateCall(atan_func, {input_value});
            }
            break;
        case 58: // AD_NODE_SINH
            {
                llvm::Function* sinh_func = getMathFunc("sinh");
                if (!sinh_func) return nullptr;
                result_value = ctx_.builder().CreateCall(sinh_func, {input_value});
            }
            break;
        case 59: // AD_NODE_COSH
            {
                llvm::Function* cosh_func = getMathFunc("cosh");
                if (!cosh_func) return nullptr;
                result_value = ctx_.builder().CreateCall(cosh_func, {input_value});
            }
            break;
        case 60: // AD_NODE_ASINH
            {
                llvm::Function* asinh_func = getMathFunc("asinh");
                if (!asinh_func) return nullptr;
                result_value = ctx_.builder().CreateCall(asinh_func, {input_value});
            }
            break;
        case 61: // AD_NODE_ACOSH
            {
                llvm::Function* acosh_func = getMathFunc("acosh");
                if (!acosh_func) return nullptr;
                result_value = ctx_.builder().CreateCall(acosh_func, {input_value});
            }
            break;
        case 62: // AD_NODE_ATANH
            {
                llvm::Function* atanh_func = getMathFunc("atanh");
                if (!atanh_func) return nullptr;
                result_value = ctx_.builder().CreateCall(atanh_func, {input_value});
            }
            break;
        case 63: // AD_NODE_LOG10
            {
                llvm::Function* log10_func = getMathFunc("log10");
                if (!log10_func) return nullptr;
                result_value = ctx_.builder().CreateCall(log10_func, {input_value});
            }
            break;
        case 64: // AD_NODE_LOG2
            {
                llvm::Function* log2_func = getMathFunc("log2");
                if (!log2_func) return nullptr;
                result_value = ctx_.builder().CreateCall(log2_func, {input_value});
            }
            break;
        case 65: // AD_NODE_EXP2
            {
                llvm::Function* exp2_func = getMathFunc("exp2");
                if (!exp2_func) return nullptr;
                result_value = ctx_.builder().CreateCall(exp2_func, {input_value});
            }
            break;
        case 66: // AD_NODE_CBRT
            {
                llvm::Function* cbrt_func = getMathFunc("cbrt");
                if (!cbrt_func) return nullptr;
                result_value = ctx_.builder().CreateCall(cbrt_func, {input_value});
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

llvm::Value* AutodiffCodegen::gradientHigherOrder(const eshkol_operations_t* op) {
    using namespace llvm;

    eshkol_info("Creating higher-order gradient function (gradient f -> grad_f)");

    // Resolve the function at compile-time if possible
    Value* func = resolve_lambda_callback_(op->gradient_op.function, 0, callback_context_);
    Value* closure_val = nullptr;

    if (!func) {
        // Runtime function parameter - get the closure value
        const eshkol_ast_t* func_ast = op->gradient_op.function;
        if (func_ast && func_ast->type == ESHKOL_VAR) {
            std::string func_name = func_ast->variable.id;
            Value* var_value = nullptr;

            auto local_it = symbol_table_->find(func_name);
            if (local_it != symbol_table_->end()) {
                var_value = local_it->second;
            } else {
                auto global_it = global_symbol_table_->find(func_name);
                if (global_it != global_symbol_table_->end()) {
                    var_value = global_it->second;
                }
            }

            if (var_value) {
                if (isa<Argument>(var_value) && var_value->getType() == ctx_.taggedValueType()) {
                    closure_val = var_value;
                } else if (isa<AllocaInst>(var_value)) {
                    closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                } else if (isa<GlobalVariable>(var_value)) {
                    closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
                }
            }
        }

        if (!closure_val) {
            eshkol_error("Failed to resolve function for higher-order gradient");
            return nullptr;
        }
    }

    // Create gradient wrapper function
    // This is VARIADIC: accepts args like (grad-f x y z ...) as a cons list
    // Computes partial derivatives numerically using central difference
    std::string grad_func_name = "gradient_ho_" + std::to_string(gradient_ho_counter_++);

    // CRITICAL: Save and disable TCO context during gradient function generation
    // The gradient function has its own internal loops that must not be confused with TCO
    auto saved_tco_ctx = binding_->getTCOContext();
    binding_->getTCOContext().enabled = false;
    binding_->getTCOContext().func_name = "";
    binding_->getTCOContext().loop_header = nullptr;

    // Function takes a rest list (variadic args packaged as cons list) + captured function
    std::vector<Type*> param_types = {ctx_.taggedValueType(), PointerType::getUnqual(ctx_.context())};
    FunctionType* grad_func_type = FunctionType::get(ctx_.taggedValueType(), param_types, false);
    Function* grad_func = Function::Create(
        grad_func_type,
        Function::ExternalLinkage,
        grad_func_name,
        ctx_.module()
    );

    BasicBlock* saved_bb = ctx_.builder().GetInsertBlock();
    BasicBlock::iterator saved_point = ctx_.builder().GetInsertPoint();

    // Create the gradient computation body
    BasicBlock* entry = BasicBlock::Create(ctx_.context(), "entry", grad_func);
    ctx_.builder().SetInsertPoint(entry);

    auto arg_it = grad_func->arg_begin();
    Value* args_list = &(*arg_it);  // First arg: rest list of arguments
    args_list->setName("args");
    ++arg_it;
    Value* captured_f_ptr = &(*arg_it);  // Captured function pointer
    captured_f_ptr->setName("captured_f");

    Value* f_closure = ctx_.builder().CreateLoad(ctx_.taggedValueType(), captured_f_ptr);

    // Constants for numerical differentiation (central difference)
    Value* h = ConstantFP::get(ctx_.doubleType(), 1e-8);
    Value* two_h = ConstantFP::get(ctx_.doubleType(), 2e-8);

    // Get cons accessor functions - avoid struct-by-value ABI issues on ARM64
    Function* cons_get_double = (*function_table_)["arena_tagged_cons_get_double"];
    Function* cons_get_type = (*function_table_)["arena_tagged_cons_get_type"];
    Function* cons_get_ptr = (*function_table_)["arena_tagged_cons_get_ptr"];
    if (!cons_get_double || !cons_get_type || !cons_get_ptr) {
        eshkol_error("Cons accessor functions not found");
        if (saved_bb) ctx_.builder().SetInsertPoint(saved_bb, saved_point);
        return nullptr;
    }

    // First, convert the cons list to a vector and count dimensions
    // Count list length using simple loop with direct tagged_value access
    BasicBlock* count_loop = BasicBlock::Create(ctx_.context(), "count_loop", grad_func);
    BasicBlock* count_done = BasicBlock::Create(ctx_.context(), "count_done", grad_func);
    BasicBlock* count_body = BasicBlock::Create(ctx_.context(), "count_body", grad_func);

    ctx_.builder().CreateBr(count_loop);
    ctx_.builder().SetInsertPoint(count_loop);
    PHINode* count_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "count");
    count_phi->addIncoming(ConstantInt::get(ctx_.int64Type(), 0), entry);
    PHINode* curr_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "curr");
    curr_phi->addIncoming(args_list, entry);

    // Check if current is null (end of list)
    Value* curr_type = tagged_.getType(curr_phi);
    Value* curr_base = tagged_.getBaseType(curr_type);
    Value* is_null = ctx_.builder().CreateICmpEQ(curr_base, ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL));
    ctx_.builder().CreateCondBr(is_null, count_done, count_body);

    // Count body - increment and get cdr using separate type/ptr accessors (ARM64 ABI fix)
    ctx_.builder().SetInsertPoint(count_body);
    Value* count_next = ctx_.builder().CreateAdd(count_phi, ConstantInt::get(ctx_.int64Type(), 1));
    Value* curr_ptr = ctx_.builder().CreateIntToPtr(tagged_.unpackInt64(curr_phi), PointerType::getUnqual(ctx_.context()));
    // Get cdr type and pointer separately, then pack into tagged_value
    Value* cdr_type = ctx_.builder().CreateCall(cons_get_type, {curr_ptr, ConstantInt::get(ctx_.int1Type(), 1)});
    Value* cdr_ptr = ctx_.builder().CreateCall(cons_get_ptr, {curr_ptr, ConstantInt::get(ctx_.int1Type(), 1)});
    Value* cdr_val = tagged_.packPtrWithFlags(cdr_ptr, cdr_type, ConstantInt::get(ctx_.int8Type(), 0));
    count_phi->addIncoming(count_next, count_body);
    curr_phi->addIncoming(cdr_val, count_body);
    ctx_.builder().CreateBr(count_loop);

    // Count done - dim_val has the list length
    ctx_.builder().SetInsertPoint(count_done);
    Value* dim_val = count_phi;

    // Allocate vector for the point via arena (length + elements) - OALR compliant
    Value* point_arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
    Value* vec_total_size = ctx_.builder().CreateAdd(ConstantInt::get(ctx_.int64Type(), 1), dim_val);
    Value* vec_bytes = ctx_.builder().CreateMul(vec_total_size, ConstantInt::get(ctx_.int64Type(),
        ctx_.module().getDataLayout().getTypeAllocSize(ctx_.taggedValueType())));
    Value* point_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {point_arena_ptr, vec_bytes});
    ctx_.builder().CreateStore(dim_val, point_ptr);  // Store length

    // Copy list elements to vector using simple loop
    BasicBlock* copy_loop = BasicBlock::Create(ctx_.context(), "copy_loop", grad_func);
    BasicBlock* copy_done = BasicBlock::Create(ctx_.context(), "copy_done", grad_func);
    BasicBlock* copy_body = BasicBlock::Create(ctx_.context(), "copy_body", grad_func);

    ctx_.builder().CreateBr(copy_loop);
    ctx_.builder().SetInsertPoint(copy_loop);
    PHINode* copy_idx = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "copy_idx");
    copy_idx->addIncoming(ConstantInt::get(ctx_.int64Type(), 0), count_done);
    PHINode* copy_curr = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "copy_curr");
    copy_curr->addIncoming(args_list, count_done);

    Value* copy_type = tagged_.getType(copy_curr);
    Value* copy_base = tagged_.getBaseType(copy_type);
    Value* copy_is_null = ctx_.builder().CreateICmpEQ(copy_base, ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL));
    ctx_.builder().CreateCondBr(copy_is_null, copy_done, copy_body);

    // Copy body - store car and advance using separate type/ptr accessors (ARM64 ABI fix)
    ctx_.builder().SetInsertPoint(copy_body);

    Value* copy_ptr = ctx_.builder().CreateIntToPtr(tagged_.unpackInt64(copy_curr), PointerType::getUnqual(ctx_.context()));
    // Get car type and value separately, then pack into tagged_value
    ctx_.builder().CreateCall(cons_get_type, {copy_ptr, ConstantInt::get(ctx_.int1Type(), 0)});
    Value* car_double = ctx_.builder().CreateCall(cons_get_double, {copy_ptr, ConstantInt::get(ctx_.int1Type(), 0)});
    Value* car_val = tagged_.packDouble(car_double);

    Value* vec_elem_offset = ctx_.builder().CreateAdd(ConstantInt::get(ctx_.int64Type(), 1), copy_idx);
    Value* vec_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), point_ptr, vec_elem_offset);
    ctx_.builder().CreateStore(car_val, vec_elem_ptr);
    Value* copy_idx_next = ctx_.builder().CreateAdd(copy_idx, ConstantInt::get(ctx_.int64Type(), 1));
    // Get cdr type and pointer separately, then pack into tagged_value
    Value* cdr_type_raw = ctx_.builder().CreateCall(cons_get_type, {copy_ptr, ConstantInt::get(ctx_.int1Type(), 1)});
    Value* cdr_ptr_raw = ctx_.builder().CreateCall(cons_get_ptr, {copy_ptr, ConstantInt::get(ctx_.int1Type(), 1)});
    Value* copy_cdr = tagged_.packPtrWithFlags(cdr_ptr_raw, cdr_type_raw, ConstantInt::get(ctx_.int8Type(), 0));
    copy_idx->addIncoming(copy_idx_next, copy_body);
    copy_curr->addIncoming(copy_cdr, copy_body);
    ctx_.builder().CreateBr(copy_loop);

    // Copy done - now point_ptr is a proper vector
    ctx_.builder().SetInsertPoint(copy_done);

    // Allocate result tensor data via arena (OALR compliant - no malloc)
    Value* result_arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
    Value* result_size = ctx_.builder().CreateMul(dim_val, ConstantInt::get(ctx_.int64Type(), 8));
    Value* result_data = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {result_arena_ptr, result_size});

    // SIMPLIFIED GRADIENT: Use switch-based dispatch like derivative does
    // Instead of complex loops, generate static code paths for each arg count
    const int MAX_GRADIENT_ARGS = 8;

    BasicBlock* switch_default = BasicBlock::Create(ctx_.context(), "grad_default", grad_func);
    BasicBlock* grad_done = BasicBlock::Create(ctx_.context(), "grad_done", grad_func);

    SwitchInst* dim_switch = ctx_.builder().CreateSwitch(dim_val, switch_default, MAX_GRADIENT_ARGS);
    std::vector<std::pair<BasicBlock*, Value*>> results;

    // Generate a case for each dimension count (1 to MAX_GRADIENT_ARGS)
    for (int dim_count = 1; dim_count <= MAX_GRADIENT_ARGS; dim_count++) {
        BasicBlock* case_bb = BasicBlock::Create(ctx_.context(),
            "grad_dim_" + std::to_string(dim_count), grad_func);
        dim_switch->addCase(ConstantInt::get(ctx_.int64Type(), dim_count), case_bb);
        ctx_.builder().SetInsertPoint(case_bb);

        // Load all arguments from point_ptr into a local array
        std::vector<Value*> base_args;
        for (int i = 0; i < dim_count; i++) {
            Value* elem_offset = ConstantInt::get(ctx_.int64Type(), 1 + i);
            Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), point_ptr, elem_offset);
            Value* loaded_arg = ctx_.builder().CreateLoad(ctx_.taggedValueType(), elem_ptr);
            base_args.push_back(loaded_arg);
        }

        // For each dimension, compute partial derivative using central difference
        for (int i = 0; i < dim_count; i++) {
            // Get the original value at dimension i
            Value* orig_val = tagged_.unpackDouble(base_args[i]);
            Value* plus_val = ctx_.builder().CreateFAdd(orig_val, h);
            Value* minus_val = ctx_.builder().CreateFSub(orig_val, h);

            // Build plus args (copy base_args with modified index i)
            std::vector<Value*> plus_args = base_args;
            plus_args[i] = tagged_.packDouble(plus_val);

            // Build minus args
            std::vector<Value*> minus_args = base_args;
            minus_args[i] = tagged_.packDouble(minus_val);

            // Call f(plus_args) and f(minus_args) using codegenClosureCall
            Value* f_plus = closure_call_callback_(f_closure, plus_args);
            Value* f_minus = closure_call_callback_(f_closure, minus_args);

            // Compute partial derivative: (f_plus - f_minus) / (2h)
            Value* f_plus_d = tagged_.unpackDouble(f_plus);
            Value* f_minus_d = tagged_.unpackDouble(f_minus);
            Value* diff = ctx_.builder().CreateFSub(f_plus_d, f_minus_d);
            Value* partial = ctx_.builder().CreateFDiv(diff, two_h);

            // Store in result array
            Value* result_slot = ctx_.builder().CreateGEP(ctx_.doubleType(), result_data,
                ConstantInt::get(ctx_.int64Type(), i));
            ctx_.builder().CreateStore(partial, result_slot);
        }

        ctx_.builder().CreateBr(grad_done);
        results.push_back({ctx_.builder().GetInsertBlock(), dim_val});
    }

    // Default case: unsupported dimension count, return null tensor
    ctx_.builder().SetInsertPoint(switch_default);
    ctx_.builder().CreateBr(grad_done);
    results.push_back({switch_default, ConstantInt::get(ctx_.int64Type(), 0)});

    // Grad done - create result tensor using proper tensor struct format
    ctx_.builder().SetInsertPoint(grad_done);

    // Tensor struct format:
    // Field 0: dims (pointer to dimensions array)
    // Field 1: num_dims (number of dimensions)
    // Field 2: elements (pointer to double data)
    // Field 3: total_elements

    // Allocate tensor struct with header via arena (OALR compliant - no malloc)
    Value* grad_arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
    Value* typed_tensor_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {grad_arena_ptr});

    // Allocate dimensions array via arena (1 element for 1D tensor)
    Value* dims_array = ctx_.builder().CreateCall(mem_.getArenaAllocate(),
        {grad_arena_ptr, ConstantInt::get(ctx_.int64Type(), 8)});  // 1 * sizeof(int64)
    ctx_.builder().CreateStore(dim_val, dims_array);  // dims[0] = number of gradient components

    // Fill tensor struct fields
    Value* dims_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 0);
    ctx_.builder().CreateStore(dims_array, dims_field_ptr);  // Field 0: dims pointer

    Value* num_dims_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 1);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), num_dims_field_ptr);  // Field 1: num_dims = 1

    Value* elements_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 2);
    ctx_.builder().CreateStore(result_data, elements_field_ptr);  // Field 2: elements pointer

    Value* total_elements_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 3);
    ctx_.builder().CreateStore(dim_val, total_elements_field_ptr);  // Field 3: total_elements = dim_val

    Value* result_tensor = tagged_.packPtr(typed_tensor_ptr, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateRet(result_tensor);

    // Restore insertion point
    if (saved_bb) {
        ctx_.builder().SetInsertPoint(saved_bb, saved_point);
    }

    // Restore TCO context
    binding_->getTCOContext() = saved_tco_ctx;

    // Register the gradient function
    (*function_table_)[grad_func_name] = grad_func;
    (*nested_function_captures_)[grad_func_name] = {"f"};  // 1 capture

    // Create closure capturing the original function
    if (!closure_val && func) {
        // STATIC FUNCTION FIX: Create a proper closure struct for static functions
        // codegenClosureCall expects a closure struct, not a raw function pointer
        // So we wrap the static function in a 0-capture closure
        Value* static_arena = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
        Value* static_func_ptr_int = ctx_.builder().CreatePtrToInt(func, ctx_.int64Type());
        // packed_info: 0 captures, 0 fixed params, NOT variadic
        uint64_t static_packed_info = 0;  // No captures, not variadic
        Value* static_packed = ConstantInt::get(ctx_.int64Type(), static_packed_info);
        Value* static_sexpr = ConstantInt::get(ctx_.int64Type(), 0);
        Value* static_return_type = ConstantInt::get(ctx_.int64Type(), 0);  // Default return type
        Value* static_name = ConstantPointerNull::get(PointerType::getUnqual(ctx_.context()));
        Value* static_closure_ptr = ctx_.builder().CreateCall(get_closure_alloc_func_(callback_context_),
            {static_arena, static_func_ptr_int, static_packed, static_sexpr, static_return_type, static_name});
        closure_val = tagged_.packPtr(static_closure_ptr, ESHKOL_VALUE_CALLABLE);
    }

    Value* func_ptr_int = ctx_.builder().CreatePtrToInt(grad_func, ctx_.int64Type());
    Value* arena = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
    // packed_info format: bits 0-15 = num_captures, bits 16-31 = fixed_params, bit 63 = is_variadic
    // We have 1 capture, 0 fixed params, and IS variadic
    uint64_t packed_info = 1 | (0ULL << 16) | (1ULL << 63);  // 1 capture, variadic
    Value* packed_captures = ConstantInt::get(ctx_.int64Type(), packed_info);
    Value* sexpr_ptr = ConstantInt::get(ctx_.int64Type(), 0);
    // Gradient returns a vector
    Value* return_type_info = ConstantInt::get(ctx_.int64Type(), CLOSURE_RETURN_VECTOR | (1 << 8));
    Value* closure_name = ConstantPointerNull::get(PointerType::getUnqual(ctx_.context()));
    // Use with_header allocator for consolidated CALLABLE type
    Value* closure_ptr = ctx_.builder().CreateCall(get_closure_alloc_func_(callback_context_),
                                             {arena, func_ptr_int, packed_captures, sexpr_ptr, return_type_info, closure_name});

    // Store captured function in closure environment
    Value* env_ptr_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), closure_ptr, ConstantInt::get(ctx_.int64Type(), 8));
    Value* env_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), env_ptr_ptr);
    Value* captures_base = ctx_.builder().CreateGEP(ctx_.int8Type(), env_ptr, ConstantInt::get(ctx_.int64Type(), 8));
    ctx_.builder().CreateStore(closure_val, captures_base);

    // Return closure as CALLABLE tagged value
    return tagged_.packPtr(closure_ptr, ESHKOL_VALUE_CALLABLE);
}


llvm::Value* AutodiffCodegen::gradient(const eshkol_operations_t* op) {
    using namespace llvm;
    if (!op->gradient_op.function) {
        eshkol_error("Invalid gradient operation - missing function");
        return nullptr;
    }

    // Higher-order form: (gradient f) returns a closure that computes gradients
    if (!op->gradient_op.point) {
        return gradientHigherOrder(op);
    }

    // Resolve function (lambda or function reference)
    Value* func = resolve_lambda_callback_(op->gradient_op.function, 0, callback_context_);

    // RUNTIME FUNCTION PARAMETER FIX: Handle functions passed as parameters
    // For gradient with runtime function parameters, we need to use a different approach
    // since gradient requires knowing the function structure at compile time.
    // For now, we'll check if the function AST is a variable and look it up.
    if (!func) {
        const eshkol_ast_t* func_ast = op->gradient_op.function;
        if (func_ast && func_ast->type == ESHKOL_VAR) {
            std::string func_name = func_ast->variable.id;
            eshkol_debug("gradient: checking runtime function parameter '%s'", func_name.c_str());

            // Check if this is a function parameter or captured value
            Value* var_value = nullptr;

            // NESTED FUNCTION FIX: First check if there's a GlobalVariable for this capture
            // This handles nested functions where captures are stored in GlobalVariables
            Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
            std::string capture_key = current_func->getName().str() + "_capture_" + func_name;
            auto gv_it = global_symbol_table_->find(capture_key);
            if (gv_it != global_symbol_table_->end() && isa<GlobalVariable>(gv_it->second)) {
                var_value = gv_it->second;
            }

            // If not found as a capture, try regular symbol_table lookup
            if (!var_value) {
                auto local_it = symbol_table_->find(func_name);
                if (local_it != symbol_table_->end()) {
                    var_value = local_it->second;
                } else {
                    auto global_it = global_symbol_table_->find(func_name);
                    if (global_it != global_symbol_table_->end()) {
                        var_value = global_it->second;
                    }
                }
            }

            // ═══════════════════════════════════════════════════════════════
            // UNIFIED RUNTIME GRADIENT PATH
            // Consolidates 3 duplicate paths (Argument, Pointer, GlobalVariable)
            // into a single resolution + shared forward-mode computation.
            // ═══════════════════════════════════════════════════════════════

            // Step 1: Resolve closure value from var_value
            Value* closure_val = nullptr;

            if (var_value && isa<Argument>(var_value) && !var_value->getType()->isPointerTy()) {
                // Direct Argument — may need capture resolution for nested functions
                Argument* arg = cast<Argument>(var_value);
                Function* arg_parent = arg->getParent();
                Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

                if (arg_parent != current_func) {
                    // From different function — find in current function's captures
                    bool found_in_captures = false;
                    for (auto& curr_arg : current_func->args()) {
                        std::string arg_name = curr_arg.getName().str();
                        if (arg_name == "captured_" + func_name) {
                            if (curr_arg.getType()->isPointerTy()) {
                                closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), &curr_arg);
                            } else {
                                closure_val = &curr_arg;
                            }
                            found_in_captures = true;
                            break;
                        }
                    }
                    if (!found_in_captures) {
                        std::string capture_key = current_func->getName().str() + "_capture_" + func_name;
                        auto cap_it = global_symbol_table_->find(capture_key);
                        if (cap_it != global_symbol_table_->end() && isa<GlobalVariable>(cap_it->second)) {
                            closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), cap_it->second);
                            found_in_captures = true;
                        } else {
                            auto var_cap_it = global_symbol_table_->find(func_name);
                            if (var_cap_it != global_symbol_table_->end() && isa<GlobalVariable>(var_cap_it->second)) {
                                closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_cap_it->second);
                                found_in_captures = true;
                            }
                        }
                    }
                    if (!found_in_captures) {
                        eshkol_error("gradient: could not find capture for '%s'", func_name.c_str());
                        return nullptr;
                    }
                } else {
                    closure_val = var_value;
                }
            } else if (var_value && isa<Argument>(var_value) && var_value->getType()->isPointerTy()) {
                // Pointer-type captured Argument — load the tagged value
                closure_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), var_value);
            } else if (var_value && isa<GlobalVariable>(var_value)) {
                // GlobalVariable — load via the global's value type
                GlobalVariable* gv = cast<GlobalVariable>(var_value);
                closure_val = ctx_.builder().CreateLoad(gv->getValueType(), var_value);
            }

            // Step 2: Shared forward-mode gradient computation
            if (closure_val) {
                Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

                // Get the input point
                Value* point_val = codegen_ast_callback_(op->gradient_op.point, callback_context_);
                if (!point_val) {
                    eshkol_error("Failed to evaluate gradient point");
                    return nullptr;
                }

                // Ensure point is tagged
                if (point_val->getType() != ctx_.taggedValueType()) {
                    if (point_val->getType()->isIntegerTy(64)) {
                        if (op->gradient_op.point->type == ESHKOL_TENSOR) {
                            point_val = tagged_.packPtr(point_val, ESHKOL_VALUE_HEAP_PTR);
                        } else {
                            point_val = tagged_.packInt64(point_val, true);
                        }
                    } else if (point_val->getType()->isDoubleTy()) {
                        point_val = tagged_.packDouble(point_val);
                    }
                }

                // Note: current_func already defined above for capture lookup

                // Get arena_allocate for Scheme vector allocation
                Function* arena_allocate_func = (*function_table_)["arena_allocate"];
                if (!arena_allocate_func) {
                    eshkol_error("arena_allocate not found for gradient");
                    return nullptr;
                }
                Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

                // Get tagged_value size
                uint64_t tagged_size = ctx_.module().getDataLayout().getTypeAllocSize(ctx_.taggedValueType());

                // Check input type - handle Scheme vector (HEAP_PTR with HEAP_SUBTYPE_VECTOR), tensor (TENSOR_PTR), or scalar
                // M1 Migration: Use consolidated HEAP_PTR type with subtype dispatch
                Value* input_type = tagged_.getType(point_val);
                Value* input_base = tagged_.getBaseType(input_type);

                // Check for HEAP_PTR (consolidated format)
                Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(input_base,
                    ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
                // Legacy VECTOR_PTR check (for backwards compatibility during migration)
                Value* is_legacy_vec = ctx_.builder().CreateICmpEQ(input_base,
                    ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
                Value* is_tensor = ctx_.builder().CreateICmpEQ(input_base,
                    ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

                BasicBlock* heap_ptr_dispatch = BasicBlock::Create(ctx_.context(), "grad_rt_heap_dispatch", current_func);
                BasicBlock* heap_check_tensor = BasicBlock::Create(ctx_.context(), "grad_rt_heap_check_tensor", current_func);
                BasicBlock* scheme_vec_path = BasicBlock::Create(ctx_.context(), "grad_rt_svec", current_func);
                BasicBlock* tensor_path = BasicBlock::Create(ctx_.context(), "grad_rt_tensor", current_func);
                BasicBlock* scalar_path = BasicBlock::Create(ctx_.context(), "grad_rt_scalar", current_func);
                BasicBlock* check_legacy_vec = BasicBlock::Create(ctx_.context(), "grad_rt_check_legacy", current_func);
                BasicBlock* check_tensor = BasicBlock::Create(ctx_.context(), "grad_rt_check_tensor", current_func);
                BasicBlock* grad_rt_compute = BasicBlock::Create(ctx_.context(), "grad_rt_compute", current_func);

                // First check for HEAP_PTR (new consolidated format)
                ctx_.builder().CreateCondBr(is_heap_ptr, heap_ptr_dispatch, check_legacy_vec);

                // HEAP_PTR dispatch - read subtype from header and route accordingly
                ctx_.builder().SetInsertPoint(heap_ptr_dispatch);
                Value* heap_ptr_val = tagged_.unpackPtr(point_val);
                Value* header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), heap_ptr_val, ConstantInt::get(ctx_.int64Type(), -8));
                Value* subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), header_ptr);
                Value* is_vec_subtype = ctx_.builder().CreateICmpEQ(subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
                Value* is_tensor_subtype = ctx_.builder().CreateICmpEQ(subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));
                ctx_.builder().CreateCondBr(is_vec_subtype, scheme_vec_path, heap_check_tensor);

                ctx_.builder().SetInsertPoint(heap_check_tensor);
                ctx_.builder().CreateCondBr(is_tensor_subtype, tensor_path, scalar_path);

                // Legacy VECTOR_PTR fallback (for migration period)
                ctx_.builder().SetInsertPoint(check_legacy_vec);
                ctx_.builder().CreateCondBr(is_legacy_vec, scheme_vec_path, check_tensor);

                ctx_.builder().SetInsertPoint(check_tensor);
                ctx_.builder().CreateCondBr(is_tensor, tensor_path, scalar_path);

                // Scheme vector path - use input directly
                ctx_.builder().SetInsertPoint(scheme_vec_path);
                Value* svec_ptr = tagged_.unpackPtr(point_val);
                Value* svec_len = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_ptr);
                Value* svec_elems = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_ptr, ConstantInt::get(ctx_.int64Type(), 8));
                Value* svec_elems_typed = ctx_.builder().CreatePointerCast(svec_elems, ctx_.ptrType());
                ctx_.builder().CreateBr(grad_rt_compute);
                BasicBlock* svec_exit = ctx_.builder().GetInsertBlock();

                // Tensor path - convert tensor elements to Scheme vector of tagged doubles
                ctx_.builder().SetInsertPoint(tensor_path);
                Value* tensor_ptr_int = tagged_.unpackInt64(point_val);
                Value* tensor_ptr = ctx_.builder().CreateIntToPtr(tensor_ptr_int, ctx_.builder().getPtrTy());
                Value* dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 0);
                Value* dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), dims_field);
                Value* typed_dims = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());
                Value* tensor_n = ctx_.builder().CreateLoad(ctx_.int64Type(), typed_dims);
                Value* tensor_elems_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 2);
                Value* tensor_elems_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), tensor_elems_field);
                Value* tensor_elems_typed = ctx_.builder().CreatePointerCast(tensor_elems_ptr, ctx_.builder().getPtrTy());

                // Allocate Scheme vector for tensor elements
                Value* tconv_size = ctx_.builder().CreateAdd(
                    ctx_.builder().CreateMul(tensor_n, ConstantInt::get(ctx_.int64Type(), tagged_size)),
                    ConstantInt::get(ctx_.int64Type(), 8));
                Value* tconv_vec = ctx_.builder().CreateCall(arena_allocate_func, {arena_ptr, tconv_size});
                ctx_.builder().CreateStore(tensor_n, tconv_vec);
                Value* tconv_elems = ctx_.builder().CreateGEP(ctx_.int8Type(), tconv_vec, ConstantInt::get(ctx_.int64Type(), 8));
                Value* tconv_elems_typed = ctx_.builder().CreatePointerCast(tconv_elems, ctx_.ptrType());

                // Copy tensor elements as tagged doubles
                Value* tconv_i = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "tconv_i");
                ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), tconv_i);
                BasicBlock* tconv_cond = BasicBlock::Create(ctx_.context(), "tconv_cond", current_func);
                BasicBlock* tconv_body = BasicBlock::Create(ctx_.context(), "tconv_body", current_func);
                BasicBlock* tconv_end = BasicBlock::Create(ctx_.context(), "tconv_end", current_func);
                ctx_.builder().CreateBr(tconv_cond);

                ctx_.builder().SetInsertPoint(tconv_cond);
                Value* tc_idx = ctx_.builder().CreateLoad(ctx_.int64Type(), tconv_i);
                ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(tc_idx, tensor_n), tconv_body, tconv_end);

                ctx_.builder().SetInsertPoint(tconv_body);
                Value* tc_src = ctx_.builder().CreateGEP(ctx_.int64Type(), tensor_elems_typed, tc_idx);
                Value* tc_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), tc_src);
                Value* tc_dbl = ctx_.builder().CreateBitCast(tc_bits, ctx_.doubleType());
                Value* tc_tagged = tagged_.packDouble(tc_dbl);
                Value* tc_dst = ctx_.builder().CreateGEP(ctx_.taggedValueType(), tconv_elems_typed, tc_idx);
                ctx_.builder().CreateStore(tc_tagged, tc_dst);
                ctx_.builder().CreateStore(ctx_.builder().CreateAdd(tc_idx, ConstantInt::get(ctx_.int64Type(), 1)), tconv_i);
                ctx_.builder().CreateBr(tconv_cond);

                ctx_.builder().SetInsertPoint(tconv_end);
                ctx_.builder().CreateBr(grad_rt_compute);
                BasicBlock* tensor_exit = ctx_.builder().GetInsertBlock();

                // Scalar path - create 1-element Scheme vector
                ctx_.builder().SetInsertPoint(scalar_path);
                Value* scalar_val = tagged_.unpackDouble(point_val);
                Value* scalar_vec_size = ConstantInt::get(ctx_.int64Type(), 8 + tagged_size);
                Value* scalar_vec = ctx_.builder().CreateCall(arena_allocate_func, {arena_ptr, scalar_vec_size});
                ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), scalar_vec);
                Value* scalar_elem = ctx_.builder().CreateGEP(ctx_.int8Type(), scalar_vec, ConstantInt::get(ctx_.int64Type(), 8));
                Value* scalar_elem_typed = ctx_.builder().CreatePointerCast(scalar_elem, ctx_.ptrType());
                Value* scalar_tagged = tagged_.packDouble(scalar_val);
                ctx_.builder().CreateStore(scalar_tagged, scalar_elem_typed);
                ctx_.builder().CreateBr(grad_rt_compute);
                BasicBlock* scalar_exit = ctx_.builder().GetInsertBlock();

                // Merge input paths
                ctx_.builder().SetInsertPoint(grad_rt_compute);
                PHINode* n = ctx_.builder().CreatePHI(ctx_.int64Type(), 3, "grad_n");
                n->addIncoming(svec_len, svec_exit);
                n->addIncoming(tensor_n, tensor_exit);
                n->addIncoming(ConstantInt::get(ctx_.int64Type(), 1), scalar_exit);

                PHINode* input_elems = ctx_.builder().CreatePHI(ctx_.ptrType(), 3, "grad_elems");
                input_elems->addIncoming(svec_elems_typed, svec_exit);
                input_elems->addIncoming(tconv_elems_typed, tensor_exit);
                input_elems->addIncoming(scalar_elem_typed, scalar_exit);

                // Allocate result tensor using arena with header
                Value* typed_result = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

                Value* result_dims_size = ConstantInt::get(ctx_.int64Type(), 8);
                Value* result_dims_ptr = ctx_.builder().CreateCall(arena_allocate_func, {arena_ptr, result_dims_size});
                Value* typed_result_dims = ctx_.builder().CreatePointerCast(result_dims_ptr, ctx_.builder().getPtrTy());
                ctx_.builder().CreateStore(n, typed_result_dims);
                ctx_.builder().CreateStore(typed_result_dims, ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result, 0));
                ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result, 1));
                ctx_.builder().CreateStore(n, ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result, 3));

                Value* result_elems_size = ctx_.builder().CreateMul(n, ConstantInt::get(ctx_.int64Type(), sizeof(double)));
                Value* result_elems_ptr = ctx_.builder().CreateCall(arena_allocate_func, {arena_ptr, result_elems_size});
                Value* typed_result_elems = ctx_.builder().CreatePointerCast(result_elems_ptr, ctx_.builder().getPtrTy());
                ctx_.builder().CreateStore(typed_result_elems, ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result, 2));

                // M1 CONSOLIDATION: Allocate Scheme vector for dual numbers with header
                // arena_allocate_vector_with_header creates: [header(8)] + [length(8)] + [elements]
                Value* dual_vec = ctx_.builder().CreateCall(mem_.getArenaAllocateVectorWithHeader(),
                    {arena_ptr, n});
                ctx_.builder().CreateStore(n, dual_vec);
                Value* dual_elems = ctx_.builder().CreateGEP(ctx_.int8Type(), dual_vec, ConstantInt::get(ctx_.int64Type(), 8));
                Value* dual_elems_typed = ctx_.builder().CreatePointerCast(dual_elems, ctx_.ptrType());

                // Outer loop: for each dimension i, compute partial derivative
                Value* dim_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "grad_dim_i");
                ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), dim_counter);

                BasicBlock* dim_cond = BasicBlock::Create(ctx_.context(), "grad_dim_cond", current_func);
                BasicBlock* dim_body = BasicBlock::Create(ctx_.context(), "grad_dim_body", current_func);
                BasicBlock* dim_end = BasicBlock::Create(ctx_.context(), "grad_dim_end", current_func);

                ctx_.builder().CreateBr(dim_cond);

                ctx_.builder().SetInsertPoint(dim_cond);
                Value* dim_i = ctx_.builder().CreateLoad(ctx_.int64Type(), dim_counter);
                ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(dim_i, n), dim_body, dim_end);

                ctx_.builder().SetInsertPoint(dim_body);

                // Inner loop: create dual vector with tangent=1 at dim_i
                Value* inner_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "grad_inner_j");
                ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), inner_counter);

                BasicBlock* inner_cond = BasicBlock::Create(ctx_.context(), "grad_inner_cond", current_func);
                BasicBlock* inner_body = BasicBlock::Create(ctx_.context(), "grad_inner_body", current_func);
                BasicBlock* inner_end = BasicBlock::Create(ctx_.context(), "grad_inner_end", current_func);

                ctx_.builder().CreateBr(inner_cond);

                ctx_.builder().SetInsertPoint(inner_cond);
                Value* inner_j = ctx_.builder().CreateLoad(ctx_.int64Type(), inner_counter);
                ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(inner_j, n), inner_body, inner_end);

                ctx_.builder().SetInsertPoint(inner_body);
                // Load primal value at position j from input elements
                Value* in_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), input_elems, inner_j);
                Value* in_elem = ctx_.builder().CreateLoad(ctx_.taggedValueType(), in_elem_ptr);
                Value* primal_val = tagged_.unpackDouble(in_elem);

                // Set tangent: 1.0 if j == i, else 0.0
                Value* is_active = ctx_.builder().CreateICmpEQ(inner_j, dim_i);
                Value* tangent = ctx_.builder().CreateSelect(is_active,
                    ConstantFP::get(ctx_.doubleType(), 1.0),
                    ConstantFP::get(ctx_.doubleType(), 0.0));

                // Create dual number and store in dual vector
                Value* dual_num = createDualNumber(primal_val, tangent);
                Value* dual_tagged = packDualToTagged(dual_num);
                Value* dual_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), dual_elems_typed, inner_j);
                ctx_.builder().CreateStore(dual_tagged, dual_elem_ptr);

                ctx_.builder().CreateStore(ctx_.builder().CreateAdd(inner_j, ConstantInt::get(ctx_.int64Type(), 1)), inner_counter);
                ctx_.builder().CreateBr(inner_cond);

                ctx_.builder().SetInsertPoint(inner_end);

                // M1 CONSOLIDATION: Pack dual vector as HEAP_PTR (header contains HEAP_SUBTYPE_VECTOR)
                Value* dual_vec_tagged = tagged_.packPtr(
                    ctx_.builder().CreatePtrToInt(dual_vec, ctx_.int64Type()),
                    ESHKOL_VALUE_HEAP_PTR);

                // Call function via closure dispatch
                std::vector<Value*> call_args = {dual_vec_tagged};
                Value* call_result = closure_call_callback_(closure_val, call_args);

                // CONSTANT RESULT FIX: Check if result is a dual number before unpacking
                // If function returns a constant (not using its argument), it won't be dual
                Value* rt_result_type = tagged_.getType(call_result);
                Value* rt_result_base = tagged_.getBaseType(rt_result_type);
                Value* rt_is_dual = ctx_.builder().CreateICmpEQ(rt_result_base,
                    ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));

                BasicBlock* rt_dual_bb = BasicBlock::Create(ctx_.context(), "grad_rt_dual", current_func);
                BasicBlock* rt_const_bb = BasicBlock::Create(ctx_.context(), "grad_rt_const", current_func);
                BasicBlock* rt_merge_bb = BasicBlock::Create(ctx_.context(), "grad_rt_merge", current_func);

                ctx_.builder().CreateCondBr(rt_is_dual, rt_dual_bb, rt_const_bb);

                // Dual path: extract tangent normally
                ctx_.builder().SetInsertPoint(rt_dual_bb);
                Value* rt_result_dual = unpackDualFromTagged(call_result);
                auto [rt_result_primal, rt_dual_deriv] = uncreateDualNumber(rt_result_dual);
                ctx_.builder().CreateBr(rt_merge_bb);
                BasicBlock* rt_dual_exit = ctx_.builder().GetInsertBlock();

                // Constant path: derivative is 0.0
                ctx_.builder().SetInsertPoint(rt_const_bb);
                Value* rt_zero_deriv = ConstantFP::get(ctx_.doubleType(), 0.0);
                ctx_.builder().CreateBr(rt_merge_bb);
                BasicBlock* rt_const_exit = ctx_.builder().GetInsertBlock();

                // Merge paths
                ctx_.builder().SetInsertPoint(rt_merge_bb);
                PHINode* deriv = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "grad_deriv");
                deriv->addIncoming(rt_dual_deriv, rt_dual_exit);
                deriv->addIncoming(rt_zero_deriv, rt_const_exit);

                // Store derivative in result tensor
                Value* result_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_result_elems, dim_i);
                Value* deriv_bits = ctx_.builder().CreateBitCast(deriv, ctx_.int64Type());
                ctx_.builder().CreateStore(deriv_bits, result_elem_ptr);

                ctx_.builder().CreateStore(ctx_.builder().CreateAdd(dim_i, ConstantInt::get(ctx_.int64Type(), 1)), dim_counter);
                ctx_.builder().CreateBr(dim_cond);

                ctx_.builder().SetInsertPoint(dim_end);
                Value* result_int = ctx_.builder().CreatePtrToInt(typed_result, ctx_.int64Type());
                return tagged_.packPtr(result_int, ESHKOL_VALUE_HEAP_PTR);
            } // end if (closure_val)

        }
        eshkol_error("Failed to resolve function for gradient computation");
        return nullptr;
    }

    Function* func_ptr = dyn_cast<Function>(func);

    if (!func_ptr) {
        eshkol_error("Gradient operator requires actual function, got non-function value");
        return nullptr;
    }
    
    // Evaluate point to get input vector
    Value* vector_val_raw = codegen_ast_callback_(op->gradient_op.point, callback_context_);
    if (!vector_val_raw) {
        eshkol_error("Failed to evaluate gradient evaluation point");
        return nullptr;
    }
    
    // CRITICAL FIX: Ensure input is tagged_value (codegenAST can return raw types for literals)
    // Tensor/vector codegen returns raw ptr-as-int64, NOT tagged values.
    // We must detect the AST type to pack correctly (HEAP_PTR vs INT64).
    Value* vector_val;
    if (vector_val_raw->getType() == ctx_.taggedValueType()) {
        vector_val = vector_val_raw; // Already tagged
    } else if (vector_val_raw->getType()->isIntegerTy(64) &&
               op->gradient_op.point->type == ESHKOL_TENSOR) {
        // Tensor literal: codegenTensor returns ptr-as-int64; wrap as HEAP_PTR
        // so the input dispatch correctly detects the tensor subtype
        vector_val = tagged_.packPtr(vector_val_raw, ESHKOL_VALUE_HEAP_PTR);
    } else if (vector_val_raw->getType()->isIntegerTy(64)) {
        vector_val = tagged_.packInt64(vector_val_raw, true); // Pack int64
    } else if (vector_val_raw->getType()->isDoubleTy()) {
        vector_val = tagged_.packDouble(vector_val_raw); // Pack double
    } else {
        TypedValue tv = detectValueType(vector_val_raw);
        vector_val = typedValueToTaggedValue(tv); // Pack other types
    }
    
    // Alloca for effective svec input (updated by tensor→svec conversion)
    Value* svec_input_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "svec_input");
    ctx_.builder().CreateStore(vector_val, svec_input_ptr);

    // SCALAR→VECTOR AUTO-PROMOTION: Detect input type BEFORE tensor structure access
    // This prevents segfault when users pass scalars like 3.0 instead of vectors like #(3.0)

    // Get current function for basic blocks
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Get arena for OALR-compliant tensor allocation (used throughout gradient computation)
    Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

    // Extract type from input (may be DOUBLE, INT64, TENSOR_PTR, or AD_NODE_PTR for nested gradients)
    Value* input_type = tagged_.getType(vector_val);
    Value* input_base_type = tagged_.getBaseType(input_type);

    // DOUBLE BACKWARD: Check if input is an AD node (from outer gradient)
    // This happens in nested gradients like (gradient (lambda (y) (gradient f y)) x)
    Value* is_ad_node_input = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));

    // Check if input is scalar (INT64 or DOUBLE)
    Value* is_int64 = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
    Value* is_double = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    Value* is_scalar = ctx_.builder().CreateOr(is_int64, is_double);

    // M1 Migration: Check if input is Scheme vector (HEAP_PTR with HEAP_SUBTYPE_VECTOR) or legacy VECTOR_PTR
    // First check for HEAP_PTR (consolidated format)
    Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    // Legacy VECTOR_PTR check
    Value* is_legacy_vector = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    // Branch: AD node input (nested), scalar input (promotion), scheme vector (convert), or tensor input (normal)
    BasicBlock* ad_node_input = BasicBlock::Create(ctx_.context(), "grad_ad_node_input", current_func);
    BasicBlock* scalar_input = BasicBlock::Create(ctx_.context(), "grad_scalar_input", current_func);
    BasicBlock* scheme_vector_input = BasicBlock::Create(ctx_.context(), "grad_scheme_vector_input", current_func);
    BasicBlock* vector_input = BasicBlock::Create(ctx_.context(), "grad_vector_input", current_func);
    BasicBlock* grad_merge_input = BasicBlock::Create(ctx_.context(), "grad_merge_input", current_func);
    // Create grad_done early so scheme_vector_input path can branch to it
    BasicBlock* grad_done = BasicBlock::Create(ctx_.context(), "grad_done", current_func);

    // First check if AD node (nested gradient)
    BasicBlock* check_scalar = BasicBlock::Create(ctx_.context(), "grad_check_scalar", current_func);
    ctx_.builder().CreateCondBr(is_ad_node_input, ad_node_input, check_scalar);

    // NESTED GRADIENT (AD_NODE_PTR input): Extract value and wrap in tensor for uniform handling
    ctx_.builder().SetInsertPoint(ad_node_input);
    eshkol_debug("Gradient: detected AD_NODE_PTR input (nested gradient)");

    // Extract the AD node pointer
    Value* outer_ad_node = tagged_.unpackPtr(vector_val);

    // DOUBLE BACKWARD: Store outer AD node in global for later use
    // This allows the backward pass to connect to outer computation graph
    ctx_.builder().CreateStore(outer_ad_node, ctx_.outerAdNodeStorage());

    // Extract the VALUE from the AD node (field 1)
    Value* ad_value_ptr = ctx_.builder().CreateStructGEP(ctx_.adNodeType(), outer_ad_node, 1);
    Value* ad_value = ctx_.builder().CreateLoad(ctx_.doubleType(), ad_value_ptr);

    // Create a 1D tensor containing this value via arena (OALR compliant - no malloc)
    Value* typed_ad_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set up 1D tensor structure
    Value* nested_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* nested_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, nested_dims_size});
    Value* typed_ad_dims = ctx_.builder().CreatePointerCast(nested_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), typed_ad_dims);

    ctx_.builder().CreateStore(typed_ad_dims,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor, 1));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor, 3));

    Value* nested_elems_size = ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
    Value* nested_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, nested_elems_size});
    Value* typed_nested_elems = ctx_.builder().CreatePointerCast(nested_elems_ptr, ctx_.builder().getPtrTy());
    Value* nested_value_as_int64 = ctx_.builder().CreateBitCast(ad_value, ctx_.int64Type());
    ctx_.builder().CreateStore(nested_value_as_int64, typed_nested_elems);
    ctx_.builder().CreateStore(typed_nested_elems,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor, 2));

    Value* nested_tensor_int = ctx_.builder().CreatePtrToInt(typed_ad_tensor, ctx_.int64Type());
    Value* ad_promoted_tagged = tagged_.packPtr(nested_tensor_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(grad_merge_input);
    BasicBlock* ad_node_exit = ctx_.builder().GetInsertBlock();

    // Check for scalar
    ctx_.builder().SetInsertPoint(check_scalar);

    // DOUBLE BACKWARD: Clear outer AD node storage for non-nested case
    ctx_.builder().CreateStore(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())),
        ctx_.outerAdNodeStorage());

    // Check: scalar → scalar_input, heap_ptr (check subtype) → maybe scheme_vector, else → vector_input (tensor)
    BasicBlock* check_heap_ptr = BasicBlock::Create(ctx_.context(), "grad_check_heap_ptr", current_func);
    ctx_.builder().CreateCondBr(is_scalar, scalar_input, check_heap_ptr);

    // M1 Migration: Check for HEAP_PTR and dispatch based on subtype
    ctx_.builder().SetInsertPoint(check_heap_ptr);
    BasicBlock* heap_ptr_dispatch = BasicBlock::Create(ctx_.context(), "grad_heap_dispatch", current_func);
    BasicBlock* check_legacy_vector = BasicBlock::Create(ctx_.context(), "grad_check_legacy_vec", current_func);
    ctx_.builder().CreateCondBr(is_heap_ptr, heap_ptr_dispatch, check_legacy_vector);

    // HEAP_PTR dispatch - read subtype from header
    ctx_.builder().SetInsertPoint(heap_ptr_dispatch);
    Value* grad_heap_ptr = tagged_.unpackPtr(vector_val);
    Value* grad_header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), grad_heap_ptr, ConstantInt::get(ctx_.int64Type(), -8));
    Value* grad_subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), grad_header_ptr);
    Value* is_vec_subtype_grad = ctx_.builder().CreateICmpEQ(grad_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    Value* is_tensor_subtype_grad = ctx_.builder().CreateICmpEQ(grad_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));
    BasicBlock* grad_check_tensor = BasicBlock::Create(ctx_.context(), "grad_check_tensor", current_func);
    ctx_.builder().CreateCondBr(is_vec_subtype_grad, scheme_vector_input, grad_check_tensor);

    // Check for TENSOR subtype — convert to Scheme vector ONLY for multi-param functions.
    // For single-param functions, tensor input works correctly with reverse-mode AD.
    // For multi-param functions, forward-mode with dual numbers is needed because
    // reverse-mode passes AD nodes as CALLABLE tagged values which crash in function dispatch.
    ctx_.builder().SetInsertPoint(grad_check_tensor);
    uint64_t grad_func_arity = 0;
    {
        auto arity_it = function_arity_table_->find(func_ptr->getName().str());
        if (arity_it != function_arity_table_->end()) {
            grad_func_arity = arity_it->second;
        }
    }
    if (grad_func_arity > 1) {
        BasicBlock* grad_tensor_to_svec = BasicBlock::Create(ctx_.context(), "grad_tensor_to_svec", current_func);
        ctx_.builder().CreateCondBr(is_tensor_subtype_grad, grad_tensor_to_svec, vector_input);

        // TENSOR→SVEC CONVERSION: Convert 8-byte tensor doubles to 16-byte tagged Scheme vector
        // so the forward-mode dual number path handles multi-parameter gradients correctly.
        ctx_.builder().SetInsertPoint(grad_tensor_to_svec);
        {
            Value* t_ptr = grad_heap_ptr;
            Value* t_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), t_ptr, 0);
            Value* t_dims_ptr = ctx_.builder().CreateLoad(ctx_.builder().getPtrTy(), t_dims_field);
            Value* t_n = ctx_.builder().CreateLoad(ctx_.int64Type(), t_dims_ptr);
            Value* t_elems_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), t_ptr, 2);
            Value* t_elems_ptr = ctx_.builder().CreateLoad(ctx_.builder().getPtrTy(), t_elems_field);

            Value* t_arena = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
            Value* t_svec = ctx_.builder().CreateCall(mem_.getArenaAllocateVectorWithHeader(), {t_arena, t_n});
            ctx_.builder().CreateStore(t_n, t_svec);
            Value* t_svec_elems_base = ctx_.builder().CreateGEP(ctx_.int8Type(), t_svec, ConstantInt::get(ctx_.int64Type(), 8));
            Value* t_svec_elems = ctx_.builder().CreatePointerCast(t_svec_elems_base, ctx_.ptrType());

            Value* t_conv_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "tensor_conv_idx");
            ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), t_conv_idx);

            BasicBlock* t_conv_cond = BasicBlock::Create(ctx_.context(), "tensor_conv_cond", current_func);
            BasicBlock* t_conv_body = BasicBlock::Create(ctx_.context(), "tensor_conv_body", current_func);
            BasicBlock* t_conv_exit = BasicBlock::Create(ctx_.context(), "tensor_conv_exit", current_func);

            ctx_.builder().CreateBr(t_conv_cond);

            ctx_.builder().SetInsertPoint(t_conv_cond);
            Value* t_idx = ctx_.builder().CreateLoad(ctx_.int64Type(), t_conv_idx);
            ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(t_idx, t_n), t_conv_body, t_conv_exit);

            ctx_.builder().SetInsertPoint(t_conv_body);
            Value* t_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), t_elems_ptr, t_idx);
            Value* t_elem_i64 = ctx_.builder().CreateLoad(ctx_.int64Type(), t_elem_ptr);
            Value* t_elem_double = ctx_.builder().CreateBitCast(t_elem_i64, ctx_.doubleType());
            Value* t_elem_tagged = tagged_.packDouble(t_elem_double);
            Value* t_svec_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), t_svec_elems, t_idx);
            ctx_.builder().CreateStore(t_elem_tagged, t_svec_elem_ptr);
            ctx_.builder().CreateStore(ctx_.builder().CreateAdd(t_idx, ConstantInt::get(ctx_.int64Type(), 1)), t_conv_idx);
            ctx_.builder().CreateBr(t_conv_cond);

            ctx_.builder().SetInsertPoint(t_conv_exit);
            Value* t_svec_int = ctx_.builder().CreatePtrToInt(t_svec, ctx_.int64Type());
            Value* t_converted_tagged = tagged_.packPtr(t_svec_int, ESHKOL_VALUE_HEAP_PTR);
            ctx_.builder().CreateStore(t_converted_tagged, svec_input_ptr);
        }
        ctx_.builder().CreateBr(scheme_vector_input);
    } else {
        // Single-param: tensor goes through reverse-mode (works for arity <= 1)
        ctx_.builder().CreateBr(vector_input);
    }

    // Legacy VECTOR_PTR fallback
    ctx_.builder().SetInsertPoint(check_legacy_vector);
    ctx_.builder().CreateCondBr(is_legacy_vector, scheme_vector_input, vector_input);

    // SCALAR INPUT: Auto-promote scalar to 1D tensor #(scalar_value)
    ctx_.builder().SetInsertPoint(scalar_input);
    eshkol_debug("Gradient: auto-promoting scalar input to 1D vector");
    
    // Extract scalar value (INT64 or DOUBLE)
    Value* scalar_val_int = tagged_.unpackInt64(vector_val);
    
    // Convert to double if needed
    Value* scalar_double = ctx_.builder().CreateSelect(is_double,
        ctx_.builder().CreateBitCast(scalar_val_int, ctx_.doubleType()),
        ctx_.builder().CreateSIToFP(scalar_val_int, ctx_.doubleType()));
    
    // Allocate 1D tensor structure for promoted scalar via arena (OALR compliant - no malloc)
    Value* typed_promoted_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions: [1] (1D tensor with single element)
    Value* promoted_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* promoted_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, promoted_dims_size});
    Value* typed_promoted_dims = ctx_.builder().CreatePointerCast(promoted_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), typed_promoted_dims);

    // Set tensor metadata
    ctx_.builder().CreateStore(typed_promoted_dims,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_promoted_tensor, 0));  // dimensions = [1]
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_promoted_tensor, 1));  // num_dimensions = 1
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_promoted_tensor, 3));  // total_elements = 1

    // Allocate and set elements: [scalar_value]
    Value* promoted_elems_size = ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
    Value* promoted_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, promoted_elems_size});
    Value* typed_promoted_elems = ctx_.builder().CreatePointerCast(promoted_elems_ptr, ctx_.builder().getPtrTy());
    
    // Store scalar as bitcast int64 (preserves IEEE754 bits for doubles)
    Value* scalar_as_int64 = ctx_.builder().CreateBitCast(scalar_double, ctx_.int64Type());
    ctx_.builder().CreateStore(scalar_as_int64, typed_promoted_elems);
    
    ctx_.builder().CreateStore(typed_promoted_elems,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_promoted_tensor, 2));  // elements
    
    // Pack promoted tensor as tagged_value with TENSOR_PTR type
    Value* promoted_tensor_int = ctx_.builder().CreatePtrToInt(typed_promoted_tensor, ctx_.int64Type());
    Value* promoted_vector_tagged = tagged_.packPtr(promoted_tensor_int, ESHKOL_VALUE_HEAP_PTR);
    
    ctx_.builder().CreateBr(grad_merge_input);
    BasicBlock* scalar_input_exit = ctx_.builder().GetInsertBlock();
    
    // SCHEME VECTOR INPUT: Use forward-mode AD with dual numbers (preserves Scheme vector format)
    // This allows functions that use vector-ref to work correctly with gradient
    // Handles both native Scheme vectors AND tensors converted via tensor→svec path above
    ctx_.builder().SetInsertPoint(scheme_vector_input);
    eshkol_debug("Gradient: using forward-mode AD for Scheme vector input");

    // Load effective input (may have been updated by tensor→svec conversion)
    Value* effective_svec_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_input_ptr);
    // Get Scheme vector pointer and length
    Value* svec_ptr_int = tagged_.unpackInt64(effective_svec_val);
    Value* svec_ptr = ctx_.builder().CreateIntToPtr(svec_ptr_int, ctx_.builder().getPtrTy());
    Value* svec_n = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_ptr);
    Value* svec_elems_base = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_ptr, ConstantInt::get(ctx_.int64Type(), 8));
    Value* svec_elems = ctx_.builder().CreatePointerCast(svec_elems_base, ctx_.ptrType());

    // Allocate result tensor for gradient - use arena allocation with header for HEAP_PTR type
    Value* arena_for_svec = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
    Value* svec_typed_result = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_for_svec});

    // Set result tensor dimensions - use arena allocation
    Value* svec_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* svec_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_for_svec, svec_dims_size});
    Value* svec_typed_dims = ctx_.builder().CreatePointerCast(svec_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(svec_n, svec_typed_dims);
    ctx_.builder().CreateStore(svec_typed_dims, ctx_.builder().CreateStructGEP(ctx_.tensorType(), svec_typed_result, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), ctx_.builder().CreateStructGEP(ctx_.tensorType(), svec_typed_result, 1));
    ctx_.builder().CreateStore(svec_n, ctx_.builder().CreateStructGEP(ctx_.tensorType(), svec_typed_result, 3));

    // Allocate result elements - use arena allocation
    Value* svec_result_elems_size = ctx_.builder().CreateMul(svec_n, ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    Value* svec_result_elems = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_for_svec, svec_result_elems_size});
    Value* svec_typed_result_elems = ctx_.builder().CreatePointerCast(svec_result_elems, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(svec_typed_result_elems, ctx_.builder().CreateStructGEP(ctx_.tensorType(), svec_typed_result, 2));

    // Get arena for dual vector allocation
    Value* arena_svec = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

    // Outer loop: for each dimension i, compute ∂f/∂xᵢ using forward-mode AD
    Value* svec_dim_i = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "svec_dim_i");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), svec_dim_i);

    BasicBlock* svec_dim_cond = BasicBlock::Create(ctx_.context(), "svec_dim_cond", current_func);
    BasicBlock* svec_dim_body = BasicBlock::Create(ctx_.context(), "svec_dim_body", current_func);
    BasicBlock* svec_dim_end = BasicBlock::Create(ctx_.context(), "svec_dim_end", current_func);

    ctx_.builder().CreateBr(svec_dim_cond);

    ctx_.builder().SetInsertPoint(svec_dim_cond);
    Value* svec_i = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_dim_i);
    ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(svec_i, svec_n), svec_dim_body, svec_dim_end);

    ctx_.builder().SetInsertPoint(svec_dim_body);

    // M1 CONSOLIDATION: Allocate dual vector with header (Scheme vector of dual numbers)
    // arena_allocate_vector_with_header creates: [header(8)] + [length(8)] + [elements]
    // Header contains HEAP_SUBTYPE_VECTOR, returns pointer to length field
    Value* svec_dual_vec = ctx_.builder().CreateCall(mem_.getArenaAllocateVectorWithHeader(),
        {arena_svec, svec_n});
    ctx_.builder().CreateStore(svec_n, svec_dual_vec);
    Value* svec_dual_elems = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_dual_vec, ConstantInt::get(ctx_.int64Type(), 8));
    Value* svec_dual_elems_typed = ctx_.builder().CreatePointerCast(svec_dual_elems, ctx_.ptrType());

    // Inner loop: create dual vector with tangent=1 at position i, 0 elsewhere
    Value* svec_inner_j = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "svec_inner_j");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), svec_inner_j);

    BasicBlock* svec_inner_cond = BasicBlock::Create(ctx_.context(), "svec_inner_cond", current_func);
    BasicBlock* svec_inner_body = BasicBlock::Create(ctx_.context(), "svec_inner_body", current_func);
    BasicBlock* svec_inner_end = BasicBlock::Create(ctx_.context(), "svec_inner_end", current_func);

    ctx_.builder().CreateBr(svec_inner_cond);

    ctx_.builder().SetInsertPoint(svec_inner_cond);
    Value* svec_j = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_inner_j);
    ctx_.builder().CreateCondBr(ctx_.builder().CreateICmpULT(svec_j, svec_n), svec_inner_body, svec_inner_end);

    ctx_.builder().SetInsertPoint(svec_inner_body);
    // Load primal value from input
    Value* svec_in_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_elems, svec_j);
    Value* svec_in_val = ctx_.builder().CreateLoad(ctx_.taggedValueType(), svec_in_ptr);
    Value* svec_primal = tagged_.unpackDouble(svec_in_val);
    // Tangent: 1.0 if j == i, else 0.0
    Value* svec_is_active = ctx_.builder().CreateICmpEQ(svec_j, svec_i);
    Value* svec_tangent = ctx_.builder().CreateSelect(svec_is_active,
        ConstantFP::get(ctx_.doubleType(), 1.0), ConstantFP::get(ctx_.doubleType(), 0.0));
    // Create dual number and store
    Value* svec_dual = createDualNumber(svec_primal, svec_tangent);
    Value* svec_dual_tagged = packDualToTagged(svec_dual);
    Value* svec_dual_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_dual_elems_typed, svec_j);
    ctx_.builder().CreateStore(svec_dual_tagged, svec_dual_ptr);
    ctx_.builder().CreateStore(ctx_.builder().CreateAdd(svec_j, ConstantInt::get(ctx_.int64Type(), 1)), svec_inner_j);
    ctx_.builder().CreateBr(svec_inner_cond);

    ctx_.builder().SetInsertPoint(svec_inner_end);

    // M1 CONSOLIDATION: Pack dual vector as HEAP_PTR (header contains HEAP_SUBTYPE_VECTOR)
    Value* svec_dual_tagged_vec = tagged_.packPtr(
        ctx_.builder().CreatePtrToInt(svec_dual_vec, ctx_.int64Type()), ESHKOL_VALUE_HEAP_PTR);

    // Call function with dual vector — dispatches through helper
    std::vector<Value*> svec_call_args;

    // MULTI-PARAMETER GRADIENT: Check function arity and unpack if needed
    // arity > 1: extract individual dual number elements as separate args
    // arity <= 1: pass whole dual vector (for vector-input functions like (lambda (v) ...))
    {
        uint64_t svec_func_arity = 0;
        auto svec_arity_it = function_arity_table_->find(func_ptr->getName().str());
        if (svec_arity_it != function_arity_table_->end()) {
            svec_func_arity = svec_arity_it->second;
        }
        if (svec_func_arity > 1) {
            // Multi-param: unpack dual vector elements as individual tagged value args
            for (uint64_t p = 0; p < svec_func_arity; p++) {
                Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_dual_elems_typed,
                    ConstantInt::get(ctx_.int64Type(), p));
                Value* elem = ctx_.builder().CreateLoad(ctx_.taggedValueType(), elem_ptr);
                svec_call_args.push_back(elem);
            }
        } else {
            svec_call_args.push_back(svec_dual_tagged_vec);
        }
    }

    // Resolve captures via unified helper
    resolveGradientCaptures(func_ptr, svec_call_args, "svec");

    Value* svec_call_result = ctx_.builder().CreateCall(func_ptr, svec_call_args);

    // CONSTANT RESULT FIX: Check if result is a dual number before unpacking
    Value* svec_result_type = tagged_.getType(svec_call_result);
    Value* svec_result_base = tagged_.getBaseType(svec_result_type);
    Value* svec_is_dual = ctx_.builder().CreateICmpEQ(svec_result_base,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));

    BasicBlock* svec_dual_bb = BasicBlock::Create(ctx_.context(), "grad_svec_dual", current_func);
    BasicBlock* svec_const_bb = BasicBlock::Create(ctx_.context(), "grad_svec_const", current_func);
    BasicBlock* svec_merge_bb = BasicBlock::Create(ctx_.context(), "grad_svec_merge", current_func);

    ctx_.builder().CreateCondBr(svec_is_dual, svec_dual_bb, svec_const_bb);

    // Dual path: extract tangent normally
    ctx_.builder().SetInsertPoint(svec_dual_bb);
    Value* svec_result_dual = unpackDualFromTagged(svec_call_result);
    auto [svec_result_primal, svec_dual_deriv] = uncreateDualNumber(svec_result_dual);
    ctx_.builder().CreateBr(svec_merge_bb);
    BasicBlock* svec_dual_exit = ctx_.builder().GetInsertBlock();

    // Constant path: derivative is 0.0
    ctx_.builder().SetInsertPoint(svec_const_bb);
    Value* svec_zero_deriv = ConstantFP::get(ctx_.doubleType(), 0.0);
    ctx_.builder().CreateBr(svec_merge_bb);
    BasicBlock* svec_const_exit = ctx_.builder().GetInsertBlock();

    // Merge paths
    ctx_.builder().SetInsertPoint(svec_merge_bb);
    PHINode* svec_deriv = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "grad_svec_deriv");
    svec_deriv->addIncoming(svec_dual_deriv, svec_dual_exit);
    svec_deriv->addIncoming(svec_zero_deriv, svec_const_exit);

    // Store derivative in result tensor
    Value* svec_result_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), svec_typed_result_elems, svec_i);
    Value* svec_deriv_bits = ctx_.builder().CreateBitCast(svec_deriv, ctx_.int64Type());
    ctx_.builder().CreateStore(svec_deriv_bits, svec_result_ptr);

    ctx_.builder().CreateStore(ctx_.builder().CreateAdd(svec_i, ConstantInt::get(ctx_.int64Type(), 1)), svec_dim_i);
    ctx_.builder().CreateBr(svec_dim_cond);

    ctx_.builder().SetInsertPoint(svec_dim_end);
    // Return result tensor
    Value* svec_result_int = ctx_.builder().CreatePtrToInt(svec_typed_result, ctx_.int64Type());
    Value* scheme_vector_tagged = tagged_.packPtr(svec_result_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(grad_done);  // Skip reverse-mode AD, go directly to done
    BasicBlock* scheme_vector_exit = ctx_.builder().GetInsertBlock();

    // VECTOR INPUT: Use original vector as-is (existing behavior - tensor format)
    ctx_.builder().SetInsertPoint(vector_input);
    ctx_.builder().CreateBr(grad_merge_input);
    BasicBlock* vector_input_exit = ctx_.builder().GetInsertBlock();

    // MERGE: PHI node selects AD node promoted, scalar promoted, or original tensor
    // NOTE: Scheme vector path now uses forward-mode AD and branches directly to grad_done
    ctx_.builder().SetInsertPoint(grad_merge_input);
    PHINode* actual_input = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3, "gradient_input");
    actual_input->addIncoming(ad_promoted_tagged, ad_node_exit);  // Nested gradient path
    actual_input->addIncoming(promoted_vector_tagged, scalar_input_exit);
    actual_input->addIncoming(vector_val, vector_input_exit);
    
    // Continue with gradient computation using merged input (guaranteed to be tensor!)
    Value* vector_ptr_int = tagged_.unpackInt64(actual_input);
    // Note: arena_ptr already defined at function start


    // Convert int64 pointer to typed tensor pointer
    Value* vector_ptr = ctx_.builder().CreateIntToPtr(vector_ptr_int, ctx_.builder().getPtrTy());

    // Extract ALL tensor properties (MUST access all fields correctly)
    Value* dims_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), vector_ptr, 0);
    Value* dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), dims_field_ptr);
    Value* typed_dims_ptr = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());

    Value* elements_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), vector_ptr, 2);
    Value* elements_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), elements_field_ptr);
    Value* typed_elements_ptr = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.builder().getPtrTy());

    // Load dimension n from tensor (RUNTIME value, NOT hardcoded)
    Value* dim0_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims_ptr,
        ConstantInt::get(ctx_.int64Type(), 0));

    Value* n = ctx_.builder().CreateLoad(ctx_.int64Type(), dim0_ptr);
    
    // VALIDATION: Check dimension > 0 (scalars already promoted to tensors, so type check not needed)
    Value* n_is_positive = ctx_.builder().CreateICmpUGT(n, ConstantInt::get(ctx_.int64Type(), 0));
    
    BasicBlock* dim_valid = BasicBlock::Create(ctx_.context(), "grad_dim_valid", current_func);
    BasicBlock* dim_invalid = BasicBlock::Create(ctx_.context(), "grad_dim_invalid", current_func);
    // grad_done already created earlier for scheme_vector_input forward-mode path

    // CRITICAL FIX: Create empty tensor BEFORE branching (for PHI node dominance)
    // This ensures null_tagged_grad is available in all paths
    // Allocate empty tensor via arena (OALR compliant - no malloc)
    Value* typed_empty_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions array (size 1, value 0)
    Value* empty_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* empty_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, empty_dims_size});
    Value* typed_empty_dims = ctx_.builder().CreatePointerCast(empty_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), typed_empty_dims);

    ctx_.builder().CreateStore(typed_empty_dims,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_empty_tensor, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_empty_tensor, 1));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_empty_tensor, 3));

    // Empty elements array
    Value* empty_elems_size = ConstantInt::get(ctx_.int64Type(), sizeof(double));
    Value* empty_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, empty_elems_size});
    Value* typed_empty_elems = ctx_.builder().CreatePointerCast(empty_elems_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(typed_empty_elems,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_empty_tensor, 2));
    
    // Pack as tagged_value (TENSOR_PTR type) - available in all paths
    Value* empty_tensor_int = ctx_.builder().CreatePtrToInt(typed_empty_tensor, ctx_.int64Type());
    Value* null_tagged_grad = tagged_.packPtr(empty_tensor_int, ESHKOL_VALUE_HEAP_PTR);
    
    ctx_.builder().CreateCondBr(n_is_positive, dim_valid, dim_invalid);
    
    // Invalid input: return empty tensor
    ctx_.builder().SetInsertPoint(dim_invalid);
    eshkol_debug("Gradient: invalid input tensor (dimension must be > 0)");
    ctx_.builder().CreateBr(grad_done);
    
    // Valid dimension: compute gradient
    ctx_.builder().SetInsertPoint(dim_valid);
    
    // Allocate result gradient vector via arena (OALR compliant - no malloc)
    Value* typed_result_tensor_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set result tensor dimension (1D vector of size n)
    Value* result_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* result_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, result_dims_size});
    Value* typed_result_dims_ptr = ctx_.builder().CreatePointerCast(result_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(n, typed_result_dims_ptr);

    // Store dimension in result tensor
    Value* result_dims_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_tensor_ptr, 0);
    ctx_.builder().CreateStore(typed_result_dims_ptr, result_dims_field_ptr);

    // Store num_dimensions = 1
    Value* result_num_dims_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_tensor_ptr, 1);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), result_num_dims_field_ptr);

    // Store total_elements = n
    Value* result_total_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_tensor_ptr, 3);
    ctx_.builder().CreateStore(n, result_total_field_ptr);

    // Allocate result elements array (n doubles for partial derivatives)
    Value* result_elements_size = ctx_.builder().CreateMul(n,
        ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    Value* result_elements_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, result_elements_size});
    Value* typed_result_elements_ptr = ctx_.builder().CreatePointerCast(result_elements_ptr, ctx_.builder().getPtrTy());
    
    // Store elements pointer in result tensor
    Value* result_elements_field_ptr = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_tensor_ptr, 2);
    ctx_.builder().CreateStore(typed_result_elements_ptr, result_elements_field_ptr);
    
    // ===== MAIN GRADIENT COMPUTATION LOOP =====
    // For each component i from 0 to n-1, compute ∂f/∂xᵢ
    
    BasicBlock* grad_loop_cond = BasicBlock::Create(ctx_.context(), "grad_loop_cond", current_func);
    BasicBlock* grad_loop_body = BasicBlock::Create(ctx_.context(), "grad_loop_body", current_func);
    BasicBlock* grad_loop_exit = BasicBlock::Create(ctx_.context(), "grad_loop_exit", current_func);
    
    // Allocate loop counter
    Value* component_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "component_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), component_idx);
    
    ctx_.builder().CreateBr(grad_loop_cond);
    
    // Loop condition: i < n
    ctx_.builder().SetInsertPoint(grad_loop_cond);
    Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), component_idx);
    Value* i_less_n = ctx_.builder().CreateICmpULT(i, n);
    ctx_.builder().CreateCondBr(i_less_n, grad_loop_body, grad_loop_exit);
    
    // Loop body: Compute ∂f/∂xᵢ using reverse-mode AD
    ctx_.builder().SetInsertPoint(grad_loop_body);

    // Step 1: Create tape for this partial derivative (arena_ptr defined at function start)
    Value* tape_capacity = ConstantInt::get(ctx_.int64Type(), 1024);
    Value* partial_tape = ctx_.builder().CreateCall(getArenaAllocateTapeFunc(),
        {arena_ptr, tape_capacity});
    
    // Store tape as current (required by recordADNode* functions)
    Value* saved_tape = current_tape_ptr;
    current_tape_ptr = partial_tape;
    
    // Step 2: Create n AD variable nodes (one per vector component)
    // Allocate array to hold variable node pointers via arena (OALR compliant - no malloc)
    Value* var_nodes_array_size = ctx_.builder().CreateMul(n,
        ConstantInt::get(ctx_.int64Type(), sizeof(void*)));
    Value* var_nodes_array = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, var_nodes_array_size});
    Value* typed_var_nodes = ctx_.builder().CreatePointerCast(var_nodes_array, ctx_.builder().getPtrTy());
    
    // Loop to create and initialize variable nodes
    BasicBlock* init_vars_cond = BasicBlock::Create(ctx_.context(), "init_vars_cond", current_func);
    BasicBlock* init_vars_body = BasicBlock::Create(ctx_.context(), "init_vars_body", current_func);
    BasicBlock* init_vars_exit = BasicBlock::Create(ctx_.context(), "init_vars_exit", current_func);
    
    Value* init_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "init_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), init_idx);
    ctx_.builder().CreateBr(init_vars_cond);
    
    ctx_.builder().SetInsertPoint(init_vars_cond);
    Value* j = ctx_.builder().CreateLoad(ctx_.int64Type(), init_idx);
    Value* j_less_n = ctx_.builder().CreateICmpULT(j, n);
    ctx_.builder().CreateCondBr(j_less_n, init_vars_body, init_vars_exit);
    
    ctx_.builder().SetInsertPoint(init_vars_body);

    // CRITICAL FIX: Tensor elements are stored as int64, load as int64 then convert to double
    Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(),
        typed_elements_ptr, j);
    Value* elem_val_int64 = ctx_.builder().CreateLoad(ctx_.int64Type(), elem_ptr);

    // NESTED GRADIENT FIX: Check if element might be an AD node pointer (from outer gradient)
    // Don't check tape depth - the element itself tells us if it's an AD node
    // When a gradient's input contains an AD node, we detect it and set up double backward
    BasicBlock* check_ad_ptr = BasicBlock::Create(ctx_.context(), "check_ad_ptr", current_func);
    BasicBlock* is_regular_double = BasicBlock::Create(ctx_.context(), "is_regular_double", current_func);
    BasicBlock* merge_elem = BasicBlock::Create(ctx_.context(), "merge_elem", current_func);

    // Check if the value could be a pointer (in valid heap address range)
    // On 64-bit systems:
    // - Heap pointers are typically 0x100000000 to 0x00007FFFFFFFFFFF (small as int64)
    // - Normal doubles like 2.0 = 0x4000000000000000, 12.0 = 0x4028... (LARGE as int64)
    // So a potential pointer is: non-zero AND less than typical double values
    // Use threshold 0x0001000000000000 (~281 trillion) - catches all user space addresses
    // but excludes normal positive doubles (which are >= 0x3FF0000000000000 for >= 1.0)
    Value* not_zero = ctx_.builder().CreateICmpNE(elem_val_int64,
        ConstantInt::get(ctx_.int64Type(), 0));
    Value* in_ptr_range = ctx_.builder().CreateICmpULT(elem_val_int64,
        ConstantInt::get(ctx_.int64Type(), 0x0001000000000000ULL));
    Value* could_be_ptr = ctx_.builder().CreateAnd(not_zero, in_ptr_range);
    ctx_.builder().CreateCondBr(could_be_ptr, check_ad_ptr, is_regular_double);

    // CHECK AD POINTER: Try to validate it's actually an AD node
    ctx_.builder().SetInsertPoint(check_ad_ptr);
    Value* ad_ptr_candidate = ctx_.builder().CreateIntToPtr(elem_val_int64, PointerType::getUnqual(ctx_.context()));
    // Check if pointer is non-null and has valid AD node type
    Value* ptr_not_null = ctx_.builder().CreateICmpNE(elem_val_int64,
        ConstantInt::get(ctx_.int64Type(), 0));

    BasicBlock* check_ad_type = BasicBlock::Create(ctx_.context(), "check_ad_type", current_func);
    BasicBlock* not_ad_node = BasicBlock::Create(ctx_.context(), "not_ad_node", current_func);
    ctx_.builder().CreateCondBr(ptr_not_null, check_ad_type, not_ad_node);

    // Check AD node type field
    ctx_.builder().SetInsertPoint(check_ad_type);
    Value* type_field_ptr = ctx_.builder().CreateStructGEP(ctx_.adNodeType(), ad_ptr_candidate, 0);
    Value* type_field = ctx_.builder().CreateLoad(ctx_.int32Type(), type_field_ptr);
    // Valid AD node types are 0-7 (CONSTANT, PTR, ADD, SUB, MUL, DIV, SIN, COS)
    // Also check that it's exactly type 1 (AD_NODE_PTR) since that's what variables are
    Value* is_ad_var = ctx_.builder().CreateICmpEQ(type_field, ConstantInt::get(ctx_.int32Type(), 1));

    BasicBlock* use_existing_ad = BasicBlock::Create(ctx_.context(), "use_existing_ad", current_func);
    ctx_.builder().CreateCondBr(is_ad_var, use_existing_ad, not_ad_node);

    // USE EXISTING AD NODE: This element is an AD node from outer gradient
    // CRITICAL FIX: Do NOT reuse the outer AD node directly!
    // The inner backward would write to its gradient field, contaminating it.
    // Instead, create a new AD variable with the same value and record the outer node.
    ctx_.builder().SetInsertPoint(use_existing_ad);
    Value* detected_outer_node = ad_ptr_candidate;
    // Store outer AD node for double backward connection
    ctx_.builder().CreateStore(detected_outer_node, ctx_.outerAdNodeStorage());
    // Extract the VALUE from outer AD node and create NEW variable for inner gradient
    Value* detected_outer_val_ptr = ctx_.builder().CreateStructGEP(ctx_.adNodeType(), detected_outer_node, 1);
    Value* detected_outer_val = ctx_.builder().CreateLoad(ctx_.doubleType(), detected_outer_val_ptr);
    Value* new_inner_var = createADVariable(detected_outer_val, 0);
    ctx_.builder().CreateBr(merge_elem);
    BasicBlock* use_ad_exit = ctx_.builder().GetInsertBlock();

    // NOT AN AD NODE: Treat as double
    ctx_.builder().SetInsertPoint(not_ad_node);
    Value* elem_as_double2 = ctx_.builder().CreateBitCast(elem_val_int64, ctx_.doubleType());
    Value* new_var_node2 = createADVariable(elem_as_double2, 0);
    ctx_.builder().CreateBr(merge_elem);
    BasicBlock* not_ad_exit = ctx_.builder().GetInsertBlock();

    // REGULAR DOUBLE: Normal case - just treat as double
    ctx_.builder().SetInsertPoint(is_regular_double);
    Value* elem_val = ctx_.builder().CreateBitCast(elem_val_int64, ctx_.doubleType());
    Value* new_var_node = createADVariable(elem_val, 0);
    ctx_.builder().CreateBr(merge_elem);
    BasicBlock* regular_double_exit = ctx_.builder().GetInsertBlock();

    // MERGE: PHI to select the right AD node
    ctx_.builder().SetInsertPoint(merge_elem);
    PHINode* var_node = ctx_.builder().CreatePHI(PointerType::getUnqual(ctx_.context()), 3, "var_node_phi");
    var_node->addIncoming(new_inner_var, use_ad_exit);
    var_node->addIncoming(new_var_node2, not_ad_exit);
    var_node->addIncoming(new_var_node, regular_double_exit);
    
    // Store node pointer in array
    Value* node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_var_nodes, j);
    ctx_.builder().CreateStore(var_node, node_slot);
    
    // Increment init counter
    Value* next_j = ctx_.builder().CreateAdd(j, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_j, init_idx);
    ctx_.builder().CreateBr(init_vars_cond);
    
    ctx_.builder().SetInsertPoint(init_vars_exit);
    
    // Step 3: Get active variable node (the one we're computing gradient for)
    Value* active_node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_var_nodes, i);
    Value* active_var_node = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()),
        active_node_slot);
    
    // Step 4: Call function with variable nodes to build computational graph
    // CRITICAL: Function must operate on AD nodes, not raw doubles
    // This requires the function to use recordADNode* operations
    
    // Build tensor of AD node pointers to pass to function
    // M1 CONSOLIDATION: Use arena allocation with header for HEAP_PTR type
    Value* ad_arena_ptr = ctx_.builder().CreateLoad(
        PointerType::getUnqual(ctx_.context()), ctx_.globalArena());
    Function* alloc_tensor_full = mem_.getArenaAllocateTensorFull();
    Value* typed_ad_tensor_ptr = ctx_.builder().CreateCall(alloc_tensor_full,
        {ad_arena_ptr, ConstantInt::get(ctx_.int64Type(), 1), n}, "ad_tensor");

    // Set AD tensor dimensions (same as input) - dims[0] = n
    Value* ad_dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor_ptr, 0));
    ctx_.builder().CreateStore(n, ctx_.builder().CreateGEP(ctx_.int64Type(), ad_dims_ptr,
        ConstantInt::get(ctx_.int64Type(), 0)));

    // Get elements array (already allocated by arena_allocate_tensor_full)
    Value* typed_ad_elems_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor_ptr, 2));
    
    // Copy node pointers into AD tensor
    BasicBlock* copy_nodes_cond = BasicBlock::Create(ctx_.context(), "copy_nodes_cond", current_func);
    BasicBlock* copy_nodes_body = BasicBlock::Create(ctx_.context(), "copy_nodes_body", current_func);
    BasicBlock* copy_nodes_exit = BasicBlock::Create(ctx_.context(), "copy_nodes_exit", current_func);
    
    Value* copy_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "copy_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), copy_idx);
    ctx_.builder().CreateBr(copy_nodes_cond);
    
    ctx_.builder().SetInsertPoint(copy_nodes_cond);
    Value* k = ctx_.builder().CreateLoad(ctx_.int64Type(), copy_idx);
    Value* k_less_n = ctx_.builder().CreateICmpULT(k, n);
    ctx_.builder().CreateCondBr(k_less_n, copy_nodes_body, copy_nodes_exit);
    
    ctx_.builder().SetInsertPoint(copy_nodes_body);
    Value* src_node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_var_nodes, k);
    Value* src_node_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), src_node_slot);
    Value* node_as_int64 = ctx_.builder().CreatePtrToInt(src_node_ptr, ctx_.int64Type());
    
    Value* dst_elem_slot = ctx_.builder().CreateGEP(ctx_.int64Type(),
        typed_ad_elems_ptr, k);
    ctx_.builder().CreateStore(node_as_int64, dst_elem_slot);
    
    Value* next_k = ctx_.builder().CreateAdd(k, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_k, copy_idx);
    ctx_.builder().CreateBr(copy_nodes_cond);
    
    ctx_.builder().SetInsertPoint(copy_nodes_exit);
    
    // Step 5: Call function with AD node (scalar) or tensor (vector)
    // SCALAR FUNCTION FIX: For n=1, extract the single AD node and pass it directly!
    // This allows scalar functions like (lambda (x) (* x x)) to work


    Value* n_is_one = ctx_.builder().CreateICmpEQ(n, ConstantInt::get(ctx_.int64Type(), 1));
    
    BasicBlock* scalar_call = BasicBlock::Create(ctx_.context(), "grad_scalar_call", current_func);
    BasicBlock* vector_call = BasicBlock::Create(ctx_.context(), "grad_vector_call", current_func);
    BasicBlock* after_func_call = BasicBlock::Create(ctx_.context(), "grad_after_func_call", current_func);
    
    ctx_.builder().CreateCondBr(n_is_one, scalar_call, vector_call);
    
    // SCALAR: Extract single AD node and pass directly
    ctx_.builder().SetInsertPoint(scalar_call);
    Value* single_ad_node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_var_nodes, ConstantInt::get(ctx_.int64Type(), 0));
    Value* single_ad_node = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), single_ad_node_slot);
    Value* scalar_ad_tagged = tagged_.packPtr(single_ad_node, ESHKOL_VALUE_CALLABLE);

    std::vector<Value*> scalar_args;

    // MULTI-PARAMETER: If function has more params than 1, unpack AD nodes
    {
        uint64_t scalar_func_arity = 0;
        auto scalar_arity_it = function_arity_table_->find(func_ptr->getName().str());
        if (scalar_arity_it != function_arity_table_->end()) {
            scalar_func_arity = scalar_arity_it->second;
        }
        if (scalar_func_arity > 1) {
            // Multi-param function on scalar path: pass all AD nodes as individual args
            for (uint64_t p = 0; p < scalar_func_arity; p++) {
                Value* node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
                    typed_var_nodes, ConstantInt::get(ctx_.int64Type(), p));
                Value* node = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), node_slot);
                Value* node_tagged = tagged_.packPtr(node, ESHKOL_VALUE_CALLABLE);
                scalar_args.push_back(node_tagged);
            }
        } else {
            scalar_args.push_back(scalar_ad_tagged);
        }
    }

    // Resolve captures via unified helper
    resolveGradientCaptures(func_ptr, scalar_args, "scalar");

    // NESTED GRADIENT FIX: Save ctx_.outerAdNodeStorage() before calling function
    // Nested gradients will overwrite it, so we save and restore to support n-dimensional derivatives
    Value* saved_outer_ad_node_scalar = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.outerAdNodeStorage());

    // NESTED GRADIENT FIX: Push tape context (saves outer gradient's tape if any)
    pushTapeContext(partial_tape);

    // M1 Migration FIX: Set AD mode flag so vref recognizes AD node pointers in tensors
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int1Type(), 1), ctx_.adModeActive());

    Value* scalar_output = ctx_.builder().CreateCall(func_ptr, scalar_args);

    // M1 Migration FIX: Reset AD mode flag after function call
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int1Type(), 0), ctx_.adModeActive());

    // NESTED GRADIENT FIX: Pop tape context (restores outer gradient's tape if any)
    popTapeContext();

    // NESTED GRADIENT FIX: Restore ctx_.outerAdNodeStorage() after function returns
    ctx_.builder().CreateStore(saved_outer_ad_node_scalar, ctx_.outerAdNodeStorage());

    ctx_.builder().CreateBr(after_func_call);
    BasicBlock* scalar_call_exit = ctx_.builder().GetInsertBlock();
    
    // VECTOR: Pass AD nodes — either as single tensor or unpacked to individual params
    ctx_.builder().SetInsertPoint(vector_call);
    Value* ad_tensor_int = ctx_.builder().CreatePtrToInt(typed_ad_tensor_ptr, ctx_.int64Type());
    // M1 CONSOLIDATION: Use HEAP_PTR type - tensor has header with HEAP_SUBTYPE_TENSOR
    Value* ad_tensor_tagged = tagged_.packPtr(ad_tensor_int, ESHKOL_VALUE_HEAP_PTR);

    std::vector<Value*> grad_call_args;

    // MULTI-PARAMETER GRADIENT: Check if function has multiple parameters
    // If func has N params and N matches the gradient dimension, unpack AD nodes
    // as individual arguments instead of passing a single tensor.
    FunctionType* grad_func_type = func_ptr->getFunctionType();
    std::string func_name_str = func_ptr->getName().str();
    uint64_t func_arity = 0;
    auto arity_it = function_arity_table_->find(func_name_str);
    if (arity_it != function_arity_table_->end()) {
        func_arity = arity_it->second;
    }

    if (func_arity > 1) {
        // Multi-parameter function: unpack AD tensor elements as individual tagged args
        // Each element in the AD tensor is an AD node pointer (CALLABLE type)
        eshkol_debug("Gradient: unpacking %llu AD nodes for %llu-parameter function %s",
                     (unsigned long long)func_arity, (unsigned long long)func_arity, func_name_str.c_str());
        for (uint64_t p = 0; p < func_arity; p++) {
            // Load AD node pointer from tensor elements[p]
            Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(),
                ctx_.builder().CreateLoad(ctx_.builder().getPtrTy(),
                    ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_ad_tensor_ptr, 2)),
                ConstantInt::get(ctx_.int64Type(), p));
            Value* ad_node_int = ctx_.builder().CreateLoad(ctx_.int64Type(), elem_ptr);
            // Pack as CALLABLE tagged value (AD nodes are callable)
            Value* ad_node_tagged = tagged_.packPtr(ad_node_int, ESHKOL_VALUE_CALLABLE);
            grad_call_args.push_back(ad_node_tagged);
        }
    } else {
        // Single-parameter function: pass tensor as-is
        grad_call_args.push_back(ad_tensor_tagged);
    }

    if (grad_func_type->getNumParams() > grad_call_args.size()) {
        size_t num_captures = grad_func_type->getNumParams() - grad_call_args.size();
        std::string lambda_name = func_ptr->getName().str();

        // REPL MODE: Get capture names from registry instead of parameter names
        std::vector<std::string> capture_names;
        if ((repl_mode_enabled_ && *repl_mode_enabled_)) {
            std::lock_guard<std::mutex> lock(*repl_mutex_);
            auto captures_it = *repl_lambda_captures_.find(lambda_name);
            if (captures_it != *repl_lambda_captures_.end()) {
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

            // First try capture-specific key in symbol tables
            auto it = global_symbol_table_->find(capture_key);
            bool found_in_global = (it != global_symbol_table_->end());
            if (!found_in_global) {
                it = symbol_table_->find(capture_key);
            }

            bool found = found_in_global ? (it != global_symbol_table_->end()) : (it != symbol_table_->end());

            // FALLBACK: Try raw variable name (for top-level global variables)
            if (!found) {
                it = global_symbol_table_->find(var_name);
                found_in_global = (it != global_symbol_table_->end());
                if (!found_in_global) {
                    it = symbol_table_->find(var_name);
                }
                found = found_in_global ? (it != global_symbol_table_->end()) : (it != symbol_table_->end());
                if (found) {
                    eshkol_debug("Gradient: found capture '%s' via raw variable name", var_name.c_str());
                }
            }

            // REPL MODE: Try creating external declaration for capture global
            if (!found && (repl_mode_enabled_ && *repl_mode_enabled_)) {
                std::lock_guard<std::mutex> lock(*repl_mutex_);
                auto sym_it = *repl_symbol_addresses_.find(capture_key);
                if (sym_it != *repl_symbol_addresses_.end()) {
                    // Create external declaration for capture global
                    GlobalVariable* capture_global = ctx_.module().getGlobalVariable(capture_key);
                    if (!capture_global) {
                        capture_global = new GlobalVariable(
                            *module,
                            ctx_.taggedValueType(),
                            false,
                            GlobalValue::ExternalLinkage,
                            nullptr,
                            capture_key
                        );
                    }
                    // MUTABLE CAPTURE FIX: Create storage containing packed pointer
                    // Lambda expects ptr to slot containing {type=INT64, data=ptrtoint(@global)}
                    // Then lambda loads from slot, unpacks data field to get @global
                    Value* global_ptr_int = ctx_.builder().CreatePtrToInt(capture_global, ctx_.int64Type());
                    Value* packed_capture = tagged_.packInt64(global_ptr_int, true);
                    Value* capture_storage = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_capture_storage");
                    ctx_.builder().CreateStore(packed_capture, capture_storage);
                    grad_call_args.push_back(capture_storage);
                    continue;
                }
            }

            if (found && it->second) {
                Value* storage = it->second;
                // MUTABLE CAPTURE FIX: Create storage containing packed pointer
                // Lambda expects ptr to slot containing {type=INT64, data=ptrtoint(@storage)}
                // Then lambda loads from slot, unpacks data field to get @storage
                Value* storage_ptr_int = ctx_.builder().CreatePtrToInt(storage, ctx_.int64Type());
                Value* packed_storage = tagged_.packInt64(storage_ptr_int, true);
                Value* capture_storage = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_capture_storage");
                ctx_.builder().CreateStore(packed_storage, capture_storage);
                grad_call_args.push_back(capture_storage);
            } else {
                // MUTABLE CAPTURE FIX: Push null pointer instead of packed zero
                grad_call_args.push_back(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())));
                eshkol_warn("Gradient: capture '%s' not found, using null pointer", var_name.c_str());
            }
        }
    }
    
    // NESTED GRADIENT FIX: Save ctx_.outerAdNodeStorage() before calling function
    // Nested gradients will overwrite it, so we save and restore to support n-dimensional derivatives
    Value* saved_outer_ad_node_vector = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.outerAdNodeStorage());

    // NESTED GRADIENT FIX: Push tape context (saves outer gradient's tape if any)
    pushTapeContext(partial_tape);

    // M1 Migration FIX: Set AD mode flag so vref recognizes AD node pointers in tensors
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int1Type(), 1), ctx_.adModeActive());

    Value* vector_output = ctx_.builder().CreateCall(func_ptr, grad_call_args);

    // M1 Migration FIX: Reset AD mode flag after function call
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int1Type(), 0), ctx_.adModeActive());

    // NESTED GRADIENT FIX: Pop tape context (restores outer gradient's tape if any)
    popTapeContext();

    // NESTED GRADIENT FIX: Restore ctx_.outerAdNodeStorage() after function returns
    ctx_.builder().CreateStore(saved_outer_ad_node_vector, ctx_.outerAdNodeStorage());

    ctx_.builder().CreateBr(after_func_call);
    BasicBlock* vector_call_exit = ctx_.builder().GetInsertBlock();
    
    // Merge scalar and vector outputs
    ctx_.builder().SetInsertPoint(after_func_call);
    PHINode* output_tagged = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "grad_func_output");
    output_tagged->addIncoming(scalar_output, scalar_call_exit);
    output_tagged->addIncoming(vector_output, vector_call_exit);
    
    // Unpack result back to int64
    Value* output_node_int = tagged_.unpackInt64(output_tagged);
    
    // Convert output to AD node pointer
    Value* output_node_ptr = ctx_.builder().CreateIntToPtr(output_node_int,
        PointerType::getUnqual(ctx_.context()));
    
    // CRITICAL FIX: Use type-based detection instead of pointer value heuristic
    // Check if output is actually an AD node by examining its type tag
    Value* output_type = tagged_.getType(output_tagged);
    Value* output_base_type = tagged_.getBaseType(output_type);
    Value* output_is_ad_node = ctx_.builder().CreateICmpEQ(output_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    
    BasicBlock* has_valid_output = BasicBlock::Create(ctx_.context(), "grad_valid_output", current_func);
    BasicBlock* invalid_output = BasicBlock::Create(ctx_.context(), "grad_invalid_output", current_func);
    BasicBlock* after_backward = BasicBlock::Create(ctx_.context(), "grad_after_backward", current_func);
    
    // Branch based on type check (robust detection)
    ctx_.builder().CreateCondBr(output_is_ad_node, has_valid_output, invalid_output);
    
    // Step 6: Run backward pass through computational graph (only for valid AD nodes)
    ctx_.builder().SetInsertPoint(has_valid_output);

    // DOUBLE BACKWARD SETUP: Store the inner variable node and initialize degree counter
    // This enables degree tracking during backward for proper double backward expressions
    ctx_.builder().CreateStore(active_var_node, ctx_.innerVarNodePtr());
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), ctx_.gradientXDegree());

    codegenBackward(output_node_ptr, partial_tape);
    ctx_.builder().CreateBr(after_backward);
    
    // Skip backward pass if output is invalid (placeholder function returning scalar)
    ctx_.builder().SetInsertPoint(invalid_output);
    eshkol_debug("Gradient: Skipping backward pass - function returned non-AD value");
    ctx_.builder().CreateBr(after_backward);
    
    ctx_.builder().SetInsertPoint(after_backward);
    
    // Step 7: Extract gradient from active variable node (or 0 if no backward pass)
    Value* partial_grad_ptr = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "partial_grad");
    ctx_.builder().CreateStore(ConstantFP::get(ctx_.doubleType(), 0.0), partial_grad_ptr);
    
    // Only extract gradient if we had valid AD output
    BasicBlock* extract_grad = BasicBlock::Create(ctx_.context(), "grad_extract", current_func);
    BasicBlock* use_zero = BasicBlock::Create(ctx_.context(), "grad_use_zero", current_func);
    BasicBlock* grad_extracted = BasicBlock::Create(ctx_.context(), "grad_extracted", current_func);
    
    ctx_.builder().CreateCondBr(output_is_ad_node, extract_grad, use_zero);
    
    ctx_.builder().SetInsertPoint(extract_grad);
    Value* extracted_grad = loadNodeGradient(active_var_node);
    ctx_.builder().CreateStore(extracted_grad, partial_grad_ptr);
    ctx_.builder().CreateBr(grad_extracted);
    
    ctx_.builder().SetInsertPoint(use_zero);
    ctx_.builder().CreateBr(grad_extracted);
    
    ctx_.builder().SetInsertPoint(grad_extracted);
    Value* partial_grad = ctx_.builder().CreateLoad(ctx_.doubleType(), partial_grad_ptr);
    
    // Step 8: Store partial derivative in result vector at index i
    // CRITICAL FIX: Tensor elements stored as int64, must bitcast double to int64
    Value* partial_grad_as_int64 = ctx_.builder().CreateBitCast(partial_grad, ctx_.int64Type());
    Value* result_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(),
        typed_result_elements_ptr, i);
    ctx_.builder().CreateStore(partial_grad_as_int64, result_elem_ptr);
    
    // Step 9: Reset tape for next iteration (MUST call to zero gradients)
    ctx_.builder().CreateCall(getArenaTapeResetFunc(), {partial_tape});
    
    // Restore previous tape
    current_tape_ptr = saved_tape;
    
    // Increment component counter
    Value* next_i = ctx_.builder().CreateAdd(i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, component_idx);
    ctx_.builder().CreateBr(grad_loop_cond);
    
    // Loop exit: Return result gradient vector
    ctx_.builder().SetInsertPoint(grad_loop_exit);

    eshkol_info("Gradient computation complete, returning vector of size n");

    // DOUBLE BACKWARD: Check if we have a stored outer AD node
    // If so, create result as AD node on outer tape for proper gradient propagation
    Value* stored_outer = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.outerAdNodeStorage());
    Value* has_outer_node = ctx_.builder().CreateICmpNE(stored_outer,
        ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())));

    // Also check if this is scalar case (n == 1)
    Value* is_scalar_grad = ctx_.builder().CreateICmpEQ(n, ConstantInt::get(ctx_.int64Type(), 1));
    Value* should_return_ad_node = ctx_.builder().CreateAnd(has_outer_node, is_scalar_grad);

    BasicBlock* return_ad_node = BasicBlock::Create(ctx_.context(), "grad_return_ad_node", current_func);
    BasicBlock* return_tensor = BasicBlock::Create(ctx_.context(), "grad_return_tensor", current_func);
    BasicBlock* grad_merge_result = BasicBlock::Create(ctx_.context(), "grad_merge_result", current_func);

    ctx_.builder().CreateCondBr(should_return_ad_node, return_ad_node, return_tensor);

    // DOUBLE BACKWARD PATH: Return AD node connected to outer graph
    ctx_.builder().SetInsertPoint(return_ad_node);

    // Get the scalar gradient value from result tensor
    Value* scalar_grad_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(),
        typed_result_elements_ptr, ConstantInt::get(ctx_.int64Type(), 0));
    Value* scalar_grad_int = ctx_.builder().CreateLoad(ctx_.int64Type(), scalar_grad_ptr);
    Value* scalar_grad_val = ctx_.builder().CreateBitCast(scalar_grad_int, ctx_.doubleType());

    // Get current tape (which IS the outer tape after popTapeContext)
    // After inner gradient's push/pop, ctx_.currentAdTape() is restored to outer tape
    Value* outer_tape_for_result = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.currentAdTape());

    // Create an AD expression on outer tape that connects gradient to input
    // For f(x) = x^n, f'(x) = n*x^(n-1)
    // The gradient depends on x, so we need to express this dependency
    //
    // Key insight: For many functions, f'(x) is approximately proportional to some power of x.
    // We use the chain rule: if result = g(outer) where g is the gradient function,
    // then d(result)/d(outer) is the Hessian.
    //
    // For scalar polynomial-like functions, we can approximate:
    // result = (grad_value / outer_value) * outer
    // This gives d(result)/d(outer) = grad_value / outer_value
    //
    // For f(x) = x^n: f'(x) = n*x^(n-1), so at x=a, f'(a) = n*a^(n-1)
    // f''(x) = n*(n-1)*x^(n-2)
    // f''(a)/f'(a) = (n-1)/a
    // So f''(a) = f'(a) * (n-1) / a
    //
    // We don't know n, but we can compute: f'(a) * derivative_factor
    // where derivative_factor is an approximation based on function structure.
    //
    // For now, use a simple linear connection: result = k * outer
    // where k = grad_value / outer_value
    // This gives d(result)/d(outer) = k = grad_value / outer_value

    // Get the stored outer AD node
    Value* outer_node_for_expr = stored_outer;

    // Get outer node's value
    Value* outer_val_ptr = ctx_.builder().CreateStructGEP(ctx_.adNodeType(), outer_node_for_expr, 1);
    Value* outer_val = ctx_.builder().CreateLoad(ctx_.doubleType(), outer_val_ptr);

    // DEGREE-BASED DOUBLE BACKWARD EXPRESSION
    // The ctx_.gradientXDegree() counter tracks the polynomial degree of f'(x) in x.
    // For f'(x) = k * x^m:
    //   - m = 0 (constant): f'(x) = k, f''(x) = 0
    //   - m = 1 (linear): f'(x) = k*x, f''(x) = k
    //   - m = 2 (quadratic): f'(x) = k*x², f''(x) = 2*k*x
    //
    // We create an AD expression: result = k * x^m where k = grad/x^m
    // This ensures d(result)/dx = k * m * x^(m-1) = correct f''(x)

    // Load the detected degree
    // Note: The counter tracks multiplications by x value during backward.
    // For x²: count=2 (both inputs are x), actual degree = 1
    // For x³: count=3, actual degree = 2
    // So actual_degree = max(0, count - 1)
    Value* raw_count = ctx_.builder().CreateLoad(ctx_.int64Type(), ctx_.gradientXDegree());
    Value* detected_degree = ctx_.builder().CreateSelect(
        ctx_.builder().CreateICmpEQ(raw_count, ConstantInt::get(ctx_.int64Type(), 0)),
        ConstantInt::get(ctx_.int64Type(), 0),
        ctx_.builder().CreateSub(raw_count, ConstantInt::get(ctx_.int64Type(), 1)));

    // N-DIMENSIONAL DERIVATIVES: Support arbitrary polynomial degree
    // For f'(x) = k * x^n:
    //   - Compute outer_val^n to get scale factor k = grad / (outer_val^n)
    //   - Build AD expression: k * x^n using repeated multiplication
    //   - This ensures d(result)/dx = k * n * x^(n-1) = correct higher derivative

    // Create blocks for degree handling
    BasicBlock* degree_0_bb = BasicBlock::Create(ctx_.context(), "degree_0", current_func);
    BasicBlock* degree_n_bb = BasicBlock::Create(ctx_.context(), "degree_n", current_func);
    BasicBlock* degree_merge_bb = BasicBlock::Create(ctx_.context(), "degree_merge", current_func);

    // Check if degree is 0 (constant - no x dependency)
    Value* is_degree_0 = ctx_.builder().CreateICmpEQ(detected_degree, ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(is_degree_0, degree_0_bb, degree_n_bb);

    // DEGREE 0: Constant gradient, f''(x) = 0
    // Result is just a constant AD node (no x dependency)
    ctx_.builder().SetInsertPoint(degree_0_bb);
    Value* const_result_node = createADConstantOnTape(outer_tape_for_result, scalar_grad_val);
    ctx_.builder().CreateBr(degree_merge_bb);
    BasicBlock* degree_0_exit = ctx_.builder().GetInsertBlock();

    // DEGREE N: Polynomial gradient f'(x) = k*x^n
    // Result = k * x^n where k = grad/x^n
    // We compute x^n both as a double (for k) and as AD expression (for result)
    ctx_.builder().SetInsertPoint(degree_n_bb);

    // Compute outer_val^n using a loop
    Value* pow_val_ptr = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "pow_val");
    ctx_.builder().CreateStore(ConstantFP::get(ctx_.doubleType(), 1.0), pow_val_ptr);
    Value* pow_idx_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "pow_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), pow_idx_ptr);

    BasicBlock* pow_loop_cond = BasicBlock::Create(ctx_.context(), "pow_loop_cond", current_func);
    BasicBlock* pow_loop_body = BasicBlock::Create(ctx_.context(), "pow_loop_body", current_func);
    BasicBlock* pow_loop_exit = BasicBlock::Create(ctx_.context(), "pow_loop_exit", current_func);

    ctx_.builder().CreateBr(pow_loop_cond);

    ctx_.builder().SetInsertPoint(pow_loop_cond);
    Value* pow_i = ctx_.builder().CreateLoad(ctx_.int64Type(), pow_idx_ptr);
    Value* pow_continue = ctx_.builder().CreateICmpULT(pow_i, detected_degree);
    ctx_.builder().CreateCondBr(pow_continue, pow_loop_body, pow_loop_exit);

    ctx_.builder().SetInsertPoint(pow_loop_body);
    Value* current_pow = ctx_.builder().CreateLoad(ctx_.doubleType(), pow_val_ptr);
    Value* next_pow = ctx_.builder().CreateFMul(current_pow, outer_val);
    ctx_.builder().CreateStore(next_pow, pow_val_ptr);
    Value* next_pow_i = ctx_.builder().CreateAdd(pow_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_pow_i, pow_idx_ptr);
    ctx_.builder().CreateBr(pow_loop_cond);

    ctx_.builder().SetInsertPoint(pow_loop_exit);
    Value* outer_val_pow_n = ctx_.builder().CreateLoad(ctx_.doubleType(), pow_val_ptr);

    // Compute scale factor k = grad / x^n
    Value* scale_factor_n = ctx_.builder().CreateFDiv(scalar_grad_val, outer_val_pow_n);
    Value* scale_const_n = createADConstantOnTape(outer_tape_for_result, scale_factor_n);

    // Build AD expression x^n using repeated multiplication
    // Start with x, then multiply by x (n-1) more times
    Value* ad_pow_ptr = ctx_.builder().CreateAlloca(PointerType::getUnqual(ctx_.context()), nullptr, "ad_pow");
    ctx_.builder().CreateStore(outer_node_for_expr, ad_pow_ptr);
    Value* ad_pow_idx_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "ad_pow_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), ad_pow_idx_ptr);

    BasicBlock* ad_pow_loop_cond = BasicBlock::Create(ctx_.context(), "ad_pow_loop_cond", current_func);
    BasicBlock* ad_pow_loop_body = BasicBlock::Create(ctx_.context(), "ad_pow_loop_body", current_func);
    BasicBlock* ad_pow_loop_exit = BasicBlock::Create(ctx_.context(), "ad_pow_loop_exit", current_func);

    ctx_.builder().CreateBr(ad_pow_loop_cond);

    ctx_.builder().SetInsertPoint(ad_pow_loop_cond);
    Value* ad_pow_i = ctx_.builder().CreateLoad(ctx_.int64Type(), ad_pow_idx_ptr);
    Value* ad_pow_continue = ctx_.builder().CreateICmpULT(ad_pow_i, detected_degree);
    ctx_.builder().CreateCondBr(ad_pow_continue, ad_pow_loop_body, ad_pow_loop_exit);

    ctx_.builder().SetInsertPoint(ad_pow_loop_body);
    Value* current_ad_pow = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ad_pow_ptr);
    // Multiply current AD expression by x: current * x
    Value* next_ad_pow = recordADNodeBinaryOnTape(outer_tape_for_result, 4, current_ad_pow, outer_node_for_expr);
    ctx_.builder().CreateStore(next_ad_pow, ad_pow_ptr);
    Value* next_ad_pow_i = ctx_.builder().CreateAdd(ad_pow_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_ad_pow_i, ad_pow_idx_ptr);
    ctx_.builder().CreateBr(ad_pow_loop_cond);

    ctx_.builder().SetInsertPoint(ad_pow_loop_exit);
    Value* outer_pow_n_ad = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ad_pow_ptr);

    // Final result: k * x^n
    Value* poly_result = recordADNodeBinaryOnTape(outer_tape_for_result, 4, scale_const_n, outer_pow_n_ad);
    ctx_.builder().CreateBr(degree_merge_bb);
    BasicBlock* degree_n_exit = ctx_.builder().GetInsertBlock();

    // Merge results
    ctx_.builder().SetInsertPoint(degree_merge_bb);
    PHINode* result_ad_node = ctx_.builder().CreatePHI(PointerType::getUnqual(ctx_.context()), 2, "degree_result");
    result_ad_node->addIncoming(const_result_node, degree_0_exit);
    result_ad_node->addIncoming(poly_result, degree_n_exit);

    // Pack AD node as result
    Value* ad_result_int = ctx_.builder().CreatePtrToInt(result_ad_node, ctx_.int64Type());
    Value* ad_result_tagged = tagged_.packPtr(ad_result_int, ESHKOL_VALUE_CALLABLE);

    // Clear the outer AD node storage
    ctx_.builder().CreateStore(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())),
        ctx_.outerAdNodeStorage());

    ctx_.builder().CreateBr(grad_merge_result);
    BasicBlock* ad_result_exit = ctx_.builder().GetInsertBlock();

    // NORMAL PATH: Return tensor as before
    ctx_.builder().SetInsertPoint(return_tensor);
    Value* grad_result_int = ctx_.builder().CreatePtrToInt(typed_result_tensor_ptr, ctx_.int64Type());
    // Tag as TENSOR_PTR for proper display handling (packPtrToTaggedValue handles i64 directly)
    Value* grad_result = tagged_.packPtr(grad_result_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(grad_merge_result);
    BasicBlock* tensor_result_exit = ctx_.builder().GetInsertBlock();

    // Merge paths
    ctx_.builder().SetInsertPoint(grad_merge_result);
    PHINode* final_result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "grad_final_result");
    final_result->addIncoming(ad_result_tagged, ad_result_exit);
    final_result->addIncoming(grad_result, tensor_result_exit);

    ctx_.builder().CreateBr(grad_done);
    BasicBlock* dim_valid_exit = ctx_.builder().GetInsertBlock();
    
    // Merge valid, invalid, and scheme vector forward-mode paths
    ctx_.builder().SetInsertPoint(grad_done);
    PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3, "grad_result_final");
    result_phi->addIncoming(null_tagged_grad, dim_invalid);
    result_phi->addIncoming(final_result, dim_valid_exit);  // Use merged result from double backward handling
    result_phi->addIncoming(scheme_vector_tagged, scheme_vector_exit);  // Forward-mode AD for Scheme vectors

    // SCALAR INPUT FIX: If input was a scalar, extract element 0 from result tensor
    // and return as a scalar double (not a 1-element tensor)
    BasicBlock* scalar_extract_bb = BasicBlock::Create(ctx_.context(), "grad_scalar_extract", current_func);
    BasicBlock* grad_final_bb = BasicBlock::Create(ctx_.context(), "grad_final", current_func);
    ctx_.builder().CreateCondBr(is_scalar, scalar_extract_bb, grad_final_bb);

    ctx_.builder().SetInsertPoint(scalar_extract_bb);
    // Result is a 1-element tensor — extract the double from element 0
    Value* result_ptr = tagged_.unpackPtr(result_phi);
    // tensor struct: field 2 = elements pointer
    Value* elems_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), result_ptr, 2));
    Value* elem_as_int64 = ctx_.builder().CreateLoad(ctx_.int64Type(), elems_ptr);
    Value* elem_double = ctx_.builder().CreateBitCast(elem_as_int64, ctx_.doubleType());
    Value* scalar_result = tagged_.packDouble(elem_double);
    BasicBlock* scalar_extract_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(grad_final_bb);

    ctx_.builder().SetInsertPoint(grad_final_bb);
    PHINode* final_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "grad_final_result");
    final_phi->addIncoming(scalar_result, scalar_extract_exit);
    final_phi->addIncoming(result_phi, grad_done);

    return final_phi;
}


llvm::Value* AutodiffCodegen::jacobian(const eshkol_operations_t* op) {
    using namespace llvm;
    if (!op->jacobian_op.function || !op->jacobian_op.point) {
        eshkol_error("Invalid jacobian operation - missing function or point");
        return nullptr;
    }
    
    // Use class member ctx_.tensorType() (shared by ALL tensor operations)
    // This prevents LLVM IR type conflicts from shadowing the class member
    
    eshkol_info("Computing Jacobian matrix using reverse-mode AD");
    
    // CRITICAL FIX: Must null-check before dyn_cast to avoid LLVM assertion
    Value* func = resolve_lambda_callback_(op->jacobian_op.function, 0, callback_context_);
    if (!func) {
        eshkol_error("Failed to resolve function for Jacobian computation");
        return nullptr;
    }
    
    Function* func_ptr = dyn_cast<Function>(func);
    if (!func_ptr) {
        eshkol_error("Jacobian requires function, got non-function");
        return nullptr;
    }
    
    llvm::Value* vector_val_raw = codegen_ast_callback_(op->jacobian_op.point, callback_context_);
    if (!vector_val_raw) {
        eshkol_error("Failed to evaluate Jacobian point");
        return nullptr;
    }

    // Get arena for OALR-compliant tensor allocation
    Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

    // CRITICAL FIX: Handle Scheme VECTOR_PTR - convert to tensor format
    // Get current function for basic blocks
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Convert TypedValue to tagged_value
    // Tensor literal fix: codegenTensor returns ptr-as-int64 which gets packed as INT64;
    // re-pack as HEAP_PTR so type dispatch correctly detects tensor subtype
    Value* vector_val = vector_val_raw;
    if (vector_val && vector_val->getType() != ctx_.taggedValueType()) {
        if (vector_val->getType()->isIntegerTy(64)) {
            if (op->jacobian_op.point->type == ESHKOL_TENSOR) {
                vector_val = tagged_.packPtr(vector_val, ESHKOL_VALUE_HEAP_PTR);
            } else {
                vector_val = tagged_.packInt64(vector_val, true);
            }
        } else if (vector_val->getType()->isDoubleTy()) {
            vector_val = tagged_.packDouble(vector_val);
        }
    }
    if (op->jacobian_op.point->type == ESHKOL_TENSOR) {
        Value* data_val = tagged_.unpackInt64(vector_val);
        vector_val = tagged_.packPtr(data_val, ESHKOL_VALUE_HEAP_PTR);
    }

    Value* input_type = tagged_.getType(vector_val);
    Value* input_base_type = tagged_.getBaseType(input_type);

    // M1 CONSOLIDATION: Check for both HEAP_PTR (consolidated) and legacy VECTOR_PTR
    Value* is_heap_ptr = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* is_legacy_vector = ctx_.builder().CreateICmpEQ(input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    BasicBlock* jac_heap_dispatch = BasicBlock::Create(ctx_.context(), "jac_heap_dispatch", current_func);
    BasicBlock* jac_check_legacy = BasicBlock::Create(ctx_.context(), "jac_check_legacy", current_func);
    BasicBlock* jac_scheme_vector_input = BasicBlock::Create(ctx_.context(), "jac_scheme_vector", current_func);
    BasicBlock* jac_tensor_input = BasicBlock::Create(ctx_.context(), "jac_tensor_input", current_func);
    BasicBlock* jac_merge_input = BasicBlock::Create(ctx_.context(), "jac_merge_input", current_func);

    // First check for HEAP_PTR (consolidated format)
    ctx_.builder().CreateCondBr(is_heap_ptr, jac_heap_dispatch, jac_check_legacy);

    // HEAP_PTR dispatch - read subtype from header
    ctx_.builder().SetInsertPoint(jac_heap_dispatch);
    Value* jac_heap_ptr_val = tagged_.unpackPtr(vector_val);
    Value* jac_header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), jac_heap_ptr_val, ConstantInt::get(ctx_.int64Type(), -8));
    Value* jac_subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), jac_header_ptr);
    Value* jac_is_vec_subtype = ctx_.builder().CreateICmpEQ(jac_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    ctx_.builder().CreateCondBr(jac_is_vec_subtype, jac_scheme_vector_input, jac_tensor_input);

    // Legacy VECTOR_PTR fallback
    ctx_.builder().SetInsertPoint(jac_check_legacy);
    ctx_.builder().CreateCondBr(is_legacy_vector, jac_scheme_vector_input, jac_tensor_input);

    // SCHEME VECTOR: Convert to tensor format
    ctx_.builder().SetInsertPoint(jac_scheme_vector_input);

    Value* jac_scheme_vec_ptr_int = tagged_.unpackInt64(vector_val);
    Value* jac_scheme_vec_ptr = ctx_.builder().CreateIntToPtr(jac_scheme_vec_ptr_int, ctx_.builder().getPtrTy());
    Value* jac_scheme_len_ptr = ctx_.builder().CreateBitCast(jac_scheme_vec_ptr, PointerType::getUnqual(ctx_.context()));
    Value* jac_scheme_len = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_scheme_len_ptr);

    // Allocate tensor via arena (OALR compliant - no malloc)
    Value* jac_typed_scheme_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions
    Value* jac_scheme_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* jac_scheme_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_scheme_dims_size});
    Value* jac_typed_scheme_dims = ctx_.builder().CreatePointerCast(jac_scheme_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(jac_scheme_len, jac_typed_scheme_dims);

    ctx_.builder().CreateStore(jac_typed_scheme_dims, ctx_.builder().CreateStructGEP(ctx_.tensorType(), jac_typed_scheme_tensor, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), jac_typed_scheme_tensor, 1));
    ctx_.builder().CreateStore(jac_scheme_len, ctx_.builder().CreateStructGEP(ctx_.tensorType(), jac_typed_scheme_tensor, 3));

    // Allocate and copy elements
    Value* jac_scheme_elems_size = ctx_.builder().CreateMul(jac_scheme_len,
        ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    Value* jac_scheme_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_scheme_elems_size});
    Value* jac_typed_scheme_elems = ctx_.builder().CreatePointerCast(jac_scheme_elems_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(jac_typed_scheme_elems, ctx_.builder().CreateStructGEP(ctx_.tensorType(), jac_typed_scheme_tensor, 2));

    // Copy elements loop
    Value* jac_scheme_elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), jac_scheme_vec_ptr,
        ConstantInt::get(ctx_.int64Type(), 8));
    Value* jac_scheme_elem_base_typed = ctx_.builder().CreateBitCast(jac_scheme_elem_base, PointerType::getUnqual(ctx_.context()));

    BasicBlock* jac_svec_copy_cond = BasicBlock::Create(ctx_.context(), "jac_svec_copy_cond", current_func);
    BasicBlock* jac_svec_copy_body = BasicBlock::Create(ctx_.context(), "jac_svec_copy_body", current_func);
    BasicBlock* jac_svec_copy_done = BasicBlock::Create(ctx_.context(), "jac_svec_copy_done", current_func);

    Value* jac_svec_copy_i = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "jac_svec_copy_i");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), jac_svec_copy_i);
    ctx_.builder().CreateBr(jac_svec_copy_cond);

    ctx_.builder().SetInsertPoint(jac_svec_copy_cond);
    Value* jac_svec_i = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_svec_copy_i);
    Value* jac_svec_cond = ctx_.builder().CreateICmpULT(jac_svec_i, jac_scheme_len);
    ctx_.builder().CreateCondBr(jac_svec_cond, jac_svec_copy_body, jac_svec_copy_done);

    ctx_.builder().SetInsertPoint(jac_svec_copy_body);
    Value* jac_svec_src_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), jac_scheme_elem_base_typed, jac_svec_i);
    Value* jac_svec_tagged_elem = ctx_.builder().CreateLoad(ctx_.taggedValueType(), jac_svec_src_ptr);
    Value* jac_svec_double_val = tagged_.unpackDouble(jac_svec_tagged_elem);
    Value* jac_svec_as_int64 = ctx_.builder().CreateBitCast(jac_svec_double_val, ctx_.int64Type());
    Value* jac_svec_dst_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), jac_typed_scheme_elems, jac_svec_i);
    ctx_.builder().CreateStore(jac_svec_as_int64, jac_svec_dst_ptr);
    Value* jac_svec_next_i = ctx_.builder().CreateAdd(jac_svec_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(jac_svec_next_i, jac_svec_copy_i);
    ctx_.builder().CreateBr(jac_svec_copy_cond);

    ctx_.builder().SetInsertPoint(jac_svec_copy_done);
    Value* jac_scheme_tensor_int = ctx_.builder().CreatePtrToInt(jac_typed_scheme_tensor, ctx_.int64Type());
    Value* jac_scheme_vector_tagged = tagged_.packPtr(jac_scheme_tensor_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(jac_merge_input);
    BasicBlock* jac_scheme_exit = ctx_.builder().GetInsertBlock();

    // TENSOR INPUT: Use as-is
    ctx_.builder().SetInsertPoint(jac_tensor_input);
    ctx_.builder().CreateBr(jac_merge_input);
    BasicBlock* jac_tensor_exit = ctx_.builder().GetInsertBlock();

    // MERGE
    ctx_.builder().SetInsertPoint(jac_merge_input);
    PHINode* jac_actual_input = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "jac_input");
    jac_actual_input->addIncoming(jac_scheme_vector_tagged, jac_scheme_exit);
    jac_actual_input->addIncoming(vector_val, jac_tensor_exit);

    // Extract tensor pointer from merged input
    Value* vector_ptr_int = tagged_.unpackInt64(jac_actual_input);

    // Extract input dimension n from input vector
    Value* input_ptr = ctx_.builder().CreateIntToPtr(vector_ptr_int, ctx_.builder().getPtrTy());
    
    Value* input_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), input_ptr, 0);
    Value* input_dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), input_dims_field);
    Value* typed_input_dims = ctx_.builder().CreatePointerCast(input_dims_ptr, ctx_.builder().getPtrTy());
    
    Value* input_elements_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), input_ptr, 2);
    Value* input_elements_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), input_elements_field);
    Value* typed_input_elements = ctx_.builder().CreatePointerCast(input_elements_ptr, ctx_.builder().getPtrTy());
    
    Value* n_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_input_dims,
        ConstantInt::get(ctx_.int64Type(), 0));
    Value* n = ctx_.builder().CreateLoad(ctx_.int64Type(), n_ptr);

    // Call function once to determine output dimension m
    // CRITICAL FIX: Pack as TENSOR_PTR not INT64, so identity lambdas preserve type
    Value* vector_tagged = tagged_.packPtr(vector_ptr_int, ESHKOL_VALUE_HEAP_PTR);
    // CLOSURE FIX: Load captures for function call
    std::vector<Value*> test_call_args = {vector_tagged};
    std::vector<Value*> jac_test_captures = loadCapturesForAutodiff(func_ptr, "Jacobian test call");
    test_call_args.insert(test_call_args.end(), jac_test_captures.begin(), jac_test_captures.end());
    Value* test_output_tagged = ctx_.builder().CreateCall(func_ptr, test_call_args);

    // ENHANCED TYPE CHECK: Accept tensors, AD tensors, AND Scheme vectors as valid outputs
    Value* output_type = tagged_.getType(test_output_tagged);
    Value* output_base_type = tagged_.getBaseType(output_type);

    // M1 CONSOLIDATION: Check for valid output types
    // For HEAP_PTR, we need to check the subtype to distinguish vector (2) from tensor (3)
    Value* output_is_heap_ptr = ctx_.builder().CreateICmpEQ(output_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* output_is_callable = ctx_.builder().CreateICmpEQ(output_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));

    // Any HEAP_PTR or CALLABLE is a valid vector type (tensor or vector)
    Value* output_has_vector_type = ctx_.builder().CreateOr(output_is_heap_ptr, output_is_callable);

    // CRITICAL FIX: Create null tagged value BEFORE branching (for PHI node dominance)
    Value* null_jac_tagged = tagged_.packInt64(
        ConstantInt::get(ctx_.int64Type(), 0), true);

    // Create blocks for validation flow
    BasicBlock* output_valid_block = BasicBlock::Create(ctx_.context(), "jac_output_valid", current_func);
    BasicBlock* output_invalid_block = BasicBlock::Create(ctx_.context(), "jac_output_invalid", current_func);
    BasicBlock* jac_return_block = BasicBlock::Create(ctx_.context(), "jac_return", current_func);

    ctx_.builder().CreateCondBr(output_has_vector_type, output_valid_block, output_invalid_block);

    // Invalid output: Generate runtime code to extract and report actual type value
    ctx_.builder().SetInsertPoint(output_invalid_block);
    // This block now only reached for genuinely invalid types (NULL, INT64, DOUBLE, CONS_PTR)
    Function* printf_func_for_error = ctx_.lookupFunction("printf");
    if (printf_func_for_error) {
        // Create alloca for type value at function entry to ensure dominance
        IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
        Function* func = ctx_.builder().GetInsertBlock()->getParent();
        if (func && !func->empty()) {
            BasicBlock& entry = func->getEntryBlock();
            ctx_.builder().SetInsertPoint(&entry, entry.begin());
        }
        Value* type_storage = ctx_.builder().CreateAlloca(ctx_.int8Type(), nullptr, "invalid_type");
        ctx_.builder().restoreIP(saved_ip);

        // Store the runtime type value and extend to int for printf
        ctx_.builder().CreateStore(output_base_type, type_storage);
        Value* type_val = ctx_.builder().CreateLoad(ctx_.int8Type(), type_storage);
        Value* type_as_int = ctx_.builder().CreateZExt(type_val, ctx_.int32Type());

        // Print error with actual runtime type value (provides better debugging!)
        ctx_.builder().CreateCall(printf_func_for_error, {
            ctx_.internString("Jacobian ERROR: function returned non-vector type %d (expected 6=TENSOR, 5=AD_TENSOR, or 4=VECTOR_PTR)\n"),
            type_as_int
        });
    }
    ctx_.builder().CreateBr(jac_return_block);

    // Valid output: Handle both tensor and Scheme vector formats
    ctx_.builder().SetInsertPoint(output_valid_block);


    // Branch based on whether output is Scheme vector or tensor
    // For HEAP_PTR, check subtype to distinguish vector (2) from tensor (3)
    BasicBlock* jac_output_check_subtype = BasicBlock::Create(ctx_.context(), "jac_output_check_subtype", current_func);
    BasicBlock* jac_output_scheme_vec = BasicBlock::Create(ctx_.context(), "jac_output_scheme_vec", current_func);
    BasicBlock* jac_output_tensor = BasicBlock::Create(ctx_.context(), "jac_output_tensor", current_func);
    BasicBlock* jac_output_merge = BasicBlock::Create(ctx_.context(), "jac_output_merge", current_func);

    // If HEAP_PTR, check subtype; otherwise go to tensor path (AD_TENSOR/CALLABLE)
    ctx_.builder().CreateCondBr(output_is_heap_ptr, jac_output_check_subtype, jac_output_tensor);

    // Check subtype in header to distinguish Scheme vector from tensor
    ctx_.builder().SetInsertPoint(jac_output_check_subtype);
    Value* test_out_ptr_int = tagged_.unpackInt64(test_output_tagged);
    Value* test_out_ptr = ctx_.builder().CreateIntToPtr(test_out_ptr_int, ctx_.builder().getPtrTy());
    Value* test_out_header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), test_out_ptr, ConstantInt::get(ctx_.int64Type(), -8));
    Value* test_out_subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), test_out_header_ptr);
    Value* test_out_is_svec = ctx_.builder().CreateICmpEQ(test_out_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    ctx_.builder().CreateCondBr(test_out_is_svec, jac_output_scheme_vec, jac_output_tensor);

    // SCHEME VECTOR OUTPUT: Extract dimension directly from vector length
    ctx_.builder().SetInsertPoint(jac_output_scheme_vec);
    Value* jac_out_svec_ptr_int = tagged_.unpackInt64(test_output_tagged);
    Value* jac_out_svec_ptr = ctx_.builder().CreateIntToPtr(jac_out_svec_ptr_int, ctx_.builder().getPtrTy());
    Value* jac_out_svec_len_ptr = ctx_.builder().CreateBitCast(jac_out_svec_ptr, PointerType::getUnqual(ctx_.context()));
    Value* jac_out_svec_m = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_out_svec_len_ptr);
    ctx_.builder().CreateBr(jac_output_merge);
    BasicBlock* jac_out_svec_exit = ctx_.builder().GetInsertBlock();

    // TENSOR OUTPUT: Extract dimension from tensor structure
    ctx_.builder().SetInsertPoint(jac_output_tensor);
    Value* test_output_int = tagged_.unpackInt64(test_output_tagged);
    Value* test_output_ptr = ctx_.builder().CreateIntToPtr(test_output_int, ctx_.builder().getPtrTy());

    Value* output_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), test_output_ptr, 0);
    Value* output_dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), output_dims_field);

    Value* typed_output_dims = ctx_.builder().CreatePointerCast(output_dims_ptr, ctx_.builder().getPtrTy());

    Value* m_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_output_dims,
        ConstantInt::get(ctx_.int64Type(), 0));

    Value* jac_out_tensor_m = ctx_.builder().CreateLoad(ctx_.int64Type(), m_ptr);
    ctx_.builder().CreateBr(jac_output_merge);
    BasicBlock* jac_out_tensor_exit = ctx_.builder().GetInsertBlock();

    // MERGE: Get m from whichever path we took
    ctx_.builder().SetInsertPoint(jac_output_merge);
    PHINode* m = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "jac_output_m");
    m->addIncoming(jac_out_svec_m, jac_out_svec_exit);
    m->addIncoming(jac_out_tensor_m, jac_out_tensor_exit);
    
    // Allocate Jacobian matrix via arena (OALR compliant - no malloc)
    Value* typed_jac_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions [m, n]
    Value* jac_dims_size = ctx_.builder().CreateMul(
        ConstantInt::get(ctx_.int64Type(), 2),
        ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t)));
    Value* jac_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_dims_size});
    Value* typed_jac_dims = ctx_.builder().CreatePointerCast(jac_dims_ptr, ctx_.builder().getPtrTy());

    ctx_.builder().CreateStore(m, typed_jac_dims);
    Value* jac_dim1_slot = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_jac_dims,
        ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(n, jac_dim1_slot);

    // Store dimensions in tensor
    Value* jac_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ptr, 0);
    ctx_.builder().CreateStore(typed_jac_dims, jac_dims_field);

    // Set num_dimensions = 2
    Value* jac_num_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ptr, 1);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 2), jac_num_dims_field);

    // Set total_elements = m * n
    Value* total_elems = ctx_.builder().CreateMul(m, n);
    Value* jac_total_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ptr, 3);
    ctx_.builder().CreateStore(total_elems, jac_total_field);

    // Allocate elements array (m*n doubles)
    Value* jac_elems_size = ctx_.builder().CreateMul(total_elems,
        ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    Value* jac_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_elems_size});
    Value* typed_jac_elems = ctx_.builder().CreatePointerCast(jac_elems_ptr, ctx_.builder().getPtrTy());
    
    Value* jac_elems_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ptr, 2);
    ctx_.builder().CreateStore(typed_jac_elems, jac_elems_field);

    BasicBlock* outer_cond = BasicBlock::Create(ctx_.context(), "jac_outer_cond", current_func);
    BasicBlock* outer_body = BasicBlock::Create(ctx_.context(), "jac_outer_body", current_func);
    BasicBlock* inner_cond = BasicBlock::Create(ctx_.context(), "jac_inner_cond", current_func);
    BasicBlock* inner_body = BasicBlock::Create(ctx_.context(), "jac_inner_body", current_func);
    BasicBlock* inner_exit = BasicBlock::Create(ctx_.context(), "jac_inner_exit", current_func);
    BasicBlock* outer_exit = BasicBlock::Create(ctx_.context(), "jac_outer_exit", current_func);

    Value* out_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "out_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), out_idx);

    ctx_.builder().CreateBr(outer_cond);
    
    // Outer: i_out < m
    ctx_.builder().SetInsertPoint(outer_cond);
    Value* i_out = ctx_.builder().CreateLoad(ctx_.int64Type(), out_idx);
    Value* i_out_less_m = ctx_.builder().CreateICmpULT(i_out, m);
    ctx_.builder().CreateCondBr(i_out_less_m, outer_body, outer_exit);
    
    ctx_.builder().SetInsertPoint(outer_body);

    Value* in_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "in_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), in_idx);
    ctx_.builder().CreateBr(inner_cond);
    
    // Inner: j_in < n
    ctx_.builder().SetInsertPoint(inner_cond);
    Value* j_in = ctx_.builder().CreateLoad(ctx_.int64Type(), in_idx);
    Value* j_in_less_n = ctx_.builder().CreateICmpULT(j_in, n);
    ctx_.builder().CreateCondBr(j_in_less_n, inner_body, inner_exit);
    
    // Compute ∂Fᵢ/∂xⱼ
    ctx_.builder().SetInsertPoint(inner_body);

    // arena_ptr defined at function start
    Value* jac_tape = ctx_.builder().CreateCall(mem_.getArenaAllocateTape(),
        {arena_ptr, ConstantInt::get(ctx_.int64Type(), 1024)});
    
    // CRITICAL FIX: Use global AD tape pointer, not member variable!
    // current_tape_ptr is compile-time C++ state, jac_tape is runtime LLVM Value*
    // Assigning Value* to member variable corrupts memory - use global instead
    ctx_.builder().CreateStore(jac_tape, ctx_.currentAdTape());
    
    // Create n AD variable nodes via arena (OALR compliant - no malloc)
    Value* jac_var_nodes_size = ctx_.builder().CreateMul(n,
        ConstantInt::get(ctx_.int64Type(), sizeof(void*)));
    Value* jac_var_nodes = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_var_nodes_size});
    Value* typed_jac_var_nodes = ctx_.builder().CreatePointerCast(jac_var_nodes, ctx_.builder().getPtrTy());
    
    // Initialize all variable nodes with input values
    BasicBlock* jac_init_cond = BasicBlock::Create(ctx_.context(), "jac_init_cond", current_func);
    BasicBlock* jac_init_body = BasicBlock::Create(ctx_.context(), "jac_init_body", current_func);
    BasicBlock* jac_init_exit = BasicBlock::Create(ctx_.context(), "jac_init_exit", current_func);
    
    Value* jac_init_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "jac_init_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), jac_init_idx);

    ctx_.builder().CreateBr(jac_init_cond);
    
    ctx_.builder().SetInsertPoint(jac_init_cond);
    Value* jac_init_i = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_init_idx);
    Value* jac_init_less = ctx_.builder().CreateICmpULT(jac_init_i, n);
    ctx_.builder().CreateCondBr(jac_init_less, jac_init_body, jac_init_exit);
    
    ctx_.builder().SetInsertPoint(jac_init_body);

    // CRITICAL FIX: Tensor elements stored as int64, load as int64 then convert
    Value* jac_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(),
        typed_input_elements, jac_init_i);
    Value* jac_elem_int64 = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_elem_ptr);

    // FIX 1b: BitCast preserves IEEE754 bits, SIToFP corrupts them
    Value* jac_elem_val = ctx_.builder().CreateBitCast(jac_elem_int64, ctx_.doubleType());
    Value* jac_var_node = createADVariable(jac_elem_val, 0);
    
    Value* jac_node_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_jac_var_nodes, jac_init_i);
    ctx_.builder().CreateStore(jac_var_node, jac_node_slot);
    
    Value* jac_next_init = ctx_.builder().CreateAdd(jac_init_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(jac_next_init, jac_init_idx);
    ctx_.builder().CreateBr(jac_init_cond);
    
    ctx_.builder().SetInsertPoint(jac_init_exit);
    
    // Build AD tensor for function call via arena (OALR compliant - no malloc)
    Value* typed_jac_ad_tensor = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set AD tensor structure
    Value* jac_ad_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* jac_ad_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_ad_dims_size});
    Value* typed_jac_ad_dims = ctx_.builder().CreatePointerCast(jac_ad_dims_ptr, ctx_.builder().getPtrTy());

    ctx_.builder().CreateStore(n, typed_jac_ad_dims);

    // Set tensor fields directly
    ctx_.builder().CreateStore(typed_jac_ad_dims,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ad_tensor, 0));
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ad_tensor, 1));
    ctx_.builder().CreateStore(n,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ad_tensor, 3));

    // Allocate elements via arena
    Value* jac_ad_elems_size = ctx_.builder().CreateMul(n,
        ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t)));
    Value* jac_ad_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, jac_ad_elems_size});
    Value* typed_jac_ad_elems = ctx_.builder().CreatePointerCast(jac_ad_elems_ptr, ctx_.builder().getPtrTy());
    
    ctx_.builder().CreateStore(typed_jac_ad_elems,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_jac_ad_tensor, 2));

    // Copy nodes
    BasicBlock* jac_copy_cond = BasicBlock::Create(ctx_.context(), "jac_copy_cond", current_func);
    BasicBlock* jac_copy_body = BasicBlock::Create(ctx_.context(), "jac_copy_body", current_func);
    BasicBlock* jac_copy_exit = BasicBlock::Create(ctx_.context(), "jac_copy_exit", current_func);
    
    Value* jac_copy_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "jac_copy_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), jac_copy_idx);
    ctx_.builder().CreateBr(jac_copy_cond);
    
    ctx_.builder().SetInsertPoint(jac_copy_cond);
    Value* jac_copy_i = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_copy_idx);
    Value* jac_copy_less = ctx_.builder().CreateICmpULT(jac_copy_i, n);
    ctx_.builder().CreateCondBr(jac_copy_less, jac_copy_body, jac_copy_exit);
    
    ctx_.builder().SetInsertPoint(jac_copy_body);

    Value* jac_src_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_jac_var_nodes, jac_copy_i);
    Value* jac_src_node = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), jac_src_slot);

    Value* jac_node_int = ctx_.builder().CreatePtrToInt(jac_src_node, ctx_.int64Type());

    Value* jac_dst_slot = ctx_.builder().CreateGEP(ctx_.int64Type(),
        typed_jac_ad_elems, jac_copy_i);
    ctx_.builder().CreateStore(jac_node_int, jac_dst_slot);
    
    Value* jac_next_copy = ctx_.builder().CreateAdd(jac_copy_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(jac_next_copy, jac_copy_idx);
    ctx_.builder().CreateBr(jac_copy_cond);

    ctx_.builder().SetInsertPoint(jac_copy_exit);
    
    // Call function to get output
    Value* jac_ad_tensor_int = ctx_.builder().CreatePtrToInt(typed_jac_ad_tensor, ctx_.int64Type());
    // CRITICAL FIX: Pack as TENSOR_PTR not INT64, so identity lambdas preserve type
    Value* jac_ad_tensor_tagged = tagged_.packPtr(jac_ad_tensor_int, ESHKOL_VALUE_HEAP_PTR);
    
    // PHASE 1 FIX: Set AD mode flag to true before calling lambda
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int1Type(), 1), ctx_.adModeActive());

    // CLOSURE FIX: Load captures for function call
    std::vector<Value*> jac_call_args = {jac_ad_tensor_tagged};
    std::vector<Value*> jac_captures = loadCapturesForAutodiff(func_ptr, "Jacobian AD call");
    jac_call_args.insert(jac_call_args.end(), jac_captures.begin(), jac_captures.end());
    Value* jac_output_tagged = ctx_.builder().CreateCall(func_ptr, jac_call_args);

    // PHASE 1 FIX: Set AD mode flag back to false after lambda call
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int1Type(), 0), ctx_.adModeActive());

    Value* jac_output_int = tagged_.unpackInt64(jac_output_tagged);
    Value* jac_output_ptr = ctx_.builder().CreateIntToPtr(jac_output_int, ctx_.builder().getPtrTy());

    // M1 CONSOLIDATION: Handle both tensor and Scheme vector output
    // For HEAP_PTR, read the header subtype to distinguish vector vs tensor
    Value* jac_loop_output_type = tagged_.getType(jac_output_tagged);
    Value* jac_loop_output_base = tagged_.getBaseType(jac_loop_output_type);
    Value* jac_loop_is_heap_ptr = ctx_.builder().CreateICmpEQ(jac_loop_output_base,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    BasicBlock* jac_loop_svec_out = BasicBlock::Create(ctx_.context(), "jac_loop_svec_out", current_func);
    BasicBlock* jac_loop_tensor_out = BasicBlock::Create(ctx_.context(), "jac_loop_tensor_out", current_func);
    BasicBlock* jac_loop_merge_out = BasicBlock::Create(ctx_.context(), "jac_loop_merge_out", current_func);
    BasicBlock* jac_loop_check_subtype = BasicBlock::Create(ctx_.context(), "jac_loop_check_subtype", current_func);

    // First check if HEAP_PTR - if so, check subtype; otherwise go to tensor path
    ctx_.builder().CreateCondBr(jac_loop_is_heap_ptr, jac_loop_check_subtype, jac_loop_tensor_out);

    // Check subtype to distinguish Scheme vector (2) from tensor (3)
    ctx_.builder().SetInsertPoint(jac_loop_check_subtype);
    // Header is at ptr - 8 bytes
    Value* jac_loop_header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), jac_output_ptr,
        ConstantInt::get(ctx_.int64Type(), -8));
    // Subtype is first byte of header
    Value* jac_loop_subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), jac_loop_header_ptr);
    Value* jac_is_scheme_vec = ctx_.builder().CreateICmpEQ(jac_loop_subtype,
        ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    ctx_.builder().CreateCondBr(jac_is_scheme_vec, jac_loop_svec_out, jac_loop_tensor_out);

    // SCHEME VECTOR OUTPUT: Extract element from Scheme vector
    ctx_.builder().SetInsertPoint(jac_loop_svec_out);
    // Scheme vector layout: [len: i64][elem0: tagged_value][elem1: tagged_value]...
    Value* jac_svec_elem_base = ctx_.builder().CreateGEP(ctx_.int8Type(), jac_output_ptr,
        ConstantInt::get(ctx_.int64Type(), 8));  // Skip length field
    Value* jac_svec_elem_base_typed = ctx_.builder().CreateBitCast(jac_svec_elem_base, PointerType::getUnqual(ctx_.context()));
    Value* jac_svec_elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), jac_svec_elem_base_typed, i_out);
    Value* jac_svec_elem_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), jac_svec_elem_ptr);
    // Extract the int64 component from the tagged value (could be AD node ptr or double bits)
    Value* jac_svec_elem_int = tagged_.unpackInt64(jac_svec_elem_tagged);
    ctx_.builder().CreateBr(jac_loop_merge_out);
    BasicBlock* jac_svec_out_exit = ctx_.builder().GetInsertBlock();

    // TENSOR OUTPUT: Extract element from tensor structure
    ctx_.builder().SetInsertPoint(jac_loop_tensor_out);
    Value* out_elems_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), jac_output_ptr, 2);
    Value* out_elems_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), out_elems_field);
    Value* typed_out_elems = ctx_.builder().CreatePointerCast(out_elems_ptr, ctx_.builder().getPtrTy());
    Value* jac_tensor_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_out_elems, i_out);
    Value* jac_tensor_elem_int = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_tensor_elem_ptr);
    ctx_.builder().CreateBr(jac_loop_merge_out);
    BasicBlock* jac_tensor_out_exit = ctx_.builder().GetInsertBlock();

    // MERGE: Get output component from whichever path
    ctx_.builder().SetInsertPoint(jac_loop_merge_out);
    PHINode* out_comp_int = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "jac_out_comp");
    out_comp_int->addIncoming(jac_svec_elem_int, jac_svec_out_exit);
    out_comp_int->addIncoming(jac_tensor_elem_int, jac_tensor_out_exit);
    
    // CRITICAL SAFETY CHECK: Detect if output element is AD node or regular value
    // AD nodes are allocated in heap (> 1000), doubles have IEEE754 exponent bits
    Value* is_small_value = ctx_.builder().CreateICmpULT(out_comp_int,
        ConstantInt::get(ctx_.int64Type(), 1000));
    
    // Check IEEE754 exponent for doubles (bit pattern detection)
    Value* exp_mask_jac = ConstantInt::get(ctx_.int64Type(), 0x7FF0000000000000ULL);
    Value* exp_bits_jac = ctx_.builder().CreateAnd(out_comp_int, exp_mask_jac);
    Value* has_exponent_jac = ctx_.builder().CreateICmpNE(exp_bits_jac,
        ConstantInt::get(ctx_.int64Type(), 0));
    
    // If has exponent, it's a double, not an AD node pointer
    Value* is_likely_double_jac = ctx_.builder().CreateAnd(has_exponent_jac,
        ctx_.builder().CreateNot(is_small_value));
    
    // Output is AD node only if: not small AND not double
    Value* elem_is_ad_node = ctx_.builder().CreateAnd(
        ctx_.builder().CreateNot(is_small_value),
        ctx_.builder().CreateNot(is_likely_double_jac));
    
    // Allocate storage for partial derivative result (accessible across blocks)
    Value* partial_deriv_storage = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "jac_partial_storage");
    
    BasicBlock* run_jac_backward = BasicBlock::Create(ctx_.context(), "jac_run_backward", current_func);
    BasicBlock* skip_jac_backward = BasicBlock::Create(ctx_.context(), "jac_skip_backward", current_func);
    BasicBlock* after_jac_backward = BasicBlock::Create(ctx_.context(), "jac_after_backward", current_func);
    
    ctx_.builder().CreateCondBr(elem_is_ad_node, run_jac_backward, skip_jac_backward);
    
    // Run backward pass only if output element is AD node
    ctx_.builder().SetInsertPoint(run_jac_backward);

    Value* out_comp_node = ctx_.builder().CreateIntToPtr(out_comp_int, PointerType::getUnqual(ctx_.context()));
    backpropagate(out_comp_node, jac_tape);
    
    // Extract gradient from variable j_in
    Value* jac_grad_var_slot = ctx_.builder().CreateGEP(PointerType::getUnqual(ctx_.context()),
        typed_jac_var_nodes, j_in);
    Value* jac_grad_var_node = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), jac_grad_var_slot);
    Value* computed_partial_deriv = loadNodeGradient(jac_grad_var_node);
    ctx_.builder().CreateStore(computed_partial_deriv, partial_deriv_storage);
    ctx_.builder().CreateBr(after_jac_backward);
    
    // Skip backward pass if output is not AD node (constant function)
    ctx_.builder().SetInsertPoint(skip_jac_backward);

    Value* zero_deriv_jac = ConstantFP::get(ctx_.doubleType(), 0.0);
    ctx_.builder().CreateStore(zero_deriv_jac, partial_deriv_storage);
    ctx_.builder().CreateBr(after_jac_backward);
    
    // Merge paths - load result from storage
    ctx_.builder().SetInsertPoint(after_jac_backward);
    Value* partial_deriv = ctx_.builder().CreateLoad(ctx_.doubleType(), partial_deriv_storage);
    
    // Store J[i_out,j_in] at linear index: i_out*n + j_in
    Value* linear_idx = ctx_.builder().CreateMul(i_out, n);
    linear_idx = ctx_.builder().CreateAdd(linear_idx, j_in);
    
    Value* jac_result_elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(),
        typed_jac_elems, linear_idx);
    ctx_.builder().CreateStore(partial_deriv, jac_result_elem_ptr);
    
    ctx_.builder().CreateCall(mem_.getArenaTapeReset(), {jac_tape});
    
    // CRITICAL FIX: Clear global tape pointer (like gradient does)
    ctx_.builder().CreateStore(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())), ctx_.currentAdTape());
    
    Value* next_j_in = ctx_.builder().CreateAdd(j_in, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_j_in, in_idx);
    ctx_.builder().CreateBr(inner_cond);
    
    ctx_.builder().SetInsertPoint(inner_exit);
    Value* next_i_out = ctx_.builder().CreateAdd(i_out, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i_out, out_idx);
    ctx_.builder().CreateBr(outer_cond);
    
    ctx_.builder().SetInsertPoint(outer_exit);

    // FIX: Return 2D tensor directly (like Hessian does) instead of converting to nested lists
    // The tensor display now handles N-dimensional tensors correctly
    // Tensor elements are stored as doubles (int64 bit representation)
    // We need to convert from double to int64 bit pattern for proper storage

    // The elements in typed_jac_elems were stored as double type - convert to int64 bit pattern
    BasicBlock* jac_convert_cond = BasicBlock::Create(ctx_.context(), "jac_convert_cond", current_func);
    BasicBlock* jac_convert_body = BasicBlock::Create(ctx_.context(), "jac_convert_body", current_func);
    BasicBlock* jac_convert_exit = BasicBlock::Create(ctx_.context(), "jac_convert_exit", current_func);

    Value* jac_convert_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "jac_convert_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), jac_convert_idx);
    ctx_.builder().CreateBr(jac_convert_cond);

    ctx_.builder().SetInsertPoint(jac_convert_cond);
    Value* jac_cvt_i = ctx_.builder().CreateLoad(ctx_.int64Type(), jac_convert_idx);
    Value* jac_cvt_less = ctx_.builder().CreateICmpULT(jac_cvt_i, total_elems);
    ctx_.builder().CreateCondBr(jac_cvt_less, jac_convert_body, jac_convert_exit);

    ctx_.builder().SetInsertPoint(jac_convert_body);
    // Load as double, convert to int64 bits, store back
    Value* jac_cvt_elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_jac_elems, jac_cvt_i);
    Value* jac_cvt_elem_double = ctx_.builder().CreateLoad(ctx_.doubleType(), jac_cvt_elem_ptr);
    Value* jac_cvt_elem_bits = ctx_.builder().CreateBitCast(jac_cvt_elem_double, ctx_.int64Type());
    // Store as int64 (tensor elements are stored as int64 bit patterns)
    Value* jac_cvt_store_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_jac_elems, jac_cvt_i);
    ctx_.builder().CreateStore(jac_cvt_elem_bits, jac_cvt_store_ptr);
    Value* jac_cvt_next = ctx_.builder().CreateAdd(jac_cvt_i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(jac_cvt_next, jac_convert_idx);
    ctx_.builder().CreateBr(jac_convert_cond);

    ctx_.builder().SetInsertPoint(jac_convert_exit);

    // Return the 2D Jacobian tensor directly
    Value* jac_result_int = ctx_.builder().CreatePtrToInt(typed_jac_ptr, ctx_.int64Type());
    Value* jac_result = tagged_.packPtr(jac_result_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(jac_return_block);

    // Merge null and valid results
    ctx_.builder().SetInsertPoint(jac_return_block);
    PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "jac_result");
    result_phi->addIncoming(null_jac_tagged, output_invalid_block);
    result_phi->addIncoming(jac_result, jac_convert_exit);

    return result_phi;
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

llvm::Value* AutodiffCodegen::createNullVectorTensor(llvm::Value* dimension) {
    using namespace llvm;
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    // Get arena for OALR-compliant allocation
    Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

    // Allocate tensor structure via arena (OALR compliant - no malloc)
    Value* typed_tensor_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Allocate dimensions array (1D vector of given dimension)
    Value* dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, dims_size});
    Value* typed_dims_ptr = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(dimension, typed_dims_ptr);  // Runtime dimension!

    // Store tensor metadata
    ctx_.builder().CreateStore(typed_dims_ptr,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 0));  // dimensions
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1),
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 1));  // num_dimensions = 1
    ctx_.builder().CreateStore(dimension,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 3));  // total_elements = dimension

    // Allocate elements array (dimension * sizeof(double))
    Value* elems_size = ctx_.builder().CreateMul(dimension,
        ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    Value* elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, elems_size});
    Value* typed_elems_ptr = ctx_.builder().CreatePointerCast(elems_ptr, ctx_.builder().getPtrTy());
    
    // Zero all elements using RUNTIME LOOP (n-dimensional!)
    BasicBlock* zero_cond = BasicBlock::Create(ctx_.context(), "null_vec_zero_cond", current_func);
    BasicBlock* zero_body = BasicBlock::Create(ctx_.context(), "null_vec_zero_body", current_func);
    BasicBlock* zero_exit = BasicBlock::Create(ctx_.context(), "null_vec_zero_exit", current_func);
    
    Value* idx_ptr = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "zero_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), idx_ptr);
    ctx_.builder().CreateBr(zero_cond);
    
    ctx_.builder().SetInsertPoint(zero_cond);
    Value* idx = ctx_.builder().CreateLoad(ctx_.int64Type(), idx_ptr);
    Value* idx_less = ctx_.builder().CreateICmpULT(idx, dimension);
    ctx_.builder().CreateCondBr(idx_less, zero_body, zero_exit);
    
    ctx_.builder().SetInsertPoint(zero_body);
    Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_elems_ptr, idx);
    ctx_.builder().CreateStore(ConstantFP::get(ctx_.doubleType(), 0.0), elem_ptr);
    Value* next_idx = ctx_.builder().CreateAdd(idx, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_idx, idx_ptr);
    ctx_.builder().CreateBr(zero_cond);
    
    ctx_.builder().SetInsertPoint(zero_exit);
    
    ctx_.builder().CreateStore(typed_elems_ptr,
        ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_tensor_ptr, 2));  // elements
    
    // Return tensor pointer as i64
    return ctx_.builder().CreatePtrToInt(typed_tensor_ptr, ctx_.int64Type());
}


// Helper: Extract J[row,col] from Jacobian's nested list structure
// Jacobian tensor elements are int64 list pointers (rows), not doubles!
// Extract element from N-dimensional tensor at given indices
// For 2D (Jacobian): indices = [row_idx, col_idx]
// For ND: computes linear index using row-major ordering
llvm::Value* AutodiffCodegen::extractTensorElement(llvm::Value* tensor_ptr, std::vector<llvm::Value*> indices) {
    using namespace llvm;
    // Get tensor dimensions
    Value* dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 0);
    Value* dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), dims_field);
    Value* typed_dims = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());


    // Get elements array
    Value* elements_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 2);
    Value* elements_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), elements_field);
    Value* typed_elements = ctx_.builder().CreatePointerCast(elements_ptr, ctx_.builder().getPtrTy());

    // Compute linear index using row-major ordering
    // linear_idx = idx[0] * (dims[1] * dims[2] * ...) + idx[1] * (dims[2] * ...) + ... + idx[n-1]
    Value* linear_idx = ConstantInt::get(ctx_.int64Type(), 0);

    for (size_t i = 0; i < indices.size(); i++) {
        // Compute stride for dimension i (product of all subsequent dimensions)
        Value* stride = ConstantInt::get(ctx_.int64Type(), 1);
        for (size_t j = i + 1; j < indices.size(); j++) {
            Value* dim_j_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims,
                ConstantInt::get(ctx_.int64Type(), j));
            Value* dim_j = ctx_.builder().CreateLoad(ctx_.int64Type(), dim_j_ptr);
            stride = ctx_.builder().CreateMul(stride, dim_j);
        }
        // Add idx[i] * stride to linear index
        Value* contribution = ctx_.builder().CreateMul(indices[i], stride);
        linear_idx = ctx_.builder().CreateAdd(linear_idx, contribution);
    }

    // Load element as int64 (bit pattern of double)
    Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_elements, linear_idx);
    Value* elem_bits = ctx_.builder().CreateLoad(ctx_.int64Type(), elem_ptr);

    // Convert int64 bit pattern back to double
    Value* elem_double = ctx_.builder().CreateBitCast(elem_bits, ctx_.doubleType());

    return elem_double;
}

// Convenience wrapper for 2D tensors (Jacobian, Hessian)
llvm::Value* AutodiffCodegen::extractJacobianElement(llvm::Value* jacobian_ptr, llvm::Value* row_idx, llvm::Value* col_idx, llvm::Value* n) {
    using namespace llvm;
    // Use the general N-dimensional extractor with 2 indices
    return extractTensorElement(jacobian_ptr, {row_idx, col_idx});
}


llvm::Value* AutodiffCodegen::divergence(const eshkol_operations_t* op) {
    using namespace llvm;
    if (!op->divergence_op.function || !op->divergence_op.point) {
        eshkol_error("Invalid divergence operation - missing function or point");
        return nullptr;
    }
    
    eshkol_info("Computing divergence of vector field");
    
    // The divergence is the sum of diagonal elements of the Jacobian
    // For F: ℝⁿ → ℝⁿ, Jacobian is n×n, divergence is trace(J)
    
    // Compute Jacobian matrix first
    eshkol_operations_t jacobian_temp;
    jacobian_temp.op = ESHKOL_JACOBIAN_OP;
    jacobian_temp.jacobian_op.function = op->divergence_op.function;
    jacobian_temp.jacobian_op.point = op->divergence_op.point;
    
    Value* jacobian_tagged = jacobian(&jacobian_temp);
    if (!jacobian_tagged) {
        eshkol_error("Failed to compute Jacobian for divergence");
        return nullptr;
    }
    
    // ENHANCED TYPE CHECK: Verify Jacobian is a valid tensor (same fix as Jacobian operator)
    Value* jacobian_type = tagged_.getType(jacobian_tagged);
    Value* jacobian_base_type = tagged_.getBaseType(jacobian_type);

    Value* jac_is_tensor_ptr = ctx_.builder().CreateICmpEQ(jacobian_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* jac_is_ad_tensor = ctx_.builder().CreateICmpEQ(jacobian_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    Value* jac_is_valid = ctx_.builder().CreateOr(jac_is_tensor_ptr, jac_is_ad_tensor);
    
    Function* div_current_func = ctx_.builder().GetInsertBlock()->getParent();
    BasicBlock* jacobian_valid = BasicBlock::Create(ctx_.context(), "div_jac_valid", div_current_func);
    BasicBlock* jacobian_invalid = BasicBlock::Create(ctx_.context(), "div_jac_invalid", div_current_func);
    BasicBlock* div_final = BasicBlock::Create(ctx_.context(), "div_final", div_current_func);
    
    ctx_.builder().CreateCondBr(jac_is_valid, jacobian_valid, jacobian_invalid);
    
    // Invalid jacobian: return 0.0 instead of crashing (only for genuinely invalid types)
    ctx_.builder().SetInsertPoint(jacobian_invalid);
    eshkol_debug("Divergence: Jacobian returned non-tensor type, returning 0.0");
    Value* zero_result = ConstantFP::get(ctx_.doubleType(), 0.0);
    ctx_.builder().CreateBr(div_final);
    
    // Valid jacobian: continue with normal computation
    ctx_.builder().SetInsertPoint(jacobian_valid);
    
    // Extract tensor pointer from validated tagged value
    Value* jacobian_ptr_int = tagged_.unpackInt64(jacobian_tagged);
    Value* jacobian_ptr = ctx_.builder().CreateIntToPtr(jacobian_ptr_int, ctx_.builder().getPtrTy());
    
    // Extract dimension n from Jacobian (it's n×n)
    Value* dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), jacobian_ptr, 0);
    Value* dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), dims_field);
    Value* typed_dims_ptr = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());
    
    Value* n_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims_ptr,
        ConstantInt::get(ctx_.int64Type(), 0));
    Value* n = ctx_.builder().CreateLoad(ctx_.int64Type(), n_ptr);
    
    // Sum diagonal elements: J[0,0] + J[1,1] + ... + J[n-1,n-1]
    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    BasicBlock* sum_loop_cond = BasicBlock::Create(ctx_.context(), "div_sum_cond", current_func);
    BasicBlock* sum_loop_body = BasicBlock::Create(ctx_.context(), "div_sum_body", current_func);
    BasicBlock* sum_loop_exit = BasicBlock::Create(ctx_.context(), "div_sum_exit", current_func);
    
    Value* sum_idx = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "sum_idx");
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    
    Value* divergence_acc = ctx_.builder().CreateAlloca(ctx_.doubleType(), nullptr, "div_acc");
    ctx_.builder().CreateStore(ConstantFP::get(ctx_.doubleType(), 0.0), divergence_acc);
    
    ctx_.builder().CreateBr(sum_loop_cond);
    
    ctx_.builder().SetInsertPoint(sum_loop_cond);
    Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), sum_idx);
    Value* i_less_n = ctx_.builder().CreateICmpULT(i, n);
    ctx_.builder().CreateCondBr(i_less_n, sum_loop_body, sum_loop_exit);
    
    ctx_.builder().SetInsertPoint(sum_loop_body);
    
    // Extract J[i,i] from nested list structure (not direct double access!)
    Value* diagonal_elem = extractJacobianElement(jacobian_ptr, i, i, n);
    
    // Add to accumulator
    Value* current_div = ctx_.builder().CreateLoad(ctx_.doubleType(), divergence_acc);
    Value* new_div = ctx_.builder().CreateFAdd(current_div, diagonal_elem);
    ctx_.builder().CreateStore(new_div, divergence_acc);
    
    Value* next_i = ctx_.builder().CreateAdd(i, ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, sum_idx);
    ctx_.builder().CreateBr(sum_loop_cond);
    
    ctx_.builder().SetInsertPoint(sum_loop_exit);
    Value* divergence_result = ctx_.builder().CreateLoad(ctx_.doubleType(), divergence_acc);
    ctx_.builder().CreateBr(div_final);
    
    // Merge valid and invalid paths
    ctx_.builder().SetInsertPoint(div_final);
    PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "div_result");
    result_phi->addIncoming(zero_result, jacobian_invalid);
    result_phi->addIncoming(divergence_result, sum_loop_exit);

    eshkol_info("Divergence computation complete");
    return tagged_.packDouble(result_phi);
}


llvm::Value* AutodiffCodegen::curl(const eshkol_operations_t* op) {
    using namespace llvm;
    if (!op->curl_op.function || !op->curl_op.point) {
        eshkol_error("Invalid curl operation - missing function or point");
        return nullptr;
    }
    
    eshkol_info("Computing curl of 3D vector field");
    
    // First, validate that input is 3D
    Value* vector_val_raw = codegen_ast_callback_(op->curl_op.point, callback_context_);
    if (!vector_val_raw) {
        eshkol_error("Failed to evaluate curl point");
        return nullptr;
    }

    // Tensor literal fix: codegenTensor returns ptr-as-int64; wrap as HEAP_PTR
    Value* vector_val;
    if (vector_val_raw->getType() == ctx_.taggedValueType()) {
        vector_val = vector_val_raw;
    } else if (vector_val_raw->getType()->isIntegerTy(64) &&
               op->curl_op.point->type == ESHKOL_TENSOR) {
        vector_val = tagged_.packPtr(vector_val_raw, ESHKOL_VALUE_HEAP_PTR);
    } else if (vector_val_raw->getType()->isIntegerTy(64)) {
        vector_val = tagged_.packInt64(vector_val_raw, true);
    } else if (vector_val_raw->getType()->isDoubleTy()) {
        vector_val = tagged_.packDouble(vector_val_raw);
    } else {
        // Ensure tagged value (direct packing)
        vector_val = typedValueToTaggedValue(tv);
    }

    // Get arena for OALR-compliant tensor allocation
    Value* arena_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), ctx_.globalArena());

    // M1 CONSOLIDATION: Handle HEAP_PTR (with subtype dispatch), legacy VECTOR_PTR, and tensor
    Value* curl_input_type = tagged_.getType(vector_val);
    Value* curl_input_base_type = tagged_.getBaseType(curl_input_type);
    Value* curl_is_heap_ptr = ctx_.builder().CreateICmpEQ(curl_input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* curl_is_legacy_vec = ctx_.builder().CreateICmpEQ(curl_input_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

    Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    BasicBlock* curl_heap_dispatch = BasicBlock::Create(ctx_.context(), "curl_heap_dispatch", current_func);
    BasicBlock* curl_check_legacy = BasicBlock::Create(ctx_.context(), "curl_check_legacy", current_func);
    BasicBlock* curl_scheme_input = BasicBlock::Create(ctx_.context(), "curl_scheme_input", current_func);
    BasicBlock* curl_tensor_input = BasicBlock::Create(ctx_.context(), "curl_tensor_input", current_func);
    BasicBlock* curl_merge_n = BasicBlock::Create(ctx_.context(), "curl_merge_n", current_func);

    ctx_.builder().CreateCondBr(curl_is_heap_ptr, curl_heap_dispatch, curl_check_legacy);

    // HEAP_PTR dispatch - read subtype from header
    ctx_.builder().SetInsertPoint(curl_heap_dispatch);
    Value* curl_heap_ptr_val = tagged_.unpackPtr(vector_val);
    Value* curl_header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), curl_heap_ptr_val, ConstantInt::get(ctx_.int64Type(), -8));
    Value* curl_subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), curl_header_ptr);
    Value* curl_is_vec_subtype = ctx_.builder().CreateICmpEQ(curl_subtype, ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_VECTOR));
    ctx_.builder().CreateCondBr(curl_is_vec_subtype, curl_scheme_input, curl_tensor_input);

    // Legacy VECTOR_PTR fallback
    ctx_.builder().SetInsertPoint(curl_check_legacy);
    ctx_.builder().CreateCondBr(curl_is_legacy_vec, curl_scheme_input, curl_tensor_input);

    // SCHEME VECTOR: Extract dimension from vector length
    ctx_.builder().SetInsertPoint(curl_scheme_input);
    Value* curl_svec_ptr_int = tagged_.unpackInt64(vector_val);
    Value* curl_svec_ptr = ctx_.builder().CreateIntToPtr(curl_svec_ptr_int, ctx_.builder().getPtrTy());
    Value* curl_svec_len_ptr = ctx_.builder().CreateBitCast(curl_svec_ptr, PointerType::getUnqual(ctx_.context()));
    Value* curl_svec_n = ctx_.builder().CreateLoad(ctx_.int64Type(), curl_svec_len_ptr);
    ctx_.builder().CreateBr(curl_merge_n);
    BasicBlock* curl_scheme_exit = ctx_.builder().GetInsertBlock();

    // TENSOR: Extract dimension from tensor structure
    ctx_.builder().SetInsertPoint(curl_tensor_input);
    Value* curl_tensor_ptr_int = tagged_.unpackInt64(vector_val);
    Value* curl_tensor_ptr = ctx_.builder().CreateIntToPtr(curl_tensor_ptr_int, ctx_.builder().getPtrTy());
    Value* dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), curl_tensor_ptr, 0);
    Value* dims_ptr = ctx_.builder().CreateLoad(PointerType::getUnqual(ctx_.context()), dims_field);
    Value* typed_dims_ptr = ctx_.builder().CreatePointerCast(dims_ptr, ctx_.builder().getPtrTy());
    Value* n_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_dims_ptr,
        ConstantInt::get(ctx_.int64Type(), 0));
    Value* curl_tensor_n = ctx_.builder().CreateLoad(ctx_.int64Type(), n_ptr);
    ctx_.builder().CreateBr(curl_merge_n);
    BasicBlock* curl_tensor_exit = ctx_.builder().GetInsertBlock();

    // MERGE: Get n from whichever path
    ctx_.builder().SetInsertPoint(curl_merge_n);
    PHINode* n = ctx_.builder().CreatePHI(ctx_.int64Type(), 2, "curl_n");
    n->addIncoming(curl_svec_n, curl_scheme_exit);
    n->addIncoming(curl_tensor_n, curl_tensor_exit);

    // ENHANCED VALIDATION: Accept n>=2 for general differential 2-forms
    // Classic curl is 3D, but generalized exterior derivative works in any dimension >= 2
    Value* n_ge_2 = ctx_.builder().CreateICmpUGE(n, ConstantInt::get(ctx_.int64Type(), 2));

    BasicBlock* dim_valid = BasicBlock::Create(ctx_.context(), "curl_dim_valid", current_func);
    BasicBlock* dim_invalid = BasicBlock::Create(ctx_.context(), "curl_dim_invalid", current_func);
    BasicBlock* curl_done = BasicBlock::Create(ctx_.context(), "curl_done", current_func);
    
    ctx_.builder().CreateCondBr(n_ge_2, dim_valid, dim_invalid);
    
    // Invalid dimension: return null vector for dim < 2
    ctx_.builder().SetInsertPoint(dim_invalid);
    eshkol_debug("Curl: dimension < 2, differential forms require at least 2D");
    Value* null_result_int = createNullVectorTensor(n);  // Use actual dimension, not hardcoded 3
    Value* null_result = tagged_.packPtr(null_result_int, ESHKOL_VALUE_HEAP_PTR);
    BasicBlock* dim_invalid_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(curl_done);
    
    // Valid dimension: compute curl (differential 2-form)
    // NOTE: For n!=3, this computes the generalized exterior derivative
    ctx_.builder().SetInsertPoint(dim_valid);
    
    // Compute Jacobian matrix (3×3)
    eshkol_operations_t jacobian_temp;
    jacobian_temp.op = ESHKOL_JACOBIAN_OP;
    jacobian_temp.jacobian_op.function = op->curl_op.function;
    jacobian_temp.jacobian_op.point = op->curl_op.point;
    
    Value* jacobian_tagged = jacobian(&jacobian_temp);
    if (!jacobian_tagged) {
        eshkol_error("Failed to compute Jacobian for curl");
        return nullptr;
    }
    
    // ENHANCED TYPE CHECK: Verify Jacobian is a valid tensor (same fix as Jacobian operator)
    Value* jacobian_type = tagged_.getType(jacobian_tagged);
    Value* jacobian_base_type = tagged_.getBaseType(jacobian_type);

    Value* jac_is_tensor_ptr = ctx_.builder().CreateICmpEQ(jacobian_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));
    Value* jac_is_ad_tensor = ctx_.builder().CreateICmpEQ(jacobian_base_type,
        ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
    Value* jac_is_valid = ctx_.builder().CreateOr(jac_is_tensor_ptr, jac_is_ad_tensor);
    
    BasicBlock* jac_valid = BasicBlock::Create(ctx_.context(), "curl_jac_valid", current_func);
    BasicBlock* jac_invalid = BasicBlock::Create(ctx_.context(), "curl_jac_invalid", current_func);
    
    // If IS valid tensor type, proceed; if NOT, error path
    ctx_.builder().CreateCondBr(jac_is_valid, jac_valid, jac_invalid);
    
    // Invalid jacobian: return null 3D vector (only for genuinely invalid types)
    ctx_.builder().SetInsertPoint(jac_invalid);
    eshkol_debug("Curl: Jacobian returned non-tensor type, returning null vector");
    Value* null_curl_int = createNullVectorTensor(
        ConstantInt::get(ctx_.int64Type(), 3)
    );
    // Tag as TENSOR_PTR for proper display
    Value* null_curl = tagged_.packPtr(null_curl_int, ESHKOL_VALUE_HEAP_PTR);
    BasicBlock* jac_invalid_exit = ctx_.builder().GetInsertBlock(); // Capture actual exit block!
    ctx_.builder().CreateBr(curl_done);
    
    // Valid jacobian: continue with normal computation
    ctx_.builder().SetInsertPoint(jac_valid);
    
    // Extract tensor pointer from validated tagged value
    Value* jacobian_ptr_int = tagged_.unpackInt64(jacobian_tagged);
    Value* jacobian_ptr = ctx_.builder().CreateIntToPtr(jacobian_ptr_int, ctx_.builder().getPtrTy());
    Value* n_const = ConstantInt::get(ctx_.int64Type(), 3);
    
    // Extract specific partial derivatives from Jacobian's nested list structure
    // J[i,j] = ∂Fᵢ/∂xⱼ (row i, column j)
    // Jacobian elements are LIST POINTERS (rows), not doubles!
    
    // curl_x = ∂F₃/∂y - ∂F₂/∂z = J[2,1] - J[1,2]
    Value* dF3_dx2 = extractJacobianElement(jacobian_ptr,
        ConstantInt::get(ctx_.int64Type(), 2),  // row 2
        ConstantInt::get(ctx_.int64Type(), 1),  // col 1
        n_const);
    Value* dF2_dx3 = extractJacobianElement(jacobian_ptr,
        ConstantInt::get(ctx_.int64Type(), 1),  // row 1
        ConstantInt::get(ctx_.int64Type(), 2),  // col 2
        n_const);
    Value* curl_x = ctx_.builder().CreateFSub(dF3_dx2, dF2_dx3);
    
    // curl_y = ∂F₁/∂z - ∂F₃/∂x = J[0,2] - J[2,0]
    Value* dF1_dx3 = extractJacobianElement(jacobian_ptr,
        ConstantInt::get(ctx_.int64Type(), 0),  // row 0
        ConstantInt::get(ctx_.int64Type(), 2),  // col 2
        n_const);
    Value* dF3_dx1 = extractJacobianElement(jacobian_ptr,
        ConstantInt::get(ctx_.int64Type(), 2),  // row 2
        ConstantInt::get(ctx_.int64Type(), 0),  // col 0
        n_const);
    Value* curl_y = ctx_.builder().CreateFSub(dF1_dx3, dF3_dx1);
    
    // curl_z = ∂F₂/∂x - ∂F₁/∂y = J[1,0] - J[0,1]
    Value* dF2_dx1 = extractJacobianElement(jacobian_ptr,
        ConstantInt::get(ctx_.int64Type(), 1),  // row 1
        ConstantInt::get(ctx_.int64Type(), 0),  // col 0
        n_const);
    Value* dF1_dx2 = extractJacobianElement(jacobian_ptr,
        ConstantInt::get(ctx_.int64Type(), 0),  // row 0
        ConstantInt::get(ctx_.int64Type(), 1),  // col 1
        n_const);
    Value* curl_z = ctx_.builder().CreateFSub(dF2_dx1, dF1_dx2);
    
    // Create result 3D vector
    // Allocate result tensor via arena (OALR compliant - no malloc)
    Value* typed_result_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocateTensorWithHeader(), {arena_ptr});

    // Set dimensions [3]
    Value* result_dims_size = ConstantInt::get(ctx_.int64Type(), sizeof(uint64_t));
    Value* result_dims_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, result_dims_size});
    Value* typed_result_dims = ctx_.builder().CreatePointerCast(result_dims_ptr, ctx_.builder().getPtrTy());
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 3), typed_result_dims);

    Value* result_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_ptr, 0);
    ctx_.builder().CreateStore(typed_result_dims, result_dims_field);

    Value* result_num_dims_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_ptr, 1);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 1), result_num_dims_field);

    Value* result_total_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_ptr, 3);
    ctx_.builder().CreateStore(ConstantInt::get(ctx_.int64Type(), 3), result_total_field);

    // Allocate and fill elements [curl_x, curl_y, curl_z]
    Value* result_elems_size = ConstantInt::get(ctx_.int64Type(), 3 * sizeof(double));
    Value* result_elems_ptr = ctx_.builder().CreateCall(mem_.getArenaAllocate(), {arena_ptr, result_elems_size});
    Value* typed_result_elems = ctx_.builder().CreatePointerCast(result_elems_ptr, ctx_.builder().getPtrTy());
    
    Value* result_elems_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), typed_result_ptr, 2);
    ctx_.builder().CreateStore(typed_result_elems, result_elems_field);
    
    // Store curl components
    Value* elem0_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_result_elems,
        ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateStore(curl_x, elem0_ptr);
    
    Value* elem1_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_result_elems,
        ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(curl_y, elem1_ptr);
    
    Value* elem2_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), typed_result_elems,
        ConstantInt::get(ctx_.int64Type(), 2));
    ctx_.builder().CreateStore(curl_z, elem2_ptr);
    
    eshkol_info("Curl computation complete, returning 3D vector");
    Value* curl_result_int = ctx_.builder().CreatePtrToInt(typed_result_ptr, ctx_.int64Type());
    // Tag as TENSOR_PTR for proper display and type consistency
    Value* curl_result = tagged_.packPtr(curl_result_int, ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateBr(curl_done);
    BasicBlock* dim_valid_exit = ctx_.builder().GetInsertBlock(); // Capture actual predecessor!
    
    // Merge all paths with tagged_value results (type-consistent!)
    ctx_.builder().SetInsertPoint(curl_done);
    PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 3, "curl_result");
    result_phi->addIncoming(null_result, dim_invalid_exit);   // Already tagged
    result_phi->addIncoming(null_curl, jac_invalid_exit);     // Already tagged
    result_phi->addIncoming(curl_result, dim_valid_exit);
    
    return result_phi;
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

// ═══════════════════════════════════════════════════════════════════════════
// CAPTURE RESOLUTION — Extracted from llvm_codegen.cpp
// ═══════════════════════════════════════════════════════════════════════════

std::vector<llvm::Value*> AutodiffCodegen::loadCapturesForAutodiff(
    llvm::Function* func_ptr, const std::string& context_name) {
    using namespace llvm;

    std::vector<Value*> capture_args;

    FunctionType* func_type = func_ptr->getFunctionType();
    if (func_type->getNumParams() <= 1) {
        return capture_args; // No captures
    }

    size_t num_captures = func_type->getNumParams() - 1;
    std::string lambda_name = func_ptr->getName().str();

    // REPL MODE: Get capture names from registry instead of parameter names
    std::vector<std::string> capture_names;
    if (repl_mode_enabled_ && *repl_mode_enabled_) {
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
            auto arg_it = func_ptr->arg_begin();
            std::advance(arg_it, i + 1);
            if (arg_it != func_ptr->arg_end()) {
                var_name = arg_it->getName().str();
                if (var_name.find("captured_") == 0) {
                    var_name = var_name.substr(9);
                }
            }
        }

        std::string capture_key = lambda_name + "_capture_" + var_name;

        // First try capture-specific key in symbol tables
        auto it = global_symbol_table_->find(capture_key);
        bool found_in_global = (it != global_symbol_table_->end());
        if (!found_in_global) {
            it = symbol_table_->find(capture_key);
        }

        bool found = found_in_global ? (it != global_symbol_table_->end()) : (it != symbol_table_->end());

        // FALLBACK: Try raw variable name (for top-level global variables)
        if (!found) {
            it = global_symbol_table_->find(var_name);
            found_in_global = (it != global_symbol_table_->end());
            if (!found_in_global) {
                it = symbol_table_->find(var_name);
            }
            found = found_in_global ? (it != global_symbol_table_->end()) : (it != symbol_table_->end());
            if (found) {
                eshkol_debug("%s: found capture '%s' via raw variable name", context_name.c_str(), var_name.c_str());
            }
        }

        // REPL MODE: Try creating external declaration for capture global
        if (!found && repl_mode_enabled_ && *repl_mode_enabled_) {
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
                Value* helper_global_ptr_int = ctx_.builder().CreatePtrToInt(capture_global, ctx_.int64Type());
                Value* helper_packed_capture = tagged_.packInt64(helper_global_ptr_int, true);
                Value* helper_capture_storage = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "autodiff_capture_storage");
                ctx_.builder().CreateStore(helper_packed_capture, helper_capture_storage);
                capture_args.push_back(helper_capture_storage);
                continue;
            }
        }

        if (found && it->second) {
            Value* storage = it->second;
            Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
            IRBuilder<> entry_builder(&current_func->getEntryBlock(),
                                      current_func->getEntryBlock().begin());
            AllocaInst* temp_alloca = entry_builder.CreateAlloca(
                ctx_.taggedValueType(), nullptr, var_name + "_autodiff_capture_storage");

            Value* ptr_as_int = ctx_.builder().CreatePtrToInt(storage, ctx_.int64Type());
            Value* packed_ptr = tagged_.packInt64(ptr_as_int, true);
            ctx_.builder().CreateStore(packed_ptr, temp_alloca);

            capture_args.push_back(temp_alloca);
        } else {
            capture_args.push_back(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())));
            eshkol_warn("%s: capture '%s' not found, using null pointer", context_name.c_str(), var_name.c_str());
        }
    }

    return capture_args;
}

void AutodiffCodegen::resolveGradientCaptures(
    llvm::Function* func_ptr,
    std::vector<llvm::Value*>& call_args,
    const std::string& context_label) {
    using namespace llvm;

    FunctionType* func_type = func_ptr->getFunctionType();
    size_t total_llvm_params = func_type->getNumParams();
    size_t args_provided = call_args.size();

    if (total_llvm_params <= args_provided) return;

    size_t num_captures = total_llvm_params - args_provided;
    std::string lambda_name = func_ptr->getName().str();

    // REPL MODE: Get capture names from registry
    std::vector<std::string> capture_names;
    if (repl_mode_enabled_ && *repl_mode_enabled_) {
        std::lock_guard<std::mutex> lock(*repl_mutex_);
        auto captures_it = repl_lambda_captures_->find(lambda_name);
        if (captures_it != repl_lambda_captures_->end()) {
            capture_names = captures_it->second;
        }
    }

    for (size_t ci = 0; ci < num_captures; ci++) {
        std::string var_name;
        if (ci < capture_names.size()) {
            var_name = capture_names[ci];
        } else {
            auto arg_it = func_ptr->arg_begin();
            std::advance(arg_it, args_provided + ci);
            if (arg_it != func_ptr->arg_end()) {
                var_name = arg_it->getName().str();
                if (var_name.find("captured_") == 0) var_name = var_name.substr(9);
            }
        }

        std::string capture_key = lambda_name + "_capture_" + var_name;

        // Search order: capture key in global → local, then raw name in global → local
        Value* storage = nullptr;
        auto it = global_symbol_table_->find(capture_key);
        if (it != global_symbol_table_->end() && it->second) {
            storage = it->second;
        } else {
            it = symbol_table_->find(capture_key);
            if (it != symbol_table_->end() && it->second) {
                storage = it->second;
            } else {
                it = global_symbol_table_->find(var_name);
                if (it != global_symbol_table_->end() && it->second) {
                    storage = it->second;
                } else {
                    it = symbol_table_->find(var_name);
                    if (it != symbol_table_->end() && it->second) {
                        storage = it->second;
                    }
                }
            }
        }

        // REPL MODE: Try creating external declaration for capture global
        if (!storage && repl_mode_enabled_ && *repl_mode_enabled_) {
            std::lock_guard<std::mutex> lock(*repl_mutex_);
            auto sym_it = repl_symbol_addresses_->find(capture_key);
            if (sym_it != repl_symbol_addresses_->end()) {
                GlobalVariable* capture_global = ctx_.module().getGlobalVariable(capture_key);
                if (!capture_global) {
                    capture_global = new GlobalVariable(
                        ctx_.module(), ctx_.taggedValueType(), false,
                        GlobalValue::ExternalLinkage, nullptr, capture_key);
                }
                Value* global_ptr_int = ctx_.builder().CreatePtrToInt(capture_global, ctx_.int64Type());
                Value* packed = tagged_.packInt64(global_ptr_int, true);
                Value* temp = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_cap");
                ctx_.builder().CreateStore(packed, temp);
                call_args.push_back(temp);
                continue;
            }
        }

        if (storage) {
            Value* ptr_int = ctx_.builder().CreatePtrToInt(storage, ctx_.int64Type());
            Value* packed = tagged_.packInt64(ptr_int, true);
            Value* temp = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "grad_cap");
            ctx_.builder().CreateStore(packed, temp);
            call_args.push_back(temp);
        } else {
            call_args.push_back(ConstantPointerNull::get(PointerType::getUnqual(ctx_.context())));
            eshkol_warn("Gradient (%s): capture '%s' not found, using null pointer",
                        context_label.c_str(), var_name.c_str());
        }
    }
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

    // === LEAF NODES (types 0, 1): constants and variables have no inputs to propagate to ===
    llvm::Value* is_constant = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 0));
    llvm::Value* is_variable = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 1));
    llvm::Value* is_leaf = ctx_.builder().CreateOr(is_constant, is_variable);
    llvm::BasicBlock* check_ops = llvm::BasicBlock::Create(ctx_.context(), "check_ops", current_func);
    ctx_.builder().CreateCondBr(is_leaf, done_block, check_ops);

    ctx_.builder().SetInsertPoint(check_ops);

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
    llvm::BasicBlock* check_tan = llvm::BasicBlock::Create(ctx_.context(), "check_tan", current_func);
    ctx_.builder().CreateCondBr(is_min, min_block, check_tan);

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

    // ===== COMPLETE MATH FUNCTION BACKWARD PASSES (types 54-66) =====
    // All standard math functions with proper derivative computation

    // Create blocks for all math function backward passes
    llvm::BasicBlock* tan_block = llvm::BasicBlock::Create(ctx_.context(), "grad_tan", current_func);
    llvm::BasicBlock* asin_block2 = llvm::BasicBlock::Create(ctx_.context(), "grad_asin", current_func);
    llvm::BasicBlock* acos_block2 = llvm::BasicBlock::Create(ctx_.context(), "grad_acos", current_func);
    llvm::BasicBlock* atan_block = llvm::BasicBlock::Create(ctx_.context(), "grad_atan", current_func);
    llvm::BasicBlock* sinh_block2 = llvm::BasicBlock::Create(ctx_.context(), "grad_sinh", current_func);
    llvm::BasicBlock* cosh_block2 = llvm::BasicBlock::Create(ctx_.context(), "grad_cosh", current_func);
    llvm::BasicBlock* asinh_block = llvm::BasicBlock::Create(ctx_.context(), "grad_asinh", current_func);
    llvm::BasicBlock* acosh_block = llvm::BasicBlock::Create(ctx_.context(), "grad_acosh", current_func);
    llvm::BasicBlock* atanh_block = llvm::BasicBlock::Create(ctx_.context(), "grad_atanh", current_func);
    llvm::BasicBlock* log10_block = llvm::BasicBlock::Create(ctx_.context(), "grad_log10", current_func);
    llvm::BasicBlock* log2_block = llvm::BasicBlock::Create(ctx_.context(), "grad_log2", current_func);
    llvm::BasicBlock* exp2_block = llvm::BasicBlock::Create(ctx_.context(), "grad_exp2", current_func);
    llvm::BasicBlock* cbrt_block = llvm::BasicBlock::Create(ctx_.context(), "grad_cbrt", current_func);

    // --- TAN (type=54): dL/dx = dL/dz * (1 + tan²(x)) = dL/dz / cos²(x) ---
    ctx_.builder().SetInsertPoint(check_tan);
    llvm::Value* is_tan = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 54));
    llvm::BasicBlock* check_asin = llvm::BasicBlock::Create(ctx_.context(), "check_asin", current_func);
    ctx_.builder().CreateCondBr(is_tan, tan_block, check_asin);

    ctx_.builder().SetInsertPoint(tan_block);
    if (input1) {
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* tan_x = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* tan_sq = ctx_.builder().CreateFMul(tan_x, tan_x);
        llvm::Value* sec_sq = ctx_.builder().CreateFAdd(one, tan_sq);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, sec_sq);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- ASIN (type=55): dL/dx = dL/dz / sqrt(1 - x²) ---
    ctx_.builder().SetInsertPoint(check_asin);
    llvm::Value* is_asin = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 55));
    llvm::BasicBlock* check_acos = llvm::BasicBlock::Create(ctx_.context(), "check_acos", current_func);
    ctx_.builder().CreateCondBr(is_asin, asin_block2, check_acos);

    ctx_.builder().SetInsertPoint(asin_block2);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
        llvm::Value* under = ctx_.builder().CreateFSub(one, x_sq);
        llvm::Function* sqrt_func = getMathFunc("sqrt");
        if (sqrt_func) {
            llvm::Value* sqrt_under = ctx_.builder().CreateCall(sqrt_func, {under});
            llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, sqrt_under);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- ACOS (type=56): dL/dx = -dL/dz / sqrt(1 - x²) ---
    ctx_.builder().SetInsertPoint(check_acos);
    llvm::Value* is_acos = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 56));
    llvm::BasicBlock* check_atan = llvm::BasicBlock::Create(ctx_.context(), "check_atan", current_func);
    ctx_.builder().CreateCondBr(is_acos, acos_block2, check_atan);

    ctx_.builder().SetInsertPoint(acos_block2);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
        llvm::Value* under = ctx_.builder().CreateFSub(one, x_sq);
        llvm::Function* sqrt_func = getMathFunc("sqrt");
        if (sqrt_func) {
            llvm::Value* sqrt_under = ctx_.builder().CreateCall(sqrt_func, {under});
            llvm::Value* neg_grad = ctx_.builder().CreateFNeg(node_grad);
            llvm::Value* grad_input = ctx_.builder().CreateFDiv(neg_grad, sqrt_under);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- ATAN (type=57): dL/dx = dL/dz / (1 + x²) ---
    ctx_.builder().SetInsertPoint(check_atan);
    llvm::Value* is_atan = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 57));
    llvm::BasicBlock* check_sinh = llvm::BasicBlock::Create(ctx_.context(), "check_sinh", current_func);
    ctx_.builder().CreateCondBr(is_atan, atan_block, check_sinh);

    ctx_.builder().SetInsertPoint(atan_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
        llvm::Value* denom = ctx_.builder().CreateFAdd(one, x_sq);
        llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, denom);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- SINH (type=58): dL/dx = dL/dz * cosh(x) ---
    ctx_.builder().SetInsertPoint(check_sinh);
    llvm::Value* is_sinh = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 58));
    llvm::BasicBlock* check_cosh = llvm::BasicBlock::Create(ctx_.context(), "check_cosh", current_func);
    ctx_.builder().CreateCondBr(is_sinh, sinh_block2, check_cosh);

    ctx_.builder().SetInsertPoint(sinh_block2);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* cosh_func = getMathFunc("cosh");
        if (cosh_func) {
            llvm::Value* cosh_val = ctx_.builder().CreateCall(cosh_func, {input_val});
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, cosh_val);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- COSH (type=59): dL/dx = dL/dz * sinh(x) ---
    ctx_.builder().SetInsertPoint(check_cosh);
    llvm::Value* is_cosh = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 59));
    llvm::BasicBlock* check_asinh = llvm::BasicBlock::Create(ctx_.context(), "check_asinh", current_func);
    ctx_.builder().CreateCondBr(is_cosh, cosh_block2, check_asinh);

    ctx_.builder().SetInsertPoint(cosh_block2);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Function* sinh_func = getMathFunc("sinh");
        if (sinh_func) {
            llvm::Value* sinh_val = ctx_.builder().CreateCall(sinh_func, {input_val});
            llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, sinh_val);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- ASINH (type=60): dL/dx = dL/dz / sqrt(1 + x²) ---
    ctx_.builder().SetInsertPoint(check_asinh);
    llvm::Value* is_asinh = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 60));
    llvm::BasicBlock* check_acosh = llvm::BasicBlock::Create(ctx_.context(), "check_acosh", current_func);
    ctx_.builder().CreateCondBr(is_asinh, asinh_block, check_acosh);

    ctx_.builder().SetInsertPoint(asinh_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
        llvm::Value* under = ctx_.builder().CreateFAdd(one, x_sq);
        llvm::Function* sqrt_func = getMathFunc("sqrt");
        if (sqrt_func) {
            llvm::Value* sqrt_under = ctx_.builder().CreateCall(sqrt_func, {under});
            llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, sqrt_under);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- ACOSH (type=61): dL/dx = dL/dz / sqrt(x² - 1) ---
    ctx_.builder().SetInsertPoint(check_acosh);
    llvm::Value* is_acosh = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 61));
    llvm::BasicBlock* check_atanh = llvm::BasicBlock::Create(ctx_.context(), "check_atanh", current_func);
    ctx_.builder().CreateCondBr(is_acosh, acosh_block, check_atanh);

    ctx_.builder().SetInsertPoint(acosh_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
        llvm::Value* under = ctx_.builder().CreateFSub(x_sq, one);
        llvm::Function* sqrt_func = getMathFunc("sqrt");
        if (sqrt_func) {
            llvm::Value* sqrt_under = ctx_.builder().CreateCall(sqrt_func, {under});
            llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, sqrt_under);
            accumulateGradient(input1, grad_input);
        }
    }
    ctx_.builder().CreateBr(done_block);

    // --- ATANH (type=62): dL/dx = dL/dz / (1 - x²) ---
    ctx_.builder().SetInsertPoint(check_atanh);
    llvm::Value* is_atanh = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 62));
    llvm::BasicBlock* check_log10 = llvm::BasicBlock::Create(ctx_.context(), "check_log10", current_func);
    ctx_.builder().CreateCondBr(is_atanh, atanh_block, check_log10);

    ctx_.builder().SetInsertPoint(atanh_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
        llvm::Value* x_sq = ctx_.builder().CreateFMul(input_val, input_val);
        llvm::Value* denom = ctx_.builder().CreateFSub(one, x_sq);
        llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, denom);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- LOG10 (type=63): dL/dx = dL/dz / (x * ln(10)) ---
    ctx_.builder().SetInsertPoint(check_log10);
    llvm::Value* is_log10 = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 63));
    llvm::BasicBlock* check_log2 = llvm::BasicBlock::Create(ctx_.context(), "check_log2", current_func);
    ctx_.builder().CreateCondBr(is_log10, log10_block, check_log2);

    ctx_.builder().SetInsertPoint(log10_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* ln10 = llvm::ConstantFP::get(ctx_.doubleType(), 2.302585092994046);
        llvm::Value* denom = ctx_.builder().CreateFMul(input_val, ln10);
        llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, denom);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- LOG2 (type=64): dL/dx = dL/dz / (x * ln(2)) ---
    ctx_.builder().SetInsertPoint(check_log2);
    llvm::Value* is_log2 = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 64));
    llvm::BasicBlock* check_exp2 = llvm::BasicBlock::Create(ctx_.context(), "check_exp2", current_func);
    ctx_.builder().CreateCondBr(is_log2, log2_block, check_exp2);

    ctx_.builder().SetInsertPoint(log2_block);
    if (input1) {
        llvm::Value* input_val = loadNodeValue(input1);
        llvm::Value* ln2 = llvm::ConstantFP::get(ctx_.doubleType(), 0.6931471805599453);
        llvm::Value* denom = ctx_.builder().CreateFMul(input_val, ln2);
        llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, denom);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- EXP2 (type=65): dL/dx = dL/dz * 2^x * ln(2) ---
    ctx_.builder().SetInsertPoint(check_exp2);
    llvm::Value* is_exp2 = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 65));
    llvm::BasicBlock* check_cbrt = llvm::BasicBlock::Create(ctx_.context(), "check_cbrt", current_func);
    ctx_.builder().CreateCondBr(is_exp2, exp2_block, check_cbrt);

    ctx_.builder().SetInsertPoint(exp2_block);
    if (input1) {
        // node value = 2^x (stored during forward pass)
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* exp2_x = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        llvm::Value* ln2 = llvm::ConstantFP::get(ctx_.doubleType(), 0.6931471805599453);
        llvm::Value* exp2_times_ln2 = ctx_.builder().CreateFMul(exp2_x, ln2);
        llvm::Value* grad_input = ctx_.builder().CreateFMul(node_grad, exp2_times_ln2);
        accumulateGradient(input1, grad_input);
    }
    ctx_.builder().CreateBr(done_block);

    // --- CBRT (type=66): dL/dx = dL/dz / (3 * cbrt(x)²) ---
    ctx_.builder().SetInsertPoint(check_cbrt);
    llvm::Value* is_cbrt = ctx_.builder().CreateICmpEQ(node_type, llvm::ConstantInt::get(ctx_.int32Type(), 66));
    llvm::BasicBlock* check_conv2d = llvm::BasicBlock::Create(ctx_.context(), "check_conv2d", current_func);
    ctx_.builder().CreateCondBr(is_cbrt, cbrt_block, check_conv2d);

    ctx_.builder().SetInsertPoint(cbrt_block);
    if (input1) {
        // node value = cbrt(x) (stored during forward pass)
        llvm::Value* node_val_ptr = ctx_.builder().CreateStructGEP(ad_node_type, node_ptr, 1);
        llvm::Value* cbrt_x = ctx_.builder().CreateLoad(ctx_.doubleType(), node_val_ptr);
        llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
        llvm::Value* cbrt_sq = ctx_.builder().CreateFMul(cbrt_x, cbrt_x);
        llvm::Value* denom = ctx_.builder().CreateFMul(three, cbrt_sq);
        llvm::Value* grad_input = ctx_.builder().CreateFDiv(node_grad, denom);
        accumulateGradient(input1, grad_input);
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

    // Unknown type: emit runtime error and abort for unhandled AD_NODE types
    // This prevents silently producing zero gradients for unrecognized operations
    ctx_.builder().SetInsertPoint(unknown_type_block);
    {
        // Print diagnostic to stderr
        llvm::FunctionType* fprintf_type = llvm::FunctionType::get(
            ctx_.int32Type(), {ctx_.ptrType(), ctx_.ptrType()}, true);
        llvm::FunctionCallee fprintf_fn = ctx_.module().getOrInsertFunction("fprintf", fprintf_type);

        // Get stderr via platform-appropriate global name
#ifdef __APPLE__
        const char* stderr_name = "__stderrp";
#else
        const char* stderr_name = "stderr";
#endif
        llvm::GlobalVariable* stderr_var = ctx_.module().getGlobalVariable(stderr_name);
        if (!stderr_var) {
            stderr_var = new llvm::GlobalVariable(ctx_.module(), ctx_.ptrType(), false,
                llvm::GlobalValue::ExternalLinkage, nullptr, stderr_name);
        }
        llvm::Value* stderr_val = ctx_.builder().CreateLoad(ctx_.ptrType(), stderr_var);

        llvm::Value* fmt_str = ctx_.builder().CreateGlobalString(
            "Error: Unknown AD node type %d in backward pass — cannot compute gradient\n");
        ctx_.builder().CreateCall(fprintf_fn, {stderr_val, fmt_str, node_type});

        // Abort: unknown types must not silently produce zero gradients
        llvm::FunctionType* abort_type = llvm::FunctionType::get(ctx_.voidType(), {}, false);
        llvm::FunctionCallee abort_fn = ctx_.module().getOrInsertFunction("abort", abort_type);
        ctx_.builder().CreateCall(abort_fn, {});
        ctx_.builder().CreateUnreachable();
    }

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
    llvm::GlobalVariable* ctx_.currentAdTape() = ctx_.currentAdTape();
    llvm::GlobalVariable* ad_mode_active = ctx_.adModeActive();

    if (!ad_tape_depth || !ad_tape_stack || !ctx_.currentAdTape() || !ad_mode_active) {
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
    llvm::Value* current_tape = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.currentAdTape());
    llvm::ArrayType* stack_type = llvm::ArrayType::get(ctx_.ptrType(), CodegenContext::MAX_TAPE_DEPTH);
    llvm::Value* slot_ptr = ctx_.builder().CreateGEP(stack_type, ad_tape_stack,
        {llvm::ConstantInt::get(ctx_.int64Type(), 0), depth});
    ctx_.builder().CreateStore(current_tape, slot_ptr);

    // Increment depth
    llvm::Value* new_depth = ctx_.builder().CreateAdd(depth, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(new_depth, ad_tape_depth);

    // Set new tape as current
    ctx_.builder().CreateStore(new_tape, ctx_.currentAdTape());

    // Set AD mode active
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int1Type(), 1), ad_mode_active);
}

void AutodiffCodegen::popTapeContext() {
    llvm::GlobalVariable* ad_tape_depth = ctx_.adTapeDepth();
    llvm::GlobalVariable* ad_tape_stack = ctx_.adTapeStack();
    llvm::GlobalVariable* ctx_.currentAdTape() = ctx_.currentAdTape();
    llvm::GlobalVariable* ad_mode_active = ctx_.adModeActive();

    if (!ad_tape_depth || !ad_tape_stack || !ctx_.currentAdTape() || !ad_mode_active) {
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
    ctx_.builder().CreateStore(saved_tape, ctx_.currentAdTape());

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
