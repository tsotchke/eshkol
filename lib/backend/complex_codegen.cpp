/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * ComplexCodegen implementation - LLVM code generation for complex numbers
 */

#include <eshkol/backend/complex_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/memory_codegen.h>
#include <eshkol/backend/type_system.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>

namespace eshkol {

ComplexCodegen::ComplexCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem)
    : ctx_(ctx), tagged_(tagged), mem_(mem) {}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX NUMBER CREATION AND ACCESS
// ═══════════════════════════════════════════════════════════════════════════

llvm::Value* ComplexCodegen::createComplex(llvm::Value* real, llvm::Value* imag) {
    // Allocate complex struct on stack at function entry for optimal codegen
    llvm::IRBuilderBase::InsertPoint saved_ip = ctx_.builder().saveIP();
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock& entry = func->getEntryBlock();
    ctx_.builder().SetInsertPoint(&entry, entry.begin());

    llvm::Value* complex_ptr = ctx_.builder().CreateAlloca(
        ctx_.complexNumberType(), nullptr, "complex_alloca");
    ctx_.builder().restoreIP(saved_ip);

    // Store real component (field 0)
    llvm::Value* real_ptr = ctx_.builder().CreateStructGEP(
        ctx_.complexNumberType(), complex_ptr, TypeSystem::COMPLEX_REAL_IDX, "real_ptr");
    ctx_.builder().CreateStore(real, real_ptr);

    // Store imaginary component (field 1)
    llvm::Value* imag_ptr = ctx_.builder().CreateStructGEP(
        ctx_.complexNumberType(), complex_ptr, TypeSystem::COMPLEX_IMAG_IDX, "imag_ptr");
    ctx_.builder().CreateStore(imag, imag_ptr);

    // Load and return the complex struct
    return ctx_.builder().CreateLoad(ctx_.complexNumberType(), complex_ptr, "complex");
}

llvm::Value* ComplexCodegen::getComplexReal(llvm::Value* complex) {
    // Extract real part from complex struct (field 0)
    return ctx_.builder().CreateExtractValue(complex, {TypeSystem::COMPLEX_REAL_IDX}, "real");
}

llvm::Value* ComplexCodegen::getComplexImag(llvm::Value* complex) {
    // Extract imaginary part from complex struct (field 1)
    return ctx_.builder().CreateExtractValue(complex, {TypeSystem::COMPLEX_IMAG_IDX}, "imag");
}

// ═══════════════════════════════════════════════════════════════════════════
// TAGGED VALUE CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

llvm::Value* ComplexCodegen::packComplexToTagged(llvm::Value* complex) {
    // Get global arena for heap allocation
    llvm::GlobalVariable* arena_global = ctx_.module().getNamedGlobal("__global_arena");
    if (!arena_global) {
        // Create external declaration if not present
        arena_global = new llvm::GlobalVariable(
            ctx_.module(),
            ctx_.ptrType(),
            false,
            llvm::GlobalValue::ExternalLinkage,
            nullptr,
            "__global_arena");
    }
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), arena_global, "arena");

    // Allocate 16 bytes for complex number on heap
    llvm::Value* size = llvm::ConstantInt::get(ctx_.int64Type(), 16);
    llvm::Function* alloc_func = mem_.getArenaAllocate();
    llvm::Value* complex_heap_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr, size}, "complex_ptr");

    // Null check: arena allocation can fail
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* alloc_ok_bb = llvm::BasicBlock::Create(ctx_.context(), "complex_alloc_ok", current_func);
    llvm::BasicBlock* alloc_fail_bb = llvm::BasicBlock::Create(ctx_.context(), "complex_alloc_fail", current_func);

    llvm::Value* is_null = ctx_.builder().CreateICmpEQ(complex_heap_ptr,
        llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_.context(), 0)), "alloc_null");
    ctx_.builder().CreateCondBr(is_null, alloc_fail_bb, alloc_ok_bb);

    // Fail path: print error and exit
    ctx_.builder().SetInsertPoint(alloc_fail_bb);
    llvm::Function* printf_func = ctx_.lookupFunction("printf");
    llvm::Function* exit_func = ctx_.lookupFunction("exit");
    if (printf_func && exit_func) {
        llvm::Value* err_msg = ctx_.builder().CreateGlobalStringPtr(
            "Error: arena allocation failed for complex number (16 bytes)\n");
        ctx_.builder().CreateCall(printf_func, {err_msg});
        ctx_.builder().CreateCall(exit_func, {llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx_.context()), 1)});
    }
    ctx_.builder().CreateUnreachable();

    // Success path: continue
    ctx_.builder().SetInsertPoint(alloc_ok_bb);

    // Store complex struct to heap
    ctx_.builder().CreateStore(complex, complex_heap_ptr);

    // Pack pointer as tagged value with COMPLEX type
    llvm::Value* ptr_as_int = ctx_.builder().CreatePtrToInt(complex_heap_ptr, ctx_.int64Type(), "ptr_int");
    return tagged_.packPtr(complex_heap_ptr, ESHKOL_VALUE_COMPLEX);
}

llvm::Value* ComplexCodegen::unpackComplexFromTagged(llvm::Value* tagged_val) {
    // Extract pointer from tagged value
    llvm::Value* ptr = tagged_.unpackPtr(tagged_val);

    // Load and return complex struct
    return ctx_.builder().CreateLoad(ctx_.complexNumberType(), ptr, "complex");
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX ARITHMETIC OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

llvm::Value* ComplexCodegen::complexAdd(llvm::Value* z1, llvm::Value* z2) {
    // (a+bi) + (c+di) = (a+c) + (b+d)i
    llvm::Value* a = getComplexReal(z1);
    llvm::Value* b = getComplexImag(z1);
    llvm::Value* c = getComplexReal(z2);
    llvm::Value* d = getComplexImag(z2);

    llvm::Value* real = ctx_.builder().CreateFAdd(a, c, "add_real");
    llvm::Value* imag = ctx_.builder().CreateFAdd(b, d, "add_imag");

    return createComplex(real, imag);
}

llvm::Value* ComplexCodegen::complexSub(llvm::Value* z1, llvm::Value* z2) {
    // (a+bi) - (c+di) = (a-c) + (b-d)i
    llvm::Value* a = getComplexReal(z1);
    llvm::Value* b = getComplexImag(z1);
    llvm::Value* c = getComplexReal(z2);
    llvm::Value* d = getComplexImag(z2);

    llvm::Value* real = ctx_.builder().CreateFSub(a, c, "sub_real");
    llvm::Value* imag = ctx_.builder().CreateFSub(b, d, "sub_imag");

    return createComplex(real, imag);
}

llvm::Value* ComplexCodegen::complexMul(llvm::Value* z1, llvm::Value* z2) {
    // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    llvm::Value* a = getComplexReal(z1);
    llvm::Value* b = getComplexImag(z1);
    llvm::Value* c = getComplexReal(z2);
    llvm::Value* d = getComplexImag(z2);

    // real = a*c - b*d
    llvm::Value* ac = ctx_.builder().CreateFMul(a, c, "ac");
    llvm::Value* bd = ctx_.builder().CreateFMul(b, d, "bd");
    llvm::Value* real = ctx_.builder().CreateFSub(ac, bd, "mul_real");

    // imag = a*d + b*c
    llvm::Value* ad = ctx_.builder().CreateFMul(a, d, "ad");
    llvm::Value* bc = ctx_.builder().CreateFMul(b, c, "bc");
    llvm::Value* imag = ctx_.builder().CreateFAdd(ad, bc, "mul_imag");

    return createComplex(real, imag);
}

llvm::Value* ComplexCodegen::complexDiv(llvm::Value* z1, llvm::Value* z2) {
    // Smith's formula: overflow-safe complex division
    // If |d| <= |c|: r = d/c, denom = c + d*r
    //   real = (a + b*r) / denom, imag = (b - a*r) / denom
    // Else: r = c/d, denom = d + c*r
    //   real = (a*r + b) / denom, imag = (b*r - a) / denom
    llvm::Value* a = getComplexReal(z1);
    llvm::Value* b = getComplexImag(z1);
    llvm::Value* c = getComplexReal(z2);
    llvm::Value* d = getComplexImag(z2);

    llvm::Function* fabs_fn = llvm::Intrinsic::getDeclaration(
        &ctx_.module(), llvm::Intrinsic::fabs, {ctx_.doubleType()});
    llvm::Value* abs_c = ctx_.builder().CreateCall(fabs_fn, {c}, "abs_c");
    llvm::Value* abs_d = ctx_.builder().CreateCall(fabs_fn, {d}, "abs_d");
    llvm::Value* d_le_c = ctx_.builder().CreateFCmpOLE(abs_d, abs_c, "d_le_c");

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* path1_bb = llvm::BasicBlock::Create(ctx_.context(), "div_path1", current_func);
    llvm::BasicBlock* path2_bb = llvm::BasicBlock::Create(ctx_.context(), "div_path2", current_func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "div_merge", current_func);

    ctx_.builder().CreateCondBr(d_le_c, path1_bb, path2_bb);

    // Path 1: |d| <= |c| — divide by c
    ctx_.builder().SetInsertPoint(path1_bb);
    llvm::Value* r1 = ctx_.builder().CreateFDiv(d, c, "r1");
    llvm::Value* denom1 = ctx_.builder().CreateFAdd(c,
        ctx_.builder().CreateFMul(d, r1, "dr1"), "denom1");
    llvm::Value* real1 = ctx_.builder().CreateFDiv(
        ctx_.builder().CreateFAdd(a,
            ctx_.builder().CreateFMul(b, r1, "br1"), "num_re1"),
        denom1, "re1");
    llvm::Value* imag1 = ctx_.builder().CreateFDiv(
        ctx_.builder().CreateFSub(b,
            ctx_.builder().CreateFMul(a, r1, "ar1"), "num_im1"),
        denom1, "im1");
    ctx_.builder().CreateBr(merge_bb);

    // Path 2: |d| > |c| — divide by d
    ctx_.builder().SetInsertPoint(path2_bb);
    llvm::Value* r2 = ctx_.builder().CreateFDiv(c, d, "r2");
    llvm::Value* denom2 = ctx_.builder().CreateFAdd(d,
        ctx_.builder().CreateFMul(c, r2, "cr2"), "denom2");
    llvm::Value* real2 = ctx_.builder().CreateFDiv(
        ctx_.builder().CreateFAdd(
            ctx_.builder().CreateFMul(a, r2, "ar2"), b, "num_re2"),
        denom2, "re2");
    llvm::Value* imag2 = ctx_.builder().CreateFDiv(
        ctx_.builder().CreateFSub(
            ctx_.builder().CreateFMul(b, r2, "br2"), a, "num_im2"),
        denom2, "im2");
    ctx_.builder().CreateBr(merge_bb);

    // Merge
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* real = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "div_real");
    real->addIncoming(real1, path1_bb);
    real->addIncoming(real2, path2_bb);
    llvm::PHINode* imag = ctx_.builder().CreatePHI(ctx_.doubleType(), 2, "div_imag");
    imag->addIncoming(imag1, path1_bb);
    imag->addIncoming(imag2, path2_bb);

    return createComplex(real, imag);
}

llvm::Value* ComplexCodegen::complexNeg(llvm::Value* z) {
    // -(a+bi) = -a - bi
    llvm::Value* a = getComplexReal(z);
    llvm::Value* b = getComplexImag(z);

    llvm::Value* neg_a = ctx_.builder().CreateFNeg(a, "neg_real");
    llvm::Value* neg_b = ctx_.builder().CreateFNeg(b, "neg_imag");

    return createComplex(neg_a, neg_b);
}

llvm::Value* ComplexCodegen::complexConj(llvm::Value* z) {
    // conj(a+bi) = a - bi
    llvm::Value* a = getComplexReal(z);
    llvm::Value* b = getComplexImag(z);

    llvm::Value* neg_b = ctx_.builder().CreateFNeg(b, "neg_imag");

    return createComplex(a, neg_b);
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX MATHEMATICAL FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

llvm::Value* ComplexCodegen::complexMagnitude(llvm::Value* z) {
    // Overflow-safe magnitude: max(|a|,|b|) * sqrt((a/max)² + (b/max)²)
    // Prevents overflow when a or b are near DBL_MAX (~1e308)
    llvm::Value* a = getComplexReal(z);
    llvm::Value* b = getComplexImag(z);

    llvm::Function* fabs_fn = llvm::Intrinsic::getDeclaration(
        &ctx_.module(), llvm::Intrinsic::fabs, {ctx_.doubleType()});
    llvm::Function* sqrt_fn = getSqrtIntrinsic();
    llvm::Function* maxnum_fn = llvm::Intrinsic::getDeclaration(
        &ctx_.module(), llvm::Intrinsic::maxnum, {ctx_.doubleType()});

    llvm::Value* abs_a = ctx_.builder().CreateCall(fabs_fn, {a}, "abs_a");
    llvm::Value* abs_b = ctx_.builder().CreateCall(fabs_fn, {b}, "abs_b");
    llvm::Value* max_ab = ctx_.builder().CreateCall(maxnum_fn, {abs_a, abs_b}, "max_ab");

    // Guard against zero (avoid 0/0)
    llvm::Value* is_zero = ctx_.builder().CreateFCmpOEQ(max_ab,
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0), "is_zero");

    llvm::Value* a_scaled = ctx_.builder().CreateFDiv(a, max_ab, "a_scaled");
    llvm::Value* b_scaled = ctx_.builder().CreateFDiv(b, max_ab, "b_scaled");
    llvm::Value* a2 = ctx_.builder().CreateFMul(a_scaled, a_scaled, "a2");
    llvm::Value* b2 = ctx_.builder().CreateFMul(b_scaled, b_scaled, "b2");
    llvm::Value* sum = ctx_.builder().CreateFAdd(a2, b2, "sum_sq");
    llvm::Value* scaled_mag = ctx_.builder().CreateFMul(max_ab,
        ctx_.builder().CreateCall(sqrt_fn, {sum}, "sqrt_sum"), "scaled_mag");

    // Return 0 if both components are zero, otherwise scaled magnitude
    return ctx_.builder().CreateSelect(is_zero,
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0), scaled_mag, "magnitude");
}

llvm::Value* ComplexCodegen::complexAngle(llvm::Value* z) {
    // arg(a+bi) = atan2(b, a)
    llvm::Value* a = getComplexReal(z);
    llvm::Value* b = getComplexImag(z);

    llvm::Function* atan2_fn = getAtan2Intrinsic();
    return ctx_.builder().CreateCall(atan2_fn, {b, a}, "angle");
}

llvm::Value* ComplexCodegen::complexExp(llvm::Value* z) {
    // exp(a+bi) = exp(a)(cos(b) + i*sin(b))
    llvm::Value* a = getComplexReal(z);
    llvm::Value* b = getComplexImag(z);

    llvm::Function* exp_fn = getExpIntrinsic();
    llvm::Function* sin_fn = getSinIntrinsic();
    llvm::Function* cos_fn = getCosIntrinsic();

    llvm::Value* exp_a = ctx_.builder().CreateCall(exp_fn, {a}, "exp_a");
    llvm::Value* cos_b = ctx_.builder().CreateCall(cos_fn, {b}, "cos_b");
    llvm::Value* sin_b = ctx_.builder().CreateCall(sin_fn, {b}, "sin_b");

    llvm::Value* real = ctx_.builder().CreateFMul(exp_a, cos_b, "exp_real");
    llvm::Value* imag = ctx_.builder().CreateFMul(exp_a, sin_b, "exp_imag");

    return createComplex(real, imag);
}

llvm::Value* ComplexCodegen::complexLog(llvm::Value* z) {
    // log(z) = log|z| + i*arg(z)
    llvm::Value* mag = complexMagnitude(z);
    llvm::Value* ang = complexAngle(z);

    llvm::Function* log_fn = getLogIntrinsic();
    llvm::Value* real = ctx_.builder().CreateCall(log_fn, {mag}, "log_mag");

    return createComplex(real, ang);
}

llvm::Value* ComplexCodegen::complexSqrt(llvm::Value* z) {
    // sqrt(z) = sqrt(|z|) * (cos(arg(z)/2) + i*sin(arg(z)/2))
    llvm::Value* mag = complexMagnitude(z);
    llvm::Value* ang = complexAngle(z);

    llvm::Function* sqrt_fn = getSqrtIntrinsic();
    llvm::Function* sin_fn = getSinIntrinsic();
    llvm::Function* cos_fn = getCosIntrinsic();

    llvm::Value* sqrt_mag = ctx_.builder().CreateCall(sqrt_fn, {mag}, "sqrt_mag");
    llvm::Value* half_ang = ctx_.builder().CreateFMul(ang,
        llvm::ConstantFP::get(ctx_.doubleType(), 0.5), "half_ang");

    llvm::Value* cos_half = ctx_.builder().CreateCall(cos_fn, {half_ang}, "cos_half");
    llvm::Value* sin_half = ctx_.builder().CreateCall(sin_fn, {half_ang}, "sin_half");

    llvm::Value* real = ctx_.builder().CreateFMul(sqrt_mag, cos_half, "sqrt_real");
    llvm::Value* imag = ctx_.builder().CreateFMul(sqrt_mag, sin_half, "sqrt_imag");

    return createComplex(real, imag);
}

llvm::Value* ComplexCodegen::complexSin(llvm::Value* z) {
    // sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
    // Using: cosh(x) = (exp(x) + exp(-x))/2, sinh(x) = (exp(x) - exp(-x))/2
    llvm::Value* a = getComplexReal(z);
    llvm::Value* b = getComplexImag(z);

    llvm::Function* sin_fn = getSinIntrinsic();
    llvm::Function* cos_fn = getCosIntrinsic();
    llvm::Function* exp_fn = getExpIntrinsic();

    llvm::Value* sin_a = ctx_.builder().CreateCall(sin_fn, {a}, "sin_a");
    llvm::Value* cos_a = ctx_.builder().CreateCall(cos_fn, {a}, "cos_a");

    // cosh(b) = (exp(b) + exp(-b)) / 2
    llvm::Value* exp_b = ctx_.builder().CreateCall(exp_fn, {b}, "exp_b");
    llvm::Value* neg_b = ctx_.builder().CreateFNeg(b, "neg_b");
    llvm::Value* exp_neg_b = ctx_.builder().CreateCall(exp_fn, {neg_b}, "exp_neg_b");
    llvm::Value* cosh_b = ctx_.builder().CreateFMul(
        ctx_.builder().CreateFAdd(exp_b, exp_neg_b, "sum"),
        llvm::ConstantFP::get(ctx_.doubleType(), 0.5), "cosh_b");

    // sinh(b) = (exp(b) - exp(-b)) / 2
    llvm::Value* sinh_b = ctx_.builder().CreateFMul(
        ctx_.builder().CreateFSub(exp_b, exp_neg_b, "diff"),
        llvm::ConstantFP::get(ctx_.doubleType(), 0.5), "sinh_b");

    llvm::Value* real = ctx_.builder().CreateFMul(sin_a, cosh_b, "sin_real");
    llvm::Value* imag = ctx_.builder().CreateFMul(cos_a, sinh_b, "sin_imag");

    return createComplex(real, imag);
}

llvm::Value* ComplexCodegen::complexCos(llvm::Value* z) {
    // cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
    llvm::Value* a = getComplexReal(z);
    llvm::Value* b = getComplexImag(z);

    llvm::Function* sin_fn = getSinIntrinsic();
    llvm::Function* cos_fn = getCosIntrinsic();
    llvm::Function* exp_fn = getExpIntrinsic();

    llvm::Value* sin_a = ctx_.builder().CreateCall(sin_fn, {a}, "sin_a");
    llvm::Value* cos_a = ctx_.builder().CreateCall(cos_fn, {a}, "cos_a");

    // cosh(b) and sinh(b) as above
    llvm::Value* exp_b = ctx_.builder().CreateCall(exp_fn, {b}, "exp_b");
    llvm::Value* neg_b = ctx_.builder().CreateFNeg(b, "neg_b");
    llvm::Value* exp_neg_b = ctx_.builder().CreateCall(exp_fn, {neg_b}, "exp_neg_b");
    llvm::Value* cosh_b = ctx_.builder().CreateFMul(
        ctx_.builder().CreateFAdd(exp_b, exp_neg_b, "sum"),
        llvm::ConstantFP::get(ctx_.doubleType(), 0.5), "cosh_b");
    llvm::Value* sinh_b = ctx_.builder().CreateFMul(
        ctx_.builder().CreateFSub(exp_b, exp_neg_b, "diff"),
        llvm::ConstantFP::get(ctx_.doubleType(), 0.5), "sinh_b");

    llvm::Value* real = ctx_.builder().CreateFMul(cos_a, cosh_b, "cos_real");
    llvm::Value* imag_pos = ctx_.builder().CreateFMul(sin_a, sinh_b, "temp");
    llvm::Value* imag = ctx_.builder().CreateFNeg(imag_pos, "cos_imag");

    return createComplex(real, imag);
}

// ═══════════════════════════════════════════════════════════════════════════
// POLAR FORM CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

llvm::Value* ComplexCodegen::makeFromPolar(llvm::Value* magnitude, llvm::Value* angle) {
    // r * e^(i*theta) = r*cos(theta) + i*r*sin(theta)
    llvm::Function* sin_fn = getSinIntrinsic();
    llvm::Function* cos_fn = getCosIntrinsic();

    llvm::Value* cos_ang = ctx_.builder().CreateCall(cos_fn, {angle}, "cos_ang");
    llvm::Value* sin_ang = ctx_.builder().CreateCall(sin_fn, {angle}, "sin_ang");

    llvm::Value* real = ctx_.builder().CreateFMul(magnitude, cos_ang, "polar_real");
    llvm::Value* imag = ctx_.builder().CreateFMul(magnitude, sin_ang, "polar_imag");

    return createComplex(real, imag);
}

// ═══════════════════════════════════════════════════════════════════════════
// INTRINSIC HELPERS
// ═══════════════════════════════════════════════════════════════════════════

llvm::Function* ComplexCodegen::getSqrtIntrinsic() {
    return llvm::Intrinsic::getDeclaration(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
}

llvm::Function* ComplexCodegen::getSinIntrinsic() {
    return llvm::Intrinsic::getDeclaration(&ctx_.module(),
        llvm::Intrinsic::sin, {ctx_.doubleType()});
}

llvm::Function* ComplexCodegen::getCosIntrinsic() {
    return llvm::Intrinsic::getDeclaration(&ctx_.module(),
        llvm::Intrinsic::cos, {ctx_.doubleType()});
}

llvm::Function* ComplexCodegen::getExpIntrinsic() {
    return llvm::Intrinsic::getDeclaration(&ctx_.module(),
        llvm::Intrinsic::exp, {ctx_.doubleType()});
}

llvm::Function* ComplexCodegen::getLogIntrinsic() {
    return llvm::Intrinsic::getDeclaration(&ctx_.module(),
        llvm::Intrinsic::log, {ctx_.doubleType()});
}

llvm::Function* ComplexCodegen::getAtan2Intrinsic() {
    // atan2 is not an LLVM intrinsic, we need to call the C library function
    llvm::FunctionType* atan2_type = llvm::FunctionType::get(
        ctx_.doubleType(),
        {ctx_.doubleType(), ctx_.doubleType()},
        false);

    llvm::Function* atan2_fn = ctx_.module().getFunction("atan2");
    if (!atan2_fn) {
        atan2_fn = llvm::Function::Create(atan2_type,
            llvm::Function::ExternalLinkage, "atan2", &ctx_.module());
    }
    return atan2_fn;
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
