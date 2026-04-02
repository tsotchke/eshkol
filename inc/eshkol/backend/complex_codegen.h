/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * ComplexCodegen - LLVM code generation for complex number operations
 *
 * This module generates LLVM IR for complex number arithmetic and functions,
 * following the same pattern as AutodiffCodegen for dual numbers.
 *
 * Complex number representation:
 *   - Stored as heap-allocated struct { double real; double imag; }
 *   - Tagged with ESHKOL_VALUE_COMPLEX (type tag 7)
 *   - 16 bytes total (same as dual numbers)
 *
 * Supported operations:
 *   - Arithmetic: add, sub, mul, div, neg
 *   - Accessors: real-part, imag-part, magnitude, angle
 *   - Constructors: make-rectangular, make-polar
 *   - Predicates: complex?
 *   - Transcendental: exp, log, sin, cos, sqrt (complex versions)
 */
#ifndef ESHKOL_BACKEND_COMPLEX_CODEGEN_H
#define ESHKOL_BACKEND_COMPLEX_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <llvm/IR/Value.h>

namespace eshkol {

class CodegenContext;
class TaggedValueCodegen;
class MemoryCodegen;

/**
 * ComplexCodegen handles LLVM IR generation for complex number operations.
 *
 * Usage:
 *   ComplexCodegen complex(ctx, tagged, mem);
 *   auto z = complex.createComplex(real, imag);
 *   auto sum = complex.complexAdd(z1, z2);
 *   auto tagged = complex.packComplexToTagged(sum);
 */
class ComplexCodegen {
public:
    ComplexCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem);

    // ═══════════════════════════════════════════════════════════════════════
    // COMPLEX NUMBER CREATION AND ACCESS
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Create a complex number from real and imaginary components.
     * @param real Real part (double)
     * @param imag Imaginary part (double)
     * @return LLVM value representing the complex struct
     */
    llvm::Value* createComplex(llvm::Value* real, llvm::Value* imag);

    /**
     * Extract the real part from a complex number.
     * @param complex Complex number struct
     * @return Real component as double
     */
    llvm::Value* getComplexReal(llvm::Value* complex);

    /**
     * Extract the imaginary part from a complex number.
     * @param complex Complex number struct
     * @return Imaginary component as double
     */
    llvm::Value* getComplexImag(llvm::Value* complex);

    // ═══════════════════════════════════════════════════════════════════════
    // TAGGED VALUE CONVERSION
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Pack a complex number struct into a tagged value.
     * Allocates heap memory and stores the complex number.
     * @param complex Complex number struct
     * @return Tagged value with ESHKOL_VALUE_COMPLEX type
     */
    llvm::Value* packComplexToTagged(llvm::Value* complex);

    /**
     * Unpack a complex number from a tagged value.
     * @param tagged_val Tagged value containing complex number
     * @return Complex number struct
     */
    llvm::Value* unpackComplexFromTagged(llvm::Value* tagged_val);

    // ═══════════════════════════════════════════════════════════════════════
    // COMPLEX ARITHMETIC OPERATIONS
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i
     */
    llvm::Value* complexAdd(llvm::Value* z1, llvm::Value* z2);

    /**
     * Complex subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i
     */
    llvm::Value* complexSub(llvm::Value* z1, llvm::Value* z2);

    /**
     * Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
     */
    llvm::Value* complexMul(llvm::Value* z1, llvm::Value* z2);

    /**
     * Complex division: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
     */
    llvm::Value* complexDiv(llvm::Value* z1, llvm::Value* z2);

    /**
     * Complex negation: -(a+bi) = -a + (-b)i
     */
    llvm::Value* complexNeg(llvm::Value* z);

    /**
     * Complex conjugate: conj(a+bi) = a - bi
     */
    llvm::Value* complexConj(llvm::Value* z);

    // ═══════════════════════════════════════════════════════════════════════
    // COMPLEX MATHEMATICAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Complex magnitude: |a+bi| = sqrt(a² + b²)
     * @return Double value
     */
    llvm::Value* complexMagnitude(llvm::Value* z);

    /**
     * Complex angle (argument): arg(a+bi) = atan2(b, a)
     * @return Double value in radians
     */
    llvm::Value* complexAngle(llvm::Value* z);

    /**
     * Complex exponential: exp(a+bi) = exp(a)(cos(b) + i*sin(b))
     */
    llvm::Value* complexExp(llvm::Value* z);

    /**
     * Complex natural logarithm: log(z) = log|z| + i*arg(z)
     */
    llvm::Value* complexLog(llvm::Value* z);

    /**
     * Complex square root: sqrt(z) using principal branch
     */
    llvm::Value* complexSqrt(llvm::Value* z);

    /**
     * Complex sine: sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
     */
    llvm::Value* complexSin(llvm::Value* z);

    /**
     * Complex cosine: cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
     */
    llvm::Value* complexCos(llvm::Value* z);

    // ═══════════════════════════════════════════════════════════════════════
    // POLAR FORM CONVERSION
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Create complex from polar form: r * e^(i*theta) = r*cos(theta) + i*r*sin(theta)
     * @param magnitude Radius (r)
     * @param angle Angle in radians (theta)
     */
    llvm::Value* makeFromPolar(llvm::Value* magnitude, llvm::Value* angle);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    MemoryCodegen& mem_;

    // Helper to get intrinsic functions
    llvm::Function* getSqrtIntrinsic();
    llvm::Function* getSinIntrinsic();
    llvm::Function* getCosIntrinsic();
    llvm::Function* getExpIntrinsic();
    llvm::Function* getLogIntrinsic();
    llvm::Function* getAtan2Intrinsic();
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_COMPLEX_CODEGEN_H
