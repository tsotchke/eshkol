/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Non-colliding access to the libm scalar math functions from codegen.
 *
 * WHY THIS EXISTS
 * ---------------
 * Codegen used to materialise a libm function by BARE NAME:
 *
 *     llvm::Function* exp_func = ctx_.module().getFunction("exp");
 *     if (!exp_func) exp_func = llvm::Function::Create(
 *         llvm::FunctionType::get(double, {double}, false),
 *         llvm::Function::ExternalLinkage, "exp", &ctx_.module());
 *
 * A module-level name is not a namespace. `@exp` in an Eshkol module is
 * whatever claimed the name first, and there are three claimants: the builtin
 * factory's libm declarations, a lowering like this one, and the user program
 * itself (`(define (exp x) ...)` emits a `%eshkol_tagged_value (%eshkol_tagged_value)`
 * function). The pattern above takes whatever it finds and calls it with
 * `double` arguments WITHOUT checking the type, so both of the following were
 * real, reproducible defects:
 *
 *   (define t (tensor (list -2.0 1.0)))
 *   (display (tensor-ref (elu t 2) 0))   ; declares a bare @exp
 *   (display (exp -2.0))                 ; SIGSEGV in the compiler
 *
 *   (define (exp x) (+ x 1))             ; user @exp, tagged ABI
 *   (display (tensor-ref (elu t 2) 0))   ; emits call double @exp(double) -> to it
 *
 * The fix is to stop naming the symbol at all where LLVM gives a namespaced
 * alternative. `llvm.exp.f64` is an INTRINSIC: its name is reserved, no module
 * symbol can shadow it, and the backend lowers it to the same libm call. That
 * is already how TensorCodegen::tensorSigmoid obtained `exp`, which is exactly
 * why `sigmoid` never reproduced the crash while `elu`/`selu`/`celu`/`silu`/
 * `mish`/`softplus` all did.
 *
 * Not every libm function has an intrinsic on every LLVM major Eshkol builds
 * against (18/19/21, plus the StableHLO lane's 24) — `tanh` only arrived in
 * LLVM 19 and `atan2` in LLVM 20. For those, this header falls back to a
 * name lookup that VERIFIES the found function's type, and refuses to bind to
 * a same-named function with a different signature: it declares a distinctly
 * named one instead, so a collision that somehow survives every guard is a
 * loud link failure rather than a call through the wrong ABI.
 *
 * The remaining libm names Eshkol uses that have no intrinsic on any supported
 * LLVM (asinh/acosh/atanh/cbrt/fmod/remainder/nextafter) are declared by
 * BuiltinFactoryCodegen at module init, BEFORE any user or lowering code runs,
 * so the type-verified path finds the libm declaration itself.
 */
#ifndef ESHKOL_BACKEND_LIBM_CODEGEN_H
#define ESHKOL_BACKEND_LIBM_CODEGEN_H

#include <llvm/Config/llvm-config.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <string>

namespace eshkol {
namespace libm_codegen {

/// Intrinsic::getDeclaration (<21) / getOrInsertDeclaration (>=21).
///
/// The pre-existing ESHKOL_GET_INTRINSIC macro does the same thing but is
/// defined per translation unit with non-identical replacement lists (see the
/// note at the top of llvm_compat.h), so this header carries its own spelling
/// rather than forcing a macro redefinition on every includer.
inline llvm::Function* getOrInsertIntrinsic(llvm::Module& module,
                                            llvm::Intrinsic::ID id,
                                            llvm::ArrayRef<llvm::Type*> overload_types) {
#if LLVM_VERSION_MAJOR >= 21
    return llvm::Intrinsic::getOrInsertDeclaration(&module, id, overload_types);
#else
    return llvm::Intrinsic::getDeclaration(&module, id, overload_types);
#endif
}

/// The LLVM intrinsic that IS this libm function, or `not_intrinsic` when this
/// LLVM major has none.
///
/// Only entries that are exact replacements are listed: `llvm.exp.f64` has the
/// semantics of `exp(double)` and lowers to it. Rounding modes are deliberately
/// absent — `round`/`trunc`/`floor`/`ceil` map to their intrinsics, but
/// `nearbyint`/`rint` do not appear here because no caller needs them and their
/// FP-environment behaviour is not interchangeable.
///
/// The one difference these intrinsics carry is that they do not set errno.
/// Eshkol never reads errno after a math call -- the scalar builtins dispatch
/// on the VALUE (R7RS numeric-tower promotion for a negative exact sqrt/log,
/// IEEE NaN/inf for an inexact one), decided before the call is emitted -- so
/// the observable answers are unchanged.
inline llvm::Intrinsic::ID intrinsicForLibm(llvm::StringRef name) {
    // Present on every LLVM major Eshkol builds against (18 and later).
    if (name == "exp")   return llvm::Intrinsic::exp;
    if (name == "exp2")  return llvm::Intrinsic::exp2;
    if (name == "log")   return llvm::Intrinsic::log;
    if (name == "log2")  return llvm::Intrinsic::log2;
    if (name == "log10") return llvm::Intrinsic::log10;
    if (name == "sqrt")  return llvm::Intrinsic::sqrt;
    if (name == "sin")   return llvm::Intrinsic::sin;
    if (name == "cos")   return llvm::Intrinsic::cos;
    if (name == "pow")   return llvm::Intrinsic::pow;
    if (name == "fabs")  return llvm::Intrinsic::fabs;
    if (name == "floor") return llvm::Intrinsic::floor;
    if (name == "ceil")  return llvm::Intrinsic::ceil;
    if (name == "trunc") return llvm::Intrinsic::trunc;
    if (name == "round") return llvm::Intrinsic::round;
    if (name == "fmin")  return llvm::Intrinsic::minnum;
    if (name == "fmax")  return llvm::Intrinsic::maxnum;
    if (name == "copysign") return llvm::Intrinsic::copysign;
#if LLVM_VERSION_MAJOR >= 19
    // The trigonometric/hyperbolic family landed in LLVM 19.
    if (name == "tan")   return llvm::Intrinsic::tan;
    if (name == "asin")  return llvm::Intrinsic::asin;
    if (name == "acos")  return llvm::Intrinsic::acos;
    if (name == "atan")  return llvm::Intrinsic::atan;
    if (name == "sinh")  return llvm::Intrinsic::sinh;
    if (name == "cosh")  return llvm::Intrinsic::cosh;
    if (name == "tanh")  return llvm::Intrinsic::tanh;
#endif
#if LLVM_VERSION_MAJOR >= 20
    if (name == "atan2") return llvm::Intrinsic::atan2;
#endif
    return llvm::Intrinsic::not_intrinsic;
}

namespace detail {

/// Name-lookup fallback for a libm function with no intrinsic on this LLVM
/// major. Never binds to a same-named function of a different type.
inline llvm::Function* declareByVerifiedName(llvm::Module& module,
                                             llvm::StringRef name,
                                             llvm::FunctionType* wanted) {
    if (llvm::Function* existing = module.getFunction(name)) {
        if (existing->getFunctionType() == wanted) {
            return existing;
        }
        // The name is taken by something that is NOT this libm function.
        // Binding to it would emit a call through the wrong ABI, so declare a
        // distinctly named one: unresolved at link is a loud failure, a wrong
        // call is a silent one. BuiltinFactoryCodegen claims every libm name
        // at module init precisely so this branch is unreachable in practice.
        std::string distinct = ("eshkol_libm_" + name).str();
        if (llvm::Function* prior = module.getFunction(distinct)) {
            if (prior->getFunctionType() == wanted) return prior;
        }
        return llvm::Function::Create(wanted, llvm::Function::ExternalLinkage,
                                      distinct, &module);
    }
    return llvm::Function::Create(wanted, llvm::Function::ExternalLinkage,
                                  name, &module);
}

}  // namespace detail

/// `double name(double)` — as an intrinsic where one exists, so no module
/// symbol of the same name can ever be called instead.
///
/// `ty` may be a vector of doubles for an intrinsic-backed name, which yields
/// the vector overload (e.g. `llvm.exp.v4f64`); a vector type is rejected on
/// the name-lookup fallback, where there is no vector libm symbol to call.
inline llvm::Function* unary(llvm::Module& module, llvm::StringRef name, llvm::Type* ty) {
    llvm::Intrinsic::ID id = intrinsicForLibm(name);
    if (id != llvm::Intrinsic::not_intrinsic) {
        return getOrInsertIntrinsic(module, id, {ty});
    }
    if (ty->isVectorTy()) return nullptr;
    return detail::declareByVerifiedName(
        module, name, llvm::FunctionType::get(ty, {ty}, false));
}

/// `double name(double, double)` — see unary().
inline llvm::Function* binary(llvm::Module& module, llvm::StringRef name, llvm::Type* ty) {
    llvm::Intrinsic::ID id = intrinsicForLibm(name);
    if (id != llvm::Intrinsic::not_intrinsic) {
        return getOrInsertIntrinsic(module, id, {ty});
    }
    if (ty->isVectorTy()) return nullptr;
    return detail::declareByVerifiedName(
        module, name, llvm::FunctionType::get(ty, {ty, ty}, false));
}

}  // namespace libm_codegen
}  // namespace eshkol

#endif  // ESHKOL_BACKEND_LIBM_CODEGEN_H
