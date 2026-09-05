/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * LLVM API compatibility surface for the Eshkol codegen.
 *
 * Eshkol builds against several LLVM major versions at once: the CI lanes use
 * LLVM 18/19/21, and the StableHLO/XLA lane pins LLVM 24.  Rather than
 * scattering `#if LLVM_VERSION_MAJOR` conditionals through ~14 codegen
 * translation units, every API whose spelling changed between those versions
 * gets ONE shim here.  Each shim is written so that a wrong assumption is a
 * COMPILE error on the affected version, never silently different IR.
 *
 * NOT here: ESHKOL_GET_INTRINSIC(mod, id, types), the pre-existing
 * getDeclaration (<21) / getOrInsertDeclaration (>=21) shim.  It is defined
 * per-file in 16 translation units and those definitions are NOT all
 * token-identical — lib/backend/llvm_codegen.cpp spells it without the `llvm::`
 * qualifier, everything else spells it with.  A macro redefinition is only
 * well-formed when the replacement lists match token for token, so defining it
 * here as well would break exactly the file that needs this header most.
 * Consolidating that macro is a separate cleanup; doing it as part of the
 * LLVM 24 port would risk the LLVM 18/19/21 lanes for no benefit.
 *
 * Contents
 * --------
 *  eshkol::llvm_compat::createGlobalString(builder, str, name)
 *      Replacement for IRBuilder::CreateGlobalStringPtr, which was deprecated
 *      in LLVM 21/22 and REMOVED in LLVM 24.
 *
 *  eshkol::llvm_compat::UncondBranchInst
 *      The type IRBuilder::CreateBr returns.  LLVM <=22 spells it
 *      llvm::BranchInst; LLVM 24 split that class into llvm::UncondBrInst and
 *      llvm::CondBrInst, so `llvm::BranchInst` no longer names anything.
 *
 *  eshkol::llvm_compat::intrinsicSignatureMatches(id, fty, overload_types)
 *      Replacement for Intrinsic::matchIntrinsicSignature, removed in LLVM 24.
 */
#ifndef ESHKOL_BACKEND_LLVM_COMPAT_H
#define ESHKOL_BACKEND_LLVM_COMPAT_H

#include <llvm/Config/llvm-config.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Type.h>

#include <type_traits>
#include <utility>

namespace eshkol {
namespace llvm_compat {

// ---------------------------------------------------------------------------
// String constants.
//
// IRBuilder::CreateGlobalStringPtr was deprecated in LLVM 21/22 ("Use
// CreateGlobalString instead") and removed outright in LLVM 24.  In every
// LLVM version Eshkol supports, CreateGlobalStringPtr is implemented as
// CreateGlobalString plus an all-zero inbounds GEP, which constant-folds back
// to the global itself under opaque pointers — so the emitted IR is identical.
//
// The shim returns llvm::Constant* rather than the GlobalVariable* that
// CreateGlobalString returns, i.e. exactly the static type the old call had.
// That keeps overload resolution at every call site (notably brace-initialised
// ArrayRef<Value*> argument lists) bit-for-bit unchanged.
//
// Templated on the builder type so it works whether CreateGlobalString is
// declared on llvm::IRBuilderBase or on llvm::IRBuilder<> in a given release.
// ---------------------------------------------------------------------------
template <typename BuilderT>
inline llvm::Constant* createGlobalString(BuilderT& builder,
                                          llvm::StringRef str,
                                          const llvm::Twine& name = "") {
    return builder.CreateGlobalString(str, name);
}

// ---------------------------------------------------------------------------
// Unconditional branch instruction type.
//
// LLVM <= 22:  IRBuilder::CreateBr returns llvm::BranchInst*.
// LLVM 24:     llvm::BranchInst was split into llvm::UncondBrInst and
//              llvm::CondBrInst, and CreateBr returns llvm::UncondBrInst*.
//
// Deriving the alias from CreateBr's own return type keeps this correct on
// every release without a version guard, and keeps it TIGHT: it is the
// unconditional-branch type only, so passing the result of CreateCondBr where
// one of these is expected stays a compile error rather than being widened
// away to llvm::Instruction*.
// ---------------------------------------------------------------------------
using UncondBranchInst = std::remove_pointer_t<decltype(
    std::declval<llvm::IRBuilder<>&>().CreateBr(
        std::declval<llvm::BasicBlock*>()))>;

// ---------------------------------------------------------------------------
// Intrinsic signature verification.
//
// Eshkol's `llvm-intrinsic` low-level builtin lets user code name an intrinsic
// and a function type; before emitting the call it must prove the requested
// type really is a legal instantiation of that intrinsic, otherwise it would
// emit a mismatched intrinsic call (miscompile).  It must also recover the
// list of overload types needed to declare the intrinsic.
//
// The spelling of that check has changed twice, so there are three paths.
//
//   LLVM 18 to 20:  Intrinsic::matchIntrinsicSignature(), driven by an IIT
//                   descriptor table the caller fetches itself, compared
//                   against MatchIntrinsicTypes_Match.
//   LLVM 21 and 22: bool Intrinsic::getIntrinsicSignature(ID, FunctionType*,
//                   SmallVectorImpl<Type*>&). Verified present in both.
//   LLVM 23 onward: the same function renamed to
//                   bool Intrinsic::isSignatureValid(ID, FunctionType*,
//                   SmallVectorImpl<Type*>&, raw_ostream& = nulls()).
//                   Verified present in 24; both matchIntrinsicSignature and
//                   getIntrinsicSignature are gone there.
//
// LLVM 23 itself is not a target Eshkol builds against, so which side of the
// rename it falls on is untested. Getting that boundary wrong is a compile
// error naming this function, never a silent behaviour change.
//
// Each newer form performs the same descriptor match internally and also
// checks the vararg tail, which the oldest path skipped. So the check has only
// ever become stricter across these versions, never weaker.
//
// Returns true iff `fty` is a valid signature for `id`; on success
// `overload_types` holds the overload types to pass to ESHKOL_GET_INTRINSIC.
// Both paths fail closed: an unmatched signature returns false and the caller
// must refuse to emit the call.
// ---------------------------------------------------------------------------
inline bool intrinsicSignatureMatches(llvm::Intrinsic::ID id,
                                      llvm::FunctionType* fty,
                                      llvm::SmallVectorImpl<llvm::Type*>& overload_types) {
#if LLVM_VERSION_MAJOR >= 23
    // isSignatureValid explains a mismatch on the stream it is given, and the
    // caller then refuses to emit the call. A refusal with no reason printed
    // presents as a function with an empty body and no return, which is what
    // a verifier failure looks like far downstream. Let the reason reach
    // stderr so a mismatch is diagnosable at the point it happens.
    return llvm::Intrinsic::isSignatureValid(id, fty, overload_types, llvm::errs());
#elif LLVM_VERSION_MAJOR >= 21
    return llvm::Intrinsic::getIntrinsicSignature(id, fty, overload_types);
#else
    llvm::SmallVector<llvm::Intrinsic::IITDescriptor, 8> infos;
    llvm::Intrinsic::getIntrinsicInfoTableEntries(id, infos);
    llvm::ArrayRef<llvm::Intrinsic::IITDescriptor> info_ref(infos);
    return llvm::Intrinsic::matchIntrinsicSignature(fty, info_ref, overload_types) ==
           llvm::Intrinsic::MatchIntrinsicTypes_Match;
#endif
}

}  // namespace llvm_compat
}  // namespace eshkol

#endif  // ESHKOL_BACKEND_LLVM_COMPAT_H
