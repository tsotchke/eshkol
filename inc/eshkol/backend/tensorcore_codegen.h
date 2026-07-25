/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_BACKEND_TENSORCORE_CODEGEN_H
#define ESHKOL_BACKEND_TENSORCORE_CODEGEN_H

#include <cstddef>
#include <string>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

namespace llvm {
class IntegerType;
class Module;
}

namespace eshkol {

class CodegenContext;

/* Declare the canonical Eshkol adapter ABI in an LLVM module. The operation is
 * idempotent and rejects a pre-existing symbol with a conflicting signature.
 * Returns the number of declarations added, or 0 on failure with *error set.
 * Returns 0 with *error CLEARED when the ABI does not apply to the module's
 * target at all — it is 64-bit by definition, so a 32-bit target (wasm32) has
 * nothing to declare and that is not a failure. */
std::size_t declareTensorcoreAdapterAbi(llvm::Module& module,
                                        llvm::IntegerType* size_type,
                                        std::string* error = nullptr);

/* Declare the ABI and register every declaration in CodegenContext's function
 * table so extern lowering, JIT, AOT, REPL, and SDK compilation share it. */
std::size_t registerTensorcoreBuiltins(CodegenContext& ctx,
                                       std::string* error = nullptr);

/* A direct verification hook used by the release gate and downstream embedders. */
bool verifyTensorcoreAdapterModule(const llvm::Module& module,
                                   std::string* error = nullptr);

}  // namespace eshkol

/* Register the adapter declarations for a codegen context. Returns the number
 * declared, 0 when the ABI does not apply to the target (not an error, and no
 * diagnostic is emitted), or a negative value when registration genuinely
 * failed, having reported why. */
extern "C" int eshkol_register_tensorcore_builtins(eshkol::CodegenContext* ctx);

#endif  /* ESHKOL_LLVM_BACKEND_ENABLED */
#endif  /* ESHKOL_BACKEND_TENSORCORE_CODEGEN_H */
