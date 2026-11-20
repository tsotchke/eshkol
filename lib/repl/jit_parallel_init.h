//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//

#ifndef ESHKOL_JIT_PARALLEL_INIT_H
#define ESHKOL_JIT_PARALLEL_INIT_H

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <string>

namespace eshkol {

inline constexpr const char* kJitStdlibParallelInitializer =
    "__repl_stdlib_init_parallel_workers";

/**
 * Give the JIT-owned worker initializer an explicitly resolvable name.
 *
 * Windows AOT modules deliberately keep this initializer internal to avoid
 * duplicate COFF symbols when a program links stdlib.o. LLJIT::lookup cannot
 * resolve that local symbol. Export it only in the JIT's IR copy, with a name
 * unique to this module, so lookup cannot select another module's initializer.
 * Renaming the Function also updates its existing llvm.global_ctors references.
 * The worker implementations retain their original linkage.
 */
inline std::string expose_jit_parallel_initializer(
    llvm::Module& module, llvm::StringRef unique_name
) {
    auto* initializer = module.getFunction("__eshkol_init_parallel_workers");
    if (!initializer || initializer->isDeclaration()) {
        return {};
    }
    initializer->setName(unique_name);
    initializer->setLinkage(llvm::GlobalValue::ExternalLinkage);
    initializer->setVisibility(llvm::GlobalValue::DefaultVisibility);
    initializer->setDLLStorageClass(llvm::GlobalValue::DefaultStorageClass);
    initializer->setComdat(nullptr);
    return initializer->getName().str();
}

}  // namespace eshkol

#endif  // ESHKOL_JIT_PARALLEL_INIT_H
