//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//

#ifndef ESHKOL_JIT_TARGET_CONFIG_H
#define ESHKOL_JIT_TARGET_CONFIG_H

#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>
#include <llvm/TargetParser/Triple.h>

#include <cstddef>

namespace eshkol {

/**
 * @brief Returns whether @p triple is the Windows ARM64 COFF JIT target.
 */
inline bool is_windows_aarch64_coff(const llvm::Triple& triple) {
    return triple.getArch() == llvm::Triple::aarch64 &&
           triple.isOSWindows() &&
           triple.getObjectFormat() == llvm::Triple::COFF;
}

/**
 * @brief Applies target-specific IR contracts required before ORC codegen.
 *
 * LLVM 21's AArch64 COFF Small model emits direct ADRP/PAGEOFF references for
 * ordinary external data declarations. ORC's Windows ARM64 object layer is
 * RuntimeDyldCOFFAArch64, whose direct PAGEBASE relocation has only +/-4 GiB
 * reach; JIT allocations and data exported by the host executable routinely
 * occupy unrelated parts of the 64-bit address space. RuntimeDyld already has
 * the correct unlimited-reach mechanism for COFF imports: a per-section import
 * address cell populated with an absolute 64-bit relocation to the host datum.
 * Marking declarations dllimport makes LLVM emit the required __imp_ load and
 * lets RuntimeDyld build that nearby cell.
 *
 * Only declarations are changed. Definitions remain ordinary JIT-owned data,
 * and non-Windows targets retain their native visibility/relocation contract.
 * The function is idempotent and returns the number of declarations changed.
 */
inline std::size_t prepare_jit_module_for_target(
    llvm::Module& module,
    const llvm::Triple& triple
) {
    if (!is_windows_aarch64_coff(triple)) {
        return 0;
    }

    std::size_t changed = 0;
    for (llvm::GlobalVariable& global : module.globals()) {
        if (!global.isDeclaration() ||
            global.getDLLStorageClass() != llvm::GlobalValue::DefaultStorageClass) {
            continue;
        }

        global.setDLLStorageClass(llvm::GlobalValue::DLLImportStorageClass);
        ++changed;
    }
    return changed;
}

}  // namespace eshkol

#endif  // ESHKOL_JIT_TARGET_CONFIG_H
