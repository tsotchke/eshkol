//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//

#ifndef ESHKOL_JIT_COFF_MEMORY_MANAGER_H
#define ESHKOL_JIT_COFF_MEMORY_MANAGER_H

#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/Support/Alignment.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace eshkol {

/**
 * @brief Owns the mapper used by CoLocatedSectionMemoryManager.
 *
 * This base is deliberately declared before SectionMemoryManager in the
 * derived class. It is therefore constructed first and destroyed last, which
 * keeps the mapper alive while SectionMemoryManager releases its sub-blocks.
 */
class CoLocatedMemoryMapperOwner {
protected:
    CoLocatedMemoryMapperOwner();
    ~CoLocatedMemoryMapperOwner();

    llvm::SectionMemoryManager::MemoryMapper* mapper();
    void reserveMapper(
        std::uintptr_t code_size,
        llvm::Align code_align,
        std::uintptr_t ro_data_size,
        llvm::Align ro_data_align,
        std::uintptr_t rw_data_size,
        llvm::Align rw_data_align
    );

private:
    class Mapper;
    std::unique_ptr<Mapper> mapper_;
};

/**
 * @brief RuntimeDyld memory manager for the Windows ARM64 Small code model.
 *
 * RuntimeDyld normally allocates code, read-only data, and writable data from
 * three independent virtual-memory pools. Windows may place those pools more
 * than 4 GiB apart, outside AArch64 ADRP/PAGEOFF reach. This manager reserves
 * one arena per object and partitions all three pools inside it.
 */
class CoLocatedSectionMemoryManager final
    : private CoLocatedMemoryMapperOwner,
      public llvm::SectionMemoryManager {
public:
    static constexpr std::uintptr_t kMaximumCodeSpan =
        std::uintptr_t{120} * 1024 * 1024;
    static constexpr std::uintptr_t kMaximumArenaSpan =
        std::uintptr_t{2} * 1024 * 1024 * 1024;

    CoLocatedSectionMemoryManager();
    ~CoLocatedSectionMemoryManager() override;

    bool needsToReserveAllocationSpace() override;

    void reserveAllocationSpace(
        std::uintptr_t code_size,
        llvm::Align code_align,
        std::uintptr_t ro_data_size,
        llvm::Align ro_data_align,
        std::uintptr_t rw_data_size,
        llvm::Align rw_data_align
    ) override;
};

}  // namespace eshkol

#endif  // ESHKOL_JIT_COFF_MEMORY_MANAGER_H
