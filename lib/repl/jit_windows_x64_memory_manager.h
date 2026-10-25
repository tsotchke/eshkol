// Copyright (C) tsotchke
// SPDX-License-Identifier: MIT

#ifndef ESHKOL_JIT_WINDOWS_X64_MEMORY_MANAGER_H
#define ESHKOL_JIT_WINDOWS_X64_MEMORY_MANAGER_H

#include <llvm/ExecutionEngine/SectionMemoryManager.h>

#include <cstdint>
#include <limits>
#include <vector>

#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
namespace eshkol {

/**
 * Register RuntimeDyld's x64 COFF .pdata with the native Windows unwinder.
 * LLVM's default memory manager uses __register_frame, which does not register
 * Windows RUNTIME_FUNCTION records. Keep the tables alive with their object
 * and remove them before SectionMemoryManager releases the underlying pages.
 */
class WindowsX64SectionMemoryManager final : public llvm::SectionMemoryManager {
public:
    ~WindowsX64SectionMemoryManager() override;

    uint8_t* allocateCodeSection(uintptr_t size, unsigned alignment,
                                unsigned section_id, llvm::StringRef name) override;
    uint8_t* allocateDataSection(uintptr_t size, unsigned alignment,
                                unsigned section_id, llvm::StringRef name,
                                bool read_only) override;
    void registerEHFrames(uint8_t* address, uint64_t load_address, size_t size) override;
    void deregisterEHFrames() override;

private:
    void recordSection(uint8_t* address, uintptr_t size);
    // RuntimeDyldCOFFX86_64 chooses the lowest loaded section as __ImageBase.
    uint64_t image_base_ = std::numeric_limits<uint64_t>::max();
    std::vector<uint8_t*> function_tables_;
};

}  // namespace eshkol
#endif
#endif  // ESHKOL_JIT_WINDOWS_X64_MEMORY_MANAGER_H
