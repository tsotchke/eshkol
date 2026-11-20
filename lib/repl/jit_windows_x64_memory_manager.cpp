// Copyright (C) tsotchke
// SPDX-License-Identifier: MIT

#include "jit_windows_x64_memory_manager.h"

#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <llvm/Support/ErrorHandling.h>

#include <algorithm>
#include <limits>

namespace eshkol {

WindowsX64SectionMemoryManager::~WindowsX64SectionMemoryManager() {
    deregisterEHFrames();
}

void WindowsX64SectionMemoryManager::recordSection(uint8_t* address, uintptr_t size) {
    if (address && size != 0) {
        image_base_ = std::min(image_base_, reinterpret_cast<uint64_t>(address));
    }
}

uint8_t* WindowsX64SectionMemoryManager::allocateCodeSection(
    uintptr_t size, unsigned alignment, unsigned section_id, llvm::StringRef name
) {
    auto* address = llvm::SectionMemoryManager::allocateCodeSection(
        size, alignment, section_id, name);
    recordSection(address, size);
    return address;
}

uint8_t* WindowsX64SectionMemoryManager::allocateDataSection(
    uintptr_t size, unsigned alignment, unsigned section_id, llvm::StringRef name,
    bool read_only
) {
    auto* address = llvm::SectionMemoryManager::allocateDataSection(
        size, alignment, section_id, name, read_only);
    recordSection(address, size);
    return address;
}

void WindowsX64SectionMemoryManager::registerEHFrames(
    uint8_t* address, uint64_t load_address, size_t size
) {
    if (size == 0) return;
    if (!address || load_address != reinterpret_cast<uint64_t>(address) ||
        image_base_ == std::numeric_limits<uint64_t>::max() ||
        size % sizeof(RUNTIME_FUNCTION) != 0 ||
        size / sizeof(RUNTIME_FUNCTION) > std::numeric_limits<DWORD>::max()) {
        llvm::report_fatal_error("invalid local Windows x64 JIT unwind table");
    }
    if (std::find(function_tables_.begin(), function_tables_.end(), address) !=
        function_tables_.end()) {
        return;
    }
    // Retain ownership before registering so an allocation failure cannot
    // leave a Windows table pointing at pages the memory manager will free.
    function_tables_.push_back(address);
    if (!RtlAddFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(address),
                             static_cast<DWORD>(size / sizeof(RUNTIME_FUNCTION)),
                             image_base_)) {
        function_tables_.pop_back();
        llvm::report_fatal_error("could not register Windows x64 JIT unwind table");
    }
}

void WindowsX64SectionMemoryManager::deregisterEHFrames() {
    for (auto* address : function_tables_) {
        if (!RtlDeleteFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(address))) {
            llvm::report_fatal_error("could not remove Windows x64 JIT unwind table");
        }
    }
    function_tables_.clear();
}

}  // namespace eshkol
#endif
