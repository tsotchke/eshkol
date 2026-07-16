//
// Copyright (C) tsotchke
//
// SPDX-License-Identifier: MIT
//

#include "jit_coff_memory_manager.h"

#include <llvm/Support/MathExtras.h>
#include <llvm/Support/Memory.h>
#include <llvm/Support/Process.h>

#include <algorithm>
#include <array>
#include <limits>
#include <system_error>

namespace eshkol {
namespace {

constexpr std::size_t kMinimumPoolSlack = 1024 * 1024;

std::error_code allocation_error() {
    return std::make_error_code(std::errc::not_enough_memory);
}

std::size_t purpose_index(
    llvm::SectionMemoryManager::AllocationPurpose purpose
) {
    switch (purpose) {
    case llvm::SectionMemoryManager::AllocationPurpose::Code:
        return 0;
    case llvm::SectionMemoryManager::AllocationPurpose::ROData:
        return 1;
    case llvm::SectionMemoryManager::AllocationPurpose::RWData:
        return 2;
    }
    return 0;
}

}  // namespace

class CoLocatedMemoryMapperOwner::Mapper final
    : public llvm::SectionMemoryManager::MemoryMapper {
public:
    ~Mapper() override {
        if (arena_.base()) {
            llvm::sys::Memory::releaseMappedMemory(arena_);
        }
    }

    void reserve(
        std::uintptr_t code_size,
        llvm::Align code_align,
        std::uintptr_t ro_data_size,
        llvm::Align ro_data_align,
        std::uintptr_t rw_data_size,
        llvm::Align rw_data_align
    ) {
        if (reservation_attempted_) {
            reservation_error_ =
                std::make_error_code(std::errc::operation_not_permitted);
            return;
        }
        reservation_attempted_ = true;

        const std::size_t page_size =
            llvm::sys::Process::getPageSizeEstimate();
        const std::array<std::uintptr_t, 3> requested = {
            code_size, ro_data_size, rw_data_size};
        const std::array<std::size_t, 3> alignments = {
            code_align.value(), ro_data_align.value(), rw_data_align.value()};

        std::size_t total = 0;
        for (std::size_t index = 0; index < requested.size(); ++index) {
            if (requested[index] == 0) {
                continue;
            }
            if (requested[index] > std::numeric_limits<std::size_t>::max()) {
                reservation_error_ = allocation_error();
                return;
            }

            const std::size_t size = static_cast<std::size_t>(requested[index]);
            const std::size_t slack =
                std::max(kMinimumPoolSlack, size / 16);
            const std::size_t alignment =
                std::max(page_size, alignments[index]);
            if (size > std::numeric_limits<std::size_t>::max() - slack ||
                size + slack >
                    std::numeric_limits<std::size_t>::max() - alignment) {
                reservation_error_ = allocation_error();
                return;
            }

            pools_[index].size =
                llvm::alignTo(size + slack, alignment);
            pools_[index].size =
                llvm::alignTo(pools_[index].size, page_size);
            if (total > std::numeric_limits<std::size_t>::max() -
                            pools_[index].size) {
                reservation_error_ = allocation_error();
                return;
            }
            total += pools_[index].size;
        }

        if (pools_[0].size >
                CoLocatedSectionMemoryManager::kMaximumCodeSpan ||
            total == 0 ||
            total > CoLocatedSectionMemoryManager::kMaximumArenaSpan) {
            reservation_error_ = allocation_error();
            return;
        }

        std::error_code error;
        arena_ = llvm::sys::Memory::allocateMappedMemory(
            total, nullptr,
            llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_WRITE,
            error);
        if (error || !arena_.base()) {
            reservation_error_ = error ? error : allocation_error();
            return;
        }

        auto* cursor = static_cast<std::uint8_t*>(arena_.base());
        for (Pool& pool : pools_) {
            if (pool.size != 0) {
                pool.base = cursor;
                cursor += pool.size;
            }
        }
    }

    llvm::sys::MemoryBlock allocateMappedMemory(
        llvm::SectionMemoryManager::AllocationPurpose purpose,
        std::size_t num_bytes,
        const llvm::sys::MemoryBlock* const,
        unsigned,
        std::error_code& error
    ) override {
        if (!reservation_attempted_ || reservation_error_) {
            error = reservation_error_ ? reservation_error_ : allocation_error();
            return {};
        }

        Pool& pool = pools_[purpose_index(purpose)];
        if (pool.handled_out || !pool.base || num_bytes > pool.size) {
            error = allocation_error();
            return {};
        }

        pool.handled_out = true;
        error.clear();
        return llvm::sys::MemoryBlock(pool.base, pool.size);
    }

    std::error_code protectMappedMemory(
        const llvm::sys::MemoryBlock& block,
        unsigned flags
    ) override {
        return llvm::sys::Memory::protectMappedMemory(block, flags);
    }

    std::error_code releaseMappedMemory(
        llvm::sys::MemoryBlock& block
    ) override {
        // SectionMemoryManager sees each pool as a separate allocation. They
        // are sub-blocks of arena_, so releasing any one of them through the
        // operating system would invalidate the others. The owner releases the
        // whole arena after SectionMemoryManager has finished its teardown.
        block = llvm::sys::MemoryBlock();
        return {};
    }

private:
    struct Pool {
        std::uint8_t* base = nullptr;
        std::size_t size = 0;
        bool handled_out = false;
    };

    std::array<Pool, 3> pools_;
    llvm::sys::MemoryBlock arena_;
    std::error_code reservation_error_;
    bool reservation_attempted_ = false;
};

CoLocatedMemoryMapperOwner::CoLocatedMemoryMapperOwner()
    : mapper_(std::make_unique<Mapper>()) {}

CoLocatedMemoryMapperOwner::~CoLocatedMemoryMapperOwner() = default;

llvm::SectionMemoryManager::MemoryMapper*
CoLocatedMemoryMapperOwner::mapper() {
    return mapper_.get();
}

void CoLocatedMemoryMapperOwner::reserveMapper(
    std::uintptr_t code_size,
    llvm::Align code_align,
    std::uintptr_t ro_data_size,
    llvm::Align ro_data_align,
    std::uintptr_t rw_data_size,
    llvm::Align rw_data_align
) {
    mapper_->reserve(
        code_size, code_align, ro_data_size, ro_data_align,
        rw_data_size, rw_data_align);
}

CoLocatedSectionMemoryManager::CoLocatedSectionMemoryManager()
    : CoLocatedMemoryMapperOwner(),
      llvm::SectionMemoryManager(mapper()) {}

CoLocatedSectionMemoryManager::~CoLocatedSectionMemoryManager() = default;

bool CoLocatedSectionMemoryManager::needsToReserveAllocationSpace() {
    return true;
}

void CoLocatedSectionMemoryManager::reserveAllocationSpace(
    std::uintptr_t code_size,
    llvm::Align code_align,
    std::uintptr_t ro_data_size,
    llvm::Align ro_data_align,
    std::uintptr_t rw_data_size,
    llvm::Align rw_data_align
) {
    reserveMapper(
        code_size, code_align, ro_data_size, ro_data_align,
        rw_data_size, rw_data_align);
}

}  // namespace eshkol
