#include <eshkol/core/gensym_sequence.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

void eshkol_gensym_suffix(char buffer[21]) {
    static std::atomic<uint64_t> sequence{1};
    uint64_t value = sequence.load(std::memory_order_relaxed);
    do {
        if (value == std::numeric_limits<uint64_t>::max()) std::abort();
    } while (!sequence.compare_exchange_weak(value, value + 1, std::memory_order_relaxed));
    std::snprintf(buffer, 21, "%llu", static_cast<unsigned long long>(value));
}
