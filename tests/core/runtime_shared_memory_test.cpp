#include "../../lib/core/arena_memory.h"

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

int destructor_calls = 0;

void counting_destructor(void* ptr) {
    if (!ptr) return;
    auto* bytes = static_cast<unsigned char*>(ptr);
    if (bytes[0] == 0xAB && bytes[1] == 0xCD) {
        destructor_calls++;
    }
}

int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

}  // namespace

int main() {
    void* raw = shared_allocate_typed(8, 77, counting_destructor);
    if (!raw) return fail("shared allocation returned null");

    auto* payload = static_cast<unsigned char*>(raw);
    payload[0] = 0xAB;
    payload[1] = 0xCD;

    eshkol_shared_header_t* header = shared_get_header(raw);
    if (!header) return fail("shared header missing");
    if (header->ref_count != 1) return fail("initial ref count is not 1");
    if (header->weak_count != 0) return fail("initial weak count is not 0");
    if (header->value_type != 77) return fail("value type was not preserved");
    if (shared_ref_count(raw) != 1) return fail("shared_ref_count mismatch");

    shared_retain(raw);
    if (shared_ref_count(raw) != 2) return fail("retain did not increment");

    eshkol_weak_ref_t* weak = weak_ref_create(raw);
    if (!weak) return fail("weak ref create returned null");
    if (!weak_ref_is_alive(weak)) return fail("new weak ref is not alive");
    if (header->weak_count != 1) return fail("weak count did not increment");

    void* upgraded = weak_ref_upgrade(weak);
    if (upgraded != raw) return fail("weak upgrade returned wrong pointer");
    if (shared_ref_count(raw) != 3) return fail("weak upgrade did not retain");

    shared_release(upgraded);
    if (shared_ref_count(raw) != 2) return fail("release upgraded ref failed");
    shared_release(raw);
    if (shared_ref_count(raw) != 1) return fail("release retained ref failed");

    shared_release(raw);
    if (destructor_calls != 1) return fail("destructor was not called once");
    if (weak_ref_is_alive(weak)) return fail("weak ref stayed alive after final release");
    if (weak_ref_upgrade(weak) != nullptr) return fail("dead weak ref upgraded");

    weak_ref_release(weak);

    std::cout << "PASS\n";
    return 0;
}
