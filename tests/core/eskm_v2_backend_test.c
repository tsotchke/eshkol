#include "../../lib/core/eskm_v2_experimental.h"
#include <assert.h>

static void put(unsigned char* p, uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i) p[i] = (unsigned char)(value >> (i * 8));
}
static void checksum(unsigned char* bytes, size_t size) {
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < size - 4; ++i) {
        crc ^= bytes[i];
        for (unsigned bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1)));
    }
    put(bytes + size - 4, ~crc, 4);
}
int main(void) {
    /* One named rank-1 tensor containing raw negative zero. Independent wire. */
    unsigned char bytes[54] = {'E', 'S', 'K', 'M', 2};
    put(bytes + 8, 1, 4);
    put(bytes + 24, 1, 4); bytes[28] = 'x';
    put(bytes + 29, 1, 4); put(bytes + 33, 1, 8);
    put(bytes + 42, UINT64_C(0x8000000000000000), 8);
    checksum(bytes, sizeof(bytes));
    eskm_v2_backend_limits limits = ESKM_V2_BACKEND_DEFAULT;
    eskm_v2_result result;
    eskm_v2_budget budget;
#define ADMIT() eskm_v2_backend_admit(bytes, sizeof(bytes), &limits, &result, &budget)
    assert(ADMIT());
    uint64_t charged = budget.charged_bytes;
    limits.records = limits.name_bytes = limits.rank = limits.elements = 1;
    limits.memory_bytes = charged;
    assert(ADMIT());
#define LOWER(field) do { --limits.field; assert(!ADMIT()); ++limits.field; } while (0)
    LOWER(records); LOWER(name_bytes); LOWER(rank); LOWER(elements); LOWER(memory_bytes);
    bytes[12] = 1; checksum(bytes, sizeof(bytes)); assert(!ADMIT());
    bytes[12] = 0; bytes[28] = 0; checksum(bytes, sizeof(bytes)); assert(!ADMIT());
    bytes[28] = 'x'; checksum(bytes, sizeof(bytes)); bytes[50] ^= 1; assert(!ADMIT());
    checksum(bytes, sizeof(bytes)); assert(ADMIT());
    unsetenv("ESHKOL_EXPERIMENTAL_ESKM_V2"); assert(eskm_v2_mode() == 0);
    setenv("ESHKOL_EXPERIMENTAL_ESKM_V2", "read", 1); assert(eskm_v2_mode() == 1);
    FILE* file = tmpfile();
    assert(file && fwrite(bytes, 1, 8, file) == 8 && fseek(file, 0, SEEK_SET) == 0);
    assert(eskm_v2_file_admit(file, ESKM_V2_MAX_BUFFER_BYTES));
    assert(!eskm_v2_file_admit(file, ESKM_V2_MAX_BUFFER_BYTES + 1));
    unsetenv("ESHKOL_EXPERIMENTAL_ESKM_V2");
    assert(!eskm_v2_file_admit(file, sizeof(bytes)));
    fclose(file);
    setenv("ESHKOL_EXPERIMENTAL_ESKM_V2", "write", 1); assert(eskm_v2_mode() == 2);
    setenv("ESHKOL_EXPERIMENTAL_ESKM_V2", "typo", 1); assert(eskm_v2_mode() == -1);
    puts("PASS: v2 backend limits, malformed controls and explicit policy");
    return 0;
}
