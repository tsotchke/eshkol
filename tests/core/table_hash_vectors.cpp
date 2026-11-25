#include <eshkol/core/table_hash.h>
#include <cstring>

extern "C" int eshkol_random_bytes(char* out, size_t size) {
    const uint64_t key[] = {0x0706050403020100ULL, 0x0f0e0d0c0b0a0908ULL};
    if (size != sizeof(key)) return -1;
    std::memcpy(out, key, size);
    return 0;
}

int main() {
    unsigned char input[15];
    for (unsigned i = 0; i < sizeof(input); ++i) input[i] = static_cast<unsigned char>(i);
    return eshkol_table_hash(input, 0) == 0x726fdb47dd0e0e31ULL &&
           eshkol_table_hash(input, 8) == 0x93f5f5799a932462ULL &&
           eshkol_table_hash(input, 15) == 0xa129ca6149be45e5ULL ? 0 : 1;
}
