#include <eshkol/core/seed_entropy.h>
#include <eshkol/core/table_hash.h>

namespace {
struct HashKey {
    uint64_t words[2];
    HashKey() { eshkol_seed_entropy(words, sizeof(words)); }
};

uint64_t rotate(uint64_t x, unsigned bits) {
    return (x << bits) | (x >> (64 - bits));
}

struct SipHash {
    uint64_t a, b, c, d;
    void round() {
        a += b;
        b = rotate(b, 13) ^ a;
        a = rotate(a, 32);
        c += d;
        d = rotate(d, 16) ^ c;
        a += d;
        d = rotate(d, 21) ^ a;
        c += b;
        b = rotate(b, 17) ^ c;
        c = rotate(c, 32);
    }
    void absorb(uint64_t word) {
        d ^= word;
        round();
        round();
        a ^= word;
    }
};
}

uint64_t eshkol_table_hash(const void* data, size_t length) {
    static const HashKey key;
    SipHash state{
        0x736f6d6570736575ULL ^ key.words[0], 0x646f72616e646f6dULL ^ key.words[1],
        0x6c7967656e657261ULL ^ key.words[0], 0x7465646279746573ULL ^ key.words[1]
    };
    const auto* bytes = static_cast<const unsigned char*>(data);
    uint64_t tail = uint64_t(length) << 56;
    while (length >= 8) {
        uint64_t word = 0;
        for (unsigned i = 0; i < 8; ++i) word |= uint64_t(bytes[i]) << (8 * i);
        state.absorb(word);
        bytes += 8;
        length -= 8;
    }
    for (unsigned i = 0; i < length; ++i) tail |= uint64_t(bytes[i]) << (8 * i);
    state.absorb(tail);
    state.c ^= 0xff;
    for (int i = 0; i < 4; ++i) state.round();
    return state.a ^ state.b ^ state.c ^ state.d;
}
