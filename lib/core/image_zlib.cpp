/**
 * @file image_zlib.cpp
 * @brief zlib streams for the built-in image codecs: CRC-32, Adler-32,
 *        inflate (RFC 1950/1951) and a deflate encoder.
 *
 * The encoder reproduces zlib's default compression level (lazy matching
 * with hash chains, good/lazy/nice/chain = 8/16/128/128), its block-flush
 * rule, its Huffman tree construction (heap order, depth tie-break, length
 * limiting) and its stored/static/dynamic block choice.  Given the same input
 * chunking it therefore emits the same bytes as zlib, which keeps PNG files
 * written by Eshkol identical to the ones the system libpng wrote before the
 * codecs moved into the runtime.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "image_codecs.h"

#include <array>
#include <cstring>

namespace eshkol::image {

/* ======================================================================== */
/* Checksums                                                                */
/* ======================================================================== */

namespace {

struct CrcTable {
    uint32_t entries[256];
    CrcTable() {
        for (uint32_t n = 0; n < 256; ++n) {
            uint32_t c = n;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            entries[n] = c;
        }
    }
};

const CrcTable& crc_table() {
    static const CrcTable table;
    return table;
}

} // namespace

uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t size) {
    const CrcTable& t = crc_table();
    uint32_t c = crc ^ 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        c = t.entries[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

uint32_t adler32_update(uint32_t adler, const uint8_t* data, size_t size) {
    constexpr uint32_t kBase = 65521u;
    constexpr size_t kNmax = 5552;  // largest n with 255n(n+1)/2 + (n+1)(BASE-1) < 2^32
    uint32_t a = adler & 0xFFFFu;
    uint32_t b = (adler >> 16) & 0xFFFFu;
    while (size > 0) {
        size_t n = size < kNmax ? size : kNmax;
        size -= n;
        while (n--) {
            a += *data++;
            b += a;
        }
        a %= kBase;
        b %= kBase;
    }
    return (b << 16) | a;
}

/* ======================================================================== */
/* Inflate                                                                  */
/* ======================================================================== */

namespace {

constexpr int kMaxCodeBits = 15;
constexpr int kFastBits = 10;

const uint16_t kLengthBase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
                                  35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
const uint8_t kLengthExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                  3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
const uint16_t kDistBase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
                                257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
                                8193, 12289, 16385, 24577};
const uint8_t kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                                7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
const uint8_t kCodeLengthOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5,
                                      11, 4, 12, 3, 13, 2, 14, 1, 15};

class BitReader {
public:
    BitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    /** Ensure at least @p n (<= 32) bits are buffered; past the end, zero bits
     *  are supplied and overrun() reports it. */
    void need(int n) {
        while (count_ < n) {
            uint64_t byte = 0;
            if (pos_ < size_) {
                byte = data_[pos_];
            } else {
                ++overrun_bytes_;
            }
            ++pos_;
            bits_ |= byte << count_;
            count_ += 8;
        }
    }
    uint32_t peek(int n) {
        need(n);
        return static_cast<uint32_t>(bits_ & ((uint64_t{1} << n) - 1));
    }
    void consume(int n) {
        bits_ >>= n;
        count_ -= n;
    }
    uint32_t take(int n) {
        if (n == 0) return 0;
        const uint32_t v = peek(n);
        consume(n);
        return v;
    }
    void align_to_byte() { consume(count_ & 7); }
    /** True when bits beyond the end of the input were consumed. */
    bool overrun() const {
        // Buffered-but-unconsumed padding bits are fine; only consumed ones count.
        const size_t consumed_bits = pos_ * 8 - static_cast<size_t>(count_);
        return consumed_bits > size_ * 8;
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
    size_t overrun_bytes_ = 0;
    uint64_t bits_ = 0;
    int count_ = 0;
};

/** Canonical Huffman decoder with a direct lookup table for short codes. */
class HuffmanDecoder {
public:
    /** Build from code lengths.  Over-subscribed sets are rejected.  With
     *  @p strict, incomplete sets are rejected too, except an empty set or a
     *  single one-bit code (the rule zlib applies to stream-supplied trees). */
    bool build(const uint8_t* lengths, int count, bool strict, bool code_lengths = false) {
        std::memset(counts_, 0, sizeof(counts_));
        for (int i = 0; i < count; ++i) counts_[lengths[i]]++;
        counts_[0] = 0;
        int max_len = 0;
        for (int len = 1; len <= kMaxCodeBits; ++len) {
            if (counts_[len] != 0) max_len = len;
        }
        int left = 1;
        for (int len = 1; len <= kMaxCodeBits; ++len) {
            left <<= 1;
            left -= counts_[len];
            if (left < 0) return false;  // over-subscribed
        }
        if (strict && max_len != 0 && left > 0 && (code_lengths || max_len != 1)) return false;
        int offsets[kMaxCodeBits + 2];
        offsets[1] = 0;
        for (int len = 1; len <= kMaxCodeBits; ++len) offsets[len + 1] = offsets[len] + counts_[len];
        symbols_.assign(static_cast<size_t>(count), 0);
        for (int i = 0; i < count; ++i) {
            if (lengths[i] != 0) symbols_[static_cast<size_t>(offsets[lengths[i]]++)] = static_cast<uint16_t>(i);
        }
        // Fast table: index = next kFastBits input bits (LSB first).
        fast_.fill(0);
        int code = 0;
        int index = 0;
        for (int len = 1; len <= kFastBits; ++len) {
            for (int k = 0; k < counts_[len]; ++k) {
                // Reverse the canonical code to LSB-first order.
                int reversed = 0;
                for (int b = 0; b < len; ++b) reversed |= ((code >> b) & 1) << (len - 1 - b);
                const uint16_t entry = static_cast<uint16_t>((symbols_[static_cast<size_t>(index)] << 4) | len);
                for (int fill = reversed; fill < (1 << kFastBits); fill += (1 << len)) {
                    fast_[static_cast<size_t>(fill)] = entry;
                }
                ++code;
                ++index;
            }
            code <<= 1;
        }
        return true;
    }

    /** Decode one symbol; returns -1 for an invalid (incomplete-code) input. */
    int decode(BitReader& in) const {
        const uint32_t look = in.peek(kFastBits);
        const uint16_t entry = fast_[look];
        if (entry != 0) {
            in.consume(entry & 15);
            return entry >> 4;
        }
        // Slow path: walk canonical code lengths bit by bit.
        int code = 0;
        int first = 0;
        int index = 0;
        for (int len = 1; len <= kMaxCodeBits; ++len) {
            code |= static_cast<int>(in.take(1));
            const int count = counts_[len];
            if (code - count < first) return symbols_[static_cast<size_t>(index + (code - first))];
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        return -1;
    }

private:
    int counts_[kMaxCodeBits + 1] = {};
    std::vector<uint16_t> symbols_;
    std::array<uint16_t, 1u << kFastBits> fast_{};
};

bool build_predefined_codes(HuffmanDecoder* lit, HuffmanDecoder* dist) {
    uint8_t lengths[288];
    int i = 0;
    for (; i < 144; ++i) lengths[i] = 8;
    for (; i < 256; ++i) lengths[i] = 9;
    for (; i < 280; ++i) lengths[i] = 7;
    for (; i < 288; ++i) lengths[i] = 8;
    uint8_t dlen[30];
    for (int d = 0; d < 30; ++d) dlen[d] = 5;
    return lit->build(lengths, 288, false) && dist->build(dlen, 30, false);
}

} // namespace

bool zlib_inflate_exact(const uint8_t* data, size_t size, size_t expected,
                        std::vector<uint8_t>* out) {
    if (!out || !data || size < 2) return false;
    const unsigned cmf = data[0];
    const unsigned flg = data[1];
    if ((cmf & 0x0Fu) != 8u || (cmf >> 4) > 7u) return false;
    if (((cmf << 8) | flg) % 31u != 0u) return false;
    if (flg & 0x20u) return false;  // preset dictionary: not used by PNG

    out->clear();
    out->reserve(expected);
    BitReader in(data + 2, size - 2);
    HuffmanDecoder lit;
    HuffmanDecoder dist;
    bool last = false;
    while (!last && out->size() < expected) {
        last = in.take(1) != 0;
        const uint32_t type = in.take(2);
        if (type == 0) {
            in.align_to_byte();
            const uint32_t len = in.take(16);
            const uint32_t nlen = in.take(16);
            if ((len ^ 0xFFFFu) != nlen) return false;
            for (uint32_t i = 0; i < len && out->size() < expected; ++i) {
                out->push_back(static_cast<uint8_t>(in.take(8)));
            }
            if (in.overrun()) return false;
            continue;
        }
        if (type == 1) {
            if (!build_predefined_codes(&lit, &dist)) return false;
        } else if (type == 2) {
            const int hlit = static_cast<int>(in.take(5)) + 257;
            const int hdist = static_cast<int>(in.take(5)) + 1;
            const int hclen = static_cast<int>(in.take(4)) + 4;
            if (hlit > 286 || hdist > 30) return false;
            uint8_t cl_lengths[19] = {0};
            for (int i = 0; i < hclen; ++i) cl_lengths[kCodeLengthOrder[i]] = static_cast<uint8_t>(in.take(3));
            HuffmanDecoder cl;
            if (!cl.build(cl_lengths, 19, true, true)) return false;
            uint8_t lengths[286 + 30] = {0};
            int n = 0;
            while (n < hlit + hdist) {
                const int sym = cl.decode(in);
                if (sym < 0) return false;
                if (sym < 16) {
                    lengths[n++] = static_cast<uint8_t>(sym);
                    continue;
                }
                int repeat = 0;
                uint8_t value = 0;
                if (sym == 16) {
                    if (n == 0) return false;
                    value = lengths[n - 1];
                    repeat = 3 + static_cast<int>(in.take(2));
                } else if (sym == 17) {
                    repeat = 3 + static_cast<int>(in.take(3));
                } else {
                    repeat = 11 + static_cast<int>(in.take(7));
                }
                if (n + repeat > hlit + hdist) return false;
                while (repeat--) lengths[n++] = value;
            }
            if (lengths[256] == 0) return false;  // no end-of-block code
            if (!lit.build(lengths, hlit, true) || !dist.build(lengths + hlit, hdist, true)) return false;
        } else {
            return false;
        }
        for (;;) {
            if (out->size() >= expected) break;
            const int sym = lit.decode(in);
            if (sym < 0) return false;
            if (sym < 256) {
                out->push_back(static_cast<uint8_t>(sym));
                continue;
            }
            if (sym == 256) break;
            const int li = sym - 257;
            if (li >= 29) return false;
            const size_t length = kLengthBase[li] + in.take(kLengthExtra[li]);
            const int dsym = dist.decode(in);
            if (dsym < 0 || dsym >= 30) return false;
            const size_t distance = kDistBase[dsym] + in.take(kDistExtra[dsym]);
            if (distance > out->size()) return false;
            size_t from = out->size() - distance;
            for (size_t k = 0; k < length && out->size() < expected; ++k) {
                out->push_back((*out)[from++]);
            }
        }
        if (in.overrun()) return false;
    }
    if (in.overrun()) return false;
    return out->size() >= expected;
}

/* ======================================================================== */
/* Deflate encoder                                                          */
/* ======================================================================== */

namespace {

constexpr int kMinMatch = 3;
constexpr int kMaxMatch = 258;
constexpr int kMinLookahead = kMaxMatch + kMinMatch + 1;
constexpr int kLengthCodes = 29;
constexpr int kLiterals = 256;
constexpr int kLCodes = kLiterals + 1 + kLengthCodes;
constexpr int kDCodes = 30;
constexpr int kBLCodes = 19;
constexpr int kHeapSize = 2 * kLCodes + 1;
constexpr int kMaxBits = 15;
constexpr int kMaxBLBits = 7;
constexpr int kEndBlock = 256;
constexpr int kRep3To6 = 16;
constexpr int kRepZ3To10 = 17;
constexpr int kRepZ11To138 = 18;
constexpr unsigned kTooFar = 4096;
constexpr unsigned kWinInit = kMaxMatch;
constexpr int kMemLevel = 8;

// zlib's default level: good_length, max_lazy, nice_length, max_chain.
constexpr unsigned kGoodLength = 8;
constexpr unsigned kMaxLazy = 16;
constexpr unsigned kNiceLength = 128;
constexpr unsigned kMaxChain = 128;

const int kExtraLBits[kLengthCodes] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                       3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
const int kExtraDBits[kDCodes] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                                  7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
const int kExtraBLBits[kBLCodes] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7};

/** Tree node: (freq | code) and (dad | len) share storage, as in zlib, because
 *  the tree builder overwrites one with the other in a defined order. */
struct TreeNode {
    uint16_t fc = 0;  // frequency while building, code afterwards
    uint16_t dl = 0;  // parent while building, bit length afterwards
};

struct StaticTables {
    TreeNode ltree[kLCodes + 2];
    TreeNode dtree[kDCodes];
    uint8_t dist_code[512];
    uint8_t length_code[kMaxMatch - kMinMatch + 1];
    int base_length[kLengthCodes];
    int base_dist[kDCodes];

    static unsigned reverse(unsigned code, int len) {
        unsigned res = 0;
        do {
            res |= code & 1u;
            code >>= 1;
            res <<= 1;
        } while (--len > 0);
        return res >> 1;
    }

    StaticTables() {
        int length = 0;
        int code = 0;
        for (code = 0; code < kLengthCodes - 1; ++code) {
            base_length[code] = length;
            for (int n = 0; n < (1 << kExtraLBits[code]); ++n) {
                length_code[length++] = static_cast<uint8_t>(code);
            }
        }
        // Length 258 uses code 28; it overwrites the last entry of code 27.
        length_code[length - 1] = static_cast<uint8_t>(code);
        base_length[kLengthCodes - 1] = 0;  // unused for code 28 (no extra bits)

        int dist = 0;
        for (code = 0; code < 16; ++code) {
            base_dist[code] = dist;
            for (int n = 0; n < (1 << kExtraDBits[code]); ++n) dist_code[dist++] = static_cast<uint8_t>(code);
        }
        dist >>= 7;
        for (; code < kDCodes; ++code) {
            base_dist[code] = dist << 7;
            for (int n = 0; n < (1 << (kExtraDBits[code] - 7)); ++n) {
                dist_code[256 + dist++] = static_cast<uint8_t>(code);
            }
        }

        int bl_count[kMaxBits + 1] = {0};
        int n = 0;
        while (n <= 143) ltree[n++].dl = 8, bl_count[8]++;
        while (n <= 255) ltree[n++].dl = 9, bl_count[9]++;
        while (n <= 279) ltree[n++].dl = 7, bl_count[7]++;
        while (n <= 287) ltree[n++].dl = 8, bl_count[8]++;
        // gen_codes over the static literal tree
        uint16_t next_code[kMaxBits + 1];
        unsigned c = 0;
        for (int bits = 1; bits <= kMaxBits; ++bits) {
            c = (c + static_cast<unsigned>(bl_count[bits - 1])) << 1;
            next_code[bits] = static_cast<uint16_t>(c);
        }
        for (n = 0; n <= kLCodes + 1; ++n) {
            const int len = ltree[n].dl;
            if (len == 0) continue;
            ltree[n].fc = static_cast<uint16_t>(reverse(next_code[len]++, len));
        }
        for (n = 0; n < kDCodes; ++n) {
            dtree[n].dl = 5;
            dtree[n].fc = static_cast<uint16_t>(reverse(static_cast<unsigned>(n), 5));
        }
    }
};

const StaticTables& static_tables() {
    static const StaticTables tables;
    return tables;
}

struct TreeDesc {
    TreeNode* dyn_tree = nullptr;
    int max_code = 0;
    const TreeNode* static_tree = nullptr;
    const int* extra_bits = nullptr;
    int extra_base = 0;
    int elems = 0;
    int max_length = 0;
};

} // namespace

struct DeflateEncoder::State {
    // Input stream
    const uint8_t* next_in = nullptr;
    size_t avail_in = 0;
    uint32_t adler = 1;
    std::vector<uint8_t>* out = nullptr;

    int strategy = 0;
    unsigned w_bits = 15;
    unsigned w_size = 0;
    unsigned w_mask = 0;
    std::vector<uint8_t> window;
    uint64_t window_size = 0;
    std::vector<uint16_t> prev;
    std::vector<uint16_t> head;
    unsigned ins_h = 0;
    unsigned hash_size = 0;
    unsigned hash_mask = 0;
    unsigned hash_shift = 0;
    long block_start = 0;
    unsigned match_length = 0;
    unsigned prev_match = 0;
    int match_available = 0;
    unsigned strstart = 0;
    unsigned match_start = 0;
    unsigned lookahead = 0;
    unsigned prev_length = 0;
    unsigned insert = 0;
    uint64_t high_water = 0;
    bool header_written = false;
    bool finished = false;

    TreeNode dyn_ltree[kHeapSize];
    TreeNode dyn_dtree[2 * kDCodes + 1];
    TreeNode bl_tree[2 * kBLCodes + 1];
    TreeDesc l_desc;
    TreeDesc d_desc;
    TreeDesc bl_desc;
    uint16_t bl_count[kMaxBits + 1] = {0};
    int heap[2 * kLCodes + 1] = {0};
    int heap_len = 0;
    int heap_max = 0;
    uint8_t depth[2 * kLCodes + 1] = {0};
    std::vector<uint8_t> sym_buf;
    unsigned lit_bufsize = 0;
    unsigned sym_next = 0;
    unsigned sym_end = 0;
    uint64_t opt_len = 0;
    uint64_t static_len = 0;

    uint64_t bi_buf = 0;
    int bi_valid = 0;

    unsigned max_dist() const { return w_size - kMinLookahead; }

    /* ---- bit output ---------------------------------------------------- */
    void put_byte(uint8_t b) { out->push_back(b); }
    void send_bits(unsigned value, int length) {
        bi_buf |= static_cast<uint64_t>(value) << bi_valid;
        bi_valid += length;
        while (bi_valid >= 8) {
            put_byte(static_cast<uint8_t>(bi_buf & 0xFFu));
            bi_buf >>= 8;
            bi_valid -= 8;
        }
    }
    void send_code(int c, const TreeNode* tree) { send_bits(tree[c].fc, tree[c].dl); }
    void bi_windup() {
        if (bi_valid > 0) put_byte(static_cast<uint8_t>(bi_buf & 0xFFu));
        bi_buf = 0;
        bi_valid = 0;
    }

    /* ---- trees --------------------------------------------------------- */
    void init_block() {
        for (int n = 0; n < kLCodes; ++n) dyn_ltree[n].fc = 0;
        for (int n = 0; n < kDCodes; ++n) dyn_dtree[n].fc = 0;
        for (int n = 0; n < kBLCodes; ++n) bl_tree[n].fc = 0;
        dyn_ltree[kEndBlock].fc = 1;
        opt_len = static_len = 0;
        sym_next = 0;
    }

    bool smaller(const TreeNode* tree, int n, int m) const {
        return tree[n].fc < tree[m].fc || (tree[n].fc == tree[m].fc && depth[n] <= depth[m]);
    }

    void pqdownheap(const TreeNode* tree, int k) {
        const int v = heap[k];
        int j = k << 1;
        while (j <= heap_len) {
            if (j < heap_len && smaller(tree, heap[j + 1], heap[j])) j++;
            if (smaller(tree, v, heap[j])) break;
            heap[k] = heap[j];
            k = j;
            j <<= 1;
        }
        heap[k] = v;
    }

    void gen_bitlen(TreeDesc* desc) {
        TreeNode* tree = desc->dyn_tree;
        const int max_code = desc->max_code;
        const TreeNode* stree = desc->static_tree;
        const int* extra = desc->extra_bits;
        const int base = desc->extra_base;
        const int max_length = desc->max_length;
        int h = 0;
        int overflow = 0;
        for (int bits = 0; bits <= kMaxBits; ++bits) bl_count[bits] = 0;
        tree[heap[heap_max]].dl = 0;
        for (h = heap_max + 1; h < kHeapSize; ++h) {
            const int n = heap[h];
            int bits = tree[tree[n].dl].dl + 1;
            if (bits > max_length) bits = max_length, overflow++;
            tree[n].dl = static_cast<uint16_t>(bits);
            if (n > max_code) continue;
            bl_count[bits]++;
            int xbits = 0;
            if (n >= base) xbits = extra[n - base];
            const uint16_t f = tree[n].fc;
            opt_len += static_cast<uint64_t>(f) * static_cast<unsigned>(bits + xbits);
            if (stree) static_len += static_cast<uint64_t>(f) * static_cast<unsigned>(stree[n].dl + xbits);
        }
        if (overflow == 0) return;
        do {
            int bits = max_length - 1;
            while (bl_count[bits] == 0) bits--;
            bl_count[bits]--;
            bl_count[bits + 1] = static_cast<uint16_t>(bl_count[bits + 1] + 2);
            bl_count[max_length]--;
            overflow -= 2;
        } while (overflow > 0);
        for (int bits = max_length; bits != 0; bits--) {
            int n = bl_count[bits];
            while (n != 0) {
                const int m = heap[--h];
                if (m > max_code) continue;
                if (tree[m].dl != static_cast<unsigned>(bits)) {
                    opt_len += (static_cast<uint64_t>(bits) - tree[m].dl) * tree[m].fc;
                    tree[m].dl = static_cast<uint16_t>(bits);
                }
                n--;
            }
        }
    }

    static void gen_codes(TreeNode* tree, int max_code, const uint16_t* counts) {
        uint16_t next_code[kMaxBits + 1];
        unsigned code = 0;
        for (int bits = 1; bits <= kMaxBits; ++bits) {
            code = (code + counts[bits - 1]) << 1;
            next_code[bits] = static_cast<uint16_t>(code);
        }
        for (int n = 0; n <= max_code; ++n) {
            const int len = tree[n].dl;
            if (len == 0) continue;
            tree[n].fc = static_cast<uint16_t>(StaticTables::reverse(next_code[len]++, len));
        }
    }

    void build_tree(TreeDesc* desc) {
        TreeNode* tree = desc->dyn_tree;
        const TreeNode* stree = desc->static_tree;
        const int elems = desc->elems;
        int max_code = -1;
        heap_len = 0;
        heap_max = kHeapSize;
        for (int n = 0; n < elems; ++n) {
            if (tree[n].fc != 0) {
                heap[++heap_len] = max_code = n;
                depth[n] = 0;
            } else {
                tree[n].dl = 0;
            }
        }
        while (heap_len < 2) {
            const int node = heap[++heap_len] = (max_code < 2 ? ++max_code : 0);
            tree[node].fc = 1;
            depth[node] = 0;
            opt_len--;
            if (stree) static_len -= stree[node].dl;
        }
        desc->max_code = max_code;
        for (int n = heap_len / 2; n >= 1; n--) pqdownheap(tree, n);
        int node = elems;
        do {
            const int n = heap[1];
            heap[1] = heap[heap_len--];
            pqdownheap(tree, 1);
            const int m = heap[1];
            heap[--heap_max] = n;
            heap[--heap_max] = m;
            tree[node].fc = static_cast<uint16_t>(tree[n].fc + tree[m].fc);
            depth[node] = static_cast<uint8_t>((depth[n] >= depth[m] ? depth[n] : depth[m]) + 1);
            tree[n].dl = tree[m].dl = static_cast<uint16_t>(node);
            heap[1] = node++;
            pqdownheap(tree, 1);
        } while (heap_len >= 2);
        heap[--heap_max] = heap[1];
        gen_bitlen(desc);
        gen_codes(tree, max_code, bl_count);
    }

    void scan_tree(TreeNode* tree, int max_code) {
        int prevlen = -1;
        int nextlen = tree[0].dl;
        int count = 0;
        int max_count = 7;
        int min_count = 4;
        if (nextlen == 0) max_count = 138, min_count = 3;
        tree[max_code + 1].dl = 0xFFFF;  // guard
        for (int n = 0; n <= max_code; ++n) {
            const int curlen = nextlen;
            nextlen = tree[n + 1].dl;
            if (++count < max_count && curlen == nextlen) {
                continue;
            } else if (count < min_count) {
                bl_tree[curlen].fc = static_cast<uint16_t>(bl_tree[curlen].fc + count);
            } else if (curlen != 0) {
                if (curlen != prevlen) bl_tree[curlen].fc++;
                bl_tree[kRep3To6].fc++;
            } else if (count <= 10) {
                bl_tree[kRepZ3To10].fc++;
            } else {
                bl_tree[kRepZ11To138].fc++;
            }
            count = 0;
            prevlen = curlen;
            if (nextlen == 0) {
                max_count = 138, min_count = 3;
            } else if (curlen == nextlen) {
                max_count = 6, min_count = 3;
            } else {
                max_count = 7, min_count = 4;
            }
        }
    }

    void send_tree(const TreeNode* tree, int max_code) {
        int prevlen = -1;
        int nextlen = tree[0].dl;
        int count = 0;
        int max_count = 7;
        int min_count = 4;
        if (nextlen == 0) max_count = 138, min_count = 3;
        for (int n = 0; n <= max_code; ++n) {
            const int curlen = nextlen;
            nextlen = tree[n + 1].dl;
            if (++count < max_count && curlen == nextlen) {
                continue;
            } else if (count < min_count) {
                do {
                    send_code(curlen, bl_tree);
                } while (--count != 0);
            } else if (curlen != 0) {
                if (curlen != prevlen) {
                    send_code(curlen, bl_tree);
                    count--;
                }
                send_code(kRep3To6, bl_tree);
                send_bits(static_cast<unsigned>(count - 3), 2);
            } else if (count <= 10) {
                send_code(kRepZ3To10, bl_tree);
                send_bits(static_cast<unsigned>(count - 3), 3);
            } else {
                send_code(kRepZ11To138, bl_tree);
                send_bits(static_cast<unsigned>(count - 11), 7);
            }
            count = 0;
            prevlen = curlen;
            if (nextlen == 0) {
                max_count = 138, min_count = 3;
            } else if (curlen == nextlen) {
                max_count = 6, min_count = 3;
            } else {
                max_count = 7, min_count = 4;
            }
        }
    }

    int build_bl_tree() {
        static const uint8_t bl_order[kBLCodes] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5,
                                                   11, 4, 12, 3, 13, 2, 14, 1, 15};
        scan_tree(dyn_ltree, l_desc.max_code);
        scan_tree(dyn_dtree, d_desc.max_code);
        build_tree(&bl_desc);
        int max_blindex = 0;
        for (max_blindex = kBLCodes - 1; max_blindex >= 3; max_blindex--) {
            if (bl_tree[bl_order[max_blindex]].dl != 0) break;
        }
        opt_len += 3 * (static_cast<uint64_t>(max_blindex) + 1) + 5 + 5 + 4;
        return max_blindex;
    }

    void send_all_trees(int lcodes, int dcodes, int blcodes) {
        static const uint8_t bl_order[kBLCodes] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5,
                                                   11, 4, 12, 3, 13, 2, 14, 1, 15};
        send_bits(static_cast<unsigned>(lcodes - 257), 5);
        send_bits(static_cast<unsigned>(dcodes - 1), 5);
        send_bits(static_cast<unsigned>(blcodes - 4), 4);
        for (int rank = 0; rank < blcodes; ++rank) send_bits(bl_tree[bl_order[rank]].dl, 3);
        send_tree(dyn_ltree, lcodes - 1);
        send_tree(dyn_dtree, dcodes - 1);
    }

    static int d_code(unsigned dist) {
        const StaticTables& t = static_tables();
        return dist < 256 ? t.dist_code[dist] : t.dist_code[256 + (dist >> 7)];
    }

    void compress_block(const TreeNode* ltree, const TreeNode* dtree) {
        const StaticTables& t = static_tables();
        unsigned sx = 0;
        if (sym_next != 0) {
            do {
                unsigned dist = sym_buf[sx++];
                dist += static_cast<unsigned>(sym_buf[sx++]) << 8;
                int lc = sym_buf[sx++];
                if (dist == 0) {
                    send_code(lc, ltree);
                } else {
                    int code = t.length_code[lc];
                    send_code(code + kLiterals + 1, ltree);
                    int extra = kExtraLBits[code];
                    if (extra != 0) {
                        lc -= t.base_length[code];
                        send_bits(static_cast<unsigned>(lc), extra);
                    }
                    dist--;
                    code = d_code(dist);
                    send_code(code, dtree);
                    extra = kExtraDBits[code];
                    if (extra != 0) {
                        dist -= static_cast<unsigned>(t.base_dist[code]);
                        send_bits(dist, extra);
                    }
                }
            } while (sx < sym_next);
        }
        send_code(kEndBlock, ltree);
    }

    void stored_block(const uint8_t* buf, uint64_t stored_len, int last) {
        send_bits(static_cast<unsigned>((0 << 1) + last), 3);
        bi_windup();
        put_byte(static_cast<uint8_t>(stored_len & 0xFFu));
        put_byte(static_cast<uint8_t>((stored_len >> 8) & 0xFFu));
        put_byte(static_cast<uint8_t>(~stored_len & 0xFFu));
        put_byte(static_cast<uint8_t>((~stored_len >> 8) & 0xFFu));
        if (stored_len) out->insert(out->end(), buf, buf + stored_len);
    }

    void flush_block(const uint8_t* buf, uint64_t stored_len, int last) {
        build_tree(&l_desc);
        build_tree(&d_desc);
        const int max_blindex = build_bl_tree();
        uint64_t opt_lenb = (opt_len + 3 + 7) >> 3;
        const uint64_t static_lenb = (static_len + 3 + 7) >> 3;
        if (static_lenb <= opt_lenb) opt_lenb = static_lenb;
        if (stored_len + 4 <= opt_lenb && buf != nullptr) {
            stored_block(buf, stored_len, last);
        } else if (static_lenb == opt_lenb) {
            const StaticTables& t = static_tables();
            send_bits(static_cast<unsigned>((1 << 1) + last), 3);
            compress_block(t.ltree, t.dtree);
        } else {
            send_bits(static_cast<unsigned>((2 << 1) + last), 3);
            send_all_trees(l_desc.max_code + 1, d_desc.max_code + 1, max_blindex + 1);
            compress_block(dyn_ltree, dyn_dtree);
        }
        init_block();
        if (last) bi_windup();
    }

    void flush_block_only(int last) {
        flush_block(block_start >= 0L ? &window[static_cast<unsigned>(block_start)] : nullptr,
                    static_cast<uint64_t>(static_cast<long>(strstart) - block_start), last);
        block_start = static_cast<long>(strstart);
    }

    bool tally_lit(uint8_t c) {
        sym_buf[sym_next++] = 0;
        sym_buf[sym_next++] = 0;
        sym_buf[sym_next++] = c;
        dyn_ltree[c].fc++;
        return sym_next == sym_end;
    }

    bool tally_dist(unsigned dist, unsigned len) {
        sym_buf[sym_next++] = static_cast<uint8_t>(dist);
        sym_buf[sym_next++] = static_cast<uint8_t>(dist >> 8);
        sym_buf[sym_next++] = static_cast<uint8_t>(len);
        dist--;
        dyn_ltree[static_tables().length_code[len] + kLiterals + 1].fc++;
        dyn_dtree[d_code(dist)].fc++;
        return sym_next == sym_end;
    }

    /* ---- matching ------------------------------------------------------ */
    void update_hash(unsigned c) { ins_h = ((ins_h << hash_shift) ^ c) & hash_mask; }

    unsigned insert_string(unsigned str) {
        update_hash(window[str + (kMinMatch - 1)]);
        const unsigned match_head = prev[str & w_mask] = head[ins_h];
        head[ins_h] = static_cast<uint16_t>(str);
        return match_head;
    }

    void slide_hash() {
        const unsigned wsize = w_size;
        for (unsigned n = 0; n < hash_size; ++n) {
            const unsigned m = head[n];
            head[n] = static_cast<uint16_t>(m >= wsize ? m - wsize : 0);
        }
        for (unsigned n = 0; n < wsize; ++n) {
            const unsigned m = prev[n];
            prev[n] = static_cast<uint16_t>(m >= wsize ? m - wsize : 0);
        }
    }

    unsigned read_buf(uint8_t* buf, unsigned size) {
        unsigned len = static_cast<unsigned>(avail_in < size ? avail_in : size);
        if (len == 0) return 0;
        avail_in -= len;
        std::memcpy(buf, next_in, len);
        adler = adler32_update(adler, next_in, len);
        next_in += len;
        return len;
    }

    void fill_window() {
        const unsigned wsize = w_size;
        do {
            unsigned more = static_cast<unsigned>(window_size - lookahead - strstart);
            if (strstart >= wsize + max_dist()) {
                std::memcpy(window.data(), window.data() + wsize, wsize - more);
                match_start -= wsize;
                strstart -= wsize;
                block_start -= static_cast<long>(wsize);
                if (insert > strstart) insert = strstart;
                slide_hash();
                more += wsize;
            }
            if (avail_in == 0) break;
            const unsigned n = read_buf(window.data() + strstart + lookahead, more);
            lookahead += n;
            if (lookahead + insert >= kMinMatch) {
                unsigned str = strstart - insert;
                ins_h = window[str];
                update_hash(window[str + 1]);
                while (insert) {
                    update_hash(window[str + kMinMatch - 1]);
                    prev[str & w_mask] = head[ins_h];
                    head[ins_h] = static_cast<uint16_t>(str);
                    str++;
                    insert--;
                    if (lookahead + insert < kMinMatch) break;
                }
            }
        } while (lookahead < kMinLookahead && avail_in != 0);

        if (high_water < window_size) {
            const uint64_t curr = static_cast<uint64_t>(strstart) + lookahead;
            if (high_water < curr) {
                uint64_t init = window_size - curr;
                if (init > kWinInit) init = kWinInit;
                std::memset(window.data() + curr, 0, static_cast<size_t>(init));
                high_water = curr + init;
            } else if (high_water < curr + kWinInit) {
                uint64_t init = curr + kWinInit - high_water;
                if (init > window_size - high_water) init = window_size - high_water;
                std::memset(window.data() + high_water, 0, static_cast<size_t>(init));
                high_water += init;
            }
        }
    }

    unsigned longest_match(unsigned cur_match) {
        unsigned chain_length = kMaxChain;
        const uint8_t* win = window.data();
        const uint8_t* scan = win + strstart;
        int best_len = static_cast<int>(prev_length);
        int nice_match = static_cast<int>(kNiceLength);
        const unsigned limit = strstart > max_dist() ? strstart - max_dist() : 0;
        const uint8_t* strend = win + strstart + kMaxMatch;
        uint8_t scan_end1 = scan[best_len - 1];
        uint8_t scan_end = scan[best_len];
        if (prev_length >= kGoodLength) chain_length >>= 2;
        if (static_cast<unsigned>(nice_match) > lookahead) nice_match = static_cast<int>(lookahead);
        do {
            const uint8_t* match = win + cur_match;
            if (match[best_len] != scan_end || match[best_len - 1] != scan_end1 ||
                match[0] != scan[0] || match[1] != scan[1]) {
                continue;
            }
            // Position 2 is implied equal by the hash; compare from 3 onward.
            scan += 2;
            match += 2;
            do {
            } while (*++scan == *++match && *++scan == *++match && *++scan == *++match &&
                     *++scan == *++match && *++scan == *++match && *++scan == *++match &&
                     *++scan == *++match && *++scan == *++match && scan < strend);
            const int len = kMaxMatch - static_cast<int>(strend - scan);
            scan = strend - kMaxMatch;
            if (len > best_len) {
                match_start = cur_match;
                best_len = len;
                if (len >= nice_match) break;
                scan_end1 = scan[best_len - 1];
                scan_end = scan[best_len];
            }
        } while ((cur_match = prev[cur_match & w_mask]) > limit && --chain_length != 0);
        if (static_cast<unsigned>(best_len) <= lookahead) return static_cast<unsigned>(best_len);
        return lookahead;
    }

    enum class BlockState { NeedMore, FinishDone };

    BlockState deflate_slow(bool finish) {
        for (;;) {
            if (lookahead < kMinLookahead) {
                fill_window();
                if (lookahead < kMinLookahead && !finish) return BlockState::NeedMore;
                if (lookahead == 0) break;
            }
            unsigned hash_head = 0;
            if (lookahead >= kMinMatch) hash_head = insert_string(strstart);
            prev_length = match_length;
            prev_match = match_start;
            match_length = kMinMatch - 1;
            if (hash_head != 0 && prev_length < kMaxLazy && strstart - hash_head <= max_dist()) {
                match_length = longest_match(hash_head);
                if (match_length <= 5 &&
                    (strategy == 1 || (match_length == kMinMatch && strstart - match_start > kTooFar))) {
                    match_length = kMinMatch - 1;
                }
            }
            if (prev_length >= kMinMatch && match_length <= prev_length) {
                const unsigned max_insert = strstart + lookahead - kMinMatch;
                const bool bflush = tally_dist(strstart - 1 - prev_match, prev_length - kMinMatch);
                lookahead -= prev_length - 1;
                prev_length -= 2;
                do {
                    if (++strstart <= max_insert) insert_string(strstart);
                } while (--prev_length != 0);
                match_available = 0;
                match_length = kMinMatch - 1;
                strstart++;
                if (bflush) flush_block_only(0);
            } else if (match_available) {
                const bool bflush = tally_lit(window[strstart - 1]);
                if (bflush) flush_block_only(0);
                strstart++;
                lookahead--;
            } else {
                match_available = 1;
                strstart++;
                lookahead--;
            }
        }
        if (match_available) {
            tally_lit(window[strstart - 1]);
            match_available = 0;
        }
        insert = strstart < kMinMatch - 1 ? strstart : kMinMatch - 1;
        flush_block_only(1);
        return BlockState::FinishDone;
    }

    void write_header() {
        unsigned header = (8u + ((w_bits - 8u) << 4)) << 8;
        unsigned level_flags = 2;  // default level
        header |= (level_flags << 6);
        header += 31 - (header % 31);
        put_byte(static_cast<uint8_t>(header >> 8));
        put_byte(static_cast<uint8_t>(header & 0xFFu));
        header_written = true;
    }
};

DeflateEncoder::DeflateEncoder(int window_bits, int strategy) : s_(new State) {
    State& s = *s_;
    if (window_bits < 9) window_bits = 9;
    if (window_bits > 15) window_bits = 15;
    s.out = &out_;
    s.strategy = strategy;
    s.w_bits = static_cast<unsigned>(window_bits);
    s.w_size = 1u << s.w_bits;
    s.w_mask = s.w_size - 1;
    const unsigned hash_bits = kMemLevel + 7;
    s.hash_size = 1u << hash_bits;
    s.hash_mask = s.hash_size - 1;
    s.hash_shift = (hash_bits + kMinMatch - 1) / kMinMatch;
    s.window.assign(static_cast<size_t>(s.w_size) * 2, 0);
    s.window_size = static_cast<uint64_t>(s.w_size) * 2;
    s.prev.assign(s.w_size, 0);
    s.head.assign(s.hash_size, 0);
    s.lit_bufsize = 1u << (kMemLevel + 6);
    s.sym_buf.assign(static_cast<size_t>(s.lit_bufsize) * 4, 0);
    s.sym_end = (s.lit_bufsize - 1) * 3;
    const StaticTables& t = static_tables();
    s.l_desc = TreeDesc{s.dyn_ltree, 0, t.ltree, kExtraLBits, kLiterals + 1, kLCodes, kMaxBits};
    s.d_desc = TreeDesc{s.dyn_dtree, 0, t.dtree, kExtraDBits, 0, kDCodes, kMaxBits};
    s.bl_desc = TreeDesc{s.bl_tree, 0, nullptr, kExtraBLBits, 0, kBLCodes, kMaxBLBits};
    s.init_block();
    s.match_length = s.prev_length = kMinMatch - 1;
}

DeflateEncoder::~DeflateEncoder() { delete s_; }

void DeflateEncoder::write(const uint8_t* data, size_t size) {
    State& s = *s_;
    if (s.finished) return;
    if (!s.header_written) s.write_header();
    s.next_in = data;
    s.avail_in = size;
    if (s.avail_in != 0 || s.lookahead != 0) {
        s.deflate_slow(false);
    }
}

void DeflateEncoder::finish() {
    State& s = *s_;
    if (s.finished) return;
    if (!s.header_written) s.write_header();
    s.next_in = nullptr;
    s.avail_in = 0;
    s.deflate_slow(true);
    const uint32_t a = s.adler;
    s.put_byte(static_cast<uint8_t>(a >> 24));
    s.put_byte(static_cast<uint8_t>(a >> 16));
    s.put_byte(static_cast<uint8_t>(a >> 8));
    s.put_byte(static_cast<uint8_t>(a));
    s.finished = true;
}

} // namespace eshkol::image
