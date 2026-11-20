/**
 * @file image_jpeg_decode.cpp
 * @brief Built-in JPEG decoder (baseline, extended-sequential and progressive
 *        Huffman) for the Eshkol runtime.
 *
 * The decoder follows the arithmetic of the reference IJG decoder in its
 * default configuration, so decoded pixels are identical to what the
 * system libjpeg produced for image-read before the codecs moved into the
 * runtime:
 *
 *   - entropy decoding per ITU T.81 Annex F/G, including restart-marker
 *     resynchronisation, and zero-filled data past a premature marker;
 *   - the accurate integer inverse DCT (13-bit constants, 2 pass bits) with
 *     the IJG post-IDCT range-limit table;
 *   - triangle ("fancy") chroma upsampling for 2:1 ratios, replication for
 *     other integral ratios;
 *   - 16-bit scaled-integer YCbCr to RGB conversion.
 *
 * Huffman tables 0 and 1 default to the Annex K tables when a stream omits
 * them (motion-JPEG frames).  Output is always 3-channel RGB; grayscale
 * images are replicated to RGB.  Arithmetic coding, lossless, hierarchical,
 * 12-bit and four-component (CMYK/YCCK) images are rejected.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "image_codecs.h"

#include <cstring>

namespace eshkol::image {

namespace {

const int kNaturalOrder[64 + 16] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    // Extra entries absorb run lengths that step past the end of a block.
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

/* Annex K default Huffman tables (bits[1..16], values). */
const uint8_t kDcLumBits[17] = {0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
const uint8_t kDcLumVals[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const uint8_t kDcChromBits[17] = {0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
const uint8_t kDcChromVals[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const uint8_t kAcLumBits[17] = {0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
const uint8_t kAcLumVals[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61,
    0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52,
    0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25,
    0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3,
    0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8,
    0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};
const uint8_t kAcChromBits[17] = {0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77};
const uint8_t kAcChromVals[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61,
    0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33,
    0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18,
    0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
    0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63,
    0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a,
    0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
    0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
    0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

constexpr int kLookahead = 9;

struct HuffTable {
    bool defined = false;
    uint8_t bits[17] = {0};
    uint8_t vals[256] = {0};
};

/** Decoding tables derived from a HuffTable (IJG layout). */
struct DerivedTable {
    int32_t maxcode[18];
    int32_t valoffset[18];
    const HuffTable* src = nullptr;
    // lookahead: (length << 8) | symbol, or 0 when the code is longer.
    uint16_t look[1 << kLookahead];
};

bool derive_table(const HuffTable& t, bool is_dc, DerivedTable* d) {
    int huffsize[257];
    unsigned huffcode[257];
    int p = 0;
    for (int l = 1; l <= 16; ++l) {
        int count = t.bits[l];
        if (p + count > 256) return false;
        while (count--) huffsize[p++] = l;
    }
    huffsize[p] = 0;
    const int numsymbols = p;
    unsigned code = 0;
    int si = huffsize[0];
    p = 0;
    while (huffsize[p]) {
        while (huffsize[p] == si) {
            huffcode[p++] = code;
            code++;
        }
        if (code >= (1u << si)) return false;
        code <<= 1;
        si++;
    }
    p = 0;
    for (int l = 1; l <= 16; ++l) {
        if (t.bits[l]) {
            d->valoffset[l] = p - static_cast<int32_t>(huffcode[p]);
            p += t.bits[l];
            d->maxcode[l] = static_cast<int32_t>(huffcode[p - 1]);
        } else {
            d->maxcode[l] = -1;
        }
    }
    d->valoffset[17] = 0;
    d->maxcode[17] = 0xFFFFF;
    std::memset(d->look, 0, sizeof(d->look));
    p = 0;
    for (int l = 1; l <= kLookahead; ++l) {
        for (int i = 1; i <= t.bits[l]; ++i, ++p) {
            int lookbits = static_cast<int>(huffcode[p]) << (kLookahead - l);
            for (int ctr = 1 << (kLookahead - l); ctr > 0; --ctr) {
                d->look[lookbits++] = static_cast<uint16_t>((l << 8) | t.vals[p]);
            }
        }
    }
    if (is_dc) {
        for (int i = 0; i < numsymbols; ++i) {
            if (t.vals[i] > 15) return false;
        }
    }
    d->src = &t;
    return true;
}

/** Entropy-coded segment reader: MSB-first bits, 0xFF00 unstuffing, stops at
 *  markers and then supplies zero bits (flagging insufficient data). */
struct BitStream {
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t pos = 0;
    uint64_t buf = 0;
    int bits = 0;
    bool at_marker = false;
    bool insufficient = false;

    bool fetch_byte(uint8_t* out) {
        if (at_marker || pos >= size) {
            at_marker = true;
            return false;
        }
        uint8_t c = data[pos];
        if (c != 0xFF) {
            ++pos;
            *out = c;
            return true;
        }
        size_t q = pos + 1;
        while (q < size && data[q] == 0xFF) ++q;
        if (q < size && data[q] == 0) {
            pos = q + 1;
            *out = 0xFF;
            return true;
        }
        // A marker: leave pos on the FF fill run so the marker parser sees it.
        pos = q - 1;
        at_marker = true;
        return false;
    }
    /** Load bytes until @p n bits are buffered or data stops; no padding. */
    void fill_soft(int n) {
        while (bits < n && bits <= 56) {
            uint8_t c;
            if (!fetch_byte(&c)) return;
            buf = (buf << 8) | c;
            bits += 8;
        }
    }
    /** Ensure @p n bits; past the data, zero bits are supplied. */
    void fill(int n) {
        fill_soft(n);
        if (bits < n) {
            insufficient = true;
            while (bits < n) {
                buf <<= 8;
                bits += 8;
            }
        }
    }
    int get(int n) {
        if (n == 0) return 0;
        fill(n);
        bits -= n;
        return static_cast<int>((buf >> bits) & ((uint64_t{1} << n) - 1));
    }
    int get1() { return get(1); }
    void reset_bits() {
        buf = 0;
        bits = 0;
    }

    int decode(const DerivedTable& t) {
        fill_soft(kLookahead);
        int l = 1;
        int32_t code = 0;
        if (bits >= kLookahead) {
            const int look = static_cast<int>((buf >> (bits - kLookahead)) & ((1 << kLookahead) - 1));
            const uint16_t e = t.look[look];
            if (e != 0) {
                bits -= e >> 8;
                return e & 0xFF;
            }
            code = get(kLookahead + 1);
            l = kLookahead + 1;
        } else {
            code = get(1);
        }
        while (code > t.maxcode[l]) {
            code = (code << 1) | get1();
            l++;
        }
        if (l > 16) return 0;  // invalid code: IJG decodes it as zero
        return t.src->vals[(code + t.valoffset[l]) & 0xFF];
    }
};

inline int extend(int x, int s) {
    return x < (1 << (s - 1)) ? x + ((-1) * (1 << s)) + 1 : x;
}

struct Component {
    int id = 0;
    int h = 1;
    int v = 1;
    int tq = 0;
    int width_in_blocks = 0;
    int height_in_blocks = 0;
    int blocks_per_line = 0;  // padded to MCU multiple
    int block_rows = 0;
    int downsampled_width = 0;
    int downsampled_height = 0;
    bool quant_latched = false;
    uint16_t quant[64] = {0};
    std::vector<int16_t> coefs;
    // Per-scan
    int dc_tbl = 0;
    int ac_tbl = 0;
    int last_dc = 0;
};

uint32_t div_round_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

/* ---- inverse DCT (accurate integer) ----------------------------------- */

constexpr int kConstBits = 13;
constexpr int kPass1Bits = 2;
constexpr int64_t FIX_0_298631336 = 2446;
constexpr int64_t FIX_0_390180644 = 3196;
constexpr int64_t FIX_0_541196100 = 4433;
constexpr int64_t FIX_0_765366865 = 6270;
constexpr int64_t FIX_0_899976223 = 7373;
constexpr int64_t FIX_1_175875602 = 9633;
constexpr int64_t FIX_1_501321110 = 12299;
constexpr int64_t FIX_1_847759065 = 15137;
constexpr int64_t FIX_1_961570560 = 16069;
constexpr int64_t FIX_2_053119869 = 16819;
constexpr int64_t FIX_2_562915447 = 20995;
constexpr int64_t FIX_3_072711026 = 25172;

inline int64_t descale(int64_t x, int n) { return (x + (int64_t{1} << (n - 1))) >> n; }

struct RangeLimit {
    // Post-IDCT table indexed by (value & 1023), centred as in IJG.
    uint8_t post_idct[1024];
    RangeLimit() {
        for (int i = 0; i < 1024; ++i) {
            if (i < 128) post_idct[i] = static_cast<uint8_t>(128 + i);
            else if (i < 512) post_idct[i] = 255;
            else if (i < 896) post_idct[i] = 0;
            else post_idct[i] = static_cast<uint8_t>(i - 896);
        }
    }
};

const RangeLimit& range_limit() {
    static const RangeLimit table;
    return table;
}

void idct_islow(const int16_t* coef, const uint16_t* quant, uint8_t* out, size_t stride) {
    int32_t ws[64];
    const uint8_t* rl = range_limit().post_idct;
    for (int c = 0; c < 8; ++c) {
        auto dq = [&](int r) -> int64_t {
            // 16-bit multiplier, as the IJG SIMD/16-bit configuration uses.
            return static_cast<int64_t>(coef[r * 8 + c]) * static_cast<int16_t>(quant[r * 8 + c]);
        };
        int64_t z2 = dq(2);
        int64_t z3 = dq(6);
        int64_t z1 = (z2 + z3) * FIX_0_541196100;
        int64_t tmp2 = z1 + z3 * (-FIX_1_847759065);
        int64_t tmp3 = z1 + z2 * FIX_0_765366865;
        z2 = dq(0);
        z3 = dq(4);
        int64_t tmp0 = (z2 + z3) * (int64_t{1} << kConstBits);
        int64_t tmp1 = (z2 - z3) * (int64_t{1} << kConstBits);
        const int64_t tmp10 = tmp0 + tmp3;
        const int64_t tmp13 = tmp0 - tmp3;
        const int64_t tmp11 = tmp1 + tmp2;
        const int64_t tmp12 = tmp1 - tmp2;
        tmp0 = dq(7);
        tmp1 = dq(5);
        tmp2 = dq(3);
        tmp3 = dq(1);
        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        int64_t z4 = tmp1 + tmp3;
        const int64_t z5 = (z3 + z4) * FIX_1_175875602;
        tmp0 = tmp0 * FIX_0_298631336;
        tmp1 = tmp1 * FIX_2_053119869;
        tmp2 = tmp2 * FIX_3_072711026;
        tmp3 = tmp3 * FIX_1_501321110;
        z1 = z1 * (-FIX_0_899976223);
        z2 = z2 * (-FIX_2_562915447);
        z3 = z3 * (-FIX_1_961570560);
        z4 = z4 * (-FIX_0_390180644);
        z3 += z5;
        z4 += z5;
        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;
        const int sh = kConstBits - kPass1Bits;
        ws[0 * 8 + c] = static_cast<int32_t>(descale(tmp10 + tmp3, sh));
        ws[7 * 8 + c] = static_cast<int32_t>(descale(tmp10 - tmp3, sh));
        ws[1 * 8 + c] = static_cast<int32_t>(descale(tmp11 + tmp2, sh));
        ws[6 * 8 + c] = static_cast<int32_t>(descale(tmp11 - tmp2, sh));
        ws[2 * 8 + c] = static_cast<int32_t>(descale(tmp12 + tmp1, sh));
        ws[5 * 8 + c] = static_cast<int32_t>(descale(tmp12 - tmp1, sh));
        ws[3 * 8 + c] = static_cast<int32_t>(descale(tmp13 + tmp0, sh));
        ws[4 * 8 + c] = static_cast<int32_t>(descale(tmp13 - tmp0, sh));
    }
    for (int r = 0; r < 8; ++r) {
        const int32_t* w = ws + r * 8;
        uint8_t* o = out + static_cast<size_t>(r) * stride;
        int64_t z2 = w[2];
        int64_t z3 = w[6];
        int64_t z1 = (z2 + z3) * FIX_0_541196100;
        int64_t tmp2 = z1 + z3 * (-FIX_1_847759065);
        int64_t tmp3 = z1 + z2 * FIX_0_765366865;
        int64_t tmp0 = (static_cast<int64_t>(w[0]) + w[4]) * (int64_t{1} << kConstBits);
        int64_t tmp1 = (static_cast<int64_t>(w[0]) - w[4]) * (int64_t{1} << kConstBits);
        const int64_t tmp10 = tmp0 + tmp3;
        const int64_t tmp13 = tmp0 - tmp3;
        const int64_t tmp11 = tmp1 + tmp2;
        const int64_t tmp12 = tmp1 - tmp2;
        tmp0 = w[7];
        tmp1 = w[5];
        tmp2 = w[3];
        tmp3 = w[1];
        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        int64_t z4 = tmp1 + tmp3;
        const int64_t z5 = (z3 + z4) * FIX_1_175875602;
        tmp0 = tmp0 * FIX_0_298631336;
        tmp1 = tmp1 * FIX_2_053119869;
        tmp2 = tmp2 * FIX_3_072711026;
        tmp3 = tmp3 * FIX_1_501321110;
        z1 = z1 * (-FIX_0_899976223);
        z2 = z2 * (-FIX_2_562915447);
        z3 = z3 * (-FIX_1_961570560);
        z4 = z4 * (-FIX_0_390180644);
        z3 += z5;
        z4 += z5;
        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;
        const int sh = kConstBits + kPass1Bits + 3;
        o[0] = rl[static_cast<int>(descale(tmp10 + tmp3, sh)) & 1023];
        o[7] = rl[static_cast<int>(descale(tmp10 - tmp3, sh)) & 1023];
        o[1] = rl[static_cast<int>(descale(tmp11 + tmp2, sh)) & 1023];
        o[6] = rl[static_cast<int>(descale(tmp11 - tmp2, sh)) & 1023];
        o[2] = rl[static_cast<int>(descale(tmp12 + tmp1, sh)) & 1023];
        o[5] = rl[static_cast<int>(descale(tmp12 - tmp1, sh)) & 1023];
        o[3] = rl[static_cast<int>(descale(tmp13 + tmp0, sh)) & 1023];
        o[4] = rl[static_cast<int>(descale(tmp13 - tmp0, sh)) & 1023];
    }
}

/* ---- the decoder ------------------------------------------------------- */

class Decoder {
public:
    Decoder(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    bool run(Raster* out);

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;

    HuffTable dc_tables_[4];
    HuffTable ac_tables_[4];
    bool qt_defined_[4] = {false, false, false, false};
    uint16_t qt_[4][64];
    int restart_interval_ = 0;
    bool saw_jfif_ = false;
    bool saw_adobe_ = false;
    int adobe_transform_ = 0;

    bool have_frame_ = false;
    bool progressive_ = false;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    std::vector<Component> comps_;
    int hmax_ = 1;
    int vmax_ = 1;
    int mcus_per_row_ = 0;
    int mcu_rows_ = 0;
    bool any_scan_ = false;
    bool multiple_scans_ = false;

    // Scan state
    int scan_comps_[4] = {0, 0, 0, 0};
    int ns_ = 0;
    int ss_ = 0, se_ = 63, ah_ = 0, al_ = 0;
    DerivedTable dc_derived_[4];
    DerivedTable ac_derived_[4];
    unsigned eobrun_ = 0;
    int next_restart_num_ = 0;
    int restarts_to_go_ = 0;
    BitStream bits_;

    bool read_u16(uint16_t* v) {
        if (pos_ + 2 > size_) return false;
        *v = static_cast<uint16_t>((data_[pos_] << 8) | data_[pos_ + 1]);
        pos_ += 2;
        return true;
    }

    /** Find the next marker from pos_ (skipping garbage and fill bytes).
     *  Returns the marker code, or -1 at end of data. */
    int next_marker() {
        for (;;) {
            while (pos_ < size_ && data_[pos_] != 0xFF) ++pos_;
            if (pos_ >= size_) return -1;
            while (pos_ < size_ && data_[pos_] == 0xFF) ++pos_;
            if (pos_ >= size_) return -1;
            const uint8_t c = data_[pos_++];
            if (c != 0) return c;
            // FF00 outside a scan: stuffed data, keep scanning
        }
    }

    bool parse_sof(int marker, const uint8_t* seg, size_t len);
    bool parse_dht(const uint8_t* seg, size_t len);
    bool parse_dqt(const uint8_t* seg, size_t len);
    bool parse_sos(const uint8_t* seg, size_t len);
    bool decode_scan();
    bool process_restart();
    void decode_block_sequential(Component& c, int16_t* block);
    bool output(Raster* out);
};

bool Decoder::parse_sof(int marker, const uint8_t* seg, size_t len) {
    if (have_frame_) return false;
    if (len < 6) return false;
    const int precision = seg[0];
    height_ = static_cast<uint32_t>((seg[1] << 8) | seg[2]);
    width_ = static_cast<uint32_t>((seg[3] << 8) | seg[4]);
    const int nc = seg[5];
    if (height_ == 0 || width_ == 0 || nc <= 0) return false;
    if (height_ > 65500 || width_ > 65500) return false;
    if (len != 6 + static_cast<size_t>(nc) * 3) return false;
    if (precision != 8 || nc > 10) return false;
    progressive_ = marker == 0xC2;
    comps_.assign(static_cast<size_t>(nc), Component{});
    hmax_ = vmax_ = 1;
    for (int i = 0; i < nc; ++i) {
        Component& c = comps_[static_cast<size_t>(i)];
        c.id = seg[6 + i * 3];
        c.h = seg[7 + i * 3] >> 4;
        c.v = seg[7 + i * 3] & 15;
        c.tq = seg[8 + i * 3];
        if (c.h < 1 || c.h > 4 || c.v < 1 || c.v > 4) return false;
        if (c.tq > 3) return false;
        if (c.h > hmax_) hmax_ = c.h;
        if (c.v > vmax_) vmax_ = c.v;
    }
    for (Component& c : comps_) {
        c.width_in_blocks = static_cast<int>(div_round_up(width_ * static_cast<uint32_t>(c.h),
                                                          static_cast<uint32_t>(hmax_ * 8)));
        c.height_in_blocks = static_cast<int>(div_round_up(height_ * static_cast<uint32_t>(c.v),
                                                           static_cast<uint32_t>(vmax_ * 8)));
        c.downsampled_width = static_cast<int>(div_round_up(width_ * static_cast<uint32_t>(c.h),
                                                            static_cast<uint32_t>(hmax_)));
        c.downsampled_height = static_cast<int>(div_round_up(height_ * static_cast<uint32_t>(c.v),
                                                             static_cast<uint32_t>(vmax_)));
        c.blocks_per_line = (c.width_in_blocks + c.h - 1) / c.h * c.h;
        c.block_rows = (c.height_in_blocks + c.v - 1) / c.v * c.v;
    }
    mcus_per_row_ = static_cast<int>(div_round_up(width_, static_cast<uint32_t>(hmax_ * 8)));
    mcu_rows_ = static_cast<int>(div_round_up(height_, static_cast<uint32_t>(vmax_ * 8)));
    for (Component& c : comps_) {
        // Interleaved scans address blocks up to the MCU grid.
        const int need_w = mcus_per_row_ * c.h;
        const int need_h = mcu_rows_ * c.v;
        if (need_w > c.blocks_per_line) c.blocks_per_line = need_w;
        if (need_h > c.block_rows) c.block_rows = need_h;
        c.coefs.assign(static_cast<size_t>(c.blocks_per_line) * static_cast<size_t>(c.block_rows) * 64, 0);
    }
    have_frame_ = true;
    return true;
}

bool Decoder::parse_dht(const uint8_t* seg, size_t len) {
    size_t p = 0;
    while (p < len) {
        if (p + 17 > len) return false;
        const int index = seg[p];
        HuffTable t;
        int count = 0;
        for (int i = 1; i <= 16; ++i) {
            t.bits[i] = seg[p + static_cast<size_t>(i)];
            count += t.bits[i];
        }
        p += 17;
        if (count > 256 || p + static_cast<size_t>(count) > len) return false;
        std::memcpy(t.vals, seg + p, static_cast<size_t>(count));
        p += static_cast<size_t>(count);
        t.defined = true;
        const int slot = index & 0x0F;
        if (slot > 3) return false;
        if (index & 0x10) ac_tables_[slot] = t;
        else dc_tables_[slot] = t;
    }
    return true;
}

bool Decoder::parse_dqt(const uint8_t* seg, size_t len) {
    size_t p = 0;
    while (p < len) {
        const int pq = seg[p] >> 4;
        const int tq = seg[p] & 0x0F;
        ++p;
        if (tq > 3 || pq > 1) return false;
        const size_t need = pq ? 128 : 64;
        if (p + need > len) return false;
        for (int i = 0; i < 64; ++i) {
            const unsigned v = pq ? static_cast<unsigned>((seg[p + 2 * i] << 8) | seg[p + 2 * i + 1])
                                  : seg[p + static_cast<size_t>(i)];
            qt_[tq][kNaturalOrder[i]] = static_cast<uint16_t>(v);
        }
        qt_defined_[tq] = true;
        p += need;
    }
    return true;
}

bool Decoder::parse_sos(const uint8_t* seg, size_t len) {
    if (!have_frame_ || len < 1) return false;
    ns_ = seg[0];
    if (ns_ < 1 || ns_ > 4 || len != 4 + static_cast<size_t>(ns_) * 2) return false;
    for (int i = 0; i < ns_; ++i) {
        const int id = seg[1 + i * 2];
        int found = -1;
        for (size_t k = 0; k < comps_.size(); ++k) {
            if (comps_[k].id == id) {
                found = static_cast<int>(k);
                break;
            }
        }
        if (found < 0) return false;
        for (int j = 0; j < i; ++j) {
            if (scan_comps_[j] == found) return false;
        }
        scan_comps_[i] = found;
        comps_[static_cast<size_t>(found)].dc_tbl = seg[2 + i * 2] >> 4;
        comps_[static_cast<size_t>(found)].ac_tbl = seg[2 + i * 2] & 15;
        if (comps_[static_cast<size_t>(found)].dc_tbl > 3 || comps_[static_cast<size_t>(found)].ac_tbl > 3) return false;
    }
    const size_t t = 1 + static_cast<size_t>(ns_) * 2;
    ss_ = seg[t];
    se_ = seg[t + 1];
    ah_ = seg[t + 2] >> 4;
    al_ = seg[t + 2] & 15;
    if (progressive_) {
        if (ss_ == 0) {
            if (se_ != 0) return false;
        } else {
            if (se_ < ss_ || se_ > 63 || ns_ != 1) return false;
        }
        if (ah_ != 0 && al_ != ah_ - 1) return false;
        if (al_ > 13) return false;
    } else {
        ss_ = 0;
        se_ = 63;
    }
    // Latch quantization tables at a component's first scan.
    for (int i = 0; i < ns_; ++i) {
        Component& c = comps_[static_cast<size_t>(scan_comps_[i])];
        if (c.quant_latched) continue;
        if (!qt_defined_[c.tq]) return false;
        std::memcpy(c.quant, qt_[c.tq], sizeof(c.quant));
        c.quant_latched = true;
    }
    // Default Annex K tables fill empty slots 0 and 1.
    auto install = [](HuffTable* t, const uint8_t* b, const uint8_t* v, int n) {
        if (t->defined) return;
        std::memcpy(t->bits, b, 17);
        std::memset(t->vals, 0, sizeof(t->vals));
        std::memcpy(t->vals, v, static_cast<size_t>(n));
        t->defined = true;
    };
    install(&dc_tables_[0], kDcLumBits, kDcLumVals, 12);
    install(&ac_tables_[0], kAcLumBits, kAcLumVals, 162);
    install(&dc_tables_[1], kDcChromBits, kDcChromVals, 12);
    install(&ac_tables_[1], kAcChromBits, kAcChromVals, 162);
    for (int i = 0; i < ns_; ++i) {
        Component& c = comps_[static_cast<size_t>(scan_comps_[i])];
        const bool need_dc = !progressive_ || (ss_ == 0 && ah_ == 0);
        const bool need_ac = !progressive_ || ss_ != 0;
        if (need_dc) {
            if (!dc_tables_[c.dc_tbl].defined) return false;
            if (!derive_table(dc_tables_[c.dc_tbl], true, &dc_derived_[c.dc_tbl])) return false;
        }
        if (need_ac) {
            if (!ac_tables_[c.ac_tbl].defined) return false;
            if (!derive_table(ac_tables_[c.ac_tbl], false, &ac_derived_[c.ac_tbl])) return false;
        }
    }
    return true;
}

bool Decoder::process_restart() {
    bits_.reset_bits();
    // Locate the marker that ends this restart interval.
    pos_ = bits_.pos;
    int marker = -1;
    if (bits_.at_marker && pos_ + 1 < size_ && data_[pos_] == 0xFF) {
        size_t q = pos_;
        while (q < size_ && data_[q] == 0xFF) ++q;
        if (q < size_) {
            marker = data_[q];
            pos_ = q + 1;
        }
    } else {
        marker = next_marker();
    }
    const int desired = next_restart_num_;
    bool consumed = false;
    for (;;) {
        int action;
        if (marker < 0) {
            action = 3;
        } else if (marker < 0xC0) {
            action = 2;
        } else if (marker < 0xD0 || marker > 0xD7) {
            action = 3;
        } else {
            const int n = marker - 0xD0;
            if (n == ((desired + 1) & 7) || n == ((desired + 2) & 7)) action = 3;
            else if (n == ((desired - 1) & 7) || n == ((desired - 2) & 7)) action = 2;
            else action = 1;
        }
        if (action == 1) {
            consumed = true;
            break;
        }
        if (action == 2) {
            marker = next_marker();
            continue;
        }
        break;  // action 3: leave the marker for the caller
    }
    if (consumed) {
        bits_.at_marker = false;
        bits_.pos = pos_;
        bits_.insufficient = false;
    } else {
        // Back up onto the marker so the entropy reader stops there.
        if (marker >= 0) {
            size_t back = pos_ - 1;
            while (back > 0 && data_[back - 1] == 0xFF) --back;
            pos_ = back;
            bits_.pos = back;
        } else {
            bits_.pos = size_;
        }
        bits_.at_marker = true;
    }
    for (Component& c : comps_) c.last_dc = 0;
    eobrun_ = 0;
    restarts_to_go_ = restart_interval_;
    next_restart_num_ = (next_restart_num_ + 1) & 7;
    return true;
}

void Decoder::decode_block_sequential(Component& c, int16_t* block) {
    int s = bits_.decode(dc_derived_[c.dc_tbl]);
    if (s) {
        const int r = bits_.get(s);
        s = extend(r, s);
    }
    s += c.last_dc;
    c.last_dc = s;
    block[0] = static_cast<int16_t>(s);
    const DerivedTable& ac = ac_derived_[c.ac_tbl];
    for (int k = 1; k < 64; ++k) {
        s = bits_.decode(ac);
        int r = s >> 4;
        s &= 15;
        if (s) {
            k += r;
            r = bits_.get(s);
            s = extend(r, s);
            block[kNaturalOrder[k]] = static_cast<int16_t>(s);
        } else {
            if (r != 15) break;
            k += 15;
        }
    }
}

bool Decoder::decode_scan() {
    bits_ = BitStream{};
    bits_.data = data_;
    bits_.size = size_;
    bits_.pos = pos_;
    for (Component& c : comps_) c.last_dc = 0;
    eobrun_ = 0;
    next_restart_num_ = 0;
    restarts_to_go_ = restart_interval_;

    int units_w = 0;
    int units_h = 0;
    if (ns_ == 1) {
        const Component& c = comps_[static_cast<size_t>(scan_comps_[0])];
        units_w = c.width_in_blocks;
        units_h = c.height_in_blocks;
    } else {
        units_w = mcus_per_row_;
        units_h = mcu_rows_;
    }

    const int p1 = 1 << al_;
    const int m1 = -1 * (1 << al_);
    for (int my = 0; my < units_h; ++my) {
        for (int mx = 0; mx < units_w; ++mx) {
            if (restart_interval_) {
                if (restarts_to_go_ == 0) process_restart();
                restarts_to_go_--;
            }
            // As in IJG, running out of data is checked once per MCU: the MCU
            // that hits the end is decoded from zero bits, later ones are skipped.
            const bool skip = bits_.insufficient;
            for (int ci = 0; ci < ns_; ++ci) {
                Component& c = comps_[static_cast<size_t>(scan_comps_[ci])];
                const int bw = ns_ == 1 ? 1 : c.h;
                const int bh = ns_ == 1 ? 1 : c.v;
                for (int by = 0; by < bh; ++by) {
                    for (int bx = 0; bx < bw; ++bx) {
                        const int row = ns_ == 1 ? my : my * c.v + by;
                        const int col = ns_ == 1 ? mx : mx * c.h + bx;
                        int16_t* block = c.coefs.data() +
                                         (static_cast<size_t>(row) * static_cast<size_t>(c.blocks_per_line) +
                                          static_cast<size_t>(col)) * 64;
                        if (!progressive_) {
                            if (!skip) decode_block_sequential(c, block);
                        } else if (ss_ == 0 && ah_ == 0) {
                            if (skip) continue;
                            int s = bits_.decode(dc_derived_[c.dc_tbl]);
                            if (s) {
                                const int r = bits_.get(s);
                                s = extend(r, s);
                            }
                            s += c.last_dc;
                            c.last_dc = s;
                            block[0] = static_cast<int16_t>(static_cast<uint32_t>(s) << al_);
                        } else if (ss_ == 0) {
                            if (bits_.get1()) block[0] = static_cast<int16_t>(block[0] | p1);
                        } else if (ah_ == 0) {
                            if (skip) continue;
                            if (eobrun_ > 0) {
                                eobrun_--;
                                continue;
                            }
                            const DerivedTable& ac = ac_derived_[c.ac_tbl];
                            for (int k = ss_; k <= se_; ++k) {
                                int s = bits_.decode(ac);
                                int r = s >> 4;
                                s &= 15;
                                if (s) {
                                    k += r;
                                    r = bits_.get(s);
                                    s = extend(r, s);
                                    block[kNaturalOrder[k]] = static_cast<int16_t>(static_cast<uint32_t>(s) << al_);
                                } else if (r == 15) {
                                    k += 15;
                                } else {
                                    eobrun_ = 1u << r;
                                    if (r) eobrun_ += static_cast<unsigned>(bits_.get(r));
                                    eobrun_--;
                                    break;
                                }
                            }
                        } else {
                            if (skip) continue;
                            const DerivedTable& ac = ac_derived_[c.ac_tbl];
                            int k = ss_;
                            if (eobrun_ == 0) {
                                for (; k <= se_; ++k) {
                                    int s = bits_.decode(ac);
                                    int r = s >> 4;
                                    s &= 15;
                                    if (s) {
                                        s = bits_.get1() ? p1 : m1;
                                    } else if (r != 15) {
                                        eobrun_ = 1u << r;
                                        if (r) eobrun_ += static_cast<unsigned>(bits_.get(r));
                                        break;
                                    }
                                    do {
                                        int16_t* coef = block + kNaturalOrder[k];
                                        if (*coef != 0) {
                                            if (bits_.get1()) {
                                                if ((*coef & p1) == 0) {
                                                    *coef = static_cast<int16_t>(*coef >= 0 ? *coef + p1 : *coef + m1);
                                                }
                                            }
                                        } else {
                                            if (--r < 0) break;
                                        }
                                        k++;
                                    } while (k <= se_);
                                    if (s) block[kNaturalOrder[k]] = static_cast<int16_t>(s);
                                }
                            }
                            if (eobrun_ > 0) {
                                for (; k <= se_; ++k) {
                                    int16_t* coef = block + kNaturalOrder[k];
                                    if (*coef != 0) {
                                        if (bits_.get1()) {
                                            if ((*coef & p1) == 0) {
                                                *coef = static_cast<int16_t>(*coef >= 0 ? *coef + p1 : *coef + m1);
                                            }
                                        }
                                    }
                                }
                                eobrun_--;
                            }
                        }
                    }
                }
            }
        }
    }
    pos_ = bits_.pos;
    any_scan_ = true;
    return true;
}

bool Decoder::run(Raster* out) {
    if (size_ < 2 || data_[0] != 0xFF || data_[1] != 0xD8) return false;
    pos_ = 2;
    for (;;) {
        const int marker = next_marker();
        if (marker < 0) break;  // end of data acts as EOI
        if (marker == 0xD9) break;
        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) continue;  // TEM, stray RSTn
        if (marker == 0xD8) return false;  // second SOI
        uint16_t length = 0;
        if (!read_u16(&length) || length < 2) {
            // Truncated header: stop as if at end of data.
            break;
        }
        const size_t seg_len = static_cast<size_t>(length) - 2;
        if (pos_ + seg_len > size_) break;
        const uint8_t* seg = data_ + pos_;
        pos_ += seg_len;
        switch (marker) {
            case 0xC0:
            case 0xC1:
            case 0xC2:
                if (!parse_sof(marker, seg, seg_len)) return false;
                break;
            case 0xC4:
                if (!parse_dht(seg, seg_len)) return false;
                break;
            case 0xDB:
                if (!parse_dqt(seg, seg_len)) return false;
                break;
            case 0xDD:
                if (seg_len != 2) return false;
                restart_interval_ = (seg[0] << 8) | seg[1];
                break;
            case 0xDA:
                // A single-scan sequential image ends after its scan.
                if (any_scan_ && !multiple_scans_) return false;
                if (!parse_sos(seg, seg_len)) return false;
                if (!any_scan_) multiple_scans_ = progressive_ || ns_ < static_cast<int>(comps_.size());
                if (!decode_scan()) return false;
                break;
            case 0xDC:  // DNL: ignored
            case 0xFE:  // COM
                break;
            default:
                if (marker >= 0xE0 && marker <= 0xEF) {
                    if (marker == 0xE0 && seg_len >= 14 && std::memcmp(seg, "JFIF\0", 5) == 0) saw_jfif_ = true;
                    if (marker == 0xEE && seg_len >= 12 && std::memcmp(seg, "Adobe", 5) == 0) {
                        saw_adobe_ = true;
                        adobe_transform_ = seg[11];
                    }
                    break;
                }
                // SOF3/5-7/9-11/13-15 (lossless, hierarchical, arithmetic) and
                // reserved markers are not supported.
                return false;
        }
    }
    if (!have_frame_ || !any_scan_) return false;
    return output(out);
}

/* ---- upsampling and colour conversion ---------------------------------- */

struct ColorTables {
    int cr_r[256];
    int cb_b[256];
    int64_t cr_g[256];
    int64_t cb_g[256];
    ColorTables() {
        constexpr int kScale = 16;
        constexpr int64_t kHalf = int64_t{1} << (kScale - 1);
        auto fix = [](double x) { return static_cast<int64_t>(x * (1L << kScale) + 0.5); };
        for (int i = 0, x = -128; i <= 255; ++i, ++x) {
            cr_r[i] = static_cast<int>((fix(1.40200) * x + kHalf) >> kScale);
            cb_b[i] = static_cast<int>((fix(1.77200) * x + kHalf) >> kScale);
            cr_g[i] = (-fix(0.71414)) * x;
            cb_g[i] = (-fix(0.34414)) * x + kHalf;
        }
    }
};

const ColorTables& color_tables() {
    static const ColorTables tables;
    return tables;
}

inline uint8_t clamp255(int v) { return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v)); }

/** Produce a full-resolution (width x height) plane for one component. */
void upsample(const Component& c, const std::vector<uint8_t>& plane, int plane_stride, int hmax, int vmax,
              uint32_t width, uint32_t height, std::vector<uint8_t>* full) {
    const size_t W = width;
    const size_t H = height;
    full->assign(W * H, 0);
    const int hexp = hmax / c.h;
    const int vexp = vmax / c.v;
    const int dw = c.downsampled_width;
    const int dh = c.downsampled_height;
    auto in = [&](int y, int x) -> int {
        return plane[static_cast<size_t>(y) * static_cast<size_t>(plane_stride) + static_cast<size_t>(x)];
    };
    auto clamp_row = [&](int y) { return y < 0 ? 0 : (y >= dh ? dh - 1 : y); };
    const bool h2v1 = c.h * 2 == hmax && c.v == vmax;
    const bool h1v2 = c.h == hmax && c.v * 2 == vmax;
    const bool h2v2 = c.h * 2 == hmax && c.v * 2 == vmax;
    if (c.h == hmax && c.v == vmax) {
        for (size_t y = 0; y < H; ++y) {
            std::memcpy(full->data() + y * W, plane.data() + y * static_cast<size_t>(plane_stride), W);
        }
    } else if (h2v1 && dw > 2) {
        std::vector<uint8_t> row(static_cast<size_t>(dw) * 2);
        for (size_t y = 0; y < H; ++y) {
            const int iy = static_cast<int>(y);
            int v0 = in(iy, 0);
            row[0] = static_cast<uint8_t>(v0);
            row[1] = static_cast<uint8_t>((v0 * 3 + in(iy, 1) + 2) >> 2);
            for (int x = 1; x < dw - 1; ++x) {
                const int v = in(iy, x) * 3;
                row[static_cast<size_t>(2 * x)] = static_cast<uint8_t>((v + in(iy, x - 1) + 1) >> 2);
                row[static_cast<size_t>(2 * x + 1)] = static_cast<uint8_t>((v + in(iy, x + 1) + 2) >> 2);
            }
            const int vl = in(iy, dw - 1);
            row[static_cast<size_t>(2 * (dw - 1))] = static_cast<uint8_t>((vl * 3 + in(iy, dw - 2) + 1) >> 2);
            row[static_cast<size_t>(2 * (dw - 1) + 1)] = static_cast<uint8_t>(vl);
            std::memcpy(full->data() + y * W, row.data(), W);
        }
    } else if (h1v2) {
        for (size_t y = 0; y < H; ++y) {
            const int iy = static_cast<int>(y / 2);
            const int v = static_cast<int>(y & 1);
            const int other = clamp_row(v == 0 ? iy - 1 : iy + 1);
            const int bias = v == 0 ? 1 : 2;
            for (size_t x = 0; x < W; ++x) {
                const int sum = in(iy, static_cast<int>(x)) * 3 + in(other, static_cast<int>(x));
                (*full)[y * W + x] = static_cast<uint8_t>((sum + bias) >> 2);
            }
        }
    } else if (h2v2 && dw > 2) {
        std::vector<uint8_t> row(static_cast<size_t>(dw) * 2);
        for (size_t y = 0; y < H; ++y) {
            const int iy = static_cast<int>(y / 2);
            const int v = static_cast<int>(y & 1);
            const int other = clamp_row(v == 0 ? iy - 1 : iy + 1);
            auto colsum = [&](int x) { return in(iy, x) * 3 + in(other, x); };
            int thiscol = colsum(0);
            int nextcol = colsum(1);
            row[0] = static_cast<uint8_t>((thiscol * 4 + 8) >> 4);
            row[1] = static_cast<uint8_t>((thiscol * 3 + nextcol + 7) >> 4);
            int lastcol = thiscol;
            thiscol = nextcol;
            for (int x = 1; x < dw - 1; ++x) {
                nextcol = colsum(x + 1);
                row[static_cast<size_t>(2 * x)] = static_cast<uint8_t>((thiscol * 3 + lastcol + 8) >> 4);
                row[static_cast<size_t>(2 * x + 1)] = static_cast<uint8_t>((thiscol * 3 + nextcol + 7) >> 4);
                lastcol = thiscol;
                thiscol = nextcol;
            }
            row[static_cast<size_t>(2 * (dw - 1))] = static_cast<uint8_t>((thiscol * 3 + lastcol + 8) >> 4);
            row[static_cast<size_t>(2 * (dw - 1) + 1)] = static_cast<uint8_t>((thiscol * 4 + 7) >> 4);
            std::memcpy(full->data() + y * W, row.data(), W);
        }
    } else {
        // Integral replication (also the 2:1 cases when too narrow to filter).
        for (size_t y = 0; y < H; ++y) {
            const int iy = static_cast<int>(y / static_cast<size_t>(vexp));
            for (size_t x = 0; x < W; ++x) {
                (*full)[y * W + x] = static_cast<uint8_t>(in(iy, static_cast<int>(x / static_cast<size_t>(hexp))));
            }
        }
    }
}

bool Decoder::output(Raster* out) {
    const size_t nc = comps_.size();
    enum class Space { Gray, YCbCr, Rgb } space;
    if (nc == 1) {
        space = Space::Gray;
    } else if (nc == 3) {
        if (saw_jfif_) {
            space = Space::YCbCr;
        } else if (saw_adobe_) {
            space = adobe_transform_ == 0 ? Space::Rgb : Space::YCbCr;
        } else {
            const int c0 = comps_[0].id, c1 = comps_[1].id, c2 = comps_[2].id;
            space = (c0 == 82 && c1 == 71 && c2 == 66) ? Space::Rgb : Space::YCbCr;
        }
    } else {
        return false;  // CMYK/YCCK or unknown layouts
    }
    if (width_ > static_cast<uint32_t>(kMaxDimension) || height_ > static_cast<uint32_t>(kMaxDimension)) return false;
    for (const Component& c : comps_) {
        // A component that never appeared in a scan keeps an all-zero
        // dequantization table, so it reconstructs as mid-gray (as in IJG).
        if (hmax_ % c.h != 0 || vmax_ % c.v != 0) return false;
    }

    std::vector<std::vector<uint8_t>> full(nc);
    for (size_t ci = 0; ci < nc; ++ci) {
        const Component& c = comps_[ci];
        const int stride = c.width_in_blocks * 8;
        std::vector<uint8_t> plane(static_cast<size_t>(stride) * static_cast<size_t>(c.height_in_blocks) * 8);
        for (int by = 0; by < c.height_in_blocks; ++by) {
            for (int bx = 0; bx < c.width_in_blocks; ++bx) {
                const int16_t* block = c.coefs.data() +
                                       (static_cast<size_t>(by) * static_cast<size_t>(c.blocks_per_line) +
                                        static_cast<size_t>(bx)) * 64;
                idct_islow(block, c.quant,
                           plane.data() + static_cast<size_t>(by) * 8 * static_cast<size_t>(stride) +
                               static_cast<size_t>(bx) * 8,
                           static_cast<size_t>(stride));
            }
        }
        upsample(c, plane, stride, hmax_, vmax_, width_, height_, &full[ci]);
    }

    const size_t count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    out->width = static_cast<int>(width_);
    out->height = static_cast<int>(height_);
    out->channels = 3;
    out->pixels.resize(count * 3);
    uint8_t* dst = out->pixels.data();
    if (space == Space::Gray) {
        const uint8_t* y = full[0].data();
        for (size_t i = 0; i < count; ++i) dst[3 * i] = dst[3 * i + 1] = dst[3 * i + 2] = y[i];
    } else if (space == Space::Rgb) {
        for (size_t i = 0; i < count; ++i) {
            dst[3 * i] = full[0][i];
            dst[3 * i + 1] = full[1][i];
            dst[3 * i + 2] = full[2][i];
        }
    } else {
        const ColorTables& t = color_tables();
        for (size_t i = 0; i < count; ++i) {
            const int y = full[0][i];
            const int cb = full[1][i];
            const int cr = full[2][i];
            dst[3 * i] = clamp255(y + t.cr_r[cr]);
            dst[3 * i + 1] = clamp255(y + static_cast<int>((t.cb_g[cb] + t.cr_g[cr]) >> 16));
            dst[3 * i + 2] = clamp255(y + t.cb_b[cb]);
        }
    }
    return true;
}

} // namespace

bool jpeg_is_signature(const uint8_t* data, size_t size) {
    return data && size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF;
}

bool jpeg_decode(const uint8_t* data, size_t size, Raster* out) {
    if (!out || !jpeg_is_signature(data, size)) return false;
    Decoder decoder(data, size);
    return decoder.run(out);
}

} // namespace eshkol::image
