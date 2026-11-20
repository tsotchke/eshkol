/**
 * @file image_jpeg_encode.cpp
 * @brief Built-in baseline JPEG encoder for the Eshkol runtime.
 *
 * The encoder reproduces the reference IJG compressor's default settings
 * byte for byte: JFIF 1.01 header, Annex K quantization tables scaled by the
 * IJG quality formula (clamped to baseline), Annex K Huffman tables, 2x2
 * chroma subsampling with the alternating-bias box filter, 16-bit
 * scaled-integer RGB to YCbCr conversion, the accurate integer forward DCT,
 * reciprocal quantization, edge replication to whole blocks and DC-copy
 * dummy blocks at MCU edges.  image-write therefore keeps producing the same
 * JPEG files the system libjpeg wrote before.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "image_codecs.h"

#include <cstring>

namespace eshkol::image {

namespace {

const int kZigzag[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
                         12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
                         35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
                         58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

const unsigned kStdLuminanceQuant[64] = {
    16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};
const unsigned kStdChrominanceQuant[64] = {
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};

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

struct EncodeTable {
    unsigned code[256] = {0};
    int size[256] = {0};
};

void derive_encode_table(const uint8_t* bits, const uint8_t* vals, EncodeTable* t) {
    int huffsize[257];
    unsigned huffcode[257];
    int p = 0;
    for (int l = 1; l <= 16; ++l) {
        for (int i = 1; i <= bits[l]; ++i) huffsize[p++] = l;
    }
    huffsize[p] = 0;
    const int lastp = p;
    unsigned code = 0;
    int si = huffsize[0];
    p = 0;
    while (huffsize[p]) {
        while (huffsize[p] == si) {
            huffcode[p++] = code;
            code++;
        }
        code <<= 1;
        si++;
    }
    for (p = 0; p < lastp; ++p) {
        t->code[vals[p]] = huffcode[p];
        t->size[vals[p]] = huffsize[p];
    }
}

class Writer {
public:
    explicit Writer(std::vector<uint8_t>* out) : out_(out) {}
    void byte(uint8_t b) { out_->push_back(b); }
    void u16(unsigned v) {
        byte(static_cast<uint8_t>(v >> 8));
        byte(static_cast<uint8_t>(v));
    }
    void marker(uint8_t m) {
        byte(0xFF);
        byte(m);
    }
    void bits(unsigned code, int size) {
        if (size == 0) return;
        acc_ = (acc_ << size) | (code & ((1u << size) - 1u));
        count_ += size;
        while (count_ >= 8) {
            const uint8_t c = static_cast<uint8_t>((acc_ >> (count_ - 8)) & 0xFFu);
            byte(c);
            if (c == 0xFF) byte(0);
            count_ -= 8;
        }
        acc_ &= (uint64_t{1} << count_) - 1;
    }
    void flush_bits() {
        // Pad the final partial byte with one bits.
        bits(0x7F, 7);
        acc_ = 0;
        count_ = 0;
    }

private:
    std::vector<uint8_t>* out_;
    uint64_t acc_ = 0;
    int count_ = 0;
};

/* ---- forward DCT (accurate integer) ------------------------------------ */

constexpr int kConstBits = 13;
constexpr int kPass1Bits = 2;
constexpr int32_t FIX_0_298631336 = 2446;
constexpr int32_t FIX_0_390180644 = 3196;
constexpr int32_t FIX_0_541196100 = 4433;
constexpr int32_t FIX_0_765366865 = 6270;
constexpr int32_t FIX_0_899976223 = 7373;
constexpr int32_t FIX_1_175875602 = 9633;
constexpr int32_t FIX_1_501321110 = 12299;
constexpr int32_t FIX_1_847759065 = 15137;
constexpr int32_t FIX_1_961570560 = 16069;
constexpr int32_t FIX_2_053119869 = 16819;
constexpr int32_t FIX_2_562915447 = 20995;
constexpr int32_t FIX_3_072711026 = 25172;

inline int32_t descale(int32_t x, int n) { return (x + (int32_t{1} << (n - 1))) >> n; }

/** In-place forward DCT on 16-bit elements (results scaled up by 8). */
void fdct_islow(int16_t* data) {
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < 8; ++i) {
            const int step = pass == 0 ? 1 : 8;
            int16_t* d = pass == 0 ? data + i * 8 : data + i;
            const int32_t tmp0 = d[0 * step] + d[7 * step];
            int32_t tmp7 = d[0 * step] - d[7 * step];
            const int32_t tmp1 = d[1 * step] + d[6 * step];
            int32_t tmp6 = d[1 * step] - d[6 * step];
            const int32_t tmp2 = d[2 * step] + d[5 * step];
            int32_t tmp5 = d[2 * step] - d[5 * step];
            const int32_t tmp3 = d[3 * step] + d[4 * step];
            int32_t tmp4 = d[3 * step] - d[4 * step];

            const int32_t tmp10 = tmp0 + tmp3;
            const int32_t tmp13 = tmp0 - tmp3;
            const int32_t tmp11 = tmp1 + tmp2;
            const int32_t tmp12 = tmp1 - tmp2;

            const int fine = pass == 0 ? kConstBits - kPass1Bits : kConstBits + kPass1Bits;
            if (pass == 0) {
                d[0 * step] = static_cast<int16_t>((tmp10 + tmp11) * (1 << kPass1Bits));
                d[4 * step] = static_cast<int16_t>((tmp10 - tmp11) * (1 << kPass1Bits));
            } else {
                d[0 * step] = static_cast<int16_t>(descale(tmp10 + tmp11, kPass1Bits));
                d[4 * step] = static_cast<int16_t>(descale(tmp10 - tmp11, kPass1Bits));
            }
            int32_t z1 = (tmp12 + tmp13) * FIX_0_541196100;
            d[2 * step] = static_cast<int16_t>(descale(z1 + tmp13 * FIX_0_765366865, fine));
            d[6 * step] = static_cast<int16_t>(descale(z1 + tmp12 * (-FIX_1_847759065), fine));

            z1 = tmp4 + tmp7;
            int32_t z2 = tmp5 + tmp6;
            int32_t z3 = tmp4 + tmp6;
            int32_t z4 = tmp5 + tmp7;
            const int32_t z5 = (z3 + z4) * FIX_1_175875602;
            tmp4 = tmp4 * FIX_0_298631336;
            tmp5 = tmp5 * FIX_2_053119869;
            tmp6 = tmp6 * FIX_3_072711026;
            tmp7 = tmp7 * FIX_1_501321110;
            z1 = z1 * (-FIX_0_899976223);
            z2 = z2 * (-FIX_2_562915447);
            z3 = z3 * (-FIX_1_961570560);
            z4 = z4 * (-FIX_0_390180644);
            z3 += z5;
            z4 += z5;
            d[7 * step] = static_cast<int16_t>(descale(tmp4 + z1 + z3, fine));
            d[5 * step] = static_cast<int16_t>(descale(tmp5 + z2 + z4, fine));
            d[3 * step] = static_cast<int16_t>(descale(tmp6 + z2 + z3, fine));
            d[1 * step] = static_cast<int16_t>(descale(tmp7 + z1 + z4, fine));
        }
    }
}

/** Reciprocal quantizer for one coefficient position (divisor = q * 8). */
struct Divisor {
    uint32_t recip = 1;
    uint32_t corr = 0;
    int shift = 0;
};

Divisor make_divisor(unsigned divisor) {
    Divisor d;
    int b = 0;
    while ((divisor >> (b + 1)) != 0) ++b;  // floor(log2(divisor))
    int r = 16 + b;
    uint32_t fq = static_cast<uint32_t>((uint64_t{1} << r) / divisor);
    const uint32_t fr = static_cast<uint32_t>((uint64_t{1} << r) % divisor);
    uint32_t c = divisor / 2;
    if (fr == 0) {
        fq >>= 1;
        r--;
    } else if (fr <= divisor / 2u) {
        c++;
    } else {
        fq++;
    }
    d.recip = fq & 0xFFFFu;
    d.corr = c & 0xFFFFu;
    d.shift = r - 16;
    return d;
}

inline int16_t quantize(int16_t value, const Divisor& d) {
    const bool neg = value < 0;
    const uint32_t mag = static_cast<uint32_t>(neg ? -static_cast<int32_t>(value) : value) & 0xFFFFu;
    uint32_t product = ((mag + d.corr) & 0xFFFFFFFFu) * d.recip;
    product >>= d.shift + 16;
    const int16_t q = static_cast<int16_t>(product);
    return neg ? static_cast<int16_t>(-q) : q;
}

struct Plane {
    int width = 0;   // padded sample width (blocks * 8)
    int height = 0;  // padded sample height
    std::vector<uint8_t> samples;
    uint8_t at(int y, int x) const {
        return samples[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
    }
};

} // namespace

bool jpeg_encode(const uint8_t* pixels, int width, int height, int channels,
                 int quality, std::vector<uint8_t>* out) {
    if (!pixels || !out || width <= 0 || height <= 0 || width > 65500 || height > 65500 ||
        !(channels == 1 || channels == 3 || channels == 4)) {
        return false;
    }
    if (quality <= 0) quality = 1;
    if (quality > 100) quality = 100;
    const int scale = quality < 50 ? 5000 / quality : 200 - quality * 2;
    const bool color = channels != 1;
    const int ncomp = color ? 3 : 1;

    unsigned qtables[2][64];
    for (int t = 0; t < 2; ++t) {
        const unsigned* basic = t == 0 ? kStdLuminanceQuant : kStdChrominanceQuant;
        for (int i = 0; i < 64; ++i) {
            long temp = (static_cast<long>(basic[i]) * scale + 50L) / 100L;
            if (temp <= 0L) temp = 1L;
            if (temp > 32767L) temp = 32767L;
            if (temp > 255L) temp = 255L;  // baseline
            qtables[t][i] = static_cast<unsigned>(temp);
        }
    }
    Divisor divisors[2][64];
    for (int t = 0; t < 2; ++t) {
        for (int i = 0; i < 64; ++i) divisors[t][i] = make_divisor(qtables[t][i] << 3);
    }

    EncodeTable dc_tables[2];
    EncodeTable ac_tables[2];
    derive_encode_table(kDcLumBits, kDcLumVals, &dc_tables[0]);
    derive_encode_table(kAcLumBits, kAcLumVals, &ac_tables[0]);
    derive_encode_table(kDcChromBits, kDcChromVals, &dc_tables[1]);
    derive_encode_table(kAcChromBits, kAcChromVals, &ac_tables[1]);

    const int hmax = color ? 2 : 1;
    const int vmax = color ? 2 : 1;
    const int comp_h[3] = {hmax, 1, 1};
    const int comp_v[3] = {vmax, 1, 1};
    const int comp_tbl[3] = {0, 1, 1};
    const int W = width;
    const int H = height;
    const int mcu_rows = (H + vmax * 8 - 1) / (vmax * 8);
    const int mcus_per_row = (W + hmax * 8 - 1) / (hmax * 8);

    // Colour conversion into full-resolution planes.
    std::vector<uint8_t> full[3];
    for (int ci = 0; ci < ncomp; ++ci) full[ci].resize(static_cast<size_t>(W) * static_cast<size_t>(H));
    if (!color) {
        for (size_t i = 0; i < static_cast<size_t>(W) * static_cast<size_t>(H); ++i) full[0][i] = pixels[i];
    } else {
        constexpr int kScale = 16;
        constexpr int32_t kHalf = 1 << (kScale - 1);
        constexpr int32_t kOffset = 128 << kScale;
        auto fix = [](double x) { return static_cast<int32_t>(x * (1L << kScale) + 0.5); };
        int32_t tab[8][256];
        for (int i = 0; i < 256; ++i) {
            tab[0][i] = fix(0.29900) * i;
            tab[1][i] = fix(0.58700) * i;
            tab[2][i] = fix(0.11400) * i + kHalf;
            tab[3][i] = (-fix(0.16874)) * i;
            tab[4][i] = (-fix(0.33126)) * i;
            tab[5][i] = fix(0.50000) * i + kOffset + kHalf - 1;  // B=>Cb and R=>Cr
            tab[6][i] = (-fix(0.41869)) * i;
            tab[7][i] = (-fix(0.08131)) * i;
        }
        const size_t n = static_cast<size_t>(W) * static_cast<size_t>(H);
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* px = pixels + i * static_cast<size_t>(channels);
            const int r = px[0], g = px[1], b = px[2];
            full[0][i] = static_cast<uint8_t>((tab[0][r] + tab[1][g] + tab[2][b]) >> kScale);
            full[1][i] = static_cast<uint8_t>((tab[3][r] + tab[4][g] + tab[5][b]) >> kScale);
            full[2][i] = static_cast<uint8_t>((tab[5][r] + tab[6][g] + tab[7][b]) >> kScale);
        }
    }

    // Edge-extend and downsample each component into a block-padded plane.
    Plane planes[3];
    int width_in_blocks[3];
    int height_in_blocks[3];
    for (int ci = 0; ci < ncomp; ++ci) {
        width_in_blocks[ci] = (W * comp_h[ci] + hmax * 8 - 1) / (hmax * 8);
        height_in_blocks[ci] = (H * comp_v[ci] + vmax * 8 - 1) / (vmax * 8);
        Plane& p = planes[ci];
        p.width = width_in_blocks[ci] * 8;
        p.height = mcu_rows * comp_v[ci] * 8;
        p.samples.assign(static_cast<size_t>(p.width) * static_cast<size_t>(p.height), 0);
        // Rows produced from image data: one per input row (full-size), or one
        // per row pair after 2x2 averaging; the input is first padded to an
        // even row count and to the downsampler's width by replication.
        const int ratio_h = hmax / comp_h[ci];
        const int ratio_v = vmax / comp_v[ci];
        const int in_rows = (H + vmax - 1) / vmax * vmax;
        const int in_cols = p.width * ratio_h;
        auto src = [&](int y, int x) -> int {
            const int yy = y < H ? y : H - 1;
            const int xx = x < W ? x : W - 1;
            return full[ci][static_cast<size_t>(yy) * static_cast<size_t>(W) + static_cast<size_t>(xx)];
        };
        const int data_rows = in_rows / ratio_v;
        for (int y = 0; y < data_rows; ++y) {
            uint8_t* row = p.samples.data() + static_cast<size_t>(y) * static_cast<size_t>(p.width);
            if (ratio_h == 1 && ratio_v == 1) {
                for (int x = 0; x < p.width; ++x) row[x] = static_cast<uint8_t>(src(y, x));
            } else {
                int bias = 1;
                for (int x = 0; x < p.width; ++x) {
                    const int sx = x * 2;
                    const int sy = y * 2;
                    const int sum = src(sy, sx) + src(sy, sx + 1) + src(sy + 1, sx) + src(sy + 1, sx + 1);
                    row[x] = static_cast<uint8_t>((sum + bias) >> 2);
                    bias ^= 3;
                }
            }
        }
        (void)in_cols;
        // Pad to a whole number of MCU rows by repeating the last row.
        for (int y = data_rows; y < p.height; ++y) {
            std::memcpy(p.samples.data() + static_cast<size_t>(y) * static_cast<size_t>(p.width),
                        p.samples.data() + static_cast<size_t>(data_rows - 1) * static_cast<size_t>(p.width),
                        static_cast<size_t>(p.width));
        }
    }

    out->clear();
    Writer w(out);
    w.marker(0xD8);
    // JFIF APP0
    w.marker(0xE0);
    w.u16(16);
    w.byte('J');
    w.byte('F');
    w.byte('I');
    w.byte('F');
    w.byte(0);
    w.byte(1);
    w.byte(1);
    w.byte(0);
    w.u16(1);
    w.u16(1);
    w.byte(0);
    w.byte(0);
    // DQT
    for (int t = 0; t < (color ? 2 : 1); ++t) {
        w.marker(0xDB);
        w.u16(67);
        w.byte(static_cast<uint8_t>(t));
        for (int i = 0; i < 64; ++i) w.byte(static_cast<uint8_t>(qtables[t][kZigzag[i]]));
    }
    // SOF0
    w.marker(0xC0);
    w.u16(static_cast<unsigned>(8 + 3 * ncomp));
    w.byte(8);
    w.u16(static_cast<unsigned>(H));
    w.u16(static_cast<unsigned>(W));
    w.byte(static_cast<uint8_t>(ncomp));
    for (int ci = 0; ci < ncomp; ++ci) {
        w.byte(static_cast<uint8_t>(ci + 1));
        w.byte(static_cast<uint8_t>((comp_h[ci] << 4) | comp_v[ci]));
        w.byte(static_cast<uint8_t>(comp_tbl[ci]));
    }
    // DHT: DC then AC for each table, in component order.
    auto emit_dht = [&](int index, const uint8_t* bits, const uint8_t* vals) {
        int count = 0;
        for (int i = 1; i <= 16; ++i) count += bits[i];
        w.marker(0xC4);
        w.u16(static_cast<unsigned>(2 + 1 + 16 + count));
        w.byte(static_cast<uint8_t>(index));
        for (int i = 1; i <= 16; ++i) w.byte(bits[i]);
        for (int i = 0; i < count; ++i) w.byte(vals[i]);
    };
    emit_dht(0x00, kDcLumBits, kDcLumVals);
    emit_dht(0x10, kAcLumBits, kAcLumVals);
    if (color) {
        emit_dht(0x01, kDcChromBits, kDcChromVals);
        emit_dht(0x11, kAcChromBits, kAcChromVals);
    }
    // SOS
    w.marker(0xDA);
    w.u16(static_cast<unsigned>(6 + 2 * ncomp));
    w.byte(static_cast<uint8_t>(ncomp));
    for (int ci = 0; ci < ncomp; ++ci) {
        w.byte(static_cast<uint8_t>(ci + 1));
        w.byte(static_cast<uint8_t>((comp_tbl[ci] << 4) | comp_tbl[ci]));
    }
    w.byte(0);
    w.byte(63);
    w.byte(0);

    int last_dc[3] = {0, 0, 0};
    int16_t block[64];
    int16_t coefs[64];
    auto encode_block = [&](const int16_t* q, int ci) {
        const EncodeTable& dc = dc_tables[comp_tbl[ci]];
        const EncodeTable& ac = ac_tables[comp_tbl[ci]];
        int temp = q[0] - last_dc[ci];
        int temp2 = temp;
        last_dc[ci] = q[0];
        if (temp < 0) {
            temp = -temp;
            temp2--;
        }
        int nbits = 0;
        while (temp) {
            nbits++;
            temp >>= 1;
        }
        w.bits(dc.code[nbits], dc.size[nbits]);
        if (nbits) w.bits(static_cast<unsigned>(temp2), nbits);
        int r = 0;
        for (int k = 1; k < 64; ++k) {
            temp = q[kZigzag[k]];
            if (temp == 0) {
                r++;
                continue;
            }
            while (r > 15) {
                w.bits(ac.code[0xF0], ac.size[0xF0]);
                r -= 16;
            }
            temp2 = temp;
            if (temp < 0) {
                temp = -temp;
                temp2--;
            }
            nbits = 1;
            while ((temp >>= 1)) nbits++;
            const int i = (r << 4) + nbits;
            w.bits(ac.code[i], ac.size[i]);
            w.bits(static_cast<unsigned>(temp2), nbits);
            r = 0;
        }
        if (r > 0) w.bits(ac.code[0], ac.size[0]);
    };

    auto forward = [&](int ci, int by, int bx, int16_t* q) {
        const Plane& p = planes[ci];
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                block[y * 8 + x] = static_cast<int16_t>(p.at(by * 8 + y, bx * 8 + x) - 128);
            }
        }
        fdct_islow(block);
        const Divisor* d = divisors[comp_tbl[ci]];
        for (int i = 0; i < 64; ++i) q[i] = quantize(block[i], d[i]);
    };

    std::vector<int16_t> mcu_blocks(64 * 4);
    for (int my = 0; my < mcu_rows; ++my) {
        for (int mx = 0; mx < mcus_per_row; ++mx) {
            for (int ci = 0; ci < ncomp; ++ci) {
                const int mw = comp_h[ci];
                const int mh = comp_v[ci];
                const int last_col_width = width_in_blocks[ci] % mw == 0 ? mw : width_in_blocks[ci] % mw;
                const int last_row_height = height_in_blocks[ci] % mh == 0 ? mh : height_in_blocks[ci] % mh;
                const int blockcnt = mx < mcus_per_row - 1 ? mw : last_col_width;
                int blkn = 0;
                for (int yi = 0; yi < mh; ++yi) {
                    int16_t* rowblocks = mcu_blocks.data() + static_cast<size_t>(blkn) * 64;
                    if (my < mcu_rows - 1 || yi < last_row_height) {
                        for (int bi = 0; bi < blockcnt; ++bi) {
                            forward(ci, my * mh + yi, mx * mw + bi, rowblocks + bi * 64);
                        }
                        for (int bi = blockcnt; bi < mw; ++bi) {
                            std::memset(rowblocks + bi * 64, 0, 64 * sizeof(int16_t));
                            rowblocks[bi * 64] = rowblocks[(bi - 1) * 64];
                        }
                    } else {
                        const int16_t dc = mcu_blocks[static_cast<size_t>(blkn - 1) * 64];
                        for (int bi = 0; bi < mw; ++bi) {
                            std::memset(rowblocks + bi * 64, 0, 64 * sizeof(int16_t));
                            rowblocks[bi * 64] = dc;
                        }
                    }
                    blkn += mw;
                }
                for (int b = 0; b < mw * mh; ++b) {
                    std::memcpy(coefs, mcu_blocks.data() + static_cast<size_t>(b) * 64, sizeof(coefs));
                    encode_block(coefs, ci);
                }
            }
        }
    }
    w.flush_bits();
    w.marker(0xD9);
    return true;
}

} // namespace eshkol::image
