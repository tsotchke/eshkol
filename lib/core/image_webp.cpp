/**
 * @file image_webp.cpp
 * @brief Built-in WebP decoder (VP8 lossy, VP8L lossless, ALPH alpha, VP8X
 *        extended container) and lossless VP8L encoder.
 *
 * Decoding follows the reference decoder's arithmetic so that pixels match
 * what the system libwebp produced for image-read:
 *
 *   - RIFF parsing: simple VP8 / VP8L files and VP8X files (ICCP, EXIF and
 *     XMP are skipped; animation is rejected; the canvas must equal the
 *     frame);
 *   - VP8L: transforms (predictor, cross-colour, subtract-green, colour
 *     indexing), colour cache, meta prefix codes, LZ77 with the 2-D
 *     distance map;
 *   - VP8: boolean decoder, segment/filter/quantizer headers, coefficient
 *     probability updates, intra 16x16/4x4/chroma prediction, Walsh-Hadamard
 *     and inverse DCT, simple and normal loop filters;
 *   - ALPH: raw or VP8L-coded alpha plane with horizontal, vertical or
 *     gradient unfiltering;
 *   - YUV 4:2:0 to RGB with the "fancy" (bilinear) chroma upsampler and the
 *     14-bit scaled-integer conversion.
 *
 * Encoding writes a lossless VP8L bitstream (subtract-green and predictor
 * transforms, LZ77 runs from the left and upper pixel, canonical Huffman
 * codes limited to 15 bits), so image-write "webp" round-trips exactly.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "image_codecs.h"
#include "image_webp_tables.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <queue>

namespace eshkol::image {

namespace {

namespace wt = webp_tables;

uint32_t le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint32_t le24(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16);
}

/* ======================================================================== */
/* VP8L (lossless)                                                          */
/* ======================================================================== */

class LBitReader {
public:
    LBitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
    /** At least 56 upcoming bits, LSB first; zeros past the end. */
    uint64_t window() const {
        const size_t byte = pos_ >> 3;
        uint64_t w = 0;
        for (int i = 0; i < 8; ++i) {
            const size_t b = byte + static_cast<size_t>(i);
            if (b < size_) w |= static_cast<uint64_t>(data_[b]) << (8 * i);
        }
        return w >> (pos_ & 7);
    }
    uint32_t read(int bits) {
        if (bits == 0) return 0;
        const uint32_t v = static_cast<uint32_t>(window() & ((uint64_t{1} << bits) - 1));
        pos_ += static_cast<size_t>(bits);
        return v;
    }
    void skip(int bits) { pos_ += static_cast<size_t>(bits); }
    /** True once a bit beyond the end has been consumed. */
    bool eos() const { return pos_ > size_ * 8; }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
};

constexpr int kMaxCodeLength = 15;
constexpr int kFastBits = 10;

/** Canonical prefix code with the VP8L rules: a code with exactly one
 *  symbol takes zero bits; any other code must be complete. */
class PrefixCode {
public:
    bool build(const uint8_t* lengths, int n) {
        std::memset(counts_, 0, sizeof(counts_));
        single_ = -1;
        int nonzero = 0;
        int last = -1;
        for (int i = 0; i < n; ++i) {
            if (lengths[i] > kMaxCodeLength) return false;
            if (lengths[i]) {
                counts_[lengths[i]]++;
                ++nonzero;
                last = i;
            }
        }
        if (nonzero == 0) return false;
        if (nonzero == 1) {
            single_ = last;
            return true;
        }
        int left = 1;
        for (int len = 1; len <= kMaxCodeLength; ++len) {
            left <<= 1;
            left -= counts_[len];
            if (left < 0) return false;
        }
        if (left != 0) return false;
        int offsets[kMaxCodeLength + 2];
        offsets[1] = 0;
        for (int len = 1; len <= kMaxCodeLength; ++len) offsets[len + 1] = offsets[len] + counts_[len];
        symbols_.assign(static_cast<size_t>(nonzero), 0);
        for (int i = 0; i < n; ++i) {
            if (lengths[i]) symbols_[static_cast<size_t>(offsets[lengths[i]]++)] = static_cast<uint16_t>(i);
        }
        fast_.assign(1u << kFastBits, 0);
        int code = 0;
        int index = 0;
        for (int len = 1; len <= kFastBits; ++len) {
            for (int k = 0; k < counts_[len]; ++k) {
                int reversed = 0;
                for (int b = 0; b < len; ++b) reversed |= ((code >> b) & 1) << (len - 1 - b);
                const uint32_t entry = (static_cast<uint32_t>(symbols_[static_cast<size_t>(index)]) << 4) |
                                       static_cast<uint32_t>(len);
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

    int decode(LBitReader& br) const {
        if (single_ >= 0) return single_;
        const uint64_t w = br.window();
        const uint32_t e = fast_[static_cast<size_t>(w & ((1u << kFastBits) - 1))];
        if (e != 0) {
            br.skip(static_cast<int>(e & 15));
            return static_cast<int>(e >> 4);
        }
        int code = 0;
        int first = 0;
        int index = 0;
        for (int len = 1; len <= kMaxCodeLength; ++len) {
            code |= static_cast<int>((w >> (len - 1)) & 1);
            const int count = counts_[len];
            if (code - count < first) {
                br.skip(len);
                return symbols_[static_cast<size_t>(index + (code - first))];
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        br.skip(kMaxCodeLength);
        return 0;
    }

private:
    int counts_[kMaxCodeLength + 1] = {};
    int single_ = -1;
    std::vector<uint16_t> symbols_;
    std::vector<uint32_t> fast_;
};

constexpr int kNumLiteralCodes = 256;
constexpr int kNumLengthCodes = 24;
constexpr int kNumDistanceCodes = 40;

struct CodeGroup {
    PrefixCode codes[5];  // green, red, blue, alpha, distance
};

uint32_t div_round_up_shift(uint32_t size, int bits) { return (size + (1u << bits) - 1) >> bits; }

bool read_code_lengths(LBitReader& br, const uint8_t* cl_lengths, int num_symbols, uint8_t* lengths) {
    PrefixCode cl;
    if (!cl.build(cl_lengths, 19)) return false;
    int max_symbol = num_symbols;
    if (br.read(1)) {
        const int length_nbits = 2 + 2 * static_cast<int>(br.read(3));
        max_symbol = 2 + static_cast<int>(br.read(length_nbits));
        if (max_symbol > num_symbols) return false;
    }
    int prev = 8;
    int symbol = 0;
    while (symbol < num_symbols) {
        if (max_symbol-- == 0) break;
        const int code_len = cl.decode(br);
        if (code_len < 16) {
            lengths[symbol++] = static_cast<uint8_t>(code_len);
            if (code_len != 0) prev = code_len;
        } else {
            static const int kExtra[3] = {2, 3, 7};
            static const int kOffset[3] = {3, 3, 11};
            const int slot = code_len - 16;
            const int repeat = static_cast<int>(br.read(kExtra[slot])) + kOffset[slot];
            if (symbol + repeat > num_symbols) return false;
            const int value = code_len == 16 ? prev : 0;
            for (int k = 0; k < repeat; ++k) lengths[symbol++] = static_cast<uint8_t>(value);
        }
    }
    return true;
}

bool read_prefix_code(LBitReader& br, int alphabet, PrefixCode* out) {
    std::vector<uint8_t> lengths(static_cast<size_t>(std::max(alphabet, 256)), 0);
    if (br.read(1)) {
        const int num_symbols = static_cast<int>(br.read(1)) + 1;
        const int first_8bits = static_cast<int>(br.read(1));
        const int s0 = static_cast<int>(br.read(first_8bits ? 8 : 1));
        lengths[static_cast<size_t>(s0)] = 1;
        if (num_symbols == 2) lengths[br.read(8)] = 1;
    } else {
        uint8_t cl_lengths[19] = {0};
        const int num_codes = static_cast<int>(br.read(4)) + 4;
        for (int i = 0; i < num_codes; ++i) cl_lengths[wt::kCodeLengthCodeOrder[i]] = static_cast<uint8_t>(br.read(3));
        if (!read_code_lengths(br, cl_lengths, alphabet, lengths.data())) return false;
    }
    if (br.eos()) return false;
    return out->build(lengths.data(), alphabet);
}

inline uint32_t add_pixels(uint32_t a, uint32_t b) {
    const uint32_t ag = (a & 0xff00ff00u) + (b & 0xff00ff00u);
    const uint32_t rb = (a & 0x00ff00ffu) + (b & 0x00ff00ffu);
    return (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
}

inline uint32_t average2(uint32_t a, uint32_t b) { return (((a ^ b) & 0xfefefefeu) >> 1) + (a & b); }

inline int clip255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

inline uint32_t clamped_add_subtract_full(uint32_t c0, uint32_t c1, uint32_t c2) {
    uint32_t out = 0;
    for (int s = 0; s < 32; s += 8) {
        const int a = static_cast<int>((c0 >> s) & 0xff) + static_cast<int>((c1 >> s) & 0xff) -
                      static_cast<int>((c2 >> s) & 0xff);
        out |= static_cast<uint32_t>(clip255(a)) << s;
    }
    return out;
}

inline uint32_t clamped_add_subtract_half(uint32_t c0, uint32_t c1, uint32_t c2) {
    const uint32_t ave = average2(c0, c1);
    uint32_t out = 0;
    for (int s = 0; s < 32; s += 8) {
        const int a = static_cast<int>((ave >> s) & 0xff);
        const int b = static_cast<int>((c2 >> s) & 0xff);
        out |= static_cast<uint32_t>(clip255(a + (a - b) / 2)) << s;
    }
    return out;
}

inline int sub3(int a, int b, int c) {
    const int pb = b - c;
    const int pa = a - c;
    return std::abs(pb) - std::abs(pa);
}

inline uint32_t select_pred(uint32_t a, uint32_t b, uint32_t c) {
    const int pa_minus_pb = sub3(static_cast<int>(a >> 24), static_cast<int>(b >> 24), static_cast<int>(c >> 24)) +
                            sub3(static_cast<int>((a >> 16) & 0xff), static_cast<int>((b >> 16) & 0xff),
                                 static_cast<int>((c >> 16) & 0xff)) +
                            sub3(static_cast<int>((a >> 8) & 0xff), static_cast<int>((b >> 8) & 0xff),
                                 static_cast<int>((c >> 8) & 0xff)) +
                            sub3(static_cast<int>(a & 0xff), static_cast<int>(b & 0xff), static_cast<int>(c & 0xff));
    return pa_minus_pb <= 0 ? a : b;
}

/** Predictor for mode @p m given left L, top T, top-left TL, top-right TR. */
uint32_t predict(int m, uint32_t L, uint32_t T, uint32_t TL, uint32_t TR) {
    switch (m) {
        case 1: return L;
        case 2: return T;
        case 3: return TR;
        case 4: return TL;
        case 5: return average2(average2(L, TR), T);
        case 6: return average2(L, TL);
        case 7: return average2(L, T);
        case 8: return average2(TL, T);
        case 9: return average2(T, TR);
        case 10: return average2(average2(L, TL), average2(T, TR));
        case 11: return select_pred(T, L, TL);
        case 12: return clamped_add_subtract_full(L, T, TL);
        case 13: return clamped_add_subtract_half(L, T, TL);
        default: return 0xff000000u;
    }
}

inline int8_t as_i8(uint32_t v) { return static_cast<int8_t>(static_cast<uint8_t>(v)); }
inline int color_delta(int8_t t, int8_t c) { return (static_cast<int>(t) * static_cast<int>(c)) >> 5; }

struct Transform {
    int type = 0;
    int bits = 0;
    uint32_t xsize = 0;
    uint32_t ysize = 0;
    std::vector<uint32_t> data;
};

enum { kPredictor = 0, kCrossColor = 1, kSubtractGreen = 2, kColorIndexing = 3 };

bool decode_image_stream(LBitReader& br, uint32_t xsize, uint32_t ysize, bool level0, std::vector<uint32_t>* out);

bool decode_pixels(LBitReader& br, uint32_t width, uint32_t height, int cache_bits, const std::vector<CodeGroup>& groups,
                   const std::vector<uint32_t>& group_image, int group_bits, std::vector<uint32_t>* out) {
    const size_t total = static_cast<size_t>(width) * height;
    out->assign(total, 0);
    std::vector<uint32_t> cache(cache_bits ? (1u << cache_bits) : 0, 0);
    const uint32_t group_w = group_bits ? div_round_up_shift(width, group_bits) : 1;
    auto group_for = [&](uint32_t x, uint32_t y) -> const CodeGroup& {
        if (!group_bits) return groups[0];
        const uint32_t g = (group_image[(y >> group_bits) * group_w + (x >> group_bits)] >> 8) & 0xffffu;
        return groups[g];
    };
    auto insert = [&](uint32_t argb) {
        if (cache_bits) cache[(0x1e35a7bdu * argb) >> (32 - cache_bits)] = argb;
    };
    auto prefix_value = [&](int symbol) -> uint32_t {
        if (symbol < 4) return static_cast<uint32_t>(symbol) + 1;
        const int extra = (symbol - 2) >> 1;
        const uint32_t offset = static_cast<uint32_t>(2 + (symbol & 1)) << extra;
        return offset + br.read(extra) + 1;
    };
    size_t pos = 0;
    uint32_t* px = out->data();
    while (pos < total) {
        const uint32_t x = static_cast<uint32_t>(pos % width);
        const uint32_t y = static_cast<uint32_t>(pos / width);
        const CodeGroup& g = group_for(x, y);
        const int code = g.codes[0].decode(br);
        if (code < kNumLiteralCodes) {
            const uint32_t red = static_cast<uint32_t>(g.codes[1].decode(br));
            const uint32_t blue = static_cast<uint32_t>(g.codes[2].decode(br));
            const uint32_t alpha = static_cast<uint32_t>(g.codes[3].decode(br));
            if (br.eos()) return false;
            px[pos] = (alpha << 24) | (red << 16) | (static_cast<uint32_t>(code) << 8) | blue;
            insert(px[pos]);
            ++pos;
        } else if (code < kNumLiteralCodes + kNumLengthCodes) {
            const uint32_t length = prefix_value(code - kNumLiteralCodes);
            const int dist_symbol = g.codes[4].decode(br);
            const uint32_t dist_code = prefix_value(dist_symbol);
            size_t dist;
            if (dist_code > 120) {
                dist = dist_code - 120;
            } else {
                const int plane = wt::kCodeToPlane[dist_code - 1];
                const int yoff = plane >> 4;
                const int xoff = 8 - (plane & 0xf);
                const long d = static_cast<long>(yoff) * static_cast<long>(width) + xoff;
                dist = d >= 1 ? static_cast<size_t>(d) : 1;
            }
            if (br.eos()) return false;
            if (pos < dist || total - pos < length) return false;
            for (uint32_t k = 0; k < length; ++k) {
                px[pos] = px[pos - dist];
                insert(px[pos]);
                ++pos;
            }
        } else {
            const int key = code - kNumLiteralCodes - kNumLengthCodes;
            if (!cache_bits || key >= (1 << cache_bits)) return false;
            px[pos] = cache[static_cast<size_t>(key)];
            insert(px[pos]);
            ++pos;
        }
        if (br.eos()) return false;
    }
    return true;
}

void inverse_transform(const Transform& t, std::vector<uint32_t>* img) {
    const uint32_t w = t.xsize;
    const uint32_t h = t.ysize;
    std::vector<uint32_t>& p = *img;
    switch (t.type) {
        case kSubtractGreen:
            for (uint32_t& v : p) {
                const uint32_t green = (v >> 8) & 0xff;
                const uint32_t rb = (v & 0x00ff00ffu) + ((green << 16) | green);
                v = (v & 0xff00ff00u) | (rb & 0x00ff00ffu);
            }
            break;
        case kPredictor: {
            const uint32_t tiles_w = div_round_up_shift(w, t.bits);
            for (uint32_t y = 0; y < h; ++y) {
                uint32_t* row = p.data() + static_cast<size_t>(y) * w;
                const uint32_t* top = y ? row - w : nullptr;
                for (uint32_t x = 0; x < w; ++x) {
                    uint32_t pred;
                    if (y == 0) {
                        pred = x == 0 ? 0xff000000u : row[x - 1];
                    } else if (x == 0) {
                        pred = top[0];
                    } else {
                        const int mode = static_cast<int>(
                            (t.data[(y >> t.bits) * tiles_w + (x >> t.bits)] >> 8) & 0xf);
                        // The top-right of the last column is the first pixel of this row.
                        pred = predict(mode, row[x - 1], top[x], top[x - 1], x + 1 < w ? top[x + 1] : row[0]);
                    }
                    row[x] = add_pixels(row[x], pred);
                }
            }
            break;
        }
        case kCrossColor: {
            const uint32_t tiles_w = div_round_up_shift(w, t.bits);
            for (uint32_t y = 0; y < h; ++y) {
                for (uint32_t x = 0; x < w; ++x) {
                    const uint32_t m = t.data[(y >> t.bits) * tiles_w + (x >> t.bits)];
                    const int8_t g2r = as_i8(m);
                    const int8_t g2b = as_i8(m >> 8);
                    const int8_t r2b = as_i8(m >> 16);
                    uint32_t& v = p[static_cast<size_t>(y) * w + x];
                    const int8_t green = as_i8(v >> 8);
                    int new_red = static_cast<int>((v >> 16) & 0xff);
                    int new_blue = static_cast<int>(v & 0xff);
                    new_red += color_delta(g2r, green);
                    new_red &= 0xff;
                    new_blue += color_delta(g2b, green);
                    new_blue += color_delta(r2b, static_cast<int8_t>(new_red));
                    new_blue &= 0xff;
                    v = (v & 0xff00ff00u) | (static_cast<uint32_t>(new_red) << 16) | static_cast<uint32_t>(new_blue);
                }
            }
            break;
        }
        case kColorIndexing: {
            const int bits_per_pixel = 8 >> t.bits;
            const uint32_t packed_w = div_round_up_shift(w, t.bits);
            std::vector<uint32_t> outp(static_cast<size_t>(w) * h);
            if (bits_per_pixel < 8) {
                const uint32_t count_mask = (1u << t.bits) - 1;
                const uint32_t bit_mask = (1u << bits_per_pixel) - 1;
                for (uint32_t y = 0; y < h; ++y) {
                    const uint32_t* src = p.data() + static_cast<size_t>(y) * packed_w;
                    uint32_t packed = 0;
                    for (uint32_t x = 0; x < w; ++x) {
                        if ((x & count_mask) == 0) packed = (*src++ >> 8) & 0xff;
                        outp[static_cast<size_t>(y) * w + x] = t.data[packed & bit_mask];
                        packed >>= bits_per_pixel;
                    }
                }
            } else {
                for (size_t i = 0; i < outp.size(); ++i) outp[i] = t.data[(p[i] >> 8) & 0xff];
            }
            p.swap(outp);
            break;
        }
        default:
            break;
    }
}

bool decode_image_stream(LBitReader& br, uint32_t xsize, uint32_t ysize, bool level0, std::vector<uint32_t>* out) {
    uint32_t coded_w = xsize;
    std::vector<Transform> transforms;
    if (level0) {
        unsigned seen = 0;
        while (br.read(1)) {
            Transform t;
            t.type = static_cast<int>(br.read(2));
            if (seen & (1u << t.type)) return false;
            seen |= 1u << t.type;
            t.xsize = coded_w;
            t.ysize = ysize;
            if (t.type == kPredictor || t.type == kCrossColor) {
                t.bits = static_cast<int>(br.read(3)) + 2;
                if (!decode_image_stream(br, div_round_up_shift(t.xsize, t.bits), div_round_up_shift(ysize, t.bits),
                                         false, &t.data)) {
                    return false;
                }
            } else if (t.type == kColorIndexing) {
                const int num_colors = static_cast<int>(br.read(8)) + 1;
                t.bits = num_colors > 16 ? 0 : (num_colors > 4 ? 1 : (num_colors > 2 ? 2 : 3));
                coded_w = div_round_up_shift(t.xsize, t.bits);
                std::vector<uint32_t> table;
                if (!decode_image_stream(br, static_cast<uint32_t>(num_colors), 1, false, &table)) return false;
                const size_t final_colors = size_t{1} << (8 >> t.bits);
                t.data.assign(final_colors, 0);
                t.data[0] = table[0];
                for (int i = 1; i < num_colors; ++i) t.data[static_cast<size_t>(i)] = add_pixels(table[static_cast<size_t>(i)], t.data[static_cast<size_t>(i - 1)]);
            }
            transforms.push_back(std::move(t));
        }
    }
    int cache_bits = 0;
    if (br.read(1)) {
        cache_bits = static_cast<int>(br.read(4));
        if (cache_bits < 1 || cache_bits > 11) return false;
    }
    int group_bits = 0;
    std::vector<uint32_t> group_image;
    size_t num_groups = 1;
    if (level0 && br.read(1)) {
        group_bits = static_cast<int>(br.read(3)) + 2;
        if (!decode_image_stream(br, div_round_up_shift(coded_w, group_bits), div_round_up_shift(ysize, group_bits),
                                 false, &group_image)) {
            return false;
        }
        uint32_t max_group = 0;
        for (uint32_t v : group_image) max_group = std::max(max_group, (v >> 8) & 0xffffu);
        num_groups = static_cast<size_t>(max_group) + 1;
    }
    if (br.eos()) return false;
    std::vector<CodeGroup> groups(num_groups);
    const int green_alphabet = kNumLiteralCodes + kNumLengthCodes + (cache_bits ? (1 << cache_bits) : 0);
    for (CodeGroup& g : groups) {
        const int alphabets[5] = {green_alphabet, 256, 256, 256, kNumDistanceCodes};
        for (int k = 0; k < 5; ++k) {
            if (!read_prefix_code(br, alphabets[k], &g.codes[k])) return false;
        }
    }
    std::vector<uint32_t> pixels;
    if (!decode_pixels(br, coded_w, ysize, cache_bits, groups, group_image, group_bits, &pixels)) return false;
    for (size_t i = transforms.size(); i-- > 0;) inverse_transform(transforms[i], &pixels);
    out->swap(pixels);
    return true;
}

/** Decode a VP8L chunk: header + image stream. */
bool vp8l_decode(const uint8_t* data, size_t size, uint32_t* w, uint32_t* h, bool* alpha_used, std::vector<uint32_t>* argb) {
    if (size < 5 || data[0] != 0x2f) return false;
    LBitReader br(data + 1, size - 1);
    *w = br.read(14) + 1;
    *h = br.read(14) + 1;
    *alpha_used = br.read(1) != 0;
    if (br.read(3) != 0) return false;
    return decode_image_stream(br, *w, *h, true, argb) && !br.eos();
}

/* ======================================================================== */
/* VP8 (lossy)                                                              */
/* ======================================================================== */

class BoolDecoder {
public:
    void init(const uint8_t* data, size_t size) {
        p_ = data;
        end_ = data + size;
        value_ = 0;
        bits_ = -8;
        range_ = 255 - 1;
        eof_ = false;
        load();
    }
    int get(int prob) {
        if (bits_ < 0) load();
        uint32_t range = range_;
        const int pos = bits_;
        const uint32_t split = (range * static_cast<uint32_t>(prob)) >> 8;
        const uint32_t value = static_cast<uint32_t>(value_ >> pos);
        int bit;
        if (value > split) {
            range -= split;
            value_ -= static_cast<uint64_t>(split + 1) << pos;
            bit = 1;
        } else {
            range = split + 1;
            bit = 0;
        }
        int log2 = 0;
        while ((range >> (log2 + 1)) != 0) ++log2;
        const int shift = 7 ^ log2;
        range <<= shift;
        bits_ -= shift;
        range_ = range - 1;
        return bit;
    }
    uint32_t value(int bits) {
        uint32_t v = 0;
        while (bits-- > 0) v |= static_cast<uint32_t>(get(0x80)) << bits;
        return v;
    }
    int signed_value(int bits) {
        const int v = static_cast<int>(value(bits));
        return get(0x80) ? -v : v;
    }
    bool eof() const { return eof_; }

private:
    void load() {
        if (p_ < end_) {
            value_ = (value_ << 8) | *p_++;
            bits_ += 8;
        } else if (!eof_) {
            value_ <<= 8;
            bits_ += 8;
            eof_ = true;
        } else {
            bits_ = 0;
        }
    }
    const uint8_t* p_ = nullptr;
    const uint8_t* end_ = nullptr;
    uint64_t value_ = 0;
    int bits_ = 0;
    uint32_t range_ = 254;
    bool eof_ = false;
};

enum { B_DC_PRED = 0, B_TM_PRED, B_VE_PRED, B_HE_PRED, B_RD_PRED, B_VR_PRED, B_LD_PRED, B_VL_PRED, B_HD_PRED, B_HU_PRED };
enum { DC_PRED = B_DC_PRED, V_PRED = B_VE_PRED, H_PRED = B_HE_PRED, TM_PRED = B_TM_PRED };

struct Quant {
    int y1[2], y2[2], uv[2];
};

struct FilterInfo {
    int limit = 0;
    int ilevel = 0;
    int hev = 0;
    bool inner = false;
};

struct MbInfo {
    bool i4x4 = false;
    int segment = 0;
    bool skip = false;
    uint8_t imodes[16] = {0};
    int uvmode = 0;
    int16_t coeffs[384];
    bool zero = true;  // no non-zero residual
};

inline uint8_t clip8(int v) { return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v)); }
inline int sclip1(int v) { return v < -128 ? -128 : (v > 127 ? 127 : v); }
inline int sclip2(int v) { return v < -16 ? -16 : (v > 15 ? 15 : v); }

constexpr int BPS = 32;

#define AVG3(a, b, c) static_cast<uint8_t>(((a) + 2 * (b) + (c) + 2) >> 2)
#define AVG2(a, b) static_cast<uint8_t>(((a) + (b) + 1) >> 1)
#define DST(x, y) dst[(x) + (y) * BPS]

void true_motion(uint8_t* dst, int size) {
    const uint8_t* top = dst - BPS;
    const int tl = top[-1];
    for (int y = 0; y < size; ++y) {
        const int l = dst[-1 + y * BPS];
        for (int x = 0; x < size; ++x) dst[x + y * BPS] = clip8(top[x] + l - tl);
    }
}

void predict4(int mode, uint8_t* dst) {
    const uint8_t* top = dst - BPS;
    const int X = dst[-1 - BPS];
    const int A = top[0], B = top[1], C = top[2], D = top[3], E = top[4], F = top[5], G = top[6], H = top[7];
    const int I = dst[-1], J = dst[-1 + BPS], K = dst[-1 + 2 * BPS], L = dst[-1 + 3 * BPS];
    switch (mode) {
        case B_DC_PRED: {
            unsigned dc = 4;
            for (int i = 0; i < 4; ++i) dc += dst[i - BPS] + dst[-1 + i * BPS];
            dc >>= 3;
            for (int y = 0; y < 4; ++y) std::memset(dst + y * BPS, static_cast<int>(dc), 4);
            break;
        }
        case B_TM_PRED: true_motion(dst, 4); break;
        case B_VE_PRED: {
            const uint8_t vals[4] = {AVG3(X, A, B), AVG3(A, B, C), AVG3(B, C, D), AVG3(C, D, E)};
            for (int y = 0; y < 4; ++y) std::memcpy(dst + y * BPS, vals, 4);
            break;
        }
        case B_HE_PRED: {
            const uint8_t r0 = AVG3(X, I, J), r1 = AVG3(I, J, K), r2 = AVG3(J, K, L), r3 = AVG3(K, L, L);
            std::memset(dst, r0, 4);
            std::memset(dst + BPS, r1, 4);
            std::memset(dst + 2 * BPS, r2, 4);
            std::memset(dst + 3 * BPS, r3, 4);
            break;
        }
        case B_RD_PRED:
            DST(0, 3) = AVG3(J, K, L);
            DST(1, 3) = DST(0, 2) = AVG3(I, J, K);
            DST(2, 3) = DST(1, 2) = DST(0, 1) = AVG3(X, I, J);
            DST(3, 3) = DST(2, 2) = DST(1, 1) = DST(0, 0) = AVG3(A, X, I);
            DST(3, 2) = DST(2, 1) = DST(1, 0) = AVG3(B, A, X);
            DST(3, 1) = DST(2, 0) = AVG3(C, B, A);
            DST(3, 0) = AVG3(D, C, B);
            break;
        case B_LD_PRED:
            DST(0, 0) = AVG3(A, B, C);
            DST(1, 0) = DST(0, 1) = AVG3(B, C, D);
            DST(2, 0) = DST(1, 1) = DST(0, 2) = AVG3(C, D, E);
            DST(3, 0) = DST(2, 1) = DST(1, 2) = DST(0, 3) = AVG3(D, E, F);
            DST(3, 1) = DST(2, 2) = DST(1, 3) = AVG3(E, F, G);
            DST(3, 2) = DST(2, 3) = AVG3(F, G, H);
            DST(3, 3) = AVG3(G, H, H);
            break;
        case B_VR_PRED:
            DST(0, 0) = DST(1, 2) = AVG2(X, A);
            DST(1, 0) = DST(2, 2) = AVG2(A, B);
            DST(2, 0) = DST(3, 2) = AVG2(B, C);
            DST(3, 0) = AVG2(C, D);
            DST(0, 3) = AVG3(K, J, I);
            DST(0, 2) = AVG3(J, I, X);
            DST(0, 1) = DST(1, 3) = AVG3(I, X, A);
            DST(1, 1) = DST(2, 3) = AVG3(X, A, B);
            DST(2, 1) = DST(3, 3) = AVG3(A, B, C);
            DST(3, 1) = AVG3(B, C, D);
            break;
        case B_VL_PRED:
            DST(0, 0) = AVG2(A, B);
            DST(1, 0) = DST(0, 2) = AVG2(B, C);
            DST(2, 0) = DST(1, 2) = AVG2(C, D);
            DST(3, 0) = DST(2, 2) = AVG2(D, E);
            DST(0, 1) = AVG3(A, B, C);
            DST(1, 1) = DST(0, 3) = AVG3(B, C, D);
            DST(2, 1) = DST(1, 3) = AVG3(C, D, E);
            DST(3, 1) = DST(2, 3) = AVG3(D, E, F);
            DST(3, 2) = AVG3(E, F, G);
            DST(3, 3) = AVG3(F, G, H);
            break;
        case B_HU_PRED:
            DST(0, 0) = AVG2(I, J);
            DST(2, 0) = DST(0, 1) = AVG2(J, K);
            DST(2, 1) = DST(0, 2) = AVG2(K, L);
            DST(1, 0) = AVG3(I, J, K);
            DST(3, 0) = DST(1, 1) = AVG3(J, K, L);
            DST(3, 1) = DST(1, 2) = AVG3(K, L, L);
            DST(3, 2) = DST(2, 2) = DST(0, 3) = DST(1, 3) = DST(2, 3) = DST(3, 3) = static_cast<uint8_t>(L);
            break;
        case B_HD_PRED:
            DST(0, 0) = DST(2, 1) = AVG2(I, X);
            DST(0, 1) = DST(2, 2) = AVG2(J, I);
            DST(0, 2) = DST(2, 3) = AVG2(K, J);
            DST(0, 3) = AVG2(L, K);
            DST(3, 0) = AVG3(A, B, C);
            DST(2, 0) = AVG3(X, A, B);
            DST(1, 0) = DST(3, 1) = AVG3(I, X, A);
            DST(1, 1) = DST(3, 2) = AVG3(J, I, X);
            DST(1, 2) = DST(3, 3) = AVG3(K, J, I);
            DST(1, 3) = AVG3(L, K, J);
            break;
        default:
            break;
    }
}

/** 16x16 luma (size 16) or 8x8 chroma (size 8) prediction; DC variants
 *  depend on which edges exist. */
void predict_block(int mode, uint8_t* dst, int size, bool has_top, bool has_left) {
    const int shift = size == 16 ? 4 : 3;
    switch (mode) {
        case DC_PRED: {
            int dc;
            if (has_top && has_left) {
                dc = size;
                for (int i = 0; i < size; ++i) dc += dst[i - BPS] + dst[-1 + i * BPS];
                dc >>= shift + 1;
            } else if (has_left) {
                dc = size / 2;
                for (int i = 0; i < size; ++i) dc += dst[-1 + i * BPS];
                dc >>= shift;
            } else if (has_top) {
                dc = size / 2;
                for (int i = 0; i < size; ++i) dc += dst[i - BPS];
                dc >>= shift;
            } else {
                dc = 0x80;
            }
            for (int y = 0; y < size; ++y) std::memset(dst + y * BPS, dc, static_cast<size_t>(size));
            break;
        }
        case V_PRED:
            for (int y = 0; y < size; ++y) std::memcpy(dst + y * BPS, dst - BPS, static_cast<size_t>(size));
            break;
        case H_PRED:
            for (int y = 0; y < size; ++y) std::memset(dst + y * BPS, dst[-1 + y * BPS], static_cast<size_t>(size));
            break;
        case TM_PRED:
            true_motion(dst, size);
            break;
        default:
            break;
    }
}

inline int mul1(int a) { return ((a * 20091) >> 16) + a; }
inline int mul2(int a) { return (a * 35468) >> 16; }

void inverse_dct_add(const int16_t* in, uint8_t* dst) {
    int C[16];
    int* tmp = C;
    for (int i = 0; i < 4; ++i) {
        const int a = in[0] + in[8];
        const int b = in[0] - in[8];
        const int c = mul2(in[4]) - mul1(in[12]);
        const int d = mul1(in[4]) + mul2(in[12]);
        tmp[0] = a + d;
        tmp[1] = b + c;
        tmp[2] = b - c;
        tmp[3] = a - d;
        tmp += 4;
        ++in;
    }
    tmp = C;
    for (int i = 0; i < 4; ++i) {
        const int dc = tmp[0] + 4;
        const int a = dc + tmp[8];
        const int b = dc - tmp[8];
        const int c = mul2(tmp[4]) - mul1(tmp[12]);
        const int d = mul1(tmp[4]) + mul2(tmp[12]);
        dst[0] = clip8(dst[0] + ((a + d) >> 3));
        dst[1] = clip8(dst[1] + ((b + c) >> 3));
        dst[2] = clip8(dst[2] + ((b - c) >> 3));
        dst[3] = clip8(dst[3] + ((a - d) >> 3));
        ++tmp;
        dst += BPS;
    }
}

void inverse_wht(const int16_t* in, int16_t* out) {
    int tmp[16];
    for (int i = 0; i < 4; ++i) {
        const int a0 = in[0 + i] + in[12 + i];
        const int a1 = in[4 + i] + in[8 + i];
        const int a2 = in[4 + i] - in[8 + i];
        const int a3 = in[0 + i] - in[12 + i];
        tmp[0 + i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }
    for (int i = 0; i < 4; ++i) {
        const int dc = tmp[0 + i * 4] + 3;
        const int a0 = dc + tmp[3 + i * 4];
        const int a1 = tmp[1 + i * 4] + tmp[2 + i * 4];
        const int a2 = tmp[1 + i * 4] - tmp[2 + i * 4];
        const int a3 = dc - tmp[3 + i * 4];
        out[0] = static_cast<int16_t>((a0 + a1) >> 3);
        out[16] = static_cast<int16_t>((a3 + a2) >> 3);
        out[32] = static_cast<int16_t>((a0 - a1) >> 3);
        out[48] = static_cast<int16_t>((a3 - a2) >> 3);
        out += 64;
    }
}

/* ---- loop filter ------------------------------------------------------- */

inline void filter2(uint8_t* p, int step) {
    const int p1 = p[-2 * step], p0 = p[-step], q0 = p[0], q1 = p[step];
    const int a = 3 * (q0 - p0) + sclip1(p1 - q1);
    const int a1 = sclip2((a + 4) >> 3);
    const int a2 = sclip2((a + 3) >> 3);
    p[-step] = clip8(p0 + a2);
    p[0] = clip8(q0 - a1);
}

inline void filter4(uint8_t* p, int step) {
    const int p1 = p[-2 * step], p0 = p[-step], q0 = p[0], q1 = p[step];
    const int a = 3 * (q0 - p0);
    const int a1 = sclip2((a + 4) >> 3);
    const int a2 = sclip2((a + 3) >> 3);
    const int a3 = (a1 + 1) >> 1;
    p[-2 * step] = clip8(p1 + a3);
    p[-step] = clip8(p0 + a2);
    p[0] = clip8(q0 - a1);
    p[step] = clip8(q1 - a3);
}

inline void filter6(uint8_t* p, int step) {
    const int p2 = p[-3 * step], p1 = p[-2 * step], p0 = p[-step];
    const int q0 = p[0], q1 = p[step], q2 = p[2 * step];
    const int a = sclip1(3 * (q0 - p0) + sclip1(p1 - q1));
    const int a1 = (27 * a + 63) >> 7;
    const int a2 = (18 * a + 63) >> 7;
    const int a3 = (9 * a + 63) >> 7;
    p[-3 * step] = clip8(p2 + a3);
    p[-2 * step] = clip8(p1 + a2);
    p[-step] = clip8(p0 + a1);
    p[0] = clip8(q0 - a1);
    p[step] = clip8(q1 - a2);
    p[2 * step] = clip8(q2 - a3);
}

inline bool hev(const uint8_t* p, int step, int thresh) {
    const int p1 = p[-2 * step], p0 = p[-step], q0 = p[0], q1 = p[step];
    return std::abs(p1 - p0) > thresh || std::abs(q1 - q0) > thresh;
}

inline bool needs_filter(const uint8_t* p, int step, int t) {
    const int p1 = p[-2 * step], p0 = p[-step], q0 = p[0], q1 = p[step];
    return 4 * std::abs(p0 - q0) + std::abs(p1 - q1) <= t;
}

inline bool needs_filter2(const uint8_t* p, int step, int t, int it) {
    const int p3 = p[-4 * step], p2 = p[-3 * step], p1 = p[-2 * step], p0 = p[-step];
    const int q0 = p[0], q1 = p[step], q2 = p[2 * step], q3 = p[3 * step];
    if (4 * std::abs(p0 - q0) + std::abs(p1 - q1) > t) return false;
    return std::abs(p3 - p2) <= it && std::abs(p2 - p1) <= it && std::abs(p1 - p0) <= it &&
           std::abs(q3 - q2) <= it && std::abs(q2 - q1) <= it && std::abs(q1 - q0) <= it;
}

void simple_filter(uint8_t* p, int step, int stride, int count, int thresh) {
    const int t = 2 * thresh + 1;
    for (int i = 0; i < count; ++i) {
        if (needs_filter(p + i * stride, step, t)) filter2(p + i * stride, step);
    }
}

void normal_filter(uint8_t* p, int hstride, int vstride, int size, int thresh, int ithresh, int hev_t, bool mb_edge) {
    const int t = 2 * thresh + 1;
    while (size-- > 0) {
        if (needs_filter2(p, hstride, t, ithresh)) {
            if (hev(p, hstride, hev_t)) {
                filter2(p, hstride);
            } else if (mb_edge) {
                filter6(p, hstride);
            } else {
                filter4(p, hstride);
            }
        }
        p += vstride;
    }
}

class Vp8Decoder {
public:
    bool decode(const uint8_t* data, size_t size, uint32_t* out_w, uint32_t* out_h, std::vector<uint8_t>* y,
                std::vector<uint8_t>* u, std::vector<uint8_t>* v, int* ystride, int* uvstride);

private:
    BoolDecoder br_;
    std::vector<BoolDecoder> parts_;
    int mb_w_ = 0, mb_h_ = 0;
    bool use_segment_ = false, update_map_ = false, absolute_delta_ = true;
    int quantizer_[4] = {0, 0, 0, 0};
    int filter_strength_[4] = {0, 0, 0, 0};
    uint8_t segment_proba_[3] = {255, 255, 255};
    bool simple_ = false;
    int level_ = 0, sharpness_ = 0;
    bool use_lf_delta_ = false;
    int ref_lf_delta_[4] = {0, 0, 0, 0};
    int mode_lf_delta_[4] = {0, 0, 0, 0};
    int filter_type_ = 0;
    Quant dqm_[4];
    uint8_t proba_[4][8][3][11];
    bool use_skip_proba_ = false;
    int skip_p_ = 0;
    FilterInfo fstrengths_[4][2];

    int get_coeffs(BoolDecoder& br, int type, int ctx, const int* dq, int n, int16_t* out);
};

int Vp8Decoder::get_coeffs(BoolDecoder& br, int type, int ctx, const int* dq, int n, int16_t* out) {
    const uint8_t* p = proba_[type][wt::kBands[n]][ctx];
    for (; n < 16; ++n) {
        if (!br.get(p[0])) return n;
        while (!br.get(p[1])) {
            p = proba_[type][wt::kBands[++n]][0];
            if (n == 16) return 16;
        }
        int v;
        const int next_band = wt::kBands[n + 1];
        if (!br.get(p[2])) {
            v = 1;
            p = proba_[type][next_band][1];
        } else {
            if (!br.get(p[3])) {
                if (!br.get(p[4])) {
                    v = 2;
                } else {
                    v = 3 + br.get(p[5]);
                }
            } else {
                if (!br.get(p[6])) {
                    if (!br.get(p[7])) {
                        v = 5 + br.get(159);
                    } else {
                        v = 7 + 2 * br.get(165);
                        v += br.get(145);
                    }
                } else {
                    const int bit1 = br.get(p[8]);
                    const int bit0 = br.get(p[9 + bit1]);
                    const int cat = 2 * bit1 + bit0;
                    static const uint8_t* const kCat3456[4] = {wt::kCat3, wt::kCat4, wt::kCat5, wt::kCat6};
                    v = 0;
                    for (const uint8_t* tab = kCat3456[cat]; *tab; ++tab) v += v + br.get(*tab);
                    v += 3 + (8 << cat);
                }
            }
            p = proba_[type][next_band][2];
        }
        const int signed_v = br.get(0x80) ? -v : v;
        out[wt::kZigzag[n]] = static_cast<int16_t>(signed_v * dq[n > 0]);
    }
    return 16;
}

bool Vp8Decoder::decode(const uint8_t* data, size_t size, uint32_t* out_w, uint32_t* out_h, std::vector<uint8_t>* yp,
                        std::vector<uint8_t>* up, std::vector<uint8_t>* vp, int* ystride_out, int* uvstride_out) {
    if (size < 10) return false;
    const uint32_t bits = le24(data);
    const bool key_frame = !(bits & 1);
    const int profile = static_cast<int>((bits >> 1) & 7);
    const bool show = ((bits >> 4) & 1) != 0;
    const uint32_t partition_length = bits >> 5;
    if (!key_frame || profile > 3 || !show) return false;
    if (data[3] != 0x9d || data[4] != 0x01 || data[5] != 0x2a) return false;
    const uint32_t width = (data[6] | (data[7] << 8)) & 0x3fff;
    const uint32_t height = (data[8] | (data[9] << 8)) & 0x3fff;
    if (width == 0 || height == 0) return false;
    if (partition_length >= size) return false;
    const uint8_t* buf = data + 10;
    size_t buf_size = size - 10;
    if (partition_length > buf_size) return false;
    mb_w_ = static_cast<int>((width + 15) >> 4);
    mb_h_ = static_cast<int>((height + 15) >> 4);

    br_.init(buf, partition_length);
    buf += partition_length;
    buf_size -= partition_length;
    br_.get(0x80);  // colour space
    br_.get(0x80);  // clamping type
    // Segment header
    use_segment_ = br_.value(1) != 0;
    if (use_segment_) {
        update_map_ = br_.value(1) != 0;
        if (br_.value(1)) {
            absolute_delta_ = br_.value(1) != 0;
            for (int s = 0; s < 4; ++s) quantizer_[s] = br_.value(1) ? br_.signed_value(7) : 0;
            for (int s = 0; s < 4; ++s) filter_strength_[s] = br_.value(1) ? br_.signed_value(6) : 0;
        }
        if (update_map_) {
            for (int s = 0; s < 3; ++s) segment_proba_[s] = br_.value(1) ? static_cast<uint8_t>(br_.value(8)) : 255;
        }
    } else {
        update_map_ = false;
    }
    if (br_.eof()) return false;
    // Filter header
    simple_ = br_.value(1) != 0;
    level_ = static_cast<int>(br_.value(6));
    sharpness_ = static_cast<int>(br_.value(3));
    use_lf_delta_ = br_.value(1) != 0;
    if (use_lf_delta_) {
        if (br_.value(1)) {
            for (int i = 0; i < 4; ++i) {
                if (br_.value(1)) ref_lf_delta_[i] = br_.signed_value(6);
            }
            for (int i = 0; i < 4; ++i) {
                if (br_.value(1)) mode_lf_delta_[i] = br_.signed_value(6);
            }
        }
    }
    filter_type_ = level_ == 0 ? 0 : (simple_ ? 1 : 2);
    if (br_.eof()) return false;
    // Partitions
    const int num_parts = 1 << br_.value(2);
    const size_t last = static_cast<size_t>(num_parts - 1);
    if (buf_size < 3 * last) return false;
    parts_.assign(static_cast<size_t>(num_parts), BoolDecoder{});
    {
        const uint8_t* sz = buf;
        const uint8_t* part_start = buf + last * 3;
        size_t size_left = buf_size - last * 3;
        for (size_t p = 0; p < last; ++p) {
            size_t psize = le24(sz);
            if (psize > size_left) psize = size_left;
            parts_[p].init(part_start, psize);
            part_start += psize;
            size_left -= psize;
            sz += 3;
        }
        parts_[last].init(part_start, size_left);
        if (part_start >= buf + buf_size) return false;
    }
    // Quantizer
    {
        const int base_q0 = static_cast<int>(br_.value(7));
        const int dqy1_dc = br_.value(1) ? br_.signed_value(4) : 0;
        const int dqy2_dc = br_.value(1) ? br_.signed_value(4) : 0;
        const int dqy2_ac = br_.value(1) ? br_.signed_value(4) : 0;
        const int dquv_dc = br_.value(1) ? br_.signed_value(4) : 0;
        const int dquv_ac = br_.value(1) ? br_.signed_value(4) : 0;
        auto clip = [](int v, int m) { return v < 0 ? 0 : (v > m ? m : v); };
        for (int i = 0; i < 4; ++i) {
            int q;
            if (use_segment_) {
                q = quantizer_[i];
                if (!absolute_delta_) q += base_q0;
            } else {
                if (i > 0) {
                    dqm_[i] = dqm_[0];
                    continue;
                }
                q = base_q0;
            }
            Quant& m = dqm_[i];
            m.y1[0] = wt::kDcTable[clip(q + dqy1_dc, 127)];
            m.y1[1] = wt::kAcTable[clip(q, 127)];
            m.y2[0] = wt::kDcTable[clip(q + dqy2_dc, 127)] * 2;
            m.y2[1] = (wt::kAcTable[clip(q + dqy2_ac, 127)] * 101581) >> 16;
            if (m.y2[1] < 8) m.y2[1] = 8;
            m.uv[0] = wt::kDcTable[clip(q + dquv_dc, 117)];
            m.uv[1] = wt::kAcTable[clip(q + dquv_ac, 127)];
        }
    }
    br_.value(1);  // update_proba: ignored for key frames
    // Coefficient probabilities
    for (int t = 0; t < 4; ++t) {
        for (int b = 0; b < 8; ++b) {
            for (int c = 0; c < 3; ++c) {
                for (int p = 0; p < 11; ++p) {
                    const size_t idx = ((static_cast<size_t>(t) * 8 + b) * 3 + c) * 11 + p;
                    proba_[t][b][c][p] = br_.get(wt::kCoeffsUpdateProba[idx]) ? static_cast<uint8_t>(br_.value(8))
                                                                               : wt::kCoeffsProba0[idx];
                }
            }
        }
    }
    use_skip_proba_ = br_.value(1) != 0;
    if (use_skip_proba_) skip_p_ = static_cast<int>(br_.value(8));
    if (br_.eof()) return false;

    // Filter strengths
    if (filter_type_ > 0) {
        for (int s = 0; s < 4; ++s) {
            int base_level;
            if (use_segment_) {
                base_level = filter_strength_[s];
                if (!absolute_delta_) base_level += level_;
            } else {
                base_level = level_;
            }
            for (int i4 = 0; i4 <= 1; ++i4) {
                FilterInfo& info = fstrengths_[s][i4];
                int level = base_level;
                if (use_lf_delta_) {
                    level += ref_lf_delta_[0];
                    if (i4) level += mode_lf_delta_[0];
                }
                level = level < 0 ? 0 : (level > 63 ? 63 : level);
                if (level > 0) {
                    int ilevel = level;
                    if (sharpness_ > 0) {
                        if (sharpness_ > 4) {
                            ilevel >>= 2;
                        } else {
                            ilevel >>= 1;
                        }
                        if (ilevel > 9 - sharpness_) ilevel = 9 - sharpness_;
                    }
                    if (ilevel < 1) ilevel = 1;
                    info.ilevel = ilevel;
                    info.limit = 2 * level + ilevel;
                    info.hev = level >= 40 ? 2 : (level >= 15 ? 1 : 0);
                } else {
                    info.limit = 0;
                }
                info.inner = i4 != 0;
            }
        }
    }

    // Frame buffers (whole macroblocks).
    const int ystride = mb_w_ * 16;
    const int uvstride = mb_w_ * 8;
    std::vector<uint8_t>& Y = *yp;
    std::vector<uint8_t>& U = *up;
    std::vector<uint8_t>& V = *vp;
    Y.assign(static_cast<size_t>(ystride) * static_cast<size_t>(mb_h_) * 16, 0);
    U.assign(static_cast<size_t>(uvstride) * static_cast<size_t>(mb_h_) * 8, 0);
    V.assign(static_cast<size_t>(uvstride) * static_cast<size_t>(mb_h_) * 8, 0);
    std::vector<FilterInfo> finfo(static_cast<size_t>(mb_w_) * static_cast<size_t>(mb_h_));

    std::vector<uint8_t> intra_t(static_cast<size_t>(mb_w_) * 4, B_DC_PRED);
    std::vector<uint8_t> top_nz_y(static_cast<size_t>(mb_w_) * 4, 0), top_nz_u(static_cast<size_t>(mb_w_) * 2, 0),
        top_nz_v(static_cast<size_t>(mb_w_) * 2, 0), top_nz_dc(static_cast<size_t>(mb_w_), 0);
    std::vector<MbInfo> row(static_cast<size_t>(mb_w_));

    for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
        // Intra modes for the whole row (first partition).
        uint8_t intra_l[4] = {B_DC_PRED, B_DC_PRED, B_DC_PRED, B_DC_PRED};
        for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
            MbInfo& mb = row[static_cast<size_t>(mb_x)];
            uint8_t* top = intra_t.data() + 4 * mb_x;
            if (update_map_) {
                mb.segment = !br_.get(segment_proba_[0]) ? br_.get(segment_proba_[1])
                                                         : br_.get(segment_proba_[2]) + 2;
            } else {
                mb.segment = 0;
            }
            mb.skip = use_skip_proba_ ? br_.get(skip_p_) != 0 : false;
            mb.i4x4 = !br_.get(145);
            if (!mb.i4x4) {
                const int ymode = br_.get(156) ? (br_.get(128) ? TM_PRED : H_PRED) : (br_.get(163) ? V_PRED : DC_PRED);
                mb.imodes[0] = static_cast<uint8_t>(ymode);
                std::memset(top, ymode, 4);
                std::memset(intra_l, ymode, 4);
            } else {
                uint8_t* modes = mb.imodes;
                for (int y = 0; y < 4; ++y) {
                    int ymode = intra_l[y];
                    for (int x = 0; x < 4; ++x) {
                        const uint8_t* prob = wt::kBModesProba + (static_cast<size_t>(top[x]) * 10 + ymode) * 9;
                        if (!br_.get(prob[0])) ymode = B_DC_PRED;
                        else if (!br_.get(prob[1])) ymode = B_TM_PRED;
                        else if (!br_.get(prob[2])) ymode = B_VE_PRED;
                        else if (!br_.get(prob[3])) {
                            if (!br_.get(prob[4])) ymode = B_HE_PRED;
                            else ymode = !br_.get(prob[5]) ? B_RD_PRED : B_VR_PRED;
                        } else {
                            if (!br_.get(prob[6])) ymode = B_LD_PRED;
                            else if (!br_.get(prob[7])) ymode = B_VL_PRED;
                            else ymode = !br_.get(prob[8]) ? B_HD_PRED : B_HU_PRED;
                        }
                        top[x] = static_cast<uint8_t>(ymode);
                    }
                    std::memcpy(modes, top, 4);
                    modes += 4;
                    intra_l[y] = static_cast<uint8_t>(ymode);
                }
            }
            mb.uvmode = !br_.get(142) ? DC_PRED : (!br_.get(114) ? V_PRED : (br_.get(183) ? TM_PRED : H_PRED));
        }
        if (br_.eof()) return false;

        // Residuals (token partition of this row).
        BoolDecoder& tbr = parts_[static_cast<size_t>(mb_y) & last];
        uint8_t left_nz_y[4] = {0, 0, 0, 0}, left_nz_u[2] = {0, 0}, left_nz_v[2] = {0, 0}, left_nz_dc = 0;
        for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
            MbInfo& mb = row[static_cast<size_t>(mb_x)];
            std::memset(mb.coeffs, 0, sizeof(mb.coeffs));
            bool zero = true;
            uint8_t* tny = top_nz_y.data() + 4 * mb_x;
            uint8_t* tnu = top_nz_u.data() + 2 * mb_x;
            uint8_t* tnv = top_nz_v.data() + 2 * mb_x;
            if (!mb.skip) {
                const Quant& q = dqm_[mb.segment];
                int first = 0;
                int ac_type = 3;
                int16_t* dst = mb.coeffs;
                if (!mb.i4x4) {
                    int16_t dc[16] = {0};
                    const int ctx = top_nz_dc[static_cast<size_t>(mb_x)] + left_nz_dc;
                    const int nz = get_coeffs(tbr, 1, ctx, q.y2, 0, dc);
                    top_nz_dc[static_cast<size_t>(mb_x)] = left_nz_dc = nz > 0 ? 1 : 0;
                    if (nz > 1) {
                        inverse_wht(dc, dst);
                    } else {
                        const int dc0 = (dc[0] + 3) >> 3;
                        for (int i = 0; i < 16 * 16; i += 16) dst[i] = static_cast<int16_t>(dc0);
                    }
                    first = 1;
                    ac_type = 0;
                }
                for (int y = 0; y < 4; ++y) {
                    int l = left_nz_y[y];
                    for (int x = 0; x < 4; ++x) {
                        const int ctx = l + tny[x];
                        const int nz = get_coeffs(tbr, ac_type, ctx, q.y1, first, dst);
                        l = nz > first ? 1 : 0;
                        tny[x] = static_cast<uint8_t>(l);
                        if (nz > 1 || dst[0] != 0) zero = false;
                        dst += 16;
                    }
                    left_nz_y[y] = static_cast<uint8_t>(l);
                }
                for (int ch = 0; ch < 2; ++ch) {
                    uint8_t* tn = ch == 0 ? tnu : tnv;
                    uint8_t* ln = ch == 0 ? left_nz_u : left_nz_v;
                    for (int y = 0; y < 2; ++y) {
                        int l = ln[y];
                        for (int x = 0; x < 2; ++x) {
                            const int ctx = l + tn[x];
                            const int nz = get_coeffs(tbr, 2, ctx, q.uv, 0, dst);
                            l = nz > 0 ? 1 : 0;
                            tn[x] = static_cast<uint8_t>(l);
                            if (nz > 1 || dst[0] != 0) zero = false;
                            dst += 16;
                        }
                        ln[y] = static_cast<uint8_t>(l);
                    }
                }
            } else {
                std::memset(tny, 0, 4);
                std::memset(tnu, 0, 2);
                std::memset(tnv, 0, 2);
                std::memset(left_nz_y, 0, 4);
                std::memset(left_nz_u, 0, 2);
                std::memset(left_nz_v, 0, 2);
                if (!mb.i4x4) {
                    top_nz_dc[static_cast<size_t>(mb_x)] = 0;
                    left_nz_dc = 0;
                }
            }
            mb.zero = zero;
            if (filter_type_ > 0) {
                FilterInfo fi = fstrengths_[mb.segment][mb.i4x4 ? 1 : 0];
                fi.inner = fi.inner || !(mb.skip || zero);
                finfo[static_cast<size_t>(mb_y) * static_cast<size_t>(mb_w_) + static_cast<size_t>(mb_x)] = fi;
            }
            if (tbr.eof()) return false;
        }

        // Reconstruction of the row (prediction from unfiltered samples).
        for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
            const MbInfo& mb = row[static_cast<size_t>(mb_x)];
            uint8_t ybuf[BPS * 17];
            uint8_t ubuf[BPS * 9];
            uint8_t vbuf[BPS * 9];
            uint8_t* yd = ybuf + BPS + 8;
            uint8_t* ud = ubuf + BPS + 8;
            uint8_t* vd = vbuf + BPS + 8;
            const int x0 = mb_x * 16, y0 = mb_y * 16;
            const int cx0 = mb_x * 8, cy0 = mb_y * 8;
            auto Yat = [&](int x, int y) { return Y[static_cast<size_t>(y) * ystride + static_cast<size_t>(x)]; };
            // Left column.
            for (int j = 0; j < 16; ++j) yd[j * BPS - 1] = mb_x > 0 ? Yat(x0 - 1, y0 + j) : 129;
            for (int j = 0; j < 8; ++j) {
                ud[j * BPS - 1] = mb_x > 0 ? U[static_cast<size_t>(cy0 + j) * uvstride + cx0 - 1] : 129;
                vd[j * BPS - 1] = mb_x > 0 ? V[static_cast<size_t>(cy0 + j) * uvstride + cx0 - 1] : 129;
            }
            // Top row, top-left and top-right.
            if (mb_y > 0) {
                for (int i = 0; i < 16; ++i) yd[i - BPS] = Yat(x0 + i, y0 - 1);
                for (int i = 0; i < 8; ++i) {
                    ud[i - BPS] = U[static_cast<size_t>(cy0 - 1) * uvstride + cx0 + i];
                    vd[i - BPS] = V[static_cast<size_t>(cy0 - 1) * uvstride + cx0 + i];
                }
                yd[-1 - BPS] = mb_x > 0 ? Yat(x0 - 1, y0 - 1) : 129;
                ud[-1 - BPS] = mb_x > 0 ? U[static_cast<size_t>(cy0 - 1) * uvstride + cx0 - 1] : 129;
                vd[-1 - BPS] = mb_x > 0 ? V[static_cast<size_t>(cy0 - 1) * uvstride + cx0 - 1] : 129;
                for (int i = 16; i < 20; ++i) {
                    yd[i - BPS] = mb_x < mb_w_ - 1 ? Yat(x0 + i, y0 - 1) : Yat(x0 + 15, y0 - 1);
                }
            } else {
                std::memset(yd - BPS - 1, 127, 16 + 4 + 1);
                std::memset(ud - BPS - 1, 127, 8 + 1);
                std::memset(vd - BPS - 1, 127, 8 + 1);
            }
            if (mb.i4x4) {
                for (int k = 1; k <= 3; ++k) std::memcpy(yd + 16 + (4 * k - 1) * BPS, yd + 16 - BPS, 4);
                for (int n = 0; n < 16; ++n) {
                    uint8_t* dst = yd + (n & 3) * 4 + (n >> 2) * 4 * BPS;
                    predict4(mb.imodes[n], dst);
                    inverse_dct_add(mb.coeffs + n * 16, dst);
                }
            } else {
                predict_block(mb.imodes[0], yd, 16, mb_y > 0, mb_x > 0);
                for (int n = 0; n < 16; ++n) {
                    inverse_dct_add(mb.coeffs + n * 16, yd + (n & 3) * 4 + (n >> 2) * 4 * BPS);
                }
            }
            predict_block(mb.uvmode, ud, 8, mb_y > 0, mb_x > 0);
            predict_block(mb.uvmode, vd, 8, mb_y > 0, mb_x > 0);
            for (int n = 0; n < 4; ++n) {
                inverse_dct_add(mb.coeffs + 256 + n * 16, ud + (n & 1) * 4 + (n >> 1) * 4 * BPS);
                inverse_dct_add(mb.coeffs + 320 + n * 16, vd + (n & 1) * 4 + (n >> 1) * 4 * BPS);
            }
            for (int j = 0; j < 16; ++j) std::memcpy(&Y[static_cast<size_t>(y0 + j) * ystride + x0], yd + j * BPS, 16);
            for (int j = 0; j < 8; ++j) {
                std::memcpy(&U[static_cast<size_t>(cy0 + j) * uvstride + cx0], ud + j * BPS, 8);
                std::memcpy(&V[static_cast<size_t>(cy0 + j) * uvstride + cx0], vd + j * BPS, 8);
            }
        }
    }

    // Loop filter over the whole frame, macroblocks in raster order.
    if (filter_type_ > 0) {
        // Filtering runs after the whole frame is predicted, because
        // prediction reads unfiltered samples.
        for (int mb_y = 0; mb_y < mb_h_; ++mb_y) {
            for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
                const FilterInfo& fi = finfo[static_cast<size_t>(mb_y) * static_cast<size_t>(mb_w_) + static_cast<size_t>(mb_x)];
                const int limit = fi.limit;
                if (limit == 0) continue;
                uint8_t* yd = &Y[static_cast<size_t>(mb_y) * 16 * ystride + static_cast<size_t>(mb_x) * 16];
                if (filter_type_ == 1) {
                    if (mb_x > 0) simple_filter(yd, 1, ystride, 16, limit + 4);
                    if (fi.inner) {
                        for (int k = 1; k <= 3; ++k) simple_filter(yd + 4 * k, 1, ystride, 16, limit);
                    }
                    if (mb_y > 0) simple_filter(yd, ystride, 1, 16, limit + 4);
                    if (fi.inner) {
                        for (int k = 1; k <= 3; ++k) simple_filter(yd + 4 * k * ystride, ystride, 1, 16, limit);
                    }
                } else {
                    uint8_t* ud = &U[static_cast<size_t>(mb_y) * 8 * uvstride + static_cast<size_t>(mb_x) * 8];
                    uint8_t* vd = &V[static_cast<size_t>(mb_y) * 8 * uvstride + static_cast<size_t>(mb_x) * 8];
                    const int il = fi.ilevel;
                    const int ht = fi.hev;
                    if (mb_x > 0) {
                        normal_filter(yd, 1, ystride, 16, limit + 4, il, ht, true);
                        normal_filter(ud, 1, uvstride, 8, limit + 4, il, ht, true);
                        normal_filter(vd, 1, uvstride, 8, limit + 4, il, ht, true);
                    }
                    if (fi.inner) {
                        for (int k = 1; k <= 3; ++k) normal_filter(yd + 4 * k, 1, ystride, 16, limit, il, ht, false);
                        normal_filter(ud + 4, 1, uvstride, 8, limit, il, ht, false);
                        normal_filter(vd + 4, 1, uvstride, 8, limit, il, ht, false);
                    }
                    if (mb_y > 0) {
                        normal_filter(yd, ystride, 1, 16, limit + 4, il, ht, true);
                        normal_filter(ud, uvstride, 1, 8, limit + 4, il, ht, true);
                        normal_filter(vd, uvstride, 1, 8, limit + 4, il, ht, true);
                    }
                    if (fi.inner) {
                        for (int k = 1; k <= 3; ++k) {
                            normal_filter(yd + 4 * k * ystride, ystride, 1, 16, limit, il, ht, false);
                        }
                        normal_filter(ud + 4 * uvstride, uvstride, 1, 8, limit, il, ht, false);
                        normal_filter(vd + 4 * uvstride, uvstride, 1, 8, limit, il, ht, false);
                    }
                }
            }
        }
    }
    *out_w = width;
    *out_h = height;
    *ystride_out = ystride;
    *uvstride_out = uvstride;
    return true;
}

/* ---- YUV to RGB --------------------------------------------------------- */

inline int mult_hi(int v, int coeff) { return (v * coeff) >> 8; }
inline uint8_t yuv_clip8(int v) {
    constexpr int kMask = (256 << 6) - 1;
    return static_cast<uint8_t>((v & ~kMask) == 0 ? (v >> 6) : (v < 0 ? 0 : 255));
}
inline void yuv_to_rgb(int y, int u, int v, uint8_t* rgb) {
    rgb[0] = yuv_clip8(mult_hi(y, 19077) + mult_hi(v, 26149) - 14234);
    rgb[1] = yuv_clip8(mult_hi(y, 19077) - mult_hi(u, 6419) - mult_hi(v, 13320) + 8708);
    rgb[2] = yuv_clip8(mult_hi(y, 19077) + mult_hi(u, 33050) - 17685);
}

/** One pair of output rows of the bilinear chroma upsampler. */
void upsample_pair(const uint8_t* top_y, const uint8_t* bottom_y, const uint8_t* top_u, const uint8_t* top_v,
                   const uint8_t* cur_u, const uint8_t* cur_v, uint8_t* top_dst, uint8_t* bottom_dst, int len,
                   int step) {
    auto load = [](uint8_t u, uint8_t v) { return static_cast<uint32_t>(u) | (static_cast<uint32_t>(v) << 16); };
    const int last_pixel_pair = (len - 1) >> 1;
    uint32_t tl_uv = load(top_u[0], top_v[0]);
    uint32_t l_uv = load(cur_u[0], cur_v[0]);
    {
        const uint32_t uv0 = (3 * tl_uv + l_uv + 0x00020002u) >> 2;
        yuv_to_rgb(top_y[0], static_cast<int>(uv0 & 0xff), static_cast<int>(uv0 >> 16), top_dst);
    }
    if (bottom_y) {
        const uint32_t uv0 = (3 * l_uv + tl_uv + 0x00020002u) >> 2;
        yuv_to_rgb(bottom_y[0], static_cast<int>(uv0 & 0xff), static_cast<int>(uv0 >> 16), bottom_dst);
    }
    for (int x = 1; x <= last_pixel_pair; ++x) {
        const uint32_t t_uv = load(top_u[x], top_v[x]);
        const uint32_t uv = load(cur_u[x], cur_v[x]);
        const uint32_t avg = tl_uv + t_uv + l_uv + uv + 0x00080008u;
        const uint32_t diag_12 = (avg + 2 * (t_uv + l_uv)) >> 3;
        const uint32_t diag_03 = (avg + 2 * (tl_uv + uv)) >> 3;
        {
            const uint32_t uv0 = (diag_12 + tl_uv) >> 1;
            const uint32_t uv1 = (diag_03 + t_uv) >> 1;
            yuv_to_rgb(top_y[2 * x - 1], static_cast<int>(uv0 & 0xff), static_cast<int>((uv0 >> 16) & 0xff),
                       top_dst + (2 * x - 1) * step);
            yuv_to_rgb(top_y[2 * x], static_cast<int>(uv1 & 0xff), static_cast<int>((uv1 >> 16) & 0xff),
                       top_dst + (2 * x) * step);
        }
        if (bottom_y) {
            const uint32_t uv0 = (diag_03 + l_uv) >> 1;
            const uint32_t uv1 = (diag_12 + uv) >> 1;
            yuv_to_rgb(bottom_y[2 * x - 1], static_cast<int>(uv0 & 0xff), static_cast<int>((uv0 >> 16) & 0xff),
                       bottom_dst + (2 * x - 1) * step);
            yuv_to_rgb(bottom_y[2 * x], static_cast<int>(uv1 & 0xff), static_cast<int>((uv1 >> 16) & 0xff),
                       bottom_dst + (2 * x) * step);
        }
        tl_uv = t_uv;
        l_uv = uv;
    }
    if (!(len & 1)) {
        {
            const uint32_t uv0 = (3 * tl_uv + l_uv + 0x00020002u) >> 2;
            yuv_to_rgb(top_y[len - 1], static_cast<int>(uv0 & 0xff), static_cast<int>((uv0 >> 16) & 0xff),
                       top_dst + (len - 1) * step);
        }
        if (bottom_y) {
            const uint32_t uv0 = (3 * l_uv + tl_uv + 0x00020002u) >> 2;
            yuv_to_rgb(bottom_y[len - 1], static_cast<int>(uv0 & 0xff), static_cast<int>((uv0 >> 16) & 0xff),
                       bottom_dst + (len - 1) * step);
        }
    }
}

/* ---- alpha ---------------------------------------------------------------- */

bool decode_alpha(const uint8_t* data, size_t size, uint32_t w, uint32_t h, std::vector<uint8_t>* alpha) {
    if (size <= 1) return false;
    const int method = data[0] & 3;
    const int filter = (data[0] >> 2) & 3;
    const int pre = (data[0] >> 4) & 3;
    const int rsrv = (data[0] >> 6) & 3;
    if (method > 1 || pre > 1 || rsrv != 0) return false;
    const size_t count = static_cast<size_t>(w) * h;
    std::vector<uint8_t> raw(count);
    if (method == 0) {
        if (size - 1 < count) return false;
        std::memcpy(raw.data(), data + 1, count);
    } else {
        LBitReader br(data + 1, size - 1);
        std::vector<uint32_t> argb;
        if (!decode_image_stream(br, w, h, true, &argb)) return false;
        for (size_t i = 0; i < count; ++i) raw[i] = static_cast<uint8_t>((argb[i] >> 8) & 0xff);
    }
    alpha->assign(count, 0);
    for (uint32_t y = 0; y < h; ++y) {
        const uint8_t* in = raw.data() + static_cast<size_t>(y) * w;
        uint8_t* out = alpha->data() + static_cast<size_t>(y) * w;
        const uint8_t* prev = y ? out - w : nullptr;
        auto horizontal = [&]() {
            uint8_t pred = prev ? prev[0] : 0;
            for (uint32_t i = 0; i < w; ++i) {
                out[i] = static_cast<uint8_t>(pred + in[i]);
                pred = out[i];
            }
        };
        switch (filter) {
            case 0: std::memcpy(out, in, w); break;
            case 1: horizontal(); break;
            case 2:
                if (!prev) {
                    horizontal();
                } else {
                    for (uint32_t i = 0; i < w; ++i) out[i] = static_cast<uint8_t>(prev[i] + in[i]);
                }
                break;
            default:
                if (!prev) {
                    horizontal();
                } else {
                    uint8_t top = prev[0], top_left = top, left = top;
                    for (uint32_t i = 0; i < w; ++i) {
                        top = prev[i];
                        const int g = left + top - top_left;
                        const int predv = (g & ~0xff) == 0 ? g : (g < 0 ? 0 : 255);
                        left = static_cast<uint8_t>(in[i] + predv);
                        top_left = top;
                        out[i] = left;
                    }
                }
                break;
        }
    }
    return true;
}

} // namespace

bool webp_is_signature(const uint8_t* data, size_t size) {
    return data && size >= 12 && std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WEBP", 4) == 0;
}

bool webp_decode(const uint8_t* data, size_t size, Raster* out) {
    if (!out || !webp_is_signature(data, size)) return false;
    const uint32_t riff_size = le32(data + 4);
    if (riff_size < 12 || riff_size > 0xfffffff6u) return false;
    if (static_cast<size_t>(riff_size) > size - 8) return false;
    const size_t end = static_cast<size_t>(riff_size) + 8;
    size_t pos = 12;
    bool vp8x = false;
    bool flag_alpha = false;
    uint32_t canvas_w = 0, canvas_h = 0;
    const uint8_t* alph = nullptr;
    size_t alph_size = 0;
    const uint8_t* frame = nullptr;
    size_t frame_size = 0;
    bool lossless = false;
    while (pos + 8 <= end) {
        const uint8_t* tag = data + pos;
        const uint32_t csize = le32(data + pos + 4);
        const size_t body = pos + 8;
        if (csize > end - body) return false;
        if (std::memcmp(tag, "VP8X", 4) == 0) {
            if (pos != 12 || csize < 10) return false;
            const uint8_t flags = data[body];
            if (flags & 0x02) return false;  // animation
            flag_alpha = (flags & 0x10) != 0;
            canvas_w = le24(data + body + 4) + 1;
            canvas_h = le24(data + body + 7) + 1;
            vp8x = true;
        } else if (std::memcmp(tag, "ALPH", 4) == 0) {
            if (!alph) {
                alph = data + body;
                alph_size = csize;
            }
        } else if (std::memcmp(tag, "VP8 ", 4) == 0 || std::memcmp(tag, "VP8L", 4) == 0) {
            frame = data + body;
            frame_size = csize;
            lossless = tag[3] == 'L';
            break;
        } else if (std::memcmp(tag, "ANIM", 4) == 0 || std::memcmp(tag, "ANMF", 4) == 0) {
            return false;
        } else if (!vp8x) {
            return false;  // a simple file starts with its frame
        }
        pos = body + csize + (csize & 1);
    }
    if (!frame) return false;
    if (lossless) {
        uint32_t w = 0, h = 0;
        bool alpha_used = false;
        std::vector<uint32_t> argb;
        if (!vp8l_decode(frame, frame_size, &w, &h, &alpha_used, &argb)) return false;
        if (vp8x && (w != canvas_w || h != canvas_h)) return false;
        if (w > static_cast<uint32_t>(kMaxDimension) || h > static_cast<uint32_t>(kMaxDimension)) return false;
        const int c = alpha_used ? 4 : 3;
        out->width = static_cast<int>(w);
        out->height = static_cast<int>(h);
        out->channels = c;
        out->pixels.resize(static_cast<size_t>(w) * h * static_cast<size_t>(c));
        uint8_t* d = out->pixels.data();
        for (size_t i = 0; i < argb.size(); ++i) {
            const uint32_t v = argb[i];
            d[0] = static_cast<uint8_t>(v >> 16);
            d[1] = static_cast<uint8_t>(v >> 8);
            d[2] = static_cast<uint8_t>(v);
            if (c == 4) d[3] = static_cast<uint8_t>(v >> 24);
            d += c;
        }
        return true;
    }
    Vp8Decoder dec;
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> Y, U, V;
    int ystride = 0, uvstride = 0;
    if (!dec.decode(frame, frame_size, &w, &h, &Y, &U, &V, &ystride, &uvstride)) return false;
    if (vp8x && (w != canvas_w || h != canvas_h)) return false;
    if (w > static_cast<uint32_t>(kMaxDimension) || h > static_cast<uint32_t>(kMaxDimension)) return false;
    const bool has_alpha = flag_alpha || alph != nullptr;
    std::vector<uint8_t> alpha;
    if (has_alpha) {
        if (alph) {
            if (!decode_alpha(alph, alph_size, w, h, &alpha)) return false;
        } else {
            alpha.assign(static_cast<size_t>(w) * h, 255);
        }
    }
    const int c = has_alpha ? 4 : 3;
    out->width = static_cast<int>(w);
    out->height = static_cast<int>(h);
    out->channels = c;
    out->pixels.assign(static_cast<size_t>(w) * h * static_cast<size_t>(c), 255);
    std::vector<uint8_t> rgb_top(static_cast<size_t>(w) * 3), rgb_bottom(static_cast<size_t>(w) * 3);
    auto store = [&](const std::vector<uint8_t>& rgb, uint32_t y) {
        uint8_t* d = out->pixels.data() + static_cast<size_t>(y) * w * static_cast<size_t>(c);
        for (uint32_t x = 0; x < w; ++x) {
            d[x * c + 0] = rgb[x * 3 + 0];
            d[x * c + 1] = rgb[x * 3 + 1];
            d[x * c + 2] = rgb[x * 3 + 2];
            if (c == 4) d[x * c + 3] = alpha[static_cast<size_t>(y) * w + x];
        }
    };
    auto yrow = [&](uint32_t y) { return Y.data() + static_cast<size_t>(y) * ystride; };
    auto urow = [&](uint32_t y) { return U.data() + static_cast<size_t>(y) * uvstride; };
    auto vrow = [&](uint32_t y) { return V.data() + static_cast<size_t>(y) * uvstride; };
    const int len = static_cast<int>(w);
    upsample_pair(yrow(0), nullptr, urow(0), vrow(0), urow(0), vrow(0), rgb_top.data(), nullptr, len, 3);
    store(rgb_top, 0);
    uint32_t y = 0;
    for (; y + 2 < h; y += 2) {
        const uint32_t k = y / 2;
        upsample_pair(yrow(y + 1), yrow(y + 2), urow(k), vrow(k), urow(k + 1), vrow(k + 1), rgb_top.data(),
                      rgb_bottom.data(), len, 3);
        store(rgb_top, y + 1);
        store(rgb_bottom, y + 2);
    }
    if (!(h & 1)) {
        const uint32_t k = y / 2;
        upsample_pair(yrow(y + 1), nullptr, urow(k), vrow(k), urow(k), vrow(k), rgb_top.data(), nullptr, len, 3);
        store(rgb_top, y + 1);
    }
    return true;
}

/* ======================================================================== */
/* VP8L encoder (lossless)                                                  */
/* ======================================================================== */

namespace {

class LBitWriter {
public:
    void put(uint32_t value, int bits) {
        acc_ |= static_cast<uint64_t>(value) << count_;
        count_ += bits;
        while (count_ >= 8) {
            out_.push_back(static_cast<uint8_t>(acc_));
            acc_ >>= 8;
            count_ -= 8;
        }
    }
    std::vector<uint8_t> finish() {
        if (count_ > 0) out_.push_back(static_cast<uint8_t>(acc_));
        acc_ = 0;
        count_ = 0;
        return out_;
    }

private:
    std::vector<uint8_t> out_;
    uint64_t acc_ = 0;
    int count_ = 0;
};

/** Huffman code lengths limited to @p limit bits (counts are floored and
 *  the tree rebuilt until it fits). */
std::vector<uint8_t> code_lengths(const std::vector<uint32_t>& freq, int limit) {
    const size_t n = freq.size();
    std::vector<uint8_t> lengths(n, 0);
    std::vector<size_t> used;
    for (size_t i = 0; i < n; ++i) {
        if (freq[i]) used.push_back(i);
    }
    if (used.size() <= 1) {
        if (used.size() == 1) lengths[used[0]] = 1;
        return lengths;
    }
    for (uint32_t floor = 1;; floor *= 2) {
        struct Node {
            uint64_t weight;
            int left, right;
        };
        std::vector<Node> nodes;
        using Item = std::pair<uint64_t, int>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
        for (size_t i : used) {
            nodes.push_back({std::max<uint64_t>(freq[i], floor), -1, static_cast<int>(i)});
            heap.push({nodes.back().weight, static_cast<int>(nodes.size() - 1)});
        }
        while (heap.size() > 1) {
            const Item a = heap.top();
            heap.pop();
            const Item b = heap.top();
            heap.pop();
            nodes.push_back({a.first + b.first, a.second, b.second});
            heap.push({nodes.back().weight, static_cast<int>(nodes.size() - 1)});
        }
        std::fill(lengths.begin(), lengths.end(), 0);
        int max_len = 0;
        std::vector<std::pair<int, int>> stack = {{heap.top().second, 0}};
        while (!stack.empty()) {
            const auto [idx, depth] = stack.back();
            stack.pop_back();
            const Node& nd = nodes[static_cast<size_t>(idx)];
            if (nd.left < 0) {
                lengths[static_cast<size_t>(nd.right)] = static_cast<uint8_t>(depth);
                max_len = std::max(max_len, depth);
            } else {
                stack.push_back({nd.left, depth + 1});
                stack.push_back({nd.right, depth + 1});
            }
        }
        if (max_len <= limit) return lengths;
    }
}

/** Canonical codes (bit-reversed for LSB-first emission). */
std::vector<uint16_t> canonical_codes(const std::vector<uint8_t>& lengths) {
    int counts[16] = {0};
    for (uint8_t l : lengths) counts[l]++;
    counts[0] = 0;
    int next[16] = {0};
    int code = 0;
    for (int len = 1; len < 16; ++len) {
        code = (code + counts[len - 1]) << 1;
        next[len] = code;
    }
    std::vector<uint16_t> codes(lengths.size(), 0);
    for (size_t i = 0; i < lengths.size(); ++i) {
        const int len = lengths[i];
        if (!len) continue;
        const int c = next[len]++;
        int rev = 0;
        for (int b = 0; b < len; ++b) rev |= ((c >> b) & 1) << (len - 1 - b);
        codes[i] = static_cast<uint16_t>(rev);
    }
    return codes;
}

struct EncCode {
    std::vector<uint8_t> lengths;
    std::vector<uint16_t> codes;
    bool zero_bits = false;  // single-symbol code
    void emit(LBitWriter& bw, int symbol) const {
        if (zero_bits) return;
        bw.put(codes[static_cast<size_t>(symbol)], lengths[static_cast<size_t>(symbol)]);
    }
};

/** Write one prefix code and return its emitter. */
EncCode write_code(LBitWriter& bw, const std::vector<uint32_t>& freq) {
    EncCode ec;
    ec.lengths = code_lengths(freq, 15);
    int nonzero = 0;
    int first_symbol = -1;
    for (size_t i = 0; i < ec.lengths.size(); ++i) {
        if (ec.lengths[i]) {
            ++nonzero;
            if (first_symbol < 0) first_symbol = static_cast<int>(i);
        }
    }
    if (nonzero <= 1) {
        // Simple code with one symbol (an unused alphabet names symbol 0).
        const int s = nonzero ? first_symbol : 0;
        if (s < 256) {
            bw.put(1, 1);
            bw.put(0, 1);
            if (s < 2) {
                bw.put(0, 1);
                bw.put(static_cast<uint32_t>(s), 1);
            } else {
                bw.put(1, 1);
                bw.put(static_cast<uint32_t>(s), 8);
            }
            ec.zero_bits = true;
            return ec;
        }
        // A lone symbol above 255 needs the normal form; give it a partner
        // so the code is complete (two one-bit codes).
        ec.lengths[static_cast<size_t>(s == 0 ? 1 : 0)] = 1;
        ec.lengths[static_cast<size_t>(s)] = 1;
    }
    ec.codes = canonical_codes(ec.lengths);
    // Run-length the code lengths with the 19-symbol code-length alphabet.
    std::vector<std::pair<int, int>> tokens;  // (symbol, extra)
    const size_t n = ec.lengths.size();
    size_t i = 0;
    while (i < n) {
        const int len = ec.lengths[i];
        size_t run = 1;
        while (i + run < n && ec.lengths[i + run] == len) ++run;
        if (len == 0 && run >= 3) {
            size_t r = run;
            while (r >= 3) {
                if (r >= 11) {
                    const size_t take = std::min<size_t>(r, 138);
                    tokens.push_back({18, static_cast<int>(take - 11)});
                    r -= take;
                } else {
                    const size_t take = std::min<size_t>(r, 10);
                    tokens.push_back({17, static_cast<int>(take - 3)});
                    r -= take;
                }
            }
            for (; r > 0; --r) tokens.push_back({0, 0});
        } else if (len != 0 && run >= 4) {
            tokens.push_back({len, 0});
            size_t r = run - 1;
            while (r >= 3) {
                const size_t take = std::min<size_t>(r, 6);
                tokens.push_back({16, static_cast<int>(take - 3)});
                r -= take;
            }
            for (; r > 0; --r) tokens.push_back({len, 0});
        } else {
            for (size_t k = 0; k < run; ++k) tokens.push_back({len, 0});
        }
        i += run;
    }
    std::vector<uint32_t> cl_freq(19, 0);
    for (const auto& t : tokens) cl_freq[static_cast<size_t>(t.first)]++;
    std::vector<uint8_t> cl_lengths = code_lengths(cl_freq, 7);
    int cl_nonzero = 0;
    for (uint8_t l : cl_lengths) cl_nonzero += l ? 1 : 0;
    const std::vector<uint16_t> cl_codes = canonical_codes(cl_lengths);
    int num_codes = 19;
    while (num_codes > 4 && cl_lengths[wt::kCodeLengthCodeOrder[num_codes - 1]] == 0) --num_codes;
    bw.put(0, 1);  // normal code
    bw.put(static_cast<uint32_t>(num_codes - 4), 4);
    for (int k = 0; k < num_codes; ++k) bw.put(cl_lengths[wt::kCodeLengthCodeOrder[k]], 3);
    bw.put(0, 1);  // max_symbol = alphabet size
    for (const auto& t : tokens) {
        if (cl_nonzero > 1) bw.put(cl_codes[static_cast<size_t>(t.first)], cl_lengths[static_cast<size_t>(t.first)]);
        if (t.first == 16) bw.put(static_cast<uint32_t>(t.second), 2);
        if (t.first == 17) bw.put(static_cast<uint32_t>(t.second), 3);
        if (t.first == 18) bw.put(static_cast<uint32_t>(t.second), 7);
    }
    return ec;
}

void prefix_encode(uint32_t value, int* symbol, int* extra_bits, uint32_t* extra) {
    // value >= 1
    const uint32_t v = value - 1;
    if (v < 4) {
        *symbol = static_cast<int>(v);
        *extra_bits = 0;
        *extra = 0;
        return;
    }
    int hb = 0;
    while ((v >> (hb + 1)) != 0) ++hb;
    const int second = static_cast<int>((v >> (hb - 1)) & 1);
    *extra_bits = hb - 1;
    *symbol = 2 * hb + second;
    *extra = v & ((1u << *extra_bits) - 1);
}

struct Token {
    bool copy;
    uint32_t argb;     // literal
    uint32_t length;   // copy
    uint32_t dist_code;
};

/** Encode @p argb (w x h) as an entropy-coded image with one code group:
 *  colour-cache bit, (for the main image) meta-code bit, five codes, pixels. */
void write_entropy_image(LBitWriter& bw, const std::vector<uint32_t>& argb, uint32_t w, bool level0) {
    const bool allow_copies = level0;
    std::vector<Token> tokens;
    const size_t total = argb.size();
    size_t pos = 0;
    while (pos < total) {
        // Runs copied from the left pixel (plane code 2) or the row above (1).
        uint32_t best_len = 0;
        uint32_t best_code = 0;
        if (allow_copies) {
            if (pos >= 1) {
                uint32_t l = 0;
                while (pos + l < total && l < 4096 && argb[pos + l] == argb[pos + l - 1]) ++l;
                if (l > best_len) best_len = l, best_code = 2;
            }
            if (pos >= w) {
                uint32_t l = 0;
                while (pos + l < total && l < 4096 && argb[pos + l] == argb[pos + l - w]) ++l;
                if (l > best_len) best_len = l, best_code = 1;
            }
        }
        if (best_len >= 3) {
            tokens.push_back({true, 0, best_len, best_code});
            pos += best_len;
        } else {
            tokens.push_back({false, argb[pos], 0, 0});
            ++pos;
        }
    }
    std::vector<uint32_t> f_green(256 + 24, 0), f_red(256, 0), f_blue(256, 0), f_alpha(256, 0), f_dist(40, 0);
    for (const Token& t : tokens) {
        if (t.copy) {
            int s, eb;
            uint32_t ex;
            prefix_encode(t.length, &s, &eb, &ex);
            f_green[static_cast<size_t>(256 + s)]++;
            prefix_encode(t.dist_code, &s, &eb, &ex);
            f_dist[static_cast<size_t>(s)]++;
        } else {
            f_green[(t.argb >> 8) & 0xff]++;
            f_red[(t.argb >> 16) & 0xff]++;
            f_blue[t.argb & 0xff]++;
            f_alpha[t.argb >> 24]++;
        }
    }
    bw.put(0, 1);  // no colour cache
    if (level0) bw.put(0, 1);  // no meta prefix codes
    const EncCode cg = write_code(bw, f_green);
    const EncCode cr = write_code(bw, f_red);
    const EncCode cb = write_code(bw, f_blue);
    const EncCode ca = write_code(bw, f_alpha);
    const EncCode cd = write_code(bw, f_dist);
    for (const Token& t : tokens) {
        if (t.copy) {
            int s, eb;
            uint32_t ex;
            prefix_encode(t.length, &s, &eb, &ex);
            cg.emit(bw, 256 + s);
            bw.put(ex, eb);
            prefix_encode(t.dist_code, &s, &eb, &ex);
            cd.emit(bw, s);
            bw.put(ex, eb);
        } else {
            cg.emit(bw, static_cast<int>((t.argb >> 8) & 0xff));
            cr.emit(bw, static_cast<int>((t.argb >> 16) & 0xff));
            cb.emit(bw, static_cast<int>(t.argb & 0xff));
            ca.emit(bw, static_cast<int>(t.argb >> 24));
        }
    }
}

inline uint32_t sub_pixels(uint32_t a, uint32_t b) {
    const uint32_t ag = 0x00ff00ffu + (a & 0xff00ff00u) - (b & 0xff00ff00u);
    const uint32_t rb = 0xff00ff00u + (a & 0x00ff00ffu) - (b & 0x00ff00ffu);
    return (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
}

} // namespace

bool webp_encode_lossless(const uint8_t* pixels, int width, int height, int channels, std::vector<uint8_t>* out) {
    if (!pixels || !out || width <= 0 || height <= 0 || width > 16384 || height > 16384 ||
        !(channels == 1 || channels == 3 || channels == 4)) {
        return false;
    }
    const uint32_t w = static_cast<uint32_t>(width), h = static_cast<uint32_t>(height);
    std::vector<uint32_t> argb(static_cast<size_t>(w) * h);
    bool alpha_used = false;
    for (size_t i = 0; i < argb.size(); ++i) {
        const uint8_t* s = pixels + i * static_cast<size_t>(channels);
        const uint32_t r = s[0];
        const uint32_t g = channels == 1 ? s[0] : s[1];
        const uint32_t b = channels == 1 ? s[0] : s[2];
        const uint32_t a = channels == 4 ? s[3] : 255;
        if (a != 255) alpha_used = true;
        argb[i] = (a << 24) | (r << 16) | (g << 8) | b;
    }
    // Forward transforms in bitstream order: subtract green, then predictor
    // (the best of a few modes per 32x32 tile by sum of absolute residuals).
    // The decoder undoes them in reverse order.
    for (uint32_t& v : argb) {
        const uint32_t green = (v >> 8) & 0xff;
        const uint32_t red = ((v >> 16) - green) & 0xff;
        const uint32_t blue = (v - green) & 0xff;
        v = (v & 0xff00ff00u) | (red << 16) | blue;
    }
    constexpr int kTileBits = 5;
    const uint32_t tiles_w = div_round_up_shift(w, kTileBits);
    const uint32_t tiles_h = div_round_up_shift(h, kTileBits);
    std::vector<uint32_t> modes(static_cast<size_t>(tiles_w) * tiles_h, 0);
    static const int kCandidates[] = {1, 2, 5, 7, 10, 11, 12, 13};
    auto pred_at = [&](int mode, uint32_t x, uint32_t y) -> uint32_t {
        const uint32_t* row = argb.data() + static_cast<size_t>(y) * w;
        if (y == 0) return x == 0 ? 0xff000000u : row[x - 1];
        const uint32_t* top = row - w;
        if (x == 0) return top[0];
        return predict(mode, row[x - 1], top[x], top[x - 1], x + 1 < w ? top[x + 1] : row[0]);
    };
    for (uint32_t ty = 0; ty < tiles_h; ++ty) {
        for (uint32_t tx = 0; tx < tiles_w; ++tx) {
            uint64_t best = UINT64_MAX;
            int best_mode = 11;
            for (int mode : kCandidates) {
                uint64_t cost = 0;
                for (uint32_t y = ty << kTileBits; y < std::min(h, (ty + 1) << kTileBits); ++y) {
                    for (uint32_t x = tx << kTileBits; x < std::min(w, (tx + 1) << kTileBits); ++x) {
                        const uint32_t r = sub_pixels(argb[static_cast<size_t>(y) * w + x], pred_at(mode, x, y));
                        for (int sh = 0; sh < 32; sh += 8) {
                            const int c = static_cast<int>((r >> sh) & 0xff);
                            cost += static_cast<uint64_t>(c < 128 ? c : 256 - c);
                        }
                    }
                }
                if (cost < best) {
                    best = cost;
                    best_mode = mode;
                }
            }
            modes[static_cast<size_t>(ty) * tiles_w + tx] = 0xff000000u | (static_cast<uint32_t>(best_mode) << 8);
        }
    }
    std::vector<uint32_t> res(argb.size());
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const int mode = static_cast<int>((modes[(y >> kTileBits) * tiles_w + (x >> kTileBits)] >> 8) & 0xf);
            res[static_cast<size_t>(y) * w + x] = sub_pixels(argb[static_cast<size_t>(y) * w + x], pred_at(mode, x, y));
        }
    }

    LBitWriter bw;
    bw.put(0x2f, 8);
    bw.put(w - 1, 14);
    bw.put(h - 1, 14);
    bw.put(alpha_used ? 1u : 0u, 1);
    bw.put(0, 3);
    bw.put(1, 1);
    bw.put(kSubtractGreen, 2);
    bw.put(1, 1);
    bw.put(kPredictor, 2);
    bw.put(kTileBits - 2, 3);
    write_entropy_image(bw, modes, tiles_w, false);
    bw.put(0, 1);  // no further transforms
    write_entropy_image(bw, res, w, true);
    std::vector<uint8_t> vp8l = bw.finish();

    out->clear();
    const uint32_t chunk = static_cast<uint32_t>(vp8l.size());
    const uint32_t riff = 4 + 8 + chunk + (chunk & 1);
    auto put32 = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) out->push_back(static_cast<uint8_t>(v >> (8 * i)));
    };
    out->insert(out->end(), {'R', 'I', 'F', 'F'});
    put32(riff);
    out->insert(out->end(), {'W', 'E', 'B', 'P', 'V', 'P', '8', 'L'});
    put32(chunk);
    out->insert(out->end(), vp8l.begin(), vp8l.end());
    if (chunk & 1) out->push_back(0);
    return true;
}

} // namespace eshkol::image
