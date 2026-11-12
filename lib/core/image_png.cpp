/**
 * @file image_png.cpp
 * @brief Built-in PNG decoder and encoder for the Eshkol runtime.
 *
 * Decoding accepts every colour type and bit depth of the PNG specification,
 * Adam7 interlacing, palettes and tRNS transparency, and produces 8-bit RGB
 * or RGBA samples: 16-bit samples keep their high byte, low-bit-depth gray is
 * scaled to 8 bits, palettes and gray are expanded to RGB, and tRNS becomes
 * an alpha channel.  Critical chunks are CRC-checked; damaged ancillary
 * chunks are skipped.
 *
 * Encoding writes IHDR, IDAT and IEND only, chooses a per-row filter by the
 * minimum-sum-of-absolute-differences heuristic, sizes the zlib window to the
 * image data and splits the stream into 8 KiB IDAT chunks.  These are the
 * choices of the system libpng the runtime used before, so files written by
 * image-write keep the same bytes.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "image_codecs.h"

#include <cstring>

namespace eshkol::image {

namespace {

const uint8_t kSignature[8] = {137, 80, 78, 71, 13, 10, 26, 10};

constexpr size_t kIdatChunkSize = 8192;
constexpr uint32_t kMaxChunkLength = 0x7FFFFFFFu;

uint32_t be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

uint16_t be16(const uint8_t* p) {
    return static_cast<uint16_t>((p[0] << 8) | p[1]);
}

void put_be32(std::vector<uint8_t>* out, uint32_t v) {
    out->push_back(static_cast<uint8_t>(v >> 24));
    out->push_back(static_cast<uint8_t>(v >> 16));
    out->push_back(static_cast<uint8_t>(v >> 8));
    out->push_back(static_cast<uint8_t>(v));
}

bool is_letter(uint8_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }

int channels_for(int color_type) {
    switch (color_type) {
        case 0: return 1;
        case 2: return 3;
        case 3: return 1;
        case 4: return 2;
        case 6: return 4;
        default: return 0;
    }
}

bool valid_depth(int color_type, int depth) {
    switch (color_type) {
        case 0: return depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16;
        case 3: return depth == 1 || depth == 2 || depth == 4 || depth == 8;
        case 2:
        case 4:
        case 6: return depth == 8 || depth == 16;
        default: return false;
    }
}

uint8_t paeth(int a, int b, int c) {
    const int p = a + b - c;
    int pa = p - a;
    int pb = p - b;
    int pc = p - c;
    if (pa < 0) pa = -pa;
    if (pb < 0) pb = -pb;
    if (pc < 0) pc = -pc;
    if (pa <= pb && pa <= pc) return static_cast<uint8_t>(a);
    if (pb <= pc) return static_cast<uint8_t>(b);
    return static_cast<uint8_t>(c);
}

/** Reverse the per-row filters of one (sub)image in place.  @p rows points
 *  at h rows of (1 + rowbytes) bytes; the filter bytes are left in place. */
bool unfilter(uint8_t* rows, size_t rowbytes, size_t h, size_t bpp) {
    const size_t stride = rowbytes + 1;
    for (size_t y = 0; y < h; ++y) {
        uint8_t* row = rows + y * stride;
        const uint8_t filter = row[0];
        uint8_t* cur = row + 1;
        const uint8_t* up = y > 0 ? rows + (y - 1) * stride + 1 : nullptr;
        switch (filter) {
            case 0:
                break;
            case 1:
                for (size_t i = bpp; i < rowbytes; ++i) cur[i] = static_cast<uint8_t>(cur[i] + cur[i - bpp]);
                break;
            case 2:
                if (up) {
                    for (size_t i = 0; i < rowbytes; ++i) cur[i] = static_cast<uint8_t>(cur[i] + up[i]);
                }
                break;
            case 3:
                for (size_t i = 0; i < rowbytes; ++i) {
                    const unsigned left = i >= bpp ? cur[i - bpp] : 0u;
                    const unsigned above = up ? up[i] : 0u;
                    cur[i] = static_cast<uint8_t>(cur[i] + ((left + above) >> 1));
                }
                break;
            case 4:
                for (size_t i = 0; i < rowbytes; ++i) {
                    const int left = i >= bpp ? cur[i - bpp] : 0;
                    const int above = up ? up[i] : 0;
                    const int upleft = (up && i >= bpp) ? up[i - bpp] : 0;
                    cur[i] = static_cast<uint8_t>(cur[i] + paeth(left, above, upleft));
                }
                break;
            default:
                return false;
        }
    }
    return true;
}

struct PngHeader {
    uint32_t width = 0;
    uint32_t height = 0;
    int depth = 0;
    int color_type = 0;
    int interlace = 0;
};

struct Transparency {
    bool present = false;
    uint16_t gray = 0;
    uint16_t red = 0, green = 0, blue = 0;
    uint8_t alpha[256];
    int count = 0;
};

/** Read one raw sample (0 .. 2^depth-1) at sample index @p i of a row. */
inline unsigned sample_at(const uint8_t* row, size_t i, int depth) {
    switch (depth) {
        case 8: return row[i];
        case 16: return (static_cast<unsigned>(row[2 * i]) << 8) | row[2 * i + 1];
        case 4: return (row[i >> 1] >> (4 - 4 * (i & 1))) & 0x0Fu;
        case 2: return (row[i >> 2] >> (6 - 2 * (i & 3))) & 0x03u;
        default: return (row[i >> 3] >> (7 - (i & 7))) & 0x01u;
    }
}

/** Convert one unfiltered row of @p count pixels to 8-bit RGB(A). */
void convert_row(const PngHeader& hdr, const uint8_t* row, size_t count,
                 const uint8_t* palette, int palette_entries, const Transparency& trns,
                 bool has_alpha, uint8_t* dst, size_t dst_step) {
    const int depth = hdr.depth;
    const int out_c = has_alpha ? 4 : 3;
    for (size_t x = 0; x < count; ++x) {
        uint8_t r = 0, g = 0, b = 0, a = 255;
        switch (hdr.color_type) {
            case 0: {
                const unsigned v = sample_at(row, x, depth);
                uint8_t v8;
                if (depth == 16) v8 = static_cast<uint8_t>(v >> 8);
                else if (depth == 8) v8 = static_cast<uint8_t>(v);
                else if (depth == 4) v8 = static_cast<uint8_t>(v * 0x11u);
                else if (depth == 2) v8 = static_cast<uint8_t>(v * 0x55u);
                else v8 = static_cast<uint8_t>(v * 0xFFu);
                r = g = b = v8;
                if (trns.present) {
                    if (depth == 16) {
                        a = v == trns.gray ? 0 : 255;
                    } else {
                        const unsigned mask = (1u << depth) - 1u;
                        unsigned key = trns.gray & mask;
                        if (depth == 8) key = trns.gray & 0xFFu;
                        a = v == key ? 0 : 255;
                    }
                }
                break;
            }
            case 2: {
                const unsigned rv = sample_at(row, 3 * x, depth);
                const unsigned gv = sample_at(row, 3 * x + 1, depth);
                const unsigned bv = sample_at(row, 3 * x + 2, depth);
                if (depth == 16) {
                    r = static_cast<uint8_t>(rv >> 8);
                    g = static_cast<uint8_t>(gv >> 8);
                    b = static_cast<uint8_t>(bv >> 8);
                    if (trns.present) a = (rv == trns.red && gv == trns.green && bv == trns.blue) ? 0 : 255;
                } else {
                    r = static_cast<uint8_t>(rv);
                    g = static_cast<uint8_t>(gv);
                    b = static_cast<uint8_t>(bv);
                    if (trns.present) {
                        a = (rv == (trns.red & 0xFFu) && gv == (trns.green & 0xFFu) &&
                             bv == (trns.blue & 0xFFu)) ? 0 : 255;
                    }
                }
                break;
            }
            case 3: {
                const unsigned idx = sample_at(row, x, depth);
                if (static_cast<int>(idx) < palette_entries) {
                    r = palette[3 * idx];
                    g = palette[3 * idx + 1];
                    b = palette[3 * idx + 2];
                }
                if (trns.present) a = static_cast<int>(idx) < trns.count ? trns.alpha[idx] : 255;
                break;
            }
            case 4: {
                const unsigned v = sample_at(row, 2 * x, depth);
                const unsigned av = sample_at(row, 2 * x + 1, depth);
                r = g = b = static_cast<uint8_t>(depth == 16 ? v >> 8 : v);
                a = static_cast<uint8_t>(depth == 16 ? av >> 8 : av);
                break;
            }
            case 6: {
                const unsigned rv = sample_at(row, 4 * x, depth);
                const unsigned gv = sample_at(row, 4 * x + 1, depth);
                const unsigned bv = sample_at(row, 4 * x + 2, depth);
                const unsigned av = sample_at(row, 4 * x + 3, depth);
                const int shift = depth == 16 ? 8 : 0;
                r = static_cast<uint8_t>(rv >> shift);
                g = static_cast<uint8_t>(gv >> shift);
                b = static_cast<uint8_t>(bv >> shift);
                a = static_cast<uint8_t>(av >> shift);
                break;
            }
            default:
                break;
        }
        uint8_t* d = dst + x * dst_step;
        d[0] = r;
        d[1] = g;
        d[2] = b;
        if (out_c == 4) d[3] = a;
    }
}

size_t row_bytes(uint32_t width, int bits_per_pixel) {
    return (static_cast<size_t>(width) * static_cast<size_t>(bits_per_pixel) + 7u) / 8u;
}

const int kPassXStart[7] = {0, 4, 0, 2, 0, 1, 0};
const int kPassYStart[7] = {0, 0, 4, 0, 2, 0, 1};
const int kPassXStep[7] = {8, 8, 4, 4, 2, 2, 1};
const int kPassYStep[7] = {8, 8, 8, 4, 4, 2, 2};

uint32_t pass_extent(uint32_t size, int start, int step) {
    if (size <= static_cast<uint32_t>(start)) return 0;
    return (size - static_cast<uint32_t>(start) + static_cast<uint32_t>(step) - 1u) /
           static_cast<uint32_t>(step);
}

} // namespace

bool png_is_signature(const uint8_t* data, size_t size) {
    return data && size >= 8 && std::memcmp(data, kSignature, 8) == 0;
}

bool png_decode(const uint8_t* data, size_t size, Raster* out) {
    if (!out || !png_is_signature(data, size)) return false;
    size_t pos = 8;
    PngHeader hdr;
    bool have_header = false;
    bool have_palette = false;
    bool seen_idat = false;
    bool idat_closed = false;
    bool have_end = false;
    uint8_t palette[768] = {0};
    int palette_entries = 0;
    Transparency trns;
    std::vector<uint8_t> compressed;

    while (pos + 12 <= size) {
        const uint32_t length = be32(data + pos);
        const uint8_t* type = data + pos + 4;
        if (length > kMaxChunkLength) return false;
        if (pos + 12 + static_cast<size_t>(length) > size) return false;
        for (int i = 0; i < 4; ++i) {
            if (!is_letter(type[i])) return false;
        }
        const uint8_t* body = data + pos + 8;
        const uint32_t stored_crc = be32(body + length);
        const bool crc_ok = crc32_update(0, type, 4 + static_cast<size_t>(length)) == stored_crc;
        const bool critical = (type[0] & 0x20u) == 0;
        pos += 12 + static_cast<size_t>(length);
        if (critical && !crc_ok) return false;
        if (!critical && !crc_ok) continue;

        if (!have_header) {
            if (std::memcmp(type, "IHDR", 4) != 0) return false;
        }
        if (std::memcmp(type, "IDAT", 4) != 0 && seen_idat) idat_closed = true;

        if (std::memcmp(type, "IHDR", 4) == 0) {
            if (have_header || length != 13) return false;
            hdr.width = be32(body);
            hdr.height = be32(body + 4);
            hdr.depth = body[8];
            hdr.color_type = body[9];
            if (body[10] != 0 || body[11] != 0 || body[12] > 1) return false;
            hdr.interlace = body[12];
            if (hdr.width == 0 || hdr.height == 0) return false;
            if (hdr.width > static_cast<uint32_t>(kMaxDimension) ||
                hdr.height > static_cast<uint32_t>(kMaxDimension)) {
                return false;
            }
            if (channels_for(hdr.color_type) == 0 || !valid_depth(hdr.color_type, hdr.depth)) return false;
            have_header = true;
        } else if (std::memcmp(type, "PLTE", 4) == 0) {
            if (have_palette || seen_idat) return false;
            if (hdr.color_type == 0 || hdr.color_type == 4) continue;  // ignored for gray
            if (length % 3 != 0 || length > 768 || length == 0) {
                if (hdr.color_type == 3) return false;
                continue;
            }
            const int entries = static_cast<int>(length / 3);
            if (hdr.color_type == 3 && entries > (1 << hdr.depth)) return false;
            std::memcpy(palette, body, length);
            palette_entries = entries;
            have_palette = true;
        } else if (std::memcmp(type, "tRNS", 4) == 0) {
            if (seen_idat || trns.present) continue;
            if (hdr.color_type == 3) {
                if (!have_palette || length == 0 || length > static_cast<uint32_t>(palette_entries)) continue;
                std::memset(trns.alpha, 255, sizeof(trns.alpha));
                std::memcpy(trns.alpha, body, length);
                trns.count = static_cast<int>(length);
                trns.present = true;
            } else if (hdr.color_type == 0) {
                if (length != 2) continue;
                trns.gray = be16(body);
                trns.present = true;
            } else if (hdr.color_type == 2) {
                if (length != 6) continue;
                trns.red = be16(body);
                trns.green = be16(body + 2);
                trns.blue = be16(body + 4);
                trns.present = true;
            }
        } else if (std::memcmp(type, "IDAT", 4) == 0) {
            if (hdr.color_type == 3 && !have_palette) return false;
            if (idat_closed) continue;  // a later, separate IDAT run is ignored
            seen_idat = true;
            compressed.insert(compressed.end(), body, body + length);
        } else if (std::memcmp(type, "IEND", 4) == 0) {
            have_end = true;
            break;
        } else if (critical) {
            return false;  // unknown critical chunk
        }
    }
    if (!have_header || !seen_idat || !have_end) return false;

    const int src_channels = channels_for(hdr.color_type);
    const int bits_per_pixel = src_channels * hdr.depth;
    const size_t bpp = static_cast<size_t>((bits_per_pixel + 7) / 8);
    const bool has_alpha = hdr.color_type == 4 || hdr.color_type == 6 || trns.present;
    const int out_c = has_alpha ? 4 : 3;
    const size_t w = hdr.width;
    const size_t h = hdr.height;

    size_t expected = 0;
    if (hdr.interlace == 0) {
        expected = h * (row_bytes(hdr.width, bits_per_pixel) + 1);
    } else {
        for (int p = 0; p < 7; ++p) {
            const uint32_t pw = pass_extent(hdr.width, kPassXStart[p], kPassXStep[p]);
            const uint32_t ph = pass_extent(hdr.height, kPassYStart[p], kPassYStep[p]);
            if (pw == 0 || ph == 0) continue;
            expected += static_cast<size_t>(ph) * (row_bytes(pw, bits_per_pixel) + 1);
        }
    }
    std::vector<uint8_t> raw;
    if (!zlib_inflate_exact(compressed.data(), compressed.size(), expected, &raw)) return false;

    out->width = static_cast<int>(w);
    out->height = static_cast<int>(h);
    out->channels = out_c;
    out->pixels.assign(w * h * static_cast<size_t>(out_c), 0);

    if (hdr.interlace == 0) {
        const size_t rb = row_bytes(hdr.width, bits_per_pixel);
        if (!unfilter(raw.data(), rb, h, bpp)) return false;
        for (size_t y = 0; y < h; ++y) {
            convert_row(hdr, raw.data() + y * (rb + 1) + 1, w, palette, palette_entries, trns, has_alpha,
                        out->pixels.data() + y * w * static_cast<size_t>(out_c), static_cast<size_t>(out_c));
        }
    } else {
        size_t offset = 0;
        std::vector<uint8_t> pass_pixels;
        for (int p = 0; p < 7; ++p) {
            const uint32_t pw = pass_extent(hdr.width, kPassXStart[p], kPassXStep[p]);
            const uint32_t ph = pass_extent(hdr.height, kPassYStart[p], kPassYStep[p]);
            if (pw == 0 || ph == 0) continue;
            const size_t rb = row_bytes(pw, bits_per_pixel);
            uint8_t* rows = raw.data() + offset;
            if (!unfilter(rows, rb, ph, bpp)) return false;
            pass_pixels.assign(static_cast<size_t>(pw) * static_cast<size_t>(out_c), 0);
            for (uint32_t py = 0; py < ph; ++py) {
                convert_row(hdr, rows + static_cast<size_t>(py) * (rb + 1) + 1, pw, palette, palette_entries,
                            trns, has_alpha, pass_pixels.data(), static_cast<size_t>(out_c));
                const size_t y = static_cast<size_t>(kPassYStart[p]) + static_cast<size_t>(py) * kPassYStep[p];
                for (uint32_t px = 0; px < pw; ++px) {
                    const size_t x = static_cast<size_t>(kPassXStart[p]) + static_cast<size_t>(px) * kPassXStep[p];
                    std::memcpy(out->pixels.data() + (y * w + x) * static_cast<size_t>(out_c),
                                pass_pixels.data() + static_cast<size_t>(px) * static_cast<size_t>(out_c),
                                static_cast<size_t>(out_c));
                }
            }
            offset += static_cast<size_t>(ph) * (rb + 1);
        }
    }
    return true;
}

/* ======================================================================== */
/* Encoder                                                                  */
/* ======================================================================== */

namespace {

constexpr unsigned kFilterNone = 0x08;
constexpr unsigned kFilterSub = 0x10;
constexpr unsigned kFilterUp = 0x20;
constexpr unsigned kFilterAvg = 0x40;
constexpr unsigned kFilterPaeth = 0x80;

inline size_t abs_sum_byte(unsigned v) { return v < 128 ? v : 256 - v; }

void write_chunk(std::vector<uint8_t>* out, const char* type, const uint8_t* body, size_t length) {
    put_be32(out, static_cast<uint32_t>(length));
    const size_t type_at = out->size();
    out->insert(out->end(), type, type + 4);
    if (length) out->insert(out->end(), body, body + length);
    const uint32_t crc = crc32_update(0, out->data() + type_at, 4 + length);
    put_be32(out, crc);
}

/** Rewrite the zlib header's window field to the smallest that covers the
 *  image data (the header is advisory for a stream this short). */
void optimize_cmf(uint8_t* data, size_t data_size) {
    if (data_size > 16384) return;
    unsigned z_cmf = data[0];
    if ((z_cmf & 0x0Fu) != 8u || (z_cmf & 0xF0u) > 0x70u) return;
    unsigned z_cinfo = z_cmf >> 4;
    unsigned half_window = 1u << (z_cinfo + 7);
    if (data_size > half_window) return;
    do {
        half_window >>= 1;
        --z_cinfo;
    } while (z_cinfo > 0 && data_size <= half_window);
    z_cmf = (z_cmf & 0x0Fu) | (z_cinfo << 4);
    data[0] = static_cast<uint8_t>(z_cmf);
    unsigned tmp = data[1] & 0xE0u;
    tmp += 0x1Fu - ((z_cmf << 8) + tmp) % 0x1Fu;
    data[1] = static_cast<uint8_t>(tmp);
}

} // namespace

bool png_encode(const uint8_t* pixels, int width, int height, int channels,
                std::vector<uint8_t>* out) {
    if (!pixels || !out || width <= 0 || height <= 0 || width > kMaxDimension ||
        height > kMaxDimension || !(channels == 1 || channels == 3 || channels == 4)) {
        return false;
    }
    const int color_type = channels == 1 ? 0 : (channels == 3 ? 2 : 6);
    const size_t rowbytes = static_cast<size_t>(width) * static_cast<size_t>(channels);
    const size_t bpp = static_cast<size_t>(channels);

    unsigned filters = kFilterNone | kFilterSub | kFilterUp | kFilterAvg | kFilterPaeth;
    if (height == 1) filters &= ~(kFilterUp | kFilterAvg | kFilterPaeth);
    if (width == 1) filters &= ~(kFilterSub | kFilterAvg | kFilterPaeth);
    if (filters == 0) filters = kFilterNone;
    const int strategy = filters != kFilterNone ? 1 : 0;

    size_t data_size = 0xFFFFFFFFu;
    if (rowbytes < 32768 && height < 32768) data_size = (rowbytes + 1) * static_cast<size_t>(height);
    int window_bits = 15;
    if (data_size <= 16384) {
        unsigned half_window = 1u << (window_bits - 1);
        while (data_size + 262 <= half_window) {
            half_window >>= 1;
            --window_bits;
        }
    }

    DeflateEncoder deflater(window_bits, strategy);
    std::vector<uint8_t> prev(rowbytes, 0);
    std::vector<uint8_t> best(rowbytes + 1);
    std::vector<uint8_t> trial(rowbytes + 1);
    std::vector<uint8_t> none_row(rowbytes + 1);
    for (int y = 0; y < height; ++y) {
        const uint8_t* cur = pixels + static_cast<size_t>(y) * rowbytes;
        const uint8_t* up = prev.data();
        none_row[0] = 0;
        std::memcpy(none_row.data() + 1, cur, rowbytes);
        const uint8_t* chosen = none_row.data();
        size_t mins = SIZE_MAX - 256;
        if ((filters & kFilterNone) && filters != kFilterNone) {
            size_t sum = 0;
            for (size_t i = 0; i < rowbytes; ++i) sum += abs_sum_byte(cur[i]);
            mins = sum;
        }
        auto consider = [&](unsigned flag, uint8_t type) {
            if (!(filters & flag)) return;
            trial[0] = type;
            size_t sum = 0;
            for (size_t i = 0; i < rowbytes; ++i) {
                const int left = i >= bpp ? cur[i - bpp] : 0;
                const int above = up[i];
                const int upleft = i >= bpp ? up[i - bpp] : 0;
                int pred = 0;
                switch (type) {
                    case 1: pred = left; break;
                    case 2: pred = above; break;
                    case 3: pred = (left + above) >> 1; break;
                    default: pred = paeth(left, above, upleft); break;
                }
                const unsigned v = static_cast<uint8_t>(cur[i] - pred);
                trial[i + 1] = static_cast<uint8_t>(v);
                sum += abs_sum_byte(v);
            }
            if (filters == flag || sum < mins) {
                mins = sum;
                best.swap(trial);
                chosen = best.data();
            }
        };
        consider(kFilterSub, 1);
        consider(kFilterUp, 2);
        consider(kFilterAvg, 3);
        consider(kFilterPaeth, 4);
        deflater.write(chosen, rowbytes + 1);
        std::memcpy(prev.data(), cur, rowbytes);
    }
    deflater.finish();

    std::vector<uint8_t> stream = deflater.output();
    if (stream.size() >= 2) optimize_cmf(stream.data(), data_size);

    out->clear();
    out->insert(out->end(), kSignature, kSignature + 8);
    uint8_t ihdr[13];
    ihdr[0] = static_cast<uint8_t>(static_cast<uint32_t>(width) >> 24);
    ihdr[1] = static_cast<uint8_t>(static_cast<uint32_t>(width) >> 16);
    ihdr[2] = static_cast<uint8_t>(static_cast<uint32_t>(width) >> 8);
    ihdr[3] = static_cast<uint8_t>(width);
    ihdr[4] = static_cast<uint8_t>(static_cast<uint32_t>(height) >> 24);
    ihdr[5] = static_cast<uint8_t>(static_cast<uint32_t>(height) >> 16);
    ihdr[6] = static_cast<uint8_t>(static_cast<uint32_t>(height) >> 8);
    ihdr[7] = static_cast<uint8_t>(height);
    ihdr[8] = 8;
    ihdr[9] = static_cast<uint8_t>(color_type);
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    write_chunk(out, "IHDR", ihdr, sizeof(ihdr));
    for (size_t off = 0; off < stream.size(); off += kIdatChunkSize) {
        const size_t n = stream.size() - off < kIdatChunkSize ? stream.size() - off : kIdatChunkSize;
        write_chunk(out, "IDAT", stream.data() + off, n);
    }
    write_chunk(out, "IEND", nullptr, 0);
    return true;
}

} // namespace eshkol::image
