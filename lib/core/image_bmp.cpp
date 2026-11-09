/**
 * @file image_bmp.cpp
 * @brief Built-in BMP (Windows bitmap) decoder and encoder.
 *
 * Decoding accepts OS/2 core headers and Windows info headers (versions 3, 4
 * and 5), bottom-up and top-down rows, 1/4/8-bit palettes, RLE4/RLE8, 16- and
 * 32-bit bit fields and 24-bit BGR.  Images with an alpha bit-field decode to
 * RGBA, all others to RGB.
 *
 * Encoding writes 24-bit BGR for gray and RGB input, and 32-bit BGRA with a
 * version 4 header and explicit bit-field masks for RGBA input, so the alpha
 * channel survives a write/read round trip.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "image_codecs.h"

#include <cstring>

namespace eshkol::image {

namespace {

uint32_t le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint16_t le16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

void put_le32(std::vector<uint8_t>* out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out->push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void put_le16(std::vector<uint8_t>* out, uint16_t v) {
    out->push_back(static_cast<uint8_t>(v));
    out->push_back(static_cast<uint8_t>(v >> 8));
}

/** Extract a channel described by a bit mask and scale it to 8 bits. */
struct MaskChannel {
    uint32_t mask = 0;
    int shift = 0;
    int bits = 0;
    explicit MaskChannel(uint32_t m = 0) : mask(m) {
        if (!m) return;
        while (!((m >> shift) & 1u)) ++shift;
        while (shift + bits < 32 && ((m >> (shift + bits)) & 1u)) ++bits;
    }
    uint8_t extract(uint32_t v) const {
        if (!mask) return 0;
        const uint32_t raw = (v & mask) >> shift;
        if (bits >= 8) return static_cast<uint8_t>(raw >> (bits - 8));
        const uint32_t max = (1u << bits) - 1u;
        return static_cast<uint8_t>((raw * 255u + max / 2u) / max);
    }
};

} // namespace

bool bmp_is_signature(const uint8_t* data, size_t size) {
    return data && size >= 18 && data[0] == 'B' && data[1] == 'M';
}

bool bmp_decode(const uint8_t* data, size_t size, Raster* out) {
    if (!out || !bmp_is_signature(data, size)) return false;
    const uint32_t pixel_offset = le32(data + 10);
    const uint32_t header_size = le32(data + 14);
    if (header_size < 12 || 14 + static_cast<size_t>(header_size) > size) return false;
    const uint8_t* h = data + 14;
    int64_t width = 0;
    int64_t height = 0;
    int bpp = 0;
    uint32_t compression = 0;
    uint32_t colors_used = 0;
    uint32_t masks[4] = {0, 0, 0, 0};
    bool have_masks = false;
    size_t palette_entry = 4;
    if (header_size == 12) {
        width = le16(h + 4);
        height = static_cast<int16_t>(le16(h + 6));
        bpp = le16(h + 10);
        palette_entry = 3;
    } else {
        if (header_size < 40) return false;
        width = static_cast<int32_t>(le32(h + 4));
        height = static_cast<int32_t>(le32(h + 8));
        bpp = le16(h + 14);
        compression = le32(h + 16);
        colors_used = le32(h + 32);
        if (header_size >= 52) {
            masks[0] = le32(h + 40);
            masks[1] = le32(h + 44);
            masks[2] = le32(h + 48);
            have_masks = true;
        }
        if (header_size >= 56) masks[3] = le32(h + 52);
    }
    size_t after_header = 14 + static_cast<size_t>(header_size);
    if ((compression == 3 || compression == 6) && header_size == 40) {
        // Bit-field masks follow a version 3 header.
        const size_t n = compression == 6 ? 4 : 3;
        if (after_header + n * 4 > size) return false;
        for (size_t i = 0; i < n; ++i) masks[i] = le32(data + after_header + i * 4);
        after_header += n * 4;
        have_masks = true;
    }
    const bool top_down = height < 0;
    if (top_down) height = -height;
    if (width <= 0 || height <= 0 || width > kMaxDimension || height > kMaxDimension) return false;
    if (!(bpp == 1 || bpp == 4 || bpp == 8 || bpp == 16 || bpp == 24 || bpp == 32)) return false;
    if (!(compression == 0 || compression == 1 || compression == 2 || compression == 3 || compression == 6)) {
        return false;
    }
    if (compression == 1 && bpp != 8) return false;
    if (compression == 2 && bpp != 4) return false;
    if ((compression == 3 || compression == 6) && !(bpp == 16 || bpp == 32)) return false;

    // Palette
    std::vector<uint8_t> palette;  // RGB triples
    if (bpp <= 8) {
        uint32_t entries = colors_used ? colors_used : (1u << bpp);
        if (entries > 256) entries = 256;
        const size_t pal_at = after_header;
        const size_t avail = pixel_offset > pal_at ? (pixel_offset - pal_at) / palette_entry : 0;
        if (entries > avail) entries = static_cast<uint32_t>(avail);
        if (pal_at + entries * palette_entry > size) return false;
        palette.assign(256 * 3, 0);
        for (uint32_t i = 0; i < entries; ++i) {
            const uint8_t* e = data + pal_at + i * palette_entry;
            palette[i * 3 + 0] = e[2];
            palette[i * 3 + 1] = e[1];
            palette[i * 3 + 2] = e[0];
        }
    }

    if (!have_masks || compression == 0) {
        if (bpp == 16) {
            masks[0] = 0x7C00;
            masks[1] = 0x03E0;
            masks[2] = 0x001F;
            masks[3] = 0;
        } else if (bpp == 32) {
            masks[0] = 0x00FF0000;
            masks[1] = 0x0000FF00;
            masks[2] = 0x000000FF;
            masks[3] = 0;
        }
    }
    const bool has_alpha = (bpp == 16 || bpp == 32) && masks[3] != 0;
    const int out_c = has_alpha ? 4 : 3;
    const size_t W = static_cast<size_t>(width);
    const size_t H = static_cast<size_t>(height);
    out->width = static_cast<int>(W);
    out->height = static_cast<int>(H);
    out->channels = out_c;
    out->pixels.assign(W * H * static_cast<size_t>(out_c), 0);
    if (has_alpha) {
        for (size_t i = 0; i < W * H; ++i) out->pixels[i * 4 + 3] = 255;
    }
    if (pixel_offset >= size) return false;
    const uint8_t* pix = data + pixel_offset;
    const size_t pix_size = size - pixel_offset;
    auto dst_row = [&](size_t file_row) -> uint8_t* {
        const size_t y = top_down ? file_row : H - 1 - file_row;
        return out->pixels.data() + y * W * static_cast<size_t>(out_c);
    };
    auto put_index = [&](uint8_t* row, size_t x, unsigned idx) {
        uint8_t* d = row + x * static_cast<size_t>(out_c);
        d[0] = palette[idx * 3];
        d[1] = palette[idx * 3 + 1];
        d[2] = palette[idx * 3 + 2];
    };

    if (compression == 1 || compression == 2) {
        // Run-length encoded palette data; unspecified pixels stay index 0.
        for (size_t y = 0; y < H; ++y) {
            uint8_t* row = dst_row(y);
            for (size_t x = 0; x < W; ++x) put_index(row, x, 0);
        }
        size_t p = 0;
        size_t x = 0;
        size_t y = 0;
        while (p + 1 < pix_size && y < H) {
            const uint8_t count = pix[p];
            const uint8_t value = pix[p + 1];
            p += 2;
            if (count > 0) {
                for (unsigned i = 0; i < count && x < W; ++i, ++x) {
                    const unsigned idx = compression == 1 ? value : ((i & 1) ? (value & 0x0F) : (value >> 4));
                    put_index(dst_row(y), x, idx);
                }
                continue;
            }
            if (value == 0) {
                x = 0;
                ++y;
            } else if (value == 1) {
                break;
            } else if (value == 2) {
                if (p + 1 >= pix_size) break;
                x += pix[p];
                y += pix[p + 1];
                p += 2;
            } else {
                const unsigned n = value;
                const size_t bytes = compression == 1 ? n : (n + 1) / 2;
                if (p + bytes > pix_size) return false;
                for (unsigned i = 0; i < n; ++i, ++x) {
                    const unsigned idx = compression == 1 ? pix[p + i]
                                                          : ((i & 1) ? (pix[p + i / 2] & 0x0F) : (pix[p + i / 2] >> 4));
                    if (x < W && y < H) put_index(dst_row(y), x, idx);
                }
                p += (bytes + 1) & ~static_cast<size_t>(1);
            }
        }
        return true;
    }

    const size_t stride = ((W * static_cast<size_t>(bpp) + 31) / 32) * 4;
    if (stride * H > pix_size) return false;
    const MaskChannel r(masks[0]), g(masks[1]), b(masks[2]), a(masks[3]);
    for (size_t fy = 0; fy < H; ++fy) {
        const uint8_t* src = pix + fy * stride;
        uint8_t* row = dst_row(fy);
        for (size_t x = 0; x < W; ++x) {
            uint8_t* d = row + x * static_cast<size_t>(out_c);
            switch (bpp) {
                case 1: put_index(row, x, (src[x >> 3] >> (7 - (x & 7))) & 1u); break;
                case 4: put_index(row, x, (src[x >> 1] >> (4 - 4 * (x & 1))) & 0x0Fu); break;
                case 8: put_index(row, x, src[x]); break;
                case 24:
                    d[0] = src[x * 3 + 2];
                    d[1] = src[x * 3 + 1];
                    d[2] = src[x * 3];
                    break;
                default: {
                    const uint32_t v = bpp == 16 ? le16(src + x * 2) : le32(src + x * 4);
                    d[0] = r.extract(v);
                    d[1] = g.extract(v);
                    d[2] = b.extract(v);
                    if (has_alpha) d[3] = a.extract(v);
                    break;
                }
            }
        }
    }
    return true;
}

bool bmp_encode(const uint8_t* pixels, int width, int height, int channels,
                std::vector<uint8_t>* out) {
    if (!pixels || !out || width <= 0 || height <= 0 || width > kMaxDimension ||
        height > kMaxDimension || !(channels == 1 || channels == 3 || channels == 4)) {
        return false;
    }
    const bool alpha = channels == 4;
    const uint32_t header_size = alpha ? 108 : 40;
    const int bpp = alpha ? 32 : 24;
    const size_t W = static_cast<size_t>(width);
    const size_t H = static_cast<size_t>(height);
    const size_t stride = ((W * static_cast<size_t>(bpp) + 31) / 32) * 4;
    const size_t image_size = stride * H;
    const uint32_t offset = 14 + header_size;
    out->clear();
    out->reserve(offset + image_size);
    out->push_back('B');
    out->push_back('M');
    put_le32(out, static_cast<uint32_t>(offset + image_size));
    put_le32(out, 0);
    put_le32(out, offset);
    put_le32(out, header_size);
    put_le32(out, static_cast<uint32_t>(width));
    put_le32(out, static_cast<uint32_t>(height));  // bottom-up
    put_le16(out, 1);
    put_le16(out, static_cast<uint16_t>(bpp));
    put_le32(out, alpha ? 3u : 0u);
    put_le32(out, static_cast<uint32_t>(image_size));
    put_le32(out, 2835);  // 72 DPI
    put_le32(out, 2835);
    put_le32(out, 0);
    put_le32(out, 0);
    if (alpha) {
        put_le32(out, 0x00FF0000u);
        put_le32(out, 0x0000FF00u);
        put_le32(out, 0x000000FFu);
        put_le32(out, 0xFF000000u);
        put_le32(out, 0x73524742u);  // 'sRGB'
        for (int i = 0; i < 12; ++i) put_le32(out, 0);  // endpoints + gamma
    }
    std::vector<uint8_t> row(stride, 0);
    for (size_t fy = 0; fy < H; ++fy) {
        const size_t y = H - 1 - fy;
        const uint8_t* src = pixels + y * W * static_cast<size_t>(channels);
        for (size_t x = 0; x < W; ++x) {
            const uint8_t* s = src + x * static_cast<size_t>(channels);
            const uint8_t rr = s[0];
            const uint8_t gg = channels == 1 ? s[0] : s[1];
            const uint8_t bb = channels == 1 ? s[0] : s[2];
            if (alpha) {
                row[x * 4 + 0] = bb;
                row[x * 4 + 1] = gg;
                row[x * 4 + 2] = rr;
                row[x * 4 + 3] = s[3];
            } else {
                row[x * 3 + 0] = bb;
                row[x * 3 + 1] = gg;
                row[x * 3 + 2] = rr;
            }
        }
        out->insert(out->end(), row.begin(), row.end());
    }
    return true;
}

/* WebP support is Planned (see image_io.c); only the container check lives
 * here so the reader can report the format precisely. */
bool webp_is_signature(const uint8_t* data, size_t size) {
    return data && size >= 12 && std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WEBP", 4) == 0;
}

} // namespace eshkol::image
