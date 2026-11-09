/**
 * @file image_io.c
 * @brief Image I/O for Eshkol tensors, backed by the runtime's own codecs.
 *
 * This file keeps the historical .c path for build-system compatibility, but
 * CMake compiles it as C++.  Decoding and encoding are done by the built-in
 * codecs declared in image_codecs.h (PNG, JPEG, BMP, WebP), identically on every
 * platform; no system or third-party codec library is linked.  This file
 * owns the public C API, the capability checks and the conversion between
 * 8-bit samples and normalized double tensors.
 */

#include "../../inc/eshkol/core/image_io.h"
#include "../../inc/eshkol/runtime_exports.h"
#include "arena_memory.h"
#include "image_codecs.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kMaxImageDimension = 65535;
constexpr int kMaxChannels = 16;

/** Reject non-positive or absurdly large image dimensions/channel counts
 *  (guards against overflow and pathological allocations downstream). */
bool valid_dimensions(int w, int h, int channels) {
    return w > 0 && h > 0 && channels > 0 && channels <= kMaxChannels &&
           w <= kMaxImageDimension && h <= kMaxImageDimension;
}

/** Compute w*h*channels into @p out_total with overflow checks at each
 *  multiplication step (plus a final check against the double-array byte
 *  size); returns false on invalid dimensions or any overflow. */
bool checked_total(int w, int h, int channels, size_t* out_total) {
    if (!valid_dimensions(w, h, channels) || !out_total) {
        return false;
    }
    const size_t sw = static_cast<size_t>(w);
    const size_t sh = static_cast<size_t>(h);
    const size_t sc = static_cast<size_t>(channels);
    const size_t wh = sw * sh;
    if (sw != 0 && wh / sw != sh) {
        return false;
    }
    const size_t total = wh * sc;
    if (wh != 0 && total / wh != sc) {
        return false;
    }
    if (total > std::numeric_limits<size_t>::max() / sizeof(double)) {
        return false;
    }
    *out_total = total;
    return true;
}

/** Allocate an array of @p count doubles from the global arena, or nullptr on
 *  overflow/allocation failure. */
double* arena_doubles(size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(double)) {
        return nullptr;
    }
    return static_cast<double*>(arena_allocate(get_global_arena(), count * sizeof(double)));
}

/** Convert a normalized [0,1] double sample to an 8-bit byte, clamping out-of-range values. */
uint8_t double_to_byte(double value) {
    double scaled = value * 255.0;
    if (!(scaled >= 0.0)) scaled = 0.0;
    if (scaled > 255.0) scaled = 255.0;
    return static_cast<uint8_t>(scaled + 0.5);
}

/** Return an ASCII-lowercased copy of @p s (treats NULL as empty string); used to
 *  normalize format names like "PNG"/"png" before comparison. */
std::string lower_ascii(const char* s) {
    std::string out = s ? s : "";
    for (char& ch : out) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return out;
}

/** Convert a decoded byte pixel buffer into an arena-allocated array of
 *  doubles normalized to [0,1] (used by every native decode path to produce
 *  the tensor representation the Scheme side expects). */
double* bytes_to_arena_doubles(const uint8_t* pixels, int w, int h, int channels) {
    size_t total = 0;
    if (!pixels || !checked_total(w, h, channels, &total)) {
        return nullptr;
    }
    double* result = arena_doubles(total);
    if (!result) {
        return nullptr;
    }
    for (size_t i = 0; i < total; ++i) {
        result[i] = static_cast<double>(pixels[i]) / 255.0;
    }
    return result;
}

/** Convert a normalized [0,1] double tensor into an 8-bit byte buffer @p out,
 *  same channel count/layout as the input (used before handing pixels to
 *  format-specific encoders). Returns false on invalid dimensions. */
bool tensor_to_bytes(const double* data, int w, int h, int channels,
                     std::vector<uint8_t>* out) {
    size_t total = 0;
    if (!data || !out || !checked_total(w, h, channels, &total)) {
        return false;
    }
    out->resize(total);
    for (size_t i = 0; i < total; ++i) {
        (*out)[i] = double_to_byte(data[i]);
    }
    return true;
}

/** Read the entire contents of file @p path into @p out.  Returns false on
 *  any I/O error. */
bool read_file_bytes(const char* path, std::vector<uint8_t>* out) {
    if (!path || !out) return false;
    FILE* fp = eshkol_fopen(path, "rb");
    if (!fp) return false;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return false;
    }
    const long size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return false;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return false;
    }
    out->resize(static_cast<size_t>(size));
    const size_t bytes_read = out->empty() ? 0 : fread(out->data(), 1, out->size(), fp);
    fclose(fp);
    return bytes_read == out->size();
}

/** Write @p bytes to @p path, replacing the file.  Returns false on any I/O error. */
bool write_file_bytes(const char* path, const std::vector<uint8_t>& bytes) {
    FILE* fp = eshkol_fopen(path, "wb");
    if (!fp) return false;
    const size_t written = bytes.empty() ? 0 : fwrite(bytes.data(), 1, bytes.size(), fp);
    const bool closed = fclose(fp) == 0;
    return written == bytes.size() && closed;
}

/**
 * @brief Decode @p path with the built-in codec matching its signature.
 *
 * PNG, JPEG and WebP are recognized by content; BMP by its "BM" header.  The
 * result has 3 (RGB) or 4 (RGBA) channels.  Returns nullptr for unreadable
 * files and unrecognized or malformed data.
 */
double* read_image_native(const char* path, int* out_w, int* out_h, int* out_c) {
    std::vector<uint8_t> encoded;
    if (!read_file_bytes(path, &encoded) || encoded.empty()) return nullptr;
    eshkol::image::Raster raster;
    const uint8_t* bytes = encoded.data();
    const size_t size = encoded.size();
    bool ok = false;
    if (eshkol::image::png_is_signature(bytes, size)) {
        ok = eshkol::image::png_decode(bytes, size, &raster);
    } else if (eshkol::image::jpeg_is_signature(bytes, size)) {
        ok = eshkol::image::jpeg_decode(bytes, size, &raster);
    } else if (eshkol::image::bmp_is_signature(bytes, size)) {
        ok = eshkol::image::bmp_decode(bytes, size, &raster);
    } else if (eshkol::image::webp_is_signature(bytes, size)) {
        ok = eshkol::image::webp_decode(bytes, size, &raster);
    }
    if (!ok || !valid_dimensions(raster.width, raster.height, raster.channels)) return nullptr;
    double* result = bytes_to_arena_doubles(raster.pixels.data(), raster.width, raster.height,
                                            raster.channels);
    if (!result) return nullptr;
    *out_w = raster.width;
    *out_h = raster.height;
    *out_c = raster.channels;
    return result;
}

/**
 * @brief Encode @p data to @p path with the built-in codec named by @p format.
 *
 * "png" (also the default for NULL/empty), "jpg"/"jpeg" (quality 90, alpha
 * dropped), "bmp" and "webp" (lossless).  Channel counts 1, 3 and 4 are
 * accepted.
 *
 * @return 0 on success, -1 on invalid input, an unsupported format or an I/O error.
 */
int write_image_native(const char* path, const double* data,
                       int w, int h, int channels, const char* format) {
    const std::string fmt = lower_ascii(format ? format : "png");
    std::vector<uint8_t> pixels;
    if (!tensor_to_bytes(data, w, h, channels, &pixels)) return -1;
    std::vector<uint8_t> encoded;
    bool ok = false;
    if (fmt.empty() || fmt == "png") {
        ok = eshkol::image::png_encode(pixels.data(), w, h, channels, &encoded);
    } else if (fmt == "jpg" || fmt == "jpeg") {
        ok = eshkol::image::jpeg_encode(pixels.data(), w, h, channels, 90, &encoded);
    } else if (fmt == "bmp") {
        ok = eshkol::image::bmp_encode(pixels.data(), w, h, channels, &encoded);
    } else if (fmt == "webp") {
        ok = eshkol::image::webp_encode_lossless(pixels.data(), w, h, channels, &encoded);
    }
    if (!ok) return -1;
    return write_file_bytes(path, encoded) ? 0 : -1;
}

} // namespace

/**
 * @brief Public entry point: read an image file into a normalized [0,1] double tensor (row-major HWC).
 *
 * Validates arguments, checks the "file-read" capability (denying and
 * returning nullptr if not permitted), then decodes with the built-in codecs.
 *
 * @param[out] out_w Image width.
 * @param[out] out_h Image height.
 * @param[out] out_c Channel count (1=gray, 3=RGB, 4=RGBA).
 * @return Arena-allocated array of w*h*c doubles, or NULL on failure/denial. Do not free().
 */
double* eshkol_image_read(const char* path, int* out_w, int* out_h, int* out_c) {
    if (!path || !out_w || !out_h || !out_c) {
        return nullptr;
    }
    if (eshkol_capability_runtime_allows("file-read") == 0) {
        eshkol_capability_runtime_deny("file-read");
        return nullptr;
    }
    return read_image_native(path, out_w, out_h, out_c);
}

/**
 * @brief Public entry point: write a normalized [0,1] double tensor to an image file.
 *
 * Validates arguments and dimensions, checks the "file-write" capability
 * (denying and returning -1 if not permitted), then encodes with the
 * built-in codecs.
 *
 * @param format "png", "jpg"/"jpeg", "bmp" or "webp"; NULL defaults to "png".
 * @return 0 on success, -1 on failure or capability denial.
 */
int eshkol_image_write(const char* path, const double* data,
                       int w, int h, int channels, const char* format) {
    if (!path || !data || !valid_dimensions(w, h, channels)) {
        return -1;
    }
    if (eshkol_capability_runtime_allows("file-write") == 0) {
        eshkol_capability_runtime_deny("file-write");
        return -1;
    }
    return write_image_native(path, data, w, h, channels, format);
}

/**
 * @brief Convert a color image tensor to single-channel grayscale using ITU-R BT.709 luma weights.
 *
 * If @p channels is already 1, just copies the data. Otherwise computes
 * 0.2126*R + 0.7152*G + 0.0722*B per pixel (missing G/B channels fall back
 * to R, matching the other helpers' single/dual-channel handling).
 *
 * @return Arena-allocated array of w*h doubles, or NULL on invalid input/allocation failure.
 */
double* eshkol_image_to_grayscale(const double* data, int w, int h, int channels) {
    if (!data || !valid_dimensions(w, h, channels)) {
        return nullptr;
    }
    const size_t pixels = static_cast<size_t>(w) * static_cast<size_t>(h);
    double* result = arena_doubles(pixels);
    if (!result) return nullptr;
    if (channels == 1) {
        memcpy(result, data, pixels * sizeof(double));
        return result;
    }
    for (size_t i = 0; i < pixels; ++i) {
        const size_t base = i * static_cast<size_t>(channels);
        const double r = data[base];
        const double g = channels >= 2 ? data[base + 1] : r;
        const double b = channels >= 3 ? data[base + 2] : r;
        result[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    return result;
}

/**
 * @brief Resize an image tensor to (new_w, new_h) using bilinear interpolation.
 *
 * For each destination pixel, maps back to source coordinates (half-pixel
 * center convention), clamps to the source bounds, and blends the four
 * surrounding source pixels per channel.
 *
 * @return Arena-allocated array of new_w*new_h*channels doubles, or NULL on invalid dimensions/allocation failure.
 */
double* eshkol_image_resize(const double* data, int w, int h, int channels,
                            int new_w, int new_h) {
    size_t src_total = 0;
    size_t dst_total = 0;
    if (!data || !checked_total(w, h, channels, &src_total) ||
        !checked_total(new_w, new_h, channels, &dst_total)) {
        return nullptr;
    }
    double* result = arena_doubles(dst_total);
    if (!result) return nullptr;

    const double scale_x = static_cast<double>(w) / static_cast<double>(new_w);
    const double scale_y = static_cast<double>(h) / static_cast<double>(new_h);
    const size_t csz = static_cast<size_t>(channels);
    for (int y = 0; y < new_h; ++y) {
        const double sy = (static_cast<double>(y) + 0.5) * scale_y - 0.5;
        const int y0 = std::max(0, static_cast<int>(std::floor(sy)));
        const int y1 = std::min(h - 1, y0 + 1);
        const double fy = std::clamp(sy - static_cast<double>(y0), 0.0, 1.0);
        for (int x = 0; x < new_w; ++x) {
            const double sx = (static_cast<double>(x) + 0.5) * scale_x - 0.5;
            const int x0 = std::max(0, static_cast<int>(std::floor(sx)));
            const int x1 = std::min(w - 1, x0 + 1);
            const double fx = std::clamp(sx - static_cast<double>(x0), 0.0, 1.0);
            for (int ch = 0; ch < channels; ++ch) {
                const size_t p00 = (static_cast<size_t>(y0) * static_cast<size_t>(w) +
                                    static_cast<size_t>(x0)) * csz + static_cast<size_t>(ch);
                const size_t p10 = (static_cast<size_t>(y0) * static_cast<size_t>(w) +
                                    static_cast<size_t>(x1)) * csz + static_cast<size_t>(ch);
                const size_t p01 = (static_cast<size_t>(y1) * static_cast<size_t>(w) +
                                    static_cast<size_t>(x0)) * csz + static_cast<size_t>(ch);
                const size_t p11 = (static_cast<size_t>(y1) * static_cast<size_t>(w) +
                                    static_cast<size_t>(x1)) * csz + static_cast<size_t>(ch);
                const double top = data[p00] * (1.0 - fx) + data[p10] * fx;
                const double bottom = data[p01] * (1.0 - fx) + data[p11] * fx;
                const size_t dst = (static_cast<size_t>(y) * static_cast<size_t>(new_w) +
                                    static_cast<size_t>(x)) * csz + static_cast<size_t>(ch);
                result[dst] = top * (1.0 - fy) + bottom * fy;
            }
        }
    }
    return result;
}
