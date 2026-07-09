/**
 * @file image_io.c
 * @brief Native image I/O for Eshkol tensors.
 *
 * This file keeps the historical .c path for build-system compatibility, but
 * CMake compiles it as C++ so the runtime can call platform/system image APIs.
 * It uses ImageIO/CoreGraphics on macOS, GDI+ on Windows, and libpng/libjpeg/
 * libwebp on Linux/Unix. No vendored image decoder is used.
 */

#include "../../inc/eshkol/core/image_io.h"
#include "../../inc/eshkol/runtime_exports.h"
#include "arena_memory.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#if defined(ESHKOL_IMAGE_IO_APPLE)
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#endif

#if defined(ESHKOL_IMAGE_IO_GDIPLUS)
#define NOMINMAX
#include <windows.h>
#include <gdiplus.h>
#include <mutex>
#pragma comment(lib, "gdiplus.lib")
#endif

#if defined(ESHKOL_IMAGE_IO_LIBPNG)
#include <png.h>
#endif

#if defined(ESHKOL_IMAGE_IO_LIBJPEG)
#include <setjmp.h>
#include <jpeglib.h>
#endif

#if defined(ESHKOL_IMAGE_IO_LIBWEBP)
#include <webp/decode.h>
#include <webp/encode.h>
#endif

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

/**
 * @brief Convert a normalized double tensor of any supported channel count into 8-bit RGBA bytes.
 *
 * Missing channels are filled in (grayscale replicated to R/G/B, alpha
 * defaults to 255), producing a 4-channel byte buffer regardless of the
 * source channel count. If @p premultiply is set, RGB is premultiplied by
 * alpha (needed by APIs like CoreGraphics that expect premultiplied alpha).
 *
 * @param premultiply Whether to premultiply RGB by alpha in the output.
 * @param[out] out Resulting RGBA byte buffer, resized to w*h*4.
 * @return false on invalid dimensions or pixel-count overflow.
 */
bool tensor_to_rgba_bytes(const double* data, int w, int h, int channels,
                          bool premultiply, std::vector<uint8_t>* out) {
    size_t ignored = 0;
    if (!data || !out || !checked_total(w, h, channels, &ignored)) {
        return false;
    }
    const size_t pixels = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (pixels > std::numeric_limits<size_t>::max() / 4) {
        return false;
    }
    out->assign(pixels * 4, 255);
    for (size_t i = 0; i < pixels; ++i) {
        const size_t src = i * static_cast<size_t>(channels);
        const uint8_t r = double_to_byte(data[src]);
        const uint8_t g = channels >= 2 ? double_to_byte(data[src + 1]) : r;
        const uint8_t b = channels >= 3 ? double_to_byte(data[src + 2]) : r;
        const uint8_t a = channels >= 4 ? double_to_byte(data[src + 3]) : 255;
        const size_t dst = i * 4;
        if (premultiply) {
            (*out)[dst + 0] = static_cast<uint8_t>((static_cast<unsigned>(r) * a + 127) / 255);
            (*out)[dst + 1] = static_cast<uint8_t>((static_cast<unsigned>(g) * a + 127) / 255);
            (*out)[dst + 2] = static_cast<uint8_t>((static_cast<unsigned>(b) * a + 127) / 255);
        } else {
            (*out)[dst + 0] = r;
            (*out)[dst + 1] = g;
            (*out)[dst + 2] = b;
        }
        (*out)[dst + 3] = a;
    }
    return true;
}

/**
 * @brief Convert a decoded 4-channel RGBA byte buffer to the desired output channel layout.
 *
 * Drops the alpha channel if @p has_alpha is false. If @p unpremultiply is
 * set and a pixel's alpha is neither 0 nor 255, reverses alpha
 * premultiplication on R/G/B (needed after decoding from APIs, like
 * CoreGraphics, that hand back premultiplied pixels).
 *
 * @return Byte buffer with 3 (RGB) or 4 (RGBA) channels per pixel, per @p has_alpha.
 */
std::vector<uint8_t> rgba_to_output_bytes(const uint8_t* rgba, int w, int h,
                                          bool has_alpha, bool unpremultiply) {
    const int channels = has_alpha ? 4 : 3;
    std::vector<uint8_t> out(static_cast<size_t>(w) * static_cast<size_t>(h) *
                             static_cast<size_t>(channels));
    const size_t pixels = static_cast<size_t>(w) * static_cast<size_t>(h);
    for (size_t i = 0; i < pixels; ++i) {
        const size_t src = i * 4;
        const size_t dst = i * static_cast<size_t>(channels);
        uint8_t r = rgba[src + 0];
        uint8_t g = rgba[src + 1];
        uint8_t b = rgba[src + 2];
        const uint8_t a = rgba[src + 3];
        if (has_alpha && unpremultiply && a > 0 && a < 255) {
            r = static_cast<uint8_t>(std::min(255u, (static_cast<unsigned>(r) * 255u + a / 2u) / a));
            g = static_cast<uint8_t>(std::min(255u, (static_cast<unsigned>(g) * 255u + a / 2u) / a));
            b = static_cast<uint8_t>(std::min(255u, (static_cast<unsigned>(b) * 255u + a / 2u) / a));
        }
        out[dst + 0] = r;
        out[dst + 1] = g;
        out[dst + 2] = b;
        if (has_alpha) {
            out[dst + 3] = a;
        }
    }
    return out;
}

#if defined(ESHKOL_IMAGE_IO_APPLE)

/** Determine whether a CGImageAlphaInfo value indicates the image carries an alpha channel. */
bool apple_alpha_info_has_alpha(CGImageAlphaInfo info) {
    switch (info & kCGBitmapAlphaInfoMask) {
        case kCGImageAlphaPremultipliedLast:
        case kCGImageAlphaPremultipliedFirst:
        case kCGImageAlphaLast:
        case kCGImageAlphaFirst:
            return true;
        default:
            return false;
    }
}

/** Build a CFURLRef for a local file @p path (UTF-8, POSIX style); caller owns the returned reference. */
CFURLRef apple_file_url(const char* path) {
    CFStringRef cf_path = CFStringCreateWithCString(kCFAllocatorDefault, path,
                                                     kCFStringEncodingUTF8);
    if (!cf_path) return nullptr;
    CFURLRef url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, cf_path,
                                                  kCFURLPOSIXPathStyle, false);
    CFRelease(cf_path);
    return url;
}

/**
 * @brief Apple (ImageIO/CoreGraphics) backend: decode an image file into a normalized double tensor.
 *
 * Loads @p path via CGImageSource, draws it into an RGBA8 bitmap context
 * (premultiplied alpha), then converts to the output channel layout (RGB or
 * RGBA depending on whether the source has alpha), un-premultiplying and
 * normalizing to [0,1] doubles.
 *
 * @param[out] out_w Image width.
 * @param[out] out_h Image height.
 * @param[out] out_c Channel count (3 or 4).
 * @return Arena-allocated double array, or nullptr on any failure.
 */
double* read_image_native(const char* path, int* out_w, int* out_h, int* out_c) {
    CFURLRef url = apple_file_url(path);
    if (!url) return nullptr;
    CGImageSourceRef source = CGImageSourceCreateWithURL(url, nullptr);
    CFRelease(url);
    if (!source) return nullptr;
    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (!image) return nullptr;

    const int w = static_cast<int>(CGImageGetWidth(image));
    const int h = static_cast<int>(CGImageGetHeight(image));
    const bool has_alpha = apple_alpha_info_has_alpha(CGImageGetAlphaInfo(image));
    const int channels = has_alpha ? 4 : 3;
    if (!valid_dimensions(w, h, channels)) {
        CGImageRelease(image);
        return nullptr;
    }

    std::vector<uint8_t> rgba(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        rgba.data(), static_cast<size_t>(w), static_cast<size_t>(h), 8,
        static_cast<size_t>(w) * 4, color_space,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color_space);
    if (!ctx) {
        CGImageRelease(image);
        return nullptr;
    }
    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), image);
    CGContextRelease(ctx);
    CGImageRelease(image);

    std::vector<uint8_t> pixels = rgba_to_output_bytes(rgba.data(), w, h, has_alpha, true);
    double* result = bytes_to_arena_doubles(pixels.data(), w, h, channels);
    if (!result) return nullptr;
    *out_w = w;
    *out_h = h;
    *out_c = channels;
    return result;
}

/** Map a lowercase format name ("png", "jpg"/"jpeg", "bmp", "tif"/"tiff", "webp")
 *  to its ImageIO UTI, or nullptr if unrecognized. */
CFStringRef apple_type_for_format(const std::string& format) {
    if (format.empty() || format == "png") return CFSTR("public.png");
    if (format == "jpg" || format == "jpeg") return CFSTR("public.jpeg");
    if (format == "bmp") return CFSTR("com.microsoft.bmp");
    if (format == "tif" || format == "tiff") return CFSTR("public.tiff");
    if (format == "webp") return CFSTR("org.webmproject.webp");
    return nullptr;
}

/**
 * @brief Apple (ImageIO/CoreGraphics) backend: encode a normalized double tensor to an image file.
 *
 * Converts @p data to premultiplied RGBA bytes, wraps them in a CGImage, and
 * writes it to @p path using the ImageIO destination API for the UTI
 * resolved from @p format (defaults to "png" if NULL/empty).
 *
 * @return 0 on success, -1 on any failure (unsupported format, bad dimensions, I/O error).
 */
int write_image_native(const char* path, const double* data,
                       int w, int h, int channels, const char* format) {
    const std::string fmt = lower_ascii(format ? format : "png");
    CFStringRef type = apple_type_for_format(fmt);
    if (!type) return -1;

    std::vector<uint8_t> rgba;
    if (!tensor_to_rgba_bytes(data, w, h, channels, true, &rgba)) return -1;

    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, rgba.data(),
                                                              rgba.size(), nullptr);
    if (!color_space || !provider) {
        if (provider) CGDataProviderRelease(provider);
        if (color_space) CGColorSpaceRelease(color_space);
        return -1;
    }
    CGImageRef image = CGImageCreate(static_cast<size_t>(w), static_cast<size_t>(h),
                                     8, 32, static_cast<size_t>(w) * 4,
                                     color_space,
                                     kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big,
                                     provider, nullptr, false, kCGRenderingIntentDefault);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(color_space);
    if (!image) return -1;

    CFURLRef url = apple_file_url(path);
    if (!url) {
        CGImageRelease(image);
        return -1;
    }
    CGImageDestinationRef dest = CGImageDestinationCreateWithURL(url, type, 1, nullptr);
    CFRelease(url);
    if (!dest) {
        CGImageRelease(image);
        return -1;
    }
    CGImageDestinationAddImage(dest, image, nullptr);
    const bool ok = CGImageDestinationFinalize(dest);
    CFRelease(dest);
    CGImageRelease(image);
    return ok ? 0 : -1;
}

#elif defined(ESHKOL_IMAGE_IO_GDIPLUS)

ULONG_PTR gdiplus_token = 0;
std::once_flag gdiplus_once;

/** One-time GDI+ startup (invoked via std::call_once from ensure_gdiplus()). */
void init_gdiplus_once() {
    Gdiplus::GdiplusStartupInput input;
    Gdiplus::GdiplusStartup(&gdiplus_token, &input, nullptr);
}

/** Lazily initialize GDI+ exactly once; returns true if a valid startup token was obtained. */
bool ensure_gdiplus() {
    std::call_once(gdiplus_once, init_gdiplus_once);
    return gdiplus_token != 0;
}

/** Convert a UTF-8 (falling back to the system ANSI code page) @p path to a wide string for Win32 APIs. */
std::wstring utf8_to_wide(const char* path) {
    if (!path) return std::wstring();
    int len = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
    UINT code_page = CP_UTF8;
    if (len <= 0) {
        len = MultiByteToWideChar(CP_ACP, 0, path, -1, nullptr, 0);
        code_page = CP_ACP;
    }
    if (len <= 0) return std::wstring();
    std::wstring out(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(code_page, 0, path, -1, out.data(), len);
    if (!out.empty() && out.back() == L'\0') out.pop_back();
    return out;
}

/** Look up the GDI+ image encoder CLSID matching @p mime_type, writing it to @p clsid.
 *  @return 0 on success, -1 if no matching encoder is registered. */
int get_encoder_clsid(const WCHAR* mime_type, CLSID* clsid) {
    UINT num = 0;
    UINT size = 0;
    Gdiplus::GetImageEncodersSize(&num, &size);
    if (size == 0) return -1;
    std::vector<uint8_t> buffer(size);
    auto* info = reinterpret_cast<Gdiplus::ImageCodecInfo*>(buffer.data());
    if (Gdiplus::GetImageEncoders(num, size, info) != Gdiplus::Ok) return -1;
    for (UINT i = 0; i < num; ++i) {
        if (wcscmp(info[i].MimeType, mime_type) == 0) {
            *clsid = info[i].Clsid;
            return 0;
        }
    }
    return -1;
}

/** Map a lowercase format name ("png", "jpg"/"jpeg", "bmp", "gif") to its MIME
 *  type string for GDI+ encoder lookup, or nullptr if unrecognized. */
const WCHAR* mime_for_format(const std::string& format) {
    if (format.empty() || format == "png") return L"image/png";
    if (format == "jpg" || format == "jpeg") return L"image/jpeg";
    if (format == "bmp") return L"image/bmp";
    if (format == "gif") return L"image/gif";
    return nullptr;
}

/**
 * @brief GDI+ backend: decode an image file into a normalized double tensor.
 *
 * Loads @p path via Gdiplus::Bitmap and reads it back pixel-by-pixel with
 * GetPixel(), converting each channel to [0,1]. Output channel count is 4
 * (RGBA) if the source has an alpha flag, else 3 (RGB).
 *
 * @param[out] out_w Image width.
 * @param[out] out_h Image height.
 * @param[out] out_c Channel count (3 or 4).
 * @return Arena-allocated double array, or nullptr on any failure.
 */
double* read_image_native(const char* path, int* out_w, int* out_h, int* out_c) {
    if (!ensure_gdiplus()) return nullptr;
    std::wstring wide_path = utf8_to_wide(path);
    if (wide_path.empty()) return nullptr;
    Gdiplus::Bitmap bitmap(wide_path.c_str(), false);
    if (bitmap.GetLastStatus() != Gdiplus::Ok) return nullptr;

    const int w = static_cast<int>(bitmap.GetWidth());
    const int h = static_cast<int>(bitmap.GetHeight());
    const bool has_alpha = (bitmap.GetFlags() & Gdiplus::ImageFlagsHasAlpha) != 0;
    const int channels = has_alpha ? 4 : 3;
    if (!valid_dimensions(w, h, channels)) return nullptr;

    std::vector<uint8_t> pixels(static_cast<size_t>(w) * static_cast<size_t>(h) *
                                static_cast<size_t>(channels));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            Gdiplus::Color color;
            if (bitmap.GetPixel(x, y, &color) != Gdiplus::Ok) return nullptr;
            const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(w) +
                                static_cast<size_t>(x)) * static_cast<size_t>(channels);
            pixels[idx + 0] = color.GetR();
            pixels[idx + 1] = color.GetG();
            pixels[idx + 2] = color.GetB();
            if (has_alpha) pixels[idx + 3] = color.GetA();
        }
    }
    double* result = bytes_to_arena_doubles(pixels.data(), w, h, channels);
    if (!result) return nullptr;
    *out_w = w;
    *out_h = h;
    *out_c = channels;
    return result;
}

/**
 * @brief GDI+ backend: encode a normalized double tensor to an image file.
 *
 * Builds a 32bpp ARGB Gdiplus::Bitmap pixel-by-pixel from @p data, then
 * saves it via the encoder CLSID resolved from @p format's MIME type
 * (defaults to "png"); uses quality 90 for JPEG output.
 *
 * @return 0 on success, -1 on any failure (GDI+ not available, unsupported format, bad dimensions).
 */
int write_image_native(const char* path, const double* data,
                       int w, int h, int channels, const char* format) {
    if (!ensure_gdiplus()) return -1;
    const std::string fmt = lower_ascii(format ? format : "png");
    const WCHAR* mime = mime_for_format(fmt);
    if (!mime) return -1;
    CLSID clsid;
    if (get_encoder_clsid(mime, &clsid) != 0) return -1;
    std::wstring wide_path = utf8_to_wide(path);
    if (wide_path.empty() || !valid_dimensions(w, h, channels)) return -1;

    Gdiplus::Bitmap bitmap(w, h, PixelFormat32bppARGB);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(w) +
                                static_cast<size_t>(x)) * static_cast<size_t>(channels);
            const uint8_t r = double_to_byte(data[idx + 0]);
            const uint8_t g = channels >= 2 ? double_to_byte(data[idx + 1]) : r;
            const uint8_t b = channels >= 3 ? double_to_byte(data[idx + 2]) : r;
            const uint8_t a = channels >= 4 ? double_to_byte(data[idx + 3]) : 255;
            if (bitmap.SetPixel(x, y, Gdiplus::Color(a, r, g, b)) != Gdiplus::Ok) return -1;
        }
    }
    Gdiplus::EncoderParameters params;
    ULONG quality = 90;
    params.Count = 1;
    params.Parameter[0].Guid = Gdiplus::EncoderQuality;
    params.Parameter[0].Type = Gdiplus::EncoderParameterValueTypeLong;
    params.Parameter[0].NumberOfValues = 1;
    params.Parameter[0].Value = &quality;
    const bool is_jpeg = fmt == "jpg" || fmt == "jpeg";
    return bitmap.Save(wide_path.c_str(), &clsid, is_jpeg ? &params : nullptr) == Gdiplus::Ok ? 0 : -1;
}

#elif defined(ESHKOL_IMAGE_IO_LIBPNG)

/**
 * @brief libpng backend: decode a PNG file into a normalized double tensor.
 *
 * Normalizes source formats via libpng transforms (16-bit strip, palette
 * expansion, sub-8-bit gray expansion, tRNS-to-alpha, gray-to-RGB) so the
 * output is always RGB or RGBA. Uses libpng's setjmp-based error handling to
 * clean up and return nullptr on any decode error.
 *
 * @param[out] out_w Image width.
 * @param[out] out_h Image height.
 * @param[out] out_c Channel count (3 or 4).
 * @return Arena-allocated double array, or nullptr on any failure.
 */
double* read_png_native(const char* path, int* out_w, int* out_h, int* out_c) {
    FILE* fp = eshkol_fopen(path, "rb");
    if (!fp) return nullptr;
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(fp); return nullptr; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, nullptr, nullptr); fclose(fp); return nullptr; }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        return nullptr;
    }

    png_init_io(png, fp);
    png_read_info(png, info);
    const png_uint_32 width = png_get_image_width(png, info);
    const png_uint_32 height = png_get_image_height(png, info);
    const int color_type = png_get_color_type(png, info);
    const int bit_depth = png_get_bit_depth(png, info);
    const bool has_alpha = color_type == PNG_COLOR_TYPE_RGBA ||
                           color_type == PNG_COLOR_TYPE_GRAY_ALPHA ||
                           png_get_valid(png, info, PNG_INFO_tRNS);
    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    png_read_update_info(png, info);

    const int w = static_cast<int>(width);
    const int h = static_cast<int>(height);
    const int src_channels = static_cast<int>(png_get_channels(png, info));
    const int out_channels = has_alpha ? 4 : 3;
    if (!valid_dimensions(w, h, src_channels) || !valid_dimensions(w, h, out_channels)) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        return nullptr;
    }

    const size_t rowbytes = png_get_rowbytes(png, info);
    std::vector<uint8_t> raw(rowbytes * static_cast<size_t>(h));
    std::vector<png_bytep> rows(static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) {
        rows[static_cast<size_t>(y)] = raw.data() + static_cast<size_t>(y) * rowbytes;
    }
    png_read_image(png, rows.data());
    png_read_end(png, nullptr);

    std::vector<uint8_t> pixels(static_cast<size_t>(w) * static_cast<size_t>(h) *
                                static_cast<size_t>(out_channels));
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = raw.data() + static_cast<size_t>(y) * rowbytes;
        for (int x = 0; x < w; ++x) {
            const size_t src = static_cast<size_t>(x) * static_cast<size_t>(src_channels);
            const size_t dst = (static_cast<size_t>(y) * static_cast<size_t>(w) +
                                static_cast<size_t>(x)) * static_cast<size_t>(out_channels);
            pixels[dst + 0] = row[src + 0];
            pixels[dst + 1] = src_channels >= 2 ? row[src + 1] : row[src + 0];
            pixels[dst + 2] = src_channels >= 3 ? row[src + 2] : row[src + 0];
            if (out_channels == 4) pixels[dst + 3] = src_channels >= 4 ? row[src + 3] : 255;
        }
    }

    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);
    double* result = bytes_to_arena_doubles(pixels.data(), w, h, out_channels);
    if (!result) return nullptr;
    *out_w = w;
    *out_h = h;
    *out_c = out_channels;
    return result;
}

/**
 * @brief libpng backend: encode a byte pixel buffer to a PNG file.
 * @param channels Must be 1 (gray), 3 (RGB), or 4 (RGBA); other values fail.
 * @return true on success, false on invalid input or any libpng error.
 */
bool write_png_native(const char* path, const uint8_t* pixels, int w, int h, int channels) {
    if (!pixels || !valid_dimensions(w, h, channels) ||
        !(channels == 1 || channels == 3 || channels == 4)) {
        return false;
    }
    FILE* fp = eshkol_fopen(path, "wb");
    if (!fp) return false;
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(fp); return false; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, nullptr); fclose(fp); return false; }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return false;
    }
    int color_type = PNG_COLOR_TYPE_RGB;
    if (channels == 1) color_type = PNG_COLOR_TYPE_GRAY;
    if (channels == 4) color_type = PNG_COLOR_TYPE_RGBA;
    png_init_io(png, fp);
    png_set_IHDR(png, info, static_cast<png_uint_32>(w), static_cast<png_uint_32>(h),
                 8, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    std::vector<png_bytep> rows(static_cast<size_t>(h));
    const size_t stride = static_cast<size_t>(w) * static_cast<size_t>(channels);
    for (int y = 0; y < h; ++y) {
        rows[static_cast<size_t>(y)] = const_cast<png_bytep>(pixels + static_cast<size_t>(y) * stride);
    }
    png_write_image(png, rows.data());
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return true;
}

/** Read the entire contents of file @p path into @p out (used to load an
 *  encoded image blob for sniffing/decoding, e.g. WebP). Returns false on any I/O error. */
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

#if defined(ESHKOL_IMAGE_IO_LIBJPEG)

struct JpegErrorManager {
    jpeg_error_mgr pub;
    jmp_buf jump_buffer;
};

/** libjpeg error callback: longjmp back to the enclosing setjmp instead of
 *  letting libjpeg call exit() on a fatal error. */
void jpeg_error_exit(j_common_ptr cinfo) {
    auto* err = reinterpret_cast<JpegErrorManager*>(cinfo->err);
    longjmp(err->jump_buffer, 1);
}

/**
 * @brief libjpeg backend: decode a JPEG file into a normalized double tensor (always RGB).
 *
 * Decompresses scanline-by-scanline into a malloc'd byte buffer via
 * libjpeg's public API, forcing JCS_RGB output, then converts to an
 * arena-allocated normalized double array. Uses jpeg_error_exit()'s longjmp
 * to recover from libjpeg errors and return nullptr instead of crashing.
 *
 * @param[out] out_w Image width.
 * @param[out] out_h Image height.
 * @param[out] out_c Channel count (always 3).
 * @return Arena-allocated double array, or nullptr on any failure.
 */
double* read_jpeg_native(const char* path, int* out_w, int* out_h, int* out_c) {
    FILE* fp = eshkol_fopen(path, "rb");
    if (!fp) return nullptr;

    jpeg_decompress_struct cinfo;
    memset(&cinfo, 0, sizeof(cinfo));
    JpegErrorManager jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;
    bool cinfo_created = false;
    uint8_t* pixels = nullptr;

    if (setjmp(jerr.jump_buffer)) {
        if (pixels) free(pixels);
        if (cinfo_created) jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        return nullptr;
    }

    jpeg_create_decompress(&cinfo);
    cinfo_created = true;
    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    const int w = static_cast<int>(cinfo.output_width);
    const int h = static_cast<int>(cinfo.output_height);
    const int channels = 3;
    if (!valid_dimensions(w, h, channels) || cinfo.output_components != channels) {
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        return nullptr;
    }

    const size_t pixel_count = static_cast<size_t>(w) * static_cast<size_t>(h);
    pixels = static_cast<uint8_t*>(malloc(pixel_count * static_cast<size_t>(channels)));
    if (!pixels) {
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        return nullptr;
    }
    const size_t stride = static_cast<size_t>(w) * static_cast<size_t>(channels);
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row[1] = {
            pixels + static_cast<size_t>(cinfo.output_scanline) * stride
        };
        jpeg_read_scanlines(&cinfo, row, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    double* result = bytes_to_arena_doubles(pixels, w, h, channels);
    free(pixels);
    if (!result) return nullptr;
    *out_w = w;
    *out_h = h;
    *out_c = channels;
    return result;
}

/**
 * @brief libjpeg backend: encode a byte pixel buffer to a JPEG file at quality 90.
 *
 * Grayscale (1-channel) input is compressed as JCS_GRAYSCALE; 4-channel
 * (RGBA) input has its alpha dropped per-scanline into a scratch RGB row
 * before compression, since JPEG has no alpha channel.
 *
 * @param channels Must be 1, 3, or 4; other values fail.
 * @return true on success, false on invalid input or any libjpeg error.
 */
bool write_jpeg_native(const char* path, const uint8_t* pixels, int w, int h, int channels) {
    if (!pixels || !valid_dimensions(w, h, channels) ||
        !(channels == 1 || channels == 3 || channels == 4)) {
        return false;
    }

    FILE* fp = eshkol_fopen(path, "wb");
    if (!fp) return false;

    jpeg_compress_struct cinfo;
    memset(&cinfo, 0, sizeof(cinfo));
    JpegErrorManager jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;
    bool cinfo_created = false;
    uint8_t* rgb_row = nullptr;

    if (setjmp(jerr.jump_buffer)) {
        if (rgb_row) free(rgb_row);
        if (cinfo_created) jpeg_destroy_compress(&cinfo);
        fclose(fp);
        return false;
    }

    jpeg_create_compress(&cinfo);
    cinfo_created = true;
    jpeg_stdio_dest(&cinfo, fp);
    cinfo.image_width = static_cast<JDIMENSION>(w);
    cinfo.image_height = static_cast<JDIMENSION>(h);
    cinfo.input_components = channels == 1 ? 1 : 3;
    cinfo.in_color_space = channels == 1 ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    const size_t src_stride = static_cast<size_t>(w) * static_cast<size_t>(channels);
    const size_t rgb_stride = static_cast<size_t>(w) * 3u;
    if (channels == 4) {
        rgb_row = static_cast<uint8_t*>(malloc(rgb_stride));
        if (!rgb_row) {
            jpeg_destroy_compress(&cinfo);
            fclose(fp);
            return false;
        }
    }
    while (cinfo.next_scanline < cinfo.image_height) {
        const uint8_t* row_data = pixels + static_cast<size_t>(cinfo.next_scanline) * src_stride;
        if (channels == 4) {
            for (int x = 0; x < w; ++x) {
                const size_t src = static_cast<size_t>(x) * 4u;
                const size_t dst = static_cast<size_t>(x) * 3u;
                rgb_row[dst + 0] = row_data[src + 0];
                rgb_row[dst + 1] = row_data[src + 1];
                rgb_row[dst + 2] = row_data[src + 2];
            }
            row_data = rgb_row;
        }
        JSAMPROW row[1] = {const_cast<JSAMPROW>(row_data)};
        jpeg_write_scanlines(&cinfo, row, 1);
    }

    jpeg_finish_compress(&cinfo);
    if (rgb_row) free(rgb_row);
    jpeg_destroy_compress(&cinfo);
    fclose(fp);
    return true;
}

#endif

#if defined(ESHKOL_IMAGE_IO_LIBWEBP)

/**
 * @brief libwebp backend: decode a WebP file into a normalized double tensor.
 *
 * Reads the whole file, inspects its bitstream features to check for alpha,
 * decodes via WebPDecodeRGBA()/WebPDecodeRGB() accordingly, then converts to
 * an arena-allocated normalized double array.
 *
 * @param[out] out_w Image width.
 * @param[out] out_h Image height.
 * @param[out] out_c Channel count (3 or 4).
 * @return Arena-allocated double array, or nullptr on any failure.
 */
double* read_webp_native(const char* path, int* out_w, int* out_h, int* out_c) {
    std::vector<uint8_t> encoded;
    if (!read_file_bytes(path, &encoded) || encoded.empty()) return nullptr;

    WebPBitstreamFeatures features;
    if (WebPGetFeatures(encoded.data(), encoded.size(), &features) != VP8_STATUS_OK) {
        return nullptr;
    }
    const bool has_alpha = features.has_alpha != 0;
    const int channels = has_alpha ? 4 : 3;
    if (!valid_dimensions(features.width, features.height, channels)) return nullptr;

    int w = 0;
    int h = 0;
    uint8_t* decoded = has_alpha
        ? WebPDecodeRGBA(encoded.data(), encoded.size(), &w, &h)
        : WebPDecodeRGB(encoded.data(), encoded.size(), &w, &h);
    if (!decoded || w != features.width || h != features.height) {
        if (decoded) WebPFree(decoded);
        return nullptr;
    }

    double* result = bytes_to_arena_doubles(decoded, w, h, channels);
    WebPFree(decoded);
    if (!result) return nullptr;
    *out_w = w;
    *out_h = h;
    *out_c = channels;
    return result;
}

/**
 * @brief libwebp backend: encode a byte pixel buffer to a WebP file at quality 90.
 *
 * Grayscale (1-channel) input is expanded to an RGB scratch buffer first
 * (WebP's encoder has no grayscale entry point), then encoded via
 * WebPEncodeRGBA()/WebPEncodeRGB() depending on channel count.
 *
 * @param channels Must be 1, 3, or 4; other values fail.
 * @return true on success, false on invalid input or encode/write failure.
 */
bool write_webp_native(const char* path, const uint8_t* pixels, int w, int h, int channels) {
    if (!pixels || !valid_dimensions(w, h, channels) ||
        !(channels == 1 || channels == 3 || channels == 4)) {
        return false;
    }

    const uint8_t* source = pixels;
    int webp_channels = channels;
    std::vector<uint8_t> rgb;
    if (channels == 1) {
        rgb.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3u);
        const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h);
        for (size_t i = 0; i < count; ++i) {
            rgb[i * 3u + 0u] = pixels[i];
            rgb[i * 3u + 1u] = pixels[i];
            rgb[i * 3u + 2u] = pixels[i];
        }
        source = rgb.data();
        webp_channels = 3;
    }

    uint8_t* encoded = nullptr;
    size_t encoded_size = 0;
    if (webp_channels == 4) {
        encoded_size = WebPEncodeRGBA(source, w, h, w * 4, 90.0f, &encoded);
    } else {
        encoded_size = WebPEncodeRGB(source, w, h, w * 3, 90.0f, &encoded);
    }
    if (!encoded || encoded_size == 0) {
        if (encoded) WebPFree(encoded);
        return false;
    }

    FILE* fp = eshkol_fopen(path, "wb");
    if (!fp) {
        WebPFree(encoded);
        return false;
    }
    const size_t written = fwrite(encoded, 1, encoded_size, fp);
    fclose(fp);
    WebPFree(encoded);
    return written == encoded_size;
}

#endif

/**
 * @brief Unix/libpng backend dispatcher: sniff @p path's file signature and decode with the matching backend.
 *
 * Reads the first 12 bytes and checks for the PNG magic number, then (if
 * compiled in) the JPEG SOI marker, then the RIFF/WEBP container tag,
 * dispatching to read_png_native()/read_jpeg_native()/read_webp_native()
 * accordingly. Returns nullptr if the signature doesn't match any compiled-in format.
 */
double* read_image_native(const char* path, int* out_w, int* out_h, int* out_c) {
    uint8_t sig[12] = {0};
    FILE* fp = eshkol_fopen(path, "rb");
    if (!fp) return nullptr;
    const size_t n = fread(sig, 1, sizeof(sig), fp);
    fclose(fp);
    static const uint8_t png_sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (n >= 8 && memcmp(sig, png_sig, 8) == 0) {
        return read_png_native(path, out_w, out_h, out_c);
    }
#if defined(ESHKOL_IMAGE_IO_LIBJPEG)
    if (n >= 3 && sig[0] == 0xff && sig[1] == 0xd8 && sig[2] == 0xff) {
        return read_jpeg_native(path, out_w, out_h, out_c);
    }
#endif
#if defined(ESHKOL_IMAGE_IO_LIBWEBP)
    if (n >= 12 && memcmp(sig, "RIFF", 4) == 0 && memcmp(sig + 8, "WEBP", 4) == 0) {
        return read_webp_native(path, out_w, out_h, out_c);
    }
#endif
    return nullptr;
}

/**
 * @brief Unix/libpng backend dispatcher: encode @p data to @p path using the backend matching @p format.
 *
 * Converts to byte pixels once, then dispatches by lowercased @p format
 * (default "png") to write_png_native(), and (if compiled in)
 * write_jpeg_native() or write_webp_native().
 *
 * @return 0 on success, -1 on invalid input or an unsupported/uncompiled format.
 */
int write_image_native(const char* path, const double* data,
                       int w, int h, int channels, const char* format) {
    const std::string fmt = lower_ascii(format ? format : "png");
    std::vector<uint8_t> pixels;
    if (!tensor_to_bytes(data, w, h, channels, &pixels)) return -1;
    if (fmt.empty() || fmt == "png") {
        return write_png_native(path, pixels.data(), w, h, channels) ? 0 : -1;
    }
#if defined(ESHKOL_IMAGE_IO_LIBJPEG)
    if (fmt == "jpg" || fmt == "jpeg") {
        return write_jpeg_native(path, pixels.data(), w, h, channels) ? 0 : -1;
    }
#endif
#if defined(ESHKOL_IMAGE_IO_LIBWEBP)
    if (fmt == "webp") {
        return write_webp_native(path, pixels.data(), w, h, channels) ? 0 : -1;
    }
#endif
    return -1;
}

#else

/** Fallback backend (no platform image API compiled in): always fails. */
double* read_image_native(const char*, int*, int*, int*) {
    return nullptr;
}

/** Fallback backend (no platform image API compiled in): always fails. */
int write_image_native(const char*, const double*, int, int, int, const char*) {
    return -1;
}

#endif

} // namespace

/**
 * @brief Public entry point: read an image file into a normalized [0,1] double tensor (row-major HWC).
 *
 * Validates arguments, checks the "file-read" capability (denying and
 * returning nullptr if not permitted), then dispatches to the
 * platform-specific read_image_native() (Apple/GDI+/libpng+friends/stub).
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
 * (denying and returning -1 if not permitted), then dispatches to the
 * platform-specific write_image_native().
 *
 * @param format "png", "jpg"/"jpeg", "webp", "bmp" depending on backend support; NULL defaults to "png".
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
