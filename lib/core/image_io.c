/**
 * @file image_io.c
 * @brief Native image I/O for Eshkol tensors.
 *
 * This file keeps the historical .c path for build-system compatibility, but
 * CMake compiles it as C++ so the runtime can call platform/system image APIs.
 * It uses ImageIO/CoreGraphics on macOS, GDI+ on Windows, and libpng on
 * Linux/Unix. No vendored image decoder is used.
 */

#include "../../inc/eshkol/core/image_io.h"
#include "../../inc/eshkol/runtime_exports.h"
#include "arena_memory.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

namespace {

constexpr int kMaxImageDimension = 65535;
constexpr int kMaxChannels = 16;

bool valid_dimensions(int w, int h, int channels) {
    return w > 0 && h > 0 && channels > 0 && channels <= kMaxChannels &&
           w <= kMaxImageDimension && h <= kMaxImageDimension;
}

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

double* arena_doubles(size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(double)) {
        return nullptr;
    }
    return static_cast<double*>(arena_allocate(get_global_arena(), count * sizeof(double)));
}

uint8_t double_to_byte(double value) {
    double scaled = value * 255.0;
    if (!(scaled >= 0.0)) scaled = 0.0;
    if (scaled > 255.0) scaled = 255.0;
    return static_cast<uint8_t>(scaled + 0.5);
}

std::string lower_ascii(const char* s) {
    std::string out = s ? s : "";
    for (char& ch : out) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return out;
}

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

CFURLRef apple_file_url(const char* path) {
    CFStringRef cf_path = CFStringCreateWithCString(kCFAllocatorDefault, path,
                                                     kCFStringEncodingUTF8);
    if (!cf_path) return nullptr;
    CFURLRef url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, cf_path,
                                                  kCFURLPOSIXPathStyle, false);
    CFRelease(cf_path);
    return url;
}

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

CFStringRef apple_type_for_format(const std::string& format) {
    if (format.empty() || format == "png") return CFSTR("public.png");
    if (format == "jpg" || format == "jpeg") return CFSTR("public.jpeg");
    if (format == "bmp") return CFSTR("com.microsoft.bmp");
    if (format == "tif" || format == "tiff") return CFSTR("public.tiff");
    if (format == "webp") return CFSTR("org.webmproject.webp");
    return nullptr;
}

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

void init_gdiplus_once() {
    Gdiplus::GdiplusStartupInput input;
    Gdiplus::GdiplusStartup(&gdiplus_token, &input, nullptr);
}

bool ensure_gdiplus() {
    std::call_once(gdiplus_once, init_gdiplus_once);
    return gdiplus_token != 0;
}

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

const WCHAR* mime_for_format(const std::string& format) {
    if (format.empty() || format == "png") return L"image/png";
    if (format == "jpg" || format == "jpeg") return L"image/jpeg";
    if (format == "bmp") return L"image/bmp";
    if (format == "gif") return L"image/gif";
    return nullptr;
}

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

double* read_image_native(const char* path, int* out_w, int* out_h, int* out_c) {
    uint8_t sig[8] = {0};
    FILE* fp = eshkol_fopen(path, "rb");
    if (!fp) return nullptr;
    const size_t n = fread(sig, 1, sizeof(sig), fp);
    fclose(fp);
    static const uint8_t png_sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (n >= 8 && memcmp(sig, png_sig, 8) == 0) {
        return read_png_native(path, out_w, out_h, out_c);
    }
    return nullptr;
}

int write_image_native(const char* path, const double* data,
                       int w, int h, int channels, const char* format) {
    const std::string fmt = lower_ascii(format ? format : "png");
    if (!(fmt.empty() || fmt == "png")) return -1;
    std::vector<uint8_t> pixels;
    if (!tensor_to_bytes(data, w, h, channels, &pixels)) return -1;
    return write_png_native(path, pixels.data(), w, h, channels) ? 0 : -1;
}

#else

double* read_image_native(const char*, int*, int*, int*) {
    return nullptr;
}

int write_image_native(const char*, const double*, int, int, int, const char*) {
    return -1;
}

#endif

} // namespace

double* eshkol_image_read(const char* path, int* out_w, int* out_h, int* out_c) {
    if (!path || !out_w || !out_h || !out_c ||
        eshkol_capability_runtime_allows("file-read") == 0) {
        return nullptr;
    }
    return read_image_native(path, out_w, out_h, out_c);
}

int eshkol_image_write(const char* path, const double* data,
                       int w, int h, int channels, const char* format) {
    if (!path || !data || !valid_dimensions(w, h, channels) ||
        eshkol_capability_runtime_allows("file-write") == 0) {
        return -1;
    }
    return write_image_native(path, data, w, h, channels, format);
}

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
