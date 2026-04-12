/**
 * @file image_io.c
 * @brief Image I/O for Eshkol — read/write PNG, JPEG, BMP as tensors.
 *
 * Uses stb_image (read), stb_image_write (write), stb_image_resize2 (resize).
 * All single-header, MIT/public domain.
 *
 * Images are represented as tensors:
 *   - Color:     shape (H, W, 3) — RGB channels, values 0.0-1.0
 *   - Grayscale: shape (H, W)    — single channel, values 0.0-1.0
 *   - RGBA:      shape (H, W, 4) — with alpha
 *
 * API:
 *   eshkol_image_read(path)                → tensor (H,W,C) or NULL
 *   eshkol_image_write(path, tensor, fmt)  → 0 on success
 *   eshkol_image_to_grayscale(tensor)      → tensor (H,W) or NULL
 *   eshkol_image_resize(tensor, new_h, new_w) → tensor or NULL
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "../../deps/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../deps/stb/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../deps/stb/stb_image_resize2.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Read image file → flat double array (normalized 0-1) ── */

double* eshkol_image_read(const char* path, int* out_w, int* out_h, int* out_c) {
    if (!path || !out_w || !out_h || !out_c) return NULL;

    int w, h, channels;
    unsigned char* data = stbi_load(path, &w, &h, &channels, 0);
    if (!data) return NULL;

    *out_w = w;
    *out_h = h;
    *out_c = channels;

    /* Convert to normalized doubles */
    size_t total = (size_t)w * (size_t)h * (size_t)channels;
    double* result = (double*)malloc(total * sizeof(double));
    if (!result) { stbi_image_free(data); return NULL; }

    for (size_t i = 0; i < total; i++) {
        result[i] = (double)data[i] / 255.0;
    }

    stbi_image_free(data);
    return result;
}

/* ── Write tensor data to image file ── */

int eshkol_image_write(const char* path, const double* data,
                       int w, int h, int channels, const char* format) {
    if (!path || !data || w <= 0 || h <= 0 || channels <= 0) return -1;

    /* Convert doubles back to uint8 */
    size_t total = (size_t)w * (size_t)h * (size_t)channels;
    unsigned char* pixels = (unsigned char*)malloc(total);
    if (!pixels) return -1;

    for (size_t i = 0; i < total; i++) {
        double v = data[i] * 255.0;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        pixels[i] = (unsigned char)(v + 0.5);
    }

    int result = -1;
    if (!format || strcmp(format, "png") == 0) {
        result = stbi_write_png(path, w, h, channels, pixels, w * channels) ? 0 : -1;
    } else if (strcmp(format, "bmp") == 0) {
        result = stbi_write_bmp(path, w, h, channels, pixels) ? 0 : -1;
    } else if (strcmp(format, "jpg") == 0 || strcmp(format, "jpeg") == 0) {
        result = stbi_write_jpg(path, w, h, channels, pixels, 90) ? 0 : -1;
    } else if (strcmp(format, "tga") == 0) {
        result = stbi_write_tga(path, w, h, channels, pixels) ? 0 : -1;
    }

    free(pixels);
    return result;
}

/* ── Convert color image to grayscale ── */

double* eshkol_image_to_grayscale(const double* data, int w, int h, int channels) {
    if (!data || w <= 0 || h <= 0) return NULL;
    if (channels == 1) {
        /* Already grayscale — copy */
        size_t total = (size_t)w * (size_t)h;
        double* result = (double*)malloc(total * sizeof(double));
        if (!result) return NULL;
        memcpy(result, data, total * sizeof(double));
        return result;
    }

    size_t pixels = (size_t)w * (size_t)h;
    double* result = (double*)malloc(pixels * sizeof(double));
    if (!result) return NULL;

    /* Luminance: Y = 0.2126*R + 0.7152*G + 0.0722*B (ITU-R BT.709) */
    for (size_t i = 0; i < pixels; i++) {
        size_t base = i * (size_t)channels;
        double r = data[base];
        double g = (channels >= 2) ? data[base + 1] : r;
        double b = (channels >= 3) ? data[base + 2] : r;
        result[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    return result;
}

/* ── Resize image ── */

double* eshkol_image_resize(const double* data, int w, int h, int channels,
                            int new_w, int new_h) {
    if (!data || w <= 0 || h <= 0 || new_w <= 0 || new_h <= 0 || channels <= 0)
        return NULL;

    /* Convert to uint8 for stbir */
    size_t src_total = (size_t)w * (size_t)h * (size_t)channels;
    size_t dst_total = (size_t)new_w * (size_t)new_h * (size_t)channels;
    unsigned char* src_pixels = (unsigned char*)malloc(src_total);
    unsigned char* dst_pixels = (unsigned char*)malloc(dst_total);
    if (!src_pixels || !dst_pixels) {
        free(src_pixels); free(dst_pixels);
        return NULL;
    }

    for (size_t i = 0; i < src_total; i++) {
        double v = data[i] * 255.0;
        if (v < 0) v = 0; if (v > 255) v = 255;
        src_pixels[i] = (unsigned char)(v + 0.5);
    }

    /* Use stb_image_resize2 */
    stbir_pixel_layout layout;
    switch (channels) {
        case 1: layout = STBIR_1CHANNEL; break;
        case 2: layout = STBIR_2CHANNEL; break;
        case 3: layout = STBIR_RGB; break;
        case 4: layout = STBIR_RGBA; break;
        default: layout = STBIR_RGB; break;
    }

    stbir_resize_uint8_linear(src_pixels, w, h, w * channels,
                               dst_pixels, new_w, new_h, new_w * channels,
                               layout);

    /* Convert back to normalized doubles */
    double* result = (double*)malloc(dst_total * sizeof(double));
    if (!result) { free(src_pixels); free(dst_pixels); return NULL; }

    for (size_t i = 0; i < dst_total; i++) {
        result[i] = (double)dst_pixels[i] / 255.0;
    }

    free(src_pixels);
    free(dst_pixels);
    return result;
}
