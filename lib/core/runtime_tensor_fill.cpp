/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Freestanding-safe tensor fill helpers.
 *
 * These helpers operate only on tagged tensor layout and raw memory. They do
 * not allocate, raise, touch files, read environment state, or depend on host
 * threads/signals/processes, so they belong to the runtime-core source set.
 */

#include <eshkol/eshkol.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" {

/**
 * @brief Unwrap a tagged tensor value into its raw element buffer and HWC extents.
 *
 * Validates that `t_tv` is a HEAP_PTR (tolerating a legacy low-nibble-only
 * type tag) referencing a heap object whose header subtype is
 * HEAP_SUBTYPE_TENSOR, then reads the tensor's dimensions according to a
 * local `fill_tensor_layout` view matching eshkol_tensor_t's field order.
 * Only rank-2 (height, width; channels forced to 1) and rank-3 (height,
 * width, channels) tensors are supported — any other rank fails. Does not
 * allocate; this is a read-only view into the existing arena-owned tensor's
 * element storage.
 *
 * @param t_tv    Tagged value expected to reference a tensor.
 * @param out_h   Set to the tensor's height (dimension 0), or 0 on failure.
 * @param out_w   Set to the tensor's width (dimension 1), or 0 on failure.
 * @param out_c   Set to the channel count (1 for rank-2, dimension 2 for rank-3), or 0 on failure.
 * @return        Pointer to the tensor's raw element storage (int64 bit patterns
 *                of doubles), or nullptr if t_tv is not a rank-2/3 tensor.
 */
static inline void* eshkol_tensor_payload(
    const eshkol_tagged_value_t* t_tv,
    int64_t* out_h, int64_t* out_w, int64_t* out_c) {
    *out_h = 0;
    *out_w = 0;
    *out_c = 0;
    if (!t_tv) return nullptr;

    eshkol_tagged_value_t value = *t_tv;
    uint8_t base = value.type;
    if (base < 8) base &= 0x0F;
    if (base != ESHKOL_VALUE_HEAP_PTR || value.data.ptr_val == 0) return nullptr;

    const eshkol_object_header_t* header = ESHKOL_GET_HEADER((void*)value.data.ptr_val);
    if (header->subtype != HEAP_SUBTYPE_TENSOR) return nullptr;

    struct fill_tensor_layout {
        uint64_t* dimensions;
        uint64_t num_dimensions;
        int64_t* elements;
        uint64_t total_elements;
    };

    const fill_tensor_layout* tensor =
        (const fill_tensor_layout*)(uintptr_t)value.data.ptr_val;
    if (tensor->num_dimensions == 2) {
        *out_h = (int64_t)tensor->dimensions[0];
        *out_w = (int64_t)tensor->dimensions[1];
        *out_c = 1;
    } else if (tensor->num_dimensions == 3) {
        *out_h = (int64_t)tensor->dimensions[0];
        *out_w = (int64_t)tensor->dimensions[1];
        *out_c = (int64_t)tensor->dimensions[2];
    } else {
        return nullptr;
    }

    return tensor->elements;
}

/**
 * @brief Write one pixel's channel values into a tensor's raw element buffer.
 *
 * Computes the base index `(y * width + x) * channels` into `elements` and
 * writes up to `channels` values there, each stored as the raw int64 bit
 * pattern of the corresponding double (matching eshkol_tensor_t's
 * doubles-as-int64-bits element encoding). If exactly one fill value is
 * given and the tensor has more than one channel, that single value is
 * broadcast to every channel (e.g. filling an RGB tensor with one
 * grayscale-style value); otherwise, values are copied channel-for-channel
 * up to `min(fill_value_count, channels)`, leaving any remaining channels
 * untouched.
 *
 * @param elements          Tensor's raw element buffer (caller-validated bounds).
 * @param y                 Row coordinate of the pixel.
 * @param x                 Column coordinate of the pixel.
 * @param width             Tensor width, used to compute the row stride.
 * @param channels          Number of channels per pixel in the tensor.
 * @param fill_values       Array of double values to write.
 * @param fill_value_count  Number of entries in fill_values.
 */
static inline void eshkol_tensor_write_pixel(
    int64_t* elements, int64_t y, int64_t x,
    int64_t width, int64_t channels,
    const double* fill_values, int64_t fill_value_count) {
    const int64_t base = (y * width + x) * channels;
    if (fill_value_count == 1 && channels > 1) {
        int64_t bits;
        std::memcpy(&bits, &fill_values[0], sizeof(double));
        for (int64_t channel = 0; channel < channels; channel++) {
            elements[base + channel] = bits;
        }
    } else {
        const int64_t writes =
            fill_value_count < channels ? fill_value_count : channels;
        for (int64_t channel = 0; channel < writes; channel++) {
            int64_t bits;
            std::memcpy(&bits, &fill_values[channel], sizeof(double));
            elements[base + channel] = bits;
        }
    }
}

/**
 * @brief Fill an axis-aligned rectangular region of a 2D/3D tensor with a color.
 *
 * Resolves `t_tv` to its element buffer and extents via
 * eshkol_tensor_payload (no-op if it isn't a rank-2/3 tensor), normalizes
 * the rectangle so row0<=row1 and col0<=col1 (swapping if needed), clamps
 * it to the tensor's bounds ([0, height) x [0, width)), and then writes
 * `channels`/`num_channels` into every pixel in the clamped rectangle via
 * eshkol_tensor_write_pixel.
 *
 * @param t_tv          Tagged value expected to reference a rank-2/3 tensor.
 * @param row0          One row endpoint of the rectangle (any order relative to row1).
 * @param col0          One column endpoint of the rectangle (any order relative to col1).
 * @param row1          Other row endpoint of the rectangle.
 * @param col1          Other column endpoint of the rectangle.
 * @param channels      Fill color/value(s) to write to each pixel.
 * @param num_channels  Number of entries in channels.
 */
void eshkol_tensor_rect_fill(
    const eshkol_tagged_value_t* t_tv,
    int64_t row0, int64_t col0,
    int64_t row1, int64_t col1,
    const double* channels, int64_t num_channels) {
    int64_t height;
    int64_t width;
    int64_t channel_count;
    int64_t* elements =
        (int64_t*)eshkol_tensor_payload(t_tv, &height, &width, &channel_count);
    if (!elements) return;

    if (row0 > row1) {
        const int64_t tmp = row0;
        row0 = row1;
        row1 = tmp;
    }
    if (col0 > col1) {
        const int64_t tmp = col0;
        col0 = col1;
        col1 = tmp;
    }
    if (row0 < 0) row0 = 0;
    if (col0 < 0) col0 = 0;
    if (row1 > height) row1 = height;
    if (col1 > width) col1 = width;

    for (int64_t y = row0; y < row1; y++) {
        for (int64_t x = col0; x < col1; x++) {
            eshkol_tensor_write_pixel(
                elements, y, x, width, channel_count, channels, num_channels);
        }
    }
}

/**
 * @brief Fill a filled disk (circle) region of a 2D/3D tensor with a color.
 *
 * Resolves `t_tv` to its element buffer and extents via
 * eshkol_tensor_payload (no-op if it isn't a rank-2/3 tensor) and returns
 * immediately if `radius < 0`. Computes the disk's bounding box
 * `[cy-radius, cy+radius] x [cx-radius, cx+radius]`, clamps it to the
 * tensor's bounds, then for every pixel in that box writes
 * `channels`/`num_channels` (via eshkol_tensor_write_pixel) only if the
 * pixel's squared distance from `(cy, cx)` is within `radius^2` (a standard
 * midpoint-free circle membership test, not anti-aliased).
 *
 * @param t_tv          Tagged value expected to reference a rank-2/3 tensor.
 * @param cy            Row coordinate of the disk's center.
 * @param cx            Column coordinate of the disk's center.
 * @param radius        Disk radius in pixels; negative values are a no-op.
 * @param channels      Fill color/value(s) to write to each pixel inside the disk.
 * @param num_channels  Number of entries in channels.
 */
void eshkol_tensor_disk_fill(
    const eshkol_tagged_value_t* t_tv,
    int64_t cy, int64_t cx, int64_t radius,
    const double* channels, int64_t num_channels) {
    int64_t height;
    int64_t width;
    int64_t channel_count;
    int64_t* elements =
        (int64_t*)eshkol_tensor_payload(t_tv, &height, &width, &channel_count);
    if (!elements) return;
    if (radius < 0) return;

    int64_t y0 = cy - radius;
    int64_t y1 = cy + radius + 1;
    int64_t x0 = cx - radius;
    int64_t x1 = cx + radius + 1;
    if (y0 < 0) y0 = 0;
    if (x0 < 0) x0 = 0;
    if (y1 > height) y1 = height;
    if (x1 > width) x1 = width;
    const int64_t radius_squared = radius * radius;

    for (int64_t y = y0; y < y1; y++) {
        const int64_t dy = y - cy;
        const int64_t dy_squared = dy * dy;
        for (int64_t x = x0; x < x1; x++) {
            const int64_t dx = x - cx;
            if (dx * dx + dy_squared > radius_squared) continue;
            eshkol_tensor_write_pixel(
                elements, y, x, width, channel_count, channels, num_channels);
        }
    }
}

}  // extern "C"
