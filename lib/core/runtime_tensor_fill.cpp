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
