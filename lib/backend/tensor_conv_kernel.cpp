/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * tensor_conv_kernel.cpp — runtime entry point for the canonical conv2d
 * forward, called directly from LLVM-generated code (-r / JIT and AOT).
 *
 * The numeric work is performed by eshkol_conv2d_kernel in
 * tensor_conv_kernel.h, the SAME inline kernel the embedded VM uses, so the
 * codegen and VM conv2d can never diverge (ESH-0068). This file only adapts
 * the eshkol heap-tensor representation (eshkol_tensor_t, doubles stored as
 * int64 bit patterns) to the flat-double kernel and computes the output shape
 * per the canonical contract documented in tensor_conv_kernel.h.
 */

#include "tensor_conv_kernel.h"
#include "../../lib/core/arena_memory.h"

extern "C" {

/*
 * eshkol_rt_conv2d — allocate and compute conv2d(input, kernel) into a fresh
 * arena tensor. Returns NULL on shape errors (the generated code treats a
 * NULL result as a runtime fault path; callers pass validated tensors).
 *
 *   input  : eshkol_tensor_t*  (rank >= 2; last two dims are H, W)
 *   kernel : eshkol_tensor_t*  (rank 2 -> single-plane; rank >= 4 -> NCHW)
 *   stride : spatial stride (>= 1; values <= 0 are clamped to 1)
 *   pad    : symmetric zero-padding (>= 0; negatives clamped to 0)
 */
eshkol_tensor_t* eshkol_rt_conv2d(void* arena_void,
                                  const eshkol_tensor_t* input,
                                  const eshkol_tensor_t* kernel,
                                  int64_t stride, int64_t pad) {
    if (!input || !kernel || !arena_void) return nullptr;
    arena_t* arena = static_cast<arena_t*>(arena_void);
    if (stride <= 0) stride = 1;
    if (pad < 0) pad = 0;

    int64_t in_nd = static_cast<int64_t>(input->num_dimensions);
    int64_t k_nd  = static_cast<int64_t>(kernel->num_dimensions);
    if (in_nd < 2 || k_nd < 2) return nullptr;

    const uint64_t* ind = input->dimensions;
    const uint64_t* kd  = kernel->dimensions;
    if (!ind || !kd) return nullptr;

    int64_t H  = static_cast<int64_t>(ind[in_nd - 2]);
    int64_t W  = static_cast<int64_t>(ind[in_nd - 1]);
    int64_t kH = static_cast<int64_t>(kd[k_nd - 2]);
    int64_t kW = static_cast<int64_t>(kd[k_nd - 1]);

    int64_t oH = eshkol_conv_out_dim(H, kH, stride, pad);
    int64_t oW = eshkol_conv_out_dim(W, kW, stride, pad);
    if (oH <= 0 || oW <= 0) return nullptr;

    int64_t batch, in_c, out_c, out_nd;

    if (k_nd >= 4) {
        /* Full NCHW: kernel [out_c, in_c, kH, kW], input [..., in_c, H, W]. */
        out_c = static_cast<int64_t>(kd[0]);
        in_c  = static_cast<int64_t>(kd[1]);
        int64_t in_channels = (in_nd >= 3) ? static_cast<int64_t>(ind[in_nd - 3]) : 1;
        if (in_channels != in_c) return nullptr;  /* channel mismatch */
        batch = 1;
        for (int64_t i = 0; i < in_nd - 3; i++) batch *= static_cast<int64_t>(ind[i]);
        out_nd = (in_nd - 3) + 3;  /* leading dims + out_c + oH + oW */
    } else {
        /* Single-plane / depth-wise: rank-2 kernel, in_c = out_c = 1.
         * Every leading input dim is an independent plane. */
        out_c = 1;
        in_c  = 1;
        batch = 1;
        for (int64_t i = 0; i < in_nd - 2; i++) batch *= static_cast<int64_t>(ind[i]);
        out_nd = in_nd;  /* preserve leading dims + oH + oW */
    }

    int64_t out_total = batch * out_c * oH * oW;
    if (out_total <= 0) return nullptr;

    eshkol_tensor_t* out =
        arena_allocate_tensor_full(arena, static_cast<uint64_t>(out_nd),
                                   static_cast<uint64_t>(out_total));
    if (!out) return nullptr;

    /* Fill output dimension array per the contract. */
    if (k_nd >= 4) {
        int64_t lead = in_nd - 3;
        for (int64_t i = 0; i < lead; i++) out->dimensions[i] = ind[i];
        out->dimensions[lead]     = static_cast<uint64_t>(out_c);
        out->dimensions[lead + 1] = static_cast<uint64_t>(oH);
        out->dimensions[lead + 2] = static_cast<uint64_t>(oW);
    } else {
        for (int64_t i = 0; i < in_nd - 2; i++) out->dimensions[i] = ind[i];
        out->dimensions[in_nd - 2] = static_cast<uint64_t>(oH);
        out->dimensions[in_nd - 1] = static_cast<uint64_t>(oW);
    }

    /* Tensor elements are doubles stored as int64 bit patterns; a double* view
     * is bit-identical, so the shared kernel operates on them directly. */
    eshkol_conv2d_kernel(
        reinterpret_cast<const double*>(input->elements),  batch, in_c, H, W,
        reinterpret_cast<const double*>(kernel->elements), out_c, kH, kW,
        stride, pad,
        reinterpret_cast<double*>(out->elements));

    return out;
}

}  /* extern "C" */
