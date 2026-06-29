/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * tensor_conv_kernel.h — Canonical convolution / pooling numeric kernels.
 *
 * SINGLE SOURCE OF TRUTH for the forward numerics of conv2d (and the 2-D
 * pooling ops) shared by BOTH execution paths:
 *
 *   1. the LLVM codegen runtime (-r / JIT and AOT), via the extern "C"
 *      wrappers in tensor_conv_kernel.cpp that the generated IR calls; and
 *   2. the embedded bytecode VM (lib/backend/vm_tensor_ops.c), which calls
 *      these inline kernels directly.
 *
 * Because both paths compile the exact same source, conv2d can never again
 * diverge in shape or value between -r, AOT and the VM (ESH-0068).
 *
 * ── Canonical conv2d contract ───────────────────────────────────────────────
 *   Cross-correlation (NOT a flipped convolution), row-major, IEEE doubles.
 *   NCHW layout:
 *       input  : [batch, in_channels,  H,  W]
 *       kernel : [out_channels, in_channels, kH, kW]
 *       output : [batch, out_channels, oH, oW]
 *   with
 *       oH = (H + 2*pad - kH) / stride + 1
 *       oW = (W + 2*pad - kW) / stride + 1
 *   `pad` is symmetric zero-padding; `stride` applies to both spatial axes.
 *   The accumulation runs over (in_channels, kH, kW); leading batch / channel
 *   dimensions are preserved.
 *
 *   A rank-2 kernel [kH, kW] is the single-plane / depth-wise special case:
 *   in_channels = out_channels = 1 and every leading dimension of the input is
 *   treated as an independent plane convolved by the same kernel. This keeps
 *   the historical 2-D `(conv2d <HxW> <kHxkW> stride)` surface working while
 *   the rank-4 form implements full NCHW convolution. Both reduce to the one
 *   kernel below (the 2-D case is just batch = prod(leading dims), C = 1).
 */
#ifndef ESHKOL_TENSOR_CONV_KERNEL_H
#define ESHKOL_TENSOR_CONV_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Output extent of one spatial axis for cross-correlation with zero padding. */
static inline int64_t eshkol_conv_out_dim(int64_t in_dim, int64_t k_dim,
                                          int64_t stride, int64_t pad) {
    if (stride <= 0) stride = 1;
    if (pad < 0) pad = 0;
    int64_t span = in_dim + 2 * pad - k_dim;
    if (span < 0) return 0;
    return span / stride + 1;
}

/*
 * Canonical conv2d cross-correlation kernel.
 *   in  : [batch, in_c,  H,  W]   (row-major doubles)
 *   ker : [out_c, in_c, kH, kW]   (row-major doubles)
 *   out : [batch, out_c, oH, oW]  (pre-allocated by the caller; oH/oW must come
 *                                  from eshkol_conv_out_dim with the same args)
 * Out-of-range taps created by padding contribute zero (skipped).
 */
static inline void eshkol_conv2d_kernel(
        const double* in, int64_t batch, int64_t in_c, int64_t H, int64_t W,
        const double* ker, int64_t out_c, int64_t kH, int64_t kW,
        int64_t stride, int64_t pad, double* out) {
    if (!in || !ker || !out) return;
    if (stride <= 0) stride = 1;
    if (pad < 0) pad = 0;
    int64_t oH = eshkol_conv_out_dim(H, kH, stride, pad);
    int64_t oW = eshkol_conv_out_dim(W, kW, stride, pad);
    if (oH <= 0 || oW <= 0) return;

    const int64_t in_s0 = in_c * H * W, in_s1 = H * W, in_s2 = W;
    const int64_t k_s0  = in_c * kH * kW, k_s1 = kH * kW, k_s2 = kW;
    const int64_t o_s0  = out_c * oH * oW, o_s1 = oH * oW, o_s2 = oW;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t oc = 0; oc < out_c; oc++) {
            for (int64_t oh = 0; oh < oH; oh++) {
                for (int64_t ow = 0; ow < oW; ow++) {
                    double sum = 0.0;
                    for (int64_t ic = 0; ic < in_c; ic++) {
                        const double* in_plane  = in  + b * in_s0 + ic * in_s1;
                        const double* ker_plane = ker + oc * k_s0 + ic * k_s1;
                        for (int64_t kh = 0; kh < kH; kh++) {
                            int64_t ih = oh * stride + kh - pad;
                            if (ih < 0 || ih >= H) continue;
                            for (int64_t kw = 0; kw < kW; kw++) {
                                int64_t iw = ow * stride + kw - pad;
                                if (iw < 0 || iw >= W) continue;
                                sum += in_plane[ih * in_s2 + iw] *
                                       ker_plane[kh * k_s2 + kw];
                            }
                        }
                    }
                    out[b * o_s0 + oc * o_s1 + oh * o_s2 + ow] = sum;
                }
            }
        }
    }
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ESHKOL_TENSOR_CONV_KERNEL_H */
