/**
 * @file image_io.h
 * @brief Image I/O API — read/write PNG, JPEG, BMP as tensor data.
 *
 * Images are normalized doubles (0.0-1.0). Shapes: (H,W,C) for color,
 * (H,W) for grayscale. Caller must free() returned arrays.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */
#ifndef ESHKOL_IMAGE_IO_H
#define ESHKOL_IMAGE_IO_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Read image file → flat double array (normalized 0-1, row-major, HWC layout).
 * @param path      Path to image file (PNG, JPEG, BMP, TGA, GIF, PSD, HDR, PIC)
 * @param out_w     Output: image width
 * @param out_h     Output: image height
 * @param out_c     Output: number of channels (1=gray, 3=RGB, 4=RGBA)
 * @return malloc'd array of (w*h*c) doubles, or NULL on failure. Caller frees.
 */
double* eshkol_image_read(const char* path, int* out_w, int* out_h, int* out_c);

/**
 * Write tensor data to image file.
 * @param path      Output file path
 * @param data      Flat double array (normalized 0-1)
 * @param w         Image width
 * @param h         Image height
 * @param channels  Number of channels (1, 3, or 4)
 * @param format    "png", "jpg"/"jpeg", "bmp", "tga" (NULL defaults to "png")
 * @return 0 on success, -1 on failure
 */
int eshkol_image_write(const char* path, const double* data,
                       int w, int h, int channels, const char* format);

/**
 * Convert color image to grayscale using ITU-R BT.709 luminance.
 * @return malloc'd array of (w*h) doubles, or NULL. Caller frees.
 */
double* eshkol_image_to_grayscale(const double* data, int w, int h, int channels);

/**
 * Resize image using high-quality interpolation.
 * @return malloc'd array of (new_w*new_h*channels) doubles, or NULL. Caller frees.
 */
double* eshkol_image_resize(const double* data, int w, int h, int channels,
                            int new_w, int new_h);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_IMAGE_IO_H */
