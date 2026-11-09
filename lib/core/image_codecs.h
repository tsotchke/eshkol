/**
 * @file image_codecs.h
 * @brief Internal interface of the Eshkol runtime's built-in image codecs.
 *
 * The runtime decodes and encodes image files itself; it links no external
 * codec library on any platform.  image_io.c owns the public C API
 * (inc/eshkol/core/image_io.h), capability checks and the conversion between
 * 8-bit samples and normalized double tensors.  The codecs here work purely
 * on byte buffers:
 *
 *   - zlib streams (RFC 1950/1951): inflate, and a deflate encoder whose
 *     output is the byte stream zlib produces at its default level;
 *   - PNG (all critical-chunk colour types, bit depths and Adam7);
 *   - JPEG: baseline and progressive Huffman decoding, baseline encoding;
 *   - BMP: uncompressed and bit-field bitmaps, 24/32-bit writer.
 *
 * Decoders return interleaved 8-bit samples.  The channel layout follows the
 * contract image-read has always exposed: 3 channels (RGB), or 4 (RGBA) when
 * the file carries transparency.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */
#ifndef ESHKOL_CORE_IMAGE_CODECS_H
#define ESHKOL_CORE_IMAGE_CODECS_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace eshkol::image {

/** Largest width or height any codec accepts (matches image_io.c). */
constexpr int kMaxDimension = 65535;

/** A decoded raster: row-major, interleaved, 8 bits per sample. */
struct Raster {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> pixels;
};

/* ---- checksums --------------------------------------------------------- */

uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t size);
uint32_t adler32_update(uint32_t adler, const uint8_t* data, size_t size);

/* ---- zlib streams ------------------------------------------------------ */

/**
 * Inflate a zlib stream until @p expected bytes have been produced.
 *
 * Decoding stops as soon as the expected amount exists, so trailing data
 * (end-of-block code, Adler-32 trailer, padding) is not required, matching
 * how PNG readers consume image data.  Returns false when the header is
 * invalid, the stream is malformed before @p expected bytes are produced, or
 * the stream ends early.
 */
bool zlib_inflate_exact(const uint8_t* data, size_t size, size_t expected,
                        std::vector<uint8_t>* out);

/**
 * Streaming deflate encoder producing a zlib stream.
 *
 * The match finder, lazy evaluation, block splitting and Huffman tree
 * construction follow zlib's default compression level exactly, so for the
 * same input chunking the output bytes equal zlib's.  Strategy values follow
 * zlib: 0 = default, 1 = filtered.
 */
class DeflateEncoder {
public:
    DeflateEncoder(int window_bits, int strategy);
    ~DeflateEncoder();
    DeflateEncoder(const DeflateEncoder&) = delete;
    DeflateEncoder& operator=(const DeflateEncoder&) = delete;

    /** Feed one chunk of input (zlib Z_NO_FLUSH semantics). */
    void write(const uint8_t* data, size_t size);
    /** Finish the stream (zlib Z_FINISH) and append the Adler-32 trailer. */
    void finish();
    /** Compressed bytes produced so far. */
    const std::vector<uint8_t>& output() const { return out_; }

private:
    struct State;
    State* s_;
    std::vector<uint8_t> out_;
};

/* ---- PNG --------------------------------------------------------------- */

bool png_is_signature(const uint8_t* data, size_t size);
bool png_decode(const uint8_t* data, size_t size, Raster* out);
/** Encode 1 (gray), 3 (RGB) or 4 (RGBA) channel 8-bit samples. */
bool png_encode(const uint8_t* pixels, int width, int height, int channels,
                std::vector<uint8_t>* out);

/* ---- JPEG -------------------------------------------------------------- */

bool jpeg_is_signature(const uint8_t* data, size_t size);
/** Decode baseline or progressive Huffman JPEG to RGB (3 channels). */
bool jpeg_decode(const uint8_t* data, size_t size, Raster* out);
/** Encode 1 (gray), 3 (RGB) or 4 (RGBA, alpha dropped) channels, baseline. */
bool jpeg_encode(const uint8_t* pixels, int width, int height, int channels,
                 int quality, std::vector<uint8_t>* out);

/* ---- BMP --------------------------------------------------------------- */

bool bmp_is_signature(const uint8_t* data, size_t size);
bool bmp_decode(const uint8_t* data, size_t size, Raster* out);
bool bmp_encode(const uint8_t* pixels, int width, int height, int channels,
                std::vector<uint8_t>* out);

/* ---- WebP -------------------------------------------------------------- */

bool webp_is_signature(const uint8_t* data, size_t size);

} // namespace eshkol::image

#endif /* ESHKOL_CORE_IMAGE_CODECS_H */
