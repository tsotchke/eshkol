/*******************************************************************************
 * Compression Primitives for Eshkol Agent (B.14)
 *
 * Provides: deflate, inflate, gzip, gunzip via zlib.
 *
 * Compile with -DHAS_ZLIB and link -lz. Without HAS_ZLIB, returns -1 stubs.
 *
 * Copyright (c) 2025 Eshkol Project — tsotchke
 ******************************************************************************/

#include <stdint.h>
#include <string.h>

#ifdef HAS_ZLIB

#include <zlib.h>

/**
 * @brief Compresses @p data into zlib (RFC 1950) format using the default compression level.
 *
 * Wraps zlib's compress2(). The compressed output is written to @p buf,
 * whose capacity is given by @p buf_size.
 *
 * @return Number of compressed bytes written to @p buf, or -1 on invalid
 *         arguments or if @p buf is too small to hold the compressed data.
 */
int32_t eshkol_deflate(const char* data, int32_t data_len,
                        char* buf, int32_t buf_size) {
    if (!data || !buf || data_len <= 0 || buf_size <= 0) return -1;
    uLongf dest_len = (uLongf)buf_size;
    int r = compress2((Bytef*)buf, &dest_len,
                       (const Bytef*)data, (uLong)data_len, Z_DEFAULT_COMPRESSION);
    return r == Z_OK ? (int32_t)dest_len : -1;
}

/**
 * @brief Decompresses a zlib (RFC 1950) buffer produced by eshkol_deflate() back into @p buf.
 *
 * Wraps zlib's uncompress(). @p buf_size gives the caller-supplied capacity
 * of @p buf and is used as the initial estimate of the decompressed size.
 *
 * @return Number of decompressed bytes written to @p buf, or -1 on invalid
 *         arguments or decompression failure.
 */
int32_t eshkol_inflate_data(const char* data, int32_t data_len,
                             char* buf, int32_t buf_size) {
    if (!data || !buf || data_len <= 0 || buf_size <= 0) return -1;
    uLongf dest_len = (uLongf)buf_size;
    int r = uncompress((Bytef*)buf, &dest_len,
                        (const Bytef*)data, (uLong)data_len);
    return r == Z_OK ? (int32_t)dest_len : -1;
}

/**
 * @brief Compresses @p data into gzip (RFC 1952) format.
 *
 * Configures a zlib deflate stream with windowBits = 15+16 so the output
 * includes a gzip header/trailer, then compresses the entire input in one
 * Z_FINISH call.
 *
 * @return Number of compressed bytes written to @p buf, or -1 if
 *         initialization fails, arguments are invalid, or @p buf is too
 *         small to hold the full compressed stream.
 */
int32_t eshkol_gzip(const char* data, int32_t data_len,
                      char* buf, int32_t buf_size) {
    if (!data || !buf || data_len <= 0 || buf_size <= 0) return -1;
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    /* windowBits=15+16 for gzip format */
    if (deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                      15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) return -1;
    strm.next_in = (Bytef*)data;
    strm.avail_in = (uInt)data_len;
    strm.next_out = (Bytef*)buf;
    strm.avail_out = (uInt)buf_size;
    int r = deflate(&strm, Z_FINISH);
    int32_t result = (r == Z_STREAM_END) ? (int32_t)strm.total_out : -1;
    deflateEnd(&strm);
    return result;
}

/**
 * @brief Decompresses a gzip (RFC 1952) buffer produced by eshkol_gzip() (or any gzip stream).
 *
 * Configures a zlib inflate stream with windowBits = 15+16 to accept the
 * gzip header/trailer, then decompresses the entire input in one Z_FINISH
 * call.
 *
 * @return Number of decompressed bytes written to @p buf, or -1 if
 *         initialization fails, arguments are invalid, or decompression
 *         does not complete in a single pass.
 */
int32_t eshkol_gunzip(const char* data, int32_t data_len,
                        char* buf, int32_t buf_size) {
    if (!data || !buf || data_len <= 0 || buf_size <= 0) return -1;
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    /* windowBits=15+16 for gzip format */
    if (inflateInit2(&strm, 15 + 16) != Z_OK) return -1;
    strm.next_in = (Bytef*)data;
    strm.avail_in = (uInt)data_len;
    strm.next_out = (Bytef*)buf;
    strm.avail_out = (uInt)buf_size;
    int r = inflate(&strm, Z_FINISH);
    int32_t result = (r == Z_STREAM_END) ? (int32_t)strm.total_out : -1;
    inflateEnd(&strm);
    return result;
}

/** @brief Reports that zlib-backed compression support is compiled in. @return Always 1. */
int32_t eshkol_compression_available(void) { return 1; }

#else /* !HAS_ZLIB */

/** @brief Stub used when built without HAS_ZLIB; deflate compression is unavailable. @return Always -1. */
int32_t eshkol_deflate(const char* d, int32_t dl, char* b, int32_t bs)
    { (void)d;(void)dl;(void)b;(void)bs; return -1; }
/** @brief Stub used when built without HAS_ZLIB; inflate decompression is unavailable. @return Always -1. */
int32_t eshkol_inflate_data(const char* d, int32_t dl, char* b, int32_t bs)
    { (void)d;(void)dl;(void)b;(void)bs; return -1; }
/** @brief Stub used when built without HAS_ZLIB; gzip compression is unavailable. @return Always -1. */
int32_t eshkol_gzip(const char* d, int32_t dl, char* b, int32_t bs)
    { (void)d;(void)dl;(void)b;(void)bs; return -1; }
/** @brief Stub used when built without HAS_ZLIB; gunzip decompression is unavailable. @return Always -1. */
int32_t eshkol_gunzip(const char* d, int32_t dl, char* b, int32_t bs)
    { (void)d;(void)dl;(void)b;(void)bs; return -1; }
/** @brief Reports that zlib-backed compression support is not compiled in. @return Always 0. */
int32_t eshkol_compression_available(void) { return 0; }

#endif
