/*******************************************************************************
 * Compression Primitives for Eshkol Agent (B.14)
 *
 * Provides: deflate, inflate, gzip, gunzip via zlib.
 *
 * zlib 1.3.1 is pinned, built, and packaged by the release CMake graph.
 * This translation unit intentionally has no unavailable/stub branch: a
 * production agent build must fail at configure/link time if zlib is absent.
 *
 * Copyright (c) 2025 Eshkol Project — tsotchke
 ******************************************************************************/

#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <eshkol/agent_capabilities.h>
#include <zlib.h>

static const Bytef g_empty_input = 0;

static const Bytef* compression_input(const char* data, int32_t data_len) {
    if (data_len < 0 || (!data && data_len != 0)) return NULL;
    return data_len == 0 ? &g_empty_input : (const Bytef*)data;
}

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
    const Bytef* input = compression_input(data, data_len);
    if (!input || !buf || buf_size <= 0) return -1;
    uLongf dest_len = (uLongf)buf_size;
    int r = compress2((Bytef*)buf, &dest_len,
                       input, (uLong)data_len, Z_DEFAULT_COMPRESSION);
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
    const Bytef* input = compression_input(data, data_len);
    if (!input || !buf || buf_size <= 0) return -1;
    uLongf dest_len = (uLongf)buf_size;
    int r = uncompress((Bytef*)buf, &dest_len,
                        input, (uLong)data_len);
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
    const Bytef* input = compression_input(data, data_len);
    if (!input || !buf || buf_size <= 0) return -1;
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    /* windowBits=15+16 for gzip format */
    if (deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                      15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) return -1;
    strm.next_in = (Bytef*)input;
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
    const Bytef* input = compression_input(data, data_len);
    if (!input || !buf || buf_size <= 0) return -1;
    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    /* windowBits=15+16 for gzip format */
    if (inflateInit2(&strm, 15 + 16) != Z_OK) return -1;
    strm.next_in = (Bytef*)input;
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

static int32_t compression_alloc_compress(const char* data, int32_t data_len,
                                           int32_t max_output, int gzip_format,
                                           char** out, int32_t* out_len) {
    const Bytef* input = compression_input(data, data_len);
    if (!input || !out || !out_len || max_output <= 0) return -1;
    *out = NULL;
    *out_len = 0;

    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    int init = deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                            gzip_format ? 15 + 16 : 15, 8,
                            Z_DEFAULT_STRATEGY);
    if (init != Z_OK) return -1;

    uLong bound = deflateBound(&strm, (uLong)data_len);
    if (bound == 0) bound = 1;
    if (bound > (uLong)max_output || bound > (uLong)INT32_MAX) {
        deflateEnd(&strm);
        return -1;
    }

    char* buffer = (char*)malloc((size_t)bound);
    if (!buffer) {
        deflateEnd(&strm);
        return -1;
    }
    strm.next_in = (Bytef*)input;
    strm.avail_in = (uInt)data_len;
    strm.next_out = (Bytef*)buffer;
    strm.avail_out = (uInt)bound;
    int rc = deflate(&strm, Z_FINISH);
    if (rc != Z_STREAM_END || strm.total_out > (uLong)INT32_MAX) {
        free(buffer);
        deflateEnd(&strm);
        return -1;
    }
    *out = buffer;
    *out_len = (int32_t)strm.total_out;
    deflateEnd(&strm);
    return 0;
}

static int32_t compression_alloc_decompress(const char* data, int32_t data_len,
                                             int32_t max_output, int gzip_format,
                                             char** out, int32_t* out_len) {
    const Bytef* input = compression_input(data, data_len);
    if (!input || !out || !out_len || max_output <= 0) return -1;
    *out = NULL;
    *out_len = 0;

    z_stream strm;
    memset(&strm, 0, sizeof(strm));
    if (inflateInit2(&strm, gzip_format ? 15 + 16 : 15) != Z_OK) return -1;

    size_t capacity = (size_t)data_len * 4u;
    if (capacity < 65536u) capacity = 65536u;
    if (capacity > (size_t)max_output) capacity = (size_t)max_output;
    if (capacity == 0) capacity = 1;
    char* buffer = (char*)malloc(capacity);
    if (!buffer) {
        inflateEnd(&strm);
        return -1;
    }

    strm.next_in = (Bytef*)input;
    strm.avail_in = (uInt)data_len;
    int rc = Z_OK;
    for (;;) {
        size_t produced = (size_t)strm.total_out;
        if (produced == capacity) {
            if (capacity >= (size_t)max_output) {
                rc = Z_BUF_ERROR;
                break;
            }
            size_t next = capacity > (size_t)max_output / 2u
                ? (size_t)max_output : capacity * 2u;
            char* grown = (char*)realloc(buffer, next);
            if (!grown) {
                rc = Z_MEM_ERROR;
                break;
            }
            buffer = grown;
            capacity = next;
        }
        strm.next_out = (Bytef*)buffer + strm.total_out;
        strm.avail_out = (uInt)(capacity - (size_t)strm.total_out);
        rc = inflate(&strm, Z_NO_FLUSH);
        if (rc == Z_STREAM_END) break;
        if (rc != Z_OK) break;
        if (strm.avail_in == 0 && strm.avail_out != 0) {
            rc = Z_DATA_ERROR;
            break;
        }
    }

    if (rc != Z_STREAM_END || strm.total_out > (uLong)INT32_MAX) {
        free(buffer);
        inflateEnd(&strm);
        return -1;
    }
    *out = buffer;
    *out_len = (int32_t)strm.total_out;
    inflateEnd(&strm);
    return 0;
}

int32_t eshkol_deflate_alloc(const char* data, int32_t data_len,
                             int32_t max_output, char** out, int32_t* out_len) {
    return compression_alloc_compress(data, data_len, max_output, 0, out, out_len);
}

int32_t eshkol_inflate_alloc(const char* data, int32_t data_len,
                             int32_t max_output, char** out, int32_t* out_len) {
    return compression_alloc_decompress(data, data_len, max_output, 0, out, out_len);
}

int32_t eshkol_gzip_alloc(const char* data, int32_t data_len,
                          int32_t max_output, char** out, int32_t* out_len) {
    return compression_alloc_compress(data, data_len, max_output, 1, out, out_len);
}

int32_t eshkol_gunzip_alloc(const char* data, int32_t data_len,
                            int32_t max_output, char** out, int32_t* out_len) {
    return compression_alloc_decompress(data, data_len, max_output, 1, out, out_len);
}

void eshkol_compression_free(void* ptr) { free(ptr); }
