/*******************************************************************************
 * Tensor Persistence for Eshkol Agent
 *
 * Save/load tensors to binary files for persistent neural state across
 * agent sessions (embeddings, learned weights, token estimation models).
 *
 * File format (ESHT v1):
 *   4 bytes: magic "ESHT"
 *   4 bytes: version (1)
 *   4 bytes: ndim
 *   4 bytes: reserved (0)
 *   ndim * 8 bytes: shape (int64_t per dimension)
 *   total * 8 bytes: data (float64, row-major)
 *
 * Copyright (c) 2025 Eshkol Project — tsotchke
 ******************************************************************************/

#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#else
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define ESHT_MAGIC 0x54485345  /* "ESHT" as little-endian uint32 */
#define ESHT_VERSION 1

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t ndim;
    uint32_t reserved;
} EshtHeader;

/**
 * @brief Writes tensor data and shape to a binary file in the ESHT v1 format.
 *
 * Validates that @p total matches the product of @p shape before writing the
 * fixed EshtHeader, the shape array, and the raw float64 data in sequence.
 *
 * @param data Contiguous row-major float64 array of @p total elements.
 * @param shape Array of @p ndim dimension sizes.
 * @param ndim Number of dimensions (1-32).
 * @param total Total element count; must equal the product of @p shape.
 * @param path Output file path (overwritten if it exists).
 * @return 0 on success, -1 on invalid arguments, a shape/total mismatch, or
 *         a file I/O error.
 */
/*
 * Save tensor data to a binary file.
 *
 * data:  pointer to contiguous float64 array
 * shape: pointer to int64_t array of dimension sizes
 * ndim:  number of dimensions
 * total: total number of elements (product of shape)
 * path:  output file path
 *
 * Returns: 0 success, -1 error
 */
int32_t eshkol_tensor_save(const double* data, const int64_t* shape,
                             int32_t ndim, int64_t total, const char* path) {
    if (!data || !shape || ndim <= 0 || ndim > 32 || total <= 0 || !path)
        return -1;

    /* Verify total matches product of shape */
    int64_t computed_total = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return -1;
        computed_total *= shape[i];
    }
    if (computed_total != total) return -1;

    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    EshtHeader hdr = {
        .magic = ESHT_MAGIC,
        .version = ESHT_VERSION,
        .ndim = (uint32_t)ndim,
        .reserved = 0
    };

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) goto fail;
    if (fwrite(shape, sizeof(int64_t), (size_t)ndim, f) != (size_t)ndim) goto fail;
    if (fwrite(data, sizeof(double), (size_t)total, f) != (size_t)total) goto fail;

    fclose(f);
    return 0;

fail:
    fclose(f);
    return -1;
}

/**
 * @brief Reads a tensor previously written by eshkol_tensor_save() from a
 * binary ESHT v1 file.
 *
 * Validates the file's magic/version header and each shape dimension
 * (rejecting non-positive dimensions), then mallocs a buffer for the data
 * and reads it in.
 *
 * @param path Input file path.
 * @param data_out Receives a malloc'd float64 array; caller must free it via
 *        eshkol_tensor_free_loaded(). Set to NULL on failure.
 * @param shape_out Caller-allocated array of at least 32 int64_t; receives
 *        the tensor's per-dimension sizes.
 * @param ndim_out Receives the number of dimensions.
 * @param total_out Receives the total element count (product of shape).
 * @return 0 on success, -1 on invalid arguments, missing/corrupt file, or OOM.
 */
/*
 * Load tensor data from a binary file.
 *
 * path:      input file path
 * data_out:  receives malloc'd float64 array (caller must free via eshkol_tensor_free_loaded)
 * shape_out: receives shape (must be pre-allocated, at least 32 int64_t)
 * ndim_out:  receives number of dimensions
 * total_out: receives total number of elements
 *
 * Returns: 0 success, -1 error (file not found, corrupt, OOM)
 */
int32_t eshkol_tensor_load(const char* path, double** data_out,
                             int64_t* shape_out, int32_t* ndim_out,
                             int64_t* total_out) {
    if (!path || !data_out || !shape_out || !ndim_out || !total_out)
        return -1;

    *data_out = NULL;
    *ndim_out = 0;
    *total_out = 0;

    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    EshtHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) goto fail;
    if (hdr.magic != ESHT_MAGIC || hdr.version != ESHT_VERSION) goto fail;
    if (hdr.ndim == 0 || hdr.ndim > 32) goto fail;

    int32_t ndim = (int32_t)hdr.ndim;
    if (fread(shape_out, sizeof(int64_t), (size_t)ndim, f) != (size_t)ndim) goto fail;

    int64_t total = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape_out[i] <= 0) goto fail;
        total *= shape_out[i];
        /* Sanity check: no single tensor > 1GB */
        if (total > 134217728) goto fail;  /* 128M elements * 8 bytes = 1GB */
    }

    double* data = (double*)malloc((size_t)total * sizeof(double));
    if (!data) goto fail;

    if (fread(data, sizeof(double), (size_t)total, f) != (size_t)total) {
        free(data);
        goto fail;
    }

    fclose(f);
    *data_out = data;
    *ndim_out = ndim;
    *total_out = total;
    return 0;

fail:
    fclose(f);
    return -1;
}

/**
 * @brief Frees a float64 buffer previously returned by eshkol_tensor_load().
 *
 * @param data Buffer to free (NULL is safely ignored, as with free()).
 */
/*
 * Free tensor data allocated by eshkol_tensor_load.
 */
void eshkol_tensor_free_loaded(double* data) {
    free(data);
}

/**
 * @brief Reads an ESHT file's header and shape without loading its tensor data.
 *
 * Validates the magic/version header, then optionally reads the shape array
 * and computes the total element count, without ever reading the (much
 * larger) data payload.
 *
 * @param path Input file path.
 * @param ndim_out Receives the number of dimensions.
 * @param shape_out Optional caller-allocated array receiving per-dimension
 *        sizes; pass NULL to skip reading the shape.
 * @param total_out Optional pointer receiving the total element count
 *        (only computed if @p shape_out is also non-NULL).
 * @return 0 on success, -1 on invalid arguments, missing file, or a bad/
 *         mismatched header.
 */
/*
 * Get tensor file metadata without loading data.
 *
 * Returns: 0 success, -1 error
 */
int32_t eshkol_tensor_file_info(const char* path, int32_t* ndim_out,
                                  int64_t* shape_out, int64_t* total_out) {
    if (!path || !ndim_out) return -1;

    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    EshtHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) goto fail;
    if (hdr.magic != ESHT_MAGIC || hdr.version != ESHT_VERSION) goto fail;

    *ndim_out = (int32_t)hdr.ndim;

    if (shape_out && hdr.ndim > 0) {
        if (fread(shape_out, sizeof(int64_t), hdr.ndim, f) != hdr.ndim) goto fail;

        if (total_out) {
            int64_t total = 1;
            for (uint32_t i = 0; i < hdr.ndim; i++) total *= shape_out[i];
            *total_out = total;
        }
    }

    fclose(f);
    return 0;

fail:
    fclose(f);
    return -1;
}
