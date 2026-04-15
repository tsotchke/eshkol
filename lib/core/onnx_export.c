/**
 * @file onnx_export.c
 * @brief Minimal ONNX model export for Eshkol tensor checkpoints.
 *
 * Writes tensors in ONNX protobuf wire format. No protobuf library needed —
 * the wire format is encoded manually. Supports:
 *   - Named tensors with double (float64) elements
 *   - Multi-dimensional shapes
 *   - ONNX opset 13+ compatible
 *
 * ONNX format: ModelProto → GraphProto → TensorProto (initializers)
 * Wire format: protobuf varint + length-delimited fields
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Forward declarations */
extern void* get_global_arena(void);
extern char* arena_allocate_string_with_header(void* arena, size_t length);

/* ── Protobuf wire format helpers ── */

/* Write a varint (variable-length integer) to buffer. Returns bytes written. */
static size_t pb_write_varint(uint8_t* buf, uint64_t value) {
    size_t n = 0;
    while (value > 0x7F) {
        buf[n++] = (uint8_t)(value & 0x7F) | 0x80;
        value >>= 7;
    }
    buf[n++] = (uint8_t)value;
    return n;
}

/* Write a field tag (field_number << 3 | wire_type) */
static size_t pb_write_tag(uint8_t* buf, uint32_t field, uint32_t wire_type) {
    return pb_write_varint(buf, ((uint64_t)field << 3) | wire_type);
}

/* Wire types */
#define PB_VARINT  0
#define PB_FIXED64 1
#define PB_LENDELIM 2
#define PB_FIXED32 5

/* ONNX data types */
#define ONNX_DOUBLE 11   /* TensorProto.DataType.DOUBLE */
#define ONNX_FLOAT  1    /* TensorProto.DataType.FLOAT */

/* ── TensorProto encoding ── */

/* Compute size of a TensorProto message for given tensor */
static size_t tensor_proto_size(const char* name, const int64_t* dims, int ndims,
                                 const double* data, int64_t total) {
    size_t size = 0;
    uint8_t tmp[16];

    /* Field 1: dims (repeated int64, varint encoding) */
    for (int i = 0; i < ndims; i++) {
        size += pb_write_tag(tmp, 1, PB_VARINT);
        size += pb_write_varint(tmp, (uint64_t)dims[i]);
    }

    /* Field 2: data_type (int32, varint) */
    size += pb_write_tag(tmp, 2, PB_VARINT);
    size += pb_write_varint(tmp, ONNX_DOUBLE);

    /* Field 5: double_data (packed repeated double = length-delimited) */
    size_t data_bytes = (size_t)total * sizeof(double);
    size += pb_write_tag(tmp, 5, PB_LENDELIM);
    size += pb_write_varint(tmp, data_bytes);
    size += data_bytes;

    /* Field 8: name (string = length-delimited) */
    if (name) {
        size_t name_len = strlen(name);
        size += pb_write_tag(tmp, 8, PB_LENDELIM);
        size += pb_write_varint(tmp, name_len);
        size += name_len;
    }

    return size;
}

/* Write a TensorProto message to buffer. Returns bytes written. */
static size_t write_tensor_proto(uint8_t* buf, const char* name,
                                  const int64_t* dims, int ndims,
                                  const double* data, int64_t total) {
    size_t pos = 0;

    /* Field 1: dims */
    for (int i = 0; i < ndims; i++) {
        pos += pb_write_tag(buf + pos, 1, PB_VARINT);
        pos += pb_write_varint(buf + pos, (uint64_t)dims[i]);
    }

    /* Field 2: data_type = DOUBLE */
    pos += pb_write_tag(buf + pos, 2, PB_VARINT);
    pos += pb_write_varint(buf + pos, ONNX_DOUBLE);

    /* Field 5: double_data (packed) */
    size_t data_bytes = (size_t)total * sizeof(double);
    pos += pb_write_tag(buf + pos, 5, PB_LENDELIM);
    pos += pb_write_varint(buf + pos, data_bytes);
    memcpy(buf + pos, data, data_bytes);
    pos += data_bytes;

    /* Field 8: name */
    if (name) {
        size_t name_len = strlen(name);
        pos += pb_write_tag(buf + pos, 8, PB_LENDELIM);
        pos += pb_write_varint(buf + pos, name_len);
        memcpy(buf + pos, name, name_len);
        pos += name_len;
    }

    return pos;
}

/* ── Public API ── */

/**
 * Export tensors to ONNX format.
 *
 * @param path Output file path
 * @param names Array of tensor names
 * @param dims Array of dimension arrays (one per tensor)
 * @param ndims Array of dimension counts
 * @param data Array of data pointers
 * @param totals Array of total element counts
 * @param n_tensors Number of tensors
 * @return 0 on success, -1 on failure
 */
int eshkol_onnx_export(const char* path,
                        const char** names,
                        const int64_t** dims,
                        const int* ndims,
                        const double** data,
                        const int64_t* totals,
                        int n_tensors) {
    if (!path || n_tensors <= 0) return -1;

    /* Compute total size of all TensorProto messages */
    size_t* tensor_sizes = (size_t*)malloc((size_t)n_tensors * sizeof(size_t));
    if (!tensor_sizes) return -1;

    size_t total_tensor_bytes = 0;
    for (int i = 0; i < n_tensors; i++) {
        tensor_sizes[i] = tensor_proto_size(names[i], dims[i], ndims[i], data[i], totals[i]);
        total_tensor_bytes += tensor_sizes[i];
    }

    /* GraphProto size: initializer fields (field 5, length-delimited) */
    uint8_t tmp[16];
    size_t graph_size = 0;
    for (int i = 0; i < n_tensors; i++) {
        graph_size += pb_write_tag(tmp, 5, PB_LENDELIM);
        graph_size += pb_write_varint(tmp, tensor_sizes[i]);
        graph_size += tensor_sizes[i];
    }

    /* ModelProto fields:
     *   Field 1: ir_version (int64, varint) = 7
     *   Field 7: graph (GraphProto, length-delimited)
     *   Field 8: opset_import (OperatorSetIdProto, length-delimited)
     */

    /* opset_import: field 2 = version (int64) = 13 */
    size_t opset_size = 0;
    opset_size += pb_write_tag(tmp, 2, PB_VARINT);
    opset_size += pb_write_varint(tmp, 13);

    size_t model_size = 0;
    model_size += pb_write_tag(tmp, 1, PB_VARINT);   /* ir_version tag */
    model_size += pb_write_varint(tmp, 7);            /* ir_version value */
    model_size += pb_write_tag(tmp, 7, PB_LENDELIM);  /* graph tag */
    model_size += pb_write_varint(tmp, graph_size);    /* graph length */
    model_size += graph_size;
    model_size += pb_write_tag(tmp, 8, PB_LENDELIM);  /* opset_import tag */
    model_size += pb_write_varint(tmp, opset_size);    /* opset_import length */
    model_size += opset_size;

    /* Allocate buffer and write */
    uint8_t* buf = (uint8_t*)malloc(model_size + 64); /* +64 for safety */
    if (!buf) { free(tensor_sizes); return -1; }

    size_t pos = 0;

    /* ir_version = 7 */
    pos += pb_write_tag(buf + pos, 1, PB_VARINT);
    pos += pb_write_varint(buf + pos, 7);

    /* graph */
    pos += pb_write_tag(buf + pos, 7, PB_LENDELIM);
    pos += pb_write_varint(buf + pos, graph_size);

    /* graph.initializer (repeated TensorProto) */
    for (int i = 0; i < n_tensors; i++) {
        pos += pb_write_tag(buf + pos, 5, PB_LENDELIM);
        pos += pb_write_varint(buf + pos, tensor_sizes[i]);
        pos += write_tensor_proto(buf + pos, names[i], dims[i], ndims[i], data[i], totals[i]);
    }

    /* opset_import */
    pos += pb_write_tag(buf + pos, 8, PB_LENDELIM);
    pos += pb_write_varint(buf + pos, opset_size);
    pos += pb_write_tag(buf + pos, 2, PB_VARINT);
    pos += pb_write_varint(buf + pos, 13);

    /* Write to file */
    FILE* f = fopen(path, "wb");
    if (!f) { free(buf); free(tensor_sizes); return -1; }
    fwrite(buf, 1, pos, f);
    fclose(f);

    free(buf);
    free(tensor_sizes);
    return 0;
}
