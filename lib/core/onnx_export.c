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
 * Wire format: protobuf varint + length-delimited fields, written through the
 * shared wire primitives in inc/eshkol/core/protobuf_wire.h.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "eshkol/core/protobuf_wire.h"

/* Arena allocation */
typedef struct arena arena_t;
extern arena_t* get_global_arena(void);
extern void* arena_allocate(arena_t* arena, size_t size);
extern char* arena_allocate_string_with_header(arena_t* arena, size_t length);

/* ── Protobuf wire format ──
 * Sizes and bytes both come from inc/eshkol/core/protobuf_wire.h, so the
 * size pass and the write pass cannot disagree about a varint's length. */

#define PB_VARINT   ESHKOL_PB_WIRE_VARINT
#define PB_LENDELIM ESHKOL_PB_WIRE_LEN

/* Size of a tag followed by a varint value. */
static size_t pb_varint_field_size(uint32_t field, uint64_t value) {
    return eshkol_pb_tag_size(field, PB_VARINT) + eshkol_pb_varint_size(value);
}

/* Size of a tag and length prefix for a length-delimited payload of `len`
 * bytes, excluding the payload itself. */
static size_t pb_len_header_size(uint32_t field, uint64_t len) {
    return eshkol_pb_tag_size(field, PB_LENDELIM) + eshkol_pb_varint_size(len);
}

/* ONNX data types */
#define ONNX_DOUBLE 11   /* TensorProto.DataType.DOUBLE */
#define ONNX_FLOAT  1    /* TensorProto.DataType.FLOAT */

/* ── TensorProto encoding ── */

/* Compute size of a TensorProto message for given tensor */
static size_t tensor_proto_size(const char* name, const int64_t* dims, int ndims,
                                 const double* data, int64_t total) {
    size_t size = 0;

    /* Field 1: dims (repeated int64, varint encoding) */
    for (int i = 0; i < ndims; i++)
        size += pb_varint_field_size(1, (uint64_t)dims[i]);

    /* Field 2: data_type (int32, varint) */
    size += pb_varint_field_size(2, ONNX_DOUBLE);

    /* Field 10: double_data (packed repeated double = length-delimited).
     * TensorProto.proto: field 5 is int32_data, field 10 is double_data.
     * Previously off-by-five: onnx.checker rejected with "should be stored
     * in field 'double_data' instead of 'int32_data'". */
    size_t data_bytes = (size_t)total * sizeof(double);
    size += pb_len_header_size(10, data_bytes) + data_bytes;

    /* Field 8: name (string = length-delimited) */
    if (name) {
        size_t name_len = strlen(name);
        size += pb_len_header_size(8, name_len) + name_len;
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
        pos += eshkol_pb_write_tag(buf + pos, 1, PB_VARINT);
        pos += eshkol_pb_write_varint(buf + pos, (uint64_t)dims[i]);
    }

    /* Field 2: data_type = DOUBLE */
    pos += eshkol_pb_write_tag(buf + pos, 2, PB_VARINT);
    pos += eshkol_pb_write_varint(buf + pos, ONNX_DOUBLE);

    /* Field 10: double_data (packed) */
    size_t data_bytes = (size_t)total * sizeof(double);
    pos += eshkol_pb_write_tag(buf + pos, 10, PB_LENDELIM);
    pos += eshkol_pb_write_varint(buf + pos, data_bytes);
    memcpy(buf + pos, data, data_bytes);
    pos += data_bytes;

    /* Field 8: name */
    if (name) {
        size_t name_len = strlen(name);
        pos += eshkol_pb_write_tag(buf + pos, 8, PB_LENDELIM);
        pos += eshkol_pb_write_varint(buf + pos, name_len);
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
    size_t* tensor_sizes = (size_t*)arena_allocate(get_global_arena(), (size_t)n_tensors * sizeof(size_t));
    if (!tensor_sizes) return -1;

    size_t total_tensor_bytes = 0;
    for (int i = 0; i < n_tensors; i++) {
        tensor_sizes[i] = tensor_proto_size(names[i], dims[i], ndims[i], data[i], totals[i]);
        total_tensor_bytes += tensor_sizes[i];
    }

    /* GraphProto size. Required field: name (field 2, string).
     * Required enough that onnx.checker rejects models without it. */
    const char* graph_name = "eshkol_graph";
    size_t graph_name_len = strlen(graph_name);
    size_t graph_size = 0;
    graph_size += pb_len_header_size(2, graph_name_len) + graph_name_len;
    for (int i = 0; i < n_tensors; i++)
        graph_size += pb_len_header_size(5, tensor_sizes[i]) + tensor_sizes[i];

    /* ModelProto fields:
     *   Field 1: ir_version (int64, varint) = 7
     *   Field 7: graph (GraphProto, length-delimited)
     *   Field 8: opset_import (OperatorSetIdProto, length-delimited)
     */

    /* opset_import: field 2 = version (int64) = 13 */
    size_t opset_size = 0;
    opset_size += pb_varint_field_size(2, 13);

    size_t model_size = 0;
    model_size += pb_varint_field_size(1, 7);                         /* ir_version */
    model_size += pb_len_header_size(7, graph_size) + graph_size;     /* graph */
    model_size += pb_len_header_size(8, opset_size) + opset_size;     /* opset_import */

    /* Allocate buffer and write */
    uint8_t* buf = (uint8_t*)arena_allocate(get_global_arena(), model_size + 64);
    if (!buf) return -1;

    size_t pos = 0;

    /* ir_version = 7 */
    pos += eshkol_pb_write_tag(buf + pos, 1, PB_VARINT);
    pos += eshkol_pb_write_varint(buf + pos, 7);

    /* graph */
    pos += eshkol_pb_write_tag(buf + pos, 7, PB_LENDELIM);
    pos += eshkol_pb_write_varint(buf + pos, graph_size);

    /* graph.name (required by onnx.checker) */
    pos += eshkol_pb_write_tag(buf + pos, 2, PB_LENDELIM);
    pos += eshkol_pb_write_varint(buf + pos, graph_name_len);
    memcpy(buf + pos, graph_name, graph_name_len);
    pos += graph_name_len;

    /* graph.initializer (repeated TensorProto) */
    for (int i = 0; i < n_tensors; i++) {
        pos += eshkol_pb_write_tag(buf + pos, 5, PB_LENDELIM);
        pos += eshkol_pb_write_varint(buf + pos, tensor_sizes[i]);
        pos += write_tensor_proto(buf + pos, names[i], dims[i], ndims[i], data[i], totals[i]);
    }

    /* opset_import */
    pos += eshkol_pb_write_tag(buf + pos, 8, PB_LENDELIM);
    pos += eshkol_pb_write_varint(buf + pos, opset_size);
    pos += eshkol_pb_write_tag(buf + pos, 2, PB_VARINT);
    pos += eshkol_pb_write_varint(buf + pos, 13);

    /* Write to file.  Pre-fix, fwrite + fclose return values were
     * dropped — disk-full mid-export or commit-time error left a
     * truncated/corrupt .onnx file but onnx-export reported success. */
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    int wrote_ok = (fwrite(buf, 1, pos, f) == (size_t)pos);
    int closed_ok = (fclose(f) == 0);
    return (wrote_ok && closed_ok) ? 0 : -1;
}
