/**
 * @file vm_model_io.c
 * @brief Tensor/model checkpoint serialization helpers for the bytecode VM.
 */

#ifndef VM_MODEL_IO_C_INCLUDED
#define VM_MODEL_IO_C_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VM_MODEL_IO_BASE 800

static const unsigned char VM_MODEL_MAGIC[4] = {'E', 'S', 'K', 'M'};
static const unsigned int VM_MODEL_VERSION = 1u;

static unsigned int vm_model_crc32_update(unsigned int crc, const unsigned char* data, size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; bit++) {
            unsigned int mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

typedef struct {
    FILE* file;
    unsigned int crc;
    int ok;
} VmModelWriter;

static int vm_model_write_bytes(VmModelWriter* writer, const void* data, size_t size, int include_crc) {
    if (!writer || !writer->file || !writer->ok) return 0;
    if (size > 0 && fwrite(data, 1, size, writer->file) != size) {
        writer->ok = 0;
        return 0;
    }
    if (include_crc && size > 0) {
        writer->crc = vm_model_crc32_update(writer->crc, (const unsigned char*)data, size);
    }
    return 1;
}

static int vm_model_write_u8(VmModelWriter* writer, unsigned char value, int include_crc) {
    return vm_model_write_bytes(writer, &value, 1, include_crc);
}

static int vm_model_write_u32(VmModelWriter* writer, unsigned int value, int include_crc) {
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(value & 0xFFu);
    bytes[1] = (unsigned char)((value >> 8) & 0xFFu);
    bytes[2] = (unsigned char)((value >> 16) & 0xFFu);
    bytes[3] = (unsigned char)((value >> 24) & 0xFFu);
    return vm_model_write_bytes(writer, bytes, sizeof(bytes), include_crc);
}

static int vm_model_write_u64(VmModelWriter* writer, uint64_t value, int include_crc) {
    unsigned char bytes[8];
    bytes[0] = (unsigned char)(value & 0xFFu);
    bytes[1] = (unsigned char)((value >> 8) & 0xFFu);
    bytes[2] = (unsigned char)((value >> 16) & 0xFFu);
    bytes[3] = (unsigned char)((value >> 24) & 0xFFu);
    bytes[4] = (unsigned char)((value >> 32) & 0xFFu);
    bytes[5] = (unsigned char)((value >> 40) & 0xFFu);
    bytes[6] = (unsigned char)((value >> 48) & 0xFFu);
    bytes[7] = (unsigned char)((value >> 56) & 0xFFu);
    return vm_model_write_bytes(writer, bytes, sizeof(bytes), include_crc);
}

static int vm_model_read_u32(const unsigned char* data, size_t size, size_t* offset, unsigned int* out) {
    if (!offset || !out || *offset + 4 > size) return 0;
    *out = (unsigned int)data[*offset] |
           ((unsigned int)data[*offset + 1] << 8) |
           ((unsigned int)data[*offset + 2] << 16) |
           ((unsigned int)data[*offset + 3] << 24);
    *offset += 4;
    return 1;
}

static int vm_model_read_u64(const unsigned char* data, size_t size, size_t* offset, uint64_t* out) {
    if (!offset || !out || *offset + 8 > size) return 0;
    *out = (uint64_t)data[*offset] |
           ((uint64_t)data[*offset + 1] << 8) |
           ((uint64_t)data[*offset + 2] << 16) |
           ((uint64_t)data[*offset + 3] << 24) |
           ((uint64_t)data[*offset + 4] << 32) |
           ((uint64_t)data[*offset + 5] << 40) |
           ((uint64_t)data[*offset + 6] << 48) |
           ((uint64_t)data[*offset + 7] << 56);
    *offset += 8;
    return 1;
}

static int vm_model_read_u8(const unsigned char* data, size_t size, size_t* offset, unsigned char* out) {
    if (!offset || !out || *offset + 1 > size) return 0;
    *out = data[*offset];
    *offset += 1;
    return 1;
}

static int vm_model_compute_total(const uint64_t* dims, unsigned int ndims, int64_t* total) {
    uint64_t value = 1;
    for (unsigned int i = 0; i < ndims; i++) {
        if (dims[i] == 0) {
            value = 0;
            break;
        }
        if (value > UINT64_MAX / dims[i]) return 0;
        value *= dims[i];
    }
    if (value > INT64_MAX) return 0;
    *total = (int64_t)value;
    return 1;
}

static const char* vm_model_string_ptr(VM* vm, Value value, int* len) {
    if (!vm || value.type != VAL_STRING || value.as.ptr < 0) return NULL;
    VmString* str = (VmString*)vm->heap.objects[value.as.ptr]->opaque.ptr;
    if (!str) return NULL;
    if (len) *len = str->byte_len;
    return str->data;
}

static VmTensor* vm_model_value_tensor(VM* vm, Value value) {
    if (!vm || value.type != VAL_TENSOR || value.as.ptr < 0) return NULL;
    return (VmTensor*)vm->heap.objects[value.as.ptr]->opaque.ptr;
}

static int vm_model_make_string_value(VM* vm, const char* data, int len, Value* out) {
    if (!vm || !out || len < 0) return 0;
    char* buf = (char*)vm_alloc(&vm->heap.regions, (size_t)len + 1);
    if (!buf) return 0;
    if (len > 0) memcpy(buf, data, (size_t)len);
    buf[len] = '\0';
    VmString* str = vm_string_new(&vm->heap.regions, buf, len);
    if (!str) return 0;
    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) return 0;
    vm->heap.objects[ptr]->type = HEAP_STRING;
    vm->heap.objects[ptr]->opaque.ptr = str;
    *out = (Value){.type = VAL_STRING, .as.ptr = ptr};
    return 1;
}

static int vm_model_make_tensor_value(VM* vm,
                                      unsigned int ndims,
                                      const uint64_t* dims,
                                      const unsigned char* data,
                                      size_t size,
                                      size_t* offset,
                                      Value* out) {
    if (!vm || !offset || !out) return 0;

    int64_t shape[VM_TENSOR_MAX_DIMS];
    if (ndims > VM_TENSOR_MAX_DIMS) return 0;
    for (unsigned int i = 0; i < ndims; i++) shape[i] = (int64_t)dims[i];

    int64_t total = 1;
    if (!vm_model_compute_total(dims, ndims, &total)) return 0;
    if (*offset + (size_t)total * 8 > size) return 0;

    double* elements = NULL;
    if (total > 0) {
        elements = (double*)malloc((size_t)total * sizeof(double));
        if (!elements) return 0;
    }

    for (int64_t i = 0; i < total; i++) {
        uint64_t bits = 0;
        if (!vm_model_read_u64(data, size, offset, &bits)) {
            free(elements);
            return 0;
        }
        union { uint64_t u; double d; } conv;
        conv.u = bits;
        if (elements) elements[i] = conv.d;
    }

    VmTensor* tensor = vm_tensor_from_data(&vm->heap.regions, elements, shape, (int)ndims);
    free(elements);
    if (!tensor) return 0;

    int32_t ptr = heap_alloc(&vm->heap);
    if (ptr < 0) return 0;
    vm->heap.objects[ptr]->type = HEAP_TENSOR;
    vm->heap.objects[ptr]->opaque.ptr = tensor;
    *out = (Value){.type = VAL_TENSOR, .as.ptr = ptr};
    return 1;
}

static int vm_model_write_tensor_record(VmModelWriter* writer,
                                        const char* name,
                                        int name_len,
                                        const VmTensor* tensor) {
    if (!writer || !tensor || name_len < 0) return 0;
    if (!vm_model_write_u32(writer, (unsigned int)name_len, 1)) return 0;
    if (name_len > 0 && !vm_model_write_bytes(writer, name, (size_t)name_len, 1)) return 0;
    if (!vm_model_write_u32(writer, (unsigned int)tensor->n_dims, 1)) return 0;
    for (int i = 0; i < tensor->n_dims; i++) {
        if (!vm_model_write_u64(writer, (uint64_t)tensor->shape[i], 1)) return 0;
    }
    if (!vm_model_write_u8(writer, 0, 1)) return 0;
    for (int64_t i = 0; i < tensor->total; i++) {
        union { uint64_t u; double d; } conv;
        conv.d = tensor->data[i];
        if (!vm_model_write_u64(writer, conv.u, 1)) return 0;
    }
    return 1;
}

static int vm_model_list_length(Value list, VM* vm) {
    int count = 0;
    while (list.type == VAL_PAIR) {
        count++;
        list = vm->heap.objects[list.as.ptr]->cons.cdr;
    }
    return (list.type == VAL_NIL) ? count : -1;
}

static Value vm_model_reverse_list(VM* vm, Value list) {
    Value reversed = NIL_VAL;
    while (list.type == VAL_PAIR) {
        int32_t node = heap_alloc(&vm->heap);
        if (node < 0) return NIL_VAL;
        vm->heap.objects[node]->type = HEAP_CONS;
        vm->heap.objects[node]->cons.car = vm->heap.objects[list.as.ptr]->cons.car;
        vm->heap.objects[node]->cons.cdr = reversed;
        reversed = PAIR_VAL(node);
        list = vm->heap.objects[list.as.ptr]->cons.cdr;
    }
    return reversed;
}

static int vm_model_save_tensor_file(VM* vm, Value path_value, Value tensor_value) {
    const char* path = vm_model_string_ptr(vm, path_value, NULL);
    VmTensor* tensor = vm_model_value_tensor(vm, tensor_value);
    if (!path || !tensor) return 0;

    VmModelWriter writer = { fopen(path, "wb"), 0u, 1 };
    if (!writer.file) return 0;

    int ok = vm_model_write_bytes(&writer, VM_MODEL_MAGIC, sizeof(VM_MODEL_MAGIC), 1) &&
             vm_model_write_u32(&writer, VM_MODEL_VERSION, 1) &&
             vm_model_write_u32(&writer, 1u, 1) &&
             vm_model_write_u32(&writer, 0u, 1) &&
             vm_model_write_tensor_record(&writer, "", 0, tensor) &&
             vm_model_write_u32(&writer, writer.crc, 0);

    fclose(writer.file);
    return ok && writer.ok;
}

static int vm_model_save_model_file(VM* vm, Value path_value, Value entries_value) {
    const char* path = vm_model_string_ptr(vm, path_value, NULL);
    if (!path) return 0;

    int count = vm_model_list_length(entries_value, vm);
    if (count < 0) return 0;

    VmModelWriter writer = { fopen(path, "wb"), 0u, 1 };
    if (!writer.file) return 0;

    int ok = vm_model_write_bytes(&writer, VM_MODEL_MAGIC, sizeof(VM_MODEL_MAGIC), 1) &&
             vm_model_write_u32(&writer, VM_MODEL_VERSION, 1) &&
             vm_model_write_u32(&writer, (unsigned int)count, 1) &&
             vm_model_write_u32(&writer, 0u, 1);

    Value current = entries_value;
    while (ok && current.type == VAL_PAIR) {
        Value entry = vm->heap.objects[current.as.ptr]->cons.car;
        if (entry.type != VAL_PAIR) { ok = 0; break; }
        Value name_value = vm->heap.objects[entry.as.ptr]->cons.car;
        Value tensor_value = vm->heap.objects[entry.as.ptr]->cons.cdr;
        int name_len = 0;
        const char* name = vm_model_string_ptr(vm, name_value, &name_len);
        VmTensor* tensor = vm_model_value_tensor(vm, tensor_value);
        if (!name || !tensor) { ok = 0; break; }
        ok = vm_model_write_tensor_record(&writer, name, name_len, tensor);
        current = vm->heap.objects[current.as.ptr]->cons.cdr;
    }

    ok = ok && vm_model_write_u32(&writer, writer.crc, 0);
    fclose(writer.file);
    return ok && writer.ok;
}

static int vm_model_load_bytes(const char* path, unsigned char** data, size_t* size) {
    FILE* file = fopen(path, "rb");
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return 0; }
    long file_size = ftell(file);
    if (file_size < 0 || fseek(file, 0, SEEK_SET) != 0) { fclose(file); return 0; }
    *size = (size_t)file_size;
    *data = (unsigned char*)malloc(*size > 0 ? *size : 1);
    if (!*data) { fclose(file); return 0; }
    if (*size > 0 && fread(*data, 1, *size, file) != *size) {
        free(*data);
        fclose(file);
        return 0;
    }
    fclose(file);
    return 1;
}

static int vm_model_parse_header(const unsigned char* data,
                                 size_t size,
                                 unsigned int* tensor_count,
                                 size_t* payload_size,
                                 size_t* offset) {
    if (!data || size < 16 || !tensor_count || !payload_size || !offset) return 0;
    *payload_size = size - 4;
    unsigned int stored_crc = 0;
    size_t footer_offset = *payload_size;
    if (!vm_model_read_u32(data, size, &footer_offset, &stored_crc)) return 0;
    if (stored_crc != vm_model_crc32_update(0u, data, *payload_size)) return 0;
    if (memcmp(data, VM_MODEL_MAGIC, sizeof(VM_MODEL_MAGIC)) != 0) return 0;

    *offset = sizeof(VM_MODEL_MAGIC);
    unsigned int version = 0;
    unsigned int flags = 0;
    if (!vm_model_read_u32(data, *payload_size, offset, &version) ||
        !vm_model_read_u32(data, *payload_size, offset, tensor_count) ||
        !vm_model_read_u32(data, *payload_size, offset, &flags)) {
        return 0;
    }
    (void)flags;
    return version == VM_MODEL_VERSION;
}

static void vm_model_tensor_load(VM* vm) {
    Value path_value = vm_pop(vm);
    const char* path = vm_model_string_ptr(vm, path_value, NULL);
    if (!path) { vm_push(vm, NIL_VAL); return; }

    unsigned char* data = NULL;
    size_t size = 0;
    unsigned int tensor_count = 0;
    size_t payload_size = 0;
    size_t offset = 0;
    if (!vm_model_load_bytes(path, &data, &size) ||
        !vm_model_parse_header(data, size, &tensor_count, &payload_size, &offset) ||
        tensor_count != 1) {
        free(data);
        vm_push(vm, NIL_VAL);
        return;
    }

    unsigned int name_len = 0;
    unsigned int ndims = 0;
    if (!vm_model_read_u32(data, payload_size, &offset, &name_len) ||
        offset + name_len > payload_size) {
        free(data);
        vm_push(vm, NIL_VAL);
        return;
    }
    offset += name_len;
    if (!vm_model_read_u32(data, payload_size, &offset, &ndims)) {
        free(data);
        vm_push(vm, NIL_VAL);
        return;
    }

    uint64_t dims[VM_TENSOR_MAX_DIMS];
    if (ndims > VM_TENSOR_MAX_DIMS) {
        free(data);
        vm_push(vm, NIL_VAL);
        return;
    }
    for (unsigned int i = 0; i < ndims; i++) {
        if (!vm_model_read_u64(data, payload_size, &offset, &dims[i])) {
            free(data);
            vm_push(vm, NIL_VAL);
            return;
        }
    }
    unsigned char dtype = 0;
    if (!vm_model_read_u8(data, payload_size, &offset, &dtype) || dtype != 0) {
        free(data);
        vm_push(vm, NIL_VAL);
        return;
    }

    Value tensor_value;
    if (!vm_model_make_tensor_value(vm, ndims, dims, data, payload_size, &offset, &tensor_value) ||
        offset != payload_size) {
        free(data);
        vm_push(vm, NIL_VAL);
        return;
    }

    free(data);
    vm_push(vm, tensor_value);
}

static void vm_model_model_load(VM* vm) {
    Value path_value = vm_pop(vm);
    const char* path = vm_model_string_ptr(vm, path_value, NULL);
    if (!path) { vm_push(vm, NIL_VAL); return; }

    unsigned char* data = NULL;
    size_t size = 0;
    unsigned int tensor_count = 0;
    size_t payload_size = 0;
    size_t offset = 0;
    if (!vm_model_load_bytes(path, &data, &size) ||
        !vm_model_parse_header(data, size, &tensor_count, &payload_size, &offset)) {
        free(data);
        vm_push(vm, NIL_VAL);
        return;
    }

    Value list = NIL_VAL;
    for (unsigned int t = 0; t < tensor_count; t++) {
        unsigned int name_len = 0;
        unsigned int ndims = 0;
        if (!vm_model_read_u32(data, payload_size, &offset, &name_len) ||
            offset + name_len > payload_size) {
            free(data);
            vm_push(vm, NIL_VAL);
            return;
        }
        const char* name_ptr = (const char*)(data + offset);
        offset += name_len;
        if (!vm_model_read_u32(data, payload_size, &offset, &ndims) || ndims > VM_TENSOR_MAX_DIMS) {
            free(data);
            vm_push(vm, NIL_VAL);
            return;
        }

        uint64_t dims[VM_TENSOR_MAX_DIMS];
        for (unsigned int i = 0; i < ndims; i++) {
            if (!vm_model_read_u64(data, payload_size, &offset, &dims[i])) {
                free(data);
                vm_push(vm, NIL_VAL);
                return;
            }
        }
        unsigned char dtype = 0;
        if (!vm_model_read_u8(data, payload_size, &offset, &dtype) || dtype != 0) {
            free(data);
            vm_push(vm, NIL_VAL);
            return;
        }

        Value name_value;
        Value tensor_value;
        if (!vm_model_make_string_value(vm, name_ptr, (int)name_len, &name_value) ||
            !vm_model_make_tensor_value(vm, ndims, dims, data, payload_size, &offset, &tensor_value)) {
            free(data);
            vm_push(vm, NIL_VAL);
            return;
        }

        int32_t pair_ptr = heap_alloc(&vm->heap);
        int32_t node_ptr = heap_alloc(&vm->heap);
        if (pair_ptr < 0 || node_ptr < 0) {
            free(data);
            vm_push(vm, NIL_VAL);
            return;
        }
        vm->heap.objects[pair_ptr]->type = HEAP_CONS;
        vm->heap.objects[pair_ptr]->cons.car = name_value;
        vm->heap.objects[pair_ptr]->cons.cdr = tensor_value;
        vm->heap.objects[node_ptr]->type = HEAP_CONS;
        vm->heap.objects[node_ptr]->cons.car = PAIR_VAL(pair_ptr);
        vm->heap.objects[node_ptr]->cons.cdr = list;
        list = PAIR_VAL(node_ptr);
    }

    free(data);
    if (offset != payload_size) {
        vm_push(vm, NIL_VAL);
        return;
    }
    vm_push(vm, vm_model_reverse_list(vm, list));
}

#endif
