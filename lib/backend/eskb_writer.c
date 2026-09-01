/**
 * @file eskb_writer.c
 * @brief Serialize compiled bytecode to .eskb binary format.
 *
 * Can be used as:
 *   1. Library: #include and call eskb_write_file()
 *   2. Standalone: linked with eshkol_compiler.c
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "eskb_format.h"
#include <stdio.h>
#include <stdlib.h>

/* Write a compiled function chunk to .eskb file.
 * Takes the same Instr/Value types used in eshkol_compiler.c and eshkol_vm.c */

typedef struct { uint8_t op; int32_t operand; } EskbInstr;
typedef struct {
    uint8_t type;  /* ESKB_CONST_* */
    union { int64_t i; double f; int b; } as;
    char str[256]; /* for string constants */
    int str_len;
} EskbConst;

typedef struct {
    const char* name;
    uint8_t n_params;
    uint32_t n_locals;
    uint8_t n_upvalues;
    const EskbInstr* code;
    int code_len;
    int code_base;
} EskbFunctionDef;

static int eskb_writer_operand_is_code_target(uint8_t op) {
    enum {
        ESKB_OP_JUMP = 28,
        ESKB_OP_JUMP_IF_FALSE = 29,
        ESKB_OP_LOOP = 30,
        ESKB_OP_PUSH_HANDLER = 57,
    };
    return op == ESKB_OP_JUMP ||
           op == ESKB_OP_JUMP_IF_FALSE ||
           op == ESKB_OP_LOOP ||
           op == ESKB_OP_PUSH_HANDLER;
}

static int eskb_write_instruction(EskbBuffer* b, uint8_t op, int32_t operand) {
    if (eskb_buf_write_u8(b, op) < 0) return -1;
    uint64_t zigzag = (uint64_t)((int64_t)operand << 1) ^
                      (uint64_t)((int64_t)operand >> 63);
    return eskb_buf_write_leb128(b, zigzag);
}

static int eskb_write_function(EskbBuffer* code_buf,
                               const EskbFunctionDef* fn) {
    if (!code_buf || !fn || !fn->name || !fn->name[0] ||
        !fn->code || fn->code_len <= 0) {
        return -1;
    }

    size_t name_len = strlen(fn->name);
    if (name_len > 255) return -1;
    if (eskb_buf_write_string(code_buf, fn->name, name_len) < 0) return -1;
    if (eskb_buf_write_u8(code_buf, fn->n_params) < 0) return -1;
    if (eskb_buf_write_leb128(code_buf, fn->n_locals) < 0) return -1;
    if (eskb_buf_write_u8(code_buf, fn->n_upvalues) < 0) return -1;
    if (eskb_buf_write_leb128(code_buf, (uint64_t)fn->code_len) < 0) return -1;

    for (int i = 0; i < fn->code_len; i++) {
        int32_t operand = fn->code[i].operand;
        if (fn->code_base != 0 && eskb_writer_operand_is_code_target(fn->code[i].op)) {
            int32_t lo = fn->code_base;
            int32_t hi = fn->code_base + fn->code_len;
            if (operand < lo || operand >= hi) return -1;
            operand -= fn->code_base;
        }
        if (eskb_write_instruction(code_buf, fn->code[i].op, operand) < 0) return -1;
    }

    return 0;
}

/* Write a complete .eskb file */
int eskb_write_file_with_functions(const char* path,
                                   const EskbConst* constants,
                                   int n_constants,
                                   const EskbFunctionDef* functions,
                                   int n_functions,
                                   const char* source_file) {
    if (!path || !constants || n_constants < 0 || !functions || n_functions <= 0) {
        return -1;
    }

    /* Build sections into buffers */

    /* Section 0: CONST */
    EskbBuffer const_buf;
    eskb_buf_init(&const_buf);
    eskb_buf_write_leb128(&const_buf, n_constants);
    for (int i = 0; i < n_constants; i++) {
        eskb_buf_write_u8(&const_buf, constants[i].type);
        switch (constants[i].type) {
        case ESKB_CONST_NIL:
            break; /* no data */
        case ESKB_CONST_INT64:
            eskb_buf_write_i64(&const_buf, constants[i].as.i);
            break;
        case ESKB_CONST_F64:
            eskb_buf_write_f64(&const_buf, constants[i].as.f);
            break;
        case ESKB_CONST_BOOL:
            eskb_buf_write_u8(&const_buf, constants[i].as.b ? 1 : 0);
            break;
        case ESKB_CONST_STRING:
            eskb_buf_write_string(&const_buf, constants[i].str, constants[i].str_len);
            break;
        default:
            eskb_buf_write_i64(&const_buf, constants[i].as.i);
            break;
        }
    }

    /* Section 1: CODE */
    EskbBuffer code_buf;
    eskb_buf_init(&code_buf);
    if (eskb_buf_write_leb128(&code_buf, (uint64_t)n_functions) < 0) {
        eskb_buf_free(&const_buf);
        eskb_buf_free(&code_buf);
        return -1;
    }
    for (int i = 0; i < n_functions; i++) {
        if (eskb_write_function(&code_buf, &functions[i]) < 0) {
            eskb_buf_free(&const_buf);
            eskb_buf_free(&code_buf);
            return -1;
        }
    }

    /* Section 2: META (optional debug info) */
    EskbBuffer meta_buf;
    eskb_buf_init(&meta_buf);
    if (source_file) {
        size_t slen = strlen(source_file);
        eskb_buf_write_string(&meta_buf, source_file, slen);
    }

    /* Build section table */
    int n_sections = source_file ? 3 : 2;

    /* Compute payload: section table + section data */
    EskbBuffer payload;
    eskb_buf_init(&payload);
    eskb_buf_write_leb128(&payload, n_sections);

    /* Section descriptors */
    eskb_buf_write_u8(&payload, ESKB_SECTION_CONST);
    eskb_buf_write_leb128(&payload, const_buf.len);
    eskb_buf_write_u8(&payload, ESKB_SECTION_CODE);
    eskb_buf_write_leb128(&payload, code_buf.len);
    if (source_file) {
        eskb_buf_write_u8(&payload, ESKB_SECTION_META);
        eskb_buf_write_leb128(&payload, meta_buf.len);
    }

    /* Section data */
    eskb_buf_write(&payload, const_buf.data, const_buf.len);
    eskb_buf_write(&payload, code_buf.data, code_buf.len);
    if (source_file) {
        eskb_buf_write(&payload, meta_buf.data, meta_buf.len);
    }

    /* Build header */
    EskbHeader hdr;
    hdr.magic = ESKB_MAGIC;
    hdr.version = ESKB_VERSION;
    hdr.flags = ESKB_FLAG_LITTLE_ENDIAN;
    hdr.checksum = eskb_crc32(payload.data, payload.len);

    /* Write to file */
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s for writing\n", path);
        eskb_buf_free(&const_buf);
        eskb_buf_free(&code_buf);
        eskb_buf_free(&meta_buf);
        eskb_buf_free(&payload);
        return -1;
    }
    fwrite(&hdr, sizeof(hdr), 1, f);
    fwrite(payload.data, 1, payload.len, f);
    fclose(f);

    size_t total = sizeof(hdr) + payload.len;
    int total_code_len = 0;
    for (int i = 0; i < n_functions; i++) total_code_len += functions[i].code_len;
    printf("[ESKB] Wrote %zu bytes to %s (%d functions, %d instructions, %d constants)\n",
           total, path, n_functions, total_code_len, n_constants);

    eskb_buf_free(&const_buf);
    eskb_buf_free(&code_buf);
    eskb_buf_free(&meta_buf);
    eskb_buf_free(&payload);
    return 0;
}

int eskb_write_file(const char* path,
                    const EskbInstr* code, int code_len,
                    const EskbConst* constants, int n_constants,
                    const char* source_file) {
    EskbFunctionDef main_function;
    main_function.name = "main";
    main_function.n_params = 0;
    main_function.n_locals = 0;
    main_function.n_upvalues = 0;
    main_function.code = code;
    main_function.code_len = code_len;
    main_function.code_base = 0;
    return eskb_write_file_with_functions(path, constants, n_constants,
                                          &main_function, 1, source_file);
}
