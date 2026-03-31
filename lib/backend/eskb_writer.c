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

/* Write a complete .eskb file */
int eskb_write_file(const char* path,
                    const EskbInstr* code, int code_len,
                    const EskbConst* constants, int n_constants,
                    const char* source_file) {
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

    /* Section 1: CODE (single function for now) */
    EskbBuffer code_buf;
    eskb_buf_init(&code_buf);
    eskb_buf_write_leb128(&code_buf, 1); /* n_functions = 1 */
    /* Function 0: main */
    eskb_buf_write_string(&code_buf, "main", 4);
    eskb_buf_write_u8(&code_buf, 0);  /* n_params */
    eskb_buf_write_leb128(&code_buf, 0);  /* n_locals (filled at load) */
    eskb_buf_write_u8(&code_buf, 0);  /* n_upvalues */
    eskb_buf_write_leb128(&code_buf, code_len);  /* code_len */
    for (int i = 0; i < code_len; i++) {
        eskb_buf_write_u8(&code_buf, code[i].op);
        /* Operand as signed LEB128 — store as zigzag-encoded unsigned */
        uint64_t zigzag = (uint64_t)((int64_t)code[i].operand << 1) ^
                          (uint64_t)((int64_t)code[i].operand >> 63);
        eskb_buf_write_leb128(&code_buf, zigzag);
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
    printf("[ESKB] Wrote %zu bytes to %s (%d instructions, %d constants)\n",
           total, path, code_len, n_constants);

    eskb_buf_free(&const_buf);
    eskb_buf_free(&code_buf);
    eskb_buf_free(&meta_buf);
    eskb_buf_free(&payload);
    return 0;
}
