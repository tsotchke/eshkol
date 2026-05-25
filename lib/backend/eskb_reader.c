/**
 * @file eskb_reader.c
 * @brief Deserialize .eskb binary format into executable bytecode.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "eskb_format.h"
#include "eshkol/backend/vm_limits.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Loaded bytecode module */
typedef struct {
    char* name;
    int n_params;
    int n_locals;
    int n_upvalues;
    int code_offset;
    int code_len;
} EskbFunction;

typedef struct {
    /* Instructions */
    uint8_t* opcodes;
    int32_t* operands;
    int code_len;

    /* Constants */
    uint8_t* const_types;   /* ESKB_CONST_* for each constant */
    int64_t* const_ints;    /* int64 values (or 0 if not int) */
    double*  const_floats;  /* double values (or 0 if not float) */
    char**   const_strings; /* string values (or NULL) */
    int n_constants;

    EskbFunction* functions;
    int n_functions;

    /* Metadata */
    char source_file[256];
    int has_debug;
} EskbModule;

void eskb_module_free(EskbModule* m) {
    free(m->opcodes);
    free(m->operands);
    free(m->const_types);
    free(m->const_ints);
    free(m->const_floats);
    if (m->const_strings) {
        for (int i = 0; i < m->n_constants; i++) free(m->const_strings[i]);
        free(m->const_strings);
    }
    if (m->functions) {
        for (int i = 0; i < m->n_functions; i++) free(m->functions[i].name);
        free(m->functions);
    }
    memset(m, 0, sizeof(*m));
}

static int eskb_read_string_alloc(EskbReader* r, char** out, size_t* actual_len) {
    uint64_t slen64;
    if (!out) return -1;
    *out = NULL;
    if (eskb_read_leb(r, &slen64) < 0 || slen64 > SIZE_MAX - 1) return -1;

    size_t slen = (size_t)slen64;
    char* s = (char*)malloc(slen + 1);
    if (!s) return -1;
    if (eskb_read_bytes(r, s, slen) < 0) {
        free(s);
        return -1;
    }
    s[slen] = 0;
    *out = s;
    if (actual_len) *actual_len = slen;
    return 0;
}

static int eskb_read_instruction(EskbReader* r, uint8_t* op, int32_t* operand) {
    uint64_t zigzag;
    if (eskb_read_u8(r, op) < 0 || eskb_read_leb(r, &zigzag) < 0) return -1;
    if (zigzag > UINT32_MAX) return -1;

    int64_t decoded = (int64_t)(zigzag >> 1);
    if (zigzag & 1) decoded = -decoded - 1;
    if (decoded < INT32_MIN || decoded > INT32_MAX) return -1;

    *operand = (int32_t)decoded;
    return 0;
}

static int eskb_operand_is_code_target(uint8_t op) {
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

static int eskb_parse_payload(const uint8_t* payload, size_t payload_len, EskbModule* mod) {
    EskbReader r;
    eskb_reader_init(&r, payload, payload_len);

    uint64_t n_sections;
    if (eskb_read_leb(&r, &n_sections) < 0 || n_sections > 16) return -1;

    EskbSectionDesc sections[16];
    size_t section_offsets[16];

    for (uint64_t i = 0; i < n_sections; i++) {
        uint8_t sid;
        uint64_t ssize;
        if (eskb_read_u8(&r, &sid) < 0 || eskb_read_leb(&r, &ssize) < 0) return -1;
        if (ssize > UINT32_MAX) return -1;
        sections[i].id = sid;
        sections[i].size = (uint32_t)ssize;
    }

    size_t offset = r.pos;
    for (uint64_t i = 0; i < n_sections; i++) {
        if ((size_t)sections[i].size > payload_len - offset) return -1;
        section_offsets[i] = offset;
        offset += sections[i].size;
    }
    if (offset != payload_len) return -1;

    for (uint64_t s = 0; s < n_sections; s++) {
        EskbReader sr;
        eskb_reader_init(&sr, payload + section_offsets[s], sections[s].size);

        if (sections[s].id == ESKB_SECTION_CONST) {
            uint64_t nc;
            if (eskb_read_leb(&sr, &nc) < 0 || nc > INT_MAX) return -1;
            if (mod->const_types) return -1;

            mod->n_constants = (int)nc;
            size_t alloc_count = nc ? (size_t)nc : 1;
            mod->const_types = (uint8_t*)calloc(alloc_count, sizeof(uint8_t));
            mod->const_ints = (int64_t*)calloc(alloc_count, sizeof(int64_t));
            mod->const_floats = (double*)calloc(alloc_count, sizeof(double));
            mod->const_strings = (char**)calloc(alloc_count, sizeof(char*));
            if (!mod->const_types || !mod->const_ints || !mod->const_floats || !mod->const_strings) return -1;

            for (uint64_t i = 0; i < nc; i++) {
                uint8_t ctype;
                if (eskb_read_u8(&sr, &ctype) < 0) return -1;
                mod->const_types[i] = ctype;
                switch (ctype) {
                case ESKB_CONST_NIL:
                    break;
                case ESKB_CONST_INT64:
                    if (eskb_read_i64(&sr, &mod->const_ints[i]) < 0) return -1;
                    break;
                case ESKB_CONST_F64:
                    if (eskb_read_f64(&sr, &mod->const_floats[i]) < 0) return -1;
                    break;
                case ESKB_CONST_BOOL: {
                    uint8_t bv;
                    if (eskb_read_u8(&sr, &bv) < 0) return -1;
                    mod->const_ints[i] = bv;
                    break;
                }
                case ESKB_CONST_STRING: {
                    size_t slen;
                    if (eskb_read_string_alloc(&sr, &mod->const_strings[i], &slen) < 0) return -1;
                    mod->const_ints[i] = (int64_t)slen;
                    break;
                }
                default:
                    if (eskb_read_i64(&sr, &mod->const_ints[i]) < 0) return -1;
                    break;
                }
            }
            if (sr.pos != sr.len) return -1;
        } else if (sections[s].id == ESKB_SECTION_CODE) {
            uint64_t nf;
            if (eskb_read_leb(&sr, &nf) < 0 || nf > 4096 || nf > INT_MAX) return -1;
            if (mod->opcodes || mod->functions) return -1;

            mod->n_functions = (int)nf;
            mod->functions = (EskbFunction*)calloc(nf ? (size_t)nf : 1, sizeof(EskbFunction));
            if (!mod->functions) return -1;
            for (uint64_t fi = 0; fi < nf; fi++) {
                char* fname = NULL;
                size_t fname_len;
                if (eskb_read_string_alloc(&sr, &fname, &fname_len) < 0) return -1;
                if (fname_len == 0 || fname_len > 255) {
                    free(fname);
                    return -1;
                }
                for (uint64_t prev = 0; prev < fi; prev++) {
                    if (strcmp(mod->functions[prev].name, fname) == 0) {
                        free(fname);
                        return -1;
                    }
                }

                uint8_t np;
                uint64_t nl;
                uint8_t nuv;
                uint64_t cl;
                if (eskb_read_u8(&sr, &np) < 0) { free(fname); return -1; }
                if (eskb_read_leb(&sr, &nl) < 0 || nl > INT_MAX) { free(fname); return -1; }
                if (eskb_read_u8(&sr, &nuv) < 0) { free(fname); return -1; }
                if (eskb_read_leb(&sr, &cl) < 0 || cl > ESHKOL_VM_MAX_CODE || cl > INT_MAX) {
                    free(fname);
                    return -1;
                }
                if (mod->code_len > INT_MAX - (int)cl) {
                    free(fname);
                    return -1;
                }

                int code_offset = mod->code_len;
                int new_code_len = mod->code_len + (int)cl;
                if (new_code_len > ESHKOL_VM_MAX_CODE) {
                    free(fname);
                    return -1;
                }
                size_t alloc_count = new_code_len ? (size_t)new_code_len : 1;
                uint8_t* new_opcodes = (uint8_t*)realloc(mod->opcodes, alloc_count * sizeof(uint8_t));
                if (!new_opcodes) { free(fname); return -1; }
                mod->opcodes = new_opcodes;
                int32_t* new_operands = (int32_t*)realloc(mod->operands, alloc_count * sizeof(int32_t));
                if (!new_operands) { free(fname); return -1; }
                mod->operands = new_operands;

                mod->functions[fi].name = fname;
                mod->functions[fi].n_params = np;
                mod->functions[fi].n_locals = (int)nl;
                mod->functions[fi].n_upvalues = nuv;
                mod->functions[fi].code_offset = code_offset;
                mod->functions[fi].code_len = (int)cl;

                for (uint64_t i = 0; i < cl; i++) {
                    if (eskb_read_instruction(&sr,
                                              &mod->opcodes[code_offset + (int)i],
                                              &mod->operands[code_offset + (int)i]) < 0) {
                        return -1;
                    }
                    if (eskb_operand_is_code_target(mod->opcodes[code_offset + (int)i])) {
                        int32_t operand = mod->operands[code_offset + (int)i];
                        if (operand < 0 || operand >= (int32_t)cl) return -1;
                        mod->operands[code_offset + (int)i] = operand + code_offset;
                    }
                }
                mod->code_len = new_code_len;
            }
            if (sr.pos != sr.len) return -1;
        } else if (sections[s].id == ESKB_SECTION_META) {
            if (mod->has_debug) return -1;
            mod->has_debug = 1;
            if (eskb_read_string(&sr, mod->source_file,
                                 sizeof(mod->source_file), NULL) < 0) {
                return -1;
            }
            if (sr.pos != sr.len) return -1;
        }
    }

    return 0;
}

int eskb_load_memory(const void* data, size_t size, EskbModule* mod) {
    if (!data || !mod) return -1;
    memset(mod, 0, sizeof(*mod));
    if (size < sizeof(EskbHeader)) return -1;

    const uint8_t* file_data = (const uint8_t*)data;

    EskbHeader hdr;
    memcpy(&hdr, file_data, sizeof(hdr));
    if (hdr.magic != ESKB_MAGIC) return -1;
    if (hdr.version != ESKB_VERSION) return -1;

    const uint8_t* payload = file_data + sizeof(hdr);
    size_t payload_len = size - sizeof(hdr);
    uint32_t computed_crc = eskb_crc32(payload, payload_len);
    if (computed_crc != hdr.checksum) {
        eskb_module_free(mod);
        return -1;
    }

    if (eskb_parse_payload(payload, payload_len, mod) < 0) {
        eskb_module_free(mod);
        return -1;
    }

    return 0;
}

/* Load .eskb file, validate, return module */
int eskb_load_file(const char* path, EskbModule* mod) {
    if (!path || !mod) return -1;

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }

    /* Read entire file */
    fseek(f, 0, SEEK_END);
    long file_len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_len < (long)sizeof(EskbHeader)) { fclose(f); return -1; }

    uint8_t* file_data = (uint8_t*)malloc((size_t)file_len);
    if (!file_data) { fclose(f); return -1; }
    if (fread(file_data, 1, (size_t)file_len, f) != (size_t)file_len) {
        free(file_data); fclose(f); return -1;
    }
    fclose(f);

    int result = eskb_load_memory(file_data, (size_t)file_len, mod);
    if (result != 0) {
        EskbHeader hdr;
        if ((size_t)file_len >= sizeof(hdr)) {
            memcpy(&hdr, file_data, sizeof(hdr));
            if (hdr.magic != ESKB_MAGIC) {
                fprintf(stderr, "ERROR: bad magic 0x%08X (expected 0x%08X)\n", hdr.magic, ESKB_MAGIC);
            } else if (hdr.version != ESKB_VERSION) {
                fprintf(stderr, "ERROR: unsupported version %u (expected %u)\n", hdr.version, ESKB_VERSION);
            } else {
                const uint8_t* payload = file_data + sizeof(hdr);
                size_t payload_len = (size_t)file_len - sizeof(hdr);
                uint32_t computed_crc = eskb_crc32(payload, payload_len);
                if (computed_crc != hdr.checksum) {
                    fprintf(stderr, "ERROR: CRC32 mismatch (file=0x%08X computed=0x%08X)\n", hdr.checksum, computed_crc);
                } else {
                    fprintf(stderr, "ERROR: invalid ESKB payload\n");
                }
            }
        }
        free(file_data);
        return -1;
    }

    free(file_data);
    printf("[ESKB] Loaded %s: %d instructions, %d constants\n",
           path, mod->code_len, mod->n_constants);
    return 0;
}
