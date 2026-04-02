/**
 * @file eskb_reader.c
 * @brief Deserialize .eskb binary format into executable bytecode.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "eskb_format.h"
#include <stdio.h>
#include <stdlib.h>

/* Loaded bytecode module */
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

    /* Metadata */
    char source_file[256];
    int has_debug;
} EskbModule;

static void eskb_module_free(EskbModule* m) {
    free(m->opcodes);
    free(m->operands);
    free(m->const_types);
    free(m->const_ints);
    free(m->const_floats);
    if (m->const_strings) {
        for (int i = 0; i < m->n_constants; i++) free(m->const_strings[i]);
        free(m->const_strings);
    }
    memset(m, 0, sizeof(*m));
}

/* Load .eskb file, validate, return module */
static int eskb_load_file(const char* path, EskbModule* mod) {
    memset(mod, 0, sizeof(*mod));

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }

    /* Read entire file */
    fseek(f, 0, SEEK_END);
    long file_len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_len < (long)sizeof(EskbHeader)) { fclose(f); return -1; }

    uint8_t* file_data = (uint8_t*)malloc(file_len);
    if (!file_data) { fclose(f); return -1; }
    if (fread(file_data, 1, file_len, f) != (size_t)file_len) {
        free(file_data); fclose(f); return -1;
    }
    fclose(f);

    /* Parse header */
    EskbHeader hdr;
    memcpy(&hdr, file_data, sizeof(hdr));
    if (hdr.magic != ESKB_MAGIC) {
        fprintf(stderr, "ERROR: bad magic 0x%08X (expected 0x%08X)\n", hdr.magic, ESKB_MAGIC);
        free(file_data); return -1;
    }
    if (hdr.version != ESKB_VERSION) {
        fprintf(stderr, "ERROR: unsupported version %u (expected %u)\n", hdr.version, ESKB_VERSION);
        free(file_data); return -1;
    }

    /* Validate CRC32 */
    const uint8_t* payload = file_data + sizeof(hdr);
    size_t payload_len = file_len - sizeof(hdr);
    uint32_t computed_crc = eskb_crc32(payload, payload_len);
    if (computed_crc != hdr.checksum) {
        fprintf(stderr, "ERROR: CRC32 mismatch (file=0x%08X computed=0x%08X)\n", hdr.checksum, computed_crc);
        free(file_data); return -1;
    }

    /* Parse section table */
    EskbReader r;
    eskb_reader_init(&r, payload, payload_len);

    uint64_t n_sections;
    if (eskb_read_leb(&r, &n_sections) < 0 || n_sections > 16) {
        fprintf(stderr, "ERROR: invalid section count\n");
        free(file_data); return -1;
    }

    EskbSectionDesc sections[16];
    size_t section_offsets[16];

    for (uint64_t i = 0; i < n_sections; i++) {
        uint8_t sid;
        uint64_t ssize;
        if (eskb_read_u8(&r, &sid) < 0 || eskb_read_leb(&r, &ssize) < 0) {
            fprintf(stderr, "ERROR: truncated section table\n");
            free(file_data); return -1;
        }
        sections[i].id = sid;
        sections[i].size = (uint32_t)ssize;
    }

    /* Compute section data offsets */
    size_t data_start = r.pos;
    size_t offset = data_start;
    for (uint64_t i = 0; i < n_sections; i++) {
        section_offsets[i] = offset;
        offset += sections[i].size;
    }

    /* Parse each section */
    for (uint64_t s = 0; s < n_sections; s++) {
        EskbReader sr;
        eskb_reader_init(&sr, payload + section_offsets[s], sections[s].size);

        if (sections[s].id == ESKB_SECTION_CONST) {
            uint64_t nc;
            if (eskb_read_leb(&sr, &nc) < 0) { free(file_data); return -1; }
            mod->n_constants = (int)nc;
            mod->const_types = (uint8_t*)calloc(nc, sizeof(uint8_t));
            mod->const_ints = (int64_t*)calloc(nc, sizeof(int64_t));
            mod->const_floats = (double*)calloc(nc, sizeof(double));
            mod->const_strings = (char**)calloc(nc, sizeof(char*));
            if (!mod->const_types || !mod->const_ints || !mod->const_floats || !mod->const_strings) {
                free(file_data); eskb_module_free(mod); return -1;
            }
            for (uint64_t i = 0; i < nc; i++) {
                uint8_t ctype;
                if (eskb_read_u8(&sr, &ctype) < 0) { free(file_data); eskb_module_free(mod); return -1; }
                mod->const_types[i] = ctype;
                switch (ctype) {
                case ESKB_CONST_NIL: break;
                case ESKB_CONST_INT64:
                    if (eskb_read_i64(&sr, &mod->const_ints[i]) < 0) { free(file_data); eskb_module_free(mod); return -1; }
                    break;
                case ESKB_CONST_F64:
                    if (eskb_read_f64(&sr, &mod->const_floats[i]) < 0) { free(file_data); eskb_module_free(mod); return -1; }
                    break;
                case ESKB_CONST_BOOL: {
                    uint8_t bv;
                    if (eskb_read_u8(&sr, &bv) < 0) { free(file_data); eskb_module_free(mod); return -1; }
                    mod->const_ints[i] = bv;
                    break;
                }
                case ESKB_CONST_STRING: {
                    char tmp[256];
                    size_t slen;
                    if (eskb_read_string(&sr, tmp, 256, &slen) < 0) { free(file_data); eskb_module_free(mod); return -1; }
                    mod->const_strings[i] = (char*)malloc(slen + 1);
                    if (mod->const_strings[i]) { memcpy(mod->const_strings[i], tmp, slen + 1); }
                    mod->const_ints[i] = (int64_t)slen;
                    break;
                }
                default:
                    if (eskb_read_i64(&sr, &mod->const_ints[i]) < 0) { free(file_data); eskb_module_free(mod); return -1; }
                    break;
                }
            }
        }
        else if (sections[s].id == ESKB_SECTION_CODE) {
            uint64_t nf;
            if (eskb_read_leb(&sr, &nf) < 0) { free(file_data); eskb_module_free(mod); return -1; }
            /* Read first function (main) */
            for (uint64_t fi = 0; fi < nf; fi++) {
                char fname[128];
                size_t fname_len;
                if (eskb_read_string(&sr, fname, 128, &fname_len) < 0) { free(file_data); eskb_module_free(mod); return -1; }
                uint8_t np; uint64_t nl; uint8_t nuv; uint64_t cl;
                if (eskb_read_u8(&sr, &np) < 0) { free(file_data); eskb_module_free(mod); return -1; }
                if (eskb_read_leb(&sr, &nl) < 0) { free(file_data); eskb_module_free(mod); return -1; }
                if (eskb_read_u8(&sr, &nuv) < 0) { free(file_data); eskb_module_free(mod); return -1; }
                if (eskb_read_leb(&sr, &cl) < 0 || cl > 100000) { free(file_data); eskb_module_free(mod); return -1; }

                if (fi == 0) { /* Only load first function as main */
                    mod->code_len = (int)cl;
                    mod->opcodes = (uint8_t*)calloc(cl, sizeof(uint8_t));
                    mod->operands = (int32_t*)calloc(cl, sizeof(int32_t));
                    if (!mod->opcodes || !mod->operands) { free(file_data); eskb_module_free(mod); return -1; }
                    for (uint64_t i = 0; i < cl; i++) {
                        if (eskb_read_u8(&sr, &mod->opcodes[i]) < 0) { free(file_data); eskb_module_free(mod); return -1; }
                        uint64_t zigzag;
                        if (eskb_read_leb(&sr, &zigzag) < 0) { free(file_data); eskb_module_free(mod); return -1; }
                        /* Decode zigzag to signed int32 */
                        mod->operands[i] = (int32_t)((int64_t)(zigzag >> 1) ^ -(int64_t)(zigzag & 1));
                    }
                }
            }
        }
        else if (sections[s].id == ESKB_SECTION_META) {
            mod->has_debug = 1;
            eskb_read_string(&sr, mod->source_file, 256, NULL);
        }
    }

    free(file_data);
    printf("[ESKB] Loaded %s: %d instructions, %d constants\n",
           path, mod->code_len, mod->n_constants);
    return 0;
}
