/**
 * @file kb_persistence.cpp
 * @brief Knowledge base save/load — binary serialization of facts.
 *
 * Format (ESKB v2):
 *   Header: magic(4) + version(4) + num_facts(4) + reserved(4)
 *   Per fact:
 *     predicate_name_len(4) + predicate_name(N) + arity(4)
 *     Per arg:
 *       type(1) + flags(1) + payload
 *         payload for IMMEDIATE types (NULL, INT64, DOUBLE, BOOL, CHAR, LOGIC_VAR):
 *           data(8)
 *         payload for HEAP_PTR (8):
 *           subtype(1) + per-subtype content:
 *             STRING(1) / SYMBOL(10) : len(4) + bytes(N)
 *             BIGNUM(11)             : str_len(4) + base-10 bytes(N)
 *           other subtypes: unsupported (save aborts with error, load aborts with error)
 *
 * v1 (pointer-valued) is REJECTED on load — the format was unsound because it
 * serialized runtime pointer values that do not survive a process restart.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <eshkol/eshkol.h>
#include <eshkol/core/logic.h>
#include <eshkol/core/bignum.h>
#include <eshkol/logger.h>
#include "../../lib/core/arena_memory.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

#define ESKB_MAGIC   0x45534B42 /* "ESKB" */
#define ESKB_VERSION 2

/* Allocate a SYMBOL-subtyped heap object (defined in arena_memory.cpp). */
extern "C" void* arena_allocate_symbol_with_header(arena_t* arena, size_t length);
/* Predicate interning (defined in logic.cpp). */
extern "C" const char* eshkol_intern_predicate(const char* name);

extern "C" {

/* ═══════════════════════════════════════════════════════════════════
 * Serialization helpers — one arg at a time.
 * Return true on success, false on error (then save/load should abort).
 * ═══════════════════════════════════════════════════════════════════ */

static bool write_u8(FILE* f, uint8_t v)   { return fwrite(&v, 1, 1, f) == 1; }
static bool write_u32(FILE* f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1; }
static bool write_u64(FILE* f, uint64_t v) { return fwrite(&v, 8, 1, f) == 1; }
static bool write_bytes(FILE* f, const void* p, size_t n) {
    return n == 0 || fwrite(p, 1, n, f) == n;
}

static bool read_u8(FILE* f, uint8_t* v)   { return fread(v, 1, 1, f) == 1; }
static bool read_u32(FILE* f, uint32_t* v) { return fread(v, 4, 1, f) == 1; }
static bool read_u64(FILE* f, uint64_t* v) { return fread(v, 8, 1, f) == 1; }
static bool read_bytes(FILE* f, void* p, size_t n) {
    return n == 0 || fread(p, 1, n, f) == n;
}

/* Write one tagged-value argument. Returns false on unsupported content. */
static bool write_arg(FILE* f, const eshkol_tagged_value_t* arg) {
    uint8_t type  = arg->type;
    uint8_t flags = arg->flags;
    if (!write_u8(f, type) || !write_u8(f, flags)) return false;

    switch (type) {
        case ESHKOL_VALUE_NULL:
        case ESHKOL_VALUE_INT64:
        case ESHKOL_VALUE_DOUBLE:
        case ESHKOL_VALUE_BOOL:
        case ESHKOL_VALUE_CHAR:
        case ESHKOL_VALUE_LOGIC_VAR:
            return write_u64(f, arg->data.raw_val);

        case ESHKOL_VALUE_HEAP_PTR: {
            void* ptr = (void*)(uintptr_t)arg->data.ptr_val;
            if (!ptr) {
                /* Null heap pointer encoded as subtype=0 with empty content */
                return write_u8(f, 0xFF) && write_u32(f, 0);
            }
            eshkol_object_header_t* hdr = ESHKOL_GET_HEADER(ptr);
            uint8_t subtype = hdr->subtype;
            if (!write_u8(f, subtype)) return false;

            switch (subtype) {
                case HEAP_SUBTYPE_STRING:
                case HEAP_SUBTYPE_SYMBOL: {
                    /* Data is NUL-terminated UTF-8 starting at ptr */
                    const char* s = (const char*)ptr;
                    size_t len = strlen(s);
                    if (len > 0x7FFFFFFF) {
                        eshkol_error("kb-save: string too large (%zu bytes)", len);
                        return false;
                    }
                    return write_u32(f, (uint32_t)len) &&
                           write_bytes(f, s, len);
                }
                case HEAP_SUBTYPE_BIGNUM: {
                    /* Emit base-10 representation */
                    char* str = eshkol_bignum_to_string(nullptr,
                        (const eshkol_bignum_t*)ptr);
                    if (!str) {
                        eshkol_error("kb-save: bignum serialization failed");
                        return false;
                    }
                    size_t len = strlen(str);
                    bool ok = write_u32(f, (uint32_t)len) &&
                              write_bytes(f, str, len);
                    free(str);
                    return ok;
                }
                default:
                    eshkol_error(
                        "kb-save: HEAP_PTR subtype %u not supported in persistence",
                        (unsigned)subtype);
                    return false;
            }
        }

        default:
            eshkol_error("kb-save: tagged type %u not supported in persistence",
                (unsigned)type);
            return false;
    }
}

/* Read one tagged-value argument. Returns false on I/O error or unsupported. */
static bool read_arg(arena_t* arena, FILE* f, eshkol_tagged_value_t* out) {
    memset(out, 0, sizeof(*out));

    uint8_t type, flags;
    if (!read_u8(f, &type) || !read_u8(f, &flags)) return false;
    out->type = type;
    out->flags = flags;

    switch (type) {
        case ESHKOL_VALUE_NULL:
        case ESHKOL_VALUE_INT64:
        case ESHKOL_VALUE_DOUBLE:
        case ESHKOL_VALUE_BOOL:
        case ESHKOL_VALUE_CHAR:
        case ESHKOL_VALUE_LOGIC_VAR:
            return read_u64(f, &out->data.raw_val);

        case ESHKOL_VALUE_HEAP_PTR: {
            uint8_t subtype;
            if (!read_u8(f, &subtype)) return false;

            if (subtype == 0xFF) {
                /* Null heap pointer sentinel */
                uint32_t zero;
                if (!read_u32(f, &zero)) return false;
                out->data.ptr_val = 0;
                return true;
            }

            uint32_t len;
            if (!read_u32(f, &len)) return false;

            /* #192 MEDIUM: cap attacker-supplied length before any
             * allocation. A 4GB string claim from a crafted file would
             * otherwise trigger a multi-GB arena allocation or (with
             * the +1 for the NUL) wrap to a tiny buffer on 32-bit
             * size_t platforms. 16 MB is more than enough for any
             * legitimate serialized KB string/symbol/bignum. */
            if (len > 16u * 1024u * 1024u) {
                eshkol_error("kb-load: string/symbol/bignum length %u exceeds 16 MB cap",
                             (unsigned)len);
                return false;
            }

            switch (subtype) {
                case HEAP_SUBTYPE_STRING: {
                    char* buf = arena_allocate_string_with_header(arena, len);
                    if (!buf) return false;
                    if (!read_bytes(f, buf, len)) return false;
                    buf[len] = '\0';
                    out->data.ptr_val = (uint64_t)(uintptr_t)buf;
                    return true;
                }
                case HEAP_SUBTYPE_SYMBOL: {
                    char* buf = (char*)arena_allocate_symbol_with_header(arena, len);
                    if (!buf) return false;
                    if (!read_bytes(f, buf, len)) return false;
                    buf[len] = '\0';
                    out->data.ptr_val = (uint64_t)(uintptr_t)buf;
                    return true;
                }
                case HEAP_SUBTYPE_BIGNUM: {
                    /* Read the base-10 string onto a stack/heap buffer, reconstruct */
                    char* tmp = (char*)malloc(len + 1);
                    if (!tmp) return false;
                    if (!read_bytes(f, tmp, len)) { free(tmp); return false; }
                    tmp[len] = '\0';
                    eshkol_bignum_t* bn = eshkol_bignum_from_string(arena, tmp, len);
                    free(tmp);
                    if (!bn) return false;
                    out->data.ptr_val = (uint64_t)(uintptr_t)bn;
                    return true;
                }
                default:
                    eshkol_error(
                        "kb-load: HEAP_PTR subtype %u not supported in persistence",
                        (unsigned)subtype);
                    return false;
            }
        }

        default:
            eshkol_error("kb-load: tagged type %u not supported in persistence",
                (unsigned)type);
            return false;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Public: kb-save and kb-load tagged-value entry points.
 * ═══════════════════════════════════════════════════════════════════ */

void eshkol_kb_save_tagged(arena_t* arena,
                           const eshkol_tagged_value_t* path_tv,
                           const eshkol_tagged_value_t* kb_tv,
                           eshkol_tagged_value_t* result) {
    (void)arena;

    auto fail = [&]() {
        if (!result) return;
        result->type = ESHKOL_VALUE_BOOL;
        result->flags = 0;
        result->reserved = 0;
        result->data.raw_val = 0; /* #f */
    };

    if (!result) return;
    memset(result, 0, sizeof(*result));

    /* Validate path: HEAP_PTR to a STRING */
    if (!path_tv || path_tv->type != ESHKOL_VALUE_HEAP_PTR || !path_tv->data.ptr_val) {
        eshkol_error("kb-save: first argument is not a string path");
        fail(); return;
    }
    void* path_ptr = (void*)(uintptr_t)path_tv->data.ptr_val;
    eshkol_object_header_t* path_hdr = ESHKOL_GET_HEADER(path_ptr);
    if (path_hdr->subtype != HEAP_SUBTYPE_STRING) {
        eshkol_error("kb-save: first argument is not a string path (subtype=%u)",
            (unsigned)path_hdr->subtype);
        fail(); return;
    }
    const char* path = (const char*)path_ptr;

    /* Validate KB: HEAP_PTR pointing at a KNOWLEDGE_BASE-subtyped object */
    if (!kb_tv || kb_tv->type != ESHKOL_VALUE_HEAP_PTR || !kb_tv->data.ptr_val) {
        eshkol_error("kb-save: second argument is not a knowledge base");
        fail(); return;
    }
    void* kb_ptr = (void*)(uintptr_t)kb_tv->data.ptr_val;
    eshkol_object_header_t* kb_hdr = ESHKOL_GET_HEADER(kb_ptr);
    if (kb_hdr->subtype != HEAP_SUBTYPE_KNOWLEDGE_BASE) {
        eshkol_error("kb-save: second argument is not a knowledge base (subtype=%u)",
            (unsigned)kb_hdr->subtype);
        fail(); return;
    }
    eshkol_knowledge_base_t* kb = (eshkol_knowledge_base_t*)kb_ptr;

    FILE* f = fopen(path, "wb");
    if (!f) { fail(); return; }

    /* Header */
    if (!write_u32(f, ESKB_MAGIC) ||
        !write_u32(f, ESKB_VERSION) ||
        !write_u32(f, kb->num_facts) ||
        !write_u32(f, 0 /* reserved */)) {
        fclose(f); fail(); return;
    }

    /* Facts */
    for (uint32_t i = 0; i < kb->num_facts; i++) {
        eshkol_fact_t* fact = kb->facts[i];
        if (!fact) { fclose(f); fail(); return; }

        const char* pred_name = (const char*)(uintptr_t)fact->predicate;
        uint32_t name_len = pred_name ? (uint32_t)strlen(pred_name) : 0;
        if (!write_u32(f, name_len) ||
            !write_bytes(f, pred_name, name_len) ||
            !write_u32(f, fact->arity)) {
            fclose(f); fail(); return;
        }

        eshkol_tagged_value_t* args = FACT_ARGS(fact);
        for (uint32_t j = 0; j < fact->arity; j++) {
            if (!write_arg(f, &args[j])) {
                fclose(f); fail(); return;
            }
        }
    }

    fclose(f);

    /* #t */
    result->type = ESHKOL_VALUE_BOOL;
    result->flags = 0;
    result->reserved = 0;
    result->data.raw_val = 1;
}

void eshkol_kb_load_tagged(arena_t* arena,
                           const eshkol_tagged_value_t* path_tv,
                           eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    result->type = ESHKOL_VALUE_NULL;

    if (!arena || !path_tv || path_tv->type != ESHKOL_VALUE_HEAP_PTR ||
        !path_tv->data.ptr_val) {
        eshkol_error("kb-load: first argument is not a string path");
        return;
    }
    void* path_ptr = (void*)(uintptr_t)path_tv->data.ptr_val;
    eshkol_object_header_t* path_hdr = ESHKOL_GET_HEADER(path_ptr);
    if (path_hdr->subtype != HEAP_SUBTYPE_STRING) {
        eshkol_error("kb-load: first argument is not a string path (subtype=%u)",
            (unsigned)path_hdr->subtype);
        return;
    }
    const char* path = (const char*)path_ptr;

    FILE* f = fopen(path, "rb");
    if (!f) {
        /* #194: don't swallow. kb-load returning () (set at line 315)
         * looked identical to "file exists but is empty"; log the
         * errno reason so operators can distinguish "no such file"
         * from permission denied. */
        eshkol_error("kb-load: cannot open '%s' (errno=%d)", path, errno);
        return;
    }

    uint32_t magic, version, num_facts, reserved;
    if (!read_u32(f, &magic) || magic != ESKB_MAGIC) {
        eshkol_error("kb-load: '%s' is not an ESKB file (bad magic)", path);
        fclose(f); return;
    }
    if (!read_u32(f, &version) ||
        !read_u32(f, &num_facts) ||
        !read_u32(f, &reserved)) {
        eshkol_error("kb-load: '%s' truncated in header", path);
        fclose(f); return;
    }

    if (version != ESKB_VERSION) {
        eshkol_error(
            "kb-load: unsupported ESKB version %u (expected %u). "
            "v1 files are unsound and cannot be loaded — regenerate with kb-save.",
            (unsigned)version, (unsigned)ESKB_VERSION);
        fclose(f); return;
    }

    eshkol_knowledge_base_t* kb = eshkol_make_kb(arena);
    if (!kb) {
        eshkol_error("kb-load: failed to allocate knowledge base");
        fclose(f); return;
    }

    char name_buf[256];
    for (uint32_t i = 0; i < num_facts; i++) {
        uint32_t name_len;
        if (!read_u32(f, &name_len)) {
            eshkol_error("kb-load: truncated at fact %u/%u (name length)",
                         (unsigned)i, (unsigned)num_facts);
            break;
        }
        if (name_len > 255) {
            eshkol_error("kb-load: predicate name too long (%u bytes)",
                (unsigned)name_len);
            fclose(f); return;
        }
        if (!read_bytes(f, name_buf, name_len)) {
            eshkol_error("kb-load: truncated at fact %u/%u (predicate name)",
                         (unsigned)i, (unsigned)num_facts);
            break;
        }
        name_buf[name_len] = '\0';

        uint32_t arity;
        if (!read_u32(f, &arity)) {
            eshkol_error("kb-load: truncated at fact %u/%u (arity)",
                         (unsigned)i, (unsigned)num_facts);
            break;
        }

        /* #192 CRITICAL: arity comes straight out of an untrusted file.
         * The multiply (size_t)arity * sizeof(tagged_value_t) wraps for
         * arity near UINT32_MAX, producing a tiny fact_size that then
         * gets a tiny arena allocation — and the subsequent j<arity
         * loop walks off the end of the buffer. Reject pathological
         * arities up front. 4096 is a generous practical cap
         * (predicates never have this many args); the multiply bound
         * also catches the wrap. */
        if (arity > 4096) {
            eshkol_error("kb-load: predicate arity=%u exceeds limit", arity);
            fclose(f); return;
        }
        if ((size_t)arity > (SIZE_MAX - sizeof(eshkol_fact_t)) / sizeof(eshkol_tagged_value_t)) {
            eshkol_error("kb-load: arity=%u would overflow fact_size", arity);
            fclose(f); return;
        }

        size_t fact_size = sizeof(eshkol_fact_t) +
            (size_t)arity * sizeof(eshkol_tagged_value_t);
        eshkol_fact_t* fact = (eshkol_fact_t*)arena_allocate_with_header(
            arena, fact_size, HEAP_SUBTYPE_FACT, 0);
        if (!fact) {
            eshkol_error("kb-load: out of memory allocating fact %u/%u",
                         (unsigned)i, (unsigned)num_facts);
            break;
        }

        /* Intern predicate so pointer equality works with runtime make-fact. */
        const char* interned = eshkol_intern_predicate(name_buf);
        fact->predicate = (uint64_t)(uintptr_t)interned;
        fact->arity = arity;
        fact->_pad = 0;

        eshkol_tagged_value_t* args = FACT_ARGS(fact);
        bool ok = true;
        for (uint32_t j = 0; j < arity; j++) {
            if (!read_arg(arena, f, &args[j])) { ok = false; break; }
        }
        if (!ok) {
            eshkol_error("kb-load: truncated at fact %u/%u (args)",
                         (unsigned)i, (unsigned)num_facts);
            fclose(f); return;
        }

        eshkol_kb_assert(arena, kb, fact);
    }

    fclose(f);

    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->flags = 0;
    result->reserved = 0;
    result->data.ptr_val = (uint64_t)kb;
}

} /* extern "C" */
