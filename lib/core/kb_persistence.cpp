/**
 * @file kb_persistence.cpp
 * @brief Knowledge base save/load — binary serialization of facts.
 *
 * Format (ESKB v1):
 *   Header: magic(4) + version(4) + num_facts(4) + reserved(4)
 *   Per fact:
 *     predicate_name_len(4) + predicate_name(N) + arity(4)
 *     Per arg:
 *       type(1) + data(8)
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <eshkol/eshkol.h>
#include <eshkol/core/logic.h>
#include "../../lib/core/arena_memory.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

#define ESKB_MAGIC   0x45534B42 /* "ESKB" */
#define ESKB_VERSION 1

extern "C" {

void eshkol_kb_save_tagged(arena_t* arena,
                           const eshkol_tagged_value_t* path_tv,
                           const eshkol_tagged_value_t* kb_tv,
                           eshkol_tagged_value_t* result) {
    (void)arena;

    /* Extract path string */
    if (path_tv->type != ESHKOL_VALUE_HEAP_PTR) {
        *result = eshkol_make_int64(0, true);
        return;
    }
    const char* path = (const char*)(uintptr_t)path_tv->data.ptr_val;
    if (!path) { *result = eshkol_make_int64(0, true); return; }

    /* Extract KB pointer */
    if (kb_tv->type != ESHKOL_VALUE_HEAP_PTR) {
        *result = eshkol_make_int64(0, true);
        return;
    }
    eshkol_knowledge_base_t* kb = (eshkol_knowledge_base_t*)(uintptr_t)kb_tv->data.ptr_val;
    if (!kb) { *result = eshkol_make_int64(0, true); return; }

    FILE* f = fopen(path, "wb");
    if (!f) { *result = eshkol_make_int64(0, true); return; }

    /* Write header */
    uint32_t magic = ESKB_MAGIC;
    uint32_t version = ESKB_VERSION;
    uint32_t num_facts = kb->num_facts;
    uint32_t reserved = 0;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&num_facts, 4, 1, f);
    fwrite(&reserved, 4, 1, f);

    /* Write each fact */
    for (uint32_t i = 0; i < num_facts; i++) {
        eshkol_fact_t* fact = kb->facts[i];
        if (!fact) continue;

        /* Get predicate name from the interned symbol */
        const char* pred_name = (const char*)(uintptr_t)fact->predicate;
        uint32_t name_len = pred_name ? (uint32_t)strlen(pred_name) : 0;
        fwrite(&name_len, 4, 1, f);
        if (name_len > 0) fwrite(pred_name, 1, name_len, f);

        uint32_t arity = fact->arity;
        fwrite(&arity, 4, 1, f);

        /* Write each argument as type(1) + data(8) */
        eshkol_tagged_value_t* args = FACT_ARGS(fact);
        for (uint32_t j = 0; j < arity; j++) {
            uint8_t type = args[j].type;
            uint64_t data = args[j].data.raw_val;
            fwrite(&type, 1, 1, f);
            fwrite(&data, 8, 1, f);
        }
    }

    fclose(f);

    /* Return #t */
    result->type = ESHKOL_VALUE_BOOL;
    result->flags = 0;
    result->reserved = 0;
    result->data.raw_val = 1;
}

void eshkol_kb_load_tagged(arena_t* arena,
                           const eshkol_tagged_value_t* path_tv,
                           eshkol_tagged_value_t* result) {
    /* Extract path string */
    if (path_tv->type != ESHKOL_VALUE_HEAP_PTR) {
        result->type = ESHKOL_VALUE_NULL;
        result->data.raw_val = 0;
        return;
    }
    const char* path = (const char*)(uintptr_t)path_tv->data.ptr_val;
    if (!path) { result->type = ESHKOL_VALUE_NULL; return; }

    FILE* f = fopen(path, "rb");
    if (!f) { result->type = ESHKOL_VALUE_NULL; return; }

    /* Read and verify header */
    uint32_t magic, version, num_facts, reserved;
    if (fread(&magic, 4, 1, f) != 1 || magic != ESKB_MAGIC ||
        fread(&version, 4, 1, f) != 1 || version != ESKB_VERSION ||
        fread(&num_facts, 4, 1, f) != 1 ||
        fread(&reserved, 4, 1, f) != 1) {
        fclose(f);
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Create new KB */
    eshkol_knowledge_base_t* kb = eshkol_make_kb(arena);
    if (!kb) { fclose(f); result->type = ESHKOL_VALUE_NULL; return; }

    /* Read each fact */
    for (uint32_t i = 0; i < num_facts; i++) {
        uint32_t name_len;
        if (fread(&name_len, 4, 1, f) != 1) break;

        /* Read predicate name */
        char name_buf[256];
        if (name_len > 255) name_len = 255;
        if (name_len > 0 && fread(name_buf, 1, name_len, f) != name_len) break;
        name_buf[name_len] = '\0';

        uint32_t arity;
        if (fread(&arity, 4, 1, f) != 1) break;

        /* Allocate fact with args */
        size_t fact_size = sizeof(eshkol_fact_t) + arity * sizeof(eshkol_tagged_value_t);
        eshkol_fact_t* fact = (eshkol_fact_t*)arena_allocate_with_header(arena, fact_size,
            HEAP_SUBTYPE_FACT, 0);
        if (!fact) break;

        /* Set predicate — allocate string in arena */
        char* pred_str = arena_allocate_string_with_header(arena, name_len);
        if (pred_str) {
            memcpy(pred_str, name_buf, name_len + 1);
            fact->predicate = (uint64_t)pred_str;
        } else {
            fact->predicate = 0;
        }
        fact->arity = arity;
        fact->_pad = 0;

        /* Read args */
        eshkol_tagged_value_t* args = FACT_ARGS(fact);
        for (uint32_t j = 0; j < arity; j++) {
            uint8_t type;
            uint64_t data;
            if (fread(&type, 1, 1, f) != 1 || fread(&data, 8, 1, f) != 1) break;
            args[j].type = type;
            args[j].flags = 0;
            args[j].reserved = 0;
            args[j].data.raw_val = data;
        }

        /* Add to KB */
        eshkol_kb_assert(arena, kb, fact);
    }

    fclose(f);

    /* Return KB as HEAP_PTR */
    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->flags = 0;
    result->reserved = 0;
    result->data.ptr_val = (uint64_t)kb;
}

} /* extern "C" */
