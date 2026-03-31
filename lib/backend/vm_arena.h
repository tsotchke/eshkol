/**
 * @file vm_arena.h
 * @brief OALR-aligned arena memory for bytecode VMs.
 *
 * Implements Ownership-Aware Lexical Regions for the bytecode VM:
 *   - Arena: linked-list of 8KB blocks, bump-pointer O(1) allocation
 *   - Region: lexical scope with its own arena, auto-freed on pop
 *   - Region stack: max 64 deep (matching Eshkol's arena_memory.h)
 *
 * No garbage collection. Deterministic cleanup at region exit.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef VM_ARENA_H
#define VM_ARENA_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Configuration ── */
#define VM_ARENA_DEFAULT_BLOCK_SIZE  8192  /* 8KB blocks (matches Eshkol's arena_memory.h) */
#define VM_ARENA_MAX_REGIONS         64    /* Max region nesting depth */
#define VM_ARENA_ALIGNMENT           8     /* 8-byte alignment */

/* ── Object Header (8 bytes, matches Eshkol's eshkol_object_header) ── */
typedef struct {
    uint8_t  subtype;     /* Type identifier (0=cons, 1=closure, 2=string, 3=vector, etc.) */
    uint8_t  flags;       /* LINEAR, BORROWED, SHARED, etc. */
    uint16_t ref_count;   /* Reference count (0 = not ref-counted) */
    uint32_t size;        /* Object size in bytes (excluding header) */
} VmObjectHeader;

/* Subtypes matching Eshkol's heap subtypes */
#define VM_SUBTYPE_CONS     0
#define VM_SUBTYPE_CLOSURE  1
#define VM_SUBTYPE_STRING   2
#define VM_SUBTYPE_VECTOR   3
#define VM_SUBTYPE_CONT     4

/* Flags */
#define VM_FLAG_LINEAR   0x01
#define VM_FLAG_BORROWED 0x02
#define VM_FLAG_SHARED   0x04

/* ── Arena Block ── */
typedef struct VmArenaBlock {
    uint8_t* data;
    size_t   size;
    size_t   used;
    struct VmArenaBlock* next;  /* next block in chain */
} VmArenaBlock;

/* ── Arena ── */
typedef struct {
    VmArenaBlock* current;
    size_t        default_block_size;
    size_t        total_allocated;
    size_t        total_used;
    int           n_blocks;
} VmArena;

/* ── Region ── */
typedef struct VmRegion {
    VmArena          arena;
    struct VmRegion* parent;
    size_t           escape_count;
    const char*      name;  /* optional, for debugging */
} VmRegion;

/* ── Region Stack (global, matches thread_local pattern from Eshkol) ── */
typedef struct {
    VmRegion* stack[VM_ARENA_MAX_REGIONS];
    int       depth;
    VmArena   global_arena;  /* fallback when no region is active */

    /* Metrics */
    size_t    peak_allocated;
    int       peak_depth;
    int       total_regions_pushed;
} VmRegionStack;

/*******************************************************************************
 * Arena Operations
 ******************************************************************************/

static inline VmArenaBlock* vm_arena_block_create(size_t size) {
    VmArenaBlock* b = (VmArenaBlock*)malloc(sizeof(VmArenaBlock));
    if (!b) return NULL;
    b->data = (uint8_t*)malloc(size);
    if (!b->data) { free(b); return NULL; }
    b->size = size;
    b->used = 0;
    b->next = NULL;
    return b;
}

static inline void vm_arena_block_destroy(VmArenaBlock* b) {
    if (!b) return;
    free(b->data);
    free(b);
}

static inline void vm_arena_init(VmArena* a, size_t block_size) {
    a->default_block_size = block_size ? block_size : VM_ARENA_DEFAULT_BLOCK_SIZE;
    a->current = vm_arena_block_create(a->default_block_size);
    a->total_allocated = a->current ? a->current->size : 0;
    a->total_used = 0;
    a->n_blocks = a->current ? 1 : 0;
}

static inline void vm_arena_destroy(VmArena* a) {
    VmArenaBlock* b = a->current;
    while (b) {
        VmArenaBlock* next = b->next;
        vm_arena_block_destroy(b);
        b = next;
    }
    a->current = NULL;
    a->total_allocated = 0;
    a->total_used = 0;
    a->n_blocks = 0;
}

/* O(1) bump-pointer allocation */
static inline void* vm_arena_alloc(VmArena* a, size_t size) {
    /* Align to VM_ARENA_ALIGNMENT */
    size = (size + VM_ARENA_ALIGNMENT - 1) & ~(VM_ARENA_ALIGNMENT - 1);

    if (!a->current || a->current->used + size > a->current->size) {
        /* Need new block */
        size_t block_size = a->default_block_size;
        if (size > block_size) block_size = size * 2;
        VmArenaBlock* nb = vm_arena_block_create(block_size);
        if (!nb) return NULL;
        nb->next = a->current;
        a->current = nb;
        a->total_allocated += block_size;
        a->n_blocks++;
    }

    void* ptr = a->current->data + a->current->used;
    a->current->used += size;
    a->total_used += size;
    return ptr;
}

/* Allocate object with header (matches Eshkol's pattern: header at offset -8) */
static inline void* vm_arena_alloc_object(VmArena* a, uint8_t subtype, size_t data_size) {
    size_t total = sizeof(VmObjectHeader) + data_size;
    uint8_t* raw = (uint8_t*)vm_arena_alloc(a, total);
    if (!raw) return NULL;

    VmObjectHeader* hdr = (VmObjectHeader*)raw;
    hdr->subtype = subtype;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    return raw + sizeof(VmObjectHeader);  /* return pointer past header */
}

/* Reset arena (free all blocks except first, reset that one) */
static inline void vm_arena_reset(VmArena* a) {
    /* Keep first block, free the rest */
    if (!a->current) return;
    VmArenaBlock* keep = NULL;
    VmArenaBlock* b = a->current;
    while (b) {
        VmArenaBlock* next = b->next;
        if (!keep) {
            keep = b;
            keep->used = 0;
            keep->next = NULL;
        } else {
            vm_arena_block_destroy(b);
        }
        b = next;
    }
    a->current = keep;
    a->total_used = 0;
    a->n_blocks = keep ? 1 : 0;
    a->total_allocated = keep ? keep->size : 0;
}

/*******************************************************************************
 * Region Operations
 ******************************************************************************/

static inline void vm_region_stack_init(VmRegionStack* rs) {
    memset(rs, 0, sizeof(*rs));
    vm_arena_init(&rs->global_arena, VM_ARENA_DEFAULT_BLOCK_SIZE);
}

static inline void vm_region_stack_destroy(VmRegionStack* rs) {
    /* Pop all regions */
    while (rs->depth > 0) {
        rs->depth--;
        VmRegion* r = rs->stack[rs->depth];
        vm_arena_destroy(&r->arena);
        free(r);
    }
    vm_arena_destroy(&rs->global_arena);
}

/* Get the active arena (current region's, or global) */
static inline VmArena* vm_active_arena(VmRegionStack* rs) {
    if (rs->depth > 0) return &rs->stack[rs->depth - 1]->arena;
    return &rs->global_arena;
}

/* Push a new region */
static inline VmRegion* vm_region_push(VmRegionStack* rs, const char* name, size_t size_hint) {
    if (rs->depth >= VM_ARENA_MAX_REGIONS) {
        fprintf(stderr, "ERROR: region stack overflow (max %d)\n", VM_ARENA_MAX_REGIONS);
        return NULL;
    }

    VmRegion* r = (VmRegion*)malloc(sizeof(VmRegion));
    if (!r) return NULL;
    vm_arena_init(&r->arena, size_hint ? size_hint : VM_ARENA_DEFAULT_BLOCK_SIZE);
    r->parent = rs->depth > 0 ? rs->stack[rs->depth - 1] : NULL;
    r->escape_count = 0;
    r->name = name;

    rs->stack[rs->depth++] = r;
    rs->total_regions_pushed++;
    if (rs->depth > rs->peak_depth) rs->peak_depth = rs->depth;

    return r;
}

/* Pop current region — frees all memory allocated in it */
static inline void vm_region_pop(VmRegionStack* rs) {
    if (rs->depth <= 0) {
        fprintf(stderr, "ERROR: region stack underflow\n");
        return;
    }
    rs->depth--;
    VmRegion* r = rs->stack[rs->depth];
    vm_arena_destroy(&r->arena);
    free(r);
}

/* Escape a value from current region to parent (copy to parent's arena) */
static inline void* vm_region_escape(VmRegionStack* rs, const void* src, size_t size) {
    if (rs->depth <= 0) return NULL; /* can't escape from global */

    VmRegion* current = rs->stack[rs->depth - 1];
    VmArena* target = current->parent ? &current->parent->arena : &rs->global_arena;

    void* dst = vm_arena_alloc(target, size);
    if (dst) {
        memcpy(dst, src, size);
        current->escape_count++;
    }
    return dst;
}

/* Allocate in the active region (or global arena) */
static inline void* vm_alloc(VmRegionStack* rs, size_t size) {
    VmArena* a = vm_active_arena(rs);
    void* ptr = vm_arena_alloc(a, size);
    size_t total = a->total_allocated;
    if (total > rs->peak_allocated) rs->peak_allocated = total;
    return ptr;
}

/* Allocate object with header in active region */
static inline void* vm_alloc_object(VmRegionStack* rs, uint8_t subtype, size_t data_size) {
    VmArena* a = vm_active_arena(rs);
    void* ptr = vm_arena_alloc_object(a, subtype, data_size);
    size_t total = a->total_allocated;
    if (total > rs->peak_allocated) rs->peak_allocated = total;
    return ptr;
}

/* Print metrics */
static inline void vm_arena_print_metrics(const VmRegionStack* rs) {
    const VmArena* a = rs->depth > 0 ? &rs->stack[rs->depth - 1]->arena : &rs->global_arena;
    printf("  [arena] %zu bytes used / %zu allocated (%d blocks), "
           "peak=%zu, regions_pushed=%d, peak_depth=%d\n",
           a->total_used, a->total_allocated, a->n_blocks,
           rs->peak_allocated, rs->total_regions_pushed, rs->peak_depth);
}

#endif /* VM_ARENA_H */
