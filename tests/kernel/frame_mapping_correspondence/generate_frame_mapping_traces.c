/*
 * generate_frame_mapping_traces.c
 * Purpose: drive the native frame pool and three x86-64 address spaces of
 * lib/kernel/k2_mmu_frames.c through random allocate, free, map, remap and unmap
 * operations, and record after every operation:
 *   - the status and the complete state (for EshkolKernelFrameMappings.lean);
 *   - the invariants of the actual mapping relation, computed from native state;
 *   - for each PERM_ALIAS refusal, whether a rule over the actual mappings would
 *     also refuse (a genuine conflict) or accept (a precision failure).
 *
 * Spaces: 0 kernel (is_user_space 0), 1 and 2 user. ASIDs: distinct (configuration
 * "unique": 0, 1, 2) or colliding (configuration "collide": the kernel space has
 * ASID 2, the same as user space 2).
 * Pages are small integers; the virtual address is page << 12.
 *
 * Output per operation:
 *   <operation> => <STATUS> | <state> | inv <RKPCWMS> | precision <class>
 *   state = counts <free> <allocated> frames <owner,gen,refcount,union,lastAsid|-> ...
 *           entries <space:page:frame:perms> ... (sorted by space, page)
 *   inv bits: RefcountExact ReferentialIntegrity KernelFramesNotInUser PermUnionCovers
 *             NoWriteExecuteAlias MutableIsolation SingleUserAsid
 * Usage: <seed> <traces> <operations> <unique|collide>
 */
#include <eshkol/kernel/k2_mmu_frames.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXIMUM_TOTAL_FRAMES 12u
#define MAXIMUM_HANDLES 512u
#define SPACE_COUNT 3u
#define PAGE_COUNT 6u
#define PHYSICAL_BASE 0x200000ULL

static uint64_t random_state;
static uint32_t random_below(uint32_t bound) {
    random_state ^= random_state << 13; random_state ^= random_state >> 7; random_state ^= random_state << 17;
    return (uint32_t)(random_state % bound);
}

static const char* status_name(k2_status_t s) {
    switch (s) {
    case K2_OK: return "OK";
    case K2_ERR_INVALID_ARG: return "INVALID_ARG";
    case K2_ERR_OUT_OF_FRAMES: return "OUT_OF_FRAMES";
    case K2_ERR_DOUBLE_FREE: return "DOUBLE_FREE";
    case K2_ERR_STALE_HANDLE: return "STALE_HANDLE";
    case K2_ERR_KERNEL_FRAME_LOCKED: return "KERNEL_FRAME_LOCKED";
    case K2_ERR_FRAME_STILL_MAPPED: return "FRAME_STILL_MAPPED";
    case K2_ERR_WX_VIOLATION: return "WX_VIOLATION";
    case K2_ERR_PRIV_VIOLATION: return "PRIV_VIOLATION";
    case K2_ERR_KERNEL_ALIAS: return "KERNEL_ALIAS";
    case K2_ERR_PERM_ALIAS: return "PERM_ALIAS";
    case K2_ERR_VA_ALREADY_MAPPED: return "VA_ALREADY_MAPPED";
    case K2_ERR_VA_NOT_MAPPED: return "VA_NOT_MAPPED";
    default: return "OTHER";
    }
}

static k2_frame_pool_t pool;
static k2_aspace_t space[SPACE_COUNT];

static int frame_of(const k2_aspace_t* a, uint32_t page, uint32_t* perms) {
    for (uint32_t i = 0; i < K2_MAX_PT_ENTRIES; i++)
        if (a->entries[i].valid && a->entries[i].va == ((uint64_t)page << K2_PAGE_SHIFT)) {
            if (perms) *perms = a->entries[i].perms;
            return (int)a->entries[i].frame_idx;
        }
    return -1;
}

static void print_state_and_invariants(void) {
    printf(" | counts %u %u frames", pool.free_frames, pool.allocated_frames);
    for (uint32_t i = 0; i < pool.total_frames; i++) {
        const k2_frame_meta_t* m = &pool.frames[i];
        printf(" %u,%u,%u,%u,", (unsigned)m->owner, m->generation, m->map_refcount, m->mapped_perms & 7u);
        if (m->mapped_asid == K2_INVALID_INDEX) printf("-"); else printf("%u", m->mapped_asid);
    }
    printf(" entries");
    int refcount_exact = 1, referential = 1, kernel_not_user = 1, union_covers = 1,
        no_wx_alias = 1, mutable_isolation = 1, single_user_asid = 1;
    uint32_t counted[MAXIMUM_TOTAL_FRAMES] = {0};
    for (uint32_t s = 0; s < SPACE_COUNT; s++)
        for (uint32_t page = 0; page < PAGE_COUNT; page++) {
            uint32_t perms;
            int f = frame_of(&space[s], page, &perms);
            if (f < 0) continue;
            printf(" %u:%u:%d:%u", s, page, f, perms);
            if ((uint32_t)f >= pool.total_frames) { referential = 0; continue; }
            counted[f]++;
            const k2_frame_meta_t* m = &pool.frames[f];
            if (m->owner == K2_OWNER_FREE) referential = 0;
            if (space[s].is_user_space && m->owner == K2_OWNER_KERNEL) kernel_not_user = 0;
            if (((perms & K2_PERM_WRITE) && !(m->mapped_perms & K2_PERM_WRITE)) ||
                ((perms & K2_PERM_EXEC) && !(m->mapped_perms & K2_PERM_EXEC))) union_covers = 0;
            for (uint32_t s2 = 0; s2 < SPACE_COUNT; s2++)
                for (uint32_t page2 = 0; page2 < PAGE_COUNT; page2++) {
                    uint32_t perms2;
                    if (frame_of(&space[s2], page2, &perms2) != f) continue;
                    if ((perms & K2_PERM_WRITE) && (perms2 & K2_PERM_EXEC)) no_wx_alias = 0;
                    if (space[s].is_user_space && space[s2].is_user_space && space[s].asid != space[s2].asid) {
                        single_user_asid = 0;
                        if ((perms & K2_PERM_WRITE) || (perms2 & K2_PERM_WRITE)) mutable_isolation = 0;
                    }
                }
        }
    for (uint32_t i = 0; i < pool.total_frames; i++)
        if (pool.frames[i].map_refcount != counted[i]) refcount_exact = 0;
    printf(" | inv %d%d%d%d%d%d%d", refcount_exact, referential, kernel_not_user, union_covers,
           no_wx_alias, mutable_isolation, single_user_asid);
}

/* Would a rule over the actual mappings refuse mapping (or remapping) frame f with
 * perms p in space s?  `skip_page` excludes the entry being remapped. */
static const char* classify_alias(uint32_t s, int skip_page, int f, uint32_t p) {
    int wx_conflict = 0, isolation_conflict = 0;
    for (uint32_t s2 = 0; s2 < SPACE_COUNT; s2++)
        for (uint32_t page2 = 0; page2 < PAGE_COUNT; page2++) {
            if (s2 == s && (int)page2 == skip_page) continue;
            uint32_t perms2;
            if (frame_of(&space[s2], page2, &perms2) != f) continue;
            if (((perms2 & K2_PERM_WRITE) && (p & K2_PERM_EXEC)) ||
                ((perms2 & K2_PERM_EXEC) && (p & K2_PERM_WRITE))) wx_conflict = 1;
            if (space[s].is_user_space && space[s2].is_user_space && space[s].asid != space[s2].asid &&
                ((perms2 & K2_PERM_WRITE) || (p & K2_PERM_WRITE))) isolation_conflict = 1;
        }
    if (wx_conflict) return "genuine-write-execute";
    if (isolation_conflict) return "genuine-isolation";
    /* Which native check refused: the permission union, or the last-ASID rule. */
    const k2_frame_meta_t* m = &pool.frames[f];
    int union_fired = ((m->mapped_perms & K2_PERM_WRITE) && (p & K2_PERM_EXEC)) ||
                      ((m->mapped_perms & K2_PERM_EXEC) && (p & K2_PERM_WRITE));
    if (union_fired) return "precision-stale-permission-union";
    int read_only_sharing = 0;
    for (uint32_t s2 = 0; s2 < SPACE_COUNT; s2++)
        for (uint32_t page2 = 0; page2 < PAGE_COUNT; page2++)
            if (frame_of(&space[s2], page2, 0) == f && space[s2].is_user_space &&
                space[s2].asid != space[s].asid) read_only_sharing = 1;
    if (read_only_sharing) return "precision-read-only-sharing";
    /* The recorded last ASID: a kernel space's, or a user space's that no longer maps f. */
    for (uint32_t s2 = 0; s2 < SPACE_COUNT; s2++)
        if (!space[s2].is_user_space && space[s2].asid == m->mapped_asid) return "precision-last-asid-is-kernel";
    return "precision-stale-last-asid";
}

static uint32_t random_perms(int user) {
    if (random_below(7) == 0) return random_below(32);
    uint32_t p = K2_PERM_READ;
    uint32_t c = random_below(3);
    if (c == 1) p |= K2_PERM_WRITE;
    if (c == 2) p |= K2_PERM_EXEC;
    if (user) p |= K2_PERM_USER;
    else if (random_below(2)) p |= K2_PERM_KERNEL_ONLY;
    return p;
}

int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s <seed> <traces> <operations> <unique|collide>\n", argv[0]); return 1; }
    random_state = strtoull(argv[1], 0, 10) | 1u;
    unsigned trace_count = (unsigned)atoi(argv[2]), operation_count = (unsigned)atoi(argv[3]);
    int collide = strcmp(argv[4], "collide") == 0;
    uint32_t asid[SPACE_COUNT] = {collide ? 2u : 0u, 1u, 2u};
    static k2_page_frame_t handle[MAXIMUM_HANDLES];
    for (unsigned t = 0; t < trace_count; t++) {
        uint32_t total = 2u + random_below(MAXIMUM_TOTAL_FRAMES - 1u);
        uint32_t reserved = random_below(3u);
        if (reserved > total) reserved = total;
        if (k2_frame_pool_init(&pool, PHYSICAL_BASE, total, reserved) != K2_OK) return 3;
        for (uint32_t s = 0; s < SPACE_COUNT; s++)
            if (k2_aspace_init(&space[s], K2_MMU_X86_64_4L, asid[s], s != 0u) != K2_OK) return 3;
        printf("trace %u total %u reserved %u asids %u %u %u\n", t, total, reserved, asid[0], asid[1], asid[2]);
        unsigned handle_count = 0;
        for (uint32_t i = 0; i < reserved; i++)
            if (k2_frame_get_kernel_handle(&pool, i, &handle[handle_count]) == K2_OK) handle_count++;
        for (unsigned k = 0; k < operation_count; k++) {
            uint32_t choice = random_below(20);
            const char* precision = "-";
            if (choice < 3 || handle_count == 0) {
                k2_page_frame_t f;
                int kernel = choice == 0;
                k2_status_t st = kernel ? k2_frame_alloc_kernel(&pool, &f) : k2_frame_alloc_user(&pool, &f);
                if (st == K2_OK && handle_count < MAXIMUM_HANDLES) handle[handle_count++] = f;
                printf("%s => %s %u", kernel ? "alloc_kernel" : "alloc_user", status_name(st), st == K2_OK ? f.frame_idx : 0u);
            } else {
                k2_page_frame_t f = handle[random_below(handle_count)];
                uint32_t tamper = random_below(16);
                if (tamper == 0) f.generation += 1u;
                if (tamper == 1) f.phys_addr += K2_PAGE_SIZE;
                int phys_ok = f.phys_addr == PHYSICAL_BASE + ((uint64_t)f.frame_idx << K2_PAGE_SHIFT);
                uint32_t s = random_below(SPACE_COUNT), page = random_below(PAGE_COUNT);
                if (choice < 6) {
                    k2_status_t st = k2_frame_free_user(&pool, &f);
                    printf("free %u %u %d => %s", f.frame_idx, f.generation, phys_ok, status_name(st));
                } else if (choice < 13) {
                    uint32_t p = random_perms(s != 0u);
                    k2_status_t st = k2_aspace_map(&space[s], &pool, (uint64_t)page << K2_PAGE_SHIFT, &f, p);
                    if (st == K2_ERR_PERM_ALIAS) precision = classify_alias(s, -1, (int)f.frame_idx, p);
                    printf("map %u %u %u %u %d %u => %s", s, page, f.frame_idx, f.generation, phys_ok, p, status_name(st));
                } else if (choice < 17) {
                    uint32_t p = random_perms(s != 0u);
                    int target = frame_of(&space[s], page, 0);
                    k2_status_t st = k2_aspace_remap_perms(&space[s], &pool, (uint64_t)page << K2_PAGE_SHIFT, p);
                    if (st == K2_ERR_PERM_ALIAS) precision = classify_alias(s, (int)page, target, p);
                    printf("remap %u %u %u => %s", s, page, p, status_name(st));
                } else {
                    k2_status_t st = k2_aspace_unmap(&space[s], &pool, (uint64_t)page << K2_PAGE_SHIFT);
                    printf("unmap %u %u => %s", s, page, status_name(st));
                }
            }
            print_state_and_invariants();
            printf(" | precision %s\n", precision);
        }
    }
    return 0;
}
