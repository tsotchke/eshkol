/*
 * generate_frame_mapping_traces.c
 * Purpose: drive the native frame pool and three x86-64 address spaces of
 * lib/kernel/k2_mmu_frames.c through random allocate, free, map, remap and unmap
 * operations, and record after every operation:
 *   - the status and the complete state (for EshkolKernelFrameMappings.lean);
 *   - the invariants of the actual mapping relation, computed from native state;
 *   - for each PERM_ALIAS refusal, what a rule over the actual mappings would say.
 * Before the traces, it records native's verdict on a fixed list of address-space
 * initializations, including those the ASID partition refuses.
 *
 * Spaces: 0 kernel (is_user_space 0), 1 and 2 user. ASIDs: distinct (configuration
 * "distinct": 0, 1, 2) or shared between the two user spaces (configuration "shared":
 * 0, 2, 2). The partition admits both.
 * Pages are small integers; the virtual address is page << 12.
 *
 * Output:
 *   initprobe <asid> <is_user_space> => <STATUS>
 *   trace <t> total <n> reserved <k> asids <a0> <a1> <a2>
 *   <operation> => <STATUS> | <state> | inv <bits> | precision <class>
 *   state = counts <free> <allocated> frames <frame> ... entries <space:page:frame:perms> ...
 *   frame = owner,generation,refcount,user,read,write,exec,perms,asid|-,space|-
 *           (entries sorted by space, page; space is the index of mapped_user_aspace)
 *   inv bits, in order: RefcountExact UserRefcountExact ReadCountExact WriteCountExact
 *     ExecCountExact PermsExact OwnerExact UserEntriesOwned ReferentialIntegrity
 *     KernelFramesNotInUser NoWriteExecuteAlias SingleUserSpace MutableIsolation
 * Usage: <seed> <traces> <operations> <distinct|shared>
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
    case K2_ERR_CONSERVATION_VIOLATION: return "CONSERVATION_VIOLATION";
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

/* The index of an address space of this harness; -1 for NULL, -2 for any other pointer. */
static int space_index(const k2_aspace_t* a) {
    for (uint32_t s = 0; s < SPACE_COUNT; s++) if (a == &space[s]) return (int)s;
    return a ? -2 : -1;
}

static void print_state_and_invariants(void) {
    printf(" | counts %u %u frames", pool.free_frames, pool.allocated_frames);
    for (uint32_t i = 0; i < pool.total_frames; i++) {
        const k2_frame_meta_t* m = &pool.frames[i];
        printf(" %u,%u,%u,%u,%u,%u,%u,%u,", (unsigned)m->owner, m->generation, m->map_refcount,
               m->user_map_refcount, m->read_refcount, m->write_refcount, m->exec_refcount,
               m->mapped_perms);
        if (m->mapped_asid == K2_INVALID_INDEX) printf("-,"); else printf("%u,", m->mapped_asid);
        int si = space_index(m->mapped_user_aspace);
        if (si == -1) printf("-"); else printf("%d", si);
    }
    printf(" entries");
    int refcount_exact = 1, user_exact = 1, read_exact = 1, write_exact = 1, exec_exact = 1,
        perms_exact = 1, owner_exact = 1, entries_owned = 1, referential = 1, kernel_not_user = 1,
        no_wx_alias = 1, single_user_space = 1, mutable_isolation = 1;
    uint32_t counted[MAXIMUM_TOTAL_FRAMES][5];
    memset(counted, 0, sizeof counted);
    for (uint32_t s = 0; s < SPACE_COUNT; s++)
        for (uint32_t page = 0; page < PAGE_COUNT; page++) {
            uint32_t perms;
            int f = frame_of(&space[s], page, &perms);
            if (f < 0) continue;
            printf(" %u:%u:%d:%u", s, page, f, perms);
            if ((uint32_t)f >= pool.total_frames) { referential = 0; continue; }
            counted[f][0]++;
            if (space[s].is_user_space) counted[f][1]++;
            if (perms & K2_PERM_READ) counted[f][2]++;
            if (perms & K2_PERM_WRITE) counted[f][3]++;
            if (perms & K2_PERM_EXEC) counted[f][4]++;
            const k2_frame_meta_t* m = &pool.frames[f];
            if (m->owner == K2_OWNER_FREE) referential = 0;
            if (space[s].is_user_space && m->owner == K2_OWNER_KERNEL) kernel_not_user = 0;
            if (space[s].is_user_space && m->mapped_user_aspace != &space[s]) entries_owned = 0;
            for (uint32_t s2 = 0; s2 < SPACE_COUNT; s2++)
                for (uint32_t page2 = 0; page2 < PAGE_COUNT; page2++) {
                    uint32_t perms2;
                    if (frame_of(&space[s2], page2, &perms2) != f) continue;
                    if ((perms & K2_PERM_WRITE) && (perms2 & K2_PERM_EXEC)) no_wx_alias = 0;
                    if (space[s].is_user_space && space[s2].is_user_space && s != s2) {
                        single_user_space = 0;
                        if (space[s].asid != space[s2].asid &&
                            ((perms & K2_PERM_WRITE) || (perms2 & K2_PERM_WRITE))) mutable_isolation = 0;
                    }
                }
        }
    for (uint32_t i = 0; i < pool.total_frames; i++) {
        const k2_frame_meta_t* m = &pool.frames[i];
        if (m->map_refcount != counted[i][0]) refcount_exact = 0;
        if (m->user_map_refcount != counted[i][1]) user_exact = 0;
        if (m->read_refcount != counted[i][2]) read_exact = 0;
        if (m->write_refcount != counted[i][3]) write_exact = 0;
        if (m->exec_refcount != counted[i][4]) exec_exact = 0;
        uint32_t recomputed = (m->read_refcount ? K2_PERM_READ : 0u) |
                              (m->write_refcount ? K2_PERM_WRITE : 0u) |
                              (m->exec_refcount ? K2_PERM_EXEC : 0u);
        if (m->mapped_perms != recomputed) perms_exact = 0;
        if (m->user_map_refcount == 0) {
            if (m->mapped_user_aspace != NULL || m->mapped_asid != K2_INVALID_INDEX) owner_exact = 0;
        } else {
            int si = space_index(m->mapped_user_aspace);
            if (si < 0 || !space[si].is_user_space || m->mapped_asid != space[si].asid) owner_exact = 0;
        }
    }
    printf(" | inv %d%d%d%d%d%d%d%d%d%d%d%d%d", refcount_exact, user_exact, read_exact, write_exact,
           exec_exact, perms_exact, owner_exact, entries_owned, referential, kernel_not_user,
           no_wx_alias, single_user_space, mutable_isolation);
}

/* What a rule over the actual mappings says about mapping (or remapping) frame f with
 * perms p in space s; `skip_page` excludes the entry being remapped.
 *   genuine-write-execute      a W/X alias would result
 *   cross-space-writable       a user mapping shared with another user space, one writable
 *   cross-space-read-only      read-only sharing with another user space (refused by design)
 *   unexplained                none of these: a precision failure */
static const char* classify_alias(uint32_t s, int skip_page, int f, uint32_t p) {
    int wx_conflict = 0, writable_sharing = 0, read_only_sharing = 0;
    for (uint32_t s2 = 0; s2 < SPACE_COUNT; s2++)
        for (uint32_t page2 = 0; page2 < PAGE_COUNT; page2++) {
            if (s2 == s && (int)page2 == skip_page) continue;
            uint32_t perms2;
            if (frame_of(&space[s2], page2, &perms2) != f) continue;
            if (((perms2 & K2_PERM_WRITE) && (p & K2_PERM_EXEC)) ||
                ((perms2 & K2_PERM_EXEC) && (p & K2_PERM_WRITE))) wx_conflict = 1;
            if (space[s].is_user_space && space[s2].is_user_space && s2 != s) {
                if ((perms2 & K2_PERM_WRITE) || (p & K2_PERM_WRITE)) writable_sharing = 1;
                else read_only_sharing = 1;
            }
        }
    if (wx_conflict) return "genuine-write-execute";
    if (writable_sharing) return "cross-space-writable";
    if (read_only_sharing) return "cross-space-read-only";
    return "unexplained";
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

static void probe_initialization(void) {
    static const uint32_t asids[] = {0u, 1u, 2u, 7u, K2_INVALID_INDEX - 1u, K2_INVALID_INDEX};
    for (uint32_t user = 0; user < 2u; user++)
        for (uint32_t i = 0; i < sizeof asids / sizeof asids[0]; i++) {
            k2_aspace_t probe;
            k2_status_t st = k2_aspace_init(&probe, K2_MMU_X86_64_4L, asids[i], user);
            printf("initprobe %u %u => %s\n", asids[i], user, status_name(st));
        }
}

int main(int argc, char** argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s <seed> <traces> <operations> <distinct|shared>\n", argv[0]); return 1; }
    random_state = strtoull(argv[1], 0, 10) | 1u;
    unsigned trace_count = (unsigned)atoi(argv[2]), operation_count = (unsigned)atoi(argv[3]);
    int shared = strcmp(argv[4], "shared") == 0;
    if (!shared && strcmp(argv[4], "distinct") != 0) { fprintf(stderr, "unknown configuration %s\n", argv[4]); return 1; }
    uint32_t asid[SPACE_COUNT] = {K2_KERNEL_ASID, shared ? 2u : 1u, 2u};
    static k2_page_frame_t handle[MAXIMUM_HANDLES];
    probe_initialization();
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
