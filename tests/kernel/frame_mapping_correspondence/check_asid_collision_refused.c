/* check_asid_collision_refused.c
 * Directed check that the cross-ASID isolation failure reproduced before the ASID
 * partition (tag historical/frame-mapping-composition-before-asid-partition,
 * reproduce_asid_collision.c) is closed in lib/kernel/k2_mmu_frames.c.
 *   1. The colliding configuration is refused at initialization: a kernel space with
 *      ASID 2 gives INVALID_ARG.
 *   2. The former sequence with a valid kernel space: user space A (ASID 1) maps frame F
 *      read-only; the kernel space maps F read-only; user space B (ASID 2) maps F
 *      writable. The last map is refused with PERM_ALIAS.
 *   3. Two user spaces that share ASID 2, which initialization admits: A maps F
 *      read-only; B's writable and read-only maps of F are both refused with PERM_ALIAS.
 * Exit 0 when all three hold. */
#include <eshkol/kernel/k2_mmu_frames.h>
#include <stdio.h>

static int setup(k2_frame_pool_t* pool, k2_aspace_t* kernel, k2_aspace_t* a, k2_aspace_t* b,
                 uint32_t asid_a, uint32_t asid_b, k2_page_frame_t* f) {
    return k2_frame_pool_init(pool, 0x200000ULL, 4, 0) == K2_OK &&
           k2_aspace_init(kernel, K2_MMU_X86_64_4L, K2_KERNEL_ASID, 0) == K2_OK &&
           k2_aspace_init(a, K2_MMU_X86_64_4L, asid_a, 1) == K2_OK &&
           k2_aspace_init(b, K2_MMU_X86_64_4L, asid_b, 1) == K2_OK &&
           k2_frame_alloc_user(pool, f) == K2_OK;
}

int main(void) {
    k2_frame_pool_t pool; k2_aspace_t kernel, a, b; k2_page_frame_t f;
    const uint32_t ru = K2_PERM_READ | K2_PERM_USER, rwu = ru | K2_PERM_WRITE;
    k2_status_t s0 = k2_aspace_init(&kernel, K2_MMU_X86_64_4L, 2, 0);
    printf("1. kernel space with ASID 2: init -> %d (INVALID_ARG = %d)\n", (int)s0, (int)K2_ERR_INVALID_ARG);

    if (!setup(&pool, &kernel, &a, &b, 1, 2, &f)) return 2;
    k2_status_t s1 = k2_aspace_map(&a, &pool, 0x2000, &f, ru);
    k2_status_t s2 = k2_aspace_map(&kernel, &pool, 0x0000, &f, K2_PERM_READ | K2_PERM_KERNEL_ONLY);
    k2_status_t s3 = k2_aspace_map(&b, &pool, 0x5000, &f, rwu);
    printf("2. A maps R|U -> %d; kernel maps R|K -> %d; B (ASID 2) maps R|W|U -> %d\n",
           (int)s1, (int)s2, (int)s3);

    if (!setup(&pool, &kernel, &a, &b, 2, 2, &f)) return 2;
    k2_status_t t1 = k2_aspace_map(&a, &pool, 0x2000, &f, ru);
    k2_status_t t2 = k2_aspace_map(&b, &pool, 0x5000, &f, rwu);
    k2_status_t t3 = k2_aspace_map(&b, &pool, 0x5000, &f, ru);
    printf("3. A and B share ASID 2: A maps R|U -> %d; B maps R|W|U -> %d; B maps R|U -> %d\n",
           (int)t1, (int)t2, (int)t3);
    printf("K2_OK = %d, K2_ERR_PERM_ALIAS = %d\n", (int)K2_OK, (int)K2_ERR_PERM_ALIAS);
    return (s0 == K2_ERR_INVALID_ARG && s1 == K2_OK && s2 == K2_OK && s3 == K2_ERR_PERM_ALIAS &&
            t1 == K2_OK && t2 == K2_ERR_PERM_ALIAS && t3 == K2_ERR_PERM_ALIAS) ? 0 : 1;
}
