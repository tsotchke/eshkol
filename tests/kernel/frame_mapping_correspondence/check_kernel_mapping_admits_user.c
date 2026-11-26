/* check_kernel_mapping_admits_user.c
 * Directed check that the precision failure reproduced before the ASID partition (tag
 * historical/frame-mapping-composition-before-asid-partition,
 * reproduce_kernel_last_asid_refusal.c) is gone in lib/kernel/k2_mmu_frames.c: a user
 * frame mapped only by the kernel space is admitted to a user space.
 * Sequence: user frame F; the kernel space maps F read-only; user space A maps F
 * read-only; the kernel space maps F read-only again at another page.
 * Exit 0 when every map returns K2_OK. */
#include <eshkol/kernel/k2_mmu_frames.h>
#include <stdio.h>
int main(void) {
    k2_frame_pool_t pool; k2_aspace_t kernel, a; k2_page_frame_t f;
    if (k2_frame_pool_init(&pool, 0x200000ULL, 4, 0) ||
        k2_aspace_init(&kernel, K2_MMU_X86_64_4L, K2_KERNEL_ASID, 0) ||
        k2_aspace_init(&a, K2_MMU_X86_64_4L, 1, 1) || k2_frame_alloc_user(&pool, &f)) return 2;
    k2_status_t s1 = k2_aspace_map(&kernel, &pool, 0x0000, &f, K2_PERM_READ | K2_PERM_KERNEL_ONLY);
    k2_status_t s2 = k2_aspace_map(&a, &pool, 0x2000, &f, K2_PERM_READ | K2_PERM_USER);
    k2_status_t s3 = k2_aspace_map(&kernel, &pool, 0x1000, &f, K2_PERM_READ);
    printf("kernel maps R|K -> %d; A maps R|U -> %d; kernel maps R -> %d (K2_OK = %d)\n",
           (int)s1, (int)s2, (int)s3, (int)K2_OK);
    return (s1 == K2_OK && s2 == K2_OK && s3 == K2_OK) ? 0 : 1;
}
