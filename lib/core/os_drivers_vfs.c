/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K4: Freestanding zero-LLVM, zero-POSIX, zero-malloc
 * userspace OALR driver runtime, VirtIO split-virtqueue block/net driver,
 * and atomic SHA-256 content-addressed VFS.
 *
 * External symbol closure: ZERO undefined external symbols (pure C11).
 */

#include <eshkol/core/os_drivers_vfs.h>

#include <stddef.h>
#include <stdint.h>

/* ── Internal slot states ── */
enum {
    DRV_SLOT_FREE     = 0u,
    DRV_SLOT_ACTIVE   = 1u,
    DRV_SLOT_RELEASED = 2u,
    DRV_SLOT_REVOKED  = 3u
};

/* ── CAS VFS on-disk geometry (512-byte sectors) ──
 *   Sector 0       : Superblock A (512 bytes, CRC32C at [508..511])
 *   Sector 1       : Superblock B (512 bytes, CRC32C at [508..511])
 *   Sectors 2..7   : Index Slot A (32 * 96 = 3072 bytes = 6 sectors)
 *   Sectors 8..13  : Index Slot B (32 * 96 = 3072 bytes = 6 sectors)
 *   Sectors 14..N  : Append-only copy-on-write content-addressed payload pool
 */
#define VFS_SB_MAGIC            0x45534856u /* "ESHV" */
#define VFS_SB_VERSION          1u
#define VFS_INDEX_BYTES         (ESHKOL_OS_VFS_MAX_ENTRIES * sizeof(eshkol_os_vfs_entry_t))
#define VFS_INDEX_SECTORS       (VFS_INDEX_BYTES / ESHKOL_OS_VFS_SECTOR_SIZE) /* 6 */
#define VFS_SLOT_A_INDEX_SECTOR 2u
#define VFS_SLOT_B_INDEX_SECTOR (2u + VFS_INDEX_SECTORS) /* 8 */
#define VFS_FIRST_DATA_SECTOR   (2u + 2u * VFS_INDEX_SECTORS) /* 14 */

/* ── Freestanding memory & endian helpers (volatile loops avoid libc lowering) ── */

static void drv_zero(void* dst, size_t n) {
    volatile uint8_t* p = (volatile uint8_t*)dst;
    for (size_t i = 0u; i < n; i++) {
        p[i] = 0u;
    }
}

static void drv_copy(void* dst, const void* src, size_t n) {
    volatile uint8_t* d = (volatile uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    for (size_t i = 0u; i < n; i++) {
        d[i] = s[i];
    }
}

static int drv_memeq(const void* a, const void* b, size_t n) {
    const uint8_t* pa = (const uint8_t*)a;
    const uint8_t* pb = (const uint8_t*)b;
    uint8_t diff = 0u;
    for (size_t i = 0u; i < n; i++) {
        diff |= (uint8_t)(pa[i] ^ pb[i]);
    }
    return diff == 0u;
}

static size_t drv_strlen(const char* s) {
    if (!s) return 0u;
    size_t n = 0u;
    while (s[n] != '\0') n++;
    return n;
}

static int drv_streq(const char* a, const char* b) {
    if (!a || !b) return 0;
    size_t i = 0u;
    while (a[i] != '\0' && b[i] != '\0') {
        if (a[i] != b[i]) return 0;
        i++;
    }
    return a[i] == b[i];
}

static uint32_t drv_rd_u32_le(const uint8_t* p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t drv_rd_u64_le(const uint8_t* p) {
    return ((uint64_t)drv_rd_u32_le(p)) |
           ((uint64_t)drv_rd_u32_le(p + 4u) << 32);
}

static void drv_wr_u32_le(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static void drv_wr_u64_le(uint8_t* p, uint64_t v) {
    drv_wr_u32_le(p, (uint32_t)(v & 0xFFFFFFFFULL));
    drv_wr_u32_le(p + 4u, (uint32_t)((v >> 32) & 0xFFFFFFFFULL));
}

static uint32_t drv_rd_u32_be(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) |
           ((uint32_t)p[3]);
}

static void drv_wr_u32_be(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)((v >> 24) & 0xFFu);
    p[1] = (uint8_t)((v >> 16) & 0xFFu);
    p[2] = (uint8_t)((v >> 8) & 0xFFu);
    p[3] = (uint8_t)(v & 0xFFu);
}

/* ── CRC32C (Castagnoli polynomial 0x82F63B78) ── */

static uint32_t drv_crc32c(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0u; i < len; i++) {
        crc ^= (uint32_t)data[i];
        for (int b = 0; b < 8; b++) {
            uint32_t mask = (crc & 1u) ? 0x82F63B78u : 0u;
            crc = (crc >> 1) ^ mask;
        }
    }
    return ~crc;
}

/* ── FIPS 180-4 Streaming SHA-256 (zero external deps) ── */

typedef struct drv_sha256_ctx {
    uint32_t state[8];
    uint64_t bit_len;
    uint8_t block[64];
    uint32_t block_used;
} drv_sha256_ctx_t;

static uint32_t sha_rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

static void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    static const uint32_t k[64] = {
        0x428A2F98u, 0x71374491u, 0xB5C0FBCFu, 0xE9B5DBA5u,
        0x3956C25Bu, 0x59F111F1u, 0x923F82A4u, 0xAB1C5ED5u,
        0xD807AA98u, 0x12835B01u, 0x243185BEu, 0x550C7DC3u,
        0x72BE5D74u, 0x80DEB1FEu, 0x9BDC06A7u, 0xC19BF174u,
        0xE49B69C1u, 0xEFBE4786u, 0x0FC19DC6u, 0x240CA1CCu,
        0x2DE92C6Fu, 0x4A7484AAu, 0x5CB0A9DCu, 0x76F988DAu,
        0x983E5152u, 0xA831C66Du, 0xB00327C8u, 0xBF597FC7u,
        0xC6E00BF3u, 0xD5A79147u, 0x06CA6351u, 0x14292967u,
        0x27B70A85u, 0x2E1B2138u, 0x4D2C6DFCu, 0x53380D13u,
        0x650A7354u, 0x766A0ABBu, 0x81C2C92Eu, 0x92722C85u,
        0xA2BFE8A1u, 0xA81A664Bu, 0xC24B8B70u, 0xC76C51A3u,
        0xD192E819u, 0xD6990624u, 0xF40E3585u, 0x106AA070u,
        0x19A4C116u, 0x1E376C08u, 0x2748774Cu, 0x34B0BCB5u,
        0x391C0CB3u, 0x4ED8AA4Au, 0x5B9CCA4Fu, 0x682E6FF3u,
        0x748F82EEu, 0x78A5636Fu, 0x84C87814u, 0x8CC70208u,
        0x90BEFFFAu, 0xA4506CEBu, 0xBEF9A3F7u, 0xC67178F2u
    };

    uint32_t w[64];
    for (uint32_t i = 0u; i < 16u; i++) {
        w[i] = drv_rd_u32_be(block + i * 4u);
    }
    for (uint32_t i = 16u; i < 64u; i++) {
        uint32_t s0 = sha_rotr(w[i - 15u], 7u) ^ sha_rotr(w[i - 15u], 18u) ^ (w[i - 15u] >> 3);
        uint32_t s1 = sha_rotr(w[i - 2u], 17u) ^ sha_rotr(w[i - 2u], 19u) ^ (w[i - 2u] >> 10);
        w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (uint32_t i = 0u; i < 64u; i++) {
        uint32_t S1 = sha_rotr(e, 6u) ^ sha_rotr(e, 11u) ^ sha_rotr(e, 25u);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + k[i] + w[i];
        uint32_t S0 = sha_rotr(a, 2u) ^ sha_rotr(a, 13u) ^ sha_rotr(a, 22u);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

static void sha256_init(drv_sha256_ctx_t* ctx) {
    drv_zero(ctx, sizeof(*ctx));
    ctx->state[0] = 0x6A09E667u;
    ctx->state[1] = 0xBB67AE85u;
    ctx->state[2] = 0x3C6EF372u;
    ctx->state[3] = 0xA54FF53Au;
    ctx->state[4] = 0x510E527Fu;
    ctx->state[5] = 0x9B05688Cu;
    ctx->state[6] = 0x1F83D9ABu;
    ctx->state[7] = 0x5BE0CD19u;
}

static void sha256_update(drv_sha256_ctx_t* ctx, const uint8_t* data, size_t len) {
    for (size_t i = 0u; i < len; i++) {
        ctx->block[ctx->block_used++] = data[i];
        if (ctx->block_used == 64u) {
            sha256_compress(ctx->state, ctx->block);
            ctx->bit_len += 512ULL;
            ctx->block_used = 0u;
        }
    }
}

static void sha256_final(drv_sha256_ctx_t* ctx, uint8_t out[ESHKOL_OS_SHA256_BYTES]) {
    uint64_t total_bits = ctx->bit_len + (uint64_t)ctx->block_used * 8ULL;
    ctx->block[ctx->block_used++] = 0x80u;
    if (ctx->block_used > 56u) {
        while (ctx->block_used < 64u) ctx->block[ctx->block_used++] = 0u;
        sha256_compress(ctx->state, ctx->block);
        ctx->block_used = 0u;
    }
    while (ctx->block_used < 56u) ctx->block[ctx->block_used++] = 0u;
    drv_wr_u32_be(ctx->block + 56u, (uint32_t)(total_bits >> 32));
    drv_wr_u32_be(ctx->block + 60u, (uint32_t)(total_bits & 0xFFFFFFFFULL));
    sha256_compress(ctx->state, ctx->block);
    for (uint32_t i = 0u; i < 8u; i++) {
        drv_wr_u32_be(out + i * 4u, ctx->state[i]);
    }
}

void eshkol_os_sha256(const void* data,
                      size_t len,
                      uint8_t out_digest[ESHKOL_OS_SHA256_BYTES]) {
    if (!out_digest) return;
    drv_sha256_ctx_t ctx;
    sha256_init(&ctx);
    if (data && len > 0u) {
        sha256_update(&ctx, (const uint8_t*)data, len);
    }
    sha256_final(&ctx, out_digest);
}

/* ── Capability syscall helper ── */

static int64_t drv_syscall(eshkol_os_drv_runtime_t* rt,
                           uint64_t sysno,
                           uint64_t a0, uint64_t a1, uint64_t a2,
                           uint64_t a3, uint64_t a4, uint64_t a5) {
    rt->total_syscalls++;
    return rt->syscall_fn(sysno, a0, a1, a2, a3, a4, a5);
}

static eshkol_os_drv_status_t drv_map_syscall_err(int64_t rc) {
    if (rc == ESHKOL_OS_ERR_REVOKED ||
        rc == ESHKOL_OS_ERR_RIGHTS_DENIED ||
        rc == ESHKOL_OS_ERR_INVALID_CAP) {
        return ESHKOL_DRV_ERR_REVOKED;
    }
    return ESHKOL_DRV_ERR_SYSCALL_FAILED;
}

/* ── Driver Runtime & Linear Resource Handles ── */

eshkol_os_drv_status_t eshkol_os_drv_runtime_init(
    eshkol_os_drv_runtime_t* rt,
    const eshkol_os_drv_runtime_config_t* config) {
    if (!rt || !config || !config->syscall_fn) {
        return ESHKOL_DRV_ERR_INVALID_ARG;
    }
    drv_zero(rt, sizeof(*rt));
    rt->syscall_fn = config->syscall_fn;
    rt->page_size = config->page_size ? config->page_size : 4096ULL;
    rt->next_generation = 1ULL;
    rt->initialized = 1u;
    return ESHKOL_DRV_OK;
}

static eshkol_os_drv_status_t drv_validate_region(
    const eshkol_os_drv_runtime_t* rt,
    const eshkol_os_device_region_t* handle,
    int for_release) {
    if (!rt || !rt->initialized || !handle) return ESHKOL_DRV_ERR_INVALID_ARG;
    if (handle->kind != ESHKOL_DRV_KIND_DEVICE_REGION || handle->slot >= ESHKOL_OS_DRV_MAX_SLOTS) {
        return ESHKOL_DRV_ERR_STALE_HANDLE;
    }
    const eshkol_os_dev_region_slot_t* s = &rt->regions[handle->slot];
    if (s->state == DRV_SLOT_REVOKED) return ESHKOL_DRV_ERR_REVOKED;
    if (handle->generation == 0ULL) {
        return for_release ? ESHKOL_DRV_ERR_DOUBLE_FREE : ESHKOL_DRV_ERR_STALE_HANDLE;
    }
    if (s->state == DRV_SLOT_RELEASED) {
        return for_release ? ESHKOL_DRV_ERR_DOUBLE_FREE : ESHKOL_DRV_ERR_STALE_HANDLE;
    }
    if (s->state != DRV_SLOT_ACTIVE || s->generation != handle->generation ||
        s->cap_id != handle->cap_id) {
        return ESHKOL_DRV_ERR_STALE_HANDLE;
    }
    return ESHKOL_DRV_OK;
}

static eshkol_os_drv_status_t drv_validate_irq(
    const eshkol_os_drv_runtime_t* rt,
    const eshkol_os_irq_lease_t* handle,
    int for_release) {
    if (!rt || !rt->initialized || !handle) return ESHKOL_DRV_ERR_INVALID_ARG;
    if (handle->kind != ESHKOL_DRV_KIND_IRQ_LEASE || handle->slot >= ESHKOL_OS_DRV_MAX_SLOTS) {
        return ESHKOL_DRV_ERR_STALE_HANDLE;
    }
    const eshkol_os_irq_lease_slot_t* s = &rt->irqs[handle->slot];
    if (s->state == DRV_SLOT_REVOKED) return ESHKOL_DRV_ERR_REVOKED;
    if (handle->generation == 0ULL) {
        return for_release ? ESHKOL_DRV_ERR_DOUBLE_FREE : ESHKOL_DRV_ERR_STALE_HANDLE;
    }
    if (s->state == DRV_SLOT_RELEASED) {
        return for_release ? ESHKOL_DRV_ERR_DOUBLE_FREE : ESHKOL_DRV_ERR_STALE_HANDLE;
    }
    if (s->state != DRV_SLOT_ACTIVE || s->generation != handle->generation ||
        s->cap_id != handle->cap_id) {
        return ESHKOL_DRV_ERR_STALE_HANDLE;
    }
    return ESHKOL_DRV_OK;
}

static eshkol_os_drv_status_t drv_validate_dma(
    const eshkol_os_drv_runtime_t* rt,
    const eshkol_os_dma_buffer_t* handle,
    int for_release) {
    if (!rt || !rt->initialized || !handle) return ESHKOL_DRV_ERR_INVALID_ARG;
    if (handle->kind != ESHKOL_DRV_KIND_DMA_BUFFER || handle->slot >= ESHKOL_OS_DRV_MAX_SLOTS) {
        return ESHKOL_DRV_ERR_STALE_HANDLE;
    }
    const eshkol_os_dma_buffer_slot_t* s = &rt->dmas[handle->slot];
    if (s->state == DRV_SLOT_REVOKED) return ESHKOL_DRV_ERR_REVOKED;
    if (handle->generation == 0ULL) {
        return for_release ? ESHKOL_DRV_ERR_DOUBLE_FREE : ESHKOL_DRV_ERR_STALE_HANDLE;
    }
    if (s->state == DRV_SLOT_RELEASED) {
        return for_release ? ESHKOL_DRV_ERR_DOUBLE_FREE : ESHKOL_DRV_ERR_STALE_HANDLE;
    }
    if (s->state != DRV_SLOT_ACTIVE || s->generation != handle->generation ||
        s->cap_id != handle->cap_id) {
        return ESHKOL_DRV_ERR_STALE_HANDLE;
    }
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_device_region_acquire(
    eshkol_os_drv_runtime_t* rt,
    eshkol_cap_id_t cap_id,
    void* mapped_base,
    size_t region_bytes,
    eshkol_os_device_region_t* out_handle) {
    if (!rt || !rt->initialized || cap_id == ESHKOL_CAP_ID_INVALID ||
        !mapped_base || region_bytes < 4u || !out_handle) {
        return ESHKOL_DRV_ERR_INVALID_ARG;
    }
    /* Allocate a fresh monotonic slot without reusing a recently released slot
     * while free slots remain so late double-free checks remain unambiguous. */
    int chosen = -1;
    for (uint32_t i = 0u; i < ESHKOL_OS_DRV_MAX_SLOTS; i++) {
        if (rt->regions[i].state == DRV_SLOT_FREE) { chosen = (int)i; break; }
    }
    if (chosen < 0) return ESHKOL_DRV_ERR_NO_SLOTS;

    uint64_t gen = rt->next_generation++;
    eshkol_os_dev_region_slot_t* s = &rt->regions[chosen];
    s->state = DRV_SLOT_ACTIVE;
    s->generation = gen;
    s->cap_id = cap_id;
    s->mapped_base = (uint8_t*)mapped_base;
    s->region_bytes = region_bytes;

    out_handle->slot = (uint32_t)chosen;
    out_handle->kind = ESHKOL_DRV_KIND_DEVICE_REGION;
    out_handle->generation = gen;
    out_handle->cap_id = cap_id;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_device_region_transfer(
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_device_region_t* src,
    eshkol_os_device_region_t* dst) {
    if (!dst) return ESHKOL_DRV_ERR_INVALID_ARG;
    eshkol_os_drv_status_t st = drv_validate_region(rt, src, 0);
    if (st != ESHKOL_DRV_OK) return st;

    uint64_t new_gen = rt->next_generation++;
    rt->regions[src->slot].generation = new_gen;

    dst->slot = src->slot;
    dst->kind = ESHKOL_DRV_KIND_DEVICE_REGION;
    dst->generation = new_gen;
    dst->cap_id = src->cap_id;

    /* Invalidate source handle token immediately */
    src->generation = 0ULL;
    src->cap_id = ESHKOL_CAP_ID_INVALID;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_device_region_read32(
    const eshkol_os_drv_runtime_t* rt,
    const eshkol_os_device_region_t* handle,
    size_t offset,
    uint32_t* out_val) {
    if (!out_val) return ESHKOL_DRV_ERR_INVALID_ARG;
    eshkol_os_drv_status_t st = drv_validate_region(rt, handle, 0);
    if (st != ESHKOL_DRV_OK) return st;

    const eshkol_os_dev_region_slot_t* s = &rt->regions[handle->slot];
    if ((offset & 3u) != 0u || offset > s->region_bytes || 4u > s->region_bytes - offset) {
        return ESHKOL_DRV_ERR_MMIO_BOUNDS;
    }
    const volatile uint8_t* p = (const volatile uint8_t*)(s->mapped_base + offset);
    uint32_t v = ((uint32_t)p[0]) |
                 ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) |
                 ((uint32_t)p[3] << 24);
    *out_val = v;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_device_region_write32(
    eshkol_os_drv_runtime_t* rt,
    const eshkol_os_device_region_t* handle,
    size_t offset,
    uint32_t val) {
    eshkol_os_drv_status_t st = drv_validate_region(rt, handle, 0);
    if (st != ESHKOL_DRV_OK) return st;

    eshkol_os_dev_region_slot_t* s = &rt->regions[handle->slot];
    if ((offset & 3u) != 0u || offset > s->region_bytes || 4u > s->region_bytes - offset) {
        return ESHKOL_DRV_ERR_MMIO_BOUNDS;
    }
    volatile uint8_t* p = (volatile uint8_t*)(s->mapped_base + offset);
    p[0] = (uint8_t)(val & 0xFFu);
    p[1] = (uint8_t)((val >> 8) & 0xFFu);
    p[2] = (uint8_t)((val >> 16) & 0xFFu);
    p[3] = (uint8_t)((val >> 24) & 0xFFu);
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_device_region_release(
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_device_region_t* handle) {
    eshkol_os_drv_status_t st = drv_validate_region(rt, handle, 1);
    if (st != ESHKOL_DRV_OK) return st;

    eshkol_os_dev_region_slot_t* s = &rt->regions[handle->slot];
    s->state = DRV_SLOT_RELEASED;
    s->mapped_base = NULL;
    s->region_bytes = 0u;
    handle->generation = 0ULL;
    handle->cap_id = ESHKOL_CAP_ID_INVALID;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_irq_lease_acquire(
    eshkol_os_drv_runtime_t* rt,
    eshkol_cap_id_t cap_id,
    uint32_t irq_vector,
    eshkol_os_irq_lease_t* out_handle) {
    if (!rt || !rt->initialized || cap_id == ESHKOL_CAP_ID_INVALID || !out_handle) {
        return ESHKOL_DRV_ERR_INVALID_ARG;
    }
    int chosen = -1;
    for (uint32_t i = 0u; i < ESHKOL_OS_DRV_MAX_SLOTS; i++) {
        if (rt->irqs[i].state == DRV_SLOT_FREE) { chosen = (int)i; break; }
    }
    if (chosen < 0) return ESHKOL_DRV_ERR_NO_SLOTS;

    uint64_t gen = rt->next_generation++;
    eshkol_os_irq_lease_slot_t* s = &rt->irqs[chosen];
    s->state = DRV_SLOT_ACTIVE;
    s->irq_vector = irq_vector;
    s->generation = gen;
    s->cap_id = cap_id;
    s->ack_count = 0ULL;

    out_handle->slot = (uint32_t)chosen;
    out_handle->kind = ESHKOL_DRV_KIND_IRQ_LEASE;
    out_handle->generation = gen;
    out_handle->cap_id = cap_id;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_irq_lease_transfer(
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_irq_lease_t* src,
    eshkol_os_irq_lease_t* dst) {
    if (!dst) return ESHKOL_DRV_ERR_INVALID_ARG;
    eshkol_os_drv_status_t st = drv_validate_irq(rt, src, 0);
    if (st != ESHKOL_DRV_OK) return st;

    uint64_t new_gen = rt->next_generation++;
    rt->irqs[src->slot].generation = new_gen;

    dst->slot = src->slot;
    dst->kind = ESHKOL_DRV_KIND_IRQ_LEASE;
    dst->generation = new_gen;
    dst->cap_id = src->cap_id;

    src->generation = 0ULL;
    src->cap_id = ESHKOL_CAP_ID_INVALID;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_irq_lease_ack(
    eshkol_os_drv_runtime_t* rt,
    const eshkol_os_irq_lease_t* handle) {
    eshkol_os_drv_status_t st = drv_validate_irq(rt, handle, 0);
    if (st != ESHKOL_DRV_OK) return st;

    eshkol_os_irq_lease_slot_t* s = &rt->irqs[handle->slot];
    int64_t rc = drv_syscall(
        rt, ESHKOL_SYS_IPC_COMPLETE, s->cap_id, (uint64_t)s->irq_vector, 1ULL, 0ULL, 0ULL, 0ULL);
    if (rc < 0) {
        if (rc == ESHKOL_OS_ERR_REVOKED) s->state = DRV_SLOT_REVOKED;
        return drv_map_syscall_err(rc);
    }
    s->ack_count++;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_irq_lease_release(
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_irq_lease_t* handle) {
    eshkol_os_drv_status_t st = drv_validate_irq(rt, handle, 1);
    if (st != ESHKOL_DRV_OK) return st;

    rt->irqs[handle->slot].state = DRV_SLOT_RELEASED;
    handle->generation = 0ULL;
    handle->cap_id = ESHKOL_CAP_ID_INVALID;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_dma_buffer_acquire(
    eshkol_os_drv_runtime_t* rt,
    eshkol_cap_id_t space_cap,
    eshkol_cap_id_t dma_cap,
    void* buf_base,
    size_t lease_bytes,
    eshkol_os_dma_buffer_t* out_handle) {
    if (!rt || !rt->initialized || space_cap == ESHKOL_CAP_ID_INVALID ||
        dma_cap == ESHKOL_CAP_ID_INVALID || !buf_base || lease_bytes == 0u || !out_handle) {
        return ESHKOL_DRV_ERR_INVALID_ARG;
    }
    int chosen = -1;
    for (uint32_t i = 0u; i < ESHKOL_OS_DRV_MAX_SLOTS; i++) {
        if (rt->dmas[i].state == DRV_SLOT_FREE) { chosen = (int)i; break; }
    }
    if (chosen < 0) return ESHKOL_DRV_ERR_NO_SLOTS;

    uint64_t pages = ((uint64_t)lease_bytes + rt->page_size - 1ULL) / rt->page_size;
    uint64_t vaddr = (uint64_t)(uintptr_t)buf_base;
    int64_t rc = drv_syscall(
        rt, ESHKOL_SYS_PAGE_MAP, space_cap, vaddr, pages,
        ESHKOL_PAGE_FLAG_READ | ESHKOL_PAGE_FLAG_WRITE | ESHKOL_PAGE_FLAG_ZEROED,
        0ULL, 0ULL);
    if (rc < 0) {
        return drv_map_syscall_err(rc);
    }

    drv_zero(buf_base, lease_bytes);
    uint64_t gen = rt->next_generation++;
    eshkol_os_dma_buffer_slot_t* s = &rt->dmas[chosen];
    s->state = DRV_SLOT_ACTIVE;
    s->generation = gen;
    s->space_cap = space_cap;
    s->cap_id = dma_cap;
    s->buf_base = (uint8_t*)buf_base;
    s->dma_vaddr = vaddr;
    s->lease_bytes = lease_bytes;
    s->mapped_pages = pages;
    rt->live_dma_pages += pages;

    out_handle->slot = (uint32_t)chosen;
    out_handle->kind = ESHKOL_DRV_KIND_DMA_BUFFER;
    out_handle->generation = gen;
    out_handle->cap_id = dma_cap;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_dma_buffer_transfer(
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_dma_buffer_t* src,
    eshkol_os_dma_buffer_t* dst) {
    if (!dst) return ESHKOL_DRV_ERR_INVALID_ARG;
    eshkol_os_drv_status_t st = drv_validate_dma(rt, src, 0);
    if (st != ESHKOL_DRV_OK) return st;

    uint64_t new_gen = rt->next_generation++;
    rt->dmas[src->slot].generation = new_gen;

    dst->slot = src->slot;
    dst->kind = ESHKOL_DRV_KIND_DMA_BUFFER;
    dst->generation = new_gen;
    dst->cap_id = src->cap_id;

    src->generation = 0ULL;
    src->cap_id = ESHKOL_CAP_ID_INVALID;
    return ESHKOL_DRV_OK;
}

static int drv_dma_range_ok(size_t lease_bytes, size_t offset, size_t len) {
    if (offset > lease_bytes) return 0;
    if (len > lease_bytes - offset) return 0;
    return 1;
}

eshkol_os_drv_status_t eshkol_os_dma_write(
    eshkol_os_drv_runtime_t* rt,
    const eshkol_os_dma_buffer_t* handle,
    size_t offset,
    const void* src,
    size_t len) {
    if (!src && len > 0u) return ESHKOL_DRV_ERR_INVALID_ARG;
    eshkol_os_drv_status_t st = drv_validate_dma(rt, handle, 0);
    if (st != ESHKOL_DRV_OK) return st;

    eshkol_os_dma_buffer_slot_t* s = &rt->dmas[handle->slot];
    if (!drv_dma_range_ok(s->lease_bytes, offset, len)) {
        return ESHKOL_DRV_ERR_DMA_BOUNDS;
    }
    if (len > 0u) {
        drv_copy(s->buf_base + offset, src, len);
    }
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_dma_read(
    const eshkol_os_drv_runtime_t* rt,
    const eshkol_os_dma_buffer_t* handle,
    size_t offset,
    void* dst,
    size_t len) {
    if (!dst && len > 0u) return ESHKOL_DRV_ERR_INVALID_ARG;
    eshkol_os_drv_status_t st = drv_validate_dma(rt, handle, 0);
    if (st != ESHKOL_DRV_OK) return st;

    const eshkol_os_dma_buffer_slot_t* s = &rt->dmas[handle->slot];
    if (!drv_dma_range_ok(s->lease_bytes, offset, len)) {
        return ESHKOL_DRV_ERR_DMA_BOUNDS;
    }
    if (len > 0u) {
        drv_copy(dst, s->buf_base + offset, len);
    }
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_dma_buffer_release(
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_dma_buffer_t* handle) {
    eshkol_os_drv_status_t st = drv_validate_dma(rt, handle, 1);
    if (st != ESHKOL_DRV_OK) return st;

    eshkol_os_dma_buffer_slot_t* s = &rt->dmas[handle->slot];
    if (s->mapped_pages > 0u) {
        (void)drv_syscall(
            rt, ESHKOL_SYS_PAGE_UNMAP, s->space_cap, s->dma_vaddr, s->mapped_pages, 0ULL, 0ULL, 0ULL);
        if (s->mapped_pages <= rt->live_dma_pages) {
            rt->live_dma_pages -= s->mapped_pages;
        } else {
            rt->live_dma_pages = 0ULL;
        }
        s->mapped_pages = 0ULL;
    }
    s->state = DRV_SLOT_RELEASED;
    s->buf_base = NULL;
    s->lease_bytes = 0u;
    handle->generation = 0ULL;
    handle->cap_id = ESHKOL_CAP_ID_INVALID;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_drv_revoke_cap(
    eshkol_os_drv_runtime_t* rt,
    eshkol_cap_id_t cap_id) {
    if (!rt || !rt->initialized || cap_id == ESHKOL_CAP_ID_INVALID) {
        return ESHKOL_DRV_ERR_INVALID_ARG;
    }
    int64_t rc = drv_syscall(rt, ESHKOL_SYS_CAP_REVOKE, cap_id, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    if (rc < 0) return drv_map_syscall_err(rc);

    for (uint32_t i = 0u; i < ESHKOL_OS_DRV_MAX_SLOTS; i++) {
        if (rt->regions[i].state == DRV_SLOT_ACTIVE && rt->regions[i].cap_id == cap_id) {
            rt->regions[i].state = DRV_SLOT_REVOKED;
            rt->regions[i].mapped_base = NULL;
            rt->regions[i].region_bytes = 0u;
        }
        if (rt->irqs[i].state == DRV_SLOT_ACTIVE && rt->irqs[i].cap_id == cap_id) {
            rt->irqs[i].state = DRV_SLOT_REVOKED;
        }
        if (rt->dmas[i].state == DRV_SLOT_ACTIVE &&
            (rt->dmas[i].cap_id == cap_id || rt->dmas[i].space_cap == cap_id)) {
            if (rt->dmas[i].mapped_pages > 0u) {
                (void)drv_syscall(
                    rt, ESHKOL_SYS_PAGE_UNMAP, rt->dmas[i].space_cap,
                    rt->dmas[i].dma_vaddr, rt->dmas[i].mapped_pages, 0ULL, 0ULL, 0ULL);
                if (rt->dmas[i].mapped_pages <= rt->live_dma_pages) {
                    rt->live_dma_pages -= rt->dmas[i].mapped_pages;
                } else {
                    rt->live_dma_pages = 0ULL;
                }
                rt->dmas[i].mapped_pages = 0ULL;
            }
            rt->dmas[i].state = DRV_SLOT_REVOKED;
            rt->dmas[i].buf_base = NULL;
            rt->dmas[i].lease_bytes = 0u;
        }
    }
    return ESHKOL_DRV_OK;
}

/* ── VirtIO Split-Virtqueue Implementation ── */

static eshkol_os_drv_status_t vq_validate_handles(const eshkol_os_virtq_t* vq) {
    if (!vq || !vq->initialized || !vq->rt) return ESHKOL_DRV_ERR_INVALID_ARG;
    eshkol_os_drv_status_t st = drv_validate_region(vq->rt, &vq->dev_region, 0);
    if (st != ESHKOL_DRV_OK) return st;
    st = drv_validate_irq(vq->rt, &vq->irq_lease, 0);
    if (st != ESHKOL_DRV_OK) return st;
    st = drv_validate_dma(vq->rt, &vq->dma_buf, 0);
    if (st != ESHKOL_DRV_OK) return st;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_virtq_init(
    eshkol_os_virtq_t* vq,
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_device_region_t dev_region,
    eshkol_os_irq_lease_t irq_lease,
    eshkol_os_dma_buffer_t dma_buf,
    eshkol_cap_id_t notify_cap,
    uint32_t queue_index) {
    if (!vq || !rt || !rt->initialized || notify_cap == ESHKOL_CAP_ID_INVALID) {
        return ESHKOL_DRV_ERR_INVALID_ARG;
    }
    eshkol_os_drv_status_t st = drv_validate_region(rt, &dev_region, 0);
    if (st != ESHKOL_DRV_OK) return st;
    st = drv_validate_irq(rt, &irq_lease, 0);
    if (st != ESHKOL_DRV_OK) return st;
    st = drv_validate_dma(rt, &dma_buf, 0);
    if (st != ESHKOL_DRV_OK) return st;

    drv_zero(vq, sizeof(*vq));
    vq->rt = rt;
    vq->dev_region = dev_region;
    vq->irq_lease = irq_lease;
    vq->dma_buf = dma_buf;
    vq->notify_cap = notify_cap;
    vq->queue_index = queue_index;
    vq->initialized = 1u;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_virtq_submit_chain(
    eshkol_os_virtq_t* vq,
    const eshkol_os_virtq_seg_t* segs,
    uint32_t n_segs) {
    if (!segs || n_segs == 0u || n_segs > ESHKOL_OS_VIRTQ_SIZE) {
        return ESHKOL_DRV_ERR_INVALID_ARG;
    }
    eshkol_os_drv_status_t st = vq_validate_handles(vq);
    if (st != ESHKOL_DRV_OK) return st;
    if (vq->fifo_count >= ESHKOL_OS_VIRTQ_SIZE) return ESHKOL_DRV_ERR_VQ_FULL;

    const eshkol_os_dma_buffer_slot_t* dma = &vq->rt->dmas[vq->dma_buf.slot];

    /* Pass 1: Atomic pre-validation (zero state mutation on any failure) */
    uint32_t chain_mask = 0u;
    for (uint32_t i = 0u; i < n_segs; i++) {
        uint16_t idx = segs[i].desc_idx;
        if (idx >= ESHKOL_OS_VIRTQ_SIZE) {
            return ESHKOL_DRV_ERR_VQ_BAD_DESC;
        }
        uint32_t bit = 1u << idx;
        if ((chain_mask & bit) != 0u || (vq->in_flight_mask & bit) != 0u) {
            return ESHKOL_DRV_ERR_VQ_DUPLICATE_DESC;
        }
        if (!drv_dma_range_ok(dma->lease_bytes, segs[i].dma_offset, (size_t)segs[i].len)) {
            return ESHKOL_DRV_ERR_DMA_BOUNDS;
        }
        chain_mask |= bit;
    }

    /* Pass 2: Populate hardware descriptor table */
    for (uint32_t i = 0u; i < n_segs; i++) {
        uint16_t idx = segs[i].desc_idx;
        uint16_t flags = (uint16_t)(segs[i].flags & (uint16_t)(~ESHKOL_VIRTQ_DESC_F_NEXT));
        uint16_t next = 0u;
        if (i + 1u < n_segs) {
            flags |= ESHKOL_VIRTQ_DESC_F_NEXT;
            next = segs[i + 1u].desc_idx;
        }
        vq->desc[idx].addr = dma->dma_vaddr + (uint64_t)segs[i].dma_offset;
        vq->desc[idx].len = segs[i].len;
        vq->desc[idx].flags = flags;
        vq->desc[idx].next = next;
    }

    /* Publish chain head to avail ring and record expected completion FIFO */
    uint16_t head_id = segs[0].desc_idx;
    vq->in_flight_mask |= chain_mask;
    vq->expected_heads[vq->fifo_tail] = head_id;
    vq->fifo_tail = (uint16_t)((vq->fifo_tail + 1u) % ESHKOL_OS_VIRTQ_SIZE);
    vq->fifo_count++;

    vq->avail.ring[vq->avail.idx % ESHKOL_OS_VIRTQ_SIZE] = head_id;
    vq->avail.idx = (uint16_t)(vq->avail.idx + 1u);
    vq->last_avail_idx = vq->avail.idx;

    /* Ring device notify register via capability MMIO + IPC submit doorbell */
    (void)eshkol_os_device_region_write32(vq->rt, &vq->dev_region, 0u, vq->queue_index);
    int64_t rc = drv_syscall(
        vq->rt, ESHKOL_SYS_IPC_SUBMIT, vq->notify_cap,
        (uint64_t)(uintptr_t)&vq->avail, 1ULL, (uint64_t)vq->queue_index, 0ULL, 0ULL);
    if (rc < 0) return drv_map_syscall_err(rc);
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_virtq_complete_one(
    eshkol_os_virtq_t* vq,
    eshkol_os_virtq_completion_t* out_comp) {
    eshkol_os_drv_status_t st = vq_validate_handles(vq);
    if (st != ESHKOL_DRV_OK) return st;

    if (vq->used.idx == vq->last_used_idx) {
        return ESHKOL_DRV_ERR_VQ_EMPTY;
    }
    const eshkol_virtq_used_elem_t elem = vq->used.ring[vq->last_used_idx % ESHKOL_OS_VIRTQ_SIZE];

    /* Strict monotonic FIFO ordering & active in-flight validation */
    if (vq->fifo_count == 0u ||
        elem.id >= ESHKOL_OS_VIRTQ_SIZE ||
        elem.id != (uint32_t)vq->expected_heads[vq->fifo_head] ||
        (vq->in_flight_mask & (1u << elem.id)) == 0u) {
        return ESHKOL_DRV_ERR_VQ_OUT_OF_ORDER;
    }

    /* Release every descriptor in the completed chain */
    uint16_t cur = (uint16_t)elem.id;
    for (uint32_t step = 0u; step < ESHKOL_OS_VIRTQ_SIZE; step++) {
        uint32_t bit = 1u << cur;
        if ((vq->in_flight_mask & bit) == 0u) {
            return ESHKOL_DRV_ERR_VQ_OUT_OF_ORDER;
        }
        vq->in_flight_mask &= ~bit;
        uint16_t flags = vq->desc[cur].flags;
        uint16_t nxt = vq->desc[cur].next;
        if ((flags & ESHKOL_VIRTQ_DESC_F_NEXT) == 0u) break;
        if (nxt >= ESHKOL_OS_VIRTQ_SIZE) return ESHKOL_DRV_ERR_VQ_BAD_DESC;
        cur = nxt;
    }

    vq->fifo_head = (uint16_t)((vq->fifo_head + 1u) % ESHKOL_OS_VIRTQ_SIZE);
    vq->fifo_count--;
    vq->last_used_idx = (uint16_t)(vq->last_used_idx + 1u);

    st = eshkol_os_irq_lease_ack(vq->rt, &vq->irq_lease);
    if (st != ESHKOL_DRV_OK) return st;

    if (out_comp) {
        out_comp->head_id = elem.id;
        out_comp->bytes_written = elem.len;
    }
    return ESHKOL_DRV_OK;
}

/* ── VirtIO Block & Network Transport Helpers ── */

eshkol_os_drv_status_t eshkol_os_virtio_blk_io(
    eshkol_os_blk_dev_t* blk,
    int is_write,
    uint64_t sector,
    void* buf,
    uint32_t sector_count) {
    if (!blk || !blk->vq || !blk->kick_fn || !buf || sector_count == 0u) {
        return ESHKOL_DRV_ERR_INVALID_ARG;
    }
    if (sector >= blk->total_sectors || (uint64_t)sector_count > blk->total_sectors - sector) {
        return ESHKOL_DRV_ERR_DMA_BOUNDS;
    }
    eshkol_os_virtq_t* vq = blk->vq;
    eshkol_os_drv_status_t st = vq_validate_handles(vq);
    if (st != ESHKOL_DRV_OK) return st;

    size_t payload_bytes = (size_t)sector_count * ESHKOL_OS_VFS_SECTOR_SIZE;
    size_t hdr_off = blk->dma_scratch_offset;
    size_t data_off = hdr_off + sizeof(eshkol_virtio_blk_req_hdr_t);
    size_t status_off = data_off + payload_bytes;

    eshkol_virtio_blk_req_hdr_t hdr;
    hdr.type = is_write ? ESHKOL_VIRTIO_BLK_T_OUT : ESHKOL_VIRTIO_BLK_T_IN;
    hdr.reserved = 0u;
    hdr.sector = sector;

    st = eshkol_os_dma_write(vq->rt, &vq->dma_buf, hdr_off, &hdr, sizeof(hdr));
    if (st != ESHKOL_DRV_OK) return st;
    if (is_write) {
        st = eshkol_os_dma_write(vq->rt, &vq->dma_buf, data_off, buf, payload_bytes);
        if (st != ESHKOL_DRV_OK) return st;
    }
    uint8_t init_status = 0xFFu;
    st = eshkol_os_dma_write(vq->rt, &vq->dma_buf, status_off, &init_status, 1u);
    if (st != ESHKOL_DRV_OK) return st;

    eshkol_os_virtq_seg_t segs[3];
    segs[0].desc_idx = 0u;
    segs[0].flags = 0u;
    segs[0].len = (uint32_t)sizeof(hdr);
    segs[0].dma_offset = hdr_off;

    segs[1].desc_idx = 1u;
    segs[1].flags = is_write ? 0u : ESHKOL_VIRTQ_DESC_F_WRITE;
    segs[1].len = (uint32_t)payload_bytes;
    segs[1].dma_offset = data_off;

    segs[2].desc_idx = 2u;
    segs[2].flags = ESHKOL_VIRTQ_DESC_F_WRITE;
    segs[2].len = 1u;
    segs[2].dma_offset = status_off;

    st = eshkol_os_virtq_submit_chain(vq, segs, 3u);
    if (st != ESHKOL_DRV_OK) return st;

    eshkol_os_dma_buffer_slot_t* dma = &vq->rt->dmas[vq->dma_buf.slot];
    if (blk->kick_fn(vq, dma->buf_base, dma->lease_bytes, blk->dev_ctx) != 0) {
        return ESHKOL_DRV_ERR_DEVICE_IO;
    }

    eshkol_os_virtq_completion_t comp;
    st = eshkol_os_virtq_complete_one(vq, &comp);
    if (st != ESHKOL_DRV_OK) return st;

    uint8_t blk_status = 0xFFu;
    st = eshkol_os_dma_read(vq->rt, &vq->dma_buf, status_off, &blk_status, 1u);
    if (st != ESHKOL_DRV_OK) return st;
    if (blk_status != ESHKOL_VIRTIO_BLK_S_OK) return ESHKOL_DRV_ERR_DEVICE_IO;

    if (!is_write) {
        st = eshkol_os_dma_read(vq->rt, &vq->dma_buf, data_off, buf, payload_bytes);
        if (st != ESHKOL_DRV_OK) return st;
    }
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_virtio_net_xfer(
    eshkol_os_virtq_t* vq,
    eshkol_os_virtio_device_kick_fn_t kick_fn,
    void* dev_ctx,
    int is_tx,
    size_t dma_offset,
    void* frame_buf,
    size_t frame_len,
    size_t* out_len) {
    if (!vq || !kick_fn || !frame_buf || frame_len == 0u) {
        return ESHKOL_DRV_ERR_INVALID_ARG;
    }
    eshkol_os_drv_status_t st = vq_validate_handles(vq);
    if (st != ESHKOL_DRV_OK) return st;

    size_t hdr_off = dma_offset;
    size_t payload_off = hdr_off + sizeof(eshkol_virtio_net_hdr_t);

    eshkol_virtio_net_hdr_t nhdr;
    drv_zero(&nhdr, sizeof(nhdr));
    st = eshkol_os_dma_write(vq->rt, &vq->dma_buf, hdr_off, &nhdr, sizeof(nhdr));
    if (st != ESHKOL_DRV_OK) return st;
    if (is_tx) {
        st = eshkol_os_dma_write(vq->rt, &vq->dma_buf, payload_off, frame_buf, frame_len);
        if (st != ESHKOL_DRV_OK) return st;
    }

    eshkol_os_virtq_seg_t segs[2];
    segs[0].desc_idx = is_tx ? 0u : 2u;
    segs[0].flags = is_tx ? 0u : ESHKOL_VIRTQ_DESC_F_WRITE;
    segs[0].len = (uint32_t)sizeof(nhdr);
    segs[0].dma_offset = hdr_off;

    segs[1].desc_idx = is_tx ? 1u : 3u;
    segs[1].flags = is_tx ? 0u : ESHKOL_VIRTQ_DESC_F_WRITE;
    segs[1].len = (uint32_t)frame_len;
    segs[1].dma_offset = payload_off;

    st = eshkol_os_virtq_submit_chain(vq, segs, 2u);
    if (st != ESHKOL_DRV_OK) return st;

    eshkol_os_dma_buffer_slot_t* dma = &vq->rt->dmas[vq->dma_buf.slot];
    if (kick_fn(vq, dma->buf_base, dma->lease_bytes, dev_ctx) != 0) {
        return ESHKOL_DRV_ERR_DEVICE_IO;
    }

    eshkol_os_virtq_completion_t comp;
    st = eshkol_os_virtq_complete_one(vq, &comp);
    if (st != ESHKOL_DRV_OK) return st;

    size_t rx_bytes = frame_len;
    if (!is_tx) {
        if (comp.bytes_written >= sizeof(nhdr)) {
            size_t avail_payload = (size_t)comp.bytes_written - sizeof(nhdr);
            if (avail_payload < rx_bytes) rx_bytes = avail_payload;
        }
        st = eshkol_os_dma_read(vq->rt, &vq->dma_buf, payload_off, frame_buf, rx_bytes);
        if (st != ESHKOL_DRV_OK) return st;
    }
    if (out_len) *out_len = rx_bytes;
    return ESHKOL_DRV_OK;
}

/* ── Atomic Content-Addressed VFS (Dual-Slot A/B Superblock + CoW Data) ── */

typedef struct vfs_sb_decoded {
    uint64_t generation;
    uint32_t active_slot;
    uint32_t entry_count;
    uint32_t next_free_sector;
    uint32_t next_entry_id;
    uint32_t index_crc32c;
} vfs_sb_decoded_t;

static void vfs_encode_sb(const vfs_sb_decoded_t* sb, uint8_t sector[ESHKOL_OS_VFS_SECTOR_SIZE]) {
    drv_zero(sector, ESHKOL_OS_VFS_SECTOR_SIZE);
    drv_wr_u32_le(sector + 0u, VFS_SB_MAGIC);
    drv_wr_u32_le(sector + 4u, VFS_SB_VERSION);
    drv_wr_u64_le(sector + 8u, sb->generation);
    drv_wr_u32_le(sector + 16u, sb->active_slot);
    drv_wr_u32_le(sector + 20u, sb->entry_count);
    drv_wr_u32_le(sector + 24u, sb->next_free_sector);
    drv_wr_u32_le(sector + 28u, sb->next_entry_id);
    drv_wr_u32_le(sector + 32u, sb->index_crc32c);
    uint32_t crc = drv_crc32c(sector, ESHKOL_OS_VFS_SECTOR_SIZE - 4u);
    drv_wr_u32_le(sector + ESHKOL_OS_VFS_SECTOR_SIZE - 4u, crc);
}

static int vfs_decode_sb(const uint8_t sector[ESHKOL_OS_VFS_SECTOR_SIZE],
                         uint32_t expected_slot,
                         vfs_sb_decoded_t* out) {
    if (drv_rd_u32_le(sector + 0u) != VFS_SB_MAGIC) return 0;
    if (drv_rd_u32_le(sector + 4u) != VFS_SB_VERSION) return 0;
    uint32_t stored_crc = drv_rd_u32_le(sector + ESHKOL_OS_VFS_SECTOR_SIZE - 4u);
    uint32_t calc_crc = drv_crc32c(sector, ESHKOL_OS_VFS_SECTOR_SIZE - 4u);
    if (stored_crc != calc_crc) return 0;

    uint32_t slot = drv_rd_u32_le(sector + 16u);
    if (slot != expected_slot) return 0;

    out->generation = drv_rd_u64_le(sector + 8u);
    out->active_slot = slot;
    out->entry_count = drv_rd_u32_le(sector + 20u);
    out->next_free_sector = drv_rd_u32_le(sector + 24u);
    out->next_entry_id = drv_rd_u32_le(sector + 28u);
    out->index_crc32c = drv_rd_u32_le(sector + 32u);
    if (out->generation == 0ULL || out->entry_count > ESHKOL_OS_VFS_MAX_ENTRIES ||
        out->next_free_sector < VFS_FIRST_DATA_SECTOR) {
        return 0;
    }
    return 1;
}

static uint32_t vfs_index_sector_for_slot(uint32_t slot) {
    return (slot == 0u) ? VFS_SLOT_A_INDEX_SECTOR : VFS_SLOT_B_INDEX_SECTOR;
}

static eshkol_os_drv_status_t vfs_write_index_slot(
    eshkol_os_blk_dev_t* blk,
    uint32_t slot,
    const eshkol_os_vfs_entry_t entries[ESHKOL_OS_VFS_MAX_ENTRIES],
    uint32_t* out_crc) {
    const uint8_t* raw = (const uint8_t*)entries;
    *out_crc = drv_crc32c(raw, VFS_INDEX_BYTES);
    uint32_t base_sec = vfs_index_sector_for_slot(slot);
    uint8_t sec_buf[ESHKOL_OS_VFS_SECTOR_SIZE];
    for (uint32_t i = 0u; i < VFS_INDEX_SECTORS; i++) {
        drv_copy(sec_buf, raw + (size_t)i * ESHKOL_OS_VFS_SECTOR_SIZE, ESHKOL_OS_VFS_SECTOR_SIZE);
        eshkol_os_drv_status_t st = eshkol_os_virtio_blk_io(blk, 1, base_sec + i, sec_buf, 1u);
        if (st != ESHKOL_DRV_OK) return st;
    }
    return ESHKOL_DRV_OK;
}

static eshkol_os_drv_status_t vfs_read_index_slot(
    eshkol_os_blk_dev_t* blk,
    uint32_t slot,
    uint32_t expected_crc,
    eshkol_os_vfs_entry_t out_entries[ESHKOL_OS_VFS_MAX_ENTRIES]) {
    uint8_t* raw = (uint8_t*)out_entries;
    uint32_t base_sec = vfs_index_sector_for_slot(slot);
    uint8_t sec_buf[ESHKOL_OS_VFS_SECTOR_SIZE];
    for (uint32_t i = 0u; i < VFS_INDEX_SECTORS; i++) {
        eshkol_os_drv_status_t st = eshkol_os_virtio_blk_io(blk, 0, base_sec + i, sec_buf, 1u);
        if (st != ESHKOL_DRV_OK) return st;
        drv_copy(raw + (size_t)i * ESHKOL_OS_VFS_SECTOR_SIZE, sec_buf, ESHKOL_OS_VFS_SECTOR_SIZE);
    }
    uint32_t calc_crc = drv_crc32c(raw, VFS_INDEX_BYTES);
    if (calc_crc != expected_crc) return ESHKOL_DRV_ERR_VFS_CORRUPT;
    return ESHKOL_DRV_OK;
}

static eshkol_os_drv_status_t vfs_commit_staged_index(
    eshkol_os_cas_vfs_t* vfs,
    const eshkol_os_vfs_entry_t staged[ESHKOL_OS_VFS_MAX_ENTRIES],
    uint32_t new_entry_count,
    uint32_t new_next_free_sector,
    uint32_t new_next_entry_id) {
    uint32_t next_slot = vfs->active_slot ^ 1u;
    uint32_t idx_crc = 0u;
    eshkol_os_drv_status_t st = vfs_write_index_slot(vfs->blk, next_slot, staged, &idx_crc);
    if (st != ESHKOL_DRV_OK) return st;

    vfs_sb_decoded_t sb;
    sb.generation = vfs->generation + 1ULL;
    sb.active_slot = next_slot;
    sb.entry_count = new_entry_count;
    sb.next_free_sector = new_next_free_sector;
    sb.next_entry_id = new_next_entry_id;
    sb.index_crc32c = idx_crc;

    uint8_t sb_sec[ESHKOL_OS_VFS_SECTOR_SIZE];
    vfs_encode_sb(&sb, sb_sec);

    /* Single-sector atomic superblock flip to inactive slot (sector 0 or 1) */
    st = eshkol_os_virtio_blk_io(vfs->blk, 1, (uint64_t)next_slot, sb_sec, 1u);
    if (st != ESHKOL_DRV_OK) return st;

    /* Commit succeeded on storage: publish into live in-memory VFS state */
    drv_copy(vfs->entries, staged, sizeof(vfs->entries));
    vfs->generation = sb.generation;
    vfs->active_slot = next_slot;
    vfs->entry_count = new_entry_count;
    vfs->next_free_sector = new_next_free_sector;
    vfs->next_entry_id = new_next_entry_id;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_cas_vfs_format(
    eshkol_os_cas_vfs_t* vfs,
    eshkol_os_blk_dev_t* blk) {
    if (!vfs || !blk || blk->total_sectors <= VFS_FIRST_DATA_SECTOR + 4u) {
        return ESHKOL_DRV_ERR_INVALID_ARG;
    }
    drv_zero(vfs, sizeof(*vfs));
    vfs->blk = blk;

    /* Initialize empty root directory (entry_id = 0) and zero slot B superblock */
    uint8_t zero_sec[ESHKOL_OS_VFS_SECTOR_SIZE];
    drv_zero(zero_sec, sizeof(zero_sec));
    eshkol_os_drv_status_t st = eshkol_os_virtio_blk_io(blk, 1, 1ULL, zero_sec, 1u);
    if (st != ESHKOL_DRV_OK) return st;

    uint32_t idx_crc = 0u;
    st = vfs_write_index_slot(blk, 0u, vfs->entries, &idx_crc);
    if (st != ESHKOL_DRV_OK) return st;

    vfs_sb_decoded_t sb;
    sb.generation = 1ULL;
    sb.active_slot = 0u;
    sb.entry_count = 0u;
    sb.next_free_sector = VFS_FIRST_DATA_SECTOR;
    sb.next_entry_id = 1u;
    sb.index_crc32c = idx_crc;

    uint8_t sb_sec[ESHKOL_OS_VFS_SECTOR_SIZE];
    vfs_encode_sb(&sb, sb_sec);
    st = eshkol_os_virtio_blk_io(blk, 1, 0ULL, sb_sec, 1u);
    if (st != ESHKOL_DRV_OK) return st;

    vfs->generation = 1ULL;
    vfs->active_slot = 0u;
    vfs->entry_count = 0u;
    vfs->next_free_sector = VFS_FIRST_DATA_SECTOR;
    vfs->next_entry_id = 1u;
    vfs->mounted = 1u;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_cas_vfs_mount(
    eshkol_os_cas_vfs_t* vfs,
    eshkol_os_blk_dev_t* blk) {
    if (!vfs || !blk || blk->total_sectors <= VFS_FIRST_DATA_SECTOR) {
        return ESHKOL_DRV_ERR_INVALID_ARG;
    }
    drv_zero(vfs, sizeof(*vfs));
    vfs->blk = blk;

    uint8_t sb_raw[2][ESHKOL_OS_VFS_SECTOR_SIZE];
    vfs_sb_decoded_t sb[2];
    int valid[2] = {0, 0};
    eshkol_os_vfs_entry_t cand_entries[2][ESHKOL_OS_VFS_MAX_ENTRIES];

    for (uint32_t slot = 0u; slot < 2u; slot++) {
        if (eshkol_os_virtio_blk_io(blk, 0, (uint64_t)slot, sb_raw[slot], 1u) != ESHKOL_DRV_OK) {
            continue;
        }
        if (!vfs_decode_sb(sb_raw[slot], slot, &sb[slot])) continue;
        if (sb[slot].next_free_sector > blk->total_sectors) continue;
        if (vfs_read_index_slot(blk, slot, sb[slot].index_crc32c, cand_entries[slot]) == ESHKOL_DRV_OK) {
            valid[slot] = 1;
        }
    }

    int chosen = -1;
    if (valid[0] && valid[1]) {
        chosen = (sb[1].generation > sb[0].generation) ? 1 : 0;
    } else if (valid[0]) {
        chosen = 0;
    } else if (valid[1]) {
        chosen = 1;
    } else {
        return ESHKOL_DRV_ERR_VFS_CORRUPT;
    }

    drv_copy(vfs->entries, cand_entries[chosen], sizeof(vfs->entries));
    vfs->generation = sb[chosen].generation;
    vfs->active_slot = (uint32_t)chosen;
    vfs->entry_count = sb[chosen].entry_count;
    vfs->next_free_sector = sb[chosen].next_free_sector;
    vfs->next_entry_id = sb[chosen].next_entry_id;
    vfs->mounted = 1u;
    return ESHKOL_DRV_OK;
}

static int vfs_parent_valid(const eshkol_os_cas_vfs_t* vfs, uint32_t parent_id) {
    if (parent_id == 0u) return 1; /* root directory */
    for (uint32_t i = 0u; i < ESHKOL_OS_VFS_MAX_ENTRIES; i++) {
        if (vfs->entries[i].active &&
            vfs->entries[i].entry_id == parent_id &&
            vfs->entries[i].kind == ESHKOL_VFS_KIND_DIR) {
            return 1;
        }
    }
    return 0;
}

eshkol_os_drv_status_t eshkol_os_cas_vfs_mkdir(
    eshkol_os_cas_vfs_t* vfs,
    const char* name,
    uint32_t parent_id,
    uint32_t* out_entry_id) {
    if (!vfs || !vfs->mounted || !name) return ESHKOL_DRV_ERR_INVALID_ARG;
    size_t nlen = drv_strlen(name);
    if (nlen == 0u || nlen > ESHKOL_OS_VFS_MAX_NAME) return ESHKOL_DRV_ERR_INVALID_ARG;
    if (!vfs_parent_valid(vfs, parent_id)) return ESHKOL_DRV_ERR_VFS_NOT_FOUND;

    int free_slot = -1;
    for (uint32_t i = 0u; i < ESHKOL_OS_VFS_MAX_ENTRIES; i++) {
        if (vfs->entries[i].active) {
            if (vfs->entries[i].parent_id == parent_id && drv_streq(vfs->entries[i].name, name)) {
                return ESHKOL_DRV_ERR_VFS_EXISTS;
            }
        } else if (free_slot < 0) {
            free_slot = (int)i;
        }
    }
    if (free_slot < 0) return ESHKOL_DRV_ERR_VFS_NO_SPACE;

    eshkol_os_vfs_entry_t staged[ESHKOL_OS_VFS_MAX_ENTRIES];
    drv_copy(staged, vfs->entries, sizeof(staged));

    uint32_t new_id = vfs->next_entry_id;
    eshkol_os_vfs_entry_t* e = &staged[free_slot];
    drv_zero(e, sizeof(*e));
    drv_copy(e->name, name, nlen);
    e->entry_id = new_id;
    e->parent_id = parent_id;
    e->kind = ESHKOL_VFS_KIND_DIR;
    e->active = 1u;

    eshkol_os_drv_status_t st = vfs_commit_staged_index(
        vfs, staged, vfs->entry_count + 1u, vfs->next_free_sector, new_id + 1u);
    if (st != ESHKOL_DRV_OK) return st;
    if (out_entry_id) *out_entry_id = new_id;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_cas_vfs_put_file(
    eshkol_os_cas_vfs_t* vfs,
    const char* name,
    uint32_t parent_id,
    const void* data,
    size_t len,
    uint8_t out_sha256[ESHKOL_OS_SHA256_BYTES]) {
    if (!vfs || !vfs->mounted || !name || (!data && len > 0u)) {
        return ESHKOL_DRV_ERR_INVALID_ARG;
    }
    size_t nlen = drv_strlen(name);
    if (nlen == 0u || nlen > ESHKOL_OS_VFS_MAX_NAME) return ESHKOL_DRV_ERR_INVALID_ARG;
    if (!vfs_parent_valid(vfs, parent_id)) return ESHKOL_DRV_ERR_VFS_NOT_FOUND;

    uint8_t digest[ESHKOL_OS_SHA256_BYTES];
    eshkol_os_sha256(data, len, digest);

    /* Find existing entry (atomic overwrite) or free slot */
    int target_slot = -1;
    int existing_idx = -1;
    for (uint32_t i = 0u; i < ESHKOL_OS_VFS_MAX_ENTRIES; i++) {
        if (vfs->entries[i].active) {
            if (vfs->entries[i].parent_id == parent_id && drv_streq(vfs->entries[i].name, name)) {
                if (vfs->entries[i].kind != ESHKOL_VFS_KIND_FILE) {
                    return ESHKOL_DRV_ERR_VFS_EXISTS;
                }
                existing_idx = (int)i;
                break;
            }
        } else if (target_slot < 0) {
            target_slot = (int)i;
        }
    }
    if (existing_idx >= 0) {
        target_slot = existing_idx;
    } else if (target_slot < 0) {
        return ESHKOL_DRV_ERR_VFS_NO_SPACE;
    }

    uint32_t sec_needed = (len == 0u)
                              ? 0u
                              : (uint32_t)((len + ESHKOL_OS_VFS_SECTOR_SIZE - 1u) /
                                           ESHKOL_OS_VFS_SECTOR_SIZE);
    if ((uint64_t)vfs->next_free_sector + (uint64_t)sec_needed > vfs->blk->total_sectors) {
        return ESHKOL_DRV_ERR_VFS_NO_SPACE;
    }

    /* Phase 1: Stage payload into fresh copy-on-write sectors (never overwrites committed data) */
    uint32_t first_sec = vfs->next_free_sector;
    const uint8_t* bytes = (const uint8_t*)data;
    uint8_t sec_buf[ESHKOL_OS_VFS_SECTOR_SIZE];
    for (uint32_t i = 0u; i < sec_needed; i++) {
        drv_zero(sec_buf, sizeof(sec_buf));
        size_t off = (size_t)i * ESHKOL_OS_VFS_SECTOR_SIZE;
        size_t chunk = (len - off > ESHKOL_OS_VFS_SECTOR_SIZE)
                           ? ESHKOL_OS_VFS_SECTOR_SIZE
                           : (len - off);
        if (chunk > 0u) drv_copy(sec_buf, bytes + off, chunk);
        eshkol_os_drv_status_t st = eshkol_os_virtio_blk_io(
            vfs->blk, 1, (uint64_t)(first_sec + i), sec_buf, 1u);
        if (st != ESHKOL_DRV_OK) return st;
    }

    /* Phase 2: Read-back digest verification before flipping superblock */
    drv_sha256_ctx_t verify_ctx;
    sha256_init(&verify_ctx);
    size_t remaining = len;
    for (uint32_t i = 0u; i < sec_needed; i++) {
        eshkol_os_drv_status_t st = eshkol_os_virtio_blk_io(
            vfs->blk, 0, (uint64_t)(first_sec + i), sec_buf, 1u);
        if (st != ESHKOL_DRV_OK) return st;
        size_t take = remaining > ESHKOL_OS_VFS_SECTOR_SIZE ? ESHKOL_OS_VFS_SECTOR_SIZE : remaining;
        if (take > 0u) {
            sha256_update(&verify_ctx, sec_buf, take);
            remaining -= take;
        }
    }
    uint8_t readback_digest[ESHKOL_OS_SHA256_BYTES];
    sha256_final(&verify_ctx, readback_digest);
    if (!drv_memeq(digest, readback_digest, ESHKOL_OS_SHA256_BYTES)) {
        return ESHKOL_DRV_ERR_VFS_CORRUPT;
    }

    /* Phase 3: Stage updated index to inactive slot + atomic single-sector superblock flip */
    eshkol_os_vfs_entry_t staged[ESHKOL_OS_VFS_MAX_ENTRIES];
    drv_copy(staged, vfs->entries, sizeof(staged));

    uint32_t entry_id = (existing_idx >= 0) ? staged[target_slot].entry_id : vfs->next_entry_id;
    uint32_t new_next_id = (existing_idx >= 0) ? vfs->next_entry_id : (vfs->next_entry_id + 1u);
    uint32_t new_count = (existing_idx >= 0) ? vfs->entry_count : (vfs->entry_count + 1u);

    eshkol_os_vfs_entry_t* e = &staged[target_slot];
    drv_zero(e, sizeof(*e));
    drv_copy(e->name, name, nlen);
    e->entry_id = entry_id;
    e->parent_id = parent_id;
    e->kind = ESHKOL_VFS_KIND_FILE;
    e->size_bytes = (uint32_t)len;
    e->first_sector = first_sec;
    e->sector_count = sec_needed;
    e->active = 1u;
    drv_copy(e->sha256, digest, ESHKOL_OS_SHA256_BYTES);

    eshkol_os_drv_status_t st = vfs_commit_staged_index(
        vfs, staged, new_count, first_sec + sec_needed, new_next_id);
    if (st != ESHKOL_DRV_OK) return st;

    if (out_sha256) drv_copy(out_sha256, digest, ESHKOL_OS_SHA256_BYTES);
    return ESHKOL_DRV_OK;
}

static eshkol_os_drv_status_t vfs_read_entry_verified(
    const eshkol_os_cas_vfs_t* vfs,
    const eshkol_os_vfs_entry_t* e,
    void* out_buf,
    size_t out_cap,
    size_t* out_len) {
    if (e->size_bytes > out_cap) return ESHKOL_DRV_ERR_VFS_BUF_TOO_SMALL;

    drv_sha256_ctx_t verify_ctx;
    sha256_init(&verify_ctx);
    uint8_t sec_buf[ESHKOL_OS_VFS_SECTOR_SIZE];
    uint8_t* dst = (uint8_t*)out_buf;
    size_t remaining = e->size_bytes;
    size_t written = 0u;

    for (uint32_t i = 0u; i < e->sector_count; i++) {
        eshkol_os_drv_status_t st = eshkol_os_virtio_blk_io(
            vfs->blk, 0, (uint64_t)(e->first_sector + i), sec_buf, 1u);
        if (st != ESHKOL_DRV_OK) return st;
        size_t take = remaining > ESHKOL_OS_VFS_SECTOR_SIZE ? ESHKOL_OS_VFS_SECTOR_SIZE : remaining;
        if (take > 0u) {
            if (dst) drv_copy(dst + written, sec_buf, take);
            sha256_update(&verify_ctx, sec_buf, take);
            written += take;
            remaining -= take;
        }
    }

    uint8_t calc[ESHKOL_OS_SHA256_BYTES];
    sha256_final(&verify_ctx, calc);
    if (!drv_memeq(calc, e->sha256, ESHKOL_OS_SHA256_BYTES)) {
        return ESHKOL_DRV_ERR_VFS_CORRUPT;
    }
    if (out_len) *out_len = e->size_bytes;
    return ESHKOL_DRV_OK;
}

eshkol_os_drv_status_t eshkol_os_cas_vfs_read_file(
    const eshkol_os_cas_vfs_t* vfs,
    const char* name,
    uint32_t parent_id,
    void* out_buf,
    size_t out_cap,
    size_t* out_len,
    uint8_t out_sha256[ESHKOL_OS_SHA256_BYTES]) {
    if (!vfs || !vfs->mounted || !name) return ESHKOL_DRV_ERR_INVALID_ARG;
    for (uint32_t i = 0u; i < ESHKOL_OS_VFS_MAX_ENTRIES; i++) {
        const eshkol_os_vfs_entry_t* e = &vfs->entries[i];
        if (e->active && e->kind == ESHKOL_VFS_KIND_FILE &&
            e->parent_id == parent_id && drv_streq(e->name, name)) {
            eshkol_os_drv_status_t st = vfs_read_entry_verified(vfs, e, out_buf, out_cap, out_len);
            if (st == ESHKOL_DRV_OK && out_sha256) {
                drv_copy(out_sha256, e->sha256, ESHKOL_OS_SHA256_BYTES);
            }
            return st;
        }
    }
    return ESHKOL_DRV_ERR_VFS_NOT_FOUND;
}

eshkol_os_drv_status_t eshkol_os_cas_vfs_read_by_hash(
    const eshkol_os_cas_vfs_t* vfs,
    const uint8_t sha256[ESHKOL_OS_SHA256_BYTES],
    void* out_buf,
    size_t out_cap,
    size_t* out_len) {
    if (!vfs || !vfs->mounted || !sha256) return ESHKOL_DRV_ERR_INVALID_ARG;
    for (uint32_t i = 0u; i < ESHKOL_OS_VFS_MAX_ENTRIES; i++) {
        const eshkol_os_vfs_entry_t* e = &vfs->entries[i];
        if (e->active && e->kind == ESHKOL_VFS_KIND_FILE &&
            drv_memeq(e->sha256, sha256, ESHKOL_OS_SHA256_BYTES)) {
            return vfs_read_entry_verified(vfs, e, out_buf, out_cap, out_len);
        }
    }
    return ESHKOL_DRV_ERR_VFS_NOT_FOUND;
}

const char* eshkol_os_drv_status_name(eshkol_os_drv_status_t status) {
    switch (status) {
    case ESHKOL_DRV_OK:                    return "ESHKOL_DRV_OK";
    case ESHKOL_DRV_ERR_INVALID_ARG:       return "ESHKOL_DRV_ERR_INVALID_ARG";
    case ESHKOL_DRV_ERR_NO_SLOTS:          return "ESHKOL_DRV_ERR_NO_SLOTS";
    case ESHKOL_DRV_ERR_STALE_HANDLE:      return "ESHKOL_DRV_ERR_STALE_HANDLE";
    case ESHKOL_DRV_ERR_DOUBLE_FREE:       return "ESHKOL_DRV_ERR_DOUBLE_FREE";
    case ESHKOL_DRV_ERR_REVOKED:           return "ESHKOL_DRV_ERR_REVOKED";
    case ESHKOL_DRV_ERR_DMA_BOUNDS:        return "ESHKOL_DRV_ERR_DMA_BOUNDS";
    case ESHKOL_DRV_ERR_MMIO_BOUNDS:       return "ESHKOL_DRV_ERR_MMIO_BOUNDS";
    case ESHKOL_DRV_ERR_SYSCALL_FAILED:    return "ESHKOL_DRV_ERR_SYSCALL_FAILED";
    case ESHKOL_DRV_ERR_VQ_FULL:           return "ESHKOL_DRV_ERR_VQ_FULL";
    case ESHKOL_DRV_ERR_VQ_EMPTY:          return "ESHKOL_DRV_ERR_VQ_EMPTY";
    case ESHKOL_DRV_ERR_VQ_BAD_DESC:       return "ESHKOL_DRV_ERR_VQ_BAD_DESC";
    case ESHKOL_DRV_ERR_VQ_DUPLICATE_DESC: return "ESHKOL_DRV_ERR_VQ_DUPLICATE_DESC";
    case ESHKOL_DRV_ERR_VQ_OUT_OF_ORDER:   return "ESHKOL_DRV_ERR_VQ_OUT_OF_ORDER";
    case ESHKOL_DRV_ERR_DEVICE_IO:         return "ESHKOL_DRV_ERR_DEVICE_IO";
    case ESHKOL_DRV_ERR_VFS_CORRUPT:       return "ESHKOL_DRV_ERR_VFS_CORRUPT";
    case ESHKOL_DRV_ERR_VFS_NOT_FOUND:     return "ESHKOL_DRV_ERR_VFS_NOT_FOUND";
    case ESHKOL_DRV_ERR_VFS_NO_SPACE:      return "ESHKOL_DRV_ERR_VFS_NO_SPACE";
    case ESHKOL_DRV_ERR_VFS_EXISTS:        return "ESHKOL_DRV_ERR_VFS_EXISTS";
    case ESHKOL_DRV_ERR_VFS_BUF_TOO_SMALL: return "ESHKOL_DRV_ERR_VFS_BUF_TOO_SMALL";
    }
    return "ESHKOL_DRV_UNKNOWN";
}
