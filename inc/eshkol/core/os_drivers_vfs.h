/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K4: Userspace OALR isolated driver framework, VirtIO
 * split-virtqueue block/net driver, and atomic content-addressed VFS ABI.
 *
 * Pure C11/C++17 freestanding userspace library operating strictly over
 * capability handles (`eshkol_cap_id_t`) and `eshkol_os_syscall6` without
 * raw physical frame access, LLVM, POSIX/libc, or dynamic heap allocation.
 *
 * Design invariants:
 *   1. Linear single-ownership handles: `DeviceRegion`, `IrqLease`, and
 *      `DmaBuffer` carry monotonic generation tokens; transfer invalidates
 *      the prior handle, double-free fails closed with
 *      `ESHKOL_DRV_ERR_DOUBLE_FREE`, and post-revocation access fails closed
 *      with `ESHKOL_DRV_ERR_REVOKED`.
 *   2. Strict DMA lease bounds: every DMA transfer and VirtIO descriptor
 *      segment enforces overflow-safe `[offset, offset + len] <= lease_bytes`.
 *   3. VirtIO split-virtqueue discipline: matches the U2 hardware layout
 *      (`virtq_desc`, `virtq_avail`, `virtq_used`), rejects duplicate or
 *      already-in-flight descriptors (`ESHKOL_DRV_ERR_VQ_DUPLICATE_DESC`),
 *      and rejects out-of-order or unsubmitted completions
 *      (`ESHKOL_DRV_ERR_VQ_OUT_OF_ORDER`).
 *   4. Atomic content-addressed VFS: SHA-256 content addressing over dual-slot
 *      A/B CRC32C superblocks and copy-on-write data sectors ensures zero
 *      partial or uncommitted state across mid-write power loss.
 */

#ifndef ESHKOL_CORE_OS_DRIVERS_VFS_H
#define ESHKOL_CORE_OS_DRIVERS_VFS_H

#include <stddef.h>
#include <stdint.h>

#include <eshkol/core/os_syscall_abi.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum number of concurrent linear handle slots per resource class.
 */
#define ESHKOL_OS_DRV_MAX_SLOTS 32u

/**
 * @brief Number of descriptors per VirtIO split virtqueue (matches U2 layout).
 */
#define ESHKOL_OS_VIRTQ_SIZE 16u

/**
 * @brief VirtIO descriptor flag: buffer continues via `next` field.
 */
#define ESHKOL_VIRTQ_DESC_F_NEXT 1u

/**
 * @brief VirtIO descriptor flag: buffer is device-writable (otherwise read-only).
 */
#define ESHKOL_VIRTQ_DESC_F_WRITE 2u

/**
 * @brief VirtIO descriptor flag: buffer contains a table of indirect descriptors.
 */
#define ESHKOL_VIRTQ_DESC_F_INDIRECT 4u

/**
 * @brief VirtIO block request type: read sectors from device.
 */
#define ESHKOL_VIRTIO_BLK_T_IN 0u

/**
 * @brief VirtIO block request type: write sectors to device.
 */
#define ESHKOL_VIRTIO_BLK_T_OUT 1u

/**
 * @brief VirtIO block request status byte: completed cleanly.
 */
#define ESHKOL_VIRTIO_BLK_S_OK 0u

/**
 * @brief Sector size in bytes for VirtIO block and CAS VFS storage.
 */
#define ESHKOL_OS_VFS_SECTOR_SIZE 512u

/**
 * @brief Maximum number of committed directory and file entries in one CAS VFS.
 */
#define ESHKOL_OS_VFS_MAX_ENTRIES 32u

/**
 * @brief Maximum byte length of a VFS entry path (excluding trailing NUL).
 */
#define ESHKOL_OS_VFS_MAX_NAME 31u

/**
 * @brief Byte length of a SHA-256 content digest.
 */
#define ESHKOL_OS_SHA256_BYTES 32u

/**
 * @brief Function pointer type matching `eshkol_os_syscall6` from `os_syscall_abi.h`.
 */
typedef int64_t (*eshkol_os_syscall6_fn_t)(uint64_t sysno,
                                           uint64_t arg0,
                                           uint64_t arg1,
                                           uint64_t arg2,
                                           uint64_t arg3,
                                           uint64_t arg4,
                                           uint64_t arg5);

/**
 * @brief Explicit status codes returned by the Phase K4 driver and VFS subsystem.
 */
typedef enum eshkol_os_drv_status {
    ESHKOL_DRV_OK                   = 0,
    ESHKOL_DRV_ERR_INVALID_ARG      = 1,
    ESHKOL_DRV_ERR_NO_SLOTS         = 2,
    ESHKOL_DRV_ERR_STALE_HANDLE     = 3,
    ESHKOL_DRV_ERR_DOUBLE_FREE      = 4,
    ESHKOL_DRV_ERR_REVOKED          = 5,
    ESHKOL_DRV_ERR_DMA_BOUNDS       = 6,
    ESHKOL_DRV_ERR_MMIO_BOUNDS      = 7,
    ESHKOL_DRV_ERR_SYSCALL_FAILED   = 8,
    ESHKOL_DRV_ERR_VQ_FULL          = 9,
    ESHKOL_DRV_ERR_VQ_EMPTY         = 10,
    ESHKOL_DRV_ERR_VQ_BAD_DESC      = 11,
    ESHKOL_DRV_ERR_VQ_DUPLICATE_DESC = 12,
    ESHKOL_DRV_ERR_VQ_OUT_OF_ORDER  = 13,
    ESHKOL_DRV_ERR_DEVICE_IO        = 14,
    ESHKOL_DRV_ERR_VFS_CORRUPT      = 15,
    ESHKOL_DRV_ERR_VFS_NOT_FOUND    = 16,
    ESHKOL_DRV_ERR_VFS_NO_SPACE     = 17,
    ESHKOL_DRV_ERR_VFS_EXISTS       = 18,
    ESHKOL_DRV_ERR_VFS_BUF_TOO_SMALL = 19
} eshkol_os_drv_status_t;

/**
 * @brief Resource class tag encoded inside linear driver handles.
 */
typedef enum eshkol_os_drv_handle_kind {
    ESHKOL_DRV_KIND_NONE          = 0,
    ESHKOL_DRV_KIND_DEVICE_REGION = 1,
    ESHKOL_DRV_KIND_IRQ_LEASE     = 2,
    ESHKOL_DRV_KIND_DMA_BUFFER    = 3
} eshkol_os_drv_handle_kind_t;

/**
 * @brief Linear single-ownership handle for a capability-mapped MMIO device region.
 */
typedef struct eshkol_os_device_region {
    uint32_t slot;
    uint32_t kind;
    uint64_t generation;
    eshkol_cap_id_t cap_id;
} eshkol_os_device_region_t;

/**
 * @brief Linear single-ownership handle for a capability-bound IRQ delivery lease.
 */
typedef struct eshkol_os_irq_lease {
    uint32_t slot;
    uint32_t kind;
    uint64_t generation;
    eshkol_cap_id_t cap_id;
} eshkol_os_irq_lease_t;

/**
 * @brief Linear single-ownership handle for a bounded capability-mapped DMA buffer.
 */
typedef struct eshkol_os_dma_buffer {
    uint32_t slot;
    uint32_t kind;
    uint64_t generation;
    eshkol_cap_id_t cap_id;
} eshkol_os_dma_buffer_t;

/**
 * @brief Internal slot record for one linear device MMIO region.
 */
typedef struct eshkol_os_dev_region_slot {
    uint32_t state;
    uint32_t reserved;
    uint64_t generation;
    eshkol_cap_id_t cap_id;
    uint8_t* mapped_base;
    size_t region_bytes;
} eshkol_os_dev_region_slot_t;

/**
 * @brief Internal slot record for one linear IRQ lease.
 */
typedef struct eshkol_os_irq_lease_slot {
    uint32_t state;
    uint32_t irq_vector;
    uint64_t generation;
    eshkol_cap_id_t cap_id;
    uint64_t ack_count;
} eshkol_os_irq_lease_slot_t;

/**
 * @brief Internal slot record for one linear DMA lease buffer.
 */
typedef struct eshkol_os_dma_buffer_slot {
    uint32_t state;
    uint32_t reserved;
    uint64_t generation;
    eshkol_cap_id_t space_cap;
    eshkol_cap_id_t cap_id;
    uint8_t* buf_base;
    uint64_t dma_vaddr;
    size_t lease_bytes;
    uint64_t mapped_pages;
} eshkol_os_dma_buffer_slot_t;

/**
 * @brief Configuration for initializing a userspace OALR driver runtime.
 */
typedef struct eshkol_os_drv_runtime_config {
    eshkol_os_syscall6_fn_t syscall_fn; /* Mandatory capability syscall dispatcher */
    uint64_t page_size;                 /* OS page size in bytes (0 -> 4096) */
} eshkol_os_drv_runtime_config_t;

/**
 * @brief Caller-owned userspace OALR driver runtime state (zero heap allocation).
 */
typedef struct eshkol_os_drv_runtime {
    eshkol_os_syscall6_fn_t syscall_fn;
    uint64_t page_size;
    uint64_t next_generation;
    uint64_t total_syscalls;
    uint64_t live_dma_pages;
    uint32_t initialized;
    uint32_t reserved;
    eshkol_os_dev_region_slot_t regions[ESHKOL_OS_DRV_MAX_SLOTS];
    eshkol_os_irq_lease_slot_t irqs[ESHKOL_OS_DRV_MAX_SLOTS];
    eshkol_os_dma_buffer_slot_t dmas[ESHKOL_OS_DRV_MAX_SLOTS];
} eshkol_os_drv_runtime_t;

/**
 * @brief VirtIO split-virtqueue descriptor matching the Phase U2 hardware ABI.
 */
typedef struct eshkol_virtq_desc {
    ESHKOL_OS_ALIGNAS(16) uint64_t addr;
    uint32_t len;
    uint16_t flags;
    uint16_t next;
} eshkol_virtq_desc_t;

/**
 * @brief VirtIO split-virtqueue available ring matching the VirtIO 1.2 specification.
 */
typedef struct eshkol_virtq_avail {
    uint16_t flags;
    uint16_t idx;
    uint16_t ring[ESHKOL_OS_VIRTQ_SIZE];
} eshkol_virtq_avail_t;

/**
 * @brief VirtIO split-virtqueue used ring element matching the Phase U2 hardware ABI.
 */
typedef struct eshkol_virtq_used_elem {
    uint32_t id;
    uint32_t len;
} eshkol_virtq_used_elem_t;

/**
 * @brief VirtIO split-virtqueue used ring matching the Phase U2 hardware ABI.
 */
typedef struct eshkol_virtq_used {
    ESHKOL_OS_ALIGNAS(4) uint16_t flags;
    uint16_t idx;
    eshkol_virtq_used_elem_t ring[ESHKOL_OS_VIRTQ_SIZE];
} eshkol_virtq_used_t;

/**
 * @brief Descriptor segment specification passed to `eshkol_os_virtq_submit_chain`.
 */
typedef struct eshkol_os_virtq_seg {
    uint16_t desc_idx;
    uint16_t flags;
    uint32_t len;
    size_t dma_offset;
} eshkol_os_virtq_seg_t;

/**
 * @brief Completed descriptor chain notification returned by `eshkol_os_virtq_complete_one`.
 */
typedef struct eshkol_os_virtq_completion {
    uint32_t head_id;
    uint32_t bytes_written;
} eshkol_os_virtq_completion_t;

/**
 * @brief Userspace VirtIO split-virtqueue instance bound to linear capability handles.
 */
typedef struct eshkol_os_virtq {
    eshkol_os_drv_runtime_t* rt;
    eshkol_os_device_region_t dev_region;
    eshkol_os_irq_lease_t irq_lease;
    eshkol_os_dma_buffer_t dma_buf;
    eshkol_cap_id_t notify_cap;
    uint32_t queue_index;
    uint32_t in_flight_mask;
    uint16_t last_avail_idx;
    uint16_t last_used_idx;
    uint16_t expected_heads[ESHKOL_OS_VIRTQ_SIZE];
    uint16_t fifo_head;
    uint16_t fifo_tail;
    uint16_t fifo_count;
    uint16_t initialized;
    eshkol_virtq_desc_t desc[ESHKOL_OS_VIRTQ_SIZE];
    eshkol_virtq_avail_t avail;
    eshkol_virtq_used_t used;
} eshkol_os_virtq_t;

/**
 * @brief VirtIO block request header (16 bytes, little-endian).
 */
typedef struct eshkol_virtio_blk_req_hdr {
    uint32_t type;
    uint32_t reserved;
    uint64_t sector;
} eshkol_virtio_blk_req_hdr_t;

/**
 * @brief VirtIO network packet header (12 bytes, VirtIO 1.2 legacy/modern base).
 */
typedef struct eshkol_virtio_net_hdr {
    uint8_t flags;
    uint8_t gso_type;
    uint16_t hdr_len;
    uint16_t gso_size;
    uint16_t csum_start;
    uint16_t csum_offset;
    uint16_t num_buffers;
} eshkol_virtio_net_hdr_t;

/**
 * @brief Device-side kick callback invoked after a VirtIO queue submission.
 *
 * Models the hardware or VMM device processing the published `avail` ring and
 * writing completion records into `used`. Returns 0 on transport success or a
 * negative error code on hardware fault / simulated power loss.
 */
typedef int (*eshkol_os_virtio_device_kick_fn_t)(eshkol_os_virtq_t* vq,
                                                 uint8_t* dma_base,
                                                 size_t dma_bytes,
                                                 void* dev_ctx);

/**
 * @brief Abstract VirtIO block device handle consumed by the atomic CAS VFS.
 */
typedef struct eshkol_os_blk_dev {
    eshkol_os_virtq_t* vq;
    eshkol_os_virtio_device_kick_fn_t kick_fn;
    void* dev_ctx;
    uint64_t total_sectors;
    size_t dma_scratch_offset;
} eshkol_os_blk_dev_t;

/**
 * @brief VFS entry kind discriminator.
 */
typedef enum eshkol_os_vfs_entry_kind {
    ESHKOL_VFS_KIND_NONE = 0,
    ESHKOL_VFS_KIND_FILE = 1,
    ESHKOL_VFS_KIND_DIR  = 2
} eshkol_os_vfs_entry_kind_t;

/**
 * @brief On-disk and in-memory metadata record for one committed VFS entry (96 bytes).
 */
typedef struct eshkol_os_vfs_entry {
    char name[32];
    uint32_t entry_id;
    uint32_t parent_id;
    uint32_t kind;
    uint32_t size_bytes;
    uint32_t first_sector;
    uint32_t sector_count;
    uint32_t active;
    uint32_t reserved;
    uint8_t sha256[ESHKOL_OS_SHA256_BYTES];
} eshkol_os_vfs_entry_t;

/**
 * @brief Caller-owned atomic content-addressed VFS instance (zero heap allocation).
 */
typedef struct eshkol_os_cas_vfs {
    eshkol_os_blk_dev_t* blk;
    uint64_t generation;
    uint32_t active_slot;
    uint32_t next_free_sector;
    uint32_t next_entry_id;
    uint32_t mounted;
    uint32_t entry_count;
    uint32_t reserved;
    eshkol_os_vfs_entry_t entries[ESHKOL_OS_VFS_MAX_ENTRIES];
} eshkol_os_cas_vfs_t;

/* ── ABI layout assertions matching Phase U2 hardware layouts ── */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(eshkol_virtq_desc_t) == 16, "virtq_desc must be 16 bytes");
_Static_assert(_Alignof(eshkol_virtq_desc_t) == 16, "virtq_desc must be 16-byte aligned");
_Static_assert(sizeof(eshkol_virtq_used_elem_t) == 8, "virtq_used_elem must be 8 bytes");
_Static_assert(sizeof(eshkol_virtq_used_t) == 132, "virtq_used (16 entries) must be 132 bytes");
_Static_assert(sizeof(eshkol_virtio_blk_req_hdr_t) == 16, "virtio_blk_req_hdr must be 16 bytes");
_Static_assert(sizeof(eshkol_virtio_net_hdr_t) == 12, "virtio_net_hdr must be 12 bytes");
_Static_assert(sizeof(eshkol_os_vfs_entry_t) == 96, "vfs_entry must be 96 bytes");
#elif defined(__cplusplus) && __cplusplus >= 201103L
static_assert(sizeof(eshkol_virtq_desc_t) == 16, "virtq_desc must be 16 bytes");
static_assert(alignof(eshkol_virtq_desc_t) == 16, "virtq_desc must be 16-byte aligned");
static_assert(sizeof(eshkol_virtq_used_elem_t) == 8, "virtq_used_elem must be 8 bytes");
static_assert(sizeof(eshkol_virtq_used_t) == 132, "virtq_used (16 entries) must be 132 bytes");
static_assert(sizeof(eshkol_virtio_blk_req_hdr_t) == 16, "virtio_blk_req_hdr must be 16 bytes");
static_assert(sizeof(eshkol_virtio_net_hdr_t) == 12, "virtio_net_hdr must be 12 bytes");
static_assert(sizeof(eshkol_os_vfs_entry_t) == 96, "vfs_entry must be 96 bytes");
#endif

/* ── Driver runtime & linear handle API ── */

/**
 * @brief Initialize a userspace OALR driver runtime bound to a capability syscall dispatcher.
 */
eshkol_os_drv_status_t eshkol_os_drv_runtime_init(
    eshkol_os_drv_runtime_t* rt,
    const eshkol_os_drv_runtime_config_t* config);

/**
 * @brief Acquire a linear single-ownership `DeviceRegion` handle bound to `cap_id`.
 */
eshkol_os_drv_status_t eshkol_os_device_region_acquire(
    eshkol_os_drv_runtime_t* rt,
    eshkol_cap_id_t cap_id,
    void* mapped_base,
    size_t region_bytes,
    eshkol_os_device_region_t* out_handle);

/**
 * @brief Linearly move ownership of a `DeviceRegion` handle, invalidating `src`.
 */
eshkol_os_drv_status_t eshkol_os_device_region_transfer(
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_device_region_t* src,
    eshkol_os_device_region_t* dst);

/**
 * @brief Perform a bounds-checked volatile 32-bit MMIO read from a `DeviceRegion`.
 */
eshkol_os_drv_status_t eshkol_os_device_region_read32(
    const eshkol_os_drv_runtime_t* rt,
    const eshkol_os_device_region_t* handle,
    size_t offset,
    uint32_t* out_val);

/**
 * @brief Perform a bounds-checked volatile 32-bit MMIO write into a `DeviceRegion`.
 */
eshkol_os_drv_status_t eshkol_os_device_region_write32(
    eshkol_os_drv_runtime_t* rt,
    const eshkol_os_device_region_t* handle,
    size_t offset,
    uint32_t val);

/**
 * @brief Release a linear `DeviceRegion` handle (double-free fails closed).
 */
eshkol_os_drv_status_t eshkol_os_device_region_release(
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_device_region_t* handle);

/**
 * @brief Acquire a linear single-ownership `IrqLease` handle bound to `cap_id`.
 */
eshkol_os_drv_status_t eshkol_os_irq_lease_acquire(
    eshkol_os_drv_runtime_t* rt,
    eshkol_cap_id_t cap_id,
    uint32_t irq_vector,
    eshkol_os_irq_lease_t* out_handle);

/**
 * @brief Linearly move ownership of an `IrqLease` handle, invalidating `src`.
 */
eshkol_os_drv_status_t eshkol_os_irq_lease_transfer(
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_irq_lease_t* src,
    eshkol_os_irq_lease_t* dst);

/**
 * @brief Acknowledge an IRQ through the bound capability via `ESHKOL_SYS_IPC_COMPLETE`.
 */
eshkol_os_drv_status_t eshkol_os_irq_lease_ack(
    eshkol_os_drv_runtime_t* rt,
    const eshkol_os_irq_lease_t* handle);

/**
 * @brief Release a linear `IrqLease` handle (double-free fails closed).
 */
eshkol_os_drv_status_t eshkol_os_irq_lease_release(
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_irq_lease_t* handle);

/**
 * @brief Acquire a linear `DmaBuffer` handle and map its pages via `ESHKOL_SYS_PAGE_MAP`.
 */
eshkol_os_drv_status_t eshkol_os_dma_buffer_acquire(
    eshkol_os_drv_runtime_t* rt,
    eshkol_cap_id_t space_cap,
    eshkol_cap_id_t dma_cap,
    void* buf_base,
    size_t lease_bytes,
    eshkol_os_dma_buffer_t* out_handle);

/**
 * @brief Linearly move ownership of a `DmaBuffer` handle, invalidating `src`.
 */
eshkol_os_drv_status_t eshkol_os_dma_buffer_transfer(
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_dma_buffer_t* src,
    eshkol_os_dma_buffer_t* dst);

/**
 * @brief Copy bytes into a `DmaBuffer` with overflow-safe lease bounds enforcement.
 */
eshkol_os_drv_status_t eshkol_os_dma_write(
    eshkol_os_drv_runtime_t* rt,
    const eshkol_os_dma_buffer_t* handle,
    size_t offset,
    const void* src,
    size_t len);

/**
 * @brief Copy bytes out of a `DmaBuffer` with overflow-safe lease bounds enforcement.
 */
eshkol_os_drv_status_t eshkol_os_dma_read(
    const eshkol_os_drv_runtime_t* rt,
    const eshkol_os_dma_buffer_t* handle,
    size_t offset,
    void* dst,
    size_t len);

/**
 * @brief Release a linear `DmaBuffer` handle and unmap its pages via `ESHKOL_SYS_PAGE_UNMAP`.
 */
eshkol_os_drv_status_t eshkol_os_dma_buffer_release(
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_dma_buffer_t* handle);

/**
 * @brief Revoke a capability via `ESHKOL_SYS_CAP_REVOKE` and invalidate all bound handles.
 */
eshkol_os_drv_status_t eshkol_os_drv_revoke_cap(
    eshkol_os_drv_runtime_t* rt,
    eshkol_cap_id_t cap_id);

/* ── VirtIO split-virtqueue & block/net driver API ── */

/**
 * @brief Initialize a userspace VirtIO split virtqueue bound to linear driver handles.
 */
eshkol_os_drv_status_t eshkol_os_virtq_init(
    eshkol_os_virtq_t* vq,
    eshkol_os_drv_runtime_t* rt,
    eshkol_os_device_region_t dev_region,
    eshkol_os_irq_lease_t irq_lease,
    eshkol_os_dma_buffer_t dma_buf,
    eshkol_cap_id_t notify_cap,
    uint32_t queue_index);

/**
 * @brief Validate and submit a descriptor chain onto the VirtIO available ring.
 *
 * Rejects duplicate or already-in-flight descriptors (`ESHKOL_DRV_ERR_VQ_DUPLICATE_DESC`),
 * out-of-range indices (`ESHKOL_DRV_ERR_VQ_BAD_DESC`), DMA lease overflows
 * (`ESHKOL_DRV_ERR_DMA_BOUNDS`), and revoked or stale linear handles.
 */
eshkol_os_drv_status_t eshkol_os_virtq_submit_chain(
    eshkol_os_virtq_t* vq,
    const eshkol_os_virtq_seg_t* segs,
    uint32_t n_segs);

/**
 * @brief Consume one completion from the VirtIO used ring with strict ordering checks.
 *
 * Rejects out-of-order, spurious, or unsubmitted descriptor completions with
 * `ESHKOL_DRV_ERR_VQ_OUT_OF_ORDER`.
 */
eshkol_os_drv_status_t eshkol_os_virtq_complete_one(
    eshkol_os_virtq_t* vq,
    eshkol_os_virtq_completion_t* out_comp);

/**
 * @brief Execute a synchronous VirtIO block sector read or write over a split virtqueue.
 */
eshkol_os_drv_status_t eshkol_os_virtio_blk_io(
    eshkol_os_blk_dev_t* blk,
    int is_write,
    uint64_t sector,
    void* buf,
    uint32_t sector_count);

/**
 * @brief Execute a synchronous VirtIO network TX or RX frame transfer over a split virtqueue.
 */
eshkol_os_drv_status_t eshkol_os_virtio_net_xfer(
    eshkol_os_virtq_t* vq,
    eshkol_os_virtio_device_kick_fn_t kick_fn,
    void* dev_ctx,
    int is_tx,
    size_t dma_offset,
    void* frame_buf,
    size_t frame_len,
    size_t* out_len);

/* ── Freestanding SHA-256 & Atomic Content-Addressed VFS API ── */

/**
 * @brief Compute a FIPS 180-4 SHA-256 digest over an in-memory byte buffer.
 */
void eshkol_os_sha256(const void* data,
                      size_t len,
                      uint8_t out_digest[ESHKOL_OS_SHA256_BYTES]);

/**
 * @brief Format an empty atomic content-addressed VFS onto a VirtIO block device.
 */
eshkol_os_drv_status_t eshkol_os_cas_vfs_format(
    eshkol_os_cas_vfs_t* vfs,
    eshkol_os_blk_dev_t* blk);

/**
 * @brief Mount an existing CAS VFS by verifying both A/B superblocks and index CRC32Cs.
 */
eshkol_os_drv_status_t eshkol_os_cas_vfs_mount(
    eshkol_os_cas_vfs_t* vfs,
    eshkol_os_blk_dev_t* blk);

/**
 * @brief Atomically create a directory entry in the CAS VFS.
 */
eshkol_os_drv_status_t eshkol_os_cas_vfs_mkdir(
    eshkol_os_cas_vfs_t* vfs,
    const char* name,
    uint32_t parent_id,
    uint32_t* out_entry_id);

/**
 * @brief Atomically stage, SHA-256-verify, and commit a file into the CAS VFS.
 */
eshkol_os_drv_status_t eshkol_os_cas_vfs_put_file(
    eshkol_os_cas_vfs_t* vfs,
    const char* name,
    uint32_t parent_id,
    const void* data,
    size_t len,
    uint8_t out_sha256[ESHKOL_OS_SHA256_BYTES]);

/**
 * @brief Read a committed file by name and verify its SHA-256 digest end-to-end.
 */
eshkol_os_drv_status_t eshkol_os_cas_vfs_read_file(
    const eshkol_os_cas_vfs_t* vfs,
    const char* name,
    uint32_t parent_id,
    void* out_buf,
    size_t out_cap,
    size_t* out_len,
    uint8_t out_sha256[ESHKOL_OS_SHA256_BYTES]);

/**
 * @brief Read committed content directly by its 32-byte SHA-256 content address.
 */
eshkol_os_drv_status_t eshkol_os_cas_vfs_read_by_hash(
    const eshkol_os_cas_vfs_t* vfs,
    const uint8_t sha256[ESHKOL_OS_SHA256_BYTES],
    void* out_buf,
    size_t out_cap,
    size_t* out_len);

/**
 * @brief Return a static human-readable name for a driver/VFS status code.
 */
const char* eshkol_os_drv_status_name(eshkol_os_drv_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_OS_DRIVERS_VFS_H */
