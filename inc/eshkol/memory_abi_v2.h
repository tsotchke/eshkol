/**
 * @file memory_abi_v2.h
 * @brief OALR object/layout ABI v2 — type definitions and layout pinning (ADR-0001 §3).
 *
 * ADR-0000 Stage 4 (v1.4.1) lands the OALR memory-ABI break: the 8-byte
 * ::eshkol_object_header_t grows to a 32-byte header that carries an exact
 * layout descriptor id, a stable object identity, and the object's residence
 * (home). This header is **Phase A** of that work:
 *
 *  - it *defines* the v2 header and the layout-descriptor record exactly as
 *    ADR-0001 §3 specifies them;
 *  - it *pins both* layouts with static assertions (v1 = 8 bytes, v2 = 32
 *    bytes with the field offsets the deep-walk evacuators depend on), so a
 *    later migration cannot silently move a field;
 *  - it exposes which ABI the current build actually uses
 *    (::ESHKOL_MEMORY_ABI_ACTIVE / ::eshkol_memory_abi_active()).
 *
 * It **does not** migrate any allocator, codegen site, or runtime consumer.
 * Selecting v2 is off by default (`-DESHKOL_MEMORY_ABI_V2=ON` at configure
 * time defines `ESHKOL_MEMORY_ABI_V2_ENABLED`); with the flag off every
 * definition here is inert type-level material and the live object ABI is
 * unchanged. See docs/design/ABI_V2_MIGRATION_INVENTORY.md for the full
 * inventory of sites the migration must convert, and for why the migration
 * cannot be staged one subsystem at a time.
 *
 * Naming: `ESHKOL_MEMORY_ABI_V2` is the ABI *version number* (value 2), as
 * written in ADR-0001. `ESHKOL_MEMORY_ABI_V2_ENABLED` is the build switch.
 * They are deliberately different spellings so that testing for the switch
 * can never be confused with testing for the version constant.
 */
#ifndef ESHKOL_MEMORY_ABI_V2_H
#define ESHKOL_MEMORY_ABI_V2_H

#include <stdint.h>
#include <stddef.h>

/* Not standalone: this header pins the ABI v1 layout next to the v2 one, so it
 * needs eshkol_object_header_t and ESHKOL_STATIC_ASSERT to already be in scope.
 * eshkol/eshkol.h includes it at exactly the right point. */
#ifndef ESHKOL_STATIC_ASSERT
#error "include <eshkol/eshkol.h> instead of <eshkol/memory_abi_v2.h> directly"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * ABI version constants
 * ───────────────────────────────────────────────────────────────────────── */

/** Object ABI v1: the shipped 8-byte ::eshkol_object_header_t. */
#define ESHKOL_MEMORY_ABI_V1 1u

/** Object ABI v2: the 32-byte ::eshkol_object_header_v2_t (ADR-0001 §3). */
#define ESHKOL_MEMORY_ABI_V2 2u

#if defined(ESHKOL_MEMORY_ABI_V2_ENABLED) && (ESHKOL_MEMORY_ABI_V2_ENABLED)
/** ABI version this translation unit was compiled against. */
#define ESHKOL_MEMORY_ABI_ACTIVE ESHKOL_MEMORY_ABI_V2
#else
#define ESHKOL_MEMORY_ABI_ACTIVE ESHKOL_MEMORY_ABI_V1
#endif

/* `alignas` is a keyword in C++ and (since C11) a macro in <stdalign.h>. */
#ifdef __cplusplus
#define ESHKOL_ABI_ALIGNAS(n) alignas(n)
#else
#define ESHKOL_ABI_ALIGNAS(n) _Alignas(n)
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * Residence (opaque in Phase A)
 * ───────────────────────────────────────────────────────────────────────── */

/**
 * @brief The reclaimable memory domain an object lives in (ADR-0001 §2).
 *
 * Opaque here: Phase A only needs the header to be able to *name* a home, so
 * that `eq?` identity and cross-thread residence queries become O(1) instead
 * of a scan of one thread's arena block lists. The concrete definition
 * (epoch, state, owner thread, retired-epoch ring) lands with the memctx work
 * in Phase B.
 */
typedef struct eshkol_residence eshkol_residence_t;

/* ─────────────────────────────────────────────────────────────────────────
 * Object header, ABI v2
 * ───────────────────────────────────────────────────────────────────────── */

/**
 * @brief 32-byte object header (ADR-0001 §3), payload 16-byte aligned.
 *
 * Field-by-field rationale, versus the v1 header:
 *  - `payload_size` — unchanged in meaning from v1 `size` (bytes, excluding
 *    the header).
 *  - `layout_id`    — NEW. Index into the layout-descriptor registry. This is
 *    the field the deep-walk evacuator needs: "unknown means leaf" is
 *    forbidden under v2, so every pointer-bearing allocation names an exact
 *    tracer/finalizer instead of being guessed at by the
 *    integer-looks-like-an-arena-address heuristic in runtime_regions.cpp.
 *  - `subtype`,`flags` — unchanged in meaning from v1.
 *  - `object_id`    — NEW. Stable identity across evacuation/resident copies,
 *    so `eq?` survives a region exit that moved the object. Never reused
 *    while an object is live. Pointer address stays the fast path when the
 *    two objects' `home` and epoch match.
 *  - `home`         — NEW. O(1), cross-thread-safe residence query.
 *  - `aux`          — reserved; MUST be zero in ABI v2.
 *
 * v1's `ref_count` has no v2 counterpart: refcounting is subsumed by
 * residence ownership plus the transfer-capsule protocol (ADR-0001 §5).
 */
typedef struct eshkol_object_header_v2 {
    /* The alignment sits on the first member rather than on the struct tag:
     * `_Alignas` is a declaration specifier in C and may not appear after
     * `struct`, so the C++-only `struct alignas(16) X {}` spelling in ADR-0001
     * does not compile in the C translation units that include this header.
     * Raising the first member's alignment raises the struct's, which is what
     * the ADR's `alignas(16)` means. */
    ESHKOL_ABI_ALIGNAS(16)
    uint32_t           payload_size;  /**< Payload bytes, excluding this header. */
    uint16_t           layout_id;     /**< Exact tracer/finalizer descriptor id. */
    uint8_t            subtype;       /**< Same subtype space as ABI v1. */
    uint8_t            flags;         /**< Same object flag bits as ABI v1. */
    uint64_t           object_id;     /**< Stable identity; never reused while live. */
    eshkol_residence_t *home;         /**< Owning residence; O(1) residence query. */
    uint64_t           aux;           /**< Reserved; zero in ABI v2. */
} eshkol_object_header_v2_t;

/* ─────────────────────────────────────────────────────────────────────────
 * Layout descriptors
 * ───────────────────────────────────────────────────────────────────────── */

/** @brief Callback invoked once per traced tagged slot of an object. */
typedef void (*eshkol_visit_slot_fn)(struct eshkol_tagged_value *slot, void *ctx);

/** @brief Exact tracer for one layout: visits every pointer-bearing slot. */
typedef int (*eshkol_trace_fn)(void *payload, eshkol_visit_slot_fn visit, void *ctx);

/** @brief Optional finalizer for one layout (external resources only). */
typedef void (*eshkol_finalize_fn)(void *payload);

/** @brief Layout descriptor flag bits (ADR-0001 §3). */
typedef enum {
    ESHKOL_LAYOUT_FLAG_NONE     = 0u,
    ESHKOL_LAYOUT_FLAG_LEAF     = 1u << 0, /**< Carries no traceable pointers. */
    ESHKOL_LAYOUT_FLAG_MUTABLE  = 1u << 1, /**< Slots may be stored into after creation. */
    ESHKOL_LAYOUT_FLAG_EXTERNAL = 1u << 2, /**< Wraps an OS/foreign resource handle. */
    ESHKOL_LAYOUT_FLAG_PINNED   = 1u << 3  /**< Must not be moved by evacuation. */
} eshkol_layout_flag_t;

/**
 * @brief One registered object layout.
 *
 * ADR-0001 §3: "every pointer-bearing layout registers a descriptor or
 * startup fails". Conses, vectors/records, multiple values,
 * closures/environments, hash backing arrays, exceptions, tensors, promises,
 * substitutions, knowledge bases, factor graphs, workspaces, DNC/SDNC, Taylor
 * values, parameters and continuations each get one.
 */
typedef struct eshkol_layout_desc {
    uint16_t           layout_id;  /**< Matches eshkol_object_header_v2::layout_id. */
    uint16_t           flags;      /**< Bitwise OR of ::eshkol_layout_flag_t. */
    uint32_t           min_size;   /**< Smallest legal payload for this layout. */
    eshkol_trace_fn    trace;      /**< Exact tracer; NULL only when LEAF. */
    eshkol_finalize_fn finalize;   /**< Finalizer, or NULL. */
} eshkol_layout_desc_t;

/** @brief Reserved layout id meaning "not yet assigned" — never registrable. */
#define ESHKOL_LAYOUT_ID_INVALID ((uint16_t)0)

/* ─────────────────────────────────────────────────────────────────────────
 * Layout pinning
 *
 * Both ABIs are pinned unconditionally, in every build, whichever one is
 * selected. A migration PR that moves a field or changes a size fails to
 * compile rather than silently producing objects the other half of the
 * toolchain (stdlib bitcode, JIT modules, AOT objects) cannot read.
 * ───────────────────────────────────────────────────────────────────────── */

ESHKOL_STATIC_ASSERT(sizeof(eshkol_object_header_t) == 8,
                     "ABI v1 object header must remain 8 bytes");
ESHKOL_STATIC_ASSERT(offsetof(eshkol_object_header_t, subtype) == 0,
                     "ABI v1: subtype must stay at offset 0");
ESHKOL_STATIC_ASSERT(offsetof(eshkol_object_header_t, flags) == 1,
                     "ABI v1: flags must stay at offset 1");
ESHKOL_STATIC_ASSERT(offsetof(eshkol_object_header_t, size) == 4,
                     "ABI v1: size must stay at offset 4");

ESHKOL_STATIC_ASSERT(sizeof(eshkol_object_header_v2_t) == 32,
                     "ABI v2 object header must be exactly 32 bytes (ADR-0001 SS3)");
ESHKOL_STATIC_ASSERT(offsetof(eshkol_object_header_v2_t, payload_size) == 0,
                     "ABI v2: payload_size at offset 0");
ESHKOL_STATIC_ASSERT(offsetof(eshkol_object_header_v2_t, layout_id) == 4,
                     "ABI v2: layout_id at offset 4");
ESHKOL_STATIC_ASSERT(offsetof(eshkol_object_header_v2_t, subtype) == 6,
                     "ABI v2: subtype at offset 6");
ESHKOL_STATIC_ASSERT(offsetof(eshkol_object_header_v2_t, flags) == 7,
                     "ABI v2: flags at offset 7");
ESHKOL_STATIC_ASSERT(offsetof(eshkol_object_header_v2_t, object_id) == 8,
                     "ABI v2: object_id at offset 8");
ESHKOL_STATIC_ASSERT(offsetof(eshkol_object_header_v2_t, home) == 16,
                     "ABI v2: home at offset 16");
ESHKOL_STATIC_ASSERT(offsetof(eshkol_object_header_v2_t, aux) == 24,
                     "ABI v2: aux at offset 24");
/* The payload begins immediately after the header, so a 32-byte header whose
 * own alignment is 16 keeps every payload 16-byte aligned — which is what the
 * 16-byte eshkol_tagged_value_t ABI requires of every slot it lands in. */
ESHKOL_STATIC_ASSERT(sizeof(eshkol_object_header_v2_t) % 16 == 0,
                     "ABI v2 header size must keep the payload 16-byte aligned");

/* ─────────────────────────────────────────────────────────────────────────
 * Active-ABI selection
 * ───────────────────────────────────────────────────────────────────────── */

#if ESHKOL_MEMORY_ABI_ACTIVE == ESHKOL_MEMORY_ABI_V2
/** @brief The object header layout this build actually allocates. */
typedef eshkol_object_header_v2_t eshkol_object_header_active_t;
#else
typedef eshkol_object_header_t eshkol_object_header_active_t;
#endif

/** @brief Size in bytes of the header prefix this build actually allocates. */
#define ESHKOL_OBJECT_HEADER_SIZE (sizeof(eshkol_object_header_active_t))

/** @brief Payload alignment this build's allocator must guarantee. */
#if ESHKOL_MEMORY_ABI_ACTIVE == ESHKOL_MEMORY_ABI_V2
#define ESHKOL_OBJECT_PAYLOAD_ALIGN ((size_t)16)
#else
#define ESHKOL_OBJECT_PAYLOAD_ALIGN ((size_t)8)
#endif

/**
 * @brief Report the object ABI version the *runtime library* was built with.
 *
 * The compile-time ::ESHKOL_MEMORY_ABI_ACTIVE describes the translation unit
 * asking the question; this function describes the runtime it is linked
 * against. AOT objects, JIT modules, precompiled stdlib bitcode and the
 * runtime must all agree, so the migration's first job in Phase B is to make
 * a mismatch a hard startup error rather than a silent miscompile. Phase A
 * only makes the two observable so a test can compare them.
 *
 * @return ::ESHKOL_MEMORY_ABI_V1 or ::ESHKOL_MEMORY_ABI_V2.
 */
uint32_t eshkol_memory_abi_active(void);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_MEMORY_ABI_V2_H */
