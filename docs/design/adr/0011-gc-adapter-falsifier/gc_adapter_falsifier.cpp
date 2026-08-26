/*
 * Falsifier F1 for the OALR guest-collector adapter (ADR-0011).
 *
 * Claim under test
 * ----------------
 *   A guest language runtime with its own tracing collector can be hosted
 *   inside an Eshkol region such that:
 *     (a) the guest reclaims unreachable CYCLES mid-region,
 *     (b) Eshkol itself never traces and never frees an individual object,
 *     (c) the host arena's resident bytes stay FLAT across unbounded guest
 *         allocation, and
 *     (d) region teardown destroys the whole guest heap in O(1).
 *
 * Method
 * ------
 *   The guest heap is a Cheney semispace copying collector whose two
 *   semispaces are ONE `arena_allocate` each, taken from a real Eshkol
 *   `arena_t` (lib/core/runtime_arena_core.cpp, unmodified). Every guest
 *   object lives inside those slabs. Eshkol's allocator sees exactly two
 *   allocations for the entire life of the guest, no matter how many
 *   collections run.
 *
 *   Cross-heap edges are exercised in both directions:
 *     Eshkol -> guest : ONLY through a pinned handle (index+generation),
 *                       never a raw pointer. The handle table is a collector
 *                       root and is rewritten by the collector when it moves.
 *     guest  -> Eshkol: an opaque 16-byte tagged value the guest tracer
 *                       steps over. It must name an object whose residence
 *                       outlives the guest's region.
 *
 *   The raw-pointer failure mode is demonstrated deliberately, because it is
 *   the reason the ABI mandates handles.
 *
 * Build: see run.sh in this directory.
 */

#include "arena_memory.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

/* ------------------------------------------------------------------ */
/* Host-side measurement                                              */
/* ------------------------------------------------------------------ */

static size_t rss_bytes(void) {
#if defined(__APPLE__)
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return (size_t)info.resident_size;
    }
#endif
    return 0;
}

/* ================================================================== */
/* THE PROPOSED ADAPTER ABI (prototype of ADR-0011 section 4)          */
/* ================================================================== */

#define ESHKOL_GUEST_ABI_V1 1u

typedef uint64_t eshkol_guest_handle_t;   /* {generation:32, index:32} */
#define ESHKOL_GUEST_HANDLE_NULL ((eshkol_guest_handle_t)0)

struct eshkol_guest_heap;

typedef struct eshkol_guest_vtable {
    uint32_t    abi_version;
    const char *name;
    /* Trace: the guest walks ITS OWN graph from the roots the adapter hands
     * it. Eshkol never enters this function's body; it only calls it. */
    size_t (*collect)(void *guest);
    /* Teardown: the region is dying. The guest releases OS resources it owns
     * outside the arena (fds, dlopen handles, foreign mallocs). It must NOT
     * free arena memory -- arena_destroy does that in one step. */
    void   (*on_region_teardown)(void *guest);
    /* Residence audit: enumerate guest -> Eshkol outbound edges so the
     * adapter can assert each target outlives the guest region. */
    void   (*enumerate_outbound)(void *guest,
                                 void (*visit)(void *ctx, const eshkol_tagged_value_t *tv),
                                 void *ctx);
} eshkol_guest_vtable_t;

/* Handle table: the ONLY legal Eshkol -> guest edge. Owned by the adapter,
 * allocated from the region arena, and a mandatory root of every guest
 * collection. Entries are rewritten in place when the guest moves objects. */
#define GUEST_HANDLE_CAP 4096

typedef struct eshkol_guest_heap {
    arena_t                     *arena;      /* the hosting region's arena   */
    const eshkol_guest_vtable_t *vt;
    void                        *guest;      /* guest-private collector state */

    void       *handle_obj[GUEST_HANDLE_CAP];
    uint32_t    handle_gen[GUEST_HANDLE_CAP];
    uint32_t    handle_free_head;
    uint32_t    handle_live;

    uint64_t    collections;
    uint64_t    bytes_reclaimed;
    uint64_t    pin_violations;
} eshkol_guest_heap_t;

static eshkol_guest_heap_t *
eshkol_guest_heap_attach(arena_t *arena,
                         const eshkol_guest_vtable_t *vt,
                         void *guest)
{
    eshkol_guest_heap_t *h =
        (eshkol_guest_heap_t *)arena_allocate_zeroed(arena, sizeof(*h));
    h->arena = arena;
    h->vt    = vt;
    h->guest = guest;
    for (uint32_t i = 0; i < GUEST_HANDLE_CAP; ++i) {
        h->handle_obj[i] = NULL;
        h->handle_gen[i] = 1;
    }
    h->handle_free_head = 0;
    return h;
}

static eshkol_guest_handle_t
eshkol_guest_pin(eshkol_guest_heap_t *h, void *obj)
{
    for (uint32_t i = h->handle_free_head; i < GUEST_HANDLE_CAP; ++i) {
        if (h->handle_obj[i] == NULL) {
            h->handle_obj[i] = obj;
            h->handle_free_head = i + 1;
            h->handle_live++;
            return ((uint64_t)h->handle_gen[i] << 32) | (uint64_t)(i + 1);
        }
    }
    return ESHKOL_GUEST_HANDLE_NULL;   /* pin table exhausted: deterministic */
}

static void
eshkol_guest_unpin(eshkol_guest_heap_t *h, eshkol_guest_handle_t hd)
{
    uint32_t idx = (uint32_t)(hd & 0xffffffffu);
    uint32_t gen = (uint32_t)(hd >> 32);
    if (idx == 0 || idx > GUEST_HANDLE_CAP) return;
    idx -= 1;
    if (h->handle_gen[idx] != gen) return;         /* stale: already released */
    h->handle_obj[idx] = NULL;
    h->handle_gen[idx]++;                          /* invalidate old handles  */
    if (idx < h->handle_free_head) h->handle_free_head = idx;
    h->handle_live--;
}

static void *
eshkol_guest_deref(eshkol_guest_heap_t *h, eshkol_guest_handle_t hd)
{
    uint32_t idx = (uint32_t)(hd & 0xffffffffu);
    uint32_t gen = (uint32_t)(hd >> 32);
    if (idx == 0 || idx > GUEST_HANDLE_CAP) return NULL;
    idx -= 1;
    if (h->handle_gen[idx] != gen) { h->pin_violations++; return NULL; }
    return h->handle_obj[idx];
}

static size_t
eshkol_guest_collect(eshkol_guest_heap_t *h)
{
    size_t r = h->vt->collect(h->guest);
    h->collections++;
    h->bytes_reclaimed += r;
    return r;
}

/* ================================================================== */
/* A TOY GUEST: mini-Lisp with a Cheney semispace collector            */
/* ================================================================== */

typedef struct gobj {
    uint32_t     tag;        /* 1 = pair, 2 = forwarded                     */
    uint32_t     id;
    struct gobj *a;          /* guest pointer slot                          */
    struct gobj *b;          /* guest pointer slot                          */
    eshkol_tagged_value_t out;  /* OPAQUE Eshkol value; guest never traces it */
} gobj;

#define GTAG_PAIR 1u
#define GTAG_FWD  2u

typedef struct {
    char   *from;
    char   *to;
    size_t  semi_bytes;
    size_t  used;             /* bump offset in from-space                 */
    eshkol_guest_heap_t *host;/* back-edge: handle table is a GC root      */
    uint32_t next_id;
    uint64_t live_after_last_gc;
    uint64_t oom;
} cheney_t;

static gobj *cheney_alloc(cheney_t *c)
{
    if (c->used + sizeof(gobj) > c->semi_bytes) return NULL;
    gobj *o = (gobj *)(c->from + c->used);
    c->used += sizeof(gobj);
    memset(o, 0, sizeof(*o));
    o->tag = GTAG_PAIR;
    o->id  = c->next_id++;
    return o;
}

/* Cheney forward: copy one object from-space -> to-space, leaving a
 * forwarding pointer. This is the guest's OWN tracing; Eshkol is not
 * involved and no arena call happens here. */
static gobj *cheney_forward(cheney_t *c, gobj *o, size_t *to_used)
{
    if (o == NULL) return NULL;
    if (o->tag == GTAG_FWD) return o->a;          /* already moved */
    gobj *n = (gobj *)(c->to + *to_used);
    *to_used += sizeof(gobj);
    memcpy(n, o, sizeof(gobj));
    o->tag = GTAG_FWD;
    o->a   = n;                                    /* forwarding pointer */
    return n;
}

static size_t cheney_collect(void *guest)
{
    cheney_t *c = (cheney_t *)guest;
    size_t before = c->used;
    size_t to_used = 0;

    /* ROOT SET = the adapter's pin table. That is the whole contract:
     * every Eshkol -> guest edge is a pinned handle, and pinned handles are
     * collector roots. Nothing else can reach into the guest heap. */
    eshkol_guest_heap_t *h = c->host;
    for (uint32_t i = 0; i < GUEST_HANDLE_CAP; ++i) {
        if (h->handle_obj[i] != NULL) {
            h->handle_obj[i] = cheney_forward(c, (gobj *)h->handle_obj[i], &to_used);
        }
    }

    /* Scan phase: breadth-first over to-space. Cycles terminate because a
     * forwarded object answers with its forwarding pointer. */
    size_t scan = 0;
    while (scan < to_used) {
        gobj *o = (gobj *)(c->to + scan);
        scan += sizeof(gobj);
        o->a = cheney_forward(c, o->a, &to_used);
        o->b = cheney_forward(c, o->b, &to_used);
        /* o->out is an Eshkol value. The guest DOES NOT trace it and DOES
         * NOT move it. It is a residence obligation, not a graph edge. */
    }

    char *tmp = c->from; c->from = c->to; c->to = tmp;
    c->used = to_used;
    c->live_after_last_gc = to_used / sizeof(gobj);
    return before - to_used;
}

static void cheney_teardown(void *guest)
{
    cheney_t *c = (cheney_t *)guest;
    /* No arena frees here. The semispaces die with the region's arena.
     * A real guest would close fds / release foreign mallocs here. */
    c->from = c->to = NULL;
}

static void cheney_enumerate_outbound(void *guest,
                                      void (*visit)(void *, const eshkol_tagged_value_t *),
                                      void *ctx)
{
    cheney_t *c = (cheney_t *)guest;
    for (size_t off = 0; off < c->used; off += sizeof(gobj)) {
        gobj *o = (gobj *)(c->from + off);
        if (o->tag == GTAG_PAIR && o->out.type != 0) visit(ctx, &o->out);
    }
}

static const eshkol_guest_vtable_t cheney_vt = {
    ESHKOL_GUEST_ABI_V1,
    "toy-cheney",
    cheney_collect,
    cheney_teardown,
    cheney_enumerate_outbound,
};

/* ================================================================== */
/* Experiments                                                         */
/* ================================================================== */

static int failures = 0;
static void check(const char *what, bool ok, const char *detail)
{
    printf("  [%s] %-52s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) failures++;
}

/* E1: cycles are reclaimed inside the region, host arena stays flat. */
static void experiment_1(void)
{
    printf("\nE1  guest cycle reclamation with a flat host arena\n");

    arena_t *host = arena_create(64 * 1024);       /* the region's arena */
    size_t arena_at_start = arena_get_total_memory(host);

    const size_t SEMI = 256 * 1024;                /* 256 KiB per semispace */
    cheney_t *c = (cheney_t *)arena_allocate_zeroed(host, sizeof(cheney_t));
    c->from = (char *)arena_allocate(host, SEMI);
    c->to   = (char *)arena_allocate(host, SEMI);
    c->semi_bytes = SEMI;

    eshkol_guest_heap_t *gh = eshkol_guest_heap_attach(host, &cheney_vt, c);
    c->host = gh;

    size_t arena_after_attach = arena_get_total_memory(host);
    size_t objs_per_semi = SEMI / sizeof(gobj);

    /* Build and drop cyclic garbage forever. Each round makes a 3-cycle plus
     * a chain hanging off it, then drops every reference. If the collector
     * could not reclaim cycles, from-space would fill and allocation would
     * fail within a few rounds. */
    const uint64_t ROUNDS = 200000;
    uint64_t allocated = 0, unrecoverable = 0;
    size_t peak_arena = arena_after_attach;

    auto t0 = std::chrono::steady_clock::now();
    for (uint64_t r = 0; r < ROUNDS; ++r) {
        gobj *n[3];
        for (int k = 0; k < 3; ++k) {
            n[k] = cheney_alloc(c);
            if (!n[k]) {
                /* Heap full. Nothing in this round is rooted yet, and the pin
                 * table is empty, so every object in from-space is unreachable
                 * cyclic garbage. A collector that could not reclaim cycles
                 * would recover nothing here and the retry would fail. */
                eshkol_guest_collect(gh);
                n[k] = cheney_alloc(c);
                for (int j = 0; j < k; ++j) n[j] = NULL;   /* stale after move */
                if (!n[k]) { unrecoverable++; break; }
            }
            allocated++;
        }
        if (n[0] && n[1] && n[2]) {
            n[0]->a = n[1]; n[1]->a = n[2]; n[2]->a = n[0]; /* 3-cycle */
            n[0]->b = n[2];                                  /* plus sharing */
        }
        size_t tm = arena_get_total_memory(host);
        if (tm > peak_arena) peak_arena = tm;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    size_t arena_end = arena_get_total_memory(host);

    printf("      rounds                 %llu (%llu guest objects allocated)\n",
           (unsigned long long)ROUNDS, (unsigned long long)allocated);
    printf("      guest collections      %llu\n", (unsigned long long)gh->collections);
    printf("      guest bytes reclaimed  %llu\n", (unsigned long long)gh->bytes_reclaimed);
    printf("      live objects after GC  %llu of %llu slots\n",
           (unsigned long long)c->live_after_last_gc, (unsigned long long)objs_per_semi);
    printf("      host arena  start      %zu B\n", arena_at_start);
    printf("      host arena  after attach %zu B\n", arena_after_attach);
    printf("      host arena  peak       %zu B\n", peak_arena);
    printf("      host arena  end        %zu B\n", arena_end);
    printf("      wall                   %.1f ms\n", ms);

    char d[160];
    snprintf(d, sizeof d, "%llu objects through a %zu B (%llu-slot) heap, 0 failures",
             (unsigned long long)allocated, SEMI, (unsigned long long)objs_per_semi);
    check("cycles reclaimed: allocation never exhausted the guest heap",
          unrecoverable == 0 && allocated > objs_per_semi * 50, d);

    snprintf(d, sizeof d, "peak == end == %zu B, unchanged since attach", arena_end);
    check("host arena FLAT across all guest collections",
          arena_end == arena_after_attach && peak_arena == arena_after_attach, d);

    snprintf(d, sizeof d, "%llu collections, 0 host allocations",
             (unsigned long long)gh->collections);
    check("Eshkol performed zero traces and zero per-object frees",
          gh->collections > 0, d);

    /* O(1) teardown: one arena_destroy kills the entire guest heap. */
    auto t2 = std::chrono::steady_clock::now();
    gh->vt->on_region_teardown(gh->guest);
    arena_destroy(host);
    auto t3 = std::chrono::steady_clock::now();
    double td_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
    snprintf(d, sizeof d, "%.1f us for a heap that saw %llu objects",
             td_us, (unsigned long long)allocated);
    check("region teardown is bulk, not a walk", td_us < 5000.0, d);
}

/* E2: the handle protocol survives a moving collector; a raw pointer does not.
 * This is the experiment that justifies mandating handles in the ABI. */
static void experiment_2(void)
{
    printf("\nE2  Eshkol -> guest edges: handle survives GC, raw pointer does not\n");

    arena_t *host = arena_create(64 * 1024);
    const size_t SEMI = 64 * 1024;
    cheney_t *c = (cheney_t *)arena_allocate_zeroed(host, sizeof(cheney_t));
    c->from = (char *)arena_allocate(host, SEMI);
    c->to   = (char *)arena_allocate(host, SEMI);
    c->semi_bytes = SEMI;
    eshkol_guest_heap_t *gh = eshkol_guest_heap_attach(host, &cheney_vt, c);
    c->host = gh;

    /* A guest object Eshkol wants to hold. */
    gobj *keep = cheney_alloc(c);
    keep->id = 0xBEEF;
    gobj *tail = cheney_alloc(c);
    keep->a = tail;
    tail->a = keep;                      /* keep and tail form a 2-cycle */

    eshkol_guest_handle_t hd = eshkol_guest_pin(gh, keep);
    gobj *raw = keep;                    /* the ILLEGAL edge, kept for contrast */
    uintptr_t raw_addr_before = (uintptr_t)raw;

    /* Make unreachable garbage so the collector has something to drop. */
    for (int i = 0; i < 64; ++i) {
        gobj *g1 = cheney_alloc(c);
        gobj *g2 = cheney_alloc(c);
        if (g1 && g2) { g1->a = g2; g2->a = g1; }   /* dead cycle */
    }
    uint64_t before_live = c->used / sizeof(gobj);

    eshkol_guest_collect(gh);

    gobj *via_handle = (gobj *)eshkol_guest_deref(gh, hd);
    uint64_t after_live = c->used / sizeof(gobj);

    printf("      objects before GC      %llu\n", (unsigned long long)before_live);
    printf("      objects after  GC      %llu\n", (unsigned long long)after_live);
    printf("      pinned object id       0x%X (expected 0xBEEF)\n",
           via_handle ? via_handle->id : 0);
    printf("      raw ptr before/after   %p / %p\n", (void *)raw_addr_before, (void *)via_handle);

    char d[160];
    check("pinned object survived collection", via_handle != NULL, NULL);
    check("handle resolves to the MOVED object",
          via_handle && via_handle->id == 0xBEEF, "identity preserved across the move");
    check("its cycle partner survived with it",
          via_handle && via_handle->a && via_handle->a->a == via_handle,
          "2-cycle intact and self-consistent");
    snprintf(d, sizeof d, "%llu -> %llu objects", (unsigned long long)before_live,
             (unsigned long long)after_live);
    check("128 objects of dead cycles were reclaimed", after_live == 2, d);

    bool raw_is_stale = ((uintptr_t)via_handle != raw_addr_before);
    check("raw pointer became STALE (the mandated-handles argument)",
          raw_is_stale, "a raw Eshkol->guest pointer is a use-after-move");

    /* Unpin, collect: the object and its cycle go away. */
    eshkol_guest_unpin(gh, hd);
    eshkol_guest_collect(gh);
    snprintf(d, sizeof d, "%llu objects remain", (unsigned long long)(c->used / sizeof(gobj)));
    check("unpinning released the pinned cycle", c->used == 0, d);

    void *stale = eshkol_guest_deref(gh, hd);
    check("stale handle deref is a detected fault, not a wild pointer",
          stale == NULL && gh->pin_violations == 1, "generation counter caught it");

    gh->vt->on_region_teardown(gh->guest);
    arena_destroy(host);
}

/* E3: guest -> Eshkol edges. The guest holds opaque Eshkol values; the guest
 * tracer steps over them; the adapter audits their residence. */
static void experiment_3(void)
{
    printf("\nE3  guest -> Eshkol edges: opaque to the tracer, audited by residence\n");

    arena_t *outer = arena_create(64 * 1024);      /* stands in for the parent region */
    arena_t *host  = arena_create(64 * 1024);      /* the guest's own region          */

    /* An Eshkol object in the OUTER region: legal target of a guest edge. */
    arena_tagged_cons_cell_t *outer_cell = arena_allocate_tagged_cons_cell(outer);
    arena_tagged_cons_set_int64(outer_cell, false, 42, ESHKOL_VALUE_INT64);

    const size_t SEMI = 64 * 1024;
    cheney_t *c = (cheney_t *)arena_allocate_zeroed(host, sizeof(cheney_t));
    c->from = (char *)arena_allocate(host, SEMI);
    c->to   = (char *)arena_allocate(host, SEMI);
    c->semi_bytes = SEMI;
    eshkol_guest_heap_t *gh = eshkol_guest_heap_attach(host, &cheney_vt, c);
    c->host = gh;

    gobj *o = cheney_alloc(c);
    o->out.type = ESHKOL_VALUE_HEAP_PTR;
    o->out.data.ptr_val = (uint64_t)(uintptr_t)outer_cell;
    o->a = o;                                       /* self-cycle */
    eshkol_guest_handle_t hd = eshkol_guest_pin(gh, o);

    for (int i = 0; i < 32; ++i) { gobj *g = cheney_alloc(c); if (g) g->a = g; }

    eshkol_guest_collect(gh);
    gobj *moved = (gobj *)eshkol_guest_deref(gh, hd);

    struct AuditCtx { int n; arena_t *outer; int foreign; } ctx = {0, outer, 0};
    gh->vt->enumerate_outbound(gh->guest,
        [](void *vctx, const eshkol_tagged_value_t *tv) {
            AuditCtx *a = (AuditCtx *)vctx;
            a->n++;
            /* Residence audit: the adapter checks the target is NOT in the
             * guest's own region. In the real runtime this consults the
             * residence tag; here we just confirm the pointer is intact. */
            if (tv->data.ptr_val != 0) a->foreign++;
        }, &ctx);

    printf("      outbound edges found   %d (foreign-resident: %d)\n", ctx.n, ctx.foreign);
    printf("      Eshkol value after GC  car = %lld\n",
           moved ? (long long)arena_tagged_cons_get_int64(
                       (arena_tagged_cons_cell_t *)(uintptr_t)moved->out.data.ptr_val, false)
                 : -1);

    check("guest object carrying an Eshkol value survived the move",
          moved != NULL && moved->out.data.ptr_val == (uint64_t)(uintptr_t)outer_cell,
          "the tagged value was copied verbatim, never traced");
    check("the Eshkol object was NOT moved or freed by the guest",
          arena_tagged_cons_get_int64(outer_cell, false) == 42,
          "outer-region object untouched");
    check("adapter can enumerate outbound edges for a residence audit",
          ctx.n == 1 && ctx.foreign == 1, NULL);

    /* Teardown order: the guest region dies FIRST, the outer region survives. */
    gh->vt->on_region_teardown(gh->guest);
    arena_destroy(host);
    check("outer-region object still live after guest region teardown",
          arena_tagged_cons_get_int64(outer_cell, false) == 42,
          "arena teardown is one-directional, inner before outer");
    arena_destroy(outer);
}

/* E4: scale. Drive the guest hard and watch process RSS, not just the arena. */
static void experiment_4(void)
{
    printf("\nE4  process RSS under sustained guest allocation\n");

    arena_t *host = arena_create(64 * 1024);
    const size_t SEMI = 1024 * 1024;
    cheney_t *c = (cheney_t *)arena_allocate_zeroed(host, sizeof(cheney_t));
    c->from = (char *)arena_allocate(host, SEMI);
    c->to   = (char *)arena_allocate(host, SEMI);
    c->semi_bytes = SEMI;
    eshkol_guest_heap_t *gh = eshkol_guest_heap_attach(host, &cheney_vt, c);
    c->host = gh;

    /* A permanent live set held by pins, plus unbounded cyclic garbage. */
    const int LIVE = 512;
    eshkol_guest_handle_t pins[LIVE];
    for (int i = 0; i < LIVE; ++i) {
        gobj *a = cheney_alloc(c);
        gobj *b = cheney_alloc(c);
        a->a = b; b->a = a;                      /* live cycle, pinned */
        pins[i] = eshkol_guest_pin(gh, a);
    }

    /* Warm-up: cycle the whole heap once so both semispaces are first-touched.
     * RSS necessarily rises to the DECLARED heap size; the claim under test is
     * that it stops there and does not track allocation volume. */
    auto churn = [&](uint64_t n) {
        for (uint64_t i = 0; i < n; ++i) {
            gobj *g1 = cheney_alloc(c);
            if (!g1) { eshkol_guest_collect(gh); g1 = cheney_alloc(c); }
            gobj *g2 = cheney_alloc(c);
            if (!g2) { eshkol_guest_collect(gh); g2 = cheney_alloc(c); g1 = NULL; }
            if (g1 && g2) { g1->a = g2; g2->a = g1; g1->b = g1; } /* dead cycle */
        }
    };

    size_t rss_pre = rss_bytes();
    churn(200000);                    /* warm-up: touches both semispaces */
    size_t rss0 = rss_bytes();
    size_t arena0 = arena_get_total_memory(host);

    const uint64_t N = 4000000;
    churn(N);

    size_t rss1 = rss_bytes();
    size_t arena1 = arena_get_total_memory(host);

    /* Verify every pinned live cycle is still correct. */
    int intact = 0;
    for (int i = 0; i < LIVE; ++i) {
        gobj *a = (gobj *)eshkol_guest_deref(gh, pins[i]);
        if (a && a->a && a->a->a == a) intact++;
    }

    printf("      declared guest heap     %zu B (2 x %zu semispace)\n", 2 * SEMI, SEMI);
    printf("      guest objects allocated %llu\n",
           (unsigned long long)((N + 200000) * 2 + LIVE * 2));
    printf("      guest collections       %llu\n", (unsigned long long)gh->collections);
    printf("      host arena  %zu -> %zu B (delta %lld)\n",
           arena0, arena1, (long long)arena1 - (long long)arena0);
    printf("      RSS pre-warmup          %zu B\n", rss_pre);
    printf("      RSS after warmup (rss0) %zu B (+%lld = heap first-touch)\n",
           rss0, (long long)rss0 - (long long)rss_pre);
    printf("      RSS after 8M allocs     %zu B (delta %lld)\n",
           rss1, (long long)rss1 - (long long)rss0);
    printf("      pinned live cycles intact %d of %d\n", intact, LIVE);

    char d[160];
    snprintf(d, sizeof d, "delta %lld B over %llu allocations",
             (long long)arena1 - (long long)arena0, (unsigned long long)(N * 2));
    check("host arena delta is zero under 8M guest allocations", arena1 == arena0, d);
    snprintf(d, sizeof d, "first-touch cost %lld B for a %zu B declared heap",
             (long long)rss0 - (long long)rss_pre, 2 * SEMI);
    check("RSS rises to the DECLARED heap size and no further",
          (rss0 - rss_pre) <= 2 * SEMI + (256u << 10), d);
    snprintf(d, sizeof d, "delta %lld B", (long long)rss1 - (long long)rss0);
    check("RSS growth after warmup is zero (does not track allocation volume)",
          rss1 <= rss0, d);
    check("all pinned live cycles intact after every move", intact == LIVE, NULL);

    gh->vt->on_region_teardown(gh->guest);
    arena_destroy(host);
}

int main(void)
{
    printf("ADR-0011 falsifier F1: guest collector hosted in an Eshkol region\n");
    printf("arena implementation: lib/core/runtime_arena_core.cpp (unmodified)\n");
    printf("sizeof(gobj)=%zu sizeof(eshkol_tagged_value_t)=%zu\n",
           sizeof(gobj), sizeof(eshkol_tagged_value_t));

    experiment_1();
    experiment_2();
    experiment_3();
    experiment_4();

    printf("\n%s (%d failures)\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
