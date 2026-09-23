// Region promotion is all-or-nothing (#713, ADR-0001 "All-or-nothing
// promotion").
//
// A store of a region-young value into a longer-lived destination promotes the
// value's whole reachable graph out of the region first. This test makes that
// promotion fail at every point it can fail and checks the contract each time:
//
//   * the store does not happen: the barrier's output is untouched;
//   * a catchable allocation error is raised;
//   * the destination arena's bump pointer is back where it was;
//   * the forwarding relation holds nothing from the failed attempt, so a later
//     successful promotion builds a complete graph rather than reusing a
//     half-built copy;
//   * after a success, no reachable edge points into the young region.
//
// Failure is injected the way the issue's reproducer does it: the destination
// arena is marked bounded and its remaining capacity set exactly. The last case
// uses the other allocator failure, the operating system refusing a new arena
// block, on an ordinary unbounded destination.

#include "../../lib/core/arena_memory.h"

#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// The backing-block case asks malloc for an impossible size on purpose; under
// AddressSanitizer that must return NULL rather than abort.
extern "C" const char* __asan_default_options() {
    return "allocator_may_return_null=1";
}

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

bool arena_holds(const arena_t* a, const void* p) {
    const auto* q = static_cast<const uint8_t*>(p);
    for (const arena_block_t* b = a->current_block; b; b = b->next) {
        if (q >= b->memory && q < b->memory + b->used) return true;
    }
    return false;
}

eshkol_tagged_value_t heap_value(void* p) {
    eshkol_tagged_value_t v{};
    v.type = ESHKOL_VALUE_HEAP_PTR;
    v.data.ptr_val = reinterpret_cast<uint64_t>(p);
    return v;
}

eshkol_tagged_value_t int_value(int64_t n) {
    eshkol_tagged_value_t v{};
    v.type = ESHKOL_VALUE_INT64;
    v.data.int_val = n;
    return v;
}

char* make_string(arena_t* a, const char* text) {
    const size_t n = std::strlen(text);
    char* s = arena_allocate_string_with_header(a, n);
    std::memcpy(s, text, n + 1);
    return s;
}

eshkol_tagged_value_t* make_vector(arena_t* a, int64_t n) {
    void* v = arena_allocate_vector_with_header(a, (size_t)n);
    *static_cast<int64_t*>(v) = n;
    auto* slots = reinterpret_cast<eshkol_tagged_value_t*>(static_cast<uint8_t*>(v) + 8);
    for (int64_t i = 0; i < n; ++i) slots[i] = int_value(0);
    return slots;
}

void* vector_of(eshkol_tagged_value_t* slots) {
    return reinterpret_cast<uint8_t*>(slots) - 8;
}

// Every heap edge reachable from @p v through conses, vectors and strings;
// counts the ones that point into @p young.
size_t young_edges(const eshkol_tagged_value_t& v, const arena_t* young,
                   std::vector<const void*>& seen) {
    if (v.type != ESHKOL_VALUE_HEAP_PTR || v.data.ptr_val == 0) return 0;
    void* p = reinterpret_cast<void*>(v.data.ptr_val);
    size_t n = arena_holds(young, p) ? 1 : 0;
    for (const void* s : seen) if (s == p) return n;
    seen.push_back(p);
    const uint8_t sub = ESHKOL_GET_HEADER(p)->subtype;
    if (sub == HEAP_SUBTYPE_CONS) {
        auto* c = static_cast<arena_tagged_cons_cell_t*>(p);
        n += young_edges(c->car, young, seen);
        n += young_edges(c->cdr, young, seen);
    } else if (sub == HEAP_SUBTYPE_VECTOR) {
        const int64_t len = *static_cast<int64_t*>(p);
        auto* slots = reinterpret_cast<eshkol_tagged_value_t*>(static_cast<uint8_t*>(p) + 8);
        for (int64_t i = 0; i < len; ++i) n += young_edges(slots[i], young, seen);
    }
    return n;
}

size_t young_edges(const eshkol_tagged_value_t& v, const arena_t* young) {
    std::vector<const void*> seen;
    return young_edges(v, young, seen);
}

// Cap @p a at exactly @p remaining free bytes in its current block.
struct CapacityGuard {
    arena_t* a;
    bool bounded;
    size_t used;
    CapacityGuard(arena_t* arena, size_t remaining)
        : a(arena), bounded(arena->bounded), used(arena->current_block->used) {
        a->bounded = true;
        a->current_block->used = a->current_block->size - remaining;
    }
    ~CapacityGuard() {
        a->bounded = bounded;
        a->current_block->used = used;
    }
};

jmp_buf g_landing;

// Run one barrier store under a handler. Returns true when it raised.
bool store_raises(eshkol_tagged_value_t* out, const void* dst,
                  const eshkol_tagged_value_t* value) {
    eshkol_push_exception_handler(&g_landing);
    if (setjmp(g_landing) == 0) {
        eshkol_region_write_barrier_into(out, dst, value);
        eshkol_pop_exception_handler();
        return false;
    }
    eshkol_pop_exception_handler();
    return true;
}

bool raised_allocation_error() {
    return g_current_exception && g_current_exception->message &&
           std::strstr(g_current_exception->message, "region promotion") != nullptr &&
           std::strstr(g_current_exception->message, "out of memory") != nullptr;
}

// Case 1: the issue's reproducer. A string promoted into the global arena,
// which is at capacity.
void reporter_case() {
    arena_t* root = get_global_arena_shared();
    eshkol_region_t* region = region_create("promotion-probe", 4096);
    region_push(region);
    char* source = make_string(region->arena, "test");
    const eshkol_tagged_value_t value = heap_value(source);
    eshkol_tagged_value_t output = int_value(73);

    bool raised;
    size_t used_after;
    {
        CapacityGuard cap(root, 0);
        const size_t used_before = root->current_block->used;
        raised = store_raises(&output, nullptr, &value);
        used_after = root->current_block->used;
        check(used_after == used_before, "reporter: destination bump pointer not rewound");
    }
    check(raised, "reporter: failed promotion did not raise");
    check(raised_allocation_error(), "reporter: raised condition is not the allocation error");
    check(output.type == ESHKOL_VALUE_INT64 && output.data.int_val == 73,
          "reporter: barrier wrote its output after a failed promotion");

    // With capacity back, the same store succeeds and is fully promoted.
    raised = store_raises(&output, nullptr, &value);
    check(!raised, "reporter: store raised with capacity available");
    check(output.type == ESHKOL_VALUE_HEAP_PTR && output.data.ptr_val != value.data.ptr_val &&
              !arena_holds(region->arena, reinterpret_cast<void*>(output.data.ptr_val)) &&
              std::strcmp(reinterpret_cast<const char*>(output.data.ptr_val), "test") == 0,
          "reporter: retry did not promote the string");
    region_pop();
}

// Case 2: the parent fits, its child does not. The parent copy must not
// survive in the forwarding relation.
void partial_graph_case() {
    eshkol_region_t* outer = region_create("outer", 1 << 16);
    region_push(outer);
    eshkol_tagged_value_t* holder = make_vector(outer->arena, 1);
    eshkol_region_t* young = region_create("young", 4096);
    region_push(young);

    eshkol_tagged_value_t* parent = make_vector(young->arena, 1);
    parent[0] = heap_value(make_string(young->arena, "child string"));
    const eshkol_tagged_value_t value = heap_value(vector_of(parent));
    eshkol_tagged_value_t output = int_value(7);

    // Room for the one-slot parent vector (32 bytes) and nothing more.
    bool raised;
    {
        CapacityGuard cap(outer->arena, 32);
        const size_t used_before = outer->arena->current_block->used;
        raised = store_raises(&output, holder, &value);
        check(outer->arena->current_block->used == used_before,
              "partial: destination bump pointer not rewound");
    }
    check(raised && raised_allocation_error(), "partial: failed child copy did not raise");
    check(output.type == ESHKOL_VALUE_INT64 && output.data.int_val == 7,
          "partial: barrier published a parent whose child was not promoted");

    raised = store_raises(&output, holder, &value);
    check(!raised, "partial: retry raised with capacity available");
    check(output.type == ESHKOL_VALUE_HEAP_PTR &&
              young_edges(output, young->arena) == 0,
          "partial: retry reused a half-built copy (young edge survived)");
    holder[0] = output;
    region_pop();
    region_pop();
}

// Case 3: a cons -> record -> vector graph with a shared edge and a cycle.
// Every copy prefix fails in turn; each failure must leave everything as it
// was, and the first success must be a complete graph.
void every_prefix_case() {
    eshkol_region_t* outer = region_create("outer", 1 << 16);
    region_push(outer);
    eshkol_tagged_value_t* holder = make_vector(outer->arena, 1);
    eshkol_region_t* young = region_create("young", 1 << 14);
    region_push(young);

    char* token = make_string(young->arena, "token");
    eshkol_tagged_value_t* three = make_vector(young->arena, 3);
    three[0] = heap_value(token);
    three[1] = heap_value(token);                       // shared edge
    three[2] = heap_value(make_string(young->arena, "tail"));
    eshkol_tagged_value_t* record = make_vector(young->arena, 2);
    record[0] = heap_value(vector_of(three));
    record[1] = heap_value(token);
    auto* cell = arena_allocate_cons_with_header(young->arena);
    cell->car = heap_value(vector_of(record));
    cell->cdr = heap_value(cell);                       // cycle
    three[2] = heap_value(cell);                        // and a back edge
    const eshkol_tagged_value_t value = heap_value(cell);

    size_t failures = 0;
    bool succeeded = false;
    for (size_t remaining = 0; remaining <= 1024 && !succeeded; remaining += 16) {
        eshkol_tagged_value_t output = int_value(-1);
        bool raised;
        {
            CapacityGuard cap(outer->arena, remaining);
            const size_t used_before = outer->arena->current_block->used;
            raised = store_raises(&output, holder, &value);
            if (raised) {
                check(outer->arena->current_block->used == used_before,
                      "prefix: destination bump pointer not rewound");
            }
        }
        if (raised) {
            ++failures;
            check(raised_allocation_error(), "prefix: raised condition is not the allocation error");
            check(output.type == ESHKOL_VALUE_INT64 && output.data.int_val == -1,
                  "prefix: barrier wrote its output after a failed promotion");
            continue;
        }
        succeeded = true;
        check(young_edges(output, young->arena) == 0,
              "prefix: promoted graph still has an edge into the young region");
        auto* copy = reinterpret_cast<arena_tagged_cons_cell_t*>(output.data.ptr_val);
        check(copy->cdr.data.ptr_val == output.data.ptr_val, "prefix: cycle not preserved");
        auto* rec = reinterpret_cast<eshkol_tagged_value_t*>(
            reinterpret_cast<uint8_t*>(copy->car.data.ptr_val) + 8);
        auto* thr = reinterpret_cast<eshkol_tagged_value_t*>(
            reinterpret_cast<uint8_t*>(rec[0].data.ptr_val) + 8);
        check(thr[0].data.ptr_val == thr[1].data.ptr_val &&
                  thr[0].data.ptr_val == rec[1].data.ptr_val,
              "prefix: shared edge not preserved");
        check(thr[2].data.ptr_val == output.data.ptr_val, "prefix: back edge not preserved");
        holder[0] = output;
    }
    check(succeeded, "prefix: promotion never succeeded");
    check(failures >= 6, "prefix: fewer failing prefixes than objects in the graph");
    std::printf("every_prefix: %zu failing prefixes, then a complete promotion\n", failures);
    region_pop();
    region_pop();
}

// Case 4: the range barrier used by vector-copy!: values are staged and
// promoted in one transaction; a failure leaves the staged values unchanged.
void range_case() {
    eshkol_region_t* outer = region_create("outer", 1 << 16);
    region_push(outer);
    eshkol_tagged_value_t* holder = make_vector(outer->arena, 2);
    eshkol_region_t* young = region_create("young", 4096);
    region_push(young);
    eshkol_tagged_value_t staged[2] = {
        heap_value(make_string(young->arena, "first")),
        heap_value(make_string(young->arena, "second")),
    };
    const eshkol_tagged_value_t original[2] = {staged[0], staged[1]};

    bool raised = false;
    {
        CapacityGuard cap(outer->arena, 16);   // room for the first string only
        eshkol_push_exception_handler(&g_landing);
        if (setjmp(g_landing) == 0) {
            eshkol_region_write_barrier_range(vector_of(holder), staged, 2);
        } else {
            raised = true;
        }
        eshkol_pop_exception_handler();
    }
    check(raised && raised_allocation_error(), "range: failed promotion did not raise");
    check(std::memcmp(staged, original, sizeof(staged)) == 0,
          "range: staged values changed after a failed promotion");
    region_pop();
    region_pop();
}

// Case 5: the ordinary backing-allocation path. The destination is unbounded;
// the copy needs a new arena block the operating system cannot provide.
void backing_block_case() {
    eshkol_region_t* outer = region_create("outer", 1 << 16);
    region_push(outer);
    eshkol_tagged_value_t* holder = make_vector(outer->arena, 1);
    eshkol_region_t* young = region_create("young", 4096);
    region_push(young);

    // A tensor whose element buffer claims 2^57 doubles: its copy asks the
    // destination for a 2^60-byte block, which malloc refuses. The buffer is
    // never read -- the allocation fails before any copy.
    eshkol_tensor_t* t = arena_allocate_tensor_with_header(young->arena);
    auto* dims = static_cast<uint64_t*>(arena_allocate_aligned(young->arena, 8, 8));
    dims[0] = (uint64_t)1 << 57;
    t->dimensions = dims;
    t->num_dimensions = 1;
    t->elements = static_cast<int64_t*>(arena_allocate_aligned(young->arena, 64, 16));
    t->total_elements = (uint64_t)1 << 57;
    t->dtype = 0;
    const eshkol_tagged_value_t value = heap_value(t);
    eshkol_tagged_value_t output = int_value(5);

    arena_block_t* const block_before = outer->arena->current_block;
    const size_t used_before = block_before->used;
    const bool raised = store_raises(&output, holder, &value);
    check(raised && raised_allocation_error(), "backing: failed block allocation did not raise");
    check(output.type == ESHKOL_VALUE_INT64 && output.data.int_val == 5,
          "backing: barrier wrote its output after a failed promotion");
    check(outer->arena->current_block == block_before &&
              block_before->used == used_before,
          "backing: destination arena not rewound");
    region_pop();
    region_pop();
}

// Case 6: parameter binding order (credit: Gabriel Kahen, #714). A parameter's
// value stack is malloc-owned, so a young value bound there is promoted first.
// When that promotion fails, the binding must not be published: `top` still
// names the previous binding, and reading the parameter answers it.
void parameter_case() {
    arena_t* root = get_global_arena_shared();
    eshkol_tagged_value_t base = int_value(11);
    void* param = eshkol_make_parameter(root, base);
    check(param != nullptr, "parameter: make-parameter outside a region failed");

    eshkol_region_t* young = region_create("young", 4096);
    region_push(young);
    const eshkol_tagged_value_t value = heap_value(make_string(young->arena, "young binding"));

    bool raised = false;
    {
        CapacityGuard cap(root, 0);
        eshkol_push_exception_handler(&g_landing);
        if (setjmp(g_landing) == 0) eshkol_parameter_push(param, value);
        else raised = true;
        eshkol_pop_exception_handler();
    }
    check(raised && raised_allocation_error(), "parameter: failed push did not raise");
    const eshkol_tagged_value_t now = eshkol_parameter_ref(param);
    check(now.type == ESHKOL_VALUE_INT64 && now.data.int_val == 11,
          "parameter: a failed push published an unwritten binding");

    raised = false;
    {
        CapacityGuard cap(root, 0);
        eshkol_push_exception_handler(&g_landing);
        if (setjmp(g_landing) == 0) (void)eshkol_make_parameter(root, value);
        else raised = true;
        eshkol_pop_exception_handler();
    }
    check(raised && raised_allocation_error(), "parameter: failed make-parameter did not raise");

    // With capacity, the same push binds the promoted value.
    eshkol_parameter_push(param, value);
    const eshkol_tagged_value_t bound = eshkol_parameter_ref(param);
    check(bound.type == ESHKOL_VALUE_HEAP_PTR &&
              !arena_holds(young->arena, reinterpret_cast<void*>(bound.data.ptr_val)) &&
              std::strcmp(reinterpret_cast<const char*>(bound.data.ptr_val), "young binding") == 0,
          "parameter: push with capacity did not bind the promoted value");
    region_pop();
}

// Case 7: the failpoint matrix (approach credit: Gabriel Kahen, #714). Every
// allocation site a promotion depends on fails in turn, at every occurrence:
// the destination copy, the OS providing a new arena block, the forwarding-map
// insert, the saved-bytes record and the inserted-keys record. The graph has a
// cycle, a shared edge, and a boxed tensor whose element buffer lives in the
// OUTER arena and so is rewritten in place (a saved-bytes record). The
// destination is unbounded with a full current block, so the first copy needs
// a new block. Each failure must leave: no output, the destination's block
// chain and bump pointer as they were, the forwarding relation as it was, the
// shared buffer's bytes as they were, and a catchable allocation error.
void failpoint_matrix_case() {
    static const char* const kSiteName[ESHKOL_ALLOC_FAILPOINT_COUNT] = {
        "destination copy", "arena block", "forwarding insert",
        "saved-bytes record", "inserted-keys record"};

    for (int site = 0; site < ESHKOL_ALLOC_FAILPOINT_COUNT; ++site) {
        eshkol_region_t* outer = region_create("outer", 1 << 16);
        region_push(outer);
        eshkol_tagged_value_t* holder = make_vector(outer->arena, 1);
        auto* shared_slots = static_cast<eshkol_tagged_value_t*>(
            arena_allocate_aligned(outer->arena, 2 * sizeof(eshkol_tagged_value_t), 16));
        eshkol_region_t* young = region_create("young", 1 << 14);
        region_push(young);

        char* token = make_string(young->arena, "token");
        shared_slots[0] = heap_value(token);
        shared_slots[1] = heap_value(make_string(young->arena, "element"));
        eshkol_tensor_t* t = arena_allocate_tensor_with_header(young->arena);
        auto* dims = static_cast<uint64_t*>(arena_allocate_aligned(young->arena, 8, 8));
        dims[0] = 2;
        t->dimensions = dims;
        t->num_dimensions = 1;
        t->elements = reinterpret_cast<int64_t*>(shared_slots);
        t->total_elements = 2;
        t->dtype = ESHKOL_TENSOR_DTYPE_BOXED;
        eshkol_tagged_value_t* three = make_vector(young->arena, 3);
        three[0] = heap_value(token);
        three[1] = heap_value(t);
        auto* cell = arena_allocate_cons_with_header(young->arena);
        cell->car = heap_value(vector_of(three));
        cell->cdr = heap_value(cell);
        three[2] = heap_value(cell);
        const eshkol_tagged_value_t value = heap_value(cell);
        eshkol_tagged_value_t shared_before[2] = {shared_slots[0], shared_slots[1]};

        size_t injected = 0;
        bool completed = false;
        for (uint64_t nth = 0; nth < 256 && !completed; ++nth) {
            arena_t* dst = outer->arena;
            const size_t used = dst->current_block->used;
            dst->current_block->used = dst->current_block->size;   // next copy needs a block
            arena_block_t* const block_before = dst->current_block;
            const size_t full = block_before->used;
            const size_t fwd_before = eshkol_region_forwarding_size(young);

            eshkol_tagged_value_t output = int_value(-7);
            eshkol_alloc_failpoint_arm(site, nth);
            const bool raised = store_raises(&output, holder, &value);
            const uint64_t hits = eshkol_alloc_failpoint_hits(site);
            eshkol_alloc_failpoint_disarm();

            if (raised) {
                ++injected;
                check(hits > nth, "failpoint: raised without reaching the armed site");
                check(raised_allocation_error(), "failpoint: raised condition is not the allocation error");
                check(output.type == ESHKOL_VALUE_INT64 && output.data.int_val == -7,
                      "failpoint: barrier wrote its output after a failed promotion");
                check(dst->current_block == block_before && block_before->used == full,
                      "failpoint: destination not rewound");
                check(eshkol_region_forwarding_size(young) == fwd_before,
                      "failpoint: forwarding relation changed by a failed promotion");
                check(std::memcmp(shared_slots, shared_before, sizeof(shared_before)) == 0,
                      "failpoint: shared buffer not restored");
                block_before->used = used;
                continue;
            }
            // The armed occurrence was never reached: the site is exhausted.
            completed = true;
            check(hits <= nth, "failpoint: armed occurrence reached but nothing raised");
            check(young_edges(output, young->arena) == 0,
                  "failpoint: completed promotion still has a young edge");
            holder[0] = output;
        }
        if (!completed || injected == 0) {
            std::fprintf(stderr, "FAIL: failpoint site '%s': %zu injections, completed=%d\n",
                         kSiteName[site], injected, (int)completed);
            ++g_failures;
        } else {
            std::printf("failpoint %-22s %zu failing occurrences, then a complete promotion\n",
                        kSiteName[site], injected);
        }
        region_pop();
        region_pop();
    }
}

}  // namespace

int main() {
    reporter_case();
    partial_graph_case();
    every_prefix_case();
    range_case();
    backing_block_case();
    parameter_case();
    failpoint_matrix_case();
    if (g_failures) {
        std::fprintf(stderr, "region_promotion_failure_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS: region promotion is all-or-nothing under allocation failure\n");
    return 0;
}
