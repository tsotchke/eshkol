#include "../../lib/core/arena_memory.h"
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <thread>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
static eshkol_tagged_value_t heap(void* p) {
    eshkol_tagged_value_t value{}; value.type = ESHKOL_VALUE_HEAP_PTR;
    value.data.ptr_val = reinterpret_cast<uint64_t>(p); return value;
}
static jmp_buf landing;
static bool compact_raises(int64_t owner) {
    eshkol_push_exception_handler(&landing);
    if (setjmp(landing) == 0) {
        eshkol_resident_compact(owner, 1);
        eshkol_pop_exception_handler(); return false;
    }
    eshkol_pop_exception_handler(); return true;
}
static bool export_raises(int64_t owner, arena_tagged_cons_cell_t* holder) {
    eshkol_push_exception_handler(&landing);
    if (setjmp(landing) == 0) {
        eshkol_resident_export(owner, holder);
        eshkol_pop_exception_handler(); return false;
    }
    eshkol_pop_exception_handler(); return true;
}
int main() {
    arena_t* outside = eshkol_current_arena();
    auto* cycle = arena_allocate_cons_with_header(outside);
    cycle->car = {}; cycle->car.type = ESHKOL_VALUE_INT64; cycle->car.data.int_val = 7;
    cycle->cdr = heap(cycle);
    const int64_t owner = eshkol_resident_open(65536);
    arena_tagged_cons_cell_t holder{}; holder.car = heap(cycle);
    const int64_t a = eshkol_resident_hold(owner, &holder);
    eshkol_resident_enter(owner); eshkol_resident_get(owner, a, &holder);
    auto* stored_cycle = reinterpret_cast<arena_tagged_cons_cell_t*>(holder.car.data.ptr_val);
    check(stored_cycle != cycle, "import copies caller graph");
    check(stored_cycle->cdr.data.ptr_val == holder.car.data.ptr_val, "import preserves cycle");
    auto* pair = arena_allocate_cons_with_header(eshkol_current_arena());
    pair->car = holder.car; pair->cdr = holder.car; holder.car = heap(pair);
    const int64_t b = eshkol_resident_hold(owner, &holder);
    holder = {}; eshkol_resident_leave(owner);
    check(eshkol_current_arena() == outside, "borrow restores allocation domain");
    check(eshkol_resident_stat(owner, 0) == 2, "two explicit retained roots");
    std::atomic<bool> foreign_refused{false};
    std::thread foreign([&] {
        jmp_buf local_landing;
        eshkol_push_exception_handler(&local_landing);
        if (setjmp(local_landing) == 0) (void)eshkol_resident_stat(owner, 0);
        else foreign_refused = true;
        eshkol_pop_exception_handler();
    });
    foreign.join();
    check(foreign_refused, "owner token cannot address another thread's table");

    // Inject every early graph-copy failure. Root publication and compaction
    // counters must remain unchanged; later successful movement still works.
    for (uint64_t nth = 0; nth < 2; ++nth) {
        const int64_t before = eshkol_resident_stat(owner, 2);
        eshkol_alloc_failpoint_arm(ESHKOL_ALLOC_FAILPOINT_EVAC_COPY, nth);
        const bool raised = compact_raises(owner);
        eshkol_alloc_failpoint_disarm();
        check(raised, "injected compaction failure raises");
        check(eshkol_resident_stat(owner, 2) == before, "failed compaction publishes nothing");
        check(eshkol_resident_stat(owner, 0) == 2, "failed compaction retains both handles");
    }
    eshkol_resident_compact(owner, 1);
    eshkol_resident_enter(owner);
    arena_tagged_cons_cell_t ha{}, hb{};
    eshkol_resident_get(owner, a, &ha); eshkol_resident_get(owner, b, &hb);
    auto* moved = reinterpret_cast<arena_tagged_cons_cell_t*>(ha.car.data.ptr_val);
    auto* moved_pair = reinterpret_cast<arena_tagged_cons_cell_t*>(hb.car.data.ptr_val);
    check(moved->car.data.int_val == 7, "moved graph preserves value");
    check(moved->cdr.data.ptr_val == ha.car.data.ptr_val, "compaction preserves cycle");
    check(moved_pair->car.data.ptr_val == ha.car.data.ptr_val &&
          moved_pair->cdr.data.ptr_val == ha.car.data.ptr_val, "one forwarding map preserves cross-snapshot sharing");
    const uint64_t original = hb.car.data.ptr_val;
    eshkol_alloc_failpoint_arm(ESHKOL_ALLOC_FAILPOINT_EVAC_COPY, 0);
    const bool refused = export_raises(owner, &hb);
    eshkol_alloc_failpoint_disarm();
    check(refused && hb.car.data.ptr_val == original, "failed export leaves borrowed root unchanged");
    const char* export_error = g_current_exception ? g_current_exception->message : nullptr;
    eshkol_resident_export(owner, &hb);
    ha = {}; eshkol_resident_leave(owner);
    eshkol_resident_release(owner, a); eshkol_resident_release(owner, b);
    eshkol_resident_compact(owner, 1);
    check(eshkol_resident_stat(owner, 0) == 0 && eshkol_resident_stat(owner, 1) == 0,
          "released roots leave no live arena payload after compaction");
    eshkol_resident_close(owner);
    check(export_error && std::strstr(export_error, "export"), "export refusal condition survives owner closure");
    auto* exported_pair = reinterpret_cast<arena_tagged_cons_cell_t*>(hb.car.data.ptr_val);
    auto* exported_cycle = reinterpret_cast<arena_tagged_cons_cell_t*>(exported_pair->car.data.ptr_val);
    check(exported_cycle->car.data.int_val == 7 && exported_cycle->cdr.data.ptr_val == exported_pair->car.data.ptr_val,
          "exported cyclic observation survives owner destruction");
    check(exported_pair->car.data.ptr_val == exported_pair->cdr.data.ptr_val,
          "export preserves aliasing inside observation");
    // Modern zero-capture procedures use LAMBDA_SEXPR with the declared closure
    // layout. Owned name/source data must move alongside the callable payload.
    auto* name = arena_allocate_string_with_header(outside, 5);
    std::memcpy(name, "named", 6);
    auto* closure = arena_allocate_closure_with_header(outside, 0, 0,
        reinterpret_cast<uint64_t>(cycle), 0, name);
    const int64_t callback_owner = eshkol_resident_open(65536);
    holder.car = heap(closure); holder.car.type = ESHKOL_VALUE_CALLABLE;
    const int64_t callback = eshkol_resident_hold(callback_owner, &holder);
    eshkol_resident_compact(callback_owner, 1);
    eshkol_resident_enter(callback_owner);
    eshkol_resident_get(callback_owner, callback, &holder);
    eshkol_resident_export(callback_owner, &holder);
    eshkol_resident_leave(callback_owner); eshkol_resident_close(callback_owner);
    auto* observed = reinterpret_cast<eshkol_closure_t*>(holder.car.data.ptr_val);
    auto* source = reinterpret_cast<arena_tagged_cons_cell_t*>(observed->sexpr_ptr);
    check(observed->env == nullptr && std::strcmp(observed->name, "named") == 0,
          "zero-capture closure name survives movement and owner destruction");
    check(source->car.data.int_val == 7 && source->cdr.data.ptr_val == observed->sexpr_ptr,
          "zero-capture closure source graph survives owner destruction");
    // LLVM can emit canonical mutable literals outside every arena. They must
    // still acquire writable, independently owned storage at the boundary.
    static const struct alignas(8) {
        eshkol_object_header_t header;
        char text[8];
    } literal{{HEAP_SUBTYPE_STRING, 0, 0, 4}, "abc"};
    const int64_t literal_owner = eshkol_resident_open(65536);
    holder.car = heap(const_cast<char*>(literal.text));
    const int64_t literal_root = eshkol_resident_hold(literal_owner, &holder);
    eshkol_resident_enter(literal_owner);
    eshkol_resident_get(literal_owner, literal_root, &holder);
    auto* imported_text = reinterpret_cast<char*>(holder.car.data.ptr_val);
    check(imported_text != literal.text, "static mutable literal is imported by value");
    if (imported_text != literal.text) imported_text[0] = 'X';
    eshkol_resident_leave(literal_owner);
    eshkol_resident_compact(literal_owner, 1);
    eshkol_resident_enter(literal_owner);
    eshkol_resident_get(literal_owner, literal_root, &holder);
    eshkol_resident_export(literal_owner, &holder);
    eshkol_resident_leave(literal_owner); eshkol_resident_close(literal_owner);
    check(std::strcmp(reinterpret_cast<char*>(holder.car.data.ptr_val), "Xbc") == 0 &&
          std::strcmp(literal.text, "abc") == 0,
          "static literal import/export remains isolated through movement and close");
    if (failures) return 1;
    std::puts("PASS: resident owner movement, sharing and rollback");
    return 0;
}
