// Fixed-condition identity and ordinary raise semantics. Only POD locals cross
// setjmp; this fixture does not claim general allocation-disabled Scheme unwind.
#include "../../lib/core/arena_memory.h"
#include "../../lib/core/runtime_region_promotion_internal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <setjmp.h>

extern "C" void eshkol_set_raised_value(const eshkol_tagged_value_t*);
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t*);
using Value = eshkol_tagged_value_t;

static void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}
static Value integer(int64_t n) {
    Value value{}; value.type = ESHKOL_VALUE_INT64; value.data.int_val = n;
    return value;
}
static Value caught(int condition) {
    jmp_buf handler;
    eshkol_push_exception_handler(&handler);
    eshkol_promotion_test_emergency_reset();
    if (setjmp(handler) == 0) eshkol_runtime_emergency_raise_v1(condition);
    Value value{};
    eshkol_get_raised_value(&value);
    check(value.type == ESHKOL_VALUE_HEAP_PTR && !value.flags && !value.reserved,
          "canonical emergency tag");
    check(reinterpret_cast<eshkol_exception_t*>(value.data.ptr_val) == g_current_exception,
          "tagged and current exception identities agree");
    check(eshkol_promotion_test_emergency_transfers() == 1, "one emergency transfer");
    eshkol_pop_exception_handler();
    return value;
}
static void returns_normally(const Value* value) {
    jmp_buf handler;
    eshkol_push_exception_handler(&handler);
    eshkol_promotion_test_emergency_reset();
    if (setjmp(handler) != 0) check(false, "noncanonical value must not rethrow");
    eshkol_runtime_emergency_rethrow_if_v1(value);
    check(eshkol_promotion_test_emergency_transfers() == 0, "no reserved transfer");
    eshkol_pop_exception_handler();
}
static void rethrows(const Value* value, bool direct) {
    jmp_buf handler;
    eshkol_push_exception_handler(&handler);
    eshkol_promotion_test_emergency_reset();
    const Value stale = integer(971);
    eshkol_set_raised_value(&stale);
    if (setjmp(handler) == 0) {
        if (direct) eshkol_raise(reinterpret_cast<eshkol_exception_t*>(value->data.ptr_val));
        else eshkol_runtime_emergency_rethrow_if_v1(value);
        check(false, "reserved identity must rethrow");
    }
    Value output{};
    eshkol_get_raised_value(&output);
    check(!std::memcmp(&output, value, sizeof output), "rethrow preserves canonical identity");
    check(eshkol_promotion_test_emergency_transfers() == 1, "one exact rethrow transfer");
    eshkol_pop_exception_handler();
}
static void ordinary_raise() {
    auto* exception = eshkol_make_exception_with_header(ESHKOL_EXCEPTION_USER_DEFINED, "ordinary");
    check(exception, "ordinary exception allocation");
    Value tagged{}; tagged.type = ESHKOL_VALUE_HEAP_PTR;
    tagged.data.ptr_val = reinterpret_cast<uintptr_t>(exception);
    returns_normally(&tagged);
    const Value original = integer(492);
    jmp_buf handler;
    eshkol_push_exception_handler(&handler);
    eshkol_promotion_test_emergency_reset();
    if (setjmp(handler) == 0) {
        eshkol_set_raised_value(&original);
        eshkol_raise(exception);
        check(false, "ordinary raise transfers");
    }
    Value output{}; eshkol_get_raised_value(&output);
    check(!std::memcmp(&output, &original, sizeof output), "ordinary raised operand unchanged");
    check(g_current_exception == exception, "ordinary current exception unchanged");
    check(eshkol_promotion_test_emergency_transfers() == 0, "ordinary raise not reserved");
    eshkol_pop_exception_handler();
}
int main() {
    check(get_global_arena_shared(), "global arena");
    __repl_shared_arena.store(get_global_arena_shared());
    Value identities[5]{};
    for (int condition = 1; condition <= 5; ++condition) {
        identities[condition - 1] = caught(condition);
        const Value again = caught(condition);
        check(!std::memcmp(&again, &identities[condition - 1], sizeof again), "stable identity");
        for (int prior = 0; prior < condition - 1; ++prior)
            check(identities[prior].data.ptr_val != again.data.ptr_val, "distinct conditions");
        auto* exception = reinterpret_cast<eshkol_exception_t*>(again.data.ptr_val);
        const eshkol_exception_t before = *exception;
        const eshkol_object_header_t header = *ESHKOL_GET_HEADER(exception);
        check(header.subtype == HEAP_SUBTYPE_EXCEPTION && header.size == sizeof *exception,
              "valid static exception header");
        check(eshkol_error_object_p(&again) != 0, "reserved value remains error object");
        const Value irritant = integer(4);
        eshkol_exception_add_irritant(exception, irritant);
        eshkol_exception_add_irritant_ptr(exception, &irritant);
        eshkol_exception_set_location(exception, 1, 2, "must not attach");
        check(!std::memcmp(exception, &before, sizeof before) &&
              !std::memcmp(ESHKOL_GET_HEADER(exception), &header, sizeof header),
              "reserved metadata remains immutable");
        rethrows(&again, false);
        rethrows(&again, true);
    }
    for (int condition : {0, -1, 6, 2147483647}) {
        const Value invalid = caught(condition);
        check(invalid.data.ptr_val == identities[3].data.ptr_val, "invalid selector maps to four");
    }
    returns_normally(nullptr);
    Value altered = identities[0]; altered.flags = 1; returns_normally(&altered);
    altered = identities[0]; altered.reserved = 1; returns_normally(&altered);
    altered = identities[0]; altered.type = ESHKOL_VALUE_INT64; returns_normally(&altered);
    altered = {}; returns_normally(&altered);
    struct Copy { eshkol_object_header_t header; eshkol_exception_t exception; } copy;
    copy.header = *ESHKOL_GET_HEADER(reinterpret_cast<void*>(identities[0].data.ptr_val));
    copy.exception = *reinterpret_cast<eshkol_exception_t*>(identities[0].data.ptr_val);
    altered = identities[0]; altered.data.ptr_val = reinterpret_cast<uintptr_t>(&copy.exception);
    returns_normally(&altered);
    ordinary_raise();
    std::puts("PASS native fixed emergency identity and ordinary raise semantics");
}
