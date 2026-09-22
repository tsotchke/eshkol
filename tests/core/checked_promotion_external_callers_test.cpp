// Real-engine external publication regression. Build with
// ESHKOL_PROMOTION_TESTING and -Wl,--wrap=malloc,--wrap=realloc,--wrap=free.
// The wrappers measure caller-owned parameter storage, not engine allocations.
#include "../../lib/core/runtime_region_promotion_internal.h"

#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef ESHKOL_PROMOTION_TESTING
#error This fixture requires the promotion engine's test-only failure hooks.
#endif

extern "C" {
void* eshkol_make_parameter(void*, eshkol_tagged_value_t);
void eshkol_parameter_push(void*, eshkol_tagged_value_t);
void eshkol_parameter_set(void*, eshkol_tagged_value_t);
void eshkol_parameter_set_converter(void*, eshkol_tagged_value_t);
void eshkol_clear_current_exception(void);
void* __real_malloc(size_t);
void* __real_realloc(void*, size_t);
void __real_free(void*);
}

namespace {
#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); std::abort(); \
} } while (false)

bool tracking = false;
void* initial_stack = nullptr;
unsigned stack_allocations = 0, stack_frees = 0, reallocations = 0;

// Inspect the existing private control layout to verify publication metadata.
struct Parameter {
    eshkol_tagged_value_t* stack;
    int top;
    int capacity;
    eshkol_tagged_value_t converter;
};

eshkol_tagged_value_t integer(int64_t value) {
    eshkol_tagged_value_t out{};
    out.type = ESHKOL_VALUE_INT64;
    out.data.int_val = value;
    return out;
}
eshkol_tagged_value_t pointer(void* value) {
    eshkol_tagged_value_t out{};
    out.type = ESHKOL_VALUE_HEAP_PTR;
    out.data.ptr_val = reinterpret_cast<uint64_t>(value);
    return out;
}
eshkol_tagged_value_t* slots(void* vector) {
    return reinterpret_cast<eshkol_tagged_value_t*>(
        static_cast<unsigned char*>(vector) + sizeof(int64_t));
}
void* make_vector(arena_t* arena, size_t count) {
    void* value = arena_allocate_vector_with_header(arena, count);
    CHECK(value);
    *static_cast<int64_t*>(value) = static_cast<int64_t>(count);
    for (size_t i = 0; i < count; ++i) slots(value)[i] = integer(-1);
    return value;
}
void* make_sequence(arena_t* arena, bool dual, size_t count) {
    if (!dual) return make_vector(arena, count);
    auto* tensor = arena_allocate_tensor_with_header(arena);
    CHECK(tensor);
    tensor->num_dimensions = 1;
    tensor->total_elements = count;
    tensor->dtype = ESHKOL_TENSOR_DTYPE_DUAL;
    tensor->dimensions = static_cast<uint64_t*>(arena_allocate(arena, sizeof(uint64_t)));
    CHECK(tensor->dimensions);
    *tensor->dimensions = count;
    tensor->elements = static_cast<int64_t*>(
        arena_allocate(arena, count * sizeof(eshkol_tagged_value_t)));
    CHECK(tensor->elements);
    auto* values = reinterpret_cast<eshkol_tagged_value_t*>(tensor->elements);
    for (size_t i = 0; i < count; ++i) values[i] = integer(-1);
    return tensor;
}
eshkol_tagged_value_t* sequence_slots(void* value, bool dual) {
    return dual ? reinterpret_cast<eshkol_tagged_value_t*>(
        static_cast<eshkol_tensor_t*>(value)->elements) : slots(value);
}
eshkol_tagged_value_t graph(arena_t* arena) {
    void* value = make_vector(arena, 2);
    char* text = arena_allocate_string_with_header(arena, 6);
    CHECK(text);
    std::memcpy(text, "nested", 7);
    slots(value)[0] = pointer(text);
    slots(value)[1] = pointer(text); // Cross-root and interior sharing survive.
    return pointer(value);
}
void check_graph(eshkol_tagged_value_t value, uint64_t old_address) {
    CHECK(value.type == ESHKOL_VALUE_HEAP_PTR);
    CHECK(value.data.ptr_val != old_address);
    auto* children = slots(reinterpret_cast<void*>(value.data.ptr_val));
    CHECK(children[0].data.ptr_val == children[1].data.ptr_val);
    CHECK(std::strcmp(reinterpret_cast<const char*>(children[0].data.ptr_val), "nested") == 0);
}

// Install inside the source region: the failed graph remains live for retry.
// No dynamic-wind callbacks or C++ owning objects cross this longjmp.
template<class Action>
void expect_failure(Action action, int site, int64_t index = 0) {
    std::jmp_buf handler;
    eshkol_push_exception_handler(&handler);
    stack_allocations = stack_frees = reallocations = 0;
    initial_stack = nullptr;
    if (setjmp(handler) == 0) {
        eshkol_promotion_test_arm(site, index);
        tracking = true;
        action();
        CHECK(false); // A failed promotion must never return to publication.
    }
    tracking = false;
    CHECK(g_current_exception);
    CHECK(ESHKOL_GET_HEADER(g_current_exception)->subtype == HEAP_SUBTYPE_EXCEPTION);
    eshkol_promotion_test_reset();
    eshkol_pop_exception_handler();
    eshkol_clear_current_exception();
}

void parameter_failures() {
    for (int site = 0; site < 5; ++site) {
        auto* param = static_cast<Parameter*>(eshkol_make_parameter(get_global_arena(), integer(0)));
        CHECK(param);
        for (int i = 1; i < 8; ++i) eshkol_parameter_push(param, integer(i));
        CHECK(param->top == 7 && param->capacity == 8);
        const Parameter before = *param;
        eshkol_tagged_value_t values[8];
        std::memcpy(values, param->stack, sizeof(values));
        auto* region = region_create("checked-parameter", 4096);
        CHECK(region);
        region_push(region);
        const auto incoming = graph(region->arena);
        const auto live_before = eshkol_promotion_test_snapshot();
        expect_failure([&] { (void)eshkol_make_parameter(get_global_arena(), incoming); }, site);
        CHECK(stack_allocations == 1 && stack_frees == 1 && reallocations == 0);
        expect_failure([&] { eshkol_parameter_push(param, incoming); }, site);
        CHECK(reallocations == 0);
        CHECK(std::memcmp(param, &before, sizeof(before)) == 0);
        CHECK(std::memcmp(param->stack, values, sizeof(values)) == 0);
        expect_failure([&] { eshkol_parameter_set(param, incoming); }, site);
        CHECK(std::memcmp(param, &before, sizeof(before)) == 0);
        CHECK(std::memcmp(param->stack, values, sizeof(values)) == 0);
        expect_failure([&] { eshkol_parameter_set_converter(param, incoming); }, site);
        CHECK(std::memcmp(param, &before, sizeof(before)) == 0);
        const auto live_after = eshkol_promotion_test_snapshot();
        for (int i = 0; i < 4; ++i) CHECK(live_after.live_bytes[i] == live_before.live_bytes[i]);
        CHECK(region_current() == region);
        eshkol_parameter_push(param, incoming);
        CHECK(param->top == 8 && param->capacity == 16);
        eshkol_parameter_set_converter(param, incoming);
        CHECK(param->converter.data.ptr_val == param->stack[8].data.ptr_val);
        region_pop();
        check_graph(param->stack[8], incoming.data.ptr_val);
        std::free(param->stack);
    }
}

void region_owned_parameter_converter() {
    auto* region = region_create("checked-parameter-converter", 4096);
    CHECK(region);
    region_push(region);
    auto* param = static_cast<Parameter*>(eshkol_make_parameter(region->arena, integer(0)));
    CHECK(param);
    const auto incoming = graph(region->arena);
    const auto before = param->converter;
    expect_failure([&] { eshkol_parameter_set_converter(param, incoming); }, 4);
    CHECK(std::memcmp(&param->converter, &before, sizeof(before)) == 0);
    const size_t root_used_before = arena_get_used_memory(region->escape_base);
    eshkol_parameter_set_converter(param, incoming);
    const size_t root_used_after = arena_get_used_memory(region->escape_base);
    const auto promoted = eshkol_promotion_test_snapshot();
    check_graph(param->converter, incoming.data.ptr_val);
    // A second store of this already-forwarded graph retains no additional copy.
    eshkol_parameter_set_converter(param, incoming);
    CHECK(arena_get_used_memory(region->escape_base) == root_used_after);
    CHECK(eshkol_promotion_test_snapshot().target_bytes == promoted.target_bytes);
    std::printf("parameter converter retention: copied bytes=%llu, root arena delta=%zu, reuse delta=0\n",
                static_cast<unsigned long long>(promoted.target_bytes),
                root_used_after - root_used_before);
    eshkol_tagged_value_t escaped{};
    const auto source = pointer(param);
    CHECK(eshkol_region_write_barrier_checked_v1(&escaped, nullptr, &source) == 0);
    auto* copy = reinterpret_cast<Parameter*>(escaped.data.ptr_val);
    CHECK(copy != param && copy->stack == param->stack);
    CHECK(copy->converter.data.ptr_val == param->converter.data.ptr_val);
    region_pop();
    check_graph(copy->converter, incoming.data.ptr_val);
    std::free(copy->stack);
}

void sequence_failures() {
    for (int dst_dual = 0; dst_dual < 2; ++dst_dual) {
        for (int src_dual = 0; src_dual < 2; ++src_dual) {
            for (int stage = 0; stage < 6; ++stage) {
                void* dst = make_sequence(get_global_arena(), dst_dual, 4);
                auto* destination = sequence_slots(dst, dst_dual);
                eshkol_tagged_value_t before[4];
                std::memcpy(before, destination, sizeof(before));
                auto* region = region_create("checked-copy", 4096);
                CHECK(region);
                region_push(region);
                void* src = make_sequence(region->arena, src_dual, 3);
                auto* source = sequence_slots(src, src_dual);
                const auto incoming = graph(region->arena);
                source[0] = integer(42);
                source[1] = source[2] = incoming;
                eshkol_tagged_value_t source_before[3];
                std::memcpy(source_before, source, sizeof(source_before));
                const auto live_before = eshkol_promotion_test_snapshot();
                expect_failure([&] { (void)eshkol_vector_copy_mutating(dst, 1, src, 0, 3); },
                               stage == 5 ? 4 : stage, stage == 5 ? 1 : 0);
                CHECK(std::memcmp(destination, before, sizeof(before)) == 0);
                CHECK(std::memcmp(source, source_before, sizeof(source_before)) == 0);
                const auto live_after = eshkol_promotion_test_snapshot();
                for (int i = 0; i < 4; ++i) CHECK(live_after.live_bytes[i] == live_before.live_bytes[i]);
                CHECK(region_current() == region);
                CHECK(eshkol_vector_copy_mutating(dst, 1, src, 0, 3) == ESHKOL_VECTOR_COPY_OK);
                CHECK(destination[0].data.int_val == -1 && destination[1].data.int_val == 42);
                CHECK(destination[2].data.ptr_val == destination[3].data.ptr_val);
                region_pop();
                check_graph(destination[2], incoming.data.ptr_val);
            }
        }
    }
}

void overlap_and_validation() {
    for (int dual = 0; dual < 2; ++dual) {
        void* value = make_sequence(get_global_arena(), dual, 5);
        auto* elements = sequence_slots(value, dual);
        for (int direction = 0; direction < 2; ++direction) {
            for (int i = 0; i < 5; ++i) elements[i] = integer(i);
            eshkol_promotion_test_arm(-1, 0);
            CHECK(eshkol_vector_copy_mutating(value, direction ? 0 : 1,
                  value, direction ? 1 : 0, direction ? 5 : 4) == ESHKOL_VECTOR_COPY_OK);
            const auto stats = eshkol_promotion_test_snapshot();
            for (auto attempts : stats.attempts) CHECK(attempts == 0);
            eshkol_promotion_test_reset();
            for (int i = 0; i < 5; ++i) {
                CHECK(elements[i].data.int_val == (direction ? (i < 4 ? i + 1 : 4) : (i > 0 ? i - 1 : 0)));
            }
        }
        eshkol_tagged_value_t before[5];
        std::memcpy(before, elements, sizeof(before));
        CHECK(eshkol_vector_copy_mutating(value, -1, value, 0, 3) == ESHKOL_VECTOR_COPY_BOUNDS);
        CHECK(eshkol_vector_copy_mutating(value, 0, value, 3, 2) == ESHKOL_VECTOR_COPY_BOUNDS);
        CHECK(eshkol_vector_copy_mutating(value, 4, value, 0, 2) == ESHKOL_VECTOR_COPY_BOUNDS);
        CHECK(eshkol_vector_copy_mutating(value, 5, value, 5, 5) == ESHKOL_VECTOR_COPY_OK);
        CHECK(std::memcmp(before, elements, sizeof(before)) == 0);
    }
    auto* numeric = arena_allocate_tensor_full(get_global_arena(), 1, 2);
    CHECK(numeric);
    numeric->dimensions[0] = 2;
    numeric->elements[0] = 123;
    numeric->elements[1] = 456;
    void* vector = make_vector(get_global_arena(), 2);
    slots(vector)[0] = integer(9);
    slots(vector)[1] = pointer(vector);
    CHECK(eshkol_vector_copy_mutating(numeric, 0, vector, 0, 2) == ESHKOL_VECTOR_COPY_TYPE);
    CHECK(numeric->elements[0] == 123 && numeric->elements[1] == 456);
    void* dual = make_sequence(get_global_arena(), true, 2);
    CHECK(eshkol_vector_copy_mutating(numeric, 0, dual, 0, 2) == ESHKOL_VECTOR_COPY_TYPE);
    CHECK(numeric->elements[0] == 123 && numeric->elements[1] == 456);
}
} // namespace

extern "C" void* __wrap_malloc(size_t bytes) {
    void* value = __real_malloc(bytes);
    if (tracking && bytes == 8 * sizeof(eshkol_tagged_value_t)) {
        CHECK(initial_stack == nullptr);
        initial_stack = value;
        ++stack_allocations;
    }
    return value;
}
extern "C" void* __wrap_realloc(void* pointer, size_t bytes) {
    if (tracking) ++reallocations;
    return __real_realloc(pointer, bytes);
}
extern "C" void __wrap_free(void* pointer) {
    if (tracking && pointer == initial_stack && pointer) ++stack_frees;
    __real_free(pointer);
}

int main() {
    parameter_failures();
    region_owned_parameter_converter();
    sequence_failures();
    overlap_and_validation();
    std::puts("PASS checked external callers: 21 parameter and 24 batch failures, retry, overlap, validation");
}
