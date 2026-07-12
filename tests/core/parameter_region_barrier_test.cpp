// Direct C-ABI reproduction/regression test for the make-parameter dynamic
// value-stack region dangling-pointer bug.
//
// `eshkol_param_t` (lib/core/runtime_parameters_hosted.cpp) is the runtime
// object behind R7RS `make-parameter`/`parameterize`. Its control block is
// arena-allocated, but its dynamic binding stack (`eshkol_param_t::stack`) is
// a plain malloc'd buffer that `eshkol_parameter_push`/`eshkol_make_parameter`
// grow/seed directly. Before the fix, a tagged value written into that stack
// was never routed through the region write barrier (ESH-0214c,
// lib/core/runtime_regions.cpp): if the value pointed into a region arena
// that was still active at push time, and that region later popped, the
// malloc'd stack kept a dangling pointer into freed (and, under
// ESHKOL_ARENA_POISON, 0xCB-poisoned) memory -- a use-after-free the next
// time the parameter was read.
//
// Source-level `(make-parameter x)` now constructs this object and
// `parameterize` pushes/pops its dynamic stack through this same C ABI. This
// direct test remains intentional: the regression is in the C storage layer,
// where a value copied out of an active region must remain valid after that
// region closes, independent of parser and closure-call lowering.

#include "../../lib/core/arena_memory.h"

#include <cstdint>
#include <cstring>
#include <iostream>

extern "C" {
void* eshkol_make_parameter(void* arena, eshkol_tagged_value_t default_val);
void eshkol_parameter_push(void* param_ptr, eshkol_tagged_value_t val);
void eshkol_parameter_pop(void* param_ptr);
eshkol_tagged_value_t eshkol_parameter_ref(void* param_ptr);
}

namespace {

int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// Allocate a header-backed, NUL-terminated string in `arena` and return a
// HEAP_PTR tagged value pointing at it.
eshkol_tagged_value_t make_region_string(arena_t* arena, const char* text) {
    const size_t len = std::strlen(text);
    char* buf = arena_allocate_string_with_header(arena, len);
    std::memcpy(buf, text, len + 1);

    eshkol_tagged_value_t v{};
    v.type = ESHKOL_VALUE_HEAP_PTR;
    v.flags = 0;
    v.reserved = 0;
    v.data.ptr_val = reinterpret_cast<uint64_t>(buf);
    return v;
}

const char* tagged_cstr(const eshkol_tagged_value_t& v) {
    return reinterpret_cast<const char*>(static_cast<uintptr_t>(v.data.ptr_val));
}

}  // namespace

int main() {
    // ─── Case 1: eshkol_make_parameter's default value ─────────────────────
    // Construct the parameter's control block from the global arena (as
    // codegen would), but seed its default value with a string allocated
    // inside an ACTIVE region. Popping that region must not leave the
    // parameter's bottom-of-stack binding dangling.
    {
        eshkol_region_t* r = region_create("param_default_region", 4096);
        if (!r) return fail("case1: region_create failed");
        region_push(r);

        eshkol_tagged_value_t default_val = make_region_string(r->arena, "default-in-region");

        void* param = eshkol_make_parameter(get_global_arena(), default_val);
        if (!param) return fail("case1: eshkol_make_parameter failed");

        region_pop();  // destroys the region's arena (poisoned under ESHKOL_ARENA_POISON=1)

        eshkol_tagged_value_t read_back = eshkol_parameter_ref(param);
        if (read_back.type != ESHKOL_VALUE_HEAP_PTR) {
            return fail("case1: read-back value lost its HEAP_PTR type");
        }
        if (std::strcmp(tagged_cstr(read_back), "default-in-region") != 0) {
            return fail("case1: default value did not survive region pop (dangling read)");
        }
    }

    // ─── Case 2: eshkol_parameter_push's dynamic binding ────────────────────
    // A parameter created OUTSIDE any region, then pushed with a value that
    // lives inside a region entered afterward (mirroring `parameterize`
    // binding a parameter to a freshly-consed region-local value). Popping
    // that region while the binding is still on the parameter's stack must
    // not leave the top-of-stack binding dangling.
    {
        eshkol_tagged_value_t outer_default{};
        outer_default.type = ESHKOL_VALUE_INT64;
        outer_default.data.int_val = 0;
        void* param = eshkol_make_parameter(get_global_arena(), outer_default);
        if (!param) return fail("case2: eshkol_make_parameter failed");

        eshkol_region_t* r = region_create("param_push_region", 4096);
        if (!r) return fail("case2: region_create failed");
        region_push(r);

        eshkol_tagged_value_t pushed_val = make_region_string(r->arena, "pushed-in-region");
        eshkol_parameter_push(param, pushed_val);

        region_pop();  // frees/poisons the region the pushed value was allocated in

        eshkol_tagged_value_t read_back = eshkol_parameter_ref(param);
        if (read_back.type != ESHKOL_VALUE_HEAP_PTR) {
            return fail("case2: read-back value lost its HEAP_PTR type");
        }
        if (std::strcmp(tagged_cstr(read_back), "pushed-in-region") != 0) {
            return fail("case2: pushed value did not survive region pop (dangling read/UAF)");
        }

        eshkol_parameter_pop(param);
        eshkol_tagged_value_t restored = eshkol_parameter_ref(param);
        if (restored.type != ESHKOL_VALUE_INT64 || restored.data.int_val != 0) {
            return fail("case2: pop did not restore the outer default binding");
        }
    }

    // ─── Case 3: nested regions, several pushes, pop innermost only ────────
    // Two bindings pushed from two different (nested) regions; only the
    // innermost pops. The still-live (outer) binding must be unaffected, and
    // the popped (inner) binding -- now the current top-of-stack value --
    // must have been evacuated when it was pushed, not when its region popped.
    {
        eshkol_tagged_value_t outer_default{};
        outer_default.type = ESHKOL_VALUE_INT64;
        outer_default.data.int_val = 0;
        void* param = eshkol_make_parameter(get_global_arena(), outer_default);
        if (!param) return fail("case3: eshkol_make_parameter failed");

        eshkol_region_t* outer_r = region_create("param_outer_region", 4096);
        region_push(outer_r);
        eshkol_tagged_value_t outer_val = make_region_string(outer_r->arena, "outer-binding");
        eshkol_parameter_push(param, outer_val);

        eshkol_region_t* inner_r = region_create("param_inner_region", 4096);
        region_push(inner_r);
        eshkol_tagged_value_t inner_val = make_region_string(inner_r->arena, "inner-binding");
        eshkol_parameter_push(param, inner_val);

        region_pop();  // destroys inner_r only; outer_r is still active

        eshkol_tagged_value_t top = eshkol_parameter_ref(param);
        if (std::strcmp(tagged_cstr(top), "inner-binding") != 0) {
            return fail("case3: inner binding did not survive inner region pop");
        }

        eshkol_parameter_pop(param);
        eshkol_tagged_value_t restored_outer = eshkol_parameter_ref(param);
        if (std::strcmp(tagged_cstr(restored_outer), "outer-binding") != 0) {
            return fail("case3: outer binding corrupted after inner pop/restore");
        }

        region_pop();  // destroys outer_r
        eshkol_parameter_pop(param);
        eshkol_tagged_value_t final_val = eshkol_parameter_ref(param);
        if (final_val.type != ESHKOL_VALUE_INT64 || final_val.data.int_val != 0) {
            return fail("case3: final restore did not reach the original default");
        }
    }

    std::cout << "PASS\n";
    return 0;
}
