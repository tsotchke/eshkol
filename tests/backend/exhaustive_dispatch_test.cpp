/**
 * @file exhaustive_dispatch_test.cpp
 * @brief SW-70: the AD node registry is total, and a node type with no
 *        registered backward REFUSES rather than returning a plausible zero.
 *
 * WHAT THIS GUARDS
 *   PR #498 found four tensor-valued AD node types falling into an early
 *   `default:` in eshkol_tensor_backward_dispatch that did nothing at all —
 *   gradients exactly 0.0, no diagnostic, exit 0 — behind a comment asserting
 *   the case was impossible, reasoned from a numeric band of enum values that
 *   those four node types had already stepped outside of. Auditing the class
 *   found four MORE still live on master: AD_NODE_HYPERBOLIC_DISTANCE,
 *   AD_NODE_POINCARE_EXP_MAP, AD_NODE_POINCARE_LOG_MAP and
 *   AD_NODE_GEODESIC_ATTENTION are recorded as TENSOR nodes by
 *   lib/bridge/qllm_bridge.cpp and land in the same default.
 *
 *   The primary fix is a COMPILE-TIME one: no `default:`, plus
 *   -Werror=switch -Werror=switch-enum, so an unhandled node type is a build
 *   failure. This file guards the parts a compiler cannot see:
 *
 *     1. the registry is DENSE — one row per value, 0..COUNT-1, no gaps and
 *        no duplicates. The backward table is a flat array indexed by node
 *        type; a gap would make it index a hole;
 *     2. every node type has a NAME, because the abort messages are the whole
 *        user-visible surface of a missing backward;
 *     3. the payload declarations are consistent — every disposition that only
 *        makes sense for a tensor node is on a TENSOR row;
 *     4. every row declaring BRIDGE resolves to a real registered function,
 *        and every row that does NOT declare BRIDGE resolves to nothing, so
 *        the table and the dispositions cannot drift apart;
 *     5. an UNREGISTERED node type with a live upstream gradient REFUSES.
 *        This is the assertion that would have failed before the fix: the old
 *        dispatcher returned normally and left every gradient at 0.0.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "eshkol/eshkol.h"
#include "eshkol/backend/tensor_backward.h"

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#define ESHKOL_HAVE_FORK_DEATH_TESTS 1
#endif

extern "C" {
    typedef struct arena arena_t;
    arena_t* get_global_arena(void);
    void* arena_allocate_zeroed(arena_t* arena, size_t size);
    ad_node_t* arena_allocate_ad_node(arena_t* arena);
    typedef void (*backward_fn_t)(ad_node_t*);
    backward_fn_t get_tensor_backward_fn(int node_type);
}

namespace {

int g_passed = 0;
int g_failed = 0;

void report(const char* name, bool ok, const std::string& detail = std::string()) {
    std::printf("  %-58s %s", name, ok ? "PASS" : "FAIL");
    if (!detail.empty()) std::printf("   [%s]", detail.c_str());
    std::printf("\n");
    if (ok) ++g_passed; else ++g_failed;
}

/* ── The registry, read back in C++ from the same rows the compiler used ──
 * Not a second copy of the table: the SAME .def, expanded again. A drift
 * between this and the dispatch is impossible by construction, which is the
 * point — the test can then check properties OF the rows rather than trying
 * to re-state them. */
enum Disposition { LEAF, SCALAR_ADJOINT, INLINE_ARM, BRIDGE, CUSTOM_VJP, UNREGISTERED };

struct Row {
    const char* name;
    int value;
    bool tensor_payload;
    Disposition disposition;
};

#define ESHKOL_AD_NO_BRIDGE 0
#define ESHKOL_AD_TEST_PAYLOAD_SCALAR false
#define ESHKOL_AD_TEST_PAYLOAD_TENSOR true
#define ESHKOL_AD_TEST_DISP_LEAF           LEAF
#define ESHKOL_AD_TEST_DISP_SCALAR_ADJOINT SCALAR_ADJOINT
#define ESHKOL_AD_TEST_DISP_INLINE         INLINE_ARM
#define ESHKOL_AD_TEST_DISP_BRIDGE         BRIDGE
#define ESHKOL_AD_TEST_DISP_CUSTOM_VJP     CUSTOM_VJP
#define ESHKOL_AD_TEST_DISP_UNREGISTERED   UNREGISTERED

const Row kRegistry[] = {
#define ESHKOL_AD_NODE(NAME, VALUE, PAYLOAD, TENSOR_BACKWARD, BRIDGE_FN) \
    { "AD_NODE_" #NAME, (VALUE), ESHKOL_AD_TEST_PAYLOAD_##PAYLOAD,       \
      ESHKOL_AD_TEST_DISP_##TENSOR_BACKWARD },
#include "eshkol/ad_node_registry.def"
#undef ESHKOL_AD_NODE
};

const size_t kRegistryRows = sizeof(kRegistry) / sizeof(kRegistry[0]);

/* ───────────────────────────── 1. density ───────────────────────────── */

void test_registry_is_dense() {
    bool ok = (kRegistryRows == (size_t)AD_NODE_TYPE_COUNT);
    std::string detail = std::to_string(kRegistryRows) + " rows, COUNT=" +
                         std::to_string((int)AD_NODE_TYPE_COUNT);
    for (size_t i = 0; ok && i < kRegistryRows; ++i) {
        if (kRegistry[i].value != (int)i) {
            ok = false;
            detail = std::string(kRegistry[i].name) + " declares value " +
                     std::to_string(kRegistry[i].value) + " at ordinal " +
                     std::to_string(i);
        }
    }
    report("ad_node_registry_is_dense", ok, detail);
}

/* ───────────────────────────── 2. names ───────────────────────────── */

void test_every_node_type_has_a_name() {
    bool ok = true;
    std::string detail;
    std::set<std::string> seen;
    for (size_t i = 0; i < kRegistryRows; ++i) {
        const char* got = eshkol_ad_node_type_name(kRegistry[i].value);
        if (!got || std::strcmp(got, kRegistry[i].name) != 0) {
            ok = false;
            detail = std::string("value ") + std::to_string(kRegistry[i].value) +
                     " named " + (got ? got : "(null)") + ", expected " +
                     kRegistry[i].name;
            break;
        }
        if (!seen.insert(got).second) {
            ok = false;
            detail = std::string("duplicate name ") + got;
            break;
        }
    }
    report("every_node_type_has_a_unique_name", ok, detail);

    // Out of range must not read past the table.
    const char* oob = eshkol_ad_node_type_name((int)AD_NODE_TYPE_COUNT + 4096);
    report("out_of_range_node_type_is_named_not_indexed",
           oob != nullptr && std::strstr(oob, "out-of-range") != nullptr,
           oob ? oob : "(null)");
}

/* ─────────────────────── 3. payload consistency ─────────────────────── */

void test_payload_declarations_are_consistent() {
    bool ok = true;
    std::string detail;
    for (size_t i = 0; i < kRegistryRows; ++i) {
        const Row& r = kRegistry[i];
        const bool needs_tensor = (r.disposition == INLINE_ARM ||
                                   r.disposition == BRIDGE ||
                                   r.disposition == UNREGISTERED);
        if (needs_tensor && !r.tensor_payload) {
            ok = false;
            detail = std::string(r.name) + " has a tensor-only disposition on a SCALAR payload";
            break;
        }
        if (eshkol_ad_node_type_is_tensor(r.value) != r.tensor_payload) {
            ok = false;
            detail = std::string(r.name) + ": generated payload predicate disagrees with its row";
            break;
        }
    }
    report("payload_declarations_are_consistent", ok, detail);
}

/* ──────────────────── 4. the bridge table is total ──────────────────── */

void test_every_bridge_row_resolves_to_a_registered_backward() {
    bool ok = true;
    std::string detail;
    int bridged = 0;
    for (size_t i = 0; i < kRegistryRows; ++i) {
        const Row& r = kRegistry[i];
        backward_fn_t fn = get_tensor_backward_fn(r.value);
        if (r.disposition == BRIDGE) {
            ++bridged;
            if (!fn) {
                ok = false;
                detail = std::string(r.name) + " declares BRIDGE but the table has no entry";
                break;
            }
        } else if (fn) {
            ok = false;
            detail = std::string(r.name) + " is not BRIDGE but the table has an entry";
            break;
        }
    }
    if (ok && bridged == 0) {
        ok = false;
        detail = "no BRIDGE rows at all — the table would be vacuously total";
    }
    report("every_bridge_row_resolves_to_a_registered_backward", ok,
           ok ? (std::to_string(bridged) + " bridged") : detail);

    // A node type outside the enum must not index the flat table.
    report("out_of_range_node_type_does_not_index_the_table",
           get_tensor_backward_fn(-1) == nullptr &&
           get_tensor_backward_fn((int)AD_NODE_TYPE_COUNT) == nullptr);
}

/* ─────────── 5. an UNREGISTERED node refuses, never returns 0.0 ─────────── */

#if defined(ESHKOL_HAVE_FORK_DEATH_TESTS)

/** @brief True iff running @p body in a forked child terminates abnormally or
 *  exits nonzero — i.e. it REFUSED rather than returning. */
bool refuses(void (*body)()) {
    std::fflush(stdout);
    std::fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        FILE* devnull = std::freopen("/dev/null", "w", stderr);
        (void)devnull;
        body();
        std::_Exit(0);   /* reached only if the call did NOT refuse */
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return false;
    if (WIFSIGNALED(status)) return true;
    return WIFEXITED(status) && WEXITSTATUS(status) != 0;
}

/** @brief Build a tensor node of @p type with a live upstream gradient and a
 *  variable input, then run the reverse dispatch on it. Before this change
 *  every UNREGISTERED type returned from here having done nothing, leaving
 *  the input gradient at exactly 0.0. */
void drive_unregistered(ad_node_type_t type) {
    arena_t* a = get_global_arena();
    const size_t n = 4;
    static int64_t shape[1];
    shape[0] = (int64_t)n;

    ad_node_t* in = arena_allocate_ad_node(a);
    in->type = AD_NODE_VARIABLE;
    in->tensor_value = arena_allocate_zeroed(a, n * sizeof(double));
    in->shape = shape;
    in->ndim = 1;

    ad_node_t* out = arena_allocate_ad_node(a);
    out->type = type;
    out->tensor_value = arena_allocate_zeroed(a, n * sizeof(double));
    out->tensor_gradient = arena_allocate_zeroed(a, n * sizeof(double));
    for (size_t i = 0; i < n; ++i) ((double*)out->tensor_gradient)[i] = 1.0;
    out->shape = shape;
    out->ndim = 1;
    out->input1 = in;

    eshkol_tensor_backward_dispatch(out);
}

void body_tangent_project() { drive_unregistered(AD_NODE_TANGENT_PROJECT); }
void body_mobius_add()      { drive_unregistered(AD_NODE_MOBIUS_ADD); }
void body_mobius_matmul()   { drive_unregistered(AD_NODE_MOBIUS_MATMUL); }
void body_gyrovector()      { drive_unregistered(AD_NODE_GYROVECTOR_SPACE); }

void test_unregistered_tensor_backward_is_declared_not_silent() {
    // The four UNREGISTERED rows that remain after SW-65 registered exact
    // BRIDGE backwards for the ops the qLLM bridge actually produces
    // (HYPERBOLIC_DISTANCE / POINCARE_EXP_MAP / POINCARE_LOG_MAP /
    // GEODESIC_ATTENTION — those are now exercised end-to-end by
    // qllm_bridge_geometric_gradcheck instead).  Nothing writes these four
    // today; a future producer must inherit the abort, not the silence.
    // The same drive on the pre-registry dispatcher returned normally and the
    // caller read a gradient of exactly 0.0 as if it were the answer.
    report("tangent_project_backward_refuses",  refuses(body_tangent_project));
    report("mobius_add_backward_refuses",       refuses(body_mobius_add));
    report("mobius_matmul_backward_refuses",    refuses(body_mobius_matmul));
    report("gyrovector_space_backward_refuses", refuses(body_gyrovector));
}

#else

void test_unregistered_tensor_backward_is_declared_not_silent() {
    std::printf("  %-58s SKIP  [no fork()]\n",
                "unregistered_tensor_backward_refuses");
}

#endif

/** @brief Every UNREGISTERED row must be reachable as a refusal, i.e. it must
 *  not silently also have a bridge entry. Cheap, but it is the property that
 *  keeps the death tests above and the registry rows in agreement. */
void test_all_unregistered_rows_have_no_backward() {
    bool ok = true;
    std::string detail;
    int count = 0;
    for (size_t i = 0; i < kRegistryRows; ++i) {
        if (kRegistry[i].disposition != UNREGISTERED) continue;
        ++count;
        if (get_tensor_backward_fn(kRegistry[i].value) != nullptr) {
            ok = false;
            detail = std::string(kRegistry[i].name) +
                     " is UNREGISTERED but a backward is registered for it";
            break;
        }
    }
    report("all_unregistered_rows_have_no_backward", ok,
           ok ? (std::to_string(count) + " unregistered") : detail);
}

}  // namespace

int main() {
    std::printf("SW-70 closed-enum dispatch exhaustiveness\n");
    test_registry_is_dense();
    test_every_node_type_has_a_name();
    test_payload_declarations_are_consistent();
    test_every_bridge_row_resolves_to_a_registered_backward();
    test_all_unregistered_rows_have_no_backward();
    test_unregistered_tensor_backward_is_declared_not_silent();
    std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
