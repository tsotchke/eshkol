/*
 * CE-8 Native Verification Suite:
 *   1. Staged Zero-Allocation Compiled Backward Graphs (Guards 1, 2, 3)
 *   2. Incremental Sharded DBSP Z-Set Knowledge Graph (Guards 4, 5, 6)
 *   3. Graded Order-2 Interval-Taylor Certified Neural World Model Enclosures (Guards 7, 8, 9)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/staged_value_grad.h"
#include "eshkol/core/dbsp_zset_kg.h"
#include "eshkol/core/interval_taylor_enclosure.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

/* ============================================================================
 * Global Heap Allocation Instrumentation (Guards 1 & 2)
 * ============================================================================ */

namespace {

std::atomic<bool> g_alloc_guard_armed{false};
std::atomic<uint64_t> g_heap_alloc_count{0};
volatile void* g_escape_sink = nullptr;

inline void* instrumented_alloc(std::size_t size) {
    if (g_alloc_guard_armed.load(std::memory_order_relaxed)) {
        g_heap_alloc_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (size == 0) size = 1;
    return std::malloc(size);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
int* force_heap_probe_alloc(std::size_t count) {
    int* p = new int[count];
    g_escape_sink = p;
    p[0] = 42;
    return p;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void force_heap_probe_free(int* p) {
    g_escape_sink = p;
    delete[] p;
}

} /* namespace */

void* operator new(std::size_t size) {
    void* p = instrumented_alloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void* operator new[](std::size_t size) {
    void* p = instrumented_alloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    return instrumented_alloc(size);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return instrumented_alloc(size);
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr, const std::nothrow_t&) noexcept {
    std::free(ptr);
}

/* ============================================================================
 * Minimal Deterministic PRNG & Test Harness
 * ============================================================================ */

namespace {

int g_pass = 0;
int g_fail = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (cond) {                                                           \
            ++g_pass;                                                         \
        } else {                                                              \
            ++g_fail;                                                         \
            std::printf("  [FAIL] %s (line %d)\n", (msg), __LINE__);          \
        }                                                                     \
    } while (0)

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double next_unit() {
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }
    double next_range(double lo, double hi) {
        return lo + (hi - lo) * next_unit();
    }
};

/* ============================================================================
 * Independent Allocating Reference Autograd (for Guard 3 Drift Verification)
 * ============================================================================ */

void reference_allocating_value_grad(const std::vector<double>& x,
                                     const std::vector<double>& target,
                                     const std::vector<double>& params,
                                     double& out_val,
                                     std::vector<double>& out_grads) {
    /* Parameter layout: W1[8x6=48], b1[8], W2[4x8=32], b2[4], u[4] -> total 96 */
    const uint32_t IN = 6;
    const uint32_t H = 8;
    const uint32_t OUT = 4;
    const size_t off_W1 = 0;
    const size_t off_b1 = 48;
    const size_t off_W2 = 56;
    const size_t off_b2 = 88;
    const size_t off_u = 92;

    std::vector<double> h1_mv(H, 0.0);
    for (uint32_t r = 0; r < H; ++r) {
        double s = 0.0;
        for (uint32_t c = 0; c < IN; ++c) {
            s += params[off_W1 + r * IN + c] * x[c];
        }
        h1_mv[r] = s;
    }

    std::vector<double> h1_lin(H, 0.0);
    std::vector<double> h1_a(H, 0.0);
    std::vector<double> h1_b(H, 0.0);
    std::vector<double> h1_prod(H, 0.0);
    for (uint32_t i = 0; i < H; ++i) {
        h1_lin[i] = h1_mv[i] + params[off_b1 + i];
        h1_a[i] = std::tanh(h1_lin[i]);
        double z = h1_lin[i];
        h1_b[i] = (z > 0.0) ? (z + std::log1p(std::exp(-z)))
                            : std::log1p(std::exp(z));
        h1_prod[i] = h1_a[i] * h1_b[i];
    }

    std::vector<double> y_mv(OUT, 0.0);
    std::vector<double> y_lin(OUT, 0.0);
    for (uint32_t r = 0; r < OUT; ++r) {
        double s = 0.0;
        for (uint32_t c = 0; c < H; ++c) {
            s += params[off_W2 + r * H + c] * h1_prod[c];
        }
        y_mv[r] = s;
        y_lin[r] = s + params[off_b2 + r];
    }

    /* Scalar energy terms */
    double m = y_lin[0];
    for (uint32_t i = 1; i < OUT; ++i) {
        if (y_lin[i] > m) m = y_lin[i];
    }
    double sum_exp = 0.0;
    for (uint32_t i = 0; i < OUT; ++i) {
        sum_exp += std::exp(y_lin[i] - m);
    }
    double lse = m + std::log(sum_exp);

    double mse = 0.0;
    for (uint32_t i = 0; i < OUT; ++i) {
        double d = y_lin[i] - target[i];
        mse += 0.5 * d * d;
    }

    double dot_term = 0.0;
    for (uint32_t i = 0; i < OUT; ++i) {
        dot_term += y_lin[i] * params[off_u + i];
    }

    double sum_term = 0.0;
    for (uint32_t i = 0; i < H; ++i) {
        sum_term += h1_prod[i];
    }

    const double c_lse = 0.35;
    const double c_dot = -0.20;
    const double c_sum = 0.15;

    out_val = mse + c_lse * lse + c_dot * dot_term + c_sum * sum_term;

    /* Independent reverse-mode adjoint pass */
    out_grads.assign(params.size(), 0.0);
    std::vector<double> g_y_lin(OUT, 0.0);
    std::vector<double> g_h1_prod(H, 0.0);
    std::vector<double> g_h1_lin(H, 0.0);

    for (uint32_t i = 0; i < OUT; ++i) {
        double sm = std::exp(y_lin[i] - m) / sum_exp;
        g_y_lin[i] += (y_lin[i] - target[i]);
        g_y_lin[i] += c_lse * sm;
        g_y_lin[i] += c_dot * params[off_u + i];
        out_grads[off_u + i] += c_dot * y_lin[i];
        out_grads[off_b2 + i] += g_y_lin[i];
    }

    for (uint32_t r = 0; r < OUT; ++r) {
        for (uint32_t c = 0; c < H; ++c) {
            out_grads[off_W2 + r * H + c] += g_y_lin[r] * h1_prod[c];
            g_h1_prod[c] += g_y_lin[r] * params[off_W2 + r * H + c];
        }
    }

    for (uint32_t i = 0; i < H; ++i) {
        g_h1_prod[i] += c_sum;
        double g_a = g_h1_prod[i] * h1_b[i];
        double g_b = g_h1_prod[i] * h1_a[i];
        double dtanh = 1.0 - h1_a[i] * h1_a[i];
        double z = h1_lin[i];
        double dsoftplus = (z >= 0.0) ? (1.0 / (1.0 + std::exp(-z)))
                                      : (std::exp(z) / (1.0 + std::exp(z)));
        g_h1_lin[i] = g_a * dtanh + g_b * dsoftplus;
        out_grads[off_b1 + i] += g_h1_lin[i];
    }

    for (uint32_t r = 0; r < H; ++r) {
        for (uint32_t c = 0; c < IN; ++c) {
            out_grads[off_W1 + r * IN + c] += g_h1_lin[r] * x[c];
        }
    }
}

/* ============================================================================
 * Test Group A: Staged Zero-Alloc AD (Guards 1, 2, 3)
 * ============================================================================ */

void test_staged_zero_alloc_ad() {
    std::printf("\n=== CE-8 Part 1: Staged Zero-Allocation Compiled Value+Grad ===\n");

    eshkol_staged_builder_t* b = eshkol_staged_builder_create();
    CHECK(b != nullptr, "builder created");

    const uint32_t IN = 6;
    const uint32_t H = 8;
    const uint32_t OUT = 4;

    uint32_t n_x      = eshkol_staged_add_input(b, IN);
    uint32_t n_target = eshkol_staged_add_input(b, OUT);

    uint32_t n_W1 = eshkol_staged_add_param(b, H * IN);
    uint32_t n_b1 = eshkol_staged_add_param(b, H);
    uint32_t n_W2 = eshkol_staged_add_param(b, OUT * H);
    uint32_t n_b2 = eshkol_staged_add_param(b, OUT);
    uint32_t n_u  = eshkol_staged_add_param(b, OUT);

    uint32_t n_h1_mv   = eshkol_staged_add_matvec(b, n_W1, n_x, H, IN);
    uint32_t n_h1_lin  = eshkol_staged_add_binary(b, ESHKOL_STAGED_OP_ADD, n_h1_mv, n_b1);
    uint32_t n_h1_a    = eshkol_staged_add_unary(b, ESHKOL_STAGED_OP_TANH, n_h1_lin);
    uint32_t n_h1_b    = eshkol_staged_add_unary(b, ESHKOL_STAGED_OP_SOFTPLUS, n_h1_lin);
    uint32_t n_h1_prod = eshkol_staged_add_binary(b, ESHKOL_STAGED_OP_MUL, n_h1_a, n_h1_b);

    uint32_t n_y_mv    = eshkol_staged_add_matvec(b, n_W2, n_h1_prod, OUT, H);
    uint32_t n_y_lin   = eshkol_staged_add_binary(b, ESHKOL_STAGED_OP_ADD, n_y_mv, n_b2);

    uint32_t n_lse     = eshkol_staged_add_unary(b, ESHKOL_STAGED_OP_LOGSUMEXP, n_y_lin);
    uint32_t n_mse     = eshkol_staged_add_binary(b, ESHKOL_STAGED_OP_MSE, n_y_lin, n_target);
    uint32_t n_dot     = eshkol_staged_add_binary(b, ESHKOL_STAGED_OP_DOT, n_y_lin, n_u);
    uint32_t n_sum     = eshkol_staged_add_unary(b, ESHKOL_STAGED_OP_SUM, n_h1_prod);

    uint32_t n_e1      = eshkol_staged_add_scale(b, n_lse, 0.35);
    uint32_t n_e2      = eshkol_staged_add_scale(b, n_dot, -0.20);
    uint32_t n_e3      = eshkol_staged_add_scale(b, n_sum, 0.15);

    uint32_t n_s1      = eshkol_staged_add_binary(b, ESHKOL_STAGED_OP_ADD, n_mse, n_e1);
    uint32_t n_s2      = eshkol_staged_add_binary(b, ESHKOL_STAGED_OP_ADD, n_s1, n_e2);
    uint32_t n_loss    = eshkol_staged_add_binary(b, ESHKOL_STAGED_OP_ADD, n_s2, n_e3);

    eshkol_staged_plan_t* plan = eshkol_compile_staged_value_grad(b, n_loss);
    eshkol_staged_builder_destroy(b);

    CHECK(plan != nullptr, "staged plan compiled");
    CHECK(eshkol_staged_plan_total_inputs(plan) == IN + OUT, "total input count == 10");
    CHECK(eshkol_staged_plan_total_params(plan) == 96, "total param count == 96");

    const size_t req_scratch = eshkol_staged_plan_required_scratch(plan);
    CHECK(req_scratch > 0, "required scratch bytes > 0");

    /* Dedicated caller-owned pre-sized scratch arena (allocated BEFORE arming guard). */
    std::vector<uint8_t> scratch_storage(req_scratch + 256, 0);
    eshkol_staged_scratch_arena_t scratch;
    CHECK(eshkol_staged_scratch_init(&scratch, scratch_storage.data(), scratch_storage.size()),
          "scratch arena initialized");

    /* Pre-generate deterministic parameters and 10,000 input vectors BEFORE arming guard. */
    SplitMix64 rng(0xCE800001ULL);
    std::vector<double> params(96, 0.0);
    for (double& v : params) v = rng.next_range(-0.6, 0.6);

    const int N_PASSES = 10000;
    std::vector<double> all_inputs(static_cast<size_t>(N_PASSES) * (IN + OUT), 0.0);
    for (double& v : all_inputs) v = rng.next_range(-1.2, 1.2);

    std::vector<double> staged_values(N_PASSES, 0.0);
    std::vector<double> staged_grads_all(static_cast<size_t>(N_PASSES) * 96, 0.0);

    /* Warmup call (1 pass). */
    double warm_val = 0.0;
    double warm_grad[96] = {0.0};
    uint64_t warm_fd = 999;
    bool warm_ok = eshkol_staged_eval_value_grad(
        plan, &scratch,
        all_inputs.data(), IN + OUT,
        params.data(), 96,
        &warm_val, warm_grad, 96, &warm_fd);
    CHECK(warm_ok, "warmup staged eval succeeded");
    CHECK(warm_fd == 0, "warmup fd_evals == 0");

    /* --- GUARD 1: Staged Zero-Alloc Guard across 10,000 consecutive passes --- */
    uint64_t total_fd_evals = 0;
    bool all_ok = true;

    const uint64_t allocs_before = g_heap_alloc_count.load(std::memory_order_seq_cst);
    g_alloc_guard_armed.store(true, std::memory_order_seq_cst);

    for (int pass = 0; pass < N_PASSES; ++pass) {
        uint64_t fd_evals = 777;
        bool ok = eshkol_staged_eval_value_grad(
            plan, &scratch,
            all_inputs.data() + static_cast<size_t>(pass) * (IN + OUT), IN + OUT,
            params.data(), 96,
            &staged_values[pass],
            staged_grads_all.data() + static_cast<size_t>(pass) * 96, 96,
            &fd_evals);
        if (!ok) all_ok = false;
        total_fd_evals += fd_evals;
    }

    g_alloc_guard_armed.store(false, std::memory_order_seq_cst);
    const uint64_t allocs_after = g_heap_alloc_count.load(std::memory_order_seq_cst);
    const uint64_t guard1_alloc_delta = allocs_after - allocs_before;

    std::printf("  [Guard 1] Staged 10,000-pass heap_alloc_delta=%llu, scratch_heap_allocs=%llu, "
                "fd_evals=%llu, scratch_high_water=%zu/%zu B\n",
                (unsigned long long)guard1_alloc_delta,
                (unsigned long long)scratch.heap_allocs,
                (unsigned long long)total_fd_evals,
                scratch.high_water, req_scratch);

    CHECK(all_ok, "Guard 1: all 10,000 staged passes returned true");
    CHECK(guard1_alloc_delta == 0, "Guard 1: ZERO global heap allocations across 10,000 staged calls");
    CHECK(scratch.heap_allocs == 0, "Guard 1: scratch arena recorded 0 heap allocations");
    CHECK(total_fd_evals == 0, "Guard 1: ZERO finite-difference evaluations");
    CHECK(scratch.resets == static_cast<uint64_t>(N_PASSES + 1), "Guard 1: scratch reset once per call");
    CHECK(scratch.high_water <= req_scratch, "Guard 1: scratch high-water strictly within pre-sized budget");

    /* --- GUARD 2: Staged Allocator Instrumentation Negative Control --- */
    const uint64_t neg_before = g_heap_alloc_count.load(std::memory_order_seq_cst);
    g_alloc_guard_armed.store(true, std::memory_order_seq_cst);

    int* probe = force_heap_probe_alloc(16);
    force_heap_probe_free(probe);

    g_alloc_guard_armed.store(false, std::memory_order_seq_cst);
    const uint64_t neg_after = g_heap_alloc_count.load(std::memory_order_seq_cst);
    const uint64_t guard2_alloc_delta = neg_after - neg_before;

    std::printf("  [Guard 2] Negative control deliberate heap alloc_delta=%llu (must be > 0)\n",
                (unsigned long long)guard2_alloc_delta);
    CHECK(guard2_alloc_delta > 0, "Guard 2: allocator guard detects deliberate heap allocation");

    /* --- GUARD 3: 10,000-Pass Drift Check vs Independent Allocating Reference Autograd --- */
    double max_val_rel_err = 0.0;
    double max_grad_rel_err = 0.0;
    std::vector<double> x_vec(IN);
    std::vector<double> t_vec(OUT);
    std::vector<double> ref_grads;

    for (int pass = 0; pass < N_PASSES; ++pass) {
        const double* inp = all_inputs.data() + static_cast<size_t>(pass) * (IN + OUT);
        for (uint32_t i = 0; i < IN; ++i) x_vec[i] = inp[i];
        for (uint32_t i = 0; i < OUT; ++i) t_vec[i] = inp[IN + i];

        double ref_val = 0.0;
        reference_allocating_value_grad(x_vec, t_vec, params, ref_val, ref_grads);

        double val_denom = std::max(1.0, std::fabs(ref_val));
        double val_err = std::fabs(staged_values[pass] - ref_val) / val_denom;
        if (val_err > max_val_rel_err) max_val_rel_err = val_err;

        const double* sg = staged_grads_all.data() + static_cast<size_t>(pass) * 96;
        for (size_t k = 0; k < 96; ++k) {
            double denom = std::max(1.0, std::fabs(ref_grads[k]));
            double err = std::fabs(sg[k] - ref_grads[k]) / denom;
            if (err > max_grad_rel_err) max_grad_rel_err = err;
        }
    }

    /* Spot-check pass 0 reference autograd against central differences (verification only). */
    double max_ref_fd_err = 0.0;
    {
        for (uint32_t i = 0; i < IN; ++i) x_vec[i] = all_inputs[i];
        for (uint32_t i = 0; i < OUT; ++i) t_vec[i] = all_inputs[IN + i];
        double base_v = 0.0;
        reference_allocating_value_grad(x_vec, t_vec, params, base_v, ref_grads);
        const double eps = 1e-6;
        std::vector<double> p_pert = params;
        std::vector<double> dummy_g;
        for (size_t k = 0; k < 96; ++k) {
            p_pert[k] = params[k] + eps;
            double vp = 0.0;
            reference_allocating_value_grad(x_vec, t_vec, p_pert, vp, dummy_g);
            p_pert[k] = params[k] - eps;
            double vm = 0.0;
            reference_allocating_value_grad(x_vec, t_vec, p_pert, vm, dummy_g);
            p_pert[k] = params[k];
            double fd = (vp - vm) / (2.0 * eps);
            double denom = std::max(1.0, std::fabs(fd));
            double err = std::fabs(ref_grads[k] - fd) / denom;
            if (err > max_ref_fd_err) max_ref_fd_err = err;
        }
    }

    std::printf("  [Guard 3] 10k-pass drift vs allocating reference: max_val_rel_err=%.3e, "
                "max_grad_rel_err=%.3e (< 1e-12), ref_fd_sanity=%.3e\n",
                max_val_rel_err, max_grad_rel_err, max_ref_fd_err);

    CHECK(max_val_rel_err < 1e-12, "Guard 3: 10k-pass value max_rel_err < 1e-12");
    CHECK(max_grad_rel_err < 1e-12, "Guard 3: 10k-pass gradient max_rel_err < 1e-12");
    CHECK(max_ref_fd_err < 1e-7, "Guard 3: independent reference autograd matches central FD");

    eshkol_staged_plan_destroy(plan);
}

/* ============================================================================
 * Test Group B: Incremental Sharded DBSP Z-Set KG (Guards 4, 5, 6)
 * ============================================================================ */

struct BatchDelta {
    std::vector<eshkol_kg_triple_t> a_ops;
    std::vector<eshkol_kg_triple_t> b_ops;
};

int64_t non_commutative_merge_fn(int64_t existing_weight, int64_t delta_weight, void*) {
    return 2 * existing_weight + delta_weight;
}

void test_dbsp_zset_kg() {
    std::printf("\n=== CE-8 Part 2: Incremental Sharded DBSP Z-Set Knowledge Graph ===\n");

    const uint64_t JOIN_PRED = 9001;
    const uint32_t NUM_SHARDS = 8;

    /* Generate a 25-batch stream of mixed inserts and retractions across relations A and B. */
    SplitMix64 rng(0xDB590008ULL);
    const int N_BATCHES = 25;
    std::vector<BatchDelta> stream(N_BATCHES);
    for (int b = 0; b < N_BATCHES; ++b) {
        for (int k = 0; k < 18; ++k) {
            uint64_t s = 1 + (rng.next_u64() % 12);
            uint64_t p = 100 + (rng.next_u64() % 3);
            uint64_t o = 1 + (rng.next_u64() % 10);
            int64_t w = static_cast<int64_t>(rng.next_u64() % 5) - 2; /* in [-2, +2] */
            if (w == 0) w = 1;
            stream[b].a_ops.push_back({s, p, o, w});
        }
        for (int k = 0; k < 18; ++k) {
            uint64_t s = 1 + (rng.next_u64() % 10);
            uint64_t p = 200 + (rng.next_u64() % 3);
            uint64_t o = 1 + (rng.next_u64() % 12);
            int64_t w = static_cast<int64_t>(rng.next_u64() % 5) - 2;
            if (w == 0) w = -1;
            stream[b].b_ops.push_back({s, p, o, w});
        }
    }

    /* --- GUARD 4: DBSP Z-Set Replay Idempotency + Exact Bilinear Incremental Join --- */
    eshkol_dbsp_join_view_t* view1 = eshkol_dbsp_join_create(JOIN_PRED, NUM_SHARDS);
    CHECK(view1 != nullptr, "DBSP join view1 created");
    CHECK(eshkol_zset_num_shards(eshkol_dbsp_join_result(view1)) == NUM_SHARDS,
          "DBSP join view has 8 shards");

    bool all_batches_match_full = true;
    for (int b = 0; b < N_BATCHES; ++b) {
        eshkol_zset_t* da = eshkol_zset_create(NUM_SHARDS);
        eshkol_zset_t* db = eshkol_zset_create(NUM_SHARDS);
        for (const auto& t : stream[b].a_ops) {
            eshkol_zset_add(da, t.subject, t.predicate, t.object, t.weight);
        }
        for (const auto& t : stream[b].b_ops) {
            eshkol_zset_add(db, t.subject, t.predicate, t.object, t.weight);
        }

        eshkol_zset_t* dj = nullptr;
        bool ok = eshkol_dbsp_join_step(view1, da, db, &dj);
        CHECK(ok && dj != nullptr, "DBSP bilinear join step succeeded");
        eshkol_zset_destroy(dj);
        eshkol_zset_destroy(da);
        eshkol_zset_destroy(db);

        /* Compare incremental view against full from-scratch relational join on every batch. */
        eshkol_zset_t* full_j = eshkol_zset_join_full(
            eshkol_dbsp_join_state_a(view1),
            eshkol_dbsp_join_state_b(view1),
            JOIN_PRED, NUM_SHARDS);
        uint64_t h_inc = eshkol_zset_canonical_hash(eshkol_dbsp_join_result(view1));
        uint64_t h_full = eshkol_zset_canonical_hash(full_j);
        if (h_inc != h_full ||
            eshkol_zset_active_count(eshkol_dbsp_join_result(view1)) != eshkol_zset_active_count(full_j)) {
            all_batches_match_full = false;
        }
        eshkol_zset_destroy(full_j);
    }

    /* Replay entire stream from scratch on view2 (8 shards) AND view3 (13 shards). */
    eshkol_dbsp_join_view_t* view2 = eshkol_dbsp_join_create(JOIN_PRED, NUM_SHARDS);
    eshkol_dbsp_join_view_t* view3 = eshkol_dbsp_join_create(JOIN_PRED, 13);
    for (int b = 0; b < N_BATCHES; ++b) {
        eshkol_zset_t* da = eshkol_zset_create(NUM_SHARDS);
        eshkol_zset_t* db = eshkol_zset_create(NUM_SHARDS);
        for (const auto& t : stream[b].a_ops) {
            eshkol_zset_add(da, t.subject, t.predicate, t.object, t.weight);
        }
        for (const auto& t : stream[b].b_ops) {
            eshkol_zset_add(db, t.subject, t.predicate, t.object, t.weight);
        }
        eshkol_dbsp_join_step(view2, da, db, nullptr);
        eshkol_dbsp_join_step(view3, da, db, nullptr);
        eshkol_zset_destroy(da);
        eshkol_zset_destroy(db);
    }

    uint8_t* bytes1 = nullptr; size_t len1 = 0;
    uint8_t* bytes2 = nullptr; size_t len2 = 0;
    uint8_t* bytes3 = nullptr; size_t len3 = 0;
    CHECK(eshkol_zset_canonical_serialize(eshkol_dbsp_join_result(view1), &bytes1, &len1),
          "canonical serialize view1");
    CHECK(eshkol_zset_canonical_serialize(eshkol_dbsp_join_result(view2), &bytes2, &len2),
          "canonical serialize view2");
    CHECK(eshkol_zset_canonical_serialize(eshkol_dbsp_join_result(view3), &bytes3, &len3),
          "canonical serialize view3 (13 shards)");

    uint64_t hash1 = eshkol_zset_canonical_hash(eshkol_dbsp_join_result(view1));
    uint64_t hash2 = eshkol_zset_canonical_hash(eshkol_dbsp_join_result(view2));
    uint64_t hash3 = eshkol_zset_canonical_hash(eshkol_dbsp_join_result(view3));

    bool bit_identical_12 = (len1 == len2) && (std::memcmp(bytes1, bytes2, len1) == 0);
    bool bit_identical_13 = (len1 == len3) && (std::memcmp(bytes1, bytes3, len1) == 0);

    std::printf("  [Guard 4] DBSP 25-batch replay: active=%zu, bytes=%zu, hash1=0x%016llx, "
                "hash2=0x%016llx, hash3(13-shard)=0x%016llx, full_match=%d\n",
                eshkol_zset_active_count(eshkol_dbsp_join_result(view1)), len1,
                (unsigned long long)hash1, (unsigned long long)hash2,
                (unsigned long long)hash3, (int)all_batches_match_full);

    CHECK(all_batches_match_full, "Guard 4: incremental DBSP join matches full join on all 25 batches");
    CHECK(hash1 == hash2 && bit_identical_12, "Guard 4: replay from scratch is bit-identical");
    CHECK(hash1 == hash3 && bit_identical_13, "Guard 4: replay canonical bytes are shard-count invariant");

    eshkol_zset_buffer_free(bytes1);
    eshkol_zset_buffer_free(bytes2);
    eshkol_zset_buffer_free(bytes3);
    eshkol_dbsp_join_destroy(view1);
    eshkol_dbsp_join_destroy(view2);
    eshkol_dbsp_join_destroy(view3);

    /* --- GUARD 5: DBSP Non-Commutative Order Negative Control --- */
    eshkol_zset_t* d1 = eshkol_zset_create(NUM_SHARDS);
    eshkol_zset_t* d2 = eshkol_zset_create(NUM_SHARDS);
    eshkol_zset_add(d1, 10, 1, 20, 3);
    eshkol_zset_add(d1, 11, 2, 21, 5);
    eshkol_zset_add(d1, 12, 3, 22, -2);
    eshkol_zset_add(d2, 10, 1, 20, 2);
    eshkol_zset_add(d2, 11, 2, 21, -1);
    eshkol_zset_add(d2, 12, 3, 22, 4);

    /* Standard commutative Abelian Z-set addition: [d1, d2] == [d2, d1]. */
    eshkol_zset_t* comm_12 = eshkol_zset_create(NUM_SHARDS);
    eshkol_zset_t* comm_21 = eshkol_zset_create(NUM_SHARDS);
    eshkol_zset_apply_delta(comm_12, d1);
    eshkol_zset_apply_delta(comm_12, d2);
    eshkol_zset_apply_delta(comm_21, d2);
    eshkol_zset_apply_delta(comm_21, d1);
    uint64_t h_comm_12 = eshkol_zset_canonical_hash(comm_12);
    uint64_t h_comm_21 = eshkol_zset_canonical_hash(comm_21);

    /* Deliberately non-commutative merge(old, d) = 2*old + d: [d1, d2] != [d2, d1]. */
    eshkol_zset_t* nc_12 = eshkol_zset_create(NUM_SHARDS);
    eshkol_zset_t* nc_21 = eshkol_zset_create(NUM_SHARDS);
    eshkol_zset_apply_delta_custom(nc_12, d1, non_commutative_merge_fn, nullptr);
    eshkol_zset_apply_delta_custom(nc_12, d2, non_commutative_merge_fn, nullptr);
    eshkol_zset_apply_delta_custom(nc_21, d2, non_commutative_merge_fn, nullptr);
    eshkol_zset_apply_delta_custom(nc_21, d1, non_commutative_merge_fn, nullptr);
    uint64_t h_nc_12 = eshkol_zset_canonical_hash(nc_12);
    uint64_t h_nc_21 = eshkol_zset_canonical_hash(nc_21);

    std::printf("  [Guard 5] Commutative hashes: 0x%016llx vs 0x%016llx (equal=%d); "
                "Non-commutative hashes: 0x%016llx vs 0x%016llx (distinct=%d)\n",
                (unsigned long long)h_comm_12, (unsigned long long)h_comm_21,
                (int)(h_comm_12 == h_comm_21),
                (unsigned long long)h_nc_12, (unsigned long long)h_nc_21,
                (int)(h_nc_12 != h_nc_21));

    CHECK(h_comm_12 == h_comm_21, "Guard 5: Abelian Z-set addition is order-independent");
    CHECK(h_nc_12 != h_nc_21, "Guard 5: non-commutative merge fails reordered replay check");
    CHECK(eshkol_zset_weight(nc_12, 10, 1, 20) == 8 && eshkol_zset_weight(nc_21, 10, 1, 20) == 7,
          "Guard 5: exact non-commutative multiplicity divergence (2*3+2=8 vs 2*2+3=7)");

    eshkol_zset_destroy(d1);
    eshkol_zset_destroy(d2);
    eshkol_zset_destroy(comm_12);
    eshkol_zset_destroy(comm_21);
    eshkol_zset_destroy(nc_12);
    eshkol_zset_destroy(nc_21);

    /* --- GUARD 6: DBSP Zero-Zombie-Node Leak Check (10,000 insert/retract cycles) --- */
    eshkol_zset_t* churn = eshkol_zset_create(NUM_SHARDS);
    const size_t BASELINE_NODES = 64;
    for (size_t i = 0; i < BASELINE_NODES; ++i) {
        eshkol_zset_add(churn, 1000 + i, 50, 2000 + i, 1);
    }
    const uint64_t baseline_hash = eshkol_zset_canonical_hash(churn);
    CHECK(eshkol_zset_active_count(churn) == BASELINE_NODES, "Guard 6: baseline active == 64");

    bool all_checkpoints_clean = true;
    for (uint64_t cycle = 0; cycle < 10000; ++cycle) {
        uint64_t s = 500000ULL + cycle;
        uint64_t p = 77ULL + (cycle % 7);
        uint64_t o = 900000ULL + (cycle * 13ULL);
        /* Multi-step add (+3) and split retract (-1, -2) -> net weight 0 -> pruned immediately. */
        eshkol_zset_add(churn, s, p, o, 3);
        eshkol_zset_add(churn, s, p, o, -1);
        eshkol_zset_add(churn, s, p, o, -2);
        if (eshkol_zset_weight(churn, s, p, o) != 0) all_checkpoints_clean = false;

        if (((cycle + 1) % 1000) == 0) {
            if (eshkol_zset_active_count(churn) != BASELINE_NODES) {
                all_checkpoints_clean = false;
            }
        }
    }

    const size_t final_active = eshkol_zset_active_count(churn);
    const uint64_t final_hash = eshkol_zset_canonical_hash(churn);
    const size_t zombie_nodes = (final_active >= BASELINE_NODES) ? (final_active - BASELINE_NODES) : 999999;

    std::printf("  [Guard 6] 10,000 insert/retract cycles: baseline=%zu, final=%zu, "
                "zombie_nodes=%zu, hash_preserved=%d\n",
                BASELINE_NODES, final_active, zombie_nodes,
                (int)(final_hash == baseline_hash));

    CHECK(all_checkpoints_clean, "Guard 6: zero zombie nodes at every 1k-cycle checkpoint");
    CHECK(final_active == BASELINE_NODES, "Guard 6: final active count strictly equals baseline");
    CHECK(final_hash == baseline_hash, "Guard 6: canonical hash preserved across 10k churn cycles");

    eshkol_zset_destroy(churn);
}

/* ============================================================================
 * Test Group C: Graded Interval-Taylor Certified Neural World Model (Guards 7, 8, 9)
 * ============================================================================ */

void test_interval_taylor_enclosure() {
    std::printf("\n=== CE-8 Part 3: Graded Interval-Taylor Certified Neural World Model ===\n");

    /*
     * Construct a 4-layer smooth neural world model (3D input -> 2D output) with
     * correlated weights and residual skip connections where naive order-0 IBP
     * suffers dependency inflation while order-2 Interval-Taylor cancels linear
     * dependencies cleanly.
     */
    eshkol_tm_world_model_t* model = eshkol_tm_model_create(3);
    CHECK(model != nullptr, "world model created");

    /* Layer 1: 3 -> 3, tanh + residual skip */
    const double W1[9] = {
         0.55, -0.45,  0.25,
        -0.50,  0.60, -0.30,
         0.35,  0.40, -0.55
    };
    const double b1[3] = { 0.10, -0.05, 0.08 };
    CHECK(eshkol_tm_model_add_layer(model, 3, W1, b1, ESHKOL_TM_ACT_TANH, true),
          "layer 1 (tanh + residual) added");

    /* Layer 2: 3 -> 3, softplus + residual skip */
    const double W2[9] = {
        -0.40,  0.50, -0.20,
         0.45, -0.35,  0.30,
        -0.25, -0.45,  0.50
    };
    const double b2[3] = { -0.10, 0.15, -0.05 };
    CHECK(eshkol_tm_model_add_layer(model, 3, W2, b2, ESHKOL_TM_ACT_SOFTPLUS, true),
          "layer 2 (softplus + residual) added");

    /* Layer 3: 3 -> 2, sin */
    const double W3[6] = {
         0.60, -0.55,  0.30,
        -0.45,  0.50, -0.40
    };
    const double b3[2] = { 0.05, -0.10 };
    CHECK(eshkol_tm_model_add_layer(model, 2, W3, b3, ESHKOL_TM_ACT_SIN, false),
          "layer 3 (sin) added");

    /* Layer 4: 2 -> 2, identity readout with opposing linear combination */
    const double W4[4] = {
         1.20, -0.95,
        -0.85,  1.15
    };
    const double b4[2] = { 0.02, -0.03 };
    CHECK(eshkol_tm_model_add_layer(model, 2, W4, b4, ESHKOL_TM_ACT_IDENTITY, false),
          "layer 4 (identity readout) added");

    const eshkol_tm_interval_t input_box[3] = {
        { -0.15,  0.15 },
        {  0.05,  0.35 },
        { -0.25,  0.05 }
    };

    eshkol_tm_interval_t cert_bounds[2];
    eshkol_tm_interval_t naive_bounds[2];

    eshkol_tm_status_t st_cert = eshkol_tm_model_certify(model, input_box, cert_bounds);
    eshkol_tm_status_t st_naive = eshkol_tm_model_naive_interval(model, input_box, naive_bounds);
    CHECK(st_cert == ESHKOL_TM_OK, "Guard 7: Taylor-2 certification returned ESHKOL_TM_OK");
    CHECK(st_naive == ESHKOL_TM_OK, "Guard 8: naive interval returned ESHKOL_TM_OK");

    /* --- GUARD 7: 100,000-Sample Monte Carlo Soundness Guard (+ all 8 box corners) --- */
    const int N_SAMPLES = 100000;
    SplitMix64 rng(0x7A710202ULL);
    double mc_min[2] = { std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity() };
    double mc_max[2] = { -std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity() };
    uint64_t taylor_violations = 0;
    uint64_t naive_violations = 0;

    /* Evaluate all 8 corners first. */
    for (int mask = 0; mask < 8; ++mask) {
        double x[3] = {
            (mask & 1) ? input_box[0].hi : input_box[0].lo,
            (mask & 2) ? input_box[1].hi : input_box[1].lo,
            (mask & 4) ? input_box[2].hi : input_box[2].lo
        };
        double y[2] = { 0.0, 0.0 };
        eshkol_tm_model_eval_point(model, x, y);
        for (int o = 0; o < 2; ++o) {
            if (y[o] < mc_min[o]) mc_min[o] = y[o];
            if (y[o] > mc_max[o]) mc_max[o] = y[o];
            if (y[o] < cert_bounds[o].lo || y[o] > cert_bounds[o].hi) ++taylor_violations;
            if (y[o] < naive_bounds[o].lo || y[o] > naive_bounds[o].hi) ++naive_violations;
        }
    }

    /* Evaluate 100,000 uniform random points across the input box. */
    for (int s = 0; s < N_SAMPLES; ++s) {
        double x[3] = {
            rng.next_range(input_box[0].lo, input_box[0].hi),
            rng.next_range(input_box[1].lo, input_box[1].hi),
            rng.next_range(input_box[2].lo, input_box[2].hi)
        };
        double y[2] = { 0.0, 0.0 };
        eshkol_tm_model_eval_point(model, x, y);
        for (int o = 0; o < 2; ++o) {
            if (y[o] < mc_min[o]) mc_min[o] = y[o];
            if (y[o] > mc_max[o]) mc_max[o] = y[o];
            if (y[o] < cert_bounds[o].lo || y[o] > cert_bounds[o].hi) ++taylor_violations;
            if (y[o] < naive_bounds[o].lo || y[o] > naive_bounds[o].hi) ++naive_violations;
        }
    }

    /* --- GUARD 8: Tightness Ratio vs Monte Carlo Range & Naive Interval Comparison --- */
    double max_taylor_ratio = 0.0;
    for (int o = 0; o < 2; ++o) {
        double mc_w = mc_max[o] - mc_min[o];
        double tm_w = cert_bounds[o].hi - cert_bounds[o].lo;
        double ibp_w = naive_bounds[o].hi - naive_bounds[o].lo;
        double tm_ratio = tm_w / mc_w;
        double ibp_ratio = ibp_w / mc_w;
        if (tm_ratio > max_taylor_ratio) max_taylor_ratio = tm_ratio;

        std::printf("  [Guard 7/8] out[%d]: mc=[%.5f, %.5f] (w=%.5f) | "
                    "taylor2=[%.5f, %.5f] (w=%.5f, %.2fx MC) | "
                    "naive_ibp=[%.5f, %.5f] (w=%.5f, %.2fx MC)\n",
                    o, mc_min[o], mc_max[o], mc_w,
                    cert_bounds[o].lo, cert_bounds[o].hi, tm_w, tm_ratio,
                    naive_bounds[o].lo, naive_bounds[o].hi, ibp_w, ibp_ratio);

        CHECK(mc_w > 1e-4, "Guard 8: non-degenerate Monte Carlo range");
        CHECK(tm_ratio < 10.0, "Guard 8: Taylor-2 tightness ratio < 10.0x MC true range");
        CHECK(tm_w < ibp_w, "Guard 8: Taylor-2 enclosure strictly tighter than naive IBP");
    }

    std::printf("  [Guard 7] Soundness over 100,000 random + 8 corner samples: "
                "taylor_violations=%llu, naive_violations=%llu, max_taylor_ratio=%.3fx\n",
                (unsigned long long)taylor_violations,
                (unsigned long long)naive_violations,
                max_taylor_ratio);

    CHECK(taylor_violations == 0, "Guard 7: 100,000/100,000 samples strictly inside certified Taylor-2 enclosure");
    CHECK(naive_violations == 0, "Guard 7: naive interval baseline also sound");

    eshkol_tm_model_destroy(model);

    /* Secondary smooth model verifying SIGMOID and SQUARE activations as well. */
    eshkol_tm_world_model_t* model2 = eshkol_tm_model_create(2);
    const double M2_W1[4] = { 0.6, -0.5, -0.4, 0.7 };
    const double M2_b1[2] = { 0.1, -0.2 };
    const double M2_W2[2] = { 0.8, -0.6 };
    const double M2_b2[1] = { 0.05 };
    CHECK(eshkol_tm_model_add_layer(model2, 2, M2_W1, M2_b1, ESHKOL_TM_ACT_SIGMOID, false),
          "secondary model sigmoid layer added");
    CHECK(eshkol_tm_model_add_layer(model2, 1, M2_W2, M2_b2, ESHKOL_TM_ACT_SQUARE, false),
          "secondary model square layer added");
    const eshkol_tm_interval_t box2[2] = { { -0.2, 0.2 }, { -0.1, 0.3 } };
    eshkol_tm_interval_t cert2[1];
    CHECK(eshkol_tm_model_certify(model2, box2, cert2) == ESHKOL_TM_OK,
          "secondary sigmoid+square model certified");
    uint64_t v2 = 0;
    for (int s = 0; s < 10000; ++s) {
        double x[2] = { rng.next_range(-0.2, 0.2), rng.next_range(-0.1, 0.3) };
        double y[1] = { 0.0 };
        eshkol_tm_model_eval_point(model2, x, y);
        if (y[0] < cert2[0].lo || y[0] > cert2[0].hi) ++v2;
    }
    CHECK(v2 == 0, "secondary sigmoid+square model: 0/10,000 soundness violations");
    eshkol_tm_model_destroy(model2);

    /* --- GUARD 9: Interval-Taylor Non-Lipschitz Negative Control --- */
    eshkol_tm_world_model_t* step_model = eshkol_tm_model_create(2);
    CHECK(eshkol_tm_model_add_layer(step_model, 2, M2_W1, M2_b1, ESHKOL_TM_ACT_TANH, false),
          "Guard 9: smooth prefix layer added");
    CHECK(eshkol_tm_model_add_layer(step_model, 1, M2_W2, M2_b2, ESHKOL_TM_ACT_HEAVISIDE_STEP, false),
          "Guard 9: discontinuous Heaviside step layer added");

    eshkol_tm_interval_t uncert_tm[1] = { { 0.0, 0.0 } };
    eshkol_tm_interval_t uncert_ibp[1] = { { 0.0, 0.0 } };
    eshkol_tm_status_t st_step_tm = eshkol_tm_model_certify(step_model, box2, uncert_tm);
    eshkol_tm_status_t st_step_ibp = eshkol_tm_model_naive_interval(step_model, box2, uncert_ibp);

    bool tm_uncert_inf = std::isinf(uncert_tm[0].lo) && (uncert_tm[0].lo < 0.0) &&
                         std::isinf(uncert_tm[0].hi) && (uncert_tm[0].hi > 0.0);
    bool ibp_uncert_inf = std::isinf(uncert_ibp[0].lo) && (uncert_ibp[0].lo < 0.0) &&
                          std::isinf(uncert_ibp[0].hi) && (uncert_ibp[0].hi > 0.0);

    std::printf("  [Guard 9] Heaviside non-Lipschitz negative control: "
                "tm_status=%d (expected %d), bounds=[%.1f, %.1f]; ibp_status=%d\n",
                (int)st_step_tm, (int)ESHKOL_TM_ERR_NON_LIPSCHITZ,
                uncert_tm[0].lo, uncert_tm[0].hi, (int)st_step_ibp);

    CHECK(st_step_tm == ESHKOL_TM_ERR_NON_LIPSCHITZ,
          "Guard 9: Taylor-2 fails closed with ESHKOL_TM_ERR_NON_LIPSCHITZ");
    CHECK(tm_uncert_inf,
          "Guard 9: Taylor-2 outputs uncertified [-inf, +inf] on Heaviside step");
    CHECK(st_step_ibp == ESHKOL_TM_ERR_NON_LIPSCHITZ && ibp_uncert_inf,
          "Guard 9: naive IBP also fails closed with ESHKOL_TM_ERR_NON_LIPSCHITZ and [-inf, +inf]");

    eshkol_tm_model_destroy(step_model);
}

} /* namespace */

int main() {
    test_staged_zero_alloc_ad();
    test_dbsp_zset_kg();
    test_interval_taylor_enclosure();

    std::printf("\nCE-8 Summary: %d passed, %d failed\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
