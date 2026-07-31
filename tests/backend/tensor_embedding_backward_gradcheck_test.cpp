/**
 * @file tensor_embedding_backward_gradcheck_test.cpp
 * @brief Gradient check for the embedding-lookup backward pass (ESH-0230).
 *
 * The embedding forward is a gather, y[i,:] = W[idx[i],:], so its adjoint is a
 * scatter-add, dL/dW[idx[i],:] += dL/dy[i,:]. Two properties of that adjoint
 * are what this test exists to pin down, because both have a failure mode that
 * produces a plausible number rather than an obvious break:
 *
 *   DUPLICATE INDICES ACCUMULATE.  A row looked up k times must receive the
 *   sum of all k upstream rows. Writing `=` instead of `+=` is the classic
 *   scatter-add bug: the gradient stays the right shape and the right order of
 *   magnitude, it is simply too small for every repeated token — which is most
 *   tokens in real text. The fixture below looks row 3 up TWICE.
 *
 *   UNSELECTED ROWS ARE EXACTLY ZERO.  The adjoint of a gather is genuinely
 *   sparse. A rule that spreads the upstream gradient over other rows (the
 *   pre-ESH-0230 "accumulate into row 0" stub did exactly that) is wrong, not
 *   approximate. This test asserts bitwise 0.0, not "small".
 *
 * The loss is linear in W, so central finite differences of the real forward
 * are exact to rounding and the FD cross-check runs at a ~1e-12 bar instead of
 * the usual 1e-6.
 *
 * The test also pins the gradient-buffer LIFETIME. The dispatcher brackets each
 * node's backward in an arena scope; the destination buffer W.tensor_gradient is
 * allocated lazily on first accumulation. When those two shared one arena, the
 * scope pop rewound over the destination buffer and the whole chain returned
 * zero. Check 5 reproduces that precisely: it allocates from the global arena
 * between two dispatches, so if the destination had been reclaimed the squatter
 * would land on top of it and the second accumulation would not read 2x.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <eshkol/eshkol.h>
#include <eshkol/backend/tensor_backward.h>

#include "../../lib/core/arena_memory.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#define ESHKOL_HAVE_FORK_DEATH_TESTS 1
#endif

extern "C" arena_t* get_global_arena();

namespace {

int g_passed = 0;
int g_failed = 0;

void report(const char* name, bool ok, const char* detail = nullptr) {
    std::printf("  %-46s %s", name, ok ? "PASS" : "FAIL");
    if (detail) std::printf("   [%s]", detail);
    std::printf("\n");
    if (ok) ++g_passed; else ++g_failed;
}

/* ---- Fixture -----------------------------------------------------------
 * vocab_size = 5, d_model = 3, four lookups hitting rows 3, 0, 3, 1.
 *   - row 3 is looked up TWICE (duplicate accumulation)
 *   - rows 2 and 4 are never looked up (exact zero)
 *   - row 0 is looked up once and is NOT the first lookup, so a rule that
 *     dumped everything into row 0 would be visibly wrong here.
 */
constexpr int64_t kVocab      = 5;
constexpr int64_t kDModel     = 3;
constexpr int64_t kNumIndices = 4;

const double kIndices[kNumIndices] = { 3.0, 0.0, 3.0, 1.0 };

/* Weight matrix [kVocab, kDModel]; values only matter to the forward. */
const double kWeights[kVocab * kDModel] = {
    /* row 0 */  0.5,  -1.25,   2.0,
    /* row 1 */ -0.75,  0.125, -3.5,
    /* row 2 */  1.5,   0.25,  -0.5,
    /* row 3 */  2.25, -0.5,    0.75,
    /* row 4 */ -1.0,   3.0,    0.375,
};

/* Upstream cotangent dL/dy, [kNumIndices, kDModel]. Deliberately distinct per
 * row so that summing rows 0 and 2 into weight row 3 is distinguishable from
 * taking either one alone. */
const double kCotangent[kNumIndices * kDModel] = {
    /* lookup 0 -> row 3 */  1.0,   2.0,  -0.5,
    /* lookup 1 -> row 0 */  0.25, -1.5,   3.0,
    /* lookup 2 -> row 3 */ -4.0,   0.75,  1.25,
    /* lookup 3 -> row 1 */  0.125, 0.5,  -2.25,
};

/** @brief The forward: L(W) = sum_{i,d} c[i,d] * W[idx[i],d]. */
double loss(const double* W) {
    double acc = 0.0;
    for (int64_t i = 0; i < kNumIndices; i++) {
        int64_t row = (int64_t)kIndices[i];
        for (int64_t d = 0; d < kDModel; d++)
            acc += kCotangent[i * kDModel + d] * W[row * kDModel + d];
    }
    return acc;
}

/** @brief Closed-form scatter-add reference: dL/dW[r,:] = sum over lookups
 *  landing on r of c[i,:]. Written as a literal restatement of the definition
 *  so it cannot share a bug with the implementation under test. */
std::vector<double> expected_grad() {
    std::vector<double> g((size_t)(kVocab * kDModel), 0.0);
    for (int64_t i = 0; i < kNumIndices; i++) {
        int64_t row = (int64_t)kIndices[i];
        for (int64_t d = 0; d < kDModel; d++)
            g[(size_t)(row * kDModel + d)] += kCotangent[i * kDModel + d];
    }
    return g;
}

/** @brief Assemble the AD nodes for one embedding backward and dispatch it.
 *  @p w_node keeps its tensor_gradient across calls so the caller can test
 *  accumulation over repeated dispatches. */
struct Fixture {
    ad_node_t w{};
    ad_node_t idx{};
    ad_node_t out{};
    int64_t   w_shape[2]   = { kVocab, kDModel };
    int64_t   idx_shape[1] = { kNumIndices };
    int64_t   out_shape[2] = { kNumIndices, kDModel };
    std::vector<double> y;
    std::vector<double> dy;

    Fixture() {
        y.assign((size_t)(kNumIndices * kDModel), 0.0);
        for (int64_t i = 0; i < kNumIndices; i++) {
            int64_t row = (int64_t)kIndices[i];
            for (int64_t d = 0; d < kDModel; d++)
                y[(size_t)(i * kDModel + d)] = kWeights[row * kDModel + d];
        }
        dy.assign(kCotangent, kCotangent + (size_t)(kNumIndices * kDModel));

        w.type          = AD_NODE_VARIABLE;
        w.tensor_value  = (void*)kWeights;
        w.shape         = w_shape;
        w.ndim          = 2;
        w.tensor_gradient = nullptr;   /* force the lazy-allocation path */

        idx.type         = AD_NODE_CONSTANT;
        idx.tensor_value = (void*)kIndices;
        idx.shape        = idx_shape;
        idx.ndim         = 1;

        out.type   = AD_NODE_TENSOR_EMBEDDING;
        out.input1 = &w;
        out.input2 = &idx;
        out.shape  = out_shape;
        out.ndim   = 2;
        out.tensor_value    = y.data();
        out.tensor_gradient = dy.data();

        int64_t* p = (int64_t*)&out.params;
        p[0] = kNumIndices;
        p[1] = kDModel;
        p[2] = kVocab;
    }

    void dispatch() { eshkol_tensor_backward_dispatch(&out); }
    const double* grad() const { return (const double*)w.tensor_gradient; }
};

double max_abs_diff(const double* a, const double* b, size_t n) {
    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

#if defined(ESHKOL_HAVE_FORK_DEATH_TESTS)
/** @brief Run @p body in a forked child and report whether it exited nonzero.
 *  The refusals under test go through eshkol_fatal(), which exits the process;
 *  a child is the only way to assert "this call refuses" without taking the
 *  test down with it. */
bool refuses(void (*body)()) {
    std::fflush(stdout);
    std::fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        /* Silence the child's diagnostic so the test output stays readable;
         * the exit status is what we assert on. */
        FILE* devnull = std::freopen("/dev/null", "w", stderr);
        (void)devnull;
        body();
        /* Reached only if the call did NOT refuse. */
        std::_Exit(0);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return false;
    if (WIFSIGNALED(status)) return true;                 /* abort also counts */
    return WIFEXITED(status) && WEXITSTATUS(status) != 0;
}

void body_missing_index() {
    Fixture f;
    f.out.input2 = nullptr;      /* producer failed to thread the index tensor */
    f.dispatch();
}

void body_fractional_index() {
    static double bad[kNumIndices] = { 3.0, 0.0, 2.5, 1.0 };
    Fixture f;
    f.idx.tensor_value = bad;
    f.dispatch();
}

void body_out_of_range_index() {
    static double bad[kNumIndices] = { 3.0, 0.0, kVocab + 1.0, 1.0 };
    Fixture f;
    f.idx.tensor_value = bad;
    f.dispatch();
}
#endif  /* ESHKOL_HAVE_FORK_DEATH_TESTS */

}  // namespace

int main() {
    std::printf("=== embedding backward gradient check (ESH-0230) ===\n");

    const std::vector<double> want = expected_grad();
    const size_t w_total = (size_t)(kVocab * kDModel);

    /* ---- 1. exact scatter-add against the closed form ------------------ */
    Fixture f;
    f.dispatch();
    const double* got = f.grad();
    if (!got) {
        report("scatter-add vs closed form", false, "no gradient buffer produced");
    } else {
        double worst = max_abs_diff(got, want.data(), w_total);
        char detail[96];
        std::snprintf(detail, sizeof detail, "max abs err = %.3e", worst);
        report("scatter-add vs closed form", worst == 0.0, detail);
    }

    /* ---- 2. central finite differences of the real forward ------------- */
    if (got) {
        std::vector<double> fd(w_total, 0.0);
        std::vector<double> Wp(kWeights, kWeights + w_total);
        /* L is EXACTLY linear in W, so the central difference has no truncation
         * term at any step size and only the cancellation term eps*|L|/(2h)
         * remains. A large step is therefore strictly better here: h = 1e-4 sits
         * on a ~1e-11 rounding floor, h = 1 pushes it to ~1e-15. Picking the
         * step for the function's actual analytic structure rather than reaching
         * for a habitual small h is what lets this run at a 1e-12 bar. */
        const double h = 1.0;
        for (size_t k = 0; k < w_total; k++) {
            double saved = Wp[k];
            Wp[k] = saved + h; double lp = loss(Wp.data());
            Wp[k] = saved - h; double lm = loss(Wp.data());
            Wp[k] = saved;
            fd[k] = (lp - lm) / (2.0 * h);
        }
        double worst = max_abs_diff(got, fd.data(), w_total);
        char detail[96];
        std::snprintf(detail, sizeof detail, "max abs err = %.3e (bar 1e-12)", worst);
        report("scatter-add vs central finite differences", worst < 1e-12, detail);
    } else {
        report("scatter-add vs central finite differences", false, "no gradient");
    }

    /* ---- 3. rows never looked up are EXACTLY zero ---------------------- */
    if (got) {
        bool ok = true;
        char detail[96] = "rows 2 and 4 untouched";
        for (int64_t row : { (int64_t)2, (int64_t)4 }) {
            for (int64_t d = 0; d < kDModel; d++) {
                double v = got[row * kDModel + d];
                if (v != 0.0) {
                    ok = false;
                    std::snprintf(detail, sizeof detail,
                                  "W[%lld][%lld] = %.17g, want exactly 0",
                                  (long long)row, (long long)d, v);
                }
            }
        }
        report("unselected rows are bitwise zero", ok, detail);
    } else {
        report("unselected rows are bitwise zero", false, "no gradient");
    }

    /* ---- 4. duplicate indices ACCUMULATE, not overwrite ---------------- */
    if (got) {
        /* Row 3 is looked up by lookups 0 and 2. The sum is the only correct
         * answer; either lookup alone is what an overwriting scatter yields. */
        bool ok = true;
        char detail[128] = "row 3 = c[0] + c[2]";
        for (int64_t d = 0; d < kDModel; d++) {
            double sum   = kCotangent[0 * kDModel + d] + kCotangent[2 * kDModel + d];
            double first = kCotangent[0 * kDModel + d];
            double last  = kCotangent[2 * kDModel + d];
            double v     = got[3 * kDModel + d];
            if (v != sum) {
                ok = false;
                const char* looks_like =
                    (v == first) ? " (== first lookup: overwritten by the LAST write?)"
                  : (v == last)  ? " (== last lookup: overwriting scatter)"
                                 : "";
                std::snprintf(detail, sizeof detail,
                              "W[3][%lld] = %.17g, want %.17g%s",
                              (long long)d, v, sum, looks_like);
            }
        }
        report("duplicate indices accumulate", ok, detail);
    } else {
        report("duplicate indices accumulate", false, "no gradient");
    }

    /* ---- 5. the destination buffer outlives the backward arena scope ---- */
    {
        /* Squat on the global arena between the two dispatches. If the first
         * dispatch's lazily-allocated destination buffer had been reclaimed by
         * the backward scope pop, this allocation would be handed the same
         * address range and the second accumulation would not read 2x. */
        const size_t squat_n = 4096;
        double* squat = (double*)arena_allocate_zeroed(get_global_arena(),
                                                      squat_n * sizeof(double));
        if (squat) for (size_t i = 0; i < squat_n; i++) squat[i] = 1.2345e300;

        f.dispatch();   /* second accumulation into the same W node */
        const double* twice = f.grad();
        if (!twice) {
            report("gradient survives the backward arena scope", false, "buffer lost");
        } else {
            double worst = 0.0;
            for (size_t k = 0; k < w_total; k++) {
                double d = std::fabs(twice[k] - 2.0 * want[k]);
                if (d > worst) worst = d;
            }
            char detail[96];
            std::snprintf(detail, sizeof detail, "2x accumulation, max abs err = %.3e", worst);
            report("gradient survives the backward arena scope", worst == 0.0, detail);
        }
    }

    /* ---- 6-8. refusals rather than plausible wrong gradients ----------- */
#if defined(ESHKOL_HAVE_FORK_DEATH_TESTS)
    report("refuses a missing index operand",  refuses(&body_missing_index));
    report("refuses a fractional index",       refuses(&body_fractional_index));
    report("refuses an out-of-range index",    refuses(&body_out_of_range_index));
#else
    std::printf("  (refusal checks skipped: no fork on this platform)\n");
#endif

    /* ---- 9. the native (non-bridge) scatter also accumulates ----------- */
    {
        /* eshkol_backward_embedding is the AD_NODE_EMBEDDING rule; it takes the
         * indices as int64 rather than off the node. Same duplicate-index
         * property, so cover it here rather than leaving it unexercised. */
        const int64_t idx_i64[kNumIndices] = { 3, 0, 3, 1 };
        std::vector<double> dW(w_total, 0.0);
        eshkol_backward_embedding(kCotangent, idx_i64, dW.data(),
                                  kNumIndices, kDModel, kVocab);
        double worst = max_abs_diff(dW.data(), want.data(), w_total);
        char detail[96];
        std::snprintf(detail, sizeof detail, "max abs err = %.3e", worst);
        report("native scatter accumulates duplicates", worst == 0.0, detail);
    }

    std::printf("=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
