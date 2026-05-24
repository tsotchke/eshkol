/* test_repl_jit_executetagged.cpp
 *
 * Regression test for the qLLM-bridge value-capture bug fixed on
 * 2026-05-08.  Symptom: every call to eshkol_eval_string("42", NULL)
 * (and any other top-level expression with no user-defined `main`)
 * came back tagged INT64 but with data.int_val = 0 — the JIT-emitted
 * `main` always returned `ret i32 0`, throwing away the actual result.
 *
 * Fix: codegen now emits `eshkol_repl_capture_last_value(&tagged)` for
 * the last value-producing top-level expression and
 * ReplJITContext::executeTagged reads the captured slot in preference
 * to the truncated-i32 main return.
 *
 * This test exercises the smallest possible surface — the public C API
 * `eshkol_eval_string` — and asserts that the value-fill path is
 * actually wired through.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <eshkol/eshkol.h>
#include <eshkol/core/introspection.h>

namespace {

int g_failures = 0;
int g_total    = 0;

#define CHECK(cond, name) do {                                          \
        ++g_total;                                                      \
        if (cond) {                                                     \
            std::printf("[PASS] %s\n", name);                           \
        } else {                                                        \
            std::printf("[FAIL] %s  (%s:%d)\n",                         \
                        name, __FILE__, __LINE__);                      \
            ++g_failures;                                               \
        }                                                               \
    } while (0)

void test_int_literal(void) {
    eshkol_tagged_value_t v = eshkol_eval_string("42", nullptr);
    CHECK(v.type == ESHKOL_VALUE_INT64,
          "int literal: type == INT64");
    CHECK(v.data.int_val == 42,
          "int literal: int_val == 42 (value-capture bug regression)");
}

void test_int_literal_other(void) {
    eshkol_tagged_value_t v = eshkol_eval_string("123", nullptr);
    CHECK(v.type == ESHKOL_VALUE_INT64,
          "int literal 123: type == INT64");
    CHECK(v.data.int_val == 123,
          "int literal 123: int_val == 123");
}

void test_addition(void) {
    eshkol_tagged_value_t v = eshkol_eval_string("(+ 1 2)", nullptr);
    CHECK(v.type == ESHKOL_VALUE_INT64,
          "(+ 1 2): type == INT64");
    CHECK(v.data.int_val == 3,
          "(+ 1 2): int_val == 3");
}

void test_multiplication(void) {
    eshkol_tagged_value_t v = eshkol_eval_string("(* 7 6)", nullptr);
    CHECK(v.type == ESHKOL_VALUE_INT64,
          "(* 7 6): type == INT64");
    CHECK(v.data.int_val == 42,
          "(* 7 6): int_val == 42");
}

void test_double_literal(void) {
    eshkol_tagged_value_t v = eshkol_eval_string("3.14", nullptr);
    CHECK(v.type == ESHKOL_VALUE_DOUBLE,
          "3.14: type == DOUBLE");
    CHECK(std::fabs(v.data.double_val - 3.14) < 1e-9,
          "3.14: double_val ~= 3.14");
}

void test_repl_user_variable_shadows_math_builtin(void) {
    (void)eshkol_eval_string("(define log2 \"\")", nullptr);
    eshkol_tagged_value_t v = eshkol_eval_string("log2", nullptr);
    CHECK(ESHKOL_IS_STRING_COMPAT(v),
          "REPL user variable log2 shadows math builtin across evals");
}

}  // namespace

int main(void) {
    std::printf("=== ReplJITContext::executeTagged value-capture tests ===\n");

    test_int_literal();
    test_int_literal_other();
    test_addition();
    test_multiplication();
    test_double_literal();
    test_repl_user_variable_shadows_math_builtin();

    std::printf("\nResults: %d/%d checks passed\n",
                g_total - g_failures, g_total);
    return g_failures == 0 ? 0 : 1;
}
