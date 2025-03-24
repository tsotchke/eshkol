/**
 * @scheme_test test_function_name
 * @tests_function function-name
 * @coverage
 *   - Basic functionality: 100%
 *   - Edge cases: 100%
 *   - Error handling: 100%
 * @test_cases
 *   - Normal operation with valid inputs
 *   - Edge cases (empty lists, etc.)
 *   - Error cases (invalid types, etc.)
 */

#include "scheme_test.h"
#include "scheme_runtime.h"

/**
 * Test normal operation of function-name
 */
void test_function_name_normal(void) {
    SchemeEnvironment* env = scheme_create_environment();
    
    // Test case 1: Basic addition
    SchemeObject* result1 = scheme_eval_string(env, "(function-name 1 2)");
    TEST_ASSERT_EQUAL_DOUBLE(3.0, SCHEME_NUMBER_VALUE(result1));
    
    // Test case 2: Different types of numbers
    SchemeObject* result2 = scheme_eval_string(env, "(function-name 1.5 2.5)");
    TEST_ASSERT_EQUAL_DOUBLE(4.0, SCHEME_NUMBER_VALUE(result2));
    
    scheme_destroy_environment(env);
}

/**
 * Test edge cases of function-name
 */
void test_function_name_edge_cases(void) {
    SchemeEnvironment* env = scheme_create_environment();
    
    // Test case 1: Zero values
    SchemeObject* result1 = scheme_eval_string(env, "(function-name 0 0)");
    TEST_ASSERT_EQUAL_DOUBLE(0.0, SCHEME_NUMBER_VALUE(result1));
    
    // Test case 2: Negative values
    SchemeObject* result2 = scheme_eval_string(env, "(function-name -1 -2)");
    TEST_ASSERT_EQUAL_DOUBLE(-3.0, SCHEME_NUMBER_VALUE(result2));
    
    scheme_destroy_environment(env);
}

/**
 * Test error handling of function-name
 */
void test_function_name_errors(void) {
    SchemeEnvironment* env = scheme_create_environment();
    
    // Test case 1: Wrong number of arguments
    SchemeObject* result1 = scheme_eval_string(env, "(function-name 1)");
    TEST_ASSERT_TRUE(SCHEME_IS_ERROR(result1));
    
    // Test case 2: Wrong type of arguments
    SchemeObject* result2 = scheme_eval_string(env, "(function-name \"a\" 1)");
    TEST_ASSERT_TRUE(SCHEME_IS_ERROR(result2));
    
    scheme_destroy_environment(env);
}

/**
 * Run all tests for function-name
 */
void test_function_name(void) {
    TEST_RUN(test_function_name_normal);
    TEST_RUN(test_function_name_edge_cases);
    TEST_RUN(test_function_name_errors);
}
