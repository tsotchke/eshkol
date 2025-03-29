/**
 * @file test_list.c
 * @brief Unit tests for the list operations
 */

#include "core/list.h"
#include "core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Test list initialization
 */
static void test_list_init(void) {
    printf("Testing list initialization...\n");
    
    // Initialize the list module
    bool result = eshkol_list_init();
    assert(result);
    
    // Check that the empty list is not NULL
    assert(ESHKOL_EMPTY_LIST != NULL);
    
    // Clean up
    eshkol_list_cleanup();
    
    printf("PASS: eshkol_list_init\n");
}

/**
 * @brief Test cons operation
 */
static void test_cons(void) {
    printf("Testing cons operation...\n");
    
    // Initialize the list module
    bool result = eshkol_list_init();
    assert(result);
    
    // Create some test data
    int a = 42;
    int b = 123;
    
    // Create a pair
    EshkolPair* pair = eshkol_cons(&a, &b);
    assert(pair != NULL);
    
    // Check the pair contents
    assert(*(int*)eshkol_car(pair) == 42);
    assert(*(int*)eshkol_cdr(pair) == 123);
    
    // Clean up
    free(pair);
    eshkol_list_cleanup();
    
    printf("PASS: eshkol_cons\n");
}

/**
 * @brief Test car and cdr operations
 */
static void test_car_cdr(void) {
    printf("Testing car and cdr operations...\n");
    
    // Initialize the list module
    bool result = eshkol_list_init();
    assert(result);
    
    // Create some test data
    int a = 42;
    int b = 123;
    
    // Create a pair
    EshkolPair* pair = eshkol_cons(&a, &b);
    assert(pair != NULL);
    
    // Test car
    void* car_result = eshkol_car(pair);
    assert(car_result != NULL);
    assert(*(int*)car_result == 42);
    
    // Test cdr
    void* cdr_result = eshkol_cdr(pair);
    assert(cdr_result != NULL);
    assert(*(int*)cdr_result == 123);
    
    // Test car/cdr with NULL
    assert(eshkol_car(NULL) == NULL);
    assert(eshkol_cdr(NULL) == NULL);
    
    // Clean up
    free(pair);
    eshkol_list_cleanup();
    
    printf("PASS: eshkol_car and eshkol_cdr\n");
}

/**
 * @brief Test set-car! and set-cdr! operations
 */
static void test_set_car_cdr(void) {
    printf("Testing set-car! and set-cdr! operations...\n");
    
    // Initialize the list module
    bool result = eshkol_list_init();
    assert(result);
    
    // Create some test data
    int a = 42;
    int b = 123;
    int c = 456;
    int d = 789;
    
    // Create a pair
    EshkolPair* pair = eshkol_cons(&a, &b);
    assert(pair != NULL);
    
    // Test set-car!
    bool set_car_result = eshkol_set_car(pair, &c);
    assert(set_car_result);
    assert(*(int*)eshkol_car(pair) == 456);
    
    // Test set-cdr!
    bool set_cdr_result = eshkol_set_cdr(pair, &d);
    assert(set_cdr_result);
    assert(*(int*)eshkol_cdr(pair) == 789);
    
    // Test with NULL
    assert(!eshkol_set_car(NULL, &c));
    assert(!eshkol_set_cdr(NULL, &d));
    
    // Test with immutable pair
    pair->is_immutable = true;
    assert(!eshkol_set_car(pair, &a));
    assert(!eshkol_set_cdr(pair, &b));
    
    // Clean up
    free(pair);
    eshkol_list_cleanup();
    
    printf("PASS: eshkol_set_car and eshkol_set_cdr\n");
}

/**
 * @brief Test list creation
 */
static void test_list(void) {
    printf("Testing list creation...\n");
    
    // Initialize the list module
    bool result = eshkol_list_init();
    assert(result);
    
    // Create some test data
    int a = 1;
    int b = 2;
    int c = 3;
    
    // Create an empty list
    EshkolPair* empty_list = eshkol_list(0);
    assert(empty_list == ESHKOL_EMPTY_LIST);
    
    // Create a list with one element
    EshkolPair* list1 = eshkol_list(1, &a);
    assert(list1 != NULL);
    assert(*(int*)eshkol_car(list1) == 1);
    assert(eshkol_cdr(list1) == ESHKOL_EMPTY_LIST);
    
    // Create a list with multiple elements
    EshkolPair* list3 = eshkol_list(3, &a, &b, &c);
    assert(list3 != NULL);
    assert(*(int*)eshkol_car(list3) == 1);
    assert(eshkol_is_pair(eshkol_cdr(list3)));
    assert(*(int*)eshkol_car((EshkolPair*)eshkol_cdr(list3)) == 2);
    assert(eshkol_is_pair(eshkol_cdr((EshkolPair*)eshkol_cdr(list3))));
    assert(*(int*)eshkol_car((EshkolPair*)eshkol_cdr((EshkolPair*)eshkol_cdr(list3))) == 3);
    assert(eshkol_cdr((EshkolPair*)eshkol_cdr((EshkolPair*)eshkol_cdr(list3))) == ESHKOL_EMPTY_LIST);
    
    // Clean up
    eshkol_free_pair_chain(list1);
    eshkol_free_pair_chain(list3);
    eshkol_list_cleanup();
    
    printf("PASS: eshkol_list\n");
}

/**
 * @brief Test list predicates
 */
static void test_predicates(void) {
    printf("Testing list predicates...\n");
    
    // Initialize the list module
    bool result = eshkol_list_init();
    assert(result);
    
    // Create some test data
    int a = 1;
    int b = 2;
    
    // Create a pair and a list
    EshkolPair* pair = eshkol_cons(&a, &b);
    EshkolPair* list = eshkol_list(2, &a, &b);
    
    // Test is_pair
    assert(eshkol_is_pair(pair));
    assert(eshkol_is_pair(list));
    assert(!eshkol_is_pair(NULL));
    assert(!eshkol_is_pair(ESHKOL_EMPTY_LIST));
    
    // Test is_null
    assert(eshkol_is_null(ESHKOL_EMPTY_LIST));
    assert(!eshkol_is_null(pair));
    assert(!eshkol_is_null(list));
    assert(!eshkol_is_null(NULL));
    
    // Test is_list
    assert(eshkol_is_list(ESHKOL_EMPTY_LIST));
    assert(eshkol_is_list(list));
    assert(!eshkol_is_list(pair)); // pair is not a proper list
    assert(!eshkol_is_list(NULL));
    
    // Clean up
    free(pair);
    eshkol_free_pair_chain(list);
    eshkol_list_cleanup();
    
    printf("PASS: eshkol_is_pair, eshkol_is_null, eshkol_is_list\n");
}

/**
 * @brief Test list length
 */
static void test_list_length(void) {
    printf("Testing list length...\n");
    
    // Initialize the list module
    bool result = eshkol_list_init();
    assert(result);
    
    // Create some test data
    int a = 1;
    int b = 2;
    int c = 3;
    
    // Create lists of different lengths
    EshkolPair* empty_list = ESHKOL_EMPTY_LIST;
    EshkolPair* list1 = eshkol_list(1, &a);
    EshkolPair* list3 = eshkol_list(3, &a, &b, &c);
    
    // Create an improper list (pair)
    EshkolPair* pair = eshkol_cons(&a, &b);
    
    // Test list_length
    assert(eshkol_list_length(empty_list) == 0);
    assert(eshkol_list_length(list1) == 1);
    assert(eshkol_list_length(list3) == 3);
    assert(eshkol_list_length(pair) == -1); // improper list
    assert(eshkol_list_length(NULL) == -1); // NULL
    
    // Clean up
    eshkol_free_pair_chain(list1);
    eshkol_free_pair_chain(list3);
    free(pair);
    eshkol_list_cleanup();
    
    printf("PASS: eshkol_list_length\n");
}

/**
 * @brief Test nested car/cdr operations
 */
static void test_nested_car_cdr(void) {
    printf("Testing nested car/cdr operations...\n");
    
    // Initialize the list module
    bool result = eshkol_list_init();
    assert(result);
    
    // Create some test data
    int a = 1;
    int b = 2;
    int c = 3;
    int d = 4;
    
    // Create a nested structure
    // ((a . b) . (c . d))
    EshkolPair* inner1 = eshkol_cons(&a, &b);
    EshkolPair* inner2 = eshkol_cons(&c, &d);
    EshkolPair* outer = eshkol_cons(inner1, inner2);
    
    // Test caar
    assert(*(int*)eshkol_caar(outer) == 1);
    
    // Test cadr
    assert(*(int*)eshkol_cadr(outer) == 3);
    
    // Test cdar
    assert(*(int*)eshkol_cdar(outer) == 2);
    
    // Test cddr
    assert(*(int*)eshkol_cddr(outer) == 4);
    
    // Clean up
    free(inner1);
    free(inner2);
    free(outer);
    eshkol_list_cleanup();
    
    printf("PASS: nested car/cdr operations\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running list tests...\n");
    
    test_list_init();
    test_cons();
    test_car_cdr();
    test_set_car_cdr();
    test_list();
    test_predicates();
    test_list_length();
    test_nested_car_cdr();
    
    printf("All list tests passed!\n");
    return 0;
}
