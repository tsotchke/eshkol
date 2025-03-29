/**
 * @file test_type_predicates.c
 * @brief Test cases for type predicate functions
 * 
 * This file contains test cases for the type predicate functions in the Eshkol project.
 * Each test case verifies that the predicate correctly identifies objects of the specified type
 * and correctly rejects objects of other types.
 * 
 * NOTE: This is a template file and is not meant to be compiled directly.
 * It contains placeholder code that needs to be adapted to the actual implementation.
 * The identifiers used in this file (EshkolObject, eshkol_create_boolean, etc.) are
 * placeholders that should be replaced with the actual types and functions in your project.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/type.h"
#include "core/memory.h"
// Add other necessary includes here

/**
 * @brief Test the boolean? predicate
 * 
 * This function tests the boolean? predicate with various inputs:
 * - Boolean values (should return true)
 * - Non-boolean values (should return false)
 * - NULL (should return false)
 * - Invalid objects (should report an error and return false)
 */
void test_boolean_predicate() {
    // Setup
    printf("Testing boolean? predicate...\n");
    
    // Test with boolean values (should return true)
    EshkolObject* true_obj = eshkol_create_boolean(true);
    EshkolObject* false_obj = eshkol_create_boolean(false);
    assert(eshkol_is_boolean(true_obj));
    assert(eshkol_is_boolean(false_obj));
    
    // Test with non-boolean values (should return false)
    EshkolObject* integer_obj = eshkol_create_integer(42);
    EshkolObject* float_obj = eshkol_create_float(3.14);
    EshkolObject* string_obj = eshkol_create_string("hello");
    EshkolObject* symbol_obj = eshkol_create_symbol("symbol");
    EshkolObject* pair_obj = eshkol_cons(integer_obj, float_obj);
    EshkolObject* empty_list = eshkol_create_empty_list();
    
    assert(!eshkol_is_boolean(integer_obj));
    assert(!eshkol_is_boolean(float_obj));
    assert(!eshkol_is_boolean(string_obj));
    assert(!eshkol_is_boolean(symbol_obj));
    assert(!eshkol_is_boolean(pair_obj));
    assert(!eshkol_is_boolean(empty_list));
    
    // Test with NULL (should return false)
    assert(!eshkol_is_boolean(NULL));
    
    // Test with invalid object (should report an error and return false)
    // This test is optional and depends on how the project handles invalid objects
    // EshkolObject* invalid_obj = malloc(sizeof(EshkolObject));
    // memset(invalid_obj, 0, sizeof(EshkolObject));
    // assert(!eshkol_is_boolean(invalid_obj));
    // free(invalid_obj);
    
    // Cleanup
    eshkol_free_object(true_obj);
    eshkol_free_object(false_obj);
    eshkol_free_object(integer_obj);
    eshkol_free_object(float_obj);
    eshkol_free_object(string_obj);
    eshkol_free_object(symbol_obj);
    eshkol_free_object(pair_obj);
    eshkol_free_object(empty_list);
    
    printf("boolean? predicate tests passed!\n");
}

/**
 * @brief Test the number? predicate
 * 
 * This function tests the number? predicate with various inputs:
 * - Integer values (should return true)
 * - Float values (should return true)
 * - Non-number values (should return false)
 * - NULL (should return false)
 * - Invalid objects (should report an error and return false)
 */
void test_number_predicate() {
    // Setup
    printf("Testing number? predicate...\n");
    
    // Test with number values (should return true)
    EshkolObject* integer_obj = eshkol_create_integer(42);
    EshkolObject* float_obj = eshkol_create_float(3.14);
    assert(eshkol_is_number(integer_obj));
    assert(eshkol_is_number(float_obj));
    
    // Test with non-number values (should return false)
    EshkolObject* true_obj = eshkol_create_boolean(true);
    EshkolObject* false_obj = eshkol_create_boolean(false);
    EshkolObject* string_obj = eshkol_create_string("hello");
    EshkolObject* symbol_obj = eshkol_create_symbol("symbol");
    EshkolObject* pair_obj = eshkol_cons(integer_obj, float_obj);
    EshkolObject* empty_list = eshkol_create_empty_list();
    
    assert(!eshkol_is_number(true_obj));
    assert(!eshkol_is_number(false_obj));
    assert(!eshkol_is_number(string_obj));
    assert(!eshkol_is_number(symbol_obj));
    assert(!eshkol_is_number(pair_obj));
    assert(!eshkol_is_number(empty_list));
    
    // Test with NULL (should return false)
    assert(!eshkol_is_number(NULL));
    
    // Test with invalid object (should report an error and return false)
    // This test is optional and depends on how the project handles invalid objects
    // EshkolObject* invalid_obj = malloc(sizeof(EshkolObject));
    // memset(invalid_obj, 0, sizeof(EshkolObject));
    // assert(!eshkol_is_number(invalid_obj));
    // free(invalid_obj);
    
    // Cleanup
    eshkol_free_object(integer_obj);
    eshkol_free_object(float_obj);
    eshkol_free_object(true_obj);
    eshkol_free_object(false_obj);
    eshkol_free_object(string_obj);
    eshkol_free_object(symbol_obj);
    eshkol_free_object(pair_obj);
    eshkol_free_object(empty_list);
    
    printf("number? predicate tests passed!\n");
}

/**
 * @brief Test the string? predicate
 * 
 * This function tests the string? predicate with various inputs:
 * - String values (should return true)
 * - Non-string values (should return false)
 * - NULL (should return false)
 * - Invalid objects (should report an error and return false)
 */
void test_string_predicate() {
    // Setup
    printf("Testing string? predicate...\n");
    
    // Test with string values (should return true)
    EshkolObject* empty_string = eshkol_create_string("");
    EshkolObject* hello_string = eshkol_create_string("hello");
    assert(eshkol_is_string(empty_string));
    assert(eshkol_is_string(hello_string));
    
    // Test with non-string values (should return false)
    EshkolObject* integer_obj = eshkol_create_integer(42);
    EshkolObject* float_obj = eshkol_create_float(3.14);
    EshkolObject* true_obj = eshkol_create_boolean(true);
    EshkolObject* false_obj = eshkol_create_boolean(false);
    EshkolObject* symbol_obj = eshkol_create_symbol("symbol");
    EshkolObject* pair_obj = eshkol_cons(integer_obj, float_obj);
    EshkolObject* empty_list = eshkol_create_empty_list();
    
    assert(!eshkol_is_string(integer_obj));
    assert(!eshkol_is_string(float_obj));
    assert(!eshkol_is_string(true_obj));
    assert(!eshkol_is_string(false_obj));
    assert(!eshkol_is_string(symbol_obj));
    assert(!eshkol_is_string(pair_obj));
    assert(!eshkol_is_string(empty_list));
    
    // Test with NULL (should return false)
    assert(!eshkol_is_string(NULL));
    
    // Test with invalid object (should report an error and return false)
    // This test is optional and depends on how the project handles invalid objects
    // EshkolObject* invalid_obj = malloc(sizeof(EshkolObject));
    // memset(invalid_obj, 0, sizeof(EshkolObject));
    // assert(!eshkol_is_string(invalid_obj));
    // free(invalid_obj);
    
    // Cleanup
    eshkol_free_object(empty_string);
    eshkol_free_object(hello_string);
    eshkol_free_object(integer_obj);
    eshkol_free_object(float_obj);
    eshkol_free_object(true_obj);
    eshkol_free_object(false_obj);
    eshkol_free_object(symbol_obj);
    eshkol_free_object(pair_obj);
    eshkol_free_object(empty_list);
    
    printf("string? predicate tests passed!\n");
}

/**
 * @brief Test the symbol? predicate
 * 
 * This function tests the symbol? predicate with various inputs:
 * - Symbol values (should return true)
 * - Non-symbol values (should return false)
 * - NULL (should return false)
 * - Invalid objects (should report an error and return false)
 */
void test_symbol_predicate() {
    // Setup
    printf("Testing symbol? predicate...\n");
    
    // Test with symbol values (should return true)
    EshkolObject* symbol_obj = eshkol_create_symbol("symbol");
    assert(eshkol_is_symbol(symbol_obj));
    
    // Test with non-symbol values (should return false)
    EshkolObject* integer_obj = eshkol_create_integer(42);
    EshkolObject* float_obj = eshkol_create_float(3.14);
    EshkolObject* true_obj = eshkol_create_boolean(true);
    EshkolObject* false_obj = eshkol_create_boolean(false);
    EshkolObject* string_obj = eshkol_create_string("hello");
    EshkolObject* pair_obj = eshkol_cons(integer_obj, float_obj);
    EshkolObject* empty_list = eshkol_create_empty_list();
    
    assert(!eshkol_is_symbol(integer_obj));
    assert(!eshkol_is_symbol(float_obj));
    assert(!eshkol_is_symbol(true_obj));
    assert(!eshkol_is_symbol(false_obj));
    assert(!eshkol_is_symbol(string_obj));
    assert(!eshkol_is_symbol(pair_obj));
    assert(!eshkol_is_symbol(empty_list));
    
    // Test with NULL (should return false)
    assert(!eshkol_is_symbol(NULL));
    
    // Test with invalid object (should report an error and return false)
    // This test is optional and depends on how the project handles invalid objects
    // EshkolObject* invalid_obj = malloc(sizeof(EshkolObject));
    // memset(invalid_obj, 0, sizeof(EshkolObject));
    // assert(!eshkol_is_symbol(invalid_obj));
    // free(invalid_obj);
    
    // Cleanup
    eshkol_free_object(symbol_obj);
    eshkol_free_object(integer_obj);
    eshkol_free_object(float_obj);
    eshkol_free_object(true_obj);
    eshkol_free_object(false_obj);
    eshkol_free_object(string_obj);
    eshkol_free_object(pair_obj);
    eshkol_free_object(empty_list);
    
    printf("symbol? predicate tests passed!\n");
}

/**
 * @brief Test the procedure? predicate
 * 
 * This function tests the procedure? predicate with various inputs:
 * - Procedure values (should return true)
 * - Non-procedure values (should return false)
 * - NULL (should return false)
 * - Invalid objects (should report an error and return false)
 */
void test_procedure_predicate() {
    // Setup
    printf("Testing procedure? predicate...\n");
    
    // Test with procedure values (should return true)
    // Create a lambda expression
    EshkolObject* params = eshkol_create_empty_list();
    EshkolObject* body = eshkol_create_integer(42);
    EshkolObject* lambda = eshkol_create_lambda(params, body);
    assert(eshkol_is_procedure(lambda));
    
    // Test with non-procedure values (should return false)
    EshkolObject* integer_obj = eshkol_create_integer(42);
    EshkolObject* float_obj = eshkol_create_float(3.14);
    EshkolObject* true_obj = eshkol_create_boolean(true);
    EshkolObject* false_obj = eshkol_create_boolean(false);
    EshkolObject* string_obj = eshkol_create_string("hello");
    EshkolObject* symbol_obj = eshkol_create_symbol("symbol");
    EshkolObject* pair_obj = eshkol_cons(integer_obj, float_obj);
    EshkolObject* empty_list = eshkol_create_empty_list();
    
    assert(!eshkol_is_procedure(integer_obj));
    assert(!eshkol_is_procedure(float_obj));
    assert(!eshkol_is_procedure(true_obj));
    assert(!eshkol_is_procedure(false_obj));
    assert(!eshkol_is_procedure(string_obj));
    assert(!eshkol_is_procedure(symbol_obj));
    assert(!eshkol_is_procedure(pair_obj));
    assert(!eshkol_is_procedure(empty_list));
    
    // Test with NULL (should return false)
    assert(!eshkol_is_procedure(NULL));
    
    // Test with invalid object (should report an error and return false)
    // This test is optional and depends on how the project handles invalid objects
    // EshkolObject* invalid_obj = malloc(sizeof(EshkolObject));
    // memset(invalid_obj, 0, sizeof(EshkolObject));
    // assert(!eshkol_is_procedure(invalid_obj));
    // free(invalid_obj);
    
    // Cleanup
    eshkol_free_object(lambda);
    eshkol_free_object(integer_obj);
    eshkol_free_object(float_obj);
    eshkol_free_object(true_obj);
    eshkol_free_object(false_obj);
    eshkol_free_object(string_obj);
    eshkol_free_object(symbol_obj);
    eshkol_free_object(pair_obj);
    eshkol_free_object(empty_list);
    
    printf("procedure? predicate tests passed!\n");
}

/**
 * @brief Test the vector? predicate
 * 
 * This function tests the vector? predicate with various inputs:
 * - Vector values (should return true)
 * - Non-vector values (should return false)
 * - NULL (should return false)
 * - Invalid objects (should report an error and return false)
 */
void test_vector_predicate() {
    // Setup
    printf("Testing vector? predicate...\n");
    
    // Test with vector values (should return true)
    EshkolObject* empty_vector = eshkol_create_vector(0);
    EshkolObject* vector = eshkol_create_vector(3);
    eshkol_vector_set(vector, 0, eshkol_create_integer(1));
    eshkol_vector_set(vector, 1, eshkol_create_integer(2));
    eshkol_vector_set(vector, 2, eshkol_create_integer(3));
    assert(eshkol_is_vector(empty_vector));
    assert(eshkol_is_vector(vector));
    
    // Test with non-vector values (should return false)
    EshkolObject* integer_obj = eshkol_create_integer(42);
    EshkolObject* float_obj = eshkol_create_float(3.14);
    EshkolObject* true_obj = eshkol_create_boolean(true);
    EshkolObject* false_obj = eshkol_create_boolean(false);
    EshkolObject* string_obj = eshkol_create_string("hello");
    EshkolObject* symbol_obj = eshkol_create_symbol("symbol");
    EshkolObject* pair_obj = eshkol_cons(integer_obj, float_obj);
    EshkolObject* empty_list = eshkol_create_empty_list();
    
    assert(!eshkol_is_vector(integer_obj));
    assert(!eshkol_is_vector(float_obj));
    assert(!eshkol_is_vector(true_obj));
    assert(!eshkol_is_vector(false_obj));
    assert(!eshkol_is_vector(string_obj));
    assert(!eshkol_is_vector(symbol_obj));
    assert(!eshkol_is_vector(pair_obj));
    assert(!eshkol_is_vector(empty_list));
    
    // Test with NULL (should return false)
    assert(!eshkol_is_vector(NULL));
    
    // Test with invalid object (should report an error and return false)
    // This test is optional and depends on how the project handles invalid objects
    // EshkolObject* invalid_obj = malloc(sizeof(EshkolObject));
    // memset(invalid_obj, 0, sizeof(EshkolObject));
    // assert(!eshkol_is_vector(invalid_obj));
    // free(invalid_obj);
    
    // Cleanup
    eshkol_free_object(empty_vector);
    eshkol_free_object(vector);
    eshkol_free_object(integer_obj);
    eshkol_free_object(float_obj);
    eshkol_free_object(true_obj);
    eshkol_free_object(false_obj);
    eshkol_free_object(string_obj);
    eshkol_free_object(symbol_obj);
    eshkol_free_object(pair_obj);
    eshkol_free_object(empty_list);
    
    printf("vector? predicate tests passed!\n");
}

/**
 * @brief Run all type predicate tests
 * 
 * This function runs all the type predicate tests.
 */
int main() {
    printf("Running type predicate tests...\n");
    
    test_boolean_predicate();
    test_number_predicate();
    test_string_predicate();
    test_symbol_predicate();
    test_procedure_predicate();
    test_vector_predicate();
    
    printf("All type predicate tests passed!\n");
    
    return 0;
}
