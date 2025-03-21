/**
 * @file test_string_table.c
 * @brief Unit tests for the string table
 */

#include "core/string_table.h"
#include "core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Test string table creation
 */
static void test_string_table_create(void) {
    printf("Testing string table creation...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    StringTable* table = string_table_create(arena, 16);
    assert(table != NULL);
    assert(string_table_get_count(table) == 0);
    
    arena_destroy(arena);
    
    printf("PASS: string_table_create\n");
}

/**
 * @brief Test string interning
 */
static void test_string_table_intern(void) {
    printf("Testing string interning...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    StringTable* table = string_table_create(arena, 16);
    assert(table != NULL);
    
    // Intern some strings
    const char* s1 = string_table_intern(table, "hello");
    const char* s2 = string_table_intern(table, "world");
    const char* s3 = string_table_intern(table, "hello");
    
    // Check that the strings were interned
    assert(s1 != NULL);
    assert(s2 != NULL);
    assert(s3 != NULL);
    
    // Check that identical strings have the same pointer
    assert(s1 == s3);
    
    // Check that different strings have different pointers
    assert(s1 != s2);
    
    // Check the string contents
    assert(strcmp(s1, "hello") == 0);
    assert(strcmp(s2, "world") == 0);
    
    // Check the string count
    assert(string_table_get_count(table) == 2);
    
    arena_destroy(arena);
    
    printf("PASS: string_table_intern\n");
}

/**
 * @brief Test string interning with explicit length
 */
static void test_string_table_intern_n(void) {
    printf("Testing string interning with explicit length...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    StringTable* table = string_table_create(arena, 16);
    assert(table != NULL);
    
    // Create a string with embedded nulls
    const char str[] = "hello\0world";
    
    // Intern the first part
    const char* s1 = string_table_intern_n(table, str, 5);
    
    // Intern the whole thing
    const char* s2 = string_table_intern_n(table, str, sizeof(str) - 1);
    
    // Check that the strings were interned
    assert(s1 != NULL);
    assert(s2 != NULL);
    
    // Check that they are different strings
    assert(s1 != s2);
    
    // Check the string contents
    assert(strcmp(s1, "hello") == 0);
    assert(memcmp(s2, str, sizeof(str) - 1) == 0);
    
    // Check the string count
    assert(string_table_get_count(table) == 2);
    
    arena_destroy(arena);
    
    printf("PASS: string_table_intern_n\n");
}

/**
 * @brief Test string table contains
 */
static void test_string_table_contains(void) {
    printf("Testing string table contains...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    StringTable* table = string_table_create(arena, 16);
    assert(table != NULL);
    
    // Intern some strings
    string_table_intern(table, "hello");
    string_table_intern(table, "world");
    
    // Check if strings are in the table
    assert(string_table_contains(table, "hello"));
    assert(string_table_contains(table, "world"));
    assert(!string_table_contains(table, "foo"));
    
    arena_destroy(arena);
    
    printf("PASS: string_table_contains\n");
}

/**
 * @brief Test string table with many strings
 */
static void test_string_table_many_strings(void) {
    printf("Testing string table with many strings...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    StringTable* table = string_table_create(arena, 16);
    assert(table != NULL);
    
    // Create a buffer for string generation
    char buffer[32];
    
    // Intern many strings
    const int count = 1000;
    for (int i = 0; i < count; i++) {
        sprintf(buffer, "string%d", i);
        const char* s = string_table_intern(table, buffer);
        assert(s != NULL);
        assert(strcmp(s, buffer) == 0);
    }
    
    // Check the string count
    assert(string_table_get_count(table) == count);
    
    // Check that all strings are in the table
    for (int i = 0; i < count; i++) {
        sprintf(buffer, "string%d", i);
        assert(string_table_contains(table, buffer));
    }
    
    arena_destroy(arena);
    
    printf("PASS: string_table_many_strings\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running string table tests...\n");
    
    test_string_table_create();
    test_string_table_intern();
    test_string_table_intern_n();
    test_string_table_contains();
    test_string_table_many_strings();
    
    printf("All string table tests passed!\n");
    return 0;
}
