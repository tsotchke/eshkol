/**
 * @file test_arena.c
 * @brief Unit tests for the arena allocator
 */

#include "eshkol/core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Test arena creation and destruction
 */
static void test_arena_create_destroy(void) {
    printf("Testing arena creation and destruction...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    arena_destroy(arena);
    
    printf("PASS: arena_create_destroy\n");
}

/**
 * @brief Test basic memory allocation
 */
static void test_arena_alloc(void) {
    printf("Testing arena allocation...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Allocate some memory
    void* ptr1 = arena_alloc(arena, 100);
    assert(ptr1 != NULL);
    
    // Allocate more memory
    void* ptr2 = arena_alloc(arena, 200);
    assert(ptr2 != NULL);
    
    // Pointers should be different
    assert(ptr1 != ptr2);
    
    // Check memory usage
    assert(arena_get_used_memory(arena) >= 300);
    assert(arena_get_allocation_count(arena) == 2);
    
    // Write to the allocated memory
    memset(ptr1, 0xAA, 100);
    memset(ptr2, 0xBB, 200);
    
    // Verify the memory is distinct
    assert(((unsigned char*)ptr1)[0] == 0xAA);
    assert(((unsigned char*)ptr2)[0] == 0xBB);
    
    arena_destroy(arena);
    
    printf("PASS: arena_alloc\n");
}

/**
 * @brief Test aligned memory allocation
 */
static void test_arena_alloc_aligned(void) {
    printf("Testing aligned arena allocation...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Allocate aligned memory
    void* ptr = arena_alloc_aligned(arena, 100, 16);
    assert(ptr != NULL);
    
    // Check alignment
    assert(((uintptr_t)ptr & 0xF) == 0);
    
    arena_destroy(arena);
    
    printf("PASS: arena_alloc_aligned\n");
}

/**
 * @brief Test arena reset
 */
static void test_arena_reset(void) {
    printf("Testing arena reset...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Allocate some memory
    void* ptr1 = arena_alloc(arena, 100);
    assert(ptr1 != NULL);
    
    // Check memory usage
    assert(arena_get_used_memory(arena) >= 100);
    assert(arena_get_allocation_count(arena) == 1);
    
    // Reset the arena
    arena_reset(arena);
    
    // Memory usage should be reset
    assert(arena_get_used_memory(arena) == 0);
    assert(arena_get_allocation_count(arena) == 0);
    
    // Allocate more memory
    void* ptr2 = arena_alloc(arena, 100);
    assert(ptr2 != NULL);
    
    // After reset, we should get the same memory back
    assert(ptr1 == ptr2);
    
    arena_destroy(arena);
    
    printf("PASS: arena_reset\n");
}

/**
 * @brief Test allocating more memory than the initial capacity
 */
static void test_arena_capacity_expansion(void) {
    printf("Testing arena capacity expansion...\n");
    
    // Create a small arena
    Arena* arena = arena_create(128);
    assert(arena != NULL);
    
    // Allocate more memory than the initial capacity
    void* ptr1 = arena_alloc(arena, 100);
    assert(ptr1 != NULL);
    
    void* ptr2 = arena_alloc(arena, 100);
    assert(ptr2 != NULL);
    
    void* ptr3 = arena_alloc(arena, 100);
    assert(ptr3 != NULL);
    
    // Check memory usage
    assert(arena_get_used_memory(arena) >= 300);
    assert(arena_get_allocation_count(arena) == 3);
    
    arena_destroy(arena);
    
    printf("PASS: arena_capacity_expansion\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running arena allocator tests...\n");
    
    test_arena_create_destroy();
    test_arena_alloc();
    test_arena_alloc_aligned();
    test_arena_reset();
    test_arena_capacity_expansion();
    
    printf("All arena tests passed!\n");
    return 0;
}
