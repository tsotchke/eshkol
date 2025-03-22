/**
 * @file test_object_pool.c
 * @brief Unit tests for the object pool
 */

#include "core/object_pool.h"
#include "core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Test object pool creation
 */
static void test_object_pool_create(void) {
    printf("Testing object pool creation...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    ObjectPool* pool = object_pool_create(arena, 16, 8);
    assert(pool != NULL);
    assert(object_pool_get_allocated_count(pool) == 0);
    assert(object_pool_get_free_count(pool) == 0);
    assert(object_pool_get_capacity(pool) == 0);
    assert(object_pool_get_object_size(pool) == 16);
    
    arena_destroy(arena);
    
    printf("PASS: object_pool_create\n");
}

/**
 * @brief Test object allocation
 */
static void test_object_pool_alloc(void) {
    printf("Testing object pool allocation...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    ObjectPool* pool = object_pool_create(arena, 16, 8);
    assert(pool != NULL);
    
    // Allocate an object
    void* obj1 = object_pool_alloc(pool);
    assert(obj1 != NULL);
    assert(object_pool_get_allocated_count(pool) == 1);
    assert(object_pool_get_free_count(pool) == 7);
    assert(object_pool_get_capacity(pool) == 8);
    
    // Allocate another object
    void* obj2 = object_pool_alloc(pool);
    assert(obj2 != NULL);
    assert(obj1 != obj2);
    assert(object_pool_get_allocated_count(pool) == 2);
    assert(object_pool_get_free_count(pool) == 6);
    assert(object_pool_get_capacity(pool) == 8);
    
    // Write to the objects
    memset(obj1, 0xAA, 16);
    memset(obj2, 0xBB, 16);
    
    // Verify the objects are distinct
    assert(((unsigned char*)obj1)[0] == 0xAA);
    assert(((unsigned char*)obj2)[0] == 0xBB);
    
    arena_destroy(arena);
    
    printf("PASS: object_pool_alloc\n");
}

/**
 * @brief Test object freeing
 */
static void test_object_pool_free(void) {
    printf("Testing object pool freeing...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    ObjectPool* pool = object_pool_create(arena, 16, 8);
    assert(pool != NULL);
    
    // Allocate an object
    void* obj1 = object_pool_alloc(pool);
    assert(obj1 != NULL);
    assert(object_pool_get_allocated_count(pool) == 1);
    assert(object_pool_get_free_count(pool) == 7);
    
    // Free the object
    object_pool_free(pool, obj1);
    assert(object_pool_get_allocated_count(pool) == 0);
    assert(object_pool_get_free_count(pool) == 8);
    
    // Allocate again, should get the same object
    void* obj2 = object_pool_alloc(pool);
    assert(obj2 != NULL);
    assert(obj1 == obj2);
    assert(object_pool_get_allocated_count(pool) == 1);
    assert(object_pool_get_free_count(pool) == 7);
    
    arena_destroy(arena);
    
    printf("PASS: object_pool_free\n");
}

/**
 * @brief Test object pool reset
 */
static void test_object_pool_reset(void) {
    printf("Testing object pool reset...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    ObjectPool* pool = object_pool_create(arena, 16, 8);
    assert(pool != NULL);
    
    // Allocate some objects
    void* obj1 = object_pool_alloc(pool);
    void* obj2 = object_pool_alloc(pool);
    void* obj3 = object_pool_alloc(pool);
    assert(obj1 != NULL);
    assert(obj2 != NULL);
    assert(obj3 != NULL);
    assert(object_pool_get_allocated_count(pool) == 3);
    assert(object_pool_get_free_count(pool) == 5);
    
    // Reset the pool
    object_pool_reset(pool);
    assert(object_pool_get_allocated_count(pool) == 0);
    assert(object_pool_get_free_count(pool) == 8);
    
    // Allocate again, should get the same objects
    void* obj4 = object_pool_alloc(pool);
    void* obj5 = object_pool_alloc(pool);
    void* obj6 = object_pool_alloc(pool);
    assert(obj4 != NULL);
    assert(obj5 != NULL);
    assert(obj6 != NULL);
    
    // The objects may not be returned in the same order after reset
    // but they should be among the previously allocated objects
    bool obj4_found = (obj4 == obj1 || obj4 == obj2 || obj4 == obj3);
    bool obj5_found = (obj5 == obj1 || obj5 == obj2 || obj5 == obj3);
    bool obj6_found = (obj6 == obj1 || obj6 == obj2 || obj6 == obj3);
    assert(obj4_found);
    assert(obj5_found);
    assert(obj6_found);
    
    arena_destroy(arena);
    
    printf("PASS: object_pool_reset\n");
}

/**
 * @brief Test allocating more objects than the initial capacity
 */
static void test_object_pool_capacity_expansion(void) {
    printf("Testing object pool capacity expansion...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    ObjectPool* pool = object_pool_create(arena, 16, 4);
    assert(pool != NULL);
    
    // Allocate objects up to capacity
    void* obj1 = object_pool_alloc(pool);
    void* obj2 = object_pool_alloc(pool);
    void* obj3 = object_pool_alloc(pool);
    void* obj4 = object_pool_alloc(pool);
    assert(obj1 != NULL);
    assert(obj2 != NULL);
    assert(obj3 != NULL);
    assert(obj4 != NULL);
    assert(object_pool_get_allocated_count(pool) == 4);
    assert(object_pool_get_free_count(pool) == 0);
    assert(object_pool_get_capacity(pool) == 4);
    
    // Allocate one more object, should expand capacity
    void* obj5 = object_pool_alloc(pool);
    assert(obj5 != NULL);
    assert(object_pool_get_allocated_count(pool) == 5);
    assert(object_pool_get_free_count(pool) == 3);
    assert(object_pool_get_capacity(pool) == 8);
    
    arena_destroy(arena);
    
    printf("PASS: object_pool_capacity_expansion\n");
}

/**
 * @brief Test with a custom object type
 */
static void test_object_pool_custom_type(void) {
    printf("Testing object pool with custom type...\n");
    
    // Define a custom object type
    typedef struct {
        int id;
        float value;
        char name[32];
    } CustomObject;
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    ObjectPool* pool = object_pool_create(arena, sizeof(CustomObject), 8);
    assert(pool != NULL);
    assert(object_pool_get_object_size(pool) == sizeof(CustomObject));
    
    // Allocate and initialize objects
    CustomObject* obj1 = object_pool_alloc(pool);
    CustomObject* obj2 = object_pool_alloc(pool);
    assert(obj1 != NULL);
    assert(obj2 != NULL);
    
    obj1->id = 1;
    obj1->value = 3.14f;
    strcpy(obj1->name, "Object 1");
    
    obj2->id = 2;
    obj2->value = 2.71f;
    strcpy(obj2->name, "Object 2");
    
    // Verify object values
    assert(obj1->id == 1);
    assert(obj1->value == 3.14f);
    assert(strcmp(obj1->name, "Object 1") == 0);
    
    assert(obj2->id == 2);
    assert(obj2->value == 2.71f);
    assert(strcmp(obj2->name, "Object 2") == 0);
    
    // Free and reallocate
    object_pool_free(pool, obj1);
    CustomObject* obj3 = object_pool_alloc(pool);
    assert(obj3 == obj1);
    
    // Initialize new object
    obj3->id = 3;
    obj3->value = 1.62f;
    strcpy(obj3->name, "Object 3");
    
    // Verify object values
    assert(obj3->id == 3);
    assert(obj3->value == 1.62f);
    assert(strcmp(obj3->name, "Object 3") == 0);
    
    arena_destroy(arena);
    
    printf("PASS: object_pool_custom_type\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running object pool tests...\n");
    
    test_object_pool_create();
    test_object_pool_alloc();
    test_object_pool_free();
    test_object_pool_reset();
    test_object_pool_capacity_expansion();
    test_object_pool_custom_type();
    
    printf("All object pool tests passed!\n");
    return 0;
}
