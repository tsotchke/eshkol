/**
 * @file test_dynamic_array.c
 * @brief Unit tests for the dynamic array
 */

#include "core/dynamic_array.h"
#include "core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Test dynamic array creation
 */
static void test_dynamic_array_create(void) {
    printf("Testing dynamic array creation...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create an array with initial capacity
    DynamicArray* array = dynamic_array_create(arena, sizeof(int), 10);
    assert(array != NULL);
    assert(dynamic_array_size(array) == 0);
    assert(dynamic_array_capacity(array) == 10);
    assert(dynamic_array_element_size(array) == sizeof(int));
    assert(dynamic_array_is_empty(array));
    
    // Create an array with zero initial capacity
    DynamicArray* array2 = dynamic_array_create(arena, sizeof(double), 0);
    assert(array2 != NULL);
    assert(dynamic_array_size(array2) == 0);
    assert(dynamic_array_capacity(array2) == 0);
    assert(dynamic_array_element_size(array2) == sizeof(double));
    assert(dynamic_array_is_empty(array2));
    
    arena_destroy(arena);
    
    printf("PASS: dynamic_array_create\n");
}

/**
 * @brief Test dynamic array push and pop
 */
static void test_dynamic_array_push_pop(void) {
    printf("Testing dynamic array push and pop...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    DynamicArray* array = dynamic_array_create(arena, sizeof(int), 2);
    assert(array != NULL);
    
    // Push elements
    int values[] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++) {
        bool result = dynamic_array_push(array, &values[i]);
        assert(result);
        assert(dynamic_array_size(array) == (size_t)(i + 1));
    }
    
    // Check capacity (should have grown)
    assert(dynamic_array_capacity(array) >= 5);
    
    // Check elements
    for (int i = 0; i < 5; i++) {
        int* value = dynamic_array_get(array, i);
        assert(value != NULL);
        assert(*value == values[i]);
    }
    
    // Pop elements
    for (int i = 4; i >= 0; i--) {
        int value;
        bool result = dynamic_array_pop(array, &value);
        assert(result);
        assert(value == values[i]);
        assert(dynamic_array_size(array) == (size_t)i);
    }
    
    // Check that the array is empty
    assert(dynamic_array_is_empty(array));
    
    // Try to pop from an empty array
    int value;
    bool result = dynamic_array_pop(array, &value);
    assert(!result);
    
    arena_destroy(arena);
    
    printf("PASS: dynamic_array_push_pop\n");
}

/**
 * @brief Test dynamic array get and set
 */
static void test_dynamic_array_get_set(void) {
    printf("Testing dynamic array get and set...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    DynamicArray* array = dynamic_array_create(arena, sizeof(int), 5);
    assert(array != NULL);
    
    // Push elements
    int values[] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 5; i++) {
        bool result = dynamic_array_push(array, &values[i]);
        assert(result);
    }
    
    // Get elements
    for (int i = 0; i < 5; i++) {
        int* value = dynamic_array_get(array, i);
        assert(value != NULL);
        assert(*value == values[i]);
    }
    
    // Get out of bounds
    void* value = dynamic_array_get(array, 5);
    assert(value == NULL);
    
    // Set elements
    int new_values[] = {100, 200, 300, 400, 500};
    for (int i = 0; i < 5; i++) {
        bool result = dynamic_array_set(array, i, &new_values[i]);
        assert(result);
    }
    
    // Check elements
    for (int i = 0; i < 5; i++) {
        int* value = dynamic_array_get(array, i);
        assert(value != NULL);
        assert(*value == new_values[i]);
    }
    
    // Set out of bounds
    bool result = dynamic_array_set(array, 5, &new_values[0]);
    assert(!result);
    
    arena_destroy(arena);
    
    printf("PASS: dynamic_array_get_set\n");
}

/**
 * @brief Test dynamic array insert and remove
 */
static void test_dynamic_array_insert_remove(void) {
    printf("Testing dynamic array insert and remove...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    DynamicArray* array = dynamic_array_create(arena, sizeof(int), 5);
    assert(array != NULL);
    
    // Insert elements
    int values[] = {10, 20, 30, 40, 50};
    
    // Insert at the beginning
    bool result = dynamic_array_insert(array, 0, &values[0]);
    assert(result);
    assert(dynamic_array_size(array) == 1);
    
    // Insert at the end
    result = dynamic_array_insert(array, 1, &values[4]);
    assert(result);
    assert(dynamic_array_size(array) == 2);
    
    // Insert in the middle
    result = dynamic_array_insert(array, 1, &values[2]);
    assert(result);
    assert(dynamic_array_size(array) == 3);
    
    // Check elements
    int* value = dynamic_array_get(array, 0);
    assert(value != NULL);
    assert(*value == values[0]);
    
    value = dynamic_array_get(array, 1);
    assert(value != NULL);
    assert(*value == values[2]);
    
    value = dynamic_array_get(array, 2);
    assert(value != NULL);
    assert(*value == values[4]);
    
    // Insert out of bounds
    result = dynamic_array_insert(array, 4, &values[1]);
    assert(!result);
    
    // Remove elements
    int removed;
    
    // Remove from the middle
    result = dynamic_array_remove(array, 1, &removed);
    assert(result);
    assert(removed == values[2]);
    assert(dynamic_array_size(array) == 2);
    
    // Check elements
    value = dynamic_array_get(array, 0);
    assert(value != NULL);
    assert(*value == values[0]);
    
    value = dynamic_array_get(array, 1);
    assert(value != NULL);
    assert(*value == values[4]);
    
    // Remove from the beginning
    result = dynamic_array_remove(array, 0, &removed);
    assert(result);
    assert(removed == values[0]);
    assert(dynamic_array_size(array) == 1);
    
    // Remove from the end
    result = dynamic_array_remove(array, 0, &removed);
    assert(result);
    assert(removed == values[4]);
    assert(dynamic_array_size(array) == 0);
    
    // Remove from an empty array
    result = dynamic_array_remove(array, 0, &removed);
    assert(!result);
    
    arena_destroy(arena);
    
    printf("PASS: dynamic_array_insert_remove\n");
}

/**
 * @brief Test dynamic array resize and reserve
 */
static void test_dynamic_array_resize_reserve(void) {
    printf("Testing dynamic array resize and reserve...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    DynamicArray* array = dynamic_array_create(arena, sizeof(int), 2);
    assert(array != NULL);
    
    // Reserve capacity
    bool result = dynamic_array_reserve(array, 10);
    assert(result);
    assert(dynamic_array_capacity(array) >= 10);
    assert(dynamic_array_size(array) == 0);
    
    // Resize to increase size
    result = dynamic_array_resize(array, 5);
    assert(result);
    assert(dynamic_array_size(array) == 5);
    
    // Check that new elements are initialized to zero
    for (size_t i = 0; i < 5; i++) {
        int* value = dynamic_array_get(array, i);
        assert(value != NULL);
        assert(*value == 0);
    }
    
    // Set some values
    for (int i = 0; i < 5; i++) {
        int value = i * 10;
        result = dynamic_array_set(array, i, &value);
        assert(result);
    }
    
    // Resize to decrease size
    result = dynamic_array_resize(array, 3);
    assert(result);
    assert(dynamic_array_size(array) == 3);
    
    // Check remaining elements
    for (int i = 0; i < 3; i++) {
        int* value = dynamic_array_get(array, i);
        assert(value != NULL);
        assert(*value == i * 10);
    }
    
    // Shrink to fit
    result = dynamic_array_shrink_to_fit(array);
    assert(result);
    assert(dynamic_array_capacity(array) == 3);
    
    // Clear the array
    dynamic_array_clear(array);
    assert(dynamic_array_size(array) == 0);
    assert(dynamic_array_is_empty(array));
    
    // Shrink to fit on empty array
    result = dynamic_array_shrink_to_fit(array);
    assert(result);
    assert(dynamic_array_capacity(array) == 0);
    
    arena_destroy(arena);
    
    printf("PASS: dynamic_array_resize_reserve\n");
}

/**
 * @brief Comparison function for integers
 */
static int compare_ints(const void* a, const void* b) {
    const int* ia = (const int*)a;
    const int* ib = (const int*)b;
    return *ia - *ib;
}

/**
 * @brief Function to apply to each element
 */
static void increment_int(void* element, void* user_data) {
    int* value = (int*)element;
    int increment = *(int*)user_data;
    *value += increment;
}

/**
 * @brief Test dynamic array find, sort, and foreach
 */
static void test_dynamic_array_find_sort_foreach(void) {
    printf("Testing dynamic array find, sort, and foreach...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    DynamicArray* array = dynamic_array_create(arena, sizeof(int), 5);
    assert(array != NULL);
    
    // Push elements in unsorted order
    int values[] = {30, 10, 50, 20, 40};
    for (int i = 0; i < 5; i++) {
        bool result = dynamic_array_push(array, &values[i]);
        assert(result);
    }
    
    // Find an element
    int target = 30;
    size_t index = dynamic_array_find(array, &target, compare_ints);
    assert(index == 0);
    
    // Find a non-existent element
    target = 60;
    index = dynamic_array_find(array, &target, compare_ints);
    assert(index == (size_t)-1);
    
    // Sort the array
    dynamic_array_sort(array, compare_ints);
    
    // Check that the array is sorted
    for (int i = 0; i < 4; i++) {
        int* a = dynamic_array_get(array, i);
        int* b = dynamic_array_get(array, i + 1);
        assert(a != NULL);
        assert(b != NULL);
        assert(*a <= *b);
    }
    
    // Apply a function to each element
    int increment = 5;
    dynamic_array_foreach(array, increment_int, &increment);
    
    // Check that each element was incremented
    for (int i = 0; i < 5; i++) {
        int* value = dynamic_array_get(array, i);
        assert(value != NULL);
        assert(*value == (i + 1) * 10 + increment);
    }
    
    arena_destroy(arena);
    
    printf("PASS: dynamic_array_find_sort_foreach\n");
}

/**
 * @brief Test dynamic array with custom struct
 */
static void test_dynamic_array_custom_struct(void) {
    printf("Testing dynamic array with custom struct...\n");
    
    // Define a custom struct
    typedef struct {
        int id;
        char name[32];
        double value;
    } TestStruct;
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    DynamicArray* array = dynamic_array_create(arena, sizeof(TestStruct), 3);
    assert(array != NULL);
    
    // Create test structs
    TestStruct structs[3] = {
        {1, "First", 1.1},
        {2, "Second", 2.2},
        {3, "Third", 3.3}
    };
    
    // Push structs
    for (int i = 0; i < 3; i++) {
        bool result = dynamic_array_push(array, &structs[i]);
        assert(result);
    }
    
    // Check structs
    for (int i = 0; i < 3; i++) {
        TestStruct* s = dynamic_array_get(array, i);
        assert(s != NULL);
        assert(s->id == structs[i].id);
        assert(strcmp(s->name, structs[i].name) == 0);
        assert(s->value == structs[i].value);
    }
    
    // Modify a struct
    TestStruct modified = {4, "Modified", 4.4};
    bool result = dynamic_array_set(array, 1, &modified);
    assert(result);
    
    // Check the modified struct
    TestStruct* s = dynamic_array_get(array, 1);
    assert(s != NULL);
    assert(s->id == modified.id);
    assert(strcmp(s->name, modified.name) == 0);
    assert(s->value == modified.value);
    
    arena_destroy(arena);
    
    printf("PASS: dynamic_array_custom_struct\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running dynamic array tests...\n");
    
    test_dynamic_array_create();
    test_dynamic_array_push_pop();
    test_dynamic_array_get_set();
    test_dynamic_array_insert_remove();
    test_dynamic_array_resize_reserve();
    test_dynamic_array_find_sort_foreach();
    test_dynamic_array_custom_struct();
    
    printf("All dynamic array tests passed!\n");
    return 0;
}
