/**
 * @file dynamic_array.c
 * @brief Implementation of the dynamic array data structure
 */

#include "core/dynamic_array.h"
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Dynamic array structure
 */
struct DynamicArray {
    Arena* arena;           /**< Arena for allocations */
    void* data;             /**< Array data */
    size_t size;            /**< Number of elements */
    size_t capacity;        /**< Capacity (number of elements) */
    size_t element_size;    /**< Size of each element in bytes */
};

/**
 * @brief Calculate the new capacity when resizing
 * 
 * @param current_capacity Current capacity
 * @param min_capacity Minimum required capacity
 * @return New capacity
 */
static size_t calculate_new_capacity(size_t current_capacity, size_t min_capacity) {
    // Double the capacity, or use min_capacity if it's larger
    size_t new_capacity = current_capacity == 0 ? 1 : current_capacity * 2;
    return new_capacity > min_capacity ? new_capacity : min_capacity;
}

DynamicArray* dynamic_array_create(Arena* arena, size_t element_size, size_t initial_capacity) {
    assert(arena != NULL);
    assert(element_size > 0);
    
    // Allocate the array structure
    DynamicArray* array = arena_alloc(arena, sizeof(DynamicArray));
    if (!array) return NULL;
    
    // Initialize the array
    array->arena = arena;
    array->size = 0;
    array->capacity = initial_capacity;
    array->element_size = element_size;
    
    // Allocate the data if initial capacity is non-zero
    if (initial_capacity > 0) {
        array->data = arena_alloc(arena, element_size * initial_capacity);
        if (!array->data) return NULL;
    } else {
        array->data = NULL;
    }
    
    return array;
}

size_t dynamic_array_size(const DynamicArray* array) {
    assert(array != NULL);
    return array->size;
}

size_t dynamic_array_capacity(const DynamicArray* array) {
    assert(array != NULL);
    return array->capacity;
}

size_t dynamic_array_element_size(const DynamicArray* array) {
    assert(array != NULL);
    return array->element_size;
}

void* dynamic_array_get(const DynamicArray* array, size_t index) {
    assert(array != NULL);
    
    // Check if index is out of bounds
    if (index >= array->size) {
        return NULL;
    }
    
    // Calculate the address of the element
    return (char*)array->data + index * array->element_size;
}

bool dynamic_array_set(DynamicArray* array, size_t index, const void* element) {
    assert(array != NULL);
    assert(element != NULL);
    
    // Check if index is out of bounds
    if (index >= array->size) {
        return false;
    }
    
    // Calculate the address of the element
    void* dest = (char*)array->data + index * array->element_size;
    
    // Copy the element
    memcpy(dest, element, array->element_size);
    
    return true;
}

bool dynamic_array_push(DynamicArray* array, const void* element) {
    assert(array != NULL);
    assert(element != NULL);
    
    // Check if we need to resize
    if (array->size >= array->capacity) {
        size_t new_capacity = calculate_new_capacity(array->capacity, array->size + 1);
        if (!dynamic_array_reserve(array, new_capacity)) {
            return false;
        }
    }
    
    // Calculate the address of the new element
    void* dest = (char*)array->data + array->size * array->element_size;
    
    // Copy the element
    memcpy(dest, element, array->element_size);
    
    // Increment the size
    array->size++;
    
    return true;
}

bool dynamic_array_pop(DynamicArray* array, void* element) {
    assert(array != NULL);
    
    // Check if the array is empty
    if (array->size == 0) {
        return false;
    }
    
    // Decrement the size
    array->size--;
    
    // If element is not NULL, copy the popped element
    if (element) {
        void* src = (char*)array->data + array->size * array->element_size;
        memcpy(element, src, array->element_size);
    }
    
    return true;
}

bool dynamic_array_insert(DynamicArray* array, size_t index, const void* element) {
    assert(array != NULL);
    assert(element != NULL);
    
    // Check if index is out of bounds (allow insertion at the end)
    if (index > array->size) {
        return false;
    }
    
    // Check if we need to resize
    if (array->size >= array->capacity) {
        size_t new_capacity = calculate_new_capacity(array->capacity, array->size + 1);
        if (!dynamic_array_reserve(array, new_capacity)) {
            return false;
        }
    }
    
    // If not inserting at the end, shift elements
    if (index < array->size) {
        void* src = (char*)array->data + index * array->element_size;
        void* dest = (char*)src + array->element_size;
        size_t bytes_to_move = (array->size - index) * array->element_size;
        memmove(dest, src, bytes_to_move);
    }
    
    // Copy the element
    void* dest = (char*)array->data + index * array->element_size;
    memcpy(dest, element, array->element_size);
    
    // Increment the size
    array->size++;
    
    return true;
}

bool dynamic_array_remove(DynamicArray* array, size_t index, void* element) {
    assert(array != NULL);
    
    // Check if index is out of bounds
    if (index >= array->size) {
        return false;
    }
    
    // Calculate the address of the element
    void* src = (char*)array->data + index * array->element_size;
    
    // If element is not NULL, copy the removed element
    if (element) {
        memcpy(element, src, array->element_size);
    }
    
    // Shift elements
    if (index < array->size - 1) {
        void* dest = src;
        void* src_next = (char*)src + array->element_size;
        size_t bytes_to_move = (array->size - index - 1) * array->element_size;
        memmove(dest, src_next, bytes_to_move);
    }
    
    // Decrement the size
    array->size--;
    
    return true;
}

void dynamic_array_clear(DynamicArray* array) {
    assert(array != NULL);
    array->size = 0;
}

bool dynamic_array_reserve(DynamicArray* array, size_t capacity) {
    assert(array != NULL);
    
    // If the requested capacity is less than or equal to the current capacity, do nothing
    if (capacity <= array->capacity) {
        return true;
    }
    
    // Allocate new data
    void* new_data = arena_alloc(array->arena, array->element_size * capacity);
    if (!new_data) {
        return false;
    }
    
    // Copy existing data
    if (array->data) {
        memcpy(new_data, array->data, array->size * array->element_size);
    }
    
    // Update the array
    array->data = new_data;
    array->capacity = capacity;
    
    return true;
}

bool dynamic_array_resize(DynamicArray* array, size_t size) {
    assert(array != NULL);
    
    // If the requested size is greater than the current capacity, reserve more space
    if (size > array->capacity) {
        if (!dynamic_array_reserve(array, size)) {
            return false;
        }
    }
    
    // If the requested size is greater than the current size, initialize new elements to zero
    if (size > array->size) {
        void* dest = (char*)array->data + array->size * array->element_size;
        size_t bytes_to_zero = (size - array->size) * array->element_size;
        memset(dest, 0, bytes_to_zero);
    }
    
    // Update the size
    array->size = size;
    
    return true;
}

bool dynamic_array_shrink_to_fit(DynamicArray* array) {
    assert(array != NULL);
    
    // If the array is empty, free the data
    if (array->size == 0) {
        array->data = NULL;
        array->capacity = 0;
        return true;
    }
    
    // If the size is less than the capacity, resize the data
    if (array->size < array->capacity) {
        void* new_data = arena_alloc(array->arena, array->element_size * array->size);
        if (!new_data) {
            return false;
        }
        
        // Copy existing data
        memcpy(new_data, array->data, array->size * array->element_size);
        
        // Update the array
        array->data = new_data;
        array->capacity = array->size;
    }
    
    return true;
}

void* dynamic_array_data(const DynamicArray* array) {
    assert(array != NULL);
    return array->data;
}

bool dynamic_array_is_empty(const DynamicArray* array) {
    assert(array != NULL);
    return array->size == 0;
}

size_t dynamic_array_find(const DynamicArray* array, const void* element,
                         int (*compare)(const void*, const void*)) {
    assert(array != NULL);
    assert(element != NULL);
    assert(compare != NULL);
    
    // Linear search
    for (size_t i = 0; i < array->size; i++) {
        void* current = (char*)array->data + i * array->element_size;
        if (compare(current, element) == 0) {
            return i;
        }
    }
    
    // Not found
    return (size_t)-1;
}

void dynamic_array_sort(DynamicArray* array, int (*compare)(const void*, const void*)) {
    assert(array != NULL);
    assert(compare != NULL);
    
    // Use qsort
    qsort(array->data, array->size, array->element_size, compare);
}

void dynamic_array_foreach(const DynamicArray* array,
                          void (*func)(void* element, void* user_data),
                          void* user_data) {
    assert(array != NULL);
    assert(func != NULL);
    
    // Apply the function to each element
    for (size_t i = 0; i < array->size; i++) {
        void* element = (char*)array->data + i * array->element_size;
        func(element, user_data);
    }
}
