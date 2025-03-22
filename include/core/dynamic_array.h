/**
 * @file dynamic_array.h
 * @brief Dynamic array implementation for Eshkol
 * 
 * This file defines a generic dynamic array data structure for Eshkol,
 * which provides automatic resizing and efficient element access.
 */

#ifndef ESHKOL_DYNAMIC_ARRAY_H
#define ESHKOL_DYNAMIC_ARRAY_H

#include "core/memory.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Dynamic array data structure
 * 
 * The dynamic array provides automatic resizing and efficient element access.
 * It is a generic container that can store elements of any type.
 */
typedef struct DynamicArray DynamicArray;

/**
 * @brief Create a new dynamic array
 * 
 * @param arena Arena to allocate from
 * @param element_size Size of each element in bytes
 * @param initial_capacity Initial capacity (number of elements)
 * @return A new dynamic array, or NULL on failure
 */
DynamicArray* dynamic_array_create(Arena* arena, size_t element_size, size_t initial_capacity);

/**
 * @brief Get the number of elements in the array
 * 
 * @param array The dynamic array
 * @return Number of elements
 */
size_t dynamic_array_size(const DynamicArray* array);

/**
 * @brief Get the capacity of the array
 * 
 * @param array The dynamic array
 * @return Capacity (number of elements)
 */
size_t dynamic_array_capacity(const DynamicArray* array);

/**
 * @brief Get the size of each element
 * 
 * @param array The dynamic array
 * @return Size of each element in bytes
 */
size_t dynamic_array_element_size(const DynamicArray* array);

/**
 * @brief Get a pointer to the element at the specified index
 * 
 * @param array The dynamic array
 * @param index Index of the element
 * @return Pointer to the element, or NULL if index is out of bounds
 */
void* dynamic_array_get(const DynamicArray* array, size_t index);

/**
 * @brief Set the element at the specified index
 * 
 * @param array The dynamic array
 * @param index Index of the element
 * @param element Pointer to the element to copy
 * @return true if successful, false otherwise
 */
bool dynamic_array_set(DynamicArray* array, size_t index, const void* element);

/**
 * @brief Append an element to the end of the array
 * 
 * @param array The dynamic array
 * @param element Pointer to the element to copy
 * @return true if successful, false otherwise
 */
bool dynamic_array_push(DynamicArray* array, const void* element);

/**
 * @brief Remove the last element from the array
 * 
 * @param array The dynamic array
 * @param element Pointer to store the removed element (can be NULL)
 * @return true if successful, false if the array is empty
 */
bool dynamic_array_pop(DynamicArray* array, void* element);

/**
 * @brief Insert an element at the specified index
 * 
 * @param array The dynamic array
 * @param index Index to insert at
 * @param element Pointer to the element to copy
 * @return true if successful, false otherwise
 */
bool dynamic_array_insert(DynamicArray* array, size_t index, const void* element);

/**
 * @brief Remove the element at the specified index
 * 
 * @param array The dynamic array
 * @param index Index of the element to remove
 * @param element Pointer to store the removed element (can be NULL)
 * @return true if successful, false if the index is out of bounds
 */
bool dynamic_array_remove(DynamicArray* array, size_t index, void* element);

/**
 * @brief Clear the array, removing all elements
 * 
 * @param array The dynamic array
 */
void dynamic_array_clear(DynamicArray* array);

/**
 * @brief Reserve capacity for the array
 * 
 * @param array The dynamic array
 * @param capacity New capacity (number of elements)
 * @return true if successful, false otherwise
 */
bool dynamic_array_reserve(DynamicArray* array, size_t capacity);

/**
 * @brief Resize the array
 * 
 * If the new size is larger than the current size, new elements are
 * initialized to zero. If the new size is smaller than the current size,
 * elements at the end are truncated.
 * 
 * @param array The dynamic array
 * @param size New size (number of elements)
 * @return true if successful, false otherwise
 */
bool dynamic_array_resize(DynamicArray* array, size_t size);

/**
 * @brief Shrink the array capacity to fit the current size
 * 
 * @param array The dynamic array
 * @return true if successful, false otherwise
 */
bool dynamic_array_shrink_to_fit(DynamicArray* array);

/**
 * @brief Get a pointer to the underlying data
 * 
 * @param array The dynamic array
 * @return Pointer to the data
 */
void* dynamic_array_data(const DynamicArray* array);

/**
 * @brief Check if the array is empty
 * 
 * @param array The dynamic array
 * @return true if the array is empty, false otherwise
 */
bool dynamic_array_is_empty(const DynamicArray* array);

/**
 * @brief Find an element in the array
 * 
 * @param array The dynamic array
 * @param element Pointer to the element to find
 * @param compare Comparison function (returns 0 if elements are equal)
 * @return Index of the element, or (size_t)-1 if not found
 */
size_t dynamic_array_find(const DynamicArray* array, const void* element,
                         int (*compare)(const void*, const void*));

/**
 * @brief Sort the array
 * 
 * @param array The dynamic array
 * @param compare Comparison function (as for qsort)
 */
void dynamic_array_sort(DynamicArray* array, int (*compare)(const void*, const void*));

/**
 * @brief Apply a function to each element in the array
 * 
 * @param array The dynamic array
 * @param func Function to apply
 * @param user_data User data to pass to the function
 */
void dynamic_array_foreach(const DynamicArray* array,
                          void (*func)(void* element, void* user_data),
                          void* user_data);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_DYNAMIC_ARRAY_H */
