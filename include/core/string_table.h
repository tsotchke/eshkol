/**
 * @file string_table.h
 * @brief String interning system for Eshkol
 * 
 * This file defines the string table interface for Eshkol,
 * which provides string interning for efficient string storage and comparison.
 */

#ifndef ESHKOL_STRING_TABLE_H
#define ESHKOL_STRING_TABLE_H

#include "core/memory.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief String table for interning strings
 * 
 * The string table provides efficient string storage and comparison
 * by ensuring that each unique string is stored only once.
 */
typedef struct StringTable StringTable;

/**
 * @brief Create a new string table
 * 
 * @param arena Arena to allocate from
 * @param initial_capacity Initial capacity (number of strings)
 * @return A new string table, or NULL on failure
 */
StringTable* string_table_create(Arena* arena, size_t initial_capacity);

/**
 * @brief Intern a string
 * 
 * If the string is already in the table, returns a pointer to the
 * existing string. Otherwise, adds the string to the table and
 * returns a pointer to the new string.
 * 
 * @param table The string table
 * @param string The string to intern
 * @return Pointer to the interned string, or NULL on failure
 */
const char* string_table_intern(StringTable* table, const char* string);

/**
 * @brief Intern a string with explicit length
 * 
 * Like string_table_intern, but takes an explicit length parameter.
 * This is useful for interning substrings or strings that are not
 * null-terminated.
 * 
 * @param table The string table
 * @param string The string to intern
 * @param length Length of the string
 * @return Pointer to the interned string, or NULL on failure
 */
const char* string_table_intern_n(StringTable* table, const char* string, size_t length);

/**
 * @brief Check if a string is interned
 * 
 * @param table The string table
 * @param string The string to check
 * @return true if the string is interned, false otherwise
 */
bool string_table_contains(StringTable* table, const char* string);

/**
 * @brief Get the number of interned strings
 * 
 * @param table The string table
 * @return Number of interned strings
 */
size_t string_table_get_count(StringTable* table);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_STRING_TABLE_H */
