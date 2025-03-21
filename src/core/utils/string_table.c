/**
 * @file string_table.c
 * @brief Implementation of the string table for string interning
 */

#include "core/string_table.h"
#include "core/memory.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Entry in the string table
 */
typedef struct Entry {
    const char* string;    /**< The interned string */
    size_t length;         /**< Length of the string */
    uint32_t hash;         /**< Hash of the string */
    struct Entry* next;    /**< Next entry in the hash bucket */
} Entry;

/**
 * @brief String table structure
 */
struct StringTable {
    Arena* arena;           /**< Arena for allocations */
    Entry** buckets;        /**< Hash buckets */
    size_t bucket_count;    /**< Number of buckets */
    size_t count;           /**< Number of strings in the table */
    size_t resize_threshold; /**< Threshold for resizing */
};

/**
 * @brief FNV-1a hash function
 * 
 * @param string String to hash
 * @param length Length of the string
 * @return Hash value
 */
static uint32_t hash_string(const char* string, size_t length) {
    uint32_t hash = 2166136261u; // FNV offset basis
    for (size_t i = 0; i < length; i++) {
        hash ^= (uint8_t)string[i];
        hash *= 16777619u; // FNV prime
    }
    return hash;
}

StringTable* string_table_create(Arena* arena, size_t initial_capacity) {
    assert(arena != NULL);
    
    // Ensure power of 2 capacity
    size_t capacity = 16; // Minimum capacity
    while (capacity < initial_capacity) {
        capacity *= 2;
    }
    
    // Allocate string table
    StringTable* table = arena_alloc(arena, sizeof(StringTable));
    if (!table) return NULL;
    
    // Allocate buckets
    size_t buckets_size = capacity * sizeof(Entry*);
    Entry** buckets = arena_alloc(arena, buckets_size);
    if (!buckets) return NULL;
    
    // Initialize buckets
    memset(buckets, 0, buckets_size);
    
    // Initialize table
    table->arena = arena;
    table->buckets = buckets;
    table->bucket_count = capacity;
    table->count = 0;
    table->resize_threshold = capacity * 3 / 4; // 75% load factor
    
    return table;
}

/**
 * @brief Resize the string table
 * 
 * @param table The string table
 */
static void string_table_resize(StringTable* table) {
    assert(table != NULL);
    
    // Double the capacity
    size_t new_capacity = table->bucket_count * 2;
    size_t new_buckets_size = new_capacity * sizeof(Entry*);
    
    // Allocate new buckets
    Entry** new_buckets = arena_alloc(table->arena, new_buckets_size);
    if (!new_buckets) return; // Failed to resize
    
    // Initialize new buckets
    memset(new_buckets, 0, new_buckets_size);
    
    // Rehash all entries
    for (size_t i = 0; i < table->bucket_count; i++) {
        Entry* entry = table->buckets[i];
        while (entry) {
            Entry* next = entry->next;
            
            // Compute new bucket index
            size_t bucket = entry->hash & (new_capacity - 1);
            
            // Add to new bucket
            entry->next = new_buckets[bucket];
            new_buckets[bucket] = entry;
            
            entry = next;
        }
    }
    
    // Update table
    table->buckets = new_buckets;
    table->bucket_count = new_capacity;
    table->resize_threshold = new_capacity * 3 / 4;
}

const char* string_table_intern_n(StringTable* table, const char* string, size_t length) {
    assert(table != NULL);
    assert(string != NULL);
    
    // Compute hash
    uint32_t hash = hash_string(string, length);
    
    // Compute bucket index
    size_t bucket = hash & (table->bucket_count - 1);
    
    // Look for existing entry
    Entry* entry = table->buckets[bucket];
    while (entry) {
        if (entry->hash == hash && entry->length == length &&
            memcmp(entry->string, string, length) == 0) {
            return entry->string;
        }
        entry = entry->next;
    }
    
    // String not found, create new entry
    
    // Check if resize needed
    if (table->count >= table->resize_threshold) {
        string_table_resize(table);
        // Recompute bucket after resize
        bucket = hash & (table->bucket_count - 1);
    }
    
    // Allocate string
    char* new_string = arena_alloc(table->arena, length + 1);
    if (!new_string) return NULL;
    
    // Copy string
    memcpy(new_string, string, length);
    new_string[length] = '\0';
    
    // Allocate entry
    entry = arena_alloc(table->arena, sizeof(Entry));
    if (!entry) return NULL;
    
    // Initialize entry
    entry->string = new_string;
    entry->length = length;
    entry->hash = hash;
    entry->next = table->buckets[bucket];
    
    // Add to bucket
    table->buckets[bucket] = entry;
    table->count++;
    
    return new_string;
}

const char* string_table_intern(StringTable* table, const char* string) {
    assert(table != NULL);
    assert(string != NULL);
    
    return string_table_intern_n(table, string, strlen(string));
}

bool string_table_contains(StringTable* table, const char* string) {
    assert(table != NULL);
    assert(string != NULL);
    
    // Compute hash
    size_t length = strlen(string);
    uint32_t hash = hash_string(string, length);
    
    // Compute bucket index
    size_t bucket = hash & (table->bucket_count - 1);
    
    // Look for existing entry
    Entry* entry = table->buckets[bucket];
    while (entry) {
        if (entry->hash == hash && entry->length == length &&
            memcmp(entry->string, string, length) == 0) {
            return true;
        }
        entry = entry->next;
    }
    
    return false;
}

size_t string_table_get_count(StringTable* table) {
    assert(table != NULL);
    return table->count;
}
