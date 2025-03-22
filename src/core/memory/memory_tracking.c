/**
 * @file memory_tracking.c
 * @brief Implementation of memory tracking utilities
 */

#include "core/memory_tracking.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

/**
 * @brief Memory allocation record
 */
typedef struct AllocationRecord {
    void* ptr;                  /**< Pointer to the allocated memory */
    size_t size;                /**< Size of the allocation in bytes */
    const char* file;           /**< Source file where the allocation occurred */
    int line;                   /**< Line number where the allocation occurred */
    const char* func;           /**< Function where the allocation occurred */
    struct AllocationRecord* next; /**< Next record in the hash table */
} AllocationRecord;

/**
 * @brief Memory tracking context
 */
typedef struct {
    AllocationRecord** records;  /**< Hash table of allocation records */
    size_t table_size;           /**< Size of the hash table */
    size_t total_allocations;    /**< Total number of allocations */
    size_t active_allocations;   /**< Number of active allocations */
    size_t total_bytes_allocated; /**< Total bytes allocated */
    size_t active_bytes_allocated; /**< Active bytes allocated */
    size_t peak_bytes_allocated;  /**< Peak bytes allocated */
    bool enabled;                /**< Whether memory tracking is enabled */
    pthread_mutex_t mutex;       /**< Mutex for thread safety */
} MemoryTrackingContext;

/**
 * @brief Global memory tracking context
 */
static MemoryTrackingContext g_context;

/**
 * @brief Hash function for pointers
 * 
 * @param ptr Pointer to hash
 * @param table_size Size of the hash table
 * @return Hash value
 */
static size_t hash_pointer(void* ptr, size_t table_size) {
    return ((uintptr_t)ptr >> 3) % table_size;
}

void memory_tracking_init(void) {
    // Initialize the context
    memset(&g_context, 0, sizeof(g_context));
    
    // Set default table size
    g_context.table_size = 1024;
    
    // Allocate hash table
    g_context.records = calloc(g_context.table_size, sizeof(AllocationRecord*));
    if (!g_context.records) {
        fprintf(stderr, "Failed to allocate memory for tracking records\n");
        return;
    }
    
    // Initialize mutex
    pthread_mutex_init(&g_context.mutex, NULL);
    
    // Enable memory tracking by default
    g_context.enabled = true;
}

bool memory_tracking_shutdown(void) {
    bool no_leaks = true;
    
    // Print memory leak report
    size_t leak_count = memory_tracking_print_leaks();
    if (leak_count > 0) {
        no_leaks = false;
    }
    
    // Print final statistics
    memory_tracking_print_stats();
    
    // Free all allocation records
    if (g_context.records) {
        for (size_t i = 0; i < g_context.table_size; i++) {
            AllocationRecord* record = g_context.records[i];
            while (record) {
                AllocationRecord* next = record->next;
                free(record);
                record = next;
            }
        }
        
        // Free the hash table
        free(g_context.records);
        g_context.records = NULL;
    }
    
    // Destroy mutex
    pthread_mutex_destroy(&g_context.mutex);
    
    return no_leaks;
}

void memory_tracking_track_alloc(void* ptr, size_t size, const char* file, int line, const char* func) {
    if (!g_context.enabled || !ptr) {
        return;
    }
    
    pthread_mutex_lock(&g_context.mutex);
    
    // Create a new allocation record
    AllocationRecord* record = malloc(sizeof(AllocationRecord));
    if (!record) {
        fprintf(stderr, "Failed to allocate memory for tracking record\n");
        pthread_mutex_unlock(&g_context.mutex);
        return;
    }
    
    // Initialize the record
    record->ptr = ptr;
    record->size = size;
    record->file = file;
    record->line = line;
    record->func = func;
    
    // Add the record to the hash table
    size_t index = hash_pointer(ptr, g_context.table_size);
    record->next = g_context.records[index];
    g_context.records[index] = record;
    
    // Update statistics
    g_context.total_allocations++;
    g_context.active_allocations++;
    g_context.total_bytes_allocated += size;
    g_context.active_bytes_allocated += size;
    
    // Update peak memory usage
    if (g_context.active_bytes_allocated > g_context.peak_bytes_allocated) {
        g_context.peak_bytes_allocated = g_context.active_bytes_allocated;
    }
    
    pthread_mutex_unlock(&g_context.mutex);
}

void memory_tracking_track_free(void* ptr, const char* file, int line, const char* func) {
    if (!g_context.enabled || !ptr) {
        return;
    }
    
    pthread_mutex_lock(&g_context.mutex);
    
    // Find the allocation record
    size_t index = hash_pointer(ptr, g_context.table_size);
    AllocationRecord** pp = &g_context.records[index];
    AllocationRecord* record = *pp;
    
    while (record) {
        if (record->ptr == ptr) {
            // Remove the record from the hash table
            *pp = record->next;
            
            // Update statistics
            g_context.active_allocations--;
            g_context.active_bytes_allocated -= record->size;
            
            // Free the record
            free(record);
            
            pthread_mutex_unlock(&g_context.mutex);
            return;
        }
        
        pp = &record->next;
        record = record->next;
    }
    
    // If we get here, the pointer was not found
    fprintf(stderr, "Memory tracking error: Attempt to free untracked pointer %p at %s:%d in %s\n",
            ptr, file, line, func);
    
    pthread_mutex_unlock(&g_context.mutex);
}

MemoryStats memory_tracking_get_stats(void) {
    MemoryStats stats;
    
    pthread_mutex_lock(&g_context.mutex);
    
    stats.total_allocations = g_context.total_allocations;
    stats.active_allocations = g_context.active_allocations;
    stats.total_bytes_allocated = g_context.total_bytes_allocated;
    stats.active_bytes_allocated = g_context.active_bytes_allocated;
    stats.peak_bytes_allocated = g_context.peak_bytes_allocated;
    
    pthread_mutex_unlock(&g_context.mutex);
    
    return stats;
}

void memory_tracking_print_stats(void) {
    MemoryStats stats = memory_tracking_get_stats();
    
    printf("Memory Tracking Statistics:\n");
    printf("  Total allocations: %zu\n", stats.total_allocations);
    printf("  Active allocations: %zu\n", stats.active_allocations);
    printf("  Total bytes allocated: %zu\n", stats.total_bytes_allocated);
    printf("  Active bytes allocated: %zu\n", stats.active_bytes_allocated);
    printf("  Peak bytes allocated: %zu\n", stats.peak_bytes_allocated);
}

size_t memory_tracking_print_leaks(void) {
    size_t leak_count = 0;
    
    pthread_mutex_lock(&g_context.mutex);
    
    if (g_context.active_allocations > 0) {
        printf("Memory Leak Report:\n");
        
        for (size_t i = 0; i < g_context.table_size; i++) {
            AllocationRecord* record = g_context.records[i];
            
            while (record) {
                printf("  Leak: %zu bytes at %p, allocated at %s:%d in %s\n",
                       record->size, record->ptr, record->file, record->line, record->func);
                
                leak_count++;
                record = record->next;
            }
        }
        
        printf("Total leaks: %zu\n", leak_count);
    } else {
        printf("No memory leaks detected.\n");
    }
    
    pthread_mutex_unlock(&g_context.mutex);
    
    return leak_count;
}

void memory_tracking_set_enabled(bool enabled) {
    pthread_mutex_lock(&g_context.mutex);
    g_context.enabled = enabled;
    pthread_mutex_unlock(&g_context.mutex);
}

bool memory_tracking_is_enabled(void) {
    bool enabled;
    
    pthread_mutex_lock(&g_context.mutex);
    enabled = g_context.enabled;
    pthread_mutex_unlock(&g_context.mutex);
    
    return enabled;
}
