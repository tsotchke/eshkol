/**
 * @file test_memory_tracking.c
 * @brief Unit tests for the memory tracking system
 */

#include "core/memory_tracking.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Test memory tracking initialization and shutdown
 */
static void test_memory_tracking_init_shutdown(void) {
    printf("Testing memory tracking initialization and shutdown...\n");
    
    // Initialize memory tracking
    memory_tracking_init();
    
    // Check that memory tracking is enabled
    assert(memory_tracking_is_enabled());
    
    // Get initial statistics
    MemoryStats stats = memory_tracking_get_stats();
    assert(stats.total_allocations == 0);
    assert(stats.active_allocations == 0);
    assert(stats.total_bytes_allocated == 0);
    assert(stats.active_bytes_allocated == 0);
    assert(stats.peak_bytes_allocated == 0);
    
    // Shutdown memory tracking
    bool no_leaks = memory_tracking_shutdown();
    assert(no_leaks);
    
    printf("PASS: memory_tracking_init_shutdown\n");
}

/**
 * @brief Test memory tracking allocation and deallocation
 */
static void test_memory_tracking_alloc_free(void) {
    printf("Testing memory tracking allocation and deallocation...\n");
    
    // Initialize memory tracking
    memory_tracking_init();
    
    // Allocate some memory
    void* ptr1 = malloc(100);
    MEMORY_TRACK_ALLOC(ptr1, 100);
    
    // Check statistics
    MemoryStats stats = memory_tracking_get_stats();
    assert(stats.total_allocations == 1);
    assert(stats.active_allocations == 1);
    assert(stats.total_bytes_allocated == 100);
    assert(stats.active_bytes_allocated == 100);
    assert(stats.peak_bytes_allocated == 100);
    
    // Allocate more memory
    void* ptr2 = malloc(200);
    MEMORY_TRACK_ALLOC(ptr2, 200);
    
    // Check statistics
    stats = memory_tracking_get_stats();
    assert(stats.total_allocations == 2);
    assert(stats.active_allocations == 2);
    assert(stats.total_bytes_allocated == 300);
    assert(stats.active_bytes_allocated == 300);
    assert(stats.peak_bytes_allocated == 300);
    
    // Free the first allocation
    MEMORY_TRACK_FREE(ptr1);
    free(ptr1);
    
    // Check statistics
    stats = memory_tracking_get_stats();
    assert(stats.total_allocations == 2);
    assert(stats.active_allocations == 1);
    assert(stats.total_bytes_allocated == 300);
    assert(stats.active_bytes_allocated == 200);
    assert(stats.peak_bytes_allocated == 300);
    
    // Free the second allocation
    MEMORY_TRACK_FREE(ptr2);
    free(ptr2);
    
    // Check statistics
    stats = memory_tracking_get_stats();
    assert(stats.total_allocations == 2);
    assert(stats.active_allocations == 0);
    assert(stats.total_bytes_allocated == 300);
    assert(stats.active_bytes_allocated == 0);
    assert(stats.peak_bytes_allocated == 300);
    
    // Shutdown memory tracking
    bool no_leaks = memory_tracking_shutdown();
    assert(no_leaks);
    
    printf("PASS: memory_tracking_alloc_free\n");
}

/**
 * @brief Test memory tracking with memory leaks
 */
static void test_memory_tracking_leaks(void) {
    printf("Testing memory tracking with memory leaks...\n");
    
    // Initialize memory tracking
    memory_tracking_init();
    
    // Allocate some memory that we won't free
    void* ptr = malloc(100);
    MEMORY_TRACK_ALLOC(ptr, 100);
    
    // Check statistics
    MemoryStats stats = memory_tracking_get_stats();
    assert(stats.total_allocations == 1);
    assert(stats.active_allocations == 1);
    assert(stats.total_bytes_allocated == 100);
    assert(stats.active_bytes_allocated == 100);
    
    // Print leaks
    size_t leak_count = memory_tracking_print_leaks();
    assert(leak_count == 1);
    
    // Shutdown memory tracking
    bool no_leaks = memory_tracking_shutdown();
    assert(!no_leaks);
    
    // Free the memory to avoid actual leaks in the test
    free(ptr);
    
    printf("PASS: memory_tracking_leaks\n");
}

/**
 * @brief Test memory tracking enable/disable
 */
static void test_memory_tracking_enable_disable(void) {
    printf("Testing memory tracking enable/disable...\n");
    
    // Initialize memory tracking
    memory_tracking_init();
    
    // Disable memory tracking
    memory_tracking_set_enabled(false);
    assert(!memory_tracking_is_enabled());
    
    // Allocate some memory
    void* ptr = malloc(100);
    MEMORY_TRACK_ALLOC(ptr, 100);
    
    // Check statistics (should be unchanged)
    MemoryStats stats = memory_tracking_get_stats();
    assert(stats.total_allocations == 0);
    assert(stats.active_allocations == 0);
    assert(stats.total_bytes_allocated == 0);
    assert(stats.active_bytes_allocated == 0);
    
    // Enable memory tracking
    memory_tracking_set_enabled(true);
    assert(memory_tracking_is_enabled());
    
    // Free the memory
    MEMORY_TRACK_FREE(ptr);
    free(ptr);
    
    // Shutdown memory tracking
    bool no_leaks = memory_tracking_shutdown();
    assert(no_leaks);
    
    printf("PASS: memory_tracking_enable_disable\n");
}

/**
 * @brief Test memory tracking statistics
 */
static void test_memory_tracking_stats(void) {
    printf("Testing memory tracking statistics...\n");
    
    // Initialize memory tracking
    memory_tracking_init();
    
    // Allocate and free memory to generate statistics
    void* ptr1 = malloc(100);
    MEMORY_TRACK_ALLOC(ptr1, 100);
    
    void* ptr2 = malloc(200);
    MEMORY_TRACK_ALLOC(ptr2, 200);
    
    void* ptr3 = malloc(300);
    MEMORY_TRACK_ALLOC(ptr3, 300);
    
    MEMORY_TRACK_FREE(ptr1);
    free(ptr1);
    
    // Check statistics
    MemoryStats stats = memory_tracking_get_stats();
    assert(stats.total_allocations == 3);
    assert(stats.active_allocations == 2);
    assert(stats.total_bytes_allocated == 600);
    assert(stats.active_bytes_allocated == 500);
    assert(stats.peak_bytes_allocated == 600);
    
    // Print statistics
    memory_tracking_print_stats();
    
    // Free remaining memory
    MEMORY_TRACK_FREE(ptr2);
    free(ptr2);
    
    MEMORY_TRACK_FREE(ptr3);
    free(ptr3);
    
    // Shutdown memory tracking
    bool no_leaks = memory_tracking_shutdown();
    assert(no_leaks);
    
    printf("PASS: memory_tracking_stats\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running memory tracking tests...\n");
    
    test_memory_tracking_init_shutdown();
    test_memory_tracking_alloc_free();
    test_memory_tracking_leaks();
    test_memory_tracking_enable_disable();
    test_memory_tracking_stats();
    
    printf("All memory tracking tests passed!\n");
    return 0;
}
