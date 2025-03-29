#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#include "core/vector.h"
#include "core/memory.h"
#include "core/autodiff.h"

// Minimal Arena implementation
typedef struct Arena {
    void* memory;
    size_t size;
    size_t used;
} Arena;

Arena* arena_create(size_t size) {
    Arena* arena = malloc(sizeof(Arena));
    if (!arena) return NULL;
    arena->memory = malloc(size);
    if (!arena->memory) {
        free(arena);
        return NULL;
    }
    arena->size = size;
    arena->used = 0;
    return arena;
}

void arena_destroy(Arena* arena) {
    if (arena) {
        free(arena->memory);
        free(arena);
    }
}

void* arena_alloc(Arena* arena, size_t size) {
    if (!arena || arena->used + size > arena->size) return NULL;
    void* ptr = (char*)arena->memory + arena->used;
    arena->used += size;
    return ptr;
}

// Global arena for memory allocations
Arena* arena = NULL;

// Eshkol value type
typedef union {
    long integer;
    double floating;
    bool boolean;
    char character;
    char* string;
    void* pointer;
} eshkol_value_t;

// Forward declarations
float factorial(float);
int32_t main();

float factorial(float n) {
    // Tail call optimization loop
    while (1) {
        return (((n < 2)) ? (1) : ((n * factorial((n - 1)))));
    }
}
int32_t main() {
    // Tail call optimization loop
    while (1) {
        printf("Factorial of 5 is %d\n", (int)(factorial(5)));
        printf("Factorial of 10 is %d\n", (int)(factorial(10)));
        return 0;
    }
}
// Initialize arena before main
static void __attribute__((constructor)) init_arena() {
    arena = arena_create(1024 * 1024);
    if (!arena) {
        fprintf(stderr, "Failed to create memory arena\n");
        exit(1);
    }
}

// Clean up arena after main
static void __attribute__((destructor)) cleanup_arena() {
    if (arena) {
        arena_destroy(arena);
        arena = NULL;
    }
}

