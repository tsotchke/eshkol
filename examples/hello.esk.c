#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "core/vector.h"
#include "core/memory.h"
#include "core/autodiff.h"

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
int32_t main();

