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
float factorial(float);
int32_t main();

float factorial(float n) {
    return ((n < 2) ? 1 : (n * factorial((n - 1))));
}

int32_t main() {
    return ({ printf("Factorial of 5 is %d\n", factorial(5)); printf("Factorial of 10 is %d\n", factorial(10)); 0; });
}

