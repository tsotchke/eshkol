#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "core/vector.h"
#include "core/memory.h"

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
int main();

int main() {
    return ({ printf("Hello, Eshkol!\n"); 0; });
}

