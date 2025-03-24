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
int factorial(int);
int main();

int factorial(int n) {
    return ((n < 2) ? 1 : (n * factorial((n - 1))));
}

int main() {
    return ({ printf("Factorial of 5 is %d\n", factorial(5)); printf("Factorial of 10 is %d\n", factorial(10)); 0; });
}

