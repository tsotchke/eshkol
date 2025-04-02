#include <stdio.h>
#include <stdlib.h>

// Define a function pointer type for functions that take an int and return an int
typedef int (*FXN)(int);

// Define a function pointer type for a composed function that takes an int and returns an int
typedef int (*ComposedFXN)(int);

// Define some simple functions for testing
int square(int x) { return x * x; }
int add1(int x) { return x + 1; }
int double_value(int x) { return x * 2; }
int negate(int x) { return -x; }

// Define a closure structure to hold the function array and its size
typedef struct {
    FXN* fs;
    int size;
} Closure;

// Define a trampoline function that applies the functions in the closure to the input value
// This is the actual function that will be called when the composed function is invoked
int apply_closure(void* closure_ptr, int x) {
    Closure* closure = (Closure*)closure_ptr;
    
    // Apply each function in the array to the input value
    for (int i = 0; i < closure->size; i++) {
        x = closure->fs[i](x);
    }
    
    return x;
}

// Define a wrapper function that will be returned by compose
// This function captures the closure and forwards the call to apply_closure
int composed_function_wrapper(int x) {
    // This is a placeholder - in a real implementation, this would be generated dynamically
    // for each composed function, with the closure pointer baked in
    Closure* closure = NULL; // This would be set to the actual closure
    return apply_closure(closure, x);
}

// Define a function that creates a new function that is the composition of the given functions
// The functions are applied in the order they appear in the array
ComposedFXN compose(FXN fs[], int size) {
    // Create a closure to hold the function array and its size
    Closure* closure = (Closure*)malloc(sizeof(Closure));
    closure->fs = (FXN*)malloc(size * sizeof(FXN));
    closure->size = size;
    
    // Copy the functions into the closure
    for (int i = 0; i < size; i++) {
        closure->fs[i] = fs[i];
    }
    
    // In a real implementation, we would generate a function that captures the closure
    // and forwards the call to apply_closure
    // For now, we'll just return a pointer to the wrapper function
    return composed_function_wrapper;
}

// Alternative implementation using a global array of closures
#define MAX_CLOSURES 100
Closure* closures[MAX_CLOSURES];
int closure_count = 0;

// Define a set of wrapper functions, one for each possible closure
int composed_function_0(int x) { return apply_closure(closures[0], x); }
int composed_function_1(int x) { return apply_closure(closures[1], x); }
int composed_function_2(int x) { return apply_closure(closures[2], x); }
// ... and so on for as many closures as needed

// Array of wrapper functions
ComposedFXN wrapper_functions[] = {
    composed_function_0,
    composed_function_1,
    composed_function_2,
    // ... and so on for as many closures as needed
};

// Define a function that creates a new function that is the composition of the given functions
// using the global array of closures
ComposedFXN compose_global(FXN fs[], int size) {
    // Check if we've reached the maximum number of closures
    if (closure_count >= MAX_CLOSURES) {
        printf("Error: Maximum number of closures reached\n");
        return NULL;
    }
    
    // Create a closure to hold the function array and its size
    Closure* closure = (Closure*)malloc(sizeof(Closure));
    closure->fs = (FXN*)malloc(size * sizeof(FXN));
    closure->size = size;
    
    // Copy the functions into the closure
    for (int i = 0; i < size; i++) {
        closure->fs[i] = fs[i];
    }
    
    // Store the closure in the global array
    closures[closure_count] = closure;
    
    // Return the corresponding wrapper function
    return wrapper_functions[closure_count++];
}

// Function to swap two functions in an array
void swap_functions(FXN fs[], int i, int j) {
    FXN temp = fs[i];
    fs[i] = fs[j];
    fs[j] = temp;
}

// Direct implementation of function composition without using closures
// This approach avoids the need for dynamic function generation
int compose2_impl(int x, FXN f, FXN g) {
    return f(g(x));
}

int compose3_impl(int x, FXN f, FXN g, FXN h) {
    return f(g(h(x)));
}

// Wrapper functions for specific compositions
int square_then_add1(int x) {
    return compose2_impl(x, add1, square);
}

int add1_then_square(int x) {
    return compose2_impl(x, square, add1);
}

int double_then_add1_then_square(int x) {
    return compose3_impl(x, square, add1, double_value);
}

// Implementation using function pointers stored in a struct
typedef struct {
    FXN f;
    FXN g;
} BinaryComposition;

int apply_binary_composition(BinaryComposition* comp, int x) {
    return comp->f(comp->g(x));
}

BinaryComposition* create_binary_composition(FXN f, FXN g) {
    BinaryComposition* comp = (BinaryComposition*)malloc(sizeof(BinaryComposition));
    comp->f = f;
    comp->g = g;
    return comp;
}

// Example of how to use the binary composition
int use_binary_composition(BinaryComposition* comp, int x) {
    return apply_binary_composition(comp, x);
}

int main() {
    // Create a function array
    FXN fs[] = {square, add1, double_value};
    int size = 3;
    
    // Test the direct implementation
    printf("Testing direct implementation:\n");
    printf("square_then_add1(5) = %d\n", square_then_add1(5));  // add1(square(5)) = add1(25) = 26
    printf("add1_then_square(5) = %d\n", add1_then_square(5));  // square(add1(5)) = square(6) = 36
    printf("double_then_add1_then_square(5) = %d\n", double_then_add1_then_square(5));  // square(add1(double(5))) = square(add1(10)) = square(11) = 121
    
    // Test the binary composition implementation
    printf("\nTesting binary composition implementation:\n");
    BinaryComposition* comp1 = create_binary_composition(add1, square);
    printf("add1(square(5)) = %d\n", use_binary_composition(comp1, 5));  // add1(square(5)) = add1(25) = 26
    
    BinaryComposition* comp2 = create_binary_composition(square, add1);
    printf("square(add1(5)) = %d\n", use_binary_composition(comp2, 5));  // square(add1(5)) = square(6) = 36
    
    // Test the global closure implementation
    printf("\nTesting global closure implementation:\n");
    ComposedFXN composed1 = compose_global(fs, size);
    printf("composed1(5) = %d\n", composed1(5));  // double_value(add1(square(5))) = double_value(add1(25)) = double_value(26) = 52
    
    // Reorder the functions in the array
    swap_functions(fs, 0, 2);
    
    // Create a new composed function with the reordered array
    ComposedFXN composed2 = compose_global(fs, size);
    printf("composed2(5) = %d\n", composed2(5));  // square(add1(double_value(5))) = square(add1(10)) = square(11) = 121
    
    // Clean up
    for (int i = 0; i < closure_count; i++) {
        free(closures[i]->fs);
        free(closures[i]);
    }
    
    free(comp1);
    free(comp2);
    
    return 0;
}
