#include <stdio.h>
#include <stdlib.h>

// Define a function pointer type for functions that take an int and return an int
typedef int (*FXN)(int);

// Define some simple functions for testing
int square(int x) { return x * x; }
int add1(int x) { return x + 1; }
int double_value(int x) { return x * 2; }
int negate(int x) { return -x; }

// Define a function that applies a sequence of functions to a value
// This is the 'eval' function from the original example
int eval(FXN fs[], int size, int x)
{
   for (int i = 0; i < size; i++) {
       x = fs[i](x);
   }
   return x;
}

// Define a function that creates a new function that is the composition of the given functions
// The functions are applied in the order they appear in the array
FXN* compose(FXN fs[], int size)
{
    // Create a closure structure to hold the function array and its size
    typedef struct {
        FXN* fs;
        int size;
    } Closure;
    
    // Allocate memory for the closure
    Closure* closure = (Closure*)malloc(sizeof(Closure));
    closure->fs = (FXN*)malloc(size * sizeof(FXN));
    closure->size = size;
    
    // Copy the functions into the closure
    for (int i = 0; i < size; i++) {
        closure->fs[i] = fs[i];
    }
    
    // Return a function pointer that captures the closure
    // Note: This is a simplified representation - in a real implementation,
    // we would need to use a trampoline or other technique to handle the closure
    return (FXN*)closure;
}

// Function to swap two functions in an array
void swap_functions(FXN fs[], int i, int j)
{
    FXN temp = fs[i];
    fs[i] = fs[j];
    fs[j] = temp;
}

int main()
{
    // Create a function array
    FXN fs[] = {square, add1, double_value};
    int size = 3;
    
    // Test the eval function
    printf("Testing eval function:\n");
    printf("eval(fs, 3, 5) = %d\n", eval(fs, size, 5));  // double_value(add1(square(5))) = double_value(add1(25)) = double_value(26) = 52
    
    // Reorder the functions in the array
    swap_functions(fs, 0, 2);
    
    // Test the eval function with reordered functions
    printf("\nTesting eval function with reordered functions:\n");
    printf("eval(fs, 3, 5) = %d\n", eval(fs, size, 5));  // square(add1(double_value(5))) = square(add1(10)) = square(11) = 121
    
    return 0;
}
