# Closures and Global Variables in Eshkol

## Table of Contents
- [Introduction](#introduction)
- [Closure System](#closure-system)
  - [Closure Structure](#closure-structure)
  - [Environment Structure](#environment-structure)
  - [Closure Operations](#closure-operations)
  - [Memory Management](#memory-management)
- [Global Variables](#global-variables)
  - [Initialization Challenges](#initialization-challenges)
  - [Two-Phase Initialization](#two-phase-initialization)
  - [Safe Usage Patterns](#safe-usage-patterns)
  - [Static Constructor Approach](#static-constructor-approach)
- [Code Examples](#code-examples)
  - [Basic Closures](#basic-closures)
  - [Captured Variables](#captured-variables)
  - [Function Composition](#function-composition)
  - [Global Closures](#global-closures)
- [Common Issues](#common-issues)
  - [NULL Closures](#null-closures)
  - [Validation Failures](#validation-failures)
  - [Memory Leaks](#memory-leaks)
- [Best Practices](#best-practices)

## Introduction

Closures and global variables are fundamental components in Eshkol programming. This document explains their implementation, usage patterns, and best practices.

## Closure System

Eshkol implements a comprehensive closure system that supports first-class functions, lexical scoping, and proper variable capture. This enables functional programming patterns like higher-order functions, currying, and function composition.

### Closure Structure

The core of the closure system is the `EshkolClosure` structure:

```c
typedef struct EshkolClosure {
    // Function pointer for closure implementation
    void* (*fn)(struct EshkolEnvironment*, void**);
    
    // Environment containing captured variables
    struct EshkolEnvironment* env;
    
    // Function metadata (optional)
    struct {
        const char* name;
        unsigned int param_count;
        void* signature;
    } meta;
    
    // Composition metadata (for composed closures)
    struct {
        struct EshkolClosure* f;
        struct EshkolClosure* g;
    } composition;
} EshkolClosure;
```

Key components:
- `fn`: The function pointer that implements the closure's behavior
- `env`: The environment containing captured variables
- `meta`: Metadata about the function (name, parameter count, etc.)
- `composition`: Used for composed closures to track the component functions

### Environment Structure

The environment structure stores captured variables and maintains the lexical scope chain:

```c
typedef struct EshkolEnvironment {
    // Captured variables
    void** variables;
    
    // Number of captured variables
    size_t count;
    
    // Parent environment (for lexical scoping)
    struct EshkolEnvironment* parent;
    
    // Reference count for memory management
    uint32_t ref_count;
} EshkolEnvironment;
```

Key concepts:
- Each environment can have a parent environment, forming a chain of lexical scopes
- Variables are stored as opaque pointers (`void*`) and must be cast to the appropriate type when accessed
- Reference counting is used to manage environment lifetime

### Closure Operations

The closure system provides several key operations:

1. **Creation**: Creating a new closure with a function and environment
```c
EshkolClosure* eshkol_closure_create(
    void* (*fn)(EshkolEnvironment*, void**),
    EshkolEnvironment* env,
    const char* name,
    void* signature,
    unsigned int param_count
);
```

2. **Invocation**: Calling a closure with arguments
```c
void* eshkol_closure_call(EshkolClosure* closure, void** args);
```

3. **Validation**: Checking if a pointer is a valid closure
```c
bool eshkol_is_closure(void* ptr);
```

4. **Composition**: Composing two closures into a new closure
```c
EshkolClosure* eshkol_closure_compose(
    EshkolClosure* f, 
    EshkolClosure* g
);
```

5. **Safe Invocation**: Safely calling a closure with validation
```c
void* call_closure(void* closure, void* arg) {
    if (closure == NULL) {
        fprintf(stderr, "Error: NULL closure in call\n");
        return NULL;
    }
    
    if (eshkol_is_closure(closure)) {
        return eshkol_closure_call((EshkolClosure*)closure, (void*[]){arg});
    } else {
        return ((void* (*)(void*))closure)(arg);
    }
}
```

### Memory Management

Closures and environments use a combination of arena allocation and reference counting:

1. **Arena Allocation**: Closures and environments are typically allocated from an arena for efficient memory management

2. **Reference Counting**: Environments use reference counting to track when they can be safely cleaned up
```c
void eshkol_environment_retain(EshkolEnvironment* env);
void eshkol_environment_release(EshkolEnvironment* env);
```

3. **Shared Environments**: Multiple closures can share an environment, which is tracked through reference counting

4. **Cleanup**: When a closure is no longer needed, its environment should be released

## Global Variables

Global variables in Eshkol, especially those containing closures, require special handling due to limitations in C's initialization capabilities.

### Initialization Challenges

1. **C Initialization Limitations**: C doesn't allow non-constant expressions for global variable initialization
2. **Function Calls**: Function calls cannot be used to initialize globals at compile time
3. **Closures in Globals**: Closures cannot be directly initialized in global scope
4. **Initialization Order**: Dependencies between globals require careful initialization order

### Two-Phase Initialization

To address these challenges, Eshkol uses a two-phase initialization approach:

#### 1. Declaration Phase

Global variables are declared with default values (typically NULL):

```c
// Global variable declarations
void* global_closure = NULL;
int* global_data = NULL;
```

#### 2. Initialization Phase

Proper initialization happens in the `main` function:

```c
int main(int argc, char** argv) {
    // Initialize arena and environment
    arena = arena_create(1024 * 1024);
    env = eshkol_environment_create(NULL, 10, 0);
    
    // Initialize global variables
    global_closure = create_my_closure();
    global_data = create_my_data();
    
    // Rest of program...
    
    return 0;
}
```

This approach ensures that:
- Global variables are properly declared in the global scope
- Initialization happens in a context where function calls are allowed
- Variables are initialized before they're used

### Safe Usage Patterns

To safely use global variables, especially those containing closures:

1. **Validation**: Always validate closures before use
```c
if (global_closure == NULL) {
    fprintf(stderr, "Error: Global closure not initialized\n");
    return ERROR_CODE;
}
```

2. **Safe Calling**: Use the `call_closure` helper function for safe invocation
```c
result = call_closure(global_closure, arg);
```

3. **Initialization Checks**: Consider adding initialization flags for complex systems
```c
bool globals_initialized = false;

void initialize_globals() {
    if (globals_initialized) return;
    
    // Initialize globals...
    
    globals_initialized = true;
}

void use_globals() {
    if (!globals_initialized) {
        initialize_globals();
    }
    
    // Use globals...
}
```

### Static Constructor Approach

On platforms that support it, static constructors can provide a cleaner approach:

```c
void* global_closure = NULL;

// Static constructor runs before main
static void __attribute__((constructor)) initialize_globals() {
    arena = arena_create(1024 * 1024);
    env = eshkol_environment_create(NULL, 10, 0);
    global_closure = create_my_closure();
}

// Static destructor runs after main
static void __attribute__((destructor)) cleanup_globals() {
    // Release global resources
    arena_destroy(arena);
}
```

**Note**: This approach is not portable across all platforms and should be used with caution.

## Code Examples

### Basic Closures

```scheme
;; Define a simple lambda function
(define add1 (lambda (x) (+ x 1)))

;; Use the closure
(display (add1 5))  ;; Displays 6
```

Generated C code (simplified):

```c
// Lambda function implementation
void* add1_lambda(EshkolEnvironment* env, void** args) {
    int x = (int)(intptr_t)args[0];
    return (void*)(intptr_t)(x + 1);
}

// Create the closure
void* add1 = eshkol_closure_create(add1_lambda, NULL, "add1", NULL, 1);

// Call the closure
int result = (int)(intptr_t)eshkol_closure_call(add1, (void*[]){(void*)5});
printf("%d\n", result);  // Prints 6
```

### Captured Variables

```scheme
;; Create a counter with a captured variable
(define (make-counter initial-value)
  (let ((count initial-value))
    (lambda ()
      (set! count (+ count 1))
      count)))

;; Create two distinct counters
(define counter1 (make-counter 0))
(define counter2 (make-counter 10))

;; Each counter has its own captured state
(display (counter1))  ;; Displays 1
(display (counter1))  ;; Displays 2
(display (counter2))  ;; Displays 11
```

Generated C code (simplified):

```c
// Counter lambda implementation
void* counter_lambda(EshkolEnvironment* env, void** args) {
    // Access captured variable
    int* count = (int*)env->variables[0];
    
    // Update count
    *count = *count + 1;
    
    // Return new count
    return (void*)(intptr_t)*count;
}

// Function to create a counter
void* make_counter(int initial_value) {
    // Create environment
    EshkolEnvironment* env = eshkol_environment_create(arena, 1, NULL);
    
    // Capture variable
    int* count = arena_alloc(arena, sizeof(int));
    *count = initial_value;
    env->variables[0] = count;
    
    // Create and return closure
    return eshkol_closure_create(counter_lambda, env, "counter", NULL, 0);
}

// Create two counters
void* counter1 = make_counter(0);
void* counter2 = make_counter(10);

// Use the counters
printf("%d\n", (int)(intptr_t)eshkol_closure_call(counter1, NULL));  // Prints 1
printf("%d\n", (int)(intptr_t)eshkol_closure_call(counter1, NULL));  // Prints 2
printf("%d\n", (int)(intptr_t)eshkol_closure_call(counter2, NULL));  // Prints 11
```

### Function Composition

```scheme
;; Define the compose function
(define (compose f g)
  (lambda (x) (f (g x))))

;; Define simple functions
(define (add1 x) (+ x 1))
(define (mul2 x) (* x 2))

;; Compose them
(define add1-then-mul2 (compose mul2 add1))
(define mul2-then-add1 (compose add1 mul2))

;; Use the composed functions
(display (add1-then-mul2 5))  ;; (5+1)*2 = 12
(display (mul2-then-add1 5))  ;; (5*2)+1 = 11
```

Generated C code (simplified):

```c
// Compose lambda implementation
void* compose_lambda(EshkolEnvironment* env, void** args) {
    // Get captured functions f and g
    EshkolClosure* f = (EshkolClosure*)env->variables[0];
    EshkolClosure* g = (EshkolClosure*)env->variables[1];
    
    // Get argument x
    void* x = args[0];
    
    // Apply g to x
    void* g_result = eshkol_closure_call(g, (void*[]){x});
    
    // Apply f to the result of g
    return eshkol_closure_call(f, (void*[]){g_result});
}

// Function to compose two functions
void* compose(void* f, void* g) {
    // Create environment
    EshkolEnvironment* env = eshkol_environment_create(arena, 2, NULL);
    
    // Capture functions
    env->variables[0] = f;
    env->variables[1] = g;
    
    // Retain captured closures
    eshkol_closure_retain((EshkolClosure*)f);
    eshkol_closure_retain((EshkolClosure*)g);
    
    // Create and return composition closure
    return eshkol_closure_create(compose_lambda, env, "compose_result", NULL, 1);
}
```

### Global Closures

```scheme
;; Define a function that creates a closure
(define (identity-maker)
  (lambda (x) x))

;; Global variable to hold the closure
(define identity (identity-maker))

;; Use the global closure
(display (identity 42))  ;; Displays 42
```

Generated C code:

```c
// Function that creates a closure
void* identity_maker() {
    // Create environment
    EshkolEnvironment* env = eshkol_environment_create(arena, 0, NULL);
    
    // Create and return closure
    return eshkol_closure_create(identity_lambda, env, "identity", NULL, 1);
}

// Lambda function implementation
void* identity_lambda(EshkolEnvironment* env, void** args) {
    void* x = args[0];
    return x;
}

// Global variable declaration with NULL initial value
void* identity = NULL;

int main(int argc, char** argv) {
    // Initialize arena
    arena = arena_create(1024 * 1024);
    
    // Initialize environment
    env = eshkol_environment_create(NULL, 10, 0);
    
    // Initialize global closure
    identity = identity_maker();
    
    // Use global closure with safe calling
    printf("identity(42): %ld\n", (long)call_closure(identity, (void*)42));
    
    // Cleanup
    arena_destroy(arena);
    
    return 0;
}
```

## Common Issues

### NULL Closures

**Problem**: Attempting to call a NULL closure causes a runtime error.

**Symptoms**:
- "Error: NULL closure in call" message
- Segmentation fault (if not properly validated)

**Causes**:
- Global closure not properly initialized
- Closure creation failed
- Closure was released prematurely

**Solution**:
- Always validate closures before use
- Use the `call_closure` helper function for safe invocation
- Ensure proper initialization of global closures in `main`

### Validation Failures

**Problem**: A pointer that is not a valid closure is passed to a closure operation.

**Symptoms**:
- "Error: Not a valid closure" message
- Segmentation fault

**Causes**:
- Memory corruption
- Using a regular function pointer as a closure
- Using a freed closure

**Solution**:
- Use `eshkol_is_closure` to validate closures before operations
- Ensure proper memory management
- Don't use freed closures

### Memory Leaks

**Problem**: Memory leaks occur due to improper management of closure environments.

**Symptoms**:
- Increasing memory usage over time
- Not enough memory errors

**Causes**:
- Not releasing closures properly
- Circular references between closures
- Not using reference counting correctly

**Solution**:
- Properly implement and use reference counting
- Avoid circular references in closures
- Use arenas for automatic cleanup in many cases

## Best Practices

1. **Validation**: Always validate closures before use, especially for global variables.
```c
if (eshkol_is_closure(closure)) {
    result = eshkol_closure_call(closure, args);
} else {
    // Handle error...
}
```

2. **Safety Wrapper**: Use the `call_closure` helper function for safer closure invocation.
```c
result = call_closure(my_closure, arg);
```

3. **Global Initialization**: Initialize global closures in the main function, not at global scope.
```c
// Declaration
void* global_closure = NULL;

int main() {
    // Initialization
    global_closure = create_my_closure();
    
    // Usage
    call_closure(global_closure, arg);
}
```

4. **Reference Counting**: Use proper reference counting for closures that are shared or have long lifetimes.
```c
// Retain a closure (increment reference count)
eshkol_closure_retain(closure);

// Release a closure (decrement reference count)
eshkol_closure_release(closure);
```

5. **Memory Ownership**: Be clear about ownership of closures and environments.
```c
// Taking ownership of a closure
void take_ownership(EshkolClosure* closure) {
    // Store the closure
    my_stored_closure = closure;
    
    // No need to retain, as we're assuming ownership
}

// Borrowing a closure
void borrow_closure(EshkolClosure* closure) {
    // Use the closure temporarily
    result = eshkol_closure_call(closure, args);
    
    // Don't release it when done, as we don't own it
}
```

6. **Error Handling**: Implement proper error handling for closure operations.
```c
// Safe closure calling with error handling
void* safe_call(void* closure, void* arg, bool* success) {
    if (closure == NULL) {
        *success = false;
        return NULL;
    }
    
    if (eshkol_is_closure(closure)) {
        *success = true;
        return eshkol_closure_call((EshkolClosure*)closure, (void*[]){arg});
    } else {
        *success = false;
        return NULL;
    }
}
```

7. **Documentation**: Document the closure contracts for functions that create, use, or store closures.
```c
/**
 * Creates a filter function that filters values by a predicate.
 *
 * @param predicate A closure that takes a value and returns true/false.
 *                  Ownership is shared - the closure will be retained.
 * @return A new closure that takes a list and returns a filtered list.
 *         Caller takes ownership and should release when done.
 */
EshkolClosure* create_filter(EshkolClosure* predicate);
