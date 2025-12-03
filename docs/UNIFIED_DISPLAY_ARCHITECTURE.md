# Unified Display Architecture for Eshkol

## Overview

This document describes the architecture for a unified display system that:
1. Handles all value types consistently
2. Provides full homoiconicity (lambdas display as their source s-expressions)
3. Eliminates duplicate display loops in LLVM codegen
4. Is maintainable and extensible

## Current Problems

### Multiple Display Loops
Currently there are 4+ display implementations in `llvm_codegen.cpp`:
- `display_sexpr_list_func` (~line 1870) - recursive s-expression display
- Tagged value display (~line 8720) - handles `(display expr)` for tagged values
- Sexpr display loop (~line 9008) - another s-expression path
- Element display loop (~line 9660) - yet another display path

Each handles types differently, leading to bugs when one loop doesn't handle a type that another does.

### LAMBDA_SEXPR Mishandling
- LAMBDA_SEXPR stores a **function pointer**, not an s-expression pointer
- Display code tries to recursively display the function pointer as a list
- This causes type=244 corruption errors

### No Lambda Registry
- No runtime mechanism to map function pointers back to their s-expressions
- `lambda_sexpr_map` exists at compile time but isn't available at runtime

## Proposed Architecture

### 1. Type System (Existing)

```c
typedef enum {
    ESHKOL_VALUE_NULL        = 0,   // Empty/null value - display as "()"
    ESHKOL_VALUE_INT64       = 1,   // Integer - display as "%lld"
    ESHKOL_VALUE_DOUBLE      = 2,   // Double - display as "%g"
    ESHKOL_VALUE_CONS_PTR    = 3,   // List - display recursively
    ESHKOL_VALUE_DUAL_NUMBER = 4,   // Dual number - display as "(dual real dual)"
    ESHKOL_VALUE_AD_NODE_PTR = 5,   // AD node - display as "#<ad-node>"
    ESHKOL_VALUE_TENSOR_PTR  = 6,   // Tensor - display as "#(...)"
    ESHKOL_VALUE_LAMBDA_SEXPR = 7,  // Lambda - lookup sexpr and display
    ESHKOL_VALUE_STRING_PTR  = 8,   // String - display as "%s" or "\"%s\""
    ESHKOL_VALUE_CHAR        = 9,   // Character - display as "#\\c"
    ESHKOL_VALUE_VECTOR_PTR  = 10,  // Vector - display as "#(...)"
    ESHKOL_VALUE_SYMBOL      = 11,  // Symbol - display as "%s"
    ESHKOL_VALUE_CLOSURE_PTR = 12,  // Closure - extract sexpr and display
    ESHKOL_VALUE_BOOL        = 13,  // Boolean - display as "#t" or "#f"
} eshkol_value_type_t;
```

### 2. Lambda Registry (New)

A runtime table mapping function pointers to their s-expressions.

```c
// In eshkol.h
typedef struct {
    uint64_t func_ptr;    // Function pointer as uint64
    uint64_t sexpr_ptr;   // Pointer to s-expression cons cell
    const char* name;     // Optional: function name for debugging
} eshkol_lambda_entry_t;

typedef struct {
    eshkol_lambda_entry_t* entries;
    size_t count;
    size_t capacity;
} eshkol_lambda_registry_t;

// Global registry
extern eshkol_lambda_registry_t* g_lambda_registry;

// API
void eshkol_lambda_registry_init(void);
void eshkol_lambda_registry_add(uint64_t func_ptr, uint64_t sexpr_ptr, const char* name);
uint64_t eshkol_lambda_registry_lookup(uint64_t func_ptr);
void eshkol_lambda_registry_destroy(void);
```

### 3. Unified Display API (New)

All display goes through these C functions:

```c
// In eshkol.h

// Display options
typedef struct {
    int max_depth;        // Maximum recursion depth (default: 100)
    bool quote_strings;   // Whether to quote strings (default: false for display, true for write)
    bool show_types;      // Debug: show type tags (default: false)
    FILE* output;         // Output stream (default: stdout)
} eshkol_display_opts_t;

// Main display functions
void eshkol_display(const eshkol_tagged_value_t* value);
void eshkol_display_with_opts(const eshkol_tagged_value_t* value, const eshkol_display_opts_t* opts);
void eshkol_write(const eshkol_tagged_value_t* value);  // Scheme 'write' semantics

// Internal helpers (also exported for LLVM to call directly)
void eshkol_display_list(uint64_t cons_ptr, int depth, const eshkol_display_opts_t* opts);
void eshkol_display_atom(const eshkol_tagged_value_t* value, const eshkol_display_opts_t* opts);
```

### 4. Display Implementation

```c
// In arena_memory.cpp (or new display.cpp)

void eshkol_display(const eshkol_tagged_value_t* value) {
    eshkol_display_opts_t opts = {
        .max_depth = 100,
        .quote_strings = false,
        .show_types = false,
        .output = stdout
    };
    eshkol_display_with_opts(value, &opts);
}

void eshkol_display_with_opts(const eshkol_tagged_value_t* value,
                               const eshkol_display_opts_t* opts) {
    if (!value) {
        fprintf(opts->output, "()");
        return;
    }

    uint8_t type = value->type & 0x0F;  // Mask to get base type

    switch (type) {
        case ESHKOL_VALUE_NULL:
            fprintf(opts->output, "()");
            break;

        case ESHKOL_VALUE_INT64:
            fprintf(opts->output, "%lld", (long long)value->data.int_val);
            break;

        case ESHKOL_VALUE_DOUBLE:
            fprintf(opts->output, "%g", value->data.double_val);
            break;

        case ESHKOL_VALUE_BOOL:
            fprintf(opts->output, value->data.int_val ? "#t" : "#f");
            break;

        case ESHKOL_VALUE_STRING_PTR:
            if (opts->quote_strings) {
                fprintf(opts->output, "\"%s\"", (const char*)value->data.ptr_val);
            } else {
                fprintf(opts->output, "%s", (const char*)value->data.ptr_val);
            }
            break;

        case ESHKOL_VALUE_SYMBOL:
            fprintf(opts->output, "%s", (const char*)value->data.ptr_val);
            break;

        case ESHKOL_VALUE_CONS_PTR:
            eshkol_display_list(value->data.ptr_val, 0, opts);
            break;

        case ESHKOL_VALUE_LAMBDA_SEXPR:
            eshkol_display_lambda(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_CLOSURE_PTR:
            eshkol_display_closure(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_TENSOR_PTR:
            eshkol_display_tensor(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_VECTOR_PTR:
            eshkol_display_vector(value->data.ptr_val, opts);
            break;

        case ESHKOL_VALUE_CHAR:
            eshkol_display_char((uint32_t)value->data.int_val, opts);
            break;

        default:
            fprintf(opts->output, "#<unknown-type-%d>", type);
            break;
    }
}

// Display a lambda by looking up its s-expression
void eshkol_display_lambda(uint64_t func_ptr, const eshkol_display_opts_t* opts) {
    uint64_t sexpr = eshkol_lambda_registry_lookup(func_ptr);
    if (sexpr != 0) {
        // Found s-expression - display it
        eshkol_display_list(sexpr, 0, opts);
    } else {
        // No s-expression found - display placeholder with address
        fprintf(opts->output, "(lambda #<0x%llx>)", (unsigned long long)func_ptr);
    }
}

// Display a closure by extracting its s-expression
void eshkol_display_closure(uint64_t closure_ptr, const eshkol_display_opts_t* opts) {
    if (closure_ptr == 0) {
        fprintf(opts->output, "#<closure>");
        return;
    }

    // Closure struct: { func_ptr (8), env (8), sexpr_ptr (8) }
    uint64_t* closure = (uint64_t*)closure_ptr;
    uint64_t sexpr = closure[2];  // sexpr_ptr at offset 16

    if (sexpr != 0) {
        eshkol_display_list(sexpr, 0, opts);
    } else {
        fprintf(opts->output, "#<closure>");
    }
}

// Display a list (cons cells)
void eshkol_display_list(uint64_t cons_ptr, int depth, const eshkol_display_opts_t* opts) {
    if (depth > opts->max_depth) {
        fprintf(opts->output, "...");
        return;
    }

    if (cons_ptr == 0) {
        fprintf(opts->output, "()");
        return;
    }

    fprintf(opts->output, "(");

    uint64_t current = cons_ptr;
    bool first = true;

    while (current != 0) {
        arena_tagged_cons_cell_t* cell = (arena_tagged_cons_cell_t*)current;

        if (!first) {
            fprintf(opts->output, " ");
        }
        first = false;

        // Display car
        eshkol_display_with_opts(&cell->car, opts);

        // Check cdr type
        uint8_t cdr_type = cell->cdr.type & 0x0F;

        if (cdr_type == ESHKOL_VALUE_NULL) {
            // Proper list end
            break;
        } else if (cdr_type == ESHKOL_VALUE_CONS_PTR) {
            // Continue to next cell
            current = cell->cdr.data.ptr_val;
        } else {
            // Dotted pair
            fprintf(opts->output, " . ");
            eshkol_display_with_opts(&cell->cdr, opts);
            break;
        }
    }

    fprintf(opts->output, ")");
}
```

### 5. LLVM Codegen Simplification

Replace all inline display loops with:

```cpp
// In llvm_codegen.cpp

// Declare the C display functions
Function* eshkol_display_func;
Function* eshkol_display_list_func;

void declareDisplayFunctions() {
    // void eshkol_display(const eshkol_tagged_value_t* value)
    std::vector<Type*> display_args = {PointerType::getUnqual(*context)};
    FunctionType* display_type = FunctionType::get(
        Type::getVoidTy(*context), display_args, false);
    eshkol_display_func = Function::Create(
        display_type, Function::ExternalLinkage, "eshkol_display", module.get());
}

// When generating (display expr):
Value* codegenDisplay(ASTNode* expr) {
    Value* value = codegenAST(expr);

    // Convert to tagged_value if needed
    Value* tagged = ensureTaggedValue(value);

    // Store to stack for passing by pointer
    Value* ptr = builder->CreateAlloca(tagged_value_type);
    builder->CreateStore(tagged, ptr);

    // Call unified display
    builder->CreateCall(eshkol_display_func, {ptr});

    return ConstantInt::get(Type::getInt32Ty(*context), 0);
}
```

### 6. Lambda Registry Population

At program initialization, populate the registry:

```cpp
// In llvm_codegen.cpp - generate __eshkol_init__ function

void generateLambdaRegistryInit() {
    // For each lambda in lambda_sexpr_map:
    for (auto& [func, name] : lambda_sexpr_map) {
        std::string sexpr_global = name + "_sexpr";
        GlobalVariable* sexpr_var = module->getNamedGlobal(sexpr_global);

        if (sexpr_var) {
            // Emit: eshkol_lambda_registry_add(func_ptr, sexpr_ptr, name)
            Value* func_ptr = builder->CreatePtrToInt(func, Type::getInt64Ty(*context));
            Value* sexpr_ptr = builder->CreateLoad(Type::getInt64Ty(*context), sexpr_var);
            Value* name_str = codegenString(name.c_str());

            builder->CreateCall(lambda_registry_add_func, {func_ptr, sexpr_ptr, name_str});
        }
    }
}
```

## Migration Plan

### Phase 1: Add Infrastructure
1. Add `eshkol_lambda_registry_t` to `eshkol.h`
2. Implement registry functions in `arena_memory.cpp`
3. Add `eshkol_display*` function declarations

### Phase 2: Implement C Display
1. Implement `eshkol_display_with_opts()` in `arena_memory.cpp`
2. Implement `eshkol_display_list()`
3. Implement `eshkol_display_lambda()` with registry lookup
4. Implement other type-specific display functions

### Phase 3: LLVM Integration
1. Declare display functions in LLVM codegen
2. Generate lambda registry population in `__eshkol_init__`
3. Replace inline display code with calls to C functions

### Phase 4: Cleanup
1. Remove old display loops from llvm_codegen.cpp
2. Remove `display_sexpr_list_func` generation
3. Test all display cases

## Testing

```scheme
;; Test cases for unified display

;; Atoms
(display 42)         ; => 42
(display 3.14)       ; => 3.14
(display #t)         ; => #t
(display #f)         ; => #f
(display "hello")    ; => hello
(display 'symbol)    ; => symbol

;; Lists
(display '())        ; => ()
(display '(1 2 3))   ; => (1 2 3)
(display '(1 . 2))   ; => (1 . 2)
(display '((1 2) (3 4))) ; => ((1 2) (3 4))

;; Lambdas (homoiconic!)
(define (double x) (* x 2))
(display double)                    ; => (lambda (x) (* x 2))
(display (list double))             ; => ((lambda (x) (* x 2)))
(display (cons double '()))         ; => ((lambda (x) (* x 2)))

;; Closures
(define (make-adder n) (lambda (x) (+ x n)))
(define add5 (make-adder 5))
(display add5)                      ; => (lambda (x) (+ x n))

;; Mixed lists
(display (list 1 double "hello" #t))
; => (1 (lambda (x) (* x 2)) hello #t)
```

## Benefits

1. **Single Source of Truth**: One display implementation in C
2. **Consistent Type Handling**: All types handled uniformly
3. **Full Homoiconicity**: Lambdas display their source code
4. **Maintainability**: Much less code in llvm_codegen.cpp
5. **Extensibility**: Easy to add new types
6. **Debuggability**: Clear control flow, easy to trace
