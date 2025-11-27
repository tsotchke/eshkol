# Symbolic Differentiation Architecture Plan

## Executive Summary

The `diff` operator should return **S-expression** (list-based) symbolic formulas that can be:
1. Displayed in human-readable form
2. Evaluated numerically at any point
3. Manipulated symbolically (future CAS foundation)

## Current Problem

```scheme
(display (diff (* x x) x))
```

**Current behavior**: Shows garbage (function pointer)  
**Desired behavior**: Shows `"2*x"` or equivalent readable form

## Best Architecture: S-Expression Symbolic Algebra

### Core Concept

Return symbolic derivatives as **quoted lists** (S-expressions):

```scheme
(diff (* x x) x)  →  '(* 2 x)      ; List representing 2*x
(diff (sin x) x)  →  '(cos x)      ; List representing cos(x)  
(diff (+ x 1) x)  →  1             ; Constants for simple cases
```

### Why S-Expressions?

1. **Homoiconic**: Code is data, data is code (Lisp fundamental principle)
2. **Native**: Lists are already first-class values in Eshkol
3. **Manipulable**: Can apply transformations, simplifications
4. **Displayable**: Convert to strings for human reading
5. **Evaluable**: Can evaluate at any variable binding
6. **Extensible**: Foundation for full Computer Algebra System

## Architecture Components

```
┌──────────────────────────────────────────────────┐
│  diff Operator (Compile-Time)                    │
│  ────────────────────────────────────────────    │
│  Input:  AST expression + variable               │
│  Output: Runtime S-expression (list structure)   │
│                                                   │
│  Example:                                         │
│    AST: (* x x)                                  │
│     ↓                                            │
│    Analyze: product rule                         │
│     ↓                                            │
│    Build: '(* 2 x)                               │
│     ↓                                            │
│    Return: cons cell list                        │
└──────────────────────────────────────────────────┘
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
┌───────────────┐   ┌──────────────────────┐
│   Display     │   │  Eval-Symbolic       │
│   Handler     │   │  (Future Phase)      │
│               │   │                      │
│ '(* 2 x)      │   │ (eval-sym '(* 2 x)   │
│   ↓           │   │   '((x . 5)))        │
│ Detect list   │   │    ↓                 │
│ with symbol   │   │   10                 │
│   ↓           │   │                      │
│ Format:       │   └──────────────────────┘
│ "2*x" or      │
│ "(* 2 x)"     │
└───────────────┘
```

## Implementation Phases

### Phase 1: Basic S-Expression Return (CURRENT SPRINT)

**Goal**: `diff` returns list structures

**Tasks**:
1. Create `buildSymbolicDerivative()` that returns `eshkol_ast_t*` (list structure)
2. Implement derivative rules that construct lists:
   ```cpp
   // d/dx(c*x) → constant c
   // d/dx(x*x) → list '(* 2 x)
   // d/dx(sin x) → list '(cos x)
   ```
3. Convert `differentiate()` from IR builder to AST builder
4. Return constructed list from `codegenDiff()`

**Output Examples**:
```scheme
(diff (* 2 x) x)     →  2        ; Constant (already works)
(diff (* x x) x)     →  '(* 2 x) ; List
(diff (sin x) x)     →  '(cos x) ; List
(diff (+ x 1) x)     →  1        ; Constant
```

### Phase 2: Smart Display Formatting (NEXT SPRINT)

**Goal**: Lists display in readable infix notation

**Tasks**:
1. Enhance `codegenDisplay()` to detect symbolic expression lists
2. Pattern match on list structure
3. Convert to infix string:
   - `'(* 2 x)` → `"2*x"`
   - `'(+ (* 2 x) 3)` → `"2*x + 3"`
   - `'(cos x)` → `"cos(x)"`

### Phase 3: Symbolic Evaluation (FUTURE)

**Goal**: Evaluate symbolic expressions at specific points

```scheme
(define deriv (diff (* x x) x))  ; deriv = '(* 2 x)
(eval-symbolic deriv '((x . 5))) ; → 10
```

**Requirements**:
- Variable binding environment
- Expression evaluator walking lists
- Integrate with numerical `derivative` operator

### Phase 4: Symbolic Simplification (FUTURE CAS)

**Goal**: Algebraic rewriting and simplification

```scheme
(simplify '(+ (* 1 x) (* x 1)))  →  '(* 2 x)
(expand '(* (+ x 1) (+ x 2)))    →  '(+ (* x x) (* 3 x) 2)
```

## Technical Implementation Details

### AST-Based Derivative Builder

Replace current IR-based `differentiate()` with AST builder:

```cpp
// OLD: Returns LLVM Value* (IR)
Value* differentiate(const eshkol_ast_t* expr, const char* var);

// NEW: Returns AST (list structure)
eshkol_ast_t* buildSymbolicDerivative(const eshkol_ast_t* expr, const char* var);
```

### Derivative Rules (AST → AST)

```cpp
eshkol_ast_t* buildSymbolicDerivative(const eshkol_ast_t* expr, const char* var) {
    switch (expr->type) {
        case ESHKOL_INT64:
        case ESHKOL_DOUBLE:
            return makeConstantAST(0);  // d/dx(c) = 0
            
        case ESHKOL_VAR:
            if (strcmp(expr->variable.id, var) == 0)
                return makeConstantAST(1);  // d/dx(x) = 1
            else
                return makeConstantAST(0);  // d/dx(y) = 0
                
        case ESHKOL_OP:
            return differentiateOperation(expr, var);
    }
}
```

### Product Rule Example

```cpp
// d/dx(f * g) = f'*g + f*g'
if (func_name == "*" && num_args == 2) {
    eshkol_ast_t* f = &args[0];
    eshkol_ast_t* g = &args[1];
    eshkol_ast_t* f_prime = buildSymbolicDerivative(f, var);
    eshkol_ast_t* g_prime = buildSymbolicDerivative(g, var);
    
    // Special case: x*x → 2*x (not x*1 + 1*x)
    if (isSameVariable(f, g, var)) {
        return makeList('*', 2, makeVar(var));  // '(* 2 x)
    }
    
    // General case: f'*g + f*g'
    eshkol_ast_t* term1 = makeList('*', f_prime, g);
    eshkol_ast_t* term2 = makeList('*', f, g_prime);
    return makeList('+', term1, term2);
}
```

### List Construction Helpers

```cpp
// Helper to build list AST nodes at compile-time
eshkol_ast_t* makeList(char op, eshkol_ast_t* arg1, eshkol_ast_t* arg2) {
    eshkol_ast_t* result = allocateAST();
    result->type = ESHKOL_CONS;
    // Build: (op arg1 arg2) as nested cons cells
    return result;
}

eshkol_ast_t* makeVar(const char* name) {
    eshkol_ast_t* result = allocateAST();
    result->type = ESHKOL_VAR;
    result->variable.id = strdup(name);
    return result;
}

eshkol_ast_t* makeConstantAST(int64_t value) {
    eshkol_ast_t* result = allocateAST();
    result->type = ESHKOL_INT64;
    result->int64_val = value;
    return result;
}
```

### Codegen Integration

```cpp
Value* codegenDiff(const eshkol_operations_t* op) {
    // Build symbolic derivative as AST
    eshkol_ast_t* symbolic_deriv = buildSymbolicDerivative(
        op->diff_op.expression, 
        op->diff_op.variable
    );
    
    // Generate runtime code for the symbolic expression
    // This creates a list structure at runtime
    Value* result = codegenAST(symbolic_deriv);
    
    // Result is now a cons cell list that can be displayed/evaluated
    return result;
}
```

## Display Integration

### Smart List Detection

Enhance `codegenDisplay()` to recognize symbolic math lists:

```cpp
// Check if list looks like symbolic expression
bool isSymbolicExpression(Value* list_ptr) {
    // Check if first element is math operator
    // Check if contains variable symbols
    return true/false;
}

// Format symbolic list to infix string
std::string formatSymbolic(Value* list_ptr) {
    // Walk list structure
    // Convert to infix notation
    // Return "2*x", "cos(x)", etc.
}
```

## Benefits of This Architecture

1. **Immediate**: Fixes display issue with proper symbolic output
2. **Scalable**: Foundation for full symbolic algebra
3. **Dual-mode**: Both symbolic (lists) and numeric (via evaluation)
4. **Clean**: Separates symbolic from numeric autodiff
5. **Future-proof**: Enables CAS features incrementally

## Migration Strategy

### Step 1: Minimal Working Version
- `diff` returns simple lists for basic cases
- Constants for trivial derivatives (d/dx(c) = 0)
- No simplification yet

### Step 2: Complete Derivative Rules
- All calculus rules return proper lists
- Product rule, chain rule, etc.

### Step 3: Display Formatting
- Pretty-print symbolic lists
- Infix notation converter

### Step 4: Symbolic Evaluation
- `eval-symbolic` operator
- Full variable binding support

## Comparison with Alternatives

| Feature | Lambda | String | S-Expression |
|---------|--------|--------|--------------|
| Symbolic Display | ❌ | ✅ | ✅ |
| Numeric Eval | ✅ | ⚠️ | ✅ |
| Manipulation | ❌ | ❌ | ✅ |
| Simplification | ❌ | ⚠️ | ✅ |
| Type Safety | ✅ | ❌ | ✅ |
| Future CAS | ❌ | ❌ | ✅ |
| Implementation Cost | Low | Medium | **Medium-High** |
| Long-term Value | Low | Low | **Very High** |

## Recommendation

**Implement S-Expression based symbolic differentiation** because:

1. Aligns with Lisp/Scheme philosophy (homoiconicity)
2. Provides true symbolic algebra foundation
3. Enables neuro-symbolic AI integration
4. Supports scientific computing workflows
5. Extensible to full CAS capabilities

This is the **only** architecture that delivers symbolic formulas that can be displayed, evaluated, AND manipulated - essential for your stated goals of scientific computing and neuro-symbolic AI.

## Next Steps

Should I proceed with:
1. Creating detailed implementation specification?
2. Building the S-expression symbolic derivative system?
3. Alternative approach?