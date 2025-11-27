# Symbolic Differentiation: Complete Implementation Guide

## Vision

Build a symbolic algebra system for `diff` operator that returns S-expression formulas, enabling:
- Human-readable symbolic output (`"2*x"`, `"cos(x)"`)
- Numerical evaluation at any point
- Symbolic manipulation (future CAS foundation)
- Neuro-symbolic AI integration

## Multi-Phase Implementation Roadmap

### Phase 1: AST-Based Symbolic Derivative Builder (Week 1)
**Goal**: Replace IR-based differentiate() with AST builder

### Phase 2: Runtime List Generation (Week 2)
**Goal**: Generate S-expression lists at runtime  

### Phase 3: Display Enhancement (Week 2-3)
**Goal**: Pretty-print symbolic expressions

### Phase 4: Symbolic Evaluation (Week 4+)
**Goal**: Evaluate S-expressions numerically

---

## PHASE 1: AST-Based Symbolic Derivative Builder

### 1.1 New AST Helper Functions

Add to `lib/core/ast.cpp`:

```cpp
// Memory management for symbolic AST nodes
static eshkol_ast_t* symbolic_ast_arena = nullptr;
static size_t symbolic_ast_count = 0;

eshkol_ast_t* eshkol_alloc_symbolic_ast() {
    eshkol_ast_t* node = (eshkol_ast_t*)malloc(sizeof(eshkol_ast_t));
    memset(node, 0, sizeof(eshkol_ast_t));
    return node;
}

// Helper: Create variable AST node
eshkol_ast_t* eshkol_make_var_ast(const char* name) {
    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_VAR;
    ast->variable.id = strdup(name);
    return ast;
}

// Helper: Create integer constant AST node
eshkol_ast_t* eshkol_make_int_ast(int64_t value) {
    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_INT64;
    ast->int64_val = value;
    return ast;
}

// Helper: Create double constant AST node
eshkol_ast_t* eshkol_make_double_ast(double value) {
    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_DOUBLE;
    ast->double_val = value;
    return ast;
}

// Helper: Create binary operation AST node (*, +, -, /)
eshkol_ast_t* eshkol_make_binary_op_ast(const char* op, 
                                         eshkol_ast_t* left, 
                                         eshkol_ast_t* right) {
    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_CALL_OP;
    
    // Create function AST for operator
    ast->operation.call_op.func = eshkol_make_var_ast(op);
    
    // Create arguments array
    ast->operation.call_op.variables = 
        (eshkol_ast_t*)malloc(2 * sizeof(eshkol_ast_t));
    ast->operation.call_op.variables[0] = *left;
    ast->operation.call_op.variables[1] = *right;
    ast->operation.call_op.num_vars = 2;
    
    return ast;
}

// Helper: Create unary function call AST node (sin, cos, exp, log)
eshkol_ast_t* eshkol_make_unary_call_ast(const char* func, eshkol_ast_t* arg) {
    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_CALL_OP;
    
    ast->operation.call_op.func = eshkol_make_var_ast(func);
    ast->operation.call_op.variables = 
        (eshkol_ast_t*)malloc(sizeof(eshkol_ast_t));
    ast->operation.call_op.variables[0] = *arg;
    ast->operation.call_op.num_vars = 1;
    
    return ast;
}
```

### 1.2 Core Symbolic Differentiation Function

Add to `lib/backend/llvm_codegen.cpp`:

```cpp
class EshkolLLVMCodeGen {
private:
    // NEW: Symbolic differentiation (AST → AST transformation)
    eshkol_ast_t* buildSymbolicDerivative(const eshkol_ast_t* expr, 
                                          const char* var) {
        if (!expr || !var) return eshkol_make_int_ast(0);
        
        switch (expr->type) {
            case ESHKOL_INT64:
            case ESHKOL_DOUBLE:
                // d/dx(c) = 0
                return eshkol_make_int_ast(0);
                
            case ESHKOL_VAR:
                // d/dx(x) = 1, d/dx(y) = 0
                if (strcmp(expr->variable.id, var) == 0)
                    return eshkol_make_int_ast(1);
                else
                    return eshkol_make_int_ast(0);
                    
            case ESHKOL_OP:
                return differentiateOperationSymbolic(&expr->operation, var);
                
            default:
                return eshkol_make_int_ast(0);
        }
    }
    
    eshkol_ast_t* differentiateOperationSymbolic(const eshkol_operations_t* op,
                                                  const char* var) {
        if (op->op != ESHKOL_CALL_OP) {
            return eshkol_make_int_ast(0);
        }
        
        const char* func_name = op->call_op.func->variable.id;
        
        // ADDITION RULE: d/dx(f + g) = f' + g'
        if (strcmp(func_name, "+") == 0 && op->call_op.num_vars >= 2) {
            eshkol_ast_t* result = buildSymbolicDerivative(
                &op->call_op.variables[0], var);
            
            for (uint64_t i = 1; i < op->call_op.num_vars; i++) {
                eshkol_ast_t* term = buildSymbolicDerivative(
                    &op->call_op.variables[i], var);
                result = eshkol_make_binary_op_ast("+", result, term);
            }
            return result;
        }
        
        // SUBTRACTION RULE: d/dx(f - g) = f' - g'
        if (strcmp(func_name, "-") == 0 && op->call_op.num_vars == 2) {
            eshkol_ast_t* f_prime = buildSymbolicDerivative(
                &op->call_op.variables[0], var);
            eshkol_ast_t* g_prime = buildSymbolicDerivative(
                &op->call_op.variables[1], var);
            return eshkol_make_binary_op_ast("-", f_prime, g_prime);
        }
        
        // PRODUCT RULE: d/dx(f * g) = f'*g + f*g'
        if (strcmp(func_name, "*") == 0 && op->call_op.num_vars == 2) {
            const eshkol_ast_t* f = &op->call_op.variables[0];
            const eshkol_ast_t* g = &op->call_op.variables[1];
            
            // SPECIAL CASE: d/dx(x * x) = 2*x (simplified)
            if (f->type == ESHKOL_VAR && g->type == ESHKOL_VAR &&
                strcmp(f->variable.id, var) == 0 &&
                strcmp(g->variable.id, var) == 0) {
                // Return '(* 2 x) as S-expression
                return eshkol_make_binary_op_ast("*",
                    eshkol_make_int_ast(2),
                    eshkol_make_var_ast(var));
            }
            
            // SPECIAL CASE: d/dx(c * x) = c (constant * variable)
            if (isConstant(f) && isVariable(g, var)) {
                // Copy constant f
                return copyAST(f);
            }
            if (isVariable(f, var) && isConstant(g)) {
                // Copy constant g
                return copyAST(g);
            }
            
            // GENERAL CASE: f'*g + f*g'
            eshkol_ast_t* f_prime = buildSymbolicDerivative(f, var);
            eshkol_ast_t* g_prime = buildSymbolicDerivative(g, var);
            
            eshkol_ast_t* term1 = eshkol_make_binary_op_ast("*", f_prime, 
                                                            copyAST(g));
            eshkol_ast_t* term2 = eshkol_make_binary_op_ast("*", copyAST(f),
                                                            g_prime);
            return eshkol_make_binary_op_ast("+", term1, term2);
        }
        
        // CHAIN RULE: d/dx(sin(f)) = cos(f) * f'
        if (strcmp(func_name, "sin") == 0 && op->call_op.num_vars == 1) {
            const eshkol_ast_t* f = &op->call_op.variables[0];
            eshkol_ast_t* f_prime = buildSymbolicDerivative(f, var);
            eshkol_ast_t* cos_f = eshkol_make_unary_call_ast("cos", copyAST(f));
            
            // Special case: d/dx(sin(x)) = cos(x) (not cos(x)*1)
            if (isConstantOne(f_prime)) {
                return cos_f;
            }
            
            return eshkol_make_binary_op_ast("*", cos_f, f_prime);
        }
        
        // Similar for cos, exp, log, pow, sqrt...
        
        return eshkol_make_int_ast(0);
    }
    
    // Helper: Check if AST is a constant (number)
    bool isConstant(const eshkol_ast_t* ast) {
        return ast && (ast->type == ESHKOL_INT64 || ast->type == ESHKOL_DOUBLE);
    }
    
    // Helper: Check if AST is specific variable
    bool isVariable(const eshkol_ast_t* ast, const char* var_name) {
        return ast && ast->type == ESHKOL_VAR && 
               strcmp(ast->variable.id, var_name) == 0;
    }
    
    // Helper: Check if constant equals 1
    bool isConstantOne(const eshkol_ast_t* ast) {
        if (ast->type == ESHKOL_INT64) return ast->int64_val == 1;
        if (ast->type == ESHKOL_DOUBLE) return ast->double_val == 1.0;
        return false;
    }
    
    // Helper: Deep copy AST node
    eshkol_ast_t* copyAST(const eshkol_ast_t* ast) {
        if (!ast) return nullptr;
        eshkol_ast_t* copy = eshkol_alloc_symbolic_ast();
        memcpy(copy, ast, sizeof(eshkol_ast_t));
        // Deep copy string fields if needed
        if (ast->type == ESHKOL_VAR && ast->variable.id) {
            copy->variable.id = strdup(ast->variable.id);
        }
        return copy;
    }
};
```

### 1.3 Modified codegenDiff() 

```cpp
Value* codegenDiff(const eshkol_operations_t* op) {
    if (!op->diff_op.expression || !op->diff_op.variable) {
        eshkol_error("Invalid diff operation");
        return nullptr;
    }
    
    const char* var = op->diff_op.variable;
    eshkol_info("Building symbolic derivative S-expression for %s", var);
    
    // STEP 1: Build symbolic derivative as AST (compile-time)
    eshkol_ast_t* symbolic_deriv = buildSymbolicDerivative(
        op->diff_op.expression, 
        var
    );
    
    if (!symbolic_deriv) {
        eshkol_error("Failed to build symbolic derivative");
        return packInt64ToTaggedValue(ConstantInt::get(Type::getInt64Ty(*context), 0), true);
    }
    
    // STEP 2: Generate runtime code that constructs the S-expression
    // This uses existing codegenAST() to build list structures at runtime
    Value* result = codegenAST(symbolic_deriv);
    
    // STEP 3: Return the S-expression (list or constant)
    eshkol_info("Generated symbolic derivative S-expression");
    
    return result;
}
```

---

## PHASE 2: Runtime S-Expression Construction

### 2.1 Challenge

AST nodes generated in Phase 1 are **compile-time C++ objects**. We need them as **runtime list structures** that can be displayed/evaluated.

### 2.2 Solution: Quote Mechanism

Implement `quote` functionality to convert AST to runtime lists:

```cpp
Value* codegenQuotedAST(const eshkol_ast_t* ast) {
    if (!ast) return packInt64ToTaggedValue(
        ConstantInt::get(Type::getInt64Ty(*context), 0), true);
    
    switch (ast->type) {
        case ESHKOL_INT64:
            // Return integer directly
            return packInt64ToTaggedValue(
                ConstantInt::get(Type::getInt64Ty(*context), ast->int64_val), 
                true);
            
        case ESHKOL_VAR:
            // Return symbol (quoted variable name as string)
            return codegenString(ast->variable.id);
            
        case ESHKOL_OP:
            if (ast->operation.op == ESHKOL_CALL_OP) {
                // Build list: (op arg1 arg2 ...)
                return codegenQuotedList(&ast->operation.call_op);
            }
            break;
            
        default:
            return packInt64ToTaggedValue(
                ConstantInt::get(Type::getInt64Ty(*context), 0), true);
    }
}

Value* codegenQuotedList(const eshkol_call_op_t* call_op) {
    // Build list from right to left: (op arg1 arg2)
    
    // Start with empty list
    TypedValue result(ConstantInt::get(Type::getInt64Ty(*context), 0), 
                     ESHKOL_VALUE_NULL);
    
    // Add arguments in reverse
    for (int64_t i = call_op->num_vars - 1; i >= 0; i--) {
        TypedValue elem = codegenQuotedElement(&call_op->variables[i]);
        Value* cons_cell = codegenTaggedArenaConsCell(elem, result);
        result = TypedValue(cons_cell, ESHKOL_VALUE_CONS_PTR, true);
    }
    
    // Add operator symbol at front
    TypedValue op_symbol(codegenString(call_op->func->variable.id),
                        ESHKOL_VALUE_CONS_PTR, true);
    Value* final_list = codegenTaggedArenaConsCell(op_symbol, result);
    
    return final_list;
}
```

### 2.3 Updated codegenDiff()

```cpp
Value* codegenDiff(const eshkol_operations_t* op) {
    // Build symbolic derivative AST
    eshkol_ast_t* symbolic_deriv = buildSymbolicDerivative(
        op->diff_op.expression, 
        op->diff_op.variable
    );
    
    // Convert AST to runtime S-expression (quoted list)
    Value* sexpr_list = codegenQuotedAST(symbolic_deriv);
    
    // Clean up temporary AST
    eshkol_ast_clean(symbolic_deriv);
    
    return sexpr_list;
}
```

---

## PHASE 3: Display Enhancement

### 3.1 Symbolic Expression Detection

Add to `codegenDisplay()`:

```cpp
Value* codegenDisplay(const eshkol_operations_t* op) {
    Value* arg = codegenAST(&op->call_op.variables[0]);
    
    // NEW: Check if arg is a symbolic expression list
    if (isSymbolicExpression(arg)) {
        return displaySymbolicExpression(arg);
    }
    
    // Existing display logic...
}

bool isSymbolicExpression(Value* val) {
    // Detect if list contains mathematical operators and variables
    // Pattern: (op arg1 arg2...) where op is +, -, *, /, sin, cos, etc.
    
    if (!val || val->getType() != Type::getInt64Ty(*context)) {
        return false;
    }
    
    // Check first element (car) is operator symbol
    // This requires runtime inspection
    return true; // Simplified for now
}
```

### 3.2 Infix Formatter

```cpp
Value* displaySymbolicExpression(Value* sexpr) {
    // Convert S-expression list to infix string
    // Example: '(* 2 x) → "2*x"
    //          '(cos x) → "cos(x)"
    //          '(+ (* 2 x) 3) → "(2*x + 3)"
    
    Function* printf_func = function_table["printf"];
    
    // Extract operator
    Value* car = extractCarAsTaggedValue(sexpr);
    // Extract arguments
    Value* cdr = extractCdrAsTaggedValue(sexpr);
    
    // Format based on operator type
    // This requires complex runtime logic - see Phase 3 details
    
    return ConstantInt::get(Type::getInt32Ty(*context), 0);
}
```

---

## PHASE 4: Symbolic Evaluation

### 4.1 New Operator: eval-symbolic

```scheme
(define f-prime (diff (* x x) x))  ; f-prime = '(* 2 x)
(eval-symbolic f-prime '((x . 5))) ; → 10
```

### 4.2 Implementation

```cpp
Value* codegenEvalSymbolic(const eshkol_operations_t* op) {
    // Args: (eval-symbolic expr bindings)
    // expr: S-expression like '(* 2 x)
    // bindings: association list '((x . 5) (y . 3))
    
    Value* expr = codegenAST(&op->call_op.variables[0]);
    Value* bindings = codegenAST(&op->call_op.variables[1]);
    
    // Walk expression tree, substitute variables, evaluate
    return evaluateSymbolicExpression(expr, bindings);
}
```

---

## Technical Details: Derivative Rules

### Complete Rule Set

```cpp
// 1. CONSTANT RULE
d/dx(c) → 0

// 2. VARIABLE RULE  
d/dx(x) → 1
d/dx(y) → 0  (y ≠ x)

// 3. SUM RULE
d/dx(f + g) → f' + g'

// 4. DIFFERENCE RULE
d/dx(f - g) → f' - g'

// 5. PRODUCT RULE (with simplifications)
d/dx(c * x) → c           // Constant times variable
d/dx(x * x) → 2*x         // Same variable squared
d/dx(f * g) → f'*g + f*g' // General case

// 6. QUOTIENT RULE
d/dx(f / g) → (f'*g - f*g') / g²

// 7. CHAIN RULE (trig)
d/dx(sin(f)) → cos(f) * f'
d/dx(cos(f)) → -sin(f) * f'

// 8. CHAIN RULE (exp/log)
d/dx(exp(f)) → exp(f) * f'
d/dx(log(f)) → f' / f

// 9. POWER RULE
d/dx(f^n) → n * f^(n-1) * f'  (constant n)

// 10. SQRT RULE
d/dx(sqrt(f)) → f' / (2*sqrt(f))
```

### Simplification Rules (Phase 1)

```cpp
// Aggressive simplification during derivative construction

// Multiply by 1: eliminate
if (isConstantOne(left)) return right;
if (isConstantOne(right)) return left;

// Multiply by 0: return 0
if (isConstantZero(left) || isConstantZero(right)) 
    return eshkol_make_int_ast(0);

// Add 0: eliminate
if (isConstantZero(left)) return right;
if (isConstantZero(right)) return left;

// Same variable: x + x → 2*x
if (isSameAST(left, right)) {
    return eshkol_make_binary_op_ast("*",
        eshkol_make_int_ast(2), copyAST(left));
}
```

---

## Integration with Existing System

### Coexistence Strategy

**Numeric Autodiff** (gradient, jacobian, hessian):
- Keep current LLVM IR-based approach
- Uses dual numbers and computational graphs
- Optimized for numerical computation

**Symbolic Diff** (`diff` operator):
- New S-expression based approach
- Returns symbolic formulas as lists
- Can be evaluated numerically via new `eval-symbolic`

### Unified Workflow

```scheme
;; Symbolic: Get formula
(define f-prime (diff (* x x) x))  
;; → '(* 2 x)

(display f-prime)
;; → "2*x" (via enhanced display)

;; Numeric: Evaluate at point
(eval-symbolic f-prime '((x . 5)))
;; → 10

;; Or use numeric autodiff directly
(derivative (lambda (x) (* x x)) 5)
;; → 10.0
```

---

## Testing Strategy

### Phase 1 Tests

```scheme
;; Test 1: Simple constant multiplication
(diff (* 2 x) x)
;; Expected: 2 (constant, not list)
;; Actual S-expr: Would be 2 directly

;; Test 2: Variable squared
(diff (* x x) x)
;; Expected S-expr: '(* 2 x)
;; Display: "(* 2 x)" initially, "2*x" after Phase 3

;; Test 3: Chain rule
(diff (sin (* 2 x)) x)
;; Expected S-expr: '(* (cos (* 2 x)) 2)
;; Display: "(* (cos (* 2 x)) 2)"
```

### Validation Approach

1. **Symbolic Correctness**: Verify S-expression structure matches mathematical formula
2. **Evaluation Correctness**: Once eval-symbolic works, compare with numeric autodiff
3. **Display Correctness**: Check formatted output is readable

---

## Memory Management

### AST Lifecycle

```
Compile-Time:
  buildSymbolicDerivative() → temp AST nodes (malloc)
  ↓
  codegenQuotedAST() → converts to runtime lists
  ↓
  Clean up temp AST nodes (free)

Runtime:
  S-expression lists live in arena memory
  ↓
  Garbage collected with arena cleanup
```

### Arena Integration

Quoted lists use existing arena-based cons cells:
- No special memory management needed
- Lists persist through program execution
- Cleaned up with global arena

---

## Implementation Timeline

### Week 1: Phase 1 Foundation
- Day 1-2: AST helper functions
- Day 3-4: buildSymbolicDerivative() with all rules
- Day 5: Integration and basic tests

### Week 2: Phase 2 Runtime Lists
- Day 1-2: codegenQuotedAST() implementation
- Day 3-4: Testing and debugging
- Day 5: Performance optimization

### Week 3: Phase 3 Display
- Day 1-3: Symbolic expression formatter
- Day 4-5: Integration and UI polish

### Week 4+: Phase 4 Evaluation
- As needed for neuro-symbolic features
- Not blocking for basic symbolic diff

---

## Success Criteria

### Phase 1 Complete When:
- ✅ `diff` returns S-expressions for all calculus rules
- ✅ Simple cases return constants (0, 1, etc.)
- ✅ Complex cases return list structures
- ✅ Product rule works: `d/dx(x*x)` →  `'(* 2 x)`
- ✅ Chain rule works: `d/dx(sin(x))` → `'(cos x)`

### Phase 2 Complete When:
- ✅ S-expressions are proper runtime lists
- ✅ Can be passed to other functions
- ✅ Display shows list structure (even if not pretty)

### Phase 3 Complete When:
- ✅ Display shows `"2*x"` instead of `"(* 2 x)"`
- ✅ All mathematical notation is readable
- ✅ Nested expressions format correctly

### Phase 4 Complete When:
- ✅ `eval-symbolic` evaluates formulas numerically
- ✅ Variable binding works correctly
- ✅ Results match numeric autodiff

---

## Risk Mitigation

### Compatibility Risk
**Issue**: Breaking existing numerical autodiff  
**Mitigation**: Keep separate code paths, no shared logic

### Complexity Risk
**Issue**: S-expression manipulation is complex  
**Mitigation**: Incremental phases, each independently useful

### Performance Risk
**Issue**: Runtime list construction overhead  
**Mitigation**: Lazy evaluation, caching, optimization in later phases

---

## Future Extensions (Beyond Initial Implementation)

### Symbolic Simplification

```scheme
(simplify '(+ (* 1 x) (* x 1)))  →  '(* 2 x)
(simplify '(* x (/ 1 x)))        →  1
```

### Symbolic Integration

```scheme
(integrate '(* 2 x) 'x)  →  '(* x x)  ; x²
```

### Equation Solving

```scheme
(solve '(= (* 2 x) 10) 'x)  →  5
```

### Pattern Matching

```scheme
(match-pattern '(* 2 x) '(* ?c ?v))  
→  '((c . 2) (v . x))
```

### Neural-Symbolic Bridge

```scheme
;; Symbolic reasoning
(define grad-formula (diff loss-function params))

;; Compile to neural network update rule
(compile-to-network grad-formula learning-rate)
```

---

## Conclusion

This S-expression based architecture provides:

1. **Immediate Value**: Readable symbolic derivatives
2. **Mathematical Correctness**: True symbolic algebra
3. **Future Proof**: CAS and neuro-symbolic AI foundation
4. **Lisp Philosophy**: Homoiconic, data-driven design
5. **Incremental Path**: Each phase delivers standalone value

**Recommendation**: Proceed with Phase 1 implementation now, as it directly addresses the current `diff` display issue while laying groundwork for advanced symbolic capabilities essential for AI research.
