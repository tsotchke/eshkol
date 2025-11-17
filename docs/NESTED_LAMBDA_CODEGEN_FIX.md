# Task: Fix Codegen for Nested Lambda (Immediate Lambda Application)

## Status: Parser Fixed, Codegen Broken

**Priority: HIGH** - Essential for functional programming idioms in Eshkol

## Problem Statement

Eshkol's parser now correctly handles nested lambda expressions like `((lambda (x) (* x x)) 5.0)` (immediate lambda application), but the LLVM codegen does not handle them correctly.

**Current Status:**
- ✅ Parser accepts nested lambda syntax
- ❌ Codegen fails (returns no value or segfaults)
- ❌ Test 8+ in phase2_forward_test.esk blocked

## Examples

### Simple Nested Lambda (Currently Broken)
```scheme
; This should return 9.0
(define result ((lambda (u) (* u u)) 3.0))
(display result)  ; Currently displays nothing
```

### Nested Lambda in Derivative Context (Currently Broken)
```scheme
; f(x) = (2x+1)³ computed via intermediate variable
(derivative (lambda (x) 
    ((lambda (u) (* u (* u u))) (+ (* 2.0 x) 1.0))) 1.0)
; Expected: 54.0 (currently broken)
```

### Why This Matters

Immediate lambda application is a fundamental Scheme/Lisp idiom used for:
1. **Local variable binding without `let`**: `((lambda (x y) (+ x y)) 3 4)`
2. **Encapsulation**: Creating local scopes
3. **Functional composition**: Higher-order function patterns
4. **Educational clarity**: Direct lambda semantics

## Technical Analysis

### Parser Changes (Completed)

File: [`lib/frontend/parser.cpp`](../lib/frontend/parser.cpp:278)

Added handling for when first element of a list is `TOKEN_LPAREN`:

```cpp
// Handle case where first element is a lambda or other expression
if (token.type == TOKEN_LPAREN) {
    eshkol_ast_t func_expr = parse_list(tokenizer);
    // ... parse arguments ...
    // Set up as CALL_OP with func_expr as the function
    ast.operation.op = ESHKOL_CALL_OP;
    ast.operation.call_op.func = new eshkol_ast_t;
    *ast.operation.call_op.func = func_expr;
    // ... set up arguments ...
}
```

### Codegen Issue (Needs Fix)

File: [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp)

The `codegenCall` function expects `call_op.func` to be:
- A `ESHKOL_VAR` (function name to look up)
- Not an `ESHKOL_OP` with `LAMBDA_OP` (anonymous lambda)

**Root Cause:** When `call_op.func` is a lambda expression, the codegen doesn't:
1. Generate the lambda function
2. Get its function pointer
3. Call it with the provided arguments

## Implementation Plan

### Step 1: Modify `codegenCall` to Handle Lambda Functions

In [`lib/backend/llvm_codegen.cpp`](../lib/backend/llvm_codegen.cpp), update `codegenCall`:

```cpp
Value* EshkolLLVMCodeGen::codegenCall(eshkol_ast_t *ast) {
    // ... existing code ...
    
    // NEW: Check if func is a lambda expression
    if (ast->operation.call_op.func->type == ESHKOL_OP &&
        ast->operation.call_op.func->operation.op == ESHKOL_LAMBDA_OP) {
        
        // Generate the lambda as an anonymous function
        Value* lambda_func = codegenLambda(ast->operation.call_op.func);
        
        // Cast to function pointer and call immediately
        // ... implement immediate invocation ...
        
        return /* result */;
    }
    
    // ... rest of existing code ...
}
```

### Step 2: Test Cases

Create comprehensive tests in `tests/nested_lambda_test.esk`:

```scheme
; Test 1: Simple immediate application
(display "Test 1: ")
(display ((lambda (x) (* x x)) 5.0))  ; Expected: 25.0
(newline)

; Test 2: Multiple arguments
(display "Test 2: ")
(display ((lambda (x y) (+ x y)) 3.0 4.0))  ; Expected: 7.0
(newline)

; Test 3: Nested immediate applications
(display "Test 3: ")
(display ((lambda (x) ((lambda (y) (+ x y)) 2.0)) 3.0))  ; Expected: 5.0
(newline)

; Test 4: With derivative (the Phase 2 blocker)
(display "Test 4: ")
(define result (derivative (lambda (x) 
    ((lambda (u) (* u (* u u))) (+ (* 2.0 x) 1.0))) 1.0))
(display result)  ; Expected: 54.0
(newline)
```

### Step 3: Alternative Approach - Let Bindings

If immediate lambda application proves too complex, consider implementing `let` bindings:

```scheme
; Instead of: ((lambda (u) (* u (* u u))) (+ (* 2.0 x) 1.0))
; Use: (let ((u (+ (* 2.0 x) 1.0))) (* u (* u u)))
```

This would require:
1. Parser support for `let` syntax
2. Codegen for `let` bindings (simpler than nested lambdas)
3. Updating test cases

## Success Criteria

- ✅ Simple nested lambda: `((lambda (x) (* x x)) 5.0)` returns `25.0`
- ✅ Multiple arguments: `((lambda (x y) (+ x y)) 3 4)` returns `7.0`
- ✅ Nested applications work without segfaults
- ✅ Test 8+ in phase2_forward_test.esk pass
- ✅ All existing tests still pass

## Files to Modify

1. **lib/backend/llvm_codegen.cpp** - Main codegen fix
2. **tests/nested_lambda_test.esk** - Comprehensive test suite
3. **tests/autodiff/phase2_forward_test.esk** - Currently blocked by this issue

## References

- Parser fix: [`lib/frontend/parser.cpp:278-335`](../lib/frontend/parser.cpp:278)
- Test case: [`tests/autodiff/test_nested_lambda.esk`](../tests/autodiff/test_nested_lambda.esk)
- Phase 2 test suite: [`tests/autodiff/phase2_forward_test.esk`](../tests/autodiff/phase2_forward_test.esk)

## Estimated Effort

**2-4 hours** - Medium complexity
- Understand current closure/lambda codegen
- Implement immediate invocation logic  
- Handle argument passing correctly
- Test thoroughly with dual numbers

## Next Steps

1. Study `codegenLambda` implementation
2. Determine how to invoke lambda immediately after creation
3. Implement and test simple case first
4. Extend to nested and dual number cases
5. Update all affected tests