# Eshkol Closure Implementation Plan

## Executive Summary

Implement proper lexical closures to support nested functions that capture parent scope variables, enabling natural neural network code patterns without LLVM cross-function reference errors.

## Current Problem

**Error:** "Referring to an instruction in another function!"

```scheme
(define (outer)
  (define x 2.0)          ; Parent scope
  (define (inner w)       ; Nested function
    (* w x))              ; ERROR: Accesses parent's x
  (derivative inner 1.0))
```

**Root Cause:** LLVM rejects instructions in one function referencing values from another function. Our current implementation creates nested LLVM functions with direct variable references, violating LLVM's invariants.

## Solution Architecture

### 1. Closure Environment Structure

**CRITICAL:** Eshkol uses **arena-based memory**, NOT garbage collection!

Environments are allocated in the arena and automatically cleaned up:

```c
// Runtime closure environment (arena-allocated)
typedef struct eshkol_closure_env {
    size_t num_captures;        // Number of captured variables
    eshkol_tagged_value_t captures[];  // Flexible array of captured values
} eshkol_closure_env_t;

// This structure lives in the arena - no manual free needed!
// When arena scope ends, all environments are automatically reclaimed.
```

**Key Insight:** We follow the same pattern as `arena_tagged_cons_cell_t` - arena allocation means automatic cleanup with function scope.

### 2. AST Representation

Extend the AST to track closure information:

```cpp
// In ast.cpp
struct ClosureInfo {
    std::vector<std::string> captured_vars;  // Names of captured variables
    std::map<std::string, int> capture_indices; // Index in environment array
    ASTNode* parent_scope;  // Reference to parent function/scope
};

// Add to ASTNode
struct ASTNode {
    // ... existing fields ...
    ClosureInfo* closure_info;  // NULL for non-closures
};
```

### 3. Parser Phase: Capture Analysis

Modify [`parser.cpp`](lib/frontend/parser.cpp) to detect and record captured variables:

```cpp
// New function in parser.cpp
ClosureInfo* analyzeCapturedVariables(ASTNode* lambda_node, 
                                      ASTNode* parent_scope) {
    ClosureInfo* info = new ClosureInfo();
    info->parent_scope = parent_scope;
    
    // 1. Collect all variable references in lambda body
    std::set<std::string> referenced_vars;
    collectVariableReferences(lambda_node->body, referenced_vars);
    
    // 2. Determine which are local vs captured
    for (const auto& var : referenced_vars) {
        if (!isLocalToLambda(lambda_node, var) && 
            isDefinedInParentScope(parent_scope, var)) {
            // This is a captured variable
            int index = info->captured_vars.size();
            info->captured_vars.push_back(var);
            info->capture_indices[var] = index;
        }
    }
    
    return info;
}
```

### 4. Codegen Phase: Environment Creation

Modify [`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) to generate closure code:

#### 4.1 Function Signature Modification

**Before:**
```llvm
define tagged_value @inner(tagged_value %w)
```

**After (with environment):**
```llvm
define tagged_value @inner(ptr %env, tagged_value %w)
```

All functions with closures get an additional first parameter: the environment pointer.

#### 4.2 Environment Allocation

```cpp
// In codegenLambda or codegenDefine for nested functions
llvm::Value* createClosureEnvironment(const ClosureInfo* info) {
    if (!info || info->captured_vars.empty()) {
        return nullptr;  // No environment needed
    }
    
    // Calculate size: sizeof(header) + num_captures * sizeof(tagged_value)
    size_t header_size = sizeof(int) + sizeof(int);  // ref_count + num_captures
    size_t data_size = info->captured_vars.size() * sizeof(eshkol_tagged_value);
    size_t total_size = header_size + data_size;
    
    // Allocate environment
    llvm::Value* env_ptr = Builder.CreateCall(
        module->getFunction("malloc"),
        llvm::ConstantInt::get(Context, APInt(64, total_size))
    );
    
    // Initialize header
    llvm::Value* ref_count_ptr = Builder.CreateStructGEP(env_type, env_ptr, 0);
    Builder.CreateStore(llvm::ConstantInt::get(Context, APInt(32, 1)), ref_count_ptr);
    
    llvm::Value* num_captures_ptr = Builder.CreateStructGEP(env_type, env_ptr, 1);
    Builder.CreateStore(
        llvm::ConstantInt::get(Context, APInt(32, info->captured_vars.size())),
        num_captures_ptr
    );
    
    // Copy captured variables into environment
    for (size_t i = 0; i < info->captured_vars.size(); i++) {
        const std::string& var_name = info->captured_vars[i];
        llvm::Value* var_value = lookupVariable(var_name);  // Current value
        
        llvm::Value* capture_ptr = Builder.CreateGEP(
            tagged_value_type,
            Builder.CreateStructGEP(env_type, env_ptr, 2),  // captures array
            llvm::ConstantInt::get(Context, APInt(64, i))
        );
        Builder.CreateStore(var_value, capture_ptr);
    }
    
    return env_ptr;
}
```

#### 4.3 Variable Access Transformation

**When generating code for a captured variable:**

```cpp
llvm::Value* codegenVariableAccess(const std::string& var_name, 
                                   const ClosureInfo* closure_info,
                                   llvm::Value* env_param) {
    if (closure_info && closure_info->capture_indices.count(var_name)) {
        // This is a captured variable - load from environment
        int index = closure_info->capture_indices.at(var_name);
        
        llvm::Value* capture_ptr = Builder.CreateGEP(
            tagged_value_type,
            Builder.CreateStructGEP(env_type, env_param, 2),  // captures array
            llvm::ConstantInt::get(Context, APInt(64, index))
        );
        
        return Builder.CreateLoad(tagged_value_type, capture_ptr);
    } else {
        // Normal local variable
        return lookupVariable(var_name);
    }
}
```

#### 4.4 Function Call Transformation

**When calling a closure:**

```cpp
llvm::Value* codegenClosureCall(llvm::Function* func, 
                                llvm::Value* env_ptr,
                                const std::vector<llvm::Value*>& args) {
    std::vector<llvm::Value*> call_args;
    
    if (env_ptr) {
        // Pass environment as first argument
        call_args.push_back(env_ptr);
    } else {
        // No environment - pass null
        call_args.push_back(llvm::ConstantPointerNull::get(env_ptr_type));
    }
    
    // Add actual function arguments
    call_args.insert(call_args.end(), args.begin(), args.end());
    
    return Builder.CreateCall(func, call_args);
}
```

### 5. Integration with Autodiff

**Critical:** The derivative operator must handle closures:

```cpp
llvm::Value* codegenDerivative(ASTNode* lambda_node, llvm::Value* point) {
    // Get the lambda's closure environment
    llvm::Value* env_ptr = lambda_node->closure_info 
        ? getCurrentEnvironment() 
        : nullptr;
    
    // Create AD graph with environment context
    ADGraph* graph = createADGraph(lambda_node, env_ptr);
    
    // When evaluating the graph, pass environment to function calls
    // ...
}
```

## Implementation Phases

### Phase 1: AST Extensions (2 hours)
- [ ] Add `ClosureInfo` struct to AST
- [ ] Add `closure_info` field to `ASTNode`
- [ ] Add helper methods for closure analysis
- [ ] Test: Parse nested functions and detect structure

### Phase 2: Parser Analysis (2-3 hours)
- [ ] Implement `analyzeCapturedVariables()`
- [ ] Implement `collectVariableReferences()`
- [ ] Implement `isLocalToLambda()` and `isDefinedInParentScope()`
- [ ] Integrate analysis into lambda/define parsing
- [ ] Test: Verify captured variables are correctly identified

### Phase 3: LLVM Runtime Support (2 hours)
- [ ] Define `eshkol_closure_env` type in LLVM IR
- [ ] Create environment allocation helper functions
- [ ] Create environment access helper functions
- [ ] Test: Manually create and access environment

### Phase 4: Codegen Transformation (3-4 hours)
- [ ] Modify function signatures to accept environment parameter
- [ ] Implement environment creation in `codegenLambda`
- [ ] Transform variable access to use environment when needed
- [ ] Transform function calls to pass environment
- [ ] Test: Simple closure without autodiff

### Phase 5: Autodiff Integration (1-2 hours)
- [ ] Update autodiff to handle environment context
- [ ] Ensure gradient computation works with closures
- [ ] Test: Neural network tests with nested functions

### Phase 6: Testing & Validation (1 hour)
- [ ] Test all 4 neural network demos
- [ ] Test edge cases (multiple captures, nested closures)
- [ ] Performance testing
- [ ] Documentation updates

## Expected Timeline

**Total: 11-14 hours**

- Phase 1: 2 hours
- Phase 2: 2-3 hours  
- Phase 3: 2 hours
- Phase 4: 3-4 hours
- Phase 5: 1-2 hours
- Phase 6: 1 hour

## Code Examples

### Example 1: Simple Closure

**Input:**
```scheme
(define (make-adder x)
  (lambda (y) (+ x y)))

(define add5 (make-adder 5.0))
(add5 3.0)  ; Returns 8.0
```

**Generated LLVM (simplified):**
```llvm
define ptr @make_adder(tagged_value %x) {
    ; Allocate environment
    %env = call ptr @malloc(i64 24)
    
    ; Store captured x
    %x_ptr = getelementptr %closure_env, ptr %env, i32 0, i32 2, i32 0
    store tagged_value %x, ptr %x_ptr
    
    ; Return closure (function + environment)
    %closure = call ptr @create_closure(ptr @lambda_1, ptr %env)
    ret ptr %closure
}

define tagged_value @lambda_1(ptr %env, tagged_value %y) {
    ; Load captured x from environment
    %x_ptr = getelementptr %closure_env, ptr %env, i32 0, i32 2, i32 0
    %x = load tagged_value, ptr %x_ptr
    
    ; Compute x + y
    %result = call tagged_value @add(tagged_value %x, tagged_value %y)
    ret tagged_value %result
}
```

### Example 2: Closure with Autodiff

**Input:**
```scheme
(define (demo4)
  (define x 2.0)
  (define target 10.0)
  (define (loss w)
    (define pred (* w x))
    (define diff (- pred target))
    (* diff diff))
  (derivative loss 1.0))
```

**Generated LLVM (simplified):**
```llvm
define tagged_value @demo4() {
    %x = ...
    %target = ...
    
    ; Create environment capturing x and target
    %env = call ptr @malloc(i64 32)
    ; Store x at index 0
    %x_ptr = getelementptr %closure_env, ptr %env, i32 0, i32 2, i32 0
    store tagged_value %x, ptr %x_ptr
    ; Store target at index 1
    %target_ptr = getelementptr %closure_env, ptr %env, i32 0, i32 2, i32 1
    store tagged_value %target, ptr %target_ptr
    
    ; Call derivative with closure
    %result = call tagged_value @derivative(ptr @loss, ptr %env, tagged_value 1.0)
    ret tagged_value %result
}

define tagged_value @loss(ptr %env, tagged_value %w) {
    ; Load captured x
    %x_ptr = getelementptr %closure_env, ptr %env, i32 0, i32 2, i32 0
    %x = load tagged_value, ptr %x_ptr
    
    ; Load captured target
    %target_ptr = getelementptr %closure_env, ptr %env, i32 0, i32 2, i32 1
    %target = load tagged_value, ptr %target_ptr
    
    ; Compute loss
    %pred = call tagged_value @mul(tagged_value %w, tagged_value %x)
    %diff = call tagged_value @sub(tagged_value %pred, tagged_value %target)
    %result = call tagged_value @mul(tagged_value %diff, tagged_value %diff)
    ret tagged_value %result
}
```

## Success Criteria

1. All 4 neural network tests compile and run successfully
2. Nested functions can access parent scope variables
3. Autodiff works with closures
4. No LLVM verification errors
5. Performance degradation < 10% for non-closure code

## Risks & Mitigations

### Risk 1: Autodiff Complexity
**Mitigation:** Start with simple closures, add autodiff support incrementally

### Risk 2: Memory Management
**Mitigation:** Simple ref counting initially, proper GC in future version

### Risk 3: Performance
**Mitigation:** Only allocate environments when needed, optimize common cases

### Risk 4: Debugging Difficulty  
**Mitigation:** Add extensive logging, create minimal test cases

## Future Enhancements

1. **Garbage Collection:** Proper ref counting or tracing GC
2. **Optimization:** Inline closures that don't escape
3. **Multiple Levels:** Support arbitrary nesting depth
4. **Mutable Captures:** Support captured variables that can be modified
5. **Partial Application:** Support currying and partial function application

## References

- LLVM Language Reference: https://llvm.org/docs/LangRef.html
- Closure Implementation in Scheme: R5RS Section 6.4
- "Compiling with Continuations" by Andrew Appel
- "Modern Compiler Implementation in ML" by Andrew Appel