# Closure Implementation Technical Specification

## Current State Analysis

### ✅ GOOD NEWS: Infrastructure Already Exists!

The AST [`lambda_op`](inc/eshkol/eshkol.h:286) structure already has closure support:

```c
struct {
    struct eshkol_ast *parameters;
    uint64_t num_params;
    struct eshkol_ast *body;
    struct eshkol_ast *captured_vars;  // ✓ ALREADY EXISTS!
    uint64_t num_captured;             // ✓ ALREADY EXISTS!
} lambda_op;
```

### ❌ PROBLEM: Missing Implementation

The fields exist but aren't properly utilized:
1. Parser doesn't populate `captured_vars` when detecting nested scope access
2. Codegen doesn't generate environment passing code
3. LLVM functions don't accept environment parameter
4. Variable lookups don't check environment

## Root Cause Deep Dive

**The Error:**
```
LLVM module verification failed: Referring to an instruction in another function!
  %0 = load double, ptr %x, align 8
```

**Why It Happens:**

```scheme
(define (outer)
  (define x 2.0)          ; Local to 'outer'
  (define (inner w)       ; Nested function
    (* w x))              ; References outer's x
  (derivative inner 1.0))
```

**Current Codegen (WRONG):**
```llvm
define tagged_value @outer() {
  %x = alloca double
  store double 2.0, ptr %x
  ...
  %result = call @derivative(@inner, ...)
}

define tagged_value @inner(tagged_value %w) {
  %0 = load double, ptr %x    ; ERROR! %x is in @outer!
  ...
}
```

LLVM rejects this because `%x` lives in `@outer`'s stack frame, but `@inner` tries to reference it directly.

## Solution Strategy

### Architecture Decision

**Use Arena Scopes for Environment Lifetime Management**

Key insight: Environments can live in the **same arena scope** as the function that creates them. When the outer function returns, the arena scope pops and the environment is automatically freed. Perfect match for Eshkol's existing memory model!

### Memory Model

```
Outer Function Call
  ↓
Arena Scope Push
  ↓
Allocate Local Variables
  ↓
Create Closure Environment ← Arena allocates here
  ↓
Call Inner Function (with environment)
  ↓
Inner Function Uses Environment
  ↓
Inner Function Returns
  ↓
Outer Function Returns
  ↓
Arena Scope Pop ← Environment automatically freed!
```

## Implementation Plan

### Phase 1: Extend Arena Memory (2 hours)

Add closure environment allocation to [`arena_memory.h`](lib/core/arena_memory.h):

```c
// Closure environment structure
typedef struct eshkol_closure_env {
    size_t num_captures;                  // Number of captured values
    eshkol_tagged_value_t captures[];     // Flexible array member
} eshkol_closure_env_t;

// Allocation function
eshkol_closure_env_t* arena_allocate_closure_env(arena_t* arena, size_t num_captures);
```

Implementation in [`arena_memory.cpp`](lib/core/arena_memory.cpp):

```cpp
eshkol_closure_env_t* arena_allocate_closure_env(arena_t* arena, size_t num_captures) {
    if (!arena) {
        eshkol_error("Cannot allocate closure environment: null arena");
        return nullptr;
    }
    
    // Calculate size: header + captures array
    size_t size = sizeof(eshkol_closure_env_t) + 
                  (num_captures * sizeof(eshkol_tagged_value_t));
    
    eshkol_closure_env_t* env = (eshkol_closure_env_t*)
        arena_allocate_aligned(arena, size, 16);
    
    if (env) {
        env->num_captures = num_captures;
        // Initialize all captures to null
        for (size_t i = 0; i < num_captures; i++) {
            env->captures[i].type = ESHKOL_VALUE_NULL;
            env->captures[i].flags = 0;
            env->captures[i].reserved = 0;
            env->captures[i].data.raw_val = 0;
        }
    }
    
    return env;
}
```

### Phase 2: Parser - Detect Captured Variables (3 hours)

Modify [`parser.cpp`](lib/frontend/parser.cpp) to analyze variable scope and populate `captured_vars`:

```cpp
// Key function to implement
void analyzeClosureCaptures(eshkol_ast_t* lambda_node, 
                           const std::set<std::string>& parent_scope_vars) {
    // 1. Collect all variable references in lambda body
    std::set<std::string> referenced_vars;
    collectVariableReferences(lambda_node->operation.lambda_op.body, referenced_vars);
    
    // 2. Determine which are captured (not in parameters, but in parent scope)
    std::set<std::string> param_names;
    for (uint64_t i = 0; i < lambda_node->operation.lambda_op.num_params; i++) {
        param_names.insert(lambda_node->operation.lambda_op.parameters[i].variable.id);
    }
    
    // 3. Find captures: referenced but not in params, and in parent scope
    std::vector<std::string> captures;
    for (const auto& var : referenced_vars) {
        if (param_names.find(var) == param_names.end() &&
            parent_scope_vars.find(var) != parent_scope_vars.end()) {
            captures.push_back(var);
        }
    }
    
    // 4. Store in AST
    if (!captures.empty()) {
        lambda_node->operation.lambda_op.num_captured = captures.size();
        lambda_node->operation.lambda_op.captured_vars = 
            new eshkol_ast_t[captures.size()];
        
        for (size_t i = 0; i < captures.size(); i++) {
            lambda_node->operation.lambda_op.captured_vars[i].type = ESHKOL_VAR;
            lambda_node->operation.lambda_op.captured_vars[i].variable.id = 
                strdup(captures[i].c_str());
        }
    }
}
```

**Integration Point:** Call this in `parseDefine` and `parseLambda` whenever a nested function is detected.

### Phase 3: Codegen - Environment Creation (4 hours)

Modify [`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) to generate closure code:

#### 3.1 Function Signature Transformation

```cpp
llvm::Function* createFunctionWithEnvironment(const std::string& name,
                                              llvm::Type* return_type,
                                              const std::vector<llvm::Type*>& param_types,
                                              bool needs_environment) {
    std::vector<llvm::Type*> func_params;
    
    if (needs_environment) {
        // First parameter is environment pointer
        func_params.push_back(llvm::PointerType::get(Context, 0));
    }
    
    // Add actual parameters
    func_params.insert(func_params.end(), param_types.begin(), param_types.end());
    
    llvm::FunctionType* func_type = llvm::FunctionType::get(
        return_type, func_params, false);
    
    return llvm::Function::Create(func_type, 
                                 llvm::Function::ExternalLinkage,
                                 name, module);
}
```

#### 3.2 Environment Allocation

```cpp
llvm::Value* codegenClosureEnvironment(eshkol_ast_t* lambda_node) {
    uint64_t num_captured = lambda_node->operation.lambda_op.num_captured;
    
    if (num_captured == 0) {
        return nullptr;  // No environment needed
    }
    
    // Call arena_allocate_closure_env
    llvm::Function* alloc_env = module->getFunction("arena_allocate_closure_env");
    llvm::Value* arena_ptr = getCurrentArenaPointer();  // Get current arena
    llvm::Value* num_cap = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(Context), num_captured);
    
    llvm::Value* env_ptr = Builder.CreateCall(alloc_env, {arena_ptr, num_cap});
    
    // Store captured variables
    for (uint64_t i = 0; i < num_captured; i++) {
        const char* var_name = lambda_node->operation.lambda_op.captured_vars[i].variable.id;
        llvm::Value* var_value = lookupVariable(var_name);
        
        // Get pointer to captures[i]
        llvm::Value* indices[] = {
            llvm::ConstantInt::get(Context, APInt(32, 0)),  // env_ptr
            llvm::ConstantInt::get(Context, APInt(32, 1)),  // captures field
            llvm::ConstantInt::get(Context, APInt(64, i))   // index i
        };
        llvm::Value* capture_ptr = Builder.CreateGEP(
            closure_env_type, env_ptr, indices);
        
        Builder.CreateStore(var_value, capture_ptr);
    }
    
    return env_ptr;
}
```

#### 3.3 Variable Lookup Transformation

```cpp
llvm::Value* lookupVariableOrCapture(const std::string& var_name,
                                     llvm::Value* env_ptr,
                                     const std::vector<std::string>& captured_vars) {
    // Check if this is a captured variable
    auto it = std::find(captured_vars.begin(), captured_vars.end(), var_name);
    
    if (it != captured_vars.end() && env_ptr) {
        // Load from environment
        size_t index = std::distance(captured_vars.begin(), it);
        
        llvm::Value* indices[] = {
            llvm::ConstantInt::get(Context, APInt(32, 0)),
            llvm::ConstantInt::get(Context, APInt(32, 1)),
            llvm::ConstantInt::get(Context, APInt(64, index))
        };
        llvm::Value* capture_ptr = Builder.CreateGEP(
            closure_env_type, env_ptr, indices);
        
        return Builder.CreateLoad(tagged_value_type, capture_ptr);
    } else {
        // Normal local variable lookup
        return lookupVariable(var_name);
    }
}
```

#### 3.4 Function Call Transformation

```cpp
llvm::Value* codegenFunctionCall(llvm::Function* func,
                                llvm::Value* env_ptr,
                                const std::vector<llvm::Value*>& args) {
    std::vector<llvm::Value*> call_args;
    
    // Check if function expects environment parameter
    if (functionNeedsEnvironment(func)) {
        call_args.push_back(env_ptr ? env_ptr : getNullPointer());
    }
    
    call_args.insert(call_args.end(), args.begin(), args.end());
    
    return Builder.CreateCall(func, call_args);
}
```

### Phase 4: Autodiff Integration (2 hours)

The `derivative` operator must handle environments:

```cpp
llvm::Value* codegenDerivative(eshkol_ast_t* node) {
    eshkol_ast_t* func_ast = node->operation.derivative_op.function;
    eshkol_ast_t* point_ast = node->operation.derivative_op.point;
    
    // Get the lambda node
    if (func_ast->type == ESHKOL_OP && 
        func_ast->operation.op == ESHKOL_LAMBDA_OP) {
        
        // Check if lambda has captured variables
        uint64_t num_captured = func_ast->operation.lambda_op.num_captured;
        
        if (num_captured > 0) {
            // Create environment for this closure
            llvm::Value* env_ptr = codegenClosureEnvironment(func_ast);
            
            // Generate AD code with environment context
            return codegenDerivativeWithEnvironment(func_ast, point_ast, env_ptr);
        }
    }
    
    // Normal derivative without environment
    return codegenDerivativeNormal(func_ast, point_ast);
}
```

## Concrete Code Locations

### Files to Modify

1. **[`inc/eshkol/eshkol.h`](inc/eshkol/eshkol.h:407)** - Add closure env type (after line 407)
2. **[`lib/core/arena_memory.h`](lib/core/arena_memory.h:154)** - Add allocation function (after line 154)
3. **[`lib/core/arena_memory.cpp`](lib/core/arena_memory.cpp:815)** - Implement allocation (at end)
4. **[`lib/frontend/parser.cpp`](lib/frontend/parser.cpp)** - Add capture analysis in lambda parsing
5. **[`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)** - Transform codegen

### Parser Entry Points

Search for these functions in [`parser.cpp`](lib/frontend/parser.cpp):
- `parseLambda()` - Add capture analysis
- `parseDefine()` - Detect nested functions and track parent scope

### Codegen Entry Points

Search for these functions in [`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp):
- `codegenLambda()` - Create environment if needed
- `codegenDefine()` - Handle nested function definitions
- `codegenVariable()` - Check environment before local lookup
- `codegenDerivative()` - Pass environment to AD system

## Testing Strategy

### Test 1: Simple Closure (No Autodiff)

```scheme
(define (make-adder x)
  (lambda (y) (+ x y)))

(define add5 (make-adder 5.0))
(add5 3.0)  ; Should return 8.0
```

**Expected:** Compiles and runs, returns 8.0

### Test 2: Closure with Derivative

```scheme
(define (demo)
  (define x 2.0)
  (define (f w) (* w x))
  (derivative f 1.0))  ; Should return 2.0
```

**Expected:** Compiles and runs, returns 2.0 (the value of x)

### Test 3: Multiple Captures

```scheme
(define (demo)
  (define x 2.0)
  (define y 3.0)
  (define (f w) (+ (* w x) y))
  (derivative f 1.0))  ; Should return 2.0
```

**Expected:** Compiles and runs, gradient w.r.t w is 2.0 (coefficient of w)

### Test 4: Neural Network Tests

Once basic closures work, test all 4 neural network demos:
- [`nn_minimal.esk`](tests/neural/nn_minimal.esk) - Already works
- [`nn_simple.esk`](tests/neural/nn_simple.esk) - Fix with closures
- [`nn_computation.esk`](tests/neural/nn_computation.esk) - Fix with closures
- [`nn_complete.esk`](tests/neural/nn_complete.esk) - Fix with closures

## Implementation Timeline

### Day 1 (4-5 hours)
- [ ] Phase 1: Add arena allocation for closure environments
- [ ] Phase 2: Implement parser capture analysis
- [ ] Test with simple closure (no autodiff)

### Day 2 (4-5 hours)
- [ ] Phase 3: Implement codegen environment passing
- [ ] Phase 4: Integrate with autodiff system
- [ ] Test with closures + derivative

### Day 3 (2-3 hours)
- [ ] Test all 4 neural network demos
- [ ] Fix any edge cases
- [ ] Update documentation

**Total: 10-13 hours**

## Success Criteria

1. ✅ All 4 neural network tests compile without LLVM errors
2. ✅ All 4 tests run and produce correct output
3. ✅ Closures work with autodiff (derivative, gradient)
4. ✅ No memory leaks (arena handles cleanup)
5. ✅ Performance degradation < 10% for non-closure code

## Risk Mitigation

### Risk: Complex Autodiff Integration
**Mitigation:** Test closures WITHOUT autodiff first, then add derivative support

### Risk: Arena Lifetime Issues
**Mitigation:** Use arena scopes - environment lives with creating function

### Risk: Multiple Nesting Levels
**Mitigation:** Start with single level, extend to arbitrary depth later

### Risk: Parser Complexity
**Mitigation:** Build incrementally - detect captures first, populate AST second

## Key Design Decisions

### 1. Environment Lifetime = Function Scope
Environments are allocated in the same arena scope as the function that creates them. When function returns, scope pops, environment is freed.

### 2. Environment as First Parameter
All closure functions get environment as hidden first parameter. This is explicit in LLVM IR but transparent to Eshkol code.

### 3. Null Environment for Non-Closures
Functions without captures pass NULL as environment parameter. Codegen checks for NULL and skips environment operations.

### 4. Captured Variables are Immutable
Initial implementation treats captures as read-only. Mutable captures require additional complexity (box/ref cells).

## Next Steps

After reviewing this plan, we should:

1. Add closure environment allocation to arena system
2. Implement parser capture analysis
3. Transform codegen to use environments
4. Test incrementally with simple cases
5. Integrate with autodiff
6. Verify all neural network tests work

Ready to proceed with implementation?