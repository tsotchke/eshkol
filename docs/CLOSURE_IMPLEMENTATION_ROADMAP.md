# Closure Implementation Roadmap - Executive Summary

## Mission

Implement proper lexical closures in Eshkol to enable natural neural network code patterns, fixing the "Referring to an instruction in another function!" LLVM error.

## The Problem

```scheme
(define (demo)
  (define x 2.0)
  (define (loss w) (* w x))  ; ERROR: 'loss' can't access parent's 'x'
  (derivative loss 1.0))
```

LLVM rejects cross-function variable references. We need closure environments to pass parent variables to nested functions.

## The Solution

### Architecture

**Use Arena-Based Closure Environments**

```c
typedef struct eshkol_closure_env {
    size_t num_captures;
    eshkol_tagged_value_t captures[];  // Captured parent variables
} eshkol_closure_env_t;
```

- Allocated from arena (automatic cleanup!)
- Passed as hidden first parameter to closure functions
- Accessed when nested function needs parent variables

### Key Insight

The AST **already has** the infrastructure:
- [`lambda_op.captured_vars`](inc/eshkol/eshkol.h:290) exists but is unused
- [`lambda_op.num_captured`](inc/eshkol/eshkol.h:291) exists but is unused

We just need to:
1. **Parser:** Populate these fields when detecting captures
2. **Codegen:** Generate environment code when `num_captured > 0`

## Implementation Phases

### Phase 1: Arena Support (2 hours)
**Files:** [`arena_memory.h`](lib/core/arena_memory.h), [`arena_memory.cpp`](lib/core/arena_memory.cpp)

Add one function:
```c
eshkol_closure_env_t* arena_allocate_closure_env(arena_t* arena, size_t num_captures);
```

**Safety:** New function, doesn't affect existing code

### Phase 2: Parser Analysis (3 hours)
**Files:** [`parser.cpp`](lib/frontend/parser.cpp)

Implement variable capture detection:
1. When parsing lambda/define, track parent scope variables
2. Analyze lambda body for variable references
3. Determine which references are captures (in parent, not in params)
4. Populate `captured_vars` and `num_captured` in AST

**Safety:** Just populates AST fields, doesn't change codegen yet

### Phase 3: Codegen Transformation (4 hours)
**Files:** [`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)

Transform function generation when `num_captured > 0`:
1. Add environment parameter to function signature
2. Allocate environment before creating closure
3. Load variables from environment instead of parent scope
4. Pass environment when calling closures

**Safety:** Guarded by `if (num_captured > 0)` - existing code uses old path

### Phase 4: Autodiff Integration (2 hours)
**Files:** [`llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)

Update AD operators to handle environments:
1. `codegenDerivative` - pass environment to AD graph construction
2. `codegenGradient` - handle closure functions
3. AD graph evaluation - use environment when evaluating closures

**Safety:** Check for environment before using it

### Phase 5: Testing & Validation (2 hours)
**Files:** [`tests/closure_*`](tests), neural network tests

Test progression:
1. Simple closure without autodiff
2. Closure with derivative
3. Multiple captures
4. All 4 neural network tests

**Safety:** Comprehensive regression testing

## Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 1. Arena Support | 2h | 2h |
| 2. Parser Analysis | 3h | 5h |
| 3. Codegen Transform | 4h | 9h |
| 4. Autodiff Integration | 2h | 11h |
| 5. Testing & Validation | 2h | 13h |

**Total: 13 hours** (with buffer)

## Safety Guarantees

### Backward Compatibility

**CRITICAL:** All existing tests MUST continue working!

**Strategy:**
1. Additive changes only - no deletions
2. Conditional logic based on `num_captured`
3. Default to existing behavior when `num_captured == 0`
4. Test after EVERY change

### Validation Checkpoints

**After Each Phase:**
```bash
cmake --build build && ./scripts/run_all_tests.sh
```

**If ANY test fails:**
1. STOP immediately
2. Revert changes
3. Analyze and fix in isolation
4. Re-test before continuing

## Success Criteria

### ✅ Must Achieve

1. All existing tests pass unchanged
2. Neural network tests compile and run
3. Map/fold/filter still work perfectly
4. Autodiff works on regular functions  
5. No LLVM verification errors

### ✅ Goals

1. All 4 neural network demos working
2. Natural closure syntax supported
3. Performance impact < 5% for non-closures
4. Code remains maintainable

## Code Entry Points

### Files to Modify (in order)

1. [`inc/eshkol/eshkol.h`](inc/eshkol/eshkol.h:407) - Add closure env type definition
2. [`lib/core/arena_memory.h`](lib/core/arena_memory.h:154) - Add allocation declaration
3. [`lib/core/arena_memory.cpp`](lib/core/arena_memory.cpp:815) - Implement allocation
4. [`lib/frontend/parser.cpp`](lib/frontend/parser.cpp) - Add capture analysis
5. [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) - Transform codegen

### Functions to Search For

**Parser:**
- `parseLambda` - Add capture detection
- `parseDefine` - Track parent scope for nested functions

**Codegen:**
- `codegenLambda` - Create environment if needed
- `codegenDefine` - Handle nested definitions
- `codegenVariable` - Check environment before local lookup
- `codegenDerivative` - Pass environment to AD system
- `codegenCall` - Pass environment to closures

## Documentation Created

1. **[`CLOSURE_IMPLEMENTATION_PLAN.md`](docs/CLOSURE_IMPLEMENTATION_PLAN.md)** - High-level architecture
2. **[`CLOSURE_TECHNICAL_SPECIFICATION.md`](docs/CLOSURE_TECHNICAL_SPECIFICATION.md)** - Detailed technical design
3. **[`CLOSURE_BACKWARD_COMPATIBILITY_PLAN.md`](docs/CLOSURE_BACKWARD_COMPATIBILITY_PLAN.md)** - Safety strategy
4. **`CLOSURE_IMPLEMENTATION_ROADMAP.md`** (this document) - Executive summary

## Next Step

Switch to **Code mode** and begin Phase 1: Arena Support

The plan is solid, the architecture is clear, backward compatibility is guaranteed. Ready to implement!