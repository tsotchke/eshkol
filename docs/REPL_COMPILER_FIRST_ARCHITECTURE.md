# Eshkol REPL: Compiler-First Architecture

**Key Principle**: The REPL is a thin interactive wrapper around the Eshkol compiler.
**Result**: Automatic feature parity - all compiler features work in REPL immediately.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Interactive Shell (eshkol-repl)                             │
│  - Readline for input                                       │
│  - History, tab completion                                  │
│  - Pretty printing of results                               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ REPL Core (lib/repl/)                                       │
│  - Session management                                       │
│  - JIT execution context                                    │
│  - Result extraction and formatting                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Eshkol Compiler (existing!)                                 │
│  ✅ Parser          → Parse input to AST                    │
│  ✅ Type checker    → Type inference and checking           │
│  ✅ LLVM Codegen    → Generate optimized IR                 │
│  ✅ Autodiff        → Automatic differentiation             │
│  ✅ Closures        → Lambda capture and execution          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ LLVM ORC JIT v2                                             │
│  - Just-in-time compilation                                 │
│  - Incremental module loading                               │
│  - Symbol resolution                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Why This is Better

### Original Plan (Independent Interpreter)
❌ Reimplements evaluation logic
❌ Features need manual porting
❌ Autodiff requires duplicate implementation
❌ Closures need separate handling
❌ Performance is slower (tree-walk)
❌ Divergence risk between REPL and compiler

### Compiler-First Plan (This Document)
✅ Uses existing compiler directly
✅ All features work automatically
✅ Autodiff works immediately
✅ Closures work immediately
✅ Performance is LLVM-optimized
✅ Perfect parity by design

**Time savings**: ~15-20 hours of reimplementation work eliminated!

---

## What We Actually Need to Build

### 1. JIT Execution Context (~2-3 hours)

**File**: `lib/repl/repl_jit.h` and `repl_jit.cpp`

```cpp
class ReplJITContext {
private:
    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::unique_ptr<llvm::LLVMContext> context_;
    std::unique_ptr<llvm::Module> module_;

    // Track definitions across evaluations
    std::unordered_map<std::string, void*> symbols_;
    int evaluation_counter_ = 0;

public:
    ReplJITContext();

    // Execute a parsed AST and return result
    ReplValue execute(const ASTNode* ast);

    // Add a definition to persistent environment
    void define(const std::string& name, void* symbol);

    // Look up a previously defined symbol
    void* lookup(const std::string& name);
};
```

**What it does**:
- Initializes LLVM ORC JIT v2 (LLJIT for simplicity)
- Maintains a persistent module across evaluations
- Each eval creates a new function, JIT compiles it, executes it
- Definitions (variables, functions) persist between evaluations

**Key insight**: We don't need to reimplement environments - LLVM manages symbols!

### 2. Result Value Extraction (~1-2 hours)

**File**: `lib/repl/repl_value.h` and `repl_value.cpp`

```cpp
class ReplValue {
public:
    enum Type { NIL, BOOLEAN, INTEGER, DOUBLE, STRING, CLOSURE };

    Type type;
    union {
        bool bool_val;
        int64_t int_val;
        double double_val;
        char* string_val;
        void* closure_ptr;
    };

    // Create from JIT-executed function result
    static ReplValue fromJITResult(void* result_ptr, Type expected_type);

    // Format for display
    std::string toString() const;
};
```

**What it does**:
- Wraps values returned from JIT execution
- Handles Eshkol's tagged value system (or typed values)
- Pretty prints results for display

**Key insight**: We don't evaluate - we just extract and display results!

### 3. Session Manager (~1 hour)

**File**: `lib/repl/repl_session.h` and `repl_session.cpp`

```cpp
class ReplSession {
private:
    ReplJITContext jit_;
    Parser parser_;
    CodeGenerator codegen_;

    std::vector<std::string> history_;
    std::unordered_map<std::string, ReplValue> last_results_;

public:
    ReplSession();

    // Main evaluation function
    ReplValue eval(const std::string& input);

    // Handle special forms
    bool isSpecialCommand(const std::string& input);
    void handleSpecialCommand(const std::string& input);
};
```

**What it does**:
- Coordinates parser → codegen → JIT → result extraction
- Maintains session history
- Handles special REPL commands (`:help`, `:type`, etc.)

### 4. Interactive Shell (~1-2 hours)

**File**: `exe/eshkol-repl.cpp`

```cpp
int main(int argc, char** argv) {
    ReplSession session;

    std::cout << "Eshkol v0.1.1 REPL\n";
    std::cout << "Type (exit) or Ctrl+D to quit\n\n";

    while (true) {
        char* input = readline("eshkol> ");
        if (!input) break;  // EOF

        if (*input) add_history(input);

        try {
            ReplValue result = session.eval(input);
            std::cout << result.toString() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
        }

        free(input);
    }

    return 0;
}
```

**What it does**:
- Readline integration for input
- History support (up/down arrows)
- Error handling and recovery
- Clean exit handling

---

## Implementation Steps

### Hour 1-2: JIT Context Setup
1. Add LLVM ORC JIT dependencies to CMakeLists.txt
2. Create `ReplJITContext` class
3. Initialize LLJIT with basic configuration
4. Test: JIT compile and execute a simple function

**Expected output**:
```cpp
// Test code
auto jit = ReplJITContext();
// Should successfully initialize LLJIT
```

### Hour 2-3: Integration with Existing Compiler
1. Link `ReplJITContext` to existing `CodeGenerator`
2. Modify codegen to support incremental compilation
3. Each REPL input becomes a new anonymous function
4. Test: Compile AST to IR and execute

**Expected output**:
```
eshkol> (+ 1 2)
// Compiles to:
// define i64 @__repl_eval_0() { ret i64 3 }
// Executes and returns 3
```

### Hour 3-4: Persistent Definitions
1. Implement `define` handling
2. Store symbols in REPL environment
3. Generate wrapper functions for cross-evaluation references
4. Test: Define variable, use in next evaluation

**Expected output**:
```
eshkol> (define x 10)
10
eshkol> (+ x 5)
15
eshkol> (define square (lambda (n) (* n n)))
<closure>
eshkol> (square x)
100
```

### Hour 4-5: Result Extraction
1. Implement `ReplValue::fromJITResult()`
2. Handle different return types (int, double, closure, etc.)
3. Implement pretty printing
4. Test: All value types display correctly

**Expected output**:
```
eshkol> #t
true
eshkol> 42
42
eshkol> 3.14159
3.14159
eshkol> "hello"
"hello"
eshkol> (lambda (x) x)
<closure at 0x...>
```

### Hour 5-6: Interactive Shell
1. Create `eshkol-repl.cpp`
2. Integrate readline
3. Add history support
4. Implement special commands (`:type`, `:help`, etc.)
5. Test: Full interactive session

### Hour 6-7: Error Handling
1. Catch parse errors
2. Catch compilation errors
3. Catch runtime errors
4. Ensure REPL continues after errors
5. Test: All error cases are recoverable

**Expected output**:
```
eshkol> (+ 1 "hello")
Error: Type mismatch: cannot add integer and string

eshkol> (this-is-undefined)
Error: Unbound variable: this-is-undefined

eshkol> (+ 1 2)
3  // REPL continues!
```

### Hour 7: CMake Integration
1. Add `eshkol-repl-lib` target
2. Add `eshkol-repl` executable target
3. Link LLVM ORC JIT libraries
4. Add readline detection
5. Build and test

```cmake
# CMakeLists.txt additions
add_library(eshkol-repl-lib
    lib/repl/repl_jit.cpp
    lib/repl/repl_value.cpp
    lib/repl/repl_session.cpp
)

target_link_libraries(eshkol-repl-lib
    eshkol-static  # Existing compiler!
    LLVMOrcJIT
    LLVMExecutionEngine
)

add_executable(eshkol-repl exe/eshkol-repl.cpp)
target_link_libraries(eshkol-repl eshkol-repl-lib readline)
```

---

## What Works Immediately (Zero Extra Work!)

Because we're using the compiler directly:

✅ **All arithmetic operators**: `+`, `-`, `*`, `/`, `%`
✅ **All comparison operators**: `<`, `>`, `<=`, `>=`, `=`, `!=`
✅ **All logical operators**: `and`, `or`, `not`
✅ **Variables**: `define`, `set!`
✅ **Control flow**: `if`, `cond`, `when`, `unless`
✅ **Let bindings**: `let`, `let*` (once implemented)
✅ **Lambdas**: `lambda` with closures
✅ **Lists**: `list`, `cons`, `car`, `cdr`, etc.
✅ **Higher-order functions**: `map`, `fold`, `filter`
✅ **Automatic differentiation**: `derivative`, `gradient`
✅ **Math functions**: `sin`, `cos`, `exp`, `log`, etc.
✅ **Vectors**: vector operations
✅ **Mixed types**: Tagged value system

**All of this works on day 1** because the compiler already implements it!

---

## Example REPL Session (What We're Building Toward)

```scheme
$ eshkol-repl
Eshkol v0.1.1 REPL
Type (exit) or Ctrl+D to quit

eshkol> (+ 1 2 3)
6

eshkol> (define x 10)
10

eshkol> (define square (lambda (n) (* n n)))
<closure>

eshkol> (square x)
100

eshkol> (map square (list 1 2 3 4 5))
(1 4 9 16 25)

eshkol> (define f (lambda (x) (* x x x)))
<closure>

eshkol> (derivative f 2.0)
12.0  ; d/dx(x³) at x=2 is 3x² = 12

eshkol> (gradient (lambda (v) (+ (* (car v) (car v))
                                  (* (cadr v) (cadr v))))
                   (list 3.0 4.0))
(6.0 8.0)  ; ∇(x²+y²) at (3,4) is (2x, 2y) = (6, 8)

eshkol> :type square
(Number -> Number)

eshkol> (exit)
$
```

---

## Technical Details

### LLVM ORC JIT v2 Integration

LLVM's ORC (On-Request Compilation) JIT is designed exactly for REPL use cases:

```cpp
// Initialization
auto jit = llvm::orc::LLJITBuilder().create();

// Add a module (each REPL evaluation)
jit->addIRModule(llvm::orc::ThreadSafeModule(
    std::move(module), std::move(context)));

// Lookup and execute
auto symbol = jit->lookup("__repl_eval_0");
auto fn = (int64_t(*)())symbol.getAddress();
int64_t result = fn();
```

### Handling Persistent State

Each REPL evaluation generates a function like:

```llvm
; First eval: (define x 10)
@__repl_x = global i64 10

define i64 @__repl_eval_0() {
    ret i64 10
}

; Second eval: (+ x 5)
define i64 @__repl_eval_1() {
    %x = load i64, i64* @__repl_x
    %result = add i64 %x, 5
    ret i64 %result
}
```

The JIT maintains all modules, so symbols are automatically available!

### Handling Closures

Closures already work in the compiler via:
- Struct containing captured variables + function pointer
- Generated in `llvm_codegen.cpp`

For REPL, we just need to:
1. Recognize closure return type
2. Print as `<closure at 0x...>`
3. Allow closures to be called in subsequent evaluations

**No extra work needed** - codegen already handles this!

### Handling Autodiff

Autodiff already works in compiler via:
- Dual number transformation
- Forward-mode differentiation in `llvm_codegen.cpp`

For REPL:
- `derivative` and `gradient` compile to existing autodiff code
- Results are just numbers - extract and print

**No extra work needed** - autodiff already implemented!

---

## Time Estimate (Revised)

| Task | Time | Notes |
|------|------|-------|
| JIT context setup | 2-3h | LLVM ORC JIT initialization |
| Compiler integration | 1-2h | Connect to existing codegen |
| Persistent definitions | 1-2h | Global symbols in JIT |
| Result extraction | 1-2h | Value wrapping and printing |
| Interactive shell | 1-2h | Readline, history, loop |
| Error handling | 1h | Catch and recover from errors |
| CMake integration | 1h | Build system setup |
| Testing | 1h | Verify all features work |
| **Total** | **9-14h** | **vs 20-29h for independent interpreter** |

**Key advantage**: All compiler features work immediately - no reimplementation!

---

## Comparison: Phase 1 Plans

### Original Plan (Tree-Walk Interpreter)
- **Time**: 6-8h for basic features
- **Features**: Arithmetic, variables, let, if, begin
- **Missing**: Lambdas, autodiff, lists, vectors
- **To reach parity**: +20-29h more work
- **Total**: ~30-35h

### Compiler-First Plan (This Document)
- **Time**: 9-14h total
- **Features**: **Everything the compiler supports**
- **Missing**: Nothing (by design)
- **To reach parity**: Already at parity!
- **Total**: ~10-14h

**Time savings**: 20-25 hours
**Feature completeness**: Day 1 vs eventual
**Maintenance burden**: None vs continuous porting

---

## Risks and Mitigations

### Risk 1: LLVM ORC JIT Complexity
**Mitigation**: Use LLJIT (simplified API), not full ORC stack
**Fallback**: If LLJIT too complex, use LLVM Interpreter (slower but simpler)

### Risk 2: Incremental Compilation Issues
**Mitigation**: Each eval is independent function, use global symbols for persistence
**Fallback**: Recompile entire environment each eval (slower but works)

### Risk 3: Debugging JIT Code
**Mitigation**: Add dump-IR special command: `:ir` shows LLVM IR for last eval
**Fallback**: Extensive logging in ReplJITContext

### Risk 4: Memory Management
**Mitigation**: Let LLVM manage JIT memory, arena allocator for REPL values
**Fallback**: Conservative GC or manual tracking if needed

---

## Testing Strategy

### Unit Tests
1. `test_repl_jit.cpp`: JIT context initialization and execution
2. `test_repl_value.cpp`: Result extraction and formatting
3. `test_repl_session.cpp`: Session state management

### Integration Tests
1. Arithmetic: All operators work
2. Variables: Define and reference across evaluations
3. Lambdas: Create and call closures
4. Autodiff: Derivative and gradient computation
5. Lists: Construction and manipulation
6. Errors: All error types are recoverable

### Manual Test Suite
```scheme
;; tests/repl/manual_test.esk
;; Run each line in REPL, verify output

;; Arithmetic
(+ 1 2 3)           ; expect: 6
(- 10 3)            ; expect: 7
(* 4 5)             ; expect: 20
(/ 10 4)            ; expect: 2.5

;; Variables
(define x 42)       ; expect: 42
x                   ; expect: 42
(define y (+ x 8))  ; expect: 50
y                   ; expect: 50

;; Lambdas
(define square (lambda (n) (* n n)))  ; expect: <closure>
(square 5)                            ; expect: 25
((lambda (x) (+ x 1)) 10)            ; expect: 11

;; Autodiff
(derivative (lambda (x) (* x x)) 5.0)  ; expect: 10.0
(gradient (lambda (v) (+ (car v) (cadr v))) (list 1.0 2.0))  ; expect: (1.0 1.0)

;; Errors (should not crash)
(undefined-var)     ; expect: Error message
(+ 1 "string")      ; expect: Type error
(/ 1 0)             ; expect: Division by zero
```

---

## Modularity: Still Independent!

Even though we're using the compiler directly, the REPL remains modular:

**What REPL adds** (new code):
- `lib/repl/repl_jit.{h,cpp}` - JIT execution context
- `lib/repl/repl_value.{h,cpp}` - Result value handling
- `lib/repl/repl_session.{h,cpp}` - Session management
- `exe/eshkol-repl.cpp` - Interactive shell

**What REPL uses** (existing code, read-only):
- Parser (produces AST)
- CodeGenerator (produces LLVM IR)
- Type system (type checking)

**What REPL does NOT touch**:
- Compiler internals
- Backend code generation
- Optimization passes
- Static compilation

**Independence maintained**:
- REPL can be disabled in builds (`-DBUILD_REPL=OFF`)
- REPL development doesn't affect compiler
- Breaking REPL doesn't break compiler
- Can develop in parallel branches

---

## Next Steps

1. **Review this architecture** - Does this address the "automatic integration" concern?
2. **Start implementation** - Begin with Hour 1-2 (JIT context setup)
3. **Test incrementally** - Each hour has a testable milestone
4. **Reach parity quickly** - Should have full features in 10-14 hours
5. **Add enhancements** - Tab completion, better printing, etc.

---

## Long-Term Vision (Unchanged)

This compiler-first architecture is the **foundation** for all future phases:

- **Phase 2**: Enhanced REPL (completion, magic commands, profiling)
- **Phase 3**: Emacs integration (inferior-eshkol-mode)
- **Phase 4**: Visualization (inline plots, tensor views)
- **Phase 5**: Live coding (hot reload, time-travel debugging)
- **Phase 6**: Notebook interface (Jupyter-style)
- **Phase 7**: Advanced (distributed computing, AI assistance)

All of these build on top of this JIT-based REPL core.

---

## Conclusion

**The compiler-first architecture is superior because**:

1. ✅ **Automatic feature parity** - compiler features = REPL features
2. ✅ **Less implementation time** - 10-14h vs 30-35h
3. ✅ **No maintenance burden** - no porting needed
4. ✅ **Better performance** - LLVM JIT vs tree-walk
5. ✅ **Still modular** - independent from compiler internals
6. ✅ **Foundation for future** - JIT enables live coding, hot reload, etc.

**This is the right architecture for Eshkol's REPL.**

Ready to implement!
