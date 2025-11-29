# Eshkol REPL - Phase 1 Minimal Implementation

**Status**: Ready to implement
**Timeline**: 6-8 hours
**Target**: Standalone, working REPL with basic features
**Philosophy**: **Ship something useful, iterate later**

---

## What We're Building (Phase 1 ONLY)

A **standalone terminal REPL** that:
- ✅ Evaluates basic expressions (arithmetic, variables, let, if)
- ✅ Maintains session state (variables persist)
- ✅ Has history (up/down arrow)
- ✅ **Does NOT depend on main compiler** (tree-walk interpreter)
- ✅ Can be developed independently
- ✅ Can be enhanced incrementally later

**Out of scope for Phase 1**:
- ❌ Lambdas/functions (Phase 2)
- ❌ Compiler integration (Phase 2)
- ❌ LLVM JIT (Phase 2)
- ❌ Emacs integration (Phase 4)
- ❌ Visualization (Phase 5)
- ❌ Live coding (Phase 6)

---

## Architecture (Simplified)

```
┌─────────────────────────────────────┐
│  eshkol-repl (single executable)    │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  Main Loop                  │   │
│  │  ├─ readline (input)        │   │
│  │  ├─ parse (reuse parser)   │   │
│  │  ├─ eval (tree-walk)       │   │
│  │  └─ print (format output)  │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  Environment                │   │
│  │  (variable bindings)        │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

**That's it.** No daemons, no protocols, no services.

---

## File Structure

```
eshkol/
├── lib/
│   └── repl/                    # NEW
│       ├── repl_value.h         # Simple value representation
│       ├── repl_value.cpp
│       ├── repl_env.h           # Environment (bindings)
│       ├── repl_env.cpp
│       ├── repl_eval.h          # Tree-walk evaluator
│       └── repl_eval.cpp
│
└── exe/
    └── eshkol-repl.cpp          # NEW - Main REPL program
```

**Total new code: ~800-1000 lines** (very manageable!)

---

## Implementation Guide

### Step 1: Value Representation (1.5 hours)

**File**: `lib/repl/repl_value.h`

```cpp
#ifndef ESHKOL_REPL_VALUE_H
#define ESHKOL_REPL_VALUE_H

#include <string>
#include <memory>
#include <variant>

namespace eshkol::repl {

class ReplValue {
public:
    enum Type {
        NIL,
        BOOLEAN,
        INTEGER,
        DOUBLE,
        STRING,
        CONS,
    };

private:
    Type type_;

    // Simple variant storage
    std::variant<
        std::nullptr_t,                          // NIL
        bool,                                    // BOOLEAN
        int64_t,                                 // INTEGER
        double,                                  // DOUBLE
        std::string,                             // STRING
        std::pair<std::shared_ptr<ReplValue>,    // CONS (car, cdr)
                  std::shared_ptr<ReplValue>>
    > data_;

public:
    // Constructors
    ReplValue();  // Default: NIL

    static ReplValue makeNil();
    static ReplValue makeBool(bool val);
    static ReplValue makeInt(int64_t val);
    static ReplValue makeDouble(double val);
    static ReplValue makeString(const std::string& val);
    static ReplValue makeCons(const ReplValue& car, const ReplValue& cdr);

    // Accessors
    Type type() const { return type_; }
    bool isNil() const { return type_ == NIL; }
    bool isCons() const { return type_ == CONS; }

    // Getters (throw if wrong type)
    bool asBool() const;
    int64_t asInt() const;
    double asDouble() const;
    std::string asString() const;
    ReplValue car() const;
    ReplValue cdr() const;

    // Display
    std::string toString() const;

    // Truthiness (Scheme-style: only #f and '() are false)
    bool isTruthy() const;
};

} // namespace eshkol::repl

#endif
```

**File**: `lib/repl/repl_value.cpp` (straightforward implementation)

---

### Step 2: Environment (1 hour)

**File**: `lib/repl/repl_env.h`

```cpp
#ifndef ESHKOL_REPL_ENV_H
#define ESHKOL_REPL_ENV_H

#include "repl_value.h"
#include <map>
#include <string>
#include <memory>

namespace eshkol::repl {

class Environment {
public:
    using Ptr = std::shared_ptr<Environment>;

private:
    std::map<std::string, ReplValue> bindings_;
    Ptr parent_;  // For nested scopes (let)

public:
    Environment() : parent_(nullptr) {}
    explicit Environment(Ptr parent) : parent_(parent) {}

    // Define new binding in this environment
    void define(const std::string& name, const ReplValue& value);

    // Lookup (searches parent chain)
    ReplValue* lookup(const std::string& name);

    // Set existing binding (searches parent chain)
    bool set(const std::string& name, const ReplValue& value);

    // List all bindings (for debugging)
    std::vector<std::string> listBindings() const;
};

} // namespace eshkol::repl

#endif
```

**File**: `lib/repl/repl_env.cpp` (simple map operations)

---

### Step 3: Evaluator (2.5 hours)

**File**: `lib/repl/repl_eval.h`

```cpp
#ifndef ESHKOL_REPL_EVAL_H
#define ESHKOL_REPL_EVAL_H

#include <eshkol/eshkol.h>  // For AST types
#include "repl_value.h"
#include "repl_env.h"

namespace eshkol::repl {

class Evaluator {
private:
    Environment::Ptr global_env_;

public:
    Evaluator();

    // Evaluate AST (uses global environment)
    ReplValue eval(const eshkol_ast_t* ast);

    // Evaluate AST in specific environment
    ReplValue eval(const eshkol_ast_t* ast, Environment::Ptr env);

private:
    // Setup built-in functions
    void setupBuiltins();

    // Evaluation helpers
    ReplValue evalDefine(const eshkol_ast_t* ast, Environment::Ptr env);
    ReplValue evalIf(const eshkol_ast_t* ast, Environment::Ptr env);
    ReplValue evalLet(const eshkol_ast_t* ast, Environment::Ptr env);
    ReplValue evalBegin(const eshkol_ast_t* ast, Environment::Ptr env);
    ReplValue evalQuote(const eshkol_ast_t* ast, Environment::Ptr env);
    ReplValue evalApplication(const eshkol_ast_t* ast, Environment::Ptr env);

    // Built-in operators (Phase 1: just arithmetic)
    ReplValue applyBuiltin(const std::string& op,
                          const std::vector<ReplValue>& args);
};

} // namespace eshkol::repl

#endif
```

**What we support in Phase 1**:
```scheme
;; Literals
42                  ; integers
3.14                ; doubles
#t #f               ; booleans
"hello"             ; strings (if parser supports)

;; Arithmetic
(+ 1 2 3)          ; addition
(- 10 3)           ; subtraction
(* 2 3 4)          ; multiplication
(/ 10 2)           ; division

;; Variables
(define x 10)
x                  ; => 10

;; Conditionals
(if (> x 5) "big" "small")

;; Let bindings
(let ((y 5)
      (z 3))
  (+ x y z))       ; => 18

;; Begin (sequencing)
(begin
  (define a 1)
  (define b 2)
  (+ a b))         ; => 3

;; Display
(display "Hello!")
(newline)

;; Lists (if time permits)
(cons 1 2)         ; => (1 . 2)
(car (cons 1 2))   ; => 1
(cdr (cons 1 2))   ; => 2
```

---

### Step 4: Main REPL Loop (1 hour)

**File**: `exe/eshkol-repl.cpp`

```cpp
#include <eshkol/eshkol.h>
#include "../lib/repl/repl_eval.h"
#include <iostream>
#include <sstream>

// Readline support (optional but recommended)
#ifdef HAVE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

using namespace eshkol::repl;

void printBanner() {
    std::cout << "Eshkol REPL v0.1 (Minimal)" << std::endl;
    std::cout << "Type (exit) to quit, (help) for help" << std::endl;
    std::cout << std::endl;
}

void printHelp() {
    std::cout << "Eshkol REPL Commands:" << std::endl;
    std::cout << "  (exit)         - Exit REPL" << std::endl;
    std::cout << "  (help)         - Show this help" << std::endl;
    std::cout << "  (env)          - List all bindings" << std::endl;
    std::cout << std::endl;
    std::cout << "Supported features:" << std::endl;
    std::cout << "  - Arithmetic: + - * /" << std::endl;
    std::cout << "  - Variables: define" << std::endl;
    std::cout << "  - Conditionals: if" << std::endl;
    std::cout << "  - Let bindings: let" << std::endl;
    std::cout << "  - Sequencing: begin" << std::endl;
    std::cout << "  - Display: display, newline" << std::endl;
}

std::string readLine(const char* prompt) {
#ifdef HAVE_READLINE
    char* input = readline(prompt);
    if (!input) return "";

    std::string result(input);
    if (!result.empty()) {
        add_history(input);
    }
    free(input);
    return result;
#else
    std::cout << prompt;
    std::string line;
    if (!std::getline(std::cin, line)) {
        return "";
    }
    return line;
#endif
}

int main(int argc, char** argv) {
    printBanner();

    Evaluator evaluator;

    while (true) {
        // Read
        std::string input = readLine("eshkol> ");

        if (input.empty()) {
            continue;
        }

        // Check for special commands
        if (input == "(exit)" || input == "exit") {
            break;
        }

        if (input == "(help)" || input == "help") {
            printHelp();
            continue;
        }

        // Parse
        std::istringstream iss(input);
        eshkol_ast_t ast = eshkol_parse_next_ast(iss);

        if (ast.type == ESHKOL_INVALID) {
            std::cerr << "Parse error" << std::endl;
            continue;
        }

        // Eval
        try {
            ReplValue result = evaluator.eval(&ast);

            // Print (unless nil - from define, display, etc.)
            if (!result.isNil()) {
                std::cout << result.toString() << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        // Cleanup AST
        eshkol_ast_destroy(&ast);
    }

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
```

---

### Step 5: CMake Integration (0.5 hours)

Add to `CMakeLists.txt`:

```cmake
# REPL library
add_library(eshkol-repl-lib STATIC
    lib/repl/repl_value.cpp
    lib/repl/repl_env.cpp
    lib/repl/repl_eval.cpp
)

target_include_directories(eshkol-repl-lib PUBLIC
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/lib
)

target_compile_features(eshkol-repl-lib PUBLIC cxx_std_17)

# REPL executable
add_executable(eshkol-repl
    exe/eshkol-repl.cpp
)

target_link_libraries(eshkol-repl
    eshkol-repl-lib
    eshkol-static  # Only for parser! Not the whole compiler
)

# Optional: Readline support
find_package(Readline)
if(READLINE_FOUND)
    target_compile_definitions(eshkol-repl PRIVATE HAVE_READLINE)
    target_include_directories(eshkol-repl PRIVATE ${READLINE_INCLUDE_DIR})
    target_link_libraries(eshkol-repl ${READLINE_LIBRARIES})
endif()

# Install
install(TARGETS eshkol-repl
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT repl)
```

---

### Step 6: Testing (0.5 hours)

Create `tests/repl/repl_basic_test.esk`:

```scheme
;; Arithmetic
(+ 1 2 3)
;; Expected: 6

(* 2 3 4)
;; Expected: 24

(- 10 3)
;; Expected: 7

(/ 10 2)
;; Expected: 5.0

;; Variables
(define x 42)
;; Expected: 42

x
;; Expected: 42

(define y (+ x 8))
;; Expected: 50

;; Conditionals
(if #t 1 2)
;; Expected: 1

(if #f 1 2)
;; Expected: 2

(if (> x 40) "big" "small")
;; Expected: "big"

;; Let
(let ((a 5)
      (b 3))
  (+ a b))
;; Expected: 8

;; Nested let
(let ((x 1))
  (let ((x 2))
    x))
;; Expected: 2

;; Begin
(begin
  (define z 100)
  (+ z 1))
;; Expected: 101
```

Manual testing:
```bash
$ cmake --build build
$ ./build/eshkol-repl

eshkol> (+ 1 2 3)
6
eshkol> (define x 10)
10
eshkol> (* x 5)
50
eshkol> (let ((y 3)) (+ x y))
13
eshkol> (if (> x 5) "yes" "no")
"yes"
eshkol> (exit)
Goodbye!
```

---

## What You Get After 6-8 Hours

### ✅ Working REPL
- Evaluates expressions
- Maintains state
- Has history (if readline available)
- Clean error messages

### ✅ Useful for Development
- Can test expressions quickly
- Can define helper functions (once lambdas added in Phase 2)
- Good for experimentation

### ✅ Foundation for Future
- Clean architecture
- Easy to extend
- Documented

### ✅ Independent
- Doesn't touch compiler code
- Won't break if compiler changes
- Can develop in parallel with v1.0 work

---

## Known Limitations (Phase 1)

### ❌ No lambdas
```scheme
eshkol> (lambda (x) (* x 2))
Error: Lambda not yet supported
```

**Workaround**: Add in Phase 2 (another 2-3 hours)

### ❌ No lists (except cons)
```scheme
eshkol> (list 1 2 3)
Error: list not yet supported
```

**Workaround**: Use `(cons 1 (cons 2 (cons 3 '())))` or add in Phase 2

### ❌ No autodiff
```scheme
eshkol> (derivative (lambda (x) (* x x)) 5.0)
Error: derivative requires compiler integration
```

**Workaround**: Add in Phase 2 when integrating with compiler

### ❌ Slow (interpreted)
No JIT compilation yet - pure tree-walk interpreter.

**Workaround**: Add LLVM JIT in Phase 2

---

## Phase 2 Additions (Future)

Once Phase 1 is working, you can add:

### Lambdas (2-3 hours)
```cpp
struct Lambda {
    std::vector<std::string> params;
    const eshkol_ast_t* body;
    Environment::Ptr closure;
};
```

### Compiler Integration (3-4 hours)
- Start `eshkol-compiler-daemon`
- Send compilation requests via JSON-RPC
- Get back compiled functions
- Execute with LLVM JIT

### Lists (1 hour)
- Implement `list`, `append`, `reverse`
- Implement `map`, `filter`, `fold`

---

## Build & Test Instructions

```bash
# Build
cd /Users/tyr/Desktop/eshkol
cmake --build build --target eshkol-repl

# Run
./build/eshkol-repl

# Test
eshkol> (+ 1 2 3)
6
eshkol> (define x 10)
10
eshkol> (if (> x 5) "yes" "no")
"yes"
eshkol> (exit)

# Install (optional)
cd build
sudo make install
eshkol-repl  # Should work from anywhere
```

---

## Timeline Breakdown

| Task | Time | Deliverable |
|------|------|-------------|
| ReplValue implementation | 1.5h | Value representation working |
| Environment implementation | 1h | Variable bindings working |
| Evaluator core | 1.5h | Basic eval working |
| Built-in operators | 1h | Arithmetic, if, let working |
| Main REPL loop | 1h | Interactive loop working |
| CMake integration | 0.5h | Builds successfully |
| Testing & debugging | 0.5h | All tests pass |
| **Total** | **7 hours** | **Working REPL!** |

Add 1-2 hours buffer for unexpected issues = **8-9 hours realistic**

---

## Success Criteria

Phase 1 is complete when:

- [ ] REPL starts and shows prompt
- [ ] Can evaluate `(+ 1 2 3)` → `6`
- [ ] Can define variables: `(define x 10)` → `10`
- [ ] Can use variables: `x` → `10`
- [ ] Can use if: `(if #t 1 2)` → `1`
- [ ] Can use let: `(let ((y 5)) (+ x y))` → `15`
- [ ] History works (up arrow recalls previous command)
- [ ] Can exit cleanly with `(exit)`
- [ ] Doesn't crash on parse errors
- [ ] Gives reasonable error messages

---

## What NOT to Do in Phase 1

❌ **Don't** try to integrate with compiler yet
❌ **Don't** implement lambdas yet
❌ **Don't** worry about performance
❌ **Don't** implement visualization
❌ **Don't** implement Emacs integration
❌ **Don't** implement advanced features

**Just get the basics working.**

You can iterate later!

---

## Ready to Start?

The implementation is straightforward:

1. Start with `ReplValue` (simple variant class)
2. Add `Environment` (simple map)
3. Implement `Evaluator` (tree-walk pattern matching)
4. Wire up main loop (readline + eval)
5. Test

**This is achievable in one focused day of work.**

After v1.0-foundation ships, you can enhance the REPL incrementally without rushing.

**Want me to generate the actual code files to get you started?** 🚀
