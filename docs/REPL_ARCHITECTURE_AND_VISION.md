# Eshkol Interactive Environment (EIE)
## Architecture, Vision & Implementation Strategy

**Version**: 1.0-design
**Status**: Design Document
**Target**: v1.1+ (Post v1.0-foundation)
**Created**: November 27, 2025

---

## Table of Contents

1. [Vision & Philosophy](#vision--philosophy)
2. [Design Principles](#design-principles)
3. [Architecture Overview](#architecture-overview)
4. [Modularity Boundaries](#modularity-boundaries)
5. [Component Specifications](#component-specifications)
6. [Emacs Integration Strategy](#emacs-integration-strategy)
7. [Implementation Phases](#implementation-phases)
8. [Advanced Features Roadmap](#advanced-features-roadmap)
9. [Technical Specifications](#technical-specifications)
10. [References & Inspiration](#references--inspiration)

---

## Vision & Philosophy

### The Dream

**Eshkol Interactive Environment (EIE)** will be a live coding environment that combines:
- **Edwin's** elegant Scheme IDE integration
- **IPython's** rich interactive experience
- **GHCi's** powerful type introspection
- **Julia REPL's** performance and live plotting
- **HolyC/TempleOS's** direct system interaction and immediacy

### Core Philosophy

> "The REPL is not a debugging tool. It's a creative canvas where neural networks are sculpted, gradients are visualized in real-time, and mathematical ideas flow directly from thought to execution."

**Key Tenets**:
1. **Immediacy**: Code executes instantly, feedback is immediate
2. **Transparency**: Show the computation graph, gradients, types, everything
3. **Interactivity**: Modify running programs, hot-reload code, inspect state
4. **Integration**: Deep Emacs integration for professional workflows
5. **Performance**: LLVM-backed JIT for production-speed experimentation
6. **Visualization**: Neural networks, gradients, tensors - all visualizable inline

### What Makes EIE Unique

**Automatic Differentiation + Live Coding** = Unprecedented ML Development Experience

```scheme
;; Train a neural network LIVE
eshkol> (define model (make-mlp [2 4 1]))
eshkol> (visualize model)           ; Opens graph visualization
eshkol> (train-live model xor-data   ; Trains with LIVE updates
          :watch-gradients           ; Shows gradient flow in real-time
          :plot-loss                 ; Plots loss curve as it trains
          :interactive)              ; Can pause/modify/continue
Epoch 0:  Loss 2.451  [████████░░] Gradient norm: 1.23
Epoch 50: Loss 0.234  [█████░░░░░] Gradient norm: 0.45
Epoch 100: Loss 0.012 [██░░░░░░░░] Gradient norm: 0.02
[Press 'p' to pause, 'e' to examine weights, 'q' to quit]

eshkol> (examine-weights model)
Layer 0: weights shape [2x4]
  [[0.45  0.23  -0.12  0.56]
   [0.34  -0.28  0.41  0.19]]
  Gradient: max=0.034, min=-0.021, mean=0.007

eshkol> (modify-weight! model 0 1 2 0.5)  ; Hot-patch weights!
eshkol> (continue-training)                ; Resume training
```

**This doesn't exist anywhere else.**

---

## Design Principles

### 1. **Total Modularity**

```
┌─────────────────────────────────────────────────────┐
│          Eshkol Compiler (eshkol-run)               │
│                                                      │
│  ┌──────────┬──────────┬──────────┬──────────┐     │
│  │  Parser  │ Frontend │  LLVM    │  Binary  │     │
│  │          │ Codegen  │ Backend  │  Output  │     │
│  └──────────┴──────────┴──────────┴──────────┘     │
│                                                      │
│  Public API: libeshkol-compiler.so                  │
└─────────────────────────────────────────────────────┘
                         ▲
                         │ Clean API boundary
                         │ (Uses, never modifies)
                         ▼
┌─────────────────────────────────────────────────────┐
│     Eshkol Interactive Environment (eshkol-repl)    │
│                                                      │
│  ┌──────────┬──────────┬──────────┬──────────┐     │
│  │   REPL   │  Editor  │   Live   │  Visual  │     │
│  │   Core   │  Bridge  │  Coding  │  Engine  │     │
│  └──────────┴──────────┴──────────┴──────────┘     │
│                                                      │
│  Public API: libeshkol-interactive.so               │
└─────────────────────────────────────────────────────┘
                         ▲
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Emacs Integration (eshkol-mode)        │
│                                                      │
│  ┌──────────┬──────────┬──────────┬──────────┐     │
│  │  Major   │  Comint  │  Company │ Flycheck │     │
│  │   Mode   │  Backend │  Backend │ Backend  │     │
│  └──────────┴──────────┴──────────┴──────────┘     │
└─────────────────────────────────────────────────────┘
```

**Guarantee**: EIE can be removed entirely without affecting compiler functionality.

### 2. **Compiler as Backend, Not Dependency**

The compiler is a **service** that EIE calls, not a library EIE links against.

**Communication Model**:
```
EIE Process                    Compiler Service
    │                               │
    ├──── Parse request ───────────>│
    │<─── AST ──────────────────────┤
    │                               │
    ├──── Compile request ─────────>│
    │<─── LLVM IR or Binary ────────┤
    │                               │
    ├──── Execute request ─────────>│
    │<─── Result ───────────────────┤
```

**Benefits**:
- EIE can be restarted without recompiling
- Compiler can be updated without breaking EIE
- Process isolation prevents crashes in one from affecting the other
- Can run on different machines (remote REPL!)

### 3. **Staged Implementation**

Each phase is **independently useful** and **incrementally adds value**.

### 4. **Emacs-First Philosophy**

Emacs integration is not an afterthought - it's the **primary interface**.

Terminal REPL is a **fallback** for non-Emacs users.

### 5. **Performance Through LLVM**

Unlike interpreters, EIE uses **JIT compilation** for production speed.

```scheme
;; This runs at COMPILED SPEED
eshkol> (define (fib n)
          (if (<= n 1) n
              (+ (fib (- n 1)) (fib (- n 2)))))

eshkol> (time (fib 35))
Compiling fib... done (12ms)
Execution time: 45ms  ; FAST! (not interpreted speed)
Result: 9227465
```

---

## Architecture Overview

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Interactive Shell (Terminal/Emacs)        │
│  ┌────────────────────────────────────────────┐    │
│  │ • Input handling (readline/Emacs)           │    │
│  │ • Display rendering (terminal/buffer)       │    │
│  │ • History management                        │    │
│  │ • Multi-line editing                        │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  Layer 2: Interactive Core (libeshkol-repl)         │
│  ┌────────────────────────────────────────────┐    │
│  │ REPL Engine                                 │    │
│  │ ├─ Session management                       │    │
│  │ ├─ Environment (bindings)                   │    │
│  │ ├─ Evaluation strategy                      │    │
│  │ └─ Result formatting                        │    │
│  │                                             │    │
│  │ Live Coding Engine                          │    │
│  │ ├─ Hot reload                               │    │
│  │ ├─ State inspection                         │    │
│  │ ├─ Breakpoints                              │    │
│  │ └─ Time-travel debugging                    │    │
│  │                                             │    │
│  │ Visualization Engine                        │    │
│  │ ├─ Gradient visualization                   │    │
│  │ ├─ Computation graph rendering              │    │
│  │ ├─ Plot generation                          │    │
│  │ └─ Interactive graphics                     │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│  Layer 3: Compiler Service (eshkol-compiler-daemon) │
│  ┌────────────────────────────────────────────┐    │
│  │ • AST generation (parsing)                  │    │
│  │ • LLVM IR generation                        │    │
│  │ • JIT compilation                           │    │
│  │ • Execution (sandboxed)                     │    │
│  │ • Type inference                            │    │
│  │ • Gradient computation                      │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### Communication Protocol

**JSON-RPC over Unix Domain Socket** (or TCP for remote)

```json
// Request
{
  "jsonrpc": "2.0",
  "method": "compile_and_execute",
  "params": {
    "code": "(+ 1 2 3)",
    "session_id": "abc123",
    "mode": "jit"
  },
  "id": 1
}

// Response
{
  "jsonrpc": "2.0",
  "result": {
    "value": 6,
    "type": "int64",
    "execution_time_ms": 0.23
  },
  "id": 1
}
```

---

## Modularity Boundaries

### File Structure

```
eshkol/
├── exe/
│   ├── eshkol-run.cpp              # Main compiler (unchanged)
│   ├── eshkol-compiler-daemon.cpp  # NEW - Compiler service
│   └── eshkol-repl.cpp             # NEW - Terminal REPL
│
├── lib/
│   ├── backend/                    # Compiler internals (unchanged)
│   ├── frontend/                   # Compiler internals (unchanged)
│   ├── core/                       # Compiler internals (unchanged)
│   │
│   ├── repl/                       # NEW - REPL library (independent!)
│   │   ├── repl_core.h/cpp         # Core REPL engine
│   │   ├── repl_session.h/cpp      # Session management
│   │   ├── repl_eval.h/cpp         # Evaluation strategies
│   │   ├── repl_env.h/cpp          # Environment (bindings)
│   │   ├── repl_display.h/cpp      # Display/formatting
│   │   └── repl_protocol.h/cpp     # Compiler communication
│   │
│   ├── live/                       # NEW - Live coding engine
│   │   ├── hot_reload.h/cpp        # Hot reload functionality
│   │   ├── state_inspect.h/cpp     # State inspection
│   │   ├── breakpoint.h/cpp        # Breakpoint management
│   │   └── time_travel.h/cpp       # Time-travel debugging
│   │
│   └── visual/                     # NEW - Visualization engine
│       ├── gradient_viz.h/cpp      # Gradient visualization
│       ├── graph_render.h/cpp      # Computation graph
│       ├── plot.h/cpp              # Plotting
│       └── term_graphics.h/cpp     # Terminal graphics
│
├── emacs/                          # NEW - Emacs integration
│   ├── eshkol-mode.el              # Major mode
│   ├── eshkol-repl.el              # REPL interface
│   ├── eshkol-company.el           # Completion
│   ├── eshkol-flycheck.el          # Syntax checking
│   └── eshkol-inspector.el         # Inspector UI
│
└── docs/
    ├── REPL_ARCHITECTURE_AND_VISION.md  # This file
    ├── REPL_API_SPECIFICATION.md        # API docs
    ├── REPL_PROTOCOL.md                 # Protocol spec
    └── EMACS_INTEGRATION_GUIDE.md       # Emacs integration
```

### API Boundaries

#### Compiler Service API

```cpp
// lib/compiler/compiler_service.h
namespace eshkol::compiler {

class CompilerService {
public:
    // Parse source code to AST
    Result<AST> parse(const std::string& source);

    // Compile to LLVM IR
    Result<Module> compile(const AST& ast);

    // JIT execute
    Result<Value> execute(const Module& module,
                          const ExecutionContext& context);

    // Type inference
    Result<Type> infer_type(const AST& ast);

    // Gradient computation
    Result<Module> compute_gradient(const AST& function,
                                     const std::vector<Value>& point);
};

} // namespace eshkol::compiler
```

**Guarantee**: This API is **read-only** with respect to compiler internals.

#### REPL API

```cpp
// lib/repl/repl_core.h
namespace eshkol::repl {

class ReplCore {
public:
    // Start a new session
    SessionId create_session();

    // Evaluate code in a session
    Result<Value> eval(SessionId session, const std::string& code);

    // Query session state
    Environment get_environment(SessionId session);

    // Modify session
    void define(SessionId session, const std::string& name, const Value& value);
    void undefine(SessionId session, const std::string& name);

    // History
    std::vector<HistoryEntry> get_history(SessionId session);

    // Visualization
    void visualize_gradient(const Value& function, const Value& point);
    void visualize_graph(const AST& ast);
};

} // namespace eshkol::repl
```

### Build Independence

```cmake
# CMakeLists.txt

# Compiler library (existing)
add_library(eshkol-compiler SHARED
    lib/backend/*.cpp
    lib/frontend/*.cpp
    lib/core/*.cpp
)

# REPL library (NEW - completely independent)
add_library(eshkol-repl SHARED
    lib/repl/*.cpp
)
target_link_libraries(eshkol-repl
    # NO direct link to eshkol-compiler!
    # Communication via protocol only
)

# Live coding library (NEW)
add_library(eshkol-live SHARED
    lib/live/*.cpp
)
target_link_libraries(eshkol-live
    eshkol-repl  # Only depends on REPL, not compiler
)

# Visualization library (NEW)
add_library(eshkol-visual SHARED
    lib/visual/*.cpp
)
target_link_libraries(eshkol-visual
    eshkol-repl
)

# Can build REPL without touching compiler!
add_custom_target(repl-only
    DEPENDS eshkol-repl eshkol-live eshkol-visual eshkol-repl-exe
)
```

**Test**: `cmake --build build --target repl-only` should work even if compiler is broken.

---

## Component Specifications

### 1. REPL Core (`lib/repl/`)

**Responsibilities**:
- Session management (multiple concurrent sessions)
- Environment management (variable bindings)
- History (persistent, searchable)
- Evaluation coordination (calls compiler service)
- Result formatting

**Does NOT**:
- Parse code (delegates to compiler)
- Generate LLVM IR (delegates to compiler)
- Execute directly (delegates to compiler)

**Key Abstractions**:

```cpp
class Session {
    SessionId id;
    Environment env;              // Current bindings
    History history;              // Command history
    CompilerServiceClient compiler;  // Connection to compiler

    Result<Value> eval(const std::string& code);
    void define(const std::string& name, const Value& value);
    Value* lookup(const std::string& name);
};

class Environment {
    std::map<std::string, Value> bindings;
    Environment* parent;  // For nested scopes

    void bind(const std::string& name, const Value& value);
    Value* lookup(const std::string& name);
    std::vector<std::string> list_bindings();
};

class History {
    std::vector<HistoryEntry> entries;

    void add(const std::string& input, const Value& output);
    std::vector<HistoryEntry> search(const std::string& pattern);
    void save_to_file(const std::string& path);
    void load_from_file(const std::string& path);
};
```

### 2. Live Coding Engine (`lib/live/`)

**Responsibilities**:
- Hot reload (modify running code)
- State inspection (examine variables, closures, etc.)
- Breakpoints (pause execution, inspect state)
- Time-travel debugging (rewind/replay execution)

**Inspired by**: Smalltalk image-based development, Lisp machines

**Key Features**:

```scheme
;; Hot reload example
eshkol> (define (f x) (* x 2))
eshkol> (define running-process (start-server f))
Server started with function f

;; In another window, modify f
eshkol> (redefine (f x) (* x 3))  ; Hot patch!
Hot reload: f updated in 1 running process

;; The server now uses new definition!

;; State inspection
eshkol> (inspect running-process)
Process #1234
  Function: f (version 2)
  Captured state: {}
  Call stack: [...]

;; Breakpoint
eshkol> (break-before (derivative loss-fn w))
Breakpoint set

eshkol> (train model data)
Breakpoint hit at (derivative loss-fn w)
  w = 0.534
  loss-fn = <closure capturing: x=2.0, y=4.0>

eshkol-debug> (examine w)
0.534 (type: double)

eshkol-debug> (set! w 0.6)  ; Modify on the fly!
eshkol-debug> (continue)
```

### 3. Visualization Engine (`lib/visual/`)

**Responsibilities**:
- Gradient visualization (show gradient flow)
- Computation graph rendering (show autodiff graph)
- Plotting (loss curves, weights, etc.)
- Terminal graphics (for non-Emacs users)

**Inspired by**: Julia's Plots.jl, Matplotlib, Graphviz

**Output Modes**:
1. **Terminal** - ASCII art, Unicode box drawing
2. **Emacs** - Images embedded in buffer
3. **Web** - HTML/SVG export
4. **File** - PNG/PDF export

**Examples**:

```scheme
;; Visualize gradient descent
eshkol> (plot-gradient-descent
          (lambda (x) (* x x))
          :initial 2.0
          :learning-rate 0.1
          :steps 10)

    f(x) = x²

    4.0 ┤     ●
        │      ╲
    3.0 ┤       ●
        │        ╲
    2.0 ┤         ●
        │          ╲
    1.0 ┤           ●──●──●──●──●──●
        │
    0.0 └─────────────────────────────
       -2  -1   0   1   2   3   4   5

;; Visualize neural network
eshkol> (visualize-network model)

    Input Layer      Hidden Layer     Output Layer
        ●                  ●                 ●
         ╲╲              ╱  ╲              ╱
          ╲╲            ╱    ╲            ╱
        ●  ╲╲────────●        ●────────●
             ╲╲      ╱ ╲      ╱
              ╲╲    ╱   ╲    ╱
        ●       ╲──●     ●──●

    Weights: min=-0.45, max=0.67, mean=0.12
    Gradient norm: 0.034

;; Live training visualization (animated!)
eshkol> (train-with-viz model xor-data)
[Shows animated gradient descent in terminal/Emacs buffer]
```

### 4. Compiler Service Daemon

**Responsibilities**:
- Long-running process that serves compilation requests
- Maintains compilation cache (avoid recompiling)
- Manages LLVM contexts
- Provides type information
- Computes gradients

**Protocol**: JSON-RPC 2.0

**Methods**:
```json
{
  "parse": "Parse source to AST",
  "compile": "Compile AST to LLVM IR",
  "execute": "JIT execute LLVM IR",
  "infer_type": "Infer type of expression",
  "compute_gradient": "Compute gradient of function",
  "get_completion": "Get completions for partial input",
  "get_documentation": "Get documentation for symbol"
}
```

**Benefits**:
- Persistent compilation cache (faster REPL)
- Process isolation (crash recovery)
- Remote compilation (REPL on one machine, compile on another)
- Resource management (limit memory/CPU)

---

## Emacs Integration Strategy

### Goal: Make Eshkol Feel Native to Emacs

Like SLIME for Common Lisp, or Geiser for Scheme.

### Components

#### 1. `eshkol-mode.el` - Major Mode

```elisp
;; Syntax highlighting
;; Indentation
;; Keybindings
;; Integration with other modes

(define-derived-mode eshkol-mode lisp-mode "Eshkol"
  "Major mode for editing Eshkol code."
  (setq-local comment-start ";")
  (setq-local comment-start-skip ";+\\s-*")
  (eshkol-setup-syntax-table)
  (eshkol-setup-font-lock))
```

#### 2. `eshkol-repl.el` - REPL Interface

```elisp
;; Comint-based REPL buffer
;; Send code from source buffer to REPL
;; Result display
;; Multi-line input

(defun eshkol-repl ()
  "Start Eshkol REPL."
  (interactive)
  (let ((buffer (make-comint "eshkol" "eshkol-repl")))
    (with-current-buffer buffer
      (eshkol-repl-mode))
    (pop-to-buffer buffer)))

(defun eshkol-eval-defun ()
  "Evaluate current top-level form."
  (interactive)
  (let ((form (eshkol-defun-at-point)))
    (eshkol-repl-send form)))
```

#### 3. `eshkol-company.el` - Completion

```elisp
;; Completion at point
;; Queries REPL for available symbols
;; Type-aware completion

(defun eshkol-company-backend (command &optional arg &rest ignored)
  "Company backend for Eshkol."
  (interactive (list 'interactive))
  (case command
    (interactive (company-begin-backend 'eshkol-company-backend))
    (prefix (eshkol-company-prefix))
    (candidates (eshkol-company-candidates arg))
    (meta (eshkol-company-meta arg))
    (annotation (eshkol-company-annotation arg))))
```

#### 4. `eshkol-flycheck.el` - Syntax Checking

```elisp
;; Real-time syntax checking
;; Type error highlighting
;; Queries compiler for type info

(flycheck-define-checker eshkol
  "Eshkol syntax checker."
  :command ("eshkol-compiler-daemon" "--check" source)
  :error-patterns
  ((error line-start (file-name) ":" line ":" column ": error: " (message))
   (warning line-start (file-name) ":" line ":" column ": warning: " (message))))
```

#### 5. `eshkol-inspector.el` - Inspector UI

```elisp
;; Inspect values in REPL
;; Tree view of data structures
;; Visualizations embedded in buffer

(defun eshkol-inspect (value)
  "Inspect VALUE in separate buffer."
  (interactive (list (eshkol-symbol-at-point)))
  (let ((buffer (get-buffer-create "*Eshkol Inspector*")))
    (with-current-buffer buffer
      (eshkol-inspector-mode)
      (eshkol-inspector-display value))
    (pop-to-buffer buffer)))
```

### Workflow Example

```
┌─────────────────────────────────────────────────────┐
│ neural-network.esk                                   │
│                                                      │
│ (define (train-step w x y lr)                       │
│   (let* ((pred (* w x))                             │
│          (loss (- pred y))                          │
│          (grad (* 2.0 (* loss x))))█                │
│     (- w (* lr grad))))                             │
│                                                      │
│ C-c C-c → Eval defun                                │
└─────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│ *eshkol-repl*                                        │
│                                                      │
│ eshkol> ;; Evaluating train-step...                 │
│ Compiling... done (15ms)                            │
│ => <function train-step>                            │
│                                                      │
│ eshkol> (train-step 0.5 2.0 4.0 0.1)                │
│ => 0.8                                              │
│                                                      │
│ eshkol> (visualize-gradient train-step '(0.5 ...)) │
└─────────────────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│ *eshkol-visualization*                               │
│                                                      │
│ [Gradient flow diagram appears here]                │
│                                                      │
│ [Interactive plot with mouse hover showing values]  │
└─────────────────────────────────────────────────────┘
```

### Key Emacs Features

1. **Inline Evaluation**: `C-c C-c` evaluates and shows result inline
2. **Jump to Definition**: Click on symbol to jump to definition
3. **Documentation Lookup**: Hover over symbol to see docs
4. **Live Completion**: As-you-type completion with type info
5. **Error Navigation**: Jump to errors from compilation buffer
6. **Inspector**: Rich tree view of complex values
7. **Debugger**: Step through code, inspect stack
8. **Profiler**: See timing breakdown of functions
9. **Plots**: Embedded inline plots (using Emacs image support)
10. **Hot Reload**: Modify code and see changes immediately

---

## Implementation Phases

### Phase 0: Documentation & Planning (Current)
**Duration**: 1 week
**Deliverables**:
- ✅ Architecture document (this file)
- [ ] API specification
- [ ] Protocol specification
- [ ] Emacs integration guide

### Phase 1: Minimal Terminal REPL (v1.1)
**Duration**: 2 weeks (20-30 hours)
**Deliverables**:
- Basic REPL loop (readline)
- Simple evaluator (arithmetic, variables, let, if)
- Compiler service communication (JSON-RPC)
- History support
- Session save/load

**Success Criteria**:
```bash
$ eshkol-repl
eshkol> (define x 10)
10
eshkol> (+ x 5)
15
eshkol> (let ((y 3)) (* x y))
30
```

### Phase 2: Compiler Service Daemon (v1.1)
**Duration**: 2 weeks (25-35 hours)
**Deliverables**:
- Standalone daemon process
- JSON-RPC server
- Compilation caching
- Type inference support
- Gradient computation support

**Success Criteria**:
```bash
$ eshkol-compiler-daemon &
[Daemon started on port 9876]

$ eshkol-repl --daemon localhost:9876
Connected to compiler daemon
eshkol> (define (f x) (* x x))
[Compiled and cached]
eshkol> (derivative f 5.0)
10.0
```

### Phase 3: Enhanced REPL Features (v1.2)
**Duration**: 3 weeks (30-40 hours)
**Deliverables**:
- Multi-line input
- Syntax highlighting (terminal)
- Tab completion
- Magic commands (%timeit, %save, %load, etc.)
- Better error messages
- Help system

**Success Criteria**:
```scheme
eshkol> %help derivative
derivative: Compute derivative of function
  (derivative f x) => df/dx at point x

eshkol> %timeit (fib 30)
Execution time: 2.3ms ± 0.1ms (1000 runs)

eshkol> %save session.esk
Session saved to session.esk
```

### Phase 4: Basic Emacs Integration (v1.3)
**Duration**: 4 weeks (40-50 hours)
**Deliverables**:
- `eshkol-mode.el` (major mode)
- `eshkol-repl.el` (comint-based REPL)
- `eshkol-company.el` (completion)
- `eshkol-flycheck.el` (syntax checking)
- Documentation

**Success Criteria**:
```elisp
M-x eshkol-repl RET
;; REPL opens in Emacs

;; In source buffer:
C-c C-c  ; Evaluates current form
C-c C-r  ; Evaluates region
C-c C-l  ; Loads file
```

### Phase 5: Visualization Engine (v1.4)
**Duration**: 4 weeks (40-60 hours)
**Deliverables**:
- Gradient visualization
- Computation graph rendering
- Plot generation
- Terminal graphics (Unicode art)
- Emacs inline images

**Success Criteria**:
```scheme
eshkol> (visualize-gradient (lambda (x) (* x x)) 5.0)
[Shows gradient visualization in Emacs or terminal]

eshkol> (plot-function (lambda (x) (* x x)) -5.0 5.0)
[Shows plot]
```

### Phase 6: Live Coding Engine (v1.5)
**Duration**: 6 weeks (60-80 hours)
**Deliverables**:
- Hot reload
- State inspection
- Breakpoints
- Time-travel debugging
- Interactive training

**Success Criteria**:
```scheme
eshkol> (define (f x) (* x 2))
eshkol> (start-process f)
Process started

;; Modify f while running
eshkol> (redefine (f x) (* x 3))
Hot reload complete

;; Process now uses new definition!
```

### Phase 7: Advanced Features (v2.0+)
**Duration**: Ongoing
**Features**:
- Distributed REPL (run on cluster)
- GPU visualization
- Notebook interface (Jupyter kernel)
- VSCode extension
- Web-based interface
- AI-assisted coding (completion using Eshkol's own neural networks!)

---

## Advanced Features Roadmap

### Live Neural Network Training

```scheme
eshkol> (define model (make-mlp [2 4 1]))

eshkol> (train-interactive model xor-data
          :watch-gradients     ; Show gradient flow
          :plot-loss          ; Plot loss curve
          :plot-weights       ; Show weight evolution
          :update-rate 10)    ; Update display every 10 epochs

[Terminal splits into 4 panels]
┌────────────────┬────────────────┐
│ Network Graph  │  Loss Curve    │
│                │   ^            │
│    ●──●──●    │   │ \          │
│   ╱│╲╱│╲╱│   │   │  \___      │
│  ● ●─● ●─●    │   │      ----  │
│                │   └──────────> │
├────────────────┼────────────────┤
│ Gradient Flow  │  Console       │
│                │                │
│ Layer 0: ████  │ Epoch 0: ...   │
│ Layer 1: ██    │ Epoch 10: ...  │
│ Layer 2: █     │                │
└────────────────┴────────────────┘
```

### HolyC-Style System Integration

**Vision**: Direct system interaction like TempleOS

```scheme
;; Draw directly to framebuffer
eshkol> (graphics-mode 800 600)
Graphics mode enabled

eshkol> (draw-pixel 400 300 #xFF0000)  ; Red pixel at center

eshkol> (define (mandelbrot-live)
          (for-each-pixel (lambda (x y)
            (let ((c (compute-mandelbrot x y)))
              (draw-pixel x y c)))))

eshkol> (mandelbrot-live)
[Renders Mandelbrot set in real-time]

;; System calls
eshkol> (syscall 'write 1 "Hello\n" 6)  ; Direct syscall!
Hello
6

;; Memory inspection
eshkol> (peek-memory 0x1000 16)  ; Read 16 bytes at address
#(00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F)
```

### Distributed REPL

```scheme
;; Connect to remote compiler
eshkol> (connect-remote "gpu-cluster.example.com")
Connected to remote compiler (8 GPUs available)

;; Execute on remote
eshkol> (with-remote
          (train-large-model model huge-dataset
            :use-gpus 8))
[Training on remote cluster...]
Epoch 0: Loss 2.45 (GPU utilization: 98%)
...

;; Fetch results
eshkol> (fetch-model)
Model synced to local
```

### AI-Assisted Development

```scheme
;; Autocomplete using neural network trained on Eshkol code!
eshkol> (define (train-█
        ^^^ Suggestions:
        train-step (65% confidence)
        train-model (23% confidence)
        train-network (12% confidence)

;; Explain code
eshkol> (explain '(gradient loss-fn weights))
"This computes the gradient of the loss function with respect
to the weights. The gradient indicates the direction of steepest
ascent, so we typically move in the opposite direction to minimize
the loss."

;; Generate tests
eshkol> (generate-tests train-step)
Generated 5 test cases:
  (test "zero gradient at minimum" ...)
  (test "positive gradient when below minimum" ...)
  ...
```

---

## Technical Specifications

### JSON-RPC Protocol

**Transport**: Unix domain socket (local) or TCP (remote)

**Socket path**: `/tmp/eshkol-compiler.sock`

**Message format**:
```json
{
  "jsonrpc": "2.0",
  "method": "method_name",
  "params": { ... },
  "id": 123
}
```

**Methods**:

#### `parse`
```json
{
  "method": "parse",
  "params": {
    "source": "(+ 1 2 3)"
  }
}
// Response:
{
  "result": {
    "ast": { ... },
    "errors": []
  }
}
```

#### `compile`
```json
{
  "method": "compile",
  "params": {
    "ast": { ... },
    "optimize": true
  }
}
// Response:
{
  "result": {
    "ir": "define i64 @add...",
    "module_id": "abc123"
  }
}
```

#### `execute`
```json
{
  "method": "execute",
  "params": {
    "module_id": "abc123",
    "function": "main",
    "args": []
  }
}
// Response:
{
  "result": {
    "value": 6,
    "type": "int64",
    "execution_time_ms": 0.15
  }
}
```

#### `compute_gradient`
```json
{
  "method": "compute_gradient",
  "params": {
    "function_ast": { ... },
    "point": [1.0, 2.0]
  }
}
// Response:
{
  "result": {
    "gradient": [2.0, 4.0],
    "module_id": "gradient_abc123"
  }
}
```

### Session Storage Format

**File**: `~/.eshkol/sessions/default.json`

```json
{
  "session_id": "default",
  "created": "2025-11-27T10:30:00Z",
  "bindings": {
    "x": {
      "type": "int64",
      "value": 10
    },
    "f": {
      "type": "function",
      "module_id": "func_f_123",
      "source": "(lambda (x) (* x x))"
    }
  },
  "history": [
    {
      "input": "(define x 10)",
      "output": { "type": "int64", "value": 10 },
      "timestamp": "2025-11-27T10:31:00Z"
    }
  ]
}
```

### Visualization Data Format

**Protocol**: Send visualization data over JSON-RPC

```json
{
  "method": "visualize",
  "params": {
    "type": "gradient_flow",
    "data": {
      "layers": [
        {
          "name": "input",
          "nodes": [...]
        },
        {
          "name": "hidden",
          "nodes": [...],
          "gradients": [0.12, 0.34, ...]
        }
      ]
    },
    "output_mode": "emacs"  // or "terminal", "web", "file"
  }
}
```

**Emacs rendering**: Send SVG as base64
**Terminal rendering**: Send Unicode art as string

---

## References & Inspiration

### Edwin (MIT Scheme)
- Emacs-based Scheme IDE
- Tight integration with Scheme runtime
- Inspector, debugger, REPL all in Emacs

### IPython
- Rich REPL with magic commands
- Inline visualization
- Jupyter notebook integration
- History, completion, introspection

### GHCi (GHC Haskell)
- Type introspection (`:t` command)
- Load/reload modules
- Powerful debugging (`:break`, `:step`)

### Julia REPL
- Multiple modes (julia>, shell>, help?>)
- Package manager integrated
- Plotting inline
- Performance analysis tools

### HolyC / TempleOS
- Direct system access
- Live graphics programming
- Immediate feedback
- System feels "alive"

### SLIME (Superior Lisp Interaction Mode for Emacs)
- Gold standard for Lisp IDE integration
- Inspector, debugger, profiler
- Cross-referencing, documentation lookup
- Trace facility

---

## Success Metrics

### Phase 1 Success
- [ ] Can evaluate basic expressions
- [ ] Can define variables and functions
- [ ] History works
- [ ] Doesn't crash

### Phase 4 Success (Emacs)
- [ ] Can send code from source buffer to REPL
- [ ] Completion works
- [ ] Syntax highlighting works
- [ ] Error navigation works

### Phase 6 Success (Live Coding)
- [ ] Can hot-reload function while program runs
- [ ] Can inspect running state
- [ ] Can set breakpoints
- [ ] Can time-travel debug

### Ultimate Success
**Can develop and train a neural network entirely in the REPL with live visualization, hot-reloading, and it feels AMAZING.**

---

## Next Steps

1. **Review this document** - Does the vision align with your goals?
2. **Refine architecture** - Any changes needed?
3. **Create API specification** - Detail each API
4. **Create protocol specification** - Detail JSON-RPC messages
5. **Start Phase 1 implementation** - Build minimal REPL

---

**This is the foundation for something truly unique in the ML/AI development space.**

The combination of:
- Automatic differentiation
- Live coding
- Visual feedback
- LLVM performance
- Emacs integration

...creates an environment that doesn't exist anywhere else.

**Ready to make this real?** 🚀
