# Eshkol Neuro-Symbolic Programming: Complete Architecture Report

## Abstract

This report presents a comprehensive architecture for transforming Eshkol into a fully-featured neuro-symbolic programming language. Based on analysis of the existing compiler implementation (29,352 lines of LLVM codegen), we identify Eshkol's exceptional strengths in differentiable computing and outline the precise additions required for complete symbolic reasoning integration. The proposed architecture enables seamless bidirectional flow between neural (subsymbolic, differentiable) and symbolic (discrete, logical) computation.

---

## Part I: Foundation Analysis

### 1.1 What Neuro-Symbolic AI Requires

Neuro-symbolic AI systems must unify two historically separate paradigms:

| Paradigm | Strengths | Weaknesses |
|----------|-----------|------------|
| **Neural (Connectionist)** | Learning from data, pattern recognition, generalization, handling noise | Black-box, data-hungry, no explicit reasoning, hallucination |
| **Symbolic (Classical AI)** | Explicit reasoning, interpretability, compositionality, few-shot learning | Brittleness, knowledge acquisition bottleneck, no learning |

A complete neuro-symbolic system requires:

1. **Neural computation** - Differentiable tensor operations, gradient-based learning
2. **Symbolic computation** - Logic, inference, knowledge representation
3. **Bidirectional bridge** - Neural→Symbolic (perception to symbols) and Symbolic→Neural (reasoning guides learning)
4. **Differentiable interface** - Gradients flow through symbolic operations
5. **Unified representation** - Common substrate for both paradigms

### 1.2 Eshkol's Current Position

Based on compiler source code analysis:

#### 1.2.1 Neural/Differentiable Computing (COMPLETE)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ESHKOL AUTODIFF STACK                        │
├─────────────────────────────────────────────────────────────────┤
│  SYMBOLIC DIFFERENTIATION (diff)                                │
│  ├── AST → AST transformation at compile time                   │
│  ├── Rules: sum, product, quotient, chain                       │
│  ├── Functions: sin, cos, exp, log, sqrt, pow                   │
│  └── Returns S-expressions: (diff (* x x) x) → (* 2 x)          │
├─────────────────────────────────────────────────────────────────┤
│  FORWARD-MODE AD (derivative)                                   │
│  ├── Dual numbers: (value, derivative) pairs                    │
│  ├── Operator overloading through type dispatch                 │
│  └── Efficient for f: R → R^n (few inputs, many outputs)        │
├─────────────────────────────────────────────────────────────────┤
│  REVERSE-MODE AD (gradient, jacobian, hessian)                  │
│  ├── Tape-based computational graph                             │
│  ├── AD nodes track operations and inputs                       │
│  ├── Backward pass accumulates gradients                        │
│  ├── Nested gradient support (tape stack, depth 32)             │
│  └── Efficient for f: R^n → R (many inputs, few outputs)        │
├─────────────────────────────────────────────────────────────────┤
│  VECTOR CALCULUS                                                │
│  ├── gradient: ∇f (partial derivatives)                         │
│  ├── jacobian: J_F (matrix of partials)                         │
│  ├── hessian: H_f (second derivatives)                          │
│  ├── divergence: ∇·F (scalar from vector field)                 │
│  ├── curl: ∇×F (rotation of vector field)                       │
│  ├── laplacian: ∇²f (sum of second partials)                    │
│  └── directional-derivative: D_v f                              │
├─────────────────────────────────────────────────────────────────┤
│  TENSOR OPERATIONS                                              │
│  ├── Creation: zeros, ones, eye, arange, linspace, reshape      │
│  ├── Arithmetic: tensor-add, tensor-sub, tensor-mul, tensor-div │
│  ├── Linear algebra: matmul, transpose, tensor-dot              │
│  ├── Reduction: tensor-sum, tensor-mean                         │
│  └── Access: vref, flatten                                      │
└─────────────────────────────────────────────────────────────────┘
```

**Code evidence** (from llvm_codegen.cpp):
- Lines 17750-18200: Symbolic differentiation (`buildSymbolicDerivative`)
- Lines 18440-18800: Dual number operations
- Lines 19060-19950: Computational graph construction
- Lines 19954-21100: Gradient operator implementation
- Lines 21102-23200: Jacobian, Hessian, vector calculus operators

#### 1.2.2 Functional Programming Infrastructure (COMPLETE)

```
┌─────────────────────────────────────────────────────────────────┐
│                 ESHKOL FUNCTIONAL CORE                          │
├─────────────────────────────────────────────────────────────────┤
│  FIRST-CLASS FUNCTIONS                                          │
│  ├── Lambda expressions with lexical scoping                    │
│  ├── Closure environments (up to 32 captures)                   │
│  ├── Variadic functions                                         │
│  └── Higher-order function support                              │
├─────────────────────────────────────────────────────────────────┤
│  HOMOICONICITY                                                  │
│  ├── Code represented as data (S-expressions)                   │
│  ├── quote returns AST as runtime list                          │
│  ├── Lambda S-expression preservation for introspection         │
│  └── Foundation for metaprogramming                             │
├─────────────────────────────────────────────────────────────────┤
│  STANDARD LIBRARY (stdlib.esk)                                  │
│  ├── Combinators: compose, identity, constantly, flip           │
│  ├── Currying: curry2, curry3, uncurry2, partial2               │
│  ├── List ops: map, filter, fold, for-each, sort                │
│  ├── Predicates: all?, none?, count-if, partition               │
│  └── Generators: iota, repeat                                   │
├─────────────────────────────────────────────────────────────────┤
│  MATH LIBRARY (math.esk)                                        │
│  ├── Linear algebra: det, inv, solve, cross, dot, normalize     │
│  ├── Eigenvalues: power-iteration                               │
│  ├── Numerical: integrate (Simpson), newton (root-finding)      │
│  └── Statistics: variance, std, covariance                      │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.2.3 Runtime Infrastructure (COMPLETE)

```
┌─────────────────────────────────────────────────────────────────┐
│                   ESHKOL RUNTIME                                │
├─────────────────────────────────────────────────────────────────┤
│  TAGGED VALUE SYSTEM                                            │
│  ├── 16-byte tagged values (type + flags + 8-byte data)         │
│  ├── Type tags: INT64, DOUBLE, STRING_PTR, CONS_PTR,            │
│  │              VECTOR_PTR, TENSOR_PTR, CLOSURE_PTR,            │
│  │              AD_NODE_PTR, LAMBDA_SEXPR, NULL                 │
│  └── Runtime type dispatch for polymorphism                     │
├─────────────────────────────────────────────────────────────────┤
│  ARENA MEMORY MANAGEMENT                                        │
│  ├── Stack-based allocation (no GC needed)                      │
│  ├── Scope-based deallocation                                   │
│  ├── Specialized allocators: cons cells, AD nodes, closures     │
│  └── Shared arena for REPL persistence                          │
├─────────────────────────────────────────────────────────────────┤
│  JIT COMPILATION (LLVM ORC)                                     │
│  ├── Incremental compilation per expression                     │
│  ├── Cross-module symbol persistence                            │
│  ├── Runtime symbol registration                                │
│  └── Hot code execution                                         │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.2.4 What's Missing (SYMBOLIC REASONING)

```
┌─────────────────────────────────────────────────────────────────┐
│              MISSING: SYMBOLIC REASONING STACK                  │
├─────────────────────────────────────────────────────────────────┤
│  LOGIC PROGRAMMING                                              │
│  ├── Unification engine                                         │
│  ├── Logic variables (?x, ?y)                                   │
│  ├── Backtracking search                                        │
│  └── Cut, negation-as-failure                                   │
├─────────────────────────────────────────────────────────────────┤
│  KNOWLEDGE REPRESENTATION                                       │
│  ├── Fact database                                              │
│  ├── Rule definitions                                           │
│  ├── Knowledge graphs (triple store)                            │
│  └── Ontology support                                           │
├─────────────────────────────────────────────────────────────────┤
│  INFERENCE ENGINES                                              │
│  ├── Forward chaining                                           │
│  ├── Backward chaining                                          │
│  ├── Resolution                                                 │
│  └── Constraint propagation                                     │
├─────────────────────────────────────────────────────────────────┤
│  NEURAL-SYMBOLIC BRIDGE                                         │
│  ├── Differentiable symbolic operations                         │
│  ├── Symbol embeddings                                          │
│  ├── Attention over knowledge                                   │
│  └── Gradient estimators for discrete ops                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part II: Complete Architecture Design

### 2.1 Architectural Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ESHKOL NEURO-SYMBOLIC ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        USER PROGRAMS                                 │   │
│  │   (define (intelligent-agent percept)                               │   │
│  │     (let* ((symbols (perceive percept))        ; Neural→Symbolic    │   │
│  │            (plan (reason symbols kb))          ; Symbolic reasoning │   │
│  │            (action (decide plan policy)))      ; Neural policy      │   │
│  │       action))                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    NEURO-SYMBOLIC INTERFACE                         │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │   │
│  │  │   perceive    │  │    reason     │  │    decide     │           │   │
│  │  │ (Neural→Sym)  │  │  (Symbolic)   │  │ (Sym→Neural)  │           │   │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘           │   │
│  │          │                  │                  │                    │   │
│  │          ▼                  ▼                  ▼                    │   │
│  │  ┌───────────────────────────────────────────────────────────┐     │   │
│  │  │              DIFFERENTIABLE BRIDGE LAYER                  │     │   │
│  │  │  • Soft unification    • Attention over KB                │     │   │
│  │  │  • Symbol embeddings   • Gradient estimators              │     │   │
│  │  └───────────────────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│          ┌─────────────────────────┴─────────────────────────┐             │
│          ▼                                                   ▼             │
│  ┌───────────────────────────────┐   ┌───────────────────────────────┐    │
│  │      NEURAL SUBSTRATE         │   │     SYMBOLIC SUBSTRATE        │    │
│  │  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐  │    │
│  │  │ Autodiff (3 modes)      │  │   │  │ Unification Engine      │  │    │
│  │  │ • Symbolic (diff)       │  │   │  │ • Pattern matching      │  │    │
│  │  │ • Forward (derivative)  │  │   │  │ • Variable binding      │  │    │
│  │  │ • Reverse (gradient)    │  │   │  │ • Occurs check          │  │    │
│  │  └─────────────────────────┘  │   │  └─────────────────────────┘  │    │
│  │  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐  │    │
│  │  │ Tensor Operations       │  │   │  │ Knowledge Base          │  │    │
│  │  │ • matmul, transpose     │  │   │  │ • Facts & rules         │  │    │
│  │  │ • tensor-dot, reshape   │  │   │  │ • Triple store          │  │    │
│  │  │ • Broadcasting          │  │   │  │ • Indexing              │  │    │
│  │  └─────────────────────────┘  │   │  └─────────────────────────┘  │    │
│  │  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐  │    │
│  │  │ Vector Calculus         │  │   │  │ Inference Engines       │  │    │
│  │  │ • ∇, ∇·, ∇×, ∇²        │  │   │  │ • Forward chaining      │  │    │
│  │  │ • Jacobian, Hessian     │  │   │  │ • Backward chaining     │  │    │
│  │  │ • Directional deriv     │  │   │  │ • Constraint solving    │  │    │
│  │  └─────────────────────────┘  │   │  └─────────────────────────┘  │    │
│  └───────────────────────────────┘   └───────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      RUNTIME FOUNDATION                             │   │
│  │  • Tagged values (16-byte, type-dispatched)                         │   │
│  │  • Arena memory management (GC-free)                                │   │
│  │  • JIT compilation (LLVM ORC)                                       │   │
│  │  • Homoiconicity (code = data)                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Specifications

#### 2.2.1 Unification Engine

The unification engine is the foundation of symbolic reasoning.

**Data Structures:**

```scheme
;; Logic variable representation
;; Prefix '?' indicates unbound variable
;; Internal: tagged value with LOGIC_VAR type

(define-type LogicVar
  (name : Symbol)
  (id : Int64))        ;; Unique identifier for occurs check

;; Substitution: mapping from variables to terms
(define-type Substitution
  (bindings : (HashMap LogicVar Term)))

;; Term: recursive structure
(define-type Term
  (variant
    (Var LogicVar)
    (Atom Symbol)
    (Num Number)
    (Compound Symbol (List Term))))
```

**Core Algorithm:**

```scheme
;; Robinson's unification algorithm with occurs check
(define (unify term1 term2 subst)
  (let ((t1 (walk term1 subst))
        (t2 (walk term2 subst)))
    (cond
      ;; Same term - success
      ((equal? t1 t2) subst)

      ;; Variable cases
      ((logic-var? t1) (extend-subst t1 t2 subst))
      ((logic-var? t2) (extend-subst t2 t1 subst))

      ;; Compound terms - unify recursively
      ((and (compound? t1) (compound? t2)
            (eq? (functor t1) (functor t2))
            (= (arity t1) (arity t2)))
       (unify-args (args t1) (args t2) subst))

      ;; Failure
      (else #f))))

;; Walk substitution chain to find actual binding
(define (walk term subst)
  (if (and (logic-var? term) (bound? term subst))
      (walk (lookup term subst) subst)
      term))

;; Extend substitution with occurs check
(define (extend-subst var term subst)
  (if (occurs? var term subst)
      #f  ;; Infinite term - fail
      (subst-add subst var term)))
```

**Implementation in Compiler:**

New AST node type: `ESHKOL_LOGIC_VAR`
New value type: `ESHKOL_VALUE_LOGIC_VAR` (add to 4-bit type enum)
New operations: `ESHKOL_UNIFY_OP`, `ESHKOL_QUERY_OP`

```cpp
// In eshkol.h
typedef enum {
    // ... existing types ...
    ESHKOL_VALUE_LOGIC_VAR = 12,    // Logic variable
    ESHKOL_VALUE_SUBST_PTR = 13,    // Substitution pointer
} eshkol_value_type_t;

// In llvm_codegen.cpp
Value* codegenUnify(const eshkol_operations_t* op) {
    // Generate code for unification
    // Returns tagged Substitution or NULL (failure)
}
```

#### 2.2.2 Knowledge Base

**Fact Storage:**

```scheme
;; Fact: ground term (no variables)
(define-type Fact
  (predicate : Symbol)
  (args : (Vector Term)))

;; Rule: head :- body (Horn clause)
(define-type Rule
  (head : Term)
  (body : (List Term)))  ;; Conjunction of goals

;; Knowledge Base
(define-type KnowledgeBase
  (facts : (HashMap Symbol (List Fact)))     ;; Indexed by predicate
  (rules : (HashMap Symbol (List Rule)))     ;; Indexed by head predicate
  (triples : (HashMap Symbol (List Triple))) ;; For graph queries
)
```

**Triple Store (Knowledge Graph):**

```scheme
;; RDF-style triple
(define-type Triple
  (subject : Term)
  (predicate : Symbol)
  (object : Term))

;; Indexing for efficient queries
;; SPO index: subject → predicate → objects
;; POS index: predicate → object → subjects
;; OSP index: object → subject → predicates
```

**Operations:**

```scheme
;; Assertion
(assert! kb (parent alice bob))
(assert-triple! kb alice :knows bob)

;; Queries
(query kb (parent ?x bob))           ;; → ({?x: alice})
(query-triple kb ?who :knows bob)    ;; → ({?who: alice})

;; Rule definition
(define-rule kb (grandparent ?x ?z)
  (parent ?x ?y)
  (parent ?y ?z))
```

#### 2.2.3 Inference Engines

**Backward Chaining (Goal-Directed):**

```scheme
(define (prove goal kb subst)
  ;; Try facts first
  (let ((fact-solutions (match-facts goal kb subst)))
    (if (not (null? fact-solutions))
        fact-solutions
        ;; Try rules
        (apply append
          (map (lambda (rule)
                 (let ((renamed (rename-vars rule)))
                   (let ((head-subst (unify goal (rule-head renamed) subst)))
                     (if head-subst
                         (prove-all (rule-body renamed) kb head-subst)
                         '()))))
               (lookup-rules (functor goal) kb))))))

(define (prove-all goals kb subst)
  (if (null? goals)
      (list subst)  ;; Success
      (apply append
        (map (lambda (s) (prove-all (cdr goals) kb s))
             (prove (car goals) kb subst)))))
```

**Forward Chaining (Data-Driven):**

```scheme
(define (forward-chain kb)
  (let loop ((new-facts '())
             (changed #t))
    (if (not changed)
        kb  ;; Fixed point reached
        (let ((derived (derive-new-facts kb)))
          (if (null? derived)
              kb
              (loop derived
                    (add-facts! kb derived)))))))

(define (derive-new-facts kb)
  (apply append
    (map (lambda (rule)
           (let ((bindings (match-body (rule-body rule) kb)))
             (map (lambda (subst)
                    (apply-subst subst (rule-head rule)))
                  bindings)))
         (all-rules kb))))
```

#### 2.2.4 Differentiable Bridge Layer

This is the **critical innovation** for neuro-symbolic AI.

**Symbol Embeddings:**

```scheme
;; Learnable embedding table
(define-type SymbolEmbedding
  (table : (HashMap Symbol (Tensor Float)))
  (dim : Int64))

;; Embed a symbol
(define (embed symbol embeddings)
  (hash-ref (embedding-table embeddings) symbol
            (random-embedding (embedding-dim embeddings))))

;; Embed a term (recursive)
(define (embed-term term embeddings)
  (cond
    ((symbol? term) (embed term embeddings))
    ((number? term) (number-embedding term))
    ((compound? term)
     (compose-embeddings
       (embed (functor term) embeddings)
       (map (lambda (arg) (embed-term arg embeddings))
            (args term))))))
```

**Soft Unification:**

```scheme
;; Differentiable unification - returns similarity score with gradient
(define (soft-unify term1 term2 embeddings)
  (let ((e1 (embed-term term1 embeddings))
        (e2 (embed-term term2 embeddings)))
    ;; Cosine similarity (differentiable)
    (/ (tensor-dot e1 e2)
       (* (tensor-norm e1) (tensor-norm e2)))))

;; Soft substitution - weighted combination
(define (soft-extend-subst var term weight subst)
  (let ((current (soft-lookup var subst)))
    (weighted-combine current term weight)))
```

**Attention Over Knowledge:**

```scheme
;; Neural attention over facts in KB
(define (attend-facts query kb embeddings)
  (let* ((query-emb (embed-term query embeddings))
         (fact-embs (map (lambda (f) (embed-term f embeddings))
                         (all-facts kb)))
         ;; Compute attention scores
         (scores (map (lambda (f-emb)
                        (tensor-dot query-emb f-emb))
                      fact-embs))
         ;; Softmax for probability distribution
         (weights (softmax scores)))
    ;; Weighted combination of fact embeddings
    (weighted-sum fact-embs weights)))

;; Differentiable KB query
(define (neural-query query kb embeddings)
  (let* ((attention (attend-facts query kb embeddings))
         ;; Decode attention into symbolic answer
         (answer (decode-attention attention embeddings)))
    answer))
```

**Gradient Estimators:**

```scheme
;; Gumbel-Softmax for differentiable discrete sampling
(define (gumbel-softmax logits temperature)
  (let* ((gumbels (map (lambda (_) (- (log (- (log (random))))))
                       logits))
         (perturbed (map + logits gumbels)))
    (softmax (map (lambda (x) (/ x temperature)) perturbed))))

;; Straight-through estimator
;; Forward: hard argmax
;; Backward: soft gradients
(define (straight-through logits)
  (let* ((soft (softmax logits))
         (hard (one-hot (argmax logits) (length logits))))
    ;; Custom gradient: use soft for backward pass
    (stop-gradient (- hard soft)) + soft))
```

#### 2.2.5 Program Synthesis

**Type-Directed Synthesis:**

```scheme
;; Typed hole - placeholder for synthesis
(define-syntax ??
  (syntax-rules ()
    [(?? type) (synthesis-hole 'type)]))

;; Example: synthesize a function
(define unknown-fn (?? (-> Number Number)))

;; Synthesis from examples
(define (synthesize-from-examples examples type)
  (let ((candidates (enumerate-programs type 5)))  ;; depth 5
    (find (lambda (prog)
            (all? (lambda (ex)
                    (equal? (eval prog (car ex)) (cdr ex)))
                  examples))
          candidates)))
```

**Neural-Guided Search:**

```scheme
;; Neural network scores program candidates
(define (neural-guided-synthesis spec heuristic-net)
  (let loop ((frontier (initial-programs spec))
             (best #f)
             (best-score -inf))
    (if (null? frontier)
        best
        (let* ((scores (map (lambda (p)
                              (neural-score heuristic-net p spec))
                            frontier))
               (sorted (sort-by-score frontier scores))
               (top (car sorted)))
          (if (satisfies-spec? top spec)
              top
              (loop (expand-programs (take 10 sorted))
                    (if (> (car scores) best-score) top best)
                    (max (car scores) best-score)))))))
```

---

## Part III: Implementation Specification

### 3.1 New Type System Extensions

```cpp
// New value types in eshkol.h
typedef enum {
    ESHKOL_VALUE_NULL = 0,
    ESHKOL_VALUE_INT64 = 1,
    ESHKOL_VALUE_DOUBLE = 2,
    ESHKOL_VALUE_STRING_PTR = 3,
    ESHKOL_VALUE_CONS_PTR = 4,
    ESHKOL_VALUE_VECTOR_PTR = 5,
    ESHKOL_VALUE_TENSOR_PTR = 6,
    ESHKOL_VALUE_CLOSURE_PTR = 7,
    ESHKOL_VALUE_AD_NODE_PTR = 8,
    ESHKOL_VALUE_LAMBDA_SEXPR = 9,
    // NEW: Symbolic reasoning types
    ESHKOL_VALUE_LOGIC_VAR = 10,      // Logic variable ?x
    ESHKOL_VALUE_SUBST_PTR = 11,      // Substitution map
    ESHKOL_VALUE_FACT_PTR = 12,       // Fact/rule pointer
    ESHKOL_VALUE_KB_PTR = 13,         // Knowledge base pointer
    ESHKOL_VALUE_EMBEDDING_PTR = 14,  // Symbol embedding table
    // Future: probabilistic types
    ESHKOL_VALUE_DIST_PTR = 15,       // Probability distribution
} eshkol_value_type_t;
```

### 3.2 New AST Node Types

```cpp
// New operation types in eshkol.h
typedef enum {
    // ... existing ops ...

    // Logic programming
    ESHKOL_UNIFY_OP,           // (unify term1 term2)
    ESHKOL_QUERY_OP,           // (query goal kb)
    ESHKOL_ASSERT_OP,          // (assert! fact kb)
    ESHKOL_RETRACT_OP,         // (retract! fact kb)

    // Knowledge base
    ESHKOL_DEFINE_RULE_OP,     // (define-rule head body...)
    ESHKOL_KB_CREATE_OP,       // (make-kb)
    ESHKOL_KB_QUERY_OP,        // (kb-query pattern kb)

    // Neural-symbolic bridge
    ESHKOL_SOFT_UNIFY_OP,      // (soft-unify t1 t2 embeddings)
    ESHKOL_ATTEND_OP,          // (attend query kb embeddings)
    ESHKOL_EMBED_OP,           // (embed symbol table)

    // Gradient estimators
    ESHKOL_GUMBEL_SOFTMAX_OP,  // (gumbel-softmax logits temp)
    ESHKOL_STRAIGHT_THROUGH_OP,// (straight-through logits)

    // Program synthesis
    ESHKOL_HOLE_OP,            // (?? type)
    ESHKOL_SYNTHESIZE_OP,      // (synthesize examples type)
} eshkol_op_t;
```

### 3.3 New Runtime Structures

```cpp
// Logic variable structure
typedef struct {
    char* name;           // Variable name (e.g., "x" for ?x)
    uint64_t id;          // Unique ID for occurs check
} eshkol_logic_var_t;

// Substitution structure (hash map)
typedef struct {
    uint64_t num_bindings;
    eshkol_logic_var_t* vars;
    eshkol_tagged_value_t* terms;
} eshkol_substitution_t;

// Fact structure
typedef struct {
    char* predicate;
    uint64_t arity;
    eshkol_tagged_value_t* args;
} eshkol_fact_t;

// Rule structure
typedef struct {
    eshkol_fact_t* head;
    uint64_t num_body;
    eshkol_fact_t* body;
} eshkol_rule_t;

// Knowledge base structure
typedef struct {
    // Fact index: predicate -> facts
    uint64_t num_predicates;
    char** predicates;
    eshkol_fact_t** fact_lists;
    uint64_t* fact_counts;

    // Rule index: head predicate -> rules
    eshkol_rule_t** rule_lists;
    uint64_t* rule_counts;

    // Triple index (for graph queries)
    // SPO, POS, OSP indexes
} eshkol_knowledge_base_t;

// Symbol embedding table
typedef struct {
    uint64_t num_symbols;
    char** symbols;
    double** embeddings;  // Each embedding is a vector
    uint64_t embedding_dim;
} eshkol_embedding_table_t;
```

### 3.4 Arena Memory Extensions

```cpp
// In arena_memory.h

// Logic variable allocation
eshkol_logic_var_t* arena_allocate_logic_var(arena_t* arena, const char* name);

// Substitution allocation
eshkol_substitution_t* arena_allocate_substitution(arena_t* arena, size_t capacity);

// Knowledge base allocation
eshkol_knowledge_base_t* arena_allocate_kb(arena_t* arena, size_t initial_capacity);

// Embedding table allocation
eshkol_embedding_table_t* arena_allocate_embedding_table(
    arena_t* arena, size_t num_symbols, size_t embedding_dim);
```

### 3.5 Codegen Functions

```cpp
// In llvm_codegen.cpp

// Unification
Value* codegenUnify(const eshkol_operations_t* op);
Value* codegenWalk(Value* term, Value* subst);
Value* codegenExtendSubst(Value* var, Value* term, Value* subst);
Value* codegenOccursCheck(Value* var, Value* term, Value* subst);

// Knowledge base operations
Value* codegenAssert(const eshkol_operations_t* op);
Value* codegenQuery(const eshkol_operations_t* op);
Value* codegenDefineRule(const eshkol_operations_t* op);

// Inference
Value* codegenProve(Value* goal, Value* kb, Value* subst);
Value* codegenForwardChain(Value* kb);

// Neural-symbolic bridge
Value* codegenSoftUnify(const eshkol_operations_t* op);
Value* codegenAttend(const eshkol_operations_t* op);
Value* codegenEmbed(const eshkol_operations_t* op);

// Gradient estimators
Value* codegenGumbelSoftmax(const eshkol_operations_t* op);
Value* codegenStraightThrough(const eshkol_operations_t* op);
```

---

## Part IV: Integration with Existing Systems

### 4.1 Integration with Autodiff

The differentiable bridge must integrate with all three autodiff modes:

```scheme
;; Symbolic diff of soft-unify
(diff (soft-unify (f ?x) (f 1) emb) ?x)
;; → Derivative of similarity w.r.t. embedding parameters

;; Gradient through knowledge attention
(gradient
  (lambda (params)
    (let ((emb (make-embeddings params)))
      (attend '(parent ?x bob) kb emb)))
  initial-params)
;; → Gradients for updating embeddings

;; Jacobian of rule application
(jacobian
  (lambda (emb-vec)
    (soft-apply-rule rule facts (vec->embeddings emb-vec)))
  embedding-vector)
```

### 4.2 Integration with Type System

When HoTT types are implemented, symbolic reasoning gets dependent types:

```scheme
;; Dependent type for proofs
(: Proof (Π (goal : Term) (kb : KnowledgeBase) Type))

;; Query returns proof object
(: query (Π (g : Term) (kb : KnowledgeBase) (Maybe (Proof g kb))))

;; Type-safe rule definitions
(: define-rule
   (Π (head : Term)
      (body : (List Term))
      (-> KnowledgeBase KnowledgeBase)))
```

### 4.3 Integration with Homoiconicity

Symbolic reasoning leverages code-as-data:

```scheme
;; Rules are data, can be manipulated
(define rule1 '(grandparent ?x ?z) '((parent ?x ?y) (parent ?y ?z)))

;; Meta-reasoning: reason about rules
(query '(implies (rule ?name ?head ?body)
                 (derivable ?head))
       meta-kb)

;; Self-modifying knowledge base
(define (learn-rule examples kb)
  (let ((induced-rule (induce-rule examples)))
    (assert! induced-rule kb)))
```

---

## Part V: Use Cases and Applications

### 5.1 Neuro-Symbolic Theorem Proving

```scheme
;; Neural network guides proof search
(define (neural-theorem-prover goal kb neural-heuristic)
  (let loop ((frontier (list (make-proof-state goal)))
             (visited (make-hash)))
    (if (null? frontier)
        #f  ;; No proof found
        (let* ((state (car frontier))
               (goal (current-goal state)))
          (if (proved? state)
              (extract-proof state)
              ;; Neural heuristic ranks next steps
              (let* ((tactics (applicable-tactics goal kb))
                     (scores (map (lambda (t)
                                    (neural-score neural-heuristic state t))
                                  tactics))
                     (sorted (sort-by-score tactics scores))
                     (new-states (apply-tactics sorted state)))
                (loop (merge-frontiers new-states (cdr frontier))
                      (hash-set visited goal #t))))))))
```

### 5.2 Self-Improving Code

```scheme
;; Program improves itself through gradient descent on performance
(define (self-improve program examples iterations)
  (let loop ((prog program) (i 0))
    (if (>= i iterations)
        prog
        (let* (;; Embed program as differentiable representation
               (prog-emb (embed-program prog))
               ;; Compute loss on examples
               (loss (program-loss prog-emb examples))
               ;; Gradient w.r.t. program embedding
               (grad (gradient (lambda (e) (program-loss e examples)) prog-emb))
               ;; Update embedding
               (new-emb (tensor-sub prog-emb (scalar-mul 0.01 grad)))
               ;; Decode back to program
               (new-prog (decode-program new-emb)))
          (loop new-prog (+ i 1))))))
```

### 5.3 Knowledge-Grounded Language Model

```scheme
;; Language model with knowledge base grounding
(define (grounded-generate prompt kb embeddings)
  (let loop ((tokens (tokenize prompt))
             (generated '()))
    (let* (;; Query KB for relevant facts
           (context-query (extract-query tokens))
           (relevant-facts (neural-query context-query kb embeddings))
           ;; Combine with neural LM
           (lm-logits (language-model tokens))
           (kb-bias (fact-to-logits relevant-facts))
           (combined (tensor-add lm-logits kb-bias))
           ;; Sample next token
           (next-token (sample (softmax combined))))
      (if (eos? next-token)
          (reverse generated)
          (loop (append tokens (list next-token))
                (cons next-token generated))))))
```

### 5.4 Explainable Neural Networks

```scheme
;; Neural network with symbolic explanation
(define (explainable-classifier input rules embeddings)
  (let* (;; Get neural prediction
         (logits (neural-network input))
         (prediction (argmax logits))
         ;; Find supporting rules
         (activated-rules
           (filter (lambda (rule)
                     (let ((relevance (soft-match input (rule-conditions rule) embeddings)))
                       (> relevance 0.5)))
                   rules))
         ;; Build explanation
         (explanation
           (map (lambda (rule)
                  (list 'because (rule-name rule)
                        'with-confidence (soft-match input (rule-conditions rule) embeddings)))
                activated-rules)))
    (list 'prediction prediction
          'explanation explanation)))
```

---

## Part VI: HoTT Type System Integration

The neuro-symbolic architecture builds on Eshkol's HoTT (Homotopy Type Theory) type system extension. See `HOTT_TYPE_SYSTEM_EXTENSION.md` for full specification.

### 6.1 Type System Foundation

The HoTT type system provides:

```
                         𝒰₂ (Universe of Propositions)
                            │
                            ▼
                         𝒰₁ (Universe of Types)
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
         Value          Resource        Proposition
            │               │               │
     ┌──────┼──────┐       │           ┌───┴───┐
     │      │      │       │           │       │
     ▼      ▼      ▼       ▼           ▼       ▼
  Number  Text  Collection Handle    Proof   Witness
```

### 6.2 Dependent Types for Neural-Symbolic Operations

```scheme
;; Typed knowledge base operations
(: assert! (Π (kb : KnowledgeBase) (fact : Fact) KnowledgeBase))
(: query (Π (kb : KnowledgeBase) (pattern : Term) (List Substitution)))

;; Dependent types for tensor shapes in neural operations
(: matmul (Π (m n k : Natural)
           (-> (Tensor Float m n) (Tensor Float n k) (Tensor Float m k))))

;; Refined types for probability distributions
(: softmax (-> (Vector Float n) (Refine (Vector Float n)
                                        (lambda (v) (= (sum v) 1.0)))))

;; Linear types for knowledge base updates
(: transaction (Π (kb : (Linear KnowledgeBase))
                 (ops : (List KBOperation))
                 (Linear KnowledgeBase)))
```

### 6.3 Proof-Carrying Inference

```scheme
;; Queries return proofs that can be type-checked
(: prove (Π (goal : Term) (kb : KnowledgeBase)
           (Maybe (Σ (subst : Substitution) (Proof (entails kb (apply subst goal)))))))

;; Forward chaining produces derivation certificates
(: forward-chain (Π (kb : KnowledgeBase) (rules : (List Rule))
                   (Σ (kb' : KnowledgeBase)
                      (Π (fact : Fact) (member fact kb')
                         (Derivation kb rules fact)))))
```

### 6.4 Type-Safe Neural-Symbolic Bridge

```scheme
;; Embeddings are typed by dimension
(: embed (Π (d : Natural) (-> Symbol (Tensor Float d))))

;; Attention preserves type structure
(: attend (Π (d : Natural) (n : Natural)
            (-> (Tensor Float d)           ;; query
                (Vector (Tensor Float d) n) ;; keys
                (Vector (Tensor Float d) n) ;; values
                (Tensor Float d))))         ;; output

;; Soft unification returns probability with gradient
(: soft-unify (Π (emb : EmbeddingTable)
                (-> Term Term (Dual Float))))
```

---

## Part VII: Six-Month Implementation Roadmap

### Overview Timeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     ESHKOL NEURO-SYMBOLIC: 6-MONTH PLAN                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Month 1          Month 2          Month 3          Month 4          Month 5          Month 6
│  ════════         ════════         ════════         ════════         ════════         ════════
│                                                                              │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
│  │ HoTT    │      │ Logic   │      │Knowledge│      │ Neural- │      │ Program │      │ Polish  │
│  │ Types   │ ──▶  │ Core    │ ──▶  │ Base    │ ──▶  │Symbolic │ ──▶  │Synthesis│ ──▶  │  and    │
│  │ Phase 1 │      │ Phase 2 │      │ Phase 3 │      │ Phase 4 │      │ Phase 5 │      │ Release │
│  └─────────┘      └─────────┘      └─────────┘      └─────────┘      └─────────┘      └─────────┘
│                                                                              │
│  • Type infra     • Unification    • Fact DB       • Embeddings    • Holes/??       • Perf opt
│  • Parser ext     • Backtracking   • Rules         • Soft unify    • Enumeration    • Examples
│  • Type checker   • Logic vars     • Triple store  • Attention     • Neural guide   • Docs
│  • Proof erasure  • Testing        • Inference     • Grad estim    • Testing        • Release
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Month 1: HoTT Type System Foundation

**Week 1-2: Type Infrastructure**
- Implement `TypeId`, `TypeNode`, `TypeEnvironment` classes
- Define supertype hierarchy (Value > Number > Integer > Int64)
- Universe levels (𝒰₀, 𝒰₁, 𝒰₂)
- Subtype checking with caching

**Week 3-4: Parser Extensions**
- Add type annotation syntax: `(: name type)`
- Inline annotations: `(define (f x : type) : return-type ...)`
- Type expression parsing (arrows, forall, dependent)
- AST extensions for type information

**Deliverables:**
- `lib/types/type_system.h` and `.cpp`
- Parser modifications for type syntax
- Unit tests for type infrastructure
- ~1,500 lines of new code

### Month 2: Logic Programming Core

**Week 5-6: Unification Engine**
- Add `ESHKOL_VALUE_LOGIC_VAR` type tag
- Implement `eshkol_logic_var_t` structure
- Robinson's unification algorithm with occurs check
- Parser support for `?variable` syntax

**Week 7-8: Substitutions and Backtracking**
- `eshkol_substitution_t` (hash map of bindings)
- `walk`, `extend-subst`, `apply-subst` functions
- Backtracking via explicit choice stack
- `amb`, `require`, `fail` operators
- Integration with arena memory

**Deliverables:**
- Unification engine in codegen (~500 lines)
- Backtracking monad (~400 lines)
- Parser extensions for logic syntax
- Test suite for logic programs (append, member, etc.)

### Month 3: Knowledge Representation

**Week 9-10: Fact Database**
- `eshkol_knowledge_base_t` structure
- `assert!`, `retract!`, `query` operations
- Predicate indexing for efficient lookup
- Arena allocation for KB structures

**Week 11-12: Rules and Inference**
- `define-rule` syntax and AST
- Backward chaining (Prolog-style `prove`)
- Forward chaining (production rules)
- Rule variable renaming to avoid capture

**Week 13: Triple Store (Knowledge Graphs)**
- RDF-style triple representation
- SPO, POS, OSP indexes
- Graph pattern matching
- Path queries (transitive closure)

**Deliverables:**
- Knowledge base implementation (~800 lines)
- Inference engines (~600 lines)
- Triple store (~400 lines)
- Classic logic program tests (genealogy, etc.)

### Month 4: Neural-Symbolic Bridge

**Week 14-15: Symbol Embeddings**
- `eshkol_embedding_table_t` structure
- `embed` operator for symbols
- Term embedding (recursive over structure)
- Integration with tensor operations
- Learnable parameters (gradient tracking)

**Week 16-17: Soft Unification**
- Differentiable similarity function
- `soft-unify` with cosine similarity
- Integration with reverse-mode autodiff
- Gradient flow through matching

**Week 18: Knowledge Attention**
- `attend` operator implementation
- Softmax attention over facts
- Differentiable KB queries
- Key-value attention mechanism

**Week 19: Gradient Estimators**
- Gumbel-Softmax for discrete sampling
- Straight-through estimator
- Custom backward passes in autodiff
- Testing on discrete choice scenarios

**Deliverables:**
- Embedding system (~300 lines)
- Soft unification (~400 lines)
- Attention mechanism (~500 lines)
- Gradient estimators (~300 lines)
- End-to-end gradient tests

### Month 5: Program Synthesis

**Week 20-21: Type-Directed Holes**
- `??` syntax for synthesis targets
- Typed hole representation in AST
- Constraint collection from holes
- Integration with type checker

**Week 22-23: Program Enumeration**
- Grammar-based program generation
- Type-directed pruning
- Depth-bounded search
- Memoization for efficiency

**Week 24: Neural-Guided Search**
- Program encoding for neural network
- Heuristic scoring model
- Beam search with neural guidance
- Learning from synthesis examples

**Deliverables:**
- Hole syntax and semantics (~300 lines)
- Program enumeration (~500 lines)
- Neural guidance integration (~400 lines)
- Synthesis examples and tests

### Month 6: Polish and Release

**Week 25: Performance Optimization**
- Profile critical paths
- Optimize unification (indexing, caching)
- Optimize KB queries
- Memory optimization

**Week 26: Example Applications**
- Neuro-symbolic theorem prover example
- Knowledge-grounded generation example
- Explainable classifier example
- Self-improving code example

**Week 27: Documentation**
- API documentation
- Tutorial: "Neuro-Symbolic Programming in Eshkol"
- Architecture guide
- Migration guide from pure neural/symbolic

**Week 28: Final Testing and Release**
- Integration testing
- Performance benchmarks
- Bug fixes
- Release preparation

**Deliverables:**
- Optimized implementation
- 4+ example applications
- Complete documentation
- Release-ready codebase

---

## Part VIII: Milestone Checkpoints

### End of Month 1
- [ ] Type system compiles and passes unit tests
- [ ] Parser accepts type annotations
- [ ] Simple typed programs type-check correctly

### End of Month 2
- [ ] `(unify '(f ?x) '(f 1))` returns `{?x: 1}`
- [ ] Backtracking finds all solutions
- [ ] Classic logic programs work (append, member)

### End of Month 3
- [ ] Knowledge base stores and retrieves facts
- [ ] Rules derive new facts
- [ ] Backward chaining proves goals
- [ ] Triple store supports graph queries

### End of Month 4
- [ ] Symbols can be embedded as vectors
- [ ] Soft unification returns differentiable similarity
- [ ] Gradients flow through KB attention
- [ ] Discrete choices have gradient estimators

### End of Month 5
- [ ] Holes can be declared with types
- [ ] Simple programs can be synthesized from examples
- [ ] Neural guidance improves synthesis speed

### End of Month 6
- [ ] All features integrated and tested
- [ ] Documentation complete
- [ ] Example applications working
- [ ] Performance meets targets

---

## Part IX: Success Metrics

### 9.1 Correctness Metrics

| Test Category | Target |
|---------------|--------|
| Unification tests | 100% pass |
| Logic program tests (e.g., append, member) | 100% pass |
| Knowledge query tests | 100% pass |
| Gradient correctness (finite difference check) | <1e-5 error |
| Soft unification differentiability | Verified |

### 9.2 Performance Metrics

| Operation | Target |
|-----------|--------|
| Unification (simple terms) | <1μs |
| KB query (1000 facts) | <10ms |
| Forward chaining (100 rules) | <100ms |
| Soft unification (dim=128) | <1ms |
| Attention over 1000 facts | <10ms |

### 9.3 Integration Metrics

| Feature | Requirement |
|---------|-------------|
| Works with all 3 autodiff modes | Yes |
| Compatible with closures | Yes |
| JIT compilation support | Yes |
| REPL integration | Yes |
| Arena memory compatible | Yes |

---

## Part X: Conclusion

Eshkol is uniquely positioned to become a premier neuro-symbolic programming language:

1. **Exceptional neural foundation** - Three autodiff modes, full vector calculus, tensor operations
2. **Homoiconicity** - Code as data enables meta-reasoning and program synthesis
3. **Performance** - LLVM backend provides native speed
4. **Elegant design** - Scheme heritage provides clean semantics

The missing piece is the symbolic reasoning stack. This report provides a complete specification for:

1. **Unification engine** - Foundation for pattern matching and logic
2. **Knowledge base** - Facts, rules, and graph storage
3. **Inference engines** - Forward and backward chaining
4. **Differentiable bridge** - Soft operations with gradient flow
5. **Program synthesis** - Neural-guided code generation

Total estimated implementation: ~6,000-8,000 lines of new code across:
- `llvm_codegen.cpp` additions (~4,000 lines)
- `arena_memory.cpp` additions (~500 lines)
- `eshkol.h` extensions (~200 lines)
- `parser.cpp` extensions (~300 lines)
- `stdlib_logic.esk` (~1,000 lines)
- Tests and examples (~2,000 lines)

The result will be a language where neural networks and symbolic reasoning are not separate systems bolted together, but a unified computational model where:

- **Learning and reasoning** are seamlessly integrated
- **Gradients flow** through symbolic operations
- **Knowledge is differentiable** and learnable
- **Programs can reason** about themselves
- **AI systems can explain** their decisions

This is the future of AI programming, and Eshkol can lead the way.
