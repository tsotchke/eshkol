# Eshkol Consciousness Engine: Technical Reference

## 1. Introduction

The Eshkol Consciousness Engine is a runtime subsystem that integrates three
foundational paradigms from artificial intelligence and cognitive science into a unified computational framework. These paradigms are:

1. **Logic programming** -- symbolic reasoning via unification and knowledge
   bases, grounded in Robinson's resolution principle (Robinson, 1965).
2. **Active inference** -- probabilistic inference over generative models using
   factor graphs and belief propagation, grounded in the Free Energy Principle
   (Friston, 2010).
3. **Global Workspace Theory** -- attention-mediated competition among
   specialist modules with a shared broadcast mechanism, grounded in the
   cognitive architecture of Baars (1988) and the computational formulation
   of Bengio (2017).

Each pillar is implemented as a self-contained C++ runtime library with arena
allocation and tagged-value dispatch. The three components compose naturally:
logic programming provides the symbolic substrate, active inference provides
the probabilistic substrate, and the global workspace provides the attentional
substrate that selects among competing hypotheses.

### Source Files

| Component        | Implementation                  | Header                              | Lines |
|------------------|---------------------------------|-------------------------------------|-------|
| Logic engine     | `lib/core/logic.cpp`            | `inc/eshkol/core/logic.h`           | 805   |
| Active inference | `lib/core/inference.cpp`        | `inc/eshkol/core/inference.h`       | 912   |
| Global workspace | `lib/core/workspace.cpp`        | `inc/eshkol/core/workspace.h`       | 308   |

### Type System Integration

The consciousness engine introduces one new value type and five new heap
subtypes into Eshkol's tagged value system:

| Entity          | Type Tag                     | Heap Subtype              | Value |
|-----------------|------------------------------|---------------------------|-------|
| Logic variable  | `ESHKOL_VALUE_LOGIC_VAR`     | --                        | 10    |
| Substitution    | `ESHKOL_VALUE_HEAP_PTR`      | `HEAP_SUBTYPE_SUBSTITUTION` | 12  |
| Fact            | `ESHKOL_VALUE_HEAP_PTR`      | `HEAP_SUBTYPE_FACT`         | 13  |
| Knowledge base  | `ESHKOL_VALUE_HEAP_PTR`      | `HEAP_SUBTYPE_KNOWLEDGE_BASE` | 15 |
| Factor graph    | `ESHKOL_VALUE_HEAP_PTR`      | `HEAP_SUBTYPE_FACTOR_GRAPH` | 16  |
| Workspace       | `ESHKOL_VALUE_HEAP_PTR`      | `HEAP_SUBTYPE_WORKSPACE`    | 17  |

Logic variables use the `?x` syntax in source code, which the parser transforms
into `ESHKOL_LOGIC_VAR_OP` AST nodes. The `?` prefix is a valid identifier
start character under R7RS, so this extension requires no grammar modifications.

---

## 2. Logic Programming

### 2.1 Logic Variables

Logic variables are globally registered by name. A static registry maps names
to monotonically increasing 64-bit identifiers. The registry is thread-safe:
a mutex serializes the find-or-register operation, and an atomic counter
provides the variable ID.

```
Registry capacity: LOGIC_VAR_MAX = 65536
Name storage:      static pool, 64 bytes per name (intern_var_name)
Thread safety:     std::mutex + std::atomic<uint64_t>
```

Variable names are interned into a static string pool
(`logic.cpp:39-58`) to avoid heap allocation and dangling pointer hazards.
The `eshkol_make_logic_var` function (`logic.cpp:61-87`) performs
deduplication: the same name always returns the same `var_id`.

In the tagged value representation, a logic variable has type tag
`ESHKOL_VALUE_LOGIC_VAR` (10), with the `data.int_val` field storing the
variable ID.

### 2.2 Substitutions

A substitution is an immutable mapping from variable IDs to terms. It is
represented as a pair of parallel arrays -- one of `uint64_t` variable IDs
and one of `eshkol_tagged_value_t` terms -- stored contiguously after the
struct header for cache-friendly access.

**Memory layout** (`logic.h:41-52`):

```
[eshkol_object_header_t]
[eshkol_substitution_t: {num_bindings: u32, capacity: u32}]
[var_ids: uint64_t[capacity]]
[terms:   eshkol_tagged_value_t[capacity]]
```

Substitutions follow a **copy-on-extend** discipline. The `eshkol_extend_subst`
function (`logic.cpp:149-186`) allocates a new substitution containing all
existing bindings plus one new binding. The original substitution is never
modified. This immutability is critical for backtracking during unification
and knowledge base queries: failed branches simply discard the extended
substitution without needing an undo operation.

Capacity grows by doubling (minimum 8 slots). Lookup is linear scan
(`logic.cpp:188-199`), which is sufficient for the binding counts typical in
v1.1 workloads. Predicate-indexed lookup is planned for v1.2.

### 2.3 The Walk Operation

The walk operation resolves a term through a substitution by following
variable chains. If `?x` is bound to `?y` and `?y` is bound to `42`, then
`walk(?x, subst)` returns `42`.

**Shallow walk** (`logic.cpp:203-223`):

```
walk(term, subst):
    current = term
    while current.type == LOGIC_VAR and subst is not null:
        bound = lookup(subst, current.var_id)
        if bound is null: break    // unbound variable
        current = bound
    return current
```

**Deep walk** (`logic.cpp:227-274`) extends this to recursively resolve
compound structures (facts). Given a fact `(parent ?x ?y)` where `?x` is
bound to `alice` and `?y` to `bob`, deep walk produces a new fact
`(parent alice bob)` with all logic variables fully resolved. A depth limit
of 10,000 (`WALK_DEEP_MAX_DEPTH`) prevents stack overflow on cyclic structures
that bypass the occurs check.

### 2.4 Unification

The unification algorithm (`logic.cpp:339-399`) implements Robinson's
unification with occurs check, extended for Eshkol's tagged value system.
The algorithm follows the Martelli-Montanari refinement: walk both terms
first, then dispatch on their types.

```
unify(t1, t2, subst):
    w1 = walk(t1, subst)
    w2 = walk(t2, subst)

    if w1 == w2:                      // identical terms (includes same var)
        return subst                  // success, no new bindings

    if w1 is logic_var:
        if occurs(w1.id, w2, subst):  // occurs check
            return FAIL
        return extend(subst, w1.id, w2)

    if w2 is logic_var:
        if occurs(w2.id, w1, subst):  // occurs check
            return FAIL
        return extend(subst, w2.id, w1)

    if w1 is fact and w2 is fact:
        if w1.predicate != w2.predicate: return FAIL
        if w1.arity != w2.arity:         return FAIL
        for i in 0..arity:
            subst = unify(w1.args[i], w2.args[i], subst)
            if subst is FAIL: return FAIL
        return subst

    return FAIL                       // incompatible ground terms
```

The **occurs check** (`logic.cpp:280-308`) prevents the creation of circular
bindings (e.g., binding `?x` to a term containing `?x`). It recursively
inspects the target term, walking through substitution chains and descending
into fact arguments. A depth limit of 1,000 (`OCCURS_CHECK_MAX_DEPTH`)
provides a safety bound.

Value equality (`logic.cpp:312-335`) dispatches on type tags: integers,
doubles, booleans, and characters compare by value; heap pointers (including
interned symbols) compare by address; logic variables compare by ID.

### 2.5 Facts and Knowledge Base

A **fact** (`logic.h:61-69`) is a predicate symbol plus an array of argument
terms:

```
[eshkol_object_header_t]
[eshkol_fact_t: {predicate: u64, arity: u32, _pad: u32}]
[args: eshkol_tagged_value_t[arity]]
```

The predicate is stored as a pointer to an interned symbol
(`HEAP_SUBTYPE_SYMBOL`), enabling O(1) predicate comparison via pointer
equality.

A **knowledge base** (`logic.h:78-82`) is a growable array of fact pointers
with initial capacity 16 (`KB_INITIAL_CAPACITY`). The `eshkol_kb_assert`
function (`logic.cpp:442-458`) appends facts, doubling capacity when full.

The `eshkol_kb_query` function (`logic.cpp:460-533`) performs pattern matching
by attempting to unify a query pattern against every fact in the knowledge base.
Matching proceeds as follows:

1. Quick-reject on predicate mismatch (pointer inequality).
2. Quick-reject on arity mismatch.
3. Attempt pairwise unification of pattern arguments with fact arguments.
4. On success, collect the resulting substitution into a cons list.

The return value is a list of substitutions -- one per matching fact --
enabling the caller to enumerate all solutions.

---

## 3. Active Inference

### 3.1 Factor Graphs

A factor graph `G = (V, F, E)` consists of:

- **V**: a set of discrete random variables, each with a finite state space.
- **F**: a set of factors, each encoding a conditional probability table (CPT).
- **E**: edges connecting each factor to its participating variables.

In Eshkol, the factor graph structure (`inference.h:55-66`) stores:

```c
typedef struct eshkol_factor_graph {
    uint32_t  num_vars;       // |V|
    uint32_t  num_factors;    // |F|
    uint32_t  max_factors;    // capacity of factors array
    uint32_t  _pad;
    double**  beliefs;        // beliefs[v][s] = log q(X_v = s)
    uint32_t* var_dims;       // var_dims[v] = |state space of X_v|
    eshkol_factor_t** factors;
    double**  msg_fv;         // factor-to-variable messages
    double**  msg_vf;         // variable-to-factor messages
    uint32_t  total_messages; // sum of factor arities
} eshkol_factor_graph_t;
```

Beliefs are initialized to the uniform distribution in log-space:
`log(1/dim)` for each state (`inference.cpp:137-145`). Messages are
allocated lazily on the first call to `fg-infer!`.

### 3.2 Conditional Probability Tables

Each factor (`inference.h:37-43`) stores a CPT as a flat array of
log-probabilities. For a factor connecting variables with dimensions
`d_0, d_1, ..., d_{n-1}`, the CPT has `d_0 * d_1 * ... * d_{n-1}` entries.

The CPT uses **row-major ordering**: for a configuration `(s_0, s_1, ..., s_{n-1})`,
the flat index is computed as:

```
index = s_0 * (d_1 * d_2 * ... * d_{n-1})
      + s_1 * (d_2 * ... * d_{n-1})
      + ...
      + s_{n-1}
```

In the implementation, configurations are decoded from a flat index by
repeated modular arithmetic (`inference.cpp:296-299`):

```c
for (int32_t k = (int32_t)f->num_vars - 1; k >= 0; k--) {
    state[k] = remaining % f->dims[k];
    remaining /= f->dims[k];
}
```

All CPT values are stored in **log-space** (natural logarithm of
probabilities). This prevents numerical underflow during message
multiplication and enables the use of log-sum-exp for marginalization.

### 3.3 Log-Space Arithmetic

The inference engine operates entirely in log-space for numerical stability.
Two core operations are defined (`inference.cpp:54-89`):

**Pairwise log-sum-exp**:

```
logsumexp2(a, b) = max(a, b) + log(1 + exp(-|a - b|))
```

This computes `log(exp(a) + exp(b))` without forming the exponentials
directly. The `LOG_ZERO` sentinel (`-1e30`) represents log(0) = negative
infinity.

**Array log-sum-exp**:

```
logsumexp(arr, n):
    m = max(arr)
    return m + log(sum_i exp(arr[i] - m))
```

**Log-normalization** adjusts a log-probability vector so its exponentials
sum to 1:

```
log_normalize(arr, n):
    z = logsumexp(arr, n)
    for i in 0..n: arr[i] -= z
```

### 3.4 Sum-Product Belief Propagation

The inference engine implements the **sum-product algorithm** (also known as
belief propagation) for loopy factor graphs. The algorithm iteratively passes
messages between factors and variables until beliefs converge.

Two message types are exchanged:

**Factor-to-variable message** (`inference.cpp:274-318`):

```
msg_{f -> v}(x_v) = log-sum-exp over x_{\v} of
    [ f(x) + sum_{u in ne(f)\v} msg_{u -> f}(x_u) ]
```

This marginalizes the factor potential over all variables except the target,
weighting each configuration by the incoming variable-to-factor messages.
The implementation enumerates all configurations of the factor's state space,
decoding each flat index into per-variable states.

**Variable-to-factor message** (`inference.cpp:325-356`):

```
msg_{v -> f}(x_v) = sum_{g in ne(v)\f} msg_{g -> v}(x_v)
```

This sums (in log-space, corresponding to multiplication in probability
space) all incoming factor-to-variable messages except from the target factor.

**Belief update** (`inference.cpp:410-431`):

After each round of message passing, beliefs are updated as the product
(sum in log-space) of all incoming factor-to-variable messages:

```
b(x_v) = sum_{f in ne(v)} msg_{f -> v}(x_v)
```

followed by log-normalization.

**Convergence** (`inference.cpp:358-440`):

The main loop alternates between variable-to-factor and factor-to-variable
message updates, tracking the maximum absolute change in any message. When
this maximum delta falls below the tolerance threshold (default `1e-6`), the
algorithm declares convergence and returns `true`. Otherwise, it runs for
the specified maximum number of iterations.

The message schedule is synchronous (flooding schedule): all messages of
one type are updated before any messages of the other type. This is simpler
than asynchronous schedules but may converge more slowly on some graph
structures.

### 3.5 Variational Free Energy

The variational free energy (`inference.cpp:444-516`) quantifies the
divergence between the approximate posterior `q(s)` (beliefs) and the true
posterior:

```
F = E_q[log q(s)] - E_q[log p(o, s)]
  = -H(q) - E_q[log p(o, s)]
```

where:

- `H(q)` is the entropy of the approximate posterior (computed as the sum
  of marginal entropies under the mean-field assumption).
- `E_q[log p(o, s)]` is the expected log-joint probability under `q`,
  computed by summing over all factor configurations weighted by the
  product of marginal beliefs.

**Entropy computation** (`inference.cpp:462-470`):

```
H(q) = -sum_v sum_s q(x_v = s) * log q(x_v = s)
```

**Expected log-joint** (`inference.cpp:473-496`):

```
E_q[log p(o,s)] = sum_f sum_config q(config) * log f(config)
```

where `q(config) = product_v q(x_v = s_v)` under the mean-field assumption.

**Observation clamping** (`inference.cpp:500-510`):

Observations are provided as `(var_index, observed_state)` pairs. The
observation term adds `log q(x_v = o_v)` for each observed variable, which
corresponds to the surprise (negative log-probability) under the model.

### 3.6 Expected Free Energy

The expected free energy (`inference.cpp:518-591`) evaluates the desirability
of a specific action by estimating the free energy that would result from
taking that action. It decomposes into two components:

```
G(a) = pragmatic_value + epistemic_value
```

**Pragmatic value** (goal-seeking): measures how well the action achieves
desired outcomes.

```
pragmatic = -sum_config q(config | a) * log f(config)
```

**Epistemic value** (uncertainty reduction): measures how much the action
reduces uncertainty about hidden states.

```
epistemic = sum_config q(config | a) * log q(config | a)
```

The implementation (`inference.cpp:542-588`) iterates over all factors
connected to the action variable, marginalizes the CPT over the specified
action state, and accumulates both the pragmatic and epistemic terms.

Lower expected free energy indicates a more preferred action -- one that
both achieves goals and reduces uncertainty.

### 3.7 CPT Update and Learning

The `fg-update-cpt!` function (`inference.cpp:832-900`) enables online
learning by replacing a factor's CPT with new log-probabilities. After
the CPT is overwritten, the message arrays are set to `NULL`
(`inference.cpp:894-895`), forcing reallocation and reinitialization on
the next call to `fg-infer!`. This ensures that stale messages do not
bias the new beliefs.

```
fg-update-cpt!(fg, factor-idx, new-cpt):
    validate factor_idx < num_factors
    validate new_cpt.size == factor.cpt_size
    copy new_cpt data into factor.cpt
    reset msg_fv = NULL, msg_vf = NULL    // force reconvergence
    return fg
```

---

## 4. Global Workspace Theory

### 4.1 Architecture

The global workspace (`workspace.h:58-65`) implements an attention bottleneck:
multiple specialist modules compete for access to a shared content buffer,
and the winner's output is broadcast to all modules on the next cycle.

```c
typedef struct eshkol_workspace {
    uint32_t num_modules;
    uint32_t max_modules;
    uint32_t dim;           // workspace vector dimension
    uint32_t step_count;    // cognitive cycle counter
    double*  content;       // current workspace content
    // Followed by: eshkol_workspace_module_t modules[max_modules]
} eshkol_workspace_t;
```

The workspace content is a flat `double` array of dimension `dim`, initialized
to zeros. Modules are stored inline after the struct using the `WS_MODULES`
macro (`workspace.h:68`).

### 4.2 Module Protocol

Each module (`workspace.h:40-44`) is a named Eshkol closure that conforms to
the following protocol:

```scheme
;; Module signature
(lambda (content-tensor) -> (cons salience proposal-tensor))
```

**Input**: the current workspace content, wrapped as a 1D tensor of
dimension `dim`.

**Output**: a cons pair where:
- The `car` is a scalar salience score (double), indicating the module's
  confidence or urgency.
- The `cdr` is a proposal tensor of dimension `dim`, representing the
  module's suggested update to the workspace content.

Module registration (`workspace.cpp:51-70`) copies the module name to arena
storage and stores the closure tagged value. The maximum number of modules
is fixed at construction time.

### 4.3 Softmax Competition

The `ws-step!` operation executes one cognitive cycle. The implementation is
split between LLVM codegen (closure invocation) and the C runtime
(tensor wrapping and softmax finalization).

**Step 1: Content tensor creation** (`workspace.cpp:171-211`)

The `eshkol_ws_make_content_tensor` helper wraps the workspace's `double*`
content array into a tensor tagged value suitable for passing to closures.
Double values are stored as int64 bit patterns in the tensor element array,
matching Eshkol's tensor representation.

**Step 2: Closure invocation** (LLVM codegen)

The codegen loop calls each module's closure via `codegenClosureCall`,
passing the content tensor as the single argument. This is implemented in
LLVM IR rather than C because Eshkol closures capture free variables and
require the closure calling convention.

**Step 3: Softmax finalization** (`workspace.cpp:213-296`)

The `eshkol_ws_step_finalize` function processes the array of cons-pair
results from all modules:

1. **Extract** salience scores and proposal tensors from each cons pair.
   Invalid results receive a salience of `-1e30` (effectively zero after
   softmax).

2. **Softmax normalization** over salience scores:

```
sigma(z_i) = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
```

   The max-subtraction prevents overflow. Each module's normalized salience
   is stored in its `salience` field for inspection.

3. **Winner selection**: the module with the highest normalized salience
   wins the competition.

4. **Broadcast**: the winner's proposal tensor is copied into the
   workspace content buffer, replacing the previous content.

5. **Step counter**: `ws->step_count` is incremented.

### 4.4 The Cognitive Cycle

The full cognitive cycle implemented by `ws-step!` proceeds as:

```
ws-step!(ws):
    content_tensor = wrap_content(ws.content, ws.dim)
    for each module m in ws.modules:
        results[m] = m.process_fn(content_tensor)    // closure call
    ws_step_finalize(ws, results, ws.num_modules)     // softmax + broadcast
    return ws
```

This cycle can be iterated to model sustained attention, information
integration, and convergence of beliefs across modules. The step count
tracks the number of completed cycles.

---

## 5. Complete API Reference

### 5.1 Logic Primitives

#### `(make-substitution)`

Creates an empty substitution with default capacity (8 slots).

- **Returns**: substitution object (HEAP_PTR, subtype 12)
- **Impl**: `logic.cpp:650-662`, calls `eshkol_make_substitution(arena, 8)`

#### `(unify term1 term2 substitution)`

Unifies two terms under the given substitution. Returns an extended
substitution on success, or `#f` (null) on failure.

- **Arguments**: any two tagged values, a substitution
- **Returns**: substitution on success, null on failure
- **Occurs check**: yes, depth limit 1000
- **Impl**: `logic.cpp:537-556`

#### `(walk term substitution)`

Shallow walk: resolves a term through a substitution by following
variable chains. Non-variable terms are returned as-is.

- **Arguments**: any tagged value, a substitution
- **Returns**: resolved tagged value
- **Impl**: `logic.cpp:558-570`

#### `(make-fact predicate arg1 arg2 ...)`

Creates a fact with the given predicate symbol and arguments.

- **Arguments**: symbol (predicate), followed by argument terms
- **Returns**: fact object (HEAP_PTR, subtype 13)
- **Impl**: `logic.cpp:572-598`

#### `(make-kb)`

Creates an empty knowledge base with initial capacity 16.

- **Returns**: knowledge base object (HEAP_PTR, subtype 15)
- **Impl**: `logic.cpp:600-611`

#### `(kb-assert! kb fact)`

Asserts a fact into the knowledge base. The KB grows by doubling when full.
This is a mutating operation.

- **Arguments**: knowledge base, fact
- **Returns**: void
- **Impl**: `logic.cpp:613-623`

#### `(kb-query kb pattern)`

Queries the knowledge base by unifying the pattern against all facts.
Returns a list of substitutions for successful matches.

- **Arguments**: knowledge base, fact (pattern with logic variables)
- **Returns**: list of substitutions, or `()` if no matches
- **Impl**: `logic.cpp:625-648`

#### `(logic-var? value)`

Predicate: returns `#t` if the value is a logic variable.

- **Returns**: boolean

#### `(substitution? value)`

Predicate: returns `#t` if the value is a substitution.

- **Returns**: boolean

#### `(kb? value)`

Predicate: returns `#t` if the value is a knowledge base.

- **Returns**: boolean

#### `(fact? value)`

Predicate: returns `#t` if the value is a fact.

- **Returns**: boolean

### 5.2 Active Inference Primitives

#### `(make-factor-graph n-vars dims-tensor)`

Creates a factor graph with `n-vars` random variables. The `dims-tensor`
specifies the number of discrete states per variable.

- **Arguments**: integer (number of variables), tensor (state dimensions)
- **Returns**: factor graph object (HEAP_PTR, subtype 16)
- **Beliefs**: initialized to uniform `log(1/dim)` per variable
- **Impl**: `inference.cpp:595-636`

**Example**:
```scheme
;; 3 variables: var0 has 2 states, var1 has 3 states, var2 has 2 states
(define fg (make-factor-graph 3 #(2 3 2)))
```

#### `(fg-add-factor! fg var-indices cpt)`

Adds a factor connecting the specified variables with a CPT of
log-probabilities. The CPT size must equal the product of the connected
variables' dimensions.

- **Arguments**: factor graph, tensor (variable indices), tensor (CPT data)
- **Returns**: void (mutates factor graph)
- **CPT format**: flat tensor of log-probabilities in row-major order
- **Impl**: `inference.cpp:638-689`

**Example**:
```scheme
;; Factor connecting var0 (2 states) and var1 (3 states)
;; CPT has 2*3 = 6 entries (log-probabilities)
(fg-add-factor! fg
  #(0 1)                                    ;; variable indices
  #(-0.693 -1.386 -1.386                    ;; log P(var1 | var0=0)
    -1.386 -0.693 -1.386))                  ;; log P(var1 | var0=1)
```

#### `(fg-infer! fg max-iterations)`

Runs loopy belief propagation on the factor graph. Updates beliefs in-place.
Returns a flat tensor of all beliefs (converted from log-space to probability
space).

- **Arguments**: factor graph, integer (max iterations; default 20)
- **Returns**: tensor of probabilities (beliefs for all variables concatenated)
- **Convergence**: tolerance = 1e-6
- **Message schedule**: synchronous (flooding)
- **Impl**: `inference.cpp:691-757`

**Example**:
```scheme
(define beliefs (fg-infer! fg 50))
;; beliefs is a flat tensor: [P(v0=0), P(v0=1), P(v1=0), P(v1=1), P(v1=2), ...]
```

#### `(fg-update-cpt! fg factor-idx new-cpt)`

Replaces the CPT of the specified factor with new log-probabilities.
Resets all messages, forcing reconvergence on the next `fg-infer!` call.

- **Arguments**: factor graph, integer (factor index), tensor (new CPT)
- **Returns**: factor graph (same object, mutated)
- **Validation**: CPT size must match existing factor size
- **Impl**: `inference.cpp:832-900`

#### `(free-energy fg observations)`

Computes the variational free energy of the current beliefs given
observations.

- **Arguments**: factor graph, tensor of `(var_index observed_state)` pairs
- **Returns**: double (free energy scalar)
- **Formula**: `F = -H(q) - E_q[log p(o,s)]`
- **Observation format**: flat tensor `#(var0_idx var0_state var1_idx var1_state ...)`
- **Impl**: `inference.cpp:759-794`

**Example**:
```scheme
;; Observe var0 in state 1 and var2 in state 0
(define fe (free-energy fg #(0 1  2 0)))
```

#### `(expected-free-energy fg action-var action-state)`

Computes the expected free energy for a specific action, decomposed into
pragmatic (goal-seeking) and epistemic (uncertainty-reducing) components.
Lower values indicate more preferred actions.

- **Arguments**: factor graph, integer (action variable index),
  integer (action state)
- **Returns**: double (expected free energy scalar)
- **Impl**: `inference.cpp:796-828`

**Example**:
```scheme
;; Evaluate action variable 2, state 0
(define g0 (expected-free-energy fg 2 0))
;; Evaluate action variable 2, state 1
(define g1 (expected-free-energy fg 2 1))
;; Choose action with lower EFE
(if (< g0 g1) 0 1)
```

#### `(factor-graph? value)`

Predicate: returns `#t` if the value is a factor graph.

- **Returns**: boolean

### 5.3 Global Workspace Primitives

#### `(make-workspace content-dim max-modules)`

Creates a global workspace with the specified content vector dimension and
maximum number of registrable modules.

- **Arguments**: integer (content dimension), integer (max modules)
- **Returns**: workspace object (HEAP_PTR, subtype 17)
- **Content**: initialized to zero vector
- **Impl**: `workspace.cpp:96-132`

#### `(ws-register! ws name closure)`

Registers a cognitive module with the workspace. The closure must conform
to the module protocol: `(tensor -> (cons salience proposal-tensor))`.

- **Arguments**: workspace, string (module name), closure
- **Returns**: void (mutates workspace)
- **Impl**: `workspace.cpp:134-156`

**Example**:
```scheme
(ws-register! ws "detector"
  (lambda (content)
    (let ((energy (tensor-sum (tensor-mul content content))))
      (cons energy content))))
```

#### `(ws-step! ws)`

Executes one cognitive cycle: invokes all module closures with the current
workspace content, applies softmax competition over salience scores, and
broadcasts the winner's proposal as the new workspace content.

- **Arguments**: workspace
- **Returns**: workspace (same object, mutated)
- **Softmax**: numerically stable (max-subtraction)
- **Max modules per step**: 16
- **Impl**: LLVM codegen (closure loop) + `workspace.cpp:213-296` (finalize)

#### `(workspace? value)`

Predicate: returns `#t` if the value is a workspace.

- **Returns**: boolean

---

## 6. Code Examples

### 6.1 Logic Unification

```scheme
;; Basic unification of logic variables
(define s (make-substitution))

;; Unify ?x with 42
(define s1 (unify ?x 42 s))
(display (walk ?x s1))        ;; => 42

;; Chain: unify ?y with ?x, then walk ?y
(define s2 (unify ?y ?x s1))
(display (walk ?y s2))        ;; => 42

;; Unification failure
(define s3 (unify 1 2 s))
(display s3)                  ;; => #f (null)
```

### 6.2 Knowledge Base Queries

```scheme
;; Build a knowledge base of family relationships
(define kb (make-kb))
(kb-assert! kb (make-fact 'parent 'alice 'bob))
(kb-assert! kb (make-fact 'parent 'alice 'charlie))
(kb-assert! kb (make-fact 'parent 'bob 'dave))

;; Query: who are Alice's children?
(define results (kb-query kb (make-fact 'parent 'alice ?child)))
;; results is a list of substitutions: ({?child -> bob}, {?child -> charlie})

;; Walk each result to extract the child
(for-each
  (lambda (subst)
    (display (walk ?child subst))
    (newline))
  results)
;; => bob
;; => charlie
```

### 6.3 Factor Graph Inference

```scheme
;; Model a simple weather/sprinkler/grass-wet Bayesian network
;; Variables: 0=weather(2), 1=sprinkler(2), 2=grass-wet(2)
(define fg (make-factor-graph 3 #(2 2 2)))

;; Factor: P(sprinkler | weather)
;; weather=sunny: P(sprinkler=on)=0.4, P(sprinkler=off)=0.6
;; weather=rainy: P(sprinkler=on)=0.01, P(sprinkler=off)=0.99
(fg-add-factor! fg #(0 1)
  #(-0.916 -0.511     ;; log P(spr|sunny) = [log(0.4), log(0.6)]
    -4.605 -0.010))   ;; log P(spr|rainy) = [log(0.01), log(0.99)]

;; Factor: P(grass-wet | sprinkler, weather)
(fg-add-factor! fg #(1 0 2)
  #(-2.303 -0.105     ;; spr=on,  sunny: [log(0.1), log(0.9)]
    -0.051 -2.996     ;; spr=on,  rainy: [log(0.95), log(0.05)]
    -0.693 -0.693     ;; spr=off, sunny: [log(0.5), log(0.5)]
    -0.020 -3.912))   ;; spr=off, rainy: [log(0.98), log(0.02)]

;; Run belief propagation
(define beliefs (fg-infer! fg 100))
(display beliefs)
;; => tensor of probabilities for each variable's states

;; Compute free energy given observation: grass is wet (var2 = state0)
(define fe (free-energy fg #(2 0)))
(display fe)
```

### 6.4 Active Inference for Action Selection

```scheme
;; Agent with 2 hidden states and 3 possible actions
(define fg (make-factor-graph 3 #(2 2 3)))

;; Prior over hidden states
(fg-add-factor! fg #(0) #(-0.693 -0.693))  ;; uniform

;; Transition model: P(next_state | state, action)
(fg-add-factor! fg #(0 1 2)
  ;; 2*2*3 = 12 entries
  #(-0.105 -2.303  ;; s0,s'0: action 0 stays, action 1 stays
    -0.693 -0.693  ;; s0,s'0: action 2 uncertain
    -2.303 -0.105  ;; s0,s'1: action 0 unlikely, action 1 likely
    -0.693 -0.693  ;; s1,s'0: uncertain
    -0.105 -2.303  ;; s1,s'0: action 1 stays
    -2.303 -0.105));; s1,s'1: action 2 switches

;; Run inference
(fg-infer! fg 50)

;; Compare expected free energy for each action
(define g0 (expected-free-energy fg 2 0))
(define g1 (expected-free-energy fg 2 1))
(define g2 (expected-free-energy fg 2 2))

;; Select action with lowest EFE
(display "Best action: ")
(cond
  ((and (<= g0 g1) (<= g0 g2)) (display 0))
  ((and (<= g1 g0) (<= g1 g2)) (display 1))
  (else (display 2)))
```

### 6.5 Global Workspace Competition

```scheme
;; Create a workspace with 4-dimensional content and 3 modules
(define ws (make-workspace 4 3))

;; Module 1: "energy detector" -- high salience when content has high norm
(ws-register! ws "energy"
  (lambda (content)
    (let ((norm (tensor-sum (tensor-mul content content))))
      (cons norm (tensor-map (lambda (x) (* x 1.1)) content)))))

;; Module 2: "novelty detector" -- high salience when content differs from zero
(ws-register! ws "novelty"
  (lambda (content)
    (let ((novelty (tensor-sum (tensor-map abs content))))
      (cons novelty #(1.0 0.0 0.0 0.0)))))

;; Module 3: "inhibitor" -- always low salience, proposes zeros
(ws-register! ws "inhibitor"
  (lambda (content)
    (cons 0.01 #(0.0 0.0 0.0 0.0))))

;; Run 10 cognitive cycles
(let loop ((i 0))
  (when (< i 10)
    (ws-step! ws)
    (loop (+ i 1))))
```

### 6.6 Integrated Example: Logic + Inference + Workspace

```scheme
;; A cognitive agent that uses:
;; - Logic programming for rule-based reasoning
;; - Active inference for probabilistic state estimation
;; - Global workspace for attention-mediated integration

;; Knowledge base of domain rules
(define kb (make-kb))
(kb-assert! kb (make-fact 'causes 'rain 'wet-ground))
(kb-assert! kb (make-fact 'causes 'sprinkler 'wet-ground))

;; Factor graph for probabilistic inference
(define fg (make-factor-graph 2 #(2 2)))
(fg-add-factor! fg #(0 1)
  #(-0.223 -1.609    ;; P(wet|rain) = [0.8, 0.2]
    -1.204 -0.357))  ;; P(wet|dry) = [0.3, 0.7]

;; Workspace integrating both subsystems
(define ws (make-workspace 4 2))

;; Module 1: logic-based reasoning
(ws-register! ws "logic"
  (lambda (content)
    (let ((results (kb-query kb (make-fact 'causes ?cause 'wet-ground))))
      (cons (if (null? results) 0.0 1.0)
            #(1.0 0.0 0.0 0.0)))))

;; Module 2: probabilistic inference
(ws-register! ws "inference"
  (lambda (content)
    (let ((beliefs (fg-infer! fg 20)))
      (cons (free-energy fg #(1 0))  ;; observe wet ground
            beliefs))))

;; Run cognitive cycle
(ws-step! ws)
```

---

## 7. Implementation Notes

### 7.1 Arena Allocation

All consciousness engine objects are arena-allocated using the
`arena_allocate_with_header` function, following the same pattern established
for bignums and rationals. Each object is preceded by an `eshkol_object_header_t`
containing the heap subtype, enabling runtime type identification through the
`ESHKOL_GET_HEADER` macro.

Arena allocation provides two key properties:
- **No manual deallocation**: objects are freed in bulk when the arena is reset.
- **Cache locality**: related objects are allocated contiguously in memory.

### 7.2 Tagged Value Dispatch

Each builtin has a `_tagged` variant (e.g., `eshkol_unify_tagged`,
`eshkol_fg_infer_tagged`) that extracts arguments from tagged values and
packs the result into a tagged value. These functions follow the same
alloca/store/call/load pattern used by bignum operations, which the LLVM
codegen emits directly.

### 7.3 ws-step! Codegen Architecture

The `ws-step!` function is unique among the consciousness engine builtins
because it must invoke Eshkol closures. Closures cannot be called from C
runtime code -- they require the LLVM closure calling convention managed by
`codegenClosureCall`. Therefore, `ws-step!` is implemented as a hybrid:

1. The LLVM codegen emits a loop that iterates over workspace modules.
2. For each module, it calls `eshkol_ws_make_content_tensor` to wrap the
   content, then invokes the module's closure via `codegenClosureCall`.
3. After the loop, it calls `eshkol_ws_step_finalize` to perform softmax
   competition and broadcast.

This split avoids the PHI-node corruption issues documented in the closure-
in-loop bug (see project memory): the loop uses alloca/store/load rather
than PHI nodes for all mutable state.

### 7.4 Numerical Considerations

- **Log-space arithmetic**: All probability computations use log-probabilities
  to prevent underflow. The `LOG_ZERO = -1e30` sentinel approximates negative
  infinity.
- **Softmax stability**: The workspace softmax subtracts `max(z)` before
  exponentiation to prevent overflow.
- **Belief normalization**: Beliefs are log-normalized after each update to
  ensure they represent valid probability distributions.
- **Convergence tolerance**: The default BP tolerance of `1e-6` provides
  a practical balance between accuracy and computational cost.

## Unification Algorithm

The logic engine implements Robinson's unification algorithm with occurs-check, operating over Eshkol's tagged value system. The core entry point is `eshkol_unify` in `/Users/tyr/Desktop/eshkol/lib/core/logic.cpp` (line 339).

### Algorithm Structure

Unification takes two terms `t1` and `t2` plus an existing substitution, and either returns an extended substitution that makes the terms identical or returns `NULL` (failure). The algorithm proceeds in five cases:

```
UNIFY(t1, t2, subst):
  1. w1 = WALK(t1, subst)          // resolve any variable chains
     w2 = WALK(t2, subst)
  2. if w1 == w2:                   // identical after walking
       return subst                 // (includes same unbound variable)
  3. if w1 is LOGIC_VAR:
       if OCCURS(w1.id, w2, subst): return FAIL
       return EXTEND(subst, w1.id, w2)
  4. if w2 is LOGIC_VAR:
       if OCCURS(w2.id, w1, subst): return FAIL
       return EXTEND(subst, w2.id, w1)
  5. if w1 is FACT and w2 is FACT:
       if w1.predicate != w2.predicate: return FAIL  // pointer equality
       if w1.arity != w2.arity: return FAIL
       s = subst
       for i in 0..arity-1:
         s = UNIFY(w1.args[i], w2.args[i], s)
         if s == NULL: return FAIL
       return s
  6. return FAIL
```

### Concrete Walkthrough

Consider unifying `(parent ?x (child ?y))` with `(parent alice (child bob))`, starting from an empty substitution `{}`.

**Step 1**: Both terms are facts with predicate `parent`, arity 2. Enter the structural case (case 5).

**Step 2**: Unify arg 0: `?x` vs `alice`. Walk `?x` in `{}` -- unbound, stays `?x`. Walk `alice` -- not a variable, stays `alice`. Case 3 applies: `?x` is a logic variable. Occurs-check: does `?x` appear in `alice`? No (it is an atom). Extend: return `{?x -> alice}`.

**Step 3**: Unify arg 1: `(child ?y)` vs `(child bob)` under `{?x -> alice}`. Both are facts with predicate `child`, arity 1. Recurse into structural unification.

**Step 4**: Unify `?y` vs `bob` under `{?x -> alice}`. Walk `?y` -- unbound. Occurs-check passes. Extend: return `{?x -> alice, ?y -> bob}`.

Final result: `{?x -> alice, ?y -> bob}`.

### Walk Operation

The walk operation (line 203) follows substitution chains to their terminal value. It is a simple loop:

```
WALK(term, subst):
  current = term
  while current is LOGIC_VAR and subst is not null:
    bound = LOOKUP(subst, current.var_id)
    if bound is null: break       // unbound variable -- stop
    current = bound
  return current
```

This handles transitive bindings. If `?x -> ?y` and `?y -> 42`, then `WALK(?x) = 42`. The loop terminates either at a non-variable value or at an unbound variable.

Deep walk (`eshkol_walk_deep`, line 227) additionally recurses into compound structures: after performing a shallow walk, if the result is a fact, it creates a new fact with each argument deeply walked. This is used to fully resolve a term after unification is complete. It has a depth limit of 10,000 to guard against pathological chains.

### Occurs-Check

The occurs-check (line 280) prevents creation of circular substitutions like `{?x -> (f ?x)}`, which would cause walk to loop infinitely. The algorithm:

```
OCCURS(var_id, term, subst):
  walked = WALK(term, subst)
  if walked is LOGIC_VAR:
    return walked.id == var_id
  if walked is FACT:
    for each arg in walked.args:
      if OCCURS(var_id, arg, subst): return true
  return false
```

It walks the term first (to resolve variables) then checks recursively through fact arguments. A depth limit of 1,000 is imposed as a safety bound; at that depth, the check returns `false` rather than stack-overflowing.

### Copy-on-Extend Discipline

Substitutions are immutable. `eshkol_extend_subst` (line 149) always allocates a **new** substitution in the arena, copies all existing bindings from the old one, and appends the new binding. The old substitution remains untouched. This discipline enables backtracking without undo operations -- failed unification branches simply discard the extended substitution (arena reclaims it). The layout uses parallel arrays for var IDs and terms (accessed via `SUBST_VAR_IDS` and `SUBST_TERMS` macros) for cache-friendly linear scanning.

### Complexity

For a term of size `n` (total number of nodes in both terms), unification runs in O(n) time with the occurs-check contribution bounded by the same term traversal. Substitution lookup is O(k) linear scan where k is the number of bindings. The total cost for a single unification of two terms with `m` total variables and `n` total nodes is O(n * m) in the worst case, though in practice the substitution is small and lookup is fast.

---

## Knowledge Base Query

The knowledge base (`eshkol_knowledge_base_t`) is a growable array of fact pointers, starting at capacity 16 and doubling when full. All storage is arena-allocated.

### Query Algorithm

`eshkol_kb_query` (line 460) performs a linear scan over all facts in the KB, attempting to unify each fact's arguments with the query pattern's arguments:

```
KB_QUERY(kb, pattern, initial_subst):
  result_list = NULL  (empty cons list)
  base_subst = initial_subst or empty substitution

  for each fact in kb.facts:
    // Fast-reject: predicate must match (pointer equality on interned symbols)
    if pattern.predicate != 0 and fact.predicate != 0
       and pattern.predicate != fact.predicate:
      skip

    // Arity must match
    if pattern.arity != fact.arity: skip

    // Try to unify each argument pair
    s = base_subst
    for j in 0..arity-1:
      s = UNIFY(pattern.args[j], fact.args[j], s)
      if s == NULL: break  // unification failed for this fact

    if s != NULL:
      result_list = cons(s, result_list)  // prepend substitution

  return result_list
```

Key details:

**Predicate matching** uses pointer equality (line 485). Eshkol symbols are interned -- every occurrence of the same symbol name resolves to the same heap address. This makes predicate comparison O(1). A predicate value of 0 acts as a wildcard (matches anything).

**Multiple result collection**: Every fact that successfully unifies contributes a substitution to the result list. The result is a cons list (Scheme-style linked list) built in reverse order. Each cons cell is arena-allocated with an `eshkol_object_header_t` and contains the substitution as its car and the rest of the list as its cdr.

**Argument unification** proceeds left-to-right, threading the substitution through each argument pair. If argument 0 unification binds `?x -> alice`, that binding is available when unifying argument 1. A failure on any argument rejects the entire fact.

The current implementation is O(n * k) per query where n is the number of facts and k is the average argument count. The header comment notes that v1.2 will add predicate indexing for O(1) lookup by predicate.

---

## Sum-Product Belief Propagation

The inference engine implements loopy belief propagation using the sum-product algorithm entirely in log-space. The implementation is in `/Users/tyr/Desktop/eshkol/lib/core/inference.cpp`.

### Factor Graph Structure

A factor graph consists of:
- **Variables**: discrete random variables, each with a finite number of states (`var_dims[i]`).
- **Factors**: functions connecting subsets of variables via conditional probability tables (CPTs). A factor over variables `{v0, v1, ..., vn}` with dimensions `{d0, d1, ..., dn}` has a CPT of size `d0 * d1 * ... * dn` storing **log-probabilities**.
- **Messages**: one message per factor-variable edge, in each direction. `msg_fv[e][s]` is the log-message from factor to variable for state `s`; `msg_vf[e][s]` is the reverse.
- **Beliefs**: `beliefs[v][s]` is the log-marginal probability of variable `v` being in state `s`.

CPT indexing uses row-major order: for a factor over variables with dimensions `(d0, d1, ..., dn)`, configuration `(s0, s1, ..., sn)` maps to flat index `s0 * (d1*...*dn) + s1 * (d2*...*dn) + ... + sn`.

### Algorithm in Pseudocode

```
FG_INFER(fg, max_iterations, tolerance):
  // Allocate messages if not already present
  for each factor-variable edge e:
    msg_fv[e] = [log(1/dim), ..., log(1/dim)]   // uniform initialization
    msg_vf[e] = [log(1/dim), ..., log(1/dim)]

  for iter in 1..max_iterations:
    max_delta = 0

    // Phase 1: Update all variable-to-factor messages
    for each factor fi, for each connected variable vi:
      new_msg[s] = 0    // log(1) = 0, neutral for product
      for each OTHER factor g connected to variable vi:
        new_msg[s] += msg_fv[g->vi][s]     // sum in log = product in linear
      NORMALIZE(new_msg)
      max_delta = max(max_delta, max |new_msg - old_msg|)
      msg_vf[fi,vi] = new_msg

    // Phase 2: Update all factor-to-variable messages
    for each factor fi, for each connected variable vi (target):
      out_msg[s] = LOG_ZERO    // -infinity, neutral for log-sum-exp
      for each configuration (s0, s1, ..., sn) of factor fi:
        log_val = CPT[config]
        for each variable vj in fi where vj != vi:
          log_val += msg_vf[fi,vj][sj]    // incoming var-to-factor messages
        out_msg[s_target] = LOG_SUM_EXP(out_msg[s_target], log_val)
      NORMALIZE(out_msg)
      max_delta = max(max_delta, max |out_msg - old_msg|)
      msg_fv[fi,vi] = out_msg

    // Phase 3: Update beliefs
    for each variable v:
      beliefs[v][s] = 0    // log-space: start at log(1)
      for each factor fi connected to v:
        beliefs[v][s] += msg_fv[fi->v][s]
      NORMALIZE(beliefs[v])

    // Phase 4: Check convergence
    if max_delta < tolerance:
      return CONVERGED

  return NOT_CONVERGED
```

### Log-Space Arithmetic

All computation uses log-probabilities to prevent underflow. The key primitive is log-sum-exp:

```
LOG_SUM_EXP(a, b):
  if a == LOG_ZERO: return b
  if b == LOG_ZERO: return a
  m = max(a, b)
  return m + log(1 + exp(-|a - b|))
```

The sentinel value `LOG_ZERO = -1e30` approximates negative infinity. It is not `-inf` to avoid NaN propagation in arithmetic. The `logsumexp` function over arrays (line 67) finds the maximum first, then computes `max + log(sum(exp(x_i - max)))`, which is the standard numerically stable log-sum-exp.

Normalization (`log_normalize`, line 82) subtracts the log-sum-exp of the entire vector from each element, ensuring `sum(exp(arr[i])) = 1` after normalization.

### Factor-to-Variable Message

The factor-to-variable message computation (`compute_factor_to_var_message`, line 274) is the most expensive operation. It marginalizes the factor's CPT over all variables except the target:

For a factor `f` connecting variables `{v0, v1, v2}` and target variable `v1`:

```
msg_{f->v1}(s1) = log-sum-exp over (s0, s2):
    [ CPT[s0, s1, s2] + msg_{v0->f}(s0) + msg_{v2->f}(s2) ]
```

The implementation enumerates all `cpt_size` configurations using integer division to decode the flat index back into per-variable states. For each configuration, it accumulates the log-CPT value plus all incoming variable-to-factor messages (excluding the target), then merges into the target state's accumulator via `logsumexp2`.

### Variable-to-Factor Message

The variable-to-factor message (`compute_var_to_factor_message`, line 325) is simpler: it sums (in log-space) all incoming factor-to-variable messages for the variable **except** from the target factor. This is the product (in probability space) of all other factors' opinions about this variable.

### Convergence

The convergence criterion is the maximum absolute change in any message value across a full iteration. The default tolerance is `1e-6` (hardcoded in the tagged dispatch at line 715). The default maximum iteration count is 20.

**Loopy BP note**: For tree-structured factor graphs, this algorithm converges to exact marginals. For graphs with cycles (loopy BP), convergence is not guaranteed, but the algorithm often converges in practice to approximate marginals. The fixed iteration limit ensures termination regardless.

### Message Indexing

Messages are organized as a flat array indexed by `get_msg_idx(fg, fi, vi)` (line 258), which sums the `num_vars` of all factors before `fi` plus the local variable position `vi`. This avoids a 2D allocation but makes indexing O(num_factors) -- acceptable for the small factor graphs expected in v1.1.

---

## Free Energy Computation

Variational free energy (`eshkol_free_energy`, line 444) quantifies how well the model's beliefs match observations. It is the central quantity minimized in active inference.

### Formula

```
F = E_q[log q(s)] - E_q[log p(o, s)]
  = -H(q) - E_q[log p(o, s)]
```

Where:
- `q(s)` is the approximate posterior (current beliefs, product of marginals under mean-field)
- `p(o, s)` is the generative model (factor potentials)
- `H(q)` is the entropy of the beliefs
- The first term `-H(q)` is the **complexity cost**: how concentrated the beliefs are
- The second term `-E_q[log p(o,s)]` is the **negative accuracy**: how poorly the model predicts

Lower free energy means the model explains the observations better with less complexity.

### Computation

The implementation computes the two terms separately:

**Entropy term** (mean-field approximation):
```
H(q) = sum over variables v:
          sum over states s:
            -q(v,s) * log q(v,s)
```
where `q(v,s) = exp(beliefs[v][s])`. A guard `q > 1e-30` prevents `0 * log(0)`.

**Expected log-joint** (factor potentials):
```
E_q[log p(o,s)] = sum over factors f:
                    sum over configurations (s0,...,sn):
                      q(s0,...,sn) * CPT[s0,...,sn]
```
where `q(s0,...,sn) = prod(q(vi, si))` under mean-field. The implementation decodes each configuration, computes `log_q_config` as the sum of log-beliefs, exponentiates to get `q_config`, and accumulates `q_config * CPT[config]`.

**Observation term**: If observations are provided as `#(var_index observed_state)` pairs, each observation adds `beliefs[var_idx][obs_state]` to the expected log-joint. This effectively clamps the observed variable to its observed state -- the belief at that state acts as log p(o|s).

The final free energy is `F = -entropy - expected_log_joint`.

### Interpretation

- **F = 0**: Beliefs perfectly explain observations with maximum entropy (ideal).
- **F > 0**: Model is surprised by the data (beliefs do not predict well).
- **F decreasing** after `fg-infer!`: Belief propagation is working -- beliefs are becoming more consistent with the factor structure.

---

## Expected Free Energy

Expected free energy (EFE, `eshkol_expected_free_energy`, line 518) evaluates how good an action would be **before taking it**. It is the core quantity for action selection in active inference.

### Decomposition

```
G(a) = E_q(o,s|a)[log q(s|a) - log p(o,s)]
     = -E_q(o|a)[log p(o|a)]     (pragmatic value: goal-seeking)
     + E_q[D_KL(q(s|o,a) || q(s|a))]  (epistemic value: information gain)
```

The **pragmatic value** penalizes actions that lead to unlikely (surprising) outcomes under the model. The **epistemic value** rewards actions that reduce uncertainty -- actions whose outcomes would be informative about hidden states.

### Implementation

The implementation conditions on a specific action (variable `action_var` in state `action_state`) and evaluates the resulting expected surprise:

```
EFE(action_var, action_state):
  efe = 0
  for each factor f involving action_var:
    for each configuration where action_var = action_state:
      // Compute q(other vars | action) as product of marginals
      log_q = sum over non-action vars: beliefs[var][state]
      q = exp(log_q)
      if q > threshold:
        efe -= q * CPT[config]     // pragmatic: -E[log p]
        efe += q * log_q           // epistemic: E[log q]
  return efe
```

Lower EFE means a more preferred action: actions that both achieve goals (low pragmatic surprise) and resolve uncertainty (high epistemic value).

### Difference from Free Energy

| Property | Free Energy (F) | Expected Free Energy (G) |
|---|---|---|
| Temporal | Retrospective (current beliefs) | Prospective (hypothetical futures) |
| Purpose | Inference (update beliefs) | Planning (select actions) |
| Minimized by | Belief propagation | Action selection |
| Input | Current observations | Candidate action |

---

## CPT Learning via fg-update-cpt!

`eshkol_fg_update_cpt_tagged` (line 832) enables online learning by mutating a factor's conditional probability table in-place.

### Procedure

1. **Validate**: Check that the factor graph, factor index, and new CPT tensor are valid. The new CPT must have exactly `cpt_size` elements matching the factor's existing CPT dimensions.

2. **Mutate CPT**: Copy the new CPT values element-by-element from the tensor into the factor's `cpt` array (line 889):
   ```c
   for (uint32_t i = 0; i < f->cpt_size; i++) {
       f->cpt[i] = tensor_get(cpt_tensor, i);
   }
   ```

3. **Reset messages**: Set both `msg_fv` and `msg_vf` to `NULL` (lines 894-895). This forces the next call to `fg-infer!` to reallocate and reinitialize all messages to uniform distributions. Resetting is essential because stale messages were computed under the old CPT -- using them with new CPT values would produce incorrect beliefs and potentially prevent convergence.

4. **Return**: The mutated factor graph is returned.

### Learning Cycle

The full learning loop in user code follows this pattern:

```scheme
;; Observe data
(define obs #(0 1))              ; variable 0 observed in state 1

;; Infer beliefs under current model
(fg-infer! fg 20)
(define fe1 (free-energy fg obs))

;; Update CPT based on observation (e.g., increase probability of observed state)
(define new-cpt #(-0.1 -2.3 -2.3 -0.1))
(fg-update-cpt! fg 0 new-cpt)

;; Re-infer: messages were reset, beliefs reconverge under new CPT
(fg-infer! fg 20)
(define fe2 (free-energy fg obs))

;; fe2 < fe1 if the update was good (model explains data better)
```

The message reset is the critical implementation detail. Without it, belief propagation would start from messages computed under the old CPT, potentially converging to a wrong fixed point or oscillating.

---

## Global Workspace Competition (ws-step!)

The global workspace implements Baars' Global Workspace Theory (1988) as a computational attention mechanism. The `ws-step!` operation is split between LLVM codegen (closure invocation) and C runtime helpers (tensor wrapping and softmax finalization).

### Data Structure

A workspace (`eshkol_workspace_t`) contains:
- `dim`: the dimension of the shared content vector (fixed at creation)
- `content`: a `double*` array of `dim` elements (the "blackboard")
- `modules[]`: array of registered modules, each containing:
  - `name`: human-readable label
  - `process_fn`: an Eshkol closure (tagged value of type `ESHKOL_VALUE_CALLABLE`)
  - `salience`: the most recently computed softmax probability

### ws-step! Execution

The cognitive cycle proceeds in five phases:

**Phase 1 -- Content Tensor Creation** (`eshkol_ws_make_content_tensor`, workspace.cpp line 171):

The C runtime wraps the workspace's raw `double*` content into a tensor tagged value that Eshkol closures can receive. It allocates a `tensor_layout_t` with object header (subtype `HEAP_SUBTYPE_TENSOR`), 1D shape `[dim]`, and copies the content doubles into the tensor's `elements` array as int64 bit patterns (the standard Eshkol tensor storage format).

**Phase 2 -- Closure Invocation** (LLVM codegen, not in the C runtime):

For each registered module `i`, the codegen calls `modules[i].process_fn` with the content tensor as the sole argument using `codegenClosureCall`. Each closure returns a cons pair `(salience . proposal)` where `salience` is a double and `proposal` is a tensor of dimension `dim`.

This phase must be in LLVM codegen rather than C because Eshkol closures are LLVM-generated functions with captured environments -- they cannot be called from plain C. The codegen uses the `closure_call_callback_` mechanism established for `vector-for-each`, `reduce`, etc.

**Phase 3 -- Result Extraction** (`eshkol_ws_step_finalize`, workspace.cpp line 213):

For each module result (a cons cell), extract:
- `car` (first 16 bytes): the salience score as a double
- `cdr` (next 16 bytes): pointer to the proposal tensor

Modules that return invalid results get salience `-1e30` (effectively zero after softmax).

**Phase 4 -- Softmax Competition** (workspace.cpp lines 259-280):

```
// Numerically stable softmax
max_sal = max(salience[0..n-1])
for i in 0..n-1:
  exp_sal[i] = exp(salience[i] - max_sal)
sum_exp = sum(exp_sal[0..n-1])

// Normalize and find winner
winner = argmax(exp_sal / sum_exp)
for i in 0..n-1:
  modules[i].salience = exp_sal[i] / sum_exp
```

The softmax converts raw salience scores into a probability distribution. The implementation subtracts `max_sal` before exponentiating for numerical stability (prevents overflow). The winner is the module with the highest normalized probability.

Note that the current implementation uses an implicit temperature of `T = 1.0` (the raw salience scores are passed directly to softmax). A higher temperature `T` would divide each salience by `T` before softmax, making competition more uniform. A lower temperature pushes toward winner-take-all. User code controls the effective temperature by scaling the salience values returned from closures.

**Phase 5 -- Broadcast** (workspace.cpp lines 284-296):

The winner's proposal tensor is copied into the workspace's content buffer:

```c
for (uint32_t i = 0; i < copy_dim; i++) {
    union { int64_t ii; double d; } u;
    u.ii = proposals[winner]->elements[i];
    ws->content[i] = u.d;
}
ws->step_count++;
```

The tensor elements are stored as int64 bit patterns and must be reinterpreted as doubles via union cast. If the proposal tensor is smaller than the workspace dimension, only the available elements are copied. The step counter is incremented.

### C Runtime Helpers

| Function | Purpose |
|---|---|
| `eshkol_ws_make_content_tensor` | Wraps `ws->content` (raw doubles) into a 1D tensor tagged value with arena-allocated header, dimensions array, and elements array. Converts doubles to int64 bit patterns. |
| `eshkol_ws_step_finalize` | Extracts salience/proposal pairs from cons cells, runs softmax competition, copies winner's proposal tensor back into `ws->content`. Maximum 16 modules (stack-allocated arrays). |

### Cognitive Cycle Summary

After `ws-step!` completes, all modules see the winner's content on the next cycle. This implements the "broadcast" aspect of Global Workspace Theory: the winning module's representation becomes globally available, enabling information integration across specialized processors.

---

## References

- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge
  University Press.
- Bengio, Y. (2017). The consciousness prior. *arXiv:1709.08568*.
- Friston, K. (2010). The free-energy principle: a unified brain theory?
  *Nature Reviews Neuroscience*, 11(2), 127-138.
- Kschischang, F. R., Frey, B. J., & Loeliger, H.-A. (2001). Factor graphs
  and the sum-product algorithm. *IEEE Transactions on Information Theory*,
  47(2), 498-519.
- Robinson, J. A. (1965). A machine-oriented logic based on the resolution
  principle. *Journal of the ACM*, 12(1), 23-41.
