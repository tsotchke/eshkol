# Eshkol Consciousness Engine — Technical Reference

This document is the canonical reference for the v1.1-accelerate
consciousness engine: the unification + factor-graph + global-workspace
runtime stack that ships as a set of compiler builtins under
`lib/core/{logic,inference,workspace}.{h,cpp}` and is wired into LLVM
codegen by `lib/backend/logic_workspace_codegen.cpp`.

The text below is grounded in source. Every formula, struct field, and
heap-subtype number is traceable to a specific line of the implementation;
the source citations are inline as `lib/core/inference.cpp §<symbol>`.

---

## 1. Overview

Three independently-developed paradigms from artificial intelligence and
cognitive science compose the engine:

1. **Symbolic logic programming.** A first-order substrate built on
   Robinson's resolution principle (J. A. Robinson, *A machine-oriented
   logic based on the resolution principle*, Journal of the ACM 12(1),
   1965). It provides logic variables, immutable substitutions,
   unification with occurs-check, and a knowledge base of ground facts.
2. **Active inference.** A probabilistic substrate built on sum-product
   belief propagation over factor graphs (Kschischang, Frey, Loeliger,
   *Factor graphs and the sum-product algorithm*, IEEE Trans. Inf. Theory
   47(2), 2001) and on the variational free-energy principle (K. Friston,
   *The free-energy principle: a unified brain theory?*, Nature Reviews
   Neuroscience 11(2), 2010). It provides discrete factor graphs in
   log-space, loopy BP, variational free energy, expected free energy,
   evidence clamping, and online CPT learning.
3. **Global Workspace Theory.** An attentional substrate inspired by
   B. J. Baars, *A Cognitive Theory of Consciousness* (Cambridge,
   1988), and the computational reformulation of Y. Bengio,
   *The consciousness prior* (arXiv:1709.08568, 2017). It provides
   a shared content tensor, a roster of cognitive specialist modules
   implemented as Eshkol closures, salience-based softmax competition,
   and broadcast of the winner.

The three components compose: logic provides the symbolic substrate
(predicates, terms, queries), inference provides the probabilistic
substrate (beliefs, evidence, action selection), and the workspace
provides the attentional substrate (which hypothesis wins access to
working memory this cycle).

### 1.1 Source files

| Component         | Header                                | Implementation                          | Lines |
|-------------------|---------------------------------------|------------------------------------------|-------|
| Logic engine      | `inc/eshkol/core/logic.h`             | `lib/core/logic.cpp`                     | 246 / 961 |
| Active inference  | `inc/eshkol/core/inference.h`         | `lib/core/inference.cpp`                 | 183 / 1029 |
| Global workspace  | `inc/eshkol/core/workspace.h`         | `lib/core/workspace.cpp`                 | 146 / 308 |
| Auxiliary builtins| —                                     | `lib/core/logic_builtins.cpp`            | 164  |
| LLVM codegen      | `inc/eshkol/backend/logic_workspace_codegen.h` | `lib/backend/logic_workspace_codegen.cpp` | 665  |
| Op dispatch       | `lib/backend/llvm_codegen.cpp` §`codegenOperation` | (lines 8648–8697)                  | —    |
| Type predicates   | `lib/backend/llvm_codegen.cpp` (lines 32223–32430) | —                                  | —    |
| Parser keywords   | `lib/frontend/parser.cpp` (lines 985–1011) and §`TOKEN_SYMBOL` (line 888) | —          | —    |
| Tagged-value type | `inc/eshkol/eshkol.h` (lines 87, 351–356) and `lib/backend/type_system.cpp` §`createStructTypes` | — | — |

### 1.2 Vocabulary of cited identifiers

Throughout this document, identifiers in the form `eshkol_…` denote C
runtime functions or structures. Identifiers in CamelCase
(`LogicWorkspaceCodegen::codegenWalk`) denote C++ codegen methods.
Hyphenated identifiers in Scheme code (`fg-add-factor!`) are user-facing
builtins.

---

## 2. Type system integration

The engine introduces one new value type tag and six new heap subtypes
into Eshkol's 16-byte tagged value system. The numeric values below are
verified against `inc/eshkol/eshkol.h`.

### 2.1 Tagged-value layout

The tagged value is a 16-byte C struct (`inc/eshkol/eshkol.h` line 140;
LLVM mirror at `lib/backend/type_system.cpp` lines 54–60):

```c
struct eshkol_tagged_value {
    uint8_t  type;        // offset 0 — eshkol_value_type_t
    uint8_t  flags;       // offset 1 — exactness, ports, etc.
    uint16_t reserved;    // offset 2 — future use
    /* implicit 4 bytes padding for 8-byte alignment of data union */
    union { int64_t int_val; double double_val; uint64_t ptr_val;
            uint64_t raw_val; } data;   // offset 8
};
```

The five-field LLVM `StructType` `{i8, i8, i16, i32, i64}` exposes the
data slot at field index **4** (not 3); this is the discipline that the
codegen predicate functions use (`codegenLogicVarPred`,
`codegenSubstPred`, `codegenKBPred`, `codegenFactPred`,
`codegenFactorGraphPred`, `codegenWorkspacePred` —
`lib/backend/llvm_codegen.cpp` lines 32223–32430).

### 2.2 New value-type tag

| Tag                        | Value | Source                          |
|----------------------------|-------|---------------------------------|
| `ESHKOL_VALUE_LOGIC_VAR`   | 10    | `inc/eshkol/eshkol.h` line 87   |

Logic variables fit entirely in a single tagged value: the type byte is
10 and `data.int_val` is the variable ID. There is no heap object for
the variable itself; the name lives in a static intern pool managed by
`eshkol_make_logic_var` (`lib/core/logic.cpp` lines 125–151).

### 2.3 New heap subtypes

All six subtypes are members of the `heap_subtype_t` enum in
`inc/eshkol/eshkol.h` (lines 337–360):

| Subtype                          | Value | Source line | Backing struct                   |
|----------------------------------|-------|-------------|----------------------------------|
| `HEAP_SUBTYPE_SUBSTITUTION`      | 12    | line 351    | `eshkol_substitution_t`          |
| `HEAP_SUBTYPE_FACT`              | 13    | line 352    | `eshkol_fact_t`                  |
| *(reserved for RULE)*            | 14    | line 353    | —                                |
| `HEAP_SUBTYPE_KNOWLEDGE_BASE`    | 15    | line 354    | `eshkol_knowledge_base_t`        |
| `HEAP_SUBTYPE_FACTOR_GRAPH`      | 16    | line 355    | `eshkol_factor_graph_t`          |
| `HEAP_SUBTYPE_WORKSPACE`         | 17    | line 356    | `eshkol_workspace_t`             |

Value `14` is reserved for the rule heap subtype slated for v1.2
backward chaining. The non-monotonic gap is intentional and load-bearing
for forward source compatibility.

### 2.4 Object header

Every heap object carries an 8-byte `eshkol_object_header_t` prefix
(`inc/eshkol/eshkol.h` lines 322–327):

```c
struct eshkol_object_header {
    uint8_t  subtype;     // distinguishes heap kinds
    uint8_t  flags;       // GC / linear / shared / pinned / external
    uint16_t ref_count;   // 0 = not ref-counted
    uint32_t size;        // object size in bytes (excluding header)
};
```

The macro `ESHKOL_GET_HEADER(data_ptr)` recovers the header by
subtracting 8 bytes from the data pointer
(`inc/eshkol/eshkol.h` line 446). Every consciousness-engine runtime
predicate (e.g., `eshkol_is_logic_var`, the `codegen…Pred` family) uses
this offset-by-8 pattern to read the subtype byte.

### 2.5 LLVM dispatch sketch

When a Scheme program calls `(factor-graph? x)`, the LLVM IR that
codegen emits is (paraphrased from
`lib/backend/llvm_codegen.cpp` §`codegenFactorGraphPred`, lines
32358–32393):

```
%type   = extractvalue {i8,i8,i16,i32,i64} %x_tagged, 0
%isheap = icmp eq i8 %type, 8                          ; ESHKOL_VALUE_HEAP_PTR
br i1 %isheap, label %check_fg_sub, label %not_fg
check_fg_sub:
  %data    = extractvalue {i8,…} %x_tagged, 4
  %dataptr = inttoptr i64 %data to ptr
  %hdrptr  = getelementptr i8, ptr %dataptr, i64 -8
  %sub     = load i8, ptr %hdrptr
  %is_fg   = icmp eq i8 %sub, 16                       ; HEAP_SUBTYPE_FACTOR_GRAPH
  br label %fg_merge
not_fg:
  br label %fg_merge
fg_merge:
  %pred = phi i1 [ %is_fg, %check_fg_sub ], [ false, %not_fg ]
```

The same pattern (offset 0 for the type byte, offset 4 for the data
field, offset −8 from data for the header) is repeated for the other
predicates.

---

## 3. The compiler builtins

The Eshkol parser maps a fixed set of identifier strings to dedicated AST
operation codes (`lib/frontend/parser.cpp` lines 985–1011). Each
operation code is then dispatched by
`LLVMCodegen::codegenOperation` (`lib/backend/llvm_codegen.cpp` lines
8648–8697) to a codegen method, the bulk of which live in
`LogicWorkspaceCodegen` and the six predicates of which remain in
`llvm_codegen.cpp`.

This table is the authoritative inventory of consciousness-engine
builtins. The leading tutorial doc and project memory cite "22
builtins"; in fact the parser emits **24** dedicated op codes plus a
further **3** that route through the name-dispatched `SystemCodegen`
fallback (`fg-marginal`, `fg-entropy`, `kb-retract!`).

### 3.1 Logic and KB builtins

| Builtin                | Arity | Parser op                          | Codegen method                                  | Runtime helper                                  | Returns                       |
|------------------------|-------|------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------|
| `make-substitution`    | 0     | `ESHKOL_MAKE_SUBST_OP`             | `LogicWorkspaceCodegen::codegenMakeSubst`       | `eshkol_make_substitution_tagged`               | substitution                  |
| `unify`                | 3     | `ESHKOL_UNIFY_OP`                  | `LogicWorkspaceCodegen::codegenUnify`           | `eshkol_unify_tagged`                           | substitution or null          |
| `walk`                 | 2     | `ESHKOL_WALK_OP`                   | `LogicWorkspaceCodegen::codegenWalk`            | `eshkol_walk_tagged`                            | term                          |
| `make-fact`            | 1+    | `ESHKOL_MAKE_FACT_OP`              | `LogicWorkspaceCodegen::codegenMakeFact`        | `eshkol_make_fact_tagged`                       | fact                          |
| `make-kb`              | 0     | `ESHKOL_MAKE_KB_OP`                | `LogicWorkspaceCodegen::codegenMakeKB`          | `eshkol_make_kb_tagged`                         | knowledge base                |
| `kb-assert!`           | 2     | `ESHKOL_KB_ASSERT_OP`              | `LogicWorkspaceCodegen::codegenKBAssert`        | `eshkol_kb_assert_tagged`                       | null                          |
| `kb-query`             | 2     | `ESHKOL_KB_QUERY_OP`               | `LogicWorkspaceCodegen::codegenKBQuery`         | `eshkol_kb_query_tagged`                        | list of substitutions         |
| `kb-query-prefix`      | 2     | `ESHKOL_KB_QUERY_PREFIX_OP`        | `LogicWorkspaceCodegen::codegenKBQueryPrefix`   | `eshkol_kb_query_prefix_tagged`                 | list of substitutions         |
| `logic-var?`           | 1     | `ESHKOL_LOGIC_VAR_PRED_OP`         | `LLVMCodegen::codegenLogicVarPred`              | inline IR (no runtime call)                     | bool                          |
| `substitution?`        | 1     | `ESHKOL_SUBSTITUTION_PRED_OP`      | `LLVMCodegen::codegenSubstPred`                 | inline IR                                       | bool                          |
| `kb?`                  | 1     | `ESHKOL_KB_PRED_OP`                | `LLVMCodegen::codegenKBPred`                    | inline IR                                       | bool                          |
| `fact?`                | 1     | `ESHKOL_FACT_PRED_OP`              | `LLVMCodegen::codegenFactPred`                  | inline IR                                       | bool                          |

### 3.2 Active inference builtins

| Builtin                | Arity | Parser op                          | Codegen method                                  | Runtime helper                                  | Returns                       |
|------------------------|-------|------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------|
| `make-factor-graph`    | 2     | `ESHKOL_MAKE_FACTOR_GRAPH_OP`      | `LogicWorkspaceCodegen::codegenMakeFactorGraph` | `eshkol_make_factor_graph_tagged`               | factor graph                  |
| `fg-add-factor!`       | 3     | `ESHKOL_FG_ADD_FACTOR_OP`          | `LogicWorkspaceCodegen::codegenFGAddFactor`     | `eshkol_fg_add_factor_tagged`                   | null (mutates fg)             |
| `fg-infer!`            | 2     | `ESHKOL_FG_INFER_OP`               | `LogicWorkspaceCodegen::codegenFGInfer`         | `eshkol_fg_infer_tagged`                        | tensor of probabilities       |
| `fg-observe!`          | 3     | `ESHKOL_FG_OBSERVE_OP`             | `LogicWorkspaceCodegen::codegenFGObserve`       | `eshkol_fg_observe_tagged`                      | bool                          |
| `fg-update-cpt!`       | 3     | `ESHKOL_FG_UPDATE_CPT_OP`          | `LogicWorkspaceCodegen::codegenFGUpdateCPT`     | `eshkol_fg_update_cpt_tagged`                   | factor graph (mutated)        |
| `free-energy`          | 2     | `ESHKOL_FREE_ENERGY_OP`            | `LogicWorkspaceCodegen::codegenFreeEnergy`      | `eshkol_free_energy_tagged`                     | double                        |
| `expected-free-energy` | 3     | `ESHKOL_EXPECTED_FREE_ENERGY_OP`   | `LogicWorkspaceCodegen::codegenEFE`             | `eshkol_efe_tagged`                             | double                        |
| `factor-graph?`        | 1     | `ESHKOL_FACTOR_GRAPH_PRED_OP`      | `LLVMCodegen::codegenFactorGraphPred`           | inline IR                                       | bool                          |

### 3.3 Global workspace builtins

| Builtin                | Arity | Parser op                          | Codegen method                                  | Runtime helper                                  | Returns                       |
|------------------------|-------|------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------|
| `make-workspace`       | 2     | `ESHKOL_MAKE_WORKSPACE_OP`         | `LogicWorkspaceCodegen::codegenMakeWorkspace`   | `eshkol_make_workspace_tagged`                  | workspace                     |
| `ws-register!`         | 3     | `ESHKOL_WS_REGISTER_OP`            | `LogicWorkspaceCodegen::codegenWSRegister`      | `eshkol_ws_register_tagged`                     | null (mutates ws)             |
| `ws-step!`             | 1     | `ESHKOL_WS_STEP_OP`                | `LogicWorkspaceCodegen::codegenWSStep`          | `eshkol_ws_make_content_tensor` + `eshkol_ws_step_finalize` (inline IR closure loop) | workspace (mutated)         |
| `workspace?`           | 1     | `ESHKOL_WORKSPACE_PRED_OP`         | `LLVMCodegen::codegenWorkspacePred`             | inline IR                                       | bool                          |

### 3.4 Auxiliary builtins (name-dispatched)

`fg-marginal`, `fg-entropy`, and `kb-retract!` do not get dedicated AST
operation codes; they reach codegen via the function-name fallback in
`LLVMCodegen::codegenOperation`'s call-handling branch
(`lib/backend/llvm_codegen.cpp` lines 12271–12273) which routes through
`SystemCodegen` (`lib/backend/system_codegen.cpp` lines 1521–1523). The
runtime implementations live in `lib/core/logic_builtins.cpp` lines
21–162.

| Builtin       | Codegen method                | Runtime helper                | Returns                       |
|---------------|-------------------------------|-------------------------------|-------------------------------|
| `fg-marginal` | `SystemCodegen::fgMarginal`   | `eshkol_fg_marginal_tagged`   | tensor of probabilities       |
| `fg-entropy`  | `SystemCodegen::fgEntropy`    | `eshkol_fg_entropy_tagged`    | double                        |
| `kb-retract!` | `SystemCodegen::kbRetract`    | `eshkol_kb_retract_tagged`    | bool                          |

---

## 4. Logic engine

### 4.1 Logic variables

A logic variable is a parser primitive: when `tokenize` returns
`TOKEN_SYMBOL` and the string starts with `?` and has length > 1
(`lib/frontend/parser.cpp` lines 888–899), the AST node is rewritten to
`ESHKOL_LOGIC_VAR_OP` with `logic_var_op.name` copied and
`logic_var_op.var_id` initialised by the call
`eshkol_make_logic_var(token.value.c_str())`. The leading `?` is the
canonical R7RS-compatible cue — `?` is a valid identifier-start
character under R7RS lexical rules, so no grammar change is required.

The registry (`lib/core/logic.cpp` lines 21–151) provides
deduplication: distinct textual occurrences of `?x` resolve to the same
`var_id`. Two structures guard concurrent use:

- A `std::mutex g_var_mutex` serialises the `find-or-register`
  critical section of `eshkol_make_logic_var`.
- A `std::atomic<uint64_t> g_var_count` is the monotonic ID counter,
  and a parallel `std::atomic<size_t> g_var_name_pool_offset` tracks
  the static name pool, using `compare_exchange_weak` to allocate
  contiguous slots without locking.

Capacity is `LOGIC_VAR_MAX = 65536` (line 29) and per-name storage is
fixed at 64 bytes (line 40) for a total of 4 MiB of static pool. Names
longer than 63 characters are truncated (line 109).

The companion function `eshkol_intern_predicate` (lines 76–105) maintains
a second pool of 64 KiB / 1024 slots for fact-predicate canonical
pointers; this is what makes pointer equality work in
`eshkol_unify`'s structural case (see §4.6 below).

The runtime predicate `eshkol_is_logic_var(const eshkol_tagged_value_t*)`
(lines 159–162) is the canonical type test:

```c
bool eshkol_is_logic_var(const eshkol_tagged_value_t* tv) {
    if (!tv) return false;
    return tv->type == ESHKOL_VALUE_LOGIC_VAR;
}
```

For test isolation, `eshkol_logic_registry_reset` (lines 62–74) drops
both the variable-name and predicate intern tables and is wired to
`(reset-tests!)`. It is documented as not thread-safe.

### 4.2 Substitutions

A substitution is a flat, immutable mapping from variable IDs to terms.
The struct (`inc/eshkol/core/logic.h` lines 40–45):

```c
struct eshkol_substitution {
    uint32_t num_bindings;
    uint32_t capacity;
    /* Followed by: uint64_t var_ids[capacity] */
    /* Followed by: eshkol_tagged_value_t terms[capacity] */
};
```

Two macros recover the parallel arrays (lines 48, 51):

```c
SUBST_VAR_IDS(s) ≡ (uint64_t*)((uint8_t*)s + sizeof(eshkol_substitution_t))
SUBST_TERMS(s)   ≡ (eshkol_tagged_value_t*)((uint8_t*)s
                     + sizeof(eshkol_substitution_t)
                     + s->capacity * sizeof(uint64_t))
```

Parallel arrays — rather than an array of `{id, term}` pairs — keep
the var-ID scan cache-dense: an `n`-binding lookup touches `8n` bytes
of var IDs before any term load.

`eshkol_make_substitution(arena, capacity)` (lines 207–211) defaults
to capacity 8 when zero is requested. `eshkol_extend_subst`
(lines 213–250) is the only mutator and is **copy-on-extend**: it
allocates a fresh substitution with at least double the old capacity,
`memcpy`s the existing bindings, appends the new
`(var_id, term)` pair, and returns the new substitution. The old
substitution is never modified; the arena reclaims it when reset. This
discipline enables backtracking through failed unification branches
without any undo bookkeeping.

`eshkol_subst_lookup` (lines 252–263) is a linear scan over the var-ID
array. For the substitution sizes typical of consciousness-engine
workloads (single-digit to low-double-digit bindings per query branch)
this is uniformly faster than a hash; predicate-indexed and hashed
lookup are slated for v1.2.

### 4.3 Walk

The walk operation resolves a term through a substitution by chasing
variable bindings to a fixed point. The shallow variant
(`eshkol_walk`, `lib/core/logic.cpp` lines 267–287):

$$
\textsc{walk}(t, \sigma) \;=\; \begin{cases}
\textsc{walk}(\sigma(v), \sigma) & \text{if } t \text{ is a logic var } v \text{ and } v \in \mathrm{dom}(\sigma) \\
t & \text{otherwise}
\end{cases}
$$

A `null` substitution short-circuits the loop, returning `t` as-is.

The deep variant `eshkol_walk_deep` (lines 289–338) extends this
recursively into compound structures: when the shallow walk yields a
fact, the deep walk constructs a fresh fact (in the arena) whose
arguments are themselves deep-walked. A depth limit of
`WALK_DEEP_MAX_DEPTH = 10000` (line 289) guards against pathological
chains, emitting a warning and returning the term unchanged at the
limit.

### 4.4 Occurs check

The occurs check (`lib/core/logic.cpp` lines 342–372) is mandatory in
`eshkol_unify`: it prevents the construction of circular bindings such
as $\{?x \mapsto f(?x)\}$ that would cause `walk` to loop indefinitely.
The algorithm:

$$
\textsc{occurs}(v, t, \sigma) \;=\; \begin{cases}
\mathrm{true} & \text{if } \textsc{walk}(t,\sigma) \text{ is the logic var } v \\
\bigvee_{i} \textsc{occurs}(v, t.\mathrm{args}[i], \sigma) & \text{if } \textsc{walk}(t,\sigma) \text{ is a fact} \\
\mathrm{false} & \text{otherwise}
\end{cases}
$$

A depth limit of `OCCURS_CHECK_MAX_DEPTH = 1000` (line 342) caps the
recursion. At the limit the check returns `false` rather than
overflowing the stack — this is a soundness-vs-safety tradeoff the
implementer explicitly chose; in practice, no production query has
been observed to reach this depth.

### 4.5 Tagged-value equality for unification

Before dispatching, `eshkol_unify` reduces both terms via
`eshkol_walk` and then tests equality. Equality for unification is
**structural** under `(equal? …)` semantics with one extension: two
`ESHKOL_VALUE_LOGIC_VAR` tagged values are equal iff their `var_id`
fields are equal (`lib/core/logic.cpp` lines 383–391):

```c
static bool tagged_values_equal(const eshkol_tagged_value_t* a,
                                const eshkol_tagged_value_t* b) {
    if (a->type == ESHKOL_VALUE_LOGIC_VAR && b->type == ESHKOL_VALUE_LOGIC_VAR) {
        return a->data.int_val == b->data.int_val;
    }
    return eshkol_deep_equal(a, b);
}
```

`eshkol_deep_equal` lives in `arena_memory.cpp` and recurses through
cons cells, strings (by content), bignums (by numeric value), tensors,
and vectors; for everything else it falls back to bit-pattern equality
of the tagged value.

### 4.6 Unification

`eshkol_unify(arena, t1, t2, subst)` (`lib/core/logic.cpp` lines
395–462) implements Robinson's unification with the
Martelli–Montanari refinement (walk both terms before any case
dispatch). It runs in $O(n \cdot m)$ time where $n$ is the total size
of both terms and $m$ is the number of bindings in `subst`, plus the
occurs-check contribution which is bounded by the same term traversal.

The full case analysis:

```
unify(t1, t2, σ):
    w1 = walk(t1, σ)
    w2 = walk(t2, σ)
    if tagged_values_equal(w1, w2):              return σ
    if w1 is LOGIC_VAR:
        if occurs(w1.var_id, w2, σ):             return NULL
        return extend(σ, w1.var_id, w2)
    if w2 is LOGIC_VAR:
        if occurs(w2.var_id, w1, σ):             return NULL
        return extend(σ, w2.var_id, w1)
    if w1, w2 are HEAP_PTR with subtype FACT:
        if w1.predicate ≠ w2.predicate:          return NULL
        if w1.arity ≠ w2.arity:                  return NULL
        σ' = σ
        for i in 0..w1.arity-1:
            σ' = unify(w1.args[i], w2.args[i], σ')
            if σ' is NULL:                       return NULL
        return σ'
    return NULL
```

Predicate equality is **pointer comparison** (line 436), which works
because every predicate symbol routes through
`eshkol_intern_predicate`. A defensive fallback (lines 437–442)
performs a `strcmp` if either side has an un-interned predicate
(observed in `kb-load` scenarios when predicates are reconstructed from
on-disk serialisations).

The substitution is threaded left-to-right through the argument list:
binding $\{?x \mapsto \mathit{alice}\}$ that emerges from arg 0 is
visible to the recursive call on arg 1. Failure on any argument
discards every binding accumulated so far simply by returning `NULL` —
the caller continues to hold the original `σ`, and arena allocations
made on the failed branch are recovered by the next arena reset.

#### 4.6.1 Worked example

Unify $(parent\ ?x\ (child\ ?y))$ with $(parent\ \mathit{alice}\ (child\ \mathit{bob}))$ under $\sigma_0 = \{\}$:

1. Both heads are facts with the same interned predicate `parent` and
   arity 2 — enter the structural case.
2. Recurse on arg 0: walk $?x$ → $?x$, walk $\mathit{alice}$ →
   $\mathit{alice}$. Case "w1 is logic var" applies. Occurs check:
   does $?x$ appear in $\mathit{alice}$? No. Extend:
   $\sigma_1 = \{?x \mapsto \mathit{alice}\}$.
3. Recurse on arg 1: walk both yields two fresh facts with predicate
   `child`, arity 1. Structural case.
4. Recurse on `child`'s arg 0: walk $?y$ → $?y$, walk $\mathit{bob}$ →
   $\mathit{bob}$. Extend:
   $\sigma_2 = \{?x \mapsto \mathit{alice}, ?y \mapsto \mathit{bob}\}$.

Result $\sigma_2$. Now $\textsc{walk}(?x, \sigma_2) = \mathit{alice}$,
$\textsc{walk}(?y, \sigma_2) = \mathit{bob}$.

### 4.7 Facts

The fact struct (`inc/eshkol/core/logic.h` lines 60–65):

```c
struct eshkol_fact {
    uint64_t predicate;    /* Pointer to interned symbol (HEAP_SUBTYPE_SYMBOL) */
    uint32_t arity;
    uint32_t _pad;
    /* Followed by: eshkol_tagged_value_t args[arity] */
};
```

with the macro `FACT_ARGS(f)` recovering the trailing argument array
(line 68). `eshkol_make_fact` (lines 466–480) arena-allocates the fact
with header subtype 13 and `memcpy`s the caller's argument array.

The `eshkol_make_fact_tagged` wrapper (lines 703–735) is what
codegen actually calls; it interns the predicate string before
constructing the fact, so two facts whose Scheme-level predicate
symbol prints the same will compare equal under
`f1.predicate == f2.predicate`.

### 4.8 Knowledge base

The KB struct (`inc/eshkol/core/logic.h` lines 77–81):

```c
struct eshkol_knowledge_base {
    uint32_t num_facts;
    uint32_t capacity;
    eshkol_fact_t** facts;    /* Arena-allocated array of fact pointers */
};
```

Initial capacity is `KB_INITIAL_CAPACITY = 16` (`lib/core/logic.cpp`
line 484). `eshkol_kb_assert` (lines 505–521) appends a fact pointer
and doubles the array (via fresh arena allocation + `memcpy`) when full.

`eshkol_kb_retract_tagged` (`lib/core/logic_builtins.cpp` lines
124–162) removes a fact by **pointer identity**: it scans the array,
finds the matching pointer, and shifts the tail down by one. There is
no structural-equality fallback — the caller must retain the fact
pointer that was originally asserted.

### 4.9 KB query

`eshkol_kb_query` (`lib/core/logic.cpp` lines 523–596) performs a
linear scan over `kb->facts`, attempting to unify the query pattern's
arguments against each fact's arguments under a shared base
substitution. The algorithm:

```
kb-query(kb, pattern, σ₀):
    result = ()
    σ₀ = σ₀ or make-substitution(arena, 8)
    for fact in kb.facts:
        if pattern.predicate ≠ 0 and fact.predicate ≠ 0 and
           pattern.predicate ≠ fact.predicate: continue
        if pattern.arity ≠ fact.arity: continue
        σ = σ₀
        success = true
        for j in 0..pattern.arity-1:
            σ = unify(pattern.args[j], fact.args[j], σ)
            if σ is NULL: success = false; break
        if success:
            result = cons(σ, result)            ; arena CONS cell
    return result
```

The result is a cons list of every matching substitution, built in
reverse order with no final reversal. Each cons cell is arena-allocated
with `HEAP_SUBTYPE_CONS` (lines 573–574). The caller can `for-each`
through the list and `walk` the query's logic variables in each
substitution to retrieve concrete bindings.

The variant `eshkol_kb_query_prefix` (lines 603–664) relaxes the arity
check to `pattern.arity <= fact.arity`, unifying only the first
`pattern.arity` argument positions. This supports KBs with
provenance-extended fact tails (e.g., where a fact's last argument
is a source citation that the pattern does not need to match).

The cost is $O(N \cdot k \cdot u)$ where $N$ is the number of facts,
$k$ is the average pattern arity, and $u$ is the cost of unification
per argument pair. For a KB of a few thousand facts and small arities
this is empirically negligible compared to surrounding work.

Predicate indexing (an inverted index from predicate symbol to facts
asserting it) is planned for v1.2 to bring the cost to $O(N_p \cdot k
\cdot u)$ where $N_p$ is the number of facts sharing the queried
predicate.

---

## 5. Active inference

The active-inference subsystem is structured around three concepts:
the factor graph (a discrete generative model), loopy belief
propagation (sum-product in log-space), and free-energy functionals
(variational and expected). All computation happens in log-probability
space using the log-sum-exp identity to avoid underflow.

### 5.1 Factor graph

The struct (`inc/eshkol/core/inference.h` lines 55–67):

```c
struct eshkol_factor_graph {
    uint32_t num_vars;         /* |V| */
    uint32_t num_factors;      /* current factor count */
    uint32_t max_factors;      /* factor-array capacity (starts at 16) */
    uint32_t _pad;
    double** beliefs;          /* beliefs[v][s] = log q(X_v = s) */
    uint32_t* var_dims;        /* var_dims[v] = |state space of X_v| */
    eshkol_factor_t** factors; /* factor array */
    double** msg_fv;           /* factor-to-variable messages */
    double** msg_vf;           /* variable-to-factor messages */
    uint32_t total_messages;   /* sum of factor arities */
    bool* observed;            /* observed[v] = true → clamped */
};
```

Construction is `eshkol_make_factor_graph` (`lib/core/inference.cpp`
lines 149–194). Beliefs are initialised uniformly in log-space to
$\log(1/d_v)$ where $d_v$ is the variable's state-space size
(lines 175–181); `safe_dim` (line 177) coerces a zero dimension to 1 to
avoid `log(0)`. Message arrays are deferred: `msg_fv` and `msg_vf` are
`NULL` until the first call to `eshkol_fg_infer`, at which point
`allocate_messages` (lines 253–293) populates them. The `observed` bool
array stays `NULL` until the first `fg-observe!` call.

### 5.2 Factor

The factor struct (`inc/eshkol/core/inference.h` lines 37–43):

```c
struct eshkol_factor {
    uint32_t num_vars;         /* arity */
    uint32_t cpt_size;         /* ∏ dims */
    double*  cpt;              /* log-probability tensor (row-major) */
    uint32_t* dims;            /* per-variable state-space size */
    /* Followed by: uint32_t var_indices[num_vars] */
};
```

Factor objects do not get their own heap subtype; they are
arena-allocated raw via `arena_allocate_aligned`
(`lib/core/inference.cpp` lines 128–137) and live as raw pointers
inside the factor graph's `factors` array. Variable indices live
inline after the struct, recovered by the macro `FACTOR_VAR_INDICES(f)`
(line 46).

The CPT is **flat, row-major, log-space**. For a factor over variables
with state-space sizes $d_0, d_1, \dots, d_{n-1}$, the flat index of
the configuration $(s_0, s_1, \dots, s_{n-1})$ is

$$
\mathrm{idx}(s_0, \dots, s_{n-1}) \;=\; \sum_{k=0}^{n-1} s_k \cdot \prod_{j=k+1}^{n-1} d_j.
$$

The implementation decodes a flat index back into per-variable states
by repeated modular reduction (`lib/core/inference.cpp` §`compute_factor_to_var_message`, lines 333–337):

```c
uint32_t remaining = config;
for (int32_t k = (int32_t)f->num_vars - 1; k >= 0; k--) {
    state[k] = remaining % f->dims[k];
    remaining /= f->dims[k];
}
```

### 5.3 Log-space arithmetic

All probabilities are stored as natural-log values; messages and CPTs
that would underflow as raw probabilities remain in range. Three
primitives (`lib/core/inference.cpp` lines 91–124):

- `LOG_ZERO = -1e30` (line 91) is the sentinel for $\log 0$. It is
  chosen finite, rather than `-INFINITY`, so that arithmetic on it
  does not propagate `NaN`.

- Pairwise log-sum-exp (lines 94–99):

  $$
  \mathrm{logsumexp2}(a, b) \;=\; \max(a, b) + \log\bigl(1 + \exp(-|a - b|)\bigr).
  $$

  Both `LOG_ZERO + 1.0` short-circuits suppress contributions that are
  computationally indistinguishable from zero.

- Array log-sum-exp (lines 102–114):

  $$
  \mathrm{logsumexp}(\mathbf{x}) \;=\; m + \log\Bigl(\textstyle\sum_i \exp(x_i - m)\Bigr), \quad m = \max_i x_i.
  $$

- In-place log-normalisation (lines 117–124):

  $$
  \mathrm{log\_normalize}(\mathbf{x}) : \quad x_i \leftarrow x_i - \mathrm{logsumexp}(\mathbf{x}).
  $$

  After this operation, $\sum_i \exp(x_i) = 1$ up to floating-point.

### 5.4 Sum-product belief propagation

`eshkol_fg_infer` (`lib/core/inference.cpp` lines 396–481) implements
loopy belief propagation with the standard sum-product algorithm,
entirely in log-space. The message schedule is **synchronous flooding**:
all variable-to-factor messages are updated, then all
factor-to-variable messages, then beliefs, then a convergence test.

#### 5.4.1 Message index

Messages are stored in flat arrays `msg_fv[]` and `msg_vf[]` indexed by
the helper `get_msg_idx(fg, fi, vi)` (lines 296–303):

$$
\mathrm{msgidx}(f_i, v_i) \;=\; v_i + \sum_{j < i} |\mathrm{ne}(f_j)|.
$$

This is computed by linear scan over the factors preceding $f_i$,
which is $O(F)$ per index but acceptable for the factor counts
expected in v1.1 (≤ 32 typical).

#### 5.4.2 Factor-to-variable message

Implemented by `compute_factor_to_var_message`
(`lib/core/inference.cpp` lines 312–356):

$$
m_{f \to v}(x_v) \;=\; \log\!\!\sum_{\mathbf{x}_{\backslash v}} \exp\!\Biggl[ \log f(\mathbf{x}) + \!\!\sum_{u \in \mathrm{ne}(f) \setminus v} m_{u \to f}(x_u) \Biggr].
$$

Concretely:

1. Initialise the output message to `LOG_ZERO` for every target state
   (lines 321–323).
2. Enumerate every flat configuration of the factor's joint state
   space ($\prod d_k$ configurations) (lines 331–352).
3. For each configuration, accumulate the log-CPT entry plus the
   incoming variable-to-factor messages for every variable other than
   the target (lines 340–347).
4. Merge into the target state's output bin via `logsumexp2`
   (lines 350–351).
5. Log-normalise (line 355).

This is the most expensive operation; its cost per call is
$O(\prod_k d_k \cdot |\mathrm{ne}(f)|)$.

#### 5.4.3 Variable-to-factor message

Implemented by `compute_var_to_factor_message`
(`lib/core/inference.cpp` lines 363–394):

$$
m_{v \to f}(x_v) \;=\; \sum_{g \in \mathrm{ne}(v) \setminus f} m_{g \to v}(x_v).
$$

That is, log-sum (= probability-product) all factor-to-variable
messages reaching $v$, **except** from the target factor $f$ itself.
The implementation walks every factor in the graph, locates each
factor's edges that connect to $v$, and accumulates (lines 378–391).
Log-normalisation closes (line 393).

#### 5.4.4 Belief update

After the two message passes complete, beliefs are recomputed as the
log-product of all incoming factor messages
(`lib/core/inference.cpp` lines 448–472):

$$
b(x_v) \;=\; \sum_{f \in \mathrm{ne}(v)} m_{f \to v}(x_v),
$$

then log-normalised. Crucially, **observed variables are skipped**
(line 452): their beliefs were clamped by `fg-observe!` and must not
be overwritten by message-driven updates. This is the BP-equivalent of
classical evidence-clamping in junction trees.

#### 5.4.5 Convergence

The main loop tracks `max_delta` (lines 405, 421–423, 440–443): the
maximum absolute change of any message component across the current
iteration. When `max_delta < tolerance` (default $10^{-6}$, see
`eshkol_fg_infer_tagged` line 768), the function sets `converged = true`
and returns. Otherwise, after `max_iterations` passes (default 20,
line 763), it returns with whatever beliefs were last produced.

For tree-structured factor graphs, this algorithm is exact and
converges in $O(\mathrm{diameter})$ iterations. For graphs with cycles,
the algorithm is "loopy BP" — convergence is not guaranteed in
general, but in practice the messages stabilise on most well-posed
models within a few dozen iterations.

The schedule is synchronous (flooding). Asynchronous and
priority-scheduled variants (residual BP, tree-reweighted BP) would
likely improve convergence on highly loopy graphs; they are deferred.

### 5.5 Variational free energy

`eshkol_free_energy(fg, observations, num_obs)` (`lib/core/inference.cpp`
lines 485–557) returns the scalar

$$
F \;=\; \mathbb{E}_q[\log q(s)] \;-\; \mathbb{E}_q[\log p(o, s)],
$$

with $q$ approximated by the **mean-field** product of current
beliefs:

$$
q(s_0, s_1, \dots, s_{n-1}) \;\approx\; \prod_v q(s_v) \;=\; \prod_v \exp(\mathrm{beliefs}[v][s_v]).
$$

Three terms enter the implementation:

1. **Entropy** (lines 502–511): the negative-entropy term of $F$,
   computed as the sum of marginal entropies under the mean-field
   factorisation:

   $$
   -H(q) \;=\; \sum_v \sum_s q(x_v = s) \log q(x_v = s).
   $$

   The guard `q > 1e-30` (line 507) prevents $0 \log 0 = \mathrm{NaN}$.

2. **Expected log-joint** (lines 513–537): the model accuracy term,

   $$
   \mathbb{E}_q[\log p(s)] \;=\; \sum_f \sum_{\mathbf{x}} q(\mathbf{x}) \cdot \log f(\mathbf{x}).
   $$

   For each factor and each configuration, the implementation decodes
   the configuration, computes $\log q(\mathbf{x}) = \sum_k \mathrm{beliefs}[v_k][s_k]$
   (lines 522–530), exponentiates, and accumulates against the
   log-CPT entry.

3. **Observation contribution** (lines 540–551): when an observation
   tensor is provided as flat pairs $(\mathit{var\_idx}, \mathit{obs\_state})$
   (so a tensor of length $2N_o$ for $N_o$ observations), each
   observation adds $\mathrm{beliefs}[v][o_v]$ to the expected
   log-joint. This is the analytic form of the surprise term
   $\log p(o \mid s)$ when the observation likelihood is treated as a
   one-hot indicator at the clamped state.

The final scalar is

$$
F \;=\; -H(q) \;-\; \mathbb{E}_q[\log p(o, s)].
$$

**Observation format.** The observations tensor is a 1-D tensor of
$2N_o$ doubles: `#(var0_idx var0_state var1_idx var1_state …)`. The
tagged wrapper `eshkol_free_energy_tagged` (lines 812–847) infers
$N_o = \lfloor \mathrm{total\_elements}/2 \rfloor$ (line 834) and
ignores any trailing element if the count is odd.

### 5.6 Expected free energy

`eshkol_expected_free_energy(arena, fg, action_var, action_state)`
(`lib/core/inference.cpp` lines 559–632) returns the prospective
quantity used to compare candidate actions:

$$
G(a) \;=\; \mathbb{E}_{q(o, s \mid a)}[\log q(s \mid a) - \log p(o, s)].
$$

The Friston decomposition into pragmatic (goal-seeking, $-\log p$) and
epistemic (uncertainty-reducing, $+\log q$) terms is implemented
directly. For every factor that involves the action variable
(lines 583–598), the implementation enumerates only the configurations
in which `action_var` takes `action_state` (line 611), computes the
mean-field $\log q$ over the *non-action* variables in that factor
(lines 614–619), and accumulates:

$$
G \;\mathrel{{+}{=}}\; q(\mathbf{x}_{\backslash a}) \cdot \bigl[ \log q(\mathbf{x}_{\backslash a}) - \log f(\mathbf{x}) \bigr].
$$

Or in code (line 624–626):

```c
double q = exp(log_q);
if (q > 1e-30) {
    efe -= q * f->cpt[config];    // pragmatic:  -q · log p
    efe += q * log_q;              // epistemic:  +q · log q
}
```

Lower $G(a)$ indicates a preferred action; an active-inference agent
selects $a^* = \arg\min_a G(a)$.

### 5.7 Evidence clamping with `fg-observe!`

`eshkol_fg_observe_tagged` (`lib/core/inference.cpp` lines 982–1029)
applies hard evidence to a variable. After validation
(lines 994–1012) it:

1. Clamps the variable's log-beliefs (lines 1014–1018): the observed
   state gets $\log 1 = 0$, every other state gets $-10^{30}$.
2. Lazily allocates the `observed` boolean array if it is `NULL`
   (lines 1021–1023), then sets `observed[var_id] = true`
   (line 1025). The BP loop honours this flag and skips message-driven
   updates to clamped variables (§5.4.4).

The function returns a boolean `#t` if the clamp was applied,
`#f` if validation failed. The implementation note (line 451) is
explicit that this clamp must match the VM path in `vm_inference.c`
line 388 for AOT and VM agreement.

### 5.8 Online CPT learning with `fg-update-cpt!`

`eshkol_fg_update_cpt_tagged` (`lib/core/inference.cpp` lines
885–962) enables online model revision: the caller supplies a new
tensor or vector of $|\mathrm{cpt}|$ log-probabilities, the function
overwrites the factor's CPT element-by-element, and — critically —
resets the message arrays:

```c
fg->msg_fv = NULL;
fg->msg_vf = NULL;
```

(lines 956–957). The next call to `fg-infer!` will re-enter
`allocate_messages` (§5.4) and start from uniform messages. Without
this reset, BP would resume from messages computed under the previous
CPT and could converge to a wrong fixed point or oscillate.

The function accepts the new CPT as either a homogeneous tensor
(`#(…)`, the fast path) or a heterogeneous vector
(`(vector …)`, the ergonomic path when building CPTs from computed
per-element expressions — `extract_doubles_from_vector` handles the
mixed `INT64`/`DOUBLE` element types and the exactness-flag masking,
lines 65–87).

### 5.9 Marginal extraction (`fg-marginal`) and entropy (`fg-entropy`)

`eshkol_fg_marginal_tagged` (`lib/core/logic_builtins.cpp` lines
21–79) extracts the marginal belief vector for a single variable as a
1-D tensor of probabilities. It (i) walks the log-beliefs, (ii)
subtracts the max for numerical stability, (iii) exponentiates, (iv)
sums and normalises, and (v) returns the result as a tensor of
`var_dims[v]` doubles.

`eshkol_fg_entropy_tagged` (lines 81–122) computes the Shannon
entropy of the same marginal:

$$
H_v \;=\; -\sum_s q(x_v = s) \log q(x_v = s).
$$

It uses a fixed-size stack array `probs[256]` (line 108), which caps
the supported per-variable state space at 256 for this builtin
specifically.

### 5.10 Tagged-value dispatch for inference builtins

Every active-inference builtin reaches the runtime through an
`_tagged` wrapper that accepts pointers to caller-stack tagged-value
slots and writes its result through a final `result` pointer
(LLVM `sret`-style convention). For example,
`eshkol_fg_infer_tagged` (`lib/core/inference.cpp` lines 744–810)
unpacks `max_iters_tv` (default 20), runs `eshkol_fg_infer` with a
hardcoded tolerance of $10^{-6}$ (line 768), then allocates a fresh
1-D `HEAP_SUBTYPE_TENSOR` tagged value (lines 778–810), exponentiates
every belief (converting back from log-space), and packs the flat
result.

The total length of the returned belief tensor is
$\sum_v d_v$, in the order
$\bigl[q_{0,0}, \dots, q_{0,d_0-1}, q_{1,0}, \dots, q_{N-1,d_{N-1}-1}\bigr]$
(lines 772–775, 799–806).

---

## 6. Global workspace

The global workspace is the engine's attentional substrate. Its job is
to take a roster of competing "specialist" modules, run them all on the
current shared content, then arbitrate among their proposals using
softmax over salience and broadcast the winner.

### 6.1 Workspace and module structures

The module struct (`inc/eshkol/core/workspace.h` lines 40–44):

```c
struct eshkol_workspace_module {
    char* name;                        /* arena-allocated string */
    eshkol_tagged_value_t process_fn;  /* closure tagged value (16 bytes) */
    double salience;                   /* last computed normalised salience */
};
```

A module is therefore **32 bytes** on a 64-bit ABI: 8 bytes for the
name pointer + 16 bytes for the closure tagged value + 8 bytes for the
salience double. The codegen layer hard-codes this 32-byte stride when
walking the module array (see §6.4 below).

The workspace struct (`inc/eshkol/core/workspace.h` lines 58–65):

```c
struct eshkol_workspace {
    uint32_t num_modules;
    uint32_t max_modules;
    uint32_t dim;                      /* content vector dimension */
    uint32_t step_count;               /* cognitive cycle counter */
    double*  content;                  /* shared content buffer */
    /* Followed by: eshkol_workspace_module_t modules[max_modules] */
};
```

The trailing module array is recovered by the macro `WS_MODULES(ws)`
at offset `sizeof(eshkol_workspace_t)` from the struct base. The
content buffer is allocated separately via `arena_allocate_aligned`
(`lib/core/workspace.cpp` lines 44–46) and pointer-stored into
`content` — it is *not* inline.

Construction (`eshkol_make_workspace`, lines 26–49) zeros the content
buffer; the workspace begins each life with a quiescent shared state.

### 6.2 Module registration

`eshkol_ws_register` (`lib/core/workspace.cpp` lines 51–70) appends a
module. It arena-allocates a fresh copy of the name string (lines
60–64) and stores the caller's `process_fn` tagged value verbatim.
Modules can be registered up to `max_modules` (the static capacity set
at construction); past that the call is a no-op (line 54).

The Scheme protocol for a module is:

```scheme
(lambda (content-tensor)
  (cons salience-score                ; double
        proposal-tensor))             ; tensor of dimension ws.dim
```

with no other shape constraints.

### 6.3 The cognitive cycle in C

The actual `ws-step!` implementation is split across LLVM codegen and
two C helpers. The C side does two things:

#### 6.3.1 Wrapping content into a tensor

`eshkol_ws_make_content_tensor` (`lib/core/workspace.cpp` lines
171–211) wraps the workspace's raw `double*` content as a 1-D tensor
tagged value so that Eshkol closures can receive it. The tensor uses
the canonical Eshkol layout:

```c
struct ws_tensor_layout {
    uint64_t* dimensions;     /* points at a one-entry array {dim} */
    uint64_t  num_dimensions; /* = 1 */
    int64_t*  elements;       /* double bit patterns as int64 */
    uint64_t  total_elements; /* = dim */
};
```

(lines 164–169). The helper performs a **copy**: it allocates a fresh
`int64_t[dim]` and writes each double as its bit pattern via a
`union { double; int64_t; }` cast (lines 203–207). Mutations of the
tensor by a module therefore do not leak back into the workspace.

#### 6.3.2 Result aggregation and broadcast

`eshkol_ws_step_finalize` (`lib/core/workspace.cpp` lines 213–296)
consumes the `tagged_value[num_modules]` array of per-module results
and performs the softmax competition. The implementation:

1. **Result extraction** (lines 222–254). Each result is expected to
   be a heap pointer to a cons cell — that is, two adjacent
   `eshkol_tagged_value_t`s (a 32-byte object). The `car`
   yields the salience (double; an `INT64` is promoted to double; any
   other type produces salience 0) and the `cdr` yields a pointer to
   a tensor. Invalid results receive salience $-10^{30}$ and proposal
   `NULL`, which softmax then maps to effectively zero weight.

2. **Softmax normalisation** (lines 258–270). Two stack arrays of
   length 16 (`salience[16]`, `exp_sal[16]`) cap the per-step module
   count at 16. The softmax is the standard max-subtracting form:

   $$
   \sigma(z_i) \;=\; \frac{\exp(z_i - \max_j z_j)}{\sum_k \exp(z_k - \max_j z_j)}.
   $$

   The implicit temperature is $T = 1.0$; clients control the
   effective temperature by scaling the salience values their closures
   return.

3. **Winner selection** (lines 271–282). The module with the largest
   normalised salience is the winner. Each module's `salience` field
   is updated with its normalised value, exposing the post-softmax
   distribution for downstream inspection.

4. **Broadcast** (lines 284–293). The winner's proposal tensor is
   copied into `ws->content`. The copy is performed element-by-element
   through the same `union { int64_t; double; }` cast used for content
   wrapping; the post-copy `ws->content` then contains genuine
   doubles. If the proposal tensor is shorter than `ws->dim`, only the
   available elements are copied; if longer, the tail is dropped
   (line 287).

5. **Step counter** `ws->step_count` is incremented (line 295).

### 6.4 Codegen-emitted closure loop

`LogicWorkspaceCodegen::codegenWSStep` (`lib/backend/logic_workspace_codegen.cpp`
lines 536–661) is the LLVM IR side. It must live in IR rather than C
because Eshkol closures cannot be called from plain C — they require
the closure calling convention managed by `codegenClosureCall`. The
emitted IR:

1. **Extracts** the workspace pointer from the tagged value
   (`ExtractValue` at field index 4, `IntToPtr`, lines 547–549).

2. **Early-exits** if `num_modules == 0` (lines 555–564). This branch
   short-circuits to the merge block without calling any helpers; the
   no-modules workspace passes through unchanged.

3. **Loads `dim` and `content`** from the workspace struct at
   `+8` and `+16` byte offsets (lines 568–576). These hard-coded
   offsets reflect the struct layout in §6.1.

4. **Calls** `eshkol_ws_make_content_tensor` (lines 580–586), passing
   the arena, the raw `content` double pointer, the dim, and a stack
   `result_a` slot.

5. **Allocates** a 16-slot `results_arr` for the per-module return
   values (line 590–591). The cap of 16 here is the same as the cap in
   `eshkol_ws_step_finalize`.

6. **Loops** over modules via the alloca/store/load pattern
   (lines 598–611). The loop counter is a stack `i32`; the loop is
   *not* PHI-based because `codegenClosureCall` synthesises many
   internal basic blocks that would corrupt PHI predecessors (this is
   the design lesson recorded in `project memory > closure-in-loop
   PHI bug`).

7. **Indexes** into the module array at offset
   `+24 + 32·i` from the workspace base (the `+24` skips the workspace
   header struct; `+8` within the module skips the `name*` to reach
   `process_fn`) (lines 615–627).

8. **Invokes** the module closure via `codegenClosureCall` with
   `content_tv` as the single argument (lines 629–630), storing the
   result in `results_arr[i]`.

9. **Calls** `eshkol_ws_step_finalize` after the loop (lines 644–649)
   with the workspace pointer, the results array, and the module count.

10. **Returns** the original `ws_tv` so that `(ws-step! ws)` is
    expression-friendly (line 660).

The 32-byte module stride and 24-byte workspace header offsets are
load-bearing: any change to `eshkol_workspace_t` or
`eshkol_workspace_module_t` requires a corresponding update here.

### 6.5 Sequential pseudocode for the full cycle

```text
ws-step!(ws):
    if ws.num_modules == 0: return ws
    content_tv = eshkol_ws_make_content_tensor(ws.content, ws.dim)
    for i in 0..ws.num_modules-1:
        results[i] = invoke_closure(ws.modules[i].process_fn, [content_tv])
    eshkol_ws_step_finalize(ws, results, ws.num_modules):
        for i in 0..n-1:
            (salience[i], proposals[i]) = (car results[i], cdr results[i])
        z = salience - max(salience)
        p = exp(z) / sum(exp(z))
        ws.modules[i].salience = p[i]  for each i
        winner = argmax(p)
        ws.content[0..ws.dim-1] = proposals[winner][0..min(ws.dim, |prop|)-1]
        ws.step_count += 1
    return ws
```

---

## 7. Codegen architecture

The dispatch flow for any consciousness-engine call is:

```
Scheme source
   │
   ▼
parser.cpp                 line 985: operation_for_string(op_text)
   │                       returns one of ESHKOL_*_OP
   ▼
AST node {type=ESHKOL_OP,  operation.op=ESHKOL_FG_INFER_OP, ...}
   │
   ▼
LLVMCodegen::codegenOperation       lib/backend/llvm_codegen.cpp line 8648
   │                       switch on op → logic_workspace_->codegen…(op)
   ▼
LogicWorkspaceCodegen::codegen…     lib/backend/logic_workspace_codegen.cpp
   │                       emits IR: alloca tagged slots for inputs,
   │                       alloca result slot, declare external runtime fn,
   │                       CreateCall, load result
   ▼
extern "C" eshkol_*_tagged          lib/core/{logic,inference,workspace}.cpp
   │                       unpacks inputs from tagged slots,
   │                       dispatches to plain-C implementation,
   │                       writes result to caller's slot
   ▼
eshkol_fg_infer / eshkol_unify / …  same files
   │                       core algorithm
   ▼
arena_allocate_with_header  lib/core/arena_memory.cpp
                            attaches 8-byte object header with
                            subtype = HEAP_SUBTYPE_FACTOR_GRAPH, etc.
```

The codegen layer's only contribution to performance is the
`getOrDeclareRuntimeFunc` helper (lines 82–88), which memoises the
LLVM `Function*` declaration of each runtime symbol, and the
`allocaAndStore` / `loadResult` helpers (lines 96–110) that follow
the bignum `alloca → store → call → load` pattern verbatim. There is
no special calling convention — every runtime function is declared
with `ExternalLinkage` and the platform default C ABI.

The six type-predicates (`logic-var?`, `substitution?`, `kb?`,
`fact?`, `factor-graph?`, `workspace?`) are the only consciousness-
engine builtins whose IR does not call into the runtime: they read
the 16-byte tagged value, compare the type byte against
`ESHKOL_VALUE_HEAP_PTR`, dereference the data pointer at offset −8
to read the header subtype byte, compare to the relevant
`HEAP_SUBTYPE_*` constant, and PHI-merge the result. This avoids a
runtime dispatch round-trip for what is otherwise a four-instruction
test.

### 7.1 Closure-call protocol for `ws-step!`

`ws-step!` is unique among the engine builtins because it invokes
caller-supplied closures. The path is:

1. `LogicWorkspaceCodegen::codegenWSStep` (§6.4) reads the workspace
   pointer and content buffer directly from the tagged value's data
   field.
2. It calls `eshkol_ws_make_content_tensor` to produce a tagged tensor.
3. Per-module, it calls
   `closure_cb_(fn_tv, {content_tv}, "ws-step-module", cb_context_)`
   — the `closure_cb_` is `LLVMCodegen::codegenClosureCall` registered
   via `LogicWorkspaceCodegen::setCodegenClosureCallCallback` during
   subsystem wiring.
4. Each module's result tagged value is stored into a stack array
   indexed by the loop counter.
5. After the loop, the C helper `eshkol_ws_step_finalize` performs the
   softmax and broadcast.

The reason the loop body cannot be in plain C is that `closure_cb_`
emits the full closure calling convention (extract function pointer
and environment pointer from the callable tagged value, call with
trampoline, handle multi-value returns, etc.). The C ABI has no
mechanism for invoking those.

### 7.2 Arena allocation discipline

All consciousness-engine heap allocations route through
`arena_allocate_with_header` (declared at `lib/core/logic.cpp` line
167, `lib/core/inference.cpp` line 24, `lib/core/workspace.cpp` line
20):

```c
void* arena_allocate_with_header(arena_t* arena, size_t data_size,
                                  uint8_t subtype, uint8_t flags);
```

This prepends the 8-byte object header and returns the data pointer
(i.e., the pointer the caller will store in the tagged value's data
field). The arena owns the memory; no `free()` is ever called on
these objects. When the arena is reset (between test runs, between
REPL forms, etc.), every object becomes invalid at once.

This discipline interacts with the substitution-as-immutable design
(§4.2) and the BP message reset on `fg-update-cpt!` (§5.8) — in both
cases, "stale" allocations are simply allowed to leak until the next
arena reset, with no explicit lifetime management.

---

## 8. Worked examples

### 8.1 Knowledge-base query

```scheme
(define kb (make-kb))
(kb-assert! kb (make-fact 'parent 'alice 'bob))
(kb-assert! kb (make-fact 'parent 'alice 'charlie))
(kb-assert! kb (make-fact 'parent 'bob   'dave))

;; Bare ?child is parsed as ESHKOL_VALUE_LOGIC_VAR with a fresh var_id.
(define results
  (kb-query kb (make-fact 'parent 'alice ?child)))

;; results :: (cons subst (cons subst '()))
;; — one substitution per matching fact, in reverse insertion order.

(for-each
  (lambda (subst)
    (display (walk ?child subst)) (newline))
  results)
;; → charlie
;; → bob
```

Trace at the runtime level:

1. `(make-kb)` calls `eshkol_make_kb_tagged` which arena-allocates an
   `eshkol_knowledge_base_t` with `HEAP_SUBTYPE_KNOWLEDGE_BASE` and a
   16-pointer `facts` array.
2. Each `(kb-assert! …)` calls `eshkol_kb_assert_tagged`, which
   appends the fact pointer to `kb->facts` (no doubling needed at 3
   facts).
3. `(kb-query …)` calls `eshkol_kb_query_tagged` which calls
   `eshkol_kb_query`. For each of the 3 facts:
   - First fact `(parent alice bob)`: predicate match, arity match,
     unify `'alice` with `'alice` (atom equality), then `?child` with
     `'bob` → extend to $\{?child \mapsto bob\}$. Prepend.
   - Second fact `(parent alice charlie)`: same predicate, succeeds
     with $\{?child \mapsto charlie\}$. Prepend.
   - Third fact `(parent bob dave)`: predicate match, arity match,
     unify `'alice` with `'bob` — atoms not equal, fail. Skip.
4. Result list (head → last successful): `subst{?child → charlie} →
   subst{?child → bob} → ()`.

### 8.2 Factor-graph inference with explicit numbers

A three-variable Bayes net Weather → Sprinkler → GrassWet, all
binary:

```scheme
(define fg (make-factor-graph 3 #(2 2 2)))

;; Prior over Weather: P(sunny)=0.6, P(rainy)=0.4
(fg-add-factor! fg #(0)
  #(-0.510826  -0.916291))                 ;; log 0.6, log 0.4

;; P(Sprinkler | Weather): 4 entries, indexed by (weather, sprinkler)
;; weather=sunny: P(spr=on)=0.4, P(spr=off)=0.6
;; weather=rainy: P(spr=on)=0.01,P(spr=off)=0.99
(fg-add-factor! fg #(0 1)
  #(-0.916291 -0.510826                     ;; sunny → spr ∈ {on, off}
    -4.605170 -0.010050))                   ;; rainy → spr ∈ {on, off}

;; P(GrassWet | Sprinkler, Weather): 8 entries
(fg-add-factor! fg #(1 0 2)
  #(-2.302585 -0.105361                     ;; spr=on,  sunny: wet ∈ {0.1, 0.9}
    -0.051293 -2.995732                     ;; spr=on,  rainy: wet ∈ {0.95, 0.05}
    -0.693147 -0.693147                     ;; spr=off, sunny: uniform
    -0.020203 -3.912023))                   ;; spr=off, rainy: wet ∈ {0.98, 0.02}

(fg-infer! fg 50)
;; → tensor of 6 probabilities:
;;   [P(W=sunny), P(W=rainy),
;;    P(S=on),    P(S=off),
;;    P(G=wet),   P(G=dry)]
```

To observe that the grass is wet and recompute:

```scheme
(fg-observe! fg 2 0)                         ;; var 2 (Grass) state 0 (wet)
(fg-infer! fg 50)                            ;; messages reset, BP rerun
(display (fg-marginal fg 0))                 ;; updated belief on Weather
(newline)
```

The expected behaviour: $P(W=\mathit{rainy})$ rises above its prior
$0.4$, since the observation of wet grass is more consistent with rain
than with the prior balance of sprinkler activity.

### 8.3 Workspace with hand-checkable salience

A 2-D content vector, two modules, no learning:

```scheme
(define ws (make-workspace 2 2))

;; Module A returns salience 2.0 and proposes #(1 0).
(ws-register! ws "A"
  (lambda (content) (cons 2.0 #(1.0 0.0))))

;; Module B returns salience 1.0 and proposes #(0 1).
(ws-register! ws "B"
  (lambda (content) (cons 1.0 #(0.0 1.0))))

(ws-step! ws)
```

Hand-check the softmax: salience vector $[2.0, 1.0]$.

- $\max = 2.0$; $z = [0, -1]$; $\exp z = [1, e^{-1}]$.
- $\sum = 1 + e^{-1} \approx 1.36788$.
- $p_A \approx 0.7311$, $p_B \approx 0.2689$.
- Winner: module A. Broadcast: `ws.content = #(1.0 0.0)`.
- `ws.step_count = 1`.

After the call, `(modules[0].salience, modules[1].salience) ≈
(0.7311, 0.2689)` — exposed for inspection but not directly readable
from Scheme (no accessor exists in v1.1).

---

## 9. Performance characteristics

### 9.1 Logic engine

- **Variable interning:** $O(n)$ on the variable count for the
  first-time `eshkol_make_logic_var(name)` (linear scan against the
  registry); $O(1)$ for the subsequent name pool write under
  compare-exchange. With 65536-slot capacity and 64-byte slots, the
  registry occupies 4 MiB of static memory.
- **Substitution lookup:** $O(k)$ linear scan over $k$ bindings.
- **Substitution extend:** $O(k)$ for the copy plus an arena
  allocation; capacity doubling amortises the work over multiple
  extends.
- **Unification:** $O(n \cdot m)$ in the worst case for terms of total
  size $n$ over a substitution with $m$ bindings; in practice
  dominated by the term traversal, since $m$ is small (single-digit
  bindings per branch).
- **Occurs check:** $O(n)$ in the term size, with `OCCURS_CHECK_MAX_DEPTH = 1000`
  imposing a hard cap.
- **`kb-query`:** $O(N \cdot k \cdot u)$ where $N = |\mathrm{kb}|$,
  $k$ is average arity, $u$ is unification cost per pair. Predicate
  indexing would reduce $N$ to $N_p$ (number of facts sharing the
  queried predicate).
- **`kb-retract!`:** $O(N)$ for the pointer scan plus $O(N)$ for the
  shift.

### 9.2 Active inference

- **Belief initialisation:** $O(\sum_v d_v)$.
- **Per BP iteration:** Variable-to-factor: $O\bigl(\sum_f |\mathrm{ne}(f)|
  \cdot d_{\max} \cdot F\bigr)$ in the worst case (the inner factor scan
  in `compute_var_to_factor_message`). Factor-to-variable:
  $O\bigl(\sum_f |\mathrm{ne}(f)| \cdot \prod_k d_k\bigr)$, dominated
  by the configuration enumeration of each factor's joint state space.
  Belief update: $O\bigl(\sum_v d_v \cdot |\mathrm{ne}(v)|\bigr)$.
- **Convergence:** in $\le 20$ iterations by default; reduced by
  tight `tolerance`. Tree-structured graphs converge in
  $O(\mathrm{diameter})$ iterations; loopy graphs may not converge,
  but the iteration cap guarantees termination.
- **Free energy:** $O\bigl(\sum_v d_v + \sum_f \prod_k d_k\bigr)$ —
  dominated by factor-config enumeration.
- **Expected free energy:** $O\bigl(\sum_{f : a \in \mathrm{ne}(f)} \prod_k d_k\bigr)$ —
  only factors containing the action variable contribute.
- **Marginal/entropy:** $O(d_v)$ per variable.

### 9.3 Global workspace

- **Per `ws-step!`:** $O(M \cdot c_{\mathrm{module}})$ where $M$ is the
  module count and $c_{\mathrm{module}}$ is the per-module closure
  cost. The softmax is $O(M)$.
- **Tensor wrapping:** $O(d)$ copy of doubles into int64 bit
  patterns.
- **Broadcast:** $O(\min(d_{\mathrm{ws}}, d_{\mathrm{proposal}}))$
  element-by-element copy.

### 9.4 Copy-vs-pointer tradeoffs

- The substitution is **copied** on every extend; this trades
  arena traffic for backtracking simplicity.
- The content tensor passed to each module is **copied** from
  `ws->content` once per `ws-step!`. Modules can mutate the tensor
  they receive without corrupting workspace state. The cost is one
  per-cycle `8 · dim`-byte arena allocation plus an element-by-element
  bit-pattern copy.
- The winner's proposal is **copied** into `ws->content`; the proposal
  tensor itself is not retained beyond the broadcast.
- BP messages are **owned** by the factor graph; `fg-update-cpt!`
  nulls the pointers to force reallocation but does not free.
- The KB stores **pointers** to facts and shares them across queries;
  `kb-retract!` removes a pointer but does not free the fact (the
  arena does, in bulk, at reset).

---

## 10. Limitations and deferred work

The implementation flags several limitations either in source
comments or in the project memory; this section enumerates them.

### 10.1 Deferred features

- **Predicate indexing for `kb-query`**: the header note at
  `inc/eshkol/core/logic.h` line 75 (`v1.2 will add predicate indexing
  for O(1) lookup`) acknowledges that the current $O(N)$ linear scan
  is intentional for v1.1.
- **Rule heap subtype**: heap subtype 14 is reserved
  (`inc/eshkol/eshkol.h` line 353) for v1.2 backward chaining.
- **Asynchronous BP schedule**: the synchronous flooding schedule may
  converge slowly on highly loopy graphs. Residual BP, tree-reweighted
  BP, and priority schedules are unimplemented.
- **Continuous variables**: only discrete factor graphs are supported.
  Gaussian belief propagation, expectation propagation, and particle
  approximations are unimplemented.
- **Mixed-time inference**: there is no explicit support for unrolled
  temporal models; the user must replicate variables across time
  manually.
- **Per-module temperature**: the workspace softmax uses an implicit
  $T = 1.0$ that is not user-configurable. Clients must pre-scale
  their salience returns to achieve a desired sharpness.
- **More than 16 modules per workspace** in a single step: the
  `eshkol_ws_step_finalize` stack arrays cap at 16, and the codegen
  `MAX_WS_MODULES` constant matches this. Registering more than 16
  modules is allowed at construction (subject to `max_modules`), but
  only the first 16 indices will be processed per cycle.
- **Per-variable cap of 256 states for `fg-entropy`**: the stack
  `probs[256]` array in `eshkol_fg_entropy_tagged` imposes this limit
  (`lib/core/logic_builtins.cpp` line 108). Variables with more states
  will be silently truncated.
- **Module return-value robustness**: `eshkol_ws_step_finalize`
  recognises only cons cells whose `car` is `DOUBLE` or `INT64` for
  salience. Any other shape collapses to salience 0 (lines 240–244).
  There is no diagnostic emitted.
- **Occurs-check ceiling**: at depth 1000 the check returns `false`
  rather than failing safely; pathological terms could theoretically
  pass through. No occurrence has been observed in practice.

### 10.2 Concurrency

- The logic-variable registry is thread-safe (mutex + atomics) for
  `eshkol_make_logic_var`.
- The predicate intern pool is thread-safe via `g_pred_mutex`.
- `eshkol_logic_registry_reset` is explicitly **not** thread-safe and
  must be called from the main thread between test batches.
- All other state (substitutions, KBs, factor graphs, workspaces) is
  arena-resident and inherits the arena's thread-locality model. No
  internal locking is provided.

### 10.3 Serialisation

`kb-save`, `kb-load`, `tensor-save`, `tensor-load`, `model-save`,
`model-load` are exposed by `LogicWorkspaceCodegen` (lines 288–350)
and reach C runtime functions (`eshkol_kb_save_tagged`,
`eshkol_tensor_save_tagged`, etc., declared in `system_builtins.c`)
but are outside the scope of this document.

### 10.4 Test isolation caveat

`eshkol_logic_registry_reset` (`lib/core/logic.cpp` lines 62–74) is
the only mechanism for clearing the global variable-name and
predicate-intern pools between test runs. Without calling
`(reset-tests!)` between batches, var IDs and predicate canonical
pointers from a previous run will persist and can produce stale
unification hits or false-positive `eq?` matches across test
boundaries — a bug pattern documented in audit #194.

---

## 11. References

- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge
  University Press.
- Bengio, Y. (2017). *The consciousness prior*. arXiv:1709.08568.
- Friston, K. (2010). The free-energy principle: a unified brain
  theory? *Nature Reviews Neuroscience*, 11(2), 127–138.
- Kschischang, F. R., Frey, B. J., & Loeliger, H.-A. (2001). Factor
  graphs and the sum-product algorithm. *IEEE Transactions on
  Information Theory*, 47(2), 498–519.
- Martelli, A., & Montanari, U. (1982). An efficient unification
  algorithm. *ACM Transactions on Programming Languages and Systems*,
  4(2), 258–282.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*.
  Morgan Kaufmann.
- Robinson, J. A. (1965). A machine-oriented logic based on the
  resolution principle. *Journal of the ACM*, 12(1), 23–41.
- Russell, S., & Norvig, P. (2010). *Artificial Intelligence: A Modern
  Approach* (3rd ed.). Pearson. [Background reference for unification
  and probabilistic inference.]
- Yedidia, J. S., Freeman, W. T., & Weiss, Y. (2003). Understanding
  belief propagation and its generalizations. In G. Lakemeyer & B.
  Nebel (Eds.), *Exploring Artificial Intelligence in the New
  Millennium*. Morgan Kaufmann.

