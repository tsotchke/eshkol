---
kind: explanation
status: current
owner-area: language
since: v1.3.5
sources:
  - inc/eshkol/frontend/syntax_color.h
  - inc/eshkol/frontend/syntax_rules_core.h
  - inc/eshkol/frontend/syntax_datum.h
  - lib/frontend/syntax_rules.cpp
  - lib/frontend/syntax_datum.cpp
  - lib/frontend/macro_expander.cpp
  - lib/frontend/parser.cpp
  - lib/backend/vm_macro.c
  - lib/backend/vm_compiler.c
  - .icc/ledger/entries/SW-192.yaml
  - .icc/ledger/entries/SW-42.yaml
  - tests/vm_parity/corpus/93_macro_hygiene_matrix.esk
---
# ADR-0026: One `syntax-rules` engine and one renaming rule

**Status:** Accepted
**Ledger:** SW-192; SW-42 (referential transparency); the SW-30 residuals (`do` and internal `define` binders)
**Scope:** Macro expansion on the native compiler (JIT, AOT, REPL) and the
bytecode VM (hosted, ESKB, browser).

## Context

R7RS 4.3.2 requires `syntax-rules` macros to be hygienic in both directions:
an identifier a template binds can neither capture nor be captured by the
caller's code, and an identifier a template uses freely means what it meant
where the macro was defined, whatever the caller has bound with the same
spelling.

Each engine had its own `syntax-rules` implementation, and neither met the
requirement.

- The native expander matched patterns against the parser's lowered AST and
  substituted into a template that had already been parsed as code. Every AST
  payload shape needed its own substitution and renaming case, and each
  missing case miscompiled silently: a `do` or `let-values` binder in a
  template was left unrenamed or its operand unsubstituted, a macro-defining
  macro lost its pattern variables, `(thunk 7)` failed to match because a
  library procedure's formal named `thunk` did not shadow the keyword, and a
  continuation-passing macro was rejected as "circular".
- The VM matched reader syntax but renamed only the binders of `lambda` and
  the `let` family, had no dotted-tail patterns, mishandled repeated structured
  patterns, expanded operands before their scope was known (so a caller's
  local variable could not shadow a macro keyword), and did not see macros
  used before their definition.

A previous attempt refused special-form macro bindings. The owner ruled that
refusal is not completion (SW-192).

## Decision

1. **One engine.** `inc/eshkol/frontend/syntax_rules_core.h` is the only
   implementation of `syntax-rules` matching and instantiation. It works on a
   neutral tree of reader syntax and is header-only C, so the VM stays a single
   C translation unit for its browser and freestanding builds. The native
   expander (`lib/frontend/syntax_rules.cpp`) and the VM (`lib/backend/vm_macro.c`)
   are adapters that convert their reader syntax to that tree and back. It
   implements the full R7RS pattern language: literals, `_`, a custom ellipsis,
   an ellipsis followed by further patterns, dotted tails, vectors, nested
   ellipses, `(... ...)` escapes.

2. **Macros operate on reader syntax.** The native parser keeps the datum of
   every macro use and every transformer in side tables keyed by `NodeId` and by
   the definition record (`inc/eshkol/frontend/syntax_datum.h`), as ADR-0008
   prescribes for per-node data. Every symbol-headed call keeps a reference into
   its tokenizer's tape, so a call the parser could not know to be a macro use (a
   forward reference, a macro another module or another expansion defined) is
   re-read as syntax when the expander finds its keyword. An expansion is handed
   back to the parser, so a template means exactly what the same text means
   anywhere else in a program.

3. **One renaming rule** (`inc/eshkol/frontend/syntax_color.h`). Each expansion
   has a fresh color. Every identifier the template introduces, outside quoted
   data, is emitted with the color appended; pattern variables are the caller's
   syntax and keep their spelling. A colored identifier is therefore a new name:
   - a binder the template introduces binds the colored name, so it can neither
     capture caller code nor be captured by it, with no list of binding forms
     anywhere in the expander;
   - a colored identifier that no binder of its expansion binds is free in the
     template and is resolved by removing its last color and looking the name up
     in the macro's definition environment (a definition-site local, else the
     top-level binding), never among the caller's locals. Colors nest for
     macro-defining macros; one is removed per definition environment.
   - keywords are colored too. Each engine recognises keyword grammar by the
     uncolored spelling (`token_is_keyword` in the parser, `is_sym` and
     `eshkol_syntax_base_is` in the VM), so a template's `if`, `else` or `=>`
     keeps its meaning while a caller's `(let ((else 1)) ...)` cannot capture it.
     Only the pattern markers `...` and `_` are never colored.
   - a symbol that becomes data (quoted) never carries a color.
   - a top-level definition of a colored name (`define`, `define-values`,
     `define-record-type`) defines the uncolored name, as definitions produced
     by templates always did on both engines.

4. **Keyword scope.** A macro keyword is shadowed by a lexical binding of the
   same spelling created after it (a caller's `(let ((m ...)) (m ...))` calls the
   local); a top-level definition does not shadow syntax, on either engine. A
   template's own macro keywords resolve in the definition environment: the
   native expander emits an alias for the definition-site transformer, the VM
   stamps the template with its definition scope.

5. **Forward references.** Every top-level `define-syntax` of a unit is visible
   to the whole unit on both engines.

## Alternatives considered

- **Fix each engine's gaps in place.** Rejected: two implementations of one
  specification diverge by construction, and the native one could only be made
  complete by teaching substitution and renaming every AST payload shape, the
  source of the defects.
- **A binder-aware renaming pass.** Rejected: it needs a table of binding forms
  in the expander, which is a second copy of the language's scoping rules and
  misses every form it does not list (`do`, `let-values`, `guard`, internal
  `define`, named `let`, `match` patterns were all missing at some point).
  Coloring gets binders right without knowing which forms bind.
- **Refuse special-form macro bindings** (f646213d5, reverted in 01d887d91).
  Rejected by the owner: a documented refusal is not completion.
- **Compile the engine as C++ and link it into the VM.** Rejected: the browser
  and freestanding VM builds compile no C++.

## Consequences

- A use whose operands do not match any rule is a syntax error on both engines.
  In particular `()` is the empty list, not a disguised `(list)`: a pattern
  `(x)` does not match it (tests/parser/bare_null_arg_test.esk was corrected).
- Macro operands are no longer parsed as expressions before expansion, so an
  operand only needs to be syntax the template makes sense of.
- A library procedure's formal shadows a macro keyword of the same spelling
  inside its body.
- The VM expands each form when the compiler reaches it, in that form's scope.
- The legacy standalone `lib/backend/eshkol_compiler.c`, which no build target
  compiles, is not maintained against this change.

## Verification

- `tests/vm_parity/corpus/93_macro_hygiene_matrix.esk`: hand-computed checks
  run on native JIT, native AOT and the VM (ctest `macro_hygiene_matrix`) and in
  the VM parity corpus.
- `tests/frontend/syntax_rules_core_test.c`: the engine compiled as plain C,
  against hand-computed expansions (ctest `syntax_rules_core_test`).
- `tests/vm_parity/corpus/92_macro_definition_scope.esk`, the VM parity
  harness (vm-src and vm-eskb) and the existing macro suites.
