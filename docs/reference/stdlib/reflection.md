# `core.reflection` — runtime value reflection

**Source**: [`lib/core/reflection.esk`](../../../lib/core/reflection.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.reflection)`

Runtime introspection helpers (task #170). `type-name` classifies a value with a single symbol; `describe` produces a human-readable string. Both dispatch over the standard type predicates.

Related: `procedure-arity` is **not** defined here — it is a codegen builtin implemented in `lib/backend/llvm_codegen.cpp` (see `codegenProcedureArity`, dispatched at ~line 13447). It returns the fixed parameter count of a procedure and is used internally by `describe`. `record-fields` is documented in the source as **deferred** (field names are not embedded in runtime record values) and is not provided.

## Functions

### `(type-name value)`
Returns one type-tag symbol: `null`, `boolean`, `integer`, `real`, `string`, `symbol`, `char`, `pair`, `vector`, `procedure`, or `unknown`. Note the dispatch order puts `null` and `boolean` before the numeric checks.

```scheme
;; reflection.esk
(require stdlib)
(display (type-name 42)) (newline)
(display (type-name 3.14)) (newline)
(display (type-name "hi")) (newline)
(display (type-name 'foo)) (newline)
(display (type-name #t)) (newline)
(display (type-name '())) (newline)
(display (type-name '(1 2))) (newline)
(display (type-name (vector 1 2))) (newline)
(display (type-name car)) (newline)
```
```
integer
real
string
symbol
boolean
null
pair
vector
procedure
```

### `(describe value)`
Returns a descriptive string. Format varies by type: atoms show their value; strings show length and quoted text; pairs and vectors show their size; procedures show their arity (via `procedure-arity`).

```scheme
(require stdlib)
(display (describe 42)) (newline)
(display (describe 3.14)) (newline)
(display (describe "hi")) (newline)
(display (describe 'foo)) (newline)
(display (describe #t)) (newline)
(display (describe '())) (newline)
(display (describe (list 1 2 3))) (newline)
(display (describe (vector 'a 'b))) (newline)
(display (describe (lambda (x y) x))) (newline)
(display (describe car)) (newline)
```
```
integer: 42
real: 3.14
string[2]: "hi"
symbol: foo
boolean: #t
null
pair (length 3)
vector[2]
procedure: arity=2
procedure: arity=1
```

Edge cases: the docstring in the source shows the pair/vector forms with their contents appended (e.g. `pair (length 3): (1 2 3)`), but the implementation emits only the size prefix (`pair (length 3)`, `vector[2]`) — the contents are not included. A value matching no predicate returns the symbol/string `unknown`.
