# `core.reflection` — runtime value reflection

**Source**: [`lib/core/reflection.esk`](../../../lib/core/reflection.esk)
**Require**: auto-loaded via `(require stdlib)`; or individually `(require core.reflection)`

Runtime introspection helpers (task #170). `type-name` classifies a value with a single symbol; `describe` produces a human-readable string by dispatching over the standard type predicates.

`type-name` is a builtin (native codegen and bytecode VM, available without any `require`); it is listed here because it belongs to the same reflection surface. Related: `procedure-arity` is **not** defined here — it is a codegen builtin implemented in `lib/backend/llvm_codegen.cpp` (see `codegenProcedureArity`, dispatched at ~line 13447). It returns the fixed parameter count of a procedure and is used internally by `describe`. `record-fields` is documented in the source as **deferred** (field names are not embedded in runtime record values) and is not provided.

## Functions

### `(type-name value)`
Returns the value's type as a symbol from the runtime's one type-name vocabulary (`lib/core/value_type_names.h`), identical on JIT, AOT and the bytecode VM: `integer` (fixnum or bignum), `rational`, `real`, `complex`, `boolean`, `char`, `string`, `symbol`, `pair`, `null`, `vector`, `tensor`, `bytevector`, `hash-table`, `procedure` (any closure or builtin), `continuation`, `port`, `promise`, `parameter`, `exception`, `unspecified`, the domain types (`dual-number`, `logic-var`, `fact`, `knowledge-base`, `factor-graph`, `workspace`, ...), and `unknown` for a value outside the vocabulary. It is a builtin, so it is also a first-class procedure: `(map type-name (list 1 "a" 'b))` is `(integer string symbol)`. A numeric literal vector `#(1.0 2.0)` is a tensor in the native compiler and answers `tensor` there.

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
