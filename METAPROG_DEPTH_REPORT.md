# Metaprogramming + Module Depth-Parametric Report (P6e)

Each family generated at depth d=1..N with a closed-form ground truth.
MCD = max-correct-depth (largest d with 1..d all PASS). WRONG = ran but wrong value (bug). LIMIT = clean crash/compile-fail/timeout.

| family | mode | MCD | first-WRONG | first-LIMIT | max-d tested |
|---|---|---|---|---|---|
| deflib_chain | r | 16 | - | - | 16 |
| deflib_chain | aot | 16 | - | - | 16 |
| ellipsis_flat | r | 48 | - | - | 48 |
| ellipsis_flat | aot | 48 | - | - | 48 |
| match_nest | r | 48 | - | - | 48 |
| match_nest | aot | 48 | - | - | 48 |
| module_chain | r | 16 | - | - | 16 |
| module_chain | aot | 16 | - | - | 16 |
| nested_ellip | r | 0 | 1 | - | 48 |
| nested_ellip | aot | 0 | 1 | - | 48 |
| nested_macro | r | 48 | - | - | 48 |
| nested_macro | aot | 48 | - | - | 48 |
| qq_macro | r | 1 | 2 | - | 48 |
| qq_macro | aot | 1 | 2 | - | 48 |
| recmac_and | r | 48 | - | - | 48 |
| recmac_and | aot | 48 | - | - | 48 |
| recmac_list | r | 48 | - | - | 48 |
| recmac_list | aot | 48 | - | - | 48 |

## Findings (WRONG at some depth = silent-wrong bug)
- **nested_ellip** WRONG from depth 1 (got=0 want=1) [aot+r] — nested ellipsis `x ... ...` unsupported; macro-expansion error is non-fatal so the program exits 0 with a silently-wrong value (ESH-0120).
- **qq_macro** WRONG from depth 2 (got=1 want=2) [aot+r] — pattern variable inside a quasiquote-unquote in a macro template is not substituted; d1 passes only coincidentally (ESH-0119).

## Related parser finding (not a depth family — worked around in the generator)
- **ESH-0118** a `syntax-rules` rule whose whole template is quote/quasiquote *shorthand* (`'()`, `` `x ``) is rejected ("expected closing paren after macro rule template"); the `(quote ...)` long form and nested shorthand (`(cons x '())`) both work.

## Notes / capability boundaries
- Working families (recmac_and, recmac_list, nested_macro, ellipsis_flat, match_nest) are correct to the max depth tested here (48) on both `-r` and AOT; the recursive-macro expander hits a separate documented limit (>1000 macro-expansion depth, ESH-0103) far above this range.
- module_chain and deflib_chain transitively resolve and (for AOT) link the whole dependency chain correctly to depth 16 on both axes.
