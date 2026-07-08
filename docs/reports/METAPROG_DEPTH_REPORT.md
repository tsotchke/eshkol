# Metaprogramming + Module Depth-Parametric Report (P6e)

Each family generated at depth d=1..N with a closed-form ground truth.
MCD = max-correct-depth (largest d with 1..d all PASS). WRONG = ran but wrong value (bug). LIMIT = clean crash/compile-fail/timeout.

| family | mode | MCD | first-WRONG | first-LIMIT | max-d tested |
|---|---|---|---|---|---|
| deflib_chain | r | 12 | - | - | 12 |
| deflib_chain | aot | 12 | - | - | 12 |
| ellipsis_flat | r | 32 | - | - | 32 |
| ellipsis_flat | aot | 32 | - | - | 32 |
| match_nest | r | 32 | - | - | 32 |
| match_nest | aot | 32 | - | - | 32 |
| module_chain | r | 12 | - | - | 12 |
| module_chain | aot | 12 | - | - | 12 |
| nested_ellip | r | 32 | - | - | 32 |
| nested_ellip | aot | 32 | - | - | 32 |
| nested_macro | r | 32 | - | - | 32 |
| nested_macro | aot | 32 | - | - | 32 |
| qq_macro | r | 32 | - | - | 32 |
| qq_macro | aot | 32 | - | - | 32 |
| recmac_and | r | 32 | - | - | 32 |
| recmac_and | aot | 32 | - | - | 32 |
| recmac_list | r | 32 | - | - | 32 |
| recmac_list | aot | 32 | - | - | 32 |

## Findings (WRONG at some depth = silent-wrong bug)
- none: every family is either all-PASS or degrades via a clean LIMIT.
