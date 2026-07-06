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
| nested_ellip | r | 48 | - | - | 48 |
| nested_ellip | aot | 48 | - | - | 48 |
| nested_macro | r | 48 | - | - | 48 |
| nested_macro | aot | 48 | - | - | 48 |
| qq_macro | r | 48 | - | - | 48 |
| qq_macro | aot | 48 | - | - | 48 |
| recmac_and | r | 48 | - | - | 48 |
| recmac_and | aot | 48 | - | - | 48 |
| recmac_list | r | 48 | - | - | 48 |
| recmac_list | aot | 48 | - | - | 48 |

## Findings (WRONG at some depth = silent-wrong bug)
- none: every family is either all-PASS or degrades via a clean LIMIT.
