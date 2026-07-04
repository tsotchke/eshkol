# Metaprogramming + Module Depth-Parametric Report (P6e)

Each family generated at depth d=1..N with a closed-form ground truth.
MCD = max-correct-depth (largest d with 1..d all PASS). WRONG = ran but wrong value (bug). LIMIT = clean crash/compile-fail/timeout.

| family | mode | MCD | first-WRONG | first-LIMIT | max-d tested |
|---|---|---|---|---|---|
| deflib_chain | r | 12 | - | - | 12 |
| deflib_chain | aot | 12 | - | - | 12 |
| ellipsis_flat | r | 48 | - | - | 48 |
| ellipsis_flat | aot | 48 | - | - | 48 |
| match_nest | r | 48 | - | - | 48 |
| match_nest | aot | 48 | - | - | 48 |
| module_chain | r | 12 | - | - | 12 |
| module_chain | aot | 12 | - | - | 12 |
| nested_ellip | r | 0 | 1 | - | 48 |
| nested_ellip | aot | 0 | 1 | - | 48 |
| nested_macro | r | 48 | - | - | 48 |
| nested_macro | aot | 48 | - | - | 48 |
| qq_macro | r | 48 | - | - | 48 |
| qq_macro | aot | 48 | - | - | 48 |
| recmac_and | r | 48 | - | - | 48 |
| recmac_and | aot | 48 | - | - | 48 |
| recmac_list | r | 48 | - | - | 48 |
| recmac_list | aot | 48 | - | - | 48 |

## Findings (WRONG at some depth = silent-wrong bug)
- **nested_ellip** WRONG from depth 1 (got=0 want=1) [aot+r]
