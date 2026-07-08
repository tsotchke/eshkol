# Tensor / Collection / String Depth-Parametric Report (P6f)

Each family generated at depth/size d=1..N with a closed-form or numpy
ground truth. Every case runs on three axes — JIT (-r), AOT-O0, AOT-O2.

- **MCD** = max-correct depth/size per axis (largest d with 1..d all PASS).
- **WRONG** = every axis agrees on a value that is not the oracle (silent bug).
- **AXIS-DIVERGENCE** = the axes disagree (codegen/opt/JIT mismatch).
- **LIMIT** = every axis fails cleanly (documented capability boundary).

| family | MCD(-r) | MCD(O0) | MCD(O2) | max-tested | first-WRONG | first-DIVERGE | first-LIMIT |
|---|---|---|---|---|---|---|---|
| conv_scale | 64 | 64 | 64 | 64 | - | - | - |
| emoji_len | 10000 | 10000 | 10000 | 10000 | - | - | - |
| hashtable_size | 100000 | 100000 | 100000 | 100000 | - | - | - |
| list_length | 100000 | 100000 | 100000 | 1000000 | - | - | 1000000 |
| matmul_np | 16 | 16 | 16 | 16 | - | - | - |
| matmul_scale | 256 | 256 | 256 | 256 | - | - | - |
| nested_list | 32 | 32 | 32 | 32 | - | - | - |
| nested_vector | 32 | 32 | 32 | 32 | - | - | - |
| nul_len | 256 | 256 | 256 | 256 | - | - | - |
| opchain_add | 16 | 16 | 16 | 16 | - | - | - |
| opchain_mul | 16 | 16 | 16 | 16 | - | - | - |
| reshape_rank | 8 | 8 | 8 | 8 | - | - | - |
| string_length | 1000000 | 1000000 | 1000000 | 1000000 | - | - | - |
| tensor_rank | 3 | 3 | 3 | 8 | 4 | - | - |
| unicode_len | 10000 | 10000 | 10000 | 10000 | - | - | - |

## Silent-WRONG findings (all axes agree on a wrong value = bug)
- **tensor_rank** WRONG from depth/size 4: all-axis got=3 want=4

## Axis-divergence findings (-r vs AOT-O0 vs AOT-O2 disagree)
- none.

## Root cause & minimal repros

### ESH-0205 — make-tensor silently truncates rank > 3 (silent WRONG)
`make-tensor` with a dims list of length >= 4 silently drops every dimension
past the third, producing a rank-3 tensor. All three axes (-r, AOT-O0, AOT-O2)
agree on the wrong value, so it is a codegen-level defect, not an opt/JIT issue.

Minimal repro:
```scheme
(require stdlib)
(define T (make-tensor (list 2 3 4 5) 7.0))
(display (tensor-shape T))(newline)   ; prints (2 3 4) — expected (2 3 4 5)
```
Root cause: `TensorCodegen::makeTensorImpl` (lib/backend/tensor_creation_codegen.cpp,
list-path ~L457-578) only unrolls dimension extraction for the 1D/2D/3D cases —
the "3D case" reads dim3 and stops walking the cons `cdr`, so dims 4..N are never
read (the source comment even claims "up to 4D"). `reshape` uses an independent
path and handles rank 4..8 correctly (reshape_rank MCD = 8), which is why the two
constructors disagree. Max-correct rank for make-tensor = 3.

### ESH-0206 — list traversal crashes (SIGBUS) at ~10^6 elements
`(length lst)` on a ~10^6-element list terminates with SIGBUS (exit 138) on all
three axes instead of returning a value or a graceful error. Clean and correct
through 10^5 (list_length MCD = 100000). This is a non-tail-recursive traversal
overflowing the stack — a hard crash, not a documented limit.

Minimal repro:
```scheme
(require stdlib)
(define lst (let loop ((i 0) (acc '())) (if (< i 1000000) (loop (+ i 1) (cons i acc)) acc)))
(display (length lst))(newline)   ; SIGBUS instead of 1000000
```

## Axis-fidelity summary
Across all 165 cases the JIT (-r), AOT-O0 and AOT-O2 axes agreed on every value
(zero axis-divergence): every per-axis MCD is identical. The only defects are the
two above; every other family is correct to its full tested depth/size.
