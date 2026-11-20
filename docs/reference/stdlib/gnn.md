# ml.gnn and ml.autograd — graph neural networks with exact gradients

`(require ml.gnn)` (which loads `ml.autograd`). Design: ADR-0052
([0052-program-synthesis.md](../../design/adr/0052-program-synthesis.md)).

Every layer is an operation recorded on the `core.ad.tape` reverse-mode tape
with a hand-derived backward rule, so a loss over a whole network is
differentiated in one reverse sweep. `tests/ml/gnn_test.esk` checks every rule
against central differences (relative error below 1e-6).

<!-- doc-example: skip fragment: the edge list, labels and masks are in tests/ml/gnn_test.esk -->
```scheme
(require ml.gnn)
(define g (make-graph 34 karate-edges))              ; undirected edge list ((i . j) ...)
(define X (gnn-one-hot-features 34))
(define m (gnn-make-model 'gcn g 34 8 2 101))        ; kind, graph, in, hidden, classes, seed
(gnn-fit! m X labels 60 0.05 5.0e-4)                 ; labels: class or #f per node
(gnn-accuracy (gnn-predict m X) truth test-mask)
```

## ml.autograd

Host matrices are `#(agm rows cols data)`: `ag-matrix`, `ag-zeros`,
`ag-random` (deterministic, seeded), `ag-from-rows`, `ag-to-rows`, `ag-ref`,
`ag-set!`, `ag-copy`. On a tape: `ag-param` (a differentiable input),
`ag-const`, `ag-value`.

| Operation | Backward rule |
|---|---|
| `ag-matmul t A B` | dA = dC Bᵀ, dB = Aᵀ dC |
| `ag-add`, `ag-sub`, `ag-hadamard`, `ag-scale` | elementwise |
| `ag-add-bias t A b` | db = column sums of dC |
| `ag-relu`, `ag-leaky-relu`, `ag-tanh`, `ag-sigmoid` | elementwise derivative |
| `ag-concat-cols t A B` | split dC |
| `ag-spmm t rows entries A` | constant sparse matrix: dA = Sᵀ dC |
| `ag-sum`, `ag-mean-rows`, `ag-dot-rows`, `ag-l2` | reductions |
| `ag-softmax-xent t logits labels` | (softmax − one-hot) / N over labelled rows |
| `ag-sigmoid-bce t Z Y` | (sigmoid(z) − y) / N |

`(ag-loss-and-gradients params build)` runs `build` on a fresh tape and
returns `(loss . gradients)`; `ag-adam`, `ag-adam-step!`, `ag-train-step!`.

## ml.gnn

| Procedure | Meaning |
|---|---|
| `make-graph`, `graph-size`, `graph-neighbors`, `graph-degree`, `graph-with-self-loops` | graphs |
| `(gnn-propagate g h message aggregate update)` | one synchronous round of message passing with any functions |
| `(gnn-aggregate t g H kind)` | differentiable neighbourhood aggregation, `kind` = `sum`, `mean` or `max` |
| `(gnn-gcn-entries g)`, `(gnn-gcn-layer t entries n H W b act)` | GCN: act(D^-1/2 (A+I) D^-1/2 H W + b) |
| `(gnn-sage-layer t g H Ws Wn b kind act)` | message passing: act(H Ws + AGG(H) Wn + b) |
| `(gnn-gat-layer t g H W a-dst a-src act)` | graph attention: αᵢⱼ = softmaxⱼ LeakyReLU(a_dstᵀ zᵢ + a_srcᵀ zⱼ) over N(i) ∪ {i} |
| `(gnn-gat-attention g H W a-dst a-src)` | the attention coefficients, per node `(j . α)` rows summing to 1 |
| `gnn-readout` | graph-level mean pooling |
| `gnn-make-model`, `gnn-forward`, `gnn-fit!`, `gnn-predict`, `gnn-accuracy` | two-layer node classifiers, `kind` = `gcn`, `sage` or `gat` |

Node classification on Zachary's karate club (34 members, 78 ties), four
labelled members, 60 Adam steps, held-out accuracy on the other 30:

| Model | Loss (first → last) | Held-out accuracy |
|---|---|---|
| GCN | 0.696 → 0.018 | 0.967 |
| GraphSAGE (mean) | 0.664 → 0.004 | 0.967 |
| GAT | 0.693 → 0.015 | 0.933 |
