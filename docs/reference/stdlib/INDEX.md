# Eshkol Standard Library — v1.3.5 API Reference Index

Complete module-by-function map of the Eshkol standard library. Every symbol
below links to a per-module reference page in this directory; every documented
function was verified by compiling and running it with `eshkol-run … -r`.

See [module-system.md](module-system.md) for how `require` resolution,
`(require stdlib)`, and `stdlib.o` precompilation work. The v1.2-scale surface
notes remain in [../../STDLIB_V1_2_API.md](../../STDLIB_V1_2_API.md).

**Modules: 59** (plus `stdlib` itself and the module system page) — **provided symbols: 679** (plus stdlib-level helpers in [stdlib_extras.md](stdlib_extras.md)).

*Auto* = loaded automatically by `(require stdlib)`; otherwise the module must be required individually.

| Module | Auto | Reference | Provided symbols |
|---|---|---|---|
| [`core.ad.tape`](../../../lib/core/ad/tape.esk) | no | [ad_tape.md](ad_tape.md) | `ad-tape-new` `ad-const` `ad-var` `ad-add` `ad-sub` `ad-mul` `ad-div` `ad-sin` `ad-cos` `ad-exp` `ad-log` `ad-sqrt` `ad-neg` `ad-abs` `ad-relu` `ad-sigmoid` `ad-tanh` `ad-backward` `ad-gradient` `ad-node-value` `with-tape` `current-tape` `make-tape` `tape-input` `tape-const` `node-value` `node-grad` `record-op!` `record-fd-op!` `tape-add` `tape-sub` `tape-mul` `tape-div` `tape-sin` `tape-cos` `tape-exp` `tape-log` `tape-neg` `tape-square` `tape-sqrt` `tape-tanh` `tape-sigmoid` `tape-relu` `tape-pow` `tape-gradient` `tape-snapshot` `tape-restore` |
| [`core.alist`](../../../lib/core/alist.esk) | yes | [alist.md](alist.md) | `alist-ref` `alist-ref-or` `alist-set` |
| [`core.argparse`](../../../lib/core/argparse.esk) | no | [argparse.md](argparse.md) | `parse-args` `arg-get` `arg-positional` `arg-has?` `argparse-help` |
| [`core.cache`](../../../lib/core/cache.esk) | no | [cache.md](cache.md) | `make-lru-cache` `lru-get` `lru-get/default` `lru-set!` `lru-has?` `lru-delete!` `lru-size` `lru-capacity` `lru-clear!` `memoize` `memoize/cap` `memoize1` `memoize1/cap` `memoize2` `memoize2/cap` |
| [`core.capabilities`](../../../lib/core/capabilities.esk) | yes | [capabilities.md](capabilities.md) | `capability-install-policy!` `capability-clear-policy!` `capability-policy` `capability-policy-active?` `capability-allowed?` `capability-require!` |
| [`core.channels`](../../../lib/core/channels.esk) | no | [channels.md](channels.md) | `make-channel` `channel?` `channel-capacity` `channel-len` `channel-closed?` `channel-send!` `channel-recv!` `channel-try-send!` `channel-try-recv!` `channel-close!` `channel-closed-sentinel` |
| [`core.collections`](../../../lib/core/collections.esk) | no | [collections.md](collections.md) | `make-pq` `pq-push!` `pq-pop!` `pq-peek` `pq-size` `pq-empty?` `make-set` `set-add!` `set-remove!` `set-contains?` `set-size` `set->list` `set-clear!` `set-union` `set-intersect` `set-difference` `make-deque` `deque-push-front!` `deque-push-back!` `deque-pop-front!` `deque-pop-back!` `deque-peek-front` `deque-peek-back` `deque-size` `deque-empty?` `deque->list` |
| [`core.control.trampoline`](../../../lib/core/control/trampoline.esk) | yes | [control_trampoline.md](control_trampoline.md) | `trampoline` `bounce` `done` |
| [`core.data.base64`](../../../lib/core/data/base64.esk) | yes | [data_base64.md](data_base64.md) | `base64-encode` `base64-decode` `base64-encode-string` `base64-decode-string` `string->bytes` `bytes->string` `base64-char-at` `base64-value` `string->bytes-helper` `base64-encode-helper` `base64-decode-helper` `base64-remove-padding` |
| [`core.data.csv`](../../../lib/core/data/csv.esk) | yes | [data_csv.md](data_csv.md) | `csv-parse` `csv-parse-file` `csv-stringify` `csv-write-file` `csv-parse-line` `csv-parse-lines` `csv-split-fields` `csv-stringify-row` |
| [`core.data.dataframe`](../../../lib/core/data/dataframe.esk) | yes | [data_dataframe.md](data_dataframe.md) | `csv-parse-typed` `csv-header` `csv-rows` `csv-column` `csv-column-idx` `csv-select` `csv-filter` `csv-map-column` `csv-row-count` `csv-col-count` `csv-describe` |
| [`core.dbsp`](../../../lib/core/dbsp.esk) | no | [dbsp.md](dbsp.md) | `zset-empty` `zset?` `zset-singleton` `zset-from-weighted` `zset-weight` `zset-add` `zset-negate` `zset-sub` `zset-scale` `zset-consolidate` `zset-empty?` `zset-positive?` `zset=?` `zset-entries` `zset-rows` `zset-size` `zset-map` `zset-filter` `zset-project` `zset-union` `zset-join` `zset-distinct` `zset-count` `zset-sum` `stream-zero` `stream-delay` `dbsp-delay` `stream-D` `stream-I` `stream-add` `stream-negate` `stream-lift` `dbsp-compose` `dbsp-incrementalize` `incremental-map` `incremental-filter` `incremental-project` `incremental-union` `incremental-join` `incremental-distinct` |
| [`core.distributed`](../../../lib/core/distributed.esk) | no | [distributed.md](distributed.md) | `lamport-zero` `lamport-tick` `lamport-merge` `lamport-recv` `lamport-before?` `vector-clock-empty` `vector-clock-ref` `vector-clock-set` `vector-clock-tick` `vector-clock-merge` `vector-clock-compare` `vector-clock-before?` `vector-clock-after?` `vector-clock-concurrent?` `vector-clock-equal?` `vector-clock-nodes` `make-g-counter` `g-counter-counts` `g-counter-inc` `g-counter-value` `g-counter-merge` `make-pn-counter` `pn-counter-positive` `pn-counter-negative` `pn-counter-inc` `pn-counter-dec` `pn-counter-value` `pn-counter-merge` `make-or-set` `or-set-adds` `or-set-removes` `or-set-clock` `or-set-add` `or-set-remove` `or-set-member?` `or-set-elements` `or-set-merge` `make-lww-register` `lww-register-value` `lww-register-timestamp` `lww-register-writer` `lww-register-deleted?` `lww-register-present?` `lww-register-set` `lww-register-delete` `lww-register-merge` `make-lww-map` `lww-map-entries` `lww-map-set` `lww-map-remove` `lww-map-ref` `lww-map-contains?` `lww-map-visible-entries` `lww-map-merge` `make-rga` `rga-root-id` `rga-entries` `rga-clock` `rga-entry-id` `rga-entry-prev` `rga-entry-value` `rga-entry-deleted?` `rga-insert-after` `rga-append` `rga-delete` `rga-values` `rga-last-id` `rga-merge` |
| [`core.dnc`](../../../lib/core/dnc.esk) | no | [dnc.md](dnc.md) | `make-dnc-memory` `dnc-content-address` `dnc-loc-address` `dnc-read` `dnc-write!` `dnc-alloc-weights` `dnc-read-grad` `dnc-memory?` |
| [`core.files`](../../../lib/core/files.esk) | yes | [files.md](files.md) | `path-directory` `with-atomic-output-file` `atomic-write-file` |
| [`core.functional.compose`](../../../lib/core/functional/compose.esk) | yes | [functional_compose.md](functional_compose.md) | `compose` `compose3` `identity` `constantly` |
| [`core.functional.curry`](../../../lib/core/functional/curry.esk) | yes | [functional_curry.md](functional_curry.md) | `curry2` `curry3` `uncurry2` `partial1` `partial2` `partial3` `partial` |
| [`core.functional.flip`](../../../lib/core/functional/flip.esk) | yes | [functional_flip.md](functional_flip.md) | `flip` |
| [`core.http_server`](../../../lib/core/http_server.esk) | no | [http_server.md](http_server.md) | `http-request-line` `http-request-method` `http-request-target` `http-request-path` `http-request-query` `http-request-version` `http-request-header` `http-request-content-length` `http-request-body` `http-request-body-too-large?` `http-max-request-body-size` `http-response` `http-response?` `http-response-status` `http-response-content-type` `http-response-body` `http-route` `http-route-request` `http-standard-response` `http-server-respond-response` `http-server-respond-standard` |
| [`core.io`](../../../lib/core/io.esk) | yes | [io.md](io.md) | `print` `println` |
| [`core.json`](../../../lib/core/json.esk) | yes | [json.md](json.md) | `json-parse` `json-try-parse` `json-parse-result?` `json-result-ok?` `json-result-value` `json-result-error` `json-stringify` `json-read` `json-write` `json-get` `json-get-in` `json-array-ref` `hash-table->alist` `alist->hash-table` `alist->json` `json-write-file` `alist-write-json` `json-read-file` |
| [`core.json_schema`](../../../lib/core/json_schema.esk) | yes | [json_schema.md](json_schema.md) | `json-schema-valid?` `json-schema-validate` |
| [`core.list.compound`](../../../lib/core/list/compound.esk) | yes | [list_compound.md](list_compound.md) | `caar` `cadr` `cdar` `cddr` `caaar` `caadr` `cadar` `caddr` `cdaar` `cdadr` `cddar` `cdddr` `caaaar` `caaadr` `caadar` `caaddr` `cadaar` `cadadr` `caddar` `cadddr` `cdaaar` `cdaadr` `cdadar` `cdaddr` `cddaar` `cddadr` `cdddar` `cddddr` `first` `second` `third` `fourth` `fifth` `sixth` `seventh` `eighth` `ninth` `tenth` |
| [`core.list.convert`](../../../lib/core/list/convert.esk) | yes | [list_convert.md](list_convert.md) | `list->vector` `vector->list` |
| [`core.list.generate`](../../../lib/core/list/generate.esk) | yes | [list_generate.md](list_generate.md) | `iota` `iota-from` `iota-step` `repeat` `make-list` `range` `zip` |
| [`core.list.higher_order`](../../../lib/core/list/higher_order.esk) | yes | [list_higher_order.md](list_higher_order.md) | `map1` `map2` `map3` `fold` `fold-left` `foldl` `fold-right` `foldr` `for-each` `any` `every` `filter-map` |
| [`core.list.query`](../../../lib/core/list/query.esk) | yes | [list_query.md](list_query.md) | `count-if` `find` `length` |
| [`core.list.search`](../../../lib/core/list/search.esk) | yes | [list_search.md](list_search.md) | `member` `member?` `memq` `memv` `assoc` `assq` `assv` `list-ref` `list-tail` `list-set!` |
| [`core.list.sort`](../../../lib/core/list/sort.esk) | yes | [list_sort.md](list_sort.md) | `sort` |
| [`core.list.transform`](../../../lib/core/list/transform.esk) | yes | [list_transform.md](list_transform.md) | `take` `drop` `take-right` `drop-right` `append` `reverse` `filter` `unzip` `partition` `list-copy` |
| [`core.logging`](../../../lib/core/logging.esk) | no | [logging.md](logging.md) | `log-debug` `log-info` `log-warn` `log-error` `log-set-level!` `log-set-output!` `log-with-trace!` `log-level` `log-output-port` |
| [`core.logic.boolean`](../../../lib/core/logic/boolean.esk) | yes | [logic_boolean.md](logic_boolean.md) | `negate` `all?` `none?` |
| [`core.logic.predicates`](../../../lib/core/logic/predicates.esk) | yes | [logic_predicates.md](logic_predicates.md) | `is-zero?` `is-positive?` `is-negative?` `is-even?` `is-odd?` |
| [`core.logic.types`](../../../lib/core/logic/types.esk) | yes | [logic_types.md](logic_types.md) | `is-null?` `is-pair?` |
| [`core.manifold`](../../../lib/core/manifold.esk) | yes | [manifold.md](manifold.md) | `make-euclidean-manifold` `make-hyperbolic-manifold` `make-spherical-manifold` `manifold-exp-map` `manifold-log-map` `manifold-distance` `manifold-parallel-transport` `manifold-curvature` `manifold-dimension` `manifold-type` `metric-component` `manifold-metric` `manifold-metric-inverse` `christoffel-symbol` `manifold-christoffel` `manifold-sectional-curvature` `manifold-scalar-curvature` `ricci-component` `manifold-ricci` `riemann-component` |
| [`core.memory`](../../../lib/core/memory.esk) | no | [memory.md](memory.md) | `make-memory-log` `memory-append!` `memory-events` `memory-merge` `memory-verify-chain` `memory-verify-events` `memory-fold-lww` `memory-event?` `memory-event-id` `memory-event-prev` `memory-event-vclock` `memory-event-node` `memory-event-type` `memory-event-payload` `event-content-hash` |
| [`core.memory_store`](../../../lib/core/memory_store.esk) | no | [memory_store.md](memory_store.md) | `make-memory-store` `memory-store?` `memory-store-log` `memory-store-path` `memory-store-open` `memory-store-open-fast` `memory-store-append!` `memory-store-verify` `memory-store-audit` `memory-store-count` `memory-store-head` `memory-store-sanitize` |
| [`core.merkle`](../../../lib/core/merkle.esk) | no | [merkle.md](merkle.md) | `fnv1a-64` `hash->hex` `merkle-leaf` `merkle-leaf?` `merkle-inode?` `merkle-root` `merkle-data` `merkle-tree` `merkle-tree-with-hash` `merkle-leaves` `merkle-proof` `merkle-verify` `make-cas` `make-cas-with-hash` `cas?` `cas-put!` `cas-get` `cas-has?` `cas-size` `cas-keys` |
| [`core.metrics`](../../../lib/core/metrics.esk) | no | [metrics.md](metrics.md) | `make-counter` `counter-inc!` `counter-add!` `make-gauge` `gauge-set!` `gauge-inc!` `gauge-dec!` `make-histogram` `histogram-observe!` `histogram-buckets` `metrics-register!` `metrics-render` `metrics-reset!` `metric-name` `metric-help` `metric-kind` |
| [`core.ml.gradient_estimators`](../../../lib/core/ml/gradient_estimators.esk) | no | [ml_gradient_estimators.md](ml_gradient_estimators.md) | `gumbel-softmax` `gumbel-softmax-det` `straight-through` `straight-through-round` `straight-through-onehot` `categorical-pick` `argmax-onehot` `softmax` `sample-gumbel-noise` |
| [`core.ml.neurosymbolic`](../../../lib/core/ml/neurosymbolic.esk) | no | [ml_neurosymbolic.md](ml_neurosymbolic.md) | `make-embedding-table` `emb-dim` `embed!` `vdot` `soft-unify` `soft-unify-loss` `soft-unify-train!` `kb-attention` `kb-retrieve` |
| [`core.msgpack`](../../../lib/core/msgpack.esk) | no | [msgpack.md](msgpack.md) | `msgpack-null` `msgpack-null?` `msgpack-map` `msgpack-map?` `msgpack-map-entries` `msgpack-encode` `msgpack-decode` `msgpack-decode-prefix` `msgpack-bytes->bytevector` `msgpack-bytevector->bytes` |
| [`core.numeric_extras`](../../../lib/core/numeric_extras.esk) | yes | [numeric_extras.md](numeric_extras.md) | `exact-integer-sqrt` |
| [`core.operators.arithmetic`](../../../lib/core/operators/arithmetic.esk) | yes | [operators_arithmetic.md](operators_arithmetic.md) | `add` `sub` `mul` `div` |
| [`core.operators.compare`](../../../lib/core/operators/compare.esk) | yes | [operators_compare.md](operators_compare.md) | `lt` `gt` `le` `ge` `eq` |
| [`core.plot`](../../../lib/core/plot.esk) | yes | [plot.md](plot.md) | `sparkline` `bar-chart` `histogram` |
| [`core.reflection`](../../../lib/core/reflection.esk) | yes | [reflection.md](reflection.md) | `describe` `type-name` |
| [`core.sdnc`](../../../lib/core/sdnc.esk) | no | [sdnc.md](sdnc.md) | `sdnc-program` `sdnc-run` `sdnc-weight-grad` `sdnc-params` `sdnc-set-params!` `sdnc-improve!` `sdnc?` |
| [`core.sexp`](../../../lib/core/sexp.esk) | yes | [sexp.md](sexp.md) | `sexp->string` `sexp->canonical-string` |
| [`core.streams`](../../../lib/core/streams.esk) | yes | [streams.md](streams.md) | `stream-null` `stream-null?` `stream-pair?` `stream?` `stream-cons` `stream-car` `stream-cdr` `stream-take` `stream-drop` `stream-ref` `stream-map` `stream-filter` `stream-for-each` `stream-zip` `stream-append` `stream-iterate` `stream-from` `stream-take-while` `stream-drop-while` `stream-length` `stream->list` `list->stream` |
| [`core.strings`](../../../lib/core/strings.esk) | yes | [strings.md](strings.md) | `string-join` `string-trim` `string-trim-left` `string-trim-right` `string-replace` `string-reverse` `string-copy` `string-repeat` `string-starts-with?` `string-ends-with?` `string-starts-with` `string-ends-with` `string-index` `string-last-index` `string-contains` `string-contains?` `string-count` `string-find` `string-upcase` `string-downcase` `string-split-ordered` |
| [`core.testing`](../../../lib/core/testing.esk) | no | [testing.md](testing.md) | `register-test` `check-equal?` `check-true` `check-false` `check-approx` `assert-close` `certify-kernel` `check-raises` `run-tests` `reset-tests!` `*tests*` `*test-pass-count*` `*test-fail-count*` `*current-test-fails*` `*current-test-name*` |
| [`core.threads`](../../../lib/core/threads.esk) | no | [threads.md](threads.md) | `make-mutex` `mutex-lock!` `mutex-trylock!` `mutex-unlock!` `mutex-destroy!` `with-mutex` `make-condvar` `condvar-wait!` `condvar-signal!` `condvar-broadcast!` `condvar-destroy!` `make-thread` `thread-join` `thread?` `thread-result-ready?` |
| [`core.url`](../../../lib/core/url.esk) | yes | [url.md](url.md) | `url-encode` `url-decode` `base64url-encode` `base64url-decode` |
| [`math`](../../../lib/math.esk) | no | [math.md](math.md) | `mat-ref` `tensor-copy` `det` `inv` `solve` `cross` `dot` `normalize` `mat-vec-mul` `power-iteration` `integrate` `newton` `variance` `std` `covariance` |
| [`ml.activations`](../../../lib/ml/activations.esk) | no | [ml_activations.md](ml_activations.md) | `relu-scalar` `sigmoid-scalar` `tanh-scalar` `softplus-scalar` `silu` `swish` `mish` `normalize-minmax` `normalize-zscore` |
| [`ml.optimization`](../../../lib/ml/optimization.esk) | yes | [ml_optimization.md](ml_optimization.md) | `gradient-descent` `adam` `l-bfgs` `conjugate-gradient` `line-search` `tensor-dot` `tensor-norm` |
| [`signal.fft`](../../../lib/signal/fft.esk) | yes | [signal_fft.md](signal_fft.md) | `fft` `ifft` |
| [`signal.filters`](../../../lib/signal/filters.esk) | yes | [signal_filters.md](signal_filters.md) | `hamming-window` `hann-window` `blackman-window` `kaiser-window` `apply-window` `convolve` `fast-convolve` `fir-filter` `iir-filter` `butterworth-lowpass` `butterworth-highpass` `butterworth-bandpass` `frequency-response` |

## Builtin families

Some public surfaces are compiler or VM builtins rather than `.esk` modules, so
they have no `require` line or `provide` block. They are documented here because
from a user's side they are ordinary callable names.

| Family | Reference | Names | Engines |
|---|---|---|---|
| Geometric manifolds, Lie groups, forms, geodesic attention | [geometry.md](geometry.md) | 62 | bytecode VM only |
| Explicit reverse-mode AD tape and instrumentation counters | [../ad/tape.md](../ad/tape.md) | 33 | native AOT, native JIT, VM |

## DD-11 indexed export coverage

The following shipped exports were missing from the index and now have stable
reference anchors on [shipped_exports.md](shipped_exports.md).

| Source | Export | Reference |
|---|---|---|
| `agent.http` | `http-stream-error` | [http-stream-error](shipped_exports.md#http-stream-error) |
| `agent.http` | `sse-event-id` | [sse-event-id](shipped_exports.md#sse-event-id) |
| `agent.http` | `sse-event-retry-ms` | [sse-event-retry-ms](shipped_exports.md#sse-event-retry-ms) |
| `agent.quantum` | `make-pauli-hamiltonian` | [make-pauli-hamiltonian](shipped_exports.md#make-pauli-hamiltonian) |
| `core.ad.interval` | `interval-add` | [interval-add](shipped_exports.md#interval-add) |
| `core.ad.interval` | `interval-cos` | [interval-cos](shipped_exports.md#interval-cos) |
| `core.ad.interval` | `interval-div` | [interval-div](shipped_exports.md#interval-div) |
| `core.ad.interval` | `interval-exp` | [interval-exp](shipped_exports.md#interval-exp) |
| `core.ad.interval` | `interval-log` | [interval-log](shipped_exports.md#interval-log) |
| `core.ad.interval` | `interval-mid` | [interval-mid](shipped_exports.md#interval-mid) |
| `core.ad.interval` | `interval-mul` | [interval-mul](shipped_exports.md#interval-mul) |
| `core.ad.interval` | `interval-neg` | [interval-neg](shipped_exports.md#interval-neg) |
| `core.ad.interval` | `interval-pow` | [interval-pow](shipped_exports.md#interval-pow) |
| `core.ad.interval` | `interval-sin` | [interval-sin](shipped_exports.md#interval-sin) |
| `core.ad.interval` | `interval-sub` | [interval-sub](shipped_exports.md#interval-sub) |
| `core.ad.interval` | `interval-union` | [interval-union](shipped_exports.md#interval-union) |
| `core.ad.interval` | `interval-widen` | [interval-widen](shipped_exports.md#interval-widen) |
| `core.ad.interval` | `interval?` | [interval?](shipped_exports.md#interval) |
| `core.ad.interval` | `iv-abs-pad` | [iv-abs-pad](shipped_exports.md#iv-abs-pad) |
| `core.ad.interval` | `iv-eps` | [iv-eps](shipped_exports.md#iv-eps) |
| `core.ad.interval` | `iv-rel-pad` | [iv-rel-pad](shipped_exports.md#iv-rel-pad) |
| `core.ad.taylor_models` | `taylor-model?` | [taylor-model?](shipped_exports.md#taylor-model) |
| `core.ad.taylor_models` | `tm-nsamp` | [tm-nsamp](shipped_exports.md#tm-nsamp) |
| `core.ad.taylor_models` | `tm-safety` | [tm-safety](shipped_exports.md#tm-safety) |
| `core.ad.tensor_tower` | `tt-add` | [tt-add](shipped_exports.md#tt-add) |
| `core.ad.tensor_tower` | `tt-const` | [tt-const](shipped_exports.md#tt-const) |
| `core.ad.tensor_tower` | `tt-div` | [tt-div](shipped_exports.md#tt-div) |
| `core.ad.tensor_tower` | `tt-exp` | [tt-exp](shipped_exports.md#tt-exp) |
| `core.ad.tensor_tower` | `tt-hadamard-cauchy` | [tt-hadamard-cauchy](shipped_exports.md#tt-hadamard-cauchy) |
| `core.ad.tensor_tower` | `tt-neg` | [tt-neg](shipped_exports.md#tt-neg) |
| `core.ad.tensor_tower` | `tt-order` | [tt-order](shipped_exports.md#tt-order) |
| `core.ad.tensor_tower` | `tt-scale-const` | [tt-scale-const](shipped_exports.md#tt-scale-const) |
| `core.ad.tensor_tower` | `tt-sub` | [tt-sub](shipped_exports.md#tt-sub) |
| `core.ad.tensor_tower` | `tt-value` | [tt-value](shipped_exports.md#tt-value) |
| `tensor.utils` | `tensor.utils` | [tensor.utils](shipped_exports.md#tensor.utils) |
| `tensorcore` | `tc-adapter-available?` | [tc-adapter-available?](shipped_exports.md#tc-adapter-available) |
| `tensorcore` | `tc-adapter-status` | [tc-adapter-status](shipped_exports.md#tc-adapter-status) |
| `tensorcore` | `tc-attention-forward` | [tc-attention-forward](shipped_exports.md#tc-attention-forward) |
| `tensorcore` | `tc-buffer-alloc` | [tc-buffer-alloc](shipped_exports.md#tc-buffer-alloc) |
| `tensorcore` | `tc-buffer-free` | [tc-buffer-free](shipped_exports.md#tc-buffer-free) |
| `tensorcore` | `tc-buffer-map` | [tc-buffer-map](shipped_exports.md#tc-buffer-map) |
| `tensorcore` | `tc-buffer-size` | [tc-buffer-size](shipped_exports.md#tc-buffer-size) |
| `tensorcore` | `tc-device-info` | [tc-device-info](shipped_exports.md#tc-device-info) |
| `tensorcore` | `tc-device-name` | [tc-device-name](shipped_exports.md#tc-device-name) |
| `tensorcore` | `tc-gemm` | [tc-gemm](shipped_exports.md#tc-gemm) |
| `tensorcore` | `tc-gemm-bf16` | [tc-gemm-bf16](shipped_exports.md#tc-gemm-bf16) |
| `tensorcore` | `tc-gemm-fp16` | [tc-gemm-fp16](shipped_exports.md#tc-gemm-fp16) |
| `tensorcore` | `tc-gemm-fp32` | [tc-gemm-fp32](shipped_exports.md#tc-gemm-fp32) |
| `tensorcore` | `tc-init` | [tc-init](shipped_exports.md#tc-init) |
| `tensorcore` | `tc-last-backend` | [tc-last-backend](shipped_exports.md#tc-last-backend) |
| `tensorcore` | `tc-last-backend-name` | [tc-last-backend-name](shipped_exports.md#tc-last-backend-name) |
| `tensorcore` | `tc-last-status` | [tc-last-status](shipped_exports.md#tc-last-status) |
| `tensorcore` | `tc-runtime-capabilities` | [tc-runtime-capabilities](shipped_exports.md#tc-runtime-capabilities) |
| `tensorcore` | `tc-runtime-capabilities-abi-version` | [tc-runtime-capabilities-abi-version](shipped_exports.md#tc-runtime-capabilities-abi-version) |
| `tensorcore` | `tc-runtime-capabilities-status` | [tc-runtime-capabilities-status](shipped_exports.md#tc-runtime-capabilities-status) |
| `tensorcore` | `tc-shutdown` | [tc-shutdown](shipped_exports.md#tc-shutdown) |
| `tensorcore` | `tc-status-string` | [tc-status-string](shipped_exports.md#tc-status-string) |
| `tensorcore` | `tc-version` | [tc-version](shipped_exports.md#tc-version) |
| `web.web` | `web.web` | [web.web](shipped_exports.md#web.web) |

## Not covered here

- The VM geometric builtin surface is documented in
  [geometry.md](geometry.md); these names are native VM operations rather than
  a `require`-loaded standard-library module.
- `core.test-module` (`lib/core/test-module.esk`) and `ml.nested_test_module`
  (`lib/ml/nested_test_module.esk`) are module-loader test fixtures, not user API.
- `lib/agent/*` (HTTP client, sqlite, subprocess, regex, crypto/sha256, terminal, …)
  is the agent-FFI domain, documented separately from the stdlib reference.
- `stdlib`-level helpers (`random-tensor`, `time-it`, `time-ns`, keyword-argument
  internals, …) are covered in [stdlib_extras.md](stdlib_extras.md).
