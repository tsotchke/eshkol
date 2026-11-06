# Eshkol Standard Library — API Reference Index

Reference map of the documented Eshkol standard-library modules. Module rows
link to their reference pages and source; the supplemental tables below cover
additional shipped exports. Execution evidence is recorded per example and
qualification suite.

See [module-system.md](module-system.md) for how `require` resolution,
`(require stdlib)`, and `stdlib.o` precompilation work. The v1.2-scale surface
notes remain in [../../STDLIB_V1_2_API.md](../../STDLIB_V1_2_API.md).

**Primary module-table entries: 138** (plus `stdlib` itself and the module system page) — **provided symbols in that table: 2034** (plus stdlib-level helpers in [stdlib_extras.md](stdlib_extras.md)).

*Auto* = loaded automatically by `(require stdlib)`; otherwise the module must be required individually.

The interval and Taylor-model family — `core.ad.interval`, `core.ad.taylor_models` and the
proof-backed `core.ad.rigorous_interval` / `core.ad.rigorous_taylor_models` beneath them, all
reached through `(require core.ad.taylor_models)` — is documented together in
[certified-enclosures.md](certified-enclosures.md); its symbols are listed in the shipped-exports
table further down this page. The directed-rounding primitives `fl-next-up` and
`fl-next-down` that layer rests on are **runtime builtins** on both engines, so
they need no `require`; see
[certified-enclosures.md §1](certified-enclosures.md#1-directed-rounding-primitives-runtime)
and [the numeric tower](../language/numeric-tower.md#exact-roots-and-exact-expt).

| Module | Auto | Reference | Provided symbols |
|---|---|---|---|
| [`core.dbsp.sqlite_changefeed`](../../../lib/core/dbsp/sqlite_changefeed.esk) | no | [dbsp_sqlite_changefeed.md](dbsp_sqlite_changefeed.md) | `dbsp-sqlite-create!` `dbsp-sqlite-enqueue!` `dbsp-sqlite-consume!` `dbsp-sqlite-state` |
| [`core.certified_native_solve`](../../../lib/core/certified_native_solve.esk) | no | [certified_native_solve.md](certified_native_solve.md) | `solve-certified-native` `verify-certified-native-solve` |
| [`core.certified_solve_pullback`](../../../lib/core/certified_solve_pullback.esk) | no | [certified_solve_pullback.md](certified_solve_pullback.md) | `solve-certified-native-pullback` `verify-certified-native-pullback` |
| [`core.ad.tape`](../../../lib/core/ad/tape.esk) | no | [ad_tape.md](ad_tape.md) | `ad-tape-new` `ad-const` `ad-var` `ad-add` `ad-sub` `ad-mul` `ad-div` `ad-sin` `ad-cos` `ad-exp` `ad-log` `ad-sqrt` `ad-neg` `ad-abs` `ad-relu` `ad-sigmoid` `ad-tanh` `ad-backward` `ad-gradient` `ad-node-value` `with-tape` `current-tape` `make-tape` `tape-input` `tape-const` `node-value` `node-grad` `record-op!` `record-fd-op!` `tape-add` `tape-sub` `tape-mul` `tape-div` `tape-sin` `tape-cos` `tape-exp` `tape-log` `tape-neg` `tape-square` `tape-sqrt` `tape-tanh` `tape-sigmoid` `tape-relu` `tape-pow` `tape-gradient` `tape-pullback` `tape-seal!` `tape-snapshot` `tape-restore` |
| [`core.ad.tensor_circuit_reverse`](../../../lib/core/ad/tensor_circuit_reverse.esk) | no | [ad_tensor_circuit_reverse.md](ad_tensor_circuit_reverse.md) | `make-tensor-circuit-pullback` `tensor-circuit-pullback-step` `tensor-circuit-pullback-output` `tensor-circuit-pullback-work` |
| [`core.ad.algebraic_pullback`](../../../lib/core/ad/algebraic_pullback.esk) | no | [ad_algebraic_pullback.md](ad_algebraic_pullback.md) | `algebraic-solve-with-pullback` `algebraic-factor-with-pullback` `algebraic-system-with-pullback` `algebraic-reverse-solve` `algebraic-reverse-system` |
| [`core.ad.algebraic_farkas`](../../../lib/core/ad/algebraic_farkas.esk) | no | [ad_algebraic_farkas.md](ad_algebraic_farkas.md) | `algebraic-farkas-search` `algebraic-farkas-verify` |
| [`core.ad.algebraic_qp`](../../../lib/core/ad/algebraic_qp.esk) | no | [ad_algebraic_qp.md](ad_algebraic_qp.md) | `algebraic-qp-solve` `algebraic-qp-solution` `algebraic-qp-info` `algebraic-qp-verify` `algebraic-qp-pullback` |
| [`core.ad.algebraic_ridge`](../../../lib/core/ad/algebraic_ridge.esk) | no | [ad_algebraic_ridge.md](ad_algebraic_ridge.md) | `algebraic-ridge-new` `algebraic-ridge-step` `algebraic-ridge-output` `algebraic-ridge-info` `algebraic-ridge-pullback` |
| [`core.ad.certified_continuation`](../../../lib/core/ad/certified_continuation.esk) | no | [ad_certified_continuation.md](ad_certified_continuation.md) | `make-certified-continuation` `certified-continuation-advance` `certified-continuation-enclosure` `verify-certified-continuation` |
| [`core.ad.derived_operator`](../../../lib/core/ad/derived_operator.esk) | no | [ad_derived_operator.md](ad_derived_operator.md) | `make-derived-operator` `derived-operator-value` `derived-operator-jvp` `derived-operator-vjp` `derived-operator-delta` `derived-operator-info` `derived-operator-enclosure` |
| [`core.certified_nonlinear`](../../../lib/core/certified_nonlinear.esk) | no | [certified_nonlinear.md](certified_nonlinear.md) | `make-certified-polynomial-system` `certify-polynomial-root` `verify-polynomial-root-certificate` |
| [`core.ad.algebraic_spectral`](../../../lib/core/ad/algebraic_spectral.esk) | no | [ad_algebraic_spectral.md](ad_algebraic_spectral.md) | `algebraic-characteristic` `algebraic-eigenvalue-with-pullback` `algebraic-eigenvalue-lift` |
| [`core.ad.algebraic_subspace`](../../../lib/core/ad/algebraic_subspace.esk) | no | [ad_algebraic_subspace.md](ad_algebraic_subspace.md) | `make-algebraic-subspace` `algebraic-subspace-projector` `algebraic-subspace-jvp` `algebraic-subspace-vjp` `algebraic-subspace-info` `verify-algebraic-subspace-jvp` `make-algebraic-subspace-from-factors` |
| [`core.ad.root_branch`](../../../lib/core/ad/root_branch.esk) | no | [ad_root_branch.md](ad_root_branch.md) | `make-root-branch` `root-branch-step` `root-branch-enclosure` `root-branch-time` `root-branch-sensitivity` `verify-root-branch-step` |

| [`core.alist`](../../../lib/core/alist.esk) | yes | [alist.md](alist.md) | `alist-ref` `alist-ref-or` `alist-set` |
| [`core.argparse`](../../../lib/core/argparse.esk) | no | [argparse.md](argparse.md) | `parse-args` `arg-get` `arg-positional` `arg-has?` `argparse-help` |
| [`core.cache`](../../../lib/core/cache.esk) | no | [cache.md](cache.md) | `make-lru-cache` `lru-get` `lru-get/default` `lru-set!` `lru-has?` `lru-delete!` `lru-size` `lru-capacity` `lru-clear!` `memoize` `memoize/cap` `memoize1` `memoize1/cap` `memoize2` `memoize2/cap` |
| [`core.capabilities`](../../../lib/core/capabilities.esk) | yes | [capabilities.md](capabilities.md) | `capability-install-policy!` `capability-clear-policy!` `capability-policy` `capability-policy-active?` `capability-allowed?` `capability-require!` |
| [`core.certified_solve`](../../../lib/core/certified_solve.esk) | no | [certified_solve.md](certified_solve.md) | `certify-linear-solve` `verify-linear-solve-certificate` |
| [`core.channels`](../../../lib/core/channels.esk) | no | [channels.md](channels.md) | `make-channel` `channel?` `channel-capacity` `channel-len` `channel-closed?` `channel-send!` `channel-recv!` `channel-try-send!` `channel-try-recv!` `channel-close!` `channel-closed-sentinel` |
| [`core.collections`](../../../lib/core/collections.esk) | no | [collections.md](collections.md) | `make-pq` `pq-push!` `pq-pop!` `pq-peek` `pq-size` `pq-empty?` `make-set` `set-add!` `set-remove!` `set-contains?` `set-size` `set->list` `set-clear!` `set-union` `set-intersect` `set-difference` `make-deque` `deque-push-front!` `deque-push-back!` `deque-pop-front!` `deque-pop-back!` `deque-peek-front` `deque-peek-back` `deque-size` `deque-empty?` `deque->list` |
| [`core.ring`](../../../lib/core/ring.esk) | no | [ring.md](ring.md) | `make-ring-carrier` `make-family-carrier` `ring-carrier?` `ring-ref` `ring-carrier=?` `ring-carrier-name` `ring-carrier-family` `ring-carrier-family-id` `ring-carrier-parameters` `ring-carrier-abi-version` `ring-carrier-extra` `ring-abi-version` `ring-families` `ring-registered-carriers` `ring-register-recognizer!` `r0` `r1` `r+` `r-` `r*` `r-neg` `r-inv` `r/` `r-expt` `r=` `r-int` `r-zero?` `r-one?` `r-unit?` `r<` `r-sum` `r-product` `r-quotient` `r-remainder` `r-gcd` `r-analytic` `ring-primal` `ring-element` `ring-element?` `ring-element-value` `ring-carrier-of` `ring-member?` `ring-admit` `ring-join` `ring-binary` `ring-equal?` `ring-refusal?` `ring-refusal-kind` `ring-refusal-irritants` `integers` `rationals` `flonums` `numeric-tower` `make-zmod` `make-jets` `make-polys` `zset-group-ring` `make-jet` `jet-seed` `jet-constant` `jet-coefficients` `jet-coefficient` `ring-poly` `ring-poly-variable` `ring-poly-coefficients` `ring-poly-degree` `ring-poly-leading` `ring-poly-eval` `make-ring-map` `ring-map?` `ring-map-apply` `ring-map-source` `ring-map-target` `ring-map-compose` `ring-base-change` `zmod-reduction` `jet-primal` `ring-poly-evaluation` `group-ring-character` `ring-det` `ring-solve` `ring-law-violations` `exact-prime?` `coefficient-field->ring` `ring->coefficient-field` |
| [`core.multimodular`](../../../lib/core/multimodular.esk) | no | [multimodular.md](multimodular.md) | `multimodular-prime` `multimodular-primes` `chinese-remainder` `symmetric-residue` `rational-reconstruct` `hadamard-bound` `mignotte-gcd-bound` `multimodular-integer` `multimodular-rational` `multimodular-integer-report` `multimodular-rational-report` `multimodular-det` `multimodular-solve` `multimodular-poly-gcd` |
| [`core.polynomial_exact`](../../../lib/core/polynomial_exact.esk) | no | [polynomial_exact.md](polynomial_exact.md) | `exact-poly` `exact-poly-zero?` `exact-poly-sub` `exact-poly-scale` `exact-poly-mul` `exact-poly-divmod` `exact-poly-inverse-mod` `exact-poly-eval` `exact-poly-real-root-count` `exact-poly-isolate` |
| [`core.algebraic_field`](../../../lib/core/algebraic_field.esk) | no | [algebraic_field.md](algebraic_field.md) | `make-algebraic-field` `algebraic-field-degree` `algebraic-field-polynomial` `algebraic-generator` `algebraic-element` `algebraic-coefficients` `algebraic-add` `algebraic-sub` `algebraic-scale` `algebraic-mul` `algebraic-inverse` `algebraic-zero?` `algebraic-equal?` `algebraic-rational` `algebraic-enclosure` `algebraic-sign` `make-algebraic-embedding` `make-algebraic-compositum` `algebraic-field-certificate` `make-multiquadratic-field` |
| [`core.algebraic_domain`](../../../lib/core/algebraic_domain.esk) | no | [algebraic_linalg.md](algebraic_linalg.md) | `algebraic-domain` `algebraic-domain-ref` |
| [`core.algebraic_linalg`](../../../lib/core/algebraic_linalg.esk) | no | [algebraic_linalg.md](algebraic_linalg.md) | `algebraic-factor` `algebraic-factor-solve` `algebraic-factor-solve-transpose` `algebraic-factor-summary` `algebraic-matrix-mul` `algebraic-solve` `algebraic-factor-positive-definite?` |
| [`core.algebraic_update`](../../../lib/core/algebraic_update.esk) | no | [algebraic_update.md](algebraic_update.md) | `algebraic-update-factor` `algebraic-update` `algebraic-update-symmetric` `algebraic-update-solve` `algebraic-update-solve-transpose` `algebraic-update-pullback` `algebraic-update-info` |
| [`core.ad.algebraic_implicit`](../../../lib/core/ad/algebraic_implicit.esk) | no | [ad_algebraic_implicit.md](ad_algebraic_implicit.md) | `algebraic-ad-system-root` `algebraic-ad-system-residual` |
| [`core.ad.algebraic_incremental`](../../../lib/core/ad/algebraic_incremental.esk) | no | [ad_algebraic_incremental.md](ad_algebraic_incremental.md) | `make-algebraic-incremental-plan` `algebraic-incremental-new` `algebraic-incremental-step` `algebraic-incremental-output` `algebraic-incremental-info` `algebraic-incremental-moment-ref` |
| [`core.ad.algebraic_reverse`](../../../lib/core/ad/algebraic_reverse.esk) | no | [ad_algebraic_reverse.md](ad_algebraic_reverse.md) | `algebraic-value-and-pullback` `algebraic-reverse-value` `algebraic-reverse-constant` `algebraic-reverse-add` `algebraic-reverse-sub` `algebraic-reverse-mul` `algebraic-reverse-inverse` `algebraic-reverse-pow` `algebraic-hvp` |
| [`core.ad.algebraic_sparse`](../../../lib/core/ad/algebraic_sparse.esk) | no | [ad_algebraic_sparse.md](ad_algebraic_sparse.md) | `algebraic-sparse-hessian` `algebraic-sparse-hessian-ref` `algebraic-sparse-hessian-dense` `algebraic-sparse-hessian-summary` |
| [`core.ad.incremental_reverse`](../../../lib/core/ad/incremental_reverse.esk) | no | [ad_incremental_reverse.md](ad_incremental_reverse.md) | `make-incremental-pullback` `incremental-pullback-step` `incremental-pullback-output` `incremental-pullback-work` `incremental-pullback-info` |
| [`core.ad.sparsity`](../../../lib/core/ad/sparsity.esk) | no | [ad_sparsity.md](ad_sparsity.md) | `hessian-color-graph` |
| [`core.algebraic_batch`](../../../lib/core/algebraic_batch.esk) | no | [algebraic_batch.md](algebraic_batch.md) | `make-algebraic-multiplication-plan` `algebraic-batch-mul` |
| [`core.control.trampoline`](../../../lib/core/control/trampoline.esk) | yes | [control_trampoline.md](control_trampoline.md) | `trampoline` `bounce` `done` |
| [`core.cyclotomic`](../../../lib/core/cyclotomic.esk) | no | [cyclotomic.md](cyclotomic.md) | `make-cyclotomic-field` `cyclotomic-order` `cyclotomic-degree` `cyclotomic-polynomial` `cyclotomic-element` `cyclotomic-coefficients` `cyclotomic-root` `cyclotomic-add` `cyclotomic-sub` `cyclotomic-scale` `cyclotomic-mul` `cyclotomic-inverse` `cyclotomic-zero?` `cyclotomic-equal?` `cyclotomic-rational` `cyclotomic-galois` `cyclotomic-norm` |
| [`core.sheaf`](../../../lib/core/sheaf.esk) | no | [sheaf.md](sheaf.md) | (loads core.integer_linalg and the five core.sheaf modules below) |
| [`core.integer_linalg`](../../../lib/core/integer_linalg.esk) | no | [integer_linalg.md](integer_linalg.md) | `integer-matrix` `integer-identity-matrix` `integer-matrix-rows` `integer-matrix-cols` `integer-matrix-ref` `integer-matrix-set!` `integer-matrix-copy` `integer-matrix-mul` `integer-matrix-equal?` `integer-matrix-from-columns` `integer-matrix-column` `integer-matrix-vector` `integer-matrix-vstack` `integer-matrix-hstack` `smith-normal-form` `smith-diagonal` `smith-left` `smith-right` `integer-rank` `integer-kernel` `integer-solve-in-basis` `integer-in-column-span?` `integer-same-lattice?` `integer-invariant-factors` `matrix-clear-denominators` `rational-rank` `rational-kernel` `rational-solve` `rational-same-span?` `cochain-cohomology` `cochain-cohomology-generators` |
| [`core.sheaf.site`](../../../lib/core/sheaf/site.esk) | no | [sheaf.md](sheaf.md) | `make-finite-category` `make-category-from-presentation` `poset-category` `category-opposite` `finite-category?` `category-object-count` `category-arrow-count` `category-arrows` `arrow-source` `arrow-target` `arrow-name` `category-identity` `category-compose` `category-composable?` `category-hom` `category-arrows-into` `category-arrows-from` `category-object-label` `category-object-of-label` `category-laws-violations` `category-laws?` `category-arrow-named` `category-thin?` `category-meet` `category-thin-arrow` `sieve?` `sieve-maximal` `sieve-generated` `sieve-pullback` `sieve-intersection` `sieve-union` `sieve-contains?` `sieve-subset?` `sieves-on` `sieve-domains` `make-site` `make-site-from-sieves` `site?` `site-category` `site-covering?` `site-covering-sieves` `site-minimal-covering-sieve` `grothendieck-topology-violations` `grothendieck-topology?` `trivial-site` `dense-site` `make-finite-space` `finite-space?` `finite-space-size` `finite-space-leq?` `finite-space-lt?` `finite-space-down-set` `finite-space-up-set` `finite-space-open?` `finite-space-opens` `finite-space-maximal-points` `finite-space-minimal-points` `finite-space-product` `finite-space-components` `finite-space-covers` `finite-space-finest-cover` `finite-space-chains` `finite-space-site` `sheaf-subsets` `sheaf-subsets-of-size` `sheaf-intersect-all` `sheaf-set-equal?` |
| [`core.sheaf.glue`](../../../lib/core/sheaf/glue.esk) | no | [sheaf.md](sheaf.md) | `gluing-success` `gluing-failure` `gluing-result?` `glued?` `glued-value` `gluing-obstructions` `make-cover` `cover?` `cover-pieces` `cover-overlap` `cover-restrict` `cover-same?` `cover-disagreements` `glue-cover` `keyed-cover` `glue-keyed-sections` |
| [`core.sheaf.presheaf`](../../../lib/core/sheaf/presheaf.esk) | no | [sheaf.md](sheaf.md) | `make-set-presheaf` `make-set-presheaf-from-indices` `set-presheaf?` `presheaf-category` `presheaf-elements` `presheaf-element-count` `presheaf-element-index` `presheaf-restrict` `presheaf-restrict-index` `presheaf-functoriality-violations` `presheaf-functorial?` `constant-set-presheaf` `representable-presheaf` `family-incompatibilities` `matching-family?` `matching-families` `family-amalgamations` `restriction-family` `glue-family` `presheaf-sheaf-violations` `presheaf-sheaf?` `presheaf-separated?` `presheaf-plus` `sheafify` `presheaf-unit-map` `presheaf-natural?` |
| [`core.sheaf.module`](../../../lib/core/sheaf/module.esk) | no | [sheaf.md](sheaf.md) | `make-module-presheaf` `module-presheaf?` `module-presheaf-category` `module-presheaf-ring` `module-presheaf-rank` `module-presheaf-restriction` `module-presheaf-bases` `module-presheaf-functoriality-violations` `module-presheaf-functorial?` `constant-module-presheaf` `ring-kernel` `ring-rank` `ring-solve` `ring-same-span?` `ring-cochain-cohomology` `module-sieve-offsets` `module-restriction-to-sieve` `module-matching-basis` `module-family-incompatibilities` `module-glue` `module-separated-on-sieve?` `module-sheaf-on-sieve?` `module-sheaf-on-family?` `module-separated-on-family?` `module-presheaf-sheaf-violations` `module-presheaf-sheaf?` `module-presheaf-plus` `module-sheafify` `module-presheaf-unit` |
| [`core.sheaf.cohomology`](../../../lib/core/sheaf/cohomology.esk) | no | [sheaf.md](sheaf.md) | `cech-complex-from-nerve` `module-cech-complex` `module-cech-cohomology` `make-poset-module` `poset-local-system` `constant-poset-module` `poset-module?` `poset-module-space` `poset-module-ring` `poset-module-rank` `poset-module-map` `poset-module-functoriality-violations` `poset-module-functorial?` `poset-module-cochain-complex` `poset-module-cohomology` `poset-module-cohomology-generators` `poset-module-sections` `poset-module-restrict-section` `poset-module-cech-complex` `poset-module-cech-cohomology` `poset-module->module-presheaf` `finite-space-euler-characteristic` `cohomology-generators` |
| [`core.topos`](../../../lib/core/topos.esk) | no | [topos.md](topos.md) | `make-topos` `presheaf-topos` `topos?` `topos-site` `topos-category` `topos-object` `topos-terminal` `topos-omega` `topos-product` `topos-exponential` `topos-power` `topos-morphisms` `topos-type?` `topos-type-kind` `topos-type-presheaf` `topos-type-size` `topos-type-elements` `topos-type-element` `topos-type-index` `topos-type-restrict` `topos-type-factors` `topos-exponential-apply` `topos-closed-sieves` `topos-closure` `topos-closed?` `topos-omega-top` `topos-omega-bottom` `topos-omega-meet` `topos-omega-join` `topos-omega-implies` `topos-omega-not` `topos-omega-boolean?` `topos-lawvere-tierney` `topos-lawvere-tierney?` `topos-lawvere-tierney-site` `topos-classifying-map` `topos-classified-subobject` `topos-function` `topos-constant` `topos-predicate` `make-topos-language` `topos-language-topos` `topos-language-type` `topos-formula` `topos-formula-context` `topos-forces?` `topos-truth` `topos-environments` `topos-extension` `topos-valid?` `topos-counter-witness` `topos-cross-check` `topos-monotone?` `topos-local?` `topos-forcing-report` `topos-instance-count` |
| [`core.simplicial`](../../../lib/core/simplicial.esk) | no | [simplicial.md](simplicial.md) | (loads the four core.simplicial modules below) |
| [`core.simplicial.sset`](../../../lib/core/simplicial/sset.esk) | no | [simplicial.md](simplicial.md) | `simplex` `simplex?` `simplex-degree` `simplex-base` `simplex-base-degree` `simplex-surjection` `simplex-nondegenerate?` `simplex-degenerate?` `simplex=?` `make-simplicial-set` `simplicial-set?` `simplicial-set-name` `simplicial-set-max-dim` `simplicial-set-nondegenerate` `simplicial-set-count` `simplicial-set-index` `simplicial-set-nondegenerate-ref` `simplicial-set-face` `simplicial-set-degeneracy` `simplicial-set-operator` `simplicial-set-restrict` `simplicial-set-faces` `simplicial-set-front-face` `simplicial-set-back-face` `simplicial-set-vertices-of` `simplicial-set-simplices` `simplicial-set-identity-violations` `simplicial-set-euler-characteristic` `simplicial-set-from-all` `make-delta-complex` `delta-complex-faces` `make-simplicial-complex` `simplicial-complex-simplices` `delta-complex-from-tetrahedra` `lens-space` `real-projective-space` `standard-simplex` `simplex-boundary` `horn` `simplicial-sphere` `discrete-simplicial-set` `point-simplicial-set` `simplicial-skeleton` `simplicial-product` `simplicial-product-simplex` `simplicial-join` `simplicial-cone` `simplicial-suspension` `monotone-surjections` `monotone-injections` `ordered-subsets` |
| [`core.simplicial.group`](../../../lib/core/simplicial/group.esk) | no | [simplicial.md](simplicial.md) | `make-finite-group` `finite-group?` `group-order` `group-mul` `group-inv` `group-identity` `group-elements` `group-label` `group-axioms-violations` `group-abelian?` `group-element-order` `group-power` `group-conjugacy-classes` `group-centralizer` `group-commute?` `cyclic-group` `group-direct-product` `dihedral-group` `quaternion-group` `symmetric-group` `group-homomorphism?` `group-quotient` `make-crossed-module` `crossed-module?` `crossed-module-source` `crossed-module-target` `crossed-module-boundary` `crossed-module-act` `crossed-module-violations` `crossed-module-pi1` `crossed-module-pi2` `crossed-module-pi1-order` `crossed-module-pi2-order` `crossed-module-k-invariant` `group-crossed-module` `abelian-crossed-module` |
| [`core.simplicial.nerve`](../../../lib/core/simplicial/nerve.esk) | no | [simplicial.md](simplicial.md) | `finite-category-nerve` `poset-nerve` `order-complex` `classifying-space` `face-poset` `face-poset-points` `barycentric-subdivision` `mccord-finite-model` `crossed-module-nerve` `crossed-module-simplices` `crossed-module-simplex-face` `crossed-module-simplex-degeneracy` `nerve-string-normal-form` |
| [`core.simplicial.kan`](../../../lib/core/simplicial/kan.esk) | no | [simplicial.md](simplicial.md) | `simplicial-horns` `horn-filler-table` `horn-fillers` `horn-filler-report` `horn-violations` `kan-complex-through?` `quasi-category-through?` `basepoint-simplex` `kan-spheres` `kan-homotopic?` `kan-homotopy-relation` `kan-homotopy-classes` `kan-pi1-product` `simplicial-maps-fold` `simplicial-maps-count` `simplicial-map-order` |
| [`core.homological`](../../../lib/core/homological.esk) | no | [homological.md](homological.md) | (loads the four core.homological modules below) |
| [`core.homological.complex`](../../../lib/core/homological/complex.esk) | no | [homological.md](homological.md) | `coefficient-kind` `coefficient-field?` `coefficient-reduce` `coefficient-reduce-vector` `coefficient-reduce-matrix` `coefficient-rank` `coefficient-kernel` `coefficient-solve` `coefficient-in-span?` `coefficient-matrix-mul` `coefficient-matrix-vector` `coefficient-zero-vector?` `coefficient-vector-add` `coefficient-vector-scale` `make-cochain-complex` `make-chain-complex` `complex?` `complex-kind` `complex-ring` `complex-dims` `complex-dim` `complex-top-degree` `complex-map` `complex-in-map` `complex-out-map` `complex-d-squared-violations` `homology` `cohomology` `homology-betti` `homology-torsion` `homology-rank-over` `homology-group-order` `homology-generators` `homology-cycle?` `homology-boundary?` `homology-classes-independent?` `homology-coordinates` `complex-dual` `complex-change-ring` `complex-direct-sum` `complex-shift` `complex-euler-characteristic` `complex-tensor` `make-chain-map` `chain-map?` `chain-map-source` `chain-map-target` `chain-map-component` `chain-map-violations` `chain-map-compose` `chain-map-identity` `chain-homotopy-violations` `induced-map` `mapping-cone` `mapping-cone-inclusion` `mapping-cone-projection` `long-exact-sequence` `long-exact-sequence-exact?` `make-double-complex` `double-complex?` `double-complex-dim` `double-complex-horizontal` `double-complex-vertical` `double-complex-violations` `total-complex` `total-complex-offsets` `qz-coboundary?` `qz-coboundary-test` `torsion-exponent` |
| [`core.homological.abelian`](../../../lib/core/homological/abelian.esk) | no | [homological.md](homological.md) | `make-fp-group` `fp-group?` `fp-group-generators` `fp-group-relations` `abelian-group` `fp-group-invariants` `fp-hom-well-defined?` `fp-kernel` `fp-cokernel` `fp-subquotient` `lattice-basis` `lattice-quotient-invariants` `lattice-contains?` `abelian-tensor` `abelian-tor` `abelian-hom` `abelian-ext` `universal-coefficients-cohomology` `universal-coefficients-mod-p` |
| [`core.homological.spectral`](../../../lib/core/homological/spectral.esk) | no | [homological.md](homological.md) | `make-filtered-complex` `filtered-complex?` `filtered-complex-complex` `filtered-complex-degree` `filtered-complex-violations` `filtered-complex-range` `spectral-sequence-entry` `spectral-sequence-page` `spectral-sequence-differential-rank` `spectral-sequence-page-violations` `spectral-sequence-limit-page` `filtration-graded-cohomology` `spectral-sequence-convergence-violations` `double-complex-filtered-complex` |
| [`core.homological.simplicial`](../../../lib/core/homological/simplicial.esk) | no | [homological.md](homological.md) | `simplicial-boundary-matrix` `simplicial-chain-complex` `simplicial-cochain-complex` `simplicial-moore-complex` `simplicial-homology` `simplicial-cohomology` `cochain-value` `cochain-from-function` `cochain-basis-vector` `simplicial-coboundary` `cup-product` `cup-i-product` `steenrod-square` `bockstein` `group-cochain-complex` `group-cohomology` `group-homology` `group-chain-complex` `trivial-group-module` `make-group-module` |
| [`core.category`](../../../lib/core/category.esk) | no | [category.md](category.md) | (loads core.sheaf.site, core.sheaf.presheaf and the seven core.category modules below) |
| [`core.category.finite`](../../../lib/core/category/finite.esk) | no | [category.md](category.md) | `discrete-category` `terminal-category` `free-category` `walking-arrow-category` `parallel-pair-category` `span-category` `cospan-category` `commutative-square-category` `monoid-category` `cyclic-group-category` `category-product` `category-product-object` `category-product-arrow` `category-coproduct` `category-full-subcategory` `category-full-subcategory-inclusion` `category-arrow-list` `category-object-list` `category-identity?` `category-isomorphism?` `category-inverse` `category-groupoid?` `category-initial-objects` `category-terminal-objects` `category-components` `category-nerve` `category-nerve-coboundary` `category-nerve-cohomology` |
| [`core.category.functor`](../../../lib/core/category/functor.esk) | no | [category.md](category.md) | `make-functor` `functor-from-procedures` `category-functor?` `functor-source` `functor-target` `functor-object` `functor-arrow` `functor-objects` `functor-arrows` `functor-laws-violations` `functor-laws?` `functor-equal?` `identity-functor` `functor-compose` `constant-functor` `object-functor` `terminal-functor` `functor-opposite` `inclusion-functor` `product-projections` `functors-between` `natural-transformation-violations` `natural-transformation?` `natural-transformations` `identity-transformation` `transformation-compose` `whisker-left` `whisker-right` `horizontal-compose` `natural-isomorphism?` `functor-category` `functor-category-functors` `comma-category` `comma-domain-projection` `comma-codomain-projection` `slice-category` `coslice-category` `sieve-category` `make-category-diagram` `category-diagram?` `category-diagram-base` `category-diagram-category` `category-diagram-functor` `category-diagram-violations` `grothendieck-construction` `grothendieck-projection` `make-adjunction` `adjunction?` `adjunction-left` `adjunction-right` `adjunction-unit` `adjunction-counit` `adjunction-violations` `adjunction-hom-bijection-violations` `right-adjoint` `left-adjoint` `make-monad` `monad?` `monad-category` `monad-functor` `monad-unit` `monad-multiplication` `adjunction-monad` `monad-laws-violations` `monad-algebras` |
| [`core.category.set`](../../../lib/core/category/set.esk) | no | [category.md](category.md) | `make-set-functor` `make-set-functor-from-indices` `make-set-functor-from-names` `set-functor?` `set-functor-category` `set-functor-elements` `set-functor-element-count` `set-functor-element-index` `set-functor-sizes` `set-functor-apply` `set-functor-apply-index` `set-functor-functoriality-violations` `set-functor-functorial?` `presheaf->set-functor` `set-functor->presheaf` `set-functor-restrict` `constant-set-functor` `corepresentable-set-functor` `hom-set-functor` `set-functor-product` `set-natural?` `set-natural-transformations` `set-natural-transformation-count` `set-identity-transformation` `set-transformation-compose` `set-transformation-identity?` `set-transformation-bijective?` `set-natural-isomorphism?` `set-limit` `set-limit-size` `set-limit-element-values` `set-cone-count` `set-colimit` `set-colimit-size` `set-colimit-class` `set-colimit-representatives` `set-end` `set-coend` `set-coend-class` `set-natural-transformations-end` `yoneda-element` `yoneda-transformation` `yoneda-violations` `presheaf-yoneda-violations` `elements-category` `elements-projection` `presheaf-elements-category` `presheaf-elements-projection` `discrete-opfibration?` `discrete-fibration?` |
| [`core.category.kan`](../../../lib/core/category/kan.esk) | no | [category.md](category.md) | `set-lan` `set-ran` `set-lan-coend` `set-ran-end` `set-lan-comparison` `set-ran-comparison` `set-lan-unit` `set-lan-counit` `set-lan-map` `set-ran-unit` `set-ran-counit` `set-ran-map` `set-restrict-map` `set-lan-adjunction-violations` `set-ran-adjunction-violations` `presheaf-ran-along-inclusion` |
| [`core.category.linear`](../../../lib/core/category/linear.esk) | no | [category.md](category.md) | `module-presheaf-limit` `module-limit-rank` `module-limit-basis` `module-limit-component` `module-presheaf-colimit` `module-colimit-rank` `module-colimit-torsion` `module-presheaf-limit-relations` `module-presheaf-colimit-relations` `module-presheaf-restrict` `poset-module->point-presheaf` `module-presheaf-cosimplicial-complex` `module-presheaf-derived-limits` `module-presheaf-map-hypercohomology` `module-cohomology-mod-p-dimensions` |
| [`core.category.descent`](../../../lib/core/category/descent.esk) | no | [category.md](category.md) | `category-components` `cover-nerve-category` `cech-derived-limits` `finite-space-cover-descent-category` `finite-space-cech-nerve-category` `cech-nerve-cell-counts` `module-sieve-limit` `module-sheaf-at-by-limit` `module-sieve-unit-shape` `finite-space-cover-sections-presheaf` `finite-space-sections` `sieve-category` `sieve-matching-limit` `sieve-descent-violations` `presheaf-ran-along-inclusion` `cech-equalizer` `cech-descent-violations` `presheaf-sheaf-violations-by-limit` `presheaf-plus-map` `sheafification-unit` `sheafify-map` `sheafification-counit` `sheafification-adjunction-violations` `sheafification-monad-violations` |
| [`core.category.cohomology`](../../../lib/core/category/cohomology.esk) | no | [category.md](category.md) | `cover-nerve-category` `cech-derived-limits` `finite-space-cover-descent-category` `finite-space-cech-nerve-category` `cech-nerve-cell-counts` `module-sieve-limit` `module-sheaf-at-by-limit` `module-sieve-unit-shape` `finite-space-cover-sections-presheaf` `finite-space-sections` |
| [`core.effective`](../../../lib/core/effective.esk) | no | [effective.md](effective.md) | (loads the five core.effective modules below) |
| [`core.effective.chain`](../../../lib/core/effective/chain.esk) | no | [effective.md](effective.md) | `lc-zero` `lc-term` `lc-coefficient` `lc-cells` `lc-zero?` `lc-scale` `lc-negate` `lc-add` `lc-sub` `lc-sum` `lc-combination` `lc-apply` `lc-filter` `lc-relabel` `lc-equal?` `lc-max-by` `lc-size` `make-effective-complex` `effective-complex?` `effective-complex-name` `effective-complex-enumerated?` `effective-complex-basis` `effective-complex-rank` `effective-complex-d` `effective-complex-d-cell` `effective-complex-matrix` `effective-complex-chain-complex` `effective-complex-homology` `effective-complex-d-squared-violations` `effective-complex-cycle?` `effective-complex-index` `effective-complex-vector` `effective-complex-combination` `make-reduction` `reduction?` `reduction-big` `reduction-small` `reduction-f` `reduction-g` `reduction-h` `reduction-f-cell` `reduction-g-cell` `reduction-h-cell` `identity-reduction` `compose-reductions` `reduction-violations` `reduction-small-violations` |
| [`core.effective.perturbation`](../../../lib/core/effective/perturbation.esk) | no | [effective.md](effective.md) | `perturb-reduction` `perturb-small` `perturbation-series` `vector-field-reduction` `vector-field-critical?` `collapse-matching` `finite-complex-reduction` `make-word` `word?` `word-letters` `word-length` `word-degree` `word-letter-degree` `word-letter-cell` `tensor-complex` `tensor-reduction` `bar-complex` `bar-reduction` `word-tensor-d` |
| [`core.effective.simplicial`](../../../lib/core/effective/simplicial.esk) | no | [effective.md](effective.md) | `simplicial-effective-chains` `make-effective-space` `effective-space?` `effective-space-name` `effective-space-simplicial-set` `effective-space-chains` `effective-space-reduction` `effective-space-group` `effective-space-small` `effective-space-homology` `finite-effective-space` `make-simplicial-group` `simplicial-group?` `simplicial-group-set` `simplicial-group-mul` `simplicial-group-inv` `simplicial-group-identity` `simplicial-group-abelian?` `simplicial-group-violations` `simplex-collapses` `simplex-from-surjection` `simplex-expand-with` `eilenberg-zilber-status` `eilenberg-zilber-critical-simplex` `eilenberg-zilber-reduction` `eilenberg-zilber-formula-reduction` `alexander-whitney` `eilenberg-maclane-shuffle` `shih-homotopy` `shuffles` `product-space` `make-twisted-product` `twisted-product-twisting` `twisting-function-violations` `twisted-eilenberg-zilber-reduction` `twisted-product-space` `k-pi-1-group` `k-pi-1-reduction` `k-pi-1-simplex` `classifying-normal-form` `classifying-group` `classifying-status` `classifying-bar-reduction` `classifying-space-of` `eilenberg-maclane-space` `eilenberg-maclane-group-modulus` |
| [`core.effective.homotopy`](../../../lib/core/effective/homotopy.esk) | no | [effective.md](effective.md) | `eilenberg-maclane-space-cached` `cochain-pullback` `minimal-twisting` `cochain-to-eilenberg-maclane` `cocycle-twisting` `homology-functional` `cyclic-homology-functional` `whitehead-step` `homotopy-groups` `homotopy-group-report` `sphere-space` |
| [`core.effective.hypercover`](../../../lib/core/effective/hypercover.esk) | no | [effective.md](effective.md) | `finite-space-hypercover` `finite-space-cech-nerve` `hypercover-space` `hypercover-piece-point` `hypercover-open-of` `hypercover-boundaries` `hypercover-violations` `hypercover-level-counts` `mod-p-cohomology-dimensions` `two-type-cardinalities` `mapping-space-cardinality` `cup-square-three-type-cardinality` `uct-cohomology-order` `postnikov-cardinalities` |
| [`core.data.base64`](../../../lib/core/data/base64.esk) | yes | [data_base64.md](data_base64.md) | `base64-encode` `base64-decode` `base64-encode-string` `base64-decode-string` `string->bytes` `bytes->string` `base64-char-at` `base64-value` `string->bytes-helper` `base64-encode-helper` `base64-decode-helper` `base64-remove-padding` |
| [`core.data.csv`](../../../lib/core/data/csv.esk) | yes | [data_csv.md](data_csv.md) | `csv-parse` `csv-parse-file` `csv-stringify` `csv-write-file` `csv-parse-line` `csv-parse-lines` `csv-split-fields` `csv-stringify-row` |
| [`core.data.dataframe`](../../../lib/core/data/dataframe.esk) | yes | [data_dataframe.md](data_dataframe.md) | `csv-parse-typed` `csv-header` `csv-rows` `csv-column` `csv-column-idx` `csv-select` `csv-filter` `csv-map-column` `csv-row-count` `csv-col-count` `csv-describe` |
| [`core.data.radix`](../../../lib/core/data/radix.esk) | no | [data_radix.md](data_radix.md) | `radix-empty` `radix-ref` `radix-set` `radix-delete` `radix-fold` |
| [`core.dbsp`](../../../lib/core/dbsp.esk) | no | [dbsp.md](dbsp.md) | `zset-row-key` `zset-empty` `zset?` `zset-singleton` `zset-from-weighted` `zset-weight` `zset-add` `zset-negate` `zset-sub` `zset-scale` `zset-consolidate` `zset-empty?` `zset-positive?` `zset=?` `zset-entries` `zset-rows` `zset-size` `zset-map` `zset-filter` `zset-project` `zset-union` `zset-join` `zset-distinct` `zset-count` `zset-sum` `stream-zero` `stream-delay` `dbsp-delay` `stream-D` `stream-I` `stream-add` `stream-negate` `stream-lift` `dbsp-compose` `dbsp-incrementalize` `incremental-map` `incremental-filter` `incremental-project` `incremental-union` `incremental-join` `incremental-distinct` |
| [`core.dbsp.ad`](../../../lib/core/dbsp/ad.esk) | no | [dbsp_ad.md](dbsp_ad.md) | `dbsp-vector-group` `dbsp-ad-lift` |
| [`core.dbsp.circuit`](../../../lib/core/dbsp/circuit.esk) | no | [dbsp_circuit.md](dbsp_circuit.md) | `make-dbsp-group` `make-dbsp-circuit` `dbsp-circuit-step` `dbsp-circuit-output` `dbsp-circuit-input` `dbsp-circuit-work` `dbsp-circuit-checkpoint-state` |
| [`core.dbsp.ordered`](../../../lib/core/dbsp/ordered.esk) | no | [dbsp_ordered.md](dbsp_ordered.md) | `make-dbsp-ordered` `dbsp-ordered-step` `dbsp-ordered-select` `dbsp-ordered-quantile` `dbsp-ordered-rank` `dbsp-ordered-top` `dbsp-ordered-info` `dbsp-ordered-work` |
| [`core.dbsp.window`](../../../lib/core/dbsp/window.esk) | no | [dbsp_window.md](dbsp_window.md) | `make-dbsp-window` `dbsp-window-step` `dbsp-window-output` `dbsp-window-info` |
| [`core.dbsp.checkpoint`](../../../lib/core/dbsp/checkpoint.esk) | no | [dbsp_checkpoint.md](dbsp_checkpoint.md) | `make-dbsp-resumable` `dbsp-resumable-step` `dbsp-resumable-output` `dbsp-resumable-tick` `dbsp-resumable-checkpoint` `dbsp-resumable-restore` `dbsp-checkpoint-save` `dbsp-checkpoint-load` |
| [`core.dbsp.resident`](../../../lib/core/dbsp/resident.esk) | no | [dbsp_resident.md](dbsp_resident.md) | `call-with-dbsp-resident` `dbsp-resident-step` `dbsp-resident-release!` `dbsp-resident-output` `dbsp-resident-delta` `dbsp-resident-work` `dbsp-resident-stats` `dbsp-resident-compact!` |
| [`core.dbsp.tensor_circuit`](../../../lib/core/dbsp/tensor_circuit.esk) | no | [dbsp_tensor_circuit.md](dbsp_tensor_circuit.md) | `make-tensor-circuit` `tensor-circuit-step` `tensor-circuit-output` `tensor-circuit-ref` `tensor-circuit-work` `tensor-circuit-visit` `tensor-circuit-plan-work` |
| [`core.dbsp.tensor_snapshot`](../../../lib/core/dbsp/tensor_snapshot.esk) | no | [dbsp_tensor_snapshot.md](dbsp_tensor_snapshot.md) | `tensor-snapshot` `tensor-snapshot-shape` `tensor-snapshot-ref` `tensor-snapshot-apply` `tensor-snapshot->tensor` `tensor-snapshot->reader` `tensor-snapshot-work` |
| [`core.dbsp.recursive`](../../../lib/core/dbsp/recursive.esk) | no | [dbsp_recursive.md](dbsp_recursive.md) | `make-dbsp-recursive` `dbsp-recursive-step` `dbsp-recursive-output` `dbsp-recursive-work` |
| [`core.dbsp.tensor`](../../../lib/core/dbsp/tensor.esk) | no | [dbsp.md](dbsp.md#optional-sparse-tensor-bridge) | `tensor-delta` `tensor-delta?` `tensor-delta-shape` `tensor-delta-entries` `tensor-apply-delta` `zset-project-tensor` `tensor-matmul-delta` `tensor-reader` |
| [`core.distributed`](../../../lib/core/distributed.esk) | no | [distributed.md](distributed.md) | `lamport-zero` `lamport-tick` `lamport-merge` `lamport-recv` `lamport-before?` `vector-clock-empty` `vector-clock-ref` `vector-clock-set` `vector-clock-tick` `vector-clock-merge` `vector-clock-compare` `vector-clock-before?` `vector-clock-after?` `vector-clock-concurrent?` `vector-clock-equal?` `vector-clock-nodes` `make-g-counter` `g-counter-counts` `g-counter-inc` `g-counter-value` `g-counter-merge` `make-pn-counter` `pn-counter-positive` `pn-counter-negative` `pn-counter-inc` `pn-counter-dec` `pn-counter-value` `pn-counter-merge` `make-or-set` `or-set-adds` `or-set-removes` `or-set-clock` `or-set-add` `or-set-remove` `or-set-member?` `or-set-elements` `or-set-merge` `make-lww-register` `lww-register-value` `lww-register-timestamp` `lww-register-writer` `lww-register-deleted?` `lww-register-present?` `lww-register-set` `lww-register-delete` `lww-register-merge` `make-lww-map` `lww-map-entries` `lww-map-set` `lww-map-remove` `lww-map-ref` `lww-map-contains?` `lww-map-visible-entries` `lww-map-merge` `make-rga` `rga-root-id` `rga-entries` `rga-clock` `rga-entry-id` `rga-entry-prev` `rga-entry-value` `rga-entry-deleted?` `rga-insert-after` `rga-append` `rga-delete` `rga-values` `rga-last-id` `rga-merge` |
| [`core.dnc`](../../../lib/core/dnc.esk) | no | [dnc.md](dnc.md) | `make-dnc-memory` `dnc-content-address` `dnc-loc-address` `dnc-read` `dnc-write!` `dnc-alloc-weights` `dnc-read-grad` `dnc-memory?` |
| [`core.exact_linalg`](../../../lib/core/exact_linalg.esk) | yes | [exact_linalg.md](exact_linalg.md) | `exact-matrix?` `exact-matrix-ref` `exact-matrix-mul` `exact-matrix-transpose` `exact-det` `exact-solve` `exact-affine-solve` `exact-inverse` `exact-rank` `exact-nullspace` `torus-average` `torus-average-2d` |
| [`core.files`](../../../lib/core/files.esk) | yes | [files.md](files.md) | `path-directory` `with-atomic-output-file` `atomic-write-file` |
| [`core.exact_psd`](../../../lib/core/exact_psd.esk) | no | [exact_psd.md](exact_psd.md) | `exact-psd-factor` `psd-factor?` `psd-factor-rows` `psd-factor-weights` `psd-negative-direction` |
| [`core.functional.compose`](../../../lib/core/functional/compose.esk) | yes | [functional_compose.md](functional_compose.md) | `compose` `compose3` `identity` `constantly` |
| [`core.functional.curry`](../../../lib/core/functional/curry.esk) | yes | [functional_curry.md](functional_curry.md) | `curry2` `curry3` `uncurry2` `partial1` `partial2` `partial3` `partial` |
| [`core.functional.flip`](../../../lib/core/functional/flip.esk) | yes | [functional_flip.md](functional_flip.md) | `flip` |
| [`core.net`](../../../lib/core/net.esk) | no | [net.md](net.md) | `tcp-listen` `tcp-connect` `tcp-accept` `tcp-read` `tcp-write` `tcp-close` `udp-bind` `udp-sendto` `udp-recvfrom` `udp-close` `net-local-address` `net-peer-address` `net-local-port` `net-peer-port` `net-nonblocking!` `network-error?` `network-error-code` `network-error-operation` |
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
| [`core.memory_store`](../../../lib/core/memory_store.esk) | no | [memory_store.md](memory_store.md) | `make-memory-store` `memory-store?` `memory-store-log` `memory-store-path` `memory-store-open` `memory-store-open-fast` `memory-store-open-durable` `memory-store-open-fast-durable` `memory-store-close!` `memory-store-append!` `memory-store-verify` `memory-store-audit` `memory-store-audit-linear` `memory-store-linear-evidence` `memory-store-count` `memory-store-head` `memory-store-tail` `memory-store-sanitize` `memory-store-test-fail-next-first-file-parent-sync!` |
| [`core.merkle`](../../../lib/core/merkle.esk) | no | [merkle.md](merkle.md) | `fnv1a-64` `hash->hex` `merkle-leaf` `merkle-leaf?` `merkle-inode?` `merkle-root` `merkle-data` `merkle-tree` `merkle-tree-with-hash` `merkle-leaves` `merkle-proof` `merkle-verify` `make-cas` `make-cas-with-hash` `cas?` `cas-put!` `cas-get` `cas-has?` `cas-size` `cas-keys` |
| [`core.metrics`](../../../lib/core/metrics.esk) | no | [metrics.md](metrics.md) | `make-counter` `counter-inc!` `counter-add!` `make-gauge` `gauge-set!` `gauge-inc!` `gauge-dec!` `make-histogram` `histogram-observe!` `histogram-buckets` `metrics-register!` `metrics-render` `metrics-reset!` `metric-name` `metric-help` `metric-kind` |
| [`core.ml.gradient_estimators`](../../../lib/core/ml/gradient_estimators.esk) | no | [ml_gradient_estimators.md](ml_gradient_estimators.md) | `gumbel-softmax` `gumbel-softmax-det` `straight-through` `straight-through-round` `straight-through-onehot` `categorical-pick` `argmax-onehot` `softmax` `sample-gumbel-noise` |
| [`core.ml.neurosymbolic`](../../../lib/core/ml/neurosymbolic.esk) | no | [ml_neurosymbolic.md](ml_neurosymbolic.md) | `make-embedding-table` `emb-dim` `embed!` `vdot` `soft-unify` `soft-unify-loss` `soft-unify-train!` `kb-attention` `kb-retrieve` |
| [`core.msgpack`](../../../lib/core/msgpack.esk) | no | [msgpack.md](msgpack.md) | `msgpack-null` `msgpack-null?` `msgpack-map` `msgpack-map?` `msgpack-map-entries` `msgpack-encode` `msgpack-decode` `msgpack-decode-prefix` `msgpack-bytes->bytevector` `msgpack-bytevector->bytes` |
| [`core.numeric_extras`](../../../lib/core/numeric_extras.esk) | yes | [numeric_extras.md](numeric_extras.md) | `exact-integer-sqrt` |
| [`core.operators.arithmetic`](../../../lib/core/operators/arithmetic.esk) | yes | [operators_arithmetic.md](operators_arithmetic.md) | `add` `sub` `mul` `divide` |
| [`core.operators.compare`](../../../lib/core/operators/compare.esk) | yes | [operators_compare.md](operators_compare.md) | `lt` `gt` `le` `ge` `eq` |
| [`core.pde.ns-residual`](../../../lib/core/pde/ns-residual.esk) | yes | [ns-residual.md](ns-residual.md) | `ns-flow` `ns-flow-ur` `ns-flow-utheta` `ns-flow-uz` `ns-flow-p` `ns-flow-nu` `ns-cflow` `ns-cflow-u` `ns-cflow-p` `ns-cflow-nu` `ns-residual` `ns-residual-cartesian` `ns-divergence` `ns-divergence-cartesian` `ns-simpson-nodes` `ns-simpson-weights` `ns-simpson-step` `ns-grid-integral` `ns-energy` `ns-dissipation` `ns-similarity-field` `ns-residual-tau-series` `ns-singular-orders` `ns-force-smoothness-probe` |
| [`core.plot`](../../../lib/core/plot.esk) | yes | [plot.md](plot.md) | `sparkline` `bar-chart` `histogram` |
| [`core.reflection`](../../../lib/core/reflection.esk) | yes | [reflection.md](reflection.md) | `describe` `type-name` |
| [`core.sdnc`](../../../lib/core/sdnc.esk) | no | [sdnc.md](sdnc.md) | `sdnc-program` `sdnc-run` `sdnc-weight-grad` `sdnc-params` `sdnc-set-params!` `sdnc-improve!` `sdnc?` |
| [`core.sexp`](../../../lib/core/sexp.esk) | yes | [sexp.md](sexp.md) | `sexp->string` `sexp->canonical-string` |
| [`core.streams`](../../../lib/core/streams.esk) | yes | [streams.md](streams.md) | `stream-null` `stream-null?` `stream-pair?` `stream?` `stream-cons` `stream-car` `stream-cdr` `stream-take` `stream-drop` `stream-ref` `stream-map` `stream-filter` `stream-for-each` `stream-zip` `stream-append` `stream-iterate` `stream-from` `stream-take-while` `stream-drop-while` `stream-length` `stream->list` `list->stream` |
| [`core.strings`](../../../lib/core/strings.esk) | yes | [strings.md](strings.md) | `string-join` `string-trim` `string-trim-left` `string-trim-right` `string-replace` `string-reverse` `string-copy` `string-repeat` `string-starts-with?` `string-ends-with?` `string-starts-with` `string-ends-with` `string-index` `string-last-index` `string-contains` `string-contains?` `string-count` `string-find` `string-upcase` `string-downcase` `string-split-ordered` |
| [`core.symbolic`](../../../lib/core/symbolic.esk) | yes | [symbolic.md](symbolic.md) | `poly?` `poly` `poly-const` `poly-var` `poly+` `poly-` `poly*` `poly-expt` `poly-scale` `poly-eval` `poly-deriv` `poly-degree` `poly-coeff` `poly=?` `poly->string` `series?` `series` `series+` `series-` `series*` `series-compose` `series-deriv` `series-integrate` `series-inverse` `series-coeff` `series-truncate` `series->poly` `series-exp` `series-log` `series-sin` `series-cos` `series-sqrt` `series-lowest-order` `series-singular-part` `poly-derivative-of` `series-derivative-of` |
| [`core.testing`](../../../lib/core/testing.esk) | no | [testing.md](testing.md) | `register-test` `check-equal?` `check-true` `check-false` `check-approx` `assert-close` `certify-kernel` `check-raises` `run-tests` `reset-tests!` `*tests*` `*test-pass-count*` `*test-fail-count*` `*current-test-fails*` `*current-test-name*` |
| [`core.threads`](../../../lib/core/threads.esk) | no | [threads.md](threads.md) | `make-mutex` `mutex-lock!` `mutex-trylock!` `mutex-unlock!` `mutex-destroy!` `with-mutex` `make-condvar` `condvar-wait!` `condvar-signal!` `condvar-broadcast!` `condvar-destroy!` `make-thread` `thread-join` `thread?` `thread-result-ready?` |
| [`core.url`](../../../lib/core/url.esk) | yes | [url.md](url.md) | `url-encode` `url-decode` `base64url-encode` `base64url-decode` |
| [`math`](../../../lib/math.esk) | no | [math.md](math.md) | `mat-ref` `tensor-copy` `det` `inv` `solve` `cross` `dot` `normalize` `mat-vec-mul` `power-iteration` `integrate` `newton` `variance` `std` `covariance` |
| [`ml.activations`](../../../lib/ml/activations.esk) | no | [ml_activations.md](ml_activations.md) | `relu-scalar` `sigmoid-scalar` `tanh-scalar` `softplus-scalar` `silu` `swish` `mish` `normalize-minmax` `normalize-zscore` |
| [`ml.optimization`](../../../lib/ml/optimization.esk) | yes | [ml_optimization.md](ml_optimization.md) | `gradient-descent` `adam` `l-bfgs` `conjugate-gradient` `line-search` `tensor-dot` `tensor-norm` |
| [`signal.fft`](../../../lib/signal/fft.esk) | yes | [signal_fft.md](signal_fft.md) | `fft` `ifft` |
| [`signal.filters`](../../../lib/signal/filters.esk) | yes | [signal_filters.md](signal_filters.md) | `hamming-window` `hann-window` `blackman-window` `kaiser-window` `apply-window` `convolve` `fast-convolve` `fir-filter` `iir-filter` `butterworth-lowpass` `butterworth-highpass` `butterworth-bandpass` `frequency-response` |
| [`web.web`](../../../lib/web/web.esk) | no | [web.md](web.md) | `web-get-document` `web-get-window` `web-get-body` `web-create-element` `web-create-text-node` `web-get-element-by-id` `web-query-selector` `web-query-selector-all` `web-append-child` `web-remove-child` `web-insert-before` `web-replace-child` `web-clone-node` `web-get-parent` `web-get-first-child` `web-get-last-child` `web-get-next-sibling` `web-get-prev-sibling` `web-get-children-count` `web-get-child-at` `web-set-attribute` `web-get-attribute` `web-remove-attribute` `web-has-attribute` `web-set-inner-html` `web-get-inner-html` `web-set-text-content` `web-get-text-content` `web-add-class` `web-remove-class` `web-toggle-class` `web-has-class` `web-set-style` `web-get-style` `web-get-value` `web-set-value` `web-get-checked` `web-set-checked` `web-focus` `web-blur` `web-add-event-listener` `web-remove-event-listener` `web-event-prevent-default` `web-event-stop-propagation` `web-event-get-target` `web-event-get-key` `web-event-get-key-code` `web-event-get-mouse-x` `web-event-get-mouse-y` `web-set-timeout` `web-set-interval` `web-clear-timeout` `web-clear-interval` `web-request-animation-frame` `web-cancel-animation-frame` `web-console-log` `web-console-warn` `web-console-error` `web-alert` `web-confirm` `web-prompt` `web-get-window-width` `web-get-window-height` `web-get-scroll-x` `web-get-scroll-y` `web-scroll-to` `web-get-href` `web-set-href` `web-get-hash` `web-set-hash` `web-storage-get` `web-storage-set` `web-storage-remove` `web-storage-clear` `web-fetch` `web-get-context-2d` `web-canvas-fill-rect` `web-canvas-stroke-rect` `web-canvas-clear-rect` `web-canvas-fill-style` `web-canvas-stroke-style` `web-canvas-line-width` `web-canvas-begin-path` `web-canvas-close-path` `web-canvas-move-to` `web-canvas-line-to` `web-canvas-arc` `web-canvas-fill` `web-canvas-stroke` `web-canvas-fill-text` `web-canvas-font` `web-canvas-save` `web-canvas-restore` `web-canvas-translate` `web-canvas-rotate` `web-canvas-scale` `web-release-handle` |

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
| `core.certified_native_solve` | `solve-certified-native` | [solve-certified-native](shipped_exports.md#solve-certified-native) |
| `core.certified_native_solve` | `verify-certified-native-solve` | [verify-certified-native-solve](shipped_exports.md#verify-certified-native-solve) |
| `core.algebraic_batch` | `make-algebraic-multiplication-plan` | [make-algebraic-multiplication-plan](shipped_exports.md#make-algebraic-multiplication-plan) |
| `core.algebraic_batch` | `algebraic-batch-mul` | [algebraic-batch-mul](shipped_exports.md#algebraic-batch-mul) |
| `core.algebraic_domain` | `algebraic-domain` | [algebraic-domain](shipped_exports.md#algebraic-domain) |
| `core.algebraic_domain` | `algebraic-domain-ref` | [algebraic-domain-ref](shipped_exports.md#algebraic-domain-ref) |
| `core.algebraic_field` | `make-algebraic-field` | [make-algebraic-field](shipped_exports.md#make-algebraic-field) |
| `core.algebraic_field` | `algebraic-field-degree` | [algebraic-field-degree](shipped_exports.md#algebraic-field-degree) |
| `core.algebraic_field` | `algebraic-field-polynomial` | [algebraic-field-polynomial](shipped_exports.md#algebraic-field-polynomial) |
| `core.algebraic_field` | `algebraic-generator` | [algebraic-generator](shipped_exports.md#algebraic-generator) |
| `core.algebraic_field` | `algebraic-element` | [algebraic-element](shipped_exports.md#algebraic-element) |
| `core.algebraic_field` | `algebraic-coefficients` | [algebraic-coefficients](shipped_exports.md#algebraic-coefficients) |
| `core.algebraic_field` | `algebraic-add` | [algebraic-add](shipped_exports.md#algebraic-add) |
| `core.algebraic_field` | `algebraic-sub` | [algebraic-sub](shipped_exports.md#algebraic-sub) |
| `core.algebraic_field` | `algebraic-scale` | [algebraic-scale](shipped_exports.md#algebraic-scale) |
| `core.algebraic_field` | `algebraic-mul` | [algebraic-mul](shipped_exports.md#algebraic-mul) |
| `core.algebraic_field` | `algebraic-inverse` | [algebraic-inverse](shipped_exports.md#algebraic-inverse) |
| `core.algebraic_field` | `algebraic-zero?` | [algebraic-zero?](shipped_exports.md#algebraic-zero) |
| `core.algebraic_field` | `algebraic-equal?` | [algebraic-equal?](shipped_exports.md#algebraic-equal) |
| `core.algebraic_field` | `algebraic-rational` | [algebraic-rational](shipped_exports.md#algebraic-rational) |
| `core.algebraic_field` | `algebraic-enclosure` | [algebraic-enclosure](shipped_exports.md#algebraic-enclosure) |
| `core.algebraic_field` | `algebraic-sign` | [algebraic-sign](shipped_exports.md#algebraic-sign) |
| `core.algebraic_field` | `make-algebraic-embedding` | [make-algebraic-embedding](shipped_exports.md#make-algebraic-embedding) |
| `core.algebraic_field` | `make-algebraic-compositum` | [make-algebraic-compositum](shipped_exports.md#make-algebraic-compositum) |
| `core.algebraic_field` | `algebraic-field-certificate` | [algebraic-field-certificate](shipped_exports.md#algebraic-field-certificate) |
| `core.algebraic_field` | `make-multiquadratic-field` | [make-multiquadratic-field](shipped_exports.md#make-multiquadratic-field) |
| `core.algebraic_linalg` | `algebraic-factor` | [algebraic-factor](shipped_exports.md#algebraic-factor) |
| `core.algebraic_linalg` | `algebraic-factor-solve` | [algebraic-factor-solve](shipped_exports.md#algebraic-factor-solve) |
| `core.algebraic_linalg` | `algebraic-factor-solve-transpose` | [algebraic-factor-solve-transpose](shipped_exports.md#algebraic-factor-solve-transpose) |
| `core.algebraic_linalg` | `algebraic-factor-summary` | [algebraic-factor-summary](shipped_exports.md#algebraic-factor-summary) |
| `core.ad.incremental_reverse` | `make-incremental-pullback` | [`make-incremental-pullback`](shipped_exports.md#make-incremental-pullback) |
| `core.ad.incremental_reverse` | `incremental-pullback-step` | [`incremental-pullback-step`](shipped_exports.md#incremental-pullback-step) |
| `core.ad.incremental_reverse` | `incremental-pullback-output` | [`incremental-pullback-output`](shipped_exports.md#incremental-pullback-output) |
| `core.ad.incremental_reverse` | `incremental-pullback-work` | [`incremental-pullback-work`](shipped_exports.md#incremental-pullback-work) |
| `core.ad.incremental_reverse` | `incremental-pullback-info` | [`incremental-pullback-info`](shipped_exports.md#incremental-pullback-info) |
| `core.algebraic_linalg` | `algebraic-matrix-mul` | [algebraic-matrix-mul](shipped_exports.md#algebraic-matrix-mul) |
| `core.algebraic_linalg` | `algebraic-solve` | [algebraic-solve](shipped_exports.md#algebraic-solve) |
| `core.algebraic_update` | `algebraic-update-factor` | [algebraic-update-factor](shipped_exports.md#algebraic-update-factor) |
| `core.algebraic_update` | `algebraic-update` | [algebraic-update](shipped_exports.md#algebraic-update) |
| `core.algebraic_update` | `algebraic-update-symmetric` | [algebraic-update-symmetric](shipped_exports.md#algebraic-update-symmetric) |
| `core.algebraic_update` | `algebraic-update-solve` | [algebraic-update-solve](shipped_exports.md#algebraic-update-solve) |
| `core.algebraic_update` | `algebraic-update-solve-transpose` | [algebraic-update-solve-transpose](shipped_exports.md#algebraic-update-solve-transpose) |
| `core.algebraic_update` | `algebraic-update-pullback` | [algebraic-update-pullback](shipped_exports.md#algebraic-update-pullback) |
| `core.algebraic_update` | `algebraic-update-info` | [algebraic-update-info](shipped_exports.md#algebraic-update-info) |
| `core.polynomial_exact` | `exact-poly` | [exact-poly](shipped_exports.md#exact-poly) |
| `core.polynomial_exact` | `exact-poly-zero?` | [exact-poly-zero?](shipped_exports.md#exact-poly-zero) |
| `core.polynomial_exact` | `exact-poly-sub` | [exact-poly-sub](shipped_exports.md#exact-poly-sub) |
| `core.polynomial_exact` | `exact-poly-scale` | [exact-poly-scale](shipped_exports.md#exact-poly-scale) |
| `core.polynomial_exact` | `exact-poly-mul` | [exact-poly-mul](shipped_exports.md#exact-poly-mul) |
| `core.polynomial_exact` | `exact-poly-divmod` | [exact-poly-divmod](shipped_exports.md#exact-poly-divmod) |
| `core.polynomial_exact` | `exact-poly-inverse-mod` | [exact-poly-inverse-mod](shipped_exports.md#exact-poly-inverse-mod) |
| `core.polynomial_exact` | `exact-poly-eval` | [exact-poly-eval](shipped_exports.md#exact-poly-eval) |
| `core.polynomial_exact` | `exact-poly-real-root-count` | [exact-poly-real-root-count](shipped_exports.md#exact-poly-real-root-count) |
| `core.polynomial_exact` | `exact-poly-isolate` | [exact-poly-isolate](shipped_exports.md#exact-poly-isolate) |
| `core.ad.algebraic` | `make-algebraic-taylor` | [make-algebraic-taylor](shipped_exports.md#make-algebraic-taylor) |
| `core.ad.algebraic` | `make-algebraic-hyperdual` | [make-algebraic-hyperdual](shipped_exports.md#make-algebraic-hyperdual) |
| `core.ad.algebraic` | `algebraic-ad-value` | [algebraic-ad-value](shipped_exports.md#algebraic-ad-value) |
| `core.ad.algebraic` | `algebraic-ad-constant` | [algebraic-ad-constant](shipped_exports.md#algebraic-ad-constant) |
| `core.ad.algebraic` | `algebraic-ad-seed` | [algebraic-ad-seed](shipped_exports.md#algebraic-ad-seed) |
| `core.ad.algebraic` | `algebraic-ad-coefficients` | [algebraic-ad-coefficients](shipped_exports.md#algebraic-ad-coefficients) |
| `core.ad.algebraic` | `algebraic-ad-add` | [algebraic-ad-add](shipped_exports.md#algebraic-ad-add) |
| `core.ad.algebraic` | `algebraic-ad-sub` | [algebraic-ad-sub](shipped_exports.md#algebraic-ad-sub) |
| `core.ad.algebraic` | `algebraic-ad-mul` | [algebraic-ad-mul](shipped_exports.md#algebraic-ad-mul) |
| `core.ad.algebraic` | `algebraic-ad-inverse` | [algebraic-ad-inverse](shipped_exports.md#algebraic-ad-inverse) |
| `core.ad.algebraic` | `algebraic-ad-equal?` | [algebraic-ad-equal?](shipped_exports.md#algebraic-ad-equal) |
| `core.ad.algebraic` | `algebraic-ad-derivative` | [algebraic-ad-derivative](shipped_exports.md#algebraic-ad-derivative) |
| `core.ad.algebraic` | `algebraic-ad-root` | [algebraic-ad-root](shipped_exports.md#algebraic-ad-root) |
| `core.ad.algebraic` | `algebraic-ad-context` | [algebraic-ad-context](shipped_exports.md#algebraic-ad-context) |
| `core.ad.algebraic` | `call-with-algebraic-derivative` | [call-with-algebraic-derivative](shipped_exports.md#call-with-algebraic-derivative) |
| `core.ad.algebraic_implicit` | `algebraic-ad-system-root` | [algebraic-ad-system-root](shipped_exports.md#algebraic-ad-system-root) |
| `core.ad.algebraic_implicit` | `algebraic-ad-system-residual` | [algebraic-ad-system-residual](shipped_exports.md#algebraic-ad-system-residual) |
| `core.ad.algebraic_implicit` | `algebraic-system-linearization` | [algebraic-system-linearization](shipped_exports.md#algebraic-system-linearization) |
| `core.ad.algebraic_incremental` | `make-algebraic-incremental-plan` | [make-algebraic-incremental-plan](shipped_exports.md#make-algebraic-incremental-plan) |
| `core.ad.algebraic_incremental` | `algebraic-incremental-new` | [algebraic-incremental-new](shipped_exports.md#algebraic-incremental-new) |
| `core.ad.algebraic_incremental` | `algebraic-incremental-step` | [algebraic-incremental-step](shipped_exports.md#algebraic-incremental-step) |
| `core.ad.algebraic_incremental` | `algebraic-incremental-output` | [algebraic-incremental-output](shipped_exports.md#algebraic-incremental-output) |
| `core.ad.algebraic_incremental` | `algebraic-incremental-info` | [algebraic-incremental-info](shipped_exports.md#algebraic-incremental-info) |
| `core.ad.algebraic_pullback` | `algebraic-solve-with-pullback` | [algebraic-solve-with-pullback](shipped_exports.md#algebraic-solve-with-pullback) |
| `core.ad.algebraic_pullback` | `algebraic-factor-with-pullback` | [algebraic-factor-with-pullback](shipped_exports.md#algebraic-factor-with-pullback) |
| `core.ad.algebraic_pullback` | `algebraic-system-with-pullback` | [algebraic-system-with-pullback](shipped_exports.md#algebraic-system-with-pullback) |
| `core.ad.algebraic_pullback` | `algebraic-reverse-solve` | [algebraic-reverse-solve](shipped_exports.md#algebraic-reverse-solve) |
| `core.ad.algebraic_pullback` | `algebraic-reverse-system` | [algebraic-reverse-system](shipped_exports.md#algebraic-reverse-system) |
| `core.ad.algebraic_reverse` | `algebraic-value-and-pullback` | [algebraic-value-and-pullback](shipped_exports.md#algebraic-value-and-pullback) |
| `core.ad.algebraic_reverse` | `algebraic-reverse-value` | [algebraic-reverse-value](shipped_exports.md#algebraic-reverse-value) |
| `core.ad.algebraic_reverse` | `algebraic-reverse-constant` | [algebraic-reverse-constant](shipped_exports.md#algebraic-reverse-constant) |
| `core.ad.algebraic_reverse` | `algebraic-reverse-add` | [algebraic-reverse-add](shipped_exports.md#algebraic-reverse-add) |
| `core.ad.algebraic_reverse` | `algebraic-reverse-sub` | [algebraic-reverse-sub](shipped_exports.md#algebraic-reverse-sub) |
| `core.ad.algebraic_reverse` | `algebraic-reverse-mul` | [algebraic-reverse-mul](shipped_exports.md#algebraic-reverse-mul) |
| `core.ad.algebraic_reverse` | `algebraic-reverse-inverse` | [algebraic-reverse-inverse](shipped_exports.md#algebraic-reverse-inverse) |
| `core.ad.algebraic_reverse` | `algebraic-reverse-pow` | [algebraic-reverse-pow](shipped_exports.md#algebraic-reverse-pow) |
| `core.ad.algebraic_reverse` | `algebraic-hvp` | [algebraic-hvp](shipped_exports.md#algebraic-hvp) |
| `core.dbsp.circuit` | `dbsp-circuit-checkpoint-state` | [`dbsp-circuit-checkpoint-state`](shipped_exports.md#dbsp-circuit-checkpoint-state) |
| `core.dbsp.checkpoint` | `make-dbsp-resumable` | [`make-dbsp-resumable`](shipped_exports.md#make-dbsp-resumable) |
| `core.dbsp.checkpoint` | `dbsp-resumable-step` | [`dbsp-resumable-step`](shipped_exports.md#dbsp-resumable-step) |
| `core.dbsp.checkpoint` | `dbsp-resumable-output` | [`dbsp-resumable-output`](shipped_exports.md#dbsp-resumable-output) |
| `core.dbsp.checkpoint` | `dbsp-resumable-tick` | [`dbsp-resumable-tick`](shipped_exports.md#dbsp-resumable-tick) |
| `core.dbsp.checkpoint` | `dbsp-resumable-checkpoint` | [`dbsp-resumable-checkpoint`](shipped_exports.md#dbsp-resumable-checkpoint) |
| `core.dbsp.checkpoint` | `dbsp-resumable-restore` | [`dbsp-resumable-restore`](shipped_exports.md#dbsp-resumable-restore) |
| `core.dbsp.checkpoint` | `dbsp-checkpoint-save` | [`dbsp-checkpoint-save`](shipped_exports.md#dbsp-checkpoint-save) |
| `core.dbsp.checkpoint` | `dbsp-checkpoint-load` | [`dbsp-checkpoint-load`](shipped_exports.md#dbsp-checkpoint-load) |
| `core.dbsp.tensor_circuit` | `make-tensor-circuit` | [`make-tensor-circuit`](shipped_exports.md#make-tensor-circuit) |
| `core.dbsp.tensor_circuit` | `tensor-circuit-step` | [`tensor-circuit-step`](shipped_exports.md#tensor-circuit-step) |
| `core.dbsp.tensor_circuit` | `tensor-circuit-output` | [`tensor-circuit-output`](shipped_exports.md#tensor-circuit-output) |
| `core.dbsp.tensor_circuit` | `tensor-circuit-ref` | [`tensor-circuit-ref`](shipped_exports.md#tensor-circuit-ref) |
| `core.dbsp.tensor_circuit` | `tensor-circuit-work` | [`tensor-circuit-work`](shipped_exports.md#tensor-circuit-work) |
| `core.ad.algebraic_ridge` | `algebraic-ridge-new` | [`algebraic-ridge-new`](shipped_exports.md#algebraic-ridge-new) |
| `core.ad.algebraic_ridge` | `algebraic-ridge-step` | [`algebraic-ridge-step`](shipped_exports.md#algebraic-ridge-step) |
| `core.ad.algebraic_ridge` | `algebraic-ridge-output` | [`algebraic-ridge-output`](shipped_exports.md#algebraic-ridge-output) |
| `core.ad.algebraic_ridge` | `algebraic-ridge-info` | [`algebraic-ridge-info`](shipped_exports.md#algebraic-ridge-info) |
| `core.ad.algebraic_ridge` | `algebraic-ridge-pullback` | [`algebraic-ridge-pullback`](shipped_exports.md#algebraic-ridge-pullback) |
| `core.certified_solve` | `certify-linear-solve` | [`certify-linear-solve`](shipped_exports.md#certify-linear-solve) |
| `core.certified_solve` | `verify-linear-solve-certificate` | [`verify-linear-solve-certificate`](shipped_exports.md#verify-linear-solve-certificate) |
| `core.ad.algebraic_qp` | `algebraic-qp-solve` | [`algebraic-qp-solve`](shipped_exports.md#algebraic-qp-solve) |
| `core.ad.algebraic_qp` | `algebraic-qp-solution` | [`algebraic-qp-solution`](shipped_exports.md#algebraic-qp-solution) |
| `core.ad.algebraic_qp` | `algebraic-qp-info` | [`algebraic-qp-info`](shipped_exports.md#algebraic-qp-info) |
| `core.ad.algebraic_qp` | `algebraic-qp-verify` | [`algebraic-qp-verify`](shipped_exports.md#algebraic-qp-verify) |
| `core.ad.algebraic_qp` | `algebraic-qp-pullback` | [`algebraic-qp-pullback`](shipped_exports.md#algebraic-qp-pullback) |
| `core.ad.derived_operator` | `make-derived-operator` | [`make-derived-operator`](shipped_exports.md#make-derived-operator) |
| `core.ad.derived_operator` | `derived-operator-value` | [`derived-operator-value`](shipped_exports.md#derived-operator-value) |
| `core.ad.derived_operator` | `derived-operator-jvp` | [`derived-operator-jvp`](shipped_exports.md#derived-operator-jvp) |
| `core.ad.derived_operator` | `derived-operator-vjp` | [`derived-operator-vjp`](shipped_exports.md#derived-operator-vjp) |
| `core.ad.derived_operator` | `derived-operator-delta` | [`derived-operator-delta`](shipped_exports.md#derived-operator-delta) |
| `core.ad.derived_operator` | `derived-operator-info` | [`derived-operator-info`](shipped_exports.md#derived-operator-info) |
| `core.ad.derived_operator` | `derived-operator-enclosure` | [`derived-operator-enclosure`](shipped_exports.md#derived-operator-enclosure) |
| `core.certified_nonlinear` | `make-certified-polynomial-system` | [`make-certified-polynomial-system`](shipped_exports.md#make-certified-polynomial-system) |
| `core.certified_nonlinear` | `certify-polynomial-root` | [`certify-polynomial-root`](shipped_exports.md#certify-polynomial-root) |
| `core.certified_nonlinear` | `verify-polynomial-root-certificate` | [`verify-polynomial-root-certificate`](shipped_exports.md#verify-polynomial-root-certificate) |
| `core.ad.root_branch` | `make-root-branch` | [`make-root-branch`](shipped_exports.md#make-root-branch) |
| `core.ad.root_branch` | `root-branch-step` | [`root-branch-step`](shipped_exports.md#root-branch-step) |
| `core.ad.root_branch` | `root-branch-enclosure` | [`root-branch-enclosure`](shipped_exports.md#root-branch-enclosure) |
| `core.ad.root_branch` | `root-branch-time` | [`root-branch-time`](shipped_exports.md#root-branch-time) |
| `core.ad.root_branch` | `root-branch-sensitivity` | [`root-branch-sensitivity`](shipped_exports.md#root-branch-sensitivity) |
| `core.ad.root_branch` | `verify-root-branch-step` | [`verify-root-branch-step`](shipped_exports.md#verify-root-branch-step) |
| `core.ad.algebraic_spectral` | `algebraic-characteristic` | [`algebraic-characteristic`](shipped_exports.md#algebraic-characteristic) |
| `core.ad.algebraic_spectral` | `algebraic-eigenvalue-with-pullback` | [`algebraic-eigenvalue-with-pullback`](shipped_exports.md#algebraic-eigenvalue-with-pullback) |
| `core.ad.algebraic_spectral` | `algebraic-eigenvalue-lift` | [`algebraic-eigenvalue-lift`](shipped_exports.md#algebraic-eigenvalue-lift) |
| `core.algebraic_linalg` | `algebraic-factor-positive-definite?` | [`algebraic-factor-positive-definite?`](shipped_exports.md#algebraic-factor-positive-definite) |
| `core.ad.algebraic_incremental` | `algebraic-incremental-moment-ref` | [`algebraic-incremental-moment-ref`](shipped_exports.md#algebraic-incremental-moment-ref) |
| `core.ad.algebraic_sparse` | `algebraic-sparse-hessian` | [algebraic-sparse-hessian](shipped_exports.md#algebraic-sparse-hessian) |
| `core.ad.algebraic_sparse` | `algebraic-sparse-hessian-dense` | [algebraic-sparse-hessian-dense](shipped_exports.md#algebraic-sparse-hessian-dense) |
| `core.ad.algebraic_sparse` | `algebraic-sparse-hessian-ref` | [algebraic-sparse-hessian-ref](shipped_exports.md#algebraic-sparse-hessian-ref) |
| `core.ad.algebraic_sparse` | `algebraic-sparse-hessian-summary` | [algebraic-sparse-hessian-summary](shipped_exports.md#algebraic-sparse-hessian-summary) |
| `core.ad.sparsity` | `hessian-color-graph` | [hessian-color-graph](shipped_exports.md#hessian-color-graph) |
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

### Modules documented outside this directory

These shipped modules have no per-module page in this directory; their exports
are documented, with signatures, on the pages listed.

| Module | Require | Documented in |
|---|---|---|
| `math.constants` | `(require math.constants)` | [MATH_STDLIB.md §1](../../breakdown/MATH_STDLIB.md#1-mathematical-constants) |
| `math.special` | `(require math.special)` | [MATH_STDLIB.md §2](../../breakdown/MATH_STDLIB.md#2-special-functions) |
| `math.ode` | `(require math.ode)` | [MATH_STDLIB.md §3](../../breakdown/MATH_STDLIB.md#3-ode-solvers) |
| `math.statistics` | `(require math.statistics)` | [MATH_STDLIB.md §4](../../breakdown/MATH_STDLIB.md#4-statistics) |
| `random.random` | `(require random.random)` | [QUANTUM_RNG.md](../../breakdown/QUANTUM_RNG.md) |
| `tensor.utils` | `(require tensor.utils)` | [API_REFERENCE.md, Tensor Utilities](../../API_REFERENCE.md#tensor-utilities) |
| `tensorcore` | `(require tensorcore)` | [shipped_exports.md](shipped_exports.md#tensorcore) |
| `core.blc` | `(require core.blc)` | [blc.md](blc.md) (reference); [BINARY_LAMBDA_CALCULUS.md](../../guide/BINARY_LAMBDA_CALCULUS.md) (guide) |
| `core.ad.guw` | `(require core.ad.guw)` | [AUTOMATIC_DIFFERENTIATION.md §4](../../guide/AUTOMATIC_DIFFERENTIATION.md#4-multivariate-mixed-partials-guw) |
| `core.ad.tensor_tower` | `(require core.ad.tensor_tower)` | [AUTOMATIC_DIFFERENTIATION.md §5](../../guide/AUTOMATIC_DIFFERENTIATION.md#5-tensor-ad--towers-of-tensors) |
| `core.ad.sparse_guw` | `(require core.ad.sparse_guw)` | [AUTOMATIC_DIFFERENTIATION.md §7](../../guide/AUTOMATIC_DIFFERENTIATION.md#7-sparse-high-order--sparse-hessians) |
| `core.ad.algebraic_sparse` | `(require core.ad.algebraic_sparse)` | [ad_algebraic_sparse.md](ad_algebraic_sparse.md) |
| `core.ad.sparsity` | `(require core.ad.sparsity)` | [ad_sparsity.md](ad_sparsity.md) |
| `core.ad.checkpoint` | `(require core.ad.checkpoint)` | [AUTOMATIC_DIFFERENTIATION.md §8](../../guide/AUTOMATIC_DIFFERENTIATION.md#8-reverse-mode--checkpointing) |
| `core.ad.taylor_numerics` | `(require core.ad.taylor_numerics)` | [AUTOMATIC_DIFFERENTIATION.md §10](../../guide/AUTOMATIC_DIFFERENTIATION.md#10-tower-numerics--odes-roots-inversion) |
| `core.ad.interval`, `core.ad.rigorous_interval`, `core.ad.taylor_models`, `core.ad.rigorous_taylor_models` | `(require core.ad.taylor_models)` | [certified-enclosures.md](certified-enclosures.md) |
| `agent.*` | `(require agent.<name>)` | [../agent/INDEX.md](../agent/INDEX.md) |

The modules under `lib/test/modules/` are module-system test fixtures, not user API.

| `core.certified_solve` | `(require core.certified_solve)` | [certified_solve.md](certified_solve.md) |

### Numeric TCP/UDP exports

- [`tcp-listen`](shipped_exports.md#tcp-listen)
- [`tcp-connect`](shipped_exports.md#tcp-connect)
- [`tcp-accept`](shipped_exports.md#tcp-accept)
- [`tcp-read`](shipped_exports.md#tcp-read)
- [`tcp-write`](shipped_exports.md#tcp-write)
- [`tcp-close`](shipped_exports.md#tcp-close)
- [`udp-bind`](shipped_exports.md#udp-bind)
- [`udp-sendto`](shipped_exports.md#udp-sendto)
- [`udp-recvfrom`](shipped_exports.md#udp-recvfrom)
- [`udp-close`](shipped_exports.md#udp-close)
- [`net-local-address`](shipped_exports.md#net-local-address)
- [`net-peer-address`](shipped_exports.md#net-peer-address)
- [`net-local-port`](shipped_exports.md#net-local-port)
- [`net-peer-port`](shipped_exports.md#net-peer-port)
- [`net-nonblocking!`](shipped_exports.md#net-nonblocking)
- [`network-error?`](shipped_exports.md#network-error)
- [`network-error-code`](shipped_exports.md#network-error-code)
- [`network-error-operation`](shipped_exports.md#network-error-operation)

### Resident circuit exports

- [`call-with-dbsp-resident`](shipped_exports.md#call-with-dbsp-resident)
- [`dbsp-resident-step`](shipped_exports.md#dbsp-resident-step)
- [`dbsp-resident-release!`](shipped_exports.md#dbsp-resident-release)
- [`dbsp-resident-output`](shipped_exports.md#dbsp-resident-output)
- [`dbsp-resident-delta`](shipped_exports.md#dbsp-resident-delta)
- [`dbsp-resident-work`](shipped_exports.md#dbsp-resident-work)
- [`dbsp-resident-stats`](shipped_exports.md#dbsp-resident-stats)
- [`dbsp-resident-compact!`](shipped_exports.md#dbsp-resident-compact)
