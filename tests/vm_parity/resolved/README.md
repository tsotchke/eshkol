# Reclassified VM parity reproducers

These programs were formerly filed under `tests/vm_parity/found/`. The
execution-backed parity gate reran each on native LLVM and the standalone VM;
their normalized outputs now agree, so retaining them as active divergence
claims would be false. Their source comments preserve the original report and
the measured expected values.

The reclassified set is:

- `builtin_shadow_ignored_by_opcode_dispatch.esk`
- `consecutive_do_state_leak.esk`
- `define_after_do_corrupted.esk`
- `do_composition_broken.esk`
- `dynamic_wind_after_twice.esk`
- `equal_eq_structural_false.esk`
- `exact_division_lost.esk`
- `expt_bignum_to_float.esk`
- `float_display_1e10.esk`
- `force_returns_promise.esk`
- `iota_returns_empty.esk`
- `macro_set_top_level.esk`
- `map_two_lists_eskb_route.esk`
- `modulo_inexact_collapsed_vm.esk`
- `recursive_macro_zero.esk`
- `splice_middle_order.esk`
- `symbol_string_unhandled_fid.esk`
- `tensor_nested_collection_native.esk`
- `tensor_ref_component_oob_native.esk`
- `tensor_set_oob_silent_native.esk`
- `write_does_not_quote.esk`
- `weh_handler_return_swallows_condition.esk`
