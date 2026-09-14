# qLLM bridge capability and ownership

`ESHKOL_QLLM_ENABLED=ON` selects the actual `semiclassical_qllm/eshkol_bridge.h`
and shared library through `ESHKOL_QLLM_ROOT`. CMake links a probe that calls the
real tensor allocator, native registrar, eval entry point, and typed ESKB encoder.
Missing definitions reject configuration. The compiler build also checks the
linked qLLM LLVM dependency against the host LLVM major; an unprovable or different
version rejects configuration with a diagnostic. The standalone ABI test links
only qLLM's JIT dependency and therefore has no second host LLVM to conflict with.

`lib/bridge/qllm_bridge.cpp` implements Eshkol's AD operations on Eshkol-owned
buffers. `lib/bridge/qllm_interop.cpp` is the external capability boundary.
An OFF build reports `eshkol_qllm_bridge_available() == false`, returns no tensor,
and cannot initialize native registration. It never allocates a substitute
`qllm_tensor_t`. ON builds include qLLM's actual type and call its allocator and
accessors. There is no dynamic library selection or duplicated tensor layout.

Call `eshkol_qllm_tensor_destroy` on returned tensors, including after bridge
shutdown. Passing them to `free()` is invalid. `qllm_to_eshkol_tensor` accepts
contiguous CPU float32 dense tensors. Its required `out_size` argument is the
buffer capacity on entry and the required element count on return; insufficient
capacity returns false without writing output. Integer, device, and strided
representations are rejected rather than reinterpreted as float32.

`eshkol_qllm_bridge_init(NULL)` invokes the real
`qllm_eshkol_register_qllm_natives`, then evaluates a native tensor allocation and
destruction through qLLM's eval bridge. This checks tagged-value calls as well as
registration. The upstream registrar's success alone is insufficient: upstream
also returns success when no Eshkol host exists. Readiness becomes true only
after native execution returns the expected result. Shutdown clears session
readiness; the directly linked library and its native callbacks remain resident.
The old path argument accepts only NULL or an empty string; select the library
at configure time.

`eshkol_qllm_tensor_to_eskb` takes the actual `qllm_eshkol_const_entry_t` entries
and invokes `qllm_eshkol_tensor_to_eskb_chunk_typed`. Instructions must come from
qLLM's bit-packing channel, not the numeric double-to-float conversion.
The wrapper checks the resulting ESKB version before exposing bytes to Eshkol.
At qLLM source revision `e444e1c09ee10251d4c6fe469890c917a385f84c`, the encoder emits
ESKB 1; this candidate requires ESKB 2. It therefore returns
`QLLM_ERROR_INVALID_STATE_CODE`, sets the output to NULL/0, and records
`qLLM emits ESKB version 1; this Eshkol requires version 2` in qLLM's error state.
It does not rewrite version bytes. Upgrading the private encoder is required
before this path can supply executable chunks to the current Eshkol VM.

## Verification

```
python3 scripts/run_qllm_private_abi_gate.py --qllm-root /path/to/semiclassical_qllm
```

This configures OFF and ON builds, links the actual library, checks cross-library
tensor ownership and matmul, inspects mixed INT64/F64/STRING wire values, validates
the ESKB version boundary, and executes native registration without mocks.
It kills dtype, output-capacity, ESKB-version, and missing-registration mutants,
and confirms that omitting qLLM fails linking. CTest registers
`qllm_private_abi` and, for ON builds, `qllm_private_native_registration`.

Recorded mutation/link results: [qLLM ABI evidence](../reports/qllm-private-abi-evidence.json).
The candidate runtime objects also passed the existing AD bridge gradcheck
(10 checks), including enabled native initialization, when linked with the
candidate base runtime archive and the actual qLLM library using LLVM 21.1.8.
