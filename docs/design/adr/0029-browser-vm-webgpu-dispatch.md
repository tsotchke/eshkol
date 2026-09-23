# ADR-0029: WebGPU dispatch for the browser VM through a JSPI boundary

- **Status:** Accepted
- **Date:** 2026-09-22
- **Decision owners:** Eshkol GPU and WASM runtime
- **Applies to:** the browser bytecode VM (`site/static/eshkol-vm.{js,wasm}`,
  built by `scripts/build-wasm-repl.sh`), the WASM execute-and-diff module,
  `lib/backend/gpu/gpu_memory_webgpu.cpp`, and `web/eshkol-webgpu.js`
- **Relates to:** the compiled-WASM WebGPU dispatch (`eshkol_matmul_dispatch` and
  `eshkol_gpu_*` served by `web/eshkol-webgpu.js`), ADR-0003 (codegen/VM parity)

## Context

The VM's tensor natives already call the ordinary GPU seam. `vm_native.c` tries
`vm_gpu_try_matmul`, `vm_gpu_try_binary`, `vm_gpu_try_reduce`,
`vm_gpu_try_softmax` and `vm_gpu_try_transpose` (`vm_gpu_dispatch.h`), which call
`eshkol_gpu_*`: Metal or CUDA natively. The WASM build compiled that header
without `ESHKOL_GPU_ENABLED`, so in the browser every attempt returned NULL and
the VM never reached a GPU. Two further problems stood in the way:

1. `vm_gpu_dispatch.h` kept its own copy of the dispatch threshold instead of
   asking the backend's `eshkol_gpu_should_use()`, the selector every other
   caller uses.
2. WebGPU readback (`GPUBuffer.mapAsync`) is asynchronous, but the VM is
   synchronous C. Something has to let the evaluation wait for the GPU.

## Decision

**One seam.** The WASM VM links `gpu_memory_webgpu.cpp` and compiles with
`ESHKOL_GPU_ENABLED`, like any other GPU build. `vm_gpu_should_dispatch()`
delegates to `eshkol_gpu_should_use()`, so the VM uses the backend's threshold
and admission rules, and matmul uses the same size measure as
`eshkol_matmul_dispatch` (output elements). In the browser the backend is the
`EshkolWebGPU` object that the compiled-WASM loaders use, with the same sf64
kernels, tiers and policy. `gpuServe()` in `web/eshkol-webgpu.js` is the one
place that decides GPU or CPU, verifies the execution marker, and counts and
explains fallbacks, for both paths.

**The async boundary is JSPI at the module boundary.** `gpu_memory_webgpu.cpp`
declares its three compute bridges (`eshkol_webgpu_js_matmul`,
`_elementwise`, `_reduce`) as ordinary synchronous `EM_JS` imports that answer
"no device". `EshkolWebGPU.attachVm(moduleArg, backend)` installs an Emscripten
`instantiateWasm` hook that:

- replaces those three imports with `WebAssembly.Suspending` wrappers of the
  bridge;
- wraps the entry exports that run Eshkol code (`repl_eval`, `run_program`)
  with `WebAssembly.promising`.

It does this only when the browser has JSPI and a WebGPU device. This is the
same mechanism the compiled-WASM loaders use (`makeImports` and
`promisingExports`), so the browser has one suspension mechanism.

JSPI cannot suspend across a JavaScript frame. The VM's error recovery uses
`setjmp`/`longjmp`, which Emscripten lowers by default to `invoke_*`
trampolines that go through JavaScript, so a GPU readback inside any guarded
evaluation trapped ("trying to suspend JS frames"). Both WASM VM builds
therefore compile with native wasm exception handling (`-fwasm-exceptions
-sSUPPORT_LONGJMP=wasm`, in `ESHKOL_WASM_VM_FLAGS` in
`scripts/lib/wasm_vm_sources.sh`), which keeps the whole evaluation in wasm
frames.

While an evaluation is suspended it owns the VM's shadow stack. Every call into
a VM that may suspend is therefore serialised per module (`EshkolWebGPU.vmSerial`
and `vmCall`). The site routes its REPL, its runnable code blocks and the
`web_repl_eval` import through that queue.

**Fallback is explicit.** Without a device, `attachVm` sets
`moduleArg.eshkolWebGPUStatus = {ok: false, reason}` and changes nothing else.
Without JSPI it does the same. The C side then reports `ESHKOL_GPU_NONE` and the
VM runs its CPU path, producing the same output. With a device, an operation
that the dispatch selects but that has no WebGPU kernel (softmax, transpose,
axis reduction, normalisation, and the transcendental elementwise ops) is counted
in `fallbackCount` and explained in `diagnostics`. A WebGPU validation error is
rethrown, never masked by the CPU.

## Alternatives considered

**Asyncify.** Rejected. It instruments every function that can be on the stack
at a suspension point. For an interpreter that is essentially the whole VM, so
every evaluation pays in code size and speed, including the vast majority that
never touch the GPU. It would also be a second suspension mechanism beside the
JSPI one the compiled-WASM path already uses. A restricted `ASYNCIFY_ONLY` list
would have to name every frame between `repl_eval` and the bridge, including
indirect calls through the native dispatch table: a list that silently breaks
when the VM changes.

**A staged submit/await boundary at the VM's native-call seam.** Rejected. The
VM would return to JavaScript with a pending GPU operation and later resume the
native call. Natives re-enter the interpreter on the C stack (a closure passed
to `map`, a guard body, a tensor callback), so resuming would need a VM that can
suspend and continue from any native frame: in effect, the interpreter rewritten
as a resumable state machine. JSPI switches stacks in the engine for the
same effect, without touching the VM.

## Consequences

- A browser user of the VM gets the same WebGPU kernels as compiled programs:
  matmul, elementwise add/sub/mul/div, and sum/mean/max/min reductions.
  Matmul and elementwise results are bit-identical to the CPU path.
- `vmCall`, the promising entry exports and the evaluation queue make VM calls
  asynchronous when attached. Pages that call `repl_eval` synchronously
  through `cwrap` keep working unattached (for example the Node checkers),
  because attaching is opt-in.
- The browser VM requires wasm exception handling, which is supported by
  every browser that supports WebAssembly exceptions (Chrome 95+, Firefox 100+,
  Safari 15.2+).
- The GPU seam adds about 3 KB to `eshkol-vm.wasm`; `lib/core/logger.cpp`, which
  the backend reports allocation failures through, adds about 10 KB.
- The VM's GPU elementwise path now requires identical operand shapes, not only
  equal element counts. Before this change, `[6] + [1,6]` would have produced a
  result with the wrong shape on any GPU backend.

## Verification

`tests/webgpu/webgpu_vm_test.mjs` loads the shipped bundle in Chrome and runs
`tests/webgpu/vm_tensor_ops.esk` attached and unattached. It requires:

- bit-identical matmul and elementwise output;
- reductions within `GPU_GATE_TOL`;
- exactly ten GPU dispatches;
- softmax reported as an explicit fallback;
- a guard that catches a raise after a suspended GPU call;
- an explicit reason and CPU-identical output with `navigator.gpu` or
  `WebAssembly.Suspending` hidden.

`--corrupt` serves a broken ADD kernel and must fail.
