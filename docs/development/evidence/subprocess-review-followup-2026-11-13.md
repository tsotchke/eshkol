# Subprocess review followup — 2026-11-13

Tested PROCESS source `d1814038877bc4fc416f588041382a34c59a83ba`, including shared foundation `b682c40c4fbb4bb0944226ff2d49b0928c1fa785` and integration `53220efc1638c77d108ab009d4a5e6939c488c7b`.

The final source removes one redundant PROCESS disjunct in `runtime_handle.cpp`, adopting the exact qualified shared/core file. The other 528 recorded native source hashes remain unchanged. The results below were rerun on this corrected source.

Four fresh scoped Release CTests passed: native PROCESS streaming, watch-removal failure recovery, explicit-loop resource behavior and strict contract pins. This own-branch build used agent FFI OFF. Its standalone native suite passed normally and with ASan, UBSan and leak detection, including missing-checker, clock-failure, watch-removal retry and cross-kind ownership controls.

All 40 `agent.subprocess` exports have reviewed contracts and a matching source pin. Scoped callbacks borrow a PROCESS_STREAM owner; cleanup IDs inside aggregate results remain explicitly unsupported in strict mode. The adapter retains descriptors, its loop and its lease after watch-removal failure, and only retires its own PROCESS ownership.

Separate public CLI qualification used the combined production source `8d99676f6467e362da34122bbe7f36d7e6e63907` with agent FFI ON and the real force-linked production archive. The command supplied no `--support-library`. It passed 23 public checks in each of JIT and AOT, lossless 17,825,829-byte stdout and stderr transfers, reentry controls and explicit standalone-VM bridge rejection. This is combined-source CLI evidence, not a new own-PROCESS compiler or staged-package qualification.

Separate [hosted native qualification](https://github.com/Gabriel-Kahen/eshkol/actions/runs/43493801710) passed Linux epoll, Linux ASan/UBSan and macOS arm64 kqueue at that same combined source. macOS public JIT/AOT and package/release acceptance are not claimed. The [manifest](subprocess-review-followup-2026-11-13.json) records provenance and original-byte evidence hashes; raw full-suite failures remain separately recorded.

The ABI inventory separately registers the binding’s one sanctioned bytevector type guard in the lexical macro class and three measured Clang classes. Only exact semantic-count disappearance exceptions are recorded for configurations omitting the agent FFI translation unit. Paired complete Clang 22 scans and 28 report-level checker controls cover this registration and the separate Unix guards; they do not establish a clean repository inventory. The combined scan retains the same 157 unrelated violations as its control. Details are in the [socket followup](https://github.com/Gabriel-Kahen/eshkol/blob/gabe/unix-socket-core/docs/development/evidence/unix-socket-review-followup-2026-11-13.md) on the socket branch.
