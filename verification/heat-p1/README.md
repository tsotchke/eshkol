# P1 — Native grid accesses in a heat step

Can the compiled program return every expected temperature while reading an overwritten cell? This experiment uses the [P0 heat kernel](../heat-p0/README.md) and an all-zero field to make the output identical under both schedules. It observes the addresses that the Eshkol executable reads and writes during one complete call to `heat-step`.

## Replay

On Linux x86-64, use Python 3, GNU objdump, Valgrind Lackey, Lean 4.35.0 and the native Eshkol v1.3.6-evolve compiler. From this directory run

```sh
python3 native_access.py --output receipt.json
```

`LEAN` and `ESHKOL_RUN` can identify the two executables. Allow about a gigabyte of temporary disk for the raw instruction and memory trace of one process. The replay removes that trace after each case. The [recorded receipt](receipt.json) keeps the hashes, exact fields, access sequence and executable-relative instruction addresses.

## What the observer measures

1. P0 decodes the emitted kernel and runs its Lean admission gate. The generator supplies a complete Eshkol program for one step. The replay executes the compiled ELF directly and compares all output cells to the exact rational FTCS reference. It then executes the same ELF under Lackey and compares the printed field again.
2. The observer obtains the ELF load bias from the tracer. It identifies the driver allocation, kernel entry and return, and vector allocator header-store instruction from the binary. The address written by the allocator establishes each grid's physical location independently of the stencil reads. This profile uses an eight-byte object header, an eight-byte vector length and sixteen bytes per tagged cell.
3. Between entry and return it inspects every traced data access overlapping an identified grid's cells, including accesses from callees. It accounts for fresh-vector initialization, then reconstructs complete cell reads and writes from the accessed bytes. The admitted profile requires the three old neighbor reads in order, followed by an interior store. The checker tracks the cells each store overwrote. Missing bytes, unexpected operations or an unsupported allocator layout prevent acceptance.
4. In the negative control the accepted source remains fixed. The replay compiles a second candidate that aliases the output vector to the input. The independent observer checks this binary against exactly the same access requirement. It records the first store to the wrong bank and the first read that consumes the new version.

The positive cases use the dyadic samples from P0's 20-interval sine profile and an all-zero four-interval grid. The alias control uses the same zero grid. All three binaries must return the exact one-step target field in direct and instrumented execution. Thus the rejection requires information from the memory trace.

## Evidence boundary

P0 proves the rational stencil relation for all input fields of an admitted decoded schedule. P1 validates the order, coordinates and physical versions of grid accesses on the recorded native executions. The receipt binds the programs, ELF files, verifier sources and tool versions. Its program counters identify the x86-64 instructions in those ELF files. Addresses in the trace are guest memory references reported by Valgrind, and the observer trusts Lackey and the tagged-vector ABI described above.

The observation spans one time step. The full-time sine error certificate belongs to P0's separate exact profile replay. Extending native equivalence to all inputs, multiple allocation lifetimes, other lowerings or a floating-point error budget requires additional evidence.
