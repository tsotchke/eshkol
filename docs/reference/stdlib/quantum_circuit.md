---
kind: reference
status: current
owner-area: quantum
since: v1.4.0
sources:
  - lib/core/quantum/circuit.esk
  - tests/quantum/circuit_export_test.esk
  - tests/quantum/circuit_export_driver.py
  - cmake/QuantumCircuitTests.cmake
---
# `core.quantum.circuit` — Static circuits and OpenQASM export

`core.quantum.circuit` constructs and validates a small circuit description,
exports it as an OpenQASM 3.0 program, and reports logical operation counts.
It is an opt-in source module with no Moonlab or quantum hardware dependency.
It does not execute circuits, submit provider jobs, simulate noise, or provide
a general OpenQASM parser. The reference describes the source tree; release
packages require the module to be included before these calls are available.

```scheme
(require core.quantum.circuit)
(define bell (make-quantum-circuit 2 2
  '((h 0) (cx 0 1) (measure 0 0) (measure 1 1))))
(display (quantum-circuit->openqasm3 bell))
```

The complete runnable source is
[`examples/quantum/bell_openqasm3.esk`](../../../examples/quantum/bell_openqasm3.esk).
From a repository checkout with a working native compiler:

```sh
eshkol-run --no-stdlib -L ./lib examples/quantum/bell_openqasm3.esk -o bell-export
./bell-export
```

Use `bell-export.exe` on Windows. The exporter prints:

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
reset q;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
```

The leading reset explicitly prepares every qubit in the all-zero state.
Every export includes it, including an empty circuit. The circuit can request
additional resets later. The exporter uses the
[OpenQASM standard gate library](https://openqasm.com/language/standard_library.html)
and emits a defined subset of the language; acceptance by one importer does
not establish compatibility with every OpenQASM backend.

## API and validation

| Procedure | Result |
|---|---|
| `(make-quantum-circuit qubits bits operations)` | Validated circuit value; copies the outer operation list and each operation's list structure |
| `(quantum-circuit->openqasm3 circuit)` | OpenQASM 3.0 text, with explicit zero-state initialization |
| `(quantum-circuit-resources circuit)` | Association list of logical register and operation counts |

`qubits` must be a positive exact integer; `bits` a nonnegative exact integer.
Wire and classical destinations are zero-based exact integers within their
register bounds. Each operation is a nonempty acyclic proper list whose first item is
a supported symbol. Unknown symbols, string gate names, wrong arities,
out-of-range operands and repeated qubits within one operation raise an error.
Cyclic outer lists and cyclic operation lists are detected before traversal.
A measurement may overwrite a classical destination used by an earlier
measurement. An empty operation list is valid. With zero classical bits the
export omits the `bit` declaration and rejects measurements.

Angles are in radians. They must be finite real numbers whose binary64
conversion is also finite. Exact rationals are accepted, then converted to a
binary64 decimal literal for export; the resulting circuit does not preserve
arbitrary-precision rational angles. Symbolic angles and complex values are
rejected. Register sizes have no backend-specific resource limit in this
module; a consuming backend can impose tighter limits.

The value currently uses a tagged vector and contains mutable lists. Treat
that representation as private. Both export and resource accounting validate
the value again, so later invalid mutations raise an error instead of emitting
unchecked text. Mutating the original input list does not alter the circuit.

## Supported operations

Gate operand order follows OpenQASM: controlled gates list controls first,
then targets. A rotation's angle precedes its qubit operands in the Eshkol
list, for example `(crx 0.5 2 0)`.

| Gate symbols | Angle parameters | Qubit operands |
|---|---:|---:|
| `id x y z h s sdg t tdg sx` | 0 | 1 |
| `p rx ry rz` | 1 | 1 |
| `cx cy cz ch swap` | 0 | 2 |
| `cp crx cry crz` | 1 | 2 |
| `ccx cswap` | 0 | 3 |

These are 25 gate kinds. The other supported operations are:

| Operation | Meaning |
|---|---|
| `(measure qubit bit)` | Computational-basis measurement into the specified classical bit |
| `(reset qubit)` | Reset the specified qubit |
| `(barrier qubit ...)` | Barrier over one or more distinct, explicit qubits |

There are no custom gate definitions, gate modifiers, general classical
expressions, classical conditional gates, loops, symbolic parameters, pulse
calibrations, scheduling, QIR lowering, noise channels or hardware mapping.
The module does not connect this circuit representation to
[`agent.quantum`](../agent/quantum.md), the separate optional Moonlab simulator.

The [historical 2026-10-25 quantum roadmap](../../reports/quantum-2026-10-25/quantum-landscape.md)
records the earlier source baseline and 100 proposed targets. This circuit
module contributes a validated static representation and a tested export
subset; it does not complete the roadmap's typed circuit IR, import/export
round trips, hardware adapters or research goals.

## Logical resources

For the Bell example, `quantum-circuit-resources` returns:

```scheme
((qubits . 2) (classical-bits . 2) (operations . 4) (gates . 2)
 (two-qubit-gates . 1) (measurements . 2) (resets . 0) (barriers . 0)
 (initialization-resets . 2))
```

`operations` counts the supplied operation lists. `gates` counts all supported
unitary operations, including `id`. `two-qubit-gates` counts operations with
exactly two qubit operands; `ccx` and `cswap` contribute to `gates` only.
`resets` counts explicit operations, while `initialization-resets` separately
reports the number of qubits reset by the export preamble. These counts do not
estimate circuit depth, hardware duration, native gate decomposition,
physical qubits or fault-tolerant resource cost.

## Validation commands and limits

The portable default test needs Python and a working native AOT compiler:

```sh
python3 tests/quantum/circuit_export_driver.py --eshkol-run ./build/eshkol-run
```

It compiles and runs 50 contract assertions, including invalid arities,
bounds, angle types, input copying, cycle rejection and revalidation after mutation. This
lane does not need Python quantum packages or `ESHKOL_QUANTUM_ENABLED`.
It also compiles the runnable Bell example and checks its exact output. The
driver's `--self-test` mode checks rejection of empty or incorrect verdicts,
nonzero child exits and error diagnostics even when a child exits zero;
CTest registers that check as `quantum_circuit_export_driver_selftest`.

The independent oracle is an explicit extra:

```sh
python3 -m pip install numpy "openqasm3[parser]" qiskit qiskit-qasm3-import
python3 tests/quantum/circuit_export_driver.py --eshkol-run ./build/eshkol-run --qiskit-oracle
```

It parses all exports with the OpenQASM parser and the
[Qiskit OpenQASM 3 importer](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qasm3),
then compares independently constructed Qiskit circuits. The 134 fixtures
cover each of the 25 gate kinds, Bell/GHZ states, exact-rational and decimal
angles, reset, barriers, measurement destinations and overwrites, an empty
circuit, and 100 generated three- or four-qubit circuits using seed 20261025.
Unitary matrices and statevectors use absolute and relative tolerances of
`1e-12`, including global phase; probability vectors use `1e-10`. Measurement
and reset cases use exact density-matrix branching, with analytic expected
probabilities and a `1e-12` tolerance, rather than sampled shots.

The oracle fails if any dependency is missing. It does not report a skipped
oracle as a successful check. `--work-dir DIRECTORY` retains each generated
program and exported QASM for inspection. The tests establish bounded
classical interoperability; they do not measure hardware execution, quantum
advantage, broad OpenQASM conformance, or another execution engine's behavior.

CTest registers `quantum_circuit_export_aot` when tests and the native
`eshkol-run` target are enabled. Set
`-DESHKOL_QUANTUM_CIRCUIT_QISKIT_ORACLE=ON` to also register
`quantum_circuit_export_qiskit_oracle`; its Python interpreter must have the
extras above. Run the lanes with:

```sh
ctest --test-dir build --output-on-failure -R '^quantum_circuit_export_'
```
