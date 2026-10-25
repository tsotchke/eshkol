---
kind: report
status: historical
owner-area: quantum
since: v1.4.0
sources:
  - lib/agent/quantum.esk
  - lib/agent/c/agent_quantum.c
  - CMakeLists.txt
---
# Eshkol quantum landscape and 100-target improvement roadmap

Research date: 2026-10-25. Source baseline: `341ac812ed600fd9d27c28e9509d66ab71363947`. This is a research and engineering specification, not a claim that 100 features were implemented.

Eshkol is not established as the world's leading quantum language by the evidence reviewed. A credible route to leadership is to build a particularly reliable compiled environment for differentiable scientific simulation, connect it to established quantum tools, and publish reproducible results on a declared workload class. No single feature count can fairly rank a typed language, a cloud SDK, a pulse system and an open-system simulator.

There are exactly **25 distinct comparators** below. They include standalone languages, embedded languages/DSLs, an instruction specification, SDKs and scientific frameworks. There are exactly **100 unique targets**, each with a baseline, source IDs, acceptance criteria and dependencies. These are **not 100 things missing from every competitor**. Many are already normal capabilities elsewhere; some harden existing Eshkol mechanisms; the last ten are research programs with uncertain outcomes.

The official pages/repositories were browsed during this review. Feature descriptions are source-supported, but competitor implementations were not installed or benchmarked. The Eshkol baseline comes from local source inspection, not a new quantum-enabled runtime test. Existing tests are evidence of test coverage intent, not a claim that they passed here. Numerical thresholds in acceptance criteria are proposed engineering gates, not reported measurements.

## Eshkol's actual starting point

The public `agent.quantum` module exposes opaque state handles, H/X/Y/Z/CNOT/RX/RY/RZ, computational-basis measurement, Z expectation, simulated CHSH, molecular and arbitrary Pauli Hamiltonians, ideal VQE energy/optimization/gradients, and a four-parameter quantum geometric tensor. `vqe-energy` has a custom VJP bridge into host AD. These are substantive mechanisms, but the public surface is small and tied to Moonlab's classical simulator. [E01](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c).

Quantum support is opt-in; the inspected build option defaults OFF. The exact Moonlab fetch pin is `4bf83a6c47e7f1c53db14fcb2a61053499fd82df`, while nearby comments and some design prose name older versions. A shipped-source header does not establish availability in every installed binary. [E03](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/CMakeLists.txt).

The language has a linear `Qubit` type and explicit regression fixtures, including documented inference boundaries. That does not by itself prove that every public integer-handle gate call obeys linear ownership. Integrating those layers is a concrete opportunity. [E06](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/types/hott_types.cpp), [E07](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/typesystem/qubit_alias_laundering_not_enforced_test.esk).

Moonlab's broader tensor-network, dynamics, decoder and other facilities are not automatically Eshkol features. The integration document explicitly describes several as deferred bindings. General Eshkol LLVM, GPU, exact-arithmetic or AD support likewise does not prove QIR output, GPU-resident quantum states, arbitrary circuit differentiation, physical QPU access or fault tolerance. [E05](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/design/MOONLAB_INTEGRATION.md).

Simulated Bell correlations are useful simulator checks. They do not certify device-independent physical randomness. That requires a physical experiment and explicit assumptions/statistical analysis; the original Bell-certified randomness work is an example of such evidence. [R09](https://arxiv.org/abs/0911.3427).

## Comparison method and limits

The table compares documented strengths, model boundaries and practical lessons for Eshkol. It deliberately contains no invented numerical score, market rank, speed ratio or universal winner. A feature being absent means **no supported public Eshkol quantum API was found in the inspected surface**; it is not a proof that nobody could implement it using general-purpose code, FFI or an unexposed Moonlab API. Unknown means the needed runtime/deployment evidence was not established. Implemented means a mechanism exists in source, not that the stronger acceptance gate is already satisfied. Partial identifies an actual narrower mechanism.

Some documents are development or historical specifications; each affected comparator notes that. Failed retrievals used official repository fallbacks. Supplementary research sources such as Stim, PyZX, QIR and VOQC support specific targets but are not extra entries in the fixed 25-comparator count.

## The 25 comparators

| ID / system | What it is | Documented strength | Boundary and Eshkol implication |
|---|---|---|---|
| [C01: Q#](https://learn.microsoft.com/en-us/azure/quantum/qsharp-overview) | standalone quantum language; QDK; Python interoperability | Hardware-independent quantum programs with qubit management and classical control. | Language safety and hardware portability do not establish application advantage. **Eshkol:** Make resource lifetimes and target capabilities explicit in the quantum API. |
| [C02: OpenQASM 3](https://openqasm.com/) | quantum/classical language and interchange specification; toolchain-neutral | Gates, classical control, timing, measurement and calibration syntax. | A specification is not an execution service; implementations support different subsets. **Eshkol:** Export a declared subset, reject unsupported constructs and test round trips. |
| [C03: Quil](https://quil-lang.github.io/) | quantum instruction language / IR; Rigetti ecosystem and other consumers | Gate modifiers, classical memory/control and pulse-control annex. | The inspected specification labels itself 2021.1 DRAFT; do not assume every backend implements it. **Eshkol:** Separate abstract circuit intent from backend-specific instructions. |
| [C04: Quipper](https://www.mathstat.dal.ca/~selinger/quipper/) | embedded functional quantum language; Haskell | Hierarchical circuit construction, reversible synthesis and multiple staging phases. | Research-language documentation is not a current hardware-service support guarantee. **Eshkol:** Add composable circuit transforms and explicit circuit-generation versus execution stages. |
| [C05: Silq](https://silq.ethz.ch/) | standalone research quantum language; dedicated compiler | Strong static types and safe automatic uncomputation. | Its research design is useful precedent, not evidence of broad production deployment. **Eshkol:** Integrate quantum resource types with a sound uncomputation discipline. |
| [C06: Qiskit](https://quantum.cloud.ibm.com/docs/en/guides) | SDK and compiler ecosystem; Python | Circuit workflows, optimization, specialized packages and IBM QPU execution. | Qiskit, runtime services and addons are separate components; do not count the whole ecosystem as one language. **Eshkol:** Prioritize real provider adapters, portable circuits and execution provenance. |
| [C07: Cirq](https://quantumai.google/cirq) | circuit SDK / Python library; Python | Hardware-aware circuits plus state-vector, density-matrix and noisy simulation. | Hardware access and noise models are target-specific; a simulator is still classical. **Eshkol:** Represent devices, noise and shot-based results explicitly. |
| [C08: PennyLane](https://github.com/PennyLaneAI/pennylane) | differentiable quantum framework and compiler ecosystem; Python; Catalyst/Lightning | Quantum/classical differentiable workflows, chemistry tools, simulators and hardware integrations. | Gradient methods and devices have domains of validity; broad platform claims are not benchmarks. **Eshkol:** Generalize the fixed VQE bridge while testing gradients against independent oracles. |
| [C09: CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/) | hybrid programming model, embedded languages and toolchain; C++ and Python | CPU/GPU/QPU workflows, simulator backends, dynamics and multi-GPU execution. | Current docs distinguish logical tooling preview and pulse research preview from supported features. **Eshkol:** Define device/runtime interfaces and measure transfer-inclusive performance. |
| [C10: Qrisp](https://www.qrisp.eu/) | high-level embedded quantum language; Python; JAX/Jasp integration | Typed quantum variables, automatic uncomputation, arithmetic and block-encoding abstractions. | Its large-qubit sparse-simulation examples apply to some circuits, not arbitrary states. **Eshkol:** Build reusable quantum data abstractions without hiding state-size assumptions. |
| [C11: Qmod](https://docs.classiq.io/qmod-reference) | high-level quantum modeling language; native syntax and Python embedding | Functional models compiled and synthesized under circuit-property constraints. | Native and Python forms are the same language, not two comparators; graphical support is a subset. **Eshkol:** Offer explicit resource constraints and inspectable synthesized circuits. |
| [C12: BQSKit](https://bqskit.readthedocs.io/en/latest/) | quantum compiler and synthesis framework; Python | Circuit IR, numerical synthesis, extensible compiler passes and distributed compilation. | A compiler framework does not itself supply all algorithm or hardware services. **Eshkol:** Adopt pass contracts, cost functions and verification for circuit optimization. |
| [C13: TKET / pytket](https://docs.quantinuum.com/tket/) | compiler toolkit and SDK; C++ core; Python interface | Native-gate compilation, placement, routing and multiple backend integrations. | Device feasibility must be checked after transformations; optimization is workload-dependent. **Eshkol:** Treat routing and native-gate legality as independently tested compiler stages. |
| [C14: Amazon Braket SDK](https://docs.aws.amazon.com/braket/latest/developerguide/what-is-braket.html) | cloud SDK / managed-service interface; Python | Common access to simulators, gate devices, analog hardware and hybrid workflows. | Provider availability, charges and hardware characteristics are service-specific. **Eshkol:** Add explicit job state, shot budgets and result provenance, with simulator-only defaults. |
| [C15: Qibo](https://qibo.science/qibo/stable/) | quantum computing framework; Python | Backend drivers, algorithm examples, tomography and hybrid research workflows. | The fetched stable docs identify v0.3.6; optional components are not all core functions. **Eshkol:** Provide backend-independent experiments with numerical reference results. |
| [C16: Qulacs](https://docs.qulacs.org/en/latest/) | quantum circuit simulator / SDK; C++ with Python interface | Noisy and parametric circuit simulation, density matrices and circuit-learning examples. | Simulation speed and scale depend on circuit/state representation and hardware. **Eshkol:** Benchmark specialized simulation kernels against strong classical baselines. |
| [C17: Yao.jl](https://github.com/QuantumBFS/Yao.jl) | quantum framework / embedded block DSL; Julia | Composable quantum blocks, algorithm construction and GPU/tensor-network companion packages. | The repository calls itself early-release beta; companion packages are distinct dependencies. **Eshkol:** Use compositional circuits and expose backend extensions without overstating core scope. |
| [C18: QuantumOptics.jl](https://docs.qojulia.org/) | open-system scientific simulation framework; Julia | Hilbert-space-aware objects, master equations, trajectories, steady states and correlations. | This is a scientific simulator, not a gate-QPU programming language. **Eshkol:** Separate open-system dynamics from gate circuits while sharing typed observables. |
| [C19: Strawberry Fields](https://strawberryfields.readthedocs.io/en/latest/) | photonic quantum SDK / embedded circuit DSL; Python | Photonic circuits, simulation backends and differentiable program optimization. | The stable URL redirects to 0.24.0-dev documentation; current cloud access was not verified. **Eshkol:** Treat continuous-variable photonics as a distinct model with truncation checks. |
| [C20: Perceval](https://github.com/Quandela/Perceval) | discrete-variable photonic SDK; Python with compiled backends | Linear-optics components, Fock states, source-error models and Quandela QPU interface. | Discrete-photon and continuous-variable models are not interchangeable. **Eshkol:** Add a separate photonic adapter and preserve heralding and postselection statistics. |
| [C21: Pulser](https://pulser.readthedocs.io/en/stable/) | neutral-atom pulse SDK / DSL; Python | Device-constrained pulse sequences and analog neutral-atom simulation. | Pulse validity depends on a particular device, not just a circuit equivalence relation. **Eshkol:** Validate geometry, timing and waveform constraints in an analog execution layer. |
| [C22: Qadence](https://pasqal-io.github.io/qadence/latest/) | differentiable digital-analog framework; Python; PyTorch | Block-based programs, symbolic parameters, analog generators and differentiable workflows. | Model/backend support must be checked per operation; not every pulse is differentiable. **Eshkol:** Keep digital/analog differentiation contracts explicit and testable. |
| [C23: Bloqade](https://bloqade.quera.com/) | neutral-atom SDK and compiler DSL ecosystem; Python; multiple components | Analog/digital experiment workflows, atom movement, noisy near-Clifford simulation and QEC tools. | The current site exposes multiple packages and target products, not one universal language. **Eshkol:** Model hardware geometry, losses and logical execution rather than assuming ideal gates. |
| [C24: ProjectQ](https://projectq.readthedocs.io/en/latest/) | embedded quantum DSL and compiler framework; Python | Extensible compiler pipeline and quantum algorithm execution on simulators/devices. | The fetched docs are a development build; individual backend support must be rechecked. **Eshkol:** Make compiler extensions small, inspectable and covered by functional tests. |
| [C25: Guppy](https://docs.quantinuum.com/guppy/) | statically typed embedded quantum language; Python | Linear qubit ownership, measurement-dependent control, metaprogramming and fault-tolerant extensions. | Guppy, its FT extension and Selene emulator have separate responsibilities. **Eshkol:** Make the typed quantum language, runtime handles and execution model agree. |


## An actionable sequence

The phase is a release gate, not a delivery-date promise. Teams can work in parallel only when listed dependencies are satisfied; a P0 evidence task may depend on a later capability before its full gate can close.

| Phase | Outcome | Exit condition |
|---|---|---|
| P0: trustworthy baseline | Capabilities, packaging, correctness, diagnostics and evidence | Reproducible ON/OFF builds; no silent wrong-result/refusal gaps in declared scope; documented source/runtime boundaries. |
| P1: portable circuit core | Circuit IR, interchange, quantum ownership and compilation | At least two independent executors agree on a declared corpus; transforms preserve semantics and hardware constraints. |
| P2: useful scientific work | Noise, richer simulation, general differentiable circuits and algorithms | Independent numerical oracles, explicit error/shot budgets, and reproducible application examples. |
| P3: device and logical execution | Actual QPU jobs, analog controls, QEC and resource accounting | Real hardware evidence where claimed; costs, calibration and physical/logical assumptions recorded. |
| R: research | Ten specific scientific programs | A reproducible result or a useful negative result; no guaranteed discovery, speedup or universal superiority. |

First implementation slice: T001/T003/T004/T007/T008/T010/T041/T050/T081/T090. Then build T015 and a minimal T011/T018 pair. Prefer adapters to mature tools where they provide a useful reference or backend; reimplement only when Eshkol can demonstrate a maintainability, correctness or measured performance benefit. Keep new physical-device spend and credentials explicit in execution policy.

## What would justify a leadership claim

Choose one claim such as “lowest end-to-end time for this differentiable molecular-simulation workload at this error tolerance,” rather than “best language.” Publish source versions, compilation and warm-up treatment, precision, hardware, memory, transfer time, shot count, optimizer budget, failures and raw outputs. Compare tuned alternatives from the relevant subset of the 25; do not compare a classical state-vector simulator to a fault-tolerant resource estimate as if they solve the same task. Use a frozen public corpus plus held-out cases and independent replication. Report confidence intervals and Pareto tradeoffs; do not hide correctness failures inside an average score. Application-oriented benchmarking provides useful precedent for including both output quality and time to solution. [R08](https://arxiv.org/abs/2110.03137).

Useful scorecard axes are correctness/refusal behavior, supported execution models, reproducibility, time/memory at fixed accuracy, hardware portability, usability, and research validity. Their measured values are currently **not populated** by this research. Popularity, feature counts, code size and generated prose are not substitute measurements.

## Scientific aspirations and why they remain research

Fault-tolerant chemistry and Hamiltonian simulation motivate precision-aware libraries and resource models. A small ideal VQE example does not establish useful chemical accuracy for a difficult molecule. Fault-tolerant resource estimates are conditional projections, including code and hardware assumptions, not proof that the hardware exists. [R02](https://arxiv.org/abs/1605.03590), [R03](https://arxiv.org/abs/1610.06546), [R01](https://arxiv.org/abs/1905.09749).

Reliable quantum learning requires more than a differentiable API. Barren plateaus constrain some circuit families, and coherent-data learning advantages rely on a specified access model and experiment. The roadmap asks for measured scaling and honest classical baselines rather than promising generic AI superiority. [R04](https://arxiv.org/abs/1803.11173), [R05](https://arxiv.org/abs/2112.00778).

Error mitigation has sampling-cost limits. Error-corrected logical memories are an important experimental achievement, but are not equivalent to useful fault-tolerant computation. The roadmap keeps mitigation, memory experiments, algorithm execution and resource projections separate. [R06](https://arxiv.org/abs/2109.04457), [R07](https://arxiv.org/abs/2408.13687), [R12](https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation).

Verified quantum optimization already has primary research precedents. Eshkol can extend that work toward its own adaptive programs without claiming verification is absent everywhere. Distributed quantum networking similarly needs physical entanglement, communication and timing models beyond a programming-language feature. [R10](https://arxiv.org/abs/1912.02250), [R11](https://repository.tudelft.nl/record/uuid%3Af2b04894-ece4-4443-984b-71ddb11a5ab2).

## The 100 measurable targets

Each target remains **planned** in this research artifact. Baseline and acceptance are separate: existing mechanisms still need expanded validation; new capabilities need implementation plus evidence. Source links ground either the Eshkol baseline or the external precedent/problem. Acceptance thresholds and dependencies are proposed engineering judgments, not claims copied from competitors.


### Reliable foundations


**T001 — Runtime quantum capability query**

Phase: P0; class: foundation; Eshkol baseline: **partial**. Build flag and unavailable stubs exist; no unified Scheme capability report found.

Acceptance: A query reports build flag, simulator backend, ABI, supported operations and derivative modes; ON/OFF CI asserts each field without executing an unavailable operation.

Dependencies: none. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [E03: Build configuration](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/CMakeLists.txt), [C01: Q#](https://learn.microsoft.com/en-us/azure/quantum/qsharp-overview).


**T002 — Reproducible optional quantum packages**

Phase: P0; class: foundation; Eshkol baseline: **partial**. Opt-in source integration exists; this stream has not validated packaged ON builds.

Acceptance: Clean Windows, Linux and macOS jobs install a pinned ON package, run Bell/VQE smoke tests, and verify OFF builds return documented unavailability.

Dependencies: T001. Sources: [E03: Build configuration](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/CMakeLists.txt), [C09: CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/).


**T003 — Honest execution and entropy provenance**

Phase: P0; class: foundation; Eshkol baseline: **partial**. Source labels exist but some prose conflates simulated Bell tests with physical certification.

Acceptance: Every result identifies classical simulator or named physical device; entropy output reports actual source and fallback events; no simulation is labelled device-independent certified entropy.

Dependencies: T001. Sources: [E04: Quantum reference](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/reference/agent/quantum.md), [E05: Moonlab integration design](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/design/MOONLAB_INTEGRATION.md), [E10: Entropy wrapper](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/quantum/quantum_rng_wrapper.c), [R09: Pironio et al.: Random Numbers Certified by Bells Theorem](https://arxiv.org/abs/0911.3427).


**T004 — Moonlab ABI and revision manifest**

Phase: P0; class: foundation; Eshkol baseline: **partial**. CMake pins a commit while nearby comments/reference text mention other revisions.

Acceptance: Generated manifest records exact dependency commit and ABI; intentional mismatches fail before allocation; release docs are generated from the same manifest.

Dependencies: T001. Sources: [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [E03: Build configuration](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/CMakeLists.txt), [E05: Moonlab integration design](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/design/MOONLAB_INTEGRATION.md).


**T005 — Generational quantum handles**

Phase: P0; class: foundation; Eshkol baseline: **partial**. Opaque integer tables and destroy helpers exist; stale-handle reuse needs a specified contract.

Acceptance: Allocate/free/reallocate fuzzing never lets a stale handle access a new state; double destruction is defined; invalid handles produce catchable errors under sanitizers.

Dependencies: T004. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C25: Guppy](https://docs.quantinuum.com/guppy/).


**T006 — Typed qubit ownership through public APIs**

Phase: P0; class: foundation; Eshkol baseline: **partial**. Linear Qubit exists, but state-handle plus integer-index wrappers do not establish full resource typing.

Acceptance: Compile-fail tests cover duplicate use, escaping aliases and interprocedural returns; valid scoped gates compile; runtime state ownership agrees with the static qubit identity.

Dependencies: T005. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E06: Qubit type definition](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/types/hott_types.cpp), [E07: Linearity boundary regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/typesystem/qubit_alias_laundering_not_enforced_test.esk), [C05: Silq](https://silq.ethz.ch/), [C25: Guppy](https://docs.quantinuum.com/guppy/).


**T007 — Uniform quantum error contract**

Phase: P0; class: foundation; Eshkol baseline: **partial**. C shim uses error codes/NaN and wrappers translate selected failures.

Acceptance: Invalid counts, indices, angles, destroyed handles and allocation failures all produce documented catchable errors; no invalid case silently returns a plausible numeric result.

Dependencies: T005. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C24: ProjectQ](https://projectq.readthedocs.io/en/latest/).


**T008 — Dense ideal-state correctness gate**

Phase: P0; class: foundation; Eshkol baseline: **implemented**. Dense state creation, gates, measurement and Z expectation have public wrappers and source tests.

Acceptance: Exhaustively compare all exposed 1- and 2-qubit gates with independent complex matrices; norm and expectation errors stay below 1e-12 in float64 fixtures.

Dependencies: T002. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [E08: Quantum surface regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/quantum_surface_coverage_test.esk), [C16: Qulacs](https://docs.qulacs.org/en/latest/).


**T009 — Seeded simulator replay**

Phase: P0; class: foundation; Eshkol baseline: **absent**. Measurement wrapper uses a process entropy context; caller-supplied seed is not exposed.

Acceptance: A seed plus serialized program reproduces samples on the same version; different seeds pass distribution checks; deterministic replay is never advertised as physical entropy.

Dependencies: T003. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C07: Cirq](https://quantumai.google/cirq).


**T010 — Backend conformance ledger**

Phase: P0; class: foundation; Eshkol baseline: **unknown**. Source tests exist but no fresh complete native/VM/browser quantum matrix was run here.

Acceptance: Publish operation-by-engine supported/refused/untested statuses tied to executable tests; zero unsupported cases silently succeed; include release hashes and test logs.

Dependencies: T002, T007. Sources: [E03: Build configuration](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/CMakeLists.txt), [E08: Quantum surface regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/quantum_surface_coverage_test.esk), [E09: VQE adversarial AD regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/vqe_ad_adversarial.esk), [R08: Lubinski et al.: Application-Oriented Performance Benchmarks for Quantum Computing](https://arxiv.org/abs/2110.03137).


### Interoperability


**T011 — OpenQASM 3 export**

Phase: P1; class: parity; Eshkol baseline: **absent**. No public circuit export API found in the inspected module/shim.

Acceptance: Export a declared gate/measurement subset; an independent OpenQASM parser accepts 100 fixtures and simulation probabilities match within 1e-10.

Dependencies: T015. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C02: OpenQASM 3](https://openqasm.com/).


**T012 — OpenQASM 3 import**

Phase: P1; class: parity; Eshkol baseline: **absent**. No public parser/import adapter found in the quantum surface.

Acceptance: Parse the declared subset with source locations; reject pulse/unsupported dynamic constructs explicitly; imported Bell, QFT and adaptive fixtures match reference behavior.

Dependencies: T015. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C02: OpenQASM 3](https://openqasm.com/).


**T013 — QIR target profile**

Phase: P1; class: parity; Eshkol baseline: **absent**. LLVM compilation alone is not a QIR emitter; none found in inspected quantum interface.

Acceptance: Emit a named QIR profile with verified LLVM IR and required metadata; run at least 20 programs on an independent compatible executor.

Dependencies: T015, T025. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [R13: QIR specification](https://github.com/qir-alliance/qir-spec).


**T014 — Quil interchange adapter**

Phase: P1; class: parity; Eshkol baseline: **absent**. No public Quil serializer/parser found in the quantum interface.

Acceptance: Round-trip gates, classical memory and measurement for a documented Quil subset; preserve bit order and reject pulse annexes until implemented.

Dependencies: T015, T017. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C03: Quil](https://quil-lang.github.io/).


**T015 — Typed circuit intermediate representation**

Phase: P1; class: foundation; Eshkol baseline: **absent**. Current public API mutates opaque simulator state directly rather than exposing circuit objects.

Acceptance: Represent qubits, gates, parameters, measurements and dependencies with a versioned schema; validation catches dangling wires and parameter-type errors before execution.

Dependencies: T006, T007. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C12: BQSKit](https://bqskit.readthedocs.io/en/latest/), [C24: ProjectQ](https://projectq.readthedocs.io/en/latest/).


**T016 — Portable observable and Hamiltonian interchange**

Phase: P1; class: parity; Eshkol baseline: **partial**. Pauli-string Hamiltonian construction exists; no general import/export schema found.

Acceptance: Serialize arbitrary Pauli sums, units, constants and wire order; round-trip 50 fixtures with identical coefficients and expectation values within 1e-12.

Dependencies: T015. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C10: Qrisp](https://www.qrisp.eu/), [C08: PennyLane](https://github.com/PennyLaneAI/pennylane).


**T017 — Shot-result and bit-order schema**

Phase: P1; class: parity; Eshkol baseline: **absent**. Single-qubit measurement returns an integer; no portable batch-result schema found.

Acceptance: Specify endianness, register mapping, counts, shots, uncertainty and provenance; adapters pass asymmetric-bitstring tests that detect reversal errors.

Dependencies: T003, T015. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C14: Amazon Braket SDK](https://docs.aws.amazon.com/braket/latest/developerguide/what-is-braket.html), [C07: Cirq](https://quantumai.google/cirq).


**T018 — Backend plugin contract**

Phase: P1; class: parity; Eshkol baseline: **absent**. Moonlab is compiled into the shim; no public quantum backend plugin contract found.

Acceptance: Two independent simulator adapters implement capability discovery, execute, sample and observe; unsupported operations fail consistently without implicit backend substitution.

Dependencies: T001, T015, T017. Sources: [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [E03: Build configuration](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/CMakeLists.txt), [C09: CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/), [C15: Qibo](https://qibo.science/qibo/stable/).


**T019 — PennyLane device integration**

Phase: P1; class: parity; Eshkol baseline: **absent**. No PennyLane-facing device plugin found in inspected Eshkol quantum surface.

Acceptance: A separately versioned plugin passes its declared device-conformance tests and a 20-case expectation/gradient comparison without copying parameters incorrectly.

Dependencies: T018, T041. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C08: PennyLane](https://github.com/PennyLaneAI/pennylane).


**T020 — Quantum array interchange**

Phase: P1; class: parity; Eshkol baseline: **absent**. Quantum state is opaque; VQE parameters are marshalled through a dedicated context.

Acceptance: Exchange complex amplitudes and parameter arrays through a documented C/Python boundary; shape, ownership, dtype and lifetime tests catch aliasing and precision loss.

Dependencies: T005, T017. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C17: Yao.jl](https://github.com/QuantumBFS/Yao.jl), [C16: Qulacs](https://docs.qulacs.org/en/latest/).


### Circuit language and compilation


**T021 — Complete practical gate vocabulary**

Phase: P1; class: parity; Eshkol baseline: **partial**. H, X, Y, Z, CNOT, RX, RY and RZ are exposed; many standard gates are not.

Acceptance: Add S, T, CZ, SWAP and controlled rotations or verified decompositions; compare matrices including global-phase conventions and invalid-wire cases.

Dependencies: T008, T015. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E05: Moonlab integration design](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/design/MOONLAB_INTEGRATION.md), [C02: OpenQASM 3](https://openqasm.com/), [C12: BQSKit](https://bqskit.readthedocs.io/en/latest/).


**T022 — Controlled and adjoint circuit combinators**

Phase: P1; class: parity; Eshkol baseline: **absent**. No general circuit modifier API is exposed.

Acceptance: Controlled and inverse transforms compose over supported circuits; random unitary fixtures satisfy U-dagger times U within 1e-10 and measurement inversion is rejected.

Dependencies: T015, T021. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C03: Quil](https://quil-lang.github.io/), [C04: Quipper](https://www.mathstat.dal.ca/~selinger/quipper/).


**T023 — Safe automatic uncomputation**

Phase: P1; class: advanced; Eshkol baseline: **absent**. Classical region memory does not implement quantum uncomputation.

Acceptance: For a declared reversible subset, compiler-generated cleanup returns ancillas to zero and preserves output coherence; unsafe erasures fail static checking.

Dependencies: T006, T022. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E06: Qubit type definition](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/types/hott_types.cpp), [C05: Silq](https://silq.ethz.ch/), [C10: Qrisp](https://www.qrisp.eu/).


**T024 — Ancilla allocation and reuse planner**

Phase: P1; class: parity; Eshkol baseline: **absent**. State allocation exists, but no circuit-level ancilla liveness planner is exposed.

Acceptance: Report peak live ancillas and verified reuse intervals; transformed circuits match references and never reuse an entangled live ancilla.

Dependencies: T015, T023. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C04: Quipper](https://www.mathstat.dal.ca/~selinger/quipper/), [C10: Qrisp](https://www.qrisp.eu/).


**T025 — Measurement-dependent circuit control**

Phase: P1; class: parity; Eshkol baseline: **partial**. Host Scheme can branch on measurement; portable in-circuit feedback representation is not established.

Acceptance: Represent mid-circuit measurement, reset, conditionals and bounded loops; teleportation and repeat-until-success fixtures match simulator/reference distributions.

Dependencies: T015, T017. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C02: OpenQASM 3](https://openqasm.com/), [C25: Guppy](https://docs.quantinuum.com/guppy/).


**T026 — Topology-aware routing**

Phase: P1; class: parity; Eshkol baseline: **absent**. No target connectivity/routing abstraction found in public quantum interface.

Acceptance: Route 50 fixed circuits to line/grid/device graphs; every two-qubit operation is legal, semantics preserved, and swap overhead is reported.

Dependencies: T015, T018. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C13: TKET / pytket](https://docs.quantinuum.com/tket/).


**T027 — Native-gate synthesis**

Phase: P1; class: parity; Eshkol baseline: **absent**. Direct simulator gate calls do not form a target synthesis pipeline.

Acceptance: Compile arbitrary tested 1-/2-qubit unitaries to a declared basis with chosen error budget; verify achieved error and report gate count and synthesis time.

Dependencies: T021, T026. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C11: Qmod](https://docs.classiq.io/qmod-reference), [C12: BQSKit](https://bqskit.readthedocs.io/en/latest/).


**T028 — Local circuit optimization passes**

Phase: P1; class: parity; Eshkol baseline: **absent**. No quantum pass-manager API found in exposed surface.

Acceptance: Cancellation, rotation fusion and commuting passes preserve 100 random small-circuit unitaries within 1e-10; publish before/after depth and count.

Dependencies: T015, T022. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C12: BQSKit](https://bqskit.readthedocs.io/en/latest/), [C24: ProjectQ](https://projectq.readthedocs.io/en/latest/).


**T029 — ZX optimization interoperability**

Phase: P1; class: advanced; Eshkol baseline: **absent**. No ZX diagram adapter found in quantum surface.

Acceptance: Export/import a defined Clifford+T subset to PyZX; certify equality or clearly report unverifiable cases; record T-count and extraction failures.

Dependencies: T011, T028. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R15: PyZX: official diagrammatic optimization library](https://github.com/zxcalc/pyzx).


**T030 — Proof-checked unitary rewrite kernel**

Phase: P1; class: advanced; Eshkol baseline: **absent**. General Eshkol proof types do not constitute verified quantum rewrite rules.

Acceptance: Mechanize semantics and prove at least five implemented rewrites; generated optimizer only uses checked rules, with numeric tests as secondary evidence.

Dependencies: T028. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R10: Hietala et al.: A Verified Optimizer for Quantum Circuits](https://arxiv.org/abs/1912.02250).


### Simulation and noise


**T031 — Density-matrix simulator adapter**

Phase: P2; class: parity; Eshkol baseline: **absent**. Public binding exposes dense pure states, not density matrices.

Acceptance: Bell-state partial trace gives maximally mixed marginals; trace and positivity tests pass; small noisy circuits match an independent density simulator within 1e-10.

Dependencies: T018, T020. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C07: Cirq](https://quantumai.google/cirq), [C16: Qulacs](https://docs.qulacs.org/en/latest/).


**T032 — Validated Kraus and noise channels**

Phase: P2; class: parity; Eshkol baseline: **absent**. No public channel/noise-model constructor found.

Acceptance: Implement depolarizing, dephasing and amplitude damping; validate completeness within tolerance and reject non-physical channel inputs.

Dependencies: T031. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C07: Cirq](https://quantumai.google/cirq), [C16: Qulacs](https://docs.qulacs.org/en/latest/).


**T033 — Quantum trajectory simulation**

Phase: P2; class: parity; Eshkol baseline: **absent**. No trajectory ensemble API found in public binding.

Acceptance: Ensemble averages converge to density-matrix references with declared confidence intervals; logs record trajectory count, seeds and sampling error separately from solver error.

Dependencies: T009, T032. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C18: QuantumOptics.jl](https://docs.qojulia.org/), [C07: Cirq](https://quantumai.google/cirq).


**T034 — Lindblad time evolution**

Phase: P2; class: advanced; Eshkol baseline: **absent**. Moonlab integration references other engines; public module has no open-system solver.

Acceptance: Damped two-level analytic solutions and independent solver fixtures agree under a documented tolerance; step refinement verifies convergence and trace preservation.

Dependencies: T031, T032. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E05: Moonlab integration design](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/design/MOONLAB_INTEGRATION.md), [C18: QuantumOptics.jl](https://docs.qojulia.org/).


**T035 — Stabilizer simulator backend**

Phase: P2; class: parity; Eshkol baseline: **absent**. No tableau/stabilizer operations exposed in the binding.

Acceptance: Execute Clifford circuits through a tested adapter; match exact small-system probabilities and run a 1000-qubit stabilizer fixture without exponential state storage.

Dependencies: T018, T021. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R14: Stim: official stabilizer/QEC simulator](https://github.com/quantumlib/Stim), [C04: Quipper](https://www.mathstat.dal.ca/~selinger/quipper/).


**T036 — Tensor-network simulation binding**

Phase: P2; class: advanced; Eshkol baseline: **absent**. Design names Moonlab CA-MPS/TDVP APIs but public module does not bind them.

Acceptance: Expose bond/truncation controls with discarded-weight diagnostics; match small exact references and show convergence as bond dimension increases.

Dependencies: T018, T020. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E05: Moonlab integration design](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/design/MOONLAB_INTEGRATION.md), [C17: Yao.jl](https://github.com/QuantumBFS/Yao.jl).


**T037 — GPU-resident quantum state execution**

Phase: P2; class: advanced; Eshkol baseline: **unknown**. Integration config disables optional Moonlab accelerators; general Eshkol GPU support does not prove this path.

Acceptance: Demonstrate actual device-resident gate execution and synchronization tests; report memory, correctness and end-to-end speed versus CPU including transfers.

Dependencies: T018, T020. Sources: [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [E03: Build configuration](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/CMakeLists.txt), [E05: Moonlab integration design](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/design/MOONLAB_INTEGRATION.md), [C09: CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/).


**T038 — Distributed quantum simulation**

Phase: P2; class: advanced; Eshkol baseline: **absent**. No multi-node quantum state partitioning API found in exposed binding.

Acceptance: Run one reproducible distributed state-vector fixture on two nodes; compare checksum/observables to a serial reference and report communication-inclusive scaling.

Dependencies: T037. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C09: CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/).


**T039 — Qudit state and gate support**

Phase: P2; class: advanced; Eshkol baseline: **absent**. Public API uses qubit counts and binary gates.

Acceptance: Support dimension-3 registers and mixed-dimension validation; qutrit Fourier/controlled-shift fixtures preserve norm and match independent matrices.

Dependencies: T015, T020. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C12: BQSKit](https://bqskit.readthedocs.io/en/latest/).


**T040 — Continuous-variable photonic model**

Phase: P2; class: advanced; Eshkol baseline: **absent**. Public API is qubit state-vector/VQE, with no Fock/Gaussian abstraction.

Acceptance: Implement or adapt Gaussian/Fock operations; coherent/squeezed-state observables agree with references and cutoff-convergence reports quantify truncation error.

Dependencies: T018, T020. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C19: Strawberry Fields](https://strawberryfields.readthedocs.io/en/latest/).


### Differentiation and hybrid training


**T041 — Preserve ideal VQE adjoint correctness**

Phase: P2; class: foundation; Eshkol baseline: **implemented**. vqe-energy custom VJP and vqe-gradient plus adversarial source tests exist.

Acceptance: On quantum-enabled releases, compare AD, direct VJP and central differences over random nonstationary points; maximum gradient error below 1e-6 and no unexplained zero gradients.

Dependencies: T002, T010. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [E09: VQE adversarial AD regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/vqe_ad_adversarial.esk), [C08: PennyLane](https://github.com/PennyLaneAI/pennylane).


**T042 — User-defined variational ansatz**

Phase: P2; class: parity; Eshkol baseline: **absent**. Public VQE uses a fixed one-layer hardware-efficient ansatz.

Acceptance: Users construct 3 distinct trainable circuit families; arbitrary parameter mapping is preserved and expectations/gradients match independent references.

Dependencies: T015, T022, T041. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C08: PennyLane](https://github.com/PennyLaneAI/pennylane), [C22: Qadence](https://pasqal-io.github.io/qadence/latest/).


**T043 — Shot-based parameter-shift gradients**

Phase: P2; class: parity; Eshkol baseline: **partial**. A native analytic fallback is documented, but user-controlled shot-based gradient execution is not exposed.

Acceptance: Estimate eligible gate derivatives with shot budgets and confidence intervals; 95% interval coverage is checked over repeated seeds against analytic small cases.

Dependencies: T017, T042. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C15: Qibo](https://qibo.science/qibo/stable/).


**T044 — General adjoint circuit differentiation**

Phase: P2; class: advanced; Eshkol baseline: **partial**. Adjoint path exists for default ideal VQE, not arbitrary exposed circuit objects.

Acceptance: Differentiate user circuits with 1, 10 and 100 parameters; match parameter-shift references and publish peak memory/runtime as depth and width vary.

Dependencies: T042. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C08: PennyLane](https://github.com/PennyLaneAI/pennylane), [C17: Yao.jl](https://github.com/QuantumBFS/Yao.jl).


**T045 — Second-order quantum derivative contract**

Phase: P2; class: advanced; Eshkol baseline: **unknown**. Arbitrary-order host AD does not establish arbitrary higher-order derivatives through the opaque VQE primitive.

Acceptance: Specify supported Hessian/JVP compositions; test symmetry and finite differences; unsupported nesting raises a named error rather than returning zero.

Dependencies: T041, T044. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [E09: VQE adversarial AD regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/vqe_ad_adversarial.esk), [C08: PennyLane](https://github.com/PennyLaneAI/pennylane).


**T046 — General quantum geometric tensor**

Phase: P2; class: advanced; Eshkol baseline: **partial**. vqe-qgt currently assumes four parameters and returns 4x4 entries.

Acceptance: Accept variable parameter counts; compare to state-derivative references, test symmetry/positive semidefiniteness, and state gauge and regularization conventions.

Dependencies: T042, T044. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C08: PennyLane](https://github.com/PennyLaneAI/pennylane).


**T047 — Differentiation through noisy channels**

Phase: P2; class: advanced; Eshkol baseline: **absent**. Public VQE bridge describes ideal-state energy and gradients.

Acceptance: Differentiate at least two parameterized CPTP channels; compare with density-matrix finite differences and distinguish stochastic estimator variance from numerical bias.

Dependencies: T032, T044. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C22: Qadence](https://pasqal-io.github.io/qadence/latest/).


**T048 — Batched parameter sweeps and gradients**

Phase: P2; class: parity; Eshkol baseline: **absent**. Public VQE evaluator takes one parameter vector per call.

Acceptance: Batch 32 independent parameter sets; outputs match serial runs, ordering is stable, and timing includes marshaling and device transfers.

Dependencies: T042, T044. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C09: CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/), [C08: PennyLane](https://github.com/PennyLaneAI/pennylane).


**T049 — Restartable hybrid optimizer workflow**

Phase: P2; class: parity; Eshkol baseline: **partial**. vqe-optimize offers an iteration count with a fixed default optimizer; general state/checkpoint control is not exposed.

Acceptance: Export optimizer state, RNG and parameters; an interrupted run resumes to the same seeded result, with loss/gradient/shot histories preserved.

Dependencies: T009, T043, T048. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C14: Amazon Braket SDK](https://docs.aws.amazon.com/braket/latest/developerguide/what-is-braket.html), [C22: Qadence](https://pasqal-io.github.io/qadence/latest/).


**T050 — Quantum derivative refusal and coverage manifest**

Phase: P2; class: foundation; Eshkol baseline: **partial**. Dedicated VQE checks exist; generic derivative coverage across new backends is not established.

Acceptance: Every operation/backend/order combination is supported or explicitly refused; negative tests cover measurement branches, unsupported channels and nested AD without silent zeros.

Dependencies: T010, T041. Sources: [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [E09: VQE adversarial AD regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/vqe_ad_adversarial.esk), [C08: PennyLane](https://github.com/PennyLaneAI/pennylane), [R04: McClean et al.: Barren plateaus in quantum neural network training landscapes](https://arxiv.org/abs/1803.11173).


### Scientific algorithm libraries


**T051 — General chemistry Hamiltonian import**

Phase: P2; class: parity; Eshkol baseline: **partial**. Fixed H2/LiH/H2O constructors and arbitrary Pauli sums exist.

Acceptance: Import a documented electronic-integral or Pauli dataset with basis, geometry and units; validate at least five small molecules against independent diagonalization.

Dependencies: T016. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C08: PennyLane](https://github.com/PennyLaneAI/pennylane), [R02: Reiher et al.: Elucidating Reaction Mechanisms on Quantum Computers](https://arxiv.org/abs/1605.03590).


**T052 — Symmetry tapering and sector selection**

Phase: P2; class: advanced; Eshkol baseline: **absent**. Public Hamiltonian construction accepts an HF bitmask but no tapering transformation.

Acceptance: Identify supported commuting symmetries; reduced and unreduced sector spectra agree on small fixtures; invalid sector requests are rejected.

Dependencies: T016, T051. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C08: PennyLane](https://github.com/PennyLaneAI/pennylane).


**T053 — Excited-state and spectral workflows**

Phase: P2; class: advanced; Eshkol baseline: **absent**. Public helpers focus on ground-state reference/VQE energy.

Acceptance: Compute two excited energies and overlaps for small Hamiltonians with declared method; independent eigensolver agreement and orthogonality checks are mandatory.

Dependencies: T042, T051. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C16: Qulacs](https://docs.qulacs.org/en/latest/).


**T054 — Quantum phase estimation library**

Phase: P2; class: parity; Eshkol baseline: **absent**. No public QPE combinator found in quantum module.

Acceptance: Known eigenphase fixtures recover each phase to the requested bits; account for ancillas, controlled-evolution synthesis and failure probability.

Dependencies: T022, T024, T025. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C04: Quipper](https://www.mathstat.dal.ca/~selinger/quipper/).


**T055 — Amplitude estimation library**

Phase: P2; class: parity; Eshkol baseline: **absent**. No exposed amplitude-estimation routine found.

Acceptance: Estimate a set of known Bernoulli amplitudes; observed confidence coverage matches declared guarantees and oracle-call counts include state preparation.

Dependencies: T022, T025. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C15: Qibo](https://qibo.science/qibo/stable/).


**T056 — QAOA with explicit classical baseline**

Phase: P2; class: parity; Eshkol baseline: **absent**. Public quantum layer exposes VQE but no QAOA constructor.

Acceptance: Small MaxCut instances match exhaustive optimum bounds; report approximation ratio, shots and total time against named classical solvers without claiming generic advantage.

Dependencies: T042, T049. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C07: Cirq](https://quantumai.google/cirq), [C22: Qadence](https://pasqal-io.github.io/qadence/latest/).


**T057 — Hamiltonian time-evolution circuits**

Phase: P2; class: parity; Eshkol baseline: **absent**. Gate wrappers do not expose a compositional simulation algorithm library.

Acceptance: Trotter/Suzuki evolution matches matrix exponentials on small models; refinement demonstrates expected convergence and error budgets include synthesis.

Dependencies: T016, T022, T027. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C18: QuantumOptics.jl](https://docs.qojulia.org/), [R03: Low and Chuang: Hamiltonian Simulation by Qubitization](https://arxiv.org/abs/1610.06546).


**T058 — Block encoding and quantum signal processing**

Phase: P2; class: advanced; Eshkol baseline: **absent**. No public block-encoding/QSP construction API found.

Acceptance: For bounded small matrices, verify top-left block encoding and polynomial transformation numerically; report normalization, oracle and approximation costs.

Dependencies: T022, T024, T027. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C10: Qrisp](https://www.qrisp.eu/), [R03: Low and Chuang: Hamiltonian Simulation by Qubitization](https://arxiv.org/abs/1610.06546).


**T059 — Tomography and classical-shadow experiments**

Phase: P2; class: parity; Eshkol baseline: **absent**. Z expectation exists; reconstruction/measurement-design workflows are not exposed.

Acceptance: Reconstruct specified small states/observables with uncertainty intervals, record shot counts and test coverage over independent seeds.

Dependencies: T017, T021, T032. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C15: Qibo](https://qibo.science/qibo/stable/).


**T060 — Discrete-photon experiment adapter**

Phase: P2; class: advanced; Eshkol baseline: **absent**. No public linear-optics/Fock-state interface found.

Acceptance: Run Hong-Ou-Mandel and a heralded small-circuit fixture through an adapter; preserve raw/postselected counts and model source/detector errors explicitly.

Dependencies: T018, T017. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C20: Perceval](https://github.com/Quandela/Perceval).


### Hardware execution and control


**T061 — First physical gate-QPU adapter**

Phase: P3; class: parity; Eshkol baseline: **absent**. Exposed Moonlab interface is a classical simulator; no QPU submission path found.

Acceptance: Submit a validated small program to one named physical device, record job ID/calibration/shots and reproduce retrieval; mocked tests never count as hardware validation.

Dependencies: T011, T018, T017. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C06: Qiskit](https://quantum.cloud.ibm.com/docs/en/guides), [C14: Amazon Braket SDK](https://docs.aws.amazon.com/braket/latest/developerguide/what-is-braket.html).


**T062 — Device topology and calibration snapshot**

Phase: P3; class: parity; Eshkol baseline: **absent**. No public physical-device model found.

Acceptance: Version connectivity, basis gates, durations and calibration timestamp; compilation rejects incompatible circuits and records exactly which snapshot it used.

Dependencies: T026, T061. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C07: Cirq](https://quantumai.google/cirq), [C13: TKET / pytket](https://docs.quantinuum.com/tket/).


**T063 — Durable asynchronous quantum jobs**

Phase: P3; class: parity; Eshkol baseline: **absent**. Local simulator calls are synchronous and no quantum job handle API is exposed.

Acceptance: Submission, polling, cancellation and restart recovery are idempotent; mocked failure tests cannot double-submit a chargeable job and preserve final provenance.

Dependencies: T061. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C14: Amazon Braket SDK](https://docs.aws.amazon.com/braket/latest/developerguide/what-is-braket.html).


**T064 — Execution and shot budget enforcement**

Phase: P3; class: foundation; Eshkol baseline: **absent**. No QPU cost/shot budget API found.

Acceptance: A preflight policy caps shots, retries and estimated spend; tests show retries cannot exceed caps; actual usage reconciles against recorded jobs.

Dependencies: T063. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C14: Amazon Braket SDK](https://docs.aws.amazon.com/braket/latest/developerguide/what-is-braket.html).


**T065 — Pulse and timing representation**

Phase: P3; class: advanced; Eshkol baseline: **absent**. No public pulse/frame/waveform representation found.

Acceptance: Represent frames, durations and waveforms with units; enforce target sample/timing limits; round-trip a documented pulse subset with no unit conversion ambiguity.

Dependencies: T015, T062. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C02: OpenQASM 3](https://openqasm.com/), [C03: Quil](https://quil-lang.github.io/).


**T066 — Neutral-atom geometry and waveform adapter**

Phase: P3; class: advanced; Eshkol baseline: **absent**. Public API has no atom positions, blockade model or analog program object.

Acceptance: Validate inter-atom spacing, channel limits and waveform schedules against one device schema; compare Rabi and small Ising simulations with references.

Dependencies: T065, T034. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C21: Pulser](https://pulser.readthedocs.io/en/stable/), [C23: Bloqade](https://bloqade.quera.com/).


**T067 — Digital-analog program composition**

Phase: P3; class: advanced; Eshkol baseline: **absent**. Quantum module exposes gates and fixed VQE, not mixed analog blocks.

Acceptance: Compose gate and Hamiltonian-evolution blocks with explicit units; ideal small-system results and parameter derivatives match an independent solver.

Dependencies: T066, T044. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C22: Qadence](https://pasqal-io.github.io/qadence/latest/).


**T068 — Hardware feedback latency contract**

Phase: P3; class: advanced; Eshkol baseline: **absent**. Host branching does not guarantee device-resident or bounded-latency feedback.

Acceptance: Measure control round-trip distributions on one backend and declare limits; unsupported feedback is rejected, with no claim of hard real-time bounds from arena allocation.

Dependencies: T025, T061. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C25: Guppy](https://docs.quantinuum.com/guppy/), [C09: CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/).


**T069 — Readout mitigation with uncertainty**

Phase: P3; class: parity; Eshkol baseline: **absent**. No public physical measurement-calibration model found.

Acceptance: Calibrate a small readout matrix; report corrected estimates, uncertainty and conditioning; held-out tests show when correction worsens results.

Dependencies: T061, T059. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C06: Qiskit](https://quantum.cloud.ibm.com/docs/en/guides), [R06: Takagi et al.: Fundamental limits of quantum error mitigation](https://arxiv.org/abs/2109.04457).


**T070 — Error mitigation under a fixed budget**

Phase: P3; class: advanced; Eshkol baseline: **absent**. No ZNE/PEC experiment framework exposed.

Acceptance: Implement one mitigation protocol with raw and corrected outputs; compare mean-squared error at equal total shots, including calibration overhead and failure regimes.

Dependencies: T032, T064, T069. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C22: Qadence](https://pasqal-io.github.io/qadence/latest/), [R06: Takagi et al.: Fundamental limits of quantum error mitigation](https://arxiv.org/abs/2109.04457).


### Error correction and resource estimation


**T071 — Logical-versus-physical qubit model**

Phase: P3; class: foundation; Eshkol baseline: **absent**. Qubit type and simulator count do not encode code distance or logical operations.

Acceptance: Resource objects distinguish logical qubits, physical qubits, code distance and gate level; mixing incompatible levels fails validation.

Dependencies: T015, T006. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E06: Qubit type definition](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/types/hott_types.cpp), [R12: Microsoft Quantum resource estimator](https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation), [C25: Guppy](https://docs.quantinuum.com/guppy/).


**T072 — Stabilizer-code construction library**

Phase: P3; class: parity; Eshkol baseline: **absent**. No public QEC code constructor found.

Acceptance: Generate repetition and distance-3/5 surface-code circuits; ideal syndromes and every single-Pauli injection match independently checked expectations.

Dependencies: T035, T071. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R14: Stim: official stabilizer/QEC simulator](https://github.com/quantumlib/Stim), [C23: Bloqade](https://bloqade.quera.com/).


**T073 — Detector error model interchange**

Phase: P3; class: advanced; Eshkol baseline: **absent**. No detector/syndrome schema exposed.

Acceptance: Export detector models for declared noise assumptions; an independent decoder consumes them and fault-injection checks preserve detector/observable mappings.

Dependencies: T032, T072. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R14: Stim: official stabilizer/QEC simulator](https://github.com/quantumlib/Stim).


**T074 — Decoder plugin and throughput harness**

Phase: P3; class: advanced; Eshkol baseline: **absent**. Moonlab design names decoder registries; Eshkol module does not expose them.

Acceptance: Two decoders consume the same syndrome batches; publish logical error, latency percentiles and throughput on fixed seeds/hardware with no unmeasured real-time claim.

Dependencies: T073. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E05: Moonlab integration design](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/design/MOONLAB_INTEGRATION.md), [R14: Stim: official stabilizer/QEC simulator](https://github.com/quantumlib/Stim).


**T075 — Logical error scaling experiments**

Phase: P3; class: advanced; Eshkol baseline: **absent**. No public code-distance/noise sweep workflow found.

Acceptance: Sweep at least three code distances and noise rates with confidence intervals; fit suppression only in measured regimes and expose insufficient-statistics outcomes.

Dependencies: T072, T074. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R07: Acharya et al.: Quantum error correction below the surface code threshold](https://arxiv.org/abs/2408.13687), [R14: Stim: official stabilizer/QEC simulator](https://github.com/quantumlib/Stim).


**T076 — Clifford+T synthesis error budget**

Phase: P3; class: advanced; Eshkol baseline: **absent**. No fault-tolerant gate/resource compiler surface found.

Acceptance: Lower supported rotations to Clifford+T with a declared synthesis bound; verify small unitaries and account separately for algorithmic, synthesis and physical error.

Dependencies: T027, T071. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C09: CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/), [R12: Microsoft Quantum resource estimator](https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation).


**T077 — Magic-state factory resource model**

Phase: P3; class: advanced; Eshkol baseline: **absent**. No public distillation/factory model found.

Acceptance: Compute physical-qubit/time estimates for named protocols under explicit input errors; reproduce a published small reference table and include parameter sensitivity.

Dependencies: T076. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R01: Gidney and Ekera: fault-tolerant factoring resource estimate](https://arxiv.org/abs/1905.09749), [R12: Microsoft Quantum resource estimator](https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation).


**T078 — Logical scheduling and lattice-surgery plan**

Phase: P3; class: advanced; Eshkol baseline: **absent**. No logical layout/schedule representation found.

Acceptance: Compile a small logical workload into a legal schedule for one architecture model; report space-time volume and validate resource conflicts.

Dependencies: T071, T076, T077. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R12: Microsoft Quantum resource estimator](https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation), [C25: Guppy](https://docs.quantinuum.com/guppy/).


**T079 — Fault-tolerant resource estimation adapter**

Phase: P3; class: parity; Eshkol baseline: **absent**. No public estimator bridge found; language compilation alone is insufficient.

Acceptance: Export a supported workload to an external estimator; preserve assumptions and compare physical qubits/runtime across at least three hardware/code scenarios.

Dependencies: T013, T076. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R12: Microsoft Quantum resource estimator](https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation), [C01: Q#](https://learn.microsoft.com/en-us/azure/quantum/qsharp-overview).


**T080 — Cross-layer resource and error budget ledger**

Phase: P3; class: advanced; Eshkol baseline: **absent**. No unified algorithm-to-physical cost ledger found.

Acceptance: A chemistry or phase-estimation example records precision, success probability, logical operations, factories and physical resources without double-counting overhead.

Dependencies: T054, T077, T079. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R01: Gidney and Ekera: fault-tolerant factoring resource estimate](https://arxiv.org/abs/1905.09749), [R02: Reiher et al.: Elucidating Reaction Mechanisms on Quantum Computers](https://arxiv.org/abs/1605.03590), [R12: Microsoft Quantum resource estimator](https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation).


### Developer experience and reproducible evidence


**T081 — Executable quantum documentation**

Phase: P0; class: foundation; Eshkol baseline: **partial**. Quantum examples/source tests exist but this stream did not execute every published example on every target.

Acceptance: CI extracts all quantum examples, runs declared supported targets, and marks external hardware prerequisites; wrong expected outputs block release.

Dependencies: T002, T010. Sources: [E04: Quantum reference](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/reference/agent/quantum.md), [E08: Quantum surface regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/quantum_surface_coverage_test.esk), [E09: VQE adversarial AD regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/vqe_ad_adversarial.esk), [C24: ProjectQ](https://projectq.readthedocs.io/en/latest/).


**T082 — Quantum-aware diagnostics**

Phase: P0; class: foundation; Eshkol baseline: **partial**. Wrapper error messages exist; circuit source mapping and richer diagnostics are absent.

Acceptance: Invalid wires, nonunitary gates, impossible ownership and unsupported target features report source spans and actionable reasons in 30 negative fixtures.

Dependencies: T007, T015. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [C25: Guppy](https://docs.quantinuum.com/guppy/).


**T083 — Circuit and state visualization**

Phase: P0; class: parity; Eshkol baseline: **absent**. No public circuit object/visualizer found in the binding.

Acceptance: Render circuits with parameter/measurement labels and small-state plots; snapshots preserve wire order and explicitly mark simulator-only state access.

Dependencies: T015, T017. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C20: Perceval](https://github.com/Quandela/Perceval), [C07: Cirq](https://quantumai.google/cirq).


**T084 — Quantum profiling trace**

Phase: P0; class: foundation; Eshkol baseline: **absent**. No unified quantum compile/execute/transfer/shot trace exposed.

Acceptance: Emit machine-readable spans for compilation, marshaling, transfers, execution and sampling; totals reconcile with wall time within documented instrumentation overhead.

Dependencies: T018. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [C09: CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/), [R08: Lubinski et al.: Application-Oriented Performance Benchmarks for Quantum Computing](https://arxiv.org/abs/2110.03137).


**T085 — Pinned reproducible experiment bundles**

Phase: P0; class: foundation; Eshkol baseline: **partial**. Dependency pinning exists; no universal experiment/result bundle schema was established here.

Acceptance: Bundle program, dependency hashes, seeds, device snapshot, environment, raw results and analysis; a clean runner reproduces a seeded simulator experiment.

Dependencies: T004, T009, T017. Sources: [E03: Build configuration](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/CMakeLists.txt), [R08: Lubinski et al.: Application-Oriented Performance Benchmarks for Quantum Computing](https://arxiv.org/abs/2110.03137).


**T086 — Differential quantum fuzzing**

Phase: P0; class: foundation; Eshkol baseline: **partial**. Source quantum tests exist; broad cross-tool randomized differential evidence is not established.

Acceptance: Generate 1000 small valid circuits and compare independent engines with phase/shot-aware tolerances; minimize every mismatch into a retained regression.

Dependencies: T008, T018, T085. Sources: [E08: Quantum surface regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/quantum_surface_coverage_test.esk), [E09: VQE adversarial AD regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/vqe_ad_adversarial.esk), [C07: Cirq](https://quantumai.google/cirq), [C16: Qulacs](https://docs.qulacs.org/en/latest/).


**T087 — Application benchmark protocol**

Phase: P0; class: foundation; Eshkol baseline: **absent**. Existing quantum microbenchmarks do not establish a 25-system application ranking.

Acceptance: Freeze workload families, precision, hardware, seeds and time-to-solution metrics before measurement; publish failures and tuning budgets alongside successes.

Dependencies: T084, T085. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R08: Lubinski et al.: Application-Oriented Performance Benchmarks for Quantum Computing](https://arxiv.org/abs/2110.03137).


**T088 — Independent replication package**

Phase: P0; class: foundation; Eshkol baseline: **unknown**. No independent replication of this proposed comparative protocol has occurred.

Acceptance: A second team or isolated environment reruns the published bundle; discrepancies are logged and resolved before any leadership claim.

Dependencies: T087. Sources: [E08: Quantum surface regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/quantum_surface_coverage_test.esk), [R08: Lubinski et al.: Application-Oriented Performance Benchmarks for Quantum Computing](https://arxiv.org/abs/2110.03137).


**T089 — Quantum extension compatibility governance**

Phase: P0; class: foundation; Eshkol baseline: **partial**. General project contribution rules exist; quantum plugin/version promises are not established.

Acceptance: Publish semantic-version and deprecation rules; two plugin versions run through a compatibility matrix and a deliberately incompatible ABI fails clearly.

Dependencies: T004, T018. Sources: [E03: Build configuration](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/CMakeLists.txt), [C24: ProjectQ](https://projectq.readthedocs.io/en/latest/), [R13: QIR specification](https://github.com/qir-alliance/qir-spec).


**T090 — Scientific claim-to-evidence registry**

Phase: P0; class: foundation; Eshkol baseline: **partial**. Existing docs mix implementation, historical design and performance/physical claims.

Acceptance: Every public quantum capability claim links to source, passing test or measured report and declares scope; simulation, hardware, projection and open research are distinct fields.

Dependencies: T003, T081, T087. Sources: [E04: Quantum reference](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/reference/agent/quantum.md), [E05: Moonlab integration design](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/design/MOONLAB_INTEGRATION.md), [R08: Lubinski et al.: Application-Oriented Performance Benchmarks for Quantum Computing](https://arxiv.org/abs/2110.03137), [R09: Pironio et al.: Random Numbers Certified by Bells Theorem](https://arxiv.org/abs/0911.3427).


### Open scientific research programs


**T091 — Fault-tolerant chemistry beyond toy molecules**

Phase: R; class: open research; Eshkol baseline: **absent**. Public molecular constructors/VQE do not demonstrate practical fault-tolerant chemistry.

Acceptance: Reproduce one small active-space reference end to end, then publish a larger-case resource estimate with basis/truncation/error budgets; independent chemistry review is required before a scientific claim.

Dependencies: T051, T058, T080. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R02: Reiher et al.: Elucidating Reaction Mechanisms on Quantum Computers](https://arxiv.org/abs/1605.03590), [R03: Low and Chuang: Hamiltonian Simulation by Qubitization](https://arxiv.org/abs/1610.06546).


**T092 — End-to-end advantage for a materials workload**

Phase: R; class: open research; Eshkol baseline: **unknown**. No matched classical-versus-quantum advantage experiment was established here.

Acceptance: Pre-register a meaningful observable, precision and total-cost metric; beat tuned classical baselines with reproducible evidence, or publish the null result. Simulation speed alone cannot close this target.

Dependencies: T057, T087, T088. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R03: Low and Chuang: Hamiltonian Simulation by Qubitization](https://arxiv.org/abs/1610.06546), [R08: Lubinski et al.: Application-Oriented Performance Benchmarks for Quantum Computing](https://arxiv.org/abs/2110.03137).


**T093 — Trainability-aware variational compiler**

Phase: R; class: open research; Eshkol baseline: **absent**. Native AD supplies derivatives but does not remove barren plateaus or shot complexity.

Acceptance: On predefined circuit families, predict and measure gradient-variance scaling; compare initialization/ansatz strategies at equal budgets and publish counterexamples to claimed improvements.

Dependencies: T043, T044, T087. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R04: McClean et al.: Barren plateaus in quantum neural network training landscapes](https://arxiv.org/abs/1803.11173), [C08: PennyLane](https://github.com/PennyLaneAI/pennylane).


**T094 — Mitigation with defensible cost guarantees**

Phase: R; class: open research; Eshkol baseline: **absent**. No Eshkol result establishes mitigation benefit under general noise and equal sampling cost.

Acceptance: Derive assumptions and overhead bounds for one method, then test equal-cost error across increasing depth; explicitly report regimes where mitigation loses.

Dependencies: T070, T087. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R06: Takagi et al.: Fundamental limits of quantum error mitigation](https://arxiv.org/abs/2109.04457).


**T095 — Useful logical computation beyond memory experiments**

Phase: R; class: open research; Eshkol baseline: **absent**. No logical hardware execution path or result exists in the inspected binding.

Acceptance: Execute a logical algorithm with repeated correction and compare to a defined physical implementation at equal task error; distinguish code-memory success from useful algorithmic advantage.

Dependencies: T061, T075, T078. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R07: Acharya et al.: Quantum error correction below the surface code threshold](https://arxiv.org/abs/2408.13687), [C25: Guppy](https://docs.quantinuum.com/guppy/).


**T096 — Verified adaptive quantum compilation**

Phase: R; class: open research; Eshkol baseline: **absent**. No proof links the quantum interface, dynamic control and backend lowering end to end.

Acceptance: Formalize a bounded adaptive subset and prove selected lowerings preserve measurement semantics; publish assumptions and proof artifacts, with adversarial tests for unproved boundaries.

Dependencies: T025, T030, T013. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R10: Hietala et al.: A Verified Optimizer for Quantum Circuits](https://arxiv.org/abs/1912.02250), [R16: Mahadev: Classical Verification of Quantum Computations](https://arxiv.org/abs/1804.01082).


**T097 — Learning from coherent experimental quantum data**

Phase: R; class: open research; Eshkol baseline: **absent**. Simulator-generated datasets do not establish coherent-data experimental advantage.

Acceptance: Reproduce a small coherent-copy learning task with a physical-data protocol and explicit classical access model; compare sample complexity and account for state preparation/readout.

Dependencies: T061, T059, T087. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R05: Huang et al.: Quantum advantage in learning from experiments](https://arxiv.org/abs/2112.00778).


**T098 — Physical device-independent entropy integration**

Phase: R; class: open research; Eshkol baseline: **absent**. Current simulated Bell/entropy layer is not a loophole-controlled physical Bell experiment.

Acceptance: Consume authenticated physical experiment records with stated independence assumptions, finite-size entropy bounds and extractor parameters; simulation-only mode must be incapable of claiming certification.

Dependencies: T003, T063, T090. Sources: [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c), [E10: Entropy wrapper](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/quantum/quantum_rng_wrapper.c), [R09: Pironio et al.: Random Numbers Certified by Bells Theorem](https://arxiv.org/abs/0911.3427).


**T099 — Distributed quantum network programming**

Phase: R; class: open research; Eshkol baseline: **absent**. No quantum network, entanglement-distribution or remote-state API is exposed.

Acceptance: Implement a simulator protocol with loss, fidelity and classical latency, then validate one hardware-linked entanglement task; report end-to-end success rates and assumptions.

Dependencies: T018, T063, T071. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R11: Wehner, Elkouss and Hanson: Quantum internet, a vision for the road ahead](https://repository.tudelft.nl/record/uuid%3Af2b04894-ece4-4443-984b-71ddb11a5ab2).


**T100 — Reproducible frontier resource co-design**

Phase: R; class: open research; Eshkol baseline: **absent**. No end-to-end hardware/code/algorithm Pareto study is exposed in Eshkol.

Acceptance: Reproduce a published resource scenario, vary algorithm/code/hardware assumptions, and publish non-dominated cost/error tradeoffs with sensitivity and independent checks; estimates remain projections.

Dependencies: T079, T080, T088. Sources: [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk), [R01: Gidney and Ekera: fault-tolerant factoring resource estimate](https://arxiv.org/abs/1905.09749), [R12: Microsoft Quantum resource estimator](https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation), [R02: Reiher et al.: Elucidating Reaction Mechanisms on Quantum Computers](https://arxiv.org/abs/1605.03590).


## Source and coverage ledger


- [C01: Q#](https://learn.microsoft.com/en-us/azure/quantum/qsharp-overview) — Official documentation or project repository; checked 2026-10-25.

- [C02: OpenQASM 3](https://openqasm.com/) — Official documentation or project repository; checked 2026-10-25.

- [C03: Quil](https://quil-lang.github.io/) — Official documentation or project repository; checked 2026-10-25.

- [C04: Quipper](https://www.mathstat.dal.ca/~selinger/quipper/) — Official documentation or project repository; checked 2026-10-25.

- [C05: Silq](https://silq.ethz.ch/) — Official documentation or project repository; checked 2026-10-25.

- [C06: Qiskit](https://quantum.cloud.ibm.com/docs/en/guides) — Official documentation or project repository; checked 2026-10-25.

- [C07: Cirq](https://quantumai.google/cirq) — Official documentation or project repository; checked 2026-10-25.

- [C08: PennyLane](https://github.com/PennyLaneAI/pennylane) — Official documentation or project repository; checked 2026-10-25. Official stable index fetched; two deeper documentation URLs returned HTTP 429, so repository used as fallback.

- [C09: CUDA-Q](https://nvidia.github.io/cuda-quantum/latest/) — Official documentation or project repository; checked 2026-10-25.

- [C10: Qrisp](https://www.qrisp.eu/) — Official documentation or project repository; checked 2026-10-25.

- [C11: Qmod](https://docs.classiq.io/qmod-reference) — Official documentation or project repository; checked 2026-10-25.

- [C12: BQSKit](https://bqskit.readthedocs.io/en/latest/) — Official documentation or project repository; checked 2026-10-25.

- [C13: TKET / pytket](https://docs.quantinuum.com/tket/) — Official documentation or project repository; checked 2026-10-25.

- [C14: Amazon Braket SDK](https://docs.aws.amazon.com/braket/latest/developerguide/what-is-braket.html) — Official documentation or project repository; checked 2026-10-25.

- [C15: Qibo](https://qibo.science/qibo/stable/) — Official documentation or project repository; checked 2026-10-25.

- [C16: Qulacs](https://docs.qulacs.org/en/latest/) — Official documentation or project repository; checked 2026-10-25.

- [C17: Yao.jl](https://github.com/QuantumBFS/Yao.jl) — Official documentation or project repository; checked 2026-10-25. docs.yaoquantum.org/stable was inaccessible to the browser tool; official repository supplied evidence.

- [C18: QuantumOptics.jl](https://docs.qojulia.org/) — Official documentation or project repository; checked 2026-10-25.

- [C19: Strawberry Fields](https://strawberryfields.readthedocs.io/en/latest/) — Official documentation or project repository; checked 2026-10-25. strawberryfields.ai redirected to Xanadu; documentation used instead.

- [C20: Perceval](https://github.com/Quandela/Perceval) — Official documentation or project repository; checked 2026-10-25. v1.0 docs were readable and linked v1.3; v1.3 retrieval failed, so current official repository used.

- [C21: Pulser](https://pulser.readthedocs.io/en/stable/) — Official documentation or project repository; checked 2026-10-25.

- [C22: Qadence](https://pasqal-io.github.io/qadence/latest/) — Official documentation or project repository; checked 2026-10-25.

- [C23: Bloqade](https://bloqade.quera.com/) — Official documentation or project repository; checked 2026-10-25.

- [C24: ProjectQ](https://projectq.readthedocs.io/en/latest/) — Official documentation or project repository; checked 2026-10-25.

- [C25: Guppy](https://docs.quantinuum.com/guppy/) — Official documentation or project repository; checked 2026-10-25.

- [R01: Gidney and Ekera: fault-tolerant factoring resource estimate](https://arxiv.org/abs/1905.09749) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R02: Reiher et al.: Elucidating Reaction Mechanisms on Quantum Computers](https://arxiv.org/abs/1605.03590) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R03: Low and Chuang: Hamiltonian Simulation by Qubitization](https://arxiv.org/abs/1610.06546) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R04: McClean et al.: Barren plateaus in quantum neural network training landscapes](https://arxiv.org/abs/1803.11173) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R05: Huang et al.: Quantum advantage in learning from experiments](https://arxiv.org/abs/2112.00778) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R06: Takagi et al.: Fundamental limits of quantum error mitigation](https://arxiv.org/abs/2109.04457) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R07: Acharya et al.: Quantum error correction below the surface code threshold](https://arxiv.org/abs/2408.13687) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R08: Lubinski et al.: Application-Oriented Performance Benchmarks for Quantum Computing](https://arxiv.org/abs/2110.03137) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R09: Pironio et al.: Random Numbers Certified by Bells Theorem](https://arxiv.org/abs/0911.3427) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R10: Hietala et al.: A Verified Optimizer for Quantum Circuits](https://arxiv.org/abs/1912.02250) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R11: Wehner, Elkouss and Hanson: Quantum internet, a vision for the road ahead](https://repository.tudelft.nl/record/uuid%3Af2b04894-ece4-4443-984b-71ddb11a5ab2) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R12: Microsoft Quantum resource estimator](https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R13: QIR specification](https://github.com/qir-alliance/qir-spec) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R14: Stim: official stabilizer/QEC simulator](https://github.com/quantumlib/Stim) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R15: PyZX: official diagrammatic optimization library](https://github.com/zxcalc/pyzx) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [R16: Mahadev: Classical Verification of Quantum Computations](https://arxiv.org/abs/1804.01082) — Primary paper, author repository, specification, or official technical documentation; checked 2026-10-25.

- [E01: Public quantum module](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/quantum.esk) — Inspected local source snapshot; checked 2026-10-25. Full exported surface and wrapper implementations inspected; opaque integer handles, eight gate operations, Z expectation, measurement, Hamiltonian/VQE/QGT APIs.

- [E02: Quantum C shim](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/agent/c/agent_quantum.c) — Inspected local source snapshot; checked 2026-10-25. Dense-state/VQE shim, handle tables, conditional Moonlab implementation and unavailable stubs inspected; QGT limited to four parameters.

- [E03: Build configuration](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/CMakeLists.txt) — Inspected local source snapshot; checked 2026-10-25. Quantum option OFF by default; exact Moonlab GIT_TAG is 4bf83a6c47e7f1c53db14fcb2a61053499fd82df. Nearby comments mention older revisions.

- [E04: Quantum reference](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/reference/agent/quantum.md) — Inspected local source snapshot; checked 2026-10-25. Public usage, ideal VQE adjoint bridge and availability descriptions; prose alone is not runtime evidence.

- [E05: Moonlab integration design](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/docs/design/MOONLAB_INTEGRATION.md) — Inspected local source snapshot; checked 2026-10-25. Mixed shipped header/historical design. Mentions broader Moonlab tensor-network and decoder APIs, many explicitly deferred from Eshkol binding.

- [E06: Qubit type definition](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/types/hott_types.cpp) — Inspected local source snapshot; checked 2026-10-25. Qubit is registered as a linear type with Int64 representation; public state/gate wrappers still use state handles and integer indices.

- [E07: Linearity boundary regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/typesystem/qubit_alias_laundering_not_enforced_test.esk) — Inspected local source snapshot; checked 2026-10-25. Fixture describes caught immediate-lambda aliasing and a remaining separately defined unannotated-return boundary; filename alone is misleading.

- [E08: Quantum surface regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/quantum_surface_coverage_test.esk) — Inspected local source snapshot; checked 2026-10-25. Source test covers exposed gates and molecular construction; not executed in this research stream.

- [E09: VQE adversarial AD regression](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/tests/quantum/vqe_ad_adversarial.esk) — Inspected local source snapshot; checked 2026-10-25. Source contains independent finite-difference checks around the ideal VQE bridge; not executed in this research stream.

- [E10: Entropy wrapper](https://github.com/tsotchke/eshkol/blob/341ac812ed600fd9d27c28e9509d66ab71363947/lib/quantum/quantum_rng_wrapper.c) — Inspected local source snapshot; checked 2026-10-25. Build-dependent Moonlab or classical source; a simulated Bell statistic is not device-independent physical entropy certification.


## Machine-checked inventory

```json
{
  "date": "2026-10-25",
  "eshkol_source_commit": "341ac812ed600fd9d27c28e9509d66ab71363947",
  "comparator_count": 25,
  "target_count": 100,
  "universal_missing_claim": false,
  "all_targets_implemented": false,
  "comparison_method": "Primary-source qualitative capability comparison; no comparative execution benchmark was performed.",
  "baseline_scope": "Inspected Eshkol public quantum module/shim and linked source snapshot, not all Moonlab capabilities or all Eshkol modules.",
  "baseline_states": {
    "partial": 22,
    "implemented": 2,
    "absent": 71,
    "unknown": 5
  },
  "target_classes": {
    "foundation": 24,
    "parity": 37,
    "advanced": 29,
    "open research": 10
  },
  "phases": {
    "P0": 20,
    "P1": 20,
    "P2": 30,
    "P3": 20,
    "R": 10
  }
}
```


The generator validates exactly 25 distinct comparator names, exactly 100 distinct target titles, allowed classifications, existing source IDs, valid dependency IDs and an acyclic dependency graph. These checks validate the artifact structure, not the scientific feasibility or completion of the roadmap. No source repository files were edited and no external publication was performed in this research stream.
