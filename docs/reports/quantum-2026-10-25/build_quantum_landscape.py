"""Generate an auditable research roadmap; this does not implement Eshkol features."""
import csv
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
DATE = '2026-10-25'
BASE = '341ac812ed600fd9d27c28e9509d66ab71363947'
sources = {}

def source(sid, title, url, evidence='Official documentation or project repository', note=''):
    sources[sid] = dict(id=sid, title=title, url=url, evidence=evidence, checked=DATE, note=note)

competitors = []
def comp(name, kind, host, strength, boundary, implication, url, note=''):
    sid = f'C{len(competitors)+1:02}'
    source(sid, name, url, note=note)
    competitors.append(dict(id=sid, name=name, classification=kind, host=host,
                            documented_strength=strength, comparison_boundary=boundary,
                            implication_for_eshkol=implication, source_ids=[sid],
                            measured_score=None, independently_benchmarked=False))

comp('Q#','standalone quantum language','QDK; Python interoperability',
     'Hardware-independent quantum programs with qubit management and classical control.',
     'Language safety and hardware portability do not establish application advantage.',
     'Make resource lifetimes and target capabilities explicit in the quantum API.',
     'https://learn.microsoft.com/en-us/azure/quantum/qsharp-overview')
comp('OpenQASM 3','quantum/classical language and interchange specification','toolchain-neutral',
     'Gates, classical control, timing, measurement and calibration syntax.',
     'A specification is not an execution service; implementations support different subsets.',
     'Export a declared subset, reject unsupported constructs and test round trips.',
     'https://openqasm.com/')
comp('Quil','quantum instruction language / IR','Rigetti ecosystem and other consumers',
     'Gate modifiers, classical memory/control and pulse-control annex.',
     'The inspected specification labels itself 2021.1 DRAFT; do not assume every backend implements it.',
     'Separate abstract circuit intent from backend-specific instructions.',
     'https://quil-lang.github.io/')
comp('Quipper','embedded functional quantum language','Haskell',
     'Hierarchical circuit construction, reversible synthesis and multiple staging phases.',
     'Research-language documentation is not a current hardware-service support guarantee.',
     'Add composable circuit transforms and explicit circuit-generation versus execution stages.',
     'https://www.mathstat.dal.ca/~selinger/quipper/')
comp('Silq','standalone research quantum language','dedicated compiler',
     'Strong static types and safe automatic uncomputation.',
     'Its research design is useful precedent, not evidence of broad production deployment.',
     'Integrate quantum resource types with a sound uncomputation discipline.',
     'https://silq.ethz.ch/')
comp('Qiskit','SDK and compiler ecosystem','Python',
     'Circuit workflows, optimization, specialized packages and IBM QPU execution.',
     'Qiskit, runtime services and addons are separate components; do not count the whole ecosystem as one language.',
     'Prioritize real provider adapters, portable circuits and execution provenance.',
     'https://quantum.cloud.ibm.com/docs/en/guides')
comp('Cirq','circuit SDK / Python library','Python',
     'Hardware-aware circuits plus state-vector, density-matrix and noisy simulation.',
     'Hardware access and noise models are target-specific; a simulator is still classical.',
     'Represent devices, noise and shot-based results explicitly.',
     'https://quantumai.google/cirq')
comp('PennyLane','differentiable quantum framework and compiler ecosystem','Python; Catalyst/Lightning',
     'Quantum/classical differentiable workflows, chemistry tools, simulators and hardware integrations.',
     'Gradient methods and devices have domains of validity; broad platform claims are not benchmarks.',
     'Generalize the fixed VQE bridge while testing gradients against independent oracles.',
     'https://github.com/PennyLaneAI/pennylane',
     'Official stable index fetched; two deeper documentation URLs returned HTTP 429, so repository used as fallback.')
comp('CUDA-Q','hybrid programming model, embedded languages and toolchain','C++ and Python',
     'CPU/GPU/QPU workflows, simulator backends, dynamics and multi-GPU execution.',
     'Current docs distinguish logical tooling preview and pulse research preview from supported features.',
     'Define device/runtime interfaces and measure transfer-inclusive performance.',
     'https://nvidia.github.io/cuda-quantum/latest/')
comp('Qrisp','high-level embedded quantum language','Python; JAX/Jasp integration',
     'Typed quantum variables, automatic uncomputation, arithmetic and block-encoding abstractions.',
     'Its large-qubit sparse-simulation examples apply to some circuits, not arbitrary states.',
     'Build reusable quantum data abstractions without hiding state-size assumptions.',
     'https://www.qrisp.eu/')
comp('Qmod','high-level quantum modeling language','native syntax and Python embedding',
     'Functional models compiled and synthesized under circuit-property constraints.',
     'Native and Python forms are the same language, not two comparators; graphical support is a subset.',
     'Offer explicit resource constraints and inspectable synthesized circuits.',
     'https://docs.classiq.io/qmod-reference')
comp('BQSKit','quantum compiler and synthesis framework','Python',
     'Circuit IR, numerical synthesis, extensible compiler passes and distributed compilation.',
     'A compiler framework does not itself supply all algorithm or hardware services.',
     'Adopt pass contracts, cost functions and verification for circuit optimization.',
     'https://bqskit.readthedocs.io/en/latest/')
comp('TKET / pytket','compiler toolkit and SDK','C++ core; Python interface',
     'Native-gate compilation, placement, routing and multiple backend integrations.',
     'Device feasibility must be checked after transformations; optimization is workload-dependent.',
     'Treat routing and native-gate legality as independently tested compiler stages.',
     'https://docs.quantinuum.com/tket/')
comp('Amazon Braket SDK','cloud SDK / managed-service interface','Python',
     'Common access to simulators, gate devices, analog hardware and hybrid workflows.',
     'Provider availability, charges and hardware characteristics are service-specific.',
     'Add explicit job state, shot budgets and result provenance, with simulator-only defaults.',
     'https://docs.aws.amazon.com/braket/latest/developerguide/what-is-braket.html')
comp('Qibo','quantum computing framework','Python',
     'Backend drivers, algorithm examples, tomography and hybrid research workflows.',
     'The fetched stable docs identify v0.3.6; optional components are not all core functions.',
     'Provide backend-independent experiments with numerical reference results.',
     'https://qibo.science/qibo/stable/')
comp('Qulacs','quantum circuit simulator / SDK','C++ with Python interface',
     'Noisy and parametric circuit simulation, density matrices and circuit-learning examples.',
     'Simulation speed and scale depend on circuit/state representation and hardware.',
     'Benchmark specialized simulation kernels against strong classical baselines.',
     'https://docs.qulacs.org/en/latest/')
comp('Yao.jl','quantum framework / embedded block DSL','Julia',
     'Composable quantum blocks, algorithm construction and GPU/tensor-network companion packages.',
     'The repository calls itself early-release beta; companion packages are distinct dependencies.',
     'Use compositional circuits and expose backend extensions without overstating core scope.',
     'https://github.com/QuantumBFS/Yao.jl',
     'docs.yaoquantum.org/stable was inaccessible to the browser tool; official repository supplied evidence.')
comp('QuantumOptics.jl','open-system scientific simulation framework','Julia',
     'Hilbert-space-aware objects, master equations, trajectories, steady states and correlations.',
     'This is a scientific simulator, not a gate-QPU programming language.',
     'Separate open-system dynamics from gate circuits while sharing typed observables.',
     'https://docs.qojulia.org/')
comp('Strawberry Fields','photonic quantum SDK / embedded circuit DSL','Python',
     'Photonic circuits, simulation backends and differentiable program optimization.',
     'The stable URL redirects to 0.24.0-dev documentation; current cloud access was not verified.',
     'Treat continuous-variable photonics as a distinct model with truncation checks.',
     'https://strawberryfields.readthedocs.io/en/latest/',
     'strawberryfields.ai redirected to Xanadu; documentation used instead.')
comp('Perceval','discrete-variable photonic SDK','Python with compiled backends',
     'Linear-optics components, Fock states, source-error models and Quandela QPU interface.',
     'Discrete-photon and continuous-variable models are not interchangeable.',
     'Add a separate photonic adapter and preserve heralding and postselection statistics.',
     'https://github.com/Quandela/Perceval',
     'v1.0 docs were readable and linked v1.3; v1.3 retrieval failed, so current official repository used.')
comp('Pulser','neutral-atom pulse SDK / DSL','Python',
     'Device-constrained pulse sequences and analog neutral-atom simulation.',
     'Pulse validity depends on a particular device, not just a circuit equivalence relation.',
     'Validate geometry, timing and waveform constraints in an analog execution layer.',
     'https://pulser.readthedocs.io/en/stable/')
comp('Qadence','differentiable digital-analog framework','Python; PyTorch',
     'Block-based programs, symbolic parameters, analog generators and differentiable workflows.',
     'Model/backend support must be checked per operation; not every pulse is differentiable.',
     'Keep digital/analog differentiation contracts explicit and testable.',
     'https://pasqal-io.github.io/qadence/latest/')
comp('Bloqade','neutral-atom SDK and compiler DSL ecosystem','Python; multiple components',
     'Analog/digital experiment workflows, atom movement, noisy near-Clifford simulation and QEC tools.',
     'The current site exposes multiple packages and target products, not one universal language.',
     'Model hardware geometry, losses and logical execution rather than assuming ideal gates.',
     'https://bloqade.quera.com/')
comp('ProjectQ','embedded quantum DSL and compiler framework','Python',
     'Extensible compiler pipeline and quantum algorithm execution on simulators/devices.',
     'The fetched docs are a development build; individual backend support must be rechecked.',
     'Make compiler extensions small, inspectable and covered by functional tests.',
     'https://projectq.readthedocs.io/en/latest/')
comp('Guppy','statically typed embedded quantum language','Python',
     'Linear qubit ownership, measurement-dependent control, metaprogramming and fault-tolerant extensions.',
     'Guppy, its FT extension and Selene emulator have separate responsibilities.',
     'Make the typed quantum language, runtime handles and execution model agree.',
     'https://docs.quantinuum.com/guppy/')

for sid, title, url in [
 ('R01','Gidney and Ekera: fault-tolerant factoring resource estimate','https://arxiv.org/abs/1905.09749'),
 ('R02','Reiher et al.: Elucidating Reaction Mechanisms on Quantum Computers','https://arxiv.org/abs/1605.03590'),
 ('R03','Low and Chuang: Hamiltonian Simulation by Qubitization','https://arxiv.org/abs/1610.06546'),
 ('R04','McClean et al.: Barren plateaus in quantum neural network training landscapes','https://arxiv.org/abs/1803.11173'),
 ('R05','Huang et al.: Quantum advantage in learning from experiments','https://arxiv.org/abs/2112.00778'),
 ('R06','Takagi et al.: Fundamental limits of quantum error mitigation','https://arxiv.org/abs/2109.04457'),
 ('R07','Acharya et al.: Quantum error correction below the surface code threshold','https://arxiv.org/abs/2408.13687'),
 ('R08','Lubinski et al.: Application-Oriented Performance Benchmarks for Quantum Computing','https://arxiv.org/abs/2110.03137'),
 ('R09','Pironio et al.: Random Numbers Certified by Bells Theorem','https://arxiv.org/abs/0911.3427'),
 ('R10','Hietala et al.: A Verified Optimizer for Quantum Circuits','https://arxiv.org/abs/1912.02250'),
 ('R11','Wehner, Elkouss and Hanson: Quantum internet, a vision for the road ahead','https://repository.tudelft.nl/record/uuid%3Af2b04894-ece4-4443-984b-71ddb11a5ab2'),
 ('R12','Microsoft Quantum resource estimator','https://learn.microsoft.com/en-us/azure/quantum/intro-to-resource-estimation'),
 ('R13','QIR specification','https://github.com/qir-alliance/qir-spec'),
 ('R14','Stim: official stabilizer/QEC simulator','https://github.com/quantumlib/Stim'),
 ('R15','PyZX: official diagrammatic optimization library','https://github.com/zxcalc/pyzx'),
 ('R16','Mahadev: Classical Verification of Quantum Computations','https://arxiv.org/abs/1804.01082'),
]:
    source(sid,title,url,'Primary paper, author repository, specification, or official technical documentation')

local_sources = [
 ('E01','Public quantum module','lib/agent/quantum.esk','Full exported surface and wrapper implementations inspected; opaque integer handles, eight gate operations, Z expectation, measurement, Hamiltonian/VQE/QGT APIs.'),
 ('E02','Quantum C shim','lib/agent/c/agent_quantum.c','Dense-state/VQE shim, handle tables, conditional Moonlab implementation and unavailable stubs inspected; QGT limited to four parameters.'),
 ('E03','Build configuration','CMakeLists.txt','Quantum option OFF by default; exact Moonlab GIT_TAG is 4bf83a6c47e7f1c53db14fcb2a61053499fd82df. Nearby comments mention older revisions.'),
 ('E04','Quantum reference','docs/reference/agent/quantum.md','Public usage, ideal VQE adjoint bridge and availability descriptions; prose alone is not runtime evidence.'),
 ('E05','Moonlab integration design','docs/design/MOONLAB_INTEGRATION.md','Mixed shipped header/historical design. Mentions broader Moonlab tensor-network and decoder APIs, many explicitly deferred from Eshkol binding.'),
 ('E06','Qubit type definition','lib/types/hott_types.cpp','Qubit is registered as a linear type with Int64 representation; public state/gate wrappers still use state handles and integer indices.'),
 ('E07','Linearity boundary regression','tests/typesystem/qubit_alias_laundering_not_enforced_test.esk','Fixture describes caught immediate-lambda aliasing and a remaining separately defined unannotated-return boundary; filename alone is misleading.'),
 ('E08','Quantum surface regression','tests/quantum/quantum_surface_coverage_test.esk','Source test covers exposed gates and molecular construction; not executed in this research stream.'),
 ('E09','VQE adversarial AD regression','tests/quantum/vqe_ad_adversarial.esk','Source contains independent finite-difference checks around the ideal VQE bridge; not executed in this research stream.'),
 ('E10','Entropy wrapper','lib/quantum/quantum_rng_wrapper.c','Build-dependent Moonlab or classical source; a simulated Bell statistic is not device-independent physical entropy certification.'),
]
for sid,title,path,note in local_sources:
    source(sid,title,f'https://github.com/tsotchke/eshkol/blob/{BASE}/{path}',
           'Inspected local source snapshot',note)
    sources[sid]['local_path'] = str(Path('Z:/Tools/eshkol/source') / path)

targets=[]
def group(workstream, phase, default_class, text):
    for line in text.strip().splitlines():
        title, baseline, classification, evidence, refs, acceptance, deps = [x.strip() for x in line.split('|')]
        n=len(targets)+1
        targets.append(dict(id=f'T{n:03}', workstream=workstream, phase=phase,
          title=title, baseline_state=baseline, target_class=classification or default_class,
          baseline_evidence=evidence, source_ids=refs.split(','),
          acceptance_criteria=acceptance, dependencies=[] if deps=='-' else deps.split(','),
          delivery_status='planned', completed_in_this_research=False))

group('Reliable foundations','P0','foundation',r'''
Runtime quantum capability query | partial | foundation | Build flag and unavailable stubs exist; no unified Scheme capability report found. | E01,E02,E03,C01 | A query reports build flag, simulator backend, ABI, supported operations and derivative modes; ON/OFF CI asserts each field without executing an unavailable operation. | -
Reproducible optional quantum packages | partial | foundation | Opt-in source integration exists; this stream has not validated packaged ON builds. | E03,C09 | Clean Windows, Linux and macOS jobs install a pinned ON package, run Bell/VQE smoke tests, and verify OFF builds return documented unavailability. | T001
Honest execution and entropy provenance | partial | foundation | Source labels exist but some prose conflates simulated Bell tests with physical certification. | E04,E05,E10,R09 | Every result identifies classical simulator or named physical device; entropy output reports actual source and fallback events; no simulation is labelled device-independent certified entropy. | T001
Moonlab ABI and revision manifest | partial | foundation | CMake pins a commit while nearby comments/reference text mention other revisions. | E02,E03,E05 | Generated manifest records exact dependency commit and ABI; intentional mismatches fail before allocation; release docs are generated from the same manifest. | T001
Generational quantum handles | partial | foundation | Opaque integer tables and destroy helpers exist; stale-handle reuse needs a specified contract. | E01,E02,C25 | Allocate/free/reallocate fuzzing never lets a stale handle access a new state; double destruction is defined; invalid handles produce catchable errors under sanitizers. | T004
Typed qubit ownership through public APIs | partial | foundation | Linear Qubit exists, but state-handle plus integer-index wrappers do not establish full resource typing. | E01,E06,E07,C05,C25 | Compile-fail tests cover duplicate use, escaping aliases and interprocedural returns; valid scoped gates compile; runtime state ownership agrees with the static qubit identity. | T005
Uniform quantum error contract | partial | foundation | C shim uses error codes/NaN and wrappers translate selected failures. | E01,E02,C24 | Invalid counts, indices, angles, destroyed handles and allocation failures all produce documented catchable errors; no invalid case silently returns a plausible numeric result. | T005
Dense ideal-state correctness gate | implemented | foundation | Dense state creation, gates, measurement and Z expectation have public wrappers and source tests. | E01,E02,E08,C16 | Exhaustively compare all exposed 1- and 2-qubit gates with independent complex matrices; norm and expectation errors stay below 1e-12 in float64 fixtures. | T002
Seeded simulator replay | absent | foundation | Measurement wrapper uses a process entropy context; caller-supplied seed is not exposed. | E01,E02,C07 | A seed plus serialized program reproduces samples on the same version; different seeds pass distribution checks; deterministic replay is never advertised as physical entropy. | T003
Backend conformance ledger | unknown | foundation | Source tests exist but no fresh complete native/VM/browser quantum matrix was run here. | E03,E08,E09,R08 | Publish operation-by-engine supported/refused/untested statuses tied to executable tests; zero unsupported cases silently succeed; include release hashes and test logs. | T002,T007
''')

group('Interoperability','P1','parity',r'''
OpenQASM 3 export | absent | parity | No public circuit export API found in the inspected module/shim. | E01,E02,C02 | Export a declared gate/measurement subset; an independent OpenQASM parser accepts 100 fixtures and simulation probabilities match within 1e-10. | T015
OpenQASM 3 import | absent | parity | No public parser/import adapter found in the quantum surface. | E01,C02 | Parse the declared subset with source locations; reject pulse/unsupported dynamic constructs explicitly; imported Bell, QFT and adaptive fixtures match reference behavior. | T015
QIR target profile | absent | parity | LLVM compilation alone is not a QIR emitter; none found in inspected quantum interface. | E01,E02,R13 | Emit a named QIR profile with verified LLVM IR and required metadata; run at least 20 programs on an independent compatible executor. | T015,T025
Quil interchange adapter | absent | parity | No public Quil serializer/parser found in the quantum interface. | E01,C03 | Round-trip gates, classical memory and measurement for a documented Quil subset; preserve bit order and reject pulse annexes until implemented. | T015,T017
Typed circuit intermediate representation | absent | foundation | Current public API mutates opaque simulator state directly rather than exposing circuit objects. | E01,E02,C12,C24 | Represent qubits, gates, parameters, measurements and dependencies with a versioned schema; validation catches dangling wires and parameter-type errors before execution. | T006,T007
Portable observable and Hamiltonian interchange | partial | parity | Pauli-string Hamiltonian construction exists; no general import/export schema found. | E01,C10,C08 | Serialize arbitrary Pauli sums, units, constants and wire order; round-trip 50 fixtures with identical coefficients and expectation values within 1e-12. | T015
Shot-result and bit-order schema | absent | parity | Single-qubit measurement returns an integer; no portable batch-result schema found. | E01,C14,C07 | Specify endianness, register mapping, counts, shots, uncertainty and provenance; adapters pass asymmetric-bitstring tests that detect reversal errors. | T003,T015
Backend plugin contract | absent | parity | Moonlab is compiled into the shim; no public quantum backend plugin contract found. | E02,E03,C09,C15 | Two independent simulator adapters implement capability discovery, execute, sample and observe; unsupported operations fail consistently without implicit backend substitution. | T001,T015,T017
PennyLane device integration | absent | parity | No PennyLane-facing device plugin found in inspected Eshkol quantum surface. | E01,C08 | A separately versioned plugin passes its declared device-conformance tests and a 20-case expectation/gradient comparison without copying parameters incorrectly. | T018,T041
Quantum array interchange | absent | parity | Quantum state is opaque; VQE parameters are marshalled through a dedicated context. | E01,E02,C17,C16 | Exchange complex amplitudes and parameter arrays through a documented C/Python boundary; shape, ownership, dtype and lifetime tests catch aliasing and precision loss. | T005,T017
''')

group('Circuit language and compilation','P1','parity',r'''
Complete practical gate vocabulary | partial | parity | H, X, Y, Z, CNOT, RX, RY and RZ are exposed; many standard gates are not. | E01,E05,C02,C12 | Add S, T, CZ, SWAP and controlled rotations or verified decompositions; compare matrices including global-phase conventions and invalid-wire cases. | T008,T015
Controlled and adjoint circuit combinators | absent | parity | No general circuit modifier API is exposed. | E01,C03,C04 | Controlled and inverse transforms compose over supported circuits; random unitary fixtures satisfy U-dagger times U within 1e-10 and measurement inversion is rejected. | T015,T021
Safe automatic uncomputation | absent | advanced | Classical region memory does not implement quantum uncomputation. | E01,E06,C05,C10 | For a declared reversible subset, compiler-generated cleanup returns ancillas to zero and preserves output coherence; unsafe erasures fail static checking. | T006,T022
Ancilla allocation and reuse planner | absent | parity | State allocation exists, but no circuit-level ancilla liveness planner is exposed. | E01,C04,C10 | Report peak live ancillas and verified reuse intervals; transformed circuits match references and never reuse an entangled live ancilla. | T015,T023
Measurement-dependent circuit control | partial | parity | Host Scheme can branch on measurement; portable in-circuit feedback representation is not established. | E01,C02,C25 | Represent mid-circuit measurement, reset, conditionals and bounded loops; teleportation and repeat-until-success fixtures match simulator/reference distributions. | T015,T017
Topology-aware routing | absent | parity | No target connectivity/routing abstraction found in public quantum interface. | E01,C13 | Route 50 fixed circuits to line/grid/device graphs; every two-qubit operation is legal, semantics preserved, and swap overhead is reported. | T015,T018
Native-gate synthesis | absent | parity | Direct simulator gate calls do not form a target synthesis pipeline. | E01,C11,C12 | Compile arbitrary tested 1-/2-qubit unitaries to a declared basis with chosen error budget; verify achieved error and report gate count and synthesis time. | T021,T026
Local circuit optimization passes | absent | parity | No quantum pass-manager API found in exposed surface. | E01,C12,C24 | Cancellation, rotation fusion and commuting passes preserve 100 random small-circuit unitaries within 1e-10; publish before/after depth and count. | T015,T022
ZX optimization interoperability | absent | advanced | No ZX diagram adapter found in quantum surface. | E01,R15 | Export/import a defined Clifford+T subset to PyZX; certify equality or clearly report unverifiable cases; record T-count and extraction failures. | T011,T028
Proof-checked unitary rewrite kernel | absent | advanced | General Eshkol proof types do not constitute verified quantum rewrite rules. | E01,R10 | Mechanize semantics and prove at least five implemented rewrites; generated optimizer only uses checked rules, with numeric tests as secondary evidence. | T028
''')

group('Simulation and noise','P2','parity',r'''
Density-matrix simulator adapter | absent | parity | Public binding exposes dense pure states, not density matrices. | E01,E02,C07,C16 | Bell-state partial trace gives maximally mixed marginals; trace and positivity tests pass; small noisy circuits match an independent density simulator within 1e-10. | T018,T020
Validated Kraus and noise channels | absent | parity | No public channel/noise-model constructor found. | E01,C07,C16 | Implement depolarizing, dephasing and amplitude damping; validate completeness within tolerance and reject non-physical channel inputs. | T031
Quantum trajectory simulation | absent | parity | No trajectory ensemble API found in public binding. | E01,C18,C07 | Ensemble averages converge to density-matrix references with declared confidence intervals; logs record trajectory count, seeds and sampling error separately from solver error. | T009,T032
Lindblad time evolution | absent | advanced | Moonlab integration references other engines; public module has no open-system solver. | E01,E05,C18 | Damped two-level analytic solutions and independent solver fixtures agree under a documented tolerance; step refinement verifies convergence and trace preservation. | T031,T032
Stabilizer simulator backend | absent | parity | No tableau/stabilizer operations exposed in the binding. | E01,R14,C04 | Execute Clifford circuits through a tested adapter; match exact small-system probabilities and run a 1000-qubit stabilizer fixture without exponential state storage. | T018,T021
Tensor-network simulation binding | absent | advanced | Design names Moonlab CA-MPS/TDVP APIs but public module does not bind them. | E01,E05,C17 | Expose bond/truncation controls with discarded-weight diagnostics; match small exact references and show convergence as bond dimension increases. | T018,T020
GPU-resident quantum state execution | unknown | advanced | Integration config disables optional Moonlab accelerators; general Eshkol GPU support does not prove this path. | E02,E03,E05,C09 | Demonstrate actual device-resident gate execution and synchronization tests; report memory, correctness and end-to-end speed versus CPU including transfers. | T018,T020
Distributed quantum simulation | absent | advanced | No multi-node quantum state partitioning API found in exposed binding. | E01,C09 | Run one reproducible distributed state-vector fixture on two nodes; compare checksum/observables to a serial reference and report communication-inclusive scaling. | T037
Qudit state and gate support | absent | advanced | Public API uses qubit counts and binary gates. | E01,C12 | Support dimension-3 registers and mixed-dimension validation; qutrit Fourier/controlled-shift fixtures preserve norm and match independent matrices. | T015,T020
Continuous-variable photonic model | absent | advanced | Public API is qubit state-vector/VQE, with no Fock/Gaussian abstraction. | E01,C19 | Implement or adapt Gaussian/Fock operations; coherent/squeezed-state observables agree with references and cutoff-convergence reports quantify truncation error. | T018,T020
''')

group('Differentiation and hybrid training','P2','parity',r'''
Preserve ideal VQE adjoint correctness | implemented | foundation | vqe-energy custom VJP and vqe-gradient plus adversarial source tests exist. | E01,E02,E09,C08 | On quantum-enabled releases, compare AD, direct VJP and central differences over random nonstationary points; maximum gradient error below 1e-6 and no unexplained zero gradients. | T002,T010
User-defined variational ansatz | absent | parity | Public VQE uses a fixed one-layer hardware-efficient ansatz. | E01,E02,C08,C22 | Users construct 3 distinct trainable circuit families; arbitrary parameter mapping is preserved and expectations/gradients match independent references. | T015,T022,T041
Shot-based parameter-shift gradients | partial | parity | A native analytic fallback is documented, but user-controlled shot-based gradient execution is not exposed. | E01,E02,C15 | Estimate eligible gate derivatives with shot budgets and confidence intervals; 95% interval coverage is checked over repeated seeds against analytic small cases. | T017,T042
General adjoint circuit differentiation | partial | advanced | Adjoint path exists for default ideal VQE, not arbitrary exposed circuit objects. | E01,E02,C08,C17 | Differentiate user circuits with 1, 10 and 100 parameters; match parameter-shift references and publish peak memory/runtime as depth and width vary. | T042
Second-order quantum derivative contract | unknown | advanced | Arbitrary-order host AD does not establish arbitrary higher-order derivatives through the opaque VQE primitive. | E01,E02,E09,C08 | Specify supported Hessian/JVP compositions; test symmetry and finite differences; unsupported nesting raises a named error rather than returning zero. | T041,T044
General quantum geometric tensor | partial | advanced | vqe-qgt currently assumes four parameters and returns 4x4 entries. | E01,E02,C08 | Accept variable parameter counts; compare to state-derivative references, test symmetry/positive semidefiniteness, and state gauge and regularization conventions. | T042,T044
Differentiation through noisy channels | absent | advanced | Public VQE bridge describes ideal-state energy and gradients. | E01,E02,C22 | Differentiate at least two parameterized CPTP channels; compare with density-matrix finite differences and distinguish stochastic estimator variance from numerical bias. | T032,T044
Batched parameter sweeps and gradients | absent | parity | Public VQE evaluator takes one parameter vector per call. | E01,E02,C09,C08 | Batch 32 independent parameter sets; outputs match serial runs, ordering is stable, and timing includes marshaling and device transfers. | T042,T044
Restartable hybrid optimizer workflow | partial | parity | vqe-optimize offers an iteration count with a fixed default optimizer; general state/checkpoint control is not exposed. | E01,C14,C22 | Export optimizer state, RNG and parameters; an interrupted run resumes to the same seeded result, with loss/gradient/shot histories preserved. | T009,T043,T048
Quantum derivative refusal and coverage manifest | partial | foundation | Dedicated VQE checks exist; generic derivative coverage across new backends is not established. | E02,E09,C08,R04 | Every operation/backend/order combination is supported or explicitly refused; negative tests cover measurement branches, unsupported channels and nested AD without silent zeros. | T010,T041
''')

group('Scientific algorithm libraries','P2','parity',r'''
General chemistry Hamiltonian import | partial | parity | Fixed H2/LiH/H2O constructors and arbitrary Pauli sums exist. | E01,C08,R02 | Import a documented electronic-integral or Pauli dataset with basis, geometry and units; validate at least five small molecules against independent diagonalization. | T016
Symmetry tapering and sector selection | absent | advanced | Public Hamiltonian construction accepts an HF bitmask but no tapering transformation. | E01,C08 | Identify supported commuting symmetries; reduced and unreduced sector spectra agree on small fixtures; invalid sector requests are rejected. | T016,T051
Excited-state and spectral workflows | absent | advanced | Public helpers focus on ground-state reference/VQE energy. | E01,C16 | Compute two excited energies and overlaps for small Hamiltonians with declared method; independent eigensolver agreement and orthogonality checks are mandatory. | T042,T051
Quantum phase estimation library | absent | parity | No public QPE combinator found in quantum module. | E01,C04 | Known eigenphase fixtures recover each phase to the requested bits; account for ancillas, controlled-evolution synthesis and failure probability. | T022,T024,T025
Amplitude estimation library | absent | parity | No exposed amplitude-estimation routine found. | E01,C15 | Estimate a set of known Bernoulli amplitudes; observed confidence coverage matches declared guarantees and oracle-call counts include state preparation. | T022,T025
QAOA with explicit classical baseline | absent | parity | Public quantum layer exposes VQE but no QAOA constructor. | E01,C07,C22 | Small MaxCut instances match exhaustive optimum bounds; report approximation ratio, shots and total time against named classical solvers without claiming generic advantage. | T042,T049
Hamiltonian time-evolution circuits | absent | parity | Gate wrappers do not expose a compositional simulation algorithm library. | E01,C18,R03 | Trotter/Suzuki evolution matches matrix exponentials on small models; refinement demonstrates expected convergence and error budgets include synthesis. | T016,T022,T027
Block encoding and quantum signal processing | absent | advanced | No public block-encoding/QSP construction API found. | E01,C10,R03 | For bounded small matrices, verify top-left block encoding and polynomial transformation numerically; report normalization, oracle and approximation costs. | T022,T024,T027
Tomography and classical-shadow experiments | absent | parity | Z expectation exists; reconstruction/measurement-design workflows are not exposed. | E01,C15 | Reconstruct specified small states/observables with uncertainty intervals, record shot counts and test coverage over independent seeds. | T017,T021,T032
Discrete-photon experiment adapter | absent | advanced | No public linear-optics/Fock-state interface found. | E01,C20 | Run Hong-Ou-Mandel and a heralded small-circuit fixture through an adapter; preserve raw/postselected counts and model source/detector errors explicitly. | T018,T017
''')

group('Hardware execution and control','P3','parity',r'''
First physical gate-QPU adapter | absent | parity | Exposed Moonlab interface is a classical simulator; no QPU submission path found. | E01,E02,C06,C14 | Submit a validated small program to one named physical device, record job ID/calibration/shots and reproduce retrieval; mocked tests never count as hardware validation. | T011,T018,T017
Device topology and calibration snapshot | absent | parity | No public physical-device model found. | E01,C07,C13 | Version connectivity, basis gates, durations and calibration timestamp; compilation rejects incompatible circuits and records exactly which snapshot it used. | T026,T061
Durable asynchronous quantum jobs | absent | parity | Local simulator calls are synchronous and no quantum job handle API is exposed. | E01,C14 | Submission, polling, cancellation and restart recovery are idempotent; mocked failure tests cannot double-submit a chargeable job and preserve final provenance. | T061
Execution and shot budget enforcement | absent | foundation | No QPU cost/shot budget API found. | E01,C14 | A preflight policy caps shots, retries and estimated spend; tests show retries cannot exceed caps; actual usage reconciles against recorded jobs. | T063
Pulse and timing representation | absent | advanced | No public pulse/frame/waveform representation found. | E01,C02,C03 | Represent frames, durations and waveforms with units; enforce target sample/timing limits; round-trip a documented pulse subset with no unit conversion ambiguity. | T015,T062
Neutral-atom geometry and waveform adapter | absent | advanced | Public API has no atom positions, blockade model or analog program object. | E01,C21,C23 | Validate inter-atom spacing, channel limits and waveform schedules against one device schema; compare Rabi and small Ising simulations with references. | T065,T034
Digital-analog program composition | absent | advanced | Quantum module exposes gates and fixed VQE, not mixed analog blocks. | E01,C22 | Compose gate and Hamiltonian-evolution blocks with explicit units; ideal small-system results and parameter derivatives match an independent solver. | T066,T044
Hardware feedback latency contract | absent | advanced | Host branching does not guarantee device-resident or bounded-latency feedback. | E01,C25,C09 | Measure control round-trip distributions on one backend and declare limits; unsupported feedback is rejected, with no claim of hard real-time bounds from arena allocation. | T025,T061
Readout mitigation with uncertainty | absent | parity | No public physical measurement-calibration model found. | E01,C06,R06 | Calibrate a small readout matrix; report corrected estimates, uncertainty and conditioning; held-out tests show when correction worsens results. | T061,T059
Error mitigation under a fixed budget | absent | advanced | No ZNE/PEC experiment framework exposed. | E01,C22,R06 | Implement one mitigation protocol with raw and corrected outputs; compare mean-squared error at equal total shots, including calibration overhead and failure regimes. | T032,T064,T069
''')

group('Error correction and resource estimation','P3','advanced',r'''
Logical-versus-physical qubit model | absent | foundation | Qubit type and simulator count do not encode code distance or logical operations. | E01,E06,R12,C25 | Resource objects distinguish logical qubits, physical qubits, code distance and gate level; mixing incompatible levels fails validation. | T015,T006
Stabilizer-code construction library | absent | parity | No public QEC code constructor found. | E01,R14,C23 | Generate repetition and distance-3/5 surface-code circuits; ideal syndromes and every single-Pauli injection match independently checked expectations. | T035,T071
Detector error model interchange | absent | advanced | No detector/syndrome schema exposed. | E01,R14 | Export detector models for declared noise assumptions; an independent decoder consumes them and fault-injection checks preserve detector/observable mappings. | T032,T072
Decoder plugin and throughput harness | absent | advanced | Moonlab design names decoder registries; Eshkol module does not expose them. | E01,E05,R14 | Two decoders consume the same syndrome batches; publish logical error, latency percentiles and throughput on fixed seeds/hardware with no unmeasured real-time claim. | T073
Logical error scaling experiments | absent | advanced | No public code-distance/noise sweep workflow found. | E01,R07,R14 | Sweep at least three code distances and noise rates with confidence intervals; fit suppression only in measured regimes and expose insufficient-statistics outcomes. | T072,T074
Clifford+T synthesis error budget | absent | advanced | No fault-tolerant gate/resource compiler surface found. | E01,C09,R12 | Lower supported rotations to Clifford+T with a declared synthesis bound; verify small unitaries and account separately for algorithmic, synthesis and physical error. | T027,T071
Magic-state factory resource model | absent | advanced | No public distillation/factory model found. | E01,R01,R12 | Compute physical-qubit/time estimates for named protocols under explicit input errors; reproduce a published small reference table and include parameter sensitivity. | T076
Logical scheduling and lattice-surgery plan | absent | advanced | No logical layout/schedule representation found. | E01,R12,C25 | Compile a small logical workload into a legal schedule for one architecture model; report space-time volume and validate resource conflicts. | T071,T076,T077
Fault-tolerant resource estimation adapter | absent | parity | No public estimator bridge found; language compilation alone is insufficient. | E01,R12,C01 | Export a supported workload to an external estimator; preserve assumptions and compare physical qubits/runtime across at least three hardware/code scenarios. | T013,T076
Cross-layer resource and error budget ledger | absent | advanced | No unified algorithm-to-physical cost ledger found. | E01,R01,R02,R12 | A chemistry or phase-estimation example records precision, success probability, logical operations, factories and physical resources without double-counting overhead. | T054,T077,T079
''')

group('Developer experience and reproducible evidence','P0','foundation',r'''
Executable quantum documentation | partial | foundation | Quantum examples/source tests exist but this stream did not execute every published example on every target. | E04,E08,E09,C24 | CI extracts all quantum examples, runs declared supported targets, and marks external hardware prerequisites; wrong expected outputs block release. | T002,T010
Quantum-aware diagnostics | partial | foundation | Wrapper error messages exist; circuit source mapping and richer diagnostics are absent. | E01,E02,C25 | Invalid wires, nonunitary gates, impossible ownership and unsupported target features report source spans and actionable reasons in 30 negative fixtures. | T007,T015
Circuit and state visualization | absent | parity | No public circuit object/visualizer found in the binding. | E01,C20,C07 | Render circuits with parameter/measurement labels and small-state plots; snapshots preserve wire order and explicitly mark simulator-only state access. | T015,T017
Quantum profiling trace | absent | foundation | No unified quantum compile/execute/transfer/shot trace exposed. | E01,C09,R08 | Emit machine-readable spans for compilation, marshaling, transfers, execution and sampling; totals reconcile with wall time within documented instrumentation overhead. | T018
Pinned reproducible experiment bundles | partial | foundation | Dependency pinning exists; no universal experiment/result bundle schema was established here. | E03,R08 | Bundle program, dependency hashes, seeds, device snapshot, environment, raw results and analysis; a clean runner reproduces a seeded simulator experiment. | T004,T009,T017
Differential quantum fuzzing | partial | foundation | Source quantum tests exist; broad cross-tool randomized differential evidence is not established. | E08,E09,C07,C16 | Generate 1000 small valid circuits and compare independent engines with phase/shot-aware tolerances; minimize every mismatch into a retained regression. | T008,T018,T085
Application benchmark protocol | absent | foundation | Existing quantum microbenchmarks do not establish a 25-system application ranking. | E01,R08 | Freeze workload families, precision, hardware, seeds and time-to-solution metrics before measurement; publish failures and tuning budgets alongside successes. | T084,T085
Independent replication package | unknown | foundation | No independent replication of this proposed comparative protocol has occurred. | E08,R08 | A second team or isolated environment reruns the published bundle; discrepancies are logged and resolved before any leadership claim. | T087
Quantum extension compatibility governance | partial | foundation | General project contribution rules exist; quantum plugin/version promises are not established. | E03,C24,R13 | Publish semantic-version and deprecation rules; two plugin versions run through a compatibility matrix and a deliberately incompatible ABI fails clearly. | T004,T018
Scientific claim-to-evidence registry | partial | foundation | Existing docs mix implementation, historical design and performance/physical claims. | E04,E05,R08,R09 | Every public quantum capability claim links to source, passing test or measured report and declares scope; simulation, hardware, projection and open research are distinct fields. | T003,T081,T087
''')

group('Open scientific research programs','R','open research',r'''
Fault-tolerant chemistry beyond toy molecules | absent | open research | Public molecular constructors/VQE do not demonstrate practical fault-tolerant chemistry. | E01,R02,R03 | Reproduce one small active-space reference end to end, then publish a larger-case resource estimate with basis/truncation/error budgets; independent chemistry review is required before a scientific claim. | T051,T058,T080
End-to-end advantage for a materials workload | unknown | open research | No matched classical-versus-quantum advantage experiment was established here. | E01,R03,R08 | Pre-register a meaningful observable, precision and total-cost metric; beat tuned classical baselines with reproducible evidence, or publish the null result. Simulation speed alone cannot close this target. | T057,T087,T088
Trainability-aware variational compiler | absent | open research | Native AD supplies derivatives but does not remove barren plateaus or shot complexity. | E01,R04,C08 | On predefined circuit families, predict and measure gradient-variance scaling; compare initialization/ansatz strategies at equal budgets and publish counterexamples to claimed improvements. | T043,T044,T087
Mitigation with defensible cost guarantees | absent | open research | No Eshkol result establishes mitigation benefit under general noise and equal sampling cost. | E01,R06 | Derive assumptions and overhead bounds for one method, then test equal-cost error across increasing depth; explicitly report regimes where mitigation loses. | T070,T087
Useful logical computation beyond memory experiments | absent | open research | No logical hardware execution path or result exists in the inspected binding. | E01,R07,C25 | Execute a logical algorithm with repeated correction and compare to a defined physical implementation at equal task error; distinguish code-memory success from useful algorithmic advantage. | T061,T075,T078
Verified adaptive quantum compilation | absent | open research | No proof links the quantum interface, dynamic control and backend lowering end to end. | E01,R10,R16 | Formalize a bounded adaptive subset and prove selected lowerings preserve measurement semantics; publish assumptions and proof artifacts, with adversarial tests for unproved boundaries. | T025,T030,T013
Learning from coherent experimental quantum data | absent | open research | Simulator-generated datasets do not establish coherent-data experimental advantage. | E01,R05 | Reproduce a small coherent-copy learning task with a physical-data protocol and explicit classical access model; compare sample complexity and account for state preparation/readout. | T061,T059,T087
Physical device-independent entropy integration | absent | open research | Current simulated Bell/entropy layer is not a loophole-controlled physical Bell experiment. | E02,E10,R09 | Consume authenticated physical experiment records with stated independence assumptions, finite-size entropy bounds and extractor parameters; simulation-only mode must be incapable of claiming certification. | T003,T063,T090
Distributed quantum network programming | absent | open research | No quantum network, entanglement-distribution or remote-state API is exposed. | E01,R11 | Implement a simulator protocol with loss, fidelity and classical latency, then validate one hardware-linked entanglement task; report end-to-end success rates and assumptions. | T018,T063,T071
Reproducible frontier resource co-design | absent | open research | No end-to-end hardware/code/algorithm Pareto study is exposed in Eshkol. | E01,R01,R12,R02 | Reproduce a published resource scenario, vary algorithm/code/hardware assumptions, and publish non-dominated cost/error tradeoffs with sensitivity and independent checks; estimates remain projections. | T079,T080,T088
''')

assert len(competitors)==25 and len({c['name'].casefold() for c in competitors})==25
assert len(targets)==100 and len({t['title'].casefold() for t in targets})==100
ids={t['id'] for t in targets}
for t in targets:
    assert t['baseline_state'] in {'implemented','partial','absent','unknown'}
    assert t['target_class'] in {'foundation','parity','advanced','open research'}
    assert t['source_ids'] and all(s in sources for s in t['source_ids'])
    assert all(d in ids and d != t['id'] for d in t['dependencies'])
    assert len(t['acceptance_criteria'].split()) >= 15

def check_dag(node, visiting, visited):
    if node in visited: return
    assert node not in visiting, f'cycle at {node}'
    visiting.add(node)
    for d in targets[int(node[1:])-1]['dependencies']: check_dag(d,visiting,visited)
    visiting.remove(node); visited.add(node)
for node in ids: check_dag(node,set(),set())

meta = dict(date=DATE, eshkol_source_commit=BASE, comparator_count=25, target_count=100,
            universal_missing_claim=False, all_targets_implemented=False,
            comparison_method='Primary-source qualitative capability comparison; no comparative execution benchmark was performed.',
            baseline_scope='Inspected Eshkol public quantum module/shim and linked source snapshot, not all Moonlab capabilities or all Eshkol modules.',
            baseline_states=dict(Counter(t['baseline_state'] for t in targets)),
            target_classes=dict(Counter(t['target_class'] for t in targets)),
            phases=dict(Counter(t['phase'] for t in targets)))
data=dict(metadata=meta, comparators=competitors, targets=targets, sources=list(sources.values()))
(OUT/'quantum-landscape.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
with (OUT/'quantum-targets.csv').open('w',encoding='utf-8-sig',newline='') as fh:
    keys=['id','workstream','phase','title','baseline_state','target_class','baseline_evidence','source_ids','acceptance_criteria','dependencies','delivery_status']
    writer=csv.DictWriter(fh,fieldnames=keys,extrasaction='ignore');writer.writeheader()
    for t in targets: writer.writerow({**t,'source_ids':';'.join(t['source_ids']),'dependencies':';'.join(t['dependencies'])})

def link(sid):
    s=sources[sid]
    return f"[{sid}: {s['title']}]({s['url']})"

intro = '''---
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
'''
lines=[intro.rstrip()]
for c in competitors:
    lines.append(f"| {link(c['id'])} | {c['classification']}; {c['host']} | {c['documented_strength']} | {c['comparison_boundary']} **Eshkol:** {c['implication_for_eshkol']} |")
lines.append('''

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
''')
for stream in dict.fromkeys(t['workstream'] for t in targets):
    lines.append(f'\n### {stream}\n')
    for t in (t for t in targets if t['workstream']==stream):
        lines.append(f"\n**{t['id']} — {t['title']}**\n\nPhase: {t['phase']}; class: {t['target_class']}; Eshkol baseline: **{t['baseline_state']}**. {t['baseline_evidence']}\n\nAcceptance: {t['acceptance_criteria']}\n\nDependencies: {', '.join(t['dependencies']) or 'none'}. Sources: {', '.join(link(s) for s in t['source_ids'])}.\n")
lines.append('\n## Source and coverage ledger\n')
for sid,s in sources.items():
    lines.append(f"\n- {link(sid)} — {s['evidence']}; checked {s['checked']}. {s['note']}")
lines.append('\n\n## Machine-checked inventory\n\n```json\n'+json.dumps(meta,indent=2)+'\n```\n')
lines.append('\nThe generator validates exactly 25 distinct comparator names, exactly 100 distinct target titles, allowed classifications, existing source IDs, valid dependency IDs and an acyclic dependency graph. These checks validate the artifact structure, not the scientific feasibility or completion of the roadmap. No source repository files were edited and no external publication was performed in this research stream.\n')
(OUT/'quantum-landscape.md').write_text('\n'.join(line.rstrip() for line in '\n'.join(lines).splitlines())+'\n',encoding='utf-8')
print(json.dumps(meta,indent=2))
