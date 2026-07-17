/*******************************************************************************
 * Moonlab Quantum State-Vector Bindings for Eshkol (Stage S1)
 *
 * Thin C shim over Moonlab's dense state-vector core (quantum_state_t,
 * gate_*, quantum_measure, measurement_expectation_z), plus the Stage S2
 * VQE surface (molecular Pauli Hamiltonians, solver, and exact gradient).
 *
 * Compile with -DESHKOL_HAVE_MOONLAB and link against libquantumsim (set by
 * CMakeLists.txt's Agent FFI block only when configured with
 * -DESHKOL_QUANTUM_ENABLED=ON and Moonlab's FetchContent succeeded).
 * Without -DESHKOL_HAVE_MOONLAB, every function below returns a graceful,
 * honestly-labeled "unavailable" error (-1 / NaN) instead of crashing or
 * silently returning a bogus value -- mirroring the existing
 * agent_treesitter.c / agent_yoga.c degrade-gracefully convention, and
 * keeping this file safe to compile unconditionally (so JIT weak-symbol
 * resolution in lib/repl/repl_jit.cpp always has a real definition to bind,
 * instead of leaving `(require agent.quantum)` a dangling reference on the
 * default ESHKOL_QUANTUM_ENABLED=OFF build).
 *
 * Symbol names below are verified against Moonlab's actual headers at the
 * pinned revision c613234cd8498804428f3838aa46dd730b1de810 (tag v1.1.0-rc):
 *   - src/quantum/state.h        : quantum_state_create, quantum_state_destroy
 *   - src/quantum/gates.h        : gate_hadamard, gate_pauli_x/_y/_z,
 *                                   gate_cnot, gate_rx/_ry/_rz, quantum_measure,
 *                                   measurement_basis_t, measurement_result_t
 *   - src/quantum/measurement.h  : measurement_expectation_z
 *   - src/utils/quantum_entropy.h: quantum_entropy_ctx_t,
 *                                   quantum_entropy_ctx_create_hw/_destroy
 *   - src/algorithms/vqe.h         : molecular Hamiltonians, VQE solver,
 *                                   optimizer, and vqe_solve
 *   - src/applications/bell_test.h : bell_test_run_chsh CHSH experiment
 *   - src/applications/moonlab_export.h
 *                                 : moonlab_vqe_gradient stable ABI wrapper
 *
 * quantum_state_t is a public struct in Moonlab (state.h) but this shim
 * follows the same discipline as the Rust binding (the model called out in
 * docs/design/MOONLAB_INTEGRATION.md Section 1.5): treat it as opaque, never
 * mirror its layout in Scheme. Eshkol callers only ever see a small integer
 * handle; the actual quantum_state_t* lives in a process-local handle table
 * here, matching the existing agent_sqlite.c / agent_regex.c pattern.
 *
 * Measurement entropy: quantum_measure() requires a caller-supplied
 * quantum_entropy_ctx_t*. We use Moonlab's own hardware-backed helper
 * (quantum_entropy_ctx_create_hw) so simulated measurement collapse is not
 * seeded by anything Eshkol controls -- lazily created once per process and
 * released at exit, mirroring moonlab_qrng_bytes's own "process-lifetime
 * context freed at atexit" pattern (documented at moonlab_export.h:33-75
 * in the pinned Moonlab revision).
 *
 * Copyright (c) 2026 Eshkol Project
 ******************************************************************************/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

/* The VQE AD bridge needs Eshkol's arena-owned tape nodes. Do not include
 * arena_memory.h here: it exposes C++ thread_local declarations and this shim
 * is intentionally compiled as C. Keep the tiny runtime boundary explicit. */
#include "../../../inc/eshkol/eshkol.h"

typedef struct arena arena_t;
extern arena_t* eshkol_current_arena(void);
extern void* arena_allocate_aligned(arena_t* arena, size_t size, size_t alignment);
extern void* arena_allocate_zeroed(arena_t* arena, size_t size);

/* Must stay in lockstep with eshkol_tensor_t's first five fields in
 * lib/core/arena_memory.h. The AD parameter tensor uses this homogeneous
 * int64-bit-pattern element layout. */
typedef struct {
    uint64_t* dimensions;
    uint64_t num_dimensions;
    int64_t* elements;
    uint64_t total_elements;
    uint64_t dtype;
} vqe_tensor_layout_t;

/*******************************************************************************
 * CONDITIONAL COMPILATION: full Moonlab-backed implementation vs graceful stubs
 ******************************************************************************/

#ifdef ESHKOL_HAVE_MOONLAB

#include "quantum/state.h"
#include "quantum/gates.h"
#include "quantum/measurement.h"
#include "utils/quantum_entropy.h"
#include "algorithms/vqe.h"
#include "applications/bell_test.h"
#include "applications/moonlab_export.h"

/*******************************************************************************
 * Handle Table
 ******************************************************************************/

#define MAX_QSTATE_HANDLES 256

static quantum_state_t* g_qstate_handles[MAX_QSTATE_HANDLES] = {0};
static int g_next_qstate_handle = 1;

/* Stage S2 keeps Moonlab-owned Pauli Hamiltonians behind the same small,
 * integer-handle boundary as state vectors.  Scheme therefore never mirrors
 * Moonlab's mutable structs or needs to know their ABI layout. */
#define MAX_HAMILTONIAN_HANDLES 256
static pauli_hamiltonian_t* g_hamiltonian_handles[MAX_HAMILTONIAN_HANDLES] = {0};
static int g_next_hamiltonian_handle = 1;

/* A gradient computation needs a vqe_solver_t plus an ansatz and optimizer.
 * Eshkol's generic vector is heterogeneous, not a contiguous double buffer,
 * so Scheme fills this private context element by element and reads the
 * resulting gradient the same way. */
#define MAX_VQE_GRADIENT_CONTEXTS 64
typedef struct {
    vqe_solver_t* solver;
    vqe_ansatz_t* ansatz;
    vqe_optimizer_t* optimizer;
    double* gradient;
} vqe_gradient_context_t;
static vqe_gradient_context_t* g_vqe_gradient_contexts[MAX_VQE_GRADIENT_CONTEXTS] = {0};
static int g_next_vqe_gradient_context = 1;

/** @brief Human-readable last-error message, mirrored via eshkol_quantum_last_error(). */
static char g_quantum_last_error[256] = {0};

static void set_last_error(const char* msg) {
    if (!msg) { g_quantum_last_error[0] = '\0'; return; }
    size_t len = strlen(msg);
    size_t copy = len < sizeof(g_quantum_last_error) - 1 ? len : sizeof(g_quantum_last_error) - 1;
    memcpy(g_quantum_last_error, msg, copy);
    g_quantum_last_error[copy] = '\0';
}

/**
 * @brief Allocates a slot in the global quantum-state handle table.
 *
 * Scans forward from the last-used index (wrapping around once) so released
 * handles are reused rather than exhausting the table.
 */
static int alloc_qstate(quantum_state_t* state) {
    for (int i = g_next_qstate_handle; i < MAX_QSTATE_HANDLES; i++) {
        if (!g_qstate_handles[i]) { g_qstate_handles[i] = state; g_next_qstate_handle = i + 1; return i; }
    }
    for (int i = 1; i < g_next_qstate_handle; i++) {
        if (!g_qstate_handles[i]) { g_qstate_handles[i] = state; g_next_qstate_handle = i + 1; return i; }
    }
    return -1;
}

static quantum_state_t* get_qstate(int64_t h) {
    if (h < 1 || h >= MAX_QSTATE_HANDLES) return NULL;
    return g_qstate_handles[h];
}

static int alloc_hamiltonian(pauli_hamiltonian_t* hamiltonian) {
    for (int i = g_next_hamiltonian_handle; i < MAX_HAMILTONIAN_HANDLES; i++) {
        if (!g_hamiltonian_handles[i]) {
            g_hamiltonian_handles[i] = hamiltonian;
            g_next_hamiltonian_handle = i + 1;
            return i;
        }
    }
    for (int i = 1; i < g_next_hamiltonian_handle; i++) {
        if (!g_hamiltonian_handles[i]) {
            g_hamiltonian_handles[i] = hamiltonian;
            g_next_hamiltonian_handle = i + 1;
            return i;
        }
    }
    return -1;
}

static pauli_hamiltonian_t* get_hamiltonian(int64_t h) {
    if (h < 1 || h >= MAX_HAMILTONIAN_HANDLES) return NULL;
    return g_hamiltonian_handles[h];
}

static int alloc_vqe_gradient_context(vqe_gradient_context_t* context) {
    for (int i = g_next_vqe_gradient_context; i < MAX_VQE_GRADIENT_CONTEXTS; i++) {
        if (!g_vqe_gradient_contexts[i]) {
            g_vqe_gradient_contexts[i] = context;
            g_next_vqe_gradient_context = i + 1;
            return i;
        }
    }
    for (int i = 1; i < g_next_vqe_gradient_context; i++) {
        if (!g_vqe_gradient_contexts[i]) {
            g_vqe_gradient_contexts[i] = context;
            g_next_vqe_gradient_context = i + 1;
            return i;
        }
    }
    return -1;
}

static vqe_gradient_context_t* get_vqe_gradient_context(int64_t h) {
    if (h < 1 || h >= MAX_VQE_GRADIENT_CONTEXTS) return NULL;
    return g_vqe_gradient_contexts[h];
}

static void free_vqe_gradient_context(vqe_gradient_context_t* context) {
    if (!context) return;
    free(context->gradient);
    vqe_solver_free(context->solver);
    vqe_ansatz_free(context->ansatz);
    vqe_optimizer_free(context->optimizer);
    free(context);
}

/* Explicit destroy functions are the normal lifecycle path.  This atexit
 * cleanup is the final backstop for handles that survived a Scheme process,
 * including handles abandoned by a non-local exit.  Contexts go first because
 * their solvers borrow the Hamiltonians. */
static void teardown_vqe_handles(void) {
    for (int i = 1; i < MAX_VQE_GRADIENT_CONTEXTS; i++) {
        free_vqe_gradient_context(g_vqe_gradient_contexts[i]);
        g_vqe_gradient_contexts[i] = NULL;
    }
    for (int i = 1; i < MAX_HAMILTONIAN_HANDLES; i++) {
        pauli_hamiltonian_free(g_hamiltonian_handles[i]);
        g_hamiltonian_handles[i] = NULL;
    }
}

static void ensure_vqe_handle_cleanup(void) {
    static int registered = 0;
    if (!registered) {
        atexit(teardown_vqe_handles);
        registered = 1;
    }
}

static double error_double(void) {
    volatile double zero = 0.0;
    return zero / zero;
}

static double unavailable_double(const char* message) {
    set_last_error(message);
    return error_double();
}

/*******************************************************************************
 * Stage S5: Bell/CHSH acceptance experiment
 ******************************************************************************/

/**
 * Run Moonlab's public CHSH experiment with the canonical settings that
 * maximise the |Phi+> violation: Alice (0, pi/2), Bob (pi/4, -pi/4).
 *
 * bell_test_default_config() uses a historical set of angles that does not
 * match bell_test_run_chsh()'s difference-angle correlator, so the default
 * configuration reports about 1.3 rather than the physical 2*sqrt(2).  Keep
 * the public experiment, but explicitly select the canonical settings here;
 * this makes the Scheme gate a test of entanglement rather than a test of that
 * stale application default.  The returned S is positive for this ordering.
 *
 * @return CHSH S on success, or NaN with eshkol_quantum_last_error() set.
 */
double eshkol_quantum_bell_chsh(int32_t num_trials) {
    if (num_trials < 4) {
        return unavailable_double("bell-chsh: num-trials must be at least four");
    }

    bell_test_config_t config = bell_test_default_config();
    config.num_trials = num_trials;
    config.alice_angle_1 = 0.0;
    config.alice_angle_2 = 1.57079632679489661923;  /* pi / 2 */
    config.bob_angle_1 = 0.78539816339744830962;    /* pi / 4 */
    config.bob_angle_2 = -0.78539816339744830962;   /* -pi / 4 */

    bell_test_results_t results;
    if (bell_test_run_chsh(&config, &results) != 0 || results.S != results.S) {
        return unavailable_double("bell-chsh: Moonlab CHSH experiment failed");
    }
    return results.S;
}

/*******************************************************************************
 * Lazily-initialized hardware entropy context for quantum_measure()
 ******************************************************************************/

static quantum_entropy_ctx_t* g_measure_entropy = NULL;

static void teardown_measure_entropy(void) {
    if (g_measure_entropy) {
        quantum_entropy_ctx_destroy(g_measure_entropy);
        g_measure_entropy = NULL;
    }
}

/** @return Non-NULL on success. Sets the shim's last-error on failure. */
static quantum_entropy_ctx_t* ensure_measure_entropy(void) {
    if (!g_measure_entropy) {
        g_measure_entropy = quantum_entropy_ctx_create_hw();
        if (g_measure_entropy) {
            atexit(teardown_measure_entropy);
        } else {
            set_last_error("failed to initialize Moonlab hardware entropy context for measurement");
        }
    }
    return g_measure_entropy;
}

/*******************************************************************************
 * Lifecycle
 ******************************************************************************/

/**
 * @brief Allocates an n-qubit state vector initialized to |0...0>.
 * @return Handle (>= 1) on success, -1 on invalid n / OOM / handle-table full.
 */
int64_t eshkol_quantum_state_create(int32_t num_qubits) {
    if (num_qubits <= 0) {
        set_last_error("make-quantum-state: num_qubits must be positive");
        return -1;
    }
    quantum_state_t* state = quantum_state_create(num_qubits);
    if (!state) {
        set_last_error("quantum_state_create failed (Moonlab returned NULL -- invalid qubit count or OOM)");
        return -1;
    }
    int handle = alloc_qstate(state);
    if (handle < 0) {
        quantum_state_destroy(state);
        set_last_error("quantum-state handle table exhausted");
        return -1;
    }
    return (int64_t)handle;
}

/** @brief Destroys a state vector and frees its handle slot. Safe on an already-freed handle. */
void eshkol_quantum_state_destroy(int64_t handle) {
    if (handle < 1 || handle >= MAX_QSTATE_HANDLES) return;
    quantum_state_t* state = g_qstate_handles[handle];
    if (state) {
        quantum_state_destroy(state);
        g_qstate_handles[handle] = NULL;
    }
}

/*******************************************************************************
 * Gates -- each returns 0 on success, or the qs_error_t (always < 0) Moonlab
 * reported, so Scheme callers can raise a specific, catchable error.
 ******************************************************************************/

int32_t eshkol_quantum_gate_hadamard(int64_t handle, int32_t qubit) {
    quantum_state_t* state = get_qstate(handle);
    if (!state) { set_last_error("apply-hadamard: invalid quantum-state handle"); return -1; }
    qs_error_t err = gate_hadamard(state, qubit);
    if (err != QS_SUCCESS) set_last_error("gate_hadamard failed");
    return (int32_t)err;
}

int32_t eshkol_quantum_gate_pauli_x(int64_t handle, int32_t qubit) {
    quantum_state_t* state = get_qstate(handle);
    if (!state) { set_last_error("apply-pauli-x: invalid quantum-state handle"); return -1; }
    qs_error_t err = gate_pauli_x(state, qubit);
    if (err != QS_SUCCESS) set_last_error("gate_pauli_x failed");
    return (int32_t)err;
}

int32_t eshkol_quantum_gate_pauli_y(int64_t handle, int32_t qubit) {
    quantum_state_t* state = get_qstate(handle);
    if (!state) { set_last_error("apply-pauli-y: invalid quantum-state handle"); return -1; }
    qs_error_t err = gate_pauli_y(state, qubit);
    if (err != QS_SUCCESS) set_last_error("gate_pauli_y failed");
    return (int32_t)err;
}

int32_t eshkol_quantum_gate_pauli_z(int64_t handle, int32_t qubit) {
    quantum_state_t* state = get_qstate(handle);
    if (!state) { set_last_error("apply-pauli-z: invalid quantum-state handle"); return -1; }
    qs_error_t err = gate_pauli_z(state, qubit);
    if (err != QS_SUCCESS) set_last_error("gate_pauli_z failed");
    return (int32_t)err;
}

int32_t eshkol_quantum_gate_cnot(int64_t handle, int32_t control, int32_t target) {
    quantum_state_t* state = get_qstate(handle);
    if (!state) { set_last_error("apply-cnot: invalid quantum-state handle"); return -1; }
    qs_error_t err = gate_cnot(state, control, target);
    if (err != QS_SUCCESS) set_last_error("gate_cnot failed");
    return (int32_t)err;
}

int32_t eshkol_quantum_gate_rx(int64_t handle, int32_t qubit, double theta) {
    quantum_state_t* state = get_qstate(handle);
    if (!state) { set_last_error("apply-rx: invalid quantum-state handle"); return -1; }
    qs_error_t err = gate_rx(state, qubit, theta);
    if (err != QS_SUCCESS) set_last_error("gate_rx failed");
    return (int32_t)err;
}

int32_t eshkol_quantum_gate_ry(int64_t handle, int32_t qubit, double theta) {
    quantum_state_t* state = get_qstate(handle);
    if (!state) { set_last_error("apply-ry: invalid quantum-state handle"); return -1; }
    qs_error_t err = gate_ry(state, qubit, theta);
    if (err != QS_SUCCESS) set_last_error("gate_ry failed");
    return (int32_t)err;
}

int32_t eshkol_quantum_gate_rz(int64_t handle, int32_t qubit, double theta) {
    quantum_state_t* state = get_qstate(handle);
    if (!state) { set_last_error("apply-rz: invalid quantum-state handle"); return -1; }
    qs_error_t err = gate_rz(state, qubit, theta);
    if (err != QS_SUCCESS) set_last_error("gate_rz failed");
    return (int32_t)err;
}

/*******************************************************************************
 * Measurement
 ******************************************************************************/

/**
 * @brief Projectively measures one qubit in the computational (Z) basis,
 *        collapsing the state vector.
 * @return 0 or 1 (the measurement outcome), or -1 on error (invalid handle,
 *         invalid qubit, or entropy-context failure -- check
 *         eshkol_quantum_last_error()).
 */
int32_t eshkol_quantum_measure(int64_t handle, int32_t qubit) {
    quantum_state_t* state = get_qstate(handle);
    if (!state) { set_last_error("measure: invalid quantum-state handle"); return -1; }
    quantum_entropy_ctx_t* entropy = ensure_measure_entropy();
    if (!entropy) return -1; /* set_last_error already called */
    measurement_result_t result = quantum_measure(state, qubit, MEASURE_COMPUTATIONAL, entropy);
    return (int32_t)result.outcome;
}

/**
 * @brief Expectation value of the Pauli-Z observable on one qubit, <psi|Z_q|psi>.
 * @return Value in [-1, 1], or a NaN sentinel on invalid handle (check
 *         eshkol_quantum_last_error()).
 */
double eshkol_quantum_expectation_z(int64_t handle, int32_t qubit) {
    quantum_state_t* state = get_qstate(handle);
    if (!state) {
        set_last_error("expectation-z: invalid quantum-state handle");
        /* Portable NaN sentinel without a literal 0.0/0.0 constant-fold warning. */
        volatile double zero = 0.0;
        return zero / zero;
    }
    return measurement_expectation_z(state, qubit);
}

/*******************************************************************************
 * Diagnostics
 ******************************************************************************/

/** @return Number of qubits the handle was created with, or -1 on invalid handle. */
int32_t eshkol_quantum_num_qubits(int64_t handle) {
    quantum_state_t* state = get_qstate(handle);
    if (!state) return -1;
    return (int32_t)state->num_qubits;
}

/*******************************************************************************
 * Stage S2: molecular Hamiltonians and VQE
 ******************************************************************************/

/** Store a newly-created Moonlab Hamiltonian behind an Eshkol handle. */
static int64_t store_hamiltonian(pauli_hamiltonian_t* hamiltonian, const char* op) {
    if (!hamiltonian) {
        set_last_error(op);
        return -1;
    }
    ensure_vqe_handle_cleanup();
    int handle = alloc_hamiltonian(hamiltonian);
    if (handle < 0) {
        pauli_hamiltonian_free(hamiltonian);
        set_last_error("VQE Hamiltonian handle table exhausted");
        return -1;
    }
    return (int64_t)handle;
}

/** @return Handle for H2/STO-3G at @p bond_distance Angstroms, or -1 on failure. */
int64_t eshkol_vqe_make_h2_hamiltonian(double bond_distance) {
    return store_hamiltonian(vqe_create_h2_hamiltonian(bond_distance),
                             "make-h2-hamiltonian: Moonlab allocation failed");
}

/** @return Handle for LiH/6-31G at @p bond_distance Angstroms, or -1 on failure. */
int64_t eshkol_vqe_make_lih_hamiltonian(double bond_distance) {
    return store_hamiltonian(vqe_create_lih_hamiltonian(bond_distance),
                             "make-lih-hamiltonian: Moonlab allocation failed");
}

/** @return Handle for H2O/STO-3G fixed geometry, or -1 on failure. */
int64_t eshkol_vqe_make_h2o_hamiltonian(void) {
    return store_hamiltonian(vqe_create_h2o_hamiltonian(),
                             "make-h2o-hamiltonian: Moonlab allocation failed");
}

/** Release a Hamiltonian handle. Safe on an already-released handle. */
void eshkol_vqe_hamiltonian_destroy(int64_t handle) {
    if (handle < 1 || handle >= MAX_HAMILTONIAN_HANDLES) return;
    pauli_hamiltonian_free(g_hamiltonian_handles[handle]);
    g_hamiltonian_handles[handle] = NULL;
}

/**
 * @return Full-CI reference energy, including nuclear repulsion, or NaN on
 * failure. The NaN is intentional: Scheme converts it to a catchable error
 * rather than ever exposing a fabricated energy.
 */
double eshkol_vqe_hamiltonian_exact_ground_energy(int64_t handle) {
    pauli_hamiltonian_t* hamiltonian = get_hamiltonian(handle);
    if (!hamiltonian) {
        return unavailable_double("hamiltonian-exact-ground-energy: invalid Hamiltonian handle");
    }
    double energy = vqe_exact_ground_state_energy(hamiltonian);
    if (energy != energy || energy >= DBL_MAX) {
        return unavailable_double("vqe_exact_ground_state_energy failed");
    }
    return energy;
}

/**
 * Build the intentionally small, deterministic Stage S2 VQE configuration:
 * one hardware-efficient layer and ADAM.  Moonlab's own H2 VQE test uses this
 * pairing; it reaches the exact two-qubit reference while keeping a single
 * essential Scheme tuning knob (the iteration cap).  The optimizer is quiet
 * because Eshkol, not Moonlab's stdout formatter, owns user-facing output.
 * The caller owns ansatz/optimizer/solver and frees all three.
 */
static vqe_solver_t* create_default_vqe_solver(pauli_hamiltonian_t* hamiltonian,
                                                int32_t iterations,
                                                vqe_ansatz_t** ansatz_out,
                                                vqe_optimizer_t** optimizer_out) {
    *ansatz_out = NULL;
    *optimizer_out = NULL;
    if (!hamiltonian) {
        set_last_error("VQE: invalid Hamiltonian handle");
        return NULL;
    }
    if (iterations <= 0) {
        set_last_error("vqe-optimize: iterations must be positive");
        return NULL;
    }

    quantum_entropy_ctx_t* entropy = ensure_measure_entropy();
    if (!entropy) return NULL;

    vqe_ansatz_t* ansatz = vqe_create_hardware_efficient_ansatz(
        hamiltonian->num_qubits, 1);
    if (!ansatz) {
        set_last_error("vqe-optimize: could not create hardware-efficient ansatz");
        return NULL;
    }

    vqe_optimizer_t* optimizer = vqe_optimizer_create(VQE_OPTIMIZER_ADAM);
    if (!optimizer) {
        vqe_ansatz_free(ansatz);
        set_last_error("vqe-optimize: could not create ADAM optimizer");
        return NULL;
    }
    optimizer->max_iterations = (size_t)iterations;
    optimizer->tolerance = 1e-9;
    optimizer->learning_rate = 0.03;
    optimizer->verbose = 0;

    vqe_solver_t* solver = vqe_solver_create(hamiltonian, ansatz, optimizer, entropy);
    if (!solver) {
        vqe_optimizer_free(optimizer);
        vqe_ansatz_free(ansatz);
        set_last_error("vqe-optimize: could not create Moonlab VQE solver");
        return NULL;
    }

    *ansatz_out = ansatz;
    *optimizer_out = optimizer;
    return solver;
}

/**
 * Evaluate the fixed-parameter Stage S2 default ansatz.  This is deliberately
 * separate from vqe_solve(): the bridge needs E(theta), not an optimizer run.
 */
static double vqe_energy_at_parameters(int64_t handle, const double* parameters,
                                       size_t parameter_count) {
    pauli_hamiltonian_t* hamiltonian = get_hamiltonian(handle);
    if (!hamiltonian || !parameters) {
        return unavailable_double("vqe-energy: invalid Hamiltonian handle or parameter buffer");
    }

    vqe_ansatz_t* ansatz = NULL;
    vqe_optimizer_t* optimizer = NULL;
    vqe_solver_t* solver = create_default_vqe_solver(hamiltonian, 1,
                                                      &ansatz, &optimizer);
    if (!solver) return error_double();

    if (parameter_count != ansatz->num_parameters) {
        vqe_solver_free(solver);
        vqe_optimizer_free(optimizer);
        vqe_ansatz_free(ansatz);
        return unavailable_double("vqe-energy: parameter vector has the wrong length");
    }

    double energy = vqe_compute_energy(solver, parameters);
    vqe_solver_free(solver);
    vqe_optimizer_free(optimizer);
    vqe_ansatz_free(ansatz);

    if (energy != energy || energy >= DBL_MAX) {
        return unavailable_double("vqe-energy: Moonlab energy evaluation failed");
    }
    return energy;
}

/** Run exactly the same native adjoint path exposed by Stage S2's vqe-gradient. */
static int vqe_gradient_at_parameters(int64_t handle, const double* parameters,
                                      double* gradient, size_t parameter_count) {
    pauli_hamiltonian_t* hamiltonian = get_hamiltonian(handle);
    if (!hamiltonian || !parameters || !gradient) {
        set_last_error("vqe-energy AD: invalid Hamiltonian handle or gradient buffer");
        return -1;
    }

    vqe_ansatz_t* ansatz = NULL;
    vqe_optimizer_t* optimizer = NULL;
    vqe_solver_t* solver = create_default_vqe_solver(hamiltonian, 1,
                                                      &ansatz, &optimizer);
    if (!solver) return -1;

    if (parameter_count != ansatz->num_parameters) {
        vqe_solver_free(solver);
        vqe_optimizer_free(optimizer);
        vqe_ansatz_free(ansatz);
        set_last_error("vqe-energy AD: parameter vector has the wrong length");
        return -1;
    }

    /* moonlab_vqe_gradient is the frozen ABI around Moonlab's exact native
     * adjoint (with its analytic parameter-shift fallback), matching
     * eshkol_vqe_gradient_compute exactly. */
    int rc = moonlab_vqe_gradient(solver, parameters, gradient, parameter_count);
    vqe_solver_free(solver);
    vqe_optimizer_free(optimizer);
    vqe_ansatz_free(ansatz);
    if (rc != 0) {
        set_last_error("vqe-energy AD: Moonlab exact gradient failed");
        return -1;
    }
    return 0;
}

/* Converts an ordinary Scheme numeric vector/tensor to a contiguous arena
 * buffer. The compiler intrinsic calls this only outside AD mode. */
static double* vqe_copy_plain_parameters(const eshkol_tagged_value_t* params,
                                         size_t* count_out) {
    if (!params || !count_out) {
        set_last_error("vqe-energy: params must be a vector");
        return NULL;
    }

    size_t count = 0;
    double* values = NULL;
    arena_t* arena = eshkol_current_arena();
    if (!arena) {
        set_last_error("vqe-energy: no active Eshkol arena");
        return NULL;
    }

    if (ESHKOL_IS_VECTOR_COMPAT(*params)) {
        const uint8_t* raw = (const uint8_t*)(uintptr_t)params->data.ptr_val;
        if (!raw) {
            set_last_error("vqe-energy: params must be a vector");
            return NULL;
        }
        int64_t signed_count = 0;
        memcpy(&signed_count, raw, sizeof(signed_count));
        if (signed_count <= 0 || (uint64_t)signed_count > SIZE_MAX / sizeof(double)) {
            set_last_error("vqe-energy: parameter vector must be non-empty");
            return NULL;
        }
        count = (size_t)signed_count;
        values = (double*)arena_allocate_aligned(arena, count * sizeof(double), sizeof(double));
        if (!values) {
            set_last_error("vqe-energy: allocation failed");
            return NULL;
        }
        const eshkol_tagged_value_t* elements =
            (const eshkol_tagged_value_t*)(raw + sizeof(int64_t));
        for (size_t i = 0; i < count; ++i) {
            if (elements[i].type == ESHKOL_VALUE_DOUBLE) {
                values[i] = elements[i].data.double_val;
            } else if (elements[i].type == ESHKOL_VALUE_INT64) {
                values[i] = (double)elements[i].data.int_val;
            } else {
                set_last_error("vqe-energy: parameters must be real numbers");
                return NULL;
            }
            if (values[i] != values[i]) {
                set_last_error("vqe-energy: parameters must be finite numbers");
                return NULL;
            }
        }
    } else if (ESHKOL_IS_TENSOR_COMPAT(*params)) {
        const vqe_tensor_layout_t* tensor =
            (const vqe_tensor_layout_t*)(uintptr_t)params->data.ptr_val;
        if (!tensor || !tensor->elements || tensor->total_elements == 0 ||
            tensor->total_elements > SIZE_MAX / sizeof(double)) {
            set_last_error("vqe-energy: parameter tensor must be non-empty");
            return NULL;
        }
        count = (size_t)tensor->total_elements;
        values = (double*)arena_allocate_aligned(arena, count * sizeof(double), sizeof(double));
        if (!values) {
            set_last_error("vqe-energy: allocation failed");
            return NULL;
        }
        for (size_t i = 0; i < count; ++i) {
            memcpy(&values[i], &tensor->elements[i], sizeof(double));
            if (values[i] != values[i]) {
                set_last_error("vqe-energy: parameters must be finite numbers");
                return NULL;
            }
        }
    } else {
        set_last_error("vqe-energy: params must be a vector");
        return NULL;
    }

    *count_out = count;
    return values;
}

/** C entry used by the non-AD half of the compiler intrinsic. */
double eshkol_vqe_energy_from_tagged(int64_t handle,
                                     const eshkol_tagged_value_t* params) {
    size_t count = 0;
    double* values = vqe_copy_plain_parameters(params, &count);
    if (!values) return error_double();
    return vqe_energy_at_parameters(handle, values, count);
}

typedef struct {
    int64_t hamiltonian_handle;
    int parameter_count;
    double parameters[];
} vqe_custom_vjp_context_t;

typedef struct {
    double energy;
    eshkol_custom_vjp_t* vjp;
    ad_node_t** inputs;
    int64_t input_count;
} vqe_ad_prepared_t;

/* The custom-VJP convention is local partials only. `upstream` is supplied by
 * the general runtime callback ABI but is intentionally not applied here;
 * eshkol_ad_node_custom_backward multiplies it exactly once. */
static void vqe_custom_vjp_backward(void* raw_ctx, double upstream,
                                    double* out_grads, int n) {
    (void)upstream;
    if (!out_grads || n <= 0) return;
    for (int i = 0; i < n; ++i) out_grads[i] = 0.0;

    vqe_custom_vjp_context_t* ctx = (vqe_custom_vjp_context_t*)raw_ctx;
    if (!ctx || ctx->parameter_count != n) return;
    (void)vqe_gradient_at_parameters(ctx->hamiltonian_handle, ctx->parameters,
                                     out_grads, (size_t)n);
}

/**
 * Prepare an arena-owned descriptor for an AD-node tensor of VQE parameters.
 * The generated intrinsic invokes this only while Eshkol reverse mode is live:
 * tensor elements are therefore ad_node_t* bit patterns, not f64 bit patterns.
 */
void* eshkol_vqe_ad_prepare(int64_t handle, const eshkol_tagged_value_t* params) {
    if (!params || !ESHKOL_IS_TENSOR_COMPAT(*params)) {
        set_last_error("vqe-energy AD: params must be an AD tensor");
        return NULL;
    }
    const vqe_tensor_layout_t* tensor =
        (const vqe_tensor_layout_t*)(uintptr_t)params->data.ptr_val;
    if (!tensor || !tensor->elements || tensor->total_elements == 0 ||
        tensor->total_elements > (uint64_t)INT_MAX) {
        set_last_error("vqe-energy AD: invalid parameter tensor");
        return NULL;
    }

    const int n = (int)tensor->total_elements;
    arena_t* arena = eshkol_current_arena();
    if (!arena || (size_t)n > SIZE_MAX / sizeof(ad_node_t*) ||
        (size_t)n > (SIZE_MAX - sizeof(vqe_custom_vjp_context_t)) / sizeof(double)) {
        set_last_error("vqe-energy AD: allocation size overflow");
        return NULL;
    }

    ad_node_t** inputs = (ad_node_t**)arena_allocate_aligned(
        arena, (size_t)n * sizeof(*inputs), sizeof(void*));
    vqe_custom_vjp_context_t* ctx = (vqe_custom_vjp_context_t*)arena_allocate_aligned(
        arena, sizeof(*ctx) + (size_t)n * sizeof(double), sizeof(double));
    if (!inputs || !ctx) {
        set_last_error("vqe-energy AD: arena allocation failed");
        return NULL;
    }

    ctx->hamiltonian_handle = handle;
    ctx->parameter_count = n;
    if (!get_hamiltonian(handle)) {
        fprintf(stderr,
                "eshkol: vqe-energy AD error: invalid Hamiltonian handle %lld "
                "inside gradient — the gradient would silently be zero\n",
                (long long)handle);
        set_last_error("vqe-energy AD: invalid Hamiltonian handle");
        return NULL;
    }
    for (int i = 0; i < n; ++i) {
        ad_node_t* input = (ad_node_t*)(uintptr_t)tensor->elements[i];
        if (!input) {
            fprintf(stderr,
                    "eshkol: vqe-energy AD error: parameter tensor contains a "
                    "non-AD value — the gradient would silently be zero\n");
            set_last_error("vqe-energy AD: parameter tensor contains a non-AD value");
            return NULL;
        }
        inputs[i] = input;
        ctx->parameters[i] = input->value;
    }

    double energy = vqe_energy_at_parameters(handle, ctx->parameters, (size_t)n);
    if (energy != energy) return NULL;

    eshkol_custom_vjp_t* vjp = (eshkol_custom_vjp_t*)arena_allocate_zeroed(
        arena, sizeof(*vjp));
    vqe_ad_prepared_t* prepared = (vqe_ad_prepared_t*)arena_allocate_zeroed(
        arena, sizeof(*prepared));
    if (!vjp || !prepared) {
        set_last_error("vqe-energy AD: arena allocation failed");
        return NULL;
    }

    vjp->backward = vqe_custom_vjp_backward;
    vjp->ctx = ctx;
    vjp->inputs = inputs;
    vjp->n = n;
    prepared->energy = energy;
    prepared->vjp = vjp;
    prepared->inputs = inputs;
    prepared->input_count = n;
    return prepared;
}

double eshkol_vqe_ad_prepared_energy(const void* raw_prepared) {
    const vqe_ad_prepared_t* prepared = (const vqe_ad_prepared_t*)raw_prepared;
    return prepared ? prepared->energy : error_double();
}

void* eshkol_vqe_ad_prepared_inputs(void* raw_prepared) {
    vqe_ad_prepared_t* prepared = (vqe_ad_prepared_t*)raw_prepared;
    return prepared ? prepared->inputs : NULL;
}

int64_t eshkol_vqe_ad_prepared_input_count(const void* raw_prepared) {
    const vqe_ad_prepared_t* prepared = (const vqe_ad_prepared_t*)raw_prepared;
    return prepared ? prepared->input_count : 0;
}

void* eshkol_vqe_ad_prepared_vjp(void* raw_prepared) {
    vqe_ad_prepared_t* prepared = (vqe_ad_prepared_t*)raw_prepared;
    return prepared ? prepared->vjp : NULL;
}

/**
 * Run Moonlab VQE using the default one-layer hardware-efficient ansatz.
 * @return Optimized variational energy, or NaN on a genuine Moonlab failure.
 */
double eshkol_vqe_optimize(int64_t handle, int32_t iterations) {
    pauli_hamiltonian_t* hamiltonian = get_hamiltonian(handle);
    vqe_ansatz_t* ansatz = NULL;
    vqe_optimizer_t* optimizer = NULL;
    vqe_solver_t* solver = create_default_vqe_solver(hamiltonian, iterations,
                                                       &ansatz, &optimizer);
    if (!solver) return error_double();

    vqe_result_t result = vqe_solve(solver);
    double energy = result.ground_state_energy;
    free(result.optimal_parameters);
    vqe_solver_free(solver);
    vqe_optimizer_free(optimizer);
    vqe_ansatz_free(ansatz);

    if (energy != energy || energy >= DBL_MAX) {
        return unavailable_double("vqe-optimize: Moonlab VQE solve failed");
    }
    return energy;
}

/**
 * Start a private default-ansatz VQE context for vqe-gradient.  It exposes
 * scalar set/read operations because Eshkol Scheme vectors are not C double
 * buffers.  The actual gradient is still Moonlab's native exact path.
 */
int64_t eshkol_vqe_gradient_context_create(int64_t handle) {
    pauli_hamiltonian_t* hamiltonian = get_hamiltonian(handle);
    vqe_ansatz_t* ansatz = NULL;
    vqe_optimizer_t* optimizer = NULL;
    vqe_solver_t* solver = create_default_vqe_solver(hamiltonian, 1,
                                                       &ansatz, &optimizer);
    if (!solver) return -1;

    vqe_gradient_context_t* context = calloc(1, sizeof(*context));
    if (!context) {
        vqe_solver_free(solver);
        vqe_optimizer_free(optimizer);
        vqe_ansatz_free(ansatz);
        set_last_error("vqe-gradient: allocation failed");
        return -1;
    }
    context->solver = solver;
    context->ansatz = ansatz;
    context->optimizer = optimizer;

    int slot = alloc_vqe_gradient_context(context);
    if (slot < 0) {
        free_vqe_gradient_context(context);
        set_last_error("vqe-gradient context handle table exhausted");
        return -1;
    }
    ensure_vqe_handle_cleanup();
    return (int64_t)slot;
}

void eshkol_vqe_gradient_context_destroy(int64_t handle) {
    if (handle < 1 || handle >= MAX_VQE_GRADIENT_CONTEXTS) return;
    free_vqe_gradient_context(g_vqe_gradient_contexts[handle]);
    g_vqe_gradient_contexts[handle] = NULL;
}

int64_t eshkol_vqe_gradient_parameter_count(int64_t handle) {
    vqe_gradient_context_t* context = get_vqe_gradient_context(handle);
    if (!context || !context->ansatz) {
        set_last_error("vqe-gradient: invalid gradient context handle");
        return -1;
    }
    if (context->ansatz->num_parameters > INT32_MAX) {
        set_last_error("vqe-gradient: parameter count exceeds Eshkol vector limit");
        return -1;
    }
    return (int64_t)context->ansatz->num_parameters;
}

int32_t eshkol_vqe_gradient_set_parameter(int64_t handle, int32_t index, double value) {
    vqe_gradient_context_t* context = get_vqe_gradient_context(handle);
    if (!context || !context->ansatz || index < 0 ||
        (size_t)index >= context->ansatz->num_parameters) {
        set_last_error("vqe-gradient: parameter index out of range");
        return -1;
    }
    if (value != value) {
        set_last_error("vqe-gradient: parameter must be a finite number");
        return -1;
    }
    context->ansatz->parameters[index] = value;
    return 0;
}

int32_t eshkol_vqe_gradient_compute(int64_t handle) {
    vqe_gradient_context_t* context = get_vqe_gradient_context(handle);
    if (!context || !context->solver || !context->ansatz) {
        set_last_error("vqe-gradient: invalid gradient context handle");
        return -1;
    }
    size_t n = context->ansatz->num_parameters;
    double* gradient = calloc(n, sizeof(double));
    if (!gradient) {
        set_last_error("vqe-gradient: allocation failed");
        return -1;
    }

    /* moonlab_vqe_gradient is Moonlab's frozen ABI wrapper around
     * vqe_compute_gradient.  It dispatches to reverse-mode adjoint for the
     * noise-free HEA context built above, with analytic parameter-shift as
     * Moonlab's exact fallback. */
    int rc = moonlab_vqe_gradient(context->solver, context->ansatz->parameters,
                                  gradient, n);
    if (rc != 0) {
        free(gradient);
        set_last_error("vqe-gradient: Moonlab exact gradient failed");
        return -1;
    }
    free(context->gradient);
    context->gradient = gradient;
    return 0;
}

double eshkol_vqe_gradient_get(int64_t handle, int32_t index) {
    vqe_gradient_context_t* context = get_vqe_gradient_context(handle);
    if (!context || !context->gradient || index < 0 ||
        (size_t)index >= context->ansatz->num_parameters) {
        return unavailable_double("vqe-gradient: gradient unavailable or index out of range");
    }
    return context->gradient[index];
}

/**
 * @brief Copies the last shim-level error message into @p buf.
 * @return Number of bytes written (excluding NUL), or -1 if @p buf is NULL/too small.
 */
int32_t eshkol_quantum_last_error(char* buf, int64_t buf_size) {
    if (!buf || buf_size <= 0) return -1;
    size_t len = strlen(g_quantum_last_error);
    size_t copy = (size_t)buf_size - 1 < len ? (size_t)buf_size - 1 : len;
    memcpy(buf, g_quantum_last_error, copy);
    buf[copy] = '\0';
    return (int32_t)copy;
}

#else /* !ESHKOL_HAVE_MOONLAB */

/*******************************************************************************
 * Graceful stubs when Moonlab is not compiled in (ESHKOL_QUANTUM_ENABLED=OFF,
 * the default). Every entry point fails honestly and explicitly -- never a
 * silently-wrong value -- so lib/agent/quantum.esk's Scheme wrappers raise a
 * clear "Moonlab quantum support not enabled in this build" error instead of
 * segfaulting on a dangling weak symbol or fabricating quantum data.
 ******************************************************************************/

static const char* const kUnavailableMsg =
    "Moonlab quantum support not enabled in this build "
    "(reconfigure with -DESHKOL_QUANTUM_ENABLED=ON)";

/** @brief Stub: Moonlab support was not compiled in, so state creation always fails. */
int64_t eshkol_quantum_state_create(int32_t num_qubits) { (void)num_qubits; return -1; }
/** @brief Stub: no-op since no states exist without Moonlab support. */
void eshkol_quantum_state_destroy(int64_t handle) { (void)handle; }
/** @brief Stub: always fails since no states exist without Moonlab support. */
int32_t eshkol_quantum_num_qubits(int64_t handle) { (void)handle; return -1; }
/** @brief Stub: always fails since no states exist without Moonlab support. */
int32_t eshkol_quantum_gate_hadamard(int64_t handle, int32_t qubit) { (void)handle; (void)qubit; return -1; }
/** @brief Stub: always fails since no states exist without Moonlab support. */
int32_t eshkol_quantum_gate_pauli_x(int64_t handle, int32_t qubit) { (void)handle; (void)qubit; return -1; }
/** @brief Stub: always fails since no states exist without Moonlab support. */
int32_t eshkol_quantum_gate_pauli_y(int64_t handle, int32_t qubit) { (void)handle; (void)qubit; return -1; }
/** @brief Stub: always fails since no states exist without Moonlab support. */
int32_t eshkol_quantum_gate_pauli_z(int64_t handle, int32_t qubit) { (void)handle; (void)qubit; return -1; }
/** @brief Stub: always fails since no states exist without Moonlab support. */
int32_t eshkol_quantum_gate_cnot(int64_t handle, int32_t control, int32_t target) {
    (void)handle; (void)control; (void)target; return -1;
}
/** @brief Stub: always fails since no states exist without Moonlab support. */
int32_t eshkol_quantum_gate_rx(int64_t handle, int32_t qubit, double theta) {
    (void)handle; (void)qubit; (void)theta; return -1;
}
/** @brief Stub: always fails since no states exist without Moonlab support. */
int32_t eshkol_quantum_gate_ry(int64_t handle, int32_t qubit, double theta) {
    (void)handle; (void)qubit; (void)theta; return -1;
}
/** @brief Stub: always fails since no states exist without Moonlab support. */
int32_t eshkol_quantum_gate_rz(int64_t handle, int32_t qubit, double theta) {
    (void)handle; (void)qubit; (void)theta; return -1;
}
/** @brief Stub: always fails since no states exist without Moonlab support. */
int32_t eshkol_quantum_measure(int64_t handle, int32_t qubit) { (void)handle; (void)qubit; return -1; }
/** @brief Stub: always fails (NaN sentinel) since no states exist without Moonlab support. */
double eshkol_quantum_expectation_z(int64_t handle, int32_t qubit) {
    (void)handle; (void)qubit;
    volatile double zero = 0.0;
    return zero / zero;
}
/** @brief Stub: Moonlab support was not compiled in, so CHSH is unavailable. */
double eshkol_quantum_bell_chsh(int32_t num_trials) {
    (void)num_trials;
    volatile double zero = 0.0;
    return zero / zero;
}

/* Stage S2 VQE stubs.  Keep every declared symbol available in a default-OFF
 * build so agent.quantum can raise the same explicit, catchable unavailability
 * error instead of becoming an unresolved extern. */
int64_t eshkol_vqe_make_h2_hamiltonian(double bond_distance) { (void)bond_distance; return -1; }
int64_t eshkol_vqe_make_lih_hamiltonian(double bond_distance) { (void)bond_distance; return -1; }
int64_t eshkol_vqe_make_h2o_hamiltonian(void) { return -1; }
void eshkol_vqe_hamiltonian_destroy(int64_t handle) { (void)handle; }
double eshkol_vqe_hamiltonian_exact_ground_energy(int64_t handle) {
    (void)handle;
    volatile double zero = 0.0;
    return zero / zero;
}
double eshkol_vqe_optimize(int64_t handle, int32_t iterations) {
    (void)handle; (void)iterations;
    volatile double zero = 0.0;
    return zero / zero;
}
double eshkol_vqe_energy_from_tagged(int64_t handle,
                                     const eshkol_tagged_value_t* params) {
    (void)handle; (void)params;
    volatile double zero = 0.0;
    return zero / zero;
}
void* eshkol_vqe_ad_prepare(int64_t handle, const eshkol_tagged_value_t* params) {
    (void)handle; (void)params;
    return NULL;
}
double eshkol_vqe_ad_prepared_energy(const void* prepared) {
    (void)prepared;
    volatile double zero = 0.0;
    return zero / zero;
}
void* eshkol_vqe_ad_prepared_inputs(void* prepared) { (void)prepared; return NULL; }
int64_t eshkol_vqe_ad_prepared_input_count(const void* prepared) {
    (void)prepared;
    return 0;
}
void* eshkol_vqe_ad_prepared_vjp(void* prepared) { (void)prepared; return NULL; }
int64_t eshkol_vqe_gradient_context_create(int64_t handle) { (void)handle; return -1; }
void eshkol_vqe_gradient_context_destroy(int64_t handle) { (void)handle; }
int64_t eshkol_vqe_gradient_parameter_count(int64_t handle) { (void)handle; return -1; }
int32_t eshkol_vqe_gradient_set_parameter(int64_t handle, int32_t index, double value) {
    (void)handle; (void)index; (void)value; return -1;
}
int32_t eshkol_vqe_gradient_compute(int64_t handle) { (void)handle; return -1; }
double eshkol_vqe_gradient_get(int64_t handle, int32_t index) {
    (void)handle; (void)index;
    volatile double zero = 0.0;
    return zero / zero;
}
/**
 * @brief Stub: copies the fixed "not enabled" message into @p buf, so callers
 *        that always check eshkol_quantum_last_error() after a -1 get an
 *        honest, actionable reason instead of an empty string.
 */
int32_t eshkol_quantum_last_error(char* buf, int64_t buf_size) {
    if (!buf || buf_size <= 0) return -1;
    size_t len = strlen(kUnavailableMsg);
    size_t copy = (size_t)buf_size - 1 < len ? (size_t)buf_size - 1 : len;
    memcpy(buf, kUnavailableMsg, copy);
    buf[copy] = '\0';
    return (int32_t)copy;
}

#endif /* ESHKOL_HAVE_MOONLAB */
