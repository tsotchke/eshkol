/**
 * @file vm_workspace.c
 * @brief Global Workspace Theory for the Eshkol bytecode VM consciousness engine.
 *
 * Implements Baars (1988) / Bengio (2017) Global Workspace:
 *   - Workspace buffer (shared tensor region)
 *   - Module registration (closures as cognitive specialists)
 *   - Attention-based competition (softmax over salience scores)
 *   - Broadcast mechanism (winner's output -> workspace content)
 *
 * The workspace invokes VM closures through a callback function pointer
 * supplied by the VM, decoupling this module from the VM's internals.
 *
 * Ported from workspace.h / workspace.cpp (C++ w/ Eshkol arena) to pure C
 * using VmRegionStack / VmArena from vm_arena.h.
 *
 * Native call IDs: 540-549
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"   /* pulls in vm_arena.h + subtypes */
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ========================================================================
 * Data structures
 * ======================================================================== */

#define VM_WS_MAX_MODULES 32

typedef struct {
    char   name[64];
    void*  process_fn;   /* opaque VM closure pointer */
    double salience;     /* last computed salience score */
} VmWSModule;

typedef struct {
    int         n_modules;
    int         max_modules;
    int         dim;          /* workspace vector dimension */
    int         step_count;   /* cognitive cycle counter */
    double*     content;      /* arena-allocated [dim] */
    VmWSModule  modules[VM_WS_MAX_MODULES];
} VmWorkspace;

/* ========================================================================
 * Callback type for invoking VM closures
 *
 * The VM passes a function pointer with this signature to vm_ws_step.
 * It calls the closure pointed to by `closure_ptr` with `content` (dim doubles),
 * and writes the result into out_salience and out_proposal.
 *
 * Returns 1 on success, 0 on failure.
 * ======================================================================== */

typedef int (*VmClosureCallFn)(
    void*   closure_ptr,     /* opaque closure pointer from VmWSModule */
    const double* content,   /* input: current workspace content */
    int     dim,             /* dimension of content */
    double* out_salience,    /* output: salience score */
    double* out_proposal,    /* output: proposal vector [dim] */
    int     out_dim          /* dimension of proposal buffer */
);

/* ========================================================================
 * Construction
 * ======================================================================== */

/* 540: make-workspace */
static VmWorkspace* vm_ws_new(VmRegionStack* rs, int dim, int max_modules) {
    if (!rs || dim <= 0 || max_modules <= 0) return NULL;
    if (max_modules > VM_WS_MAX_MODULES) max_modules = VM_WS_MAX_MODULES;

    VmWorkspace* ws = (VmWorkspace*)vm_alloc_object(rs,
        VM_SUBTYPE_WORKSPACE, sizeof(VmWorkspace));
    if (!ws) return NULL;

    memset(ws, 0, sizeof(VmWorkspace));
    ws->n_modules   = 0;
    ws->max_modules = max_modules;
    ws->dim         = dim;
    ws->step_count  = 0;

    /* Allocate content buffer (initialized to zero) */
    ws->content = (double*)vm_alloc(rs, (size_t)dim * sizeof(double));
    if (!ws->content) return NULL;
    memset(ws->content, 0, (size_t)dim * sizeof(double));

    return ws;
}

/* 541: ws-register! */
static void vm_ws_register(VmWorkspace* ws, const char* name, void* closure_ptr) {
    if (!ws || !name) return;
    if (ws->n_modules >= ws->max_modules) return;

    VmWSModule* mod = &ws->modules[ws->n_modules];
    size_t len = strlen(name);
    if (len >= 63) len = 63;
    memcpy(mod->name, name, len);
    mod->name[len] = '\0';
    mod->process_fn = closure_ptr;
    mod->salience   = 0.0;

    ws->n_modules++;
}

/* ========================================================================
 * Content access
 * ======================================================================== */

/* 543: ws-get-content */
static const double* vm_ws_get_content(const VmWorkspace* ws) {
    if (!ws) return NULL;
    return ws->content;
}

/* 544: ws-set-content */
static void vm_ws_set_content(VmWorkspace* ws, const double* data, int dim) {
    if (!ws || !data) return;
    int copy_dim = (dim < ws->dim) ? dim : ws->dim;
    memcpy(ws->content, data, (size_t)copy_dim * sizeof(double));
}

/* 545: ws-get-dim */
static int vm_ws_get_dim(const VmWorkspace* ws) {
    return ws ? ws->dim : 0;
}

/* 546: ws-get-step-count */
static int vm_ws_get_step_count(const VmWorkspace* ws) {
    return ws ? ws->step_count : 0;
}

/* ========================================================================
 * Softmax
 * ======================================================================== */

static void softmax(const double* input, double* output, int n) {
    if (n <= 0) return;

    /* Find max for numerical stability */
    double max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    /* Compute exp(x - max) and sum */
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }

    /* Normalize */
    if (sum > 0.0) {
        for (int i = 0; i < n; i++) {
            output[i] /= sum;
        }
    }
}

/* ========================================================================
 * ws-step! — The cognitive cycle
 *
 * 1. Each module processes current workspace content
 * 2. Modules return (salience, proposal) pairs
 * 3. Softmax competition over salience scores
 * 4. Winner's proposal becomes new workspace content
 * 5. All modules see updated content next cycle
 * ======================================================================== */

/* 542: ws-step! */
static int vm_ws_step(VmRegionStack* rs, VmWorkspace* ws,
    VmClosureCallFn call_closure_fn)
{
    if (!rs || !ws || !call_closure_fn || ws->n_modules == 0) return 0;

    int n = ws->n_modules;
    int dim = ws->dim;

    /* Allocate temporary storage for saliences and proposals */
    double* saliences = (double*)vm_alloc(rs, (size_t)n * sizeof(double));
    if (!saliences) return 0;

    /* Proposals: n * dim doubles */
    double* proposals = (double*)vm_alloc(rs, (size_t)n * (size_t)dim * sizeof(double));
    if (!proposals) return 0;

    int valid_count = 0;

    /* Call each module's closure */
    for (int i = 0; i < n; i++) {
        double sal = -1e30;
        double* proposal = proposals + i * dim;
        memset(proposal, 0, (size_t)dim * sizeof(double));

        int ok = call_closure_fn(
            ws->modules[i].process_fn,
            ws->content,
            dim,
            &sal,
            proposal,
            dim
        );

        if (ok) {
            saliences[i] = sal;
            valid_count++;
        } else {
            saliences[i] = -1e30; /* effectively zero after softmax */
        }
    }

    if (valid_count == 0) return 0;

    /* Softmax over salience scores */
    double sm[VM_WS_MAX_MODULES];
    softmax(saliences, sm, n);

    /* Find winner */
    int winner = 0;
    double best_score = sm[0];
    for (int i = 1; i < n; i++) {
        if (sm[i] > best_score) {
            best_score = sm[i];
            winner = i;
        }
    }

    /* Update module saliences */
    for (int i = 0; i < n; i++) {
        ws->modules[i].salience = sm[i];
    }

    /* Copy winner's proposal to workspace content */
    memcpy(ws->content, proposals + winner * dim, (size_t)dim * sizeof(double));

    ws->step_count++;
    return 1;
}

/* ========================================================================
 * Display
 * ======================================================================== */

static void vm_display_workspace(const VmWorkspace* ws) {
    if (!ws) { printf("#<workspace: empty>"); return; }
    printf("#<workspace: dim=%d, %d modules, step=%d>",
           ws->dim, ws->n_modules, ws->step_count);
}

/* ========================================================================
 * Self-tests
 * ======================================================================== */

#ifdef VM_WORKSPACE_TEST
#include <assert.h>

/* Test closure: always returns salience=1.0, proposal=[1,0,0,...] */
static int test_module1(void* closure_ptr, const double* content, int dim,
    double* out_salience, double* out_proposal, int out_dim)
{
    (void)closure_ptr;
    (void)content;
    *out_salience = 1.0;
    memset(out_proposal, 0, (size_t)out_dim * sizeof(double));
    if (out_dim > 0) out_proposal[0] = 1.0;
    return 1;
}

/* Test closure: always returns salience=0.5, proposal=[0,1,0,...] */
static int test_module2(void* closure_ptr, const double* content, int dim,
    double* out_salience, double* out_proposal, int out_dim)
{
    (void)closure_ptr;
    (void)content;
    *out_salience = 0.5;
    memset(out_proposal, 0, (size_t)out_dim * sizeof(double));
    if (out_dim > 1) out_proposal[1] = 1.0;
    return 1;
}

/* Test closure: returns salience=2.0, proposal=[0,0,1] */
static int test_module3(void* closure_ptr, const double* content, int dim,
    double* out_salience, double* out_proposal, int out_dim)
{
    (void)closure_ptr;
    (void)content;
    *out_salience = 2.0;
    memset(out_proposal, 0, (size_t)out_dim * sizeof(double));
    if (out_dim > 2) out_proposal[2] = 1.0;
    return 1;
}

/* Dispatcher for test closures — maps closure_ptr to test functions */
static int test_call_closure(void* closure_ptr, const double* content, int dim,
    double* out_salience, double* out_proposal, int out_dim)
{
    /* closure_ptr encodes which test function to call:
     * 1 -> test_module1, 2 -> test_module2, 3 -> test_module3 */
    intptr_t id = (intptr_t)closure_ptr;
    switch (id) {
        case 1: return test_module1(closure_ptr, content, dim, out_salience, out_proposal, out_dim);
        case 2: return test_module2(closure_ptr, content, dim, out_salience, out_proposal, out_dim);
        case 3: return test_module3(closure_ptr, content, dim, out_salience, out_proposal, out_dim);
        default: return 0;
    }
}

int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    printf("=== vm_workspace self-tests ===\n");

    /* --- Test 1: Workspace construction --- */
    {
        VmWorkspace* ws = vm_ws_new(&rs, 3, 4);
        assert(ws != NULL);
        assert(ws->dim == 3);
        assert(ws->max_modules == 4);
        assert(ws->n_modules == 0);
        assert(ws->step_count == 0);

        /* Content should be zeros */
        const double* c = vm_ws_get_content(ws);
        assert(c[0] == 0.0 && c[1] == 0.0 && c[2] == 0.0);

        printf("  [PASS] workspace construction\n");
    }

    /* --- Test 2: Module registration --- */
    {
        VmWorkspace* ws = vm_ws_new(&rs, 3, 4);
        vm_ws_register(ws, "perception", (void*)(intptr_t)1);
        vm_ws_register(ws, "reasoning", (void*)(intptr_t)2);
        assert(ws->n_modules == 2);
        assert(strcmp(ws->modules[0].name, "perception") == 0);
        assert(strcmp(ws->modules[1].name, "reasoning") == 0);

        printf("  [PASS] module registration\n");
    }

    /* --- Test 3: ws-step with 2 modules, higher salience wins --- */
    {
        VmWorkspace* ws = vm_ws_new(&rs, 3, 4);
        vm_ws_register(ws, "module1", (void*)(intptr_t)1);  /* salience=1.0, proposal=[1,0,0] */
        vm_ws_register(ws, "module2", (void*)(intptr_t)2);  /* salience=0.5, proposal=[0,1,0] */

        int ok = vm_ws_step(&rs, ws, test_call_closure);
        assert(ok);
        assert(ws->step_count == 1);

        /* Module 1 has higher salience (1.0 vs 0.5), so it wins.
         * After softmax: exp(1.0) / (exp(1.0) + exp(0.5)) ≈ 0.622
         * Module 1 wins, content = [1, 0, 0] */
        const double* c = vm_ws_get_content(ws);
        assert(c[0] == 1.0);
        assert(c[1] == 0.0);
        assert(c[2] == 0.0);

        printf("  [PASS] ws-step: module 1 wins (salience 1.0 > 0.5)\n");
    }

    /* --- Test 4: ws-step with 3 modules, module 3 wins --- */
    {
        VmWorkspace* ws = vm_ws_new(&rs, 3, 4);
        vm_ws_register(ws, "m1", (void*)(intptr_t)1);  /* salience=1.0 */
        vm_ws_register(ws, "m2", (void*)(intptr_t)2);  /* salience=0.5 */
        vm_ws_register(ws, "m3", (void*)(intptr_t)3);  /* salience=2.0 */

        int ok = vm_ws_step(&rs, ws, test_call_closure);
        assert(ok);

        /* Module 3 has salience=2.0 (highest), content = [0, 0, 1] */
        const double* c = vm_ws_get_content(ws);
        assert(c[0] == 0.0);
        assert(c[1] == 0.0);
        assert(c[2] == 1.0);

        printf("  [PASS] ws-step: module 3 wins (salience 2.0 highest)\n");
    }

    /* --- Test 5: softmax correctness --- */
    {
        double input[3] = { 1.0, 2.0, 3.0 };
        double output[3];
        softmax(input, output, 3);

        /* Check sum = 1 */
        double sum = output[0] + output[1] + output[2];
        assert(fabs(sum - 1.0) < 1e-10);

        /* Check ordering: output[2] > output[1] > output[0] */
        assert(output[2] > output[1]);
        assert(output[1] > output[0]);

        /* Known values: softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652] */
        assert(fabs(output[0] - 0.0900) < 0.001);
        assert(fabs(output[1] - 0.2447) < 0.001);
        assert(fabs(output[2] - 0.6652) < 0.001);

        printf("  [PASS] softmax correctness\n");
    }

    /* --- Test 6: softmax numerical stability (large values) --- */
    {
        double input[3] = { 1000.0, 1001.0, 1002.0 };
        double output[3];
        softmax(input, output, 3);

        double sum = output[0] + output[1] + output[2];
        assert(fabs(sum - 1.0) < 1e-10);
        /* Should give same relative distribution as [0,1,2] */
        assert(output[2] > output[1]);
        assert(output[1] > output[0]);
        assert(fabs(output[0] - 0.0900) < 0.001);

        printf("  [PASS] softmax numerical stability (values ~1000)\n");
    }

    /* --- Test 7: set/get content --- */
    {
        VmWorkspace* ws = vm_ws_new(&rs, 4, 2);
        double data[4] = { 1.0, 2.0, 3.0, 4.0 };
        vm_ws_set_content(ws, data, 4);

        const double* c = vm_ws_get_content(ws);
        assert(c[0] == 1.0 && c[1] == 2.0 && c[2] == 3.0 && c[3] == 4.0);

        /* Partial set (smaller dim) */
        double data2[2] = { 10.0, 20.0 };
        vm_ws_set_content(ws, data2, 2);
        c = vm_ws_get_content(ws);
        assert(c[0] == 10.0 && c[1] == 20.0 && c[2] == 3.0 && c[3] == 4.0);

        printf("  [PASS] set/get content\n");
    }

    /* --- Test 8: multiple steps --- */
    {
        VmWorkspace* ws = vm_ws_new(&rs, 3, 4);
        vm_ws_register(ws, "m1", (void*)(intptr_t)1);
        vm_ws_register(ws, "m2", (void*)(intptr_t)2);

        for (int i = 0; i < 5; i++) {
            int ok = vm_ws_step(&rs, ws, test_call_closure);
            assert(ok);
        }
        assert(ws->step_count == 5);

        /* Module 1 always wins, content should be [1,0,0] */
        const double* c = vm_ws_get_content(ws);
        assert(c[0] == 1.0 && c[1] == 0.0 && c[2] == 0.0);

        printf("  [PASS] multiple steps (5 cycles)\n");
    }

    /* --- Test 9: max modules cap --- */
    {
        VmWorkspace* ws = vm_ws_new(&rs, 2, 2);
        vm_ws_register(ws, "a", (void*)(intptr_t)1);
        vm_ws_register(ws, "b", (void*)(intptr_t)2);
        vm_ws_register(ws, "c", (void*)(intptr_t)3); /* should be ignored */
        assert(ws->n_modules == 2); /* capped at max_modules=2 */

        printf("  [PASS] max modules cap\n");
    }

    /* --- Test 10: display --- */
    {
        VmWorkspace* ws = vm_ws_new(&rs, 3, 4);
        vm_ws_register(ws, "test", (void*)(intptr_t)1);
        printf("  display: ");
        vm_display_workspace(ws);
        printf("\n");
        printf("  [PASS] display\n");
    }

    /* --- Test 11: salience scores stored after step --- */
    {
        VmWorkspace* ws = vm_ws_new(&rs, 3, 4);
        vm_ws_register(ws, "m1", (void*)(intptr_t)1);  /* salience=1.0 */
        vm_ws_register(ws, "m2", (void*)(intptr_t)2);  /* salience=0.5 */

        vm_ws_step(&rs, ws, test_call_closure);

        /* After softmax, module saliences should sum to 1 */
        double sum = ws->modules[0].salience + ws->modules[1].salience;
        assert(fabs(sum - 1.0) < 1e-10);

        /* Module 1's softmax score should be higher than module 2's */
        assert(ws->modules[0].salience > ws->modules[1].salience);

        printf("  [PASS] salience scores stored after step\n");
    }

    /* --- Test 12: dim and step_count getters --- */
    {
        VmWorkspace* ws = vm_ws_new(&rs, 7, 3);
        assert(vm_ws_get_dim(ws) == 7);
        assert(vm_ws_get_step_count(ws) == 0);

        vm_ws_register(ws, "x", (void*)(intptr_t)1);
        vm_ws_step(&rs, ws, test_call_closure);
        assert(vm_ws_get_step_count(ws) == 1);

        printf("  [PASS] dim/step_count getters\n");
    }

    vm_region_stack_destroy(&rs);
    printf("vm_workspace: ALL TESTS PASSED\n");
    return 0;
}
#endif /* VM_WORKSPACE_TEST */
