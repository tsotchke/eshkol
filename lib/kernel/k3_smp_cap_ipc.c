/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol OS Phase K3: Freestanding C11 Zero-Alloc SMP Scheduler,
 * Capability Derivation Tree (CDT) with Immediate Atomic Revocation,
 * and Zero-Copy Session-Typed submit/complete Ring IPC.
 *
 * Strictly zero heap allocation, zero libc/POSIX/LLVM/unwind symbols,
 * and 32-bit atomic operations natively supported across all 5 ISAs.
 */

#include <eshkol/kernel/k3_smp_cap_ipc.h>

static void k3_zero_bytes(void* dst, size_t n) {
    volatile uint8_t* p = (volatile uint8_t*)dst;
    for (size_t i = 0u; i < n; ++i) {
        p[i] = 0u;
    }
}

static void k3_copy_bytes(void* dst, const void* src, size_t n) {
    volatile uint8_t* d = (volatile uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    for (size_t i = 0u; i < n; ++i) {
        d[i] = s[i];
    }
}

/* Native 32-bit spinlock helpers (zero libatomic calls on all 5 ISAs). */
static void k3_lock_acquire(uint32_t* lock_word) {
    while (__atomic_exchange_n(lock_word, 1u, __ATOMIC_ACQUIRE) != 0u) {
        __asm__ volatile("" ::: "memory");
    }
}

static void k3_lock_release(uint32_t* lock_word) {
    __atomic_store_n(lock_word, 0u, __ATOMIC_RELEASE);
}

static uint32_t k3_atomic_load_u32(const uint32_t* p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}

static void k3_atomic_store_u32(uint32_t* p, uint32_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}

static uint32_t k3_next_badge(k3_cdt_t* cdt, uint32_t slot_id, uint32_t generation) {
    uint32_t x = cdt->badge_seed ^ (slot_id * 0x9E3779B1u) ^ (generation * 0x85EBCA6Bu);
    x ^= (x >> 16u);
    x *= 0x7FEB352Du;
    x ^= (x >> 15u);
    cdt->badge_seed = x + 0xA5A55A5Au;
    return (x == 0u) ? 1u : x;
}

/* ─────────────────────────── SMP Scheduler ─────────────────────────── */

k3_status_t k3_smp_sched_init(k3_smp_sched_t* sched, uint32_t cpu_count) {
    if (!sched) {
        return K3_ERR_INVALID_ARG;
    }
    if (cpu_count == 0u || cpu_count > K3_MAX_CPUS) {
        return K3_ERR_CPU_OUT_OF_RANGE;
    }
    k3_zero_bytes(sched, sizeof(*sched));
    sched->cpu_count = cpu_count;

    for (uint32_t t = 0u; t < K3_MAX_TASKS; ++t) {
        sched->tasks[t].task_id = t;
        sched->tasks[t].state = K3_TASK_UNUSED;
    }
    for (uint32_t c = 0u; c < K3_MAX_CPUS; ++c) {
        sched->runqueues[c].cpu_id = c;
        sched->runqueues[c].current_task_id = K3_INVALID_ID;
        for (uint32_t q = 0u; q < K3_RUNQUEUE_CAP; ++q) {
            sched->runqueues[c].queue[q] = K3_INVALID_ID;
        }
    }
    return K3_OK;
}

k3_status_t k3_smp_task_spawn(k3_smp_sched_t* sched,
                              uint32_t task_id,
                              uint32_t compartment_id,
                              uint32_t cpu_id,
                              uint32_t priority) {
    if (!sched) {
        return K3_ERR_INVALID_ARG;
    }
    if (task_id >= K3_MAX_TASKS) {
        return K3_ERR_TASK_OUT_OF_RANGE;
    }
    if (cpu_id >= sched->cpu_count || cpu_id >= K3_MAX_CPUS) {
        return K3_ERR_CPU_OUT_OF_RANGE;
    }
    if (compartment_id >= K3_MAX_COMPARTMENTS || priority > 7u) {
        return K3_ERR_INVALID_ARG;
    }

    k3_task_t* t = &sched->tasks[task_id];
    k3_zero_bytes(t, sizeof(*t));
    t->task_id = task_id;
    t->compartment_id = compartment_id;
    t->cpu_affinity = cpu_id;
    t->priority = priority;
    t->state = K3_TASK_RUNNABLE;
    t->time_slice_ticks = 4u;
    sched->active_tasks += 1u;
    return k3_smp_enqueue(sched, cpu_id, task_id);
}

k3_status_t k3_smp_enqueue(k3_smp_sched_t* sched, uint32_t cpu_id, uint32_t task_id) {
    if (!sched) {
        return K3_ERR_INVALID_ARG;
    }
    if (cpu_id >= sched->cpu_count || cpu_id >= K3_MAX_CPUS) {
        return K3_ERR_CPU_OUT_OF_RANGE;
    }
    if (task_id >= K3_MAX_TASKS) {
        return K3_ERR_TASK_OUT_OF_RANGE;
    }
    if (sched->tasks[task_id].state != K3_TASK_RUNNABLE) {
        return K3_ERR_INVALID_ARG;
    }

    k3_cpu_runqueue_t* rq = &sched->runqueues[cpu_id];
    k3_lock_acquire(&rq->lock_word);
    if (rq->count >= K3_RUNQUEUE_CAP) {
        k3_lock_release(&rq->lock_word);
        return K3_ERR_RUNQUEUE_FULL;
    }
    rq->queue[rq->tail] = task_id;
    rq->tail = (rq->tail + 1u) & (K3_RUNQUEUE_CAP - 1u);
    rq->count += 1u;
    k3_lock_release(&rq->lock_word);
    return K3_OK;
}

k3_status_t k3_smp_schedule_next(k3_smp_sched_t* sched, uint32_t cpu_id, uint32_t* out_task_id) {
    if (!sched || !out_task_id) {
        return K3_ERR_INVALID_ARG;
    }
    if (cpu_id >= sched->cpu_count || cpu_id >= K3_MAX_CPUS) {
        return K3_ERR_CPU_OUT_OF_RANGE;
    }

    k3_cpu_runqueue_t* rq = &sched->runqueues[cpu_id];
    k3_lock_acquire(&rq->lock_word);
    if (rq->count == 0u) {
        rq->current_task_id = K3_INVALID_ID;
        k3_lock_release(&rq->lock_word);
        return K3_ERR_RUNQUEUE_EMPTY;
    }

    /* Select highest-priority runnable task in FIFO order within priority level. */
    uint32_t best_pos = K3_INVALID_ID;
    uint32_t best_prio = 0u;
    uint32_t n = rq->count;

    for (uint32_t i = 0u; i < n; ++i) {
        uint32_t slot = (rq->head + i) & (K3_RUNQUEUE_CAP - 1u);
        uint32_t tid = rq->queue[slot];
        if (tid < K3_MAX_TASKS && sched->tasks[tid].state == K3_TASK_RUNNABLE) {
            if (best_pos == K3_INVALID_ID || sched->tasks[tid].priority > best_prio) {
                best_pos = i;
                best_prio = sched->tasks[tid].priority;
            }
        }
    }

    if (best_pos == K3_INVALID_ID) {
        rq->head = 0u;
        rq->tail = 0u;
        rq->count = 0u;
        rq->current_task_id = K3_INVALID_ID;
        k3_lock_release(&rq->lock_word);
        return K3_ERR_RUNQUEUE_EMPTY;
    }

    uint32_t chosen_slot = (rq->head + best_pos) & (K3_RUNQUEUE_CAP - 1u);
    uint32_t chosen_tid = rq->queue[chosen_slot];

    /* Compact selected slot out while preserving FIFO order of remaining tasks. */
    for (uint32_t i = best_pos; i + 1u < n; ++i) {
        uint32_t cur = (rq->head + i) & (K3_RUNQUEUE_CAP - 1u);
        uint32_t nxt = (rq->head + i + 1u) & (K3_RUNQUEUE_CAP - 1u);
        rq->queue[cur] = rq->queue[nxt];
    }
    rq->tail = (rq->tail + K3_RUNQUEUE_CAP - 1u) & (K3_RUNQUEUE_CAP - 1u);
    rq->queue[rq->tail] = K3_INVALID_ID;
    rq->count -= 1u;

    rq->current_task_id = chosen_tid;
    rq->context_switches += 1ULL;
    sched->tasks[chosen_tid].state = K3_TASK_RUNNING;
    sched->tasks[chosen_tid].dispatch_count += 1ULL;

    *out_task_id = chosen_tid;
    k3_lock_release(&rq->lock_word);
    return K3_OK;
}

k3_status_t k3_smp_preempt_tick(k3_smp_sched_t* sched, uint32_t cpu_id, uint32_t* out_next_task) {
    if (!sched || !out_next_task) {
        return K3_ERR_INVALID_ARG;
    }
    if (cpu_id >= sched->cpu_count || cpu_id >= K3_MAX_CPUS) {
        return K3_ERR_CPU_OUT_OF_RANGE;
    }

    k3_cpu_runqueue_t* rq = &sched->runqueues[cpu_id];
    k3_lock_acquire(&rq->lock_word);
    uint32_t cur_tid = rq->current_task_id;
    if (cur_tid < K3_MAX_TASKS && sched->tasks[cur_tid].state == K3_TASK_RUNNING) {
        sched->tasks[cur_tid].state = K3_TASK_RUNNABLE;
        if (rq->count < K3_RUNQUEUE_CAP) {
            rq->queue[rq->tail] = cur_tid;
            rq->tail = (rq->tail + 1u) & (K3_RUNQUEUE_CAP - 1u);
            rq->count += 1u;
        }
        rq->preemptions += 1ULL;
        rq->current_task_id = K3_INVALID_ID;
    }
    k3_lock_release(&rq->lock_word);
    return k3_smp_schedule_next(sched, cpu_id, out_next_task);
}

k3_status_t k3_smp_fault_task(k3_smp_sched_t* sched, uint32_t task_id, uint64_t fault_code) {
    if (!sched) {
        return K3_ERR_INVALID_ARG;
    }
    if (task_id >= K3_MAX_TASKS) {
        return K3_ERR_TASK_OUT_OF_RANGE;
    }
    k3_task_t* t = &sched->tasks[task_id];
    if (t->state == K3_TASK_UNUSED) {
        return K3_ERR_TASK_OUT_OF_RANGE;
    }

    t->state = K3_TASK_FAULTED;
    t->fault_code = fault_code;
    if (sched->active_tasks > 0u) {
        sched->active_tasks -= 1u;
    }

    /* Purge ONLY the faulting task from all per-CPU runqueues; siblings stay intact. */
    for (uint32_t c = 0u; c < sched->cpu_count; ++c) {
        k3_cpu_runqueue_t* rq = &sched->runqueues[c];
        k3_lock_acquire(&rq->lock_word);
        if (rq->current_task_id == task_id) {
            rq->current_task_id = K3_INVALID_ID;
        }
        uint32_t kept = 0u;
        uint32_t temp[K3_RUNQUEUE_CAP];
        for (uint32_t i = 0u; i < rq->count; ++i) {
            uint32_t slot = (rq->head + i) & (K3_RUNQUEUE_CAP - 1u);
            uint32_t tid = rq->queue[slot];
            if (tid != task_id && tid < K3_MAX_TASKS) {
                temp[kept++] = tid;
            }
        }
        rq->head = 0u;
        rq->tail = kept & (K3_RUNQUEUE_CAP - 1u);
        rq->count = kept;
        for (uint32_t i = 0u; i < K3_RUNQUEUE_CAP; ++i) {
            rq->queue[i] = (i < kept) ? temp[i] : K3_INVALID_ID;
        }
        k3_lock_release(&rq->lock_word);
    }
    return K3_OK;
}

/* ─────────────────── Capability Derivation Tree (CDT) ─────────────────── */

void k3_cdt_init(k3_cdt_t* cdt) {
    if (!cdt) {
        return;
    }
    k3_zero_bytes(cdt, sizeof(*cdt));
    cdt->badge_seed = 0x517CC1B7u;
    for (uint32_t i = 0u; i < K3_MAX_CAPS; ++i) {
        cdt->nodes[i].generation = 1u;
        cdt->nodes[i].parent_slot = K3_INVALID_ID;
        cdt->nodes[i].first_child = K3_INVALID_ID;
        cdt->nodes[i].next_sibling = K3_INVALID_ID;
    }
}

static uint32_t k3_cdt_find_free_slot_locked(const k3_cdt_t* cdt) {
    for (uint32_t i = 0u; i < K3_MAX_CAPS; ++i) {
        if (k3_atomic_load_u32(&cdt->nodes[i].active) == 0u) {
            return i;
        }
    }
    return K3_INVALID_ID;
}

static int k3_cdt_handle_matches_locked(const k3_cdt_t* cdt, const k3_cap_handle_t* h) {
    if (!h || h->slot_id >= K3_MAX_CAPS) {
        return 0;
    }
    const k3_cdt_node_t* n = &cdt->nodes[h->slot_id];
    if (k3_atomic_load_u32(&n->active) == 0u) {
        return 0;
    }
    if (k3_atomic_load_u32(&n->generation) != h->generation) {
        return 0;
    }
    if (n->secret_badge != h->badge || n->compartment_id != h->compartment_id) {
        return 0;
    }
    return 1;
}

k3_status_t k3_cdt_mint_root(k3_cdt_t* cdt,
                             uint32_t compartment_id,
                             uint32_t resource_id,
                             uint32_t rights,
                             k3_cap_handle_t* out_handle) {
    if (!cdt || !out_handle || compartment_id >= K3_MAX_COMPARTMENTS) {
        return K3_ERR_INVALID_ARG;
    }
    if ((rights & ~K3_RIGHT_ALL) != 0u || rights == 0u) {
        return K3_ERR_INVALID_ARG;
    }

    k3_lock_acquire(&cdt->lock_word);
    uint32_t slot = k3_cdt_find_free_slot_locked(cdt);
    if (slot == K3_INVALID_ID) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CAP_POOL_FULL;
    }

    k3_cdt_node_t* n = &cdt->nodes[slot];
    uint32_t gen = n->generation + 1u;
    uint32_t badge = k3_next_badge(cdt, slot, gen);

    n->compartment_id = compartment_id;
    n->resource_id = resource_id;
    n->rights = rights;
    n->depth = 0u;
    n->parent_slot = K3_INVALID_ID;
    n->first_child = K3_INVALID_ID;
    n->next_sibling = K3_INVALID_ID;
    n->secret_badge = badge;
    k3_atomic_store_u32(&n->generation, gen);
    k3_atomic_store_u32(&n->active, 1u);
    cdt->live_count += 1u;

    out_handle->slot_id = slot;
    out_handle->generation = gen;
    out_handle->compartment_id = compartment_id;
    out_handle->badge = badge;

    k3_lock_release(&cdt->lock_word);
    return K3_OK;
}

k3_status_t k3_cdt_derive(k3_cdt_t* cdt,
                          const k3_cap_handle_t* parent_handle,
                          uint32_t target_compartment_id,
                          uint32_t child_rights,
                          k3_cap_handle_t* out_child) {
    if (!cdt || !parent_handle || !out_child || target_compartment_id >= K3_MAX_COMPARTMENTS) {
        return K3_ERR_INVALID_ARG;
    }
    if ((child_rights & ~K3_RIGHT_ALL) != 0u || child_rights == 0u) {
        return K3_ERR_INVALID_ARG;
    }

    k3_lock_acquire(&cdt->lock_word);
    if (parent_handle->slot_id >= K3_MAX_CAPS) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CAP_INVALID;
    }
    k3_cdt_node_t* parent = &cdt->nodes[parent_handle->slot_id];
    if (k3_atomic_load_u32(&parent->active) == 0u ||
        k3_atomic_load_u32(&parent->generation) != parent_handle->generation) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CAP_REVOKED;
    }
    if (parent->secret_badge != parent_handle->badge ||
        parent->compartment_id != parent_handle->compartment_id) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CAP_CONFINEMENT;
    }

    /* Enforce DERIVE right before attenuation checks. */
    if ((parent->rights & K3_RIGHT_DERIVE) == 0u) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CAP_DENIED;
    }

    /* Strict monotonic attenuation: child_rights MUST be a subset of parent->rights. */
    if ((child_rights & ~parent->rights) != 0u) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_RIGHTS_ESCALATION;
    }

    /* Strict CDT depth ceiling. */
    if (parent->depth >= K3_MAX_CDT_DEPTH) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CDT_DEPTH_OVERFLOW;
    }

    uint32_t slot = k3_cdt_find_free_slot_locked(cdt);
    if (slot == K3_INVALID_ID) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CAP_POOL_FULL;
    }

    k3_cdt_node_t* child = &cdt->nodes[slot];
    uint32_t gen = child->generation + 1u;
    uint32_t badge = k3_next_badge(cdt, slot, gen);

    child->compartment_id = target_compartment_id;
    child->resource_id = parent->resource_id;
    child->rights = child_rights;
    child->depth = parent->depth + 1u;
    child->parent_slot = parent_handle->slot_id;
    child->first_child = K3_INVALID_ID;
    child->next_sibling = parent->first_child;
    child->secret_badge = badge;
    parent->first_child = slot;

    k3_atomic_store_u32(&child->generation, gen);
    k3_atomic_store_u32(&child->active, 1u);
    cdt->live_count += 1u;

    out_child->slot_id = slot;
    out_child->generation = gen;
    out_child->compartment_id = target_compartment_id;
    out_child->badge = badge;

    k3_lock_release(&cdt->lock_word);
    return K3_OK;
}

static uint32_t k3_cdt_subtree_height_locked(const k3_cdt_t* cdt, uint32_t root_slot) {
    uint32_t stack_slot[K3_MAX_CAPS];
    uint32_t stack_rel[K3_MAX_CAPS];
    uint32_t sp = 0u;
    uint32_t max_rel = 0u;

    stack_slot[sp] = root_slot;
    stack_rel[sp] = 0u;
    sp += 1u;

    while (sp > 0u) {
        sp -= 1u;
        uint32_t cur = stack_slot[sp];
        uint32_t rel = stack_rel[sp];
        if (rel > max_rel) {
            max_rel = rel;
        }
        uint32_t ch = cdt->nodes[cur].first_child;
        uint32_t guard = 0u;
        while (ch != K3_INVALID_ID && ch < K3_MAX_CAPS && guard < K3_MAX_CAPS) {
            if (sp < K3_MAX_CAPS && k3_atomic_load_u32(&cdt->nodes[ch].active) != 0u) {
                stack_slot[sp] = ch;
                stack_rel[sp] = rel + 1u;
                sp += 1u;
            }
            ch = cdt->nodes[ch].next_sibling;
            guard += 1u;
        }
    }
    return max_rel;
}

static void k3_cdt_update_subtree_depths_locked(k3_cdt_t* cdt, uint32_t root_slot, uint32_t new_depth) {
    uint32_t stack_slot[K3_MAX_CAPS];
    uint32_t stack_depth[K3_MAX_CAPS];
    uint32_t sp = 0u;

    stack_slot[sp] = root_slot;
    stack_depth[sp] = new_depth;
    sp += 1u;

    while (sp > 0u) {
        sp -= 1u;
        uint32_t cur = stack_slot[sp];
        uint32_t d = stack_depth[sp];
        cdt->nodes[cur].depth = d;
        uint32_t ch = cdt->nodes[cur].first_child;
        uint32_t guard = 0u;
        while (ch != K3_INVALID_ID && ch < K3_MAX_CAPS && guard < K3_MAX_CAPS) {
            if (sp < K3_MAX_CAPS && k3_atomic_load_u32(&cdt->nodes[ch].active) != 0u) {
                stack_slot[sp] = ch;
                stack_depth[sp] = d + 1u;
                sp += 1u;
            }
            ch = cdt->nodes[ch].next_sibling;
            guard += 1u;
        }
    }
}

k3_status_t k3_cdt_reparent(k3_cdt_t* cdt,
                            const k3_cap_handle_t* node_handle,
                            const k3_cap_handle_t* new_parent_handle) {
    if (!cdt || !node_handle || !new_parent_handle) {
        return K3_ERR_INVALID_ARG;
    }

    k3_lock_acquire(&cdt->lock_word);
    if (!k3_cdt_handle_matches_locked(cdt, node_handle) ||
        !k3_cdt_handle_matches_locked(cdt, new_parent_handle)) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CAP_REVOKED;
    }

    uint32_t node_slot = node_handle->slot_id;
    uint32_t np_slot = new_parent_handle->slot_id;

    /* Explicit self-cycle check. */
    if (node_slot == np_slot) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CDT_CYCLE;
    }

    /* Walk ancestor chain of new_parent: if node_slot appears, reparenting would form a cycle! */
    uint32_t anc = np_slot;
    for (uint32_t hop = 0u; hop <= K3_MAX_CDT_DEPTH + 1u && anc != K3_INVALID_ID; ++hop) {
        if (anc == node_slot) {
            k3_lock_release(&cdt->lock_word);
            return K3_ERR_CDT_CYCLE;
        }
        anc = (anc < K3_MAX_CAPS) ? cdt->nodes[anc].parent_slot : K3_INVALID_ID;
    }

    k3_cdt_node_t* node = &cdt->nodes[node_slot];
    k3_cdt_node_t* new_parent = &cdt->nodes[np_slot];

    if (node->resource_id != new_parent->resource_id) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CAP_CONFINEMENT;
    }
    if ((node->rights & ~new_parent->rights) != 0u) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_RIGHTS_ESCALATION;
    }

    uint32_t sub_h = k3_cdt_subtree_height_locked(cdt, node_slot);
    if (new_parent->depth + 1u + sub_h > K3_MAX_CDT_DEPTH) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CDT_DEPTH_OVERFLOW;
    }

    /* Unlink from previous parent's child list if applicable. */
    if (node->parent_slot != K3_INVALID_ID && node->parent_slot < K3_MAX_CAPS) {
        uint32_t* link = &cdt->nodes[node->parent_slot].first_child;
        uint32_t guard = 0u;
        while (*link != K3_INVALID_ID && guard < K3_MAX_CAPS) {
            if (*link == node_slot) {
                *link = node->next_sibling;
                break;
            }
            link = &cdt->nodes[*link].next_sibling;
            guard += 1u;
        }
    }

    node->parent_slot = np_slot;
    node->next_sibling = new_parent->first_child;
    new_parent->first_child = node_slot;
    k3_cdt_update_subtree_depths_locked(cdt, node_slot, new_parent->depth + 1u);

    k3_lock_release(&cdt->lock_word);
    return K3_OK;
}

k3_status_t k3_cdt_revoke_subtree(k3_cdt_t* cdt,
                                  const k3_cap_handle_t* cap_handle,
                                  uint32_t* out_revoked_count) {
    if (!cdt || !cap_handle) {
        return K3_ERR_INVALID_ARG;
    }
    if (out_revoked_count) {
        *out_revoked_count = 0u;
    }

    k3_lock_acquire(&cdt->lock_word);
    if (cap_handle->slot_id >= K3_MAX_CAPS) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CAP_INVALID;
    }
    k3_cdt_node_t* root = &cdt->nodes[cap_handle->slot_id];
    if (k3_atomic_load_u32(&root->active) == 0u ||
        k3_atomic_load_u32(&root->generation) != cap_handle->generation) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CAP_REVOKED;
    }
    if (root->secret_badge != cap_handle->badge ||
        root->compartment_id != cap_handle->compartment_id) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CAP_CONFINEMENT;
    }
    if ((root->rights & K3_RIGHT_REVOKE) == 0u) {
        k3_lock_release(&cdt->lock_word);
        return K3_ERR_CAP_DENIED;
    }

    /* Detach root from its parent's child list if it has a parent. */
    if (root->parent_slot != K3_INVALID_ID && root->parent_slot < K3_MAX_CAPS) {
        uint32_t* link = &cdt->nodes[root->parent_slot].first_child;
        uint32_t guard = 0u;
        while (*link != K3_INVALID_ID && guard < K3_MAX_CAPS) {
            if (*link == cap_handle->slot_id) {
                *link = root->next_sibling;
                break;
            }
            link = &cdt->nodes[*link].next_sibling;
            guard += 1u;
        }
    }

    /* Explicit static DFS stack: atomically invalidate root + every transitive descendant. */
    uint32_t stack[K3_MAX_CAPS];
    uint32_t sp = 0u;
    uint32_t revoked = 0u;

    stack[sp++] = cap_handle->slot_id;
    while (sp > 0u) {
        uint32_t cur = stack[--sp];
        k3_cdt_node_t* n = &cdt->nodes[cur];

        uint32_t ch = n->first_child;
        uint32_t guard = 0u;
        while (ch != K3_INVALID_ID && ch < K3_MAX_CAPS && guard < K3_MAX_CAPS) {
            uint32_t next_sib = cdt->nodes[ch].next_sibling;
            if (sp < K3_MAX_CAPS) {
                stack[sp++] = ch;
            }
            ch = next_sib;
            guard += 1u;
        }

        if (k3_atomic_load_u32(&n->active) != 0u) {
            k3_atomic_store_u32(&n->active, 0u);
            k3_atomic_store_u32(&n->generation, n->generation + 1u);
            n->rights = 0u;
            n->secret_badge = 0u;
            n->parent_slot = K3_INVALID_ID;
            n->first_child = K3_INVALID_ID;
            n->next_sibling = K3_INVALID_ID;
            n->depth = 0u;
            if (cdt->live_count > 0u) {
                cdt->live_count -= 1u;
            }
            revoked += 1u;
        }
    }

    __atomic_fetch_add(&cdt->rev_epoch, 1u, __ATOMIC_SEQ_CST);
    if (out_revoked_count) {
        *out_revoked_count = revoked;
    }
    k3_lock_release(&cdt->lock_word);
    return K3_OK;
}

k3_status_t k3_cdt_validate(const k3_cdt_t* cdt,
                            const k3_cap_handle_t* handle,
                            uint32_t caller_compartment,
                            uint32_t required_rights,
                            uint32_t* out_resource_id) {
    if (!cdt || !handle) {
        return K3_ERR_INVALID_ARG;
    }
    if (handle->slot_id >= K3_MAX_CAPS) {
        return K3_ERR_CAP_CONFINEMENT;
    }

    /* Lock-free acquire-ordered read of active + generation + badge + confinement. */
    const k3_cdt_node_t* n = &cdt->nodes[handle->slot_id];
    uint32_t g1 = k3_atomic_load_u32(&n->generation);
    uint32_t active = k3_atomic_load_u32(&n->active);
    uint32_t badge = n->secret_badge;
    uint32_t comp = n->compartment_id;
    uint32_t rights = n->rights;
    uint32_t res = n->resource_id;
    uint32_t g2 = k3_atomic_load_u32(&n->generation);

    if (g1 != g2 || active == 0u || g1 != handle->generation) {
        /* Distinguish forged zero-badge handle from previously valid revoked handle. */
        if (handle->badge == 0u || handle->generation == 0u) {
            return K3_ERR_CAP_CONFINEMENT;
        }
        return K3_ERR_CAP_REVOKED;
    }
    if (badge != handle->badge || comp != handle->compartment_id || comp != caller_compartment) {
        return K3_ERR_CAP_CONFINEMENT;
    }
    if ((rights & required_rights) != required_rights) {
        return K3_ERR_CAP_DENIED;
    }
    if (out_resource_id) {
        *out_resource_id = res;
    }
    return K3_OK;
}

/* ────────────── Zero-Copy Session-Typed submit/complete Ring IPC ────────────── */

k3_status_t k3_ipc_channel_init(k3_ipc_channel_t* ch,
                                uint32_t channel_id,
                                uint32_t client_compartment,
                                uint32_t server_compartment) {
    if (!ch || client_compartment >= K3_MAX_COMPARTMENTS || server_compartment >= K3_MAX_COMPARTMENTS) {
        return K3_ERR_INVALID_ARG;
    }
    k3_zero_bytes(ch, sizeof(*ch));
    ch->channel_id = channel_id;
    ch->client_compartment = client_compartment;
    ch->server_compartment = server_compartment;
    ch->session_state = K3_SESSION_INIT;
    ch->pending_state = K3_SESSION_INIT;
    ch->next_seq = 1u;
    return K3_OK;
}

static int k3_session_transition_valid(k3_session_state_t cur,
                                       k3_ipc_msg_kind_t kind,
                                       uint32_t sender_comp,
                                       uint32_t client_comp,
                                       uint32_t server_comp,
                                       k3_session_state_t* out_next) {
    if (cur == K3_SESSION_CLOSED) {
        return 0;
    }
    switch (kind) {
        case K3_IPC_MSG_REQUEST:
            if (sender_comp != client_comp) {
                return 0;
            }
            if (cur == K3_SESSION_INIT || cur == K3_SESSION_RESPONSE_SENT) {
                *out_next = K3_SESSION_REQUEST_SENT;
                return 1;
            }
            return 0;
        case K3_IPC_MSG_RESPONSE:
            if (sender_comp != server_comp) {
                return 0;
            }
            if (cur == K3_SESSION_REQUEST_SENT) {
                *out_next = K3_SESSION_RESPONSE_SENT;
                return 1;
            }
            return 0;
        case K3_IPC_MSG_CLOSE:
            if (sender_comp == client_comp || sender_comp == server_comp) {
                *out_next = K3_SESSION_CLOSED;
                return 1;
            }
            return 0;
        default:
            return 0;
    }
}

k3_status_t k3_ipc_submit(k3_ipc_channel_t* ch,
                          const k3_cdt_t* cdt,
                          const k3_cap_handle_t* sender_cap,
                          uint32_t caller_compartment,
                          k3_ipc_msg_kind_t msg_kind,
                          const uint8_t* payload,
                          size_t payload_len,
                          uint64_t zero_copy_tag,
                          uint32_t* out_seq_id) {
    if (!ch || !cdt || !sender_cap || payload_len > K3_IPC_MAX_PAYLOAD) {
        return K3_ERR_INVALID_ARG;
    }
    if (payload_len > 0u && !payload) {
        return K3_ERR_INVALID_ARG;
    }

    uint32_t res_id = K3_INVALID_ID;
    k3_status_t cap_st = k3_cdt_validate(cdt, sender_cap, caller_compartment, K3_RIGHT_SEND, &res_id);
    if (cap_st != K3_OK) {
        return cap_st;
    }
    if (res_id != ch->channel_id) {
        return K3_ERR_CAP_CONFINEMENT;
    }

    k3_lock_acquire(&ch->lock_word);
    k3_session_state_t next_pending = ch->pending_state;
    if (!k3_session_transition_valid(ch->pending_state, msg_kind, caller_compartment,
                                     ch->client_compartment, ch->server_compartment,
                                     &next_pending)) {
        k3_lock_release(&ch->lock_word);
        return K3_ERR_SESSION_PROTOCOL;
    }
    if (ch->count >= K3_IPC_RING_CAP) {
        k3_lock_release(&ch->lock_word);
        return K3_ERR_IPC_RING_FULL;
    }

    k3_ipc_sqe_t* sqe = &ch->ring[ch->tail];
    k3_zero_bytes(sqe, sizeof(*sqe));
    sqe->valid = 1u;
    sqe->msg_kind = (uint32_t)msg_kind;
    sqe->seq_id = ch->next_seq++;
    sqe->payload_len = (uint32_t)payload_len;
    sqe->sender_compartment = caller_compartment;
    sqe->receiver_compartment = (caller_compartment == ch->client_compartment)
        ? ch->server_compartment : ch->client_compartment;
    sqe->sender_cap = *sender_cap;
    sqe->zero_copy_tag = zero_copy_tag;
    if (payload_len > 0u) {
        k3_copy_bytes(sqe->payload, payload, payload_len);
    }

    ch->tail = (ch->tail + 1u) & (K3_IPC_RING_CAP - 1u);
    ch->count += 1u;
    ch->pending_state = next_pending;
    ch->submitted_total += 1ULL;
    if (out_seq_id) {
        *out_seq_id = sqe->seq_id;
    }

    k3_lock_release(&ch->lock_word);
    return K3_OK;
}

k3_status_t k3_ipc_complete(k3_ipc_channel_t* ch,
                            const k3_cdt_t* cdt,
                            const k3_cap_handle_t* receiver_cap,
                            uint32_t caller_compartment,
                            uint8_t* out_buf,
                            size_t out_buf_cap,
                            size_t* out_len,
                            k3_ipc_msg_kind_t* out_kind,
                            uint64_t* out_zero_copy_tag) {
    if (!ch || !cdt || !receiver_cap) {
        return K3_ERR_INVALID_ARG;
    }

    uint32_t recv_res = K3_INVALID_ID;
    k3_status_t recv_st = k3_cdt_validate(cdt, receiver_cap, caller_compartment, K3_RIGHT_RECV, &recv_res);
    if (recv_st != K3_OK) {
        return recv_st;
    }
    if (recv_res != ch->channel_id) {
        return K3_ERR_CAP_CONFINEMENT;
    }

    k3_lock_acquire(&ch->lock_word);
    if (ch->count == 0u) {
        k3_lock_release(&ch->lock_word);
        return K3_ERR_IPC_RING_EMPTY;
    }

    k3_ipc_sqe_t* sqe = &ch->ring[ch->head];
    if (sqe->receiver_compartment != caller_compartment) {
        k3_lock_release(&ch->lock_word);
        return K3_ERR_CAP_CONFINEMENT;
    }

    /* CRITICAL ATOMIC IN-FLIGHT CHECK:
     * Re-validate sender's capability BEFORE mutating out_buf or receiver state.
     * If revoked mid-flight (directly or via any CDT ancestor), abort with ZERO side effects! */
    uint32_t send_res = K3_INVALID_ID;
    k3_status_t send_st = k3_cdt_validate(cdt, &sqe->sender_cap, sqe->sender_compartment,
                                          K3_RIGHT_SEND, &send_res);
    if (send_st != K3_OK || send_res != ch->channel_id) {
        sqe->valid = 0u;
        ch->head = (ch->head + 1u) & (K3_IPC_RING_CAP - 1u);
        ch->count -= 1u;
        ch->pending_state = ch->session_state;
        ch->revoked_aborts += 1ULL;
        k3_lock_release(&ch->lock_word);
        return K3_ERR_CAP_REVOKED;
    }

    if ((size_t)sqe->payload_len > out_buf_cap || (sqe->payload_len > 0u && !out_buf)) {
        k3_lock_release(&ch->lock_word);
        return K3_ERR_BUFFER_TOO_SMALL;
    }

    k3_session_state_t next_session = ch->session_state;
    if (!k3_session_transition_valid(ch->session_state, (k3_ipc_msg_kind_t)sqe->msg_kind,
                                     sqe->sender_compartment, ch->client_compartment,
                                     ch->server_compartment, &next_session)) {
        k3_lock_release(&ch->lock_word);
        return K3_ERR_SESSION_PROTOCOL;
    }

    if (sqe->payload_len > 0u) {
        k3_copy_bytes(out_buf, sqe->payload, (size_t)sqe->payload_len);
    }
    if (out_len) {
        *out_len = (size_t)sqe->payload_len;
    }
    if (out_kind) {
        *out_kind = (k3_ipc_msg_kind_t)sqe->msg_kind;
    }
    if (out_zero_copy_tag) {
        *out_zero_copy_tag = sqe->zero_copy_tag;
    }

    sqe->valid = 0u;
    ch->head = (ch->head + 1u) & (K3_IPC_RING_CAP - 1u);
    ch->count -= 1u;
    ch->session_state = next_session;
    ch->completed_total += 1ULL;

    k3_lock_release(&ch->lock_word);
    return K3_OK;
}

/* ─────────────────── Shared-Memory Compartment Confinement ─────────────────── */

k3_status_t k3_shm_page_init(k3_shm_page_t* page,
                             uint32_t page_id,
                             uint32_t owner_compartment,
                             uint32_t peer_compartment) {
    if (!page || owner_compartment >= K3_MAX_COMPARTMENTS || peer_compartment >= K3_MAX_COMPARTMENTS) {
        return K3_ERR_INVALID_ARG;
    }
    k3_zero_bytes(page, sizeof(*page));
    page->page_id = page_id;
    page->owner_compartment = owner_compartment;
    page->peer_compartment = peer_compartment;
    page->cap_transfer_forbidden = 1u;
    return K3_OK;
}

k3_status_t k3_shm_write(k3_shm_page_t* page,
                         const k3_cdt_t* cdt,
                         const k3_cap_handle_t* cap,
                         uint32_t caller_compartment,
                         size_t offset,
                         const uint8_t* src,
                         size_t len) {
    if (!page || !cdt || !cap || (!src && len > 0u)) {
        return K3_ERR_INVALID_ARG;
    }
    if (offset > K3_SHM_PAGE_BYTES || len > (K3_SHM_PAGE_BYTES - offset)) {
        return K3_ERR_SHM_BOUNDS;
    }
    uint32_t res_id = K3_INVALID_ID;
    k3_status_t st = k3_cdt_validate(cdt, cap, caller_compartment, K3_RIGHT_SHM_WRITE, &res_id);
    if (st != K3_OK) {
        return st;
    }
    if (res_id != page->page_id ||
        (caller_compartment != page->owner_compartment && caller_compartment != page->peer_compartment)) {
        return K3_ERR_CAP_CONFINEMENT;
    }
    if (len > 0u) {
        k3_copy_bytes(page->data + offset, src, len);
    }
    return K3_OK;
}

k3_status_t k3_shm_read(const k3_shm_page_t* page,
                        const k3_cdt_t* cdt,
                        const k3_cap_handle_t* cap,
                        uint32_t caller_compartment,
                        size_t offset,
                        uint8_t* dst,
                        size_t len) {
    if (!page || !cdt || !cap || (!dst && len > 0u)) {
        return K3_ERR_INVALID_ARG;
    }
    if (offset > K3_SHM_PAGE_BYTES || len > (K3_SHM_PAGE_BYTES - offset)) {
        return K3_ERR_SHM_BOUNDS;
    }
    uint32_t res_id = K3_INVALID_ID;
    k3_status_t st = k3_cdt_validate(cdt, cap, caller_compartment, K3_RIGHT_SHM_READ, &res_id);
    if (st != K3_OK) {
        return st;
    }
    if (res_id != page->page_id ||
        (caller_compartment != page->owner_compartment && caller_compartment != page->peer_compartment)) {
        return K3_ERR_CAP_CONFINEMENT;
    }
    if (len > 0u) {
        k3_copy_bytes(dst, page->data + offset, len);
    }
    return K3_OK;
}

k3_status_t k3_shm_attempt_import_cap(const k3_shm_page_t* page,
                                      const k3_cdt_t* cdt,
                                      uint32_t caller_compartment,
                                      size_t offset,
                                      k3_cap_handle_t* out_cap) {
    (void)page;
    (void)cdt;
    (void)caller_compartment;
    (void)offset;
    if (out_cap) {
        k3_zero_bytes(out_cap, sizeof(*out_cap));
    }
    /* Raw shared-memory bytes are pure data and may NEVER forge or transfer capabilities. */
    return K3_ERR_CAP_CONFINEMENT;
}
