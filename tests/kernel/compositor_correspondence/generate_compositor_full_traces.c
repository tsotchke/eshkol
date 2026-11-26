/*
 * generate_compositor_full_traces.c
 * Purpose: run random traces on the native K5 compositor and record the complete
 * observable state after every operation, for full-trace correspondence with
 * EshkolCompositorAligned.lean.
 *
 * Handle correspondence (stated here because it is where a security-relevant
 * difference could hide):
 *   model handle <id, badge>  <->  native handle (slot = id, generation and cap_id as
 *     issued for slot id, owner_badge = badge).
 *   A native handle whose (slot, generation, cap_id) was never issued corresponds to a
 *   model id that names no surface (the adapter uses id 1000; at most 16 exist).
 *   This trace exercises three such handles: a failed registration's handle
 *   (generation 0, never issued), an issued handle with generation + 1, and an
 *   issued handle with cap_id + 100. Each must be refused by the native code; if
 *   one is accepted, the model refuses it and the replay reports a divergence.
 *
 * Output, one line per operation:
 *   trace <n> width <W>
 *   reg <owner> <xStart> <xLen> <writable> => <STATUS> | <state>
 *   write <i> <tamper> <badge> <x> <value> => <STATUS> | <state>
 *     i: registration attempt index; tamper: none | generation | capability
 *   <state> = fb <c0> ... <cW-1> surf <owner>,<x>,<len>,<w> ...   (active slots, slot order)
 * Deterministic in the seed.
 */
#include <eshkol/core/os_k5_k6_services.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXIMUM_ATTEMPTS 64u
#define MAXIMUM_WIDTH 24u

static uint64_t random_state;
static uint32_t random_below(uint32_t bound) {
    random_state ^= random_state << 13; random_state ^= random_state >> 7; random_state ^= random_state << 17;
    return (uint32_t)(random_state % bound);
}

static const char* status_name(eshkol_os_k56_status_t s) {
    switch (s) {
    case ESHKOL_K56_OK: return "OK";
    case ESHKOL_K56_ERR_INVALID_ARG: return "INVALID_ARG";
    case ESHKOL_K56_ERR_NO_SLOTS: return "NO_SLOTS";
    case ESHKOL_COMP_ERR_SURFACE_CONFLICT: return "SURFACE_CONFLICT";
    case ESHKOL_COMP_ERR_RO_TRAP: return "RO_TRAP";
    case ESHKOL_COMP_ERR_BOUNDS: return "BOUNDS";
    case ESHKOL_COMP_ERR_BAD_SURFACE: return "BAD_SURFACE";
    default: return "OTHER";
    }
}

static void print_state(const eshkol_os_compositor_t* comp, const uint32_t* framebuffer, uint32_t width) {
    printf(" | fb");
    for (uint32_t c = 0; c < width; c++) printf(" %u", framebuffer[c]);
    printf(" surf");
    for (uint32_t i = 0; i < ESHKOL_OS_COMP_MAX_SURFACES; i++) {
        const eshkol_os_surface_desc_t* s = &comp->surfaces[i];
        if (s->active)
            printf(" %u,%u,%u,%d", s->owner_badge, s->x, s->width, (s->rights & ESHKOL_CAP_RIGHT_WRITE) != 0ULL);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <seed> <trace_count> <operations_per_trace>\n", argv[0]); return 1; }
    random_state = strtoull(argv[1], 0, 10) | 1u;
    unsigned trace_count = (unsigned)atoi(argv[2]), operation_count = (unsigned)atoi(argv[3]);
    if (operation_count > MAXIMUM_ATTEMPTS) return 1;
    static eshkol_os_compositor_t comp;
    uint32_t framebuffer[MAXIMUM_WIDTH];
    for (unsigned t = 0; t < trace_count; t++) {
        uint32_t width = 4u + random_below(MAXIMUM_WIDTH - 3u);
        if (eshkol_os_compositor_init(&comp, framebuffer, width, 1u) != ESHKOL_K56_OK) return 3;
        printf("trace %u width %u\n", t, width);
        /* One trace in ten registers aggressively, to reach the slot limit. */
        uint32_t registration_odds = random_below(10) == 0 ? 4u : 1u;
        uint32_t owner[MAXIMUM_ATTEMPTS], x_start[MAXIMUM_ATTEMPTS], x_len[MAXIMUM_ATTEMPTS];
        eshkol_os_surface_handle_t handle[MAXIMUM_ATTEMPTS];
        unsigned attempts = 0;
        for (unsigned k = 0; k < operation_count; k++) {
            if (attempts == 0 || random_below(5) < registration_odds) {
                unsigned i = attempts++;
                owner[i] = random_below(30) == 0 ? 0u : 1u + random_below(3);
                x_len[i] = random_below(30) == 0 ? 0u : 1u + random_below(width / 2u);
                uint32_t span = random_below(10) == 0 ? width + 2u : width - x_len[i] + 1u;
                x_start[i] = random_below(span);
                int writable = random_below(4) != 0;
                eshkol_os_k56_status_t st = eshkol_os_compositor_create_surface(
                    &comp, owner[i], (eshkol_cap_id_t)(i + 1u),
                    ESHKOL_CAP_RIGHT_READ | (writable ? ESHKOL_CAP_RIGHT_WRITE : 0ULL),
                    x_start[i], 0u, x_len[i], 1u, &handle[i]);
                if (st != ESHKOL_K56_OK) {
                    handle[i].surface_id = 0u; handle[i].generation = 0u;
                    handle[i].owner_badge = owner[i]; handle[i].cap_id = (eshkol_cap_id_t)(i + 1u);
                }
                printf("reg %u %u %u %d => %s", owner[i], x_start[i], x_len[i], writable, status_name(st));
            } else {
                unsigned i = random_below(attempts);
                uint32_t badge = random_below(5) == 0 ? random_below(4) : owner[i];
                uint32_t x = random_below(5) == 0 ? random_below(width + 2u)
                                                  : x_start[i] + random_below(x_len[i] + 2u);
                uint32_t value = 1u + random_below(1000);
                uint32_t tamper_choice = random_below(20);
                const char* tamper = tamper_choice == 0 ? "generation" : tamper_choice == 1 ? "capability" : "none";
                eshkol_os_surface_handle_t h = handle[i];
                h.owner_badge = badge;
                if (tamper_choice == 0) h.generation += 1u;
                if (tamper_choice == 1) h.cap_id += 100u;
                eshkol_os_k56_status_t st = eshkol_os_compositor_write_rect(&comp, &h, x, 0u, 1u, 1u, &value);
                printf("write %u %s %u %u %u => %s", i, tamper, badge, x, value, status_name(st));
            }
            print_state(&comp, framebuffer, width);
        }
    }
    return 0;
}
