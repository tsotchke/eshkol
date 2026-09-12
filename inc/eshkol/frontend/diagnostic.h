#ifndef ESHKOL_FRONTEND_DIAGNOSTIC_H
#define ESHKOL_FRONTEND_DIAGNOSTIC_H

#include <stdint.h>

#include <eshkol/frontend/node_identity.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESHKOL_DIAGNOSTIC_V1_SCHEMA 1u

typedef enum eshkol_diagnostic_severity {
    ESHKOL_DIAGNOSTIC_NOTE = 0,
    ESHKOL_DIAGNOSTIC_WARNING = 1,
    ESHKOL_DIAGNOSTIC_ERROR = 2
} eshkol_diagnostic_severity_t;

typedef struct eshkol_diagnostic_v1 {
    uint32_t schema;
    eshkol_diagnostic_severity_t severity;
    eshkol_node_id_t node_id;
    eshkol_source_span_t span;
    const char* code;
    const char* message;
} eshkol_diagnostic_v1_t;

typedef void (*eshkol_diagnostic_sink_v1)(const eshkol_diagnostic_v1_t* diagnostic,
                                           void* userdata);

void eshkol_diagnostic_set_sink_v1(eshkol_diagnostic_sink_v1 sink, void* userdata);
void eshkol_diagnostic_emit_v1(eshkol_diagnostic_severity_t severity,
                               eshkol_node_id_t node_id,
                               const char* code,
                               const char* message);
uint64_t eshkol_diagnostic_count_v1(void);
void eshkol_diagnostic_reset_v1(void);

#ifdef __cplusplus
}
#endif

#endif
