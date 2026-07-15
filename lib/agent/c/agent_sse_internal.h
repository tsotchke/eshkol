#ifndef ESHKOL_AGENT_SSE_INTERNAL_H
#define ESHKOL_AGENT_SSE_INTERNAL_H

#include <stddef.h>
#include "eshkol/agent_http.h"

typedef struct eshkol_sse_parser eshkol_sse_parser_t;

eshkol_sse_parser_t* eshkol_sse_parser_create(void);
void eshkol_sse_parser_destroy(eshkol_sse_parser_t* parser);
int eshkol_sse_parser_feed(eshkol_sse_parser_t* parser,
                           const char* bytes,
                           size_t length);
eshkol_sse_event_t* eshkol_sse_parser_next(eshkol_sse_parser_t* parser);
int eshkol_sse_parser_has_complete_event(eshkol_sse_parser_t* parser);
int eshkol_sse_parser_failed(const eshkol_sse_parser_t* parser);

#endif
