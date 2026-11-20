#ifndef ESHKOL_AGENT_HTTP_INTERNAL_H
#define ESHKOL_AGENT_HTTP_INTERNAL_H

/* The agent HTTP client reads the request grammar from the same place the
 * core builtins and the bytecode VM read it, so a method, a request target or
 * a header line that one engine refuses is refused by all of them.
 *
 * <eshkol/core/http_syntax.h> is that one place: eshkol_http_valid_method,
 * eshkol_http_valid_token, eshkol_http_valid_field_value,
 * eshkol_http_valid_header_line, eshkol_http_valid_request_target,
 * eshkol_http_valid_header_block, eshkol_http_render_header_block and
 * eshkol_http_request_offending_field and eshkol_http_reserved_field_name
 * all come from there. */
#include <eshkol/core/http_syntax.h>


/**
 * @brief True when a caller-supplied header line may go on the wire.
 *
 * The line must fit the grammar and must not name a field the client writes
 * itself. Every native client frames the body from the request it was given,
 * so a caller-supplied Content-Length or Transfer-Encoding beside that framing
 * is what RFC 9112 section 6.3 forbids a sender emitting, and what a front end
 * and a back end disagreeing about becomes request smuggling.
 */
static inline int eshkol_http_header_line_admissible(const char* line) {
    const char* colon;
    if (!line || !eshkol_http_valid_header_line(line)) return 0;
    colon = strchr(line, ':');
    if (!colon) return 0;
    return !eshkol_http_reserved_field_name(line, (size_t)(colon - line), 0);
}

#endif
