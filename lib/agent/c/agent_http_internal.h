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
 * eshkol_http_request_offending_field all come from there. */
#include <eshkol/core/http_syntax.h>

#endif
