#include "agent_sse_internal.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(int condition, const char* name) {
    if (condition) printf("PASS: %s\n", name);
    else {
        fprintf(stderr, "FAIL: %s\n", name);
        failures++;
    }
}

static void test_fragmented_bom_and_line_endings(void) {
    static const unsigned char input[] =
        "\xef\xbb\xbfid: seed\r\n"
        "event: update\r"
        "retry: 42\n"
        "data: alpha\r\n\r\n"
        "data: beta\n\n";
    eshkol_sse_parser_t* parser = eshkol_sse_parser_create();
    check(parser != NULL, "fragmented parser allocates");
    if (!parser) return;
    int fed = 1;
    for (size_t i = 0; i < sizeof(input) - 1; ++i)
        fed = fed && eshkol_sse_parser_feed(parser, (const char*)input + i, 1);
    check(fed, "one-byte SSE fragments feed");
    eshkol_sse_event_t* first = eshkol_sse_parser_next(parser);
    check(first && strcmp(eshkol_sse_event_type(first), "update") == 0 &&
          strcmp(eshkol_sse_event_data(first), "alpha") == 0 &&
          strcmp(eshkol_sse_event_id(first), "seed") == 0 &&
          eshkol_sse_event_retry_ms(first) == 42,
          "BOM, CR, LF, CRLF and fragmented fields parse exactly");
    eshkol_sse_event_free(first);
    eshkol_sse_event_t* second = eshkol_sse_parser_next(parser);
    check(second && strcmp(eshkol_sse_event_data(second), "beta") == 0 &&
          strcmp(eshkol_sse_event_id(second), "seed") == 0,
          "SSE last-event-id persists across dispatched events");
    eshkol_sse_event_free(second);
    check(!eshkol_sse_parser_failed(parser), "valid fragmented stream stays healthy");
    eshkol_sse_parser_destroy(parser);
}

static void test_non_dispatching_id_block(void) {
    static const char input[] =
        ": heartbeat\n"
        "id: checkpoint-9\n\n"
        "data: resumed\n\n";
    eshkol_sse_parser_t* parser = eshkol_sse_parser_create();
    check(parser && eshkol_sse_parser_feed(parser, input, sizeof(input) - 1),
          "directive-only stream feeds");
    if (!parser) return;
    eshkol_sse_event_t* event = eshkol_sse_parser_next(parser);
    check(event && strcmp(eshkol_sse_event_data(event), "resumed") == 0 &&
          strcmp(eshkol_sse_event_id(event), "checkpoint-9") == 0,
          "id-only blocks update state without emitting phantom events");
    eshkol_sse_event_free(event);
    eshkol_sse_parser_destroy(parser);
}

static void test_invalid_embedded_nul(void) {
    static const char input[] = {'d','a','t','a',':',' ','a','\0','b','\n','\n'};
    eshkol_sse_parser_t* parser = eshkol_sse_parser_create();
    check(parser && eshkol_sse_parser_feed(parser, input, sizeof(input)),
          "NUL fixture reaches parser");
    if (!parser) return;
    check(eshkol_sse_parser_next(parser) == NULL &&
          eshkol_sse_parser_failed(parser),
          "embedded NUL fails explicitly instead of truncating SSE data");
    eshkol_sse_parser_destroy(parser);
}

int main(void) {
    test_fragmented_bom_and_line_endings();
    test_non_dispatching_id_block();
    test_invalid_embedded_nul();
    printf("Agent SSE parser checks: %s (%d failure%s)\n",
           failures ? "FAIL" : "PASS", failures,
           failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
