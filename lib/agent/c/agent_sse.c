#include "agent_sse_internal.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ESHKOL_SSE_MAX_BUFFER_BYTES ((size_t)8 * 1024 * 1024)

struct eshkol_sse_event {
    char* type;
    char* data;
    char* id;
    int64_t retry_ms;
};

struct eshkol_sse_parser {
    char* buffer;
    size_t length;
    size_t capacity;
    char* last_event_id;
    int bom_handled;
    int failed;
};

static char* sse_copy(const char* bytes, size_t length) {
    char* out = (char*)malloc(length + 1);
    if (!out) return NULL;
    if (length) memcpy(out, bytes, length);
    out[length] = '\0';
    return out;
}

static int sse_append(char** dst, size_t* length,
                      const char* bytes, size_t count, int add_newline) {
    size_t extra = count + (add_newline ? 1u : 0u);
    if (*length > ESHKOL_SSE_MAX_BUFFER_BYTES - extra) return 0;
    char* grown = (char*)realloc(*dst, *length + extra + 1);
    if (!grown) return 0;
    *dst = grown;
    if (count) memcpy(grown + *length, bytes, count);
    *length += count;
    if (add_newline) grown[(*length)++] = '\n';
    grown[*length] = '\0';
    return 1;
}

eshkol_sse_parser_t* eshkol_sse_parser_create(void) {
    return (eshkol_sse_parser_t*)calloc(1, sizeof(eshkol_sse_parser_t));
}

void eshkol_sse_parser_destroy(eshkol_sse_parser_t* parser) {
    if (!parser) return;
    free(parser->buffer);
    free(parser->last_event_id);
    free(parser);
}

int eshkol_sse_parser_feed(eshkol_sse_parser_t* parser,
                           const char* bytes,
                           size_t length) {
    if (!parser || parser->failed || (!bytes && length)) return 0;
    if (length > ESHKOL_SSE_MAX_BUFFER_BYTES - parser->length) {
        parser->failed = 1;
        return 0;
    }
    size_t required = parser->length + length + 1;
    if (required > parser->capacity) {
        size_t capacity = parser->capacity ? parser->capacity : 4096;
        while (capacity < required) {
            if (capacity > ESHKOL_SSE_MAX_BUFFER_BYTES / 2) {
                capacity = ESHKOL_SSE_MAX_BUFFER_BYTES + 1;
                break;
            }
            capacity *= 2;
        }
        if (capacity < required || capacity > ESHKOL_SSE_MAX_BUFFER_BYTES + 1) {
            parser->failed = 1;
            return 0;
        }
        char* grown = (char*)realloc(parser->buffer, capacity);
        if (!grown) {
            parser->failed = 1;
            return 0;
        }
        parser->buffer = grown;
        parser->capacity = capacity;
    }
    if (length) memcpy(parser->buffer + parser->length, bytes, length);
    parser->length += length;
    parser->buffer[parser->length] = '\0';
    return 1;
}

static void handle_initial_bom(eshkol_sse_parser_t* parser) {
    if (!parser || parser->bom_handled || parser->length == 0) return;
    const unsigned char* bytes = (const unsigned char*)parser->buffer;
    if (bytes[0] != 0xef ||
        (parser->length >= 2 && bytes[1] != 0xbb) ||
        (parser->length >= 3 && bytes[2] != 0xbf)) {
        parser->bom_handled = 1;
        return;
    }
    if (parser->length < 3) return;
    memmove(parser->buffer, parser->buffer + 3, parser->length - 3);
    parser->length -= 3;
    parser->buffer[parser->length] = '\0';
    parser->bom_handled = 1;
}

static size_t line_end(const char* buffer, size_t length,
                       size_t start, size_t* content_length) {
    size_t cursor = start;
    while (cursor < length && buffer[cursor] != '\r' && buffer[cursor] != '\n')
        cursor++;
    if (cursor == length) return 0;
    *content_length = cursor - start;
    if (buffer[cursor] == '\r') {
        if (cursor + 1 == length) return 0; /* A fragmented CRLF is ambiguous. */
        cursor++;
        if (buffer[cursor] == '\n') cursor++;
    } else {
        cursor++;
    }
    return cursor;
}

static size_t complete_event_size(eshkol_sse_parser_t* parser) {
    if (!parser || !parser->buffer) return 0;
    handle_initial_bom(parser);
    size_t line_start = 0;
    while (line_start < parser->length) {
        size_t content_length = 0;
        size_t next = line_end(parser->buffer, parser->length,
                               line_start, &content_length);
        if (!next) return 0;
        if (content_length == 0) return next;
        line_start = next;
    }
    return 0;
}

int eshkol_sse_parser_has_complete_event(eshkol_sse_parser_t* parser) {
    return complete_event_size(parser) != 0;
}

int eshkol_sse_parser_failed(const eshkol_sse_parser_t* parser) {
    return !parser || parser->failed;
}

static int parse_retry(const char* value, size_t length, int64_t* retry_ms) {
    if (!length) return 0;
    int64_t result = 0;
    for (size_t i = 0; i < length; ++i) {
        unsigned char c = (unsigned char)value[i];
        if (!isdigit(c)) return 0;
        if (result > (INT64_MAX - (c - '0')) / 10) return 0;
        result = result * 10 + (c - '0');
    }
    *retry_ms = result;
    return 1;
}

eshkol_sse_event_t* eshkol_sse_parser_next(eshkol_sse_parser_t* parser) {
    for (;;) {
        if (!parser || parser->failed) return NULL;
        size_t consumed = complete_event_size(parser);
        if (!consumed) return NULL;

        char* type = NULL;
        char* data = NULL;
        char* id_update = NULL;
        size_t data_length = 0;
        int saw_data = 0;
        int saw_id = 0;
        int64_t retry_ms = -1;
        size_t line_start = 0;

        while (line_start < consumed) {
            size_t line_length = 0;
            size_t next = line_end(parser->buffer, consumed, line_start,
                                   &line_length);
            if (!next || line_length == 0) break;

            const char* line = parser->buffer + line_start;
            if (line[0] != ':') {
                size_t colon = 0;
                while (colon < line_length && line[colon] != ':') colon++;
                const char* value = colon < line_length
                                        ? line + colon + 1
                                        : line + line_length;
                size_t value_length = colon < line_length
                                          ? line_length - colon - 1
                                          : 0;
                if (value_length && value[0] == ' ') {
                    value++;
                    value_length--;
                }
                if (memchr(value, '\0', value_length)) goto invalid_text;

                if (colon == 4 && memcmp(line, "data", 4) == 0) {
                    if (!sse_append(&data, &data_length, value, value_length, 1))
                        goto oom;
                    saw_data = 1;
                } else if (colon == 5 && memcmp(line, "event", 5) == 0) {
                    char* replacement = sse_copy(value, value_length);
                    if (!replacement) goto oom;
                    free(type);
                    type = replacement;
                } else if (colon == 2 && memcmp(line, "id", 2) == 0) {
                    char* replacement = sse_copy(value, value_length);
                    if (!replacement) goto oom;
                    free(id_update);
                    id_update = replacement;
                    saw_id = 1;
                } else if (colon == 5 && memcmp(line, "retry", 5) == 0) {
                    (void)parse_retry(value, value_length, &retry_ms);
                }
            }
            line_start = next;
        }

        memmove(parser->buffer, parser->buffer + consumed,
                parser->length - consumed);
        parser->length -= consumed;
        parser->buffer[parser->length] = '\0';

        if (saw_id) {
            free(parser->last_event_id);
            parser->last_event_id = id_update;
            id_update = NULL;
        }

        if (!saw_data) {
            free(type);
            free(data);
            free(id_update);
            continue;
        }
        if (data_length && data[data_length - 1] == '\n')
            data[--data_length] = '\0';

        eshkol_sse_event_t* event =
            (eshkol_sse_event_t*)calloc(1, sizeof(*event));
        if (!event) goto oom;
        event->type = type ? type : sse_copy("message", 7);
        event->data = data ? data : sse_copy("", 0);
        event->id = sse_copy(parser->last_event_id ? parser->last_event_id : "",
                             parser->last_event_id
                                 ? strlen(parser->last_event_id) : 0);
        event->retry_ms = retry_ms;
        if (!event->type || !event->data || !event->id) {
            eshkol_sse_event_free(event);
            parser->failed = 1;
            return NULL;
        }
        free(id_update);
        return event;

oom:
        free(type);
        free(data);
        free(id_update);
        parser->failed = 1;
        return NULL;
invalid_text:
        free(type);
        free(data);
        free(id_update);
        parser->failed = 1;
        return NULL;
    }
}

const char* eshkol_sse_event_type(const eshkol_sse_event_t* event) {
    return event ? event->type : NULL;
}

const char* eshkol_sse_event_data(const eshkol_sse_event_t* event) {
    return event ? event->data : NULL;
}

const char* eshkol_sse_event_id(const eshkol_sse_event_t* event) {
    return event ? event->id : NULL;
}

int64_t eshkol_sse_event_retry_ms(const eshkol_sse_event_t* event) {
    return event ? event->retry_ms : -1;
}

void eshkol_sse_event_free(eshkol_sse_event_t* event) {
    if (!event) return;
    free(event->type);
    free(event->data);
    free(event->id);
    free(event);
}
