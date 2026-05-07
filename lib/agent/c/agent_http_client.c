/*
 * agent_http_client.c — libcurl-backed HTTP client.
 *
 * Provides the qllm_http_* symbols that lib/agent/http.esk binds against,
 * so Eshkol agents can make live HTTP/HTTPS calls without linking the
 * full qLLM library. Built only when libcurl is available; otherwise
 * the agent falls back to error-returning stubs (existing behavior).
 *
 * ABI mirrors the qLLM client (see header at top of http.esk):
 *   qllm_http_init/shutdown/has_ssl
 *   qllm_http_get(url, timeout_ms)
 *   qllm_http_post(url, headers[], hdr_count, body, body_len, timeout_ms)
 *   qllm_http_post_json(url, body, auth_header, timeout_ms)
 *   qllm_http_response_status/body/body_len/free
 *   qllm_http_error_string(code)
 *
 * SSE streaming (qllm_http_stream_*) is not yet implemented here — the
 * agent's streaming path still requires a follow-up. Callers that need
 * streaming today should keep falling back to the existing weak-stub
 * behavior (returns NULL, surfaces "explicit unavailable error").
 *
 * Threading: libcurl's "easy" interface is per-thread. Each request
 * creates and destroys its own CURL handle so the FFI is safe to call
 * from multiple Eshkol threads concurrently.
 *
 * Memory ownership: every successful call returns a heap-allocated
 * response struct. Callers MUST invoke qllm_http_response_free() to
 * release it, otherwise the body buffer leaks. The body string is
 * NUL-terminated for compatibility with Eshkol's string operations
 * but body_len reports the actual byte count (binary-safe).
 */

#ifdef ESHKOL_HAVE_LIBCURL

#include <curl/curl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct qllm_http_response {
    int32_t  status;        /* HTTP status code, 0 if curl couldn't connect */
    char*    body;          /* malloc'd, NUL-terminated */
    int64_t  body_len;      /* byte count excluding terminator */
    char*    error;         /* malloc'd error string or NULL on success */
} qllm_http_response_t;

/* libcurl global init is required exactly once before any curl_easy_*.
 * We track init state under a mutex so concurrent http-init calls are
 * harmless. shutdown decrements a counter — only the last shutdown
 * call actually tears libcurl down, so a worker thread that stays
 * alive past the agent's main shutdown doesn't lose curl access. */
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_init_refs = 0;

static void* xmalloc(size_t n) {
    void* p = malloc(n);
    if (!p) {
        fprintf(stderr, "agent_http_client: out of memory (%zu bytes)\n", n);
        abort();
    }
    return p;
}

static char* xstrdup_n(const char* src, size_t n) {
    char* out = (char*)xmalloc(n + 1);
    if (n > 0 && src) memcpy(out, src, n);
    out[n] = '\0';
    return out;
}

/* Body accumulator. libcurl calls write_cb in chunks; we grow the
 * buffer geometrically to keep amortized cost O(n). */
typedef struct {
    char*  data;
    size_t len;
    size_t cap;
} body_buf_t;

static size_t write_cb(void* ptr, size_t size, size_t nmemb, void* userdata) {
    body_buf_t* buf = (body_buf_t*)userdata;
    size_t add = size * nmemb;
    if (buf->len + add + 1 > buf->cap) {
        size_t new_cap = buf->cap == 0 ? 4096 : buf->cap * 2;
        while (new_cap < buf->len + add + 1) new_cap *= 2;
        char* grown = (char*)realloc(buf->data, new_cap);
        if (!grown) return 0;  /* signals curl to abort */
        buf->data = grown;
        buf->cap = new_cap;
    }
    memcpy(buf->data + buf->len, ptr, add);
    buf->len += add;
    buf->data[buf->len] = '\0';
    return add;
}

/* Configure shared curl options for all requests. Centralizing this
 * means TLS cert verification and redirect policy are consistent
 * across get/post/post_json. */
static void apply_common_opts(CURL* curl, body_buf_t* buf, int32_t timeout_ms) {
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, buf);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);  /* thread-safety: no SIGPIPE handler */
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, (long)(timeout_ms > 0 ? timeout_ms : 30000));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 10000L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "eshkol-agent/1.2");
    /* TLS verification: ON by default. Callers wanting to talk to
     * self-signed dev servers must do so via subprocess curl until we
     * add an explicit insecure-mode flag. */
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
}

static qllm_http_response_t* finalize_response(CURL* curl, body_buf_t* buf, CURLcode rc) {
    qllm_http_response_t* resp = (qllm_http_response_t*)xmalloc(sizeof(*resp));
    memset(resp, 0, sizeof(*resp));
    if (rc != CURLE_OK) {
        resp->status = 0;
        resp->error = xstrdup_n(curl_easy_strerror(rc), strlen(curl_easy_strerror(rc)));
        free(buf->data);
    } else {
        long http_status = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_status);
        resp->status = (int32_t)http_status;
        /* Always hand back a non-NULL body buffer so callers can do
         * string ops without first nullchecking. Empty body → "". */
        if (buf->data == NULL) {
            resp->body = xstrdup_n("", 0);
            resp->body_len = 0;
        } else {
            resp->body = buf->data;     /* transfer ownership */
            resp->body_len = (int64_t)buf->len;
        }
    }
    return resp;
}

/* ============================================================================
 * Public ABI — qllm_http_*
 * ============================================================================ */

int32_t qllm_http_init(void) {
    pthread_mutex_lock(&g_init_mutex);
    if (g_init_refs == 0) {
        CURLcode rc = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (rc != CURLE_OK) {
            pthread_mutex_unlock(&g_init_mutex);
            return 0;
        }
    }
    g_init_refs++;
    pthread_mutex_unlock(&g_init_mutex);
    return 1;
}

void qllm_http_shutdown(void) {
    pthread_mutex_lock(&g_init_mutex);
    if (g_init_refs > 0) {
        g_init_refs--;
        if (g_init_refs == 0) {
            curl_global_cleanup();
        }
    }
    pthread_mutex_unlock(&g_init_mutex);
}

int32_t qllm_http_has_ssl(void) {
    /* libcurl built without TLS would still link; check the feature
     * bitmap to give an accurate answer. */
    curl_version_info_data* info = curl_version_info(CURLVERSION_NOW);
    if (!info) return 0;
    return (info->features & CURL_VERSION_SSL) ? 1 : 0;
}

qllm_http_response_t* qllm_http_get(const char* url, int32_t timeout_ms) {
    if (!url) return NULL;
    /* Lazy-init: the agent may forget to call http-init before its first
     * request. One-shot init+leak is cheaper than a hard error here. */
    if (g_init_refs == 0) qllm_http_init();

    CURL* curl = curl_easy_init();
    if (!curl) return NULL;

    body_buf_t buf = {0};
    apply_common_opts(curl, &buf, timeout_ms);
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);

    CURLcode rc = curl_easy_perform(curl);
    qllm_http_response_t* resp = finalize_response(curl, &buf, rc);
    curl_easy_cleanup(curl);
    return resp;
}

/* qllm_http_post takes header strings as a flat "Name: value\0Name: value\0"
 * style array — that's the layout the qLLM C client uses. Eshkol's
 * http.esk currently doesn't pass headers through this path (it goes
 * via http-post-json-raw), but provide the full ABI for forward compat. */
qllm_http_response_t* qllm_http_post(const char* url,
                                     const char** headers,
                                     int64_t header_count,
                                     const char* body,
                                     int64_t body_len,
                                     int32_t timeout_ms) {
    if (!url) return NULL;
    if (g_init_refs == 0) qllm_http_init();

    CURL* curl = curl_easy_init();
    if (!curl) return NULL;

    body_buf_t buf = {0};
    apply_common_opts(curl, &buf, timeout_ms);
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    if (body && body_len > 0) {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)body_len);
    } else {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, 0L);
    }

    struct curl_slist* slist = NULL;
    for (int64_t i = 0; i < header_count && headers; i++) {
        if (headers[i]) slist = curl_slist_append(slist, headers[i]);
    }
    if (slist) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist);

    CURLcode rc = curl_easy_perform(curl);
    qllm_http_response_t* resp = finalize_response(curl, &buf, rc);
    if (slist) curl_slist_free_all(slist);
    curl_easy_cleanup(curl);
    return resp;
}

/* Convenience wrapper used by Anthropic API calls etc.: POSTs a JSON
 * body with Content-Type application/json plus an optional auth header.
 * The auth_header string is taken verbatim — caller is responsible
 * for choosing "Authorization: Bearer ..." vs "x-api-key: ..." etc. */
qllm_http_response_t* qllm_http_post_json(const char* url,
                                          const char* body,
                                          const char* auth_header,
                                          int32_t timeout_ms) {
    if (!url) return NULL;
    if (g_init_refs == 0) qllm_http_init();

    CURL* curl = curl_easy_init();
    if (!curl) return NULL;

    body_buf_t buf = {0};
    apply_common_opts(curl, &buf, timeout_ms);
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);

    size_t blen = body ? strlen(body) : 0;
    if (blen > 0) {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)blen);
    } else {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, 0L);
    }

    struct curl_slist* slist = NULL;
    slist = curl_slist_append(slist, "Content-Type: application/json");
    slist = curl_slist_append(slist, "Accept: application/json");
    /* The agent's http-safe-string check already rejects CRLF/NUL/etc.,
     * so it's safe to inject auth_header verbatim into the slist. We
     * still skip empty/NULL to avoid sending a blank header line. */
    if (auth_header && auth_header[0] != '\0') {
        slist = curl_slist_append(slist, auth_header);
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist);

    CURLcode rc = curl_easy_perform(curl);
    qllm_http_response_t* resp = finalize_response(curl, &buf, rc);
    curl_slist_free_all(slist);
    curl_easy_cleanup(curl);
    return resp;
}

int32_t qllm_http_response_status(qllm_http_response_t* resp) {
    return resp ? resp->status : 0;
}

const char* qllm_http_response_body(qllm_http_response_t* resp) {
    return resp ? resp->body : NULL;
}

int64_t qllm_http_response_body_len(qllm_http_response_t* resp) {
    return resp ? resp->body_len : 0;
}

void qllm_http_response_free(qllm_http_response_t* resp) {
    if (!resp) return;
    free(resp->body);
    free(resp->error);
    free(resp);
}

const char* qllm_http_error_string(int32_t code) {
    /* Map curl error codes to human strings. Negative or zero → no error. */
    if (code <= 0) return "no error";
    return curl_easy_strerror((CURLcode)code);
}

/* qllm_http_request takes a packed request struct in the qLLM ABI;
 * Eshkol's http.esk doesn't currently call this directly (it uses
 * the get/post/post_json convenience entries). Provide a NULL-returning
 * stub so the symbol resolves at link time without crashing. */
void* qllm_http_request(void* req) {
    (void)req;
    return NULL;
}

/* SSE streaming entry points: deliberately return NULL/error so the
 * Eshkol fallback path runs. Real implementation needs CURL multi
 * interface or chunked-transfer line parsing; deferred to a follow-up. */
void* qllm_http_stream_open(const char* url, const char** headers, int64_t hdr_count,
                            const char* body, int64_t body_len, int32_t timeout_ms,
                            void* callback) {
    (void)url; (void)headers; (void)hdr_count;
    (void)body; (void)body_len; (void)timeout_ms; (void)callback;
    return NULL;
}

int32_t qllm_http_stream_next(void* stream, void* event_out, int32_t timeout_ms) {
    (void)stream; (void)event_out; (void)timeout_ms;
    return -1;
}

int32_t qllm_http_stream_done(void* stream) {
    (void)stream;
    return 1;
}

void qllm_http_stream_close(void* stream) {
    (void)stream;
}

void qllm_sse_event_free(void* event) {
    (void)event;
}

#endif /* ESHKOL_HAVE_LIBCURL */
