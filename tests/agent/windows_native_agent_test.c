#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct eshkol_subprocess eshkol_subprocess_t;
typedef struct qllm_http_response qllm_http_response_t;

extern const char* eshkol_os_type(void);
extern int64_t eshkol_monotonic_time_ms(void);
extern void eshkol_sleep_ms(int64_t ms);
extern int32_t eshkol_mkdtemp_path(const char*, const char*, char*, int32_t);
extern int32_t eshkol_mkstemp_path(const char*, const char*, const char*, char*, int32_t);
extern int32_t eshkol_mkdir_recursive(const char*, int32_t);
extern int32_t eshkol_rmdir_recursive(const char*);
extern int32_t eshkol_file_copy(const char*, const char*);
extern int64_t eshkol_file_mmap(const char*, int64_t, int64_t);
extern int32_t eshkol_mmap_read(int64_t, int64_t, char*, int32_t);
extern int64_t eshkol_mmap_length(int64_t);
extern int32_t eshkol_file_munmap(int64_t);
extern int32_t eshkol_sha256_file(const char*, char*, int32_t);
extern int32_t eshkol_glob_match(const char*, const char*);

extern int64_t eshkol_regex_compile(const char*, int);
extern int eshkol_regex_match(int64_t, const char*, char*, size_t);
extern void eshkol_regex_free(int64_t);
extern int64_t eshkol_sqlite_open(const char*);
extern int eshkol_sqlite_exec(int64_t, const char*);
extern void eshkol_sqlite_close(int64_t);

extern eshkol_subprocess_t* qllm_process_spawn_argv(const char*, const char*);
extern int32_t qllm_process_wait(eshkol_subprocess_t*, int32_t);
extern int32_t qllm_process_exit_code(eshkol_subprocess_t*);
extern char* qllm_process_read_all_stdout(eshkol_subprocess_t*, int64_t, int64_t*);
extern void qllm_process_free_buffer(char*);
extern void qllm_process_destroy(eshkol_subprocess_t*);

extern int32_t eshkol_make_pipe(int32_t*, int32_t*);
extern int32_t eshkol_poll_read(int32_t, int32_t);
extern int32_t eshkol_fd_write_available(int32_t, const char*, int32_t);
extern int32_t eshkol_fd_read_available(int32_t, char*, int32_t);

extern int64_t eshkol_watch_start(const char*, int32_t);
extern int32_t eshkol_watch_poll(int64_t, char*, int32_t);
extern void eshkol_watch_stop(int64_t);

extern int64_t eshkol_http_server_create(int32_t);
extern int32_t eshkol_http_server_port(int64_t);
extern int32_t eshkol_http_server_accept(int64_t, char*, int32_t, int32_t);
extern void eshkol_http_server_respond(int64_t, int32_t, const char*, const char*);
extern void eshkol_http_server_close(int64_t);
extern int32_t qllm_http_init(void);
extern void qllm_http_shutdown(void);
extern int32_t qllm_http_has_ssl(void);
extern qllm_http_response_t* qllm_http_get(const char*, int32_t);
extern qllm_http_response_t* qllm_http_post(const char*, const char**, int64_t,
                                           const char*, int64_t, int32_t);
extern int32_t qllm_http_response_status(qllm_http_response_t*);
extern const char* qllm_http_response_body(qllm_http_response_t*);
extern void qllm_http_response_free(qllm_http_response_t*);

static int failures = 0;

static void check(int condition, const char* name) {
    if (condition) {
        printf("PASS: %s\n", name);
    } else {
        fprintf(stderr, "FAIL: %s\n", name);
        failures++;
    }
}

static int write_pattern_file(const char* path, size_t size) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    for (size_t i = 0; i < size; ++i) {
        unsigned char byte = (unsigned char)((i * 131u + 17u) & 0xffu);
        if (fwrite(&byte, 1, 1, f) != 1) { fclose(f); return 0; }
    }
    return fclose(f) == 0;
}

typedef struct server_thread_ctx {
    int64_t server;
    const char* expected;
    const char* response;
    int accepted;
} server_thread_ctx_t;

static DWORD WINAPI server_thread(LPVOID opaque) {
    server_thread_ctx_t* ctx = (server_thread_ctx_t*)opaque;
    char request[16384];
    int32_t n = eshkol_http_server_accept(ctx->server, request, sizeof(request), 10000);
    ctx->accepted = n > 0 && strstr(request, ctx->expected) != NULL;
    if (n > 0) eshkol_http_server_respond(ctx->server, 200, "text/plain", ctx->response);
    return 0;
}

static void test_platform_and_dependencies(char* root, size_t root_cap) {
    check(strcmp(eshkol_os_type(), "windows") == 0, "platform reports windows");
    int64_t before = eshkol_monotonic_time_ms();
    eshkol_sleep_ms(5);
    check(eshkol_monotonic_time_ms() >= before + 1, "monotonic clock advances");
    check(eshkol_mkdtemp_path("eshkol-native-", NULL, root, (int32_t)root_cap) > 0,
          "race-free temporary directory");

    char nested[4096];
    snprintf(nested, sizeof(nested), "%s/sub/child", root);
    check(eshkol_mkdir_recursive(nested, 0700) == 0, "recursive mkdir");

    char source[4096];
    snprintf(source, sizeof(source), "%s/pattern.bin", nested);
    check(write_pattern_file(source, 131072), "write mmap fixture");
    int64_t mapping = eshkol_file_mmap(source, 123, 8192);
    check(mapping >= 0 && eshkol_mmap_length(mapping) == 8192,
          "unaligned Windows file mapping");
    if (mapping >= 0) {
        unsigned char bytes[1024];
        int32_t got = eshkol_mmap_read(mapping, 0, (char*)bytes, sizeof(bytes));
        int valid = got == (int32_t)sizeof(bytes);
        for (int32_t i = 0; valid && i < got; ++i) {
            valid = bytes[i] == (unsigned char)((((size_t)i + 123) * 131u + 17u) & 0xffu);
        }
        check(valid, "mapped bytes preserve unaligned offset");
        check(eshkol_file_munmap(mapping) == 0, "unmap releases mapping");
    }

    char copy[4096];
    snprintf(copy, sizeof(copy), "%s/copy.bin", root);
    check(eshkol_file_copy(source, copy) == 0, "native file copy");
    char digest[65];
    check(eshkol_sha256_file(copy, digest, sizeof(digest)) == 0 && strlen(digest) == 64,
          "BCrypt SHA-256 file digest");
    check(eshkol_glob_match("*.BIN", "copy.bin") == 1,
          "Windows glob matching is case-insensitive");

    int64_t regex = eshkol_regex_compile("^native-[0-9]+$", 0);
    char match[64];
    check(regex >= 0 && eshkol_regex_match(regex, "native-133", match, sizeof(match)) > 0 &&
          strcmp(match, "native-133") == 0,
          "bundled PCRE2 regex");
    if (regex >= 0) eshkol_regex_free(regex);

    int64_t db = eshkol_sqlite_open(":memory:");
    check(db > 0 && eshkol_sqlite_exec(db,
          "CREATE TABLE t(x INTEGER); INSERT INTO t VALUES(133)") == 0,
          "bundled SQLite database");
    if (db > 0) eshkol_sqlite_close(db);
}

static void test_pipe_and_subprocess(void) {
    int32_t read_fd = -1, write_fd = -1;
    check(eshkol_make_pipe(&read_fd, &write_fd) == 0, "native CRT pipe");
    if (read_fd >= 0 && write_fd >= 0) {
        check(eshkol_fd_write_available(write_fd, "pipe-ok", 7) == 7,
              "pipe write");
        check(eshkol_poll_read(read_fd, 1000) == 1, "PeekNamedPipe readiness");
        char data[16] = {0};
        check(eshkol_fd_read_available(read_fd, data, sizeof(data)) == 7 &&
              memcmp(data, "pipe-ok", 7) == 0, "pipe read");
        _close(read_fd);
        _close(write_fd);
    }

    eshkol_subprocess_t* process = qllm_process_spawn_argv(
        "powershell.exe\t-NoLogo\t-NoProfile\t-NonInteractive\t-Command\t[Console]::Out.Write(('x' * 1048576))",
        NULL);
    check(process != NULL, "direct argv subprocess spawn");
    if (process) {
        check(qllm_process_wait(process, 30000) == 0 &&
              qllm_process_exit_code(process) == 0,
              "chatty subprocess exits without pipe deadlock");
        int64_t length = 0;
        char* output = qllm_process_read_all_stdout(process, 2 * 1024 * 1024, &length);
        check(output != NULL && length == 1048576 && output[0] == 'x' &&
              output[length - 1] == 'x', "subprocess output drained exactly");
        qllm_process_free_buffer(output);
        qllm_process_destroy(process);
    }
}

static void test_watch(const char* root) {
    int64_t watcher = eshkol_watch_start(root, 1);
    check(watcher > 0, "recursive ReadDirectoryChangesW watcher");
    if (watcher <= 0) return;
    char path[4096];
    snprintf(path, sizeof(path), "%s/watched.txt", root);
    FILE* f = fopen(path, "wb");
    if (f) { fputs("watch", f); fclose(f); }
    char event[8192] = {0};
    int found = 0;
    for (int i = 0; i < 200 && !found; ++i) {
        int32_t n = eshkol_watch_poll(watcher, event, sizeof(event));
        if (n > 0 && strstr(event, "watched.txt")) found = 1;
        if (!found) eshkol_sleep_ms(10);
    }
    check(found, "watcher reports created file");
    eshkol_watch_stop(watcher);
}

static void test_http(void) {
    check(qllm_http_init() == 1 && qllm_http_has_ssl() == 1,
          "WinHTTP initializes with TLS");
    int64_t server = eshkol_http_server_create(0);
    int32_t port = eshkol_http_server_port(server);
    check(server > 0 && port > 0, "Winsock loopback HTTP server");
    if (server > 0 && port > 0) {
        server_thread_ctx_t ctx = {server, "GET /native", "winhttp-get", 0};
        HANDLE thread = CreateThread(NULL, 0, server_thread, &ctx, 0, NULL);
        char url[256];
        snprintf(url, sizeof(url), "http://127.0.0.1:%d/native", port);
        qllm_http_response_t* response = qllm_http_get(url, 10000);
        check(response && qllm_http_response_status(response) == 200 &&
              strcmp(qllm_http_response_body(response), "winhttp-get") == 0,
              "native WinHTTP GET round-trip");
        if (response) qllm_http_response_free(response);
        WaitForSingleObject(thread, 10000);
        CloseHandle(thread);
        check(ctx.accepted, "HTTP server accepted complete GET");
        eshkol_http_server_close(server);
    }

    server = eshkol_http_server_create(0);
    port = eshkol_http_server_port(server);
    if (server > 0 && port > 0) {
        server_thread_ctx_t ctx = {server, "native-body-133", "winhttp-post", 0};
        HANDLE thread = CreateThread(NULL, 0, server_thread, &ctx, 0, NULL);
        char url[256];
        snprintf(url, sizeof(url), "http://127.0.0.1:%d/post", port);
        const char* headers[] = {"Content-Type: text/plain", "X-Eshkol-Test: native"};
        const char body[] = "native-body-133";
        qllm_http_response_t* response = qllm_http_post(
            url, headers, 2, body, (int64_t)strlen(body), 10000);
        check(response && qllm_http_response_status(response) == 200 &&
              strcmp(qllm_http_response_body(response), "winhttp-post") == 0,
              "native WinHTTP POST round-trip");
        if (response) qllm_http_response_free(response);
        WaitForSingleObject(thread, 10000);
        CloseHandle(thread);
        check(ctx.accepted, "HTTP server reads complete POST body");
        eshkol_http_server_close(server);
    }
    qllm_http_shutdown();
}

int main(void) {
    char root[4096] = {0};
    test_platform_and_dependencies(root, sizeof(root));
    test_pipe_and_subprocess();
    if (root[0]) test_watch(root);
    test_http();
    if (root[0]) check(eshkol_rmdir_recursive(root) == 0, "safe recursive cleanup");
    printf("Windows native agent checks: %s (%d failure%s)\n",
           failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}

#else
int main(void) { return 0; }
#endif
