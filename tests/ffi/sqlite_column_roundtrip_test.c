#include <sqlite3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int64_t eshkol_sqlite_open(const char* path);
void eshkol_sqlite_close(int64_t handle);
int eshkol_sqlite_exec(int64_t handle, const char* sql);
int64_t eshkol_sqlite_prepare(int64_t handle, const char* sql);
int eshkol_sqlite_step(int64_t handle);
int eshkol_sqlite_reset(int64_t handle);
void eshkol_sqlite_finalize(int64_t handle);
int eshkol_sqlite_bind_text(int64_t handle, int index, const char* text);
int eshkol_sqlite_bind_int(int64_t handle, int index, int64_t value);
int eshkol_sqlite_bind_null(int64_t handle, int index);
int64_t eshkol_sqlite_column_bytes(int64_t handle, int index);
int eshkol_sqlite_column_text(int64_t handle, int index, char* buf, size_t size);
int eshkol_sqlite_column_type(int64_t handle, int index);

typedef struct {
    uint8_t subtype;
    uint8_t flags;
    uint16_t ref_count;
    uint32_t size;
} EshkolStringHeader;

/* Match the runtime's header-based length query for this standalone ABI test.
 * The CMake target uses the real runtime helper; the direct harness build
 * supplies this small equivalent so it can run without the full agent archive. */
#ifndef ESHKOL_SQLITE_TEST_USE_RUNTIME
int64_t eshkol_string_byte_length(const char* value) {
    if (!value) return 0;
    const EshkolStringHeader* header =
        (const EshkolStringHeader*)((const unsigned char*)value - sizeof(*header));
    if (header->subtype == 1 && header->size > 0) return (int64_t)header->size - 1;
    return (int64_t)strlen(value);
}
#else
extern int64_t eshkol_string_byte_length(const char* value);
#endif

static char* make_string(const char* bytes, size_t length) {
    EshkolStringHeader* header =
        (EshkolStringHeader*)malloc(sizeof(*header) + length + 1);
    if (!header) return NULL;
    header->subtype = 1;
    header->flags = 0;
    header->ref_count = 0;
    header->size = (uint32_t)(length + 1);
    char* value = (char*)(header + 1);
    if (length && bytes) memcpy(value, bytes, length);
    value[length] = '\0';
    return value;
}

static int expect(int condition, const char* label) {
    if (condition) return 1;
    fprintf(stderr, "FAIL: %s\n", label);
    return 0;
}

int main(void) {
    int ok = 1;
    const int SQLITE_ROW_CODE = 100;
    const int SQLITE_DONE_CODE = 101;
    const int SQLITE_TEXT_CODE = 3;
    const int SQLITE_NULL_CODE = 5;
    int64_t db = eshkol_sqlite_open(":memory:");
    ok &= expect(db > 0, "open");
    ok &= expect(eshkol_sqlite_exec(db,
        "CREATE TABLE roundtrip_values(id INTEGER PRIMARY KEY, value TEXT)") == SQLITE_OK,
        "create");

    char* large = make_string(NULL, 20000);
    if (!large) return 2;
    memset(large, 'x', 20000);
    char embedded_bytes[] = {'l','e','f','t','\0','r','i','g','h','t'};
    char* embedded = make_string(embedded_bytes, sizeof(embedded_bytes));
    char* empty = make_string("", 0);
    if (!embedded || !empty) return 2;

    int64_t insert = eshkol_sqlite_prepare(db,
        "INSERT INTO roundtrip_values(id, value) VALUES(?, ?)");
    ok &= expect(insert > 0, "prepare insert");
    const char* values[] = {large, empty, NULL, embedded};
    for (int i = 0; i < 4; ++i) {
        ok &= expect(eshkol_sqlite_reset(insert) == SQLITE_OK, "reset insert");
        ok &= expect(eshkol_sqlite_bind_int(insert, 1, i + 1) == SQLITE_OK,
                     "bind id");
        int bind_rc = values[i] ? eshkol_sqlite_bind_text(insert, 2, values[i])
                                : eshkol_sqlite_bind_null(insert, 2);
        ok &= expect(bind_rc == SQLITE_OK, "bind value");
        ok &= expect(eshkol_sqlite_step(insert) == SQLITE_DONE_CODE, "insert step");
    }
    eshkol_sqlite_finalize(insert);

    int64_t query = eshkol_sqlite_prepare(db,
        "SELECT value FROM roundtrip_values ORDER BY id");
    ok &= expect(query > 0, "prepare query");
    char output[20001];
    for (int i = 0; i < 4; ++i) {
        ok &= expect(eshkol_sqlite_step(query) == SQLITE_ROW_CODE, "query row");
        int type = eshkol_sqlite_column_type(query, 0);
        int64_t bytes = eshkol_sqlite_column_bytes(query, 0);
        int copied = eshkol_sqlite_column_text(query, 0, output, sizeof(output));
        if (i == 0) {
            ok &= expect(type == SQLITE_TEXT_CODE && bytes == 20000 && copied == 20000,
                         "large TEXT round-trip");
            ok &= expect(memcmp(output, large, 20000) == 0, "large TEXT payload");
        } else if (i == 1) {
            ok &= expect(type == SQLITE_TEXT_CODE && bytes == 0 && copied == 0,
                         "empty TEXT round-trip");
        } else if (i == 2) {
            ok &= expect(type == SQLITE_NULL_CODE && bytes == 0 && copied == 0,
                         "SQL NULL discriminator");
        } else {
            ok &= expect(type == SQLITE_TEXT_CODE && bytes == 10 && copied == 10,
                         "embedded NUL round-trip");
            ok &= expect(memcmp(output, embedded_bytes, sizeof(embedded_bytes)) == 0,
                         "embedded NUL payload");
        }
    }
    ok &= expect(eshkol_sqlite_step(query) == SQLITE_DONE_CODE, "query done");
    eshkol_sqlite_finalize(query);
    eshkol_sqlite_close(db);
    free(((EshkolStringHeader*)large) - 1);
    free(((EshkolStringHeader*)embedded) - 1);
    free(((EshkolStringHeader*)empty) - 1);
    puts(ok ? "sqlite-column-roundtrip: PASS" : "sqlite-column-roundtrip: FAIL");
    return ok ? 0 : 1;
}
