/*******************************************************************************
 * SQLite3 Bindings for Eshkol
 *
 * Provides database operations via opaque int64_t handles.
 * WAL mode enabled by default for concurrent read performance.
 *
 * Copyright (c) 2025 Eshkol Project
 ******************************************************************************/

#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*******************************************************************************
 * Handle Tables
 ******************************************************************************/

#define MAX_DB_HANDLES   64
#define MAX_STMT_HANDLES 512

static sqlite3*      g_db_handles[MAX_DB_HANDLES]     = {0};
static sqlite3_stmt* g_stmt_handles[MAX_STMT_HANDLES]  = {0};
static int g_next_db   = 1;
static int g_next_stmt = 1;

/**
 * @brief Allocates a slot in the global DB handle table for an open @p db.
 *
 * Scans forward from the last-used index (wrapping around once) so released
 * handles are reused rather than exhausting the table.
 *
 * @param db Open sqlite3 connection to store.
 * @return The new handle (>= 1), or -1 if the table is full.
 */
static int alloc_db(sqlite3* db) {
    for (int i = g_next_db; i < MAX_DB_HANDLES; i++) {
        if (!g_db_handles[i]) { g_db_handles[i] = db; g_next_db = i+1; return i; }
    }
    for (int i = 1; i < g_next_db; i++) {
        if (!g_db_handles[i]) { g_db_handles[i] = db; g_next_db = i+1; return i; }
    }
    return -1;
}

/**
 * @brief Allocates a slot in the global statement handle table for a prepared @p stmt.
 *
 * Scans forward from the last-used index (wrapping around once) so released
 * handles are reused rather than exhausting the table.
 *
 * @param stmt Prepared statement to store.
 * @return The new handle (>= 1), or -1 if the table is full.
 */
static int alloc_stmt(sqlite3_stmt* stmt) {
    for (int i = g_next_stmt; i < MAX_STMT_HANDLES; i++) {
        if (!g_stmt_handles[i]) { g_stmt_handles[i] = stmt; g_next_stmt = i+1; return i; }
    }
    for (int i = 1; i < g_next_stmt; i++) {
        if (!g_stmt_handles[i]) { g_stmt_handles[i] = stmt; g_next_stmt = i+1; return i; }
    }
    return -1;
}

/**
 * @brief Looks up the open sqlite3 connection for a given handle.
 *
 * @param h Handle previously returned by eshkol_sqlite_open().
 * @return The stored sqlite3 pointer, or NULL if @p h is out of range or unused.
 */
static sqlite3* get_db(int64_t h) {
    if (h < 1 || h >= MAX_DB_HANDLES) return NULL;
    return g_db_handles[h];
}

/**
 * @brief Looks up the prepared statement for a given handle.
 *
 * @param h Handle previously returned by eshkol_sqlite_prepare().
 * @return The stored sqlite3_stmt pointer, or NULL if @p h is out of range or unused.
 */
static sqlite3_stmt* get_stmt(int64_t h) {
    if (h < 1 || h >= MAX_STMT_HANDLES) return NULL;
    return g_stmt_handles[h];
}

/*******************************************************************************
 * Database Operations
 ******************************************************************************/

/**
 * @brief Opens (or creates) a SQLite database file and returns an opaque handle.
 *
 * Opens with SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
 * then enables WAL journal mode for concurrent readers and sets a 5-second
 * busy timeout before storing the connection in the handle table.
 *
 * @param path Filesystem path to the database file.
 * @return Handle (>= 1) on success, -1 on a NULL path, open failure, or full
 *         handle table.
 */
int64_t eshkol_sqlite_open(const char* path) {
    if (!path) return -1;

    sqlite3* db = NULL;
    int rc = sqlite3_open_v2(path, &db,
                              SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE |
                              SQLITE_OPEN_FULLMUTEX, NULL);
    if (rc != SQLITE_OK) {
        if (db) sqlite3_close(db);
        return -1;
    }

    /* Enable WAL mode for concurrent reads */
    sqlite3_exec(db, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
    /* Reasonable busy timeout */
    sqlite3_busy_timeout(db, 5000);

    int handle = alloc_db(db);
    if (handle < 0) {
        sqlite3_close(db);
        return -1;
    }
    return handle;
}

/**
 * @brief Closes a database connection and releases its handle slot.
 *
 * @param handle Handle previously returned by eshkol_sqlite_open(). No-op if
 *        the handle is invalid or already closed.
 */
void eshkol_sqlite_close(int64_t handle) {
    sqlite3* db = get_db(handle);
    if (db) {
        sqlite3_close(db);
        g_db_handles[handle] = NULL;
    }
}

/**
 * @brief Executes one or more SQL statements with no result-row callback.
 *
 * Intended for DDL/DML (CREATE, INSERT, UPDATE, ...) where no rows are
 * expected back. Any error message produced by SQLite is freed internally.
 *
 * @param handle Database handle from eshkol_sqlite_open().
 * @param sql SQL text to execute.
 * @return 0 on success, the SQLite error code on failure, or -1 on invalid
 *         handle/sql.
 */
int eshkol_sqlite_exec(int64_t handle, const char* sql) {
    sqlite3* db = get_db(handle);
    if (!db || !sql) return -1;

    char* errmsg = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &errmsg);
    if (errmsg) sqlite3_free(errmsg);
    return rc == SQLITE_OK ? 0 : rc;
}

/*******************************************************************************
 * Prepared Statements
 ******************************************************************************/

/**
 * @brief Compiles a SQL statement into a reusable prepared-statement handle.
 *
 * @param db_handle Database handle from eshkol_sqlite_open().
 * @param sql SQL text to prepare.
 * @return Statement handle (>= 1) on success, -1 on invalid database handle,
 *         NULL sql, a prepare error, or a full statement handle table.
 */
int64_t eshkol_sqlite_prepare(int64_t db_handle, const char* sql) {
    sqlite3* db = get_db(db_handle);
    if (!db || !sql) return -1;

    sqlite3_stmt* stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK || !stmt) return -1;

    int handle = alloc_stmt(stmt);
    if (handle < 0) {
        sqlite3_finalize(stmt);
        return -1;
    }
    return handle;
}

/**
 * @brief Advances a prepared statement to its next result.
 *
 * @param stmt_handle Statement handle from eshkol_sqlite_prepare().
 * @return The raw sqlite3_step() result code (e.g. SQLITE_ROW = 100,
 *         SQLITE_DONE = 101), or -1 if @p stmt_handle is invalid.
 */
int eshkol_sqlite_step(int64_t stmt_handle) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    return sqlite3_step(stmt);  /* Returns SQLITE_ROW (100) or SQLITE_DONE (101) */
}

/**
 * @brief Resets a prepared statement so it can be re-executed (bindings are kept).
 *
 * @param stmt_handle Statement handle from eshkol_sqlite_prepare().
 * @return The sqlite3_reset() result code, or -1 if @p stmt_handle is invalid.
 */
int eshkol_sqlite_reset(int64_t stmt_handle) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    return sqlite3_reset(stmt);
}

/**
 * @brief Finalizes a prepared statement and releases its handle slot.
 *
 * @param stmt_handle Statement handle from eshkol_sqlite_prepare(). No-op if
 *        the handle is invalid or already finalized.
 */
void eshkol_sqlite_finalize(int64_t stmt_handle) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (stmt) {
        sqlite3_finalize(stmt);
        g_stmt_handles[stmt_handle] = NULL;
    }
}

/*******************************************************************************
 * Parameter Binding
 ******************************************************************************/

/**
 * @brief Binds a text value to a 1-based parameter index of a prepared statement.
 *
 * Uses SQLITE_TRANSIENT, so SQLite copies @p text internally and the caller's
 * buffer may be freed or reused immediately after this call.
 *
 * @param stmt_handle Statement handle from eshkol_sqlite_prepare().
 * @param index 1-based bind parameter index.
 * @param text NUL-terminated text to bind.
 * @return The sqlite3_bind_text() result code, or -1 on invalid statement/text.
 */
int eshkol_sqlite_bind_text(int64_t stmt_handle, int index, const char* text) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt || !text) return -1;
    return sqlite3_bind_text(stmt, index, text, -1, SQLITE_TRANSIENT);
}

/**
 * @brief Binds a 64-bit integer value to a 1-based parameter index.
 *
 * @param stmt_handle Statement handle from eshkol_sqlite_prepare().
 * @param index 1-based bind parameter index.
 * @param value Integer value to bind.
 * @return The sqlite3_bind_int64() result code, or -1 on invalid statement.
 */
int eshkol_sqlite_bind_int(int64_t stmt_handle, int index, int64_t value) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    return sqlite3_bind_int64(stmt, index, value);
}

/**
 * @brief Binds a floating-point value to a 1-based parameter index.
 *
 * @param stmt_handle Statement handle from eshkol_sqlite_prepare().
 * @param index 1-based bind parameter index.
 * @param value Value to bind.
 * @return The sqlite3_bind_double() result code, or -1 on invalid statement.
 */
int eshkol_sqlite_bind_double(int64_t stmt_handle, int index, double value) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    return sqlite3_bind_double(stmt, index, value);
}

/**
 * @brief Binds SQL NULL to a 1-based parameter index.
 *
 * @param stmt_handle Statement handle from eshkol_sqlite_prepare().
 * @param index 1-based bind parameter index.
 * @return The sqlite3_bind_null() result code, or -1 on invalid statement.
 */
int eshkol_sqlite_bind_null(int64_t stmt_handle, int index) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    return sqlite3_bind_null(stmt, index);
}

/*******************************************************************************
 * Column Access
 ******************************************************************************/

/**
 * @brief Returns the exact byte length of a TEXT/BLOB column without copying it.
 *
 * Lets the Eshkol wrapper size its receiving buffer correctly before calling
 * eshkol_sqlite_column_text(), avoiding silent truncation on large payloads.
 * Calls sqlite3_column_text() first per SQLite's docs, since column_bytes()
 * may require the value to have been converted to TEXT.
 *
 * @param stmt_handle Statement handle positioned on a row (after a
 *        SQLITE_ROW step).
 * @param index 0-based column index.
 * @return Byte length of the column value, or -1 if @p stmt_handle is invalid.
 */
/* Get the byte length of a TEXT/BLOB column without copying the data.
 * Required so the Eshkol wrapper can size its receiving buffer
 * correctly for large payloads (session JSON, embedded blobs, etc.)
 * — the previous fixed-8 KB scheme silently truncated and callers
 * threw on the partial JSON. Mirrors sqlite3_column_bytes which is
 * exact (binary-safe) unlike strlen on the text pointer. */
int64_t eshkol_sqlite_column_bytes(int64_t stmt_handle, int index) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    /* Note: SQLite docs require column_text() to be called before
     * column_bytes() if the underlying value isn't already TEXT,
     * because SQLite may need to convert the type. Calling text first
     * is harmless when the column is already TEXT. */
    sqlite3_column_text(stmt, index);
    return (int64_t)sqlite3_column_bytes(stmt, index);
}

/**
 * @brief Copies a TEXT column's value into a caller-provided buffer.
 *
 * Uses sqlite3_column_bytes() (binary-safe) rather than strlen() to determine
 * the source length, since the previous fixed-size scheme silently truncated
 * large payloads. If @p buf is too small, the copy is truncated but the
 * return value is encoded as -(len+1) so the caller can detect truncation
 * and retry with a larger buffer.
 *
 * @param stmt_handle Statement handle positioned on a row.
 * @param index 0-based column index.
 * @param buf Output buffer receiving the NUL-terminated text.
 * @param buf_size Size of @p buf in bytes.
 * @return Number of bytes copied (excluding the NUL) on success, 0 if the
 *         column is NULL, -(len+1) if @p buf was too small to hold the full
 *         value, or -1 on invalid arguments.
 */
int eshkol_sqlite_column_text(int64_t stmt_handle, int index,
                                char* buf, size_t buf_size) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt || !buf || buf_size == 0) return -1;

    const unsigned char* text = sqlite3_column_text(stmt, index);
    if (!text) {
        buf[0] = '\0';
        return 0;
    }
    /* Use sqlite3_column_bytes (binary-safe, includes embedded NULs)
     * not strlen (stops at first NUL). The agent stores JSON sessions
     * which never contain NUL but may contain other bytes that strlen
     * would handle correctly — still, column_bytes is the canonical
     * length and matches the buffer-sizing path below. */
    size_t len = (size_t)sqlite3_column_bytes(stmt, index);
    size_t copy = len < buf_size - 1 ? len : buf_size - 1;
    memcpy(buf, text, copy);
    buf[copy] = '\0';
    /* Signal truncation: caller passed buf_size = len, but content was
     * larger. Returning the truncated count alone hides the overflow.
     * Encoding: if we couldn't fit the full payload, return -(len+1)
     * so the Eshkol wrapper can detect and retry with a bigger buffer
     * (or surface a clear error rather than silent truncation). The
     * Eshkol wrapper calls column_bytes first nowadays so this branch
     * shouldn't fire in normal use, but it's defensive against
     * callers that haven't migrated to the new sizing path. */
    if (copy < len) {
        return -(int)(len + 1);
    }
    return (int)copy;
}

/**
 * @brief Reads the current row's column value as a 64-bit integer.
 *
 * @param stmt_handle Statement handle positioned on a row.
 * @param index 0-based column index.
 * @return The column value converted to int64_t, or 0 if @p stmt_handle is invalid.
 */
int64_t eshkol_sqlite_column_int(int64_t stmt_handle, int index) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return 0;
    return sqlite3_column_int64(stmt, index);
}

/**
 * @brief Reads the current row's column value as a double.
 *
 * @param stmt_handle Statement handle positioned on a row.
 * @param index 0-based column index.
 * @return The column value converted to double, or 0.0 if @p stmt_handle is invalid.
 */
double eshkol_sqlite_column_double(int64_t stmt_handle, int index) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return 0.0;
    return sqlite3_column_double(stmt, index);
}

/**
 * @brief Returns the number of columns in a prepared statement's result set.
 *
 * @param stmt_handle Statement handle from eshkol_sqlite_prepare().
 * @return Column count, or 0 if @p stmt_handle is invalid.
 */
int eshkol_sqlite_column_count(int64_t stmt_handle) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return 0;
    return sqlite3_column_count(stmt);
}

/**
 * @brief Copies a result column's declared name into a caller-provided buffer.
 *
 * @param stmt_handle Statement handle from eshkol_sqlite_prepare().
 * @param index 0-based column index.
 * @param buf Output buffer receiving the NUL-terminated column name.
 * @param buf_size Size of @p buf in bytes.
 * @return Number of bytes copied (excluding the NUL), 0 if the name is
 *         unavailable, or -1 on invalid arguments.
 */
int eshkol_sqlite_column_name(int64_t stmt_handle, int index,
                                char* buf, size_t buf_size) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt || !buf || buf_size == 0) return -1;

    const char* name = sqlite3_column_name(stmt, index);
    if (!name) { buf[0] = '\0'; return 0; }
    size_t len = strlen(name);
    size_t copy = len < buf_size - 1 ? len : buf_size - 1;
    memcpy(buf, name, copy);
    buf[copy] = '\0';
    return (int)copy;
}

/**
 * @brief Returns the SQLite storage class (type) of the current row's column value.
 *
 * @param stmt_handle Statement handle positioned on a row.
 * @param index 0-based column index.
 * @return One of the SQLITE_{INTEGER,FLOAT,TEXT,BLOB,NULL} constants, or -1
 *         if @p stmt_handle is invalid.
 */
int eshkol_sqlite_column_type(int64_t stmt_handle, int index) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    return sqlite3_column_type(stmt, index);
}

/*******************************************************************************
 * Error Handling
 ******************************************************************************/

/**
 * @brief Copies the database connection's last error message into a buffer.
 *
 * @param db_handle Database handle from eshkol_sqlite_open().
 * @param buf Output buffer receiving the NUL-terminated error message.
 * @param buf_size Size of @p buf in bytes.
 * @return Number of bytes copied (excluding the NUL), or -1 on invalid arguments.
 */
int eshkol_sqlite_last_error(int64_t db_handle, char* buf, size_t buf_size) {
    sqlite3* db = get_db(db_handle);
    if (!db || !buf || buf_size == 0) return -1;

    const char* msg = sqlite3_errmsg(db);
    size_t len = strlen(msg);
    size_t copy = len < buf_size - 1 ? len : buf_size - 1;
    memcpy(buf, msg, copy);
    buf[copy] = '\0';
    return (int)copy;
}

/**
 * @brief Returns the rowid of the most recent successful INSERT on this connection.
 *
 * @param db_handle Database handle from eshkol_sqlite_open().
 * @return The last inserted rowid, or -1 if @p db_handle is invalid.
 */
int64_t eshkol_sqlite_last_insert_rowid(int64_t db_handle) {
    sqlite3* db = get_db(db_handle);
    if (!db) return -1;
    return sqlite3_last_insert_rowid(db);
}

/**
 * @brief Returns the number of rows changed by the most recent INSERT/UPDATE/DELETE.
 *
 * @param db_handle Database handle from eshkol_sqlite_open().
 * @return The row-change count (which may legitimately be zero), or -1 if
 *         @p db_handle is invalid.
 */
int eshkol_sqlite_changes(int64_t db_handle) {
    sqlite3* db = get_db(db_handle);
    if (!db) return -1;
    return sqlite3_changes(db);
}
