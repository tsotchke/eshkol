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

static int alloc_db(sqlite3* db) {
    for (int i = g_next_db; i < MAX_DB_HANDLES; i++) {
        if (!g_db_handles[i]) { g_db_handles[i] = db; g_next_db = i+1; return i; }
    }
    for (int i = 1; i < g_next_db; i++) {
        if (!g_db_handles[i]) { g_db_handles[i] = db; g_next_db = i+1; return i; }
    }
    return -1;
}

static int alloc_stmt(sqlite3_stmt* stmt) {
    for (int i = g_next_stmt; i < MAX_STMT_HANDLES; i++) {
        if (!g_stmt_handles[i]) { g_stmt_handles[i] = stmt; g_next_stmt = i+1; return i; }
    }
    for (int i = 1; i < g_next_stmt; i++) {
        if (!g_stmt_handles[i]) { g_stmt_handles[i] = stmt; g_next_stmt = i+1; return i; }
    }
    return -1;
}

static sqlite3* get_db(int64_t h) {
    if (h < 1 || h >= MAX_DB_HANDLES) return NULL;
    return g_db_handles[h];
}

static sqlite3_stmt* get_stmt(int64_t h) {
    if (h < 1 || h >= MAX_STMT_HANDLES) return NULL;
    return g_stmt_handles[h];
}

/*******************************************************************************
 * Database Operations
 ******************************************************************************/

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

void eshkol_sqlite_close(int64_t handle) {
    sqlite3* db = get_db(handle);
    if (db) {
        sqlite3_close(db);
        g_db_handles[handle] = NULL;
    }
}

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

int eshkol_sqlite_step(int64_t stmt_handle) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    return sqlite3_step(stmt);  /* Returns SQLITE_ROW (100) or SQLITE_DONE (101) */
}

int eshkol_sqlite_reset(int64_t stmt_handle) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    return sqlite3_reset(stmt);
}

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

int eshkol_sqlite_bind_text(int64_t stmt_handle, int index, const char* text) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt || !text) return -1;
    return sqlite3_bind_text(stmt, index, text, -1, SQLITE_TRANSIENT);
}

int eshkol_sqlite_bind_int(int64_t stmt_handle, int index, int64_t value) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    return sqlite3_bind_int64(stmt, index, value);
}

int eshkol_sqlite_bind_double(int64_t stmt_handle, int index, double value) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    return sqlite3_bind_double(stmt, index, value);
}

int eshkol_sqlite_bind_null(int64_t stmt_handle, int index) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    return sqlite3_bind_null(stmt, index);
}

/*******************************************************************************
 * Column Access
 ******************************************************************************/

int eshkol_sqlite_column_text(int64_t stmt_handle, int index,
                                char* buf, size_t buf_size) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt || !buf || buf_size == 0) return -1;

    const unsigned char* text = sqlite3_column_text(stmt, index);
    if (!text) {
        buf[0] = '\0';
        return 0;
    }
    size_t len = strlen((const char*)text);
    size_t copy = len < buf_size - 1 ? len : buf_size - 1;
    memcpy(buf, text, copy);
    buf[copy] = '\0';
    return (int)copy;
}

int64_t eshkol_sqlite_column_int(int64_t stmt_handle, int index) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return 0;
    return sqlite3_column_int64(stmt, index);
}

double eshkol_sqlite_column_double(int64_t stmt_handle, int index) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return 0.0;
    return sqlite3_column_double(stmt, index);
}

int eshkol_sqlite_column_count(int64_t stmt_handle) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return 0;
    return sqlite3_column_count(stmt);
}

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

int eshkol_sqlite_column_type(int64_t stmt_handle, int index) {
    sqlite3_stmt* stmt = get_stmt(stmt_handle);
    if (!stmt) return -1;
    return sqlite3_column_type(stmt, index);
}

/*******************************************************************************
 * Error Handling
 ******************************************************************************/

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

int64_t eshkol_sqlite_last_insert_rowid(int64_t db_handle) {
    sqlite3* db = get_db(db_handle);
    if (!db) return -1;
    return sqlite3_last_insert_rowid(db);
}

int eshkol_sqlite_changes(int64_t db_handle) {
    sqlite3* db = get_db(db_handle);
    if (!db) return 0;
    return sqlite3_changes(db);
}
