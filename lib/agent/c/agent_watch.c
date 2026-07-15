/*******************************************************************************
 * File Watching for Eshkol Agent (B.23)
 *
 * Native filesystem event monitoring via kqueue (macOS) or inotify (Linux).
 * Replaces fswatch/inotifywait shell-outs.
 *
 * Design: watchers accumulate events in an internal buffer. The agent polls
 * via eshkol_watch_poll() in its event loop. This avoids callback complexity.
 *
 * Copyright (c) 2025 Eshkol Project — tsotchke
 ******************************************************************************/

#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#elif !defined(_WIN32)
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX 32768
#endif
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <dirent.h>
#endif

#ifdef __APPLE__
#include <sys/event.h>
#include <sys/time.h>
#define USE_KQUEUE 1
#elif defined(__linux__)
#include <sys/inotify.h>
#define USE_INOTIFY 1
#elif defined(_WIN32)
#define USE_READ_DIRECTORY_CHANGES 1
#endif

#define MAX_WATCHERS 16
#define MAX_WATCH_FDS 256
#define MAX_EVENTS 64

typedef struct {
    char path[PATH_MAX];
    int  fd;            /* File descriptor being watched */
    int  wd;            /* Watch descriptor (inotify) or fd (kqueue) */
} WatchEntry;

typedef struct {
#ifdef USE_KQUEUE
    int kq;
#elif defined(USE_INOTIFY)
    int inotify_fd;
#elif defined(USE_READ_DIRECTORY_CHANGES)
    HANDLE directory;
    HANDLE event;
    OVERLAPPED overlap;
    unsigned char change_buffer[65536];
    wchar_t root[PATH_MAX];
    int request_pending;
#endif
    WatchEntry entries[MAX_WATCH_FDS];
    int n_entries;
    int recursive;
    /* Event buffer */
    char events[MAX_EVENTS][PATH_MAX + 32]; /* "event_type\tpath" */
    int event_head;
    int event_tail;
    int event_count;
} Watcher;

static Watcher* g_watchers[MAX_WATCHERS] = {0};
static void push_event(Watcher* w, const char* type, const char* path);

#ifdef USE_READ_DIRECTORY_CHANGES
static wchar_t* watch_utf8_to_wide(const char* value) {
    int needed = value ? MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                              value, -1, NULL, 0) : 0;
    if (needed <= 0) return NULL;
    wchar_t* wide = (wchar_t*)calloc((size_t)needed, sizeof(wchar_t));
    if (!wide) return NULL;
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                            value, -1, wide, needed) <= 0) {
        free(wide); return NULL;
    }
    for (wchar_t* p = wide; *p; ++p) if (*p == L'/') *p = L'\\';
    return wide;
}

static int issue_watch_request(Watcher* watcher) {
    ResetEvent(watcher->event);
    memset(&watcher->overlap, 0, sizeof(watcher->overlap));
    watcher->overlap.hEvent = watcher->event;
    DWORD filter = FILE_NOTIFY_CHANGE_FILE_NAME |
                   FILE_NOTIFY_CHANGE_DIR_NAME |
                   FILE_NOTIFY_CHANGE_ATTRIBUTES |
                   FILE_NOTIFY_CHANGE_SIZE |
                   FILE_NOTIFY_CHANGE_LAST_WRITE |
                   FILE_NOTIFY_CHANGE_CREATION |
                   FILE_NOTIFY_CHANGE_SECURITY;
    if (!ReadDirectoryChangesW(watcher->directory, watcher->change_buffer,
                               sizeof(watcher->change_buffer),
                               watcher->recursive ? TRUE : FALSE,
                               filter, NULL, &watcher->overlap, NULL)) {
        watcher->request_pending = 0;
        return -1;
    }
    watcher->request_pending = 1;
    return 0;
}

static void push_windows_event(Watcher* watcher, DWORD action,
                               const wchar_t* relative, size_t relative_chars) {
    size_t root_chars = wcslen(watcher->root);
    int separator = root_chars && watcher->root[root_chars - 1] != L'\\';
    size_t total = root_chars + (size_t)separator + relative_chars;
    wchar_t* full = (wchar_t*)malloc((total + 1) * sizeof(wchar_t));
    if (!full) return;
    memcpy(full, watcher->root, root_chars * sizeof(wchar_t));
    size_t cursor = root_chars;
    if (separator) full[cursor++] = L'\\';
    memcpy(full + cursor, relative, relative_chars * sizeof(wchar_t));
    full[total] = L'\0';
    int needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                     full, -1, NULL, 0, NULL, NULL);
    char* utf8 = needed > 0 ? (char*)malloc((size_t)needed) : NULL;
    if (utf8 && WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                    full, -1, utf8, needed, NULL, NULL) > 0) {
        for (char* p = utf8; *p; ++p) if (*p == '\\') *p = '/';
        const char* type = "change";
        if (action == FILE_ACTION_ADDED) type = "create";
        else if (action == FILE_ACTION_REMOVED) type = "delete";
        else if (action == FILE_ACTION_RENAMED_OLD_NAME ||
                 action == FILE_ACTION_RENAMED_NEW_NAME) type = "rename";
        push_event(watcher, type, utf8);
    }
    free(utf8); free(full);
}
#endif

/**
 * @brief Appends an event to a watcher's ring buffer of pending events.
 *
 * Formats @p type and @p path as "type\tpath" for later retrieval by
 * eshkol_watch_poll(). If the buffer is full, the new event is silently
 * dropped (oldest events are kept, not overwritten).
 *
 * @param w Watcher whose event ring buffer is appended to.
 * @param type Event type string (e.g. "change", "create", "delete", "rename").
 * @param path Path the event pertains to.
 */
static void push_event(Watcher* w, const char* type, const char* path) {
    if (w->event_count >= MAX_EVENTS) return;  /* Drop oldest */
    snprintf(w->events[w->event_head], sizeof(w->events[0]),
             "%s\t%s", type, path);
    w->event_head = (w->event_head + 1) % MAX_EVENTS;
    w->event_count++;
}

#ifdef USE_KQUEUE

/**
 * @brief Opens @p path and registers a kqueue EVFILT_VNODE watch on it.
 *
 * Watches for write, delete, rename, and attribute-change events with
 * EV_CLEAR (edge-triggered) semantics, and records the opened fd/path in the
 * watcher's entry table for later lookup and cleanup.
 *
 * @param w Watcher to add the entry to.
 * @param path Filesystem path to watch.
 */
static void add_kqueue_watch(Watcher* w, const char* path) {
    if (w->n_entries >= MAX_WATCH_FDS) return;
    int fd = open(path, O_RDONLY | O_EVTONLY);
    if (fd < 0) return;

    struct kevent ev;
    EV_SET(&ev, fd, EVFILT_VNODE,
           EV_ADD | EV_CLEAR,
           NOTE_WRITE | NOTE_DELETE | NOTE_RENAME | NOTE_ATTRIB,
           0, NULL);
    kevent(w->kq, &ev, 1, NULL, 0, NULL);

    WatchEntry* e = &w->entries[w->n_entries++];
    strncpy(e->path, path, PATH_MAX - 1);
    e->fd = fd;
    e->wd = fd;
}

/**
 * @brief Recursively registers kqueue watches on @p path and every subdirectory beneath it.
 *
 * Skips dotfiles/dot-directories. Each subdirectory found via readdir() is
 * watched via add_kqueue_watch() and then recursed into.
 *
 * @param w Watcher to add entries to.
 * @param path Root directory path to watch recursively.
 */
static void add_recursive(Watcher* w, const char* path) {
    add_kqueue_watch(w, path);
    DIR* dir = opendir(path);
    if (!dir) return;
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char child[PATH_MAX];
        snprintf(child, sizeof(child), "%s/%s", path, ent->d_name);
        struct stat st;
        if (lstat(child, &st) == 0 && S_ISDIR(st.st_mode)) {
            add_recursive(w, child);
        }
    }
    closedir(dir);
}

#elif defined(USE_INOTIFY)

/**
 * @brief Registers an inotify watch on @p path for create/delete/modify/rename events.
 *
 * Records the returned watch descriptor and path in the watcher's entry
 * table for later lookup and cleanup.
 *
 * @param w Watcher to add the entry to.
 * @param path Filesystem path to watch.
 */
static void add_inotify_watch(Watcher* w, const char* path) {
    if (w->n_entries >= MAX_WATCH_FDS) return;
    int wd = inotify_add_watch(w->inotify_fd, path,
                                IN_CREATE | IN_DELETE | IN_MODIFY | IN_MOVED_FROM | IN_MOVED_TO);
    if (wd < 0) return;
    WatchEntry* e = &w->entries[w->n_entries++];
    strncpy(e->path, path, PATH_MAX - 1);
    e->fd = w->inotify_fd;
    e->wd = wd;
}

/**
 * @brief Recursively registers inotify watches on @p path and every subdirectory beneath it.
 *
 * Skips dotfiles/dot-directories. Each subdirectory found via readdir() is
 * watched via add_inotify_watch() and then recursed into.
 *
 * @param w Watcher to add entries to.
 * @param path Root directory path to watch recursively.
 */
static void add_recursive(Watcher* w, const char* path) {
    add_inotify_watch(w, path);
    DIR* dir = opendir(path);
    if (!dir) return;
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char child[PATH_MAX];
        snprintf(child, sizeof(child), "%s/%s", path, ent->d_name);
        struct stat st;
        if (lstat(child, &st) == 0 && S_ISDIR(st.st_mode)) {
            add_recursive(w, child);
        }
    }
    closedir(dir);
}

#endif

/**
 * @brief Starts watching a filesystem path for change events, returning a handle.
 *
 * Allocates a Watcher slot and backing OS resource (a kqueue on macOS, an
 * inotify instance on Linux), then registers a watch on @p path — and, if
 * @p recursive is set, on every subdirectory beneath it.
 *
 * @param path Filesystem path to watch.
 * @param recursive Nonzero to also watch all subdirectories.
 * @return Watcher handle (>= 1) on success, -1 on error or on platforms with
 *         no supported file-watching backend.
 */
/*
 * Start watching a path for filesystem events.
 * recursive: 1 to watch all subdirectories
 * Returns: watcher handle (>= 1), -1 error
 */
int64_t eshkol_watch_start(const char* path, int32_t recursive) {
    if (!path) return -1;

    int slot = -1;
    for (int i = 1; i < MAX_WATCHERS; i++) {
        if (!g_watchers[i]) { slot = i; break; }
    }
    if (slot < 0) return -1;

    Watcher* w = (Watcher*)calloc(1, sizeof(Watcher));
    if (!w) return -1;
    w->recursive = recursive;

#ifdef USE_KQUEUE
    w->kq = kqueue();
    if (w->kq < 0) { free(w); return -1; }
    if (recursive) add_recursive(w, path);
    else add_kqueue_watch(w, path);
#elif defined(USE_INOTIFY)
    w->inotify_fd = inotify_init1(IN_NONBLOCK);
    if (w->inotify_fd < 0) { free(w); return -1; }
    if (recursive) add_recursive(w, path);
    else add_inotify_watch(w, path);
#elif defined(USE_READ_DIRECTORY_CHANGES)
    wchar_t* wide = watch_utf8_to_wide(path);
    if (!wide) { free(w); return -1; }
    DWORD attrs = GetFileAttributesW(wide);
    if (attrs == INVALID_FILE_ATTRIBUTES || !(attrs & FILE_ATTRIBUTE_DIRECTORY)) {
        free(wide); free(w); return -1;
    }
    wcsncpy(w->root, wide, PATH_MAX - 1);
    w->root[PATH_MAX - 1] = L'\0';
    w->directory = CreateFileW(wide, FILE_LIST_DIRECTORY,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL, OPEN_EXISTING,
                               FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED,
                               NULL);
    free(wide);
    if (w->directory == INVALID_HANDLE_VALUE) { free(w); return -1; }
    w->event = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!w->event) { CloseHandle(w->directory); free(w); return -1; }
    if (issue_watch_request(w) != 0) {
        CloseHandle(w->event); CloseHandle(w->directory); free(w); return -1;
    }
#else
    free(w);
    return -1;  /* No file watching on this platform */
#endif

    g_watchers[slot] = w;
    return (int64_t)slot;
}

/**
 * @brief Retrieves the next pending filesystem event for a watcher, if any.
 *
 * Drains the watcher's internal event buffer first; if empty, does a
 * non-blocking poll of the underlying OS mechanism (kqueue or inotify),
 * translates any new events into "type\tpath" strings via push_event(), and
 * returns the oldest buffered event.
 *
 * @param handle Watcher handle from eshkol_watch_start().
 * @param buf Output buffer receiving "event_type\tpath", NUL-terminated.
 * @param buf_size Size of @p buf in bytes.
 * @return Length of the event string written to @p buf, 0 if no events are
 *         pending, or -1 on an invalid handle/buffer.
 */
/*
 * Poll for filesystem events.
 *
 * Returns: next event as "event_type\tpath" in buf, strlen.
 *          0 if no events pending
 *          -1 on error
 *
 * event_type: "change", "create", "delete", "rename"
 */
int32_t eshkol_watch_poll(int64_t handle, char* buf, int32_t buf_size) {
    if (handle < 1 || handle >= MAX_WATCHERS || !g_watchers[handle]) return -1;
    if (!buf || buf_size <= 0) return -1;
    Watcher* w = g_watchers[handle];

    /* First drain any buffered events */
    if (w->event_count > 0) {
        int len = (int32_t)strlen(w->events[w->event_tail]);
        if (len >= buf_size) len = buf_size - 1;
        memcpy(buf, w->events[w->event_tail], (size_t)len);
        buf[len] = '\0';
        w->event_tail = (w->event_tail + 1) % MAX_EVENTS;
        w->event_count--;
        return len;
    }

    /* Check for new OS events (non-blocking) */
#ifdef USE_KQUEUE
    struct kevent events[16];
    struct timespec timeout = {0, 0};  /* Non-blocking */
    int n = kevent(w->kq, NULL, 0, events, 16, &timeout);
    for (int i = 0; i < n; i++) {
        int fd = (int)events[i].ident;
        const char* path = "unknown";
        for (int j = 0; j < w->n_entries; j++) {
            if (w->entries[j].fd == fd) { path = w->entries[j].path; break; }
        }
        const char* type = "change";
        if (events[i].fflags & NOTE_DELETE) type = "delete";
        else if (events[i].fflags & NOTE_RENAME) type = "rename";
        push_event(w, type, path);
    }
#elif defined(USE_INOTIFY)
    char ibuf[4096];
    ssize_t len = read(w->inotify_fd, ibuf, sizeof(ibuf));
    if (len > 0) {
        char* ptr = ibuf;
        while (ptr < ibuf + len) {
            struct inotify_event* ev = (struct inotify_event*)ptr;
            /* Find path for this wd */
            const char* dir_path = "";
            for (int j = 0; j < w->n_entries; j++) {
                if (w->entries[j].wd == ev->wd) { dir_path = w->entries[j].path; break; }
            }
            char full[PATH_MAX];
            if (ev->len > 0)
                snprintf(full, sizeof(full), "%s/%s", dir_path, ev->name);
            else
                strncpy(full, dir_path, sizeof(full) - 1);

            const char* type = "change";
            if (ev->mask & IN_CREATE) type = "create";
            else if (ev->mask & IN_DELETE) type = "delete";
            else if (ev->mask & (IN_MOVED_FROM | IN_MOVED_TO)) type = "rename";
            push_event(w, type, full);

            ptr += sizeof(struct inotify_event) + ev->len;
        }
    }
#elif defined(USE_READ_DIRECTORY_CHANGES)
    if (!w->request_pending && issue_watch_request(w) != 0) return -1;
    DWORD bytes = 0;
    if (GetOverlappedResult(w->directory, &w->overlap, &bytes, FALSE)) {
        w->request_pending = 0;
        size_t offset = 0;
        while (offset < bytes) {
            FILE_NOTIFY_INFORMATION* event =
                (FILE_NOTIFY_INFORMATION*)(w->change_buffer + offset);
            push_windows_event(w, event->Action, event->FileName,
                               event->FileNameLength / sizeof(wchar_t));
            if (event->NextEntryOffset == 0) break;
            offset += event->NextEntryOffset;
        }
        if (issue_watch_request(w) != 0) return -1;
    } else {
        DWORD error = GetLastError();
        if (error != ERROR_IO_INCOMPLETE) {
            w->request_pending = 0;
            if (issue_watch_request(w) != 0) return -1;
        }
    }
#endif

    /* Return first buffered event if any */
    if (w->event_count > 0) {
        int elen = (int32_t)strlen(w->events[w->event_tail]);
        if (elen >= buf_size) elen = buf_size - 1;
        memcpy(buf, w->events[w->event_tail], (size_t)elen);
        buf[elen] = '\0';
        w->event_tail = (w->event_tail + 1) % MAX_EVENTS;
        w->event_count--;
        return elen;
    }

    return 0;  /* No events */
}

/**
 * @brief Stops a watcher and releases all of its OS and memory resources.
 *
 * Closes every per-entry fd/removes every inotify watch descriptor, closes
 * the kqueue/inotify fd, frees the Watcher struct, and clears its handle slot.
 *
 * @param handle Watcher handle from eshkol_watch_start(). No-op if invalid.
 */
/*
 * Stop watching and free resources.
 */
void eshkol_watch_stop(int64_t handle) {
    if (handle < 1 || handle >= MAX_WATCHERS || !g_watchers[handle]) return;
    Watcher* w = g_watchers[handle];

#ifdef USE_KQUEUE
    for (int i = 0; i < w->n_entries; i++) {
        close(w->entries[i].fd);
    }
    close(w->kq);
#elif defined(USE_INOTIFY)
    for (int i = 0; i < w->n_entries; i++) {
        inotify_rm_watch(w->inotify_fd, w->entries[i].wd);
    }
    close(w->inotify_fd);
#elif defined(USE_READ_DIRECTORY_CHANGES)
    if (w->request_pending) CancelIoEx(w->directory, &w->overlap);
    if (w->event) CloseHandle(w->event);
    if (w->directory && w->directory != INVALID_HANDLE_VALUE) CloseHandle(w->directory);
#endif

    free(w);
    g_watchers[handle] = NULL;
}
