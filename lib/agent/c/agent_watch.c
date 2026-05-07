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

#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#else
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <dirent.h>
#include <limits.h>

#ifdef __APPLE__
#include <sys/event.h>
#include <sys/time.h>
#define USE_KQUEUE 1
#elif defined(__linux__)
#include <sys/inotify.h>
#define USE_INOTIFY 1
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

static void push_event(Watcher* w, const char* type, const char* path) {
    if (w->event_count >= MAX_EVENTS) return;  /* Drop oldest */
    snprintf(w->events[w->event_head], sizeof(w->events[0]),
             "%s\t%s", type, path);
    w->event_head = (w->event_head + 1) % MAX_EVENTS;
    w->event_count++;
}

#ifdef USE_KQUEUE

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
#else
    free(w);
    return -1;  /* No file watching on this platform */
#endif

    g_watchers[slot] = w;
    return (int64_t)slot;
}

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
#endif

    free(w);
    g_watchers[handle] = NULL;
}
