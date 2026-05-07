/*******************************************************************************
 * System Platform Primitives for Eshkol Agent
 *
 * Provides: OS info, home directory, shell quoting, temp files, sleep,
 *           process identity, stderr output, recursive mkdir/rmdir,
 *           file stat, UUID, timestamps, executable search.
 *
 * All functions use the eshkol agent FFI conventions:
 *   - Strings: null-terminated, output via pre-allocated (buf, buf_size)
 *   - Returns: 0 success / -1 error, or strlen for string outputs
 *   - Handles: int64_t indices into static tables
 *
 * Copyright (c) 2025 Eshkol Project — tsotchke
 ******************************************************************************/

#ifdef __APPLE__
#define _DARWIN_C_SOURCE    /* flock, mkdtemp, timegm on macOS */
#else
#define _GNU_SOURCE         /* nftw, strptime, timegm, mkdtemp on Linux */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <dirent.h>
#include <fcntl.h>
#include <ftw.h>
#include <pwd.h>
#include <fnmatch.h>
#include <limits.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <copyfile.h>
#endif

/*******************************************************************************
 * D.1 / B.7: OS Information (compile-time constants)
 ******************************************************************************/

const char* eshkol_os_type(void) {
#ifdef __APPLE__
    return "darwin";
#elif defined(__linux__)
    return "linux";
#elif defined(_WIN32)
    return "windows";
#else
    return "unknown";
#endif
}

const char* eshkol_os_arch(void) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__arm__)
    return "arm";
#else
    return "unknown";
#endif
}

/*******************************************************************************
 * B.7: Home Directory
 ******************************************************************************/

int32_t eshkol_home_directory(char* buf, int32_t buf_size) {
    /* Try getpwuid first (most reliable) */
    struct passwd* pw = getpwuid(getuid());
    if (pw && pw->pw_dir && pw->pw_dir[0] != '\0') {
        int len = (int32_t)strlen(pw->pw_dir);
        if (len >= buf_size) return -1;
        memcpy(buf, pw->pw_dir, (size_t)len + 1);
        return len;
    }
    /* Fallback to $HOME */
    const char* home = getenv("HOME");
    if (home && home[0] != '\0') {
        int len = (int32_t)strlen(home);
        if (len >= buf_size) return -1;
        memcpy(buf, home, (size_t)len + 1);
        return len;
    }
    return -1;
}

/*******************************************************************************
 * B.7: Hostname, Username
 ******************************************************************************/

int32_t eshkol_hostname(char* buf, int32_t buf_size) {
    if (gethostname(buf, (size_t)buf_size) != 0) return -1;
    buf[buf_size - 1] = '\0';
    return (int32_t)strlen(buf);
}

int32_t eshkol_username(char* buf, int32_t buf_size) {
    struct passwd* pw = getpwuid(getuid());
    if (!pw || !pw->pw_name) return -1;
    int len = (int32_t)strlen(pw->pw_name);
    if (len >= buf_size) return -1;
    memcpy(buf, pw->pw_name, (size_t)len + 1);
    return len;
}

/*******************************************************************************
 * B.7: Executable Search (PATH lookup)
 ******************************************************************************/

int32_t eshkol_executable_exists(const char* name) {
    if (!name || name[0] == '\0') return 0;
    /* If absolute path, check directly */
    if (name[0] == '/') return access(name, X_OK) == 0 ? 1 : 0;

    const char* path_env = getenv("PATH");
    if (!path_env) return 0;

    char full[PATH_MAX];
    const char* p = path_env;
    while (*p) {
        const char* colon = strchr(p, ':');
        size_t dir_len = colon ? (size_t)(colon - p) : strlen(p);
        if (dir_len > 0 && dir_len + strlen(name) + 2 < PATH_MAX) {
            memcpy(full, p, dir_len);
            full[dir_len] = '/';
            strcpy(full + dir_len + 1, name);
            if (access(full, X_OK) == 0) return 1;
        }
        if (!colon) break;
        p = colon + 1;
    }
    return 0;
}

int32_t eshkol_executable_path(const char* name, char* buf, int32_t buf_size) {
    if (!name || !buf || buf_size < 2) return -1;
    if (name[0] == '/') {
        if (access(name, X_OK) == 0) {
            int len = (int32_t)strlen(name);
            if (len >= buf_size) return -1;
            memcpy(buf, name, (size_t)len + 1);
            return len;
        }
        return -1;
    }
    const char* path_env = getenv("PATH");
    if (!path_env) return -1;

    char full[PATH_MAX];
    const char* p = path_env;
    while (*p) {
        const char* colon = strchr(p, ':');
        size_t dir_len = colon ? (size_t)(colon - p) : strlen(p);
        if (dir_len > 0 && dir_len + strlen(name) + 2 < PATH_MAX) {
            memcpy(full, p, dir_len);
            full[dir_len] = '/';
            strcpy(full + dir_len + 1, name);
            if (access(full, X_OK) == 0) {
                int len = (int32_t)strlen(full);
                if (len >= buf_size) return -1;
                memcpy(buf, full, (size_t)len + 1);
                return len;
            }
        }
        if (!colon) break;
        p = colon + 1;
    }
    return -1;
}

/*******************************************************************************
 * B.7: Time (millisecond precision)
 ******************************************************************************/

int64_t eshkol_current_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
}

int64_t eshkol_monotonic_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
}

const char* eshkol_temp_directory(void) {
    const char* tmp = getenv("TMPDIR");
    if (tmp && tmp[0] != '\0') return tmp;
    return "/tmp";
}

/*******************************************************************************
 * E.6: getpid
 ******************************************************************************/

int64_t eshkol_getpid_val(void) {
    return (int64_t)getpid();
}

/*******************************************************************************
 * E.5: stderr output
 ******************************************************************************/

void eshkol_eprint(const char* str) {
    if (str) {
        fputs(str, stderr);
        fflush(stderr);
    }
}

/*******************************************************************************
 * E.7: Precise millisecond sleep
 ******************************************************************************/

void eshkol_sleep_ms(int64_t ms) {
    if (ms <= 0) return;
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

/*******************************************************************************
 * B.22: Shell Quoting (POSIX single-quote escaping)
 ******************************************************************************/

int32_t eshkol_shell_quote(const char* str, char* buf, int32_t buf_size) {
    if (!str || !buf || buf_size < 3) return -1;

    /* Check if quoting is needed */
    int needs_quoting = 0;
    for (const char* p = str; *p; p++) {
        if (*p == ' ' || *p == '\'' || *p == '"' || *p == '\\' ||
            *p == '$' || *p == '`' || *p == '!' || *p == '&' ||
            *p == '|' || *p == ';' || *p == '(' || *p == ')' ||
            *p == '<' || *p == '>' || *p == '*' || *p == '?' ||
            *p == '[' || *p == ']' || *p == '{' || *p == '}' ||
            *p == '#' || *p == '~' || *p == '\n' || *p == '\t') {
            needs_quoting = 1;
            break;
        }
    }
    if (!needs_quoting) {
        int len = (int32_t)strlen(str);
        if (len >= buf_size) return -1;
        memcpy(buf, str, (size_t)len + 1);
        return len;
    }

    /* Single-quote escaping: wrap in '' and escape internal ' as '\'' */
    int32_t out = 0;
    buf[out++] = '\'';
    for (const char* p = str; *p; p++) {
        if (*p == '\'') {
            /* Need 4 chars: '\'' */
            if (out + 4 >= buf_size) return -1;
            buf[out++] = '\'';
            buf[out++] = '\\';
            buf[out++] = '\'';
            buf[out++] = '\'';
        } else {
            if (out + 1 >= buf_size) return -1;
            buf[out++] = *p;
        }
    }
    if (out + 1 >= buf_size) return -1;
    buf[out++] = '\'';
    buf[out] = '\0';
    return out;
}

/*******************************************************************************
 * E.8: Temp File/Dir Creation (race-free)
 ******************************************************************************/

int32_t eshkol_mkstemp_path(const char* prefix, const char* suffix,
                             const char* dir, char* path_buf, int32_t buf_size) {
    if (!prefix || !path_buf || buf_size < 32) return -1;
    const char* tmpdir = dir && dir[0] != '\0' ? dir : eshkol_temp_directory();
    char tmpl[PATH_MAX];
    int n = snprintf(tmpl, sizeof(tmpl), "%s/%sXXXXXX", tmpdir, prefix);
    if (n < 0 || n >= (int)sizeof(tmpl)) return -1;

    int fd = mkstemp(tmpl);
    if (fd < 0) return -1;
    close(fd);

    /* If suffix requested, rename */
    if (suffix && suffix[0] != '\0') {
        char final_path[PATH_MAX];
        snprintf(final_path, sizeof(final_path), "%s%s", tmpl, suffix);
        rename(tmpl, final_path);
        int len = (int32_t)strlen(final_path);
        if (len >= buf_size) { unlink(final_path); return -1; }
        memcpy(path_buf, final_path, (size_t)len + 1);
        return len;
    }

    int len = (int32_t)strlen(tmpl);
    if (len >= buf_size) { unlink(tmpl); return -1; }
    memcpy(path_buf, tmpl, (size_t)len + 1);
    return len;
}

int32_t eshkol_mkdtemp_path(const char* prefix, const char* dir,
                              char* path_buf, int32_t buf_size) {
    if (!prefix || !path_buf || buf_size < 32) return -1;
    const char* tmpdir = dir && dir[0] != '\0' ? dir : eshkol_temp_directory();
    char tmpl[PATH_MAX];
    int n = snprintf(tmpl, sizeof(tmpl), "%s/%sXXXXXX", tmpdir, prefix);
    if (n < 0 || n >= (int)sizeof(tmpl)) return -1;

    if (!mkdtemp(tmpl)) return -1;

    int len = (int32_t)strlen(tmpl);
    if (len >= buf_size) return -1;
    memcpy(path_buf, tmpl, (size_t)len + 1);
    return len;
}

/*******************************************************************************
 * B.1: Recursive mkdir (create all parents)
 ******************************************************************************/

int32_t eshkol_mkdir_recursive(const char* path, int32_t mode) {
    if (!path || path[0] == '\0') return -1;
    char tmp[PATH_MAX];
    size_t len = strlen(path);
    if (len >= PATH_MAX) return -1;
    memcpy(tmp, path, len + 1);

    /* Walk each component and mkdir */
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, (mode_t)mode) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, (mode_t)mode) != 0 && errno != EEXIST) return -1;
    return 0;
}

/*******************************************************************************
 * B.1: Recursive rmdir (with safety checks)
 ******************************************************************************/

static const char* g_dangerous_paths[] = {
    "/", "/usr", "/bin", "/sbin", "/etc", "/var", "/tmp",
    "/home", "/Users", "/System", "/Library", "/opt",
    NULL
};

static int rmdir_recursive_cb(const char* fpath, const struct stat* sb,
                               int typeflag, struct FTW* ftwbuf) {
    (void)sb; (void)ftwbuf;
    if (typeflag == FTW_D || typeflag == FTW_DP) {
        return rmdir(fpath);
    }
    return unlink(fpath);
}

int32_t eshkol_rmdir_recursive(const char* path) {
    if (!path || path[0] == '\0') return -1;

    /* Safety: refuse dangerous paths */
    char resolved[PATH_MAX];
    if (!realpath(path, resolved)) {
        /* Path doesn't exist, nothing to delete */
        return 0;
    }
    for (const char** p = g_dangerous_paths; *p; p++) {
        if (strcmp(resolved, *p) == 0) {
            errno = EPERM;
            return -1;
        }
    }

    return nftw(resolved, rmdir_recursive_cb, 64, FTW_DEPTH | FTW_PHYS);
}

/*******************************************************************************
 * B.1: File Stat (structured)
 ******************************************************************************/

int32_t eshkol_file_stat_fields(const char* path,
                                  int64_t* out_size, int64_t* out_mtime,
                                  int64_t* out_ctime, int32_t* out_mode,
                                  int32_t* out_type) {
    if (!path) return -1;
    struct stat st;
    if (lstat(path, &st) != 0) return -1;

    if (out_size)  *out_size  = (int64_t)st.st_size;
    if (out_mtime) *out_mtime = (int64_t)st.st_mtime;
    if (out_ctime) *out_ctime = (int64_t)st.st_ctime;
    if (out_mode)  *out_mode  = (int32_t)st.st_mode;

    if (out_type) {
        if (S_ISREG(st.st_mode))       *out_type = 0;  /* file */
        else if (S_ISDIR(st.st_mode))  *out_type = 1;  /* directory */
        else if (S_ISLNK(st.st_mode))  *out_type = 2;  /* symlink */
        else                           *out_type = 3;  /* other */
    }
    return 0;
}

/*******************************************************************************
 * B.1: File Copy
 ******************************************************************************/

int32_t eshkol_file_copy(const char* src, const char* dst) {
    if (!src || !dst) return -1;

#ifdef __APPLE__
    /* Try CoW clone first (instant, no I/O) */
    if (copyfile(src, dst, NULL, COPYFILE_ALL | COPYFILE_CLONE) == 0) return 0;
#endif

    /* Fallback: read/write in chunks */
    int in_fd = open(src, O_RDONLY);
    if (in_fd < 0) return -1;

    int out_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) { close(in_fd); return -1; }

    char chunk[65536];
    ssize_t n;
    int result = 0;
    while ((n = read(in_fd, chunk, sizeof(chunk))) > 0) {
        ssize_t written = 0;
        while (written < n) {
            ssize_t w = write(out_fd, chunk + written, (size_t)(n - written));
            if (w < 0) { result = -1; goto done; }
            written += w;
        }
    }
    if (n < 0) result = -1;

done:
    close(in_fd);
    close(out_fd);
    return result;
}

/*******************************************************************************
 * B.1: File chmod, symlink, realpath, glob-match, file-lock
 ******************************************************************************/

int32_t eshkol_file_chmod(const char* path, int32_t mode) {
    return chmod(path, (mode_t)mode) == 0 ? 0 : -1;
}

int32_t eshkol_symlink_create(const char* target, const char* link_path) {
    return symlink(target, link_path) == 0 ? 0 : -1;
}

int32_t eshkol_symlink_read(const char* path, char* buf, int32_t buf_size) {
    ssize_t n = readlink(path, buf, (size_t)(buf_size - 1));
    if (n < 0) return -1;
    buf[n] = '\0';
    return (int32_t)n;
}

int32_t eshkol_realpath_resolve(const char* path, char* buf, int32_t buf_size) {
    char resolved[PATH_MAX];
    if (!realpath(path, resolved)) return -1;
    int len = (int32_t)strlen(resolved);
    if (len >= buf_size) return -1;
    memcpy(buf, resolved, (size_t)len + 1);
    return len;
}

int32_t eshkol_glob_match(const char* pattern, const char* path) {
    return fnmatch(pattern, path, FNM_PATHNAME) == 0 ? 1 : 0;
}

int64_t eshkol_file_lock(const char* path) {
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) return -1;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        return -1;
    }
    return (int64_t)fd;
}

int32_t eshkol_file_unlock(int64_t fd) {
    if (fd < 0) return -1;
    flock((int)fd, LOCK_UN);
    close((int)fd);
    return 0;
}

/*******************************************************************************
 * B.6: UUID v4
 ******************************************************************************/

void eshkol_uuid_v4(char* buf) {
    unsigned char bytes[16];
    /* Read from /dev/urandom */
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        read(fd, bytes, 16);
        close(fd);
    } else {
        /* Fallback: not cryptographically secure but functional */
        srand((unsigned)time(NULL) ^ (unsigned)getpid());
        for (int i = 0; i < 16; i++) bytes[i] = (unsigned char)(rand() & 0xFF);
    }
    /* Set version (4) and variant (RFC 4122) */
    bytes[6] = (bytes[6] & 0x0F) | 0x40;  /* version 4 */
    bytes[8] = (bytes[8] & 0x3F) | 0x80;  /* variant 10xx */

    snprintf(buf, 37, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             bytes[0], bytes[1], bytes[2], bytes[3],
             bytes[4], bytes[5], bytes[6], bytes[7],
             bytes[8], bytes[9], bytes[10], bytes[11],
             bytes[12], bytes[13], bytes[14], bytes[15]);
}

/*******************************************************************************
 * B.11: Date/Time Formatting
 ******************************************************************************/

int32_t eshkol_format_iso8601(int64_t epoch, char* buf, int32_t buf_size) {
    if (!buf || buf_size < 21) return -1;
    time_t t = (time_t)epoch;
    struct tm tm;
    gmtime_r(&t, &tm);
    int n = (int32_t)strftime(buf, (size_t)buf_size, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return n > 0 ? n : -1;
}

int64_t eshkol_parse_iso8601(const char* str) {
    if (!str) return -1;
    struct tm tm;
    memset(&tm, 0, sizeof(tm));
    if (!strptime(str, "%Y-%m-%dT%H:%M:%S", &tm)) return -1;
    return (int64_t)timegm(&tm);
}

int32_t eshkol_format_relative(int64_t seconds_ago, char* buf, int32_t buf_size) {
    if (!buf || buf_size < 16) return -1;
    int n;
    if (seconds_ago < 60)
        n = snprintf(buf, (size_t)buf_size, "%llds ago", (long long)seconds_ago);
    else if (seconds_ago < 3600)
        n = snprintf(buf, (size_t)buf_size, "%lldm ago", (long long)(seconds_ago / 60));
    else if (seconds_ago < 86400)
        n = snprintf(buf, (size_t)buf_size, "%lldh ago", (long long)(seconds_ago / 3600));
    else
        n = snprintf(buf, (size_t)buf_size, "%lldd ago", (long long)(seconds_ago / 86400));
    return n > 0 ? n : -1;
}

int64_t eshkol_local_timezone_offset(void) {
    time_t t = time(NULL);
    struct tm local, utc;
    localtime_r(&t, &local);
    gmtime_r(&t, &utc);
    return (int64_t)(mktime(&local) - mktime(&utc));
}

/*******************************************************************************
 * B.1: file-mmap / file-munmap
 ******************************************************************************/

#include <sys/mman.h>

/* mmap handle table */
#define MAX_MMAPS 16
static struct { void* ptr; size_t len; } g_mmaps[MAX_MMAPS] = {{0}};

int64_t eshkol_file_mmap(const char* path, int64_t offset, int64_t length) {
    if (!path) return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    /* If length is 0, map the whole file */
    if (length <= 0) {
        struct stat st;
        if (fstat(fd, &st) != 0) { close(fd); return -1; }
        length = st.st_size - offset;
        if (length <= 0) { close(fd); return -1; }
    }

    void* ptr = mmap(NULL, (size_t)length, PROT_READ, MAP_PRIVATE, fd, (off_t)offset);
    close(fd);
    if (ptr == MAP_FAILED) return -1;

    for (int i = 0; i < MAX_MMAPS; i++) {
        if (!g_mmaps[i].ptr) {
            g_mmaps[i].ptr = ptr;
            g_mmaps[i].len = (size_t)length;
            return (int64_t)i;
        }
    }
    munmap(ptr, (size_t)length);
    return -1;
}

int32_t eshkol_file_munmap(int64_t handle) {
    if (handle < 0 || handle >= MAX_MMAPS || !g_mmaps[handle].ptr) return -1;
    munmap(g_mmaps[handle].ptr, g_mmaps[handle].len);
    g_mmaps[handle].ptr = NULL;
    g_mmaps[handle].len = 0;
    return 0;
}

/* Read bytes from mmap'd region */
int32_t eshkol_mmap_read(int64_t handle, int64_t offset, char* buf, int32_t buf_size) {
    if (handle < 0 || handle >= MAX_MMAPS || !g_mmaps[handle].ptr) return -1;
    if (!buf || buf_size <= 0) return -1;
    size_t avail = g_mmaps[handle].len - (size_t)offset;
    if (offset < 0 || (size_t)offset >= g_mmaps[handle].len) return -1;
    int32_t to_read = (int32_t)(avail < (size_t)buf_size ? avail : (size_t)buf_size);
    memcpy(buf, (char*)g_mmaps[handle].ptr + offset, (size_t)to_read);
    return to_read;
}

int64_t eshkol_mmap_length(int64_t handle) {
    if (handle < 0 || handle >= MAX_MMAPS || !g_mmaps[handle].ptr) return -1;
    return (int64_t)g_mmaps[handle].len;
}

/*******************************************************************************
 * B.1: directory-walk — recursive directory traversal
 ******************************************************************************/

static int32_t directory_walk_impl(const char* base, int depth, int max_depth,
                                     char* buf, int32_t buf_size,
                                     int32_t* written, int32_t* count) {
    if (depth > max_depth && max_depth >= 0) return 0;

    DIR* dir = opendir(base);
    if (!dir) return 0;

    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;

        char full_path[PATH_MAX];
        int plen = snprintf(full_path, sizeof(full_path), "%s/%s", base, ent->d_name);
        if (plen < 0 || plen >= (int)sizeof(full_path)) continue;

        struct stat st;
        if (lstat(full_path, &st) != 0) continue;

        char type_ch = 'f';
        if (S_ISDIR(st.st_mode)) type_ch = 'd';
        else if (S_ISLNK(st.st_mode)) type_ch = 'l';

        /* Write "type:path\0" to buffer */
        if (*count > 0 && *written < buf_size - 1) {
            buf[*written] = '\0';
            (*written)++;
        }
        int n = snprintf(buf + *written, (size_t)(buf_size - *written),
                         "%c:%s", type_ch, full_path);
        if (n < 0 || *written + n >= buf_size) { closedir(dir); return -1; }
        *written += n;
        (*count)++;

        /* Recurse into directories */
        if (type_ch == 'd') {
            directory_walk_impl(full_path, depth + 1, max_depth,
                               buf, buf_size, written, count);
        }
    }
    closedir(dir);
    return 0;
}

int32_t eshkol_directory_walk(const char* path, int32_t max_depth,
                                char* buf, int32_t buf_size, int32_t* count) {
    if (!path || !buf || buf_size <= 0 || !count) return -1;
    int32_t written = 0;
    *count = 0;
    directory_walk_impl(path, 0, max_depth, buf, buf_size, &written, count);
    if (written < buf_size) buf[written] = '\0';
    return written;
}

/*******************************************************************************
 * B.1: glob-expand — walk + fnmatch
 ******************************************************************************/

static void glob_expand_impl(const char* dir_path, const char* pattern,
                               char* buf, int32_t buf_size,
                               int32_t* written, int32_t* count) {
    DIR* dir = opendir(dir_path);
    if (!dir) return;

    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;

        char full_path[PATH_MAX];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, ent->d_name);

        struct stat st;
        if (lstat(full_path, &st) != 0) continue;

        /* Check if name matches pattern */
        if (fnmatch(pattern, ent->d_name, 0) == 0) {
            if (*count > 0 && *written < buf_size - 1) {
                buf[*written] = '\0';
                (*written)++;
            }
            int n = snprintf(buf + *written, (size_t)(buf_size - *written), "%s", full_path);
            if (n >= 0 && *written + n < buf_size) {
                *written += n;
                (*count)++;
            }
        }

        /* Recurse into subdirectories */
        if (S_ISDIR(st.st_mode)) {
            glob_expand_impl(full_path, pattern, buf, buf_size, written, count);
        }
    }
    closedir(dir);
}

int32_t eshkol_glob_expand(const char* pattern, const char* root,
                             char* buf, int32_t buf_size, int32_t* count) {
    if (!pattern || !root || !buf || buf_size <= 0 || !count) return -1;
    int32_t written = 0;
    *count = 0;
    glob_expand_impl(root, pattern, buf, buf_size, &written, count);
    if (written < buf_size) buf[written] = '\0';
    return written;
}

/*******************************************************************************
 * B.22: shell-split — parse shell command into argv
 ******************************************************************************/

int32_t eshkol_shell_split(const char* cmd, char* buf, int32_t buf_size, int32_t* argc) {
    if (!cmd || !buf || buf_size <= 0 || !argc) return -1;
    *argc = 0;
    int32_t out = 0;
    const char* p = cmd;

    while (*p) {
        /* Skip whitespace */
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;

        /* Parse one argument */
        char quote = 0;
        if (*p == '\'' || *p == '"') { quote = *p; p++; }

        if (*argc > 0 && out < buf_size - 1) buf[out++] = '\0';

        while (*p) {
            if (quote) {
                if (*p == quote) { p++; quote = 0; break; }
                if (*p == '\\' && quote == '"' && *(p+1)) {
                    p++;
                    if (out < buf_size - 1) buf[out++] = *p;
                    p++;
                } else {
                    if (out < buf_size - 1) buf[out++] = *p;
                    p++;
                }
            } else {
                if (*p == ' ' || *p == '\t') break;
                if (*p == '\\' && *(p+1)) {
                    p++;
                    if (out < buf_size - 1) buf[out++] = *p;
                    p++;
                } else if (*p == '\'' || *p == '"') {
                    quote = *p; p++;
                } else {
                    if (out < buf_size - 1) buf[out++] = *p;
                    p++;
                }
            }
        }
        (*argc)++;
    }

    if (out < buf_size) buf[out] = '\0';
    return out;
}

/*******************************************************************************
 * B.8: string-display-width — Unicode-aware terminal column width
 *
 * Uses East Asian Width (UAX #11) classification:
 * - CJK ideographs (U+4E00-U+9FFF, U+3400-U+4DBF, etc.): 2 columns
 * - Fullwidth forms (U+FF01-U+FF60, U+FFE0-U+FFE6): 2 columns
 * - Emoji (U+1F300-U+1F9FF, etc.): 2 columns
 * - Zero-width marks (U+0300-U+036F, U+200B-U+200F, etc.): 0 columns
 * - ANSI escape sequences: 0 columns
 * - Everything else: 1 column
 ******************************************************************************/

static int is_wide_char(uint32_t cp) {
    /* CJK Unified Ideographs */
    if (cp >= 0x4E00 && cp <= 0x9FFF) return 1;
    if (cp >= 0x3400 && cp <= 0x4DBF) return 1;
    if (cp >= 0x20000 && cp <= 0x2A6DF) return 1;
    /* CJK Compatibility Ideographs */
    if (cp >= 0xF900 && cp <= 0xFAFF) return 1;
    /* Hangul Syllables */
    if (cp >= 0xAC00 && cp <= 0xD7AF) return 1;
    /* Fullwidth Forms */
    if (cp >= 0xFF01 && cp <= 0xFF60) return 1;
    if (cp >= 0xFFE0 && cp <= 0xFFE6) return 1;
    /* CJK Symbols */
    if (cp >= 0x2E80 && cp <= 0x303E) return 1;
    /* Katakana/Hiragana */
    if (cp >= 0x3040 && cp <= 0x30FF) return 1;
    if (cp >= 0x31F0 && cp <= 0x31FF) return 1;
    /* Emoji */
    if (cp >= 0x1F300 && cp <= 0x1F9FF) return 1;
    if (cp >= 0x1FA00 && cp <= 0x1FA6F) return 1;
    if (cp >= 0x1FA70 && cp <= 0x1FAFF) return 1;
    if (cp >= 0x2600 && cp <= 0x27BF) return 1;
    return 0;
}

static int is_zero_width(uint32_t cp) {
    /* Combining marks */
    if (cp >= 0x0300 && cp <= 0x036F) return 1;
    if (cp >= 0x1AB0 && cp <= 0x1AFF) return 1;
    if (cp >= 0x1DC0 && cp <= 0x1DFF) return 1;
    if (cp >= 0x20D0 && cp <= 0x20FF) return 1;
    if (cp >= 0xFE20 && cp <= 0xFE2F) return 1;
    /* Zero-width characters */
    if (cp == 0x200B || cp == 0x200C || cp == 0x200D || cp == 0x200E || cp == 0x200F) return 1;
    if (cp == 0xFEFF) return 1;  /* BOM / ZWNBSP */
    /* Variation selectors */
    if (cp >= 0xFE00 && cp <= 0xFE0F) return 1;
    if (cp >= 0xE0100 && cp <= 0xE01EF) return 1;
    return 0;
}

/* Decode one UTF-8 codepoint, advance *pos */
static uint32_t decode_utf8(const char* str, int len, int* pos) {
    unsigned char c = (unsigned char)str[*pos];
    uint32_t cp;
    int bytes;
    if (c < 0x80) { cp = c; bytes = 1; }
    else if (c < 0xC0) { cp = 0xFFFD; bytes = 1; }
    else if (c < 0xE0) { cp = c & 0x1F; bytes = 2; }
    else if (c < 0xF0) { cp = c & 0x0F; bytes = 3; }
    else { cp = c & 0x07; bytes = 4; }
    for (int i = 1; i < bytes && (*pos + i) < len; i++) {
        cp = (cp << 6) | ((unsigned char)str[*pos + i] & 0x3F);
    }
    *pos += bytes;
    return cp;
}

int32_t eshkol_string_display_width(const char* str) {
    if (!str) return 0;
    int len = (int)strlen(str);
    int width = 0;
    int pos = 0;
    int in_escape = 0;
    int escape_type = 0;

    while (pos < len) {
        unsigned char c = (unsigned char)str[pos];

        /* ANSI escape handling */
        if (!in_escape && c == 0x1B) {
            if (pos + 1 < len) {
                unsigned char next = (unsigned char)str[pos + 1];
                if (next == '[') { in_escape = 1; escape_type = 1; pos += 2; continue; }
                if (next == ']') { in_escape = 1; escape_type = 2; pos += 2; continue; }
                if (next >= 0x40 && next <= 0x7E) { pos += 2; continue; }
            }
            pos++;
            continue;
        }
        if (in_escape) {
            if (escape_type == 1) {
                /* CSI: consume until 0x40-0x7E */
                if (c >= 0x40 && c <= 0x7E) in_escape = 0;
                pos++;
            } else {
                /* OSC: consume until BEL or ST */
                if (c == 0x07) { in_escape = 0; pos++; }
                else if (c == 0x1B && pos + 1 < len && str[pos+1] == '\\') {
                    in_escape = 0; pos += 2;
                } else { pos++; }
            }
            continue;
        }

        /* Decode UTF-8 codepoint */
        uint32_t cp = decode_utf8(str, len, &pos);
        if (is_zero_width(cp)) continue;
        width += is_wide_char(cp) ? 2 : 1;
    }
    return width;
}

int32_t eshkol_string_truncate_display(const char* str, int32_t max_width,
                                         const char* suffix,
                                         char* buf, int32_t buf_size) {
    if (!str || !buf || buf_size <= 0 || max_width <= 0) return -1;
    int suffix_width = suffix ? eshkol_string_display_width(suffix) : 0;
    int target = max_width - suffix_width;
    if (target <= 0) target = max_width;

    int len = (int)strlen(str);
    int width = 0;
    int pos = 0;
    int last_safe = 0;
    int in_escape = 0, escape_type = 0;

    while (pos < len && width < target) {
        unsigned char c = (unsigned char)str[pos];
        if (!in_escape && c == 0x1B) {
            if (pos + 1 < len && str[pos+1] == '[') { in_escape = 1; escape_type = 1; pos += 2; continue; }
            if (pos + 1 < len && str[pos+1] == ']') { in_escape = 1; escape_type = 2; pos += 2; continue; }
            pos++;
            continue;
        }
        if (in_escape) {
            if (escape_type == 1 && c >= 0x40 && c <= 0x7E) in_escape = 0;
            else if (escape_type == 2 && (c == 0x07 || (c == 0x1B && pos+1 < len && str[pos+1] == '\\'))) {
                in_escape = 0;
                if (c == 0x1B) pos++;
            }
            pos++;
            continue;
        }
        int old_pos = pos;
        uint32_t cp = decode_utf8(str, len, &pos);
        int cw = is_zero_width(cp) ? 0 : (is_wide_char(cp) ? 2 : 1);
        if (width + cw > target) { pos = old_pos; break; }
        width += cw;
        last_safe = pos;
    }

    int full_width = eshkol_string_display_width(str);
    if (full_width <= max_width) {
        /* No truncation needed */
        int slen = (int)strlen(str);
        if (slen >= buf_size) slen = buf_size - 1;
        memcpy(buf, str, (size_t)slen);
        buf[slen] = '\0';
        return slen;
    }

    /* Truncate + add suffix */
    int out = 0;
    if (last_safe > 0 && last_safe < buf_size - 1) {
        memcpy(buf, str, (size_t)last_safe);
        out = last_safe;
    }
    if (suffix && suffix_width > 0) {
        int slen = (int)strlen(suffix);
        if (out + slen < buf_size) {
            memcpy(buf + out, suffix, (size_t)slen);
            out += slen;
        }
    }
    buf[out] = '\0';
    return out;
}

/*******************************************************************************
 * B.6: base64url-encode / decode (URL-safe, no padding)
 ******************************************************************************/

static const char b64url_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

int32_t eshkol_base64url_encode(const char* data, int32_t data_len,
                                  char* buf, int32_t buf_size) {
    if (!data || !buf || data_len < 0 || buf_size <= 0) return -1;
    int j = 0;
    int i;
    for (i = 0; i + 2 < data_len && j + 4 < buf_size; i += 3) {
        unsigned int n = ((unsigned char)data[i] << 16) |
                         ((unsigned char)data[i+1] << 8) |
                          (unsigned char)data[i+2];
        buf[j++] = b64url_table[(n >> 18) & 63];
        buf[j++] = b64url_table[(n >> 12) & 63];
        buf[j++] = b64url_table[(n >> 6) & 63];
        buf[j++] = b64url_table[n & 63];
    }
    if (i < data_len && j + 3 < buf_size) {
        unsigned int n = (unsigned char)data[i] << 16;
        if (i + 1 < data_len) n |= (unsigned char)data[i+1] << 8;
        buf[j++] = b64url_table[(n >> 18) & 63];
        buf[j++] = b64url_table[(n >> 12) & 63];
        if (i + 1 < data_len) buf[j++] = b64url_table[(n >> 6) & 63];
        /* No padding in URL-safe base64 */
    }
    buf[j] = '\0';
    return j;
}

static int b64url_val(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '-') return 62;
    if (c == '_') return 63;
    return -1;
}

int32_t eshkol_base64url_decode(const char* data, int32_t data_len,
                                  char* buf, int32_t buf_size) {
    if (!data || !buf || data_len < 0 || buf_size <= 0) return -1;
    int out = 0, val = 0, bits = 0;
    for (int i = 0; i < data_len && out < buf_size - 1; i++) {
        int v = b64url_val(data[i]);
        if (v < 0) continue;
        val = (val << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            buf[out++] = (char)((val >> bits) & 0xFF);
        }
    }
    buf[out] = '\0';
    return out;
}

/*******************************************************************************
 * B.6: constant-time-equal
 ******************************************************************************/

int32_t eshkol_constant_time_equal(const char* a, int32_t a_len,
                                     const char* b, int32_t b_len) {
    if (a_len != b_len) return 0;
    if (!a || !b) return 0;
    volatile unsigned char diff = 0;
    for (int i = 0; i < a_len; i++) {
        diff |= (unsigned char)a[i] ^ (unsigned char)b[i];
    }
    return diff == 0 ? 1 : 0;
}

/*******************************************************************************
 * B.6: sha256-file (streaming, no full-file load)
 ******************************************************************************/

#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>

int32_t eshkol_sha256_file(const char* path, char* hex_buf, int32_t buf_size) {
    if (!path || !hex_buf || buf_size < 65) return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    char chunk[65536];
    ssize_t n;
    while ((n = read(fd, chunk, sizeof(chunk))) > 0) {
        CC_SHA256_Update(&ctx, chunk, (CC_LONG)n);
    }
    close(fd);
    if (n < 0) return -1;

    unsigned char hash[32];
    CC_SHA256_Final(hash, &ctx);

    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        hex_buf[i*2]   = hex[(hash[i] >> 4) & 0xf];
        hex_buf[i*2+1] = hex[hash[i] & 0xf];
    }
    hex_buf[64] = '\0';
    return 0;
}
#else
/* Linux: use OpenSSL */
#include <openssl/sha.h>

int32_t eshkol_sha256_file(const char* path, char* hex_buf, int32_t buf_size) {
    if (!path || !hex_buf || buf_size < 65) return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    char chunk[65536];
    ssize_t n;
    while ((n = read(fd, chunk, sizeof(chunk))) > 0) {
        SHA256_Update(&ctx, chunk, (size_t)n);
    }
    close(fd);
    if (n < 0) return -1;

    unsigned char hash[32];
    SHA256_Final(hash, &ctx);

    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        hex_buf[i*2]   = hex[(hash[i] >> 4) & 0xf];
        hex_buf[i*2+1] = hex[hash[i] & 0xf];
    }
    hex_buf[64] = '\0';
    return 0;
}
#endif
