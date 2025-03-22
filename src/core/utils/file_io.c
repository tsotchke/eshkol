/**
 * @file file_io.c
 * @brief Implementation of file I/O utilities
 */

#include "core/file_io.h"
#include "core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#include <io.h>
#define mkdir(path, mode) _mkdir(path)
#define rmdir(path) _rmdir(path)
#define access(path, mode) _access(path, mode)
#define F_OK 0
#define PATH_MAX MAX_PATH
#else
#include <unistd.h>
#include <dirent.h>
#include <limits.h>
#include <libgen.h>
#endif

/**
 * @brief File structure
 */
struct File {
    FILE* handle;     /**< File handle */
    FileMode mode;    /**< File mode */
    char* path;       /**< File path */
};

/**
 * @brief Last file error
 */
static FileError g_last_error = FILE_ERROR_NONE;

/**
 * @brief Set the last file error
 * 
 * @param error The error code
 */
static void set_last_error(FileError error) {
    g_last_error = error;
}

/**
 * @brief Convert errno to FileError
 * 
 * @param err The errno value
 * @return The corresponding FileError
 */
static FileError errno_to_file_error(int err) {
    switch (err) {
        case 0:
            return FILE_ERROR_NONE;
        case ENOENT:
            return FILE_ERROR_NOT_FOUND;
        case EACCES:
        case EPERM:
            return FILE_ERROR_PERMISSION;
        case EEXIST:
            return FILE_ERROR_ALREADY_EXISTS;
        case EIO:
            return FILE_ERROR_IO;
        case EINVAL:
            return FILE_ERROR_INVALID_ARGUMENT;
        case ENOMEM:
            return FILE_ERROR_OUT_OF_MEMORY;
        default:
            return FILE_ERROR_UNKNOWN;
    }
}

/**
 * @brief Convert FileMode to fopen mode string
 * 
 * @param mode The file mode
 * @return The fopen mode string
 */
static const char* file_mode_to_string(FileMode mode) {
    switch (mode) {
        case FILE_MODE_READ:
            return "rb";
        case FILE_MODE_WRITE:
            return "wb";
        case FILE_MODE_APPEND:
            return "ab";
        case FILE_MODE_READ_WRITE:
            return "r+b";
        default:
            return "rb";
    }
}

/**
 * @brief Convert FileSeekOrigin to fseek origin
 * 
 * @param origin The file seek origin
 * @return The fseek origin
 */
static int file_seek_origin_to_int(FileSeekOrigin origin) {
    switch (origin) {
        case FILE_SEEK_SET:
            return SEEK_SET;
        case FILE_SEEK_CUR:
            return SEEK_CUR;
        case FILE_SEEK_END:
            return SEEK_END;
        default:
            return SEEK_SET;
    }
}

FileError file_get_last_error(void) {
    return g_last_error;
}

const char* file_error_to_string(FileError error) {
    switch (error) {
        case FILE_ERROR_NONE:
            return "No error";
        case FILE_ERROR_NOT_FOUND:
            return "File not found";
        case FILE_ERROR_PERMISSION:
            return "Permission denied";
        case FILE_ERROR_ALREADY_EXISTS:
            return "File already exists";
        case FILE_ERROR_IO:
            return "I/O error";
        case FILE_ERROR_INVALID_HANDLE:
            return "Invalid file handle";
        case FILE_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case FILE_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case FILE_ERROR_UNKNOWN:
            return "Unknown error";
        default:
            return "Unknown error";
    }
}

File* file_open(const char* path, FileMode mode) {
    assert(path != NULL);
    
    // Open the file
    FILE* handle = fopen(path, file_mode_to_string(mode));
    if (!handle) {
        set_last_error(errno_to_file_error(errno));
        return NULL;
    }
    
    // Allocate the file structure
    File* file = malloc(sizeof(File));
    if (!file) {
        fclose(handle);
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    // Copy the path
    file->path = strdup(path);
    if (!file->path) {
        fclose(handle);
        free(file);
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    // Initialize the file structure
    file->handle = handle;
    file->mode = mode;
    
    set_last_error(FILE_ERROR_NONE);
    return file;
}

void file_close(File* file) {
    if (!file) {
        set_last_error(FILE_ERROR_INVALID_HANDLE);
        return;
    }
    
    // Close the file
    if (file->handle) {
        fclose(file->handle);
    }
    
    // Free the path
    if (file->path) {
        free(file->path);
    }
    
    // Free the file structure
    free(file);
    
    set_last_error(FILE_ERROR_NONE);
}

size_t file_read(File* file, void* buffer, size_t size) {
    assert(buffer != NULL);
    
    if (!file || !file->handle) {
        set_last_error(FILE_ERROR_INVALID_HANDLE);
        return (size_t)-1;
    }
    
    // Read from the file
    size_t bytes_read = fread(buffer, 1, size, file->handle);
    if (bytes_read != size && !feof(file->handle)) {
        set_last_error(errno_to_file_error(errno));
        return (size_t)-1;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return bytes_read;
}

size_t file_write(File* file, const void* buffer, size_t size) {
    assert(buffer != NULL);
    
    if (!file || !file->handle) {
        set_last_error(FILE_ERROR_INVALID_HANDLE);
        return (size_t)-1;
    }
    
    // Write to the file
    size_t bytes_written = fwrite(buffer, 1, size, file->handle);
    if (bytes_written != size) {
        set_last_error(errno_to_file_error(errno));
        return (size_t)-1;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return bytes_written;
}

bool file_seek(File* file, long offset, FileSeekOrigin origin) {
    if (!file || !file->handle) {
        set_last_error(FILE_ERROR_INVALID_HANDLE);
        return false;
    }
    
    // Seek in the file
    int result = fseek(file->handle, offset, file_seek_origin_to_int(origin));
    if (result != 0) {
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return true;
}

long file_tell(File* file) {
    if (!file || !file->handle) {
        set_last_error(FILE_ERROR_INVALID_HANDLE);
        return -1;
    }
    
    // Get the current position
    long position = ftell(file->handle);
    if (position < 0) {
        set_last_error(errno_to_file_error(errno));
        return -1;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return position;
}

bool file_eof(File* file) {
    if (!file || !file->handle) {
        set_last_error(FILE_ERROR_INVALID_HANDLE);
        return true;
    }
    
    // Check for end of file
    int result = feof(file->handle);
    
    set_last_error(FILE_ERROR_NONE);
    return result != 0;
}

bool file_flush(File* file) {
    if (!file || !file->handle) {
        set_last_error(FILE_ERROR_INVALID_HANDLE);
        return false;
    }
    
    // Flush the file
    int result = fflush(file->handle);
    if (result != 0) {
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return true;
}

long file_size(File* file) {
    if (!file || !file->handle) {
        set_last_error(FILE_ERROR_INVALID_HANDLE);
        return -1;
    }
    
    // Get the current position
    long current_position = ftell(file->handle);
    if (current_position < 0) {
        set_last_error(errno_to_file_error(errno));
        return -1;
    }
    
    // Seek to the end
    if (fseek(file->handle, 0, SEEK_END) != 0) {
        set_last_error(errno_to_file_error(errno));
        return -1;
    }
    
    // Get the size
    long size = ftell(file->handle);
    if (size < 0) {
        set_last_error(errno_to_file_error(errno));
        return -1;
    }
    
    // Seek back to the original position
    if (fseek(file->handle, current_position, SEEK_SET) != 0) {
        set_last_error(errno_to_file_error(errno));
        return -1;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return size;
}

bool file_exists(const char* path) {
    assert(path != NULL);
    
    // Check if the file exists
    if (access(path, F_OK) != 0) {
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return true;
}

bool file_delete(const char* path) {
    assert(path != NULL);
    
    // Delete the file
    if (remove(path) != 0) {
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return true;
}

bool file_rename(const char* old_path, const char* new_path) {
    assert(old_path != NULL);
    assert(new_path != NULL);
    
    // Rename the file
    if (rename(old_path, new_path) != 0) {
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return true;
}

bool file_copy(const char* src_path, const char* dst_path) {
    assert(src_path != NULL);
    assert(dst_path != NULL);
    
    // Open the source file
    FILE* src_file = fopen(src_path, "rb");
    if (!src_file) {
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    // Open the destination file
    FILE* dst_file = fopen(dst_path, "wb");
    if (!dst_file) {
        fclose(src_file);
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    // Copy the file
    char buffer[4096];
    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), src_file)) > 0) {
        if (fwrite(buffer, 1, bytes_read, dst_file) != bytes_read) {
            fclose(src_file);
            fclose(dst_file);
            set_last_error(errno_to_file_error(errno));
            return false;
        }
    }
    
    // Check for errors
    if (ferror(src_file)) {
        fclose(src_file);
        fclose(dst_file);
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    // Close the files
    fclose(src_file);
    fclose(dst_file);
    
    set_last_error(FILE_ERROR_NONE);
    return true;
}

char* file_read_all(Arena* arena, const char* path, size_t* size) {
    assert(arena != NULL);
    assert(path != NULL);
    
    // Open the file
    File* file = file_open(path, FILE_MODE_READ);
    if (!file) {
        return NULL;
    }
    
    // Get the file size
    long file_size_value = file_size(file);
    if (file_size_value < 0) {
        file_close(file);
        return NULL;
    }
    
    // Allocate the buffer
    char* buffer = arena_alloc(arena, file_size_value + 1);
    if (!buffer) {
        file_close(file);
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    // Read the file
    size_t bytes_read = file_read(file, buffer, file_size_value);
    if (bytes_read != (size_t)file_size_value) {
        file_close(file);
        return NULL;
    }
    
    // Null-terminate the buffer
    buffer[file_size_value] = '\0';
    
    // Close the file
    file_close(file);
    
    // Set the size
    if (size) {
        *size = file_size_value;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return buffer;
}

bool file_write_all(const char* path, const void* buffer, size_t size) {
    assert(path != NULL);
    assert(buffer != NULL);
    
    // Open the file
    File* file = file_open(path, FILE_MODE_WRITE);
    if (!file) {
        return false;
    }
    
    // Write the file
    size_t bytes_written = file_write(file, buffer, size);
    if (bytes_written != size) {
        file_close(file);
        return false;
    }
    
    // Close the file
    file_close(file);
    
    set_last_error(FILE_ERROR_NONE);
    return true;
}

char* file_read_line(Arena* arena, File* file, size_t* size) {
    assert(arena != NULL);
    
    if (!file || !file->handle) {
        set_last_error(FILE_ERROR_INVALID_HANDLE);
        return NULL;
    }
    
    // Check for end of file
    if (feof(file->handle)) {
        set_last_error(FILE_ERROR_NONE);
        return NULL;
    }
    
    // Read the line
    char buffer[4096];
    char* line = NULL;
    size_t line_size = 0;
    
    while (fgets(buffer, sizeof(buffer), file->handle)) {
        size_t buffer_len = strlen(buffer);
        
        // Allocate or resize the line buffer
        if (!line) {
            line = arena_alloc(arena, buffer_len + 1);
            if (!line) {
                set_last_error(FILE_ERROR_OUT_OF_MEMORY);
                return NULL;
            }
            memcpy(line, buffer, buffer_len + 1);
        } else {
            char* new_line = arena_alloc(arena, line_size + buffer_len + 1);
            if (!new_line) {
                set_last_error(FILE_ERROR_OUT_OF_MEMORY);
                return NULL;
            }
            memcpy(new_line, line, line_size);
            memcpy(new_line + line_size, buffer, buffer_len + 1);
            line = new_line;
        }
        
        line_size += buffer_len;
        
        // Check for newline
        if (buffer[buffer_len - 1] == '\n') {
            break;
        }
    }
    
    // Check for errors
    if (ferror(file->handle)) {
        set_last_error(errno_to_file_error(errno));
        return NULL;
    }
    
    // Set the size
    if (size) {
        *size = line_size;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return line;
}

bool file_write_line(File* file, const char* line, size_t size) {
    assert(line != NULL);
    
    if (!file || !file->handle) {
        set_last_error(FILE_ERROR_INVALID_HANDLE);
        return false;
    }
    
    // Write the line
    size_t bytes_written = fwrite(line, 1, size, file->handle);
    if (bytes_written != size) {
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    // Write a newline
    if (fputc('\n', file->handle) == EOF) {
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return true;
}

char* file_get_absolute_path(Arena* arena, const char* path) {
    assert(arena != NULL);
    assert(path != NULL);
    
    // Get the absolute path
    char absolute_path[PATH_MAX];
    
#ifdef _WIN32
    if (_fullpath(absolute_path, path, PATH_MAX) == NULL) {
        set_last_error(errno_to_file_error(errno));
        return NULL;
    }
#else
    if (realpath(path, absolute_path) == NULL) {
        set_last_error(errno_to_file_error(errno));
        return NULL;
    }
#endif
    
    // Copy the path
    size_t path_len = strlen(absolute_path);
    char* result = arena_alloc(arena, path_len + 1);
    if (!result) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    memcpy(result, absolute_path, path_len + 1);
    
    set_last_error(FILE_ERROR_NONE);
    return result;
}

char* file_get_directory(Arena* arena, const char* path) {
    assert(arena != NULL);
    assert(path != NULL);
    
    // Copy the path
    size_t path_len = strlen(path);
    char* path_copy = arena_alloc(arena, path_len + 1);
    if (!path_copy) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    memcpy(path_copy, path, path_len + 1);
    
    // Get the directory
#ifdef _WIN32
    char drive[_MAX_DRIVE];
    char dir[_MAX_DIR];
    _splitpath(path_copy, drive, dir, NULL, NULL);
    
    // Combine drive and directory
    size_t drive_len = strlen(drive);
    size_t dir_len = strlen(dir);
    char* result = arena_alloc(arena, drive_len + dir_len + 1);
    if (!result) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    memcpy(result, drive, drive_len);
    memcpy(result + drive_len, dir, dir_len + 1);
#else
    char* dir = dirname(path_copy);
    
    // Copy the directory
    size_t dir_len = strlen(dir);
    char* result = arena_alloc(arena, dir_len + 1);
    if (!result) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    memcpy(result, dir, dir_len + 1);
#endif
    
    set_last_error(FILE_ERROR_NONE);
    return result;
}

char* file_get_filename(Arena* arena, const char* path) {
    assert(arena != NULL);
    assert(path != NULL);
    
    // Copy the path
    size_t path_len = strlen(path);
    char* path_copy = arena_alloc(arena, path_len + 1);
    if (!path_copy) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    memcpy(path_copy, path, path_len + 1);
    
    // Get the filename
#ifdef _WIN32
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];
    _splitpath(path_copy, NULL, NULL, fname, ext);
    
    // Combine filename and extension
    size_t fname_len = strlen(fname);
    size_t ext_len = strlen(ext);
    char* result = arena_alloc(arena, fname_len + ext_len + 1);
    if (!result) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    memcpy(result, fname, fname_len);
    memcpy(result + fname_len, ext, ext_len + 1);
#else
    char* filename = basename(path_copy);
    
    // Copy the filename
    size_t filename_len = strlen(filename);
    char* result = arena_alloc(arena, filename_len + 1);
    if (!result) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    memcpy(result, filename, filename_len + 1);
#endif
    
    set_last_error(FILE_ERROR_NONE);
    return result;
}

char* file_get_extension(Arena* arena, const char* path) {
    assert(arena != NULL);
    assert(path != NULL);
    
    // Find the last dot
    const char* dot = strrchr(path, '.');
    if (!dot || dot == path) {
        // No extension
        char* result = arena_alloc(arena, 1);
        if (!result) {
            set_last_error(FILE_ERROR_OUT_OF_MEMORY);
            return NULL;
        }
        
        result[0] = '\0';
        
        set_last_error(FILE_ERROR_NONE);
        return result;
    }
    
    // Copy the extension
    size_t ext_len = strlen(dot);
    char* result = arena_alloc(arena, ext_len + 1);
    if (!result) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    memcpy(result, dot, ext_len + 1);
    
    set_last_error(FILE_ERROR_NONE);
    return result;
}

char* file_join_path(Arena* arena, const char** components, size_t count) {
    assert(arena != NULL);
    assert(components != NULL);
    assert(count > 0);
    
    // Calculate the total length
    size_t total_len = 0;
    for (size_t i = 0; i < count; i++) {
        total_len += strlen(components[i]);
    }
    
    // Add space for separators
    total_len += count - 1;
    
    // Allocate the result
    char* result = arena_alloc(arena, total_len + 1);
    if (!result) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    // Copy the components
    char* p = result;
    for (size_t i = 0; i < count; i++) {
        size_t len = strlen(components[i]);
        memcpy(p, components[i], len);
        p += len;
        
        // Add separator
        if (i < count - 1) {
#ifdef _WIN32
            *p++ = '\\';
#else
            *p++ = '/';
#endif
        }
    }
    
    // Null-terminate
    *p = '\0';
    
    set_last_error(FILE_ERROR_NONE);
    return result;
}

char* file_normalize_path(Arena* arena, const char* path) {
    assert(arena != NULL);
    assert(path != NULL);
    
    // Get the absolute path
    char* absolute_path = file_get_absolute_path(arena, path);
    if (!absolute_path) {
        return NULL;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return absolute_path;
}

bool file_create_directory(const char* path) {
    assert(path != NULL);
    
    // Create the directory
#ifdef _WIN32
    if (_mkdir(path) != 0) {
#else
    if (mkdir(path, 0777) != 0) {
#endif
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return true;
}

bool file_create_directories(const char* path) {
    assert(path != NULL);
    
    // Copy the path
    char* path_copy = strdup(path);
    if (!path_copy) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return false;
    }
    
    // Create directories
    char* p = path_copy;
    
    // Skip leading slashes
    while (*p == '/' || *p == '\\') {
        p++;
    }
    
    // Create each directory
    while (*p) {
        // Find the next slash
        char* slash = p;
        while (*slash && *slash != '/' && *slash != '\\') {
            slash++;
        }
        
        // Save the slash
        char save = *slash;
        *slash = '\0';
        
        // Create the directory
        if (!file_exists(path_copy)) {
            if (!file_create_directory(path_copy)) {
                free(path_copy);
                return false;
            }
        }
        
        // Restore the slash
        *slash = save;
        
        // Skip to the next component
        if (!*slash) {
            break;
        }
        
        p = slash + 1;
        
        // Skip multiple slashes
        while (*p == '/' || *p == '\\') {
            p++;
        }
    }
    
    free(path_copy);
    
    set_last_error(FILE_ERROR_NONE);
    return true;
}

bool file_delete_directory(const char* path, bool recursive) {
    assert(path != NULL);
    
    if (recursive) {
#ifdef _WIN32
        // Windows implementation
        WIN32_FIND_DATA find_data;
        HANDLE find_handle;
        char search_path[MAX_PATH];
        
        // Create the search path
        snprintf(search_path, sizeof(search_path), "%s\\*", path);
        
        // Find the first file
        find_handle = FindFirstFile(search_path, &find_data);
        if (find_handle == INVALID_HANDLE_VALUE) {
            set_last_error(errno_to_file_error(GetLastError()));
            return false;
        }
        
        // Delete all files and subdirectories
        do {
            // Skip . and ..
            if (strcmp(find_data.cFileName, ".") == 0 ||
                strcmp(find_data.cFileName, "..") == 0) {
                continue;
            }
            
            // Create the full path
            char full_path[MAX_PATH];
            snprintf(full_path, sizeof(full_path), "%s\\%s", path, find_data.cFileName);
            
            // Check if it's a directory
            if (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                // Recursively delete the directory
                if (!file_delete_directory(full_path, true)) {
                    FindClose(find_handle);
                    return false;
                }
            } else {
                // Delete the file
                if (!DeleteFile(full_path)) {
                    FindClose(find_handle);
                    set_last_error(errno_to_file_error(GetLastError()));
                    return false;
                }
            }
        } while (FindNextFile(find_handle, &find_data));
        
        // Close the find handle
        FindClose(find_handle);
        
        // Delete the directory
        if (!RemoveDirectory(path)) {
            set_last_error(errno_to_file_error(GetLastError()));
            return false;
        }
#else
        // Unix implementation
        DIR* dir = opendir(path);
        if (!dir) {
            set_last_error(errno_to_file_error(errno));
            return false;
        }
        
        // Delete all files and subdirectories
        struct dirent* entry;
        while ((entry = readdir(dir))) {
            // Skip . and ..
            if (strcmp(entry->d_name, ".") == 0 ||
                strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            
            // Create the full path
            char full_path[PATH_MAX];
            snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);
            
            // Check if it's a directory
            struct stat st;
            if (stat(full_path, &st) != 0) {
                closedir(dir);
                set_last_error(errno_to_file_error(errno));
                return false;
            }
            
            if (S_ISDIR(st.st_mode)) {
                // Recursively delete the directory
                if (!file_delete_directory(full_path, true)) {
                    closedir(dir);
                    return false;
                }
            } else {
                // Delete the file
                if (unlink(full_path) != 0) {
                    closedir(dir);
                    set_last_error(errno_to_file_error(errno));
                    return false;
                }
            }
        }
        
        // Close the directory
        closedir(dir);
        
        // Delete the directory
        if (rmdir(path) != 0) {
            set_last_error(errno_to_file_error(errno));
            return false;
        }
#endif
    } else {
        // Delete the directory
#ifdef _WIN32
        if (!RemoveDirectory(path)) {
            set_last_error(errno_to_file_error(GetLastError()));
            return false;
        }
#else
        if (rmdir(path) != 0) {
            set_last_error(errno_to_file_error(errno));
            return false;
        }
#endif
    }
    
    set_last_error(FILE_ERROR_NONE);
    return true;
}

bool file_is_directory(const char* path) {
    assert(path != NULL);
    
    // Check if the path is a directory
    struct stat st;
    if (stat(path, &st) != 0) {
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return S_ISDIR(st.st_mode);
}

char* file_get_current_directory(Arena* arena) {
    assert(arena != NULL);
    
    // Get the current directory
    char current_dir[PATH_MAX];
    
#ifdef _WIN32
    if (_getcwd(current_dir, PATH_MAX) == NULL) {
#else
    if (getcwd(current_dir, PATH_MAX) == NULL) {
#endif
        set_last_error(errno_to_file_error(errno));
        return NULL;
    }
    
    // Copy the directory
    size_t dir_len = strlen(current_dir);
    char* result = arena_alloc(arena, dir_len + 1);
    if (!result) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    memcpy(result, current_dir, dir_len + 1);
    
    set_last_error(FILE_ERROR_NONE);
    return result;
}

bool file_set_current_directory(const char* path) {
    assert(path != NULL);
    
    // Set the current directory
#ifdef _WIN32
    if (_chdir(path) != 0) {
#else
    if (chdir(path) != 0) {
#endif
        set_last_error(errno_to_file_error(errno));
        return false;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return true;
}

char* file_get_temporary_directory(Arena* arena) {
    assert(arena != NULL);
    
    // Get the temporary directory
    const char* temp_dir = NULL;
    
#ifdef _WIN32
    temp_dir = getenv("TEMP");
    if (!temp_dir) {
        temp_dir = getenv("TMP");
    }
#else
    temp_dir = getenv("TMPDIR");
    if (!temp_dir) {
        temp_dir = "/tmp";
    }
#endif
    
    if (!temp_dir) {
        set_last_error(FILE_ERROR_UNKNOWN);
        return NULL;
    }
    
    // Copy the directory
    size_t dir_len = strlen(temp_dir);
    char* result = arena_alloc(arena, dir_len + 1);
    if (!result) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    memcpy(result, temp_dir, dir_len + 1);
    
    set_last_error(FILE_ERROR_NONE);
    return result;
}

File* file_create_temporary(Arena* arena, const char* prefix, char** path) {
    assert(arena != NULL);
    assert(prefix != NULL);
    assert(path != NULL);
    
    // Get the temporary directory
    char* temp_dir = file_get_temporary_directory(arena);
    if (!temp_dir) {
        return NULL;
    }
    
    // Create a temporary file
#ifdef _WIN32
    char temp_path[MAX_PATH];
    char temp_filename[MAX_PATH];
    
    // Get the temporary path
    if (GetTempPath(MAX_PATH, temp_path) == 0) {
        set_last_error(errno_to_file_error(GetLastError()));
        return NULL;
    }
    
    // Get a temporary filename
    if (GetTempFileName(temp_path, prefix, 0, temp_filename) == 0) {
        set_last_error(errno_to_file_error(GetLastError()));
        return NULL;
    }
    
    // Copy the path
    size_t path_len = strlen(temp_filename);
    *path = arena_alloc(arena, path_len + 1);
    if (!*path) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    memcpy(*path, temp_filename, path_len + 1);
#else
    // Create a template
    size_t prefix_len = strlen(prefix);
    size_t temp_dir_len = strlen(temp_dir);
    size_t template_len = temp_dir_len + 1 + prefix_len + 6 + 1;
    char* template = arena_alloc(arena, template_len);
    if (!template) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    snprintf(template, template_len, "%s/%sXXXXXX", temp_dir, prefix);
    
    // Create the temporary file
    int fd = mkstemp(template);
    if (fd < 0) {
        set_last_error(errno_to_file_error(errno));
        return NULL;
    }
    
    // Close the file descriptor
    close(fd);
    
    // Copy the path
    size_t path_len = strlen(template);
    *path = arena_alloc(arena, path_len + 1);
    if (!*path) {
        set_last_error(FILE_ERROR_OUT_OF_MEMORY);
        return NULL;
    }
    
    memcpy(*path, template, path_len + 1);
#endif
    
    // Open the file
    File* file = file_open(*path, FILE_MODE_READ_WRITE);
    if (!file) {
        return NULL;
    }
    
    set_last_error(FILE_ERROR_NONE);
    return file;
}
