/**
 * @file file_io.h
 * @brief File I/O utilities for Eshkol
 * 
 * This file defines the file I/O interface for Eshkol,
 * which provides utilities for reading and writing files.
 */

#ifndef ESHKOL_FILE_IO_H
#define ESHKOL_FILE_IO_H

#include "core/memory.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief File handle
 */
typedef struct File File;

/**
 * @brief File open mode
 */
typedef enum {
    FILE_MODE_READ,       /**< Open for reading */
    FILE_MODE_WRITE,      /**< Open for writing (create or truncate) */
    FILE_MODE_APPEND,     /**< Open for appending */
    FILE_MODE_READ_WRITE, /**< Open for reading and writing */
} FileMode;

/**
 * @brief File seek origin
 */
typedef enum {
    FILE_SEEK_SET, /**< Seek from the beginning of the file */
    FILE_SEEK_CUR, /**< Seek from the current position */
    FILE_SEEK_END, /**< Seek from the end of the file */
} FileSeekOrigin;

/**
 * @brief File error codes
 */
typedef enum {
    FILE_ERROR_NONE,           /**< No error */
    FILE_ERROR_NOT_FOUND,      /**< File not found */
    FILE_ERROR_PERMISSION,     /**< Permission denied */
    FILE_ERROR_ALREADY_EXISTS, /**< File already exists */
    FILE_ERROR_IO,             /**< I/O error */
    FILE_ERROR_INVALID_HANDLE, /**< Invalid file handle */
    FILE_ERROR_INVALID_ARGUMENT, /**< Invalid argument */
    FILE_ERROR_OUT_OF_MEMORY,  /**< Out of memory */
    FILE_ERROR_UNKNOWN,        /**< Unknown error */
} FileError;

/**
 * @brief Get the last file error
 * 
 * @return The last file error
 */
FileError file_get_last_error(void);

/**
 * @brief Get a string description of a file error
 * 
 * @param error The file error
 * @return A string description of the error
 */
const char* file_error_to_string(FileError error);

/**
 * @brief Open a file
 * 
 * @param path Path to the file
 * @param mode File open mode
 * @return A file handle, or NULL on failure
 */
File* file_open(const char* path, FileMode mode);

/**
 * @brief Close a file
 * 
 * @param file The file handle
 */
void file_close(File* file);

/**
 * @brief Read from a file
 * 
 * @param file The file handle
 * @param buffer Buffer to read into
 * @param size Size of the buffer in bytes
 * @return Number of bytes read, or -1 on failure
 */
size_t file_read(File* file, void* buffer, size_t size);

/**
 * @brief Write to a file
 * 
 * @param file The file handle
 * @param buffer Buffer to write from
 * @param size Size of the buffer in bytes
 * @return Number of bytes written, or -1 on failure
 */
size_t file_write(File* file, const void* buffer, size_t size);

/**
 * @brief Seek to a position in a file
 * 
 * @param file The file handle
 * @param offset Offset in bytes
 * @param origin Seek origin
 * @return true if successful, false otherwise
 */
bool file_seek(File* file, long offset, FileSeekOrigin origin);

/**
 * @brief Get the current position in a file
 * 
 * @param file The file handle
 * @return Current position in bytes, or -1 on failure
 */
long file_tell(File* file);

/**
 * @brief Check if the end of file has been reached
 * 
 * @param file The file handle
 * @return true if end of file, false otherwise
 */
bool file_eof(File* file);

/**
 * @brief Flush file buffers
 * 
 * @param file The file handle
 * @return true if successful, false otherwise
 */
bool file_flush(File* file);

/**
 * @brief Get the size of a file
 * 
 * @param file The file handle
 * @return Size of the file in bytes, or -1 on failure
 */
long file_size(File* file);

/**
 * @brief Check if a file exists
 * 
 * @param path Path to the file
 * @return true if the file exists, false otherwise
 */
bool file_exists(const char* path);

/**
 * @brief Delete a file
 * 
 * @param path Path to the file
 * @return true if successful, false otherwise
 */
bool file_delete(const char* path);

/**
 * @brief Rename a file
 * 
 * @param old_path Old path
 * @param new_path New path
 * @return true if successful, false otherwise
 */
bool file_rename(const char* old_path, const char* new_path);

/**
 * @brief Copy a file
 * 
 * @param src_path Source path
 * @param dst_path Destination path
 * @return true if successful, false otherwise
 */
bool file_copy(const char* src_path, const char* dst_path);

/**
 * @brief Read an entire file into memory
 * 
 * @param arena Arena to allocate from
 * @param path Path to the file
 * @param size Pointer to store the size of the file
 * @return Buffer containing the file contents, or NULL on failure
 */
char* file_read_all(Arena* arena, const char* path, size_t* size);

/**
 * @brief Write a buffer to a file
 * 
 * @param path Path to the file
 * @param buffer Buffer to write
 * @param size Size of the buffer in bytes
 * @return true if successful, false otherwise
 */
bool file_write_all(const char* path, const void* buffer, size_t size);

/**
 * @brief Read a line from a file
 * 
 * @param arena Arena to allocate from
 * @param file The file handle
 * @param size Pointer to store the size of the line
 * @return Buffer containing the line, or NULL on failure or end of file
 */
char* file_read_line(Arena* arena, File* file, size_t* size);

/**
 * @brief Write a line to a file
 * 
 * @param file The file handle
 * @param line Line to write
 * @param size Size of the line in bytes
 * @return true if successful, false otherwise
 */
bool file_write_line(File* file, const char* line, size_t size);

/**
 * @brief Get the absolute path of a file
 * 
 * @param arena Arena to allocate from
 * @param path Path to the file
 * @return Absolute path, or NULL on failure
 */
char* file_get_absolute_path(Arena* arena, const char* path);

/**
 * @brief Get the directory part of a path
 * 
 * @param arena Arena to allocate from
 * @param path Path to the file
 * @return Directory part, or NULL on failure
 */
char* file_get_directory(Arena* arena, const char* path);

/**
 * @brief Get the filename part of a path
 * 
 * @param arena Arena to allocate from
 * @param path Path to the file
 * @return Filename part, or NULL on failure
 */
char* file_get_filename(Arena* arena, const char* path);

/**
 * @brief Get the extension part of a path
 * 
 * @param arena Arena to allocate from
 * @param path Path to the file
 * @return Extension part, or NULL on failure
 */
char* file_get_extension(Arena* arena, const char* path);

/**
 * @brief Join path components
 * 
 * @param arena Arena to allocate from
 * @param components Array of path components
 * @param count Number of components
 * @return Joined path, or NULL on failure
 */
char* file_join_path(Arena* arena, const char** components, size_t count);

/**
 * @brief Normalize a path
 * 
 * @param arena Arena to allocate from
 * @param path Path to normalize
 * @return Normalized path, or NULL on failure
 */
char* file_normalize_path(Arena* arena, const char* path);

/**
 * @brief Create a directory
 * 
 * @param path Path to the directory
 * @return true if successful, false otherwise
 */
bool file_create_directory(const char* path);

/**
 * @brief Create directories recursively
 * 
 * @param path Path to the directory
 * @return true if successful, false otherwise
 */
bool file_create_directories(const char* path);

/**
 * @brief Delete a directory
 * 
 * @param path Path to the directory
 * @param recursive Whether to delete recursively
 * @return true if successful, false otherwise
 */
bool file_delete_directory(const char* path, bool recursive);

/**
 * @brief Check if a path is a directory
 * 
 * @param path Path to check
 * @return true if the path is a directory, false otherwise
 */
bool file_is_directory(const char* path);

/**
 * @brief Get the current working directory
 * 
 * @param arena Arena to allocate from
 * @return Current working directory, or NULL on failure
 */
char* file_get_current_directory(Arena* arena);

/**
 * @brief Set the current working directory
 * 
 * @param path Path to the directory
 * @return true if successful, false otherwise
 */
bool file_set_current_directory(const char* path);

/**
 * @brief Get the temporary directory
 * 
 * @param arena Arena to allocate from
 * @return Temporary directory, or NULL on failure
 */
char* file_get_temporary_directory(Arena* arena);

/**
 * @brief Create a temporary file
 * 
 * @param arena Arena to allocate from
 * @param prefix Prefix for the filename
 * @param path Pointer to store the path of the temporary file
 * @return A file handle, or NULL on failure
 */
File* file_create_temporary(Arena* arena, const char* prefix, char** path);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_FILE_IO_H */
