/**
 * @file test_file_io.c
 * @brief Unit tests for the file I/O utilities
 */

#include "core/file_io.h"
#include "core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Test file open and close
 */
static void test_file_open_close(void) {
    printf("Testing file open and close...\n");
    
    // Create a temporary file
    const char* test_file = "test_file.txt";
    
    // Open the file for writing
    File* file = file_open(test_file, FILE_MODE_WRITE);
    assert(file != NULL);
    
    // Close the file
    file_close(file);
    
    // Open the file for reading
    file = file_open(test_file, FILE_MODE_READ);
    assert(file != NULL);
    
    // Close the file
    file_close(file);
    
    // Delete the file
    assert(file_delete(test_file));
    
    printf("PASS: file_open_close\n");
}

/**
 * @brief Test file read and write
 */
static void test_file_read_write(void) {
    printf("Testing file read and write...\n");
    
    // Create a temporary file
    const char* test_file = "test_file.txt";
    
    // Open the file for writing
    File* file = file_open(test_file, FILE_MODE_WRITE);
    assert(file != NULL);
    
    // Write to the file
    const char* test_data = "Hello, world!";
    size_t test_data_len = strlen(test_data);
    size_t bytes_written = file_write(file, test_data, test_data_len);
    assert(bytes_written == test_data_len);
    
    // Close the file
    file_close(file);
    
    // Open the file for reading
    file = file_open(test_file, FILE_MODE_READ);
    assert(file != NULL);
    
    // Read from the file
    char buffer[256];
    size_t bytes_read = file_read(file, buffer, test_data_len);
    assert(bytes_read == test_data_len);
    
    // Verify the data
    buffer[test_data_len] = '\0';
    assert(strcmp(buffer, test_data) == 0);
    
    // Close the file
    file_close(file);
    
    // Delete the file
    assert(file_delete(test_file));
    
    printf("PASS: file_read_write\n");
}

/**
 * @brief Test file seek and tell
 */
static void test_file_seek_tell(void) {
    printf("Testing file seek and tell...\n");
    
    // Create a temporary file
    const char* test_file = "test_file.txt";
    
    // Open the file for writing
    File* file = file_open(test_file, FILE_MODE_WRITE);
    assert(file != NULL);
    
    // Write to the file
    const char* test_data = "Hello, world!";
    size_t test_data_len = strlen(test_data);
    size_t bytes_written = file_write(file, test_data, test_data_len);
    assert(bytes_written == test_data_len);
    
    // Close the file
    file_close(file);
    
    // Open the file for reading
    file = file_open(test_file, FILE_MODE_READ);
    assert(file != NULL);
    
    // Get the current position
    long position = file_tell(file);
    assert(position == 0);
    
    // Seek to the middle
    assert(file_seek(file, 7, FILE_SEEK_SET));
    
    // Get the current position
    position = file_tell(file);
    assert(position == 7);
    
    // Read from the file
    char buffer[256];
    size_t bytes_read = file_read(file, buffer, 6);
    assert(bytes_read == 6);
    
    // Verify the data
    buffer[6] = '\0';
    assert(strcmp(buffer, "world!") == 0);
    
    // Seek to the beginning
    assert(file_seek(file, 0, FILE_SEEK_SET));
    
    // Get the current position
    position = file_tell(file);
    assert(position == 0);
    
    // Close the file
    file_close(file);
    
    // Delete the file
    assert(file_delete(test_file));
    
    printf("PASS: file_seek_tell\n");
}

/**
 * @brief Test file size
 */
static void test_file_size(void) {
    printf("Testing file size...\n");
    
    // Create a temporary file
    const char* test_file = "test_file.txt";
    
    // Open the file for writing
    File* file = file_open(test_file, FILE_MODE_WRITE);
    assert(file != NULL);
    
    // Write to the file
    const char* test_data = "Hello, world!";
    size_t test_data_len = strlen(test_data);
    size_t bytes_written = file_write(file, test_data, test_data_len);
    assert(bytes_written == test_data_len);
    
    // Get the file size
    long size = file_size(file);
    assert(size == (long)test_data_len);
    
    // Close the file
    file_close(file);
    
    // Open the file for reading
    file = file_open(test_file, FILE_MODE_READ);
    assert(file != NULL);
    
    // Get the file size
    size = file_size(file);
    assert(size == (long)test_data_len);
    
    // Close the file
    file_close(file);
    
    // Delete the file
    assert(file_delete(test_file));
    
    printf("PASS: file_size\n");
}

/**
 * @brief Test file exists
 */
static void test_file_exists(void) {
    printf("Testing file exists...\n");
    
    // Create a temporary file
    const char* test_file = "test_file.txt";
    
    // Check if the file exists (it shouldn't)
    assert(!file_exists(test_file));
    
    // Open the file for writing
    File* file = file_open(test_file, FILE_MODE_WRITE);
    assert(file != NULL);
    
    // Close the file
    file_close(file);
    
    // Check if the file exists (it should)
    assert(file_exists(test_file));
    
    // Delete the file
    assert(file_delete(test_file));
    
    // Check if the file exists (it shouldn't)
    assert(!file_exists(test_file));
    
    printf("PASS: file_exists\n");
}

/**
 * @brief Test file rename
 */
static void test_file_rename(void) {
    printf("Testing file rename...\n");
    
    // Create a temporary file
    const char* test_file = "test_file.txt";
    const char* new_file = "new_file.txt";
    
    // Open the file for writing
    File* file = file_open(test_file, FILE_MODE_WRITE);
    assert(file != NULL);
    
    // Write to the file
    const char* test_data = "Hello, world!";
    size_t test_data_len = strlen(test_data);
    size_t bytes_written = file_write(file, test_data, test_data_len);
    assert(bytes_written == test_data_len);
    
    // Close the file
    file_close(file);
    
    // Rename the file
    assert(file_rename(test_file, new_file));
    
    // Check if the old file exists (it shouldn't)
    assert(!file_exists(test_file));
    
    // Check if the new file exists (it should)
    assert(file_exists(new_file));
    
    // Open the new file for reading
    file = file_open(new_file, FILE_MODE_READ);
    assert(file != NULL);
    
    // Read from the file
    char buffer[256];
    size_t bytes_read = file_read(file, buffer, test_data_len);
    assert(bytes_read == test_data_len);
    
    // Verify the data
    buffer[test_data_len] = '\0';
    assert(strcmp(buffer, test_data) == 0);
    
    // Close the file
    file_close(file);
    
    // Delete the file
    assert(file_delete(new_file));
    
    printf("PASS: file_rename\n");
}

/**
 * @brief Test file copy
 */
static void test_file_copy(void) {
    printf("Testing file copy...\n");
    
    // Create a temporary file
    const char* test_file = "test_file.txt";
    const char* copy_file = "copy_file.txt";
    
    // Open the file for writing
    File* file = file_open(test_file, FILE_MODE_WRITE);
    assert(file != NULL);
    
    // Write to the file
    const char* test_data = "Hello, world!";
    size_t test_data_len = strlen(test_data);
    size_t bytes_written = file_write(file, test_data, test_data_len);
    assert(bytes_written == test_data_len);
    
    // Close the file
    file_close(file);
    
    // Copy the file
    assert(file_copy(test_file, copy_file));
    
    // Check if the original file exists (it should)
    assert(file_exists(test_file));
    
    // Check if the copy file exists (it should)
    assert(file_exists(copy_file));
    
    // Open the copy file for reading
    file = file_open(copy_file, FILE_MODE_READ);
    assert(file != NULL);
    
    // Read from the file
    char buffer[256];
    size_t bytes_read = file_read(file, buffer, test_data_len);
    assert(bytes_read == test_data_len);
    
    // Verify the data
    buffer[test_data_len] = '\0';
    assert(strcmp(buffer, test_data) == 0);
    
    // Close the file
    file_close(file);
    
    // Delete the files
    assert(file_delete(test_file));
    assert(file_delete(copy_file));
    
    printf("PASS: file_copy\n");
}

/**
 * @brief Test file read all and write all
 */
static void test_file_read_write_all(void) {
    printf("Testing file read all and write all...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a temporary file
    const char* test_file = "test_file.txt";
    
    // Write to the file
    const char* test_data = "Hello, world!";
    size_t test_data_len = strlen(test_data);
    assert(file_write_all(test_file, test_data, test_data_len));
    
    // Read from the file
    size_t size;
    char* data = file_read_all(arena, test_file, &size);
    assert(data != NULL);
    assert(size == test_data_len);
    
    // Verify the data
    assert(strcmp(data, test_data) == 0);
    
    // Delete the file
    assert(file_delete(test_file));
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: file_read_write_all\n");
}

/**
 * @brief Test file read line and write line
 */
static void test_file_read_write_line(void) {
    printf("Testing file read line and write line...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a temporary file
    const char* test_file = "test_file.txt";
    
    // Open the file for writing
    File* file = file_open(test_file, FILE_MODE_WRITE);
    assert(file != NULL);
    
    // Write lines to the file
    const char* line1 = "Line 1";
    const char* line2 = "Line 2";
    const char* line3 = "Line 3";
    
    assert(file_write_line(file, line1, strlen(line1)));
    assert(file_write_line(file, line2, strlen(line2)));
    assert(file_write_line(file, line3, strlen(line3)));
    
    // Close the file
    file_close(file);
    
    // Open the file for reading
    file = file_open(test_file, FILE_MODE_READ);
    assert(file != NULL);
    
    // Read lines from the file
    size_t size;
    char* line = file_read_line(arena, file, &size);
    assert(line != NULL);
    assert(size == strlen(line1) + 1); // +1 for newline
    assert(strncmp(line, line1, strlen(line1)) == 0);
    
    line = file_read_line(arena, file, &size);
    assert(line != NULL);
    assert(size == strlen(line2) + 1); // +1 for newline
    assert(strncmp(line, line2, strlen(line2)) == 0);
    
    line = file_read_line(arena, file, &size);
    assert(line != NULL);
    assert(size == strlen(line3) + 1); // +1 for newline
    assert(strncmp(line, line3, strlen(line3)) == 0);
    
    // Check for end of file
    line = file_read_line(arena, file, &size);
    assert(line == NULL);
    assert(file_eof(file));
    
    // Close the file
    file_close(file);
    
    // Delete the file
    assert(file_delete(test_file));
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: file_read_write_line\n");
}

/**
 * @brief Test file path utilities
 */
static void test_file_path_utilities(void) {
    printf("Testing file path utilities...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Test file_get_directory
    char* dir = file_get_directory(arena, "/path/to/file.txt");
    assert(dir != NULL);
#ifdef _WIN32
    assert(strcmp(dir, "/path/to") == 0);
#else
    assert(strcmp(dir, "/path/to") == 0);
#endif
    
    // Test file_get_filename
    char* filename = file_get_filename(arena, "/path/to/file.txt");
    assert(filename != NULL);
    assert(strcmp(filename, "file.txt") == 0);
    
    // Test file_get_extension
    char* ext = file_get_extension(arena, "/path/to/file.txt");
    assert(ext != NULL);
    assert(strcmp(ext, ".txt") == 0);
    
    // Test file_join_path
    const char* components[] = {"path", "to", "file.txt"};
    char* path = file_join_path(arena, components, 3);
    assert(path != NULL);
#ifdef _WIN32
    assert(strcmp(path, "path\\to\\file.txt") == 0);
#else
    assert(strcmp(path, "path/to/file.txt") == 0);
#endif
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: file_path_utilities\n");
}

/**
 * @brief Test file directory operations
 */
static void test_file_directory_operations(void) {
    printf("Testing file directory operations...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Create a temporary directory
    const char* test_dir = "test_dir";
    
    // Check if the directory exists (it shouldn't)
    assert(!file_exists(test_dir));
    
    // Create the directory
    assert(file_create_directory(test_dir));
    
    // Check if the directory exists (it should)
    assert(file_exists(test_dir));
    assert(file_is_directory(test_dir));
    
    // Create a file in the directory
    const char* test_file = "test_dir/test_file.txt";
    
    // Open the file for writing
    File* file = file_open(test_file, FILE_MODE_WRITE);
    assert(file != NULL);
    
    // Write to the file
    const char* test_data = "Hello, world!";
    size_t test_data_len = strlen(test_data);
    size_t bytes_written = file_write(file, test_data, test_data_len);
    assert(bytes_written == test_data_len);
    
    // Close the file
    file_close(file);
    
    // Check if the file exists (it should)
    assert(file_exists(test_file));
    
    // Delete the directory (should fail because it's not empty)
    assert(!file_delete_directory(test_dir, false));
    
    // Delete the directory recursively (should succeed)
    assert(file_delete_directory(test_dir, true));
    
    // Check if the directory exists (it shouldn't)
    assert(!file_exists(test_dir));
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: file_directory_operations\n");
}

/**
 * @brief Test file temporary operations
 */
static void test_file_temporary_operations(void) {
    printf("Testing file temporary operations...\n");
    
    // Create an arena
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    // Get the temporary directory
    char* temp_dir = file_get_temporary_directory(arena);
    assert(temp_dir != NULL);
    
    // Create a temporary file
    char* temp_path;
    File* temp_file = file_create_temporary(arena, "test", &temp_path);
    assert(temp_file != NULL);
    assert(temp_path != NULL);
    
    // Write to the temporary file
    const char* test_data = "Hello, world!";
    size_t test_data_len = strlen(test_data);
    size_t bytes_written = file_write(temp_file, test_data, test_data_len);
    assert(bytes_written == test_data_len);
    
    // Close the temporary file
    file_close(temp_file);
    
    // Check if the temporary file exists (it should)
    assert(file_exists(temp_path));
    
    // Delete the temporary file
    assert(file_delete(temp_path));
    
    // Destroy the arena
    arena_destroy(arena);
    
    printf("PASS: file_temporary_operations\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running file I/O tests...\n");
    
    test_file_open_close();
    test_file_read_write();
    test_file_seek_tell();
    test_file_size();
    test_file_exists();
    test_file_rename();
    test_file_copy();
    test_file_read_write_all();
    test_file_read_write_line();
    test_file_path_utilities();
    test_file_directory_operations();
    test_file_temporary_operations();
    
    printf("All file I/O tests passed!\n");
    return 0;
}
