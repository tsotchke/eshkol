#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/memory.h"
#include "core/string_table.h"
#include "core/diagnostics.h"
#include "frontend/lexer/lexer.h"
#include "frontend/parser/parser.h"
#include "frontend/type_inference/type_inference.h"
#include "frontend/type_inference/analysis.h"
#include "backend/codegen.h"
#include "backend/codegen/debug.h"

// Forward declarations of functions we'll implement later
static char* read_file(const char* filename, size_t* length);

/**
 * Print usage information
 * 
 * @param program_name The name of the program
 */
static void print_usage(const char* program_name) {
    fprintf(stderr, "Usage: %s [options] <input.esk> [output.c]\n", program_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -v, --verbose         Enable verbose output\n");
    fprintf(stderr, "  -d, --debug           Enable debug output (implies verbose)\n");
    fprintf(stderr, "  -h, --help            Display this help message\n");
    fprintf(stderr, "  --dump-ast            Dump AST visualization\n");
    fprintf(stderr, "  --format=<fmt>        Output format for AST (dot, mermaid)\n");
    fprintf(stderr, "  --analyze-types       Analyze type inference\n");
    fprintf(stderr, "  --detail=<level>      Detail level for type analysis (basic, detailed, verbose)\n");
    fprintf(stderr, "  --debug-codegen       Debug code generation\n");
    fprintf(stderr, "  --stage=<stage>       Code generation stage to debug (ast, ir, c-code, all)\n");
    fprintf(stderr, "  --profile=<phase>     Profile compilation phase (can be used multiple times)\n");
    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "  <input.esk>     Input Eshkol source file\n");
    fprintf(stderr, "  [output.c]      Optional output C file (if not provided, compiles and runs)\n");
}

/**
 * Eshkol Compiler - Main Entry Point
 * 
 * This is the main entry point for the Eshkol compiler.
 * It handles command-line arguments and initiates the compilation process.
 */
int main(int argc, char** argv) {
    printf("Eshkol Compiler v0.1.0\n");
    
    // Parse command line arguments
    bool verbose_mode = true;      // Enable verbose mode by default
    bool debug_mode = true;        // Enable debug mode by default
    bool dump_ast = false;         // AST visualization
    const char* ast_format = NULL; // AST output format
    bool analyze_types = false;    // Type analysis
    const char* detail_level = "basic"; // Type analysis detail
    bool debug_codegen = false;    // Debug code generation
    const char* debug_stage = "all"; // Code gen stage to debug
    bool profile_mode = false;     // Profile compilation
    const char* input_file = NULL;
    const char* output_file = NULL;
    
    // Skip program name
    int arg_index = 1;
    
    // Parse options
    while (arg_index < argc && argv[arg_index][0] == '-') {
        const char* arg = argv[arg_index];
        
        if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            verbose_mode = true;
        } else if (strcmp(arg, "-d") == 0 || strcmp(arg, "--debug") == 0) {
            debug_mode = true;
            verbose_mode = true; // Debug implies verbose
        } else if (strcmp(arg, "--dump-ast") == 0) {
            dump_ast = true;
        } else if (strncmp(arg, "--format=", 9) == 0) {
            ast_format = arg + 9;
            if (strcmp(ast_format, "dot") != 0 && strcmp(ast_format, "mermaid") != 0) {
                fprintf(stderr, "Error: Invalid format '%s'. Must be 'dot' or 'mermaid'\n", ast_format);
                return 1;
            }
        } else if (strcmp(arg, "--analyze-types") == 0) {
            analyze_types = true;
        } else if (strncmp(arg, "--detail=", 9) == 0) {
            detail_level = arg + 9;
            if (strcmp(detail_level, "basic") != 0 && 
                strcmp(detail_level, "detailed") != 0 && 
                strcmp(detail_level, "verbose") != 0) {
                fprintf(stderr, "Error: Invalid detail level '%s'. Must be 'basic', 'detailed', or 'verbose'\n", detail_level);
                return 1;
            }
        } else if (strcmp(arg, "--debug-codegen") == 0) {
            debug_codegen = true;
        } else if (strncmp(arg, "--stage=", 8) == 0) {
            debug_stage = arg + 8;
            if (strcmp(debug_stage, "ast") != 0 && 
                strcmp(debug_stage, "ir") != 0 && 
                strcmp(debug_stage, "c-code") != 0 && 
                strcmp(debug_stage, "all") != 0) {
                fprintf(stderr, "Error: Invalid stage '%s'. Must be 'ast', 'ir', 'c-code', or 'all'\n", debug_stage);
                return 1;
            }
        } else if (strncmp(arg, "--profile=", 10) == 0) {
            const char* phase = arg + 10;
            if (strcmp(phase, "lexing") != 0 && 
                strcmp(phase, "parsing") != 0 && 
                strcmp(phase, "type-inference") != 0 && 
                strcmp(phase, "codegen") != 0) {
                fprintf(stderr, "Error: Invalid phase '%s'. Must be 'lexing', 'parsing', 'type-inference', or 'codegen'\n", phase);
                return 1;
            }
            profile_mode = true;
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: Unknown option '%s'\n", arg);
            print_usage(argv[0]);
            return 1;
        }
        
        arg_index++;
    }
    
    // Check if we have enough arguments
    if (arg_index >= argc) {
        fprintf(stderr, "Error: No input file specified\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Get input filename
    input_file = argv[arg_index++];
    
        // Get output filename if provided
    if (arg_index < argc) {
        output_file = argv[arg_index++];
    }
    
    // Print debug information
    if (debug_mode) {
        printf("Debug mode enabled\n");
        if (dump_ast) {
            printf("AST visualization enabled (format: %s)\n", ast_format ? ast_format : "mermaid");
        }
    } else if (verbose_mode) {
        printf("Verbose mode enabled\n");
    }
    
    // Read input file
    size_t source_length = 0;
    char* source = read_file(input_file, &source_length);
    if (!source) {
        return 1;
    }
    
    // Initialize memory arena
    Arena* arena = arena_create(1024 * 1024); // 1MB initial size
    if (!arena) {
        fprintf(stderr, "Error: Failed to create memory arena\n");
        free(source);
        return 1;
    }
    
    // Initialize string table
    StringTable* strings = string_table_create(arena, 1024); // 1024 initial capacity
    if (!strings) {
        fprintf(stderr, "Error: Failed to create string table\n");
        arena_destroy(arena);
        free(source);
        return 1;
    }
    
    // Initialize diagnostic context
    DiagnosticContext* diag = diagnostic_context_create(arena);
    if (!diag) {
        fprintf(stderr, "Error: Failed to create diagnostic context\n");
        arena_destroy(arena);
        free(source);
        return 1;
    }
    
    // Set verbosity level
    if (debug_mode) {
        diagnostic_context_set_verbosity(diag, VERBOSITY_DEBUG);
    } else if (verbose_mode) {
        diagnostic_context_set_verbosity(diag, VERBOSITY_VERBOSE);
    }
    
    // Initialize lexer
    Lexer* lexer = lexer_create(arena, strings, diag, source);
    if (!lexer) {
        fprintf(stderr, "Error: Failed to create lexer\n");
        arena_destroy(arena);
        free(source);
        return 1;
    }
    
    // Initialize parser
    Parser* parser = parser_create(arena, strings, diag, lexer);
    if (!parser) {
        fprintf(stderr, "Error: Failed to create parser\n");
        arena_destroy(arena);
        free(source);
        return 1;
    }
    
    // Parse program
    AstNode* ast = parser_parse_program(parser);
    if (!ast) {
        fprintf(stderr, "Error: Failed to parse program\n");
        arena_destroy(arena);
        free(source);
        return 1;
    }

    // Handle AST visualization if requested
    if (dump_ast || (output_file && strcmp(output_file, "--dump-ast") == 0)) {
        const char* format = ast_format ? ast_format : "mermaid";
        printf("Compiling %s to --dump-ast...\n", input_file);
        printf("\nAST Visualization:\n");
        ast_visualize(ast, format);
        printf("\nSuccessfully compiled %s to --dump-ast\n", input_file);
        arena_destroy(arena);
        free(source);
        return 0;
    }
    
    // Initialize type inference context
    TypeInferenceContext* type_context = type_inference_context_create(arena, diag);
    if (!type_context) {
        fprintf(stderr, "Error: Failed to create type inference context\n");
        arena_destroy(arena);
        free(source);
        return 1;
    }
    
    // Collect explicit types from the AST
    if (!type_inference_collect_explicit_types(type_context, ast)) {
        fprintf(stderr, "Error: Failed to collect explicit types\n");
        arena_destroy(arena);
        free(source);
        return 1;
    }
    
    // Infer types for the AST
    if (!type_inference_infer(type_context, ast)) {
        fprintf(stderr, "Error: Failed to infer types\n");
        arena_destroy(arena);
        free(source);
        return 1;
    }

    // Handle type analysis if requested
    if (analyze_types) {
        printf("Analyzing types with detail level: %s\n", detail_level);
        type_inference_analyze(type_context, ast, detail_level);
        arena_destroy(arena);
        free(source);
        return 0;
    }
    
    // Initialize code generator
    CodegenContext* codegen = codegen_context_create(arena, diag, type_context);
    if (!codegen) {
        fprintf(stderr, "Error: Failed to create code generator\n");
        arena_destroy(arena);
        free(source);
        return 1;
    }

    // Handle code generation debugging if requested
    if (debug_codegen) {
        printf("Debugging code generation stage: %s\n", debug_stage);
        codegen_debug(codegen, ast, debug_stage);
        arena_destroy(arena);
        free(source);
        return 0;
    }

    // Handle profiling if requested
    if (profile_mode) {
        printf("Profiling compilation...\n");
        codegen_profile(codegen, ast);
        arena_destroy(arena);
        free(source);
        return 0;
    }
    
    // Determine mode
    if (output_file && output_file[0] != '-') {
        // Compile to C file
        printf("Compiling %s to %s...\n", input_file, output_file);
        
        // Generate C code
        if (!codegen_generate(codegen, ast, output_file)) {
            fprintf(stderr, "Error: Failed to generate C code\n");
            arena_destroy(arena);
            free(source);
            return 1;
        }
        
        printf("Successfully compiled %s to %s\n", input_file, output_file);
    } else {
        // Compile and run
        printf("Compiling and running %s...\n", input_file);
        
        // Generate C code to a temporary file
        char temp_file[256];
        snprintf(temp_file, sizeof(temp_file), "%s.c", input_file);
        
        // Generate C code
        if (!codegen_generate(codegen, ast, temp_file)) {
            fprintf(stderr, "Error: Failed to generate C code\n");
            
            // Print diagnostic messages
            diagnostic_context_print(diag);
            
            // Print AST for debugging
            fprintf(stderr, "AST dump:\n");
            ast_print(ast, 0);
            
            arena_destroy(arena);
            free(source);
            return 1;
        }
        
        // Compile and execute
        int result = codegen_compile_and_execute(codegen, temp_file, NULL, 0);
        if (result != 0) {
            fprintf(stderr, "Error: Failed to compile and execute program\n");
            arena_destroy(arena);
            free(source);
            return 1;
        }
    }
    
    // Clean up
    arena_destroy(arena);
    free(source);
    return 0;
}

/**
 * Read file contents into a string
 * 
 * @param filename The name of the file to read
 * @param length Pointer to store the length of the file
 * @return The file contents as a string, or NULL on error
 */
static char* read_file(const char* filename, size_t* length) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Allocate buffer
    char* buffer = malloc(size + 1);
    if (!buffer) {
        fprintf(stderr, "Error: Out of memory\n");
        fclose(file);
        return NULL;
    }
    
    // Read file contents
    size_t read = fread(buffer, 1, size, file);
    fclose(file);
    
    if (read != size) {
        fprintf(stderr, "Error: Could not read file '%s'\n", filename);
        free(buffer);
        return NULL;
    }
    
    // Null-terminate
    buffer[size] = '\0';
    
    if (length) {
        *length = size;
    }
    
    return buffer;
}
