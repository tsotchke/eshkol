# Getting Started with Eshkol

## Table of Contents
- [Installation](#installation)
- [Your First Eshkol Program](#your-first-eshkol-program)
- [Basic Syntax](#basic-syntax)
- [Common Operations](#common-operations)
- [Development Environment Setup](#development-environment-setup)
- [Debugging](#debugging)

## Installation

### Prerequisites

Before installing Eshkol, ensure you have the following prerequisites:

- C compiler (GCC 7+ or Clang 6+)
- Make
- Git
- CMake (version 3.10+)

### Installing from Source

```bash
# Clone the repository
git clone https://github.com/tsotchke/eshkol.git
cd eshkol

# Build the compiler
make

# Add to your PATH
export PATH=$PATH:$(pwd)/bin

# Verify installation
eshkol --version
```

### Package Manager Installation

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install eshkol
```

#### macOS (Homebrew)

```bash
brew install eshkol
```

## Your First Eshkol Program

Let's create a simple "Hello, World!" program to verify your installation:

1. Create a file named `hello.esk`:

```scheme
;; hello.esk
(display "Hello, Eshkol!")
```

2. Compile the program:

```bash
eshkol hello.esk
```

3. Run the compiled program:

```bash
./hello
```

You should see the output: `Hello, Eshkol!`

## Basic Syntax

Eshkol uses S-expressions for its syntax, similar to Scheme and other Lisp dialects.

### Variables and Functions

```scheme
;; Define a variable
(define x 42)

;; Define a function
(define (square n)
  (* n n))

;; Call a function
(square 5)  ; Returns 25
```

### Conditionals

```scheme
;; If expression
(if (> x 0)
    "Positive"
    "Non-positive")

;; Cond expression
(cond
  [(< x 0) "Negative"]
  [(= x 0) "Zero"]
  [else "Positive"])
```

### Loops

```scheme
;; Iterate over a list
(for-each (lambda (item)
            (display item))
          '(1 2 3 4 5))

;; Recursive loop
(define (countdown n)
  (if (<= n 0)
      (display "Blast off!")
      (begin
        (display n)
        (countdown (- n 1)))))
```

## Common Operations

### Working with Lists

```scheme
;; Create a list
(define numbers '(1 2 3 4 5))

;; Access elements
(car numbers)    ; First element: 1
(cadr numbers)   ; Second element: 2
(cdr numbers)    ; Rest of the list: (2 3 4 5)

;; Map function over a list
(map square numbers)  ; Returns (1 4 9 16 25)

;; Filter a list
(filter (lambda (n) (even? n)) numbers)  ; Returns (2 4)

;; Reduce a list
(fold + 0 numbers)  ; Sum: 15
```

### Working with Strings

```scheme
;; String concatenation
(string-append "Hello, " "world!")

;; String formatting
(format "Value: %d" 42)

;; String operations
(string-length "Eshkol")  ; Returns 6
(substring "Eshkol" 0 3)   ; Returns "Esh"
```

### File I/O

```scheme
;; Read a file
(define content (read-file "data.txt"))

;; Write to a file
(write-file "output.txt" "Hello, Eshkol!")

;; Process a file line by line
(with-input-from-file "data.txt"
  (lambda ()
    (let loop ([line (read-line)])
      (unless (eof-object? line)
        (process-line line)
        (loop (read-line))))))
```

## Development Environment Setup

### Editor Integration

#### VS Code

1. Install the Eshkol extension:
   ```
   ext install eshkol-lang
   ```

2. Configure settings:
   ```json
   {
     "eshkol.compiler.path": "/path/to/eshkol",
     "eshkol.format.enabled": true
   }
   ```

#### Emacs

Add the following to your `.emacs` file:

```elisp
(require 'eshkol-mode)
(add-to-list 'auto-mode-alist '("\\.esk\\'" . eshkol-mode))
```

#### Vim/Neovim

Add the following to your `.vimrc` or `init.vim`:

```vim
Plug 'eshkol/vim-eshkol'
autocmd BufNewFile,BufRead *.esk setfiletype eshkol
```

### Project Structure

A typical Eshkol project structure:

```
my-project/
├── src/
│   ├── main.esk
│   └── utils.esk
├── include/
│   └── types.esk
├── tests/
│   └── test-utils.esk
├── build/
└── Makefile
```

Example Makefile:

```make
.PHONY: all clean test

all:
	eshkol build src/main.esk -o build/main

test:
	eshkol test tests/*.esk

clean:
	rm -rf build/*
```

## Debugging

### Using the Debugger

Eshkol comes with a built-in debugger:

```bash
eshkol debug my-program.esk
```

Common debugger commands:

- `break <line>` - Set a breakpoint
- `continue` - Continue execution
- `step` - Step into function calls
- `next` - Step over function calls
- `print <expr>` - Evaluate and print an expression
- `backtrace` - Show call stack

### Logging

```scheme
;; Basic logging
(log-info "Processing item" item)
(log-warning "Unusual value detected" value)
(log-error "Failed to process item" item)

;; Configure logging
(set-log-level! 'debug)
(set-log-output! "app.log")
```

### Performance Profiling

```bash
# Run with profiling enabled
eshkol profile my-program.esk

# Generate profile report
eshkol profile-report
```

This will generate a report showing:
- Function call counts
- Time spent in each function
- Memory allocation patterns

For more detailed information on specific language features, refer to the following documentation:
- [Memory Management](MEMORY_MANAGEMENT.md)
- [Type System](TYPE_SYSTEM.md)
- [Function Composition](FUNCTION_COMPOSITION.md)
- [Automatic Differentiation](AUTODIFF.md)
- [Vector Operations](VECTOR_OPERATIONS.md)
