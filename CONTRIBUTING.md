# Contributing to Eshkol

Thank you for your interest in contributing to Eshkol! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Development Environment Setup](#development-environment-setup)
  - [Building the Project](#building-the-project)
  - [Running Tests](#running-tests)
- [How to Contribute](#how-to-contribute)
  - [Reporting Issues](#reporting-issues)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Guidelines](#development-guidelines)
  - [Coding Standards](#coding-standards)
  - [Documentation](#documentation)
  - [Testing](#testing)
- [Project Structure](#project-structure)
- [Communication](#communication)

## Code of Conduct

We expect all contributors to adhere to our Code of Conduct. Please be respectful and considerate of others when participating in our community.

## Getting Started

### Development Environment Setup

To set up your development environment for Eshkol, you'll need:

1. **C/C++ Compiler**
   - GCC 9.0+ or Clang 10.0+
   - On macOS: `brew install gcc` or use the default Clang
   - On Linux: `sudo apt install build-essential`
   - On Windows: Install MinGW or use WSL

2. **CMake**
   - Version 3.12 or higher
   - On macOS: `brew install cmake`
   - On Linux: `sudo apt install cmake`
   - On Windows: Download from [cmake.org](https://cmake.org/download/)

3. **Node.js** (for MCP tools)
   - Version 14.0+ recommended
   - Download from [nodejs.org](https://nodejs.org/)

4. **Git**
   - On macOS: `brew install git`
   - On Linux: `sudo apt install git`
   - On Windows: Download from [git-scm.com](https://git-scm.com/download/win)

### Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/tsotchke/eshkol.git
   cd eshkol
   ```

2. Create a build directory and build the project:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   make
   ```

3. Install the MCP tools (optional, for development tools):
   ```bash
   cd ../eshkol-tools
   npm install
   ```

### Running Tests

Run the test suite to ensure everything is working correctly:

```bash
cd build
ctest
```

Or run specific test categories:

```bash
# Run unit tests only
ctest -R unit

# Run integration tests only
ctest -R integration
```

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. Check the [GitHub Issues](https://github.com/tsotchke/eshkol/issues) to see if the issue has already been reported.
2. If not, create a new issue with a clear title and description.
3. Include steps to reproduce the issue, expected behavior, and actual behavior.
4. If possible, include code examples, logs, or screenshots.

### Suggesting Enhancements

We welcome suggestions for new features or improvements:

1. Check the [ROADMAP.md](ROADMAP.md) to see if your suggestion is already planned.
2. Check existing issues to avoid duplicates.
3. Create a new issue with the label "enhancement".
4. Clearly describe the feature and its benefits.
5. If possible, outline how the feature might be implemented.

### Pull Requests

We follow a standard GitHub flow for contributions:

1. Fork the repository.
2. Create a new branch for your feature or bugfix: `git checkout -b feature/your-feature-name` or `git checkout -b fix/issue-description`.
3. Make your changes, following our [coding standards](#coding-standards).
4. Add or update tests as necessary.
5. Update documentation to reflect your changes.
6. Commit your changes with clear, descriptive commit messages.
7. Push your branch to your fork: `git push origin your-branch-name`.
8. Submit a pull request to the `main` branch of the Eshkol repository.
9. Respond to any feedback or questions during the review process.

## Development Guidelines

### Coding Standards

We follow these coding standards for consistency:

#### C Code

- Use 4 spaces for indentation (no tabs).
- Follow the [Linux kernel coding style](https://www.kernel.org/doc/html/latest/process/coding-style.html) with some modifications:
  - Use camelCase for function names and variables.
  - Use PascalCase for struct and enum names.
  - Use ALL_CAPS for macros and constants.
- Prefix all public functions with `eshkol_`.
- Keep lines under 100 characters when possible.
- Add comments for complex logic or non-obvious behavior.
- Use descriptive variable and function names.

#### TypeScript Code (MCP Tools)

- Use 2 spaces for indentation.
- Follow the [TypeScript Style Guide](https://google.github.io/styleguide/tsguide.html).
- Use camelCase for variables and functions.
- Use PascalCase for classes, interfaces, and type aliases.
- Use descriptive names and add JSDoc comments.

### Documentation

Good documentation is crucial for the project:

- Update relevant documentation when making changes.
- Document all public APIs with clear descriptions, parameters, and return values.
- Add examples for complex features.
- Keep the README and other high-level documentation up to date.
- Use Markdown for all documentation files.

### Testing

We strive for good test coverage:

- Add tests for new features.
- Update tests when modifying existing features.
- Ensure all tests pass before submitting a pull request.
- Follow the existing test patterns in the codebase.

## Project Structure

Understanding the project structure will help you contribute effectively:

```
eshkol/
├── build/                  # Build output (generated)
├── docs/                   # Documentation
│   ├── architecture/       # Architecture documentation
│   ├── components/         # Component documentation
│   ├── reference/          # API reference
│   ├── scheme_compatibility/ # Scheme compatibility docs
│   ├── tutorials/          # Tutorials
│   ├── type_system/        # Type system documentation
│   └── vision/             # Vision and roadmap
├── examples/               # Example Eshkol programs
├── include/                # Public header files
│   ├── backend/            # Code generation headers
│   ├── core/               # Core functionality headers
│   └── frontend/           # Parser and type system headers
├── src/                    # Source code
│   ├── backend/            # Code generation implementation
│   ├── core/               # Core functionality implementation
│   └── frontend/           # Parser and type system implementation
├── tests/                  # Test suite
│   ├── integration/        # Integration tests
│   └── unit/               # Unit tests
├── eshkol-tools/           # MCP tools for development
├── eshkol-vscode/          # VSCode extension
├── CMakeLists.txt          # Main build configuration
├── CONTRIBUTING.md         # This file
├── LICENSE                 # MIT license
├── README.md               # Project overview
└── ROADMAP.md              # Development roadmap
```

## Communication

- **GitHub Issues**: For bug reports, feature requests, and specific technical discussions.
- **GitHub Discussions**: For general questions, ideas, and community discussions.
- **Pull Requests**: For code contributions and code reviews.

## Priority Areas for Contribution

We especially welcome contributions in these areas:

1. **Function Composition**: Fixing issues with function composition (SCH-019).
2. **Type System**: Improving type inference and integration with autodiff.
3. **Scheme Compatibility**: Implementing standard Scheme functions and predicates.
4. **Documentation**: Improving tutorials, examples, and API documentation.
5. **Testing**: Adding more tests and improving test coverage.
6. **Performance**: Optimizing the compiler and runtime.

See the [ROADMAP.md](ROADMAP.md) for more details on our development priorities.

## Recognition

We value all contributions and will recognize contributors in our release notes and on the project website. Significant contributors may be invited to join the core team.

Thank you for contributing to Eshkol!
