#!/bin/bash

# Script to clean up the examples directory by removing generated files from Git tracking
# without deleting them from the filesystem

# Execute this script when you want to ensure that no generated files in the examples
# directory are tracked by Git.

echo "Cleaning up examples directory..."
echo "Removing generated files from Git tracking (without deleting them)..."

# Remove all .c files from Git tracking in examples directory
git rm --cached examples/*.c 2>/dev/null || true
git rm --cached examples/*.esk.c 2>/dev/null || true
git rm --cached examples/*.out 2>/dev/null || true

# Untrack specific files that may still be tracked
git rm --cached examples/factorial.esk.c 2>/dev/null || true
git rm --cached examples/hello.esk.c 2>/dev/null || true

# Remove any executables that might be tracked
find examples -type f -executable -not -path "*/\.*" | while read file; do
    if [ -f "$file" ]; then
        git rm --cached "$file" 2>/dev/null || true
        echo "Untracked: $file"
    fi
done

# Remove .DS_Store from Git tracking
git rm --cached examples/.DS_Store 2>/dev/null || true

echo "Cleaning complete!"
echo "Note: The files still exist in your filesystem, but they are no longer tracked by Git."
echo "You can continue to use and modify these generated files locally."
echo "Run this script again if you need to clean up the examples directory in the future."
