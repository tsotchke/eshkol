#!/bin/bash
# Test REPL without DEBUG noise

echo "(+ 1 2)" | ./build/eshkol-repl
echo ""
echo "(lambda (x) (* x x))" | ./build/eshkol-repl
