#!/bin/bash

# Eshkol REPL Test Suite Runner - With Full Output
# Shows actual REPL output for each test for verification

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0

# Output directory
OUTPUT_DIR="repl_test_outputs"
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "  Eshkol REPL Test Suite (With Output)"
echo "========================================="
echo ""
echo "Output files will be saved to: $OUTPUT_DIR/"
echo ""

# Ensure build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}Error: build directory not found. Run cmake first.${NC}"
    exit 1
fi

# Check if REPL exists
if [ ! -f "build/eshkol-repl" ]; then
    echo -e "${RED}Error: eshkol-repl not found. Run make first.${NC}"
    exit 1
fi

# Function to run a REPL test with full output
run_test_with_output() {
    local test_name="$1"
    local input="$2"
    local expected="$3"
    local safe_name=$(echo "$test_name" | tr ' ' '_' | tr -cd '[:alnum:]_')
    local output_file="$OUTPUT_DIR/${safe_name}_output.txt"

    echo "========================================="
    echo -e "${CYAN}TEST: $test_name${NC}"
    echo "========================================="
    echo ""
    echo -e "${DIM}Input:${NC}"
    echo "$input" | sed 's/^/  /'
    echo ""
    echo -e "${DIM}Expected pattern:${NC} $expected"
    echo ""

    # Run the REPL with input
    echo -e "$input\n:quit" | timeout 30 ./build/eshkol-repl 2>&1 > "$output_file" || true

    echo -e "${DIM}Actual output:${NC}"
    echo "----------------------------------------"
    cat "$output_file" | sed 's/^/  /'
    echo "----------------------------------------"

    # Check if expected pattern is in output
    if grep -qE "$expected" "$output_file" 2>/dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((PASS++)) || true
    else
        echo -e "${RED}❌ FAIL - Expected pattern not found${NC}"
        ((FAIL++)) || true
    fi
    echo ""
}

# ============================================================================
# SECTION 1: Basic Arithmetic
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 1: BASIC ARITHMETIC"
echo "###################################################################"
echo ""

run_test_with_output "Integer addition" "(+ 1 2 3)" "6"
run_test_with_output "Integer subtraction" "(- 10 3)" "7"
run_test_with_output "Integer multiplication" "(* 4 5)" "20"
run_test_with_output "Integer division" "(/ 20 4)" "5"
run_test_with_output "Floating point" "(+ 1.5 2.5)" "4"
run_test_with_output "Nested arithmetic" "(+ (* 2 3) (- 10 5))" "11"

# ============================================================================
# SECTION 2: Comparisons
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 2: COMPARISONS"
echo "###################################################################"
echo ""

run_test_with_output "Equal (true)" "(= 5 5)" "#t"
run_test_with_output "Equal (false)" "(= 5 6)" "#f"
run_test_with_output "Less than" "(< 3 5)" "#t"
run_test_with_output "Greater than" "(> 10 5)" "#t"
run_test_with_output "Less or equal" "(<= 5 5)" "#t"
run_test_with_output "Greater or equal" "(>= 6 5)" "#t"

# ============================================================================
# SECTION 3: Boolean Operations
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 3: BOOLEAN OPERATIONS"
echo "###################################################################"
echo ""

run_test_with_output "And (true)" "(and #t #t)" "#t"
run_test_with_output "And (false)" "(and #t #f)" "#f"
run_test_with_output "Or (true)" "(or #f #t)" "#t"
run_test_with_output "Or (false)" "(or #f #f)" "#f"
run_test_with_output "Not" "(not #f)" "#t"

# ============================================================================
# SECTION 4: Conditionals
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 4: CONDITIONALS"
echo "###################################################################"
echo ""

run_test_with_output "If true branch" "(if #t 1 2)" "1"
run_test_with_output "If false branch" "(if #f 1 2)" "2"
run_test_with_output "Nested if" "(if (> 5 3) (if (< 2 1) 10 20) 30)" "20"

# ============================================================================
# SECTION 5: Variable Definitions
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 5: VARIABLE DEFINITIONS"
echo "###################################################################"
echo ""

run_test_with_output "Define and reference" "(define x 42)
x" "42"

run_test_with_output "Define with expression" "(define y (+ 10 20))
y" "30"

run_test_with_output "Use defined variables" "(define a 5)
(define b 7)
(+ a b)" "12"

# ============================================================================
# SECTION 6: Function Definitions
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 6: FUNCTION DEFINITIONS"
echo "###################################################################"
echo ""

run_test_with_output "Simple function" "(define (square n) (* n n))
(square 5)" "25"

run_test_with_output "Multi-param function" "(define (add3 a b c) (+ a b c))
(add3 1 2 3)" "6"

run_test_with_output "Recursive factorial" "(define (fact n) (if (<= n 1) 1 (* n (fact (- n 1)))))
(fact 5)" "120"

run_test_with_output "Recursive fibonacci" "(define (fib n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))
(fib 10)" "55"

# ============================================================================
# SECTION 7: Lambda Expressions
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 7: LAMBDA EXPRESSIONS"
echo "###################################################################"
echo ""

run_test_with_output "Inline lambda" "((lambda (x) (* x 2)) 5)" "10"
run_test_with_output "Lambda multi-params" "((lambda (a b) (+ a b)) 3 4)" "7"
run_test_with_output "Stored lambda" "(define double (lambda (x) (* x 2)))
(double 7)" "14"
run_test_with_output "Curried lambda" "(((lambda (x) (lambda (y) (+ x y))) 3) 4)" "7"

# ============================================================================
# SECTION 8: Let Expressions
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 8: LET EXPRESSIONS"
echo "###################################################################"
echo ""

run_test_with_output "Simple let" "(let ((x 5)) x)" "5"
run_test_with_output "Let multiple bindings" "(let ((a 2) (b 3)) (+ a b))" "5"
run_test_with_output "Nested let" "(let ((x 1)) (let ((y 2)) (+ x y)))" "3"
run_test_with_output "Let with computation" "(let ((x (* 3 4))) (+ x 1))" "13"

# ============================================================================
# SECTION 9: List Operations
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 9: LIST OPERATIONS"
echo "###################################################################"
echo ""

run_test_with_output "Create list" "(list 1 2 3)" "(1 2 3)"
run_test_with_output "Quote list" "'(a b c)" "(a b c)"
run_test_with_output "Car" "(car '(1 2 3))" "1"
run_test_with_output "Cdr" "(cdr '(1 2 3))" "(2 3)"
run_test_with_output "Cons" "(cons 1 '(2 3))" "(1 2 3)"
run_test_with_output "Cadr" "(cadr '(1 2 3))" "2"
run_test_with_output "Null? empty" "(null? '())" "#t"
run_test_with_output "Null? non-empty" "(null? '(1))" "#f"
run_test_with_output "Pair?" "(pair? (cons 1 2))" "#t"

# ============================================================================
# SECTION 10: Closures
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 10: CLOSURES"
echo "###################################################################"
echo ""

run_test_with_output "Make-adder closure" "(define (make-adder n) (lambda (x) (+ x n)))
(define add5 (make-adder 5))
(add5 10)" "15"

run_test_with_output "Counter closure" "(define (make-counter)
  (let ((count 0))
    (lambda ()
      (set! count (+ count 1))
      count)))
(define counter (make-counter))
(counter)
(counter)
(counter)" "3"

# ============================================================================
# SECTION 11: Higher-Order Functions (with stdlib)
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 11: HIGHER-ORDER FUNCTIONS (STDLIB)"
echo "###################################################################"
echo ""

run_test_with_output "Map" ":stdlib
(define (sq x) (* x x))
(map sq '(1 2 3 4))" "(1 4 9 16)"

run_test_with_output "Filter" ":stdlib
(filter (lambda (x) (> x 2)) '(1 2 3 4 5))" "(3 4 5)"

run_test_with_output "Fold/reduce" ":stdlib
(fold + 0 '(1 2 3 4 5))" "15"

run_test_with_output "Length" ":stdlib
(length '(1 2 3 4 5))" "5"

run_test_with_output "Append" ":stdlib
(append '(1 2) '(3 4))" "(1 2 3 4)"

run_test_with_output "Reverse" ":stdlib
(reverse '(1 2 3))" "(3 2 1)"

# ============================================================================
# SECTION 12: Math Functions
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 12: MATH FUNCTIONS"
echo "###################################################################"
echo ""

run_test_with_output "Abs" "(abs -5)" "5"
run_test_with_output "Modulo" "(modulo 17 5)" "2"
run_test_with_output "Max" "(max 3 7 2 9 1)" "9"
run_test_with_output "Min" "(min 3 7 2 9 1)" "1"
run_test_with_output "Sqrt" "(sqrt 16)" "4"
run_test_with_output "Expt" "(expt 2 10)" "1024"

# ============================================================================
# SECTION 13: Autodiff
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 13: AUTOMATIC DIFFERENTIATION"
echo "###################################################################"
echo ""

run_test_with_output "Symbolic diff x^2" "(differentiate (lambda (x) (* x x)) 'x)" "\\* 2 x"

run_test_with_output "Forward-mode D(x^2) at 3" "(let ((f (lambda (x) (* x x))))
  (let ((df (D f)))
    (df 3)))" "6"

run_test_with_output "Forward-mode D(sin) at 0" "(let ((df (D sin)))
  (df 0))" "1"

# ============================================================================
# SECTION 14: Type Predicates
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 14: TYPE PREDICATES"
echo "###################################################################"
echo ""

run_test_with_output "number?" "(number? 42)" "#t"
run_test_with_output "number? false" "(number? \"hello\")" "#f"
run_test_with_output "string?" "(string? \"hello\")" "#t"
run_test_with_output "boolean?" "(boolean? #t)" "#t"
run_test_with_output "procedure?" "(define (f x) x)
(procedure? f)" "#t"

# ============================================================================
# SECTION 15: REPL Commands
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 15: REPL COMMANDS"
echo "###################################################################"
echo ""

run_test_with_output "Version command" ":version" "Eshkol"
run_test_with_output "Help command" ":help" "Commands"
run_test_with_output "Type command" ":type (+ 1 2)" "Type"
run_test_with_output "Env command" ":env" "Defined"

# ============================================================================
# SECTION 16: Complex Expressions
# ============================================================================

echo ""
echo "###################################################################"
echo "#  SECTION 16: COMPLEX EXPRESSIONS"
echo "###################################################################"
echo ""

run_test_with_output "Ackermann(2,3)" "(define (ack m n)
  (cond ((= m 0) (+ n 1))
        ((= n 0) (ack (- m 1) 1))
        (else (ack (- m 1) (ack m (- n 1))))))
(ack 2 3)" "9"

run_test_with_output "Y-combinator factorial" "(define Y
  (lambda (f)
    ((lambda (x) (f (lambda (v) ((x x) v))))
     (lambda (x) (f (lambda (v) ((x x) v)))))))
((Y (lambda (fact)
      (lambda (n)
        (if (<= n 1)
            1
            (* n (fact (- n 1))))))) 5)" "120"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "###################################################################"
echo "#  TEST RESULTS SUMMARY"
echo "###################################################################"
echo ""
echo "Total Tests:    $((PASS + FAIL))"
echo -e "${GREEN}Passed:         $PASS${NC}"
echo -e "${RED}Failed:         $FAIL${NC}"
echo ""
echo "Output files saved to: $OUTPUT_DIR/"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}*** ALL REPL TESTS PASSED ***${NC}"
    exit 0
else
    echo -e "${RED}*** SOME TESTS FAILED ***${NC}"
    echo ""
    echo "Failed test outputs can be found in $OUTPUT_DIR/"
    exit 1
fi
