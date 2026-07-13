CC      ?= gcc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -I src
SRC      = src/value.c src/ad.c src/reader.c src/eval.c src/main.c
BIN      = slermes-eshkol

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) -lm

test: $(BIN)
	@echo "=== eshkol example: gradient descent learns w=2 ==="
	@./$(BIN) tests/gradient_descent.sk
	@echo "=== derivative (lambda (x) (* x x x)) 2 => 12 ==="
	@echo '(display (derivative (lambda (x) (* x x x)) 2.0))' | ./$(BIN)
	@echo "=== derivative-n f 3 1 => 27 (3x^2) ==="
	@echo '(display (derivative-n (lambda (y) (* y y y)) 3.0 1))' | ./$(BIN)
	@echo "=== taylor exp@0.5 order 4 ==="
	@echo '(display (taylor (lambda (x) (exp x)) 0.5 4))' | ./$(BIN)
	@echo "=== sin/cos tower: d/dx sin x = cos x at 0 => 1 ==="
	@echo '(display (derivative (lambda (x) (sin x)) 0.0))' | ./$(BIN)

clean:
	rm -f $(BIN)

.PHONY: all test clean
