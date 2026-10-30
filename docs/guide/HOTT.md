---
kind: guide
status: current
owner-area: types
since: v1.9.0
sources:
  - inc/eshkol/types/hott_cubical.h
  - lib/types/cubical/elab.cpp
  - lib/types/hott_forms.cpp
  - lib/hott/foundations.esk
  - lib/hott/circle.esk
  - tests/hott/programs/pi1_circle.esk
---
# Homotopy Type Theory in Eshkol

Eshkol's type system includes a cubical type theory in which equality is a
path, univalence holds and computes, and spaces such as the circle are data
types. This guide builds up to the flagship computation: the fundamental
group of the circle is the integers, and the compiler computes winding
numbers. The forms and the complete syntax are in
[docs/reference/language/hott.md](../reference/language/hott.md); the design
is ADR-0043.

Every example below is a complete program; its output is shown after it.

## Paths

A path `(Path A a b)` is a function from the interval to `A` that starts at
`a` and ends at `b`. The constant path is `refl`; a path applied to `i0` or
`i1` gives its endpoints.

```scheme
(hott
  (import numbers)
  (define two-path : (Path Nat 2 2) (lambda (i) 2))
  ;; reversing a path twice gives the same path, by computation
  (example (Pi ((a b : Int) (p : (Path Int a b)))
             (Path (Path Int a b) (sym 0 Int b a (sym 0 Int a b p)) p))
           (lambda (a b p) (refl 0 (Path Int a b) p))))

(display (hott-eval (two-path i1))) (newline)
```

Output:

```text
2
```

`example` asks the compiler to check a statement. A wrong one stops the
compile: `(example (Path Nat 2 3) (refl 0 Nat 2))` is a type error.

## Transport and path induction

`(transp (i) A phi u)` moves `u` along a line of types. The library defines
`transport`, `subst` and path induction `J` with it.

```scheme
(hott (import numbers))
(display (hott-eval (subst 0 0 Nat (lambda (n) Nat) 3 3 (refl 0 Nat 3) 7))) (newline)
```

Output:

```text
7
```

## Univalence

An equivalence of types gives a path between them, `ua`, and transport along
that path applies the equivalence. The successor function of the integers is
an equivalence, so it is a path from `Int` to `Int`:

```scheme
(hott (import numbers))
(display (hott-eval (transport 0 Int Int sucPathInt (pos 4)))) (newline)
(display (hott-eval (transport 0 Int Int (sym 1 U Int Int sucPathInt) (pos 0)))) (newline)
```

Output:

```text
(pos 5)
(negsuc 0)
```

`(negsuc n)` is the integer `-(n+1)`: going backwards along the successor
path from 0 gives -1.

## The circle

The circle is a data type with a point and a path:

```scheme
(hott
  (data Circle
    pt
    (around (dims i) (boundary ((= i 0) pt) ((= i 1) pt))))
  (define around-path : (Path Circle pt pt) around))
(display (hott-eval (around-path i0))) (newline)
```

Output:

```text
pt
```

The library's `circle` module defines `S1` this way, with its universal
cover `helix`: the fibre over `base` is `Int`, and going once around `loop`
is the successor path. The winding number of a loop is the transport of 0
along its lift.

```scheme
(hott (import circle))
(display (hott-eval (winding (trans 0 S1 base base base loop loop)))) (newline)
(display (hott-eval (winding (trans 0 S1 base base base
                                    (trans 0 S1 base base base loop loop)
                                    (sym 0 S1 base base loop))))) (newline)
(display (hott-eval (winding (intLoop (negsuc 3))))) (newline)
```

Output:

```text
(pos 2)
(pos 1)
(negsuc 3)
```

The module also proves that `winding` and `intLoop` are inverse (by path
induction and by induction on the integers), turns this into an equivalence
`OmegaS1-equiv-Int`, and by univalence a path `OmegaS1=Int` from the loop
space of the circle to the integers. Transport along that path is the
winding number, and transport back builds a loop:

```scheme
(hott (import circle))
(display (hott-eval (transport 0 OmegaS1 Int OmegaS1=Int (trans 0 S1 base base base loop loop)))) (newline)
(display (hott-eval (winding (transport 0 Int OmegaS1 (sym 1 U OmegaS1 Int OmegaS1=Int) (pos 3))))) (newline)
```

Output:

```text
(pos 2)
(pos 3)
```

The same definitions, stated in Cubical Agda, compute the same values:
`tests/hott/agda/Pi1Circle.agda`.

## Propositions and truncation

`isProp A` says any two elements of `A` are equal. The propositional
truncation `Trunc` is a data type with a path constructor that makes any two
elements equal; a map into a proposition factors through it:

```scheme
(hott
  (import truncation)
  (import suspension)
  (define is-prop : (isProp 0 (Trunc 0 Nat)) (isProp-Trunc 0 Nat)))
(display (hott-eval (S1->SuspBool base))) (newline)
```

Output:

```text
north
```

The `suspension` module gives `Susp`, with maps between the circle and the
suspension of the booleans.

## Where to go next

- [docs/reference/language/hott.md](../reference/language/hott.md): every form, declaration and library module.
- `lib/hott/*.esk`: the library, written in the same syntax.
- `tests/hott/reject/`: programs the kernel rejects, one per rule.
