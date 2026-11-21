import Init

namespace Verificatum.Obligation

/-- The two shapes of one rank-two matrix multiplication in typed HIR.
    Identifiers refer to stable, resolved natural-number dimensions. -/
structure Matmul where
  rows : Nat
  leftCols : Nat
  rightRows : Nat
  cols : Nat

abbrev Dimensions := Nat → Nat
abbrev Fact := Nat × Nat

def Holds (dims : Dimensions) (fact : Fact) : Prop :=
  dims fact.1 = dims fact.2

def Satisfies (dims : Dimensions) (facts : List Fact) : Prop :=
  ∀ f ∈ facts, Holds dims f

/-- The source rule consumes the complete left row and right column together. -/
inductive Pairable : Nat → Nat → Prop where
  | empty : Pairable 0 0
  | next : Pairable n m → Pairable (n+1) (m+1)

def SourceSafe (dims : Dimensions) (call : Matmul) : Prop :=
  Pairable (dims call.leftCols) (dims call.rightRows)

structure Goal where
  premises : List Fact
  conclusion : Fact
deriving DecidableEq

def Meaning (g : Goal) : Prop :=
  ∀ dims, Satisfies dims g.premises → Holds dims g.conclusion

/-- Construct this target from the source site and its authorized facts.
    The proof producer may choose the submitted goal and its derivation. -/
def expected (facts : List Fact) (call : Matmul) : Goal :=
  ⟨facts, call.leftCols, call.rightRows⟩

def bindTarget (facts : List Fact) (call : Matmul) (submitted : Goal) : Bool :=
  decide (submitted = expected facts call)

private theorem pairSelf (n : Nat) : Pairable n n := by
  induction n with
  | zero => exact .empty
  | succ _ h => exact .next h

theorem discharge_sound {facts call submitted}
    (bound : bindTarget facts call submitted = true)
    (verified : Meaning submitted) (dims : Dimensions)
    (contextValid : Satisfies dims facts) : SourceSafe dims call := by
  have same : submitted = expected facts call := of_decide_eq_true bound
  subst submitted
  have eq : dims call.leftCols = dims call.rightRows := verified dims contextValid
  unfold SourceSafe
  rw [eq]
  exact pairSelf _

-- Three different source dimensions. Only the two inner dimensions matter.
private def site : Matmul := ⟨2, 0, 1, 2⟩
private def inner : Fact := (0, 1)

-- Positive control with a genuine source hypothesis.
example : bindTarget [inner] site ⟨[inner], inner⟩ = true := rfl
example : Meaning (expected [inner] site) := fun _ h => h inner (.head _)

-- Both forged propositions have proofs. One weakens the conclusion to
-- reflexivity. The other invents the very assumption needed to justify it.
example : Meaning ⟨[], (0, 0)⟩ := fun _ _ => rfl
example : bindTarget [] site ⟨[], (0, 0)⟩ = false := rfl
example : Meaning ⟨[inner], inner⟩ := fun _ h => h inner (.head _)
example : bindTarget [] site ⟨[inner], inner⟩ = false := rfl

-- A concrete valuation refutes the real source rule for that unguarded site.
example : ¬ SourceSafe (fun i => i) site := by
  intro h
  change Pairable 0 1 at h
  cases h

end Verificatum.Obligation
