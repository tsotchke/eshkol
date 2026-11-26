namespace Fixture

def double (n : Nat) : Nat := n + n

def unused (n : Nat) : Nat := n

theorem double_even (n : Nat) : double n % 2 = 0 := by
  unfold double
  omega

-- The target's own axiom report. In a mutant that breaks double_even's proof it names
-- sorryAx; the runner must judge a witness by the witness's report alone (expect-refuted).
#print axioms double_even

theorem double_ge (n : Nat) : n ≤ double n := by
  unfold double
  omega

end Fixture
