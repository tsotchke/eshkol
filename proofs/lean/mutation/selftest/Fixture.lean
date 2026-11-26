namespace Fixture

def double (n : Nat) : Nat := n + n

def unused (n : Nat) : Nat := n

theorem double_even (n : Nat) : double n % 2 = 0 := by
  unfold double
  omega

theorem double_ge (n : Nat) : n ≤ double n := by
  unfold double
  omega

end Fixture
