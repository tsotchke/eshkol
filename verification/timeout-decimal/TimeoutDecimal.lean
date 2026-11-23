import Init

namespace Verificatum.TimeoutDecimal

-- The linked x86-64 loop uses MUL's high word followed by SHR 3.
theorem quotient (n : Nat) (word : n < 2^64) :
    n * 14757395258967641293 / 2^64 / 8 = n / 10 := by
  rw [Nat.div_div_eq_div_mul]
  have decomp := Nat.mod_add_div n 10
  have remBound := Nat.mod_lt n (by decide : 0 < 10)
  omega

-- The loop subtracts ten times the quotient and adds ASCII '0'.
theorem digit (n : Nat) (word : n < 2^64) :
    n - 10 * (n * 14757395258967641293 / 2^64 / 8) + 48 = 48 + n % 10 := by
  rw [quotient n word]
  have := Nat.mod_add_div n 10
  omega

-- A do-while loop emits a digit even for zero. None means fuel ran out.
def writes : Nat → Nat → Option Nat
  | 0, _ => none
  | fuel + 1, n =>
    if n < 10 then some 1 else (writes fuel (n / 10)).map Nat.succ

theorem writesBound {fuel n : Nat} (positive : 0 < fuel) (fits : n < 10^fuel) :
    ∃ count, writes fuel n = some count ∧ 0 < count ∧ count ≤ fuel := by
  induction fuel generalizing n with
  | zero => omega
  | succ fuel ih =>
    by_cases last : n < 10
    · exact ⟨1, by simp [writes, last], by decide, by omega⟩
    · have smaller : n / 10 < 10^fuel := by
        apply (Nat.div_lt_iff_lt_mul (by decide : 0 < 10)).mpr
        simpa [Nat.pow_succ] using fits
      have rest : 0 < fuel := by
        cases fuel with
        | zero => simp at fits; omega
        | succ => omega
      obtain ⟨count, h, _, le⟩ := ih rest smaller
      exact ⟨count + 1, by simp [writes, last, h], by omega, by omega⟩

-- Relative to the 64-byte local buffer: suffix at [61,64), digits
-- written successively at 60,59,... down to (and including) 61-count.
theorem formatterExtent (n : Nat) (word : n < 2^64) :
    ∃ count, writes 20 n = some count ∧ 1 ≤ count ∧ count + 3 ≤ 23 ∧
      ∀ i, i < count → 41 ≤ 60 - i ∧ 60 - i < 61 := by
  obtain ⟨count, h, pos, le⟩ := writesBound (fuel := 20) (n := n) (by decide)
    (Nat.lt_trans word (by decide))
  exact ⟨count, h, pos, by omega, by intro i hi; omega⟩

#print axioms quotient
#print axioms formatterExtent
end Verificatum.TimeoutDecimal
