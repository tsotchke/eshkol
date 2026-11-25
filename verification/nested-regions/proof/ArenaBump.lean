namespace Verificatum.ArenaBump

abbrev modulus : Nat := 2^64

-- This is the x86-64 computation (v + a - 1) & -a, in unsigned words.
def rounded (v a : Nat) := ((v + a - 1) % modulus) &&& (modulus - a)

private theorem mask (x k : Nat) (hk : k = 3 ∨ k = 4 ∨ k = 6) :
    (x % modulus) &&& (modulus - 2^k) = (x % modulus / 2^k) * 2^k := by
  have hm : modulus - 2^k = (2^(64-k)-1)*2^k := by
    rcases hk with rfl | rfl | rfl <;> decide
  rw [hm]
  apply Nat.eq_of_testBit_eq
  intro i
  simp only [Nat.testBit_and, Nat.testBit_mul_two_pow, Nat.testBit_two_pow_sub_one,
             Nat.testBit_div_two_pow, Nat.testBit_mod_two_pow]
  by_cases lo : k ≤ i
  · have d : i-k+k = i := by simp only [modulus, Nat.reducePow] at * <;> omega
    have b : i-k < 64-k ↔ i < 64 := by rcases hk with rfl | rfl | rfl <;> omega
    simp [lo, d, b, Bool.and_comm, Bool.and_left_comm]
  · simp [lo]

private theorem rounds (x a : Nat) (ha : a = 8 ∨ a = 16 ∨ a = 64) (small : x < 2^62) :
    x ≤ rounded x a ∧ rounded x a < x + a ∧ rounded x a % a = 0 := by
  rcases ha with rfl | rfl | rfl
  · unfold rounded
    rw [show (8:Nat) = 2^3 by decide, mask _ 3 (Or.inl rfl)]
    simp only [modulus, Nat.reducePow]
    omega
  · unfold rounded
    rw [show (16:Nat) = 2^4 by decide, mask _ 4 (Or.inr (Or.inl rfl))]
    simp only [modulus, Nat.reducePow]
    omega
  · unfold rounded
    rw [show (64:Nat) = 2^6 by decide, mask _ 6 (Or.inr (Or.inr rfl))]
    simp only [modulus, Nat.reducePow]
    omega

def offset (base used a : Nat) := (modulus + rounded ((base + used) % modulus) a - base) % modulus
def nextUsed (base used n a : Nat) := (offset base used a + rounded n a) % modulus
def result (base used a : Nat) := (offset base used a + base) % modulus

theorem ordinary_aligned (base used capacity n a : Nat)
    (hb : base < 2^60) (hc : capacity < 2^60) (hu : used ≤ capacity)
    (hn : n < 2^60) (positive : 0 < n) (ha : a = 8 ∨ a = 16 ∨ a = 64)
    (fits : nextUsed base used n a ≤ capacity) :
    let p := result base used a
    p % a = 0 ∧ base + used ≤ p ∧ p < p + n ∧
      p + n ≤ base + nextUsed base used n a ∧
      base + nextUsed base used n a ≤ base + capacity := by
  have hab : a ≤ 64 := by omega
  have addr := rounds (base+used) a ha (by omega)
  have req := rounds n a ha (by omega)
  have noWrap : (base+used) % modulus = base+used := Nat.mod_eq_of_lt (by unfold modulus; omega)
  simp only [nextUsed, result, offset, noWrap] at *
  have off : (modulus + rounded (base+used) a - base) % modulus = rounded (base+used) a - base := by
    simp only [modulus, Nat.reducePow] at * <;> omega
  simp only [off] at *
  have ret : (rounded (base+used) a - base + base) % modulus = rounded (base+used) a := by simp only [modulus, Nat.reducePow] at * <;> omega
  rw [ret]
  simp only [modulus, Nat.reducePow] at *
  exact ⟨addr.2.2, addr.1, by omega, by omega, by omega⟩

theorem ordinary (base used capacity n a : Nat)
    (hb : base < 2^60) (hc : capacity < 2^60) (hu : used ≤ capacity)
    (hn : n < 2^60) (positive : 0 < n) (ha : a = 8 ∨ a = 64)
    (fits : nextUsed base used n a ≤ capacity) :
    let p := result base used a
    p % a = 0 ∧ base + used ≤ p ∧ p < p + n ∧
      p + n ≤ base + nextUsed base used n a ∧
      base + nextUsed base used n a ≤ base + capacity :=
  ordinary_aligned _ _ _ _ _ hb hc hu hn positive (ha.imp_right Or.inr) fits

-- A live interval may be in the used prefix or in a different block.
-- This applies to every reserved physical copy, including forwarded ones.
theorem apart (oldLo oldHi prefixEnd blockEnd p endPos : Nat)
    (reserved : oldHi ≤ prefixEnd ∨ blockEnd ≤ oldLo)
    (lower : prefixEnd ≤ p) (upper : endPos ≤ blockEnd) :
    oldHi ≤ p ∨ endPos ≤ oldLo := by omega

-- Model a byte store to the eight-byte cursor field. All data in the owned
-- block and every external live range must be separate from this metadata.
def storeCursor (mem : Nat → UInt8) (field : Nat) (bytes : Fin 8 → UInt8) (x : Nat) :=
  if h : x - field < 8 then
    if field ≤ x then bytes ⟨x - field, h⟩ else mem x
  else mem x

theorem preserves_byte (outside : x < field ∨ field + 8 ≤ x) :
    storeCursor mem field bytes x = mem x := by
  unfold storeCursor
  split <;> try rfl
  split <;> first | rfl | omega

theorem header_layout (base : Nat) (aligned : base % 64 = 0) :
    (base+8) % 64 = 8 ∧ (base+8) % 8 = 0 ∧ base+8+40 = base+48 := by omega

-- All root-visible intervals, including aliases and retained forwarding copies,
-- are supplied by the consumer. Completeness of that inventory is a premise.
theorem reserves_fresh (hb : base < 2^60) (hc : capacity < 2^60) (hu : used ≤ capacity)
    (hn : n < 2^60) (positive : 0 < n) (ha : a = 8 ∨ a = 64)
    (fits : nextUsed base used n a ≤ capacity)
    (live : Nat → Nat → Prop)
    (separated : ∀ lo hi, live lo hi → hi ≤ base + used ∨ base + capacity ≤ lo) :
    ∀ lo hi, live lo hi → hi ≤ result base used a ∨ result base used a + n ≤ lo := by
  have h := ordinary base used capacity n a hb hc hu hn positive ha fits
  exact fun lo hi prior => apart lo hi _ _ _ _ (separated lo hi prior) h.2.1
    (Nat.le_trans h.2.2.2.1 h.2.2.2.2)

-- Either early test returns null before touching the public heap.
def earlyReturn (arena n : Nat) (heap : Nat → UInt8) :=
  if arena = 0 ∨ n = 0 then some (0, heap) else none

theorem early_failure (h : arena = 0 ∨ n = 0) :
    earlyReturn arena n heap = some (0, heap) := by simp [earlyReturn, h]

end Verificatum.ArenaBump
