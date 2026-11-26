import Init

/-!
# Exact foundations of differentiable SBLC proposals

C/Scheme correspondence for `lib/core/sblc/relax.esk` and item 5 of
`docs/design/adr/0017-stochastic-binary-lambda-calculus.md`:

* `ExactQ`, `jointMean` and `total_variance` give the finite population moments
  underlying `sblc-relax-oracle` and the inner choices in `sblc-relax-noisy`.
* `reinforce` and `gradient_second_decomposition` model the coordinates passed to
  `score-function-gradient` in `sblc-relax-rb`, for a baseline held constant
  across the current draw. The oracle target is `categorical-expectation-gradient`.
* `mdlPrior`, `evidence`, `posterior` and `maximum_mdl_joint_perm` use the prior,
  likelihood and joint weight in `sblc-relax-candidate` and `sblc-relax-better?`.
* `RelaxReceipt` models successful method provenance in `sblc-relax-ok`.
  `confidenceFlag` classifies the variance gain in `sblc-relax-rb`.
  `projectStrict` abstracts the validity, closedness and codec checks into one
  predicate. `schemeErrorResidual` models the unit diagnostic that
  `sblc-relax-project-strict` assigns to rejected nonpositive soft distances.

The probability table is an input to the mathematical model. All moment identities
concern population values and exact arithmetic.
-/

namespace EshkolSBLCRelaxation

/-- A representative with strictly positive denominator. -/
structure Fraction where
  num : Int
  den : Int
  pos : 0 < den

namespace Fraction

def equiv (a b : Fraction) : Prop := a.num * b.den = b.num * a.den

private theorem chain {a b c : Fraction} (h : equiv a b) (g : equiv b c) : equiv a c := by
  apply Int.eq_of_mul_eq_mul_right (Int.ne_of_gt b.pos)
  calc
    _ = (a.num * b.den) * c.den := by ac_rfl
    _ = (b.num * c.den) * a.den := by rw [h]; ac_rfl
    _ = _ := by rw [g]; ac_rfl

instance : Setoid Fraction where
  r := equiv
  iseqv := ⟨fun _ => rfl, Eq.symm, chain⟩

end Fraction

abbrev ExactQ := Quotient (inferInstance : Setoid Fraction)

namespace ExactQ

local notation "⟦" a "⟧" => Quotient.mk (inferInstance : Setoid Fraction) a

def frac (n d : Int) (h : 0 < d) : ExactQ := ⟦⟨n, d, h⟩⟧

instance (n : Nat) : OfNat ExactQ n := ⟨frac n 1 (by decide)⟩

private theorem crossAdd (a b c d : Fraction) (h : a ≈ c) (g : b ≈ d) :
    (a.num*b.den + b.num*a.den)*(c.den*d.den) =
    (c.num*d.den + d.num*c.den)*(a.den*b.den) := by
  simp only [Int.add_mul]
  congr 1
  · calc
      _ = (a.num*c.den)*(b.den*d.den) := by ac_rfl
      _ = _ := by rw [show a.num*c.den = c.num*a.den from h]; ac_rfl
  · calc
      _ = (b.num*d.den)*(a.den*c.den) := by ac_rfl
      _ = _ := by rw [show b.num*d.den = d.num*b.den from g]; ac_rfl

instance : Add ExactQ := ⟨Quotient.lift₂ (fun a b : Fraction =>
  frac (a.num*b.den + b.num*a.den) (a.den*b.den) (Int.mul_pos a.pos b.pos))
  (fun a b c d h g => Quotient.sound (crossAdd a b c d h g))⟩

instance : Mul ExactQ := ⟨Quotient.lift₂ (fun a b : Fraction =>
  frac (a.num*b.num) (a.den*b.den) (Int.mul_pos a.pos b.pos)) (by
    intro a b c d h g
    apply Quotient.sound
    change (a.num*b.num)*(c.den*d.den) = (c.num*d.num)*(a.den*b.den)
    calc
      _ = (a.num*c.den)*(b.num*d.den) := by ac_rfl
      _ = _ := by rw [show a.num*c.den = c.num*a.den from h,
                        show b.num*d.den = d.num*b.den from g]; ac_rfl)⟩

instance : Neg ExactQ := ⟨Quotient.lift (fun a : Fraction => frac (-a.num) a.den a.pos)
  (by intro a b h; apply Quotient.sound
      change -a.num * b.den = -b.num * a.den
      simpa only [Int.neg_mul] using congrArg Neg.neg (show a.num*b.den = b.num*a.den from h))⟩

instance : Sub ExactQ := ⟨fun a b => a + -b⟩

private theorem le_scale {a b d : Int} (hd : 0 < d) : a*d ≤ b*d ↔ a ≤ b := by
  constructor
  · intro h
    apply Decidable.byContradiction
    intro hn
    exact Int.not_le_of_gt (Int.mul_lt_mul_of_pos_right (Int.not_le.mp hn) hd) h
  · exact fun h => Int.mul_le_mul_of_nonneg_right h (Int.le_of_lt hd)

private theorem crossLe (a b c d : Fraction) (h : a ≈ c) (g : b ≈ d) :
    (a.num*b.den ≤ b.num*a.den) = (c.num*d.den ≤ d.num*c.den) := by
  apply propext
  rw [← le_scale (d := c.den*d.den) (Int.mul_pos c.pos d.pos),
      ← le_scale (a := c.num*d.den) (d := a.den*b.den) (Int.mul_pos a.pos b.pos)]
  have shift (x y u v : Int) : (x*y)*(u*v) = (x*u)*(y*v) := by ac_rfl
  rw [shift a.num, show a.num*c.den = c.num*a.den from h]
  have hb : (b.num*a.den)*(c.den*d.den) = (b.num*d.den)*(a.den*c.den) := by ac_rfl
  rw [hb, show b.num*d.den = d.num*b.den from g]
  simp only [Int.mul_assoc, Int.mul_left_comm, Int.mul_comm]

instance : LE ExactQ := ⟨Quotient.lift₂ (fun a b : Fraction => a.num*b.den ≤ b.num*a.den) crossLe⟩
instance : LT ExactQ := ⟨fun a b => ¬ b ≤ a⟩

instance (a b : ExactQ) : Decidable (a ≤ b) :=
  Quotient.recOnSubsingleton a fun _ => Quotient.recOnSubsingleton b fun _ =>
    (inferInstance : Decidable ((_ : Int) ≤ _))
instance (a b : ExactQ) : Decidable (a < b) := inferInstanceAs (Decidable (¬ b ≤ a))

theorem frac_eq {a b c d : Int} {hb : 0 < b} {hd : 0 < d} :
    frac a b hb = frac c d hd ↔ a*d = c*b := ⟨fun h => Quotient.exact h, fun h => Quotient.sound (s := inferInstanceAs (Setoid Fraction)) h⟩

theorem frac_le {a b c d : Int} {hb : 0 < b} {hd : 0 < d} :
    frac a b hb ≤ frac c d hd ↔ a*d ≤ c*b := Iff.rfl

theorem frac_lt {a b c d : Int} {hb : 0 < b} {hd : 0 < d} :
    frac a b hb < frac c d hd ↔ a*d < c*b := Int.not_le

instance (a b : ExactQ) : Decidable (a = b) :=
  Quotient.recOnSubsingleton a fun x => Quotient.recOnSubsingleton b fun y =>
    decidable_of_iff (x.num*y.den = y.num*x.den) frac_eq.symm

theorem frac_add (a b : Fraction) : (⟦a⟧ : ExactQ) + ⟦b⟧ =
    frac (a.num*b.den + b.num*a.den) (a.den*b.den) (Int.mul_pos a.pos b.pos) := rfl
theorem frac_mul (a b : Fraction) : (⟦a⟧ : ExactQ) * ⟦b⟧ =
    frac (a.num*b.num) (a.den*b.den) (Int.mul_pos a.pos b.pos) := rfl
theorem frac_neg (a : Fraction) : -(⟦a⟧ : ExactQ) = frac (-a.num) a.den a.pos := rfl

/-- Reduce a polynomial identity on representatives to integer distributivity and AC. -/
local macro "fraction_identity" : tactic => `(tactic|
  (apply Quotient.sound
   change Fraction.equiv _ _
   dsimp only [Fraction.equiv]
   simp [Int.add_mul, Int.mul_add, Int.mul_assoc, Int.mul_left_comm, Int.mul_comm,
     Int.add_assoc, Int.add_left_comm, Int.add_comm, Int.neg_mul, Int.mul_neg] <;> omega))

theorem add_comm (a b : ExactQ) : a+b = b+a := by
  induction a, b using Quotient.inductionOn₂ with | _ a b => fraction_identity
theorem add_assoc (a b c : ExactQ) : (a+b)+c = a+(b+c) := by
  induction a, b, c using Quotient.inductionOn₃ with | _ a b c => fraction_identity
theorem mul_comm (a b : ExactQ) : a*b = b*a := by
  induction a, b using Quotient.inductionOn₂ with | _ a b => fraction_identity
theorem mul_assoc (a b c : ExactQ) : (a*b)*c = a*(b*c) := by
  induction a, b, c using Quotient.inductionOn₃ with | _ a b c => fraction_identity
theorem mul_add (a b c : ExactQ) : a*(b+c) = a*b+a*c := by
  induction a, b, c using Quotient.inductionOn₃ with | _ a b c => fraction_identity
@[simp] theorem add_zero (a : ExactQ) : a+0 = a := by
  induction a using Quotient.inductionOn with | _ a => fraction_identity
@[simp] theorem mul_one (a : ExactQ) : a*1 = a := by
  induction a using Quotient.inductionOn with | _ a => fraction_identity
@[simp] theorem mul_zero (a : ExactQ) : a*0 = 0 := by
  induction a using Quotient.inductionOn with | _ a => fraction_identity
@[simp] theorem add_neg (a : ExactQ) : a + -a = 0 := by
  induction a using Quotient.inductionOn with | _ a => fraction_identity

@[simp] theorem zero_add (a : ExactQ) : 0+a = a := by rw [add_comm, add_zero]
@[simp] theorem one_mul (a : ExactQ) : 1*a = a := by rw [mul_comm, mul_one]
@[simp] theorem zero_mul (a : ExactQ) : 0*a = 0 := by rw [mul_comm, mul_zero]
theorem add_mul (a b c : ExactQ) : (a+b)*c = a*c+b*c := by
  rw [mul_comm, mul_add, mul_comm c a, mul_comm c b]
theorem add_left_comm (a b c : ExactQ) : a+(b+c) = b+(a+c) := by
  rw [← add_assoc, add_comm a b, add_assoc]
theorem mul_left_comm (a b c : ExactQ) : a*(b*c) = b*(a*c) := by
  rw [← mul_assoc, mul_comm a b, mul_assoc]
theorem sub_def (a b : ExactQ) : a-b = a + -b := rfl
@[simp] theorem neg_mul (a b : ExactQ) : (-a)*b = -(a*b) := by
  induction a, b using Quotient.inductionOn₂ with | _ a b => fraction_identity
@[simp] theorem mul_neg (a b : ExactQ) : a*(-b) = -(a*b) := by
  rw [mul_comm, neg_mul, mul_comm]
@[simp] theorem neg_neg (a : ExactQ) : - -a = a := by
  induction a using Quotient.inductionOn with | _ a => fraction_identity
theorem neg_add (a b : ExactQ) : -(a+b) = -a + -b := by
  induction a, b using Quotient.inductionOn₂ with | _ a b => fraction_identity
theorem mul_sub (a b c : ExactQ) : a*(b-c) = a*b-a*c := by simp [sub_def, mul_add]
@[simp] theorem sub_self (a : ExactQ) : a-a = 0 := add_neg a
theorem zero_ne_one : (0 : ExactQ) ≠ 1 := by decide

/-- Totalized reciprocal. A nonzero numerator supplies its magnitude as denominator. -/
private def reciprocalRep (x : Fraction) : ExactQ :=
  if h : x.num = 0 then 0 else
    frac (x.den * x.num.sign) x.num.natAbs (Int.ofNat_pos.mpr (Int.natAbs_pos.mpr h))

private theorem reciprocal_spec (x : Fraction) (hn : x.num ≠ 0) :
    (⟦x⟧ : ExactQ) * reciprocalRep x = 1 := by
  simp only [reciprocalRep, dif_neg hn]
  apply Quotient.sound
  change (x.num*(x.den*x.num.sign))*1 = 1*(x.den*(x.num.natAbs : Int))
  rw [Int.mul_one, Int.one_mul, Int.mul_left_comm, Int.mul_sign]

private theorem rep_zero (x : Fraction) : (⟦x⟧ : ExactQ) = 0 ↔ x.num = 0 := by
  change frac x.num x.den x.pos = frac 0 1 _ ↔ _
  simp [frac_eq]

private theorem inverse_unique {a b c : ExactQ} (h : a*b = 1) (g : a*c = 1) : b = c := by
  calc
    b = (a*c)*b := by rw [g, one_mul]
    _ = (a*b)*c := by rw [mul_assoc, mul_comm c b, ← mul_assoc]
    _ = c := by rw [h, one_mul]

def inv : ExactQ → ExactQ := Quotient.lift reciprocalRep (by
  intro x y h
  have e : (⟦x⟧ : ExactQ) = ⟦y⟧ := Quotient.sound h
  by_cases hx : x.num = 0
  · have hy := (rep_zero y).mp (e ▸ (rep_zero x).mpr hx)
    simp [reciprocalRep, hx, hy]
  · have hy : y.num ≠ 0 := fun h0 => hx ((rep_zero x).mp (e.trans ((rep_zero y).mpr h0)))
    exact inverse_unique (reciprocal_spec x hx) (e ▸ reciprocal_spec y hy))

instance : Div ExactQ := ⟨fun a b => a * inv b⟩

theorem mul_inv (a : ExactQ) (ha : a ≠ 0) : a * inv a = 1 := by
  induction a using Quotient.inductionOn with
  | _ x => exact reciprocal_spec x (fun h => ha ((rep_zero x).mpr h))

theorem div_mul_cancel (a b : ExactQ) (hb : b ≠ 0) : (a/b)*b = a := by
  change (a*inv b)*b = a
  rw [mul_assoc, mul_comm (inv b), mul_inv b hb, mul_one]

theorem le_refl (a : ExactQ) : a ≤ a := by
  induction a using Quotient.inductionOn with | _ x => exact Int.le_refl _
theorem le_total (a b : ExactQ) : a ≤ b ∨ b ≤ a := by
  induction a, b using Quotient.inductionOn₂ with | _ x y => exact Int.le_total _ _
theorem le_antisymm {a b : ExactQ} (h : a ≤ b) (g : b ≤ a) : a = b := by
  induction a, b using Quotient.inductionOn₂ with
  | _ x y => exact Quotient.sound (show x.num*y.den = y.num*x.den from Int.le_antisymm h g)

theorem le_trans {a b c : ExactQ} (h : a ≤ b) (g : b ≤ c) : a ≤ c := by
  induction a, b, c using Quotient.inductionOn₃ with
  | _ x y z =>
    have h₁ := Int.mul_le_mul_of_nonneg_right (show x.num*y.den ≤ y.num*x.den from h)
      (Int.le_of_lt z.pos)
    have h₂ := Int.mul_le_mul_of_nonneg_right (show y.num*z.den ≤ z.num*y.den from g)
      (Int.le_of_lt x.pos)
    apply (le_scale y.pos).mp
    simp only [Int.mul_assoc, Int.mul_left_comm, Int.mul_comm] at h₁ h₂ ⊢
    exact Int.le_trans h₁ h₂

theorem add_le_add_iff (a b c : ExactQ) : a+b ≤ a+c ↔ b ≤ c := by
  induction a, b, c using Quotient.inductionOn₃ with
  | _ x y z =>
    change (x.num*y.den+y.num*x.den)*(x.den*z.den) ≤
      (x.num*z.den+z.num*x.den)*(x.den*y.den) ↔ y.num*z.den ≤ z.num*y.den
    rw [← le_scale (a := y.num*z.den) (d := x.den*x.den) (Int.mul_pos x.pos x.pos)]
    simp only [Int.add_mul]
    have factor (u v w t : Int) : (u*v)*(w*t) = (u*t)*(v*w) := by ac_rfl
    rw [factor y.num, factor z.num]
    have same : (x.num*y.den)*(x.den*z.den) = (x.num*z.den)*(x.den*y.den) := by ac_rfl
    rw [same, Int.add_le_add_iff_left]

theorem add_lt_add_iff (a b c : ExactQ) : a+b < a+c ↔ b < c :=
  not_congr (add_le_add_iff a c b)
theorem le_of_lt {a b : ExactQ} (h : a < b) : a ≤ b := (le_total a b).resolve_right h
theorem le_add_nonneg (a : ExactQ) {b : ExactQ} (h : 0 ≤ b) : a ≤ a+b := by
  simpa using (add_le_add_iff a 0 b).mpr h
theorem lt_add_pos (a : ExactQ) {b : ExactQ} (h : 0 < b) : a < a+b := by
  simpa using (add_lt_add_iff a 0 b).mpr h

theorem mul_nonneg {a b : ExactQ} (h : 0 ≤ a) (g : 0 ≤ b) : 0 ≤ a*b := by
  induction a, b using Quotient.inductionOn₂ with
  | _ x y =>
    change 0*x.den ≤ x.num*1 at h
    change 0*y.den ≤ y.num*1 at g
    change 0*_ ≤ (x.num*y.num)*1
    simp only [Int.zero_mul, Int.mul_one] at *
    exact Int.mul_nonneg h g

theorem mul_pos {a b : ExactQ} (h : 0 < a) (g : 0 < b) : 0 < a*b := by
  induction a, b using Quotient.inductionOn₂ with
  | _ x y =>
    change ¬ x.num*1 ≤ 0*x.den at h
    change ¬ y.num*1 ≤ 0*y.den at g
    change ¬ (x.num*y.num)*1 ≤ 0*_
    simp only [Int.zero_mul, Int.mul_one, Int.not_le] at *
    exact Int.mul_pos h g

def sq (a : ExactQ) := a*a
theorem sq_nonneg (a : ExactQ) : 0 ≤ sq a := by
  induction a using Quotient.inductionOn with
  | _ x =>
    change 0*_ ≤ (x.num*x.num)*1
    simp only [Int.zero_mul, Int.mul_one]
    cases Int.le_total 0 x.num with
    | inl h => exact Int.mul_nonneg h h
    | inr h => exact Int.mul_nonneg_of_nonpos_of_nonpos h h

theorem sq_pos {a : ExactQ} (ha : a ≠ 0) : 0 < sq a := by
  induction a using Quotient.inductionOn with
  | _ x =>
    have hn : x.num ≠ 0 := fun h => ha ((rep_zero x).mpr h)
    change ¬ (x.num*x.num)*1 ≤ 0*_
    simp only [Int.zero_mul, Int.mul_one, Int.not_le]
    by_cases h : 0 < x.num
    · exact Int.mul_pos h h
    · have g : x.num < 0 := by omega
      exact Int.mul_pos_of_neg_of_neg g g

theorem mul_le_iff_right (a b : ExactQ) {c : ExactQ} (hc : 0 < c) :
    a*c ≤ b*c ↔ a ≤ b := by
  induction a, b, c using Quotient.inductionOn₃ with
  | _ x y z =>
    change ¬ z.num*1 ≤ 0*z.den at hc
    simp only [Int.zero_mul, Int.mul_one, Int.not_le] at hc
    change (x.num*z.num)*(y.den*z.den) ≤ (y.num*z.num)*(x.den*z.den) ↔ _
    have collect (u v : Int) : (u*z.num)*(v*z.den) = (u*v)*(z.num*z.den) := by ac_rfl
    rw [collect, collect]
    exact le_scale (Int.mul_pos hc z.pos)

theorem mul_lt_iff_right (a b : ExactQ) {c : ExactQ} (hc : 0 < c) :
    a*c < b*c ↔ a < b := not_congr (mul_le_iff_right b a hc)

end ExactQ

open ExactQ

/-! ## Finite distributions and conditional moments -/

def sum : {n : Nat} → (Fin n → ExactQ) → ExactQ
  | 0, _ => 0
  | _+1, f => f 0 + sum (fun i => f i.succ)

theorem sum_congr {f g : Fin n → ExactQ} (h : ∀ i, f i = g i) : sum f = sum g :=
  congrArg sum (funext h)

theorem sum_add (f g : Fin n → ExactQ) :
    sum (fun i => f i + g i) = sum f + sum g := by
  induction n with
  | zero => simp [sum]
  | succ n ih =>
    simp only [sum, ih, add_assoc, add_comm, add_left_comm]

theorem sum_scale (c : ExactQ) (f : Fin n → ExactQ) :
    sum (fun i => c * f i) = c * sum f := by
  induction n with
  | zero => simp [sum]
  | succ n ih => simp only [sum, ih, mul_add]

theorem sum_zero (n : Nat) : sum (fun _ : Fin n => (0 : ExactQ)) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [sum, ih]

theorem sum_delta (f : Fin n → ExactQ) (k : Fin n) :
    sum (fun i => if i = k then f i else 0) = f k := by
  induction n with
  | zero => exact Fin.elim0 k
  | succ n ih =>
    refine Fin.cases ?_ (fun j => ?_) k
    · simp [sum, Fin.succ_ne_zero, sum_zero]
    · simp [sum, Fin.succ_inj, Ne.symm (Fin.succ_ne_zero j), ih (fun i => f i.succ) j]

theorem sum_nonneg {f : Fin n → ExactQ} (h : ∀ i, 0 ≤ f i) : 0 ≤ sum f := by
  induction n with
  | zero => exact le_refl _
  | succ n ih => exact le_trans (h 0) (le_add_nonneg _ (ih (fun j => h j.succ)))

theorem sum_pos_of_entry {f : Fin n → ExactQ} (h : ∀ i, 0 ≤ f i)
    (k : Fin n) (hk : 0 < f k) : 0 < sum f := by
  induction n with
  | zero => exact Fin.elim0 k
  | succ n ih =>
    refine Fin.cases (motive := fun i => 0 < f i → 0 < sum f) ?_ (fun j hj => ?_) k hk
    · intro h0 hn
      exact h0 (le_trans (le_add_nonneg _ (sum_nonneg (fun j => h j.succ))) hn)
    · have ht := ih (fun j => h j.succ) j hj
      intro hn
      have ha := le_add_nonneg (sum (fun j => f j.succ)) (h 0)
      rw [add_comm] at ha
      exact ht (le_trans ha hn)

structure Distribution (n : Nat) where
  weight : Fin n → ExactQ
  nonneg : ∀ i, 0 ≤ weight i
  normalized : sum weight = 1

def mean (q : Distribution n) (f : Fin n → ExactQ) : ExactQ :=
  sum (fun i => q.weight i * f i)

theorem mean_congr (q : Distribution n) {f g : Fin n → ExactQ} (h : ∀ i, f i = g i) :
    mean q f = mean q g := congrArg (mean q) (funext h)

theorem mean_add (q : Distribution n) (f g : Fin n → ExactQ) :
    mean q (fun i => f i + g i) = mean q f + mean q g := by
  simp only [mean, mul_add, sum_add]

theorem mean_scale (q : Distribution n) (c : ExactQ) (f : Fin n → ExactQ) :
    mean q (fun i => c*f i) = c*mean q f := by
  unfold mean
  rw [← sum_scale]
  exact sum_congr fun _ => mul_left_comm _ _ _

theorem mean_const (q : Distribution n) (c : ExactQ) : mean q (fun _ => c) = c := by
  unfold mean
  simp only [mul_comm _ c, sum_scale, q.normalized, mul_one]

theorem mean_nonneg (q : Distribution n) {f : Fin n → ExactQ} (h : ∀ i, 0 ≤ f i) :
    0 ≤ mean q f := sum_nonneg fun i => mul_nonneg (q.nonneg i) (h i)

theorem mean_pos_of_entry (q : Distribution n) {f : Fin n → ExactQ} (h : ∀ i, 0 ≤ f i)
    (k : Fin n) (hq : 0 < q.weight k) (hf : 0 < f k) : 0 < mean q f :=
  sum_pos_of_entry (fun i => mul_nonneg (q.nonneg i) (h i)) k (mul_pos hq hf)

def variance (q : Distribution n) (f : Fin n → ExactQ) : ExactQ :=
  mean q (fun i => sq (f i - mean q f))

theorem variance_nonneg (q : Distribution n) (f : Fin n → ExactQ) : 0 ≤ variance q f :=
  mean_nonneg q fun _ => sq_nonneg _

theorem centered_mean (q : Distribution n) (f : Fin n → ExactQ) :
    mean q (fun i => f i - mean q f) = 0 := by
  simp only [sub_def, mean_add, mean_const, add_neg]

/-- A constant multiple of a within-row deviation is orthogonal to every row constant. -/
theorem centered_cross (q : Distribution n) (f : Fin n → ExactQ) (c : ExactQ) :
    mean q (fun i => c*(f i - mean q f)) = 0 := by
  rw [mean_scale, centered_mean, mul_zero]

theorem sq_add (a b : ExactQ) : sq (a+b) = sq a + ((b+b)*a + sq b) := by
  simp only [sq, add_mul, mul_add, mul_comm, add_comm, add_left_comm, add_assoc]

theorem recenter (x μ b : ExactQ) : x-b = (x-μ)+(μ-b) := by
  simp only [sub_def, add_assoc]
  rw [← add_assoc (-μ), add_comm (-μ) μ, add_neg, zero_add]

/-- The squared error about any center splits into squared bias and centered variance. -/
theorem mean_squared_error (q : Distribution n) (f : Fin n → ExactQ) (b : ExactQ) :
    mean q (fun i => sq (f i-b)) = sq (mean q f-b) + variance q f := by
  conv => lhs; arg 2; intro i; rw [recenter (f i) (mean q f) b, sq_add]
  rw [mean_add, mean_add, centered_cross, mean_const, zero_add, add_comm]
  rfl

def conditionalMean (p : Fin m → Distribution k) (r : Fin m → Fin k → ExactQ) (i : Fin m) :=
  mean (p i) (r i)

/-- Direct integration against the joint atom mass qᵢ pᵢⱼ. -/
def jointMean (q : Distribution m) (p : Fin m → Distribution k)
    (r : Fin m → Fin k → ExactQ) :=
  sum (fun i => sum (fun j => (q.weight i * (p i).weight j) * r i j))

theorem total_expectation (q : Distribution m) (p : Fin m → Distribution k)
    (r : Fin m → Fin k → ExactQ) : jointMean q p r = mean q (conditionalMean p r) := by
  apply sum_congr
  intro i
  simp only [conditionalMean, mean, mul_assoc, sum_scale]

def jointVariance (q : Distribution m) (p : Fin m → Distribution k)
    (r : Fin m → Fin k → ExactQ) :=
  jointMean q p (fun i j => sq (r i j - jointMean q p r))

theorem joint_cross_zero (q : Distribution m) (p : Fin m → Distribution k)
    (r : Fin m → Fin k → ExactQ) (c : Fin m → ExactQ) :
    jointMean q p (fun i j => c i * (r i j - conditionalMean p r i)) = 0 := by
  rw [total_expectation]
  change mean q (fun i => mean (p i) (fun j => c i * (r i j - mean (p i) (r i)))) = 0
  simp only [centered_cross, mean_const]

theorem total_variance (q : Distribution m) (p : Fin m → Distribution k)
    (r : Fin m → Fin k → ExactQ) :
    jointVariance q p r = variance q (conditionalMean p r) +
      mean q (fun i => variance (p i) (r i)) := by
  unfold jointVariance
  unfold variance
  rw [total_expectation, total_expectation q p r, ← mean_add]
  exact mean_congr q fun i => mean_squared_error (p i) (r i) _

theorem rao_blackwell (q : Distribution m) (p : Fin m → Distribution k)
    (r : Fin m → Fin k → ExactQ) : variance q (conditionalMean p r) ≤ jointVariance q p r := by
  rw [total_variance]
  exact le_add_nonneg _ (mean_nonneg q fun i => variance_nonneg (p i) (r i))

theorem rao_blackwell_strict (q : Distribution m) (p : Fin m → Distribution k)
    (r : Fin m → Fin k → ExactQ)
    (h : ∃ i, 0 < q.weight i ∧ 0 < variance (p i) (r i)) :
    variance q (conditionalMean p r) < jointVariance q p r := by
  obtain ⟨i, hq, hv⟩ := h
  rw [total_variance]
  exact lt_add_pos _ (mean_pos_of_entry q (fun j => variance_nonneg (p j) (r j)) i hq hv)


/-! ## A closed fair-choice program -/

/-- De Bruijn variables are bounded by the surrounding binder depth. -/
inductive Term : Nat → Type where
  | var : Fin d → Term d
  | lam : Term (d+1) → Term d
  | app : Term d → Term d → Term d
  | choice : Term d → Term d → Term d
  deriving DecidableEq

def identityTerm : Term 0 := .lam (.var 0)
def constantTerm : Term 0 := .lam (.lam (.var 1))
def fairChoice : Term 0 := .choice identityTerm constantTerm

def drawBranch (t : Term 0) (coin : Fin 2) : Term 0 :=
  match t with
  | .choice left right => if coin = 0 then left else right
  | other => other

def fairReward (_ : Fin 1) (coin : Fin 2) : ExactQ :=
  if drawBranch fairChoice coin = identityTerm then 1 else 0

def singleProposal : Distribution 1 :=
  ⟨fun _ => 1, fun _ => (of_decide_eq_true rfl : (0 : ExactQ) ≤ 1), by decide⟩
def fairCoin : Distribution 2 :=
  ⟨fun _ => 1/2, fun _ => (of_decide_eq_true rfl : (0 : ExactQ) ≤ 1/2), by decide⟩

theorem fair_choice_moments (i : Fin 1) :
    conditionalMean (fun _ => fairCoin) fairReward i = 1/2 ∧
    variance fairCoin (fairReward i) = 1/4 ∧ (0 : ExactQ) < 1/4 := by
  exact (by decide : conditionalMean (fun _ => fairCoin) fairReward 0 = 1/2 ∧
    variance fairCoin (fairReward 0) = 1/4 ∧ (0 : ExactQ) < 1/4)

theorem fair_choice_strict :
    variance singleProposal (conditionalMean (fun _ => fairCoin) fairReward) <
    jointVariance singleProposal (fun _ => fairCoin) fairReward := by
  apply rao_blackwell_strict
  refine ⟨0, by decide, ?_⟩
  rw [(fair_choice_moments 0).2.1]
  exact (fair_choice_moments 0).2.2

/-! ## Categorical logit scores -/

def score (q : Distribution n) (i k : Fin n) : ExactQ :=
  (if i = k then 1 else 0) - q.weight k

theorem mean_indicator (q : Distribution n) (f : Fin n → ExactQ) (k : Fin n) :
    mean q (fun i => if i = k then f i else 0) = q.weight k * f k := by
  unfold mean
  rw [← sum_delta (fun i => q.weight i * f i) k]
  apply sum_congr
  intro i
  split <;> simp_all

theorem score_weighted_mean (q : Distribution n) (f : Fin n → ExactQ) (k : Fin n) :
    mean q (fun i => f i * score q i k) = q.weight k * (f k - mean q f) := by
  calc
    _ = mean q (fun i => (if i = k then f i else 0) + (-q.weight k) * f i) := by
      apply mean_congr
      intro i
      simp only [score, sub_def, mul_add, mul_comm (f i) (-q.weight k)]
      split <;> simp
    _ = _ := by rw [mean_add, mean_indicator, mean_scale, neg_mul, mul_sub, sub_def]

theorem zero_mean_score (q : Distribution n) (k : Fin n) :
    sum (fun i => q.weight i * score q i k) = 0 := by
  have h := score_weighted_mean q (fun _ => 1) k
  rw [mean_const, sub_self, mul_zero] at h
  simpa only [one_mul, mean] using h

theorem baseline_zero_bias (q : Distribution n) (b : ExactQ) (k : Fin n) :
    sum (fun i => q.weight i * b * score q i k) = 0 := by
  have h := score_weighted_mean q (fun _ => b) k
  rw [mean_const, sub_self, mul_zero] at h
  simpa only [mean, mul_assoc] using h

theorem mean_shift (q : Distribution n) (f : Fin n → ExactQ) (b : ExactQ) :
    mean q (fun i => f i-b) = mean q f-b := by
  simp only [sub_def, mean_add, mean_const]

theorem shift_cancel (x y b : ExactQ) : (x-b)-(y-b) = x-y := by
  simp only [sub_def, neg_add, neg_neg, add_assoc]
  rw [add_comm (-y) b, ← add_assoc (-b), add_comm (-b) b, add_neg, zero_add]

def gradientSample (reward baseline s : ExactQ) := (reward-baseline)*s

theorem reinforce (q : Distribution n) (f : Fin n → ExactQ) (b : ExactQ) (k : Fin n) :
    sum (fun i => q.weight i * (f i-b) * score q i k) = q.weight k * (f k-mean q f) := by
  have h := score_weighted_mean q (fun i => f i-b) k
  rw [mean_shift, shift_cancel] at h
  simpa only [mean, mul_assoc] using h

theorem conditional_gradient (p : Distribution n) (f : Fin n → ExactQ) (b s : ExactQ) :
    mean p (fun j => gradientSample (f j) b s) = gradientSample (mean p f) b s := by
  simp only [gradientSample, mul_comm _ s, mean_scale, mean_shift]

theorem rb_gradient_mean (q : Distribution m) (p : Fin m → Distribution n)
    (r : Fin m → Fin n → ExactQ) (b : ExactQ) (k : Fin m) :
    jointMean q p (fun i j => gradientSample (r i j) b (score q i k)) =
    mean q (fun i => gradientSample (conditionalMean p r i) b (score q i k)) := by
  rw [total_expectation]
  exact mean_congr q fun i => conditional_gradient (p i) (r i) b _

theorem rb_reinforce (q : Distribution m) (p : Fin m → Distribution n)
    (r : Fin m → Fin n → ExactQ) (b : ExactQ) (k : Fin m) :
    jointMean q p (fun i j => gradientSample (r i j) b (score q i k)) =
    q.weight k * (conditionalMean p r k - mean q (conditionalMean p r)) := by
  rw [rb_gradient_mean]
  simpa only [gradientSample, mean, mul_assoc] using reinforce q (conditionalMean p r) b k

theorem gradient_square (x b s : ExactQ) : sq (gradientSample x b s) = sq s * sq (x-b) := by
  simp only [sq, gradientSample, mul_assoc, mul_left_comm, mul_comm]

theorem conditional_gradient_second (p : Distribution n) (f : Fin n → ExactQ) (b s : ExactQ) :
    mean p (fun j => sq (gradientSample (f j) b s)) =
    sq (gradientSample (mean p f) b s) + sq s * variance p f := by
  simp only [gradient_square, mean_scale, mean_squared_error, mul_add]

def noisySecond (q : Distribution m) (p : Fin m → Distribution n)
    (r : Fin m → Fin n → ExactQ) (b : ExactQ) (k : Fin m) :=
  jointMean q p (fun i j => sq (gradientSample (r i j) b (score q i k)))

def conditionedSecond (q : Distribution m) (p : Fin m → Distribution n)
    (r : Fin m → Fin n → ExactQ) (b : ExactQ) (k : Fin m) :=
  mean q (fun i => sq (gradientSample (conditionalMean p r i) b (score q i k)))

theorem gradient_second_decomposition (q : Distribution m) (p : Fin m → Distribution n)
    (r : Fin m → Fin n → ExactQ) (b : ExactQ) (k : Fin m) :
    noisySecond q p r b k = conditionedSecond q p r b k +
      mean q (fun i => sq (score q i k) * variance (p i) (r i)) := by
  unfold noisySecond conditionedSecond
  rw [total_expectation, ← mean_add]
  exact mean_congr q fun i => conditional_gradient_second (p i) (r i) b _

theorem rb_gradient_second_le (q : Distribution m) (p : Fin m → Distribution n)
    (r : Fin m → Fin n → ExactQ) (b : ExactQ) (k : Fin m) :
    conditionedSecond q p r b k ≤ noisySecond q p r b k := by
  rw [gradient_second_decomposition]
  exact le_add_nonneg _ (mean_nonneg q fun i => mul_nonneg (sq_nonneg _) (variance_nonneg _ (r i)))

theorem rb_gradient_second_lt (q : Distribution m) (p : Fin m → Distribution n)
    (r : Fin m → Fin n → ExactQ) (b : ExactQ) (k : Fin m)
    (h : ∃ i, 0 < q.weight i ∧ score q i k ≠ 0 ∧ 0 < variance (p i) (r i)) :
    conditionedSecond q p r b k < noisySecond q p r b k := by
  obtain ⟨i, hq, hs, hv⟩ := h
  rw [gradient_second_decomposition]
  exact lt_add_pos _ (mean_pos_of_entry q
    (fun j => mul_nonneg (sq_nonneg _) (variance_nonneg _ (r j))) i hq (mul_pos (sq_pos hs) hv))


/-- Summing the coordinate identities gives the second moment of the Euclidean
    norm of the gradient. Every contribution to the gap is nonnegative. -/
theorem gradient_norm_second_decomposition (q : Distribution m) (p : Fin m → Distribution n)
    (r : Fin m → Fin n → ExactQ) (b : ExactQ) :
    sum (noisySecond q p r b) = sum (conditionedSecond q p r b) +
      sum (fun k => mean q (fun i => sq (score q i k) * variance (p i) (r i))) := by
  rw [← sum_add]
  exact sum_congr (gradient_second_decomposition q p r b)

theorem gradient_norm_second_le (q : Distribution m) (p : Fin m → Distribution n)
    (r : Fin m → Fin n → ExactQ) (b : ExactQ) :
    sum (conditionedSecond q p r b) ≤ sum (noisySecond q p r b) := by
  rw [gradient_norm_second_decomposition]
  exact le_add_nonneg _ (sum_nonneg fun _ => mean_nonneg q fun i =>
    mul_nonneg (sq_nonneg _) (variance_nonneg _ (r i)))

theorem gradient_norm_second_lt (q : Distribution m) (p : Fin m → Distribution n)
    (r : Fin m → Fin n → ExactQ) (b : ExactQ)
    (h : ∃ i k, 0 < q.weight i ∧ score q i k ≠ 0 ∧ 0 < variance (p i) (r i)) :
    sum (conditionedSecond q p r b) < sum (noisySecond q p r b) := by
  obtain ⟨i, k, hq, hs, hv⟩ := h
  have nn (c j : Fin m) : 0 ≤ sq (score q j c) * variance (p j) (r j) :=
    mul_nonneg (sq_nonneg _) (variance_nonneg _ _)
  rw [gradient_norm_second_decomposition]
  exact lt_add_pos _ (sum_pos_of_entry (fun c => mean_nonneg q (nn c)) k
    (mean_pos_of_entry q (nn k) i hq (mul_pos (sq_pos hs) hv)))

/-! ## Exact catalogue semantics -/

/-- Kraft weight of a codeword of the supplied length. A catalogue of closed
    syntax uses `α := Term 0`; the proofs also apply to any encoding of it. -/
def mdlPrior (length : α → Nat) (t : α) : ExactQ :=
  frac 1 (Int.ofNat (2 ^ length t)) (Int.ofNat_pos.mpr (Nat.pow_pos (by decide : 0 < (2 : Nat))))

theorem mdlPrior_pos (length : α → Nat) (t : α) : 0 < mdlPrior length t := by
  unfold mdlPrior
  change ¬ (1 : Int)*1 ≤ 0 * Int.ofNat (2 ^ length t)
  simp

def evidence (w L : α → ExactQ) : List α → ExactQ
  | [] => 0
  | t :: ts => w t * L t + evidence w L ts

/-- Conditional probability requires nonzero evidence, including at the API boundary. -/
def posterior (w L : α → ExactQ) (ts : List α) (_ : evidence w L ts ≠ 0) (t : α) :=
  w t * L t / evidence w L ts

theorem evidence_perm (w L : α → ExactQ) {xs ys : List α} (h : xs.Perm ys) :
    evidence w L xs = evidence w L ys := by
  induction h with
  | nil => rfl
  | cons _ _ ih => simp only [evidence, ih]
  | swap => exact add_left_comm _ _ _
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

theorem posterior_perm (w L : α → ExactQ) {xs ys : List α} (h : xs.Perm ys)
    (hx : evidence w L xs ≠ 0) (t : α) :
    posterior w L xs hx t = posterior w L ys ((evidence_perm w L h) ▸ hx) t := by
  simp only [posterior, evidence_perm w L h]

def greaterWeight (x y : ExactQ) := if x ≤ y then y else x

theorem greaterWeight_le (a b c : ExactQ) : greaterWeight a b ≤ c ↔ a ≤ c ∧ b ≤ c := by
  unfold greaterWeight
  split
  next h => exact ⟨fun g => ⟨le_trans h g, g⟩, And.right⟩
  next h => exact ⟨fun g => ⟨g, le_trans ((le_total a b).resolve_left h) g⟩, And.left⟩

/-- The empty catalogue has weight zero. For positive priors a nonempty catalogue
    attains this fold at a program, even when several programs tie. -/
def maximumWeight (w : α → ExactQ) : List α → ExactQ
  | [] => 0
  | t :: ts => greaterWeight (w t) (maximumWeight w ts)

theorem maximumWeight_le (w : α → ExactQ) (xs : List α) (c : ExactQ) :
    maximumWeight w xs ≤ c ↔ 0 ≤ c ∧ ∀ t ∈ xs, w t ≤ c := by
  induction xs with
  | nil => simp [maximumWeight]
  | cons x xs ih =>
    simp only [maximumWeight, greaterWeight_le, ih, List.mem_cons]
    constructor
    · rintro ⟨hx, h0, hs⟩
      exact ⟨h0, fun t ht => ht.elim (fun e => e ▸ hx) (hs t)⟩
    · rintro ⟨h0, hs⟩
      exact ⟨hs x (.inl rfl), h0, fun t ht => hs t (.inr ht)⟩

theorem maximumWeight_attained (w : α → ExactQ) (positive : ∀ t, 0 < w t)
    (x : α) (xs : List α) : ∃ t ∈ x :: xs, maximumWeight w (x :: xs) = w t := by
  induction xs generalizing x with
  | nil =>
    refine ⟨x, .head _, ?_⟩
    exact if_neg (positive x)
  | cons y ys ih =>
    change ∃ t ∈ x :: y :: ys, (if w x ≤ maximumWeight w (y :: ys) then _ else _) = w t
    split
    · obtain ⟨t, ht, he⟩ := ih y
      exact ⟨t, .tail _ ht, he⟩
    · exact ⟨x, .head _, rfl⟩

theorem maximumWeight_perm (w : α → ExactQ) {xs ys : List α} (h : xs.Perm ys) :
    maximumWeight w xs = maximumWeight w ys := by
  have mono {us vs : List α} (sub : us ⊆ vs) : maximumWeight w us ≤ maximumWeight w vs := by
    obtain ⟨h0, upper⟩ := (maximumWeight_le w vs _).mp (le_refl _)
    exact (maximumWeight_le w us _).mpr ⟨h0, fun t ht => upper t (sub ht)⟩
  exact le_antisymm (mono h.subset) (mono h.symm.subset)

theorem maximum_mdl_joint_perm (length : α → Nat) (L : α → ExactQ) {xs ys : List α}
    (h : xs.Perm ys) : maximumWeight (fun t => mdlPrior length t * L t) xs =
      maximumWeight (fun t => mdlPrior length t * L t) ys := maximumWeight_perm _ h

theorem maximum_mdl_perm (length : α → Nat) {xs ys : List α} (h : xs.Perm ys) :
    maximumWeight (mdlPrior length) xs = maximumWeight (mdlPrior length) ys :=
  maximumWeight_perm _ h

structure Proposed (α : Type u) where
  program : α
  proposalWeight : ExactQ
  positive : 0 < proposalWeight

def attach (q : α → ExactQ) (hq : ∀ t, 0 < q t) (t : α) : Proposed α := ⟨t, q t, hq t⟩

def verifierValue (v : α → ExactQ) (entry : Proposed α) := v entry.program

theorem annotation_pointwise (w L q : α → ExactQ) (hq : ∀ t, 0 < q t) (t : α) :
    verifierValue w (attach q hq t) = w t ∧ verifierValue L (attach q hq t) = L t :=
  ⟨rfl, rfl⟩

theorem annotated_evidence (w L q : α → ExactQ) (hq : ∀ t, 0 < q t) (ts : List α) :
    evidence (verifierValue w) (verifierValue L) (ts.map (attach q hq)) = evidence w L ts := by
  induction ts with
  | nil => rfl
  | cons x xs ih => simp only [List.map_cons, evidence, ih]; rfl

/-- Proposal-dependent traversal can be any permutation, including arbitrary tie order. -/
theorem reordered_proposal_evidence (w L q : α → ExactQ) (hq : ∀ t, 0 < q t)
    (ts : List α) (ordered : List (Proposed α)) (h : ordered.Perm (ts.map (attach q hq))) :
    evidence (verifierValue w) (verifierValue L) ordered = evidence w L ts :=
  (evidence_perm _ _ h).trans (annotated_evidence w L q hq ts)

/-! ## Receipt provenance, projection and confidence -/

inductive ProposalMethod where
  | exactOracle | reinforceRB | gumbelSoftmax
  deriving DecidableEq

inductive ReceiptKind where
  | exact | approximate
  deriving DecidableEq

/-- Diagnostics are stored only by the two approximate methods. Exact diagnostics
    are determined by construction, so a caller cannot supply conflicting fields. -/
inductive RelaxReceipt where
  | exactOracle
  | reinforceRB (projectionResidual varianceBound : ExactQ)
  | gumbelSoftmax (projectionResidual varianceBound : ExactQ)

namespace RelaxReceipt

def method : RelaxReceipt → ProposalMethod
  | .exactOracle => .exactOracle
  | .reinforceRB .. => .reinforceRB
  | .gumbelSoftmax .. => .gumbelSoftmax

def kind : RelaxReceipt → ReceiptKind
  | .exactOracle => .exact
  | .reinforceRB .. | .gumbelSoftmax .. => .approximate

def projectionResidual : RelaxReceipt → ExactQ
  | .exactOracle => 0
  | .reinforceRB r _ | .gumbelSoftmax r _ => r

def varianceBound : RelaxReceipt → ExactQ
  | .exactOracle => 0
  | .reinforceRB _ v | .gumbelSoftmax _ v => v

def isExact (r : RelaxReceipt) : Bool := decide (r.kind = .exact)

theorem exact_iff (r : RelaxReceipt) : r.kind = .exact ↔
    r.method = .exactOracle ∧ r.projectionResidual = 0 ∧ r.varianceBound = 0 := by
  cases r <;> simp [kind, method, projectionResidual, varianceBound]

theorem approximate_methods (r : RelaxReceipt)
    (h : r.method = .reinforceRB ∨ r.method = .gumbelSoftmax) :
    r.kind = .approximate ∧ r.isExact = false := by
  cases r <;> simp_all [kind, method, isExact]

end RelaxReceipt

inductive Projection (α : Type u) where
  | ok (term : α)
  | projectionError (residual : ExactQ)

/-- The residual is the caller's exact diagnostic at the original candidate. -/
def projectStrict (t : α) (isClosed constraint : α → Bool) (residual : α → ExactQ) : Projection α :=
  if isClosed t && constraint t then .ok t else .projectionError (residual t)

theorem projectStrict_ok (t u : α) (isClosed constraint : α → Bool) (residual : α → ExactQ) :
    projectStrict t isClosed constraint residual = .ok u ↔
      t = u ∧ isClosed t = true ∧ constraint t = true := by
  cases hc : isClosed t <;> cases hh : constraint t <;> simp [projectStrict, hc, hh]

theorem projectStrict_iff (t : α) (isClosed constraint : α → Bool) (residual : α → ExactQ) :
    projectStrict t isClosed constraint residual = .ok t ↔
      isClosed t = true ∧ constraint t = true := by
  simpa only [true_and, eq_self] using projectStrict_ok t t isClosed constraint residual

theorem projectStrict_error (t : α) (isClosed constraint : α → Bool) (residual : α → ExactQ)
    (h : isClosed t = false ∨ constraint t = false) :
    projectStrict t isClosed constraint residual = .projectionError (residual t) := by
  cases h with
  | inl hc => simp [projectStrict, hc]
  | inr hh => simp [projectStrict, hh]

inductive Confidence where
  | lowConfidence | high | exactZeroVariance
  deriving DecidableEq

/-- Compare the gain to a fraction of the reference variance without division. -/
def confidenceFlag (baseVar postVar threshold : ExactQ) : Confidence :=
  if 0 < baseVar ∧ baseVar-postVar < threshold*baseVar then .lowConfidence
  else if postVar = 0 then .exactZeroVariance else .high

theorem lowConfidence_iff (baseVar postVar threshold : ExactQ) :
    confidenceFlag baseVar postVar threshold = .lowConfidence ↔
      0 < baseVar ∧ baseVar-postVar < threshold*baseVar := by
  unfold confidenceFlag
  split
  · simp_all
  · split <;> simp_all

/-- On a positive reference the Scheme ratio and the division-free gate agree. -/
theorem confidence_ratio_lt {base post threshold : ExactQ} (hb : 0 < base) :
    1-post/base < threshold ↔ base-post < threshold*base := by
  have hn : base ≠ 0 := fun e => hb (e ▸ le_refl _)
  rw [← mul_lt_iff_right (1-post/base) threshold hb]
  simp only [sub_def, add_mul, one_mul, neg_mul, div_mul_cancel _ _ hn]

/-- At admitted thresholds the zero-variance branch of the Scheme receipt
    agrees with testing the gain first, including at a zero reference. -/
theorem confidence_scheme (base post threshold : ExactQ) (ht : threshold ≤ 1) :
    confidenceFlag base post threshold =
      (if post = 0 then .exactZeroVariance
       else if 0 < base ∧ 1-post/base < threshold then .lowConfidence else .high) := by
  by_cases hp : post = 0
  · subst post
    have noLow : ¬ (0 < base ∧ base-0 < threshold*base) := by
      rintro ⟨hb, h⟩
      have upper := (mul_le_iff_right threshold 1 hb).mpr ht
      have z : -(0 : ExactQ) = 0 := by decide
      simp only [one_mul] at upper
      simp only [sub_def, z, add_zero] at h
      exact h upper
    simp [confidenceFlag, noLow]
  · by_cases hb : 0 < base
    · simp [confidenceFlag, hp, hb, confidence_ratio_lt hb]
    · simp [confidenceFlag, hp, hb]

/-- Scheme's error receipt accounts for a violated hard constraint at zero soft distance. -/
def schemeErrorResidual (softDistance : ExactQ) : ExactQ :=
  if 0 < softDistance then softDistance else 1

theorem scheme_error_agrees_iff (r : ExactQ) : schemeErrorResidual r = r ↔ 0 < r := by
  unfold schemeErrorResidual
  split
  next h => simp [h]
  next h =>
    constructor
    · intro e
      rw [← e]
      decide
    · exact fun hp => False.elim (h hp)

theorem scheme_zero_error_residual : schemeErrorResidual 0 = 1 ∧ (1 : ExactQ) ≠ 0 := by decide

/-! ## Kernel dependency audit -/

#print axioms Fraction.chain
#print axioms ExactQ.crossAdd
#print axioms ExactQ.le_scale
#print axioms ExactQ.crossLe
#print axioms ExactQ.frac_eq
#print axioms ExactQ.frac_le
#print axioms ExactQ.frac_lt
#print axioms ExactQ.frac_add
#print axioms ExactQ.frac_mul
#print axioms ExactQ.frac_neg
#print axioms ExactQ.add_comm
#print axioms ExactQ.add_assoc
#print axioms ExactQ.mul_comm
#print axioms ExactQ.mul_assoc
#print axioms ExactQ.mul_add
#print axioms ExactQ.add_zero
#print axioms ExactQ.mul_one
#print axioms ExactQ.mul_zero
#print axioms ExactQ.add_neg
#print axioms ExactQ.zero_add
#print axioms ExactQ.one_mul
#print axioms ExactQ.zero_mul
#print axioms ExactQ.add_mul
#print axioms ExactQ.add_left_comm
#print axioms ExactQ.mul_left_comm
#print axioms ExactQ.sub_def
#print axioms ExactQ.neg_mul
#print axioms ExactQ.mul_neg
#print axioms ExactQ.neg_neg
#print axioms ExactQ.neg_add
#print axioms ExactQ.mul_sub
#print axioms ExactQ.sub_self
#print axioms ExactQ.zero_ne_one
#print axioms ExactQ.reciprocal_spec
#print axioms ExactQ.rep_zero
#print axioms ExactQ.inverse_unique
#print axioms ExactQ.mul_inv
#print axioms ExactQ.div_mul_cancel
#print axioms ExactQ.le_refl
#print axioms ExactQ.le_total
#print axioms ExactQ.le_antisymm
#print axioms ExactQ.le_trans
#print axioms ExactQ.add_le_add_iff
#print axioms ExactQ.add_lt_add_iff
#print axioms ExactQ.le_of_lt
#print axioms ExactQ.le_add_nonneg
#print axioms ExactQ.lt_add_pos
#print axioms ExactQ.mul_nonneg
#print axioms ExactQ.mul_pos
#print axioms ExactQ.sq_nonneg
#print axioms ExactQ.sq_pos
#print axioms ExactQ.mul_le_iff_right
#print axioms ExactQ.mul_lt_iff_right
#print axioms sum_congr
#print axioms sum_add
#print axioms sum_scale
#print axioms sum_zero
#print axioms sum_delta
#print axioms sum_nonneg
#print axioms sum_pos_of_entry
#print axioms mean_congr
#print axioms mean_add
#print axioms mean_scale
#print axioms mean_const
#print axioms mean_nonneg
#print axioms mean_pos_of_entry
#print axioms variance_nonneg
#print axioms centered_mean
#print axioms centered_cross
#print axioms sq_add
#print axioms recenter
#print axioms mean_squared_error
#print axioms total_expectation
#print axioms joint_cross_zero
#print axioms total_variance
#print axioms rao_blackwell
#print axioms rao_blackwell_strict
#print axioms fair_choice_moments
#print axioms fair_choice_strict
#print axioms mean_indicator
#print axioms score_weighted_mean
#print axioms zero_mean_score
#print axioms baseline_zero_bias
#print axioms mean_shift
#print axioms shift_cancel
#print axioms reinforce
#print axioms conditional_gradient
#print axioms rb_gradient_mean
#print axioms rb_reinforce
#print axioms gradient_square
#print axioms conditional_gradient_second
#print axioms gradient_second_decomposition
#print axioms rb_gradient_second_le
#print axioms rb_gradient_second_lt
#print axioms gradient_norm_second_decomposition
#print axioms gradient_norm_second_le
#print axioms gradient_norm_second_lt
#print axioms mdlPrior_pos
#print axioms evidence_perm
#print axioms posterior_perm
#print axioms greaterWeight_le
#print axioms maximumWeight_le
#print axioms maximumWeight_attained
#print axioms maximumWeight_perm
#print axioms maximum_mdl_joint_perm
#print axioms maximum_mdl_perm
#print axioms annotation_pointwise
#print axioms annotated_evidence
#print axioms reordered_proposal_evidence
#print axioms RelaxReceipt.exact_iff
#print axioms RelaxReceipt.approximate_methods
#print axioms projectStrict_ok
#print axioms projectStrict_iff
#print axioms projectStrict_error
#print axioms lowConfidence_iff
#print axioms confidence_ratio_lt
#print axioms confidence_scheme
#print axioms scheme_error_agrees_iff
#print axioms scheme_zero_error_residual

end EshkolSBLCRelaxation

