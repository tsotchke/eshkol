/-! # Dessins, cellular Euler characteristic, and the characteristic-p tilt -/

namespace EshkolGrothendieckEra3EsquissePerfectoid

/-- A periodic presentation of a permutation of the darts. A common period
    need not be the least one. Iterating one fewer time gives the inverse. -/
def iterate {α : Type u} (f : α → α) (x : α) : Nat → α
  | 0 => x
  | k+1 => f (iterate f x k)

theorem iterate_add {α : Type u} (f : α → α) (x : α) (n m : Nat) :
    iterate f x (n + m) = iterate f (iterate f x n) m := by
  induction m with
  | zero => rfl
  | succ m ih => exact congrArg f ih

structure Rotation (d : Nat) where
  act : Fin d → Fin d
  period : Nat
  positive : 0 < period
  closes : ∀ x, iterate act x period = x

namespace Rotation
variable {d : Nat} (s : Rotation d)

theorem reduced (x : Fin d) (k : Nat) : iterate s.act x k = iterate s.act x (k % s.period) := by
  have laps (n : Nat) : iterate s.act x (s.period * n) = x := by
    induction n with
    | zero => rfl
    | succ n ih => rw [Nat.mul_succ, iterate_add, ih, s.closes]
  calc
    _ = iterate s.act x (s.period * (k / s.period) + k % s.period) := congrArg _ (Nat.div_add_mod ..).symm
    _ = _ := by rw [iterate_add, laps]

def inverse (x : Fin d) := iterate s.act x (s.period - 1)

theorem inverse_laws (x : Fin d) : s.act (s.inverse x) = x ∧ s.inverse (s.act x) = x := by
  have h := s.positive
  have hl : s.period - 1 + 1 = s.period := by omega
  have hr : 1 + (s.period - 1) = s.period := by omega
  constructor
  · exact (congrArg (iterate s.act x) hl).trans (s.closes x)
  · exact (iterate_add s.act x 1 _).symm.trans ((congrArg (iterate s.act x) hr).trans (s.closes x))

def orbit (x y : Fin d) : Prop := ∃ k, iterate s.act x k = y

theorem orbit_trans {x y z : Fin d} : s.orbit x y → s.orbit y z → s.orbit x z := by
  rintro ⟨a, rfl⟩ ⟨b, rfl⟩
  exact ⟨a+b, iterate_add ..⟩

theorem orbit_symm {x y : Fin d} : s.orbit x y → s.orbit y x := by
  rintro ⟨k, rfl⟩
  rw [s.reduced x k]
  refine ⟨s.period - k % s.period, ?_⟩
  rw [← iterate_add, Nat.add_sub_of_le (Nat.le_of_lt (Nat.mod_lt _ s.positive)), s.closes]

/-- A cycle contributes its least dart. This test enumerates the orbit. -/
def leader (x : Fin d) : Bool :=
  (List.range s.period).all fun k => decide (x.val ≤ (iterate s.act x k).val)

theorem leader_iff (x : Fin d) : s.leader x ↔ ∀ y, s.orbit x y → x.val ≤ y.val := by
  simp only [leader, List.all_eq_true, List.mem_range, decide_eq_true_eq]
  constructor
  · rintro h _ ⟨k, rfl⟩
    rw [s.reduced x k]
    exact h _ (Nat.mod_lt _ s.positive)
  · intro h k _
    exact h _ ⟨k, rfl⟩

theorem unique_leader (x : Fin d) :
    ∃ y, s.orbit x y ∧ s.leader y ∧ ∀ z, s.orbit x z → s.leader z → z = y := by
  have existsMin (n : Nat) (x : Fin d) (hx : x.val = n) : ∃ y, s.orbit x y ∧ s.leader y := by
    induction n using Nat.strongRecOn generalizing x with
    | ind n ih =>
      by_cases h : ∀ y, s.orbit x y → x.val ≤ y.val
      · exact ⟨x, ⟨0, rfl⟩, (s.leader_iff x).2 h⟩
      · letI := Classical.propDecidable
        have he : ∃ y, s.orbit x y ∧ y.val < x.val := by
          apply Classical.byContradiction
          intro hn
          exact h fun y hy => Nat.le_of_not_gt fun hlt => hn ⟨y, hy, hlt⟩
        obtain ⟨y, hy, hlt⟩ := he
        obtain ⟨z, hz, hz'⟩ := ih y.val (hx ▸ hlt) y rfl
        exact ⟨z, s.orbit_trans hy hz, hz'⟩
  obtain ⟨y, hy, ly⟩ := existsMin x.val x rfl
  refine ⟨y, hy, ly, ?_⟩
  intro z hz lz
  have hzy := s.orbit_trans (s.orbit_symm hz) hy
  exact Fin.ext (Nat.le_antisymm ((s.leader_iff z).1 lz y hzy)
    ((s.leader_iff y).1 ly z (s.orbit_symm hzy)))

end Rotation

def darts : (d : Nat) → List (Fin d)
  | 0 => []
  | n+1 => 0 :: (darts n).map Fin.succ

theorem darts_length (d : Nat) : (darts d).length = d := by
  induction d with
  | zero => rfl
  | succ n ih => simp [darts, ih]

private theorem count_complements {α : Type} (p : α → Bool) (l : List α) :
    l.countP p + l.countP (fun x => !(p x)) = l.length := by
  induction l with
  | nil => rfl
  | cons x xs ih => cases h : p x <;> simp [List.countP_cons, h] <;> omega

def Rotation.cycles {d : Nat} (s : Rotation d) := (darts d).countP s.leader

/-- In each cycle of length e, its e−1 nonleaders contribute the local
    ramification multiplicity. -/
def Rotation.ramification {d : Nat} (s : Rotation d) := (darts d).countP (fun x => !(s.leader x))

theorem Rotation.cycles_add_ramification {d : Nat} (s : Rotation d) :
    s.cycles + s.ramification = d := (count_complements s.leader _).trans (darts_length d)

structure Dessin (d : Nat) where
  black : Rotation d
  white : Rotation d
  face : Rotation d
  monodromy : ∀ x, black.act (white.act (face.act x)) = x

/-- Cellular Euler characteristic of the compact oriented dessin surface.
    Black and white cycles give vertices, darts give edges, and exterior
    cycles give faces. The formula also applies componentwise to disconnected covers. -/
def Dessin.euler {d : Nat} (D : Dessin d) : Int :=
  (D.black.cycles : Int) + D.white.cycles - d + D.face.cycles

theorem dessin_riemann_hurwitz {d : Nat} (D : Dessin d) :
    D.euler = 2 * (d : Int) -
      (D.black.ramification + D.white.ramification + D.face.ramification : Nat) := by
  have hb := D.black.cycles_add_ramification
  have hw := D.white.cycles_add_ramification
  have hf := D.face.cycles_add_ramification
  unfold Dessin.euler
  omega

def cyc3 (x : Fin 3) : Fin 3 := ⟨(x.val + 1) % 3, Nat.mod_lt _ (by decide)⟩

def triangleRotation : Rotation 3 :=
  ⟨cyc3, 3, by decide, by decide⟩

def torusDessin : Dessin 3 := ⟨triangleRotation, triangleRotation, triangleRotation, by decide⟩

theorem dessin_torus_d3_genus_one :
    (∀ x : Fin 3, torusDessin.black.orbit 0 x) ∧
    (torusDessin.black.cycles, torusDessin.white.cycles, torusDessin.face.cycles) = (1, 1, 1) ∧
    (∀ g : Nat, 2 - 2 * (g : Int) = torusDessin.euler ↔ g = 1) := by
  refine ⟨?_, by decide, ?_⟩
  · intro x
    exact ⟨x.val, (by decide : ∀ t : Fin 3, iterate cyc3 0 t.val = t) x⟩
  · have h : torusDessin.euler = 0 := by decide
    intro g
    rw [h]
    omega

-- ============================================================================
-- 2. Hyperbolic Anabelian Euler Characteristic Separation (chi = 2 - 2g - n)
-- ============================================================================

def curveEuler (g n : Int) : Int := 2 - 2 * g - n

def isHyperbolicAnabelian (g n : Int) : Prop := curveEuler g n < 0

theorem hyperbolic_p1_minus_3pts_is_anabelian_and_elliptic_is_not :
    curveEuler 0 3 = -1 ∧
    curveEuler 1 0 = 0 ∧
    isHyperbolicAnabelian 0 3 ∧
    ¬ isHyperbolicAnabelian 1 0 ∧
    ¬ isHyperbolicAnabelian 0 0 ∧
    ¬ isHyperbolicAnabelian 0 2 := by
  unfold isHyperbolicAnabelian curveEuler
  decide

-- ============================================================================
-- 3. o-Minimal Tame Topology: 1D Cell Euler Characteristic & Additivity
-- ============================================================================

inductive Cell1D where
  | point : Cell1D
  | openInterval : Cell1D
  deriving DecidableEq

def cellEuler : Cell1D → Int
  | Cell1D.point => 1
  | Cell1D.openInterval => -1

def cellsEuler : List Cell1D → Int
  | [] => 0
  | c :: cs => cellEuler c + cellsEuler cs

/-- Exact Euler characteristic of a Boolean subset over a finite atomic CAD partition. -/
def maskEuler : List Cell1D → List Bool → Int
  | c :: cs, b :: bs => (if b then cellEuler c else 0) + maskEuler cs bs
  | _, _ => 0

def maskOr : List Bool → List Bool → List Bool
  | a :: as, b :: bs => (a || b) :: maskOr as bs
  | _, _ => []

def maskAnd : List Bool → List Bool → List Bool
  | a :: as, b :: bs => (a && b) :: maskAnd as bs
  | _, _ => []

theorem maskEuler_additivity :
    ∀ (xs : List Cell1D) (ma mb : List Bool),
      ma.length = xs.length →
      mb.length = xs.length →
      maskEuler xs (maskOr ma mb) + maskEuler xs (maskAnd ma mb) =
        maskEuler xs ma + maskEuler xs mb := by
  intro xs
  induction xs with
  | nil =>
    intro ma mb hma hmb
    cases ma with
    | nil =>
      cases mb with
      | nil => rfl
      | cons _ _ => cases hmb
    | cons _ _ => cases hma
  | cons c cs ih =>
    intro ma mb hma hmb
    cases ma with
    | nil => cases hma
    | cons a as =>
      cases mb with
      | nil => cases hmb
      | cons b bs =>
        have has : as.length = cs.length := Nat.succ.inj hma
        have hbs : bs.length = cs.length := Nat.succ.inj hmb
        have hrec := ih as bs has hbs
        simp only [maskOr, maskAnd, maskEuler]
        cases a <;> cases b <;> simp only [Bool.false_or, Bool.true_or, Bool.false_and, Bool.true_and,
          Bool.false_eq_true, ite_true, ite_false] <;> omega

/-- Subdivision into n+1 open edges and n+2 vertices. -/
def intervalCells : Nat → List Cell1D
  | 0 => [.point, .openInterval, .point]
  | n+1 => .point :: .openInterval :: intervalCells n

theorem ominimal_closed_interval_euler_one (n : Nat) : cellsEuler (intervalCells n) = 1 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [intervalCells, cellsEuler, cellEuler, ih]

-- ============================================================================
-- 4. Scholze Perfectoid vs Non-Perfectoid Frobenius Surjectivity
-- ============================================================================

/-- Elements of the classical non-perfectoid ring F_3[t]/(t^2) represented as (a, b) = a + b*t. -/
structure F3Dual where
  a : Fin 3
  b : Fin 3
  deriving DecidableEq

def f3DualMul (x y : F3Dual) : F3Dual :=
  ⟨⟨(x.a.val * y.a.val) % 3, Nat.mod_lt _ (by decide)⟩,
   ⟨(x.a.val * y.b.val + x.b.val * y.a.val) % 3, Nat.mod_lt _ (by decide)⟩⟩

def f3DualFrob (x : F3Dual) : F3Dual :=
  f3DualMul x (f3DualMul x x)

/-- In F_3[t]/(t^2), (a + b*t)^3 = a^3 + 3*a^2*b*t = a + 0*t can NEVER equal t = (0, 1). -/
theorem non_perfectoid_fp_t_mod_t2_frobenius_not_surjective :
    ∀ x : F3Dual, f3DualFrob x ≠ ⟨⟨0, by decide⟩, ⟨1, by decide⟩⟩ := by
  intro ⟨a, b⟩ h
  have hb : (f3DualFrob ⟨a, b⟩).b.val = 1 := congrArg (fun z => z.b.val) h
  match a, b with
  | ⟨0, _⟩, ⟨0, _⟩ => simp [f3DualFrob, f3DualMul] at hb
  | ⟨0, _⟩, ⟨1, _⟩ => simp [f3DualFrob, f3DualMul] at hb
  | ⟨0, _⟩, ⟨2, _⟩ => simp [f3DualFrob, f3DualMul] at hb
  | ⟨1, _⟩, ⟨0, _⟩ => simp [f3DualFrob, f3DualMul] at hb
  | ⟨1, _⟩, ⟨1, _⟩ => simp [f3DualFrob, f3DualMul] at hb
  | ⟨1, _⟩, ⟨2, _⟩ => simp [f3DualFrob, f3DualMul] at hb
  | ⟨2, _⟩, ⟨0, _⟩ => simp [f3DualFrob, f3DualMul] at hb
  | ⟨2, _⟩, ⟨1, _⟩ => simp [f3DualFrob, f3DualMul] at hb
  | ⟨2, _⟩, ⟨2, _⟩ => simp [f3DualFrob, f3DualMul] at hb

/-! ## Frobenius and the characteristic-p inverse limit -/

structure RingData (A : Type u) where
  zero : A
  one : A
  add : A → A → A
  neg : A → A
  mul : A → A → A
  add_assoc : ∀ a b c, add (add a b) c = add a (add b c)
  add_comm : ∀ a b, add a b = add b a
  add_zero : ∀ a, add a zero = a
  add_neg : ∀ a, add a (neg a) = zero
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  mul_comm : ∀ a b, mul a b = mul b a
  mul_one : ∀ a, mul a one = a
  mul_add : ∀ a b c, mul a (add b c) = add (mul a b) (mul a c)

namespace RingData
variable {A : Type u} (R : RingData A)

theorem zero_add (a : A) : R.add R.zero a = a := (R.add_comm ..).trans (R.add_zero a)
theorem one_mul (a : A) : R.mul R.one a = a := (R.mul_comm ..).trans (R.mul_one a)
theorem mul_left_comm (a b c : A) : R.mul a (R.mul b c) = R.mul b (R.mul a c) := by
  rw [← R.mul_assoc, R.mul_comm a b, R.mul_assoc]
theorem add_left_comm (a b c : A) : R.add a (R.add b c) = R.add b (R.add a c) := by
  rw [← R.add_assoc, R.add_comm a b, R.add_assoc]
theorem add_cancel {a b c : A} (h : R.add a b = R.add a c) : b = c := by
  have he := congrArg (R.add (R.neg a)) h
  simpa only [← R.add_assoc, R.add_comm (R.neg a) a, R.add_neg, R.zero_add] using he

theorem mul_zero (a : A) : R.mul a R.zero = R.zero := by
  apply R.add_cancel (a := R.mul a R.zero)
  rw [← R.mul_add, R.add_zero, R.add_zero]

def pow (a : A) : Nat → A
  | 0 => R.one
  | n+1 => R.mul a (pow a n)

theorem mul_pow (a b : A) (n : Nat) : R.pow (R.mul a b) n = R.mul (R.pow a n) (R.pow b n) := by
  induction n with
  | zero => exact (R.mul_one _).symm
  | succ n ih => simp only [pow, ih, R.mul_assoc, R.mul_left_comm, R.mul_comm]

def multiple (n : Nat) (a : A) : A :=
  match n with
  | 0 => R.zero
  | k+1 => R.add a (multiple k a)

theorem multiple_add (m n : Nat) (a : A) :
    R.multiple (m+n) a = R.add (R.multiple m a) (R.multiple n a) := by
  induction n with
  | zero => exact (R.add_zero _).symm
  | succ n ih =>
    change R.add a (R.multiple (m+n) a) = R.add (R.multiple m a) (R.add a (R.multiple n a))
    rw [ih, R.add_left_comm]

theorem mul_multiple (x y : A) (n : Nat) :
    R.mul x (R.multiple n y) = R.multiple n (R.mul x y) := by
  induction n with
  | zero => exact R.mul_zero x
  | succ n ih => simp only [multiple, R.mul_add, ih]

theorem multiple_mul (k n : Nat) (x : A) :
    R.multiple (k*n) x = R.multiple n (R.multiple k x) := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Nat.mul_succ, multiple_add, ih, multiple, R.add_comm]

def diagonal (n : Nat) (f : Nat → Nat → A) : A :=
  match n with
  | 0 => f 0 0
  | k+1 => R.add (f 0 (k+1)) (diagonal k (fun i j => f (i+1) j))

theorem diagonal_congr (n : Nat) (f g : Nat → Nat → A)
    (h : ∀ i j, i+j = n → f i j = g i j) : R.diagonal n f = R.diagonal n g := by
  induction n generalizing f g with
  | zero => exact h 0 0 rfl
  | succ n ih =>
    simp only [diagonal]
    rw [h 0 (n+1) (by omega), ih _ (fun i j => g (i+1) j) (fun i j e => h _ _ (by omega))]

theorem diagonal_last (n : Nat) (f : Nat → Nat → A) :
    R.diagonal (n+1) f = R.add (R.diagonal n (fun i j => f i (j+1))) (f (n+1) 0) := by
  induction n generalizing f with
  | zero => rfl
  | succ n ih =>
    change R.add _ (R.diagonal (n+1) _) = R.add (R.add _ _) _
    rw [ih, R.add_assoc]

theorem diagonal_add (n : Nat) (f g : Nat → Nat → A) :
    R.diagonal n (fun i j => R.add (f i j) (g i j)) = R.add (R.diagonal n f) (R.diagonal n g) := by
  induction n generalizing f g with
  | zero => rfl
  | succ n ih =>
    simp only [diagonal]
    rw [ih (fun i j => f (i+1) j) (fun i j => g (i+1) j)]
    simp only [R.add_assoc, R.add_left_comm, R.add_comm]

theorem diagonal_mul (n : Nat) (a : A) (f : Nat → Nat → A) :
    R.diagonal n (fun i j => R.mul a (f i j)) = R.mul a (R.diagonal n f) := by
  induction n generalizing f with
  | zero => rfl
  | succ n ih => simp only [diagonal, R.mul_add]; rw [ih (fun i j => f (i+1) j)]

def west (f : Nat → Nat → A) : Nat → Nat → A
  | 0, _ => R.zero
  | i+1, j => f i j

def south (f : Nat → Nat → A) : Nat → Nat → A
  | _, 0 => R.zero
  | i, j+1 => f i j

def paths : Nat → Nat → Nat
  | 0, _ => 1
  | _+1, 0 => 1
  | i+1, j+1 => paths i (j+1) + paths (i+1) j

def term (a b : A) (i j : Nat) := R.multiple (paths i j) (R.mul (R.pow a i) (R.pow b j))

private theorem term_pascal (a b : A) (i j : Nat) (h : 0 < i+j) :
    R.term a b i j =
      R.add (R.west (fun k l => R.mul a (R.term a b k l)) i j)
        (R.south (fun k l => R.mul b (R.term a b k l)) i j) := by
  cases i with
  | zero =>
    cases j with
    | zero => omega
    | succ j => simp [term, paths, pow, multiple, west, south, R.add_zero, R.zero_add, R.one_mul]
  | succ i =>
    cases j with
    | zero =>
      have h0 : paths i 0 = 1 := by cases i <;> rw [paths]
      simp [term, paths, pow, multiple, west, south, R.add_zero, R.mul_one, R.zero_add, h0]
    | succ j =>
      simp only [term, paths, R.multiple_add, west, south, R.mul_multiple, pow]
      simp only [R.mul_assoc, R.mul_left_comm, R.mul_comm]

theorem binomial_expansion (a b : A) (n : Nat) : R.pow (R.add a b) n = R.diagonal n (R.term a b) := by
  induction n with
  | zero => simp [diagonal, term, paths, multiple, pow, R.mul_one, R.add_zero]
  | succ n ih =>
    rw [R.diagonal_congr _ _ _ (fun i j h => R.term_pascal a b i j (by omega)), R.diagonal_add]
    conv => rhs; lhs; rw [diagonal]
    conv => rhs; rhs; rw [diagonal_last]
    simp only [west, south, R.zero_add, R.add_zero]
    rw [R.diagonal_mul n a, R.diagonal_mul n b, pow, ih]
    rw [R.mul_comm, R.mul_add, R.mul_comm a, R.mul_comm b]

def factorial : Nat → Nat
  | 0 => 1
  | n+1 => (n+1)*factorial n

private theorem path_factorials (i j : Nat) :
    paths i j * (factorial i * factorial j) = factorial (i+j) := by
  induction i generalizing j with
  | zero => simp [paths, factorial]
  | succ i hi =>
    induction j with
    | zero => simp [paths, factorial]
    | succ j hj =>
      rw [paths, Nat.add_mul]
      have ha : paths i (j+1) * (factorial (i+1)*factorial (j+1)) =
          (i+1)*factorial (i+(j+1)) := by
        calc
          _ = (i+1) * (paths i (j+1) * (factorial i * factorial (j+1))) := by rw [show factorial (i+1) = (i+1)*factorial i from rfl]; ac_rfl
          _ = _ := congrArg _ (hi (j+1))
      have hb : paths (i+1) j * (factorial (i+1)*factorial (j+1)) =
          (j+1)*factorial (i+1+j) := by
        calc
          _ = (j+1) * (paths (i+1) j * (factorial (i+1) * factorial j)) := by rw [show factorial (j+1) = (j+1)*factorial j from rfl]; ac_rfl
          _ = _ := congrArg _ hj
      rw [ha, hb, show i+(j+1) = i+1+j by omega, ← Nat.add_mul, Nat.add_succ, factorial]

/-- Euclid's prime divisor condition. -/
def PrimeExponent (p : Nat) := 1 < p ∧ ∀ a b, p ∣ a*b → p ∣ a ∨ p ∣ b

private theorem not_dvd_factorial {p : Nat} (hp : PrimeExponent p) (n : Nat) (hn : n < p) :
    ¬ p ∣ factorial n := by
  induction n with
  | zero => intro h; have he := Nat.eq_one_of_dvd_one h; have := hp.1; omega
  | succ n ih =>
    intro h; rcases hp.2 _ _ h with h | h
    · have := Nat.le_of_dvd (Nat.zero_lt_succ n) h; omega
    · exact ih (by omega) h

theorem prime_dvd_mixed {p i j : Nat} (hp : PrimeExponent p) (hi : 0 < i) (hj : 0 < j)
    (h : i+j=p) : p ∣ paths i j := by
  have hf : p ∣ factorial (i+j) := by
    rw [h]
    cases p with
    | zero => omega
    | succ n => exact Nat.dvd_mul_right _ _
  rw [← path_factorials] at hf
  rcases hp.2 _ _ hf with h' | h'
  · exact h'
  · rcases hp.2 _ _ h' with h' | h'
    · exact False.elim (not_dvd_factorial hp i (by omega) h')
    · exact False.elim (not_dvd_factorial hp j (by omega) h')

def Characteristic (p : Nat) := ∀ n, R.multiple n R.one = R.zero ↔ p ∣ n

private theorem mixed_term_zero {p i j : Nat} (hp : PrimeExponent p) (hc : R.Characteristic p)
    (a b : A) (hi : 0 < i) (hj : 0 < j) (h : i+j=p) : R.term a b i j = R.zero := by
  have hm := (hc _).2 (prime_dvd_mixed hp hi hj h)
  have he := congrArg (R.mul (R.mul (R.pow a i) (R.pow b j))) hm
  simpa only [term, R.mul_multiple, R.mul_one, R.mul_zero] using he

private theorem diagonal_end (n : Nat) (f : Nat → Nat → A)
    (h : ∀ i j, i+j=n → 0 < j → f i j = R.zero) : R.diagonal n f = f n 0 := by
  induction n generalizing f with
  | zero => rfl
  | succ n ih =>
    rw [diagonal, h 0 _ (by omega) (by omega), R.zero_add, ih (fun i j => f (i+1) j) (fun i j he hj => h _ _ (by omega) hj)]

theorem freshman_dream {p : Nat} (hp : PrimeExponent p) (hc : R.Characteristic p) (a b : A) :
    R.pow (R.add a b) p = R.add (R.pow a p) (R.pow b p) := by
  rw [R.binomial_expansion]
  cases p with
  | zero => have := hp.1; omega
  | succ n =>
    rw [diagonal, R.diagonal_end n (fun i j => R.term a b (i+1) j) (fun i j he hj => R.mixed_term_zero hp hc a b
      (by omega : 0 < i+1) hj (by omega))]
    simp only [term, paths, multiple, R.add_zero, pow, R.one_mul, R.mul_one]
    rw [R.add_comm]

theorem pow_one (n : Nat) : R.pow R.one n = R.one := by
  induction n with
  | zero => rfl
  | succ n ih => rw [pow, ih, R.mul_one]

theorem pow_zero_succ (n : Nat) : R.pow R.zero (n+1) = R.zero := by
  rw [pow, R.mul_comm, R.mul_zero]

theorem neg_unique {a b : A} (h : R.add a b = R.zero) : b = R.neg a :=
  R.add_cancel (h.trans (R.add_neg a).symm)

structure Tilt (p : Nat) where
  atLevel : Nat → A
  compatible : ∀ n, R.pow (atLevel (n+1)) p = atLevel n

namespace Tilt
variable {R : RingData A} {p : Nat}

theorem ext {x y : R.Tilt p} (h : ∀ n, x.atLevel n = y.atLevel n) : x = y := by
  cases x; cases y; congr; exact funext h

def mul (x y : R.Tilt p) : R.Tilt p :=
  ⟨fun n => R.mul (x.atLevel n) (y.atLevel n), fun n => by rw [R.mul_pow, x.compatible, y.compatible]⟩

def add (hp : PrimeExponent p) (hc : R.Characteristic p) (x y : R.Tilt p) : R.Tilt p :=
  ⟨fun n => R.add (x.atLevel n) (y.atLevel n), fun n => by rw [R.freshman_dream hp hc, x.compatible, y.compatible]⟩

def one : R.Tilt p := ⟨fun _ => R.one, fun _ => R.pow_one p⟩

def zero (hp : PrimeExponent p) : R.Tilt p :=
  ⟨fun _ => R.zero, fun _ => by cases p with
    | zero => have := hp.1; omega
    | succ n => exact R.pow_zero_succ n⟩

def neg (hp : PrimeExponent p) (hc : R.Characteristic p) (x : R.Tilt p) : R.Tilt p :=
  ⟨fun n => R.neg (x.atLevel n), fun n => by
    apply R.neg_unique
    rw [← x.compatible n, ← R.freshman_dream hp hc, R.add_neg]
    exact (zero (R := R) hp).compatible n⟩

def ring (hp : PrimeExponent p) (hc : R.Characteristic p) : RingData (R.Tilt p) where
  zero := zero hp
  one := one
  neg := neg hp hc
  add := add hp hc
  mul := mul
  add_assoc := fun _ _ _ => ext fun _ => R.add_assoc .. 
  add_comm := fun _ _ => ext fun _ => R.add_comm ..
  add_zero := fun _ => ext fun _ => R.add_zero ..
  add_neg := fun _ => ext fun _ => R.add_neg ..
  mul_assoc := fun _ _ _ => ext fun _ => R.mul_assoc ..
  mul_comm := fun _ _ => ext fun _ => R.mul_comm ..
  mul_one := fun _ => ext fun _ => R.mul_one ..
  mul_add := fun _ _ _ => ext fun _ => R.mul_add ..

theorem pow_level (hp : PrimeExponent p) (hc : R.Characteristic p) (x : R.Tilt p) (k n : Nat) :
    ((ring hp hc).pow x k).atLevel n = R.pow (x.atLevel n) k := by
  induction k with
  | zero => rfl
  | succ k ih => exact congrArg (R.mul (x.atLevel n)) ih

def frobenius (x : R.Tilt p) : R.Tilt p :=
  ⟨fun n => R.pow (x.atLevel n) p, fun n => congrArg (fun a => R.pow a p) (x.compatible n)⟩

theorem frobenius_is_pth_power (hp : PrimeExponent p) (hc : R.Characteristic p) (x : R.Tilt p) :
    (ring hp hc).pow x p = x.frobenius := ext (pow_level hp hc x p)

def root (x : R.Tilt p) : R.Tilt p := ⟨fun n => x.atLevel (n+1), fun n => x.compatible (n+1)⟩

theorem inverse_frobenius (x : R.Tilt p) : (root x).frobenius = x ∧ root x.frobenius = x :=
  ⟨ext x.compatible, ext x.compatible⟩

theorem frobenius_bijective :
    (∀ x y : R.Tilt p, x.frobenius = y.frobenius → x = y) ∧
    (∀ y : R.Tilt p, ∃ x : R.Tilt p, x.frobenius = y) := by
  refine ⟨fun x y h => (inverse_frobenius x).2.symm.trans ((congrArg root h).trans (inverse_frobenius y).2),
    fun y => ⟨root y, (inverse_frobenius y).1⟩⟩

end Tilt
end RingData

end EshkolGrothendieckEra3EsquissePerfectoid
