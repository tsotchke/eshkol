/-! # Two-dimensional horn filling, Čech transition data, Euler characteristics and cumulative universes -/

namespace EshkolGrothendieckEra3

/- ============================================================================
   1. Truncated Simplicial Sets, Horns, Quasi-Categories & Kan Complexes
   ============================================================================ -/

structure TruncSSet where
  V : Type
  E : Type
  T : Type
  d0E : E -> V   -- target
  d1E : E -> V   -- source
  d0T : T -> E
  d1T : T -> E
  d2T : T -> E

/-- Every compatible inner horn Lambda^2_1 (e0, e2) with d1(e0) = d0(e2) has a filler. -/
def IsQuasiCat2 (X : TruncSSet) : Prop :=
  forall e0 e2 : X.E, X.d1E e0 = X.d0E e2 ->
    exists t : X.T, X.d0T t = e0 /\ X.d2T t = e2

/-- Every compatible outer-left horn Lambda^2_0 (e1, e2) with d1(e1) = d1(e2) has a filler. -/
def HasOuter0Fillers (X : TruncSSet) : Prop :=
  forall e1 e2 : X.E, X.d1E e1 = X.d1E e2 ->
    exists t : X.T, X.d1T t = e1 /\ X.d2T t = e2

/-- Every compatible outer-right horn Lambda^2_2 (e0, e1) with d0(e0) = d0(e1) has a filler. -/
def HasOuter2Fillers (X : TruncSSet) : Prop :=
  forall e0 e1 : X.E, X.d0E e0 = X.d0E e1 ->
    exists t : X.T, X.d0T t = e0 /\ X.d1T t = e1

/-- All three two-dimensional horn-filling predicates for the given face data. -/
def IsKanComplex2 (X : TruncSSet) : Prop :=
  IsQuasiCat2 X /\ HasOuter0Fillers X /\ HasOuter2Fillers X

/- --- Nerve of Delta^1 (0 -> 1): Quasi-Category, NOT Kan! --- -/

inductive ArrowObj : Type
  | o0 | o1
  deriving DecidableEq

inductive ArrowMor : Type
  | id0 | id1 | f01
  deriving DecidableEq

def arrowSrc : ArrowMor -> ArrowObj
  | .id0 => .o0
  | .id1 => .o1
  | .f01 => .o0

def arrowTgt : ArrowMor -> ArrowObj
  | .id0 => .o0
  | .id1 => .o1
  | .f01 => .o1

inductive ArrowTri : Type
  | t_00 | t_f0 | t_1f | t_11
  deriving DecidableEq

def arrowTriD0 : ArrowTri -> ArrowMor
  | .t_00 => .id0
  | .t_f0 => .f01
  | .t_1f => .id1
  | .t_11 => .id1

def arrowTriD1 : ArrowTri -> ArrowMor
  | .t_00 => .id0
  | .t_f0 => .f01
  | .t_1f => .f01
  | .t_11 => .id1

def arrowTriD2 : ArrowTri -> ArrowMor
  | .t_00 => .id0
  | .t_f0 => .id0
  | .t_1f => .f01
  | .t_11 => .id1

def nerveArrow : TruncSSet :=
  { V := ArrowObj
    E := ArrowMor
    T := ArrowTri
    d0E := arrowTgt
    d1E := arrowSrc
    d0T := arrowTriD0
    d1T := arrowTriD1
    d2T := arrowTriD2 }

/-- Theorem 1: N(Delta^1) fills every inner horn Lambda^2_1 (quasi-category),
    yet fails the outer horn Lambda^2_0 = (id0, f01), proving Kan subsetneq QCat. -/
theorem nerve_arrow_is_quasicat_not_kan :
    IsQuasiCat2 nerveArrow /\ Not (IsKanComplex2 nerveArrow) := by
  constructor
  · intro e0 e2 h
    cases e0 <;> cases e2
    · exact ⟨.t_00, rfl, rfl⟩
    · cases h
    · cases h
    · cases h
    · exact ⟨.t_11, rfl, rfl⟩
    · exact ⟨.t_1f, rfl, rfl⟩
    · exact ⟨.t_f0, rfl, rfl⟩
    · cases h
    · cases h
  · intro hKan
    rcases hKan with ⟨_, hOut0, _⟩
    have hCompat : nerveArrow.d1E .id0 = nerveArrow.d1E .f01 := rfl
    rcases hOut0 .id0 .f01 hCompat with ⟨t, ht1, ht2⟩
    cases t <;> simp [nerveArrow, arrowTriD1, arrowTriD2] at ht1 ht2

/- --- Nerve of B(Z/2Z) (1-object groupoid): Kan Complex / infinity-Groupoid! --- -/

inductive Z2Elem : Type
  | e | g
  deriving DecidableEq

def z2Mul : Z2Elem -> Z2Elem -> Z2Elem
  | .e, y => y
  | .g, .e => .g
  | .g, .g => .e

def nerveZ2Groupoid : TruncSSet :=
  { V := Unit
    E := Z2Elem
    T := Z2Elem × Z2Elem
    d0E := fun _ => ()
    d1E := fun _ => ()
    d0T := fun p => p.1
    d1T := fun p => z2Mul p.1 p.2
    d2T := fun p => p.2 }

/-- Theorem 2: The nerve of the groupoid B(Z/2Z) fills ALL inner and outer 2-horns
    (Kan complex / infinity-groupoid). -/
theorem nerve_z2_groupoid_is_kan : IsKanComplex2 nerveZ2Groupoid := by
  refine ⟨?_, ?_, ?_⟩
  · intro e0 e2 _
    exact ⟨(e0, e2), rfl, rfl⟩
  · intro e1 e2 _
    refine ⟨(z2Mul e1 e2, e2), ?_, rfl⟩
    cases e1 <;> cases e2 <;> rfl
  · intro e0 e1 _
    refine ⟨(e0, z2Mul e0 e1), rfl, ?_⟩
    cases e0 <;> cases e1 <;> rfl

/- --- Negative Control: Bare Inner Horn Lambda^2_1 is NOT a Quasi-Category --- -/

inductive BareV : Type
  | v0 | v1 | v2
  deriving DecidableEq

inductive BareE : Type
  | e01 | e12
  deriving DecidableEq

def bareSrc : BareE -> BareV
  | .e01 => .v0
  | .e12 => .v1

def bareTgt : BareE -> BareV
  | .e01 => .v1
  | .e12 => .v2

def bareInnerHorn : TruncSSet :=
  { V := BareV
    E := BareE
    T := Empty
    d0E := bareTgt
    d1E := bareSrc
    d0T := fun t => Empty.elim t
    d1T := fun t => Empty.elim t
    d2T := fun t => Empty.elim t }

/-- Bonus Theorem: The bare inner horn Lambda^2_1 fails inner horn filling (QCat subsetneq sSet). -/
theorem bare_inner_horn_not_quasicat : Not (IsQuasiCat2 bareInnerHorn) := by
  intro hQ
  have hCompat : bareInnerHorn.d1E .e12 = bareInnerHorn.d0E .e01 := rfl
  rcases hQ .e12 .e01 hCompat with ⟨t, _⟩
  exact Empty.elim t

/- ============================================================================
   2. Higher Stacks & Cech 1-Cocycle Descent Law
   ============================================================================ -/

structure GroupData (G : Type) where
  one : G
  mul : G -> G -> G
  inv : G -> G
  mul_assoc : forall a b c : G, mul (mul a b) c = mul a (mul b c)
  one_mul : forall a : G, mul one a = a
  mul_one : forall a : G, mul a one = a
  mul_left_inv : forall a : G, mul (inv a) a = one
  mul_right_inv : forall a : G, mul a (inv a) = one

/-- 1-Cocycle condition on a cover indexed by I: g(i, j) * g(j, k) = g(i, k). -/
def IsCechCocycle {G I : Type} (grp : GroupData G) (g : I -> I -> G) : Prop :=
  forall i j k : I, grp.mul (g i j) (g j k) = g i k

/-- Gauge transformation of a cocycle by local 1-cochain h: I -> G. -/
def gaugeTransform {G I : Type} (grp : GroupData G) (h : I -> G) (g : I -> I -> G) :
    I -> I -> G :=
  fun i j => grp.mul (h i) (grp.mul (g i j) (grp.inv (h j)))

def z2Group : GroupData Z2Elem :=
  { one := .e
    mul := z2Mul
    inv := fun x => x
    mul_assoc := by intro a b c; cases a <;> cases b <;> cases c <;> rfl
    one_mul := by intro a; cases a <;> rfl
    mul_one := by intro a; cases a <;> rfl
    mul_left_inv := by intro a; cases a <;> rfl
    mul_right_inv := by intro a; cases a <;> rfl }

/-- An explicitly invalid transition assignment on a 3-open cover with triple overlap. -/
def invalidZ2Transition : Fin 3 -> Fin 3 -> Z2Elem
  | ⟨0, _⟩, ⟨1, _⟩ => .g
  | ⟨1, _⟩, ⟨2, _⟩ => .g
  | ⟨0, _⟩, ⟨2, _⟩ => .g
  | _, _ => .e

/-- Theorem 3: Gauge transformations preserve the Cech 2-cocycle condition for every group G,
    AND there exists a non-matching transition assignment violating the triple-overlap law. -/
theorem cech_cocycle_composition_law :
    (forall (G I : Type) (grp : GroupData G) (g : I -> I -> G) (h : I -> G),
      IsCechCocycle grp g -> IsCechCocycle grp (gaugeTransform grp h g)) /\
    Not (IsCechCocycle z2Group invalidZ2Transition) := by
  constructor
  · intro G I grp g h hCoc i j k
    unfold gaugeTransform
    calc
      grp.mul (grp.mul (h i) (grp.mul (g i j) (grp.inv (h j))))
              (grp.mul (h j) (grp.mul (g j k) (grp.inv (h k))))
        = grp.mul (h i) (grp.mul (grp.mul (g i j) (grp.inv (h j)))
              (grp.mul (h j) (grp.mul (g j k) (grp.inv (h k))))) := by
                rw [grp.mul_assoc]
      _ = grp.mul (h i) (grp.mul (g i j) (grp.mul (grp.inv (h j))
              (grp.mul (h j) (grp.mul (g j k) (grp.inv (h k)))))) := by
                rw [grp.mul_assoc]
      _ = grp.mul (h i) (grp.mul (g i j) (grp.mul (grp.mul (grp.inv (h j)) (h j))
              (grp.mul (g j k) (grp.inv (h k))))) := by
                rw [← grp.mul_assoc (grp.inv (h j)) (h j)]
      _ = grp.mul (h i) (grp.mul (g i j) (grp.mul grp.one
              (grp.mul (g j k) (grp.inv (h k))))) := by
                rw [grp.mul_left_inv]
      _ = grp.mul (h i) (grp.mul (g i j) (grp.mul (g j k) (grp.inv (h k)))) := by
                rw [grp.one_mul]
      _ = grp.mul (h i) (grp.mul (grp.mul (g i j) (g j k)) (grp.inv (h k))) := by
                rw [← grp.mul_assoc (g i j) (g j k)]
      _ = grp.mul (h i) (grp.mul (g i k) (grp.inv (h k))) := by
                rw [hCoc i j k]
  · intro hBad
    have h012 := hBad ⟨0, by omega⟩ ⟨1, by omega⟩ ⟨2, by omega⟩
    simp [z2Group, z2Mul, invalidZ2Transition] at h012

/- ============================================================================
   3. Les Derivateurs (1991): Homotopy Pushout/Pullback Euler Exactness
      and Triangulated Cone Non-Uniqueness
   ============================================================================ -/

/-- Alternating sign (-1)^k as an Int. -/
def altSign (k : Nat) : Int :=
  if k % 2 = 0 then (1 : Int) else (-1 : Int)

theorem altSign_succ (k : Nat) : altSign (k + 1) = - altSign k := by
  unfold altSign
  by_cases h : k % 2 = 0
  · have h1 : (k + 1) % 2 = 1 := by omega
    simp [h, h1]
  · have h1 : (k + 1) % 2 = 0 := by omega
    simp [h, h1]

/-- Finite-dimensional graded chain complex dimensions in degrees 0 .. N. -/
def eulerUpTo (dim : Nat -> Nat) : Nat -> Int
  | 0 => (dim 0 : Int)
  | n + 1 => eulerUpTo dim n + altSign (n + 1) * (dim (n + 1) : Int)

/-- Degreewise dimensions of hocolim(B <- A -> C): P_0 = B_0 + C_0, P_{k+1} = A_k + B_{k+1} + C_{k+1}. -/
def hocolimDim (dimA dimB dimC : Nat -> Nat) : Nat -> Nat
  | 0 => dimB 0 + dimC 0
  | k + 1 => dimA k + dimB (k + 1) + dimC (k + 1)

theorem eulerUpTo_succ (dim : Nat -> Nat) (n : Nat) :
    eulerUpTo dim (n + 1) = eulerUpTo dim n + altSign (n + 1) * (dim (n + 1) : Int) :=
  rfl

theorem euler_hocolim_aux (dimA dimB dimC : Nat -> Nat) :
    forall n : Nat,
      eulerUpTo (hocolimDim dimA dimB dimC) (n + 1) =
        eulerUpTo dimB (n + 1) + eulerUpTo dimC (n + 1) - eulerUpTo dimA n := by
  intro n
  induction n with
  | zero =>
    simp [eulerUpTo, hocolimDim, altSign]
    omega
  | succ k ih =>
    rw [eulerUpTo_succ (hocolimDim dimA dimB dimC) (k + 1),
        eulerUpTo_succ dimB (k + 1),
        eulerUpTo_succ dimC (k + 1),
        eulerUpTo_succ dimA k,
        ih]
    simp only [hocolimDim]
    rw [altSign_succ (k + 1)]
    have hCast :
        ((dimA (k + 1) + dimB (k + 1 + 1) + dimC (k + 1 + 1) : Nat) : Int) =
          (dimA (k + 1) : Int) + (dimB (k + 1 + 1) : Int) + (dimC (k + 1 + 1) : Int) := by
      omega
    rw [hCast]
    have hMul :
        -(altSign (k + 1)) * ((dimA (k + 1) : Int) + (dimB (k + 1 + 1) : Int) + (dimC (k + 1 + 1) : Int)) =
          -(altSign (k + 1) * (dimA (k + 1) : Int))
          + (-(altSign (k + 1))) * (dimB (k + 1 + 1) : Int)
          + (-(altSign (k + 1))) * (dimC (k + 1 + 1) : Int) := by
      rw [Int.mul_add, Int.mul_add, Int.neg_mul]
    rw [hMul]
    omega

/-- Theorem 4: Exact Mayer-Vietoris Euler characteristic additivity for the
    graded dimensions of the standard mapping-cone model of a homotopy pushout:
    when the two target dimensions vanish in degree n+1,
    chi(hocolim(B <- A -> C)) = chi(B) + chi(C) - chi(A). -/
theorem derivator_hocolim_euler_exactness
    (dimA dimB dimC : Nat -> Nat) (n : Nat)
    (hBtop : dimB (n + 1) = 0) (hCtop : dimC (n + 1) = 0) :
    eulerUpTo (hocolimDim dimA dimB dimC) (n + 1) =
      eulerUpTo dimB n + eulerUpTo dimC n - eulerUpTo dimA n := by
  have h := euler_hocolim_aux dimA dimB dimC n
  simp [eulerUpTo, hBtop, hCtop] at h
  exact h

/- --- Bonus Theorem: Triangulated Cone Filler Non-Uniqueness --- -/

/-- Two 2x2 integer endomorphisms on Cone(0: Q[0] -> Q[1]) = Q[1] (+) Q[1]:
    h0 = [[1, 0], [0, 1]] (canonical derivator cone map) and
    h1 = [[1, 0], [1, 1]] (sheared filler).
    Both strictly preserve inclusion i(b) = (0, b) and projection p(a, b) = a, yet h0 != h1. -/
def coneH0 (v : Int × Int) : Int × Int := (v.1, v.2)
def coneH1 (v : Int × Int) : Int × Int := (v.1, v.1 + v.2)

theorem triangulated_cone_nonuniqueness :
    (forall b : Int, coneH0 (0, b) = (0, b)) /\
    (forall b : Int, coneH1 (0, b) = (0, b)) /\
    (forall a b : Int, (coneH0 (a, b)).1 = a) /\
    (forall a b : Int, (coneH1 (a, b)).1 = a) /\
    coneH0 (1, 0) ≠ coneH1 (1, 0) := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro b; rfl
  · intro b; simp [coneH1]
  · intro a b; rfl
  · intro a b; rfl
  · decide

/-! ## Cumulative Grothendieck universes in an ambient set theory -/

/-- The ambient operations are specified by their elements. Separation accepts
    an arbitrary predicate and indexed union ranges over the members of a set. -/
structure SetOperations (V : Type u) where
  mem : V → V → Prop
  irreflexive : ∀ x, ¬ mem x x
  empty : V
  empty_spec : ∀ x, ¬ mem x empty
  pair : V → V → V
  pair_spec : ∀ x a b, mem x (pair a b) ↔ x = a ∨ x = b
  power : V → V
  power_spec : ∀ x a, mem x (power a) ↔ ∀ z, mem z x → mem z a
  separate : V → (V → Prop) → V
  separate_spec : ∀ x a p, mem x (separate a p) ↔ mem x a ∧ p x
  iUnion : V → (V → V) → V
  union_spec : ∀ x a f, mem x (iUnion a f) ↔ ∃ i, mem i a ∧ mem x (f i)

structure UniverseTower {V : Type u} (S : SetOperations V) where
  level : Nat → V
  next : ∀ n, S.mem (level n) (level (n+1))
  transitive : ∀ {n a x}, S.mem a (level n) → S.mem x a → S.mem x (level n)
  empty_small : ∀ n, S.mem S.empty (level n)
  pair_small : ∀ {n a b}, S.mem a (level n) → S.mem b (level n) → S.mem (S.pair a b) (level n)
  power_small : ∀ {n a}, S.mem a (level n) → S.mem (S.power a) (level n)
  union_small : ∀ {n a} {f : V → V}, S.mem a (level n) →
    (∀ i, S.mem i a → S.mem (f i) (level n)) → S.mem (S.iUnion a f) (level n)

namespace UniverseTower
variable {V : Type u} {S : SetOperations V} (T : UniverseTower S)

abbrev Small (n : Nat) (a : V) := S.mem a (T.level n)

theorem cumulative {i j : Nat} (h : i ≤ j) {a : V} (ha : T.Small i a) : T.Small j a := by
  induction h with
  | refl => exact ha
  | step _ ih => exact T.transitive (T.next _) ih

/-- The lower universe itself witnesses strict inclusion. -/
theorem strict_stratification {i j : Nat} (h : i < j) :
    (∀ a, T.Small i a → T.Small j a) ∧
    T.Small j (T.level i) ∧ ¬ T.Small i (T.level i) ∧ ¬ T.Small i (T.level j) := by
  have hij := T.cumulative h (T.next i)
  exact ⟨fun _ => T.cumulative (Nat.le_of_lt h), hij, S.irreflexive _, fun hji =>
    S.irreflexive _ (T.transitive hij hji)⟩

def ordered (S : SetOperations V) (a b : V) := S.pair (S.pair a a) (S.pair a b)

theorem ordered_small {n a b} (ha : T.Small n a) (hb : T.Small n b) :
    T.Small n (ordered S a b) := T.pair_small (T.pair_small ha ha) (T.pair_small ha hb)

/-- The dependent sum is the union of singletons of tagged elements. -/
def sigmaSet (S : SetOperations V) (a : V) (b : V → V) : V :=
  S.iUnion a fun i => S.iUnion (b i) fun x => S.pair (ordered S i x) (ordered S i x)

theorem mem_sigma {a y : V} {b : V → V} :
    S.mem y (sigmaSet S a b) ↔ ∃ i, S.mem i a ∧ ∃ x, S.mem x (b i) ∧ y = ordered S i x := by
  simp only [sigmaSet, S.union_spec, S.pair_spec, or_self]

theorem sigma_small {n a} {b : V → V} (ha : T.Small n a)
    (hb : ∀ i, S.mem i a → T.Small n (b i)) : T.Small n (sigmaSet S a b) := by
  apply T.union_small ha
  intro i hi
  apply T.union_small (hb i hi)
  intro x hx
  have hp := T.ordered_small (T.transitive ha hi) (T.transitive (hb i hi) hx)
  exact T.pair_small hp hp

theorem separation_small {n a} (ha : T.Small n a) (p : V → Prop) :
    T.Small n (S.separate a p) := by
  apply T.transitive (T.power_small ha)
  exact (S.power_spec ..).2 fun x hx => ((S.separate_spec ..).1 hx).1

/-- Dependent functions are total single-valued graphs inside the dependent sum. -/
def piSet (S : SetOperations V) (a : V) (b : V → V) : V :=
  S.separate (S.power (sigmaSet S a b)) fun graph =>
    ∀ i, S.mem i a → ∃ x, S.mem x (b i) ∧ S.mem (ordered S i x) graph ∧
      ∀ y, S.mem y (b i) → S.mem (ordered S i y) graph → y = x

theorem pi_small {n a} {b : V → V} (ha : T.Small n a)
    (hb : ∀ i, S.mem i a → T.Small n (b i)) : T.Small n (piSet S a b) :=
  T.separation_small (T.power_small (T.sigma_small ha hb)) _

theorem product_and_function_small {n a b} (ha : T.Small n a) (hb : T.Small n b) :
    T.Small n (sigmaSet S a (fun _ => b)) ∧ T.Small n (piSet S a (fun _ => b)) :=
  ⟨T.sigma_small ha (fun _ _ => hb), T.pi_small ha (fun _ _ => hb)⟩

/-- A two-point index with tags ∅ and {∅} realizes disjoint union. -/
noncomputable def sumSet (S : SetOperations V) (a b : V) : V := by
  letI := Classical.propDecidable
  exact sigmaSet S (S.pair S.empty (S.pair S.empty S.empty)) (fun t => if t = S.empty then a else b)

theorem sum_small {n a b} (ha : T.Small n a) (hb : T.Small n b) : T.Small n (sumSet S a b) := by
  letI := Classical.propDecidable
  apply T.sigma_small (T.pair_small (T.empty_small n) (T.pair_small (T.empty_small n) (T.empty_small n)))
  intro t _
  split <;> assumption

end UniverseTower
end EshkolGrothendieckEra3
