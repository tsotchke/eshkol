/-! # Intersection forms, Dieudonné matrices and sheaves on a finite coproduct -/

namespace EshkolGrothendieckPillarsSixOps

-- ===========================================================================
-- 1. Standard Conjectures: Hodge Index Signature on P^1 x P^1 & Bl_pt(P^2)
-- ===========================================================================

/-- Exact intersection pairing on H^{1,1}(P^1 x P^1, Z) = <e_1, e_2>. -/
def p1xp1Pairing (a b : Int × Int) : Int :=
  a.1 * b.2 + a.2 * b.1

/-- Hyperplane class H = e_1 + e_2 on P^1 x P^1. -/
def p1xp1Hyperplane : Int × Int := (1, 1)

/-- Primitive (1,1) class P = e_1 - e_2 on P^1 x P^1. -/
def p1xp1Primitive : Int × Int := (1, -1)

theorem hodge_index_p1xp1_signature :
    p1xp1Pairing p1xp1Hyperplane p1xp1Hyperplane = 2 ∧
    p1xp1Pairing p1xp1Hyperplane p1xp1Hyperplane > 0 ∧
    p1xp1Pairing p1xp1Hyperplane p1xp1Primitive = 0 ∧
    p1xp1Pairing p1xp1Primitive p1xp1Primitive = -2 ∧
    p1xp1Pairing p1xp1Primitive p1xp1Primitive < 0 := by
  decide

theorem hodge_index_p1xp1_all_primitive_negative (a b : Int)
    (horth : p1xp1Pairing p1xp1Hyperplane (a, b) = 0)
    (hne : a ≠ 0) :
    p1xp1Pairing (a, b) (a, b) < 0 := by
  have hb : b = -a := by
    simp [p1xp1Pairing, p1xp1Hyperplane] at horth
    omega
  subst hb
  simp [p1xp1Pairing]
  have hsq : 0 < a * a := by
    rcases Int.lt_trichotomy a 0 with hlt | heq | hgt
    · have : 0 < (-a) * (-a) := Int.mul_pos (Int.neg_pos_of_neg hlt) (Int.neg_pos_of_neg hlt)
      simpa using this
    · exact absurd heq hne
    · exact Int.mul_pos hgt hgt
  omega

/-- Intersection pairing on H^{1,1}(Bl_pt(P^2), Z) = <H, E>. -/
def blowupP2Pairing (a b : Int × Int) : Int :=
  a.1 * b.1 - a.2 * b.2

theorem hodge_index_blowup_p2_signature :
    blowupP2Pairing (1, 0) (1, 0) = 1 ∧
    blowupP2Pairing (0, 1) (0, 1) = -1 ∧
    blowupP2Pairing (1, 0) (0, 1) = 0 := by
  decide

-- ===========================================================================
-- 2. Crystalline Cohomology & Dieudonne Theory: FV = VF = pI_2 & PD Algebra
-- ===========================================================================

structure Mat2 where
  a11 : Int
  a12 : Int
  a21 : Int
  a22 : Int
  deriving DecidableEq

def matMul (A B : Mat2) : Mat2 :=
  { a11 := A.a11 * B.a11 + A.a12 * B.a21
    a12 := A.a11 * B.a12 + A.a12 * B.a22
    a21 := A.a21 * B.a11 + A.a22 * B.a21
    a22 := A.a21 * B.a12 + A.a22 * B.a22 }

def matId : Mat2 :=
  { a11 := 1, a12 := 0, a21 := 0, a22 := 1 }

def matScale (s : Int) (A : Mat2) : Mat2 :=
  { a11 := s * A.a11, a12 := s * A.a12, a21 := s * A.a21, a22 := s * A.a22 }

def ssF (p : Int) : Mat2 :=
  { a11 := 0, a12 := -p, a21 := 1, a22 := 0 }

def ssV (p : Int) : Mat2 :=
  { a11 := 0, a12 := p, a21 := -1, a22 := 0 }

def ordF (p : Int) : Mat2 :=
  { a11 := 1, a12 := 0, a21 := 0, a22 := p }

def ordV (p : Int) : Mat2 :=
  { a11 := p, a12 := 0, a21 := 0, a22 := 1 }

theorem dieudonne_supersingular_fv_equals_p (p : Int) :
    matMul (ssF p) (ssV p) = matScale p matId ∧
    matMul (ssV p) (ssF p) = matScale p matId := by
  constructor
  · simp [matMul, ssF, ssV, matScale, matId]
  · simp [matMul, ssF, ssV, matScale, matId]

theorem dieudonne_supersingular_f_sq (p : Int) :
    matMul (ssF p) (ssF p) = matScale (-p) matId := by
  simp [matMul, ssF, matScale, matId]

theorem dieudonne_ordinary_fv_equals_p (p : Int) :
    matMul (ordF p) (ordV p) = matScale p matId ∧
    matMul (ordV p) (ordF p) = matScale p matId := by
  constructor
  · simp [matMul, ordF, ordV, matScale, matId]
  · simp [matMul, ordF, ordV, matScale, matId]

/-- Exact binomial coefficient for Divided Power multiplication x^{[m]} * x^{[n]}. -/
def binom : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => binom n k + binom n (k + 1)

theorem pd_mul_2_3_eq_10 : binom (2 + 3) 2 = 10 := by
  decide

-- ===========================================================================
-- 3. Grothendieck Duality & The Six Operations on U --j--> X <--i-- Z
-- ===========================================================================

/-! ## Free abelian sheaves on the finite discrete coproduct U ⊔ Z

A sheaf is given by its stalks. Sections on an open set are all families of
stalk elements there. Morphisms are arbitrary stalkwise additive maps.
Both inclusions are clopen in this model. -/

inductive Basis where
  | standard (rank : Nat)
  | product (left right : Basis)

def Basis.Index : Basis → Type
  | .standard n => Fin n
  | .product a b => a.Index × b.Index

def Basis.rank : Basis → Nat
  | .standard n => n
  | .product a b => a.rank * b.rank

abbrev Vec (A : Basis) := A.Index → Int
abbrev emptyBasis := Basis.standard 0
abbrev Sheaf (X : Type) := X → Basis
abbrev Sections {X : Type} (F : Sheaf X) (O : X → Prop) := (x : X) → O x → Vec (F x)

def restrictSection {X : Type} {F : Sheaf X} {O P : X → Prop} (h : ∀ x, P x → O x)
    (f : Sections F O) : Sections F P := fun x hx => f x (h x hx)

/-- Compatible local sections glue uniquely, for every covering family. -/
theorem sheaf_gluing {X I : Type} (F : Sheaf X) (O : X → Prop) (U : I → X → Prop)
    (inside : ∀ i x, U i x → O x) (cover : ∀ x, O x → ∃ i, U i x)
    (s : ∀ i, Sections F (U i))
    (agree : ∀ i j x (hi : U i x) (hj : U j x), s i x hi = s j x hj) :
    ∃ g : Sections F O, (∀ i, restrictSection (inside i) g = s i) ∧
      ∀ t : Sections F O, (∀ i, restrictSection (inside i) t = s i) → t = g := by
  let g : Sections F O := fun x hx => s (Classical.choose (cover x hx)) x (Classical.choose_spec (cover x hx))
  have hg (i : I) : restrictSection (inside i) g = s i := by
    funext x hx
    apply agree
  refine ⟨g, hg, ?_⟩
  intro t ht
  funext x hx
  obtain ⟨i, hi⟩ := cover x hx
  exact congrFun (congrFun ((ht i).trans (hg i).symm) x) hi

structure HomEquiv (A B : Type) where
  forward : A → B
  inverse : B → A
  left_inv : ∀ x, inverse (forward x) = x
  right_inv : ∀ y, forward (inverse y) = y

structure Enumeration (X : Type) where
  points : List X
  distinct : points.Nodup
  complete : ∀ x, x ∈ points

/-! ## Presheaves on an arbitrary open–closed cut of a finite preorder

Specialization arrows within either stratum may be arbitrary. Arrows crossing
the cut run from closed to open. Their action is the attachment map from a
closed stalk to compatible sections on its open neighbours. -/
namespace Specialization

structure Group where
  carrier : Type
  zero : carrier
  add : carrier → carrier → carrier
  neg : carrier → carrier
  assoc : ∀ x y z, add (add x y) z = add x (add y z)
  comm : ∀ x y, add x y = add y x
  unit : ∀ x, add x zero = x
  cancel : ∀ x, add x (neg x) = zero

structure Linear (A B : Group) where
  val : A.carrier → B.carrier
  unit : val A.zero = B.zero
  additive : ∀ a b, val (A.add a b) = B.add (val a) (val b)

theorem Linear.neg {A B : Group} (f : Linear A B) (x) : f.val (A.neg x) = B.neg (f.val x) := by
  have h : B.add (f.val x) (f.val (A.neg x)) = B.zero := by rw [← f.additive, A.cancel, f.unit]
  have h' := congrArg (B.add (B.neg (f.val x))) (h.trans (B.cancel (f.val x)).symm)
  simp only [← B.assoc, B.comm (B.neg (f.val x)) (f.val x), B.cancel] at h'
  simpa only [B.comm B.zero, B.unit] using h'

def trivial : Group where
  carrier := Unit
  zero := ()
  add := fun _ _ => ()
  neg := fun _ => ()
  assoc := by intros; rfl
  comm := by intros; rfl
  unit := fun _ => rfl
  cancel := fun _ => rfl

def Linear.zero {A B : Group} : Linear A B :=
  ⟨fun _ => B.zero, rfl, fun _ _ => (B.unit _).symm⟩
def Linear.id (A : Group) : Linear A A := ⟨fun x => x, rfl, fun _ _ => rfl⟩
theorem Linear.ext {A B : Group} {f g : Linear A B} (h : f.val = g.val) : f = g := by
  cases f; cases g; congr

def Linear.comp {A B C : Group} (g : Linear B C) (f : Linear A B) : Linear A C :=
  ⟨fun x => g.val (f.val x), by simp [f.unit, g.unit], by intros; simp [f.additive, g.additive]⟩

structure Space where
  Point : Type
  le : Point → Point → Prop
  refl : ∀ x, le x x
  trans : ∀ {x y z}, le x y → le y z → le x z

structure Diagram (S : Space) where
  stalk : S.Point → Group
  res : ∀ {x y}, S.le x y → Linear (stalk x) (stalk y)
  identity : ∀ x a, (res (S.refl x)).val a = a
  transitive : ∀ {x y z} (h : S.le x y) (k : S.le y z) a, (res k).val ((res h).val a) = (res (S.trans h k)).val a

structure Map {S : Space} (F G : Diagram S) where
  app : ∀ x, Linear (F.stalk x) (G.stalk x)
  natural : ∀ {x y} (h : S.le x y) a, (G.res h).val ((app x).val a) = (app y).val ((F.res h).val a)

namespace Map
theorem ext {S : Space} {F G : Diagram S} {f g : Map F G}
    (h : ∀ x a, (f.app x).val a = (g.app x).val a) : f = g := by
  have hh : f.app = g.app := by
    funext x
    exact Linear.ext (funext (h x))
  cases f; cases g; congr

def id {S : Space} (F : Diagram S) : Map F F := ⟨fun _ => Linear.id _, fun _ _ => rfl⟩
def comp {S : Space} {F G H : Diagram S} (g : Map G H) (f : Map F G) : Map F H :=
  ⟨fun x => (g.app x).comp (f.app x), by intros; simp only [Linear.comp, g.natural, f.natural]⟩
def zero {S : Space} {F G : Diagram S} : Map F G :=
  ⟨fun _ => Linear.zero, fun h _ => (G.res h).unit⟩
end Map

def zeroDiagram (S : Space) : Diagram S :=
  ⟨fun _ => trivial, fun _ => Linear.id _, fun _ _ => rfl, fun _ _ _ => rfl⟩

/-- For a cut of a preorder, link z u means z ≤ u. These are precisely the
    two composition laws required of the crossing specialization arrows. -/
structure Cut (U Z : Space) where
  link : Z.Point → U.Point → Prop
  left : ∀ {x y u}, Z.le x y → link y u → link x u
  right : ∀ {z u v}, link z u → U.le u v → link z v

variable {P Q : Space} (C : Cut P Q)

/-- Nearby sections are the limit over open points above z. -/
structure Nearby (F : Diagram P) (z : Q.Point) where
  value : ∀ u, C.link z u → (F.stalk u).carrier
  coherent : ∀ {u v} (hu : C.link z u) (h : P.le u v), (F.res h).val (value u hu) = value v (C.right hu h)

theorem Nearby.ext {F : Diagram P} {z : Q.Point} {s t : Nearby C F z}
    (h : ∀ u hu, s.value u hu = t.value u hu) : s = t := by
  have he : s.value = t.value := funext fun u => funext (h u)
  cases s; cases t; congr

def nearbyGroup (F : Diagram P) (z : Q.Point) : Group where
  carrier := Nearby C F z
  zero := ⟨fun u _ => (F.stalk u).zero, fun _ h => (F.res h).unit⟩
  add s t := ⟨fun u hu => (F.stalk u).add (s.value u hu) (t.value u hu),
    fun hu h => by simp only [(F.res h).additive, s.coherent, t.coherent]⟩
  neg s := ⟨fun u hu => (F.stalk u).neg (s.value u hu), fun hu h => by rw [Linear.neg, s.coherent]⟩
  assoc := fun _ _ _ => Nearby.ext C fun u _ => (F.stalk u).assoc ..
  comm := fun _ _ => Nearby.ext C fun u _ => (F.stalk u).comm ..
  unit := fun _ => Nearby.ext C fun u _ => (F.stalk u).unit ..
  cancel := fun _ => Nearby.ext C fun u _ => (F.stalk u).cancel ..

def nearby (F : Diagram P) : Diagram Q where
  stalk := nearbyGroup C F
  res h := ⟨fun s => ⟨fun u hu => s.value u (C.left h hu), fun hu k => s.coherent (C.left h hu) k⟩,
    rfl, fun _ _ => rfl⟩
  identity := fun _ _ => rfl
  transitive := fun _ _ _ => rfl

def nearbyMap {F G : Diagram P} (f : Map F G) : Map (nearby C F) (nearby C G) where
  app _ := ⟨fun s => ⟨fun u hu => (f.app u).val (s.value u hu),
    fun hu h => ((f.natural h _).trans (congrArg _ (s.coherent hu h)))⟩,
    Nearby.ext C (fun u _ => (f.app u).unit), fun _ _ => Nearby.ext C (fun u _ => (f.app u).additive ..)⟩
  natural := fun _ _ => rfl

structure Presheaf where
  openPart : Diagram P
  closedPart : Diagram Q
  attach : Map closedPart (nearby C openPart)

structure Arrow (F G : Presheaf C) where
  onU : Map F.openPart G.openPart
  onZ : Map F.closedPart G.closedPart
  square : ∀ z x, (G.attach.app z).val ((onZ.app z).val x) =
    ((nearbyMap C onU).app z).val ((F.attach.app z).val x)

theorem Arrow.ext {F G : Presheaf C} {f g : Arrow C F G} (hu : f.onU = g.onU) (hz : f.onZ = g.onZ) : f = g := by
  cases f; cases g; congr

def Arrow.comp {F G H : Presheaf C} (g : Arrow C G H) (f : Arrow C F G) : Arrow C F H :=
  ⟨g.onU.comp f.onU, g.onZ.comp f.onZ, fun z x => (g.square z _).trans
    (congrArg (((nearbyMap C g.onU).app z).val) (f.square z x))⟩

def jPull (F : Presheaf C) := F.openPart
def iPull (F : Presheaf C) := F.closedPart
def jShriek (A : Diagram P) : Presheaf C := ⟨A, zeroDiagram Q, Map.zero⟩
def jStar (A : Diagram P) : Presheaf C := ⟨A, nearby C A, Map.id _⟩
def iStar (B : Diagram Q) : Presheaf C := ⟨zeroDiagram P, B, Map.zero⟩

/-- Sections on the closed part with vanishing generization on the open part. -/
def kernelStalk (F : Presheaf C) (z : Q.Point) : Group where
  carrier := {x : (F.closedPart.stalk z).carrier // (F.attach.app z).val x = (nearbyGroup C F.openPart z).zero}
  zero := ⟨_, (F.attach.app z).unit⟩
  add x y := ⟨(F.closedPart.stalk z).add x.val y.val, by rw [(F.attach.app z).additive, x.property, y.property]; exact (nearbyGroup C F.openPart z).unit _⟩
  neg x := ⟨(F.closedPart.stalk z).neg x.val, by
    rw [(F.attach.app z).neg, x.property]
    apply Nearby.ext C
    intro u _
    exact ((F.openPart.stalk u).unit _).symm.trans
                    (((F.openPart.stalk u).comm ..).trans ((F.openPart.stalk u).cancel _))⟩
  assoc := fun _ _ _ => Subtype.eq ((F.closedPart.stalk z).assoc ..)
  comm := fun _ _ => Subtype.eq ((F.closedPart.stalk z).comm ..)
  unit := fun _ => Subtype.eq ((F.closedPart.stalk z).unit ..)
  cancel := fun _ => Subtype.eq ((F.closedPart.stalk z).cancel ..)

def iShriek (F : Presheaf C) : Diagram Q where
  stalk := kernelStalk C F
  res h := ⟨fun x => ⟨(F.closedPart.res h).val x.val, by
    rw [← F.attach.natural h, x.property]; exact ((nearby C F.openPart).res h).unit⟩,
    Subtype.eq ((F.closedPart.res h).unit), fun _ _ => Subtype.eq ((F.closedPart.res h).additive ..)⟩
  identity := fun _ _ => Subtype.eq (F.closedPart.identity ..)
  transitive := fun _ _ _ => Subtype.eq (F.closedPart.transitive ..)

def Arrow.id (F : Presheaf C) : Arrow C F F := ⟨Map.id _, Map.id _, fun _ _ => rfl⟩

def jShriekMap {A B : Diagram P} (f : Map A B) : Arrow C (jShriek C A) (jShriek C B) :=
  ⟨f, Map.id _, fun z _ => ((nearbyMap C f).app z).unit.symm⟩
def jStarMap {A B : Diagram P} (f : Map A B) : Arrow C (jStar C A) (jStar C B) :=
  ⟨f, nearbyMap C f, fun _ _ => rfl⟩
def iStarMap {A B : Diagram Q} (f : Map A B) : Arrow C (iStar C A) (iStar C B) :=
  ⟨Map.id _, f, fun _ _ => rfl⟩

def iShriekMap {F G : Presheaf C} (f : Arrow C F G) : Map (iShriek C F) (iShriek C G) where
  app z := ⟨fun x => ⟨(f.onZ.app z).val x.val, (f.square z x.val).trans
    ((congrArg _ x.property).trans (((nearbyMap C f.onU).app z).unit))⟩,
    Subtype.eq ((f.onZ.app z).unit), fun x y => Subtype.eq ((f.onZ.app z).additive x.val y.val)⟩
  natural h x := Subtype.eq (f.onZ.natural h x.val)

theorem extension_identities (A : Diagram P) (B : Diagram Q) (F : Presheaf C) :
    jShriekMap C (Map.id A) = Arrow.id C _ ∧ jStarMap C (Map.id A) = Arrow.id C _ ∧
    iStarMap C (Map.id B) = Arrow.id C _ ∧ iShriekMap C (Arrow.id C F) = Map.id _ := by
  exact ⟨rfl,rfl,rfl,rfl⟩

theorem extension_composition {A B D : Diagram P} (f : Map A B) (g : Map B D) :
    jShriekMap C (g.comp f) = Arrow.comp C (jShriekMap C g) (jShriekMap C f) ∧
    jStarMap C (g.comp f) = Arrow.comp C (jStarMap C g) (jStarMap C f) := ⟨rfl,rfl⟩

theorem closed_composition {A B D : Diagram Q} (f : Map A B) (g : Map B D) :
    iStarMap C (g.comp f) = Arrow.comp C (iStarMap C g) (iStarMap C f) := rfl

theorem kernel_composition {F G H : Presheaf C} (f : Arrow C F G) (g : Arrow C G H) :
    iShriekMap C (Arrow.comp C g f) = (iShriekMap C g).comp (iShriekMap C f) := rfl

private theorem out_of_zero {S : Space} {B : Diagram S} (f : Map (zeroDiagram S) B) :
    f = Map.zero := Map.ext fun z _ => (f.app z).unit

private theorem into_zero {S : Space} {B : Diagram S} (f : Map B (zeroDiagram S)) :
    f = Map.zero := Map.ext fun _ _ => rfl

def open_left (A : Diagram P) (F : Presheaf C) :
    HomEquiv (Arrow C (jShriek C A) F) (Map A (jPull C F)) where
  forward f := f.onU
  inverse f := ⟨f, Map.zero, fun z _ => (F.attach.app z).unit.trans (((nearbyMap C f).app z).unit.symm)⟩
  left_inv f := Arrow.ext C rfl (out_of_zero f.onZ).symm
  right_inv _ := rfl

def open_right (F : Presheaf C) (A : Diagram P) :
    HomEquiv (Arrow C F (jStar C A)) (Map (jPull C F) A) where
  forward f := f.onU
  inverse f := ⟨f, (nearbyMap C f).comp F.attach, fun _ _ => rfl⟩
  left_inv f := Arrow.ext C rfl (Map.ext fun z x => (f.square z x).symm)
  right_inv _ := rfl

def closed_left (F : Presheaf C) (B : Diagram Q) :
    HomEquiv (Arrow C F (iStar C B)) (Map (iPull C F) B) where
  forward f := f.onZ
  inverse f := ⟨Map.zero, f, fun _ _ => Nearby.ext C fun _ _ => rfl⟩
  left_inv f := Arrow.ext C (into_zero f.onU).symm rfl
  right_inv _ := rfl

def sectionsSupported {B : Diagram Q} {F : Presheaf C} (f : Arrow C (iStar C B) F) :
    Map B (iShriek C F) where
  app z := ⟨fun x => ⟨(f.onZ.app z).val x, (f.square z x).trans (((nearbyMap C f.onU).app z).unit)⟩,
    Subtype.eq ((f.onZ.app z).unit), fun _ _ => Subtype.eq ((f.onZ.app z).additive ..)⟩
  natural h x := Subtype.eq (f.onZ.natural h x)

def closed_right (B : Diagram Q) (F : Presheaf C) :
    HomEquiv (Arrow C (iStar C B) F) (Map B (iShriek C F)) where
  forward := sectionsSupported C
  inverse f := ⟨Map.zero,
    ⟨fun z => ⟨fun x => ((f.app z).val x).val, congrArg Subtype.val (f.app z).unit,
      fun x y => congrArg Subtype.val ((f.app z).additive x y)⟩,
      fun h x => congrArg Subtype.val (f.natural h x)⟩,
    fun z x => ((f.app z).val x).property.trans (((nearbyMap C Map.zero).app z).unit.symm)⟩
  left_inv f := Arrow.ext C (out_of_zero f.onU).symm rfl
  right_inv _ := rfl

theorem open_left_natural {A B : Diagram P} {F G : Presheaf C} (a : Map A B) (b : Arrow C F G)
    (f : Arrow C (jShriek C B) F) :
    (open_left C A G).forward (Arrow.comp C b (Arrow.comp C f (jShriekMap C a))) =
    b.onU.comp (((open_left C B F).forward f).comp a) := rfl

theorem open_right_natural {A B : Diagram P} {F G : Presheaf C} (a : Map A B) (b : Arrow C F G)
    (f : Arrow C G (jStar C A)) :
    (open_right C F B).forward (Arrow.comp C (jStarMap C a) (Arrow.comp C f b)) =
    a.comp (((open_right C G A).forward f).comp b.onU) := rfl

theorem closed_left_natural {A B : Diagram Q} {F G : Presheaf C} (a : Map A B) (b : Arrow C F G)
    (f : Arrow C G (iStar C A)) :
    (closed_left C F B).forward (Arrow.comp C (iStarMap C a) (Arrow.comp C f b)) =
    a.comp (((closed_left C G A).forward f).comp b.onZ) := rfl

theorem closed_right_natural {A B : Diagram Q} {F G : Presheaf C} (a : Map A B) (b : Arrow C F G)
    (f : Arrow C (iStar C B) F) :
    (closed_right C A G).forward (Arrow.comp C b (Arrow.comp C f (iStarMap C a))) =
    (iShriekMap C b).comp (((closed_right C B F).forward f).comp a) := rfl

theorem proper_base_change (A : Diagram P) (B : Diagram Q) :
    iPull C (jShriek C A) = zeroDiagram Q ∧ jPull C (iStar C B) = zeroDiagram P ∧
    iPull C (iStar C B) = B := ⟨rfl,rfl,rfl⟩

/-- Every stalkwise rank with rank(0)=0 gives the cellular Euler valuation,
    with weights such as (-1)^dimension for locally closed cells. -/
def valuation {S : Space} (rank : Group → Int) (weight : S.Point → Int) (F : Diagram S) :
    List S.Point → Int
  | [] => 0
  | x :: xs => weight x * rank (F.stalk x) + valuation rank weight F xs

def euler (rank : Group → Int) (u : Enumeration P.Point) (z : Enumeration Q.Point)
    (wu : P.Point → Int) (wz : Q.Point → Int) (F : Presheaf C) :=
  valuation rank wu F.openPart u.points + valuation rank wz F.closedPart z.points

private theorem zero_valuation {S : Space} (rank : Group → Int) (h : rank trivial = 0)
    (w : S.Point → Int) (xs : List S.Point) : valuation rank w (zeroDiagram S) xs = 0 := by
  induction xs with
  | nil => rfl
  | cons x xs ih => change w x * rank trivial + _ = 0; rw [h, Int.mul_zero, ih, Int.add_zero]

theorem euler_excision (rank : Group → Int) (h : rank trivial = 0)
    (u : Enumeration P.Point) (z : Enumeration Q.Point) (wu : P.Point → Int) (wz : Q.Point → Int)
    (F : Presheaf C) : euler C rank u z wu wz F =
    euler C rank u z wu wz (jShriek C (jPull C F)) + euler C rank u z wu wz (iStar C (iPull C F)) := by
  simp [euler, jShriek, iStar, jPull, iPull, zero_valuation rank h]

/-! The integral tensor product is the group of formal sums of symbols
    a ⊗ b modulo equality under all biadditive evaluations. -/
structure Bilinear (A B D : Group) where
  val : A.carrier → B.carrier → D.carrier
  zero_l : ∀ b, val A.zero b = D.zero
  zero_r : ∀ a, val a B.zero = D.zero
  add_l : ∀ x y b, val (A.add x y) b = D.add (val x b) (val y b)
  add_r : ∀ a x y, val a (B.add x y) = D.add (val a x) (val a y)

namespace Tensor
inductive Word (A B : Group) where
  | pure (a : A.carrier) (b : B.carrier)
  | zero
  | plus (x y : Word A B)
  | neg (x : Word A B)

def eval {A B D : Group} (f : Bilinear A B D) : Word A B → D.carrier
  | .pure a b => f.val a b
  | .zero => D.zero
  | .plus x y => D.add (eval f x) (eval f y)
  | .neg x => D.neg (eval f x)

def Rel {A B : Group} (x y : Word A B) := ∀ D (f : Bilinear A B D), eval f x = eval f y
abbrev Elem (A B : Group) := Quot (@Rel A B)
abbrev mk {A B : Group} (x : Word A B) : Elem A B := Quot.mk Rel x

def plus {A B : Group} (x y : Elem A B) : Elem A B :=
  Quot.liftOn x (fun a => Quot.liftOn y (fun b => mk (.plus a b)) (fun _ _ h =>
    Quot.sound fun D f => congrArg (D.add (eval f a)) (h D f))) (fun _ _ h => by
      induction y using Quot.inductionOn with
      | _ b => exact Quot.sound fun D f => congrArg (fun t => D.add t (eval f b)) (h D f))

def neg {A B : Group} (x : Elem A B) : Elem A B :=
  Quot.liftOn x (fun a => mk (.neg a)) (fun _ _ h => Quot.sound fun D f => congrArg D.neg (h D f))

def group (A B : Group) : Group where
  carrier := Elem A B
  zero := mk .zero
  add := plus
  neg := neg
  assoc x y z := Quot.inductionOn x fun _ => Quot.inductionOn y fun _ => Quot.inductionOn z fun _ =>
    Quot.sound fun D f => D.assoc (eval f _) (eval f _) (eval f _)
  comm x y := Quot.inductionOn x fun _ => Quot.inductionOn y fun _ =>
    Quot.sound fun D f => D.comm (eval f _) (eval f _)
  unit x := Quot.inductionOn x fun _ => Quot.sound fun D f => D.unit (eval f _)
  cancel x := Quot.inductionOn x fun _ => Quot.sound fun D f => D.cancel (eval f _)

def pure (A B : Group) : Bilinear A B (group A B) where
  val a b := mk (.pure a b)
  zero_l b := Quot.sound fun _ f => f.zero_l b
  zero_r a := Quot.sound fun _ f => f.zero_r a
  add_l x y b := Quot.sound fun _ f => f.add_l x y b
  add_r a x y := Quot.sound fun _ f => f.add_r a x y

def lift {A B D : Group} (f : Bilinear A B D) : Linear (group A B) D where
  val x := Quot.liftOn x (eval f) (fun _ _ h => h D f)
  unit := rfl
  additive x y := Quot.inductionOn x fun _ => Quot.inductionOn y fun _ => rfl

theorem hom_ext {A B D : Group} {f g : Linear (group A B) D}
    (h : ∀ a b, f.val ((pure A B).val a b) = g.val ((pure A B).val a b)) : f = g := by
  apply Linear.ext; funext x
  induction x using Quot.inductionOn with
  | _ t =>
    induction t with
    | pure a b => exact h a b
    | zero => exact f.unit.trans g.unit.symm
    | plus u v hu hv =>
      change f.val ((group A B).add (mk u) (mk v)) = g.val ((group A B).add (mk u) (mk v))
      rw [f.additive, g.additive, hu, hv]
    | neg u hu =>
      change f.val ((group A B).neg (mk u)) = g.val ((group A B).neg (mk u))
      rw [f.neg, g.neg, hu]

def map {A B A' B' : Group} (f : Linear A A') (g : Linear B B') : Linear (group A B) (group A' B') :=
  lift ⟨fun a b => (pure A' B').val (f.val a) (g.val b),
    fun b => by simp only [f.unit, Bilinear.zero_l], fun a => by simp only [g.unit, Bilinear.zero_r],
    fun _ _ _ => by simp only [f.additive, Bilinear.add_l],
    fun _ _ _ => by simp only [g.additive, Bilinear.add_r]⟩

theorem map_square {A B A' B' D E : Group}
    {f : Linear A A'} {g : Linear B B'} {h : Linear A' D} {k : Linear B' E}
    {u : Linear A D} {v : Linear B E} (hf : ∀ x, h.val (f.val x) = u.val x)
    (hg : ∀ y, k.val (g.val y) = v.val y) (t : Elem A B) :
    (map h k).val ((map f g).val t) = (map u v).val t := by
  have eq : (map h k).comp (map f g) = map u v := by
    apply hom_ext; intro x y
    exact (congrArg (fun a => (pure D E).val a (k.val (g.val y))) (hf x)).trans
      (congrArg ((pure D E).val (u.val x)) (hg y))
  exact congrArg (fun l => l.val t) eq

theorem empty_left {B : Group} (x : Elem trivial B) : x = (group trivial B).zero := by
  have h : Linear.id (group trivial B) = Linear.zero := hom_ext fun _ b => (pure trivial B).zero_l b
  exact congrArg (fun l => l.val x) h

end Tensor

def tensorDiagram {S : Space} (F G : Diagram S) : Diagram S where
  stalk x := Tensor.group (F.stalk x) (G.stalk x)
  res h := Tensor.map (F.res h) (G.res h)
  identity x a := by
    have h : Tensor.map (F.res (S.refl x)) (G.res (S.refl x)) = Linear.id _ := by
      apply Tensor.hom_ext; intro u v
      change (Tensor.pure _ _).val _ _ = (Tensor.pure _ _).val u v
      rw [F.identity, G.identity]
    exact congrArg (fun f => f.val a) h
  transitive h k a := Tensor.map_square (F.transitive h k) (G.transitive h k) a

def crossing (F : Presheaf C) (z u) (h : C.link z u) : Linear (F.closedPart.stalk z) (F.openPart.stalk u) where
  val x := ((F.attach.app z).val x).value u h
  unit := congrArg (fun s => s.value u h) (F.attach.app z).unit
  additive x y := congrArg (fun s => s.value u h) ((F.attach.app z).additive x y)

def tensor (F G : Presheaf C) : Presheaf C where
  openPart := tensorDiagram F.openPart G.openPart
  closedPart := tensorDiagram F.closedPart G.closedPart
  attach :=
    { app := fun z =>
        { val := fun x =>
            { value := fun u h => (Tensor.map (crossing C F z u h) (crossing C G z u h)).val x
              coherent := fun hu k => Tensor.map_square
                (fun a => ((F.attach.app z).val a).coherent hu k)
                (fun b => ((G.attach.app z).val b).coherent hu k) x }
          unit := Nearby.ext C fun _ h => (Tensor.map (crossing C F _ _ h) (crossing C G _ _ h)).unit
          additive := fun x y => Nearby.ext C fun _ h =>
            (Tensor.map (crossing C F _ _ h) (crossing C G _ _ h)).additive x y }
      natural := fun k x => Nearby.ext C fun u h => Eq.symm (Tensor.map_square
        (fun a => congrArg (fun s => s.value u h) (F.attach.natural k a).symm)
        (fun b => congrArg (fun s => s.value u h) (G.attach.natural k b).symm) x) }

structure Iso (F G : Presheaf C) where
  forward : Arrow C F G
  inverse : Arrow C G F
  left_inv : Arrow.comp C inverse forward = Arrow.id C F
  right_inv : Arrow.comp C forward inverse = Arrow.id C G

/-- The comparison is an isomorphism of diagrams, including all crossing
    maps. Its inverse uses the universal relation 0 ⊗ G_z = 0. -/
def projection_formula (A : Diagram P) (G : Presheaf C) :
    Iso C (jShriek C (tensorDiagram A (jPull C G))) (tensor C (jShriek C A) G) where
  forward := ⟨Map.id _, Map.zero, fun z _ => ((tensor C (jShriek C A) G).attach.app z).unit⟩
  inverse := ⟨Map.id _, Map.zero, fun z x => by
    have hx := Tensor.empty_left x
    change _ = (((tensor C (jShriek C A) G).attach.app z).val x)
    rw [hx]; exact ((tensor C (jShriek C A) G).attach.app z).unit.symm⟩
  left_inv := rfl
  right_inv := Arrow.ext C rfl (Map.ext fun _ x => (Tensor.empty_left x).symm)

end Specialization

end EshkolGrothendieckPillarsSixOps
