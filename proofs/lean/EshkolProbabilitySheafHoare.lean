/-
  Independent pullbacks, common-refinement equality of extension representatives,
  and Hoare rules for predicates stable under sample-space extension.
  The extension and Hoare constructions follow Ho, Wu and Raad,
  “Sheaf Semantics of Probabilistic Programs” (arXiv:2611.30692).

  Model-to-Implementation Correspondence
  lib/core/sheaf.esk exposes finite-site restriction and gluing, exercised by
  tests/stdlib/sheaf_test.esk. In lib/core/grothendieck/probability_sheaves.esk,
  gpsh-indep-pullback constructs a conditional product on pairs of atoms.
  Its coordinate projections motivate the unique mediator for independent cones.
  gpsh-hoare-holds? checks finite extensions and gpsh-case-mask-mutual-independence
  enumerates the XOR law. tests/stdlib/grothendieck_probability_sheaves_test.esk
  checks the finite controls. The categorical proofs below take the independent
  pullback property as a hypothesis on the chosen category.
-/

namespace EshkolProbabilitySheafHoare

/-! ## 1. Independent cones and their universal spans -/

universe u

structure SmallCat where
  Obj : Type u
  Hom : Obj → Obj → Type
  id  : ∀ (A : Obj), Hom A A
  comp : ∀ {A B C : Obj}, Hom B C → Hom A B → Hom A C
  id_comp : ∀ {A B : Obj} (f : Hom A B), comp (id B) f = f
  comp_id : ∀ {A B : Obj} (f : Hom A B), comp f (id A) = f
  comp_assoc : ∀ {A B C D : Obj} (h : Hom C D) (g : Hom B C) (f : Hom A B),
    comp h (comp g f) = comp (comp h g) f

/-- A commuting span over a cospan. -/
structure SpanData (C : SmallCat) {W1 W2 W : C.Obj}
    (f : C.Hom W1 W) (g : C.Hom W2 W) where
  apex : C.Obj
  p1 : C.Hom apex W1
  p2 : C.Hom apex W2
  comm : C.comp f p1 = C.comp g p2

/-- A map into the apex preserving the two specified legs. -/
def SpanData.Factors {C : SmallCat} {A B Z X : C.Obj}
    {f : C.Hom A Z} {g : C.Hom B Z} (P : SpanData C f g)
    (s : C.Hom X A) (t : C.Hom X B) (u : C.Hom X P.apex) : Prop :=
  C.comp P.p1 u = s ∧ C.comp P.p2 u = t

/-- Every independent cone has a unique factorization into the chosen span. -/
structure IndepPullbackStruct (C : SmallCat) where
  Indep : ∀ {W1 W2 W3 W4 : C.Obj},
    C.Hom W1 W2 → C.Hom W1 W3 → C.Hom W2 W4 → C.Hom W3 W4 → Prop
  ip1 : ∀ {W1 W2 : C.Obj} (f : C.Hom W1 W2),
    Indep f (C.id W1) (C.id W2) f
  ip2 : ∀ {W1 W2 W3 W4 : C.Obj}
    {f : C.Hom W1 W2} {h : C.Hom W1 W3} {g : C.Hom W2 W4} {i : C.Hom W3 W4},
    Indep f h g i → Indep h f i g
  ip3 : ∀ {W1 W2 W3 W4 W5 W6 : C.Obj}
    {t : C.Hom W1 W3} {s : C.Hom W1 W2} {v : C.Hom W3 W4} {u : C.Hom W2 W4}
    {t' : C.Hom W3 W5} {g : C.Hom W5 W6} {f : C.Hom W4 W6},
    Indep t s v u → Indep t' v g f → Indep (C.comp t' t) s g (C.comp f u)
  pb : ∀ {W1 W2 W : C.Obj} (f : C.Hom W1 W) (g : C.Hom W2 W), SpanData C f g
  pb_indep : ∀ {W1 W2 W : C.Obj} (f : C.Hom W1 W) (g : C.Hom W2 W),
    Indep (pb f g).p1 (pb f g).p2 f g
  pb_univ : ∀ {W1 W2 W W' : C.Obj} (f : C.Hom W1 W) (g : C.Hom W2 W)
    (s : C.Hom W' W1) (t : C.Hom W' W2),
    Indep s t f g →
    ∃ u : C.Hom W' (pb f g).apex,
      (pb f g).Factors s t u ∧ ∀ v, (pb f g).Factors s t v → v = u

/-- The same universal property for an arbitrary span over the cospan. -/
structure IsIndependentPullback {C : SmallCat} (I : IndepPullbackStruct C)
    {A B Z : C.Obj} {f : C.Hom A Z} {g : C.Hom B Z} (P : SpanData C f g) : Prop where
  independent : I.Indep P.p1 P.p2 f g
  universal : ∀ {X : C.Obj} (s : C.Hom X A) (t : C.Hom X B), I.Indep s t f g →
    ∃ u, P.Factors s t u ∧ ∀ v, P.Factors s t v → v = u

theorem chosen_isIndependentPullback {C : SmallCat} (I : IndepPullbackStruct C)
    {A B Z : C.Obj} (f : C.Hom A Z) (g : C.Hom B Z) :
    IsIndependentPullback I (I.pb f g) :=
  ⟨I.pb_indep f g, I.pb_univ f g⟩

/-- An endomorphism of a universal span preserving its projections is the identity. -/
theorem pullback_endo_eq_id {C : SmallCat} {I : IndepPullbackStruct C}
    {A B Z : C.Obj} {f : C.Hom A Z} {g : C.Hom B Z}
    {P : SpanData C f g} (hP : IsIndependentPullback I P)
    {e : C.Hom P.apex P.apex} (he : P.Factors P.p1 P.p2 e) : e = C.id P.apex := by
  obtain ⟨_, _, uniq⟩ := hP.universal _ _ hP.independent
  exact (uniq e he).trans (uniq _ ⟨C.comp_id _, C.comp_id _⟩).symm

/-- Projection-preserving maps in both directions with inverse equations. -/
structure SpanIso {C : SmallCat} {A B Z : C.Obj} {f : C.Hom A Z} {g : C.Hom B Z}
    (P Q : SpanData C f g) where
  hom : C.Hom P.apex Q.apex
  inv : C.Hom Q.apex P.apex
  hom_factors : Q.Factors P.p1 P.p2 hom
  inv_factors : P.Factors Q.p1 Q.p2 inv
  inv_hom : C.comp inv hom = C.id P.apex
  hom_inv : C.comp hom inv = C.id Q.apex

theorem pullbacks_unique_up_to_unique_iso {C : SmallCat} {I : IndepPullbackStruct C}
    {A B Z : C.Obj} {f : C.Hom A Z} {g : C.Hom B Z} {P Q : SpanData C f g}
    (hP : IsIndependentPullback I P) (hQ : IsIndependentPullback I Q) :
    ∃ iso : SpanIso P Q, ∀ other : SpanIso P Q, other = iso := by
  obtain ⟨u, hu, uniqueU⟩ := hQ.universal _ _ hP.independent
  obtain ⟨v, hv, uniqueV⟩ := hP.universal _ _ hQ.independent
  have vu : C.comp v u = C.id P.apex := pullback_endo_eq_id hP
    ⟨by rw [C.comp_assoc, hv.1, hu.1], by rw [C.comp_assoc, hv.2, hu.2]⟩
  have uv : C.comp u v = C.id Q.apex := pullback_endo_eq_id hQ
    ⟨by rw [C.comp_assoc, hu.1, hv.1], by rw [C.comp_assoc, hu.2, hv.2]⟩
  refine ⟨⟨u, v, hu, hv, vu, uv⟩, ?_⟩
  intro other
  cases other with
  | mk w z hw hz _ _ =>
    cases uniqueU w hw
    cases uniqueV z hz
    rfl

/-- Concrete carrier example. Objects are small types and arrows are functions.
    Pairing gives the mediator pointwise on compatible pairs. -/
def carrierCat : SmallCat.{1} where
  Obj := Type
  Hom := fun A B => A → B
  id := fun _ x => x
  comp := fun g f x => g (f x)
  id_comp := fun _ => rfl
  comp_id := fun _ => rfl
  comp_assoc := fun _ _ _ => rfl

/-- On unweighted carriers take every commuting square as independent. -/
def carrierPullbacks : IndepPullbackStruct carrierCat where
  Indep := fun s t f g => ∀ x, f (s x) = g (t x)
  ip1 := fun _ _ => rfl
  ip2 := fun h x => (h x).symm
  ip3 := fun { _ _ _ _ _ _ } {t _ _ _ _ _ f} h k x => (k (t x)).trans (congrArg f (h x))
  pb := fun f g =>
    { apex := {p : _ × _ // f p.1 = g p.2}
      p1 := fun p => p.val.1
      p2 := fun p => p.val.2
      comm := funext (fun p => p.property) }
  pb_indep := fun _ _ p => p.property
  pb_univ := fun _ _ s t h =>
    ⟨fun x => ⟨(s x, t x), h x⟩, ⟨rfl, rfl⟩, fun _ hv =>
      funext (fun x => Subtype.ext (Prod.ext (congrFun hv.1 x) (congrFun hv.2 x)))⟩

/-- Contravariant presheaf. -/
structure PSh (C : SmallCat) where
  val : C.Obj → Type
  act : ∀ {W W' : C.Obj}, val W → C.Hom W' W → val W'
  act_id : ∀ {W : C.Obj} (x : val W), act x (C.id W) = x
  act_comp : ∀ {W1 W2 W3 : C.Obj} (x : val W3) (f : C.Hom W2 W3) (g : C.Hom W1 W2),
    act (act x f) g = act x (C.comp f g)

/-- A section agrees along every pair of arrows equalized by s. -/
def IsSInvariant {C : SmallCat} (F : PSh C) {W' W : C.Obj}
    (s : C.Hom W' W) (y : F.val W') : Prop :=
  ∀ {W'' : C.Obj} (u v : C.Hom W'' W'),
    C.comp s u = C.comp s v → F.act y u = F.act y v

/-- Every invariant section descends uniquely. -/
def IsAtomicSheaf {C : SmallCat} (F : PSh C) : Prop :=
  ∀ {W' W : C.Obj} (s : C.Hom W' W) (y : F.val W'),
    IsSInvariant F s y →
    ∃ x : F.val W, F.act x s = y ∧ ∀ x' : F.val W, F.act x' s = y → x' = x

/-- Constant presheaf on a type A. -/
def constPSh (C : SmallCat) (A : Type) : PSh C where
  val := fun _ => A
  act := fun a _ => a
  act_id := fun _ => rfl
  act_comp := fun _ _ _ => rfl

/-- Product presheaf `F x G`. -/
def prodPSh {C : SmallCat} (F G : PSh C) : PSh C where
  val := fun W => F.val W × G.val W
  act := fun p f => (F.act p.1 f, G.act p.2 f)
  act_id := fun p => Prod.ext (F.act_id p.1) (G.act_id p.2)
  act_comp := fun p f g => Prod.ext (F.act_comp p.1 f g) (G.act_comp p.2 f g)

/-- Every constant presheaf is an atomic sheaf (Section 4.1, footnote 13 & 14). -/
theorem constPSh_isAtomicSheaf (C : SmallCat) (A : Type) :
    IsAtomicSheaf (constPSh C A) := by
  intro W' W s y _
  refine ⟨y, rfl, ?_⟩
  intro x' hx'
  exact hx'

/-- Compatible pairs descend by descending each coordinate along the same arrow. -/
theorem prodPSh_isAtomicSheaf {C : SmallCat} {F G : PSh C}
    (hF : IsAtomicSheaf F) (hG : IsAtomicSheaf G) : IsAtomicSheaf (prodPSh F G) := by
  intro X Y s y hy
  obtain ⟨a, ha, ua⟩ := hF s y.1 (fun u v h => congrArg Prod.fst (hy u v h))
  obtain ⟨b, hb, ub⟩ := hG s y.2 (fun u v h => congrArg Prod.snd (hy u v h))
  exact ⟨(a, b), Prod.ext ha hb, fun p hp =>
    Prod.ext (ua p.1 (congrArg Prod.fst hp)) (ub p.2 (congrArg Prod.snd hp))⟩

/-! ## 2. Unit, bind and independent pairing on extension representatives -/

/-- Representative `(W', f : W' -> W, x : F(W'))` of `[f, x] in G F(W)`. -/
structure ExtRep (C : SmallCat) (F : PSh C) (W : C.Obj) where
  stage : C.Obj
  proj  : C.Hom stage W
  elem  : F.val stage

/-- Lemma 3.18 common-refinement equivalence `[f, x] ~ [g, y]` in `G F(W)`. -/
def ExtEq {C : SmallCat} {F : PSh C} {W : C.Obj}
    (a b : ExtRep C F W) : Prop :=
  ∃ (W3 : C.Obj) (p1 : C.Hom W3 a.stage) (p2 : C.Hom W3 b.stage),
    C.comp a.proj p1 = C.comp b.proj p2 ∧
    F.act a.elem p1 = F.act b.elem p2

theorem extEq_refl {C : SmallCat} {F : PSh C} {W : C.Obj} (a : ExtRep C F W) :
    ExtEq a a :=
  ⟨a.stage, C.id a.stage, C.id a.stage, rfl, rfl⟩

theorem extEq_symm {C : SmallCat} {F : PSh C} {W : C.Obj}
    {a b : ExtRep C F W} (h : ExtEq a b) : ExtEq b a := by
  rcases h with ⟨W3, p1, p2, hcomm, helem⟩
  exact ⟨W3, p2, p1, hcomm.symm, helem.symm⟩

theorem extEq_trans {C : SmallCat} (I : IndepPullbackStruct C)
    {F : PSh C} {W : C.Obj} {a b c : ExtRep C F W}
    (hab : ExtEq a b) (hbc : ExtEq b c) : ExtEq a c := by
  rcases hab with ⟨W12, u1, u2, h12, e12⟩
  rcases hbc with ⟨W23, v2, v3, h23, e23⟩
  let sp := I.pb u2 v2
  refine ⟨sp.apex, C.comp u1 sp.p1, C.comp v3 sp.p2, ?_, ?_⟩
  · calc
      C.comp a.proj (C.comp u1 sp.p1)
          = C.comp (C.comp a.proj u1) sp.p1 := C.comp_assoc a.proj u1 sp.p1
      _   = C.comp (C.comp b.proj u2) sp.p1 := by rw [h12]
      _   = C.comp b.proj (C.comp u2 sp.p1) := (C.comp_assoc b.proj u2 sp.p1).symm
      _   = C.comp b.proj (C.comp v2 sp.p2) := by rw [sp.comm]
      _   = C.comp (C.comp b.proj v2) sp.p2 := C.comp_assoc b.proj v2 sp.p2
      _   = C.comp (C.comp c.proj v3) sp.p2 := by rw [h23]
      _   = C.comp c.proj (C.comp v3 sp.p2) := (C.comp_assoc c.proj v3 sp.p2).symm
  · calc
      F.act a.elem (C.comp u1 sp.p1)
          = F.act (F.act a.elem u1) sp.p1 := (F.act_comp a.elem u1 sp.p1).symm
      _   = F.act (F.act b.elem u2) sp.p1 := by rw [e12]
      _   = F.act b.elem (C.comp u2 sp.p1) := F.act_comp b.elem u2 sp.p1
      _   = F.act b.elem (C.comp v2 sp.p2) := by rw [sp.comm]
      _   = F.act (F.act b.elem v2) sp.p2 := (F.act_comp b.elem v2 sp.p2).symm
      _   = F.act (F.act c.elem v3) sp.p2 := by rw [e23]
      _   = F.act c.elem (C.comp v3 sp.p2) := F.act_comp c.elem v3 sp.p2

/-- Unit `eta_F(x) = [id_W, x]`. -/
def extEta {C : SmallCat} {F : PSh C} {W : C.Obj} (x : F.val W) : ExtRep C F W :=
  ⟨W, C.id W, x⟩

/-- Postcompose the projection of an extension with an ambient arrow. -/
def ExtRep.over {C : SmallCat} {F : PSh C} {X W : C.Obj}
    (s : C.Hom X W) (m : ExtRep C F X) : ExtRep C F W :=
  ⟨m.stage, C.comp s m.proj, m.elem⟩

theorem extEq_over {C : SmallCat} {F : PSh C} {X W : C.Obj}
    (s : C.Hom X W) {a b : ExtRep C F X} (h : ExtEq a b) : ExtEq (a.over s) (b.over s) := by
  obtain ⟨T, u, v, hs, hx⟩ := h
  exact ⟨T, u, v, (C.comp_assoc s a.proj u).symm.trans
    ((congrArg (C.comp s) hs).trans (C.comp_assoc s b.proj v)), hx⟩

/-- Kleisli bind `[f, x] >>= k` (Corollary 3.20). -/
def extBind {C : SmallCat} {F G : PSh C} {W : C.Obj}
    (m : ExtRep C F W) (k : ∀ {W' : C.Obj}, F.val W' → ExtRep C G W') : ExtRep C G W :=
  (k m.elem).over m.proj

/-- A continuation compatible with restriction up to common refinement
    induces a bind on equivalence classes of inputs. -/
theorem extBind_respects_extEq {C : SmallCat} (I : IndepPullbackStruct C)
    {F G : PSh C} {W : C.Obj}
    (k : ∀ {X : C.Obj}, F.val X → ExtRep C G X)
    (natural : ∀ {X Y : C.Obj} (x : F.val X) (u : C.Hom Y X),
      ExtEq ((k (F.act x u)).over u) (k x))
    {a b : ExtRep C F W} (h : ExtEq a b) : ExtEq (extBind a k) (extBind b k) := by
  obtain ⟨T, u, v, hs, hx⟩ := h
  have left := extEq_over a.proj (natural a.elem u)
  have right := extEq_over b.proj (natural b.elem v)
  have common : ((k (F.act a.elem u)).over u).over a.proj =
      ((k (F.act b.elem v)).over v).over b.proj := by
    dsimp only [ExtRep.over]
    rw [C.comp_assoc, hs, hx, C.comp_assoc]
  rw [common] at left
  exact extEq_trans I (extEq_symm left) right

/-- Functorial action `G(alpha)([f, x]) = [f, alpha(x)]`. -/
def extMap {C : SmallCat} {F G : PSh C} {W : C.Obj}
    (alpha : ∀ {W' : C.Obj}, F.val W' → G.val W') (m : ExtRep C F W) : ExtRep C G W :=
  ⟨m.stage, m.proj, alpha m.elem⟩

/-- A natural map sends any common refinement to a common refinement. -/
theorem extMap_respects_extEq {C : SmallCat} {F G : PSh C} {W : C.Obj}
    (α : ∀ {X : C.Obj}, F.val X → G.val X)
    (natural : ∀ {X Y : C.Obj} (x : F.val X) (u : C.Hom Y X),
      G.act (α x) u = α (F.act x u))
    {a b : ExtRep C F W} (h : ExtEq a b) : ExtEq (extMap α a) (extMap α b) := by
  obtain ⟨X, u, v, hc, he⟩ := h
  exact ⟨X, u, v, hc, (natural _ u).trans ((congrArg α he).trans (natural _ v).symm)⟩

/-- Left strength `st(x, [g, y]) = [g, (x . g, y)]`. -/
def extSt {C : SmallCat} {F G : PSh C} {W : C.Obj}
    (x : F.val W) (m : ExtRep C G W) : ExtRep C (prodPSh F G) W :=
  ⟨m.stage, m.proj, (F.act x m.proj, m.elem)⟩

/-- Theorem 1 (Left Unit): `eta(x) >>= k ~ k(x)`. -/
theorem ext_left_unit {C : SmallCat} {F G : PSh C} {W : C.Obj}
    (x : F.val W) (k : ∀ {W' : C.Obj}, F.val W' → ExtRep C G W') :
    ExtEq (extBind (extEta x) k) (k x) := by
  refine ⟨(k x).stage, C.id (k x).stage, C.id (k x).stage, ?_, ?_⟩
  · exact congrArg (fun p => C.comp p (C.id (k x).stage)) (C.id_comp (k x).proj)
  · rfl

/-- Theorem 1 (Right Unit): `m >>= eta ~ m`. -/
theorem ext_right_unit {C : SmallCat} {F : PSh C} {W : C.Obj}
    (m : ExtRep C F W) :
    ExtEq (extBind m (fun y => extEta y)) m := by
  have h : extBind m (fun y => extEta y) = m := by
    cases m with
    | mk st pr el =>
      dsimp only [extBind, ExtRep.over, extEta]
      rw [C.comp_id]
  rw [h]
  exact extEq_refl m

/-- Theorem 1 (Associativity): `(m >>= f) >>= g ~ m >>= (x -> f x >>= g)`. -/
theorem ext_assoc {C : SmallCat} {F G H : PSh C} {W : C.Obj}
    (m : ExtRep C F W)
    (f : ∀ {W' : C.Obj}, F.val W' → ExtRep C G W')
    (g : ∀ {W' : C.Obj}, G.val W' → ExtRep C H W') :
    ExtEq (extBind (extBind m f) g) (extBind m (fun x => extBind (f x) g)) := by
  have h : extBind (extBind m f) g = extBind m (fun x => extBind (f x) g) := by
    dsimp only [extBind, ExtRep.over]
    rw [C.comp_assoc]
  rw [h]
  exact extEq_refl _

/-- Theorem 1 (Affineness): `G(1) ~= 1` (unused sample-space extensions vanish). -/
theorem ext_affine {C : SmallCat} (I : IndepPullbackStruct C) {W : C.Obj}
    (a b : ExtRep C (constPSh C Unit) W) :
    ExtEq a b := by
  let sp := I.pb a.proj b.proj
  exact ⟨sp.apex, sp.p1, sp.p2, sp.comm, rfl⟩

/-- Pairing two extensions via the chosen independent span gives equivalent
    ordered observations when the pullback is formed in the opposite order. -/
theorem ext_commutative {C : SmallCat} (I : IndepPullbackStruct C)
    {F G : PSh C} {W : C.Obj}
    (m1 : ExtRep C F W) (m2 : ExtRep C G W) :
    let sp1 := I.pb m1.proj m2.proj
    let sp2 := I.pb m2.proj m1.proj
    let lhs : ExtRep C (prodPSh F G) W :=
      ⟨sp1.apex, C.comp m2.proj sp1.p2, (F.act m1.elem sp1.p1, G.act m2.elem sp1.p2)⟩
    let rhs : ExtRep C (prodPSh F G) W :=
      ⟨sp2.apex, C.comp m1.proj sp2.p2, (F.act m1.elem sp2.p2, G.act m2.elem sp2.p1)⟩
    ExtEq lhs rhs := by
  intro sp1 sp2 lhs rhs
  have hsym : I.Indep sp1.p2 sp1.p1 m2.proj m1.proj :=
    I.ip2 (I.pb_indep m1.proj m2.proj)
  rcases I.pb_univ m2.proj m1.proj sp1.p2 sp1.p1 hsym with ⟨u, ⟨hu1, hu2⟩, _⟩
  refine ⟨sp1.apex, C.id sp1.apex, u, ?_, ?_⟩
  · calc
      C.comp lhs.proj (C.id sp1.apex)
          = C.comp m2.proj sp1.p2 := C.comp_id _
      _   = C.comp m1.proj sp1.p1 := sp1.comm.symm
      _   = C.comp m1.proj (C.comp sp2.p2 u) := by rw [hu2]
      _   = C.comp (C.comp m1.proj sp2.p2) u := C.comp_assoc m1.proj sp2.p2 u
  · simp only [lhs, rhs, prodPSh]
    rw [F.act_id, G.act_id, F.act_comp, G.act_comp, hu2, hu1]

/-! ## 3. Boolean observations of extensions -/

/-- Extract the Boolean section of a constant presheaf. -/
def theta {C : SmallCat} {W : C.Obj} (m : ExtRep C (constPSh C Bool) W) : Bool :=
  m.elem

/-- `Theta` is well-defined on `ExtEq` equivalence classes (Proposition 4.1). -/
theorem theta_well_defined {C : SmallCat} {W : C.Obj}
    {a b : ExtRep C (constPSh C Bool) W} (h : ExtEq a b) :
    theta a = theta b := by
  rcases h with ⟨_, _, _, _, helem⟩
  exact helem

/-- `Theta ∘ eta = id` (first Eilenberg-Moore algebra axiom). -/
theorem theta_algebra_unit {C : SmallCat} {W : C.Obj} (b : Bool) :
    theta (extEta (C := C) (F := constPSh C Bool) (W := W) b) = b :=
  rfl

/-- Observing a bind into the constant Boolean presheaf observes its continuation. -/
theorem theta_algebra_bind {C : SmallCat} {F : PSh C} {W : C.Obj}
    (m : ExtRep C F W)
    (k : ∀ {W' : C.Obj}, F.val W' → ExtRep C (constPSh C Bool) W') :
    theta (extBind m k) = theta (k m.elem) :=
  rfl

/-- `Theta` and `eta` are mutual inverses up to `ExtEq` (Proposition 4.1 iso). -/
theorem theta_right_inv {C : SmallCat} (I : IndepPullbackStruct C) {W : C.Obj}
    (m : ExtRep C (constPSh C Bool) W) :
    ExtEq (extEta (C := C) (F := constPSh C Bool) (W := W) (theta m)) m := by
  let sp := I.pb (C.id W) m.proj
  exact ⟨sp.apex, sp.p1, sp.p2, sp.comm, rfl⟩

/-- Two Boolean extensions agree up to common refinement exactly when their
    observations agree. The unit supplies a representative of each observation. -/
theorem theta_eq_iff_extEq {C : SmallCat} (I : IndepPullbackStruct C) {W : C.Obj}
    (a b : ExtRep C (constPSh C Bool) W) : theta a = theta b ↔ ExtEq a b := by
  constructor
  · intro h
    have ha := theta_right_inv I a
    rw [h] at ha
    exact extEq_trans I (extEq_symm ha) (theta_right_inv I b)
  · exact theta_well_defined

/-! ## 4. Hoare rules on extension representatives -/

/-- A stage-indexed predicate stable under further restriction. -/
structure IntPred (C : SmallCat) (A : PSh C) (W : C.Obj) where
  eval : ∀ {W' : C.Obj}, C.Hom W' W → A.val W' → Prop
  mono : ∀ {W1 W2 : C.Obj} (s : C.Hom W1 W) (u : C.Hom W2 W1) (a : A.val W1),
    eval s a → eval (C.comp s u) (A.act a u)

/-- Weakest precondition `wp t phi` (Definition 4.2 & Corollary D.3):
    for `t = [s : W' -> W, x]`, `wp t phi` holds iff `phi(s, x)` holds. -/
def wp {C : SmallCat} {A : PSh C} {W : C.Obj}
    (t : ExtRep C A W) (phi : IntPred C A W) : Prop :=
  phi.eval t.proj t.elem

/-- Internal Hoare triple `{pre} t {post} := pre -> wp t post` (Definition 4.3). -/
def Hoare {C : SmallCat} {A : PSh C} {W : C.Obj}
    (pre : Prop) (t : ExtRep C A W) (post : IntPred C A W) : Prop :=
  pre → wp t post

/-- A predicate which also reflects truth along restrictions gives a
    representative-independent weakest precondition. -/
theorem wp_respects_extEq {C : SmallCat} {A : PSh C} {W : C.Obj}
    (p : IntPred C A W)
    (reflects : ∀ {X Y : C.Obj} (s : C.Hom X W) (u : C.Hom Y X) (x : A.val X),
      p.eval (C.comp s u) (A.act x u) → p.eval s x)
    {a b : ExtRep C A W} (h : ExtEq a b) : wp a p ↔ wp b p := by
  obtain ⟨X, u, v, hs, hx⟩ := h
  constructor
  · intro ha
    apply reflects b.proj v b.elem
    rw [← hs, ← hx]
    exact p.mono a.proj u a.elem ha
  · intro hb
    apply reflects a.proj u a.elem
    rw [hs, hx]
    exact p.mono b.proj v b.elem hb

/-- Pullback of an internal predicate along `s : W' -> W`. -/
def IntPred.pull {C : SmallCat} {A : PSh C} {W W' : C.Obj}
    (phi : IntPred C A W) (s : C.Hom W' W) : IntPred C A W' where
  eval := fun u a => phi.eval (C.comp s u) a
  mono := fun u v a h => by
    dsimp only
    rw [C.comp_assoc]
    exact phi.mono (C.comp s u) v a h

/-- Constant predicate `fun _ => theta` lifted into `IntPred`. -/
def IntPred.const {C : SmallCat} (A : PSh C) (W : C.Obj) (p : Prop) : IntPred C A W where
  eval := fun _ _ => p
  mono := fun _ _ _ h => h

/-- Pointwise conjunction of internal predicates. -/
def IntPred.conj {C : SmallCat} {A : PSh C} {W : C.Obj}
    (p q : IntPred C A W) : IntPred C A W where
  eval := fun s a => p.eval s a ∧ q.eval s a
  mono := fun s u a h => ⟨p.mono s u a h.1, q.mono s u a h.2⟩

/-- Theorem 3 (HRet, Figure 5): `{phi(id, x)} return x {y. phi(y)}`. -/
theorem hoare_HRet {C : SmallCat} {A : PSh C} {W : C.Obj}
    (phi : IntPred C A W) (x : A.val W) :
    Hoare (phi.eval (C.id W) x) (extEta x) phi := by
  intro h
  exact h

/-- Theorem 3 (HComp, Figure 5): higher-order sequencing rule. -/
theorem hoare_HComp {C : SmallCat} {A B : PSh C} {W : C.Obj}
    (pre : Prop) (mid : IntPred C A W) (post : IntPred C B W)
    (t : ExtRep C A W)
    (k : ∀ {W' : C.Obj}, A.val W' → ExtRep C B W')
    (h1 : Hoare pre t mid)
    (h2 : ∀ {W' : C.Obj} (s : C.Hom W' W) (a : A.val W'),
      Hoare (mid.eval s a) (k a) (post.pull s)) :
    Hoare pre (extBind t k) post := by
  intro hpre
  have hmid : mid.eval t.proj t.elem := h1 hpre
  have hstep := h2 t.proj t.elem hmid
  exact hstep

/-- Theorem 3 (HFunc, Figure 5): higher-order function instantiation rule. -/
theorem hoare_HFunc {C : SmallCat} {A B : PSh C} {W : C.Obj}
    (pre : A.val W → Prop)
    (f : A.val W → ExtRep C B W)
    (x : A.val W)
    (post : A.val W → IntPred C B W) :
    Hoare ((∀ a : A.val W, Hoare (pre a) (f a) (post a)) ∧ pre x)
          (f x)
          (post x) := by
  intro h
  exact h.1 x h.2

/-- Theorem 3 (HCons, Figure 5): rule of consequence. -/
theorem hoare_HCons {C : SmallCat} {A : PSh C} {W : C.Obj}
    (pre pre' : Prop) (post post' : IntPred C A W) (t : ExtRep C A W)
    (hpre : pre → pre')
    (hpost : ∀ {W' : C.Obj} (s : C.Hom W' W) (a : A.val W'),
      post'.eval s a → post.eval s a)
    (ht : Hoare pre' t post') :
    Hoare pre t post := by
  intro hp
  exact hpost t.proj t.elem (ht (hpre hp))

/-- Framing a stage-independent proposition. -/
theorem hoare_HFrame {C : SmallCat} {A : PSh C} {W : C.Obj}
    (pre thetaProp : Prop) (post : IntPred C A W) (t : ExtRep C A W)
    (ht : Hoare pre t post) :
    Hoare (pre ∧ thetaProp) t (IntPred.conj post (IntPred.const A W thetaProp)) := by
  intro h
  exact ⟨ht h.1, h.2⟩

/-- A frame about an ambient section persists when the computation extends
    the stage and pairs its output with the restricted section. -/
theorem hoare_frame_strength {C : SmallCat} {A B : PSh C} {W : C.Obj}
    (t : ExtRep C B W) (x : A.val W) (frame : IntPred C A W)
    (post : IntPred C B W) (pre : Prop) (ht : Hoare pre t post) :
    Hoare (frame.eval (C.id W) x ∧ pre) (extSt x t)
      { eval := fun s p => frame.eval s p.1 ∧ post.eval s p.2
        mono := fun s u p h => ⟨frame.mono s u p.1 h.1, post.mono s u p.2 h.2⟩ } := by
  intro h
  have keep := frame.mono (C.id W) t.proj x h.1
  rw [C.id_comp] at keep
  exact ⟨keep, ht h.2⟩

/-- Theorem 3 (HConst, Lemma D.6): any proposition true before `t` remains true
    after `t`. -/
theorem hoare_HConst {C : SmallCat} {A : PSh C} {W : C.Obj}
    (thetaProp : Prop) (t : ExtRep C A W) :
    Hoare thetaProp t (IntPred.const A W thetaProp) := by
  intro h
  exact h

/-- Conditional composition for a decided Boolean guard. -/
theorem hoare_HCond {C : SmallCat} {A : PSh C} {W : C.Obj}
    (pre : Prop) (cond : Bool) (t1 t2 : ExtRep C A W) (post : IntPred C A W)
    (h1 : cond = true → Hoare pre t1 post)
    (h2 : cond = false → Hoare pre t2 post) :
    Hoare pre (if cond then t1 else t2) post := by
  cases cond with
  | true => exact h1 rfl
  | false => exact h2 rfl

/-! ## 5. Concrete Finite Probability Controls & Paper Case Studies -/

/-- Incrementing a mutable cell can invalidate a frame about its old value. -/
theorem imperative_frame_fails :
    (∀ x : Nat, x = 10 → x + 1 = 11) ∧
    ¬ (∀ x : Nat, (x = 10 ∧ x ≤ 10) → (x + 1 = 11 ∧ x + 1 ≤ 10)) := by
  constructor
  · intro x hx; rw [hx]
  · intro h
    have h10 := h 10 ⟨rfl, Nat.le_refl 10⟩
    omega

/-- All 8 equiprobable triples in `{false, true}^3`. -/
def allBoolTriples : List (Bool × Bool × Bool) :=
  [ (false, false, false), (false, false, true),
    (false, true,  false), (false, true,  true),
    (true,  false, false), (true,  false, true),
    (true,  true,  false), (true,  true,  true) ]

/-- All 4 equiprobable pairs in `{false, true}^2`. -/
def allBoolPairs : List (Bool × Bool) :=
  [(false, false), (false, true), (true, false), (true, true)]

/-- Example 4.9 (`mask` higher-order XOR):
    `x <- coin; c1 <- coin; c2 <- coin; return (x, x xor c1, x xor c2)`.
    Every output triple `(a, b, c) ∈ Bool^3` occurs in exactly 1 of 8 equiprobable
    outcomes (`P(a,b,c) = 1/8 = P(a)P(b)P(c)`), proving mutual independence
    `_|_m(x, y, z)`. -/
def maskOutput (w : Bool × Bool × Bool) : Bool × Bool × Bool :=
  (w.1, xor w.1 w.2.1, xor w.1 w.2.2)

theorem mask_triple_mutually_independent :
    allBoolTriples.length = 8 ∧
    allBoolTriples.all (fun target =>
      (allBoolTriples.filter (fun w => maskOutput w == target)).length == 1) = true := by
  decide

/-- Adversarial counterexample (pairwise independence does NOT imply mutual
    independence): `X, Y ~ Ber(1/2)` i.i.d., `Z = xor X Y`.
    All 3 pairs `(X,Y)`, `(X,Z)`, `(Y,Z)` are uniformly distributed (`1/4` each),
    yet `(X, Y, Z)` hits `(false, false, true)` with probability `0 != 1/8`. -/
def xorTriple (p : Bool × Bool) : Bool × Bool × Bool :=
  (p.1, p.2, xor p.1 p.2)

theorem xor_triple_pairwise_indep_not_mutually_indep :
    allBoolPairs.all (fun target =>
      (allBoolPairs.filter (fun p => (p.1, p.2) == target)).length == 1) = true ∧
    allBoolPairs.all (fun target =>
      (allBoolPairs.filter (fun p => (p.1, xor p.1 p.2) == target)).length == 1) = true ∧
    allBoolPairs.all (fun target =>
      (allBoolPairs.filter (fun p => (p.2, xor p.1 p.2) == target)).length == 1) = true ∧
    (allBoolPairs.filter (fun p => xorTriple p == (false, false, true))).length = 0 := by
  decide

/-- Two independent fair bits collide in two of their four outcomes. -/
theorem discrete_coin_not_nu_fresh :
    (allBoolPairs.filter (fun p => p.1 == p.2)).length = 2 ∧
    (allBoolPairs.filter (fun p => p.1 == p.2)).length ≠ 0 := by
  decide

/-- Both equiprobable outcomes of one fair coin. -/
def allBools : List Bool := [false, true]

theorem correlated_shared_coin_not_independent :
    let n := allBools.length
    let p0 := (allBools.filter (fun b => b == false)).length
    let p00 := (allBools.filter (fun b => (b, b) == (false, false))).length
    p00 * n ≠ p0 * p0 := by
  decide

/-- A fair two-branch mixture with distinct return values. -/
def branchMerge (c : Bool) : Nat := if c then 10 else 20

theorem hcond_fair_mixture_exact_law :
    (allBools.filter (fun c => branchMerge c == 10)).length = 1 ∧
    (allBools.filter (fun c => branchMerge c == 20)).length = 1 ∧
    (allBools.filter (fun c => branchMerge c == 10)).length ≠ allBools.length ∧
    (allBools.filter (fun c => branchMerge c == 20)).length ≠ allBools.length := by
  decide

/-- A shared draw has one outcome at each diagonal pair and zero elsewhere.
    Two fresh draws have one outcome at every pair. -/
theorem sblc_shared_vs_fresh_sheaf_reconciliation (x y : Bool) :
    (allBools.filter (fun b => (b, b) == (x, y))).length = (if x = y then 1 else 0) ∧
    (allBoolPairs.filter (fun p => p == (x, y))).length = 1 := by
  cases x <;> cases y <;> decide

#print axioms chosen_isIndependentPullback
#print axioms pullback_endo_eq_id
#print axioms pullbacks_unique_up_to_unique_iso
#print axioms constPSh_isAtomicSheaf
#print axioms prodPSh_isAtomicSheaf
#print axioms extEq_refl
#print axioms extEq_symm
#print axioms extEq_trans
#print axioms extEq_over
#print axioms extBind_respects_extEq
#print axioms extMap_respects_extEq
#print axioms ext_left_unit
#print axioms ext_right_unit
#print axioms ext_assoc
#print axioms ext_affine
#print axioms ext_commutative
#print axioms theta_well_defined
#print axioms theta_algebra_unit
#print axioms theta_algebra_bind
#print axioms theta_right_inv
#print axioms theta_eq_iff_extEq
#print axioms wp_respects_extEq
#print axioms hoare_HRet
#print axioms hoare_HComp
#print axioms hoare_HFunc
#print axioms hoare_HCons
#print axioms hoare_HFrame
#print axioms hoare_frame_strength
#print axioms hoare_HConst
#print axioms hoare_HCond
#print axioms imperative_frame_fails
#print axioms mask_triple_mutually_independent
#print axioms xor_triple_pairwise_indep_not_mutually_indep
#print axioms discrete_coin_not_nu_fresh
#print axioms correlated_shared_coin_not_independent
#print axioms hcond_fair_mixture_exact_law
#print axioms sblc_shared_vs_fresh_sheaf_reconciliation

end EshkolProbabilitySheafHoare
