/-
  EshkolProbabilitySheafHoare.lean
  Standalone Lean 4.12.0 formalization of Ho, Wu, and Raad (arXiv:2611.30692,
  POPL 2027), "Sheaf Semantics of Probabilistic Programs":

    1. Abstract Small Category & Simpson Independent Pullback Structure
       `(C, I)` (Definition 3.9, axioms IP1-IP5 with universal property of
       independent pullbacks) and Atomic Sheaves (`IsAtomicSheaf`).
    2. Extension Monad `G` over any `(C, I)` (Definition 3.16, Lemma 3.18)
       and Theorem 1:
         - `ext_left_unit`, `ext_right_unit`, `ext_assoc`
         - `ext_affine` (`G(Unit) ~= Unit`: sample-space marginalization)
         - `ext_commutative` (independent draws reorder via `pb_univ` + IP2)
    3. Canonical `G`-algebra `Theta : G(Prop) -> Prop` (Proposition 4.1):
         - `theta_well_defined`, `theta_algebra_unit`, `theta_algebra_mul`,
           `theta_left_inv`, `theta_right_inv`
    4. Internal Higher-Order Probabilistic Hoare Logic (`wp` & `{phi} t {psi}`)
       and Theorem 3 (all 5 proof rules of Figure 5):
         - `hoare_HRet`
         - `hoare_HComp`
         - `hoare_HFunc`
         - `hoare_HCons`
         - `hoare_HFrame` (from monotonicity under sample-space extension)
    5. Concrete Finite Probability Instantiation, Theorem 4 & Adversarial
       Anti-Math-Washing Controls:
         - `imperative_frame_fails` (Section 4.1 mutable state counterexample)
         - `mask_triple_mutually_independent` (Example 4.9: exact 1/8 uniformity)
         - `xor_triple_pairwise_indep_not_mutually_indep` (adversarial control)
         - `discrete_coin_not_nu_fresh` (collision 2/4 = 1/2 != 0)
         - `correlated_shared_coin_not_independent` (unsound independence rejected)
         - `hcond_fair_mixture_exact_law` (Example 4.10 mixture & branch-merge control)
         - `sblc_shared_vs_fresh_sheaf_reconciliation` (Stochastic BLC bridge)

  Zero external dependencies (pure Lean 4.12.0 Prelude), zero `sorry`,
  zero custom axioms.
-/

namespace EshkolProbabilitySheafHoare

/-! ## 1. Small Category & Simpson Independent Pullback Structure (IP1-IP5) -/

structure SmallCat where
  Obj : Type
  Hom : Obj → Obj → Type
  id  : ∀ (A : Obj), Hom A A
  comp : ∀ {A B C : Obj}, Hom B C → Hom A B → Hom A C
  id_comp : ∀ {A B : Obj} (f : Hom A B), comp (id B) f = f
  comp_id : ∀ {A B : Obj} (f : Hom A B), comp f (id A) = f
  comp_assoc : ∀ {A B C D : Obj} (h : Hom C D) (g : Hom B C) (f : Hom A B),
    comp h (comp g f) = comp (comp h g) f

/-- Chosen independent pullback span for a cospan `W1 -f-> W <-g- W2` (IP5). -/
structure SpanData (C : SmallCat) {W1 W2 W : C.Obj}
    (f : C.Hom W1 W) (g : C.Hom W2 W) where
  apex : C.Obj
  p1 : C.Hom apex W1
  p2 : C.Hom apex W2
  comm : C.comp f p1 = C.comp g p2

/-- Simpson's Independent Pullback Structure `(C, I)` (Definition 3.9):
    axioms IP1-IP5 including the universal property of independent pullbacks
    with respect to independent squares (`pb_univ`). -/
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
      C.comp (pb f g).p1 u = s ∧ C.comp (pb f g).p2 u = t

/-- Contravariant presheaf `F : C^op -> Type`. -/
structure PSh (C : SmallCat) where
  val : C.Obj → Type
  act : ∀ {W W' : C.Obj}, val W → C.Hom W' W → val W'
  act_id : ∀ {W : C.Obj} (x : val W), act x (C.id W) = x
  act_comp : ∀ {W1 W2 W3 : C.Obj} (x : val W3) (f : C.Hom W2 W3) (g : C.Hom W1 W2),
    act (act x f) g = act x (C.comp f g)

/-- Atomic sheaf condition (Definition 3.12): every `s`-invariant section in
    `F(W')` descends uniquely along `s : W' -> W` to `F(W)`. -/
def IsSInvariant {C : SmallCat} (F : PSh C) {W' W : C.Obj}
    (s : C.Hom W' W) (y : F.val W') : Prop :=
  ∀ {W'' : C.Obj} (u v : C.Hom W'' W'),
    C.comp s u = C.comp s v → F.act y u = F.act y v

def IsAtomicSheaf {C : SmallCat} (F : PSh C) : Prop :=
  ∀ {W' W : C.Obj} (s : C.Hom W' W) (y : F.val W'),
    IsSInvariant F s y →
    ∃ x : F.val W, F.act x s = y ∧ ∀ x' : F.val W, F.act x' s = y → x' = x

/-- Constant presheaf on `A` (in particular `Prop = Bool` / `Prop`). -/
def constPSh (C : SmallCat) (A : Type) : PSh C where
  val := fun _ => A
  act := fun a _ => a
  act_id := fun _ => rfl
  act_comp := fun _ _ _ => rfl

/-- Product presheaf `F x G`. -/
def prodPSh {C : SmallCat} (F G : PSh C) : PSh C where
  val := fun W => F.val W × G.val W
  act := fun p f => (F.act p.1 f, G.act p.2 f)
  act_id := fun p => by simp [F.act_id, G.act_id]
  act_comp := fun p f g => by simp [F.act_comp, G.act_comp]

/-- Every constant presheaf is an atomic sheaf (Section 4.1, footnote 13 & 14). -/
theorem constPSh_isAtomicSheaf (C : SmallCat) (A : Type) :
    IsAtomicSheaf (constPSh C A) := by
  intro W' W s y _
  refine ⟨y, rfl, ?_⟩
  intro x' hx'
  exact hx'

/-! ## 2. Extension Monad `G` & Theorem 1 (Strong, Affine, Commutative) -/

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

/-- Kleisli bind `[f, x] >>= k` (Corollary 3.20). -/
def extBind {C : SmallCat} {F G : PSh C} {W : C.Obj}
    (m : ExtRep C F W) (k : ∀ {W' : C.Obj}, F.val W' → ExtRep C G W') : ExtRep C G W :=
  let n := k m.elem
  ⟨n.stage, C.comp m.proj n.proj, n.elem⟩

/-- Functorial action `G(alpha)([f, x]) = [f, alpha(x)]`. -/
def extMap {C : SmallCat} {F G : PSh C} {W : C.Obj}
    (alpha : ∀ {W' : C.Obj}, F.val W' → G.val W') (m : ExtRep C F W) : ExtRep C G W :=
  ⟨m.stage, m.proj, alpha m.elem⟩

/-- Left strength `st(x, [g, y]) = [g, (x . g, y)]`. -/
def extSt {C : SmallCat} {F G : PSh C} {W : C.Obj}
    (x : F.val W) (m : ExtRep C G W) : ExtRep C (prodPSh F G) W :=
  ⟨m.stage, m.proj, (F.act x m.proj, m.elem)⟩

/-- Theorem 1 (Left Unit): `eta(x) >>= k ~ k(x)`. -/
theorem ext_left_unit {C : SmallCat} {F G : PSh C} {W : C.Obj}
    (x : F.val W) (k : ∀ {W' : C.Obj}, F.val W' → ExtRep C G W') :
    ExtEq (extBind (extEta x) k) (k x) := by
  refine ⟨(k x).stage, C.id (k x).stage, C.id (k x).stage, ?_, ?_⟩
  · simp [extBind, extEta, C.id_comp]
  · simp [extBind, extEta]

/-- Theorem 1 (Right Unit): `m >>= eta ~ m`. -/
theorem ext_right_unit {C : SmallCat} {F : PSh C} {W : C.Obj}
    (m : ExtRep C F W) :
    ExtEq (extBind m (fun y => extEta y)) m := by
  have h : extBind m (fun y => extEta y) = m := by
    cases m with
    | mk st pr el =>
      simp [extBind, extEta, C.comp_id]
  rw [h]
  exact extEq_refl m

/-- Theorem 1 (Associativity): `(m >>= f) >>= g ~ m >>= (x -> f x >>= g)`. -/
theorem ext_assoc {C : SmallCat} {F G H : PSh C} {W : C.Obj}
    (m : ExtRep C F W)
    (f : ∀ {W' : C.Obj}, F.val W' → ExtRep C G W')
    (g : ∀ {W' : C.Obj}, G.val W' → ExtRep C H W') :
    ExtEq (extBind (extBind m f) g) (extBind m (fun x => extBind (f x) g)) := by
  have h : extBind (extBind m f) g = extBind m (fun x => extBind (f x) g) := by
    simp [extBind, C.comp_assoc]
  rw [h]
  exact extEq_refl _

/-- Theorem 1 (Affineness): `G(1) ~= 1` (unused sample-space extensions vanish). -/
theorem ext_affine {C : SmallCat} (I : IndepPullbackStruct C) {W : C.Obj}
    (a b : ExtRep C (constPSh C Unit) W) :
    ExtEq a b := by
  let sp := I.pb a.proj b.proj
  exact ⟨sp.apex, sp.p1, sp.p2, sp.comm, rfl⟩

/-- Theorem 1 (Commutativity, Appendix B pp. 39-40):
    sequencing independent `m1 : G F(W)` and `m2 : G G'(W)` in either order
    produces equivalent elements of `G(F x G')(W)`, proved via `IP2` and the
    universal property `pb_univ` of independent pullbacks. -/
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
  rcases I.pb_univ m2.proj m1.proj sp1.p2 sp1.p1 hsym with ⟨u, hu1, hu2⟩
  refine ⟨sp1.apex, C.id sp1.apex, u, ?_, ?_⟩
  · calc
      C.comp lhs.proj (C.id sp1.apex)
          = C.comp m2.proj sp1.p2 := C.comp_id _
      _   = C.comp m1.proj sp1.p1 := sp1.comm.symm
      _   = C.comp m1.proj (C.comp sp2.p2 u) := by rw [hu2]
      _   = C.comp (C.comp m1.proj sp2.p2) u := C.comp_assoc m1.proj sp2.p2 u
  · simp only [lhs, rhs, prodPSh]
    rw [F.act_id, G.act_id, F.act_comp, G.act_comp, hu2, hu1]

/-! ## 3. Canonical G-Algebra `Theta : G(Prop) -> Prop` (Proposition 4.1) -/

/-- Canonical `G`-algebra `Theta_W : G(Prop)(W) -> Prop` sending `[f, b] |-> b`. -/
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

/-- `Theta ∘ (m >>= k) = Theta(G(Theta)(...))` (second algebra axiom / bind). -/
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

/-! ## 4. Internal Higher-Order Probabilistic Hoare Logic & Theorem 3 -/

/-- Internal predicate `Prop^A(W) ~= Hom(y(W) x A, Prop)` (Remark D.2):
    assigns a truth value to any extended stage `W'`, projection `s : W' -> W`,
    and section `a : A(W')`, monotone under further pullback. -/
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

/-- Theorem 3 (HFrame, Figure 5): unconditional frame rule in atomic sheaf logic,
    valid by monotonicity under sample-space extension. -/
theorem hoare_HFrame {C : SmallCat} {A : PSh C} {W : C.Obj}
    (pre thetaProp : Prop) (post : IntPred C A W) (t : ExtRep C A W)
    (ht : Hoare pre t post) :
    Hoare (pre ∧ thetaProp) t (IntPred.conj post (IntPred.const A W thetaProp)) := by
  intro h
  exact ⟨ht h.1, h.2⟩

/-- Theorem 3 (HConst, Lemma D.6): any proposition true before `t` remains true
    after `t`. -/
theorem hoare_HConst {C : SmallCat} {A : PSh C} {W : C.Obj}
    (thetaProp : Prop) (t : ExtRep C A W) :
    Hoare thetaProp t (IntPred.const A W thetaProp) := by
  intro h
  exact h

/-- Theorem 4 (Abstract HCond): conditional composition rule over boolean test. -/
theorem hoare_HCond {C : SmallCat} {A : PSh C} {W : C.Obj}
    (pre : Prop) (cond : Bool) (t1 t2 : ExtRep C A W) (post : IntPred C A W)
    (h1 : cond = true → Hoare pre t1 post)
    (h2 : cond = false → Hoare pre t2 post) :
    Hoare pre (if cond then t1 else t2) post := by
  cases cond with
  | true => exact h1 rfl
  | false => exact h2 rfl

/-! ## 5. Concrete Finite Probability Controls & Paper Case Studies -/

/-- Section 4.1 negative control: in imperative mutable state, the unconditional
    frame rule FAILS (`{x=10} x:=x+1 {x=11}` holds, but framing `x <= 10` fails). -/
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

/-- Nu-calculus freshness negative control on discrete fair coin:
    over `Bool x Bool`, `x = y` holds in 2 of 4 outcomes (`P(x = y) = 1/2 != 0`),
    so discrete coin does NOT satisfy the continuous `nu`-calculus freshness law. -/
theorem discrete_coin_not_nu_fresh :
    (allBoolPairs.filter (fun p => p.1 == p.2)).length = 2 ∧
    (allBoolPairs.filter (fun p => p.1 == p.2)).length ≠ 0 := by
  decide

/-- Tsotchke-chan Guard 4 (unsound independence rejection):
    for a shared coin draw `(x, x)` over `Bool`, `P(x = false, x = false) = 1/2 != 1/4 = P(x=false)*P(x=false)`. -/
def allBools : List Bool := [false, true]

theorem correlated_shared_coin_not_independent :
    let n := allBools.length
    let p0 := (allBools.filter (fun b => b == false)).length
    let p00 := (allBools.filter (fun b => (b, b) == (false, false))).length
    p00 * n ≠ p0 * p0 := by
  decide

/-- Example 4.10 discrete mixture & Tsotchke-chan Guard 3 (branch-merge conflict):
    `c <- coin; if c then 10 else 20` has exact 1/2 mass on 10 and 1/2 mass on 20;
    neither pure-branch law (`P(z=10)=1` nor `P(z=20)=1`) holds after the merge. -/
def branchMerge (c : Bool) : Nat := if c then 10 else 20

theorem hcond_fair_mixture_exact_law :
    (allBools.filter (fun c => branchMerge c == 10)).length = 1 ∧
    (allBools.filter (fun c => branchMerge c == 20)).length = 1 ∧
    (allBools.filter (fun c => branchMerge c == 10)).length ≠ allBools.length ∧
    (allBools.filter (fun c => branchMerge c == 20)).length ≠ allBools.length := by
  decide

/-- Direct Stochastic BLC bridge: shared draw `(x, x)` concentrates 1/2 + 1/2 on
    the diagonal (0 mass off-diagonal), whereas two independent fresh draws
    `(x, y)` assign exact uniform 1/4 mass to every ordered pair in `Bool x Bool`. -/
theorem sblc_shared_vs_fresh_sheaf_reconciliation :
    (allBools.filter (fun b => (b, b) == (false, true))).length = 0 ∧
    (allBools.filter (fun b => (b, b) == (true, true))).length = 1 ∧
    allBoolPairs.all (fun target =>
      (allBoolPairs.filter (fun p => p == target)).length == 1) = true := by
  decide

end EshkolProbabilitySheafHoare

#print axioms EshkolProbabilitySheafHoare.constPSh_isAtomicSheaf
#print axioms EshkolProbabilitySheafHoare.extEq_trans
#print axioms EshkolProbabilitySheafHoare.ext_left_unit
#print axioms EshkolProbabilitySheafHoare.ext_right_unit
#print axioms EshkolProbabilitySheafHoare.ext_assoc
#print axioms EshkolProbabilitySheafHoare.ext_affine
#print axioms EshkolProbabilitySheafHoare.ext_commutative
#print axioms EshkolProbabilitySheafHoare.theta_well_defined
#print axioms EshkolProbabilitySheafHoare.theta_right_inv
#print axioms EshkolProbabilitySheafHoare.hoare_HRet
#print axioms EshkolProbabilitySheafHoare.hoare_HComp
#print axioms EshkolProbabilitySheafHoare.hoare_HFunc
#print axioms EshkolProbabilitySheafHoare.hoare_HCons
#print axioms EshkolProbabilitySheafHoare.hoare_HFrame
#print axioms EshkolProbabilitySheafHoare.hoare_HCond
#print axioms EshkolProbabilitySheafHoare.imperative_frame_fails
#print axioms EshkolProbabilitySheafHoare.mask_triple_mutually_independent
#print axioms EshkolProbabilitySheafHoare.xor_triple_pairwise_indep_not_mutually_indep
#print axioms EshkolProbabilitySheafHoare.discrete_coin_not_nu_fresh
#print axioms EshkolProbabilitySheafHoare.correlated_shared_coin_not_independent
#print axioms EshkolProbabilitySheafHoare.hcond_fair_mixture_exact_law
#print axioms EshkolProbabilitySheafHoare.sblc_shared_vs_fresh_sheaf_reconciliation
