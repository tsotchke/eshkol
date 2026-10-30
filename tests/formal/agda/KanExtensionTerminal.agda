{-# OPTIONS --cubical --safe --guardedness -WnoUnsupportedIndexedMatch #-}
------------------------------------------------------------------------
-- Kan extension along the functor to the terminal category.
--
-- nLab, "Kan extension" (claim 2 of tests/nlab/nlab_category_test.esk):
--   "If C' = * is the terminal category, then the left Kan extension
--    operation forms the colimit of a functor; the right Kan extension
--    operation forms the limit of a functor."
--
-- General theorem (checked here for every small category C and every
-- Set-valued functor F on C):
--   * colim F, built as the set quotient of  Σ c. F c  by the arrows of C,
--     represents cocones:   (colim F → Y) ≃ Nat(F, p* Y)   for every set Y,
--     naturally in Y.  This is the defining universal property of Lan_p F
--     for p : C → *.
--   * lim F, built as the set of compatible families, represents cones:
--     (Y → lim F) ≃ Nat(p* Y, F), naturally in Y; the defining universal
--     property of Ran_p F.
-- Corollary: for a group G acting on a set X (a functor BG → Set),
--   Lan_p X = orbits X/G  and  Ran_p X = fixed points X^G.
--
-- Concrete instance computed by Eshkol (core.category, set-lan / set-ran
-- along terminal-functor): the span {a b c} <- {x y} -> {p q} with
-- x, y |-> a and x |-> p, y |-> q.  Eshkol prints (|Lan_! G| |colim G|
-- |Ran_! G| |lim G|) = (3 3 2 2).  Below: colim ≃ Fin 3 and lim ≃ Fin 2,
-- with explicit inverse maps.
------------------------------------------------------------------------
module KanExtensionTerminal where

open import Cubical.Foundations.Prelude
open import Cubical.Foundations.Function
open import Cubical.Foundations.Isomorphism
open import Cubical.Foundations.Equiv
open import Cubical.Foundations.HLevels
open import Cubical.Data.Sigma
open import Cubical.Data.Unit
open import Cubical.Data.Empty as ⊥
open import Cubical.Data.FinData
open import Cubical.Data.FinData.Properties using (znots; snotz; isSetFin)
open import Cubical.Relation.Nullary
open import Cubical.HITs.SetQuotients as SQ

------------------------------------------------------------------------
-- Small categories and Set-valued functors

record Category : Type₁ where
  field
    Ob     : Type₀
    Hom    : Ob → Ob → Type₀
    isSetHom : ∀ a b → isSet (Hom a b)
    idn    : ∀ a → Hom a a
    _⋆_    : ∀ {a b c} → Hom a b → Hom b c → Hom a c     -- diagrammatic order
    ⋆IdL   : ∀ {a b} (f : Hom a b) → idn a ⋆ f ≡ f
    ⋆IdR   : ∀ {a b} (f : Hom a b) → f ⋆ idn b ≡ f
    ⋆Assoc : ∀ {a b c d} (f : Hom a b) (g : Hom b c) (h : Hom c d)
           → (f ⋆ g) ⋆ h ≡ f ⋆ (g ⋆ h)

record SetFunctor (C : Category) : Type₁ where
  open Category C
  field
    F₀      : Ob → Type₀
    isSetF₀ : ∀ c → isSet (F₀ c)
    F₁      : ∀ {a b} → Hom a b → F₀ a → F₀ b
    F-id    : ∀ a (x : F₀ a) → F₁ (idn a) x ≡ x
    F-seq   : ∀ {a b c} (f : Hom a b) (g : Hom b c) (x : F₀ a)
            → F₁ (f ⋆ g) x ≡ F₁ g (F₁ f x)

module _ {C : Category} (F : SetFunctor C) where
  open Category C
  open SetFunctor F

  ----------------------------------------------------------------------
  -- Natural transformations into and out of the constant functor p* Y

  Cocone : Type₀ → Type₀
  Cocone Y = Σ[ α ∈ (∀ c → F₀ c → Y) ]
             (∀ a b (f : Hom a b) (x : F₀ a) → α b (F₁ f x) ≡ α a x)

  Cone : Type₀ → Type₀
  Cone Y = Σ[ β ∈ (∀ c → Y → F₀ c) ]
           (∀ a b (f : Hom a b) (y : Y) → F₁ f (β a y) ≡ β b y)

  ----------------------------------------------------------------------
  -- colim F = (Σ c. F c) / (arrows of C)

  Elt : Type₀
  Elt = Σ Ob F₀

  _~_ : Elt → Elt → Type₀
  (a , x) ~ (b , y) = Σ[ f ∈ Hom a b ] F₁ f x ≡ y

  colim : Type₀
  colim = Elt / _~_

  isSetColim : isSet colim
  isSetColim = squash/

  inj : ∀ c → F₀ c → colim
  inj c x = [ c , x ]

  -- Lan_p F = colim F : the universal property
  respIso : (Y : Type₀) → isSet Y
          → Iso (Σ[ h ∈ (Elt → Y) ] (∀ u v → u ~ v → h u ≡ h v)) (Cocone Y)
  Iso.fun (respIso Y _) (h , r) =
    (λ c x → h (c , x)) , λ a b f x → sym (r (a , x) (b , F₁ f x) (f , refl))
  Iso.inv (respIso Y _) (α , nat) =
    (λ u → α (fst u) (snd u)) ,
    λ { (a , x) (b , y) (f , p) → sym (nat a b f x) ∙ cong (α b) p }
  Iso.rightInv (respIso Y isSetY) (α , nat) =
    Σ≡Prop (λ α' → isPropΠ4 λ a b f x → isSetY _ _) refl
  Iso.leftInv (respIso Y isSetY) (h , r) =
    Σ≡Prop (λ h' → isPropΠ3 λ u v _ → isSetY _ _) refl

  Lan-universal : (Y : Type₀) → isSet Y → (colim → Y) ≃ Cocone Y
  Lan-universal Y isSetY =
    compEquiv (isoToEquiv (setQuotUniversalIso isSetY))
              (isoToEquiv (respIso Y isSetY))

  -- naturality in Y: postcomposing with g : Y → Y' commutes with the equivalence
  Lan-natural : (Y Y' : Type₀) (sY : isSet Y) (sY' : isSet Y') (g : Y → Y') (h : colim → Y)
              → fst (equivFun (Lan-universal Y' sY') (g ∘ h))
              ≡ (λ c x → g (fst (equivFun (Lan-universal Y sY) h) c x))
  Lan-natural _ _ _ _ _ _ = refl

  -- the cocone of the identity is the colimit injection
  Lan-unit : fst (equivFun (Lan-universal colim isSetColim) (idfun colim)) ≡ inj
  Lan-unit = refl

  ----------------------------------------------------------------------
  -- lim F = compatible families

  lim : Type₀
  lim = Σ[ x ∈ (∀ c → F₀ c) ] (∀ a b (f : Hom a b) → F₁ f (x a) ≡ x b)

  isSetLim : isSet lim
  isSetLim = isSetΣ (isSetΠ isSetF₀)
                    (λ x → isProp→isSet (isPropΠ3 λ a b f → isSetF₀ b _ _))

  proj : ∀ c → lim → F₀ c
  proj c l = fst l c

  -- Ran_p F = lim F : the universal property
  Ran-universal : (Y : Type₀) → (Y → lim) ≃ Cone Y
  Ran-universal Y = isoToEquiv i
    where
    i : Iso (Y → lim) (Cone Y)
    Iso.fun i h = (λ c y → fst (h y) c) , λ a b f y → snd (h y) a b f
    Iso.inv i (β , nat) y = (λ c → β c y) , λ a b f → nat a b f y
    Iso.rightInv i _ = refl
    Iso.leftInv i _ = refl

  Ran-natural : (Y Y' : Type₀) (g : Y → Y') (h : Y' → lim)
              → fst (equivFun (Ran-universal Y) (h ∘ g))
              ≡ (λ c y → fst (equivFun (Ran-universal Y') h) c (g y))
  Ran-natural _ _ _ _ = refl

------------------------------------------------------------------------
-- Corollary: a group action.  Lan = orbits, Ran = fixed points.

record Group : Type₁ where
  field
    G     : Type₀
    isSetG : isSet G
    e     : G
    _·_   : G → G → G
    ·IdL  : ∀ g → e · g ≡ g
    ·IdR  : ∀ g → g · e ≡ g
    ·Assoc : ∀ g h k → (g · h) · k ≡ g · (h · k)

module GroupAction (Gr : Group) where
  open Group Gr

  -- the delooping BG: one object, arrows = G, composition = multiplication
  BG : Category
  Category.Ob BG = Unit
  Category.Hom BG _ _ = G
  Category.isSetHom BG _ _ = isSetG
  Category.idn BG _ = e
  Category._⋆_ BG g h = h · g
  Category.⋆IdL BG = ·IdR
  Category.⋆IdR BG = ·IdL
  Category.⋆Assoc BG f g h = sym (·Assoc h g f)

  -- a left action of G on a set X is a functor BG → Set
  record Action : Type₁ where
    field
      X      : Type₀
      isSetX : isSet X
      act    : G → X → X
      act-e  : ∀ x → act e x ≡ x
      act-·  : ∀ g h x → act (h · g) x ≡ act h (act g x)

  module _ (A : Action) where
    open Action A

    asFunctor : SetFunctor BG
    SetFunctor.F₀ asFunctor _ = X
    SetFunctor.isSetF₀ asFunctor _ = isSetX
    SetFunctor.F₁ asFunctor g = act g
    SetFunctor.F-id asFunctor _ = act-e
    SetFunctor.F-seq asFunctor f g = act-· f g

    Orbits : Type₀
    Orbits = X / (λ x y → Σ[ g ∈ G ] act g x ≡ y)

    FixedPoints : Type₀
    FixedPoints = Σ[ x ∈ X ] (∀ g → act g x ≡ x)

    -- the colimit over BG is the orbit set
    Lan≃Orbits : colim asFunctor ≃ Orbits
    Lan≃Orbits = isoToEquiv i
      where
      to : colim asFunctor → Orbits
      to = SQ.rec squash/ (λ u → [ snd u ]) λ { (_ , x) (_ , y) (g , p) → eq/ x y (g , p) }
      from : Orbits → colim asFunctor
      from = SQ.rec squash/ (λ x → [ tt , x ]) λ x y r → eq/ (tt , x) (tt , y) r
      i : Iso (colim asFunctor) Orbits
      Iso.fun i = to
      Iso.inv i = from
      Iso.rightInv i = SQ.elimProp (λ _ → squash/ _ _) λ _ → refl
      Iso.leftInv i = SQ.elimProp (λ _ → squash/ _ _) λ _ → refl

    -- the limit over BG is the set of fixed points
    Ran≃FixedPoints : lim asFunctor ≃ FixedPoints
    Ran≃FixedPoints = isoToEquiv i
      where
      i : Iso (lim asFunctor) FixedPoints
      Iso.fun i (x , c) = x tt , λ g → c tt tt g
      Iso.inv i (x , c) = (λ _ → x) , λ _ _ g → c g
      Iso.rightInv i _ = refl
      Iso.leftInv i _ = refl

------------------------------------------------------------------------
-- The concrete instance: the span 1 <- 0 -> 2 of tests/nlab.

data SpanOb : Type₀ where o0 o1 o2 : SpanOb

SpanHom : SpanOb → SpanOb → Type₀
SpanHom o0 o0 = Unit
SpanHom o0 o1 = Unit
SpanHom o0 o2 = Unit
SpanHom o1 o1 = Unit
SpanHom o2 o2 = Unit
SpanHom _  _  = ⊥

isPropSpanHom : ∀ a b → isProp (SpanHom a b)
isPropSpanHom o0 o0 = isPropUnit
isPropSpanHom o0 o1 = isPropUnit
isPropSpanHom o0 o2 = isPropUnit
isPropSpanHom o1 o0 = isProp⊥
isPropSpanHom o1 o1 = isPropUnit
isPropSpanHom o1 o2 = isProp⊥
isPropSpanHom o2 o0 = isProp⊥
isPropSpanHom o2 o1 = isProp⊥
isPropSpanHom o2 o2 = isPropUnit

spanId : ∀ a → SpanHom a a
spanId o0 = tt
spanId o1 = tt
spanId o2 = tt

spanSeq : ∀ {a b c} → SpanHom a b → SpanHom b c → SpanHom a c
spanSeq {o0} {o0} {o0} _ _ = tt
spanSeq {o0} {o0} {o1} _ _ = tt
spanSeq {o0} {o0} {o2} _ _ = tt
spanSeq {o0} {o1} {o1} _ _ = tt
spanSeq {o0} {o2} {o2} _ _ = tt
spanSeq {o1} {o1} {o1} _ _ = tt
spanSeq {o2} {o2} {o2} _ _ = tt
spanSeq {o0} {o1} {o0} _ ()
spanSeq {o0} {o1} {o2} _ ()
spanSeq {o0} {o2} {o0} _ ()
spanSeq {o0} {o2} {o1} _ ()
spanSeq {o1} {o0} () _
spanSeq {o1} {o1} {o0} _ ()
spanSeq {o1} {o1} {o2} _ ()
spanSeq {o1} {o2} () _
spanSeq {o2} {o0} () _
spanSeq {o2} {o1} () _
spanSeq {o2} {o2} {o0} _ ()
spanSeq {o2} {o2} {o1} _ ()

SpanCat : Category
Category.Ob SpanCat = SpanOb
Category.Hom SpanCat = SpanHom
Category.isSetHom SpanCat a b = isProp→isSet (isPropSpanHom a b)
Category.idn SpanCat = spanId
Category._⋆_ SpanCat = spanSeq
Category.⋆IdL SpanCat {a} {b} _ = isPropSpanHom a b _ _
Category.⋆IdR SpanCat {a} {b} _ = isPropSpanHom a b _ _
Category.⋆Assoc SpanCat {a} {_} {_} {d} _ _ _ = isPropSpanHom a d _ _

data S0 : Type₀ where x y : S0
data S1 : Type₀ where a b c : S1
data S2 : Type₀ where p q : S2

-- each finite set is a set (Hedberg, via an injection into Fin)
S0→Fin : S0 → Fin 2
S0→Fin x = zero
S0→Fin y = suc zero
S1→Fin : S1 → Fin 3
S1→Fin a = zero
S1→Fin b = suc zero
S1→Fin c = suc (suc zero)
S2→Fin : S2 → Fin 2
S2→Fin p = zero
S2→Fin q = suc zero

discreteS0 : Discrete S0
discreteS0 x x = yes refl
discreteS0 y y = yes refl
discreteS0 x y = no λ e → znots (cong S0→Fin e)
discreteS0 y x = no λ e → snotz (cong S0→Fin e)

predFin' : Fin 3 → Fin 2
predFin' zero = zero
predFin' (suc i) = i

discreteS1 : Discrete S1
discreteS1 a a = yes refl
discreteS1 b b = yes refl
discreteS1 c c = yes refl
discreteS1 a b = no λ e → znots (cong S1→Fin e)
discreteS1 a c = no λ e → znots (cong S1→Fin e)
discreteS1 b a = no λ e → snotz (cong S1→Fin e)
discreteS1 c a = no λ e → snotz (cong S1→Fin e)
discreteS1 b c = no λ e → znots (cong (predFin' ∘ S1→Fin) e)
discreteS1 c b = no λ e → snotz (cong (predFin' ∘ S1→Fin) e)

discreteS2 : Discrete S2
discreteS2 p p = yes refl
discreteS2 q q = yes refl
discreteS2 p q = no λ e → znots (cong S2→Fin e)
discreteS2 q p = no λ e → snotz (cong S2→Fin e)

SpanF₀ : SpanOb → Type₀
SpanF₀ o0 = S0
SpanF₀ o1 = S1
SpanF₀ o2 = S2

m1 : S0 → S1
m1 _ = a
m2 : S0 → S2
m2 x = p
m2 y = q

SpanF₁ : ∀ {i j} → SpanHom i j → SpanF₀ i → SpanF₀ j
SpanF₁ {o0} {o0} _ s = s
SpanF₁ {o0} {o1} _ s = m1 s
SpanF₁ {o0} {o2} _ s = m2 s
SpanF₁ {o1} {o1} _ s = s
SpanF₁ {o2} {o2} _ s = s
SpanF₁ {o1} {o0} ()
SpanF₁ {o1} {o2} ()
SpanF₁ {o2} {o0} ()
SpanF₁ {o2} {o1} ()

GSpan : SetFunctor SpanCat
SetFunctor.F₀ GSpan = SpanF₀
SetFunctor.isSetF₀ GSpan o0 = Discrete→isSet discreteS0
SetFunctor.isSetF₀ GSpan o1 = Discrete→isSet discreteS1
SetFunctor.isSetF₀ GSpan o2 = Discrete→isSet discreteS2
SetFunctor.F₁ GSpan = SpanF₁
SetFunctor.F-id GSpan o0 _ = refl
SetFunctor.F-id GSpan o1 _ = refl
SetFunctor.F-id GSpan o2 _ = refl
SetFunctor.F-seq GSpan {o0} {o0} {o0} _ _ _ = refl
SetFunctor.F-seq GSpan {o0} {o0} {o1} _ _ _ = refl
SetFunctor.F-seq GSpan {o0} {o0} {o2} _ _ _ = refl
SetFunctor.F-seq GSpan {o0} {o1} {o1} _ _ _ = refl
SetFunctor.F-seq GSpan {o0} {o2} {o2} _ _ _ = refl
SetFunctor.F-seq GSpan {o1} {o1} {o1} _ _ _ = refl
SetFunctor.F-seq GSpan {o2} {o2} {o2} _ _ _ = refl
SetFunctor.F-seq GSpan {o0} {o1} {o0} _ ()
SetFunctor.F-seq GSpan {o0} {o1} {o2} _ ()
SetFunctor.F-seq GSpan {o0} {o2} {o0} _ ()
SetFunctor.F-seq GSpan {o0} {o2} {o1} _ ()
SetFunctor.F-seq GSpan {o1} {o0} ()
SetFunctor.F-seq GSpan {o1} {o1} {o0} _ ()
SetFunctor.F-seq GSpan {o1} {o1} {o2} _ ()
SetFunctor.F-seq GSpan {o1} {o2} ()
SetFunctor.F-seq GSpan {o2} {o0} ()
SetFunctor.F-seq GSpan {o2} {o1} ()
SetFunctor.F-seq GSpan {o2} {o2} {o0} _ ()
SetFunctor.F-seq GSpan {o2} {o2} {o1} _ ()

------------------------------------------------------------------------
-- |Lan_! G| = |colim G| = 3: the classes {a p q x y}, {b}, {c}

class : Elt GSpan → Fin 3
class (o0 , _) = zero
class (o1 , a) = zero
class (o1 , b) = suc zero
class (o1 , c) = suc (suc zero)
class (o2 , _) = zero

class-arrow : ∀ i j (f : SpanHom i j) (s : SpanF₀ i) → class (i , s) ≡ class (j , SpanF₁ f s)
class-arrow o0 o0 _ _ = refl
class-arrow o0 o1 _ _ = refl
class-arrow o0 o2 _ x = refl
class-arrow o0 o2 _ y = refl
class-arrow o1 o1 _ _ = refl
class-arrow o2 o2 _ _ = refl

colim→Fin3 : colim GSpan → Fin 3
colim→Fin3 = SQ.rec isSetFin class
  λ { (i , s) (j , t) (f , e) → class-arrow i j f s ∙ cong (λ z → class (j , z)) e }

Fin3→colim : Fin 3 → colim GSpan
Fin3→colim zero = [ o1 , a ]
Fin3→colim (suc zero) = [ o1 , b ]
Fin3→colim (suc (suc zero)) = [ o1 , c ]

private
  glue : ∀ i j (f : SpanHom i j) (s : SpanF₀ i) → Path (colim GSpan) [ i , s ] [ j , SpanF₁ f s ]
  glue i j f s = eq/ (i , s) (j , SpanF₁ f s) (f , refl)

section-colim : ∀ u → Fin3→colim (colim→Fin3 [ u ]) ≡ [ u ]
section-colim (o0 , x) = sym (glue o0 o1 tt x)
section-colim (o0 , y) = sym (glue o0 o1 tt y)
section-colim (o1 , a) = refl
section-colim (o1 , b) = refl
section-colim (o1 , c) = refl
section-colim (o2 , p) = sym (glue o0 o1 tt x) ∙ glue o0 o2 tt x
section-colim (o2 , q) = sym (glue o0 o1 tt y) ∙ glue o0 o2 tt y

colim≃Fin3 : colim GSpan ≃ Fin 3
colim≃Fin3 = isoToEquiv i
  where
  i : Iso (colim GSpan) (Fin 3)
  Iso.fun i = colim→Fin3
  Iso.inv i = Fin3→colim
  Iso.rightInv i zero = refl
  Iso.rightInv i (suc zero) = refl
  Iso.rightInv i (suc (suc zero)) = refl
  Iso.leftInv i = SQ.elimProp (λ _ → squash/ _ _) section-colim

------------------------------------------------------------------------
-- |Ran_! G| = |lim G| = 2: a compatible family is fixed by its value at 0

lim→Fin2 : lim GSpan → Fin 2
lim→Fin2 l = S0→Fin (fst l o0)

family : S0 → lim GSpan
family s = fam , compat
  where
  fam : ∀ i → SpanF₀ i
  fam o0 = s
  fam o1 = m1 s
  fam o2 = m2 s
  compat : ∀ i j (f : SpanHom i j) → SpanF₁ f (fam i) ≡ fam j
  compat o0 o0 _ = refl
  compat o0 o1 _ = refl
  compat o0 o2 _ = refl
  compat o1 o1 _ = refl
  compat o2 o2 _ = refl

Fin2→lim : Fin 2 → lim GSpan
Fin2→lim zero = family x
Fin2→lim (suc zero) = family y

lim-determined : (l : lim GSpan) → family (fst l o0) ≡ l
lim-determined l =
  Σ≡Prop (λ v → isPropΠ3 λ i j f → SetFunctor.isSetF₀ GSpan j _ _)
         (funExt pointwise)
  where
  pointwise : ∀ i → fst (family (fst l o0)) i ≡ fst l i
  pointwise o0 = refl
  pointwise o1 = snd l o0 o1 tt
  pointwise o2 = snd l o0 o2 tt

lim≃Fin2 : lim GSpan ≃ Fin 2
lim≃Fin2 = isoToEquiv i
  where
  i : Iso (lim GSpan) (Fin 2)
  Iso.fun i = lim→Fin2
  Iso.inv i = Fin2→lim
  Iso.rightInv i zero = refl
  Iso.rightInv i (suc zero) = refl
  Iso.leftInv i l = round (fst l o0) ∙ lim-determined l
    where
    round : (s : S0) → Fin2→lim (S0→Fin s) ≡ family s
    round x = refl
    round y = refl

-- Lan_! G and Ran_! G themselves (the representing objects) are colim G and
-- lim G by Lan-universal and Ran-universal above, so the Eshkol row
-- (|Lan_! G| |colim G| |Ran_! G| |lim G|) = (3 3 2 2) is:
Lan!-size : colim GSpan ≃ Fin 3
Lan!-size = colim≃Fin3

Ran!-size : lim GSpan ≃ Fin 2
Ran!-size = lim≃Fin2
