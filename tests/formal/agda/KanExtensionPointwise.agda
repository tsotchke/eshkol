{-# OPTIONS --cubical --safe --guardedness -WnoUnsupportedIndexedMatch #-}
------------------------------------------------------------------------
-- Pointwise Kan extensions of Set-valued functors by the coend and end
-- formulas.
--
-- nLab, "Kan extension" (tests/nlab/nlab_category_test.esk):
--   [Kan extension#1] "(Lan_p F)(c') ≅ ∫^{c ∈ C} C'(p(c),c') · F(c)" and
--                     "(Ran_p F)(c') ≅ ∫_{c ∈ C} [C'(c',p(c)),F(c)]"
--   [Kan extension#3] when p is fully faithful the unit F ⇒ p^* Lan_p F is
--                     an isomorphism ("restricts back to F")
--
-- General theorems, for every functor p : C → D between small categories
-- and every Set-valued functor F on C:
--   * the coend  (Lan_p F)(d) = (Σ c. D(p c, d) × F c) / ~  is a functor
--     D → Set and Nat(Lan_p F, G) ≃ Nat(F, G ∘ p) for every G : D → Set;
--   * the end  (Ran_p F)(d) = { ξ : ∀ c. D(d, p c) → F c, natural in c }
--     is a functor D → Set and Nat(G, Ran_p F) ≃ Nat(G ∘ p, F);
--   * if p is fully faithful, the unit  F c → (Lan_p F)(p c),
--     x ↦ [c, id, x], is a bijection for every c.
-- Eshkol computes these by the same two formulas (set-lan-coend,
-- set-ran-end) and by the comma-category colimit; [Kan extension#1] checks
-- that they agree on every functor span → arrow and arrow → BZ/2, and
-- [Kan extension#3] that the unit is bijective along the full inclusion of
-- {0, 1} into the commutative square and not along discrete 2 → arrow.
------------------------------------------------------------------------
module KanExtensionPointwise where

open import Cubical.Foundations.Prelude
open import Cubical.Foundations.Isomorphism
open import Cubical.Foundations.Equiv
open import Cubical.Foundations.HLevels
open import Cubical.Data.Sigma
open import Cubical.HITs.SetQuotients as SQ

open import KanExtensionTerminal using (Category; SetFunctor)

record Functor (C D : Category) : Type₁ where
  private
    module C = Category C
    module D = Category D
  field
    p₀    : C.Ob → D.Ob
    p₁    : ∀ {a b} → C.Hom a b → D.Hom (p₀ a) (p₀ b)
    p-id  : ∀ a → p₁ (C.idn a) ≡ D.idn (p₀ a)
    p-seq : ∀ {a b c} (f : C.Hom a b) (g : C.Hom b c)
          → p₁ (f C.⋆ g) ≡ p₁ f D.⋆ p₁ g

-- natural transformations between Set-valued functors
module _ {C : Category} where
  open Category C
  Nat : SetFunctor C → SetFunctor C → Type₀
  Nat F G = Σ[ α ∈ (∀ c → F.F₀ c → G.F₀ c) ]
            (∀ a b (f : Hom a b) x → α b (F.F₁ f x) ≡ G.F₁ f (α a x))
    where module F = SetFunctor F
          module G = SetFunctor G

  isPropNatural : (F G : SetFunctor C) (α : ∀ c → SetFunctor.F₀ F c → SetFunctor.F₀ G c)
                → isProp (∀ a b (f : Hom a b) x
                          → α b (SetFunctor.F₁ F f x) ≡ SetFunctor.F₁ G f (α a x))
  isPropNatural F G α = isPropΠ4 λ a b f x → SetFunctor.isSetF₀ G b _ _

module _ {C D : Category} (p : Functor C D) where
  private
    module C = Category C
    module D = Category D
  open Functor p

  -- restriction along p
  _∘p : SetFunctor D → SetFunctor C
  SetFunctor.F₀ (G ∘p) c = SetFunctor.F₀ G (p₀ c)
  SetFunctor.isSetF₀ (G ∘p) c = SetFunctor.isSetF₀ G (p₀ c)
  SetFunctor.F₁ (G ∘p) u = SetFunctor.F₁ G (p₁ u)
  SetFunctor.F-id (G ∘p) a x = cong (λ g → SetFunctor.F₁ G g x) (p-id a) ∙ SetFunctor.F-id G _ x
  SetFunctor.F-seq (G ∘p) f g x =
    cong (λ h → SetFunctor.F₁ G h x) (p-seq f g) ∙ SetFunctor.F-seq G (p₁ f) (p₁ g) x

  module _ (F : SetFunctor C) where
    open SetFunctor F

    --------------------------------------------------------------------
    -- Lan_p F by the coend formula

    Tri : D.Ob → Type₀
    Tri d = Σ[ c ∈ C.Ob ] (D.Hom (p₀ c) d × F₀ c)

    -- (c, p u ⋆ f₀, x) ~ (c₀, f₀, F u x): the coend relation
    Rel : ∀ d → Tri d → Tri d → Type₀
    Rel d (c , f , x) (c₀ , f₀ , x₀) =
      Σ[ u ∈ C.Hom c c₀ ] (f ≡ p₁ u D.⋆ f₀) × (F₁ u x ≡ x₀)

    LanObj : D.Ob → Type₀
    LanObj d = Tri d / Rel d

    Lan₁ : ∀ {d d'} → D.Hom d d' → LanObj d → LanObj d'
    Lan₁ h = SQ.rec squash/ (λ { (c , f , x) → [ c , f D.⋆ h , x ] })
      λ { (c , f , x) (c₀ , f₀ , x₀) (u , e , e') →
          eq/ _ _ (u , cong (D._⋆ h) e ∙ D.⋆Assoc (p₁ u) f₀ h , e') }

    Lan : SetFunctor D
    SetFunctor.F₀ Lan = LanObj
    SetFunctor.isSetF₀ Lan _ = squash/
    SetFunctor.F₁ Lan = Lan₁
    SetFunctor.F-id Lan d = SQ.elimProp (λ _ → squash/ _ _)
      λ { (c , f , x) → cong (λ g → [ c , g , x ]) (D.⋆IdR f) }
    SetFunctor.F-seq Lan h k = SQ.elimProp (λ _ → squash/ _ _)
      λ { (c , f , x) → cong (λ g → [ c , g , x ]) (sym (D.⋆Assoc f h k)) }

    -- the unit F ⇒ Lan_p F ∘ p
    η : ∀ c → F₀ c → LanObj (p₀ c)
    η c x = [ c , D.idn (p₀ c) , x ]

    module _ (G : SetFunctor D) where
      private module G = SetFunctor G

      Lan-universal : Iso (Nat Lan G) (Nat F (G ∘p))
      Iso.fun Lan-universal (α , nat) =
        (λ c x → α (p₀ c) (η c x)) ,
        λ a b u x →
            cong (α (p₀ b)) (sym (eq/ (a , p₁ u , x) (b , D.idn (p₀ b) , F₁ u x)
                                      (u , sym (D.⋆IdR (p₁ u)) , refl)))
          ∙ cong (λ g → α (p₀ b) [ a , g , x ]) (sym (D.⋆IdL (p₁ u)))
          ∙ nat (p₀ a) (p₀ b) (p₁ u) (η a x)
      Iso.inv Lan-universal (β , nat) = α , natα
        where
        α : ∀ d → LanObj d → G.F₀ d
        α d = SQ.rec (G.isSetF₀ d) (λ { (c , f , x) → G.F₁ f (β c x) })
          λ { (c , f , x) (c₀ , f₀ , x₀) (u , e , e') →
                cong (λ g → G.F₁ g (β c x)) e
              ∙ G.F-seq (p₁ u) f₀ (β c x)
              ∙ cong (G.F₁ f₀) (sym (nat c c₀ u x))
              ∙ cong (λ y → G.F₁ f₀ (β c₀ y)) e' }
        natα : ∀ a b (h : D.Hom a b) z → α b (Lan₁ h z) ≡ G.F₁ h (α a z)
        natα a b h = SQ.elimProp (λ _ → G.isSetF₀ b _ _)
          λ { (c , f , x) → G.F-seq f h (β c x) }
      Iso.rightInv Lan-universal (β , nat) =
        Σ≡Prop (isPropNatural F (G ∘p))
               (funExt λ c → funExt λ x → G.F-id (p₀ c) (β c x))
      Iso.leftInv Lan-universal (α , nat) =
        Σ≡Prop (isPropNatural Lan G)
               (funExt λ d → funExt (SQ.elimProp (λ _ → G.isSetF₀ d _ _)
                 λ { (c , f , x) →
                       sym (nat (p₀ c) d f (η c x))
                     ∙ cong (λ g → α d [ c , g , x ]) (D.⋆IdL f) }))

      Lan≃ : Nat Lan G ≃ Nat F (G ∘p)
      Lan≃ = isoToEquiv Lan-universal

    --------------------------------------------------------------------
    -- Ran_p F by the end formula

    RanObj : D.Ob → Type₀
    RanObj d = Σ[ ξ ∈ (∀ c → D.Hom d (p₀ c) → F₀ c) ]
               (∀ c c₀ (u : C.Hom c c₀) (f : D.Hom d (p₀ c)) → F₁ u (ξ c f) ≡ ξ c₀ (f D.⋆ p₁ u))

    isPropRanCompat : ∀ d (ξ : ∀ c → D.Hom d (p₀ c) → F₀ c)
      → isProp (∀ c c₀ (u : C.Hom c c₀) (f : D.Hom d (p₀ c)) → F₁ u (ξ c f) ≡ ξ c₀ (f D.⋆ p₁ u))
    isPropRanCompat d ξ = isPropΠ4 λ c c₀ u f → isSetF₀ c₀ _ _

    Ran₁ : ∀ {d d'} → D.Hom d d' → RanObj d → RanObj d'
    Ran₁ h (ξ , cmp) = (λ c f → ξ c (h D.⋆ f)) ,
      λ c c₀ u f → cmp c c₀ u (h D.⋆ f) ∙ cong (ξ c₀) (D.⋆Assoc h f (p₁ u))

    Ran : SetFunctor D
    SetFunctor.F₀ Ran = RanObj
    SetFunctor.isSetF₀ Ran d =
      isSetΣ (isSetΠ2 λ c _ → isSetF₀ c) (λ ξ → isProp→isSet (isPropRanCompat d ξ))
    SetFunctor.F₁ Ran = Ran₁
    SetFunctor.F-id Ran d (ξ , _) =
      Σ≡Prop (isPropRanCompat d) (funExt λ c → funExt λ f → cong (ξ c) (D.⋆IdL f))
    SetFunctor.F-seq Ran {c = d''} h k (ξ , _) =
      Σ≡Prop (isPropRanCompat d'') (funExt λ c → funExt λ f → cong (ξ c) (D.⋆Assoc h k f))

    -- the counit Ran_p F ∘ p ⇒ F
    ε : ∀ c → RanObj (p₀ c) → F₀ c
    ε c (ξ , _) = ξ c (D.idn (p₀ c))

    module _ (G : SetFunctor D) where
      private module G = SetFunctor G

      Ran-universal : Iso (Nat G Ran) (Nat (G ∘p) F)
      Iso.fun Ran-universal (α , nat) =
        (λ c y → ε c (α (p₀ c) y)) ,
        λ a b u y →
            cong (λ r → fst r b (D.idn (p₀ b))) (nat (p₀ a) (p₀ b) (p₁ u) y)
          ∙ cong (fst (α (p₀ a) y) b) (D.⋆IdR (p₁ u) ∙ sym (D.⋆IdL (p₁ u)))
          ∙ sym (snd (α (p₀ a) y) a b u (D.idn (p₀ a)))
      Iso.inv Ran-universal (β , nat) = α , natα
        where
        α : ∀ d → G.F₀ d → RanObj d
        α d y = (λ c f → β c (G.F₁ f y)) ,
          λ c c₀ u f → sym (nat c c₀ u (G.F₁ f y))
                     ∙ cong (β c₀) (sym (G.F-seq f (p₁ u) y))
        natα : ∀ a b (h : D.Hom a b) y → α b (G.F₁ h y) ≡ Ran₁ h (α a y)
        natα a b h y = Σ≡Prop (isPropRanCompat b)
          (funExt λ c → funExt λ f → cong (β c) (sym (G.F-seq h f y)))
      Iso.rightInv Ran-universal (β , nat) =
        Σ≡Prop (isPropNatural (G ∘p) F)
               (funExt λ c → funExt λ y → cong (β c) (G.F-id (p₀ c) y))
      Iso.leftInv Ran-universal (α , nat) =
        Σ≡Prop (isPropNatural G Ran)
               (funExt λ d → funExt λ y → Σ≡Prop (isPropRanCompat d)
                 (funExt λ c → funExt λ f →
                     cong (λ r → fst r c (D.idn (p₀ c))) (nat d (p₀ c) f y)
                   ∙ cong (fst (α d y) c) (D.⋆IdR f)))

      Ran≃ : Nat G Ran ≃ Nat (G ∘p) F
      Ran≃ = isoToEquiv Ran-universal

    --------------------------------------------------------------------
    -- [Kan extension#3] along a fully faithful p the unit is a bijection

    record FullyFaithful : Type₀ where
      field
        pinv : ∀ {a b} → D.Hom (p₀ a) (p₀ b) → C.Hom a b
        sec  : ∀ {a b} (f : D.Hom (p₀ a) (p₀ b)) → p₁ (pinv f) ≡ f
        ret  : ∀ {a b} (u : C.Hom a b) → pinv (p₁ u) ≡ u

    module _ (ff : FullyFaithful) (c : C.Ob) where
      open FullyFaithful ff

      back : LanObj (p₀ c) → F₀ c
      back = SQ.rec (isSetF₀ c) (λ { (c' , f , x) → F₁ (pinv f) x })
        λ { (c' , f , x) (c₀ , f₀ , x₀) (u , e , e') →
              cong (λ g → F₁ (pinv g) x)
                (e ∙ cong (p₁ u D.⋆_) (sym (sec f₀)) ∙ sym (p-seq u (pinv f₀)))
            ∙ cong (λ v → F₁ v x) (ret (u C.⋆ pinv f₀))
            ∙ F-seq u (pinv f₀) x
            ∙ cong (F₁ (pinv f₀)) e' }

      unit-iso : Iso (F₀ c) (LanObj (p₀ c))
      Iso.fun unit-iso = η c
      Iso.inv unit-iso = back
      Iso.leftInv unit-iso x =
          cong (λ v → F₁ v x) (cong pinv (sym (p-id c)) ∙ ret (C.idn c))
        ∙ F-id c x
      Iso.rightInv unit-iso = SQ.elimProp (λ _ → squash/ _ _)
        λ { (c' , f , x) →
            sym (eq/ (c' , f , x) (c , D.idn (p₀ c) , F₁ (pinv f) x)
                     (pinv f , sym (D.⋆IdR _ ∙ sec f) , refl)) }
