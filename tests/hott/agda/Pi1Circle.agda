{-# OPTIONS --cubical --safe #-}
-- Cross-check for Eshkol's cubical kernel (ADR-0043).
--
-- The same definitions as lib/hott/{foundations,numbers,circle}.esk, written
-- against Agda's builtin cubical primitives only (no library), so the file
-- checks with a bare `agda --cubical`. Every `_ = refl` below is a statement
-- the Eshkol kernel also proves by refl (tests/hott/programs/*.esk): Cubical
-- Agda computes the same normal forms.
module Pi1Circle where

open import Agda.Primitive
open import Agda.Primitive.Cubical
  renaming (primINeg to ~_; primIMax to _∨_; primIMin to _∧_;
            primHComp to hcomp; primTransp to transp; itIsOne to 1=1)
open import Agda.Builtin.Cubical.Path
open import Agda.Builtin.Cubical.Sub renaming (primSubOut to outS)
open import Agda.Builtin.Cubical.Glue renaming (primGlue to primGlue')
open import Agda.Builtin.Cubical.Equiv
import Agda.Builtin.Cubical.HCompU

open Agda.Builtin.Cubical.HCompU.Helpers using (hfill)
open import Agda.Builtin.Sigma
open import Agda.Builtin.Nat renaming (Nat to ℕ)
open import Agda.Builtin.Int renaming (Int to ℤ)

private variable ℓ : Level

-- ---- foundations ----
refl' : {A : Set ℓ} (a : A) → a ≡ a
refl' a = λ _ → a

sym : {A : Set ℓ} {a b : A} → a ≡ b → b ≡ a
sym p i = p (~ i)

_∙_ : {A : Set ℓ} {a b c : A} → a ≡ b → b ≡ c → a ≡ c
_∙_ {a = a} p q i = hcomp (λ j → λ { (i = i0) → a ; (i = i1) → q j }) (p i)

ap : {A B : Set ℓ} (f : A → B) {a b : A} → a ≡ b → f a ≡ f b
ap f p i = f (p i)

transport : {A B : Set ℓ} → A ≡ B → A → B
transport p x = transp (λ i → p i) i0 x

J : {A : Set ℓ} {a : A} (P : (x : A) → a ≡ x → Set ℓ) → P a (refl' a) →
    {x : A} (p : a ≡ x) → P x p
J P d p = transp (λ i → P (p i) (λ j → p (i ∧ j))) i0 d

idEquiv : (A : Set ℓ) → A ≃ A
idEquiv A .fst x = x
idEquiv A .snd .equiv-proof y = (y , λ _ → y) , λ z i → z .snd (~ i) , λ j → z .snd (~ i ∨ j)

Glue : (A : Set ℓ) {φ : I} → Partial φ (Σ (Set ℓ) λ T → T ≃ A) → Set ℓ
Glue A Te = primGlue' A (λ x → Te x .fst) (λ x → Te x .snd)

ua : {A B : Set ℓ} → A ≃ B → A ≡ B
ua {A = A} {B = B} e i = Glue B (λ { (i = i0) → (A , e) ; (i = i1) → (B , idEquiv B) })

-- isomorphisms are equivalences (Cubical.Foundations.Isomorphism)
module _ {A B : Set ℓ} (f : A → B) (g : B → A)
         (s : (y : B) → f (g y) ≡ y) (t : (x : A) → g (f x) ≡ x) where
  private
    module _ (y : B) (x0 x1 : A) (p0 : f x0 ≡ y) (p1 : f x1 ≡ y) where
      fill0 : I → I → A
      fill0 i = hfill (λ k → λ { (i = i1) → t x0 k ; (i = i0) → g y }) (inS (g (p0 (~ i))))
      fill1 : I → I → A
      fill1 i = hfill (λ k → λ { (i = i1) → t x1 k ; (i = i0) → g y }) (inS (g (p1 (~ i))))
      fill2 : I → I → A
      fill2 i = hfill (λ k → λ { (i = i1) → fill1 k i1 ; (i = i0) → fill0 k i1 }) (inS (g y))
      p : x0 ≡ x1
      p i = fill2 i i1
      sq : I → I → A
      sq i j = hcomp (λ k → λ { (i = i1) → fill1 j (~ k) ; (i = i0) → fill0 j (~ k)
                              ; (j = i1) → t (fill2 i i1) (~ k) ; (j = i0) → g y })
                     (fill2 i j)
      sq1 : I → I → B
      sq1 i j = hcomp (λ k → λ { (i = i1) → s (p1 (~ j)) k ; (i = i0) → s (p0 (~ j)) k
                               ; (j = i1) → s (f (p i)) k ; (j = i0) → s y k })
                      (f (sq i j))
      lemIso : (x0 , p0) ≡ (x1 , p1)
      lemIso i .fst = p i
      lemIso i .snd = λ j → sq1 i (~ j)

  isoToEquiv : A ≃ B
  isoToEquiv .fst = f
  isoToEquiv .snd .equiv-proof y .fst .fst = g y
  isoToEquiv .snd .equiv-proof y .fst .snd = s y
  isoToEquiv .snd .equiv-proof y .snd z = lemIso y (g y) (fst z) (s y) (snd z)

-- ---- numbers ----
sucNeg : ℕ → ℤ
sucNeg zero = pos zero
sucNeg (suc m) = negsuc m

predPos : ℕ → ℤ
predPos zero = negsuc zero
predPos (suc m) = pos m

sucℤ : ℤ → ℤ
sucℤ (pos n) = pos (suc n)
sucℤ (negsuc n) = sucNeg n

predℤ : ℤ → ℤ
predℤ (pos n) = predPos n
predℤ (negsuc n) = negsuc (suc n)

sucPred : (x : ℤ) → sucℤ (predℤ x) ≡ x
sucPred (pos zero) = refl' _
sucPred (pos (suc m)) = refl' _
sucPred (negsuc n) = refl' _

predSuc : (x : ℤ) → predℤ (sucℤ x) ≡ x
predSuc (pos n) = refl' _
predSuc (negsuc zero) = refl' _
predSuc (negsuc (suc m)) = refl' _

sucEquiv : ℤ ≃ ℤ
sucEquiv = isoToEquiv sucℤ predℤ sucPred predSuc

sucPathℤ : ℤ ≡ ℤ
sucPathℤ = ua sucEquiv

_ : transport sucPathℤ (pos 4) ≡ pos 5
_ = refl' _

_ : transport (sym sucPathℤ) (pos 0) ≡ negsuc 0
_ = refl' _

_ : transport (sucPathℤ ∙ sucPathℤ) (negsuc 4) ≡ negsuc 2
_ = refl' _

-- ---- the circle ----
data S¹ : Set where
  base : S¹
  loop : base ≡ base

ΩS¹ : Set
ΩS¹ = base ≡ base

helix : S¹ → Set
helix base = ℤ
helix (loop i) = sucPathℤ i

winding : ΩS¹ → ℤ
winding p = transp (λ i → helix (p i)) i0 (pos 0)

encode : (x : S¹) → base ≡ x → helix x
encode x p = transp (λ i → helix (p i)) i0 (pos 0)

posLoop : ℕ → ΩS¹
posLoop zero = refl' base
posLoop (suc n) = posLoop n ∙ loop

negLoop : ℕ → ΩS¹
negLoop zero = sym loop
negLoop (suc n) = negLoop n ∙ sym loop

intLoop : ℤ → ΩS¹
intLoop (pos n) = posLoop n
intLoop (negsuc n) = negLoop n

decodeSquare : (n : ℤ) → PathP (λ i → base ≡ loop i) (intLoop (predℤ n)) (intLoop n)
decodeSquare (pos zero) i j = loop (i ∨ ~ j)
decodeSquare (pos (suc n)) i j =
  hcomp (λ k → λ { (j = i0) → base ; (j = i1) → loop (i ∧ k) ; (i = i0) → posLoop n j })
        (posLoop n j)
decodeSquare (negsuc n) i j =
  hcomp (λ k → λ { (j = i0) → base ; (j = i1) → loop (~ (~ i ∧ k)) ; (i = i1) → negLoop n j })
        (negLoop n j)

decode : (x : S¹) → helix x → base ≡ x
decode base = intLoop
decode (loop i) y j =
  hcomp (λ k → λ { (i = i0) → intLoop (predSuc y k) j
                 ; (i = i1) → intLoop y j
                 ; (j = i0) → base
                 ; (j = i1) → loop i })
        (decodeSquare (prim^unglue {φ = i ∨ ~ i} y) i j)

decodeEncode : (x : S¹) (p : base ≡ x) → decode x (encode x p) ≡ p
decodeEncode x p = J (λ y q → decode y (encode y q) ≡ q) (refl' (refl' base)) p

windingPos : (n : ℕ) → winding (posLoop n) ≡ pos n
windingPos zero = refl' _
windingPos (suc n) = ap sucℤ (windingPos n)

windingNeg : (n : ℕ) → winding (negLoop n) ≡ negsuc n
windingNeg zero = refl' _
windingNeg (suc n) = ap predℤ (windingNeg n)

windingIntLoop : (n : ℤ) → winding (intLoop n) ≡ n
windingIntLoop (pos n) = windingPos n
windingIntLoop (negsuc n) = windingNeg n

ΩS¹≃ℤ : ΩS¹ ≃ ℤ
ΩS¹≃ℤ = isoToEquiv winding intLoop windingIntLoop (decodeEncode base)

ΩS¹≡ℤ : ΩS¹ ≡ ℤ
ΩS¹≡ℤ = ua ΩS¹≃ℤ

-- the flagship values (tests/hott/programs/pi1_circle.expected)
_ : winding (loop ∙ loop) ≡ pos 2
_ = refl' _

_ : winding ((loop ∙ loop) ∙ sym loop) ≡ pos 1
_ = refl' _

_ : winding (intLoop (negsuc 3)) ≡ negsuc 3
_ = refl' _

_ : winding (intLoop (pos 5)) ≡ pos 5
_ = refl' _

_ : transport ΩS¹≡ℤ (loop ∙ loop) ≡ pos 2
_ = refl' _

_ : winding (transport (sym ΩS¹≡ℤ) (pos 3)) ≡ pos 3
_ = refl' _

-- transport along ua e applies e, also on a variable (tests/hott/programs/univalence.esk)
_ : (x : ℤ) → transport sucPathℤ x ≡ sucℤ x
_ = λ x → refl' _

-- J computes on refl when the motive is an inductive type (tests/hott/programs/paths.esk)
_ : (n : ℕ) → J (λ _ _ → ℕ) n {n} (refl' n) ≡ n
_ = λ n → refl' _
