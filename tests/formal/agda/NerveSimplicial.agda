{-# OPTIONS --safe --without-K #-}
------------------------------------------------------------------------
-- The nerve of a category is a simplicial set.
--
-- nLab claims (tests/nlab/nlab_space_test.esk, nlab_simplicial_test.esk):
--   [nerve#1]  "N(C)_n := C_1 ×_{C_0} ... ×_{C_0} C_1": n-simplices are
--              strings of n composable arrows; |N([1])_n| = n + 2
--   [nerve#2]  "s_k are given by inserting an identity morphism on x_k"
--   [simplicial set#1]  |Delta[n]_m| for n, m = 0..3
--   [simplicial set#2]  face maps d_i and degeneracy maps s_i satisfy the
--              simplicial identities
--
-- General theorem, for every category C (equations up to ≡): with
-- N(C)_n the composable strings of length n, the inner face d_i composes
-- at vertex i, the outer faces drop an end, and s_k inserts an identity at
-- vertex k, all five families of simplicial identities hold:
--   d_i d_j     = d_{j-1} d_i              (i < j)
--   d_i s_j     = s_{j-1} d_i              (i < j)
--   d_j s_j     = id = d_{j+1} s_j
--   d_i s_j     = s_j d_{i-1}              (i > j + 1)
--   s_i s_j     = s_{j+1} s_i              (i ≤ j)
-- They use exactly the unit and associativity laws of C.
--
-- Instances, decided by `refl` on an enumeration of strings:
--   |N([1])_n| = n + 2 for n = 0..5 (Eshkol: (2 3 4 5 6 7)), and
--   |Delta[n]_m| = |N([n])_m| = binomial(n+m+1, m+1) for n, m = 0..3.
------------------------------------------------------------------------
module NerveSimplicial where

open import Data.Nat using (ℕ; zero; suc; _+_; _*_)
open import Data.Fin using (Fin; zero; suc; inject₁; toℕ)
open import Data.Nat.Base using (_≤_; z≤n; s≤s)
open import Data.Product using (Σ; _,_)
open import Data.List using (List; []; _∷_; length; map; concatMap; upTo)
open import Data.Nat.ListAction using (sum)
open import Data.Maybe using (Maybe; just; nothing; maybe)
import Data.Maybe
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; cong; cong₂; sym; trans)

record Category : Set₁ where
  infixr 9 _⋆_
  field
    Ob    : Set
    Hom   : Ob → Ob → Set
    idn   : ∀ {a} → Hom a a
    _⋆_   : ∀ {a b c} → Hom a b → Hom b c → Hom a c   -- diagrammatic order
    idl   : ∀ {a b} (f : Hom a b) → idn ⋆ f ≡ f
    idr   : ∀ {a b} (f : Hom a b) → f ⋆ idn ≡ f
    assoc : ∀ {a b c d} (f : Hom a b) (g : Hom b c) (h : Hom c d)
          → (f ⋆ g) ⋆ h ≡ f ⋆ (g ⋆ h)

module Nerve (C : Category) where
  open Category C

  -- strings x = x₀ → x₁ → ... → xₙ of n composable arrows starting at x
  data Str : Ob → ℕ → Set where
    nil  : (x : Ob) → Str x 0
    cons : ∀ {x y n} → Hom x y → Str y n → Str x (suc n)

  -- the n-simplices of the nerve
  N : ℕ → Set
  N n = Σ Ob λ x → Str x n

  -- faceS k removes vertex k + 1 (composing there, or dropping the end)
  faceS : ∀ {x n} → Fin (suc n) → Str x (suc n) → Str x n
  faceS {x} zero (cons f (nil _)) = nil x
  faceS zero (cons f (cons g s)) = cons (f ⋆ g) s
  faceS {n = suc n} (suc k) (cons f s) = cons f (faceS k s)

  -- degS k repeats vertex k (inserts an identity there)
  degS : ∀ {x n} → Fin (suc n) → Str x n → Str x (suc n)
  degS zero s = cons idn s
  degS (suc k) (cons f s) = cons f (degS k s)

  -- [nerve#2] the face and degeneracy maps of N(C)
  d : ∀ {n} → Fin (suc (suc n)) → N (suc n) → N n
  d zero (x , cons f s) = (_ , s)
  d (suc k) (x , s) = (x , faceS k s)

  s : ∀ {n} → Fin (suc n) → N n → N (suc n)
  s k (x , σ) = (x , degS k σ)

  -- the inner face of a 2-simplex is the composite
  inner-face : ∀ {x y z} (f : Hom x y) (g : Hom y z)
             → d (suc zero) (x , cons f (cons g (nil z))) ≡ (x , cons (f ⋆ g) (nil z))
  inner-face f g = refl

  ----------------------------------------------------------------------
  -- lemmas on strings

  faceface : ∀ {x n} (i j : Fin (suc n)) → toℕ i ≤ toℕ j → (σ : Str x (suc (suc n)))
           → faceS i (faceS (suc j) σ) ≡ faceS j (faceS (inject₁ i) σ)
  faceface zero zero _ (cons f (cons g (nil z))) = refl
  faceface zero zero _ (cons f (cons g (cons h t))) = cong (λ k → cons k t) (sym (assoc f g h))
  faceface zero (suc j) _ (cons f (cons g (cons h t))) = refl
  faceface {n = suc n} (suc i) (suc j) (s≤s p) (cons f t) = cong (cons f) (faceface i j p t)

  degdeg : ∀ {x n} (i j : Fin (suc n)) → toℕ i ≤ toℕ j → (σ : Str x n)
         → degS (inject₁ i) (degS j σ) ≡ degS (suc j) (degS i σ)
  degdeg zero j _ σ = refl
  degdeg {n = suc n} (suc i) (suc j) (s≤s p) (cons f t) = cong (cons f) (degdeg i j p t)

  face-deg-same : ∀ {x n} (j : Fin (suc n)) (σ : Str x n) → faceS j (degS j σ) ≡ σ
  face-deg-same zero (nil x) = refl
  face-deg-same zero (cons g t) = cong (λ k → cons k t) (idl g)
  face-deg-same {n = suc n} (suc j) (cons f t) = cong (cons f) (face-deg-same j t)

  face-deg-next : ∀ {x n} (j : Fin (suc n)) (σ : Str x (suc n))
                → faceS (inject₁ j) (degS (suc j) σ) ≡ σ
  face-deg-next zero (cons f t) = cong (λ k → cons k t) (idr f)
  face-deg-next {n = suc n} (suc j) (cons f t) = cong (cons f) (face-deg-next j t)

  face-deg-below : ∀ {x n} (i : Fin (suc n)) (j : Fin n) → toℕ i ≤ toℕ j → (σ : Str x (suc n))
                 → faceS (inject₁ i) (degS (suc (suc j)) σ) ≡ degS (suc j) (faceS i σ)
  face-deg-below zero j _ (cons f (cons g t)) = refl
  face-deg-below {n = suc n} (suc i) (suc j) (s≤s p) (cons f t) = cong (cons f) (face-deg-below i j p t)

  face-deg-above : ∀ {x n} (i j : Fin (suc n)) → toℕ j ≤ toℕ i → (σ : Str x (suc n))
                 → faceS (suc i) (degS (inject₁ j) σ) ≡ degS j (faceS i σ)
  face-deg-above i zero _ σ = refl
  face-deg-above {n = suc n} (suc i) (suc j) (s≤s p) (cons f t) = cong (cons f) (face-deg-above i j p t)

  ----------------------------------------------------------------------
  -- [simplicial set#2] the simplicial identities for N(C)

  dd : ∀ {n} (i j : Fin (suc (suc n))) → toℕ i ≤ toℕ j → (σ : N (suc (suc n)))
     → d i (d (suc j) σ) ≡ d j (d (inject₁ i) σ)
  dd zero zero _ (x , cons f (cons g t)) = refl
  dd zero (suc j) _ (x , cons f (cons g t)) = refl
  dd (suc i) (suc j) (s≤s p) (x , σ) = cong (x ,_) (faceface i j p σ)

  ss : ∀ {n} (i j : Fin (suc n)) → toℕ i ≤ toℕ j → (σ : N n)
     → s (inject₁ i) (s j σ) ≡ s (suc j) (s i σ)
  ss i j p (x , σ) = cong (x ,_) (degdeg i j p σ)

  ds-same : ∀ {n} (j : Fin (suc n)) (σ : N n) → d (inject₁ j) (s j σ) ≡ σ
  ds-same zero (x , σ) = refl
  ds-same {n = suc n} (suc j) (x , σ) = cong (x ,_) (face-deg-next j σ)

  ds-next : ∀ {n} (j : Fin (suc n)) (σ : N n) → d (suc j) (s j σ) ≡ σ
  ds-next j (x , σ) = cong (x ,_) (face-deg-same j σ)

  -- d_i s_j = s_{j-1} d_i  for i < j  (here j = suc j')
  ds-below : ∀ {n} (i : Fin (suc (suc n))) (j : Fin (suc n)) → toℕ i ≤ toℕ j → (σ : N (suc n))
           → d (inject₁ i) (s (suc j) σ) ≡ s j (d i σ)
  ds-below zero j _ (x , cons f t) = refl
  ds-below {n = suc n} (suc i) (suc j) (s≤s p) (x , σ) = cong (x ,_) (face-deg-below i j p σ)

  -- d_i s_j = s_j d_{i-1}  for i > j + 1  (here i = suc (suc i'))
  ds-above : ∀ {n} (i j : Fin (suc n)) → toℕ j ≤ toℕ i → (σ : N (suc n))
           → d (suc (suc i)) (s (inject₁ j) σ) ≡ s j (d (suc i) σ)
  ds-above i j p (x , σ) = cong (x ,_) (face-deg-above i j p σ)

------------------------------------------------------------------------
-- Instances: the ordinals [k] as categories (thin, arrows i → j for i ≤ j)

data _≤ℕ_ : ℕ → ℕ → Set where
  z≤ : ∀ {n} → zero ≤ℕ n
  s≤ : ∀ {m n} → m ≤ℕ n → suc m ≤ℕ suc n

≤-refl : ∀ {n} → n ≤ℕ n
≤-refl {zero} = z≤
≤-refl {suc n} = s≤ ≤-refl

≤-trans : ∀ {a b c} → a ≤ℕ b → b ≤ℕ c → a ≤ℕ c
≤-trans z≤ _ = z≤
≤-trans (s≤ p) (s≤ q) = s≤ (≤-trans p q)

≤-prop : ∀ {m n} (p q : m ≤ℕ n) → p ≡ q
≤-prop z≤ z≤ = refl
≤-prop (s≤ p) (s≤ q) = cong s≤ (≤-prop p q)

record Bounded (k : ℕ) : Set where
  constructor ⟨_,_⟩
  field
    val : ℕ
    bnd : val ≤ℕ k

Ord : ℕ → Category
Category.Ob (Ord k) = Bounded k
Category.Hom (Ord k) a b = Bounded.val a ≤ℕ Bounded.val b
Category.idn (Ord k) = ≤-refl
Category._⋆_ (Ord k) = ≤-trans
Category.idl (Ord k) f = ≤-prop _ _
Category.idr (Ord k) f = ≤-prop _ _
Category.assoc (Ord k) f g h = ≤-prop _ _

-- enumerate the strings of N([k])_n
allPoints : (k : ℕ) → List (Bounded k)
allPoints zero = ⟨ zero , z≤ ⟩ ∷ []
allPoints (suc k) = ⟨ zero , z≤ ⟩ ∷ map shift (allPoints k)
  where
  shift : Bounded k → Bounded (suc k)
  shift ⟨ v , p ⟩ = ⟨ suc v , s≤ p ⟩

dec≤ : (a b : ℕ) → Maybe (a ≤ℕ b)
dec≤ zero b = just z≤
dec≤ (suc a) zero = nothing
dec≤ (suc a) (suc b) = Data.Maybe.map s≤ (dec≤ a b)

module Enumerate (k : ℕ) where
  open Nerve (Ord k)

  stringsFrom : (x : Bounded k) (n : ℕ) → List (Str x n)
  stringsFrom x zero = nil x ∷ []
  stringsFrom x (suc n) = concatMap ext (allPoints k)
    where
    ext : Bounded k → List (Str x (suc n))
    ext y = maybe (λ f → map (cons f) (stringsFrom y n)) []
                  (dec≤ (Bounded.val x) (Bounded.val y))

  count : ℕ → ℕ
  count n = sum (map (λ x → length (stringsFrom x n)) (allPoints k))

choose : ℕ → ℕ → ℕ
choose _ zero = 1
choose zero (suc k) = 0
choose (suc n) (suc k) = choose n k + choose n (suc k)

-- [nerve#1] |N([1])_n| = n + 2 for n = 0..5
nerve-of-[1] : map (Enumerate.count 1) (upTo 6) ≡ 2 ∷ 3 ∷ 4 ∷ 5 ∷ 6 ∷ 7 ∷ []
nerve-of-[1] = refl

-- [simplicial set#1] |Delta[n]_m| = |N([n])_m| for n, m = 0..3 (rows n)
Delta-table : map (λ n → map (Enumerate.count n) (upTo 4)) (upTo 4)
            ≡ (1 ∷ 1 ∷ 1 ∷ 1 ∷ [])
            ∷ (2 ∷ 3 ∷ 4 ∷ 5 ∷ [])
            ∷ (3 ∷ 6 ∷ 10 ∷ 15 ∷ [])
            ∷ (4 ∷ 10 ∷ 20 ∷ 35 ∷ []) ∷ []
Delta-table = refl

-- ... and the closed form binomial(n + m + 1, m + 1)
Delta-binomial : map (λ n → map (Enumerate.count n) (upTo 4)) (upTo 4)
               ≡ map (λ n → map (λ m → choose (n + m + 1) (m + 1)) (upTo 4)) (upTo 4)
Delta-binomial = refl
