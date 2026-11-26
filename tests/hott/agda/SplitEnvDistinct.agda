{-# OPTIONS --cubical --safe #-}
-- Independent check for reject/split_env_captured_split.esk: the two functions the
-- Eshkol kernel identifies are distinct, and a path between them yields an
-- element of the empty type. Builtin cubical primitives only, as in
-- tests/hott/agda/Pi1Circle.agda.
module SplitEnvDistinct where

open import Agda.Primitive.Cubical renaming (primTransp to transp)
open import Agda.Builtin.Cubical.Path

data Bool : Set where
  true false : Bool

data Empty : Set where

record Unit : Set where
  constructor tt

neg : Bool → Bool
neg true  = false
neg false = true

ident : Bool → Bool
ident true  = true
ident false = false

k : (Bool → Bool) → Bool → Bool
k g true  = g true
k g false = g false

Code : Bool → Set
Code true  = Empty
Code false = Unit

_ : k neg true ≡ false
_ = λ _ → false

_ : k ident true ≡ true
_ = λ _ → true

-- Any path k neg ≡ k ident gives an element of Empty.
distinct : k neg ≡ k ident → Empty
distinct q = transp (λ i → Code (q i true)) i0 tt
