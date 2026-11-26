{-# OPTIONS --cubical --safe #-}
-- Independent check for reject/split_env_captured_variable.esk: f1 and f2 are distinct.
module VarEnvDistinct where

open import Agda.Primitive.Cubical renaming (primTransp to transp)
open import Agda.Builtin.Cubical.Path

data Bool : Set where
  true false : Bool
data Empty : Set where
record Unit : Set where
  constructor tt

c : Bool → Bool → Bool
c g true  = g
c g false = g

f1 f2 : Bool → Bool → Bool → Bool
f1 a b = c a
f2 a b = c b

Code : Bool → Set
Code true  = Unit
Code false = Empty

distinct : f1 ≡ f2 → Empty
distinct q = transp (λ i → Code (q i true false true)) i0 tt
