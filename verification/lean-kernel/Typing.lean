import Init

namespace Verificatum.Typing

-- Names have already been resolved to constants in one environment.
inductive Term where
  | var (index : Nat)
  | sort (level : Nat)
  | con (name : Nat)
  | pi (domain body : Term)
  | lam (domain body : Term)
  | app (fn arg : Term)
deriving DecidableEq, Repr
open Term

def shift (byN cutoff : Nat) : Term → Term
  | var i => var (if cutoff ≤ i then i + byN else i)
  | pi a b => pi (shift byN cutoff a) (shift byN (cutoff+1) b)
  | lam a b => lam (shift byN cutoff a) (shift byN (cutoff+1) b)
  | app f x => app (shift byN cutoff f) (shift byN cutoff x)
  | t => t

def inst (arg : Term) (depth : Nat) : Term → Term
  | var i => if i = depth then shift depth 0 arg else var (if depth < i then i-1 else i)
  | pi a b => pi (inst arg depth a) (inst arg (depth+1) b)
  | lam a b => lam (inst arg depth a) (inst arg (depth+1) b)
  | app f x => app (inst arg depth f) (inst arg depth x)
  | t => t

structure Decl where
  type : Term
  value : Option Term

abbrev Env := Nat → Option Decl

-- Declarative, symmetric beta/delta conversion with congruence. Delta assumes
-- the supplied environment's transparent definitions are closed and checked.
inductive Conv (e : Env) : Term → Term → Prop where
  | refl (x) : Conv e x x
  | sym : Conv e x y → Conv e y x
  | trans : Conv e x y → Conv e y z → Conv e x z
  | beta : Conv e (app (lam a b) x) (inst x 0 b)
  | delta : e n = some ⟨a, some v⟩ → Conv e (con n) v
  | piCong : Conv e a c → Conv e b d → Conv e (pi a b) (pi c d)
  | lamCong : Conv e a c → Conv e b d → Conv e (lam a b) (lam c d)
  | appCong : Conv e f g → Conv e x y → Conv e (app f x) (app g y)

def piLevel (u v : Nat) := if v = 0 then 0 else max u v

inductive HasType (e : Env) : List Term → Term → Term → Prop where
  | variable : ctx[i]? = some a → HasType e ctx (var i) (shift (i+1) 0 a)
  | universe : HasType e ctx (sort u) (sort (u+1))
  | constant : e n = some d → HasType e ctx (con n) d.type
  | product : HasType e ctx a (sort u) → HasType e (a :: ctx) b (sort v) →
      HasType e ctx (pi a b) (sort (piLevel u v))
  | lambda : HasType e ctx a (sort u) → HasType e (a :: ctx) b t →
      HasType e ctx (lam a b) (pi a t)
  | apply : HasType e ctx f (pi a b) → HasType e ctx x a →
      HasType e ctx (app f x) (inst x 0 b)
  | convert : HasType e ctx x a → Conv e a b → HasType e ctx x b

-- The normalizer returns evidence for every reduction it actually used.
def normal (e : Env) : (fuel : Nat) → (x : Term) → Option {y // Conv e x y}
  | 0, _ => none
  | k+1, x => match x with
    | con n => match h : e n with
      | some ⟨_,some v⟩ => do
        let ⟨r, hr⟩ ← normal e k v
        pure ⟨r, .trans (.delta h) hr⟩
      | _ => some ⟨_, .refl _⟩
    | pi a b => do
      let ⟨_,hc⟩ ← normal e k a
      let ⟨_,hd⟩ ← normal e k b
      pure ⟨_, .piCong hc hd⟩
    | lam a b => do
      let ⟨_,hc⟩ ← normal e k a
      let ⟨_,hd⟩ ← normal e k b
      pure ⟨_, .lamCong hc hd⟩
    | app f x => do
      let ⟨g,hg⟩ ← normal e k f
      let ⟨y,hy⟩ ← normal e k x
      match g, hg with
      | lam _ b, hh =>
        let ⟨_,hv⟩ ← normal e k (inst y 0 b)
        pure ⟨_, .trans (.appCong hh hy) (.trans .beta hv)⟩
      | _, hh => pure ⟨_, .appCong hh hy⟩
    | var _ | sort _ => some ⟨_, .refl _⟩

def conversion (e : Env) (k : Nat) (x y : Term) : Option (PLift (Conv e x y)) := do
  let ⟨a,ha⟩ ← normal e k x
  let ⟨b,hb⟩ ← normal e k y
  if h : a = b then pure ⟨.trans (h ▸ ha) (.sym hb)⟩ else none

def infer (e : Env) : (fuel : Nat) → (ctx : List Term) → (x : Term) →
    Option {t // HasType e ctx x t}
  | 0, _, _ => none
  | k+1, ctx, x => match x with
    | var i => match h : ctx[i]? with
      | some _ => some ⟨_, .variable h⟩
      | none => none
    | sort _ => some ⟨_, .universe⟩
    | con n => match h : e n with
      | some _ => some ⟨_, .constant h⟩
      | none => none
    | pi a b => do
      let ⟨t,ht⟩ ← infer e k ctx a
      let ⟨sort _,hu⟩ ← normal e k t | none
      let ⟨s,hs⟩ ← infer e k (a::ctx) b
      let ⟨sort _,hv⟩ ← normal e k s | none
      pure ⟨_, .product (.convert ht hu) (.convert hs hv)⟩
    | lam a b => do
      let ⟨t,ht⟩ ← infer e k ctx a
      let ⟨sort _,hu⟩ ← normal e k t | none
      let ⟨_,hv⟩ ← infer e k (a::ctx) b
      pure ⟨_, .lambda (.convert ht hu) hv⟩
    | app f x => do
      let ⟨ft,hf⟩ ← infer e k ctx f
      let ⟨pi a _,ha⟩ ← normal e k ft | none
      let ⟨xt,hx⟩ ← infer e k ctx x
      let ⟨h⟩ ← conversion e k xt a
      pure ⟨_, .apply (.convert hf ha) (.convert hx h)⟩

def certify (e : Env) (k : Nat) (x t : Term) : Option (PLift (HasType e [] x t)) := do
  let ⟨s,hs⟩ ← infer e k [] x
  let ⟨h⟩ ← conversion e k s t
  pure ⟨.convert hs h⟩

def check (e : Env) (k : Nat) (x t : Term) := (certify e k x t).isSome

theorem check_sound (h : check e k x t = true) : HasType e [] x t := by
  unfold check at h
  cases result : certify e k x t with
  | none => simp [result] at h
  | some proof => exact proof.down

-- Import boundary: 0 is the independently validated empty inductive False.
-- Recursor use is outside this model. User declarations arrive in order,
-- resolved to common IDs; opacity controls unfolding of an admitted theorem.
structure InputDecl where
  name : Nat
  type : Term
  body : Term
  isTheorem : Bool

def base : Env := fun n => if n = 0 then some ⟨sort 0, none⟩ else none
def extend (e : Env) (d : InputDecl) : Env := fun n =>
  if n = d.name then some ⟨d.type, if d.isTheorem then none else some d.body⟩ else e n

inductive Valid : Env → Prop where
  | baseValid : Valid base
  | add : Valid e → e d.name = none → HasType e [] d.type (sort u) →
      (d.isTheorem = true → u = 0) → HasType e [] d.body d.type → Valid (extend e d)

def build (k : Nat) : List InputDecl → Option {e // Valid e}
  | [] => some ⟨base, .baseValid⟩
  | d::earlier => do
    let ⟨e, he⟩ ← build k earlier
    match h : e d.name with
    | some _ => none
    | none =>
      let ⟨t, ht⟩ ← infer e k [] d.type
      let ⟨sort u, hu⟩ ← normal e k t | none
      if hp : d.isTheorem = true → u = 0 then
        let ⟨hb⟩ ← certify e k d.body d.type
        pure ⟨extend e d, .add he h (.convert ht hu) hp hb⟩
      else none

-- Bind an independent, already expanded statement. Its only permitted free
-- constant is the pinned False; a local definition named Not cannot change it.
def portable (depth : Nat) : Term → Bool
  | var i => i < depth
  | con n => n == 0
  | sort _ => true
  | pi a b | lam a b => portable depth a && portable (depth+1) b
  | app f x => portable depth f && portable depth x

def checkExport (k : Nat) (earlier : List InputDecl) (d : InputDecl) (expected : Term) : Bool :=
  match build k earlier with
  | none => false
  | some ⟨e,_⟩ =>
    d.isTheorem && (e d.name).isNone && portable 0 expected &&
    check e k expected (sort 0) && check e k d.type (sort 0) &&
    check e k d.body d.type && (conversion e k d.type expected).isSome

theorem target_binding_sound (h : checkExport k earlier d expected = true) :
    ∃ e, Valid e ∧ (build k earlier).map Subtype.val = some e ∧
      portable 0 expected = true ∧ HasType e [] expected (sort 0) ∧
      HasType e [] d.body expected := by
  unfold checkExport at h
  cases hb : build k earlier with
  | none => simp [hb] at h
  | some env =>
    simp only [hb, Bool.and_eq_true] at h
    obtain ⟨⟨⟨⟨⟨⟨_,_⟩,hp⟩,ht⟩,_⟩,hc⟩,heq⟩ := h
    refine ⟨env.val, env.property, rfl, hp, check_sound ht, ?_⟩
    cases cv : conversion env.val k d.type expected with
    | none => simp [cv] at heq
    | some proof => exact .convert (check_sound hc) proof.down

private def mtType : Term :=
  pi (sort 0) (pi (sort 0) (pi (pi (var 1) (var 1))
    (pi (pi (var 1) (con 0)) (pi (var 3) (con 0)))))
private def mtBody (fn : Nat) : Term :=
  lam (sort 0) (lam (sort 0) (lam (pi (var 1) (var 1))
    (lam (pi (var 1) (con 0)) (lam (var 3) (app (var 1) (app (var fn) (var 0)))))))
example : checkExport 32 [] ⟨1, mtType, mtBody 2, true⟩ mtType = true := by decide
example : checkExport 32 [] ⟨1, mtType, mtBody 1, true⟩ mtType = false := by decide
example : checkExport 32 [] ⟨1, mtType, mtBody 2, true⟩ (pi (sort 0) (pi (var 0) (var 1))) = false := by decide

private def emptyEnv : Env := fun _ => none
example : check emptyEnv 16 (lam (sort 0) (lam (var 0) (var 0)))
    (pi (sort 0) (pi (var 0) (var 1))) = true := by decide
example : check emptyEnv 16 (lam (sort 0) (var 0))
    (pi (sort 0) (pi (var 0) (var 1))) = false := by decide

end Verificatum.Typing
