import Init

namespace Verificatum.Lazy

-- Zero-based variables internally. Free indices are relative to the root.
inductive Term where
  | var (i : Nat)
  | lam (body : Term)
  | app (fn arg : Term)
deriving DecidableEq, Repr

abbrev Ren := Nat → Nat
abbrev Sub := Nat → Term

def push {α} (a : α) (s : Nat → α) : Nat → α
  | 0 => a
  | i+1 => s i

def upRen (r : Ren) := push 0 (fun i => r i + 1)
def ren (r : Ren) : Term → Term
  | .var i => .var (r i)
  | .lam b => .lam (ren (upRen r) b)
  | .app f a => .app (ren r f) (ren r a)
def upSub (s : Sub) := push (.var 0) (fun i => ren Nat.succ (s i))
def sub (s : Sub) : Term → Term
  | .var i => s i
  | .lam b => .lam (sub (upSub s) b)
  | .app f a => .app (sub s f) (sub s a)

theorem renRen (t : Term) (r s : Ren) :
    ren r (ren s t) = ren (fun i => r (s i)) t := by
  induction t generalizing r s with
  | var => rfl
  | app _ _ hf ha => simp [ren, hf, ha]
  | lam _ ih =>
    simp only [ren, ih, Term.lam.injEq]
    congr 1
    funext i; cases i <;> rfl

private theorem renSub (t : Term) (r : Ren) (s : Sub) :
    ren r (sub s t) = sub (fun i => ren r (s i)) t := by
  induction t generalizing r s with
  | var => rfl
  | app _ _ hf ha => simp [sub, ren, hf, ha]
  | lam _ ih =>
    simp only [sub, ren, ih, Term.lam.injEq]
    congr 1
    funext i; cases i with
    | zero => rfl
    | succ i => simp only [upSub, push, renRen]; rfl

private theorem subRen (t : Term) (s : Sub) (r : Ren) :
    sub s (ren r t) = sub (fun i => s (r i)) t := by
  induction t generalizing r s with
  | var => rfl
  | app _ _ hf ha => simp [sub, ren, hf, ha]
  | lam _ ih =>
    simp only [ren, sub, ih, Term.lam.injEq]
    congr 1
    funext i; cases i <;> rfl

private theorem subSub (t : Term) (r s : Sub) :
    sub r (sub s t) = sub (fun i => sub r (s i)) t := by
  induction t generalizing r s with
  | var => rfl
  | app _ _ hf ha => simp [sub, hf, ha]
  | lam _ ih =>
    simp only [sub, ih, Term.lam.injEq]
    congr 1
    funext i; cases i with
    | zero => rfl
    | succ i => simp only [upSub, push, subRen, renSub]

private theorem idSub (t : Term) : sub Term.var t = t := by
  induction t with
  | var => rfl
  | app _ _ hf ha => simp [sub, hf, ha]
  | lam _ ih =>
    have h : upSub Term.var = Term.var := by funext i; cases i <;> rfl
    simp [sub, h, ih]

def beta (b a : Term) := sub (push a Term.var) b

theorem enter_binder (b : Term) (s : Sub) (a : Term) :
    beta (sub (upSub s) b) a = sub (push a s) b := by
  unfold beta
  rw [subSub]
  congr 1
  funext i; cases i with
  | zero => rfl
  | succ i => change sub (push a Term.var) (ren Nat.succ (s i)) = s i
              rw [subRen]; exact idSub _

-- A persistent environment is queried only at a demanded variable. Wrapping
-- a computation in the implementation's mutable WHNF update cells requires
-- an additional heap refinement.
inductive Head where
  | free (i : Nat)
  | bound (level : Nat)
inductive Closure where
  | atom (h : Head)
  | delay (code : Term) (env : Nat → Closure)

def index (d : Nat) : Head → Nat
  | .free i => d+i
  | .bound k => d-1-k

def Valid (d : Nat) : Closure → Prop
  | .atom (.bound k) => k < d
  | .atom (.free _) => True
  | .delay _ e => ∀ i, Valid d (e i)

def erase (d : Nat) : Closure → Term
  | .atom h => .var (index d h)
  | .delay t e => sub (fun i => erase d (e i)) t

private theorem validSucc (d : Nat) (c : Closure) (h : Valid d c) : Valid (d+1) c := by
  induction c with
  | atom a => cases a <;> simp_all [Valid] <;> omega
  | delay _ _ ih => exact fun i => ih i (h i)

private theorem weaken (c : Closure) (d : Nat) (h : Valid d c) :
    erase (d+1) c = ren Nat.succ (erase d c) := by
  induction c with
  | atom a => cases a <;> simp_all [erase, index, ren, Valid] <;> omega
  | delay t e ih =>
    simp only [erase, renSub]
    congr 1; funext i; exact ih i (h i)

inductive View (α : Type) where
  | var (i : Nat)
  | app (fn arg : α)
  | lam (under : Unit → α) (enter : α → α)

def sourceView (_d : Nat) : Term → View Term
  | .var i => .var i
  | .app f a => .app f a
  | .lam b => .lam (fun _ => b) (beta b)

def closureView (d : Nat) : Closure → View Closure
  | .atom h => .var (index d h)
  | .delay (.var i) e => closureView d (e i)
  | .delay (.app f a) e => .app (.delay f e) (.delay a e)
  | .delay (.lam b) e =>
    .lam (fun _ => .delay b (push (.atom (.bound d)) e))
         (fun a => .delay b (push a e))

-- Argument stack is ordered next argument first. Backward frames eliminate
-- a complete neutral spine from left to right and read back under lambdas.
inductive Frame (α : Type) where
  | lam (d : Nat)
  | args (d : Nat) (head : Term) (more : List α)
inductive State (α : Type) where
  | seek (d : Nat) (fn : α) (args : List α) (ks : List (Frame α))
  | emit (nf : Term) (ks : List (Frame α))

def spine (d : Nat) (head : Term) (args : List α) (ks : List (Frame α)) : State α :=
  match args with
  | [] => .emit head ks
  | a :: as => .seek d a [] (.args d head as :: ks)

def tick (view : Nat → α → View α) : State α → Sum Term (State α)
  | .seek d f as ks =>
    match view d f with
    | .var i => .inr (spine d (.var i) as ks)
    | .app g x => .inr (.seek d g (x :: as) ks)
    | .lam under enter =>
      match as with
      | [] => .inr (.seek (d+1) (under ()) [] (.lam d :: ks))
      | a :: rest => .inr (.seek d (enter a) rest ks)
  | .emit n [] => .inl n
  | .emit n (.lam _ :: ks) => .inr (.emit (.lam n) ks)
  | .emit n (.args d h as :: ks) => .inr (spine d (.app h n) as ks)

def run (view : Nat → α → View α) : Nat → State α → Option Term
  | 0, _ => none
  | n+1, s => match tick view s with
    | .inl t => some t
    | .inr s' => run view n s'

def Related (d : Nat) (c : Closure) (t : Term) := Valid d c ∧ erase d c = t

inductive Views (d : Nat) : View Closure → View Term → Prop where
  | var : Views d (.var i) (.var i)
  | app : Related d f f' → Related d a a' → Views d (.app f a) (.app f' a')
  | lam : Related (d+1) (u ()) (u' ()) →
      (∀ a t, Related d a t → Related d (e a) (e' t)) →
      Views d (.lam u e) (.lam u' e')

theorem observe (c : Closure) (d : Nat) (h : Valid d c) :
    Views d (closureView d c) (sourceView d (erase d c)) := by
  induction c with
  | atom => exact .var
  | delay code env ih =>
    cases code with
    | var i => exact ih i (h i)
    | app f a => exact .app ⟨h, rfl⟩ ⟨h, rfl⟩
    | lam b =>
      apply Views.lam
      · constructor
        · intro i; cases i with
          | zero => exact Nat.lt_succ_self _
          | succ i => exact validSucc _ _ (h i)
        · change sub _ b = sub _ b
          congr 1; funext i; cases i with
          | zero => simp [push, upSub, erase, index]
          | succ i => exact weaken _ _ (h i)
      · intro a t ha
        constructor
        · intro i; cases i with
          | zero => exact ha.1
          | succ i => exact h i
        · change sub _ b = beta (sub _ b) t
          rw [enter_binder]
          congr 1; funext i; cases i with
          | zero => exact ha.2
          | succ i => rfl

inductive Args (d : Nat) : List Closure → List Term → Prop where
  | nil : Args d [] []
  | cons : Related d a b → Args d as bs → Args d (a :: as) (b :: bs)
inductive Frames : List (Frame Closure) → List (Frame Term) → Prop where
  | nil : Frames [] []
  | lam : Frames ks ks' → Frames (.lam d :: ks) (.lam d :: ks')
  | args : Args d as as' → Frames ks ks' → Frames (.args d h as :: ks) (.args d h as' :: ks')
inductive States : State Closure → State Term → Prop where
  | seek : Related d f f' → Args d as as' → Frames ks ks' →
      States (.seek d f as ks) (.seek d f' as' ks')
  | emit : Frames ks ks' → States (.emit n ks) (.emit n ks')

private theorem spine_rel (h : Args d as as') (k : Frames ks ks') (n : Term) :
    States (spine d n as ks) (spine d n as' ks') := by
  cases h with
  | nil => exact .emit k
  | cons x xs => exact .seek x .nil (.args xs k)

private inductive Outputs : Sum Term (State Closure) → Sum Term (State Term) → Prop where
  | done : Outputs (.inl n) (.inl n)
  | step : States s t → Outputs (.inr s) (.inr t)

private theorem tick_rel (pair : States s t) :
    Outputs (tick closureView s) (tick sourceView t) := by
  cases pair with
  | emit ks =>
    cases ks with
    | nil => exact .done
    | lam rest => exact .step (.emit rest)
    | args as rest => exact .step (spine_rel as rest _)
  | @seek d f f' as as' ks ks' fn args frames =>
    have view := observe f d fn.1
    rw [fn.2] at view
    simp only [tick]
    generalize closureView d f = cv at view ⊢
    generalize sourceView d f' = tv at view ⊢
    cases view with
    | var => exact .step (spine_rel args frames _)
    | app g x => exact .step (.seek g (.cons x args) frames)
    | lam u e =>
      cases args with
      | nil => exact .step (.seek u .nil (.lam frames))
      | cons a rest => exact .step (.seek (e _ _ a) rest frames)

theorem runs_agree (fuel : Nat) (pair : States s t) :
    run closureView fuel s = run sourceView fuel t := by
  induction fuel generalizing s t with
  | zero => rfl
  | succ n ih =>
    have h := tick_rel pair
    simp only [run]
    generalize tick closureView s = a at h ⊢
    generalize tick sourceView t = b at h ⊢
    cases h with
    | done => rfl
    | step ss => exact ih ss

def initial (t : Term) : Closure := .delay t (fun i => .atom (.free i))

theorem closure_correspondence (t : Term) (fuel : Nat) :
    run closureView fuel (.seek 0 (initial t) [] []) =
    run sourceView fuel (.seek 0 t [] []) := by
  apply runs_agree
  apply States.seek _ .nil .nil
  constructor
  · intro i; trivial
  · simpa [initial, erase, index] using idSub t


-- Independent specification: contract one leftmost-outermost redex. Under a
-- lambda recurse into its body; in a non-redex application try the left first.
def step : Term → Option Term
  | .var _ => none
  | .lam b => (step b).map Term.lam
  | .app f a => match f with
    | .lam b => some (beta b a)
    | _ => match step f with
      | some g => some (.app g a)
      | none => (step a).map (Term.app f)

def reference : Nat → Term → Option Term
  | 0, _ => none
  | k+1, t => match step t with
    | some u => reference k u
    | none => some t

def apps (h : Term) : List Term → Term
  | [] => h
  | a :: as => apps (.app h a) as

def plugFrames (t : Term) : List (Frame Term) → Term
  | [] => t
  | .lam _ :: ks => plugFrames (.lam t) ks
  | .args _ h as :: ks => plugFrames (apps (.app h t) as) ks

def plug : State Term → Term
  | .seek _ f as ks => plugFrames (apps f as) ks
  | .emit n ks => plugFrames n ks

def nonLam : Term → Prop
  | .lam _ => False
  | _ => True

private theorem stepApp (f a : Term) (h : nonLam f) :
    step (.app f a) = match step f with
      | some g => some (.app g a)
      | none => (step a).map (Term.app f) := by
  cases f <;> simp_all [nonLam, step]

def Neutral (t : Term) := nonLam t ∧ step t = none
private theorem neutralVar (i : Nat) : Neutral (.var i) := ⟨True.intro,rfl⟩
private theorem neutralApp (f a : Term) (hf : Neutral f) (ha : step a = none) :
    Neutral (.app f a) := by
  exact ⟨True.intro, by rw [stepApp _ _ hf.1, hf.2, ha]; rfl⟩

private theorem appsStep (t u : Term) (as : List Term) (hh : nonLam t)
    (h : step t = some u) : step (apps t as) = some (apps u as) := by
  induction as generalizing t u with
  | nil => exact h
  | cons a as ih =>
    apply ih (.app t a) (.app u a) True.intro
    rw [stepApp _ _ hh,h]

def SafeFrames : List (Frame Term) → Prop
  | [] => True
  | .lam _ :: ks => SafeFrames ks
  | .args _ h _ :: ks => Neutral h ∧ SafeFrames ks

def Safe : State Term → Prop
  | .seek _ _ _ ks => SafeFrames ks
  | .emit n ks => step n = none ∧ SafeFrames ks

private theorem plugStep (ks : List (Frame Term)) (safe : SafeFrames ks)
    (t u : Term) (h : step t = some u) :
    step (plugFrames t ks) = some (plugFrames u ks) := by
  induction ks generalizing t u with
  | nil => exact h
  | cons frame ks ih =>
    cases frame with
    | lam =>
      apply ih safe
      simp [step, h]
    | args d f as =>
      apply ih safe.2
      apply appsStep (.app f t) (.app f u) as True.intro
      rw [stepApp _ _ safe.1.1, safe.1.2, h]; rfl

-- Administrative transitions consume syntax or a pending readback frame.
-- This potential need only decrease between beta contractions.
def work : Term → Nat
  | .var _ => 2
  | .lam b => work b + 3
  | .app f a => work f + work a + 3

def argWork : List Term → Nat
  | [] => 0
  | a :: as => work a + 1 + argWork as

def frameWork : List (Frame Term) → Nat
  | [] => 0
  | .lam _ :: ks => 1 + frameWork ks
  | .args _ _ as :: ks => argWork as + 1 + frameWork ks

def weight : State Term → Nat
  | .seek _ f as ks => work f + argWork as + frameWork ks
  | .emit _ ks => frameWork ks

private inductive Progress (s : State Term) : Sum Term (State Term) → Prop where
  | done : plug s = n → step n = none → Progress s (.inl n)
  | admin : Safe t → plug s = plug t → weight t < weight s → Progress s (.inr t)
  | betaStep : Safe t → step (plug s) = some (plug t) → Progress s (.inr t)

private theorem progress (s : State Term) (h : Safe s) : Progress s (tick sourceView s) := by
  cases s with
  | seek d f as ks =>
    cases f with
    | var i =>
      cases as with
      | nil => exact .admin ⟨rfl,h⟩ rfl (by simp [spine,weight, work, argWork] <;> omega)
      | cons x xs =>
        exact .admin ⟨neutralVar _,h⟩ rfl
          (by simp [spine,weight, argWork, frameWork, work] <;> omega)
    | app f a => exact .admin h rfl (by simp [weight, argWork, work] <;> omega)
    | lam b =>
      cases as with
      | nil => exact .admin h rfl (by simp [weight, argWork, frameWork, work] <;> omega)
      | cons a as =>
        apply Progress.betaStep (t := .seek d (beta b a) as ks) h
        apply plugStep ks h
        exact appsStep (.app (.lam b) a) (beta b a) as True.intro rfl
  | emit n ks =>
    cases ks with
    | nil => exact .done rfl h.1
    | cons k ks =>
      cases k with
      | lam d => exact .admin ⟨by simp [step,h.1],h.2⟩ rfl
                   (by simp [weight,frameWork] <;> omega)
      | args d f as =>
        have hh := neutralApp f n h.2.1 h.1
        cases as with
        | nil => exact .admin ⟨hh.2,h.2.2⟩ rfl
                   (by simp [spine,weight,frameWork,argWork] <;> omega)
        | cons a as =>
          exact .admin ⟨hh,h.2.2⟩ rfl (by simp [spine,weight,frameWork,argWork] <;> omega)

private theorem complete (k : Nat) (s : State Term) (safe : Safe s)
    (h : reference k (plug s) = some n) :
    ∃ fuel, run sourceView fuel s = some n := by
  have p := progress s safe
  generalize ht : tick sourceView s = out at p
  cases p with
  | @done v hp hn =>
    cases k with
    | zero => simp [reference] at h
    | succ k =>
      simp [reference,hp,hn] at h
      exact ⟨1, by simp [run,ht,h]⟩
  | @admin t hs hp hw =>
    obtain ⟨fuel,hh⟩ := complete k t hs (hp ▸ h)
    exact ⟨fuel+1, by simp [run,ht,hh]⟩
  | @betaStep t hs hb =>
    cases k with
    | zero => simp [reference] at h
    | succ k =>
      simp [reference,hb] at h
      obtain ⟨fuel,hh⟩ := complete k t hs h
      exact ⟨fuel+1, by simp [run,ht,hh]⟩
termination_by (k, weight s)

private theorem sound (fuel : Nat) (s : State Term) (safe : Safe s)
    (h : run sourceView fuel s = some n) :
    ∃ k, reference k (plug s) = some n := by
  induction fuel generalizing s with
  | zero => simp [run] at h
  | succ fuel ih =>
    have p := progress s safe
    generalize ht : tick sourceView s = out at p
    cases p with
    | done hp hn =>
      simp [run,ht] at h
      subst n
      exact ⟨1, by simp [reference,hp,hn]⟩
    | admin hs hp hw =>
      simp [run,ht] at h
      exact hp ▸ ih _ hs h
    | betaStep hs hb =>
      simp [run,ht] at h
      obtain ⟨k,hh⟩ := ih _ hs h
      exact ⟨k+1, by simp [reference,hb,hh]⟩

-- Quantify resources separately: reference inspections and abstract-machine
-- ticks count different work. Both directions preserve the entire normal form.
theorem normal_order_correspondence (t n : Term) :
    (∃ k, reference k t = some n) ↔
    (∃ fuel, run closureView fuel (.seek 0 (initial t) [] []) = some n) := by
  constructor
  · rintro ⟨k,h⟩
    obtain ⟨fuel,hf⟩ := complete k (.seek 0 t [] []) True.intro h
    exact ⟨fuel, (closure_correspondence t fuel).trans hf⟩
  · rintro ⟨fuel,h⟩
    rw [closure_correspondence] at h
    exact sound fuel (.seek 0 t [] []) True.intro h

-- Weak-head evaluation only pushes arguments or enters a lambda. Store the
-- resulting head and any unconsumed local arguments. The key includes the
-- lexical environment. No readback depth or consumer continuation is captured.
inductive ApplyView : Closure → Closure → Closure → Prop where
  | direct : ApplyView (.delay (.app f a) e) (.delay f e) (.delay a e)
  | lookup : ApplyView (e i) f a → ApplyView (.delay (.var i) e) f a

inductive EnterView : Closure → Closure → Closure → Prop where
  | direct : EnterView (.delay (.lam b) e) a (.delay b (push a e))
  | lookup : EnterView (e i) a c → EnterView (.delay (.var i) e) a c

inductive HeadHop : Closure × List Closure → Closure × List Closure → Prop where
  | app : ApplyView f g x → HeadHop (f, as) (g, x :: as)
  | lam : EnterView f x c → HeadHop (f, x :: as) (c, as)

inductive Hops : Nat → Closure × List Closure → Closure × List Closure → Prop where
  | refl : Hops 0 x x
  | cons : HeadHop x y → Hops n y z → Hops (n+1) x z

private theorem applyView_eq (d : Nat) (h : ApplyView c f a) :
    closureView d c = .app f a := by
  induction h with
  | direct => rfl
  | lookup _ ih => exact ih
private theorem enterView_eq (d : Nat) (h : EnterView c a b) :
    ∃ u e, closureView d c = .lam u e ∧ e a = b := by
  induction h with
  | direct => exact ⟨_,_,rfl,rfl⟩
  | lookup _ ih => exact ih

def atHead (d : Nat) (outer : List Closure) (ks : List (Frame Closure))
    (x : Closure × List Closure) : State Closure := .seek d x.1 (x.2 ++ outer) ks

private theorem hop_context (h : HeadHop x y) :
    tick closureView (atHead d outer ks x) = .inr (atHead d outer ks y) := by
  cases h with
  | app ha => simp [atHead,tick,applyView_eq _ ha]
  | lam he =>
    obtain ⟨u,e,hh,ha⟩ := enterView_eq d he
    simp [atHead,tick,hh,ha]

theorem reuse_under_context (h : Hops n x y) (d : Nat) (outer : List Closure)
    (ks : List (Frame Closure)) (fuel : Nat) :
    run closureView (fuel+n) (atHead d outer ks x) =
    run closureView fuel (atHead d outer ks y) := by
  induction h with
  | refl => rfl
  | cons hop rest ih => simp [run,hop_context hop,ih]

structure Entry (c : Closure) where
  head : Closure
  args : List Closure
  hops : Nat
  path : Hops hops (c,[]) (head,args)

-- The store may change on every lookup. A store earns reuse by providing the
-- head-evaluation derivation for its key. This is an invariant of the abstract
-- cache, not a requirement to build proof terms at runtime.
abbrev Lookup (σ : Type) := σ → (c : Closure) → σ × Option (Entry c)

def shortcut (pick : Lookup σ) (store : σ) : State Closure → σ × Nat × State Closure
  | .seek d c as ks =>
    match pick store c with
    | (next,none) => (next,0,.seek d c as ks)
    | (next,some e) => (next,e.hops,.seek d e.head (e.args ++ as) ks)
  | .emit n ks => (store,0,.emit n ks)

private theorem shortcut_run (pick : Lookup σ) (store : σ) (s : State Closure)
    (h : shortcut pick store s = (next,k,v)) (fuel : Nat) :
    run closureView (fuel+k) s = run closureView fuel v := by
  cases s with
  | emit => cases h; rfl
  | seek d c as ks =>
    cases hh : pick store c with
    | mk st opt =>
      cases opt with
      | none => simp [shortcut,hh] at h; obtain ⟨_,_,_⟩ := h; subst_vars; rfl
      | some e =>
        simp [shortcut,hh] at h; obtain ⟨_,_,_⟩ := h; subst_vars
        exact reuse_under_context e.path d as ks fuel

def cached (pick : Lookup σ) : Nat → σ → State Closure → Option Term
  | 0, _, _ => none
  | fuel+1, store, s =>
    let (next,_,v) := shortcut pick store s
    match tick closureView v with
    | .inl n => some n
    | .inr s' => cached pick fuel next s'

-- An exact success length avoids treating a failed finite budget as a
-- divergence judgment. A finite head shortcut cannot cross a completed state.
private inductive Finishes : Nat → State Closure → Term → Prop where
  | done : tick closureView s = .inl n → Finishes 1 s n
  | step : tick closureView s = .inr t → Finishes k t n → Finishes (k+1) s n

private theorem finishes_of_run (fuel : Nat) (s : State Closure)
    (h : run closureView fuel s = some n) : ∃ k, Finishes k s n := by
  induction fuel generalizing s with
  | zero => simp [run] at h
  | succ k ih =>
    cases ht : tick closureView s with
    | inl v => simp [run,ht] at h; subst v; exact ⟨1,.done ht⟩
    | inr t =>
      obtain ⟨m,hm⟩ := ih t (by simpa [run,ht] using h)
      exact ⟨m+1,.step ht hm⟩

private theorem after_hops (h : Hops k x y) (d : Nat) (as : List Closure)
    (ks : List (Frame Closure)) (hf : Finishes m (atHead d as ks x) n) :
    ∃ j, m = j+k ∧ Finishes j (atHead d as ks y) n := by
  induction h generalizing m with
  | refl => exact ⟨m,rfl,hf⟩
  | cons hp tail ih =>
    have hstep := @hop_context _ _ d as ks hp
    cases hf with
    | done hd => simp [hstep] at hd
    | step ht hnext =>
      rw [hstep] at ht
      cases ht
      obtain ⟨j,he,hj⟩ := ih hnext
      exact ⟨j,by omega,hj⟩

private theorem shortcut_finishes (pick : Lookup σ) (store : σ) (s : State Closure)
    (h : shortcut pick store s = (next,k,v)) (hf : Finishes m s n) :
    ∃ j, m = j+k ∧ Finishes j v n := by
  cases s with
  | emit => cases h; exact ⟨m,rfl,hf⟩
  | seek d c as ks =>
    cases hh : pick store c with
    | mk st opt =>
      cases opt with
      | none =>
        simp [shortcut,hh] at h; obtain ⟨_,_,_⟩ := h; subst_vars
        exact ⟨m,rfl,hf⟩
      | some e =>
        simp [shortcut,hh] at h; obtain ⟨_,_,_⟩ := h; subst_vars
        exact after_hops e.path d as ks hf

private theorem cached_complete (pick : Lookup σ) (store : σ) (s : State Closure)
    (m : Nat) (hf : Finishes m s n) : ∃ fuel, cached pick fuel store s = some n := by
  generalize hs : shortcut pick store s = p
  obtain ⟨next,k,v⟩ := p
  obtain ⟨j,hm,hj⟩ := shortcut_finishes pick store s hs hf
  cases hj with
  | done ht => exact ⟨1, by simp [cached,hs,ht]⟩
  | @step _ t l _ ht hl =>
    obtain ⟨fuel,h⟩ := cached_complete pick next t l hl
    exact ⟨fuel+1, by simp [cached,hs,ht,h]⟩
termination_by m

private theorem cached_sound (pick : Lookup σ) (fuel : Nat) (store : σ) (s : State Closure)
    (h : cached pick fuel store s = some n) : ∃ k, run closureView k s = some n := by
  induction fuel generalizing store s with
  | zero => simp [cached] at h
  | succ fuel ih =>
    generalize hs : shortcut pick store s = p
    obtain ⟨next,k,v⟩ := p
    cases ht : tick closureView v with
    | inl t =>
      simp [cached,hs,ht] at h
      exact ⟨1+k, by rw [shortcut_run pick store s hs]; simp [run,ht,h]⟩
    | inr t =>
      obtain ⟨m,hm⟩ := ih next t (by simpa [cached,hs,ht] using h)
      exact ⟨(m+1)+k, by rw [shortcut_run pick store s hs]; simp [run,ht,hm]⟩

theorem cached_normal_order_correspondence (t n : Term) (pick : Lookup σ) (store : σ) :
    (∃ k, reference k t = some n) ↔
    (∃ fuel, cached pick fuel store (.seek 0 (initial t) [] []) = some n) := by
  rw [normal_order_correspondence]
  constructor
  · rintro ⟨k,h⟩
    obtain ⟨m,hm⟩ := finishes_of_run k _ h
    exact cached_complete pick store _ m hm
  · rintro ⟨fuel,h⟩
    exact cached_sound pick fuel store _ h

/-- info: 'Verificatum.Lazy.cached_normal_order_correspondence' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms cached_normal_order_correspondence

private def esk : Term → String
  | .var i => s!"(var {i+1})"
  | .lam b => s!"(lam {esk b})"
  | .app f a => s!"(app {esk f} {esk a})"

-- All trees of size ≤ 2 iterations of this grammar, with two ambient free
-- variables. Applications range independently over both subtrees.
private def terms (width : Nat) : Nat → List Term
  | 0 => (List.range width).map .var
  | h+1 =>
    let prev := terms width h
    (List.range width).map .var ++ (terms (width+1) h).map .lam ++
      prev.flatMap (fun f => prev.map (.app f ·))

private def oracleRow (t : Term) : IO Unit := do
  let some expected := reference 2000 t | throw (IO.userError "reference budget")
  let some actual := run closureView 10000 (.seek 0 (initial t) [] []) |
    throw (IO.userError "abstract closure budget")
  unless actual = expected do throw (IO.userError "model disagreement")
  IO.println s!"{esk t} {esk actual}"

def emitCorpus : IO Unit := do
  let id := Term.lam (.var 0)
  let dup := Term.lam (.app (.var 0) (.var 0))
  let omega := Term.app dup dup
  let deep := (List.range 64).foldl (fun t _ => Term.lam t) (Term.var 64)
  for t in terms 2 2 do oracleRow t
  for b in terms 3 1 do
    for a in terms 2 1 do oracleRow (.app (.lam b) a)
  let underOmega := Term.app (.lam (.lam (.var 1))) omega
  unless (reference 32 underOmega).isNone &&
         (run closureView 64 (.seek 0 (initial underOmega) [] [])).isNone do
    throw (IO.userError "expected budget exhaustion")
  for t in [Term.app (.app (.lam (.lam (.var 1))) id) omega,
            .app (.lam (.app (.var 0) deep)) (.app id (.var 1)),
            .app dup (.app id (.var 4))] do oracleRow t

-- Independent weak-head oracle for actual heap events. Stop at a lambda or
-- variable-headed spine. Arguments and lambda bodies are kept verbatim.
def weakStep : Term → Option Term
  | .app f x => match f with
    | .lam b => some (beta b x)
    | _ => (weakStep f).map (Term.app · x)
  | _ => none

def weakRun : Nat → Term → Option Term
  | 0, _ => none
  | n+1, t => match weakStep t with
    | none => some t
    | some u => weakRun n u

def parse (fuel : Nat) (s : List Char) : Option (Term × List Char) := do
  match fuel, s with
  | k+1, '0' :: '0' :: rest =>
    let (b,tail) ← parse k rest
    return (.lam b,tail)
  | k+1, '0' :: '1' :: rest =>
    let (f,mid) ← parse k rest
    let (a,tail) ← parse k mid
    return (.app f a,tail)
  | _+1, '1' :: rest =>
    let ones := rest.takeWhile (· == '1')
    let '0' :: tail := rest.drop ones.length | none
    return (.var ones.length,tail)
  | _,_ => none

def decode (s : String) : Option Term := do
  let bits := s.toList
  let (t,[]) ← parse (bits.length+1) bits | none
  return t

def auditHeads (path : System.FilePath) : IO Unit := do
  let mut fills := 0
  let mut hits := 0
  for line in (← IO.FS.lines path) do
    let [kind,source,result] := line.splitOn " " | throw (IO.userError "bad event row")
    let some t := decode source | throw (IO.userError "bad source syntax")
    let some v := decode result | throw (IO.userError "bad cached syntax")
    let some expected := weakRun 10000 t | throw (IO.userError "inconclusive head budget")
    unless v = expected do throw (IO.userError s!"wrong cached head on {kind} after {fills} fills and {hits} hits")
    match kind with
    | "FILL" => fills := fills+1
    | "HIT" => hits := hits+1
    | _ => throw (IO.userError "unknown event kind")
  if fills == 0 || hits == 0 then throw (IO.userError "missing fill or reuse")
  IO.println s!"Checked {fills} stored heads and {hits} memo hits."

end Verificatum.Lazy

def main : List String → IO Unit
  | [] => pure ()
  | ["--emit"] => Verificatum.Lazy.emitCorpus
  | ["--heads", file] => Verificatum.Lazy.auditHeads file
  | _ => throw (IO.userError "expected --emit or --heads FILE")
