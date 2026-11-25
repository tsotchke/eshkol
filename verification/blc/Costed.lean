import Init

namespace BLC.Costed

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

-- A delayed term captures its lexical environment. Every demand evaluates
-- that term anew. Bound levels and root-relative free indices also allow
-- readback under lambdas.
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

def initial (t : Term) : Closure := .delay t (fun i => .atom (.free i))

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

def spends (view : Nat → α → View α) : State α → Bool
  | .seek d f (_ :: _) _ => match view d f with
    | .lam .. => true
    | _ => false
  | _ => false

private inductive Progress (s : State Term) : Bool → Sum Term (State Term) → Prop where
  | done : plug s = n → step n = none → Progress s false (.inl n)
  | admin : Safe t → plug s = plug t → weight t < weight s → Progress s false (.inr t)
  | betaStep : Safe t → step (plug s) = some (plug t) → Progress s true (.inr t)

private theorem progress (s : State Term) (h : Safe s) : Progress s (spends sourceView s) (tick sourceView s) := by
  cases s with
  | seek d f as ks =>
    cases f with
    | var i =>
      cases as with
      | nil => exact .admin ⟨rfl,h⟩ rfl (by simp [spine,weight, work, argWork] <;> omega)
      | cons x xs =>
        exact .admin ⟨neutralVar _,h⟩ rfl
          (by simp [spine,weight, argWork, frameWork, work] <;> omega)
    | app f a => cases as <;> exact .admin h rfl (by simp [weight, argWork, work] <;> omega)
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

-- A missing inner result is the public bound error. A missing outer result
-- means that the replay stopped before observing either public outcome.
abbrev Answer := Option (Nat × Term)
def charged (a : Answer) : Answer := a.map fun (k,n) => (k+1,n)

def refBound : Nat → Term → Answer
  | 0, t => match step t with
    | none => some (0,t)
    | some _ => none
  | m+1, t => match step t with
    | none => some (0,t)
    | some u => charged (refBound m u)

def execute (v : Nat → α → View α) : Nat → Nat → State α → Option Answer
  | 0, _, _ => none
  | fuel+1, m, s => match tick v s with
    | .inl n => some (some (0,n))
    | .inr u =>
      if spends v s then match m with
        | 0 => some none
        | m+1 => (execute v fuel m u).map charged
      else execute v fuel m u

private theorem spends_rel (h : States s t) :
    spends closureView s = spends sourceView t := by
  cases h with
  | emit => rfl
  | seek fn args _ =>
    cases args with
    | nil => rfl
    | cons =>
      have vv := observe _ _ fn.1
      rw [fn.2] at vv
      simp only [spends]
      generalize closureView _ _ = cv at vv ⊢
      generalize sourceView _ _ = tv at vv ⊢
      cases vv <;> rfl

private theorem agree (fuel m : Nat) (h : States s t) :
    execute closureView fuel m s = execute sourceView fuel m t := by
  induction fuel generalizing s t m with
  | zero => rfl
  | succ f ih =>
    have tt := tick_rel h
    simp only [execute, spends_rel h]
    generalize tick closureView s = a at tt ⊢
    generalize tick sourceView t = b at tt ⊢
    cases tt with
    | done => rfl
    | step rel =>
      cases spends sourceView t <;> cases m <;> simp [ih _ rel]

private theorem complete (m : Nat) (s : State Term) (hs : Safe s) :
    ∃ fuel, execute sourceView fuel m s = some (refBound m (plug s)) := by
  have p := progress s hs
  generalize ht : tick sourceView s = out at p
  generalize hc : spends sourceView s = cost at p
  cases p with
  | done hp hn =>
    exact ⟨1, by cases m <;> simp [execute,ht,refBound,hp,hn]⟩
  | @admin u hu hp hw =>
    obtain ⟨f,h⟩ := complete m u hu
    exact ⟨f+1, by simp [execute,ht,hc,h,hp]⟩
  | @betaStep u hu hb =>
    cases m with
    | zero => exact ⟨1, by simp [execute,ht,hc,refBound,hb]⟩
    | succ m =>
      obtain ⟨f,h⟩ := complete m u hu
      exact ⟨f+1, by simp [execute,ht,hc,h,refBound,hb]⟩
termination_by (m, weight s)

private theorem sound (f m : Nat) (s : State Term) (hs : Safe s)
    (h : execute sourceView f m s = some r) : refBound m (plug s) = r := by
  induction f generalizing m s r with
  | zero => simp [execute] at h
  | succ f ih =>
    have p := progress s hs
    generalize ht : tick sourceView s = out at p
    generalize hc : spends sourceView s = cost at p
    cases p with
    | done hp hn =>
      simp [execute,ht] at h
      cases m <;> simpa [refBound,hp,hn] using h
    | admin hu hp _ =>
      exact hp ▸ ih _ _ hu (by simpa [execute,ht,hc] using h)
    | @betaStep u hu hb =>
      cases m with
      | zero => simpa [execute,ht,hc,refBound,hb] using h
      | succ m =>
        cases he : execute sourceView f m u with
        | none => simp [execute,ht,hc,he] at h
        | some v =>
          simpa [refBound,hb,ih m u hu he] using (by simpa [execute,ht,hc,he] using h : charged v = r)

def ClosureResult (t : Term) (m : Nat) (r : Answer) :=
  ∃ fuel, execute closureView fuel m (.seek 0 (initial t) [] []) = some r

theorem bounded_correspondence (t : Term) (m : Nat) (r : Answer) :
    refBound m t = r ↔ ClosureResult t m r := by
  have start : States (.seek 0 (initial t) [] []) (.seek 0 t [] []) := by
    apply States.seek _ .nil .nil
    constructor
    · intro i; trivial
    · simpa [initial, erase, index] using idSub t
  constructor
  · intro hr
    obtain ⟨f,h⟩ := complete m (.seek 0 t [] []) True.intro
    exact ⟨f,(agree f m start).trans (h.trans (congrArg some hr))⟩
  · rintro ⟨f,h⟩
    exact sound f m (.seek 0 t [] []) True.intro ((agree f m start).symm.trans h)

theorem exact_cost (t n : Term) (k : Nat) :
    refBound k t = some (k,n) ↔ ClosureResult t k (some (k,n)) :=
  bounded_correspondence ..

theorem same_limit_error (t : Term) (m : Nat) :
    refBound m t = none ↔ ClosureResult t m none := bounded_correspondence ..

#print axioms bounded_correspondence
end BLC.Costed
