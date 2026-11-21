import Lazy

namespace Verificatum.Heap

open Verificatum.Lazy

-- Mathematical call by name, with no heap, addresses, update rule or cache.
-- Neutral arguments are stored newest first, as in the nested spine pairs.
inductive WH where
  | lam (body : Term) (env : Nat → Closure)
  | neutral (head : Head) (rev : List Closure)

inductive Eval : Closure → WH → Prop where
  | atom : Eval (.atom h) (.neutral h [])
  | lam : Eval (.delay (.lam b) e) (.lam b e)
  | var : Eval (e i) v → Eval (.delay (.var i) e) v
  | beta : Eval (.delay f e) (.lam b s) →
      Eval (.delay b (push (.delay a e) s)) v → Eval (.delay (.app f a) e) v
  | neutral : Eval (.delay f e) (.neutral h rev) →
      Eval (.delay (.app f a) e) (.neutral h (.delay a e :: rev))

theorem eval_unique (p : Eval c w) (q : Eval c v) : w = v := by
  induction p generalizing v with
  | atom => cases q; rfl
  | lam => cases q; rfl
  | var _ ih => cases q with | var q => exact ih q
  | beta _ _ ihf ihb =>
    cases q with
    | beta f b => cases ihf f; exact ihb b
    | neutral f => cases ihf f
  | neutral _ ih =>
    cases q with
    | beta f _ => cases ih f
    | neutral f => cases ih f; rfl

-- One run owns its addresses. Reflected binders carry levels directly.
inductive Ref where
  | addr (i : Nat)
  | level (d : Nat)
deriving DecidableEq
abbrev Env := List Ref

deriving instance DecidableEq for Head

inductive Value where
  | lam (body : Term) (env : Env)
  | neutral (head : Head) (rev : List Ref)
deriving DecidableEq
inductive Cell where
  | pending (code : Term) (env : Env)
  | writing (value : Value)
  | memo (value : Value)
abbrev Store := Nat → Option Cell
abbrev Origins := Nat → Option Closure

def put (m : Nat → α) (i : Nat) (x : α) := fun j => if j = i then x else m j

inductive RefOK (o : Origins) : Ref → Closure → Prop where
  | addr : o i = some c → RefOK o (.addr i) c
  | level : RefOK o (.level d) (.atom (.bound d))

inductive EnvOK (o : Origins) : Env → (Nat → Closure) → Prop where
  | nil : EnvOK o [] (fun i => .atom (.free i))
  | cons : RefOK o a c → EnvOK o es s → EnvOK o (a :: es) (push c s)

inductive RefsOK (o : Origins) : List Ref → List Closure → Prop where
  | nil : RefsOK o [] []
  | cons : RefOK o a c → RefsOK o as cs → RefsOK o (a :: as) (c :: cs)

inductive ValueOK (o : Origins) : Value → WH → Prop where
  | lam : EnvOK o env e → ValueOK o (.lam b env) (.lam b e)
  | neutral : RefsOK o as cs → ValueOK o (.neutral h as) (.neutral h cs)

inductive SlotOK (o : Origins) : Option Closure → Option Cell → Prop where
  | empty : SlotOK o none none
  | pending : EnvOK o env e → SlotOK o (some (.delay t e)) (some (.pending t env))
  | writing : ValueOK o v w → Eval c w → SlotOK o (some c) (some (.writing v))
  | memo : ValueOK o v w → Eval c w → SlotOK o (some c) (some (.memo v))

structure HeapOK (o : Origins) (fresh : Nat) (heap : Store) : Prop where
  slots : ∀ i, SlotOK o (o i) (heap i)
  unused : ∀ i, fresh ≤ i → o i = none

abbrev Extends (o o' : Origins) := ∀ i c, o i = some c → o' i = some c

private theorem refExt (h : Extends o o') (r : RefOK o a c) : RefOK o' a c := by
  cases r with
  | level => constructor
  | addr ha => exact .addr (h _ _ ha)

private theorem envExt (h : Extends o o') (r : EnvOK o a c) : EnvOK o' a c := by
  induction r with
  | nil => constructor
  | cons x _ ih => exact .cons (refExt h x) ih

private theorem refsExt (h : Extends o o') (r : RefsOK o a c) : RefsOK o' a c := by
  induction r with
  | nil => constructor
  | cons x _ ih => exact .cons (refExt h x) ih

private theorem valExt (h : Extends o o') (r : ValueOK o a c) : ValueOK o' a c := by
  cases r with
  | lam e => exact .lam (envExt h e)
  | neutral xs => exact .neutral (refsExt h xs)

private theorem slotExt (h : Extends o o') (r : SlotOK o a c) : SlotOK o' a c := by
  cases r with
  | empty => constructor
  | pending e => exact .pending (envExt h e)
  | writing v ev => exact .writing (valExt h v) ev
  | memo v ev => exact .memo (valExt h v) ev

-- The active update's frame carries a provenance obligation. Requiring only
-- HeapOK would allow a forged return stack to write any unrelated value.
inductive K where
  | arg (r : Ref)
  | update (address : Nat)
inductive R where
  | quoteAt (d : Nat)
  | wrapLam
  | quoteArg (d : Nat) (r : Ref)
  | wrapApp (f : Term)
inductive Control where
  | eval (t : Term) (env : Env)
  | lookup (i : Nat) (env : Env)
  | force (r : Ref)
  | value (v : Value)
  | publish (i : Nat) (v : Value)
  | emit (t : Term)

inductive Apply : WH → Closure → WH → Prop where
  | lam : Eval (.delay b (push x e)) v → Apply (.lam b e) x v
  | neutral : Apply (.neutral h rev) x (.neutral h (x :: rev))

inductive StackOK (o : Origins) : (WH → Prop) → List K → Prop where
  | nil : StackOK o P []
  | arg : RefOK o r x → StackOK o (fun w => ∃ f, P f ∧ Apply f x w) ks →
      StackOK o P (.arg r :: ks)
  | update : o i = some c → (∀ v, P v → Eval c v) → StackOK o P ks →
      StackOK o P (.update i :: ks)

private theorem narrow (h : StackOK o Q ks) (imp : ∀ v, P v → Q v) : StackOK o P ks := by
  induction h generalizing P with
  | nil => constructor
  | arg r _ ih =>
    apply StackOK.arg r
    apply ih
    rintro w ⟨f,hp,ha⟩
    exact ⟨f,imp f hp,ha⟩
  | update key save _ ih => exact .update key (fun v hp => save v (imp v hp)) (ih imp)

private theorem stackExt (h : Extends o o') (r : StackOK o P ks) : StackOK o' P ks := by
  induction r with
  | nil => constructor
  | arg x _ ih => exact .arg (refExt h x) ih
  | update key save _ ih => exact .update (h _ _ key) save ih

def ReadOK (o : Origins) : List R → Prop
  | [] => True
  | .quoteArg _ r :: ks => (∃ c, RefOK o r c) ∧ ReadOK o ks
  | _ :: ks => ReadOK o ks

private theorem readExt (h : Extends o o') (ks : List R) (s : ReadOK o ks) : ReadOK o' ks := by
  induction ks with
  | nil => trivial
  | cons f ks ih =>
    cases f with
    | quoteArg d r =>
      obtain ⟨⟨c,hc⟩,hh⟩ := s
      exact ⟨⟨c,refExt h hc⟩,ih hh⟩
    | quoteAt | wrapLam | wrapApp => exact ih s

def ControlOK (o : Origins) (ks : List K) : Control → Prop
  | .eval t env => ∃ e, EnvOK o env e ∧ StackOK o (Eval (.delay t e)) ks
  | .lookup i env => ∃ e, EnvOK o env e ∧ StackOK o (Eval (e i)) ks
  | .force r => ∃ c, RefOK o r c ∧ StackOK o (Eval c) ks
  | .value v | .publish _ v => ∃ w, ValueOK o v w ∧ StackOK o (· = w) ks
  | .emit _ => ks = []

structure Config where
  fresh : Nat
  heap : Store
  ctl : Control
  work : List K
  readback : List R

def Invariant (s : Config) := ∃ o, HeapOK o s.fresh s.heap ∧
    ControlOK o s.work s.ctl ∧ ReadOK o s.readback

-- These rules operate only on raw data. In particular, write and publish
-- contain no Eval, Entry, origin-map or semantic-correctness premise.
inductive Step : Config → Config → Prop where
  | alloc : Step ⟨n,h,.eval (.app f a) env,k,r⟩
      ⟨n+1,put h n (some (.pending a env)),.eval f env,.arg (.addr n)::k,r⟩
  | lambda : Step ⟨n,h,.eval (.lam b) env,k,r⟩ ⟨n,h,.value (.lam b env),k,r⟩
  | beginLookup : Step ⟨n,h,.eval (.var i) env,k,r⟩ ⟨n,h,.lookup i env,k,r⟩
  | free : Step ⟨n,h,.lookup i [],k,r⟩ ⟨n,h,.value (.neutral (.free i) []),k,r⟩
  | first : Step ⟨n,h,.lookup 0 (a::env),k,r⟩ ⟨n,h,.force a,k,r⟩
  | outer : Step ⟨n,h,.lookup (i+1) (a::env),k,r⟩ ⟨n,h,.lookup i env,k,r⟩
  | demand : h i = some (.pending t env) → Step ⟨n,h,.force (.addr i),k,r⟩
      ⟨n,h,.eval t env,.update i::k,r⟩
  | hit : h i = some (.memo v) → Step ⟨n,h,.force (.addr i),k,r⟩ ⟨n,h,.value v,k,r⟩
  | level : Step ⟨n,h,.force (.level d),k,r⟩ ⟨n,h,.value (.neutral (.bound d) []),k,r⟩
  | write : Step ⟨n,h,.value v,.update i::k,r⟩
      ⟨n,put h i (some (.writing v)),.publish i v,k,r⟩
  | publish : h i = some (.writing v) → Step ⟨n,h,.publish i v,k,r⟩
      ⟨n,put h i (some (.memo v)),.value v,k,r⟩
  | enterLam : Step ⟨n,h,.value (.lam b env),.arg x::k,r⟩
      ⟨n,h,.eval b (x::env),k,r⟩
  | enterNeutral : Step ⟨n,h,.value (.neutral a as),.arg x::k,r⟩
      ⟨n,h,.value (.neutral a (x::as)),k,r⟩
  | underLam : Step ⟨n,h,.value (.lam b env),[],.quoteAt d::r⟩
      ⟨n,h,.eval b (.level d::env),[],.quoteAt (d+1):: .wrapLam::r⟩
  | head : Step ⟨n,h,.value (.neutral a []),[],.quoteAt d::r⟩
      ⟨n,h,.emit (.var (index d a)),[],r⟩
  | spine : Step ⟨n,h,.value (.neutral a (x::as)),[],.quoteAt d::r⟩
      ⟨n,h,.value (.neutral a as),[],.quoteAt d :: .quoteArg d x :: r⟩
  | wrapLam : Step ⟨n,h,.emit b,[],.wrapLam::r⟩ ⟨n,h,.emit (.lam b),[],r⟩
  | forceArg : Step ⟨n,h,.emit f,[],.quoteArg d x::r⟩
      ⟨n,h,.force x,[],.quoteAt d :: .wrapApp f :: r⟩
  | wrapApp : Step ⟨n,h,.emit a,[],.wrapApp f::r⟩ ⟨n,h,.emit (.app f a),[],r⟩

private theorem inspectSlot (hh : HeapOK o n h) (load : h i = some cell) :
    SlotOK o (o i) (some cell) := load ▸ hh.slots i

private theorem updateSlot (hh : HeapOK o n h) (slot : SlotOK o (o i) v) :
    HeapOK o n (put h i v) := by
  refine ⟨fun j => ?_,hh.unused⟩
  by_cases eq : j = i
  · subst j; simpa [put] using slot
  · simpa [put,eq] using hh.slots j

def Holds (o : Origins) (s : Config) := HeapOK o s.fresh s.heap ∧
    ControlOK o s.work s.ctl ∧ ReadOK o s.readback

theorem preserves (transition : Step s t) (inv : Holds o s) :
    ∃ o', Extends o o' ∧ Holds o' t := by
  obtain ⟨heap,ctl,read⟩ := inv
  have same : Extends o o := fun _ _ h => h
  cases transition with
  | @alloc n h f a env k r =>
    obtain ⟨e,envOK,kOK⟩ := ctl
    let c := Closure.delay a e
    let o' := put o n (some c)
    have newKey : o' n = some c := by simp [o',put]
    have fresh : o n = none := heap.unused n (by simp)
    have ext : Extends o o' := by
      intro i key hk
      have ne : i ≠ n := by rintro rfl; simp_all
      simpa [o',put,ne] using hk
    have envNew := envExt ext envOK
    refine ⟨o',ext,⟨fun i => ?_,fun i hi => ?_⟩,⟨e,envNew,?_⟩,readExt ext _ read⟩
    · by_cases eq : i = n
      · subst i; simpa [o',put,c] using (SlotOK.pending (t := a) envNew)
      · simpa [o',put,eq] using slotExt ext (heap.slots i)
    · change n+1 ≤ i at hi
      have ne : i ≠ n := by omega
      simpa [o',put,ne] using heap.unused i (show n ≤ i by omega)
    · apply StackOK.arg (.addr newKey)
      apply narrow (stackExt ext kOK)
      rintro w ⟨v,hf,ha⟩
      cases ha with
      | lam hb => exact .beta hf hb
      | neutral => exact .neutral hf
  | lambda =>
    obtain ⟨e,he,hk⟩ := ctl
    exact ⟨o,same,heap,⟨_,.lam he,narrow hk (by rintro _ rfl; exact .lam)⟩,read⟩
  | beginLookup =>
    obtain ⟨e,he,hk⟩ := ctl
    exact ⟨o,same,heap,⟨e,he,narrow hk (fun _ h => .var h)⟩,read⟩
  | free =>
    obtain ⟨e,he,hk⟩ := ctl
    cases he
    exact ⟨o,same,heap,⟨_,.neutral .nil,narrow hk (by rintro _ rfl; exact .atom)⟩,read⟩
  | first =>
    obtain ⟨e,he,hk⟩ := ctl
    cases he with
    | cons a _ => exact ⟨o,same,heap,⟨_,a,hk⟩,read⟩
  | outer =>
    obtain ⟨e,he,hk⟩ := ctl
    cases he with
    | cons _ es => exact ⟨o,same,heap,⟨_,es,hk⟩,read⟩
  | demand load =>
    obtain ⟨c,hc,hk⟩ := ctl
    cases hc with
    | addr key =>
      have slot := inspectSlot heap load
      rw [key] at slot
      cases slot with
      | pending env => exact ⟨o,same,heap,⟨_,env,.update key (fun _ h => h) hk⟩,read⟩
  | hit load =>
    obtain ⟨c,hc,hk⟩ := ctl
    cases hc with
    | addr key =>
      have slot := inspectSlot heap load
      rw [key] at slot
      cases slot with
      | memo hv ev =>
        exact ⟨o,same,heap,⟨_,hv,narrow hk (by rintro _ rfl; exact ev)⟩,read⟩
  | level =>
    obtain ⟨c,hc,hk⟩ := ctl
    cases hc
    exact ⟨o,same,heap,⟨_,.neutral .nil,narrow hk (by rintro _ rfl; exact .atom)⟩,read⟩
  | write =>
    obtain ⟨w,hv,hk⟩ := ctl
    cases hk with
    | update key provenance rest =>
      exact ⟨o,same,updateSlot heap (key ▸ SlotOK.writing hv (provenance w rfl)),
             ⟨w,hv,rest⟩,read⟩
  | publish load =>
    have slot := inspectSlot heap load
    generalize key : o _ = opt at slot
    cases slot with
    | writing hv ev =>
      exact ⟨o,same,updateSlot heap (key ▸ SlotOK.memo hv ev),ctl,read⟩
  | enterLam =>
    obtain ⟨w,hv,hk⟩ := ctl
    cases hv with
    | lam env =>
      cases hk with
      | arg x rest =>
        exact ⟨o,same,heap,⟨_,.cons x env,narrow rest (fun _ ev => ⟨_,rfl,.lam ev⟩)⟩,read⟩
  | enterNeutral =>
    obtain ⟨w,hv,hk⟩ := ctl
    cases hv with
    | neutral xs =>
      cases hk with
      | arg x rest =>
        exact ⟨o,same,heap,⟨_,.neutral (.cons x xs),
                 narrow rest (by rintro _ rfl; exact ⟨_,rfl,.neutral⟩)⟩,read⟩
  | underLam =>
    obtain ⟨w,hv,_⟩ := ctl
    cases hv with
    | lam env => exact ⟨o,same,heap,⟨_,.cons .level env,.nil⟩,read⟩
  | head | wrapLam | wrapApp => exact ⟨o,same,heap,rfl,read⟩
  | spine =>
    obtain ⟨w,hv,_⟩ := ctl
    cases hv with
    | neutral xs =>
      cases xs with
      | cons x tail => exact ⟨o,same,heap,⟨_,.neutral tail,.nil⟩,⟨_,x⟩,read⟩
  | forceArg =>
    obtain ⟨⟨_,hx⟩,hr⟩ := read
    exact ⟨o,same,heap,⟨_,hx,.nil⟩,hr⟩

-- While the car tag still says pending after the cdr write, no other
-- dispatcher action may inspect the cell. Only publication can run.
def WritesAgree (s : Config) := ∀ i v,
    s.heap i = some (.writing v) ↔ s.ctl = .publish i v

private theorem writes_preserved (st : Step s t) (agree : WritesAgree s) : WritesAgree t := by
  cases st <;> simp only [WritesAgree] at agree ⊢ <;> intro j w
  all_goals grind [put]

def start (t : Term) : Config := ⟨0,fun _ => none,.eval t [],[],[.quoteAt 0]⟩
inductive Reachable (t : Term) : Config → Prop where
  | init : Reachable t (start t)
  | next : Reachable t s → Step s u → Reachable t u

theorem reachable_invariant (h : Reachable t s) : Invariant s := by
  induction h with
  | init => exact ⟨fun _ => none,⟨fun _ => .empty,fun _ _ => rfl⟩,⟨_,.nil,.nil⟩,True.intro⟩
  | next _ st ih =>
    obtain ⟨o,hold⟩ := ih
    obtain ⟨o',_,hold'⟩ := preserves st hold
    exact ⟨o',hold'⟩

theorem reachable_writes (reach : Reachable t s) : WritesAgree s := by
  induction reach with
  | init => intro i v; simp [start]
  | next _ st ih => exact writes_preserved st ih

-- Resuming the model exactly at the cut has one enabled transition. Native
-- fuel checks occur outside this two-store region of a single dispatch.
theorem publication_only (reach : Reachable t ⟨n,h,.publish i v,k,r⟩) :
    ∃ u, Step ⟨n,h,.publish i v,k,r⟩ u ∧
      (∀ u', Step ⟨n,h,.publish i v,k,r⟩ u' → u' = u) ∧
      u = ⟨n,put h i (some (.memo v)),.value v,k,r⟩ := by
  have staged := (reachable_writes reach i v).mpr rfl
  exact ⟨_,.publish staged,by intro u' other; cases other; rfl,rfl⟩

theorem force_finds_live_cell (reach : Reachable t s) (forcing : s.ctl = .force (.addr i)) :
    (∃ code env, s.heap i = some (.pending code env)) ∨
    (∃ value, s.heap i = some (.memo value)) := by
  obtain ⟨o,heap,ctl,_⟩ := reachable_invariant reach
  rw [forcing] at ctl
  obtain ⟨c,hc,_⟩ := ctl
  cases hc with
  | addr key =>
    have slot := heap.slots i
    rw [key] at slot
    generalize load : s.heap i = cell at slot
    cases slot with
    | pending => exact Or.inl ⟨_,_,rfl⟩
    | memo => exact Or.inr ⟨_,rfl⟩
    | writing =>
      have phase := (reachable_writes reach i _).mp load
      simp [forcing] at phase

-- Relate the independent evaluation judgment back to the earlier cache theorem.
-- An administrative variable lookup costs zero abstract head hops. Retain the
-- syntactic variable as a head representative in this case.
inductive LamAt : Closure → Term → (Nat → Closure) → Prop where
  | direct : LamAt (.delay (.lam b) e) b e
  | var : LamAt (e i) b s → LamAt (.delay (.var i) e) b s
inductive AtomAt : Closure → Head → Prop where
  | direct : AtomAt (.atom h) h
  | var : AtomAt (e i) h → AtomAt (.delay (.var i) e) h

def Represents (w : WH) (q : Closure × List Closure) := match w with
  | .lam b e => LamAt q.1 b e ∧ q.2 = []
  | .neutral h rev => AtomAt q.1 h ∧ q.2 = rev.reverse

private theorem lamEnter (atLam : LamAt f b e) (x : Closure) :
    EnterView f x (.delay b (push x e)) := by
  induction atLam with
  | direct => exact .direct
  | var _ ih => exact .lookup ih

private theorem hopsTrans (h : Hops n x y) (j : Hops m y z) : Hops (m+n) x z := by
  induction h with
  | refl => exact j
  | cons one _ ih => exact .cons one (ih j)

private theorem hopsArgs (h : Hops n x y) (more : List Closure) :
    Hops n (x.1,x.2 ++ more) (y.1,y.2 ++ more) := by
  induction h with
  | refl => exact .refl
  | cons one _ ih =>
    cases one with
    | app a => exact .cons (.app a) ih
    | lam a => exact .cons (.lam a) ih

private theorem variablePath (w : WH) (path : Hops n (e i,[]) q) (rep : Represents w q) :
    ∃ m p, Hops m (.delay (.var i) e,[]) p ∧ Represents w p := by
  generalize he : (e i, ([] : List Closure)) = start at path
  cases path with
  | refl =>
    cases he
    refine ⟨0,_,Hops.refl,?_⟩
    cases w with
    | lam b s => exact ⟨.var rep.1,rep.2⟩
    | neutral h rev => exact ⟨.var rep.1,rep.2⟩
  | cons first tail =>
    cases he
    cases first with
    | app hf => exact ⟨_,_,.cons (.app (.lookup hf)) tail,rep⟩

theorem earned_path (ev : Eval c w) :
    ∃ n q, Hops n (c,[]) q ∧ Represents w q := by
  induction ev with
  | atom => exact ⟨0,_,.refl,.direct,rfl⟩
  | lam => exact ⟨0,_,.refl,.direct,rfl⟩
  | var _ ih =>
    obtain ⟨_,_,path,rep⟩ := ih
    exact variablePath _ path rep
  | @beta f e b s a v hf hb ihf ihb =>
    obtain ⟨_,⟨head,args⟩,path,hL,hargs⟩ := ihf
    change args = [] at hargs
    subst args
    obtain ⟨_,q,body,rep⟩ := ihb
    have funPath := hopsArgs path [.delay a e]
    have enter : HeadHop (head,[.delay a e]) (.delay b (push (.delay a e) s),[]) :=
      .lam (lamEnter hL _)
    exact ⟨_,q,.cons (.app .direct) (hopsTrans funPath (.cons enter body)),rep⟩
  | @neutral f e h rev a hf ih =>
    obtain ⟨_,⟨hd,args⟩,path,ha,eq⟩ := ih
    change args = rev.reverse at eq
    subst args
    exact ⟨_,_,.cons (.app .direct) (hopsArgs path [.delay a e]),ha,by simp⟩

-- One origin map works for all cells simultaneously. A hit yields the exact
-- Entry required by cached_normal_order_correspondence and represents the
-- actual stored value, including the ordered arguments of a neutral spine.
theorem reachable_memos (reach : Reachable t s) :
    ∃ o, Holds o s ∧ ∀ i v, s.heap i = some (.memo v) →
      ∃ c w, o i = some c ∧ ValueOK o v w ∧
        ∃ entry : Entry c, Represents w (entry.head,entry.args) := by
  obtain ⟨o,hold⟩ := reachable_invariant reach
  refine ⟨o,hold,?_⟩
  intro i v load
  have slot := inspectSlot hold.1 load
  generalize ho : o i = key at slot
  cases slot with
  | memo val ev =>
    obtain ⟨hops,⟨head,args⟩,path,rep⟩ := earned_path ev
    exact ⟨_,_,rfl,val,⟨⟨head,args,hops,path⟩,rep⟩⟩

inductive Steps : Config → Config → Prop where
  | refl : Steps s s
  | next : Steps s t → Step t u → Steps s u

theorem origins_persist (steps : Steps s t) (safe : Holds o s) :
    ∃ o', Extends o o' ∧ Holds o' t := by
  induction steps with
  | refl => exact ⟨o,fun _ _ h => h,safe⟩
  | next _ step ih =>
    obtain ⟨old,ext,ok⟩ := ih
    obtain ⟨fresh,ext',ok'⟩ := preserves step ok
    exact ⟨fresh,fun i c h => ext' i c (ext i c h),ok'⟩

-- Original suspension-to-suspension references strictly descend in allocation
-- order. A memoized result may refer to newer suspensions created by demand.
theorem captured_address_older (inv : Holds o ⟨n,h,.eval (.app f a) env,k,r⟩)
    (member : Ref.addr i ∈ env) : i < n := by
  obtain ⟨heap,⟨e,henv,stk⟩,read⟩ := inv
  clear stk read
  have bounded : ∀ j c, o j = some c → j < n := by
    intro j c hj
    by_cases lt : j < n
    · exact lt
    · rw [heap.unused j (Nat.le_of_not_gt lt)] at hj; contradiction
  clear heap
  induction henv with
  | nil => cases member
  | cons key _ ih =>
    rcases List.mem_cons.mp member with eq | tail
    · subst_vars; cases key with | addr hk => exact bounded _ _ hk
    · exact ih tail

-- Once allocated, this exact code and its captured environment remain the
-- origin of the address throughout every finite future of that run.
theorem allocation_survives
    (inv : Holds o ⟨n,h,.eval (.app f a) env,k,r⟩)
    (future : Steps ⟨n+1,put h n (some (.pending a env)),.eval f env,.arg (.addr n)::k,r⟩ t) :
    ∃ later e, Holds later t ∧ EnvOK later env e ∧ later n = some (.delay a e) := by
  obtain ⟨new,_,ok⟩ := preserves Step.alloc inv
  have slot := inspectSlot ok.1 (show put h n (some (.pending a env)) n = some (.pending a env) by simp [put])
  generalize key : new n = origin at slot
  cases slot with
  | pending envOK =>
    obtain ⟨later,ext,final⟩ := origins_persist future ok
    exact ⟨later,_,final,envExt ext envOK,ext _ _ key⟩

/-- info: 'Verificatum.Heap.reachable_memos' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms reachable_memos
/-- info: 'Verificatum.Heap.origins_persist' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms origins_persist

-- Each successful branch of the executable replay must construct a Step
-- proof. Its projected events are compared with the native pair operations.
private def bits : Term → String
  | .var i => String.ofList (List.replicate (i+1) '1') ++ "0"
  | .lam b => "00" ++ bits b
  | .app f a => "01" ++ bits f ++ bits a

private def refName : Ref → String
  | .addr i => s!"a{i}"
  | .level d => s!"l{d}"

private def names (refs : List Ref) := String.join (refs.map (fun x => " " ++ refName x))
private def valueName : Value → String
  | .lam body env => "FUN " ++ bits body ++ names env
  | .neutral (.free i) rev => s!"FREE {i}" ++ names rev.reverse
  | .neutral (.bound d) rev => s!"BOUND {d}" ++ names rev.reverse

private def next (s : Config) : Option (Option String × {u // Step s u}) :=
  match s with
  | ⟨n,_,.eval (.app _ a) env,_,_⟩ =>
    some (some (s!"ALLOC {n} {bits a}" ++ names env), ⟨_,.alloc⟩)
  | ⟨_,_,.eval (.var _) _,_,_⟩ => some (none,⟨_,.beginLookup⟩)
  | ⟨_,_,.eval (.lam _) _,_,_⟩ => some (none,⟨_,.lambda⟩)
  | ⟨_,_,.lookup _ [],_,_⟩ => some (none,⟨_,.free⟩)
  | ⟨_,_,.lookup 0 (_::_),_,_⟩ => some (none,⟨_,.first⟩)
  | ⟨_,_,.lookup (_+1) (_::_),_,_⟩ => some (none,⟨_,.outer⟩)
  | ⟨_,h,.force (.addr i),_,_⟩ =>
    match eq : h i with
    | some (.pending _ _) => some (some s!"FORCE {i}",⟨_,.demand eq⟩)
    | some (.memo v) => some (some s!"HIT {i} {valueName v}",⟨_,.hit eq⟩)
    | _ => none
  | ⟨_,_,.force (.level d),_,_⟩ => some (some s!"LEVEL {d}",⟨_,.level⟩)
  | ⟨_,_,.value v,.update i::_,_⟩ => some (some s!"WRITE {i} {valueName v}",⟨_,.write⟩)
  | ⟨_,h,.publish i v,_,_⟩ =>
    match eq : h i with
    | some (.writing stored) =>
      if hv : stored = v then some (some s!"PUBLISH {i} {valueName stored}",⟨_,.publish (hv ▸ eq)⟩)
      else none
    | _ => none
  | ⟨_,_,.value (.lam _ _),.arg _::_,_⟩ => some (none,⟨_,.enterLam⟩)
  | ⟨_,_,.value (.neutral _ _),.arg _::_,_⟩ => some (none,⟨_,.enterNeutral⟩)
  | ⟨_,_,.value (.lam _ _),[],.quoteAt _::_⟩ => some (none,⟨_,.underLam⟩)
  | ⟨_,_,.value (.neutral _ []),[],.quoteAt _::_⟩ => some (none,⟨_,.head⟩)
  | ⟨_,_,.value (.neutral _ (_::_)),[],.quoteAt _::_⟩ => some (none,⟨_,.spine⟩)
  | ⟨_,_,.emit _,[],.wrapLam::_⟩ => some (none,⟨_,.wrapLam⟩)
  | ⟨_,_,.emit _,[],.quoteArg _ _::_⟩ => some (none,⟨_,.forceArg⟩)
  | ⟨_,_,.emit _,[],.wrapApp _::_⟩ => some (none,⟨_,.wrapApp⟩)
  | _ => none

private def replayRun (term : Term) (fuel : Nat) : Except String (Array String) := do
  let mut events := #[]
  let mut state := start term
  for steps in [0:fuel+1] do
    if let ⟨_,_,.emit nf,[],[]⟩ := state then
      return events.push s!"STOP normal {steps} {bits nf}"
    if steps == fuel then return events.push s!"STOP exhausted {steps}"
    if let ⟨_,_,.value v,.update i :: _,_⟩ := state then
      events := events.push s!"RETURN {i} {valueName v}"
    let some (ev,u) := next state | throw s!"stuck model at dispatch {steps}"
    if let some e := ev then events := events.push e
    state := u.val
    -- The Eshkol loop executes both pair writes within a single dispatch.
    if let .publish _ _ := state.ctl then
      let some (some e,v) := next state | throw "stuck before publish"
      events := events.push e
      state := v.val
  throw "missing terminal status"

def auditHeap (path : System.FilePath) (interrupted : Bool := false) : IO Unit := do
  let mut expected : Array String := #[]
  let mut atEvent := 0
  let mut runs := 0
  let mut operations := 0
  for line in (← IO.FS.lines path) do
    match line.splitOn " " with
    | ["START", bound, input] =>
      unless atEvent == expected.size do throw (IO.userError "incomplete preceding run")
      let some t := decode input | throw (IO.userError "bad input term")
      let some fuel := bound.toNat? | throw (IO.userError "bad budget")
      match replayRun t fuel with
      | .error err => throw (IO.userError err)
      | .ok trace => expected := trace
      atEvent := 0
      runs := runs+1
    | _ =>
      unless expected[atEvent]? == some line do
        throw (IO.userError s!"heap event mismatch in run {runs} at {atEvent}: expected {repr expected[atEvent]?}, saw {line}")
      atEvent := atEvent+1
      operations := operations+1
  if interrupted then
    unless runs == 1 && atEvent > 0 && (expected[atEvent-1]?.getD "").startsWith "WRITE " &&
           (expected[atEvent]?.getD "").startsWith "PUBLISH " do
      throw (IO.userError "expected interruption between the two stores")
    IO.println "Checked interrupted prefix; publication is the only next event."
    return
  unless runs > 0 && atEvent == expected.size do throw (IO.userError "incomplete heap trace")
  IO.println s!"Checked {operations-runs} heap events and {runs} final statuses/results."

end Verificatum.Heap
