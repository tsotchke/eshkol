import Dispatch

namespace Verificatum.Readback

open Verificatum.Lazy Verificatum.Heap

-- A suspension keeps the binder-depth bound at which its argument was saved.
-- Forcing at a deeper use site must still produce a value valid at that bound.
abbrev Caps := Nat → Option Nat
def safe (c : Caps) (d : Nat) : Ref → Prop
  | .addr i => ∃ b, c i = some b ∧ b ≤ d
  | .level k => k < d
def refs (c : Caps) (d : Nat) (e : List Ref) := ∀ x ∈ e, safe c d x
def validHead (d : Nat) : Head → Prop
  | .bound k => k < d
  | .free _ => True
def wh (c : Caps) (d : Nat) : Value → Prop
  | .lam _ e => refs c d e
  | .neutral h rev => validHead d h ∧ refs c d rev
def cellOK (c : Caps) (d : Nat) : Cell → Prop
  | .pending _ e => refs c d e
  | .writing v | .memo v => wh c d v

inductive Work (c : Caps) : Nat → List K → Nat → Prop where
  | done : d ≤ q → Work c d [] q
  | arg : d ≤ b → safe c b x → Work c b k q → Work c d (.arg x::k) q
  | upd : c i = some b → d ≤ b → Work c d k q → Work c d (.update i::k) q

inductive Output (c : Caps) : Nat → List R → Prop where
  | done : Output c d []
  | lam : Output c d r → Output c (d+1) (.wrapLam::r)
  | arg : safe c d x → Output c d r → Output c d (.quoteArg d x::r)
  | app : Output c d r → Output c d (.wrapApp f::r)

private def extendsCaps (c c' : Caps) := ∀ i b, c i = some b → c' i = some b
private theorem liftRef (ext : extendsCaps c c') (le : d ≤ b) (hx : safe c d x) : safe c' b x := by
  cases x with
  | addr i => obtain ⟨m,hm,lt⟩ := hx; exact ⟨_,ext _ _ hm,Nat.le_trans lt le⟩
  | level i => exact Nat.lt_of_lt_of_le hx le
private theorem liftRefs (ext : extendsCaps c c') (le : d ≤ b) (h : refs c d e) : refs c' b e :=
  fun x mem => liftRef ext le (h x mem)
private theorem liftWh (ext : extendsCaps c c') (le : d ≤ b) (h : wh c d v) : wh c' b v := by
  cases v with
  | lam => exact liftRefs ext le h
  | neutral hd xs =>
    obtain ⟨head,tail⟩ := h
    exact ⟨by cases hd with | free => trivial | bound => exact Nat.lt_of_lt_of_le head le,
           liftRefs ext le tail⟩
private theorem liftCell (ext : extendsCaps c c') (h : cellOK c d cell) : cellOK c' d cell := by
  cases cell <;> first | exact liftRefs ext (Nat.le_refl _) h | exact liftWh ext (Nat.le_refl _) h
private theorem liftWork (ext : extendsCaps c c') (le : a ≤ d) (h : Work c d k q) : Work c' a k q := by
  induction h generalizing a with
  | done bound => exact .done (Nat.le_trans le bound)
  | arg bound hx _ ih => exact .arg (Nat.le_trans le bound) (liftRef ext (Nat.le_refl _) hx) (ih (Nat.le_refl _))
  | upd hc bound _ ih => exact .upd (ext _ _ hc) (Nat.le_trans le bound) (ih le)
private theorem liftOutput (ext : extendsCaps c c') (h : Output c d r) : Output c' d r := by
  induction h with
  | done => exact .done
  | lam _ ih => exact .lam ih
  | arg hx _ ih => exact .arg (liftRef ext (Nat.le_refl _) hx) ih
  | app _ ih => exact .app ih

private def active (c : Caps) (d : Nat) : Control → Prop
  | .eval _ e | .lookup _ e => refs c d e
  | .force x => safe c d x
  | .value v | .publish _ v => wh c d v
  | .emit _ => False

private def ready (c : Caps) (s : Config) :=
  (∃ d t, s.ctl = .emit t ∧ s.work = [] ∧ Output c d s.readback) ∨
  (∃ d q tail, active c d s.ctl ∧ Work c d s.work q ∧
    s.readback = .quoteAt q :: tail ∧ Output c q tail)

structure Scoped (c : Caps) (s : Config) : Prop where
  fresh : ∀ j, s.fresh ≤ j → c j = none
  slots : ∀ i b cell, c i = some b → s.heap i = some cell → cellOK c b cell
  staged : ∀ i v, s.ctl = .publish i v → ∃ b, c i = some b ∧ wh c b v
  control : ready c s

private theorem consRefs (hx : safe c d x) (he : refs c d e) : refs c d (x::e) := by
  intro y mem
  rcases List.mem_cons.mp mem with rfl | tail
  · exact hx
  · exact he _ tail

private theorem updateOK (slots : ∀ j b v, c j = some b → h j = some v → cellOK c b v)
    (slot : ∀ b, c i = some b → cellOK c b val) :
    ∀ j b v, c j = some b → put h i (some val) j = some v → cellOK c b v := by
  intro j b v key load
  by_cases eq : j = i
  · subst j; simp [put] at load; subst v; exact slot _ key
  · exact slots _ _ _ key (by simpa [put,eq] using load)

private theorem keep (st : Step s t) (inv : Scoped c s) : ∃ c', Scoped c' t := by
  obtain ⟨fresh,slots,staged,ok⟩ := inv
  have idExt : extendsCaps c c := fun _ _ h => h
  cases st
  all_goals rcases ok with ⟨d,term,eq,empty,out⟩ | ⟨d,q,tail,act,work,rs,out⟩
  all_goals try cases eq
  all_goals try contradiction
  case alloc.inr n h f a env k r =>
    let c' := put c n (some d)
    have ext : extendsCaps c c' := by
      intro j b hb
      have ne : j ≠ n := by intro eq; subst j; rw [fresh n (Nat.le_refl _)] at hb; contradiction
      simpa [c',put,ne] using hb
    have en := liftRefs ext (Nat.le_refl _) act
    refine ⟨c',?_,?_,?_,?_⟩
    · intro j lt; change n+1 ≤ j at lt; have ne : j ≠ n := by omega
      simpa [c',put,ne] using fresh j (show n ≤ j by omega)
    · intro j b cell key load
      by_cases eq : j = n
      · subst j; simp [c',put] at key load; subst b; subst cell; exact en
      · exact liftCell ext (slots _ _ _ (by simpa [c',put,eq] using key) (by simpa [put,eq] using load))
    · intro _ _ no; cases no
    · exact Or.inr ⟨d,q,tail,en,.arg (Nat.le_refl _) ⟨d,by simp [c',put],Nat.le_refl _⟩
        (liftWork ext (Nat.le_refl _) work),rs,liftOutput ext out⟩
  all_goals try refine ⟨c,fresh,slots,(by intro _ _ no; cases no),?_⟩
  case lambda.inr => exact Or.inr ⟨_,_,_,act,work,rs,out⟩
  case beginLookup.inr => exact Or.inr ⟨_,_,_,act,work,rs,out⟩
  case outer.inr => exact Or.inr ⟨_,_,_,fun x m => act x (List.mem_cons_of_mem _ m),work,rs,out⟩
  case free.inr => exact Or.inr ⟨_,_,_,⟨True.intro,by simp [refs]⟩,work,rs,out⟩
  case first.inr => exact Or.inr ⟨_,_,_,act _ (by simp),work,rs,out⟩
  case level.inr => exact Or.inr ⟨_,_,_,⟨act,by simp [refs]⟩,work,rs,out⟩
  case demand.inr load =>
    obtain ⟨b,key,le⟩ := act
    exact Or.inr ⟨b,_,_,slots _ _ _ key load,.upd key (Nat.le_refl _) (liftWork idExt le work),rs,out⟩
  case hit.inr load =>
    obtain ⟨b,key,le⟩ := act
    exact Or.inr ⟨d,_,_,liftWh idExt le (slots _ _ _ key load),work,rs,out⟩
  case write.inr =>
    cases work with
    | upd key le rest =>
      have hv := liftWh idExt le act
      refine ⟨c,fresh,updateOK slots (fun b hb => ?_),?_,Or.inr ⟨_,_,_,act,rest,rs,out⟩⟩
      · rw [key] at hb; cases hb; exact hv
      · intro _ _ heq; cases heq; exact ⟨_,key,hv⟩
  case publish.inr =>
    obtain ⟨b,key,hv⟩ := staged _ _ rfl
    refine ⟨c,fresh,updateOK slots (fun j hj => ?_),(by intro _ _ no; cases no),Or.inr ⟨_,_,_,act,work,rs,out⟩⟩
    rw [key] at hj; cases hj; exact hv
  case enterLam.inr =>
    cases work with
    | arg le arg rest => exact Or.inr ⟨_,_,_,consRefs arg (liftRefs idExt le act),rest,rs,out⟩
  case enterNeutral.inr n h hd xs x k r =>
    cases work with
    | arg le arg rest =>
      obtain ⟨hh,he⟩ := liftWh (v := .neutral hd xs) idExt le act
      exact Or.inr ⟨_,_,_,⟨hh,consRefs arg he⟩,rest,rs,out⟩
  case underLam.inr n h b e depth r =>
    cases rs; cases work with
    | done le => exact Or.inr ⟨depth+1,depth+1,_,consRefs (x := .level depth) (Nat.lt_succ_self _) (liftRefs idExt (by omega) act),
        .done (Nat.le_refl _),rfl,.lam out⟩
  case head.inr =>
    cases rs; exact Or.inl ⟨_,_,rfl,rfl,out⟩
  case spine.inr n h hd x xs depth r =>
    cases rs; cases work with
    | done le =>
      obtain ⟨hh,he⟩ := liftWh (v := .neutral hd (x::xs)) idExt le act
      exact Or.inr ⟨depth,depth,_,⟨hh,fun x m => he x (List.mem_cons_of_mem _ m)⟩,.done (Nat.le_refl _),rfl,
        .arg (he _ (by simp)) out⟩
  case wrapLam.inl => cases out with | lam o => exact Or.inl ⟨_,_,rfl,rfl,o⟩
  case forceArg.inl => cases out with | arg hx o => exact Or.inr ⟨_,_,_,hx,.done (Nat.le_refl _),rfl,.app o⟩
  case wrapApp.inl => cases out with | app o => exact Or.inl ⟨_,_,rfl,rfl,o⟩


theorem reachable_scoped (reach : Reachable term s) : ∃ c, Scoped c s := by
  induction reach with
  | init => exact ⟨fun _ => none,fun _ _ => rfl,by simp [start],by simp [start],
      Or.inr ⟨0,0,[],by simp [active,start,refs],.done (Nat.le_refl _),rfl,.done⟩⟩
  | next _ st ih => obtain ⟨_,ok⟩ := ih; exact keep st ok

theorem bound_below_quote (reach : Reachable t ⟨n,h,.value (.neutral (.bound j) []),[],.quoteAt depth::r⟩) :
    j < depth := by
  obtain ⟨_,_,_,_,ok⟩ := reachable_scoped reach
  rcases ok with ⟨_,_,no,_⟩ | ⟨d,q,tail,hw,stack,rs,_⟩
  · cases no
  · cases rs; cases stack with | done le => exact Nat.lt_of_lt_of_le hw.1 le

-- One top-level dispatch. Both field writes in the update arm cost a single
-- iteration; the staged heap between them is never a loop entry.
inductive Tick : Config → Config → Prop where
  | ordinary : Step s u → (∀ i v, s.ctl ≠ .publish i v) →
      (∀ i v, u.ctl ≠ .publish i v) → Tick s u
  | update : Tick ⟨n,h,.value v,.update i::k,r⟩
      ⟨n,put (put h i (some (.writing v))) i (some (.memo v)),.value v,k,r⟩

theorem tick_reaches (reach : Reachable t s) (st : Tick s u) : Reachable t u := by
  cases st with
  | ordinary one _ _ => exact .next reach one
  | update => exact .next (.next reach .write) (.publish (by simp [put]))

def terminal : Config → Option Term
  | ⟨_,_,.emit t,[],[]⟩ => some t
  | _ => none

theorem progress (reach : Reachable t s) (entry : ∀ i v, s.ctl ≠ .publish i v)
    (running : terminal s = none) : ∃ u, Tick s u := by
  obtain ⟨c,_,_,_,ok⟩ := reachable_scoped reach
  rcases s with ⟨n,h,ctl,k,r⟩
  rcases ok with ⟨d,term,he,hk,out⟩ | ⟨d,q,tail,act,work,rs,out⟩
  · cases he; cases hk; cases out
    · cases running
    all_goals exact ⟨_,.ordinary (by constructor) (by simp) (by simp)⟩
  · cases rs
    cases ctl with
    | publish i v => exact (entry i v rfl).elim
    | emit => contradiction
    | eval term env =>
      cases term <;> exact ⟨_,.ordinary (by constructor) (by simp) (by simp)⟩
    | lookup i env =>
      cases env with
      | nil => exact ⟨_,.ordinary .free (by simp) (by simp)⟩
      | cons => cases i <;> exact ⟨_,.ordinary (by constructor) (by simp) (by simp)⟩
    | force x =>
      cases x with
      | level => exact ⟨_,.ordinary .level (by simp) (by simp)⟩
      | addr i =>
        rcases force_finds_live_cell reach rfl with ⟨_,_,load⟩ | ⟨_,load⟩
        · exact ⟨_,.ordinary (.demand load) (by simp) (by simp)⟩
        · exact ⟨_,.ordinary (.hit load) (by simp) (by simp)⟩
    | value v =>
      cases k with
      | cons k ks => cases k with
        | update => exact ⟨_,.update⟩
        | arg => cases v <;> exact ⟨_,.ordinary (by constructor) (by simp) (by simp)⟩
      | nil => cases v with
        | lam => exact ⟨_,.ordinary .underLam (by simp) (by simp)⟩
        | neutral _ xs => cases xs <;> exact ⟨_,.ordinary (by constructor) (by simp) (by simp)⟩

-- One physical frame list. Addresses below are already abstracted to the
-- stable logical IDs, and Term abstracts the decoded immutable syntax pairs.
open Dispatch
inductive StackItem where
  | arg (x : Pointer Nat)
  | update (i : Nat)
  | quoteAt (d : Nat)
  | wrapLam
  | quoteArg (d : Nat) (x : Pointer Nat)
  | wrapApp (f : Term)

def wk : K → StackItem
  | .arg x => .arg (pointer id x)
  | .update i => .update i

def rk : R → StackItem
  | .quoteAt d => .quoteAt d
  | .wrapLam => .wrapLam
  | .quoteArg d x => .quoteArg d (pointer id x)
  | .wrapApp f => .wrapApp f

inductive PC where
  | eval (t : Term) (env : List (Pointer Nat))
  | lookup (i : Int) (env : List (Pointer Nat))
  | force (x : Pointer Nat)
  | ret (v : RawValue Nat)
  | emit (t : Term)
  | between

def pc : Control → PC
  | .eval t e => .eval t (environment id e)
  | .lookup i e => .lookup (Int.ofNat (i+1)) (environment id e)
  | .force x => .force (pointer id x)
  | .value v => .ret (value id v)
  | .emit t => .emit t
  | .publish .. => .between

structure Registers where
  fresh : Nat
  heap : RawHeap Nat
  mode : PC
  frames : List StackItem

def encode (s : Config) : Registers :=
  ⟨s.fresh,fun i => (s.heap i).map (Publication.encode (boxing id)),pc s.ctl,s.work.map wk ++ s.readback.map rk⟩

def dispatch (s : Registers) : Option Registers :=
  let go := fun mode frames => some {s with mode,frames}
  match s.mode, s.frames with
  | .eval (.var j) e, ks => go (.lookup (Int.ofNat (j+1)) e) ks
  | .eval (.lam b) e, ks => go (.ret (.closure b e)) ks
  | .eval (.app f a) e, ks => some ⟨s.fresh+1,put s.heap s.fresh (some (.pending,.suspension a e)),
                                    .eval f e,.arg (.cell s.fresh)::ks⟩
  | .lookup j e, ks => do
      match ← Lookup.dispatch Lookup.lookupArm j e with
      | .lookup i env => go (.lookup i env) ks
      | .force x => go (.force x) ks
      | .value i => go (.ret (.integer i)) ks
  | .force (.level d), ks => go (.ret (.integer (Int.ofNat d))) ks
  | .force (.cell i), ks =>
      match s.heap i with
      | some (.pending,.suspension t e) => go (.eval t e) (.update i::ks)
      | some (.memo,.result v) => go (.ret v) ks
      | _ => none
  | .ret v, .update i::ks => some ⟨s.fresh,put s.heap i (some (.memo,.result v)),.ret v,ks⟩
  | .ret (.closure b e), .arg x::ks => go (.eval b (x::e)) ks
  | .ret v, .arg x::ks => go (.ret (.spine v x)) ks
  | .ret (.integer j), .quoteAt d::ks =>
      let idx := Int.ofNat d - j
      if idx > 0 then go (.emit (.var (idx.toNat-1))) ks else none
  | .ret (.closure b e), .quoteAt d::ks => go (.eval b (.level d::e)) (.quoteAt (d+1):: .wrapLam::ks)
  | .ret (.spine f x), .quoteAt d::ks => go (.ret f) (.quoteAt d:: .quoteArg d x::ks)
  | .emit t, .wrapLam::ks => go (.emit (.lam t)) ks
  | .emit t, .quoteArg d x::ks => go (.force x) (.quoteAt d:: .wrapApp t::ks)
  | .emit t, .wrapApp f::ks => go (.emit (.app f t)) ks
  | _, _ => none

private theorem overwrite (h : Nat → α) (i : Nat) (a b : α) :
    put (put h i a) i b = put h i b := by funext j; by_cases eq : j = i <;> simp [put,eq]

theorem simulates (reach : Reachable t s) (st : Tick s u) : dispatch (encode s) = some (encode u) := by
  cases st with
  | update =>
    rename_i n h v i k r
    simp [dispatch,encode,pc,wk,overwrite]
    funext j; by_cases eq : j = i <;> simp [put,eq,Publication.encode,boxing]
  | ordinary one before after =>
    cases one with
    | write => exact (after _ _ rfl).elim
    | publish => exact (before _ _ rfl).elim
    | alloc =>
      rename_i n h f a env k r
      simp [dispatch,encode,pc,wk,pointer]
      funext j; by_cases eq : j = n <;> simp [put,eq,Publication.encode,boxing]
    | outer =>
      rename_i i a e k r
      have ne : (i : Int)+1+1 ≠ 1 := by omega
      simp [dispatch,encode,pc,Lookup.dispatch,Lookup.lookupArm,Lookup.tailCall,environment,ne]
    | demand load => simp [dispatch,encode,pc,wk,pointer,load,Publication.encode,boxing]
    | hit load => simp [dispatch,encode,pc,pointer,load,Publication.encode,boxing]
    | enterNeutral => rename_i a xs x k r; cases xs <;> cases a <;> rfl
    | head =>
      rename_i hd depth _
      cases hd with
      | free i =>
        simp [dispatch,encode,pc,rk,value,neutral,index]; omega
      | bound j =>
        have scope := bound_below_quote reach
        simp [dispatch,encode,pc,rk,value,neutral,index]; omega
    | free | level | first | lambda | beginLookup | enterLam | underLam | spine | wrapLam | forceArg | wrapApp => rfl


def atEnd (s : Registers) : Option Term := match s.mode,s.frames with
  | .emit t,[] => some t
  | _,_ => none

theorem dispatch_refines (reach : Reachable t s) (entry : ∀ i v, s.ctl ≠ .publish i v)
    (stepTaken : dispatch (encode s) = some raw) :
    ∃ u, Tick s u ∧ Reachable t u ∧ encode u = raw := by
  have running : terminal s = none := by
    cases stop : terminal s with
    | none => rfl
    | some t =>
      rcases s with ⟨n,h,ctl,k,r⟩
      cases ctl <;> cases k <;> cases r <;> simp [terminal] at stop
      cases stepTaken
  obtain ⟨u,one⟩ := progress reach entry running
  exact ⟨u,one,tick_reaches reach one,Option.some.inj ((simulates reach one).symm.trans stepTaken)⟩

theorem same_end (s : Config) : atEnd (encode s) = terminal s := by
  rcases s with ⟨n,h,ctl,k,r⟩
  cases ctl <;> cases k <;> cases r <;> rfl

inductive Answer where
  | normal (term : Term)
  | exhausted
  | stuck
deriving DecidableEq

-- The count is the executed dispatch count, including environment hops and
-- readback. The terminal guard precedes fuel exhaustion even on an exact tie.
def drive (s : Registers) (fuel used : Nat) : Answer × Nat :=
  match atEnd s with
  | some t => (.normal t,used)
  | none => match fuel with
    | 0 => (.exhausted,used)
    | f+1 => match dispatch s with
      | none => (.stuck,used)
      | some u => drive u f (used+1)

inductive Budget : Config → Nat → Nat → Answer × Nat → Prop where
  | normal : terminal s = some t → Budget s fuel used (.normal t,used)
  | exhausted : terminal s = none → Budget s 0 used (.exhausted,used)
  | tick : terminal s = none → Tick s u → Budget u f (used+1) result →
      Budget s (f+1) used result

private theorem next_entry (one : Tick s u) : ∀ i v, u.ctl ≠ .publish i v := by
  cases one with
  | ordinary _ _ bound => exact bound
  | update => simp

theorem exact_budget (reach : Reachable t s) (entry : ∀ i v, s.ctl ≠ .publish i v) :
    Budget s fuel used (drive (encode s) fuel used) := by
  induction fuel generalizing s used with
  | zero =>
    cases stop : terminal s with
    | some nf => simpa [drive,same_end,stop] using (Budget.normal (fuel := 0) (used := used) stop)
    | none => simpa [drive,same_end,stop] using (Budget.exhausted (used := used) stop)
  | succ f ih =>
    cases stop : terminal s with
    | some nf => simpa [drive,same_end,stop] using (Budget.normal (fuel := f+1) (used := used) stop)
    | none =>
      obtain ⟨u,one⟩ := progress reach entry stop
      simpa [drive,same_end,stop,simulates reach one] using
        Budget.tick stop one (ih (tick_reaches reach one) (next_entry one) (used := used+1))

theorem budget_result (run : Budget s fuel used answer) (reach : Reachable t s) :
    drive (encode s) fuel used = answer := by
  induction run with
  | @normal s nf f used stop => cases f <;> simp [drive,same_end,stop]
  | exhausted stop => simp [drive,same_end,stop]
  | tick stop one _ ih => simpa [drive,same_end,stop,simulates reach one] using ih (tick_reaches reach one)

theorem initial_correspondence (term : Term) (fuel : Nat) (out : Answer × Nat) :
    drive (encode (start term)) fuel 0 = out ↔ Budget (start term) fuel 0 out := by
  constructor
  · rintro rfl; exact exact_budget .init (by simp [start])
  · intro run; exact budget_result run .init

theorem no_stuck (run : Budget s fuel used answer) : answer.1 ≠ .stuck := by
  induction run <;> simp_all

theorem exhausted_count (run : Budget s fuel used answer) (cut : answer.1 = .exhausted) :
    answer.2 = used+fuel := by
  induction run with
  | normal => cases cut
  | exhausted => simp
  | tick _ _ _ ih => have := ih cut; omega

-- A neutral function is required while rebuilding the left side of an
-- application. Merely knowing that both children are normal is insufficient.
private def neutralTerm (t : Term) := step t = none ∧ ∀ b, t ≠ .lam b
private def syntaxOK (t : Term) (n : Bool) := step t = none ∧ (n = true → neutralTerm t)
private inductive Kont : Bool → List R → Prop where
  | done : Kont n []
  | lam : Kont false r → Kont n (.wrapLam::r)
  | app : neutralTerm f → Kont true r → Kont n (.wrapApp f::r)
  | arg : Kont true r → Kont true (.quoteArg d x::r)

private theorem stronger (k : Kont n r) : Kont true r := by cases k <;> constructor <;> assumption

private def rebuilding (s : Config) :=
  (∃ t b, s.ctl = .emit t ∧ s.work = [] ∧ syntaxOK t b ∧ Kont b s.readback) ∨
  (∃ d r, s.readback = .quoteAt d::r ∧
    (Kont false r ∨ (∃ h xs, s.ctl = .value (.neutral h xs) ∧ s.work = [] ∧ Kont true r)) ∧
    (∀ t, s.ctl ≠ .emit t))

private theorem rebuild_step (p : Step s t) (ok : rebuilding s) : rebuilding t := by
  cases p <;> rcases ok with ⟨term,b,eq,empty,normal,cont⟩ | ⟨d,r,rs,kind,active⟩
  all_goals try cases eq
  all_goals try exact False.elim (active _ rfl)
  all_goals try (rcases kind with k | ⟨hd,xs,no,empty,tail⟩
                 · exact Or.inr ⟨_,_,rs,Or.inl k,by simp⟩
                 · first | (solve | cases empty) | (solve | cases no))
  case underLam.inr =>
    cases rs
    rcases kind with tail | ⟨_,_,impossible,_⟩
    · exact Or.inr ⟨_,_,rfl,Or.inl (.lam tail),by simp⟩
    · cases impossible
  case head.inr n h hd depth tailStack =>
    cases rs
    have tail : Kont true tailStack := by rcases kind with k | ⟨_,_,_,_,k⟩ <;> exact stronger k
    exact Or.inl ⟨_,true,rfl,rfl,⟨rfl,fun _ => ⟨rfl,by simp⟩⟩,tail⟩
  case spine.inr n h hd x xs depth tailStack =>
    cases rs
    have tail : Kont true tailStack := by rcases kind with k | ⟨_,_,_,_,k⟩ <;> exact stronger k
    exact Or.inr ⟨_,_,rfl,Or.inr ⟨_,_,rfl,rfl,.arg tail⟩,by simp⟩
  case wrapLam.inl =>
    cases cont with | lam rest =>
      exact Or.inl ⟨_,false,rfl,rfl,⟨by simp [step,normal.1],by simp⟩,rest⟩
  case forceArg.inl =>
    cases cont with | arg rest =>
      exact Or.inr ⟨_,_,rfl,Or.inl (.app (normal.2 rfl) rest),by simp⟩
  case wrapApp.inl n h a f r =>
    cases cont with | app nf rest =>
      have good : neutralTerm (.app f a) := by
        refine ⟨?_,by simp⟩
        obtain ⟨fn,head⟩ := nf
        cases f <;> simp_all [step,normal.1]
      exact Or.inl ⟨_,true,rfl,rfl,⟨good.1,fun _ => good⟩,rest⟩

theorem terminal_normal (reach : Reachable t s) (stop : terminal s = some nf) : step nf = none := by
  have ok : rebuilding s := by
    clear stop
    induction reach with
    | init => exact Or.inr ⟨0,[],rfl,Or.inl .done,by simp [start]⟩
    | next _ one ih => exact rebuild_step one ih
  rcases s with ⟨_,_,ctl,k,r⟩
  cases ctl <;> cases k <;> cases r <;> simp [terminal] at stop
  rcases ok with ⟨term,_,he,_,good,_⟩ | ⟨_,_,bad,_⟩
  · cases he; subst nf; exact good.1
  · cases bad

theorem returns_normal (run : Budget s fuel used result) (reach : Reachable t s)
    (finished : result.1 = .normal nf) : step nf = none := by
  induction run with
  | normal stop => cases finished; exact terminal_normal reach stop
  | exhausted => cases finished
  | tick _ one _ ih => exact ih (tick_reaches reach one) finished

/-- info: 'Verificatum.Readback.initial_correspondence' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms initial_correspondence
/-- info: 'Verificatum.Readback.returns_normal' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms returns_normal
/-- info: 'Verificatum.Readback.dispatch_refines' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms dispatch_refines

private def bits : Term → String
  | .var i => String.ofList (List.replicate (i+1) '1') ++ "0"
  | .lam t => "00" ++ bits t
  | .app a b => "01" ++ bits a ++ bits b

def audit (file : System.FilePath) : IO Unit := do
  let mut pending : Option String := none
  let mut calls := 0
  for line in (← IO.FS.lines file) do
    if line.startsWith "START " then
      unless pending.isNone do throw (IO.userError "overlapping calls")
      let ["START",bound,input] := line.splitOn " " | throw (IO.userError "invalid input")
      let some t := decode input | throw (IO.userError "invalid term")
      let some fuel := bound.toNat? | throw (IO.userError "invalid budget")
      let (answer,used) := drive (encode (start t)) fuel 0
      match answer with
      | .stuck => throw (IO.userError "register model stuck")
      | .normal nf => pending := some s!"STOP normal {used} {bits nf}"
      | .exhausted => pending := some s!"STOP exhausted {used}"
    else if line.startsWith "STOP " then
      unless pending == some line do throw (IO.userError s!"register result mismatch: {line}")
      calls := calls+1
      pending := none
  unless pending.isNone && calls > 0 do throw (IO.userError "incomplete register replay")
  IO.println s!"Checked {calls} flat-stack results with exact dispatch fuel."

end Verificatum.Readback
