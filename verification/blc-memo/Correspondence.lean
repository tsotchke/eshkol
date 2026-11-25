import Readback

namespace Verificatum.Correspondence

open Verificatum.Lazy Verificatum.Heap Verificatum.Readback

-- Strong readback of a finite head evaluation. Neutral spines store the
-- newest argument first, so their function prefix is quoted before that argument.
inductive Quote : Nat → WH → Term → Prop where
  | lambda : Eval (.delay b (push (.atom (.bound d)) e)) w → Quote (d+1) w n →
      Quote d (.lam b e) (.lam n)
  | head : Quote d (.neutral h []) (.var (index d h))
  | app : Quote d (.neutral h rev) f → Eval x w → Quote d w a →
      Quote d (.neutral h (x::rev)) (.app f a)

-- The default makes decoding total. Every address used by a reachable
-- evaluation has a recorded origin, as witnessed by RefOK or HeapOK.slots.
private def resolve (o : Origins) : Ref → Closure
  | .level d => .atom (.bound d)
  | .addr i => (o i).getD (.atom (.free 0))

private def environment (o : Origins) : Env → Nat → Closure
  | [] => fun i => .atom (.free i)
  | x::xs => push (resolve o x) (environment o xs)

private def value (o : Origins) : Value → WH
  | .lam b env => .lam b (environment o env)
  | .neutral h rev => .neutral h (rev.map (resolve o))

private theorem resolve_ok (ext : Extends o p) (ok : RefOK o r c) : resolve p r = c := by
  cases ok with
  | addr key => simp [resolve,ext _ _ key]
  | level => rfl

private theorem env_ok (ext : Extends o p) (ok : EnvOK o env e) : environment p env = e := by
  induction ok with
  | nil => rfl
  | cons x _ ih => simp [environment,resolve_ok ext x,ih]

private theorem value_ok (ext : Extends o p) (ok : ValueOK o v w) : value p v = w := by
  cases ok with
  | lam env => simp [value,env_ok ext env]
  | neutral refs =>
    simp only [value, WH.neutral.injEq, true_and]
    induction refs with
    | nil => rfl
    | cons x _ ih => simp [resolve_ok ext x,ih]

private def output (o : Origins) (n : Term) : List R → Term → Prop
  | [], t => t = n
  | .wrapLam :: r, t => output o n r (.lam t)
  | .wrapApp f :: r, t => output o n r (.app f t)
  | .quoteArg d x :: r, f => ∃ w a, Eval (resolve o x) w ∧ Quote d w a ∧ output o n r (.app f a)
  | .quoteAt _ :: _, _ => False

private def consume (o : Origins) (n : Term) (r : List R) : List K → WH → Prop
  | [], w => match r with
    | .quoteAt d :: rest => ∃ t, Quote d w t ∧ output o n rest t
    | _ => False
  | .update _ :: k, w => consume o n r k w
  | .arg x :: k, w => ∃ v, Apply w (resolve o x) v ∧ consume o n r k v

private def result (o : Origins) (s : Config) (n : Term) : Prop := match s.ctl with
  | .eval t env => ∃ w, Eval (.delay t (environment o env)) w ∧ consume o n s.readback s.work w
  | .lookup i env => ∃ w, Eval (environment o env i) w ∧ consume o n s.readback s.work w
  | .force x => ∃ w, Eval (resolve o x) w ∧ consume o n s.readback s.work w
  | .value v | .publish _ v => consume o n s.readback s.work (value o v)
  | .emit t => output o n s.readback t

-- Interpret a finite prefix in any extension of its final origin map. This
-- lets earlier continuations refer to the same origins after later allocations.
private theorem backward (st : Step s u) (before : Holds o s) (after : Holds p u)
    (old : Extends o q) (newer : Extends p q) (ok : result q u n) : result q s n := by
  cases st with
  | @alloc i heap f a env k r =>
    have slot := after.1.slots i
    simp [put] at slot
    generalize key : p i = opt at slot
    cases slot with
    | pending he =>
      obtain ⟨w,ev,v,app,rest⟩ := ok
      have key' := newer _ _ key
      simp only [resolve,key',Option.getD_some,← env_ok newer he] at app
      cases app with
      | lam hb => exact ⟨_,.beta ev hb,rest⟩
      | neutral => exact ⟨_,.neutral ev,rest⟩
  | lambda => exact ⟨_,.lam,ok⟩
  | beginLookup => obtain ⟨w,ev,rest⟩ := ok; exact ⟨_,.var ev,rest⟩
  | free => exact ⟨_,.atom,ok⟩
  | first | outer => exact ok
  | @demand heap fresh i k r t env load =>
    have slot := before.1.slots i
    simp only at slot
    rw [load] at slot
    generalize key : o i = opt at slot
    cases slot with
    | pending he =>
      obtain ⟨v,ev,rest⟩ := ok
      exact ⟨_,by simpa [resolve,old _ _ key,← env_ok old he] using ev,rest⟩
  | @hit heap fresh i k r v load =>
    have slot := before.1.slots i
    simp only at slot
    rw [load] at slot
    generalize key : o i = opt at slot
    cases slot with
    | memo hv ev =>
      refine ⟨value q v,?_,ok⟩
      · simpa [resolve,old _ _ key,value_ok old hv] using ev
  | level => exact ⟨_,.atom,ok⟩
  | write | publish => exact ok
  | enterLam => obtain ⟨v,ev,rest⟩ := ok; exact ⟨_,.lam ev,rest⟩
  | enterNeutral => exact ⟨_,.neutral,ok⟩
  | underLam => obtain ⟨_,ev,_,qt,rest⟩ := ok; exact ⟨_,.lambda ev qt,rest⟩
  | head => exact ⟨_,.head,ok⟩
  | spine => obtain ⟨_,qt,_,_,ev,qa,rest⟩ := ok; exact ⟨_,.app qt ev qa,rest⟩
  | wrapLam | wrapApp => exact ok
  | forceArg => obtain ⟨_,ev,_,qa,rest⟩ := ok; exact ⟨_,_,ev,qa,rest⟩

private theorem prefix_result (reach : Reachable t s) :
    ∃ o, Holds o s ∧ ∀ q, Extends o q → ∀ n, result q s n → result q (start t) n := by
  induction reach with
  | init =>
    obtain ⟨o,hold⟩ := reachable_invariant (Reachable.init (t := t))
    exact ⟨o,hold,fun _ _ _ h => h⟩
  | next _ st ih =>
    obtain ⟨o,before,back⟩ := ih
    obtain ⟨p,ext,after⟩ := preserves st before
    refine ⟨p,after,?_⟩
    intro q newer n ok
    have old : Extends o q := fun i c h => newer i c (ext i c h)
    exact back q old n (backward st before after old newer ok)

private theorem terminal_quote (reach : Reachable t s) (stop : terminal s = some n) :
    ∃ w, Eval (initial t) w ∧ Quote 0 w n := by
  obtain ⟨o,_,back⟩ := prefix_result reach
  have ok : result o s n := by
    rcases s with ⟨_,_,ctl,k,r⟩
    cases ctl <;> cases k <;> cases r <;> simp [terminal] at stop
    exact stop
  obtain ⟨w,ev,_,qt,eq⟩ := back o (fun _ _ h => h) n ok
  exact ⟨w,ev,eq ▸ qt⟩

private abbrev Success (s : State Closure) (n : Term) := ∃ fuel, run closureView fuel s = some n
private abbrev Leads (s t : State Closure) := ∀ {n}, Success t n → Success s n

private theorem one (ht : tick closureView s = .inr t) : Leads s t := by
  rintro _ ⟨k,h⟩
  exact ⟨k+1,by simp [run,ht,h]⟩

private theorem lam_view (hl : LamAt c b e) (d : Nat) :
    closureView d c = closureView d (.delay (.lam b) e) := by
  induction hl with
  | direct => rfl
  | var _ ih => exact ih

private theorem atom_view (ha : AtomAt c h) (d : Nat) :
    closureView d c = .var (index d h) := by
  induction ha with
  | direct => rfl
  | var _ ih => exact ih

-- Quantifying over remaining arguments lets readback of a neutral prefix
-- resume at the next argument in the original closure-machine continuation.
private def Quoted (d : Nat) (w : WH) (t : Term) : Prop := match w with
  | .lam b e => ∀ ks, Leads (.seek (d+1) (.delay b (push (.atom (.bound d)) e)) [] (.lam d::ks)) (.emit t ks)
  | .neutral h rev => ∀ more ks, Leads (spine d (.var (index d h)) (rev.reverse ++ more) ks) (spine d t more ks)

private theorem eval_quoted (ev : Eval c w) (qt : Quoted d w t) (ks : List (Frame Closure)) :
    Leads (.seek d c [] ks) (.emit t ks) := by
  obtain ⟨k,⟨hd,args⟩,path,rep⟩ := earned_path ev
  intro n hn
  have atRep : Success (.seek d hd args ks) n := by
    cases w with
    | lam b e =>
      obtain ⟨hl,eq⟩ := rep
      change args = [] at eq
      subst args
      exact one (by simp [tick,lam_view hl d,closureView]) (qt ks hn)
    | neutral h rev =>
      obtain ⟨ha,eq⟩ := rep
      change args = rev.reverse at eq
      subst args
      exact one (t := spine d (.var (index d h)) rev.reverse ks) (by simp [tick,atom_view ha d]) (by simpa using qt [] ks hn)
  obtain ⟨fuel,hf⟩ := atRep
  refine ⟨fuel+k,?_⟩
  simpa [atHead] using (reuse_under_context path d [] ks fuel).trans (by simpa [atHead] using hf)

private theorem quote_run (qt : Quote d w t) : Quoted d w t := by
  induction qt with
  | lambda ev _ ih =>
    intro ks n done
    exact eval_quoted ev ih _ (one (by rfl) done)
  | head => intro _ _ _ h; exact h
  | @app d h rev f x w a _ ev _ ihf iha =>
    intro more ks n done
    have arg := eval_quoted ev iha (.args d f more :: ks) (one (by rfl) done)
    simpa [List.reverse_cons,List.append_assoc] using ihf (x :: more) ks arg

theorem terminal_reference (reach : Reachable t s) (stop : terminal s = some n) :
    ∃ k, reference k t = some n := by
  obtain ⟨w,ev,qt⟩ := terminal_quote reach stop
  exact (normal_order_correspondence t n).mpr (eval_quoted ev (quote_run qt) [] ⟨1,rfl⟩)

theorem register_sound (success : drive (encode (start t)) fuel 0 = (.normal n,used)) :
    ∃ k, reference k t = some n := by
  have budget := (initial_correspondence t fuel _).mp success
  have go {s f u out} (b : Budget s f u out) (reach : Reachable t s)
      (done : out.1 = .normal n) : ∃ k, reference k t = some n := by
    induction b with
    | normal stop => cases done; exact terminal_reference reach stop
    | exhausted => cases done
    | tick _ st _ ih => exact ih (tick_reaches reach st) done
  exact go budget .init rfl


-- Read a successful closure-machine run backwards to recover its finite
-- head evaluations and strong readback derivation.
private inductive SpineNF (d : Nat) : Term → List Closure → Term → Prop where
  | nil : SpineNF d t [] t
  | cons : Eval x w → Quote d w a → SpineNF d (.app f a) xs t → SpineNF d f (x::xs) t

private inductive SeekNF (d : Nat) : WH → List Closure → Term → Prop where
  | nil : Quote d w t → SeekNF d w [] t
  | cons : Apply w x v → SeekNF d v xs t → SeekNF d w (x::xs) t

private def frames (n : Term) : List (Frame Closure) → Term → Prop
  | [], t => t = n
  | .lam _ :: ks, t => frames n ks (.lam t)
  | .args d f xs :: ks, a => ∃ t, SpineNF d (.app f a) xs t ∧ frames n ks t

private def pureResult (n : Term) : State Closure → Prop
  | .emit t ks => frames n ks t
  | .seek d c xs ks => ∃ w t, Eval c w ∧ SeekNF d w xs t ∧ frames n ks t

private theorem spine_result (ok : pureResult n (spine d f xs ks)) :
    ∃ t, SpineNF d f xs t ∧ frames n ks t := by
  cases xs with
  | nil => exact ⟨_,.nil,ok⟩
  | cons x xs =>
    obtain ⟨_,_,ev,sn,rest⟩ := ok
    cases sn with
    | nil qt =>
      obtain ⟨_,sn,done⟩ := rest
      exact ⟨_,.cons ev qt sn,done⟩

private theorem neutral_seek (args : SpineNF d f xs t) (qt : Quote d (.neutral h rev) f) :
    SeekNF d (.neutral h rev) xs t := by
  induction args generalizing rev with
  | nil => exact .nil qt
  | cons ev qa _ ih => exact .cons .neutral (ih (.app qt ev qa))

private theorem seek_back (c : Closure) (ok : match tick closureView (.seek d c xs ks) with
    | .inl t => t = n | .inr s => pureResult n s) : pureResult n (.seek d c xs ks) := by
  induction c generalizing xs ks with
  | atom h =>
    obtain ⟨t,args,done⟩ := spine_result (d := d) (f := .var (index d h)) (xs := xs) (ks := ks) ok
    exact ⟨_,_,.atom,neutral_seek args .head,done⟩
  | delay code e ih =>
    cases code with
    | var i =>
      obtain ⟨w,t,ev,seek,done⟩ := ih i ok
      exact ⟨_,_,.var ev,seek,done⟩
    | app f a =>
      obtain ⟨w,t,ev,seek,done⟩ := ok
      cases seek with
      | cons ap rest =>
        cases ap with
        | lam body => exact ⟨_,_,.beta ev body,rest,done⟩
        | neutral => exact ⟨_,_,.neutral ev,rest,done⟩
    | lam b =>
      cases xs with
      | nil =>
        obtain ⟨w,t,ev,sn,rest⟩ := ok
        cases sn with
        | nil qt => exact ⟨_,_,.lam,.nil (.lambda ev qt),rest⟩
      | cons a xs =>
        obtain ⟨w,t,ev,sn,rest⟩ := ok
        exact ⟨_,_,.lam,.cons (.lam ev) sn,rest⟩

private theorem pure_result (ok : run closureView fuel s = some n) : pureResult n s := by
  induction fuel generalizing s with
  | zero => simp [run] at ok
  | succ k ih =>
    have next : match tick closureView s with
      | .inl t => t = n | .inr u => pureResult n u := by
      cases ht : tick closureView s with
      | inl t => simpa [run,ht] using ok
      | inr u => exact ih (by simpa [run,ht] using ok)
    cases s with
    | seek => exact seek_back _ next
    | emit t ks =>
      cases ks with
      | nil => exact next
      | cons f ks =>
        cases f with
        | lam => exact next
        | args d h xs => exact spine_result (d := d) (f := .app h t) (xs := xs) (ks := ks) next

private theorem reference_quote (run : reference k t = some n) :
    ∃ w, Eval (initial t) w ∧ Quote 0 w n := by
  obtain ⟨_,hr⟩ := (normal_order_correspondence t n).mp ⟨_,run⟩
  obtain ⟨w,t,ev,sn,done⟩ := pure_result hr
  cases sn with
  | nil qt => exact ⟨w,ev,done ▸ qt⟩

-- A finite derivation supplies an upper bound on remaining dispatches.
-- A lookup reserves one tick per skipped binder, a force reserves demand
-- and update, and a cache hit may discard all the reserved evaluation work.
private inductive HeadCost : Closure → WH → Nat → Prop where
  | atom : HeadCost (.atom h) (.neutral h []) 1
  | lam : HeadCost (.delay (.lam b) e) (.lam b e) 1
  | var : HeadCost (e i) w a → HeadCost (.delay (.var i) e) w (a+i+4)
  | beta : HeadCost (.delay f e) (.lam b s) a → HeadCost (.delay b (push (.delay x e) s)) w b' →
      HeadCost (.delay (.app f x) e) w (a+b'+2)
  | neutral : HeadCost (.delay f e) (.neutral h rev) a →
      HeadCost (.delay (.app f x) e) (.neutral h (.delay x e::rev)) (a+2)

private theorem head_cost (ev : Eval c w) : ∃ m, HeadCost c w m := by
  induction ev with
  | atom => exact ⟨_,.atom⟩
  | lam => exact ⟨_,.lam⟩
  | var _ ih => obtain ⟨_,h⟩ := ih; exact ⟨_,.var h⟩
  | beta _ _ ihf ihb => obtain ⟨_,f⟩ := ihf; obtain ⟨_,b⟩ := ihb; exact ⟨_,.beta f b⟩
  | neutral _ ih => obtain ⟨_,f⟩ := ih; exact ⟨_,.neutral f⟩

private theorem head_eval (h : HeadCost c w m) : Eval c w := by
  induction h with
  | atom => exact .atom
  | lam => exact .lam
  | var _ ih => exact .var ih
  | beta _ _ ihf ihb => exact .beta ihf ihb
  | neutral _ ih => exact .neutral ih

private inductive QuoteCost : Nat → WH → Term → Nat → Prop where
  | lam : HeadCost (.delay b (push (.atom (.bound d)) e)) w a → QuoteCost (d+1) w n q →
      QuoteCost d (.lam b e) (.lam n) (a+q+2)
  | head : QuoteCost d (.neutral h []) (.var (index d h)) 1
  | app : QuoteCost d (.neutral h rev) f p → HeadCost x w a → QuoteCost d w t q →
      QuoteCost d (.neutral h (x::rev)) (.app f t) (p+a+q+5)

private theorem quote_cost (qt : Quote d w t) : ∃ m, QuoteCost d w t m := by
  induction qt with
  | lambda ev _ ih => obtain ⟨_,e⟩ := head_cost ev; obtain ⟨_,q⟩ := ih; exact ⟨_,.lam e q⟩
  | head => exact ⟨_,.head⟩
  | app _ ev _ ihf iha =>
    obtain ⟨_,e⟩ := head_cost ev
    obtain ⟨_,f⟩ := ihf
    obtain ⟨_,a⟩ := iha
    exact ⟨_,.app f e a⟩

private inductive OutCost (p : Origins) (n : Term) : List R → Term → Nat → Prop where
  | done : OutCost p n [] n 0
  | lam : OutCost p n r (.lam b) m → OutCost p n (.wrapLam::r) b (m+1)
  | app : OutCost p n r (.app f a) m → OutCost p n (.wrapApp f::r) a (m+1)
  | arg : HeadCost (resolve p x) w e → QuoteCost d w a q → OutCost p n r (.app f a) m →
      OutCost p n (.quoteArg d x::r) f (e+q+m+4)

private inductive WorkCost (p : Origins) (n : Term) (r : List R) : List K → WH → Nat → Prop where
  | quote : r = .quoteAt d::rs → QuoteCost d w t q → OutCost p n rs t m → WorkCost p n r [] w (q+m)
  | update : WorkCost p n r k w m → WorkCost p n r (.update i::k) w (m+1)
  | lam : HeadCost (.delay b (push (resolve p x) e)) w a → WorkCost p n r k w m →
      WorkCost p n r (.arg x::k) (.lam b e) (a+m+1)
  | neutral : WorkCost p n r k (.neutral h (resolve p x::rev)) m →
      WorkCost p n r (.arg x::k) (.neutral h rev) (m+1)

private def pendingCost (p : Origins) (n : Term) (s : Config) (c : Closure) (extra m : Nat) :=
  ∃ w a k, HeadCost c w a ∧ WorkCost p n s.readback s.work w k ∧ m = a+k+extra

private def cost (p : Origins) (s : Config) (n : Term) (m : Nat) := match s.ctl with
  | .eval t env => pendingCost p n s (.delay t (environment p env)) 0 m
  | .lookup i env => pendingCost p n s (environment p env i) (i+3) m
  | .force x => pendingCost p n s (resolve p x) 2 m
  | .value v => WorkCost p n s.readback s.work (value p v) m
  | .emit t => OutCost p n s.readback t m
  | .publish _ _ => False

private theorem ordinary_decreases (st : Step s u) (entry : ∀ i v, s.ctl ≠ .publish i v)
    (exitEntry : ∀ i v, u.ctl ≠ .publish i v) (before : Holds o s) (after : Holds p u)
    (old : Extends o z) (newer : Extends p z) (cert : cost z s n m) :
    ∃ j < m, cost z u n j := by
  cases st with
  | @alloc i heap f x env k r =>
    have slot := after.1.slots i
    simp [put] at slot
    generalize key : p i = opt at slot
    cases slot with
    | pending he =>
      have arg : resolve z (.addr i) = .delay x (environment z env) := by
        simp [resolve,newer _ _ key,env_ok newer he]
      obtain ⟨w,a,b,eval,work,rfl⟩ := cert
      cases eval with
      | beta ef eb =>
        rw [← arg] at eb
        exact ⟨_,by omega,_,_,_,ef,.lam eb work,rfl⟩
      | neutral ef =>
        rw [← arg] at work
        exact ⟨_,by omega,_,_,_,ef,.neutral work,rfl⟩
  | lambda =>
    obtain ⟨_,_,_,ev,work,rfl⟩ := cert
    cases ev; exact ⟨_,by omega,work⟩
  | beginLookup =>
    obtain ⟨_,_,_,ev,work,rfl⟩ := cert
    cases ev with | var e => exact ⟨_,by omega,_,_,_,e,work,rfl⟩
  | free =>
    obtain ⟨_,_,_,ev,work,rfl⟩ := cert
    cases ev; exact ⟨_,by omega,work⟩
  | first | outer =>
    obtain ⟨_,_,_,ev,work,rfl⟩ := cert
    exact ⟨_,by omega,_,_,_,ev,work,rfl⟩
  | @demand heap fresh i k r t env load =>
    have slot := before.1.slots i
    simp only at slot
    rw [load] at slot
    generalize key : o i = opt at slot
    cases slot with
    | pending he =>
      obtain ⟨_,_,_,ev,work,rfl⟩ := cert
      simp [resolve,old _ _ key,← env_ok old he] at ev
      exact ⟨_,by omega,_,_,_,ev,.update work,rfl⟩
  | @hit heap fresh i k r v load =>
    have slot := before.1.slots i
    simp only at slot
    rw [load] at slot
    generalize key : o i = opt at slot
    cases slot with
    | memo hv memo =>
      obtain ⟨_,_,rest,ev,work,rfl⟩ := cert
      simp [resolve,old _ _ key] at ev
      cases eval_unique (head_eval ev) memo
      exact ⟨rest,by omega,by simpa [cost,value_ok old hv] using work⟩
  | level =>
    obtain ⟨_,_,_,ev,work,rfl⟩ := cert
    cases ev; exact ⟨_,by omega,work⟩
  | write => exact (exitEntry _ _ rfl).elim
  | publish => exact (entry _ _ rfl).elim
  | enterLam =>
    cases cert with | lam ev work => exact ⟨_,by omega,_,_,_,ev,work,rfl⟩
  | enterNeutral => cases cert with | neutral work => exact ⟨_,by omega,work⟩
  | underLam =>
    cases cert with
    | quote eq qt out =>
      cases eq
      cases qt with | lam ev qb => exact ⟨_,by omega,_,_,_,ev,.quote rfl qb (.lam out),rfl⟩
  | head =>
    cases cert with
    | quote eq qt out =>
      cases eq
      cases qt; exact ⟨_,by omega,out⟩
  | spine =>
    cases cert with
    | quote eq qt out =>
      cases eq
      cases qt with | app qf ev qa => exact ⟨_,by omega,.quote rfl qf (.arg ev qa out)⟩
  | wrapLam => cases cert with | lam out => exact ⟨_,by omega,out⟩
  | wrapApp => cases cert with | app out => exact ⟨_,by omega,out⟩
  | forceArg =>
    cases cert with | arg ev qa out => exact ⟨_,by omega,_,_,_,ev,.quote rfl qa (.app out),rfl⟩

private theorem tick_decreases (st : Tick s u) (before : Holds o s) :
    ∃ p, Extends o p ∧ Holds p u ∧ ∀ z, Extends p z → ∀ n m, cost z s n m →
      ∃ j < m, cost z u n j := by
  cases st with
  | ordinary one entry nextEntry =>
    obtain ⟨p,ext,after⟩ := preserves one before
    exact ⟨p,ext,after, fun _ newer _ _ => ordinary_decreases one entry nextEntry before after
      (fun i c h => newer i c (ext i c h)) newer⟩
  | update =>
    obtain ⟨p,ext,after⟩ := origins_persist (.next (.next .refl .write) (.publish (by simp [put]))) before
    refine ⟨p,ext,after,?_⟩
    intro z _ n m cert
    cases cert with | update work => exact ⟨_,by omega,work⟩

-- Each tick decreases the bound in every later extension of the origin map.
-- This includes a fused publication tick and a hit that skips a whole derivation.
private theorem bounded_run (bound : Nat) (reach : Reachable t s) (hold : Holds o s)
    (entry : ∀ i v, s.ctl ≠ .publish i v)
    (cert : ∀ z, Extends o z → ∃ m ≤ bound, cost z s n m) (used : Nat) :
    ∃ count, Budget s bound used (.normal n,count) := by
  cases stop : terminal s with
  | some nf =>
    obtain ⟨m,_,cm⟩ := cert o (fun _ _ h => h)
    have eq : nf = n := by
      rcases s with ⟨_,_,ctl,k,r⟩
      cases ctl <;> cases k <;> cases r <;> simp [terminal] at stop
      cases cm; exact stop.symm
    exact ⟨_,.normal (eq ▸ stop)⟩
  | none =>
    obtain ⟨u,st⟩ := progress reach entry stop
    obtain ⟨p,ext,after,dec⟩ := tick_decreases st hold
    have smaller (z) (ez : Extends p z) : ∃ j < bound, cost z u n j := by
      obtain ⟨m,le,cm⟩ := cert z (fun i c h => ez i c (ext i c h))
      obtain ⟨j,lt,hj⟩ := dec z ez n m cm
      exact ⟨j,by omega,hj⟩
    cases bound with
    | zero => obtain ⟨_,lt,_⟩ := smaller p (fun _ _ h => h); omega
    | succ b =>
      have nextEntry : ∀ i v, u.ctl ≠ .publish i v := by cases st with
        | ordinary _ _ h => exact h
        | update => simp
      obtain ⟨count,run⟩ := bounded_run b (tick_reaches reach st) after nextEntry
        (fun z ez => by obtain ⟨j,lt,hj⟩ := smaller z ez; exact ⟨j,by omega,hj⟩) (used+1)
      exact ⟨count,.tick stop st run⟩
termination_by bound

theorem register_complete (success : reference k t = some n) :
    ∃ fuel used, drive (encode (start t)) fuel 0 = (.normal n,used) := by
  obtain ⟨w,ev,qt⟩ := reference_quote success
  obtain ⟨a,e⟩ := head_cost ev
  obtain ⟨q,b⟩ := quote_cost qt
  obtain ⟨o,hold⟩ := reachable_invariant (Reachable.init (t := t))
  obtain ⟨used,run⟩ := bounded_run (a+q) .init hold (by simp [start])
    (fun z _ => ⟨a+q,Nat.le_refl _,_,_,_,e,.quote rfl b .done,rfl⟩) 0
  exact ⟨_,used,budget_result run .init⟩

theorem register_reference_correspondence (t n : Term) :
    (∃ k, reference k t = some n) ↔
    (∃ fuel used, drive (encode (start t)) fuel 0 = (.normal n,used)) := by
  constructor
  · rintro ⟨_,hr⟩; exact register_complete hr
  · rintro ⟨_,_,hr⟩; exact register_sound hr

/-- info: 'Verificatum.Correspondence.register_reference_correspondence' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs (whitespace := lax) in
#print axioms register_reference_correspondence

end Verificatum.Correspondence
