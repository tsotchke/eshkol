import ArenaLedger
import NativeAllocation

namespace Verificatum.Reclamation
open ArenaLedger

abbrev Extent := NativeEffect.Allocation.Span

-- The released interval is [base+mark, base+used). Includes alignment padding.
def Outside (s : State) (mark : Nat) (x : Extent) : Prop :=
  x.ends ≤ s.base + mark ∨ s.base + s.used ≤ x.base

instance : Decidable (Outside s m x) := inferInstanceAs (Decidable (_ ∨ _))

def fitsPrefix (s : State) (mark : Nat) (g : Grant) := decide (g.base + g.size ≤ s.base + mark)

def after (s : State) (mark : Nat) : State :=
  {s with used := mark, grants := s.grants.filter (fitsPrefix s mark)}

-- No split grants, and no surviving ledger view may lose its owner.
def MetadataOK (s : State) (mark : Nat) : Prop :=
  s.floor ≤ mark ∧ mark ≤ s.used ∧
  (∀ g ∈ s.grants, fitsPrefix s mark g = true ∨ s.base + mark ≤ g.base) ∧
  (∀ v ∈ s.roots ++ s.retained, fitsPrefix s mark v.owner = true)

instance : Decidable (MetadataOK s m) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ (∀ g ∈ _, _ ∨ _) ∧ (∀ v ∈ _, _)))

theorem preserved (valid : ArenaValid s) (ok : MetadataOK s m) : ArenaValid (after s m) := by
  obtain ⟨floor,limit,_,live⟩ := ok
  refine ⟨valid.baseBound, valid.capBound, floor, Nat.le_trans limit valid.cursor, ?_, ?_, ?_, ?_⟩
  · intro g hg
    obtain ⟨mem,fit⟩ := List.mem_filter.mp hg
    obtain ⟨a,b,c,d,_⟩ := valid.covered g mem
    exact ⟨a,b,c,d,of_decide_eq_true fit⟩
  · exact valid.separated.filter _
  · exact (List.filter_sublist (p := fitsPrefix s m) (l := s.grants)).map Grant.generation |>.nodup valid.generations
  · intro v hv
    obtain ⟨mem,bounds⟩ := valid.views v hv
    exact ⟨List.mem_filter.mpr ⟨mem,live v hv⟩,bounds⟩

-- The implementation supplies the interpretation of all references that can
-- survive this operation. Unenumerated holder classes cannot be silently empty.
structure Evidence (s : State) (mark : Nat) (survives : Extent → Prop) where
  accounted : Extent → Prop
  complete : ∀ x, survives x → accounted x
  outside : ∀ x, accounted x → Outside s mark x

structure Permit (s : State) (mark : Nat) (survives : Extent → Prop) where
  state : State
  result : state = after s mark
  valid : ArenaValid state
  safe : ∀ x, survives x → Outside s mark x
  revoked : ∀ g ∈ s.grants, g ∉ state.grants → s.base + mark ≤ g.base
  generation : state.next = s.next

def authorize (s : Checked) (mark : Nat) {survives : Extent → Prop}
    (e : Evidence s.state mark survives) : Option (Permit s.state mark survives) :=
  if h : MetadataOK s.state mark then
    some ⟨after s.state mark, rfl, preserved s.valid h, fun x hx => e.outside x (e.complete x hx),
      fun g hg gone => (h.2.2.1 g hg).resolve_left (fun keep => gone (List.mem_filter.mpr ⟨hg,keep⟩)), rfl⟩
  else none

theorem authorize_exists (s : Checked) (h : MetadataOK s.state m)
    (e : Evidence s.state m survives) : (authorize s m e).isSome := by
  simp [authorize,h]

-- Partial reclamation also preserves revocation across later checked
-- allocations and nested rewinds. A reset of the ledger is outside this history.
inductive History : State → State → Prop where
  | begin : History s s
  | bump : History s t → admit t n a p u = some v → History s v
  | rewind : History s t → MetadataOK t mark → History s (after t mark)

theorem stays_revoked (path : History s t) (valid : ArenaValid s)
    (old : g.generation < s.next) (gone : g ∉ s.grants) :
    ArenaValid t ∧ g.generation < t.next ∧ g ∉ t.grants := by
  induction path with
  | begin => exact ⟨valid,old,gone⟩
  | bump prev ok ih =>
    exact revoked_stays_revoked (.bump .refl ok) ih.1 ih.2.1 ih.2.2
  | rewind prev ok ih =>
    exact ⟨preserved ih.1 ok, ih.2.1, fun h => ih.2.2 (List.mem_filter.mp h).1⟩

-- A finite list establishes safety for its actual entries. Extending that
-- result to a native holder set requires a separate completeness argument.
def recorded (s : Checked) (m : Nat) (xs : List Extent) :
    Option (Permit s.state m (· ∈ xs)) :=
  if h : ∀ x ∈ xs, Outside s.state m x then
    authorize s m ⟨(· ∈ xs), fun _ captured => captured,h⟩
  else none

-- Keep an earlier live interior view while revoking a later grant. A cut
-- through an object or a surviving slot into released padding must fail.
private def partialControl : Bool := Id.run do
  let some start := beginChecked 1 4096 1024 0 | return false
  let some a := start.bump 8 8 4096 8 | return false
  let some rooted := a.keep ⟨issued start.state 8 8,2,2⟩ | return false
  let some b := rooted.bump 16 64 4160 128 | return false
  let some good := recorded b 8 [⟨4098,2⟩] | return false
  return decide (good.state.grants = a.state.grants) && good.state.next == 2 &&
    (recorded b 68 []).isNone && (recorded b 8 [⟨4107,1⟩]).isNone &&
    (recorded b 0 []).isNone

#guard partialControl

end Verificatum.Reclamation
