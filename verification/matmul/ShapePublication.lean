namespace Verificatum.ShapePublication

/-- Rank two. The model trusts the element capacity supplied when creating a draft. -/
structure Axes where
  rows : Nat
  cols : Nat
deriving DecidableEq, Repr

inductive Buffer where
  | unused
  | draft (address : Nat) (axes : Axes) (capacity : Nat)
  | published (address : Nat) (axes : Axes)
  | exposed (address : Nat) (axes : Axes)
  | retired
deriving DecidableEq

abbrev World := Nat → Buffer
abbrev Headers := Nat → Option Nat

def put (w : World) (key : Nat) (value : Buffer) : World :=
  fun k => if k = key then value else w k

/-- Native addresses are observations. World keys are logical identities which
    remain distinct after retirement, including when an address is reused. -/
inductive Change : Buffer → Buffer → Prop where
  | allocate : Change .unused (.draft p a capacity)
  | write : Change (.draft p a capacity) (.draft p b capacity)
  | publish (fits : a.rows * a.cols ≤ capacity) : Change (.draft p a capacity) (.published p a)
  | relocate : Change (.published p a) (.published q a)
  | expose : Change (.published p a) (.exposed p a)
  | exposeDraft : Change (.draft p a capacity) (.exposed p a)
  | foreignWrite : Change (.exposed p a) (.exposed p b)
  | retire : Change s .retired

inductive Step : World → World → Prop where
  | update (h : Change (w key) s) : Step w (put w key s)
  | copy (src : w i = .published p a) (fresh : w j = .unused) :
      Step w (put w j (.draft q a capacity))

inductive Steps : World → World → Prop where
  | refl : Steps w w
  | snoc : Steps w v → Step v u → Steps w u

/-- Exposure or retirement permanently removes an identity from proof use. -/
def Preserves (a : Axes) : Buffer → Prop
  | .published _ b => b = a
  | .exposed .. | .retired => True
  | _ => False

theorem local_preserves (h : Change s t) (ha : Preserves a s) : Preserves a t := by
  cases h <;> simp_all [Preserves]

private theorem step_preserves (h : Step w v) (ha : Preserves a (w k)) :
    Preserves a (v k) := by
  cases h with
  | update hc =>
    rename_i s key
    by_cases same : k = key
    · subst k
      simpa [put] using local_preserves hc ha
    · simpa [put, same] using ha
  | copy src fresh =>
    simp only [put]
    split
    · rename_i same
      simp [same, fresh, Preserves] at ha
    · exact ha

theorem published_history (start : w key = .published p a) (steps : Steps w v) :
    Preserves a (v key) := by
  induction steps with
  | refl => simp [start, Preserves]
  | snoc _ next ih => exact step_preserves next ih

structure Binding where
  header : Nat
  shape : Nat

/-- The use site supplies the current header identity and resolution. A physical
    address can change between issuance and use. -/
def Bound (w : World) (headers : Headers) (actual : Nat) (b : Binding) (a : Axes) : Prop :=
  actual = b.header ∧ headers actual = some b.shape ∧ ∃ p, w b.shape = .published p a

theorem bound_contents_at_use
    (issued : Bound w hs h b a) (steps : Steps w v) (live : Bound v hs' h' b actualAxes) :
    actualAxes = a := by
  obtain ⟨_, _, p, hp⟩ := issued
  obtain ⟨_, _, q, hq⟩ := live
  have preserved := published_history hp steps
  simpa [hq, Preserves] using preserved

theorem inner_guard_at_use
    (left : Bound w hs l bl a) (right : Bound w hs r br b)
    (eqAtIssue : a.cols = b.rows) (steps : Steps w v)
    (leftLive : Bound v hs' l' bl a') (rightLive : Bound v hs' r' br b')
    (sizeOK : Bool) : (decide (a'.cols = b'.rows) && sizeOK) = sizeOK := by
  rw [bound_contents_at_use left steps leftLive, bound_contents_at_use right steps rightLive]
  simp [eqAtIssue]

/-- Two header IDs still resolve to one object after exposure and a write.
    Neither can obtain a live published binding, even with its new, true axes. -/
theorem exposed_refuses_all_headers (hx : w key = .exposed p changed) :
    ¬ Bound w hs h ⟨h, key⟩ observed := by
  rintro ⟨_, _, q, hq⟩
  rw [hx] at hq
  contradiction

private def original : Axes := ⟨3, 2⟩
private def changed : Axes := ⟨1, 2⟩
private def empty : World := fun _ => .unused
private def draft := put empty 7 (.draft 100 original 6)
private def sealedWorld := put draft 7 (.published 100 original)
private def moved := put sealedWorld 7 (.published 300 original)
private def escaped := put moved 7 (.exposed 300 original)
private def overwritten := put escaped 7 (.exposed 300 changed)
private def aliases : Headers := fun h => if h = 11 ∨ h = 12 then some 7 else none

private theorem published_at_7 : Steps empty sealedWorld :=
  .snoc (.snoc .refl (.update (key := 7) Change.allocate))
        (.update (key := 7) (Change.publish (by decide : 3 * 2 ≤ 6)))

private theorem moved_after_publication : Step sealedWorld moved :=
  .update (key := 7) Change.relocate

private theorem alias_write_history : Steps moved overwritten :=
  .snoc (.snoc .refl (.update (key := 7) Change.expose))
        (.update (key := 7) Change.foreignWrite)

example : Bound moved aliases 11 ⟨11, 7⟩ original ∧ Bound moved aliases 12 ⟨12, 7⟩ original :=
  ⟨⟨rfl, rfl, 300, rfl⟩, ⟨rfl, rfl, 300, rfl⟩⟩

example : ¬ Bound overwritten aliases 11 ⟨11, 7⟩ original ∧
          ¬ Bound overwritten aliases 12 ⟨12, 7⟩ changed :=
  ⟨exposed_refuses_all_headers (by rfl), exposed_refuses_all_headers (by rfl)⟩

-- This is a reference shape-ownership policy. Arbitrary address labels make
-- physical sharing observable independently of axis equality.
private def relocateAt (w : World) (k : Nat) : World :=
  match w k with
  | .published _ a => put w k (.published (300 + k) a)
  | _ => w

private theorem relocateAt_steps (w : World) (k : Nat) : Steps w (relocateAt w k) := by
  unfold relocateAt
  split
  next p a h =>
    exact .snoc .refl (.update (h ▸ Change.relocate))
  all_goals exact .refl

def observation (shared older : Bool) : String :=
  let leftKey := 7
  let rightKey := if shared then leftKey else 8
  let before := put (put empty leftKey (.published (100 + leftKey) original))
                    rightKey (.published (100 + rightKey) original)
  let afterLeft := if older then before else relocateAt before leftKey
  let after := if older || shared then afterLeft else relocateAt afterLeft rightKey
  match before leftKey, after leftKey, after rightKey with
  | .published old _, .published l a, .published r b =>
    s!"{l == r} {l == old} {a.rows} {a.cols} {b.rows} {b.cols}"
  | _, _, _ => "outside publication contract"

end Verificatum.ShapePublication

def main : IO Unit := do
  for route in ["batch", "separate"] do
    for (name, shared, older) in [("shared-young", true, false), ("shared-old", true, true),
                                  ("distinct-young", false, false)] do
      IO.println s!"{route} {name} {Verificatum.ShapePublication.observation shared older}"
