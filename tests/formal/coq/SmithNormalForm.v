(* ------------------------------------------------------------------------
   Smith normal form: what the invariant factors determine.

   nLab claims (tests/nlab/nlab_ring_test.esk, nlab_space_test.esk):
     [Smith normal form#1] "There exist invertible matrices P and Q such that
                            the product matrix PAQ is" diagonal;
     [Smith normal form#2] "each a_i divides a_{i+1}";
     [real projective space#2] H_*(RP^2; Z) = (Z, Z/2, 0), computed by Eshkol
                            from the 6-vertex triangulation.

   General theorems, for integer matrices of any size m x n with
   U A V = D, U' U = I and V V' = I (U, V invertible over Z):
     image_iff            z is in the image of A  <->  U z is in the image of D;
     diag_image           for diagonal D the image is cut out by divisibility:
                          (D w)_i = v_i solvable  <->  D_ii | v_i (and v_i = 0
                          beyond the n columns);
     kernel_trivial       nonzero diagonal entries in every column => ker A = 0;
     divisors_invariant   d divides every entry of A <-> d divides every entry of D;
     first_invariant_factor  d divides every entry of A <-> d | D_00, when D_00
                          divides the diagonal: the first invariant factor is
                          the gcd of the entries, determined by A alone.
   Instances (certificates proposed by gen_snf_certificates.py, every identity
   re-checked here by computation):
     the six matrices of [Smith normal form#1/#2], with their diagonals and the
     divisibility chain; and the integral homology of the 6-vertex RP^2:
     H_0 = Z, H_1 = Z/2 (a cycle bounds iff one coordinate is even, and an odd
     cycle exists), H_2 = 0.
   ------------------------------------------------------------------------ *)
From Stdlib Require Import ZArith List Lia Bool.
Import ListNotations.
Open Scope Z_scope.

(* ---------------------------------------------------------------- sums *)

Fixpoint sumn (n : nat) (f : nat -> Z) : Z :=
  match n with O => 0 | S k => sumn k f + f k end.

Lemma sumn_ext n f g :
  (forall i, (i < n)%nat -> f i = g i) -> sumn n f = sumn n g.
Proof.
  induction n; simpl; intros H; [reflexivity|].
  rewrite IHn by (intros; apply H; lia). rewrite H by lia. reflexivity.
Qed.

Lemma sumn_add n f g : sumn n (fun i => f i + g i) = sumn n f + sumn n g.
Proof. induction n; simpl; [reflexivity|]. rewrite IHn. ring. Qed.

Lemma sumn_scale n c f : sumn n (fun i => c * f i) = c * sumn n f.
Proof. induction n; simpl; [ring|]. rewrite IHn. ring. Qed.

Lemma sumn_zero n : sumn n (fun _ => 0) = 0.
Proof. induction n; simpl; [reflexivity|]. rewrite IHn. ring. Qed.

Lemma sumn_swap n m f :
  sumn n (fun i => sumn m (fun j => f i j)) = sumn m (fun j => sumn n (fun i => f i j)).
Proof.
  induction n; simpl.
  - symmetry. apply sumn_zero.
  - rewrite IHn, sumn_add. reflexivity.
Qed.

Lemma sumn_delta n i a x :
  sumn n (fun j => (if Nat.eqb i j then a else 0) * x j) = if Nat.ltb i n then a * x i else 0.
Proof.
  induction n; simpl; [reflexivity|]. rewrite IHn.
  destruct (Nat.eqb_spec i n), (Nat.ltb_spec i n), (Nat.ltb_spec i (S n));
    subst; try lia; ring.
Qed.

Lemma sumn_delta_r n j a x :
  sumn n (fun l => x l * (if Nat.eqb l j then a else 0)) = if Nat.ltb j n then x j * a else 0.
Proof.
  induction n; simpl; [reflexivity|]. rewrite IHn.
  destruct (Nat.eqb_spec n j), (Nat.ltb_spec j n), (Nat.ltb_spec j (S n));
    subst; try lia; ring.
Qed.

(* ------------------------------------------------------- matrices, vectors *)

Definition vec := nat -> Z.
Definition mat := nat -> nat -> Z.

Definition mv (n : nat) (M : mat) (x : vec) : vec := fun i => sumn n (fun j => M i j * x j).
Definition mm (k : nat) (A B : mat) : mat := fun i j => sumn k (fun l => A i l * B l j).
Definition idm : mat := fun i j => if Nat.eqb i j then 1 else 0.

Definition veq (m : nat) (x y : vec) := forall i, (i < m)%nat -> x i = y i.
Definition meq (m n : nat) (A B : mat) :=
  forall i j, (i < m)%nat -> (j < n)%nat -> A i j = B i j.

Lemma mv_mm k n A B x i : mv n (mm k A B) x i = mv k A (mv n B x) i.
Proof.
  unfold mv, mm.
  transitivity (sumn n (fun j => sumn k (fun l => A i l * B l j * x j))).
  - apply sumn_ext; intros j _. rewrite Z.mul_comm, <- sumn_scale.
    apply sumn_ext; intros; ring.
  - rewrite sumn_swap. apply sumn_ext; intros l _. rewrite <- sumn_scale.
    apply sumn_ext; intros; ring.
Qed.

Lemma mm_assoc k l A B C i j : mm k (mm l A B) C i j = mm l A (mm k B C) i j.
Proof.
  unfold mm.
  transitivity (sumn k (fun c => sumn l (fun b => A i b * B b c * C c j))).
  - apply sumn_ext; intros c _. rewrite Z.mul_comm, <- sumn_scale.
    apply sumn_ext; intros; ring.
  - rewrite sumn_swap. apply sumn_ext; intros b _. rewrite <- sumn_scale.
    apply sumn_ext; intros; ring.
Qed.

Lemma mv_id m x i : (i < m)%nat -> mv m idm x i = x i.
Proof.
  intros H. unfold mv, idm. rewrite sumn_delta.
  destruct (Nat.ltb_spec i m); [ring | lia].
Qed.

Lemma mm_id_l m A i j : (i < m)%nat -> mm m idm A i j = A i j.
Proof.
  intros H. unfold mm, idm. rewrite (sumn_delta m i 1 (fun l => A l j)).
  destruct (Nat.ltb_spec i m); [ring | lia].
Qed.

Lemma mm_id_r n A i j : (j < n)%nat -> mm n A idm i j = A i j.
Proof.
  intros H. unfold mm, idm. rewrite (sumn_delta_r n j 1 (fun l => A i l)).
  destruct (Nat.ltb_spec j n); [ring | lia].
Qed.

Lemma mv_veq n M x y : veq n x y -> forall i, mv n M x i = mv n M y i.
Proof. intros H i. unfold mv. apply sumn_ext. intros j Hj. rewrite H by exact Hj. reflexivity. Qed.

Lemma mv_meq m n M N x : meq m n M N -> forall i, (i < m)%nat -> mv n M x i = mv n N x i.
Proof. intros H i Hi. unfold mv. apply sumn_ext. intros j Hj. rewrite H by assumption. reflexivity. Qed.

Lemma mm_ext_l m k A A' B : meq m k A A' -> forall i j, (i < m)%nat -> mm k A B i j = mm k A' B i j.
Proof. intros H i j Hi. unfold mm. apply sumn_ext. intros l Hl. rewrite H by assumption. reflexivity. Qed.

Lemma mm_ext_r k n A B B' : meq k n B B' -> forall i j, (j < n)%nat -> mm k A B i j = mm k A B' i j.
Proof. intros H i j Hj. unfold mm. apply sumn_ext. intros l Hl. rewrite H by assumption. reflexivity. Qed.

Lemma mv_zero n M i : mv n M (fun _ => 0) i = 0.
Proof. unfold mv. rewrite (sumn_ext n _ (fun _ => 0)) by (intros; ring). apply sumn_zero. Qed.

(* ------------------------------------------------- Smith normal form data *)

(* U A V = D with U' U = I and V V' = I: U and V are invertible over Z *)
Record snf_cert (m n : nat) (A U Ui V Vi D : mat) : Prop := {
  c_UAV : meq m n (mm n (mm m U A) V) D;
  c_UiU : meq m m (mm m Ui U) idm;
  c_VVi : meq n n (mm n V Vi) idm
}.

Definition diagonal (m n : nat) (D : mat) :=
  forall i j, (i < m)%nat -> (j < n)%nat -> i <> j -> D i j = 0.

Section General.
  Variables (m n : nat) (A U Ui V Vi D : mat).
  Hypothesis C : snf_cert m n A U Ui V Vi D.

  Lemma VVi_vec w : veq n (mv n V (mv n Vi w)) w.
  Proof.
    intros j Hj. rewrite <- mv_mm. rewrite (mv_meq n n _ idm w (c_VVi _ _ _ _ _ _ _ _ C) j Hj).
    apply mv_id; exact Hj.
  Qed.

  Lemma UiU_vec z : veq m (mv m Ui (mv m U z)) z.
  Proof.
    intros i Hi. rewrite <- mv_mm. rewrite (mv_meq m m _ idm z (c_UiU _ _ _ _ _ _ _ _ C) i Hi).
    apply mv_id; exact Hi.
  Qed.

  Lemma snf_forward w i : (i < m)%nat -> mv n D (mv n Vi w) i = mv m U (mv n A w) i.
  Proof.
    intros Hi.
    rewrite <- (mv_meq m n _ _ _ (c_UAV _ _ _ _ _ _ _ _ C) i Hi).
    rewrite mv_mm. rewrite (mv_veq n _ _ _ (VVi_vec w)). apply mv_mm.
  Qed.

  Lemma snf_backward w : veq m (mv n A (mv n V w)) (mv m Ui (mv n D w)).
  Proof.
    intros i Hi.
    rewrite (mv_veq m Ui (mv n D w) (mv n (mm n (mm m U A) V) w)).
    2: { intros l Hl. symmetry. apply (mv_meq m n _ _ _ (c_UAV _ _ _ _ _ _ _ _ C) l Hl). }
    rewrite (mv_veq m Ui _ (mv m U (mv n A (mv n V w)))).
    2: { intros l Hl. rewrite mv_mm. apply mv_mm. }
    symmetry. apply UiU_vec. exact Hi.
  Qed.

  Theorem image_iff z :
    (exists w, veq m (mv n A w) z) <-> (exists w, veq m (mv n D w) (mv m U z)).
  Proof.
    split.
    - intros [w Hw]. exists (mv n Vi w). intros i Hi.
      rewrite snf_forward by exact Hi. apply mv_veq. exact Hw.
    - intros [w Hw]. exists (mv n V w). intros i Hi.
      rewrite snf_backward by exact Hi.
      rewrite (mv_veq m Ui _ _ Hw). apply UiU_vec. exact Hi.
  Qed.

  Theorem kernel_trivial :
    diagonal m n D ->
    (forall i, (i < n)%nat -> (i < m)%nat /\ D i i <> 0) ->
    forall w, veq m (mv n A w) (fun _ => 0) -> veq n w (fun _ => 0).
  Proof.
    intros Hd Hnz w Hw.
    assert (Hy : forall i, (i < n)%nat -> mv n Vi w i = 0).
    { intros i Hi. destruct (Hnz i Hi) as [Him Hne].
      assert (E : mv n D (mv n Vi w) i = D i i * mv n Vi w i).
      { unfold mv at 1.
        rewrite (sumn_ext n _ (fun j => (if Nat.eqb i j then D i i else 0) * mv n Vi w j)).
        - rewrite sumn_delta. destruct (Nat.ltb_spec i n); [reflexivity | lia].
        - intros j Hj. destruct (Nat.eqb_spec i j); [subst; reflexivity|].
          rewrite Hd by assumption. ring. }
      rewrite snf_forward in E by exact Him.
      rewrite (mv_veq m U _ _ Hw), mv_zero in E.
      symmetry in E. apply Z.mul_eq_0 in E. destruct E; [contradiction | assumption]. }
    intros j Hj. rewrite <- (VVi_vec w j Hj). unfold mv at 1.
    rewrite (sumn_ext n _ (fun _ => 0)). apply sumn_zero.
    intros l Hl. rewrite Hy by exact Hl. ring.
  Qed.

  Lemma recover : meq m n A (mm n (mm m Ui D) Vi).
  Proof.
    intros i j Hi Hj. symmetry.
    rewrite (mm_ext_l m n _ (mm m Ui (mm n (mm m U A) V))).
    2: { intros a b Ha Hb. apply (mm_ext_r m n); [|exact Hb].
         intros c e Hc He. symmetry. apply (c_UAV _ _ _ _ _ _ _ _ C); assumption. }
    2: exact Hi.
    rewrite (mm_ext_l m n _ (mm n (mm m Ui (mm m U A)) V)).
    2: { intros a b Ha Hb. symmetry. apply mm_assoc. }
    2: exact Hi.
    rewrite mm_assoc.
    rewrite (mm_ext_r n n _ _ idm (c_VVi _ _ _ _ _ _ _ _ C) i j Hj).
    rewrite mm_id_r by exact Hj.
    rewrite <- mm_assoc.
    rewrite (mm_ext_l m m _ idm A (c_UiU _ _ _ _ _ _ _ _ C) i j Hi).
    apply mm_id_l. exact Hi.
  Qed.

  Definition divides_all (d : Z) (M : mat) :=
    forall i j, (i < m)%nat -> (j < n)%nat -> (d | M i j).

  Lemma divides_sumn d k f : (forall l, (l < k)%nat -> (d | f l)) -> (d | sumn k f).
  Proof.
    induction k; simpl; intros H; [apply Z.divide_0_r|].
    apply Z.divide_add_r; [apply IHk; intros; apply H; lia | apply H; lia].
  Qed.

  Theorem divisors_invariant d : divides_all d A <-> divides_all d D.
  Proof.
    split; intros H i j Hi Hj.
    - rewrite <- (c_UAV _ _ _ _ _ _ _ _ C) by assumption.
      apply divides_sumn. intros l Hl. apply Z.divide_mul_l.
      apply divides_sumn. intros r Hr. apply Z.divide_mul_r. apply H; assumption.
    - rewrite recover by assumption.
      apply divides_sumn. intros l Hl. apply Z.divide_mul_l.
      apply divides_sumn. intros r Hr. apply Z.divide_mul_r. apply H; assumption.
  Qed.

  Theorem first_invariant_factor :
    diagonal m n D -> (0 < m)%nat -> (0 < n)%nat ->
    (forall i, (i < m)%nat -> (i < n)%nat -> (D 0%nat 0%nat | D i i)) ->
    forall d, divides_all d A <-> (d | D 0%nat 0%nat).
  Proof.
    intros Hd Hm Hn Hdiv d. rewrite divisors_invariant. split.
    - intros H. apply H; assumption.
    - intros H i j Hi Hj. destruct (Nat.eq_dec i j).
      + subst. eapply Z.divide_trans; [exact H | apply Hdiv; assumption].
      + rewrite Hd by assumption. apply Z.divide_0_r.
  Qed.
End General.

Theorem diag_image m n D :
  diagonal m n D ->
  forall v, (exists w, veq m (mv n D w) v) <->
            (forall i, (i < m)%nat -> if Nat.ltb i n then (D i i | v i) else v i = 0).
Proof.
  intros Hd v.
  assert (Hmv : forall w i, (i < m)%nat -> mv n D w i = if Nat.ltb i n then D i i * w i else 0).
  { intros w i Hi. unfold mv.
    rewrite (sumn_ext n _ (fun j => (if Nat.eqb i j then D i i else 0) * w j)).
    - apply sumn_delta.
    - intros j Hj. destruct (Nat.eqb_spec i j); [subst; reflexivity|].
      rewrite Hd by assumption. ring. }
  split.
  - intros [w Hw] i Hi. rewrite <- (Hw i Hi), (Hmv w i Hi).
    destruct (Nat.ltb i n); [exists (w i); ring | reflexivity].
  - intros H.
    exists (fun j => if Z.eqb (D j j) 0 then 0 else v j / D j j).
    intros i Hi. rewrite (Hmv _ i Hi). specialize (H i Hi).
    destruct (Nat.ltb i n); [|symmetry; exact H].
    destruct (Z.eqb_spec (D i i) 0) as [E|E].
    + rewrite E in H. apply Z.divide_0_l in H. rewrite H. ring.
    + destruct H as [c Hc]. rewrite Hc, Z.div_mul by exact E. ring.
Qed.

(* ------------------------------------------------------ computed checks *)

Definition ofL (L : list (list Z)) : mat := fun i j => nth j (nth i L []) 0.
Definition ofV (L : list Z) : vec := fun i => nth i L 0.

Definition check_meq (m n : nat) (A B : mat) : bool :=
  forallb (fun i => forallb (fun j => Z.eqb (A i j) (B i j)) (seq 0 n)) (seq 0 m).

Lemma in_seq0 i k : (i < k)%nat -> In i (seq 0 k).
Proof. intros H. apply in_seq. lia. Qed.

Lemma check_meq_sound m n A B : check_meq m n A B = true -> meq m n A B.
Proof.
  unfold check_meq. intros H i j Hi Hj.
  rewrite forallb_forall in H. specialize (H i (in_seq0 _ _ Hi)).
  rewrite forallb_forall in H. specialize (H j (in_seq0 _ _ Hj)).
  apply Z.eqb_eq. exact H.
Qed.

Definition check_diag (m n : nat) (D : mat) : bool :=
  forallb (fun i => forallb (fun j => orb (Nat.eqb i j) (Z.eqb (D i j) 0)) (seq 0 n)) (seq 0 m).

Lemma check_diag_sound m n D : check_diag m n D = true -> diagonal m n D.
Proof.
  unfold check_diag. intros H i j Hi Hj Hne.
  rewrite forallb_forall in H. specialize (H i (in_seq0 _ _ Hi)).
  rewrite forallb_forall in H. specialize (H j (in_seq0 _ _ Hj)).
  apply orb_true_iff in H. destruct H as [H|H].
  - apply Nat.eqb_eq in H. contradiction.
  - apply Z.eqb_eq. exact H.
Qed.

Definition check_cert (m n : nat) (A U Ui V Vi D : mat) : bool :=
  check_meq m n (mm n (mm m U A) V) D && check_meq m m (mm m Ui U) idm
  && check_meq n n (mm n V Vi) idm && check_diag m n D.

Lemma check_cert_sound m n A U Ui V Vi D :
  check_cert m n A U Ui V Vi D = true -> snf_cert m n A U Ui V Vi D /\ diagonal m n D.
Proof.
  unfold check_cert. intros H. repeat (apply andb_true_iff in H; destruct H as [H ?]).
  split; [split|]; apply check_meq_sound || apply check_diag_sound; assumption.
Qed.

(* (a | b) checked by exact division *)
Definition divb (a b : Z) : bool := Z.eqb b ((b / a) * a).

Lemma divb_sound a b : divb a b = true -> (a | b).
Proof. unfold divb. intros H. apply Z.eqb_eq in H. exists (b / a). exact H. Qed.

(* the diagonal, and the chain d_i | d_{i+1} *)
Definition diag_list (m n : nat) (D : mat) : list Z := map (fun i => D i i) (seq 0 (Nat.min m n)).

Fixpoint chainb (l : list Z) : bool :=
  match l with
  | a :: ((b :: _) as t) => divb a b && chainb t
  | _ => true
  end.

Definition first_divides (m n : nat) (D : mat) : bool :=
  forallb (fun i => divb (D 0%nat 0%nat) (D i i)) (seq 0 (Nat.min m n)).

Lemma first_divides_sound m n D :
  first_divides m n D = true ->
  forall i, (i < m)%nat -> (i < n)%nat -> (D 0%nat 0%nat | D i i).
Proof.
  unfold first_divides. intros H i Hi Hn. rewrite forallb_forall in H.
  apply divb_sound, H, in_seq0. lia.
Qed.

(* BEGIN GENERATED CERTIFICATES *)
(* case 1: diagonal [2, 6, 12] *)
Definition snf1_A : list (list Z) :=
  [[2; 4; 4];
    [-6; 6; 12];
    [10; -4; -16]].
Definition snf1_U : list (list Z) :=
  [[1; 0; 0];
    [3; 1; 0];
    [1; 2; 1]].
Definition snf1_Ui : list (list Z) :=
  [[1; 0; 0];
    [-3; 1; 0];
    [5; -2; 1]].
Definition snf1_V : list (list Z) :=
  [[1; 0; -2];
    [0; -1; 4];
    [0; 1; -3]].
Definition snf1_Vi : list (list Z) :=
  [[1; 2; 2];
    [0; 3; 4];
    [0; 1; 1]].
Definition snf1_D : list (list Z) :=
  [[2; 0; 0];
    [0; 6; 0];
    [0; 0; 12]].
Lemma snf1_check : check_cert 3 3 (ofL snf1_A) (ofL snf1_U) (ofL snf1_Ui) (ofL snf1_V) (ofL snf1_Vi) (ofL snf1_D) = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf1_cert : snf_cert 3 3 (ofL snf1_A) (ofL snf1_U) (ofL snf1_Ui) (ofL snf1_V) (ofL snf1_Vi) (ofL snf1_D) /\ diagonal 3 3 (ofL snf1_D).
Proof. apply check_cert_sound, snf1_check. Qed.
Theorem snf1_diagonal : diag_list 3 3 (ofL snf1_D) = [2; 6; 12].
Proof. vm_compute. reflexivity. Qed.
Theorem snf1_chain : chainb [2; 6; 12] = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf1_first_invariant_factor :
  forall d, divides_all 3 3 d (ofL snf1_A) <-> (d | ofL snf1_D 0%nat 0%nat).
Proof.
  destruct snf1_cert as [C Hd].
  exact (first_invariant_factor _ _ _ _ _ _ _ _ C Hd ltac:(lia) ltac:(lia)
           (first_divides_sound 3 3 (ofL snf1_D) ltac:(vm_compute; reflexivity))).
Qed.
(* case 2: diagonal [2, 4] *)
Definition snf2_A : list (list Z) :=
  [[2; 4];
    [6; 8]].
Definition snf2_U : list (list Z) :=
  [[1; 0];
    [3; -1]].
Definition snf2_Ui : list (list Z) :=
  [[1; 0];
    [3; -1]].
Definition snf2_V : list (list Z) :=
  [[1; -2];
    [0; 1]].
Definition snf2_Vi : list (list Z) :=
  [[1; 2];
    [0; 1]].
Definition snf2_D : list (list Z) :=
  [[2; 0];
    [0; 4]].
Lemma snf2_check : check_cert 2 2 (ofL snf2_A) (ofL snf2_U) (ofL snf2_Ui) (ofL snf2_V) (ofL snf2_Vi) (ofL snf2_D) = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf2_cert : snf_cert 2 2 (ofL snf2_A) (ofL snf2_U) (ofL snf2_Ui) (ofL snf2_V) (ofL snf2_Vi) (ofL snf2_D) /\ diagonal 2 2 (ofL snf2_D).
Proof. apply check_cert_sound, snf2_check. Qed.
Theorem snf2_diagonal : diag_list 2 2 (ofL snf2_D) = [2; 4].
Proof. vm_compute. reflexivity. Qed.
Theorem snf2_chain : chainb [2; 4] = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf2_first_invariant_factor :
  forall d, divides_all 2 2 d (ofL snf2_A) <-> (d | ofL snf2_D 0%nat 0%nat).
Proof.
  destruct snf2_cert as [C Hd].
  exact (first_invariant_factor _ _ _ _ _ _ _ _ C Hd ltac:(lia) ltac:(lia)
           (first_divides_sound 2 2 (ofL snf2_D) ltac:(vm_compute; reflexivity))).
Qed.
(* case 3: diagonal [1, 3, 0] *)
Definition snf3_A : list (list Z) :=
  [[1; 2; 3];
    [4; 5; 6];
    [7; 8; 9]].
Definition snf3_U : list (list Z) :=
  [[1; 0; 0];
    [4; -1; 0];
    [1; -2; 1]].
Definition snf3_Ui : list (list Z) :=
  [[1; 0; 0];
    [4; -1; 0];
    [7; -2; 1]].
Definition snf3_V : list (list Z) :=
  [[1; -2; 1];
    [0; 1; -2];
    [0; 0; 1]].
Definition snf3_Vi : list (list Z) :=
  [[1; 2; 3];
    [0; 1; 2];
    [0; 0; 1]].
Definition snf3_D : list (list Z) :=
  [[1; 0; 0];
    [0; 3; 0];
    [0; 0; 0]].
Lemma snf3_check : check_cert 3 3 (ofL snf3_A) (ofL snf3_U) (ofL snf3_Ui) (ofL snf3_V) (ofL snf3_Vi) (ofL snf3_D) = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf3_cert : snf_cert 3 3 (ofL snf3_A) (ofL snf3_U) (ofL snf3_Ui) (ofL snf3_V) (ofL snf3_Vi) (ofL snf3_D) /\ diagonal 3 3 (ofL snf3_D).
Proof. apply check_cert_sound, snf3_check. Qed.
Theorem snf3_diagonal : diag_list 3 3 (ofL snf3_D) = [1; 3; 0].
Proof. vm_compute. reflexivity. Qed.
Theorem snf3_chain : chainb [1; 3] = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf3_first_invariant_factor :
  forall d, divides_all 3 3 d (ofL snf3_A) <-> (d | ofL snf3_D 0%nat 0%nat).
Proof.
  destruct snf3_cert as [C Hd].
  exact (first_invariant_factor _ _ _ _ _ _ _ _ C Hd ltac:(lia) ltac:(lia)
           (first_divides_sound 3 3 (ofL snf3_D) ltac:(vm_compute; reflexivity))).
Qed.
(* case 4: diagonal [1, 30, 30] *)
Definition snf4_A : list (list Z) :=
  [[6; 0; 0];
    [0; 10; 0];
    [0; 0; 15]].
Definition snf4_U : list (list Z) :=
  [[1; 1; 1];
    [5; 6; 0];
    [15; 15; 14]].
Definition snf4_Ui : list (list Z) :=
  [[-84; -1; 6];
    [70; 1; -5];
    [15; 0; -1]].
Definition snf4_V : list (list Z) :=
  [[-14; -5; 30];
    [7; 3; -15];
    [1; 0; -2]].
Definition snf4_Vi : list (list Z) :=
  [[6; 10; 15];
    [1; 2; 0];
    [3; 5; 7]].
Definition snf4_D : list (list Z) :=
  [[1; 0; 0];
    [0; 30; 0];
    [0; 0; 30]].
Lemma snf4_check : check_cert 3 3 (ofL snf4_A) (ofL snf4_U) (ofL snf4_Ui) (ofL snf4_V) (ofL snf4_Vi) (ofL snf4_D) = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf4_cert : snf_cert 3 3 (ofL snf4_A) (ofL snf4_U) (ofL snf4_Ui) (ofL snf4_V) (ofL snf4_Vi) (ofL snf4_D) /\ diagonal 3 3 (ofL snf4_D).
Proof. apply check_cert_sound, snf4_check. Qed.
Theorem snf4_diagonal : diag_list 3 3 (ofL snf4_D) = [1; 30; 30].
Proof. vm_compute. reflexivity. Qed.
Theorem snf4_chain : chainb [1; 30; 30] = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf4_first_invariant_factor :
  forall d, divides_all 3 3 d (ofL snf4_A) <-> (d | ofL snf4_D 0%nat 0%nat).
Proof.
  destruct snf4_cert as [C Hd].
  exact (first_invariant_factor _ _ _ _ _ _ _ _ C Hd ltac:(lia) ltac:(lia)
           (first_divides_sound 3 3 (ofL snf4_D) ltac:(vm_compute; reflexivity))).
Qed.
(* case 5: diagonal [3, 3, 0] *)
Definition snf5_A : list (list Z) :=
  [[0; 3; 0; 6];
    [9; 0; 12; 0];
    [0; 0; 0; 0]].
Definition snf5_U : list (list Z) :=
  [[1; 0; 0];
    [0; 1; 0];
    [0; 0; 1]].
Definition snf5_Ui : list (list Z) :=
  [[1; 0; 0];
    [0; 1; 0];
    [0; 0; 1]].
Definition snf5_V : list (list Z) :=
  [[0; -1; 4; 0];
    [1; 0; 0; -2];
    [0; 1; -3; 0];
    [0; 0; 0; 1]].
Definition snf5_Vi : list (list Z) :=
  [[0; 1; 0; 2];
    [3; 0; 4; 0];
    [1; 0; 1; 0];
    [0; 0; 0; 1]].
Definition snf5_D : list (list Z) :=
  [[3; 0; 0; 0];
    [0; 3; 0; 0];
    [0; 0; 0; 0]].
Lemma snf5_check : check_cert 3 4 (ofL snf5_A) (ofL snf5_U) (ofL snf5_Ui) (ofL snf5_V) (ofL snf5_Vi) (ofL snf5_D) = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf5_cert : snf_cert 3 4 (ofL snf5_A) (ofL snf5_U) (ofL snf5_Ui) (ofL snf5_V) (ofL snf5_Vi) (ofL snf5_D) /\ diagonal 3 4 (ofL snf5_D).
Proof. apply check_cert_sound, snf5_check. Qed.
Theorem snf5_diagonal : diag_list 3 4 (ofL snf5_D) = [3; 3; 0].
Proof. vm_compute. reflexivity. Qed.
Theorem snf5_chain : chainb [3; 3] = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf5_first_invariant_factor :
  forall d, divides_all 3 4 d (ofL snf5_A) <-> (d | ofL snf5_D 0%nat 0%nat).
Proof.
  destruct snf5_cert as [C Hd].
  exact (first_invariant_factor _ _ _ _ _ _ _ _ C Hd ltac:(lia) ltac:(lia)
           (first_divides_sound 3 4 (ofL snf5_D) ltac:(vm_compute; reflexivity))).
Qed.
(* case 6: diagonal [1, 49382715598] *)
Definition snf6_A : list (list Z) :=
  [[12345678901; 2];
    [3; 4]].
Definition snf6_U : list (list Z) :=
  [[1; 0];
    [24691357797; 1]].
Definition snf6_Ui : list (list Z) :=
  [[1; 0];
    [-24691357797; 1]].
Definition snf6_V : list (list Z) :=
  [[1; -2];
    [-6172839450; 12345678901]].
Definition snf6_Vi : list (list Z) :=
  [[12345678901; 2];
    [6172839450; 1]].
Definition snf6_D : list (list Z) :=
  [[1; 0];
    [0; 49382715598]].
Lemma snf6_check : check_cert 2 2 (ofL snf6_A) (ofL snf6_U) (ofL snf6_Ui) (ofL snf6_V) (ofL snf6_Vi) (ofL snf6_D) = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf6_cert : snf_cert 2 2 (ofL snf6_A) (ofL snf6_U) (ofL snf6_Ui) (ofL snf6_V) (ofL snf6_Vi) (ofL snf6_D) /\ diagonal 2 2 (ofL snf6_D).
Proof. apply check_cert_sound, snf6_check. Qed.
Theorem snf6_diagonal : diag_list 2 2 (ofL snf6_D) = [1; 49382715598].
Proof. vm_compute. reflexivity. Qed.
Theorem snf6_chain : chainb [1; 49382715598] = true.
Proof. vm_compute. reflexivity. Qed.
Theorem snf6_first_invariant_factor :
  forall d, divides_all 2 2 d (ofL snf6_A) <-> (d | ofL snf6_D 0%nat 0%nat).
Proof.
  destruct snf6_cert as [C Hd].
  exact (first_invariant_factor _ _ _ _ _ _ _ _ C Hd ltac:(lia) ltac:(lia)
           (first_divides_sound 2 2 (ofL snf6_D) ltac:(vm_compute; reflexivity))).
Qed.
(* RP^2: 6 vertices, 15 edges, 10 triangles *)
(* d1: invariant factors [1, 1, 1, 1, 1] *)
Definition rp2_d1 : list (list Z) :=
  [[-1; -1; -1; -1; -1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [1; 0; 0; 0; 0; -1; -1; -1; -1; 0; 0; 0; 0; 0; 0];
    [0; 1; 0; 0; 0; 1; 0; 0; 0; -1; -1; -1; 0; 0; 0];
    [0; 0; 1; 0; 0; 0; 1; 0; 0; 1; 0; 0; -1; -1; 0];
    [0; 0; 0; 1; 0; 0; 0; 1; 0; 0; 1; 0; 1; 0; -1];
    [0; 0; 0; 0; 1; 0; 0; 0; 1; 0; 0; 1; 0; 1; 1]].
Definition rp2_d1U : list (list Z) :=
  [[-1; 0; 0; 0; 0; 0];
    [-1; -1; 0; 0; 0; 0];
    [-1; -1; -1; 0; 0; 0];
    [-1; -1; -1; -1; 0; 0];
    [-1; -1; -1; -1; -1; 0];
    [1; 1; 1; 1; 1; 1]].
Definition rp2_d1Ui : list (list Z) :=
  [[-1; 0; 0; 0; 0; 0];
    [1; -1; 0; 0; 0; 0];
    [0; 1; -1; 0; 0; 0];
    [0; 0; 1; -1; 0; 0];
    [0; 0; 0; 1; -1; 0];
    [0; 0; 0; 0; 1; 1]].
Definition rp2_d1V : list (list Z) :=
  [[1; -1; 0; 0; 0; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0];
    [0; 1; -1; 0; 0; -1; 0; 0; 0; 1; 1; 1; 0; 0; 0];
    [0; 0; 1; -1; 0; 0; -1; 0; 0; -1; 0; 0; 1; 1; 0];
    [0; 0; 0; 1; -1; 0; 0; -1; 0; 0; -1; 0; -1; 0; 1];
    [0; 0; 0; 0; 1; 0; 0; 0; -1; 0; 0; -1; 0; -1; -1];
    [0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1]].
Definition rp2_d1Vi : list (list Z) :=
  [[1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 1; 1; 1; 1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0];
    [0; 0; 1; 1; 1; 0; 1; 1; 1; 1; 1; 1; 0; 0; 0];
    [0; 0; 0; 1; 1; 0; 0; 1; 1; 0; 1; 1; 1; 1; 0];
    [0; 0; 0; 0; 1; 0; 0; 0; 1; 0; 0; 1; 0; 1; 1];
    [0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 1]].
Definition rp2_d1D : list (list Z) :=
  [[1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0]].
(* d2: invariant factors [1, 1, 1, 1, 1, 1, 1, 1, 1, 2] *)
Definition rp2_d2 : list (list Z) :=
  [[1; 1; 0; 0; 0; 0; 0; 0; 0; 0];
    [-1; 0; 1; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; -1; 1; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; -1; 1; 0; 0; 0; 0; 0];
    [0; -1; 0; 0; -1; 0; 0; 0; 0; 0];
    [1; 0; 0; 0; 0; 1; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 1; 1; 0; 0];
    [0; 0; 0; 0; 0; -1; -1; 0; 0; 0];
    [0; 1; 0; 0; 0; 0; 0; -1; 0; 0];
    [0; 0; 1; 0; 0; 0; 0; 0; 1; 0];
    [0; 0; 0; 0; 0; 1; 0; 0; 0; 1];
    [0; 0; 0; 0; 0; 0; 0; 0; -1; -1];
    [0; 0; 0; 1; 0; 0; 1; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 1; 1; 0];
    [0; 0; 0; 0; 1; 0; 0; 0; 0; 1]].
Definition rp2_d2U : list (list Z) :=
  [[1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; -1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; -1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 1; 1; 1; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; -1; -1; 0; 0; 0; 0; 0; 0; 0];
    [0; 1; 0; 0; 0; 1; 1; 1; 0; -1; 0; 0; 0; 0; 0];
    [0; -1; 0; 0; 0; -1; 0; 0; 0; 1; 1; 0; 0; 0; 0];
    [0; -1; -1; 0; 0; -1; 0; 1; 0; 0; 2; 0; 1; 0; 0];
    [-1; 0; 0; 0; 0; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0];
    [0; -1; 0; 0; 0; -1; 0; 0; 0; 1; 1; 1; 0; 0; 0];
    [1; 1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; -1; 0; 0; 0; -1; 0; 0; -1; 0; 0; 1; 1; 0];
    [0; 0; 0; -1; 0; 0; 0; -1; 0; 0; -1; 0; -1; 0; 1]].
Definition rp2_d2Ui : list (list Z) :=
  [[1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [-1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; -1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; -1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; -1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 1; 0; 0];
    [1; -1; 1; 1; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; -1; -1; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 1; -1; -1; -1; 0; 1; 0; 0; 0; 1; 0; 0; 0; 0];
    [0; 0; 1; 1; 1; 0; -1; -1; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 1; 1; 1; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; -1; 0; 0; 1; 0; 0; 0];
    [0; 0; 0; 1; 1; 1; -1; -2; -2; 1; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 1; 2; -1; 0; 0; 0; 1; 0];
    [0; 0; 0; 0; 1; 0; -1; -1; -1; 1; 0; 0; 0; 0; 1]].
Definition rp2_d2V : list (list Z) :=
  [[1; -1; 1; 1; 1; 0; -1; -1; -1; 1];
    [0; 1; -1; -1; -1; 0; 1; 1; 1; -1];
    [0; 0; 1; 1; 1; 0; -1; -1; -1; 1];
    [0; 0; 0; 1; 1; 0; -1; -1; -1; 1];
    [0; 0; 0; 0; 1; 0; -1; -1; -1; 1];
    [0; 0; 0; 0; 0; 0; 1; 1; 1; -1];
    [0; 0; 0; 0; 0; 1; 0; -1; -1; 1];
    [0; 0; 0; 0; 0; 0; 0; 1; 1; -1];
    [0; 0; 0; 0; 0; 0; 0; 0; 1; -1];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 1]].
Definition rp2_d2Vi : list (list Z) :=
  [[1; 1; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 1; 1; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 1; -1; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 1; -1; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 1; 1; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 1; 1; 0; 0];
    [0; 0; 0; 0; 0; 1; 0; -1; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 1; -1; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 1; 1];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 1]].
Definition rp2_d2D : list (list Z) :=
  [[1; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 1; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 1; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 1; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 1; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 1; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 1; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 1; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 1; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 2];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0];
    [0; 0; 0; 0; 0; 0; 0; 0; 0; 0]].
(* columns 10.. of d1 . d2U' are independent: L . d1 . d2U' = -1 on them *)
Definition rp2_L : list (list Z) :=
  [[0; 1; 0; 0; 0; 0];
    [0; 0; 1; 0; 0; 0];
    [1; 0; 0; 0; 0; 0];
    [0; 0; 0; 1; 0; 0];
    [0; 0; 0; 0; 1; 0]].
Definition rp2_c0 : Z := -1.
(* END GENERATED CERTIFICATES *)

(* ---------------------------------------------- RP^2: H_0 = Z, H_1 = Z/2, H_2 = 0 *)

Definition d1 := ofL rp2_d1.
Definition d2 := ofL rp2_d2.
Definition U1 := ofL rp2_d1U.
Definition U1i := ofL rp2_d1Ui.
Definition D1 := ofL rp2_d1D.
Definition U2 := ofL rp2_d2U.
Definition U2i := ofL rp2_d2Ui.
Definition D2 := ofL rp2_d2D.

Definition check_veq (m : nat) (x y : vec) : bool :=
  forallb (fun i => Z.eqb (x i) (y i)) (seq 0 m).

Lemma check_veq_sound m x y : check_veq m x y = true -> veq m x y.
Proof.
  unfold check_veq. intros H i Hi. rewrite forallb_forall in H.
  apply Z.eqb_eq, H, in_seq0, Hi.
Qed.

Definition unit_vec (k : nat) : vec := fun j => if Nat.eqb k j then 1 else 0.

(* the chain complex: d1 . d2 = 0 *)
Lemma rp2_is_complex : meq 6 10 (mm 15 d1 d2) (fun _ _ => 0).
Proof. apply check_meq_sound. vm_compute. reflexivity. Qed.

Lemma rp2_d1_check : check_cert 6 15 d1 U1 U1i (ofL rp2_d1V) (ofL rp2_d1Vi) D1 = true.
Proof. vm_compute. reflexivity. Qed.

Lemma rp2_d2_check : check_cert 15 10 d2 U2 U2i (ofL rp2_d2V) (ofL rp2_d2Vi) D2 = true.
Proof. vm_compute. reflexivity. Qed.

Lemma rp2_tail_check :
  check_meq 5 15 (mm 6 (ofL rp2_L) (mm 15 d1 U2i))
            (fun r c => if Nat.eqb (10 + r) c then rp2_c0 else 0) = true.
Proof. vm_compute. reflexivity. Qed.

Lemma rp2_c0_nonzero : rp2_c0 <> 0.
Proof. unfold rp2_c0. discriminate. Qed.

Lemma D2_ones : forall i, (i < 9)%nat -> D2 i i = 1.
Proof.
  assert (H : forallb (fun i => Z.eqb (D2 i i) 1) (seq 0 9) = true) by (vm_compute; reflexivity).
  intros i Hi. rewrite forallb_forall in H. apply Z.eqb_eq, H, in_seq0, Hi.
Qed.

Lemma D2_nine : D2 9%nat 9%nat = 2.
Proof. vm_compute. reflexivity. Qed.

Lemma D1_ones : forall i, (i < 5)%nat -> D1 i i = 1.
Proof.
  assert (H : forallb (fun i => Z.eqb (D1 i i) 1) (seq 0 5) = true) by (vm_compute; reflexivity).
  intros i Hi. rewrite forallb_forall in H. apply Z.eqb_eq, H, in_seq0, Hi.
Qed.

Lemma D1_five : D1 5%nat 5%nat = 0.
Proof. vm_compute. reflexivity. Qed.

(* the coordinates of U2 z beyond rank d2 = 10 vanish on every 1-cycle z *)
Lemma rp2_cycle_tail :
  forall z, veq 6 (mv 15 d1 z) (fun _ => 0) ->
  forall r, (r < 5)%nat -> mv 15 U2 z (10 + r)%nat = 0.
Proof.
  intros z Hz r Hr.
  destruct (check_cert_sound _ _ _ _ _ _ _ _ rp2_d2_check) as [C _].
  set (y := mv 15 U2 z).
  assert (HM : forall k, (k < 6)%nat -> mv 15 (mm 15 d1 U2i) y k = 0).
  { intros k Hk. rewrite mv_mm.
    rewrite (mv_veq 15 d1 _ z (UiU_vec _ _ _ _ _ _ _ _ C z)). apply Hz; exact Hk. }
  assert (HE : mv 15 (fun r c => if Nat.eqb (10 + r) c then rp2_c0 else 0) y r = 0).
  { rewrite <- (mv_meq 5 15 _ _ y (check_meq_sound _ _ _ _ rp2_tail_check) r Hr).
    rewrite mv_mm. rewrite (mv_veq 6 _ _ (fun _ => 0) HM). apply mv_zero. }
  unfold mv in HE. cbv beta in HE. rewrite sumn_delta in HE.
  destruct (Nat.ltb_spec (10 + r) 15); [|lia].
  apply Z.mul_eq_0 in HE. destruct HE as [HE|HE]; [exfalso; exact (rp2_c0_nonzero HE) | exact HE].
Qed.

(* H_1(RP^2; Z) = Z/2: on 1-cycles, z |-> (U2 z)_9 mod 2 is onto Z/2 with
   kernel exactly the boundaries *)
Theorem rp2_H1 :
  (forall z, veq 6 (mv 15 d1 z) (fun _ => 0) ->
     ((exists w, veq 15 (mv 10 d2 w) z) <-> Z.Even (mv 15 U2 z 9%nat)))
  /\ (exists z, veq 6 (mv 15 d1 z) (fun _ => 0) /\ Z.Odd (mv 15 U2 z 9%nat)).
Proof.
  destruct (check_cert_sound _ _ _ _ _ _ _ _ rp2_d2_check) as [C Hd].
  split.
  - intros z Hz. rewrite (image_iff _ _ _ _ _ _ _ _ C z), (diag_image _ _ _ Hd).
    split.
    + intros H. specialize (H 9%nat ltac:(lia)). cbv [Nat.ltb Nat.leb] in H.
      rewrite D2_nine in H. destruct H as [c Hc]. exists c. lia.
    + intros [c Hc] i Hi. destruct (Nat.ltb_spec i 10).
      * destruct (Nat.eq_dec i 9) as [->|Hne].
        -- rewrite D2_nine. exists c. lia.
        -- rewrite D2_ones by lia. apply Z.divide_1_l.
      * replace i with (10 + (i - 10))%nat by lia. apply rp2_cycle_tail; [exact Hz | lia].
  - exists (mv 15 U2i (unit_vec 9)). split.
    + apply check_veq_sound. vm_compute. reflexivity.
    + assert (E : mv 15 U2 (mv 15 U2i (unit_vec 9)) 9%nat = 1) by (vm_compute; reflexivity).
      rewrite E. exists 0. reflexivity.
Qed.

(* H_2(RP^2; Z) = 0: d2 is injective *)
Theorem rp2_H2 : forall w, veq 15 (mv 10 d2 w) (fun _ => 0) -> veq 10 w (fun _ => 0).
Proof.
  destruct (check_cert_sound _ _ _ _ _ _ _ _ rp2_d2_check) as [C Hd].
  apply (kernel_trivial _ _ _ _ _ _ _ _ C Hd).
  assert (H : forallb (fun i => negb (Z.eqb (D2 i i) 0)) (seq 0 10) = true) by (vm_compute; reflexivity).
  intros i Hi. split; [lia|]. rewrite forallb_forall in H.
  specialize (H i (in_seq0 _ _ Hi)). apply negb_true_iff, Z.eqb_neq in H. exact H.
Qed.

(* H_0(RP^2; Z) = Z: a 0-chain bounds iff (U1 z)_5 = 0, and that map is onto Z *)
Theorem rp2_H0 :
  (forall z, (exists w, veq 6 (mv 15 d1 w) z) <-> mv 6 U1 z 5%nat = 0)
  /\ (exists z, mv 6 U1 z 5%nat = 1).
Proof.
  destruct (check_cert_sound _ _ _ _ _ _ _ _ rp2_d1_check) as [C Hd].
  split.
  - intros z. rewrite (image_iff _ _ _ _ _ _ _ _ C z), (diag_image _ _ _ Hd).
    split.
    + intros H. specialize (H 5%nat ltac:(lia)). cbv [Nat.ltb Nat.leb] in H.
      rewrite D1_five in H. apply Z.divide_0_l in H. exact H.
    + intros H i Hi. destruct (Nat.ltb_spec i 15); [|lia].
      destruct (Nat.eq_dec i 5) as [->|Hne].
      * rewrite D1_five, H. apply Z.divide_refl.
      * rewrite D1_ones by lia. apply Z.divide_1_l.
  - exists (mv 6 U1i (unit_vec 5)). vm_compute. reflexivity.
Qed.

(* every main theorem is closed: no axioms *)
Print Assumptions image_iff.
Print Assumptions first_invariant_factor.
Print Assumptions snf6_first_invariant_factor.
Print Assumptions rp2_H0.
Print Assumptions rp2_H1.
Print Assumptions rp2_H2.
