/- check_axioms.lean: report the axioms of the composed model's headline theorems.
   Expected: propext and Quot.sound only; never sorryAx.
   Interface: lean check_axioms.lean, with EshkolKernelFrameMappings.olean on LEAN_PATH
   (run_frame_mapping_correspondence.sh does this). -/
import EshkolKernelFrameMappings
#print axioms EshkolKernelFrameMappings.reachable_coherent
#print axioms EshkolKernelFrameMappings.reachable_mutable_isolation
#print axioms EshkolKernelFrameMappings.asid_collision_breaks_isolation
#print axioms EshkolKernelFrameMappings.kernel_last_asid_refuses_read_only_user
