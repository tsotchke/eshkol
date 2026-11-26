/- check_axioms.lean: report the axioms of the composed model's headline theorems.
   Expected: propext and Quot.sound at most; never sorryAx.
   Interface: lean check_axioms.lean, with EshkolKernelFrameMappings.olean on LEAN_PATH
   (run_frame_mapping_correspondence.sh does this). -/
import EshkolKernelFrameMappings
#print axioms EshkolKernelFrameMappings.reachable_coherent
#print axioms EshkolKernelFrameMappings.reachable_mutable_isolation
#print axioms EshkolKernelFrameMappings.reachable_space_isolation
#print axioms EshkolKernelFrameMappings.init_refuses_kernel_space_with_user_asid
#print axioms EshkolKernelFrameMappings.owner_check_asid_redundant
