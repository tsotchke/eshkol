/-
  Eshkol CE-9 Chained eshkol.certificate.v2 Derivation Proof
  Verified by Lean 4.12.0 kernel via decidable reflection (self-contained).
  Scaled-integer intervals use conservative outward rounding (10^9 scale).
-/

namespace EshkolCE9CertChain

structure IntInterval where
  lo : Int
  hi : Int
  deriving DecidableEq, Repr

structure ProgramLink where
  programDigest : String
  linkHash      : String
  deriving DecidableEq, Repr

structure TaylorEnclosureLink where
  prevHash        : String
  certifiedBounds : List IntInterval
  linkHash        : String
  deriving DecidableEq, Repr

structure SafetyPostconditionLink where
  prevHash   : String
  safeBounds : List IntInterval
  linkHash   : String
  deriving DecidableEq, Repr

structure CertChainV2 where
  version         : Nat
  contextId       : String
  issuedEpoch     : Nat
  validUntilEpoch : Nat
  link1           : ProgramLink
  link2           : TaylorEnclosureLink
  link3           : SafetyPostconditionLink
  signatureHmac   : String
  deriving DecidableEq, Repr

def intervalEnclosed (cert safe : IntInterval) : Bool :=
  decide (safe.lo ≤ cert.lo ∧ cert.lo ≤ cert.hi ∧ cert.hi ≤ safe.hi)

def allBoundsSafe : List IntInterval → List IntInterval → Bool
  | [], [] => true
  | c :: cs, s :: ss => intervalEnclosed c s && allBoundsSafe cs ss
  | _, _ => false

def verifyCertChainV2 (c : CertChainV2) (expectedCtx : String) (evalEpoch : Nat) : Bool :=
  (c.version == 2) &&
  (c.link2.prevHash == c.link1.linkHash) &&
  (c.link3.prevHash == c.link2.linkHash) &&
  (c.contextId == expectedCtx) &&
  decide (c.issuedEpoch ≤ evalEpoch ∧ evalEpoch ≤ c.validUntilEpoch) &&
  decide (c.link2.certifiedBounds.length > 0) &&
  allBoundsSafe c.link2.certifiedBounds c.link3.safeBounds

def expectedContext : String := "eshkol.ce9.world_model.safety.v2"

def evalEpoch : Nat := 1750000000

def exportedCertChain : CertChainV2 := {
  version := 2,
  contextId := "eshkol.ce9.world_model.safety.v2",
  issuedEpoch := 1700000000,
  validUntilEpoch := 1800000000,
  link1 := {
    programDigest := "fff964d6e5aa9a38aee234033513442926c0775ade3a6245877ba350ad04927e",
    linkHash := "8c6276c8bcf6ae1da0c06a4dbea240073332ec706eb0ddecbbfa6a18865ae601"
  },
  link2 := {
    prevHash := "8c6276c8bcf6ae1da0c06a4dbea240073332ec706eb0ddecbbfa6a18865ae601",
    certifiedBounds := [
      { lo := (-369362844 : Int), hi := 426469021 },
      { lo := (-340163183 : Int), hi := 383202879 }
    ],
    linkHash := "d4fa3111b476b6b1a3f4943a0766ff35547271aedc9ec19e8da1be0b77ec9d31"
  },
  link3 := {
    prevHash := "d4fa3111b476b6b1a3f4943a0766ff35547271aedc9ec19e8da1be0b77ec9d31",
    safeBounds := [
      { lo := (-750000000 : Int), hi := 750000000 },
      { lo := (-750000000 : Int), hi := 750000000 }
    ],
    linkHash := "a727987bf32f5f1b8677add17bc2ebccc54e68c0f836829bc5c6dd498fcae5a8"
  },
  signatureHmac := "1ed8b4997a5ff790b27ed9b4ce4cd846dd807473df06445fb40c2623e1d1bb79"
}

theorem cert_chain_v2_verified :
    verifyCertChainV2 exportedCertChain expectedContext evalEpoch = true := by
  decide

theorem cert_chain_hash_linkage :
    exportedCertChain.link2.prevHash = exportedCertChain.link1.linkHash ∧
    exportedCertChain.link3.prevHash = exportedCertChain.link2.linkHash := by
  decide

theorem cert_chain_safety_enclosure :
    allBoundsSafe exportedCertChain.link2.certifiedBounds exportedCertChain.link3.safeBounds = true := by
  decide

end EshkolCE9CertChain
