# Navier-Stokes construction: proof ledger

Machine-readable twin: `.icc/navier-stokes-proof-ledger.yaml`. Checked by
`scripts/check_ns_proof_ledger.py`. Every row below corresponds 1:1 to a row
of the pipeline table in
[docs/design/NAVIER_STOKES_BLOWUP_MECHANIZATION.md](NAVIER_STOKES_BLOWUP_MECHANIZATION.md)
Section 2, in the same numbering. Status vocabulary is exactly one of:

- **EXACT** -- checked with exact (rational/bignum) arithmetic, by a passing example program.
- **VALIDATED** -- checked numerically, to a stated tolerance, by a passing example program.
- **ANALYTIC-ONLY** -- no computable content exercising this row exists in v1.3.5; the missing
  capability is named.

**Counts:** EXACT 25, VALIDATED 0, ANALYTIC-ONLY 59, total 84.

| id | row | section | paper ref | operation | status | program(s) / missing capability |
|---|---|---|---|---|---|---|
| ns-proof-001 | 1 | 2.1 | Section 3.1; Section 3.6 | Fix tau=1-t, A=1/2+h, D=1/2-h and check 0<h<1/100, h<1/6 | EXACT | ns_similarity_exponents_solved |
| ns-proof-002 | 2 | 2.1 | (3.2); Lemma 4.1 | Eliminate eta from tau=q(1-eta^2) and verify q=q(z,tau) is single-valued | EXACT | ns_similarity_exponents_solved |
| ns-proof-003 | 3 | 2.1 | Section 3.1; (4.3) | Substitute the similarity ansatz into the residual and collect powers of q | EXACT | ns_leading_profile_balance |
| ns-proof-004 | 4 | 2.1 | (3.1); (4.12) | Evaluate the residual operator d_t u+(u.grad)u-nu Lap u+grad p at a point | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-005 | 5 | 2.1 | (4.7); (5.2) | Solve d_X V0 = -Z_-A U with V0(0,eta)=0 | EXACT | ns_leading_profile_balance |
| ns-proof-006 | 6 | 2.1 | Section 3.1; (4.31) | Impose the leading radial balance and normalize the pressure at infinity | EXACT | ns_leading_profile_balance |
| ns-proof-007 | 7 | 2.1 | Section 3.1 | Derive the core extents l_r, l_z and the growth of u_theta^(0) | EXACT | ns_similarity_exponents_solved |
| ns-proof-008 | 8 | 2.1 | Section 3.5 | Derive E_core, D_core and verify the finite-energy integral for h<1/6 | EXACT | ns_similarity_exponents_solved |
| ns-proof-009 | 9 | 2.1 | (10.22)-(10.23) | Verify the viscosity-scaling identity for u_nu, p_nu, f_nu | EXACT | ns_viscosity_scaling_exact |
| ns-proof-010 | 10 | 2.2 | Proposition 4.2; (4.12) | Show the tangential residuals are minus cylindrical divergences of a stress pair | ANALYTIC-ONLY | symbolic multivariate polynomial and series values / polynomial-valued duals (v1.4.0-connection build item) |
| ns-proof-011 | 11 | 2.2 | Section 3.2 | Integrate radially with the axis-regularity constant (T_r theta, T_r z) | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-012 | 12 | 2.2 | Section 3.2; Lemma A.8 | Impose the total-moment identities forcing the stress to vanish beyond the exterior radius | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-013 | 13 | 2.2 | (4.15) | Form the five cumulative radial integrals M, I, J, S, C_p | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-014 | 14 | 2.2 | Lemma 4.3; (4.16) | Reduce Q_s, N_s to closed expressions in (M,I,J,S,Pi) | ANALYTIC-ONLY | symbolic multivariate polynomial and series values / polynomial-valued duals (v1.4.0-connection build item) |
| ns-proof-015 | 15 | 2.2 | Lemma 4.4(i) | Certify the joining rule at the joining radius X_h | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-016 | 16 | 2.3 | (4.20) | Form t_s, v_s, P_c, J_c from the radial shear and integrated inviscid contribution | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-017 | 17 | 2.3 | (4.21) | Evaluate U(P_c,J_c) and test the relaxed cone condition | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-018 | 18 | 2.3 | Lemma 4.5; (4.22) | Certify the cone-condition equivalence via a quadratic root computation | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-019 | 19 | 2.3 | Lemma 4.5, second assertion | Produce the uniform threshold P_K on a compact parameter set | ANALYTIC-ONLY | rigorous (directed-rounding) enclosures and a proved Taylor-model remainder, in place of the current validated/sampled bound (v1.5.0-intelligence / v2.0-starlight build item) |
| ns-proof-020 | 20 | 2.3 | (4.23) | Rewrite the cone in stress coordinates, homogeneous in T0 | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-021 | 21 | 2.3 | Theorem 4.6 | Assemble the leading-profile theorem from rows 10-20 and 22-33 | ANALYTIC-ONLY | composite of multiple rows with no single Eshkol primitive or build item; assembled only once its constituent rows are mechanized |
| ns-proof-022 | 22 | 2.4 | Lemma A.1 | Certify the moment matrix B_ij is invertible for distinct exponents | ANALYTIC-ONLY | symbolic multivariate polynomial and series values / polynomial-valued duals (v1.4.0-connection build item) |
| ns-proof-023 | 23 | 2.4 | (A.1) | Quantify the loss when exponents collide (det B formula) | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-024 | 24 | 2.4 | Lemma A.2; (A.2)-(A.3) | Solve the quadratic moment system by contraction | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-025 | 25 | 2.4 | Corollary A.3; (A.4) | Normalize the five moments and split into U/E blocks with distinct powers | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-026 | 26 | 2.4 | Proposition A.4 | Construct the reference outer profile and read off Pi(0,eta) | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-027 | 27 | 2.4 | Lemma A.6; (A.33)-(A.38) | Build the heat exterior K(r,t) and verify its ODE and derivative values | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-028 | 28 | 2.4 | Lemma A.6, monotonicity | Certify K_r<0 and the ratio bound on ZH'/H | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-029 | 29 | 2.4 | Proposition A.7 | Restore the pressure integral after replacing the power tail by the heat flow | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-030 | 30 | 2.4 | Lemma A.8; Proposition A.10 | Under exact moment conditions conclude the leading tangential stresses vanish | ANALYTIC-ONLY | rigorous (directed-rounding) enclosures and a proved Taylor-model remainder, in place of the current validated/sampled bound (v1.5.0-intelligence / v2.0-starlight build item) |
| ns-proof-031 | 31 | 2.5 | Lemma B.1 | Establish boundedness of multiplication and the radial inverses on B_rho | ANALYTIC-ONLY | symbolic multivariate polynomial and series values / polynomial-valued duals (v1.4.0-connection build item) |
| ns-proof-032 | 32 | 2.5 | Proposition B.2; (B.12)-(B.15) | Solve the axis system by contraction about the comparison profile | ANALYTIC-ONLY | symbolic multivariate polynomial and series values / polynomial-valued duals (v1.4.0-connection build item) |
| ns-proof-033 | 33 | 2.5 | Proposition B.3; Lemma B.4; Proposition B.5 | Continue the analytic axis profile to X_i inside the relaxed cone | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-034 | 34 | 2.5 | Lemma B.7; Proposition B.8; Corollary B.10 | Match the five cumulative radial integrals at the joining radius | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-035 | 35 | 2.6 | Lemma C.1; (C.1)-(C.5) | Construct a period-one loop with prescribed mean and cone gaps | ANALYTIC-ONLY | torus (T1/T2) averaging, the Haar mean, phase-map evaluation with chain rule, and the inverse of a directional derivative on the zero-mean subspace (v1.5.0-intelligence build item) |
| ns-proof-036 | 36 | 2.6 | Lemma C.1, removable singularity | Certify the removable singularity at p_s2=0 | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-037 | 37 | 2.6 | Proposition C.2; (C.11)-(C.13) | Take zero-mean periodic antiderivatives and verify the exact shears | ANALYTIC-ONLY | torus (T1/T2) averaging, the Haar mean, phase-map evaluation with chain rule, and the inverse of a directional derivative on the zero-mean subspace (v1.5.0-intelligence build item) |
| ns-proof-038 | 38 | 2.6 | Proposition C.2; (C.14) | Bound the profile and moment perturbation by O(N^-1) | ANALYTIC-ONLY | rigorous (directed-rounding) enclosures and a proved Taylor-model remainder, in place of the current validated/sampled bound (v1.5.0-intelligence / v2.0-starlight build item) |
| ns-proof-039 | 39 | 2.6 | Proposition C.2, restoration; Proposition C.3 | Restore the five radial moments on the reserved correction patch | ANALYTIC-ONLY | torus (T1/T2) averaging, the Haar mean, phase-map evaluation with chain rule, and the inverse of a directional derivative on the zero-mean subspace (v1.5.0-intelligence build item) |
| ns-proof-040 | 40 | 2.7 | (5.1) | Set lambda_n=2nh and write the order-n physical coefficients | ANALYTIC-ONLY | symbolic multivariate polynomial and series values / polynomial-valued duals (v1.4.0-connection build item) |
| ns-proof-041 | 41 | 2.7 | (5.2)-(5.6) | Form the order-n coefficient system (azimuthal, axial, pressure equations) | ANALYTIC-ONLY | symbolic multivariate polynomial and series values / polynomial-valued duals (v1.4.0-connection build item) |
| ns-proof-042 | 42 | 2.7 | Section 5.1, splitting | Certify the order-n system is linear at each positive order | ANALYTIC-ONLY | symbolic multivariate polynomial and series values / polynomial-valued duals (v1.4.0-connection build item) |
| ns-proof-043 | 43 | 2.7 | Section 5.1, regularity | Verify Omega_k is divisible by X (axis regularity) | EXACT | ns_leading_profile_balance |
| ns-proof-044 | 44 | 2.7 | Lemma 5.1 | Solve, at every positive order, uniquely on 0<=xi<=a | ANALYTIC-ONLY | symbolic multivariate polynomial and series values / polynomial-valued duals (v1.4.0-connection build item) |
| ns-proof-045 | 45 | 2.7 | Lemma 5.2 | Extend each order radially and impose the five integral conditions | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-046 | 46 | 2.7 | Proposition 5.3 | Run the coefficient induction to a finite residual | ANALYTIC-ONLY | core.dbsp incremental knowledge-base evaluation over the closed world (GA at v1.5.0-intelligence) |
| ns-proof-047 | 47 | 2.7 | Lemma 5.4; Section 5.4 | Sum the coefficients with cutoffs, curls taken after multiplication | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-048 | 48 | 2.7 | Proposition 5.5; (5.41) | Conclude the smooth axisymmetric background (u_B,p_B) | ANALYTIC-ONLY | composite of multiple rows with no single Eshkol primitive or build item; assembled only once its constituent rows are mechanized |
| ns-proof-049 | 49 | 2.8 | (6.1) | Set the dyadic chart Q, epsilon, S_star, R, Z, T | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-050 | 50 | 2.8 | (6.3); (6.6) | Introduce the auxiliary torus variable Y and the phase map | ANALYTIC-ONLY | torus (T1/T2) averaging, the Haar mean, phase-map evaluation with chain rule, and the inverse of a directional derivative on the zero-mean subspace (v1.5.0-intelligence build item) |
| ns-proof-051 | 51 | 2.8 | Lemma 6.1; (6.13)-(6.15) | Choose color centers for disjoint auxiliary rectangles (graph coloring) | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-052 | 52 | 2.8 | Lemma 6.1, consequence | Certify F_gamma F_gamma'=0 for distinct labels | ANALYTIC-ONLY | torus (T1/T2) averaging, the Haar mean, phase-map evaluation with chain rule, and the inverse of a directional derivative on the zero-mean subspace (v1.5.0-intelligence build item) |
| ns-proof-053 | 53 | 2.8 | (6.14) | Bound the covering-index spread | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-054 | 54 | 2.8 | Definitions 6.4-6.5; Proposition 6.6 | Define the mean/moment coefficient classes closed under sums, products, derivatives | ANALYTIC-ONLY | symbolic multivariate polynomial and series values / polynomial-valued duals (v1.4.0-connection build item) |
| ns-proof-055 | 55 | 2.9 | Lemma 7.1; (7.2)-(7.3) | Construct the phase and an orthogonal frame B | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-056 | 56 | 2.9 | Proposition 7.2; Corollary 7.3; (7.5) | Solve the projected amplitude equation for each harmonic | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-057 | 57 | 2.9 | Lemma 7.4; (7.22) | Integrate the homogeneous pulse and certify the envelope | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-058 | 58 | 2.9 | Proposition 7.5; (7.27)-(7.28) | Average the quadratic product over the auxiliary torus and the angle | EXACT | ns_covariance_two_family_solve |
| ns-proof-059 | 59 | 2.9 | Proposition 7.5; (7.24),(7.29) | Solve the 2x2 system for the two pulse-family squared amplitudes | EXACT | ns_covariance_two_family_solve |
| ns-proof-060 | 60 | 2.9 | Proposition 7.5, Step 3 | Bound every fixed derivative of a_sigma via Faa di Bruno | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-061 | 61 | 2.9 | (7.30) | Sum slow boxes and bands to the leading physical stress | ANALYTIC-ONLY | composite of multiple rows with no single Eshkol primitive or build item; assembled only once its constituent rows are mechanized |
| ns-proof-062 | 62 | 2.9 | Proposition 7.6; (7.31) | Form the signed amplitude increments (first-order variation) | EXACT | ns_covariance_two_family_solve |
| ns-proof-063 | 63 | 2.9 | Lemma 7.7; Corollary 7.8; (7.35) | Take exact curls of the localized vector potentials and bound the covariance remainder | ANALYTIC-ONLY | torus (T1/T2) averaging, the Haar mean, phase-map evaluation with chain rule, and the inverse of a directional derivative on the zero-mean subspace (v1.5.0-intelligence build item) |
| ns-proof-064 | 64 | 2.10 | Proposition 8.1; (8.1)-(8.3) | Write the conservative momentum equations for the divergence-free decomposition | EXACT | ns_five_moment_vandermonde |
| ns-proof-065 | 65 | 2.10 | Lemma 8.2; Proposition 8.3; (8.12) | Build a compactly supported primitive and reconstruct pressure after each correction | EXACT | ns_five_moment_vandermonde |
| ns-proof-066 | 66 | 2.10 | Proposition 8.4; Corollary 8.5; (8.15) | Identify the three compatibility defects (P, J_theta, J_z) as covariance targets | EXACT | ns_five_moment_vandermonde |
| ns-proof-067 | 67 | 2.10 | Lemma 8.6; (8.20) | Invert the fast auxiliary-time derivative on zero Haar-mean functions on T^2 | ANALYTIC-ONLY | torus (T1/T2) averaging, the Haar mean, phase-map evaluation with chain rule, and the inverse of a directional derivative on the zero-mean subspace (v1.5.0-intelligence build item) |
| ns-proof-068 | 68 | 2.10 | Lemma 8.7; (8.24)-(8.25) | Solve the five radial moment equations with three azimuthal and two axial bumps | EXACT | ns_five_moment_vandermonde |
| ns-proof-069 | 69 | 2.10 | Lemma 8.7, Step 3 | Pass from q-normalized rows to a fixed Q chart | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-070 | 70 | 2.10 | Lemma 8.8; (8.27); Section 8.7 | Record the exact difference and certify the improved decay of the nonlinear remainder | ANALYTIC-ONLY | the underlying Eshkol primitive is shipped, but no dedicated Navier-Stokes example program exercises this row in v1.3.5 |
| ns-proof-071 | 71 | 2.11 | Section 3.3, exact increment identity | R(u_B+w,p_B+pi)=R(u_B,p_B)+L_uB(w,pi)+div(w o w) | EXACT | ns_viscosity_scaling_exact, ns_residual_exponent_ladder |
| ns-proof-072 | 72 | 2.11 | Proposition 9.1; Lemma 9.2 | Bound the residual of curl-constructed velocities and wave-mean interaction terms | ANALYTIC-ONLY | rigorous (directed-rounding) enclosures and a proved Taylor-model remainder, in place of the current validated/sampled bound (v1.5.0-intelligence / v2.0-starlight build item) |
| ns-proof-073 | 73 | 2.11 | Proposition 9.3; (9.3) | Recompute the full residual after every operation | EXACT | ns_residual_exponent_ladder |
| ns-proof-074 | 74 | 2.11 | Definition 9.4; Proposition 9.5; (9.7)-(9.10) | Initialize the temporal mean update and the five-equation correction | ANALYTIC-ONLY | torus (T1/T2) averaging, the Haar mean, phase-map evaluation with chain rule, and the inverse of a directional derivative on the zero-mean subspace (v1.5.0-intelligence build item) |
| ns-proof-075 | 75 | 2.11 | Proposition 9.6, steps (i)-(iv) | Run one full correction cycle | EXACT | ns_residual_exponent_ladder |
| ns-proof-076 | 76 | 2.11 | (9.8); (9.18) | Certify the residual-exponent ladder sigma_j=1/5+j/10 | EXACT | ns_residual_exponent_ladder |
| ns-proof-077 | 77 | 2.11 | Lemma 9.7; Lemma 9.8 | Certify a common domain and stage-independent derivative estimates | ANALYTIC-ONLY | rigorous (directed-rounding) enclosures and a proved Taylor-model remainder, in place of the current validated/sampled bound (v1.5.0-intelligence / v2.0-starlight build item) |
| ns-proof-078 | 78 | 2.11 | Proposition 9.9; (3.4); (9.20) | Sum corrections with shrinking cutoffs and certify flatness | ANALYTIC-ONLY | rigorous (directed-rounding) enclosures and a proved Taylor-model remainder, in place of the current validated/sampled bound (v1.5.0-intelligence / v2.0-starlight build item) |
| ns-proof-079 | 79 | 2.12 | Section 3.5; Proposition 10.1; (3.3),(10.4) | Localize with c=chi_x chi_t, extend by zero outside the cutoff support | EXACT | ns_localization_smooth_extension |
| ns-proof-080 | 80 | 2.12 | Lemma 10.2; Lemma 10.3; (10.5),(10.8)-(10.9) | Certify convergence as t->1 and realize the limits from t>1 in Cc^infty | EXACT | ns_localization_smooth_extension |
| ns-proof-081 | 81 | 2.12 | Lemma 10.4; (10.13) | Certify the global energy F(1)<infinity and the integrated dissipation bound | EXACT | ns_energy_dissipation_bounds |
| ns-proof-082 | 82 | 2.12 | Lemma 10.5; (10.20)-(10.21) | Verify the growth path and run the Gronwall comparison | ANALYTIC-ONLY | rigorous (directed-rounding) enclosures and a proved Taylor-model remainder, in place of the current validated/sampled bound (v1.5.0-intelligence / v2.0-starlight build item) |
| ns-proof-083 | 83 | 2.12 | (10.22)-(10.23) | Rescale to arbitrary nu>0 and verify the identity term by term | EXACT | ns_viscosity_scaling_exact |
| ns-proof-084 | 84 | 2.12 | Corollary 10.6 | Transport the construction to T^3=R^3/Z^3 | EXACT | ns_torus_corollary_scaling |

## Programs referenced

| CTest / criterion id | example program |
|---|---|
| `ns_viscosity_scaling_exact` | `examples/mathematics_navier_stokes_viscosity_scaling.esk` |
| `ns_similarity_exponents_solved` | `examples/mathematics_navier_stokes_similarity_scales.esk` |
| `ns_leading_profile_balance` | `examples/mathematics_navier_stokes_first_principles.esk` |
| `ns_covariance_two_family_solve` | `examples/mathematics_navier_stokes_pulse_stress.esk` |
| `ns_five_moment_vandermonde` | `examples/mathematics_navier_stokes_mean_corrections.esk` |
| `ns_residual_exponent_ladder` | `examples/mathematics_navier_stokes_residual_ladder.esk` |
| `ns_localization_smooth_extension` | `examples/mathematics_navier_stokes_localization.esk` |
| `ns_energy_dissipation_bounds` | `examples/mathematics_navier_stokes_localization.esk` |
| `ns_torus_corollary_scaling` | `examples/mathematics_navier_stokes_localization.esk` |
